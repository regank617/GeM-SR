
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.modeling.modules.wavelet import HaarWaveletTransform2D, InverseHaarWaveletTransform2D


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from src.modeling.modules.wavelet import HaarWaveletTransform2D, InverseHaarWaveletTransform2D
from src.modeling.generator.conditioner_layer_config import ConditionerLayerConfig


def psnr_from_mse(mse_tensor: torch.Tensor, data_range: float = 1.0) -> float:
    mse_val = mse_tensor.item()
    if mse_val == 0 or math.isnan(mse_val):
        return float('inf')
    return 20 * math.log10(data_range) - 10 * math.log10(mse_val)

class ZeroPreservingDenseBlock(nn.Module):
    """Zero-Preserving Dense Block (No biases, no BatchNorm)"""
    def __init__(self, in_channels: int, growth_rate: int = 32, conv_bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, 1, 1, bias=conv_bias)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, 1, 1, bias=conv_bias)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, 3, 1, 1, bias=conv_bias)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, 3, 1, 1, bias=conv_bias)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, 3, 1, 1, bias=conv_bias)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x  # Residual scaling factor (0.2) stabilizes training

class ZeroPreservingRRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, channels: int, growth_rate: int = 32, conv_bias: bool = False):
        super().__init__()
        self.rdb1 = ZeroPreservingDenseBlock(channels, growth_rate, conv_bias=conv_bias)
        self.rdb2 = ZeroPreservingDenseBlock(channels, growth_rate, conv_bias=conv_bias)
        self.rdb3 = ZeroPreservingDenseBlock(channels, growth_rate, conv_bias=conv_bias)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x  # Global residual scaling

class ZeroPreservingInversePhysicsStem(nn.Module):
    """Zero-Preserving mapping with Late Deep Feature Upsampling and Global Residual."""
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        hidden_dim: int = 64,
        num_rrdb: int = 4,
        layer_cfg: Optional[ConditionerLayerConfig] = None,
    ):
        super().__init__()
        _lc = layer_cfg if layer_cfg is not None else ConditionerLayerConfig()
        b = _lc.conv_bias

        # 1. Initial Low-Res Feature Extraction (32x16)
        self.conv_first = nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=b)

        # 2. Deep Feature Extraction (RRDBs operate in low-res space)
        self.rrdbs = nn.Sequential(
            *[ZeroPreservingRRDB(hidden_dim, conv_bias=b) for _ in range(num_rrdb)]
        )
        self.conv_trunk = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=b)

        # 3. Late Upsampling & High-Res Refinement
        self.upsample_refine = nn.Sequential(
            nn.Upsample(size=(48, 48), mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=b),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, padding=1, bias=b),
        )

        # 4. Global Residual Base Projection
        self.base_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=b)

    def forward(self, x):
        # Base Image Upsampling (Low-Frequency Structure) 
        # using bilinear to avoid bicubic ringing artifacts into empty space
        base_img = F.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
        base_img = self.base_proj(base_img)
        
        # Deep Feature Extraction
        feat = self.conv_first(x)
        trunk = self.conv_trunk(self.rrdbs(feat))
        feat = feat + trunk  # Local residual
        
        # Upsample rich features to get High-Frequency Residual
        residual = self.upsample_refine(feat)
        
        # Global Residual Connection
        out = base_img + residual
        return out

class SuperResWaveletConditionerV4(nn.Module):
    """SOTA Spatial Conditioner using RRDBs."""
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 512,
        hidden_dim: int = 64,
        layer_cfg: Optional[ConditionerLayerConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = ZeroPreservingInversePhysicsStem(
            in_channels, embed_dim, hidden_dim, num_rrdb=4, layer_cfg=layer_cfg
        )
        self.dwt = HaarWaveletTransform2D()

    def forward(self, x: torch.Tensor):
        curr = self.input_proj(x) 
        outputs = {}
        
        coeffs_1 = self.dwt(curr) 
        outputs['scale_2_packed'] = coeffs_1 
        curr = coeffs_1[:, :self.embed_dim]  
        
        coeffs_2 = self.dwt(curr) 
        outputs['scale_1_packed'] = coeffs_2 
        curr = coeffs_2[:, :self.embed_dim]  
        
        coeffs_3 = self.dwt(curr) 
        outputs['scale_0_packed'] = coeffs_3 
        outputs['root_ll'] = coeffs_3[:, :self.embed_dim] 
        outputs['global_style'] = coeffs_3.mean(dim=(2, 3)) 
       
        return outputs

class ConditionerWaveletPretrainer(nn.Module):
    """Pretrains the Conditioner by matching continuous outputs to Tokenizer Ground Truth."""

    def __init__(
        self,
        conditioner: nn.Module,
        embed_dim: int = 512,
        tokenizer_dim: int = 8,
        use_sparsity_weighted_l1: bool = True,
        edge_weight_multiplier: float = 10.0,
        active_threshold: float = 0.05,
        layer_cfg: Optional[ConditionerLayerConfig] = None,
    ):
        super().__init__()
        _lc = layer_cfg if layer_cfg is not None else ConditionerLayerConfig()
        b = _lc.conv_bias
        self.conditioner = conditioner
        self.use_sparsity_weighted_l1 = bool(use_sparsity_weighted_l1)
        self.edge_weight_multiplier = float(edge_weight_multiplier)
        self.active_threshold = float(active_threshold)
        self.proj_root = nn.Conv2d(embed_dim, tokenizer_dim, kernel_size=1, bias=b)
        self.proj_scales = nn.ModuleList(
            [nn.Conv2d(4 * embed_dim, 4 * tokenizer_dim, kernel_size=1, bias=b) for _ in range(3)]
        )

    def forward(self, lr_noisy_image: torch.Tensor, gt_pyramid: list, gt_root_ll: torch.Tensor):
        cond_outputs = self.conditioner(lr_noisy_image)
        metrics = {}
        total_loss = 0.0
        
        pred_root = self.proj_root(cond_outputs['root_ll'])
        loss_root = F.l1_loss(pred_root, gt_root_ll)
        mse_root = F.mse_loss(pred_root, gt_root_ll)
        total_loss += loss_root
        metrics['root_mse'] = mse_root.item()
        
        scale_weights = [1.0, 5.0, 20.0] 
        
        for i in range(3):
            gt_ll, gt_details = gt_pyramid[i]
            gt_packed = torch.cat([gt_ll, gt_details], dim=1) 
            pred_packed = self.proj_scales[i](cond_outputs[f'scale_{i}_packed'])

            if self.use_sparsity_weighted_l1:
                l1_unreduced = F.l1_loss(pred_packed, gt_packed, reduction="none")
                active_mask = (gt_packed.abs() > self.active_threshold).float()
                em = self.edge_weight_multiplier
                weight_matrix = 1.0 + ((em - 1.0) * active_mask)
                loss_scale = (l1_unreduced * weight_matrix).mean()
            else:
                loss_scale = F.l1_loss(pred_packed, gt_packed)
            
            total_loss += (loss_scale * scale_weights[i])
            mse_scale = F.mse_loss(pred_packed, gt_packed)
            metrics[f'scale_{i}_mse'] = mse_scale.item()
            
        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics

def extract_conditioner(pretrainer_module):
    if hasattr(pretrainer_module, "module"):
        return pretrainer_module.module.conditioner
    return pretrainer_module.conditioner