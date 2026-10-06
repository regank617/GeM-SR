import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional

from src.modeling.modules.wavelet import HaarWaveletTransform2D, InverseHaarWaveletTransform2D
from src.modeling.tokenizer.tokenizer_v5_layer_config import TokenizerV5LayerConfig
from src.modeling.modules.fsq import FSQ


class SpatialBandGate2D(nn.Module):
    def __init__(self, channels: int, conv_bias: bool = False):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=conv_bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=conv_bias),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.gate(x)

class ZeroPreservingRMSNorm2D(nn.Module):
    def __init__(self, num_channels, groups=1, eps=1e-5):
        super().__init__()
        self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
    def forward(self, x):
        B, C, H, W = x.shape
        G = self.groups
        x_g = x.view(B, G, C // G, H, W)
        variance = x_g.pow(2).mean(dim=2, keepdim=True)
        x_normed = x_g * torch.rsqrt(variance + self.eps)
        return (x_normed.view(B, C, H, W) * self.weight)

class ZeroPreservingResBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int = 4, conv_bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups, bias=conv_bias)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups, bias=conv_bias)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


def _groupnorm_num_groups(num_channels: int, preferred: int = 8) -> int:
    for g in (preferred, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


class StandardResBlock2D(nn.Module):
    """Conv → GroupNorm → act → Conv → GroupNorm + residual (baseline CNN path)."""

    def __init__(self, channels: int, conv_groups: int = 4, conv_bias: bool = True):
        super().__init__()
        ng = _groupnorm_num_groups(channels, 8)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=conv_groups, bias=conv_bias
        )
        self.norm1 = nn.GroupNorm(ng, channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=conv_groups, bias=conv_bias
        )
        self.norm2 = nn.GroupNorm(ng, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return x + h

class LidarTokenizerV5(nn.Module):
    """Haar + FSQ tokenizer. Set ``use_refiner=False`` to train/eval without the conv refiner (weights kept for ckpt compat).

    Ablations (paper / parity runs):

    - **Even detail FSQ:** use even integers in ``fsq_levels`` (e.g. ``[4, 4, 4, 4]``). ``FSQ`` then uses
      uniform binning (see ``fsq.FSQ``) instead of the odd-only half-integer grid.
    - **LiDAR sparsity on latents:** ``apply_sparsity_mask_to_latents`` (default ``True``) zeros quantized
      detail latents outside the physical canopy mask before the decoder.
    - **Null-token routing:** ``use_null_routing`` (paper name) or ``use_null_token_routing`` forces
      detail indices to ``null_idx_det`` off-canopy; set ``False`` for dense FSQ indices (ablation).
      If ``use_null_routing`` is set, it overrides ``use_null_token_routing``.
    """

    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 4, 
        hidden_dim: int = 64,         
        num_scales: int = 3,         
        fsq_levels: List[int] = [3, 3, 3, 3],
        # vbr_threshold: float = 0.02,
        # vbr_ratio: float = 0.1,
        inter_loss_weight: float = 2.0,
        use_refiner: bool = False,
        apply_sparsity_mask_to_latents: bool = True,
        use_null_token_routing: bool = True,
        use_null_routing: Optional[bool] = None,
        layer_cfg: Optional[TokenizerV5LayerConfig] = None,
    ):
        super().__init__()
        _lc = layer_cfg if layer_cfg is not None else TokenizerV5LayerConfig()
        b = _lc.conv_bias
        zp = _lc.use_zero_preserving_resnorm
        assert input_channels == 1, "Pure geometry DWT expects 1 channel input."
        assert latent_dim == 4, "Latent dim must be 4 to map 1:1 with 2D Haar sub-bands."
        self.input_channels = input_channels
        self.num_scales = num_scales
        #self.vbr_threshold = vbr_threshold
        #self.vbr_ratio = vbr_ratio
        self.inter_loss_weight = inter_loss_weight
        self.use_refiner = use_refiner
        self.apply_sparsity_mask_to_latents = bool(apply_sparsity_mask_to_latents)
        if use_null_routing is not None:
            self.use_null_token_routing = bool(use_null_routing)
        else:
            self.use_null_token_routing = bool(use_null_token_routing)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.fsq_levels = fsq_levels
        self.num_scales=num_scales
        self.dwt = HaarWaveletTransform2D()
        self.idwt = InverseHaarWaveletTransform2D()

        if zp:
            _root_norm: nn.Module = ZeroPreservingRMSNorm2D(3, groups=1)
        else:
            _root_norm = nn.GroupNorm(num_groups=1, num_channels=3)

        # Dense Root Encoder/Decoder
        self.proj_ll_in = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=b),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, 3, kernel_size=1, bias=b),
            _root_norm,
        )
        self.proj_ll_out = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=1, bias=b),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, bias=b),
        )
        self.fusion_attentions = nn.ModuleList([SpatialBandGate2D(4, conv_bias=b) for _ in range(num_scales)])

        def _res_pair():
            if zp:
                return [
                    ZeroPreservingResBlock2D(hidden_dim, groups=4, conv_bias=b),
                    ZeroPreservingResBlock2D(hidden_dim, groups=4, conv_bias=b),
                ]
            return [
                StandardResBlock2D(hidden_dim, conv_groups=4, conv_bias=b),
                StandardResBlock2D(hidden_dim, conv_groups=4, conv_bias=b),
            ]

        # Deep ResNet Encoders
        self.fusion_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1, groups=4, bias=b),
                    nn.LeakyReLU(0.2),
                    *_res_pair(),
                    nn.Conv2d(hidden_dim, latent_dim, kernel_size=1, groups=4, bias=b),
                )
                for _ in range(num_scales)
            ]
        )
        # Deep ResNet Decoders
        self.fusion_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(latent_dim, hidden_dim, kernel_size=1, groups=4, bias=b),
                    nn.LeakyReLU(0.2),
                    *_res_pair(),
                    nn.Conv2d(hidden_dim, 4, kernel_size=3, padding=1, groups=4, bias=b),
                )
                for _ in range(num_scales)
            ]
        )

        self.refiner = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            # Final output is 1 channel (the residual correction)
            nn.Conv2d(16, 1, kernel_size=3, padding=1) 
        )
        # ==========================================
        # NEW: Learnable Channel-Wise Quantization Scales (LSQ)
        # Initializes at 5.0, letting the network dynamically balance resolution vs. clipping
        # ==========================================
        self.quant_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, latent_dim, 1, 1) * 1.0) for _ in range(num_scales)
        ])
        # self.quant_scales = nn.Parameter(torch.ones(self.num_scales, 1, 4, 1, 1))
        self.fsq_root = FSQ(levels=[7, 5, 5], expected_dim=3, allow_even_levels=False)
        _detail_has_even_level = any(int(L) % 2 == 0 for L in fsq_levels)
        self.fsq_details = FSQ(
            levels=list(fsq_levels),
            expected_dim=latent_dim,
            allow_even_levels=_detail_has_even_level,
        )
        self.null_idx_root = self.fsq_root.zero_idx
        self.null_idx_det = self.fsq_details.zero_idx

    def _calc_entropy_and_vocab_used(self, indices, vocab_size):
        if indices.numel() == 0: return 0.0, 0
        counts = torch.histc(indices.float(), bins=vocab_size, min=0, max=vocab_size - 1)
        probs = counts / (counts.sum() + 1e-6)
        probs_pos = probs[probs > 0]
        entropy = -torch.sum(probs_pos * torch.log2(probs_pos)).item()
        vocab_used = (counts > 0).sum().item()
        return entropy, vocab_used

    def forward(self, x, test=False, diagnostics=None):
        B, C, H, W = x.shape
        
        with torch.no_grad():
            input_sparsity = (x.abs() < 0.01).float().mean().item()
        if diagnostics is not None:
            diagnostics["0_input"] = self._tensor_stats(x)
            
        # Keep base_mask for sparsity/active_mask logic in the wavelet scales
        base_mask = (x > 0.0).float()
        if base_mask.shape[1] > 1:
            base_mask = base_mask.max(dim=1, keepdim=True)[0]
            
        test_dict = {}
        gt_pyramid = []
        mask_pyramid = []
        curr = x
        curr_mask = base_mask
        
        for _ in range(self.num_scales):
            combined = self.dwt(curr)
            ll_gt = combined[:, 0:1]
            details_gt = combined[:, 1:4] 
            gt_pyramid.append((ll_gt, details_gt))
            curr = ll_gt
            curr_mask = F.max_pool2d(curr_mask, kernel_size=2, stride=2)
            mask_pyramid.append(curr_mask)
            
        gt_pyramid = list(reversed(gt_pyramid))
        mask_pyramid = list(reversed(mask_pyramid))
        
        if diagnostics is not None:
            diagnostics["2_pyramid_root_ll"] = self._tensor_stats(gt_pyramid[0][0])
            diagnostics["2_pyramid_root_details"] = self._tensor_stats(gt_pyramid[0][1])
            
        total_inter_loss = torch.tensor(0.0, device=x.device)
        all_indices = []
        debug_stats = {"diag/input_sparsity": input_sparsity}
        
        # ==========================================
        # Scale 0: Root (Dense)
        # ==========================================
        residual_root = gt_pyramid[0][0]
        z_root_enc = self.proj_ll_in(residual_root)
        
        if diagnostics is not None:
            diagnostics["3_root_z_enc"] = self._tensor_stats(z_root_enc)
            
        z_q, idx, _ = self.fsq_root(z_root_enc)
        curr_ll_recon = self.proj_ll_out(z_q)
        
        if diagnostics: test_dict['curr_ll_recon_root'] = curr_ll_recon.clone().detach().cpu()
        
        with torch.no_grad():
            ent, vocab_used = self._calc_entropy_and_vocab_used(idx, self.fsq_root.vocab_size)
            debug_stats["s0_d0/ent"] = ent
            debug_stats["s0_d0/vocab_used"] = vocab_used
            
        all_indices.append(torch.stack([idx], dim=-1))
        
        # ==========================================
        # Scales 1..N: Details (Sparse)
        # ==========================================
        final_active_mask = None 
        for s_i, (gt_ll_next, gt_details) in enumerate(gt_pyramid):
            inter_loss_step = F.mse_loss(curr_ll_recon, gt_ll_next.detach())
            total_inter_loss = total_inter_loss + inter_loss_step
            debug_stats[f"loss/inter_scale_{s_i+1}"] = inter_loss_step.item()
            
            ll_error = gt_ll_next - curr_ll_recon
            packed_target = torch.cat([ll_error, gt_details], dim=1) 
            
            # Active mask purely relies on physical LiDAR sparsity
            active_mask = (mask_pyramid[s_i] > 0)
            
            if s_i == len(gt_pyramid) - 1:
                final_active_mask = active_mask 
                
            z_enc = self.fusion_encoders[s_i](packed_target)
            scale = F.softplus(self.quant_scales[s_i]) + 1e-3 # helps fsq from clipping or zero binning features
            z_enc_scaled = z_enc / scale
            
            # Latent Sparsity Penalty for LL_err (Dimension 0)
            # Evicts edge-data squatters from the macro-residual channel
            ll_latent_l1 = torch.abs(z_enc_scaled[:, 0]).mean()
            # Note: 0.05 is a standard starting weight for latent regularizers. 
            # If the histogram doesn't spike at 0, increase this to 0.1 or 0.2.
            total_inter_loss = total_inter_loss + (ll_latent_l1 * 0.001) 
            debug_stats[f"loss/s{s_i+1}_ll_latent_l1"] = ll_latent_l1.item()
            
            # Quantize safely
            z_q, active_idx, _ = self.fsq_details(z_enc_scaled)
            
            # LSQ: Multiply to restore physical geometric amplitude
            z_q_restored = z_q * scale
            
            # Optional LiDAR sparsity: mask quantized latents / route to null token (ablations toggle these).
            if self.apply_sparsity_mask_to_latents:
                z_q_masked = z_q_restored * active_mask.float()
            else:
                z_q_masked = z_q_restored
            if self.use_null_token_routing:
                active_idx = torch.where(active_mask.squeeze(1), active_idx, self.null_idx_det)
            all_indices.append(torch.stack([active_idx], dim=-1))
            
            with torch.no_grad():
                valid_idx = active_idx[active_mask.squeeze(1)]
                ent, vocab_used = self._calc_entropy_and_vocab_used(valid_idx, self.fsq_details.vocab_size)
                debug_stats[f"s{s_i+1}/ent"] = ent
                debug_stats[f"s{s_i+1}/vocab_used"] = vocab_used
                debug_stats[f"s{s_i+1}/scale_mean"] = scale.mean().item()
                
            z_out = self.fusion_decoders[s_i](z_q_masked)
            curr_ll_corrected = curr_ll_recon + z_out[:, 0:1]
            full_recon = torch.cat([curr_ll_corrected, z_out[:, 1:4]], dim=1)
            curr_ll_recon = self.idwt(full_recon)
            
            if diagnostics: test_dict[f'curr_ll_recon_{s_i}'] = curr_ll_recon.clone().detach().cpu()

        # ==========================================
        # Refiner & Dual-Loss Strategy (optional)
        # ==========================================
        x_recon_raw = curr_ll_recon
        if self.use_refiner:
            refiner_residual = self.refiner(x_recon_raw)
            x_refined_raw = x_recon_raw + refiner_residual
        else:
            x_refined_raw = x_recon_raw

        if diagnostics is not None:
            diagnostics["5_x_recon_raw_base"] = self._tensor_stats(x_recon_raw)
            diagnostics["5_x_recon_raw_refined"] = self._tensor_stats(x_refined_raw)

        mse_base = F.mse_loss(x_recon_raw, x)
        if self.use_refiner:
            mse_refined = F.mse_loss(x_refined_raw, x)
            loss = mse_base + mse_refined + (total_inter_loss * self.inter_loss_weight)
        else:
            mse_refined = mse_base
            loss = mse_base + (total_inter_loss * self.inter_loss_weight)

        with torch.no_grad():
            debug_stats["loss/mse_base"] = mse_base.item()
            debug_stats["loss/mse_refined"] = mse_refined.item()
            if self.use_refiner:
                debug_stats["loss/refiner_improvement_percent"] = (
                    (mse_base - mse_refined) / (mse_base + 1e-6)
                ).item() * 100
            else:
                debug_stats["loss/refiner_improvement_percent"] = 0.0

        x_final = torch.clamp(x_refined_raw, 0, 1)
        clean_recon_mse = F.mse_loss(x_final, x)
        
        info_dict = {
            "recon_mse": clean_recon_mse,
            "quantizer_loss": loss,
            "indices": all_indices,
            "debug_stats": debug_stats,
            "mse_base": mse_base,      
            "mse_refined": mse_refined,
            "inter_loss_total": total_inter_loss,
            "active_mask": final_active_mask,
            "diagnostics": diagnostics,
            'test_dict': test_dict,
        }
        
        if test:
            return x_final, (x_recon_raw, x_final), info_dict
        return x_final, loss, info_dict

    # ==========================================
    # HELPERS
    # ==========================================
    def _tensor_stats(self, t: torch.Tensor) -> Dict[str, float]:
        if t.numel() == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        flat = t.detach().reshape(-1) 
        return {
            "min": flat.min().item(),
            "max": flat.max().item(),
            "mean": flat.mean().item(),
            "std": flat.std().item()
        }

    def _index_stats(self, idx: torch.Tensor, null_idx: int, vocab_size: int) -> Dict[str, float]:
        if idx.numel() == 0:
            return {"valid_frac": 0.0, "unique_used": 0.0, "vocab_utilization": 0.0}
        flat_idx = idx.detach().reshape(-1)
        valid_mask = flat_idx != null_idx
        valid_indices = flat_idx[valid_mask]
        valid_frac = valid_mask.float().mean().item()
        if valid_indices.numel() > 0:
            unique_used = len(torch.unique(valid_indices))
            vocab_utilization = unique_used / vocab_size
        else:
            unique_used = 0
            vocab_utilization = 0.0
        return {
            "valid_frac": valid_frac,
            "unique_used": float(unique_used),
            "vocab_utilization": vocab_utilization
        }

# ------------ Generative State-Space Super-Resolution helper functions ------------ #
    @torch.no_grad()
    def get_hierarchical_codes(self, high_res_img, vbr_threshold=None, vbr_ratio=None):
        """
        Extracts hierarchical discrete tokens (codes) from an input image.
        FSQ Edition: A single flat token per spatial location [B, 1, H, W].
        """
        # Run forward pass in test mode to extract indices
        #_, _, info_dict = self.forward(high_res_img, vbr_threshold, vbr_ratio, test=True)
        _, _, info_dict = self.forward(high_res_img, test=True)
        codes = {}
        # info_dict['indices'] contains lists of [B, H, W, 1] tensors
        # Permute to standard generative sequence format: [B, 1, H, W]
        indices = info_dict['indices']
        codes['root'] = indices[0].permute(0, 3, 1, 2).contiguous()
        # Detail scales
        for i in range(1, self.num_scales + 1):
            codes[f'scale_{i-1}'] = indices[i].permute(0, 3, 1, 2).contiguous()
        return codes
    def decode_root(self, root_indices):
        """
        Decodes the dense base (root) scale indices into continuous feature maps.
        Args:
            root_indices: [B, 1, H, W]
        """
        # 1. Extract the single depth index
        idx = root_indices[:, 0, :, :]
        # 2. Map indices to continuous FSQ codes: [B, H, W] -> [B, H, W, dim]
        z_q = self.fsq_root.indices_to_codes(idx)
        # 3. Permute to channel-first: [B, dim, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        # 4. Decode directly to LL features
        curr_ll_recon = self.proj_ll_out(z_q)
        return curr_ll_recon

    def inverse_step(self, current_ll_features, scale_indices, scale_idx):
        """
        Performs one step of recursive wavelet reconstruction for a packed scale.
        Applies LSQ scaling and null-token masking.
        Args:
            current_ll_features: [B, 1, H_prev, W_prev]
            scale_indices: [B, 1, H, W]
            scale_idx: Current scale (0-indexed)
        """
        # 1. Extract the single depth index
        idx = scale_indices[:, 0, :, :]
        # 2. Map indices to continuous FSQ codes: [B, dim, H, W]
        z_q = self.fsq_details.indices_to_codes(idx).permute(0, 3, 1, 2).contiguous()
        # 3. Apply LSQ Scale to restore physical amplitude!
        # This matches z_q_restored = z_q * scale in the forward pass
        scale = F.softplus(self.quant_scales[scale_idx]) + 1e-3
        z_q_restored = z_q * scale.view(1, -1, 1, 1) # Broadcast channel scale
        # 4. Optional null-token masking on decode (match forward ablation flags)
        if self.use_null_token_routing:
            is_active = (idx != self.null_idx_det).unsqueeze(1).float()
        else:
            is_active = torch.ones_like(idx, dtype=z_q_restored.dtype, device=z_q_restored.device).unsqueeze(1)
        z_q_masked = z_q_restored * is_active
        # 5. Decode the fused subbands
        z_out = self.fusion_decoders[scale_idx](z_q_masked)
        # 6. Unpack the feature map (LL_error + Details)
        curr_ll_corrected = current_ll_features + z_out[:, 0:1]
        full_recon = torch.cat([curr_ll_corrected, z_out[:, 1:4]], dim=1)
        # 7. Apply Inverse Wavelet Transform
        return self.idwt(full_recon)

    def features_to_pixels(self, final_features, refiner_strength=1.0, ret_prerefiner=False):
        """
        Takes the final IDWT raw output and applies the Convolutional Refiner.
        Replaces the old out_conv logic to match the new dual-loss forward pass.
        Args:
            final_features: The raw output from the last `inverse_step` (x_recon_raw).
        """
        x_recon_raw = final_features
        if self.use_refiner:
            refiner_residual = self.refiner(x_recon_raw)
            x_refined = x_recon_raw + (refiner_residual * refiner_strength)
        else:
            x_refined = x_recon_raw
        # Final physical clamp
        final_recon = torch.clamp(x_refined, min=0.0, max=1.0)
        if ret_prerefiner: 
            return final_recon, torch.clamp(x_recon_raw, 0.0, 1.0)
        return final_recon


"""self.fusion_encoders = nn.ModuleList([
    nn.Sequential(
        # Changed groups=4 to groups=1 for dense cross-band correlation
        nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1, bias=False), 
        nn.LeakyReLU(0.2),
        ZeroPreservingResBlock2D(hidden_dim, groups=1),
        ZeroPreservingResBlock2D(hidden_dim, groups=1),
        nn.Conv2d(hidden_dim, latent_dim, kernel_size=1, bias=False),
    )
    for _ in range(num_scales)
])
self.fusion_decoders = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(latent_dim, hidden_dim, kernel_size=1, bias=False),
        nn.LeakyReLU(0.2),
        ZeroPreservingResBlock2D(hidden_dim, groups=1),
        ZeroPreservingResBlock2D(hidden_dim, groups=1),
        nn.Conv2d(hidden_dim, 4, kernel_size=3, padding=1, bias=False),
    )
    for _ in range(num_scales)
])"""