import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.modeling.generator.conditioner_layer_config import PaperConvNeXtLayerConfig

"""
generator_merged contains:
    1. ConvNeXt and Transformer backbones, which are used in the paper.
    2. Classifier-Free Guidance (CFG) capabilities.

It is used for ablations in the paper to compare ConvNeXt and Transformer backbones, as well as
CFG vs noC CFG.

generator_main is the main ConvNeXt backbone, with the transformer and CFG code stripped for readability.
"""



# ==============================================================================
# Shared Utilities & Norms
# ==============================================================================

class ChannelRMSNorm2d(nn.Module):
    """Zero-preserving RMSNorm for 2D spatial features [B, C, H, W]"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class RMSNorm1d(nn.Module):
    """Zero-preserving RMSNorm for sequence features [B, L, D]"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class StandardHead(nn.Module):
    """Linear head to vocabulary."""
    def __init__(self, dim: int, vocab_size: int, zero_idx: Optional[int] = None):
        super().__init__()
        self.vocab = vocab_size
        self.code_proj = nn.Linear(dim, vocab_size, bias=True)
        nn.init.zeros_(self.code_proj.bias)
        nn.init.normal_(self.code_proj.weight, mean=0.0, std=0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.code_proj(x)

# ==============================================================================
# Transformer Architecture Blocks
# ==============================================================================

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, h: int, w: int):
        super().__init__()
        self.num_heads = num_heads
        self.h, self.w = h, w
        self.rel_height = nn.Parameter(torch.zeros(2 * h - 1, num_heads))
        self.rel_width = nn.Parameter(torch.zeros(2 * w - 1, num_heads))
        nn.init.trunc_normal_(self.rel_height, std=0.02)
        nn.init.trunc_normal_(self.rel_width, std=0.02)

    def forward(self, H: int, W: int) -> torch.Tensor:
        device = self.rel_height.device
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij")).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel[0] += self.h - 1
        rel[1] += self.w - 1
        bh = self.rel_height[rel[0].clamp(0, 2 * self.h - 2)].permute(2, 0, 1)
        bw = self.rel_width[rel[1].clamp(0, 2 * self.w - 2)].permute(2, 0, 1)
        return bh + bw

class AdaLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:dim].fill_(1.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * gamma + beta

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = AdaLN(dim), AdaLN(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, global_cond, spatial_cond, attn_bias=None):
        B, L, D = x.shape
        x = x + spatial_cond
        h = self.ln1(x, global_cond)
        qkv = self.qkv(h).view(B, L, 3, self.h, D // self.h)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        x = x + self.proj(out.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln2(x, global_cond))

class TransformerScaleGenerator(nn.Module):
    def __init__(
        self,
        tokenizer_dim: int,
        embed_dim: int,
        vocab_size: int,
        rq_depths: int,
        cond_in_dim: int,
        spatial_shape: Tuple[int, int],
        zero_idx: Optional[int],
        depth: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.rq, self.vocab, self.zero_idx = rq_depths, vocab_size, zero_idx
        self.spatial_shape = spatial_shape
        self.in_proj = nn.Linear(tokenizer_dim, embed_dim)
        self.abs_pos_emb = nn.Parameter(torch.zeros(1, spatial_shape[0] * spatial_shape[1], embed_dim))
        self.cond_proj = nn.Sequential(nn.Linear(cond_in_dim, embed_dim), nn.LayerNorm(embed_dim))
        
        self.rel_pos_bias = RelativePositionBias(heads, spatial_shape[0], spatial_shape[1])
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
        self.norm_f = AdaLN(embed_dim)
        self.head = StandardHead(embed_dim, vocab_size, zero_idx)

    def forward(self, hidden, cond, global_cond):
        B, L, _ = hidden.shape
        H, W = self.spatial_shape
        x = self.in_proj(hidden) + self.abs_pos_emb
        s_cond = self.cond_proj(cond)
        bias = self.rel_pos_bias(H, W)
        for block in self.blocks:
            x = block(x, global_cond, s_cond, bias)
        x = self.norm_f(x, global_cond)
        
        logits = self.head(x)
        # CE loss expects [B, rq, H, W, V] 
        return logits.view(B, H, W, self.rq, self.vocab).permute(0, 3, 1, 2, 4)


# ==============================================================================
# Fast ConvNeXt Architecture Blocks
# ==============================================================================

class SubBandFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_cfg: PaperConvNeXtLayerConfig,
    ):
        super().__init__()
        b = layer_cfg.conv_bias
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=b),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=b),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=4, bias=b)

    def forward(self, x):
        return self.proj(x * self.gate(x))

class SpatialModulation(nn.Module):
    def __init__(self, x_dim, cond_dim, layer_cfg: PaperConvNeXtLayerConfig):
        super().__init__()
        self.proj = nn.Conv2d(cond_dim, x_dim * 2, kernel_size=1, bias=layer_cfg.conv_bias)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return x * (1 + gamma) + beta

class FastConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, mlp_ratio: float, layer_cfg: PaperConvNeXtLayerConfig):
        super().__init__()
        b = layer_cfg.conv_bias
        self.spatial_mod = SpatialModulation(dim, cond_dim, layer_cfg)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=b)
        self.norm = ChannelRMSNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=b)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=b)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.spatial_mod(x, cond)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x

class FastScaleGenerator(nn.Module):
    def __init__(
        self,
        tokenizer_dim: int,
        embed_dim: int,
        rq_depths: int,
        cond_in_dim: int,
        spatial_shape: Tuple[int, int],
        depth: int = 4,
        mlp_ratio: float = 2.0,
        layer_cfg: Optional[PaperConvNeXtLayerConfig] = None,
    ):
        super().__init__()
        _lc = layer_cfg if layer_cfg is not None else PaperConvNeXtLayerConfig()
        self.rq = rq_depths
        self.H, self.W = spatial_shape
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(tokenizer_dim, embed_dim, bias=_lc.linear_bias)
        
        self.use_fusion = cond_in_dim > embed_dim
        if self.use_fusion:
            self.cond_fusion = SubBandFusion(cond_in_dim, embed_dim, _lc)
            processed_cond_dim = embed_dim
        else:
            processed_cond_dim = cond_in_dim
            
        self.blocks = nn.ModuleList(
            [
                FastConvNeXtBlock(
                    dim=embed_dim, cond_dim=processed_cond_dim, mlp_ratio=mlp_ratio, layer_cfg=_lc
                )
                for _ in range(depth)
            ]
        )
        self.norm_f = RMSNorm1d(embed_dim)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor):
        B, L, _ = hidden.shape
        x_proj = self.in_proj(hidden)
        x = x_proj
        
        x_2d = x.transpose(1, 2).view(B, self.embed_dim, self.H, self.W).contiguous()
        s_cond_2d = cond.transpose(1, 2).view(B, -1, self.H, self.W).contiguous()
        
        if self.use_fusion:
            s_cond_2d = self.cond_fusion(s_cond_2d)
            
        for block in self.blocks:
            x_2d = block(x_2d, s_cond_2d)
            
        x_out = x_2d.view(B, -1, L).transpose(1, 2).contiguous()
        x_out = x_out + x_proj
        x_out = self.norm_f(x_out)
        return x_out.view(B, self.H, self.W, self.rq, self.embed_dim).permute(0, 3, 1, 2, 4)


# ==============================================================================
# Unified Master Generator
# ==============================================================================

def _resolve_vocab_sizes(
    vocab_sizes: Optional[Dict[str, int]], vocab_size_scalar: int
) -> Dict[str, int]:
    # We enforce joint tokenizer layout implicitly now
    keys = ("root", "scale_0", "scale_1", "scale_2")
    if vocab_sizes:
        out = {}
        for k in keys:
            if k not in vocab_sizes:
                raise ValueError(f"vocab_sizes must include {k} for joint layout")
            out[k] = int(vocab_sizes[k])
        return out
    raise ValueError("vocab_sizes must be provided explicitly")


class HierarchicalGenerator(nn.Module):
    def __init__(
        self,
        tokenizer_dim: int = 1,
        embed_dim: int = 128,
        vocab_size: int = 2401, # Keep for backward compat but we use vocab_sizes
        rq_depths: int = 1,
        zero_idx: Optional[int] = None,
        root_depth: int = 4,
        s0_depth: int = 4,
        s1_depth: int = 4,
        s2_depth: int = 4,
        mlp_ratio: int = 2,
        heads: int = 8,
        backbone: str = "convnext", # 'convnext' or 'transformer'
        cfg_training: bool = False, # Toggle CFG capability (adds null tokens and dropout)
        cond_dropout_prob: float = 0.1,
        scheduled_sampling_prob: float = 0.1,
        vocab_sizes: Optional[Dict[str, int]] = None,
        paper_layer_cfg: Optional[PaperConvNeXtLayerConfig] = None,
        **kwargs # Accept legacy args like tokenizer_layout silently
    ):
        super().__init__()
        self.paper_layer_cfg = paper_layer_cfg if paper_layer_cfg is not None else PaperConvNeXtLayerConfig()
        self.backbone = backbone.lower()
        self.cfg_training = cfg_training
        self.rq = rq_depths
        self.scheduled_sampling_prob = float(scheduled_sampling_prob)
        self.cond_dropout_prob = float(cond_dropout_prob)
        
        c_dim = 4 * embed_dim
        self.embed_dim = embed_dim
        self.c_dim = c_dim

        self.zero_idx = zero_idx
        vs = _resolve_vocab_sizes(vocab_sizes, vocab_size)

        # CFG & Transformer requirements
        if self.cfg_training or self.backbone == "transformer":
            self.global_proj = nn.Sequential(
                nn.Linear(c_dim, embed_dim, bias=self.paper_layer_cfg.linear_bias),
                nn.GELU(),
                RMSNorm1d(embed_dim),
            )
            
        if self.cfg_training:
            self.null_global = nn.Parameter(torch.randn(1, c_dim))
            self.null_root_ll = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.null_s0_packed = nn.Parameter(torch.randn(1, 1, c_dim))
            self.null_s1_packed = nn.Parameter(torch.randn(1, 1, c_dim))
            self.null_s2_packed = nn.Parameter(torch.randn(1, 1, c_dim))

        if self.backbone == "transformer":
            self.root_learned_state = nn.Parameter(torch.zeros(1, 36, tokenizer_dim))
            base_kw = dict(tokenizer_dim=tokenizer_dim, embed_dim=embed_dim, rq_depths=rq_depths, zero_idx=zero_idx, heads=heads)
            self.pred_root = TransformerScaleGenerator(spatial_shape=(6, 6), depth=root_depth, cond_in_dim=embed_dim, vocab_size=vs["root"], **base_kw)
            self.pred_s0 = TransformerScaleGenerator(spatial_shape=(6, 6), depth=s0_depth, cond_in_dim=c_dim, vocab_size=vs["scale_0"], **base_kw)
            self.pred_s1 = TransformerScaleGenerator(spatial_shape=(12, 12), depth=s1_depth, cond_in_dim=c_dim, vocab_size=vs["scale_1"], **base_kw)
            self.pred_s2 = TransformerScaleGenerator(spatial_shape=(24, 24), depth=s2_depth, cond_in_dim=c_dim, vocab_size=vs["scale_2"], **base_kw)
            
            self.head_root = self.head_s0 = self.head_s1 = self.head_s2 = None # Handled inside TransformerScaleGenerator

        elif self.backbone == "convnext":
            fast_kw = dict(
                tokenizer_dim=tokenizer_dim,
                embed_dim=embed_dim,
                rq_depths=rq_depths,
                layer_cfg=self.paper_layer_cfg,
            )
            root_kw = dict(
                tokenizer_dim=embed_dim,
                embed_dim=embed_dim,
                rq_depths=rq_depths,
                layer_cfg=self.paper_layer_cfg,
            )
            
            self.pred_root = FastScaleGenerator(spatial_shape=(6, 6), depth=root_depth, cond_in_dim=embed_dim, mlp_ratio=mlp_ratio, **root_kw)
            self.pred_s0 = FastScaleGenerator(spatial_shape=(6, 6), depth=s0_depth, cond_in_dim=c_dim, mlp_ratio=mlp_ratio, **fast_kw)
            self.pred_s1 = FastScaleGenerator(spatial_shape=(12, 12), depth=s1_depth, cond_in_dim=c_dim, mlp_ratio=mlp_ratio, **fast_kw)
            self.pred_s2 = FastScaleGenerator(spatial_shape=(24, 24), depth=s2_depth, cond_in_dim=c_dim, mlp_ratio=mlp_ratio, **fast_kw)
            
            self.head_root = StandardHead(embed_dim, vs["root"], zero_idx)
            self.head_s0 = StandardHead(embed_dim, vs["scale_0"], zero_idx)
            self.head_s1 = StandardHead(embed_dim, vs["scale_1"], zero_idx)
            self.head_s2 = StandardHead(embed_dim, vs["scale_2"], zero_idx)
            
            if not self.cfg_training:
                self.global_proj = nn.Sequential(
                    nn.Linear(c_dim, embed_dim, bias=self.paper_layer_cfg.linear_bias),
                    nn.GELU(),
                    RMSNorm1d(embed_dim),
                )

        else:
            raise ValueError("Backbone must be 'convnext' or 'transformer'")

    def _apply_cond_dropout(self, cond_outputs: Dict[str, torch.Tensor], B: int, device: torch.device):
        if not self.cfg_training:
             return cond_outputs["global_style"], cond_outputs["root_ll"], cond_outputs["scale_0_packed"], cond_outputs["scale_1_packed"], cond_outputs["scale_2_packed"]

        if self.training and self.cond_dropout_prob > 0:
            drop_mask = torch.rand(B, 1, 1, device=device) < self.cond_dropout_prob
        else:
            drop_mask = torch.zeros(B, 1, 1, dtype=torch.bool, device=device)
            
        drop_mask_global = drop_mask.squeeze(2)
        
        g_style_cond = torch.where(drop_mask_global, self.null_global.expand(B, -1), cond_outputs["global_style"])
        
        cond_root = torch.where(drop_mask, self.null_root_ll.expand(B, 36, -1), cond_outputs["root_ll"].flatten(2).transpose(1, 2))
        cond_s0 = torch.where(drop_mask, self.null_s0_packed.expand(B, 36, -1), cond_outputs["scale_0_packed"].flatten(2).transpose(1, 2))
        cond_s1 = torch.where(drop_mask, self.null_s1_packed.expand(B, 144, -1), cond_outputs["scale_1_packed"].flatten(2).transpose(1, 2))
        cond_s2 = torch.where(drop_mask, self.null_s2_packed.expand(B, 576, -1), cond_outputs["scale_2_packed"].flatten(2).transpose(1, 2))
        
        return g_style_cond, cond_root, cond_s0, cond_s1, cond_s2

    def forward(self, cond_outputs: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.Tensor], tokenizer):
        B = cond_outputs["global_style"].shape[0]
        device = cond_outputs["global_style"].device
        
        raw_g, cond_root, cond_s0, cond_s1, cond_s2 = self._apply_cond_dropout(cond_outputs, B, device)
        
        g_style = self.global_proj(raw_g) if hasattr(self, 'global_proj') else None
        
        if not self.cfg_training:
            cond_root = cond_root.flatten(2).transpose(1, 2)
            cond_s0 = cond_s0.flatten(2).transpose(1, 2)
            cond_s1 = cond_s1.flatten(2).transpose(1, 2)
            cond_s2 = cond_s2.flatten(2).transpose(1, 2)

        use_pred_prob = self.scheduled_sampling_prob if self.training else 0.0

        if self.backbone == "transformer":
            root_logits = self.pred_root(self.root_learned_state.expand(B, -1, -1), cond_root, g_style)
        else:
            root_hidden = g_style.unsqueeze(1).expand(B, 36, -1)
            root_h = self.pred_root(root_hidden, cond_root)
            root_logits = self.head_root(root_h)

        if self.training and use_pred_prob > 0.0:
            use_pred_mask = torch.rand(B, 1, 1, device=device) < use_pred_prob
            pr = root_logits.argmax(dim=-1)
            chosen_root = torch.where(use_pred_mask.view(B, 1, 1, 1), pr, target_tokens["root"])
        else:
            chosen_root = target_tokens["root"]
            use_pred_mask = None 
        
        curr_ll = tokenizer.decode_root(chosen_root)

        if self.backbone == "transformer":
            s0_logits = self.pred_s0(curr_ll.flatten(2).transpose(1, 2), cond_s0, g_style)
        else:
            s0_h = self.pred_s0(curr_ll.flatten(2).transpose(1, 2), cond_s0)
            s0_logits = self.head_s0(s0_h)
            
        if self.training and use_pred_prob > 0.0:
            p0 = s0_logits.argmax(dim=-1)
            chosen_s0 = torch.where(use_pred_mask.view(B, 1, 1, 1), p0, target_tokens["scale_0"])
        else:
            chosen_s0 = target_tokens["scale_0"]
        
        curr_ll = tokenizer.inverse_step(curr_ll, chosen_s0, 0)

        if self.backbone == "transformer":
            s1_logits = self.pred_s1(curr_ll.flatten(2).transpose(1, 2), cond_s1, g_style)
        else:
            s1_h = self.pred_s1(curr_ll.flatten(2).transpose(1, 2), cond_s1)
            s1_logits = self.head_s1(s1_h)
            
        if self.training and use_pred_prob > 0.0:
            p1 = s1_logits.argmax(dim=-1)
            chosen_s1 = torch.where(use_pred_mask.view(B, 1, 1, 1), p1, target_tokens["scale_1"])
        else:
            chosen_s1 = target_tokens["scale_1"]
            
        curr_ll = tokenizer.inverse_step(curr_ll, chosen_s1, 1)

        if self.backbone == "transformer":
            s2_logits = self.pred_s2(curr_ll.flatten(2).transpose(1, 2), cond_s2, g_style)
        else:
            s2_h = self.pred_s2(curr_ll.flatten(2).transpose(1, 2), cond_s2)
            s2_logits = self.head_s2(s2_h)
        
        return {"root": root_logits, "scale_0": s0_logits, "scale_1": s1_logits, "scale_2": s2_logits}

    @staticmethod
    def _sample_logits(logits: torch.Tensor, temp: float, top_k: int, top_p: float, vocab_size: int) -> torch.Tensor:
        z = logits.float() / max(temp, 1e-4)
        
        if temp == 0.0:
            return torch.argmax(z, dim=-1)
            
        if top_k > 0:
            v, indices = torch.topk(z, min(top_k, vocab_size), dim=-1)
            
            if 0.0 < top_p < 1.0:
                probs_k = F.softmax(v, dim=-1)
                cumulative_probs = torch.cumsum(probs_k, dim=-1)
                
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                
                v = torch.where(mask, torch.full_like(v, float("-inf")), v)
                
            gumbel_noise = -torch.empty_like(v).exponential_().log()
            samp_idx_in_k = torch.argmax(v + gumbel_noise, dim=-1)
            
            samp = torch.gather(indices, -1, samp_idx_in_k.unsqueeze(-1)).squeeze(-1)
            return samp

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(z, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            
            indices_to_remove = mask.scatter(-1, sorted_indices, mask)
            z = torch.where(indices_to_remove, torch.full_like(z, float("-inf")), z)

        gumbel_noise = -torch.empty_like(z).exponential_().log()
        return torch.argmax(z + gumbel_noise, dim=-1)

    @torch.inference_mode() 
    def generate(
        self,
        cond_outputs: Dict[str, torch.Tensor],
        tokenizer,
        temp: float = 1.0,
        temps: Optional[Dict[str, float]] = None,
        top_k: int = 50,
        top_p: float = 0.9,
        ret_recon: bool = False,
        cfg_scale: float = 1.0, 
        **kwargs,
    ):
        B = cond_outputs["global_style"].shape[0]

        if temps is None:
            temps = {"root": temp, "scale_0": temp, "scale_1": temp, "scale_2": temp}

        tokens: Dict[str, torch.Tensor] = {}

        do_cfg = self.cfg_training and cfg_scale > 1.0
        if do_cfg:
            raw_g = torch.cat([cond_outputs["global_style"], self.null_global.expand(B, -1)], dim=0)
            cond_root = torch.cat([cond_outputs["root_ll"].flatten(2).transpose(1, 2), self.null_root_ll.expand(B, 36, -1)], dim=0)
            cond_s0 = torch.cat([cond_outputs["scale_0_packed"].flatten(2).transpose(1, 2), self.null_s0_packed.expand(B, 36, -1)], dim=0)
            cond_s1 = torch.cat([cond_outputs["scale_1_packed"].flatten(2).transpose(1, 2), self.null_s1_packed.expand(B, 144, -1)], dim=0)
            cond_s2 = torch.cat([cond_outputs["scale_2_packed"].flatten(2).transpose(1, 2), self.null_s2_packed.expand(B, 576, -1)], dim=0)
        else:
            raw_g = cond_outputs["global_style"]
            cond_root = cond_outputs["root_ll"].flatten(2).transpose(1, 2)
            cond_s0 = cond_outputs["scale_0_packed"].flatten(2).transpose(1, 2)
            cond_s1 = cond_outputs["scale_1_packed"].flatten(2).transpose(1, 2)
            cond_s2 = cond_outputs["scale_2_packed"].flatten(2).transpose(1, 2)

        g_style = self.global_proj(raw_g) if hasattr(self, 'global_proj') else None

        def _apply_cfg_and_sample(logits, tscale, vocab_sz):
            if do_cfg:
                cond_logits, uncond_logits = logits.chunk(2, dim=0)
                logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            return self._sample_logits(logits, tscale, top_k, top_p, vocab_sz)

        # ---------------------------------------------------------
        # ROOT
        # ---------------------------------------------------------
        if self.backbone == "transformer":
            r_state = self.root_learned_state.expand(B * 2 if do_cfg else B, -1, -1)
            root_logits = self.pred_root(r_state, cond_root, g_style)
            v_sz = self.pred_root.vocab
        else:
            root_hidden = g_style.unsqueeze(1).expand(B * 2 if do_cfg else B, 36, -1)
            root_h = self.pred_root(root_hidden, cond_root)
            root_logits = self.head_root(root_h)
            v_sz = root_logits.size(-1)

        tokens["root"] = _apply_cfg_and_sample(root_logits, temps["root"], v_sz)
        curr_ll = tokenizer.decode_root(tokens["root"])

        # ---------------------------------------------------------
        # SCALE 0
        # ---------------------------------------------------------
        h_in = curr_ll.flatten(2).transpose(1, 2)
        if do_cfg: h_in = h_in.repeat(2, 1, 1)

        if self.backbone == "transformer":
            s0_logits = self.pred_s0(h_in, cond_s0, g_style)
            v_sz = self.pred_s0.vocab
        else:
            s0_h = self.pred_s0(h_in, cond_s0)
            s0_logits = self.head_s0(s0_h)
            v_sz = s0_logits.size(-1)

        tokens["scale_0"] = _apply_cfg_and_sample(s0_logits, temps["scale_0"], v_sz)
        curr_ll = tokenizer.inverse_step(curr_ll, tokens["scale_0"], 0)

        # ---------------------------------------------------------
        # SCALE 1
        # ---------------------------------------------------------
        h_in = curr_ll.flatten(2).transpose(1, 2)
        if do_cfg: h_in = h_in.repeat(2, 1, 1)

        if self.backbone == "transformer":
            s1_logits = self.pred_s1(h_in, cond_s1, g_style)
            v_sz = self.pred_s1.vocab
        else:
            s1_h = self.pred_s1(h_in, cond_s1)
            s1_logits = self.head_s1(s1_h)
            v_sz = s1_logits.size(-1)

        tokens["scale_1"] = _apply_cfg_and_sample(s1_logits, temps["scale_1"], v_sz)
        curr_ll = tokenizer.inverse_step(curr_ll, tokens["scale_1"], 1)

        # ---------------------------------------------------------
        # SCALE 2
        # ---------------------------------------------------------
        h_in = curr_ll.flatten(2).transpose(1, 2)
        if do_cfg: h_in = h_in.repeat(2, 1, 1)

        if self.backbone == "transformer":
            s2_logits = self.pred_s2(h_in, cond_s2, g_style)
            v_sz = self.pred_s2.vocab
        else:
            s2_h = self.pred_s2(h_in, cond_s2)
            s2_logits = self.head_s2(s2_h)
            v_sz = s2_logits.size(-1)

        tokens["scale_2"] = _apply_cfg_and_sample(s2_logits, temps["scale_2"], v_sz)

        if ret_recon:
            curr_ll = tokenizer.inverse_step(curr_ll, tokens["scale_2"], 2)
            recon = tokenizer.features_to_pixels(curr_ll)
            return tokens, recon
        else:
            return tokens
