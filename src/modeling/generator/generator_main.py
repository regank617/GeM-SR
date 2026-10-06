from __future__ import annotations
"""
Unified hierarchical generator: **convnext** (fast CNN).
- Strictly Zero-Preserving (No biases, uses RMSNorms).
- No classifier-free guidance.
- Purely spatial ConvNeXt backbone.
- Global style ONLY used as the root initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
"""
generator_main is the main generator utilizing a ConvNeXt backbone.
generator_merged adds transformer backbone and CFG capability for ablations.
"""


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
    def __init__(self, dim: int, vocab_size: int, zero_idx: Optional[int] = None):
        super().__init__()
        self.vocab = vocab_size
        # Bias kept exclusively here so the model can predict the zero-bin logit from an empty vector
        self.code_proj = nn.Linear(dim, vocab_size, bias=True)
        nn.init.zeros_(self.code_proj.bias)
        #if zero_idx is not None:
        #    self.code_proj.bias.data[zero_idx] += 2.0
        nn.init.normal_(self.code_proj.weight, mean=0.0, std=0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.code_proj(x)


class SubBandFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=4, bias=False)

    def forward(self, x):
        return self.proj(x * self.gate(x))


class SpatialModulation(nn.Module):
    def __init__(self, x_dim, cond_dim):
        super().__init__()
        # SPARSITY FIX: bias=False ensures if cond is 0, gamma and beta are exactly 0
        self.proj = nn.Conv2d(cond_dim, x_dim * 2, kernel_size=1, bias=False)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return x * (1 + gamma) + beta


class FastConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.spatial_mod = SpatialModulation(dim, cond_dim)
        # SPARSITY FIX: bias=False everywhere
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = ChannelRMSNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

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
        **kwargs,
    ):
        super().__init__()
        self.rq = rq_depths
        self.H, self.W = spatial_shape
        self.embed_dim = embed_dim
        # SPARSITY FIX: No bias on input projection
        self.in_proj = nn.Linear(tokenizer_dim, embed_dim, bias=False)
        
        self.use_fusion = cond_in_dim > embed_dim
        if self.use_fusion:
            self.cond_fusion = SubBandFusion(cond_in_dim, embed_dim)
            processed_cond_dim = embed_dim
        else:
            processed_cond_dim = cond_in_dim
            
        self.blocks = nn.ModuleList(
            [FastConvNeXtBlock(dim=embed_dim, cond_dim=processed_cond_dim, mlp_ratio=mlp_ratio) for _ in range(depth)]
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


_DEFAULT_CONVNEXT_VOCABS = {"root": 175, "scale_0": 2401, "scale_1": 2401, "scale_2": 625}

def _resolve_vocab_sizes(
    tokenizer_layout: str, vocab_sizes: Optional[Dict[str, int]], vocab_size_scalar: int
) -> Dict[str, int]:
    keys = ("root", "scale_0", "scale_1", "scale_2")
    if vocab_sizes:
        if tokenizer_layout == "separate":
            out = {k: int(_DEFAULT_CONVNEXT_VOCABS[k]) for k in keys}
            for k, val in vocab_sizes.items():
                if k in out:
                    out[k] = int(val)
            return out
        out = {}
        for k in keys:
            if k not in vocab_sizes:
                raise ValueError(f"vocab_sizes must include {k} for joint layout")
            out[k] = int(vocab_sizes[k])
        return out
    if tokenizer_layout == "separate":
        return {k: int(_DEFAULT_CONVNEXT_VOCABS[k]) for k in keys}
    raise ValueError("tokenizer_layout='joint' requires explicit vocab_sizes dict")


class HierarchicalGeneratorFast(nn.Module):
    def __init__(
        self,
        tokenizer_dim: int = 1,
        embed_dim: int = 128,
        vocab_size: int = 2401,
        rq_depths: int = 1,
        zero_idx: Optional[int] = None,
        root_depth: int = 4,
        s0_depth: int = 4,
        s1_depth: int = 4,
        s2_depth: int = 4,
        mlp_ratio: int = 2,
        tokenizer_layout: str = "separate",
        scheduled_sampling_prob: float = 0.1,
        vocab_sizes: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.tokenizer_layout = tokenizer_layout
        self.rq = rq_depths
        self.scheduled_sampling_prob = float(scheduled_sampling_prob)
        
        c_dim = 4 * embed_dim
        self.embed_dim = embed_dim
        self.c_dim = c_dim

        self.global_proj = nn.Sequential(nn.Linear(c_dim, embed_dim, bias=False), nn.GELU(), RMSNorm1d(embed_dim))
        self.zero_idx = zero_idx
        vs = _resolve_vocab_sizes(self.tokenizer_layout, vocab_sizes, vocab_size)

        fast_kw = dict(tokenizer_dim=tokenizer_dim, embed_dim=embed_dim, rq_depths=rq_depths)
        root_kw = dict(tokenizer_dim=embed_dim, embed_dim=embed_dim, rq_depths=rq_depths)
        self.pred_root = FastScaleGenerator(spatial_shape=(6, 6), depth=root_depth, cond_in_dim=embed_dim, mlp_ratio=mlp_ratio, **root_kw)
        self.pred_s0 = FastScaleGenerator(spatial_shape=(6, 6), depth=s0_depth, cond_in_dim=c_dim, mlp_ratio=mlp_ratio, **fast_kw)
        self.pred_s1 = FastScaleGenerator(spatial_shape=(12, 12), depth=s1_depth, cond_in_dim=c_dim, mlp_ratio=mlp_ratio, **fast_kw)
        self.pred_s2 = FastScaleGenerator(spatial_shape=(24, 24), depth=s2_depth, cond_in_dim=c_dim, mlp_ratio=mlp_ratio, **fast_kw)
        
        self.head_root = StandardHead(embed_dim, vs["root"], zero_idx)
        self.head_s0 = StandardHead(embed_dim, vs["scale_0"], zero_idx)
        self.head_s1 = StandardHead(embed_dim, vs["scale_1"], zero_idx)
        self.head_s2 = StandardHead(embed_dim, vs["scale_2"], zero_idx)


    def forward(self, cond_outputs: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.Tensor], tokenizer):
        B = cond_outputs["global_style"].shape[0]
        device = cond_outputs["global_style"].device
        
        g_style = self.global_proj(cond_outputs["global_style"])
        cond_root = cond_outputs["root_ll"].flatten(2).transpose(1, 2)
        cond_s0 = cond_outputs["scale_0_packed"].flatten(2).transpose(1, 2)
        cond_s1 = cond_outputs["scale_1_packed"].flatten(2).transpose(1, 2)
        cond_s2 = cond_outputs["scale_2_packed"].flatten(2).transpose(1, 2)

        use_pred_prob = self.scheduled_sampling_prob if self.training else 0.0
        
        # ROOT INITIALIZATION ONLY
        root_hidden = g_style.unsqueeze(1).expand(B, 36, -1)
        root_h = self.pred_root(root_hidden, cond_root)
        root_logits = self.head_root(root_h)
        
        # OPTIMIZATION: Lazy evaluation. Only argmax if we are actually injecting predictions
        if self.training and use_pred_prob > 0.0:
            use_pred_mask = torch.rand(B, 1, 1, device=device) < use_pred_prob
            pr = root_logits.argmax(dim=-1)
            chosen_root = torch.where(use_pred_mask.view(B, 1, 1, 1), pr, target_tokens["root"])
        else:
            chosen_root = target_tokens["root"]
            use_pred_mask = None # For downstream scales if needed
        
        curr_ll = tokenizer.decode_root(chosen_root)

        s0_h = self.pred_s0(curr_ll.flatten(2).transpose(1, 2), cond_s0)
        s0_logits = self.head_s0(s0_h)
        
        if self.training and use_pred_prob > 0.0:
            p0 = s0_logits.argmax(dim=-1)
            chosen_s0 = torch.where(use_pred_mask.view(B, 1, 1, 1), p0, target_tokens["scale_0"])
        else:
            chosen_s0 = target_tokens["scale_0"]
        
        curr_ll = tokenizer.inverse_step(curr_ll, chosen_s0, 0)

        s1_h = self.pred_s1(curr_ll.flatten(2).transpose(1, 2), cond_s1)
        s1_logits = self.head_s1(s1_h)
        
        if self.training and use_pred_prob > 0.0:
            p1 = s1_logits.argmax(dim=-1)
            chosen_s1 = torch.where(use_pred_mask.view(B, 1, 1, 1), p1, target_tokens["scale_1"])
        else:
            chosen_s1 = target_tokens["scale_1"]
            
        curr_ll = tokenizer.inverse_step(curr_ll, chosen_s1, 1)

        s2_h = self.pred_s2(curr_ll.flatten(2).transpose(1, 2), cond_s2)
        s2_logits = self.head_s2(s2_h)
        
        return {"root": root_logits, "scale_0": s0_logits, "scale_1": s1_logits, "scale_2": s2_logits}

    @staticmethod
    def _sample_logits(logits: torch.Tensor, temp: float, top_k: int, top_p: float, vocab_size: int) -> torch.Tensor:
        # OPTIMIZATION: Ensure fp32 for stable math, prevents inf/NaN blowups
        z = logits.float() / max(temp, 1e-4)
        
        if temp == 0.0:
            return torch.argmax(z, dim=-1)
            
        if top_k > 0:
            # OPTIMIZATION: PyTorch topk returns values ALREADY SORTED descendingly.
            # We restrict all memory operations to K elements instead of V elements.
            v, indices = torch.topk(z, min(top_k, vocab_size), dim=-1)
            
            if 0.0 < top_p < 1.0:
                probs_k = F.softmax(v, dim=-1)
                cumulative_probs = torch.cumsum(probs_k, dim=-1)
                
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                
                v = torch.where(mask, torch.full_like(v, float("-inf")), v)
                
            # OPTIMIZATION: Gumbel-Max Trick.
            # Avoids F.softmax and torch.multinomial entirely, relying on fast argmax.
            gumbel_noise = -torch.empty_like(v).exponential_().log()
            samp_idx_in_k = torch.argmax(v + gumbel_noise, dim=-1)
            
            # Map the local K index back to the global Vocabulary index
            samp = torch.gather(indices, -1, samp_idx_in_k.unsqueeze(-1)).squeeze(-1)
            return samp

        # Fallback for pure Top-P without Top-K (not recommended for speed)
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

    @torch.inference_mode() # OPTIMIZATION: Much stricter/faster than no_grad()
    def generate(
        self,
        cond_outputs: Dict[str, torch.Tensor],
        tokenizer,
        temp: float = 1.0,
        temps: Optional[Dict[str, float]] = None,
        top_k: int = 50,
        top_p: float = 0.9,
        ret_recon: bool = False,
        **kwargs,
    ):
        B = cond_outputs["global_style"].shape[0]
        g_style = self.global_proj(cond_outputs["global_style"])        
        cond_root = cond_outputs["root_ll"].flatten(2).transpose(1, 2)
        cond_s0 = cond_outputs["scale_0_packed"].flatten(2).transpose(1, 2)
        cond_s1 = cond_outputs["scale_1_packed"].flatten(2).transpose(1, 2)
        cond_s2 = cond_outputs["scale_2_packed"].flatten(2).transpose(1, 2)

        if temps is None:
            temps = {"root": temp, "scale_0": temp, "scale_1": temp, "scale_2": temp}

        tokens: Dict[str, torch.Tensor] = {}

        def _sample_cnx(gen, head, hidden, cond, tscale):
            h = gen(hidden, cond)
            logits = head(h)
            v_sz = logits.size(-1)
            return self._sample_logits(logits, tscale, top_k, top_p, v_sz)

        # ROOT INITIALIZATION ONLY
        root_hidden = g_style.unsqueeze(1).expand(B, 36, -1)
        
        tokens["root"] = _sample_cnx(self.pred_root, self.head_root, root_hidden, cond_root, temps["root"])
        curr_ll = tokenizer.decode_root(tokens["root"])
        
        tokens["scale_0"] = _sample_cnx(self.pred_s0, self.head_s0, curr_ll.flatten(2).transpose(1, 2), cond_s0, temps["scale_0"])
        curr_ll = tokenizer.inverse_step(curr_ll, tokens["scale_0"], 0)
        
        tokens["scale_1"] = _sample_cnx(self.pred_s1, self.head_s1, curr_ll.flatten(2).transpose(1, 2), cond_s1, temps["scale_1"])
        curr_ll = tokenizer.inverse_step(curr_ll, tokens["scale_1"], 1)
        
        tokens["scale_2"] = _sample_cnx(self.pred_s2, self.head_s2, curr_ll.flatten(2).transpose(1, 2), cond_s2, temps["scale_2"])

        if ret_recon:
            curr_ll = tokenizer.inverse_step(curr_ll, tokens["scale_2"], 2)
            recon = tokenizer.features_to_pixels(curr_ll)
            return tokens, recon
        else:
            return tokens