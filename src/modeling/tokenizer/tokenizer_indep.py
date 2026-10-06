from __future__ import annotations
"""
Ablated tokenizer: **independent per-channel VQ** on the 4-D fusion latent.

Each of the four subband-aligned latent channels uses its own Euclidean codebook
(``embedding_dim=1``). ``fsq_details`` is shimmed from the first detail VQ for
API compatibility (same per-codebook ``vocab_size`` as the base tokenizer).
"""


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.modules.wavelet import HaarWaveletTransform2D, InverseHaarWaveletTransform2D
from src.modeling.tokenizer.tokenizer_v5_layer_config import TokenizerV5LayerConfig


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
        return x_normed.view(B, C, H, W) * self.weight


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


class _VQCodeShim:
    """Read-only FSQ-like view: ``vocab_size`` + ``indices_to_codes`` for tooling."""

    def __init__(self, vq: "EuclideanVectorQuantize2d"):
        self.vocab_size = int(vq.num_embeddings)
        self._vq = vq

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return self._vq.indices_to_codes(indices)


class EuclideanVectorQuantize2d(nn.Module):
    """
    VQ on ``[B, D, H, W]`` latents. Returns STE-quantized tensor, flat indices ``[B, H, W]``,
    and a scalar VQ loss (commitment + codebook), averaged over spatial batch.

    Latents are passed through ``tanh`` before distance / assignment (same bounding idea as
    ``FSQ``), so encoder activations stay in a stable range and VQ losses do not explode when
    conv stacks output large values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        commitment_cost: float = 0.25,
        use_tanh_bound: bool = True,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_embeddings = int(num_embeddings)
        self.commitment_cost = float(commitment_cost)
        self.use_tanh_bound = bool(use_tanh_bound)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Map discrete indices to code vectors, FSQ-compatible trailing layout ``[..., D]``.

        ``indices`` may be ``[B, H, W]``, ``[N]``, etc. Output has the same leading shape
        plus a last dim of ``embedding_dim`` (matches ``FSQ.indices_to_codes`` for 3D inputs).
        """
        orig_shape = indices.shape
        flat = indices.reshape(-1).long()
        z = self.embedding(flat)
        return z.view(*orig_shape, self.embedding_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z : Tensor [B, D, H, W]

        Returns
        -------
        z_q_ste : Tensor [B, D, H, W]
        indices : LongTensor [B, H, W]
        vq_loss : scalar tensor (commitment + codebook)
        """
        B, D, H, W = z.shape
        if D != self.embedding_dim:
            raise ValueError(f"VQ expected embedding_dim={self.embedding_dim}, got z with D={D}")

        z_b = torch.tanh(z) if self.use_tanh_bound else z
        z_ch = z_b.permute(0, 2, 3, 1).contiguous()  # [B, H, W, D]
        flat = z_ch.view(-1, D)

        emb = self.embedding.weight  # [K, D]
        # Squared Euclidean distance: ||z||^2 + ||e||^2 - 2 z·e
        d2 = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ emb.t()
            + emb.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices_flat = d2.argmin(dim=1)  # [B*H*W]
        z_q_flat = self.embedding(indices_flat)
        indices = indices_flat.view(B, H, W)

        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # STE in the same bounded space we quantized (gradients flow through tanh into z).
        z_q_ste = z_b + (z_q - z_b).detach()

        commit = F.mse_loss(z_q, z_b)
        codebook = F.mse_loss(z_q.detach(), z_b)
        vq_loss = codebook + self.commitment_cost * commit
        return z_q_ste, indices, vq_loss


class LidarTokenizerIndepVQ(nn.Module):
    """
    Haar pyramid + fusion CNN tokenizer with **VQ** at root (3-D) and **four independent**
    1-D detail codebooks (one per fusion latent channel).
    """

    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 4,
        hidden_dim: int = 64,
        num_scales: int = 3,
        codebook_size_root: int = 175,
        codebook_size_detail: int = 512,
        commitment_cost: float = 0.25,
        vq_loss_weight: float = 1.0,
        vq_use_tanh_bound: bool = True,
        inter_loss_weight: float = 2.0,
        use_refiner: bool = False,
        apply_sparsity_mask_to_latents: bool = True,
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
        self.inter_loss_weight = inter_loss_weight
        self.use_refiner = use_refiner
        self.apply_sparsity_mask_to_latents = bool(apply_sparsity_mask_to_latents)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.commitment_cost = commitment_cost
        self.vq_loss_weight = float(vq_loss_weight)
        self.vq_use_tanh_bound = bool(vq_use_tanh_bound)

        self.dwt = HaarWaveletTransform2D()
        self.idwt = InverseHaarWaveletTransform2D()

        if zp:
            _root_norm: nn.Module = ZeroPreservingRMSNorm2D(3, groups=1)
        else:
            _root_norm = nn.GroupNorm(num_groups=1, num_channels=3)

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
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        self.vq_root = EuclideanVectorQuantize2d(
            embedding_dim=3,
            num_embeddings=codebook_size_root,
            commitment_cost=commitment_cost,
            use_tanh_bound=self.vq_use_tanh_bound,
        )
        self.vq_detail = nn.ModuleList(
            [
                EuclideanVectorQuantize2d(
                    embedding_dim=1,
                    num_embeddings=codebook_size_detail,
                    commitment_cost=commitment_cost,
                    use_tanh_bound=self.vq_use_tanh_bound,
                )
                for _ in range(latent_dim)
            ]
        )
        self.fsq_root = _VQCodeShim(self.vq_root)
        self.fsq_details = _VQCodeShim(self.vq_detail[0])

    def _calc_entropy_and_vocab_used(self, indices: torch.Tensor, vocab_size: int):
        if indices.numel() == 0:
            return 0.0, 0
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

        total_inter_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_vq_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        all_indices = []
        debug_stats: dict = {"diag/input_sparsity": input_sparsity}

        residual_root = gt_pyramid[0][0]
        z_root_enc = self.proj_ll_in(residual_root)

        if diagnostics is not None:
            diagnostics["3_root_z_enc"] = self._tensor_stats(z_root_enc)

        z_q, idx, vq_l_root = self.vq_root(z_root_enc)
        total_vq_loss = total_vq_loss + vq_l_root
        curr_ll_recon = self.proj_ll_out(z_q)

        if diagnostics:
            test_dict["curr_ll_recon_root"] = curr_ll_recon.clone().detach().cpu()

        with torch.no_grad():
            ent, vocab_used = self._calc_entropy_and_vocab_used(idx, self.vq_root.num_embeddings)
            debug_stats["s0_d0/ent"] = ent
            debug_stats["s0_d0/vocab_used"] = vocab_used

        all_indices.append(torch.stack([idx], dim=-1))

        final_active_mask = None
        for s_i, (gt_ll_next, gt_details) in enumerate(gt_pyramid):
            inter_loss_step = F.mse_loss(curr_ll_recon, gt_ll_next.detach())
            total_inter_loss = total_inter_loss + inter_loss_step
            debug_stats[f"loss/inter_scale_{s_i+1}"] = inter_loss_step.item()

            ll_error = gt_ll_next - curr_ll_recon
            packed_target = torch.cat([ll_error, gt_details], dim=1)

            active_mask = mask_pyramid[s_i] > 0

            if s_i == len(gt_pyramid) - 1:
                final_active_mask = active_mask

            z_enc = self.fusion_encoders[s_i](packed_target)

            ll_latent_l1 = torch.abs(z_enc[:, 0]).mean()
            total_inter_loss = total_inter_loss + (ll_latent_l1 * 0.001)
            debug_stats[f"loss/s{s_i+1}_ll_latent_l1"] = ll_latent_l1.item()

            z_q_list, active_idx_list, vq_l_list = [], [], []
            for c in range(self.latent_dim):
                z_c, idx_c, l_c = self.vq_detail[c](z_enc[:, c : c + 1, :, :])
                z_q_list.append(z_c)
                active_idx_list.append(idx_c)
                vq_l_list.append(l_c)

            z_q = torch.cat(z_q_list, dim=1)
            active_idx = torch.stack(active_idx_list, dim=-1)
            vq_l_det = sum(vq_l_list)
            total_vq_loss = total_vq_loss + vq_l_det

            if self.apply_sparsity_mask_to_latents:
                z_q_masked = z_q * active_mask.float()
            else:
                z_q_masked = z_q

            all_indices.append(active_idx)

            with torch.no_grad():
                ents, vus = [], []
                for c in range(self.latent_dim):
                    ch_idx = active_idx[..., c]
                    valid_idx = ch_idx[active_mask.squeeze(1)]
                    ent_c, vu_c = self._calc_entropy_and_vocab_used(
                        valid_idx, self.vq_detail[c].num_embeddings
                    )
                    ents.append(ent_c)
                    vus.append(vu_c)
                debug_stats[f"s{s_i+1}/ent"] = sum(ents) / len(ents)
                debug_stats[f"s{s_i+1}/vocab_used"] = max(vus)

            z_out = self.fusion_decoders[s_i](z_q_masked)
            curr_ll_corrected = curr_ll_recon + z_out[:, 0:1]
            full_recon = torch.cat([curr_ll_corrected, z_out[:, 1:4]], dim=1)
            curr_ll_recon = self.idwt(full_recon)

            if diagnostics:
                test_dict[f"curr_ll_recon_{s_i}"] = curr_ll_recon.clone().detach().cpu()

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
            recon_terms = mse_base + mse_refined + (total_inter_loss * self.inter_loss_weight)
        else:
            mse_refined = mse_base
            recon_terms = mse_base + (total_inter_loss * self.inter_loss_weight)

        vq_term = total_vq_loss * self.vq_loss_weight
        loss = recon_terms + vq_term

        with torch.no_grad():
            debug_stats["loss/mse_base"] = mse_base.item()
            debug_stats["loss/mse_refined"] = mse_refined.item()
            if self.use_refiner:
                debug_stats["loss/refiner_improvement_percent"] = (
                    (mse_base - mse_refined) / (mse_base + 1e-6)
                ).item() * 100
            else:
                debug_stats["loss/refiner_improvement_percent"] = 0.0
            debug_stats["loss/vq_total"] = total_vq_loss.item()

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
            "total_vq_loss": total_vq_loss,
            "total_commit_loss": total_vq_loss,
            "active_mask": final_active_mask,
            "diagnostics": diagnostics,
            "test_dict": test_dict,
        }

        if test:
            return x_final, (x_recon_raw, x_final), info_dict
        return x_final, loss, info_dict

    def _tensor_stats(self, t: torch.Tensor):
        if t.numel() == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        flat = t.detach().reshape(-1)
        return {
            "min": flat.min().item(),
            "max": flat.max().item(),
            "mean": flat.mean().item(),
            "std": flat.std().item(),
        }

    @torch.no_grad()
    def get_hierarchical_codes(self, high_res_img, vbr_threshold=None, vbr_ratio=None):
        del vbr_threshold, vbr_ratio
        _, _, info_dict = self.forward(high_res_img, test=True)
        indices = info_dict["indices"]
        codes = {"root": indices[0].permute(0, 3, 1, 2).contiguous()}
        for i in range(1, self.num_scales + 1):
            codes[f"scale_{i-1}"] = indices[i].permute(0, 3, 1, 2).contiguous()
        return codes

    def decode_root(self, root_indices: torch.Tensor) -> torch.Tensor:
        """``root_indices``: [B, 1, H, W] -> LL recon [B, 1, H, W]."""
        idx = root_indices[:, 0, :, :]
        z_q = self.vq_root.indices_to_codes(idx)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.proj_ll_out(z_q)

    def inverse_step(self, current_ll_features: torch.Tensor, scale_indices: torch.Tensor, scale_idx: int):
        """
        One IDWT step from discrete detail codes. ``scale_indices`` must be ``[B, 4, H, W]``
        (one index map per latent channel / subband codebook).
        """
        z_q_list = []
        for c in range(self.latent_dim):
            idx_c = scale_indices[:, c, :, :]
            z_c = self.vq_detail[c].indices_to_codes(idx_c).permute(0, 3, 1, 2).contiguous()
            z_q_list.append(z_c)
        z_q = torch.cat(z_q_list, dim=1)
        z_out = self.fusion_decoders[scale_idx](z_q)
        curr_ll_corrected = current_ll_features + z_out[:, 0:1]
        full_recon = torch.cat([curr_ll_corrected, z_out[:, 1:4]], dim=1)
        return self.idwt(full_recon)

    def features_to_pixels(
        self, final_features: torch.Tensor, refiner_strength: float = 1.0, ret_prerefiner: bool = False
    ):
        x_recon_raw = final_features
        if self.use_refiner:
            refiner_residual = self.refiner(x_recon_raw)
            x_refined = x_recon_raw + (refiner_residual * refiner_strength)
        else:
            x_refined = x_recon_raw
        final_recon = torch.clamp(x_refined, min=0.0, max=1.0)
        if ret_prerefiner:
            return final_recon, torch.clamp(x_recon_raw, 0.0, 1.0)
        return final_recon
