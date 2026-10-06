from __future__ import annotations
"""
``VARSRWrapper`` for training and eval notebooks.

Loads ``VAR_RoPE`` from ``benchmarks/varsr/repo``. Clears any cached ``models``
package (SwinIR, SR3, old VAR) so ``models.var`` resolves to VARSR, not
``network_swinir``. Import this **before** constructing the LiDAR VQVAE
(same order as ``train_varsr.py``).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.varsr.varsr_repo_imports import import_varsr_var_rope

VAR_RoPE = import_varsr_var_rope()


class VARSRWrapper(nn.Module):
    def __init__(
        self,
        vae_local,
        embed_dim: int = 256,
        patch_nums=(1, 2, 3, 4, 6, 12),
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.var = VAR_RoPE(
            vae_local=vae_local,
            num_classes=1,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_nums=patch_nums,
            in_channels=1,
        )

    def _apply(self, fn):
        # VAR_RoPE keeps ``vae_local`` in ``vae_proxy`` tuple only — not a registered child — so move it here.
        ret = super()._apply(fn)
        self.var.vae_proxy[0]._apply(fn)
        return ret

    def forward(self, lr_img: torch.Tensor, x_BLCv_wo_first_l: torch.Tensor, gt_idx_Bl: list) -> tuple:
        lr_up = F.interpolate(lr_img, size=(48, 48), mode="bilinear", align_corners=False)
        B = lr_img.size(0)
        label_B = torch.zeros(B, dtype=torch.long, device=lr_img.device)
        return self.var(
            x_BLCv_wo_first_l,
            label_B,
            lr_up,
            text_hidden=None,
            last_layer_gt=gt_idx_Bl[-1],
            last_layer_gt_discrete=gt_idx_Bl[-2],
        )

    @torch.no_grad()
    def generate(
        self,
        lr_img: torch.Tensor,
        *,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        g_seed: Optional[int] = None,
        more_smooth: bool = False,
    ) -> torch.Tensor:
        """
        LR-conditioned autoregressive sampling (VARSR ``autoregressive_infer_cfg``).

        Uses the same 48×48 bilinear LR upsample as :meth:`forward`. Returns a tensor
        ``[B, C, H, W]`` in **[0, 1]** (``C`` matches the VQVAE, typically 1).

        ``text_hidden`` / ``negative_text`` are unused; ``label_B`` is all zeros (dummy class).
        """
        lr_up = F.interpolate(lr_img, size=(48, 48), mode="bilinear", align_corners=False)
        B = lr_up.size(0)
        label_B = torch.zeros(B, dtype=torch.long, device=lr_up.device)
        return self.var.autoregressive_infer_cfg(
            B,
            text_hidden=None,
            lr_inp=lr_up,
            negative_text=None,
            label_B=label_B,
            g_seed=g_seed,
            cfg=cfg,
            top_k=top_k,
            top_p=top_p,
            more_smooth=more_smooth,
        )
