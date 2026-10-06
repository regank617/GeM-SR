"""
Previous SwinIR benchmark wrapper (pixelshuffle + asymmetric conv).

Kept for reference / reverting. The active training script uses
``train_swinir.SwinIRWrapper`` (deep-feature upsample + global residual).
"""
import os
import sys
import torch.nn as nn
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from benchmarks.import_isolation import clear_benchmark_namespaces, insert_path_first

_SWINIR_REPO = str(Path(__file__).resolve().parent / "repo")
clear_benchmark_namespaces()
insert_path_first(_SWINIR_REPO)
from models.network_swinir import SwinIR  # noqa: E402


class SwinIRWrapperLegacy(nn.Module):
    """Original wrapper: 3× pixelshuffle SwinIR + stride-(2,1) conv to 48×48."""

    def __init__(
        self,
        n_colors=1,
        window_size=8,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        mlp_ratio=2,
    ):
        super().__init__()
        self.swinir = SwinIR(
            upscale=3,
            in_chans=n_colors,
            img_size=(16, 32),
            window_size=window_size,
            img_range=1.0,
            depths=list(depths),
            embed_dim=embed_dim,
            num_heads=list(num_heads),
            mlp_ratio=mlp_ratio,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
        self.fix_aspect = nn.Conv2d(n_colors, n_colors, kernel_size=3, stride=(2, 1), padding=1)

    def forward(self, x):
        return self.fix_aspect(self.swinir(x))
