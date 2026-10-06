"""Native 32×16 → 48×48 tile unfold / hard-stitch for mosaic inference."""

from __future__ import annotations

import torch


def lr_tiles(lr_full: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Unfold LR mosaic into native model tiles (N, 1, 32, 16)."""
    _batch, _channels, height, width = lr_full.shape
    n_h, n_w = height // 32, width // 16
    tiles = lr_full.unfold(2, 32, 32).unfold(3, 16, 16)
    tiles = tiles.contiguous().view(-1, 1, 32, 16)
    return tiles, n_h, n_w


def remesh_tiles(recon_tiles: torch.Tensor, batch: int, n_h: int, n_w: int) -> torch.Tensor:
    """Hard-stitch HR tiles back into a full mosaic."""
    recon_full = recon_tiles.view(batch, n_h, n_w, 1, 48, 48)
    recon_full = recon_full.permute(0, 3, 1, 4, 2, 5).contiguous()
    return recon_full.view(batch, 1, n_h * 48, n_w * 48)
