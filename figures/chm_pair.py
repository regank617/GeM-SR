"""Shared forward-model kwargs and CHM extraction for figure scripts."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from jobs.extract_chm import FORWARD_KWARGS
from src.data.canopy_plots import create_chm
from src.data.forward_model import LidarForwardImagingModel

# Re-export for figure scripts (same values as jobs/extract_chm.py).
__all__ = ["FORWARD_KWARGS", "extract_chm_pair", "make_forward_model"]


def make_forward_model() -> LidarForwardImagingModel:
    return LidarForwardImagingModel(**FORWARD_KWARGS)


def extract_chm_pair(
    cube_chw: np.ndarray,
    *,
    canopy_percentile: float = 0.98,
    forward_model: LidarForwardImagingModel | None = None,
) -> Dict[str, Any]:
    """
    Degrade a (128, H, W) cube and extract LR/HR RH98 CHMs.

    Returns raw CHM arrays (0–128 bin indices) plus LR/HR tensors scaled to [0, 1]
    for model input. Teaser viridis panels use raw CHM bins; Meta plots use ×0.5 m;
    cross-region metric code divides by 128 — do not unify units here.
    """
    if forward_model is None:
        forward_model = make_forward_model()

    cube_lr = forward_model(torch.from_numpy(cube_chw).float())
    hr_chm, hr_dtm, _, _ = create_chm(cube_chw, canopy_percentile=canopy_percentile)
    lr_chm, lr_dtm, _, _ = create_chm(
        cube_lr.numpy() if isinstance(cube_lr, torch.Tensor) else cube_lr,
        canopy_percentile=canopy_percentile,
    )

    device = torch.device("cpu")
    hr_tensor = torch.from_numpy(hr_chm).unsqueeze(0).unsqueeze(0).float().to(device) / 128.0
    lr_tensor = torch.from_numpy(lr_chm).unsqueeze(0).unsqueeze(0).float().to(device) / 128.0

    return {
        "hr_chm": hr_chm,
        "lr_chm": lr_chm,
        "hr_dtm": hr_dtm,
        "lr_dtm": lr_dtm,
        "hr_tensor": hr_tensor,
        "lr_tensor": lr_tensor,
        "cube_lr": cube_lr,
    }
