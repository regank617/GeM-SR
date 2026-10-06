# -*- coding: utf-8 -*-
"""Physical CHM error accumulators for ``benchmark_hub.py``.

Keeps the original height-range quartiles (``floor(hr * 4)``) and adds:

- signed bias (mean SR − HR)
- R² vs GT height
- population-percentile tails (pixels with HR ≥ empirical quantile)

All running sums are in **normalized** [0, 1] height. Finalize multiplies MAE/RMSE/bias
by ``physical_meters_scale``. R² is dimensionless.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

# Tails: top 10% / 5% / 1% of GT pixels (thresholds = 90th / 95th / 99th percentiles).
DEFAULT_TAIL_QUANTILES: Tuple[float, ...] = (0.90, 0.95, 0.99)
TAIL_LABELS: Tuple[str, ...] = ("top10", "top5", "top1")
PERCENTILE_HIST_BINS = 100_000


def _empty_acc() -> Dict[str, float]:
    return {
        "n": 0.0,
        "sum_e": 0.0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_hr": 0.0,
        "sum_hr_sq": 0.0,
        "sum_sr_hr": 0.0,
    }


def _add_masked(acc: Dict[str, float], e: torch.Tensor, hr: torch.Tensor, sr: torch.Tensor, mask: torch.Tensor) -> None:
    n = int(mask.sum().item())
    if n == 0:
        return
    ev = e[mask]
    hv = hr[mask]
    sv = sr[mask]
    acc["n"] += n
    acc["sum_e"] += float(ev.sum().item())
    acc["sum_abs"] += float(ev.abs().sum().item())
    acc["sum_sq"] += float((ev * ev).sum().item())
    acc["sum_hr"] += float(hv.sum().item())
    acc["sum_hr_sq"] += float((hv * hv).sum().item())
    acc["sum_sr_hr"] += float((sv * hv).sum().item())


def finalize_acc(acc: Mapping[str, float], physical_meters_scale: float) -> Dict[str, Optional[float]]:
    n = int(acc["n"])
    if n <= 0:
        return {
            "n": 0,
            "mae_m": None,
            "rmse_m": None,
            "bias_m": None,
            "r2": None,
        }
    mse = acc["sum_sq"] / n
    mae = acc["sum_abs"] / n
    bias = acc["sum_e"] / n
    mean_hr = acc["sum_hr"] / n
    ss_tot = acc["sum_hr_sq"] - n * mean_hr * mean_hr
    if ss_tot <= 1e-12:
        r2: Optional[float] = None
    else:
        r2 = 1.0 - (acc["sum_sq"] / ss_tot)
    return {
        "n": n,
        "mae_m": mae * physical_meters_scale,
        "rmse_m": (max(mse, 0.0) ** 0.5) * physical_meters_scale,
        "bias_m": bias * physical_meters_scale,
        "r2": r2,
    }


def quartile_accs() -> List[Dict[str, float]]:
    return [_empty_acc() for _ in range(4)]


def percentile_accs(n_tails: int = 3) -> List[Dict[str, float]]:
    return [_empty_acc() for _ in range(n_tails)]


def accumulate_physical_errors(
    sr: torch.Tensor,
    hr: torch.Tensor,
    overall: Dict[str, float],
    quartiles: Sequence[Dict[str, float]],
    percentile_acc: Sequence[Dict[str, float]],
    percentile_thresholds: Sequence[float],
) -> None:
    """Update overall, quartile, and percentile-tail accumulators.

    ``hr`` / ``sr`` must already be in [0, 1]. Quartile bin ``i`` is
    ``floor(hr * 4)`` clamped to {0,1,2,3} (same as the original hub).
    Percentile tail ``j`` is ``hr >= percentile_thresholds[j]``.
    """
    flat_hr = hr.reshape(-1).clamp(0.0, 1.0)
    flat_sr = sr.reshape(-1)
    e = flat_sr - flat_hr
    ones = torch.ones_like(flat_hr, dtype=torch.bool)
    _add_masked(overall, e, flat_hr, flat_sr, ones)

    q = ((flat_hr * 4.0).floor().long()).clamp(max=3)
    for i in range(4):
        _add_masked(quartiles[i], e, flat_hr, flat_sr, q == i)

    for j, t in enumerate(percentile_thresholds):
        _add_masked(percentile_acc[j], e, flat_hr, flat_sr, flat_hr >= float(t))


def finalize_quartiles(
    quartiles: Sequence[Mapping[str, float]], physical_meters_scale: float
) -> Dict[str, Any]:
    rows = [finalize_acc(a, physical_meters_scale) for a in quartiles]
    return {
        "hr_quartile_edges_normalized": [0.0, 0.25, 0.5, 0.75, 1.0],
        "mae_height_bins": [r["mae_m"] for r in rows],
        "rmse_height_bins": [r["rmse_m"] for r in rows],
        "bias_height_bins": [r["bias_m"] for r in rows],
        "r2_height_bins": [r["r2"] for r in rows],
        "pixel_counts": [int(r["n"]) for r in rows],
    }


def finalize_percentile_tails(
    accs: Sequence[Mapping[str, float]],
    thresholds: Sequence[float],
    labels: Sequence[str],
    quantiles: Sequence[float],
    physical_meters_scale: float,
) -> Dict[str, Any]:
    rows = [finalize_acc(a, physical_meters_scale) for a in accs]
    return {
        "quantiles": [float(q) for q in quantiles],
        "labels": list(labels),
        "thresholds_normalized": {lab: float(t) for lab, t in zip(labels, thresholds)},
        "mae_height_bins": [r["mae_m"] for r in rows],
        "rmse_height_bins": [r["rmse_m"] for r in rows],
        "bias_height_bins": [r["bias_m"] for r in rows],
        "r2_height_bins": [r["r2"] for r in rows],
        "pixel_counts": [int(r["n"]) for r in rows],
    }


def hr_quantile_thresholds_from_histogram(
    hist: torch.Tensor, quantiles: Iterable[float]
) -> List[float]:
    """Invert a [0, 1] histogram CDF. ``hist[i]`` counts samples in bin i."""
    hist = hist.to(dtype=torch.int64).cpu()
    cdf = hist.cumsum(0)
    total = int(cdf[-1].item())
    n_bins = hist.numel()
    if total <= 0:
        return [0.0 for _ in quantiles]
    out: List[float] = []
    cdf_f = cdf.to(dtype=torch.float64)
    for q in quantiles:
        target = float(q) * float(total)
        idx = int(torch.searchsorted(cdf_f, torch.tensor(target, dtype=torch.float64), right=False).item())
        idx = min(max(idx, 0), n_bins - 1)
        out.append((idx + 1) / n_bins)
    return out


def accumulate_hr_histogram(hr: torch.Tensor, hist: torch.Tensor) -> None:
    n_bins = hist.numel()
    idx = (hr.reshape(-1).clamp(0.0, 1.0) * n_bins).long().clamp(max=n_bins - 1)
    hist.scatter_add_(0, idx.cpu(), torch.ones_like(idx, dtype=hist.dtype, device="cpu"))


def seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
