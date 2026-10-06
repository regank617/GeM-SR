"""Thin CLI: load regional cubes + precomputed GeM recons and plot."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.chm_pair import extract_chm_pair  # noqa: E402
from figures.cube import load_hhdc_cube  # noqa: E402
from figures.plot_cross_region import plot_three_regions_compact  # noqa: E402


def calculate_psnr(mse, data_range=1.0):
    if mse == 0:
        return float("inf")
    return 10 * math.log10(data_range**2 / mse)


def compute_lpips(gt, pred, loss_fn):
    if gt.shape[1] == 1:
        gt = gt.repeat(1, 3, 1, 1)
        pred = pred.repeat(1, 3, 1, 1)
    gt = gt * 2 - 1
    pred = pred * 2 - 1
    return loss_fn(gt, pred).squeeze()


def compute_metrics(pred_np, gt_np, ssim_metric, lpips_model, device):
    # Cross-region metrics: CHM heights normalized by 128 (metres in figure colorbar use separate scaling).
    pred_t = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).float().to(device) / 128.0
    gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float().to(device) / 128.0
    pred_t = torch.clamp(pred_t, 0, 1)
    gt_t = torch.clamp(gt_t, 0, 1)
    with torch.no_grad():
        mse = F.mse_loss(pred_t, gt_t).item()
        psnr = calculate_psnr(mse, data_range=1.0)
        ssim = ssim_metric(pred_t, gt_t).item()
        lp = compute_lpips(gt_t, pred_t, lpips_model).item()
    return {"psnr": psnr, "ssim": ssim, "lpips": lp}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-regional GeM-SR ablation figure.")
    parser.add_argument(
        "--cube-dir",
        type=Path,
        default=Path("samples/test_cubes"),
        help="Directory with regional cubes and *_recon.npy files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cross_regional_ablation.pdf"),
        help="Output PDF path",
    )
    parser.add_argument("--skip-metrics", action="store_true", help="Plot only; skip GPU metric computation")
    args = parser.parse_args()

    cube_dir = args.cube_dir
    if not cube_dir.is_dir():
        raise FileNotFoundError(
            f"{cube_dir} not found. Expected the regional cubes + "
            "california/florida/maryland_recon.npy (shipped in samples/test_cubes/), or pass --cube-dir."
        )

    region_specs = [
        ("California", "California_test_cube.npz", "cube", "california_recon.npy", "Conifer / Mountains"),
        ("Maryland", "Maryland_test_cube.npz", "arr_0", "maryland_recon.npy", "Temperate Woodlands"),
        ("Florida", "Florida_test_cube_2.npz", "cube", "florida_recon.npy", "Tropical / Mangrove"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.skip_metrics else "cpu")
    ssim_metric = None
    lpips_model = None
    if not args.skip_metrics and device.type == "cuda":
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device)
        ssim_metric.eval()
        lpips_model = lpips.LPIPS(net="vgg").to(device)
        lpips_model.eval()

    regions = []
    for name, cube_file, array_key, recon_file, tag in region_specs:
        cube_path = cube_dir / cube_file
        recon_path = cube_dir / recon_file
        if not cube_path.is_file() or not recon_path.is_file():
            missing = [str(p) for p in (cube_path, recon_path) if not p.is_file()]
            raise FileNotFoundError(
                "STOP: missing cross-region artifacts:\n"
                + "\n".join(f"  - {p}" for p in missing)
                + "\nRegenerate the reconstructions with scripts/run_cross_region_mosaic.py (GPU)."
            )
        cube = load_hhdc_cube(cube_path, array_key=array_key)
        pair = extract_chm_pair(cube)
        recon = np.load(recon_path)
        entry = dict(
            name=name,
            tag=tag,
            lr=pair["lr_chm"],
            recon=recon,
            hr=pair["hr_chm"],
        )
        if ssim_metric is not None and lpips_model is not None:
            metrics = compute_metrics(recon, pair["hr_chm"], ssim_metric, lpips_model, device)
            entry.update(psnr=metrics["psnr"], ssim=metrics["ssim"])
        regions.append(entry)

    plot_three_regions_compact(regions, save_path=str(args.output), show=False)
    print(f"Done! Saved {args.output}")


if __name__ == "__main__":
    main()
