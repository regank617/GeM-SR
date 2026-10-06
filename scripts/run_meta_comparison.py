#!/usr/bin/env python3
"""Meta CHMv2 ECC alignment + comparison figure. Requires AWS tile + precomputed gemsr_recon.npy."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.chm_pair import extract_chm_pair  # noqa: E402
from figures.cube import load_coords, load_hhdc_cube  # noqa: E402
from figures.plot_meta import plot_meta_comparison  # noqa: E402

META_S3_PATH = "s3://dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3/chm/0320100332.tif"
META_LOCAL_TIF = "samples/meta_case_study/0320100332.tif"
PAD = 64


def download_meta_tile(local_tif: str = META_LOCAL_TIF, s3_path: str = META_S3_PATH) -> None:
    if os.path.exists(local_tif):
        return
    print("\n" + "=" * 70)
    print(f"NOTICE: {local_tif} not found.")
    print("Downloading from Meta's public AWS bucket to enable comparison...")
    print(f"Command: aws s3 cp --no-sign-request {s3_path} {local_tif}")
    print("=" * 70 + "\n")
    exit_code = os.system(f"aws s3 cp --no-sign-request {s3_path} {local_tif}")
    if exit_code != 0:
        raise FileNotFoundError("AWS download failed. Install aws-cli or download the GeoTIFF manually.")


def extract_and_downsample_meta_chm(
    meta_tiff_path: str,
    xy_grid: np.ndarray,
    src_crs: str = "EPSG:32618",
    pad: int = 0,
) -> np.ndarray:
    height, width, _ = xy_grid.shape
    pixel_size = 2.0
    top_left_x = xy_grid[0, 0, 0] - (pixel_size / 2.0) - pad * pixel_size
    top_left_y = xy_grid[0, 0, 1] + (pixel_size / 2.0) + pad * pixel_size
    out_h = height + 2 * pad
    out_w = width + 2 * pad
    target_transform = Affine(pixel_size, 0.0, top_left_x, 0.0, -pixel_size, top_left_y)
    meta_aligned_2m = np.zeros((out_h, out_w), dtype=np.float32)
    with rasterio.open(meta_tiff_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=meta_aligned_2m,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=src_crs,
            resampling=Resampling.max,
        )
    return np.clip(meta_aligned_2m, a_min=0.0, a_max=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta CHMv2 vs GeM-SR case study.")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("samples/meta_case_study"),
        help="Folder with maryland_cube.npz, maryland_coords.npz, gemsr_recon.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comparison_meta.png"),
        help="Output PNG path",
    )
    parser.add_argument("--skip-download", action="store_true", help="Do not attempt AWS download")
    args = parser.parse_args()

    case_dir = args.case_dir
    cube_path = case_dir / "maryland_cube.npz"
    coords_path = case_dir / "maryland_coords.npz"
    recon_path = case_dir / "gemsr_recon.npy"
    for required in (cube_path, coords_path, recon_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} not found.\n"
                "Expected in samples/meta_case_study/ (shipped with the repo), or pass --case-dir."
            )

    if not args.skip_download:
        os.makedirs(os.path.dirname(META_LOCAL_TIF), exist_ok=True)
        download_meta_tile()

    if not os.path.exists(META_LOCAL_TIF):
        raise FileNotFoundError(f"Meta GeoTIFF missing: {META_LOCAL_TIF}")

    coords = load_coords(coords_path)
    cube = load_hhdc_cube(cube_path, array_key="arr_0")
    pair = extract_chm_pair(cube)
    chm_gt = pair["hr_chm"]
    dtm_gt = pair["hr_dtm"]

    chm_meta_padded = extract_and_downsample_meta_chm(META_LOCAL_TIF, coords, pad=PAD)
    chm_recon = np.squeeze(np.load(recon_path)) * 128

    height, width = 288, 288
    gt = torch.from_numpy(cube[:, :height, :width]).float()
    chm_meta_center = chm_meta_padded[PAD : PAD + height, PAD : PAD + width]
    fixed_gray = chm_gt[:height, :width].astype(np.float32)
    moving_gray = chm_meta_center.astype(np.float32)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
    _, warp_matrix = cv2.findTransformECC(fixed_gray, moving_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
    warp_matrix_padded = warp_matrix.copy()
    warp_matrix_padded[0, 2] += PAD
    warp_matrix_padded[1, 2] += PAD
    chm_meta = (
        cv2.warpAffine(
            chm_meta_padded,
            warp_matrix_padded,
            (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        * 2
    )

    profile_row = 3 * gt.shape[1] // 4
    wave_slice = gt[:, profile_row, :].numpy()

    plot_meta_comparison(
        chm_gt[:height, :width],
        dtm_gt[:height, :width],
        chm_meta,
        chm_recon[:height, :width],
        wave_slice,
        save_path=str(args.output),
        show=False,
    )


if __name__ == "__main__":
    main()
