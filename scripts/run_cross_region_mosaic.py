#!/usr/bin/env python3
"""Generate CA / FL / MD GeM-SR mosaic recons (*.npy). Requires CUDA + GEMSR_MODEL_ROOT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.chm_pair import extract_chm_pair  # noqa: E402
from figures.cube import load_hhdc_cube  # noqa: E402
from figures.models import apply_model_root, infer_mosaic, load_gemsr_wrappers  # noqa: E402

REGION_CUBES = {
    "california": ("California_test_cube.npz", "cube"),
    "florida": ("Florida_test_cube_2.npz", "cube"),
    "maryland": ("Maryland_test_cube.npz", "arr_0"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GeM-SR mosaic inference on regional test cubes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="lidar_chm_models tree. Alternative to GEMSR_MODEL_ROOT env var.",
    )
    parser.add_argument(
        "--cube-dir",
        type=Path,
        default=Path("samples/test_cubes"),
        help="Directory with California/Florida/Maryland *_cube.npz files",
    )
    parser.add_argument(
        "--temps",
        choices=("rs", "perc"),
        default="rs",
        help="GeM-SR temperature preset (Uniform joint_2)",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="recon",
        help="Output filename suffix, e.g. california_recon.npy",
    )
    args = parser.parse_args()

    apply_model_root(args.model_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Regional mosaic inference requires CUDA.")

    cube_dir = args.cube_dir
    if not cube_dir.is_dir():
        raise FileNotFoundError(
            f"Cube directory not found: {cube_dir}\n"
            "Regional test cubes ship in samples/test_cubes/; otherwise pass --cube-dir."
        )

    gemsr_rs, gemsr_perc = load_gemsr_wrappers(device)
    model = gemsr_rs if args.temps == "rs" else gemsr_perc

    for region, (filename, array_key) in REGION_CUBES.items():
        cube_path = cube_dir / filename
        if not cube_path.is_file():
            print(f"[skip] missing {cube_path}")
            continue
        cube = load_hhdc_cube(cube_path, array_key=array_key)
        pair = extract_chm_pair(cube)
        recon = infer_mosaic(model, pair["lr_tensor"], device)
        out_path = cube_dir / f"{region}_{args.out_suffix}.npy"
        np.save(out_path, recon.squeeze().cpu().numpy())
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
