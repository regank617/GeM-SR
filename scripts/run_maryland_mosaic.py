#!/usr/bin/env python3
"""Generate Figure 1 mosaic .npy arrays (all models, seed 0). Requires CUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.chm_pair import extract_chm_pair  # noqa: E402
from figures.cube import load_hhdc_cube, resolve_cube_dir  # noqa: E402
from figures.models import MOSAIC_SAVE_ROOT, apply_model_root, run_maryland_mosaic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Maryland SERC mosaic inference for Figure 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="lidar_chm_models tree (ALL_MODELS/...). Alternative to GEMSR_MODEL_ROOT env var.",
    )
    parser.add_argument(
        "--cube-dir",
        type=Path,
        default=None,
        help="Folder with Maryland_test_cube.npz. Alternative to GEMSR_CUBE_DIR env var.",
    )
    parser.add_argument(
        "--cube-npz",
        type=Path,
        default=None,
        help="Direct path to Maryland_test_cube.npz (overrides --cube-dir)",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(MOSAIC_SAVE_ROOT),
        help="Output directory for per-model run_XX.npy files",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=1,
        help="Optional ensemble passes; Fig. 1 plot still reads run_01.npy",
    )
    args = parser.parse_args()

    apply_model_root(args.model_root)

    if args.cube_npz is not None:
        cube_path = args.cube_npz
    else:
        cube_dir = resolve_cube_dir(args.cube_dir)
        cube_path = cube_dir / "Maryland_test_cube.npz"
    if not cube_path.is_file():
        raise FileNotFoundError(cube_path)

    cube = load_hhdc_cube(cube_path, array_key="arr_0")
    pair = extract_chm_pair(cube)
    run_maryland_mosaic(
        pair["lr_tensor"],
        hr_full=pair["hr_tensor"],
        save_root=str(args.save_root),
        num_passes=max(1, args.num_passes),
    )


if __name__ == "__main__":
    main()
