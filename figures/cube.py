"""Load HHDC 3-D cubes with a consistent (128, H, W) channel-first layout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

VERTICAL_BINS = 128


def _resolve_array_key(data: np.lib.npyio.NpzFile, array_key: Optional[str]) -> str:
    if array_key is not None:
        if array_key not in data:
            raise KeyError(f"array key {array_key!r} not in {list(data.files)}")
        return array_key
    if "cube" in data:
        return "cube"
    if "arr_0" in data:
        return "arr_0"
    raise KeyError(f"No default cube key in {list(data.files)}; pass array_key=")


def ensure_chw(cube: np.ndarray) -> np.ndarray:
    """Return cube as (128, H, W). Maryland npz often stores (H, W, 128)."""
    cube = np.asarray(cube, dtype=np.float32)
    if cube.shape[0] == VERTICAL_BINS:
        return cube
    if cube.shape[-1] == VERTICAL_BINS:
        return cube.transpose(2, 0, 1)
    raise ValueError(
        f"Expected {VERTICAL_BINS} vertical bins; got shape {cube.shape}. "
        "Provide a cube stored as (128, H, W) or (H, W, 128) rather than guessing the layout."
    )


def load_hhdc_cube(
    npz_path: Union[str, Path],
    *,
    array_key: Optional[str] = None,
    max_norm: bool = True,
) -> np.ndarray:
    """
    Load an HHDC cube npz as float32 (128, H, W).

    When ``max_norm`` is True, divide by per-column max along the vertical axis.
    This is visualization-path occupancy scaling, not ``jobs/extract_chm`` behavior.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as data:
        key = _resolve_array_key(data, array_key)
        cube = ensure_chw(data[key])
    if max_norm:
        cube = cube / (cube.max(axis=0, keepdims=True) + 1e-6)
    return cube


def load_coords(npz_path: Union[str, Path], *, array_key: str = "arr_0") -> np.ndarray:
    """Load a coordinate grid npz (e.g. Maryland_test_cube_coords.npz)."""
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as data:
        if array_key not in data:
            raise KeyError(f"{array_key!r} not in {list(data.files)}")
        return np.asarray(data[array_key])


def resolve_cube_dir(
    explicit: Union[str, Path, None] = None,
    *,
    env_var: str = "GEMSR_CUBE_DIR",
) -> Path:
    """Return cube directory from *explicit*, env, or raise with CLI hint."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    raw = os.environ.get(env_var) or os.environ.get("MARYLAND_CUBE_DIR")
    if not raw:
        raise FileNotFoundError(
            f"Pass --cube-dir / set CUBE_DIR in the notebook, or set {env_var} "
            f"(or MARYLAND_CUBE_DIR) to the folder with Maryland_test_cube.npz"
        )
    path = Path(raw)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path
