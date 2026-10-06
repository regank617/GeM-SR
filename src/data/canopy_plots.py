from __future__ import annotations
"""
Canopy height model helpers.

These utilities create canopy height models (CHM) from voxelized point cloud
data cubes while applying a simple adaptive filter to smooth noisy ground
estimates (DTM). Functions are written with clarity and deterministic output in
mind for use in notebooks and scripts.
"""


from typing import Tuple

import numpy as np
from numpy.typing import NDArray

Array2D = NDArray[np.floating]
Array3D = NDArray[np.floating]


def apply_kernel(dtm: Array2D, center: Tuple[int, int], kernel_size: int) -> Array2D:
    """
    Return a window of `dtm` centered on `center`, clipped to array bounds.

    Parameters
    ----------
    dtm:
        Two-dimensional digital terrain model array.
    center:
        (row, column) index around which to extract the window.
    kernel_size:
        Size of the square window (must be positive).
    """
    if kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer.")

    half = kernel_size // 2
    row_start = max(center[0] - half, 0)
    row_end = min(center[0] + half + 1, dtm.shape[0])
    col_start = max(center[1] - half, 0)
    col_end = min(center[1] + half + 1, dtm.shape[1])

    return dtm[row_start:row_end, col_start:col_end]


def adaptive_dtm_filter(
    dtm: Array2D, kernel_size: int = 7, percentile: float = 50.0
) -> Array2D:
    """
    Smooth the DTM by replacing outliers with the mean of local lower-percentile values.

    Each pixel greater than the specified percentile in its neighborhood is
    replaced by the mean of the values at or below that percentile. This
    mitigates spikes that otherwise lead to artifacts in the derived CHM.
    """
    filtered = dtm.copy()
    rows, cols = filtered.shape

    for row in range(rows):
        for col in range(cols):
            window = apply_kernel(filtered, (row, col), kernel_size).ravel()
            if window.size == 0:
                continue

            threshold = float(np.percentile(window, percentile))
            if filtered[row, col] > threshold:
                lower_values = window[window <= threshold]
                if lower_values.size:
                    filtered[row, col] = float(lower_values.mean())

    return filtered


def hillshade(
    array: Array2D, azimuth: float = 90.0, angle_altitude: float = 60.0
) -> Array2D:
    """
    Generate a simple hillshade from a 2D array for visualization.

    Parameters use degrees, consistent with most GIS tools.
    """
    azimuth = 360.0 - azimuth

    x_gradient, y_gradient = np.gradient(array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x_gradient * x_gradient + y_gradient * y_gradient))
    aspect = np.arctan2(-x_gradient, y_gradient)
    azimuth_rad = azimuth * np.pi / 180.0
    altitude_rad = angle_altitude * np.pi / 180.0

    shaded = (
        np.sin(altitude_rad) * np.sin(slope)
        + np.cos(altitude_rad) * np.cos(slope) * np.cos((azimuth_rad - np.pi / 2.0) - aspect)
    )

    return 255.0 * (shaded + 1.0) / 2.0


def calc_height_surface(cube: Array3D, percentile: float) -> NDArray[np.int64]:
    """
    Compute the height surface where the cumulative density exceeds `percentile`.
    """
    cumulative = np.cumsum(cube, axis=0)
    maxima = cumulative.max(axis=0)
    maxima[maxima == 0] = 1  # avoid division by zero
    normalized = cumulative / maxima

    surface = normalized >= percentile
    return np.argmax(surface, axis=0).astype(np.int64)


def create_chm(
    cube: Array3D, canopy_percentile: float = 0.98, dtm_percentile: float = 0.05, kernel_size: int = 7
) -> tuple[Array2D, Array2D, Array2D, NDArray[np.int64]]:
    """
    Build a canopy height model, ground model, hillshade, and DEM from a cube.

    Returns
    -------
    chm:
        Canopy height model (DEM minus DTM).
    dtm:
        Smoothed digital terrain model.
    dtm_hillshade:
        Hillshaded representation of the DTM for visualization.
    dem:
        Digital elevation model derived from the cube percentile.
    """
    dtm = calc_height_surface(cube, dtm_percentile)
    dtm = adaptive_dtm_filter(dtm, kernel_size=kernel_size)
    dem = calc_height_surface(cube, canopy_percentile)
    chm = dem - dtm

    return chm, dtm, hillshade(dtm), dem


# Backwards-compatible aliases for existing notebooks/scripts.
calcSurf = calc_height_surface
createCHM = create_chm

__all__ = [
    "adaptive_dtm_filter",
    "apply_kernel",
    "calc_height_surface",
    "calcSurf",
    "create_chm",
    "createCHM",
    "hillshade",
]
