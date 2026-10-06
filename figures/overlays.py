"""Locked Maryland test-split overlay numbers for Figure 1 (Tables II–III).

These are **not** scores computed on the SERC mosaic. Do not recompute FID on mosaic tiles.
"""

from __future__ import annotations

SR3_LINE = dict(
    fps=[54.09, 27.01, 13.49],
    fid=[27.14, 14.56, 9.48],
    rmse=[3.65, 3.72, 3.77],
    top1=[3.09, 3.18, 3.16],
)

MODEL_METRICS = {
    "GeM-SR (RS)": dict(fps=1266.56, fid=18.81, rmse=3.62, top1=2.76),
    "GeM-SR (P)": dict(fps=1268.84, fid=14.37, rmse=3.78, top1=3.27),
    "VAR-SR": dict(fps=196.06, fid=19.43, rmse=3.99, top1=3.59),
    "Flow-Matching": dict(fps=694.62, fid=77.52, rmse=3.36, top1=3.19),
    "SwinIR": dict(fps=730.37, fid=102.17, rmse=3.30, top1=3.01),
}

# Scatter-only (Table II SR3 2000-step DDPM); no mosaic panel.
DDPM_SCATTER = dict(fps=1.37, fid=7.63, rmse=3.79, top1=3.20)

MODEL_MARKERS = {
    "GeM-SR (RS)": "*",
    "GeM-SR (P)": "*",
    "VAR-SR": "s",
    "Flow-Matching": "D",
    "SwinIR": "o",
    "DDPM": "x",
}

MODEL_COLORS = {
    "GeM-SR (RS)": "#d62728",
    "GeM-SR (Perceptual)": "#990000",
    "VAR-SR": "#9467bd",
    "Flow-Matching": "#ff7f0e",
    "SwinIR": "#2ca02c",
    "DDPM": "#888888",
}

MODEL_SIZES = {
    "GeM-SR (RS)": 500,
    "GeM-SR (Perceptual)": 450,
    "VAR-SR": 200,
    "Flow-Matching": 200,
    "SwinIR": 200,
    "DDPM": 180,
}

# (model, scatter y-key) → annotation offset in points (dx, dy)
# GeM-SR stars share ~1267 FPS: split RS left / Perceptual right on RMSE; above/below on FID.
ANNOTATE_OFFSETS = {
    ("GeM-SR (RS)", "fid"): (8, 6),
    ("GeM-SR (Perceptual)", "fid"): (8, -7),
    ("GeM-SR (RS)", "rmse"): (-8, 5),
    ("GeM-SR (Perceptual)", "rmse"): (8, -5),
    ("VAR-SR", "fid"): (7, 3),
    ("VAR-SR", "rmse"): (7, -9),
    ("Flow-Matching", "fid"): (-7, 2),
    ("Flow-Matching", "rmse"): (-7, -2),
    ("SwinIR", "fid"): (7, 5),
    ("SwinIR", "rmse"): (7, 3),
}

FPS_XTICKS = [0, 400, 800, 1200]
