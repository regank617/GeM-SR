"""Figure 1 teaser plot (mosaic panels + locked Maryland test overlays)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Union

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from figures.overlays import (
    ANNOTATE_OFFSETS,
    DDPM_SCATTER,
    FPS_XTICKS,
    MODEL_COLORS,
    MODEL_MARKERS,
    MODEL_METRICS,
    MODEL_SIZES,
    SR3_LINE,
)

PAPER_BG = "#ffffff"
PANEL_BG = "#ffffff"
BORDER = "#333333"
TEXT_PRI = "#000000"
TEXT_SEC = "#222222"
TEXT_HINT = "#555555"
C_BLUE = "#1f77b4"
C_PURPLE = "#9467bd"
C_RED = "#d62728"
C_DARKRED = "#990000"
C_GREEN = "#2ca02c"
C_AMBER = "#ff7f0e"
C_ZOOM = "#e31a1c"

MOSAIC_MODEL_DIRS = {
    "swinir": "swinir",
    "flow": "flow",
    "sr3_ddim_200": "sr3_ddim_200",
    "varsr": "varsr",
    "gemsr_rs": "gemsr_rs",
    "gemsr_perc": "gemsr_perc",
}


def _to_numpy(x, idx=0):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x, dtype=float)
    if x.ndim == 4:
        x = x[idx]
    if x.ndim == 3:
        x = x[0]
    return x


def _setup_rcparams():
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER_BG,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT_SEC,
            "xtick.color": TEXT_HINT,
            "ytick.color": TEXT_HINT,
            "text.color": TEXT_PRI,
            "grid.color": "#e0e0e0",
            "grid.alpha": 0.8,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": TEXT_PRI,
            "legend.facecolor": PAPER_BG,
            "legend.edgecolor": BORDER,
            "legend.labelcolor": TEXT_PRI,
            "savefig.facecolor": PAPER_BG,
            "savefig.edgecolor": PAPER_BG,
        }
    )


def _add_zoom_inset(
    ax,
    img,
    rect_xy_wh,
    inset_loc="lower right",
    inset_w="42%",
    inset_h="42%",
    aspect="auto",
):
    x0, y0, rw, rh = rect_xy_wh
    patch_img = img[y0 : y0 + rh, x0 : x0 + rw]
    ax.add_patch(
        mpatches.Rectangle(
            (x0 - 0.5, y0 - 0.5),
            rw,
            rh,
            linewidth=2.0,
            edgecolor=C_ZOOM,
            facecolor="none",
            zorder=5,
            clip_on=False,
        )
    )
    axins = inset_axes(ax, width=inset_w, height=inset_h, loc=inset_loc, borderpad=0)
    kw = dict(interpolation="none", cmap="viridis", origin="upper", aspect=aspect)
    axins.imshow(patch_img, **kw)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(C_ZOOM)
        spine.set_linewidth(2.0)


def _metrics_badge(ax, text, color=C_BLUE):
    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        fontsize=14,
        color=color,
        verticalalignment="bottom",
        fontfamily="monospace",
        fontweight="bold",
        bbox=dict(
            boxstyle="square,pad=0.5",
            facecolor="#ffffff",
            alpha=0.9,
            edgecolor=BORDER,
            linewidth=1.5,
        ),
        zorder=10,
    )


def _overlaid_label(ax, letter, title):
    text = f"({letter}) {title}"
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        fontsize=16,
        color=TEXT_PRI,
        verticalalignment="top",
        fontweight="bold",
        bbox=dict(
            boxstyle="square,pad=0.3",
            facecolor="#ffffff",
            alpha=0.85,
            edgecolor="none",
        ),
        zorder=20,
    )


def plot_teaser(
    lr,
    swinir_recon,
    fast_diffusion_recon,
    sr3_200_recon,
    varsr_recon,
    gemsr_rs_recon,
    gemsr_perc_recon,
    hr,
    *,
    y_metric="rmse",
    save_path=None,
    dpi=300,
    hr_zoom=(24, 12, 16, 16),
    lr_zoom=None,
    show: bool = True,
):
    """Render Figure 1 mosaic + scatter panels. Overlay badges use ``figures.overlays`` only."""
    _setup_rcparams()
    lr_img = _to_numpy(lr)
    hr_img = _to_numpy(hr)
    swinir_img = _to_numpy(swinir_recon)
    fast_img = _to_numpy(fast_diffusion_recon)
    sr3_200_img = _to_numpy(sr3_200_recon)
    varsr_img = _to_numpy(varsr_recon)
    gemsr_rs_img = _to_numpy(gemsr_rs_recon)
    gemsr_perc_img = _to_numpy(gemsr_perc_recon)

    if lr_zoom is None:
        hx, hy, hw, hh = hr_zoom
        hr_h, hr_w = hr_img.shape
        lr_h, lr_w = lr_img.shape
        scale_x = lr_w / hr_w
        scale_y = lr_h / hr_h
        lr_zoom = (
            int(hx * scale_x),
            int(hy * scale_y),
            max(2, int(hw * scale_x)),
            max(2, int(hh * scale_y)),
        )

    fig = plt.figure(figsize=(20, 13.5))
    gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[3.3, 1.0],
        hspace=0.20,
        left=0.04,
        right=0.96,
        top=0.95,
        bottom=0.06,
    )
    gs_images = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=gs[0], height_ratios=[1, 1], wspace=0.015, hspace=0.015
    )
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.28)

    ax_lr = fig.add_subplot(gs_images[0, 0])
    ax_swinir = fig.add_subplot(gs_images[0, 1])
    ax_fast = fig.add_subplot(gs_images[0, 2])
    ax_sr3_200 = fig.add_subplot(gs_images[0, 3])
    ax_varsr = fig.add_subplot(gs_images[1, 0])
    ax_gemsr_rs = fig.add_subplot(gs_images[1, 1])
    ax_gemsr_pc = fig.add_subplot(gs_images[1, 2])
    ax_gt = fig.add_subplot(gs_images[1, 3])
    ax_fid = fig.add_subplot(gs_bot[0, 0])
    ax_right = fig.add_subplot(gs_bot[0, 1])

    panels = [
        (ax_lr, lr_img, "a", "LR input", None, True),
        (ax_swinir, swinir_img, "b", "SwinIR", "730 FPS\nFID: 102.2", False),
        (ax_fast, fast_img, "c", "Flow", "695 FPS\nFID: 77.5", False),
        (ax_sr3_200, sr3_200_img, "d", "SR3 (200 steps)", "13.5 FPS\nFID: 9.48", False),
        (ax_varsr, varsr_img, "e", "VAR-SR (VQ-VAE)", "196 FPS\nFID: 19.4", False),
        (ax_gemsr_rs, gemsr_rs_img, "f", "GeM-SR (RS)", "★ 1267 FPS\nFID: 18.81", False),
        (ax_gemsr_pc, gemsr_perc_img, "g", "GeM-SR (Perceptual)", "★ 1269 FPS\nFID: 14.37", False),
        (ax_gt, hr_img, "h", "HR ground truth", None, False),
    ]

    for ax, img, letter, title, badge, is_lr in panels:
        ax.imshow(img, cmap="viridis", interpolation="none", origin="upper", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BORDER)
            spine.set_linewidth(1.5)
        _overlaid_label(ax, letter, title)
        zoom = lr_zoom if is_lr else hr_zoom
        _add_zoom_inset(ax, img, zoom, aspect="auto")
        if badge:
            color = C_RED if "★" in badge else TEXT_PRI
            _metrics_badge(ax, badge, color=color)

    y_right_label = {
        "rmse": "RMSE ↓ (m)",
        "top1": "Top-1% RMSE ↓ (tall canopy)",
        "lpips": "LPIPS ↓ (Perceptual)",
    }[y_metric]

    right_title = {
        "rmse": "Physical error vs. Speed",
        "top1": "Top Canopy Adherence vs. Speed",
        "lpips": "Perceptual error vs. Speed",
    }[y_metric]

    def _draw_scatter(ax, y_key, y_axis_label, panel_letter, title_str):
        ax.plot(
            SR3_LINE["fps"],
            SR3_LINE[y_key],
            color=C_BLUE,
            linewidth=2.5,
            linestyle="-",
            marker="o",
            markersize=8,
            markeredgecolor=PAPER_BG,
            markeredgewidth=1.0,
            label="SR3 DDIM 50→200",
            zorder=2,
        )
        ax.axvspan(
            MODEL_METRICS["GeM-SR (RS)"]["fps"] * 0.85,
            1400,
            alpha=0.08,
            color=C_RED,
            zorder=0,
        )
        ax.set_xlim(-40, 1400)

        if DDPM_SCATTER is not None and y_key in DDPM_SCATTER:
            ax.scatter(
                DDPM_SCATTER["fps"],
                DDPM_SCATTER[y_key],
                marker=MODEL_MARKERS.get("DDPM", "x"),
                color=MODEL_COLORS.get("DDPM", "#888888"),
                s=MODEL_SIZES.get("DDPM", 180),
                edgecolors=BORDER,
                linewidths=1.2,
                label="DDPM",
                zorder=3,
            )

        for name, metrics in MODEL_METRICS.items():
            ax.scatter(
                metrics["fps"],
                metrics[y_key],
                marker=MODEL_MARKERS.get(name, "o"),
                color=MODEL_COLORS.get(name, C_BLUE),
                s=MODEL_SIZES.get(name, 200),
                edgecolors=BORDER,
                linewidths=1.2,
                label=name,
                zorder=4,
            )
            offsets_key = (name, y_key)
            dx, dy = ANNOTATE_OFFSETS.get(offsets_key, (6, 6))
            h_align = "right" if dx < 0 else "left"
            v_align = "top" if dy < 0 else "bottom"
            if name.startswith("GeM-SR") and y_key == "rmse":
                # Stars nearly coincident in x; split labels horizontally.
                if name.endswith("(RS)"):
                    dx, dy, h_align, v_align = -6, 0, "right", "center"
                else:
                    dx, dy, h_align, v_align = 6, 0, "left", "center"
            annotate_label = name if name.startswith("GeM-SR") else name.split(" (")[0]
            if annotate_label == "Flow-Matching":
                annotate_label = "Flow"
            label_fs = 11 if name.startswith("GeM-SR") else 13
            ax.annotate(
                annotate_label,
                xy=(metrics["fps"], metrics[y_key]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=label_fs,
                color=BORDER,
                fontweight="bold",
                fontfamily="sans-serif",
                ha=h_align,
                va=v_align,
            )

        ax.set_xlabel("Inference throughput (FPS) ↑", fontsize=18, color=TEXT_PRI, labelpad=8)
        ax.set_ylabel(y_axis_label, fontsize=18, color=TEXT_PRI, labelpad=8)
        ax.grid(True, which="both", zorder=1)
        ax.tick_params(axis="both", labelsize=13)
        ax.xaxis.set_major_locator(FixedLocator(FPS_XTICKS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
        title_pad = 22 if panel_letter == "j" else 12
        ax.set_title(
            f"$\\mathbf{{({panel_letter})}}$ {title_str}",
            fontsize=18,
            color=TEXT_PRI,
            pad=title_pad,
            loc="left",
        )

    _draw_scatter(ax_fid, "fid", "FID ↓", "i", "Generative Realism vs. Speed")
    _draw_scatter(ax_right, y_metric, y_right_label, "j", right_title)

    handles, labels = ax_fid.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_fid.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        bbox_to_anchor=(0.97, 1.18),
        fontsize=11,
        framealpha=1.0,
        markerscale=0.65,
        handlelength=1.3,
        borderpad=0.4,
        borderaxespad=0.0,
    )

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def load_mosaic_recon(eval_root: Union[str, Path], subdir: str, run_name: str = "run_01") -> np.ndarray:
    path = Path(eval_root) / subdir / f"{run_name}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.squeeze(np.load(path))


def missing_mosaic_files(eval_root: Union[str, Path]) -> list[str]:
    eval_root = Path(eval_root)
    missing = []
    for subdir in MOSAIC_MODEL_DIRS.values():
        expected = eval_root / subdir / "run_01.npy"
        if not expected.is_file():
            missing.append(str(expected))
    return missing


def plot_teaser_from_disk(
    lr: np.ndarray,
    hr: np.ndarray,
    eval_root: Union[str, Path] = "evaluation/maryland_evals",
    *,
    save_path: Union[str, Path] = "evaluation/teaser.pdf",
    copy_to: Optional[Union[str, Path]] = None,
    hr_zoom=(48, 96, 48, 48),
    y_metric: str = "rmse",
    show: bool = False,
) -> None:
    """Load saved mosaic npy arrays and write teaser PDF."""
    eval_root = Path(eval_root)
    missing = missing_mosaic_files(eval_root)
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "STOP: mosaic recon .npy not found.\n"
            f"Expected in evaluation/maryland_evals/ (shipped with the repo); "
            f"regenerate with scripts/run_maryland_mosaic.py on GPU:\n{lines}"
        )

    arrays = {key: load_mosaic_recon(eval_root, subdir) for key, subdir in MOSAIC_MODEL_DIRS.items()}
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # LR/HR and mosaic recons are all in [0, 1] (CHM bin height / 128); see figures.chm_pair.

    plot_teaser(
        lr=lr,
        hr=hr,
        swinir_recon=arrays["swinir"],
        fast_diffusion_recon=arrays["flow"],
        sr3_200_recon=arrays["sr3_ddim_200"],
        varsr_recon=arrays["varsr"],
        gemsr_rs_recon=arrays["gemsr_rs"],
        gemsr_perc_recon=arrays["gemsr_perc"],
        y_metric=y_metric,
        save_path=str(save_path),
        hr_zoom=hr_zoom,
        show=show,
    )

    if copy_to is not None:
        copy_to = Path(copy_to)
        if copy_to.parent.exists() or copy_to.parent == Path("."):
            copy_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(save_path, copy_to)
            print(f"Copied: {copy_to}")
