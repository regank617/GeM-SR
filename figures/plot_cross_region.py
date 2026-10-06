"""Cross-regional CA / FL / MD CHM figure (plot-only)."""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REGION_COLORS = {"California": "#f4845f", "Maryland": "#57cc99", "Florida": "#48cae4"}
CMAP = "viridis"
ZOOM_CLR = "red"
SPINE_CLR = "black"
LABEL_FS = 15
BADGE_FS = 11
TITLE_FS = 16

HR_ZOOM_X, HR_ZOOM_Y, HR_ZOOM_S = 48, 48, 48


def _to2d(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x, dtype=float)
    while x.ndim > 2:
        x = x[0]
    return x


def _draw_panel(ax, img, is_lr, metrics_text=None):
    asp = 0.5 if is_lr else 1.0
    ax.imshow(img, cmap=CMAP, interpolation="none", aspect=asp, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])

    x, y, w, h = (HR_ZOOM_X, HR_ZOOM_Y, HR_ZOOM_S, HR_ZOOM_S)
    if is_lr:
        x = int(HR_ZOOM_X * 1 / 3)
        y = int(HR_ZOOM_Y * 2 / 3)
        w = int(HR_ZOOM_S * 1 / 3)
        h = int(HR_ZOOM_S * 2 / 3)
    ax.add_patch(
        mpatches.Rectangle(
            (x - 0.5, y - 0.5),
            w,
            h,
            linewidth=1.8,
            edgecolor=ZOOM_CLR,
            facecolor="none",
            zorder=10,
        )
    )

    axins = inset_axes(ax, width="35%", height="35%", loc="lower right", borderpad=0.1)
    axins.imshow(img[y : y + h, x : x + w], cmap=CMAP, interpolation="none", aspect=asp, origin="upper")
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(ZOOM_CLR)
        spine.set_linewidth(1.8)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_CLR)
        spine.set_linewidth(1.1)

    if metrics_text:
        props = dict(boxstyle="square,pad=0.3", facecolor="black", alpha=0.75, edgecolor="none")
        ax.text(
            0.04,
            0.04,
            metrics_text,
            transform=ax.transAxes,
            fontsize=BADGE_FS,
            color="white",
            bbox=props,
            fontweight="bold",
            family="monospace",
            zorder=20,
        )


def plot_three_regions_compact(regions, save_path="cross_regional_ablation.png", show: bool = True):
    fig = plt.figure(figsize=(9.2, 8.2))
    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1, 1, 1],
        width_ratios=[1, 1, 1, 0.08],
        hspace=0.03,
        wspace=0.03,
        left=0.12,
        right=0.92,
        top=0.92,
        bottom=0.05,
    )

    row_titles = [r"$Y_{\mathrm{LR}}$ (input)", r"$\hat{X}_{\mathrm{HR}}$ (Ours)", r"$X_{\mathrm{HR}}$ (GT)"]
    col_titles = ["California", "Maryland", "Florida"]

    for i, reg in enumerate(regions):
        accent = REGION_COLORS.get(reg["name"])
        badge = ""
        if "psnr" in reg:
            badge += f"P: {reg['psnr']:.1f}\n"
        if "ssim" in reg:
            badge += f"S: {reg['ssim']:.2f}"
        badge = badge.strip() or None
        row_data = [reg["lr"], reg["recon"], reg["hr"]]

        for j in range(3):
            ax = fig.add_subplot(gs[j, i])
            _draw_panel(ax, _to2d(row_data[j]), is_lr=(j == 0), metrics_text=badge if j == 1 else None)
            if i == 0:
                ax.text(
                    -0.15,
                    0.5,
                    row_titles[j],
                    transform=ax.transAxes,
                    fontsize=LABEL_FS,
                    fontweight="bold",
                    rotation=90,
                    ha="center",
                    va="center",
                )
            if j == 0:
                ax.set_title(col_titles[i], fontsize=TITLE_FS, fontweight="bold", pad=10, color=accent)

    cax = fig.add_subplot(gs[:, 3])
    norm = Normalize(vmin=0, vmax=40)
    cbar = ColorbarBase(cax, cmap=CMAP, norm=norm, orientation="vertical")
    cbar.set_label("Canopy Height (m)", fontsize=14, labelpad=12)
    cbar.ax.tick_params(labelsize=11)

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
