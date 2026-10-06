"""Meta CHMv2 case-study figure (plot-only; ECC alignment done in run_meta_comparison)."""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.titlesize": 18,
        "axes.labelsize": 14,
    }
)

GT_RES = IN_RES_Y = 2.0
Z_RES = 0.5
MAP_INT = 150.0
Z_INT = 30.0


def _diagonal_split(ax, fg_data, bg_data, fg_cmap, bg_cmap, fg_vmin=None, fg_vmax=None, bg_vmax=None):
    height, width = fg_data.shape[:2]
    ax.imshow(bg_data, cmap=bg_cmap, vmin=0, vmax=bg_vmax, aspect="auto")
    tri_verts = [(-0.5, -0.5), (width - 0.5, -0.5), (-0.5, height - 0.5)]
    im_fg = ax.imshow(fg_data, cmap=fg_cmap, vmin=fg_vmin, vmax=fg_vmax, aspect="auto")
    im_fg.set_clip_path(MplPolygon(tri_verts, closed=True, transform=ax.transData))


def _map_ticks(ax, img, y_res, show_y):
    height = img.shape[0]
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if show_y:
        y_vals = np.arange(0, height * y_res, MAP_INT)
        ax.set_yticks(y_vals / y_res)
        ax.set_yticklabels([f"{int(v)}" for v in y_vals], fontsize=18)
    else:
        ax.set_yticks([])


def _profile_line_ticks(ax, n_pixels, max_height):
    x_vals = np.arange(0, n_pixels * IN_RES_Y, MAP_INT)
    ax.set_xticks(x_vals / IN_RES_Y)
    ax.set_xticklabels([f"{int(v)}" for v in x_vals], fontsize=18)
    z_vals = np.arange(0, max_height + 1, Z_INT)
    ax.set_yticks(z_vals)
    ax.set_yticklabels([f"{int(v * 0.5)}" for v in z_vals], fontsize=18)


def plot_meta_comparison(
    chm_gt,
    dtm_gt,
    chm_meta,
    chm_recon,
    gt_wave_slice,
    *,
    save_path="comparison_meta.png",
    show: bool = True,
):
    """
    Plot Meta CHMv2 vs GeM-SR vs GT.

    CHM arrays are in 0–128 bin units; display heights use ×0.5 for metres.
    """
    error_meta = np.abs(chm_meta - chm_gt) * 0.5
    error_recon = np.abs(chm_recon - chm_gt) * 0.5
    error_vmax = max(error_meta.max(), error_recon.max())

    psnr_recon = psnr_metric(chm_gt * 0.5, chm_recon * 0.5, data_range=(chm_gt * 0.5).max())
    psnr_meta = psnr_metric(chm_gt * 0.5, chm_meta * 0.5, data_range=(chm_gt * 0.5).max())
    mae_meta = error_meta.mean()
    mae_recon = error_recon.mean()

    global_max_height = 128
    profile_row = 3 * gt_wave_slice.shape[1] // 4
    gt_chm_line = chm_gt[profile_row, :] + dtm_gt[profile_row, :]
    max_plot_height = min(global_max_height, int(np.max(gt_chm_line) * 1.3) + 5)

    wave_slice = gt_wave_slice
    x = np.arange(wave_slice.shape[1])
    extent = [0, wave_slice.shape[1], 0, global_max_height]

    model_list = [
        ("CHMv2", chm_meta, "#2CA02C", psnr_meta),
        ("Our Method", chm_recon, "#1F77B4", psnr_recon),
    ]

    psnr_kw = dict(
        fontsize=24,
        color="white",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0),
    )

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, hspace=0, wspace=0, left=0.08, right=0.98, top=0.94, bottom=0.06)
    ax_chm = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_p = [fig.add_subplot(gs[i + 1, :]) for i in range(2)]

    chm_imgs = [chm_meta * 0.5, chm_recon * 0.5, chm_gt * 0.5]
    err_imgs = [error_meta, error_recon, None]
    titles = ["CHMv2", "Our Method", "Ground Truth"]
    chm_psnrs = [psnr_meta, psnr_recon, None]
    chm_maes = [mae_meta, mae_recon, None]

    chm_vmin = min(img.min() for img in chm_imgs)
    chm_vmax = max(img.max() for img in chm_imgs)

    for i, ax in enumerate(ax_chm):
        if err_imgs[i] is not None:
            _diagonal_split(
                ax,
                chm_imgs[i],
                err_imgs[i],
                "viridis",
                "turbo",
                fg_vmin=chm_vmin,
                fg_vmax=chm_vmax,
                bg_vmax=error_vmax,
            )
            ax.text(
                0.97,
                0.03,
                f"PSNR {chm_psnrs[i]:.1f} dB\nMAE {chm_maes[i]:.2f} m",
                transform=ax.transAxes,
                **psnr_kw,
            )
        else:
            ax.imshow(chm_imgs[i], cmap="viridis", vmin=chm_vmin, vmax=chm_vmax, aspect="auto")
        ax.set_title(titles[i], fontsize=18)
        _map_ticks(ax, chm_imgs[i], IN_RES_Y, show_y=(i == 0))

    ax_chm[0].set_ylabel("Distance (m)", fontsize=18)

    for i, (ax, (name, pred_full, color, _)) in enumerate(zip(ax_p, model_list)):
        pred_chm_line = pred_full[profile_row, :] + dtm_gt[profile_row, :]
        diff = pred_chm_line - gt_chm_line
        ax.imshow(
            wave_slice,
            cmap="gray_r",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=None,
            alpha=0.8,
            aspect="auto",
        )
        ax.plot(x, gt_chm_line, color="black", linewidth=1.5, label="Ground Truth")
        ax.plot(x, pred_chm_line, color=color, linewidth=2.0, label=name)
        ax.fill_between(
            x,
            gt_chm_line,
            pred_chm_line,
            where=(diff >= 0),
            color="#FF4C4C",
            alpha=0.45,
            label="Over-prediction",
        )
        ax.fill_between(
            x,
            gt_chm_line,
            pred_chm_line,
            where=(diff < 0),
            color="#0088FF",
            alpha=0.45,
            label="Under-prediction",
        )
        ax.set_ylim(0, max_plot_height)
        ax.set_xlim(0, wave_slice.shape[1])
        _profile_line_ticks(ax, wave_slice.shape[1], max_plot_height)
        ax.set_ylabel("Height (m)", fontsize=18)
        if i == len(ax_p) - 1:
            ax.set_xlabel("Distance (m)", fontsize=18)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.legend(loc="lower right", fontsize=14, framealpha=0.85, edgecolor="none", facecolor="white")
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
