import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ========================================================================
# Plotting multiple visualizations for benchmarking models
# Code used to get the visuals in Appendix 6.14
# ========================================================================
def plot_appendix_grid(lr, hr, recons_dict, num_rows=4, hr_zoom_coords=(16, 16, 16), base_size=3.0, save_path=None):
    """
    Plots a multi-row horizontal qualitative grid for the Appendix.
    
    Args:
        lr: Tensor [B, 1, 32, 16]
        hr: Tensor [B, 1, 48, 48]
        recons_dict: Dict mapping model names to Tensors [B, 1, 48, 48]
        num_rows: Number of rows (batch items) to plot.
        hr_zoom_coords: Tuple (x, y, size) defining the zoom box in the 48x48 space.
        base_size: Float. Multiplier for the inch-dimensions of each subplot.
        save_path: Optional string to save the figure as a PDF.
    """
    # Scale font sizes dynamically based on the base_size to ensure titles stay readable
    title_font = int(22 * (base_size / 3.0))
    
    plt.rcParams.update({'font.size': int(16 * (base_size / 3.0)), 'font.family': 'sans-serif'})
    
    # Safely bound the number of rows by the actual batch size
    B = lr.shape[0]
    num_rows = min(num_rows, B)
    
    # 1. Prepare column layout and titles
    model_names = list(recons_dict.keys())
    col_titles = ["Input $Y_{LR}$"] + model_names + ["Target $X_{HR}$"]
    n_cols = len(col_titles)
    
    # 2. Setup Figure and GridSpec
    # Dynamically scale width and height
    fig = plt.figure(figsize=(base_size * n_cols, base_size * num_rows)) 
    gs = fig.add_gridspec(num_rows, n_cols, wspace=0.02, hspace=0.02, 
                          left=0.02, right=0.98, top=0.95, bottom=0.02)
    
    # 3. Mathematically map HR zoom coordinates to LR space 
    hr_zx, hr_zy, hr_zs = hr_zoom_coords
    lr_zx = hr_zx // 3
    lr_zy = int(hr_zy * (2/3))
    lr_zw = hr_zs // 3
    lr_zh = int(hr_zs * (2/3))
    
    # Use the 99th percentile instead of absolute max to prevent one outlier pixel from dimming the whole plot
    global_vmax = float(torch.quantile(hr.float(), 0.99).cpu())
    
    for row in range(num_rows):
        # Extract the images for this specific row in the batch
        panels = [lr[row, 0].cpu().numpy()]
        for name in model_names:
            panels.append(recons_dict[name][row, 0].cpu().numpy())
        panels.append(hr[row, 0].cpu().numpy())
        
        for col, img in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            is_lr = (col == 0)
            
            # 1. Main Image
            if is_lr:
                ax.imshow(img, aspect=0.5, interpolation='none', cmap='viridis', vmin=0.0, vmax=global_vmax)
                rect = (lr_zx, lr_zy, lr_zw, lr_zh)
                zoom_patch = img[lr_zy : lr_zy + lr_zh, lr_zx : lr_zx + lr_zw]
                zoom_aspect = 0.5
            else:
                ax.imshow(img, aspect=1.0, interpolation='none', cmap='viridis', vmin=0.0, vmax=global_vmax)
                rect = (hr_zx, hr_zy, hr_zs, hr_zs)
                zoom_patch = img[hr_zy : hr_zy + hr_zs, hr_zx : hr_zx + hr_zs]
                zoom_aspect = 1.0
                
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.5)
                
            if row == 0:
                ax.set_title(col_titles[col], fontsize=title_font, fontweight='bold', pad=12)
                    
            # 2. Inset Zoom 
            axins = inset_axes(ax, width="35%", height="35%", loc='lower right', borderpad=0.0)
            axins.imshow(zoom_patch, interpolation='none', aspect=zoom_aspect, cmap='viridis', vmin=0.0, vmax=global_vmax)
            
            axins.set_xticks([])
            axins.set_yticks([])
            for spine in axins.spines.values():
                spine.set_visible(True)       
                spine.set_edgecolor('red')
                spine.set_linewidth(2.5)
                
            # Draw the Red Box
            ax.add_patch(plt.Rectangle((rect[0]-0.5, rect[1]-0.5), rect[2], rect[3], 
                                       edgecolor='red', facecolor='none', 
                                       linewidth=2.5, clip_on=False))

    if save_path:
        # DPI=300 is standard for print, DPI=150 is often better for massive grids to keep PDF size down
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=150)
    
    # Display the plot
    plt.show()
    # Close the figure to free up memory if running in a loop
    plt.close(fig)
# ==========================================
# Example Usage for 4 Rows:
# ==========================================
# plot_appendix_grid(
#     lr=lr, 
#     hr=hr, 
#     recons_dict=recons_dict, 
#     num_rows=4,                  # <-- Set this dynamically to whatever looks best (e.g., 4 or 5)
#     hr_zoom_coords=(16, 16, 16), 
#     save_path="appendix_grid_4rows.pdf"
# )


# ========================================================================
# Plotting California, Maryland, Florida macro-region CHM visuals
# Implementation lives in figures.plot_cross_region (re-exported here).
# ========================================================================
from figures.plot_cross_region import plot_three_regions_compact  # noqa: F401