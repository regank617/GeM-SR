import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Local module imports
from plotting import plot_appendix_grid

# =============================================================================
# Utility & Metric Functions
# =============================================================================
def calculate_psnr(mse, data_range=1.0):
    if mse == 0:
        return float('inf')
    return 10 * math.log10(data_range ** 2 / mse)

def compute_lpips(gt, pred, loss_fn, device='cuda'):
    gt = gt.to(device)
    pred = pred.to(device)

    if gt.shape[1] == 1:
        gt = gt.repeat(1, 3, 1, 1)
        pred = pred.repeat(1, 3, 1, 1)

    gt = gt * 2 - 1
    pred = pred * 2 - 1

    return loss_fn(gt, pred).squeeze()

def print_markdown_table(header, rows):
    print("\n### Quantitative Metrics Comparison (Full Batch N=64)")
    
    col_widths = [max(len(str(item)) for item in col) for col in zip(header, *rows)]
    
    def format_row(data):
        return "| " + " | ".join(str(val).ljust(width) for val, width in zip(data, col_widths)) + " |"
    
    print(format_row(header))
    print("| " + " | ".join("-" * width for width in col_widths) + " |")
    for row in rows:
        print(format_row(row))
    print("\n")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize CHM batch reconstructions.")
    parser.add_argument('--num_rows', type=int, default=16, 
                        help="Total number of rows (samples) to extract for visualization.")
    parser.add_argument('--rows_per_page', type=int, default=4, 
                        help="How many rows to plot per PDF to maintain readability.")
    parser.add_argument('--random_sample', action='store_true', 
                        help="Randomly select the images to plot from the batch.")
    parser.add_argument('--samples_path', type=str, default='samples/test_batch/', 
                        help="Directory containing the precomputed .npy batch files.")
    parser.add_argument('--save_prefix', type=str, default='batch_chm_visuals', 
                        help="Prefix for the output PDFs. Use 'None' to display only.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Starting Batch Evaluation (Device: {device.upper()}) ---")

    # 1. Load Data
    try:
        ylr_batch = torch.from_numpy(np.load(os.path.join(args.samples_path, 'ylr_batch.npy')))
        xhr_batch = torch.from_numpy(np.load(os.path.join(args.samples_path, 'xhr_batch.npy')))
        
        recons_dict = {
            "SwinIR": torch.from_numpy(np.load(os.path.join(args.samples_path, 'swinir_recons.npy'))),
            "Flow-Matching": torch.from_numpy(np.load(os.path.join(args.samples_path, 'flow_recons.npy'))),
            "VAR-SR": torch.from_numpy(np.load(os.path.join(args.samples_path, 'varsr_recons.npy'))),
            "GeM-SR (Ours)": torch.from_numpy(np.load(os.path.join(args.samples_path, 'gemsr_recons.npy')))
        }
    except FileNotFoundError as e:
        print(f"\nError loading batch data: {e}")
        print("Please ensure you have extracted the supplementary data into the correct folder.")
        return

    total_samples = len(xhr_batch)
    print(f"Data Loaded -> LR Batch: {ylr_batch.shape}, HR Batch: {xhr_batch.shape}")

    # 2. Subset Selection for Plotting
    total_to_plot = min(args.num_rows, total_samples)
    
    if args.random_sample:
        print(f"\nRandomly selecting {total_to_plot} samples for visualization...")
        indices = torch.randperm(total_samples)[:total_to_plot]
    else:
        print(f"\nSelecting the first {total_to_plot} samples for visualization...")
        indices = torch.arange(total_to_plot)

    # 3. Paginated Plotting
    skip_save = (args.save_prefix.lower() == 'none')
    
    for page_idx, start_row in enumerate(range(0, total_to_plot, args.rows_per_page)):
        end_row = min(start_row + args.rows_per_page, total_to_plot)
        chunk_indices = indices[start_row:end_row]
        current_rows = end_row - start_row
        
        # Subset tensors for this specific page
        ylr_chunk = ylr_batch[chunk_indices]
        xhr_chunk = xhr_batch[chunk_indices]
        recons_chunk = {k: v[chunk_indices] for k, v in recons_dict.items()}
        
        # Set save path
        if skip_save:
            chunk_save_path = None
            print(f"Displaying Page {page_idx + 1} (Rows {start_row + 1} to {end_row})...")
        else:
            chunk_save_path = f"{args.save_prefix}_page{page_idx + 1}.pdf"
            print(f"Saving Page {page_idx + 1} (Rows {start_row + 1} to {end_row}) -> {chunk_save_path}")

        # Plot the chunk
        plot_appendix_grid(
            ylr_chunk, 
            xhr_chunk, 
            recons_chunk, 
            num_rows=current_rows, 
            hr_zoom_coords=(18, 18, 10), 
            base_size=3.0,
            save_path=chunk_save_path
        )

    # 4. Initialize Metrics (Calculated on the full 64-item batch)
    print(f"\nCalculating metrics on the FULL batch of {total_samples} samples for reproducibility...")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=(0,1)).to(device)
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    ssim_metric.eval()
    loss_fn.eval()

    header = ["Model", "PSNR (higher better)", "SSIM (higher better)", "LPIPS (lower better)"]
    rows = []

    # 5. Compute Metrics
    for model_name, recon in recons_dict.items():
        recon = recon.float().to(device)
        xhr = xhr_batch.float().to(device)

        mse = F.mse_loss(recon, xhr).item()
        psnr_val = calculate_psnr(mse, data_range=1.0)

        with torch.no_grad():
            ssim_val = ssim_metric(recon, xhr).item()
            lpips_scores = compute_lpips(xhr, recon, loss_fn, device=device)
            lpips_val = lpips_scores.mean().item()

        rows.append([model_name, f"{psnr_val:.2f}", f"{ssim_val:.4f}", f"{lpips_val:.4f}"])

    # 6. Output Results
    print_markdown_table(header, rows)

if __name__ == "__main__":
    main()