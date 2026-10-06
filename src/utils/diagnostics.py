import torch
import wandb

@torch.no_grad()
def calculate_batch_stats(outputs, null_idx, vocab_size, return_entropy=False):
    """
    Comprehensive LidarSSM Diagnostics.
    Bridges Information Theory and Deep Learning metrics.
    """
    pred_indices = outputs['pred_indices'] # [B, S, D, H, W]
    gt_indices = outputs['gt_indices']     # [B, S, D, H, W]
    
    stats = {}
    scales = ['root', 'scale_0', 'scale_1', 'scale_2']

    for scale in scales:
        # 1. Prepare Tensors
        pred = pred_indices[scale].flatten(1) 
        gt = gt_indices[scale].flatten(1)
        
        is_active_pred = (pred != null_idx)
        is_active_gt = (gt != null_idx)

        # Intersection/Union now strictly compare the same physical points
        intersection_spatial = (is_active_pred & is_active_gt).sum().float()
        union_spatial = (is_active_pred | is_active_gt).sum().float()
        shunt_iou = intersection_spatial / (union_spatial + 1e-6)

        # 4. Residual Token IoU (The Content)
        # Measures if the model picked the EXACT right token for active points.
        intersection_token = ((pred == gt) & is_active_gt).sum().float()
        # Union is the set of all non-null tokens predicted or present in GT
        token_iou = intersection_token / (union_spatial + 1e-6)

        # 5. Density & Sparsity
        pred_density = is_active_pred.float().mean()
        gt_density = is_active_gt.float().mean()
        density_ratio = pred_density / (gt_density + 1e-6)
        
        # Residual Sparsity: How much of the "Active" tree is actually used
        # If this is 1.0, the model is fully dense; near 0.0 is highly efficient/sparse.
        sparsity = 1.0 - pred_density

        # [NEW METRIC 1] Missed Structure Rate (False Negative Rate)
        # Fraction of GT Active pixels that were predicted as Empty (Zero)
        # Ideally should be 0.0. If high, shunting is killing features.
        missed_structure = ((gt != null_idx) & (pred == null_idx)).sum().float()
        missed_structure_rate = missed_structure / (is_active_gt.sum().float() + 1e-6)

        # [NEW METRIC 2] Unique Codes Used (Codebook Utilization)
        # If low, indicates mode collapse (predicting same texture everywhere)
        unique_codes = torch.unique(pred).numel()
        gt_unique_codes = torch.unique(gt).numel()

        scale_stats = {
            "shunt_iou": shunt_iou.item(),
            "token_iou": token_iou.item(),
            "density_ratio": density_ratio.item(),
            "pred_density": pred_density.item(),
            "gt_density": gt_density.item(),
            "sparsity": sparsity.item(),
            "missed_structure_rate": missed_structure_rate.item(),
            "unique_codes": unique_codes,
            "gt_unique_codes": gt_unique_codes,
        }

        # 6. Information Theory Metrics
        if return_entropy:
            # Helper to calculate Shannon Entropy H(X) in bits
            def get_entropy(codes):
                content = codes[codes != null_idx]
                if content.numel() == 0: return 0.0
                
                # Ensure indices are within [0, vocab_size-1] before histogram
                content = torch.clamp(content, 0, vocab_size - 1)
                
                counts = torch.histc(content.float(), bins=vocab_size, min=0, max=vocab_size-1)
                p = counts / (counts.sum() + 1e-6)
                p = p[p > 0] 
                return -torch.sum(p * torch.log2(p)).item()

            h_pred = get_entropy(pred)
            h_gt = get_entropy(gt)
            
            # Entropy Gap: Positive means model is "noisy/uncertain", Negative means "collapsed"
            scale_stats["entropy_gap"] = h_pred - h_gt
            
            # 7. Mutual Information I(Mask; Tokens) - Shunt Efficiency
            # How much does knowing the Spatial Mask (Shunt) reduce token uncertainty?
            # I(M; T) = H(T) - H(T | M)
            # In a shunted system, H(T | M=0) is 0 (it's always null_idx).
            # Thus MI is effectively the "Content Entropy" weighted by the mask's presence.
            scale_stats["shunt_efficiency_bits"] = h_pred * pred_density.detach().cpu().item()
            #sscale_stats["shunt_efficiency_bits"] = h_pred * pred_density.item()

        stats[scale] = scale_stats
        
    return stats

def format_metrics(output, batch_stats, current_penalties):
    """Flattens nested dictionaries into a flat logging format."""
    logs = {}
    # Base model losses (root, scale_0, etc.)
    if 'metrics' in output:
        for k, v in output['metrics'].items():
            logs[f"train/loss_{k}"] = v.item() if hasattr(v, 'item') else v
    
    # PID VBR stats
    for scale, stats in batch_stats.items():
        for stat_name, val in stats.items():
            logs[f"vbr_{stat_name}/{scale}"] = val

    # Dynamic penalties from the controller
    for scale, penalty in current_penalties.items():
        logs[f"dynamic/penalty_{scale}"] = penalty

    logs["train/total_loss"] = output['loss'].item()
    logs["train/is_collapsed"] = float(output.get('is_collapsed', False))
    return logs

def log_visuals(accelerator, viz_dict, step, image_colormap=None):
    """Router for image logging between WandB and TensorBoard.
    Float tensors in [0, 1] are converted to uint8 (0-255) so WandB does not re-normalize
    each image independently.

    ``image_colormap=None``: single-channel images are sent as (H, W, 3) with identical
    channels so WandB shows neutral grayscale.

    ``image_colormap="viridis"``: single-channel images are mapped with matplotlib viridis
    to RGB. True RGB inputs (3 channels with distinct content) should be passed with
    ``image_colormap="rgb"`` so all channels are kept.
    """
    if not accelerator.is_main_process:
        return
    import numpy as np

    def _to_wandb_hw3(img_tensor):
        """img_tensor: [C, H, W] float CPU."""
        arr = img_tensor.numpy()
        arr = np.clip(arr, 0.0, 1.0)
        if arr.ndim == 3 and arr.shape[0] == 3 and image_colormap == "rgb":
            arr_hw3 = np.transpose(arr, (1, 2, 0))
            return (arr_hw3 * 255).astype(np.uint8)
        if arr.ndim == 3:
            arr = arr[0]  # [C, H, W] -> [H, W] single-channel
        arr = np.clip(arr, 0.0, 1.0)
        if arr.ndim == 2 and image_colormap == "viridis":
            import matplotlib.cm as mpl_cm

            rgb = (mpl_cm.viridis(arr)[..., :3] * 255).astype(np.uint8)
            return rgb
        arr_uint8 = (arr * 255).astype(np.uint8)
        if arr_uint8.ndim == 2:
            arr_uint8 = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)
        return arr_uint8

    def _to_tb_chw01(img_tensor):
        """Tensor [C,H,W] on CPU float [0,1] for add_image."""
        if img_tensor.shape[0] == 3 and image_colormap == "rgb":
            return img_tensor
        if img_tensor.shape[0] == 1 and image_colormap == "viridis":
            import matplotlib.cm as mpl_cm

            arr = np.clip(img_tensor[0].numpy(), 0.0, 1.0)
            rgb = torch.from_numpy(mpl_cm.viridis(arr)[..., :3]).permute(2, 0, 1).float()
            return rgb
        if img_tensor.shape[0] == 1:
            return img_tensor.repeat(3, 1, 1)
        return img_tensor

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            log_dict = {}
            for k, v in viz_dict.items():
                img_tensor = v[0] if v.ndim == 4 else v
                img_tensor = img_tensor.detach().float().cpu()
                log_dict[k] = [wandb.Image(_to_wandb_hw3(img_tensor))]
            tracker.log(log_dict, step=step)
        elif tracker.name == "tensorboard":
            for k, v in viz_dict.items():
                img_tensor = v[0] if v.ndim == 4 else v
                img_tensor = img_tensor.detach().float().cpu()
                tracker.tracker.add_image(k, _to_tb_chw01(img_tensor), step)

import matplotlib.pyplot as plt
import numpy as np

def plot_cfg_sweep_grid(low_res, high_res, ae_recon, generations, n=4, cmap="viridis"):
    """
    Columns: [LR | GT | AE | CFG_1 | CFG_2 | CFG_3]
    """
    def _squeeze(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        return x[:, 0] if x.ndim == 4 else x

    lr, gt, ae = _squeeze(low_res), _squeeze(high_res), _squeeze(ae_recon)
    gen_keys = list(generations.keys())
    gen_data = [_squeeze(generations[k]) for k in gen_keys]

    num_cols = 3 + len(gen_keys)
    titles = ["Low-Res", "Ground Truth", "Tokenizer Max"] + gen_keys

    fig, axes = plt.subplots(nrows=n, ncols=num_cols, figsize=(num_cols * 3, n * 3), squeeze=False)

    for row in range(n):
        # Gather all images for this row
        row_images = [lr[row], gt[row], ae[row]] + [g[row] for g in gen_data]
        
        # Determine consistent color scale based on GT
        v_min, v_max = np.percentile(gt[row], [1, 99])

        for col, img in enumerate(row_images):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmap, vmin=v_min, vmax=v_max, aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=12, fontweight='bold')
            
            # Add colorbar only to the last column to save space
            if col == num_cols - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig

def surgery_refiner_weights(model):
    """
    Expands the Refiner's first layer from 1 channel to 2 channels.
    Initializes the new 'Satellite' channel to ZERO to preserve behavior.
    """
    refiner = model.tokenizer.refiner
    old_weight = refiner.in_conv.weight.data # Shape: [dim, 1, 3, 3]
    
    # Check if surgery is needed
    if old_weight.shape[1] == 2:
        return # Already done

    print("Scaling up Refiner channels from 1 -> 2...")
    new_weight = torch.zeros((old_weight.shape[0], 2, 3, 3), device=old_weight.device)
    
    # Copy old weights to Channel 0 (Lidar)
    new_weight[:, 0:1, :, :] = old_weight
    
    # Channel 1 (Satellite) remains 0.0 (Zero-Init)
    
    # Apply
    refiner.in_conv.weight.data = new_weight
    print("✅ Surgery Complete.")