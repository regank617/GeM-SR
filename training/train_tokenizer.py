import os
import sys

# Add project root to sys.path so we can import src.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import importlib
import sys, os, json, math, time, pprint, glob, shutil  # math used for isfinite in logging
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
import omegaconf
print(f"Imported OmegaConf and Standard Libraries")

# Accelerate & PyTorch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.utils import make_grid
import numpy as np
print(f"Imported Torch and Accelerate")

import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import Normalize, TwoSlopeNorm

import wandb
from tqdm import tqdm
# Utility 
from src.utils.meter import AverageMeter
from src.utils.logger import setup_logger
from src.utils.lr_schedulers import get_scheduler
print(f"Imported Custom Utilities")
import lpips
# TO THIS:
from src.data.lidar_dataloading import build_dataloaders

from src.modeling.tokenizer.tokenizer_fsq import LidarTokenizerV5
from src.modeling.tokenizer.tokenizer_v5_layer_config import TokenizerV5LayerConfig

logger = get_logger(__name__)

# --- PYTORCH 2.6+ FIX ---
try:
    from omegaconf.dictconfig import DictConfig
    from omegaconf.listconfig import ListConfig
    from omegaconf.base import ContainerMetadata, Node
    
    torch.serialization.add_safe_globals([
        DictConfig, 
        ListConfig, 
        ContainerMetadata, 
        Node,
        set, 
    ])
except ImportError:
    pass 
except AttributeError:
    pass 

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

_VQ_TOKENIZER_CLASSES = frozenset(
    {
        "LidarTokenizerVQ",
        "LidarTokenizerNoLL",
        "LidarTokenizerIndepVQ",
        "LidarTokenizerVanillaVQ",
    }
)


def _tokenizer_class_from_target(tokenizer_target: str) -> str:
    t = (tokenizer_target or "").strip()
    return t.rsplit(".", 1)[-1] if t else ""


def _tokenizer_target_is_vq(tokenizer_target: str) -> bool:
    return _tokenizer_class_from_target(tokenizer_target) in _VQ_TOKENIZER_CLASSES


def _import_tokenizer_class(tokenizer_target: str) -> type:
    t = (tokenizer_target or "").strip()
    if "." not in t:
        raise ValueError(f"tokenizer target must be a dotted path, got {t!r}")
    # Older configs used ``modeling....`` when ``gess`` was the implicit package root.
    if t.startswith("modeling."):
        t = "gess." + t
    mod_path, _, cls_name = t.rpartition(".")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def _detail_vocab_size(model: nn.Module) -> int:
    vd = getattr(model, "vq_detail", None)
    if isinstance(vd, nn.ModuleList):
        return int(vd[0].num_embeddings)
    if vd is None:
        return 0
    return int(vd.num_embeddings)


def save_tokenizer_snapshot(
    output_dir: Path, tokenizer_target: str = "", overwrite: bool = False
) -> Optional[Path]:
    """Copy the tokenizer module source into the run output_dir for reproducibility.
    If overwrite is False and the snapshot file already exists, skip (resume keeps original snapshot).
    """
    tokenizer_dir = Path(__file__).resolve().parent.parent / "src" / "modeling" / "tokenizer"
    if _tokenizer_target_is_vq(tokenizer_target):
        cls = _tokenizer_class_from_target(tokenizer_target)
        snap = {
            "LidarTokenizerNoLL": "tokenizer_no_ll.py",
            "LidarTokenizerIndepVQ": "tokenizer_indep.py",
            "LidarTokenizerVanillaVQ": "tokenizer_vanilla.py",
        }.get(cls, "tokenizer_vq.py")
        dest_name = snap.replace(".py", "_snapshot.py")
        src = tokenizer_dir / snap
    else:
        src = tokenizer_dir / "tokenizer_fsq.py"
        dest_name = "tokenizer3D.py"
    if not src.exists():
        raise FileNotFoundError(f"Tokenizer module not found at {src}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / dest_name
    if dest.exists() and not overwrite:
        return None
    shutil.copy2(src, dest)
    return dest


def build_lidar_tokenizer_v5_from_params(t_params) -> LidarTokenizerV5:
    """
    Build ``LidarTokenizerV5`` from ``config.model.tokenizer.params`` (OmegaConf).
    Forwards supported ``LidarTokenizerV5`` keys; unknown keys are ignored. Also builds ``layer_cfg`` from
    ``tokenizer_conv_bias`` and ``tokenizer_standard_blocks`` when present.
    """
    fsq = OmegaConf.select(t_params, "fsq_levels", default=[7, 7, 7, 7])
    fsq_levels = list(OmegaConf.to_container(fsq, resolve=True))

    kwargs = {
        "input_channels": int(OmegaConf.select(t_params, "input_channels", default=1)),
        "latent_dim": int(OmegaConf.select(t_params, "latent_dim", default=4)),
        "hidden_dim": int(OmegaConf.select(t_params, "hidden_dim", default=64)),
        "num_scales": int(OmegaConf.select(t_params, "num_scales", default=3)),
        "fsq_levels": fsq_levels,
        "inter_loss_weight": float(OmegaConf.select(t_params, "inter_loss_weight", default=2.0)),
        "use_refiner": bool(OmegaConf.select(t_params, "use_refiner", default=False)),
        "apply_sparsity_mask_to_latents": bool(
            OmegaConf.select(t_params, "apply_sparsity_mask_to_latents", default=True)
        ),
        "use_null_token_routing": bool(
            OmegaConf.select(t_params, "use_null_token_routing", default=True)
        ),
    }
    unr = OmegaConf.select(t_params, "use_null_routing", default=None)
    if unr is not None:
        kwargs["use_null_routing"] = bool(unr)

    kwargs["layer_cfg"] = TokenizerV5LayerConfig(
        conv_bias=bool(OmegaConf.select(t_params, "tokenizer_conv_bias", default=False)),
        use_zero_preserving_resnorm=not bool(
            OmegaConf.select(t_params, "tokenizer_standard_blocks", default=False)
        ),
    )

    return LidarTokenizerV5(**kwargs)


def build_lidar_tokenizer_vq_from_params(t_params, tokenizer_target: str) -> nn.Module:
    """
    Build a Haar+VQ tokenizer from ``config.model.tokenizer.params`` (OmegaConf) and
    ``tokenizer_target`` (fully qualified class name).
    """
    kwargs = {
        "input_channels": int(OmegaConf.select(t_params, "input_channels", default=1)),
        "latent_dim": int(OmegaConf.select(t_params, "latent_dim", default=4)),
        "hidden_dim": int(OmegaConf.select(t_params, "hidden_dim", default=64)),
        "num_scales": int(OmegaConf.select(t_params, "num_scales", default=3)),
        "codebook_size_root": int(OmegaConf.select(t_params, "codebook_size_root", default=175)),
        "codebook_size_detail": int(OmegaConf.select(t_params, "codebook_size_detail", default=512)),
        "commitment_cost": float(OmegaConf.select(t_params, "commitment_cost", default=0.25)),
        "vq_loss_weight": float(OmegaConf.select(t_params, "vq_loss_weight", default=1.0)),
        "vq_use_tanh_bound": bool(OmegaConf.select(t_params, "vq_use_tanh_bound", default=True)),
        "inter_loss_weight": float(OmegaConf.select(t_params, "inter_loss_weight", default=2.0)),
        "use_refiner": bool(OmegaConf.select(t_params, "use_refiner", default=False)),
        "apply_sparsity_mask_to_latents": bool(
            OmegaConf.select(t_params, "apply_sparsity_mask_to_latents", default=True)
        ),
    }

    kwargs["layer_cfg"] = TokenizerV5LayerConfig(
        conv_bias=bool(OmegaConf.select(t_params, "tokenizer_conv_bias", default=False)),
        use_zero_preserving_resnorm=not bool(
            OmegaConf.select(t_params, "tokenizer_standard_blocks", default=False)
        ),
    )

    cls = _import_tokenizer_class(tokenizer_target)
    return cls(**kwargs)


def build_tokenizer_from_config(config) -> nn.Module:
    """Instantiate tokenizer from ``config.model.tokenizer.target`` + ``params``."""
    t_params = config.model.tokenizer.params
    target = str(OmegaConf.select(config.model.tokenizer, "target", default=""))
    if _tokenizer_target_is_vq(target):
        return build_lidar_tokenizer_vq_from_params(t_params, target)
    return build_lidar_tokenizer_v5_from_params(t_params)


def get_latest_checkpoint(logging_dir: Path):
    if not logging_dir.exists():
        return None, 0
        
    checkpoints = list(logging_dir.glob("checkpoint-*"))
    checkpoints = [c for c in checkpoints if c.name.split("-")[-1].isdigit()]
    
    if not checkpoints:
        return None, 0
        
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    latest_path = checkpoints[-1]
    step = int(latest_path.name.split("-")[-1])
    return latest_path, step

def save_checkpoint(accelerator, config, global_step, model):
    output_dir = Path(config.experiment.logging_dir) / f"checkpoint-{global_step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator.save_state(output_dir)
    
    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, 'save_pretrained'):
        unwrapped_model.save_pretrained(output_dir)
    else:
        torch.save(unwrapped_model.state_dict(), output_dir / "pytorch_model.bin")
    
    metadata = {
        "global_step": global_step,
        "config": OmegaConf.to_container(config, resolve=True),
        "timestamp": time.time()
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, cls=PathEncoder)
        
    return output_dir

# ==============================================================================
# METRICS (UPDATED FOR 3D)
# ==============================================================================

def calculate_entropy(indices, vocab_size):
    flat = indices.flatten().float()
    if flat.numel() == 0: return 0.0
    counts = torch.histc(flat, bins=int(vocab_size), min=0, max=int(vocab_size)-1)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy.item()

def calculate_active_percentage(indices, null_idx):
    flat = indices.flatten()
    active = (flat != null_idx).float().sum()
    return (active / flat.numel()) * 100.0

def compute_ssim_2d(img1, img2, window_size=11):
    """Computes SSIM natively on GPU. Expects [N, 1, H, W]"""
    padding = window_size // 2
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]).to(img1.device)
    gauss = gauss / gauss.sum()
    kernel_2d = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, kernel_2d, padding=padding)
    mu2 = F.conv2d(img2, kernel_2d, padding=padding)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, kernel_2d, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel_2d, padding=padding) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel_2d, padding=padding) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



# ==============================================================================
# TRAINING LOGIC
# ==============================================================================
def train_step(model, batch, optimizer, lr_scheduler, accelerator, config, global_step, schedulers):
    _, hr_imgs = batch # [B, 1, H, W]
    model.train()
    
    """current_t =  0.0 
    current_r = 0.0
    
    if global_step >= 5000:
        # Ramps smoothly from 0.0 up to t_params.vbr.threshold
        current_t = schedulers['vbr_t'].get_value(global_step - 5000)
        current_r = min(0.05, (global_step - 5000) * 0.00001)"""
    
    #current_t = 0.0 
    #current_r = 0.0
    
    # COMMENT THIS OUT FOR RUN 1:
    # if global_step >= 5000:
    #     current_t = schedulers['vbr_t'].get_value(global_step - 5000)
    #     current_r = min(0.05, (global_step - 5000) * 0.00001)

    with accelerator.accumulate(model):
        #recon, q_loss, info = model(hr_imgs, vbr_threshold=current_t, vbr_ratio=current_r, test=False)
        recon, q_loss, info = model(hr_imgs, test=False)
        accelerator.backward(q_loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config.training.compute.max_grad_norm)
            
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Added safe .item() calls to prevent memory leaks in tracking
        metrics = {
            "train/total_loss": q_loss.item(),
            "train/rec_mse": info["recon_mse"].item(),
        }
        if "total_commit_loss" in info and isinstance(info["total_commit_loss"], torch.Tensor):
            metrics["train/commit_loss"] = info["total_commit_loss"].item()
        if "commit_loss_scaled" in info and isinstance(info["commit_loss_scaled"], torch.Tensor):
            metrics["train/commit_loss_scaled"] = info["commit_loss_scaled"].item()
        if "commit_root" in info and isinstance(info["commit_root"], torch.Tensor):
            metrics["train/commit_root"] = info["commit_root"].item()
        if "fg_loss" in info and isinstance(info["fg_loss"], torch.Tensor):
            metrics["train/fg_loss"] = info["fg_loss"].item()
        if "bg_loss" in info and isinstance(info["bg_loss"], torch.Tensor):
            metrics["train/bg_loss"] = info["bg_loss"].item()
        if "inter_loss_total" in info and isinstance(info["inter_loss_total"], torch.Tensor):
            metrics["train/inter_loss_total"] = info["inter_loss_total"].item()

        if "debug_stats" in info:
            for k, v in info["debug_stats"].items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    metrics[f"debug/{k}"] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    try:
                        val = v.item()
                        if math.isfinite(val):
                            metrics[f"debug/{k}"] = val
                    except Exception:
                        pass

        return metrics, recon

@torch.no_grad()
def validation_loop(model, val_loader, accelerator, global_step, config):
    model.eval()
    avg_psnr = AverageMeter()
    avg_mse = AverageMeter()
    avg_ssim = AverageMeter()
    
    viz_batch = None
    viz_outputs = None
    first_batch_info = None
    
    for batch_idx, batch in enumerate(val_loader):
        _, hr_imgs = batch # [B, 1, H, W]
        #output, curr_recon, info = model(hr_imgs, vbr_threshold=-1.0, vbr_ratio=0.0, test=True)
        output, curr_recon, info = model(hr_imgs, test=True)

        mse = F.mse_loss(output, hr_imgs)
        avg_mse.update(mse.item(), hr_imgs.size(0))

        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        avg_psnr.update(psnr.item(), hr_imgs.size(0))

        b, c, h, w = hr_imgs.shape
        ssim_val = compute_ssim_2d(hr_imgs, output)
        avg_ssim.update(ssim_val.item(), hr_imgs.size(0))

        if batch_idx == 0:
            viz_batch = hr_imgs
            viz_outputs = {
                "final": output,
                "refinement": output,
                "mask": info.get("active_mask", None),
            }
            first_batch_info = info

    final_psnr = accelerator.gather(torch.tensor(avg_psnr.avg, device=accelerator.device)).mean().item()
    final_mse = accelerator.gather(torch.tensor(avg_mse.avg, device=accelerator.device)).mean().item()
    final_ssim = accelerator.gather(torch.tensor(avg_ssim.avg, device=accelerator.device)).mean().item()
    
    metrics = {
        "val/psnr": final_psnr,
        "val/mse": final_mse,
        "val/ssim": final_ssim,
    }
    
    if first_batch_info is not None and "debug_stats" in first_batch_info:
        for k, v in first_batch_info["debug_stats"].items():
            try:
                if isinstance(v, (int, float)) and math.isfinite(v):
                    metrics[f"val/debug/{k}"] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    val = v.item()
                    if math.isfinite(val):
                        metrics[f"val/debug/{k}"] = val
            except Exception:
                pass
                
    if first_batch_info is not None and "intermediate_mse_per_scale" in first_batch_info:
        for i, mse_val in enumerate(first_batch_info["intermediate_mse_per_scale"]):
            name = "root" if i == 0 else f"scale{i - 1}"
            metrics[f"val/mse_{name}"] = mse_val if isinstance(mse_val, float) else mse_val

    # --- ROBUST 2D VISUALIZATION ---
    if accelerator.is_main_process and viz_batch is not None:
        try:
            n_viz = min(8, viz_batch.shape[0])
            
            def apply_cmap(tensor, cmap_name="viridis", vmin=0.0, vmax=1.0):
                x = tensor.detach().cpu().squeeze(1).numpy()
                if hasattr(matplotlib, "colormaps"):
                    cmap = matplotlib.colormaps[cmap_name]
                else:
                    cmap = matplotlib.cm.get_cmap(cmap_name)
                norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
                colored = cmap(norm(x))
                return torch.from_numpy(colored[..., :3]).permute(0, 3, 1, 2).float()

            # REMOVED .mean(dim=2) here to support strict 2D processing
            real_2d = viz_batch[:n_viz]
            recon_2d = viz_outputs['final'][:n_viz]
            error_2d = torch.abs(viz_batch[:n_viz] - viz_outputs['final'][:n_viz])
            
            real_viz = apply_cmap(real_2d, "viridis")
            recon_viz = apply_cmap(recon_2d, "viridis")
            error_viz = apply_cmap(error_2d, cmap_name="inferno", vmin=0.0, vmax=0.2)
            
            refinement_data_2d = viz_outputs['refinement'][:n_viz]
            refinement_viz = apply_cmap(torch.clamp(refinement_data_2d * 5.0 + 0.5, 0, 1), "seismic", vmin=-0.2, vmax=0.0)

            mask = viz_outputs['mask']
            if mask is not None:
                mask_2d = mask[:n_viz].float()
                mask_viz = mask_2d.repeat(1, 3, 1, 1) 
            else:
                mask_viz = torch.zeros_like(real_2d).repeat(1, 3, 1, 1)

            def grid_np(t):
                g = make_grid(t, nrow=4, normalize=False, padding=2)
                g_np = (g.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype('uint8')
                return wandb.Image(g_np)

            accelerator.log({
                "val/vis_real": grid_np(real_viz),
                "val/vis_recon": grid_np(recon_viz),
                "val/vis_error": grid_np(error_viz),
                "val/vis_refinement": grid_np(refinement_viz),
                "val/vis_mask": grid_np(mask_viz)
            }, step=global_step)
            
        except Exception as e:
            print(f"WARNING: Failed to log images to WandB at step {global_step}. Skipping visualization. Error: {e}")

    return metrics

def train_step_lpips(model, batch, optimizer, lr_scheduler, accelerator, config, global_step, lpips_loss_fn, lpips_weight=0.1):
    _, hr_imgs = batch # [B, 1, H, W]
    model.train()
    
    #current_t = 0.0 
    #current_r = 0.0
    
    # COMMENT THIS OUT FOR RUN 1:
    # if global_step >= 5000:
    #     current_t = schedulers['vbr_t'].get_value(global_step - 5000)
    #     current_r = min(0.05, (global_step - 5000) * 0.00001)

    with accelerator.accumulate(model):
        #recon, q_loss, info = model(hr_imgs, vbr_threshold=current_t, vbr_ratio=current_r, test=False)
        recon, q_loss, info = model(hr_imgs, test=False)

        # --- LPIPS CALCULATION ---
        # 1. Scale from [0, 1] to [-1, 1]
        # 2. Repeat the 1 channel to 3 channels to satisfy LPIPS VGG requirement
        hr_imgs_lpips = (hr_imgs * 2.0 - 1.0).repeat(1, 3, 1, 1)
        recon_lpips = (recon * 2.0 - 1.0).repeat(1, 3, 1, 1)
        
        lpips_loss = lpips_loss_fn(recon_lpips, hr_imgs_lpips).mean()
        
        # Add to total loss
        total_loss = q_loss + (lpips_weight * lpips_loss)
        
        accelerator.backward(total_loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config.training.compute.max_grad_norm)
            
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Added safe .item() calls to prevent memory leaks in tracking
        metrics = {
            "train/total_loss": total_loss.item(),
            "train/q_loss": q_loss.item(),
            "train/lpips_loss": lpips_loss.item(),
            "train/rec_mse": info["recon_mse"].item(),
        }
        if "total_commit_loss" in info and isinstance(info["total_commit_loss"], torch.Tensor):
            metrics["train/commit_loss"] = info["total_commit_loss"].item()
        if "commit_loss_scaled" in info and isinstance(info["commit_loss_scaled"], torch.Tensor):
            metrics["train/commit_loss_scaled"] = info["commit_loss_scaled"].item()
        if "commit_root" in info and isinstance(info["commit_root"], torch.Tensor):
            metrics["train/commit_root"] = info["commit_root"].item()
        if "fg_loss" in info and isinstance(info["fg_loss"], torch.Tensor):
            metrics["train/fg_loss"] = info["fg_loss"].item()
        if "bg_loss" in info and isinstance(info["bg_loss"], torch.Tensor):
            metrics["train/bg_loss"] = info["bg_loss"].item()
        if "inter_loss_total" in info and isinstance(info["inter_loss_total"], torch.Tensor):
            metrics["train/inter_loss_total"] = info["inter_loss_total"].item()

        if "debug_stats" in info:
            for k, v in info["debug_stats"].items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    metrics[f"debug/{k}"] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    try:
                        val = v.item()
                        if math.isfinite(val):
                            metrics[f"debug/{k}"] = val
                    except Exception:
                        pass

        return metrics, recon

@torch.no_grad()
def validation_loop_lpips(model, val_loader, accelerator, global_step, config, lpips_loss_fn):
    model.eval()
    avg_psnr = AverageMeter()
    avg_mse = AverageMeter()
    avg_ssim = AverageMeter()
    avg_lpips = AverageMeter() # Added LPIPS meter
    
    viz_batch = None
    viz_outputs = None
    first_batch_info = None
    
    for batch_idx, batch in enumerate(val_loader):
        _, hr_imgs = batch # [B, 1, H, W]
        #output, curr_recon, info = model(hr_imgs, vbr_threshold=-1.0, vbr_ratio=0.0, test=True)
        output, curr_recon, info = model(hr_imgs, test=True)
        mse = F.mse_loss(output, hr_imgs)
        avg_mse.update(mse.item(), hr_imgs.size(0))

        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        avg_psnr.update(psnr.item(), hr_imgs.size(0))

        b, c, h, w = hr_imgs.shape
        ssim_val = compute_ssim_2d(hr_imgs, output)
        avg_ssim.update(ssim_val.item(), hr_imgs.size(0))
        
        # --- LPIPS CALCULATION ---
        hr_imgs_lpips = (hr_imgs * 2.0 - 1.0).repeat(1, 3, 1, 1)
        output_lpips = (output * 2.0 - 1.0).repeat(1, 3, 1, 1)
        
        lpips_val = lpips_loss_fn(output_lpips, hr_imgs_lpips).mean()
        avg_lpips.update(lpips_val.item(), hr_imgs.size(0))

        if batch_idx == 0:
            viz_batch = hr_imgs
            viz_outputs = {
                "final": output,
                "refinement": output,
                "mask": info.get("active_mask", None),
            }
            first_batch_info = info

    final_psnr = accelerator.gather(torch.tensor(avg_psnr.avg, device=accelerator.device)).mean().item()
    final_mse = accelerator.gather(torch.tensor(avg_mse.avg, device=accelerator.device)).mean().item()
    final_ssim = accelerator.gather(torch.tensor(avg_ssim.avg, device=accelerator.device)).mean().item()
    final_lpips = accelerator.gather(torch.tensor(avg_lpips.avg, device=accelerator.device)).mean().item()
    
    metrics = {
        "val/psnr": final_psnr,
        "val/mse": final_mse,
        "val/ssim": final_ssim,
        "val/lpips": final_lpips, # Log gathered LPIPS
    }
    
    if first_batch_info is not None and "debug_stats" in first_batch_info:
        for k, v in first_batch_info["debug_stats"].items():
            try:
                if isinstance(v, (int, float)) and math.isfinite(v):
                    metrics[f"val/debug/{k}"] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    val = v.item()
                    if math.isfinite(val):
                        metrics[f"val/debug/{k}"] = val
            except Exception:
                pass
                
    if first_batch_info is not None and "intermediate_mse_per_scale" in first_batch_info:
        for i, mse_val in enumerate(first_batch_info["intermediate_mse_per_scale"]):
            name = "root" if i == 0 else f"scale{i - 1}"
            metrics[f"val/mse_{name}"] = mse_val if isinstance(mse_val, float) else mse_val

    # --- ROBUST 2D VISUALIZATION ---
    if accelerator.is_main_process and viz_batch is not None:
        try:
            n_viz = min(8, viz_batch.shape[0])
            
            def apply_cmap(tensor, cmap_name="viridis", vmin=0.0, vmax=1.0):
                x = tensor.detach().cpu().squeeze(1).numpy()
                if hasattr(matplotlib, "colormaps"):
                    cmap = matplotlib.colormaps[cmap_name]
                else:
                    cmap = matplotlib.cm.get_cmap(cmap_name)
                norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
                colored = cmap(norm(x))
                return torch.from_numpy(colored[..., :3]).permute(0, 3, 1, 2).float()

            real_2d = viz_batch[:n_viz]
            recon_2d = viz_outputs['final'][:n_viz]
            error_2d = torch.abs(viz_batch[:n_viz] - viz_outputs['final'][:n_viz])
            
            real_viz = apply_cmap(real_2d, "viridis")
            recon_viz = apply_cmap(recon_2d, "viridis")
            error_viz = apply_cmap(error_2d, cmap_name="inferno", vmin=0.0, vmax=0.2)
            
            refinement_data_2d = viz_outputs['refinement'][:n_viz]
            refinement_viz = apply_cmap(torch.clamp(refinement_data_2d * 5.0 + 0.5, 0, 1), "seismic", vmin=-0.2, vmax=0.0)

            mask = viz_outputs['mask']
            if mask is not None:
                mask_2d = mask[:n_viz].float()
                mask_viz = mask_2d.repeat(1, 3, 1, 1) 
            else:
                mask_viz = torch.zeros_like(real_2d).repeat(1, 3, 1, 1)

            def grid_np(t):
                g = make_grid(t, nrow=4, normalize=False, padding=2)
                g_np = (g.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype('uint8')
                return wandb.Image(g_np)

            accelerator.log({
                "val/vis_real": grid_np(real_viz),
                "val/vis_recon": grid_np(recon_viz),
                "val/vis_error": grid_np(error_viz),
                "val/vis_refinement": grid_np(refinement_viz),
                "val/vis_mask": grid_np(mask_viz)
            }, step=global_step)
            
        except Exception as e:
            print(f"WARNING: Failed to log images to WandB at step {global_step}. Skipping visualization. Error: {e}")

    return metrics


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Prefer POSIX flags (--config / --run). OmegaConf.from_cli() alone only accepts dotlist
    # (e.g. config=foo.yaml run=1), not --run 1 — so we use argparse for the common case.
    argv = sys.argv[1:]
    cli_overrides = OmegaConf.create()

    if "--config" in argv:
        parser = argparse.ArgumentParser(description="Train LiDAR tokenizer (V5 FSQ or VQ).")
        parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
        parser.add_argument("--run", type=str, required=True, help="Run id suffix for output path / logging")
        args, dotlist_argv = parser.parse_known_args()

        yaml_conf = OmegaConf.load(args.config)
        cli_fixed = OmegaConf.create({"run": args.run})
        if dotlist_argv:
            try:
                cli_overrides = OmegaConf.from_cli(dotlist_argv)
            except Exception as e:
                print(
                    "Error parsing extra CLI arguments as OmegaConf dotlist (key=value). "
                    f"Remainder: {dotlist_argv!r}. Underlying error: {e}"
                )
                sys.exit(1)
            config = OmegaConf.merge(yaml_conf, cli_fixed, cli_overrides)
        else:
            config = OmegaConf.merge(yaml_conf, cli_fixed)
    else:
        cli_conf = OmegaConf.from_cli()
        if not hasattr(cli_conf, "run") or cli_conf.run is None:
            print(
                "Error: missing run id. Use either:\n"
                "  python training/train_tokenizer.py --config CONFIG.yaml --run RUN_ID\n"
                "  python training/train_tokenizer.py config=CONFIG.yaml run=RUN_ID"
            )
            sys.exit(1)
        if not hasattr(cli_conf, "config") or cli_conf.config is None:
            print(
                "Error: missing config path. Use either:\n"
                "  python training/train_tokenizer.py --config CONFIG.yaml --run RUN_ID\n"
                "  python training/train_tokenizer.py config=CONFIG.yaml run=RUN_ID"
            )
            sys.exit(1)
        yaml_conf = OmegaConf.load(cli_conf.config)
        config = OmegaConf.merge(yaml_conf, cli_conf)
        cli_overrides = cli_conf

    if "rq_depths" in cli_overrides:
        print(
            f"--> [CLI] rq_depths={cli_overrides.rq_depths} (suffix on run id only; "
            "tokenizer V5/VQ configs ignore RQ — see config.model.tokenizer.params)"
        )

    set_seed(config.globals.seed)
    workspace = Path(os.environ.get("WORKSPACE", "./outputs/tokenizer_runs"))
    unique_run_id = f"{config.experiment.run_name}_{config.run}"
    
    if "rq_depths" in cli_overrides:
        unique_run_id += f"_rq{cli_overrides.rq_depths}"

    output_dir = workspace / "outputs" / config.experiment.project / config.experiment.name / unique_run_id
    logging_dir = output_dir / "logs"
    config.experiment.logging_dir = str(logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    config.experiment.logging_dir = logging_dir
    
    config_path = output_dir / "config.yaml"
    OmegaConf.save(config=OmegaConf.to_container(config, resolve=True), f=config_path)

    accelerator = Accelerator(
        log_with="wandb",
        project_dir=logging_dir,
        gradient_accumulation_steps=config.training.compute.grad_accum_steps,
        mixed_precision=config.training.compute.mixed_precision 
    )
    
    logger = setup_logger(
        name=config.experiment.project,
        output_dir=logging_dir,
        use_accelerate=True
    )

    # Snapshot the tokenizer module used for this run (next to config.yaml); skip if already present (e.g. resume)
    tok_target = str(OmegaConf.select(config.model.tokenizer, "target", default=""))
    try:
        snapshot_path = save_tokenizer_snapshot(output_dir, tokenizer_target=tok_target)
        if snapshot_path is not None:
            logger.info(f"Saved tokenizer snapshot to: {snapshot_path}")
    except Exception as e:
        logger.warning(f"Could not save tokenizer snapshot: {e}")
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.experiment.project,
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": {"name": unique_run_id}}
        )

    t_params = config.model.tokenizer.params
    model = build_tokenizer_from_config(config)
    if _tokenizer_target_is_vq(str(OmegaConf.select(config.model.tokenizer, "target", default=""))):
        logger.info(
            "Built %s (V_root=%s, V_detail=%s, commitment_cost=%s, vq_loss_weight=%s, "
            "vq_use_tanh_bound=%s, apply_sparsity_mask_to_latents=%s, use_refiner=%s)",
            type(model).__name__,
            model.vq_root.num_embeddings,
            _detail_vocab_size(model),
            model.commitment_cost,
            model.vq_loss_weight,
            model.vq_use_tanh_bound,
            model.apply_sparsity_mask_to_latents,
            model.use_refiner,
        )
    else:
        logger.info(
            "Built LidarTokenizerV5 (root FSQ [7,5,5] fixed in code; detail fsq_levels=%s, "
            "apply_sparsity_mask_to_latents=%s, use_null_token_routing=%s, use_null_routing=%s)",
            model.fsq_levels,
            model.apply_sparsity_mask_to_latents,
            model.use_null_token_routing,
            OmegaConf.select(t_params, "use_null_routing", default=None),
        )

    # Initialize LPIPS and freeze it
    lpips_loss_fn = lpips.LPIPS(net='vgg').eval()
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False
        
    # MOVE TO ACCELERATOR DEVICE MANUALLY
    lpips_loss_fn.to(accelerator.device)
        
    # Make sure to pass it to the accelerator later so it goes to the correct device!
    normalize_diffusion = config.dataset.loader.normalize_diffusion
    preload_to_ram = config.dataset.loader.preload_to_ram
    train_loader, _, val_loader = build_dataloaders(config.dataset, normalize_diffusion=normalize_diffusion, preload_to_ram=preload_to_ram)
    
    compander_params = []
    base_params_decay = []
    base_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1. Compander parameters (Low LR, NO weight decay)
        if 'compander' in name or 'gamma' in name or 'scale_factor' in name:
            compander_params.append(param)
            
        # 2. Biases and 1D parameters like Norms (Normal LR, NO weight decay)
        elif len(param.shape) == 1 or name.endswith(".bias"):
            base_params_no_decay.append(param)
            
        # 3. Convolution weights / Linear matrices (Normal LR, YES weight decay)
        else:
            base_params_decay.append(param)

    # Note: We remove weight_decay from the AdamW kwargs and specify it per-group
    optimizer = AdamW([
        {'params': base_params_decay, 'lr': config.globals.learning_rate, 'weight_decay': 1e-4},      
        {'params': base_params_no_decay, 'lr': config.globals.learning_rate, 'weight_decay': 0.0},
        {'params': compander_params, 'lr': config.globals.learning_rate * 0.1, 'weight_decay': 0.0} 
    ])

    max_steps = config.training.duration.max_steps
    lr_scheduler = get_scheduler(
        name=config.optimizer.scheduler.name, 
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.scheduler.warmup_steps,
        num_training_steps=max_steps
    )
    
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    steps_per_epoch = len(train_loader)
    total_epochs = math.ceil(max_steps / steps_per_epoch)

    resume_path, resume_step = get_latest_checkpoint(logging_dir)
    is_actually_resuming = config.experiment.resume and (resume_path is not None)
    
    global_step = 0
    current_epoch = 0
    resume_batches_to_skip = 0 

    if is_actually_resuming:
        if resume_path:
            accelerator.print(f"--> [RESUME] Loading full state from: {resume_path}")
            accelerator.print(f"    Resuming from global_step: {resume_step}")
            
            accelerator.load_state(resume_path)
            
            global_step = resume_step
            current_epoch = global_step // steps_per_epoch
            
            resume_batches_to_skip = global_step % steps_per_epoch
            
            if resume_batches_to_skip > 0:
                accelerator.print(f"    - Fast-forwarding data loader by {resume_batches_to_skip} steps...")
    else:
        accelerator.print("--> [INIT] Starting training from scratch.")

    progress_bar = tqdm(
        range(global_step, max_steps), 
        initial=global_step, 
        total=max_steps, 
        disable=not accelerator.is_main_process
    )
    
    # ADDED: Data and Compute timers to your tracker
    tracker = {
        "loss": AverageMeter(), 
        "mse": AverageMeter(),
        "data_time": AverageMeter(), 
        "compute_time": AverageMeter()
    }
    logger.info(f"Starting Training: {total_epochs} Epochs | {max_steps} Steps")

    while global_step < max_steps:
        if accelerator.is_main_process:
            progress_bar.set_description(f"Epoch {current_epoch}")

        # ADDED: Start timer for data loading before entering the batch loop
        data_start_time = time.time()
        
        for batch in train_loader:
            # Sync GPU to ensure previous batch is fully done before stopping data timer
            torch.cuda.synchronize() 
            data_time = time.time() - data_start_time
            tracker["data_time"].update(data_time)
            
            # Start timer for the actual model compute (forward + backward + step)
            compute_start_time = time.time()
            
            metrics, _ = train_step_lpips(
                model, batch, optimizer, lr_scheduler, accelerator, config, global_step, lpips_loss_fn
            )

            # Sync GPU to ensure the model step is completely finished
            torch.cuda.synchronize()
            compute_time = time.time() - compute_start_time
            tracker["compute_time"].update(compute_time)
            
            tracker["loss"].update(metrics["train/total_loss"])
            tracker["mse"].update(metrics["train/rec_mse"])
            
            if global_step % config.experiment.intervals.log_steps == 0:
                metrics["train/epoch"] = global_step / steps_per_epoch
                # Log speeds to WandB/Tensorboard as well!
                metrics["speed/data_time"] = tracker["data_time"].avg
                metrics["speed/compute_time"] = tracker["compute_time"].avg
                
                accelerator.log(metrics, step=global_step)
                
                if accelerator.is_main_process:
                    # ADDED: Cleanly formatted speed metrics in tqdm
                    progress_bar.set_postfix({
                        "Loss": f"{tracker['loss'].avg:.4f}", 
                        "MSE": f"{tracker['mse'].avg:.4f}",
                        "Data": f"{tracker['data_time'].avg:.3f}s",
                        "Step": f"{tracker['compute_time'].avg:.3f}s",
                    })
            
            if global_step % config.experiment.intervals.save_every == 0 and global_step > 0:  
                if accelerator.is_main_process:
                    saved_path = save_checkpoint(accelerator, config, global_step, model)
                    logger.info(f"Saved checkpoint to: {saved_path}")

            if global_step % config.experiment.intervals.eval_every == 0 and global_step > 0:         
                if accelerator.is_main_process:
                    #val_metrics = validation_loop(model, val_loader, accelerator, global_step, config)
                    val_metrics = validation_loop_lpips(model, val_loader, accelerator, global_step, config, lpips_loss_fn)
                    accelerator.log(val_metrics, step=global_step)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
            
            if global_step >= max_steps:
                break
                
            # ADDED: Reset the data loading timer for the next iteration
            data_start_time = time.time()
        
        current_epoch += 1

    accelerator.end_training()
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()