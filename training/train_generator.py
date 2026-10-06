import os
import sys

# Add project root to sys.path so we can import src.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""
Training Script for Hierarchical Generative Model V4 (SOTA Spatial Conditioning)

This script implements training for the V4 refactor with Two-Phase Training:
- Phase 1: Pretrains the Spatial Conditioner to invert LiDAR physics (DWT matching).
- Phase 2: Main AR Generation training with SOTA spatial injection.
- Preserves all rigorous V2/V4 checkpointing and wandb logging.
"""
print(f"[Imports] Importing Libraries:")

import sys, os, json, math, time, shutil, warnings
import importlib
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
print(f"  -> Imported basics")

# Accelerate & PyTorch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
from torch.optim import AdamW
print(f"  -> Imported torch & accelerate packages")

# Utility 
from src.utils.meter import AverageMeter
from src.utils.logger import setup_logger
from src.utils.lr_schedulers import get_scheduler
from src.utils.optimizer_utils import adamw_param_groups_from_named_pairs, adamw_param_groups_two_level
from src.utils.diagnostics import log_visuals
print(f"  -> Imported custom utilities")

# V4 Imports
from src.modeling.generator.generator_merged import HierarchicalGenerator, _resolve_vocab_sizes
from src.modeling.generator.conditioner_layer_config import PaperConvNeXtLayerConfig, ConditionerLayerConfig
from src.modeling.generator.generator_loss import compute_multi_scale_loss_v5
from src.modeling.generator.conditioner import SuperResWaveletConditionerV4, ConditionerWaveletPretrainer 
from src.modeling.tokenizer.tokenizer_fsq import LidarTokenizerV5 as LidarTokenizerV5Joint
from src.modeling.tokenizer.tokenizer_v5_layer_config import TokenizerV5LayerConfig
from src.data.lidar_dataloading import build_dataloaders 

def _normalize_tokenizer_layout(layout: str) -> str:
    return "joint"

print(f"  -> Imported model assets")

# Use Safetensors for loading if available
try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    load_file = None  # Will use torch.load as fallback

logger = get_logger(__name__)

# ==============================================================================
# 1. ROBUST HELPER FUNCTIONS (Preserved exactly as provided)
# ==============================================================================

def get_latest_checkpoint(logging_dir: Path):
    if not logging_dir.exists(): return None, 0
    checkpoints = list(logging_dir.glob("checkpoint-*"))
    valid_checkpoints = []
    for c in checkpoints:
        parts = c.name.split("-")
        if len(parts) == 2 and parts[1].isdigit():
            step = int(parts[1])
            valid_checkpoints.append((c, step))
    if not valid_checkpoints: return None, 0
    valid_checkpoints.sort(key=lambda x: x[1])
    return valid_checkpoints[-1]


def get_latest_checkpoint_across_dirs(logging_dirs: list):
    """
    Return the checkpoint with the largest global step among existing ``logs`` dirs.
    Used to resume from a legacy tree (e.g. Lidar-SSM) when the current workspace has no saves yet.
    """
    best_path, best_step = None, -1
    for d in logging_dirs:
        if d is None:
            continue
        p, s = get_latest_checkpoint(Path(d))
        if p is not None and s > best_step:
            best_path, best_step = p, s
    if best_step < 0:
        return None, 0
    return best_path, best_step


def resolve_pretrained_cond_checkpoint(
    raw,
    workspace: Path,
    legacy_workspace: Path,
    base_logging_dir: Path,
    legacy_logging_dir: Optional[Path],
) -> Optional[Path]:
    """
    Resolve ``training.pretraining.pretrained_cond_path`` to a directory containing ``model_1.safetensors``.

    Tries, in order:
    - Path as given (``expanduser``)
    - If the string is digits only: ``<run logs>/checkpoint-<step>`` under primary then legacy ``logs``
    - If relative: ``WORKSPACE / path`` then ``WORKSPACE_LEGACY / path`` (GeSS vs Lidar-SSM trees)
    - A file path ending in ``model_1.safetensors`` resolves to its parent directory
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None

    def _ok(d: Path) -> bool:
        return d.is_dir() and (d / "model_1.safetensors").exists()

    p = Path(s).expanduser()
    candidates: list[Path] = []

    if p.is_file() and p.name == "model_1.safetensors":
        return p.parent if _ok(p.parent) else None

    candidates.append(p)
    if s.isdigit():
        step = int(s)
        candidates.insert(0, base_logging_dir / f"checkpoint-{step}")
        if legacy_logging_dir is not None:
            candidates.insert(1, legacy_logging_dir / f"checkpoint-{step}")
    if not p.is_absolute():
        candidates.append(workspace / p)
        if legacy_workspace.resolve() != workspace.resolve():
            candidates.append(legacy_workspace / p)

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        try:
            if _ok(c):
                return c
        except OSError:
            continue
    return None


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path): return str(obj)
        return super().default(obj)

class CheckpointManager:
    def __init__(self, logging_dir: Path, keep_last_n: int = 3):
        self.logging_dir = Path(logging_dir)
        self.keep_last_n = keep_last_n
        self.best_metrics = {
            'best_val_loss': {'value': float('inf'), 'step': None, 'path': None},
            'best_lpips': {'value': float('inf'), 'step': None, 'path': None},
        }
        self._load_best_metrics()
    
    def _load_best_metrics(self):
        best_metrics_file = self.logging_dir / "best_metrics.json"
        if best_metrics_file.exists():
            try:
                with open(best_metrics_file, 'r') as f:
                    saved = json.load(f)
                    for key in self.best_metrics:
                        if key in saved:
                            self.best_metrics[key]['value'] = saved[key].get('value', self.best_metrics[key]['value'])
                            self.best_metrics[key]['step'] = saved[key].get('step', None)
            except Exception as e:
                print(f"Warning: Could not load best metrics: {e}")
    
    def _save_best_metrics(self):
        best_metrics_file = self.logging_dir / "best_metrics.json"
        with open(best_metrics_file, 'w') as f:
            json.dump({k: {'value': v['value'], 'step': v['step']} for k, v in self.best_metrics.items()}, f, indent=2)
    
    def _is_best_checkpoint(self, metrics: dict) -> dict:
        if metrics is None:
            return {}
        to_save = {}
        if 'val/total_loss' in metrics and isinstance(metrics['val/total_loss'], (int, float)):
            val_loss = metrics['val/total_loss']
            if val_loss < self.best_metrics['best_val_loss']['value']:
                to_save['best_val_loss'] = True
                self.best_metrics['best_val_loss']['value'] = val_loss
        lp = metrics.get('val/lpips')
        if lp is not None and isinstance(lp, (int, float)) and math.isfinite(lp):
            if lp < self.best_metrics['best_lpips']['value']:
                to_save['best_lpips'] = True
                self.best_metrics['best_lpips']['value'] = lp
        return to_save
    
    def save_checkpoint(self, accelerator, config, global_step: int, model, metrics: dict = None, is_best: bool = False) -> Path:
        save_path = self.logging_dir / f"checkpoint-{global_step}"
        accelerator.save_state(save_path)
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(save_path)
        
        metadata = {"global_step": global_step, "config": OmegaConf.to_container(config, resolve=True)}
        if metrics: metadata["metrics"] = metrics
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4, cls=PathEncoder)
        
        best_types = self._is_best_checkpoint(metrics) if metrics else ({'best_final': True} if is_best else {})
        for best_type, is_new_best in best_types.items():
            if is_new_best:
                best_path = self.logging_dir / f"{best_type}-step-{global_step}"
                if self.best_metrics.get(best_type, {}).get('path') is not None:
                    old_path = Path(self.best_metrics[best_type]['path'])
                    if old_path.exists() and old_path != save_path:
                        try: shutil.rmtree(old_path)
                        except OSError as e: print(f"Warning: Failed to delete old {best_type} checkpoint: {e}")
                try:
                    if best_path.exists(): shutil.rmtree(best_path)
                    shutil.copytree(save_path, best_path)
                    if best_type in self.best_metrics:
                        self.best_metrics[best_type]['path'] = str(best_path)
                        self.best_metrics[best_type]['step'] = global_step
                    if accelerator.is_main_process:
                        accelerator.print(f"  > 💾 New {best_type} checkpoint saved at step {global_step}")
                except Exception as e:
                    print(f"Warning: Failed to save {best_type} checkpoint: {e}")
        
        self._save_best_metrics()
        self._rotate_recent_checkpoints()
        return save_path
    
    def _rotate_recent_checkpoints(self):
        base_dir = self.logging_dir
        all_ckpts = sorted(
            [d for d in base_dir.glob("checkpoint-*") if d.name.split("-")[-1].isdigit() and not d.name.startswith("best_")],
            key=lambda x: int(x.name.split("-")[-1])
        )
        best_paths = {Path(info['path']) for info in self.best_metrics.values() if info.get('path')}
        recent_ckpts = [c for c in all_ckpts if c not in best_paths]
        if len(recent_ckpts) > self.keep_last_n:
            to_delete = recent_ckpts[:-self.keep_last_n]
            for d in to_delete:
                try: shutil.rmtree(d)
                except OSError as e: print(f"Warning: Failed to delete old checkpoint: {e}")

# Helpers for handling DDP wraps
def extract_conditioner(pretrainer_module):
    if hasattr(pretrainer_module, "module"):
        return pretrainer_module.module.conditioner
    return pretrainer_module.conditioner

# ==============================================================================
# 2. TRAINING STEPS
# ==============================================================================

def pretrain_step_v5(pretrainer, model, batch, optimizer, lr_scheduler, accelerator, config):
    """PHASE 1: Trains the Inverse Physics Stem to match Tokenizer DWT coefficients."""
    lr_imgs, hr_imgs = batch
    pretrainer.train()
    
    with accelerator.accumulate(pretrainer):
        # Generate Continuous Ground Truth states DIRECTLY from the raw image
        with torch.no_grad():
            gt_pyramid = []
            curr = hr_imgs # V5 directly transforms the image!
            
            for _ in range(model.tokenizer.num_scales):
                combined = model.tokenizer.dwt(curr)
                # V5 strictly separates the 1-channel LL and 3-channel details
                ll_gt = combined[:, 0:1]
                details_gt = combined[:, 1:4]
                
                gt_pyramid.append((ll_gt, details_gt))
                curr = ll_gt
                
            gt_pyramid = list(reversed(gt_pyramid))
            gt_root_ll = curr
            
        loss, metrics = pretrainer(lr_imgs, gt_pyramid, gt_root_ll)
        
        # Inf/NaN guard
        if not torch.isfinite(loss).all():
            optimizer.zero_grad()
            formatted_fallback = {f'pretrain/{k}': v for k, v in metrics.items()}
            formatted_fallback['pretrain/nan_batch'] = 1.0
            return loss.detach(), formatted_fallback
        
        accelerator.backward(loss)
        pretrain_grad_norm = None
        if accelerator.sync_gradients:
            pretrain_grad_norm = float(
                accelerator.clip_grad_norm_(
                    pretrainer.parameters(), config.training.compute.get("max_grad_norm", 1.0)
                )
            )
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        formatted_metrics = {f"pretrain/{k}": v for k, v in metrics.items()}
        if pretrain_grad_norm is not None:
            formatted_metrics["pretrain/grad_norm"] = pretrain_grad_norm
        
    return loss, formatted_metrics

def finetune_step_v4(model, conditioner, batch, optimizer, lr_scheduler, accelerator, config, global_step):
    """PHASE 3: Tokenizer Decoder Finetuning with Multi-Scale Regularization"""
    import random
    lr_imgs, hr_imgs = batch
    
    # 1. Stochastic Decoding: Randomize generation params so out_conv is robust
    dynamic_temp = random.uniform(0.7, 1.1)
    dynamic_top_k = random.choice([1, 10, 50, 0]) # 0 means no top-k
    
    # Freeze AR model and Conditioner for generation
    model.eval()
    conditioner.eval()
    
    with torch.no_grad():
        cond_outputs = conditioner(lr_imgs)
        # Generate tokens
        generated_tokens = model.generate(
            cond_outputs, 
            tokenizer=model.tokenizer,
            temp=dynamic_temp,
            top_k=dynamic_top_k
        )
        
    # 2. Decode using the UNFROZEN Tokenizer layers
    model.tokenizer.train() 
    
    # Get all intermediate latent states
    f_root = model.tokenizer.decode_root(generated_tokens['root'])                           # 6x6
    f_s0 = model.tokenizer.inverse_step(f_root, generated_tokens['scale_0'], scale_idx=0)    # 12x12
    f_s1 = model.tokenizer.inverse_step(f_s0, generated_tokens['scale_1'], scale_idx=1)      # 24x24
    f_s2 = model.tokenizer.inverse_step(f_s1, generated_tokens['scale_2'], scale_idx=2)      # 48x48
    
    loss = 0.0
    metrics = {}
    ft_config = config.finetuning
    
    # --- A. MULTI-SCALE INTERMEDIATE SUPERVISION ---
    total_intermediate_mse = 0.0
    for s_idx, f_latent in enumerate([f_root, f_s0, f_s1]):
        # V5: The latent IS the image geometry. Just clamp it for safety.
        inter_img = torch.clamp(f_latent, min=0.0, max=1.0)
        inter_target = F.interpolate(hr_imgs, size=inter_img.shape[2:], mode='area')
        total_intermediate_mse += F.mse_loss(inter_img, inter_target)
        
    # --- B. PRE/POST REFINER SPLIT (Final Scale) ---
    # use ret_prerefiner=True to get both the base wavelet recon and the refined recon
    generated_pixels, wavelet_recon = model.tokenizer.features_to_pixels(f_s2, ret_prerefiner=True)
    
    wavelet_mse = F.mse_loss(wavelet_recon, hr_imgs)
    recon_mse = F.mse_loss(generated_pixels, hr_imgs)
    
    # --- C. BALANCED LOSS COMPUTATION ---
    # Use the same powerful scaling factors from your pretraining script
    loss = (wavelet_mse * 125.0) + (recon_mse * 125.0) + (total_intermediate_mse * 25.0)
    
    # Optional SSIM addition for extra perceptual sharpness
    #if ft_config.get('lambda_ssim', 0.0) > 0:
    #    ssim_l = compute_ssim_loss(generated_pixels, hr_imgs)
    #    loss += ft_config.lambda_ssim * ssim_l
    #    metrics['finetune/ssim_loss'] = ssim_l.item()

    # Logging
    metrics['finetune/wavelet_mse'] = wavelet_mse.item()
    metrics['finetune/recon_mse'] = recon_mse.item()
    metrics['finetune/inter_mse'] = total_intermediate_mse.item()
    metrics['finetune/total_loss'] = loss.item()
    metrics['finetune/dynamic_temp'] = dynamic_temp

    # 3. Backprop & Step
    if not torch.isfinite(loss).all():
        optimizer.zero_grad()
        return loss.detach(), {'finetune/nan_batch': 1.0}, {}

    accelerator.backward(loss)

    finetune_grad_norm = None
    if accelerator.sync_gradients:
        ft_params = [p for p in model.tokenizer.parameters() if p.requires_grad]
        finetune_grad_norm = float(
            accelerator.clip_grad_norm_(ft_params, config.training.compute.get("max_grad_norm", 1.0))
        )

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    if finetune_grad_norm is not None:
        metrics["finetune/grad_norm"] = finetune_grad_norm

    return loss, metrics, {}


def train_step_v5(model, conditioner, batch, optimizer, lr_scheduler, accelerator, config, global_step):
    """PHASE 2: Main AR Generation Training."""
    lr_imgs, hr_imgs = batch
    model.train()
    
    if config.training.get("end_to_end", False):
        conditioner.train()
    else:
        conditioner.eval()
        
    with accelerator.accumulate(model):
        with torch.no_grad():
            # Get FSQ ground truth tokens: [B, 1, H, W]
            target_tokens = model.tokenizer.get_hierarchical_codes(hr_imgs)
        
        # Get SOTA spatial and global conditions
        cond_outputs = conditioner(lr_imgs)
        
        # Model forward pass -> {'root': logits, 'scale_0': logits, ...}
        outputs = model(cond_outputs, target_tokens, model.tokenizer)
        
        gen_diagnostics = outputs.pop('metrics', {})

        v5_sw = OmegaConf.select(config, "training.loss.v5_scale_weights", default=None)
        if v5_sw is not None:
            v5_sw = OmegaConf.to_container(v5_sw, resolve=True)
        gp = OmegaConf.select(config, "model.generator_v5.params", default=None)
        tl = OmegaConf.select(config, "training.loss", default=None)
        use_ce_w = True
        void_ce = 0.5
        if tl is not None and "use_ce_weighting" in tl:
            use_ce_w = bool(tl.use_ce_weighting)
        if tl is not None and "ce_void_class_weight" in tl:
            void_ce = float(tl.ce_void_class_weight)
        if gp is not None:
            if "use_ce_weighting" in gp:
                use_ce_w = bool(gp.use_ce_weighting)
            if "ce_void_class_weight" in gp:
                void_ce = float(gp.ce_void_class_weight)

        is_vq_tok = _is_lidar_vq_tokenizer(model.tokenizer)
        if is_vq_tok and use_ce_w:
            raise ValueError(
                "VQ tokenizer has no FSQ void index. Set training.loss.use_ce_weighting: false for dense CE "
                "(recommended), or implement an explicit void-index scheme and pass zero_indices_dict from config."
            )
        _vq_zero_dict = None
        _tok_for_ce = model.tokenizer
        if is_vq_tok:
            # FSQ null_indices_dict_from_tokenizer requires null_idx_*; use uniform CE + dummy void index 0 for metrics only.
            _vq_zero_dict = {"root": 0, "scale_0": 0, "scale_1": 0, "scale_2": 0}
            _tok_for_ce = None

        loss, metrics = compute_multi_scale_loss_v5(
            outputs,
            target_tokens,
            tokenizer=_tok_for_ce,
            zero_indices_dict=_vq_zero_dict,
            scale_loss_weights=v5_sw,
            use_ce_weighting=use_ce_w,
            void_class_weight=void_ce,
        )
        if not torch.isfinite(loss).all():
            optimizer.zero_grad()
            return loss.detach(), {'train/nan_batch': 1.0}, {}
        
        accelerator.backward(loss)

        train_grad_norm = None
        if accelerator.sync_gradients:
            # clip_grad_norm_ returns total L2 norm *before* clipping; log that (non-zero vs get_grad_norm after zero_grad).
            train_grad_norm = float(
                accelerator.clip_grad_norm_(model.parameters(), config.training.compute.get("max_grad_norm", 1.0))
            )

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if train_grad_norm is not None:
            metrics["train/grad_norm"] = train_grad_norm
        for k, v in gen_diagnostics.items():
            metrics[f'train/diag_{k}'] = v

    return loss, metrics, {}

# ==============================================================================
# 3. VALIDATION LOOP
# ==============================================================================
_val_lpips_fn = None


def _get_validation_lpips(device):
    """Lazy LPIPS (VGG) for validation; returns None if ``lpips`` is not installed."""
    global _val_lpips_fn
    try:
        import lpips as _lpips_mod
    except ImportError:
        return None
    if _val_lpips_fn is None:
        _val_lpips_fn = _lpips_mod.LPIPS(net="vgg").eval()
        for p in _val_lpips_fn.parameters():
            p.requires_grad = False
    _val_lpips_fn = _val_lpips_fn.to(device)
    return _val_lpips_fn


def _compute_ssim_batch(img1, img2, window_size=11):
    """SSIM mean over a batch; ``img1``/``img2`` are ``[N, 1, H, W]`` on the same device."""
    padding = window_size // 2
    sigma = 1.5
    gauss = torch.tensor(
        [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)],
        device=img1.device,
        dtype=img1.dtype,
    )
    gauss = gauss / gauss.sum()
    kernel_2d = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    mu1 = F.conv2d(img1, kernel_2d, padding=padding)
    mu2 = F.conv2d(img2, kernel_2d, padding=padding)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel_2d, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel_2d, padding=padding) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel_2d, padding=padding) - mu1_mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


@torch.no_grad()
def validation_loop_v5(model, conditioner, val_loader, accelerator, config, global_step, val_pct=0.10):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    val_loss_m = AverageMeter()
    val_mse_m = AverageMeter()
    val_ssim_m = AverageMeter()
    val_lpips_m = AverageMeter()
    lpips_fn = _get_validation_lpips(accelerator.device)
    dynamic_metrics_meters = {}
    
    limit = max(1, int(len(val_loader) * val_pct))
    gen_params = config.model.generator_v5.params
    viz_samples = [] 
    
    pbar = tqdm(enumerate(val_loader), total=limit, desc="Validating", disable=not accelerator.is_main_process)
    
    for i, batch in pbar:
        if i >= limit: break
        lr_imgs, hr_imgs = batch
        
        # 1. GENERATION 
        cond_outputs = conditioner(lr_imgs)
        generated_tokens = unwrapped_model.generate(
            cond_outputs,
            tokenizer=unwrapped_model.tokenizer,
            temp=float(gen_params.get("temperature", 1.0)),
            top_k=int(gen_params.get("top_k", 50)),
            top_p=float(gen_params.get("top_p", 0.9)),
            cfg_scale=float(gen_params.get("cfg_scale", 1.0)),
        )
        
        # Metrics handling
        batch_gen_metrics = generated_tokens.pop('metrics', {})
        for key, val in batch_gen_metrics.items():
            if key not in dynamic_metrics_meters:
                dynamic_metrics_meters[key] = AverageMeter()
            dynamic_metrics_meters[key].update(val, lr_imgs.size(0))
            
        # 2. RECONSTRUCTION
        f_root = unwrapped_model.tokenizer.decode_root(generated_tokens['root'])
        f_s0 = unwrapped_model.tokenizer.inverse_step(f_root, generated_tokens['scale_0'], scale_idx=0)
        f_s1 = unwrapped_model.tokenizer.inverse_step(f_s0, generated_tokens['scale_1'], scale_idx=1)
        f_s2 = unwrapped_model.tokenizer.inverse_step(f_s1, generated_tokens['scale_2'], scale_idx=2)
        generated_pixels = unwrapped_model.tokenizer.features_to_pixels(f_s2)
        
        # Standard L1 loss
        loss = F.l1_loss(generated_pixels, hr_imgs)
        val_loss_m.update(loss.item(), lr_imgs.size(0))

        sr = torch.clamp(generated_pixels, 0.0, 1.0).float()
        hr = torch.clamp(hr_imgs, 0.0, 1.0).float()
        val_mse_m.update(F.mse_loss(sr, hr).item(), lr_imgs.size(0))
        val_ssim_m.update(_compute_ssim_batch(sr, hr).item(), lr_imgs.size(0))
        if lpips_fn is not None:
            sr3 = (sr * 2.0 - 1.0).repeat(1, 3, 1, 1)
            hr3 = (hr * 2.0 - 1.0).repeat(1, 3, 1, 1)
            val_lpips_m.update(lpips_fn(sr3, hr3).mean().item(), lr_imgs.size(0))
        
        # 3. VISUALIZATION
        if accelerator.is_main_process and len(viz_samples) < 4:
            B = hr_imgs.shape[0]
            for b_idx in range(B):
                if len(viz_samples) >= 4: break
                hr_img_single = hr_imgs[b_idx:b_idx+1]
                if hr_img_single.var().item() < 1e-6: continue
                
                # Ground truth tokenizer reconstruction for comparison
                gt_indices = unwrapped_model.tokenizer.get_hierarchical_codes(hr_img_single)
                tf_root = unwrapped_model.tokenizer.decode_root(gt_indices['root'])
                tf_s0 = unwrapped_model.tokenizer.inverse_step(tf_root, gt_indices['scale_0'], scale_idx=0)
                tf_s1 = unwrapped_model.tokenizer.inverse_step(tf_s0, gt_indices['scale_1'], scale_idx=1)
                tf_s2 = unwrapped_model.tokenizer.inverse_step(tf_s1, gt_indices['scale_2'], scale_idx=2)
                tokenizer_recon = unwrapped_model.tokenizer.features_to_pixels(tf_s2)
                
                viz_samples.append((hr_img_single, tokenizer_recon, generated_pixels[b_idx:b_idx+1]))

    visuals = {}
    if accelerator.is_main_process and len(viz_samples) > 0:
        def normalize_vis(img):
            img = img.float()
            B = img.shape[0]
            normalized = []
            for b in range(B):
                img_b = img[b]
                min_v, max_v = img_b.min(), img_b.max()
                norm_b = (img_b - min_v) / (max_v - min_v) if max_v - min_v > 1e-5 else img_b
                normalized.append(norm_b)
            return torch.stack(normalized, dim=0)
        
        normalized_samples = []
        for hr_img, tokenizer_recon, gen_px in viz_samples:
            viz_row = torch.cat([normalize_vis(hr_img), normalize_vis(tokenizer_recon), normalize_vis(gen_px)], dim=-1)
            normalized_samples.append(viz_row)
        
        viz_grid = torch.cat(normalized_samples, dim=2) if len(normalized_samples) > 1 else normalized_samples[0]
        # Keep single-channel tensors; log_visuals maps them with viridis for WandB/TensorBoard.

        first_hr, first_tokenizer, first_gen = viz_samples[0]
        viz_gt_first, viz_tokenizer_first, viz_generated_first = normalize_vis(first_hr), normalize_vis(first_tokenizer), normalize_vis(first_gen)
        
        visuals = {
            "viz/validation_grid": viz_grid,
            "viz/ground_truth_images": viz_gt_first,
            "viz/tokenizer_image": viz_tokenizer_first,
            "viz/generated_image": viz_generated_first,
        }
    
    # Final metric aggregation
    eps = 1e-10
    avg_metrics = {
        "val/total_loss": val_loss_m.avg,
        "val/mse": val_mse_m.avg,
        "val/psnr": 10.0 * math.log10(1.0 / (val_mse_m.avg + eps)),
        "val/ssim": val_ssim_m.avg,
    }
    if lpips_fn is not None and val_lpips_m.count > 0:
        avg_metrics["val/lpips"] = val_lpips_m.avg
    for key, meter in dynamic_metrics_meters.items():
        avg_metrics[f"val/{key}"] = meter.avg

    model.train()
    return avg_metrics, visuals

# ==============================================================================
# 4. BUILD MODEL
# ==============================================================================

def infer_vocab_sizes_from_tokenizer(tokenizer, layout: str) -> dict:
    """FSQ vocabulary sizes for ``HierarchicalGeneratorUnified`` (matches null / codebook layout)."""
    layout = _normalize_tokenizer_layout(layout)
    root = int(tokenizer.fsq_root.vocab_size)
    if layout == "joint":
        det = int(tokenizer.fsq_details.vocab_size)
        return {"root": root, "scale_0": det, "scale_1": det, "scale_2": det}
    lst = tokenizer.fsq_details_list
    if len(lst) < 3:
        raise ValueError(f"tokenizer_layout=separate expects 3 scales in fsq_details_list, got {len(lst)}")
    return {
        "root": root,
        "scale_0": int(lst[0].vocab_size),
        "scale_1": int(lst[1].vocab_size),
        "scale_2": int(lst[2].vocab_size),
    }


def _resolve_use_null_routing(gen_params, tp) -> bool:
    """Default True when unset (matches historical SSM configs)."""
    if gen_params is not None and "use_null_routing" in gen_params:
        return bool(gen_params.use_null_routing)
    if tp is not None and "use_null_routing" in tp:
        return bool(tp.use_null_routing)
    return True


_VQ_TOKENIZER_CLASS_NAMES = frozenset(
    {
        "LidarTokenizerVQ",
        "LidarTokenizerNoLL",
        "LidarTokenizerIndepVQ",
        "LidarTokenizerVanillaVQ",
    }
)


def _tokenizer_class_suffix(tok_target: str) -> str:
    t = (tok_target or "").strip()
    return t.rsplit(".", 1)[-1] if t else ""


def _tokenizer_target_is_vq_ssm(tok_target: str) -> bool:
    return _tokenizer_class_suffix(tok_target) in _VQ_TOKENIZER_CLASS_NAMES


def _import_tokenizer_class_from_target(tok_target: str) -> type:
    """
    Retrieves the correct tokenizer class from the GeM-SR codebase based on the target string's class name,
    ignoring any legacy 'gess' paths that might be present in old config files.
    """
    suffix = _tokenizer_class_suffix(tok_target)
    
    if suffix == "LidarTokenizerVQ":
        from src.modeling.tokenizer.tokenizer_vq import LidarTokenizerVQ
        return LidarTokenizerVQ
    elif suffix == "LidarTokenizerIndepVQ":
        from src.modeling.tokenizer.tokenizer_indep import LidarTokenizerIndepVQ
        return LidarTokenizerIndepVQ
    elif suffix == "LidarTokenizerVanillaVQ":
        from src.modeling.tokenizer.tokenizer_vanilla import LidarTokenizerVanillaVQ
        return LidarTokenizerVanillaVQ
    elif suffix == "LidarTokenizerNoLL":
        from src.modeling.tokenizer.tokenizer_no_ll import LidarTokenizerNoLL
        return LidarTokenizerNoLL
    elif suffix in ["LidarTokenizerV5", "LidarTokenizerV6", "FSQ"]:
        from src.modeling.tokenizer.tokenizer_fsq import LidarTokenizerV5
        return LidarTokenizerV5
    else:
        raise ValueError(f"Unknown tokenizer class in target: {tok_target}")


def _is_lidar_vq_tokenizer(tok: torch.nn.Module) -> bool:
    """True for Haar Euclidean-VQ tokenizer family (base + ablations)."""
    return getattr(tok, "vq_root", None) is not None


def build_model_v5(ssm_config, device='cuda', is_resume=False):
    if ssm_config.model.tokenizer.pretrained.path_to_config:
        print(f"  -> Loading from pretrained tokenizer path: {ssm_config.model.tokenizer.pretrained.path_to_config}")
        tokenizer_cfg = ssm_config.model.tokenizer.pretrained.path_to_config
        _tok_cfg_loaded = OmegaConf.load(tokenizer_cfg)
        vqvae_params = _tok_cfg_loaded.model.tokenizer.params
        tokenizer_target = str(OmegaConf.select(_tok_cfg_loaded.model.tokenizer, "target", default=""))
    else:
        print(f"  -> Loading from ssm tokenizer's params")
        tokenizer_cfg = ssm_config.model.tokenizer
        vqvae_params = tokenizer_cfg.params
        tokenizer_target = str(OmegaConf.select(ssm_config.model.tokenizer, "target", default=""))

    gen_params_early = ssm_config.model.generator_v5.params
    tp_cfg = OmegaConf.select(ssm_config, "training.pretraining", default=None)
    tokenizer_layout = _normalize_tokenizer_layout(gen_params_early.get("tokenizer_layout", "separate"))

    tokenizer_kwargs = {
        'input_channels': vqvae_params.input_channels,
        #'projected_dim': vqvae_params.projected_dim,
        'latent_dim': vqvae_params.latent_dim,
        'hidden_dim': vqvae_params.hidden_dim,
        'num_scales': vqvae_params.num_scales,
        #'rq_depths': vqvae_params.rq_depths,
        #'vbr_threshold': 0.0, # vqvae_params.vbr.threshold,
        #'vbr_ratio': 0.0, #,vqvae_params.vbr.ratio,
        'inter_loss_weight': getattr(vqvae_params, "inter_loss_weight", 2.0)
        #'sparsity_weight': vqvae_params.vbr.sparsity_weight,
    }
    if tokenizer_layout == "joint":
        fl = getattr(vqvae_params, "fsq_levels", None)
        if fl is not None:
            tokenizer_kwargs["fsq_levels"] = list(OmegaConf.to_container(fl, resolve=True))
        # Tokenizer architecture / tokenizer-only flags: **vqvae_params only** (tokenizer training YAML
        # when ``path_to_config`` is set, else ``model.tokenizer.params`` on the SSM yaml). SSM must not
        # override via ``generator_v5.params`` or duplicate keys under ``model.tokenizer.params``.

        tokenizer_kwargs["use_refiner"] = bool(OmegaConf.select(vqvae_params, "use_refiner", default=True))

        def _pick_tok_bool(key: str):
            if vqvae_params is not None and key in vqvae_params:
                return bool(vqvae_params[key])
            if key == "use_null_routing" and tp_cfg is not None and "use_null_routing" in tp_cfg:
                return bool(tp_cfg.use_null_routing)
            return None

        def _pick_tok_cfg_bool(key: str, default: bool = False) -> bool:
            if vqvae_params is not None and key in vqvae_params:
                return bool(vqvae_params[key])
            return default

        tokenizer_kwargs["layer_cfg"] = TokenizerV5LayerConfig(
            conv_bias=_pick_tok_cfg_bool("tokenizer_conv_bias", False),
            use_zero_preserving_resnorm=not _pick_tok_cfg_bool("tokenizer_standard_blocks", False),
        )
        _tok_src = (
            "pretrained tokenizer YAML (path_to_config)"
            if ssm_config.model.tokenizer.pretrained.path_to_config
            else "SSM model.tokenizer.params (no path_to_config)"
        )
        _lc = tokenizer_kwargs["layer_cfg"]
        print(
            f"  -> [tokenizer build] {_tok_src} only (no SSM overrides): use_refiner={tokenizer_kwargs['use_refiner']}, "
            f"conv_bias={_lc.conv_bias}, standard_blocks={not _lc.use_zero_preserving_resnorm}"
        )

        if _tokenizer_target_is_vq_ssm(tokenizer_target):
            vq_kwargs = {
                "input_channels": int(OmegaConf.select(vqvae_params, "input_channels", default=1)),
                "latent_dim": int(OmegaConf.select(vqvae_params, "latent_dim", default=4)),
                "hidden_dim": int(OmegaConf.select(vqvae_params, "hidden_dim", default=64)),
                "num_scales": int(OmegaConf.select(vqvae_params, "num_scales", default=3)),
                "codebook_size_root": int(OmegaConf.select(vqvae_params, "codebook_size_root", default=175)),
                "codebook_size_detail": int(OmegaConf.select(vqvae_params, "codebook_size_detail", default=512)),
                "commitment_cost": float(OmegaConf.select(vqvae_params, "commitment_cost", default=0.25)),
                "vq_loss_weight": float(OmegaConf.select(vqvae_params, "vq_loss_weight", default=1.0)),
                "vq_use_tanh_bound": bool(OmegaConf.select(vqvae_params, "vq_use_tanh_bound", default=True)),
                "inter_loss_weight": float(OmegaConf.select(vqvae_params, "inter_loss_weight", default=2.0)),
                "use_refiner": tokenizer_kwargs.get("use_refiner", bool(getattr(vqvae_params, "use_refiner", False))),
                "apply_sparsity_mask_to_latents": bool(
                    OmegaConf.select(vqvae_params, "apply_sparsity_mask_to_latents", default=True)
                ),
                "layer_cfg": tokenizer_kwargs["layer_cfg"],
            }
            _tok_cls = _import_tokenizer_class_from_target(tokenizer_target)
            print(
                f"  -> Tokenizer: {_tok_cls.__name__} (joint / learned VQ), layout={tokenizer_layout}, "
                f"target={tokenizer_target!r}"
            )
            tokenizer_model = _tok_cls(**vq_kwargs).to(device)
        else:
            for _ablation_key in (
                "apply_sparsity_mask_to_latents",
                "use_null_token_routing",
                "use_null_routing",
            ):
                vb = _pick_tok_bool(_ablation_key)
                if vb is not None:
                    tokenizer_kwargs[_ablation_key] = vb
            print(f"  -> Tokenizer: LidarTokenizerV5 (joint / shared detail FSQ), layout={tokenizer_layout}")
            tokenizer_model = LidarTokenizerV5Joint(**tokenizer_kwargs).to(device)
    else:
        raise NotImplementedError("Separate tokenizer layout (LidarTokenizerV6) is deprecated in GeM-SR.")
    
    if not is_resume and ssm_config.model.tokenizer.pretrained.path_to_ckpt:
        ckpt_path = ssm_config.model.tokenizer.pretrained.path_to_ckpt
        print(f"--> [INIT] Loading Pretrained Tokenizer V4: {ckpt_path}")
        if ckpt_path.endswith(".safetensors") and HAS_SAFETENSORS:
            tok_state = load_file(ckpt_path)
            tokenizer_model.load_state_dict(tok_state, strict=False)
        else:
            state = torch.load(ckpt_path, map_location='cpu')
            tokenizer_model.load_state_dict(state['state_dict'] if 'state_dict' in state else state, strict=False)
    
    tokenizer_model.eval()
    
    gen_params = ssm_config.model.generator_v5.params
    layout = _normalize_tokenizer_layout(gen_params.get("tokenizer_layout", tokenizer_layout))

    vs_from_tok = infer_vocab_sizes_from_tokenizer(tokenizer_model, layout)
    yaml_vs = gen_params.get("vocab_sizes")
    if yaml_vs is not None and not isinstance(yaml_vs, dict):
        yaml_vs = OmegaConf.to_container(yaml_vs, resolve=True)
    vocab_sizes_resolved = dict(vs_from_tok)
    if yaml_vs:
        for k, v in yaml_vs.items():
            if k in vocab_sizes_resolved:
                vocab_sizes_resolved[k] = int(v)

    bb = gen_params.get("backbone")
    if bb is None or str(bb).strip() == "":
        backbone = "convnext" if gen_params.get("fast", True) else "transformer"
    else:
        backbone = str(bb).lower().strip()

    gen_target = OmegaConf.select(ssm_config.model.generator_v5, "target", default="")
    gen_target_l = str(gen_target).lower()
    use_paper_generator = (
        bool(gen_params.get("paper_ablation", False))
        or "hierarchial_generator_paper" in gen_target_l
    )

    zero_idx_model = gen_params.get("zero_idx")
    if zero_idx_model is None and hasattr(tokenizer_model, "null_idx_root"):
        _nr = tokenizer_model.null_idx_root
        zero_idx_model = int(_nr.item()) if isinstance(_nr, torch.Tensor) else int(_nr)
    if zero_idx_model is None and _is_lidar_vq_tokenizer(tokenizer_model):
        # No FSQ void token; generator heads still expect an int (bias init / diagnostics). Override in YAML if needed.
        zero_idx_model = int(gen_params.get("vq_generator_zero_idx", 0))

    rq_depths = int(gen_params.get("rq_depths", 1))

    model = HierarchicalGenerator(
        tokenizer_dim=1,
        embed_dim=gen_params.embed_dim,
        vocab_size=gen_params.get("vocab_size", 2401),
        rq_depths=rq_depths,
        zero_idx=zero_idx_model,
        root_depth=gen_params.get("root_depth", 2),
        s0_depth=gen_params.get("s0_depth", 2),
        s1_depth=gen_params.get("s1_depth", 4),
        s2_depth=gen_params.get("s2_depth", 6),
        mlp_ratio=gen_params.get("mlp_ratio", 2),
        heads=gen_params.get("heads", 8),
        backbone=backbone,
        cfg_training=bool(gen_params.get("cfg_training", False)),
        cond_dropout_prob=float(gen_params.get("cond_dropout_prob", 0.1)),
        scheduled_sampling_prob=gen_params.get("scheduled_sampling_prob", 0.0),
        vocab_sizes=vocab_sizes_resolved,
        paper_layer_cfg=PaperConvNeXtLayerConfig(
            conv_bias=bool(gen_params.get("conv_bias", False)),
            linear_bias=bool(gen_params.get("linear_bias", False)),
        ),
    ).to(device)

    
    
    model.tokenizer = tokenizer_model
    if hasattr(tokenizer_model, "null_idx_root"):
        _nr = tokenizer_model.null_idx_root
        model.zero_idx = int(_nr.item()) if isinstance(_nr, torch.Tensor) else int(_nr)
    elif _is_lidar_vq_tokenizer(tokenizer_model):
        model.zero_idx = int(zero_idx_model)

    # Initialize SOTA Conditioner (conv_bias default False = original ckpts)
    _cond_layer_cfg = ConditionerLayerConfig(
        conv_bias=bool(gen_params.get("conditioner_conv_bias", False)),
    )
    conditioner = SuperResWaveletConditionerV4(
        in_channels=1, embed_dim=gen_params.embed_dim, layer_cfg=_cond_layer_cfg
    ).to(device)

    def _pick_pretrain_float(key: str, default: float) -> float:
        if gen_params is not None and key in gen_params:
            return float(gen_params[key])
        if tp_cfg is not None and key in tp_cfg:
            return float(tp_cfg[key])
        return default

    use_null_routing_eff = _resolve_use_null_routing(gen_params, tp_cfg)

    _explicit_pretrain_w = None
    if gen_params is not None and "use_sparsity_weighted_pretrain_l1" in gen_params:
        _explicit_pretrain_w = gen_params["use_sparsity_weighted_pretrain_l1"]
    elif tp_cfg is not None and "use_sparsity_weighted_pretrain_l1" in tp_cfg:
        _explicit_pretrain_w = tp_cfg["use_sparsity_weighted_pretrain_l1"]
    if _explicit_pretrain_w is None:
        use_pretrain_weighted = bool(use_null_routing_eff)
    else:
        use_pretrain_weighted = bool(_explicit_pretrain_w)

    pretrain_em = _pick_pretrain_float("pretrain_edge_weight_multiplier", 10.0)
    pretrain_thr = _pick_pretrain_float("pretrain_active_threshold", 0.05)
    # CRITICAL: Pass tokenizer_dim=1 so the pretrainer maps to [B, 1, H, W] targets
    pretrainer = ConditionerWaveletPretrainer(
        conditioner,
        embed_dim=gen_params.embed_dim,
        tokenizer_dim=1,
        use_sparsity_weighted_l1=use_pretrain_weighted,
        edge_weight_multiplier=pretrain_em,
        active_threshold=pretrain_thr,
        layer_cfg=_cond_layer_cfg,
    ).to(device)
    
    return model, pretrainer


# ==============================================================================
# 5. MAIN FUNCTION
# ==============================================================================

def main():
    print(f"[Received Training Script] Starting Training Generative SSM V4.")
    
    # ==============================================================================
    # 1. CONFIGURATION & ENVIRONMENT SETUP
    # ==============================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--run", type=str, required=True, help="Run ID/Number")
    args, unknown = parser.parse_known_args()
    
    yaml_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(yaml_conf, cli_conf)
    config.run = args.run 
    set_seed(config.globals.seed)
    

    # dynamic root dir for multi-computer usage (local, compute cluster, etc)
    root_dir = None
    env_scratch = os.environ.get("MY_SCRATCH")
    if env_scratch and Path(env_scratch).exists():
        root_dir = Path(env_scratch)
        print(f"🚀 Priority Path Found: Using LOCAL SCRATCH at {root_dir}")
    else:
        for path in config.dataset.infra.get("root_dir", []):
            if Path(path).exists():
                root_dir = Path(path)
                break
    if root_dir is None: raise RuntimeError("❌ No valid root_dir found! Check your rsync or YAML.")
    config.dataset.infra.root_dir = str(root_dir)
    print(f" Using data from: {config.dataset.infra.root_dir}")
    # ==============================================================================
    # 2. MODE DETECTION & PATH ROUTING
    # ==============================================================================
    # Default new runs under GeSS; set WORKSPACE / WORKSPACE_LEGACY to override or resume Lidar-SSM trees.
    """
    workspace = Path(os.environ.get("WORKSPACE", "./outputs/ssm_runs"))
    legacy_workspace = Path(
        os.environ.get("WORKSPACE_LEGACY", "./outputs/ssm_runs")
    )
    """
    workspace = Path(os.environ.get("WORKSPACE", "./outputs/ssm_runs"))
    legacy_workspace = Path(
        os.environ.get("WORKSPACE_LEGACY", "./outputs/ssm_runs")
    )
    unique_run_id = f"{config.experiment.run_name}_{config.run}"
    rel_run = Path("outputs") / config.experiment.project / config.experiment.name / unique_run_id
    base_output_dir = workspace / rel_run
    base_logging_dir = base_output_dir / "logs"
    legacy_logging_dir = (legacy_workspace / rel_run / "logs") if legacy_workspace != workspace else None
    if legacy_logging_dir is not None and str(legacy_logging_dir.resolve()) == str(base_logging_dir.resolve()):
        legacy_logging_dir = None

    is_finetuning = config.get("finetuning", {}).get("active", False)
    resume_path, resume_step = None, 0
    is_auto_resuming = False

    if is_finetuning:
        # Determine exactly which checkpoint to load for finetuning
        ft_target = str(config.finetuning.get("checkpoint", ""))
        if ft_target.isdigit():
            resume_path = base_logging_dir / f"checkpoint-{ft_target}"
            if not resume_path.exists() and legacy_logging_dir is not None:
                alt = legacy_logging_dir / f"checkpoint-{ft_target}"
                if alt.exists():
                    resume_path = alt
        elif ft_target:
            resume_path = Path(ft_target)
        else:
            resume_path, _ = get_latest_checkpoint_across_dirs(
                [base_logging_dir, legacy_logging_dir] if legacy_logging_dir else [base_logging_dir]
            )

        if not resume_path or not resume_path.exists():
            raise FileNotFoundError(f"❌ Finetuning checkpoint not found at: {resume_path}")
        # Redirect outputs to a new folder to preserve Phase 2 models
        output_dir = base_output_dir.with_name(f"{unique_run_id}_FINETUNE")
        unique_run_id += "_FINETUNE"
        print(f"🎯 MODE: FINETUNING (Loading weights from {resume_path})")
   
    else:
        # Standard training: only discover/load training checkpoints when experiment.resume is True.
        # Otherwise weights would load from latest checkpoint but global_step stayed 0 (misleading "FRESH TRAIN").
        output_dir = base_output_dir
        if config.experiment.resume:
            resume_path, resume_step = get_latest_checkpoint_across_dirs(
                [base_logging_dir, legacy_logging_dir] if legacy_logging_dir else [base_logging_dir]
            )
        else:
            resume_path, resume_step = None, 0
        is_auto_resuming = resume_path is not None
        if is_auto_resuming:
            from_legacy = (
                legacy_logging_dir is not None
                and resume_path is not None
                and resume_path.parent.resolve() == legacy_logging_dir.resolve()
            )
            src = "legacy (WORKSPACE_LEGACY)" if from_legacy else "primary (WORKSPACE)"
            print(f"🔄 MODE: AUTO-RESUME from {src}: step {resume_step} @ {resume_path}")
        elif config.experiment.resume:
            print(
                "🌱 MODE: FRESH TRAIN (experiment.resume=True but no checkpoint-* under this run's logs/; "
                "training from tokenizer init only)"
            )
        else:
            print("🌱 MODE: FRESH TRAIN (experiment.resume=False; not loading training checkpoints)")

    cond_steps = int(OmegaConf.select(config, "training.pretraining.cond_steps", default=0) or 0)
    pre_raw = OmegaConf.select(config, "training.pretraining.pretrained_cond_path", default=None)
    pretrained_cond_dir = None
    if not is_finetuning and not is_auto_resuming:
        pretrained_cond_dir = resolve_pretrained_cond_checkpoint(
            pre_raw, workspace, legacy_workspace, base_logging_dir, legacy_logging_dir
        )
        s_pre = str(pre_raw or "").strip()
        if s_pre and s_pre.lower() not in ("null", "none", "~") and pretrained_cond_dir is None:
            raise FileNotFoundError(
                f"training.pretraining.pretrained_cond_path={pre_raw!r} did not resolve to a directory "
                f"with model_1.safetensors. Tried: literal path, checkpoint-<step> under this run's logs "
                f"(WORKSPACE then WORKSPACE_LEGACY), and relative paths under WORKSPACE / WORKSPACE_LEGACY.\n"
                f"  WORKSPACE={workspace}\n  WORKSPACE_LEGACY={legacy_workspace}"
            )

    # Set up final paths
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.to_container(config, resolve=True), f=output_dir / "config.yaml")
    config.experiment.logging_dir = str(logging_dir)
    config.experiment.output_dir = str(output_dir)

    # ==============================================================================
    # 3. ACCELERATOR & MODEL INITIALIZATION
    # ==============================================================================
    acc_config = ProjectConfiguration(project_dir=str(logging_dir), logging_dir=str(logging_dir))
    log_with = "wandb" if config.experiment.get("logger") == "wandb" else "tensorboard"
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.compute.grad_accum_steps,
        mixed_precision=config.training.compute.mixed_precision,
        log_with=log_with,
        project_config=acc_config
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.experiment.project, 
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": {"name": unique_run_id, "group": config.experiment.name}} if log_with == "wandb" else {}
        )
    
    accelerator.print(f"\n[PATHS] Data: {root_dir} | Output: {output_dir}")

    # Build model (passing True for is_resume skips downloading the raw tokenizer again)
    model, pretrainer = build_model_v5(config, device=accelerator.device, is_resume=(is_auto_resuming or is_finetuning))

    # Load Model Weights
    skipped_cond_pretrain = False
    if resume_path is not None:
        chkpt_file = resume_path / "model.safetensors"
        cond_chkpt_file = resume_path / "model_1.safetensors"  # Fixed to model_1 with underscore
        
        # 1. Load Main Model
        if chkpt_file.exists() and HAS_SAFETENSORS:
            accelerator.print(f"  > 📥 Loading safe weights from {chkpt_file}")
            state = load_file(chkpt_file) if chkpt_file.name.endswith('.safetensors') else torch.load(chkpt_file, map_location='cpu')
            if 'state_dict' in state: state = state['state_dict']
            if 'model' in state: state = state['model']
            
            # Filter to avoid loading mismatched DDP module strings
            model_sd = model.state_dict()
            filtered_state = {k.replace("module.", ""): v for k, v in state.items() if k.replace("module.", "") in model_sd or k in model_sd}
            model.load_state_dict(filtered_state, strict=False)
            accelerator.print(f"     > Loaded model weights")
            
        # 2. Load Conditioner (Pretrainer)
        if cond_chkpt_file.exists() and HAS_SAFETENSORS:
            accelerator.print(f"  > 📥 Loading safe weights from {cond_chkpt_file}")
            # FIXED: Load using cond_chkpt_file, not chkpt_file
            cond_state = load_file(cond_chkpt_file) if cond_chkpt_file.name.endswith('.safetensors') else torch.load(cond_chkpt_file, map_location='cpu')
            if 'state_dict' in cond_state: cond_state = cond_state['state_dict']
            if 'model' in cond_state: cond_state = cond_state['model']
            
            cond_sd = pretrainer.state_dict()
            filtered_state_cond = {k.replace("module.", ""): v for k, v in cond_state.items() if k.replace("module.", "") in cond_sd or k in cond_sd}
            pretrainer.load_state_dict(filtered_state_cond, strict=False)
            accelerator.print(f"     > Loaded conditioner weights")
        else:
            accelerator.print(f"  > ⚠️ WARNING: Conditioner weights not found at {cond_chkpt_file}")
    elif pretrained_cond_dir is not None:
        cond_chkpt_file = pretrained_cond_dir / "model_1.safetensors"
        if not HAS_SAFETENSORS:
            raise RuntimeError("safetensors is required to load pretrained_cond_path (install safetensors).")
        if not cond_chkpt_file.is_file():
            raise FileNotFoundError(f"pretrained conditioner missing: {cond_chkpt_file}")
        accelerator.print(f"  > 📥 Pretrained conditioner (skip phase 1): {cond_chkpt_file}")
        cond_state = load_file(cond_chkpt_file)
        if "state_dict" in cond_state:
            cond_state = cond_state["state_dict"]
        if "model" in cond_state:
            cond_state = cond_state["model"]
        cond_sd = pretrainer.state_dict()
        filtered_state_cond = {
            k.replace("module.", ""): v
            for k, v in cond_state.items()
            if k.replace("module.", "") in cond_sd or k in cond_sd
        }
        pretrainer.load_state_dict(filtered_state_cond, strict=False)
        accelerator.print("     > Loaded conditioner weights; generative training starts at cond_steps (no phase-1 train).")
        skipped_cond_pretrain = True

    # ==============================================================================
    # 4. FREEZING & OPTIMIZER SETUP
    # ==============================================================================
    if is_finetuning:
        accelerator.print("\n[LOCK] Freezing SSM & Conditioner. Unfreezing Tokenizer layers...")
        for param in model.parameters(): param.requires_grad = False
        for param in pretrainer.parameters(): param.requires_grad = False
        
        layers_to_unfreeze = config.finetuning.get("layers", ["out_conv"])
        if config.finetuning.get("refiner_finetune", False) and "refiner" not in layers_to_unfreeze:
            layers_to_unfreeze.append("refiner")
            
        for layer_name in layers_to_unfreeze:
            if hasattr(model.tokenizer, layer_name):
                layer = getattr(model.tokenizer, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
                accelerator.print(f"  -> ✅ Unfrozen: tokenizer.{layer_name}")
                
        _betas = tuple(OmegaConf.select(config, "optimizer.params.betas", default=[0.9, 0.95]))
        _eps = float(OmegaConf.select(config, "optimizer.params.eps", default=1e-8))
        _wd = float(OmegaConf.select(config, "optimizer.params.weight_decay", default=0.05))
        pairs = []
        for layer_name in layers_to_unfreeze:
            if hasattr(model.tokenizer, layer_name):
                layer = getattr(model.tokenizer, layer_name)
                for n, p in layer.named_parameters():
                    pairs.append((f"{layer_name}.{n}", p))
        ft_groups = adamw_param_groups_from_named_pairs(pairs, lr=float(config.optimizer.params.lr), weight_decay=_wd)
        optimizer = AdamW(ft_groups, betas=_betas, eps=_eps)
        
        target_steps = config.training.duration.max_steps + config.finetuning.get("finetuning_steps", 5000)
        global_step = config.training.duration.max_steps # Start logically where phase 2 ended
        
    else:
        accelerator.print("\n[LOCK] Freezing Tokenizer. Preparing standard Phase 1/2 Optimizer...")
        model.tokenizer.eval()
        for param in model.tokenizer.parameters(): param.requires_grad = False

        end_to_end = config.training.get("end_to_end", False)
        cond_lr_scale = config.training.get("cond_lr_scale", 0.1) if end_to_end else 1.0
        
        _betas = tuple(OmegaConf.select(config, "optimizer.params.betas", default=[0.9, 0.95]))
        _eps = float(OmegaConf.select(config, "optimizer.params.eps", default=1e-8))
        _wd = float(OmegaConf.select(config, "optimizer.params.weight_decay", default=0.05))
        param_groups = adamw_param_groups_two_level(
            model,
            pretrainer,
            lr=float(config.optimizer.params.lr),
            weight_decay=_wd,
            conditioner_lr_scale=cond_lr_scale,
        )
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        optimizer = AdamW(param_groups, betas=_betas, eps=_eps)
        
        target_steps = config.training.duration.max_steps
        if is_auto_resuming:
            global_step = resume_step
        elif skipped_cond_pretrain and cond_steps > 0:
            global_step = cond_steps
        else:
            global_step = 0


    # ==============================================================================
    # 5. DATA, ACCELERATOR PREP & LOOP ENTRY
    # ==============================================================================
    train_loader, _, val_loader = build_dataloaders(config.dataset)
    sched_last_epoch = -1
    if (not is_finetuning) and skipped_cond_pretrain and cond_steps > 0:
        sched_last_epoch = cond_steps - 1

    lr_scheduler = get_scheduler(
        config.optimizer.scheduler.name,
        optimizer,
        num_training_steps=target_steps,
        num_warmup_steps=config.optimizer.scheduler.warmup_steps,
        last_epoch=sched_last_epoch,
    )
    
    model, pretrainer, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, pretrainer, optimizer, train_loader, val_loader, lr_scheduler
    )
    
    # Only load accelerator state (Optimizer momentum, lr schedules) if Auto-Resuming
    if is_auto_resuming:
        accelerator.print(f"  > 🔄 Loading Accelerator Optimizer State...")
        accelerator.load_state(resume_path)
        accelerator.print("  > ✅ State Restored.")
    
    checkpoint_manager = CheckpointManager(logging_dir=logging_dir, keep_last_n=3)
    
    accelerator.print(f"\n[TRAIN START] Current Step: {global_step} | Target Steps: {target_steps}")
    progress_bar = tqdm(initial=global_step, total=target_steps, disable=not accelerator.is_local_main_process)
    
    data_time_m, batch_time_m = AverageMeter(), AverageMeter()
    end_time = time.time()
    val_metrics = None
    
    while global_step < target_steps:
        for batch in train_loader:
            data_time = time.time() - end_time
            data_time_m.update(data_time)
            
            # --- TRAINING PHASE ROUTER ---
            if is_finetuning:
                loss, metrics, _ = finetune_step_v4(model, extract_conditioner(pretrainer), batch, optimizer, lr_scheduler, accelerator, config, global_step)
                phase_label = "Finetune"

            elif global_step < cond_steps:
                loss, metrics = pretrain_step_v5(pretrainer, model, batch, optimizer, lr_scheduler, accelerator, config)
                phase_label = "Pretrain"
            else:
                if global_step == cond_steps and accelerator.is_main_process:
                    if config.training.get("end_to_end", False):
                        accelerator.print("\n[PHASE SWITCH] 🚀 End-to-End Mode: Keeping Conditioner unfrozen for Joint Training...")
                        # Keep pretrainer in train mode, keep requires_grad=True
                    else:
                        accelerator.print("\n[PHASE SWITCH] 🧊 Freezing Conditioner weights for AR Generative Training...")
                        pretrainer.eval() 
                        for param in pretrainer.parameters(): param.requires_grad = False
                        
                loss, metrics, _ = train_step_v5(model, extract_conditioner(pretrainer), batch, optimizer, lr_scheduler, accelerator, config, global_step)
                phase_label = "GenTrain"
            
            batch_time = time.time() - end_time
            batch_time_m.update(batch_time)
            
            if global_step % config.experiment.intervals.log_steps == 0:
                throughput = (batch[0].shape[0] * accelerator.num_processes) / (batch_time + 1e-6)
                metrics.update({'perf/throughput': throughput, 'perf/data_time': data_time_m.val, 'perf/batch_time': batch_time_m.val})
                accelerator.log(metrics, step=global_step)
                
                if accelerator.is_main_process:
                    loss_val = metrics.get(f'{phase_label.lower()}/total_loss', metrics.get('train/total_loss', loss.item()))
                    progress_bar.set_description(f"[{phase_label}] Loss: {loss_val:.4f} | Data: {data_time_m.val:.3f}s | Bt: {batch_time_m.val:.3f}s")
            
            # --- VALIDATION LOGIC ---
            if global_step % config.experiment.intervals.eval_every == 0 and global_step != 0:
                conditioner_eval = extract_conditioner(pretrainer)
                val_metrics, val_visuals = validation_loop_v5(model, conditioner_eval, val_loader, accelerator, config, global_step)
                accelerator.log(val_metrics, step=global_step)
                if val_visuals and accelerator.is_main_process:
                    log_visuals(accelerator, val_visuals, global_step, image_colormap="viridis")
            
            # --- CHECKPOINTING ---
            if global_step % config.experiment.intervals.save_every == 0 and global_step != 0:
                checkpoint_manager.save_checkpoint(accelerator, config, global_step, model, metrics=val_metrics)
            
            global_step += 1
            progress_bar.update(1)
            end_time = time.time()
            if global_step >= target_steps: break
            
    checkpoint_manager.save_checkpoint(accelerator, config, global_step, model, metrics=val_metrics, is_best=True)
    accelerator.end_training()

if __name__ == "__main__":
    main()