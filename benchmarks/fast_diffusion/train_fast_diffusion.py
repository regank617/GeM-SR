import os
import sys

# Add project root to sys.path so we can import src.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""
Fast Diffusion Benchmark Training Script - Accelerated
Tailored for 1-channel LiDAR CHM Super-Resolution (32x16 -> 48x48)
Using Rectified Flow (Flow Matching) for fast 1-4 NFE inference.
"""
import sys
import os
import argparse
import time
import math
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from benchmarks.config_compat import resolve_diffusion_adam_params
from benchmarks.ema_utils import ParameterEMA

# Setup Paths for GeSS imports
from src.data.lidar_dataloading import build_dataloaders
from src.utils.meter import AverageMeter
from src.utils.lr_schedulers import get_scheduler

# This package directory (so ``flow_matching`` resolves when imported from notebooks / other cwd)
_FAST_DIFFUSION_DIR = Path(__file__).resolve().parent
if str(_FAST_DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(_FAST_DIFFUSION_DIR))

# Adjust path to import from the cloned SR3 repo
_SR3_REPO = _FAST_DIFFUSION_DIR.parent / "sr3" / "repo"
if str(_SR3_REPO) not in sys.path:
    sys.path.insert(0, str(_SR3_REPO))
from model.sr3_modules.unet import UNet

from flow_matching import FlowMatching


def get_latest_checkpoint(output_dir):
    import os
    from pathlib import Path
    output_dir = Path(output_dir)
    if not output_dir.exists(): return None, 0
    checkpoints = list(output_dir.glob("checkpoint-*"))
    valid_checkpoints = []
    for c in checkpoints:
        parts = c.name.split("-")
        if len(parts) == 2 and parts[1].isdigit():
            step = int(parts[1])
            valid_checkpoints.append((c, step))
    if not valid_checkpoints: return None, 0
    valid_checkpoints.sort(key=lambda x: x[1])
    return valid_checkpoints[-1]

class FastDiffusionWrapper(nn.Module):
    """
    Wraps the SR3 UNet and FlowMatching models.
    Operates by concatenating a bilinearly-upsampled Low-Res image
    with the High-Res target/noise.
    """
    def __init__(self, n_colors=1, image_size=48, num_inference_steps=4):
        super().__init__()
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        
        # in_channel is n_colors * 2 because of the concatenation of HR_noisy and LR_upsampled
        # Register UNet only as FlowMatching.denoise_fn (avoids duplicate state_dict / shared tensors).
        unet = UNet(
            in_channel=n_colors * 2,
            out_channel=n_colors,
            inner_channel=64,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8),
            attn_res=(12,),
            res_blocks=2,
            dropout=0.0,
            image_size=image_size,
            with_noise_level_emb=True,  # Keeps the time-embedding logic active
        )
        
        self.diffusion = FlowMatching(unet)

    def load_state_dict(self, state_dict, strict=True):
        """Map legacy ``unet.*`` keys to ``diffusion.denoise_fn.*`` when needed."""
        sd = dict(state_dict)
        unet_p, denoise_p = "unet.", "diffusion.denoise_fn."
        drop = []
        add = {}
        for k in list(sd.keys()):
            if not k.startswith(unet_p):
                continue
            alt = denoise_p + k[len(unet_p) :]
            if alt in sd:
                drop.append(k)
            else:
                add[alt] = sd[k]
                drop.append(k)
        for k in drop:
            sd.pop(k, None)
        sd.update(add)
        # Checkpoints from before FlowMatching stored the SR3 noise schedule buffer.
        buf = "diffusion._sqrt_alphas_cumprod"
        if buf not in sd:
            sd[buf] = self.diffusion._sqrt_alphas_cumprod.detach().clone()
        return super().load_state_dict(sd, strict=strict)

    def forward(self, lr, hr=None, target_size=None):
        """
        Args:
            lr: Low-res input ``(B, C, H_lr, W_lr)``.
            hr: High-res target (training only), or at eval can supply only to define
                output shape via ``hr.shape[-2:]``.
            target_size: Optional ``(H, W)`` for upsampled condition / sampled output.
                If None at eval and ``hr`` is None, uses ``(self.image_size, self.image_size)``.
        """
        if self.training:
            if hr is None:
                raise ValueError("FastDiffusionWrapper training requires ``hr``.")
            out_hw = hr.shape[-2:]
        else:
            if hr is not None:
                out_hw = hr.shape[-2:]
            elif target_size is not None:
                out_hw = tuple(target_size)
            else:
                out_hw = (self.image_size, self.image_size)

        lr_up = F.interpolate(
            lr, size=out_hw, mode="bilinear", align_corners=False
        )

        if self.training:
            x_in = {"HR": hr, "SR": lr_up}
            return self.diffusion(x_in) # Returns the MSE Velocity Loss
            
        # Fast Inference (e.g., 4 steps matching your NFE budget)
        return self.diffusion.super_resolution(lr_up, num_steps=self.num_inference_steps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--run", type=str, required=True, help="Run ID")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Exponential moving average of trainable weights; validation and best checkpoints use EMA.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay (only used with --use_ema). Typical: 0.999 - 0.9999.",
    )
    args, unknown = parser.parse_known_args()
    
    yaml_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(yaml_conf, cli_conf)
    
    # Manually attach the run argument to the config so it mimics your trainV5.py behavior
    config.run = args.run
    
    set_seed(config.globals.seed)

    # Resolve Data Path
    root_dir = None
    for path in config.dataset.infra.get("root_dir", []):
        if Path(path).exists():
            root_dir = Path(path)
            break
    if root_dir is None: raise RuntimeError("❌ No valid root_dir found!")
    config.dataset.infra.root_dir = str(root_dir)

    # Disable multiprocessing to prevent SemLock errors
    config.dataset.infra.num_workers_per_gpu = 4
    config.dataset.infra.pin_memory = True
    config.dataset.infra.persistent_workers = True
    
    # Mode & Logging Setup
    unique_run_id = f"fast_diffusion_benchmark_{config.run}"
    output_dir = Path(__file__).parent / "outputs" / args.run
    output_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.compute.grad_accum_steps,
        mixed_precision=config.training.compute.mixed_precision,
        log_with="wandb" if config.experiment.logger == "wandb" else "tensorboard",
        project_dir=str(output_dir)
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name=config.experiment.project, config=OmegaConf.to_container(config, resolve=True), init_kwargs={"wandb": {"name": unique_run_id}})
    
    accelerator.print(f"🚀 Starting Fast Diffusion Benchmark. Data: {root_dir}")
    if args.use_ema:
        accelerator.print(f"   EMA enabled (decay={args.ema_decay})")

    # Build Components
    model = FastDiffusionWrapper(n_colors=1, image_size=48, num_inference_steps=args.steps)
    
    op = resolve_diffusion_adam_params(config)
    optimizer = optim.Adam(
        model.parameters(),
        lr=op["lr"],
        betas=op["betas"],
        eps=op["eps"],
        weight_decay=op["weight_decay"],
    )
    
    # [-1, 1] inputs (same as SR3); required for diffusion-style training / FlowMatching noise scale
    train_loader, _, val_loader = build_dataloaders(
        config.dataset, normalize_diffusion=True
    )
    target_steps = config.training.duration.max_steps
    
    lr_scheduler = get_scheduler(
        config.optimizer.scheduler.name, optimizer, 
        num_training_steps=target_steps, 
        num_warmup_steps=config.optimizer.scheduler.warmup_steps
    )
    
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    ema = None
    if args.use_ema:
        ema = ParameterEMA(accelerator.unwrap_model(model), decay=args.ema_decay)

    lpips_fn = lpips.LPIPS(net="vgg").to(accelerator.device)
    lpips_fn.eval()

    # Auto-Resume Logic
    resume_path, resume_step = get_latest_checkpoint(output_dir)
    if resume_path is not None:
        accelerator.print(f"🔄 Resuming from checkpoint: {resume_path} (Step {resume_step})")
        accelerator.load_state(str(resume_path))
        global_step = resume_step
        if ema is not None:
            ema_pt = Path(resume_path) / "ema_state.pt"
            if ema_pt.is_file():
                ema.load_state_dict(torch.load(ema_pt, map_location="cpu"))
            else:
                accelerator.print(
                    "  [EMA] No ema_state.pt in checkpoint; re-initializing EMA shadow from loaded weights."
                )
                ema = ParameterEMA(accelerator.unwrap_model(model), decay=args.ema_decay)
    else:
        global_step = 0

    if accelerator.is_main_process and global_step == 0:
        torch.save(accelerator.unwrap_model(model).state_dict(), output_dir / "initial_model.pth")
    
    progress_bar = tqdm(initial=global_step, total=target_steps, disable=not accelerator.is_local_main_process)
    best_val_psnr = float("-inf")
    best_val_lpips = float("inf")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(accelerator.device)
    
    while global_step < target_steps:
        model.train()
        for batch in train_loader:
            lr_imgs, hr_imgs = batch
            
            with accelerator.accumulate(model):
                # FastDiffusionWrapper computes the flow matching MSE loss internally during training
                loss = model(lr_imgs, hr_imgs)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.compute.get('max_grad_norm', 1.0))
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients and ema is not None:
                    ema.update(accelerator.unwrap_model(model))
            
            if global_step % config.experiment.intervals.log_steps == 0:
                accelerator.log({"train/loss": loss.item(), "train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                if accelerator.is_main_process:
                    progress_bar.set_description(f"Loss: {loss.item():.4f}")
            
            # Validation (fast Flow-Matching sampling; data in [-1, 1])
            if global_step % config.experiment.intervals.eval_every == 0 and global_step > 0:
                model.eval()
                unwrapped = accelerator.unwrap_model(model)
                if ema is not None:
                    ema.apply_shadow(unwrapped)
                val_loss_m = AverageMeter()
                val_lpips_m = AverageMeter()
                ssim_metric.reset()
                
                try:
                    with torch.no_grad():
                        eval_limit = 10
                        for i, (v_lr, v_hr) in enumerate(val_loader):
                            if i >= eval_limit:
                                break
                            v_sr = model(v_lr)
                            v_mse = F.mse_loss(v_sr, v_hr)
                            val_loss_m.update(v_mse.item(), v_lr.size(0))
                            v_sr3 = v_sr.repeat(1, 3, 1, 1) if v_sr.size(1) == 1 else v_sr
                            v_hr3 = v_hr.repeat(1, 3, 1, 1) if v_hr.size(1) == 1 else v_hr
                            batch_lpips = lpips_fn(v_sr3, v_hr3).mean()
                            val_lpips_m.update(batch_lpips.item(), v_lr.size(0))
                            ssim_metric.update(v_sr, v_hr)
                finally:
                    if ema is not None:
                        ema.restore(unwrapped)
                
                if val_loss_m.count > 0:
                    val_psnr = 10 * math.log10(4.0 / val_loss_m.avg)
                    val_ssim = ssim_metric.compute()
                    if isinstance(val_ssim, torch.Tensor):
                        val_ssim = val_ssim.detach().item()
                    val_lpips = val_lpips_m.avg
                    log_payload = {
                        "val/mse": val_loss_m.avg,
                        "val/psnr": val_psnr,
                        "val/ssim": val_ssim,
                        "val/lpips": val_lpips,
                    }
                    accelerator.log(log_payload, step=global_step)
                    
                    if accelerator.is_main_process:
                        accelerator.print(
                            f"  -> Step {global_step} | Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f}"
                        )
                        unwrapped_model = accelerator.unwrap_model(model)
                        best_sd = (
                            ema.merge_into_full_state_dict(unwrapped_model)
                            if ema is not None
                            else unwrapped_model.state_dict()
                        )
                        if val_psnr > best_val_psnr:
                            best_val_psnr = val_psnr
                            torch.save(best_sd, output_dir / "best_model_psnr.pth")
                            accelerator.log({"val/best_psnr": best_val_psnr}, step=global_step)
                        if val_lpips < best_val_lpips:
                            best_val_lpips = val_lpips
                            torch.save(best_sd, output_dir / "best_model_lpips.pth")
                            accelerator.log({"val/best_lpips": best_val_lpips}, step=global_step)
                
                # Save full accelerator state for resuming
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(ckpt_dir))
                if accelerator.is_main_process and ema is not None:
                    torch.save(ema.state_dict(), ckpt_dir / "ema_state.pt")
                if accelerator.is_main_process:
                    import shutil
                    checkpoints = list(output_dir.glob("checkpoint-*"))
                    if len(checkpoints) > 2:
                        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
                        for c in checkpoints[:-2]:
                            shutil.rmtree(c)
                model.train()
            
            global_step += 1
            progress_bar.update(1)
            if (
                global_step > 0
                and global_step % 100_000 == 0
                and accelerator.is_main_process
            ):
                um = accelerator.unwrap_model(model)
                torch.save(
                    ema.merge_into_full_state_dict(um) if ema is not None else um.state_dict(),
                    output_dir / f"model_step_{global_step}.pth",
                )
            if global_step >= target_steps: break

    if accelerator.is_main_process:
        um = accelerator.unwrap_model(model)
        torch.save(
            ema.merge_into_full_state_dict(um) if ema is not None else um.state_dict(),
            output_dir / "final_model.pth",
        )
    accelerator.end_training()

if __name__ == "__main__":
    main()
