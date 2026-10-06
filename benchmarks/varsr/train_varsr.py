import os
import sys

# Add project root to sys.path so we can import src.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""
VARSR Benchmark Training Script - Accelerated
Tailored for 1-channel LiDAR CHM Super-Resolution (32x16 -> 48x48)
"""
import sys
import os
import argparse
import time
import math
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.config_compat import (
    ensure_benchmark_dataset_config,
    resolve_benchmark_training_keys,
    resolve_gess_optimizer_params,
)

# Setup Paths for GeSS imports
from src.data.lidar_dataloading import build_dataloaders
from src.utils.meter import AverageMeter
from src.utils.lr_schedulers import get_scheduler

# VARSR shares the package name ``models`` with benchmarks/var/repo; wrapper import resolves VAR_RoPE.
from benchmarks.varsr.wrapper import VARSRWrapper
from models.vqvae import VQVAE  # noqa: E402 — varsr repo (path set by wrapper import)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--run", type=str, required=True, help="Run ID")
    parser.add_argument("--vqvae_ckpt", type=str, required=True, help="Path to pre-trained VQVAE checkpoint")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--depth", type=int, default=12, help="VARSR depth")
    parser.add_argument("--embed_dim", type=int, default=256, help="VARSR embed_dim")
    parser.add_argument("--num_heads", type=int, default=4, help="VARSR num_heads")
    parser.add_argument(
        "--mlp_ratio",
        type=float,
        default=4.0,
        help="FFN hidden width = round(embed_dim * mlp_ratio). Lower (e.g. 2) cuts params vs width alone.",
    )
    args = parser.parse_args()

    yaml_conf = OmegaConf.load(args.config)
    config = OmegaConf.create()
    config = OmegaConf.merge(config, yaml_conf)
    config.run = args.run

    if OmegaConf.select(config, "infra") is None:
        data_path = OmegaConf.select(config, "globals.data_path", default="./data")
        config.infra = OmegaConf.create({
            "root_dir": data_path,
            "train_data_dir": f"{data_path}/train",
            "val_data_dir": f"{data_path}/val",
            "num_workers_per_gpu": 4,
            "pin_memory": True,
            "persistent_workers": True,
        })
    ensure_benchmark_dataset_config(config)
    bc = resolve_benchmark_training_keys(config)
    opt_h = resolve_gess_optimizer_params(config)

    # Same layout as other benchmarks, e.g. benchmarks/var/outputs/vqvae/<run>
    output_dir = Path(__file__).parent / "outputs" / "varsr" / args.run
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_logger = OmegaConf.select(config, "experiment.logger", default="wandb")
    if exp_logger not in ("wandb", "tensorboard"):
        exp_logger = "wandb"

    accelerator = Accelerator(
        mixed_precision=bc["mixed_precision"],
        log_with=exp_logger,
        project_dir=str(output_dir),
    )

    set_seed(int(OmegaConf.select(config, "globals.seed", default=42)))
    unique_run_id = f"varsr_{args.run}"

    if accelerator.is_main_process:
        print(f"🚀 Starting VARSR Benchmark. Data: {config.infra.root_dir}")
        print(f"   Output dir: {output_dir}")
        accelerator.init_trackers(
            project_name=OmegaConf.select(config, "experiment.project", default="GeSS"),
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": {"name": unique_run_id}},
        )

    # Dataloaders
    dataloaders = build_dataloaders(config.dataset)
    if isinstance(dataloaders, tuple):
        train_loader = dataloaders[0]
        val_loader = dataloaders[1] if len(dataloaders) > 1 else None
    else:
        train_loader = dataloaders
        val_loader = None

    # Load VQVAE
    v_patch_nums = (1, 2, 3, 4, 6, 12)
    vae_local = VQVAE(vocab_size=4096, z_channels=32, ch=64, v_patch_nums=v_patch_nums, in_channels=1, ch_mult=(1, 2, 4))
    
    # Load VQVAE checkpoint
    vae_ckpt = torch.load(args.vqvae_ckpt, map_location='cpu')
    if 'vqvae' in vae_ckpt:
        vae_local.load_state_dict(vae_ckpt['vqvae'])
    else:
        # try to load directly or with prefix
        state_dict = vae_ckpt
        if any(k.startswith('vqvae.') for k in state_dict.keys()):
            state_dict = {k.replace('vqvae.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.')}
        vae_local.load_state_dict(state_dict)
        
    vae_local.eval()
    for p in vae_local.parameters():
        p.requires_grad = False

    # Model
    model = VARSRWrapper(
        vae_local=vae_local,
        embed_dim=args.embed_dim,
        patch_nums=v_patch_nums,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    
    if accelerator.is_main_process:
        print(f" [VAR] Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_h["lr"],
        betas=opt_h["betas"],
        weight_decay=opt_h["weight_decay"],
        eps=opt_h["eps"],
    )

    max_steps = int(args.max_steps) if args.max_steps is not None else bc["max_steps"]
    warmup_steps = bc["warmup_steps"]
    lr_scheduler = get_scheduler(
        bc["scheduler_name"],
        optimizer,
        num_training_steps=max_steps,
        num_warmup_steps=warmup_steps,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model, optimizer, train_loader, lr_scheduler, vae_local = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler, vae_local
    )

    resume_path, resume_step = get_latest_checkpoint(output_dir)
    if resume_path is not None:
        accelerator.print(f"🔄 Resuming from checkpoint: {resume_path} (Step {resume_step})")
        accelerator.load_state(str(resume_path))
        global_step = resume_step
    else:
        global_step = 0

    model.train()

    progress_bar = tqdm(initial=global_step, total=max_steps, disable=not accelerator.is_local_main_process)

    while global_step < max_steps:
        for batch in train_loader:
            if isinstance(batch, dict):
                lr_img = batch['lr']
                hr_img = batch['hr']
            else:
                lr_img, hr_img = batch
                
            # 1. Get ground truth tokens from VQVAE
            # quant.f_to_idxBl_or_fhat returns (per-scale idx + final continuous residual, idx_N_list).
            # idxBl_to_var_input expects idx_N_list (same as official VARSR trainer).
            with torch.no_grad():
                idx_out = vae_local.img_to_idxBl(hr_img)
                if isinstance(idx_out, tuple) and len(idx_out) == 2:
                    gt_idx_Bl, idx_N_list = idx_out
                else:
                    gt_idx_Bl = idx_out
                    idx_N_list = gt_idx_Bl
                SN = len(v_patch_nums)
                gt_BL = torch.cat(gt_idx_Bl[0:SN], dim=1)  # discrete scales only; exclude continuous tail
                x_BLCv_wo_first_l = vae_local.quantize.idxBl_to_var_input(idx_N_list)
                
            # 2. Forward VARSR
            logits_BLV, diff_loss, out_rgbs, mask_wo_prev_stages = model(lr_img, x_BLCv_wo_first_l, gt_idx_Bl)
            
            # Adjust gt_BL based on masking
            B = hr_img.size(0)
            last_level_pns = model.var.last_level_pns if hasattr(model, 'var') else 144
            gt_BL = torch.cat((gt_BL[:, :-last_level_pns], gt_BL[:, -last_level_pns:][mask_wo_prev_stages].view(B,-1)), dim=1)
            
            # 3. Loss
            logits_loss = criterion(logits_BLV.contiguous().view(-1, logits_BLV.size(-1)), gt_BL.view(-1))
            loss = logits_loss + diff_loss * 2.0
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), bc["max_grad_norm"])

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                if global_step % bc["log_every_n_steps"] == 0:
                    accelerator.log(
                        {"train/loss": loss.item(), "train/lr": lr_scheduler.get_last_lr()[0]},
                        step=global_step,
                    )

            global_step += 1
            
            if global_step % 5000 == 0:
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), output_dir / f"var_step_{global_step}.pth")
            
                # Save full accelerator state for resuming
                accelerator.save_state(str(output_dir / f"checkpoint-{global_step}"))
                if accelerator.is_main_process:
                    import shutil
                    checkpoints = list(output_dir.glob("checkpoint-*"))
                    if len(checkpoints) > 2:
                        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
                        for c in checkpoints[:-2]:
                            shutil.rmtree(c)

            if global_step >= max_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), output_dir / "var_final.pth")
        print(f"Saved final checkpoint to {output_dir / 'var_final.pth'}")
        accelerator.end_training()
    else:
        accelerator.end_training()

if __name__ == "__main__":
    main()
