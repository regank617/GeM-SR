import os
import sys

# Add project root to sys.path so we can import src.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""
SwinIR Benchmark Training Script - Accelerated
Tailored for 1-channel LiDAR CHM Super-Resolution (32x16 -> 48x48)

Previous pixelshuffle + asymmetric-conv wrapper lives in ``swinir_old.py`` (``SwinIRWrapperLegacy``).
"""
import sys
import os
import argparse
import math
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from benchmarks.config_compat import resolve_cnn_adam_params
from benchmarks.import_isolation import clear_benchmark_namespaces, insert_path_first

from src.data.lidar_dataloading import build_dataloaders
from src.utils.meter import AverageMeter
from src.utils.lr_schedulers import get_scheduler

# VAR/VARSR/other benchmarks cache ``models`` in sys.modules; clear and prefer this repo first
# (same idea as ``new_eval_bench.import_swinir_wrapper``).
_SWINIR_REPO = str(Path(__file__).resolve().parent / "repo")
clear_benchmark_namespaces()
insert_path_first(_SWINIR_REPO)
from models.network_swinir import SwinIR  # noqa: E402


def get_latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None, 0
    checkpoints = list(output_dir.glob("checkpoint-*"))
    valid_checkpoints = []
    for c in checkpoints:
        parts = c.name.split("-")
        if len(parts) == 2 and parts[1].isdigit():
            step = int(parts[1])
            valid_checkpoints.append((c, step))
    if not valid_checkpoints:
        return None, 0
    valid_checkpoints.sort(key=lambda x: x[1])
    return valid_checkpoints[-1]


class SwinIRWrapper(nn.Module):
    """
    SwinIR in LR space; deep features are upsampled to HR and added to a bicubic base (global residual).
    See ``swinir_old.SwinIRWrapperLegacy`` for the previous pixelshuffle + asymmetric-conv design.
    """

    def __init__(
        self,
        n_colors=1,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        img_size=(32, 16),
        out_hw=(48, 48),
    ):
        super().__init__()
        self.out_h, self.out_w = out_hw
        depths = list(depths)
        num_heads = list(num_heads)

        self.swinir = SwinIR(
            upscale=1,
            in_chans=n_colors,
            img_size=img_size,
            window_size=window_size,
            img_range=1.0,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            upsampler="",
            resi_connection="1conv",
        )

        self.upsample = nn.Sequential(
            nn.Upsample(size=(self.out_h, self.out_w), mode="bicubic", align_corners=False),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, n_colors, kernel_size=3, padding=1),
        )

    def _deep_features(self, x):
        """``[B, embed_dim, H_lr, W_lr]`` trunk output (before SwinIR ``conv_last`` / global image residual)."""
        sw = self.swinir
        H, W = x.shape[2], x.shape[3]
        x_pad = sw.check_image_size(x)
        sw.mean = sw.mean.type_as(x_pad)
        x_n = (x_pad - sw.mean) * sw.img_range
        x_first = sw.conv_first(x_n)
        feat = sw.conv_after_body(sw.forward_features(x_first)) + x_first
        return feat[:, :, :H, :W].contiguous()

    def forward(self, x):
        feat = self._deep_features(x)
        residual = self.upsample(feat)
        base_img = F.interpolate(
            x, size=(self.out_h, self.out_w), mode="bicubic", align_corners=False
        )
        out = base_img + residual
        return torch.clamp(out, min=0.0, max=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--run", type=str, required=True, help="Run ID")
    parser.add_argument("--embed_dim", type=int, default=180)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--depths", type=str, default="6,6,6,6,6,6")
    parser.add_argument("--num_heads", type=str, default="6,6,6,6,6,6")
    args, unknown = parser.parse_known_args()

    depths = [int(x) for x in args.depths.split(",")]
    num_heads = [int(x) for x in args.num_heads.split(",")]

    yaml_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(yaml_conf, cli_conf)

    config.run = args.run

    set_seed(config.globals.seed)

    root_dir = None
    for path in config.dataset.infra.get("root_dir", []):
        if Path(path).exists():
            root_dir = Path(path)
            break
    if root_dir is None:
        raise RuntimeError("❌ No valid root_dir found!")
    config.dataset.infra.root_dir = str(root_dir)

    config.dataset.infra.num_workers_per_gpu = 4
    config.dataset.infra.pin_memory = True
    config.dataset.infra.persistent_workers = True

    unique_run_id = f"swinir_benchmark_{config.run}"
    output_dir = Path(__file__).parent / "outputs" / args.run
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.compute.grad_accum_steps,
        mixed_precision=config.training.compute.mixed_precision,
        log_with="wandb" if config.experiment.logger == "wandb" else "tensorboard",
        project_dir=str(output_dir),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.experiment.project,
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": {"name": unique_run_id}},
        )

    accelerator.print(f"🚀 Starting SwinIR Benchmark. Data: {root_dir}")

    model = SwinIRWrapper(
        n_colors=1,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f" [SwinIR] Number of trainable parameters: {num_trainable:,}")

    op = resolve_cnn_adam_params(config)
    optimizer = optim.Adam(
        model.parameters(),
        lr=op["lr"],
        betas=op["betas"],
        eps=op["eps"],
        weight_decay=op["weight_decay"],
    )

    # [0, 1] targets (same as GeSS / VAR); do not use normalize_diffusion
    train_loader, _, val_loader = build_dataloaders(
        config.dataset, normalize_diffusion=False
    )
    target_steps = config.training.duration.max_steps

    lr_scheduler = get_scheduler(
        config.optimizer.scheduler.name,
        optimizer,
        num_training_steps=target_steps,
        num_warmup_steps=config.optimizer.scheduler.warmup_steps,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    lpips_fn = lpips.LPIPS(net="vgg").to(accelerator.device)
    lpips_fn.eval()

    resume_path, resume_step = get_latest_checkpoint(output_dir)
    if resume_path is not None:
        accelerator.print(f"🔄 Resuming from checkpoint: {resume_path} (Step {resume_step})")
        accelerator.load_state(str(resume_path))
        global_step = resume_step
    else:
        global_step = 0

    if accelerator.is_main_process and global_step == 0:
        torch.save(accelerator.unwrap_model(model).state_dict(), output_dir / "initial_model.pth")

    criterion = nn.L1Loss()

    progress_bar = tqdm(initial=global_step, total=target_steps, disable=not accelerator.is_local_main_process)
    best_val_psnr = float("-inf")
    best_val_lpips = float("inf")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)

    while global_step < target_steps:
        model.train()
        for batch in train_loader:
            lr_imgs, hr_imgs = batch

            with accelerator.accumulate(model):
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.training.compute.get("max_grad_norm", 1.0)
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if global_step % config.experiment.intervals.log_steps == 0:
                accelerator.log(
                    {"train/loss": loss.item(), "train/lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step,
                )
                if accelerator.is_main_process:
                    progress_bar.set_description(f"Loss: {loss.item():.4f}")

            if global_step % config.experiment.intervals.eval_every == 0 and global_step > 0:
                model.eval()
                val_loss_m = AverageMeter()
                val_lpips_m = AverageMeter()
                ssim_metric.reset()

                with torch.no_grad():
                    for v_lr, v_hr in val_loader:
                        v_sr = model(v_lr)
                        v_mse = F.mse_loss(v_sr, v_hr)
                        val_loss_m.update(v_mse.item(), v_lr.size(0))
                        v_sr3 = v_sr.repeat(1, 3, 1, 1) if v_sr.size(1) == 1 else v_sr
                        v_hr3 = v_hr.repeat(1, 3, 1, 1) if v_hr.size(1) == 1 else v_hr
                        batch_lpips = lpips_fn(v_sr3 * 2.0 - 1.0, v_hr3 * 2.0 - 1.0).mean()
                        val_lpips_m.update(batch_lpips.item(), v_lr.size(0))
                        ssim_metric.update(v_sr, v_hr)

                val_mse = val_loss_m.avg
                val_rmse = math.sqrt(val_mse) if val_mse > 0 else 0.0
                val_psnr = 10 * math.log10(1.0 / val_mse) if val_mse > 0 else 100.0
                val_ssim = ssim_metric.compute()
                if isinstance(val_ssim, torch.Tensor):
                    val_ssim = val_ssim.detach().item()
                val_lpips = val_lpips_m.avg

                accelerator.log(
                    {
                        "val/mse": val_mse,
                        "val/rmse": val_rmse,
                        "val/psnr": val_psnr,
                        "val/ssim": val_ssim,
                        "val/lpips": val_lpips,
                    },
                    step=global_step,
                )

                if accelerator.is_main_process:
                    accelerator.print(
                        f"  -> Step {global_step} | Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | "
                        f"LPIPS: {val_lpips:.4f} | RMSE: {val_rmse:.4f}"
                    )
                    unwrapped_model = accelerator.unwrap_model(model)
                    if val_psnr > best_val_psnr:
                        best_val_psnr = val_psnr
                        torch.save(unwrapped_model.state_dict(), output_dir / "best_model_psnr.pth")
                        accelerator.log({"val/best_psnr": best_val_psnr}, step=global_step)
                    if val_lpips < best_val_lpips:
                        best_val_lpips = val_lpips
                        torch.save(unwrapped_model.state_dict(), output_dir / "best_model_lpips.pth")
                        accelerator.log({"val/best_lpips": best_val_lpips}, step=global_step)

                accelerator.save_state(str(output_dir / f"checkpoint-{global_step}"))
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
            if global_step > 0 and global_step % 100_000 == 0 and accelerator.is_main_process:
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    output_dir / f"model_step_{global_step}.pth",
                )
            if global_step >= target_steps:
                break

    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), output_dir / "final_model.pth")
    accelerator.end_training()


if __name__ == "__main__":
    main()
