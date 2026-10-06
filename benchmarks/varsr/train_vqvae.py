"""VARSR stage 1: train the 1-channel LiDAR VQVAE tokenizer consumed by ``train_varsr.py --vqvae_ckpt``."""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.config_compat import (
    ensure_benchmark_dataset_config,
    resolve_benchmark_training_keys,
    resolve_gess_optimizer_params,
)
from benchmarks.import_isolation import insert_path_first
from benchmarks.varsr.varsr_repo_imports import import_varsr_vqvae_class
from src.data.lidar_dataloading import build_dataloaders
from src.utils.lr_schedulers import get_scheduler

VQVAE = import_varsr_vqvae_class()

insert_path_first(str(_REPO_ROOT / "benchmarks" / "taming" / "repo"))
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator  # noqa: E402


def get_latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None, 0
    valid_checkpoints = []
    for c in output_dir.glob("checkpoint-*"):
        parts = c.name.split("-")
        if len(parts) == 2 and parts[1].isdigit():
            valid_checkpoints.append((c, int(parts[1])))
    if not valid_checkpoints:
        return None, 0
    valid_checkpoints.sort(key=lambda x: x[1])
    return valid_checkpoints[-1]


class VAR_VQVAE(nn.Module):
    def __init__(self, in_channels=1, ch_mult=(1, 2, 4), v_patch_nums=(1, 2, 3, 4, 6, 12), vocab_size=4096, z_channels=32, ch=64):
        super().__init__()
        # test_mode=True (VAR default) freezes all params — must disable for training.
        self.vqvae = VQVAE(
            vocab_size=vocab_size, z_channels=z_channels, ch=ch,
            v_patch_nums=v_patch_nums,
            in_channels=in_channels,
            ch_mult=ch_mult,
            test_mode=False,
        )

    def forward(self, x):
        rec, usages, vq_loss = self.vqvae(x)
        return rec, vq_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--run", type=str, required=True, help="Run ID")
    args = parser.parse_args()

    config = OmegaConf.merge(OmegaConf.create(), OmegaConf.load(args.config))
    config.run = args.run
    bc = resolve_benchmark_training_keys(config)
    opt_h = resolve_gess_optimizer_params(config)
    ensure_benchmark_dataset_config(config)

    unique_run_id = f"vqgan_train_{config.run}"
    output_dir = Path(__file__).parent / "outputs" / "vqvae" / args.run
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.globals.seed)

    accelerator = Accelerator(
        mixed_precision=bc["mixed_precision"],
        log_with="wandb" if config.experiment.get("logger") == "wandb" else "tensorboard",
        project_dir=str(output_dir),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.experiment.project,
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": {"name": unique_run_id}},
        )
        print(f"Starting VQVAE pre-training. Data: {config.dataset.infra.root_dir}")

    train_dl = build_dataloaders(config.dataset)[0]

    model = VAR_VQVAE()
    loss_module = VQLPIPSWithDiscriminator(
        disc_start=config.training.get("vqgan_disc_start", 10000),
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=2,  # smaller discriminator for 48x48
        disc_in_channels=1,
        disc_weight=0.8,
        perceptual_weight=1.0,
    )

    # LPIPS expects 3 channels; repeat 1-channel CHMs.
    original_lpips_forward = loss_module.perceptual_loss.forward

    def lpips_1ch_forward(x, y):
        x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        y = y.repeat(1, 3, 1, 1) if y.shape[1] == 1 else y
        return original_lpips_forward(x, y)

    loss_module.perceptual_loss.forward = lpips_1ch_forward

    if accelerator.is_main_process:
        print(f" [VQVAE] trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f" [Discriminator] trainable parameters: {sum(p.numel() for p in loss_module.discriminator.parameters() if p.requires_grad):,}")

    opt_lr = opt_h["lr"]
    opt_ae = torch.optim.Adam(
        list(model.vqvae.encoder.parameters())
        + list(model.vqvae.decoder.parameters())
        + list(model.vqvae.quantize.parameters())
        + list(model.vqvae.quant_conv.parameters())
        + list(model.vqvae.post_quant_conv.parameters()),
        lr=opt_lr,
        betas=(0.5, 0.9),
    )
    opt_disc = torch.optim.Adam(loss_module.discriminator.parameters(), lr=opt_lr, betas=(0.5, 0.9))

    max_steps = bc["max_steps"]
    warmup_steps = bc["warmup_steps"]
    scheduler_ae = get_scheduler(
        bc["scheduler_name"], opt_ae, num_training_steps=max_steps, num_warmup_steps=warmup_steps
    )
    scheduler_disc = get_scheduler(
        bc["scheduler_name"], opt_disc, num_training_steps=max_steps, num_warmup_steps=warmup_steps
    )

    model, loss_module, opt_ae, opt_disc, train_dl, scheduler_ae, scheduler_disc = accelerator.prepare(
        model, loss_module, opt_ae, opt_disc, train_dl, scheduler_ae, scheduler_disc
    )

    resume_path, resume_step = get_latest_checkpoint(output_dir)
    if resume_path is not None:
        accelerator.print(f"Resuming from checkpoint: {resume_path} (step {resume_step})")
        accelerator.load_state(str(resume_path))
        global_step = resume_step
    else:
        global_step = 0

    progress_bar = tqdm(initial=global_step, total=max_steps, disable=not accelerator.is_main_process)

    while global_step < max_steps:
        for batch in train_dl:
            hr_img = batch["hr"] if isinstance(batch, dict) else batch[1]  # [B, 1, 48, 48]
            last_layer = accelerator.unwrap_model(model).vqvae.decoder.conv_out.weight

            opt_ae.zero_grad()
            reconstructions, qloss = model(hr_img)
            ae_loss, _ = loss_module(qloss, hr_img, reconstructions, optimizer_idx=0, global_step=global_step, last_layer=last_layer)
            accelerator.backward(ae_loss)
            opt_ae.step()
            scheduler_ae.step()

            opt_disc.zero_grad()
            discloss, _ = loss_module(qloss, hr_img, reconstructions, optimizer_idx=1, global_step=global_step, last_layer=last_layer)
            accelerator.backward(discloss)
            opt_disc.step()
            scheduler_disc.step()

            global_step += 1
            progress_bar.update(1)

            if accelerator.is_main_process and global_step % bc["log_every_n_steps"] == 0:
                accelerator.log(
                    {
                        "train/ae_loss": ae_loss.item(),
                        "train/disc_loss": discloss.item(),
                        "train/lr": scheduler_ae.get_last_lr()[0],
                        "step": global_step,
                    },
                    step=global_step,
                )
                progress_bar.set_postfix(ae_loss=f"{ae_loss.item():.4f}", disc_loss=f"{discloss.item():.4f}")

            if global_step % bc["save_every_n_steps"] == 0 or global_step == max_steps:
                save_path = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(save_path))
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_path / "vqgan.pth")
                    print(f"\nSaved checkpoint to {save_path}")

            if global_step >= max_steps:
                break

    if accelerator.is_main_process:
        accelerator.end_training()
        print("VQVAE pre-training complete.")


if __name__ == "__main__":
    main()
