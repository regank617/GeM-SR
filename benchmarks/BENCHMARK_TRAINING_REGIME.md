# Benchmark training regime

## Source of truth

GeM-SR and every baseline train from the same YAML config (e.g. `configs/generator/lidar_ssmVQ.yaml`), so all
models see the same data pipeline, step budget, warmup, and logging intervals.

Shared helpers live in `benchmarks/config_compat.py`:

- `resolve_benchmark_training_keys(config)` — `training.duration.max_steps`, `optimizer.scheduler.warmup_steps`, `experiment.intervals.*`, `training.compute.max_grad_norm`, `training.compute.mixed_precision`, `optimizer.scheduler.name`, etc.
- `resolve_cnn_adam_params(config)` — **Adam**, lr `1e-4`, β `(0.9, 0.999)`, weight decay `0` (SwinIR). LR resolution: `globals.cnn_learning_rate` → `optimizer.params.cnn_lr` → `optimizer.params.lr` → `globals.learning_rate` → `1e-4`, so the shared config can keep GeM-SR at `5e-4` while the CNN stays at `1e-4`.
- `resolve_diffusion_adam_params(config)` — same betas; **lr** from `globals.diffusion_learning_rate` → `optimizer.params.diffusion_lr` → `globals.cnn_learning_rate` → fallbacks as above (SR3, Fast Diffusion).
- `resolve_discrete_adamw_params(config)` — **AdamW**, lr `5e-4`, β `(0.9, 0.95)`, weight decay `0.05` (VARSR).
- `resolve_gess_optimizer_params(config)` — alias for `resolve_discrete_adamw_params` (historical name, kept for compatibility).
- `ensure_benchmark_dataset_config(config)` — ensures `build_dataloaders(config.dataset)` has `dataset.infra` and `dataset.loader`.

YAML `optimizer.params` overrides these defaults when present. Benchmarks should **not** duplicate long `OmegaConf.select` chains; centralize them in `config_compat.py`.

## Optimization by model family (paper table)

| Family | Models | Optimizer | LR | Betas | Weight decay |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CNN / Transformer restoration | SwinIR | Adam | `1e-4` | `(0.9, 0.999)` | `0` |
| Diffusion | SR3, Fast Diffusion | Adam | `1e-4` | `(0.9, 0.999)` | `0` |
| Discrete generative | VARSR, GeM-SR generator (`training/train_generator.py`) | AdamW | `5e-4` | `(0.9, 0.95)` | `0.05` |

## GeM-SR generator (`training/train_generator.py`)

- **AdamW** with `betas` and `eps` from `optimizer.params` (defaults: `(0.9, 0.95)`, `1e-8`).
- **Parameter groups** via `src.utils.optimizer_utils.adamw_param_groups_two_level`: weight decay `0.05` only on weights that are not biases, norm weights (LayerNorm / RMSNorm / GroupNorm), or parameters whose name contains `gamma` (ConvNeXt-style scale). The conditioner uses the same rule with optional `cond_lr_scale`.

## LR schedule

`src.utils.lr_schedulers.get_scheduler` with `optimizer.scheduler.name` (e.g. `cosine_with_minimum`) and `optimizer.scheduler.warmup_steps` from the YAML. **VARSR** uses the same scheduler API as the GeM-SR generator (`num_training_steps`, `num_warmup_steps`).

## VARSR VQVAE tokenizer (GAN-style phase)

`benchmarks/varsr/train_vqvae.py` trains the VQVAE generator and the PatchGAN discriminator with `torch.optim.Adam`
and **β = (0.5, 0.9)** (CompVis taming-transformers / VQGAN convention), **not** the table above. Only the
**learning rate** comes from the YAML (via `resolve_gess_optimizer_params` → `lr`); the betas are fixed for GAN
stability. The discriminator starts at `training.vqgan_disc_start` (default 10000 steps).

## VARSR training time

VARSR (built on VAR) is slow to train mostly because of its **long multi-scale AR sequences**, **deep transformer stacks**, **per-step VAE / token preparation**, and **data loading**. Upstream VAR also runs the conditioning / embedding prefix under `torch.cuda.amp.autocast(enabled=False)` by design (that block stays FP32; the transformer blocks run in mixed precision). If the ETA looks extreme, profile **steps/sec**, **dataloader wait**, and **GPU utilization** before blaming AMP.

## Upstream references

| Model       | Upstream |
| ----------- | -------- |
| SwinIR      | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) |
| SR3         | [Janspiry/Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) |
| VAR / VARSR | [FoundationVision/VAR](https://github.com/FoundationVision/VAR), [quyp2000/VARSR](https://github.com/quyp2000/VARSR) |
