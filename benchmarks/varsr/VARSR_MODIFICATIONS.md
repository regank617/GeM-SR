# VARSR (Visual Autoregressive Super-Resolution) Benchmark Modifications

This document outlines the modifications made to the official VARSR repository ([quyp2000/VARSR](https://github.com/quyp2000/VARSR), MIT) to adapt it for the 1-channel LiDAR CHM super-resolution task (32x16 -> 48x48).

The upstream AVA dataset index (`repo/dataloader/meta_info_AVADataset.csv`) was removed from the vendored copy; it is not used for LiDAR training or evaluation.

## 1. VQVAE Adaptation
- **Channel Modification**: Similar to the original VAR, we patched the `VQVAE` class in `models/vqvae.py` to accept `in_channels` and `ch_mult` as arguments.
- **Downsampling Factor**: We modified `ch_mult` to `(1, 2, 4)` for a factor of 4 downsampling, resulting in a 12x12 latent space for our 48x48 images.
- **Patch Numbers**: We updated `v_patch_nums` to `(1, 2, 3, 4, 6, 12)` to match the 12x12 latent space.
- **Training / eval VQVAE**: `ch=64`, `in_channels=1`, `v_patch_nums=(1,2,3,4,6,12)`. Construct it from **`benchmarks/varsr/repo`** (`models.vqvae.VQVAE` via `make_lidar_varsr_vqvae()`), matching `train_varsr.py`.
- **Stage-1 trainer (`train_vqvae.py`)**: trains this VQVAE on 48x48 HR CHMs with the LPIPS + PatchGAN loss from taming-transformers (`benchmarks/taming/`, see `TAMING_MODIFICATIONS.md`), Adam with β=(0.5, 0.9). It was originally written against the upstream VAR repo's `VQVAE`; VARSR's `VQVAE` has identical parameters and an identical `forward`, so it now builds the model from `benchmarks/varsr/repo` and its checkpoints load directly into `train_varsr.py` / the hub (`--vqvae_ckpt` / `--varsr_vqvae_ckpt`). It saves `outputs/vqvae/<run>/checkpoint-<step>/vqgan.pth` and has no validation loop; the checkpoint used in the paper was selected offline by validation-set LPIPS among the saved steps and stored as `best_model_lpips.pth` (available upon request).

## 2. VARSR Transformer Adaptation
- **ControlNetConditioningEmbedding**: VARSR uses a `ControlNetConditioningEmbedding` to process the upscaled low-resolution image into a conditioning sequence. We patched `models/var.py` to allow `in_channels=1`.
- **Downsampling Blocks**: The original `ControlNetConditioningEmbedding` uses 4 blocks (downsampling by 16). Since our VQVAE only downsamples by 4, we patched `VAR_RoPE` to initialize `ControlNetConditioningEmbedding` with `(32, 128, 256)` (2 blocks, downsampling by 4). This ensures the conditioning sequence length perfectly matches the first scale of the autoregressive generation (`first_l`).
- **Upscaling**: In our custom `VARSRWrapper`, we upscale the 32x16 low-resolution input to 48x48 using bilinear interpolation before passing it to the `VAR_RoPE` model, mimicking VARSR's default degradation/upscaling pipeline.

## 3. Path setup (benchmark layout)
VARSR, SwinIR, and SR3 all define generic top-level packages (`models`, `utils`, `model`, `data`, ...). **`benchmarks/varsr/varsr_repo_imports.py`** therefore clears those cached namespaces (`clear_benchmark_namespaces()`), prepends **`benchmarks/varsr/repo`** to **`sys.path`**, imports **`VAR_RoPE`** / **`models.vqvae.VQVAE`**, and then calls **`verify_vendor_namespaces()`**, which warns if a module from a different baseline is still cached. See `benchmarks/import_isolation.py` for details and for how to add another baseline safely.

- **`benchmarks/varsr/wrapper.py`**: **`VARSRWrapper`** — import this first, then construct the LiDAR VQVAE via **`make_lidar_varsr_vqvae()`** (same order as **`train_varsr.py`**).

Shared GeM-SR utilities (data loaders, schedulers) are imported from `src/`.

**`models/quant.py`**: same single-GPU patch as VAR — **`torch.distributed.get_world_size()`** is only used when **`torch.distributed.is_initialized()`**; otherwise world size is **1** (avoids errors under Accelerate without `init_process_group`).

## 4. Inference / evaluation
- **`VARSRWrapper._apply`**: moves the shared **`VQVAE`** when the wrapper is **`.to()` / `.cuda()`**’d (upstream stores the VAE in a tuple, not as a registered submodule).
- **`VARSRWrapper.generate`**: wraps **`VAR_RoPE.autoregressive_infer_cfg`** with the same **48×48** bilinear LR conditioning as training, dummy **`label_B = 0`**, and unused text arguments set to **`None`**. Returns **`[B, C, H, W]`** in **[0, 1]** (same convention as upstream **`fhat_to_img`**).
- **Upstream bugfix**: **`autoregressive_infer_cfg`** had a hardcoded **`1024`** when reshaping the diffusion head output; it now uses **`self.last_level_pns`** (i.e. **`patch_nums[-1]²`**) so non-default patch schedules (e.g. final scale **12×12**) work.
- **Reference**: **`benchmarks/varsr/repo/test_varsr.py`** calls **`var.autoregressive_infer_cfg(...)`** with **`lr_inp`**, **`label_B`**, **`cfg`**, **`top_k`**, **`top_p`**. **`DiffLoss.sample`** in the repo still allocates on **`.cuda()`** — run on GPU for inference.

Example:

```python
varsr_model.eval()
with torch.inference_mode():
    recon = varsr_model.generate(lr_batch, cfg=1.5, top_k=0, top_p=0.0, g_seed=42)
```

## 5. Unified Training Loop
- **Accelerate Integration**: We created `train_varsr.py` using the `accelerate` framework to ensure architectural fairness and consistency with the GeM-SR training regime.
- **Loss Computation**: VARSR appends a continuous token representation to the end of the VQVAE sequence for diffusion loss. We updated our training loop to correctly slice the discrete tokens for the cross-entropy loss and add the diffusion loss (`diff_loss`).
- **Optimizer & Scheduler**: We used `AdamW` and the same cosine annealing scheduler with warmup as GeM-SR.

## 6. Execution Commands
1. Train the LiDAR VQVAE tokenizer (outputs under **`benchmarks/varsr/outputs/vqvae/<run>/`**):

```bash
python benchmarks/varsr/train_vqvae.py --config configs/generator/lidar_ssmVQ.yaml --run 1
```

2. Train the VARSR Transformer on top of it. Use **`--run <name>`** so all checkpoints and W&B project files go under **`benchmarks/varsr/outputs/varsr/<name>/`**.

```bash
python benchmarks/varsr/train_varsr.py \
  --config configs/generator/lidar_ssmVQ.yaml \
  --run 1 \
  --vqvae_ckpt ./path/to/best_model_lpips.pth
```

Periodic weights: `var_step_<step>.pth`, full Accelerate state under `checkpoint-<step>/`, final `var_final.pth`.
