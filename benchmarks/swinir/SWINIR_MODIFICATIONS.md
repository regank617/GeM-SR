# SwinIR Modifications Log

This file tracks the exact modifications made to the original [SwinIR](https://github.com/JingyunLiang/SwinIR) repository (Apache-2.0) for the 1-channel LiDAR benchmarking task (32x16 -> 48x48).

The upstream example images under `repo/testsets/` and `repo/figs/` were removed from the vendored copy; they are not used for LiDAR training or evaluation (some links in `repo/README.md` therefore no longer resolve).

### 1. `benchmarks/swinir/repo/models/network_swinir.py`
- **Line 661-662**: Added logic to the `SwinIR` initialization class to handle `in_chans == 1`.
  - The original code hardcoded a 3-channel RGB mean normalization `(0.4488, 0.4371, 0.4040)`. 
  - We added an `elif in_chans == 1:` block that sets the mean to a single scalar `[0.5]`. Since the LiDAR data is bounded strictly between `[0, 1]`, centering it around 0.5 allows the Swin Transformer to train symmetrically without crashing due to channel mismatch during the input normalization step.

### 2. Wrapper (`train_swinir.py` — ``SwinIRWrapper``)
- SwinIR runs at LR with ``upscale=1``, ``upsampler=''`` (no pixelshuffle head).
- Deep trunk features ``[B, embed_dim, H_lr, W_lr]`` are taken **before** the internal ``conv_last`` image residual (same tensors as ``conv_after_body(forward_features(conv_first(x))) + conv_first(x)``).
- Features are upsampled to ``48×48`` (bicubic + conv refinement) and added to a **bicubic baseline** of the LR input (global residual); output is **clamped to [0, 1]**.
- The previous **pixelshuffle + stride-(2,1) conv** design is preserved in ``swinir_old.py`` as ``SwinIRWrapperLegacy``.

### 3. Training (`train_swinir.py`)
- Data: ``build_dataloaders(..., normalize_diffusion=False)`` → **[0, 1]** (same convention as GeM-SR).
- Validation: **PSNR / SSIM** (``data_range=1.0``) / **LPIPS** (VGG on inputs scaled to **[-1, 1]**); metrics logged via **Accelerate** (W&B when ``experiment.logger == "wandb"``).
- Checkpoints: ``initial_model.pth``, ``final_model.pth``, ``best_model_psnr.pth``, ``best_model_lpips.pth``, ``model_step_{100k}.pth``, plus Accelerate ``checkpoint-*`` for resume.

### 4. Training command
```bash
python benchmarks/swinir/train_swinir.py --config configs/generator/lidar_ssmVQ.yaml --run 1
```
