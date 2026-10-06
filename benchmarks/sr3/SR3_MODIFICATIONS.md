# SR3 Modifications

This document outlines the modifications made to the original SR3 repository ([Janspiry/Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement), Apache-2.0) for benchmarking against GeM-SR on the 1-channel LiDAR dataset.

The upstream example face datasets under `repo/dataset/` were removed from the vendored copy; they are not used for LiDAR training or evaluation.

## Architecture and Wrapper (`train_sr3.py`)
- The UNet is registered only as `GaussianDiffusion.denoise_fn`, not also as `self.unet`, so `state_dict` does not duplicate the same parameters (avoids PyTorch’s “Removed shared tensor … while saving” message and keeps reloads clean).
- `load_state_dict` strips legacy `unet.*` keys when `diffusion.denoise_fn.*` is present, or maps `unet.*` → `diffusion.denoise_fn.*` for older checkpoints.
- Created `SR3Wrapper` to handle the 1-channel input and asymmetrical upscaling (32x16 -> 48x48).
- SR3 natively expects square images, so the wrapper bilinearly upsamples the low-resolution input to the target high-resolution shape before concatenating it with the noise target.
- Modified the UNet initialization to accept `in_channel=2` (1 channel for the upsampled LR, 1 channel for the noisy HR) and `out_channel=1`.
- The `forward` pass dynamically determines the output shape based on the `hr` tensor during training, or an optional `target_size` argument during inference, allowing for shape-agnostic generation.

## Diffusion Process (`repo/model/sr3_modules/diffusion.py`)
- Fixed an issue in `p_sample_loop` to return `img` instead of `ret_img[-1]` to ensure correct batch handling during inference/sampling.
- Explicitly set the loss function device via `self.diffusion.set_loss(device)` to prevent device mismatch errors during training.

## DDIM Sampling Addition
- Added a `ddim_sample` method to `SR3Wrapper` to allow for fast inference without retraining the model.
- The original SR3 model is trained with 2000 DDPM steps. The new DDIM sampler uses the pre-trained `alphas_cumprod` schedule to perform strided sampling (e.g., 50, 100, 500 steps) while maintaining high generation quality.
- This allows for direct benchmarking of diffusion steps vs. inference time against GeM-SR.

## Training Loop
- Implemented a unified training loop using `accelerate` with the same data pipeline, learning rate scheduling, gradient accumulation, and logging as GeM-SR. The optimizer is Adam with the diffusion-family settings in [`BENCHMARK_TRAINING_REGIME.md`](../BENCHMARK_TRAINING_REGIME.md).
- Added auto-resume functionality to automatically recover from the latest checkpoint.

## Execution Command
```bash
python benchmarks/sr3/train_sr3.py --config configs/generator/lidar_ssmVQ.yaml --run 1
```
