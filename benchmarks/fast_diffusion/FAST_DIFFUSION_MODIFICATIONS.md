# Fast Diffusion Benchmark Modifications

This benchmark adapts the SR3 UNet architecture to use a Rectified Flow (Flow Matching) formulation, allowing for extremely fast 1-4 step inference. This provides a direct comparison against GeM-SR's fast inference capabilities.

## Architecture & Modifications

1. **UNet Backbone:** We reuse the `UNet` implementation from the SR3 benchmark (`benchmarks/sr3/repo/model/sr3_modules/unet.py`). It is configured identically to the SR3 benchmark (concatenating the upsampled Low-Res image with the High-Res target/noise).
2. **Flow Matching:** We replaced the `GaussianDiffusion` module with a custom `FlowMatching` class (`benchmarks/fast_diffusion/flow_matching.py`).
   - **Training:** Learns the straight-line vector field between pure noise ($x_0 \sim \mathcal{N}(0, I)$) and the target data ($x_1$). The loss is the MSE between the predicted velocity and the target velocity ($x_1 - x_0$).
   - **UNet time input (SR3-compatible):** SR3 does **not** pass integer timesteps into the UNet. `GaussianDiffusion.p_losses` passes **`continuous_sqrt_alpha_cumprod`** in roughly **(0, 1]**. Raw flow time $t \in [0,1]$ would sit in a narrow band for `PositionalEncoding` (WaveGrad-style: `noise_level * exp(...)`). `FlowMatching` maps $t$ linearly from **min to max** of `sqrt(alphas_cumprod)` under the same **2000-step linear** schedule as SR3 so conditioning matches the scale the embedding was designed for.
   - **Inference:** Euler integration; outputs **clamped to [-1, 1]** to match diffusion normalization.
3. **Training Script:** `benchmarks/fast_diffusion/train_fast_diffusion.py` uses `build_dataloaders(..., normalize_diffusion=True)` for **[-1, 1]** data (same as SR3). Checkpoints: `initial_model.pth`, `final_model.pth`, `best_model_psnr.pth`, `best_model_lpips.pth`, `model_step_{100k}.pth`; validation logs **PSNR / SSIM / LPIPS** to W&B (via Accelerate) when `experiment.logger == wandb`.
4. **Wrapper:** The UNet is registered only as `FlowMatching.denoise_fn` (no duplicate `self.unet`), matching the SR3 checkpoint hygiene fix.

## Execution

To train the Fast Diffusion model:

```bash
python benchmarks/fast_diffusion/train_fast_diffusion.py --config configs/generator/lidar_ssmVQ.yaml --run 1 --steps 4
```

To evaluate the trained model with the benchmark hub:

```bash
python benchmark_hub.py --config configs/generator/lidar_ssmVQ.yaml \
  --save_path benchmarks/evaluation/results \
  --fast_diffusion_ckpt path/to/best_model_lpips.pth --fast_diffusion_steps 4
```
