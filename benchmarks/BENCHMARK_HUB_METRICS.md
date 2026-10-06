# GeM-SR benchmark hub — metric definitions

This document describes how **``benchmark_hub.py``** computes each reported quantity. The implementation lives next to the repo root as ``benchmark_hub.py``; metrics are computed in ``evaluate_model`` after predictions and ground truth are aligned to **[0, 1]** (unless ``--eval_normalize_diffusion`` forces a diffusion-only [-1, 1] loader; see script docstring).

---

## Data range and alignment

- **Default loader:** ``build_dataloaders(..., normalize_diffusion=False)`` yields LR/HR as **normalized canopy height** in **[0, 1]** (height divided by ``GLOBAL_MAX_HEIGHT`` from ``src.data.lidar_dataloading``).
- **Diffusion baselines (SR3, fast diffusion):** wrappers map LR **→ [-1, 1]** inside ``forward`` and map SR **→ [0, 1]** for metrics, so tables remain comparable to CNN / AR models trained on [0, 1].
- **``prepare_sr_hr_for_metrics``:** clamps SR/HR to [0, 1] in the canonical case; if the loader is [-1, 1], HR is linearly mapped to [0, 1] while SR is assumed already in [0, 1] from the wrapper.

All **image fidelity** metrics below use these aligned tensors.

---

## PSNR (peak signal-to-noise ratio)

- **Definition:** ``PSNR = 10 · log10(1 / MSE)`` where **MSE** is the batch-averaged mean squared error between SR and HR in **[0, 1]** over the full validation loader (scalar ``mse_m.avg``).
- **Dynamic range:** Implicitly **1.0** (not 255), because inputs are floats in [0, 1].
- **Interpretation:** Higher is better; sensitive to large per-pixel errors.

---

## SSIM (structural similarity)

- **Implementation:** ``torchmetrics.image.StructuralSimilarityIndexMeasure`` with ``data_range=1.0``.
- **Per batch:** SSIM is computed on the **single-channel** CHM tensors (after alignment), then averaged with the same sample weighting as MSE (batch size).
- **Interpretation:** Higher is better (1 = identical); emphasizes local structure vs. pointwise MSE.

---

## RMSE / MAE / bias / R² (normalized and meters)

- **``rmse``:** ``sqrt(MSE)`` in **[0, 1]** space (dimensionless vs. the normalized CHM).
- **``mae_height_bins`` / ``rmse_height_bins``:** Same L1 / RMSE error multiplied by  
  ``CHM_PHYSICAL_METERS_SCALE = GLOBAL_MAX_HEIGHT × 0.5``  
  i.e. each normalized unit corresponds to **0.5 m** vertical resolution in the physical CHM convention used by the dataset (see ``GLOBAL_MAX_HEIGHT`` in ``lidar_dataloading``). This yields **meters** for reporting vertical error in physical units.
- **``bias_height_overall``:** Mean signed error (SR − HR) in meters. Negative = systematic under-prediction.
- **``r2``:** Coefficient of determination of SR vs HR in normalized height (1 = perfect linear match to GT; ``null`` if GT variance is ~0).
- **Stratified ``chm_physical_by_hr_quartile``:** Pixels are binned by **GT** height **range** in [0, 1] (indices ``floor(hr * 4)`` clamped to ``{0,1,2,3}``). This is **not** a population quantile. Q4 can have very small N on Maryland/SERC. Each bin now also reports MAE/RMSE/**bias**/R² and ``pixel_counts``.
- **``chm_physical_by_hr_percentile``:** Tails defined by empirical GT quantiles (default top 10/5/1%, i.e. HR ≥ p90/p95/p99). Thresholds are computed once from a GT-only histogram pass (or loaded from ``--hr_percentile_json``) and reused for every model. Large N by construction.
- **``--eval_seed``:** Applied at the start of each ``evaluate_model`` call so stochastic models (SR3 ``randn``, GeM-SR categorical sampling) are reproducible. Re-run with seeds 0, 1, 2 for variance on Q4 / tails.

---

## LPIPS (learned perceptual patch similarity)

- **Implementation:** ``lpips.LPIPS(net='vgg')`` — the **AlexNet-based LPIPS** variant is *not* used here; the **VGG backbone** version is used (``net='vgg'``).
- **Inputs:** Single-channel SR/HR are **replicated to 3 channels**, then mapped from **[0, 1]** to **[-1, 1]** via ``x * 2 - 1`` to match the LPIPS training distribution.
- **Reduction:** Mean over spatial dimensions and channels for each image; then averaged across the val set with batch-size weighting.
- **Interpretation:** Lower is better (more perceptually similar under the VGG feature metric).

---

## FID (Fréchet inception distance)

- **Implementation:** ``torchmetrics.image.fid.FrechetInceptionDistance`` with ``feature=2048`` and ``normalize=False``.
- **Preprocessing:** SR and HR (3× repeated, same as LPIPS) are scaled to **[0, 255]**, converted to **``uint8``**, and passed to the metric as **real** vs **fake** distributions. With ``normalize=False``, torchmetrics expects inputs in the **[0, 255]** uint8 range (not [0, 1] floats with ``normalize=True``).
- **Backbone:** Standard **Inception-v3** feature extractor (torchmetrics / torchvision stack) — the default FID pipeline used by torchmetrics.
- **Interpretation:** Lower is better; sensitive to distribution shift and blur; **not** a per-image distance (dataset-level statistic).

---

## Throughput (``fps``)

- **Procedure:** After constructing SSIM/LPIPS/FID modules, the script takes **one** validation batch for a **dummy** LR tensor, runs ``profile_model``, then runs the full eval loop (separate from the timing loop).
- **Warmup / timing:** ``PROFILE_WARMUP_RUNS`` (default 5) then ``PROFILE_TIMING_RUNS`` (default 20) forward passes of ``model(dummy_lr)`` **without** gradient.
- **CUDA:** Uses ``torch.cuda.Event`` elapsed time between start/end records, synchronized. Reported **fps** = ``B * 1000 / (elapsed_ms / num_runs)`` — **images per second** for batch size **B** of the first val batch.
- **Non-CUDA:** Uses ``time.perf_counter()`` wall time over ``num_runs``; **fps** = ``B * num_runs / elapsed`` (approximate).
- **GPU dependence:** Absolute fps and VRAM depend on **GPU model**, driver, clock, batch size **B**, and concurrent processes. Numbers are for **relative** comparison under fixed hardware and batching, not as a portable constant.

---

## Peak memory (``vram_mb``)

- **CUDA only:** After timing, resets peak memory stats, runs one extra forward, then reports ``torch.cuda.max_memory_allocated(device) / 1024²``.
- **Non-CUDA:** Reported as **0.0** (not applicable).

---

## GFLOPs (``gflops``)

- **Method:** If ``thop`` is installed and the model implements ``get_gflops(dummy_lr, device)``, that method’s return is used (each wrapper documents its own counting rules: e.g. SR3 multiplies one UNet MAC count by **NFE** or DDPM steps; VAR sums stage-wise profiles).
- **Otherwise:** ``thop.profile`` on a **deepcopy** of the model for a **single** ``forward(dummy_lr)`` (no multi-step scaling).
- **Interpretation:** MACs × 2 → FLOPs convention; **approximate** (e.g. attention with KV cache vs. dense profile). For apples-to-apples comparisons, use the same ``thop`` version and input shape.

---

## Parameter count (``params_m``)

- **Definition:** ``sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6``.
- **Note:** Frozen submodules (e.g. some VQGAN paths) may be excluded if ``requires_grad=False``.

---

## GeMSR-specific metadata

When evaluating the hierarchical SSM wrapper, ``results.json`` also includes:

- **``gemsr_eval_refiner_strength``** — passed through to the tokenizer’s ``features_to_pixels(..., refiner_strength=...)``.
- **``gemsr_eval_temps``** — dict of sampling temperatures ``root``, ``scale_0``, … used in ``ssm.generate``.

---

## Reproducibility checklist

1. Fix **random seed** in training and, where applicable, sampling (VAR / VARSR / VQGAN use stochastic decoding unless noted).
2. Record **GPU type**, **CUDA version**, **batch size** (first val batch sets profiling **B**), and **PyTorch / torchmetrics / lpips** versions.
3. For FID on small val sets, variance can be large; report **dataset split** and number of images.
