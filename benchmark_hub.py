from __future__ import annotations
import os
import sys

# Add project root to sys.path so we can import src.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -*- coding: utf-8 -*-
"""
GeM-SR benchmark hub — validation metrics for LiDAR CHM super-resolution baselines.

This script is the **publication-oriented** entry point: one file, grouped CLI flags, and the
same metric protocol as the historical ``new_eval_bench.py`` core (without large-region tiling,
transport metrics, oracle teacher-forcing, or random-init speed-only paths).

**Convention:** LR/HR from ``build_dataloaders(..., normalize_diffusion=False)`` are heights
normalized by ``GLOBAL_MAX_HEIGHT`` in **[0, 1]**. PSNR / SSIM / LPIPS / FID are computed in that
space after optional wrapper rescaling. Physical CHM errors in meters use ``CHM_PHYSICAL_METERS_SCALE``.

The hierarchical SSM evaluated here is loaded via ``training.ssm.trainV5.build_model_v5`` (same as
before); user-facing names in this module use **GeMSR** / ``gemsr_*`` CLI flags.

**Tokenizer-only validation:** ``--tokenizer_gemsr_entry`` (tokenizer training YAML + weights: VQ **or**
FSQ ``LidarTokenizerV5``; legacy ``modeling....`` class paths are normalized to ``gess.modeling....``)
and ``--tokenizer_var_vqvae_ckpt`` (VAR/VARSR VQ-VAE) iterate the val loader on **HR** reconstructions,
writing ``tokenizer_benchmark.json`` per result key (no full generative model).
"""

import argparse
import copy
import gc
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

try:
    import thop
except ImportError:
    thop = None

from src.data.lidar_dataloading import GLOBAL_MAX_HEIGHT, build_dataloaders
from src.utils.meter import AverageMeter
from training.train_generator import build_model_v5
from training.train_tokenizer import build_tokenizer_from_config
from src.modeling.generator.conditioner import extract_conditioner
from src.modeling.generator.generator_merged import HierarchicalGenerator

_CHM_VERTICAL_BIN_METERS = 0.5
CHM_PHYSICAL_METERS_SCALE = float(GLOBAL_MAX_HEIGHT) * _CHM_VERTICAL_BIN_METERS
PROFILE_WARMUP_RUNS = 5
PROFILE_TIMING_RUNS = 20

from benchmarks.import_isolation import (
    clear_benchmark_namespaces,
    insert_path_first,
    verify_vendor_namespaces,
)
from benchmarks.chm_physical_metrics import (
    DEFAULT_TAIL_QUANTILES,
    PERCENTILE_HIST_BINS,
    TAIL_LABELS,
    accumulate_hr_histogram,
    accumulate_physical_errors,
    finalize_acc,
    finalize_percentile_tails,
    finalize_quartiles,
    hr_quantile_thresholds_from_histogram,
    percentile_accs,
    quartile_accs,
    seed_torch,
    _empty_acc,
)

_BENCH_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# GeMSR (hierarchical SSM) benchmark presets — merged into YAML before build_model_v5
# ---------------------------------------------------------------------------
GEMSR_BENCHMARK_PRESETS: Dict[str, Any] = {
    "convnext_nocfg": {
        "model": {"generator_v5": {"params": {"paper_ablation": False, "fast": True}}}
    },
    "convnext_cfg": {
        "model": {
            "generator_v5": {
                "params": {
                    "paper_ablation": True,
                    "fast": False,
                    "backbone": "convnext",
                    "cfg_training": True,
                }
            }
        }
    },
    "transformer_nocfg": {
        "model": {
            "generator_v5": {
                "params": {
                    "paper_ablation": True,
                    "fast": False,
                    "backbone": "transformer",
                    "cfg_training": False,
                }
            }
        }
    },
    "legacy": {},
}


def _merge_gemsr_preset(cfg: Any, preset: str) -> Any:
    if preset == "legacy":
        return cfg
    if preset not in GEMSR_BENCHMARK_PRESETS:
        raise ValueError(
            f"Unknown GeMSR preset {preset!r}. Choose from: {sorted(GEMSR_BENCHMARK_PRESETS.keys())}"
        )
    ov = GEMSR_BENCHMARK_PRESETS[preset]
    if not ov:
        return cfg
    return OmegaConf.merge(cfg, OmegaConf.create(ov))


def _merge_gemsr_tokenizer_path_overrides(
    g_cfg: Any,
    path_to_config: Optional[str] = None,
    path_to_ckpt: Optional[str] = None,
) -> Any:
    dotlist: List[str] = []
    if path_to_config:
        dotlist.append(f"model.tokenizer.pretrained.path_to_config={path_to_config}")
    if path_to_ckpt:
        dotlist.append(f"model.tokenizer.pretrained.path_to_ckpt={path_to_ckpt}")
    if not dotlist:
        return g_cfg
    return OmegaConf.merge(g_cfg, OmegaConf.from_dotlist(dotlist))


def _parse_gemsr_cfg_scale_and_yaml(fourth: str, fifth: Optional[str]) -> Tuple[float, Optional[str]]:
    if fifth is not None:
        return float(fourth), fifth
    try:
        return float(fourth), None
    except ValueError:
        pass
    m = re.match(
        r"^([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)(?:::|:/|/)(.+)$",
        fourth.strip(),
    )
    if m:
        scale = float(m.group(1))
        path = m.group(2).strip()
        if path and not path.startswith("/"):
            path = "/" + path
        return scale, path
    raise ValueError(
        "Fourth --gemsr_entry field must be cfg_scale (float) or "
        "'SCALE::/path/to.yaml', 'SCALE:/path', or 'SCALE/path'. "
        f"Got: {fourth!r}"
    )


_GEMSR_TEMP_KEYS = ("root", "scale_0", "scale_1", "scale_2")


def _parse_gemsr_temps_arg(raw: Optional[str]) -> Optional[Dict[str, float]]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "--gemsr_temps requires exactly 4 comma-separated floats: "
            "root,scale_0,scale_1,scale_2. "
            f"Got {len(parts)} field(s) in {raw!r}"
        )
    out: Dict[str, float] = {}
    for k, p in zip(_GEMSR_TEMP_KEYS, parts):
        if not p:
            raise ValueError(f"--gemsr_temps: empty value for {k!r} in {raw!r}")
        out[k] = float(p)
    return out


def _maybe_pop_trailing_gemsr_entry_temps(
    parts: List[str],
) -> Tuple[List[str], Optional[Dict[str, float]]]:
    if len(parts) <= 4:
        return parts, None
    last = parts[-1].strip()
    try:
        parsed = _parse_gemsr_temps_arg(last)
    except ValueError:
        return parts, None
    if parsed is None:
        return parts, None
    return parts[:-1], parsed


def _gemsr_specs_from_args(
    args,
) -> List[
    Tuple[str, str, str, float, Optional[str], Optional[str], Optional[str], Optional[Dict[str, float]]]
]:
    """Parse ``--gemsr_entry`` rows and optional legacy ``--gemsr_ckpt``."""
    if getattr(args, "gemsr_entry", None):
        out: List[
            Tuple[str, str, str, float, Optional[str], Optional[str], Optional[str], Optional[Dict[str, float]]]
        ] = []
        for raw in args.gemsr_entry:
            parts = [p.strip() for p in raw.split("::")]
            parts, entry_temps = _maybe_pop_trailing_gemsr_entry_temps(parts)
            if len(parts) < 4:
                raise ValueError(
                    "Each --gemsr_entry must be "
                    "'ResultKey::/path/to_ckpt_dir::preset::cfg_scale[::optional_ssm.yaml]"
                    "[::tokenizer_config.yaml][::tokenizer_weights.safetensors][::root,s0,s1,s2_temps]'. "
                    f"Got {len(parts)} fields in: {raw!r}"
                )
            name, ckpt, preset = parts[0], parts[1], parts[2]
            fifth = parts[4] if len(parts) >= 5 else None
            scale, extra = _parse_gemsr_cfg_scale_and_yaml(parts[3], fifth)
            tok_cfg = parts[5] if len(parts) >= 6 and parts[5] else None
            tok_ckpt = parts[6] if len(parts) >= 7 and parts[6] else None
            out.append((name, ckpt, preset, scale, extra, tok_cfg, tok_ckpt, entry_temps))
        return out
    if args.gemsr_ckpt:
        return [
            (
                "GeMSR",
                args.gemsr_ckpt,
                "legacy",
                float(args.gemsr_cfg_scale),
                args.gemsr_cfg,
                None,
                None,
                None,
            )
        ]
    return []


def build_benchmark_val_loader(
    config,
    normalize_diffusion: bool = False,
    split: str = "validation",
):
    """Return the eval split from ``build_dataloaders`` (``validation`` or ``test``)."""
    split_n = str(split).strip().lower()
    if split_n not in ("validation", "test"):
        raise ValueError(f"--eval_split must be 'validation' or 'test', got {split!r}")
    if split_n == "test":
        return build_dataloaders(
            config.dataset, test_only=True, normalize_diffusion=normalize_diffusion
        )
    dl = build_dataloaders(config.dataset, normalize_diffusion=normalize_diffusion)
    if isinstance(dl, tuple) and len(dl) >= 3:
        return dl[2]
    if isinstance(dl, tuple) and len(dl) >= 2:
        return dl[1]
    return dl


def prepare_sr_hr_for_metrics(sr: torch.Tensor, hr: torch.Tensor, input_range_01: bool):
    if input_range_01:
        return torch.clamp(sr, 0.0, 1.0), torch.clamp(hr, 0.0, 1.0)
    hr = (torch.clamp(hr, -1.0, 1.0) + 1.0) * 0.5
    sr = torch.clamp(sr, 0.0, 1.0)
    return sr, hr


def _batch_hr(batch, device: torch.device) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch["hr"].to(device)
    return batch[1].to(device)


def compute_loader_hr_quantile_thresholds(
    val_loader,
    device: torch.device,
    metrics_input_range_01: bool,
    quantiles: Sequence[float] = DEFAULT_TAIL_QUANTILES,
) -> List[float]:
    """One GT-only pass; histogram CDF so all models share identical tail cutoffs."""
    hist = torch.zeros(PERCENTILE_HIST_BINS, dtype=torch.int64)
    for batch in tqdm(val_loader, desc="HR percentile scan (GT only)"):
        hr = _batch_hr(batch, device)
        if metrics_input_range_01:
            hr = torch.clamp(hr, 0.0, 1.0)
        else:
            hr = (torch.clamp(hr, -1.0, 1.0) + 1.0) * 0.5
        accumulate_hr_histogram(hr, hist)
    return hr_quantile_thresholds_from_histogram(hist, quantiles)


def load_or_compute_hr_percentile_thresholds(
    path: Optional[str],
    val_loader,
    device: torch.device,
    metrics_input_range_01: bool,
) -> List[float]:
    if path:
        p = Path(path)
        if p.is_file():
            payload = json.loads(p.read_text(encoding="utf-8"))
            thr = payload.get("thresholds_normalized") or payload.get("thresholds")
            if isinstance(thr, dict):
                return [float(thr[k]) for k in TAIL_LABELS]
            if isinstance(thr, list) and len(thr) == len(DEFAULT_TAIL_QUANTILES):
                return [float(x) for x in thr]
            raise ValueError(f"Unrecognized percentile cache schema in {p}")
    thresholds = compute_loader_hr_quantile_thresholds(val_loader, device, metrics_input_range_01)
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "quantiles": list(DEFAULT_TAIL_QUANTILES),
                    "labels": list(TAIL_LABELS),
                    "thresholds_normalized": {k: t for k, t in zip(TAIL_LABELS, thresholds)},
                    "hist_bins": PERCENTILE_HIST_BINS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[hub] wrote HR percentile thresholds to {p}")
    return thresholds


# --- Benchmark repo wrappers (vendored under benchmarks/) ---------------------------------
def import_edsr_wrapper():
    edsr_root = str(_BENCH_ROOT / "benchmarks" / "edsr" / "repo" / "src")
    clear_benchmark_namespaces()
    insert_path_first(edsr_root)
    from benchmarks.edsr.train_edsr import EDSRWrapper

    verify_vendor_namespaces(edsr_root)
    return EDSRWrapper


def import_swinir_wrapper():
    swinir_root = str(_BENCH_ROOT / "benchmarks" / "swinir" / "repo")
    clear_benchmark_namespaces()
    insert_path_first(swinir_root)
    from benchmarks.swinir.train_swinir import SwinIRWrapper

    verify_vendor_namespaces(swinir_root)
    return SwinIRWrapper


def import_sr3_wrapper():
    sr3_root = str(_BENCH_ROOT / "benchmarks" / "sr3" / "repo")
    clear_benchmark_namespaces()
    insert_path_first(sr3_root)
    from benchmarks.sr3.train_sr3 import SR3Wrapper

    verify_vendor_namespaces(sr3_root)
    return SR3Wrapper


def import_fast_diffusion_wrapper():
    sr3_root = str(_BENCH_ROOT / "benchmarks" / "sr3" / "repo")
    fast_diffusion_root = str(_BENCH_ROOT / "benchmarks" / "fast_diffusion")
    clear_benchmark_namespaces()
    insert_path_first(sr3_root)
    insert_path_first(fast_diffusion_root)
    from benchmarks.fast_diffusion.train_fast_diffusion import FastDiffusionWrapper

    verify_vendor_namespaces([sr3_root, fast_diffusion_root])
    return FastDiffusionWrapper


def import_var_wrapper():
    train_var = _BENCH_ROOT / "benchmarks" / "var" / "train_var.py"
    if not train_var.is_file():
        raise RuntimeError(
            "Plain VAR (--var_ckpt) needs benchmarks/var/, which is not in this repo. "
            "VAR-SR uses --varsr_ckpt and VQVAE from benchmarks/varsr/repo; omit --var_ckpt."
        )
    var_root = str(_BENCH_ROOT / "benchmarks" / "var" / "repo")
    clear_benchmark_namespaces()
    insert_path_first(var_root)
    from benchmarks.var.train_var import VARWrapper

    verify_vendor_namespaces(var_root)
    return VARWrapper


from benchmarks.eval_wrappers import (
    GeMSRWrapper,
    SR3EvalWrapper,
    FastDiffusionEvalWrapper,
    VAREvalWrapper,
    VARSREvalWrapper,
    VQGANTransformerEvalWrapper,
    ResizeInterpEval,
)

def import_vqgan_transformer_stack():
    taming_repo_root = str(_BENCH_ROOT / "benchmarks" / "taming-transformers" / "repo")
    taming_root = str(_BENCH_ROOT / "benchmarks" / "taming-transformers")
    clear_benchmark_namespaces()
    insert_path_first(taming_repo_root)
    insert_path_first(taming_root)
    from train_transformer import VQGANTransformerWrapper
    from train_vqgan import LiDARVQGAN

    verify_vendor_namespaces([taming_repo_root, taming_root])
    return VQGANTransformerWrapper, LiDARVQGAN


@torch.inference_mode()
def profile_model(
    model: nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    model_name: str,
):
    """Wall-clock throughput, peak memory (CUDA), GFLOPs estimate (thop), and parameter count."""
    model.eval()
    B = int(dummy_input.size(0))

    def _forward() -> torch.Tensor:
        return model(dummy_input)

    print(f"\n[Profiling] {model_name}")
    warmup = max(0, int(PROFILE_WARMUP_RUNS))
    num_runs = max(1, int(PROFILE_TIMING_RUNS))

    with torch.no_grad():
        for _ in tqdm(range(warmup), desc="warmup", leave=False):
            _ = _forward()

        if device.type == "cuda":
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in tqdm(range(num_runs), desc="timing", leave=False):
                _ = _forward()
            t1.record()
            torch.cuda.synchronize()
            fps = (B * 1000.0) / (t0.elapsed_time(t1) / num_runs)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _ = _forward()
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            t_start = time.perf_counter()
            for _ in tqdm(range(num_runs), desc="timing", leave=False):
                _ = _forward()
            elapsed = time.perf_counter() - t_start
            fps = (B * num_runs) / max(elapsed, 1e-9)
            peak_vram = 0.0

    try:
        if thop is None:
            raise ImportError("thop not installed")
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        if hasattr(model, "get_gflops"):
            gflops = float(model.get_gflops(dummy_input, device))
        else:
            temp_model = copy.deepcopy(model).to(device)
            macs, _ = thop.profile(temp_model, inputs=(dummy_input.to(device),), verbose=False)
            gflops = (macs * 2) / 1e9
            del temp_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        gflops = 0.0

    print(f"  -> fps={fps:.2f}  vram_mb={peak_vram:.1f}  gflops≈{gflops:.2f}  params_m={params_m:.2f}")
    return fps, peak_vram, gflops, params_m


def evaluate_model(
    model: nn.Module,
    val_loader,
    device: torch.device,
    model_name: str = "Model",
    *,
    metrics_input_range_01: bool = True,
    eval_seed: int = 0,
    percentile_thresholds: Optional[Sequence[float]] = None,
):
    """
    PSNR / SSIM / LPIPS / FID in **[0,1]**, CHM stratified errors in meters, plus ``profile_model`` scalars.

    See ``benchmarks/BENCHMARK_HUB_METRICS.md`` for definitions.

    ``eval_seed`` is applied at the start of this call (and therefore at the start of
    each model / SR3 step-count) so stochastic models are reproducible. VAR/VARSR
    ``g_seed`` stays ``None`` so it does not reset to the same image noise every batch;
    they follow the global PyTorch RNG seeded here.
    """
    import random

    random.seed(int(eval_seed))
    np.random.seed(int(eval_seed))
    seed_torch(int(eval_seed))

    if percentile_thresholds is None:
        percentile_thresholds = compute_loader_hr_quantile_thresholds(
            val_loader, device, metrics_input_range_01
        )
    percentile_thresholds = [float(t) for t in percentile_thresholds]

    model.eval()
    mse_m = AverageMeter()
    mae_m = AverageMeter()
    ssim_m = AverageMeter()
    lpips_m = AverageMeter()
    overall_acc = _empty_acc()
    q_acc = quartile_accs()
    p_acc = percentile_accs(len(percentile_thresholds))
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    batch = next(iter(val_loader))
    if isinstance(batch, dict):
        dummy_lr = batch["lr"].to(device)
    else:
        dummy_lr = batch[0].to(device)
    fps, vram, gflops, params = profile_model(model, dummy_lr, device, model_name)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"{model_name} val"):
            lr, hr = (
                (batch["lr"].to(device), batch["hr"].to(device))
                if isinstance(batch, dict)
                else (batch[0].to(device), batch[1].to(device))
            )
            sr = model(lr)
            sr, hr = prepare_sr_hr_for_metrics(sr, hr, metrics_input_range_01)

            mse_m.update(F.mse_loss(sr, hr).item(), lr.size(0))
            mae_m.update(F.l1_loss(sr, hr).item(), lr.size(0))
            accumulate_physical_errors(sr, hr, overall_acc, q_acc, p_acc, percentile_thresholds)
            ssim_m.update(ssim_metric(sr, hr).item(), lr.size(0))

            sr3, hr3 = sr.repeat(1, 3, 1, 1), hr.repeat(1, 3, 1, 1)
            lpips_m.update(lpips_fn(sr3 * 2 - 1, hr3 * 2 - 1).mean().item(), lr.size(0))
            fid_metric.update((hr3 * 255).byte(), real=True)
            fid_metric.update((sr3 * 255).byte(), real=False)

    psnr = 10 * math.log10(1.0 / max(mse_m.avg, 1e-12))
    rmse_norm = math.sqrt(mse_m.avg)
    overall_phys = finalize_acc(overall_acc, CHM_PHYSICAL_METERS_SCALE)
    res: Dict[str, Any] = {
        "psnr": psnr,
        "ssim": ssim_m.avg,
        "rmse": rmse_norm,
        "rmse_height_bins": rmse_norm * CHM_PHYSICAL_METERS_SCALE,
        "mae_height_bins": mae_m.avg * CHM_PHYSICAL_METERS_SCALE,
        "bias_height_overall": overall_phys["bias_m"],
        "r2": overall_phys["r2"],
        "eval_seed": int(eval_seed),
        "chm_physical_by_hr_quartile": finalize_quartiles(q_acc, CHM_PHYSICAL_METERS_SCALE),
        "chm_physical_by_hr_percentile": finalize_percentile_tails(
            p_acc,
            percentile_thresholds,
            TAIL_LABELS[: len(percentile_thresholds)],
            DEFAULT_TAIL_QUANTILES[: len(percentile_thresholds)],
            CHM_PHYSICAL_METERS_SCALE,
        ),
        "lpips": lpips_m.avg,
        "fid": float(fid_metric.compute().item()),
        "fps": fps,
        "vram_mb": vram,
        "gflops": gflops,
        "params_m": params,
    }
    meta = getattr(model, "eval_result_meta", None)
    if isinstance(meta, dict):
        res.update(meta)

    del lpips_fn, ssim_metric, fid_metric
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return res


def _entropy_from_indices(indices: torch.Tensor, vocab_size: int) -> float:
    """Shannon entropy (bits) of a discrete index tensor (same convention as tokenizer training)."""
    if indices.numel() == 0:
        return 0.0
    counts = torch.histc(indices.float(), bins=int(vocab_size), min=0, max=int(vocab_size) - 1)
    probs = counts / (counts.sum() + 1e-6)
    probs = probs[probs > 0]
    return float((-probs * torch.log2(probs)).sum().item())


def _metrics_pack_01(sr: torch.Tensor, hr: torch.Tensor, ssim_m, lpips_fn) -> Dict[str, float]:
    sr_c = torch.clamp(sr, 0.0, 1.0)
    hr_c = torch.clamp(hr, 0.0, 1.0)
    mse = F.mse_loss(sr_c, hr_c).item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    rmse = math.sqrt(max(mse, 0.0))
    ssim_v = ssim_m(sr_c, hr_c).item()
    s3, h3 = sr_c.repeat(1, 3, 1, 1), hr_c.repeat(1, 3, 1, 1)
    lp = lpips_fn(s3 * 2.0 - 1.0, h3 * 2.0 - 1.0).mean().item()
    return {
        "psnr": psnr,
        "ssim": ssim_v,
        "lpips": lp,
        "rmse": rmse,
        "rmse_height_m": rmse * CHM_PHYSICAL_METERS_SCALE,
        "mae_height_m": float(F.l1_loss(sr_c, hr_c).item()) * CHM_PHYSICAL_METERS_SCALE,
    }


@torch.no_grad()
def evaluate_gemsr_tokenizer_on_loader(
    tok: nn.Module,
    val_loader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    HR autoencoding with a GeMSR LiDAR tokenizer (``forward(..., test=True)``).

    Reports PSNR / SSIM / LPIPS / RMSE in **[0, 1]** and RMSE/MAE in **meters** for
    pre-refiner (``x_recon_raw`` clamped) and post-refiner outputs, plus mean entropy /
    vocab usage from ``info_dict['debug_stats']`` (same keys as tokenizer training).
    """
    tok.eval()
    pre_psnr, post_psnr = AverageMeter(), AverageMeter()
    pre_ssim, post_ssim = AverageMeter(), AverageMeter()
    pre_lp, post_lp = AverageMeter(), AverageMeter()
    pre_rmse, post_rmse = AverageMeter(), AverageMeter()
    pre_rmsem, post_rmsem = AverageMeter(), AverageMeter()
    pre_maem, post_maem = AverageMeter(), AverageMeter()
    dbg_acc: Dict[str, List[float]] = {}

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    for batch in tqdm(val_loader, desc="GeMSR tokenizer (HR recon)"):
        hr = batch["hr"].to(device) if isinstance(batch, dict) else batch[1].to(device)
        x_post, (x_pre_raw, _), info = tok(hr, test=True)
        x_pre = torch.clamp(x_pre_raw, 0.0, 1.0)
        x_post_c = torch.clamp(x_post, 0.0, 1.0)
        hr_c = torch.clamp(hr, 0.0, 1.0)
        n = hr.size(0)

        mp = _metrics_pack_01(x_pre, hr_c, ssim_metric, lpips_fn)
        for k, m in (
            ("psnr", pre_psnr),
            ("ssim", pre_ssim),
            ("lpips", pre_lp),
            ("rmse", pre_rmse),
            ("rmse_height_m", pre_rmsem),
            ("mae_height_m", pre_maem),
        ):
            m.update(mp[k], n)
        mp2 = _metrics_pack_01(x_post_c, hr_c, ssim_metric, lpips_fn)
        for k, m in (
            ("psnr", post_psnr),
            ("ssim", post_ssim),
            ("lpips", post_lp),
            ("rmse", post_rmse),
            ("rmse_height_m", post_rmsem),
            ("mae_height_m", post_maem),
        ):
            m.update(mp2[k], n)

        ds = info.get("debug_stats") or {}
        for k, v in ds.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                dbg_acc.setdefault(k, []).append(float(v))

    def _avg_meter(m: AverageMeter) -> float:
        return float(m.avg)

    dbg_means = {k: float(sum(vals) / len(vals)) for k, vals in dbg_acc.items() if vals}

    out: Dict[str, Any] = {
        "pre_refiner": {
            "psnr": _avg_meter(pre_psnr),
            "ssim": _avg_meter(pre_ssim),
            "lpips": _avg_meter(pre_lp),
            "rmse": _avg_meter(pre_rmse),
            "rmse_height_m": _avg_meter(pre_rmsem),
            "mae_height_m": _avg_meter(pre_maem),
        },
        "post_refiner": {
            "psnr": _avg_meter(post_psnr),
            "ssim": _avg_meter(post_ssim),
            "lpips": _avg_meter(post_lp),
            "rmse": _avg_meter(post_rmse),
            "rmse_height_m": _avg_meter(post_rmsem),
            "mae_height_m": _avg_meter(post_maem),
        },
        "debug_stats_mean_over_loader": dbg_means,
    }
    del lpips_fn, ssim_metric
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return out


def _load_vqvae_weights(vae: nn.Module, ckpt_path: str) -> None:
    path = str(ckpt_path)
    if path.endswith(".safetensors"):
        vae_ckpt = load_file(path)
    else:
        vae_ckpt = torch.load(path, map_location="cpu")
    if isinstance(vae_ckpt, dict) and "vqvae" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["vqvae"])
        return
    sd = vae_ckpt
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and any(k.startswith("vqvae.") for k in sd.keys()):
        sd = {k.replace("vqvae.", ""): v for k, v in sd.items() if k.startswith("vqvae.")}
    vae.load_state_dict(sd)


def _make_lidar_varsr_vqvae() -> nn.Module:
    from benchmarks.varsr.varsr_repo_imports import make_lidar_varsr_vqvae

    return make_lidar_varsr_vqvae()


def _load_var_vqvae_for_bench(
    ckpt_path: str,
    device: torch.device,
    *,
    v_patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 6, 12),
) -> nn.Module:
    del v_patch_nums  # architecture is fixed to the LiDAR VAR-SR tokenizer
    vae = _make_lidar_varsr_vqvae()
    _load_vqvae_weights(vae, ckpt_path)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


def _flatten_discrete_vqvae_indices(idx_out, n_scales: Optional[int] = None) -> torch.Tensor:
    """Pool discrete VAR/VAR-SR index maps for entropy.

    VAR-SR ``img_to_idxBl`` returns ``(gt_idx_Bl, idx_N_list)``; ``gt_idx_Bl``
    appends a continuous residual after the last scale. Original VAR returns a
    list of index tensors. Entropy uses discrete scales only (same slice as
    ``train_varsr.py``).
    """
    if isinstance(idx_out, tuple) and len(idx_out) == 2:
        gt_idx_Bl, _ = idx_out
    else:
        gt_idx_Bl = idx_out
    if n_scales is not None:
        gt_idx_Bl = gt_idx_Bl[: int(n_scales)]
    return torch.cat([x.reshape(-1) for x in gt_idx_Bl]).long()


@torch.no_grad()
def evaluate_var_vqvae_tokenizer_on_loader(
    vae: nn.Module,
    val_loader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    VAR/VARSR VQ-VAE: ``forward(..., ret_usages=True)`` on HR in **[0, 1]**.

    LiDAR VQVAE weights (VAR-SR tokenizer) are trained and validated with recon
    and HR both in **[0, 1]** (same as ``_validation_psnr_lpips``). Do **not** apply ``(x+1)/2``;
    that assumes a [-1, 1] decoder and badly skews metrics toward ~0.5–1.0 vs sparse CHM.
    """
    vae.eval()
    psnr_m = AverageMeter()
    ssim_m_ = AverageMeter()
    lp_m = AverageMeter()
    rmse_m = AverageMeter()
    rmsem_m = AverageMeter()
    mae_m = AverageMeter()
    ent_vals: List[float] = []
    usage_rows: List[List[float]] = []

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    V = int(vae.vocab_size)

    for batch in tqdm(val_loader, desc="VAR VQVAE tokenizer (HR recon)"):
        hr = batch["hr"].to(device) if isinstance(batch, dict) else batch[1].to(device)
        rec_raw, usages, _vq = vae(hr, ret_usages=True)
        rec = torch.clamp(rec_raw, 0.0, 1.0)
        hr_c = torch.clamp(hr, 0.0, 1.0)
        n = hr.size(0)
        mp = _metrics_pack_01(rec, hr_c, ssim_metric, lpips_fn)
        psnr_m.update(mp["psnr"], n)
        ssim_m_.update(mp["ssim"], n)
        lp_m.update(mp["lpips"], n)
        rmse_m.update(mp["rmse"], n)
        rmsem_m.update(mp["rmse_height_m"], n)
        mae_m.update(mp["mae_height_m"], n)

        v_patch = getattr(getattr(vae, "quantize", None), "v_patch_nums", None)
        n_scales = len(v_patch) if v_patch else None
        flat = _flatten_discrete_vqvae_indices(vae.img_to_idxBl(hr), n_scales)
        ent_vals.append(_entropy_from_indices(flat, V))
        if usages is not None:
            usage_rows.append([float(u) for u in usages])

    mean_ent = float(sum(ent_vals) / max(len(ent_vals), 1))
    usage_mean_by_scale: Optional[List[float]] = None
    if usage_rows:
        n_sc = len(usage_rows[0])
        usage_mean_by_scale = [
            float(sum(row[si] for row in usage_rows) / len(usage_rows)) for si in range(n_sc)
        ]

    out = {
        "reconstruction": {
            "psnr": float(psnr_m.avg),
            "ssim": float(ssim_m_.avg),
            "lpips": float(lp_m.avg),
            "rmse": float(rmse_m.avg),
            "rmse_height_m": float(rmsem_m.avg),
            "mae_height_m": float(mae_m.avg),
        },
        "entropy_bits_pooled_over_indices": mean_ent,
        "vocab_size": V,
        "code_usage_percent_per_scale_mean": usage_mean_by_scale,
    }
    del lpips_fn, ssim_metric
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return out


def _parse_tokenizer_gemsr_entry(raw: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in raw.split("::")]
    if len(parts) != 3:
        raise ValueError(
            "Each --tokenizer_gemsr_entry must be exactly "
            "'ResultKey::/path/to/tokenizer_training/config.yaml::/path/to/tokenizer_weights.safetensors'. "
            f"Got {len(parts)} fields in {raw!r}"
        )
    return parts[0], parts[1], parts[2]


def _sanitize_save_key(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key.strip()).strip("_")


def _reuse_sr_result(args, name: str, results: Dict[str, Any]) -> bool:
    """If ``--skip_existing`` and ``results.json`` is on disk, load it and skip the GPU run."""
    if not getattr(args, "skip_existing", False):
        return False
    path = os.path.join(args.save_path, name, "results.json")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        results[name] = json.load(f)
    print(f"[hub] skip existing {name}: {path}")
    return True


def _reuse_tok_result(args, key: str, tokenizer_on_disk: Dict[str, Dict[str, Any]]) -> bool:
    if not getattr(args, "skip_existing", False):
        return False
    path = os.path.join(args.save_path, key, "tokenizer_benchmark.json")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        tokenizer_on_disk[key] = json.load(f)
    print(f"[hub] skip existing tokenizer {key}: {path}")
    return True


def _hub_save_progress(
    save_path: str,
    results: Dict[str, Any],
    tokenizer_on_disk: Dict[str, Dict[str, Any]],
    *,
    label: str = "progress",
) -> None:
    """Write ``summary.json``, each ``results.json``, and each ``tokenizer_benchmark.json`` so far."""
    if not results and not tokenizer_on_disk:
        return
    os.makedirs(save_path, exist_ok=True)
    for name, metrics in results.items():
        out_dir = os.path.join(save_path, name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    for sk, metrics in tokenizer_on_disk.items():
        out_dir = os.path.join(save_path, sk)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tokenizer_benchmark.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    summary_payload: Dict[str, Any] = dict(results)
    for sk, metrics in tokenizer_on_disk.items():
        summary_payload[f"tokenizer::{sk}"] = metrics
    with open(os.path.join(save_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(
        f"[hub] saved {label}: {len(results)} SR + {len(tokenizer_on_disk)} tokenizer "
        f"-> {os.path.abspath(save_path)}"
    )


def _release_model(model: nn.Module, extras: Tuple[Any, ...] = ()) -> None:
    del model
    for x in extras:
        del x
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(
        description="GeM-SR benchmark hub: CHM SR validation metrics (see benchmarks/BENCHMARK_HUB_METRICS.md)."
    )

    # --- Global / dataloader -------------------------------------------------
    p.add_argument(
        "--config",
        type=str,
        default="configs/generator/lidar_ssmVQ.yaml",
        help="Dataset YAML (and default GeMSR SSM yaml when per-entry yaml omitted).",
    )
    p.add_argument("--save_path", type=str, default="benchmarks/evaluation", help="Root folder for per-model results.")
    p.add_argument(
        "--eval_split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Dataset split under GEMSR_DATA_ROOT (validation or test). Use a new --save_path for test.",
    )
    p.add_argument(
        "--interpolation",
        type=str,
        nargs="*",
        default=None,
        metavar="MODE",
        help="Parameter-free resize baselines (bilinear, bicubic, nearest). 32×16→48×48.",
    )
    p.add_argument(
        "--eval_normalize_diffusion",
        action="store_true",
        help="Val loader in [-1,1] (SR3 / fast diffusion only; do not mix with [0,1]-trained CNNs).",
    )
    p.add_argument(
        "--eval_seed",
        type=int,
        default=0,
        help="Seed applied at the start of each evaluate_model call (reproducible stochastic SR).",
    )
    p.add_argument(
        "--hr_percentile_json",
        type=str,
        default=None,
        help=(
            "Cache for GT height tail thresholds (top-10/5/1%%). Loaded if the file exists; "
            "otherwise computed in a GT-only pass and written here so every model shares cutoffs."
        ),
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse ``save_path/<name>/results.json`` (or tokenizer_benchmark.json) if present; skip that GPU run.",
    )

    # --- EDSR ---------------------------------------------------------------
    p.add_argument("--edsr_ckpt", type=str, default=None, help="Path to EDSR .pth checkpoint.")

    # --- SwinIR -------------------------------------------------------------
    p.add_argument("--swinir_ckpt", type=str, default=None, help="Path to SwinIR .pth checkpoint.")

    # --- SR3 (DDPM / DDIM) --------------------------------------------------
    p.add_argument("--sr3_ckpt", type=str, default=None, help="SR3 checkpoint (.pth).")
    p.add_argument("--sr3_ddpm", type=bool, default=False)
    p.add_argument(
        "--sr3_ddim_steps",
        type=int,
        nargs="+",
        default=[0],
        help="0 = full DDPM; else one eval run per DDIM step count.",
    )

    # --- Fast diffusion / rectified flow -----------------------------------
    p.add_argument("--fast_diffusion_ckpt", type=str, default=None, help="Fast diffusion / flow-matching .pth.")
    p.add_argument(
        "--fast_diffusion_steps",
        type=int,
        nargs="+",
        default=[4],
        help="NFE counts to benchmark (same weights, multiple runs).",
    )

    # --- VAR ----------------------------------------------------------------
    p.add_argument("--var_ckpt", type=str, default=None, help="VAR transformer .pth.")
    p.add_argument("--var_vqvae_ckpt", type=str, default=None, help="VQ-VAE weights (required with --var_ckpt).")
    p.add_argument("--var_depth", type=int, default=14)
    p.add_argument("--var_embed_dim", type=int, default=256)
    p.add_argument("--var_num_heads", type=int, default=4)
    p.add_argument("--var_top_k", type=int, default=900)
    p.add_argument("--var_top_p", type=float, default=0.95)

    # --- VARSR --------------------------------------------------------------
    p.add_argument("--varsr_ckpt", type=str, default=None, help="VARSR transformer .pth.")
    p.add_argument("--varsr_vqvae_ckpt", type=str, default=None, help="VQ-VAE path (falls back to --var_vqvae_ckpt).")
    p.add_argument("--varsr_depth", type=int, default=12)
    p.add_argument("--varsr_embed_dim", type=int, default=256)
    p.add_argument("--varsr_num_heads", type=int, default=4, help="Must match training.")
    p.add_argument("--varsr_mlp_ratio", type=float, default=4.0, help="FFN width ratio; must match training.")
    p.add_argument("--varsr_cfg", type=float, default=6.0, help="Classifier-free guidance scale at sampling.")
    p.add_argument("--varsr_top_k", type=int, default=900)
    p.add_argument("--varsr_top_p", type=float, default=0.95)

    # --- VQGAN + transformer -------------------------------------------------
    p.add_argument("--vqgan_ckpt", type=str, default=None, help="LiDARVQGAN decode weights.")
    p.add_argument("--vqgan_transformer_ckpt", type=str, default=None, help="AR transformer .safetensors.")
    p.add_argument("--vqgan_vocab_size", type=int, default=4096)
    p.add_argument("--vqgan_block_size", type=int, default=144)
    p.add_argument("--vqgan_embed_dim", type=int, default=256)
    p.add_argument("--vqgan_n_layer", type=int, default=19)
    p.add_argument("--vqgan_n_head", type=int, default=4)
    p.add_argument("--vqgan_temperature", type=float, default=1.0)
    p.add_argument("--vqgan_top_k", type=int, default=50)
    p.add_argument("--vqgan_top_p", type=float, default=0.9)
    p.add_argument("--vqgan_greedy", action="store_true")
    p.add_argument("--vqgan_start_token_id", type=int, default=0)

    # --- GeMSR (hierarchical SSM) -------------------------------------------
    p.add_argument(
        "--gemsr_ckpt",
        type=str,
        default=None,
        help="Legacy single run: directory with model.safetensors + model_1.safetensors. Prefer --gemsr_entry.",
    )
    p.add_argument(
        "--gemsr_cfg",
        type=str,
        default=None,
        help="GeMSR / SSM yaml (defaults to --config). Per-entry override via 5th :: field on --gemsr_entry.",
    )
    p.add_argument(
        "--gemsr_cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale for paper generator path (legacy --gemsr_ckpt only).",
    )
    p.add_argument(
        "--gemsr_layout",
        type=str,
        default="joint",
        choices=("joint", "separate"),
        help="Tokenizer layout flag passed into OmegaConf before build_model_v5.",
    )
    p.add_argument(
        "--gemsr_entry",
        action="append",
        default=None,
        help=(
            "Repeat per variant. Format: "
            "ResultKey::ckpt_dir::preset::cfg_scale[::ssm.yaml][::tokenizer.yaml][::tokenizer.safetensors]"
            "[::root,s0,s1,s2 temps]. "
            "presets: convnext_nocfg | convnext_cfg | transformer_nocfg | legacy."
        ),
    )
    p.add_argument(
        "--gemsr_tokenizer_config",
        type=str,
        default=None,
        help="Global override for model.tokenizer.pretrained.path_to_config (local tokenizer recipe yaml).",
    )
    p.add_argument(
        "--gemsr_tokenizer_ckpt",
        type=str,
        default=None,
        help="Global override for model.tokenizer.pretrained.path_to_ckpt.",
    )
    p.add_argument(
        "--gemsr_eval_refiner_strength",
        type=float,
        default=1.0,
        help="Passed to tokenizer features_to_pixels (1.0 = full refiner; 0 = ablate refiner residual).",
    )
    p.add_argument(
        "--gemsr_temps",
        type=str,
        default=None,
        help="Four floats: root,scale_0,scale_1,scale_2 sampling temperatures for rows without trailing ::temps.",
    )

    # --- Tokenizer-only (HR reconstruction; no generative SSM) --------------
    p.add_argument(
        "--tokenizer_gemsr_entry",
        action="append",
        default=None,
        help=(
            "Repeat per LiDAR tokenizer checkpoint (VQ or FSQ V5). Format (exactly three :: fields): "
            "'ResultKey::tokenizer_training/config.yaml::tokenizer_weights.safetensors'. "
            "Writes ``tokenizer_benchmark.json`` under ``--save_path/ResultKey/``."
        ),
    )
    p.add_argument(
        "--tokenizer_var_vqvae_ckpt",
        type=str,
        default=None,
        help=(
            "Path to VAR/VARSR VQ-VAE weights (.safetensors or .pth). Runs HR encode–decode on the val "
            "loader and writes ``tokenizer_benchmark.json`` under ``--save_path/VAR_VQVAE_tokenizer/``."
        ),
    )

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(args.config)
    data_root = os.environ.get("GEMSR_DATA_ROOT", "").strip()
    if data_root:
        config.dataset.infra.root_dir = data_root
        print(f"[hub] GEMSR_DATA_ROOT -> dataset.infra.root_dir={data_root}")

    wants_01_models = any(
        [
            args.edsr_ckpt,
            args.swinir_ckpt,
            args.gemsr_ckpt,
            args.gemsr_entry,
            args.var_ckpt,
            args.varsr_ckpt,
            args.vqgan_transformer_ckpt and args.vqgan_ckpt,
            bool(args.interpolation),
        ]
    )
    wants_tokenizer_bench = bool(args.tokenizer_gemsr_entry or args.tokenizer_var_vqvae_ckpt)
    wants_diffusion = bool(args.sr3_ckpt or args.fast_diffusion_ckpt)
    if args.eval_normalize_diffusion and (wants_01_models or wants_tokenizer_bench):
        raise ValueError(
            "--eval_normalize_diffusion is only valid for SR3 and/or fast diffusion alone; "
            "other baselines expect [0,1] from build_dataloaders(..., False)."
        )
    if args.eval_normalize_diffusion and not wants_diffusion:
        warnings.warn(
            "--eval_normalize_diffusion set but no diffusion checkpoint given; loader still uses [-1,1].",
            stacklevel=2,
        )

    eval_split = str(getattr(args, "eval_split", "validation") or "validation")
    val_loader = build_benchmark_val_loader(
        config, normalize_diffusion=args.eval_normalize_diffusion, split=eval_split
    )
    metrics_input_range_01 = not args.eval_normalize_diffusion
    diffusion_lr_range = "01" if metrics_input_range_01 else "pm1"

    print(
        f"[hub] eval_split={eval_split}  save_path={os.path.abspath(args.save_path)}"
    )
    print(
        f"[hub] tensors: {'[0,1] canonical' if metrics_input_range_01 else '[-1,1] HR→[0,1] for metrics'}; "
        f"diffusion LR wrapper mode: {diffusion_lr_range}"
    )

    results: Dict[str, Any] = {}
    tokenizer_on_disk: Dict[str, Dict[str, Any]] = {}
    percentile_thresholds = load_or_compute_hr_percentile_thresholds(
        args.hr_percentile_json, val_loader, device, metrics_input_range_01
    )
    print(
        "[hub] HR tail thresholds (normalized, shared): "
        + ", ".join(f"{lab}={t:.6f}" for lab, t in zip(TAIL_LABELS, percentile_thresholds))
    )
    eval_kw: Dict[str, Any] = {
        "metrics_input_range_01": metrics_input_range_01,
        "eval_seed": int(args.eval_seed),
        "percentile_thresholds": percentile_thresholds,
    }

    interp_modes = [str(m).strip().lower() for m in (args.interpolation or []) if str(m).strip()]
    _allowed_interp = {"bilinear", "bicubic", "nearest"}
    bad_interp = [m for m in interp_modes if m not in _allowed_interp]
    if bad_interp:
        raise ValueError(f"--interpolation modes must be in {_allowed_interp}, got {bad_interp}")
    for mode in interp_modes:
        tag = f"Interp_{mode}"
        print(f" -> {tag}")
        if not _reuse_sr_result(args, tag, results):
            model = ResizeInterpEval(mode=mode).to(device)
            results[tag] = evaluate_model(model, val_loader, device, tag, **eval_kw)
            _release_model(model)
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")

    if args.edsr_ckpt:
        print(" -> EDSR")
        if not _reuse_sr_result(args, "EDSR", results):
            EDSR = import_edsr_wrapper()
            model = EDSR(n_resblocks=32, n_feats=256).to(device)
            model.load_state_dict(torch.load(args.edsr_ckpt, map_location=device))
            results["EDSR"] = evaluate_model(model, val_loader, device, "EDSR", **eval_kw)
            _release_model(model)
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")

    if args.swinir_ckpt:
        print(" -> SwinIR")
        if not _reuse_sr_result(args, "SwinIR", results):
            SwinIR = import_swinir_wrapper()
            model = SwinIR(
                n_colors=1,
                embed_dim=256,
                depths=[6, 6, 6, 6, 6, 6],
                num_heads=[8, 8, 8, 8, 8, 8],
                window_size=8,
                mlp_ratio=2.0,
            ).to(device)
            model.load_state_dict(torch.load(args.swinir_ckpt, map_location=device))
            results["SwinIR"] = evaluate_model(model, val_loader, device, "SwinIR", **eval_kw)
            _release_model(model)
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")

    gemsr_specs = _gemsr_specs_from_args(args)
    gemsr_temps_global = _parse_gemsr_temps_arg(getattr(args, "gemsr_temps", None))
    if gemsr_temps_global is not None and gemsr_specs:
        print(f"[hub] global --gemsr_temps: {gemsr_temps_global}")
    if gemsr_specs:
        print(f" -> GeMSR ({len(gemsr_specs)} variant(s))")
        if args.gemsr_tokenizer_config or args.gemsr_tokenizer_ckpt:
            print(
                f"    tokenizer overrides: config={args.gemsr_tokenizer_config!r} ckpt={args.gemsr_tokenizer_ckpt!r}"
            )
        default_yaml = args.gemsr_cfg or args.config
        for res_key, ckpt_dir, preset, cfg_scale, yaml_ov, tok_c, tok_k, entry_temps in gemsr_specs:
            yaml_path = yaml_ov or default_yaml
            temps_wrap = entry_temps if entry_temps is not None else gemsr_temps_global
            print(f"\n   GeMSR[{res_key}] preset={preset} cfg_scale={cfg_scale} yaml={yaml_path}")
            if entry_temps is not None:
                print(f"      per-entry temps: {entry_temps}")
            elif gemsr_temps_global is not None:
                print(f"      temps (--gemsr_temps): {gemsr_temps_global}")
            if _reuse_sr_result(args, res_key, results):
                continue

            g_cfg = OmegaConf.load(yaml_path)
            g_cfg = _merge_gemsr_preset(g_cfg, preset)
            if args.gemsr_layout == "separate":
                g_cfg.model.generator_v5.params.tokenizer_layout = "separate"
            else:
                g_cfg.model.generator_v5.params.tokenizer_layout = "joint"
            g_cfg = _merge_gemsr_tokenizer_path_overrides(
                g_cfg,
                path_to_config=tok_c or args.gemsr_tokenizer_config,
                path_to_ckpt=tok_k or args.gemsr_tokenizer_ckpt,
            )

            ssm, pretrainer = build_model_v5(g_cfg, device=str(device), is_resume=False)
            ssm.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
            pretrainer.load_state_dict(load_file(os.path.join(ckpt_dir, "model_1.safetensors")), strict=True)
            model = GeMSRWrapper(
                ssm.eval(),
                extract_conditioner(pretrainer).eval(),
                cfg_scale=cfg_scale,
                refiner_strength=float(args.gemsr_eval_refiner_strength),
                temps=temps_wrap,
            ).to(device)
            results[res_key] = evaluate_model(model, val_loader, device, res_key, **eval_kw)
            _release_model(model, (ssm, pretrainer))
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")

    if args.var_ckpt:
        print(" -> VAR")
        if not args.var_vqvae_ckpt:
            raise ValueError("VAR requires --var_vqvae_ckpt")
        if not _reuse_sr_result(args, "VAR", results):
            VAR = import_var_wrapper()
            v_patch_nums = (1, 2, 3, 4, 6, 12)
            vae = _make_lidar_varsr_vqvae()
            _load_vqvae_weights(vae, args.var_vqvae_ckpt)
            vae = vae.to(device).eval()
            base = VAR(
                vae_local=vae,
                num_classes=1,
                depth=args.var_depth,
                embed_dim=args.var_embed_dim,
                num_heads=args.var_num_heads,
                patch_nums=v_patch_nums,
            ).to(device)
            base.load_state_dict(torch.load(args.var_ckpt, map_location=device), strict=True)
            model = VAREvalWrapper(base, top_k=args.var_top_k, top_p=args.var_top_p, g_seed=None).to(device)
            results["VAR"] = evaluate_model(model, val_loader, device, "VAR", **eval_kw)
            _release_model(model, (base, vae))
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")

    if args.varsr_ckpt:
        print(" -> VARSR")
        vqvae_path = args.varsr_vqvae_ckpt or args.var_vqvae_ckpt
        if not vqvae_path:
            raise ValueError("VARSR needs --varsr_vqvae_ckpt or --var_vqvae_ckpt")
        if not _reuse_sr_result(args, "VARSR", results):
            # Same import order as train_varsr.py: wrapper first (sys.path + VAR_RoPE), then VQVAE.
            from benchmarks.varsr.wrapper import VARSRWrapper

            v_patch_nums = (1, 2, 3, 4, 6, 12)
            vae = _make_lidar_varsr_vqvae()
            _load_vqvae_weights(vae, str(vqvae_path))
            vae.eval()
            for p in vae.parameters():
                p.requires_grad_(False)

            base_sr = VARSRWrapper(
                vae_local=vae,
                embed_dim=args.varsr_embed_dim,
                patch_nums=v_patch_nums,
                depth=args.varsr_depth,
                num_heads=args.varsr_num_heads,
                mlp_ratio=args.varsr_mlp_ratio,
            )
            base_sr.load_state_dict(torch.load(args.varsr_ckpt, map_location="cpu"), strict=True)
            model = VARSREvalWrapper(
                base_sr.to(device),
                cfg=args.varsr_cfg,
                top_k=args.varsr_top_k,
                top_p=args.varsr_top_p,
                g_seed=None,
            ).to(device)
            results["VARSR"] = evaluate_model(model, val_loader, device, "VARSR", **eval_kw)
            _release_model(model, (base_sr, vae))
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")

    if args.vqgan_transformer_ckpt and args.vqgan_ckpt:
        print(" -> VQGAN-Transformer")
        if not _reuse_sr_result(args, "VQGAN-Transformer", results):
            VQGT, LiDARVQGAN = import_vqgan_transformer_stack()
            vqgan = LiDARVQGAN().to(device)
            vqgan.load_state_dict(torch.load(args.vqgan_ckpt, map_location=device))
            vqgan.eval()
            transformer = VQGT(
                vocab_size=args.vqgan_vocab_size,
                block_size=args.vqgan_block_size,
                embed_dim=args.vqgan_embed_dim,
                n_layer=args.vqgan_n_layer,
                n_head=args.vqgan_n_head,
            ).to(device)
            transformer.load_state_dict(load_file(args.vqgan_transformer_ckpt))
            model = VQGANTransformerEvalWrapper(
                transformer,
                vqgan,
                temperature=args.vqgan_temperature,
                top_k=args.vqgan_top_k,
                top_p=args.vqgan_top_p,
                greedy=args.vqgan_greedy,
                start_token_id=args.vqgan_start_token_id,
            ).to(device)
            results["VQGAN-Transformer"] = evaluate_model(
                model, val_loader, device, "VQGAN-Transformer", **eval_kw
            )
            _release_model(model, (transformer, vqgan))
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")

    if args.tokenizer_gemsr_entry:
        if not metrics_input_range_01:
            raise ValueError(
                "Tokenizer benchmarks expect CHM HR in [0, 1] from the val loader; "
                "omit --eval_normalize_diffusion for --tokenizer_gemsr_entry."
            )
        for raw in args.tokenizer_gemsr_entry:
            key, cfg_p, w_p = _parse_tokenizer_gemsr_entry(raw)
            sk = _sanitize_save_key(key) or "tokenizer"
            print(f" -> GeMSR tokenizer-only ({sk})")
            if _reuse_tok_result(args, sk, tokenizer_on_disk):
                continue
            tok_cfg = OmegaConf.load(cfg_p)
            tok = build_tokenizer_from_config(tok_cfg).to(device)
            w_path = str(w_p)
            if w_path.endswith((".safetensors", ".sft")):
                sd = load_file(w_path)
            else:
                sd = torch.load(w_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            tok.load_state_dict(sd, strict=True)
            tok.eval()
            for p in tok.parameters():
                p.requires_grad_(False)
            metrics = evaluate_gemsr_tokenizer_on_loader(tok, val_loader, device)
            metrics["entry"] = {"result_key": key, "tokenizer_config": cfg_p, "weights": w_p}
            tokenizer_on_disk[sk] = metrics
            _release_model(tok)
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")

    if args.tokenizer_var_vqvae_ckpt:
        if not metrics_input_range_01:
            raise ValueError(
                "VAR VQ-VAE tokenizer benchmark expects HR in [0, 1]; omit --eval_normalize_diffusion."
            )
        print(" -> VAR VQ-VAE tokenizer-only")
        if not _reuse_tok_result(args, "VAR_VQVAE_tokenizer", tokenizer_on_disk):
            vae = _load_var_vqvae_for_bench(args.tokenizer_var_vqvae_ckpt, device)
            metrics = evaluate_var_vqvae_tokenizer_on_loader(vae, val_loader, device)
            metrics["entry"] = {"weights": args.tokenizer_var_vqvae_ckpt}
            tokenizer_on_disk["VAR_VQVAE_tokenizer"] = metrics
            _release_model(vae)
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")

    # Continuous diffusion last so GeM-SR / VAR-SR JSON is on disk first. DDPM (steps=0) is last among SR3.
    if args.fast_diffusion_ckpt:
        print(" -> Fast diffusion")
        fd_base = None
        for nfe in args.fast_diffusion_steps:
            tag = f"FastDiffusion_NFE{nfe}"
            print(f"   {tag}")
            if _reuse_sr_result(args, tag, results):
                continue
            if fd_base is None:
                FD = import_fast_diffusion_wrapper()
                fd_base = FD(n_colors=1, image_size=48, num_inference_steps=int(args.fast_diffusion_steps[0])).to(
                    device
                )
                fd_base.load_state_dict(torch.load(args.fast_diffusion_ckpt, map_location=device), strict=True)
            model = FastDiffusionEvalWrapper(fd_base, num_steps=int(nfe), lr_input_range=diffusion_lr_range).to(
                device
            )
            results[tag] = evaluate_model(model, val_loader, device, tag, **eval_kw)
            _release_model(model)
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")
        if fd_base is not None:
            _release_model(fd_base)

    if args.sr3_ckpt:
        print(" -> SR3")
        sr3_base = None
        for steps in args.sr3_ddim_steps:
            tag = f"SR3_step{steps}" if steps > 0 else "SR3_DDPM"
            print(f"   {tag}")
            if _reuse_sr_result(args, tag, results):
                continue
            if sr3_base is None:
                SR3 = import_sr3_wrapper()
                sr3_base = SR3(n_colors=1, image_size=48).to(device)
                sr3_base.load_state_dict(torch.load(args.sr3_ckpt, map_location=device))
            model = SR3EvalWrapper(sr3_base, ddim_steps=steps, lr_input_range=diffusion_lr_range).to(device)
            results[tag] = evaluate_model(model, val_loader, device, tag, **eval_kw)
            _release_model(model)
            _hub_save_progress(args.save_path, results, tokenizer_on_disk, label="partial")
        if sr3_base is not None:
            _release_model(sr3_base)

    if results or tokenizer_on_disk:
        n_sr = len(results)
        n_tok = len(tokenizer_on_disk)
        print(
            f"\n[hub] finished {n_sr} SR run(s) and {n_tok} tokenizer run(s); "
            f"outputs under {os.path.abspath(args.save_path)}"
        )
    else:
        print("[hub] no checkpoints selected; nothing to run.")


if __name__ == "__main__":
    main()
