"""Hub-style model loaders and mosaic inference for Figure 1 (GPU scripts only)."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_hub import (  # noqa: E402
    _load_vqvae_weights,
    _make_lidar_varsr_vqvae,
    _merge_gemsr_preset,
    import_fast_diffusion_wrapper,
    import_swinir_wrapper,
    import_sr3_wrapper,
)
from benchmarks.chm_physical_metrics import seed_torch  # noqa: E402
from benchmarks.eval_wrappers import (  # noqa: E402
    FastDiffusionEvalWrapper,
    GeMSRWrapper,
    SR3EvalWrapper,
    VARSREvalWrapper,
)
from benchmarks.import_isolation import clear_benchmark_namespaces  # noqa: E402
from figures.tiling import lr_tiles, remesh_tiles  # noqa: E402
from src.modeling.generator.conditioner import extract_conditioner  # noqa: E402
from training.train_generator import build_model_v5  # noqa: E402

EVAL_SEED = 0
TILE_BATCH = 1
MOSAIC_SAVE_ROOT = "evaluation/maryland_evals"

# Relative paths under MODEL_ROOT (jobs/tgrs_eval.yaml)
TGRS_PATHS = {
    "tokenizer_yaml": "ALL_MODELS/vq_ablation/gess_tokenizer_vq_3/config.yaml",
    "tokenizer_ckpt": "ALL_MODELS/vq_ablation/gess_tokenizer_vq_3/logs/checkpoint-60000/model.safetensors",
    "gemsr_ckpt_dir": "ALL_MODELS/vq_ablation/gess_gen_vq_joint_2/logs/best_lpips-step-40000",
    "gemsr_yaml": "ALL_MODELS/vq_ablation/gess_gen_vq_joint_2/config.yaml",
    "swinir_ckpt": "ALL_MODELS/benchmarks/swinir/best_model_lpips.pth",
    "varsr_ckpt": "ALL_MODELS/benchmarks/varsr/transformer/var_step_30000.pth",
    "varsr_vqvae_ckpt": "ALL_MODELS/benchmarks/varsr/tokenizer/best_model_lpips.pth",
    "flow_ckpt": "ALL_MODELS/benchmarks/flow/best_model_lpips.pth",
    "sr3_ckpt": "ALL_MODELS/benchmarks/sr3/best_model_lpips.pth",
}

GEMSR_TEMPS_RS = {"root": 0.4, "scale_0": 0.4, "scale_1": 0.6, "scale_2": 0.8}
GEMSR_TEMPS_PERC = {"root": 1.0, "scale_0": 1.0, "scale_1": 1.0, "scale_2": 1.0}
VARSR_CFG = 6.0
VARSR_TOP_K = 900
VARSR_TOP_P = 0.95
FLOW_NFE = 4
SR3_DDIM_STEPS = 200

_POSIX_PATH_TAG = "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath"
_WINDOWS_PATH_TAG = "tag:yaml.org,2002:python/object/apply:pathlib.WindowsPath"


def model_root(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    raw = os.environ.get("GEMSR_MODEL_ROOT") or os.environ.get("MODEL_ROOT")
    if not raw:
        raise FileNotFoundError(
            "Pass --model-root or set GEMSR_MODEL_ROOT to the lidar_chm_models tree "
            "before mosaic inference."
        )
    return Path(raw)


def apply_model_root(explicit: Path | str | None) -> None:
    """Set GEMSR_MODEL_ROOT for the current process (e.g. from CLI args)."""
    if explicit is not None:
        os.environ["GEMSR_MODEL_ROOT"] = str(Path(explicit).resolve())


def _path_from_yaml_apply(loader, node):
    parts = loader.construct_sequence(node)
    if not parts:
        return ""
    out = os.fspath(parts[0])
    for part in parts[1:]:
        out = os.path.join(out, os.fspath(part).strip("/\\"))
    return out


class _CrossPlatformYamlLoader(yaml.SafeLoader):
    pass


_CrossPlatformYamlLoader.add_constructor(_POSIX_PATH_TAG, _path_from_yaml_apply)
_CrossPlatformYamlLoader.add_constructor(_WINDOWS_PATH_TAG, _path_from_yaml_apply)


def _load_omega_yaml_cross_platform(path: str):
    path = os.fspath(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.load(f, Loader=_CrossPlatformYamlLoader)
    return OmegaConf.create(data)


_OMEGACONF_LOAD_ORIG = OmegaConf.load


def _omega_load_cross_platform(file_, *args, **kwargs):
    """OmegaConf.load wrapper: cluster YAMLs may embed pathlib.PosixPath (Linux paths)."""
    file_ = os.fspath(file_)
    try:
        return _OMEGACONF_LOAD_ORIG(file_, *args, **kwargs)
    except Exception as exc:
        if "PosixPath" not in str(exc) and "WindowsPath" not in str(exc):
            raise
    return _load_omega_yaml_cross_platform(file_)


def ensure_omega_cross_platform_load() -> None:
    """Patch OmegaConf.load once so build_model_v5 can read tokenizer YAMLs written on Linux when running on Windows."""
    if getattr(OmegaConf.load, "_gemsr_cross_platform", False):
        return
    OmegaConf.load = _omega_load_cross_platform  # type: ignore[method-assign]
    OmegaConf.load._gemsr_cross_platform = True  # type: ignore[attr-defined]


# Mosaic scripts import this module on Windows; patch before any hub YAML load.
ensure_omega_cross_platform_load()


def _under_model_root(rel: str, root: Path | None = None) -> Path:
    return model_root(root) / str(rel).replace("\\", "/").lstrip("/")


def _require_file(rel: str) -> Path:
    path = _under_model_root(rel)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _require_ckpt_dir(rel: str) -> Path:
    directory = _under_model_root(rel)
    for name in ("model.safetensors", "model_1.safetensors"):
        checkpoint = directory / name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    return directory


def _apply_eval_seed(seed: int = EVAL_SEED) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    seed_torch(int(seed))


@torch.inference_mode()
def infer_mosaic(
    model: torch.nn.Module,
    lr_full: torch.Tensor,
    device: torch.device,
    eval_seed: int = EVAL_SEED,
) -> torch.Tensor:
    _apply_eval_seed(eval_seed)
    tiles, n_h, n_w = lr_tiles(lr_full)
    model.eval().to(device)
    outputs = []
    for i in range(0, tiles.size(0), TILE_BATCH):
        batch = tiles[i : i + TILE_BATCH].to(device)
        outputs.append(model(batch).detach().cpu())
    recon_tiles = torch.cat(outputs, dim=0)
    return remesh_tiles(recon_tiles, lr_full.size(0), n_h, n_w)


def _save_mosaic_npy(
    recon: torch.Tensor,
    subdir: str,
    save_root: str,
    run_name: str = "run_01",
) -> str:
    out_dir = os.path.join(save_root, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}.npy")
    np.save(out_path, recon.squeeze().cpu().numpy())
    return out_path


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _fmt_elapsed(seconds: float) -> str:
    if seconds >= 60.0:
        return f"{seconds / 60.0:.1f} min ({seconds:.1f} s)"
    return f"{seconds:.1f} s"


def _set_gemsr_tokenizer_paths(g_cfg, path_to_config: str, path_to_ckpt: str):
    path_to_config = os.fspath(path_to_config)
    path_to_ckpt = os.fspath(path_to_ckpt)
    overrides = OmegaConf.create(
        {
            "model": {
                "tokenizer": {
                    "pretrained": {
                        "path_to_config": path_to_config,
                        "path_to_ckpt": path_to_ckpt,
                    }
                }
            }
        }
    )
    return OmegaConf.merge(g_cfg, overrides)


def load_gemsr_wrappers(device: torch.device):
    ckpt_dir = _require_ckpt_dir(TGRS_PATHS["gemsr_ckpt_dir"])
    yaml_path = os.fspath(_require_file(TGRS_PATHS["gemsr_yaml"]))
    tok_yaml = os.fspath(_require_file(TGRS_PATHS["tokenizer_yaml"]))
    tok_ckpt = os.fspath(_require_file(TGRS_PATHS["tokenizer_ckpt"]))

    g_cfg = _load_omega_yaml_cross_platform(yaml_path)
    g_cfg = _merge_gemsr_preset(g_cfg, "convnext_nocfg")
    g_cfg.model.generator_v5.params.tokenizer_layout = "joint"
    g_cfg = _set_gemsr_tokenizer_paths(g_cfg, tok_yaml, tok_ckpt)

    ssm, pretrainer = build_model_v5(g_cfg, device=str(device), is_resume=False)
    ssm.load_state_dict(load_file(str(ckpt_dir / "model.safetensors")), strict=True)
    pretrainer.load_state_dict(load_file(str(ckpt_dir / "model_1.safetensors")), strict=True)
    conditioner = extract_conditioner(pretrainer).eval()

    rs = GeMSRWrapper(
        ssm.eval(), conditioner, cfg_scale=1.0, refiner_strength=1.0, temps=GEMSR_TEMPS_RS
    ).to(device)
    perc = GeMSRWrapper(
        ssm, conditioner, cfg_scale=1.0, refiner_strength=1.0, temps=GEMSR_TEMPS_PERC
    ).to(device)
    return rs, perc


def load_swinir_wrapper(device: torch.device):
    clear_benchmark_namespaces()
    swinir_cls = import_swinir_wrapper()
    ckpt = str(_require_file(TGRS_PATHS["swinir_ckpt"]))
    model = swinir_cls(
        n_colors=1,
        embed_dim=256,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[8, 8, 8, 8, 8, 8],
        window_size=8,
        mlp_ratio=2.0,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


def load_flow_wrapper(device: torch.device):
    clear_benchmark_namespaces()
    fd_cls = import_fast_diffusion_wrapper()
    ckpt = str(_require_file(TGRS_PATHS["flow_ckpt"]))
    base = fd_cls(n_colors=1, image_size=48, num_inference_steps=FLOW_NFE).to(device)
    base.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    return FastDiffusionEvalWrapper(base, num_steps=FLOW_NFE, lr_input_range="01").to(device).eval()


def load_sr3_ddim200(device: torch.device):
    clear_benchmark_namespaces()
    sr3_cls = import_sr3_wrapper()
    ckpt = str(_require_file(TGRS_PATHS["sr3_ckpt"]))
    base = sr3_cls(n_colors=1, image_size=48).to(device)
    base.load_state_dict(torch.load(ckpt, map_location=device))
    return SR3EvalWrapper(base, ddim_steps=SR3_DDIM_STEPS, lr_input_range="01").to(device).eval()


def load_varsr_wrapper(device: torch.device):
    clear_benchmark_namespaces()
    from benchmarks.varsr.wrapper import VARSRWrapper

    vqvae_path = str(_require_file(TGRS_PATHS["varsr_vqvae_ckpt"]))
    varsr_ckpt = str(_require_file(TGRS_PATHS["varsr_ckpt"]))
    v_patch_nums = (1, 2, 3, 4, 6, 12)

    vae = _make_lidar_varsr_vqvae()
    _load_vqvae_weights(vae, vqvae_path)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    base_sr = VARSRWrapper(
        vae_local=vae,
        embed_dim=256,
        patch_nums=v_patch_nums,
        depth=12,
        num_heads=4,
        mlp_ratio=4.0,
    )
    base_sr.load_state_dict(torch.load(varsr_ckpt, map_location="cpu"), strict=True)
    return VARSREvalWrapper(
        base_sr.to(device),
        cfg=VARSR_CFG,
        top_k=VARSR_TOP_K,
        top_p=VARSR_TOP_P,
        g_seed=None,
    ).to(device).eval()


def _run_mosaic_stage(
    label: str,
    save_key: str,
    loader_fn: Callable[[torch.device], torch.nn.Module],
    lr_full: torch.Tensor,
    device: torch.device,
    save_root: str,
    outputs: dict,
    *,
    num_passes: int = 1,
    hr_full: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    print(f"\n[mosaic] === {label} ===")
    t_stage = time.perf_counter()

    print("[mosaic]   loading weights...")
    t0 = time.perf_counter()
    model = loader_fn(device)
    _sync_device(device)
    t_load = time.perf_counter() - t0

    tiles, _, _ = lr_tiles(lr_full)
    print(f"[mosaic]   inferring {tiles.size(0)} tiles × {num_passes} pass(es) (batch={TILE_BATCH})...")
    t0 = time.perf_counter()
    pass_recons = []
    pass_metrics = []
    for pass_idx in range(num_passes):
        recon = infer_mosaic(model, lr_full, device, eval_seed=EVAL_SEED + pass_idx)
        pass_recons.append(recon)
        run_name = f"run_{pass_idx + 1:02d}"
        out_path = _save_mosaic_npy(recon, save_key, save_root, run_name=run_name)
        print(f"[mosaic]   pass {pass_idx + 1}/{num_passes} -> {out_path}")
        if hr_full is not None:
            pass_metrics.append(_mosaic_diag_metrics(recon, hr_full))

    if num_passes > 1:
        stacked = torch.stack(pass_recons, dim=0)
        ensemble = stacked.mean(dim=0)
        ens_path = os.path.join(save_root, save_key, "ensemble_mean_output.npy")
        np.save(ens_path, ensemble.squeeze().cpu().numpy())
        if hr_full is not None:
            ens_metrics = _mosaic_diag_metrics(ensemble, hr_full)
            summary = {
                "model": save_key,
                "num_passes": num_passes,
                "per_pass": pass_metrics,
                "ensemble_mean": ens_metrics,
            }
            summary_path = os.path.join(save_root, save_key, "mosaic_diag_metrics.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print(f"[mosaic]   diagnostic metrics -> {summary_path}")

    _sync_device(device)
    t_infer = time.perf_counter() - t0
    recon = pass_recons[0]
    t_total = time.perf_counter() - t_stage
    outputs[save_key] = recon
    print(
        f"[mosaic]   done: load {_fmt_elapsed(t_load)}, "
        f"infer {_fmt_elapsed(t_infer)}, total {_fmt_elapsed(t_total)}"
    )
    return recon


def _mosaic_diag_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Diagnostic mosaic PSNR/SSIM/LPIPS only — never used for teaser overlay badges."""
    pred_np = pred.detach().float().cpu().numpy().squeeze()
    gt_np = gt.detach().float().cpu().numpy().squeeze()
    mse = float(np.mean((pred_np - gt_np) ** 2))
    psnr = float("inf") if mse < 1e-12 else 20 * math.log10(1.0) - 10 * math.log10(mse)
    return {"psnr": psnr, "mse": mse}


def run_maryland_mosaic(
    lr_full: torch.Tensor,
    *,
    hr_full: Optional[torch.Tensor] = None,
    save_root: str = MOSAIC_SAVE_ROOT,
    device: torch.device | None = None,
    num_passes: int = 1,
) -> Dict[str, torch.Tensor]:
    """Hard-stitch native 32×16→48×48 tiles; overlay numbers stay in figures.overlays."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Mosaic inference requires CUDA; run on a GPU node.")
    lr_full = lr_full.to(device)
    if hr_full is not None:
        hr_full = hr_full.to(device)
    outputs: Dict[str, torch.Tensor] = {}

    tiles, _, _ = lr_tiles(lr_full)
    print(
        f"[mosaic] start: {tiles.size(0)} native tiles, "
        f"save_root={save_root}, device={device}, eval_seed={EVAL_SEED}, num_passes={num_passes}"
    )
    t_all = time.perf_counter()

    stages = [
        ("SwinIR", "swinir", load_swinir_wrapper),
        ("Flow-Matching (NFE=4)", "flow", load_flow_wrapper),
        (f"SR3 DDIM-{SR3_DDIM_STEPS}", "sr3_ddim_200", load_sr3_ddim200),
        ("VAR-SR", "varsr", load_varsr_wrapper),
    ]
    for label, save_key, loader in stages:
        _run_mosaic_stage(
            label,
            save_key,
            loader,
            lr_full,
            device,
            save_root,
            outputs,
            num_passes=num_passes,
            hr_full=hr_full,
        )

    print("\n[mosaic] === GeM-SR (joint_2) ===")
    t_gem = time.perf_counter()
    print("[mosaic]   loading shared weights (RS + Perc temps)...")
    t0 = time.perf_counter()
    gemsr_rs, gemsr_perc = load_gemsr_wrappers(device)
    _sync_device(device)
    print(f"[mosaic]   load {_fmt_elapsed(time.perf_counter() - t0)}")

    for label, model, save_key in (
        ("GeM-SR (RS temps)", gemsr_rs, "gemsr_rs"),
        ("GeM-SR (Perc temps)", gemsr_perc, "gemsr_perc"),
    ):
        _run_mosaic_stage(
            label,
            save_key,
            lambda _device, m=model: m,
            lr_full,
            device,
            save_root,
            outputs,
            num_passes=num_passes,
            hr_full=hr_full,
        )

    print(f"[mosaic]   GeM-SR total {_fmt_elapsed(time.perf_counter() - t_gem)}")
    print(f"\n[mosaic] ALL DONE in {_fmt_elapsed(time.perf_counter() - t_all)}")
    return outputs
