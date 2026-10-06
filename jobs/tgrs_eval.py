#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TGRS robustness hub driver.

Builds ``benchmark_hub.py`` argv from ``jobs/tgrs_eval.yaml`` + env:

    MODEL_ROOT          checkpoint tree (…/lidar_chm_models)
    GEMSR_DATA_ROOT     Maryland splits (train/validation/test)
    GEMSR_CONFIG        optional repo yaml override
    GEMSR_SAVE_PATH     optional save_path override
    GEMSR_EVAL_SPLIT    validation (default) or test
    EVAL_SEED           optional int (default 0)
    TGRS_EVAL_MANIFEST  optional path to yaml

Usage (from GeM-SR repo root, with MODEL_ROOT and GEMSR_DATA_ROOT exported, e.g. from a
local, gitignored jobs/env.local.sh):

    python jobs/tgrs_eval.py --parse-only   # CLI shape only; no files, no GPU
    python jobs/tgrs_eval.py --preflight    # ls data + ckpts; parse CLI; no GPU
    python jobs/tgrs_eval.py --print        # print argv
    python jobs/tgrs_eval.py                # run hub on validation/ (needs GPU)
    python jobs/tgrs_eval.py --skip-existing  # resume: skip models that already have JSON
    python jobs/tgrs_eval.py --eval-split test  # held-out test/; writes tgrs_robustness_test
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "tgrs_eval.yaml"
VAL_SAVE_DIRNAME = "tgrs_robustness"
TEST_SAVE_DIRNAME = "tgrs_robustness_test"
DEFAULT_INTERP_MODES = ("bilinear", "bicubic")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _load_manifest(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"[fail] manifest missing: {path}")
    return OmegaConf.load(str(path))


def _under_model_root(model_root: Path, rel: str) -> Path:
    rel_s = str(rel).replace("\\", "/").lstrip("/")
    return (model_root / rel_s).resolve()


def _gemsr_entry_str(
    spec: Any,
    model_root: Path,
    tok_yaml: Path,
    tok_ckpt: Path,
) -> str:
    ckpt_dir = _under_model_root(model_root, spec.ckpt_dir)
    yaml_p = _under_model_root(model_root, spec.yaml)
    preset = str(spec.preset)
    scale = spec.cfg_scale
    temps = str(spec.temps)
    return (
        f"{spec.key}::{ckpt_dir}::{preset}::{scale}"
        f"::{yaml_p}::{tok_yaml}::{tok_ckpt}::{temps}"
    )


def _save_path_for_split(save_path: str, eval_split: str) -> str:
    """Keep validation JSON under tgrs_robustness; test writes tgrs_robustness_test."""
    if str(eval_split).strip().lower() != "test":
        return save_path
    p = Path(save_path)
    if p.name == VAL_SAVE_DIRNAME:
        return p.with_name(TEST_SAVE_DIRNAME).as_posix()
    return save_path


def build_hub_argv(
    manifest: Any,
    model_root: Path,
    *,
    config: str,
    save_path: str,
    eval_seed: int,
    skip_existing: bool = False,
    eval_split: str = "validation",
    interpolation: Sequence[str] | None = None,
) -> List[str]:
    tok_yaml = _under_model_root(model_root, manifest.tokenizer_shared.yaml)
    tok_ckpt = _under_model_root(model_root, manifest.tokenizer_shared.ckpt)
    argv: List[str] = [
        "--config",
        config,
        "--save_path",
        save_path,
        "--eval_seed",
        str(int(eval_seed)),
        "--hr_percentile_json",
        str(Path(save_path) / "hr_percentiles.json"),
    ]
    if skip_existing:
        argv.append("--skip_existing")
    split = str(eval_split).strip().lower() or "validation"
    argv.extend(["--eval_split", split])
    modes = list(interpolation) if interpolation is not None else list(DEFAULT_INTERP_MODES)
    if modes:
        argv.append("--interpolation")
        argv.extend(str(m) for m in modes)
    for spec in manifest.gemsr_entries:
        argv.extend(
            [
                "--gemsr_entry",
                _gemsr_entry_str(spec, model_root, tok_yaml, tok_ckpt),
            ]
        )
    b = manifest.baselines
    argv.extend(
        [
            "--swinir_ckpt",
            str(_under_model_root(model_root, b.swinir)),
            "--varsr_ckpt",
            str(_under_model_root(model_root, b.varsr)),
            "--varsr_vqvae_ckpt",
            str(_under_model_root(model_root, b.varsr_vqvae)),
            "--fast_diffusion_ckpt",
            str(_under_model_root(model_root, b.flow)),
            "--fast_diffusion_steps",
            *[str(int(x)) for x in list(b.fast_diffusion_steps)],
            "--sr3_ckpt",
            str(_under_model_root(model_root, b.sr3)),
            "--sr3_ddim_steps",
            *[str(int(x)) for x in list(b.sr3_ddim_steps)],
        ]
    )
    for spec in manifest.tokenizer_entries:
        yaml_p = _under_model_root(model_root, spec.yaml)
        ckpt_p = _under_model_root(model_root, spec.ckpt)
        argv.extend(["--tokenizer_gemsr_entry", f"{spec.key}::{yaml_p}::{ckpt_p}"])
    argv.extend(
        [
            "--tokenizer_var_vqvae_ckpt",
            str(_under_model_root(model_root, manifest.tokenizer_var_vqvae)),
        ]
    )
    return argv


def required_paths(manifest: Any, model_root: Path, data_root: Path) -> List[Tuple[str, Path, str]]:
    """(label, path, kind) with kind in {file, dir}."""
    out: List[Tuple[str, Path, str]] = []
    for split in ("train", "validation", "test"):
        out.append((f"data split {split}", data_root / split, "dir"))

    tok_yaml = _under_model_root(model_root, manifest.tokenizer_shared.yaml)
    tok_ckpt = _under_model_root(model_root, manifest.tokenizer_shared.ckpt)
    out.append(("shared tok yaml", tok_yaml, "file"))
    out.append(("shared tok ckpt", tok_ckpt, "file"))

    seen_ckpt: set[str] = set()
    for spec in manifest.gemsr_entries:
        ckpt_dir = _under_model_root(model_root, spec.ckpt_dir)
        yaml_p = _under_model_root(model_root, spec.yaml)
        key = str(ckpt_dir)
        if key not in seen_ckpt:
            seen_ckpt.add(key)
            out.append((f"{spec.key} ckpt dir", ckpt_dir, "dir"))
            out.append((f"{spec.key} model.safetensors", ckpt_dir / "model.safetensors", "file"))
            out.append((f"{spec.key} model_1.safetensors", ckpt_dir / "model_1.safetensors", "file"))
        out.append((f"{spec.key} gen yaml", yaml_p, "file"))

    b = manifest.baselines
    for name in ("swinir", "varsr", "varsr_vqvae", "flow", "sr3"):
        out.append((name, _under_model_root(model_root, getattr(b, name)), "file"))

    for spec in manifest.tokenizer_entries:
        out.append((f"{spec.key} yaml", _under_model_root(model_root, spec.yaml), "file"))
        out.append((f"{spec.key} ckpt", _under_model_root(model_root, spec.ckpt), "file"))
    out.append(
        (
            "tokenizer_var_vqvae",
            _under_model_root(model_root, manifest.tokenizer_var_vqvae),
            "file",
        )
    )
    return out


def _npz_chm_pair(path: Path) -> Tuple[Any, Any]:
    import numpy as np

    with np.load(path) as z:
        names = set(z.files)
        lr_key = "chm_input" if "chm_input" in names else "input"
        hr_key = "chm_target" if "chm_target" in names else "target"
        if lr_key not in names or hr_key not in names:
            raise KeyError(f"{path.name} keys={sorted(names)}")
        return z[lr_key], z[hr_key]


def check_npz_splits(data_root: Path) -> List[str]:
    """Hub LazyCHMDataset reads *.npz with chm_input/chm_target (or input/target)."""
    missing: List[str] = []
    val_alt = data_root / "val"
    validation = data_root / "validation"
    if (not validation.is_dir()) and val_alt.is_dir():
        missing.append(
            f"found {val_alt} but hub loaders use split name 'validation' (not 'val')"
        )
        print(f"[missing] {missing[-1]}")
    for split in ("train", "validation", "test"):
        d = data_root / split
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.npz"))
        if not files:
            msg = f"{split}: no .npz files in {d}"
            print(f"[missing] {msg}")
            missing.append(msg)
            continue
        print(f"[ok] {split}: {len(files)} .npz in {d}")
    sample_dir = data_root / "validation"
    samples = sorted(sample_dir.glob("*.npz")) if sample_dir.is_dir() else []
    if samples:
        sample = samples[0]
        try:
            lr, hr = _npz_chm_pair(sample)
            lr_shape = tuple(lr.shape)
            hr_shape = tuple(hr.shape)
            print(f"[ok] sample {sample.name}: chm_input{lr_shape} chm_target{hr_shape}")
            if lr_shape != (32, 16) or hr_shape != (48, 48):
                msg = (
                    f"{sample}: expected chm_input (32, 16) and chm_target (48, 48), "
                    f"got {lr_shape} / {hr_shape}"
                )
                print(f"[missing] {msg}")
                missing.append(msg)
        except Exception as exc:
            msg = f"could not read CHM pair from {sample}: {exc}"
            print(f"[missing] {msg}")
            missing.append(msg)
    return missing


def preflight_files(rows: Sequence[Tuple[str, Path, str]]) -> List[str]:
    missing: List[str] = []
    for label, path, kind in rows:
        ok = path.is_dir() if kind == "dir" else path.is_file()
        if ok:
            print(f"[ok] {label}: {path}")
        else:
            print(f"[missing] {label}: {path}")
            missing.append(f"{label}: {path}")
    data_root = None
    for label, path, kind in rows:
        if label == "data split validation":
            data_root = path.parent
            break
    if data_root is not None:
        missing.extend(check_npz_splits(data_root))
    return missing


def parse_hub_argv(argv: List[str]) -> argparse.Namespace:
    sys.path.insert(0, str(REPO_ROOT))
    import benchmark_hub

    return benchmark_hub.build_arg_parser().parse_args(argv)


def resolve_settings(manifest: Any) -> Dict[str, Any]:
    model_root_s = _env("MODEL_ROOT")
    data_root_s = _env("GEMSR_DATA_ROOT")
    dummy = _env("TGRS_EVAL_DUMMY_ROOT")
    if dummy:
        model_root_s = model_root_s or dummy
        data_root_s = data_root_s or dummy
    if not model_root_s:
        raise SystemExit("[fail] MODEL_ROOT unset (export it, e.g. in a gitignored jobs/env.local.sh)")
    if not data_root_s:
        raise SystemExit("[fail] GEMSR_DATA_ROOT unset (Maryland train/validation/test)")
    config = _env("GEMSR_CONFIG") or str(manifest.config)
    save_path = _env("GEMSR_SAVE_PATH") or str(manifest.save_path)
    seed_s = _env("EVAL_SEED")
    eval_seed = int(seed_s) if seed_s else 0
    eval_split = _env("GEMSR_EVAL_SPLIT") or "validation"
    eval_split = eval_split.strip().lower() or "validation"
    if eval_split not in ("validation", "test"):
        raise SystemExit(f"[fail] GEMSR_EVAL_SPLIT must be validation or test, got {eval_split!r}")
    save_path = _save_path_for_split(save_path, eval_split)
    return {
        "model_root": Path(model_root_s),
        "data_root": Path(data_root_s),
        "config": config,
        "save_path": save_path,
        "eval_seed": eval_seed,
        "eval_split": eval_split,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TGRS hub eval driver")
    p.add_argument(
        "--manifest",
        default=_env("TGRS_EVAL_MANIFEST") or str(DEFAULT_MANIFEST),
        help="Path to jobs/tgrs_eval.yaml",
    )
    p.add_argument("--parse-only", action="store_true", help="Parse hub CLI; skip file checks and GPU")
    p.add_argument("--preflight", action="store_true", help="Check data/ckpt paths and parse CLI; no GPU")
    p.add_argument("--print", dest="print_argv", action="store_true", help="Print hub argv and exit")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pass --skip_existing to the hub (reuse save_path JSON; skip those GPU runs).",
    )
    p.add_argument(
        "--eval-split",
        choices=["validation", "test"],
        default=None,
        help="Hub --eval_split (default: GEMSR_EVAL_SPLIT or validation). test/ writes tgrs_robustness_test.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    os.chdir(REPO_ROOT)
    if args.parse_only and not _env("MODEL_ROOT") and not _env("TGRS_EVAL_DUMMY_ROOT"):
        os.environ["TGRS_EVAL_DUMMY_ROOT"] = str(REPO_ROOT / "_tgrs_parse_dummy")

    manifest = _load_manifest(Path(args.manifest))
    settings = resolve_settings(manifest)
    if args.eval_split:
        settings["eval_split"] = args.eval_split
        settings["save_path"] = _save_path_for_split(
            _env("GEMSR_SAVE_PATH") or str(manifest.save_path),
            args.eval_split,
        )
    skip_env = _env("TGRS_SKIP_EXISTING").lower() in ("1", "true", "yes")
    skip_existing = bool(args.skip_existing or skip_env)
    hub_argv = build_hub_argv(
        manifest,
        settings["model_root"],
        config=settings["config"],
        save_path=settings["save_path"],
        eval_seed=settings["eval_seed"],
        skip_existing=skip_existing,
        eval_split=settings["eval_split"],
    )

    if args.print_argv or not args.parse_only:
        print("python benchmark_hub.py " + " ".join(shlex.quote(a) for a in hub_argv))

    if args.parse_only or args.preflight or args.print_argv:
        parsed = parse_hub_argv(hub_argv)
        n_gemsr = len(parsed.gemsr_entry or [])
        n_tok = len(parsed.tokenizer_gemsr_entry or [])
        print(
            f"[ok] hub argparse: gemsr_entry={n_gemsr} tokenizer_gemsr_entry={n_tok} "
            f"sr3_ddim_steps={list(parsed.sr3_ddim_steps)} eval_seed={parsed.eval_seed} "
            f"eval_split={parsed.eval_split} save_path={parsed.save_path} "
            f"skip_existing={parsed.skip_existing} "
            f"interpolation={list(parsed.interpolation or [])}"
        )

    if args.parse_only or args.print_argv:
        return 0

    os.environ["GEMSR_DATA_ROOT"] = str(settings["data_root"])
    missing = preflight_files(required_paths(manifest, settings["model_root"], settings["data_root"]))
    if missing:
        print("[fail] preflight missing paths:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print("[ok] preflight: all listed paths exist")
    if args.preflight:
        return 0

    hub = REPO_ROOT / "benchmark_hub.py"
    cmd = [sys.executable, str(hub), *hub_argv]
    print(f"[run] cwd={REPO_ROOT}")
    return int(subprocess.call(cmd, cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
