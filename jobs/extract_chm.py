#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract paired LR/HR CHM .npz files from 3-D HHDC cubes.

SERC (Maryland): use the Hugging Face dataset anfera236/HHDC with its native
train/validation/test splits (spatially separated; do not reshuffle). Example, P25 canopy:

    python -m jobs.extract_chm --source hf --hf-split all \\
        --canopy-percentile 0.25 --out-chm /path/to/chm/serc_rh25

Zero-shot transfer site (local *.npz cubes, e.g. NEON BLAN), everything into test/:

    python -m jobs.extract_chm --source dir --cube-dir /path/to/cubes/BLAN \\
        --all-to-test --canopy-percentile 0.98 \\
        --out-chm /path/to/chm/blan_rh98 --preflight-copies 8

A different canopy percentile changes the target CHM, so train a new tokenizer and
generator on that tree (GEMSR_DATA_ROOT) rather than reusing RH98 checkpoints.

Zero-shot hub eval on the extracted site:

    GEMSR_DATA_ROOT=/path/to/chm/blan_rh98
    GEMSR_SAVE_PATH=benchmarks/evaluation/tgrs_blan_zeroshot
    python jobs/tgrs_eval.py --eval-split test
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.canopy_plots import create_chm
from src.data.forward_model import LidarForwardImagingModel

HF_SPLITS = ("train", "validation", "test")
SPLIT_ALIASES = {"val": "validation"}

DTM_PERCENTILE = 0.05
FORWARD_KWARGS = dict(
    input_res_m=(2.0, 2.0),
    output_res_m=(3.0, 6.0),
    footprint_diameter_m=10.0,
    b=0.0,
    eta=0.0,
    ref_altitude=500.0,
    ref_photon_count=20.0,
)


def normalize_split_name(name: str) -> str:
    return SPLIT_ALIASES.get(name, name)


def ensure_split_dirs(root: Path, splits: Sequence[str] = HF_SPLITS) -> None:
    for split in splits:
        (root / split).mkdir(parents=True, exist_ok=True)


def discover_dir_splits(
    cube_dir: Path,
    *,
    all_to_test: bool,
) -> dict[str, list[Path]]:
    cube_dir = cube_dir.resolve()
    if not cube_dir.is_dir():
        raise FileNotFoundError(f"--cube-dir not found: {cube_dir}")

    if all_to_test:
        files = sorted(cube_dir.rglob("*.npz"))
        return {"test": files}

    split_files: dict[str, list[Path]] = {}
    for split in HF_SPLITS:
        sub = cube_dir / split
        if sub.is_dir():
            split_files[split] = sorted(sub.glob("*.npz"))

    if split_files:
        return split_files

    flat = sorted(cube_dir.glob("*.npz"))
    if flat:
        print(
            "[extract_chm] flat --cube-dir without train/validation/test subdirs; "
            "use --source hf for SERC splits or --all-to-test for BLAN.",
            file=sys.stderr,
        )
        return {"test": flat}

    raise FileNotFoundError(f"No .npz cubes under {cube_dir}")


def output_stem(source: str, split: str, idx: int, src_path: Path | None) -> str:
    if source == "hf":
        return f"chm_{idx:06d}.npz"
    assert src_path is not None
    if src_path.parent == src_path.parents[0]:
        return f"{src_path.stem}.npz"
    return f"{src_path.parent.name}_{src_path.stem}.npz"


def compute_chm_pair(
    args: tuple[np.ndarray, np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray]:
    cube_input, cube_target, canopy_percentile = args
    chm_input = create_chm(cube_input, canopy_percentile=canopy_percentile, dtm_percentile=DTM_PERCENTILE)[0]
    chm_target = create_chm(cube_target, canopy_percentile=canopy_percentile, dtm_percentile=DTM_PERCENTILE)[0]
    return chm_input.astype(np.float32), chm_target.astype(np.float32)


def make_chm_worker(canopy_percentile: float) -> Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    def _compute(args: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        cube_input, cube_target = args
        return compute_chm_pair((cube_input, cube_target, canopy_percentile))

    return _compute


class NpzCubeDataset(Dataset):
    def __init__(self, file_paths: Sequence[Path]):
        self.file_paths = list(file_paths)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        fpath = self.file_paths[idx]
        data = np.load(fpath)
        if "cube" not in data:
            raise KeyError(f"{fpath} missing 'cube' key; found {list(data.keys())}")
        cube = data["cube"].astype(np.float32)
        return torch.from_numpy(cube), str(fpath)


class HFCubeDataset(Dataset):
    def __init__(self, hf_dataset) -> None:
        self.hf_dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.hf_dataset[idx]
        cube = row["cube"]
        if not isinstance(cube, torch.Tensor):
            cube = torch.as_tensor(cube)
        return cube.float(), str(idx)


def write_batch(
    batch_args: list[tuple],
    *,
    out_chm: Path,
    out_cubes: Path | None,
) -> None:
    for split_name, fname, input_cube, target_cube, chm_input, chm_target in batch_args:
        np.savez_compressed(
            out_chm / split_name / fname,
            chm_input=chm_input,
            chm_target=chm_target,
        )
        if out_cubes is not None:
            np.savez_compressed(
                out_cubes / split_name / fname,
                input_cube=input_cube.astype(np.float32),
                target_cube=target_cube.astype(np.float32),
            )


def process_split(
    split_name: str,
    dataset: Dataset,
    *,
    source: str,
    out_chm: Path,
    out_cubes: Path | None,
    canopy_percentile: float,
    device: torch.device,
    batch_size: int,
    loader_workers: int,
    chm_workers: int,
    writer_pool: Pool,
    global_index_start: int = 0,
) -> int:
    print(f"\n[extract_chm] {split_name}: {len(dataset)} cubes")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
    )

    forward_model = LidarForwardImagingModel(**FORWARD_KWARGS).to(device)
    forward_model.eval()

    chm_pool = Pool(processes=chm_workers)
    batch_write_list: list[tuple] = []
    write_results = []
    global_index = global_index_start

    with torch.no_grad():
        for target_cubes, refs in tqdm(loader, desc=split_name):
            target_cubes = target_cubes.to(device)
            input_cubes = forward_model(target_cubes)

            target_np = target_cubes.cpu().numpy()
            input_np = input_cubes.cpu().numpy()
            batch_n = target_np.shape[0]

            chm_results = chm_pool.map(
                compute_chm_pair,
                [(inp, tgt, canopy_percentile) for inp, tgt in zip(input_np, target_np)],
            )

            for i in range(batch_n):
                src_path = Path(refs[i]) if source == "dir" else None
                fname = output_stem(source, split_name, global_index, src_path)
                batch_write_list.append(
                    (
                        split_name,
                        fname,
                        input_np[i],
                        target_np[i],
                        chm_results[i][0],
                        chm_results[i][1],
                    )
                )
                global_index += 1

            if len(batch_write_list) >= batch_size * 2:
                res = writer_pool.apply_async(
                    write_batch,
                    args=(batch_write_list,),
                    kwds={"out_chm": out_chm, "out_cubes": out_cubes},
                )
                write_results.append(res)
                batch_write_list = []

    if batch_write_list:
        res = writer_pool.apply_async(
            write_batch,
            args=(batch_write_list,),
            kwds={"out_chm": out_chm, "out_cubes": out_cubes},
        )
        write_results.append(res)

    for res in write_results:
        res.get()

    chm_pool.close()
    chm_pool.join()
    return global_index


def load_hf_dataset(split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace extract requires `pip install datasets`. "
            "Use --source dir for local .npz cubes."
        ) from exc
    return load_dataset("anfera236/HHDC", split=split)


def resolve_hf_splits(hf_split: str) -> list[str]:
    if hf_split == "all":
        return list(HF_SPLITS)
    split = normalize_split_name(hf_split)
    if split not in HF_SPLITS:
        raise ValueError(f"--hf-split must be one of {HF_SPLITS} or 'all', got {hf_split!r}")
    return [split]


def preflight_copies(out_chm: Path, n: int) -> None:
    if n <= 0:
        return
    test_dir = out_chm / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"preflight-copies needs {test_dir}")
    files = sorted(test_dir.glob("*.npz"))[:n]
    if not files:
        raise FileNotFoundError(f"No .npz files in {test_dir} for preflight-copies")
    for split in ("train", "validation"):
        dest_dir = out_chm / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            shutil.copy2(src, dest_dir / src.name)
    print(f"[extract_chm] preflight-copies: copied {len(files)} files from test/ -> train/ and validation/")


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    p = argparse.ArgumentParser(
        description="Extract paired CHM .npz files from 3-D HHDC cubes (forward model + create_chm).",
    )
    p.add_argument(
        "--source",
        choices=("hf", "dir"),
        default=defaults.get("source", "dir"),
        help="hf = anfera236/HHDC (SERC); dir = local *.npz with key 'cube'",
    )
    p.add_argument(
        "--hf-split",
        default=defaults.get("hf_split", "all"),
        choices=("train", "validation", "test", "all"),
        help="HF split(s) to extract (SERC only; preserves spatial splits)",
    )
    p.add_argument(
        "--cube-dir",
        type=Path,
        default=defaults.get("cube_dir"),
        help="Directory of *.npz cubes (dir source) or split subdirs train/validation/test",
    )
    p.add_argument(
        "--out-chm",
        type=Path,
        default=defaults.get("out_chm"),
        required=defaults.get("out_chm") is None,
        help="Output root, e.g. .../chm/serc_rh25 (writes train/validation/test/)",
    )
    p.add_argument(
        "--canopy-percentile",
        type=float,
        default=defaults.get("canopy_percentile", 0.98),
        help="Canopy height percentile (DTM fixed at 0.05). Default 0.98 (RH98).",
    )
    p.add_argument(
        "--save-3d",
        action="store_true",
        help="Also write degraded 3-D cubes (input_cube, target_cube)",
    )
    p.add_argument(
        "--out-cubes",
        type=Path,
        default=None,
        help="3-D cube output root (required when --save-3d)",
    )
    p.add_argument(
        "--all-to-test",
        action="store_true",
        help="BLAN zero-shot: every cube -> out-chm/test/ (no random split)",
    )
    p.add_argument(
        "--preflight-copies",
        type=int,
        default=0,
        metavar="N",
        help="After extract, copy first N test .npz into train/ and validation/ for --preflight",
    )
    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=defaults.get("device"),
        help="Forward-model device (default: cuda if available else cpu)",
    )
    p.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 64))
    p.add_argument("--chm-workers", type=int, default=defaults.get("chm_workers", 4))
    p.add_argument("--writer-workers", type=int, default=defaults.get("writer_workers", 2))
    p.add_argument(
        "--loader-workers",
        type=int,
        default=defaults.get("loader_workers", 4),
        help="DataLoader workers (dir/hf loading)",
    )
    return p


def run_extract(args: argparse.Namespace) -> None:
    out_chm = args.out_chm.resolve()
    out_cubes = args.out_cubes.resolve() if args.out_cubes else None
    if args.save_3d and out_cubes is None:
        raise SystemExit("--save-3d requires --out-cubes")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_chm] device={device} canopy_percentile={args.canopy_percentile}")

    if args.source == "hf":
        splits_to_run = resolve_hf_splits(args.hf_split)
    else:
        if args.cube_dir is None:
            raise SystemExit("--cube-dir is required for --source dir")
        split_map = discover_dir_splits(args.cube_dir, all_to_test=args.all_to_test)
        splits_to_run = list(split_map.keys())

    ensure_split_dirs(out_chm, splits_to_run)
    if out_cubes is not None:
        ensure_split_dirs(out_cubes, splits_to_run)

    with Pool(processes=args.writer_workers) as writer_pool:
        if args.source == "hf":
            for split in splits_to_run:
                hf_ds = load_hf_dataset(split)
                hf_ds.set_format(type="torch", columns=["cube"])
                dataset = HFCubeDataset(hf_ds)
                process_split(
                    split,
                    dataset,
                    source="hf",
                    out_chm=out_chm,
                    out_cubes=out_cubes,
                    canopy_percentile=args.canopy_percentile,
                    device=device,
                    batch_size=args.batch_size,
                    loader_workers=args.loader_workers,
                    chm_workers=args.chm_workers,
                    writer_pool=writer_pool,
                )
        else:
            split_map = discover_dir_splits(args.cube_dir, all_to_test=args.all_to_test)
            for split, paths in split_map.items():
                if not paths:
                    continue
                dataset = NpzCubeDataset(paths)
                process_split(
                    split,
                    dataset,
                    source="dir",
                    out_chm=out_chm,
                    out_cubes=out_cubes,
                    canopy_percentile=args.canopy_percentile,
                    device=device,
                    batch_size=args.batch_size,
                    loader_workers=args.loader_workers,
                    chm_workers=args.chm_workers,
                    writer_pool=writer_pool,
                )

    preflight_copies(out_chm, args.preflight_copies)
    print(f"\n[extract_chm] done -> {out_chm}")


def main(argv: Sequence[str] | None = None) -> None:
    os.chdir(REPO_ROOT)
    parser = build_parser()
    args = parser.parse_args(argv)
    run_extract(args)


if __name__ == "__main__":
    main()
