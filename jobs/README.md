# Data extraction and TGRS evaluation

| File | Purpose |
|---|---|
| `extract_chm.py` | Build paired LR/HR CHM `.npz` files from 3-D HHDC cubes (forward model + canopy height extraction). |
| `tgrs_eval.py` | Drive `benchmark_hub.py` for the paper's evaluation tables from `tgrs_eval.yaml` + environment variables. |
| `tgrs_eval.yaml` | Model/checkpoint manifest (paths relative to `MODEL_ROOT`; see `CHECKPOINT_MANIFEST.md`). |

Run everything from the repository root. Keep machine-specific paths out of committed files; a convenient
pattern is a gitignored `jobs/env.local.sh` that exports the variables below, which you `source` first.

## Data layout

Loaders expect split folders **`train/`, `validation/`, `test/`** (not `val/`), each holding `.npz` files with
`chm_input` `(32, 16)` and `chm_target` `(48, 48)` in raw height-bin units (loaders divide by 128).
Training reads the root from `dataset.infra.root_dir` in the YAML (default `./data`); evaluation reads
`GEMSR_DATA_ROOT`.

## 1. SERC (Maryland) from Hugging Face

The SERC cubes are published as [`anfera236/HHDC`](https://huggingface.co/datasets/anfera236/HHDC) with native,
spatially separated train/validation/test splits (do not reshuffle):

```bash
python -m jobs.extract_chm --source hf --hf-split all \
    --canopy-percentile 0.98 --out-chm /path/to/chm/serc_rh98
```

`--canopy-percentile` defaults to 0.98 (RH98); the P25/P50 variants use `0.25` / `0.50`. A different percentile
changes the target CHM, so train a new tokenizer and generator on that tree rather than reusing RH98 weights.

## 2. BLAN zero-shot site (NEON Blandy Experimental Farm, Virginia)

BLAN cubes are generated locally with [HHDC-Creator](https://github.com/Anfera/HHDC-Creator). Clone it next to
this repo (not inside it). Cube generation is RAM-heavy; run fewer parallel jobs if you run out of memory.

```bash
cd ..
git clone https://github.com/Anfera/HHDC-Creator.git
cd HHDC-Creator
python -m hhdc.main --site BLAN --config cube_config_sample.json \
  --download-dir "<data-root>/neon_raw" \
  --cube-output-dir "<data-root>/cubes" \
  --jobs 8 --tiles-per-batch 4
```

Cubes land in `<data-root>/cubes/BLAN/*.npz` (key `cube`, shape `[128, 48, 48]`). Extract RH98 CHMs with every
patch in `test/`; `--preflight-copies` copies a few into `train/` and `validation/` so `--preflight` sees all
three folders:

```bash
cd ../GeM-SR
python -m jobs.extract_chm --source dir --cube-dir "<data-root>/cubes/BLAN" \
  --all-to-test --canopy-percentile 0.98 --out-chm "<data-root>/chm/blan_rh98" \
  --preflight-copies 8 --device cpu
```

## 3. Evaluation (`tgrs_eval.py`)

| Variable | Meaning |
|---|---|
| `MODEL_ROOT` | Checkpoint root containing `ALL_MODELS/...` (checkpoints available upon request) |
| `GEMSR_DATA_ROOT` | Extracted CHM tree with `train/ validation/ test/` |
| `GEMSR_SAVE_PATH` | Optional output folder (default from `tgrs_eval.yaml`) |
| `GEMSR_EVAL_SPLIT` | `validation` (default) or `test` |
| `GEMSR_CONFIG` | Optional override of the repo YAML |
| `EVAL_SEED` | Optional integer seed (default 0) |

```bash
python jobs/tgrs_eval.py --parse-only        # CLI shape only; no files, no GPU
python jobs/tgrs_eval.py --preflight         # checks data + checkpoints exist; no GPU
python jobs/tgrs_eval.py --print             # print the benchmark_hub.py command
python jobs/tgrs_eval.py                     # validation split (GPU)
python jobs/tgrs_eval.py --eval-split test   # held-out test split (GPU)
```

- Validation results go to `benchmarks/evaluation/tgrs_robustness`; `--eval-split test` writes
  `benchmarks/evaluation/tgrs_robustness_test`. Bilinear/bicubic interpolation baselines are evaluated alongside the models.
- Each model writes `results.json` as soon as it finishes. `--skip-existing` reuses existing JSON to resume
  after an interruption; do not pass it on a first run, and use a new `GEMSR_SAVE_PATH` after changing metric code.
- For seed variance, rerun with `EVAL_SEED=1`, `2`, … and a new `GEMSR_SAVE_PATH` per seed.
- `--preflight` only checks that files exist; it does not prove the weights load or reproduce the paper numbers.

**BLAN zero-shot** (frozen SERC checkpoints):

```bash
export GEMSR_DATA_ROOT="<data-root>/chm/blan_rh98"
export GEMSR_SAVE_PATH="benchmarks/evaluation/tgrs_blan_zeroshot"
python jobs/tgrs_eval.py --eval-split test
```
