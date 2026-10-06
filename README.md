# GeM-SR: First-Order Markov Wavelet Generation for Satellite LiDAR Canopy Height Super-Resolution

Code accompanying our manuscript submitted to **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**. This repository provides the data processing, model training, and evaluation code for GeM-SR, a hierarchical generative model for super-resolving satellite LiDAR Canopy Height Models (CHM).

## Features
- **GeM-SR:** A 2D conditional super-resolution model operating in the wavelet domain. Uses a DWT multi-scale pyramid and discrete tokenizers (FSQ/VQ) paired with an autoregressive ConvNeXt / Transformer backbone.
- **Vendored Baselines:** Code for training and evaluating several SR baselines (SwinIR, SR3, Fast Diffusion, VARSR) against LiDAR CHMs.
- **Unified Benchmark Hub:** A comprehensive evaluation script (`benchmark_hub.py`) to systematically extract evaluation metrics across all models (PSNR, SSIM, LPIPS, FID, physical error in meters, and stratified metrics).

## 1. Installation

This project is tested with **Python 3.10** and requires **PyTorch** (we recommend the CUDA 11.8 build for reproducibility, though CUDA 12.1 is also supported).

### Requirements
1. Clone the repository:
   ```bash
   git clone https://github.com/regank617/GeM-SR.git
   cd GeM-SR
   ```
2. Create and activate a virtual environment:
   ```bash
   conda create -n gemsr python=3.10 -y
   conda activate gemsr
   pip install --upgrade pip
   ```
3. Install PyTorch (CUDA 11.8 build):
   ```bash
   pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Force correct `xformers` build (Optional but recommended):
   ```bash
   pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
   ```

*Note on `xformers`: Our generative models (ConvNeXt + attention variants) and diffusion baselines can utilize `xformers` for memory-efficient attention. It is highly recommended for memory-heavy inference but remains optional. If `xformers` is installed, it will automatically be utilized where applicable.*

*Note on Python Path:* To ensure frictionless imports across the `src`, `training`, and `benchmarks` directories without needing `pip install -e .`, all entry scripts automatically inject the project root into `sys.path`.

*Note on baselines:* `requirements.txt` covers the core GeM-SR pipeline plus the SwinIR / SR3 / Fast Diffusion baselines. The vendored VARSR baseline (`benchmarks/varsr/repo/`) has its own, separately pinned dependency set — see `benchmarks/varsr/repo/requirements.txt` if you need to run it.

## 2. Dataset Preparation

The paired LR/HR CHM data used for training and evaluation is extracted from 3-D LiDAR "HHDC" cubes via `jobs/extract_chm.py`. For the SERC (Maryland) site, the cubes are published on the Hugging Face Hub as [`anfera236/HHDC`](https://huggingface.co/datasets/anfera236/HHDC) with native `train` / `validation` / `test` splits:

```bash
python -m jobs.extract_chm --source hf --hf-split all \
    --canopy-percentile 0.25 --out-chm /path/to/chm/serc_rh25
```

**What this does:**
- Loads the 3-D LiDAR HHDC cube(s), simulates the degradation pipeline (airborne high-res → satellite-equivalent low-res via `src/data/forward_model.py`), and computes the canopy height map at the requested percentile.
- Writes paired LR/HR `.npz` files into `train` / `validation` / `test` subdirectories under `--out-chm`. Training reads them from `dataset.infra.root_dir` in the YAML config (default `./data`), so extract into `./data` or point `root_dir` at your `--out-chm` folder.
- `--source dir` is also supported for a local directory of raw `.npz` cubes (e.g. a zero-shot regional cube not on the Hub); see the script's docstring (`python -m jobs.extract_chm --help`) for the full flag set.

## 3. Training
Training scripts are provided for both the GeM-SR model and the baselines.

- **Train Tokenizer:**
  ```bash
  python training/train_tokenizer.py --config configs/tokenizer/lidar_tokenizerVQ.yaml
  ```
- **Train Generator (SSM):**
  ```bash
  python training/train_generator.py --config configs/generator/lidar_ssmVQ.yaml
  ```

For baselines, execute the scripts located in their respective directories (e.g., `python benchmarks/swinir/train_swinir.py --config configs/generator/lidar_ssmVQ.yaml`).

## 4. Evaluation and Benchmarking

Evaluation relies on `benchmark_hub.py`. This script automatically scales generated CHMs back to physical meters and computes stratified metrics alongside FID and LPIPS.

**Manual evaluation for GeM-SR:**
```bash
python benchmark_hub.py \
  --config configs/generator/lidar_ssmVQ.yaml \
  --save_path benchmarks/evaluation/results \
  --gemsr_entry "GeMSR::./path/to/generator/ckpt::convnext_nocfg::1.0::./configs/generator/lidar_ssmVQ.yaml::./configs/tokenizer/lidar_tokenizerVQ.yaml::./path/to/tokenizer.safetensors::1.0,1.0,1.0,1.0"
```

For the TGRS hub-eval / zero-shot-transfer tables specifically, see `jobs/tgrs_eval.py` (invoked as `python jobs/tgrs_eval.py --eval-split test`, with `GEMSR_DATA_ROOT` / `GEMSR_SAVE_PATH` pointed at the extracted CHM tree — see the header of `jobs/extract_chm.py` for the full data-prep-to-eval pipeline).

## Custom Inference (Tutorial)

If you want to load the models (GeM-SR or any of the baselines) and run inference yourself on a batch of data without using the full `benchmark_hub.py` suite, we have provided an easy-to-read, heavily commented python tutorial:

```bash
python inference_example.py
```

This script demonstrates how to:
1. Load LR and HR data (`.npy` tensors) — by default the 64-sample batch shipped in `samples/test_batch/`.
2. Load GeM-SR (and its tokenizer/conditioner) from weights.
3. Load baseline models like SwinIR from weights.
4. Run inference to generate super-resolved CHMs and compute simple metrics.

## Checkpoint Manifest

For reproducible weights across our paper's tables, see [CHECKPOINT_MANIFEST.md](CHECKPOINT_MANIFEST.md) for matching model weights and configuration flags. **Trained checkpoints are not hosted publicly at this time — they are available upon request** (see [Code and Data Availability](#code-and-data-availability) below).

## Structure
- `src/`: Core library (data loaders, model architectures, utilities).
  - `src/modeling/generator/generator_merged.py`: Our unified Hierarchical Generator bridging ConvNeXt and Transformer variants.
- `training/`: Entry points for training tokenizers and generators.
- `configs/`: YAML configs for various runs (VQ, FSQ, Independent, etc.).
- `benchmarks/`: Code for baselines, `benchmark_hub.py`, and `eval_wrappers.py`.
- `jobs/`: Dataset extraction (`extract_chm.py`) and the TGRS evaluation driver (`tgrs_eval.py` + `tgrs_eval.yaml`).
- `figures/`: Shared library code (cube loading, CHM-pair extraction, plotting) used by the visualization scripts below and by `figures.ipynb`.
- `scripts/`: Standalone entry points that run GeM-SR inference to (re)generate the case-study outputs used by the visualization scripts (see below).
- `samples/`, `evaluation/maryland_evals/`: Small sample cubes, a 64-sample test batch, and precomputed reconstructions for the examples and case studies.
- `tests/`: Unit tests (`pytest tests/`).

## 5. Visualizations & Case Studies

Each case study has two tiers: a `scripts/` entry point that **regenerates** model outputs from scratch (requires a GPU, trained weights via `GEMSR_MODEL_ROOT`, and the relevant regional cube data), and a root-level script that **plots** already-generated outputs (lightweight, no GPU required). The input cubes and precomputed reconstructions for every case study below ship with the repo (`samples/`, `evaluation/maryland_evals/`), so all plotting scripts run out of the box.

An interactive walkthrough of the same figure-generation code is also available in [`figures.ipynb`](figures.ipynb).

### 🌍 Cross-Regional Ablation Study (California / Maryland / Florida)
- **Regenerate:** `scripts/run_cross_region_mosaic.py` runs GeM-SR inference over regional test cubes (`--cube-dir`, default `samples/test_cubes/`, requires CUDA and `GEMSR_MODEL_ROOT`) and writes `<region>_recon.npy` files next to the input cubes.
- **Plot:** `cross_regional_ablation.py` loads those regional cubes + reconstructions and produces the comparison figure.
  ```bash
  python cross_regional_ablation.py
  ```
- **Data:** the regional test cubes (`California_test_cube.npz`, `Florida_test_cube_2.npz`, `Maryland_test_cube.npz`) and their precomputed `*_recon.npy` reconstructions ship in `samples/test_cubes/`.

### 🏞️ SERC (Maryland) Mosaic (Fig. 1)
Precomputed reconstructions for this one **are** checked into the repo (`evaluation/maryland_evals/<model>/run_01.npy`), so it plots out of the box:
- **Plot:** `figures.ipynb` → `figures.plot_teaser.plot_teaser_from_disk()`.
- **Regenerate:** `scripts/run_maryland_mosaic.py` (`--model-root` / `--cube-dir`, requires CUDA) re-runs GeM-SR over `Maryland_test_cube.npz` and refreshes `evaluation/maryland_evals/`.

### 🛰️ Meta CHMv2 Comparison
`comparison_meta.py` (a thin wrapper around `scripts/run_meta_comparison.py`) compares a GeM-SR reconstruction to Meta's CHMv2, demonstrating the complementary nature of LiDAR-based ranging alongside optical RGB methods. It reads `maryland_cube.npz`, `maryland_coords.npz`, and a precomputed `gemsr_recon.npy` from `samples/meta_case_study/` (shipped with the repo), and automatically fetches the Meta GeoTIFF from their public S3 bucket (requires the AWS CLI).
```bash
# Example for Linux: sudo apt update && sudo apt install awscli -y
python comparison_meta.py
```
A previously generated `comparison_meta.png` is included in the repo root for reference.

### 🖼️ Visualize Samples
Generates visual grids of precomputed model reconstructions to PDF and prints a PSNR/SSIM/LPIPS table for the batch. Reads `ylr_batch.npy`, `xhr_batch.npy`, and per-model `*_recons.npy` files from `--samples_path` (default `samples/test_batch/`, shipped with the repo):
```bash
python visualize_samples.py
```

Optional arguments:
- `--num_rows <int>`: Total number of samples to visualize (default: `16`).
- `--rows_per_page <int>`: Number of samples per PDF page (default: `4`).
- `--random_sample`: Randomly select samples from the batch instead of the first $N$.
- `--samples_path <str>`: Directory with the precomputed `.npy` batch files (default: `samples/test_batch/`).
- `--save_prefix <str>`: Prefix for the saved PDF files (default: `batch_chm_visuals`). Pass `'None'` to disable saving.

## 6. Tutorials and Guides

- **[Training Tutorial](TRAINING_TUTORIAL.md)**: Detailed instructions on how to configure and train GeM-SR (both Tokenizer and Generator) and the vendored baselines from scratch.
- **[Custom Inference Example](inference_example.py)**: A heavily commented, standalone Python script demonstrating how to load models (GeM-SR and baselines) from weights and perform inference without the overhead of the benchmarking hub.

## Code and Data Availability

- **Code:** [https://github.com/regank617/GeM-SR](https://github.com/regank617/GeM-SR)
- **Dataset:** [`anfera236/HHDC`](https://huggingface.co/datasets/anfera236/HHDC) on the Hugging Face Hub.
- **Trained checkpoints:** Not hosted publicly at this time. Available upon request — contact regank@udel.edu.

## Citation

If you use this code, please cite:

```bibtex
@misc{regan2026gemsr,
  title  = {GeM-SR: First-Order Markov Wavelet Generation for Satellite LiDAR Canopy Height Super-Resolution},
  author = {Regan, Kyle M. and Ramirez-Jaime, Andres and Porras-Diaz, Nestor and Stephen, Mark and Yang, Guangning and Arce, Gonzalo R.},
  note   = {Manuscript submitted to IEEE Transactions on Geoscience and Remote Sensing},
  year   = {2026}
}
```

## License

GeM-SR is released under the [MIT License](LICENSE). The vendored baseline code under `benchmarks/*/repo/` keeps its original licenses (Apache-2.0 for SwinIR, SR3, and BasicSR; MIT for VARSR); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
