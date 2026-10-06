# Model Training Tutorial

This guide explains how to configure and launch training runs for both the GeM-SR model and the vendored baseline models (SwinIR, SR3, Fast Diffusion, and VARSR).

All models in this repository are governed by Hydra-style YAML configuration files located in the `configs/` directory.

> **Training data**
> The training data is not stored in this repository. Build it from the public HHDC dataset ([`anfera236/HHDC`](https://huggingface.co/datasets/anfera236/HHDC)) with `jobs/extract_chm.py` (see the README, *Dataset Preparation*). Training expects `train/`, `validation/`, and `test/` subfolders of paired LR/HR `.npz` files under `dataset.infra.root_dir` (default `./data`), so either extract into `./data` or point `root_dir` at your `--out-chm` folder.
>
> Without that data, training stops with `FileNotFoundError: ... './data/train'`. To try the code without training, use `inference_example.py` or the visualization scripts with the small sample data shipped in `samples/`. Trained checkpoints are available upon request.

---

## 1. Understanding Configuration Files (YAML)

Configurations are structured into several main blocks. Here is an example of what you will see inside `configs/generator/lidar_ssmVQ.yaml` or `configs/tokenizer/lidar_tokenizerVQ.yaml`:

- `globals`: Defines high-level project constants like random `seed`, `resolution`, `per_gpu_batch_size`, and the base `learning_rate` which is inherited by all subsequent optimizer blocks.
- `dataset`: Points to the data loader infrastructure. By default, it expects paired LR/HR CHM `.npz` files produced by `jobs/extract_chm.py`.
- `experiment`: Dictates where logs, checkpoints, and metrics are saved.
- `model`: Defines the architecture. The `target` key dynamically loads the correct Python class (e.g., `src.modeling.generator.generator_merged.HierarchicalGenerator`).
- `training` / `finetuning`: Sets the loss functions, number of steps, and scheduler parameters.
- `optimizer`: Dictates weight decay, betas, epsilon, and learning rates.

### Modifying Configurations
You do not need to change the Python code to run ablation studies. You can just duplicate and modify a YAML file. 
For example, to change the batch size, just edit:
```yaml
globals:
  per_gpu_batch_size: 32 # Default is usually 64
```

---

## 2. Training the GeM-SR Model

GeM-SR is trained in **two stages**: Tokenizer (VQ or FSQ) training, followed by Generator (AR) training.

### Stage 1: Train the Tokenizer
The tokenizer compresses the $48 \times 48$ spatial image into discrete latent tokens.

```bash
# Example: Training the VQ Tokenizer
python training/train_tokenizer.py \
  --config configs/tokenizer/lidar_tokenizerVQ.yaml \
  --run 1
```
*Note: The `--run 1` argument appends `_1` to your experiment name folder to prevent overwriting previous runs.*

### Stage 2: Train the Generator
Once the tokenizer is trained, you use it to train the autoregressive (AR) generator.
You must update `configs/generator/lidar_ssmVQ.yaml` to point to the tokenizer you just trained:

```yaml
model:
  tokenizer:
    pretrained:
      path_to_config: "path/to/tokenizer/config.yaml"
      path_to_ckpt: "path/to/tokenizer/model.safetensors"
```

Then, launch the generator training:
```bash
python training/train_generator.py \
  --config configs/generator/lidar_ssmVQ.yaml \
  --run 1
```

---

## 3. Training Baseline Models

We provide clean training wrappers for all baseline models. They hook directly into the **same** YAML configuration structure and dataloaders to guarantee perfectly fair, identical data shuffling and learning rate warmups.

To train any baseline, you simply point its training script to a valid config file (usually you can reuse the generator configs since the dataloader rules are the same).

### Convolutional Baselines (SwinIR)
```bash
python benchmarks/swinir/train_swinir.py --config configs/generator/lidar_ssmVQ.yaml --run 1
```

### Diffusion Baselines (SR3, Fast Diffusion)
```bash
# SR3
python benchmarks/sr3/train_sr3.py --config configs/generator/lidar_ssmVQ.yaml --run 1

# Fast Diffusion (Flow Matching)
python benchmarks/fast_diffusion/train_fast_diffusion.py --config configs/generator/lidar_ssmVQ.yaml --run 1
```

### Discrete Generative Baseline (VARSR)
Like GeM-SR, VARSR is two-stage: a 1-channel LiDAR VQVAE tokenizer, then the autoregressive transformer.

```bash
# Stage 1: train the VQVAE (LPIPS + PatchGAN loss); saves outputs/vqvae/<run>/checkpoint-<step>/vqgan.pth
python benchmarks/varsr/train_vqvae.py --config configs/generator/lidar_ssmVQ.yaml --run 1

# Stage 2: train VARSR on top of the chosen VQVAE checkpoint
python benchmarks/varsr/train_varsr.py --config configs/generator/lidar_ssmVQ.yaml --run 1 \
  --vqvae_ckpt benchmarks/varsr/outputs/vqvae/1/checkpoint-<step>/vqgan.pth
```

*Note: `train_vqvae.py` has no validation loop. For the paper, the VQVAE checkpoint was chosen by validation-set LPIPS among the saved steps.*

---

## 4. Logging and Experiment Tracking (WandB / TensorBoard)

By default, all models in this repository are configured to log metrics, validation images, and gradient norms to **Weights & Biases (WandB)** to match the strict tracking used during development.

**If you use WandB:**
You will need to have an active WandB account and be authenticated in your terminal (`wandb login`). The scripts will automatically generate comprehensive dashboards.

**If you do not want to use WandB:**
You can easily fall back to local TensorBoard logging without modifying any Python code. Just change the `logger` parameter in your target `.yaml` configuration file:

```yaml
experiment:
  logger: "tensorboard" # Change from "wandb"
```
The logs will then be written locally in a `logs/` folder inside each run's output directory (for the generator, runs default to `./outputs/ssm_runs/`; override with the `WORKSPACE` environment variable).

---

## 5. Multi-GPU Training (Accelerate)

All scripts in this repository are natively written using HuggingFace `accelerate`. They will automatically run on a single GPU if you use standard `python script.py`. 

To train across multiple GPUs (e.g., 4 GPUs), use:
```bash
accelerate launch --num_processes 4 training/train_generator.py --config configs/generator/lidar_ssmVQ.yaml --run 1
```

*(Note: The global batch size will be `per_gpu_batch_size * num_processes`)*