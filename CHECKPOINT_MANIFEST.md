# GeM-SR Checkpoint Manifest

This document maps the models evaluated in our TGRS manuscript to the checkpoints and configs used by the
evaluation hub (`benchmark_hub.py`). The authoritative, machine-readable version is
[`jobs/tgrs_eval.yaml`](jobs/tgrs_eval.yaml), which `jobs/tgrs_eval.py` turns into the exact
`benchmark_hub.py` command line.

**Trained checkpoints are not hosted publicly at this time; they are available upon request**
(contact regank@udel.edu). Paths below are relative to a checkpoint root that you point the `MODEL_ROOT`
environment variable at.

## Shared tokenizer

| Role | Config | Checkpoint (under `MODEL_ROOT`) |
|---|---|---|
| VQ tokenizer used by every GeM-SR generator row | `ALL_MODELS/vq_ablation/gess_tokenizer_vq_3/config.yaml` (trained from `configs/tokenizer/lidar_tokenizerVQ.yaml`) | `ALL_MODELS/vq_ablation/gess_tokenizer_vq_3/logs/checkpoint-60000/model.safetensors` |

## GeM-SR generator rows

All rows use generator config `configs/generator/lidar_ssmVQ.yaml`, preset `convnext_nocfg`, and CFG scale 1.0.
In the result keys, `A`/`U` = Ascending / Uniform cross-entropy scale weighting (a property of the trained
weights), and the last letter is the sampling-temperature preset: `_P` = `{0.4, 0.4, 0.6, 0.8}` (the paper's
*Remote Sensing* setting) and `_R` = `{1, 1, 1, 1}` (the paper's *Perceptual* setting).

| Result key | Generator checkpoint (under `MODEL_ROOT`) | Temperatures |
|---|---|---|
| `Gess_VQ_C_S_A_R` | `ALL_MODELS/vq_ablation/gess_gen_vq_joint_1/logs/best_lpips-step-40000` | 1.0, 1.0, 1.0, 1.0 |
| `Gess_VQ_C_S_A_P` | `ALL_MODELS/vq_ablation/gess_gen_vq_joint_1/logs/best_lpips-step-40000` | 0.4, 0.4, 0.6, 0.8 |
| `Gess_VQ_C_S_U_R` | `ALL_MODELS/vq_ablation/gess_gen_vq_joint_2/logs/best_lpips-step-40000` | 1.0, 1.0, 1.0, 1.0 |
| `Gess_VQ_C_S_U_P` | `ALL_MODELS/vq_ablation/gess_gen_vq_joint_2/logs/best_lpips-step-40000` | 0.4, 0.4, 0.6, 0.8 |
| `Gess_VQ_T_L_U_R` | `ALL_MODELS/vq_ablation/gess_gen_vq_joint_t_1/logs/best_lpips-step-30000` | 1.0, 1.0, 1.0, 1.0 |
| `Gess_VQ_T_L_U_P` | `ALL_MODELS/vq_ablation/gess_gen_vq_joint_t_1/logs/best_lpips-step-30000` | 0.4, 0.4, 0.6, 0.8 |

Each generator checkpoint directory carries its own `config.yaml` next to the weights.

## Baselines

| Model | Checkpoint (under `MODEL_ROOT`) | Notes |
|---|---|---|
| SwinIR | `ALL_MODELS/benchmarks/swinir/best_model_lpips.pth` | |
| SR3 | `ALL_MODELS/benchmarks/sr3/best_model_lpips.pth` | DDIM steps evaluated: 50, 100, 200, and full DDPM |
| Fast Diffusion (Flow Matching) | `ALL_MODELS/benchmarks/flow/best_model_lpips.pth` | 4 Euler steps |
| VARSR transformer | `ALL_MODELS/benchmarks/varsr/transformer/var_step_30000.pth` | |
| VARSR VQVAE tokenizer | `ALL_MODELS/benchmarks/varsr/tokenizer/best_model_lpips.pth` | Trained with `benchmarks/varsr/train_vqvae.py`; step chosen by validation LPIPS |

## Tokenizer ablations

Tokenizer-only rows are evaluated with `--tokenizer_gemsr_entry`. Checkpoints (under `MODEL_ROOT`,
`ALL_MODELS/tokenizer_ablation/<key>/`) each carry their own `config.yaml`; the repo configs
`configs/tokenizer/lidar_tokenizerVQ_no_ll.yaml`, `lidar_tokenizerVQ_vanilla.yaml`, and
`lidar_tokenizerVQ_indep.yaml` correspond to the `no_ll`, `vanilla`, and `indep` variants.

| Key | Checkpoint step |
|---|---|
| `tokenizer_no_ll` | 299000 |
| `tokenizer_vanilla` | 299000 |
| `tokenizer_indep` | 299000 |
| `tokenizer_biases` | 60000 |
| `tokenizer_jsrq1` | 60000 |
| `tokenizer_jsrq_fsq` | 174000 |

## Running the evaluation

```bash
export MODEL_ROOT=/path/to/lidar_chm_models     # contains ALL_MODELS/...
export GEMSR_DATA_ROOT=/path/to/chm/serc_rh98   # train/ validation/ test/ from jobs/extract_chm.py
python jobs/tgrs_eval.py --print                # show the benchmark_hub.py command it will run
python jobs/tgrs_eval.py --eval-split test      # held-out test split (needs a GPU)
```

The `gess_*` directory and key names are the original run names (the project's earlier working title) and are
kept as-is so these paths keep matching the trained runs.
