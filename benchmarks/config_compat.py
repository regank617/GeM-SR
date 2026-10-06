from __future__ import annotations
"""
Map GeSS ``lidar_ssmV5.yaml`` (and compatible layouts) to keys used by benchmark scripts.

**Optimizer families** (publication table — see ``BENCHMARK_TRAINING_REGIME.md``):

- **CNN** (EDSR, SwinIR): Adam, lr ``1e-4``, ``β=(0.9, 0.999)``, weight decay ``0``.
- **Diffusion** (SR3, Fast Diffusion): same as CNN.
- **Discrete generative** (VAR, VQGAN+Transformer, GeSS): AdamW, lr ``5e-4``, ``β=(0.9, 0.95)``, weight decay ``0.05``.

YAML ``optimizer.params`` overrides these defaults when present.
"""

from typing import Any, Dict, Tuple

from omegaconf import DictConfig, OmegaConf


def resolve_benchmark_training_keys(config: DictConfig) -> Dict[str, Any]:
    """Steps, logging, LR (for legacy flat keys), grad clip, mixed precision."""
    log_every = OmegaConf.select(config, "logging.log_every_n_steps")
    if log_every is None:
        log_every = OmegaConf.select(config, "experiment.intervals.log_steps", default=100)

    max_steps = OmegaConf.select(config, "training.max_steps")
    if max_steps is None:
        max_steps = OmegaConf.select(config, "training.duration.max_steps", default=200_000)

    warmup_steps = OmegaConf.select(config, "training.warmup_steps")
    if warmup_steps is None:
        warmup_steps = OmegaConf.select(config, "optimizer.scheduler.warmup_steps", default=10_000)

    save_every = OmegaConf.select(config, "training.save_every_n_steps")
    if save_every is None:
        save_every = OmegaConf.select(config, "experiment.intervals.save_every", default=2500)

    opt_lr = OmegaConf.select(config, "optimizer.learning_rate")
    if opt_lr is None:
        opt_lr = OmegaConf.select(config, "optimizer.params.lr", default=OmegaConf.select(config, "globals.learning_rate", default=1e-4))

    max_grad_norm = OmegaConf.select(config, "training.max_grad_norm")
    if max_grad_norm is None:
        max_grad_norm = OmegaConf.select(config, "training.compute.max_grad_norm", default=1.0)

    mixed_precision = OmegaConf.select(config, "globals.mixed_precision")
    if mixed_precision is None:
        mixed_precision = OmegaConf.select(config, "training.compute.mixed_precision", default="no")

    scheduler_name = OmegaConf.select(config, "optimizer.scheduler.name", default="cosine_with_minimum")

    return {
        "log_every_n_steps": int(log_every),
        "max_steps": int(max_steps),
        "warmup_steps": int(warmup_steps),
        "save_every_n_steps": int(save_every),
        "opt_lr": float(opt_lr),
        "max_grad_norm": float(max_grad_norm),
        "mixed_precision": str(mixed_precision) if mixed_precision is not None else "no",
        "scheduler_name": str(scheduler_name),
    }


def _betas_from_config(config: DictConfig, default: Tuple[float, float]) -> Tuple[float, float]:
    betas_cfg = OmegaConf.select(config, "optimizer.params.betas", default=list(default))
    if isinstance(betas_cfg, (list, tuple)) and len(betas_cfg) >= 2:
        return (float(betas_cfg[0]), float(betas_cfg[1]))
    return default


def resolve_cnn_adam_params(config: DictConfig) -> Dict[str, Any]:
    """EDSR, SwinIR: Adam, lr 1e-4, β=(0.9, 0.999), no weight decay."""
    lr = OmegaConf.select(config, "globals.cnn_learning_rate")
    if lr is None:
        lr = OmegaConf.select(config, "optimizer.params.cnn_lr")
    if lr is None:
        lr = OmegaConf.select(config, "optimizer.params.lr")
    if lr is None:
        lr = OmegaConf.select(config, "globals.learning_rate", default=1e-4)
    betas = _betas_from_config(config, (0.9, 0.999))
    eps = float(OmegaConf.select(config, "optimizer.params.eps", default=1e-8))
    return {"lr": float(lr), "betas": betas, "eps": eps, "weight_decay": 0.0}


def resolve_diffusion_adam_params(config: DictConfig) -> Dict[str, Any]:
    """SR3, Fast Diffusion: Adam, lr 1e-4, β=(0.9, 0.999), no weight decay (defaults like CNN)."""
    lr = OmegaConf.select(config, "globals.diffusion_learning_rate")
    if lr is None:
        lr = OmegaConf.select(config, "optimizer.params.diffusion_lr")
    if lr is None:
        lr = OmegaConf.select(config, "globals.cnn_learning_rate")
    if lr is None:
        lr = OmegaConf.select(config, "optimizer.params.lr")
    if lr is None:
        lr = OmegaConf.select(config, "globals.learning_rate", default=1e-4)
    betas = _betas_from_config(config, (0.9, 0.999))
    eps = float(OmegaConf.select(config, "optimizer.params.eps", default=1e-8))
    return {"lr": float(lr), "betas": betas, "eps": eps, "weight_decay": 0.0}


def resolve_discrete_adamw_params(config: DictConfig) -> Dict[str, Any]:
    """VAR, VQGAN+Transformer, GeSS yaml: AdamW, lr 5e-4, β=(0.9, 0.95), wd 0.05."""
    lr = OmegaConf.select(config, "optimizer.params.lr")
    if lr is None:
        lr = OmegaConf.select(config, "globals.learning_rate", default=5e-4)
    weight_decay = float(OmegaConf.select(config, "optimizer.params.weight_decay", default=0.05))
    betas = _betas_from_config(config, (0.9, 0.95))
    eps = float(OmegaConf.select(config, "optimizer.params.eps", default=1e-8))
    return {"lr": float(lr), "weight_decay": weight_decay, "betas": betas, "eps": eps}


def resolve_gess_optimizer_params(config: DictConfig) -> Dict[str, Any]:
    """Backward-compatible alias for discrete-generative (AdamW) defaults."""
    return resolve_discrete_adamw_params(config)


def ensure_benchmark_dataset_config(config: DictConfig) -> None:
    """
    Ensure ``build_dataloaders(config.dataset)`` has ``infra`` and ``loader`` (GeSS layout).

    Mutates ``config`` in place.
    """
    if OmegaConf.select(config, "dataset") is None:
        config.dataset = OmegaConf.create()

    data_path = OmegaConf.select(config, "globals.data_path", default="./data")

    if OmegaConf.select(config, "dataset.infra") is None:
        if OmegaConf.select(config, "infra") is not None:
            config.dataset.infra = config.infra
        else:
            config.dataset.infra = OmegaConf.create({
                "root_dir": data_path,
                "num_workers_per_gpu": 4,
                "pin_memory": True,
                "persistent_workers": True,
            })

    if OmegaConf.select(config, "dataset.loader") is None:
        config.dataset.loader = OmegaConf.create({
            "per_gpu_batch_size": OmegaConf.select(config, "globals.per_gpu_batch_size", default=8),
            "resolution": OmegaConf.select(config, "globals.resolution", default=48),
        })
