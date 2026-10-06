from __future__ import annotations
"""
AdamW parameter grouping for GeSS: weight decay only on appropriate weights.

Excludes (weight_decay = 0): biases, LayerNorm / RMSNorm / GroupNorm parameters,
and parameters whose name contains ``gamma`` (ConvNeXt-style scale).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def param_excluded_from_weight_decay(name: str, p: torch.nn.Parameter) -> bool:
    if not p.requires_grad:
        return True
    if p.ndim == 0:
        return True
    ln = name.lower()
    if ln.endswith(".bias"):
        return True
    if "layernorm" in ln or "layer_norm" in ln:
        return True
    if "rmsnorm" in ln or "rms_norm" in ln:
        return True
    if "groupnorm" in ln or "group_norm" in ln:
        return True
    if "gamma" in ln:
        return True
    return False


def adamw_param_groups_two_level(
    model: nn.Module,
    conditioner: nn.Module,
    lr: float,
    weight_decay: float,
    conditioner_lr_scale: float,
) -> List[Dict]:
    """
    Four groups: model {decay, no_decay} × base LR; conditioner {decay, no_decay} × scaled LR.
    """
    md, mnd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if param_excluded_from_weight_decay(n, p):
            mnd.append(p)
        else:
            md.append(p)

    cd, cnd = [], []
    for n, p in conditioner.named_parameters():
        if not p.requires_grad:
            continue
        if param_excluded_from_weight_decay(n, p):
            cnd.append(p)
        else:
            cd.append(p)

    lr_c = lr * conditioner_lr_scale
    return [
        {"params": md, "lr": lr, "weight_decay": weight_decay},
        {"params": mnd, "lr": lr, "weight_decay": 0.0},
        {"params": cd, "lr": lr_c, "weight_decay": weight_decay},
        {"params": cnd, "lr": lr_c, "weight_decay": 0.0},
    ]


def adamw_param_groups_single_module(module: nn.Module, lr: float, weight_decay: float) -> List[Dict]:
    """Finetune / single-module: decay vs no-decay at one LR."""
    d, nd = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if param_excluded_from_weight_decay(n, p):
            nd.append(p)
        else:
            d.append(p)
    return [
        {"params": d, "lr": lr, "weight_decay": weight_decay},
        {"params": nd, "lr": lr, "weight_decay": 0.0},
    ]


def adamw_param_groups_from_named_pairs(
    pairs: List[Tuple[str, torch.nn.Parameter]],
    lr: float,
    weight_decay: float,
) -> List[Dict]:
    """Like ``adamw_param_groups_single_module`` but for an explicit (name, param) list (finetune heads)."""
    d, nd = [], []
    for n, p in pairs:
        if not p.requires_grad:
            continue
        if param_excluded_from_weight_decay(n, p):
            nd.append(p)
        else:
            d.append(p)
    out: List[Dict] = []
    if d:
        out.append({"params": d, "lr": lr, "weight_decay": weight_decay})
    if nd:
        out.append({"params": nd, "lr": lr, "weight_decay": 0.0})
    return out
