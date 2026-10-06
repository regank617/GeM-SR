from __future__ import annotations
"""
EMA (exponential moving average) over trainable parameters for diffusion benchmark models.

Used by ``train_sr3`` and ``train_fast_diffusion``. Shadow tracks ``named_parameters()`` with
``requires_grad``; buffers (e.g. diffusion noise schedule) are left on the live model and copied
unchanged into saved checkpoints.
"""

from typing import Any, Dict, Optional

import torch
from torch import nn


class ParameterEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        sd = state["shadow"]
        for k in list(self.shadow.keys()):
            if k in sd:
                self.shadow[k] = sd[k].float().clone().to(self.shadow[k].device)

    def merge_into_full_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Full ``model.state_dict()`` with EMA weights swapped in for tracked parameters."""
        out = model.state_dict()
        for name, tensor in self.shadow.items():
            if name in out:
                t = out[name]
                out[name] = tensor.to(device=t.device, dtype=t.dtype).clone()
        return out

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """Replace trainable params with EMA for inference; call :meth:`restore` after."""
        self._backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name].to(p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self._backup:
            return
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = None
