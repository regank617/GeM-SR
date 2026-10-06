from __future__ import annotations
"""Convolution bias toggles for ``SuperResWaveletConditionerV4`` / ``ConditionerWaveletPretrainer``."""

from dataclasses import dataclass

@dataclass(frozen=True)
class ConditionerLayerConfig:
    """When ``conv_bias`` is ``False`` (default), matches original zero-preserving Convs (checkpoint-compatible)."""
    conv_bias: bool = False

@dataclass(frozen=True)
class PaperConvNeXtLayerConfig:
    conv_bias: bool = False
    linear_bias: bool = False
