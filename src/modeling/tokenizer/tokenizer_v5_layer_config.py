from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TokenizerV5LayerConfig:
    conv_bias: bool = False
    use_zero_preserving_resnorm: bool = True
