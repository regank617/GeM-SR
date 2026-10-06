# -*- coding: utf-8 -*-
"""Shape check for parameter-free interpolation baselines."""
from __future__ import annotations

import unittest

import torch

from benchmarks.eval_wrappers import ResizeInterpEval


class TestResizeInterpEval(unittest.TestCase):
    def test_bilinear_and_bicubic_32x16_to_48x48(self) -> None:
        lr = torch.rand(2, 1, 32, 16)
        for mode in ("bilinear", "bicubic", "nearest"):
            out = ResizeInterpEval(mode=mode)(lr)
            self.assertEqual(tuple(out.shape), (2, 1, 48, 48), msg=mode)
