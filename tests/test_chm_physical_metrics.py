# -*- coding: utf-8 -*-
"""CPU tests for CHM physical metric accumulators (no dataset / GPU)."""
from __future__ import annotations

import unittest

import torch

from benchmarks.chm_physical_metrics import (
    accumulate_hr_histogram,
    accumulate_physical_errors,
    finalize_acc,
    finalize_percentile_tails,
    finalize_quartiles,
    hr_quantile_thresholds_from_histogram,
    percentile_accs,
    quartile_accs,
    _empty_acc,
)


class TestPhysicalMetrics(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        hr = torch.linspace(0.0, 1.0, 100).view(1, 1, 10, 10)
        acc = _empty_acc()
        q = quartile_accs()
        p = percentile_accs(3)
        accumulate_physical_errors(hr, hr, acc, q, p, (0.9, 0.95, 0.99))
        fin = finalize_acc(acc, physical_meters_scale=64.0)
        self.assertEqual(fin["n"], 100)
        self.assertAlmostEqual(fin["mae_m"], 0.0, places=6)
        self.assertAlmostEqual(fin["rmse_m"], 0.0, places=6)
        self.assertAlmostEqual(fin["bias_m"], 0.0, places=6)
        self.assertAlmostEqual(fin["r2"], 1.0, places=5)

    def test_constant_bias(self) -> None:
        hr = torch.full((1, 1, 4, 4), 0.5)
        sr = hr + 0.1
        acc = _empty_acc()
        q = quartile_accs()
        p = percentile_accs(3)
        accumulate_physical_errors(sr, hr, acc, q, p, (0.9, 0.95, 0.99))
        fin = finalize_acc(acc, physical_meters_scale=10.0)
        self.assertAlmostEqual(fin["bias_m"], 1.0, places=5)
        self.assertAlmostEqual(fin["mae_m"], 1.0, places=5)

    def test_quartile_binning_matches_floor_hr_times_4(self) -> None:
        hr = torch.tensor([0.0, 0.24, 0.25, 0.49, 0.50, 0.74, 0.75, 1.0]).view(1, 1, 2, 4)
        sr = torch.zeros_like(hr)
        acc = _empty_acc()
        q = quartile_accs()
        p = percentile_accs(3)
        accumulate_physical_errors(sr, hr, acc, q, p, (2.0, 2.0, 2.0))
        fin_q = finalize_quartiles(q, 1.0)
        # 0.0,0.24 → Q0; 0.25,0.49 → Q1; 0.50,0.74 → Q2; 0.75,1.0 → Q3
        self.assertEqual(fin_q["pixel_counts"], [2, 2, 2, 2])

    def test_histogram_quantiles_on_uniform_grid(self) -> None:
        hr = torch.linspace(0.0, 1.0, 1001)
        hist = torch.zeros(10_000, dtype=torch.int64)
        accumulate_hr_histogram(hr, hist)
        t90, t95, t99 = hr_quantile_thresholds_from_histogram(hist, (0.90, 0.95, 0.99))
        self.assertGreater(t90, 0.85)
        self.assertLess(t90, 0.95)
        self.assertGreater(t99, t95)
        self.assertGreater(t95, t90)

    def test_percentile_tail_counts(self) -> None:
        hr = torch.linspace(0.0, 1.0, 100).view(1, 1, 10, 10)
        sr = hr.clone()
        acc = _empty_acc()
        q = quartile_accs()
        p = percentile_accs(3)
        accumulate_physical_errors(sr, hr, acc, q, p, (0.90, 0.95, 0.99))
        fin = finalize_percentile_tails(p, (0.90, 0.95, 0.99), ("top10", "top5", "top1"), (0.9, 0.95, 0.99), 1.0)
        self.assertGreaterEqual(fin["pixel_counts"][0], fin["pixel_counts"][1])
        self.assertGreaterEqual(fin["pixel_counts"][1], fin["pixel_counts"][2])
        self.assertGreater(fin["pixel_counts"][2], 0)


if __name__ == "__main__":
    unittest.main()
