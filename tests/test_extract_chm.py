# -*- coding: utf-8 -*-
"""CPU tests for jobs/extract_chm helpers (no HF download / GPU)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestExtractChmHelpers(unittest.TestCase):
    def _load_module(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "extract_chm", REPO_ROOT / "jobs" / "extract_chm.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_create_chm_percentiles_differ(self) -> None:
        from src.data.canopy_plots import create_chm

        cube = np.zeros((128, 48, 48), dtype=np.float32)
        # Tall column in center: density concentrated high
        cube[80:, 20:28, 20:28] = 1.0
        cube[40:60, 10:15, 10:15] = 0.5

        chm_p25, *_ = create_chm(cube, canopy_percentile=0.25, dtm_percentile=0.05)
        chm_p98, *_ = create_chm(cube, canopy_percentile=0.98, dtm_percentile=0.05)
        self.assertFalse(np.allclose(chm_p25, chm_p98))
        self.assertGreater(chm_p98.max(), chm_p25.max())

    def test_discover_dir_splits_all_to_test(self) -> None:
        mod = self._load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.npz", "b.npz", "nested/c.npz"):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(p, cube=np.zeros((128, 48, 48), dtype=np.float32))
            splits = mod.discover_dir_splits(root, all_to_test=True)
        self.assertEqual(list(splits.keys()), ["test"])
        self.assertEqual(len(splits["test"]), 3)

    def test_discover_dir_splits_named_subdirs(self) -> None:
        mod = self._load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "validation", "test"):
                d = root / split
                d.mkdir()
                np.savez(d / "patch.npz", cube=np.zeros((128, 48, 48), dtype=np.float32))
            splits = mod.discover_dir_splits(root, all_to_test=False)
        self.assertEqual(set(splits.keys()), {"train", "validation", "test"})
        self.assertEqual(len(splits["validation"]), 1)

    def test_write_batch_uses_chm_keys_and_validation_dir(self) -> None:
        mod = self._load_module()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chm"
            mod.ensure_split_dirs(out)
            chm_in = np.ones((32, 16), dtype=np.float32) * 10.0
            chm_tgt = np.ones((48, 48), dtype=np.float32) * 20.0
            mod.write_batch(
                [("validation", "sample.npz", np.zeros(1), np.zeros(1), chm_in, chm_tgt)],
                out_chm=out,
                out_cubes=None,
            )
            path = out / "validation" / "sample.npz"
            self.assertTrue(path.is_file())
            with np.load(path) as data:
                self.assertIn("chm_input", data.files)
                self.assertIn("chm_target", data.files)
                self.assertNotIn("input", data.files)
                np.testing.assert_array_equal(data["chm_input"], chm_in)
                # Raw bin units — not normalized to [0, 1]
                self.assertGreater(float(data["chm_input"].max()), 0.5)

    def test_normalize_split_name_maps_val_to_validation(self) -> None:
        mod = self._load_module()
        self.assertEqual(mod.normalize_split_name("val"), "validation")

    def test_preflight_copies(self) -> None:
        mod = self._load_module()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "chm"
            mod.ensure_split_dirs(out)
            for i in range(3):
                np.savez(
                    out / "test" / f"p{i}.npz",
                    chm_input=np.zeros((32, 16), dtype=np.float32),
                    chm_target=np.zeros((48, 48), dtype=np.float32),
                )
            mod.preflight_copies(out, 2)
            self.assertEqual(len(list((out / "train").glob("*.npz"))), 2)
            self.assertEqual(len(list((out / "validation").glob("*.npz"))), 2)

    def test_compute_chm_pair_pickleable(self) -> None:
        import pickle

        from jobs.extract_chm import compute_chm_pair

        pickle.dumps(compute_chm_pair)

    def test_forward_model_output_shapes(self) -> None:
        import torch

        from src.data.forward_model import LidarForwardImagingModel

        model = LidarForwardImagingModel(
            input_res_m=(2.0, 2.0),
            output_res_m=(3.0, 6.0),
            footprint_diameter_m=10.0,
            b=0.0,
            eta=0.0,
            ref_altitude=500.0,
            ref_photon_count=20.0,
        )
        model.eval()
        cube = torch.rand(1, 128, 48, 48)
        with torch.no_grad():
            lr = model.forward_noiseless(cube)
        self.assertEqual(tuple(lr.shape), (1, 128, 32, 16))
        mod = self._load_module()
        chm_in, chm_tgt = mod.make_chm_worker(0.98)((lr[0].numpy(), cube[0].numpy()))
        self.assertEqual(chm_in.shape, (32, 16))
        self.assertEqual(chm_tgt.shape, (48, 48))


if __name__ == "__main__":
    unittest.main()
