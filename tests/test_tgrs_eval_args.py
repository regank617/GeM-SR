# -*- coding: utf-8 -*-
"""CLI construction for jobs/tgrs_eval.py (no dataset / GPU)."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestTgrsEvalArgs(unittest.TestCase):
    def _load_tgrs_eval(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tgrs_eval", REPO_ROOT / "jobs" / "tgrs_eval.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_parse_only_six_gemsr_and_sr3_steps(self) -> None:
        mod = self._load_tgrs_eval()

        os.environ.pop("MODEL_ROOT", None)
        os.environ.pop("GEMSR_DATA_ROOT", None)
        os.environ.pop("TGRS_EVAL_DUMMY_ROOT", None)
        manifest = mod._load_manifest(mod.DEFAULT_MANIFEST)
        argv = mod.build_hub_argv(
            manifest,
            Path("/dummy/model_root"),
            config="configs/generator/lidar_ssmVQ.yaml",
            save_path="benchmarks/evaluation/tgrs_robustness",
            eval_seed=0,
        )
        parsed = mod.parse_hub_argv(argv)
        self.assertEqual(len(parsed.gemsr_entry), 6)
        self.assertEqual(len(parsed.tokenizer_gemsr_entry), 6)
        self.assertEqual(list(parsed.sr3_ddim_steps), [50, 100, 200, 0])
        self.assertEqual(parsed.eval_seed, 0)
        self.assertFalse(parsed.skip_existing)
        self.assertEqual(parsed.eval_split, "validation")
        self.assertEqual(list(parsed.interpolation), ["bilinear", "bicubic"])
        keys = [e.split("::", 1)[0] for e in parsed.gemsr_entry]
        self.assertIn("Gess_VQ_C_S_U_P", keys)
        self.assertIn("Gess_VQ_T_L_U_R", keys)
        self.assertNotIn("Gess_VQ_C_L_U_R", keys)
        u_p = next(e for e in parsed.gemsr_entry if e.startswith("Gess_VQ_C_S_U_P::"))
        self.assertTrue(u_p.endswith("0.4,0.4,0.6,0.8"))
        a_r = next(e for e in parsed.gemsr_entry if e.startswith("Gess_VQ_C_S_A_R::"))
        self.assertTrue(a_r.endswith("1.0,1.0,1.0,1.0"))
        labels = [lab for lab, _, _ in mod.required_paths(manifest, Path("/dummy/m"), Path("/dummy/d"))]
        self.assertIn("data split validation", labels)
        self.assertFalse(any(lab in ("val inputs", "val targets") for lab in labels))

        argv_skip = mod.build_hub_argv(
            manifest,
            Path("/dummy/model_root"),
            config="configs/generator/lidar_ssmVQ.yaml",
            save_path="benchmarks/evaluation/tgrs_robustness",
            eval_seed=0,
            skip_existing=True,
        )
        self.assertIn("--skip_existing", argv_skip)
        self.assertTrue(mod.parse_hub_argv(argv_skip).skip_existing)

        argv_test = mod.build_hub_argv(
            manifest,
            Path("/dummy/model_root"),
            config="configs/generator/lidar_ssmVQ.yaml",
            save_path="benchmarks/evaluation/tgrs_robustness_test",
            eval_seed=0,
            eval_split="test",
        )
        parsed_test = mod.parse_hub_argv(argv_test)
        self.assertEqual(parsed_test.eval_split, "test")
        self.assertEqual(parsed_test.save_path, "benchmarks/evaluation/tgrs_robustness_test")
        self.assertEqual(
            mod._save_path_for_split("benchmarks/evaluation/tgrs_robustness", "test"),
            "benchmarks/evaluation/tgrs_robustness_test",
        )
        self.assertEqual(
            mod._save_path_for_split("benchmarks/evaluation/tgrs_robustness", "validation"),
            "benchmarks/evaluation/tgrs_robustness",
        )

    def test_npz_split_probe_accepts_chm_keys(self) -> None:
        import tempfile

        import numpy as np

        mod = self._load_tgrs_eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "validation", "test"):
                d = root / split
                d.mkdir()
                np.savez(
                    d / "patch0.npz",
                    chm_input=np.zeros((32, 16), dtype=np.float32),
                    chm_target=np.zeros((48, 48), dtype=np.float32),
                )
            missing = mod.check_npz_splits(root)
        self.assertEqual(missing, [])
