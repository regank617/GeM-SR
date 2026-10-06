from __future__ import annotations
"""
Load VARSR code from ``benchmarks/varsr/repo`` without colliding with other ``models`` trees.

SwinIR, SR3, and (if present) ``benchmarks/var/repo`` all expose a top-level ``models``
package. A leftover ``sys.modules['models']`` from SwinIR has ``network_swinir`` but not
``VAR_RoPE``, which raises ``No module named 'models.var'``. Clear that cache, put the
VARSR repo first on ``sys.path``, then import.
"""

import sys
from pathlib import Path

from benchmarks.import_isolation import (
    clear_benchmark_namespaces,
    insert_path_first,
    verify_vendor_namespaces,
)

_VARSR_REPO = Path(__file__).resolve().parent / "repo"


def purge_var_benchmark_models_cache() -> None:
    """Remove ``models`` package entries that were not loaded from ``benchmarks/varsr/repo``."""
    to_del: list[str] = []
    for name in list(sys.modules):
        if name != "models" and not name.startswith("models."):
            continue
        mod = sys.modules.get(name)
        if mod is None:
            continue
        fp = (getattr(mod, "__file__", None) or "").replace("\\", "/").lower()
        if "/varsr/repo/" not in fp:
            to_del.append(name)
    for name in to_del:
        del sys.modules[name]


def ensure_varsr_repo_on_path() -> str:
    r = str(_VARSR_REPO.resolve())
    insert_path_first(r)
    return r


def import_varsr_var_rope():
    clear_benchmark_namespaces()
    varsr_root = ensure_varsr_repo_on_path()
    from models.var import VAR_RoPE  # noqa: E402

    verify_vendor_namespaces(varsr_root)
    return VAR_RoPE


def import_varsr_vqvae_class():
    """VQVAE class from VARSR repo (matches ``train_varsr`` / varsr checkpoints)."""
    clear_benchmark_namespaces()
    varsr_root = ensure_varsr_repo_on_path()
    from models.vqvae import VQVAE  # noqa: E402

    verify_vendor_namespaces(varsr_root)
    return VQVAE


def make_lidar_varsr_vqvae():
    """1-channel CHM tokenizer used by ``train_varsr.py`` (no ``benchmarks/var/`` tree)."""
    VQVAE = import_varsr_vqvae_class()
    return VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=64,
        v_patch_nums=(1, 2, 3, 4, 6, 12),
        in_channels=1,
        ch_mult=(1, 2, 4),
    )
