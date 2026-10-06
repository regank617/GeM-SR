from __future__ import annotations
"""
Isolate vendored ``model`` / ``models`` packages (EDSR, SR3, VAR, VARSR, SwinIR, …).

Importing both ``benchmarks/var/repo`` and ``benchmarks/varsr/repo`` in one process
loads two different ``models.*`` trees. Without clearing ``sys.modules``, the second
import silently reuses the first — **this fails in notebooks and on clusters**, not
only in Jupyter. Always clear before switching benchmarks, or use the helpers below.

SwinIR (``models``, ``utils``), SR3 (``model``, ``data``, ``core``), and VARSR
(``models``, ``utils``, ``dataloader``) all expose the same generic top-level names.
``_VENDOR_COLLISION_NAMESPACES`` below is that observed set; if you vendor a new
baseline that adds another generic top-level package name, add it there too.

**Adding a new baseline loader:** follow the existing ``import_*_wrapper()`` functions
in ``benchmark_hub.py`` — call ``clear_benchmark_namespaces()``, then
``insert_path_first(vendor_repo_root)``, then your ``from ... import ...``, then
``verify_vendor_namespaces(vendor_repo_root)``. The last call is what catches it if
this sequence was skipped or done out of order (by you or a future contributor)
instead of failing silently with code running against the wrong vendor's module.
"""

import os
import sys
import warnings
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent

# Generic top-level names actually exposed by the vendored baseline repos (see module
# docstring). Any of these left in `sys.modules` from a *different* vendor than the one
# just put first on `sys.path` is exactly the collision documented in
# `INSTRUCTIONS/REPO_CODING.md` ("Vendor `models` packages collide").
_VENDOR_COLLISION_NAMESPACES = ("model", "models", "utils", "data", "core", "dataloader")


def clear_benchmark_namespaces() -> None:
    for k in list(sys.modules.keys()):
        if any(k == m or k.startswith(m + ".") for m in _VENDOR_COLLISION_NAMESPACES):
            del sys.modules[k]


def insert_path_first(p: str) -> None:
    p = os.path.abspath(p)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def verify_vendor_namespaces(vendor_roots, *, strict: bool = False) -> None:
    """
    Call this right after importing from a vendor root you just put first on
    ``sys.path`` (i.e. right after your ``from models.x import Y``). It checks that
    none of ``_VENDOR_COLLISION_NAMESPACES`` currently cached in ``sys.modules``
    actually resolves to a *different* vendor's files — the exact bug this module
    exists to prevent, except caught right where it happened instead of surfacing
    later as a confusing ``ImportError`` (or, worse, no error at all: code silently
    running against the wrong baseline's implementation).

    ``vendor_roots``: the path (or list of paths, if the loader inserts more than
    one, e.g. Fast Diffusion also needs the SR3 repo) most recently passed to
    ``insert_path_first``.

    Emits a ``UserWarning`` by default; pass ``strict=True`` to raise instead.
    """
    if isinstance(vendor_roots, (str, os.PathLike)):
        vendor_roots = [vendor_roots]
    roots = [os.path.abspath(str(r)) for r in vendor_roots]

    problems = []
    for name in _VENDOR_COLLISION_NAMESPACES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        file_ = getattr(mod, "__file__", None)
        paths = [file_] if file_ else list(getattr(mod, "__path__", []) or [])
        if not paths:
            continue
        foreign = [p for p in paths if not any(os.path.abspath(p).startswith(r) for r in roots)]
        if foreign:
            problems.append(f"sys.modules[{name!r}] -> {foreign}")

    if problems:
        msg = (
            "Vendored benchmark namespace collision: a 'model'/'models'/'utils'/... module "
            "from a DIFFERENT baseline is still cached in sys.modules (see the module "
            "docstring in benchmarks/import_isolation.py and INSTRUCTIONS/REPO_CODING.md, "
            "'Vendor `models` packages collide').\n  "
            + "\n  ".join(problems)
            + f"\nExpected everything rooted under: {roots}\n"
            "Fix: call clear_benchmark_namespaces() immediately before importing this benchmark."
        )
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)


def import_var_vqvae_class():
    """LiDAR VAR-SR tokenizer: ``models.vqvae.VQVAE`` from ``benchmarks/varsr/repo``."""
    from benchmarks.varsr.varsr_repo_imports import import_varsr_vqvae_class

    return import_varsr_vqvae_class()


def import_varsr_wrapper_class():
    """VARSR wrapper — call **after** ``clear_benchmark_namespaces()`` if VAR was imported."""
    varsr_root = str(_BENCH_ROOT / "benchmarks" / "varsr" / "repo")
    clear_benchmark_namespaces()
    insert_path_first(varsr_root)
    from benchmarks.varsr.train_varsr import VARSRWrapper

    verify_vendor_namespaces(varsr_root)
    return VARSRWrapper
