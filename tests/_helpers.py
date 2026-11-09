"""Shared test helpers for the whirlwind test suite.

Provides environment-gating guards (torch/CUDA/model) that survive broken
installs: a half-broken torch (e.g. missing ``libtorch_global_deps.so``)
raises ``OSError`` on import, not just ``ImportError``, which previously
crashed test collection. These helpers catch ``Exception`` broadly so the
suite skips cleanly instead of erroring out.
"""

from __future__ import annotations

import importlib.util
import os
import unittest


def _import_torch():
    """Import torch lazily; return None if unavailable or broken.

    Catches OSError too (broken .so / missing CUDA libs), not just ImportError.
    """
    try:
        import torch  # noqa: F401

        return torch
    except Exception:
        return None


def torch_available() -> bool:
    """True if torch imports AND a CUDA GPU is visible."""
    torch = _import_torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    except Exception:
        return False


def transformers_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def sglang_available() -> bool:
    return importlib.util.find_spec("sglang") is not None


# ---------------------------------------------------------------------------
# Skip decorators
# ---------------------------------------------------------------------------

skip_no_torch = unittest.skipIf(_import_torch() is None, "torch not installed")

skip_no_gpu = unittest.skipIf(not torch_available(), "CUDA GPU not available")

skip_no_transformers = unittest.skipIf(not transformers_available(), "transformers not installed")

# GPU + transformers + a model: the heaviest tier (model_update integration).
skip_no_model_env = unittest.skipIf(
    not (torch_available() and transformers_available()),
    "CUDA + transformers not available",
)


def model_path() -> str:
    """Resolve the small model path for tests.

    Priority:
      1. WHIRLWIND_TEST_MODEL env var (explicit override)
      2. WHIRLWIND_TEST_MODEL_DIR/<basename> on disk
      3. ``Qwen/Qwen3-0.6B`` from HF hub (downloads on first run, cached via HF_HOME)

    Returns a path/identifier usable by AutoModelForCausalLM.from_pretrained.
    """
    explicit: str | None = os.environ.get("WHIRLWIND_TEST_MODEL")
    if explicit:
        return explicit
    base: str | None = os.environ.get("WHIRLWIND_TEST_MODEL_DIR")
    if base:
        cand = os.path.join(base, "Qwen3-0.6B")
        if os.path.isdir(cand):
            return cand
    return "Qwen/Qwen3-0.6B"
