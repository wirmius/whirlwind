"""Per-tensor update-path tracing: manual FSDP2 vs HF-Trainer FSDP2 (VALUES).

Extends the forward/loss ``WW_IS_DIAG`` dump (which records loss-path tensors)
to the BACKWARD + CLIP + OPTIMIZER-STEP path. Env-gated by ``WW_UPDATE_DIAG``.
Records, for every parameter, at four stages of ``_model_update_step``:

  grad_pre_clip    — p.grad BEFORE clip (the true backward + all-reduce output)
  grad_post_clip   — p.grad AFTER clip (what the optimizer sees)
  param_pre_step   — p.data BEFORE optimizer.step (== W_s; sanity check)
  param_post_step  — p.data AFTER optimizer.step (== W'; the update result)
  optim_pre_step   — AdamW exp_avg / exp_avg_sq BEFORE step
  optim_post_step  — AdamW exp_avg / exp_avg_sq AFTER step

Each tensor is recorded as a content SHA-256 (bit-diff, not norm-diff) + norm
stats. Additionally, when ``WW_UPDATE_DIAG_SAVE=1`` is set, the FULL grad/param
tensors for the four grad/param stages are saved to ``{dir}/s{step}_{node}_rank{r}_{stage}.pt``
(one dict {name: cpu_tensor} per stage) so the differ can compute the
ELEMENT-WISE divergence (max_abs, rel, argmax, sample values) when a hash
mismatch is found. Optimizer state is hash-only (pre should match; post
diverges only because grads did).

All stats use ``.full_tensor()`` so they are sharding-invariant: a manual-vs-HF
hash mismatch is a REAL bit divergence, not an artifact of the two FSDP stacks
sharding differently. No-op + zero overhead unless ``WW_UPDATE_DIAG`` is set.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any

import torch

_state_lock = threading.Lock()
_state: dict[str, dict[str, dict[str, Any] | None]] = {}
_enabled_cache: bool | None = None


def enabled() -> bool:
    global _enabled_cache
    if _enabled_cache is None:
        _enabled_cache = os.environ.get("WW_UPDATE_DIAG", "0") == "1"
    return _enabled_cache


def _save_full() -> bool:
    return enabled() and os.environ.get("WW_UPDATE_DIAG_SAVE", "0") == "1"


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    return t.detach().cpu().contiguous()


def _sha(t: torch.Tensor) -> str | None:
    """Content hash of raw bytes (bf16 has no numpy dtype -> view as int16)."""
    tc = _to_cpu(t)
    if tc.dtype == torch.bfloat16:
        b = tc.view(torch.int16).numpy().tobytes()
    else:
        b = tc.numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def _stats(t: torch.Tensor | None) -> dict[str, Any] | None:
    if t is None:
        return None
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    if t.numel() == 0:
        return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                "sha": None, "l2_norm": 0.0, "max_abs": 0.0, "mean_abs": 0.0, "mean": 0.0}
    tf = t.float()
    s = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "sha": _sha(t),
        "l2_norm": float(tf.norm().item()),
        "max_abs": float(tf.abs().max().item()),
        "mean_abs": float(tf.abs().mean().item()),
        "mean": float(tf.mean().item()),
    }
    return s


def _bundle_path(stage: str, rank: int, node_tag: str) -> str | None:
    if not _save_full():
        return None
    out_dir = os.environ.get("WW_UPDATE_DIAG_DIR", "")
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    step = os.environ.get("WW_UPDATE_DIAG_STEP", "")
    return os.path.join(out_dir, f"s{step}_{node_tag}_rank{rank}_{stage}.pt")


def _save_bundle(model: Any, attr: str, stage: str, rank: int, node_tag: str) -> None:
    """Save a {name: cpu_tensor} bundle for grad (.grad) or param (.data)."""
    if not _save_full():
        return
    path = _bundle_path(stage, rank, node_tag)
    if path is None:
        return
    bundle: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        t = p.grad if attr == "grad" else p.data
        if t is not None:
            bundle[name] = _to_cpu(t)
    torch.save(bundle, path)


def capture_grads(model: Any, stage: str, rank: int, node_tag: str) -> None:
    if not enabled():
        return
    with _state_lock:
        _state[stage] = {name: _stats(p.grad) for name, p in model.named_parameters()}
    _save_bundle(model, "grad", stage, rank, node_tag)


def capture_params(model: Any, stage: str, rank: int, node_tag: str) -> None:
    if not enabled():
        return
    with _state_lock:
        _state[stage] = {name: _stats(p.data) for name, p in model.named_parameters()}
    _save_bundle(model, "data", stage, rank, node_tag)


def capture_optim(optimizer: Any, stage: str, model: Any = None) -> None:
    """Hash AdamW exp_avg / exp_avg_sq per param (keyed by param NAME, not id).

    ``model`` is used to build an id(p)->name map so the keys match across the
    manual and HF processes (id(p) is process-specific and would never align).
    Falls back to shape if model is None (diagnostic only)."""
    if not enabled():
        return
    id_to_name = {}
    if model is not None:
        for name, p in model.named_parameters():
            id_to_name[id(p)] = name
    d: dict[str, dict[str, Any] | None] = {}
    try:
        for group in optimizer.param_groups:
            for p in group["params"]:
                st = optimizer.state.get(p, {})
                key = id_to_name.get(id(p), f"shape{list(p.shape)}")
                d[key] = {
                    "exp_avg": _stats(st.get("exp_avg")),
                    "exp_avg_sq": _stats(st.get("exp_avg_sq")),
                }
    except Exception:
        return
    with _state_lock:
        _state[stage] = d  # type: ignore[assignment]


def write(rank: int, node_tag: str) -> None:
    if not enabled():
        return
    with _state_lock:
        stages: dict[str, Any] = {}
        for stage, d in _state.items():
            total_l2 = 0.0
            dtypes: set[str] = set()
            n_none = 0
            for v in d.values():
                if v is None:
                    n_none += 1
                    continue
                if "l2_norm" in v:
                    total_l2 += v["l2_norm"]
                    dtypes.add(v.get("dtype", "?"))
                else:
                    # optim-state sub-dict {exp_avg, exp_avg_sq}
                    for sub in v.values():
                        if sub and "l2_norm" in sub:
                            total_l2 += sub["l2_norm"]
            stages[stage] = {
                "total_l2": total_l2,
                "dtypes": sorted(dtypes),
                "n_tensors": len(d),
                "n_none": n_none,
                "tensors": d,
            }
        rec = {
            "node": node_tag,
            "step": os.environ.get("WW_UPDATE_DIAG_STEP", ""),
            "rank": rank,
            "save_full": _save_full(),
            "stages": stages,
        }
        out_dir = os.environ.get("WW_UPDATE_DIAG_DIR", "")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"s{rec['step']}_{node_tag}_rank{rank}_update.json")
            with open(path, "w") as f:
                json.dump(rec, f, indent=2)
        _state.clear()
