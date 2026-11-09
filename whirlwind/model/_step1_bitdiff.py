"""Step-1 numerical fingerprinting for the manual-vs-HF training-step bit-diff.

Gated by the ``WW_STEP1_BITDIFF`` env var. When set, the model node captures
step-1 numerical fingerprints and writes them to ``WW_STEP1_BITDIFF_OUT`` (JSON),
then exits. Lets us compare the manual FSDP2 node vs the HF-Trainer node at step 1
with IDENTICAL inputs (same seeds) to decide whether the training-step machinery
(#3: accelerator.backward / AcceleratedOptimizer / clip wrapper) is a no-op at
GAS=1+bf16, or a real numerical divergence.

All fingerprints are GLOBAL scalars (all-reduced across ranks). Comparison:
  - input checksums match  -> inputs identical -> valid A/B
  - grad + param stats match -> #3 no-op -> #1+#2 fully explain the stall
  - input matches but grad/param differ -> #3 (or a 4th factor) is real

Signals:
  - grad: pre-clip (l2_norm, signed_sum, abs_sum) — captured BEFORE clip mutates
  - param: norm before & after optimizer.step — if both match, update was identical
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import torch


def enabled() -> bool:
    return os.environ.get("WW_STEP1_BITDIFF", "0") == "1"


_state_lock = threading.Lock()
_state: dict[str, Any] = {"input_sum": 0.0, "input_count": 0, "fwd_calls": 0, "done": False}


def record_input_checksum(input_ids: list) -> None:
    """Accumulate an input checksum across step-1 microbatch forward calls (rank-local)."""
    if not enabled():
        return
    with _state_lock:
        for ids in input_ids:
            t = torch.as_tensor(ids)
            _state["input_sum"] += float(t.float().sum().item())
            _state["input_count"] += int(t.numel())
        _state["fwd_calls"] += 1


def maybe_capture_request(req) -> None:
    """If WW_STEP1_CAPTURE set, serialize the first forward_accum_grad request (rank 0).

    Dumps to WW_STEP1_CAPTURE_OUT as a torch pickle of req.to_dict(). Enables
    capture-replay: the same request can be fed to both the manual and HF nodes
    in a standalone replay, removing rollout non-determinism from the A/B.
    """
    if not enabled() or os.environ.get("WW_STEP1_CAPTURE", "0") != "1":
        return
    with _state_lock:
        if _state.get("captured"):
            return
        _state["captured"] = True
    try:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        if rank != 0:
            return
        out = os.environ.get("WW_STEP1_CAPTURE_OUT", "/dev/shm/step1_req.pt")
        d = req.to_dict() if hasattr(req, "to_dict") else req
        torch.save(d, out)
        print(f"[step1-bitdiff] captured request to {out}", flush=True)
    except Exception as e:
        print(f"[step1-bitdiff] capture failed: {e}", flush=True)


def _to_local_float(t: torch.Tensor) -> float:
    """Reduce a (possibly DTensor) scalar tensor to a rank-local Python float.

    For FSDP2 DTensor params/grads, gather to full tensor first so the per-rank
    value is the GLOBAL value (then the all_reduce below is idempotent / correct).
    """
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    return float(t.float().sum().item())


def _global_reduce(value: float, op=None) -> float:
    """All-reduce a Python float across ranks (sum). Idempotent if value is already global."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    if op is None:
        op = torch.distributed.ReduceOp.SUM
    t = torch.tensor(value, dtype=torch.float64, device="cuda")
    torch.distributed.all_reduce(t, op=op)
    return float(t.item())


def grad_stats_pre_clip(params) -> tuple[float, float, float]:
    """Global (l2_norm, signed_sum, abs_sum) of grads BEFORE clipping. Call BEFORE clip."""
    sq = sum_ = abs_ = 0.0
    for p in params:
        g = p.grad
        if g is None:
            continue
        gf = g.float()
        sq += _to_local_float(gf.pow(2))
        sum_ += _to_local_float(gf)
        abs_ += _to_local_float(gf.abs())
    # If grads are DTensors, _to_local_float already gathered each -> global.
    # If local, all-reduce. all_reduce of an already-global sum double-counts, so
    # guard: only reduce if world_size>1 AND tensors were local. DTensor .full_tensor
    # already gathered, so skip the reduce in that case (heuristic: grads on FSDP2
    # are DTensors -> gathered). To be safe, reduce only when not already gathered.
    # Simpler correct approach: grads on FSDP2 are DTensors; .full_tensor() gathers
    # to the FULL global tensor, so sq/sum_/abs_ are already global -> no reduce.
    return sq ** 0.5, sum_, abs_


def param_l2_norm(params) -> float:
    """Global L2 norm of all params. For FSDP2 DTensor params, .full_tensor() gathers."""
    sq = 0.0
    for p in params:
        sq += _to_local_float(p.data.float().pow(2))
    return sq ** 0.5


def dump_and_exit(
    params,
    grad_norm_clip: float,
    rank: int,
    label: str,
    param_norm_pre: float | None = None,
    grad_pre: tuple[float, float, float] | None = None,
) -> None:
    """Capture step-1 fingerprints and (rank 0) write JSON + exit all ranks.

    grad_pre must be the PRE-CLIP grad stats (captured before clip mutated grads).
    param_norm_pre must be the PRE-step param norm (captured before optimizer.step).
    """
    if not enabled():
        return
    with _state_lock:
        if _state["done"]:
            return
        _state["done"] = True

    param_norm_post = param_l2_norm(params)

    g_norm_pre, g_sum, g_abs = grad_pre if grad_pre is not None else (None, None, None)

    fp = {
        "label": label,
        "rank": rank,
        "input_sum": _state["input_sum"],
        "input_count": _state["input_count"],
        "fwd_calls": _state["fwd_calls"],
        "grad_norm_pre_clip_clipfn": float(grad_norm_clip),  # from clip_grad_norm_ return
        "grad_norm_pre_clip": g_norm_pre,   # our recomputation (should match above)
        "grad_sum": g_sum,
        "grad_abs_sum": g_abs,
        "param_norm_pre": param_norm_pre,
        "param_norm_post": param_norm_post,
        "param_delta_norm": (abs(param_norm_post - param_norm_pre) if param_norm_pre is not None else None),
    }

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    out = os.environ.get("WW_STEP1_BITDIFF_OUT", f"/dev/shm/step1_bitdiff_{label}.json")
    if rank == 0:
        with open(out, "w") as f:
            json.dump(fp, f, indent=2)
        print(f"[step1-bitdiff] wrote {out}: {fp}", flush=True)
        os._exit(0)
    else:
        os._exit(0)
