"""FP8 checkpoint ingestion for the model-update node (EXPERIMENTAL).

Loads HF checkpoints quantized in the *official fine-grained FP8 format* —
the same recipe as ``Qwen/Qwen3.8-27B-FP8`` and DeepSeek's FP8 releases:

* weights of target linear layers stored as ``float8_e4m3fn``,
* per-128x128-block static scales stored as ``<module>.weight_scale_inv``
  (bf16, shape ``(ceil(out/128), ceil(in/128))``),
* ``config.json`` carries ``quantization_config`` with
  ``quant_method: "fp8"``, ``fmt: "e4m3"``, ``activation_scheme: "dynamic"``,
  ``weight_block_size: [128, 128]``,
* norms, embeddings, ``lm_head``, vision towers, and other
  ``modules_to_not_convert`` stay in their original dtype.

The updater cannot train FP8 tensors directly (FSDP2 + PEFT expect standard
dtypes; the fp32-master rule of TIPS_AND_TRICKS.md §1.1 applies regardless), so this module
**dequantizes at load time**: ``w_bf16 = w_fp8 * scale_inv`` (exact-linear,
no rounding beyond the already-baked-in quantization error) and hands the
model node an ordinary BF16 in-memory model. Training then proceeds exactly
as with a native BF16 checkpoint; the engine-side FP8 question (serving +
LoRA-on-FP8-base) is deliberately out of scope here.

Memory note: the whole dequantized state dict is materialized on CPU before
``load_state_dict`` — fine at 0.6B-8B. For the 27B+ scale a streaming /
sharded init (per-tensor ``load_state_dict`` assignment) is the follow-up;
the format reader below already streams shard-by-shard.

This module is intentionally side-effect free on the import graph: it only
needs torch + safetensors + transformers lazily, and nothing imports it
unless an FP8 checkpoint is detected (extension guide: additive machinery).
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

# Scale-tensor suffixes seen in the wild. The official Qwen/DeepSeek layout
# stores the *inverse* scale (`weight_scale_inv`, dequant multiplies by it).
# `weight_scale` (non-inverted, divide on dequant) is accepted defensively.
_SCALE_SUFFIXES = ("weight_scale_inv", "weight_scale")


def is_fp8_checkpoint(path: str | os.PathLike) -> bool:
    """True if ``path`` looks like an official fine-grained FP8 checkpoint."""
    cfg_path = os.path.join(path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    qc = cfg.get("quantization_config") or {}
    return qc.get("quant_method") == "fp8"


def _shard_files(path: str) -> list[str]:
    """All safetensors shards of an HF checkpoint dir (index-aware).

    Index-aware when ``model.safetensors.index.json`` exists; otherwise GLOB
    every ``*.safetensors`` — official Qwen FP8 releases (e.g.
    Qwen3.8-27B-FP8) shard as ``layers-0.safetensors``/``mtp.safetensors``/
    ``outside.safetensors`` with NO index and NO ``model.safetensors``.
    """
    idx_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        import glob
        files = sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(path, "*.safetensors"))
        )
        if not files:
            files = ["model.safetensors"]  # let the missing-file error below fire
    missing = [fn for fn in files if not os.path.exists(os.path.join(path, fn))]
    if missing:
        raise FileNotFoundError(f"FP8 checkpoint shards missing in {path}: {missing}")
    return files


def _dequant_blockwise(w: Any, scale_inv: Any, block: int) -> Any:
    """Dequantize a 2D FP8 tensor with per-``block``-square scales.

    ``scale_inv[i, j]`` applies to rows ``[i*block, (i+1)*block)`` and cols
    ``[j*block, (j+1)*block)``. Edge blocks are clipped, not padded — the
    official quantizers compute amax over the *actual* (smaller) edge block,
    so padding with zeros before the multiply is exact.
    """
    import torch

    if w.dim() != 2:
        raise ValueError(f"FP8 block dequant expects 2D weights, got {w.dim()}D")
    out_dim, in_dim = w.shape
    w32 = w.float()

    # Per-tensor scales (shape (1,1)) broadcast trivially.
    if scale_inv.numel() == 1:
        return w32 * scale_inv.float().reshape(1, 1)

    nb_o = math.ceil(out_dim / block)
    nb_i = math.ceil(in_dim / block)
    if tuple(scale_inv.shape) != (nb_o, nb_i):
        raise ValueError(
            f"weight_scale_inv shape {tuple(scale_inv.shape)} does not match "
            f"({nb_o}, {nb_i}) for weight {tuple(w.shape)} at block {block}"
        )

    pad_o = nb_o * block - out_dim
    pad_i = nb_i * block - in_dim
    if pad_o or pad_i:
        w32 = torch.nn.functional.pad(w32, (0, pad_i, 0, pad_o))
    s32 = scale_inv.float()
    # (nb_o, block, nb_i, block) * (nb_o, 1, nb_i, 1) -> exact per-block scaling
    w32 = w32.view(nb_o, block, nb_i, block) * s32[:, None, :, None]
    w32 = w32.view(nb_o * block, nb_i * block)
    return w32[:out_dim, :in_dim]


def dequantize_state_dict(path: str, dtype: Any, block: int = 128) -> dict[str, Any]:
    """Read an official-format FP8 checkpoint into a clean ``dtype`` state dict.

    Scale tensors are consumed and dropped; everything the model expects is
    returned in ``dtype``. Raises loudly on unrecognized scale-tensor names
    so format drift fails at load, not silently mid-training.
    """
    import torch
    from safetensors.torch import load_file

    state: dict[str, Any] = {}
    scales: dict[str, tuple[str, Any]] = {}
    unknown_scale_keys: list[str] = []

    for fn in _shard_files(path):
        shard = load_file(os.path.join(path, fn))
        for key, tensor in shard.items():
            for suffix in _SCALE_SUFFIXES:
                if key.endswith("." + suffix):
                    base = key[: -len(suffix) - 1]
                    if suffix == "weight_scale_inv":
                        scales[base] = ("mul", tensor)
                    else:
                        scales[base] = ("div", tensor)
                    break
            else:
                if "scale" in key.rsplit(".", 1)[-1].lower():
                    unknown_scale_keys.append(key)
                state[key] = tensor

    if unknown_scale_keys:
        raise ValueError(
            f"Unrecognized scale tensors in FP8 checkpoint (format drift?): "
            f"{unknown_scale_keys[:5]} — refusing to guess dequant semantics"
        )

    n_quantized = 0
    for base, (op, scale) in scales.items():
        wkey = base + ".weight"
        if wkey not in state:
            raise KeyError(f"Scale tensor present for missing weight '{wkey}'")
        w = state.pop(wkey)
        w32 = _dequant_blockwise(w, scale, block)
        if op == "div":  # non-inverted scale: divide instead of multiply
            w32 = w32 / (scale.float().reshape(1, 1) if scale.numel() == 1 else scale.float())
        state[wkey] = w32.to(dtype)
        n_quantized += 1
    logger.info(
        "FP8 dequant: %d/%d tensors dequantized from %s (block=%d)",
        n_quantized, len(state), path, block,
    )
    return state


def from_pretrained_dequant(
    path: str,
    *,
    torch_dtype: Any,
    attn_implementation: str = "sdpa",
    **init_kwargs: Any,
) -> Any:
    """Drop-in ``from_pretrained`` replacement for official FP8 checkpoints.

    Mirrors ``AutoModelForCausalLM.from_pretrained(path, torch_dtype=...,
    attn_implementation=..., **init_kwargs)`` semantics: same config, same
    tokenizer-independent model construction, weights dequantized to
    ``torch_dtype``. ``quantization_config`` is stripped from the in-memory
    config so downstream consumers (PEFT, FSDP wrap, HF export) see a
    vanilla model.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(path, **init_kwargs)
    qc = getattr(config, "quantization_config", None)
    if not isinstance(qc, dict) or qc.get("quant_method") != "fp8":
        raise ValueError(
            f"{path} is not an official fp8 checkpoint "
            f"(quantization_config={qc!r})"
        )
    block = qc.get("weight_block_size") or [128, 128]
    if len(block) != 2 or block[0] != block[1]:
        raise ValueError(f"Unsupported weight_block_size {block} (want square)")
    # Strip quantization metadata — we hand back a standard-dtype model.
    config.quantization_config = None  # type: ignore[assignment]

    try:
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation=attn_implementation
        )
    except TypeError:  # older transformers: from_config lacks the kwarg
        config._attn_implementation = attn_implementation  # type: ignore[attr-defined]
        model = AutoModelForCausalLM.from_config(config)

    state = dequantize_state_dict(path, torch_dtype, block=block[0])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"FP8 checkpoint has unexpected keys: {unexpected[:5]}")
    if missing:
        # Tied-embedding checkpoints legitimately omit lm_head.weight.
        text_cfg = (
            model.config.get_text_config()
            if hasattr(model.config, "get_text_config")
            else model.config
        )
        if getattr(text_cfg, "tie_word_embeddings", False):
            model.tie_weights()
            missing = [k for k in missing if k != "lm_head.weight"]
        if missing:
            raise ValueError(f"FP8 checkpoint is missing keys: {missing[:5]}")

    model.to(torch_dtype)
    return model
