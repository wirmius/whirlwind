"""Model loading for the model-update node: picks the right loader for a checkpoint.

``load_causal_lm`` is the single ``from_pretrained`` seam of the model node:

* official fine-grained FP8 checkpoints are dequantized at load
  (:mod:`whirlwind.model.fp8_loader`, experimental);
* composite vision-language MoE checkpoints of the ``qwen3_5_moe`` family load
  their TEXT backbone only (``Qwen3_5MoeForCausalLM``; the vision tower is
  never needed for RL on text, and transformers remaps the keys);
* everything else goes through ``AutoModelForCausalLM``.

Training math is unchanged in every case — this only replaces weight ingestion.
"""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def _causal_lm_class(model_path: str, model_init_kwargs: dict[str, Any]) -> type:
    """The transformers class to load ``model_path`` with."""
    trust = bool(model_init_kwargs.get("trust_remote_code", False))
    model_type = getattr(
        AutoConfig.from_pretrained(model_path, trust_remote_code=trust), "model_type", ""
    )
    if model_type == "qwen3_5_moe":
        from transformers import Qwen3_5MoeForCausalLM

        logger.info("Model: qwen3_5_moe checkpoint — loading the TEXT backbone only")
        return Qwen3_5MoeForCausalLM
    return AutoModelForCausalLM


def load_causal_lm(
    model_path: str,
    *,
    torch_dtype: Any,
    attn_implementation: str,
    model_init_kwargs: dict[str, Any] | None = None,
):
    """Load a causal LM for training, dispatching on the checkpoint format."""
    from whirlwind.model import fp8_loader

    kwargs = dict(model_init_kwargs or {})
    if fp8_loader.is_fp8_checkpoint(model_path):
        logger.info("FP8 checkpoint detected at %s — dequantizing at load", model_path)
        return fp8_loader.from_pretrained_dequant(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
    model_cls = _causal_lm_class(model_path, kwargs)
    return model_cls.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        **kwargs,
    )
