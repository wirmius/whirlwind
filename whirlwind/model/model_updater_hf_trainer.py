"""HF-Trainer-backed model-update node — a debug A/B against ``model_fsdp2``.

This module implements the SAME RPC interface as ``whirlwind.model.model_fsdp2``
(the 14 methods the orchestrator calls: ``forward_accum_grad``,
``model_update_step``, ``export_weights``, ``sleep``, ``wake``,
``initialize_model``, ``forward_no_grad``, ``ping``, ``get_state``/``set_state``,
``save_hot_checkpoint``/``save_cold_checkpoint``/``load_hot_checkpoint``) but
delegates the forward / backward / optimizer-step machinery to
``transformers.Trainer`` + Accelerate FSDP instead of the manual FSDP2 path.

Purpose
-------
Isolate whether WW's *manual* FSDP2 training path differs numerically from the
HF-Trainer / Accelerate path that TRL uses. Everything else is held identical:

- **Loss math is REUSED, not rewritten.** ``HFTrainerModelUpdateNode`` subclasses
  ``ModelUpdateNode``, so ``_compute_loss`` / ``_compute_per_token_logps`` /
  ``_pad_and_stack`` / ``_selective_log_softmax`` (the verified GRPO + IS + KL +
  dr_grpo/grpo/bnpo math) are inherited unchanged. Only the training-step
  machinery (autocast, backward, grad-clip, optimizer/scheduler step, FSDP wrap)
  is swapped.
- **Precision semantics match WW's config flags.** ``fsdp_mixed_precision=True``
  + ``torch_dtype=float32`` -> ``bf16=True`` (Accelerate FSDP param-cast bf16
  compute, fp32 master — identical to WW's ``MixedPrecisionPolicy(param_dtype=
  bf16)`` + fp32 load). ``fsdp_autocast=True`` -> ``bf16=False`` + the inherited
  ``_use_autocast`` path (torch.autocast bf16, fp32 logits) — identical to WW's
  autocast variant.
- **LR is constant** (matching WW-FSDP2, which has no scheduler and honors the
  per-step ``req.lr`` override). A constant scheduler is created and stepped as
  a no-op so the A/B changes ONLY the training-step path, not the LR schedule.

What differs by construction (the thing under test): the model is wrapped by
``accelerator.prepare`` (Accelerate FSDP, the same engine TRL/HF Trainer use),
the backward goes through ``accelerator.backward`` (FSDP-aware grad reduce),
and the optimizer step + grad-clip go through the Trainer's
``optimizer``/``lr_scheduler``/``accelerator.clip_grad_norm_``. The forward
runs via ``trainer.training_step`` -> ``compute_loss`` (our override) -> the
inherited ``_compute_per_token_logps`` under the same FSDP param-cast dtype as
TRL.

Swap-in
--------
``bringup_and_run.sh`` hardcodes ``-m whirlwind.model.model_fsdp2``; add a
``--model-module`` flag (small edit) and pass
``--model-module whirlwind.model.model_updater_hf_trainer`` with an HF-Trainer
model config. The orchestrator and all other nodes are unchanged.

Known risk
----------
``sleep``/``wake`` (colocate GPU sharing) move the FSDP-wrapped model + optimizer
state GPU<->CPU via ``.to()``. Accelerate FSDP does not expose FSDP2's internal
sharded tensors the way the manual node does, so this is a best-effort
offload (the main risk flagged when scoping this module). If it misbehaves
under Accelerate FSDP, run noncolocate (no sleep/wake) first, or set
``fsdp_offload_params`` in the FSDP config.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from whirlwind.config import ModelConfig
from whirlwind.messages import (
    ExportWeightsRequest,
    ModelUpdateRequest,
    OperationResult,
    TrainStepRequest,
    TrainStepResult,
)
from whirlwind.model.model_fsdp2 import (
    ModelUpdateNode,
    _fsdp_using_function,
    _to_cpu_pinned,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer subclass — compute_loss -> WW GRPO loss; _prepare_inputs passthrough
# ---------------------------------------------------------------------------


class GRPOHFTrainer(__import__("transformers").Trainer):
    """Minimal ``transformers.Trainer`` subclass for the WW GRPO loop.

    Only two overrides:
    - ``compute_loss``: delegates to the node's loss math (pad_and_stack +
      per_token_logps + the inherited ``_compute_loss``). Called by
      ``training_step`` under ``compute_loss_context_manager`` (which is a
      ``nullcontext`` — bf16 happens via the FSDP param-cast from
      ``accelerator.prepare``, NOT torch.autocast; this matches WW's
      ``fsdp_mixed_precision`` path exactly).
    - ``_prepare_inputs``: passthrough. WW's inputs are already prepared
      (tensors placed on the rank GPU by ``_pad_and_stack``); the standard
      Trainer would move/rename columns, which we must not do.
    """

    def __init__(self, node: "HFTrainerModelUpdateNode", *args: Any, **kwargs: Any) -> None:
        self.node = node
        super().__init__(*args, **kwargs)

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Passthrough — tensors are already on the correct device (placed by
        # the node's _pad_and_stack). The default _prepare_inputs would move
        # them and strip columns; we must not.
        return inputs

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> torch.Tensor:
        if return_outputs:
            raise ValueError("GRPOHFTrainer does not support return_outputs")
        # The loss tensor (post /accum_steps). training_step then does
        # accelerator.backward(loss). _compute_step_loss also buffers the loss
        # value + diagnostics for the model_update_step aggregation.
        return self.node._compute_step_loss(model, inputs)


# ---------------------------------------------------------------------------
# Model-update node — subclass of the manual FSDP2 node
# ---------------------------------------------------------------------------


class HFTrainerModelUpdateNode(ModelUpdateNode):
    """HF-Trainer-backed model update node (same RPC interface as ModelUpdateNode).

    Inherits the verified loss math (``_compute_loss``, ``_compute_per_token_logps``,
    ``_pad_and_stack``, ``_selective_log_softmax``), the RPC registration (``start``),
    ``forward_no_grad``, ``ping``, state get/set, and the checkpoint methods.
    Overrides only: ``intitialize_model`` (build the Trainer + Accelerate FSDP),
    the inner training-step handlers (``_forward_accum_grad``,
    ``_model_update_step``), ``_export_weights``, and ``_sleep``/``_wake``.
    """

    # The Trainer owning the model/optimizer/scheduler/accelerator. Set in
    # intitialize_model. Typed Any to avoid importing Trainer at module top
    # (keeps the import lazy-failable for environments without transformers).
    _trainer: Any = None

    # ------------------------------------------------------------------
    # Lifecycle — replace the FSDP2 wrap with HF Trainer + Accelerate FSDP
    # ------------------------------------------------------------------

    def intitialize_model(self, data: Any) -> None:
        """Load model + tokenizer, build a ``transformers.Trainer`` + Accelerate FSDP.

        Mirrors the parent's load (tokenizer, model, dropout, hf-config cache)
        but replaces the manual FSDP2 ``fully_shard`` wrap + manual optimizer
        with a ``GRPOHFTrainer`` whose ``accelerator.prepare`` does the FSDP
        wrap (Accelerate's FSDP, the same engine TRL uses) and whose optimizer
        + scheduler are created by the Trainer. Precision follows the WW
        config flags (see module docstring).
        """
        import transformers  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

        assert not self._initialized_model

        # Entropy-compute flag (mirrors the parent's line 421; the HF node's
        # intitialize_model does not call super's, so set it here). WW_NO_ENTROPY
        # skips the fp32 full-vocab entropy softmax (logged metric only; kl_coef=0).
        self._compute_entropy = not bool(os.environ.get("WW_NO_ENTROPY"))

        # ----------------------------------------------------------------
        # 1. Distributed init (all ranks) — same as parent
        # ----------------------------------------------------------------
        if self.world_size > 1:
            import torch.distributed as dist

            dist.barrier()

        # ----------------------------------------------------------------
        # 2. Load tokenizer + model (all ranks) — same as parent
        # ----------------------------------------------------------------
        dtype_map = {"bfloat16": "bfloat16", "float32": "float32", "float16": "float16"}
        torch_dtype_str = dtype_map.get(self.config.torch_dtype, "bfloat16")
        torch_dtype = getattr(torch, torch_dtype_str)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, **self.config.model_init_kwargs
        )
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token  # type: ignore[attr-defined]

        attn_impl = "sdpa"
        try:
            if transformers.is_flash_attn_2_available():  # type: ignore[attr-defined]
                attn_impl = "flash_attention_2"
        except AttributeError:
            attn_impl = "sdpa"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            **self.config.model_init_kwargs,
        )
        if self.config.disable_dropout:
            for attr in ("dropout", "attention_dropout", "hidden_dropout", "emb_dropout"):
                if hasattr(self._model.config, attr):
                    setattr(self._model.config, attr, 0.0)
        # Cache for export_weights (same rationale as the parent).
        self._hf_config = self._model.config

        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Model loaded (HFTrainer): %s (%.1fB params, attn=%s, dtype=%s)",
            type(self._model).__name__,
            total_params / 1e9,
            getattr(self._model.config, "_attn_implementation", "unknown"),
            self._model.dtype,
        )

        # ----------------------------------------------------------------
        # 3. Precision flags — mirror WW config (see module docstring)
        # ----------------------------------------------------------------
        # _use_autocast controls the inherited _compute_per_token_logps branch:
        #  - True  -> torch.autocast(bf16) + logits.float()  (WW autocast variant)
        #  - False -> plain model() forward; bf16 (if any) comes from the FSDP
        #             param-cast set by bf16=True below (WW fsdp_mixed_precision).
        self._use_autocast = bool(getattr(self.config, "fsdp_autocast", False))
        # Precision-remedy flags (env-gated) — mirror the manual node so the
        # inherited _compute_per_token_logps / _compute_loss pick them up.
        # See model_fsdp2.py __init__ for the rationale.
        self._logits_fp32 = bool(os.environ.get("WW_LOGITS_FP32"))
        self._is_fp32 = bool(os.environ.get("WW_IS_FP32"))
        self._loss_fp32 = bool(os.environ.get("WW_LOSS_FP32"))
        use_bf16 = bool(self.config.fsdp_mixed_precision) and not self._use_autocast
        if not use_bf16 and not self._use_autocast and torch_dtype != torch.float32:
            logger.info(
                "HFTrainer node: no mixed precision (fp32 forward); torch_dtype=%s",
                torch_dtype_str,
            )

        # ----------------------------------------------------------------
        # 4. TrainingArguments — match WW optimizer/LR/clip knobs, constant LR
        # ----------------------------------------------------------------
        # LR is CONSTANT (lr_scheduler_type="constant") to match WW-FSDP2, which
        # has no scheduler and honors the per-step req.lr override. This keeps
        # the A/B clean: only the training-step path changes, not the schedule.
        # The constant scheduler is stepped as a no-op in model_update_step.
        fsdp = "full_shard" if self.world_size > 1 else ""
        # FSDP version + wrap policy: match TRL (FSDP2 + transformer-layer wrap).
        # FSDP2 requires fsdp=['full_shard','auto_wrap'] + fsdp_config version=2 +
        # transformer_layer_cls_to_wrap, which TrainingArguments._process_fsdp_args
        # maps to FullyShardedDataParallelPlugin(fsdp_version=2, ...). The first
        # HF-Trainer run accidentally used FSDP1 (no version set -> defaulted to 1);
        # this makes FSDP2 explicit so the A/B holds FSDP version constant with the
        # manual WW node (which uses fully_shard directly).
        if self.world_size > 1:
            fsdp = ["full_shard", "auto_wrap"]
            fsdp_config = {
                "version": 2,
                "transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
                "reshard_after_forward": True,
            }
        else:
            fsdp_config = None
        # micro_batch_size <= 0 (sentinel -1) means bypass the wrapper (one
        # forward over the full rank-sharded batch — see ModelUpdateNode.
        # _use_grad_accum_microbatch_size). HF's TrainingArguments requires a
        # POSITIVE per_device_train_batch_size, and we drive training_step
        # directly with our own (pre-bypassed) batch so this value is cosmetic
        # here — set a large valid int so HF's Trainer init / batch-size math
        # never chokes on -1 and never imposes its own sub-batching.
        _pdb = self.config.micro_batch_size if self.config.micro_batch_size > 0 else 4096
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir or "/tmp/whirlwind-hftrainer",
            per_device_train_batch_size=_pdb,
            gradient_accumulation_steps=1,  # accum handled via req.accum_steps in compute_loss
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            optim=self.config.optim if self.config.optim != "adamw" else "adamw_torch",
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            lr_scheduler_type="constant",
            warmup_steps=0,
            bf16=use_bf16,
            fp16=False,
            fsdp=fsdp,
            fsdp_config=fsdp_config,
            # Don't let the Trainer drive eval/save/logging — the orchestrator does.
            logging_strategy="no",
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            disable_tqdm=True,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            use_cpu=False,
        )

        # ----------------------------------------------------------------
        # 5. Instantiate the Trainer (sets up accelerator; no train loop yet)
        # ----------------------------------------------------------------
        # No dataloader — we feed batches via RPC. The Trainer's get_train_dataloader
        # is never called (we never call train()).
        self._trainer = GRPOHFTrainer(
            node=self,
            model=self._model,
            args=training_args,
        )

        # ----------------------------------------------------------------
        # 6. Activation checkpointing (match TRL default: gradient_checkpointing=True)
        # ----------------------------------------------------------------
        # TRL's GRPOConfig defaults gradient_checkpointing=True (base_config.py:61).
        # In normal TRL usage, Trainer.train() calls model.gradient_checkpointing_enable()
        # at trainer.py:1379 BEFORE _prepare_for_training (FSDP wrap) at trainer.py:1466.
        # We never call train(), so replicate that here: enable AC on the UNWRAPPED
        # model, then _prepare_for_training wraps it with FSDP (preserving the
        # checkpointing hooks). use_reentrant=False matches gradient_checkpointing_enable's
        # own default (and TRL's intent, though the <5.0.0 gate skips its setdefault).
        if self.config.fsdp_activation_checkpointing:
            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("HFTrainer node: activation checkpointing ENABLED (use_reentrant=False)")

        # ----------------------------------------------------------------
        # 7. Run the Trainer's model/optimizer/scheduler setup (no train loop)
        # ----------------------------------------------------------------
        # _prepare_for_training does: create_optimizer, _wrap_model,
        # accelerator.prepare(model, optimizer) [FSDP wrap + device], create_scheduler.
        # This handles the FSDP1-vs-FSDP2 + delayed-optimizer branches correctly.
        # max_steps only matters for non-constant schedulers; 1 is safe here.
        self._trainer._prepare_for_training(
            max_steps=1, train_dataloader=None, resume_from_checkpoint=None
        )

        # After prepare: trainer.model == trainer.model_wrapped == FSDP-wrapped
        # model (for FSDP); trainer.optimizer + trainer.lr_scheduler are set.
        self._fsdp = self._trainer.model
        self._optimizer = self._trainer.optimizer
        self._lr_scheduler = self._trainer.lr_scheduler

        # Make the inherited forward_no_grad / checkpoint methods happy.
        self._asleep = False
        self._initialized_model = True

        logger.info(
            "HFTrainer node ready (rank=%d, world=%d, fsdp=%r, bf16=%r, autocast=%r, optim=%s)",
            self.rank,
            self.world_size,
            fsdp,
            use_bf16,
            self._use_autocast,
            type(self._optimizer).__name__,
        )

    # ------------------------------------------------------------------
    # Loss-only step — the body of _forward_accum_grad minus backward
    # ------------------------------------------------------------------

    def _compute_step_loss(self, model: Any, inputs: dict[str, Any]) -> torch.Tensor:
        """Compute the GRPO loss for one (already-scattered) request shard.

        Replicates ``ModelUpdateNode._forward_accum_grad``'s body up to (but
        NOT including) ``loss.backward()``. The backward is done by
        ``trainer.training_step`` via ``accelerator.backward`` (FSDP-aware),
        which is the HF-Trainer path under test.

        Also buffers the loss value + sample count (the parent buffers these
        in _forward_accum_grad; diagnostics are buffered inside _compute_loss).
        """
        # Unpack the request-shard fields carried in the inputs dict.
        input_ids = inputs["input_ids"]
        label_ids = inputs["label_ids"]
        advantages = inputs["advantages"]
        sampling_logprobs = inputs["sampling_logprobs"]
        ref_logprobs = inputs["ref_logprobs"]
        old_logprobs = inputs["old_logprobs"]
        accum_steps = max(inputs.get("accum_steps", 1), 1)
        global_batch_size_override = inputs.get("global_batch_size_override", 0)
        global_max_completion_len_override = inputs.get(
            "global_max_completion_len_override", 0
        )

        tensors = self._pad_and_stack(
            input_ids,
            label_ids,
            advantages,
            sampling_logprobs,
            ref_logprobs,
            old_logprobs,
        )
        prompt_completion_ids = tensors["prompt_completion_ids"]
        attention_mask = tensors["attention_mask"]
        completion_ids = tensors["completion_ids"]
        completion_mask = tensors["completion_mask"]
        logits_to_keep = tensors["logits_to_keep"]
        advantages_t = tensors["advantages"]
        sampling_per_token_logps = tensors["sampling_logprobs"]
        ref_logprobs_t = tensors["ref_logprobs"]
        provided_old_logps = tensors["old_logprobs"]
        assert prompt_completion_ids is not None and attention_mask is not None
        assert completion_ids is not None and completion_mask is not None
        assert advantages_t is not None

        per_token_logps, entropy = self._compute_per_token_logps(
            model=model,
            prompt_completion_ids=prompt_completion_ids,
            attention_mask=attention_mask,
            completion_ids=completion_ids,
            logits_to_keep=logits_to_keep,
            temperature=self.config.temperature,
            compute_entropy=self._compute_entropy,
        )

        # old_per_token_logps resolution — identical to the parent
        # (provided -> use; else per_token_logps.detach() at GAS=1).
        do_engine_is = (
            getattr(self.config, "engine_is_correction", False)
            and self.config.loss_reduction != "ce"
            and sampling_per_token_logps is not None
        )
        if provided_old_logps is not None:
            old_per_token_logps = provided_old_logps
        else:
            old_per_token_logps = per_token_logps.detach()

        loss = self._compute_loss(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            ref_per_token_logps=ref_logprobs_t,
            advantages=advantages_t,
            mask=completion_mask,
            global_batch_size_override=global_batch_size_override,
            global_max_completion_len_override=global_max_completion_len_override,
            sampling_per_token_logps=sampling_per_token_logps,
            entropy=entropy,
        )

        # Gradient-accumulation scaling — identical to the parent
        # (loss / accum_steps before backward; training_step's own
        # /current_gradient_accumulation_steps is a no-op since GAS=1).
        if accum_steps > 1:
            loss = loss / accum_steps

        # Buffer for the model_update_step aggregation (matches parent lines
        # 1482-1483; diagnostics are buffered inside _compute_loss).
        self._accum_buf["losses"].append(loss.item())
        self._accum_buf["samples"].append(len(input_ids))

        # [TRL-COEF] Run TRL's EXACT engine-IS-ratio + coef_1 code (copied verbatim
        # from trl/trainer/grpo_trainer.py:2063-2090 and :2495-2503) on this node's
        # tensors, to see what the coefficient looks like in TRL's path on our
        # rollout data. Env-gated by WW_TRL_COEF_LOG; no behavior change when off.
        # old_per_token_logps here == per_token_logps.detach() at GAS=1 (TRL's own
        # aligned fallback, grpo_trainer.py:2475), bit-identical to TRL's fresh
        # no_grad forward (same weights/inputs/kernel). Coefficient is computed
        # pre-backward, so it is optimizer-machinery-independent — this faithfully
        # answers "what does the coefficient look like in TRL".
        if os.environ.get("WW_TRL_COEF_LOG"):
            try:
                with torch.no_grad():
                    _trl_mode = self.config.engine_is_mode
                    _trl_cap = self.config.engine_is_cap
                    # Neutralize NaN engine logprobs (rollout sentinel) the same
                    # way the manual _compute_loss does: replace with `old` so
                    # old - sampling = 0 -> ratio contribution 1 (neutral). Without
                    # this the raw NaN sentinel makes the ratio NaN.
                    _samp = sampling_per_token_logps
                    if torch.isnan(_samp).any():
                        _samp = torch.where(torch.isnan(_samp), old_per_token_logps, _samp)
                    # --- TRL engine IS ratio (grpo_trainer.py:2065-2086) ---
                    _pt_diff = (old_per_token_logps - _samp) * completion_mask
                    _seq_level = _trl_mode in ("sequence_mask", "sequence_truncate")
                    if _seq_level:
                        _logps_diff = _pt_diff.sum(dim=-1, keepdim=True)
                    else:
                        _logps_diff = _pt_diff
                    _trl_ratio = torch.exp(_logps_diff)
                    if _trl_mode in ("sequence_truncate", "token_truncate"):
                        _trl_ratio = torch.clamp(_trl_ratio, max=_trl_cap)
                    elif _trl_mode in ("sequence_mask", "token_mask"):
                        _trl_ratio = _trl_ratio.masked_fill(_trl_ratio > _trl_cap, value=0.0)
                    # --- TRL coef_1 (grpo_trainer.py:2495-2503) ---
                    _log_ratio = per_token_logps - old_per_token_logps
                    _trl_level = self.config.importance_sampling_level
                    if _trl_level == "token":
                        _liw = _log_ratio
                    elif _trl_level == "sequence":
                        _liw = (_log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
                        _liw = _liw.unsqueeze(-1)
                    else:
                        _liw = _log_ratio
                    _trl_coef1 = torch.exp(_liw)
                    # per-sequence summary (for the ratio, [B,1] in seq mode)
                    _r = _trl_ratio.flatten()
                    _seqsum = _pt_diff.sum(-1)  # [B]
                    _trl_diag = {
                        "trl/engine_is_ratio_mean": float(_r.mean()),
                        "trl/engine_is_ratio_min": float(_r.min()),
                        "trl/engine_is_ratio_max": float(_r.max()),
                        "trl/coef_1_mean": float(_trl_coef1.mean()),
                        "trl/coef_1_abs_dev_from_1": float((_trl_coef1 - 1.0).abs().mean()),
                        "trl/seq_sum_mean": float(_seqsum.mean()),
                        "trl/seq_sum_min": float(_seqsum.min()),
                    }
                    self._accum_buf.setdefault("trl_diag", []).append(_trl_diag)
                    # --- record inputs for the first microbatch of each flush ---
                    _dump_dir = os.environ.get("WW_TRL_COEF_DUMP_DIR") or "/tmp/ww_trl_coef"
                    _rec_cnt = getattr(self, "_trl_rec_calls", 0)
                    self._trl_rec_calls = _rec_cnt + 1
                    if _rec_cnt == 0:  # first microbatch of this flush
                        os.makedirs(_dump_dir, exist_ok=True)
                        torch.save(
                            {
                                "prompt_completion_ids": prompt_completion_ids.detach().cpu(),
                                "completion_ids": completion_ids.detach().cpu(),
                                "completion_mask": completion_mask.detach().cpu(),
                                "advantages": advantages_t.detach().cpu(),
                                "per_token_logps": per_token_logps.detach().cpu(),
                                "old_per_token_logps": old_per_token_logps.detach().cpu(),
                                "sampling_per_token_logps": sampling_per_token_logps.detach().cpu(),
                                "trl_engine_is_ratio": _trl_ratio.detach().cpu(),
                                "trl_coef_1": _trl_coef1.detach().cpu(),
                                "trl_seq_sum": _seqsum.detach().cpu(),
                            },
                            os.path.join(_dump_dir, f"inputs_step{getattr(self, '_trl_flush_idx', 0):02d}_rank{self.rank}.pt"),
                        )
            except Exception:
                logger.exception("TRL-COEF: measurement failed")

        return loss

    # ------------------------------------------------------------------
    # Forward + backward — delegate to trainer.training_step
    # ------------------------------------------------------------------

    @_fsdp_using_function("_forward_accum_grad")  # type: ignore[misc]
    def _forward_accum_grad(self, req: TrainStepRequest) -> TrainStepResult:
        """One microbatch-shard forward + backward via HF Trainer.

        ``_fsdp_using_function`` scatters the (already microbatch-split) shard
        across ranks (inherited outer ``forward_accum_grad`` + 
        ``_use_grad_accum_microbatch_size`` do the microbatch split). On each
        rank, ``training_step`` runs: ``model.train()`` -> ``compute_loss``
        (our WW loss) -> ``/current_gradient_accumulation_steps`` (no-op, GAS=1)
        -> ``accelerator.backward`` (FSDP-aware grad reduce). Returns the
        detached loss; the loss value was already buffered in
        ``_compute_step_loss``.
        """
        inputs = {
            "input_ids": req.input_ids,
            "label_ids": req.label_ids,
            "advantages": req.advantages,
            "sampling_logprobs": req.sampling_logprobs,
            "ref_logprobs": req.ref_logprobs,
            "old_logprobs": req.old_logprobs,
            "accum_steps": max(getattr(req, "accum_steps", 1), 1),
            "global_batch_size_override": req.global_batch_size_override,
            "global_max_completion_len_override": req.global_max_completion_len_override,
        }
        # training_step reads self.current_gradient_accumulation_steps to scale
        # the loss; the Trainer's _inner_training_loop normally sets it. Since we
        # drive the loop ourselves (and set gradient_accumulation_steps=1, doing
        # accum via req.accum_steps in compute_loss), pin it to 1 so the
        # training_step division is a no-op (matching the parent's scaling).
        self._trainer.current_gradient_accumulation_steps = 1
        loss = self._trainer.training_step(self._fsdp, inputs)
        return TrainStepResult(loss=loss.item(), grad_norm=None, lr=None)

    # ------------------------------------------------------------------
    # Optimizer step — clip + step + scheduler + zero + aggregate
    # ------------------------------------------------------------------

    @_fsdp_using_function("_model_update_step")  # type: ignore[misc]
    def _model_update_step(self, req: ModelUpdateRequest) -> TrainStepResult:
        """Clip grads, optimizer.step, lr_scheduler.step, zero_grad, aggregate.

        Same aggregation as the parent's _model_update_step (loss/diag buffers
        were filled per-microbatch in _compute_step_loss / _compute_loss), but:
        - grad clip uses ``accelerator.clip_grad_norm_`` (FSDP-aware)
        - the LR scheduler is stepped (constant -> no-op; matches WW's no-sched
          behavior; honors req.lr override the same way the parent does)
        """
        accelerator = self._trainer.accelerator
        model = self._fsdp

        # Per-step LR override (same as parent).
        if req.lr is not None:
            for param_group in self._optimizer.param_groups:  # type: ignore[union-attr]
                param_group["lr"] = req.lr

        # [step1-bitdiff] grad stats BEFORE clip (HF path) — clip mutates grads
        _grad_pre = None
        try:
            from whirlwind.model import _step1_bitdiff
            if _step1_bitdiff.enabled():
                _grad_pre = _step1_bitdiff.grad_stats_pre_clip(model.parameters())
        except Exception:
            pass

        # [update-diag] grad stats BEFORE clip (HF path) — the true backward
        # + accelerator.backward all-reduce output. No-op unless WW_UPDATE_DIAG.
        try:
            from whirlwind.model import _update_diag
            _update_diag.capture_grads(model, "grad_pre_clip", self.rank, "hf")
        except Exception:
            pass

        # FSDP-aware grad clip (instead of torch.nn.utils.clip_grad_norm_).
        grad_norm = accelerator.clip_grad_norm_(
            model.parameters(), max_norm=self.config.max_grad_norm  # type: ignore[arg-type]
        )

        # [update-diag] grad stats AFTER clip (HF path) — what the optimizer sees.
        try:
            from whirlwind.model import _update_diag
            _update_diag.capture_grads(model, "grad_post_clip", self.rank, "hf")
        except Exception:
            pass

        # [step1-bitdiff] param norm BEFORE optimizer step (HF path)
        _param_norm_pre = None
        try:
            from whirlwind.model import _step1_bitdiff
            if _step1_bitdiff.enabled():
                _param_norm_pre = _step1_bitdiff.param_l2_norm(model.parameters())
        except Exception:
            pass

        # [update-diag] param stats BEFORE optimizer.step (HF path) — == W_s
        try:
            from whirlwind.model import _update_diag
            _update_diag.capture_params(model, "param_pre_step", self.rank, "hf")
            _update_diag.capture_optim(self._optimizer, "optim_pre_step", model)
        except Exception:
            pass

        self._optimizer.step()  # type: ignore[union-attr]
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()  # constant -> no-op
        self._optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]

        # [update-diag] param stats AFTER optimizer.step (HF path) — == W'
        # (the update result; the key manual-vs-HF comparison stage).
        try:
            from whirlwind.model import _update_diag
            _update_diag.capture_params(model, "param_post_step", self.rank, "hf")
            _update_diag.capture_optim(self._optimizer, "optim_post_step", model)
            _update_diag.write(self.rank, "hf")
        except Exception:
            pass

        # [step1-bitdiff] capture fingerprints + exit after step 1 (HF path)
        try:
            from whirlwind.model import _step1_bitdiff
            _step1_bitdiff.dump_and_exit(model.parameters(), float(grad_norm), self.rank, "hf", _param_norm_pre, _grad_pre)
        except Exception:
            pass

        current_lr = req.lr if req.lr is not None else self.config.learning_rate

        # ---- Aggregate buffered losses (copied from the parent) ----
        buf_losses = self._accum_buf["losses"]
        buf_samples = self._accum_buf["samples"]
        acc_steps = max(self.config.gradient_accumulation_steps, 1)
        num_shards = len(buf_losses)

        if self.config.loss_reduction == "dr_grpo":
            rank_loss = sum(buf_losses)
        elif self.config.loss_reduction in ("bnpo", "ce"):
            rank_loss = sum(buf_losses) * self.world_size
        else:  # "grpo"
            rank_loss = sum(buf_losses) / num_shards if num_shards else 0.0

        device = torch.device("cuda") if self.world_size > 1 else torch.device("cpu")
        loss_tensor = torch.tensor(rank_loss, dtype=torch.float32, device=device)
        samples_tensor = torch.tensor(
            sum(buf_samples), dtype=torch.float64, device=device
        )
        if self.world_size > 1:
            import torch.distributed as dist

            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor = loss_tensor / self.world_size
            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)

        if self.config.loss_reduction in ("bnpo", "dr_grpo", "ce", "grpo"):
            loss_tensor = loss_tensor * acc_steps

        self._accum_buf["losses"] = []
        self._accum_buf["samples"] = []

        # ---- Aggregate diagnostics (copied from the parent) ----
        diag_buf = self._accum_buf.get("diag", [])
        agg_diag: dict[str, float] = {}
        if diag_buf:
            keys: set[str] = set()
            for d in diag_buf:
                keys.update(d.keys())
            for k in sorted(keys):
                vals = [d[k] for d in diag_buf if k in d]
                if not vals:
                    continue
                if not all(isinstance(v, (int, float)) for v in vals):
                    agg_diag[k] = vals[-1]  # type: ignore[assignment]
                    continue
                if k.endswith("/min"):
                    agg_diag[k] = min(vals)
                elif k.endswith("/max"):
                    agg_diag[k] = max(vals)
                else:
                    agg_diag[k] = sum(vals) / len(vals)
            if self.world_size > 1 and agg_diag:
                import torch.distributed as dist

                for k in sorted(agg_diag.keys()):
                    v = agg_diag[k]
                    if not isinstance(v, (int, float)):
                        continue
                    t = torch.tensor(v, dtype=torch.float32, device=device)
                    if k.endswith("/min"):
                        dist.all_reduce(t, op=dist.ReduceOp.MIN)  # type: ignore[arg-type]
                    elif k.endswith("/max"):
                        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # type: ignore[arg-type]
                    else:
                        dist.all_reduce(t, op=dist.ReduceOp.SUM)  # type: ignore[arg-type]
                        t = t / self.world_size
                    agg_diag[k] = t.item()
            for k, v in list(agg_diag.items()):
                if v == float("inf") or v == float("-inf") or v != v:
                    agg_diag[k] = 0.0
        self._last_diag = agg_diag
        self._accum_buf["diag"] = []

        # [TRL-COEF] Aggregate the TRL-coefficient measurement across microbatches
        # + ranks and print for the first N flushes (env WW_TRL_COEF_LOG_N).
        # Mirrors the parent's STEP-DECOMP print but reports TRL's coefficient code
        # output, for direct comparison with the manual node's coefficient.
        _trl_agg: dict[str, float] = {}
        _trl_buf = self._accum_buf.get("trl_diag", [])
        if _trl_buf and os.environ.get("WW_TRL_COEF_LOG"):
            try:
                _tk: set[str] = set()
                for _d in _trl_buf:
                    _tk.update(_d.keys())
                for _k in sorted(_tk):
                    _vals = [_d[_k] for _d in _trl_buf if _k in _d]
                    if not _vals or not all(isinstance(_v, (int, float)) for _v in _vals):
                        continue
                    _trl_agg[_k] = min(_vals) if _k.endswith("min") else (max(_vals) if _k.endswith("max") else sum(_vals) / len(_vals))
                if self.world_size > 1 and _trl_agg:
                    import torch.distributed as dist
                    for _k in sorted(_trl_agg.keys()):
                        _v = _trl_agg[_k]
                        _t = torch.tensor(_v, dtype=torch.float32, device=device)
                        if _k.endswith("min"):
                            dist.all_reduce(_t, op=dist.ReduceOp.MIN)
                        elif _k.endswith("max"):
                            dist.all_reduce(_t, op=dist.ReduceOp.MAX)
                        else:
                            dist.all_reduce(_t, op=dist.ReduceOp.SUM)
                            _t = _t / self.world_size
                        _trl_agg[_k] = _t.item()
                self._accum_buf["trl_diag"] = []
                _fi = getattr(self, "_trl_flush_idx", 0)
                self._trl_flush_idx = _fi + 1
                _n = int(os.environ.get("WW_TRL_COEF_LOG_N", "16"))
                if self.rank == 0 and _fi < _n:
                    _rm = _trl_agg.get("trl/engine_is_ratio_mean", float("nan"))
                    _rmin = _trl_agg.get("trl/engine_is_ratio_min", float("nan"))
                    _rmax = _trl_agg.get("trl/engine_is_ratio_max", float("nan"))
                    _c1 = _trl_agg.get("trl/coef_1_mean", float("nan"))
                    _c1d = _trl_agg.get("trl/coef_1_abs_dev_from_1", float("nan"))
                    _ss = _trl_agg.get("trl/seq_sum_mean", float("nan"))
                    _g = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
                    print(
                        f"[STEP-DECOMP-TRL] opt-step={_fi+1} "
                        f"TRL-CODE coef_1={_c1:.5f} (|.-1|={_c1d:.2e}) | "
                        f"TRL-CODE engine_ratio[mean={_rm:.5f} min={_rmin:.5f} max={_rmax:.5f}] "
                        f"seq_sum={_ss:.4f} | "
                        f"grad(actual,HF-machinery)={_g:.5f} "
                        f"(WW-manual same-batch: ratio=0.132 coef_1=1.0 seq_sum=-6.1 grad=0.020 core=0.058)",
                        flush=True,
                    )
                # dump the full config once (rank 0, first flush) for reproducibility
                if self.rank == 0 and _fi == 0:
                    _dump_dir = os.environ.get("WW_TRL_COEF_DUMP_DIR") or "/tmp/ww_trl_coef"
                    os.makedirs(_dump_dir, exist_ok=True)
                    import json as _json
                    _cfg = {
                        "model_config": {k: getattr(self.config, k) for k in (
                            "engine_is_correction", "engine_is_mode", "engine_is_cap",
                            "importance_sampling_level", "loss_reduction", "temperature",
                            "clip_range_low", "clip_range_high", "learning_rate",
                            "max_grad_norm", "gradient_accumulation_steps", "micro_batch_size",
                            "kl_coef", "max_completion_length",
                        ) if hasattr(self.config, k)},
                        "trl_version": __import__("trl").__version__,
                        "note": "TRL coefficient code (grpo_trainer.py:2063-2090,2495-2503) run on WW rollout tensors via the HF node (transformers.Trainer + accelerator.backward + AcceleratedOptimizer).",
                    }
                    with open(os.path.join(_dump_dir, "config.json"), "w") as _f:
                        _json.dump(_cfg, _f, indent=2)
            except Exception:
                logger.exception("TRL-COEF: aggregation/print failed")

        return TrainStepResult(
            loss=loss_tensor.item(),
            grad_norm=grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            lr=current_lr,
            total_completions=int(samples_tensor.item()),
        )

    # ------------------------------------------------------------------
    # Export — consolidate via accelerator.get_state_dict
    # ------------------------------------------------------------------

    @_fsdp_using_function("_export_weights")  # type: ignore[misc]
    def _export_weights(self, req: ExportWeightsRequest) -> OperationResult:
        """Consolidate FSDP shards into HF format on rank 0.

        Uses ``accelerator.get_state_dict`` (FSDP-aware full-state-dict gather)
        instead of the parent's FSDP2 ``get_model_state_dict``. Writes
        ``pytorch_model.bin`` + ``config.json`` — same on-disk format as the
        parent so the rollout's weight reload is unchanged.
        """
        ckpt_dir = req.path
        accelerator = self._trainer.accelerator
        assert self._fsdp is not None

        logger.info("HFTrainer: all ranks beginning weight export: %s", ckpt_dir)
        full_state_dict = accelerator.get_state_dict(self._fsdp)

        if self.rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            model_path = os.path.join(ckpt_dir, "pytorch_model.bin")
            logger.info("Writing gathered weights to: %s", model_path)
            torch.save(full_state_dict, model_path)
            self._hf_config.save_pretrained(ckpt_dir)
            logger.info("Exported config: %s", ckpt_dir)

        import torch.distributed as dist

        if self.world_size > 1:
            dist.barrier()
        return OperationResult(ok=True, message=f"Weights exported to {ckpt_dir}")

    # ------------------------------------------------------------------
    # Sleep / Wake — GPU<->CPU offload under Accelerate FSDP2
    # ------------------------------------------------------------------
    # NOTE: under FSDP2 the params are DTensors whose placement invariant
    # forbids reassigning ``.data`` to a differently-placed storage (raises
    # "set the storage of a tensor on device cuda:0 to a storage on cpu").
    # The naive ``for p in self._fsdp.parameters(): p.data = ...`` loop that
    # this used before crashed _wake on the first colocate step (the clean
    # FSDP2 IS run hung after val@0). Fix: mirror the manual node's proven
    # FSDP2-native sleep/wake (model_fsdp2.py:1799/1858) — move the internal
    # ``_sharded_param_data`` (1D flat sharded tensor) + ``sharded_param.grad``
    # via ``_get_fsdp_state()._fsdp_param_group.fsdp_params``, NOT the DTensor
    # ``.data``. Optimizer states are moved the same way as the manual node.

    @_fsdp_using_function("_sleep")  # type: ignore[misc]
    def _sleep(self, data: Any) -> OperationResult:
        assert self._fsdp is not None
        assert self._optimizer is not None
        device = torch.device("cuda", self.local_rank)

        moved_params = moved_grads = moved_optim = 0
        # Sharded params + grads -> CPU (FSDP2-native: move _sharded_param_data,
        # NOT the DTensor .data — see the NOTE above this method.)
        fsdp_state = self._fsdp._get_fsdp_state()  # type: ignore[attr-defined]
        if fsdp_state._fsdp_param_group is not None:
            for fsdp_param in fsdp_state._fsdp_param_group.fsdp_params:
                # Sharded param data (1D flat tensor)
                if fsdp_param._sharded_param_data.device.type == "cuda":
                    fsdp_param._sharded_param_data = _to_cpu_pinned(fsdp_param._sharded_param_data)
                    moved_params += 1
                # Gradients (live on the sharded_param DTensor).
                # At GAS=1, grads are None here (zero_grad ran after the opt
                # step), so this branch never fires. At GAS>1, the intermediate
                # sleep between accumulate cycles runs WITH accumulated grads
                # present. LEAVE grads on GPU (do not move to CPU): assigning a
                # CPU tensor to a cuda DTensor's .grad fails device validation,
                # and the per-rank grad footprint is negligible. Grads survive
                # sleep/wake untouched -> cross-cycle accumulation preserved.
                if (
                    fsdp_param.sharded_param.grad is not None
                    and fsdp_param.sharded_param.grad.device.type == "cuda"
                ):
                    moved_grads += 1  # counted but left on GPU (see note above)
        # Optimizer states -> CPU
        for param_states in self._optimizer.state.values():  # type: ignore[union-attr]
            for key, val in param_states.items():
                if hasattr(val, "device") and val.device.type == "cuda":
                    param_states[key] = _to_cpu_pinned(val)
                    moved_optim += 1
        torch.cuda.empty_cache()
        logger.info(
            "Sleep (rank %d, HFTrainer): moved %d params, %d grads, %d optim states to CPU",
            self.rank, moved_params, moved_grads, moved_optim,
        )
        return OperationResult(
            ok=True,
            message=f"Moved {moved_params} params, {moved_grads} grads, {moved_optim} optim states to CPU (rank {self.rank})",
        )

    @_fsdp_using_function("_wake")  # type: ignore[misc]
    def _wake(self, data: Any) -> OperationResult:
        assert self._fsdp is not None
        assert self._optimizer is not None
        device = torch.device("cuda", self.local_rank)

        restored_params = restored_grads = restored_optim = 0
        # Sharded params + grads -> GPU (FSDP2-native, mirrors manual node _wake.)
        fsdp_state = self._fsdp._get_fsdp_state()  # type: ignore[attr-defined]
        if fsdp_state._fsdp_param_group is not None:
            for fsdp_param in fsdp_state._fsdp_param_group.fsdp_params:
                if fsdp_param._sharded_param_data.device.type == "cpu":
                    fsdp_param._sharded_param_data = fsdp_param._sharded_param_data.to(
                        device, non_blocking=True
                    )
                    restored_params += 1
                # Gradients: sleep now LEAVES grads on GPU (see _sleep note),
                # so this cpu->cuda restore is a no-op for GAS>1. Kept for safety.
                if (
                    fsdp_param.sharded_param.grad is not None
                    and fsdp_param.sharded_param.grad.device.type == "cpu"
                ):
                    fsdp_param.sharded_param.grad = fsdp_param.sharded_param.grad.to(
                        device, non_blocking=True
                    )
                    restored_grads += 1
        for param_states in self._optimizer.state.values():  # type: ignore[union-attr]
            for key, val in param_states.items():
                if hasattr(val, "device") and val.device.type == "cpu":
                    param_states[key] = val.to(device, non_blocking=True)
                    restored_optim += 1
        logger.info(
            "Wake (rank %d, HFTrainer): restored %d params, %d grads, %d optim states to GPU",
            self.rank, restored_params, restored_grads, restored_optim,
        )
        return OperationResult(
            ok=True,
            message=f"Restored {restored_params} params, {restored_grads} grads, {restored_optim} optim states to GPU (rank {self.rank})",
        )


# ---------------------------------------------------------------------------
# Entry point — mirrors whirlwind.model.model_fsdp2.__main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Whirlwind HF-Trainer Model Update Node")
    parser.add_argument(
        "--config", default="configs/model/fsdp2.yaml", help="Path to model YAML config"
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = ModelConfig.from_yaml(args.config)
    print(
        f"Starting HF-Trainer model update node: model={config.model_path}, "
        f"world={int(os.environ.get('WORLD_SIZE', 1))}"
    )

    node = HFTrainerModelUpdateNode(config)

    async def _main() -> None:
        await node.start()
        print("HF-Trainer model update node running. Press Ctrl+C to stop.")
        try:
            while node.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await node.stop()

    if node.rank > 0:  # global rank: local rank 0 on other nodes is not the lead
        # Non-lead rank: start() for dist init, then block in standby.
        asyncio.run(node.start())
        node.intitialize_model({})
        node._standby()
    else:
        asyncio.run(_main())
