#!/usr/bin/env python
"""Fully Accelerate-driven model-update node — ZERO direct FSDP2 touches.

This node subclasses ``ModelUpdateNode`` (the manual FSDP2 node) ONLY to inherit:
  - The RPC contract + standby loop (``start``, ``_standby``, ``ping``, state get/set)
  - The GRPO loss math (``_compute_loss``, ``_compute_per_token_logps``,
    ``_selective_log_softmax``, ``_pad_and_stack``)
  - ``forward_no_grad`` (generation-time logprob forward)
  - The microbatch dispatch (``_use_grad_accum_microbatch_size``)

It REWRITES every model-management operation to go through Accelerate's public
API — no ``_get_fsdp_state``, no ``_sharded_param_data``, no ``FSDPModule``, no
``fully_shard``. The node could run on any Accelerate-supported backend
(FSDP1/2, DDP, DeepSpeed, single-GPU) by changing only the plugin config.

Rewritten to be fully Accelerate-driven:
  - ``intitialize_model``: HF-native AC + ``Accelerator.prepare`` (FSDP2 plugin)
  - ``_forward_accum_grad``: ``accelerator.backward(loss)``
  - ``_model_update_step``: ``accelerator.clip_grad_norm_`` + ``AcceleratedOptimizer``
  - ``_export_weights``: ``accelerator.get_state_dict``
  - ``_sleep``/``_wake``: optimizer-state CPU offload + ``torch.cuda.empty_cache``
    (backend-agnostic — no FSDP internals. Params stay on GPU; the colocate
    cycle frees optimizer-state + activation memory, which is the bulk. For
    full param offload, set ``cpu_offload: true`` on the FSDP plugin.)

Launch:
  --model-module whirlwind.model.model_accelerate_pure \\
  --model-config configs/model/fsdp2_base_colocate_gas16_fp32master_hfaligned_mbminus1.yaml
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
)

logger = logging.getLogger(__name__)


class AcceleratePureModelUpdateNode(ModelUpdateNode):
    """Fully Accelerate-driven model update node (zero direct FSDP2 touches).

    Inherits ONLY the loss math + RPC contract. Every model operation routes
    through Accelerate's public API.
    """

    _accelerator: Any = None

    # ------------------------------------------------------------------
    # Lifecycle — Accelerate-managed init
    # ------------------------------------------------------------------

    def intitialize_model(self, data: Any) -> dict:
        """Load model + tokenizer, HF-native AC, Accelerate FSDP2 wrap + optimizer."""
        import transformers  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        assert not self._initialized_model

        self._compute_entropy = not bool(os.environ.get("WW_NO_ENTROPY"))

        import torch.distributed as dist

        if self.world_size > 1:
            dist.barrier()

        # 1. Load tokenizer + model (same as parent).
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
        self._hf_config = self._model.config

        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Model loaded (AcceleratePure): %s (%.1fB params, attn=%s, dtype=%s)",
            type(self._model).__name__,
            total_params / 1e9,
            getattr(self._model.config, "_attn_implementation", "unknown"),
            self._model.dtype,
        )

        # 2. Precision flags (so inherited _compute_per_token_logps/_compute_loss
        #    pick them up). Under accelerator.prepare's autocast patching, the
        #    forward runs under torch.autocast(bf16) + convert_outputs_to_fp32,
        #    so logits come out fp32 regardless of _use_autocast/_logits_fp32.
        self._use_autocast = bool(getattr(self.config, "fsdp_autocast", False))
        self._logits_fp32 = bool(os.environ.get("WW_LOGITS_FP32"))
        self._is_fp32 = bool(os.environ.get("WW_IS_FP32"))
        self._loss_fp32 = bool(os.environ.get("WW_LOSS_FP32"))
        use_bf16 = bool(self.config.fsdp_mixed_precision) and not self._use_autocast

        # 3. HF-native activation checkpointing (on the UNWRAPPED model, before
        #    prepare — mirrors TRL's Trainer.train() ordering).
        if self.config.fsdp_activation_checkpointing:
            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("AcceleratePure: HF-native gradient checkpointing ENABLED")

        # 4. Optimizer (created on the unwrapped model; prepare wraps it).
        if self.config.optim == "adamw":
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optim == "sgd":
            self._optimizer = torch.optim.SGD(
                self._model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optim}")

        # 5. FSDP2 wrap via Accelerate (FullyShardedDataParallelPlugin).
        if self.world_size > 1:
            from accelerate import Accelerator
            from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin

            plugin = FullyShardedDataParallelPlugin(
                fsdp_version=2,
                transformer_cls_names_to_wrap=["Qwen3DecoderLayer"],
                reshard_after_forward=True,
            )
            if use_bf16:
                plugin.set_mixed_precision("bf16")
                acc = Accelerator(mixed_precision="bf16", fsdp_plugin=plugin)
            else:
                acc = Accelerator(mixed_precision="no", fsdp_plugin=plugin)
            self._accelerator = acc

            self._model, self._optimizer = acc.prepare(self._model, self._optimizer)
            logger.info(
                "AcceleratePure FSDP%d prepare done (rank=%d, world=%d, bf16=%r, "
                "autocast=%r, optim=%s)",
                fsdp_version, self.rank, self.world_size, use_bf16,
                self._use_autocast, type(self._optimizer).__name__,
            )
        else:
            self._accelerator = None
            dev = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(dev)

        self._fsdp = self._model  # inherited loss-math methods use self._fsdp

        import pathlib

        pathlib.Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._asleep = False
        self._initialized_model = True
        logger.debug("AcceleratePure model initialization complete.")
        return {}

    # ------------------------------------------------------------------
    # Forward + Backward — accelerator.backward
    # ------------------------------------------------------------------

    @_fsdp_using_function("_forward_accum_grad")  # type: ignore[misc]
    def _forward_accum_grad(self, req: TrainStepRequest) -> TrainStepResult:
        """One shard forward + backward, fully Accelerate-driven."""
        model = self._fsdp
        accelerator = self._accelerator

        model.train()

        tensors = self._pad_and_stack(
            req.input_ids, req.label_ids, req.advantages,
            req.sampling_logprobs, req.ref_logprobs, req.old_logprobs,
        )
        prompt_completion_ids = tensors["prompt_completion_ids"]
        attention_mask = tensors["attention_mask"]
        completion_ids = tensors["completion_ids"]
        completion_mask = tensors["completion_mask"]
        logits_to_keep = tensors["logits_to_keep"]
        advantages = tensors["advantages"]
        sampling_per_token_logps = tensors["sampling_logprobs"]
        ref_logprobs = tensors["ref_logprobs"]
        provided_old_logps = tensors["old_logprobs"]
        assert prompt_completion_ids is not None and attention_mask is not None
        assert completion_ids is not None and completion_mask is not None
        assert advantages is not None

        # Forward (inherited loss-math code — runs under Accelerate's autocast).
        per_token_logps, entropy = self._compute_per_token_logps(
            model=model,
            prompt_completion_ids=prompt_completion_ids,
            attention_mask=attention_mask,
            completion_ids=completion_ids,
            logits_to_keep=logits_to_keep,
            temperature=self.config.temperature,
            compute_entropy=self._compute_entropy,
        )

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
            ref_per_token_logps=ref_logprobs,
            advantages=advantages,
            mask=completion_mask,
            global_batch_size_override=req.global_batch_size_override,
            global_max_completion_len_override=req.global_max_completion_len_override,
            sampling_per_token_logps=sampling_per_token_logps,
            entropy=entropy,
        )

        accum_steps = max(getattr(req, "accum_steps", 1), 1)
        if accum_steps > 1:
            loss = loss / accum_steps

        # Backward via Accelerate (NOT loss.backward()).
        accelerator.backward(loss)

        self._accum_buf["losses"].append(loss.item())
        self._accum_buf["samples"].append(len(req.input_ids))

        return TrainStepResult(loss=loss.item(), grad_norm=None, lr=None)

    # ------------------------------------------------------------------
    # Model update — accelerator.clip_grad_norm_ + AcceleratedOptimizer
    # ------------------------------------------------------------------

    @_fsdp_using_function("_model_update_step")  # type: ignore[misc]
    def _model_update_step(self, req: ModelUpdateRequest) -> TrainStepResult:
        """Clip + optimizer step + zero grad, fully Accelerate-driven."""
        accelerator = self._accelerator
        model = self._fsdp

        if req.lr is not None:
            for param_group in self._optimizer.param_groups:  # type: ignore[union-attr]
                param_group["lr"] = req.lr

        # Grad clip via Accelerate (FSDP-aware unscale + clip).
        grad_norm = accelerator.clip_grad_norm_(
            model.parameters(), max_norm=self.config.max_grad_norm  # type: ignore[arg-type]
        )

        self._optimizer.step()  # type: ignore[union-attr]
        self._optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]

        # Aggregate loss + diagnostics (mirrors the parent's aggregation).
        buf_losses = self._accum_buf["losses"]
        buf_samples = self._accum_buf["samples"]
        total_samples = sum(buf_samples) if buf_samples else 0
        if total_samples > 0:
            loss_val = sum(l * s for l, s in zip(buf_losses, buf_samples)) / total_samples
        else:
            loss_val = sum(buf_losses) / max(len(buf_losses), 1)
        self._accum_buf["losses"] = []
        self._accum_buf["samples"] = []

        agg_diag: dict[str, float] = {}
        diag_buf = self._accum_buf.get("diag", [])
        if diag_buf:
            _tk: set[str] = set()
            for _d in diag_buf:
                _tk.update(_d.keys())
            for _k in sorted(_tk):
                _vals = [_d[_k] for _d in diag_buf if _k in _d]
                if not _vals or not all(isinstance(_v, (int, float)) for _v in _vals):
                    continue
                if _k.endswith("/min"):
                    agg_diag[_k] = min(_vals)
                elif _k.endswith("/max"):
                    agg_diag[_k] = max(_vals)
                else:
                    agg_diag[_k] = sum(_vals) / len(_vals)
            if self.world_size > 1 and agg_diag:
                import torch.distributed as dist

                for _k in sorted(agg_diag.keys()):
                    _v = agg_diag[_k]
                    if not isinstance(_v, (int, float)):
                        continue
                    _t = torch.tensor(_v, dtype=torch.float32, device=model.device)
                    if _k.endswith("/min"):
                        dist.all_reduce(_t, op=dist.ReduceOp.MIN)  # type: ignore[arg-type]
                    elif _k.endswith("/max"):
                        dist.all_reduce(_t, op=dist.ReduceOp.MAX)  # type: ignore[arg-type]
                    else:
                        dist.all_reduce(_t, op=dist.ReduceOp.SUM)  # type: ignore[arg-type]
                        _t = _t / self.world_size
                    agg_diag[_k] = _t.item()
            for _k, _v in list(agg_diag.items()):
                if _v == float("inf") or _v == float("-inf") or _v != _v:
                    agg_diag[_k] = 0.0
        self._last_diag = agg_diag
        self._accum_buf["diag"] = []

        # [STEP-DECOMP-PURE] intermediate-value trace (mirrors the manual node's
        # block, labeled PURE to distinguish in logs). Gated by WW_PRINT_STEP1_DIAG.
        try:
            import os as _os
            if _os.environ.get("WW_PRINT_STEP1_DIAG"):
                _n = int(_os.environ.get("WW_PRINT_STEP1_DIAG_N", "3"))
                cnt = getattr(self, "_step1_decomp_cnt", 0)
                if cnt < _n and self.rank == 0:
                    self._step1_decomp_cnt = cnt + 1
                    _g = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
                    _rmean = agg_diag.get("sampling/importance_sampling_ratio/mean", float("nan"))
                    _rmin = agg_diag.get("sampling/importance_sampling_ratio/min", float("nan"))
                    _rmax = agg_diag.get("sampling/importance_sampling_ratio/max", float("nan"))
                    _rumean = agg_diag.get("sampling/importance_sampling_ratio/uncapped_mean", float("nan"))
                    _seqsum = agg_diag.get("sampling/sampling_logp_difference/seq_sum_mean", float("nan"))
                    _signed = agg_diag.get("sampling/sampling_logp_difference/signed_mean", float("nan"))
                    _c1 = agg_diag.get("model_is/coef_1_mean", float("nan"))
                    _c1dev = agg_diag.get("model_is/coef_1_abs_dev_from_1", float("nan"))
                    _clip = agg_diag.get("clip_ratio/region_mean", float("nan"))
                    _lpre = agg_diag.get("loss/pre_is_mean", float("nan"))
                    _implied_core = (_g / _rmean) if _rmean and _rmean == _rmean and _rmean != 0 else float("nan")
                    print(
                        f"[STEP-DECOMP-PURE] opt-step={cnt+1} "
                        f"MODEL-IS coef_1={_c1:.5f} (|.-1|={_c1dev:.2e} clip_region={_clip:.4f}) | "
                        f"ENGINE-IS ratio[mean={_rmean:.5f} min={_rmin:.5f} max={_rmax:.5f} uncapped={_rumean:.5f}] "
                        f"logp_diff[seq_sum={_seqsum:.4f} signed={_signed:.5f}] | "
                        f"LOSS pre_is={_lpre:.5f} post_is={loss_val:.5f} | "
                        f"GRAD actual={_g:.5f} implied_core=actual/ratio={_implied_core:.5f}",
                        flush=True,
                    )
        except Exception:
            pass

        current_lr = self._optimizer.param_groups[0]["lr"] if self._optimizer else None  # type: ignore[union-attr]

        if self._node is not None:
            metrics: dict[str, float] = {
                "train/loss": loss_val,
                "train/grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
            }
            if current_lr is not None:
                metrics["train/lr"] = float(current_lr)
            if total_samples > 0:
                metrics["train/total_completions"] = float(total_samples)
            for k, v in self._last_diag.items():
                metrics[f"train/{k}"] = float(v) if isinstance(v, (int, float)) else v
            self._node.call_bg_no_return(
                "tracker", "log_metrics",
                {"metrics": metrics, "node_id": self.config.node_id},
            )

        samples_tensor = torch.tensor(total_samples, dtype=torch.float32, device=model.device)
        if self.world_size > 1:
            import torch.distributed as dist

            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)  # type: ignore[arg-type]

        return TrainStepResult(
            loss=loss_val,
            grad_norm=grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            lr=current_lr,
            total_completions=int(samples_tensor.item()),
        )

    # ------------------------------------------------------------------
    # Export — accelerator.get_state_dict (backend-agnostic)
    # ------------------------------------------------------------------

    @_fsdp_using_function("_export_weights")  # type: ignore[misc]
    def _export_weights(self, req: ExportWeightsRequest) -> OperationResult:
        """Consolidate via ``accelerator.get_state_dict`` (works for any backend)."""
        ckpt_dir = req.path
        accelerator = self._accelerator
        assert self._fsdp is not None

        logger.info("AcceleratePure: all ranks beginning weight export: %s", ckpt_dir)
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
    # Sleep / Wake — backend-agnostic GPU<->CPU offload
    # (NO FSDP2 internals: no _get_fsdp_state, no _sharded_param_data.
    #  Moves optimizer states to CPU + empty_cache. Params stay on GPU;
    #  FSDP's reshard_after_forward keeps unsharded params freed between
    #  forwards, so the dominant GPU cost during rollout is optimizer state.
    #  For full param offload, set cpu_offload on the FSDP plugin in config.)
    # ------------------------------------------------------------------

    @_fsdp_using_function("_sleep")  # type: ignore[misc]
    def _sleep(self, data: Any) -> OperationResult:
        """Offload optimizer states to CPU + free cache (backend-agnostic)."""
        assert self._fsdp is not None
        assert self._optimizer is not None

        moved = 0
        # Move optimizer states (exp_avg, exp_avg_sq for AdamW) to CPU.
        # This is pure PyTorch optimizer-state iteration — works for any backend.
        for param_states in self._optimizer.state.values():  # type: ignore[union-attr]
            for key, val in param_states.items():
                if hasattr(val, "device") and val.device.type == "cuda":
                    param_states[key] = val.to("cpu", non_blocking=True)
                    moved += 1
        torch.cuda.empty_cache()
        self._asleep = True
        logger.info(
            "Sleep (rank %d, AcceleratePure): moved %d optim states to CPU (backend-agnostic)",
            self.rank, moved,
        )
        return OperationResult(
            ok=True,
            message=f"Moved {moved} optim states to CPU (rank {self.rank})",
        )

    @_fsdp_using_function("_wake")  # type: ignore[misc]
    def _wake(self, data: Any) -> OperationResult:
        """Restore optimizer states to GPU (backend-agnostic)."""
        assert self._fsdp is not None
        assert self._optimizer is not None
        device = torch.device("cuda", self.local_rank)

        restored = 0
        for param_states in self._optimizer.state.values():  # type: ignore[union-attr]
            for key, val in param_states.items():
                if hasattr(val, "device") and val.device.type == "cpu":
                    param_states[key] = val.to(device, non_blocking=True)
                    restored += 1
        self._asleep = False
        logger.info(
            "Wake (rank %d, AcceleratePure): restored %d optim states to GPU (backend-agnostic)",
            self.rank, restored,
        )
        return OperationResult(
            ok=True,
            message=f"Restored {restored} optim states to GPU (rank {self.rank})",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Whirlwind Pure-Accelerate Model Update Node"
    )
    parser.add_argument("--config", default="configs/model/fsdp2.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = ModelConfig.from_yaml(args.config)
    print(
        f"Starting Pure-Accelerate model update node: model={config.model_path}, "
        f"world={int(os.environ.get('WORLD_SIZE', 1))}"
    )

    node = AcceleratePureModelUpdateNode(config)

    async def _main() -> None:
        await node.start()
        print("Pure-Accelerate model update node running. Press Ctrl+C to stop.")
        try:
            while node.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await node.stop()

    if node.rank > 0:  # global rank: local rank 0 on other nodes is not the lead
        asyncio.run(node.start())
        node.intitialize_model({})
        node._standby()
    else:
        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            pass
