"""Rollout node — runs vLLM Engine for LLM generation.

Drop-in replacement for ``whirlwind.rollout.rollout_sglang``: same ``RolloutNode``
class, same RPC interface (generate / update_weights / sleep / wake / ping /
get_state / set_state), same ``__main__`` CLI, same :class:`RolloutConfig`.

Differences from the SGLang module are mechanical and noted inline:
  * vLLM's ``LLM.generate`` is *synchronous*; the async ``_generate`` handler
    runs it in a thread executor (``loop.run_in_executor``) so the whirlcoms
    Node event loop is never blocked. Only one generate is ever in flight
    (the orchestrator serialises rollout calls), so this is thread-safe.
  * Weight reload from the on-disk HF checkpoint (``pytorch_model.bin`` +
    ``config.json`` — the format written by ``model_fsdp2._export_weights``)
    is done in-place via ``LLM.apply_model(lambda m: m.load_weights(iter))``:
    each worker reads the checkpoint from the shared ``/dev/shm`` path itself
    and copies weights into the live model parameters. CUDA graphs capture
    parameter *buffers*, not values, so in-place ``.copy_`` keeps them valid —
    no graph rebuild (mirrors SGLang's ``update_weights_from_disk``).
  * Memory sleep/wake maps to vLLM ``sleep(level=1)`` / ``wake_up()``.
  * ``prompt_token_ids`` come from vLLM's authoritative
    ``RequestOutput.prompt_token_ids`` (the SGLang module re-tokenises; vLLM's
    own token ids guarantee they match what the engine actually consumed).

Requires vLLM 0.21+ (developed against 0.24).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from whirlwind.config import RolloutConfig
from whirlwind.messages import (
    PromptBatch,
    RolloutResult,
    StateGetRequest,
    StateResult,
    StateSetRequest,
    WeightSyncRequest,
    _from_dict,
    _to_dict,
)
from whirlwind.whirlcoms import Node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level weight loader (must be picklable across processes)
# ---------------------------------------------------------------------------
# vLLM v1 runs the engine core in a *separate process*; ``LLM.apply_model``
# serializes its callable via msgspec and sends it across a pipe. Closures
# over ``self`` / local state cannot survive that (zmq sockets, the Node, etc.
# are not picklable), so the weight loader is a module-level function whose
# only non-picklable-free argument (the checkpoint path) is bound with
# ``functools.partial`` in ``_update_weights``. ``partial`` of a module-level
# function pickles *by reference* (module + qualname), so the worker subprocess
# just imports this module to resolve it — no closure bytecode, no ``self``.
# Requires ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` (set in ``set_up`` before
# ``LLM()`` so the spawned engine-core subprocess inherits it; the decode
# side also gates on it — serial_utils.py:370).


def _load_weights_from_file(model: Any, weight_file: str) -> Any:
    """Read an HF ``pytorch_model.bin`` and copy weights into ``model`` in-place.

    Called inside each vLLM worker via ``apply_model``: the worker passes its
    live model as ``model``; the checkpoint path was bound by ``partial``.
    ``load_weights`` copies each tensor into the live parameter buffer in-place
    (``.copy_``), so captured CUDA graphs stay valid — no rebuild.
    """
    import torch

    sd = torch.load(weight_file, map_location="cpu", weights_only=True)
    return model.load_weights(((k, v) for k, v in sd.items()))


class RolloutNode:
    """vLLM rollout node.

    Usage:
        node = RolloutNode(RolloutConfig(model_path="meta-llama/Llama-3.1-8B-Instruct"))
        await node.start()
        # ... run until done ...
        await node.stop()
    """

    def __init__(self, config: RolloutConfig) -> None:
        self.config = config
        self._node: Node | None = None
        self._engine: Any | None = None  # vllm.LLM
        self._tokenizer: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._asleep = False
        self._use_chat_template = False
        # LoRA adapter state (native adapter reload): the id/req of the
        # currently-loaded adapter, passed to vLLM.generate as lora_request so
        # generation uses base+adapter. None when no adapter is loaded yet
        # (full-weight path or before the first update_weights).
        self._active_lora_id: int | None = None
        self._active_lora_req: Any | None = None
        self._prev_lora_path: str | None = None  # prior adapter dir, deleted on next swap

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the underlying RPC node is still running (polled by the ``__main__`` keep-alive loop)."""
        return self._node.is_running if self._node is not None else False

    def set_up(self) -> None:
        """Start the vLLM engine, load tokenizer, and register with orchestrator.

        Must be called from within a running asyncio loop (the whirlcoms Node
        schedules background tasks on it). The ``__main__`` block below uses
        ``asyncio.run`` so the loop is already running when this is invoked.
        """
        from vllm import LLM

        self._loop = asyncio.get_running_loop()

        # vLLM v1's engine core runs in a subprocess; ``apply_model`` serializes
        # the callable via msgspec, which refuses functions unless this env var
        # enables a pickle fallback (serial_utils.py:214/231). Must be set BEFORE
        # ``LLM()`` so the spawned engine-core subprocess inherits it — the
        # decode side also gates on it (serial_utils.py:370). Safe here because
        # the only functions we serialize are our own module-level loader.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # Launch vLLM engine. Kwargs pass through LLM.__init__(**kwargs) ->
        # EngineArgs (verified fields: tensor_parallel_size, max_model_len,
        # gpu_memory_utilization, enable_sleep_mode, load_format).
        engine_kwargs: dict = {
            "model": self.config.model_path,
            "trust_remote_code": True,
            "dtype": "auto",
            "tensor_parallel_size": self.config.tp_size,
            "gpu_memory_utilization": self.config.mem_fraction_static,
            "max_model_len": self.config.context_length,
            "load_format": self.config.load_format,
            # vLLM sleep mode == SGLang memory_saver: enables sleep()/wake_up()
            # for the co-located sleep/wake cycle. Keep in sync with the training
            # config's memory_management_enabled (see sglang.yaml notes).
            "enable_sleep_mode": self.config.enable_memory_saver,
            # Return logprobs computed AFTER temperature scaling so they match
            # the trainer's convention. The model-update node divides logits by
            # the sampling temperature before log_softmax
            # (model_fsdp2.py:_compute_per_token_logps, temperature=
            # self.config.temperature, e.g. 0.7), so the IS ratio
            # exp(old_logps - sampling_logps) is only well-conditioned when the
            # engine's sampling_logps are at the SAME temperature. vLLM's
            # default mode ("raw_logprobs") computes log_softmax BEFORE
            # apply_temperature (sampler.py:88, pre-sample()), i.e. at T=1.0 —
            # a constant per-token mismatch of ~0.07 vs the trainer's T=0.7
            # logprobs that exp()s to ~exp(40) over a 570-token completion,
            # tripping the sequence_mask IS cap on every sequence and zeroing
            # the loss/gradient (model never learns). "processed_logprobs"
            # computes log_softmax AFTER apply_temperature (sampler.py:140,
            # inside topk_topp_sampler.forward_native), yielding post-temperature
            # logprobs that match the trainer — mirroring TRL's vLLMGeneration
            # (TRL's vllm_generation.py, comment "Important so
            # temperature scaling/logit tweaking affects the TIS log probs") and
            # SGLang's standard path (sampler.py: applies logits.div_(temperature)
            # before log_softmax). Note: setting this makes vLLM's TopKTopPSampler
            # route to forward_native (topk_topp_sampler.py:89-94: flashinfer is
            # disabled whenever processed_logprobs is set), so top_p/top_k still
            # work via the PyTorch-native sampler. Do not change without
            # updating the trainer's temperature (configs/model/fsdp2.yaml) to
            # match.
            "logprobs_mode": "processed_logprobs",
        }
        # LoRA: enable the engine's native adapter support at init. The adapter
        # itself is loaded per-step in _update_weights via add_lora; this only
        # arms the engine (allocates the LoRA manager + memory pool sized by
        # lora_max_rank). No peft import on the rollout side — the engine reads
        # the standard adapter dir (adapter_config.json + adapter_model.safetensors)
        # that the trainer's peft.save_pretrained wrote. See docs/lora_design.md.
        if self.config.enable_lora:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_loras"] = 1  # one active adapter (swap per step)
            engine_kwargs["max_lora_rank"] = self.config.lora_max_rank
        # NOTE on data parallelism: SGLang's ``dp_size`` spawns N DP workers
        # *inside one process* (so dp_size=4 works on a single GPU). vLLM's
        # ``data_parallel_size`` is fundamentally multi-process — one process
        # per DP rank, launched via the external_launcher example — and the
        # single-process ``LLM()`` refuses it (entrypoints/llm.py:291-300:
        # "LLM(data_parallel_size=N) is not supported for single-process usage
        # and may hang"). So we cannot honour dp_size>1 here. For multi-GPU
        # rollout throughput, use ``tensor_parallel_size`` instead (set
        # tp_size=N, dp_size=1 in the yaml). We warn and drop dp_size rather
        # than silently changing throughput, so a perf-matched A/B test
        # against SGLang isn't quietly corrupted.
        if self.config.dp_size > 1:
            logger.warning(
                "Rollout: vLLM single-process LLM does not support "
                "data_parallel_size>1 (it requires the multi-process DP "
                "launcher at examples/features/data_parallel/). SGLang's "
                "dp_size runs N workers in one process; vLLM cannot. "
                "Downgrading dp_size %d->1 (running single DP rank). For "
                "multi-GPU rollout use tensor_parallel_size instead "
                "(set tp_size=N, dp_size=1).",
                self.config.dp_size,
            )
        # Only forward data_parallel_size when it's 1 (the safe single-process
        # value) — passing dp_size=1 is a no-op but explicit; passing >1 would
        # hit the guard above and crash.
        if self.config.dp_size == 1:
            engine_kwargs["data_parallel_size"] = 1

        self._engine = LLM(**engine_kwargs)
        self._asleep = False
        logger.info("Rollout: vLLM LLM initialized (%s)", self.config.model_path)

        # Tokenizer for chat-template formatting (generation itself uses vLLM's
        # internal tokenizer; this mirrors the sglang module's setup).
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[misc]
            self.config.model_path,
            trust_remote_code=True,
        )
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token  # type: ignore[attr-defined]

        # Detect chat template (used to format dataset prompts)
        self._use_chat_template = bool(getattr(self._tokenizer, "chat_template", None))
        if self._use_chat_template:
            logger.info("Rollout: chat template detected in tokenizer")
        else:
            logger.info("Rollout: no chat template — using raw prompts")

        # Register as whirlcoms node on the running loop
        self._node = Node(
            self.config.node_id,
            auto_process=True,
            event_loop=self._loop,
        )
        self._node.expose("generate")(self._generate)
        self._node.expose("update_weights")(self._update_weights)
        self._node.expose("sleep")(self._sleep)
        self._node.expose("wake")(self._wake)
        self._node.expose("ping")(self._ping)
        self._node.expose("get_state")(self._get_state)
        self._node.expose("set_state")(self._set_state)

        logger.info(
            "Rollout node '%s' started (model=%s, engine=vLLM)",
            self.config.node_id,
            self.config.model_path,
        )

    async def start(self):
        await self._node.start()

    async def stop(self) -> None:
        if self._node:
            await self._node.stop()
        # vLLM's LLM has no public shutdown(); its engine workers (subprocesses
        # / threads) are cleaned up by the process exit. Drop our reference so
        # GPU memory is released promptly.
        self._engine = None
        logger.info("Rollout node stopped")

    # ------------------------------------------------------------------
    # State endpoints
    # ------------------------------------------------------------------

    def _full_state(self) -> dict[str, Any]:
        return {
            "config": _to_dict(self.config),
            "state": {"engine": "vllm"},
        }

    @StateGetRequest.decorate_serdes()
    def _get_state(self, req: StateGetRequest) -> StateResult:
        state = self._full_state()
        if req.keys:
            state = {k: v for k, v in state.items() if k in req.keys}
        return StateResult(data=state, ok=True)

    @StateSetRequest.decorate_serdes()
    def _set_state(self, req: StateSetRequest) -> StateResult:
        applied = {k: v for k, v in req.data.items() if not req.keys or k in req.keys}
        return StateResult(data=applied, ok=True)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    @PromptBatch.decorate_serdes()
    async def _generate(self, batch: PromptBatch) -> RolloutResult:
        """Generate completions for a batch of prompts.

        Returns both text (for debugging) and token_ids (for training).
        Dimensionality:
            completions:         [batch] str
            token_ids:           [batch, variable] int
            prompt_token_ids:    [batch, variable] int
            logprobs:            [batch, variable] float (per-token, optional)
        """
        t0 = time.monotonic()
        assert self._engine is not None
        assert self._tokenizer is not None
        engine = self._engine
        tokenizer = self._tokenizer

        sampling_params = self._build_sampling_params(batch)

        # Template all prompts up front (CPU, non-blocking)
        templated = [
            self._apply_chat_template(p, batch.system_prompt, batch.prefix_messages)
            for p in batch.prompts
        ]

        # vLLM.generate is synchronous — run it in a thread executor so the
        # Node's event loop keeps processing heartbeats / topology. Only one
        # generate is in flight at a time (orchestrator serialises rollouts).
        # When a LoRA adapter is loaded, pass its request so generation runs
        # base+adapter (vLLM applies the adapter to the loaded base model).
        lora_req = self._active_lora_req if self.config.enable_lora else None
        responses = await self._loop.run_in_executor(  # type: ignore[union-attr]
            None,
            lambda: engine.generate(
                templated, sampling_params, use_tqdm=False, lora_request=lora_req
            ),
        )

        # Extract per-response data (CPU-only, no awaits). vLLM returns a list
        # of RequestOutput in the same order as the input prompts; each has
        # .outputs[0] (the single completion, n=1) and .prompt_token_ids.
        completions: list[str] = []
        token_ids_list: list[list[int]] = []
        logprobs_list: list[list[float]] = []
        prompt_token_ids: list[list[int]] = []
        finish_reasons: list[str] = []

        for resp in responses:
            output = resp.outputs[0]  # n=1 → exactly one completion
            completions.append(output.text)

            comp_toks = list(output.token_ids)
            token_ids_list.append(comp_toks)

            # Per-token logprob of the *sampled* token. vLLM's
            # output.logprobs is list[dict[int, Logprob] | None], one entry
            # per generated position; the sampled token is comp_toks[i], so
            # its logprob is logprobs[i][comp_toks[i]].logprob. This mirrors
            # SGLang's meta_info.output_token_logprobs[i][0].
            if batch.return_logprobs and output.logprobs:
                probs: list[float] = []
                # logprobs=0 (set in _build_sampling_params) returns ONLY the
                # sampled token's logprob per position, so lp_dict always
                # contains comp_toks[pos] and the lookups below never miss.
                # A missing logprob (lp_dict None / sampled tok absent -- an
                # edge case that logprobs=0 makes essentially impossible) is
                # sent as NaN, NOT 0.0: 0.0 means "probability 1.0" and would
                # inject (old - 0) into the IS ratio, spuriously annihilating
                # the sequence. The model updater neutralizes NaN to a ratio
                # contribution of 1 (see model_fsdp2._compute_loss IS block).
                for pos, lp_dict in enumerate(output.logprobs):
                    if lp_dict is None or pos >= len(comp_toks):
                        probs.append(float("nan"))
                        continue
                    tok = comp_toks[pos]
                    lp = lp_dict.get(tok)
                    # vLLM 0.24 returns exactly 0.0 for high-probability sampled
                    # tokens (clamped; ~87% of tokens at temp 0.7). 0.0 is NOT
                    # a real logprob (prob=1.0 for 87% of tokens is impossible) --
                    # it leaks (old - 0.0)=old into the IS ratio and annihilates
                    # the sequence (proven from shadow-run traces: 87.3% of vLLM
                    # logprobs exactly 0.0 vs 11.6% for SGLang). Treat 0.0 as a
                    # sentinel and send NaN instead, so model_fsdp2's isnan() guard
                    # neutralizes it to `old` (old-old=0 -> ratio 1, no annihilation).
                    # This loses the real (tiny, near-0) logprob value, but the
                    # IS correction is ~a no-op at GAS=1 anyway, and neutrality is
                    # strictly better than annihilation. SGLang returns true small
                    # negatives (no clamping) so it is unaffected.
                    if lp is not None:
                        raw = float(lp.logprob)
                        probs.append(float("nan") if raw == 0.0 else raw)
                    else:
                        probs.append(float("nan"))
            else:
                # output.logprobs is None/empty for this response even though
                # it generated tokens (vLLM intermittently drops logprobs --
                # observed: 5-token completion, 0 logprobs). Start from [] and
                # fall through to the alignment below, which NaN-pads to
                # len(comp_toks). WITHOUT this alignment (the prior `append([])`)
                # the model node received a 0-length logprob list for a non-empty
                # completion; the old _pad_comp_lp then placed ZEROS across the
                # completion span -> IS ratio `old - 0` -> exp(sum(old)) ~ 3e-7
                # -> below the cap (not masked) -> per_token_loss *= 3e-7 ->
                # that sequence's loss ANNIHILATED. Intermittent (~1 in ~128
                # samples) -> progressive IS signal destruction -> the vLLM IS
                # low-plateau collapse. (SGLang's path always aligned; vLLM was
                # missing it.) NaN here -> model node's isnan() guard neutralizes
                # to ratio 1 (no annihilation). Mirrors rollout_sglang.py:232-238.
                probs = []
            # Align to len(comp_toks): truncate extras / NaN-pad shortfalls.
            # This is the guard the SGLang path has and vLLM lacked -- it makes
            # sampling_logprobs count == completion token count (the contract
            # model_fsdp2._pad_and_stack now asserts). NaN-pad (NOT 0.0): 0.0
            # means prob=1.0 and injects `old - 0` into the IS ratio (annihilates);
            # NaN is neutralized to ratio 1 at the model node.
            if len(probs) != len(comp_toks):
                if len(probs) > len(comp_toks):
                    probs = probs[: len(comp_toks)]
                else:
                    probs = probs + [float("nan")] * (len(comp_toks) - len(probs))
            logprobs_list.append(probs)

            # Authoritative prompt token ids (what vLLM actually consumed).
            # Falls back to re-tokenising the templated prompt if absent.
            pids = resp.prompt_token_ids
            if pids is None:
                pids = tokenizer(templated[len(prompt_token_ids)], return_tensors="pt").input_ids[
                    0
                ].tolist()
            prompt_token_ids.append(list(pids))

            # vLLM finish_reason strings: "stop", "length", "abort", "error",
            # "repetition" (vllm/v1/engine/__init__.py). "length" == truncated
            # by max_tokens, matching SGLang's "length".
            finish_reasons.append(output.finish_reason or "unknown")

        result = RolloutResult(
            completions=completions,
            token_ids=token_ids_list,
            prompt_token_ids=prompt_token_ids,
            logprobs=logprobs_list if any(lp for lp in logprobs_list) else None,
            dataset_ids=batch.dataset_ids,
            entry_ids=batch.entry_ids,
            rollout_ids=batch.rollout_ids,
            subrollout_ids=batch.subrollout_ids,
            finish_reasons=finish_reasons,
        )

        # Log metrics asynchronously (fire-and-forget, doesn't block)
        elapsed = time.monotonic() - t0
        total_tokens = sum(len(t) for t in token_ids_list)
        _grp = "val" if batch.log_prefix.startswith("val") else "train"
        comp_lengths = [len(t) for t in token_ids_list]
        is_truncated = [fr == "length" for fr in finish_reasons]
        n_comp = len(comp_lengths)
        term_lengths = [ln for ln, tr in zip(comp_lengths, is_truncated, strict=False) if not tr]
        rollout_metrics: dict[str, float] = {
            f"{_grp}/batch_size": len(batch.prompts),
            f"{_grp}/total_tokens": total_tokens,
            f"{_grp}/gen_time_s": elapsed,
            f"{_grp}/tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
            f"{_grp}/completions/mean_length": (sum(comp_lengths) / n_comp) if n_comp else 0.0,
            f"{_grp}/completions/min_length": min(comp_lengths) if comp_lengths else 0.0,
            f"{_grp}/completions/max_length": max(comp_lengths) if comp_lengths else 0.0,
            f"{_grp}/completions/clipped_ratio": (sum(is_truncated) / n_comp) if n_comp else 0.0,
        }
        if term_lengths:
            rollout_metrics[f"{_grp}/completions/mean_terminated_length"] = sum(term_lengths) / len(
                term_lengths
            )
            rollout_metrics[f"{_grp}/completions/min_terminated_length"] = min(term_lengths)
            rollout_metrics[f"{_grp}/completions/max_terminated_length"] = max(term_lengths)
        if self._node is not None:
            self._node.call_bg_no_return(
                "tracker",
                "log_metrics",
                {
                    "metrics": rollout_metrics,
                    "node_id": self.config.node_id,
                },
            )

            # Log prompts + completions table (fire-and-forget, throttled)
            should_log = batch.log_table
            override = self.config.log_every_call_override
            if override > 0:
                self._call_counter = getattr(self, "_call_counter", 0) + 1
                if self._call_counter % override == 0:
                    should_log = True
            if should_log:
                table_name = f"{batch.log_prefix}rollouts"
                table_rows = []
                for i in range(len(batch.prompts)):
                    row: list[str] = [
                        str(batch.step),
                        batch.dataset_ids[i] if i < len(batch.dataset_ids) else "",
                        str(batch.entry_ids[i]) if i < len(batch.entry_ids) else "",
                        str(batch.rollout_ids[i]) if i < len(batch.rollout_ids) else "",
                        str(batch.subrollout_ids[i]) if i < len(batch.subrollout_ids) else "",
                        templated[i],
                        completions[i],
                    ]
                    table_rows.append(row)
                self._node.call_bg_no_return(
                    "tracker",
                    "log_table",
                    {
                        "table_name": table_name,
                        "columns": [
                            "step",
                            "dataset_id",
                            "entry_id",
                            "rollout_id",
                            "subrollout_id",
                            "prompt",
                            "completion",
                        ],
                        "data": table_rows,
                        "node_id": self.config.node_id,
                        "step": batch.step,
                    },
                )
                logger.debug(f" >>> Sent the detailed rollout table to the tracker. {batch.step}")

        return result

    def _build_sampling_params(self, batch: PromptBatch) -> Any:
        """Build a vLLM ``SamplingParams`` from config + per-batch overrides.

        Mirrors the SGLang module's sampling_params dict (single params object
        for the whole batch). ``generation_kwargs`` are forwarded onto the
        SamplingParams fields that exist (filtered, so SGLang-only keys don't
        error).
        """
        from vllm import SamplingParams

        # vLLM: top_k=0 means disabled (SGLang uses -1 / None).
        top_k = self.config.top_k if self.config.top_k > 0 else 0

        kwargs: dict[str, Any] = {
            "max_tokens": self.config.max_gen_len,
            "temperature": max(0.0, batch.temperature or self.config.temperature),
            "top_p": batch.top_p if batch.top_p is not None else self.config.top_p,
            "top_k": top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "n": 1,
            # Include special tokens in output text/token_ids — matches SGLang
            # (skip_special_tokens=False) so downstream token ids align exactly.
            "skip_special_tokens": False,
            "detokenize": True,
        }
        # Request the sampled token's logprob when the batch asks for logprobs.
        # logprobs=0 returns ONLY the sampled token's logprob per position (vLLM
        # always includes the sampled token) -- matching TRL (grpo_trainer.py:796,
        # "we only need the generated token logprobs"). logprobs=1 (top-1+sampled)
        # was over-requesting and, with processed_logprobs_mode, vLLM returned
        # None for many positions, tripping the 0.0 fallback and annihilating
        # ~87% of sequences' IS ratios. logprobs=0 eliminates that at the source.
        # logprobs=None returns none.
        if batch.return_logprobs:
            kwargs["logprobs"] = 0

        # Forward generation_kwargs, keeping only valid SamplingParams fields.
        # SamplingParams is a msgspec.Struct in vLLM 0.24+ (not a dataclass),
        # so introspect via __struct_fields__; fall back to dataclass fields
        # for older vLLM versions (0.21-0.23 used @dataclass).
        if hasattr(SamplingParams, "__struct_fields__"):
            valid_fields = set(SamplingParams.__struct_fields__)
        else:
            import dataclasses
            valid_fields = {f.name for f in dataclasses.fields(SamplingParams)}
        for k, v in (self.config.generation_kwargs or {}).items():
            if k in valid_fields:
                kwargs[k] = v
            else:
                logger.debug("Rollout: ignoring generation_kwargs[%r] (not a SamplingParams field)", k)

        return SamplingParams(**kwargs)

    def _apply_chat_template(
        self,
        raw_prompt: str,
        system_prompt: str | None = None,
        prefix_messages: list[dict] | None = None,
    ) -> str:
        """Apply the tokenizer's chat template to a raw prompt string.

        Wraps the prompt as a single user message and formats it
        using ``apply_chat_template`` with ``add_generation_prompt=True``
        so the output ends with the assistant turn marker.

        If ``system_prompt`` is provided, a system message is prepended.

        ``prefix_messages`` (few_shot_eos): the blob style is kept —
        see the rollout_sglang twin: examples fold back into the system
        message as 'Q:/A:' pairs with the DECLARED EOS appended after each
        example; the only delta vs the legacy prompt is the EOS.
        """
        eos = getattr(self._tokenizer, "eos_token", "") or ""

        def _fold(prefix: list[dict]) -> str:
            parts: list[str] = []
            i = 0
            while i + 1 < len(prefix):
                u = str(prefix[i].get("content", ""))
                a = str(prefix[i + 1].get("content", ""))
                if eos:
                    a = a + eos
                parts.append(f"Q: {u}\nA: {a}")
                i += 2
            if i < len(prefix):
                parts.append(str(prefix[i].get("content", "")))
            return "\n\n".join(parts)

        folded = _fold(prefix_messages) if prefix_messages else ""
        sys_content = (
            (system_prompt + "\n\n" + folded)
            if (system_prompt and folded)
            else (system_prompt or folded or None)
        )
        if not self._use_chat_template:
            assert self._tokenizer is not None
            parts = [p for p in (sys_content, raw_prompt) if p]
            return "\n\n".join(parts)
        assert self._tokenizer is not None
        messages: list[dict[str, str]] = []
        if sys_content:
            messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": raw_prompt})
        return self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def _update_weights(self, data: dict[str, Any]) -> dict[str, Any]:
        """Load new checkpoint from disk into the running engine (in-place).

        Two modes:
          - LoRA adapter (enable_lora=True, dir contains adapter_config.json):
            load the adapter natively via ``engine.add_lora`` and tag it as the
            active adapter for subsequent generate calls. The base model stays
            loaded; only the adapter (MB) is swapped. No peft import — vLLM
            reads the standard PEFT adapter dir directly.
          - Full weights (default): the checkpoint is an HF directory with
            ``pytorch_model.bin`` + ``config.json`` (written by
            ``model_fsdp2._export_weights``). Each vLLM worker reads the file
            itself from the shared path and copies the weights into the live
            model parameters via ``model.load_weights``. CUDA graphs capture
            parameter buffers, not values, so in-place copies keep the captured
            graphs valid (mirrors SGLang's ``update_weights_from_disk``).
        """
        req = _from_dict(WeightSyncRequest, data)
        assert self._engine is not None
        ckpt_path = req.checkpoint_path
        logger.info("Rollout: updating weights from %s (step=%d)", ckpt_path, req.step)

        # --- LoRA adapter path ---
        if self.config.enable_lora and os.path.exists(
            os.path.join(ckpt_path, "adapter_config.json")
        ):
            from vllm.lora.request import LoRARequest

            # Swap: remove the previous adapter (if any) before loading the new
            # one to keep one active adapter (max_loras=1). lora_int_id uses the
            # step number to keep IDs unique across the run.
            if self._active_lora_id is not None:
                try:
                    self._engine.llm_engine.remove_lora(self._active_lora_id)
                except Exception as exc:  # already evicted / first step
                    logger.debug("remove_lora(%d) skipped: %s", self._active_lora_id, exc)
            lora_id = req.step + 1  # vLLM requires int_id >= 1
            lora_req = LoRARequest(
                lora_name=f"step{req.step}", lora_int_id=lora_id, lora_path=ckpt_path
            )
            self._engine.llm_engine.add_lora(lora_req)
            self._active_lora_id = lora_id
            self._active_lora_req = lora_req
            logger.info("Rollout: LoRA adapter loaded (id=%d, step=%d)", lora_id, req.step)
            # Delete the PREVIOUS adapter dir (vLLM read the current one into
            # GPU memory at add_lora time; the dir is no longer needed). Keep
            # the current one until the next swap as a safety margin.
            if self._prev_lora_path and os.path.isdir(self._prev_lora_path):
                import shutil

                shutil.rmtree(self._prev_lora_path, ignore_errors=True)
            self._prev_lora_path = ckpt_path
            return {"ok": True, "step": req.step}

        # --- Full weight path (existing) ---
        import functools

        weight_file = os.path.join(ckpt_path, "pytorch_model.bin")

        # Bind the checkpoint path to the module-level loader via ``partial``.
        # ``apply_model`` ships this to each worker subprocess (serialized via
        # msgspec→pickle); ``partial`` of a module-level function pickles by
        # reference, so the worker imports ``_load_weights_from_file`` and calls
        # it with its live model. Each worker reads pytorch_model.bin from the
        # shared /dev/shm path itself (no tensor pickling across workers), and
        # ``load_weights`` copies into the live params in-place — CUDA graphs
        # capture parameter buffers, not values, so they stay valid (mirrors
        # SGLang's ``update_weights_from_disk``).
        loader = functools.partial(_load_weights_from_file, weight_file=weight_file)

        def _load() -> None:
            self._engine.apply_model(loader)

        await self._loop.run_in_executor(None, _load)  # type: ignore[union-attr]
        logger.info("Rollout: weights updated successfully (step=%d)", req.step)
        # The export dir is deleted by its creator (the orchestrator's _call_sync)
        # once every consumer has loaded it; deleting it here raced the other pool
        # backends reading the same /dev/shm path.

        return {"ok": True, "step": req.step}

    async def _sleep(self, data: dict[str, Any]) -> dict[str, Any]:
        """Release GPU memory (offload weights to CPU + discard KV cache).

        Maps SGLang's ``release_memory_occupation`` to vLLM ``sleep(level=1)``.
        The engine must be idle (no ongoing requests) before calling this.

        LoRA + cumem: vLLM's LoRA adapter memory pool is NOT tracked by the
        cumem allocator (vllm #39934/#48297), so ``sleep()`` after an
        ``add_lora`` crashes with ``CUDART illegal memory access`` when cumem
        tries to offload the untracked region. Fix: unload the adapter BEFORE
        sleeping (frees the LoRA pool), and reload it AFTER waking. The base
        model weights ARE cumem-tracked and sleep/wake fine. The adapter is
        tiny (~MB) so the reload is cheap. See docs/lora_design.md.
        """
        assert self._engine is not None
        if self._asleep:
            return {"ok": True}
        # Unload the LoRA adapter before sleep so cumem only offloads the
        # base weights (which it tracks). Remember it for reload on wake.
        if self.config.enable_lora and self._active_lora_id is not None:
            try:
                self._engine.llm_engine.remove_lora(self._active_lora_id)
            except Exception as exc:  # already evicted
                logger.debug("pre-sleep remove_lora(%d) skipped: %s", self._active_lora_id, exc)
            logger.info(
                "Rollout: unloaded LoRA adapter (id=%d) before sleep", self._active_lora_id
            )
        logger.info("Rollout: releasing GPU memory (sleep)")
        await self._loop.run_in_executor(  # type: ignore[union-attr]
            None, lambda: self._engine.sleep(level=1)
        )
        self._asleep = True
        return {"ok": True}

    async def _wake(self, data: dict[str, Any]) -> dict[str, Any]:
        """Restore GPU memory (reload weights + reallocate KV cache) after sleep.

        Reloads the LoRA adapter that was unloaded before sleep (see _sleep),
        so generation can resume with base+adapter.
        """
        assert self._engine is not None
        if not self._asleep:
            return {"ok": True}
        logger.info("Rollout: reclaiming GPU memory (wake)")
        await self._loop.run_in_executor(  # type: ignore[union-attr]
            None, self._engine.wake_up
        )
        self._asleep = False
        # Reload the LoRA adapter that was unloaded before sleep (same id/path).
        if self.config.enable_lora and self._active_lora_req is not None:
            try:
                self._engine.llm_engine.add_lora(self._active_lora_req)
                logger.info(
                    "Rollout: reloaded LoRA adapter (id=%d) after wake",
                    self._active_lora_id,
                )
            except Exception as exc:
                logger.warning("post-wake add_lora failed: %s", exc)
        return {"ok": True}

    def _ping(self) -> dict[str, Any]:
        return {"ok": True, "engine_initialized": self._engine is not None}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whirlwind Rollout Node (vLLM)")
    parser.add_argument(
        "--config",
        default="configs/rollout/vllm.yaml",
        help="Path to rollout YAML config",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = RolloutConfig.from_yaml(args.config)
    print(f"Starting rollout node (vLLM): model={config.model_path}, tp={config.tp_size}")

    node = RolloutNode(config)

    async def _main() -> None:
        # set_up() builds the vLLM LLM (blocking model load + CUDA graph
        # capture) and registers the whirlcoms Node on this running loop.
        node.set_up()
        await node.start()
        print("Rollout node running. Press Ctrl+C to stop.")
        try:
            while node.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await node.stop()

    asyncio.run(_main())
