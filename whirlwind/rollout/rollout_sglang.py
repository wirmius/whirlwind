"""Rollout node — runs SGLang Engine for LLM generation.

Exposes:
    generate        — sample completions from prompts (returns tokens + text)
    update_weights  — load new checkpoint without rebuilding CUDA graphs
    ping            — health check
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


def _dp_dispatch_order(keys: list, dp_size: int) -> list[int]:
    """Request order that sends every copy of a prompt to the same DP worker.

    SGLang's data-parallel controller routes request i of a batch to worker
    (c0 + i) % dp_size (round robin), and all workers share one sampling seed.
    The orchestrator sends a prompt's copies back to back, so copy k landed on
    worker k and identical workers returned identical samples (8k val@0:
    copies 1-3 byte-identical for every prompt). Here each distinct key gets
    one "column" (position residue mod dp_size); within a worker the copies
    are separate batch rows and are sampled independently. If the groups
    cannot fill equal columns, the order is a fixed shuffle instead, which
    still breaks the lockstep between workers.
    """
    n = len(keys)
    if dp_size <= 1 or n <= 1:
        return list(range(n))
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    cols: list[list[int]] = [[] for _ in range(dp_size)]
    for g in sorted(groups.values(), key=len, reverse=True):
        min(cols, key=len).extend(g)
    if len({len(c) for c in cols}) == 1:
        return [cols[r][j] for j in range(len(cols[0])) for r in range(dp_size)]
    import random

    order = list(range(n))
    random.Random(n).shuffle(order)
    return order


logger = logging.getLogger(__name__)


def _finish_reason_str(meta: dict, default: str = "unknown") -> str:
    """Normalize SGLang's finish_reason to the plain string every consumer compares
    against ("length" / "stop"). SGLang reports a DICT ({"type": "length",
    "length": N} or {"type": "stop", "matched": id}); passing it through meant
    ``fr == "length"`` was never true: mask_truncated never masked, clipped_ratio
    read 0.0 even with every completion at the cap."""
    fr = meta.get("finish_reason", default)
    return str(fr.get("type", default)) if isinstance(fr, dict) else str(fr)


class RolloutNode:
    """SGLang rollout node.

    Usage:
        node = RolloutNode(RolloutConfig(model_path="meta-llama/Llama-3.1-8B-Instruct"))
        await node.start()
        # ... run until done ...
        await node.stop()
    """

    def __init__(self, config: RolloutConfig) -> None:
        self.config = config
        self._node: Node | None = None
        self._engine: Any | None = None
        self._tokenizer: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the underlying RPC node is still running (polled by the ``__main__`` keep-alive loop)."""
        return self._node.is_running if self._node is not None else False

    def _engine_kwargs(self) -> dict:
        """Map the rollout config to SGLang ``Engine`` kwargs (override to extend)."""
        engine_kwargs: dict = {
            "model_path": self.config.model_path,
            "load_format": self.config.load_format,
            "tp_size": self.config.tp_size,
            "dp_size": self.config.dp_size,
            "context_length": self.config.context_length,
            "mem_fraction_static": self.config.mem_fraction_static,
            "schedule_policy": self.config.schedule_policy,
            "enable_memory_saver": self.config.enable_memory_saver,
            "enable_weights_cpu_backup": self.config.enable_weights_cpu_backup,
            "disable_cuda_graph": self.config.disable_cuda_graph,
        }
        if getattr(self.config, "allow_auto_truncate", False):
            engine_kwargs["allow_auto_truncate"] = True
        if self.config.chunked_prefill_size > 0:
            engine_kwargs["chunked_prefill_size"] = self.config.chunked_prefill_size

        # RL on-policy mode: make SGLang compute logprobs with torch.log_softmax
        # (matching the trainer's forward pass) instead of log(softmax()), which
        # is numerically unstable in bf16. Eliminates the spurious per-token
        # drift that inflates the sequence_mask IS ratio. See sampler.py:144-154.
        if getattr(self.config, "rl_on_policy_target", None) is not None:
            engine_kwargs["rl_on_policy_target"] = self.config.rl_on_policy_target

        # LoRA: arm SGLang's native adapter support at init. The adapter is
        # loaded per-step in _update_weights via load_lora_adapter; this only
        # configures the engine (--enable-lora, --max-lora-rank,
        # --lora-target-modules). No peft import on the rollout side — SGLang
        # reads the standard PEFT adapter dir directly. See docs/lora_design.md.
        if self.config.enable_lora:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_lora_rank"] = self.config.lora_max_rank
            engine_kwargs["lora_target_modules"] = self.config.lora_target_modules

        return engine_kwargs

    def set_up(self) -> None:
        """Start the SGLang engine, load tokenizer, and register with orchestrator."""
        from sglang import Engine

        # Launch SGLang engine
        self._engine = Engine(**self._engine_kwargs())
        self._asleep = False
        # LoRA adapter state (native adapter reload): the name of the
        # currently-loaded adapter (passed to generate via lora request, or
        # applied automatically once loaded). None before the first update.
        self._active_lora_name: str | None = None
        self._active_lora_path: str | None = None  # disk path of the loaded adapter (passed to async_generate as lora_path)
        self._prev_lora_path: str | None = None  # prior adapter dir, deleted on next swap
        logger.info("Rollout: Engine initialized (%s, lora=%s)", self.config.model_path, self.config.enable_lora)

        # Load tokenizer for post-generation tokenization
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

        # Register as whirlcoms node
        self._node = Node(self.config.node_id, auto_process=True, event_loop=self._engine.loop)
        self._node.expose("generate")(self._generate)
        self._node.expose("update_weights")(self._update_weights)
        self._node.expose("sleep")(self._sleep)
        self._node.expose("wake")(self._wake)
        self._node.expose("ping")(self._ping)
        self._node.expose("get_state")(self._get_state)
        self._node.expose("set_state")(self._set_state)

        logger.info(
            "Rollout node '%s' started (model=%s)",
            self.config.node_id,
            self.config.model_path,
        )

    async def start(self):
        await self._node.start()

    async def stop(self) -> None:
        if self._node:
            await self._node.stop()
        if self._engine:
            self._engine.shutdown()
        logger.info("Rollout node stopped")

    # ------------------------------------------------------------------
    # State endpoints
    # ------------------------------------------------------------------

    def _full_state(self) -> dict[str, Any]:
        return {
            "config": _to_dict(self.config),
            "state": {},
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

        sampling_params = {
            "max_new_tokens": self.config.max_gen_len,
            "temperature": max(0.0, batch.temperature or self.config.temperature),
            "top_p": batch.top_p if batch.top_p is not None else self.config.top_p,
            "top_k": self.config.top_k if self.config.top_k > 0 else None,
            "repetition_penalty": self.config.repetition_penalty,
        }
        # Override with user-provided generation kwargs
        sampling_params.update(self.config.generation_kwargs)
        sampling_params.update(self._extra_sampling_params(batch))
        # Ensure special tokens are included in the output text and token_ids
        # so they match SGLang's internal tokenization exactly.
        sampling_params["skip_special_tokens"] = False

        # Template all prompts up front (CPU, non-blocking)
        if not getattr(self, "_logged_prefix_msgs", False):
            self._logged_prefix_msgs = True
            logger.info(
                "First batch at the rollout: system prompt %d chars, few-shot turns: %s",
                len(batch.system_prompt or ""),
                (f"{len(batch.prefix_messages)} turns" if batch.prefix_messages else "None"),
            )
        templated = self._template_prompts(batch)

        # Single batched generation call — SGLang handles variable-length
        # prompts. Its returned logprobs are full-vocabulary log_softmax values
        # (not renormalized to the top-p set; TIPS_AND_TRICKS.md §4.4).
        order = _dp_dispatch_order(templated, self.config.dp_size)
        responses = await engine.async_generate(
            [templated[j] for j in order],
            sampling_params=sampling_params,
            return_logprob=batch.return_logprobs,
            # When a LoRA adapter is loaded, pass its NAME (not disk path) as
            # lora_path — SGLang's registry is keyed by lora_name, and
            # _resolve_lora_path looks up the registry by this value. Passing
            # the disk path would miss the registry (keyed by name) and raise
            # "never been loaded".
            lora_path=self._active_lora_name if self.config.enable_lora else None,
        )
        unpermuted: list = [None] * len(order)
        for pos, j in enumerate(order):
            unpermuted[j] = responses[pos]
        responses = unpermuted

        # Extract per-response data (CPU-only, no awaits)
        completions: list[str] = []
        token_ids_list: list[list[int]] = []
        logprobs_list: list[list[float]] = []
        finish_reasons: list[str] = []

        for resp in responses:
            text = resp.get("text", "")
            completions.append(text)

            # Completion token ids come from SGLang's top-level ``output_ids``
            # field, which is always populated (independent of return_logprob).
            # Per-token logprobs are a separate field (only present when
            # return_logprob=True). Decoupling the two means token_ids survive
            # even if a future caller disables logprobs for speed.
            comp_toks = list(resp.get("output_ids", []))
            meta = resp.get("meta_info", {})
            output_logprobs = meta.get("output_token_logprobs")
            # SGLang returns a pre-aligned per-token list [(logprob, token_id,
            # text), ...] -- lp[0] is always the sampled token's logprob, so
            # (unlike vLLM's old logprobs=1 dict-lookup) there is no per-token
            # miss here. The only corruption risk is a missing/short logprob
            # list for a whole sequence: padding fills missing positions with
            # 0.0 (prob=1.0), which injects (old - 0) and annihilates that
            # sequence's IS ratio. Use a NaN sentinel instead -- the model
            # updater neutralizes NaN to a ratio contribution of 1 (see
            # model_fsdp2._compute_loss IS block). Align to len(comp_toks):
            # truncate extras (would misshape the pad tensor), NaN-pad shortfalls.
            # Engine values pass through VERBATIM: sglang's exact 0.0 is a true
            # logprob (prob ~= 1.0), not a placeholder — reinterpreting it as a
            # NaN sentinel (an earlier convention, targeting vLLM's 0.0-padding)
            # corrupted every consumer of these logprobs that is not the IS
            # ratio (reward variants, segmentation, metrics). NaN means ONLY
            # genuinely-missing data (the short-list pad below), which the
            # updater's isnan guard neutralizes.
            probs: list[float] = [float(lp[0]) for lp in output_logprobs or []]
            if len(probs) != len(comp_toks):
                if len(probs) > len(comp_toks):
                    probs = probs[: len(comp_toks)]
                else:
                    probs = probs + [float("nan")] * (len(comp_toks) - len(probs))
            token_ids_list.append(comp_toks)
            logprobs_list.append(probs)

            # Extract finish reason ("length", "stop", "abort", ...) as a plain string
            finish_reasons.append(_finish_reason_str(resp.get("meta_info", {})))

        # Tokenize prompts (right-padded later by model_update)
        # Reuse templated prompts from generation — no double chat-template call
        prompt_token_ids: list[list[int]] = [
            tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in templated
        ]

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
        self._postprocess_result(batch, completions, result)

        # Log metrics asynchronously (fire-and-forget, doesn't block)
        elapsed = time.monotonic() - t0
        total_tokens = sum(len(t) for t in token_ids_list)
        # W&B supergroup: ``train/`` for training, ``val/`` for validation.
        # ``batch.log_prefix`` is "" for train and "val." for eval — map to the
        # slash-grouping convention. The tracker preserves these prefixes.
        _grp = "val" if batch.log_prefix.startswith("val") else "train"
        # Completion length + truncation stats (mirror TRL
        # completions/{mean,min,max}_length, clipped_ratio, mean_terminated_length).
        # ``finish_reason == "length"`` means the completion hit max_gen_len
        # (truncated) — analogous to TRL's ``ids[-1] not in eos_and_pad``.
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
        # Share of samples that exactly repeat an earlier sample of the same
        # prompt — the guard for DP workers sampling in lockstep (see
        # _dp_dispatch_order). Should be ~0 for long completions at T>0.
        _seen: set = set()
        _dups = 0
        for p, t in zip(templated, token_ids_list, strict=True):
            key = (p, tuple(t))
            _dups += key in _seen
            _seen.add(key)
        rollout_metrics[f"{_grp}/completions/duplicate_frac"] = _dups / n_comp if n_comp else 0.0
        if _dups and n_comp:
            logger.warning(
                "Rollout: %d/%d completions repeat an earlier sample of the same prompt",
                _dups, n_comp,
            )
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
            # Allow per-node override for debugging
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
                        templated[i],  # full prompt (system + few-shot + question)
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

    def _extra_sampling_params(self, batch: PromptBatch) -> dict:
        """Per-batch SGLang sampling params (override to extend)."""
        if batch.sampling_constraint:
            raise ValueError(
                "PromptBatch.sampling_constraint needs the agentic rollout node "
                "(whirlwind.rollout.rollout_sglang_agent)"
            )
        return {}

    def _template_prompts(self, batch: PromptBatch) -> list[str]:
        """Render every prompt of a batch to engine text (override to extend)."""
        return [
            self._apply_chat_template(p, batch.system_prompt, batch.prefix_messages)
            for p in batch.prompts
        ]

    def _postprocess_result(
        self, batch: PromptBatch, completions: list[str], result: RolloutResult
    ) -> None:
        """Attach extra per-sample metadata to a result (override; no-op here)."""

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

        ``prefix_messages`` (few_shot_eos): the simple
        style is kept — the examples are folded back into the system
        message as a 'Q: {q}\nA: {a}' blob. The ONLY delta vs the legacy
        prompt: the tokenizer's DECLARED EOS string (the token the engine
        stops on) is appended after each example, so the demonstrated
        post-answer pattern is 'answer → EOS' (was 'answer → \n\nQ:').
        No extra template turns are introduced.
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
            load the adapter natively via ``engine.load_lora_adapter`` and keep
            its name as the active adapter. The base model stays loaded; only
            the adapter (MB) is swapped. No peft import — SGLang reads the
            standard PEFT adapter dir directly.
          - Full weights (default): swap the full model weights via
            ``update_weights_from_disk`` (the existing path).

        Calls the tokenizer manager's async method directly because the engine
        wrapper uses ``self.loop.run_until_complete()`` which fails when SGLang's
        event loop is already running.
        """
        from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

        req = _from_dict(WeightSyncRequest, data)
        assert self._engine is not None
        engine = self._engine
        ckpt_path = req.checkpoint_path
        logger.info("Rollout: updating weights from %s (step=%d)", ckpt_path, req.step)

        # --- LoRA adapter path ---
        if self.config.enable_lora and os.path.exists(
            os.path.join(ckpt_path, "adapter_config.json")
        ):
            from sglang.srt.managers.io_struct import LoadLoRAAdapterReqInput

            # Unload the previous adapter (if any) before loading the new one
            # (one active adapter at a time).
            if self._active_lora_name is not None:
                try:
                    from sglang.srt.managers.io_struct import UnloadLoRAAdapterReqInput

                    unload_obj = UnloadLoRAAdapterReqInput(lora_name=self._active_lora_name)
                    await engine.tokenizer_manager.unload_lora_adapter(unload_obj, None)
                except Exception as exc:  # already evicted / first step
                    logger.debug("unload_lora_adapter(%s) skipped: %s", self._active_lora_name, exc)
            lora_name = f"step{req.step}"
            obj = LoadLoRAAdapterReqInput(lora_name=lora_name, lora_path=ckpt_path)
            result = await engine.tokenizer_manager.load_lora_adapter(obj, None)
            # load_lora_adapter returns a LoRAUpdateOutput object (not a 3-tuple
            # like update_weights_from_disk): .success, .error_message.
            if not result.success and "already loaded" not in str(result.error_message):
                raise RuntimeError(
                    f"Rollout: LoRA adapter load FAILED at step={req.step}: {result.error_message}."
                )
            # Delete the PREVIOUS adapter dir (keep the current one on disk —
            # SGLang may re-read it if the adapter is evicted and re-requested
            # during this step's generation; _resolve_lora_path reloads from
            # lora_ref_cache[path] which points here).
            if self._prev_lora_path and os.path.isdir(self._prev_lora_path):
                import shutil

                shutil.rmtree(self._prev_lora_path, ignore_errors=True)
            self._prev_lora_path = ckpt_path
            self._active_lora_name = lora_name
            self._active_lora_path = ckpt_path
            # Flush the radix (prefix/KV) cache: SGLang flushes it on full-weight
            # updates but NOT on adapter swaps, and prefix KV computed under the
            # previous adapter must never serve prompts under the new one.
            try:
                await engine.tokenizer_manager.flush_cache()
            except Exception as exc:
                logger.warning("Rollout: radix cache flush after adapter swap failed: %s", exc)
            logger.info("Rollout: LoRA adapter loaded (name=%s, step=%d)", lora_name, req.step)
            return {"ok": True, "step": req.step}

        # --- Full weight path (existing) ---
        obj = UpdateWeightFromDiskReqInput(
            model_path=req.checkpoint_path,
            load_format=None,
        )
        success, message, _ = await engine.tokenizer_manager.update_weights_from_disk(obj, None)
        if not success:
            raise RuntimeError(
                f"Rollout: weight update FAILED at step={req.step}: {message}. "
                f"The rollout engine will keep using stale weights — aborting to "
                f"prevent silent off-policy training."
            )
        logger.info("Rollout: weights updated successfully (step=%d)", req.step)
        # The export dir is deleted by its creator (the orchestrator's _call_sync)
        # once every consumer has loaded it; deleting it here raced the other pool
        # backends reading the same /dev/shm path.

        return {"ok": True, "step": req.step}

    async def _sleep(self, data: dict[str, Any]) -> dict[str, Any]:
        """Release GPU memory (weights + KV cache) via SGLang memory saver.

        The engine must be idle (no ongoing requests) before calling this.
        """
        assert self._engine is not None
        if self._asleep:
            return {"ok": True}
        logger.info("Rollout: releasing GPU memory (sleep)")
        await self._sglang_memory_occupation("release")
        self._asleep = True
        return {"ok": True}

    async def _wake(self, data: dict[str, Any]) -> dict[str, Any]:
        """Restore GPU memory (weights + KV cache) after sleep."""
        assert self._engine is not None
        if not self._asleep:
            return {"ok": True}
        logger.info("Rollout: reclaiming GPU memory (wake)")
        await self._sglang_memory_occupation("resume")
        self._asleep = False
        return {"ok": True}

    async def _sglang_memory_occupation(self, action: str) -> None:
        """Call release/resume memory occupation on SGLang engine.

        SGLang Engine runs its own asyncio loop on a background thread.
        The public ``release_memory_occupation()`` / ``resume_memory_occupation()``
        methods call ``self.loop.run_until_complete()`` which fails when invoked
        from another thread because that loop is already running.

        Workaround: call the tokenizer_manager's async method directly via
        ``asyncio.run_coroutine_threadsafe()``.
        """
        from sglang.srt.managers.io_struct import (
            ReleaseMemoryOccupationReqInput,
            ResumeMemoryOccupationReqInput,
        )

        assert self._engine is not None
        engine = self._engine

        req_input = (
            ReleaseMemoryOccupationReqInput(tags=None)
            if action == "release"
            else ResumeMemoryOccupationReqInput(tags=None)
        )
        # Note: this must be called from SGLang's own event loop
        # (engine.loop.run_until_complete(...) would deadlock)
        coro = (
            engine.tokenizer_manager.release_memory_occupation(req_input, None)
            if action == "release"
            else engine.tokenizer_manager.resume_memory_occupation(req_input, None)
        )
        await coro

    def _ping(self) -> dict[str, Any]:
        return {"ok": True, "engine_initialized": self._engine is not None}


def main(node_cls: type[RolloutNode] = RolloutNode, config_cls: type = RolloutConfig) -> None:
    """Run a rollout node until shutdown. Sibling modules call ``main(MyNode, MyConfig)``."""
    import argparse

    parser = argparse.ArgumentParser(description="Whirlwind Rollout Node")
    parser.add_argument(
        "--config",
        default="configs/rollout/sglang.yaml",
        help="Path to rollout YAML config",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = config_cls.from_yaml(args.config)
    print(f"Starting rollout node: model={config.model_path}, tp={config.tp_size}")

    node = node_cls(config)

    async def _main() -> None:
        await node.start()
        print("Rollout node running. Press Ctrl+C to stop.")
        try:
            while node.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await node.stop()

    # piggyback on engine's event loop
    node.set_up()
    assert node._engine is not None  # noqa: S101
    node._engine.loop.run_until_complete(_main())


if __name__ == "__main__":
    main()
