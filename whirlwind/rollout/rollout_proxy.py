"""Rollout proxy — always-on serving node with a replay buffer (async GRPO).

EXPERIMENTAL — part of the async proxy-rollout mode (``docs/async_proxy_rollout.md``): the rollout
layer becomes a
persistent inference *service* for an external agentic system, while every
served rollout is logged for the async updater to train on.

    EXTERNAL AGENTIC SYSTEM ──generate_external──►  ┌──────────────────────┐
              ▲                                     │  ROLLOUT PROXY       │
              └────────────completions──────────────┤  (this node)         │
              ──report_reward(rollout_ids, rs)────► │  replay buffer       │
                                                    │  weight_version: int │
    ASYNC UPDATER ──pop_ready_batch───────────────► └─────────┬────────────┘
              ◄─────────BufferBatchResult───────────┘          │ forward
              ──update_weights (every K steps)──────►          ▼
              ◄────{"ok", "weight_version"}──────────   BACKEND ENGINE(S)
                                                     (sglang | vllm | pool —
                                                      any rollout module,
                                                      unmodified)

The proxy is a thin router (pool-style): it never touches engines directly.
Its backend is one node id (``config.backend``); if that backend is the
SGLang pool, the proxy transparently serves through N engines. The intended
weight-sync primitive is the LoRA adapter hot-swap (~9 MB, no restart, no
sleep/wake — the backend's existing ``update_weights`` path).

Endpoints beyond the rollout contract (generate / update_weights / sleep /
wake / ping / get_state / set_state — all forwarded):

    generate_external  — the external serving API: expands prompts ×group_size
                         (GRPO groups), serves via the backend with engine
                         logprobs captured, returns completions immediately
                         and logs every sample to the buffer tagged with the
                         CURRENT weight_version.
    report_reward      — the reward channel: out-of-order / delayed / batched
                         scalar rewards keyed by rollout_id. A group becomes
                         trainable when all its members are rewarded.
    pop_ready_batch    — the updater's pull: whole ready groups, FIFO,
                         staleness-bounded (design's K).

Version tagging: entries are tagged with the weight_version at REQUEST time;
the version counter is bumped only after a successful backend weight load.
Requests served while an adapter load is in flight are tagged with the OLD
version — conservative (they look one version staler than they are), never
optimistic.

The buffer lives in-process by design (the proxy already holds the served
rollouts; a separate buffer node would add an RPC hop per generate). It is a
standalone module (``replay_buffer.py``) with a clean seam, so extracting it
into its own node later is a move, not a rewrite.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from whirlwind.config import RolloutProxyConfig
from whirlwind.messages import (
    BufferBatchResult,
    BufferEntry,
    BufferPopRequest,
    ExternalGenerateRequest,
    ExternalGenerateResult,
    PromptBatch,
    ReportRewardRequest,
    RolloutResult,
    StateGetRequest,
    StateResult,
    StateSetRequest,
    WeightSyncRequest,
    _from_dict,
    _to_dict,
)
from whirlwind.rollout.replay_buffer import ReplayBuffer, ReplayBufferEntry
from whirlwind.whirlcoms import Node, RpcError

logger = logging.getLogger(__name__)

# How long generate/update forwarding waits for the backend to appear in the
# topology at bringup (matches the pool's _await_backends default).
_BACKEND_WAIT_TIMEOUT_S = 300.0


class RolloutProxyNode:
    """Thin always-on router over one backend rollout node + replay buffer."""

    def __init__(self, config: RolloutProxyConfig) -> None:
        self.config = config
        self._node: Node | None = None
        self._buffer = self._make_buffer()
        self._weight_version: int = 0  # bumped on each successful update_weights
        self._rollout_counter: int = 0  # unique rollout_id source
        self._group_counter: int = 0  # unique group_id source
        self._worker_tasks: list[asyncio.Task] = []

    def _make_buffer(self) -> ReplayBuffer:
        """Build the replay buffer (override to use a subclass)."""
        return ReplayBuffer(
            max_entries=self.config.buffer_max_entries,
            reward_ttl_s=self.config.reward_ttl_s,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._node.is_running if self._node is not None else False

    @property
    def weight_version(self) -> int:
        return self._weight_version

    async def start(self) -> None:
        """Create the Node on the running loop and register handlers."""
        self._node = Node(
            self.config.node_id,
            auto_process=False,  # we spawn N workers ourselves (serving concurrency)
            event_loop=asyncio.get_running_loop(),
        )
        self._node.expose("generate")(self._generate)
        self._node.expose("generate_external")(self._generate_external)
        self._node.expose("report_reward")(self._report_reward)
        self._node.expose("pop_ready_batch")(self._pop_ready_batch)
        self._node.expose("update_weights")(self._update_weights)
        self._node.expose("sleep")(self._sleep)
        self._node.expose("wake")(self._wake)
        self._node.expose("ping")(self._ping)
        self._node.expose("get_state")(self._get_state)
        self._node.expose("set_state")(self._set_state)
        self._expose_extra_endpoints()
        await self._node.start()
        # N concurrent workers: an external generate never queues behind a
        # validation generate (handlers only await backend RPCs; buffer ops
        # are synchronous and therefore atomic on the single event loop).
        for _ in range(max(1, self.config.n_rpc_workers)):
            self._worker_tasks.append(
                self._node.loop.create_task(self._node.process_rpc_requests())
            )
        self._start_extra_workers()
        logger.info(
            "Rollout proxy '%s' started — backend '%s', buffer cap %d, ttl %.0fs",
            self.config.node_id,
            self.config.backend,
            self.config.buffer_max_entries,
            self.config.reward_ttl_s,
        )

    def _expose_extra_endpoints(self) -> None:
        """Register additional RPC endpoints before the node starts (override)."""

    def _start_extra_workers(self) -> None:
        """Spawn additional RPC worker tasks after the serving workers (override)."""

    async def _backend_generate(self, batch: PromptBatch) -> RolloutResult:
        """One generate call to the backend engine (override to gate or account)."""
        assert self._node is not None
        return await self._node.call_typed(
            self.config.backend, "generate", RolloutResult, _to_dict(batch)
        )

    async def stop(self) -> None:
        for t in self._worker_tasks:
            t.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []
        if self._node is not None:
            await self._node.stop()

    async def _await_backend(self, timeout: float | None = None) -> None:
        """Block until the backend node is discoverable (bringup race).

        The default bound is the config's ``backend_wait_timeout_s`` when it
        has one (slow engine bringup), else the module default.
        """
        assert self._node is not None
        if timeout is None:
            timeout = float(
                getattr(self.config, "backend_wait_timeout_s", _BACKEND_WAIT_TIMEOUT_S)
            )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.config.backend in self._node.topology:
                return
            await asyncio.sleep(2.0)
        raise RuntimeError(
            f"Rollout proxy: backend '{self.config.backend}' not discovered "
            f"after {timeout}s"
        )

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _next_rollout_id(self) -> str:
        self._rollout_counter += 1
        return f"{self._weight_version}:{self._rollout_counter}"

    def _next_group_id(self) -> int:
        self._group_counter += 1
        return self._group_counter

    def _log_to_buffer(
        self,
        prompts: list[str],
        result: RolloutResult,
        group_ids: list[int],
        rollout_idxs: list[int],
        client_id: str,
        metadata: dict[str, Any] | None,
        weight_version: int,
    ) -> list[str]:
        """Log one generate call's samples to the buffer; returns rollout_ids."""
        rollout_ids: list[str] = []
        entries: list[ReplayBufferEntry] = []
        now = time.monotonic()
        for i in range(result.batch_size):
            rid = self._next_rollout_id()
            rollout_ids.append(rid)
            entries.append(
                ReplayBufferEntry(
                    rollout_id=rid,
                    group_id=group_ids[i],
                    rollout_idx=rollout_idxs[i],
                    prompt=prompts[i] if i < len(prompts) else "",
                    completion=result.completions[i],
                    prompt_token_ids=result.prompt_token_ids[i],
                    token_ids=result.token_ids[i],
                    logprobs=result.logprobs[i] if result.logprobs else None,
                    finish_reason=(
                        result.finish_reasons[i] if i < len(result.finish_reasons) else ""
                    ),
                    weight_version=weight_version,
                    created_ts=now,
                    client_id=client_id,
                    extra=dict(metadata or {}),
                )
            )
        self._buffer.put_many(entries)
        return rollout_ids

    # ------------------------------------------------------------------
    # State endpoints
    # ------------------------------------------------------------------

    def _full_state(self) -> dict[str, Any]:
        return {
            "config": _to_dict(self.config),
            "state": {
                "weight_version": self._weight_version,
                "buffer": self._buffer.stats(),
            },
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
    # Handlers — rollout contract (forwarded to the backend)
    # ------------------------------------------------------------------

    @PromptBatch.decorate_serdes()
    async def _generate(self, batch: PromptBatch) -> RolloutResult:
        """Forward an orchestrator-shaped generate to the backend.

        Validation rollouts (``log_prefix`` starting with "val") are NEVER
        logged to the buffer. Training-path logging is opt-in via
        ``log_internal_generate`` (off by default — a sync orchestrator
        pointed at the proxy already trains on its own rollouts).
        """
        assert self._node is not None
        await self._await_backend()
        result = await self._backend_generate(batch)
        if (
            self.config.log_internal_generate
            and not batch.log_prefix.startswith("val")
        ):
            # Preserve group semantics: distinct entry_ids in THIS call map to
            # fresh group ids (entry_ids repeat across dataset epochs — a raw
            # entry_id key would merge groups across calls).
            group_map: dict[int, int] = {}
            group_ids: list[int] = []
            for eid in batch.entry_ids:
                if eid not in group_map:
                    group_map[eid] = self._next_group_id()
                group_ids.append(group_map[eid])
            rollout_idxs = list(batch.rollout_ids) or [0] * result.batch_size
            self._log_to_buffer(
                prompts=batch.prompts,
                result=result,
                group_ids=group_ids,
                rollout_idxs=rollout_idxs,
                client_id="internal",
                metadata=dict(batch.extra or {}),
                weight_version=self._weight_version,
            )
        return result

    async def _update_weights(self, data: dict[str, Any]) -> dict[str, Any]:
        """Forward a weight sync to the backend, then bump weight_version.

        Raises on backend failure — never silently keep stale weights while
        reporting success (the caller would tag new rollouts with a version
        the engine doesn't actually run).
        """
        assert self._node is not None
        await self._await_backend()
        req = _from_dict(WeightSyncRequest, data)
        raw = await self._node.call(self.config.backend, "update_weights", data)
        if not (isinstance(raw, dict) and raw.get("ok")):
            raise RpcError(
                f"Rollout proxy: backend weight update FAILED at step={req.step} "
                f"(reply={raw!r}) — weight_version NOT bumped; the engine keeps "
                f"the previous policy. Aborting to prevent version-tag lies."
            )
        self._weight_version += 1
        logger.info(
            "Rollout proxy: weights updated (step=%d) — weight_version=%d",
            req.step,
            self._weight_version,
        )
        return {"ok": True, "step": req.step, "weight_version": self._weight_version}

    async def _sleep(self, data: dict[str, Any]) -> dict[str, Any]:
        assert self._node is not None
        raw = await self._node.call(self.config.backend, "sleep", data)
        return raw if isinstance(raw, dict) else {"ok": False}

    async def _wake(self, data: dict[str, Any]) -> dict[str, Any]:
        assert self._node is not None
        raw = await self._node.call(self.config.backend, "wake", data)
        return raw if isinstance(raw, dict) else {"ok": False}

    def _ping(self) -> dict[str, Any]:
        return {
            "ok": True,
            "proxy": True,
            "backend": self.config.backend,
            "weight_version": self._weight_version,
        }

    # ------------------------------------------------------------------
    # Handlers — the async-serving surface (new)
    # ------------------------------------------------------------------

    @ExternalGenerateRequest.decorate_serdes()
    async def _generate_external(
        self, req: ExternalGenerateRequest
    ) -> ExternalGenerateResult:
        """Serve the external agentic system: generate, log, return text.

        Expands each prompt ``group_size`` times (GRPO group), forwards to the
        backend with ``return_logprobs=True`` (the buffer needs the engine
        per-token logprobs captured at generation time for the IS ratio —
        TIPS_AND_TRICKS.md §4.2: NaN sentinel, never 0.0), logs every sample to the
        buffer under the CURRENT weight_version, and returns completions +
        rollout_ids immediately (rewards are reported later, out of band).

        Note: ``max_new_tokens`` is currently IGNORED — the backend engine's
        configured ``max_gen_len`` governs (a per-request override would need
        a PromptBatch field; add one if an agent ever needs it).
        """
        assert self._node is not None
        t0 = time.monotonic()
        await self._await_backend()
        if req.max_new_tokens is not None:
            logger.debug(
                "generate_external: max_new_tokens=%s ignored (backend max_gen_len governs)",
                req.max_new_tokens,
            )

        group_size = max(1, req.group_size)
        # Expand prompts ×group_size, mirroring the sync loop's _expand_prompts
        # (grouping must match so advantages z-score the right samples). Chat
        # conversations, when given, expand in lockstep: the engine templates
        # each message list, and the flat prompt string is the buffer's
        # display copy.
        use_convs = req.conversations is not None
        expanded: list[str] = []
        expanded_convs: list[list[dict[str, Any]]] | None = [] if use_convs else None
        group_ids: list[int] = []
        rollout_idxs: list[int] = []
        for i, p in enumerate(req.prompts):
            gid = self._next_group_id()
            for j in range(group_size):
                expanded.append(p)
                if expanded_convs is not None:
                    expanded_convs.append(req.conversations[i])  # type: ignore[index]
                group_ids.append(gid)
                rollout_idxs.append(j)

        version_at_request = self._weight_version
        batch = PromptBatch(
            prompts=expanded,
            system_prompt=req.system_prompt,
            step=0,
            log_table=False,
            log_prefix="",  # training-path traffic (never "val.")
            temperature=req.temperature,
            top_p=req.top_p,
            return_logprobs=True,
            conversations=expanded_convs,
            tools=req.tools,
            sampling_constraint=req.sampling_constraint,
        )
        result = await self._backend_generate(batch)
        if result.batch_size != len(expanded):
            raise RuntimeError(
                f"Backend returned {result.batch_size} completions for "
                f"{len(expanded)} prompts — refusing to log a misaligned batch"
            )

        rollout_ids = self._log_to_buffer(
            prompts=expanded,
            result=result,
            group_ids=group_ids,
            rollout_idxs=rollout_idxs,
            client_id=req.client_id,
            metadata=req.metadata,
            weight_version=version_at_request,
        )
        elapsed = time.monotonic() - t0
        if self._node is not None:
            self._node.call_bg_no_return(
                "tracker",
                "log_metrics",
                {
                    "metrics": {
                        "proxy/serve_batch_size": result.batch_size,
                        "proxy/serve_time_s": elapsed,
                        "proxy/buffer_entries": len(self._buffer),
                        "proxy/weight_version": self._weight_version,
                    },
                    "node_id": self.config.node_id,
                },
            )
        logger.info(
            "generate_external: %d prompts ×%d → %d completions logged "
            "(v=%d, %.1fs, buffer=%d)",
            len(req.prompts),
            group_size,
            result.batch_size,
            version_at_request,
            elapsed,
            len(self._buffer),
        )
        return ExternalGenerateResult(
            rollout_ids=rollout_ids,
            completions=result.completions,
            finish_reasons=result.finish_reasons,
            weight_version=version_at_request,
            # Structured tool calls parsed by the backend (its tool-call
            # parser); [] when the backend does not parse them.
            tool_calls=list(result.tool_calls) if result.tool_calls else [],
        )

    async def _report_reward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Reward channel: record scalar rewards for previously served rollouts.

        Tolerant by design: unknown ids (already popped, TTL-expired, or never
        served) are reported in ``missing`` instead of failing the whole call.
        """
        req = _from_dict(ReportRewardRequest, data)
        ids: list[str] = [str(r) for r in req.rollout_ids]
        rewards: list[float] = [float(r) for r in req.rewards]
        if len(ids) != len(rewards):
            raise ValueError(
                f"report_reward: {len(ids)} rollout_ids but {len(rewards)} rewards"
            )
        missing: list[str] = []
        for rid, r in zip(ids, rewards):
            if not self._buffer.set_reward(rid, r):
                missing.append(rid)
        if missing:
            logger.warning(
                "report_reward: %d/%d unknown rollout_ids (popped/expired/never "
                "served): %s%s",
                len(missing),
                len(ids),
                missing[:8],
                "…" if len(missing) > 8 else "",
            )
        return {"ok": True, "n_recorded": len(ids) - len(missing), "missing": missing}

    @BufferPopRequest.decorate_serdes()
    def _pop_ready_batch(self, req: BufferPopRequest) -> BufferBatchResult:
        """The async updater's pull: whole ready groups, staleness-bounded."""
        max_lag = req.max_lag if req.max_lag >= 0 else self.config.max_weight_version_lag
        entries = self._buffer.pop_ready(
            min_batch=max(1, req.min_batch),
            max_batch=req.max_batch,
            max_lag=max_lag,
            current_version=self._weight_version,
        )
        stats = self._buffer.stats()
        return BufferBatchResult(
            entries=[_to_dict(BufferEntry(
                rollout_id=e.rollout_id,
                group_id=e.group_id,
                rollout_idx=e.rollout_idx,
                prompt=e.prompt,
                completion=e.completion,
                prompt_token_ids=e.prompt_token_ids,
                token_ids=e.token_ids,
                logprobs=e.logprobs,
                finish_reason=e.finish_reason,
                reward=e.reward if e.reward is not None else 0.0,
                weight_version=e.weight_version,
                created_ts=e.created_ts,
                client_id=e.client_id,
                extra=e.extra,
            )) for e in entries],
            weight_version=self._weight_version,
            n_pending=stats["n_entries"],
            n_discarded_stale=stats["n_discarded_stale"],
            n_discarded_ttl=stats["n_discarded_ttl"],
            n_discarded_overflow=stats["n_discarded_overflow"],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(node_cls: type[RolloutProxyNode] = RolloutProxyNode, config_cls: type = RolloutProxyConfig) -> None:
    """Run a rollout proxy until shutdown. Sibling modules call ``main(MyProxy, MyConfig)``."""
    import argparse

    parser = argparse.ArgumentParser(description="Whirlwind Rollout Proxy Node")
    parser.add_argument(
        "--config",
        default="configs/rollout/async_proxy_1x.yaml",
        help="Path to rollout proxy YAML config",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = config_cls.from_yaml(args.config)
    print(
        f"Starting rollout proxy: node_id={config.node_id}, backend={config.backend}"
    )

    proxy = node_cls(config)

    async def _main() -> None:
        await proxy.start()
        print("Rollout proxy running. Press Ctrl+C to stop.")
        try:
            while proxy.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await proxy.stop()

    asyncio.run(_main())


if __name__ == "__main__":
    main()
