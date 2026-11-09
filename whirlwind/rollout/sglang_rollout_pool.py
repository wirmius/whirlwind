"""SGLang rollout pool — a thin router over N simultaneous SGLang engines.

Registers as ONE rollout endpoint (the orchestrator is unchanged) and fans out
to N backend ``rollout_sglang`` processes, each on its own GPU with
``dp_size=1``. This unblocks multi-GPU LoRA (SGLang's ``load_lora_adapter``
asserts ``dp_size==1`` per-engine) and gives sharded generation throughput
(the DP=2 win, generalized to any GPU count and engine backend).

Exposes the same four handlers the orchestrator already calls on "rollout":
    generate        — shard the batch → fan out in parallel → gather in order
    update_weights  — broadcast to all backends (one shared checkpoint path)
    sleep / wake    — broadcast to all backends (release / reclaim GPU)
    ping            — local health check

The backends are the existing :class:`rollout_sglang.RolloutNode` (launched
unchanged with distinct ``node_id``s). The pool adds one RPC hop (in-process
fan-out via ``asyncio.gather``) — negligible vs generation time.

Async model (verified against whirlcoms/node.py):
  - Outbound: each ``node.call`` gets a unique msg_id + its own asyncio.Future
    in ``_pending``; fires ``send_multipart`` (ZMQ DEALER = non-blocking,
    multiple in-flight); awaits its future. The ``_recv_loop`` routes replies
    by msg_id → resolves the right future. So ``asyncio.gather`` of N calls =
    N truly-concurrent in-flight RPCs.
  - Inbound: the orchestrator's RL loop is serial (generate → reward → train →
    sync), so the pool has one inbound call at a time. Heartbeats use a
    SEPARATE channel (``_orchestrator_request`` to the ctrl endpoint, not the
    RPC handler queue), so they never stall behind a long generate.
  - Backends: each is a separate process → true parallelism (own event loop /
    GIL). While the pool's ``_generate`` awaits the gather, the pool's single
    ``process_rpc_requests`` worker is blocked — fine, because the orchestrator
    is awaiting the generate reply and sends nothing else meanwhile.

See docs/rollout_pool_design.md for the full design + rationale.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, fields, is_dataclass
from typing import Any

from whirlwind.config import RolloutPoolConfig
from whirlwind.messages import (
    NoInputRequest,
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

# Parallel lists in PromptBatch that must be sharded by the SAME index ranges
# to keep each sub-batch self-consistent.
_BATCH_LIST_FIELDS = (
    "prompts",
    "responses",
    "dataset_ids",
    "entry_ids",
    "rollout_ids",
    "subrollout_ids",
)
# Parallel lists in RolloutResult that are gathered (concatenated) in order.
# Includes the ID lists (entry_ids, dataset_ids, etc.) which the orchestrator
# indexes by completions length for reward matching — they MUST be gathered
# alongside completions, or the orchestrator hits IndexError.
_RESULT_LIST_FIELDS = (
    "completions",
    "token_ids",
    "prompt_token_ids",
    "logprobs",
    "prompt_logprobs",
    "finish_reasons",
    "tool_calls",
    "dataset_ids",
    "entry_ids",
    "rollout_ids",
    "subrollout_ids",
)


def _has_default(cls: type, field_name: str) -> bool:
    """True if a dataclass field has a default (so omitting it in kwargs is safe)."""
    if not is_dataclass(cls):
        return False
    for f in fields(cls):
        if f.name == field_name:
            return f.default is not None or f.default_factory is not None  # type: ignore[misc]
    return False


def _shard_batch(batch: PromptBatch, n: int) -> list[PromptBatch]:
    """Split a PromptBatch into N contiguous, order-preserving shards.

    All parallel lists are split by the same index ranges so each shard is
    self-consistent. Scalars (system_prompt, step, temperature, ...) are
    copied verbatim. Shards are contiguous chunks, so concatenating them in
    backend order reproduces the original batch (round-robin did NOT: it
    returned 0,2,4,1,3,5 and broke the orchestrator's prompt/completion
    pairing and GRPO group contiguity).
    """
    total = len(batch.prompts)
    if total == 0 or n <= 1:
        return [batch]
    bounds = [(i * total) // n for i in range(n + 1)]
    ranges = [list(range(bounds[i], bounds[i + 1])) for i in range(n)]  # contiguous
    shards: list[PromptBatch] = []
    for r in ranges:
        if not r:
            continue  # fewer prompts than backends → skip empty shard
        data = asdict(batch)
        for fld in _BATCH_LIST_FIELDS:
            val = getattr(batch, fld, None)
            # Only shard lists that match the prompt length (responses may be
            # empty or shorter; scalars like system_prompt are str/None).
            if isinstance(val, list) and len(val) == total:
                data[fld] = [val[i] for i in r]
        # Only pass kwargs that have a default or a non-None value.
        clean = {k: v for k, v in data.items() if v is not None or _has_default(PromptBatch, k)}
        shards.append(PromptBatch(**clean))
    return shards


def _gather(results: list[RolloutResult]) -> RolloutResult:
    """Concatenate N RolloutResults back into one (in backend order)."""
    if len(results) == 1:
        return results[0]
    merged = asdict(results[0])
    for fld in _RESULT_LIST_FIELDS:
        concat: list = []
        for r in results:
            val = getattr(r, fld, None)
            if val is not None:
                concat.extend(val)
        merged[fld] = concat if concat else None
    clean = {k: v for k, v in merged.items() if v is not None or _has_default(RolloutResult, k)}
    return RolloutResult(**clean)


class SGLangRolloutPoolNode:
    """Thin router: registers as ``config.node_id``, fans out to N backends.

    The backends are configured in ``config.backends`` (a list of node_ids).
    Each backend must be a running rollout node (e.g. ``rollout_sglang``)
    registered under its id. The pool discovers them via the node registry.
    """

    def __init__(self, config: RolloutPoolConfig) -> None:
        self.config = config
        self._node: Node | None = None
        self._backends: list[str] = list(config.backends)

    @property
    def is_running(self) -> bool:
        return self._node.is_running if self._node is not None else False

    def set_up(self) -> None:
        """Pre-start setup (no Node creation — that happens in start() on the loop).

        Unlike rollout_sglang (which piggybacks on the SGLang engine's event
        loop), the pool owns no engine. The Node must be created from the
        running event loop (Node.start asserts ``self._loop is
        asyncio.get_running_loop()``), so we defer Node creation to start().
        """
        pass  # nothing to do pre-start; Node is created in start()

    async def start(self) -> None:
        """Create the Node on the running loop and register handlers."""
        self._node = Node(
            self.config.node_id,
            auto_process=True,
            event_loop=asyncio.get_running_loop(),
        )
        self._node.expose("generate")(self._generate)
        self._node.expose("update_weights")(self._update_weights)
        self._node.expose("sleep")(self._sleep)
        self._node.expose("wake")(self._wake)
        self._node.expose("ping")(self._ping)
        self._node.expose("get_state")(self._get_state)
        self._node.expose("set_state")(self._set_state)
        await self._node.start()
        logger.info(
            "Rollout pool '%s' started — backends: %s",
            self.config.node_id,
            self._backends,
        )

    async def stop(self) -> None:
        if self._node is not None:
            await self._node.stop()

    async def _await_backends(self, timeout: float = 300.0) -> None:
        """Wait until all backends are discoverable (registered with orchestrator).

        Uses ``Node.topology`` (the same dict the orchestrator's
        ``_wait_for_nodes`` polls) — backends register themselves on start.
        """
        assert self._node is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            topo = self._node.topology
            if all(b in topo for b in self._backends):
                return
            await asyncio.sleep(2.0)
        missing = [b for b in self._backends if b not in self._node.topology]
        raise RuntimeError(f"Rollout pool: backends not discovered after {timeout}s: {missing}")

    # ------------------------------------------------------------------
    # Handlers (the 4 the orchestrator calls + ping/state)
    # ------------------------------------------------------------------

    @PromptBatch.decorate_serdes()
    async def _generate(self, batch: PromptBatch) -> RolloutResult:
        """Shard the batch → fan out to N backends in parallel → gather in order."""
        t0 = time.monotonic()
        assert self._node is not None
        await self._await_backends()
        n = len(self._backends)
        shards = _shard_batch(batch, n)
        # Pad shards if fewer than backends (small batch) — extra backends idle.
        while len(shards) < n:
            shards.append(None)  # type: ignore[arg-type]
        logger.info(
            "Pool generate: %d prompts → %d backends (shards: %s)",
            len(batch.prompts), n, [len(s.prompts) if s else 0 for s in shards],
        )
        # Fan out: each non-None shard → its backend's 'generate'. asyncio.gather
        # runs them concurrently (each call has its own msg_id + future).
        async def _call_one(backend: str, shard: PromptBatch | None) -> RolloutResult | None:
            if shard is None:
                return None
            data = await self._node.call(backend, "generate", _to_dict(shard))
            return _from_dict(RolloutResult, data)

        raw = await asyncio.gather(*(
            _call_one(b, s) for b, s in zip(self._backends, shards)
        ))
        results = [r for r in raw if r is not None]
        merged = _gather(results)
        logger.info("Pool generate done in %.1fs (%d completions)", time.monotonic() - t0, len(merged.completions))
        return merged

    async def _update_weights(self, data: dict[str, Any]) -> dict[str, Any]:
        """Broadcast weight update to all backends (all load the same checkpoint)."""
        assert self._node is not None
        await self._await_backends()
        logger.info("Pool update_weights: broadcasting to %d backends", len(self._backends))
        # All backends load from the same shared /dev/shm path — parallel.
        raw = await asyncio.gather(*(
            self._node.call(b, "update_weights", data) for b in self._backends
        ))
        ok = all(isinstance(r, dict) and r.get("ok") for r in raw)
        step = data.get("step", -1)
        return {"ok": ok, "step": step}

    async def _sleep(self, data: dict[str, Any]) -> dict[str, Any]:
        """Broadcast sleep to all backends (release GPU for training phase)."""
        return await self._broadcast("sleep", data)

    async def _wake(self, data: dict[str, Any]) -> dict[str, Any]:
        """Broadcast wake to all backends (reclaim GPU for generation phase)."""
        return await self._broadcast("wake", data)

    async def _broadcast(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Fan out a side-effect-only call (sleep/wake) to all backends."""
        assert self._node is not None
        await self._await_backends()
        logger.info("Pool %s: broadcasting to %d backends", endpoint, len(self._backends))
        raw = await asyncio.gather(*(
            self._node.call(b, endpoint, data) for b in self._backends
        ))
        ok = all(isinstance(r, dict) and r.get("ok") for r in raw)
        return {"ok": ok}

    def _ping(self) -> dict[str, Any]:
        return {"ok": True, "pool": True, "backends": self._backends}

    @StateGetRequest.decorate_serdes()
    def _get_state(self, req: StateGetRequest) -> StateResult:
        return StateResult(data={"node_id": self.config.node_id, "backends": self._backends}, ok=True)

    @StateSetRequest.decorate_serdes()
    def _set_state(self, req: StateSetRequest) -> StateResult:
        return StateResult(data={}, ok=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whirlwind SGLang Rollout Pool")
    parser.add_argument(
        "--config",
        default="configs/rollout/pool_2x_sglang.yaml",
        help="Path to rollout pool YAML config",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = RolloutPoolConfig.from_yaml(args.config)
    print(f"Starting SGLang rollout pool: node_id={config.node_id}, backends={config.backends}")

    pool = SGLangRolloutPoolNode(config)

    async def _main() -> None:
        await pool.start()  # creates Node on the running loop + registers
        print("Rollout pool running. Press Ctrl+C to stop.")
        try:
            while pool.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await pool.stop()

    asyncio.run(_main())
