"""Rollout proxy for agent traffic (EXPERIMENTAL).

A sibling of :mod:`whirlwind.rollout.rollout_proxy` for the agentic mode
(docs/agentic_multiturn.md), configured by
:class:`whirlwind.config_agentic.AgentProxyConfig`. On top of the base proxy:

* ``touch(rollout_ids)`` — heartbeat for long-running episodes. It extends the
  pending-reward TTL of those entries, so the TTL measures *crash silence*
  rather than episode length (a crashed agent's entries still age out one
  TTL after its last heartbeat);
* ``pause_for_update`` / ``resume_after_update`` — for layouts where the
  engine shares its GPU with the updater: park new generates, wait for the
  in-flight ones to finish (fail-closed after ``drain_timeout_s``), sleep the
  engine; then wake it and release the parked generates;
* control-lane RPC workers that never take generate traffic, so control calls
  stay responsive while serving workers hold parked generates;
* a configurable backend-wait bound for slow engine bringup.

Run it like the base proxy::

    python3 -m whirlwind.rollout.rollout_proxy_agent --config configs/rollout/agentic_proxy.yaml
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from whirlwind.config_agentic import AgentProxyConfig
from whirlwind.messages import (
    PauseUpdateRequest,
    PromptBatch,
    RolloutResult,
    TouchRequest,
    _from_dict,
)
from whirlwind.rollout.replay_buffer import ReplayBuffer, ReplayBufferEntry
from whirlwind.rollout.rollout_proxy import RolloutProxyNode, main

logger = logging.getLogger(__name__)


class AgentReplayBuffer(ReplayBuffer):
    """Replay buffer with per-entry heartbeats (``touch``)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._touched: dict[str, float] = {}  # rollout_id -> last heartbeat

    def touch(self, rollout_ids: list[str]) -> tuple[int, list[str]]:
        """Extend the pending-reward TTL of the given rollouts.

        Returns ``(n_touched, missing)``: ids already popped, expired or never
        served land in ``missing`` (tolerant, like ``set_reward``).
        """
        now = self._clock()
        n_touched = 0
        missing: list[str] = []
        for rid in rollout_ids:
            if rid in self._entries:
                self._touched[rid] = now
                n_touched += 1
            else:
                missing.append(rid)
        return n_touched, missing

    def _pending_since(self, entry: ReplayBufferEntry) -> float:
        return max(entry.created_ts, self._touched.get(entry.rollout_id, 0.0))

    def _expire_ttl(self) -> None:
        super()._expire_ttl()
        # Forget heartbeats of entries that left the buffer.
        if len(self._touched) > len(self._entries):
            self._touched = {
                rid: ts for rid, ts in self._touched.items() if rid in self._entries
            }


class AgentRolloutProxyNode(RolloutProxyNode):
    """Rollout proxy with heartbeats and shared-GPU pause/resume."""

    config: AgentProxyConfig

    def __init__(self, config: AgentProxyConfig) -> None:
        super().__init__(config)
        # Pause gate: created on the running loop in _expose_extra_endpoints.
        self._serving: asyncio.Event | None = None
        self._gate_lock: asyncio.Lock | None = None
        self._inflight = 0  # backend generates currently holding the engine

    def _make_buffer(self) -> ReplayBuffer:
        return AgentReplayBuffer(
            max_entries=self.config.buffer_max_entries,
            reward_ttl_s=self.config.reward_ttl_s,
        )

    def _expose_extra_endpoints(self) -> None:
        assert self._node is not None
        self._serving = asyncio.Event()
        self._serving.set()  # serving by default
        self._gate_lock = asyncio.Lock()
        self._node.expose("touch")(self._touch)
        self._node.expose("pause_for_update")(self._pause_for_update)
        self._node.expose("resume_after_update")(self._resume_after_update)

    def _start_extra_workers(self) -> None:
        assert self._node is not None
        for _ in range(max(0, self.config.n_control_workers)):
            self._worker_tasks.append(
                self._node.loop.create_task(
                    self._node.process_rpc_requests(
                        skip_endpoints={"generate", "generate_external"}
                    )
                )
            )
        logger.info(
            "Agent proxy: %d control worker(s), backend wait %.0fs",
            self.config.n_control_workers,
            self.config.backend_wait_timeout_s,
        )

    # ------------------------------------------------------------------
    # Pause gate around every backend generate
    # ------------------------------------------------------------------

    async def _serve_gate(self) -> None:
        """Park while the engine is paused; then count as in flight.

        Registering as in flight only after passing the gate (atomically with
        respect to pause's clear) lets the drain tell parked requests (safe to
        sleep over) from in-flight ones (must finish first).
        """
        assert self._serving is not None and self._gate_lock is not None
        while True:
            await self._serving.wait()
            async with self._gate_lock:
                if self._serving.is_set():
                    self._inflight += 1
                    return

    async def _backend_generate(self, batch: PromptBatch) -> RolloutResult:
        await self._serve_gate()
        try:
            return await super()._backend_generate(batch)
        finally:
            self._inflight -= 1

    async def _pause_for_update(self, data: dict[str, Any]) -> dict[str, Any]:
        """Park new generates, drain in-flight ones, sleep the engine.

        Fail-closed: a drain longer than ``drain_timeout_s`` raises — the
        engine is never slept under live requests.
        """
        assert self._node is not None and self._serving is not None
        assert self._gate_lock is not None
        req = _from_dict(PauseUpdateRequest, data)
        t0 = time.monotonic()
        async with self._gate_lock:
            self._serving.clear()
        while self._inflight > 0:
            if time.monotonic() - t0 > req.drain_timeout_s:
                self._serving.set()  # keep serving; the update step fails instead
                raise TimeoutError(
                    f"pause_for_update: {self._inflight} request(s) still in flight "
                    f"after {req.drain_timeout_s:.0f}s — refusing to sleep the "
                    "engine under live requests"
                )
            await asyncio.sleep(0.05)
        await self._node.call(self.config.backend, "sleep", {})
        drained = time.monotonic() - t0
        logger.info("pause_for_update: drained and engine asleep (%.1fs)", drained)
        return {"ok": True, "drained_s": round(drained, 2)}

    async def _resume_after_update(self, data: dict[str, Any]) -> dict[str, Any]:
        """Wake the engine, then release the parked generates."""
        assert self._node is not None and self._serving is not None
        await self._node.call(self.config.backend, "wake", {})
        self._serving.set()
        logger.info("resume_after_update: engine awake, serving resumed")
        return {"ok": True}

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _touch(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extend the pending-reward TTL of an in-flight episode's rollouts."""
        req = _from_dict(TouchRequest, data)
        ids = [str(r) for r in req.rollout_ids]
        assert isinstance(self._buffer, AgentReplayBuffer)
        n_touched, missing = self._buffer.touch(ids)
        if missing:
            logger.warning(
                "touch: %d/%d unknown rollout_ids (popped/expired/never served): %s%s",
                len(missing),
                len(ids),
                missing[:8],
                "…" if len(missing) > 8 else "",
            )
        return {"ok": True, "n_touched": n_touched, "missing": missing}


if __name__ == "__main__":
    main(AgentRolloutProxyNode, AgentProxyConfig)
