"""Shared helpers for the async-proxy tests: event-loop decorator + a fake
backend rollout engine (no GPU, no sglang — a canned RolloutResult generator
that honours the rollout RPC contract).
"""

from __future__ import annotations

import asyncio
import contextlib

from whirlwind.messages import PromptBatch, RolloutResult
from whirlwind.whirlcoms import Node


def _async_test(coro):
    """Decorator to run async test methods (mirrors tests/test_whirlwind.py)."""

    def wrapper(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro(self))
        except Exception:
            raise
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(asyncio.sleep(0.05))
            loop.close()

    return wrapper


class FakeBackendNode:
    """A rollout backend that returns deterministic canned completions.

    Satisfies the rollout RPC contract the proxy forwards to: ``generate``
    (returns tokens + per-token logprobs), ``update_weights`` (always ok),
    ``ping``. No engine, no GPU.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._node: Node | None = None
        self.n_generates = 0
        self.n_weight_updates = 0

    @property
    def is_running(self) -> bool:
        return self._node.is_running if self._node is not None else False

    async def start(self) -> None:
        self._node = Node(self.node_id, auto_process=True)
        self._node.expose("generate")(self._generate)
        self._node.expose("update_weights")(self._update_weights)
        self._node.expose("ping")(self._ping)
        await self._node.start()

    async def stop(self) -> None:
        if self._node is not None:
            await self._node.stop()

    @PromptBatch.decorate_serdes()
    async def _generate(self, batch: PromptBatch) -> RolloutResult:
        n = batch.batch_size
        self.n_generates += 1
        return RolloutResult(
            completions=[f"comp-{self.n_generates}-{i}" for i in range(n)],
            token_ids=[[10, 20 + i] for i in range(n)],
            prompt_token_ids=[[1, 2] for _ in range(n)],
            logprobs=[[-0.5, -0.6] for _ in range(n)],
            finish_reasons=["stop"] * n,
            dataset_ids=list(batch.dataset_ids),
            entry_ids=list(batch.entry_ids),
            rollout_ids=list(batch.rollout_ids),
            subrollout_ids=list(batch.subrollout_ids),
        )

    async def _update_weights(self, data: dict) -> dict:
        self.n_weight_updates += 1
        return {"ok": True, "step": 1}

    def _ping(self) -> dict:
        return {"ok": True, "fake": True}
