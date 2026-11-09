"""Tests for the async proxy-rollout stack (design:
docs/async_proxy_rollout.md; experimental).

Covers:
  1. compute_group_advantages  — the shared TRL-matched advantage helper
     (extracted from RewardNode; delegation pinned)
  2. ReplayBuffer              — FIFO, group-atomic readiness, TTL, staleness,
                                 overflow, min/max batch gating
  3. New message roundtrips    — ExternalGenerate*, ReportReward, Buffer*
  4. RolloutProxyNode over IPC — generate_external logs to buffer, reward
                                 channel, pop_ready, weight-version bump,
                                 staleness discard, validation isolation
  5. AsyncGRPOOrchestrator     — pure helpers (entries→batch, version lag)
                                 + config plumbing + GAS=1 enforcement
  6. AsyncTrainingConfig       — YAML roundtrip with the async knobs

GPU-free: everything runs on IPC sockets + fake backend engines.
Run:  bash tests/run_tests.sh tests.test_async_proxy
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import re
import time
import unittest

# Namespace the IPC sockets BEFORE any whirlwind import (RUN_ID is captured at
# import time): keeps the test cluster off any live run's default paths.
os.environ.setdefault("WW_RUN_ID", "test-async-proxy")

from tests._async_helpers import (  # noqa: E402
    FakeBackendNode,
    _async_test,
)

from whirlwind.config import (  # noqa: E402
    AsyncTrainingConfig,
    ModelConfig,
    RewardConfig,
    RolloutConfig,
    RolloutProxyConfig,
)
from whirlwind.messages import (  # noqa: E402
    BatchRequest,
    BatchResult,
    BufferBatchResult,
    BufferEntry,
    BufferPopRequest,
    ExternalGenerateRequest,
    ExternalGenerateResult,
    PromptBatch,
    ReportRewardRequest,
    RewardResult,
    RolloutResult,
    StateGetRequest,
    StateResult,
    WeightSyncRequest,
    _from_dict,
    _to_dict,
)
from whirlwind.reward.advantages import compute_group_advantages  # noqa: E402
from whirlwind.reward.reward_function import RewardNode  # noqa: E402
from whirlwind.rollout.replay_buffer import ReplayBuffer, ReplayBufferEntry  # noqa: E402
from whirlwind.rollout.rollout_proxy import RolloutProxyNode  # noqa: E402
from whirlwind.training.training_async import AsyncGRPOOrchestrator  # noqa: E402
from whirlwind.whirlcoms import Node, Orchestrator  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    rollout_id: str,
    group_id: int,
    rollout_idx: int = 0,
    reward: float | None = None,
    weight_version: int = 0,
    created_ts: float | None = None,
) -> ReplayBufferEntry:
    return ReplayBufferEntry(
        rollout_id=rollout_id,
        group_id=group_id,
        rollout_idx=rollout_idx,
        prompt=f"prompt-{group_id}",
        completion=f"completion-{rollout_id}",
        prompt_token_ids=[1, 2, 3],
        token_ids=[4, 5],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        weight_version=weight_version,
        created_ts=time.monotonic() if created_ts is None else created_ts,
    )


def make_group(
    group_id: int,
    size: int,
    weight_version: int = 0,
    created_ts: float | None = None,
) -> list[ReplayBufferEntry]:
    return [
        make_entry(
            rollout_id=f"g{group_id}-{i}",
            group_id=group_id,
            rollout_idx=i,
            weight_version=weight_version,
            created_ts=created_ts,
        )
        for i in range(size)
    ]


# ---------------------------------------------------------------------------
# 1. Shared advantage helper
# ---------------------------------------------------------------------------


class TestComputeGroupAdvantages(unittest.TestCase):
    """TRL-matched semantics: Bessel nanstd, group/batch/none, NaN→0."""

    def test_group_zscore_bessel(self):
        # group of 4: rewards 0,0,1,1 → mean .5, sample std = sqrt(4/3 * .25)
        rewards = [0.0, 0.0, 1.0, 1.0]
        ids = [7, 7, 7, 7]
        adv = compute_group_advantages(rewards, ids, "group")
        std = math.sqrt((0.25 + 0.25 + 0.25 + 0.25) / 3)  # ddof=1
        for r in rewards:
            self.assertAlmostEqual((r - 0.5) / std, adv[rewards.index(r)], places=12)

    def test_nan_reward_gets_zero_advantage(self):
        rewards = [1.0, float("nan"), 0.0, 1.0]
        adv = compute_group_advantages(rewards, [1, 1, 1, 1], "group")
        self.assertEqual(adv[1], 0.0)

    def test_batch_mode_uses_global_stats(self):
        rewards = [0.0, 1.0, 0.0, 1.0]
        ids = [1, 1, 2, 2]  # two groups, but batch mode ignores them
        adv = compute_group_advantages(rewards, ids, "batch")
        mean = 0.5
        std = math.sqrt((0.25 * 4) / 3)  # Bessel ddof=1
        self.assertAlmostEqual((rewards[0] - mean) / std, adv[0], places=12)
        self.assertAlmostEqual(adv[0], adv[2], places=12)

    def test_none_mode_subtracts_mean_only(self):
        rewards = [0.0, 2.0]
        adv = compute_group_advantages(rewards, [1, 1], "none")
        self.assertEqual(adv, [-1.0, 1.0])

    def test_matches_reward_node_delegation(self):
        """RewardNode._compute_grpo_advantages must delegate to the helper."""
        node = RewardNode(RewardConfig(reward_type="accuracy", scale_rewards="group"))
        rewards = [0.0, 0.0, 1.0, 1.0, 0.5]
        ids = [1, 1, 1, 1, 2]
        self.assertEqual(
            node._compute_grpo_advantages(rewards, ids),
            compute_group_advantages(rewards, ids, "group"),
        )


# ---------------------------------------------------------------------------
# 2. ReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBuffer(unittest.TestCase):
    def _buffer(self, **kw) -> ReplayBuffer:
        return ReplayBuffer(max_entries=kw.pop("max_entries", 64), **kw)

    def test_partial_group_never_pops(self):
        buf = self._buffer()
        group = make_group(1, 4)
        buf.put_many(group)
        buf.set_reward("g1-0", 1.0)
        buf.set_reward("g1-1", 1.0)
        buf.set_reward("g1-2", 0.0)
        # g1-3 still missing → nothing is ready
        self.assertEqual(buf.pop_ready(min_batch=1), [])
        self.assertEqual(len(buf), 4)  # nothing consumed

    def test_full_group_pops_fifo(self):
        buf = self._buffer()
        buf.put_many(make_group(1, 2))
        buf.put_many(make_group(2, 2))
        for rid in ("g1-0", "g1-1", "g2-0", "g2-1"):
            buf.set_reward(rid, 1.0)
        popped = buf.pop_ready(min_batch=4)
        self.assertEqual([e.rollout_id for e in popped], ["g1-0", "g1-1", "g2-0", "g2-1"])
        self.assertEqual(len(buf), 0)

    def test_min_batch_gate(self):
        buf = self._buffer()
        buf.put_many(make_group(1, 4))
        for e in buf._entries.values():
            buf.set_reward(e.rollout_id, 1.0)
        # min_batch higher than available → nothing
        self.assertEqual(buf.pop_ready(min_batch=8), [])
        # min_batch met → the group
        popped = buf.pop_ready(min_batch=4)
        self.assertEqual(len(popped), 4)

    def test_max_batch_whole_groups_only(self):
        buf = self._buffer()
        buf.put_many(make_group(1, 4))
        buf.put_many(make_group(2, 4))
        buf.put_many(make_group(3, 2))
        for rid in list(buf._entries):
            buf.set_reward(rid, 1.0)
        # cap 6: greedy FIFO selects group1(4); group2(4) would overflow →
        # stop there (FIFO fairness: older groups train first). Group2+3 stay.
        popped = buf.pop_ready(min_batch=1, max_batch=6)
        ids = {e.group_id for e in popped}
        self.assertEqual(ids, {1})
        self.assertEqual(len(buf), 6)  # group2 (4) + group3 (2), both intact

    def test_under_cap_min_batch_not_consumed(self):
        """If max_batch blocks reaching min_batch, NOTHING is consumed."""
        buf = self._buffer()
        buf.put_many(make_group(1, 4))  # ready
        buf.put_many(make_group(2, 8))  # ready, but too big for the cap
        for rid in list(buf._entries):
            buf.set_reward(rid, 1.0)
        # cap 8: group1(4)+group2(8) > 8; group1 alone < min_batch 6 → empty,
        # and group1 must still be in the buffer afterwards.
        self.assertEqual(buf.pop_ready(min_batch=6, max_batch=8), [])
        self.assertEqual(len(buf), 12)

    def test_out_of_order_rewards(self):
        buf = self._buffer()
        buf.put_many(make_group(1, 3))
        buf.set_reward("g1-2", 1.0)
        buf.set_reward("g1-0", 0.0)
        buf.set_reward("g1-1", 1.0)
        popped = buf.pop_ready(min_batch=3)
        self.assertEqual(len(popped), 3)

    def test_ttl_expiry_discards_group(self):
        t = {"now": 100.0}
        buf = ReplayBuffer(max_entries=64, reward_ttl_s=10.0, clock=lambda: t["now"])
        buf.put_many(make_group(1, 2, created_ts=100.0))
        t["now"] = 120.0  # past TTL, reward never arrived
        buf.put_many(make_group(2, 2, created_ts=120.0))  # triggers nothing yet
        buf.set_reward("g2-0", 1.0)
        buf.set_reward("g2-1", 1.0)
        popped = buf.pop_ready(min_batch=2)
        self.assertEqual({e.group_id for e in popped}, {2})
        self.assertEqual(buf.n_discarded_ttl, 2)

    def test_staleness_discard_group_atomic(self):
        buf = self._buffer()
        buf.put_many(make_group(1, 2, weight_version=0))
        buf.set_reward("g1-0", 1.0)
        buf.set_reward("g1-1", 0.0)
        # current version 3, max_lag 2 → version-0 entries are stale
        popped = buf.pop_ready(min_batch=1, max_lag=2, current_version=3)
        self.assertEqual(popped, [])
        self.assertEqual(buf.n_discarded_stale, 2)
        self.assertEqual(len(buf), 0)

    def test_overflow_drops_oldest_group(self):
        buf = ReplayBuffer(max_entries=6)
        buf.put_many(make_group(1, 4, created_ts=1.0))
        buf.put_many(make_group(2, 4, created_ts=2.0))  # overflow → group1 dropped
        self.assertEqual(len(buf), 4)
        self.assertEqual(buf.n_discarded_overflow, 4)
        self.assertEqual({e.group_id for e in buf._entries.values()}, {2})

    def test_set_reward_unknown_id(self):
        buf = self._buffer()
        self.assertFalse(buf.set_reward("nope", 1.0))

    def test_stats(self):
        buf = self._buffer()
        buf.put_many(make_group(1, 2))
        buf.put_many(make_group(2, 2))
        buf.set_reward("g1-0", 1.0)
        buf.set_reward("g1-1", 1.0)
        stats = buf.stats()
        self.assertEqual(stats["n_entries"], 4)
        self.assertEqual(stats["n_groups_ready"], 1)
        self.assertEqual(stats["n_groups_partial"], 1)

    def test_put_too_large_raises(self):
        buf = ReplayBuffer(max_entries=2)
        with self.assertRaises(ValueError):
            buf.put_many(make_group(1, 4))

    def test_duplicate_rollout_id_rejected(self):
        buf = self._buffer()
        buf.put_many(make_group(1, 2))
        with self.assertRaises(ValueError):
            buf.put_many([make_entry("g1-0", group_id=9)])


# ---------------------------------------------------------------------------
# 3. Message roundtrips
# ---------------------------------------------------------------------------


class TestAsyncProxyMessages(unittest.TestCase):
    def test_external_generate_roundtrip(self):
        req = ExternalGenerateRequest(
            prompts=["a", "b"], group_size=4, temperature=0.7, client_id="x",
            metadata={"k": 1},
        )
        req2 = _from_dict(ExternalGenerateRequest, _to_dict(req))
        self.assertEqual(req2.prompts, ["a", "b"])
        self.assertEqual(req2.group_size, 4)
        self.assertEqual(req2.temperature, 0.7)
        self.assertEqual(req2.metadata, {"k": 1})

        res = ExternalGenerateResult(
            rollout_ids=["0:1", "0:2"], completions=["c1", "c2"],
            finish_reasons=["stop", "length"], weight_version=3,
        )
        res2 = _from_dict(ExternalGenerateResult, _to_dict(res))
        self.assertEqual(res2.rollout_ids, ["0:1", "0:2"])
        self.assertEqual(res2.weight_version, 3)

    def test_report_reward_roundtrip(self):
        req = ReportRewardRequest(rollout_ids=["0:1", "0:2"], rewards=[1.0, 0.0])
        d = _to_dict(req)
        req2 = _from_dict(ReportRewardRequest, d)
        self.assertEqual(req2.rollout_ids, ["0:1", "0:2"])
        self.assertEqual(req2.rewards, [1.0, 0.0])

    def test_buffer_messages_roundtrip(self):
        entry = BufferEntry(
            rollout_id="0:1", group_id=5, rollout_idx=2, prompt="p",
            completion="c", prompt_token_ids=[1], token_ids=[2],
            logprobs=[-0.5], finish_reason="stop", reward=1.0,
            weight_version=2, created_ts=123.5, client_id="sim",
        )
        d = _to_dict(entry)
        # to_dict(asdict) flattens nested defaults; from_dict restores
        entry2 = BufferEntry.from_dict(d)
        self.assertEqual(entry2.rollout_id, "0:1")
        self.assertEqual(entry2.group_id, 5)
        self.assertEqual(entry2.logprobs, [-0.5])
        self.assertEqual(entry2.reward, 1.0)

        pop = BufferPopRequest(min_batch=8, max_batch=64, max_lag=2)
        pop2 = _from_dict(BufferPopRequest, _to_dict(pop))
        self.assertEqual((pop2.min_batch, pop2.max_batch, pop2.max_lag), (8, 64, 2))

        res = BufferBatchResult(
            entries=[d], weight_version=3, n_pending=4,
            n_discarded_stale=2, n_discarded_ttl=1, n_discarded_overflow=0,
        )
        res2 = _from_dict(BufferBatchResult, _to_dict(res))
        self.assertEqual(res2.weight_version, 3)
        self.assertEqual(len(res2.entries), 1)
        # entries ride as plain dicts — reconstruct explicitly
        e = BufferEntry.from_dict(res2.entries[0])
        self.assertEqual(e.group_id, 5)


# ---------------------------------------------------------------------------
# 4. RolloutProxyNode over IPC
# ---------------------------------------------------------------------------


class _Cluster:
    """Bringup/teardown of orchestrator + fake backend + proxy + client."""

    def __init__(self, proxy_config: RolloutProxyConfig):
        self.proxy_config = proxy_config
        self.orch: Orchestrator | None = None
        self.backend: FakeBackendNode | None = None
        self.proxy: RolloutProxyNode | None = None
        self.client: Node | None = None

    async def __aenter__(self) -> "_Cluster":
        self.orch = Orchestrator()
        await self.orch.start()
        self.backend = FakeBackendNode("rollout-engine")
        await self.backend.start()
        self.proxy = RolloutProxyNode(self.proxy_config)
        await self.proxy.start()
        self.client = Node("test-client")
        await self.client.start()
        for _ in range(100):
            if (
                "rollout-engine" in self.client.topology
                and "rollout" in self.client.topology
            ):
                break
            await asyncio.sleep(0.05)
        return self

    async def __aexit__(self, *exc) -> None:
        with contextlib.suppress(Exception):
            await self.client.stop()
        with contextlib.suppress(Exception):
            await self.proxy.stop()
        with contextlib.suppress(Exception):
            await self.backend.stop()
        with contextlib.suppress(Exception):
            await self.orch.stop()


class TestRolloutProxyNodeIPC(unittest.TestCase):
    def _config(self, **kw) -> RolloutProxyConfig:
        return RolloutProxyConfig(node_id="rollout", backend="rollout-engine", **kw)

    @_async_test
    async def test_generate_external_serves_and_logs(self):
        async with _Cluster(self._config()) as c:
            req = _to_dict(
                ExternalGenerateRequest(
                    prompts=["q1", "q2"], group_size=2, client_id="t"
                )
            )
            res = _from_dict(
                ExternalGenerateResult,
                await c.client.call("rollout", "generate_external", req),
            )
            self.assertEqual(len(res.completions), 4)
            self.assertEqual(len(res.rollout_ids), 4)
            self.assertEqual(len(set(res.rollout_ids)), 4)
            self.assertEqual(res.weight_version, 0)
            # Logged, pending rewards, nothing ready
            pop = _to_dict(BufferPopRequest(min_batch=4))
            out = _from_dict(
                BufferBatchResult, await c.client.call("rollout", "pop_ready_batch", pop)
            )
            self.assertEqual(out.entries, [])
            self.assertEqual(out.n_pending, 4)

    @_async_test
    async def test_report_reward_and_pop_roundtrip(self):
        async with _Cluster(self._config()) as c:
            req = _to_dict(ExternalGenerateRequest(prompts=["q1"], group_size=4))
            res = _from_dict(
                ExternalGenerateResult,
                await c.client.call("rollout", "generate_external", req),
            )
            reply = await c.client.call(
                "rollout",
                "report_reward",
                _to_dict(ReportRewardRequest(rollout_ids=res.rollout_ids, rewards=[1.0, 0.0, 1.0, 0.0])),
            )
            self.assertTrue(reply["ok"])
            self.assertEqual(reply["n_recorded"], 4)

            pop = _to_dict(BufferPopRequest(min_batch=4))
            out = _from_dict(
                BufferBatchResult, await c.client.call("rollout", "pop_ready_batch", pop)
            )
            self.assertEqual(len(out.entries), 4)
            entries = [BufferEntry.from_dict(e) for e in out.entries]
            self.assertEqual({e.group_id for e in entries}, {1})  # one group
            self.assertEqual([e.rollout_idx for e in entries], [0, 1, 2, 3])
            self.assertEqual([e.reward for e in entries], [1.0, 0.0, 1.0, 0.0])
            self.assertTrue(all(e.logprobs for e in entries))  # engine logprobs captured
            self.assertEqual(out.weight_version, 0)

    @_async_test
    async def test_report_reward_missing_ids(self):
        async with _Cluster(self._config()) as c:
            reply = await c.client.call(
                "rollout",
                "report_reward",
                _to_dict(ReportRewardRequest(rollout_ids=["9:99"], rewards=[1.0])),
            )
            self.assertTrue(reply["ok"])
            self.assertEqual(reply["n_recorded"], 0)
            self.assertEqual(reply["missing"], ["9:99"])

    @_async_test
    async def test_update_weights_bumps_version_and_tags(self):
        async with _Cluster(self._config()) as c:
            # Serve + reward a batch under version 0
            res = _from_dict(
                ExternalGenerateResult,
                await c.client.call(
                    "rollout",
                    "generate_external",
                    _to_dict(ExternalGenerateRequest(prompts=["q1"], group_size=2)),
                ),
            )
            await c.client.call(
                "rollout",
                "report_reward",
                _to_dict(ReportRewardRequest(rollout_ids=res.rollout_ids, rewards=[1.0, 1.0])),
            )
            # Sync weights (fake backend accepts) → version 1
            up = await c.client.call(
                "rollout",
                "update_weights",
                _to_dict(WeightSyncRequest(checkpoint_path="/tmp/fake", step=1)),
            )
            self.assertTrue(up["ok"])
            self.assertEqual(up["weight_version"], 1)
            # New traffic is tagged v1; reward it too
            res2 = _from_dict(
                ExternalGenerateResult,
                await c.client.call(
                    "rollout",
                    "generate_external",
                    _to_dict(ExternalGenerateRequest(prompts=["q2"], group_size=2)),
                ),
            )
            self.assertEqual(res2.weight_version, 1)
            await c.client.call(
                "rollout",
                "report_reward",
                _to_dict(ReportRewardRequest(rollout_ids=res2.rollout_ids, rewards=[1.0, 1.0])),
            )
            # max_lag=0 → the v0 group is stale, only the v1 group trains
            pop = _to_dict(BufferPopRequest(min_batch=2, max_lag=0))
            out = _from_dict(
                BufferBatchResult, await c.client.call("rollout", "pop_ready_batch", pop)
            )
            entries = [BufferEntry.from_dict(e) for e in out.entries]
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(e.weight_version == 1 for e in entries))
            self.assertGreaterEqual(out.n_discarded_stale, 2)

    @_async_test
    async def test_validation_generate_isolated_from_buffer(self):
        async with _Cluster(self._config()) as c:
            batch = _to_dict(
                PromptBatch(
                    prompts=["v1", "v2"],
                    step=1,
                    log_prefix="val.",
                    dataset_ids=["d", "d"],
                    entry_ids=[1, 2],
                    rollout_ids=[0, 1],
                )
            )
            res = _from_dict(
                RolloutResult, await c.client.call("rollout", "generate", batch)
            )
            self.assertEqual(res.batch_size, 2)  # pass-through works
            pop = _to_dict(BufferPopRequest(min_batch=1))
            out = _from_dict(
                BufferBatchResult, await c.client.call("rollout", "pop_ready_batch", pop)
            )
            self.assertEqual(out.n_pending, 0)  # ...and nothing was logged

    @_async_test
    async def test_internal_generate_opt_in_logging(self):
        async with _Cluster(self._config(log_internal_generate=True)) as c:
            # entry_id 10 ×2 → one group; entry_id 11 ×1 → another group
            batch = _to_dict(
                PromptBatch(
                    prompts=["t1", "t1", "t2"],
                    step=1,
                    dataset_ids=["d"] * 3,
                    entry_ids=[10, 10, 11],
                    rollout_ids=[0, 1, 0],
                )
            )
            await c.client.call("rollout", "generate", batch)
            state = _from_dict(
                StateResult,
                await c.client.call(
                    "rollout", "get_state", _to_dict(StateGetRequest(keys=["state"]))
                ),
            )
            buf = state.data["state"]["buffer"]
            self.assertEqual(buf["n_entries"], 3)
            self.assertEqual(buf["n_groups"], 2)
            # And nothing is trainable without rewards
            pop = _to_dict(BufferPopRequest(min_batch=1))
            out = _from_dict(
                BufferBatchResult, await c.client.call("rollout", "pop_ready_batch", pop)
            )
            self.assertEqual(out.entries, [])

    @_async_test
    async def test_ping_and_state(self):
        async with _Cluster(self._config()) as c:
            pong = await c.client.call("rollout", "ping")
            self.assertTrue(pong["proxy"])
            self.assertEqual(pong["backend"], "rollout-engine")
            self.assertEqual(pong["weight_version"], 0)


def _mk_state_req():
    return StateGetRequest(keys=["state"])


# ---------------------------------------------------------------------------
# 5. AsyncGRPOOrchestrator — pure helpers + config plumbing
# ---------------------------------------------------------------------------


def _async_config(**kw) -> AsyncTrainingConfig:
    cfg = AsyncTrainingConfig(**kw)
    cfg.rollout = RolloutConfig(model_path="test")
    cfg.reward = RewardConfig(reward_type="accuracy", scale_rewards="group")
    cfg.model = ModelConfig(model_path="test", gradient_accumulation_steps=1)
    return cfg


class TestAsyncOrchestratorHelpers(unittest.TestCase):
    def test_entries_to_batch_groups_and_tokens(self):
        orch = AsyncGRPOOrchestrator(_async_config())
        entries = [
            BufferEntry(
                rollout_id="0:1", group_id=1, rollout_idx=0, prompt="p", completion="c1",
                prompt_token_ids=[1, 2], token_ids=[3], logprobs=[-0.1],
                finish_reason="stop", reward=1.0, weight_version=0,
            ),
            BufferEntry(
                rollout_id="0:2", group_id=1, rollout_idx=1, prompt="p", completion="c2",
                prompt_token_ids=[1, 2], token_ids=[4], logprobs=[-0.2],
                finish_reason="stop", reward=0.0, weight_version=0,
            ),
            BufferEntry(
                rollout_id="0:3", group_id=2, rollout_idx=0, prompt="q", completion="c3",
                prompt_token_ids=[5], token_ids=[6], logprobs=None,
                finish_reason="stop", reward=1.0, weight_version=1,
            ),
        ]
        rollouts, rewards = orch._entries_to_batch(entries)
        self.assertEqual(rollouts.batch_size, 3)
        self.assertEqual(rollouts.entry_ids, [1, 1, 2])  # GRPO groups preserved
        self.assertEqual(rollouts.rollout_ids, [0, 1, 0])
        self.assertEqual(rewards, [1.0, 0.0, 1.0])
        self.assertEqual(rollouts.logprobs, [[-0.1], [-0.2], None])
        # Advantages group correctly through the shared helper
        adv = compute_group_advantages(rewards, rollouts.entry_ids, "group")
        self.assertEqual(adv[0], -adv[1])  # symmetric pair within group 1
        self.assertEqual(adv[2], 0.0)  # solo group → zero std → eps → 0

    def test_version_lag_metric(self):
        orch = AsyncGRPOOrchestrator(_async_config())
        entries = [
            BufferEntry(
                rollout_id=f"0:{i}", group_id=1, rollout_idx=i, prompt="p",
                completion="c", prompt_token_ids=[1], token_ids=[2], logprobs=[-1.0],
                finish_reason="stop", reward=1.0, weight_version=0 if i < 2 else 1,
            )
            for i in range(4)
        ]
        orch._last_pop = BufferBatchResult(entries=[], weight_version=2)
        mean_lag, max_lag = orch._batch_version_lag(entries)
        self.assertAlmostEqual(mean_lag, 1.5)
        self.assertEqual(max_lag, 2)

    @_async_test
    async def test_gas_must_be_one(self):
        cfg = _async_config()
        cfg.model.gradient_accumulation_steps = 4
        orch = AsyncGRPOOrchestrator(cfg)
        with self.assertRaises(RuntimeError):
            await orch.run()  # must raise before touching the network


# ---------------------------------------------------------------------------
# 6. AsyncTrainingConfig YAML roundtrip
# ---------------------------------------------------------------------------


class TestAsyncTrainingConfig(unittest.TestCase):
    def test_yaml_roundtrip_async_knobs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            RolloutConfig(model_path="m").save_yaml(os.path.join(tmp, "rollout.yaml"))
            RewardConfig(reward_type="accuracy").save_yaml(
                os.path.join(tmp, "reward.yaml")
            )
            ModelConfig(model_path="m").save_yaml(os.path.join(tmp, "model.yaml"))
            cfg_path = os.path.join(tmp, "training.yaml")
            cfg = AsyncTrainingConfig(
                rollout_yaml="rollout.yaml",
                reward_yaml="reward.yaml",
                model_yaml="model.yaml",
                total_steps=65,
                min_train_batch=32,
                max_train_batch=128,
                max_weight_version_lag=2,
                adapter_sync_every_k_steps=1,
            )
            cfg.save_yaml(cfg_path)
            loaded = AsyncTrainingConfig.from_yaml(cfg_path)
            self.assertEqual(loaded.min_train_batch, 32)
            self.assertEqual(loaded.max_train_batch, 128)
            self.assertEqual(loaded.max_weight_version_lag, 2)
            self.assertEqual(loaded.total_steps, 65)
            self.assertIsInstance(loaded.rollout, RolloutConfig)
            self.assertIsInstance(loaded.reward, RewardConfig)
            self.assertIsInstance(loaded.model, ModelConfig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
