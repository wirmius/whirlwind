"""Tests for the experimental agentic (multi-turn) mode.

Covers the agentic sibling modules (docs/agentic_multiturn.md):

  1. AgentReplayBuffer          heartbeats (touch) extend the pending-reward TTL
  2. AgentRolloutProxyNode      touch, backend-wait bound, drain-aware
                                pause/resume, control-lane workers (over IPC)
  3. training_agentic helpers   episode identity, episode advantages, padding
  4. AgenticGRPOOrchestrator    episode staging, stale/timeout drops, batch
                                alignment, pause/resume around the update
  5. rollout_sglang_agent       serve gate, gated handlers, template kwargs,
                                special-token escaping, tool-call parsing,
                                engine kwargs
  6. sim_reflect_agent          episode logic, carriers, one wave over IPC
  7. sim_swe_agent              tool-call parser, chat episode flow
  8. sim_os_agent               task generation, simulated tools, grading,
                                call parsing, rewards, requests
  9. Configs                    YAML round-trips of the agentic configs

GPU-free: IPC sockets, fake backends and stubs; no SGLang engine is started.
Run:  bash tests/run_tests.sh tests.test_agentic
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import math
import os
import pathlib
import posixpath
import re
import tempfile
import time
import unittest
from unittest import mock

# Namespace the IPC sockets BEFORE any whirlwind import (RUN_ID is captured at
# import time): keeps the test cluster off any live run's default paths.
os.environ.setdefault("WW_RUN_ID", "test-agentic")

from tests._async_helpers import FakeBackendNode, _async_test  # noqa: E402
from tests.test_async_proxy import make_group  # noqa: E402

from whirlwind.config import (  # noqa: E402
    AsyncTrainingConfig,
    ModelConfig,
    RewardConfig,
    RolloutConfig,
    RolloutProxyConfig,
)
from whirlwind.config_agentic import (  # noqa: E402
    AgenticTrainingConfig,
    AgentProxyConfig,
    AgentRolloutConfig,
)
from whirlwind.messages import (  # noqa: E402
    BatchResult,
    BufferBatchResult,
    BufferEntry,
    BufferPopRequest,
    ExternalGenerateRequest,
    ExternalGenerateResult,
    PauseUpdateRequest,
    PromptBatch,
    ReportRewardRequest,
    RolloutResult,
    TouchRequest,
    _from_dict,
    _to_dict,
)
from whirlwind.reward.advantages import compute_group_advantages  # noqa: E402
from whirlwind.rollout import rollout_sglang_agent  # noqa: E402
from whirlwind.rollout.replay_buffer import ReplayBuffer  # noqa: E402
from whirlwind.rollout.rollout_proxy import RolloutProxyNode  # noqa: E402
from whirlwind.rollout.rollout_proxy_agent import (  # noqa: E402
    AgentReplayBuffer,
    AgentRolloutProxyNode,
)
from whirlwind.rollout.rollout_sglang import RolloutNode  # noqa: E402
from whirlwind.rollout.rollout_sglang_agent import (  # noqa: E402
    AgentRolloutNode,
    _merge_template_kwargs,
    _ServeGate,
    apply_chat_template_messages,
    escape_conversation,
    escape_special_tokens,
)
from whirlwind.training import sim_os_agent  # noqa: E402
from whirlwind.training.sim_os_agent import (  # noqa: E402
    ANSWER_TOOL,
    CONTAINER_LABEL,
    LAST_TURN_HINT,
    TASK_KINDS,
    TOOLS,
    OsEpisode,
    PodmanSandbox,
    SimOsAgent,
    VirtualFS,
    call_error,
    episode_stats,
    grade,
    make_tasks,
    posix_tz,
    reply_calls,
    tool_call_grammar,
)
from whirlwind.training.sim_reflect_agent import (  # noqa: E402
    CARRIERS,
    DEFAULT_RETRY_TEMPLATE,
    ReflectEpisode,
    SimReflectAgent,
    grade_gsm8k,
)
from whirlwind.training.sim_swe_agent import (  # noqa: E402
    BASH_TOOL_SCHEMA,
    CHAT_SYSTEM_PROMPT,
    STYLE_CHAT,
    SimSweAgent,
    SweEpisode,
    SweInstance,
    parse_tool_call,
)
from whirlwind.training.training_agentic import (  # noqa: E402
    AgenticGRPOOrchestrator,
    episode_advantages,
    episode_identity,
    is_pad,
    pad_to_multiple,
)
from whirlwind.whirlcoms import Node, Orchestrator, RpcError  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

# Chat special-token and tool-structure markers, built by concatenation so the
# literal sequences never appear in source (same convention as the modules).
_IM_END = "<|" + "im_end|>"
_TC_OPEN = "<tool" + "_call>"
_TC_CLOSE = "</tool" + "_call>"
_THINK_CLOSE = "</think" + ">"
_FN_OPEN = "<function" + "="
_FN_CLOSE = "</function" + ">"
_P_OPEN = "<parameter" + "="
_P_CLOSE = "</parameter" + ">"


async def _wait_topology(node: Node, *node_ids: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(n in node.topology for n in node_ids):
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"{node_ids} not in topology after {timeout}s")


# ---------------------------------------------------------------------------
# 1. AgentReplayBuffer.touch
# ---------------------------------------------------------------------------


class TestAgentReplayBuffer(unittest.TestCase):
    def _buffer(self, **kw) -> AgentReplayBuffer:
        return AgentReplayBuffer(max_entries=kw.pop("max_entries", 64), **kw)

    def test_touch_extends_ttl(self):
        """A touched entry outlives its creation TTL: the deadline counts
        from the last heartbeat."""
        t = {"now": 100.0}
        buf = self._buffer(reward_ttl_s=10.0, clock=lambda: t["now"])
        buf.put_many(make_group(1, 2, created_ts=100.0))
        t["now"] = 105.0
        self.assertEqual(buf.touch(["g1-0", "g1-1"]), (2, []))
        t["now"] = 114.0  # past the creation TTL: alive only through the heartbeat
        self.assertEqual(buf.pop_ready(min_batch=1), [])  # runs the TTL sweep
        self.assertEqual((len(buf), buf.n_discarded_ttl), (2, 0))
        self.assertEqual(buf.touch(["g1-0", "g1-1"]), (2, []))
        t["now"] = 123.0
        self.assertEqual(buf.pop_ready(min_batch=1), [])
        self.assertEqual(len(buf), 2)
        buf.set_reward("g1-0", 1.0)
        buf.set_reward("g1-1", 1.0)
        self.assertEqual(len(buf.pop_ready(min_batch=2)), 2)
        self.assertEqual(buf.n_discarded_ttl, 0)

    def test_touch_crash_silence_still_expires(self):
        """No heartbeat for a full TTL window -> the group is discarded."""
        t = {"now": 100.0}
        buf = self._buffer(reward_ttl_s=10.0, clock=lambda: t["now"])
        buf.put_many(make_group(1, 2, created_ts=100.0))
        t["now"] = 105.0
        buf.touch(["g1-0", "g1-1"])  # last heartbeat
        t["now"] = 120.0  # 15s of silence
        buf.put_many(make_group(2, 1, created_ts=120.0))
        buf.set_reward("g2-0", 1.0)
        popped = buf.pop_ready(min_batch=1)
        self.assertEqual({e.group_id for e in popped}, {2})
        self.assertEqual(buf.n_discarded_ttl, 2)

    def test_touch_missing_and_after_pop(self):
        buf = self._buffer()
        buf.put_many(make_group(1, 1))
        self.assertEqual(buf.touch(["g1-0", "nope-1"]), (1, ["nope-1"]))
        buf.set_reward("g1-0", 1.0)
        buf.pop_ready(min_batch=1)
        # Already popped: tolerant, reported as missing.
        self.assertEqual(buf.touch(["g1-0"]), (0, ["g1-0"]))

    def test_heartbeats_of_departed_entries_are_forgotten(self):
        buf = self._buffer()
        buf.put_many(make_group(1, 2))
        buf.touch(["g1-0", "g1-1"])
        buf.set_reward("g1-0", 1.0)
        buf.set_reward("g1-1", 0.0)
        self.assertEqual(len(buf.pop_ready(min_batch=2)), 2)
        buf.pop_ready(min_batch=1)  # the next TTL sweep prunes the heartbeats
        self.assertEqual(buf._touched, {})

    def test_touch_lives_only_on_the_agent_buffer(self):
        self.assertFalse(hasattr(ReplayBuffer, "touch"))
        base = RolloutProxyNode(RolloutProxyConfig())
        agent = AgentRolloutProxyNode(AgentProxyConfig())
        self.assertNotIsInstance(base._buffer, AgentReplayBuffer)
        self.assertIsInstance(agent._buffer, AgentReplayBuffer)
        self.assertEqual(agent._buffer.reward_ttl_s, AgentProxyConfig().reward_ttl_s)


# ---------------------------------------------------------------------------
# 2. AgentRolloutProxyNode over IPC
# ---------------------------------------------------------------------------


class _SleepyBackend(FakeBackendNode):
    """Fake engine with sleep/wake endpoints, an optional generate delay and
    an optional generate failure. Serves calls concurrently (several RPC
    workers), records every batch, and notes a sleep that arrives while a
    generate is still running."""

    N_WORKERS = 4

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.n_sleeps = 0
        self.n_wakes = 0
        self.generate_delay_s = 0.0
        self.fail_generates = False
        self.batches: list[PromptBatch] = []
        self.active_generates = 0
        self.slept_under_live_generate = False
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        self._node = Node(self.node_id, auto_process=False, event_loop=asyncio.get_running_loop())
        self._node.expose("generate")(self._generate)
        self._node.expose("update_weights")(self._update_weights)
        self._node.expose("sleep")(self._sleep)
        self._node.expose("wake")(self._wake)
        self._node.expose("ping")(self._ping)
        await self._node.start()
        self._workers = [
            self._node.loop.create_task(self._node.process_rpc_requests())
            for _ in range(self.N_WORKERS)
        ]

    async def stop(self) -> None:
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        await super().stop()

    def _texts(self, batch: PromptBatch) -> list[str]:
        return [f"comp-{self.n_generates}-{i}" for i in range(batch.batch_size)]

    @PromptBatch.decorate_serdes()
    async def _generate(self, batch: PromptBatch) -> RolloutResult:
        self.n_generates += 1  # counted on entry
        self.batches.append(batch)
        self.active_generates += 1
        try:
            if self.generate_delay_s:
                await asyncio.sleep(self.generate_delay_s)
            if self.fail_generates:
                raise RuntimeError("fake engine failure")
        finally:
            self.active_generates -= 1
        n = batch.batch_size
        return RolloutResult(
            completions=self._texts(batch),
            token_ids=[[10, 20 + i] for i in range(n)],
            prompt_token_ids=[[1, 2] for _ in range(n)],
            logprobs=[[-0.5, -0.6] for _ in range(n)],
            finish_reasons=["stop"] * n,
            dataset_ids=list(batch.dataset_ids),
            entry_ids=list(batch.entry_ids),
            rollout_ids=list(batch.rollout_ids),
            subrollout_ids=list(batch.subrollout_ids),
        )

    async def _sleep(self, data: dict) -> dict:
        self.n_sleeps += 1
        if self.active_generates:
            self.slept_under_live_generate = True
        return {"ok": True}

    async def _wake(self, data: dict) -> dict:
        self.n_wakes += 1
        return {"ok": True}


class _AgentCluster:
    """Orchestrator + fake engine + agent proxy + client, in-process over IPC."""

    def __init__(self, proxy_config: AgentProxyConfig, backend_cls: type = _SleepyBackend):
        self.proxy_config = proxy_config
        self.backend_cls = backend_cls
        self.orch: Orchestrator | None = None
        self.backend: _SleepyBackend | None = None
        self.proxy: AgentRolloutProxyNode | None = None
        self.client: Node | None = None

    async def __aenter__(self) -> "_AgentCluster":
        self.orch = Orchestrator()
        await self.orch.start()
        self.backend = self.backend_cls("rollout-engine")
        await self.backend.start()
        self.proxy = AgentRolloutProxyNode(self.proxy_config)
        await self.proxy.start()
        self.client = Node("test-client")
        await self.client.start()
        await _wait_topology(self.client, "rollout-engine", "rollout")
        await _wait_topology(self.proxy._node, "rollout-engine")
        return self

    async def __aexit__(self, *exc) -> None:
        for part in (self.client, self.proxy, self.backend, self.orch):
            if part is not None:
                with contextlib.suppress(Exception):
                    await part.stop()

    async def generate(self, prompt: str = "q", group_size: int = 1) -> ExternalGenerateResult:
        req = _to_dict(ExternalGenerateRequest(prompts=[prompt], group_size=group_size))
        return _from_dict(
            ExternalGenerateResult,
            await self.client.call("rollout", "generate_external", req),
        )

    def generate_task(self) -> asyncio.Task:
        return asyncio.get_running_loop().create_task(self.generate())

    async def pause(self, drain_timeout_s: float = 5.0):
        req = _to_dict(PauseUpdateRequest(drain_timeout_s=drain_timeout_s))
        return await self.client.call("rollout", "pause_for_update", req)

    async def resume(self):
        req = _to_dict(PauseUpdateRequest(drain_timeout_s=5.0))
        return await self.client.call("rollout", "resume_after_update", req)

    async def pop(self, min_batch: int = 1) -> BufferBatchResult:
        req = _to_dict(BufferPopRequest(min_batch=min_batch))
        return _from_dict(
            BufferBatchResult, await self.client.call("rollout", "pop_ready_batch", req)
        )

    async def report(self, rollout_ids: list[str], rewards: list[float]) -> dict:
        req = _to_dict(ReportRewardRequest(rollout_ids=rollout_ids, rewards=rewards))
        return await self.client.call("rollout", "report_reward", req)


def _proxy_config(**kw) -> AgentProxyConfig:
    return AgentProxyConfig(node_id="rollout", backend="rollout-engine", **kw)


class TestAgentProxyIPC(unittest.TestCase):
    @_async_test
    async def test_touch_roundtrip(self):
        async with _AgentCluster(_proxy_config()) as c:
            res = await c.generate("q1", group_size=2)
            reply = await c.client.call(
                "rollout",
                "touch",
                _to_dict(TouchRequest(rollout_ids=res.rollout_ids + ["9:99"])),
            )
            self.assertTrue(reply["ok"])
            self.assertEqual(reply["n_touched"], 2)
            self.assertEqual(reply["missing"], ["9:99"])
            # Touched entries are still trainable.
            await c.report(res.rollout_ids, [1.0, 0.0])
            self.assertEqual(len((await c.pop(min_batch=2)).entries), 2)

    @_async_test
    async def test_touch_keeps_a_long_episode_alive(self):
        """Over the wire: the touched rollout survives past the TTL, the
        untouched one is discarded."""
        async with _AgentCluster(_proxy_config(reward_ttl_s=1.0)) as c:
            kept = await c.generate("kept")
            lost = await c.generate("lost")
            await asyncio.sleep(0.7)
            await c.client.call(
                "rollout", "touch", _to_dict(TouchRequest(rollout_ids=kept.rollout_ids))
            )
            await asyncio.sleep(0.7)
            out = await c.pop()  # the TTL sweep runs on pop
            self.assertEqual(out.entries, [])
            self.assertEqual(out.n_discarded_ttl, 1)
            self.assertEqual((await c.report(kept.rollout_ids, [1.0]))["n_recorded"], 1)
            self.assertEqual((await c.report(lost.rollout_ids, [1.0]))["missing"], lost.rollout_ids)
            out = await c.pop()
            self.assertEqual(
                [BufferEntry.from_dict(e).rollout_id for e in out.entries], kept.rollout_ids
            )

    @_async_test
    async def test_backend_wait_timeout_knob(self):
        """backend_wait_timeout_s bounds the wait for the engine to register."""
        self.assertEqual(AgentProxyConfig().backend_wait_timeout_s, 300.0)
        self.assertFalse(hasattr(RolloutProxyConfig(), "backend_wait_timeout_s"))
        cfg = _proxy_config(backend_wait_timeout_s=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "proxy.yaml")
            cfg.save_yaml(path)
            self.assertEqual(AgentProxyConfig.from_yaml(path).backend_wait_timeout_s, 0.5)

        orch = Orchestrator()
        await orch.start()
        proxy = AgentRolloutProxyNode(cfg)
        await proxy.start()
        backend: _SleepyBackend | None = None
        try:
            # No backend: fails closed after the configured wait.
            t0 = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, r"after 0\.5s"):
                await proxy._await_backend()
            self.assertLess(time.monotonic() - t0, 5.0)
            # Backend registers: the same call returns promptly.
            backend = _SleepyBackend("rollout-engine")
            await backend.start()
            await _wait_topology(proxy._node, "rollout-engine")
            await asyncio.wait_for(proxy._await_backend(), timeout=5.0)
        finally:
            for part in (backend, proxy, orch):
                if part is not None:
                    with contextlib.suppress(Exception):
                        await part.stop()


class TestPauseResumeForUpdate(unittest.TestCase):
    """Drain-aware pause_for_update / resume_after_update on the agent proxy."""

    @_async_test
    async def test_pause_parks_new_generates_until_resume(self):
        async with _AgentCluster(_proxy_config()) as c:
            t0 = time.monotonic()
            reply = await c.pause()
            self.assertTrue(reply["ok"])
            self.assertLess(time.monotonic() - t0, 1.0)  # nothing to drain
            self.assertEqual(c.backend.n_sleeps, 1)

            gen = c.generate_task()
            await asyncio.sleep(0.3)
            self.assertFalse(gen.done())  # parked at the proxy
            self.assertEqual(c.backend.n_generates, 0)

            self.assertTrue((await c.resume())["ok"])
            self.assertEqual(c.backend.n_wakes, 1)
            await asyncio.wait_for(gen, timeout=5)
            self.assertEqual(c.backend.n_generates, 1)

    @_async_test
    async def test_pause_drains_inflight_before_sleep(self):
        async with _AgentCluster(_proxy_config()) as c:
            c.backend.generate_delay_s = 0.4
            gen = c.generate_task()
            await asyncio.sleep(0.1)  # past the gate: in flight
            self.assertEqual(c.backend.n_generates, 1)

            t0 = time.monotonic()
            await c.pause(drain_timeout_s=10.0)
            self.assertGreaterEqual(time.monotonic() - t0, 0.25)  # waited for it
            self.assertEqual(c.backend.n_sleeps, 1)
            self.assertFalse(c.backend.slept_under_live_generate)
            await asyncio.wait_for(gen, timeout=5)
            await c.resume()
            self.assertEqual(c.backend.n_wakes, 1)

    @_async_test
    async def test_pause_timeout_is_fail_closed(self):
        async with _AgentCluster(_proxy_config()) as c:
            c.backend.generate_delay_s = 2.0
            gen = c.generate_task()
            await asyncio.sleep(0.1)
            with self.assertRaises(Exception) as ctx:
                await c.pause(drain_timeout_s=0.2)
            self.assertIn("refusing to sleep", str(ctx.exception))
            self.assertEqual(c.backend.n_sleeps, 0)  # never slept under a live request
            await asyncio.wait_for(gen, timeout=10)
            # Serving was restored: a new generate is not parked.
            c.backend.generate_delay_s = 0.0
            await asyncio.wait_for(c.generate(), timeout=5)

    @_async_test
    async def test_resume_unblocks_multiple_parked_generates(self):
        async with _AgentCluster(_proxy_config()) as c:
            await c.pause()
            tasks = [c.generate_task() for _ in range(3)]
            await asyncio.sleep(0.2)
            self.assertEqual(c.backend.n_generates, 0)
            await c.resume()
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            self.assertEqual(len(results), 3)
            self.assertEqual(c.backend.n_generates, 3)

    @_async_test
    async def test_failed_generate_releases_its_inflight_slot(self):
        async with _AgentCluster(_proxy_config()) as c:
            c.backend.fail_generates = True
            with self.assertRaises(Exception):
                await asyncio.wait_for(c.generate(), timeout=5)
            self.assertEqual(c.proxy._inflight, 0)
            await asyncio.wait_for(c.pause(drain_timeout_s=1.0), timeout=5)
            self.assertEqual(c.backend.n_sleeps, 1)

    @_async_test
    async def test_control_lane_serves_resume_while_serving_workers_park(self):
        """With every serving worker parked on the gate, resume still runs on
        a control worker."""
        async with _AgentCluster(_proxy_config(n_rpc_workers=1, n_control_workers=1)) as c:
            await c.pause()
            tasks = [c.generate_task() for _ in range(2)]
            await asyncio.sleep(0.2)
            await asyncio.wait_for(c.resume(), timeout=3)
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            self.assertEqual(c.backend.n_generates, 2)

    @_async_test
    async def test_without_control_lane_resume_queues_behind_parked_generate(self):
        """Counter-check for the control lane: with no control worker the
        resume call waits behind the parked generate."""
        async with _AgentCluster(_proxy_config(n_rpc_workers=1, n_control_workers=0)) as c:
            await c.pause()
            gen = c.generate_task()
            await asyncio.sleep(0.2)
            try:
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(c.resume(), timeout=1.0)
            finally:
                gen.cancel()
                with contextlib.suppress(BaseException):
                    await gen


# ---------------------------------------------------------------------------
# 3. training_agentic helpers
# ---------------------------------------------------------------------------

_gid = itertools.count(1000)


def _tag(group, episode, turn=1, group_size=2, problem="p1", task="reflect") -> dict:
    return {
        "task": task,
        "problem": problem,
        "group": group,
        "group_size": group_size,
        "episode": episode,
        "turn": turn,
    }


def _buf_entry(
    rid: str,
    *,
    reward: float = 0.0,
    tag: dict | None = None,
    group_id: int | None = None,
    version: int = 0,
    client: str = "agent",
    n_tokens: int = 2,
) -> BufferEntry:
    """One buffer entry; every agent turn is its own one-sample buffer group."""
    return BufferEntry(
        rollout_id=rid,
        group_id=next(_gid) if group_id is None else group_id,
        rollout_idx=0,
        prompt="p",
        completion="c",
        prompt_token_ids=[1, 2],
        token_ids=list(range(n_tokens)),
        logprobs=[-0.1] * n_tokens,
        finish_reason="stop",
        reward=reward,
        weight_version=version,
        client_id=client,
        extra={"ww_tag": tag} if tag is not None else {},
    )


def _episode(group, episode, n_turns, reward, *, group_size=2, version=0, problem="p1"):
    return [
        _buf_entry(
            f"g{group}e{episode}t{t}",
            reward=reward,
            version=version,
            tag=_tag(group, episode, t, group_size, problem),
        )
        for t in range(1, n_turns + 1)
    ]


class TestEpisodeIdentity(unittest.TestCase):
    def test_tagged_entry(self):
        e = _buf_entry("r1", tag=_tag(7, 1, 3, group_size=4, problem="abc", task="t"), client="c1")
        self.assertEqual(episode_identity(e), (("tag", "c1", "t", "abc", 7), 1, 4))

    def test_untagged_entry_keeps_buffer_group(self):
        e = _buf_entry("r1", group_id=42)
        self.assertEqual(episode_identity(e), (("buffer", 42), "r1", 0))

    def test_tag_without_group_size_falls_back(self):
        for tag in (
            {"group": 1, "episode": 0},
            {"group": 1, "episode": 0, "group_size": 0},
            {"group": 1, "group_size": 2},  # no episode
            {"episode": 0, "group_size": 2},  # no group
        ):
            e = _buf_entry("r1", group_id=5, tag=tag)
            self.assertEqual(episode_identity(e), (("buffer", 5), "r1", 0), tag)

    def test_clients_do_not_share_groups(self):
        a = _buf_entry("a", tag=_tag(1, 0), client="agent-a")
        b = _buf_entry("b", tag=_tag(1, 0), client="agent-b")
        self.assertNotEqual(episode_identity(a)[0], episode_identity(b)[0])


class TestEpisodeAdvantages(unittest.TestCase):
    def test_every_turn_gets_its_episode_advantage(self):
        entries = _episode(1, 0, 1, 1.0) + _episode(1, 1, 3, 0.0)
        adv, stats = episode_advantages(entries, "group")
        a = 1 / math.sqrt(2)  # returns [1, 0], Bessel std: z = +-0.707
        self.assertAlmostEqual(adv[0], a, places=6)
        for x in adv[1:]:
            self.assertAlmostEqual(x, -a, places=6)
        self.assertEqual(
            set(stats),
            {
                "groups",
                "episodes",
                "turns_per_episode",
                "groups_with_signal_frac",
                "mean_return",
                "inconsistent_episode_rewards",
            },
        )
        self.assertEqual(stats["groups"], 1.0)
        self.assertEqual(stats["episodes"], 2.0)
        self.assertEqual(stats["turns_per_episode"], 2.0)
        self.assertEqual(stats["groups_with_signal_frac"], 1.0)
        self.assertAlmostEqual(stats["mean_return"], 0.5)
        self.assertEqual(stats["inconsistent_episode_rewards"], 0.0)

    def test_groups_with_signal_frac(self):
        signal = _episode(1, 0, 2, 1.0) + _episode(1, 1, 1, 0.0)
        flat = _episode(2, 0, 1, 1.0) + _episode(2, 1, 2, 1.0)
        adv, stats = episode_advantages(signal + flat, "group")
        self.assertEqual(stats["groups"], 2.0)
        self.assertEqual(stats["groups_with_signal_frac"], 0.5)
        self.assertEqual(adv[len(signal):], [0.0, 0.0, 0.0])  # no-signal group

    def test_inconsistent_episode_rewards_counted(self):
        bad = _episode(1, 0, 2, 1.0)
        bad[1].reward = 0.0  # the turns disagree; the mean is the return
        entries = bad + _episode(1, 1, 1, 0.0)
        adv, stats = episode_advantages(entries, "group")
        self.assertEqual(stats["inconsistent_episode_rewards"], 1.0)
        self.assertAlmostEqual(stats["mean_return"], 0.25)
        self.assertGreater(adv[0], 0.0)
        self.assertEqual(adv[0], adv[1])

    def test_mixed_tagged_and_untagged(self):
        tagged = _episode(1, 0, 2, 1.0) + _episode(1, 1, 1, 0.0)
        untagged = [
            _buf_entry("u0", reward=1.0, group_id=9),
            _buf_entry("u1", reward=0.0, group_id=9),
        ]
        adv, stats = episode_advantages(tagged + untagged, "group")
        a = 1 / math.sqrt(2)
        self.assertEqual(stats["groups"], 2.0)
        self.assertEqual(stats["episodes"], 4.0)  # 2 episodes + 2 plain samples
        self.assertAlmostEqual(adv[3], a, places=6)
        self.assertAlmostEqual(adv[4], -a, places=6)
        # Plain samples behave exactly like the shared GRPO helper.
        self.assertEqual(adv[3:], compute_group_advantages([1.0, 0.0], [9, 9], "group"))

    def test_scale_none_subtracts_the_mean_return(self):
        entries = _episode(1, 0, 1, 1.0) + _episode(1, 1, 2, 0.0)
        adv, _ = episode_advantages(entries, "none")
        self.assertEqual(adv, [0.5, -0.5, -0.5])


class TestPadding(unittest.TestCase):
    def test_pads_up_to_a_multiple(self):
        entries = [_buf_entry(f"r{i}", n_tokens=n) for i, n in enumerate([5, 1, 3, 2, 4])]
        out = pad_to_multiple(entries, 4)
        self.assertEqual(len(out), 8)
        self.assertEqual(out[:5], entries)
        pads = out[5:]
        self.assertTrue(all(is_pad(p) for p in pads))
        self.assertFalse(any(is_pad(e) for e in entries))
        self.assertTrue(all(p.extra == {"ww_pad": True} for p in pads))
        self.assertEqual(len({e.rollout_id for e in out}), 8)
        # Shortest entries are copied first.
        self.assertEqual([len(p.token_ids) for p in pads], [1, 2, 3])
        self.assertEqual([p.group_id for p in pads], [entries[1].group_id, entries[3].group_id, entries[2].group_id])

    def test_more_pads_than_entries_cycle(self):
        out = pad_to_multiple([_buf_entry("only")], 4)
        self.assertEqual(len(out), 4)
        self.assertEqual(len({e.rollout_id for e in out}), 4)

    def test_no_op_when_aligned_or_empty(self):
        entries = [_buf_entry(f"r{i}") for i in range(4)]
        self.assertEqual(pad_to_multiple(entries, 4), entries)
        self.assertEqual(pad_to_multiple(entries, 2), entries)
        self.assertEqual(pad_to_multiple(entries, 1), entries)
        self.assertEqual(pad_to_multiple(entries[:3], 0), entries[:3])
        self.assertEqual(pad_to_multiple([], 8), [])

    def test_untagged_entry_is_not_a_pad(self):
        self.assertFalse(is_pad(_buf_entry("r")))


# ---------------------------------------------------------------------------
# 4. AgenticGRPOOrchestrator
# ---------------------------------------------------------------------------


def _agentic_config(*, mbs: int = 1, fsdp: int = 1, **kw) -> AgenticTrainingConfig:
    kw.setdefault("workdir", "")
    kw.setdefault("validation_every_n_steps", 0)
    cfg = AgenticTrainingConfig(**kw)
    cfg.rollout = AgentRolloutConfig(model_path="test")
    cfg.reward = RewardConfig(reward_type="accuracy", scale_rewards="group")
    cfg.model = ModelConfig(
        model_path="test", gradient_accumulation_steps=1, micro_batch_size=mbs, fsdp_size=fsdp
    )
    return cfg


class TestAgenticStaging(unittest.TestCase):
    def _orch(self, **kw) -> AgenticGRPOOrchestrator:
        kw.setdefault("min_train_batch", 1)
        kw.setdefault("max_train_batch", 0)
        return AgenticGRPOOrchestrator(_agentic_config(**kw))

    def test_incomplete_group_waits(self):
        orch = self._orch()
        orch._stage(_episode(1, 0, 2, 1.0))
        self.assertEqual(orch._take_complete_groups(0), [])
        self.assertEqual(len(orch._staged), 1)
        orch._stage(_episode(1, 1, 1, 0.0))
        self.assertEqual(len(orch._take_complete_groups(0)), 3)
        self.assertEqual(orch._staged, {})
        self.assertEqual(orch.n_groups_trained, 1)

    def test_complete_groups_wait_for_min_train_batch(self):
        orch = self._orch(min_train_batch=4, max_train_batch=64)
        orch._stage(_episode(1, 0, 2, 1.0) + _episode(1, 1, 1, 0.0))  # 3 turns
        self.assertEqual(orch._take_complete_groups(0), [])
        self.assertEqual(len(orch._staged), 1)  # kept, not consumed
        orch._stage(_episode(2, 0, 1, 1.0) + _episode(2, 1, 1, 0.0))  # 2 turns
        batch = orch._take_complete_groups(0)
        self.assertEqual(len(batch), 5)
        self.assertEqual([e.extra["ww_tag"]["group"] for e in batch], [1, 1, 1, 2, 2])

    def test_fifo_and_max_train_batch(self):
        orch = self._orch(min_train_batch=4, max_train_batch=8)
        for g in (1, 2, 3):
            orch._stage(_episode(g, 0, 2, 1.0) + _episode(g, 1, 2, 0.0))
        first = orch._take_complete_groups(0)
        self.assertEqual(len(first), 8)
        self.assertEqual({e.extra["ww_tag"]["group"] for e in first}, {1, 2})
        second = orch._take_complete_groups(0)
        self.assertEqual({e.extra["ww_tag"]["group"] for e in second}, {3})
        self.assertEqual(orch._take_complete_groups(0), [])

    def test_turns_sorted_within_an_episode(self):
        orch = self._orch()
        ep = _episode(1, 0, 3, 1.0)
        orch._stage([ep[2], ep[0]])
        orch._stage(_episode(1, 1, 1, 0.0) + [ep[1]])
        batch = orch._take_complete_groups(0)
        ep0 = [e.extra["ww_tag"]["turn"] for e in batch if e.extra["ww_tag"]["episode"] == 0]
        self.assertEqual(ep0, [1, 2, 3])

    def test_stale_groups_dropped_and_counted(self):
        orch = self._orch(max_weight_version_lag=2)
        orch._stage(_episode(1, 0, 1, 1.0, version=0) + _episode(1, 1, 1, 0.0, version=3))
        orch._stage(_episode(2, 0, 1, 1.0, version=3) + _episode(2, 1, 1, 0.0, version=3))
        batch = orch._take_complete_groups(current_version=3)  # group 1 lags 3 > 2
        self.assertEqual({e.extra["ww_tag"]["group"] for e in batch}, {2})
        self.assertEqual(orch.n_groups_dropped_stale, 1)
        self.assertEqual(orch._staged, {})

    def test_expire_staged_drops_incomplete_groups(self):
        orch = self._orch(group_complete_timeout_s=60.0)
        orch._stage(_episode(1, 0, 1, 1.0))  # incomplete
        orch._stage(_episode(2, 0, 1, 1.0) + _episode(2, 1, 1, 0.0))  # complete
        for g in orch._staged.values():
            g.first_ts -= 120.0
        orch._expire_staged()
        self.assertEqual(orch.n_groups_dropped_timeout, 1)
        self.assertEqual([k[4] for k in orch._staged], [2])  # complete groups stay

    def test_expire_staged_respects_timeout_and_off_switch(self):
        orch = self._orch(group_complete_timeout_s=60.0)
        orch._stage(_episode(1, 0, 1, 1.0))
        orch._expire_staged()  # fresh
        self.assertEqual(len(orch._staged), 1)
        orch.config.group_complete_timeout_s = 0.0  # disabled
        next(iter(orch._staged.values())).first_ts -= 1e6
        orch._expire_staged()
        self.assertEqual(len(orch._staged), 1)
        self.assertEqual(orch.n_groups_dropped_timeout, 0)

    def test_untagged_entries_complete_on_arrival(self):
        orch = self._orch()
        orch._stage([_buf_entry("u0", group_id=5, reward=1.0), _buf_entry("u1", group_id=5)])
        self.assertEqual([e.rollout_id for e in orch._take_complete_groups(0)], ["u0", "u1"])

    def test_tag_without_group_size_warns_once(self):
        orch = self._orch()
        with self.assertLogs("whirlwind.training.training_agentic", level="WARNING") as logs:
            orch._stage([_buf_entry("a", tag={"group": 1, "episode": 0})])
            orch._stage([_buf_entry("b", tag={"group": 1, "episode": 1})])
        self.assertEqual(len([r for r in logs.records if "group_size" in r.getMessage()]), 1)


class _PopNode:
    """Stands in for the orchestrator's Node: serves canned pop results."""

    def __init__(self, pops: list[list[BufferEntry]], weight_version: int = 0):
        self.pops = list(pops)
        self.weight_version = weight_version
        self.requests: list[BufferPopRequest] = []

    async def call_typed(self, target, endpoint, returns, req):
        self.requests.append(_from_dict(BufferPopRequest, req))
        entries = self.pops.pop(0) if self.pops else []
        return BufferBatchResult(
            entries=[_to_dict(e) for e in entries],
            weight_version=self.weight_version,
            n_pending=0,
        )


class TestAgenticBatchAssembly(unittest.TestCase):
    def test_compute_advantages_gives_pads_zero(self):
        orch = AgenticGRPOOrchestrator(_agentic_config(fsdp=4))
        entries = pad_to_multiple(_episode(1, 0, 1, 1.0) + _episode(1, 1, 2, 0.0), 4)
        rollouts, rewards = orch._entries_to_batch(entries)
        adv = orch._compute_advantages(entries, rollouts, rewards)
        self.assertEqual(len(adv), 4)
        self.assertEqual(adv[3], 0.0)  # the pad
        expected, _ = episode_advantages(entries[:3], "group")
        self.assertEqual(adv[:3], expected)
        self.assertEqual(orch._last_stats["pad_entries"], 1.0)
        self.assertEqual(orch._last_stats["episodes"], 2.0)

    def test_batch_alignment(self):
        cases = [((2, 4), 8), ((1, 1), 1), ((0, 4), 4), ((-1, 4), 4), ((3, 0), 3)]
        for (mbs, fsdp), want in cases:
            orch = AgenticGRPOOrchestrator(_agentic_config(mbs=mbs, fsdp=fsdp))
            self.assertEqual(orch._batch_alignment(), want, (mbs, fsdp))

    def test_group_episodes_false_uses_parent_advantages(self):
        entries = _episode(1, 0, 1, 1.0) + _episode(1, 1, 1, 0.0)
        on = AgenticGRPOOrchestrator(_agentic_config())
        off = AgenticGRPOOrchestrator(_agentic_config(group_episodes=False))
        rollouts, rewards = off._entries_to_batch(entries)
        self.assertEqual(
            off._compute_advantages(entries, rollouts, rewards),
            compute_group_advantages(rewards, rollouts.entry_ids, "group"),
        )
        self.assertEqual(off._compute_advantages(entries, rollouts, rewards), [0.0, 0.0])
        self.assertNotEqual(on._compute_advantages(entries, rollouts, rewards), [0.0, 0.0])
        self.assertEqual(off._last_stats, {})

    def test_buffer_drains_only_with_max_batch(self):
        # One-entry groups (one per agent turn): min_batch=1 with max_batch=0
        # stops after the first ready group; a max_batch drains up to it.
        buf = ReplayBuffer(max_entries=64, reward_ttl_s=0.0)
        for g in range(10):
            buf.put_many(make_group(g, 1))
            buf.set_reward(f"g{g}-0", 1.0)
        self.assertEqual(len(buf.pop_ready(min_batch=1, max_batch=0)), 1)
        self.assertEqual(len(buf.pop_ready(min_batch=1, max_batch=100)), 9)

    @_async_test
    async def test_pop_stages_until_a_group_completes_then_pads(self):
        orch = AgenticGRPOOrchestrator(
            _agentic_config(fsdp=4, min_train_batch=1, pop_poll_interval_s=0.01, step_timeout_s=5.0)
        )
        orch._node = _PopNode([_episode(1, 0, 2, 1.0), _episode(1, 1, 1, 0.0)])
        batch = await orch._pop_ready_entries(step=0)
        self.assertEqual(len(batch), 4)  # 3 turns + 1 pad
        self.assertEqual(sum(is_pad(e) for e in batch), 1)
        self.assertEqual(len(orch._node.requests), 2)
        req = orch._node.requests[0]
        # max_batch must be set: with 0 the buffer stops after one group per poll.
        cfg = orch.config
        self.assertEqual((req.min_batch, req.max_lag), (1, 8))
        self.assertEqual(req.max_batch, 4 * max(1, cfg.min_train_batch, cfg.max_train_batch))

    @_async_test
    async def test_pop_idles_out_and_keeps_the_partial_group(self):
        orch = AgenticGRPOOrchestrator(
            _agentic_config(pop_poll_interval_s=0.01, step_timeout_s=0.05)
        )
        orch._node = _PopNode([_episode(1, 0, 1, 1.0)])
        self.assertEqual(await orch._pop_ready_entries(step=0), [])
        self.assertEqual(len(orch._staged), 1)

    @_async_test
    async def test_pop_falls_back_to_parent_without_grouping(self):
        orch = AgenticGRPOOrchestrator(
            _agentic_config(group_episodes=False, min_train_batch=2, max_train_batch=6, fsdp=4)
        )
        entries = _episode(1, 0, 1, 1.0)  # incomplete as an episode group
        orch._node = _PopNode([entries])
        batch = await orch._pop_ready_entries(step=0)
        self.assertEqual([e.rollout_id for e in batch], [e.rollout_id for e in entries])
        req = orch._node.requests[0]
        self.assertEqual((req.min_batch, req.max_batch), (2, 6))
        self.assertEqual(orch._staged, {})


class _RecordingNode:
    """Stands in for the orchestrator's Node: records RPCs in a shared event log."""

    def __init__(self, events: list[str], fail: tuple[str, ...] = ()):
        self.events = events
        self.fail = set(fail)
        self.payloads: dict[str, list] = {}
        self.topology: dict = {}
        self.shutdown_reasons: list[str] = []

    async def call(self, target, endpoint, payload=None):
        self.events.append(endpoint)
        self.payloads.setdefault(endpoint, []).append(payload)
        if endpoint in self.fail:
            raise RpcError(f"{endpoint} failed")
        if endpoint == "update_weights":
            return {"ok": True, "weight_version": 1}
        return {"ok": True}

    async def broadcast_shutdown(self, reason):
        self.shutdown_reasons.append(reason)


class TestAgenticPauseResume(unittest.TestCase):
    """Engine pause/resume around the update step (shared-GPU layout)."""

    def _orch(self, events, fail=(), **kw) -> AgenticGRPOOrchestrator:
        kw.setdefault("engine_sleep_during_update", True)
        kw.setdefault("pause_drain_timeout_s", 12.0)
        orch = AgenticGRPOOrchestrator(_agentic_config(**kw))
        orch._node = _RecordingNode(events, fail)
        return orch

    def _stub_loop(self, orch, events, train_update_error: Exception | None = None):
        """Replace bringup, pop and the model RPCs; the hooks stay real."""
        batch = _episode(1, 0, 1, 1.0) + _episode(1, 1, 1, 0.0)

        async def bringup():
            return None

        async def pop(step):
            orch._last_pop = BufferBatchResult(entries=[], weight_version=0)
            return list(batch)

        async def accum(rollouts, reward_result, step):
            events.append("train_accum")
            return {"loss": 0.0}

        async def update(step):
            events.append("train_update")
            if train_update_error is not None:
                raise train_update_error
            return {"loss": 0.0, "grad_norm": 0.0, "lr": 0.0}

        async def export(step):
            events.append("export_weights")
            return f"/nonexistent/export-{step}"

        orch._coordinate_bringup_sequence = bringup
        orch._pop_ready_entries = pop
        orch._call_train_accum = accum
        orch._call_train_update = update
        orch._call_export_weights = export

    def test_engine_sleep_defaults_off(self):
        cfg = AgenticTrainingConfig()
        self.assertFalse(cfg.engine_sleep_during_update)
        self.assertEqual(cfg.pause_drain_timeout_s, 300.0)
        self.assertFalse(hasattr(AsyncTrainingConfig(), "engine_sleep_during_update"))

    @_async_test
    async def test_hooks_are_no_ops_without_engine_sleep(self):
        events: list[str] = []
        orch = self._orch(events, engine_sleep_during_update=False)
        await orch._before_update(1)
        await orch._after_update(1)
        self.assertEqual(events, [])

    @_async_test
    async def test_syncing_step_wakes_syncs_then_resumes(self):
        events: list[str] = []
        orch = self._orch(events, total_steps=1, adapter_sync_every_k_steps=1)
        self._stub_loop(orch, events)
        logs = await orch.run()
        self.assertEqual(len(logs), 1)
        self.assertEqual(
            events,
            [
                "pause_for_update",
                "train_accum",
                "train_update",
                "export_weights",
                "wake",
                "update_weights",
                "resume_after_update",
            ],
        )
        pause = _from_dict(PauseUpdateRequest, orch._node.payloads["pause_for_update"][0])
        self.assertEqual(pause.drain_timeout_s, 12.0)
        self.assertFalse(orch._engine_paused)

    @_async_test
    async def test_resume_exactly_once_per_step(self):
        """Step 1 has no adapter sync (K=2) and resumes after the update;
        step 2 wakes, syncs, then resumes. One resume each."""
        events: list[str] = []
        orch = self._orch(events, total_steps=2, adapter_sync_every_k_steps=2)
        self._stub_loop(orch, events)
        await orch.run()
        step1 = ["pause_for_update", "train_accum", "train_update", "resume_after_update"]
        step2 = [
            "pause_for_update",
            "train_accum",
            "train_update",
            "export_weights",
            "wake",
            "update_weights",
            "resume_after_update",
        ]
        self.assertEqual(events, step1 + step2)
        self.assertEqual(events.count("resume_after_update"), 2)

    @_async_test
    async def test_failed_step_still_resumes(self):
        events: list[str] = []
        orch = self._orch(events, total_steps=1, shutdown_on_node_failure=False)
        self._stub_loop(orch, events, train_update_error=RpcError("model died"))
        logs = await orch.run()
        self.assertIn("error", logs[0])
        self.assertEqual(
            events, ["pause_for_update", "train_accum", "train_update", "resume_after_update"]
        )

    @_async_test
    async def test_pause_failure_shuts_down(self):
        events: list[str] = []
        orch = self._orch(events, fail=("pause_for_update",))
        with self.assertRaises(RpcError):
            await orch._before_update(1)
        self.assertTrue(orch._shutting_down)
        self.assertIn("engine pause failed at step 1:", orch.shutdown_reason)
        await orch._after_update(1)
        self.assertEqual(events, ["pause_for_update"])  # never resumed a non-paused engine

    @_async_test
    async def test_wake_failure_aborts_the_adapter_sync(self):
        events: list[str] = []
        orch = self._orch(events, fail=("wake",))
        await orch._before_update(1)
        with self.assertRaises(RpcError):
            await orch._call_proxy_sync("/nonexistent/export", 1)
        await orch._after_update(1)
        self.assertEqual(events, ["pause_for_update", "wake"])  # no sync, no resume
        self.assertIn("engine wake failed at step 1", orch.shutdown_reason)

    @_async_test
    async def test_resume_failure_after_sync_shuts_down(self):
        events: list[str] = []
        orch = self._orch(events, fail=("resume_after_update",))
        await orch._before_update(1)
        with self.assertRaises(RpcError):
            await orch._call_proxy_sync("/nonexistent/export", 1)
        await orch._after_update(1)
        self.assertEqual(events, ["pause_for_update", "wake", "update_weights", "resume_after_update"])
        self.assertIn("engine resume failed", orch.shutdown_reason)

    @_async_test
    async def test_dedicated_layout_syncs_without_wake(self):
        events: list[str] = []
        orch = self._orch(events, engine_sleep_during_update=False)
        await orch._before_update(1)
        await orch._call_proxy_sync("/nonexistent/export", 1)
        await orch._after_update(1)
        self.assertEqual(events, ["update_weights"])


# ---------------------------------------------------------------------------
# 5. rollout_sglang_agent
# ---------------------------------------------------------------------------


class TestServeGate(unittest.TestCase):
    """Reader/writer gate: generations share, engine-state changes exclude."""

    @_async_test
    async def test_readers_run_concurrently(self):
        gate = _ServeGate()
        inside = peak = 0

        async def reader():
            nonlocal inside, peak
            async with gate.shared():
                inside += 1
                peak = max(peak, inside)
                await asyncio.sleep(0.05)
                inside -= 1

        await asyncio.gather(*(reader() for _ in range(4)))
        self.assertEqual(peak, 4)

    @_async_test
    async def test_writer_waits_for_inflight_readers(self):
        gate = _ServeGate()
        events: list[str] = []
        release = asyncio.Event()

        async def reader():
            async with gate.shared():
                events.append("reader-in")
                await release.wait()
                events.append("reader-out")

        async def writer():
            async with gate.exclusive():
                events.append("writer")

        r = asyncio.create_task(reader())
        await asyncio.sleep(0.01)
        w = asyncio.create_task(writer())
        await asyncio.sleep(0.05)
        self.assertEqual(events, ["reader-in"])
        release.set()
        await asyncio.gather(r, w)
        self.assertEqual(events, ["reader-in", "reader-out", "writer"])

    @_async_test
    async def test_waiting_writer_blocks_new_readers(self):
        gate = _ServeGate()
        events: list[str] = []
        release = asyncio.Event()

        async def first_reader():
            async with gate.shared():
                events.append("r1-in")
                await release.wait()
                events.append("r1-out")

        async def writer():
            async with gate.exclusive():
                events.append("w-in")
                await asyncio.sleep(0.02)
                events.append("w-out")

        async def late_reader():
            async with gate.shared():
                events.append("r2-in")

        t1 = asyncio.create_task(first_reader())
        await asyncio.sleep(0.01)
        tw = asyncio.create_task(writer())
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(late_reader())
        await asyncio.sleep(0.05)
        self.assertEqual(events, ["r1-in"])  # r2 is held behind the waiting writer
        release.set()
        await asyncio.gather(t1, tw, t2)
        self.assertEqual(events, ["r1-in", "r1-out", "w-in", "w-out", "r2-in"])

    @_async_test
    async def test_writers_are_mutually_exclusive(self):
        gate = _ServeGate()
        inside = peak = 0

        async def writer():
            nonlocal inside, peak
            async with gate.exclusive():
                inside += 1
                peak = max(peak, inside)
                await asyncio.sleep(0.02)
                inside -= 1

        await asyncio.gather(*(writer() for _ in range(3)))
        self.assertEqual(peak, 1)

    @_async_test
    async def test_cancelled_waiting_writer_does_not_wedge_the_gate(self):
        gate = _ServeGate()
        release = asyncio.Event()

        async def reader():
            async with gate.shared():
                await release.wait()

        async def writer():
            async with gate.exclusive():
                pass

        r = asyncio.create_task(reader())
        await asyncio.sleep(0.01)
        w = asyncio.create_task(writer())
        await asyncio.sleep(0.01)
        w.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await w

        async def new_reader():
            async with gate.shared():
                return True

        # Enters while the first reader still holds the gate.
        self.assertTrue(await asyncio.wait_for(new_reader(), timeout=1.0))
        release.set()
        await r

    @_async_test
    async def test_exception_releases_the_gate(self):
        gate = _ServeGate()
        with self.assertRaises(ValueError):
            async with gate.exclusive():
                raise ValueError("update failed")
        with self.assertRaises(ValueError):
            async with gate.shared():
                raise ValueError("generate failed")

        async def writer():
            async with gate.exclusive():
                return True

        self.assertTrue(await asyncio.wait_for(writer(), timeout=1.0))


class _ExposeRecorder:
    """Stands in for the whirlcoms Node: records exposed handlers."""

    def __init__(self) -> None:
        self.handlers: dict = {}
        self._tasks: list[asyncio.Task] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.started = False

    def expose(self, name):
        def deco(fn):
            self.handlers[name] = fn
            return fn

        return deco

    async def start(self) -> None:
        self.started = True

    async def process_rpc_requests(self) -> None:
        await asyncio.Event().wait()


class _RecordingTokenizer:
    """Duck-typed tokenizer: records apply_chat_template calls."""

    chat_template = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[list, dict]] = []

    def apply_chat_template(self, conversation, *args, **kwargs):
        self.calls.append((conversation, kwargs))
        return "|".join(str(m.get("content", "")) for m in conversation)


class TestAgentRolloutNodeWiring(unittest.TestCase):
    """AgentRolloutNode without an engine: the base set_up is stubbed out."""

    def _wired(self, handlers: dict | None = None, **cfg_kw):
        node = AgentRolloutNode(AgentRolloutConfig(model_path="m", **cfg_kw))
        for name, fn in (handlers or {}).items():
            setattr(node, name, fn)  # captured by the gate at set_up time
        recorder = _ExposeRecorder()
        tok = _RecordingTokenizer()

        def base_set_up(this):
            this._tokenizer = tok
            this._use_chat_template = True
            this._node = recorder

        with mock.patch.object(RolloutNode, "set_up", base_set_up):
            node.set_up()
        return node, recorder, tok

    def test_construction_boots_no_engine(self):
        node = AgentRolloutNode(AgentRolloutConfig(model_path="m"))
        self.assertIsNone(node._engine)
        self.assertIsNone(node._node)
        self.assertIsInstance(node._gate, _ServeGate)

    def test_reserved_template_kwargs_fail_at_construction(self):
        with self.assertRaises(ValueError):
            AgentRolloutNode(AgentRolloutConfig(chat_template_kwargs={"tokenize": True}))

    def test_engine_handlers_are_gated(self):
        node, recorder, _ = self._wired()
        for name in ("generate", "update_weights", "sleep", "wake"):
            self.assertIn(name, recorder.handlers)
        self.assertIsNot(recorder.handlers["generate"], node._generate)

    @_async_test
    async def test_update_waits_for_generates_and_holds_new_ones(self):
        events: list[str] = []

        def slow(name, delay):
            async def handler(data):
                events.append(f"{name}-in")
                await asyncio.sleep(delay)
                events.append(f"{name}-out")
                return {"ok": True, "name": name}

            return handler

        _, recorder, _ = self._wired(
            {"_generate": slow("gen", 0.1), "_update_weights": slow("update", 0.05)}
        )
        gen, update = recorder.handlers["generate"], recorder.handlers["update_weights"]
        g1 = asyncio.create_task(gen({"n": 1}))
        g2 = asyncio.create_task(gen({"n": 2}))
        await asyncio.sleep(0.02)
        u = asyncio.create_task(update({}))
        await asyncio.sleep(0.02)
        g3 = asyncio.create_task(gen({"n": 3}))
        results = await asyncio.gather(g1, g2, u, g3)
        self.assertEqual(results[2], {"ok": True, "name": "update"})  # raw args/result pass through
        self.assertEqual(events[:2], ["gen-in", "gen-in"])  # generates overlap
        i_upd = events.index("update-in")
        self.assertEqual(events[i_upd - 2 : i_upd], ["gen-out", "gen-out"])
        self.assertEqual(events[i_upd:], ["update-in", "update-out", "gen-in", "gen-out"])

    @_async_test
    async def test_sleep_and_wake_are_exclusive(self):
        events: list[str] = []

        def mark(name, delay=0.0):
            async def handler(data):
                events.append(f"{name}-in")
                await asyncio.sleep(delay)
                events.append(f"{name}-out")
                return {"ok": True}

            return handler

        _, recorder, _ = self._wired(
            {"_generate": mark("gen", 0.05), "_sleep": mark("sleep"), "_wake": mark("wake")}
        )
        g = asyncio.create_task(recorder.handlers["generate"]({}))
        await asyncio.sleep(0.01)
        await asyncio.gather(recorder.handlers["sleep"]({}), g)
        await recorder.handlers["wake"]({})
        self.assertEqual(
            events, ["gen-in", "gen-out", "sleep-in", "sleep-out", "wake-in", "wake-out"]
        )

    def test_chat_template_kwargs_injected_explicit_kwargs_win(self):
        node, _, tok = self._wired(chat_template_kwargs={"enable_thinking": False})
        node._tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
        self.assertEqual(tok.calls[-1][1], {"enable_thinking": False, "tokenize": False})
        node._tokenizer.apply_chat_template([{"role": "user", "content": "x"}], enable_thinking=True)
        self.assertEqual(tok.calls[-1][1], {"enable_thinking": True})

    def test_apply_chat_template_override(self):
        node, _, _ = self._wired(apply_chat_template=False)
        self.assertFalse(node._use_chat_template)
        node, _, _ = self._wired()  # None = keep the tokenizer's detection
        self.assertTrue(node._use_chat_template)

    @_async_test
    async def test_start_adds_rpc_workers(self):
        for n_workers, extra in ((4, 3), (1, 0), (0, 0)):
            node = AgentRolloutNode(AgentRolloutConfig(model_path="m", n_rpc_workers=n_workers))
            node._node = _ExposeRecorder()
            node._node.loop = asyncio.get_running_loop()
            await node.start()
            self.assertTrue(node._node.started)
            self.assertEqual(len(node._node._tasks), extra, n_workers)
            for t in node._node._tasks:
                t.cancel()
            await asyncio.gather(*node._node._tasks, return_exceptions=True)

    def test_conversations_are_templated_and_escaped(self):
        node = AgentRolloutNode(AgentRolloutConfig(model_path="m"))
        node._tokenizer = _RecordingTokenizer()
        node._use_chat_template = True
        batch = PromptBatch(
            prompts=["a", "b"],
            conversations=[
                [{"role": "user", "content": "x" + _IM_END}],
                [{"role": "user", "content": "y"}],
            ],
            tools=[BASH_TOOL_SCHEMA],
        )
        out = node._template_prompts(batch)
        self.assertEqual(out, ["x\\" + _IM_END, "y"])
        for _, kwargs in node._tokenizer.calls:
            self.assertEqual(kwargs["tools"], [BASH_TOOL_SCHEMA])
            self.assertTrue(kwargs["add_generation_prompt"])
            self.assertFalse(kwargs["tokenize"])

    def test_conversations_refused_without_templating(self):
        node = AgentRolloutNode(AgentRolloutConfig(model_path="m"))
        node._tokenizer = _RecordingTokenizer()
        node._use_chat_template = False
        batch = PromptBatch(prompts=["a"], conversations=[[{"role": "user", "content": "x"}]])
        with self.assertRaises(RuntimeError):
            node._template_prompts(batch)

    def test_flat_prompts_take_the_base_path(self):
        node = AgentRolloutNode(AgentRolloutConfig(model_path="m"))
        node._apply_chat_template = lambda p, sp, pm: f"[{sp}]{p}"
        self.assertEqual(
            node._template_prompts(PromptBatch(prompts=["a"], system_prompt="s")), ["[s]a"]
        )

    def test_postprocess_attaches_parsed_tool_calls(self):
        calls = []

        def fake_parse(completions, tools, parser_name):
            calls.append((list(completions), tools, parser_name))
            return [[{"name": "bash", "arguments": {}}] for _ in completions]

        conv_batch = PromptBatch(
            prompts=["a"], conversations=[[{"role": "user", "content": "x"}]], tools=[BASH_TOOL_SCHEMA]
        )
        flat_batch = PromptBatch(prompts=["a"])

        def result():
            return RolloutResult(
                completions=["c"], token_ids=[[1]], prompt_token_ids=[[0]], finish_reasons=["stop"]
            )

        with mock.patch.object(rollout_sglang_agent, "_parse_tool_calls", fake_parse):
            on = AgentRolloutNode(AgentRolloutConfig(model_path="m", tool_call_parser="hermes"))
            r = result()
            on._postprocess_result(conv_batch, ["c"], r)
            self.assertEqual(r.tool_calls, [[{"name": "bash", "arguments": {}}]])
            self.assertEqual(calls, [(["c"], [BASH_TOOL_SCHEMA], "hermes")])
            r = result()
            on._postprocess_result(flat_batch, ["c"], r)  # no conversations/tools
            self.assertIsNone(r.tool_calls)
            off = AgentRolloutNode(AgentRolloutConfig(model_path="m"))
            r = result()
            off._postprocess_result(conv_batch, ["c"], r)  # parser off
            self.assertIsNone(r.tool_calls)
        self.assertEqual(len(calls), 1)


class TestChatTemplateSafety(unittest.TestCase):
    def test_escape_special_tokens(self):
        markers = (_IM_END, "<|" + "im_start|>", _TC_OPEN, _TC_CLOSE, "<" + "think>", _THINK_CLOSE, "<" + "assistant>")
        for marker in markers:
            self.assertEqual(escape_special_tokens(f"a {marker} b"), f"a \\{marker} b")
        self.assertEqual(escape_special_tokens("plain <b>text</b>"), "plain <b>text</b>")

    def test_escape_is_idempotent(self):
        once = escape_special_tokens("x" + _IM_END + _TC_OPEN)
        self.assertEqual(escape_special_tokens(once), once)
        # A caller-side backslash normalizes to exactly one.
        self.assertEqual(escape_special_tokens("\\" + _IM_END), "\\" + _IM_END)

    def test_escape_conversation_walks_contents_and_arguments(self):
        conv = [
            {"role": "user", "content": "q" + _IM_END},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "1", "function": {"name": "bash", "arguments": {"command": "echo " + _TC_CLOSE}}}
                ],
            },
            {"role": "tool", "content": None, "n": 3, "ok": True},
        ]
        out = escape_conversation(conv)
        self.assertEqual(out[0]["content"], "q\\" + _IM_END)
        self.assertEqual(
            out[1]["tool_calls"][0]["function"]["arguments"]["command"], "echo \\" + _TC_CLOSE
        )
        self.assertEqual(out[2], {"role": "tool", "content": None, "n": 3, "ok": True})
        self.assertEqual(conv[0]["content"], "q" + _IM_END)  # input untouched

    def test_apply_chat_template_messages(self):
        tok = _RecordingTokenizer()
        msgs = [{"role": "user", "content": "x" + _IM_END}]
        out = apply_chat_template_messages(tok, msgs, None, {"enable_thinking": False})
        self.assertEqual(out, "x\\" + _IM_END)
        self.assertEqual(
            tok.calls[-1][1],
            {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
        )
        apply_chat_template_messages(tok, msgs, [BASH_TOOL_SCHEMA])
        self.assertEqual(tok.calls[-1][1]["tools"], [BASH_TOOL_SCHEMA])
        with self.assertRaises(ValueError):
            apply_chat_template_messages(tok, msgs, None, {"add_generation_prompt": False})


class TestTemplateKwargsKnob(unittest.TestCase):
    """chat_template_kwargs: pass-through and fail-closed reserved keys."""

    def test_merge_passthrough(self):
        self.assertEqual(_merge_template_kwargs({"enable_thinking": False}), {"enable_thinking": False})

    def test_none_and_empty(self):
        self.assertEqual(_merge_template_kwargs(None), {})
        self.assertEqual(_merge_template_kwargs({}), {})

    def test_reserved_keys_rejected(self):
        for key in ("tokenize", "add_generation_prompt", "tools"):
            with self.assertRaises(ValueError):
                _merge_template_kwargs({key: True})

    def test_config_field_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rollout.yaml")
            AgentRolloutConfig(model_path="m", chat_template_kwargs={"enable_thinking": False}).save_yaml(path)
            loaded = AgentRolloutConfig.from_yaml(path)
            self.assertEqual(loaded.chat_template_kwargs, {"enable_thinking": False})
            # The base config rejects the agent-only key.
            with self.assertRaises(ValueError):
                RolloutConfig.from_yaml(path)


class TestToolCallParserKnob(unittest.TestCase):
    """AgentRolloutConfig.tool_call_parser: server-side structured tool calls."""

    def test_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rollout.yaml")
            AgentRolloutConfig(model_path="m", tool_call_parser="qwen3_coder").save_yaml(path)
            self.assertEqual(AgentRolloutConfig.from_yaml(path).tool_call_parser, "qwen3_coder")
            AgentRolloutConfig(model_path="m").save_yaml(path)
            self.assertIsNone(AgentRolloutConfig.from_yaml(path).tool_call_parser)

    def test_rollout_result_tool_calls_shard_join(self):
        r = RolloutResult(
            completions=["a", "b", "c", "d"],
            token_ids=[[1], [2], [3], [4]],
            prompt_token_ids=[[0], [0], [0], [0]],
            finish_reasons=["stop"] * 4,
            tool_calls=[
                [{"name": "bash", "arguments": {"command": "ls"}}],
                [],
                [],
                [{"name": "finish", "arguments": {}}],
            ],
        )
        self.assertEqual(_from_dict(RolloutResult, _to_dict(r)).tool_calls, r.tool_calls)
        shards = r.split_into(2)
        self.assertEqual(shards[0].tool_calls, [[{"name": "bash", "arguments": {"command": "ls"}}], []])
        self.assertEqual(RolloutResult.join(shards).tool_calls, r.tool_calls)
        self.assertEqual(r.slice_by([3, 1]).tool_calls, [[{"name": "finish", "arguments": {}}], []])

    def test_rollout_result_without_tool_calls(self):
        r = _from_dict(RolloutResult, {"completions": ["a"], "token_ids": [[1]], "prompt_token_ids": [[0]]})
        self.assertIsNone(r.tool_calls)

    def test_external_result_tool_calls_roundtrip(self):
        e = ExternalGenerateResult(
            rollout_ids=["r1"], completions=["x"],
            tool_calls=[[{"name": "bash", "arguments": {"command": "ls"}}]],
        )
        self.assertEqual(_from_dict(ExternalGenerateResult, _to_dict(e)).tool_calls, e.tool_calls)
        # A peer that sends no tool_calls -> [] (the agent parses the text).
        e3 = _from_dict(ExternalGenerateResult, {"rollout_ids": ["r1"], "completions": ["x"]})
        self.assertEqual(e3.tool_calls, [])

    def _require_sglang(self):
        try:
            from sglang.srt.function_call.function_call_parser import (  # noqa: F401
                FunctionCallParser,
            )
        except Exception:
            self.skipTest("sglang not installed")

    def test_sglang_parser_on_xml_grammar(self):
        self._require_sglang()
        completion = (
            "Let me explore the repo.\n" + _THINK_CLOSE + "\n\n" + _TC_OPEN + "\n"
            + _FN_OPEN + "bash>\n" + _P_OPEN + "command>\nfind /testbed -name x.py\n"
            + _P_CLOSE + "\n" + _FN_CLOSE + "\n" + _TC_CLOSE
        )
        calls = rollout_sglang_agent._parse_tool_calls(
            [completion, "reasoning-only prose, no call"], [BASH_TOOL_SCHEMA], "qwen3_coder"
        )
        self.assertEqual(calls[0], [{"name": "bash", "arguments": {"command": "find /testbed -name x.py"}}])
        self.assertEqual(calls[1], [])

    def test_sglang_parser_on_hermes_grammar(self):
        self._require_sglang()
        completion = _TC_OPEN + "\n" + json.dumps({"name": "bash", "arguments": {"command": "ls"}}) + "\n" + _TC_CLOSE
        calls = rollout_sglang_agent._parse_tool_calls([completion], [BASH_TOOL_SCHEMA], "hermes")
        self.assertEqual(calls, [[{"name": "bash", "arguments": {"command": "ls"}}]])


class TestEngineKwargs(unittest.TestCase):
    """Config -> SGLang Engine kwargs, without booting an engine."""

    def _kwargs(self, **kw) -> dict:
        return AgentRolloutNode(AgentRolloutConfig(**kw))._engine_kwargs()

    def test_spec_disabled_by_default(self):
        kwargs = self._kwargs()
        self.assertNotIn("speculative_algorithm", kwargs)
        self.assertNotIn("speculative_num_draft_tokens", kwargs)

    def test_ngram_mapped(self):
        kwargs = self._kwargs(speculative_algorithm="NGRAM", speculative_num_draft_tokens=4)
        self.assertEqual(kwargs["speculative_algorithm"], "NGRAM")
        self.assertEqual(kwargs["speculative_num_draft_tokens"], 4)

    def test_algorithm_without_draft_tokens(self):
        kwargs = self._kwargs(speculative_algorithm="NGRAM")
        self.assertEqual(kwargs["speculative_algorithm"], "NGRAM")
        self.assertNotIn("speculative_num_draft_tokens", kwargs)

    def test_draft_tokens_without_algorithm_ignored(self):
        self.assertNotIn("speculative_num_draft_tokens", self._kwargs(speculative_num_draft_tokens=4))

    def test_base_kwargs_flow(self):
        kwargs = self._kwargs(
            model_path="m",
            dp_size=1,
            mem_fraction_static=0.6,
            chunked_prefill_size=2048,
            enable_lora=True,
            lora_max_rank=16,
            disable_radix_cache=False,
            enable_memory_saver=True,
        )
        self.assertEqual(kwargs["model_path"], "m")
        self.assertEqual(kwargs["dp_size"], 1)
        self.assertEqual(kwargs["mem_fraction_static"], 0.6)
        self.assertEqual(kwargs["chunked_prefill_size"], 2048)
        self.assertTrue(kwargs["enable_lora"])
        self.assertEqual(kwargs["max_lora_rank"], 16)
        self.assertIs(kwargs["disable_radix_cache"], False)  # radix cache on
        self.assertTrue(kwargs["enable_memory_saver"])

    def test_disable_radix_cache_mapped(self):
        self.assertIs(self._kwargs(disable_radix_cache=True)["disable_radix_cache"], True)

    def test_chunked_prefill_zero_omitted(self):
        self.assertNotIn("chunked_prefill_size", self._kwargs(chunked_prefill_size=0))

    def test_base_node_has_no_agent_keys(self):
        kwargs = RolloutNode(RolloutConfig(model_path="m"))._engine_kwargs()
        for key in ("disable_radix_cache", "speculative_algorithm", "speculative_num_draft_tokens"):
            self.assertNotIn(key, kwargs)
        self.assertEqual(
            {k: v for k, v in self._kwargs(model_path="m").items() if k != "disable_radix_cache"},
            kwargs,
        )

    def test_agentic_engine_yaml(self):
        path = CONFIGS / "rollout" / "agentic_engine.yaml"
        cfg = AgentRolloutConfig.from_yaml(path)
        self.assertEqual(cfg.n_rpc_workers, 64)
        self.assertFalse(cfg.disable_radix_cache)
        self.assertTrue(cfg.enable_lora)
        self.assertIsNone(cfg.speculative_algorithm)
        self.assertIsNone(cfg.tool_call_parser)
        self.assertIsNone(cfg.apply_chat_template)
        kwargs = AgentRolloutNode(cfg)._engine_kwargs()
        self.assertIs(kwargs["disable_radix_cache"], False)
        self.assertTrue(kwargs["enable_lora"])
        self.assertEqual(kwargs["context_length"], 8192)
        self.assertNotIn("speculative_algorithm", kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "engine.yaml")
            cfg.save_yaml(out)
            self.assertEqual(AgentRolloutConfig.from_yaml(out), cfg)
        # The base config rejects the agent-only keys.
        with self.assertRaisesRegex(ValueError, "n_rpc_workers"):
            RolloutConfig.from_yaml(path)

    def test_yaml_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "engine.yaml")
            with open(path, "w") as f:
                f.write("model_path: m\nspeculative_algorithm_not_a_field: NGRAM\n")
            with self.assertRaisesRegex(ValueError, "speculative_algorithm_not_a_field"):
                AgentRolloutConfig.from_yaml(path)


# ---------------------------------------------------------------------------
# 6. sim_reflect_agent
# ---------------------------------------------------------------------------


def _mk_episode(max_turns: int = 3, turn: int = 1, attempts: list[str] | None = None) -> ReflectEpisode:
    return ReflectEpisode(
        group_id=7,
        episode=1,
        problem="What is 6 * 7?",
        reference="#### 42",
        max_turns=max_turns,
        problem_sha="abc123def4567890",
        group_size=4,
        turn=turn,
        attempts=list(attempts or []),
    )


class TestSimReflectAgentLogic(unittest.TestCase):
    """Pure episode logic: no IPC, no GPU."""

    def setUp(self):
        self.pattern = re.compile(r"#####?\s*(\d[\d,]*)")

    def test_grade_gsm8k(self):
        self.assertTrue(grade_gsm8k("so, 6*7=42 #### 42", "#### 42", self.pattern))
        self.assertTrue(grade_gsm8k("#### 4,200", "#### 4200", self.pattern))
        self.assertFalse(grade_gsm8k("#### 41", "#### 42", self.pattern))
        self.assertFalse(grade_gsm8k("no answer marker", "#### 42", self.pattern))
        self.assertFalse(grade_gsm8k("#### 42", "no reference marker", self.pattern))

    def test_render_prompt_turn1_is_bare_problem(self):
        self.assertEqual(_mk_episode().render_prompt(DEFAULT_RETRY_TEMPLATE), "What is 6 * 7?")

    def test_render_prompt_embeds_history(self):
        ep = _mk_episode(turn=3, attempts=["attempt one", "attempt two"])
        self.assertEqual(
            ep.render_prompt(DEFAULT_RETRY_TEMPLATE).split("\n\n"),
            [
                "What is 6 * 7?",
                "Your previous answer:\nattempt one",
                DEFAULT_RETRY_TEMPLATE,
                "Your previous answer:\nattempt two",
                DEFAULT_RETRY_TEMPLATE,
            ],
        )

    def test_render_messages_structure(self):
        ep = _mk_episode(turn=3, attempts=["attempt one", "attempt two"])
        msgs = ep.render_messages(DEFAULT_RETRY_TEMPLATE, "sys")
        self.assertEqual(
            [m["role"] for m in msgs], ["system", "user", "assistant", "user", "assistant", "user"]
        )
        self.assertEqual(msgs[0]["content"], "sys")
        self.assertEqual(msgs[1]["content"], "What is 6 * 7?")
        self.assertEqual([m["content"] for m in msgs[2::2]], ["attempt one", "attempt two"])
        self.assertTrue(all(m["content"] == DEFAULT_RETRY_TEMPLATE for m in msgs[3::2]))
        # No system prompt -> no system message; turn 1 is the bare problem.
        self.assertEqual(
            _mk_episode().render_messages(DEFAULT_RETRY_TEMPLATE, None),
            [{"role": "user", "content": "What is 6 * 7?"}],
        )

    def test_make_tag_grammar(self):
        self.assertEqual(
            _mk_episode(max_turns=3, turn=3).make_tag()["ww_tag"],
            {
                "task": "gsm8k-reflect",
                "problem": "abc123def4567890",
                "group": 7,
                "group_size": 4,
                "episode": 1,
                "turn": 3,
                "max_turns": 3,
                "terminal": True,  # last turn by budget
                "reference": "#### 42",
            },
        )
        # Earlier turns are not terminal at request time, even if the episode
        # later ends by a correct answer (the reward report closes it).
        self.assertFalse(_mk_episode(max_turns=3, turn=2).make_tag()["ww_tag"]["terminal"])

    def test_problem_sha_is_hash_of_prompt(self):
        problem = "What is 6 * 7?"
        sha = hashlib.sha256(problem.encode()).hexdigest()[:16]
        ep = ReflectEpisode(group_id=1, episode=0, problem=problem, reference="#### 42", max_turns=2, problem_sha=sha)
        self.assertEqual(ep.make_tag()["ww_tag"]["problem"], sha)

    def test_record_turn_first_try_correct(self):
        ep = _mk_episode()
        ep.record_turn("r1", "#### 42", correct=True, discount=1.0)
        self.assertTrue(ep.terminal and ep.correct and ep.first_try_correct)
        self.assertEqual(ep.reward, 1.0)
        self.assertEqual(ep.n_turns, 1)
        self.assertEqual(ep.rollout_ids, ["r1"])

    def test_record_turn_retry_then_discounted_success(self):
        ep = _mk_episode()
        ep.record_turn("r1", "#### 13", correct=False, discount=0.5)
        self.assertFalse(ep.terminal)
        self.assertEqual(ep.turn, 2)
        ep.record_turn("r2", "#### 42", correct=True, discount=0.5)
        self.assertTrue(ep.terminal and ep.correct)
        self.assertFalse(ep.first_try_correct)
        self.assertEqual(ep.reward, 0.5)  # 0.5 ** (turn - 1)
        self.assertEqual(ep.rollout_ids, ["r1", "r2"])

    def test_record_turn_budget_exhausted(self):
        ep = _mk_episode(max_turns=2)
        ep.record_turn("r1", "#### 13", correct=False, discount=1.0)
        ep.record_turn("r2", "#### 13", correct=False, discount=1.0)
        self.assertTrue(ep.terminal)
        self.assertFalse(ep.correct)
        self.assertEqual(ep.reward, 0.0)
        self.assertEqual(ep.n_turns, 2)

    def test_constructor_guards(self):
        self.assertEqual(CARRIERS, ("flat", "messages"))
        with self.assertRaises(ValueError):
            SimReflectAgent(carrier="bogus")
        with self.assertRaises(ValueError):
            SimReflectAgent(discount=0.0)
        with self.assertRaises(ValueError):
            SimReflectAgent(max_turns=0)
        with self.assertLogs("whirlwind.training.sim_reflect_agent", level="WARNING"):
            SimReflectAgent(group_size=1)  # warns, does not refuse


class _CaptureAgentNode:
    """Stands in for the agent's Node: captures requests and reward reports."""

    def __init__(self):
        self.requests: list[ExternalGenerateRequest] = []
        self.reports: list[ReportRewardRequest] = []

    async def call_typed(self, target, endpoint, returns, request):
        self.requests.append(request)
        return ExternalGenerateResult(rollout_ids=["r"], completions=["#### 42"])

    async def call(self, target, endpoint, payload):
        self.reports.append(_from_dict(ReportRewardRequest, payload))
        return {"ok": True}


class TestSimReflectAgentRequests(unittest.TestCase):
    @_async_test
    async def test_flat_carrier_sends_system_prompt(self):
        agent = SimReflectAgent(carrier="flat")
        agent._node = _CaptureAgentNode()
        await agent._serve_one(_mk_episode(turn=2, attempts=["a1"]), "sys")
        req = agent._node.requests[0]
        self.assertEqual(req.system_prompt, "sys")
        self.assertIsNone(req.conversations)
        self.assertIn("Your previous answer:\na1", req.prompts[0])
        self.assertEqual(req.metadata["ww_tag"]["group_size"], 4)

    @_async_test
    async def test_messages_carrier_sends_conversations(self):
        agent = SimReflectAgent(carrier="messages")
        agent._node = _CaptureAgentNode()
        ep = _mk_episode(turn=2, attempts=["a1"])
        await agent._serve_one(ep, "sys")
        req = agent._node.requests[0]
        self.assertIsNone(req.system_prompt)  # travels inside the conversation
        self.assertEqual(req.conversations, [ep.render_messages(DEFAULT_RETRY_TEMPLATE, "sys")])
        self.assertEqual(req.prompts, [ep.render_prompt(DEFAULT_RETRY_TEMPLATE)])
        self.assertEqual((req.group_size, req.metadata), (1, ep.make_tag()))

    @_async_test
    async def test_all_turns_reported_in_one_call(self):
        agent = SimReflectAgent()
        agent._node = _CaptureAgentNode()
        a, b = _mk_episode(), _mk_episode()
        a.record_turn("a1", "#### 42", correct=True, discount=1.0)
        b.record_turn("b1", "#### 1", correct=False, discount=1.0)
        b.record_turn("b2", "#### 1", correct=False, discount=1.0)
        b.record_turn("b3", "#### 1", correct=False, discount=1.0)
        await agent._report_terminal([a, b])
        self.assertEqual(len(agent._node.reports), 1)
        rep = agent._node.reports[0]
        self.assertEqual(list(rep.rollout_ids), ["a1", "b1", "b2", "b3"])
        self.assertEqual(list(rep.rewards), [1.0, 0.0, 0.0, 0.0])


class _GradingBackend(_SleepyBackend):
    """Wrong on the first attempt, right once the retry history is present."""

    def _texts(self, batch: PromptBatch) -> list[str]:
        retry = bool(batch.prompts) and "Your previous answer:" in batch.prompts[0]
        text = "reflecting carefully... #### 42" if retry else "hasty guess #### 13"
        return [text] * batch.batch_size


class _FakeDataloaderNode:
    """Canned ``next_batch`` responder (one problem per wave)."""

    def __init__(self, prompts, responses, system_prompt="few-shot prefix"):
        self.prompts = prompts
        self.responses = responses
        self.system_prompt = system_prompt
        self._node: Node | None = None

    async def start(self):
        self._node = Node("dataloader", auto_process=True)
        self._node.expose("next_batch")(self._next_batch)
        await self._node.start()

    async def stop(self):
        if self._node is not None:
            await self._node.stop()

    async def _next_batch(self, data):
        return _to_dict(
            BatchResult(
                ok=True,
                prompts=list(self.prompts),
                responses=list(self.responses),
                system_prompt=self.system_prompt,
            )
        )


class TestSimReflectAgentIPC(unittest.TestCase):
    """One wave over IPC: agent -> agent proxy -> grading engine."""

    async def _run_wave(self, c: _AgentCluster, **agent_kw) -> None:
        dataloader = _FakeDataloaderNode(prompts=["What is 6 * 7?"], responses=["#### 42"])
        await dataloader.start()
        agent = SimReflectAgent(node_id="sim-reflect-agent", group_size=2, max_turns=3, waves=1, **agent_kw)
        await agent.start()
        try:
            await _wait_topology(agent._node, "rollout", "dataloader")
            await agent.run()
        finally:
            await agent.stop()
            await dataloader.stop()

    @_async_test
    async def test_flat_wave_end_to_end(self):
        """Wrong turn 1, retry, correct turn 2: every turn carries the episode
        return, tags persist, and the turns regroup into one episode group."""
        async with _AgentCluster(_proxy_config(), backend_cls=_GradingBackend) as c:
            await self._run_wave(c)
            out = await c.pop(min_batch=4)
            entries = [BufferEntry.from_dict(e) for e in out.entries]
            self.assertEqual(len(entries), 4)  # 2 episodes x 2 turns
            self.assertEqual([e.reward for e in entries], [1.0] * 4)
            tags = [e.extra["ww_tag"] for e in entries]
            self.assertEqual({t["turn"] for t in tags}, {1, 2})
            self.assertEqual({t["episode"] for t in tags}, {0, 1})
            self.assertEqual({t["group"] for t in tags}, {1})
            self.assertTrue(all(t["group_size"] == 2 for t in tags))
            self.assertTrue(all(t["reference"] == "#### 42" for t in tags))
            # Turn-2 requests were sent before the answer was known.
            self.assertTrue(all(t["terminal"] is False for t in tags if t["turn"] == 2))
            for e in entries:
                if e.extra["ww_tag"]["turn"] == 2:
                    self.assertIn("Your previous answer:", e.prompt)
                    self.assertIn("#### 42", e.completion)
                else:
                    self.assertNotIn("Your previous answer:", e.prompt)
                    self.assertIn("#### 13", e.completion)
            self.assertTrue(all(b.system_prompt == "few-shot prefix" for b in c.backend.batches))
            self.assertTrue(all(b.conversations is None for b in c.backend.batches))

            # The agentic loop regroups the one-sample buffer groups.
            orch = AgenticGRPOOrchestrator(_agentic_config(min_train_batch=4))
            orch._stage(entries)
            batch = orch._take_complete_groups(out.weight_version)
            self.assertEqual(len(batch), 4)
            self.assertEqual(orch.n_groups_trained, 1)

    @_async_test
    async def test_messages_wave_carries_conversations(self):
        async with _AgentCluster(_proxy_config(), backend_cls=_GradingBackend) as c:
            await self._run_wave(c, carrier="messages")
            batches = c.backend.batches
            self.assertEqual(len(batches), 4)
            self.assertTrue(all(b.system_prompt is None for b in batches))
            convs = [b.conversations[0] for b in batches]
            self.assertEqual(
                sorted(len(conv) for conv in convs), [2, 2, 4, 4]
            )  # [system, user] then [system, user, assistant, user]
            retry = [conv for conv in convs if len(conv) == 4]
            for conv in retry:
                self.assertEqual([m["role"] for m in conv], ["system", "user", "assistant", "user"])
                self.assertEqual(conv[0]["content"], "few-shot prefix")
                self.assertEqual(conv[2]["content"], "hasty guess #### 13")
            out = await c.pop(min_batch=4)
            self.assertEqual([BufferEntry.from_dict(e).reward for e in out.entries], [1.0] * 4)


# ---------------------------------------------------------------------------
# 7. sim_swe_agent: tool-call parser and chat episode flow
# ---------------------------------------------------------------------------


def _tool_call_text(name: str, arguments) -> str:
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return _TC_OPEN + '\n{"name": "' + name + '", "arguments": ' + arguments + "}\n" + _TC_CLOSE


def _xml_call(name: str, params: dict) -> str:
    body = "".join(f"{_P_OPEN}{k}>{v}{_P_CLOSE}" for k, v in params.items())
    return f"{_FN_OPEN}{name}>{body}{_FN_CLOSE}"


class TestParseToolCall(unittest.TestCase):
    """Pure parser logic: no IPC, no GPU."""

    def test_valid_call_dict_arguments(self):
        self.assertEqual(
            parse_tool_call(_tool_call_text("bash", {"command": "ls -la"})),
            {"name": "bash", "arguments": {"command": "ls -la"}},
        )

    def test_valid_call_string_arguments(self):
        # Arguments given as a JSON-encoded string.
        text = _tool_call_text("bash", json.dumps(json.dumps({"command": "echo hi"})))
        self.assertEqual(parse_tool_call(text)["arguments"]["command"], "echo hi")

    def test_no_marker_returns_none(self):
        self.assertIsNone(parse_tool_call("I will now run a command."))
        self.assertIsNone(parse_tool_call(""))

    def test_malformed_json_returns_none(self):
        self.assertIsNone(parse_tool_call(_TC_OPEN + "\n{not json at all\n" + _TC_CLOSE))

    def test_truncated_close_marker_still_parses(self):
        tc = parse_tool_call(_TC_OPEN + '\n{"name": "bash", "arguments": {"command": "pwd"}}')
        self.assertEqual(tc, {"name": "bash", "arguments": {"command": "pwd"}})

    def test_xml_grammar_single_param(self):
        self.assertEqual(
            parse_tool_call(_xml_call("bash", {"command": "echo hi > f.txt"})),
            {"name": "bash", "arguments": {"command": "echo hi > f.txt"}},
        )

    def test_xml_grammar_multiline_value(self):
        call = _xml_call("bash", {"command": "sed -i 's/a/b/' x.py"}).replace(">sed", ">\nsed")
        self.assertEqual(parse_tool_call(call)["arguments"]["command"], "sed -i 's/a/b/' x.py")

    def test_xml_grammar_multi_param(self):
        tc = parse_tool_call(_xml_call("edit", {"path": "a.py", "content": "x = 1"}))
        self.assertEqual(tc, {"name": "edit", "arguments": {"path": "a.py", "content": "x = 1"}})

    def test_xml_grammar_malformed_returns_none(self):
        self.assertIsNone(parse_tool_call(_FN_OPEN + "bash>" + _P_OPEN + "command>echo hi"))
        self.assertIsNone(parse_tool_call("plain text, no grammar at all"))

    def test_hermes_preferred_when_present(self):
        both = _tool_call_text("bash", {"command": "from-hermes"}) + _xml_call("bash", {"command": "from-xml"})
        self.assertEqual(parse_tool_call(both)["arguments"]["command"], "from-hermes")

    def test_non_dict_or_missing_name_returns_none(self):
        self.assertIsNone(parse_tool_call(_TC_OPEN + "\n[1,2]\n" + _TC_CLOSE))
        self.assertIsNone(parse_tool_call(_TC_OPEN + '\n{"arguments": {}}\n' + _TC_CLOSE))

    def test_wrapped_xml_grammar(self):
        # The wrapper may hold the XML grammar instead of hermes JSON.
        wrapped = _TC_OPEN + "\n" + _xml_call("bash", {"command": "ls /testbed"}) + "\n" + _TC_CLOSE
        self.assertEqual(parse_tool_call(wrapped), {"name": "bash", "arguments": {"command": "ls /testbed"}})

    def test_wrapped_xml_after_reasoning(self):
        wrapped = (
            "Let me explore the repo.\n" + _THINK_CLOSE + "\n\n" + _TC_OPEN + "\n"
            + _xml_call("bash", {"command": "find /testbed -name x.py"}) + "\n" + _TC_CLOSE
        )
        self.assertEqual(parse_tool_call(wrapped)["arguments"]["command"], "find /testbed -name x.py")

    def test_wrapped_xml_with_braces_in_value(self):
        # Braces in an XML value are not a hermes payload.
        wrapped = _TC_OPEN + "\n" + _xml_call("bash", {"command": "sed 's/{a}/{b}/' x.py"}) + "\n" + _TC_CLOSE
        self.assertEqual(parse_tool_call(wrapped)["arguments"]["command"], "sed 's/{a}/{b}/' x.py")

    def test_wrapped_xml_truncated_returns_none(self):
        truncated = _TC_OPEN + "\n" + _FN_OPEN + "bash>\n" + _P_OPEN + "command>\nfind /testbed -type f"
        self.assertIsNone(parse_tool_call(truncated))

    def test_wrapped_prose_returns_none(self):
        self.assertIsNone(parse_tool_call(_TC_OPEN + "\nI will just think out loud.\n" + _TC_CLOSE))

    def test_hermes_valid_beats_xml_in_same_wrapper(self):
        both = (
            _TC_OPEN + "\n" + json.dumps({"name": "bash", "arguments": {"command": "hermes"}})
            + "\n" + _xml_call("bash", {"command": "xml"}) + "\n" + _TC_CLOSE
        )
        self.assertEqual(parse_tool_call(both)["arguments"]["command"], "hermes")


class _FakeRunner:
    """ContainerRunner stand-in: records execs, returns canned output."""

    def __init__(self, outputs: list[str] | None = None):
        self.execs: list[str] = []
        self.outputs = list(outputs or [""])

    async def create(self, name: str, image: str):
        self.created = name

    async def copy_in(self, name: str, src: str, dest: str):
        pass

    async def exec(self, name: str, command: str, timeout: float | None = None):
        self.execs.append(command)
        return 0, (self.outputs.pop(0) if self.outputs else "")

    async def remove(self, name: str):
        pass


class _FakeSweNode:
    """Stands in for the agent's Node: records heartbeats."""

    def __init__(self):
        self.touches: list[list[str]] = []

    async def call(self, target: str, method: str, payload: dict):
        if method == "touch":
            self.touches.append(list(_from_dict(TouchRequest, payload).rollout_ids))
        return {"ok": True, "n_touched": 1, "missing": [], "n_recorded": 1}


class _JinjaTokenizer:
    """Renders a small chat template that iterates tool-call arguments as a
    mapping, like some real templates do."""

    TEMPLATE = (
        "{% for m in messages %}<{{ m.role }}>{{ m.content }}"
        "{% for tc in (m.tool_calls or []) %}[{{ tc.function.name }}"
        "{% for k, v in tc.function.arguments.items() %} {{ k }}={{ v }}{% endfor %}]"
        "{% endfor %}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )

    def apply_chat_template(self, conversation, tools=None, tokenize=True, add_generation_prompt=False, **kw):
        import jinja2

        return jinja2.Environment().from_string(self.TEMPLATE).render(
            messages=conversation, tools=tools, add_generation_prompt=add_generation_prompt
        )


class TestChatEpisodeFlow(unittest.TestCase):
    """Chat-style SWE episode loop with faked containers and generation."""

    def _agent(self) -> SimSweAgent:
        agent = SimSweAgent(task_family="swe-gym", prompt_style=STYLE_CHAT, run_id="chat-test")
        agent._node = _FakeSweNode()
        return agent

    def _episode(self, agent: SimSweAgent, problem: str = "Make the thing work.") -> SweEpisode:
        inst = SweInstance(
            instance_id="moto__moto-1234",
            repo="getmoto/moto",
            base_commit="abc",
            problem_statement=problem,
            test_patch="diff --git a/test.py b/test.py",
            fail_to_pass=["test_thing"],
            pass_to_pass=[],
        )
        return SweEpisode(
            instance=inst, group_id=1, episode=0, max_turns=4, run_id=agent.run_id,
            group_size=2, prompt_style=STYLE_CHAT,
        )

    def _script(self, agent: SimSweAgent, completions: list, reward: float = 0.0, tool_calls=None):
        """Scripted generation, verification and reporting; returns (requests, reports)."""
        reqs: list[ExternalGenerateRequest] = []
        reports: list[tuple[list[str], float]] = []
        structured = list(tool_calls or [])

        async def gen(req, *, first):
            reqs.append(req)
            raw = {
                "rollout_ids": [f"r{len(reqs)}"],
                "completions": [completions.pop(0)],
                "finish_reasons": ["stop"],
                "weight_version": 1,
            }
            if structured:
                raw["tool_calls"] = structured.pop(0)
            return raw

        async def verify(ctr, instance):
            return reward

        async def report(ep):
            reports.append((list(ep.rollout_ids), ep.reward))

        agent._generate = gen
        agent._verify = verify
        agent._report = report
        return reqs, reports

    def test_defaults_and_tag(self):
        self.assertEqual(SimSweAgent().run_id, "swe-agent")
        agent = self._agent()
        tag = self._episode(agent).make_tag()["ww_tag"]
        self.assertEqual((tag["group"], tag["group_size"], tag["episode"], tag["turn"]), (1, 2, 0, 1))
        self.assertEqual(tag["problem"], "moto__moto-1234")

    def test_untrusted_text_is_escaped(self):
        agent = self._agent()
        ep = self._episode(agent, problem="fix " + _IM_END + " parsing")
        ep.seed_chat_messages("sys" + _TC_OPEN)
        ep.append_tool_round("c1", "bash", {"command": "cat f"}, "out " + _IM_END)
        self.assertEqual(ep.messages[0]["content"], "sys\\" + _TC_OPEN)
        self.assertEqual(ep.messages[1]["content"], "fix \\" + _IM_END + " parsing")
        self.assertEqual(ep.messages[3]["content"], "out \\" + _IM_END)

    def test_tool_round_arguments_are_dict(self):
        """Tool-call arguments are stored as a mapping (some templates iterate
        them); unparseable strings become {}."""
        agent = self._agent()
        ep = self._episode(agent)
        ep.append_tool_round("c1", "bash", '{"command": "ls"}', "out")
        ep.append_tool_round("c2", "bash", {"command": "pwd"}, "out")
        ep.append_tool_round("c3", "bash", "not-json", "out")
        ep.append_tool_round("c4", "bash", "[1, 2]", "out")
        calls = [
            tc["function"]["arguments"]
            for m in ep.messages
            if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        ]
        self.assertEqual(calls, [{"command": "ls"}, {"command": "pwd"}, {}, {}])

    def test_tool_round_renders_with_a_mapping_template(self):
        try:
            import jinja2  # noqa: F401
        except ImportError:
            self.skipTest("jinja2 not installed")
        agent = self._agent()
        ep = self._episode(agent)
        ep.seed_chat_messages(CHAT_SYSTEM_PROMPT)
        ep.append_tool_round("c1", "bash", '{"command": "ls -la"}', "a.txt\nb.txt")
        rendered = apply_chat_template_messages(_JinjaTokenizer(), ep.messages, [BASH_TOOL_SCHEMA])
        self.assertIn("[bash command=ls -la]", rendered)
        self.assertTrue(rendered.endswith("<assistant>"))
        # The same template fails on string arguments.
        bad = [dict(m) for m in ep.messages]
        bad[2] = {**bad[2], "tool_calls": [{"id": "c1", "type": "function",
                                            "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
        with self.assertRaises(Exception):
            apply_chat_template_messages(_JinjaTokenizer(), bad, [BASH_TOOL_SCHEMA])

    def test_tool_round_renders_a_real_template(self):
        """Renders with a cached Qwen3-family tokenizer; skips if none is cached."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers not installed")
        tok = None
        for name in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"):
            try:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
                break
            except Exception:
                continue
        if tok is None:
            self.skipTest("no Qwen3-family tokenizer cached locally")
        agent = self._agent()
        ep = self._episode(agent)
        ep.seed_chat_messages(CHAT_SYSTEM_PROMPT)
        ep.append_tool_round("c1", "bash", '{"command": "ls -la"}', "a.txt\nb.txt")
        rendered = apply_chat_template_messages(tok, ep.messages, [BASH_TOOL_SCHEMA], {"enable_thinking": True})
        self.assertIn("ls -la", rendered)

    @_async_test
    async def test_bash_then_finish_flow(self):
        agent = self._agent()
        agent._runner = _FakeRunner(outputs=[""])
        reqs, reports = self._script(
            agent,
            [_tool_call_text("bash", {"command": "echo hi > f.txt"}), _tool_call_text("finish", {})],
            reward=1.0,
        )
        ep = self._episode(agent)
        await agent._run_episode(ep, asyncio.Semaphore(1))

        self.assertEqual(len(ep.rollout_ids), 2)
        self.assertTrue(ep.terminal and ep.ended_early)
        self.assertEqual(ep.history, [("echo hi > f.txt", "")])
        self.assertEqual(ep.reward, 1.0)
        # Request 1: seeded conversation (system + user) and the tools.
        r1 = reqs[0]
        self.assertEqual([m["role"] for m in r1.conversations[0]], ["system", "user"])
        self.assertIn("Make the thing work.", r1.conversations[0][1]["content"])
        self.assertEqual([t["function"]["name"] for t in r1.tools], ["bash", "finish"])
        self.assertIsNone(r1.system_prompt)
        self.assertEqual(r1.metadata["ww_tag"]["group_size"], 2)
        # Request 2: the assistant tool call and its result.
        conv2 = reqs[1].conversations[0]
        self.assertEqual(len(conv2), 4)
        self.assertEqual(conv2[2]["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(conv2[3]["role"], "tool")
        self.assertEqual(conv2[3]["tool_call_id"], conv2[2]["tool_calls"][0]["id"])
        self.assertEqual(agent._runner.execs, ["echo hi > f.txt"])
        # One heartbeat per turn after the first, covering the whole episode.
        self.assertEqual(agent._node.touches, [["r1", "r2"]])
        self.assertEqual(reports, [(["r1", "r2"], 1.0)])

    @_async_test
    async def test_malformed_tool_call_is_fail_closed_round(self):
        agent = self._agent()
        agent._runner = _FakeRunner(outputs=[""])
        reqs, _ = self._script(
            agent,
            [
                _tool_call_text("bash", {"nope": 1}),
                _tool_call_text("bash", {"command": "true"}),
                _tool_call_text("finish", {}),
            ],
        )
        ep = self._episode(agent)
        await agent._run_episode(ep, asyncio.Semaphore(1))
        self.assertEqual(agent._runner.execs, ["true"])
        self.assertEqual(ep.n_cmds_failed, 1)
        conv3 = reqs[2].conversations[0]
        self.assertEqual(len(conv3), 6)
        self.assertEqual((conv3[2]["role"], conv3[3]["role"]), ("assistant", "tool"))
        self.assertIn("tool error", conv3[3]["content"])
        self.assertEqual(len(ep.rollout_ids), 3)
        self.assertEqual(agent._node.touches, [["r1", "r2"], ["r1", "r2", "r3"]])

    @_async_test
    async def test_repetition_guard_nudge_then_terminate(self):
        """Identical consecutive commands: two execute, the third nudges,
        the fourth ends the episode."""
        agent = self._agent()
        agent._runner = _FakeRunner(outputs=["err"] * 8)
        same = _tool_call_text("bash", {"command": "failing repro"})
        reqs, _ = self._script(agent, [same] * 5)
        ep = self._episode(agent)
        await agent._run_episode(ep, asyncio.Semaphore(1))
        self.assertEqual(agent._runner.execs, ["failing repro", "failing repro"])
        self.assertTrue(ep.terminal and ep.ended_early)
        self.assertEqual((ep.n_identical, ep.n_nudged), (4, 1))
        self.assertEqual(len(ep.rollout_ids), 4)
        conv4 = reqs[3].conversations[0]
        self.assertEqual(conv4[-1]["role"], "tool")
        self.assertIn("Do not repeat it", conv4[-1]["content"])

    @_async_test
    async def test_plain_prose_is_fail_closed_round_not_finish(self):
        """A reply without a tool call is an error round, not a finish."""
        agent = self._agent()
        agent._runner = _FakeRunner()
        self._script(
            agent,
            ["Let me think about this problem.\n\n" + _IM_END, _tool_call_text("finish", {})],
            reward=0.5,
        )
        ep = self._episode(agent)
        await agent._run_episode(ep, asyncio.Semaphore(1))
        self.assertTrue(ep.terminal and ep.ended_early)
        self.assertEqual(agent._runner.execs, [])
        self.assertEqual(len(ep.rollout_ids), 2)
        tool_rounds = [m for m in ep.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_rounds), 1)
        self.assertIn("no bash tool call", tool_rounds[0]["content"])

    @_async_test
    async def test_unparseable_rounds_consume_the_turn_budget(self):
        agent = self._agent()
        agent._runner = _FakeRunner()
        reqs, reports = self._script(agent, ["reasoning-only prose, no call"] * 8)
        ep = self._episode(agent)  # max_turns=4
        await agent._run_episode(ep, asyncio.Semaphore(1))
        self.assertTrue(ep.terminal)
        self.assertFalse(ep.ended_early)
        self.assertEqual(agent._runner.execs, [])
        self.assertEqual(len(ep.rollout_ids), 4)
        self.assertEqual(ep.n_cmds_failed, 4)
        self.assertEqual([r.metadata["ww_tag"]["turn"] for r in reqs], [1, 2, 3, 4])
        self.assertEqual(len([m for m in ep.messages if m.get("role") == "tool"]), 4)
        self.assertEqual(reports, [(["r1", "r2", "r3", "r4"], 0.0)])

    @_async_test
    async def test_engine_parsed_calls_take_precedence(self):
        """Structured calls from the engine win over the text parser."""
        agent = self._agent()
        agent._runner = _FakeRunner(outputs=[""])
        self._script(
            agent,
            ["reasoning prose the engine parsed into a call", _tool_call_text("finish", {})],
            reward=1.0,
            tool_calls=[
                [[{"name": "bash", "arguments": {"command": "echo structured"}}]],
                [[{"name": "finish", "arguments": {}}]],
            ],
        )
        ep = self._episode(agent)
        await agent._run_episode(ep, asyncio.Semaphore(1))
        self.assertEqual(agent._runner.execs, ["echo structured"])
        self.assertTrue(ep.terminal)
        self.assertEqual(len(ep.rollout_ids), 2)

    @_async_test
    async def test_explicit_end_finishes_episode(self):
        agent = self._agent()
        agent._runner = _FakeRunner()
        self._script(agent, ["END"], reward=0.5)
        ep = self._episode(agent)
        await agent._run_episode(ep, asyncio.Semaphore(1))
        self.assertTrue(ep.ended_early)
        self.assertEqual(agent._runner.execs, [])
        self.assertEqual(len(ep.rollout_ids), 1)
        self.assertEqual(agent._node.touches, [])  # single-turn episode: no heartbeat


# ---------------------------------------------------------------------------
# 8. Configs
# ---------------------------------------------------------------------------

_CONFIG_ENV = (
    "WHIRLWIND_ROLLOUT_CONFIG",
    "WHIRLWIND_REWARD_CONFIG",
    "WHIRLWIND_MODEL_CONFIG",
    "WW_ASYNC_TOTAL_STEPS",
)


# ---------------------------------------------------------------------------
# sim_os_agent — procedurally generated tool-use tasks
# ---------------------------------------------------------------------------


def _os_oracle(ep: OsEpisode) -> tuple[str, list[dict]]:
    """A scripted policy that solves every task kind with the intended tools."""
    t = ep.task
    last = ep.messages[-1]["content"].split("\n" + LAST_TURN_HINT)[0]
    if t.kind in ("time", "date", "weekday"):
        if ep.turn == 1:
            return "", [{"name": "get_time", "arguments": {}}]
        day, clock, weekday = last.split()
        return {"time": clock[:5], "date": day, "weekday": weekday.strip("()")}[t.kind], []
    if t.kind == "find_dir":
        name = t.question.split("file ")[1].split("?")[0]
        if ep.turn == 1:
            return "", [{"name": "find_file", "arguments": {"name": name}}]
        return last.rsplit("/", 1)[0], []
    if t.kind == "read_file":
        path = t.question.split("written in ")[1].split("?")[0]
        if ep.turn == 1:
            return "", [{"name": "read_file", "arguments": {"path": path}}]
        return f"`{last.split('access code: ')[1].strip()}`", []
    if t.kind == "find_read":
        name = t.question.split("named ")[1].split(" somewhere")[0]
        if ep.turn == 1:
            return "", [{"name": "find_file", "arguments": {"name": name}}]
        if ep.turn == 2:
            return "", [{"name": "read_file", "arguments": {"path": last}}]
        return last.split("access code: ")[1].strip(), []
    d = t.question.split("inside ")[1].split("?")[0]
    if ep.turn == 1:
        return "", [{"name": "list_dir", "arguments": {"path": d}}]
    return str(sum(1 for ln in last.splitlines() if not ln.endswith("/"))), []


def _os_episode(task, **kw) -> OsEpisode:
    kw.setdefault("max_turns", 4)
    return OsEpisode(task=task, group_id=1, group_size=8, episode=0, **kw)


class TestOsTasks(unittest.TestCase):
    def test_reproducible_and_seed_dependent(self):
        a, b, c = make_tasks(16, 0), make_tasks(16, 0), make_tasks(16, 1)
        self.assertEqual([t.question for t in a], [t.question for t in b])
        self.assertEqual([t.answer for t in a], [t.answer for t in b])
        self.assertNotEqual([t.answer for t in a], [t.answer for t in c])

    def test_kinds_cycle_evenly(self):
        tasks = make_tasks(7 * 4, 3)
        for kind in TASK_KINDS:
            self.assertEqual(sum(t.kind == kind for t in tasks), 4)

    def test_file_names_unique_and_evidence_set(self):
        for t in make_tasks(64, 5):
            names = [f.rsplit("/", 1)[1] for f in t.fs.files]
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(t.evidence)

    def test_oracle_solves_every_task_exactly(self):
        for t in make_tasks(128, 0):
            ep = _os_episode(t)
            while not ep.terminal:
                ep.step(*_os_oracle(ep))
            self.assertTrue(ep.correct, (t.kind, t.question, t.answer, ep.final_reply))
            self.assertTrue(ep.found)
            self.assertEqual(ep.reward, 1.0)
            self.assertEqual(ep.n_valid_calls, ep.n_calls)


class TestVirtualFS(unittest.TestCase):
    def _fs(self) -> VirtualFS:
        from datetime import datetime

        return VirtualFS(
            files={"/home/user/docs/a.txt": "hello\n", "/home/user/docs/sub/b.md": "x"},
            dirs={"/", "/home", "/home/user", "/home/user/docs", "/home/user/docs/sub"},
            now=datetime(2025, 3, 4, 5, 6, 7),
        )

    def test_list_dir_and_path_forms(self):
        fs = self._fs()
        self.assertEqual(fs.list_dir("/home/user/docs"), "a.txt\nsub/")
        self.assertEqual(fs.list_dir("docs/"), "a.txt\nsub/")  # relative to home
        self.assertEqual(fs.list_dir("~"), "docs/")
        self.assertEqual(fs.list_dir("/home/user/docs/sub/"), "b.md")
        self.assertIn("no such directory", fs.list_dir("/nope"))
        self.assertIn("not a directory", fs.list_dir("/home/user/docs/a.txt"))

    def test_find_and_read(self):
        fs = self._fs()
        self.assertEqual(fs.find_file("b.md"), "/home/user/docs/sub/b.md")
        self.assertIn("no files named", fs.find_file("c.md"))
        self.assertEqual(fs.read_file("docs/a.txt"), "hello\n")
        self.assertIn("is a directory", fs.read_file("/home/user/docs"))
        self.assertIn("no such file", fs.read_file("/home/user/zz.txt"))

    def test_time_and_call_validation(self):
        fs = self._fs()
        self.assertEqual(fs.get_time(), "2025-03-04 05:06:07 (Tuesday)")
        self.assertEqual(fs.call("get_time", {}), (fs.get_time(), True))
        obs, ok = fs.call("read_file", {})  # missing argument
        self.assertFalse(ok)
        self.assertIn("missing or invalid arguments", obs)
        obs, ok = fs.call("rm", {"path": "/"})
        self.assertFalse(ok)
        self.assertIn("unknown tool", obs)

    def test_tool_schemas(self):
        names = [t["function"]["name"] for t in TOOLS]
        self.assertEqual(names, ["list_dir", "find_file", "read_file", "get_time"])
        for t in TOOLS:
            self.assertEqual(t["type"], "function")
            params = t["function"]["parameters"]
            self.assertEqual(set(params["required"]), set(params["properties"]))


class TestOsGrading(unittest.TestCase):
    def test_exact_prefix_none(self):
        cases = [
            ("09:27", "09:27", "exact"),
            ("Answer: Q4ZT7B", "Q4ZT7B", "exact"),
            ("The answer is `Q4ZT7B`.", "Q4ZT7B", "exact"),
            ("Wednesday.", "Wednesday", "exact"),
            ("/home/user/docs/", "/home/user/docs", "exact"),
            ("14:49 sure", "14:49", "prefix"),
            ("Wednesday, I think", "Wednesday", "prefix"),
            ("1 file", "1", "prefix"),
            ("10 files", "1", "none"),
            ("/home/user/docs/alpha", "/home/user/docs", "none"),
            ("junk\n09:27", "09:27", "none"),  # only the first line counts
            ("", "09:27", "none"),
        ]
        for reply, want, expected in cases:
            self.assertEqual(grade(reply, want), expected, (reply, want))

    def test_long_line_gets_no_exact_credit(self):
        reply = "Q4ZT7B " + "x" * 200
        self.assertEqual(grade(reply, "Q4ZT7B"), "prefix")
        listing = "/home/user/a /home/user/b /home/user/docs " * 10
        self.assertEqual(grade(listing, "/home/user/docs"), "none")

    def test_special_tokens_and_think_ignored(self):
        reply = "<" + "think>" + "hmm" + _THINK_CLOSE + "09:27" + _IM_END
        self.assertEqual(grade(reply, "09:27"), "exact")


class TestOsReplyCalls(unittest.TestCase):
    def test_structured_calls_win(self):
        calls, wrapped = reply_calls("text", [{"name": "get_time", "arguments": {}}])
        self.assertEqual((calls, wrapped), ([{"name": "get_time", "arguments": {}}], True))

    def test_wrapped_text_call(self):
        text = _tool_call_text("list_dir", {"path": "/home/user"})
        calls, wrapped = reply_calls(text, [])
        self.assertTrue(wrapped)
        self.assertEqual(calls, [{"name": "list_dir", "arguments": {"path": "/home/user"}}])

    def test_bare_json_call_counts_but_is_not_wrapped(self):
        calls, wrapped = reply_calls(' []{"name": "get_time", "arguments": {}}\njunk', None)
        self.assertEqual((calls, wrapped), ([{"name": "get_time", "arguments": {}}], False))
        calls, _ = reply_calls('{"name": "read_file", "arguments": "{\\"path\\": \\"a\\"}"}', None)
        self.assertEqual(calls, [{"name": "read_file", "arguments": {"path": "a"}}])

    def test_answers_and_late_json_are_not_calls(self):
        self.assertEqual(reply_calls("09:27", None), ([], False))
        late = "x" * 40 + '{"name": "get_time", "arguments": {}}'
        self.assertEqual(reply_calls(late, None), ([], False))
        self.assertEqual(reply_calls('{"day": "Monday"}', None), ([], False))


class TestOsEpisode(unittest.TestCase):
    def setUp(self):
        self.task = next(t for t in make_tasks(14, 0) if t.kind == "find_read")

    def test_messages_open_with_demos(self):
        ep = _os_episode(self.task)
        self.assertEqual(ep.messages[0]["role"], "system")
        self.assertEqual(ep.messages[-1], {"role": "user", "content": self.task.question})
        self.assertTrue(any(m.get("tool_calls") for m in ep.messages[:-1]))
        bare = _os_episode(self.task, fewshot=False)
        self.assertEqual([m["role"] for m in bare.messages], ["system", "user"])

    def test_tool_round_appends_call_and_result(self):
        ep = _os_episode(self.task)
        n = len(ep.messages)
        name = self.task.question.split("named ")[1].split(" somewhere")[0]
        ep.step("", [{"name": "find_file", "arguments": {"name": name}}], wrapped=False)
        self.assertEqual(ep.turn, 2)
        call_msg, result = ep.messages[n], ep.messages[n + 1]
        self.assertEqual(call_msg["role"], "assistant")
        self.assertEqual(call_msg["tool_calls"][0]["function"]["arguments"], {"name": name})
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], call_msg["tool_calls"][0]["id"])
        self.assertEqual((ep.n_calls, ep.n_valid_calls, ep.n_wrapped_calls), (1, 1, 0))

    def test_last_turn_hint_and_budget(self):
        ep = _os_episode(self.task, max_turns=2, evidence_reward=0.2)
        ep.step("", [{"name": "get_time", "arguments": {}}])
        self.assertTrue(ep.messages[-1]["content"].endswith(LAST_TURN_HINT))
        ep.step("", [{"name": "get_time", "arguments": {}}])  # never answers
        self.assertTrue(ep.terminal)
        self.assertFalse(ep.answered)
        self.assertEqual((ep.found, ep.reward), (False, 0.0))

    def test_reward_levels(self):
        path = next(p for p, c in self.task.fs.files.items() if "access code" in c)

        def run(answer, **kw):
            ep = _os_episode(self.task, **kw)
            ep.step("", [{"name": "read_file", "arguments": {"path": path}}])
            ep.step(answer, [])
            return ep

        self.assertEqual(run(self.task.answer).reward, 1.0)
        self.assertEqual(run(self.task.answer + " ok", prefix_reward=0.5).reward, 0.5)
        self.assertEqual(run("no idea", evidence_reward=0.2).reward, 0.2)  # found, wrong answer
        self.assertEqual(run("no idea").reward, 0.0)
        ep = _os_episode(self.task, evidence_reward=0.2)
        ep.step("no idea", [])  # answered without looking
        self.assertEqual((ep.found, ep.reward), (False, 0.0))

    def test_tag_and_stats(self):
        ep = _os_episode(self.task)
        tag = ep.make_tag()["ww_tag"]
        self.assertEqual(
            {k: tag[k] for k in ("task", "problem", "group", "group_size", "episode", "turn")},
            {"task": "os-tools", "problem": self.task.task_id, "group": 1, "group_size": 8,
             "episode": 0, "turn": 1},
        )
        ep.step(self.task.answer, [])
        stats = episode_stats([ep], "os")
        self.assertEqual(stats["os/success"], 1.0)
        self.assertEqual(stats[f"os/success/{self.task.kind}"], 1.0)
        self.assertEqual(stats["os/no_tool_answer_frac"], 1.0)


class TestOsLiveClock(unittest.TestCase):
    def test_posix_tz_strings(self):
        self.assertEqual((posix_tz(9), posix_tz(-5), posix_tz(0)), ("SIM-9", "SIM+5", "SIM+0"))

    def test_tz_offsets_are_spread_and_do_not_change_the_tasks(self):
        tasks = make_tasks(64, 0)
        self.assertTrue(all(-10 <= t.fs.tz_offset_h <= 12 for t in tasks))
        self.assertGreater(len({t.fs.tz_offset_h for t in tasks}), 10)

    def test_live_clock_accepts_the_episode_window(self):
        from datetime import datetime, timedelta, timezone

        task = next(t for t in make_tasks(7, 0) if t.kind == "time")
        ep = _os_episode(task, live_clock=True)
        now = datetime.now(timezone(timedelta(hours=task.fs.tz_offset_h)))
        self.assertIn(now.strftime("%H:%M"), ep.expected_answers())
        ep.step("", [{"name": "get_time", "arguments": {}}], results=[("x", True)])
        self.assertTrue(ep.found)  # any successful get_time is the evidence
        ep.step(now.strftime("%H:%M"), [])
        self.assertTrue(ep.correct)
        fixed = _os_episode(task)  # memory backend: the task's fixed clock
        self.assertEqual(fixed.expected_answers(), [task.answer])

    def test_external_results_are_recorded(self):
        task = next(t for t in make_tasks(7, 0) if t.kind == "find_dir")
        ep = _os_episode(task)
        ep.step("", [{"name": "find_file", "arguments": {"name": "x"}}], results=[("out", True)])
        self.assertEqual(ep.messages[-1]["content"], "out")
        self.assertEqual(call_error("find_file", {}), "error: missing or invalid arguments for find_file: {}")


def _podman_usable() -> bool:
    import shutil
    import subprocess

    if shutil.which("podman") is None:
        return False
    try:
        out = subprocess.run(
            ["podman", "image", "exists", "docker.io/library/python:3.12-slim"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


@unittest.skipUnless(_podman_usable(), "podman with python:3.12-slim not available")
class TestPodmanSandbox(unittest.TestCase):
    """Real container: the oracle solves tasks through real ls/find/cat/date."""

    def _left(self, label: str) -> list[str]:
        import subprocess

        return subprocess.run(
            ["podman", "ps", "-aq", "--filter", f"label={CONTAINER_LABEL}={label}"],
            capture_output=True, text=True,
        ).stdout.split()

    @_async_test
    async def test_oracle_solves_tasks_in_the_shared_container_and_cleans_up(self):
        label = f"test-{os.getpid()}"
        tasks = make_tasks(7, 11)  # one task per kind
        box = PodmanSandbox(label=label, concurrency=8)
        try:
            await box.start(tasks)
            self.assertEqual(len(self._left(label)), 1)  # one container for every task
            host_dir = box._host_dir
            for t in tasks:
                ep = _os_episode(t, live_clock=True)
                while not ep.terminal:
                    reply, calls = _os_oracle(ep)
                    results = [await box.call(t, c["name"], c["arguments"]) for c in calls]
                    ep.step(reply, calls, results=results or None)
                self.assertTrue(ep.correct, (t.kind, t.question, ep.expected_answers(), ep.final_reply))
                self.assertTrue(ep.found)
            # Real OS behaviour; errors come from the commands, in the model's paths.
            obs, ok = await box.call(tasks[0], "read_file", {"path": "/nope"})
            self.assertTrue(ok)
            self.assertEqual(obs, "error: cat: /nope: No such file or directory")
            obs, _ = await box.call(tasks[0], "list_dir", {"path": "/"})
            self.assertEqual(obs, "home/")
            obs, ok = await box.call(tasks[0], "rm", {"path": "/"})
            self.assertFalse(ok)
            # Locked down: no network, read-only.
            rc, out, _ = await box._run("exec", box.name, "ls", "/sys/class/net")
            interfaces = [x for x in out.split() if x != "bonding_masters"]  # a control file
            self.assertEqual((rc, interfaces), (0, ["lo"]))
            rc, _, _ = await box._run("exec", box.name, "touch", "/tasks/t0/x")
            self.assertNotEqual(rc, 0)
        finally:
            await box.remove_all()
        self.assertEqual(self._left(label), [])
        self.assertFalse(os.path.exists(host_dir))

    @_async_test
    async def test_tasks_cannot_see_each_other(self):
        label = f"test-iso-{os.getpid()}"
        a, b = make_tasks(2, 23)
        box = PodmanSandbox(label=label)
        try:
            await box.start([a, b])
            b_file = sorted(b.fs.files)[0]
            name = posixpath.basename(b_file)
            if name not in {posixpath.basename(f) for f in a.fs.files}:
                obs, _ = await box.call(a, "find_file", {"name": name})
                self.assertEqual(obs, f"no files named {name!r}")
            for path in (b_file, "/tasks/t1" + b_file, "../../../tasks/t1" + b_file, "/../t1" + b_file):
                obs, _ = await box.call(a, "read_file", {"path": path})
                self.assertTrue(obs.startswith("error:"), (path, obs))
                self.assertNotIn(box._roots[a.task_id], obs)  # the real root never shows
            obs, _ = await box.call(b, "read_file", {"path": b_file})
            self.assertEqual(obs, b.fs.files[b_file])
            obs, _ = await box.call(b, "find_file", {"name": name})
            self.assertEqual(obs, b_file)  # mapped back to the model's view
        finally:
            await box.remove_all()

    @_async_test
    async def test_agent_episode_runs_its_tools_in_the_container(self):
        label = f"test-agent-{os.getpid()}"
        agent = SimOsAgent(
            n_tasks=7, n_eval_tasks=0, group_size=2, backend="podman", container_label=label
        )
        task = next(t for t in agent.tasks if t.kind == "find_dir")
        name = task.question.split("file ")[1].split("?")[0]
        call = json.dumps({"name": "find_file", "arguments": {"name": name}})
        agent._node = _OsCaptureNode([call, task.answer])
        try:
            await agent.sandboxes.start([task])
            ep = OsEpisode(task=task, group_id=1, group_size=2, episode=0, max_turns=4,
                          **agent.episode_kwargs)
            await agent._run_episode(ep)
        finally:
            await agent.sandboxes.remove_all()
        tool_msg = agent._node.requests[1].conversations[0][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["content"], f"{task.answer}/{name}")  # real `find` output
        self.assertEqual((ep.correct, ep.found, ep.reward), (True, True, 1.0))
        report = _from_dict(ReportRewardRequest, agent._node.calls[-1][1])
        self.assertEqual(list(report.rewards), [1.0, 1.0])


class TestOsConstrainedCalls(unittest.TestCase):
    """--constrain-calls: grammar-enforced <tool_call> turns plus the answer tool."""

    def _accepts(self):
        try:
            import xgrammar as xgr
            from xgrammar.testing import _is_grammar_accept_string
        except ImportError:
            self.skipTest("xgrammar not installed")
        grammar = xgr.Grammar.from_ebnf(tool_call_grammar(TOOLS + [ANSWER_TOOL]))
        return lambda text: _is_grammar_accept_string(grammar, text)

    @staticmethod
    def _call(name, args):
        return _TC_OPEN + "\n" + json.dumps({"name": name, "arguments": args}) + "\n" + _TC_CLOSE

    def test_grammar_accepts_only_template_format_calls(self):
        accepts = self._accepts()
        for name, args in (("get_time", {}), ("list_dir", {"path": "/home/user"}),
                           ("find_file", {"name": "a_1.txt"}), ("read_file", {"path": 'x "y".txt'}),
                           ("answer", {"text": "09:27"})):
            self.assertTrue(accepts(self._call(name, args)), name)
        for text in (
            json.dumps({"name": "get_time", "arguments": {}}),  # bare JSON
            self._call("rm", {"path": "/"}),  # unknown tool
            self._call("read_file", {}),  # missing parameter
            self._call("answer", {"text": "a\nb"}).replace("\\n", "\n"),  # newline in a string
            "09:27",  # free text
            " " + self._call("get_time", {}),  # leading text
        ):
            self.assertFalse(accepts(text), text)

    def test_grammar_matches_the_template_rendering(self):
        """The grammar's call is byte-identical to what the chat template renders."""
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True)
        except Exception:
            self.skipTest("Qwen3 tokenizer not cached")
        accepts = self._accepts()
        msgs = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                    "name": "read_file", "arguments": {"path": "/home/user/a.txt"}}}]}]
        rendered = apply_chat_template_messages(tok, msgs, TOOLS + [ANSWER_TOOL])
        start = rendered.rindex(_TC_OPEN)
        call = rendered[start: rendered.index(_TC_CLOSE, start) + len(_TC_CLOSE)]
        self.assertTrue(accepts(call), call)

    def test_answer_tool_episode(self):
        task = next(t for t in make_tasks(7, 0) if t.kind == "read_file")
        ep = _os_episode(task, answer_tool=True)
        self.assertIn("answer tool", ep.messages[0]["content"])
        demo_answers = [m for m in ep.messages if any(
            c["function"]["name"] == "answer" for c in m.get("tool_calls", []))]
        self.assertEqual(len(demo_answers), 2)
        path = task.question.split("written in ")[1].split("?")[0]
        ep.step("", [{"name": "read_file", "arguments": {"path": path}}])
        self.assertFalse(ep.terminal)
        ep.step("", [{"name": "answer", "arguments": {"text": task.answer}}])
        self.assertTrue(ep.terminal and ep.answered and ep.correct)
        self.assertEqual((ep.reward, ep.n_calls, ep.trace[-1]["name"]), (1.0, 2, "answer"))
        stats = episode_stats([ep], "os")
        self.assertEqual(stats["os/no_tool_answer_frac"], 0.0)  # it looked first

    def test_answer_without_text_is_wrong(self):
        task = make_tasks(1, 0)[0]
        ep = _os_episode(task, answer_tool=True)
        ep.step("", [{"name": "answer", "arguments": {}}])
        self.assertTrue(ep.terminal)
        self.assertFalse(ep.correct)
        self.assertEqual(ep.n_valid_calls, 0)

    @_async_test
    async def test_agent_sends_the_constraint_and_the_answer_tool(self):
        agent = SimOsAgent(n_tasks=7, n_eval_tasks=3, group_size=2, constrain_calls=True)
        task = next(t for t in agent.tasks if t.kind == "time")
        call = self._call("get_time", {})
        agent._node = _OsCaptureNode([call, self._call("answer", {"text": task.answer})])
        ep = OsEpisode(task=task, group_id=1, group_size=2, episode=0, max_turns=4, **agent.episode_kwargs)
        await agent._run_episode(ep)
        req = agent._node.requests[0]
        self.assertEqual(req.sampling_constraint, {"ebnf": tool_call_grammar(TOOLS + [ANSWER_TOOL])})
        self.assertEqual([t["function"]["name"] for t in req.tools][-1], "answer")
        self.assertTrue(ep.correct)
        self.assertEqual(ep.n_wrapped_calls, 2)
        await agent.evaluate()
        self.assertEqual(agent._node.calls[-1][0], "generate")

    def test_unconstrained_agent_sends_no_constraint(self):
        agent = SimOsAgent(n_tasks=7, n_eval_tasks=0)
        self.assertIsNone(agent.constraint)
        self.assertEqual(agent.tools, TOOLS)


class TestSamplingConstraintPlumbing(unittest.TestCase):
    def test_prompt_batch_split_and_join_keep_the_constraint(self):
        b = PromptBatch(prompts=["a", "b", "c"], sampling_constraint={"regex": "x+"})
        shards = b.split_into(2)
        self.assertTrue(all(sh.sampling_constraint == {"regex": "x+"} for sh in shards))
        self.assertEqual(PromptBatch.join(shards).sampling_constraint, {"regex": "x+"})
        self.assertEqual(_from_dict(PromptBatch, _to_dict(b)).sampling_constraint, {"regex": "x+"})

    def test_base_node_rejects_and_agent_node_passes_constraints(self):
        base = object.__new__(RolloutNode)
        self.assertEqual(base._extra_sampling_params(PromptBatch(prompts=["a"])), {})
        with self.assertRaises(ValueError):
            base._extra_sampling_params(PromptBatch(prompts=["a"], sampling_constraint={"ebnf": "x"}))
        node = object.__new__(AgentRolloutNode)
        batch = PromptBatch(prompts=["a"], sampling_constraint={"ebnf": "root ::= \"a\""})
        self.assertEqual(node._extra_sampling_params(batch), {"ebnf": "root ::= \"a\""})
        for bad in ({"grammar": "x"}, {"ebnf": "x", "regex": "y"}):
            with self.assertRaises(ValueError):
                node._extra_sampling_params(PromptBatch(prompts=["a"], sampling_constraint=bad))

    @_async_test
    async def test_proxy_forwards_the_constraint_to_the_backend(self):
        async with _AgentCluster(_proxy_config()) as cluster:
            req = ExternalGenerateRequest(
                prompts=["p"],
                conversations=[[{"role": "user", "content": "p"}]],
                sampling_constraint={"ebnf": 'root ::= "c"'},
            )
            await cluster.client.call("rollout", "generate_external", _to_dict(req))
            self.assertEqual(cluster.backend.batches[-1].sampling_constraint, {"ebnf": 'root ::= "c"'})


class _OsCaptureNode:
    """The agent's Node: answers generate_external from a script, records calls."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.requests: list[ExternalGenerateRequest] = []
        self.calls: list[tuple[str, dict]] = []
        self._n = 0

    async def call_typed(self, target, endpoint, returns, request):
        if endpoint == "generate":
            n = len(request.prompts)
            self.calls.append((endpoint, {"n": n, "log_prefix": request.log_prefix}))
            return RolloutResult(
                completions=["no idea"] * n,
                token_ids=[[1]] * n,
                prompt_token_ids=[[1]] * n,
                tool_calls=[[] for _ in range(n)],
            )
        self.requests.append(request)
        self._n += 1
        return ExternalGenerateResult(
            rollout_ids=[f"r{self._n}"], completions=[self.replies.pop(0)], tool_calls=[[]]
        )

    async def call(self, target, endpoint, payload):
        self.calls.append((endpoint, payload))
        return {"ok": True}


class TestSimOsAgentRequests(unittest.TestCase):
    @_async_test
    async def test_episode_requests_touch_and_one_report(self):
        agent = SimOsAgent(n_tasks=7, n_eval_tasks=0, group_size=2, tasks_per_wave=1)
        task = next(t for t in agent.tasks if t.kind == "time")
        agent._node = _OsCaptureNode(['{"name": "get_time", "arguments": {}}', task.answer])
        ep = OsEpisode(task=task, group_id=3, group_size=2, episode=1, max_turns=4)
        await agent._run_episode(ep)
        reqs = agent._node.requests
        self.assertEqual(len(reqs), 2)
        for i, req in enumerate(reqs, start=1):
            self.assertEqual(req.group_size, 1)
            self.assertEqual(req.tools, TOOLS)
            self.assertEqual(len(req.conversations), 1)
            self.assertEqual(req.metadata["ww_tag"]["turn"], i)
            self.assertEqual(req.metadata["ww_tag"]["group_size"], 2)
        self.assertEqual(reqs[1].conversations[0][-1]["role"], "tool")
        endpoints = [e for e, _ in agent._node.calls]
        self.assertEqual(endpoints, ["touch", "report_reward"])
        report = _from_dict(ReportRewardRequest, agent._node.calls[-1][1])
        self.assertEqual((list(report.rollout_ids), list(report.rewards)), (["r1", "r2"], [1.0, 1.0]))

    @_async_test
    async def test_evaluation_uses_internal_generate(self):
        agent = SimOsAgent(n_tasks=7, n_eval_tasks=5, group_size=2)
        agent._node = _OsCaptureNode([])
        stats = await agent.evaluate()
        self.assertEqual(agent._node.calls, [("generate", {"n": 5, "log_prefix": "val/os_tools"})])
        self.assertEqual(agent._node.requests, [])  # nothing served through the buffer
        self.assertEqual(stats["os_eval/success"], 0.0)

    def test_task_pool_cycles_before_repeating(self):
        agent = SimOsAgent(n_tasks=10, n_eval_tasks=0, tasks_per_wave=4)
        seen = [t.task_id for _ in range(5) for t in agent._next_tasks()]
        self.assertEqual(len(set(seen[:10])), 10)
        self.assertEqual(len(set(seen[10:20])), 10)

    def test_eval_seed_must_differ(self):
        with self.assertRaises(ValueError):
            SimOsAgent(seed=3, eval_seed=3)



class TestAgenticConfigs(unittest.TestCase):
    """The shipped agentic YAMLs load with their "$ENV:default" defaults."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in _CONFIG_ENV:
            os.environ.pop(var, None)

    def test_agentic_training_yaml(self):
        cfg = AgenticTrainingConfig.from_yaml(CONFIGS / "training" / "agentic_reflect_gsm8k.yaml")
        self.assertIsInstance(cfg.rollout, AgentRolloutConfig)
        self.assertIsInstance(cfg.reward, RewardConfig)
        self.assertIsInstance(cfg.model, ModelConfig)
        self.assertEqual(cfg.rollout.n_rpc_workers, 64)
        self.assertEqual(cfg.total_steps, 40)
        self.assertEqual((cfg.min_train_batch, cfg.max_train_batch), (256, 320))
        self.assertEqual(cfg.max_weight_version_lag, 8)
        self.assertTrue(cfg.group_episodes)
        self.assertEqual(cfg.group_complete_timeout_s, 900.0)
        self.assertFalse(cfg.engine_sleep_during_update)
        self.assertEqual(cfg.rollout_per_prompt, 4)

    def test_agentic_training_yaml_env_override(self):
        os.environ["WW_ASYNC_TOTAL_STEPS"] = "7"
        cfg = AgenticTrainingConfig.from_yaml(CONFIGS / "training" / "agentic_reflect_gsm8k.yaml")
        self.assertEqual(cfg.total_steps, 7)

    def test_async_loop_rejects_the_agentic_yaml(self):
        with self.assertRaisesRegex(ValueError, "group_episodes"):
            AsyncTrainingConfig.from_yaml(CONFIGS / "training" / "agentic_reflect_gsm8k.yaml")

    def test_agentic_proxy_yaml(self):
        path = CONFIGS / "rollout" / "agentic_proxy.yaml"
        cfg = AgentProxyConfig.from_yaml(path)
        self.assertEqual((cfg.node_id, cfg.backend), ("rollout", "rollout-engine"))
        self.assertEqual(cfg.backend_wait_timeout_s, 600.0)
        self.assertEqual(cfg.n_control_workers, 2)
        self.assertEqual(cfg.n_rpc_workers, 128)
        self.assertEqual(cfg.buffer_max_entries, 8192)
        self.assertEqual(cfg.reward_ttl_s, 600.0)
        with self.assertRaisesRegex(ValueError, "backend_wait_timeout_s"):
            RolloutProxyConfig.from_yaml(path)

    def test_agentic_training_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            AgentRolloutConfig(model_path="m", n_rpc_workers=8, tool_call_parser="hermes").save_yaml(
                os.path.join(tmp, "rollout.yaml")
            )
            RewardConfig(reward_type="accuracy").save_yaml(os.path.join(tmp, "reward.yaml"))
            ModelConfig(model_path="m").save_yaml(os.path.join(tmp, "model.yaml"))
            path = os.path.join(tmp, "training.yaml")
            AgenticTrainingConfig(
                rollout_yaml="rollout.yaml",
                reward_yaml="reward.yaml",
                model_yaml="model.yaml",
                group_episodes=False,
                group_complete_timeout_s=30.0,
                engine_sleep_during_update=True,
                pause_drain_timeout_s=45.0,
            ).save_yaml(path)
            cfg = AgenticTrainingConfig.from_yaml(path)
            self.assertFalse(cfg.group_episodes)
            self.assertEqual(cfg.group_complete_timeout_s, 30.0)
            self.assertTrue(cfg.engine_sleep_during_update)
            self.assertEqual(cfg.pause_drain_timeout_s, 45.0)
            self.assertIsInstance(cfg.rollout, AgentRolloutConfig)
            self.assertEqual((cfg.rollout.n_rpc_workers, cfg.rollout.tool_call_parser), (8, "hermes"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
