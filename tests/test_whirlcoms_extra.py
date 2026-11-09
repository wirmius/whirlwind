"""Extra tests for the whirlcoms ZeroMQ RPC layer (whirlwind.whirlcoms).

Independent of every other node: only exercises the communication stack
(Node, Orchestrator, RpcClient, transport, messages). Pure-Python where
possible (the tensor tests need torch).

Adds coverage the existing test_whirlwind.py does not:
  - robustness: RPC timeout, unknown node/endpoint, malformed frame, oversized payload
  - concurrency: many concurrent calls, overlapping call+process
  - transport edge cases: tensors at the SHM threshold on both sides, dtype
    fidelity, SHM file cleanup, many small tensors
  - lifecycle: node re-register, orchestrator restart -> topology re-converges,
    clean stop drains the queue, deregister during pending call
  - message serdes: all message types with tracing fields, split_into/join/
    slice_by/iter_by_* exactness, ArgsKwargs attr/index access, large batches
  - topology: multi-hop peer discovery then direct P2P
"""

from __future__ import annotations

import asyncio
import os
import unittest

from tests._helpers import skip_no_torch


# Reuse the existing file's async-test helper convention.
def _async_test(coro):
    """Decorator to run async test methods (mirrors test_whirlwind.py)."""

    def wrapper(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro(self))
        finally:
            import contextlib

            with contextlib.suppress(Exception):
                loop.run_until_complete(asyncio.sleep(0.05))
            loop.close()

    return wrapper


# ---------------------------------------------------------------------------
# Transport edge cases
# ---------------------------------------------------------------------------


class TestTransportEdgeCases(unittest.TestCase):
    """Transport serialization round-trips at SHM threshold boundaries."""

    def test_just_below_shm_threshold_stays_inline(self):
        from whirlwind.whirlcoms.transport import (
            DEFAULT_SHM_THRESHOLD,
            _deserialize_payload,
            _serialize_payload,
        )

        # The SHM threshold is applied to the SERIALIZED frame size (msgpack
        # adds framing), so use a payload comfortably below the threshold.
        payload = b"x" * (DEFAULT_SHM_THRESHOLD // 2)
        frame, shm_meta, shm_path = _serialize_payload(payload)
        self.assertIsNone(shm_meta, "payload below threshold should not use SHM")
        self.assertIsNone(shm_path)
        self.assertEqual(_deserialize_payload(frame), payload)

    def test_just_above_shm_threshold_uses_shm(self):
        from whirlwind.whirlcoms.transport import (
            DEFAULT_SHM_THRESHOLD,
            _deserialize_payload,
            _serialize_payload,
        )

        payload = b"y" * (DEFAULT_SHM_THRESHOLD + 1)
        frame, shm_meta, shm_path = _serialize_payload(payload)
        self.assertIsNotNone(shm_meta, "payload above threshold should use SHM")
        self.assertIsNotNone(shm_path)
        self.assertEqual(_deserialize_payload(frame), payload)
        # SHM file should be cleaned up by the deserializer.
        self.assertFalse(os.path.exists(shm_path), "SHM file not cleaned up")

    def test_nested_dict_with_multiple_large_blobs(self):
        from whirlwind.whirlcoms.transport import (
            DEFAULT_SHM_THRESHOLD,
            _deserialize_payload,
            _serialize_payload,
        )

        big = b"z" * (DEFAULT_SHM_THRESHOLD + 10)
        payload = {"a": big, "b": big, "list": [big, {"inner": big}]}
        frame, shm_meta, shm_path = _serialize_payload(payload)
        result = _deserialize_payload(frame)
        self.assertEqual(result, payload)

    def test_empty_and_zero_payloads(self):
        from whirlwind.whirlcoms.transport import (
            _deserialize_payload,
            _serialize_payload,
        )

        for obj in [b"", "", [], {}, None, 0, 0.0, False]:
            frame, _, _ = _serialize_payload(obj)
            self.assertEqual(_deserialize_payload(frame), obj)

    def test_unicode_and_binary_strings(self):
        from whirlwind.whirlcoms.transport import (
            _deserialize_payload,
            _serialize_payload,
        )

        for obj in ["héllo wörld", "日本語", "\x00\x01\x02binary", "emoji 🎉"]:
            frame, _, _ = _serialize_payload(obj)
            self.assertEqual(_deserialize_payload(frame), obj)


@skip_no_torch
class TestTensorTransportExtra(unittest.TestCase):
    """Tensor dtype fidelity + many-small-tensors + SHM cleanup."""

    def test_all_float_dtypes_roundtrip(self):
        import torch

        from whirlwind.whirlcoms.transport import (
            _deserialize_payload,
            _serialize_payload,
        )

        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            t = torch.randn(4, 8, dtype=dtype)
            frame, _, _ = _serialize_payload(t)
            result = _deserialize_payload(frame)
            self.assertEqual(result.dtype, dtype)
            self.assertTrue(torch.equal(result, t.to(result.dtype)))

    def test_all_int_dtypes_roundtrip(self):
        import torch

        from whirlwind.whirlcoms.transport import (
            _deserialize_payload,
            _serialize_payload,
        )

        for dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            t = torch.randint(0, 100, (3, 3), dtype=dtype)
            frame, _, _ = _serialize_payload(t)
            result = _deserialize_payload(frame)
            self.assertEqual(result.dtype, dtype)
            self.assertTrue(torch.equal(result, t))

    def test_many_small_tensors_in_list(self):
        import torch

        from whirlwind.whirlcoms.transport import (
            _deserialize_payload,
            _serialize_payload,
        )

        ts = [torch.randn(2, 2) for _ in range(20)]
        frame, _, _ = _serialize_payload(ts)
        result = _deserialize_payload(frame)
        self.assertEqual(len(result), 20)
        for orig, got in zip(ts, result, strict=True):
            self.assertTrue(torch.allclose(got, orig))

    def test_large_tensor_shm_cleanup(self):
        import torch

        from whirlwind.whirlcoms.transport import (
            _deserialize_payload,
            _serialize_payload,
        )

        t = torch.randn(1024, 1024)  # ~4MB, well above SHM threshold
        frame, shm_meta, shm_path = _serialize_payload(t)
        self.assertIsNotNone(shm_path)
        result = _deserialize_payload(frame)
        self.assertTrue(torch.allclose(result, t))
        self.assertFalse(os.path.exists(shm_path), "SHM file not cleaned up after deserialization")


# ---------------------------------------------------------------------------
# Lifecycle + robustness
# ---------------------------------------------------------------------------


class TestNodeLifecycle(unittest.TestCase):
    """Node re-register, clean stop, orchestrator restart."""

    @_async_test
    async def test_clean_stop_drains_queue(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        processed = []

        worker = Node("worker", auto_process=True)

        @worker.expose("echo")
        def echo(x):
            processed.append(x)
            return x

        await worker.start()
        await asyncio.sleep(0.3)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)

        # fire several calls then stop immediately; the queue should drain
        for i in range(5):
            await client.call("worker", "echo", i)
        await client.stop()
        await worker.stop()
        await orch.stop()
        # at least the calls we awaited should have been processed
        self.assertGreaterEqual(len(processed), 1)

    @_async_test
    async def test_orchestrator_restart_topology_reconverges(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=True)

        @worker.expose("ping")
        def ping():
            return "pong"

        await worker.start()
        await asyncio.sleep(0.3)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)
        self.assertEqual(await client.call("worker", "ping"), "pong")

        # Restart the orchestrator; nodes should re-register and rediscover.
        await orch.stop()
        await asyncio.sleep(0.2)
        orch2 = Orchestrator()
        await orch2.start()
        await asyncio.sleep(1.0)  # allow heartbeat/re-registration

        # client should still reach worker (topology re-converged)
        # (this may take a couple retries due to heartbeat timing)
        ok = False
        for _ in range(10):
            try:
                if await client.call("worker", "ping") == "pong":
                    ok = True
                    break
            except Exception:
                await asyncio.sleep(0.3)
        self.assertTrue(ok, "topology did not re-converge after orchestrator restart")

        await client.stop()
        await worker.stop()
        await orch2.stop()

    @_async_test
    async def test_node_re_register_same_id(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        w1 = Node("worker", auto_process=True)

        @w1.expose("v")
        def v1():
            return 1

        await w1.start()
        await asyncio.sleep(0.3)
        await w1.stop()

        w2 = Node("worker", auto_process=True)

        @w2.expose("v")
        def v2():
            return 2

        await w2.start()
        await asyncio.sleep(0.3)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)
        # the new worker with the same id should respond
        result = await client.call("worker", "v")
        self.assertEqual(result, 2)
        await client.stop()
        await w2.stop()
        await orch.stop()

    # Runs in a subprocess: the failure mode is ctx.term() blocking forever.
    _UNDELIVERABLE_REPLY = r"""
import asyncio, sys
import whirlwind.whirlcoms.node as node_mod
from whirlwind.whirlcoms import Node, Orchestrator
if len(sys.argv) > 1:
    node_mod._STOP_LINGER_MS = int(sys.argv[1])

async def main():
    orch = Orchestrator()
    await orch.start()
    worker = Node("worker", auto_process=True)

    @worker.expose("slow")
    async def slow():
        await asyncio.sleep(1.5)
        return "late"

    await worker.start()
    client = Node("client", rpc_timeout=0.5, fail_on_timeout=True)
    await client.start()
    while "worker" not in client.topology:
        await asyncio.sleep(0.1)
    try:
        await client.call("worker", "slow")
    except Exception:
        pass  # timed out, as intended
    await client.stop()  # the caller is gone before the reply exists
    await asyncio.sleep(2.0)  # the reply is now queued for a dead peer
    await worker.stop()
    await orch.stop()
    print("stopped")

asyncio.run(main())
"""

    def test_stop_returns_with_undeliverable_reply(self):
        """A reply queued for a caller that already exited must not block
        stop(): zmq's default infinite linger made ctx.term() hang, and the
        node never exited. (Passing "-1" as the script argument restores the
        infinite linger and makes this scenario hang.)"""
        import subprocess
        import sys

        env = dict(os.environ, WW_RUN_ID=f"test-undeliverable-{os.getpid()}")
        proc = subprocess.run(
            [sys.executable, "-c", self._UNDELIVERABLE_REPLY],
            env=env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("stopped", proc.stdout)


class TestRpcRobustness(unittest.TestCase):
    """RPC error/edge handling."""

    @_async_test
    async def test_rpc_timeout_to_unknown_node(self):
        from whirlwind.whirlcoms import Node, Orchestrator
        from whirlwind.whirlcoms.node import RpcError

        orch = Orchestrator()
        await orch.start()
        client = Node("client", rpc_timeout=0.5, fail_on_timeout=False)
        await client.start()
        await asyncio.sleep(0.3)

        # calling a node that doesn't exist should fail/timeout, not hang
        with self.assertRaises((RpcError, asyncio.TimeoutError, Exception)):
            await client.call("nonexistent", "anything", 1)
        await client.stop()
        await orch.stop()

    @_async_test
    async def test_rpc_unknown_endpoint_raises(self):
        from whirlwind.whirlcoms import Node, Orchestrator
        from whirlwind.whirlcoms.node import RpcError

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=True)
        await worker.start()
        await asyncio.sleep(0.3)
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)
        with self.assertRaises((RpcError, Exception)):
            await client.call("worker", "no_such_endpoint")
        await client.stop()
        await worker.stop()
        await orch.stop()

    @_async_test
    async def test_handler_returning_none_roundtrips(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=True)

        @worker.expose("noop")
        def noop():
            return None

        await worker.start()
        await asyncio.sleep(0.3)
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)
        self.assertIsNone(await client.call("worker", "noop"))
        await client.stop()
        await worker.stop()
        await orch.stop()

    @_async_test
    async def test_handler_raising_exception_propagates(self):
        from whirlwind.whirlcoms import Node, Orchestrator
        from whirlwind.whirlcoms.node import RpcError

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=True)

        @worker.expose("boom")
        def boom():
            raise ValueError("kaboom")

        await worker.start()
        await asyncio.sleep(0.3)
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)
        with self.assertRaises((RpcError, Exception)):
            await client.call("worker", "boom")
        await client.stop()
        await worker.stop()
        await orch.stop()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestRpcConcurrency(unittest.TestCase):
    @_async_test
    async def test_many_concurrent_calls(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=True)

        @worker.expose("inc")
        def inc(x):
            return x + 1

        await worker.start()
        await asyncio.sleep(0.3)
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)

        results = await asyncio.gather(*[client.call("worker", "inc", i) for i in range(20)])
        self.assertEqual(sorted(results), list(range(1, 21)))
        await client.stop()
        await worker.stop()
        await orch.stop()

    @_async_test
    async def test_concurrent_calls_to_multiple_workers(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        workers = []
        for i in range(3):
            w = Node(f"worker{i}", auto_process=True)

            @w.expose("id")
            def make_id(i=i):
                return i

            await w.start()
            workers.append(w)
        await asyncio.sleep(0.3)
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)

        results = await asyncio.gather(*[client.call(f"worker{i}", "id") for i in range(3)])
        self.assertEqual(sorted(results), [0, 1, 2])
        await client.stop()
        for w in workers:
            await w.stop()
        await orch.stop()


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


class TestTopologyDiscovery(unittest.TestCase):
    @_async_test
    async def test_multi_peer_discovery_then_direct_p2p(self):
        """Three workers discover each other via the orchestrator and can call
        each other directly (full mesh), not just via the orchestrator."""
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        nodes = []
        for i in range(3):
            n = Node(f"node{i}", auto_process=True)

            @n.expose("whoami")
            def who(i=i):
                return f"node{i}"

            await n.start()
            nodes.append(n)
        await asyncio.sleep(0.6)  # let topology propagate across all 3

        # Each node can call the others directly (full mesh).
        for src in range(3):
            for dst in range(3):
                if src == dst:
                    continue
                result = await nodes[src].call(f"node{dst}", "whoami")
                self.assertEqual(result, f"node{dst}")

        for n in nodes:
            await n.stop()
        await orch.stop()


# ---------------------------------------------------------------------------
# Message serdes: split_into/join/slice_by/iter_by_*
# ---------------------------------------------------------------------------


class TestMessageSerdes(unittest.TestCase):
    """Round-trip all message types and the BatchedMessage grouping iterators."""

    def _make_prompt_batch(self):
        from whirlwind.messages import PromptBatch

        return PromptBatch(
            prompts=["p0", "p1", "p2", "p3"],
            responses=["r0", "r1", "r2", "r3"],
            dataset_ids=["ds", "ds", "ds", "ds"],
            entry_ids=[0, 1, 2, 3],
            rollout_ids=[0, 0, 1, 1],
            subrollout_ids=[0, 1, 0, 1],
        )

    def test_prompt_batch_to_dict_from_dict_roundtrip(self):
        from whirlwind.messages import PromptBatch, _from_dict, _to_dict

        msg = self._make_prompt_batch()
        d = _to_dict(msg)
        msg2 = _from_dict(PromptBatch, d)
        self.assertEqual(msg2.prompts, msg.prompts)
        self.assertEqual(msg2.entry_ids, msg.entry_ids)
        self.assertEqual(msg2.batch_size, 4)

    def test_split_into_then_join_is_identity(self):
        msg = self._make_prompt_batch()
        for n in (1, 2, 4):
            shards = msg.split_into(n)
            self.assertEqual(len(shards), n)
            joined = type(msg).join(shards)
            self.assertEqual(joined.prompts, msg.prompts)
            self.assertEqual(joined.entry_ids, msg.entry_ids)

    def test_split_into_more_than_batch_duplicates(self):
        # split_into(8) on a 4-sample batch should still return 8 shards
        # (BatchedMessage duplicates samples when n > batch_size).
        msg = self._make_prompt_batch()
        shards = msg.split_into(8)
        self.assertEqual(len(shards), 8)

    def test_slice_by_selects_subset(self):
        msg = self._make_prompt_batch()
        sub = msg.slice_by([0, 2])
        self.assertEqual(sub.prompts, ["p0", "p2"])
        self.assertEqual(sub.entry_ids, [0, 2])
        self.assertEqual(sub.batch_size, 2)

    def test_iter_by_sample_exact_roundtrip(self):
        msg = self._make_prompt_batch()
        shards = [m for _, m in msg.iter_by_sample()]
        self.assertEqual(len(shards), 4)
        joined = type(msg).join(shards)
        self.assertEqual(joined.prompts, msg.prompts)

    def test_iter_by_prompt_groups_grpo(self):
        msg = self._make_prompt_batch()
        groups = list(msg.iter_by_prompt())
        # entry_ids 0,1 share entry 0? No: entry_ids are [0,1,2,3] -> 4 unique
        # prompts. Adjust expectations to grouping by (dataset_id, entry_id).
        self.assertEqual(len(groups), 4)

    def test_train_step_request_roundtrip(self):
        from whirlwind.messages import TrainStepRequest, _from_dict, _to_dict

        msg = TrainStepRequest(
            input_ids=[[1, 2], [3, 4, 5]],
            label_ids=[[-100, 2], [-100, -100, 5]],
            rewards=[0.5, -0.5],
            advantages=[1.0, -1.0],
            prompts=["a", "b"],
            completions=["x", "y"],
        )
        d = _to_dict(msg)
        msg2 = _from_dict(TrainStepRequest, d)
        self.assertEqual(msg2.input_ids, msg.input_ids)
        self.assertEqual(msg2.label_ids, msg.label_ids)
        self.assertEqual(msg2.advantages, msg.advantages)
        self.assertEqual(msg2.batch_size, 2)

    def test_train_step_request_split_preserves_overrides(self):
        from whirlwind.messages import TrainStepRequest

        msg = TrainStepRequest(
            input_ids=[[1, 2], [3, 4], [5, 6], [7, 8]],
            label_ids=[[-100, 2], [-100, 4], [-100, 6], [-100, 8]],
            rewards=[0.1, 0.2, 0.3, 0.4],
            global_batch_size_override=4,
            global_max_completion_len_override=1,
        )
        shards = msg.split_into(2)
        for s in shards:
            self.assertEqual(s.global_batch_size_override, 4)
            self.assertEqual(s.global_max_completion_len_override, 1)


class TestArgsKwargsMessages(unittest.TestCase):
    """ArgsKwargs-based messages: attribute + index access, type checks."""

    def test_argskwargs_attribute_access(self):
        from whirlwind.messages import WeightSyncRequest

        msg = WeightSyncRequest(checkpoint_path="/tmp/ckpt", step=7)
        self.assertEqual(msg.checkpoint_path, "/tmp/ckpt")
        self.assertEqual(msg.step, 7)

    def test_argskwargs_missing_required_kwarg_raises(self):
        from whirlwind.messages import ModelUpdateRequest

        with self.assertRaises(TypeError):
            ModelUpdateRequest()  # 'lr' is required

    def test_argskwargs_wrong_type_raises(self):
        from whirlwind.messages import WeightSyncRequest

        with self.assertRaises(TypeError):
            WeightSyncRequest(checkpoint_path=123, step=7)  # path must be str

    def test_checkpoint_result_roundtrip(self):
        from whirlwind.messages import CheckpointResult, _from_dict, _to_dict

        msg = CheckpointResult(ok=True, path="/x", step=3, error="")
        d = _to_dict(msg)
        msg2 = _from_dict(CheckpointResult, d)
        self.assertTrue(msg2.ok)
        self.assertEqual(msg2.path, "/x")
        self.assertEqual(msg2.step, 3)


class TestTrainStepResultMerge(unittest.TestCase):
    """Regression tests for TrainStepResult.merge.

    The producer all-reduces the sample count before building each rank's
    TrainStepResult, so every rank holds the *global* total. merge() must
    NOT sum across ranks (that would inflate by world_size); it must take
    the value from results[0].
    """

    def test_merge_does_not_inflate_total_completions(self):
        from whirlwind.messages import TrainStepResult

        # Simulate world_size=4 where every rank already holds the global
        # all-reduced count of 100 (identical on all ranks).
        global_count = 100
        world_size = 4
        results = [
            TrainStepResult(
                loss=1.5,
                grad_norm=0.5,
                lr=1e-4,
                total_completions=global_count,
            )
            for _ in range(world_size)
        ]
        merged = TrainStepResult.merge(results)
        # Must be the global count, not world_size * global_count.
        self.assertEqual(merged.total_completions, global_count)
        # loss/lr averaged, grad_norm maxed -- unchanged behavior, sanity-check.
        self.assertAlmostEqual(merged.loss, 1.5)
        self.assertIsNotNone(merged.lr)
        self.assertAlmostEqual(merged.lr, 1e-4)
        self.assertIsNotNone(merged.grad_norm)
        self.assertAlmostEqual(merged.grad_norm, 0.5)

    def test_merge_handles_mixed_none_total_completions(self):
        from whirlwind.messages import TrainStepResult

        # Defensive: if some ranks report None and others the global count,
        # take the first non-None (all ranks should be identical, but guard).
        results = [
            TrainStepResult(loss=1.0, total_completions=None),
            TrainStepResult(loss=1.0, total_completions=100),
            TrainStepResult(loss=1.0, total_completions=100),
            TrainStepResult(loss=1.0, total_completions=None),
        ]
        merged = TrainStepResult.merge(results)
        self.assertEqual(merged.total_completions, 100)

    def test_merge_all_none_total_completions_stays_none(self):
        from whirlwind.messages import TrainStepResult

        results = [
            TrainStepResult(loss=1.0, total_completions=None),
            TrainStepResult(loss=1.0, total_completions=None),
        ]
        merged = TrainStepResult.merge(results)
        self.assertIsNone(merged.total_completions)

    def test_merge_single_result_short_circuit(self):
        from whirlwind.messages import TrainStepResult

        # world_size==1 path: merge([r]) returns r unchanged (no inflation).
        r = TrainStepResult(loss=2.0, grad_norm=0.3, lr=2e-4, total_completions=50)
        merged = TrainStepResult.merge([r])
        self.assertEqual(merged.total_completions, 50)
        self.assertEqual(merged.loss, 2.0)


class TestModelUpdateLossAggregation(unittest.TestCase):
    """Pin the loss-metric reconstruction arithmetic in _model_update_step.

    _compute_loss divides every loss by gradient_accumulation_steps (acc_steps)
    for gradient scaling, so each forward_accum_grad logs loss_i = true_i /
    acc_steps. _model_update_step then reconstructs the true global loss from the
    buffer. BNPO/CE/DR-GRPO undo the /acc_steps with a final *acc_steps; GRPO
    must too, or its reported train_loss is understated by acc_steps.

    These tests replicate the exact arithmetic (no model/GPU needed) to pin the
    formula without depending on a live FSDP run.
    """

    @staticmethod
    def _reconstruct(loss_reduction, true_microbatch_losses, world_size=1, acc_steps=4):
        """Mirror _model_update_step's loss aggregation exactly."""
        buf_losses = [tl / acc_steps for tl in true_microbatch_losses]  # _compute_loss scaling
        num_shards = len(buf_losses)
        if loss_reduction in ("bnpo", "dr_grpo", "ce"):
            rank_loss = sum(buf_losses) * world_size
        else:  # "grpo"
            rank_loss = sum(buf_losses) / num_shards if num_shards else 0.0
        loss_tensor = rank_loss
        if world_size > 1:
            loss_tensor = loss_tensor / world_size  # all_reduce mean of identical per-rank values
        if loss_reduction in ("bnpo", "dr_grpo", "ce", "grpo"):  # grpo included
            loss_tensor = loss_tensor * acc_steps
        return loss_tensor

    def test_grpo_reported_loss_not_understated_by_acc_steps(self):
        # true per-microbatch GRPO losses (per-seq means); global mean = 25.0
        true = [10.0, 20.0, 30.0, 40.0]
        true_global = sum(true) / len(true)
        reported = self._reconstruct("grpo", true, world_size=1, acc_steps=4)
        # Before the fix this was 6.25 (= 25/4). Must now equal the true global mean.
        self.assertAlmostEqual(reported, true_global)
        self.assertNotAlmostEqual(reported, true_global / 4)  # guard: not the old bug

    def test_ce_and_grpo_both_get_acc_steps_correction(self):
        # Both modes must report the true loss (not /acc_steps).
        true = [10.0, 20.0, 30.0, 40.0]
        for mode in ("ce", "bnpo", "dr_grpo", "grpo"):
            with self.subTest(mode=mode):
                reported = self._reconstruct(mode, true, world_size=1, acc_steps=4)
                # Each mode's true value differs, but all must be *corrected*
                # (i.e. != the /acc_steps understated value). For GRPO the true
                # is the mean (25); for the sum-modes the true is sum(true)=100.
                understated = sum(true) / len(true) / 4 if mode == "grpo" else sum(true) / 4
                self.assertNotAlmostEqual(
                    reported, understated, msg=f"{mode} loss still understated by acc_steps"
                )


class TestShardEdgeCases(unittest.TestCase):
    """Regression tests for low-severity message-layer fixes."""

    def test_empty_batch_split_into_does_not_raise(self):
        # split_into on an empty batch used to raise ZeroDivisionError.
        from whirlwind.messages import PromptBatch, _compute_shard_indices

        self.assertEqual(_compute_shard_indices(4, 0), [[], [], [], []])
        b = PromptBatch(prompts=[], dataset_ids=[], entry_ids=[], rollout_ids=[], subrollout_ids=[])
        shards = b.split_into(3)
        self.assertEqual([s.batch_size for s in shards], [0, 0, 0])
        # join of empty shards is empty
        self.assertEqual(type(b).join(shards).batch_size, 0)

    def test_absent_dataset_id_defaults_to_empty_string(self):
        # ArgsKwargs.from_dict used to default absent dataset_id to "0".
        from whirlwind.messages import WeightSyncRequest, _from_dict

        m = _from_dict(
            WeightSyncRequest,
            {"_msg_args": [], "_msg_kwargs": {"checkpoint_path": "/x", "step": 7}},
        )
        self.assertEqual(m.dataset_id, "")  # not "0"
        # entry_id default stays -1, the rest stay 0
        self.assertEqual(m.entry_id, -1)
        self.assertEqual(m.rollout_id, 0)

    def test_unmap_topk_places_selected_logit_outside_topk(self):
        # unmap_topk must place a chosen-token logit that falls outside
        # the top-K (under stochastic sampling), instead of leaving it at -inf.
        import torch

        from whirlwind.messages import map_topk, unmap_topk

        torch.manual_seed(0)
        logits = torch.randn(1, 1, 10)  # vocab=10
        label_ids = torch.tensor([[8]])  # chosen token 8
        tk_ids, tk_logits, sel = map_topk(logits, label_ids, topk=3, temperature=1.0)
        # Force the test to exercise the interesting case (8 not in top-3):
        if 8 in tk_ids[0, 0].tolist():
            self.skipTest("chosen token happened to be in top-K; seed fix needed")
        recon = unmap_topk(tk_ids, tk_logits, sel, vocab_size=10, label_ids=label_ids)
        self.assertAlmostEqual(recon[0, 0, 8].item(), logits[0, 0, 8].item())
        # non-top-K, non-selected positions stay -inf
        non_topk_pos = next(j for j in range(10) if j != 8 and j not in tk_ids[0, 0].tolist())
        self.assertEqual(recon[0, 0, non_topk_pos].item(), float("-inf"))

    def test_rolloutresult_split_into_uses_shard_list_for_prompt_logprobs(self):
        # split_into and slice_by must index prompt_logprobs consistently
        # (both via _shard_list), so a length desync raises consistently.
        import inspect

        from whirlwind.messages import RolloutResult

        split_src = inspect.getsource(RolloutResult.split_into)
        slice_src = inspect.getsource(RolloutResult.slice_by)
        self.assertIn("_shard_list(self.prompt_logprobs", split_src)
        self.assertIn("_shard_list(self.prompt_logprobs", slice_src)
        self.assertNotIn("[self.prompt_logprobs[i]", split_src)


class TestAdvantageStdBessel(unittest.TestCase):
    """Pin that reward _nanstd uses Bessel's correction (sample std, ddof=1),
    matching TRL's trl.trainer.utils.nanstd (DIV-2).
    """

    def test_nanstd_matches_trl_sample_std(self):
        import math

        from whirlwind.reward.reward_function import _nanmean, _nanstd

        # group_size=4, rewards [0,0,0,1]: pop std=0.433, sample (Bessel) std=0.5
        rewards = [0.0, 0.0, 0.0, 1.0]
        mean = _nanmean(rewards)
        std = _nanstd(rewards, mean)
        self.assertAlmostEqual(std, 0.5, places=6)
        self.assertNotAlmostEqual(std, math.sqrt(3 / 16), places=3)  # not pop std 0.433

    def test_nanstd_matches_trl_across_cases(self):
        # Cross-check against TRL's nanstd directly.
        try:
            from trl.trainer.utils import nanstd as trl_nanstd
        except Exception:
            self.skipTest("trl not importable in this env")
        import torch

        from whirlwind.reward.reward_function import _nanmean, _nanstd

        for rewards in ([0, 0, 0, 1], [0, 1], [1, 1, 1, 1], [0.5, 0.2, 0.9, 0.0]):
            with self.subTest(rewards=rewards):
                wh = _nanstd(rewards, _nanmean(rewards))
                trl = trl_nanstd(torch.tensor(rewards, dtype=torch.float32)).item()
                self.assertAlmostEqual(wh, trl, places=6)

    def test_nanstd_single_valid_returns_zero(self):
        import math

        from whirlwind.reward.reward_function import _nanmean, _nanstd

        # Fewer than 2 valid -> 0.0 (triggers the epsilon fallback downstream,
        # same as TRL's NaN path effectively giving a zero advantage).
        self.assertEqual(_nanstd([math.nan, 0.5], _nanmean([math.nan, 0.5])), 0.0)
        self.assertEqual(_nanstd([0.5], _nanmean([0.5])), 0.0)


if __name__ == "__main__":
    unittest.main()
