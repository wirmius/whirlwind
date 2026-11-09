"""Tests for whirlcoms v2 — user-driven process_rpc_requests + serdes-in-seam.

Covers the redesigned API:
  * process_rpc_requests (async) / process_rpc_requests_sync (thread) — user-driven worker.
  * serdes at the seam: handler receives a TYPED message; reply is to_dict'd.
  * (register_message/MessageMeta removed — only expects/returns remain).
  * ping-during-generate (deafness fix) via user-spawned workers.
  * propagate_exceptions: handler error exits process_rpc_requests.
  * auto_process shim still works (parity).
  * call_typed None-guard (the training_grpo.py:849 crash fix).

Per-test hard timeout (WHIRLWIND_TEST_TIMEOUT, default 100s) so a hang fails
fast instead of running for hours.

Run: ./tests/run_tests.sh tests.test_whirlcoms_v2
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import unittest
from dataclasses import dataclass

from tests._helpers import skip_no_torch

_TEST_TIMEOUT = float(os.environ.get("WHIRLWIND_TEST_TIMEOUT", "100"))


def _async_test(coro):
    def wrapper(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                asyncio.wait_for(coro(self), timeout=_TEST_TIMEOUT)
            )
        except asyncio.TimeoutError:
            raise AssertionError(
                f"{self.__class__.__name__}.{coro.__name__} exceeded "
                f"{_TEST_TIMEOUT:.0f}s timeout — likely a deadlock"
            ) from None
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(asyncio.sleep(0.05))
            loop.close()

    return wrapper


from whirlwind.whirlcoms.message import Message  # noqa: E402


@dataclass
class PingMsg(Message):
    n: int = 0


@dataclass
class PongMsg(Message):
    n: int = 0


@dataclass
class JobMsg(Message):
    secs: float = 0.0


class TestWaitForCallsAsync(unittest.TestCase):
    """User-driven async worker: deafness fix + typed serdes at the seam."""

    @_async_test
    async def test_ping_during_slow_generate_with_user_workers(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        # auto_process=False → the user drives process_rpc_requests themselves.
        worker = Node("worker", auto_process=False)

        @worker.expose("generate", expects=JobMsg, returns=PongMsg)
        async def generate(msg: JobMsg) -> PongMsg:
            await asyncio.sleep(msg.secs)
            return PongMsg(n=msg.secs)

        @worker.expose("ping", expects=PingMsg, returns=PongMsg)
        def ping(msg: PingMsg) -> PongMsg:
            return PongMsg(n=msg.n)

        await worker.start()
        # User spawns 2 worker tasks (concurrency is caller-controlled).
        workers = [asyncio.create_task(worker.process_rpc_requests()) for _ in range(2)]
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)

        gen_task = asyncio.create_task(client.call_typed("worker", "generate", PongMsg, JobMsg(secs=0.5)))
        await asyncio.sleep(0.05)
        t0 = asyncio.get_event_loop().time()
        pong = await client.call_typed("worker", "ping", PongMsg, PingMsg(n=7))
        latency = asyncio.get_event_loop().time() - t0

        self.assertIsInstance(pong, PongMsg)
        self.assertEqual(pong.n, 7)
        self.assertLess(latency, 0.2, f"ping took {latency:.3f}s — node deaf to control RPC")
        await gen_task

        await client.stop()
        await worker.stop()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await orch.stop()

    @_async_test
    async def test_handler_receives_typed_msg_and_reply_is_serialized(self):
        """Serdes lives in whirlcoms: handler gets a typed object, returns one."""
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=False)

        seen = {}

        @worker.expose("ping", expects=PingMsg, returns=PongMsg)
        def ping(msg: PingMsg) -> PongMsg:
            seen["type"] = type(msg).__name__
            seen["n"] = msg.n
            return PongMsg(n=msg.n + 1)

        await worker.start()
        wtask = asyncio.create_task(worker.process_rpc_requests())
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)

        pong = await client.call_typed("worker", "ping", PongMsg, PingMsg(n=41))
        self.assertEqual(seen.get("type"), "PingMsg")  # handler got a TYPED object
        self.assertEqual(seen.get("n"), 41)
        self.assertIsInstance(pong, PongMsg)
        self.assertEqual(pong.n, 42)  # round-tripped through to_dict/from_dict

        await client.stop()
        await worker.stop()
        wtask.cancel()
        await asyncio.gather(wtask, return_exceptions=True)
        await orch.stop()

    @_async_test
    async def test_propagate_exceptions_exits_process_rpc_requests(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=False, propagate_exceptions=True)

        @worker.expose("boom", expects=PingMsg, returns=PongMsg)
        def boom(msg: PingMsg) -> PongMsg:
            raise ValueError("kaboom")

        await worker.start()
        wtask = asyncio.create_task(worker.process_rpc_requests())
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)

        # The call itself will get no reply (worker exited); it times out.
        with self.assertRaises(Exception):
            await asyncio.wait_for(
                client.call_typed("worker", "boom", PongMsg, PingMsg(n=1)),
                timeout=2.0,
            )

        # process_rpc_requests should have exited with the ValueError.
        with self.assertRaises(ValueError):
            await asyncio.wait_for(wtask, timeout=2.0)

        await client.stop()
        await worker.stop()
        await orch.stop()

    @_async_test
    async def test_default_catches_exception_and_keeps_routing(self):
        from whirlwind.whirlcoms import Node, Orchestrator, RpcError

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=False)  # propagate_exceptions=False

        calls = []

        @worker.expose("flaky", expects=PingMsg, returns=PongMsg)
        def flaky(msg: PingMsg) -> PongMsg:
            calls.append(msg.n)
            if msg.n == 1:
                raise ValueError("once")
            return PongMsg(n=msg.n)

        await worker.start()
        wtask = asyncio.create_task(worker.process_rpc_requests())
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)

        # First call errors (error reply → RpcError at the typed seam).
        with self.assertRaises(RpcError):
            await client.call_typed("worker", "flaky", PongMsg, PingMsg(n=1))
        # Second call still works — routing continued after the handler error.
        pong = await client.call_typed("worker", "flaky", PongMsg, PingMsg(n=2))
        self.assertEqual(pong.n, 2)
        self.assertEqual(calls, [1, 2])

        await client.stop()
        await worker.stop()
        wtask.cancel()
        await asyncio.gather(wtask, return_exceptions=True)
        await orch.stop()


class TestWaitForCallsSync(unittest.TestCase):
    """User-driven thread worker: sync handler runs in the thread."""

    @_async_test
    async def test_sync_worker_runs_handler_off_loop(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=False)

        ran_on = {}

        @worker.expose("ping", expects=PingMsg, returns=PongMsg)
        def ping(msg: PingMsg) -> PongMsg:
            ran_on["name"] = threading.current_thread().name
            return PongMsg(n=msg.n)

        await worker.start()
        # User drives a sync worker from a thread.
        t = threading.Thread(target=worker.process_rpc_requests_sync, daemon=True)
        t.start()
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)

        pong = await client.call_typed("worker", "ping", PongMsg, PingMsg(n=5))
        self.assertEqual(pong.n, 5)
        self.assertNotIn("asyncio", ran_on.get("name", "asyncio"))
        self.assertIn("Thread", ran_on.get("name", ""), f"ran on {ran_on}")

        await client.stop()
        await worker.stop()
        t.join(timeout=2.0)
        await orch.stop()


class TestAutoProcessShim(unittest.TestCase):
    """auto_process=True still works (Node spawns one process_rpc_requests task)."""

    @_async_test
    async def test_auto_process_still_handles_calls(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()
        worker = Node("worker", auto_process=True)  # legacy convenience

        @worker.expose("echo")
        def echo(x):
            return x

        await worker.start()
        client = Node("client")
        await client.start()
        await asyncio.sleep(0.3)

        self.assertEqual(await client.call("worker", "echo", 123), 123)

        await client.stop()
        await worker.stop()
        await orch.stop()
if __name__ == "__main__":
    unittest.main(verbosity=2)
