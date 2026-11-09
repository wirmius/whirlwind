"""Teardown robustness tests: node heartbeat fuse, trainer stop() survival.

The live 65-step runs exposed three teardown failure modes:

1. A node whose orchestrator is gone logs "heartbeat missed" forever and
   never exits (service nodes + sglang workers hung for hours).
2. ``TrainingOrchestrator.stop()`` crashed on ``CancelledError`` (it awaited
   the just-cancelled failure-monitor task under ``suppress(Exception)``,
   which does not catch BaseException-derived CancelledError), skipping the
   node/orchestrator teardown.
3. Normal completion never broadcast a cluster shutdown, so service nodes
   were never told to exit.

These tests pin the fixes. The fuse test runs the node in a SUBPROCESS
(the fuse ends in ``os._exit(1)`` — it must not kill the test runner).
"""

import os
import signal
import subprocess
import sys
import textwrap
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FUSE_CHILD = textwrap.dedent(
    """
    import asyncio, os, sys
    from unittest.mock import patch
    from whirlwind.whirlcoms.node import Node

    async def main():
        # A node that never registered is fine for the fuse test: the
        # heartbeat loop only needs a context; patch the orchestrator
        # request to simulate a dead orchestrator (always None).
        with patch.object(Node, "_orchestrator_request", return_value=None):
            node = Node("fuse-test-node", auto_process=False)
            node._running = True
            node._ctx = True  # placate the assert
            node._tasks = []
            os.environ["WW_HEARTBEAT_TEST"] = "1"
            await node._heartbeat_loop()  # must never return normally
        print("UNREACHABLE: heartbeat loop returned", file=sys.stderr)
        os._exit(99)

    asyncio.run(main())
    """
)


def _env() -> dict:
    env = os.environ.copy()
    lib = os.path.join(sys.prefix, "lib")
    env["LD_LIBRARY_PATH"] = lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


class TestHeartbeatFuse(unittest.TestCase):
    """WHIRLWIND_HEARTBEAT_FATAL_MISSES consecutive misses -> os._exit(1)."""

    def test_fuse_exits_process(self):
        env = _env()
        env["WHIRLWIND_HEARTBEAT_FATAL_MISSES"] = "3"
        proc = subprocess.Popen(
            [sys.executable, "-c", FUSE_CHILD],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            rc = proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            self.fail("heartbeat fuse did not fire within 90s (process hung)")
        self.assertEqual(rc, 1, f"expected os._exit(1), got rc={rc}")

    def test_fuse_resets_on_success(self):
        """Alternating None/reply keeps the streak below the fuse (no exit)."""
        child = FUSE_CHILD.replace("return_value=None", "side_effect=_flaky_reply")
        child = (
            "def _flaky_reply(*a, _n=[0], **k):\n"
            "    _n[0] += 1\n"
            "    return None if _n[0] % 2 == 1 else b'reply'\n\n" + child
        )
        env = _env()
        env["WHIRLWIND_HEARTBEAT_FATAL_MISSES"] = "2"
        proc = subprocess.Popen(
            [sys.executable, "-c", child],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            rc = proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = None
        # The streak never reaches 2 -> the loop runs forever -> the outer
        # timeout kills the child (rc None). Fuse firing would be rc=1.
        if rc is not None:
            self.fail(
                f"fuse fired despite successful replies (rc={rc}); "
                f"child stderr: {proc.stderr.read()[:800]}"
            )


class TestTrainerStopSurvival(unittest.TestCase):
    """stop() must survive the CancelledError from the cancelled monitor."""

    def _make_orch(self):
        from whirlwind.training import training_grpo

        orch = training_grpo.TrainingOrchestrator.__new__(
            training_grpo.TrainingOrchestrator
        )
        orch._running = True
        orch._failure_monitor_task = None
        orch._node = None
        orch._orch = None
        orch._shutting_down = False
        return orch

    def test_stop_with_cancelled_monitor_task(self):
        import asyncio

        async def run():
            orch = self._make_orch()

            async def forever():
                while True:
                    await asyncio.sleep(5.0)

            orch._failure_monitor_task = asyncio.get_running_loop().create_task(
                forever()
            )
            await asyncio.sleep(0.05)
            await orch.stop()  # must NOT raise CancelledError
            return True

        self.assertTrue(asyncio.run(run()))

    def test_shutdown_cluster_public_wrapper(self):
        """shutdown_cluster() is callable and broadcasts via the node."""
        import asyncio

        async def run():
            orch = self._make_orch()
            sent = []

            class FakeNode:
                async def broadcast_shutdown(self, reason):
                    sent.append(reason)

            orch._node = FakeNode()
            await orch.shutdown_cluster("training complete")
            await orch.shutdown_cluster("again")  # idempotent
            self.assertEqual(sent, ["training complete"])
            return True

        self.assertTrue(asyncio.run(run()))


if __name__ == "__main__":
    unittest.main()
