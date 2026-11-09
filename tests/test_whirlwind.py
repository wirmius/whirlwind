"""Comprehensive tests for whirlwind RPC system.

Covers:
  - Transport layer (serialization, SHM, tensors)
  - Message framing (request/reply round-trips)
  - Orchestrator (registration, discovery, heartbeat)
  - Node (expose, input queue, dispatch)
  - RPC calls (node-to-node, errors, timeouts)
  - Integration (multi-node scenarios)

Run with:  python3 -m unittest tests/test_whirlwind
           python3 -m pytest tests/test_whirlwind -v
"""

import asyncio
import os
import unittest

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------


def _import_torch():
    """Import torch lazily — skip tensor tests if unavailable.

    Catches OSError too (broken .so / missing CUDA libs), not just ImportError,
    so test collection skips cleanly on a half-broken torch install instead
    of crashing.
    """
    try:
        import torch

        return torch
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Transport layer tests
# ---------------------------------------------------------------------------


class TestTransportSerialization(unittest.TestCase):
    """Test msgpack serialization/deserialization round-trips."""

    def test_basic_types(self):
        from whirlwind.whirlcoms.transport import _serialize_payload, _deserialize_payload

        for obj in [42, 3.14, "hello", [1, 2, 3], {"a": 1, "b": 2}, None, True]:
            frame, shm_meta, shm_path = _serialize_payload(obj)
            result = _deserialize_payload(frame)
            self.assertEqual(result, obj, f"Round-trip failed for {obj!r}")

    def test_nested_structures(self):
        from whirlwind.whirlcoms.transport import _serialize_payload, _deserialize_payload

        obj = {
            "list_of_dicts": [{"x": i, "y": i * 2} for i in range(10)],
            "deep": {"a": {"b": {"c": [1, 2, 3]}}},
        }
        frame, _, _ = _serialize_payload(obj)
        result = _deserialize_payload(frame)
        self.assertEqual(result, obj)

    def test_small_bytes_inline(self):
        from whirlwind.whirlcoms.transport import _serialize_payload, _deserialize_payload

        obj = b"small binary data"
        frame, shm_meta, shm_path = _serialize_payload(obj)
        self.assertIsNone(shm_meta)  # small → inline
        result = _deserialize_payload(frame)
        self.assertEqual(result, obj)

    def test_large_bytes_shm(self):
        from whirlwind.whirlcoms.transport import _serialize_payload, _deserialize_payload

        # Create data above 64KB threshold
        large_data = {"data": list(range(100_000))}
        frame, shm_meta, shm_path = _serialize_payload(large_data)
        self.assertIsNotNone(shm_meta)
        self.assertIsNotNone(shm_path)
        self.assertTrue(os.path.exists(shm_path))
        result = _deserialize_payload(frame)
        self.assertEqual(result, large_data)
        # Receiver cleanup: SHM should be gone after deserialize
        self.assertFalse(os.path.exists(shm_path))


class TestTensorTransport(unittest.TestCase):
    """Test torch.Tensor serialization via SHM and inline paths."""

    @unittest.skipIf(_import_torch() is None, "torch not installed")
    def test_small_tensor_inline(self):
        import torch
        from whirlwind.whirlcoms.transport import _serialize_payload, _deserialize_payload

        t = torch.randn(4, 4)  # 64 bytes — well below threshold
        frame, shm_meta, _ = _serialize_payload(t)
        self.assertIsNone(shm_meta)  # small → inline
        result = _deserialize_payload(frame)
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.allclose(t, result))

    @unittest.skipIf(_import_torch() is None, "torch not installed")
    def test_large_tensor_shm(self):
        import torch
        from whirlwind.whirlcoms.transport import _serialize_payload, _deserialize_payload

        # 1000x1000 float32 = 4MB — above 64KB threshold
        t = torch.randn(1000, 1000)
        frame, shm_meta, shm_path = _serialize_payload(t)
        self.assertIsNotNone(shm_meta)
        self.assertTrue(os.path.exists(shm_path))
        result = _deserialize_payload(frame)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, t.shape)
        self.assertEqual(result.dtype, t.dtype)
        self.assertTrue(torch.allclose(t, result))
        self.assertFalse(os.path.exists(shm_path))  # cleaned up

    @unittest.skipIf(_import_torch() is None, "torch not installed")
    def test_tensor_dtypes(self):
        import torch
        from whirlwind.whirlcoms.transport import _serialize_payload, _deserialize_payload

        for dtype in [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.uint8,
        ]:
            with self.subTest(dtype=dtype):
                t = torch.tensor([1, 2, 3], dtype=dtype)
                frame, _, _ = _serialize_payload(t)
                result = _deserialize_payload(frame)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(torch.equal(t, result))

    @unittest.skipIf(_import_torch() is None, "torch not installed")
    def test_tensor_in_nested_structure(self):
        import torch
        from whirlwind.whirlcoms.transport import _serialize_payload, _deserialize_payload

        t = torch.randn(10, 10)
        obj = {
            "tensor": t,
            "metadata": {"name": "test"},
            "list": [t, t * 2],
        }
        frame, _, _ = _serialize_payload(obj)
        result = _deserialize_payload(frame)
        self.assertIsInstance(result["tensor"], torch.Tensor)
        self.assertTrue(torch.allclose(result["tensor"], t))
        self.assertEqual(len(result["list"]), 2)

    @unittest.skipIf(_import_torch() is None, "torch not installed")
    def test_zero_copy_shm(self):
        """Verify SHM tensor transfer doesn't corrupt data."""
        import torch
        from whirlwind.whirlcoms.transport import _serialize_payload, _deserialize_payload

        # Large tensor with known pattern
        t = torch.arange(100_000, dtype=torch.float32).reshape(500, 200)
        frame, shm_meta, shm_path = _serialize_payload(t)
        result = _deserialize_payload(frame)
        self.assertTrue(torch.equal(t, result))
        self.assertEqual(result.nbytes, t.nbytes)


# ---------------------------------------------------------------------------
# 2. Message framing tests
# ---------------------------------------------------------------------------


class TestMessageFraming(unittest.TestCase):
    """Test request/reply frame construction and parsing."""

    def test_request_round_trip(self):
        from whirlwind.whirlcoms.transport import make_request_frame, parse_request_frame

        header, payload, cleanup = make_request_frame(
            msg_id="abc123",
            src_node="sender",
            dst_node="receiver",
            endpoint="greet",
            args=("world",),
            kwargs={"greeting": "hello"},
        )
        parts = [header, payload]
        frame = parse_request_frame(parts)
        self.assertEqual(frame["msg_id"], "abc123")
        self.assertEqual(frame["src"], "sender")
        self.assertEqual(frame["dst"], "receiver")
        self.assertEqual(frame["endpoint"], "greet")
        # msgpack converts tuples to lists — that's standard msgpack behavior
        self.assertEqual(frame["payload"]["args"], ["world"])
        self.assertEqual(frame["payload"]["kwargs"], {"greeting": "hello"})

    def test_reply_round_trip(self):
        from whirlwind.whirlcoms.transport import make_reply_frame, parse_reply_frame

        header, payload = make_reply_frame(
            msg_id="abc123",
            src_node="receiver",
            dst_node="sender",
            result={"status": "ok"},
        )
        result, error = parse_reply_frame([header, payload])
        self.assertIsNone(error)
        self.assertEqual(result, {"status": "ok"})

    def test_reply_with_error(self):
        from whirlwind.whirlcoms.transport import make_reply_frame, parse_reply_frame

        header, payload = make_reply_frame(
            msg_id="abc123",
            src_node="receiver",
            dst_node="sender",
            result=None,
            error="Something went wrong",
        )
        result, error = parse_reply_frame([header, payload])
        self.assertIsNotNone(error)
        self.assertIn("Something went wrong", error)

    @unittest.skipIf(_import_torch() is None, "torch not installed")
    def test_request_with_tensor(self):
        import torch
        from whirlwind.whirlcoms.transport import make_request_frame, parse_request_frame

        t = torch.randn(32, 32)
        header, payload, cleanup = make_request_frame(
            msg_id="t1",
            src_node="a",
            dst_node="b",
            endpoint="process",
            args=(t,),
            kwargs={},
        )
        frame = parse_request_frame([header, payload])
        received_tensor = frame["payload"]["args"][0]
        self.assertIsInstance(received_tensor, torch.Tensor)
        self.assertTrue(torch.allclose(received_tensor, t))


# ---------------------------------------------------------------------------
# 3. Async integration tests
# ---------------------------------------------------------------------------


def _async_test(coro):
    """Decorator to run async test methods."""

    def wrapper(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro(self))
        except Exception:
            raise
        finally:
            import contextlib

            with contextlib.suppress(Exception):
                loop.run_until_complete(asyncio.sleep(0.05))
            loop.close()

    return wrapper


class TestOrchestrator(unittest.TestCase):
    """Test orchestrator registration and discovery."""

    @_async_test
    async def test_register_and_query(self):
        from whirlwind.whirlcoms.orchestrator import Orchestrator
        import zmq.asyncio
        import msgpack

        orch = Orchestrator()
        await orch.start()
        await asyncio.sleep(0.1)  # let sockets bind

        # Register a node
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.connect(orch._ctrl_endpoint)

        await sock.send(
            msgpack.packb(
                {
                    "cmd": "register",
                    "node_id": "test-node",
                    "address": "ipc:///tmp/whirlwind-test",
                    "endpoints": ["hello", "world"],
                },
                use_bin_type=True,
            )
        )
        await sock.recv()

        # Query
        await sock.send(
            msgpack.packb(
                {
                    "cmd": "query",
                    "node_id": "test-node",
                },
                use_bin_type=True,
            )
        )
        reply = msgpack.unpackb(await sock.recv(), raw=False)
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["node"]["endpoints"], ["hello", "world"])

        sock.close()
        ctx.term()
        await orch.stop()

    @_async_test
    async def test_list_nodes(self):
        from whirlwind.whirlcoms.orchestrator import Orchestrator
        import zmq.asyncio
        import msgpack

        orch = Orchestrator()
        await orch.start()
        await asyncio.sleep(0.1)

        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.connect(orch._ctrl_endpoint)

        # Register two nodes
        for node_id in ["node-a", "node-b"]:
            await sock.send(
                msgpack.packb(
                    {
                        "cmd": "register",
                        "node_id": node_id,
                        "address": f"ipc:///tmp/whirlwind-{node_id}",
                        "endpoints": ["work"],
                    },
                    use_bin_type=True,
                )
            )
            await sock.recv()

        # List all
        await sock.send(msgpack.packb({"cmd": "list"}, use_bin_type=True))
        reply = msgpack.unpackb(await sock.recv(), raw=False)
        self.assertTrue(reply["ok"])
        self.assertEqual(len(reply["nodes"]), 2)
        self.assertIn("node-a", reply["nodes"])
        self.assertIn("node-b", reply["nodes"])

        sock.close()
        ctx.term()
        await orch.stop()

    @_async_test
    async def test_deregister(self):
        from whirlwind.whirlcoms.orchestrator import Orchestrator
        import zmq.asyncio
        import msgpack

        orch = Orchestrator()
        await orch.start()
        await asyncio.sleep(0.1)

        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.connect(orch._ctrl_endpoint)

        # Register then deregister
        await sock.send(
            msgpack.packb(
                {
                    "cmd": "register",
                    "node_id": "temp-node",
                    "address": "ipc:///tmp/whirlwind-temp",
                    "endpoints": [],
                },
                use_bin_type=True,
            )
        )
        await sock.recv()

        await sock.send(
            msgpack.packb(
                {
                    "cmd": "deregister",
                    "node_id": "temp-node",
                },
                use_bin_type=True,
            )
        )
        await sock.recv()

        # Should not exist anymore
        await sock.send(
            msgpack.packb(
                {
                    "cmd": "query",
                    "node_id": "temp-node",
                },
                use_bin_type=True,
            )
        )
        reply = msgpack.unpackb(await sock.recv(), raw=False)
        self.assertIsNone(reply["node"])

        sock.close()
        ctx.term()
        await orch.stop()


class TestNodeExpose(unittest.TestCase):
    """Test node endpoint registration and queue mechanics."""

    @staticmethod
    async def _serve(node, n=1):
        """Serve n queued calls as process_rpc_requests does (replaces process_one/process_loop)."""
        calls = []
        for _ in range(n):
            calls.append(await node.input_queue.get())
            await node._serve_async(calls[-1])
            await node.output_queue.put(calls[-1])
        return calls

    @_async_test
    async def test_expose_decorator(self):
        from whirlwind.whirlcoms.node import Node

        node = Node("test-node")

        @node.expose("greet")
        def greet(name):
            return f"hello {name}"

        @node.expose()  # uses function name
        def add(a, b):
            return a + b

        self.assertEqual(node.endpoints, ["greet", "add"])
        self.assertEqual(node._handlers["greet"]("world"), "hello world")
        self.assertEqual(node._handlers["add"](2, 3), 5)

    @_async_test
    async def test_queue_and_process(self):
        """Test that process_one correctly dispatches to handlers."""
        from whirlwind.whirlcoms.node import Node
        from whirlwind.whirlcoms.orchestrator import Orchestrator
        import msgpack

        orch = Orchestrator()
        await orch.start()
        await asyncio.sleep(0.1)

        node = Node("test-node")

        @node.expose("echo")
        def echo(msg):
            return f"echo: {msg}"

        await node.start()
        await asyncio.sleep(0.1)

        # Manually push a request onto the queue
        header = msgpack.packb(
            {
                "msg_id": "test-1",
                "src": "fake-client",
                "dst": "test-node",
                "endpoint": "echo",
            },
            use_bin_type=True,
        )
        payload = msgpack.packb(
            {
                "args": ("hello",),
                "kwargs": {},
            },
            use_bin_type=True,
        )
        node.input_queue.put_nowait(node._build_call([header, payload]))

        self.assertEqual(node.pending_count, 1)
        (call,) = await self._serve(node)
        self.assertEqual(node.pending_count, 0)
        self.assertEqual(call.result, "echo: hello")

        await node.stop()
        await orch.stop()

    @_async_test
    async def test_process_loop_batch(self):
        from whirlwind.whirlcoms.node import Node
        from whirlwind.whirlcoms.orchestrator import Orchestrator
        import msgpack

        orch = Orchestrator()
        await orch.start()
        await asyncio.sleep(0.1)

        node = Node("batch-node")

        @node.expose("double")
        def double(x):
            return x * 2

        await node.start()
        await asyncio.sleep(0.1)

        # Push 3 requests
        for i in range(3):
            header = msgpack.packb(
                {
                    "msg_id": f"batch-{i}",
                    "src": "client",
                    "dst": "batch-node",
                    "endpoint": "double",
                },
                use_bin_type=True,
            )
            payload = msgpack.packb({"args": (i,), "kwargs": {}}, use_bin_type=True)
            node.input_queue.put_nowait(node._build_call([header, payload]))

        self.assertEqual(node.pending_count, 3)
        await self._serve(node, 2)
        self.assertEqual(node.pending_count, 1)  # 2 processed, 1 left

        await node.stop()
        await orch.stop()

    @_async_test
    async def test_unknown_endpoint(self):
        """Request to non-existent endpoint should return error reply."""
        from whirlwind.whirlcoms.node import Node
        from whirlwind.whirlcoms.orchestrator import Orchestrator
        import msgpack

        # A registry, like the sibling tests: with NODE_COMMS_ENSURE_REGISTER=1
        # (bringup's default) a node with nothing to register with retries and
        # then raises in start().
        orch = Orchestrator()
        await orch.start()
        node = Node("err-node")
        await node.start()
        await asyncio.sleep(0.1)

        # Push request to unknown endpoint
        header = msgpack.packb(
            {
                "msg_id": "err-1",
                "src": "client",
                "dst": "err-node",
                "endpoint": "nonexistent",
            },
            use_bin_type=True,
        )
        payload = msgpack.packb({"args": (), "kwargs": {}}, use_bin_type=True)
        node.input_queue.put_nowait(node._build_call([header, payload]))

        (call,) = await self._serve(node)
        self.assertIn("Unknown endpoint", call.error)  # error reply, no crash

        await node.stop()
        await orch.stop()


class TestNodeRpcCalls(unittest.TestCase):
    """Test node-to-node RPC via the full stack."""

    @_async_test
    async def test_basic_rpc(self):
        """Two nodes: one exposes, one calls."""
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        worker = Node("worker", auto_process=True)

        @worker.expose("greet")
        def greet(name):
            return f"hello {name}"

        await worker.start()
        await asyncio.sleep(0.5)  # let topology propagate

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.5)

        # Call worker from client
        result = await client.call("worker", "greet", "world")
        self.assertEqual(result, "hello world")

        await client.stop()
        await worker.stop()
        await orch.stop()

    @_async_test
    async def test_rpc_with_kwargs(self):
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        worker = Node("worker", auto_process=True)

        @worker.expose("add")
        def add(a, b, mult=1):
            return (a + b) * mult

        await worker.start()
        await asyncio.sleep(0.5)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.5)

        result = await client.call("worker", "add", 2, 3, mult=4)
        self.assertEqual(result, 20)

        await client.stop()
        await worker.stop()
        await orch.stop()

    @_async_test
    async def test_rpc_unknown_node(self):
        from whirlwind.whirlcoms import Node, Orchestrator
        from whirlwind.whirlcoms.node import RpcError

        orch = Orchestrator()
        await orch.start()

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.5)

        with self.assertRaises(RpcError) as ctx:
            await client.call("nonexistent", "anything")

        self.assertIn("Unknown node", str(ctx.exception))

        await client.stop()
        await orch.stop()

    @_async_test
    async def test_rpc_unknown_endpoint(self):
        from whirlwind.whirlcoms import Node, Orchestrator
        from whirlwind.whirlcoms.node import RpcError

        orch = Orchestrator()
        await orch.start()

        worker = Node("worker", auto_process=True)

        @worker.expose("exists")
        def exists():
            return True

        await worker.start()
        await asyncio.sleep(0.5)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.5)

        with self.assertRaises(RpcError):
            await client.call("worker", "does_not_exist")

        await client.stop()
        await worker.stop()
        await orch.stop()

    @_async_test
    async def test_multi_node_mesh(self):
        """Three nodes can all call each other."""
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        # Three workers
        nodes = {}
        for name in ["alpha", "beta", "gamma"]:
            node = Node(name, auto_process=True)

            @node.expose("identity")
            def ident(_name=name):
                return _name

            @node.expose("compute")
            def compute(x, _name=name):
                return _name, x * 2

            nodes[name] = node
            await node.start()

        await asyncio.sleep(1)  # let topology propagate

        # Every node calls every other node
        for src_name, src_node in nodes.items():
            for dst_name, _ in nodes.items():
                if src_name == dst_name:
                    continue
                result = await src_node.call(dst_name, "identity")
                self.assertEqual(result, dst_name)
                result = await src_node.call(dst_name, "compute", 21)
                # msgpack converts tuples to lists
                self.assertEqual(list(result), [dst_name, 42])

        for node in nodes.values():
            await node.stop()
        await orch.stop()

    @_async_test
    async def test_async_handler(self):
        """Async handlers should work correctly."""
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        worker = Node("worker", auto_process=True)

        @worker.expose("delayed")
        async def delayed(value, delay=0.01):
            await asyncio.sleep(delay)
            return f"got:{value}"

        await worker.start()
        await asyncio.sleep(0.5)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.5)

        result = await client.call("worker", "delayed", "test", delay=0.01)
        self.assertEqual(result, "got:test")

        await client.stop()
        await worker.stop()
        await orch.stop()


class TestTensorRpc(unittest.TestCase):
    """Test tensor transfer via RPC calls between nodes."""

    @unittest.skipIf(_import_torch() is None, "torch not installed")
    @_async_test
    async def test_small_tensor_rpc(self):
        """Small tensor via inline path."""
        import torch
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        worker = Node("worker", auto_process=True)

        @worker.expose("sum_tensor")
        def sum_tensor(t):
            return float(t.sum())

        await worker.start()
        await asyncio.sleep(0.5)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.5)

        t = torch.tensor([1.0, 2.0, 3.0])
        result = await client.call("worker", "sum_tensor", t)
        self.assertAlmostEqual(result, 6.0)

        await client.stop()
        await worker.stop()
        await orch.stop()

    @unittest.skipIf(_import_torch() is None, "torch not installed")
    @_async_test
    async def test_large_tensor_rpc_shm(self):
        """Large tensor via SHM — verify zero-copy transfer."""
        import torch
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        worker = Node("worker", auto_process=True)

        @worker.expose("stats")
        def stats(t):
            return {
                "mean": float(t.mean()),
                "std": float(t.std()),
                "shape": list(t.shape),
            }

        await worker.start()
        await asyncio.sleep(0.5)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.5)

        # 2MB tensor — definitely goes through SHM
        t = torch.randn(500, 500)
        result = await client.call("worker", "stats", t)
        self.assertAlmostEqual(result["mean"], float(t.mean()), places=5)
        self.assertEqual(result["shape"], [500, 500])

        await client.stop()
        await worker.stop()
        await orch.stop()

    @unittest.skipIf(_import_torch() is None, "torch not installed")
    @_async_test
    async def test_tensor_transform_rpc(self):
        """Send tensor, transform, get tensor back."""
        import torch
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        worker = Node("worker", auto_process=True)

        @worker.expose("normalize")
        def normalize(t, mean=0.0, std=1.0):
            return (t - mean) / std

        await worker.start()
        await asyncio.sleep(0.5)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.5)

        t = torch.randn(100, 100)
        result = await client.call("worker", "normalize", t, mean=0.5, std=0.5)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, t.shape)
        expected = (t - 0.5) / 0.5
        self.assertTrue(torch.allclose(result, expected))

        await client.stop()
        await worker.stop()
        await orch.stop()


class TestTopologyDiscovery(unittest.TestCase):
    """Test that nodes discover each other via orchestrator broadcasts."""

    @_async_test
    async def test_topology_populates(self):
        """Node should see other nodes after they register."""
        from whirlwind.whirlcoms import Node, Orchestrator

        orch = Orchestrator()
        await orch.start()

        worker = Node("worker", auto_process=True)
        await worker.start()
        await asyncio.sleep(0.3)

        client = Node("client")
        await client.start()
        await asyncio.sleep(0.5)  # wait for broadcast

        self.assertIn("worker", client.topology)
        self.assertEqual(client.topology["worker"]["endpoints"], [])

        await client.stop()
        await worker.stop()
        await orch.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
