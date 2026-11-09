#!/usr/bin/env python3
"""
Basic reference implementation — whirlwind RPC

Usage:
    python3 example.py worker   # start a worker node
    python3 example.py client   # start a client node (auto-exits after test)
"""

import asyncio
import sys
import torch

from whirlwind.whirlcoms import Node, Orchestrator


async def run_orchestrator() -> None:
    """Run the central orchestrator (service registry + topology broker)."""
    orch = Orchestrator()
    await orch.start()
    print("[orch] started")

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await orch.stop()


async def run_worker() -> None:
    """Register endpoints and process incoming RPCs."""
    node = Node("worker", auto_process=True)

    @node.expose("greet")
    def greet(name: str) -> str:
        return f"hello {name}"

    @node.expose("matmul")
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

    @node.expose("process_batch")
    async def process_batch(tensors: list[torch.Tensor]) -> torch.Tensor:
        # Async handler — simulates async I/O before compute
        await asyncio.sleep(0.01)
        stacked = torch.stack(tensors)
        return stacked.mean(dim=0)

    await node.start()
    print(f"[worker] ready on {node._rpc_endpoint}")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await node.stop()


async def run_client() -> None:
    """Discover peers and call RPCs."""
    node = Node("client")
    await node.start()

    # Wait for topology to propagate
    await asyncio.sleep(1)
    print(f"[client] discovered: {list(node.topology.keys())}")

    # --- String RPC ---
    result = await node.call("worker", "greet", "world")
    print(f"[client] greet('world') = {result!r}")

    # --- Tensor RPC (small — sent inline) ---
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    c = await node.call("worker", "matmul", a, b)
    expected = torch.matmul(a, b)
    print(f"[client] matmul shape={c.shape}, matches={torch.allclose(c, expected)}")

    # --- Large tensor RPC (zero-copy via /dev/shm) ---
    big_a = torch.randn(1000, 1000)  # ~8 MB
    big_b = torch.randn(1000, 1000)
    big_c = await node.call("worker", "matmul", big_a, big_b)
    expected_big = torch.matmul(big_a, big_b)
    print(
        f"[client] matmul(shape={big_c.shape}) matches={torch.allclose(big_c, expected_big)}"
    )

    # --- Async handler RPC ---
    tensors = [torch.randn(32, 32) for _ in range(5)]
    mean = await node.call("worker", "process_batch", tensors)
    expected_mean = torch.stack(tensors).mean(dim=0)
    print(
        f"[client] process_batch mean shape={mean.shape}, matches={torch.allclose(mean, expected_mean)}"
    )

    await node.stop()
    print("[client] done")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "client"

    if mode == "orchestrator":
        asyncio.run(run_orchestrator())
    elif mode == "worker":
        asyncio.run(run_worker())
    elif mode == "client":
        asyncio.run(run_client())
    else:
        print(f"Usage: {sys.argv[0]} {{orchestrator, worker, client}}")
        sys.exit(1)


if __name__ == "__main__":
    main()
