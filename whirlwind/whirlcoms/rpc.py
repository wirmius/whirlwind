"""RPC client — call remote endpoints with retry/reconnect logic."""

from __future__ import annotations

import logging
from typing import Any

from whirlwind.whirlcoms.node import Node, RpcError

logger = logging.getLogger(__name__)


class RpcClient:
    """Standalone RPC client that calls endpoints on other nodes.

    Uses a Node's topology cache and socket infrastructure.
    Retries failed calls with reconnection.

    Usage:
        rpc = RpcClient(node)
        result = await rpc.call("worker-a", "greet", "world")
    """

    def __init__(
        self,
        node: Node,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self._node = node
        self._max_retries = max_retries
        self._timeout = timeout

    async def call(
        self,
        target_node: str,
        endpoint: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call an RPC endpoint on another node with retry logic.

        Args:
            target_node: The node_id of the target node.
            endpoint: The exposed endpoint name to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The result returned by the remote endpoint.

        Raises:
            RpcError: If the call fails after all retries.
        """
        last_error: RpcError | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._node.call(
                    target_node,
                    endpoint,
                    *args,
                    **kwargs,
                )
            except RpcError as exc:
                last_error = exc
                logger.warning(
                    "RPC attempt %d/%d to %s/%s failed: %s",
                    attempt,
                    self._max_retries,
                    target_node,
                    endpoint,
                    exc,
                )
                if attempt < self._max_retries:
                    await self._node._refresh_topology()
        raise last_error  # type: ignore[misc]


def call(
    node: Node,
    target: str,
    endpoint: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Module-level convenience function for RPC calls.

    Usage:
        import asyncio
        from whirlwind import Node, call

        node = Node("me")
        await node.start()

        result = await call(node, "worker-a", "greet", "world")
    """
    return node.call(target, endpoint, *args, **kwargs)


# Singleton rpc object for convenience — users create via RpcClient(node)
rpc = RpcClient  # type: ignore
