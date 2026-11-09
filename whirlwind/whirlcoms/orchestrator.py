"""Orchestrator — central service registry and topology broadcast."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import msgpack
import zmq.asyncio

from whirlwind.whirlcoms._runid import ipc_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default endpoints
# ---------------------------------------------------------------------------
# Namespaced by WW_RUN_ID when set (multi-run coexistence); unchanged otherwise.
DEFAULT_CTRL_ENDPOINT = ipc_path("ctrl")
DEFAULT_PUB_ENDPOINT = ipc_path("events")

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Central node that maintains the service registry.

    Nodes register their address and exposed endpoints here.
    The orchestrator broadcasts topology changes via a PUB socket
    so every node can maintain a full-mesh cache.

    Usage:
        orchestrator = Orchestrator()
        await orchestrator.start()
        # ... run until done ...
        await orchestrator.stop()
    """

    def __init__(
        self,
        ctrl_endpoint: str = DEFAULT_CTRL_ENDPOINT,
        pub_endpoint: str = DEFAULT_PUB_ENDPOINT,
        heartbeat_timeout: float = 15.0,
    ) -> None:
        self._ctrl_endpoint = ctrl_endpoint
        self._pub_endpoint = pub_endpoint
        self._heartbeat_timeout = heartbeat_timeout

        self._registry: dict[str, dict[str, Any]] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._ctx: zmq.asyncio.Context | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start listening for registrations and broadcasting topology."""
        if self._running:
            return

        self._running = True
        self._ctx = zmq.asyncio.Context()

        # REQ/REP for commands (REP pairs naturally with REQ sockets)
        self._ctrl_sock = self._ctx.socket(zmq.REP)
        self._ctrl_sock.bind(self._ctrl_endpoint)

        # PUB for topology events
        self._pub_sock = self._ctx.socket(zmq.PUB)
        self._pub_sock.bind(self._pub_endpoint)

        self._tasks = [
            asyncio.create_task(self._accept_loop()),
            asyncio.create_task(self._heartbeat_check_loop()),
        ]
        logger.info(
            "Orchestrator started on %s (pub: %s)",
            self._ctrl_endpoint,
            self._pub_endpoint,
        )

    async def stop(self) -> None:
        """Gracefully shut down the orchestrator."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Bounded linger (zmq's default is infinite): a queued message for a
        # peer that is gone must not block ctx.term(); 1 s still flushes the
        # shutdown broadcast to live subscribers.
        if self._ctx:
            if hasattr(self, "_ctrl_sock"):
                self._ctrl_sock.close(linger=1000)
            if hasattr(self, "_pub_sock"):
                self._pub_sock.close(linger=1000)
            self._ctx.term()
        logger.info("Orchestrator stopped")

    # -- public API ----------------------------------------------------------

    def get_node(self, node_id: str) -> dict | None:
        """Look up a node's address and endpoints."""
        return self._registry.get(node_id)

    def list_nodes(self) -> dict[str, dict]:
        """Return a copy of the full registry."""
        return dict(self._registry)

    # -- internal ------------------------------------------------------------

    async def _accept_loop(self) -> None:
        """Accept and dispatch incoming REQ/REP commands."""
        while self._running:
            try:
                msg = await self._ctrl_sock.recv()
                cmd = msgpack.unpackb(msg, raw=False)
                response = self._handle_command(cmd)
                response_bytes = msgpack.packb(response, use_bin_type=True)
                await self._ctrl_sock.send(response_bytes)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Error in orchestrator accept loop")
                # A REP socket must answer every recv, or the next recv raises
                # EFSM and this loop spins; send a best-effort error reply.
                with contextlib.suppress(Exception):
                    await self._ctrl_sock.send(
                        msgpack.packb({"ok": False, "error": str(exc)}, use_bin_type=True)
                    )

    def _handle_command(self, cmd: dict) -> dict:
        """Process a single command and return a response dict."""
        action = cmd.get("cmd")

        if action == "register":
            return self._handle_register(cmd)
        if action == "heartbeat":
            return self._handle_heartbeat(cmd)
        if action == "deregister":
            return self._handle_deregister(cmd)
        if action == "query":
            return self._handle_query(cmd)
        if action == "list":
            return {"ok": True, "nodes": self.list_nodes()}
        if action == "shutdown_all":
            return self._handle_shutdown_all(cmd)
        return {"ok": False, "error": f"Unknown command: {action}"}

    def _handle_register(self, cmd: dict) -> dict:
        """Register or re-register a node."""
        node_id = cmd["node_id"]
        self._registry[node_id] = {
            "address": cmd["address"],
            "endpoints": cmd.get("endpoints", []),
            "last_heartbeat": time.monotonic(),
        }
        logger.info(
            "Node registered: %s at %s [%s]",
            node_id,
            cmd["address"],
            cmd.get("endpoints", []),
        )
        self._broadcast("registered", node_id, cmd)
        return {"ok": True}

    def _handle_heartbeat(self, cmd: dict) -> dict:
        """Ack a heartbeat from a node; nack one the registry no longer knows.

        A node dropped by ``_heartbeat_check_loop`` keeps heartbeating; acking it
        would leave it out of the registry forever. ``ok: False``
        tells the node to re-register.
        """
        node_id = cmd["node_id"]
        if node_id not in self._registry:
            return {"ok": False, "unknown": True}
        self._registry[node_id]["last_heartbeat"] = time.monotonic()
        return {"ok": True}

    def _handle_deregister(self, cmd: dict) -> dict:
        """Remove a node from the registry."""
        node_id = cmd["node_id"]
        if node_id in self._registry:
            del self._registry[node_id]
            logger.info("Node deregistered: %s", node_id)
            self._broadcast("deregistered", node_id, cmd)
        return {"ok": True}

    def _handle_query(self, cmd: dict) -> dict:
        """Query info about a specific node."""
        node_id = cmd.get("node_id")
        if node_id:
            info = self._registry.get(node_id)
            return {"ok": True, "node": info}
        return {"ok": True, "nodes": self.list_nodes()}

    def _handle_shutdown_all(self, cmd: dict) -> dict:
        """Broadcast a cluster-wide shutdown event to all subscribed nodes.

        Any node whose ``_topology_loop`` is running receives the ``shutdown``
        event and exits its run loop gracefully (see ``Node._topology_loop``).
        The node that requested the shutdown is responsible for its own exit.
        """
        reason = cmd.get("reason", "unspecified")
        logger.warning("Cluster shutdown broadcast: reason=%s", reason)
        self._broadcast("shutdown", "orchestrator", {"reason": reason})
        return {"ok": True}

    async def _heartbeat_check_loop(self) -> None:
        """Periodically check for stale heartbeats and remove dead nodes."""
        while self._running:
            await asyncio.sleep(self._heartbeat_timeout / 2)
            now = time.monotonic()
            stale = [
                nid
                for nid, info in self._registry.items()
                if now - info["last_heartbeat"] > self._heartbeat_timeout
            ]
            for nid in stale:
                del self._registry[nid]
                logger.warning("Node %s removed: heartbeat timeout", nid)
                self._broadcast("deregistered", nid, {"node_id": nid})

    def _broadcast(self, event_type: str, node_id: str, data: dict) -> None:
        """Publish a topology event to all subscribers."""
        data["event"] = event_type
        data["node_id"] = node_id
        msg = msgpack.packb(data, use_bin_type=True)
        import contextlib

        with contextlib.suppress(zmq.ZMQError):
            self._pub_sock.send(msg, zmq.NOBLOCK)
