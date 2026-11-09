"""Node — RPC endpoint runtime with expose decorator and heartbeat."""

from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import os
import queue
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import msgpack
import zmq.asyncio

from whirlwind.whirlcoms._runid import ipc_path
from whirlwind.whirlcoms.message import (
    Message,
    MessageRegistry,
    _from_dict,
    _to_dict,
)
from whirlwind.whirlcoms.transport import (
    DEFAULT_SHM_THRESHOLD,
    make_reply_frame,
    parse_request_frame,
)

logger = logging.getLogger(__name__)

# Debug-level node-to-node RPC comms output.
# Set NODE_COMMS_DBG_STDOUT=1 to print these to stderr in real time.
_COMMS_DBG = os.environ.get("NODE_COMMS_DBG_STDOUT") in ("1", "true", "yes", "")

# Block node startup until orchestrator accepts registration.
# Set NODE_COMMS_ENSURE_REGISTER=1 to enable.
_ENSURE_REGISTER = os.environ.get("NODE_COMMS_ENSURE_REGISTER") in (
    "1",
    "true",
    "yes",
    "",
)

# When set, RPC timeouts raise RpcError instead of logging a warning.
# Set NODE_COMMS_RPC_FAIL_TIMEOUT=1 to enable strict mode.
_FAIL_ON_TIMEOUT = os.environ.get("NODE_COMMS_RPC_FAIL_TIMEOUT") in (
    "1",
    "true",
    "yes",
    "",
)


def _dbg(msg: str, *args: object) -> None:
    """Print debug-level RPC comms to stderr when NODE_COMMS_DBG_STDOUT is set."""
    if not _COMMS_DBG:
        return
    formatted = msg % args if args else msg
    print(f"[DBG-node] {formatted}", file=sys.stderr)


# RPC call timing (enabled via NODE_COMMS_TIMING env var).
# Keeps the last 1000 completed calls per node.
_COMMS_TIMING = os.environ.get("NODE_COMMS_TIMING") in ("1", "true", "yes", "")
_MAX_TIMING_ENTRIES = 1000

# How long stop() lets unsent messages flush before dropping them (ms).
_STOP_LINGER_MS = 1000

# Shared deque: list of dicts with keys
#   target_node, endpoint, t_init, t_send, t_recv, t_deserialize, latency_ms
_timing_buf: collections.deque[dict[str, Any]] = collections.deque(
    maxlen=_MAX_TIMING_ENTRIES
)


# ---------------------------------------------------------------------------
# Default endpoints
# ---------------------------------------------------------------------------
DEFAULT_CTRL_ENDPOINT = ipc_path("ctrl")
DEFAULT_PUB_ENDPOINT = ipc_path("events")

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass
class _RPCCall:
    """One RPC's lifecycle on the worker side (Node-internal).

    The IO layer fills the request fields (``msg_id``/``src``/
    ``routing_frames``/``handler``/``msg``/``args``/``meta``) and puts the call
    on :attr:`Node.input_queue`. The worker runs the handler and fills
    ``result``/``error``; the IO layer's output drain reads those plus the
    routing fields to send the reply. One small object for both directions —
    the Node owns the queues, so this is a private implementation detail, not
    a public type.
    """

    msg_id: str
    src: str
    endpoint: str
    handler: Callable[..., Any] | None
    routing_frames: list[bytes]
    msg: Any = None  # typed message (from_dict done at the seam); None = legacy raw-args
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    result: Any = None  # filled by the worker
    error: str | None = None  # filled by the worker
    src_addr: str | None = None  # caller's RPC endpoint from the header (reply routing)


class Node:
    """An RPC node that exposes callable endpoints.

    Each node binds a DEALER socket for peer-to-peer RPC and
    registers with the orchestrator for service discovery.

    Usage:
        node = Node("worker-a")

        @node.expose("greet")
        def greet(name: str) -> str:
            return f"hello {name}"

        await node.start()
        # ... run until done ...
        await node.stop()
    """

    def __init__(
        self,
        node_id: str,
        endpoint_suffix: str | None = None,
        ctrl_endpoint: str = DEFAULT_CTRL_ENDPOINT,
        pub_endpoint: str = DEFAULT_PUB_ENDPOINT,
        shm_threshold: int = DEFAULT_SHM_THRESHOLD,
        heartbeat_interval: float = 5.0,
        auto_process: bool = False,
        rpc_timeout: float | None = None,
        fail_on_timeout: bool = _FAIL_ON_TIMEOUT,
        event_loop: asyncio.AbstractEventLoop | None = None,
        propagate_exceptions: bool = False,
    ) -> None:
        self.node_id = node_id
        # Allow overriding the RPC timeout via env var (e.g. for slow colocate
        # rollout engines where a full generation batch can exceed the 300s
        # default). Falls back to 300s if unset.
        if rpc_timeout is None:
            rpc_timeout = float(os.environ.get("WHIRLWIND_RPC_TIMEOUT", "300.0"))
        self._ctrl_endpoint = ctrl_endpoint
        self._pub_endpoint = pub_endpoint
        self._shm_threshold = shm_threshold
        self._heartbeat_interval = heartbeat_interval
        self._auto_process = auto_process
        self._rpc_timeout = rpc_timeout
        self._fail_on_timeout = fail_on_timeout
        self._event_loop = event_loop

        # Per-node DEALER endpoint for RPC (namespaced by WW_RUN_ID when set)
        suffix = endpoint_suffix or node_id
        self._rpc_endpoint = ipc_path(suffix)

        # Endpoint registry: name -> callable
        self._handlers: dict[str, Callable] = {}

        # Topology cache: node_id -> {address, endpoints}
        self._topology: dict[str, dict] = {}

        # Transient client sockets (dst_node -> DEALER socket)
        self._client_sockets: dict[str, zmq.asyncio.Socket] = {}

        # Pending RPC futures: msg_id -> asyncio.Future
        self._pending: dict[str, asyncio.Future] = {}

        # Pending RPC timing (msg_id -> {target_node, endpoint, t_init, t_send})
        # Only populated when _COMMS_TIMING is True.
        self._pending_timing: dict[str, dict[str, Any]] = {}

        # Two first-class queues forming the Node<->work seam:
        #   input_queue  : IO layer ──► worker  (_RPCCall, request deserialized)
        #   output_queue : worker ──► IO layer  (_RPCCall, result/error filled)
        # The IO layer recv→from_dict→input_queue; the user-driven
        # process_rpc_requests drains input_queue, runs the handler,
        # to_dict→output_queue; the IO output_drain sends. zmq sockets stay
        # loop-only.
        self._input_queue: asyncio.Queue[_RPCCall] = asyncio.Queue()
        self._output_queue: asyncio.Queue[_RPCCall] = asyncio.Queue()

        # Optional callback fired when a cluster "shutdown" event is received
        # on the topology SUB socket. Used by the training orchestrator to break
        # its run loop; worker nodes simply rely on ``is_running`` going False.
        self._on_shutdown: Callable[[], Any] | None = None

        # Thread-safe queue for background (fire-and-forget) RPC sends.
        # Drained by _bg_send_drain_task on the event loop.
        self._bg_send_queue: queue.Queue[tuple[bytes, bytes, str, str]] = queue.Queue()

        # Message registry: endpoint -> (expects, returns).
        # Populated by expose(); read at the serdes seam (from_dict on recv,
        # to_dict on reply).
        self._registry = MessageRegistry()

        # If True, a handler exception propagates out of process_rpc_requests (fail-
        # fast). Default False: handler exceptions become error replies and the
        # worker keeps routing (preserves the legacy robustness contract).
        self._propagate_exceptions = propagate_exceptions

        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._ctx: zmq.asyncio.Context | None = None

    @property
    def _loop(self) -> asyncio.AbstractEventLoop:
        """Return the user-provided loop, or fall back to the running one."""
        return self._event_loop or asyncio.get_running_loop()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Public access to the event loop this node runs on."""
        return self._loop

    # -- decorator -----------------------------------------------------------

    def expose(
        self,
        name: str | None = None,
        *,
        expects: type | None = None,
        returns: type | None = None,
    ) -> Callable:
        """Decorator to register a function as an RPC endpoint.

        Optionally binds a message contract (``expects``/``returns`` types) so
        the Node can own typed serdes at the seam: the IO layer does
        ``from_dict(expects)`` on recv; ``process_rpc_requests`` does
        ``to_dict(returns)`` on the reply. Omitting them preserves today's
        behaviour (raw args in / raw result out — legacy ``@decorate_serdes``
        handlers still work unchanged via the raw-args path).
        """

        def decorator(func: Callable) -> Callable:
            endpoint_name = name or func.__name__
            self._handlers[endpoint_name] = func
            self._registry.register_endpoint(
                endpoint_name, expects=expects, returns=returns
            )
            logger.debug(
                "Exposed endpoint '%s' on node '%s'", endpoint_name, self.node_id
            )
            return func

        return decorator

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the node: bind sockets, register with orchestrator.

        Must be called from the configured event loop (``self._loop``).
        """
        if self._running:
            return

        assert self._loop is asyncio.get_running_loop(), (
            f"Node '{self.node_id}' must be started from the configured event loop"
        )
        self._running = True
        # Cache the loop so process_rpc_requests_sync (running in a thread) can reach
        # it without a running loop in that thread.
        if self._event_loop is None:
            self._event_loop = asyncio.get_running_loop()
        self._ctx = zmq.asyncio.Context()

        # DEALER socket for incoming/outgoing RPC
        self._dealer = self._ctx.socket(zmq.DEALER)
        self._dealer.bind(self._rpc_endpoint)

        # SUB socket for topology events from orchestrator
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(self._pub_endpoint)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all

        # Register with orchestrator (best-effort, single attempt) then fetch topology
        await self._register_with_orchestrator()
        reply = await self._orchestrator_request({"cmd": "list"})
        if reply:
            data = msgpack.unpackb(reply, raw=False)
            if data.get("ok"):
                nodes = data.get("nodes", {})
                for nid, info in nodes.items():
                    if nid != self.node_id:
                        self._topology[nid] = info
                logger.debug("Topology loaded: %d nodes", len(self._topology))

        # Start background tasks — always schedule on self._loop so they run
        # on the configured event loop even when start() is called externally.
        # IO layer: _recv_loop (recv→from_dict→input_queue) + _output_drain_loop
        # (output_queue→send), plus topology/heartbeat/bg_send. zmq sockets
        # stay loop-only; handlers never run here.
        self._tasks = [
            self._loop.create_task(self._recv_loop()),
            self._loop.create_task(self._output_drain_loop()),
            self._loop.create_task(self._topology_loop()),
            self._loop.create_task(self._heartbeat_loop()),
            self._loop.create_task(self._bg_send_drain_loop()),
        ]
        # auto_process = convenience: the Node drives one process_rpc_requests
        # worker itself, so existing nodes that don't call it explicitly still
        # process RPCs (parity with the legacy behaviour). For concurrency or
        # user control, omit auto_process and call process_rpc_requests yourself.
        if self._auto_process:
            self._tasks.append(self._loop.create_task(self.process_rpc_requests()))
        logger.info("Node '%s' started on %s", self.node_id, self._rpc_endpoint)

    async def stop(self) -> None:
        """Gracefully shut down: deregister and close sockets."""
        self._running = False
        await self._deregister()

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close sockets with a bounded linger. zmq's default linger is
        # infinite: a reply queued for a caller that has already exited (it
        # timed out and the run ended) would block ctx.term() forever and the
        # node would never exit. Messages to live peers still get
        # _STOP_LINGER_MS to flush.
        if self._ctx:
            self._dealer.close(linger=_STOP_LINGER_MS)
            self._sub.close(linger=_STOP_LINGER_MS)
            for sock in self._client_sockets.values():
                sock.close(linger=_STOP_LINGER_MS)
            self._client_sockets.clear()
            self._ctx.term()
        logger.info("Node '%s' stopped", self.node_id)

    # -- user-driven work loop -----------------------------------------------

    @property
    def input_queue(self) -> asyncio.Queue[_RPCCall]:
        """Incoming RPC queue (IO layer → worker). Drained by
        :meth:`process_rpc_requests`; each item is a call with the request
        already deserialized (``msg`` typed) when the endpoint declared
        ``expects``."""
        return self._input_queue

    @property
    def output_queue(self) -> asyncio.Queue[_RPCCall]:
        """Outgoing reply queue (worker → IO layer). Filled by
        :meth:`process_rpc_requests`; drained by the IO layer's output drain."""
        return self._output_queue

    async def process_rpc_requests(
        self, skip_endpoints: set[str] | None = None
    ) -> None:
        """Route incoming RPC requests to their handlers and produce replies.

        The user-driven worker loop. Pulls a call from :attr:`input_queue` (the
        IO layer has already deserialized it — ``call.msg`` is typed when the
        endpoint declared ``expects``), runs the registered handler, serializes
        the return (``to_dict`` if ``returns`` was declared), and pushes the
        call back on :attr:`output_queue` for the IO layer to send. Routing is
        automatic via :meth:`expose`; the caller does no dispatch.

        Exits when ``is_running`` goes False (shutdown). A handler exception
        becomes an error reply by default (routing continues); with
        ``propagate_exceptions`` it propagates out and the loop exits.

        Concurrency is caller-controlled: spawn N tasks each running this
        method (asyncio) or N threads each running
        :meth:`process_rpc_requests_sync`.

        ``skip_endpoints`` makes this a CONTROL worker: calls whose endpoint
        is in the set are requeued (tail) instead of served — a control lane
        that never starves on long-held serving handlers (e.g. rollout
        generates parked while the engine sleeps for an update step).
        """
        while self._running:
            try:
                call = await self._input_queue.get()
            except asyncio.CancelledError:
                break
            if call is None:  # poison pill (stop())
                break
            if skip_endpoints and call.endpoint in skip_endpoints:
                await self._input_queue.put(call)
                await asyncio.sleep(0.01)  # let serving workers drain first
                continue
            await self._serve_async(call)
            await self._output_queue.put(call)

    def process_rpc_requests_sync(self) -> None:
        """Blocking worker loop for a plain thread.

        Same as :meth:`process_rpc_requests` but blocking: pulls from
        :attr:`input_queue` via the loop (``run_coroutine_threadsafe``), runs
        sync handlers in *this* thread (true CPU parallel across multiple
        threads), and pushes replies to :attr:`output_queue` thread-safely.
        Async handlers run on the loop. Exits when ``is_running`` goes False.
        """
        while self._running:
            try:
                call = asyncio.run_coroutine_threadsafe(
                    self._input_queue.get(), self._loop
                ).result()  # blocks this thread until a call arrives
            except Exception:
                if not self._running:
                    break
                continue
            if call is None:  # poison pill
                break
            self._serve_sync(call)
            self._loop.call_soon_threadsafe(self._output_queue.put_nowait, call)

    async def _serve_async(self, call: _RPCCall) -> None:
        """Run one call's handler on the loop and fill its reply (async path)."""
        if call.handler is None:
            call.error = f"Unknown endpoint: {call.endpoint}"
            return
        try:
            if asyncio.iscoroutinefunction(call.handler):
                result = await call.handler(*self._handler_args(call))
            else:
                result = await self._loop.run_in_executor(
                    None, lambda: self._call_handler(call)
                )
        except Exception as exc:
            self._error_or_propagate(call, exc)  # may raise if propagate
            return
        self._set_result(call, result)

    def _serve_sync(self, call: _RPCCall) -> None:
        """Run one call's handler in the current thread and fill its reply."""
        if call.handler is None:
            call.error = f"Unknown endpoint: {call.endpoint}"
            return
        try:
            if asyncio.iscoroutinefunction(call.handler):
                # Async handler is loop-bound — run there, block this thread.
                result = asyncio.run_coroutine_threadsafe(
                    call.handler(*self._handler_args(call)), self._loop
                ).result()
            else:
                result = self._call_handler(call)
        except Exception as exc:
            self._error_or_propagate(call, exc)  # may raise → thread exits
            return
        self._set_result(call, result)

    def _handler_args(self, call: _RPCCall) -> tuple:
        """Positional args for the handler: typed msg if expects registered,
        else the raw frame args (legacy ``@decorate_serdes`` path)."""
        if call.msg is not None:
            return (call.msg,)
        return call.args

    def _call_handler(self, call: _RPCCall) -> Any:
        """Invoke a sync handler with typed msg or raw args (legacy path)."""
        assert call.handler is not None
        return call.handler(*self._handler_args(call), **call.kwargs)

    def _set_result(self, call: _RPCCall, result: Any) -> None:
        """Serialize the result (to_dict) if the endpoint declared returns."""
        returns = self._registry.returns_for(call.endpoint)
        if returns is not None and isinstance(result, Message):
            result = _to_dict(result)
        call.result = result

    def _error_or_propagate(self, call: _RPCCall, exc: Exception) -> None:
        """Handler raised: error-reply by default (fills call.error); propagate
        if ``propagate_exceptions`` (re-raises → the worker loop exits)."""
        if self._propagate_exceptions:
            raise exc
        logger.exception("Error in handler '%s'", call.endpoint)
        call.error = f"{type(exc).__name__}: {exc}"

    async def _output_drain_loop(self) -> None:
        """IO layer: drain output_queue → frame → zmq send (loop-only)."""
        while self._running:
            try:
                call = await self._output_queue.get()
            except asyncio.CancelledError:
                break
            if call is None:
                continue
            if call.error is not None:
                header, body = make_reply_frame(
                    msg_id=call.msg_id,
                    src_node=self.node_id,
                    dst_node=call.src,
                    result=None,
                    error=call.error,
                )
            else:
                header, body = make_reply_frame(
                    msg_id=call.msg_id,
                    src_node=self.node_id,
                    dst_node=call.src,
                    result=call.result,
                    shm_threshold=self._shm_threshold,
                )
            self._send_reply(call.routing_frames, header, body, call.src, call.src_addr)
            _dbg("RPC reply: to=%s endpoint=%s", call.src, call.endpoint)

    @property
    def pending_count(self) -> int:
        """Number of incoming RPCs waiting to be processed."""
        return self._input_queue.qsize()

    # -- public RPC ----------------------------------------------------------

    async def call(
        self,
        target_node: str,
        endpoint: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call an RPC endpoint on another node."""
        t_init = time.monotonic() if _COMMS_TIMING else 0

        address = self._resolve(target_node)
        if address is None:
            raise RpcError(f"Unknown node: {target_node}")

        msg_id = uuid.uuid4().hex
        from whirlwind.whirlcoms.transport import make_request_frame

        header, payload, _cleanup = make_request_frame(
            msg_id=msg_id,
            src_node=self.node_id,
            dst_node=target_node,
            endpoint=endpoint,
            args=args,
            kwargs=kwargs,
            shm_threshold=self._shm_threshold,
            src_addr=self._rpc_endpoint,
        )

        t_send = time.monotonic() if _COMMS_TIMING else 0
        if _COMMS_TIMING:
            self._pending_timing[msg_id] = {
                "target_node": target_node,
                "endpoint": endpoint,
                "t_init": t_init,
                "t_send": t_send,
            }

        sock = self._get_client_socket(address)
        future = self._loop.create_future()
        self._pending[msg_id] = future

        try:
            await sock.send_multipart([header, payload])
            _dbg("RPC send: to=%s endpoint=%s msg_id=%s", target_node, endpoint, msg_id)
            return await asyncio.wait_for(future, timeout=self._rpc_timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(msg_id, None)
            self._pending_timing.pop(msg_id, None)
            if self._fail_on_timeout:
                raise RpcError(
                    f"RPC call to {target_node}/{endpoint} "
                    f"timed out after {self._rpc_timeout}s"
                ) from exc
            logger.warning(
                "RPC call to %s/%s timed out after %.1fs — returning None",
                target_node,
                endpoint,
                self._rpc_timeout,
            )
            return None

    async def call_typed(
        self,
        target_node: str,
        endpoint: str,
        returns: type,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Typed RPC call with a None-guard at the client seam.

        Like :meth:`call` but reconstructs the typed result via
        ``returns.from_dict`` and — crucially — raises :class:`RpcError` if the
        call returned ``None`` (timeout / unreachable when
        ``fail_on_timeout`` is off). This closes the live crash at
        ``training_grpo.py:849`` where ``_from_dict(RolloutResult, None)``
        raised an ``AttributeError`` that escaped ``except RpcError`` and killed
        the run. With this seam the same timeout becomes an ``RpcError`` the
        orchestrator already handles.

        Migration is incremental: callers can adopt ``call_typed`` one at a
        time; the legacy ``call`` + manual ``_from_dict`` path is unchanged.
        """
        # Client-side serdes at the seam: to_dict any outbound Message args
        # before they hit the wire (msgpack can't serialize dataclass objects).
        ser_args = tuple(
            _to_dict(a) if isinstance(a, Message) else a for a in args
        )
        ser_kwargs = {
            k: (_to_dict(v) if isinstance(v, Message) else v)
            for k, v in kwargs.items()
        }
        data = await self.call(target_node, endpoint, *ser_args, **ser_kwargs)
        if data is None:
            raise RpcError(
                f"RPC call to {target_node}/{endpoint} returned no result "
                f"(timed out after {self._rpc_timeout}s or unreachable)"
            )
        return _from_dict(returns, data)

    def call_bg_no_return(
        self,
        target_node: str,
        endpoint: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Fire-and-forget RPC call — does NOT block the caller.

        Spawns a background task that sends the request and drops the reply.
        Silently skipped if the target node isn't in the topology.

        Synchronous by design — safe to call from both sync and async handlers
        without ``await``.

        Use this for best-effort work inside RPC handlers: logging metrics,
        flushing buffers, triggering downstream events, etc.

        Usage (inside an RPC handler)::

            def _generate(self, data):
                # ... do work ...
                self._node.call_bg_no_return(
                    "tracker",
                    "log_metrics",
                    {"tokens_generated": 512},
                )
                return result  # returned immediately, logging runs in bg
        """
        if target_node not in self._topology:
            return

        address = self._resolve(target_node)
        if address is None:
            return

        msg_id = uuid.uuid4().hex
        from whirlwind.whirlcoms.transport import make_request_frame

        header, payload, _cleanup = make_request_frame(
            msg_id=msg_id,
            src_node=self.node_id,
            dst_node=target_node,
            endpoint=endpoint,
            args=args,
            kwargs=kwargs,
            shm_threshold=self._shm_threshold,
            src_addr=self._rpc_endpoint,
        )

        # Push onto thread-safe queue; drain loop sends asynchronously.
        self._bg_send_queue.put_nowait((header, payload, target_node, endpoint))

    async def _send_bg_no_return(
        self,
        sock: zmq.asyncio.Socket,
        header: bytes,
        payload: bytes,
        target_node: str,
        endpoint: str,
    ) -> None:
        """Best-effort send for background RPC — never raises."""
        try:
            await sock.send_multipart([header, payload])
            _dbg("BG RPC send: to=%s endpoint=%s", target_node, endpoint)
        except Exception:
            _dbg("BG RPC send failed: to=%s endpoint=%s", target_node, endpoint)

    async def _bg_send_drain_loop(self) -> None:
        """Drain thread-safe queue and send frames asynchronously."""
        while self._running:
            try:
                header, payload, target_node, endpoint = (
                    self._bg_send_queue.get_nowait()
                )
                address = self._resolve(target_node)
                if address is None:
                    self._bg_send_queue.task_done()
                    continue
                sock = self._get_client_socket(address)
                try:
                    await sock.send_multipart([header, payload])
                    _dbg("BG RPC send: to=%s endpoint=%s", target_node, endpoint)
                except Exception:
                    _dbg("BG RPC send failed: to=%s endpoint=%s", target_node, endpoint)
                finally:
                    self._bg_send_queue.task_done()
            except queue.Empty:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break

    # -- internal ------------------------------------------------------------

    def _resolve(self, target_node: str) -> str | None:
        """Look up a node's address in the local topology cache."""
        info = self._topology.get(target_node)
        return info.get("address") if info else None

    def _get_client_socket(self, address: str) -> zmq.asyncio.Socket:
        """Get or create a transient DEALER socket connected to a peer."""
        assert self._ctx is not None
        if address not in self._client_sockets:
            sock = self._ctx.socket(zmq.DEALER)
            sock.connect(address)
            self._client_sockets[address] = sock
        return self._client_sockets[address]

    async def _orchestrator_request(
        self, cmd: dict, timeout: float = 2.0
    ) -> bytes | None:
        """Send a request to the orchestrator and optionally receive a reply.

        Uses a REQ socket with explicit timeout via asyncio shield + poll.
        Returns the reply bytes, or None if the orchestrator is unreachable.
        """
        assert self._ctx is not None
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._ctrl_endpoint)

        msg = msgpack.packb(cmd, use_bin_type=True)
        poller = zmq.asyncio.Poller()
        poller.register(sock, zmq.POLLOUT)

        try:
            events = dict(
                await asyncio.wait_for(
                    poller.poll(int(timeout * 1000)), timeout=timeout
                )
            )
            if sock not in events:
                sock.close()
                return None

            sock.send(msg, zmq.NOBLOCK)

            poller.register(sock, zmq.POLLIN)
            events = dict(
                await asyncio.wait_for(
                    poller.poll(int(timeout * 1000)), timeout=timeout
                )
            )
            if sock not in events:
                sock.close()
                return None

            reply = await sock.recv(zmq.NOBLOCK)
            sock.close()
            return reply  # type: ignore[return-value]
        except (asyncio.TimeoutError, zmq.Again):
            sock.close()
            return None
        except Exception:
            sock.close()
            return None

    async def _register_with_orchestrator(self) -> None:
        """Send a register command to the orchestrator (single attempt, best-effort).

        If NODE_COMMS_ENSURE_REGISTER is set, blocks until registration succeeds
        or retries are exhausted (exponential backoff, capped at 10s between attempts).
        """
        if _ENSURE_REGISTER:
            await self._register_and_wait()
            return

        reply = await self._orchestrator_request(
            {
                "cmd": "register",
                "node_id": self.node_id,
                "address": self._rpc_endpoint,
                "endpoints": list(self._handlers.keys()),
            }
        )
        if reply is None:
            logger.warning("Node '%s': orchestrator not reachable", self.node_id)

    async def _register_and_wait(self, max_retries: int = 12) -> None:
        """Register with the orchestrator, retrying with exponential backoff.

        Blocks until registration succeeds or max_retries is exhausted.
        This handles the race where the node starts before the orchestrator.
        """
        for attempt in range(1, max_retries + 1):
            reply = await self._orchestrator_request(
                {
                    "cmd": "register",
                    "node_id": self.node_id,
                    "address": self._rpc_endpoint,
                    "endpoints": list(self._handlers.keys()),
                }
            )
            if reply is not None:
                logger.debug(
                    "Node '%s' registered with orchestrator (attempt %d/%d)",
                    self.node_id,
                    attempt,
                    max_retries,
                )
                return
            backoff = min(2**attempt, 10)  # 2, 4, 8, 10, 10...
            logger.warning(
                "Node '%s': orchestrator not reachable (attempt %d/%d, retry in %.1fs)",
                self.node_id,
                attempt,
                max_retries,
                backoff,
            )
            await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Node '{self.node_id}': could not reach orchestrator after {max_retries} attempts"
        )

    async def _deregister(self) -> None:
        """Send a deregister command to the orchestrator."""
        await self._orchestrator_request(
            {
                "cmd": "deregister",
                "node_id": self.node_id,
            },
            timeout=1.0,
        )

    async def _refresh_topology(self) -> None:
        """Re-query the orchestrator for the full topology and re-register."""
        reply = await self._orchestrator_request({"cmd": "list"})
        if reply:
            data = msgpack.unpackb(reply, raw=False)
            if data.get("ok"):
                nodes = data.get("nodes", {})
                self._topology.clear()
                for nid, info in nodes.items():
                    if nid != self.node_id:
                        self._topology[nid] = info
                logger.debug("Topology refreshed: %d nodes", len(self._topology))
        else:
            logger.warning("Node '%s': topology refresh failed", self.node_id)

        await self._register_with_orchestrator()

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeats to the orchestrator.

        A sustained run of missed heartbeats means the orchestrator is gone
        (crashed trainer, torn-down run, dead IPC). Such a node can never
        receive further work or a shutdown broadcast, so after
        WHIRLWIND_HEARTBEAT_FATAL_MISSES consecutive misses (default 12 ≈
        80s at the 5s interval + 2s RPC timeout) it shuts itself down
        instead of logging "heartbeat missed" forever (the pre-fix behaviour
        left service nodes — and their sglang engine workers — hanging for
        hours after a completed or crashed run).
        """
        assert self._ctx is not None
        fatal_misses = int(
            os.environ.get("WHIRLWIND_HEARTBEAT_FATAL_MISSES", "12")
        )
        miss_streak = 0
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            reply = await self._orchestrator_request(
                {
                    "cmd": "heartbeat",
                    "node_id": self.node_id,
                }
            )
            if reply is not None:
                miss_streak = 0
                try:
                    known = msgpack.unpackb(reply, raw=False).get("ok", True)
                except Exception:  # any reply proves the registry is alive; only a nack re-registers
                    known = True
                if not known:
                    # The registry dropped us (heartbeat timeout during a stall):
                    # re-register so peers can route replies to us again.
                    logger.warning(
                        "Node '%s': registry no longer knows us — re-registering",
                        self.node_id,
                    )
                    await self._register_with_orchestrator()
                continue
            miss_streak += 1
            logger.warning(
                "Node '%s': heartbeat missed (%d consecutive)",
                self.node_id,
                miss_streak,
            )
            if miss_streak < fatal_misses:
                continue
            logger.error(
                "Node '%s': orchestrator unreachable after %d consecutive "
                "missed heartbeats — shutting this node down",
                self.node_id,
                miss_streak,
            )
            # stop() cancels THIS task too, so it must run as a fresh task;
            # if the graceful path does not finish quickly, force the exit
            # (the process is useless without the orchestrator, and a hard
            # exit is the only reliable way out of e.g. a wedged NCCL call).
            stop_task = asyncio.get_running_loop().create_task(self.stop())
            try:
                await asyncio.wait_for(asyncio.shield(stop_task), timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            if not stop_task.done() or self._running:
                logger.error(
                    "Node '%s': graceful stop incomplete — forcing process exit",
                    self.node_id,
                )
            os._exit(1)

    async def _recv_loop(self) -> None:
        """IO layer: async ZMQ receive loop.

        On a request frame: parse it, deserialize the payload
        (``from_dict(expects)`` when the endpoint declared a contract), build
        a :class:`_RPCCall`, and push it to :attr:`input_queue` for
        :meth:`process_rpc_requests` to consume. On a reply frame: resolve the
        pending client future via :meth:`_handle_reply`. zmq sockets stay
        loop-only; no handler runs here.
        """
        assert self._dealer is not None

        while self._running:
            try:
                parts = await self._dealer.recv_multipart()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error receiving RPC message")
                continue

            if not parts:
                continue

            try:
                header = msgpack.unpackb(parts[0], raw=False)
                if "endpoint" in header:
                    item = self._build_call(parts)
                    if item is not None:
                        self._input_queue.put_nowait(item)
                elif "msg_id" in header and "src" in header:
                    await self._handle_reply(parts)
                else:
                    logger.debug("Unknown message type, ignoring")
            except Exception:
                logger.exception("Error parsing RPC message")

    def _build_call(self, parts: list[bytes]) -> _RPCCall | None:
        """Parse a request frame into an _RPCCall, deserializing at the seam.

        If the endpoint declared ``expects``, the payload is reconstructed via
        ``from_dict(expects)`` and carried as ``call.msg`` (typed). Otherwise
        the raw frame args/kwargs are carried (legacy ``@decorate_serdes``
        handlers do their own from_dict). Either way the worker just calls the
        handler; serdes lives in whirlcoms, not in the handler.
        """
        routing_frames: list[bytes] = []
        header_idx = 0
        for i, part in enumerate(parts):
            try:
                h = msgpack.unpackb(part, raw=False)
                if isinstance(h, dict) and ("endpoint" in h or "msg_id" in h):
                    header_idx = i
                    break
            except Exception:
                continue
            routing_frames.append(part)

        frame = parse_request_frame(parts[header_idx : header_idx + 2])
        endpoint = frame["endpoint"]
        msg_id = frame["msg_id"]
        src = frame["src"]
        payload = frame["payload"]
        args = payload.get("args", ())
        kwargs = payload.get("kwargs", {})

        _dbg("RPC recv: from=%s endpoint=%s msg_id=%s", src, endpoint, msg_id)
        handler = self._handlers.get(endpoint)

        # Serdes at the seam: if the endpoint declared an expects type,
        # deserialize the (single) message argument now. The handler will
        # receive a typed object, not a raw dict.
        typed_msg: Any = None
        expects = self._registry.expects_for(endpoint)
        if expects is not None:
            try:
                # The request carries the message as the first positional arg
                # or under the ``data`` kwarg (decorator convention).
                raw = args[0] if args else kwargs.get("data")
                if raw is not None:
                    typed_msg = _from_dict(expects, raw)
            except Exception:
                logger.exception(
                    "Error deserializing expects=%s for '%s'", expects, endpoint
                )
                typed_msg = None

        return _RPCCall(
            msg_id=msg_id,
            src=src,
            endpoint=endpoint,
            handler=handler,
            routing_frames=routing_frames,
            msg=typed_msg,
            args=tuple(args),
            kwargs=dict(kwargs),
            src_addr=frame.get("src_addr"),
        )


    def _send_reply(
        self,
        routing: list[bytes],
        header: bytes,
        body: bytes,
        dst_node: str,
        dst_addr: str | None = None,
    ) -> None:
        """Send a reply frame to the target node. Silently drops if unreachable.

        Prefers the address the request carried (``src_addr`` header) over the
        topology cache, so a caller the registry dropped still gets its reply;
        legacy requests without it fall back to the cache.
        """
        import contextlib

        dst_addr = dst_addr or self._topology.get(dst_node, {}).get("address")
        if not dst_addr:
            return

        if dst_addr not in self._client_sockets:
            assert self._ctx is not None
            sock = self._ctx.socket(zmq.DEALER)
            sock.connect(dst_addr)
            self._client_sockets[dst_addr] = sock
        else:
            sock = self._client_sockets[dst_addr]

        with contextlib.suppress(zmq.Again, zmq.ZMQError):
            sock.send_multipart([header, body], zmq.NOBLOCK | zmq.DONTWAIT)

    async def _handle_reply(self, parts: list[bytes]) -> None:
        """Handle an incoming reply to our RPC call."""
        from whirlwind.whirlcoms.transport import parse_reply_frame

        t_recv = time.monotonic() if _COMMS_TIMING else 0
        msg_id = None
        if _COMMS_TIMING:
            try:
                raw_header = msgpack.unpackb(parts[0], raw=False)
                msg_id = raw_header.get("msg_id")
            except Exception:
                pass

        result, error = parse_reply_frame(parts)
        t_deser = time.monotonic() if _COMMS_TIMING else 0

        header = msgpack.unpackb(parts[0], raw=False)
        msg_id = msg_id or header["msg_id"]

        future = self._pending.pop(msg_id, None)
        _dbg(
            "RPC recv reply: from=%s msg_id=%s error=%s",
            header.get("src_node"),
            msg_id,
            bool(error),
        )

        if _COMMS_TIMING and msg_id in self._pending_timing:
            timing = self._pending_timing.pop(msg_id)
            _timing_buf.append(
                {
                    "target_node": timing["target_node"],
                    "endpoint": timing["endpoint"],
                    "t_init": timing["t_init"],
                    "t_send": timing["t_send"],
                    "t_recv": t_recv,
                    "t_deserialize": t_deser,
                    "latency_ms": round((t_deser - timing["t_init"]) * 1000, 3),
                }
            )

        if future and not future.done():
            if error:
                future.set_exception(RpcError(error))
            else:
                future.set_result(result)

    async def _topology_loop(self) -> None:
        """Listen for topology broadcast events from the orchestrator."""
        while self._running:
            try:
                msg = await self._sub.recv()
                event = msgpack.unpackb(msg, raw=False)
                event_type = event.get("event")
                node_id = event.get("node_id")

                if event_type == "registered":
                    self._topology[node_id] = {
                        "address": event["address"],
                        "endpoints": event.get("endpoints", []),
                    }
                    logger.debug("Topology: node '%s' registered", node_id)
                elif event_type == "deregistered":
                    self._topology.pop(node_id, None)
                    logger.debug("Topology: node '%s' deregistered", node_id)
                elif event_type == "shutdown":
                    reason = event.get("reason", "unspecified")
                    logger.warning(
                        "Node '%s' received cluster shutdown (reason=%s) — exiting",
                        self.node_id,
                        reason,
                    )
                    self._running = False
                    if self._on_shutdown is not None:
                        try:
                            res = self._on_shutdown()
                            if inspect.isawaitable(res):
                                self._loop.create_task(res)
                        except Exception:
                            logger.exception("shutdown callback failed")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in topology loop")

    @property
    def topology(self) -> dict[str, dict]:
        """Return a copy of the current topology cache."""
        return dict(self._topology)

    @property
    def endpoints(self) -> list[str]:
        """List of exposed endpoint names."""
        return list(self._handlers.keys())

    @property
    def is_running(self) -> bool:
        """Whether the node is still running (False after stop() or shutdown)."""
        return self._running

    def set_shutdown_callback(self, cb: Callable[[], Any]) -> None:
        """Register a callback fired when a cluster "shutdown" event arrives.

        The training orchestrator uses this to break out of its long-running
        ``run()`` loop. Worker nodes do not need it: their ``__main__`` loops
        poll ``is_running`` and exit when it goes False.
        """
        self._on_shutdown = cb

    async def broadcast_shutdown(self, reason: str = "unspecified") -> dict | None:
        """Ask the orchestrator registry to broadcast a cluster-wide shutdown.

        Sends the ``shutdown_all`` command over the control REQ socket. The
        orchestrator then publishes a ``shutdown`` event on its PUB socket,
        which every node's ``_topology_loop`` receives → ``is_running`` goes
        False → graceful exit. Best-effort: returns None if the orchestrator
        is unreachable (it may itself be the failing component).
        """
        return await self._orchestrator_request(
            {"cmd": "shutdown_all", "node_id": self.node_id, "reason": reason}
        )


class RpcError(Exception):
    """Raised when an RPC call fails."""
