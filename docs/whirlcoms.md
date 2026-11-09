# whirlcoms — ZeroMQ RPC Layer

P2P RPC over ZeroMQ DEALER sockets with a central service registry. This doc
covers both the base layer (framing, transport, discovery) and the **v2 node
seam** (typed serdes at the seam, two-queue worker split) that is now the
as-built design.

## Overview

`whirlcoms` provides peer-to-peer RPC: each node binds a DEALER socket,
registers with the orchestrator (discovery-only), and processes requests.
All data RPC is direct P2P full-mesh; the orchestrator never relays data.

```python
node = Node("worker", auto_process=True)

@node.expose("greet")
def greet(name: str) -> str:
    return f"hello {name}"

await node.start()
# ... run until done ...
await node.stop()
```

**Processing modes**:
- `auto_process=True` — background task drains the queue (spawns one
  `process_rpc_requests` worker; parity with the legacy auto-drain)
- `auto_process=False` — user drives the worker loop via
  `process_rpc_requests()` (async) / `process_rpc_requests_sync()` (thread)

## The v2 node seam (as built)

### Goals

1. **Keep RPC obligations alive during heavy callbacks.** A long
   `rollout.generate` previously made the node deaf to `ping`/`update_weights`
   (one-request-at-a-time loop). Control-plane RPCs are now answered while a
   data-plane handler is in flight.
2. **Typed serdes at the seam, owned by the Node.** Previously serdes was
   split: per-handler decorators server-side, manual `_from_dict` at 14+
   client call sites with no None/error guard (the live-crash pattern,
   §`call_typed` below).
3. **Behavior preservation by default.** A handler with no typed contract
   behaves exactly as before (raw args in / raw result out — the legacy
   `@decorate_serdes` path).
4. **User-driven execution.** The Node owns IO + queues + serdes; users drive
   the worker loop. Concurrency = "spawn N tasks/threads each calling the
   loop." No runtime/strategy classes.

### Boundary

```
whirlcoms/
  ├── message.py    ← MessageRegistry (endpoint → expects/returns);
  │                   re-exports Message/BatchedMessage/_to_dict/_from_dict
  └── node.py       ← _RPCCall (private), IO layer, the two queues,
                      process_rpc_requests / _sync, expose, call_typed

app layer (users)
  ├── messages.py   ← PromptBatch, RolloutResult, TrainStepRequest, …
  └── rollout/...   ← @node.expose("generate", expects=PromptBatch, returns=RolloutResult)
```

The *mechanism* (serdes, queues, dispatch) lives in `whirlcoms/`; the *domain
message types* stay with users. There is no `runtime.py` — worker execution
lives directly on the Node.

### The data path (one request)

```
 peer            loop thread (IO layer, never blocks)               worker
 ──── ─────────────────────────────────────────────────────────── ───────────
  send ─► [1] _recv_loop: recv_multipart
           [2] _build_call: parse + from_dict(expects)   ◄── serdes at the seam
           [3] _RPCCall{msg_id, src, endpoint, msg(TYPED), ...}
           [4] input_queue.put(call)
                ────────────────────────────────────────────────────────────►
                                                                 [5] process_rpc_requests:
                                                                     call = input_queue.get()
                                                                     result = handler(call.msg)
                                                                 [6] call.result = to_dict(result)
                                                                     output_queue.put(call)
                ◄────────────────────────────────────────────────────────────
           [7] _output_drain_loop: output_queue.get()
           [8] make_reply_frame(call.result or call.error)
           [9] _send_reply ──send──► zmq DEALER           (loop-only, safe)
  ◄──reply──
```

**Invariant:** ZMQ sockets and `_send_reply` are touched *only* on the loop
thread; only the handler call is offloaded. The IO layer (recv, output-drain,
topology, heartbeat) never runs handlers and never blocks.

### Two queues, one call object

- **`input_queue`** — IO → worker; one `_RPCCall` per incoming RPC, request
  already deserialized (`call.msg` typed when the endpoint declared `expects`).
- **`output_queue`** — worker → IO; the *same* `_RPCCall` with
  `result`/`error` filled.

`_RPCCall` is a single private dataclass carrying a call's whole lifecycle —
no separate WorkItem/ReplyItem types; nothing carried-but-unused.

### The message registry

```python
@node.expose("generate", expects=PromptBatch, returns=RolloutResult)
async def generate(batch: PromptBatch) -> RolloutResult: ...   # typed, no decorator
```

The registry maps endpoint → (`expects`, `returns`) — its **entire** live
surface (`register_endpoint`/`expects_for`/`returns_for`), driving
`from_dict` on recv and `to_dict` on reply. Handlers without a contract
(`@node.expose("ping")`) get raw `args` — the unchanged legacy path, so all
existing `@decorate_serdes` handlers keep working; migration is per-handler
and optional.

### The crash fix — `call_typed` (client seam)

```python
result: RolloutResult = await node.call_typed("rollout", "generate", RolloutResult, batch)
```

`call_typed` `to_dict`s outbound `Message` args, sends, does `from_dict` on
the reply, and **raises `RpcError` if the call returned `None`** (timeout /
unreachable). This closes the pattern where `_from_dict(RolloutResult, None)`
raised `AttributeError` that escaped `except RpcError` and killed a long
run. The orchestrator constructs its own `Node` with `fail_on_timeout=True`,
so its untyped `call` raises `RpcError` on a timeout as well.

### Design history (what was deliberately removed)

- **No `runtime.py` / `WorkerRuntime` strategy hierarchy** — became vestigial
  after the user-driven loop redesign; deleted. Sync/async/thread execution
  is just `process_rpc_requests` (+caller-spawned tasks/threads).
- **No `MessageMeta`/`TransportType` per-message property set** — designed,
  built, found carried-but-never-consumed (priority, timeout_s, exclusive,
  max_concurrency all unread), removed. Re-add *inline* when a real run
  proves the need, not as an upfront abstraction.

## Orchestrator (discovery)

Central service registry (REP + PUB). Nodes register address + endpoints;
topology changes broadcast on PUB.

```python
orch = Orchestrator()
await orch.start()
orch.get_node("worker")    # → {"address": "ipc:///tmp/whirlwind-worker", "endpoints": [...]}
orch.list_nodes()
```

Heartbeats every 5s; nodes stale > 15s are dropped and deregistration is
broadcast.

## RpcClient (retry wrapper)

```python
rpc = RpcClient(node, max_retries=3, timeout=30.0)
result = await rpc.call("worker", "greet", "world")
```

**Note:** the orchestrator does not use it — it calls `node.call` directly,
so there is no retry in the data path.

## Message framing

- **Request** `[header, payload]`: header `{msg_id, src, dst, endpoint}`;
  payload `{args, kwargs}` (msgpack).
- **Reply** `[header, payload]`: header `{msg_id, src, dst, error}`; payload
  `{result}`.

## Tensor transport

| Size | Method |
|------|--------|
| ≤ 64 KB | inline pickle in the msgpack frame |
| > 64 KB | write `/dev/shm/whirlwind-shm-<uuid>`, send metadata only; receiver reads via `torch.frombuffer()` and **unlinks** (receiver-owned cleanup) |

⚠ SHM files leak if the receiver dies before reading; sweep `/dev/shm/whirlwind-shm-*` after a crash.

## IPC addresses

| Component | Pattern |
|-----------|---------|
| Orchestrator REP | `ipc:///tmp/whirlwind-ctrl` |
| Orchestrator PUB | `ipc:///tmp/whirlwind-events` |
| Node DEALER | `ipc:///tmp/whirlwind-<node_id>` |

## Debug environment variables

| Variable | Effect |
|----------|--------|
| `NODE_COMMS_DBG_STDOUT=1` | print RPC comms to stderr live |
| `NODE_COMMS_ENSURE_REGISTER=1` | block node startup until registration accepted |
| `NODE_COMMS_TIMING=1` | track RPC latencies (last 1000) |
| `NODE_COMMS_RPC_FAIL_TIMEOUT=1` | `call` raises `RpcError` on timeout instead of returning `None` |
| `WHIRLWIND_RPC_TIMEOUT` | RPC timeout, default 300s |

## Error handling

`RpcError` on: unknown node, timeout (300s default, returns `None` instead
unless `NODE_COMMS_RPC_FAIL_TIMEOUT=1`), remote-handler exception (forwarded).
Background calls (`call_bg_no_return`) silently drop if unreachable.

## Tests

- `tests/test_whirlcoms_v2.py` — the seam: ping-during-generate, typed
  round-trip, sync worker off-loop, exception propagation, `auto_process`
  shim, `call_typed` None→`RpcError`.
- `tests/test_whirlcoms_extra.py` — 44 parity tests (refactor is
  behavior-preserving).
- Per-test hard timeout (`WHIRLWIND_TEST_TIMEOUT`, default 100s).
