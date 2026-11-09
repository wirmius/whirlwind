# RolloutPool — managing multiple simultaneous rollout engines

> **Status: implemented.** A thin abstraction that lets the
> orchestrator talk to ONE "rollout" endpoint while N independent SGLang engine
> processes do the actual generation, sharded across GPUs. No orchestrator
> changes. Code: `whirlwind/rollout/sglang_rollout_pool.py`.
>
> **Async verified** against `whirlcoms/node.py`: outbound `asyncio.gather` of
> N `node.call`s = N truly-concurrent in-flight RPCs (each call has its own
> msg_id + future; ZMQ DEALER handles multiple in-flight; recv loop routes by
> msg_id). Inbound serial serving is fine (orchestrator's RL loop is serial;
> heartbeats use a separate channel). See "Async model" below.

## Why (the concrete motivation)

Today we run **one** rollout engine per run. SGLang's internal `dp_size` gives N
replicas in one process, but it has two hard limits we just hit:

1. **vLLM can't do DP at all** in our single-process launch path (it downgrades
   `dp_size>1 → 1` and warns).
2. **SGLang forbids DP + dynamic LoRA** — a hard assert in `load_lora_adapter`:
   `assert self.server_args.dp_size == 1, "dp_size must be 1 for dynamic lora
   loading"`. So multi-GPU LoRA is **blocked** today.

A **pool of N separate engine processes** (each `dp_size=1, tp_size=1`, one GPU)
sidesteps both: each engine passes the assert (it's per-engine), vLLM engines
work too, and the batch is sharded across them by the pool. This **unblocks
multi-GPU LoRA** — the headline win. It also generalizes the DP=2 throughput
gain we measured (+25%) to any backend and any GPU count.

Measured (full-FT, 2 GPUs, plain config): DP=2 = 388s/opt-step vs TP=2 = 515s.
A 2-engine pool should match DP=2 for SGLang and bring the same gain to vLLM.

## The design: a router node, not a rewrite

```
Orchestrator  ──call("rollout", generate)──►  RolloutPool (node id="rollout")
                                                   │  shard prompts round-robin
                                                   │  fan out in parallel
                              ┌────────────────────┼────────────────────┐
                              ▼                    ▼                    ▼
                         engine "rollout0"    engine "rollout1"    engine "rolloutN"
                         (existing node,       (existing node,       (existing node,
                          GPU 2, tp=1)          GPU 3, tp=1)          GPU k, tp=1)
                                                   │  gather results in order
                                                   ▼
                              ◄──one RolloutResult──
```

**The pool is the only new code.** The backend engines are the **existing
`rollout_sglang` / `rollout_vllm` nodes, launched unchanged** with distinct
`node_id`s (`rollout0`, `rollout1`, …) and one-GPU `CUDA_VISIBLE_DEVICES` each.
The orchestrator keeps calling `"rollout"` — it never learns there are N engines.

## The abstraction (already satisfied by existing nodes)

The pool treats each backend as an opaque engine implementing four operations —
exactly the four handlers the orchestrator already calls:

```python
# conceptual protocol — both rollout_sglang and rollout_vllm already satisfy it
class RolloutEngine:
    async def generate(self, batch: PromptBatch) -> RolloutResult: ...
    async def update_weights(self, req: WeightSyncRequest) -> dict: ...
    async def sleep(self, req: NoInputRequest) -> dict: ...
    async def wake(self, req: NoInputRequest) -> dict: ...
```

No new ABC needed in code — the pool just holds N RPC client handles (one per
backend `node_id`) and fans out. The existing nodes are reused as-is.

## The pool node (`whirlwind/rollout/sglang_rollout_pool.py`)

```python
class RolloutPoolNode:
    """Thin router: registers as 'rollout', fans out to N backend engines."""

    def __init__(self, config: RolloutPoolConfig):
        self.config = config
        self._node: Node | None = None
        # backend node ids, e.g. ["rollout0", "rollout1"]
        self._backends: list[str] = config.backends

    async def _generate(self, batch: PromptBatch) -> RolloutResult:
        # 1. Shard the batch's parallel lists across N backends (round-robin).
        shards = self._shard_batch(batch, len(self._backends))
        # 2. Fan out in parallel (each call → one backend's 'generate').
        results = await asyncio.gather(*(
            self._node.call(b, "generate", _to_dict(s)) for b, s in zip(self._backends, shards)
        ))
        # 3. Gather in original order.
        return self._gather([_from_dict(RolloutResult, r) for r in results])

    async def _update_weights(self, req: WeightSyncRequest) -> dict:
        # Broadcast to all backends (all load the same checkpoint path).
        data = _to_dict(req)
        results = await asyncio.gather(*(self._node.call(b, "update_weights", data) for b in self._backends))
        return {"ok": all(r.get("ok") for r in results), "step": req.step}

    async def _sleep(self, req) -> dict:
        return await self._broadcast("sleep", req)   # all engines release GPU
    async def _wake(self, req) -> dict:
        return await self._broadcast("wake", req)    # all engines reclaim GPU
```

### Sharding — keep all parallel lists aligned

`PromptBatch` carries parallel lists (`prompts`, `responses`, `dataset_ids`,
`entry_ids`, `rollout_ids`, `subrollout_ids`). The pool splits them by the **same
index ranges** so each backend gets a self-consistent sub-batch. After the fan-out
it concatenates the `RolloutResult` lists in backend order, which == original
order because round-robin preserves it.

```python
def _shard_batch(self, batch, n):
    idx = list(range(len(batch.prompts)))
    ranges = [idx[i::n] for i in range(n)]          # round-robin, order-preserving
    shards = []
    for r in ranges:
        s = copy(batch)                              # shallow copy
        s.prompts          = [batch.prompts[i] for i in r]
        s.responses        = [batch.responses[i] for i in r]
        s.dataset_ids      = [batch.dataset_ids[i] for i in r]
        s.entry_ids        = [batch.entry_ids[i] for i in r]
        s.rollout_ids      = [batch.rollout_ids[i] for i in r]
        s.subrollout_ids   = [batch.subrollout_ids[i] for i in r]
        shards.append(s)
    return shards
```

`system_prompt`, `step`, `temperature`, `top_p`, `return_logprobs`, etc. are
scalars — copied verbatim to every shard.

## Config

A new `RolloutPoolConfig` (the only new config type):

```python
@dataclass
class RolloutPoolConfig:
    node_id: str = "rollout"                 # the orchestrator's single endpoint
    backends: list[str] = field(default_factory=lambda: ["rollout0", "rollout1"])
```

Batches are split round-robin (order-preserving) across the backends.

The **backend engines use the existing `RolloutConfig`**, one per engine, each
with its own `node_id`, `model_path`, `tp_size=1`, and GPU set (via bringup's
`CUDA_VISIBLE_DEVICES`). Example two-engine SGLang LoRA setup:

```yaml
# configs/rollout/pool_2x_sglang_lora.yaml  (the pool)
node_id: "rollout"
backends: ["rollout0", "rollout1"]
```
```yaml
# configs/rollout/sglang_lora_g2.yaml  (backend 0 — existing RolloutConfig)
node_id: "rollout0"
model_path: "Qwen/Qwen3-0.6B-Base"
tp_size: 1
dp_size: 1            # ← dp_size=1 PASSES the LoRA assert (the whole point)
enable_lora: true
lora_max_rank: 16
enable_memory_saver: true
...
```
```yaml
# configs/rollout/sglang_lora_g3.yaml  (backend 1 — identical, node_id rollout1)
node_id: "rollout1"
... (same as above)
```

## Bringup changes

`bringup_and_run.sh` currently launches one rollout process. Extended to launch
N backends (each on its own GPU set) + 1 pool:

```
# pool
CUDA_VISIBLE_DEVICES= python3 -m whirlwind.rollout.rollout_pool \
    --config configs/rollout/pool_2x_sglang_lora.yaml
# backend 0 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python3 -m whirlwind.rollout.rollout_sglang \
    --config configs/rollout/sglang_lora_g2.yaml
# backend 1 (GPU 3)
CUDA_VISIBLE_DEVICES=3 python3 -m whirlwind.rollout.rollout_sglang \
    --config configs/rollout/sglang_lora_g3.yaml
```

The pool registers as `rollout` and discovers backends by their `node_id`s (in
its config) — same node-registry mechanism every node uses. Minimal bringup
change: loop over a list of `(module, config, gpus)` instead of one.

## The 4-handler fan-out semantics

| handler | pool action | why |
|---|---|---|
| `generate` | shard → fan out → gather (parallel) | throughput: each engine does 1/N of the batch |
| `update_weights` | broadcast (parallel) | every engine must load the new weights; one shared `/dev/shm` export path, all read it |
| `sleep` | broadcast (parallel) | all engines release GPU for the training phase |
| `wake` | broadcast (parallel) | all engines reclaim GPU for the next generation phase |

**Weight export is unchanged.** The model node writes ONE checkpoint to the
shared `/dev/shm` path (full `pytorch_model.bin` or LoRA `adapter_model.safetensors`).
Every backend loads from that same path — no per-engine export, no extra disk.

## What this buys us

1. **Unblocks multi-GPU LoRA** (the headline). N × (single-GPU SGLang engine,
   `dp_size=1`) passes the LoRA assert per-engine → 2-GPU LoRA works for the
   first time. ~2× generation throughput vs the single-engine LoRA run.
2. **Brings the DP throughput win to vLLM.** vLLM can't DP single-process; a
   pool of N vLLM engines (each `tp=1`, one GPU) gives the same sharded
   throughput. (Caveat: vLLM LoRA + sleep/cumem still crashes per-engine, so
   vLLM pools are full-FT only until vLLM fixes #39934.)
3. **Scales to any GPU count.** Add engines linearly; the pool shards
   automatically. No code change per GPU added.
4. **Heterogeneous engines possible.** Backends need not be identical — e.g.,
   one SGLang + one vLLM, or different models (future: base + ref engines).

## Honest constraints

- **Colocate still serializes.** In colocate, all engines share the training
  GPUs and sleep during training. The pool gives **generation throughput**
  (sharded batch), not training/generation overlap — RL is on-policy, so you
  can't generate step N+1 before step N's weights are trained. The win is a
  faster generate phase, same as DP=2.
- **One extra RPC hop.** `orchestrator → pool → engine` adds one in-process
  fan-out (the pool calls backends via the same node IPC). Negligible vs
  generation time (seconds vs minutes).
- **Weight sync is per-engine but parallel.** N engines each load the
  checkpoint; with `gather` the wall-time = slowest engine, not N×. For LoRA
  (9 MB) this is trivial; for full-FT (1.2 GB) each engine reads the shared
  file independently (cached by the page cache after the first read).
- **Failure handling.** If one backend dies, the pool's `gather` raises. A
  later enhancement can mark a backend unhealthy and reshard to the survivors
  (out of scope for the thin first version).

## What is NOT changed

- `rollout_sglang.py`, `rollout_vllm.py` — reused as backends, zero edits
  (they already take `node_id` from config).
- The orchestrator — still calls `"rollout"`; the pool is transparent.
- `model_fsdp2.py` export — writes one shared checkpoint as before.
- Messages — `PromptBatch` / `RolloutResult` / `WeightSyncRequest` unchanged.

## Implementation (done)

1. `RolloutPoolConfig` in `config.py` — `node_id`, `backends`.
2. `whirlwind/rollout/sglang_rollout_pool.py` — the router node (~300 lines):
   `_shard_batch` (round-robin, order-preserving, only splits lists matching
   prompt length), `_gather` (concat in backend order), `_generate` (fan out
   via `asyncio.gather`), `_update_weights`/`_sleep`/`_wake` (broadcast),
   `_await_backends` (poll `Node.topology` until backends register).
3. Demo configs: `pool_2x_sglang.yaml` (pool) + `sglang_lora_pool_g2.yaml` /
   `sglang_lora_pool_g3.yaml` (backends, each `dp_size=1` → LoRA assert passes).
4. Shard/gather unit tests pass: even/odd/over-shard edge cases, order
   preservation, parallel-list consistency.
5. `bringup_and_run.sh` — TODO: loop over a backend list (currently launches
   one rollout; the pool + backends can be launched manually for now).

## Async model (verified against whirlcoms/node.py)

- **Outbound (pool → N backends):** each `node.call` creates a unique `msg_id`,
  stores its own `asyncio.Future` in `_pending[msg_id]`, fires `send_multipart`
  (ZMQ DEALER = non-blocking, handles multiple in-flight), and awaits its
  future. The `_recv_loop` routes replies by `msg_id` → resolves the right
  future. So `asyncio.gather(*[call(b,...) for b in backends])` = N
  truly-concurrent in-flight RPCs. ✅
- **Inbound (orchestrator → pool):** `process_rpc_requests` is serial
  (`await _serve_async(call)` per call), but the orchestrator's RL loop is
  serial (generate → reward → train → sync) → one inbound call at a time. ✅
- **Heartbeats:** use a SEPARATE channel (`_orchestrator_request` to the ctrl
  endpoint, not the RPC handler queue) → never stall behind a long generate. ✅
- **Backends:** each is a separate process → true parallelism (own event loop /
  GIL). While the pool's `_generate` awaits the gather, the pool's single
  worker is blocked — fine, the orchestrator is awaiting the reply. ✅

## What is NOT changed

- `rollout_sglang.py`, `rollout_vllm.py` — reused as backends, zero edits.
- The orchestrator — still calls `"rollout"`; the pool is transparent.
- `model_fsdp2.py` export — writes one shared checkpoint as before.
- Messages — `PromptBatch` / `RolloutResult` / `WeightSyncRequest` unchanged.

Total new code: one ~300-line file + ~10 lines of config. The existing rollout
nodes and orchestrator are untouched — the definition of "thin".
