# Specification: Agentic-Harness Connector to the Whirlwind Optimization Harness

> **Status: SPEC (v1), for the experimental async proxy-rollout mode
> ([async_proxy_rollout.md](async_proxy_rollout.md)).** This document
> specifies a connector module to be implemented **inside the agentic
> harness** (the "agent"). It is written to be executable by an implementing
> agent with no other context. The server side (the rollout proxy) is
> implemented in this repository; the connector must NOT modify it.
> Reference client: `whirlwind/training/sim_external_agent.py` (in the
> whirlwind repo, path references below are relative to that repo root).

## 0. Goal

The agent generates completions through a **rollout proxy** and reports
scalar rewards for them. The optimization harness (an asynchronous
GRPO/actor-learner system) consumes those (completion, reward) pairs to
update the generating model via LoRA, then hot-swaps the updated weights
back into the serving engine — the agent transparently benefits from every
update. The connector is the agent's only interface to this loop.

```
┌──────────────┐  generate_external  ┌───────────────┐  generate   ┌─────────┐
│  agentic     │ ──────────────────► │ rollout proxy │ ──────────► │ engine  │
│  harness     │ ◄────────────────── │ (replay       │ ◄────────── │ (SGLang)│
│  + connector │  completions +      │  buffer,      │             └─────────┘
└──────┬───────┘  rollout_ids        │  versioning)  │   update_weights (hot-swap)
       │                               └──────▲──────┘
       │ report_reward(rollout_ids,           │ pop_ready_batch
       │                rewards)              │
       └──────────────────────────────►───────┴──────► async GRPO updater (LoRA)
```

The connector talks **only** to the proxy. It must never talk to the engine,
the updater, or any other node directly.

## 1. Transport (v1: whirlcoms RPC)

The proxy exposes two RPC endpoints on a whirlcoms `Node` with `node_id`
`"rollout"` (configurable; the connector takes it as a constructor arg).

**Environment the connector process needs:**

| Variable | Meaning |
|---|---|
| `WW_RUN_ID` | IPC namespace. MUST equal the optimization run's id (all sockets are namespaced by it). |
| `NODE_COMMS_ENSURE_REGISTER` | set to `"1"` — makes registration retry until the proxy is up |
| `WHIRLWIND_RPC_TIMEOUT` | optional; seconds for RPC timeouts (default applies otherwise) |

**Client mechanics** (mirror `sim_external_agent.py`):

```python
from whirlwind.whirlcoms.node import Node
node = Node(<unique_agent_node_id>, auto_process=True)
await node.start()   # registers with the run's registry

# generate (typed call — validates the reply):
result = await node.call_typed(
    "rollout", "generate_external", ExternalGenerateResult,
    _to_dict(ExternalGenerateRequest(...)),
)
# reward (untyped call):
reply = await node.call(
    "rollout", "report_reward",
    _to_dict(ReportRewardRequest(rollout_ids=[...], rewards=[...])),
)
```

If the agent's environment cannot import `whirlwind`, STOP and report back:
an HTTP/OpenAI-compatible facade on the proxy is planned but not built; the
whirlcoms client is the only supported transport in v1.

## 2. Wire protocol (exact shapes)

### 2.1 `generate_external` — request

```python
@dataclass(kw_only=True)
class ExternalGenerateRequest(Message):
    prompts:        List[str]           # raw prompt strings (NO chat template —
                                        # the engine applies it from prompt+system_prompt)
    system_prompt:  Optional[str] = None
    group_size:     int = 1             # completions served PER prompt (GRPO group)
    temperature:    Optional[float] = None   # None = engine config default
    top_p:          Optional[float] = None   # None = engine config default
    max_new_tokens: Optional[int] = None     # ACCEPTED BUT IGNORED (engine governs)
    client_id:      str = ""                 # caller identity (audit)
    metadata:       Optional[Dict[str, Any]] = None  # stored with buffer entries
```

### 2.2 `generate_external` — result (index-aligned lists)

```python
@dataclass(kw_only=True)
class ExternalGenerateResult(Message):
    rollout_ids:     List[str]   # opaque stable handles — the ONLY thing needed to report rewards
    completions:     List[str]   # completion text, index i ↔ rollout_ids[i]
    finish_reasons:  List[str] = []   # e.g. "stop" | "length"
    weight_version:  int = 0     # policy version that generated ALL samples in this call
```

`len(result.rollout_ids) == len(req.prompts) * req.group_size` — the proxy
raises if the backend misaligns; the connector SHOULD still assert this
client-side. Ordering: for prompt p and replicate j, flat index
`i = p * group_size + j`.

### 2.3 `report_reward` — request / reply

```python
class ReportRewardRequest(ArgsKwargs):     # kwargs:
    rollout_ids: list    # same length as rewards (else the proxy raises ValueError)
    rewards:      list   # scalar float per rollout_id (any range; agent-defined semantics)
```

Reply (always HTTP-200-equivalent — tolerant by design):

```python
{"ok": True, "n_recorded": <int>, "missing": [<rollout_id>, ...]}
```

`missing` = ids that are unknown **because already popped, TTL-expired, or
never served**. A non-empty `missing` is NOT a transport error; see §5.

**Retry semantics (verified server-side):** re-reporting an id that is still
pending **overwrites** (last-write-wins); re-reporting after pop/expiry lands
in `missing`. Therefore: **retries after RPC timeouts are safe**; keep the
reward value deterministic per rollout_id (never recompute a different value
on retry).

## 3. Connector interface (required deliverable)

One module, framework-agnostic async (must work under `asyncio`), no
dependency on whirlwind beyond the transport, structured like this:

```python
@dataclass
class GroupResult:
    prompt: str
    system_prompt: str | None
    rollout_ids: list[str]      # n = group_size handles
    completions: list[str]      # index-aligned with rollout_ids
    finish_reasons: list[str]
    weight_version: int
    created_ts: float           # time.time() at generate time

@dataclass
class PendingStats:
    n_inflight: int             # rollout_ids generated but not yet rewarded
    oldest_age_s: float         # age of the oldest unrewarded rollout_id

@dataclass
class ReportSummary:
    n_recorded: int
    missing: list[str]

class RolloutConnector:
    def __init__(self, *, proxy_node_id="rollout", client_id="<agent-name>",
                 node_id="<agent-comms-id>", default_group_size=4): ...
    async def start(self) -> None: ...           # create + start whirlcoms Node
    async def close(self) -> None: ...           # deregister + stop node

    async def generate_group(self, prompt: str, *, system_prompt: str | None = None,
                             n: int | None = None, temperature: float | None = None,
                             top_p: float | None = None,
                             metadata: dict | None = None) -> GroupResult:
        """One prompt × n completions in ONE generate_external call (see R1)."""

    async def report_rewards(self, rewards: Mapping[str, float]) -> ReportSummary:
        """Batched reward report. MUST be called exactly once per rollout_id
        (retry-safe per §2.3). Recommended batch cadence: every few seconds,
        not one RPC per sample."""

    def pending(self) -> PendingStats: ...
    async def await_all_reported(self, timeout_s: float) -> bool: ...
```

`n` defaults to `default_group_size` (4). The connector MAY additionally
expose a raw `generate_groups(prompts, ...)` passthrough for multi-prompt
calls; the single-prompt form is the required contract.

## 4. Behavioral rules (MUST/SHOULD, RFC sense)

- **R1 — groups, never solo samples (MUST).** GRPO learns from *within-group
  reward variance*: one prompt × `n ≥ 2` completions in **one** call. A
  group of 1 z-scores to advantage 0 → contributes nothing to training. The
  connector MUST default `n=4` and MUST reject `n < 2` unless the caller
  passes an explicit `allow_zero_learning=True` escape hatch.
- **R2 — exactly one terminal reward per rollout_id (MUST).** One scalar,
  terminal reward per id, reported once (retries with the same value are
  fine per §2.3). No intermediate/step rewards — they would overwrite.
- **R3 — reward latency budget (MUST respect, SHOULD monitor).** A pending
  group is discarded when its oldest member exceeds **TTL 300 s**; entries
  whose `weight_version` lags the live version by more than **K=3** syncs
  are dropped at pop. With sync-every-step (~20–30 s/opt-step), rewards
  SHOULD be reported within **~90 s** of generation, MUST be within 300 s.
  The connector MUST expose `pending().oldest_age_s` and SHOULD log a
  warning when it exceeds 120 s.
- **R4 — prompt format consistency (SHOULD).** Pass prompts in the same
  shape the optimization run trains on: a fixed `system_prompt` (e.g. the
  few-shot task template) + the raw task text as `prompt`. The engine
  applies the chat template itself — never pre-template. Mismatched formats
  are not a correctness bug (importance sampling corrects per-sample drift)
  but degrade group comparability.
- **R5 — never mutate or re-tokenize completions (MUST).** `rollout_ids` are
  the only handles the agent needs. Logprob/token bookkeeping is
  server-side; the agent must not re-tokenize, stitch, or reformat
  completions before reward scoring beyond what its own scoring requires.
- **R6 — single-turn against the async loop (MUST).** Against
  `training_async`, each call is an independent single-turn group;
  multi-turn episodes would be mis-baselined. Multi-turn agents MUST target
  the agentic mode (`training_agentic`, docs/agentic_multiturn.md): one call
  per (episode, turn) with `metadata["ww_tag"]`, one `report_reward` per
  finished episode covering all its turns, and `touch` heartbeats for long
  episodes.
- **R7 — pacing (SHOULD).** The updater trains when ≥ `min_train_batch`
  (currently **256**) reward-complete samples are buffered. The connector
  SHOULD sustain ≥ 256 reward-complete samples per ~30 s window (i.e.
  ~32 groups of 8, or 64 groups of 4) or the run config must be changed
  server-side. Idling is safe (the updater waits) but pending entries age
  toward TTL.
- **R8 — no version pinning (MUST).** The proxy serves whatever weight
  version is live; the connector MUST NOT try to select, pin, or detect
  versions beyond reading `weight_version` for observability.
- **R9 — error handling (MUST).** `RpcError` on `generate_external` → the
  call failed atomically; safe to retry the whole call (no orphan ids can
  be trained — the proxy logs to the buffer only after a fully-aligned
  backend result). `RpcError` on `report_reward` → retry per §2.3. Proxy
  unreachable at startup → block in `start()` (ENSURE_REGISTER retries);
  surface after its retry budget.
- **R10 — observability (SHOULD).** Expose counters: groups generated,
  samples, rewards reported, rewards missing, generate latency p50/p95,
  `pending()` stats. Log `missing` ids at WARNING with the proxy's reason
  hint (popped/expired/never-served are indistinguishable client-side).

## 5. Server-side failure modes (what the proxy does — for defense-in-depth)

| Event | Proxy behavior | Connector implication |
|---|---|---|
| Missing reward for some group members | the **whole group** never becomes ready → TTL (300 s) discards it | reward every id you were given |
| Reward after pop | `missing` in reply | benign; log and move on |
| Reward for unknown/never-served id | `missing` | bug on agent side — alert |
| Buffer overflow (cap 4096 samples) | oldest **whole groups** dropped | don't outpace training by >4096 samples |
| Version drift > K=3 at pop | entries dropped (whole groups) | respect R3 latency budget |
| Backend engine down | `generate_external` RPC errors | retry with backoff; surface after N failures |

## 6. Constants (single source of truth: server config — do not hardcode beyond defaults)

| Constant | Value (current run) | Where configurable |
|---|---|---|
| `reward_ttl_s` | 300.0 | `RolloutProxyConfig` |
| `max_weight_version_lag` (K) | 3 | proxy + `AsyncTrainingConfig` |
| `buffer_max_entries` | 4096 | `RolloutProxyConfig` |
| `min_train_batch` / `max_train_batch` | 256 / 256 | `AsyncTrainingConfig` |
| engine `max_gen_len` | 1024 (governs completion length; `max_new_tokens` is ignored) | engine rollout config |
| default sampling | temperature 0.7, top_p 0.9 | engine rollout config |

## 7. Acceptance criteria (the implementing agent MUST deliver)

1. **Unit tests against a fake proxy** implementing §2 exactly (tolerant
   `report_reward` with `missing`, index alignment, overwrite-while-pending).
   Cover: R1 rejection of `n=1` without the escape hatch; R3 warning at age
   > 120 s; R9 retry paths (generate atomic-retry, reward idempotent-retry);
   index-alignment assertion on misaligned results.
2. **Contract test vs the real proxy**: run the whirlwind async smoke
   (`tests/test_async_proxy.py` infra or the sim driver), then drive it with
   the connector: generate 4 groups of 4, report rewards, observe the
   updater's buffer metrics advance (`async/buffer_pending` etc.).
3. **A 10-line usage example** in the module docstring (async context,
   `start` → loop `generate_group`/score/`report_rewards` → `close`).
4. **No server-side changes** — the diff must not touch the whirlwind repo
   (read-only dependency). If a server change seems required, STOP and
   report the blocker instead.

## 8. Out of scope / future adapters

- **OpenAI-compatible HTTP facade** (planned server-side): when it lands,
  the connector gains a second transport (`OpenAIFacadeTransport`)
  implementing the same `RolloutConnector` interface; nothing above changes
  except §1. Design the transport as a swappable internal layer from day 1.
- **Multi-turn sessions**: implemented by the agentic mode
  (docs/agentic_multiturn.md) as an interface extension — the episode tag in
  `metadata["ww_tag"]`, `conversations` / `tools` on the generate call, and
  the `touch` endpoint; nothing above changes.
- **Version pinning / stable-policy serving**: explicitly not offered.

## Appendix A — Reference client excerpt (verified working, smoke-validated)

From `whirlwind/training/sim_external_agent.py`:

```python
result = await self._node.call_typed(
    self.proxy_node, "generate_external", ExternalGenerateResult,
    ExternalGenerateRequest(
        prompts=prompts.prompts, system_prompt=prompts.system_prompt,
        group_size=self.group_size, client_id="sim-external-agent",
    ),
)
# ... score completions ...
reply = await self._node.call(
    self.proxy_node, "report_reward",
    _to_dict(ReportRewardRequest(
        rollout_ids=result.rollout_ids, rewards=rewards,
    )),
)
```
