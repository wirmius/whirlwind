# Whirlwind — technical overview

How the system is put together, how a step runs, and what to know before
changing it. For the quick tour, see [README.md](README.md); for the rules
learned the hard way, see [TIPS_AND_TRICKS.md](TIPS_AND_TRICKS.md).

## Architecture

```
                  ┌────────────────────────────────────────────┐
                  │   training_grpo.py — TrainingOrchestrator │
                  │   CPU; drives the training loop           │
                  └──┬─────────┬─────────┬─────────┬──────────┘
        DEALER P2P    │         │         │         │      (discovery-only
                     ▼         ▼         ▼         ▼       registry: REP+PUB)
              ┌──────────┐ ┌────────┐ ┌──────────────┐ ┌────────────┐
              │ dataloader│ │ rollout│ │ reward       │ │ model      │
              │ (HF data, │ │ SGLang │ │ rewards +    │ │ FSDP2      │
              │ CPU)      │ │ / vLLM │ │ group-       │ │ (torchrun, │
              │           │ │ / pool │ │ relative     │ │  N GPUs)   │
              │           │ │  GPUs  │ │ advantages   │ │ + HF-Trainer
              └──────────┘ └────────┘ └──────────────┘ │   sibling  │
                              ▲                        └─────┬──────┘
                              └── weight sync (hot reload) ───┘
                                                   tracker (W&B, CPU)
```

Each box is a separate process. The orchestrator talks to **node names**
(`dataloader`, `rollout`, `reward`, `model`, `tracker`), never to modules, so
any implementation that serves the same RPC endpoints can be dropped in.

### Process and RPC model (`whirlwind/whirlcoms/`)

- Every node owns a ZeroMQ DEALER socket on an IPC endpoint and calls peers
  directly. A small registry (REP for register/heartbeat/list, PUB for
  topology events) lives in the training orchestrator's process. It only does
  discovery and liveness; no data passes through it.
- Messages are typed dataclasses (`whirlwind/messages.py`) serialized with
  msgpack. Payloads over 64 KB (tensors, long token lists) go through
  `/dev/shm` instead of the socket.
- `WW_RUN_ID` namespaces every IPC endpoint, checkpoint dir and the torchrun
  port, so several runs can share a machine.
- Liveness: nodes heartbeat the registry. A node the registry dropped during a
  stall re-registers; a node whose registry is gone shuts itself down after
  `WHIRLWIND_HEARTBEAT_FATAL_MISSES` misses (default 12, about 80 s), so
  nothing is left holding GPUs after a crash.
- The model node runs under `torchrun`. Rank 0 serves RPC and scatters work to
  the other ranks, which wait in a standby loop. Only rank 0 needs the IPC
  sockets, so the other ranks can run on other machines
  (`fsdp_sharding_strategy: HYBRID_SHARD`,
  [docs/launch.md](docs/launch.md#multi-node-model-update)).

Details: [docs/whirlcoms.md](docs/whirlcoms.md), [docs/messages.md](docs/messages.md).

## The step lifecycle

1. **Data** — the dataloader returns a batch of prompts (with few-shot
   prefixes and reference answers).
2. **Expand** — each prompt is repeated `rollout_per_prompt` times; the copies
   form a group.
3. **Rollout** — the engine generates completions with per-token logprobs and
   finish reasons.
4. **Reward** — per-sample rewards, then group-relative advantages
   (z-scored per group by default; `scale_rewards: group | batch | none`).
5. **Model update** — forward/backward over microbatches with gradient
   accumulation; optimizer step every `gradient_accumulation_steps`.
6. **Sync** — weights are exported and the rollout engine hot-reloads them.
7. **Track** — metrics and sampled tables go to the tracker; validation runs
   every `validation_every_n_steps`.

**Colocate mode** (`memory_management_enabled: true`) lets the rollout engine
and the trainer share GPUs: the engine sleeps (releases KV cache and weights)
during the update and the trainer offloads optimizer state during rollout.
`steps_per_generation > 1` is refused under colocate.

## Implementations and how they are selected

| stage | modules | selected by |
|---|---|---|
| rollout | `rollout_sglang`, `rollout_vllm`, `sglang_rollout_pool` | `--rollout-module`, `configs/rollout/*.yaml` |
| model update | `model_fsdp2` (manual FSDP2), `model_updater_hf_trainer`, `model_accelerate_pure` | `--model-module`, `configs/model/*.yaml` |
| reward | `reward_function` (accuracy, length) or a subclass | `--reward-module`, `configs/reward/*.yaml` |
| training loop | `training_grpo`, `training_grpo_capture` (batch capture for replay) | `--training-module`, `configs/training/*.yaml` |
| data | `dataloader_node` (Hugging Face datasets) | `configs/dataloader/*.yaml` |
| tracking | `tracker_wandb` | `configs/tracker/*.yaml` |

`bringup_and_run.sh` is the composition root: it starts each node with its own
`CUDA_VISIBLE_DEVICES`, runs the model under `torchrun`, waits for
registration, and tears everything down on exit. Flags: `--rollout-gpus`,
`--model-gpus`, `--<stage>-config`, `--<stage>-module`, `--master-port`,
`--dry-run` (validate configs only), `--nodes` (start a subset and print the
commands for the rest). The script exits with the training loop's status.
See [docs/launch.md](docs/launch.md).

## Configuration

- One YAML per stage under `configs/<stage>/`; the training config points at
  the rollout, reward and model files (`rollout_yaml`, …) with
  `$ENV_VAR:fallback` substitution. Bringup exports `WHIRLWIND_ROLLOUT_CONFIG`,
  `WHIRLWIND_REWARD_CONFIG`, `WHIRLWIND_MODEL_CONFIG`,
  `WHIRLWIND_TRAINING_CONFIG`, `WHIRLWIND_TRACKER_CONFIG` and
  `WHIRLWIND_DATA_LOADER_CONFIG`.
- The dataclasses in `whirlwind/config.py` are the authoritative field list.
  Loading rejects unknown keys and warns when the rollout and model disagree on
  generation length or temperature. A variant node may own an extra top-level
  section of a shared YAML (listed in `_EXTRA_SECTIONS` in `config.py`).
- `workdir` (training config) holds checkpoints and run artifacts. Hot
  checkpoints (`hot_checkpoint_every_n_steps`) resume a killed run; cold ones
  (`cold_checkpoint_steps`) are kept. DCP shards are written to a temp dir and
  swapped in, and `state.json` is written only after them.
- `lr_schedule_steps` (model config) sets the learning-rate horizon
  independently of `total_steps`, so short diagnostic runs can keep the
  schedule of the long run they stand in for.

## Objective and numerics

- **Loss reduction** (`loss_reduction`): `grpo` (per-sequence mean, then batch
  mean), `dr_grpo` (sum over tokens / (batch × `max_completion_length`)),
  `bnpo` (sum / total tokens). PPO-style clipping via `clip_range_low/high`;
  optional KL via `kl_coef`.
- **Engine importance sampling** (`engine_is_correction`, `engine_is_mode`:
  `sequence_mask | sequence_truncate | token_mask | token_truncate`,
  `engine_is_cap`) corrects for the gap between engine and trainer logprobs.
  Engines report missing logprobs as NaN, never 0.0.
- **Precision:** bf16 compute with fp32 master weights and fp32 gradient
  reduction; `logits_chunk_size` computes the lm_head in chunks so the full
  `[B, T, V]` logits are never materialized.
- **Multi-GPU scale:** FSDP2 averages gradients across ranks, so every loss
  denominator is per-rank (global count / `world_size`). The microbatched and
  full-batch (`micro_batch_size: -1`) paths produce the same gradient at any
  world size.

The rules behind these choices, and how to check them, are in
[TIPS_AND_TRICKS.md](TIPS_AND_TRICKS.md). Read it before changing precision,
loss scaling, sleep/wake, or engine logprob handling.

## Experimental: async proxy-rollout (agent post-training)

For training on the traffic of an external agentic system, Whirlwind has an
**experimental** actor-learner mode. It replaces the synchronous loop's
generate-then-train cadence with three decoupled parts:

- **Rollout proxy** (`whirlwind/rollout/rollout_proxy.py`, node id `rollout`)
  in front of an unmodified engine node (`rollout-engine`). Agents call
  `generate_external`; every sample is logged to a replay buffer with the
  policy's `weight_version` and waits for its reward, which the agent reports
  through `report_reward` at any time and in any order.
- **Replay buffer** (`whirlwind/rollout/replay_buffer.py`, in the proxy's
  process): group-atomic and FIFO, with a reward TTL, a staleness bound K
  (`max_weight_version_lag`) and drop-oldest overflow. Only complete groups
  leave it, so GRPO group statistics stay intact.
- **Async updater loop** (`whirlwind/training/training_async.py`): pops a
  batch, computes advantages with the same function the reward node uses
  (`whirlwind/reward/advantages.py`), takes one optimizer step on a LoRA
  adapter, and pushes the adapter to the engine every
  `adapter_sync_every_k_steps`. Staleness is corrected by the existing bounded
  engine IS ratio (`engine_is_mode: token_mask`) and bounded by K.

The engine never sleeps, so the engine and the updater need a GPU each.
`examples/experimental_async_proxy.sh` runs it on GSM8K with a simulated agent
(`whirlwind/training/sim_external_agent.py`); a 65-step run raised validation
accuracy from 0.146 to 0.609, against 0.636 for the synchronous LoRA recipe.
LoRA updates only. Details:
[docs/async_proxy_rollout.md](docs/async_proxy_rollout.md) and
[docs/agent_connector_spec.md](docs/agent_connector_spec.md).

**Multi-turn agents** (also experimental, [docs/agentic_multiturn.md](docs/agentic_multiturn.md))
use sibling modules that leave the async mode untouched: an engine node with
chat conversations, tools, tool-call parsing and concurrent serving
(`rollout_sglang_agent.py`), a proxy with episode heartbeats and engine
pause/resume for shared-GPU updates (`rollout_proxy_agent.py`), and a
training loop (`training_agentic.py`) that regroups turns tagged with
`metadata["ww_tag"]` into episodes, z-scores episode returns within each
group, and trains every turn with its episode's advantage.
`examples/experimental_agentic.sh` runs a tool-use agent on procedurally
generated file-system and clock tasks whose tools run in a podman container
(`sim_os_agent.py`, the default) or a
reflect-and-retry GSM8K agent (`TASK=reflect`, `sim_reflect_agent.py`);
`sim_swe_agent.py` drives containerized SWE-Gym episodes.

Also experimental: `whirlwind/model/fp8_loader.py` lets the model node start
from an official fine-grained FP8 checkpoint by dequantizing it at load; the
training math is unchanged.

## Runtime knobs (environment)

| variable | default | effect |
|---|---|---|
| `WW_RUN_ID` | unset | namespaces IPC endpoints, checkpoints and the torchrun port |
| `WHIRLWIND_PYTHON`, `WHIRLWIND_TORCHRUN` | `python3`, `torchrun` | interpreter and launcher bringup uses for every node |
| `WHIRLWIND_TORCHRUN_ARGS` | unset | extra torchrun arguments for the model node (multi-node: [docs/launch.md](docs/launch.md#multi-node-model-update)) |
| `WHIRLWIND_RPC_TIMEOUT` | 300 | RPC timeout in seconds (raise for long generations) |
| `WHIRLWIND_NCCL_TIMEOUT` | 600 | process-group timeout for the model ranks (raise for long-context data-parallel runs) |
| `WHIRLWIND_HEARTBEAT_TIMEOUT` | 15 | seconds of silence before the registry drops a node |
| `WHIRLWIND_HEARTBEAT_FATAL_MISSES` | 12 | missed heartbeats before a node exits on its own |
| `NODE_COMMS_ENSURE_REGISTER` | 1 (bringup) | block node startup until registration succeeds (backoff retries) |
| `WW_SDP_DISABLE_CUDNN` | unset | drop the cuDNN SDPA backend (NaN backward on some H100 drivers) |
| `WW_ATTN_IMPL` | unset | force an attention implementation (e.g. `eager`) |
| `WW_DETECT_ANOMALY` | unset | wrap the backward in anomaly detection to localize the first NaN gradient (2–3× slower) |
| `WANDB_MODE` | online | `offline` / `disabled` to run without W&B |

Other `WW_*` variables enable diagnostics (update dumps, step fingerprints,
batch capture); see `whirlwind/model/_update_diag.py` and the debugging
toolbox in [TIPS_AND_TRICKS.md](TIPS_AND_TRICKS.md). All are off by default.
To run one node under a debugger inside a live run, and the timeouts to raise
while it sits at a breakpoint, see [docs/debugging.md](docs/debugging.md).

## Validation results

GSM8K validation reward, Qwen3-0.6B-Base, 200 steps:

| path | reward | notes |
|---|---|---|
| manual FSDP2 + chunked lm_head + `dr_grpo` | **0.6152** @128 (peak) | [`examples/gsm8k_dr_grpo_colocate.sh`](examples/gsm8k_dr_grpo_colocate.sh) |
| manual baselines | 0.6162 / 0.6260 | |
| HF-Trainer-backed model node | 0.6504 | parity reference; the gap is in the training-step machinery |

Measured before the per-rank denominator fix (0.5.0): at world size 2 the
logged loss and `grad_norm` are now 2× larger; the update itself is nearly
unchanged (AdamW, clipping off in this recipe).

## Testing

```bash
python3 -m pytest tests/ -q          # unit + integration; GPU tests skip without CUDA
bash tests/run_tests.sh              # same, with the library-path setup some envs need
```

Covers transport and RPC, message split/join, config round trips and strict
loading, rewards and advantages, padding alignment against TRL (exact
reference), chunked lm_head, FSDP2 smoke, teardown, and the experimental async
and agentic modes (replay buffer, proxy over IPC with a fake engine, async
loop helpers, episode grouping, pause/resume, the example agents). `tests/conftest.py` destroys any zmq context a
test leaves open, so one failing test cannot hang the session.

## Layout

```
whirlwind/
├── config.py / messages.py / utils.py    # config dataclasses, message contracts
├── whirlcoms/                            # ZMQ RPC: node, registry, transport
├── training/                             # TrainingOrchestrator (+ capture sibling);
│                                         # training_async + sim_external_agent,
│                                         # training_agentic + sim_os/reflect/swe_agent (experimental)
├── rollout/                              # rollout_sglang / rollout_vllm / pool;
│                                         # rollout_proxy + replay_buffer,
│                                         # rollout_sglang_agent + rollout_proxy_agent (experimental)
├── model/                                # model_fsdp2 / model_updater_hf_trainer /
│                                         # model_accelerate_pure + updater_utils;
│                                         # fp8_loader (experimental)
├── reward/                               # reward_function (accuracy/length), advantages
├── data/                                 # HF dataloader node
└── tracker/                              # W&B tracker
configs/{model,rollout,reward,training,tracker,dataloader}/
examples/                                 # runnable recipes and an example reward
tests/                                    # pytest suite
docs/                                     # reference and design docs (docs/INDEX.md)
.claude/skills/                           # skill files for AI coding assistants (see AGENTS.md)
bringup_and_run.sh                        # launcher (composition root)
```

## Environment

Python 3.11, torch (FSDP2), transformers, SGLang or vLLM, pyzmq, msgpack,
wandb. Same machine only (IPC + `/dev/shm`). Tested with torch 2.11–2.13,
transformers 5.8–5.12, SGLang 0.5.14–0.5.20.
