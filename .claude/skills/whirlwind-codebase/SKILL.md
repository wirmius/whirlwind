---
name: whirlwind-codebase
description: Architecture and module map of Whirlwind, the asyncio + ZeroMQ GRPO/RLHF training system (FSDP2 model node, SGLang/vLLM rollout, reward node, dataloader, W&B tracker). Load when asked to analyze, navigate, extend, or review this codebase, before touching the training loop, the FSDP2 updater, the rollout engines, or the whirlcoms RPC layer.
---

# Whirlwind — architecture and module map

## What it is

A from-scratch RLHF/GRPO training system for one machine with several GPUs
(only the model node's non-lead FSDP ranks may run on other machines:
`fsdp_sharding_strategy: HYBRID_SHARD`, `docs/launch.md`). Every stage is its own process; they talk over ZeroMQ DEALER peer-to-peer RPC
(`whirlcoms`), and a discovery-only registry lives in the training
orchestrator's process. Reference workload: Qwen3-0.6B-Base on GSM8K,
benchmarked against TRL's `GRPOTrainer`. Extension model: **new machinery =
new sibling file**, selected by a launch flag or config (`docs/extension_guide.md`).

## The loop (`whirlwind/training/training_grpo.py`)

Per step: fetch batch → expand prompts × `rollout_per_prompt` (the copies form a
group, tracked by `entry_id`) → rollout (completions, per-token logprobs,
finish reasons) → reward (rewards + group-relative advantages) → model
forward/backward over microbatches → every
`gradient_accumulation_steps`: optimizer step, export weights, rollout
hot-reload → validation / checkpoint on schedule → tracker flush.

Colocate mode (`memory_management_enabled`): the rollout engine sleeps
(releases weights + KV cache) during the update and the model node offloads
optimizer state during rollout. Order matters: the model sleeps **before** the
rollout wakes. `steps_per_generation > 1` is refused under colocate.

The orchestrator talks to node IDs (`dataloader`, `rollout`, `reward`, `model`,
`tracker`), never to modules. Its `Node` is constructed with
`fail_on_timeout=True`, so a timed-out RPC raises `RpcError` instead of
returning `None`. `training_grpo_capture.py` is the sibling that dumps batches
for offline replay.

## Selection mechanisms

| What | Mechanism | Example |
|---|---|---|
| Rollout engine | `--rollout-module` | `whirlwind.rollout.rollout_vllm` |
| Model updater | `--model-module` (runs under torchrun) | `whirlwind.model.model_updater_hf_trainer` |
| Reward node | `--reward-module` | `examples.length_penalty_reward` |
| Training loop | `--training-module` | `whirlwind.training.training_grpo_capture` |
| Node configs | `configs/<kind>/*.yaml` via `--<kind>-config` / `WHIRLWIND_*_CONFIG` | `configs/model/fsdp2_base_colocate_gas1_fp32master_chunked_dr_grpo.yaml` |
| Loss / IS / precision | fields in the model config | `loss_reduction: dr_grpo`, `engine_is_correction: true` |

`bringup_and_run.sh` is the composition root (per-node `CUDA_VISIBLE_DEVICES`,
torchrun for the model node, `WW_RUN_ID` namespacing, teardown, and the training
loop's exit status; `--nodes` starts a subset and prints the commands for the
rest, which is how one node runs under a debugger in a live run
(`docs/debugging.md`, including the timeouts to raise). `WHIRLWIND_PYTHON` /
`WHIRLWIND_TORCHRUN` pick the interpreter. The
`examples/` recipes choose 1- or 2-GPU configs from `GPUS`. `validate_configs.py` loads
YAMLs through the dataclasses without starting anything.

## Module map

### `model/model_fsdp2.py` — the mainline updater (`ModelUpdateNode`)
- **FP32 master is mandatory** (`torch_dtype="float32"`): FSDP2 keeps the
  sharded master in the load dtype and casts to bf16 only for compute.
- Rank 0 serves RPC. `_fsdp_using_function` splits a `BatchedMessage` by
  world size, scatters, each rank computes, gathers, joins. Non-lead ranks sit
  in `_standby`; a handler exception on rank 0 still takes part in the gather
  (error sentinel) and releases the lock, so collectives stay in lockstep.
- Microbatch loop with `global_batch_size_override` /
  `global_max_completion_len_override`. The microbatch split runs *before*
  the rank scatter, so the denominators are `data.batch_size // world_size`
  and the token total `// world_size`: per-rank, because FSDP2 reduces
  gradients with `ReduceOp.AVG`. `micro_batch_size: -1` bypasses the loop and
  must give the same gradient at any world size.
- `_pad_and_stack`: TRL-aligned left-padded prompts / right-padded
  completions, completion-shaped `[B, C]` tensors, `logits_to_keep=C_max`,
  `selective_log_softmax`; asserts engine-logprob length alignment.
- Loss math in `updater_utils.compute_grpo_loss_core` (returns every stage:
  log ratio, coefficients, per-token losses, KL, masked loss). Knobs:
  `loss_reduction` (`grpo|dr_grpo|bnpo|ce`), `engine_is_correction`,
  `engine_is_mode` (`sequence_mask|sequence_truncate|token_mask|token_truncate`),
  `engine_is_cap`, `logits_chunk_size` (chunked lm_head, tied embeddings),
  `fsdp_grad_reduce_dtype`, `gradient_accumulation_steps`, `lr_schedule_steps`,
  CPU offload (`fsdp_cpu_offload`, with custom clip / fp32-Adam / state-sync adapters).
- Sleep/wake: offloads **optimizer state only**; FSDP param shards stay on
  the GPU (reassigning them breaks FSDP2's storage alias, TIPS §3.1).
- Checkpoints: DCP hot checkpoints written to a temp dir and swapped in,
  `state.json` last; cold checkpoints are HF weights only.
- Diagnostics (all env-gated, off by default): `WW_DETECT_ANOMALY`,
  `WW_SDP_DISABLE_CUDNN`, `WW_ATTN_IMPL`, update dumps (`_update_diag.py`),
  step fingerprints, batch capture. `WHIRLWIND_NCCL_TIMEOUT` scales the
  process-group watchdog for long-context data-parallel runs.
- Siblings: `model_updater_hf_trainer.py` (HF `Trainer.training_step`
  machinery; the parity reference), `model_accelerate_pure.py`.

### `rollout/` — engines behind one RPC contract
- Contract: `generate(PromptBatch) -> RolloutResult` (completions, `token_ids`,
  per-token `logprobs` with NaN for missing — never 0.0 — and `finish_reasons`
  normalized to `"length"`/`"stop"`), `update_weights(path)` (raise on
  failure), `sleep()` / `wake()`, `ping`.
- `rollout_sglang.py`: colocate needs `disable_cuda_graph: true` and
  `enable_weights_cpu_backup: true`; every copy of a prompt goes to the same
  data-parallel worker; exact-0.0 logprobs pass through verbatim; `prefix_messages`
  render few-shot examples with the declared EOS after each (`few_shot_eos`).
- `rollout_vllm.py`: `logprobs_mode="processed_logprobs"` is mandatory
  (log-softmax **after** temperature); same `prefix_messages` rendering.
- `sglang_rollout_pool.py`: N engines behind one node ID, contiguous shards,
  LoRA pools (`docs/rollout_pool_design.md`, `docs/lora_design.md`).

### `whirlcoms/` — RPC layer
- `node.py`: DEALER socket per node, `call` / `call_typed` (raises `RpcError`
  on `None`), `@node.expose(expects=, returns=)` seam with a `MessageRegistry`,
  heartbeats with re-registration when the registry dropped the node, reply
  routing by the caller's address.
- `orchestrator.py`: the registry (REP register/heartbeat/list, PUB topology).
- `transport.py`: msgpack framing; payloads > 64 KB go through
  `/dev/shm/whirlwind-shm-*` (receiver unlinks; files leak if the receiver
  dies first); bf16 views for SHM copies; scalar-list fast path.
- Env: `WHIRLWIND_RPC_TIMEOUT`, `WHIRLWIND_HEARTBEAT_TIMEOUT`,
  `WHIRLWIND_HEARTBEAT_FATAL_MISSES`, `NODE_COMMS_ENSURE_REGISTER`.
  Details: `docs/whirlcoms.md`, `docs/messages.md`.

### `reward/`
- `reward_function.py` (`RewardNode`): `_raw_rewards` is the override point
  (`accuracy` regex or `length` built in); `_compute_grpo_advantages` groups
  by `entry_id`, Bessel `nanstd`, `scale_rewards: group|batch|none`
  (TRL-matched); NaN-aware metrics; truncation masking; `main(node_cls)` runs
  a subclass as a node. Example: `examples/length_penalty_reward.py`.
- `advantages.py`: `compute_group_advantages`, the one advantage
  implementation shared by the reward node and the async updater.

### `data/dataloader_node.py`
Multi-dataset RPC node over Hugging Face datasets: cached per-epoch shuffle,
`entry_id` tracking for GRPO grouping, epoch/index restore, few-shot prompt
building (its own seed; `few_shot_eos` appends the EOS after each example and
ships the examples as `prefix_messages`). `data_hf.py` is the legacy in-process
loader.

### `tracker/tracker_wandb.py`
Metrics prefixed by source node, per-step flush (fire-and-forget metrics settle
before it), tables on a cadence, resume via `wandb_run_id` + `resume_step`,
`backend: "none"` for offline runs.

### `config.py` and `messages.py`
Dataclasses are the authoritative field lists; `from_yaml` rejects unknown keys
and warns when rollout and model disagree on generation length or temperature;
`$ENV:fallback` substitution in training configs; `_EXTRA_SECTIONS` lists the
top-level YAML sections owned by variant nodes (none by default). Messages implement
`split_into` / `join` / `slice_by` for FSDP sharding (`join` is a left inverse
of `split_into` only when the batch is at least `n` long); `RewardResult.advantages`
carries optional per-token advantages; `RewardRequest.prompt_token_ids` /
`completion_token_ids` let reward variants avoid re-tokenizing.

## Experimental: async proxy-rollout (agent post-training)

An actor-learner mode for training on traffic from an external agentic
system. Experimental: single-turn groups, LoRA updates. Design, API and limitations:
`docs/async_proxy_rollout.md`; wire protocol for agents:
`docs/agent_connector_spec.md`; runnable example:
`examples/experimental_async_proxy.sh`.

- `rollout/rollout_proxy.py` (`RolloutProxyNode`, registers as `rollout`):
  fronts one unmodified backend rollout node (`rollout-engine`). Endpoints:
  `generate_external` (serve an agent; every sample is logged to the buffer
  with the current `weight_version`), `report_reward` (out-of-order, batched,
  missing-tolerant), `pop_ready_batch` (updater pull), and `update_weights`
  (forwards, then bumps the monotonic `weight_version`).
- `rollout/replay_buffer.py`: in-process on the proxy; group-atomic FIFO with
  reward TTL, staleness bound K (`max_weight_version_lag`) and drop-oldest
  overflow. Partial groups are never returned, so GRPO group statistics stay
  intact.
- `training/training_async.py` (`AsyncGRPOOrchestrator(TrainingOrchestrator)`):
  pop → advantages (`reward/advantages.py`) → one optimizer step → LoRA
  adapter sync every `adapter_sync_every_k_steps`. Requires
  `gradient_accumulation_steps: 1`; batch size comes from
  `min_train_batch`/`max_train_batch`. Non-colocate: the engine never sleeps.
  Uses `call_typed` throughout.
- `training/sim_external_agent.py`: a simulated external agent on GSM8K that
  drives `generate_external` and reports rewards out of band.
- Messages `ExternalGenerateRequest/Result`, `ReportRewardRequest`,
  `BufferEntry`, `BufferPopRequest`, `BufferBatchResult`; configs
  `RolloutProxyConfig`, `AsyncTrainingConfig`.
- `model/fp8_loader.py` (experimental, independent of the async mode):
  dequantizes official fine-grained FP8 checkpoints at load; the model node
  takes this branch only when the checkpoint declares FP8 quantization.
- Tests: `tests/test_async_proxy.py` (GPU-free, fake engine over IPC).

## Experimental: multi-turn agentic mode

Sibling modules on top of the async mode (`docs/agentic_multiturn.md`;
example `examples/experimental_agentic.sh`). The base async modules
only carry generic hooks; agent-only config keys live in
`config_agentic.py`, so the base nodes reject them.

- `config_agentic.py`: `AgentRolloutConfig(RolloutConfig)`,
  `AgentProxyConfig(RolloutProxyConfig)`,
  `AgenticTrainingConfig(AsyncTrainingConfig)`.
- `rollout/rollout_sglang_agent.py` (`AgentRolloutNode(RolloutNode)`):
  `PromptBatch.conversations` + `tools` rendered with the chat template
  (special-token-escaped), server-side tool-call parsing
  (`RolloutResult.tool_calls`), per-request constrained decoding
  (`sampling_constraint`: one of ebnf/regex/json_schema/structural_tag,
  passed to SGLang; the base node rejects it), template/radix/speculative
  knobs, and
  `n_rpc_workers` concurrent serving behind a reader/writer gate (generates
  shared; update_weights/sleep/wake exclusive).
- `rollout/rollout_proxy_agent.py` (`AgentRolloutProxyNode`,
  `AgentReplayBuffer`): `touch` heartbeats, `pause_for_update` /
  `resume_after_update` (drain-aware, fail-closed), control-lane workers.
- `training/training_agentic.py` (`AgenticGRPOOrchestrator`): stages popped
  entries by their `metadata["ww_tag"]` group key until all `group_size`
  episodes arrived, z-scores episode returns (`episode_advantages`), pads the
  batch to `micro_batch_size × fsdp_size` with zero-advantage copies, and
  pauses a shared-GPU engine around updates (wake → adapter sync → release
  the parked requests).
- Example agents: `training/sim_os_agent.py` (tool calls on procedurally
  generated file-system / clock tasks; `PodmanSandbox` runs them as real
  commands in one locked-down container shared by all tasks (per-task
  workdir, TZ and path mapping over a read-only bind mount), `--backend memory`
  simulates them; held-out eval through the proxy's non-buffered
  `generate`),
  `training/sim_reflect_agent.py` (reflect-and-retry GSM8K; `flat` or
  `messages` carrier), `training/sim_swe_agent.py` (podman SWE-Gym bash
  episodes, tool calls).
- Tests: `tests/test_agentic.py`.

## Extension rules (condensed)

1. New sibling file next to the original; subclass or import it.
2. Edit originals only for a minimal, generic hook (dispatch entry, config-gated
   default-off branch). Never variant-specific logic inline.
3. Config selects the sibling; new knobs default to the current behavior.
4. Base regression tests stay green with the variant not selected; add
   `tests/test_<variant>_*.py` with exact-reference checks.
5. No long-lived per-variant branches; a run is reproducible from commit +
   driver + config tree + seed.

## Where to look first

- A metric looks wrong → `model_fsdp2.py:_model_update_step` (diagnostics
  dict), `updater_utils.compute_grpo_loss_core`, `tracker_wandb.py`.
- A run hangs → `_standby` / collectives in `model_fsdp2.py`, heartbeats in
  `whirlcoms/node.py`, `WHIRLWIND_RPC_TIMEOUT`, `/dev/shm` space.
- Rollout/trainer disagreement → `_pad_and_stack` asserts, engine logprob
  mode, temperature cross-check in `config.py`.
- The experimental async / agentic modes → the experimental sections above,
  `docs/async_proxy_rollout.md` and `docs/agentic_multiturn.md`.
