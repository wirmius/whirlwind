# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.0] — 2026-11-09 — first public release

### Added
- **Experimental: async proxy-rollout mode for agent post-training**
  (`docs/async_proxy_rollout.md`). An always-on rollout proxy
  (`whirlwind/rollout/rollout_proxy.py`) serves an external agent through
  `generate_external`, logs every sample with its policy version to a
  group-atomic replay buffer (`whirlwind/rollout/replay_buffer.py`), and takes
  rewards back through `report_reward`; an async updater loop
  (`whirlwind/training/training_async.py`) trains a LoRA adapter on fresh,
  complete groups and hot-swaps it into the engine. A simulated GSM8K agent
  (`whirlwind/training/sim_external_agent.py`), the agent-side protocol
  (`docs/agent_connector_spec.md`), configs (`RolloutProxyConfig`,
  `AsyncTrainingConfig`) and `examples/experimental_async_proxy.sh`. On GSM8K a
  65-step run (one hot-swap per step, version lag at the K=3 bound, no TTL or
  overflow discards) raised validation accuracy from 0.146 to 0.609, against
  0.636 for the synchronous LoRA recipe.
- **Experimental: multi-turn agentic mode** (`docs/agentic_multiturn.md`), in
  sibling modules that leave the async mode unchanged: an engine node with chat
  conversations, tools, server-side tool-call parsing, special-token escaping
  and concurrent serving (`rollout_sglang_agent.py`); a proxy with episode
  heartbeats (`touch`) and drain-aware engine pause/resume for shared-GPU
  updates (`rollout_proxy_agent.py`); a training loop that regroups turns
  tagged with `metadata["ww_tag"]` into episodes and trains every turn with
  its episode's group-relative advantage (`training_agentic.py`); configs in
  `whirlwind/config_agentic.py`; `examples/experimental_agentic.sh`. Three
  example agents: tool use on procedurally generated file-system and clock
  tasks, with the tools running as real commands in a podman container and
  the `<tool_call>` format enforced by constrained decoding
  (`sampling_constraint` on generate requests; `sim_os_agent.py`; on
  held-out tasks Qwen3-0.6B-Base went from 72% to 100% exact answers in 100
  steps), reflect-and-retry GSM8K (`sim_reflect_agent.py`)
  and containerized SWE-Gym bash episodes (`sim_swe_agent.py`).
- **Experimental: FP8 checkpoint ingest** (`whirlwind/model/fp8_loader.py`): the
  model node can start from an official fine-grained FP8 checkpoint,
  dequantized at load.
- `whirlwind/reward/advantages.py`: the group-relative advantage computation,
  shared by the reward node and the async updater (behavior unchanged).
- `lr_schedule_steps` (model config) sets the learning-rate horizon
  independently of `total_steps`.
- `few_shot_eos` (dataloader) appends the tokenizer's EOS after each few-shot
  example so the demonstrated pattern is "answer, then stop"; `few_shot_seed`
  picks the examples independently of the shuffle seed.
- The dataloader loads local `.parquet` files.
- `bringup_and_run.sh` exits with the training loop's status, and the loop
  exits non-zero when it stops early because a required node was lost.
  `WHIRLWIND_PYTHON` / `WHIRLWIND_TORCHRUN` select the interpreter; the
  example recipes choose 1- or 2-GPU configs from `GPUS` and pass extra flags
  through.
- **Multi-node model update** (`docs/launch.md`). `fsdp_sharding_strategy:
  HYBRID_SHARD` shards each parameter within a node and replicates it across
  nodes, on a nodes × GPUs-per-node mesh taken from torchrun (it used to
  guess the node count as the square root of the rank count), so only one
  gradient all-reduce per step crosses nodes. `WHIRLWIND_TORCHRUN_ARGS`
  passes `--nnodes`, `--node_rank` and `--master_addr` to the model node's
  torchrun. Only the extra model ranks run on other machines; every other
  node, and model rank 0, stays on the head node.
- `docs/debugging.md`: running one node under a debugger inside a live run
  (bringup's `--nodes`, the timeouts to raise while a node sits at a
  breakpoint, the command for each node, VS Code launch configurations). The
  README now opens with the project's motivation and the whirlcoms contract,
  and shows example launches for a 4-GPU machine.
- `TIPS_AND_TRICKS.md` (the distilled engineering rules), `AGENTS.md` and
  skill files under `.claude/skills/` for AI coding assistants;
  `docs/configuration.md` is generated from the config dataclasses.

### Fixed
- FSDP2: per-layer wrapping found no decoder layers on recent transformers
  versions, so models were sharded as one root unit and activation
  checkpointing did nothing. Decoder layers are now found by a structural scan,
  and the node logs how many units it wrapped (warning loudly on zero).
- FSDP2: training forwards run with the model's KV cache off. With activation
  checkpointing active (see above), the backward recompute of each decoder
  layer re-appended its keys to the cache and attention failed on the doubled
  key length.
- Model node: the lead rank (the one that answers RPC) is global rank 0. The
  entry points picked it by local rank, so on a multi-node launch local rank
  0 of every other node took the lead path, exited, and hung the process
  group.
- Model node shutdown: rank 0 releases the other ranks from their standby
  loop before it destroys the process group. They used to stay blocked in
  the next collective, so torchrun outlived a finished run until bringup
  killed it 120 s later, and ranks on other nodes died with an NCCL error.
- FSDP2: the CPU-offload gradient clip (`fsdp_cpu_offload: true`) read the
  all-reduced norm before its asynchronous copy to the CPU had landed, so the
  logged gradient norm and the clip factor could come from a stale value.
- Rollout: the radix (prefix) cache is flushed after a LoRA adapter swap, so
  prefix KV computed under the previous adapter is never reused.
- Rollout: every copy of a prompt is sent to the same SGLang data-parallel
  worker (copies on different workers sampled in lockstep and returned
  identical completions), with a `completions/duplicate_frac` metric as a
  guard; SGLang's exact-0.0 logprobs pass through verbatim instead of being
  mistaken for a missing-value sentinel.
- Data: few-shot examples no longer change with the shuffle seed; a template
  without a few-shot placeholder is used verbatim.
- RPC: a heartbeat reply that does not parse no longer kills the heartbeat loop.
- RPC: a node with a reply it could not deliver (the caller had already
  exited, e.g. after the call timed out) hung forever in `stop()`, because
  zmq's default infinite linger blocked the context shutdown; bringup then
  had to kill it. Sockets now close with a 1 s linger.
- `configs/rollout/sglang.yaml` asked for four data-parallel engines
  (`dp_size: 4`), so the documented `--rollout-gpus 0` launch with the
  default configs could not start; it now runs one engine.
- The default model config uses an fp32 optimizer master (`torch_dtype:
  float32`); a bf16 master freezes GRPO updates.

### Removed
- `launch.py` (an unmaintained single-process launcher); use
  `bringup_and_run.sh` (`docs/launch.md`).
- Config options that were broken, only reproduced known failures, or had no
  effect (config loading is strict: delete these keys from your YAMLs):
  `qlora` (never tested; bitsandbytes 4-bit weights do not survive FSDP2
  sharding), `fsdp_sleep_offload_params` (its only alternative,
  `shard_reassign`, reproduced the storage-alias collapse of TIPS §3.1),
  `renorm_logprobs` / `renorm_top_logprobs_n` (reproduced the collapse of
  §4.4), `hftrainer_fsdp_version` (the HF-Trainer and accelerate siblings
  always use FSDP2), and the no-op `max_subrollouts`, `default_group_size`
  and `shard_strategy`.

## [0.5.0]

### Fixed
- **Microbatch loss denominators are per-rank.** The microbatch split runs
  before the batch is scattered to ranks, so every rank divided by the global
  batch and FSDP2's AVG reduction made the gradient `world_size`× smaller than
  the `micro_batch_size: -1` path (measured 0.5001× at world size 2). Logged
  loss, `grad_norm` and `max_grad_norm` clipping are affected; with AdamW and
  clipping off, the update is nearly unchanged. Single-GPU runs are unaffected.
- **Teardown:** guaranteed process cleanup on completion and on failure;
  nodes exit after a run of missed heartbeats instead of hanging when the
  orchestrator is gone; the rank-teardown wedge is bounded by a grace
  period; orphaned rank workers are swept after a crash.
- **RPC (whirlcoms):** a node the registry dropped during a stall
  re-registers (heartbeats from unknown nodes are nacked) and replies route
  by the caller's address; RPC timeouts raise `RpcError` instead of returning
  `None`; bf16 tensors over SHM and malformed registry commands; faster
  deserialization of large scalar lists.
- **Model node:** FSDP collectives stay in lockstep when a handler raises,
  and the lock is released; CPU-offload AdamW weight decay had the wrong sign
  and no lr factor; `WW_SDP_DISABLE_CUDNN` works around a cuDNN SDPA backward
  that returns NaN on some H100 drivers.
- **Training loop:** checkpoints record the completed step and resumed runs
  no longer double-offset W&B steps; `steps_per_generation > 1` under
  colocate is refused instead of warned about; a `tracker_ready` race that
  silently disabled validation; two cold-checkpoint bugs.
- **Checkpoints:** DCP shards are written to a temp dir and swapped in, and
  `state.json` only lands after them.
- **Rollout:** pool shards were round-robin and permuted the batch, and the
  export dir was deleted by its creator; SGLang's dict-valued `finish_reason`
  is normalized to `"length"`/`"stop"`; long requests can opt into
  `allow_auto_truncate`.
- **Tracker:** fire-and-forget metrics land before the per-step flush.
- **Config:** unknown YAML keys are rejected, and rollout/model length and
  temperature drift is warned about.
- **Tests:** a test that leaks a zmq context can no longer hang the suite
  (`tests/conftest.py`); stale tests ported to the current queue API.

### Added
- `RewardNode._raw_rewards` (override to define a reward) and a reusable
  `reward_function.main(node_cls)`; `examples/length_penalty_reward.py` shows
  a custom reward in ~10 lines.
- `README.md` (crash course) split from `TECHNICAL_README.md` (internals).
- `examples/gsm8k_dr_grpo_colocate.sh` with `configs/training/gsm8k_dr_grpo_colocate.yaml`
  and `configs/dataloader/gsm8k.yaml` — the validated mainline recipe.
- Environment knobs for long-context runs: `WHIRLWIND_NCCL_TIMEOUT`,
  `WHIRLWIND_HEARTBEAT_TIMEOUT`, and the `validation_log_table` config option.
- `bringup_and_run.sh --training-module` / `--reward-module`, completing the
  module-selection flags; `NODE_COMMS_ENSURE_REGISTER` on by default.
- `pyproject.toml` project metadata with `sglang`, `vllm`, `lora`, `dev`
  extras; MIT license.

## [0.4.0]

The scale-correct training core: chunked lm_head validated, FP32 master
weights with optimizer-only sleep/wake, LoRA and rollout-pool paths.
