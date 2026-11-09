---
name: whirlwind-training-pitfalls
description: Checklist of the precision, loss-scale, sleep/wake, engine-logprob, importance-sampling, collective and RPC-timeout rules that Whirlwind's GRPO training depends on, with the symptom of each violation and how to verify. Load before editing whirlwind/model/*, whirlwind/rollout/*, whirlwind/training/training_grpo.py, whirlwind/whirlcoms/*, configs/model/* or configs/rollout/*, and when diagnosing a run that does not learn, collapses, hangs, or logs impossible values.
---

# Whirlwind training pitfalls — the checklist

The full reasoning and the evidence are in `TIPS_AND_TRICKS.md` (same section
numbers). This file is the pre-flight list: before you change a path, check
the rules that path depends on; when a run misbehaves, match the symptom.

## Before you change…

### …the model node's precision or FSDP setup (`model_fsdp2.py`, `configs/model/*`)
- `torch_dtype: "float32"` stays. A bf16 master freezes GRPO updates below the
  bf16 ULP (§1.1). The node warns at init if the master is bf16; the default
  recipe configs set float32.
- Do not add upcasts inside `_compute_loss` expecting a numerical change;
  only `WW_LOGITS_FP32` (logits before log-softmax) does anything (§1.4).
- Never reassign `_sharded_param_data` or `_local_tensor` to move tensors
  across devices (§3.1). Offload optimizer state only, or use
  `fsdp_cpu_offload: true`.
- Every rank runs identical collectives in identical order; rank-dependent
  branches deadlock. Iterate diagnostic dict keys in sorted order before
  `all_reduce` (§5.5, §6.6).
- Anything that touches CPU tensors needs its own derivation: bf16 Adam state
  NaNs on CPU, `clip_grad_norm_` has no CPU mesh backend (§1.7).

### …loss reduction, microbatching or accumulation
- Denominators are per-rank; FSDP2 averages across ranks. The microbatch
  split runs before the rank scatter, so `global_batch_size_override` and the
  token total are divided by `world_size` (§2.1).
- Microbatch losses of fixed-denominator reductions are summed, not averaged
  (§2.2). The `micro_batch_size: -1` path and the microbatched path must give
  the same gradient — run the pinning test on two ranks (§6.7).
- `max_completion_length` (model) == `max_gen_len` (rollout) (§2.9).
- Accumulation is orchestrator-owned via `TrainStepRequest.accum_steps`; the
  HF-Trainer node pins its own accumulation to 1 (§2.9).

### …sleep/wake or colocate choreography (`_sleep`/`_wake`, `training_grpo.py`)
- Model sleeps before the rollout wakes; the reward phase runs with the engine
  asleep (§3.4). A reward node that generates must wake/generate/sleep the
  rollout itself in `try/finally`.
- SGLang colocate: `enable_memory_saver`, `disable_cuda_graph`,
  `enable_weights_cpu_backup` all true (§3.3).
- `steps_per_generation > 1` stays refused under colocate.

### …rollout engines or logprob handling (`rollout_*.py`, `updater_utils.py`, `configs/rollout/*`)
- Logprobs are post-temperature (`logprobs_mode: processed_logprobs` for vLLM)
  and the trainer temperature equals the rollout's (§4.1).
- Missing logprobs are NaN, never 0.0; keep the length-alignment assert in
  `_pad_and_stack` (§4.2).
- Do not truncate or renormalize the trainer's logprobs to the engine's top-p
  set (§4.3, §4.4); do not fold engine logprobs into the PPO ratio (§4.10).
- Do not patch SGLang's decode sampler in place; use `rl_on_policy_target`
  or vLLM when the −0.014 nats/token bias matters (§4.5).
- Long completions: `engine_is_mode: token_mask` (§4.7).
- `finish_reason` from SGLang is a dict; normalize before comparing (§4.14).
- Every typed handler gets `@<Message>.decorate_serdes()` (§6.8).

### …RPC, timeouts, long-context configs
- The longest single RPC (validation generation, generation inside a reward
  node, the async proxy's `generate_external`) fits in
  `WHIRLWIND_RPC_TIMEOUT`; standby waits fit in `WHIRLWIND_NCCL_TIMEOUT`;
  heartbeat budget covers payload parsing (`WHIRLWIND_HEARTBEAT_TIMEOUT`)
  (§6.1–6.3).
- Long context: `validation_log_table: false`, `allow_auto_truncate: true`
  (§6.4–6.5).
- `node.call` without `fail_on_timeout=True` returns `None`; use
  `call_typed` or guard (§6.1).

### …the experimental async proxy-rollout mode
- Non-colocate by design: the engine never sleeps, so the engine config keeps
  `enable_memory_saver: false`; give the engine and the updater separate GPUs.
- `gradient_accumulation_steps: 1`; the buffer's `min_train_batch` /
  `max_train_batch` set the batch.
- Staleness is bounded by `max_weight_version_lag`; keep `engine_is_mode:
  token_mask` (§4.7) and watch the logged version lag.
- Advantages come from `reward/advantages.py`, shared with the synchronous
  reward node (§4.15); change them there, not in a copy.
- Agentic mode: an agent must tag every turn with `group` / `group_size` /
  `episode` and report all of an episode's turns in one `report_reward`
  call; otherwise groups never complete (or train as one-sample groups with
  zero advantage). Watch `agentic/groups_with_signal_frac` and the
  `agentic/groups_dropped_*` counters.
- Agentic batches vary in size; the loop pads them to `micro_batch_size ×
  fsdp_size` with zero-advantage copies. Keep `kl_coef: 0` there, or pads
  add a KL gradient.
- Shared-GPU agentic layout: `enable_memory_saver: true` on the engine, and
  never sleep it under live requests (the proxy's drain enforces that).

## Symptom → first suspects

| symptom | check |
|---|---|
| reward flat from step 0, `grad_norm` non-zero | bf16 master (§1.1); count bit-unchanged parameters across steps |
| microbatched and full-batch (`micro_batch_size: -1`) gradients differ by a factor of `world_size` | microbatch denominators not per-rank (§2.1); compare both at world 2 |
| IS ratio collapses within ~15–30 steps, `signed_mean` drifts to −0.07 | sleep/wake alias break (§3.1); stale forward vs live optimizer |
| loss zero or masked everywhere from step 1 | temperature-mismatched or raw engine logprobs (§4.1); sequence-mask over long completions (§4.7) |
| `signed_mean` ≈ −0.02, exact 0.0 on certain tokens | vLLM top-p renormalized logprobs; expected, do not "fix" (§4.3) |
| `is_ratio/mean < 0`, `uncapped_min > uncapped_max` | hash-ordered keys before `all_reduce` (§5.5) |
| `clipped_ratio` = 0 while `mean_length` ≈ cap | `finish_reason` dict compared as string (§4.14) |
| loss swings 40–128% between micro-batch sizes, gradient stable | bf16 reporter noise, not a bug (§1.3, §1.4) |
| reported loss n× another framework's at equal gradient | accumulation or microbatch loss scale (§2.3) |
| run wedges after a long validation | heartbeat budget, RPC timeout, validation tables (§6.1–6.4) |
| standby ranks die mid-broadcast | `WHIRLWIND_NCCL_TIMEOUT` (§6.3) |
| rollout node never registers, no error, multi-GPU engine | GPUs on different sockets; pick a linked pair (`nvidia-smi topo -m`) (§6.11) |
| SIGSEGV in `ncclCommInitRank`, rollout never registers or model node dies after the first export | host InfiniBand stack; `NCCL_IB_DISABLE=1` (§6.10) |
| first generate at long context fails | `allow_auto_truncate` (§6.5) |
| NaN at the first post-update step on an H100 | cuDNN SDPA backward; `WW_SDP_DISABLE_CUDNN=1` (§1.6) |
| engine emits gibberish after wake | `enable_weights_cpu_backup` (§3.3) |
| OOM at long context despite chunked lm_head | autograd keeps every chunk's logits; lower `micro_batch_size` (§2.8) |
| hundreds of Hub requests per run | `AutoConfig.from_pretrained` per export; `HF_HUB_OFFLINE=1` (§3.7) |

## Do not rewalk

Retracted or dead for the collapse investigation: bf16 forward gradient bias,
IS-cap clipping, rollout↔trainer weight desync, microbatch aggregation,
activation-checkpointing presence or mechanism, the clip path, optimizer
wrapping, early-basin selection, top_p 1.0, the 0.0→NaN sentinel as a fix,
truncating the trainer's logprobs to the top-p set, patching SGLang's sampler
in place. Confirmed no-ops: `WW_IS_FP32`, `WW_LOSS_FP32`.

## How to verify a numerics change

1. `python3 -m pytest tests/ -q` with the variant not selected.
2. An exact-reference test for the changed math (`tests/test_trl_padding_alignment.py`
   and `tests/test_chunked_lm_head.py` show the style).
3. Capture one round (`WW_CAPTURE_EVERY`/`WW_CAPTURE_DIR`) and replay through
   the old and new path with `WW_IS_DIAG_DIR` set; diff the dumps (§5.3, §7.5).
4. Two ranks, not one (§6.7). Same `micro_batch_size` on both sides (§7.5).
5. Only then a 65-step run with `lr_schedule_steps` matching the long run
   (§2.5), read `grad_norm`, length and entropy next to the reward.
