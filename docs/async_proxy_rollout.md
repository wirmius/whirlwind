# Async proxy-rollout GRPO (experimental)

> **Status: experimental.** The pieces below are implemented and covered by
> GPU-free tests (`tests/test_async_proxy.py`). On GSM8K with the simulated
> agent, a 65-step run (one adapter hot-swap per step, version lag at the K=3
> bound, no TTL or overflow discards) raised validation accuracy from 0.146
> to 0.609, against 0.636 for the synchronous LoRA recipe; the engine never
> paused for training. Multi-turn episodes are handled by the sibling
> [agentic mode](agentic_multiturn.md). Not done yet: restart-safe buffers.
> APIs and configs may change.

This mode is for **agent post-training**: an external agentic system uses
Whirlwind's rollout engine to generate, decides for itself what to do with
the completions (tools, retries, scoring), and reports a reward for each one.
Whirlwind trains on that traffic in the background and hot-swaps the improved
policy back into the engine, so the agent benefits from every update without
restarting anything.

It is the actor-learner / experience-replay pattern. The engine never sleeps;
generation runs at the agent's pace and training at the updater's pace, and
the only coupling between them is a replay buffer with bounded staleness.

## Architecture

```
 external agent ── generate_external ──►  rollout proxy  ── generate ──►  rollout engine
       │          ◄── completions,  ──    (node "rollout")               (node "rollout-engine",
       │              rollout_ids            │  replay buffer             SGLang, LoRA)
       │                                     │  weight_version              ▲
       └─────── report_reward ─────────────►─┤                              │ update_weights
                                             │ pop_ready_batch              │ (adapter hot-swap)
                                             ▼                              │
                                      async updater loop ──► model node (FSDP2 + LoRA)
                                      (training_async.py)        └── adapter export ──┘
```

1. The agent calls `generate_external` on the proxy. The proxy forwards the
   request to the engine, returns the completions with one `rollout_id` each,
   and logs every sample to the buffer with its token ids, sampling logprobs
   and the current `weight_version`.
2. The agent reports rewards with `report_reward(rollout_ids, rewards)`,
   seconds or minutes later, batched or out of order. Missing rewards expire
   after `reward_ttl_s`.
3. The updater pops a batch of complete, rewarded groups that are at most
   `max_weight_version_lag` versions old, computes group-relative advantages
   (`whirlwind/reward/advantages.py`, the same code the synchronous reward node
   uses), and takes one optimizer step on the LoRA adapter.
4. Every `adapter_sync_every_k_steps` steps the updater exports the adapter
   (about 9 MB for the reference model), the proxy forwards it to the engine,
   which hot-swaps it, and the proxy bumps `weight_version`.

## Components

| piece | module | role |
|---|---|---|
| rollout proxy | `whirlwind/rollout/rollout_proxy.py` | `RolloutProxyNode`: fronts one unmodified backend rollout node (SGLang, vLLM or a pool). Endpoints `generate_external`, `report_reward`, `pop_ready_batch`; forwards `generate` (validation) and `update_weights`. |
| replay buffer | `whirlwind/rollout/replay_buffer.py` | In the proxy's process. Group-atomic FIFO: only complete groups leave it, so GRPO group statistics stay intact. Reward TTL, staleness bound, drop-oldest overflow. |
| async updater loop | `whirlwind/training/training_async.py` | `AsyncGRPOOrchestrator`, a subclass of the synchronous loop: pop → advantages → optimizer step → adapter sync. Validation, checkpoints and tracking work as in the synchronous loop. |
| simulated agent | `whirlwind/training/sim_external_agent.py` | A stand-in for the external system on GSM8K: fetches prompts from the dataloader, calls `generate_external`, scores accuracy locally, reports rewards. |
| messages | `whirlwind/messages.py` | `ExternalGenerateRequest/Result`, `ReportRewardRequest`, `BufferEntry`, `BufferPopRequest`, `BufferBatchResult`. |
| configs | `RolloutProxyConfig`, `AsyncTrainingConfig` in `whirlwind/config.py` | `configs/rollout/async_proxy_1x.yaml` (proxy), `configs/rollout/async_proxy_engine.yaml` (engine), `configs/model/fsdp2_async_lora_1gpu.yaml` (LoRA updater), `configs/training/async_proxy_gsm8k.yaml`. |

The protocol an agent uses is specified in
[agent_connector_spec.md](agent_connector_spec.md); the simulated agent is the
reference client.

## Running it

```bash
ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_async_proxy.sh            # 65 steps
STEPS=3 ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_async_proxy.sh    # smoke
```

The script starts the engine, the proxy, the reward, dataloader and tracker
nodes, the LoRA updater, the async loop and the simulated agent, each in its
own process group; it tears everything down when the loop finishes and exits
with the loop's status. `WHIRLWIND_PYTHON` / `WHIRLWIND_TORCHRUN` pick the
interpreter; `TRAINING_CONFIG`, `TRACKER_CONFIG` and `DATALOADER_CONFIG`
override the configs. To attach a real agent, start the same nodes without
the simulated one and point the agent at the proxy (node id `rollout`).

## Knobs

| knob | where | effect |
|---|---|---|
| `min_train_batch`, `max_train_batch` | training config | samples per optimizer step; the loop waits until `min_train_batch` ready samples exist |
| `max_weight_version_lag` | training config (and the proxy's fallback) | K: samples generated more than K adapter versions ago are discarded |
| `adapter_sync_every_k_steps` | training config | how often the adapter is pushed to the engine |
| `pop_poll_interval_s`, `step_timeout_s` | training config | buffer polling and how long a step waits for data before giving up |
| `reward_ttl_s` | proxy config | pending-reward lifetime; an expired sample drops its whole group |
| `buffer_max_entries` | proxy config | capacity; overflow drops the oldest groups |
| `log_internal_generate` | proxy config | whether the updater's own validation generations enter the buffer (default no) |

## Staleness and importance sampling

A sample generated under adapter version *v* is trained under *v + Δ*. The
loss path is the synchronous one: the sampling logprobs captured at generation
time are the "sampling" side of the bounded engine IS ratio, and the updater's
fresh forward gives the rest. Larger Δ widens the ratio exactly as a
logprob mismatch does in the synchronous loop (TIPS_AND_TRICKS.md §4), so:

- keep Δ small with frequent syncs (the adapter is cheap to push) and a small
  K (1–3 for small models);
- use `engine_is_mode: token_mask` (the shipped LoRA config does), since
  sequence-level IS is fragile on long completions;
- watch `version_lag_mean` / `version_lag_max` per step and the IS-ratio
  metrics; a collapsing ratio means K is too large or syncs too rare. The
  proxy and loop also log `async/discarded_stale`, `async/discarded_ttl`,
  `async/discarded_overflow`, `async/buffer_pending`, `proxy/weight_version`
  and `proxy/buffer_entries`.

No extra clipping for stale samples is applied beyond the bounded engine IS
ratio and the K bound.

## Requirements and limitations

- **Two GPUs minimum, used exclusively.** The engine owns its GPU and never
  sleeps (its config keeps `enable_memory_saver: false` and claims 85% of the
  GPU's memory with `mem_fraction_static: 0.85`); the updater needs its own.
  Sharing the engine's GPU with another job crashed the engine with a CUDA
  illegal-memory-access error in testing; the example script warns when
  either GPU is already in use. At scale the engine becomes a pool behind the
  proxy and the updater a multi-GPU FSDP2 node.
- **LoRA only.** The adapter hot-swap is the weight-sync primitive; full
  fine-tuning would mean pushing full weights to a serving engine.
- **`gradient_accumulation_steps: 1`.** Each popped batch is one optimizer
  step; batch size is set by `min_train_batch` / `max_train_batch`.
- **Single-turn groups.** Each `generate_external` call forms its own
  group(s). For multi-turn episodes use the [agentic mode](agentic_multiturn.md),
  which regroups tagged turns into episodes.
- **No restart safety for the buffer.** A proxy restart loses pending
  samples.

## Experimental: FP8 checkpoint ingest

Independent of the async mode, `whirlwind/model/fp8_loader.py` lets the model
node start from an official fine-grained FP8 checkpoint (per-128×128-block
E4M3 weights with a `weight_scale_inv` scale). The model node detects the
checkpoint's `quantization_config.quant_method == "fp8"`, dequantizes to the
configured `torch_dtype` at load, and trains as usual. The dequantization is
done on CPU for the whole state dict, which is fine up to roughly 8B
parameters, and the loader assumes a plain `AutoModelForCausalLM`
architecture.
