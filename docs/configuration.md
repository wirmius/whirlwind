# Configuration reference

One YAML per node kind under `configs/<kind>/`; bringup passes each file to its node with `--<kind>-config`.
The dataclasses in `whirlwind/config.py` (and `whirlwind/config_agentic.py` for the experimental agentic mode) are the
authoritative field lists: loading **rejects unknown keys**, and the training config warns when the rollout
and model files disagree on generation length or temperature. This page is generated from the dataclass
definitions and their comments; when a field changes, regenerate it rather than editing the tables.

## TrainingConfig — `configs/training/*.yaml`

The training orchestrator. `rollout_yaml`, `reward_yaml` and `model_yaml` point at the other files and accept `$ENV_VAR:fallback` substitution (bringup exports `WHIRLWIND_ROLLOUT_CONFIG`, `WHIRLWIND_REWARD_CONFIG`, `WHIRLWIND_MODEL_CONFIG`).

| field | type | default | notes |
|---|---|---|---|
| `rollout_yaml` | `str` | `"rollout.yaml"` | Sub-config YAML paths (each node loads its own file; bringup exports WHIRLWIND_*_CONFIG) |
| `reward_yaml` | `str` | `"reward.yaml"` |  |
| `model_yaml` | `str` | `"model.yaml"` |  |
| `rollout` | `RolloutConfig` | `None` | Loaded sub-config objects (populated by from_yaml) type: ignore[assignment] |
| `reward` | `RewardConfig` | `None` | type: ignore[assignment] |
| `model` | `ModelConfig` | `None` | type: ignore[assignment] |
| `total_steps` | `int` | `1000` | Loop control |
| `lr_schedule_steps` | `int` | `0` | Horizon of the LR schedule (warmup + decay); 0 = total_steps. A 65-step run with lr_schedule_steps 192 follows the first 65 steps of a 192-step run, so runs of different length compare at the same per-step LR. |
| `rollout_per_prompt` | `int` | `4` | N generations per prompt (GRPO group size) |
| `log_every_n_steps` | `int` | `1` |  |
| `memory_management_enabled` | `bool` | `True` | sleep/wake nodes between phases |
| `steps_per_generation` | `int` | `1` | Generation cadence (TRL ``steps_per_generation``). Generate a batch of completions once, then run ``steps_per_generation`` optimizer steps over contiguous slices of it (each slice = one GRPO group batch), syncing the rollout weights only once per … |
| `shutdown_on_node_failure` | `bool` | `True` | Failure handling: if a required node is detected as failed (RPC error or heartbeat timeout), broadcast a cluster-wide graceful shutdown to all nodes and stop, instead of spinning on the dead node. Active by default. |
| `rollout_log_table_every_n_steps` | `int` | `-1` | Table logging (moved from sub-node configs; orchestrator controls step-based gating) log rollout table every N steps (-1 = disabled) |
| `reward_log_table_every_n_steps` | `int` | `-1` | log reward table every N steps (-1 = disabled) |
| `validation_every_n_steps` | `int` | `0` | Validation run validation every N optimizer steps (0 = disabled) |
| `validation_at_step_0` | `bool` | `True` | run validation before any training |
| `validation_log_table` | `bool` | `True` | Log val rollouts/rewards as W&B tables. At long completion budgets the full-text serialization (~30-60MB) stalls the tracker RPC long enough to trip the heartbeat watchdogs — set false for 8k-context runs. |
| `workdir` | `str` | `""` | Artifacts & checkpointing working directory for checkpoints, logs, configs (empty = disabled) |
| `hot_checkpoint_every_n_steps` | `int` | `0` | hot checkpoint every N optimizer updates (0 = disabled) |
| `cold_checkpoint_steps` | `list` | `(computed)` | list of specific steps |
| `seed` | `int` | `42` | Misc |
| `orchestrator_node_id` | `str` | `"orchestrator"` |  |

## RolloutConfig — `configs/rollout/*.yaml`

The rollout engine node (SGLang or vLLM).

| field | type | default | notes |
|---|---|---|---|
| `model_path` | `str` | `""` | Model HuggingFace / local model path |
| `load_format` | `str` | `"auto"` | auto, safetensors, dummy, ... |
| `tp_size` | `int` | `1` | Tensor parallelism number of GPUs for SGLang TP |
| `dp_size` | `int` | `1` | number of data parallel workers |
| `context_length` | `int` | `4096` | Context / generation limits max context window |
| `max_gen_len` | `int` | `1024` | max completion length |
| `allow_auto_truncate` | `bool` | `False` | sglang allow_auto_truncate: when a request's prompt + max_new_tokens exceeds context_length, trim max_new_tokens per request instead of hard-rejecting (REQUIRED when max_gen_len leaves little prompt head- room — the 8196-window runs died on first generate … |
| `temperature` | `float` | `1.0` | Sampling hyperparameters (TRL-style) |
| `top_p` | `float` | `1.0` |  |
| `top_k` | `int` | `-1` | -1 = disabled |
| `repetition_penalty` | `float` | `1.0` |  |
| `generation_kwargs` | `dict[str, Any]` | `(computed)` | Extra generation params (forwarded to SGLang SamplingParams) e.g. min_p, skip_special_tokens, spaces_between_special_tokens |
| `mem_fraction_static` | `float` | `0.9` | Engine knobs SGLang KV cache fraction |
| `chunked_prefill_size` | `int` | `128` |  |
| `schedule_policy` | `str` | `"lpm"` | lpm, fcfs |
| `enable_memory_saver` | `bool` | `False` | torch_memory_saver (memory_saver / cpu_backup) is only useful when the orchestrator performs co-located sleep/wake during the RL cycle, i.e. when ``TrainingConfig.memory_management_enabled`` is True. With it off, leaving ``enable_memory_saver`` on … |
| `enable_weights_cpu_backup` | `bool` | `False` | mirror TrainingConfig.memory_management_enabled |
| `enable_lora` | `bool` | `False` | LoRA (native adapter reload). When ``enable_lora`` is True, the engine is initialized with LoRA support and ``_update_weights`` loads the exported adapter dir (adapter_config.json + adapter_model.safetensors) via the engine's native add_lora / … |
| `lora_max_rank` | `int` | `64` | vLLM max_lora_rank / SGLang --max-lora-rank |
| `lora_target_modules` | `list[str]` | `["q_proj", "v_proj"]` | SGLang --lora-target-modules |
| `disable_cuda_graph` | `bool` | `False` | Disable SGLang CUDA graph capture (decode). Colocate (rollout+FSDP share GPUs) can hit ``illegal memory access`` in the decode ``copy_done`` during cuda-graph REPLAY when an FSDP2 process co-resides on the GPU — the graph's static buffers are invalidated … |
| `rl_on_policy_target` | `str | None` | `None` | SGLang RL on-policy target. When set (e.g. "fsdp"), SGLang computes per-token logprobs via torch.log_softmax(logits/temp) instead of the standard log(softmax(logits/temp)). The standard path is numerically unstable in bf16 (softmax→0 then log→-inf), … |
| `log_every_call_override` | `int` | `-1` | Table logging override force table logging every N calls (-1 = disabled, use orchestrator's decision) |
| `node_id` | `str` | `"rollout"` | Misc |

## RolloutPoolConfig — `configs/rollout/pool_*.yaml`

A pool of engines behind one node id; each backend entry is a RolloutConfig YAML.

| field | type | default | notes |
|---|---|---|---|
| `node_id` | `str` | `"rollout"` |  |
| `backends` | `list[str]` | `["rollout0", "rollout1"]` |  |

## RewardConfig — `configs/reward/*.yaml`

The base reward node.

| field | type | default | notes |
|---|---|---|---|
| `reward_type` | `str` | | Reward type (mandatory — no default) "length" \| "accuracy" \| "model" \| "custom" |
| `length_scale` | `float` | `1.0` | Length-based reward params multiplier for raw length reward |
| `length_min` | `int` | `1` | minimum count for normalization |
| `group_size` | `int` | `16` | Group normalization (TRL GRPO-style) items per normalization group |
| `scale_rewards` | `str` | `"group"` | "group" \| "batch" \| "none" — std scope for z-score |
| `log_every_call_override` | `int` | `-1` | Table logging override force table logging every N calls (-1 = disabled, use orchestrator's decision) |
| `node_id` | `str` | `"reward"` | Misc |
| `accuracy_regex` | `str` | `r"#####?\s*(\d[\d,]*)"` | Accuracy reward params (for GSM8K-style datasets) ####?\s*(\d[\d,]*)" # pattern to extract answer from text |
| `mask_truncated_completions` | `bool` | `False` | Truncation masking (TRL GRPOConfig.mask_truncated_completions) zero advantage for length-truncated samples |

## ModelConfig — `configs/model/*.yaml`

The model-update node (manual FSDP2, HF-Trainer or accelerate sibling).

| field | type | default | notes |
|---|---|---|---|
| `model_path` | `str` | `""` | Model initial checkpoint / HF model |
| `torch_dtype` | `str` | `"bfloat16"` | bfloat16, float32, float16 |
| `model_init_kwargs` | `dict[str, Any]` | `(computed)` | trust_remote_code, attn_implementation, device_map, etc. |
| `fsdp_size` | `int` | `8` | FSDP2 number of GPUs for FSDP sharding (should match WORLD_SIZE) |
| `fsdp_wrap_granularity` | `str` | `"layer"` | layer, transformer_block |
| `fsdp_cpu_offload` | `bool` | `False` | CPUOffloadPolicy at init time |
| `fsdp_reshard_after_forward` | `bool` | `True` | free unsharded params after forward (per-LAYER wrap) |
| `fsdp_grad_reduce_dtype` | `str` | `"float32"` | Gradient reduce-scatter dtype for FSDP2 MixedPrecisionPolicy.reduce_dtype. Manual node default "float32" (reduce grads in fp32); HF/Accelerate uses "bfloat16" (set_mixed_precision('bf16') -> reduce_dtype=bf16). The one genuine FSDP-option numerical … |
| `fsdp_root_reshard_after_forward` | `bool` | `False` | reshard_after_forward for the ROOT fully_shard wrap. Manual node default False (root stays unsharded after forward so backward starts without re-gather); HF/Accelerate + TRL use True (root reshards, re-gathers for backward — saves memory). Memory/compute … |
| `fsdp_activation_checkpointing` | `bool` | `True` | checkpoint activations to save memory |
| `fsdp_mixed_precision` | `bool` | `True` | MixedPrecisionPolicy (bf16 params, fp32 grads) |
| `fsdp_autocast` | `bool` | `False` | TRL-faithful autocast: keep FSDP params in torch_dtype (fp32 master) and run the forward under torch.autocast(bf16) with logits cast to fp32, so softmax/lm_head run in fp32 (autocast op-allowlist) like HF Trainer's bf16=True (accelerator.py:1807-1818). … |
| `fsdp_sharding_strategy` | `str` | `"FULL_SHARD"` | FULL_SHARD: a 1D mesh, every rank holds a shard of every parameter. HYBRID_SHARD (multi-node): shard within each node, replicate across nodes — a (nodes, GPUs per node) mesh from torchrun's LOCAL_WORLD_SIZE — so only one gradient all-reduce per step … |
| `disable_dropout` | `bool` | `True` | Optimizations disable all dropout layers (recommended for RLHF) |
| `torch_compile` | `bool` | `False` | torch.compile the FSDP-wrapped model. Works with per-layer FSDP2, activation checkpointing and the colocate sleep/wake cycle; at 0.6B it added ~45 s of compilation to the first step and no speedup after. |
| `torch_compile_backend` | `str` | `"inductor"` |  |
| `torch_compile_mode` | `str` | `"default"` | "default", "reduce-overhead", "max-autotune" |
| `checkpoint_dir` | `str` | `(computed)` | Checkpointing Namespaced by WW_RUN_ID when set (multi-run coexistence); unchanged otherwise. Read at dataclass-construction time (process start). |
| `peft_lora` | `bool` | `False` | ------------------------------------------------------------------ PEFT / LoRA (optional). When ``peft_lora`` is False (default) the node does full fine-tuning and imports neither peft nor bitsandbytes. When True, the node attaches LoRA adapters … |
| `lora_r` | `int` | `16` | rank |
| `lora_alpha` | `int` | `32` |  |
| `lora_dropout` | `float` | `0.0` |  |
| `lora_target_modules` | `list[str]` | `(computed)` | modules to attach adapters to |
| `lora_bias` | `str` | `"none"` | "none" \| "all" \| "lora_only" |
| `loss_reduction` | `str` | `"grpo"` | GRPO loss hyperparameters (TRL defaults) "grpo" \| "dr_grpo" \| "bnpo" — aggregation strategy |
| `importance_sampling_level` | `str` | `"token"` | "token" \| "sequence" — coef_1 granularity |
| `engine_is_correction` | `bool` | `True` | Engine/trainer mismatch importance-sampling correction (mirrors TRL ``vllm_importance_sampling_correction`` / ``_mode`` / ``_cap``). TRL keeps two "old" logprob concepts separate: ``old_per_token_logps`` from a fresh detached forward pass (feeds … |
| `engine_is_mode` | `str` | `"sequence_mask"` | "sequence_mask" \| "sequence_truncate" \| "token_mask" \| "token_truncate" (TRL default sequence_mask) |
| `engine_is_cap` | `float` | `3.0` | mirror TRL vllm_importance_sampling_cap (default 3.0) |
| `kl_coef` | `float` | `0.0` | KL regularization strength (beta). 0 = disabled. TRL default since 2025. |
| `clip_range_low` | `float` | `0.2` | PPO low clipping epsilon (1 - ε_low) |
| `clip_range_high` | `float` | `0.2` | PPO high clipping epsilon (1 + ε_high) |
| `temperature` | `float` | `1.0` | logit temperature scaling during loss computation |
| `max_completion_length` | `int` | `1024` | Fixed max completion length cap, used as the dr_grpo loss denominator (per the DR-GRPO paper / TRL). Must match the rollout's max_gen_len. The loss divides by batch_size * max_completion_length (the *cap*, not the actual longest completion), making the … |
| `learning_rate` | `float` | `1e-5` | Optimizer (TRL-style) |
| `weight_decay` | `float` | `0.01` |  |
| `max_grad_norm` | `float` | `1.0` | gradient clipping |
| `optim` | `str` | `"adamw"` | adamw, sgd, paged_adamw_8bit |
| `adam_beta1` | `float` | `0.9` |  |
| `adam_beta2` | `float` | `0.999` |  |
| `adam_epsilon` | `float` | `1e-8` |  |
| `lr_scheduler_type` | `str` | `"cosine"` | Scheduler cosine, linear, constant, constant_with_warmup |
| `warmup_ratio` | `float` | `0.05` |  |
| `warmup_steps` | `int` | `0` | overrides ratio if > 0 |
| `gradient_accumulation_steps` | `int` | `1` | Training batch forward accumulations per optimizer step |
| `micro_batch_size` | `int` | `1` | per rank per time, broken into those pieces if exceeds them |
| `logits_chunk_size` | `int` | `0` | Chunked lm_head forward (memory): when >0, the lm_head + selective_log_softmax is computed in chunks of this many completion tokens, so the full [B, C, V] logits are never materialized at once (peak = one chunk). The per-token logps are assembled to [B, C] … |
| `node_id` | `str` | `"model"` | Misc |

## DataLoaderConfig — `configs/dataloader/*.yaml`

The dataloader node. Each entry of `datasets` is a mapping with `name`, `dataset_config`, `split`, `prompt_key`, `response_key`, `batch_size`, `max_samples`, `seed`, `shuffle` (see `configs/dataloader/gsm8k.yaml`).

| field | type | default | notes |
|---|---|---|---|
| `datasets` | `dict[str, dict[str, Any]]` | `(computed)` | Named datasets: {name: {config}} |
| `default_dataset` | `str` | `"train"` | Default dataset name for when no dataset_name is specified |
| `system_prompt_template` | `str` | `""` | System prompt (few-shot examples injected from the default dataset) template with {few_shot_examples} placeholder |
| `few_shot_count` | `int` | `0` | number of few-shot examples to sample |
| `few_shot_eos` | `bool` | `False` | few_shot_eos ('few-shot EOS demonstration'): the prompt keeps the plain style (instruction + 'Q:/A:' example blob in the system message, single user question) — the ONLY delta is that the model's DECLARED EOS token (tokenizer.eos_token, the token the … |
| `few_shot_seed` | `int | None` | `None` | Seed for picking the few-shot examples. None = the default dataset's `seed` (legacy), which ties the prompt to the shuffle seed: two runs that differ only in shuffle order then also train on different prompts. |
| `node_id` | `str` | `"dataloader"` | Misc |

## AsyncTrainingConfig — `configs/training/async_proxy_gsm8k.yaml`

**Experimental** (docs/async_proxy_rollout.md). The async updater loop; a TrainingConfig plus the fields below (inherited fields are listed in the TrainingConfig table).

| field | type | default | notes |
|---|---|---|---|
| `min_train_batch` | `int` | `64` | Batch assembly from the buffer pop returns nothing below this many samples |
| `max_train_batch` | `int` | `256` | pop cap per optimizer step (0 = uncapped) |
| `max_weight_version_lag` | `int` | `3` | Staleness bound K: discard buffered groups generated > K policy versions ago. K=1 ≈ near-sync; K=3 accepts more drift (engine IS ratio still corrects; TIPS_AND_TRICKS.md §4.7 favors token-level IS modes under drift). |
| `adapter_sync_every_k_steps` | `int` | `1` | Adapter sync cadence: export + proxy update_weights every K optimizer steps (1 = sync every step — the safest default; ~9 MB per sync). |
| `pop_poll_interval_s` | `float` | `2.0` | Idle behaviour when the buffer can't fill min_train_batch sleep between pop attempts |
| `step_timeout_s` | `float` | `120.0` | give up on this step after this long idle |

## RolloutProxyConfig — `configs/rollout/async_proxy_1x.yaml`

**Experimental** (docs/async_proxy_rollout.md). The rollout proxy in front of one engine node.

| field | type | default | notes |
|---|---|---|---|
| `node_id` | `str` | `"rollout"` |  |
| `backend` | `str` | `"rollout-engine"` |  |
| `buffer_max_entries` | `int` | `4096` | Replay buffer (see whirlwind/rollout/replay_buffer.py) capacity cap (samples); overflow drops oldest groups |
| `reward_ttl_s` | `float` | `300.0` | pending-reward TTL; expiry discards the group |
| `max_weight_version_lag` | `int` | `3` | Default staleness bound K (design: K=1-3 for small models). The updater sends its own max_lag on every pop; this is the fallback for direct pops. |
| `n_rpc_workers` | `int` | `2` | Serving concurrent process_rpc_requests workers (serving concurrency) |
| `log_internal_generate` | `bool` | `False` | Whether the internal ``generate`` endpoint (orchestrator/validation path) also logs to the buffer. Default False: validation rollouts must never be trained on, and a sync orchestrator pointed at the proxy already trains on its own rollouts (double-logging … |

## AgenticTrainingConfig — `configs/training/agentic_reflect_gsm8k.yaml`

**Experimental** (docs/agentic_multiturn.md). The multi-turn agentic loop; an AsyncTrainingConfig plus the fields below. Its `rollout_yaml` loads as an AgentRolloutConfig.

| field | type | default | notes |
|---|---|---|---|
| `max_weight_version_lag` | `int` | `8` | Agentic horizons span many policy versions; K is a training-semantics knob (the FIFO pop trains the oldest batch within K under oversupply). |
| `group_episodes` | `bool` | `True` | Episode grouping (the session registry): buffer entries tagged with metadata["ww_tag"] = {task, problem, group, group_size, episode, turn, ...} are staged until all `group_size` episodes of a group have arrived, and every turn of an episode gets that … |
| `group_complete_timeout_s` | `float` | `900.0` | Incomplete groups older than this are dropped (an agent crashed, or an episode never reported its reward). |
| `engine_sleep_during_update` | `bool` | `False` | Shared-GPU layout only: drain + sleep the engine around each optimizer step and wake it after the adapter sync. Dedicated engine GPUs keep this False (always-on serving). |
| `pause_drain_timeout_s` | `float` | `300.0` | Fail-closed drain bound: never sleep the engine under live requests. |

## AgentRolloutConfig — `configs/rollout/agentic_engine.yaml`

**Experimental** (docs/agentic_multiturn.md). The engine node behind the agent proxy (`rollout_sglang_agent`); a RolloutConfig plus the fields below.

| field | type | default | notes |
|---|---|---|---|
| `apply_chat_template` | `bool | None` | `None` | Chat-template handling: None = auto-detect from the tokenizer, False = force raw prompts (base models that inherit a template they were never trained on), True = force templating. |
| `chat_template_kwargs` | `dict | None` | `None` | Extra kwargs for every apply_chat_template call, e.g. Qwen3's {enable_thinking: false}. Keys the engine controls (tokenize, add_generation_prompt, tools) are rejected at startup. |
| `tool_call_parser` | `str | None` | `None` | Server-side structured tool-call parsing for the conversations + tools path: a name from SGLang's FunctionCallParser registry ("qwen25"/"hermes" = hermes JSON, "qwen3_coder" = the tool_call-wrapped XML grammar, "mistral", ...). Parsed calls ride along as … |
| `disable_radix_cache` | `bool` | `False` | Radix (prefix/KV) cache: False = enabled. Multi-turn traffic shares the whole conversation as a prefix; the cache is flushed on every adapter swap. |
| `speculative_algorithm` | `str | None` | `None` | Speculative decoding. With enable_lora, SGLang accepts only "NGRAM" (n-gram lookup drafting; no draft weights, so sleep/wake never touches it). Works with a hot-loaded adapter; it pays off only on repetitive output (tool results, code) and slowed short … |
| `speculative_num_draft_tokens` | `int | None` | `None` |  |
| `n_rpc_workers` | `int` | `1` | Concurrent RPC workers. Agents send one small generate per (episode, turn); with several workers SGLang batches them (continuous batching) instead of serving them one at a time. Weight updates, sleep and wake still run exclusively. |

## AgentProxyConfig — `configs/rollout/agentic_proxy.yaml`

**Experimental** (docs/agentic_multiturn.md). The agent proxy (`rollout_proxy_agent`); a RolloutProxyConfig plus the fields below.

| field | type | default | notes |
|---|---|---|---|
| `backend_wait_timeout_s` | `float` | `300.0` | How long a handler waits for the backend engine to register before failing. Large engines can take many minutes to capture CUDA graphs; keep it below WHIRLWIND_RPC_TIMEOUT. |
| `n_control_workers` | `int` | `2` | Control-lane RPC workers that never serve generate traffic, so pause/resume/pop/report/touch stay responsive while every serving worker holds a generate parked on the pause gate. |

## ProgressTrackerConfig — `configs/tracker/*.yaml`

The tracker node (`backend: wandb` or `none`).

| field | type | default | notes |
|---|---|---|---|
| `backend` | `str` | `"wandb"` | Backend selection "wandb" \| "none" |
| `wandb_project` | `str` | `"whirlwind"` | W&B settings |
| `wandb_entity` | `str` | `""` | empty = personal account |
| `wandb_run_name` | `str` | `""` | empty = auto-generated |
| `wandb_mode` | `str` | `"online"` | "online" \| "offline" \| "disabled" |
| `node_id` | `str` | `"tracker"` | Misc |
