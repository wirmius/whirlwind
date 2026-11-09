"""Shared configuration dataclasses for whirlwind RLHF system.

Inspired by TRL's GRPO/SFT configs. Each config loads from a YAML file via
``from_yaml`` / ``save_yaml`` classmethods.
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass, field, fields
from typing import Any

logger = logging.getLogger(__name__)

# Top-level YAML sections that are not dataclass fields but are consumed by
# another loader (e.g. a variant node's own settings block). A variant that
# reads such a section from a shared YAML adds its name here.
_EXTRA_SECTIONS: set[str] = set()


def _checkpoint_dir_default() -> str:
    """Namespaced checkpoint dir when WW_RUN_ID is set, else the default."""
    rid = os.environ.get("WW_RUN_ID")
    return f"/dev/shm/checkpoints-{rid}" if rid else "/dev/shm/checkpoints"


try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def _get_yaml() -> Any:
    """Return the yaml module, raising if missing."""
    if yaml is None:
        raise RuntimeError("PyYAML is required (pip install pyyaml)")
    return yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _substitute_env_vars(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively substitute $VAR and $VAR:default in string values.

    The substituted value is re-parsed through YAML so that, e.g.,
    ``$LR`` → ``"1e-06"`` becomes ``1e-06`` (float) automatically.

    ``$VAR:`` (with empty default) returns ``None``.
    Missing required var (no ``:``) raises ``KeyError``.
    """
    y = _get_yaml()

    def _sub(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sub(item) for item in value]
        if isinstance(value, str) and value.startswith("$"):
            rest = value[1:]  # everything after '$'
            # Split on first ':' for optional default
            if ":" in rest:
                var_name, default = rest.split(":", 1)
                env_val = os.environ.get(var_name)
                if env_val is not None:
                    raw = env_val
                else:
                    # Empty default ("$VAR:") → None; otherwise YAML-parse default
                    raw = None if default == "" else default
            else:
                env_val = os.environ.get(rest)
                if env_val is None:
                    raise KeyError(
                        f"Environment variable '{rest}' is not set "
                        f"(required by config value '{value}')"
                    )
                raw = env_val
            # If we already got a non-string (None), skip re-parsing
            if not isinstance(raw, str):
                return raw
            # Re-parse through YAML so numbers/booleans are typed correctly
            return y.safe_load(raw)
        return value

    return {k: _sub(v) for k, v in data.items()}


def _from_yaml(cls: type, path: str | pathlib.Path) -> Any:
    """Load a config dataclass from a YAML file."""
    y = _get_yaml()
    with open(path) as f:
        data = y.safe_load(f) or {}
    data = _substitute_env_vars(data)
    valid = {f.name for f in fields(cls)}
    # Unknown keys used to be dropped silently, so a typo or a key left over from
    # a removed option quietly ran with the default. Fail loudly.
    unknown = sorted(set(data) - valid - _EXTRA_SECTIONS)
    if unknown:
        raise ValueError(
            f"{path}: unknown {cls.__name__} key(s) {unknown} (typo, or a removed option?)"
        )
    filtered = {k: v for k, v in data.items() if k in valid}
    return cls(**filtered)  # type: ignore[arg-type]


def _save_yaml(cls: Any, path: str | pathlib.Path) -> None:
    """Dump a config instance to a YAML file."""
    y = _get_yaml()
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        y.dump(cls.__dict__, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# RolloutConfig — SGLang engine settings
# ---------------------------------------------------------------------------


@dataclass
class RolloutConfig:
    """Configuration for the SGLang rollout node.

    Parameters match common SGLang engine args and sampling hyperparameters.
    """

    # Model
    model_path: str = ""  # HuggingFace / local model path
    load_format: str = "auto"  # auto, safetensors, dummy, ...

    # CUDA device assignment (for bringup script / torchrun compatibility)
    # cuda_devices: str = ""  # e.g. "0" or "0,1" — sets CUDA_VISIBLE_DEVICES

    # Tensor parallelism
    tp_size: int = 1  # number of GPUs for SGLang TP
    dp_size: int = 1  # number of data parallel workers

    # Context / generation limits
    context_length: int = 4096  # max context window
    max_gen_len: int = 1024  # max completion length
    # sglang allow_auto_truncate: when a request's prompt + max_new_tokens
    # exceeds context_length, trim max_new_tokens per request instead of
    # hard-rejecting (REQUIRED when max_gen_len leaves little prompt head-
    # room — the 8196-window runs died on first generate otherwise).
    allow_auto_truncate: bool = False

    # Sampling hyperparameters (TRL-style)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 = disabled
    repetition_penalty: float = 1.0

    # Extra generation params (forwarded to SGLang SamplingParams)
    # e.g. min_p, skip_special_tokens, spaces_between_special_tokens
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


    # Engine knobs
    mem_fraction_static: float = 0.9  # SGLang KV cache fraction
    chunked_prefill_size: int = 128
    schedule_policy: str = "lpm"  # lpm, fcfs
    # torch_memory_saver (memory_saver / cpu_backup) is only useful when the
    # orchestrator performs co-located sleep/wake during the RL cycle, i.e.
    # when ``TrainingConfig.memory_management_enabled`` is True. With it off,
    # leaving ``enable_memory_saver`` on allocates the KV-cache pool under a
    # custom CUDA MemPool that is never paused/resumed — a state SGLang's
    # weight-update path can leave inaccessible to Triton (sglang#10025).
    # Keep both flags in sync with ``memory_management_enabled``.
    enable_memory_saver: bool = False  # mirror TrainingConfig.memory_management_enabled
    enable_weights_cpu_backup: bool = False  # mirror TrainingConfig.memory_management_enabled

    # LoRA (native adapter reload). When ``enable_lora`` is True, the engine is
    # initialized with LoRA support and ``_update_weights`` loads the exported
    # adapter dir (adapter_config.json + adapter_model.safetensors) via the
    # engine's native add_lora / load_lora_adapter, instead of swapping full
    # weights. ``lora_max_rank`` must be >= the training lora_r. Set in sync
    # with the model config's peft_lora. See docs/lora_design.md.
    enable_lora: bool = False
    lora_max_rank: int = 64  # vLLM max_lora_rank / SGLang --max-lora-rank
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])  # SGLang --lora-target-modules

    # Disable SGLang CUDA graph capture (decode). Colocate (rollout+FSDP share
    # GPUs) can hit ``illegal memory access`` in the decode ``copy_done`` during
    # cuda-graph REPLAY when an FSDP2 process co-resides on the GPU — the
    # graph's static buffers are invalidated by the constrained/shared memory
    # pool (val@0 runs eager at >max_bs and succeeds; a 256-seq step-1 decode in
    # the graph-capture range crashes; isolated SGLang at 256 seq is fine).
    # Disabling costs ~1.5-2x decode throughput but makes colocate stable.
    disable_cuda_graph: bool = False

    # SGLang RL on-policy target. When set (e.g. "fsdp"), SGLang computes
    # per-token logprobs via torch.log_softmax(logits/temp) instead of the
    # standard log(softmax(logits/temp)). The standard path is numerically
    # unstable in bf16 (softmax→0 then log→-inf), producing ~0.02/token drift
    # from the trainer's forward-pass logprobs. With sequence_mask IS
    # correction this drift accumulates over hundreds of tokens then exp()'s,
    # creating a large spurious IS-ratio mismatch. The log_softmax path
    # matches the trainer's numerics exactly, so the IS ratio stays ≈1.0.
    # See sglang/srt/layers/sampler.py:144-154. Currently only "fsdp" is valid.
    rl_on_policy_target: str | None = None

    # Table logging override
    log_every_call_override: int = (
        -1
    )  # force table logging every N calls (-1 = disabled, use orchestrator's decision)

    # Misc
    node_id: str = "rollout"

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> RolloutConfig:
        return _from_yaml(cls, path)

    def save_yaml(self, path: str | pathlib.Path) -> None:
        _save_yaml(self, path)


# ---------------------------------------------------------------------------
# RolloutPoolConfig (SGLang multi-engine pool)
# ---------------------------------------------------------------------------


@dataclass
class RolloutPoolConfig:
    """Configuration for the SGLang rollout pool router node.

    A thin router that registers as one ``node_id`` (the single endpoint the
    orchestrator calls) and fans out to N backend rollout engines (each a
    separate ``rollout_sglang`` process on its own GPU). The pool shards the
    generate batch across backends (parallel ``asyncio.gather``), broadcasts
    update_weights / sleep / wake to all, and gathers results in order.

    The backends are configured via the existing :class:`RolloutConfig` (one
    per engine, each ``dp_size=1`` so SGLang's ``load_lora_adapter`` assert
    passes per-engine — the point of the pool: unblocks multi-GPU LoRA).

    Attributes:
        node_id:        the orchestrator's single rollout endpoint (default
                        "rollout" — the orchestrator is unchanged).
        backends:       backend engine node_ids to fan out to (e.g.
                        ["rollout0", "rollout1"]). Each must be a running
                        rollout node registered under that id.
    """

    node_id: str = "rollout"
    backends: list[str] = field(default_factory=lambda: ["rollout0", "rollout1"])

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> RolloutPoolConfig:
        return _from_yaml(cls, path)

    def save_yaml(self, path: str | pathlib.Path) -> None:
        _save_yaml(self, path)


# ---------------------------------------------------------------------------
# RolloutProxyConfig (async proxy-rollout: always-on serving + replay buffer)
# ---------------------------------------------------------------------------


@dataclass
class RolloutProxyConfig:
    """Configuration for the rollout proxy node (async actor-learner mode).

    The proxy registers as ONE rollout endpoint (``node_id``, default
    "rollout" — the orchestrator is unchanged) and fronts a single backend
    rollout node (``backend``, default "rollout-engine" — any engine module:
    sglang / vllm / the pool). It adds four things the plain engines don't
    have (EXPERIMENTAL; see ``docs/async_proxy_rollout.md``):

      1. ``generate_external``  — the external agentic system's serving API;
         every served rollout is logged to the replay buffer tagged with the
         current ``weight_version``.
      2. ``report_reward``      — the reward channel: out-of-order, delayed,
         batched scalar rewards keyed by ``rollout_id``.
      3. ``pop_ready_batch``    — the async updater's pull (whole ready GRPO
         groups, staleness-bounded).
      4. a ``weight_version`` counter bumped on every successful
         ``update_weights`` (LoRA adapter hot-swap is the intended sync —
         ~9 MB, no engine restart, no sleep/wake).

    The proxy never touches engines directly — the backend is selected with
    the existing ``--rollout-module`` / node config machinery, so a pool of
    engines works unchanged behind the proxy.

    Generation hyperparameters (temperature etc.) live in the BACKEND's
    :class:`RolloutConfig`; the orchestrator's ``rollout_yaml`` should point
    at the backend config too (same convention as the pool drivers).
    """

    node_id: str = "rollout"
    backend: str = "rollout-engine"

    # Replay buffer (see whirlwind/rollout/replay_buffer.py)
    buffer_max_entries: int = 4096  # capacity cap (samples); overflow drops oldest groups
    reward_ttl_s: float = 300.0  # pending-reward TTL; expiry discards the group
    # Default staleness bound K (design: K=1-3 for small models). The updater
    # sends its own max_lag on every pop; this is the fallback for direct pops.
    max_weight_version_lag: int = 3

    # Serving
    n_rpc_workers: int = 2  # concurrent process_rpc_requests workers (serving concurrency)

    # Whether the internal ``generate`` endpoint (orchestrator/validation path)
    # also logs to the buffer. Default False: validation rollouts must never be
    # trained on, and a sync orchestrator pointed at the proxy already trains
    # on its own rollouts (double-logging would double-train). Enable only for
    # deliberate hybrid experiments.
    log_internal_generate: bool = False

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> RolloutProxyConfig:
        return _from_yaml(cls, path)

    def save_yaml(self, path: str | pathlib.Path) -> None:
        _save_yaml(self, path)


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Configuration for the reward computation node.

    Supports function-based rewards (length, accuracy) and is
    extensible to learned reward models.

    ``reward_type`` is mandatory — there is no default.
    """

    # Reward type (mandatory — no default)
    reward_type: str  # "length" | "accuracy" | "model" | "custom"

    # Length-based reward params
    length_scale: float = 1.0  # multiplier for raw length reward
    length_min: int = 1  # minimum count for normalization

    # Group normalization (TRL GRPO-style)
    group_size: int = 16  # items per normalization group
    scale_rewards: str = "group"  # "group" | "batch" | "none" — std scope for z-score

    # Table logging override
    log_every_call_override: int = (
        -1
    )  # force table logging every N calls (-1 = disabled, use orchestrator's decision)

    # Misc
    node_id: str = "reward"

    # Accuracy reward params (for GSM8K-style datasets)
    accuracy_regex: str = r"#####?\s*(\d[\d,]*)"  # pattern to extract answer from text

    # Truncation masking (TRL GRPOConfig.mask_truncated_completions)
    mask_truncated_completions: bool = False  # zero advantage for length-truncated samples

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> RewardConfig:
        return _from_yaml(cls, path)

    def save_yaml(self, path: str | pathlib.Path) -> None:
        _save_yaml(self, path)


# ---------------------------------------------------------------------------
# ModelConfig — FSDP2 + optimizer settings
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for the FSDP2 model-update node.

    Mirrors TRL SFTTrainer / GRPOTrainer optimizer & FSDP knobs.
    """

    # Model
    model_path: str = ""  # initial checkpoint / HF model
    torch_dtype: str = "bfloat16"  # bfloat16, float32, float16
    model_init_kwargs: dict[str, Any] = field(
        default_factory=dict,  # trust_remote_code, attn_implementation, device_map, etc.
    )

    # FSDP2
    fsdp_size: int = 8  # number of GPUs for FSDP sharding (should match WORLD_SIZE)
    fsdp_wrap_granularity: str = "layer"  # layer, transformer_block
    fsdp_cpu_offload: bool = False  # CPUOffloadPolicy at init time
    fsdp_reshard_after_forward: bool = True  # free unsharded params after forward (per-LAYER wrap)
    # Gradient reduce-scatter dtype for FSDP2 MixedPrecisionPolicy.reduce_dtype.
    # Manual node default "float32" (reduce grads in fp32); HF/Accelerate uses
    # "bfloat16" (set_mixed_precision('bf16') -> reduce_dtype=bf16). The one
    # genuine FSDP-option numerical difference vs the HF path. Set "bfloat16"
    # to align to HF for the manual-vs-HF A/B.
    fsdp_grad_reduce_dtype: str = "float32"
    # reshard_after_forward for the ROOT fully_shard wrap. Manual node default
    # False (root stays unsharded after forward so backward starts without
    # re-gather); HF/Accelerate + TRL use True (root reshards, re-gathers for
    # backward — saves memory). Memory/compute tradeoff, not obviously
    # numerical. Set True to align to HF for the A/B.
    fsdp_root_reshard_after_forward: bool = False
    fsdp_activation_checkpointing: bool = True  # checkpoint activations to save memory
    fsdp_mixed_precision: bool = True  # MixedPrecisionPolicy (bf16 params, fp32 grads)
    # TRL-faithful autocast: keep FSDP params in torch_dtype (fp32 master) and
    # run the forward under torch.autocast(bf16) with logits cast to fp32, so
    # softmax/lm_head run in fp32 (autocast op-allowlist) like HF Trainer's
    # bf16=True (accelerator.py:1807-1818). Overrides fsdp_mixed_precision's
    # param_dtype/output_dtype cast when true. Default false = unchanged.
    fsdp_autocast: bool = False
    # FULL_SHARD: a 1D mesh, every rank holds a shard of every parameter.
    # HYBRID_SHARD (multi-node): shard within each node, replicate across
    # nodes — a (nodes, GPUs per node) mesh from torchrun's LOCAL_WORLD_SIZE —
    # so only one gradient all-reduce per step crosses nodes. On one node it
    # equals FULL_SHARD. Multi-node launch: docs/launch.md. FSDP2 node only;
    # the HF-Trainer and accelerate siblings always shard over every rank.
    fsdp_sharding_strategy: str = "FULL_SHARD"

    # Optimizations
    disable_dropout: bool = True  # disable all dropout layers (recommended for RLHF)
    # torch.compile the FSDP-wrapped model. Works with per-layer FSDP2,
    # activation checkpointing and the colocate sleep/wake cycle; at 0.6B it
    # added ~45 s of compilation to the first step and no speedup after.
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"

    # Checkpointing
    # Namespaced by WW_RUN_ID when set (multi-run coexistence); unchanged
    # otherwise. Read at dataclass-construction time (process start).
    checkpoint_dir: str = field(default_factory=_checkpoint_dir_default)

    # ------------------------------------------------------------------
    # PEFT / LoRA (optional). When ``peft_lora`` is False (default) the node
    # does full fine-tuning and imports neither peft nor bitsandbytes. When
    # True, the node attaches LoRA adapters (trainable) over a frozen base;
    # weight export is adapter-only (MB) and the rollout loads the adapter
    # natively (vLLM ``add_lora`` / SGLang ``load_lora_adapter``) instead of a
    # full merged model. See docs/lora_design.md. peft must be installed
    # (gated import — the node raises a clear error if peft_lora=True but peft
    # is missing, rather than crashing at import time).
    peft_lora: bool = False
    lora_r: int = 16  # rank
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(  # modules to attach adapters to
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_bias: str = "none"  # "none" | "all" | "lora_only"

    # GRPO loss hyperparameters (TRL defaults)
    loss_reduction: str = "grpo"  # "grpo" | "dr_grpo" | "bnpo" — aggregation strategy
    importance_sampling_level: str = "token"  # "token" | "sequence" — coef_1 granularity
    # Engine/trainer mismatch importance-sampling correction (mirrors TRL
    # ``vllm_importance_sampling_correction`` / ``_mode`` / ``_cap``).
    # TRL keeps two "old" logprob concepts separate: ``old_per_token_logps``
    # from a fresh detached forward pass (feeds ``coef_1``, so ``coef_1``≈1.0
    # and PPO clipping rarely fires), and ``sampling_per_token_logps`` from the
    # generation engine (vLLM/SGLang) used ONLY for a bounded IS ratio that
    # corrects the engine↔trainer distribution mismatch. Whirlwind previously
    # fed SGLang's generation logprobs straight into ``coef_1``, contaminating
    # the PPO ratio with engine numerics. With ``engine_is_correction=True``
    # we compute a detached forward for ``old_per_token_logps`` and apply the
    # bounded IS ratio from the SGLang logprobs, matching TRL exactly.
    engine_is_correction: bool = (
        True  # mirror TRL vllm_importance_sampling_correction (default True)
    )
    engine_is_mode: str = "sequence_mask"  # "sequence_mask" | "sequence_truncate" | "token_mask" | "token_truncate" (TRL default sequence_mask)
    engine_is_cap: float = 3.0  # mirror TRL vllm_importance_sampling_cap (default 3.0)
    kl_coef: float = 0.0  # KL regularization strength (beta). 0 = disabled. TRL default since 2025.
    clip_range_low: float = 0.2  # PPO low clipping epsilon (1 - ε_low)
    clip_range_high: float = 0.2  # PPO high clipping epsilon (1 + ε_high)
    temperature: float = 1.0  # logit temperature scaling during loss computation
    # Fixed max completion length cap, used as the dr_grpo loss denominator
    # (per the DR-GRPO paper / TRL). Must match the rollout's max_gen_len.
    # The loss divides by batch_size * max_completion_length (the *cap*, not
    # the actual longest completion), making the loss length-invariant.
    max_completion_length: int = 1024

    # Optimizer (TRL-style)
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # gradient clipping
    optim: str = "adamw"  # adamw, sgd, paged_adamw_8bit
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Scheduler
    lr_scheduler_type: str = "cosine"  # cosine, linear, constant, constant_with_warmup
    warmup_ratio: float = 0.05
    warmup_steps: int = 0  # overrides ratio if > 0

    # Training batch
    gradient_accumulation_steps: int = 1  # forward accumulations per optimizer step
    micro_batch_size: int = 1  # per rank per time, broken into those pieces if exceeds them
    # Chunked lm_head forward (memory): when >0, the lm_head + selective_log_softmax
    # is computed in chunks of this many completion tokens, so the full
    # [B, C, V] logits are never materialized at once (peak = one chunk).
    # The per-token logps are assembled to [B, C] and the STANDARD loss is
    # called once with global normalizers — the loss is not modified. Works for
    # all reductions (grpo/dr_grpo/bnpo/ce). 0 = off (single forward).
    logits_chunk_size: int = 0

    # Misc
    node_id: str = "model"

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> ModelConfig:
        return _from_yaml(cls, path)

    def save_yaml(self, path: str | pathlib.Path) -> None:
        _save_yaml(self, path)


# ---------------------------------------------------------------------------
# TrainingConfig — top-level orchestrator settings
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for the training orchestrator.

    Controls the GRPO / general RL loop: rollout count, logging, stopping.

    Loads sub-configs (RolloutConfig, RewardConfig, ModelConfig) from YAML
    during ``from_yaml`` so the orchestrator can access rollout/model/reward
    settings via ``config.rollout``, ``config.model``, ``config.reward``.
    """

    # Sub-config YAML paths (each node loads its own file; bringup exports WHIRLWIND_*_CONFIG)
    rollout_yaml: str = "rollout.yaml"
    reward_yaml: str = "reward.yaml"
    model_yaml: str = "model.yaml"

    # Loaded sub-config objects (populated by from_yaml)
    rollout: RolloutConfig = None  # type: ignore[assignment]
    reward: RewardConfig = None  # type: ignore[assignment]
    model: ModelConfig = None  # type: ignore[assignment]

    # Loop control
    total_steps: int = 1000
    # Horizon of the LR schedule (warmup + decay); 0 = total_steps. A 65-step
    # run with lr_schedule_steps 192 follows the first 65 steps of a 192-step
    # run, so runs of different length compare at the same per-step LR.
    lr_schedule_steps: int = 0
    rollout_per_prompt: int = 4  # N generations per prompt (GRPO group size)
    log_every_n_steps: int = 1
    memory_management_enabled: bool = True  # sleep/wake nodes between phases

    # Generation cadence (TRL ``steps_per_generation``). Generate a batch of
    # completions once, then run ``steps_per_generation`` optimizer steps over
    # contiguous slices of it (each slice = one GRPO group batch), syncing the
    # rollout weights only once per generation. 1 = legacy behaviour (generate
    # every step). >1 decouples generation from optimization: fewer (expensive)
    # rollouts per optimizer step, and each per-step slice is small enough to
    # forward in one shot — replacing the per-step micro/nano-batch split as
    # the batch-scaling strategy. The engine IS correction (token-level) still
    # corrects the growing off-policy drift across the slices.
    steps_per_generation: int = 1

    # Failure handling: if a required node is detected as failed (RPC error or
    # heartbeat timeout), broadcast a cluster-wide graceful shutdown to all
    # nodes and stop, instead of spinning on the dead node. Active by default.
    shutdown_on_node_failure: bool = True

    # Table logging (moved from sub-node configs; orchestrator controls step-based gating)
    rollout_log_table_every_n_steps: int = -1  # log rollout table every N steps (-1 = disabled)
    reward_log_table_every_n_steps: int = -1  # log reward table every N steps (-1 = disabled)

    # Validation
    validation_every_n_steps: int = 0  # run validation every N optimizer steps (0 = disabled)
    validation_at_step_0: bool = True  # run validation before any training
    # Log val rollouts/rewards as W&B tables. At long completion budgets the
    # full-text serialization (~30-60MB) stalls the tracker RPC long enough
    # to trip the heartbeat watchdogs — set false for 8k-context runs.
    validation_log_table: bool = True

    # Artifacts & checkpointing
    workdir: str = ""  # working directory for checkpoints, logs, configs (empty = disabled)
    hot_checkpoint_every_n_steps: int = 0  # hot checkpoint every N optimizer updates (0 = disabled)
    cold_checkpoint_steps: list = field(default_factory=list)  # list of specific steps

    # Misc
    seed: int = 42
    orchestrator_node_id: str = "orchestrator"

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> TrainingConfig:
        path = pathlib.Path(path)
        base = path.parent
        config = _from_yaml(cls, path)
        # Load sub-configs (each component still loads its own YAML independently)
        config.rollout = RolloutConfig.from_yaml(base / config.rollout_yaml)
        config.reward = RewardConfig.from_yaml(base / config.reward_yaml)
        config.model = ModelConfig.from_yaml(base / config.model_yaml)
        # Cross-config consistency: these pairs must agree but live in
        # different YAMLs. Warn rather than raise: the length mismatch only
        # rescales the dr_grpo loss, the temperature mismatch breaks the engine
        # IS ratio (see TIPS_AND_TRICKS.md) and deserves a hard look either way.
        if config.model.max_completion_length != config.rollout.max_gen_len:
            logger.warning("model.max_completion_length=%d != rollout.max_gen_len=%d (dr_grpo cap vs generation cap)",
                           config.model.max_completion_length, config.rollout.max_gen_len)
        if abs(config.model.temperature - config.rollout.temperature) > 1e-9:
            logger.warning("model.temperature=%.3f != rollout.temperature=%.3f (engine logprobs and trainer logits must share one temperature)",
                           config.model.temperature, config.rollout.temperature)
        return config  # type: ignore[return-value]

    def save_yaml(self, path: str | pathlib.Path) -> None:
        """Dump this config to YAML (excludes loaded sub-config objects)."""
        if yaml is None:
            raise RuntimeError("PyYAML is required (pip install pyyaml)")
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclude the sub-config objects; only write scalar fields + paths
        exclude = {"rollout", "reward", "model"}
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in exclude}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# AsyncTrainingConfig — buffer-driven (actor-learner) orchestrator settings
# ---------------------------------------------------------------------------


@dataclass
class AsyncTrainingConfig(TrainingConfig):
    """Config for the async GRPO orchestrator (buffer-driven actor-learner). EXPERIMENTAL.

    Subclass of :class:`TrainingConfig` used by
    ``whirlwind.training.training_async.AsyncGRPOOrchestrator``. The loop:
    pop ready (externally rewarded) groups from the rollout proxy → compute
    GRPO advantages locally → forward/backward + optimizer step → every K
    steps export weights and hot-swap the proxy's policy (bumping its
    ``weight_version``). No dataloader in the training path — generation is
    driven by the external agentic system; the dataloader/reward nodes are
    only needed for periodic validation.

    Non-colocate by construction (the proxy is always-on); sleep/wake is
    skipped even if ``memory_management_enabled`` is left true.
    """

    # Batch assembly from the buffer
    min_train_batch: int = 64  # pop returns nothing below this many samples
    max_train_batch: int = 256  # pop cap per optimizer step (0 = uncapped)
    # Staleness bound K: discard buffered groups generated > K policy versions
    # ago. K=1 ≈ near-sync; K=3 accepts more drift (engine IS ratio still
    # corrects; TIPS_AND_TRICKS.md §4.7 favors token-level IS modes under drift).
    max_weight_version_lag: int = 3

    # Adapter sync cadence: export + proxy update_weights every K optimizer
    # steps (1 = sync every step — the safest default; ~9 MB per sync).
    adapter_sync_every_k_steps: int = 1

    # Idle behaviour when the buffer can't fill min_train_batch
    pop_poll_interval_s: float = 2.0  # sleep between pop attempts
    step_timeout_s: float = 120.0  # give up on this step after this long idle

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> AsyncTrainingConfig:
        path = pathlib.Path(path)
        base = path.parent
        config = _from_yaml(cls, path)
        config.rollout = RolloutConfig.from_yaml(base / config.rollout_yaml)
        config.reward = RewardConfig.from_yaml(base / config.reward_yaml)
        config.model = ModelConfig.from_yaml(base / config.model_yaml)
        return config  # type: ignore[return-value]

    def save_yaml(self, path: str | pathlib.Path) -> None:
        """Dump this config to YAML (excludes loaded sub-config objects)."""
        if yaml is None:
            raise RuntimeError("PyYAML is required (pip install pyyaml)")
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        exclude = {"rollout", "reward", "model"}
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in exclude}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# ProgressTrackerConfig — experiment tracking settings
# ---------------------------------------------------------------------------


@dataclass
class ProgressTrackerConfig:
    """Configuration for the progress-tracker node (W&B backend).

    When ``backend`` is ``"none"`` the tracker node is a no-op —
    the orchestrator simply skips starting it.
    """

    # Backend selection
    backend: str = "wandb"  # "wandb" | "none"

    # W&B settings
    wandb_project: str = "whirlwind"
    wandb_entity: str = ""  # empty = personal account
    wandb_run_name: str = ""  # empty = auto-generated
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"

    # Misc
    node_id: str = "tracker"

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> ProgressTrackerConfig:
        return _from_yaml(cls, path)

    def save_yaml(self, path: str | pathlib.Path) -> None:
        _save_yaml(self, path)


# ---------------------------------------------------------------------------
# DataLoaderConfig — dataset and dataloader settings
# ---------------------------------------------------------------------------


@dataclass
class DataLoaderConfig:
    """Configuration for the dataloader node.

    Manages multiple named datasets (train, val, etc.) with independent
    shuffling, batching, and state tracking.
    """

    # Named datasets: {name: {config}}
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Default dataset name for when no dataset_name is specified
    default_dataset: str = "train"

    # System prompt (few-shot examples injected from the default dataset)
    system_prompt_template: str = ""  # template with {few_shot_examples} placeholder
    few_shot_count: int = 0  # number of few-shot examples to sample
    # few_shot_eos ('few-shot EOS demonstration'): the prompt keeps the plain
    # style (instruction + 'Q:/A:' example blob in the system
    # message, single user question) — the ONLY delta is that the model's
    # DECLARED EOS token (tokenizer.eos_token, the token the engine stops
    # on) is appended after each few-shot example, so the in-context pattern
    # becomes 'answer → EOS' instead of 'answer → \n\nQ:'. When true: the
    # system prompt carries the instruction only; the example texts travel
    # on PromptBatch.prefix_messages and the ROLLOUT folds them back into
    # the old blob style with the EOS appended (the tokenizer lives there).
    # False (default) = the legacy blob, bit-identical prompts.
    few_shot_eos: bool = False
    # Seed for picking the few-shot examples. None = the default dataset's
    # `seed` (legacy), which ties the prompt to the shuffle seed: two runs that
    # differ only in shuffle order then also train on different prompts.
    few_shot_seed: int | None = None

    # Misc
    node_id: str = "dataloader"

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> DataLoaderConfig:
        return _from_yaml(cls, path)

    def save_yaml(self, path: str | pathlib.Path) -> None:
        _save_yaml(self, path)
