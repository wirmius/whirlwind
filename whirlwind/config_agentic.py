"""Config dataclasses for the experimental agentic (multi-turn) mode.

Each class extends a base config with the knobs only the agentic siblings
read, so the base nodes keep rejecting them as unknown keys:

* :class:`AgentRolloutConfig` — ``whirlwind.rollout.rollout_sglang_agent``
  (chat conversations + tools, server-side tool-call parsing, template
  control, speculative decoding);
* :class:`AgentProxyConfig` — ``whirlwind.rollout.rollout_proxy_agent``
  (episode heartbeats, engine pause/resume for shared-GPU updates);
* :class:`AgenticTrainingConfig` — ``whirlwind.training.training_agentic``
  (episode grouping, shared-GPU update choreography).

See docs/agentic_multiturn.md.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from whirlwind.config import (
    AsyncTrainingConfig,
    ModelConfig,
    RewardConfig,
    RolloutConfig,
    RolloutProxyConfig,
    _from_yaml,
)


@dataclass
class AgentRolloutConfig(RolloutConfig):
    """Rollout engine config for agent traffic (experimental)."""

    # Chat-template handling: None = auto-detect from the tokenizer, False =
    # force raw prompts (base models that inherit a template they were never
    # trained on), True = force templating.
    apply_chat_template: bool | None = None
    # Extra kwargs for every apply_chat_template call, e.g. Qwen3's
    # {enable_thinking: false}. Keys the engine controls (tokenize,
    # add_generation_prompt, tools) are rejected at startup.
    chat_template_kwargs: dict | None = None
    # Server-side structured tool-call parsing for the conversations + tools
    # path: a name from SGLang's FunctionCallParser registry ("qwen25"/"hermes"
    # = hermes JSON, "qwen3_coder" = the tool_call-wrapped XML grammar,
    # "mistral", ...). Parsed calls ride along as metadata
    # (RolloutResult.tool_calls); the raw completion text stays canonical for
    # training. None = off (agents parse the text themselves).
    tool_call_parser: str | None = None
    # Radix (prefix/KV) cache: False = enabled. Multi-turn traffic shares the
    # whole conversation as a prefix; the cache is flushed on every adapter swap.
    disable_radix_cache: bool = False
    # Speculative decoding. With enable_lora, SGLang accepts only "NGRAM"
    # (n-gram lookup drafting; no draft weights, so sleep/wake never touches
    # it). Works with a hot-loaded adapter; it pays off only on repetitive
    # output (tool results, code) and slowed short free-form answers down.
    speculative_algorithm: str | None = None
    speculative_num_draft_tokens: int | None = None
    # Concurrent RPC workers. Agents send one small generate per (episode,
    # turn); with several workers SGLang batches them (continuous batching)
    # instead of serving them one at a time. Weight updates, sleep and wake
    # still run exclusively.
    n_rpc_workers: int = 1

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> AgentRolloutConfig:
        return _from_yaml(cls, path)


@dataclass
class AgentProxyConfig(RolloutProxyConfig):
    """Rollout proxy config for agent traffic (experimental)."""

    # How long a handler waits for the backend engine to register before
    # failing. Large engines can take many minutes to capture CUDA graphs;
    # keep it below WHIRLWIND_RPC_TIMEOUT.
    backend_wait_timeout_s: float = 300.0
    # Control-lane RPC workers that never serve generate traffic, so
    # pause/resume/pop/report/touch stay responsive while every serving worker
    # holds a generate parked on the pause gate.
    n_control_workers: int = 2

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> AgentProxyConfig:
        return _from_yaml(cls, path)


@dataclass
class AgenticTrainingConfig(AsyncTrainingConfig):
    """Agentic (multi-turn) async training loop config (experimental)."""

    # Agentic horizons span many policy versions; K is a training-semantics
    # knob (the FIFO pop trains the oldest batch within K under oversupply).
    max_weight_version_lag: int = 8
    # Episode grouping (the session registry): buffer entries tagged with
    # metadata["ww_tag"] = {task, problem, group, group_size, episode, turn, ...}
    # are staged until all `group_size` episodes of a group have arrived, and
    # every turn of an episode gets that episode's group-relative advantage.
    # Untagged entries train as plain single-turn groups.
    group_episodes: bool = True
    # Incomplete groups older than this are dropped (an agent crashed, or an
    # episode never reported its reward).
    group_complete_timeout_s: float = 900.0
    # Shared-GPU layout only: drain + sleep the engine around each optimizer
    # step and wake it after the adapter sync. Dedicated engine GPUs keep this
    # False (always-on serving).
    engine_sleep_during_update: bool = False
    # Fail-closed drain bound: never sleep the engine under live requests.
    pause_drain_timeout_s: float = 300.0

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> AgenticTrainingConfig:
        path = pathlib.Path(path)
        base = path.parent
        config = _from_yaml(cls, path)
        # The engine config may carry agent-only keys.
        config.rollout = AgentRolloutConfig.from_yaml(base / config.rollout_yaml)
        config.reward = RewardConfig.from_yaml(base / config.reward_yaml)
        config.model = ModelConfig.from_yaml(base / config.model_yaml)
        return config  # type: ignore[return-value]
