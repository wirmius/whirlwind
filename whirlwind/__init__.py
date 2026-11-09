"""Whirlwind — RLHF training system with ZeroMQ RPC communication.

Subpackages:
    whirlcoms — RPC communication (Node, Orchestrator, transport)
    rollout — SGLang rollout node
    reward — function-based reward computation
    model — FSDP2 training node
    tracker — W&B progress tracking
    training — GRPO training orchestrator
    data — HuggingFace dataset loader

Modules:
    config — configuration dataclasses (YAML-serializable)
    messages — shared message/data classes

All public symbols listed in ``__dir__`` are exported lazily via
``__getattr__``.  Importing ``whirlwind`` alone does NOT pull in any
heavy back-ends (torch, sglang, etc.).  Each symbol is resolved on first
access and then cached on the module.

Usage:

    # Short form — lazy, only loads what you use
    from whirlwind import TrainingOrchestrator, TrainingConfig

    # Long form — same effect, explicit module path
    from whirlwind.training import TrainingOrchestrator
    from whirlwind.config import TrainingConfig
"""

from __future__ import annotations

__version__ = "0.4.0"


# ---------------------------------------------------------------------------
# Lazy exports (PEP 562)
#
# Every public symbol is resolved on first access.  The resolved module
# object is stored on ``whirlwind`` so subsequent attribute reads hit the
# cached value and ``__getattr__`` is skipped.
# ---------------------------------------------------------------------------


def __getattr__(name: str):  # noqa: C901
    if name in (
        "RolloutConfig",
        "RewardConfig",
        "ModelConfig",
        "TrainingConfig",
        "ProgressTrackerConfig",
        "DataLoaderConfig",
    ):
        from whirlwind import config

        return getattr(config, name)

    if name in (
        "BatchedMessage",
        "BatchRequest",
        "BatchResult",
        "CheckpointRequest",
        "CheckpointResult",
        "MetricsBatch",
        "ModelUpdateRequest",
        "OperationResult",
        "StateGetRequest",
        "StateResult",
        "StateSetRequest",
        "TopKLogitsRequest",
        "TopKLogitsResult",
    ):
        from whirlwind import messages

        return getattr(messages, name)

    if name == "DatasetLoader":
        from whirlwind.data import DatasetLoader

        return DatasetLoader

    if name == "RolloutNode":
        from whirlwind.rollout import RolloutNode

        return RolloutNode

    if name == "RewardNode":
        from whirlwind.reward import RewardNode

        return RewardNode

    if name == "ModelUpdateNode":
        from whirlwind.model import ModelUpdateNode

        return ModelUpdateNode

    if name == "ProgressTrackerNode":
        from whirlwind.tracker import ProgressTrackerNode

        return ProgressTrackerNode

    if name == "DataLoaderNode":
        from whirlwind.data import DataLoaderNode

        return DataLoaderNode

    if name == "TrainingOrchestrator":
        from whirlwind.training import TrainingOrchestrator

        return TrainingOrchestrator

    if name == "DatasetLoader":
        from whirlwind.data import DatasetLoader

        return DatasetLoader

    if name == "compute_lr":
        from whirlwind import utils

        return utils.compute_lr

    if name in (
        "Orchestrator",
        "Node",
        "RpcError",
        "RpcClient",
        "call",
    ):
        from whirlwind import whirlcoms

        return getattr(whirlcoms, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return [
        # whirlcoms
        "Orchestrator",
        "Node",
        "RpcError",
        "RpcClient",
        "call",
        # utils
        "compute_lr",
        # config
        "RolloutConfig",
        "RewardConfig",
        "ModelConfig",
        "TrainingConfig",
        "ProgressTrackerConfig",
        "DataLoaderConfig",
        # messages
        "BatchedMessage",
        "BatchRequest",
        "BatchResult",
        "CheckpointRequest",
        "CheckpointResult",
        "MetricsBatch",
        "ModelUpdateRequest",
        "OperationResult",
        "StateGetRequest",
        "StateResult",
        "StateSetRequest",
        "TopKLogitsRequest",
        "TopKLogitsResult",
        # nodes
        "RolloutNode",
        "RewardNode",
        "ModelUpdateNode",
        "ProgressTrackerNode",
        "TrainingOrchestrator",
        # dataloader
        "DatasetLoader",
    ]
