"""Validate whirlwind config files.

Usage:
    python3 validate_configs.py configs/rollout/sglang.yaml configs/reward/function.yaml ...
    python3 validate_configs.py configs/*/*.yaml

The config type is taken from the path (configs/<kind>/...). Where a kind has
several config classes (a rollout engine, a pool of engines, the experimental
async proxy, the experimental agentic engine and proxy; the synchronous, the
experimental async or the experimental agentic training loop), the first
class that loads the file without unknown keys wins.
"""

from __future__ import annotations

import sys

from whirlwind.config import (
    AsyncTrainingConfig,
    DataLoaderConfig,
    ModelConfig,
    ProgressTrackerConfig,
    RewardConfig,
    RolloutConfig,
    RolloutPoolConfig,
    RolloutProxyConfig,
    TrainingConfig,
)
from whirlwind.config_agentic import (
    AgenticTrainingConfig,
    AgentProxyConfig,
    AgentRolloutConfig,
)

# (path substring, candidate classes); checked in order, first match decides.
_KINDS = [
    ("dataloader", (DataLoaderConfig,)),
    (
        "rollout",
        (
            RolloutConfig,
            RolloutPoolConfig,
            RolloutProxyConfig,
            AgentRolloutConfig,
            AgentProxyConfig,
        ),
    ),
    ("reward", (RewardConfig,)),
    ("model", (ModelConfig,)),
    ("tracker", (ProgressTrackerConfig,)),
    ("", (TrainingConfig, AsyncTrainingConfig, AgenticTrainingConfig)),
]


def _load(path: str):
    lowered = path.lower()
    for key, classes in _KINDS:
        if key in lowered:
            first_error: Exception | None = None
            for cls in classes:
                try:
                    return cls.from_yaml(path)
                except ValueError as exc:  # unknown keys: try the next class
                    if "unknown" not in str(exc):
                        raise
                    first_error = first_error or exc
            assert first_error is not None
            raise first_error
    raise ValueError(f"cannot tell the config type of {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 validate_configs.py <config1.yaml> [config2.yaml ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"  Validating: {path}")
        cfg = _load(path)
        print(f"    OK: {type(cfg).__name__}")

    print("\nConfig validation passed.")


if __name__ == "__main__":
    main()
