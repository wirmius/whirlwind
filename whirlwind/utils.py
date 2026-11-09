"""Pure utility functions."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def compute_lr(
    step: int,
    base_lr: float,
    total_steps: int,
    warmup_ratio: float = 0.0,
    warmup_steps: int = 0,
    scheduler_type: str = "cosine",
) -> float:
    """Compute the learning rate for a given training step.

    Pure function — no internal state, so it works correctly on resume.
    Supports the same scheduler types as ``transformers.get_scheduler``.

    Args:
        step: Current training step (1-indexed).
        base_lr: Base learning rate.
        total_steps: Total number of training steps.
        warmup_ratio: Fraction of total steps for warmup (used if warmup_steps == 0).
        warmup_steps: Absolute number of warmup steps (overrides ratio if > 0).
        scheduler_type: Scheduler name — "cosine", "linear", "constant", "constant_with_warmup".

    Returns:
        The learning rate for the given step.
    """
    total_steps = max(total_steps, 1)
    warmup_steps = (
        max(warmup_steps, 1)
        if warmup_steps > 0
        else max(int(total_steps * warmup_ratio), 1)
    )

    # Warmup phase (all schedulers support this)
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps

    # Decay phase
    if scheduler_type == "cosine":
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    elif scheduler_type == "linear":
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return base_lr * max(1.0 - progress, 0.0)

    elif scheduler_type == "constant":
        return base_lr

    elif scheduler_type == "constant_with_warmup":
        return base_lr  # warmup already handled above

    else:
        logger.warning(
            "Unknown scheduler type '%s', falling back to constant",
            scheduler_type,
        )
        return base_lr
