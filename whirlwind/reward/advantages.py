"""Group-relative advantage computation (TRL-matched), shared by all consumers.

Extracted verbatim from :meth:`RewardNode._compute_grpo_advantages`
(``reward_function.py``) so that every path that turns raw scalar rewards into
GRPO advantages — the reward node (sync loop) and the async updater
(buffered, externally-rewarded rollouts) — uses the *same* numerics:

- groups keyed by ``entry_id`` (= one prompt's ``group_size`` completions),
- NaN-aware mean, Bessel-corrected sample std (ddof=1) — matches TRL's
  ``trl.trainer.utils.nanstd`` exactly (TIPS_AND_TRICKS.md §4.15: keep it that way),
- NaN rewards → advantage 0; ``scale_rewards`` = ``group`` | ``batch`` | ``none``.

Behavior-preserving extraction: ``RewardNode._compute_grpo_advantages`` now
delegates here; the reward tests pin the semantics.
"""

from __future__ import annotations

import math

EPSILON = 1e-4


def nanmean(vals: list[float]) -> float:
    """NaN-aware mean. Returns 0.0 if all values are NaN."""
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else 0.0


def nanstd(vals: list[float], mean: float) -> float:
    """NaN-aware sample std (Bessel's correction, ddof=1).

    Returns 0.0 if fewer than 2 valid values. Matches TRL's
    ``trl.trainer.utils.nanstd``: the ``count/(count-1)`` correction makes this
    the sample std, not the population std, so advantages scale exactly like
    TRL's (for group_size=4 the std is sqrt(4/3)~1.155x the population std).
    """
    valid = [v for v in vals if not math.isnan(v)]
    if len(valid) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
    return variance**0.5


def compute_group_advantages(
    rewards: list[float],
    entry_ids: list[int],
    scale_rewards: str = "group",
    epsilon: float = EPSILON,
) -> list[float]:
    """Compute group-normalized advantages (TRL GRPO-style).

    Groups samples by their entry_id (dataset row index / prompt group),
    computes the per-group mean and standard deviation, and returns z-score
    normalized advantages.

    Unlike a flat-stride approach, this correctly handles arbitrary group
    sizes and samples that may not be perfectly contiguous in the batch.

    Args:
        rewards:      raw per-sample rewards [batch]
        entry_ids:    per-sample group ids [batch] (one group per prompt)
        scale_rewards: ``"group"`` (per-group z-score, TRL default),
                      ``"batch"`` (global z-score), or ``"none"`` (subtract
                      group mean only)

    Returns:
        advantages: z-score normalized per-sample advantages [batch]
    """
    # Build group assignment: entry_id -> list of sample positions
    groups: dict[int, list[int]] = {}
    for pos, eid in enumerate(entry_ids):
        groups.setdefault(eid, []).append(pos)

    advantages: list[float] = [0.0] * len(rewards)

    if scale_rewards == "group":
        # Per-group mean and std (TRL default). NaN rewards → advantage = 0.
        for members in groups.values():
            group_rewards = [rewards[i] for i in members]
            group_mean = nanmean(group_rewards)
            group_std = nanstd(group_rewards, group_mean)
            std = group_std if group_std > epsilon else epsilon
            for i in members:
                if math.isnan(rewards[i]):
                    advantages[i] = 0.0
                else:
                    advantages[i] = (rewards[i] - group_mean) / std

    elif scale_rewards == "batch":
        # Global mean and std across the entire batch. NaN → advantage = 0.
        batch_mean = nanmean(rewards)
        batch_std = nanstd(rewards, batch_mean)
        std = batch_std if batch_std > epsilon else epsilon
        for i in range(len(rewards)):
            if math.isnan(rewards[i]):
                advantages[i] = 0.0
            else:
                advantages[i] = (rewards[i] - batch_mean) / std

    else:  # "none" — just subtract group mean, no std normalization
        for members in groups.values():
            group_rewards = [rewards[i] for i in members]
            group_mean = nanmean(group_rewards)
            for i in members:
                if math.isnan(rewards[i]):
                    advantages[i] = 0.0
                else:
                    advantages[i] = rewards[i] - group_mean

    return advantages
