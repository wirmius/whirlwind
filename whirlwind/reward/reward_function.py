"""Reward computation node.

Computes raw per-sample rewards and group-normalized advantages for GRPO.

GRPO advantage computation (TRL-style):
    1. Samples are grouped by ``indices`` (prompt ID). Each prompt is
       generated ``num_generations`` times, forming one group.
    2. Within each group, compute mean and std of raw rewards.
    3. Advantages = (reward - group_mean) / (group_std + epsilon) — z-score.
    4. Unscorable completions (all rewards NaN) get advantage = 0.

The reward node returns both raw rewards (for logging) and advantages
(for the GRPO loss computation in model_update).

Accepts RewardRequest with both text (completions) and token (input_ids)
representations.  The reward implementation chooses whichever is convenient.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from whirlwind.config import RewardConfig
from whirlwind.messages import (
    RewardRequest,
    RewardResult,
    StateGetRequest,
    StateResult,
    StateSetRequest,
    _to_dict,
)
from whirlwind.reward.advantages import compute_group_advantages
from whirlwind.whirlcoms import Node

logger = logging.getLogger(__name__)


def _nanmean(vals: list[float]) -> float:
    """NaN-aware mean. Returns 0.0 if all values are NaN."""
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else 0.0


def _nanstd(vals: list[float], mean: float) -> float:
    """NaN-aware sample std (Bessel's correction, ddof=1). Returns 0.0 if fewer than 2 valid values.

    Matches TRL's ``trl.trainer.utils.nanstd``: applies Bessel's correction
    ``count/(count-1)`` so the result is the sample std, not the population
    std. For group_size=4 this makes the std sqrt(4/3)~1.155x larger than the
    population std, which scales advantages down to match TRL exactly.
    """
    valid = [v for v in vals if not math.isnan(v)]
    if len(valid) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
    return variance**0.5


class RewardNode:
    """Reward computation node.

    Usage:
        node = RewardNode(RewardConfig())
        await node.start()
        # ... run until done ...
        await node.stop()
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self._node: Node | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the underlying RPC node is still running (polled by the ``__main__`` keep-alive loop)."""
        return self._node.is_running if self._node is not None else False

    async def start(self) -> None:
        """Register with the whirlcoms orchestrator."""
        self._node = Node(self.config.node_id, auto_process=True)
        self._node.expose("compute_rewards")(self._compute_rewards)
        self._node.expose("ping")(self._ping)
        self._node.expose("get_state")(self._get_state)
        self._node.expose("set_state")(self._set_state)
        await self._node.start()
        logger.info(
            "Reward node '%s' started (type=%s)",
            self.config.node_id,
            self.config.reward_type,
        )

    async def stop(self) -> None:
        if self._node:
            await self._node.stop()
        logger.info("Reward node stopped")

    # ------------------------------------------------------------------
    # State endpoints
    # ------------------------------------------------------------------

    def _full_state(self) -> dict[str, Any]:
        return {
            "config": _to_dict(self.config),
            "state": {},
        }

    @StateGetRequest.decorate_serdes()
    def _get_state(self, req: StateGetRequest) -> StateResult:
        state = self._full_state()
        if req.keys:
            state = {k: v for k, v in state.items() if k in req.keys}
        return StateResult(data=state, ok=True)

    @StateSetRequest.decorate_serdes()
    def _set_state(self, req: StateSetRequest) -> StateResult:
        applied = {k: v for k, v in req.data.items() if not req.keys or k in req.keys}
        return StateResult(data=applied, ok=True)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    @RewardRequest.decorate_serdes()
    def _compute_rewards(self, req: RewardRequest) -> RewardResult:
        """Compute rewards and advantages for a batch of prompt/completion pairs.

        Pipeline:
            1. Compute raw per-sample rewards from the chosen reward function.
            2. Group-normalize rewards into advantages (z-score per prompt group).
            3. Return both raw rewards (for logging) and advantages (for GRPO loss).
        """

        rewards = self._raw_rewards(req)

        # Compute GRPO-style group-normalized advantages
        advantages = self._compute_grpo_advantages(rewards, req.entry_ids)

        # Mask truncated completions: zero out advantage if finish_reason == "length"
        if self.config.mask_truncated_completions:
            for i, finish_reason in enumerate(req.finish_reasons):
                if finish_reason == "length":
                    advantages[i] = 0.0

        result = RewardResult(
            rewards=rewards,
            advantages=advantages,
            dataset_ids=req.dataset_ids,
            entry_ids=req.entry_ids,
            rollout_ids=req.rollout_ids,
            subrollout_ids=req.subrollout_ids,
        )

        if self._node is not None:
            # W&B supergroup: ``train/`` for training, ``val/`` for validation.
            # ``req.log_prefix`` is "" for train and "val." for eval — map to the
            # slash-grouping convention. The tracker preserves these prefixes.
            _grp = "val" if req.log_prefix.startswith("val") else "train"
            # Log reward scalar metrics (fire-and-forget, accumulated by tracker).
            # NaN-aware: rewards/advantages can be NaN for unscorable completions
            # (the advantage computation handles this, but statistics.mean/stdev
            # are NOT NaN-aware -- a single NaN would make every logged stat NaN).
            reward_metrics: dict[str, float] = {
                f"{_grp}/reward_mean": _nanmean(rewards),
                f"{_grp}/reward_std": _nanstd(rewards, _nanmean(rewards))
                if len(rewards) > 1
                else 0.0,
                f"{_grp}/reward_min": min((r for r in rewards if not math.isnan(r)), default=0.0),
                f"{_grp}/reward_max": max((r for r in rewards if not math.isnan(r)), default=0.0),
                f"{_grp}/adv_mean": _nanmean(advantages),
                f"{_grp}/adv_std": _nanstd(advantages, _nanmean(advantages))
                if len(advantages) > 1
                else 0.0,
                f"{_grp}/batch_size": len(rewards),
            }
            # TRL ``frac_reward_zero_std``: fraction of samples belonging to
            # groups whose reward std is ~0 (no learning signal). Matches TRL's
            # ``is_std_zero = isclose(std_rewards, 0); mean(is_std_zero)`` where
            # std_rewards is broadcast per-sample within each group.
            _EPS = 1e-4
            _groups: dict[int, list[int]] = {}
            for _pos, _eid in enumerate(req.entry_ids):
                _groups.setdefault(_eid, []).append(_pos)
            _zero_std_members = 0
            for _members in _groups.values():
                _gr = [rewards[i] for i in _members]
                _gs = _nanstd(_gr, _nanmean(_gr))
                if _gs <= _EPS:
                    _zero_std_members += len(_members)
            reward_metrics[f"{_grp}/frac_reward_zero_std"] = (
                _zero_std_members / len(rewards) if rewards else 0.0
            )
            logger.debug(f"Reward metrics: {reward_metrics}")
            self._node.call_bg_no_return(
                "tracker",
                "log_metrics",
                {
                    "metrics": reward_metrics,
                    "node_id": self.config.node_id,
                },
            )

            # Log prompts + completions + rewards table (fire-and-forget, throttled)
            should_log = req.log_table
            override = self.config.log_every_call_override
            if override > 0:
                self._call_counter = getattr(self, "_call_counter", 0) + 1
                if self._call_counter % override == 0:
                    should_log = True
            if should_log:
                table_name = f"{req.log_prefix}rewards"
                table_rows = []
                for i in range(len(req.completions)):
                    row: list[str] = [
                        str(req.step),
                        req.dataset_ids[i] if i < len(req.dataset_ids) else "",
                        str(req.entry_ids[i]) if i < len(req.entry_ids) else "",
                        str(req.rollout_ids[i]) if i < len(req.rollout_ids) else "",
                        str(req.subrollout_ids[i]) if i < len(req.subrollout_ids) else "",
                        req.prompts[i] if i < len(req.prompts) else "",
                        req.completions[i],
                        str(rewards[i]),
                        str(advantages[i]),
                    ]
                    table_rows.append(row)
                self._node.call_bg_no_return(
                    "tracker",
                    "log_table",
                    {
                        "table_name": table_name,
                        "columns": [
                            "step",
                            "dataset_id",
                            "entry_id",
                            "rollout_id",
                            "subrollout_id",
                            "prompt",
                            "completion",
                            "reward",
                            "advantage",
                        ],
                        "data": table_rows,
                        "node_id": self.config.node_id,
                        "step": req.step,
                    },
                )
                logger.debug(f" >>> Sent the detailed reward table to the tracker. {req.step}")

        return result

    def _raw_rewards(self, req: RewardRequest) -> list[float]:
        """Per-sample raw rewards [batch]. The extension point: a variant node
        overrides this and inherits advantages, masking, and logging."""
        if self.config.reward_type == "length":
            return self._length_based_reward(req)
        if self.config.reward_type == "accuracy":
            return self._accuracy_reward(req)
        raise NotImplementedError(
            f"Reward type '{self.config.reward_type}' not implemented yet"
        )

    def _accuracy_reward(self, req: RewardRequest) -> list[float]:
        """Compute accuracy-based reward.

        Extracts the answer from the reference (ground-truth) using
        ``accuracy_regex`` and searches for a matching answer in the
        generated completion.  Returns 1.0 on match, 0.0 otherwise.
        """
        import re

        pattern = re.compile(self.config.accuracy_regex)
        rewards: list[float] = []
        for completion, response in zip(req.completions, req.responses, strict=True):
            # Extract answer from reference
            ref_match = pattern.search(response)
            if not ref_match:
                rewards.append(0.0)
                continue
            ref_answer = ref_match.group(1).replace(",", "")

            # Search for matching answer in completion
            comp_match = pattern.search(completion)
            if comp_match:
                comp_answer = comp_match.group(1).replace(",", "")
                rewards.append(1.0 if comp_answer == ref_answer else 0.0)
            else:
                rewards.append(0.0)

        # Sequence dump for cross-pipeline diagnostic (env-gated).
        # Schema matches the TRL ``_seq_dump`` so the two .jsonl files can be
        # diffed directly. ``req.log_prefix`` is "" for train, "val." for eval.
        import json as _json
        import os as _os
        _dump_path = _os.environ.get("WHIRLWIND_SEQ_DUMP")
        if _dump_path:
            _mode = "eval" if req.log_prefix.startswith("val") else "train"
            with open(_dump_path, "a") as _f:
                for _i, (_p, _c, _r, _rw) in enumerate(
                    zip(req.prompts, req.completions, req.responses, rewards, strict=False)
                ):
                    _f.write(_json.dumps({
                        "src": "ww",
                        "step": int(req.step),
                        "mode": _mode,
                        "idx": _i,
                        "prompt": str(_p) if _p is not None else "",
                        "completion": str(_c),
                        "reference": str(_r) if _r is not None else "",
                        "reward": float(_rw),
                    }) + "\n")
        return rewards

    def _length_based_reward(self, req: RewardRequest) -> list[float]:
        """Compute raw length-based reward (no normalization).

        Uses token-length from input_ids when available (more accurate than
        string length).  Falls back to string length of completions.

        Raw reward = scale * completion_length.
        """
        lengths: list[int]
        if req.input_ids and req.input_ids[0]:
            # Token-based: use length of the input sequence
            lengths = [len(seq) for seq in req.input_ids]
        else:
            # String-based: use character length of completion
            lengths = [len(c) for c in req.completions]

        return [
            self.config.length_scale * max(length, self.config.length_min) for length in lengths
        ]

    def _compute_grpo_advantages(
        self,
        rewards: list[float],
        entry_ids: list[int],
    ) -> list[float]:
        """Compute group-normalized advantages (TRL GRPO-style).

        Delegates to the shared :func:`whirlwind.reward.advantages.
        compute_group_advantages` (extracted verbatim from this method so the
        async updater path shares the exact same TRL-matched numerics).
        """
        return compute_group_advantages(rewards, entry_ids, self.config.scale_rewards)

    def _ping(self) -> dict[str, Any]:
        return {"ok": True, "reward_type": self.config.reward_type}


def main(node_cls: type[RewardNode] = RewardNode) -> None:
    """Run a reward node until shutdown. Variant modules call ``main(MyNode)``."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Whirlwind Reward Node")
    parser.add_argument(
        "--config",
        default="configs/reward/function.yaml",
        help="Path to reward YAML config",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = RewardConfig.from_yaml(args.config)
    print(f"Starting reward node: type={config.reward_type}, group_size={config.group_size}")

    node = node_cls(config)

    async def _main() -> None:
        await node.start()
        print("Reward node running. Press Ctrl+C to stop.")
        try:
            while node.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await node.stop()

    asyncio.run(_main())


if __name__ == "__main__":
    main()
