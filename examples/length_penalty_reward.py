"""A custom reward in ~10 lines: accuracy minus a small length penalty.

    ./bringup_and_run.sh --reward-module examples.length_penalty_reward ...

The subclass only changes the per-sample reward; group-normalized
advantages, truncation masking and W&B logging come from RewardNode.
"""

from whirlwind.reward.reward_function import RewardNode, main


class LengthPenaltyReward(RewardNode):
    def _raw_rewards(self, req):
        correct = super()._raw_rewards(req)  # 1.0 / 0.0 under reward_type: accuracy
        return [r - 1e-4 * len(c) for r, c in zip(correct, req.completions)]


if __name__ == "__main__":
    main(LengthPenaltyReward)
