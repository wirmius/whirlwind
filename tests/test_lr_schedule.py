"""lr_schedule_steps: a short run can follow a longer run's per-step LR.

A 65-step run on a 65-step cosine spends 91% of its LR budget by step 40,
while a 192-step run is still near peak LR there, so step 64 of the short run
corresponds to roughly step 37 of the long one.
"""

import unittest
from types import SimpleNamespace

from whirlwind.training import training_grpo
from whirlwind.utils import compute_lr


def _orch(total_steps, lr_schedule_steps=0):
    orch = training_grpo.TrainingOrchestrator.__new__(training_grpo.TrainingOrchestrator)
    orch.config = SimpleNamespace(
        total_steps=total_steps,
        lr_schedule_steps=lr_schedule_steps,
        model=SimpleNamespace(learning_rate=1e-6, warmup_ratio=0.05, warmup_steps=0,
                              lr_scheduler_type="cosine"),
    )
    return orch


class TestLrSchedule(unittest.TestCase):
    def test_default_horizon_is_total_steps(self):
        self.assertEqual([_orch(65)._compute_lr(s) for s in range(1, 66)],
                         [_orch(65, 65)._compute_lr(s) for s in range(1, 66)])

    def test_short_run_follows_the_long_schedule(self):
        short, long = _orch(65, lr_schedule_steps=192), _orch(192)
        self.assertEqual([short._compute_lr(s) for s in range(1, 66)],
                         [long._compute_lr(s) for s in range(1, 66)])
        # 5.68e-5 over the 65 steps vs 3.25e-5 on the 65-step cosine
        self.assertGreater(sum(short._compute_lr(s) for s in range(1, 66)),
                           1.7 * sum(_orch(65)._compute_lr(s) for s in range(1, 66)))

    def test_cosine_stays_at_zero_past_the_horizon(self):
        self.assertAlmostEqual(compute_lr(250, 1e-6, 192, warmup_ratio=0.05), 0.0)
        self.assertAlmostEqual(compute_lr(192, 1e-6, 192, warmup_ratio=0.05), 0.0)


if __name__ == "__main__":
    unittest.main()
