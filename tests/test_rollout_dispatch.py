"""SGLang DP dispatch order: every copy of a prompt goes to one DP worker.

SGLang routes request i of a batch to worker (c0 + i) % dp_size and all
workers share one sampling seed, so copies of a prompt sent back to back
were sampled in lockstep (identical completions). ``_dp_dispatch_order``
must give all copies of a prompt the same position residue mod dp_size.
"""

import unittest

from whirlwind.rollout.rollout_sglang import _dp_dispatch_order


def _workers(order, keys, dp, c0=0):
    """key -> set of DP workers its copies are routed to under round robin."""
    routed: dict = {}
    for pos, j in enumerate(order):
        routed.setdefault(keys[j], set()).add((c0 + pos) % dp)
    return routed


class TestDpDispatchOrder(unittest.TestCase):
    def test_copies_share_a_worker(self):
        # prompt-major, as the orchestrator sends them: p0 x4, p1 x4, ...
        for n_prompts, copies, dp in [(256, 4, 4), (64, 4, 4), (256, 4, 2), (64, 4, 2), (8, 8, 4)]:
            keys = [f"p{e}" for e in range(n_prompts) for _ in range(copies)]
            order = _dp_dispatch_order(keys, dp)
            self.assertEqual(sorted(order), list(range(len(keys))))
            for c0 in range(dp):  # the round-robin counter persists across calls
                routed = _workers(order, keys, dp, c0)
                self.assertTrue(all(len(w) == 1 for w in routed.values()), (n_prompts, copies, dp))
            # load stays balanced: each worker gets the same number of requests
            per_worker = [sum(1 for pos in range(len(order)) if pos % dp == r) for r in range(dp)]
            self.assertEqual(len(set(per_worker)), 1)

    def test_old_order_split_copies(self):
        # the pre-fix behaviour this guards against: identity order puts copy k on worker k
        keys = [f"p{e}" for e in range(4) for _ in range(4)]
        routed = _workers(list(range(16)), keys, 4)
        self.assertTrue(all(len(w) == 4 for w in routed.values()))

    def test_unbalanced_groups_fall_back_to_a_permutation(self):
        keys = ["a"] * 4 + ["b"] * 4 + ["c"] * 4  # 3 groups cannot fill 4 equal columns
        order = _dp_dispatch_order(keys, 4)
        self.assertEqual(sorted(order), list(range(12)))
        self.assertNotEqual(order, list(range(12)))

    def test_single_worker_keeps_order(self):
        keys = ["a", "a", "b", "b"]
        self.assertEqual(_dp_dispatch_order(keys, 1), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
