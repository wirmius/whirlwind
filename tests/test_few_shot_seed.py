"""few_shot_seed decouples the few-shot examples from the dataset shuffle seed.

A scan that varies the train dataset's `seed` to get a second draw used to
change the few-shot examples too, because the sampler read the same field, so
the "seed replicate" also ran on different prompts.
"""

import unittest
from types import SimpleNamespace

from whirlwind.config import DataLoaderConfig
from whirlwind.data.dataloader_node import DataLoaderNode

ROWS = [{"question": f"q{i}?", "answer": f"a{i}\n#### {i}"} for i in range(50)]


def _prompt(dataset_seed, few_shot_seed=None):
    cfg = DataLoaderConfig(
        datasets={"train": {"seed": dataset_seed}},
        system_prompt_template="Solve.\n\n{few_shot_examples}",
        few_shot_count=4,
        few_shot_seed=few_shot_seed,
    )
    node = DataLoaderNode.__new__(DataLoaderNode)  # no event loop needed
    node.config = cfg
    handle = SimpleNamespace(
        config={"seed": dataset_seed, "prompt_key": "question", "response_key": "answer"},
        _dataset=ROWS,
    )
    node._datasets = {"train": handle}
    return node._build_system_prompt()


class TestFewShotSeed(unittest.TestCase):
    def test_legacy_follows_the_dataset_seed(self):
        self.assertNotEqual(_prompt(42), _prompt(43))

    def test_fixed_few_shot_seed_keeps_the_prompt(self):
        self.assertEqual(_prompt(42, few_shot_seed=42), _prompt(43, few_shot_seed=42))
        self.assertEqual(_prompt(43, few_shot_seed=42), _prompt(42))  # same examples as legacy seed 42


if __name__ == "__main__":
    unittest.main()
