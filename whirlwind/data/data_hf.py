"""Dataset loader — iterator over HF Datasets.

Yields batches of raw prompt strings from the configured ``prompt_key``
column.  Supports shuffling, epoch cycling, and optional sample capping.

Usage:
    loader = DatasetLoader(config)
    await loader.load()
    for batch in loader.iter_batches():
        # batch is PromptBatch with indices, prompts, step
        ...
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from whirlwind.config import TrainingConfig
from whirlwind.messages import PromptBatch

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Batch iterator over a HuggingFace dataset.

    The dataset is loaded fully into memory (via ``load()``), then
    ``iter_batches()`` yields shuffled PromptBatch objects, cycling
    through epochs indefinitely.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._dataset: Any | None = None
        self._num_samples: int = 0
        self._step: int = 0
        self._few_shot_examples: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load (or fetch) the HF dataset into memory."""
        import asyncio

        from datasets import load_dataset  # type: ignore[import-not-found]

        # Run HF download in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        dataset = await loop.run_in_executor(
            None,
            lambda: load_dataset(
                self.config.dataset_name,
                name=self.config.dataset_config or None,
                split=self.config.dataset_split,
            ),
        )

        # Cap samples if requested
        if self.config.dataset_max_samples > 0:
            dataset = dataset.select(
                range(min(self.config.dataset_max_samples, len(dataset)))
            )

        # Store original row indices so they survive shuffling.
        # This makes entry_id a stable identifier of the source row.
        dataset = dataset.add_column("_orig_idx", list(range(len(dataset))))

        self._dataset = dataset
        self._num_samples = len(dataset)

        # Sample few-shot examples at load time
        self._sample_few_shot()

        logger.info(
            "Loaded dataset '%s' split='%s': %d samples (key='%s')",
            self.config.dataset_name,
            self.config.dataset_split,
            self._num_samples,
            self.config.prompt_key,
        )

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def few_shot_examples(self) -> list[tuple[str, str]]:
        """List of (question, answer) tuples sampled at load time."""
        return self._few_shot_examples

    def _sample_few_shot(self) -> None:
        """Sample few-shot examples from the dataset.

        Extracts (question, answer) pairs using the configured keys.
        Answers are expected to contain the full reasoning text
        (e.g., GSM8K format ending with #### {answer}).
        """
        import random

        if self._dataset is None:
            return
        count = self.config.few_shot_count
        if count <= 0 or len(self._dataset) == 0:
            return

        rng = random.Random(self.config.seed + 99)
        indices = rng.sample(range(len(self._dataset)), min(count, len(self._dataset)))

        prompt_key = self.config.prompt_key
        response_key = self.config.response_key
        examples: list[tuple[str, str]] = []
        for idx in indices:
            q = str(self._dataset[idx][prompt_key])  # type: ignore[index]
            a = (
                str(self._dataset[idx][response_key])
                if response_key in self._dataset.column_names
                else ""
            )  # type: ignore[index]
            examples.append((q, a))
        self._few_shot_examples = examples
        logger.info("Sampled %d few-shot examples", len(examples))

    def iter_batches(self) -> Iterator[PromptBatch]:
        """Sync generator yielding PromptBatch objects, cycling epochs.

        Each batch contains ``dataset_batch_size`` unique prompts.
        The caller is responsible for ``rollout_per_prompt`` expansion
        (done by ``TrainingOrchestrator._expand_prompts``).
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded — call load() first")

        batch_size = getattr(self.config, "dataset_batch_size", 32) or 32
        key = self.config.prompt_key

        epoch = 0
        while True:
            # Shuffle at epoch boundaries
            shuffled = self._dataset.shuffle(seed=self.config.seed + epoch)
            epoch += 1

            for start in range(0, self._num_samples, batch_size):
                end = min(start + batch_size, self._num_samples)

                # Access the column directly (avoids Pyright __getitem__ typing issues)
                prompts = [str(p) for p in shuffled[key][start:end]]  # type: ignore[index]
                responses = [
                    str(shuffled[self.config.response_key][i])  # type: ignore[index]
                    for i in range(start, end)
                    if self.config.response_key in shuffled.column_names
                ]
                if not responses:
                    responses = [""] * len(prompts)

                dataset_ids = [str(self.config.dataset_name)] * len(prompts)
                entry_ids = shuffled["_orig_idx"][start:end]
                rollout_ids = [0] * len(prompts)  # single rollout per prompt
                subrollout_ids = [0] * len(prompts)

                self._step += 1
                yield PromptBatch(
                    prompts=prompts,
                    responses=responses,
                    dataset_ids=dataset_ids,
                    entry_ids=entry_ids,
                    rollout_ids=rollout_ids,
                    subrollout_ids=subrollout_ids,
                    step=self._step,
                )
