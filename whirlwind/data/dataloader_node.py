"""Dataloader node — serves batches from HF Datasets via RPC.

Manages multiple named datasets (train, val, etc.) with independent
shuffling, batching, and state tracking.

Usage:
    node = DataLoaderNode(DataLoaderConfig.from_yaml("dataloader.yaml"))
    await node.start()
    # ... run until done ...
    await node.stop()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from whirlwind.config import DataLoaderConfig
from whirlwind.messages import (
    BatchRequest,
    BatchResult,
    StateGetRequest,
    StateResult,
    StateSetRequest,
    _to_dict,
)
from whirlwind.whirlcoms import Node

logger = logging.getLogger(__name__)


class _DatasetHandle:
    """Tracks one dataset's state and iteration position."""

    def __init__(
        self,
        name: str,
        dataset_config: dict[str, Any],
    ) -> None:
        self.name = name
        self.config = dataset_config
        self._dataset: Any | None = None
        self._num_samples: int = 0

        # Current iteration state (for checkpointing)
        self._epoch: int = 0
        self._index: int = 0
        # Cached per-epoch shuffle (re-materialized only when the epoch rolls
        # over), so next_batch is O(batch) not O(dataset) per call.
        self._shuffled: Any | None = None

    async def load(self) -> None:
        """Load the dataset into memory."""
        from datasets import load_dataset  # type: ignore[import-not-found]

        loop = asyncio.get_running_loop()

        def _load():
            name = self.config.get("name", "")
            split = self.config.get("split", "train")
            # datasets>=5 dropped single-file local-path resolution for
            # load_dataset(path) — use the packaged builder for local
            # .parquet files instead (hub datasets take the original path).
            if name.endswith(".parquet"):
                return load_dataset("parquet", data_files=name, split=split)
            return load_dataset(
                name,
                name=self.config.get("dataset_config") or "default",
                split=split,
            )

        dataset = await loop.run_in_executor(None, _load)

        # Cap samples if requested
        max_samples = self.config.get("max_samples", 0)
        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Store original row indices
        dataset = dataset.add_column("_orig_idx", list(range(len(dataset))))

        self._dataset = dataset
        self._num_samples = len(dataset)

        logger.info(
            "Dataset '%s': %d samples (split=%s)",
            self.name,
            self._num_samples,
            self.config.get("split", "train"),
        )

    def next_batch(self, batch_size: int, step: int = 0) -> BatchResult:
        """Return the next batch, advancing the iterator.

        Cycles through epochs indefinitely. Shuffles at epoch boundaries
        using seed + epoch for deterministic reproducibility.
        """
        if self._dataset is None:
            raise RuntimeError(f"Dataset '{self.name}' not loaded")

        cfg = self.config
        key = cfg.get("prompt_key", "prompt")
        response_key = cfg.get("response_key", "response")
        seed = cfg.get("seed", 42)
        shuffle = cfg.get("shuffle", True)

        # Move to next epoch if needed (re-shuffle when the epoch rolls over;
        # cache the result so we don't re-permute the whole dataset per batch).
        if self._index >= self._num_samples:
            self._index = 0
            self._epoch += 1
            self._shuffled = None
        if self._shuffled is None:
            # When shuffle is disabled, serve rows in raw order (no permutation).
            # This makes the dataloader deterministic and index-aligned with
            # external pipelines (e.g. the TRL reference run) for A/B comparison.
            self._shuffled = (
                self._dataset.shuffle(seed=seed + self._epoch) if shuffle else self._dataset
            )
        shuffled = self._shuffled

        batch_sz = max(batch_size, 1)
        end = min(self._index + batch_sz, self._num_samples)

        # Extract columns
        prompts = [str(p) for p in shuffled[key][self._index : end]]  # type: ignore[index]
        responses: list[str] = []
        if response_key in shuffled.column_names:
            responses = [str(shuffled[response_key][i]) for i in range(self._index, end)]
        if not responses:
            responses = [""] * len(prompts)

        dataset_ids = [self.name] * len(prompts)
        entry_ids = shuffled["_orig_idx"][self._index : end]  # type: ignore[index]

        # Advance index
        self._index = end

        return BatchResult(
            ok=True,
            prompts=prompts,
            responses=responses,
            step=step,
            dataset_ids=dataset_ids,
            entry_ids=entry_ids.tolist() if hasattr(entry_ids, "tolist") else entry_ids,
            rollout_ids=[0] * len(prompts),
            subrollout_ids=[0] * len(prompts),
        )

    def get_state(self) -> dict[str, Any]:
        """Return this dataset's iteration state."""
        return {
            "epoch": self._epoch,
            "index": self._index,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore this dataset's iteration state."""
        if isinstance(state, dict):
            self._epoch = int(state.get("epoch", 0))
            self._index = int(state.get("index", 0))


class DataLoaderNode:
    """Dataloader node serving batches from multiple named HF datasets.

    Each dataset is tracked independently with its own epoch, index, and
    shuffle seed.  Supports checkpointing and resumption.
    """

    def __init__(self, config: DataLoaderConfig) -> None:
        self.config = config
        self._node: Node | None = None
        self._datasets: dict[str, _DatasetHandle] = {}
        self._system_prompt: str = ""
        self._prefix_messages: list[dict] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the underlying RPC node is still running (polled by the ``__main__`` keep-alive loop)."""
        return self._node.is_running if self._node is not None else False

    async def start(self) -> None:
        """Load all datasets, then register with the whirlcoms orchestrator.

        Datasets are loaded *before* ``self._node.start()`` so the event loop
        is not blocked on HF Hub downloads while heartbeats need processing.
        """
        self._node = Node(self.config.node_id, auto_process=True)
        self._node.expose("next_batch")(self._handle_next_batch)
        self._node.expose("get_state")(self._get_state)
        self._node.expose("set_state")(self._set_state)
        self._node.expose("ping")(self._ping)

        # Load all datasets BEFORE starting the node.
        # If we start the node first, the blocking HF Hub download will starve
        # the event loop and cause the orchestrator to deregister us for missed
        # heartbeats.
        for name, dcfg in self.config.datasets.items():
            handle = _DatasetHandle(name, dcfg)
            await handle.load()
            self._datasets[name] = handle

        # Build system prompt from template + few-shot examples
        self._system_prompt = self._build_system_prompt()

        # Now register — event loop is free to handle heartbeats.
        await self._node.start()

        logger.info(
            "Dataloader node '%s' started with datasets: %s",
            self.config.node_id,
            sorted(self._datasets.keys()),
        )

    async def stop(self) -> None:
        if self._node:
            await self._node.stop()
        logger.info("Dataloader node stopped")

    def _build_system_prompt(self) -> str:
        """Build system prompt from template + few-shot examples.

        Uses the configured system_prompt_template and samples
        few_shot_count examples from the default (train) dataset.
        """
        import random

        self._prefix_messages: list[dict] | None = None
        template = self.config.system_prompt_template
        if not template:
            return ""

        count = self.config.few_shot_count
        eos_mode = bool(getattr(self.config, "few_shot_eos", False))
        if count <= 0:
            # No few-shot examples: if the template has no placeholder either,
            # return it VERBATIM — str.format would otherwise choke on literal
            # braces in hardcoded example text (LaTeX like \binom{10}{5}).
            if "{few_shot_examples}" not in template:
                return template
            return template.format(few_shot_examples="")

        # Sample from the default dataset (train)
        handle = self._datasets.get(self.config.default_dataset)
        if handle is None or handle._dataset is None or len(handle._dataset) == 0:
            return template.format(few_shot_examples="")

        cfg = handle.config
        seed = self.config.few_shot_seed
        if seed is None:
            seed = cfg.get("seed", 42)
        prompt_key = cfg.get("prompt_key", "prompt")
        response_key = cfg.get("response_key", "response")

        rng = random.Random(seed)  # match TRL few-shot seed (Random(42)); was seed+99=141
        indices = rng.sample(range(len(handle._dataset)), min(count, len(handle._dataset)))

        dataset = handle._dataset
        if eos_mode:
            # few_shot_eos: the examples travel as raw turn
            # contents (transport only). The ROLLOUT folds them back into
            # the 'Q:/A:' blob style inside the system message and
            # appends the DECLARED EOS after each example — the rendering
            # stays the old simplistic style; only the EOS is new.
            # System message = instruction only (examples re-attached at
            # the rollout, tokenizer-side).
            self._prefix_messages = []
            for idx in indices:
                self._prefix_messages.append(
                    {"role": "user", "content": str(dataset[idx][prompt_key])}
                )
                self._prefix_messages.append(
                    {"role": "assistant", "content": str(dataset[idx].get(response_key, ""))}
                )
            # instruction-only system prompt (no examples blob)
            if "{few_shot_examples}" in template:
                logger.info(
                    "Built system prompt (few_shot_eos): instruction only + "
                    "%d example turns (folded rollout-side with the declared EOS)",
                    len(self._prefix_messages),
                )
                return template.format(few_shot_examples="").rstrip()
            return template
        few_shot = "\n\n".join(
            f"Q: {dataset[idx][prompt_key]}\nA: {dataset[idx].get(response_key, '')}"
            for idx in indices
        )

        logger.info("Built system prompt with %d few-shot examples", count)
        return template.format(few_shot_examples=few_shot)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    @BatchRequest.decorate_serdes()
    def _handle_next_batch(self, req: BatchRequest) -> BatchResult:
        """Serve the next batch for the requested dataset."""
        name = req.dataset_name or self.config.default_dataset
        handle = self._datasets.get(name)

        if handle is None:
            return BatchResult(
                ok=False,
                error=f"Unknown dataset: '{name}'. Available: {sorted(self._datasets.keys())}",
            )

        batch_size = req.batch_size if req.batch_size > 0 else (handle.config.get("batch_size", 32))
        result = handle.next_batch(batch_size, req.step)
        result.system_prompt = self._system_prompt
        if self._prefix_messages:
            # few_shot_multiturn: example turns for the rollout's chat template
            result.prefix_messages = self._prefix_messages
        return result

    @StateGetRequest.decorate_serdes()
    def _get_state(self, req: StateGetRequest) -> StateResult:
        """Return dataloader state (config + per-dataset iteration state)."""
        state = self._full_state()
        if req.keys:
            state = {k: v for k, v in state.items() if k in req.keys}
        return StateResult(data=state, ok=True)

    @StateSetRequest.decorate_serdes()
    def _set_state(self, req: StateSetRequest) -> StateResult:
        """Restore dataloader state."""
        applied = {k: v for k, v in req.data.items() if not req.keys or k in req.keys}

        if "state" in applied and isinstance(applied["state"], dict):
            ds_states = applied["state"].get("datasets", {})
            for name, ds_state in ds_states.items():
                handle = self._datasets.get(name)
                if handle is not None:
                    handle.set_state(ds_state)

        return StateResult(data=applied, ok=True)

    def _full_state(self) -> dict[str, Any]:
        """Construct the full state dict."""
        datasets_state: dict[str, Any] = {}
        for name, handle in self._datasets.items():
            datasets_state[name] = handle.get_state()

        return {
            "config": _to_dict(self.config),
            "state": {
                "datasets": datasets_state,
            },
        }

    def _ping(self) -> dict[str, Any]:
        return {
            "ok": True,
            "datasets": sorted(self._datasets.keys()),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whirlwind Dataloader Node")
    parser.add_argument(
        "--config",
        default="configs/dataloader/datasets.yaml",
        help="Path to dataloader YAML config",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = DataLoaderConfig.from_yaml(args.config)
    print(f"Starting dataloader node: datasets={sorted(config.datasets.keys())}")

    node = DataLoaderNode(config)

    async def _main() -> None:
        await node.start()
        print("Dataloader node running. Press Ctrl+C to stop.")
        try:
            while node.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await node.stop()

    asyncio.run(_main())
