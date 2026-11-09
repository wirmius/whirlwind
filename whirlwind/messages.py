"""Shared message/data classes for inter-component communication.

Every message is a plain dataclass -- serializable via msgpack (through the
``to_dict`` / ``from_dict`` helpers) and carry tensors by reference (SHM)
when routed through the whirlcoms transport layer.

Messages that carry per-sample lists (prompts, completions, rewards, etc.)
inherit from ``BatchedMessage``, which declares ``split_into(n)`` and
``join(shards)`` as abstract methods implemented by each subclass.

Both ``Message`` and ``BatchedMessage`` provide ``decorate_serdes()`` classmethods
that produce decorators for RPC endpoints, converting dict ↔ message automatically.

All messages carry tracing fields (dataset_id, entry_id, rollout_id,
subrollout_id) so any message can be traced back to its source data.
Batched messages carry plural versions of these as per-sample lists.

Dimensionality summary:

    PromptBatch         [batch]  -- string prompts for SGLang
    RolloutResult       [batch]  -- strings (debugging) + [batch, variable] token lists
    RewardRequest       [batch]  -- strings (optional) + [batch, variable] input_ids
    RewardResult        [batch]  -- scalar rewards per sample
    TrainStepRequest    [batch]  -- [batch, variable] input_ids/label_ids + [batch] rewards
    TrainStepResult     [1]      -- scalar metrics (loss, grad_norm, lr) — one per rank
    WeightSyncRequest   ()       -- single checkpoint path
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Self, Tuple

import torch


def _compute_shard_indices(n: int, size: int) -> List[List[int]]:
    """Compute per-shard index lists for splitting a batch of *size* across *n* ranks.

    If ``size < n``, samples are duplicated so every shard has at least one item.
    If ``size == 0``, every shard gets an empty list (empty batches arise at
    end-of-epoch remainders, empty val splits, or filtered datasets).
    """
    n = max(n, 1)
    if size == 0:
        return [[] for _ in range(n)]
    padded_size = max(size, n)
    indices = [i % size for i in range(padded_size)]

    remainders = padded_size % n
    base = padded_size // n
    shard_sizes = [base + (1 if i < remainders else 0) for i in range(n)]

    shard_idx_lists: List[List[int]] = []
    offset = 0
    for sz in shard_sizes:
        shard_idx_lists.append(indices[offset : offset + sz])
        offset += sz
    return shard_idx_lists


def _shard_list(lst: Any, idxs: List[int]) -> List[Any]:
    """Shard a list using the given indices, returning empty list if lst is empty.

    Indices are expected to be in ``[0, len(lst))`` (``_compute_shard_indices``
    guarantees this for a correctly-sized batch). Uses plain indexing so a
    per-sample list whose length disagrees with ``batch_size`` raises a loud
    ``IndexError`` instead of being silently remapped by a modulo wrap.
    """
    if not lst:
        return []
    return [lst[i] for i in idxs]


def max_subrollouts_per_group(msg: "BatchedMessage") -> Dict[Tuple[int, int], int]:
    """Compute the maximum subrollout_id + 1 for each (entry_id, rollout_id) group.

    Returns a dict mapping (entry_id, rollout_id) → subrollout count.
    Used by the orchestrator to expand ground-truth responses to match
    variable sub-rollout counts in a RolloutResult.

    Example::

        >>> counts = max_subrollouts_per_group(rollout_result)
        # { (0, 0): 3, (0, 1): 2, (1, 0): 3, (1, 1): 3 }

    Then ``raw_responses`` (one per prompt) can be expanded to
    ``rollout_result.batch_size`` by repeating each response
    ``sum(counts[(eid, rid)] for rid in all_rollouts_of_eid)`` times.
    """
    groups: Dict[Tuple[int, int], int] = {}
    for i in range(msg.batch_size):
        key = (msg.entry_ids[i], msg.rollout_ids[i])
        current = groups.get(key, 0)
        groups[key] = max(current, msg.subrollout_ids[i] + 1)
    return groups


# ---------------------------------------------------------------------------
# Base message
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Message:
    """Base class for all inter-component messages.

    Provides ``to_dict`` / ``from_dict`` so subclasses don't need to
    import the module-level helpers.

    Tracing fields:
        dataset_id:     unique identifier for the source dataset (empty if unspecified)
        entry_id:       index within the dataset (-1 if unspecified)
        rollout_id:     index of the rollout among many
        subrollout_id:  index of the sub-rollout (e.g. generation stage)
    """

    dataset_id: str = ""
    entry_id: int = -1
    rollout_id: int = 0
    subrollout_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert this message to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Reconstruct this message from a dict, ignoring extra keys."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})  # type: ignore[arg-type]

    @classmethod
    def decorate_serdes(cls) -> Callable[[Callable], Callable]:
        """Decorator that converts dict ↔ message automatically for RPC endpoints.

        Extracts the input dict from ``data`` in kwargs or the second
        positional arg (first after ``self``). Converts it to this message
        type via ``from_dict``. Converts the return value back to dict
        via ``to_dict`` if it is a Message subclass.

        Usage::

            @MyMessage.decorate_serdes()
            def my_endpoint(self, msg: MyMessage) -> OtherMessage:
                ...
        """

        import asyncio

        def decorator(func: Callable) -> Callable:
            # ---- Async path ----
            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                    # Extract the input dict
                    input_dict: Any = kwargs.pop("data", None)
                    if input_dict is None and len(args) >= 1:
                        input_dict = args[0]

                    # Convert dict → message
                    if isinstance(input_dict, cls):
                        msg = input_dict
                    elif isinstance(input_dict, dict):
                        msg = cls.from_dict(input_dict)  # type: ignore[arg-type]
                    else:
                        raise ValueError(
                            f"Got wrong type: {type(input_dict)}; expected dict or {cls}"
                        )

                    # Call the decorated function
                    result = await func(self, msg, *args[1:], **kwargs)

                    # Convert message → dict for RPC reply
                    if isinstance(result, Message):
                        return result.to_dict()
                    return result

                return wrapper

            # ---- Sync path ----
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                # Extract the input dict
                input_dict: Any = kwargs.pop("data", None)
                if input_dict is None and len(args) >= 1:
                    input_dict = args[0]

                # Convert dict → message
                if isinstance(input_dict, cls):
                    msg = input_dict
                elif isinstance(input_dict, dict):
                    msg = cls.from_dict(input_dict)  # type: ignore[arg-type]
                else:
                    raise ValueError(f"Got wrong type: {type(input_dict)}; expected dict or {cls}")

                # Call the decorated function
                result = func(self, msg, *args[1:], **kwargs)

                # Convert message → dict for RPC reply
                if isinstance(result, Message):
                    return result.to_dict()
                return result

            return wrapper

        return decorator


# ---------------------------------------------------------------------------
# ArgsKwargs — generic message for trivial payloads
# ---------------------------------------------------------------------------


@dataclass
class ArgsKwargs(Message):
    """Generic message for handlers with trivial positional/keyword payloads.

    Stores positional arguments in ``_msg_args`` and keyword arguments in
    ``_msg_kwargs``.  Supports attribute-style access (``msg.path`` →
    ``msg._msg_kwargs["path"]``) and positional indexing (``msg[0]`` →
    ``msg._msg_args[0]``).

    Subclasses can restrict payload types by setting class-level variables::

        class HotCheckpointRequest(ArgsKwargs):
            _must_have_args_types = [str, int]   # (path, step)
            _must_have_kwargs_types = None        # no keyword validation

    If ``_must_have_args_types`` or ``_must_have_kwargs_types`` is not None,
    ``__init__`` validates that the corresponding positions/keys match.
    Raises ``TypeError`` on mismatch.

    Serdes (``to_dict`` / ``from_dict``) is overridden to correctly
    round-trip ``_msg_args`` and ``_msg_kwargs`` through JSON.
    """

    _msg_args: tuple[Any, ...] = field(default_factory=tuple)
    _msg_kwargs: dict[str, Any] = field(default_factory=dict)

    # Subclass overrides — None means "skip validation"
    _must_have_args_types: ClassVar[list[type] | None] = None
    _must_have_kwargs_types: ClassVar[dict[str, type] | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._msg_args = args
        self._msg_kwargs = kwargs
        self._check_types()

    def _check_types(self) -> None:
        """Validate arg/kwarg types if the subclass declared constraints."""
        at = self._must_have_args_types
        if at is not None:
            if len(self._msg_args) != len(at):
                raise TypeError(
                    f"{type(self).__name__} expected {len(at)} positional "
                    f"arg(s), got {len(self._msg_args)}"
                )
            for i, (got, expected) in enumerate(zip(self._msg_args, at)):
                if not isinstance(got, expected):
                    raise TypeError(
                        f"{type(self).__name__} positional arg {i}: "
                        f"expected {expected.__name__}, got {type(got).__name__}"
                    )
        kt = self._must_have_kwargs_types
        if kt is not None:
            for key, expected in kt.items():
                if key not in self._msg_kwargs:
                    raise TypeError(f"{type(self).__name__} missing required keyword arg {key!r}")
                got = self._msg_kwargs[key]
                if not isinstance(got, expected):
                    raise TypeError(
                        f"{type(self).__name__} keyword arg {key!r}: "
                        f"expected {expected.__name__}, got {type(got).__name__}"
                    )

    def __getattr__(self, name: str) -> Any:
        """Allow ``msg.path`` → ``msg._msg_kwargs["path"]``."""
        if name.startswith("_") or name in (
            "_msg_args",
            "_msg_kwargs",
            "dataset_id",
            "entry_id",
            "rollout_id",
            "subrollout_id",
        ):
            raise AttributeError(name)
        try:
            return self._msg_kwargs[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, idx: int) -> Any:
        """Allow ``msg[0]`` → ``msg._msg_args[0]``."""
        return self._msg_args[idx]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, preserving args/kwargs plus tracing fields."""
        d: dict[str, Any] = {
            "_msg_args": list(self._msg_args),
            "_msg_kwargs": dict(self._msg_kwargs),
        }
        # Include tracing fields from Message
        d["dataset_id"] = self.dataset_id
        d["entry_id"] = self.entry_id
        d["rollout_id"] = self.rollout_id
        d["subrollout_id"] = self.subrollout_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct from a dict, restoring args/kwargs + tracing fields."""
        args = tuple(data.get("_msg_args", []))
        kwargs = dict(data.get("_msg_kwargs", {}))
        # Pop tracing fields so they don't end up in kwargs
        tracing_keys = {"dataset_id", "entry_id", "rollout_id", "subrollout_id"}
        # Match each key's declared default in Message: dataset_id is a str
        # (default ""), entry_id is int (default -1), the rest default 0.
        # An absent key must not become a bogus "0"/0 value.
        tracing = {
            k: data.get(k, "" if k == "dataset_id" else (-1 if k == "entry_id" else 0))
            for k in tracing_keys
        }
        msg = cls(*args, **kwargs)
        msg.dataset_id = str(tracing["dataset_id"])
        msg.entry_id = int(tracing["entry_id"])
        msg.rollout_id = int(tracing["rollout_id"])
        msg.subrollout_id = int(tracing["subrollout_id"])
        return msg


# ---------------------------------------------------------------------------
# Batched message base
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BatchedMessage(Message, ABC):
    """Abstract base for messages that carry per-sample batch lists.

    Subclasses must implement ``batch_size``, ``split_into``, and ``join``.

    Tracing fields (plural, per-sample):
        dataset_ids:      per-sample dataset identifiers
        entry_ids:        per-sample dataset indices
        rollout_ids:      per-sample rollout indices
        subrollout_ids:   per-sample sub-rollout indices (generation stages)
    """

    dataset_ids: List[str] = field(default_factory=list)
    entry_ids: List[int] = field(default_factory=list)
    rollout_ids: List[int] = field(default_factory=list)
    subrollout_ids: List[int] = field(default_factory=list)

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Return the number of samples in the batch."""
        ...

    @abstractmethod
    def split_into(self, n: int) -> List[Self]:  # type: ignore[override]
        """Split self into n shards for distributed execution.

        If ``batch_size < n``, samples are duplicated so every shard has
        at least one item (ensures all ranks participate in FSDP).
        """
        ...

    @classmethod
    @abstractmethod
    def join(cls, shards: List[Self]) -> Self:  # type: ignore[override]
        """Join shards back into a single message by concatenating batch fields.

        Note: ``join`` is a left-inverse of ``split_into`` only when
        ``batch_size >= n``. When ``batch_size < n``, ``split_into`` duplicates
        samples so every rank gets >=1 item (for FSDP participation), and
        ``join`` concatenates those shards back -- so the joined batch is larger
        than the original and contains duplicates. Callers that need the
        original size should slice the joined result back to it.
        """
        ...

    @classmethod
    def decorate_serdes(cls) -> Callable[[Callable], Callable]:
        """Decorator that converts dict ↔ BatchedMessage for RPC endpoints.

        Uses ``cls.from_dict`` (not ``Message.from_dict``) so the correct
        subclass is instantiated even when called from a subclass like
        ``TrainStepRequest``.
        """

        import asyncio

        def decorator(func: Callable) -> Callable:
            # ---- Async path ----
            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def wrapper(*args: Any, **kwargs: Any) -> Any:
                    # Extract the input dict
                    input_dict: Any = kwargs.pop("data", None)
                    sel_idx = -1
                    if input_dict is None and len(args) >= 1:
                        for idx, a in enumerate(args):
                            if isinstance(a, dict) or isinstance(a, cls):
                                input_dict = a
                                sel_idx = idx
                                break
                    if input_dict is None:
                        raise ValueError("Could not find any suitable arguments in the decoratr!")

                    # Convert dict → message (uses correct subclass)
                    if not isinstance(input_dict, cls):
                        msg = cls.from_dict(input_dict)  # type: ignore[arg-type]
                    else:
                        msg = input_dict

                    # Call the decorated function.
                    # sel_idx == -1 means the message arrived via the ``data=``
                    # kwarg (popped above); args[0] is the bound ``self``.
                    if sel_idx == -1:
                        result = await func(args[0], msg, *args[1:], **kwargs)
                    else:
                        result = await func(*args[:sel_idx], msg, *args[sel_idx + 1 :], **kwargs)

                    # Convert message → dict for RPC reply
                    if isinstance(result, Message):
                        return result.to_dict()
                    return result

                return wrapper

            # ---- Sync path ----
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Extract the input dict
                input_dict: Any = kwargs.pop("data", None)
                sel_idx = -1
                if input_dict is None and len(args) >= 1:
                    for idx, a in enumerate(args):
                        if isinstance(a, dict) or isinstance(a, cls):
                            input_dict = a
                            sel_idx = idx
                            break
                if input_dict is None:
                    raise ValueError("Could not find any suitable arguments in the decoratr!")

                # Convert dict → message (uses correct subclass)
                if not isinstance(input_dict, cls):
                    msg = cls.from_dict(input_dict)  # type: ignore[arg-type]
                else:
                    msg = input_dict

                # Call the decorated function.
                # sel_idx == -1 means the message arrived via the ``data=``
                # kwarg (popped above); args[0] is the bound ``self``.
                if sel_idx == -1:
                    result = func(args[0], msg, *args[1:], **kwargs)
                else:
                    result = func(*args[:sel_idx], msg, *args[sel_idx + 1 :], **kwargs)

                # Convert message → dict for RPC reply
                if isinstance(result, Message):
                    return result.to_dict()
                return result

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Slicing (concrete subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def slice_by(self, indices: List[int]) -> Self:  # type: ignore[override]
        """Select a subset of samples by index, returning a new message.

        Each subclass reimplements this to slice its own fields correctly.
        Used by the group-by iterators below.
        """
        ...

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    def group_indices(self, level: str = "sample") -> List[Tuple[Tuple, List[int]]]:
        """Compute group keys and member index lists.

        The ``level`` string controls which tracking fields form the group key.
        Accepted values:

            - ``"dataset"``     → group by (dataset_id,)
            - ``"prompt"``      → group by (dataset_id, entry_id)
            - ``"rollout"``     → group by (dataset_id, entry_id, rollout_id)
            - ``"sample"``      → group by all four (default)

        Samples need not be ordered by the grouping fields — every position is
        scanned and members are collected into their group regardless of where
        they appear in the batch.

        Args:
            level: Which set of tracking fields to group by.

        Returns:
            List of (key_tuple, [indices]) in first-seen (batch) order.
        """
        if level == "dataset":
            _key = lambda i: (self.dataset_ids[i],)
        elif level == "prompt":
            _key = lambda i: (self.dataset_ids[i], self.entry_ids[i])
        elif level == "rollout":
            _key = lambda i: (
                self.dataset_ids[i],
                self.entry_ids[i],
                self.rollout_ids[i],
            )
        else:  # "sample" — all four tracking fields
            _key = lambda i: (
                self.dataset_ids[i],
                self.entry_ids[i],
                self.rollout_ids[i],
                self.subrollout_ids[i],
            )

        groups: dict[Tuple, List[int]] = {}
        for i in range(self.batch_size):
            groups.setdefault(_key(i), []).append(i)

        # Return in first-seen (batch) order
        ordered: List[Tuple[Tuple, List[int]]] = []
        seen: set[Tuple] = set()
        for i in range(self.batch_size):
            key = _key(i)
            if key not in seen:
                seen.add(key)
                ordered.append((key, groups[key]))
        return ordered

    # ------------------------------------------------------------------
    # Group-by iterators
    # ------------------------------------------------------------------

    def iter_by_sample(self) -> Iterator[Tuple[Tuple, "BatchedMessage"]]:
        """Iterate over unique samples (dataset_id, entry_id, rollout_id, subrollout_id).

        Yields (key, msg) where key is the four-field tuple and msg is a
        single-sample message.
        """
        for key, indices in self.group_indices(level="sample"):
            yield key, self.slice_by(indices)

    def iter_by_rollout(self) -> Iterator[Tuple[Tuple, "BatchedMessage"]]:
        """Iterate over rollouts (dataset_id, entry_id, rollout_id).

        Yields (key, msg) where key is the three-field tuple and msg may
        contain multiple sub-rollout stages.
        """
        for key, indices in self.group_indices(level="rollout"):
            yield key, self.slice_by(indices)

    def iter_by_prompt(self) -> Iterator[Tuple[Tuple, "BatchedMessage"]]:
        """Iterate over prompts (dataset_id, entry_id).

        Yields (key, msg) where key is the two-field tuple and msg contains
        all rollout replicas (and sub-rollout stages) for that prompt.
        """
        for key, indices in self.group_indices(level="prompt"):
            yield key, self.slice_by(indices)

    def iter_by_dataset(self) -> Iterator[Tuple[Tuple, "BatchedMessage"]]:
        """Iterate over datasets (dataset_id).

        Yields (key, msg) where key is the one-field tuple and msg contains
        all samples from that dataset.
        """
        for key, indices in self.group_indices(level="dataset"):
            yield key, self.slice_by(indices)


# ---------------------------------------------------------------------------
# Prompt / rollout messages
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class PromptBatch(BatchedMessage):
    """A batch of prompts sent from the orchestrator to the rollout node.

    Dimensionality: [batch]

    Attributes:
        prompts:    raw prompt strings (tokenizer will be applied by rollout)
        step:       current training step
        log_table:  whether to log a W&B table for this batch (orchestrator decides)
        log_prefix: prefix for metric keys and table names (e.g. "" for train, "val." for eval)
        extra:      free-form metadata forwarded downstream
    """

    prompts: List[str]
    responses: List[str] = field(default_factory=list)  # ground-truth answers (for reward)
    system_prompt: Optional[str] = None  # system message for chat template
    # few_shot_multiturn: example turns inserted between the
    # system message and the real user question by the rollout's chat
    # template (None = legacy behavior). Batch-constant, like system_prompt.
    prefix_messages: Optional[List[Dict[str, str]]] = None
    # Per-prompt chat message lists (parallel to prompts; template applied by
    # the rollout engine). None = legacy flat-prompt path.
    conversations: Optional[List[List[Dict[str, Any]]]] = None
    # Tool schemas for the chat template (requires conversations).
    tools: Optional[List[Dict[str, Any]]] = None
    # Constrained decoding for every prompt of the batch: one SGLang
    # structured-output key, e.g. {"ebnf": "..."} or {"regex": "..."}.
    # Served by the agentic rollout node only; the base node rejects it.
    sampling_constraint: Optional[Dict[str, Any]] = None
    step: int = 0
    log_table: bool = False  # orchestrator-controlled table logging
    log_prefix: str = ""  # metric/table naming prefix
    temperature: Optional[float] = None  # override rollout config
    top_p: Optional[float] = None  # override rollout config
    return_logprobs: bool = False  # request per-token logprobs from rollout
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}

    @property
    def batch_size(self) -> int:
        return len(self.prompts)

    def split_into(self, n: int) -> List[Self]:  # type: ignore[override]
        if n == 1:
            return [self]
        shard_idx_lists = _compute_shard_indices(n, self.batch_size)
        return [
            PromptBatch(
                prompts=_shard_list(self.prompts, idxs),
                responses=_shard_list(self.responses, idxs),
                system_prompt=self.system_prompt,
                prefix_messages=self.prefix_messages,
                conversations=(
                    _shard_list(self.conversations, idxs)
                    if self.conversations is not None
                    else None
                ),
                tools=self.tools,
                sampling_constraint=self.sampling_constraint,
                dataset_ids=_shard_list(self.dataset_ids, idxs),
                entry_ids=_shard_list(self.entry_ids, idxs),
                rollout_ids=_shard_list(self.rollout_ids, idxs),
                subrollout_ids=_shard_list(self.subrollout_ids, idxs),
                step=self.step,
                log_table=self.log_table,
                log_prefix=self.log_prefix,
                temperature=self.temperature,
                top_p=self.top_p,
                return_logprobs=self.return_logprobs,
                extra=dict(self.extra or {}),
            )
            for idxs in shard_idx_lists
        ]

    @classmethod
    def join(cls, shards: List[Self]) -> Self:  # type: ignore[override]
        if not shards:
            raise ValueError("Cannot join empty shard list")
        if len(shards) == 1:
            return shards[0]
        return cls(
            prompts=[p for s in shards for p in s.prompts],
            responses=[r for s in shards for r in s.responses],
            system_prompt=shards[0].system_prompt,
            prefix_messages=shards[0].prefix_messages,
            conversations=(
                [c for s in shards for c in (s.conversations or [])]
                if any(s.conversations is not None for s in shards)
                else None
            ),
            tools=shards[0].tools,
            sampling_constraint=shards[0].sampling_constraint,
            dataset_ids=[d for s in shards for d in s.dataset_ids],
            entry_ids=[e for s in shards for e in s.entry_ids],
            rollout_ids=[r for s in shards for r in s.rollout_ids],
            subrollout_ids=[s_id for s in shards for s_id in s.subrollout_ids],
            step=shards[0].step,
            log_table=shards[0].log_table,
            log_prefix=shards[0].log_prefix,
            temperature=shards[0].temperature,
            top_p=shards[0].top_p,
            return_logprobs=shards[0].return_logprobs,
            extra=dict(shards[0].extra or {}),
        )

    def slice_by(self, idxs: List[int]) -> Self:  # type: ignore[override]
        return PromptBatch(
            prompts=_shard_list(self.prompts, idxs),
            responses=_shard_list(self.responses, idxs),
            system_prompt=self.system_prompt,
            dataset_ids=_shard_list(self.dataset_ids, idxs),
            entry_ids=_shard_list(self.entry_ids, idxs),
            rollout_ids=_shard_list(self.rollout_ids, idxs),
            subrollout_ids=_shard_list(self.subrollout_ids, idxs),
            step=self.step,
            log_table=self.log_table,
            log_prefix=self.log_prefix,
            temperature=self.temperature,
            top_p=self.top_p,
            return_logprobs=self.return_logprobs,
            extra=dict(self.extra or {}),
        )


@dataclass(kw_only=True)
class RolloutResult(BatchedMessage):
    """Completions returned by the rollout node.

    Dimensionality:
        completions / logprobs (scalar):   [batch]
        token_ids / prompt_token_ids:      [batch, variable]
        logprobs (per-token):              [batch, variable] (optional)

    Attributes:
        completions:       generated text (one per prompt, kept for debugging)
        token_ids:         tokenized completions [batch, variable]
        prompt_token_ids:  tokenized prompts [batch, variable]
        logprobs:          per-token log-probs [batch, variable] if requested
        prompt_logprobs:   per-token log-probs for prompt portion (optional)
    """

    completions: List[str]
    token_ids: List[List[int]]
    prompt_token_ids: List[List[int]]
    logprobs: Optional[List[List[float]]] = None
    prompt_logprobs: Optional[List[List[float]]] = None
    finish_reasons: List[str] = field(default_factory=list)
    # Structured tool calls parsed by the agent rollout node
    # (AgentRolloutConfig.tool_call_parser; SGLang FunctionCallParser registry).
    # One list of {"name", "arguments"} dicts per sample ([] = no call).
    # None = not parsed (knob off); the completion text is always canonical.
    tool_calls: Optional[List[List[Dict[str, Any]]]] = None

    @property
    def batch_size(self) -> int:
        return len(self.completions)

    def split_into(self, n: int) -> List[Self]:  # type: ignore[override]
        if n == 1:
            return [self]
        shard_idx_lists = _compute_shard_indices(n, self.batch_size)
        return [
            RolloutResult(
                completions=_shard_list(self.completions, idxs),
                token_ids=_shard_list(self.token_ids, idxs),
                prompt_token_ids=_shard_list(self.prompt_token_ids, idxs),
                dataset_ids=_shard_list(self.dataset_ids, idxs),
                entry_ids=_shard_list(self.entry_ids, idxs),
                rollout_ids=_shard_list(self.rollout_ids, idxs),
                subrollout_ids=_shard_list(self.subrollout_ids, idxs),
                logprobs=_shard_list(self.logprobs, idxs) if self.logprobs else None,
                prompt_logprobs=_shard_list(self.prompt_logprobs, idxs)
                if self.prompt_logprobs
                else None,
                finish_reasons=_shard_list(self.finish_reasons, idxs),
                tool_calls=_shard_list(self.tool_calls, idxs) if self.tool_calls else None,
            )
            for idxs in shard_idx_lists
        ]

    @classmethod
    def join(cls, shards: List[Self]) -> Self:  # type: ignore[override]
        if not shards:
            raise ValueError("Cannot join empty shard list")
        if len(shards) == 1:
            return shards[0]
        return cls(
            completions=[c for s in shards for c in s.completions],
            token_ids=[t for s in shards for t in s.token_ids],
            prompt_token_ids=[t for s in shards for t in s.prompt_token_ids],
            dataset_ids=[d for s in shards for d in s.dataset_ids],
            entry_ids=[e for s in shards for e in s.entry_ids],
            rollout_ids=[r for s in shards for r in s.rollout_ids],
            subrollout_ids=[s_id for s in shards for s_id in s.subrollout_ids],
            logprobs=(
                [lp for s in shards if s.logprobs is not None for lp in s.logprobs]
                if shards[0].logprobs is not None
                else None
            ),
            prompt_logprobs=(
                [lp for s in shards if s.prompt_logprobs is not None for lp in s.prompt_logprobs]
                if shards[0].prompt_logprobs is not None
                else None
            ),
            finish_reasons=[fr for s in shards for fr in s.finish_reasons],
            tool_calls=(
                [tc for s in shards if s.tool_calls is not None for tc in s.tool_calls]
                if shards[0].tool_calls is not None
                else None
            ),
        )

    def slice_by(self, idxs: List[int]) -> Self:  # type: ignore[override]
        return RolloutResult(
            completions=_shard_list(self.completions, idxs),
            token_ids=_shard_list(self.token_ids, idxs),
            prompt_token_ids=_shard_list(self.prompt_token_ids, idxs),
            dataset_ids=_shard_list(self.dataset_ids, idxs),
            entry_ids=_shard_list(self.entry_ids, idxs),
            rollout_ids=_shard_list(self.rollout_ids, idxs),
            subrollout_ids=_shard_list(self.subrollout_ids, idxs),
            logprobs=_shard_list(self.logprobs, idxs) if self.logprobs else None,
            prompt_logprobs=_shard_list(self.prompt_logprobs, idxs)
            if self.prompt_logprobs
            else None,
            finish_reasons=_shard_list(self.finish_reasons, idxs),
            tool_calls=_shard_list(self.tool_calls, idxs) if self.tool_calls else None,
        )


# ---------------------------------------------------------------------------
# Reward messages
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RewardRequest(BatchedMessage):
    """Sent from the orchestrator (or training loop) to the reward node.

    Dimensionality:
        prompts / completions:          [batch] (optional, for text-based rewards)
        input_ids:                      [batch, variable] (token sequences)

    Attributes:
        prompts:      prompt strings (optional, kept for text-based rewards)
        completions:  rollout completions (optional, kept for text-based rewards)
        input_ids:    concatenated prompt+completion token sequences [batch, variable]
        step:         training step
        log_table:    whether to log a W&B table (orchestrator decides)
        log_prefix:   prefix for metric keys and table names
        extra:        arbitrary metadata (e.g. reference answers)
    """

    prompts: List[str]
    completions: List[str]
    responses: List[str] = field(default_factory=list)  # ground-truth answers
    input_ids: Optional[List[List[int]]] = None
    # Prompt/completion token split for token-based reward variants
    # (additive; the base reward node ignores these). Populated by the
    # training loop alongside input_ids so token-level rewards never have
    # to re-tokenize text.
    prompt_token_ids: Optional[List[List[int]]] = None
    completion_token_ids: Optional[List[List[int]]] = None
    logprobs: Optional[List[List[float]]] = None
    ref_logprobs: Optional[List[List[float]]] = None
    step: int = 0
    log_table: bool = False  # orchestrator-controlled table logging
    log_prefix: str = ""  # metric/table naming prefix
    finish_reasons: List[str] = field(default_factory=list)
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}

    @property
    def batch_size(self) -> int:
        return len(self.completions)

    def split_into(self, n: int) -> List[Self]:  # type: ignore[override]
        if n == 1:
            return [self]
        shard_idx_lists = _compute_shard_indices(n, self.batch_size)
        return [
            RewardRequest(
                prompts=_shard_list(self.prompts, idxs),
                completions=_shard_list(self.completions, idxs),
                responses=_shard_list(self.responses, idxs),
                input_ids=_shard_list(self.input_ids, idxs) if self.input_ids else None,
                prompt_token_ids=(
                    _shard_list(self.prompt_token_ids, idxs) if self.prompt_token_ids else None
                ),
                completion_token_ids=(
                    _shard_list(self.completion_token_ids, idxs)
                    if self.completion_token_ids
                    else None
                ),
                logprobs=_shard_list(self.logprobs, idxs) if self.logprobs else None,
                ref_logprobs=(_shard_list(self.ref_logprobs, idxs) if self.ref_logprobs else None),
                dataset_ids=_shard_list(self.dataset_ids, idxs),
                entry_ids=_shard_list(self.entry_ids, idxs),
                rollout_ids=_shard_list(self.rollout_ids, idxs),
                subrollout_ids=_shard_list(self.subrollout_ids, idxs),
                step=self.step,
                log_table=self.log_table,
                log_prefix=self.log_prefix,
                finish_reasons=_shard_list(self.finish_reasons, idxs),
                extra=dict(self.extra or {}),
            )
            for idxs in shard_idx_lists
        ]

    @classmethod
    def join(cls, shards: List[Self]) -> Self:  # type: ignore[override]
        if not shards:
            raise ValueError("Cannot join empty shard list")
        if len(shards) == 1:
            return shards[0]
        return cls(
            prompts=[p for s in shards for p in s.prompts],
            completions=[c for s in shards for c in s.completions],
            responses=[r for s in shards for r in s.responses],
            input_ids=(
                [inp for s in shards if s.input_ids is not None for inp in s.input_ids]
                if shards[0].input_ids is not None
                else None
            ),
            prompt_token_ids=(
                [p for s in shards if s.prompt_token_ids is not None for p in s.prompt_token_ids]
                if shards[0].prompt_token_ids is not None
                else None
            ),
            completion_token_ids=(
                [
                    c
                    for s in shards
                    if s.completion_token_ids is not None
                    for c in s.completion_token_ids
                ]
                if shards[0].completion_token_ids is not None
                else None
            ),
            logprobs=(
                [lp for s in shards if s.logprobs is not None for lp in s.logprobs]
                if shards[0].logprobs is not None
                else None
            ),
            ref_logprobs=(
                [rlp for s in shards if s.ref_logprobs is not None for rlp in s.ref_logprobs]
                if shards[0].ref_logprobs is not None
                else None
            ),
            dataset_ids=[d for s in shards for d in s.dataset_ids],
            entry_ids=[e for s in shards for e in s.entry_ids],
            rollout_ids=[r for s in shards for r in s.rollout_ids],
            subrollout_ids=[s_id for s in shards for s_id in s.subrollout_ids],
            step=shards[0].step,
            log_table=shards[0].log_table,
            log_prefix=shards[0].log_prefix,
            finish_reasons=[fr for s in shards for fr in s.finish_reasons],
            extra=dict(shards[0].extra or {}),
        )

    def slice_by(self, idxs: List[int]) -> Self:  # type: ignore[override]
        return RewardRequest(
            prompts=_shard_list(self.prompts, idxs),
            completions=_shard_list(self.completions, idxs),
            responses=_shard_list(self.responses, idxs),
            input_ids=_shard_list(self.input_ids, idxs) if self.input_ids else None,
            prompt_token_ids=(
                _shard_list(self.prompt_token_ids, idxs) if self.prompt_token_ids else None
            ),
            completion_token_ids=(
                _shard_list(self.completion_token_ids, idxs)
                if self.completion_token_ids
                else None
            ),
            logprobs=_shard_list(self.logprobs, idxs) if self.logprobs else None,
            ref_logprobs=_shard_list(self.ref_logprobs, idxs) if self.ref_logprobs else None,
            dataset_ids=_shard_list(self.dataset_ids, idxs),
            entry_ids=_shard_list(self.entry_ids, idxs),
            rollout_ids=_shard_list(self.rollout_ids, idxs),
            subrollout_ids=_shard_list(self.subrollout_ids, idxs),
            step=self.step,
            log_table=self.log_table,
            log_prefix=self.log_prefix,
            finish_reasons=_shard_list(self.finish_reasons, idxs),
            extra=dict(self.extra or {}),
        )


@dataclass(kw_only=True)
class RewardResult(BatchedMessage):
    """Rewards and advantages returned by the reward node.

    Dimensionality: [batch]

    Attributes:
        rewards:      scalar reward per completion (raw, pre-normalization)
        advantages:   group-normalized advantages (z-score) for GRPO loss
    """

    rewards: List[float]
    advantages: Optional[List[float] | List[List[float]]] = (
        None  # [batch] or [batch, completion_len]
    )

    @property
    def batch_size(self) -> int:
        return len(self.rewards)

    def split_into(self, n: int) -> List[Self]:  # type: ignore[override]
        if n == 1:
            return [self]
        shard_idx_lists = _compute_shard_indices(n, self.batch_size)
        return [
            RewardResult(
                rewards=_shard_list(self.rewards, idxs),
                advantages=(
                    _shard_list(self.advantages, idxs)  # type: ignore[assignment]
                    if self.advantages
                    else None
                ),
                dataset_ids=_shard_list(self.dataset_ids, idxs),
                entry_ids=_shard_list(self.entry_ids, idxs),
                rollout_ids=_shard_list(self.rollout_ids, idxs),
                subrollout_ids=_shard_list(self.subrollout_ids, idxs),
            )
            for idxs in shard_idx_lists
        ]

    @classmethod
    def join(cls, shards: List[Self]) -> Self:  # type: ignore[override]
        if not shards:
            raise ValueError("Cannot join empty shard list")
        if len(shards) == 1:
            return shards[0]
        return cls(
            rewards=[r for s in shards for r in s.rewards],
            advantages=(
                [a for s in shards if s.advantages is not None for a in s.advantages]  # type: ignore[list-item]
                if shards[0].advantages is not None
                else None
            ),
            dataset_ids=[d for s in shards for d in s.dataset_ids],
            entry_ids=[e for s in shards for e in s.entry_ids],
            rollout_ids=[r for s in shards for r in s.rollout_ids],
            subrollout_ids=[s_id for s in shards for s_id in s.subrollout_ids],
        )

    def slice_by(self, idxs: List[int]) -> Self:  # type: ignore[override]
        return RewardResult(
            rewards=_shard_list(self.rewards, idxs),
            advantages=_shard_list(self.advantages, idxs) if self.advantages else None,
            dataset_ids=_shard_list(self.dataset_ids, idxs),
            entry_ids=_shard_list(self.entry_ids, idxs),
            rollout_ids=_shard_list(self.rollout_ids, idxs),
            subrollout_ids=_shard_list(self.subrollout_ids, idxs),
        )


# ---------------------------------------------------------------------------
# Training / model-update messages
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TrainStepRequest(BatchedMessage):
    """Sent from the orchestrator to the model-update node.

    Dimensionality:
        input_ids / label_ids:    [batch, variable]
        rewards:                  [batch]
        prompts / completions:    [batch] (optional, for debugging)

    Contains everything needed to compute one training step.

    Attributes:
        input_ids:     concatenated prompt+completion token sequences [batch, variable]
        label_ids:     loss mask -- token ids for completion, -100 for prompt [batch, variable]
        rewards:       scalar rewards per sample
        prompts:       prompt strings (optional, kept for debugging)
        completions:   rollout completions (optional, kept for debugging)
        step:          global training step
        extra:         optional metadata
    """

    input_ids: List[List[int]]
    label_ids: List[List[int]]
    rewards: List[float]
    advantages: Optional[List[float] | List[List[float]]] = (
        None  # [batch] or [batch, completion_len]
    )
    # Engine/sampling logprobs: per-token logprobs recorded by the rollout
    # engine (vLLM/SGLang) AT GENERATION TIME, for the sampled tokens. These
    # are TRL's ``sampling_per_token_logps`` -- used ONLY for the bounded
    # engine<->trainer importance-sampling ratio, NEVER for the PPO ratio
    # ``coef_1``. (Historically misnamed ``old_logprobs``; renamed for clarity
    # -- "old" in the PPO sense is a different quantity, see ``old_logprobs``
    # below.) Shape [batch, completion_len].
    sampling_logprobs: Optional[List[List[float]]] = None
    # PPO old-policy logprobs: per-token logprobs of the BEHAVIOR/old policy
    # that collected the rollout, for the sampled tokens. These feed the PPO
    # ratio ``coef_1 = exp(curr - old)`` AND the ``old`` term of the engine IS
    # ratio. Normally the model-update node computes these itself via a fresh
    # detached forward pass on the synced weights (so coef_1 ~= 1 at GAS=1);
    # a caller MAY provide them instead (e.g. a replay/diagnostic with captured
    # old-policy logprobs, or a future num_iterations>1 path that caches them),
    # in which case the model updater SKIPS recomputing them. None = let the
    # updater compute them. Shape [batch, completion_len].
    old_logprobs: Optional[List[List[float]]] = None
    ref_logprobs: Optional[List[List[float]]] = None
    prompts: List[str] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    extra: Optional[Dict[str, Any]] = None

    # Full-batch overrides for loss normalization across microbatches.
    # Set to 0 to use local shard dimensions (default / backward-compatible).
    global_batch_size_override: int = 0  # full batch size for DR-GRPO denominator
    global_max_completion_len_override: int = (  # global max completion len for DR-GRPO
        0
    )
    global_total_tokens_override: int = 0  # global real-token count for bnpo/ce

    # Gradient-accumulation steps for THIS optimizer update (orchestrator-owned).
    # The orchestrator sets this = its gradient_accumulation_steps on every
    # forward_accum_grad call; the model divides the per-call loss by it before
    # backward so accumulated gradients form a correct mean over `accum_steps`
    # calls. 1 = no accumulation (byte-identical to legacy behaviour). This is
    # the GAS scale communicated across the orchestrator->model boundary -- the
    # model node no longer reads its config for this. Propagated unchanged to
    # every microbatch shard by split_into/slice_by.
    accum_steps: int = 1

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}

    @property
    def batch_size(self) -> int:
        return len(self.input_ids)

    def split_into(self, n: int) -> List[Self]:  # type: ignore[override]
        if n == 1:
            return [self]
        shard_idx_lists = _compute_shard_indices(n, self.batch_size)
        return [
            TrainStepRequest(
                input_ids=_shard_list(self.input_ids, idxs),
                label_ids=_shard_list(self.label_ids, idxs),
                rewards=_shard_list(self.rewards, idxs),
                advantages=(
                    _shard_list(self.advantages, idxs)  # type: ignore[assignment]
                    if self.advantages
                    else None
                ),
                sampling_logprobs=(_shard_list(self.sampling_logprobs, idxs) if self.sampling_logprobs else None),
                old_logprobs=(_shard_list(self.old_logprobs, idxs) if self.old_logprobs else None),
                ref_logprobs=(_shard_list(self.ref_logprobs, idxs) if self.ref_logprobs else None),
                prompts=_shard_list(self.prompts, idxs),
                completions=_shard_list(self.completions, idxs),
                dataset_ids=_shard_list(self.dataset_ids, idxs),
                entry_ids=_shard_list(self.entry_ids, idxs),
                rollout_ids=_shard_list(self.rollout_ids, idxs),
                subrollout_ids=_shard_list(self.subrollout_ids, idxs),
                extra=dict(self.extra or {}),
                global_batch_size_override=self.global_batch_size_override,
                global_max_completion_len_override=self.global_max_completion_len_override,
                global_total_tokens_override=self.global_total_tokens_override,
                accum_steps=self.accum_steps,
            )
            for idxs in shard_idx_lists
        ]

    @classmethod
    def join(cls, shards: List[Self]) -> Self:  # type: ignore[override]
        if not shards:
            raise ValueError("Cannot join empty shard list")
        if len(shards) == 1:
            return shards[0]
        return cls(
            input_ids=[inp for s in shards for inp in s.input_ids],
            label_ids=[lb for s in shards for lb in s.label_ids],
            rewards=[r for s in shards for r in s.rewards],
            advantages=(
                [a for s in shards if s.advantages is not None for a in s.advantages]  # type: ignore[list-item]
                if shards[0].advantages is not None
                else None
            ),
            sampling_logprobs=(
                [slp for s in shards if s.sampling_logprobs is not None for slp in s.sampling_logprobs]
                if shards[0].sampling_logprobs is not None
                else None
            ),
            old_logprobs=(
                [olp for s in shards if s.old_logprobs is not None for olp in s.old_logprobs]
                if shards[0].old_logprobs is not None
                else None
            ),
            ref_logprobs=(
                [rlp for s in shards if s.ref_logprobs is not None for rlp in s.ref_logprobs]
                if shards[0].ref_logprobs is not None
                else None
            ),
            prompts=[p for s in shards for p in s.prompts],
            completions=[c for s in shards for c in s.completions],
            dataset_ids=[d for s in shards for d in s.dataset_ids],
            entry_ids=[e for s in shards for e in s.entry_ids],
            rollout_ids=[r for s in shards for r in s.rollout_ids],
            subrollout_ids=[s_id for s in shards for s_id in s.subrollout_ids],
            extra=dict(shards[0].extra or {}),
            global_batch_size_override=shards[0].global_batch_size_override,
            global_max_completion_len_override=shards[0].global_max_completion_len_override,
            global_total_tokens_override=shards[0].global_total_tokens_override,
            accum_steps=shards[0].accum_steps,
        )

    def slice_by(self, idxs: List[int]) -> Self:  # type: ignore[override]
        return TrainStepRequest(
            input_ids=_shard_list(self.input_ids, idxs),
            label_ids=_shard_list(self.label_ids, idxs),
            rewards=_shard_list(self.rewards, idxs),
            advantages=_shard_list(self.advantages, idxs) if self.advantages else None,
            sampling_logprobs=_shard_list(self.sampling_logprobs, idxs) if self.sampling_logprobs else None,
            old_logprobs=_shard_list(self.old_logprobs, idxs) if self.old_logprobs else None,
            ref_logprobs=_shard_list(self.ref_logprobs, idxs) if self.ref_logprobs else None,
            prompts=_shard_list(self.prompts, idxs),
            completions=_shard_list(self.completions, idxs),
            dataset_ids=_shard_list(self.dataset_ids, idxs),
            entry_ids=_shard_list(self.entry_ids, idxs),
            rollout_ids=_shard_list(self.rollout_ids, idxs),
            subrollout_ids=_shard_list(self.subrollout_ids, idxs),
            extra=dict(self.extra or {}),
            global_batch_size_override=self.global_batch_size_override,
            global_max_completion_len_override=self.global_max_completion_len_override,
            global_total_tokens_override=self.global_total_tokens_override,
            accum_steps=self.accum_steps,
        )


@dataclass(kw_only=True)
class TrainStepResult(BatchedMessage):
    """Returned by the model-update node after one training step.

    Dimensionality: [1] per rank -- scalar metrics.

    Each rank produces one TrainStepResult (batch_size == 1). The
    ``join()`` method delegates to ``merge()`` which averages loss/lr
    and takes max grad_norm across ranks.
    """

    loss: float
    grad_norm: Optional[float] = None
    lr: Optional[float] = None
    total_completions: Optional[int] = None
    checkpoint_path: Optional[str] = None

    @property
    def batch_size(self) -> int:
        return 1

    def split_into(self, n: int) -> List[Self]:  # type: ignore[override]
        return [self] * n

    @classmethod
    def join(cls, shards: List[Self]) -> Self:  # type: ignore[override]
        return cls.merge(shards)  # type: ignore[return-value]

    @classmethod
    def merge(cls, results: List["TrainStepResult"]) -> "TrainStepResult":
        """Merge results from multiple ranks into a single TrainStepResult.

        Averages loss and lr across ranks, takes max grad_norm for safety,
        and uses rank 0's checkpoint path.

        ``total_completions`` is taken from ``results[0]`` rather than summed:
        the producer (``_model_update_step``) ``all_reduce``s the sample count
        with ``op=SUM`` *before* building the per-rank ``TrainStepResult``, so
        every rank already holds the *global* total. Summing across ranks would
        inflate the value by ``world_size``.
        """
        if not results:
            raise ValueError("Cannot merge empty result list")
        if len(results) == 1:
            return results[0]

        # Average loss
        losses = [r.loss for r in results if r.loss is not None]
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        # Max grad norm (for safety reporting)
        grad_norms = [r.grad_norm for r in results if r.grad_norm is not None]
        max_grad_norm = max(grad_norms) if grad_norms else None

        # Average lr
        lrs = [r.lr for r in results if r.lr is not None]
        avg_lr = sum(lrs) / len(lrs) if lrs else None

        # Every rank holds the same all-reduced global count (see docstring);
        # summing would multiply by world_size. Take the first non-None value.
        total_completions = next(
            (r.total_completions for r in results if r.total_completions is not None),
            None,
        )

        return cls(
            loss=avg_loss,
            grad_norm=max_grad_norm,
            lr=avg_lr,
            total_completions=total_completions,
            checkpoint_path=results[0].checkpoint_path,
        )

    def slice_by(self, indices: List[int]) -> Self:  # type: ignore[override]
        return self


# ---------------------------------------------------------------------------
# Weight-sync messages
# ---------------------------------------------------------------------------


class PauseUpdateRequest(ArgsKwargs):
    """Agentic mode, shared GPU: drain + sleep the engine around an optimizer step.

    Kwargs:
        drain_timeout_s: fail-closed bound on waiting for in-flight requests
            (the engine is never slept under live requests).
    """

    _must_have_kwargs_types = {"drain_timeout_s": float}


class WeightSyncRequest(ArgsKwargs):
    """Sent from the orchestrator to the rollout node after a checkpoint save.

    Kwargs:
        checkpoint_path: path to the HF weights directory
        step: training step (for logging / audit)
    """

    _must_have_kwargs_types = {"checkpoint_path": str, "step": int}


@dataclass(kw_only=True)
class MetricsBatch(Message):
    """Scalar metrics sent from any node to the progress-tracker node.

    Dimensionality: () -- flat dict of metric name → float.

    Attributes:
        metrics:     arbitrary {name: value} scalars (loss, reward_mean, lr, ...)
        node_id:     source node identifier ("orchestrator", "rollout", "model", …)
    """

    metrics: Dict[str, float]
    node_id: str = ""


@dataclass(kw_only=True)
class TableBatch(Message):
    """Table data sent from any node to the progress-tracker node.

    Dimensionality: [rows, columns] -- tabular data.

    The tracker converts this to ``wandb.Table(columns=..., data=...)``
    and logs it at the current step.

    Attributes:
        table_name:  table key in W&B (e.g. "rollouts", "rewards")
        columns:     column headers
        data:        list of rows, each row is a list of values
        node_id:     source node identifier
        step:        training step (for W&B logging)
    """

    table_name: str
    columns: List[str]
    data: List[List[str]]
    node_id: str = ""
    step: int = 0


@dataclass(kw_only=True)
class TopKLogitsRequest(BatchedMessage):
    """Sent from the orchestrator to the model-update node for top-K logits.

    Dimensionality:
        input_ids / label_ids:    [batch, variable]
        rewards:                  [batch]

    Attributes:
        input_ids:   concatenated prompt+completion token sequences [batch, variable]
        label_ids:   chosen token ids [batch, variable] (-100 for prompt portion)
        rewards:     scalar rewards per sample
        topk:        number of top candidates to extract per position
        temperature: temperature applied before ranking (default 1.0)
        step:        global training step
    """

    input_ids: List[List[int]]
    label_ids: List[List[int]]
    rewards: List[float]
    topk: int = 5
    temperature: float = 1.0
    step: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.input_ids)

    def split_into(self, n: int) -> List[Self]:  # type: ignore[override]
        if n == 1:
            return [self]
        shard_idx_lists = _compute_shard_indices(n, self.batch_size)
        return [
            TopKLogitsRequest(
                input_ids=_shard_list(self.input_ids, idxs),
                label_ids=_shard_list(self.label_ids, idxs),
                rewards=_shard_list(self.rewards, idxs),
                dataset_ids=_shard_list(self.dataset_ids, idxs),
                entry_ids=_shard_list(self.entry_ids, idxs),
                rollout_ids=_shard_list(self.rollout_ids, idxs),
                subrollout_ids=_shard_list(self.subrollout_ids, idxs),
                topk=self.topk,
                temperature=self.temperature,
                step=self.step,
            )
            for idxs in shard_idx_lists
        ]

    @classmethod
    def join(cls, shards: List[Self]) -> Self:  # type: ignore[override]
        if not shards:
            raise ValueError("Cannot join empty shard list")
        if len(shards) == 1:
            return shards[0]
        return cls(
            input_ids=[inp for s in shards for inp in s.input_ids],
            label_ids=[lb for s in shards for lb in s.label_ids],
            rewards=[r for s in shards for r in s.rewards],
            dataset_ids=[d for s in shards for d in s.dataset_ids],
            entry_ids=[e for s in shards for e in s.entry_ids],
            rollout_ids=[r for s in shards for r in s.rollout_ids],
            subrollout_ids=[s_id for s in shards for s_id in s.subrollout_ids],
            topk=shards[0].topk,
            temperature=shards[0].temperature,
            step=shards[0].step,
        )

    def slice_by(self, idxs: List[int]) -> Self:  # type: ignore[override]
        return TopKLogitsRequest(
            input_ids=_shard_list(self.input_ids, idxs),
            label_ids=_shard_list(self.label_ids, idxs),
            rewards=_shard_list(self.rewards, idxs),
            dataset_ids=_shard_list(self.dataset_ids, idxs),
            entry_ids=_shard_list(self.entry_ids, idxs),
            rollout_ids=_shard_list(self.rollout_ids, idxs),
            subrollout_ids=_shard_list(self.subrollout_ids, idxs),
            topk=self.topk,
            temperature=self.temperature,
            step=self.step,
        )


@dataclass(kw_only=True)
class TopKLogitsResult(BatchedMessage):
    """Returned by the model-update node after a top-K logits query.

    Dimensionality:
        topk_ids:          [batch, seq_len, topk]
        topk_logits:       [batch, seq_len, topk]
        selected_logits:   [batch, seq_len] — logit of the actually-chosen token

    Attributes:
        topk_ids:        token ids of the top-K candidates per position
        topk_logits:     logits of the top-K candidates per position
        selected_logits: logit value of the selected (label) token per position
        step:            global training step
    """

    topk_ids: List[List[List[int]]]
    topk_logits: List[List[List[float]]]
    selected_logits: List[List[float]]
    step: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.topk_ids)

    def split_into(self, n: int) -> List[Self]:  # type: ignore[override]
        if n == 1:
            return [self]
        shard_idx_lists = _compute_shard_indices(n, self.batch_size)
        return [
            TopKLogitsResult(
                topk_ids=_shard_list(self.topk_ids, idxs),
                topk_logits=_shard_list(self.topk_logits, idxs),
                selected_logits=_shard_list(self.selected_logits, idxs),
                dataset_ids=_shard_list(self.dataset_ids, idxs),
                entry_ids=_shard_list(self.entry_ids, idxs),
                rollout_ids=_shard_list(self.rollout_ids, idxs),
                subrollout_ids=_shard_list(self.subrollout_ids, idxs),
                step=self.step,
            )
            for idxs in shard_idx_lists
        ]

    @classmethod
    def join(cls, shards: List[Self]) -> Self:  # type: ignore[override]
        if not shards:
            raise ValueError("Cannot join empty shard list")
        if len(shards) == 1:
            return shards[0]
        return cls(
            topk_ids=[t for s in shards for t in s.topk_ids],
            topk_logits=[t for s in shards for t in s.topk_logits],
            selected_logits=[t for s in shards for t in s.selected_logits],
            dataset_ids=[d for s in shards for d in s.dataset_ids],
            entry_ids=[e for s in shards for e in s.entry_ids],
            rollout_ids=[r for s in shards for r in s.rollout_ids],
            subrollout_ids=[s_id for s in shards for s_id in s.subrollout_ids],
            step=shards[0].step,
        )

    def slice_by(self, idxs: List[int]) -> Self:  # type: ignore[override]
        return TopKLogitsResult(
            topk_ids=_shard_list(self.topk_ids, idxs),
            topk_logits=_shard_list(self.topk_logits, idxs),
            selected_logits=_shard_list(self.selected_logits, idxs),
            dataset_ids=_shard_list(self.dataset_ids, idxs),
            entry_ids=_shard_list(self.entry_ids, idxs),
            rollout_ids=_shard_list(self.rollout_ids, idxs),
            subrollout_ids=_shard_list(self.subrollout_ids, idxs),
            step=self.step,
        )


class ModelUpdateRequest(ArgsKwargs):
    """Sent from the orchestrator to trigger a gradient update step.

    Kwargs:
        lr: optional per-step learning rate override (None = use scheduler)
    """

    _must_have_kwargs_types = {"lr": (type(None), float)}


class ExportWeightsRequest(ArgsKwargs):
    """Sent from the orchestrator to export consolidated HF weights.

    All FSDP ranks participate (all-reduce to consolidate), but only rank 0
    writes to disk -- no expensive per-rank DCP I/O.

    Kwargs:
        path: directory path where weights will be saved
    """

    _must_have_kwargs_types = {"path": str}


class OperationResult(ArgsKwargs):
    """Generic success/failure message for model-update operations.

    Used by: sleep, wake, export_weights.
    """

    _must_have_kwargs_types = {"ok": bool}


# ---------------------------------------------------------------------------
# Async proxy / replay-buffer messages (actor-learner decoupled serving)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ExternalGenerateRequest(Message):
    """External agentic system → rollout proxy: generate completions.

    The proxy expands each prompt ``group_size`` times (GRPO group), serves
    them through its backend engine(s), returns the completions immediately,
    and logs every sample to its replay buffer tagged with the current
    ``weight_version`` — ready for training once rewards are reported.

    Attributes:
        prompts:        raw prompt strings (chat template applied by the engine node)
        system_prompt:  optional system message (chat template)
        group_size:     completions per prompt (GRPO group size; 1 = agentic solo)
        temperature / top_p / max_new_tokens: sampling overrides (None = engine
                        config defaults)
        client_id:      caller identity (logging / audit)
        metadata:       free-form caller metadata stored alongside the buffer entries
    """

    prompts: List[str]
    system_prompt: Optional[str] = None
    # Full chat conversations, parallel to ``prompts`` (same length; the
    # prompt string doubles as the audit/buffer display copy). When set, the
    # engine applies the model's chat template to each message list instead of
    # wrapping the flat prompt as a single user message — native multi-turn
    # structure (system / user / assistant / tool roles) and tool rendering.
    conversations: Optional[List[List[Dict[str, Any]]]] = None
    # OpenAI-style tool schemas rendered into the template (Qwen3 default
    # template emits tool docs + the <tool_call> tool_call grammar). Requires
    # ``conversations``; the flat-prompt path has no tool rendering.
    tools: Optional[List[Dict[str, Any]]] = None
    # Constrained decoding, e.g. {"ebnf": "..."}: completions can only take
    # shapes the grammar allows (see PromptBatch.sampling_constraint).
    sampling_constraint: Optional[Dict[str, Any]] = None
    group_size: int = 1
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
    client_id: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.tools is not None and self.conversations is None:
            raise ValueError(
                "tools requires conversations (flat prompts have no tool rendering)"
            )
        if (
            self.conversations is not None
            and len(self.conversations) != len(self.prompts)
        ):
            raise ValueError(
                f"conversations ({len(self.conversations)}) must be parallel to "
                f"prompts ({len(self.prompts)})"
            )


@dataclass(kw_only=True)
class ExternalGenerateResult(Message):
    """Rollout proxy → external agentic system: served completions.

    ``rollout_ids`` are the stable handles the caller uses to report rewards
    (``report_reward(rollout_ids, rewards)``). Index i of every list belongs
    to the same sample.
    """

    rollout_ids: List[str]
    completions: List[str]
    finish_reasons: List[str] = field(default_factory=list)
    weight_version: int = 0  # policy version that generated these samples
    # Structured tool calls parsed by the rollout backend
    # (AgentRolloutConfig.tool_call_parser). One list of {"name", "arguments"}
    # dicts per completion ([] = none, or the parser is off); the completion
    # text is always canonical.
    tool_calls: List[List[Dict[str, Any]]] = field(default_factory=list)


class ReportRewardRequest(ArgsKwargs):
    """External system → rollout proxy: rewards for previously served rollouts.

    Kwargs:
        rollout_ids:  handles returned by ``generate_external``
        rewards:      scalar reward per rollout (may arrive out of order / batched)
    """

    _must_have_kwargs_types = {"rollout_ids": list, "rewards": list}


class TouchRequest(ArgsKwargs):
    """External system → rollout proxy: heartbeat / TTL keepalive.

    Extends the pending-reward TTL for the given rollouts, so long-running
    agentic episodes (minutes-to-hours between generation and the terminal
    ``report_reward``) survive the buffer's crash-detection deadline: the
    TTL measures *crash silence*, not episode length. Call after every turn,
    or on a timer during long environment work.

    Kwargs:
        rollout_ids:  handles returned by ``generate_external``
    """

    _must_have_kwargs_types = {"rollout_ids": list}


@dataclass(kw_only=True)
class BufferEntry(Message):
    """One replay-buffer entry as handed to the async updater.

    Everything the model-update node needs to (re)train one sample: the token
    sequences, the per-token engine logprobs captured AT GENERATION TIME under
    ``weight_version`` (TRL's ``sampling_per_token_logps`` — feeds only the
    bounded engine IS ratio), the externally reported scalar reward, and the
    group identity for GRPO advantage computation.
    """

    rollout_id: str  # unique handle (external systems report rewards against it)
    group_id: int  # GRPO group (one per prompt; advantages are z-scored per group)
    rollout_idx: int  # index within the group (0..group_size-1)
    prompt: str
    completion: str
    prompt_token_ids: List[int] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    logprobs: Optional[List[float]] = None  # engine per-token logprobs (gen time)
    finish_reason: str = ""
    reward: float = 0.0  # externally reported scalar reward
    weight_version: int = 0  # policy version that generated this sample
    created_ts: float = 0.0  # arrival time (monotonic clock, for diagnostics)
    client_id: str = ""
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


@dataclass(kw_only=True)
class BufferPopRequest(Message):
    """Async updater → rollout proxy: pop a batch of ready (rewarded) rollouts.

    A group is ready iff ALL its members carry rewards (partial groups are
    never returned — group statistics must stay intact). Entries older than
    ``max_lag`` policy versions are discarded, not returned.
    """

    min_batch: int = 1  # return nothing unless this many samples are ready
    max_batch: int = 0  # 0 = no cap
    max_lag: int = -1  # staleness bound K; -1 = use the proxy config default


@dataclass(kw_only=True)
class BufferBatchResult(Message):
    """Rollout proxy → async updater: popped replay-buffer entries.

    ``entries`` are serialized :class:`BufferEntry` dicts (reconstruct with
    ``BufferEntry.from_dict`` — msgpack roundtrips carry them as plain dicts).
    Buffer-side stats are echoed for observability (W&B).
    """

    entries: List[Dict[str, Any]] = field(default_factory=list)
    weight_version: int = 0  # proxy's current policy version
    n_pending: int = 0  # entries awaiting reward
    n_discarded_stale: int = 0  # entries dropped by the staleness filter (cumulative)
    n_discarded_ttl: int = 0  # entries dropped by TTL expiry (cumulative)
    n_discarded_overflow: int = 0  # entries dropped by capacity overflow (cumulative)


# ---------------------------------------------------------------------------
# Top-K logits utility functions
# ---------------------------------------------------------------------------


def map_topk(
    logits: torch.Tensor,
    label_ids: torch.Tensor,
    topk: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract top-K logits + the selected token logit from full logits tensor.

    Args:
        logits:     [batch, seq_len, vocab_size] raw logits
        label_ids:  [batch, seq_len] chosen token ids (-100 for masked positions)
        topk:       number of top candidates to extract per position
        temperature: temperature applied before ranking (applied to logits, not softmax).
          Note: dividing logits by a positive scalar preserves order, so this does
          NOT change which tokens are selected as the top-K -- it only scales the
          returned topk_logits/selected_logits values (affects downstream
          softmax/logprob computations).

    Returns:
        (topk_ids, topk_logits, selected_logits) where:
            topk_ids:         [batch, seq_len, topk] token ids
            topk_logits:      [batch, seq_len, topk] logit values (temperature-scaled)
            selected_logits:  [batch, seq_len] logit of the label token (temperature-scaled)
                              -inf for masked positions
    """
    scaled = logits / max(temperature, 1e-9)

    # Top-K across vocab dimension (dim=-1)
    topk_logits, topk_ids = scaled.topk(topk, dim=-1)

    # Selected token logit: gather the logit at label_ids position
    mask = label_ids != -100  # [batch, seq_len]
    safe_labels = label_ids.clamp(min=0, max=scaled.size(-1) - 1)  # clamp for gather
    selected_logits = scaled.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
    selected_logits = selected_logits.masked_fill(~mask, float("-inf"))

    return topk_ids, topk_logits, selected_logits


def unmap_topk(
    topk_ids: torch.Tensor,
    topk_logits: torch.Tensor,
    selected_logits: torch.Tensor,
    vocab_size: int,
    label_ids: Optional[torch.Tensor] = None,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """Reconstruct a full logits tensor from top-K results.

    Non-top-K positions are filled with ``mask_value``.

    The chosen (selected) token's logit is written from ``selected_logits`` when
    ``label_ids`` is provided, so a chosen token that is NOT among the top-K
    (common under stochastic sampling when ``topk < vocab_size``) keeps its true
    logit instead of being left at ``mask_value`` -- which would yield logprob
    ``-inf`` and break downstream importance-sampling ratios. If ``label_ids``
    is None, the selected logit is only placed where it already matches a top-K
    entry (i.e. the old assumption that chosen \u2286 top-K); non-top-K chosen
    tokens are left at ``mask_value`` and a warning is logged via the docstring.

    Args:
        topk_ids:         [batch, seq_len, topk] token ids
        topk_logits:      [batch, seq_len, topk] logit values
        selected_logits:  [batch, seq_len] selected token logit (-inf at masked
                          positions; only written where finite)
        vocab_size:       full vocabulary size
        label_ids:        [batch, seq_len] chosen token ids (-100 for masked).
                          Optional but recommended; needed to place a selected
                          logit that falls outside the top-K.
        mask_value:       fill value for non-top-K, non-selected positions

    Returns:
        reconstructed: [batch, seq_len, vocab_size] logits tensor
    """
    batch, seq_len, _ = topk_ids.shape
    # Build the reconstructed tensor in a float dtype (logits are floats); using
    # topk_ids.new_full would inherit int64, which can't hold a -inf mask_value.
    reconstructed = topk_logits.new_full((batch, seq_len, vocab_size), fill_value=mask_value)
    reconstructed.scatter_(-1, topk_ids.long(), topk_logits)
    # Place the chosen token's true logit where it is finite (not masked).
    # This matters when the chosen token is NOT in the top-K (under stochastic
    # sampling with topk < vocab_size): without this, its position stays at
    # mask_value == -inf, giving logprob -inf and a 0 importance-sampling ratio.
    sel_mask = torch.isfinite(selected_logits)  # [batch, seq_len]
    if sel_mask.any() and label_ids is not None:
        safe_labels = label_ids.clamp(min=0, max=vocab_size - 1)  # clamp for gather/scatter
        # Only scatter at finite (valid) positions; keep -inf elsewhere.
        vals = selected_logits.where(sel_mask, torch.zeros_like(selected_logits))
        reconstructed.scatter_(-1, safe_labels.unsqueeze(-1), vals.unsqueeze(-1))
    return reconstructed


# ---------------------------------------------------------------------------
# State checkpoint messages
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class StateGetRequest(Message):
    """Request to read a node's state.

    Attributes:
        keys: top-level keys to return (e.g. ["config"], ["state"], or [] for all).
              Empty list means return every key the node exposes.
    """

    keys: List[str] = field(default_factory=list)


@dataclass(kw_only=True)
class StateSetRequest(Message):
    """Request to write a node's state.

    Attributes:
        keys: top-level keys to apply (e.g. ["config"], ["state"]).
              Only the named keys in ``data`` are applied; others are ignored.
        data: mapping of key → value (e.g. {"config": {...}, "state": {...}}).
    """

    keys: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class StateResult(Message):
    """Response from get_state or set_state.

    Attributes:
        data:  for get_state — requested {key: value} pairs.
               for set_state — echoed {key: value} that were applied.
        ok:    True if the operation succeeded, False if some key(s) failed.
        error: human-readable error string (populated when ok is False).
    """

    data: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""


class NoInputRequest(ArgsKwargs):
    """Empty request for side-effect-only endpoints (wake, sleep, etc.)."""

    pass


class FlushMetrics(ArgsKwargs):
    """Trigger the tracker to flush all accumulated metrics to W&B.

    Sub-nodes send ``log_metrics`` to the tracker which accumulates
    locally.  The orchestrator sends ``flush`` once per global step,
    at which point the tracker calls
    ``wandb.log(accumulated, step=step)`` with the correct global step.

    Kwargs:
        step: the global training step to associate with this flush
    """

    _must_have_kwargs_types = {"step": int}


class CheckpointRequest(ArgsKwargs):
    """Request for any checkpoint operation (save/load, hot/cold).

    Carries a path and step — the endpoint name (save_hot_checkpoint,
    save_cold_checkpoint, load_hot_checkpoint) disambiguates semantics.

    Kwargs:
        path: directory for checkpoint files
        step: current training step (for logging)
    """

    _must_have_kwargs_types = {"path": str, "step": int}


class CheckpointResult(ArgsKwargs):
    """Response from any checkpoint operation."""

    _must_have_kwargs_types = {"ok": bool, "path": str, "step": int, "error": str}


# ---------------------------------------------------------------------------
# Dataloader messages
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BatchRequest(Message):
    """Request a batch from the dataloader node.

    Attributes:
        dataset_name: which dataset to pull from (e.g. 'train', 'val')
        batch_size: optional override of the dataset's configured batch size
        step: current training step (for logging)
    """

    dataset_name: str = ""
    batch_size: int = 0  # 0 = use dataset's configured batch_size
    step: int = 0


@dataclass(kw_only=True)
class BatchResult(Message):
    """Response from the dataloader node.

    Wraps the fields of a PromptBatch plus success/error info.
    """

    ok: bool = True
    error: str = ""
    # PromptBatch fields
    prompts: List[str] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    # few_shot_eos: the example turns MUST survive this wrapper —
    # _from_dict(BatchResult, …) filters to these fields, and to_prompt_batch()
    # rebuilds from them; missing here = examples silently dropped = the
    # model trains on an instruction-only prompt (seen as the end-of-thought
    # rate dropping from 61% to 7% and 7000-token example-free rambles).
    prefix_messages: Optional[List[Dict[str, str]]] = None
    step: int = 0
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    return_logprobs: bool = False
    extra: Optional[Dict[str, Any]] = None
    # Tracing fields
    dataset_ids: List[str] = field(default_factory=list)
    entry_ids: List[int] = field(default_factory=list)
    rollout_ids: List[int] = field(default_factory=list)
    subrollout_ids: List[int] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.prompts)

    def to_prompt_batch(self) -> "PromptBatch":
        """Convert to a PromptBatch for the orchestrator."""
        return PromptBatch(
            prompts=self.prompts,
            responses=self.responses,
            system_prompt=self.system_prompt,
            prefix_messages=self.prefix_messages,
            step=self.step,
            temperature=self.temperature,
            top_p=self.top_p,
            return_logprobs=self.return_logprobs,
            extra=self.extra,
            dataset_ids=self.dataset_ids,
            entry_ids=self.entry_ids,
            rollout_ids=self.rollout_ids,
            subrollout_ids=self.subrollout_ids,
        )


# ---------------------------------------------------------------------------
# Legacy helpers (for backward compatibility with existing call sites)
# ---------------------------------------------------------------------------


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a message instance to a plain dict."""
    if isinstance(obj, Message):
        return obj.to_dict()
    return asdict(obj)


def _from_dict(cls: type, data: Dict[str, Any]) -> Any:
    """Reconstruct a message from a dict, ignoring extra keys."""
    if isinstance(cls, type) and issubclass(cls, Message):
        return cls.from_dict(data)
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})  # type: ignore[arg-type]
