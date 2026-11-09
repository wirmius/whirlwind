"""Replay buffer — decoupled experience store for async (actor-learner) GRPO.

EXPERIMENTAL — part of the async proxy-rollout mode (``docs/async_proxy_rollout.md``):
generation (serving an external agentic system) and training (the async
updater) are decoupled through an in-memory buffer keyed by ``rollout_id``.

Contract:

- ``put_many(entries)``      — the proxy logs every externally served rollout
  (prompt/completion tokens, per-token ENGINE logprobs captured at generation
  time, ``weight_version`` of the generating policy).
- ``set_reward(rollout_id, r)`` — the external system reports scalar rewards,
  possibly late, batched, or out of order. A group (one prompt's completions)
  is *ready* iff ALL its members carry rewards — partial groups are never
  returned, so GRPO group statistics stay intact.
- ``pop_ready(min_batch, max_batch, max_lag, current_version)`` — the updater
  pulls whole ready groups, FIFO, until ``min_batch`` samples are collected
  (else nothing — the updater idles and retries) and never more than
  ``max_batch``. Entries generated under a policy version more than
  ``max_lag`` versions behind ``current_version`` are DISCARDED (the
  staleness bound K from the design — bounded drift instead of hoping).

Eviction is group-atomic: TTL expiry, staleness, and capacity overflow drop
whole groups, never individual members (a half group would corrupt the
z-scored advantages — TIPS_AND_TRICKS.md §4.15).

Everything is plain in-memory state mutated synchronously: on the proxy node
all handlers run on one event loop and no buffer method awaits, so each call
is atomic without locks.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ReplayBufferEntry:
    """One logged rollout sample (the buffer's internal representation).

    ``rollout_id`` is the stable handle the external system reports rewards
    against; ``group_id`` identifies the GRPO group (all completions of one
    prompt from one generate call share it); ``logprobs`` are the ENGINE
    per-token logprobs captured at generation time under ``weight_version``
    (TRL's ``sampling_per_token_logps`` — they feed only the bounded engine
    IS ratio in the loss, never the PPO ratio).
    """

    rollout_id: str
    group_id: int
    rollout_idx: int
    prompt: str
    completion: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    logprobs: list[float] | None
    finish_reason: str
    weight_version: int
    created_ts: float
    client_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None  # None until the external system reports it


class ReplayBuffer:
    """In-memory FIFO replay buffer with group-atomic readiness and eviction.

    Args:
        max_entries:   capacity cap (samples, not groups). On overflow the
                       oldest groups are dropped (they are the stalest anyway).
        reward_ttl_s:  seconds a reward may be pending before the group is
                       discarded. Prefer discard over training on guessed
                       rewards.
        clock:         monotonic time source (injectable for tests).
    """

    def __init__(
        self,
        max_entries: int = 4096,
        reward_ttl_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self.reward_ttl_s = reward_ttl_s
        self._clock = clock

        self._entries: dict[str, ReplayBufferEntry] = {}
        self._order: deque[str] = deque()  # rollout_ids, arrival order (FIFO)
        self._groups: dict[int, list[str]] = {}  # group_id -> [rollout_id, ...]

        # Cumulative eviction counters (echoed in BufferBatchResult for W&B).
        self.n_discarded_stale = 0
        self.n_discarded_ttl = 0
        self.n_discarded_overflow = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        """Snapshot counters: pending entries, ready/partial groups, evictions."""
        ready = partial = 0
        for members in self._groups.values():
            if all(self._entries[rid].reward is not None for rid in members):
                ready += 1
            else:
                partial += 1
        return {
            "n_entries": len(self._entries),
            "n_groups": len(self._groups),
            "n_groups_ready": ready,
            "n_groups_partial": partial,
            "n_discarded_stale": self.n_discarded_stale,
            "n_discarded_ttl": self.n_discarded_ttl,
            "n_discarded_overflow": self.n_discarded_overflow,
        }

    # ------------------------------------------------------------------
    # Write paths
    # ------------------------------------------------------------------

    def put_many(self, entries: list[ReplayBufferEntry]) -> None:
        """Log a batch of entries (one generate call; groups complete here)."""
        if not entries:
            return
        if len(entries) > self.max_entries:
            raise ValueError(
                f"generate call produced {len(entries)} samples but the buffer "
                f"caps at {self.max_entries} — raise buffer_max_entries"
            )
        ids = {e.rollout_id for e in entries}
        if ids & set(self._entries):
            raise ValueError(f"duplicate rollout_id(s): {sorted(ids & set(self._entries))}")

        # Capacity: evict oldest groups until the new batch fits.
        need = len(self._entries) + len(entries) - self.max_entries
        while need > 0 and self._order:
            oldest_id = self._order[0]
            gid = self._entries[oldest_id].group_id
            need -= self._drop_group(gid, "overflow")

        for e in entries:
            self._entries[e.rollout_id] = e
            self._order.append(e.rollout_id)
            self._groups.setdefault(e.group_id, []).append(e.rollout_id)

    def set_reward(self, rollout_id: str, reward: float) -> bool:
        """Record an externally reported reward. False if the id is unknown
        (already popped, expired, or never served)."""
        entry = self._entries.get(rollout_id)
        if entry is None:
            return False
        entry.reward = float(reward)
        return True

    # ------------------------------------------------------------------
    # Read path (the updater's pull)
    # ------------------------------------------------------------------

    def pop_ready(
        self,
        min_batch: int = 1,
        max_batch: int = 0,
        max_lag: int = 3,
        current_version: int = 0,
    ) -> list[ReplayBufferEntry]:
        """Pop whole ready groups, FIFO, respecting the staleness bound.

        Returns [] unless at least ``min_batch`` samples are ready (the
        updater idles and retries — generation is externally driven). Never
        returns more than ``max_batch`` samples (0 = uncapped). Groups whose
        generating version lags more than ``max_lag`` behind
        ``current_version`` are discarded (the design's K bound), as are
        groups whose rewards exceeded the TTL.
        """
        self._expire_ttl()

        # Pass 1 — select whole ready groups (non-destructive): staleness
        # drops commit immediately (discarded regardless), but nothing ready
        # is consumed until we know the batch meets min_batch.
        selected: list[int] = []  # group ids
        selected_size = 0
        seen_groups: set[int] = set()

        for rid in list(self._order):
            if rid not in self._entries:
                continue  # already dropped
            entry = self._entries[rid]
            gid = entry.group_id
            if gid in seen_groups:
                continue
            seen_groups.add(gid)

            # Staleness bound K (group-atomic: all members share the version).
            if current_version - entry.weight_version > max_lag:
                self._drop_group(gid, "stale")
                continue

            members = self._groups.get(gid, [])
            if not all(self._entries[m].reward is not None for m in members):
                continue  # partial rewards — group not ready yet

            group_size = len(members)
            if max_batch and selected_size + group_size > max_batch:
                # Would exceed the cap. If min_batch is already met, stop;
                # otherwise keep scanning for smaller groups (rare).
                if selected_size >= min_batch:
                    break
                continue

            selected.append(gid)
            selected_size += group_size

            if selected_size >= min_batch and (
                not max_batch or selected_size >= max_batch
            ):
                break

        if selected_size < min_batch:
            return []

        # Pass 2 — commit: pop the selected groups (whole groups only).
        popped: list[ReplayBufferEntry] = []
        popped_ids: set[str] = set()
        for gid in selected:
            for m in self._groups.pop(gid, []):
                popped.append(self._entries.pop(m))
                popped_ids.add(m)
        if popped_ids:
            self._order = deque(rid for rid in self._order if rid not in popped_ids)
        return popped

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pending_since(self, entry: ReplayBufferEntry) -> float:
        """Timestamp a pending entry's reward TTL counts from (override to extend)."""
        return entry.created_ts

    def _expire_ttl(self) -> None:
        """Discard groups whose rewards have been pending past the TTL."""
        if self.reward_ttl_s <= 0:
            return
        deadline = self._clock() - self.reward_ttl_s
        stale_groups: set[int] = set()
        for rid in self._order:
            entry = self._entries.get(rid)
            if entry is not None and entry.reward is None and self._pending_since(entry) < deadline:
                stale_groups.add(entry.group_id)
        for gid in stale_groups:
            self._drop_group(gid, "ttl")

    def _drop_group(self, group_id: int, reason: str) -> int:
        """Remove a whole group; returns how many entries were dropped."""
        members = self._groups.pop(group_id, [])
        dropped = 0
        for m in members:
            if self._entries.pop(m, None) is not None:
                dropped += 1
        if dropped:
            member_set = set(members)
            self._order = deque(rid for rid in self._order if rid not in member_set)
            attr = f"n_discarded_{reason}"
            setattr(self, attr, getattr(self, attr) + dropped)
            logger.debug(
                "ReplayBuffer: dropped group %d (%d entries, reason=%s)",
                group_id, dropped, reason,
            )
        return dropped
