"""Agentic (multi-turn) async GRPO orchestrator (EXPERIMENTAL).

A sibling of :mod:`whirlwind.training.training_async` for multi-turn agent
traffic (docs/agentic_multiturn.md), configured by
:class:`whirlwind.config_agentic.AgenticTrainingConfig`.

**Episode grouping (the session registry).** An agent serves one
``generate_external`` call per (episode, turn) and tags each call with
``metadata["ww_tag"]``::

    {"task": ..., "problem": ..., "group": <id>, "group_size": <N>,
     "episode": <id>, "turn": <t>, ...}

At the end of an episode it reports the episode's return for every turn's
``rollout_id``. The buffer groups per call, so on its own every turn would be
a one-sample group with a zero advantage. This loop therefore *stages*
popped entries by ``(client, task, problem, group)`` until all ``group_size``
episodes of the group have arrived, z-scores the episodes' returns within
the group, and gives every turn of an episode its episode's advantage. Each
turn then trains as one sequence (its prompt is the conversation so far).
Groups that stay incomplete past ``group_complete_timeout_s`` or fall more
than ``max_weight_version_lag`` versions behind while staged are dropped and
counted. Untagged entries (plain single-turn traffic) train exactly like in
the async loop.

Contract for agents: a group counts an episode as present once any of its
turns is popped, so report an episode's return for all of its turns in ONE
``report_reward`` call (they then become ready together).

**Batch alignment.** The model node needs a batch that is a multiple of
``micro_batch_size x fsdp_size``, and turn counts vary, so each batch is
padded with copies of its shortest turns marked ``extra["ww_pad"]``. Pads get
a zero advantage, so (with ``kl_coef: 0``) they add no gradient; they only
grow the dr_grpo denominator by fewer than ``micro_batch_size x fsdp_size``
samples.

**Shared-GPU updates.** With ``engine_sleep_during_update`` the loop asks
the agent proxy (:mod:`whirlwind.rollout.rollout_proxy_agent`) to drain and
sleep the engine before each optimizer step. Once the step's adapter is
exported it wakes the engine (requests stay parked), pushes the adapter, and
then releases the parked requests, so nothing is served between the wake and
the swap. Any failure there shuts the cluster down (a paused engine with the
loop alive would park every agent forever).

Run it like the async loop::

    python3 -m whirlwind.training.training_agentic --config configs/training/agentic_reflect_gsm8k.yaml
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from whirlwind.config_agentic import AgenticTrainingConfig
from whirlwind.messages import (
    BufferBatchResult,
    BufferEntry,
    BufferPopRequest,
    MetricsBatch,
    PauseUpdateRequest,
    RolloutResult,
    _to_dict,
)
from whirlwind.reward.advantages import compute_group_advantages
from whirlwind.training.training_async import AsyncGRPOOrchestrator, main
from whirlwind.whirlcoms import RpcError

logger = logging.getLogger(__name__)

GroupKey = tuple[Any, ...]

# Each buffer poll drains up to this many training batches' worth of entries.
_DRAIN_BATCHES = 4


@dataclass
class _StagedGroup:
    """Entries of one episode group, collected across pops."""

    key: GroupKey
    expected: int  # episodes expected; 0 = untagged buffer group (complete on arrival)
    first_ts: float
    episodes: dict[Any, list[BufferEntry]] = field(default_factory=dict)

    @property
    def n_entries(self) -> int:
        return sum(len(v) for v in self.episodes.values())

    @property
    def complete(self) -> bool:
        return self.expected == 0 or len(self.episodes) >= self.expected

    def min_version(self) -> int:
        return min(e.weight_version for v in self.episodes.values() for e in v)


def episode_identity(entry: BufferEntry) -> tuple[GroupKey, Any, int]:
    """``(group key, episode key, expected episodes)`` of a buffer entry.

    Tagged entries (``extra["ww_tag"]`` with ``group``, ``episode`` and
    ``group_size``) group by ``(client, task, problem, group)``; anything else
    keeps the buffer's own group, one "episode" per sample (= plain GRPO).
    """
    tag = (entry.extra or {}).get("ww_tag")
    if (
        isinstance(tag, dict)
        and "group" in tag
        and "episode" in tag
        and int(tag.get("group_size") or 0) > 0
    ):
        key = ("tag", entry.client_id, tag.get("task", ""), tag.get("problem", ""), tag["group"])
        return key, tag["episode"], int(tag["group_size"])
    return ("buffer", entry.group_id), entry.rollout_id, 0


def episode_advantages(
    entries: list[BufferEntry], scale_rewards: str = "group"
) -> tuple[list[float], dict[str, float]]:
    """Per-entry advantages from episode-level group statistics.

    Each episode's return (the mean reward of its turns — agents report the
    same value for every turn) is z-scored against the other episodes of its
    group, and every turn gets its episode's advantage. Returns the advantages
    aligned with ``entries`` plus summary stats.
    """
    ep_index: dict[tuple[GroupKey, Any], int] = {}
    ep_rewards: list[list[float]] = []
    ep_group: list[int] = []
    group_index: dict[GroupKey, int] = {}
    entry_ep: list[int] = []
    for e in entries:
        gkey, ekey, _ = episode_identity(e)
        gi = group_index.setdefault(gkey, len(group_index))
        k = (gkey, ekey)
        if k not in ep_index:
            ep_index[k] = len(ep_rewards)
            ep_rewards.append([])
            ep_group.append(gi)
        ep_rewards[ep_index[k]].append(float(e.reward))
        entry_ep.append(ep_index[k])
    returns = [sum(r) / len(r) for r in ep_rewards]
    ep_adv = compute_group_advantages(returns, ep_group, scale_rewards)
    by_group: dict[int, list[float]] = {}
    for gi, ret in zip(ep_group, returns, strict=True):
        by_group.setdefault(gi, []).append(ret)
    n_signal = sum(1 for rets in by_group.values() if max(rets) > min(rets))
    stats = {
        "groups": float(len(by_group)),
        "episodes": float(len(returns)),
        "turns_per_episode": len(entries) / max(1, len(returns)),
        "groups_with_signal_frac": n_signal / max(1, len(by_group)),
        "mean_return": sum(returns) / max(1, len(returns)),
        "inconsistent_episode_rewards": float(
            sum(1 for r in ep_rewards if max(r) - min(r) > 1e-9)
        ),
    }
    return [ep_adv[i] for i in entry_ep], stats


def is_pad(entry: BufferEntry) -> bool:
    return bool((entry.extra or {}).get("ww_pad"))


def pad_to_multiple(entries: list[BufferEntry], align: int) -> list[BufferEntry]:
    """Append zero-advantage copies of the shortest entries up to a multiple of ``align``."""
    short = (-len(entries)) % max(1, align)
    if not entries or short == 0:
        return entries
    by_len = sorted(entries, key=lambda e: len(e.prompt_token_ids) + len(e.token_ids))
    pads = [
        dataclasses.replace(
            by_len[i % len(by_len)],
            rollout_id=f"{by_len[i % len(by_len)].rollout_id}:pad{i}",
            extra={"ww_pad": True},
        )
        for i in range(short)
    ]
    return entries + pads


class AgenticGRPOOrchestrator(AsyncGRPOOrchestrator):
    """Async GRPO loop with episode grouping and shared-GPU pause/resume."""

    config: AgenticTrainingConfig

    def __init__(self, config: AgenticTrainingConfig) -> None:
        super().__init__(config)
        self._staged: dict[GroupKey, _StagedGroup] = {}
        self._engine_paused = False
        self._last_stats: dict[str, float] = {}
        self.n_groups_dropped_timeout = 0
        self.n_groups_dropped_stale = 0
        self.n_groups_trained = 0
        self._warned_untagged_size = False

    # ------------------------------------------------------------------
    # Session registry: stage, complete, assemble
    # ------------------------------------------------------------------

    def _stage(self, entries: list[BufferEntry]) -> None:
        now = time.monotonic()
        for e in entries:
            gkey, ekey, expected = episode_identity(e)
            tag = (e.extra or {}).get("ww_tag")
            if gkey[0] == "buffer" and isinstance(tag, dict) and not self._warned_untagged_size:
                self._warned_untagged_size = True
                logger.warning(
                    "Tagged entries without ww_tag.group_size train as one-sample "
                    "groups (zero advantage) — add group_size to the agent's tag"
                )
            g = self._staged.get(gkey)
            if g is None:
                g = self._staged[gkey] = _StagedGroup(key=gkey, expected=expected, first_ts=now)
            g.episodes.setdefault(ekey, []).append(e)

    def _expire_staged(self) -> None:
        timeout = self.config.group_complete_timeout_s
        if timeout <= 0:
            return
        now = time.monotonic()
        for key in [k for k, g in self._staged.items() if not g.complete and now - g.first_ts > timeout]:
            g = self._staged.pop(key)
            self.n_groups_dropped_timeout += 1
            logger.warning(
                "Agentic: dropped incomplete group %s (%d/%d episodes after %.0fs)",
                key[1:],
                len(g.episodes),
                g.expected,
                now - g.first_ts,
            )

    def _take_complete_groups(self, current_version: int) -> list[BufferEntry]:
        """Complete groups, oldest first, once they reach ``min_train_batch`` samples.

        Groups are never split, and the cap is soft: while the batch is still
        below ``min_train_batch`` the next group is taken even if it overshoots
        ``max_train_batch`` (strict FIFO; a group larger than
        ``max_train_batch - min_train_batch`` could otherwise never train).
        """
        cfg = self.config
        chosen: list[GroupKey] = []
        total = 0
        for key, g in list(self._staged.items()):
            if not g.complete:
                continue
            if current_version - g.min_version() > cfg.max_weight_version_lag:
                self._staged.pop(key)
                self.n_groups_dropped_stale += 1
                continue
            n = g.n_entries
            if chosen and cfg.max_train_batch > 0 and total + n > cfg.max_train_batch and total >= cfg.min_train_batch:
                break
            chosen.append(key)
            total += n
            if total >= cfg.min_train_batch and (cfg.max_train_batch <= 0 or total >= cfg.max_train_batch):
                break
        if total < cfg.min_train_batch:
            return []
        out: list[BufferEntry] = []
        for key in chosen:
            g = self._staged.pop(key)
            self.n_groups_trained += 1
            for ep_entries in g.episodes.values():
                ep_entries.sort(key=lambda e: int(((e.extra or {}).get("ww_tag") or {}).get("turn", 0)))
                out.extend(ep_entries)
        return out

    async def _pop_ready_entries(self, step: int) -> list[BufferEntry]:
        if not self.config.group_episodes:
            return await super()._pop_ready_entries(step)
        assert self._node is not None
        cfg = self.config
        deadline = time.monotonic() + cfg.step_timeout_s
        # Drain what is ready; batch assembly happens here, per episode group.
        # Each turn is its own one-entry buffer group, and the buffer stops at
        # min_batch unless max_batch is set (max_batch=0 would pop ONE group
        # per poll), so ask for a few batches' worth per poll.
        drain = _DRAIN_BATCHES * max(1, cfg.min_train_batch, cfg.max_train_batch)
        req = _to_dict(
            BufferPopRequest(min_batch=1, max_batch=drain, max_lag=cfg.max_weight_version_lag)
        )
        while not self._shutting_down:
            result = await self._node.call_typed("rollout", "pop_ready_batch", BufferBatchResult, req)
            self._last_pop = result
            self._stage([BufferEntry.from_dict(e) for e in result.entries])
            self._expire_staged()
            batch = self._take_complete_groups(result.weight_version)
            if batch:
                return pad_to_multiple(batch, self._batch_alignment())
            if time.monotonic() >= deadline:
                logger.warning(
                    "Step %d: no complete episode groups after %.0fs (staged groups=%d, "
                    "entries=%d, buffer pending=%d) — idling one cycle",
                    step,
                    cfg.step_timeout_s,
                    len(self._staged),
                    sum(g.n_entries for g in self._staged.values()),
                    result.n_pending,
                )
                return []
            await asyncio.sleep(cfg.pop_poll_interval_s)
        return []

    def _batch_alignment(self) -> int:
        model = self.config.model
        mbs = int(getattr(model, "micro_batch_size", 1) or 1)
        if mbs <= 0:  # micro-batching bypassed: one shard per rank
            mbs = 1
        return mbs * max(1, int(getattr(model, "fsdp_size", 1) or 1))

    def _compute_advantages(
        self, entries: list[BufferEntry], rollouts: RolloutResult, rewards: list[float]
    ) -> list[float]:
        if not self.config.group_episodes:
            return super()._compute_advantages(entries, rollouts, rewards)
        real = [i for i, e in enumerate(entries) if not is_pad(e)]
        real_adv, stats = episode_advantages(
            [entries[i] for i in real], self.config.reward.scale_rewards
        )
        adv = [0.0] * len(entries)
        for i, a in zip(real, real_adv, strict=True):
            adv[i] = a
        stats["pad_entries"] = float(len(entries) - len(real))
        self._last_stats = stats
        if stats["inconsistent_episode_rewards"]:
            logger.warning(
                "Agentic: %d episode(s) reported different rewards for their turns; "
                "using the mean as the episode return",
                int(stats["inconsistent_episode_rewards"]),
            )
        logger.info(
            "Agentic batch: %d groups, %d episodes, %.2f turns/episode, mean return %.3f, "
            "%.0f%% of groups with signal",
            int(stats["groups"]),
            int(stats["episodes"]),
            stats["turns_per_episode"],
            stats["mean_return"],
            100 * stats["groups_with_signal_frac"],
        )
        return adv

    async def _log_async_metrics(self) -> None:
        await super()._log_async_metrics()
        if not self.config.group_episodes or self._node is None:
            return
        metrics = {f"agentic/{k}": float(v) for k, v in self._last_stats.items()}
        metrics.update(
            {
                "agentic/staged_groups": float(len(self._staged)),
                "agentic/staged_entries": float(sum(g.n_entries for g in self._staged.values())),
                "agentic/groups_trained": float(self.n_groups_trained),
                "agentic/groups_dropped_timeout": float(self.n_groups_dropped_timeout),
                "agentic/groups_dropped_stale": float(self.n_groups_dropped_stale),
            }
        )
        try:
            await self._node.call(
                "tracker",
                "log_metrics",
                _to_dict(MetricsBatch(metrics=metrics, node_id="orchestrator")),
            )
        except RpcError as exc:
            logger.debug("Agentic metrics log failed: %s", exc)

    # ------------------------------------------------------------------
    # Shared-GPU pause/resume around the update
    # ------------------------------------------------------------------

    async def _before_update(self, step: int) -> None:
        if not self.config.engine_sleep_during_update:
            return
        assert self._node is not None
        req = _to_dict(PauseUpdateRequest(drain_timeout_s=float(self.config.pause_drain_timeout_s)))
        try:
            await self._node.call("rollout", "pause_for_update", req)
        except Exception as exc:  # noqa: BLE001 — fail closed on anything
            logger.error("pause_for_update failed at step %d: %s", step, exc)
            await self._shutdown_cluster(f"engine pause failed at step {step}: {exc}")
            raise
        self._engine_paused = True
        logger.info("Engine paused (drained + asleep) for the update step")

    async def _call_proxy_sync(self, checkpoint_path: str, step: int) -> int:
        if not self._engine_paused:
            return await super()._call_proxy_sync(checkpoint_path, step)
        # Shared GPU: wake the engine while requests stay parked, load the new
        # adapter, and only then release them — nothing is served between the
        # wake and the swap.
        assert self._node is not None
        try:
            await self._node.call("rollout", "wake", {})
        except Exception as exc:  # noqa: BLE001 — a paused engine is unrecoverable
            logger.error("engine wake failed at step %d: %s", step, exc)
            self._engine_paused = False
            await self._shutdown_cluster(f"engine wake failed at step {step}: {exc}")
            raise RpcError(f"engine wake failed at step {step}: {exc}") from exc
        version = await super()._call_proxy_sync(checkpoint_path, step)
        await self._resume_engine(step)
        if self._shutting_down:
            raise RpcError(f"engine resume failed after the adapter sync at step {step}")
        return version

    async def _after_update(self, step: int) -> None:
        # Steps without an adapter sync (or a failed step) resume here.
        await self._resume_engine(step)

    async def _resume_engine(self, step: int) -> None:
        if not self._engine_paused:
            return
        self._engine_paused = False
        if self._shutting_down:
            return
        assert self._node is not None
        req = _to_dict(PauseUpdateRequest(drain_timeout_s=float(self.config.pause_drain_timeout_s)))
        try:
            await self._node.call("rollout", "resume_after_update", req)
            logger.info("Engine awake — serving resumed")
        except Exception as exc:  # noqa: BLE001 — a paused engine is unrecoverable
            logger.error("resume_after_update failed at step %d: %s", step, exc)
            if self.config.shutdown_on_node_failure:
                await self._shutdown_cluster(f"engine resume failed at step {step}: {exc}")


if __name__ == "__main__":
    main(AgenticGRPOOrchestrator, AgenticTrainingConfig)
