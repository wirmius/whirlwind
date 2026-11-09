"""Async GRPO orchestrator — buffer-driven actor-learner training loop.

EXPERIMENTAL — part of the async proxy-rollout mode
(``docs/async_proxy_rollout.md``). Replaces the serial
``generate → reward → train → sync`` loop with decoupled service-oriented RL:

    while step < total:
        entries  = proxy.pop_ready_batch(min_batch, max_batch, max_lag)
        #        ↑ externally driven generation, rewards reported out-of-band;
        #          whole ready GRPO groups only
        advantages = group z-score of the reported rewards   (shared helper)
        train one optimizer step on the popped batch         (parent helpers)
        every K steps: export weights → proxy.update_weights  (bumps its
                       weight_version; LoRA adapter hot-swap on the engine)

Generation is NOT driven by this loop: an external agentic system (or the
simulated agent) serves itself through the proxy's ``generate_external``
and reports rewards via ``report_reward``. The dataloader node is only
needed for periodic validation (val prompts + reference answers); the reward
node likewise (accuracy scoring of validation rollouts).

Differences vs the sync loop (``training_grpo.TrainingOrchestrator``):

- No dataloader in the training path; required nodes are ``rollout`` (the
  proxy) + ``model``.
- No colocate sleep/wake: the proxy is always-on by construction. If
  ``memory_management_enabled`` is left true it is ignored with a warning
  (non-colocate is the whole point).
- ``gradient_accumulation_steps`` must be 1 — each popped batch is one
  optimizer step (GAS across pops is a different, future cadence).
- Staleness is a first-class, bounded quantity: entries older than
  ``max_weight_version_lag`` policy versions are discarded by the buffer;
  the per-batch version lag is logged (the drift guardrail).

The loss path is UNCHANGED: engine logprobs captured at generation time ride
in ``sampling_logprobs`` (the bounded engine IS ratio quarantines them from
the PPO ratio — TIPS_AND_TRICKS.md §4.10), and the model node's fresh detached forward
provides ``old_logprobs`` under the CURRENT weights, which is exactly the
IS correction for version drift.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from whirlwind.config import AsyncTrainingConfig
from whirlwind.messages import (
    BufferBatchResult,
    BufferEntry,
    BufferPopRequest,
    MetricsBatch,
    RewardResult,
    RolloutResult,
    WeightSyncRequest,
    _from_dict,
    _to_dict,
)
from whirlwind.reward.advantages import compute_group_advantages
from whirlwind.training.training_grpo import TrainingOrchestrator
from whirlwind.whirlcoms import RpcError

logger = logging.getLogger(__name__)


class AsyncGRPOOrchestrator(TrainingOrchestrator):
    """Orchestrator for the buffer-driven (async) GRPO loop.

    Usage:
        config = AsyncTrainingConfig.from_yaml("async_grpo.yaml")
        orch = AsyncGRPOOrchestrator(config)
        await orch.run()
    """

    def __init__(self, config: AsyncTrainingConfig) -> None:
        super().__init__(config)
        self.config: AsyncTrainingConfig = config
        # The proxy registers as "rollout" (the orchestrator-facing endpoint).
        # Required nodes: proxy + model. Dataloader/reward are validation-only.
        self._required_nodes = ["rollout", "model"]
        self._last_pop: BufferBatchResult | None = None  # observability

    # ------------------------------------------------------------------
    # Bringup
    # ------------------------------------------------------------------

    async def _coordinate_bringup_sequence(self) -> None:
        """Async bringup: no colocate staggering, no sleep/wake — ever."""
        assert self._node is not None
        if self.config.memory_management_enabled:
            logger.warning(
                "Async mode is non-colocate by construction (the proxy is an "
                "always-on service); memory_management_enabled=true is ignored."
            )
        logger.info("Waiting for rollout proxy + model nodes...")
        await self._wait_for_nodes(["rollout", "model"], timeout=600.0)
        # The model node needs its explicit init trigger (same as the parent's
        # non-colocate path) — the proxy serves immediately after registration.
        await self._node.call("model", "initialize_model", {})
        logger.info("Model node initializing.")
        if self._validation_wanted():
            logger.info("Validation enabled — waiting for dataloader + reward nodes...")
            await self._wait_for_nodes(["dataloader", "reward"], timeout=300.0)
        logger.info("Async bringup complete.")

    def _validation_wanted(self) -> bool:
        return self.config.validation_every_n_steps > 0

    # ------------------------------------------------------------------
    # Buffer pull
    # ------------------------------------------------------------------

    async def _pop_ready_entries(self, step: int) -> list[BufferEntry]:
        """Poll the proxy until min_batch is ready or the step times out.

        ``call_typed`` raises on None (the v2 seam) so a proxy timeout is an
        RpcError, never an AttributeError on None (TIPS_AND_TRICKS.md §6.1).
        """
        assert self._node is not None
        cfg = self.config
        deadline = time.monotonic() + cfg.step_timeout_s
        req = _to_dict(
            BufferPopRequest(
                min_batch=cfg.min_train_batch,
                max_batch=cfg.max_train_batch,
                max_lag=cfg.max_weight_version_lag,
            )
        )
        while not self._shutting_down:
            result = await self._node.call_typed(
                "rollout", "pop_ready_batch", BufferBatchResult, req
            )
            self._last_pop = result
            if result.entries:
                return [BufferEntry.from_dict(e) for e in result.entries]
            if time.monotonic() >= deadline:
                logger.warning(
                    "Step %d: no ready batch after %.0fs (buffer pending=%d) — "
                    "idling one cycle",
                    step,
                    cfg.step_timeout_s,
                    result.n_pending,
                )
                return []
            await asyncio.sleep(cfg.pop_poll_interval_s)
        return []

    # ------------------------------------------------------------------
    # Batch assembly
    # ------------------------------------------------------------------

    def _entries_to_batch(
        self, entries: list[BufferEntry]
    ) -> tuple[RolloutResult, list[float]]:
        """Convert popped buffer entries into a RolloutResult + reward list.

        ``entry_ids`` = the buffer's group ids — every group was popped whole,
        so grouping by entry_id reproduces the GRPO groups exactly (the same
        contract the sync loop's _expand_prompts establishes).
        """
        n = len(entries)
        logprobs = [e.logprobs for e in entries]
        rollouts = RolloutResult(
            completions=[e.completion for e in entries],
            token_ids=[e.token_ids for e in entries],
            prompt_token_ids=[e.prompt_token_ids for e in entries],
            logprobs=logprobs if any(lp is not None for lp in logprobs) else None,
            finish_reasons=[e.finish_reason for e in entries],
            dataset_ids=[
                str(e.extra.get("dataset_id", "external")) for e in entries
            ],
            entry_ids=[e.group_id for e in entries],
            rollout_ids=[e.rollout_idx for e in entries],
            subrollout_ids=[0] * n,
        )
        rewards = [float(e.reward) for e in entries]
        return rollouts, rewards

    def _compute_advantages(
        self, entries: list[BufferEntry], rollouts: RolloutResult, rewards: list[float]
    ) -> list[float]:
        """Per-sample advantages for a popped batch (override to regroup).

        Default: group-relative over the buffer's groups (``entry_ids``).
        """
        return compute_group_advantages(
            rewards, rollouts.entry_ids, self.config.reward.scale_rewards
        )

    async def _before_update(self, step: int) -> None:
        """Runs before each optimizer step (override; no-op here)."""

    async def _after_update(self, step: int) -> None:
        """Runs after each optimizer step and adapter sync, also on failure (override)."""

    def _batch_version_lag(self, entries: list[BufferEntry]) -> tuple[float, int]:
        """Mean/max policy-version lag of a popped batch (staleness metric)."""
        if not entries:
            return 0.0, 0
        assert self._last_pop is not None
        current = self._last_pop.weight_version
        lags = [current - e.weight_version for e in entries]
        return sum(lags) / len(lags), max(lags)

    # ------------------------------------------------------------------
    # Weight sync (adapter hot-swap through the proxy)
    # ------------------------------------------------------------------

    async def _call_proxy_sync(self, checkpoint_path: str, step: int) -> int:
        """Export → proxy.update_weights; returns the new weight_version."""
        assert self._node is not None
        req = _to_dict(WeightSyncRequest(checkpoint_path=checkpoint_path, step=step))
        raw = await self._node.call("rollout", "update_weights", req)
        if not (isinstance(raw, dict) and raw.get("ok")):
            raise RpcError(
                f"Proxy weight sync failed at step {step} (reply={raw!r})"
            )
        version = int(raw.get("weight_version", -1))
        logger.info(
            "Step %d: weights synced via proxy — weight_version=%d", step, version
        )
        return version

    # ------------------------------------------------------------------
    # The async loop
    # ------------------------------------------------------------------

    async def run(self) -> list[dict[str, Any]]:
        """Execute the buffer-driven training loop. Returns step-level logs."""
        if self.config.model.gradient_accumulation_steps != 1:
            raise RuntimeError(
                "Async mode requires gradient_accumulation_steps=1: each popped "
                "buffer batch is one optimizer step. Batching is controlled by "
                "min_train_batch/max_train_batch (buffer-side), not GAS."
            )

        if self._node is None:
            await self.start()

        await self._coordinate_bringup_sequence()

        assert self._node is not None
        tracker_ready = "tracker" in self._node.topology
        if tracker_ready:
            logger.info("Tracker node discovered")

        # Hot-checkpoint resume (parent path syncs the proxy after load —
        # its _call_sync targets "rollout", which is the proxy).
        if self._load_hot_checkpoint_path:
            hot_path = self._load_hot_checkpoint_path
            print(f"Loading hot checkpoint from {hot_path}...")
            configs, states = await self.load_hot_checkpoint(hot_path)
            print(f"  Loaded configs for: {sorted(configs.keys())}")
            print(f"  Loaded states for: {sorted(states.keys())}")
        elif self.config.workdir and os.path.isdir(os.path.join(self.config.workdir, "hot_cp")):
            print(f"Auto-resuming from hot checkpoint in {self.config.workdir}...")
            configs, states = await self.load_hot_checkpoint(self.config.workdir)
            print(f"  Loaded configs for: {sorted(configs.keys())}")
            print(f"  Loaded states for: {sorted(states.keys())}")

        await self._maybe_save_config()

        cfg = self.config
        total = cfg.total_steps
        sync_every = max(1, cfg.adapter_sync_every_k_steps)
        logs: list[dict[str, Any]] = []

        step = self._own_step

        if self.config.shutdown_on_node_failure:
            self._failure_monitor_task = asyncio.create_task(self._failure_monitor_loop())

        # Baseline validation before any training (needs dataloader + reward)
        if (
            cfg.validation_every_n_steps > 0
            and cfg.validation_at_step_0
            and step == 0
            and tracker_ready
        ):
            await self._run_validation(step, total)

        while step < total:
            if self._shutting_down:
                logger.warning("Shutdown requested — exiting async loop at step %d", step)
                break

            try:
                entries = await self._pop_ready_entries(step)
            except RpcError as exc:
                logger.error("Step %d: pop_ready_batch failed: %s", step, exc)
                if self.config.shutdown_on_node_failure:
                    await self._shutdown_cluster(f"RPC failure at step {step}: {exc}")
                    break
                continue
            if not entries:
                continue  # idled out or shutdown mid-poll — not a training step
            rollouts, rewards = self._entries_to_batch(entries)
            advantages = self._compute_advantages(entries, rollouts, rewards)
            reward_result = RewardResult(
                rewards=rewards,
                advantages=advantages,
                dataset_ids=rollouts.dataset_ids,
                entry_ids=rollouts.entry_ids,
                rollout_ids=rollouts.rollout_ids,
                subrollout_ids=rollouts.subrollout_ids,
            )

            step += 1
            t0 = time.monotonic()
            train_result: dict[str, Any] = {}
            await self._before_update(step)
            try:
                fwd_result = await self._call_train_accum(rollouts, reward_result, step)
                train_result = await self._call_train_update(step)

                # Adapter sync every K optimizer steps (K=1 default).
                if step % sync_every == 0:
                    export_path = await self._call_export_weights(step)
                    await self._call_proxy_sync(export_path, step)
            except RpcError as exc:
                logger.error("Step %d failed: %s", step, exc)
                if self.config.shutdown_on_node_failure:
                    await self._shutdown_cluster(f"RPC failure at step {step}: {exc}")
                    break
                # Legacy best-effort path (shutdown disabled): the popped
                # batch is consumed either way; count the step and move on
                # (parent-sync behaviour — failures are counted, not retried).
                logs.append({"step": step, "error": str(exc)})
                continue
            finally:
                await self._after_update(step)

            elapsed = time.monotonic() - t0
            lag_mean, lag_max = self._batch_version_lag(entries)
            log = {
                "step": step,
                "loss": train_result.get("loss", fwd_result.get("loss")),
                "grad_norm": train_result.get("grad_norm"),
                "lr": train_result.get("lr"),
                "elapsed_s": round(elapsed, 2),
                "batch_size": rollouts.batch_size,
                "version_lag_mean": round(lag_mean, 3),
                "version_lag_max": lag_max,
            }
            logs.append(log)

            if tracker_ready:
                await self._log_metrics(log, step, total)
                await self._log_async_metrics()
                await self._call_flush(step)

            if step % self.config.log_every_n_steps == 0:
                _gn = log.get("grad_norm")
                _lr = log.get("lr")
                _loss = log.get("loss")
                logger.info(
                    "Step %3d/%d  loss=%.2e  grad=%.2f  lr=%.2e  batch=%d  "
                    "lag=%.1f  %.1fs",
                    step,
                    total,
                    _loss if _loss is not None else -1.0,
                    _gn if _gn is not None else -1.0,
                    _lr if _lr is not None else 0.0,
                    log["batch_size"],
                    log["version_lag_mean"],
                    log["elapsed_s"],
                )

            # Validation cadence is per optimizer step (GAS==1 here).
            if (
                cfg.validation_every_n_steps > 0
                and step % cfg.validation_every_n_steps == 0
                and tracker_ready
            ):
                await self._run_validation(step, total)

            if (
                cfg.workdir
                and cfg.hot_checkpoint_every_n_steps > 0
                and step % cfg.hot_checkpoint_every_n_steps == 0
            ):
                await self._save_hot_checkpoint(step)

            if cfg.workdir and cfg.cold_checkpoint_steps:
                cold_steps = cfg.cold_checkpoint_steps
                is_cold_step = (
                    isinstance(cold_steps, int) and step % cold_steps == 0
                    if isinstance(cold_steps, (int, list))
                    else step in cold_steps
                    if isinstance(cold_steps, list)
                    else False
                )
                if is_cold_step:
                    await self._save_cold_checkpoint(step)

            self._own_step = step

        if tracker_ready:
            await self._call_tracker_finish()
        return logs

    async def _log_async_metrics(self) -> None:
        """Log buffer/async observability to the tracker (fire-and-forget)."""
        assert self._node is not None
        pop = self._last_pop
        if pop is None:
            return
        metrics = {
            "async/buffer_pending": float(pop.n_pending),
            "async/discarded_stale": float(pop.n_discarded_stale),
            "async/discarded_ttl": float(pop.n_discarded_ttl),
            "async/discarded_overflow": float(pop.n_discarded_overflow),
            "async/proxy_weight_version": float(pop.weight_version),
        }
        try:
            await self._node.call(
                "tracker",
                "log_metrics",
                _to_dict(MetricsBatch(metrics=metrics, node_id="orchestrator")),
            )
        except RpcError as exc:
            logger.debug("Async metrics log failed: %s", exc)

    async def _run_validation(self, step: int, total: int) -> None:
        """Parent validation, gated on the validation-only nodes being up."""
        assert self._node is not None
        missing = [
            nid
            for nid in ("dataloader", "reward")
            if nid not in self._node.topology
        ]
        if missing:
            logger.warning(
                "Validation skipped at step %d — node(s) %s not running "
                "(async runs need dataloader + reward only for validation)",
                step,
                missing,
            )
            return
        await super()._run_validation(step, total)


def main(
    orch_cls: type[AsyncGRPOOrchestrator] = AsyncGRPOOrchestrator,
    config_cls: type = AsyncTrainingConfig,
) -> None:
    """Run an async training loop to completion. Sibling modules call ``main(MyLoop, MyConfig)``."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Whirlwind Async GRPO Orchestrator")
    parser.add_argument(
        "--config",
        default="configs/training/async_proxy_gsm8k.yaml",
        help="Path to async training YAML config",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--load-hot-checkpoint",
        default=None,
        help="Path to load hot checkpoint before training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = config_cls.from_yaml(args.config)
    print(
        f"Starting async training orchestrator: steps={config.total_steps}, "
        f"min_batch={config.min_train_batch}, K={config.max_weight_version_lag}"
    )

    orch = orch_cls(config)
    if args.load_hot_checkpoint:
        orch._load_hot_checkpoint_path = args.load_hot_checkpoint

    async def _main() -> None:
        await orch.start()
        logs = await orch.run()
        # Set when the loop was cut short (RPC failure, a required node lost).
        failure = orch.shutdown_reason

        print("\n=== Async Training Summary ===")
        for log in logs[-5:]:
            print(
                f"  Step {log['step']}: loss={log.get('loss', 'N/A')}, "
                f"grad={log.get('grad_norm', 'N/A')}, "
                f"batch={log.get('batch_size', 'N/A')}, "
                f"lag={log.get('version_lag_mean', 'N/A')}, "
                f"time={log.get('elapsed_s', 'N/A')}s"
            )

        # Tell the service nodes (proxy, engine, reward, dataloader, tracker)
        # to exit; failure paths already broadcast.
        try:
            await orch.shutdown_cluster("training complete")
        except Exception as exc:  # best-effort
            logger.warning("Completion shutdown broadcast failed: %s", exc)
        await orch.stop()
        if failure:
            print(f"Async training stopped early: {failure}")
            raise SystemExit(1)
        print("Async training complete.")

    asyncio.run(_main())


if __name__ == "__main__":
    main()
