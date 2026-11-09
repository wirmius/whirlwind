"""Progress-tracker node — logs metrics to Weights & Biases.

Exposes:
    log_metrics  — accumulate scalar metrics (called from any node)
    log_table    — log a table (columns + rows) to W&B
    flush        — flush accumulated metrics to W&B with a global step
    finish       — close the W&B run and flush
    ping         — health check
"""

from __future__ import annotations

import logging
from typing import Any

from whirlwind.config import ProgressTrackerConfig
from whirlwind.messages import (
    FlushMetrics,
    MetricsBatch,
    StateGetRequest,
    StateResult,
    StateSetRequest,
    TableBatch,
    _from_dict,
    _to_dict,
)
from whirlwind.whirlcoms import Node

logger = logging.getLogger(__name__)


class ProgressTrackerNode:
    """Progress-tracker node backed by Weights & Biases.

    Metrics are **accumulated** locally via ``log_metrics`` from any node
    (rollout throughput, reward stats, training loss) and only flushed
    to W&B when the orchestrator calls ``flush``, which provides the
    correct global step.  This ensures one ``wandb.log()`` per global
    step with all metrics aligned on the same x-axis.

    W&B run is initialized **lazily** on the first metrics call rather
    than at startup.  This allows state restore (run ID + resume step)
    to arrive via ``set_state`` *before* the run is created, so that
    a resumed training can continue logging into the **same** W&B run
    at the correct step.

    When ``config.backend`` is ``"none"``, all endpoints are no-ops and
    the W&B SDK is never imported.

    Usage:
        node = ProgressTrackerNode(ProgressTrackerConfig())
        await node.start()
        # ... nodes send metrics via RPC ...
        await node.stop()
    """

    def __init__(self, config: ProgressTrackerConfig) -> None:
        self.config = config
        self._node: Node | None = None
        self._wandb: Any | None = None
        self._run: Any | None = None
        self._resume_step: int = 0
        self._resume_run_id: str | None = None
        # Accumulator: node_id.metric_name → value, reset on each flush
        self._accumulated: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the underlying RPC node is still running (polled by the ``__main__`` keep-alive loop)."""
        return self._node.is_running if self._node is not None else False

    async def start(self) -> None:
        """Register with whirlcoms.  W&B init is deferred to first log."""
        self._node = Node(self.config.node_id, auto_process=True)
        self._node.expose("log_metrics")(self._handle_log_metrics)
        self._node.expose("log_table")(self._handle_log_table)
        self._node.expose("flush")(self._handle_flush)
        self._node.expose("finish")(self._handle_finish)
        self._node.expose("ping")(self._ping)
        self._node.expose("get_state")(self._get_state)
        self._node.expose("set_state")(self._set_state)
        await self._node.start()

        if self.config.backend == "wandb":
            import wandb

            self._wandb = wandb
            logger.info(
                "Progress tracker '%s' started (wandb project=%s, mode=%s)",
                self.config.node_id,
                self.config.wandb_project,
                self.config.wandb_mode,
            )
        else:
            logger.info(
                "Progress tracker '%s' started (backend=%s, no-op)",
                self.config.node_id,
                self.config.backend,
            )

    def _ensure_run(self) -> None:
        """Lazily create (or resume) the W&B run on first metrics call."""
        if self._run is not None or self._wandb is None:
            return

        init_kwargs: dict[str, Any] = dict(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity or None,
            mode=self.config.wandb_mode,
            config={
                "wandb_project": self.config.wandb_project,
                "wandb_entity": self.config.wandb_entity,
                "wandb_mode": self.config.wandb_mode,
            },
        )

        if self._resume_run_id:
            # Resume an existing run — continues logging into the same run
            init_kwargs["id"] = self._resume_run_id
            init_kwargs["resume"] = "allow"
            if self.config.wandb_run_name:
                init_kwargs["name"] = self.config.wandb_run_name
            logger.info(
                "Resuming W&B run %s (step offset=%d)",
                self._resume_run_id,
                self._resume_step,
            )
        else:
            # Fresh run
            if self.config.wandb_run_name:
                init_kwargs["name"] = self.config.wandb_run_name
            logger.info(
                "Creating new W&B run (project=%s)",
                self.config.wandb_project,
            )

        self._run = self._wandb.init(**init_kwargs)

    async def stop(self) -> None:
        # Flush any remaining accumulated metrics before closing
        if self._run is not None and self._accumulated:
            self._wandb.log(self._accumulated, step=self._resume_step or None)
            self._accumulated.clear()
        if self._run is not None:
            self._run.finish()
            logger.info("W&B run finished")
        if self._node:
            await self._node.stop()
        logger.info("Progress tracker node stopped")

    # ------------------------------------------------------------------
    # State endpoints
    # ------------------------------------------------------------------

    def _full_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self._run is not None:
            state["wandb_run_id"] = self._run.id
            state["wandb_run_name"] = self._run.name
        return {
            "config": _to_dict(self.config),
            "state": state,
        }

    @StateGetRequest.decorate_serdes()
    def _get_state(self, req: StateGetRequest) -> StateResult:
        full_state = self._full_state()
        if req.keys:
            full_state = {k: v for k, v in full_state.items() if k in req.keys}
        return StateResult(data=full_state, ok=True)

    @StateSetRequest.decorate_serdes()
    def _set_state(self, req: StateSetRequest) -> StateResult:
        applied: dict[str, Any] = {}
        for k, v in req.data.items():
            if req.keys and k not in req.keys:
                continue
            if k == "state" and isinstance(v, dict):
                self._resume_run_id = v.get("wandb_run_id", self._resume_run_id)
                self._resume_step = v.get("resume_step", self._resume_step)
            applied[k] = v
        return StateResult(data=applied, ok=True)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_log_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Accumulate a batch of scalar metrics (no W&B call yet).

        Metrics are prefixed with the source node ID and merged into
        ``_accumulated``.  The actual ``wandb.log()`` happens in
        ``flush`` when the orchestrator provides the global step.
        """
        batch = _from_dict(MetricsBatch, data)
        self._ensure_run()

        for name, value in batch.metrics.items():
            # Group metrics into W&B supergroups: keys already prefixed with
            # ``train/`` or ``val/`` (by the emitting node, which knows the
            # train-vs-val context) are preserved as-is so all training
            # metrics collapse under ``train/*`` and validation under ``val/*``
            # regardless of which node emitted them. Bare/unprefixed keys fall
            # back to the legacy ``{node_id}.{name}`` grouping for backward compat.
            if name.startswith("train/") or name.startswith("val/"):
                key = name
            else:
                key = f"{batch.node_id}.{name}" if batch.node_id else name
            # Numeric metrics -> float; string values (e.g. worst_offenders
            # log) pass through as-is so wandb can record them.
            self._accumulated[key] = float(value) if isinstance(value, (int, float)) else value

        return {"ok": True, "count": len(batch.metrics)}

    def _handle_log_table(self, data: dict[str, Any]) -> dict[str, Any]:
        """Log a table (columns + rows) to W&B."""
        batch = _from_dict(TableBatch, data)
        self._ensure_run()

        if self._run is not None and self._wandb is not None:
            table = self._wandb.Table(columns=batch.columns, data=batch.data)
            # Log at the same resumed global step as scalar flush
            # (req.step + self._resume_step in _handle_flush) so tables and
            # their scalars share the W&B x-axis instead of diverging on resume.
            global_step = batch.step + self._resume_step
            self._wandb.log({batch.table_name: table, "Step": batch.step}, step=global_step)
            logger.debug(
                "Logged table '%s': %d rows x %d cols (step=%d)",
                batch.table_name,
                len(batch.data),
                len(batch.columns),
                global_step,
            )

        return {"ok": True, "rows": len(batch.data)}

    @FlushMetrics.decorate_serdes()
    def _handle_flush(self, req: FlushMetrics) -> dict[str, Any]:
        """Flush accumulated metrics to W&B with the given global step."""
        if not self._accumulated:
            logger.debug("Flush called but no accumulated metrics")
            return {"ok": True, "flushed": 0}

        global_step = req.step + self._resume_step
        if self._run is not None and self._wandb is not None:
            self._wandb.log(self._accumulated, step=global_step)

        count = len(self._accumulated)
        self._accumulated.clear()
        logger.debug("Flushed %d metrics at step %d", count, global_step)
        return {"ok": True, "flushed": count}

    def _handle_finish(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Signal that training is complete — flush and close the run."""
        # Flush any remaining accumulated metrics
        if self._run is not None and self._accumulated:
            if self._wandb is not None:
                self._wandb.log(self._accumulated, step=self._resume_step or None)
                self._accumulated.clear()
        if self._run is not None:
            self._run.finish()
            self._run = None
            logger.info("W&B run finished via RPC")
        return {"ok": True}

    def _ping(self) -> dict[str, Any]:
        return {
            "ok": True,
            "backend": self.config.backend,
            "active": self._run is not None,
        }


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Whirlwind Progress Tracker Node")
    parser.add_argument(
        "--config",
        default="configs/tracker/wandb.yaml",
        help="Path to tracker YAML config",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = ProgressTrackerConfig.from_yaml(args.config)
    print(f"Starting tracker node: backend={config.backend}, project={config.wandb_project}")

    node = ProgressTrackerNode(config)

    async def _main() -> None:
        await node.start()
        print("Tracker node running. Press Ctrl+C to stop.")
        try:
            while node.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await node.stop()

    asyncio.run(_main())
