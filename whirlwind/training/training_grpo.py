"""Training orchestrator — coordinates the GRPO/RL training loop.

Embeds the whirlcoms Orchestrator (service registry) and runs the main
training loop: rollout → reward → train → weight sync.

Usage:
    from whirlwind import TrainingOrchestrator, TrainingConfig

    config = TrainingConfig.from_yaml("training.yaml")
    orch = TrainingOrchestrator(config)
    await orch.run()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from typing import Any

from whirlwind.config import TrainingConfig
from whirlwind.messages import (
    CheckpointRequest,
    CheckpointResult,
    FlushMetrics,
    MetricsBatch,
    ModelUpdateRequest,
    NoInputRequest,
    PromptBatch,
    RewardRequest,
    RewardResult,
    RolloutResult,
    StateResult,
    TrainStepRequest,
    TrainStepResult,
    WeightSyncRequest,
    _from_dict,
    _to_dict,
)
from whirlwind.whirlcoms import Node, Orchestrator, RpcError

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Main orchestrator for the GRPO/RL training loop.

    Embeds the whirlcoms Orchestrator for service discovery and health checks,
    then coordinates the four-phase training step:

        1. Send prompts → rollout node → completions + token_ids
        2. Send {input_ids, prompts, completions} → reward node → rewards
        3. Send {input_ids, label_ids, rewards} → model node → train step → checkpoint
        4. Send checkpoint path → rollout node → update weights (hot-reload)

    Memory management (when ``memory_management_enabled=True``):
        Sleeps/wakes rollout and model nodes between phases to allow
        co-location of both on the same GPUs without OOM.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._orch: Orchestrator | None = None
        self._node: Node | None = None
        self._running = False
        self._own_step: int = 0  # training step (restored on load_checkpoint)
        self._load_hot_checkpoint_path: str | None = (
            None  # set before run() to load hot checkpoint after bringup
        )
        # Failure handling
        self._shutting_down = False
        # Why the cluster was shut down before the loop finished (None while
        # running or after a normal finish); __main__ turns it into the exit code.
        self.shutdown_reason: str | None = None
        self._failure_monitor_task: asyncio.Task | None = None
        # Required peer nodes for the training loop. If any disappears from the
        # topology (heartbeat timeout) the failure monitor triggers a shutdown.
        self._required_nodes = ["rollout", "reward", "dataloader", "model"]

    # ------------------------------------------------------------------
    # LR Scheduler (moved from model node)
    # ------------------------------------------------------------------

    def _compute_lr(self, step: int) -> float:
        """Compute the learning rate for a given training step."""
        from whirlwind.utils import compute_lr

        return compute_lr(
            step=step,
            base_lr=self.config.model.learning_rate,
            total_steps=self.config.lr_schedule_steps or self.config.total_steps,
            warmup_ratio=self.config.model.warmup_ratio,
            warmup_steps=self.config.model.warmup_steps,
            scheduler_type=self.config.model.lr_scheduler_type,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the whirlcoms orchestrator and register this process as a node."""
        self._running = True
        # Heartbeat staleness budget. The 15s default was sized for
        # 1024-token payloads; at 8k budgets the val RolloutResult
        # (~8M token ids) blocks this loop in sync JSON parsing for
        # minutes — the trainer's OWN registration goes stale and the
        # orchestrator self-removes it mid-RPC, wedging the run forever
        # (every subsequent RPC unroutable). Scale via env for long-
        # context runs.
        _hb = float(os.environ.get("WHIRLWIND_HEARTBEAT_TIMEOUT", "15.0"))
        self._orch = Orchestrator(heartbeat_timeout=_hb)
        await self._orch.start()
        logger.info("Orchestrator socket started (heartbeat_timeout=%.0fs)", _hb)

        # Also register as a Node so we can call peers. fail_on_timeout: a timed-out
        # RPC raises RpcError (which the training loop turns into a cluster
        # shutdown) instead of returning None, which crashed the loop with an
        # AttributeError in _from_dict and left every service node running
        # Bringup RPCs that outlive WHIRLWIND_RPC_TIMEOUT
        # (e.g. a very slow initialize_model) now fail loudly instead of silently.
        self._node = Node(
            self.config.orchestrator_node_id, auto_process=False, fail_on_timeout=True
        )
        await self._node.start()
        # React to a cluster "shutdown" event broadcast (incl. one we initiate)
        # by marking the run loop for exit.
        self._node.set_shutdown_callback(self._on_cluster_shutdown)
        # FIXME: possibly make configurable in the future
        await asyncio.sleep(0.3)  # FIXME: possibly make configurable in the future
        logger.info("Training orchestrator '%s' started", self.config.orchestrator_node_id)

    async def stop(self) -> None:
        self._running = False
        if self._failure_monitor_task is not None:
            self._failure_monitor_task.cancel()
            # NOTE: CancelledError derives from BaseException (py3.8+), so
            # suppress(Exception) does NOT catch it — awaiting the task we
            # just cancelled would crash this stop() path and skip the
            # orchestrator/node teardown below (a completed run would leave
            # every service node running because stop() raised).
            with contextlib.suppress(asyncio.CancelledError):
                await self._failure_monitor_task
        if self._node:
            await self._node.stop()
        if self._orch:
            await self._orch.stop()
        logger.info("Training orchestrator stopped")

    # ------------------------------------------------------------------
    # Failure handling / graceful cluster shutdown
    # ------------------------------------------------------------------

    def _on_cluster_shutdown(self) -> None:
        """Callback fired when a cluster ``shutdown`` event reaches our node.

        Marks the run loop for exit. The actual stop() happens after run()
        breaks and ``__main__`` calls ``orch.stop()``.
        """
        if not self._shutting_down:
            logger.warning("Cluster shutdown received — will exit after current step")
            if self.shutdown_reason is None:
                self.shutdown_reason = "cluster shutdown received from another node"
        self._shutting_down = True

    async def shutdown_cluster(self, reason: str) -> None:
        """Broadcast a graceful cluster shutdown (idempotent, best-effort).

        Public wrapper so the normal-completion path (_main) can stop the
        service nodes without reaching into the failure-path internals.
        """
        await self._shutdown_cluster(reason)

    async def _shutdown_cluster(self, reason: str) -> None:
        """Broadcast a graceful shutdown to all nodes and flag the loop to exit.

        Best-effort: if the orchestrator registry itself is unreachable the
        broadcast is skipped and we just stop locally. Idempotent.
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        if "complete" not in reason and self.shutdown_reason is None:
            self.shutdown_reason = reason
        # "training complete" is the expected path — log it as info; genuine
        # failures keep the error level.
        log = logger.info if "complete" in reason else logger.error
        log("Initiating cluster shutdown: %s", reason)
        if self._node is not None:
            try:
                await self._node.broadcast_shutdown(reason)
            except Exception as exc:  # registry may itself be the failing piece
                logger.warning("Cluster shutdown broadcast failed: %s", exc)

    async def _failure_monitor_loop(self) -> None:
        """Background task: detect required nodes that vanished from topology.

        The whirlcoms Orchestrator drops nodes whose heartbeat timed out and
        broadcasts ``deregistered``; our Node's topology cache then no longer
        lists them. If a required node is missing for two consecutive checks,
        trigger a cluster shutdown (avoids acting on a momentary refresh gap).
        """
        assert self._node is not None
        miss_streak: dict[str, int] = {}
        while self._running and not self._shutting_down:
            await asyncio.sleep(5.0)
            topo = self._node.topology
            for nid in self._required_nodes:
                if nid not in topo:
                    miss_streak[nid] = miss_streak.get(nid, 0) + 1
                else:
                    miss_streak.pop(nid, None)
            if self.config.shutdown_on_node_failure:
                for nid, n in miss_streak.items():
                    if n >= 2:
                        await self._shutdown_cluster(
                            f"required node '{nid}' lost (heartbeat timeout)"
                        )
                        return

    @property
    def orchestrator(self) -> Orchestrator:
        """Access the underlying whirlcoms Orchestrator (for health checks, etc.)."""
        assert self._orch is not None
        return self._orch

    @property
    def topology(self) -> dict[str, dict]:
        """Current topology from the orchestrator's perspective."""
        assert self._orch is not None
        return self._orch.list_nodes()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    async def run(self) -> list[dict[str, Any]]:
        """Execute the full training loop.

        Returns:
            List of step-level log dicts.
        """
        if self._node is None:
            await self.start()

        # make sure that the collocation functions properly
        await self._coordinate_bringup_sequence()

        assert self._node is not None
        tracker_ready = "tracker" in self._node.topology
        if not tracker_ready:
            # The tracker is started by bringup before the training loop,
            # but its registration event can land after this one-shot check
            # (lost the race twice in production: validation + all wandb
            # metrics silently skipped for the whole run). Poll briefly
            # instead of accepting a false negative.
            for _ in range(15):
                await asyncio.sleep(1.0)
                if "tracker" in self._node.topology:
                    tracker_ready = True
                    break
        if tracker_ready:
            logger.info("Tracker node discovered")
        else:
            logger.warning(
                "Tracker node NOT discovered after 15s — validation and "
                "wandb metric logging will be skipped this run"
            )

        # Dataloader node is required
        if "dataloader" not in self._node.topology:
            raise RuntimeError("Dataloader node not found. Start it before training.")
        logger.info("Dataloader node discovered")

        # Load hot checkpoint from workdir if available
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

        # Auto-save config snapshot to workdir if it doesn't exist yet
        await self._maybe_save_config()

        total = self.config.total_steps
        reps = self.config.rollout_per_prompt
        acc_steps = self.config.model.gradient_accumulation_steps
        gen_steps = max(1, self.config.steps_per_generation)
        if gen_steps > 1 and acc_steps != 1:
            raise RuntimeError(
                "steps_per_generation>1 requires gradient_accumulation_steps=1: "
                "the two mechanisms overlap (both accumulate across multiple "
                "forward passes before an optimizer step). Use steps_per_generation "
                "for the TRL generate-once / train-N cadence, or "
                "gradient_accumulation_steps for legacy micro-accumulation — not both."
            )
        if gen_steps > 1 and self.config.memory_management_enabled:
            # The bringup sleeps both nodes and the window loop never wakes them:
            # the first generate would hit a released engine and the optimizer
            # step would run with its state on CPU. Refuse instead of warning
            # (the sleep/wake choreography is not wired for this loop).
            raise RuntimeError(
                "steps_per_generation>1 with memory_management_enabled=true is not "
                "supported: run it disjoint (memory_management_enabled: false)."
            )
        logs: list[dict[str, Any]] = []
        train_result: dict[str, Any] = {}

        step = self._own_step

        # Start the node-failure monitor (graceful cluster shutdown)
        if self.config.shutdown_on_node_failure:
            self._failure_monitor_task = asyncio.create_task(self._failure_monitor_loop())

        # Run baseline validation before any training (if enabled)
        if (
            self.config.validation_every_n_steps > 0
            and self.config.validation_at_step_0
            and step == 0
            and tracker_ready
        ):
            await self._run_validation(step, total)

        while step < total:
            if self._shutting_down:
                logger.warning("Shutdown requested — exiting training loop at step %d", step)
                break
            if gen_steps > 1:
                # TRL-style generate-once / train-N cadence (own loop).
                # step is the last COMPLETED step (0 fresh, N on resume); the
                # window loop numbers steps 1..total like this loop does.
                return await self._run_generation_loop(
                    step + 1, total, reps, gen_steps, tracker_ready, logs
                )
            step += 1
            t0 = time.monotonic()
            try:
                # ------------------------------------------------------------------
                # Memory management: wake rollout, sleep model before rollout
                # ------------------------------------------------------------------
                if self.config.memory_management_enabled:
                    await self._wake_rollout()
                    await self._sleep_model()

                # Fetch batch from dataloader node
                batch_result = await self._call_next_batch(step)
                raw_prompts = batch_result.prompts
                raw_responses = batch_result.responses or []
                batch_dataset_ids = batch_result.dataset_ids
                batch_entry_ids = batch_result.entry_ids
                system_prompt = batch_result.system_prompt
                prefix_messages = batch_result.prefix_messages

                # Expand: repeat each prompt `reps` times for diverse sampling
                (
                    expanded,
                    exp_dataset_ids,
                    exp_entry_ids,
                    exp_rollout_ids,
                    exp_subrollout_ids,
                ) = self._expand_prompts(raw_prompts, reps, batch_dataset_ids, batch_entry_ids)

                # Build entry_id → reference_answer map for response expansion.
                # Detect duplicate entry_ids (possible via the size<n duplication
                # in _compute_shard_indices or a dataloader collision): dict(zip)
                # would silently keep only the last response per entry, so the
                # duplicated samples would get the wrong reference answer.
                if len(set(batch_entry_ids)) != len(batch_entry_ids):
                    dupes = [e for e in set(batch_entry_ids) if batch_entry_ids.count(e) > 1]
                    raise RuntimeError(
                        f"Duplicate entry_ids in batch: {sorted(set(dupes))}; "
                        f"cannot build a 1:1 entry_id→response map"
                    )
                ref_answers = dict(zip(batch_entry_ids, raw_responses, strict=True))

                # Determine whether to request table logging for this step
                log_rollout = (
                    self.config.rollout_log_table_every_n_steps > 0
                    and step % self.config.rollout_log_table_every_n_steps == 0
                )
                log_reward = (
                    self.config.reward_log_table_every_n_steps > 0
                    and step % self.config.reward_log_table_every_n_steps == 0
                )
                log_prefix = ""  # "" for train, "val." for validation

                # Phase 1: Rollout (returns full RolloutResult with tokens)
                rollouts = await self._call_rollout(
                    expanded,
                    step,
                    system_prompt,
                    exp_dataset_ids,
                    exp_entry_ids,
                    exp_rollout_ids,
                    exp_subrollout_ids,
                    log_table=log_rollout,
                    log_prefix=log_prefix,
                    prefix_messages=prefix_messages,
                )

                # ------------------------------------------------------------------
                # Memory management: sleep rollout before reward + training
                # ------------------------------------------------------------------
                if self.config.memory_management_enabled:
                    await self._sleep_rollout()

                # Phase 2: Reward (pass tokens + strings + ground-truth to reward node)
                # Pass ref_answers; _call_reward expands them to match RolloutResult size
                reward_result = await self._call_reward(
                    rollouts,
                    expanded,
                    ref_answers,
                    step,
                    log_table=log_reward,
                    log_prefix=log_prefix,
                )

                # ------------------------------------------------------------------
                # Memory management: wake model before training
                # ------------------------------------------------------------------
                if self.config.memory_management_enabled:
                    await self._wake_model()

                # Phase 3: Train
                # Accumulate gradients over acc_steps rollouts, then flush
                fwd_result = await self._call_train_accum(rollouts, reward_result, step)

                logger.debug(
                    f"Step {step}; acc_steps: {acc_steps}; do we update? {step % acc_steps == 0}"
                )
                if step % acc_steps == 0:
                    # Flush accumulated gradients
                    train_result = await self._call_train_update(step)

                    # Export weights (HF consolidate only — no DCP/optimizer)
                    export_path = await self._call_export_weights(step)

                    # Phase 4: Weight sync
                    # Sleep the model FIRST (frees its GPU memory — optimizer
                    # states created by the just-run optimizer.step) so the
                    # rollout can resume its KV pool (resume_memory_occupation)
                    # without OOM'ing. Safe because _call_sync only RPCs the
                    # rollout (update_weights from the already-written export
                    # file at line above); the model is not needed awake for
                    # the sync. Fixes the colocate step-1 OOM where the rollout
                    # woke while the model still held ~optim-state GPU memory.
                    if self.config.memory_management_enabled:
                        await self._sleep_model()
                        await self._wake_rollout()
                    await self._call_sync(export_path, step)
                    logger.debug("Weights synced from %s", export_path)

                    # Sleep rollout after sync (model already asleep; the next
                    # iteration wakes the rollout + no-ops sleep_model).
                    if self.config.memory_management_enabled:
                        await self._sleep_rollout()

            except RpcError as exc:
                logger.error("Step %d failed: %s", step, exc)
                if self.config.shutdown_on_node_failure:
                    await self._shutdown_cluster(f"RPC failure at step {step}: {exc}")
                    break
                # Legacy behaviour (shutdown disabled): best-effort recovery so a
                # failed flush step does not leak accumulated gradients.
                if step % acc_steps == 0:
                    try:
                        await self._call_train_update(step)
                    except RpcError as exc2:
                        logger.error("Recovery zero_grad for step %d also failed: %s", step, exc2)
                logs.append({"step": step, "error": str(exc)})
                continue

            # The step is complete: record it NOW, so a hot checkpoint taken below
            # saves THIS step. It used to be set at the end of the iteration, so
            # state.json lagged the DCP shards by one and resume re-ran a step
            # that was already applied.
            self._own_step = step

            elapsed = time.monotonic() - t0
            log = {
                "step": step,
                "loss": train_result.get("loss", fwd_result.get("loss")),
                "grad_norm": train_result.get("grad_norm"),
                "lr": train_result.get("lr"),
                "elapsed_s": round(elapsed, 2),
            }
            logger.debug(f"Current log item: {log}")
            logs.append(log)

            # Log metrics to tracker node
            if tracker_ready:
                await self._log_metrics(log, step, total)
                await self._call_flush(step)

            if step % self.config.log_every_n_steps == 0:
                # grad_norm/lr are None on non-flush steps (GAS>1: the optimizer
                # flush + LR compute only happen at step % accum_steps == 0).
                # Use safe sentinels so the %.2f/%.2e format doesn't crash on None.
                _gn = log.get("grad_norm")
                _lr = log.get("lr")
                _loss = log.get("loss")
                logger.info(
                    "Step %3d/%d  loss=%.2e  grad=%.2f  lr=%.2e  %.1fs",
                    step,
                    total,
                    _loss if _loss is not None else -1.0,
                    _gn if _gn is not None else -1.0,
                    _lr if _lr is not None else 0.0,
                    log["elapsed_s"],
                )

            # Validation after optimizer update (if enabled)
            if (
                step % acc_steps == 0
                and self.config.validation_every_n_steps > 0
                and (step // acc_steps) % self.config.validation_every_n_steps == 0
                and tracker_ready
            ):
                await self._run_validation(step, total)

            # Hot checkpoint (only on optimizer update steps)
            if (
                step % acc_steps == 0
                and self.config.workdir
                and self.config.hot_checkpoint_every_n_steps > 0
                and (step // acc_steps) % self.config.hot_checkpoint_every_n_steps == 0
            ):
                await self._save_hot_checkpoint(step)

            # Cold checkpoint (HF-compatible weights only)
            if self.config.workdir and self.config.cold_checkpoint_steps:
                cold_steps = self.config.cold_checkpoint_steps
                # NOTE: this used to be a nested ternary whose precedence
                # made list-valued cold_steps ALWAYS False (isinstance(int)
                # and ... evaluated first) — the `step in cold_steps` branch
                # was unreachable. Runs with cold_checkpoint_steps: [] never
                # hit it ([] is falsy). Explicit branching fixes it.
                if isinstance(cold_steps, int):
                    is_cold_step = step % cold_steps == 0
                elif isinstance(cold_steps, list):
                    is_cold_step = step in cold_steps
                else:
                    is_cold_step = False
                if is_cold_step:
                    await self._save_cold_checkpoint(step)

        # Signal tracker that training is complete
        if tracker_ready:
            await self._call_tracker_finish()

        return logs

    async def _run_generation_loop(
        self,
        step: int,
        total: int,
        reps: int,
        gen_steps: int,
        tracker_ready: bool,
        logs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """TRL-style cadence: generate a batch of ``gen_steps`` GRPO groups once,
        then run ``gen_steps`` optimizer steps over contiguous slices of it,
        syncing the rollout weights only once per generation.

        Replaces the per-step micro/nano-batch split as the batch-scaling
        strategy: the per-step slice is small enough to forward in one shot
        (set ``batch_size`` small + ``micro_batch_size`` >= slice), and
        ``gen_steps`` keeps generation throughput up. The token-level engine IS
        correction still corrects the off-policy drift that accumulates across
        the slices (the rollout engine keeps the pre-generation weights until
        the end-of-generation sync).

        Requires ``gradient_accumulation_steps == 1`` (checked in run()).
        """
        assert self._node is not None

        # Generation cache (refilled at the start of each generation window)
        rollout_shards: list | None = None
        reward_shards: list | None = None
        gen_pos = 0  # index of the next slice to train within the current generation

        while step <= total:
            if self._shutting_down:
                logger.warning("Shutdown requested — exiting generation loop at step %d", step)
                break

            # ---- Generation phase (only at the start of a window) ----
            if gen_pos == 0:
                t_gen = time.monotonic()
                all_prompts: list[str] = []
                all_responses: list[str] = []
                all_dataset_ids: list[str] = []
                all_entry_ids: list[int] = []
                system_prompt: str | None = None
                prefix_messages: list[dict] | None = None
                for _ in range(gen_steps):
                    batch_result = await self._call_next_batch(step)
                    raw_prompts = batch_result.prompts
                    raw_responses = batch_result.responses or []
                    b_entry_ids = batch_result.entry_ids
                    if len(set(b_entry_ids)) != len(b_entry_ids):
                        dupes = [e for e in set(b_entry_ids) if b_entry_ids.count(e) > 1]
                        raise RuntimeError(
                            f"Duplicate entry_ids in batch: {sorted(set(dupes))}; "
                            f"cannot build a 1:1 entry_id→response map"
                        )
                    all_prompts.extend(raw_prompts)
                    all_responses.extend(raw_responses)
                    all_dataset_ids.extend(batch_result.dataset_ids)
                    all_entry_ids.extend(b_entry_ids)
                    system_prompt = batch_result.system_prompt
                    prefix_messages = batch_result.prefix_messages
                ref_answers = dict(zip(all_entry_ids, all_responses, strict=True))
                (
                    expanded,
                    exp_dataset_ids,
                    exp_entry_ids,
                    exp_rollout_ids,
                    exp_subrollout_ids,
                ) = self._expand_prompts(all_prompts, reps, all_dataset_ids, all_entry_ids)

                rollouts = await self._call_rollout(
                    expanded,
                    step,
                    system_prompt,
                    exp_dataset_ids,
                    exp_entry_ids,
                    exp_rollout_ids,
                    exp_subrollout_ids,
                    log_table=(
                        self.config.rollout_log_table_every_n_steps > 0
                        and step % self.config.rollout_log_table_every_n_steps == 0
                    ),
                    log_prefix="",
                    prefix_messages=prefix_messages,
                )
                reward_result = await self._call_reward(
                    rollouts,
                    expanded,
                    ref_answers,
                    step,
                    log_table=(
                        self.config.reward_log_table_every_n_steps > 0
                        and step % self.config.reward_log_table_every_n_steps == 0
                    ),
                    log_prefix="",
                )
                # Contiguous split preserves GRPO groups (each group = reps
                # consecutive completions for one entry_id).
                rollout_shards = rollouts.split_into(gen_steps)
                reward_shards = reward_result.split_into(gen_steps)
                logger.info(
                    "Generation window at step %d: %d completions → %d slices (gen took %.1fs)",
                    step,
                    rollouts.batch_size,
                    gen_steps,
                    time.monotonic() - t_gen,
                )

            # ---- Training phase: one optimizer step on the current slice ----
            t0 = time.monotonic()
            train_result: dict[str, Any] = {}
            fwd_result: dict[str, Any] = {}
            assert rollout_shards is not None and reward_shards is not None
            roll_slice = rollout_shards[gen_pos]
            rew_slice = reward_shards[gen_pos]
            end_of_generation = gen_pos == gen_steps - 1
            gen_pos = (gen_pos + 1) % gen_steps

            try:
                fwd_result = await self._call_train_accum(roll_slice, rew_slice, step)
                train_result = await self._call_train_update(step)

                # Weight sync to the rollout engine ONCE per generation.
                if end_of_generation:
                    export_path = await self._call_export_weights(step)
                    await self._call_sync(export_path, step)
                    logger.debug("Weights synced from %s (end of generation)", export_path)
            except RpcError as exc:
                logger.error("Step %d failed: %s", step, exc)
                if self.config.shutdown_on_node_failure:
                    await self._shutdown_cluster(f"RPC failure at step {step}: {exc}")
                    break
                logs.append({"step": step, "error": str(exc)})
                continue

            # The step is complete: record it NOW, so a hot checkpoint taken below
            # saves THIS step. It used to be set at the end of the iteration, so
            # state.json lagged the DCP shards by one and resume re-ran a step
            # that was already applied.
            self._own_step = step

            elapsed = time.monotonic() - t0
            log = {
                "step": step,
                "loss": train_result.get("loss", fwd_result.get("loss")),
                "grad_norm": train_result.get("grad_norm"),
                "lr": train_result.get("lr"),
                "elapsed_s": round(elapsed, 2),
            }
            logs.append(log)
            if tracker_ready:
                await self._log_metrics(log, step, total)
                await self._call_flush(step)
            if step % self.config.log_every_n_steps == 0:
                # None-safe (see the main-loop log line): grad_norm/lr are None
                # on non-flush steps under GAS>1.
                _gn = log.get("grad_norm")
                _lr = log.get("lr")
                _loss = log.get("loss")
                logger.info(
                    "Step %3d/%d  loss=%.2e  grad=%.2f  lr=%.2e  %.1fs  %s",
                    step,
                    total,
                    _loss if _loss is not None else -1.0,
                    _gn if _gn is not None else -1.0,
                    _lr if _lr is not None else 0.0,
                    log["elapsed_s"],
                    "[sync]" if end_of_generation else "",
                )

            if (
                self.config.validation_every_n_steps > 0
                and step % self.config.validation_every_n_steps == 0
                and tracker_ready
            ):
                await self._run_validation(step, total)

            if (
                self.config.workdir
                and self.config.hot_checkpoint_every_n_steps > 0
                and step % self.config.hot_checkpoint_every_n_steps == 0
            ):
                await self._save_hot_checkpoint(step)

            if self.config.workdir and self.config.cold_checkpoint_steps:
                cold_steps = self.config.cold_checkpoint_steps
                # explicit branching — see the note at the train-loop copy
                # (the old nested ternary made list cold_steps unreachable)
                if isinstance(cold_steps, int):
                    is_cold_step = step % cold_steps == 0
                elif isinstance(cold_steps, list):
                    is_cold_step = step in cold_steps
                else:
                    is_cold_step = False
                if is_cold_step:
                    await self._save_cold_checkpoint(step)

            step += 1

        if tracker_ready:
            await self._call_tracker_finish()
        return logs

    # ------------------------------------------------------------------
    # Memory management (sleep/wake)
    # ------------------------------------------------------------------

    async def _wake_rollout(self) -> None:
        """Wake the rollout node (restore GPU memory)."""
        assert self._node is not None
        try:
            await self._node.call("rollout", "wake", _to_dict(NoInputRequest()))
        except RpcError as exc:
            logger.warning("Rollout wake failed (continuing): %s", exc)

    async def _sleep_rollout(self) -> None:
        """Sleep the rollout node (release GPU memory)."""
        assert self._node is not None
        try:
            await self._node.call("rollout", "sleep", _to_dict(NoInputRequest()))
        except RpcError as exc:
            logger.warning("Rollout sleep failed (continuing): %s", exc)

    async def _wake_model(self) -> None:
        """Wake the model node (restore GPU memory)."""
        assert self._node is not None
        try:
            await self._node.call("model", "wake", _to_dict(NoInputRequest()))
        except RpcError as exc:
            logger.warning("Model wake failed (continuing): %s", exc)

    async def _sleep_model(self) -> None:
        """Sleep the model node (release GPU memory)."""
        assert self._node is not None
        try:
            await self._node.call("model", "sleep", _to_dict(NoInputRequest()))
        except RpcError as exc:
            logger.warning("Model sleep failed (continuing): %s", exc)

    # ------------------------------------------------------------------
    # Tracker RPC helpers
    # ------------------------------------------------------------------

    async def _log_metrics(self, log: dict[str, Any], step: int, total: int) -> None:
        """Send metrics to the tracker node (accumulator). Silently skipped if unreachable."""
        assert self._node is not None
        # Group all orchestrator training metrics under the ``train/`` W&B supergroup.
        metrics: dict[str, float] = {}
        for key in ("loss", "grad_norm", "lr"):
            value = log.get(key)
            if value is not None:
                metrics[f"train/{key}"] = float(value)
        if "elapsed_s" in log:
            metrics["train/elapsed_s"] = float(log["elapsed_s"])
        metrics["train/step_fraction"] = step / total
        metrics["train/current_step"] = step

        batch = _to_dict(
            MetricsBatch(
                metrics=metrics,
                node_id="orchestrator",
            )
        )
        try:
            await self._node.call("tracker", "log_metrics", batch)
        except RpcError as exc:
            logger.debug("Tracker log_metrics failed: %s", exc)

    async def _call_flush(self, step: int) -> None:
        """Flush accumulated tracker metrics with the current global step."""
        assert self._node is not None
        # Nodes ship their metrics fire-and-forget (call_bg_no_return, drained by
        # a 10 ms poller) right before replying; give those a moment to land so
        # they are flushed at THIS step instead of the next one.
        await asyncio.sleep(0.05)
        try:
            await self._node.call(
                "tracker",
                "flush",
                _to_dict(FlushMetrics(step=step)),
            )
        except RpcError as exc:
            logger.debug("Tracker flush failed: %s", exc)

    async def _call_tracker_finish(self) -> None:
        """Tell the tracker to flush and close. Silently skipped if unreachable."""
        assert self._node is not None
        try:
            await self._node.call("tracker", "finish")
        except RpcError as exc:
            logger.debug("Tracker finish failed: %s", exc)

    # ------------------------------------------------------------------
    # RPC helpers
    # ------------------------------------------------------------------

    async def _coordinate_bringup_sequence(
        self,
    ) -> None:
        if self.config.memory_management_enabled:
            # Competitive colocation: stagger bringup so rollout and model
            # take turns on the shared GPU during initialization.
            logger.info("Waiting for the rollout node to start first...")
            await self._wait_for_nodes(["rollout"], timeout=300.0)
            logger.info("Waiting for the model node to start...")
            await self._wait_for_nodes(["model"], timeout=600.0)
            logger.info("Sleeping rollout, initializing model on GPU...")
            await self._release_model_updater_init()
        else:
            # Spacious / disjoint colocation: no staggering.
            # Model still needs an explicit RPC trigger to initialize.
            logger.info("Waiting for model node to appear...")
            await self._wait_for_nodes(["model"], timeout=300.0)
            await self._node.call("model", "initialize_model", {})
            logger.info("Model node initializing; waiting for all nodes...")

        logger.info("Waiting for all nodes to come online (final stage).")
        await self._wait_for_nodes(["dataloader", "rollout", "reward", "model"], timeout=300.0)
        logger.info("Bringup complete.")

    async def _release_model_updater_init(self):
        assert self._node is not None
        # put rollout to sleep
        logger.debug("Sleeping the rollout.")
        await self._sleep_rollout()

        # allow the model update node to use gpus
        logger.debug("Initializing the model updater.")
        data = {}
        data = await self._node.call("model", "initialize_model", data)

        # sleep the model update node
        logger.debug("Sleeping the model updater before the loop.")
        await asyncio.sleep(1.0)
        await self._sleep_model()

    async def _call_rollout(
        self,
        prompts: list[str],
        step: int,
        system_prompt: str | None = None,
        dataset_ids: list[str] | None = None,
        entry_ids: list[int] | None = None,
        rollout_ids: list[int] | None = None,
        subrollout_ids: list[int] | None = None,
        log_table: bool = False,
        log_prefix: str = "",
        prefix_messages: list[dict] | None = None,
    ) -> RolloutResult:
        """Call rollout node → generate completions + token_ids."""
        assert self._node is not None
        batch = _to_dict(
            PromptBatch(
                prompts=prompts,
                system_prompt=system_prompt,
                prefix_messages=prefix_messages,
                step=step,
                log_table=log_table,
                log_prefix=log_prefix,
                dataset_ids=dataset_ids or [],
                entry_ids=entry_ids or [],
                rollout_ids=rollout_ids or [],
                subrollout_ids=subrollout_ids or [],
                temperature=(
                    self.config.rollout.temperature if self.config.rollout.temperature > 0 else None
                ),
                top_p=(self.config.rollout.top_p if self.config.rollout.top_p < 1.0 else None),
                return_logprobs=True,
            )
        )
        data = await self._node.call("rollout", "generate", batch)
        return _from_dict(RolloutResult, data)

    async def _call_reward(
        self,
        rollouts: RolloutResult,
        expanded_prompts: list[str],
        ref_answers: dict[int, str],
        step: int,
        log_table: bool = False,
        log_prefix: str = "",
    ) -> RewardResult:
        """Call reward node → compute rewards and GRPO advantages.

        Passes both text (for text-based rewards) and input_ids (for token-based
        rewards).  Also passes per-token logprobs from the rollout for potential
        learned reward models.

        ``ref_answers`` maps entry_id → ground-truth answer.  It is expanded
        to match ``rollouts.batch_size`` using the rollout's ``entry_ids``.

        Returns the full RewardResult containing both raw rewards and advantages.
        """
        assert self._node is not None

        # Build concatenated input_ids: prompt_tokens + completion_tokens
        input_ids: list[list[int]] = [
            pt + ct for pt, ct in zip(rollouts.prompt_token_ids, rollouts.token_ids, strict=True)
        ]

        # Expand reference answers to match rollout batch size.
        # Each sample gets the answer for its entry_id.
        responses: list[str] = [
            ref_answers.get(rollouts.entry_ids[i], "") for i in range(rollouts.batch_size)
        ]

        req = _to_dict(
            RewardRequest(
                prompts=expanded_prompts,
                completions=rollouts.completions,
                responses=responses,
                input_ids=input_ids,
                # Prompt/completion token split for token-based reward variants
                # (additive; the base reward node ignores these).
                prompt_token_ids=rollouts.prompt_token_ids,
                completion_token_ids=rollouts.token_ids,
                logprobs=rollouts.logprobs,
                step=step,
                log_table=log_table,
                log_prefix=log_prefix,
                dataset_ids=rollouts.dataset_ids,
                entry_ids=rollouts.entry_ids,
                rollout_ids=rollouts.rollout_ids,
                subrollout_ids=rollouts.subrollout_ids,
                finish_reasons=rollouts.finish_reasons,
            )
        )
        data = await self._node.call("reward", "compute_rewards", req)
        return _from_dict(RewardResult, data)

    async def _call_train_accum(
        self,
        rollouts: RolloutResult,
        reward_result: RewardResult,
        step: int,
    ) -> dict[str, Any]:
        """Call model node → forward + backward (accumulates gradients).

        Passes advantages (GRPO z-score normalized) and old logprobs for
        PPO-style importance sampling in the loss computation.
        """
        assert self._node is not None

        input_ids: list[list[int]] = [
            pt + ct for pt, ct in zip(rollouts.prompt_token_ids, rollouts.token_ids, strict=True)
        ]
        label_ids: list[list[int]] = [
            [-100] * len(pt) + ct
            for pt, ct in zip(rollouts.prompt_token_ids, rollouts.token_ids, strict=True)
        ]

        req = _to_dict(
            TrainStepRequest(
                input_ids=input_ids,
                label_ids=label_ids,
                rewards=reward_result.rewards,
                advantages=reward_result.advantages,
                sampling_logprobs=rollouts.logprobs,
                completions=rollouts.completions,
                dataset_ids=rollouts.dataset_ids,
                entry_ids=rollouts.entry_ids,
                rollout_ids=rollouts.rollout_ids,
                subrollout_ids=rollouts.subrollout_ids,
                # GAS scale — orchestrator-owned. The model divides the
                # per-call loss by this before backward so accumulated
                # gradients form a correct mean over `accum_steps` calls.
                # 1 = no accumulation (legacy behaviour).
                accum_steps=self.config.model.gradient_accumulation_steps,
            )
        )

        # --- diagnostic capture (env-gated, one-shot) ---------------------
        # When WW_CAPTURE_DIR is set, save the FULL inputs to the gradient
        # computation on the FIRST train-accum call, so the model_update
        # forward/IS-ratio path can be probed in isolation (no cluster).
        # Captures: input_ids, label_ids, advantages, vLLM sampling_logprobs,
        # rewards, completions, prompt/completion token_ids, finish_reasons,
        # + metadata (step, temperature, model/rollout config paths).
        # If WW_CAPTURE_EXIT=1, stop the run cleanly after capturing.
        cap_dir = os.environ.get("WW_CAPTURE_DIR")
        if cap_dir and not getattr(self, "_capture_done", False):
            self._capture_done = True
            import pickle
            try:
                os.makedirs(cap_dir, exist_ok=True)
                cap = {
                    "step": step,
                    "accum_steps": self.config.model.gradient_accumulation_steps,
                    "temperature": getattr(self.config.model, "temperature", None),
                    "engine_is_correction": getattr(self.config.model, "engine_is_correction", None),
                    "engine_is_mode": getattr(self.config.model, "engine_is_mode", None),
                    "engine_is_cap": getattr(self.config.model, "engine_is_cap", None),
                    "loss_reduction": getattr(self.config.model, "loss_reduction", None),
                    "input_ids": input_ids,
                    "label_ids": label_ids,
                    "rewards": reward_result.rewards,
                    "advantages": reward_result.advantages,
                    "sampling_logprobs": rollouts.logprobs,
                    "completions": rollouts.completions,
                    "prompt_token_ids": rollouts.prompt_token_ids,
                    "token_ids": rollouts.token_ids,
                    "finish_reasons": rollouts.finish_reasons,
                }
                cap_path = os.path.join(cap_dir, f"capture_step{step}.pkl")
                with open(cap_path, "wb") as f:
                    pickle.dump(cap, f)
                logger.info(
                    "WW_CAPTURE: saved %d completions to %s "
                    "(temp=%s, accum_steps=%s)",
                    len(input_ids), cap_path,
                    cap["temperature"], cap["accum_steps"],
                )
            except Exception:
                logger.exception("WW_CAPTURE: failed to save capture")
            if os.environ.get("WW_CAPTURE_EXIT"):
                logger.info("WW_CAPTURE: WW_CAPTURE_EXIT set — stopping run after capture.")
                self._shutting_down = True
                return {"loss": 0.0, "captured": True}
        # --- end diagnostic capture ---------------------------------------

        fwd_data = await self._node.call("model", "forward_accum_grad", req)
        return _to_dict(_from_dict(TrainStepResult, fwd_data))

    async def _call_train_update(self, step: int) -> dict[str, Any]:
        """Call model node → optimizer step (flushes accumulated gradients)."""
        assert self._node is not None

        lr = self._compute_lr(step)
        update_req = _to_dict(ModelUpdateRequest(lr=lr))
        upd_data = await self._node.call("model", "model_update_step", update_req)
        return _to_dict(_from_dict(TrainStepResult, upd_data))

    async def _call_export_weights(self, step: int) -> str:
        """Call model node → export consolidated HF weights.

        Returns the path to the exported weights directory.
        """
        import tempfile

        assert self._node is not None
        export_dir = tempfile.mkdtemp(
            prefix=f"weights-step{step}-",
            dir=self.config.model.checkpoint_dir,
        )

        from whirlwind.messages import ExportWeightsRequest

        export_req = _to_dict(ExportWeightsRequest(path=export_dir))
        await self._node.call("model", "export_weights", export_req)
        return export_dir

    async def _call_next_batch(self, step: int = 0, dataset: str = "train") -> PromptBatch:
        """Call dataloader node → fetch next batch of prompts."""
        from whirlwind.messages import BatchRequest, BatchResult

        assert self._node is not None
        req = _to_dict(BatchRequest(dataset_name=dataset, step=step))
        raw = await self._node.call("dataloader", "next_batch", req)
        if raw is None:
            raise RpcError("Dataloader returned None")
        batch_result = _from_dict(BatchResult, raw)
        if not batch_result.ok:
            raise RpcError(f"Dataloader error: {batch_result.error}")
        pb = batch_result.to_prompt_batch()
        if not getattr(self, "_logged_prefix_msgs", False):
            self._logged_prefix_msgs = True
            logger.info(
                "First batch at the orchestrator: system prompt %d chars, few-shot turns: %s",
                len(pb.system_prompt or ""),
                (f"{len(pb.prefix_messages)} turns" if pb.prefix_messages else "None"),
            )
        return pb

    async def _run_validation(self, step: int, total: int) -> None:
        """Run a validation step: rollout + reward (no training).

        Fetches from the "val" dataset, runs generation and reward,
        logs metrics with a "val." prefix and tables with a "val_" prefix,
        then flushes the tracker so validation metrics appear at this step.
        """
        assert self._node is not None
        logger.info("Running validation (step %d)...", step)

        reps = self.config.rollout_per_prompt

        # Memory management: ensure rollout is awake
        if self.config.memory_management_enabled:
            await self._wake_rollout()

        # Fetch a batch from the validation dataset
        val_batch = await self._call_next_batch(step, dataset="val")
        raw_prompts = val_batch.prompts
        batch_entry_ids = val_batch.entry_ids

        # Expand prompts
        (
            expanded,
            exp_dataset_ids,
            exp_entry_ids,
            exp_rollout_ids,
            exp_subrollout_ids,
        ) = self._expand_prompts(raw_prompts, reps, val_batch.dataset_ids, batch_entry_ids)

        # Phase 1: Rollout (no gradients needed)
        # Val tables serialize EVERY completion text — at 8k-token budgets
        # that's ~30-60MB through wandb's (pathologically slow) serializer,
        # which stalled the tracker RPC long enough to trip every heartbeat
        # watchdog and wedge the cluster (the 8196 runs died here on val@0).
        # validation_log_table: false disables ONLY the logging.
        _val_table = bool(getattr(self.config, "validation_log_table", True))
        val_rollouts = await self._call_rollout(
            expanded,
            step,
            val_batch.system_prompt,
            exp_dataset_ids,
            exp_entry_ids,
            exp_rollout_ids,
            exp_subrollout_ids,
            log_table=_val_table,
            log_prefix="val.",
            prefix_messages=val_batch.prefix_messages,
        )

        # Phase 2: Reward. (Same duplicate-entry_id guard as the training loop.)
        if len(set(batch_entry_ids)) != len(batch_entry_ids):
            dupes = [e for e in set(batch_entry_ids) if batch_entry_ids.count(e) > 1]
            raise RuntimeError(
                f"Duplicate entry_ids in val batch: {sorted(set(dupes))}; "
                f"cannot build a 1:1 entry_id→response map"
            )
        ref_answers = dict(zip(batch_entry_ids, val_batch.responses or [], strict=True))
        val_reward = await self._call_reward(
            val_rollouts,
            expanded,
            ref_answers,
            step,
            log_table=_val_table,
            log_prefix="val.",
        )

        # Sleep rollout if in competitive mode
        if self.config.memory_management_enabled:
            await self._sleep_rollout()

        # Log validation metrics under the ``val/`` W&B supergroup (consistent
        # slash grouping with training; previously used a "val." dot prefix).
        import statistics

        val_log: dict[str, Any] = {
            "val/reward_mean": statistics.mean(val_reward.rewards),
            "val/reward_std": (
                statistics.stdev(val_reward.rewards) if len(val_reward.rewards) > 1 else 0.0
            ),
            "val/reward_min": min(val_reward.rewards),
            "val/reward_max": max(val_reward.rewards),
            "val/adv_mean": statistics.mean(val_reward.advantages),
            "val/batch_size": len(val_reward.rewards),
            "step": step,
            "current_step": step,
            "step_fraction": step / total,
        }

        val_metrics_batch = _to_dict(
            MetricsBatch(
                metrics=val_log,
                node_id="orchestrator",
            )
        )
        try:
            await self._node.call("tracker", "log_metrics", val_metrics_batch)
        except RpcError as exc:
            logger.debug("Validation log_metrics failed: %s", exc)

        await self._call_flush(step)
        logger.info(
            "Validation complete (step %d): reward_mean=%.4f",
            step,
            val_log["val/reward_mean"],
        )

    async def _call_sync(self, checkpoint_path: str, step: int) -> None:
        """Call rollout node → update weights from checkpoint."""
        assert self._node is not None
        req = _to_dict(WeightSyncRequest(checkpoint_path=checkpoint_path, step=step))
        await self._node.call("rollout", "update_weights", req)
        # We created the export dir (mkdtemp in _call_export_weights): delete it
        # here, after EVERY consumer (pool backends included) has loaded it. The
        # backends used to delete it themselves and raced each other.
        # LoRA adapter dirs are kept: the engines may re-read an evicted adapter
        # and already manage their own previous-adapter cleanup.
        if not getattr(self.config.model, "peft_lora", False) and os.path.isdir(checkpoint_path):
            shutil.rmtree(checkpoint_path, ignore_errors=True)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _get_own_state(self, step: int = 0) -> dict[str, Any]:
        """Return the orchestrator's own state (config + training progress).

        Args:
            step: current training step (0 if not in training loop).
        """
        return {
            "config": asdict(self.config),
            "state": {
                "seed": self.config.seed,
                "step": self._own_step if step == 0 else step,
            },
        }

    async def get_node_states(
        self,
        keys: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get state from all discovered nodes.

        Returns a dict mapping node_id → {key: value} for requested keys.
        If keys is None, returns all keys.

        The orchestrator's own state is fetched directly (not via RPC).
        """
        assert self._node is not None
        keys = keys or []
        result: dict[str, dict[str, Any]] = {}

        # Own state — direct call, no RPC
        own = self._get_own_state()
        if keys:
            own = {k: v for k, v in own.items() if k in keys}
        result[self.config.orchestrator_node_id] = own

        # Other nodes — RPC
        for node_id in self._node.topology:
            if node_id == self.config.orchestrator_node_id:
                continue
            try:
                raw = await self._node.call(
                    node_id,
                    "get_state",
                    {"keys": keys},
                )
                if raw is None:
                    continue
                resp = _from_dict(StateResult, raw)
                if resp.ok:
                    result[node_id] = resp.data
                else:
                    logger.warning("Node %s get_state error: %s", node_id, resp.error)
            except RpcError:
                # Node doesn't expose get_state — skip silently
                pass
        return result

    async def set_node_states(
        self,
        states: dict[str, dict[str, Any]],
    ) -> dict[str, bool]:
        """Set state on multiple nodes.

        Returns a dict mapping node_id → True/False (success/failure).

        The orchestrator's own state is applied directly (not via RPC).
        """
        assert self._node is not None
        result: dict[str, bool] = {}
        own_id = self.config.orchestrator_node_id

        for node_id, data in states.items():
            if node_id == own_id:
                # Direct apply — no RPC to self
                if "state" in data and isinstance(data["state"], dict):
                    s = data["state"]
                    if "step" in s:
                        self._own_step = int(s["step"])
                    if "seed" in s:
                        self.config.seed = int(s["seed"])
                result[node_id] = True
                continue

            # Other nodes — RPC
            keys = list(data.keys())
            try:
                raw = await self._node.call(
                    node_id,
                    "set_state",
                    {"keys": keys, "data": data},
                )
                if raw is not None:
                    resp = _from_dict(StateResult, raw)
                    result[node_id] = resp.ok
                else:
                    result[node_id] = False
            except RpcError:
                result[node_id] = False
                logger.warning("Node %s set_state failed", node_id)
        return result

    async def _maybe_save_config(self) -> None:
        """Auto-save config snapshot to workdir if no config exists yet.

        Called at the start of the training loop. Writes only config.json
        (not state) to the workdir root. If config.json already exists,
        this is a no-op.
        """
        workdir = self.config.workdir
        if not workdir:
            return

        config_file = os.path.join(workdir, "config.json")
        if os.path.isfile(config_file):
            logger.debug("Config snapshot already exists at %s; skipping", config_file)
            return

        os.makedirs(workdir, exist_ok=True)

        # Collect configs from all nodes
        node_configs = await self.get_node_states(keys=["config"])
        configs: dict[str, Any] = {
            "orchestrator": self._get_own_state().get("config", {}),
        }
        for node_id, data in node_configs.items():
            if "config" in data:
                configs[node_id] = data["config"]

        with open(config_file, "w") as f:
            json.dump(configs, f, indent=2)

        logger.info(
            "Config snapshot saved to %s (nodes: %s)",
            config_file,
            sorted(configs.keys()),
        )

    async def _save_hot_checkpoint(self, step: int) -> None:
        """Save a hot checkpoint: state.json + DCP model + optimizer shards.

        Overwrites the previous hot checkpoint in workdir/hot_cp/.
        """
        workdir = self.config.workdir
        if not workdir:
            return

        hot_dir = os.path.join(workdir, "hot_cp")
        # hot_cp/ is created by the model node's atomic swap, never pre-made here:
        # an empty dir would make run()'s auto-resume check see a "checkpoint".

        # Snapshot node states now (they belong to THIS step) but write state.json
        # only once the DCP shards have landed, so the two never disagree.
        node_states = await self.get_node_states(keys=["state"])
        states: dict[str, Any] = {}
        for node_id, data in node_states.items():
            if "state" in data:
                states[f"{node_id}"] = data["state"]
        state_file = os.path.join(workdir, "state.json")

        # Call model node to save DCP shards (atomic swap into hot_cp/)
        assert self._node is not None
        try:
            result = await self._node.call(
                "model",
                "save_hot_checkpoint",
                _to_dict(CheckpointRequest(path=hot_dir, step=step)),
            )
            if result is not None:
                resp = _from_dict(CheckpointResult, result)
                if resp.ok:
                    with open(state_file, "w") as f:
                        json.dump(states, f, indent=2)
                    logger.info("Hot checkpoint saved at step %d → %s", step, hot_dir)
                else:
                    logger.warning("Hot checkpoint failed: %s", resp.error)
        except RpcError as exc:
            logger.warning("Hot checkpoint RPC failed: %s", exc)

    async def _save_cold_checkpoint(self, step: int) -> None:
        """Save a cold checkpoint: HF-compatible weights only."""
        workdir = self.config.workdir
        if not workdir:
            return

        cold_dir = os.path.join(workdir, f"cold_cp_step_{step}")

        # Call model node to save its cold checkpoint
        assert self._node is not None
        try:
            result = await self._node.call(
                "model",
                "save_cold_checkpoint",
                _to_dict(CheckpointRequest(path=cold_dir, step=step)),
            )
            if result is not None:
                resp = _from_dict(CheckpointResult, result)
                if resp.ok:
                    logger.info("Cold checkpoint saved at step %d → %s", step, cold_dir)
                else:
                    logger.warning("Cold checkpoint failed: %s", resp.error)
        except RpcError as exc:
            logger.warning("Cold checkpoint RPC failed: %s", exc)

    async def load_hot_checkpoint(
        self,
        path: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load a hot checkpoint from *path* and restore node states + model weights.

        Expects *path* to contain:
          path/config.json  — all node configs
          path/state.json   — all node states
          path/hot_cp/      — DCP shards (model + optimizer)

        Restores orchestrator step counter and node states,
        loads DCP model + optimizer shards, then exports HF weights
        and syncs the rollout node for generation.

        Returns (configs, states) dicts for the caller to inspect.
        """
        config_path = os.path.join(path, "config.json")
        state_path = os.path.join(path, "state.json")

        if not os.path.isfile(config_path) or not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"Checkpoint incomplete at {path}: need config.json and state.json"
            )

        with open(config_path) as f:
            configs: dict[str, Any] = json.load(f)
        with open(state_path) as f:
            states: dict[str, Any] = json.load(f)

        # Restore state on each node + the orchestrator itself
        node_states: dict[str, dict[str, Any]] = {}
        for key, value in states.items():
            if key == "orchestrator":
                # Restore orchestrator's own state directly
                if isinstance(value, dict) and "step" in value:
                    self._own_step = int(value["step"])
                continue
            node_states[key] = {"state": value}

        # No resume_step offset for the tracker: the step counter above is
        # restored as an ABSOLUTE step and the tracker logs req.step directly.
        # Injecting the offset made W&B jump to 2N-1 after a resume at N.

        if node_states:
            await self.set_node_states(node_states)

        # Load model DCP shards from hot_cp subdirectory
        hot_cp_path = os.path.join(path, "hot_cp")
        if os.path.isdir(hot_cp_path):
            assert self._node is not None
            try:
                result = await self._node.call(
                    "model",
                    "load_hot_checkpoint",
                    _to_dict(CheckpointRequest(path=hot_cp_path, step=self._own_step)),
                )
                if result is not None:
                    resp = _from_dict(CheckpointResult, result)
                    if resp.ok:
                        logger.info(
                            "Model weights loaded from %s (step=%d)",
                            hot_cp_path,
                            self._own_step,
                        )
                        # Sync rollout to match restored model weights
                        try:
                            await self._wake_rollout()
                            export_path = await self._call_export_weights(self._own_step)
                            await self._call_sync(export_path, self._own_step)
                            logger.info(
                                "Rollout weights synced to match restored model (step=%d)",
                                self._own_step,
                            )
                            await self._sleep_rollout()
                        except Exception:
                            logger.exception(
                                "Failed to sync rollout after checkpoint load — first step will generate with stale weights"
                            )
                    else:
                        logger.warning("Model weight load failed: %s", resp.error)
                else:
                    logger.warning("Model weight load returned None")
            except RpcError as exc:
                logger.warning("Model weight load RPC failed: %s", exc)
        else:
            logger.info(
                "No hot_cp found at %s; skipping model weight restore",
                hot_cp_path,
            )

        logger.info("Checkpoint loaded from %s (nodes: %s)", path, sorted(configs.keys()))
        return configs, states

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _expand_prompts(
        self,
        prompts: list[str],
        reps: int,
        dataset_ids: list[str] | None = None,
        entry_ids: list[int] | None = None,
    ) -> tuple[
        list[str],
        list[str],
        list[int],
        list[int],
        list[int],
    ]:
        """Repeat each prompt `reps` times for diverse sampling groups.

        Returns (expanded_prompts, dataset_ids, entry_ids, rollout_ids, subrollout_ids).
        """
        if dataset_ids is None:
            dataset_ids = [""] * len(prompts)
        if entry_ids is None:
            entry_ids = list(range(len(prompts)))
        expanded: list[str] = []
        expanded_dataset_ids: list[str] = []
        expanded_entry_ids: list[int] = []
        expanded_rollout_ids: list[int] = []
        expanded_subrollout_ids: list[int] = []
        for i, p in enumerate(prompts):
            for j in range(reps):
                expanded.append(p)
                expanded_dataset_ids.append(dataset_ids[i] if i < len(dataset_ids) else "")
                expanded_entry_ids.append(entry_ids[i] if i < len(entry_ids) else i)
                expanded_rollout_ids.append(j)  # which replica (0..reps-1)
                expanded_subrollout_ids.append(0)  # single-fragment case
        return (
            expanded,
            expanded_dataset_ids,
            expanded_entry_ids,
            expanded_rollout_ids,
            expanded_subrollout_ids,
        )

    async def _wait_for_nodes(self, node_ids: list[str], timeout: float = 300.0) -> None:
        """Block until all expected nodes appear in topology."""
        assert self._node is not None
        deadline = time.monotonic() + timeout
        topo: dict[str, dict] = {}
        while time.monotonic() < deadline:
            topo = self._node.topology
            logger.debug(f"Topology contains the following: {topo.keys()}")
            logger.debug(f"Membership list: {[nid in topo for nid in node_ids]}")
            if all(nid in topo for nid in node_ids):
                logger.info("All nodes discovered: %s", sorted(topo.keys()))
                return
            missing = [n for n in node_ids if n not in topo]
            logger.debug("Waiting for nodes: %s", missing)
            await asyncio.sleep(5.0)
        raise TimeoutError(
            f"Not all nodes discovered within {timeout}s. "
            f"Found: {sorted(topo.keys())}, Expected: {node_ids}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whirlwind Training Orchestrator")
    parser.add_argument(
        "--config",
        default="configs/training/grpo.yaml",
        help="Path to training YAML config",
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
    config = TrainingConfig.from_yaml(args.config)
    print(
        f"Starting training orchestrator: steps={config.total_steps}, "
        f"rollout_per={config.rollout_per_prompt}"
    )

    orch = TrainingOrchestrator(config)
    if args.load_hot_checkpoint:
        orch._load_hot_checkpoint_path = args.load_hot_checkpoint

    # Capture orchestrator (env-gated): when WW_CAPTURE_EVERY is set, wrap the
    # run in CaptureOrchestrator which snapshots weights + messages every N
    # steps for offline replay. See whirlwind/training/training_grpo_capture.py.
    _cap_every = int(os.environ.get("WW_CAPTURE_EVERY", "0"))
    if _cap_every > 0:
        from whirlwind.training.training_grpo_capture import CaptureOrchestrator
        cap_dir = os.environ.get("WW_CAPTURE_DIR")
        orch = CaptureOrchestrator(config, capture_every=_cap_every, capture_dir=cap_dir)
        if args.load_hot_checkpoint:
            orch._load_hot_checkpoint_path = args.load_hot_checkpoint
        print(f"CaptureOrchestrator active: every {_cap_every} steps -> {orch.capture_dir}")

    async def _main() -> None:
        await orch.start()
        logs = await orch.run()
        # Set when the loop was cut short (a required node lost, a cluster
        # shutdown from elsewhere); checked after teardown for the exit code.
        failure = orch.shutdown_reason

        # Print summary
        print("\n=== Training Summary ===")
        for log in logs[-5:]:
            print(
                f"  Step {log['step']}: loss={log.get('loss', 'N/A')}, "
                f"grad={log.get('grad_norm', 'N/A')}, "
                f"time={log.get('elapsed_s', 'N/A')}s"
            )

        # Normal completion must tell the SERVICE nodes (rollout / reward /
        # dataloader / tracker) to exit: they are long-running processes that
        # only stop on a cluster-shutdown broadcast or a signal. Failure
        # paths already broadcast via _shutdown_cluster; without this call a
        # cleanly finished run leaves every node (and the sglang engine
        # workers) alive indefinitely.
        try:
            await orch.shutdown_cluster("training complete")
        except Exception as exc:  # broadcast is best-effort
            logger.warning("Completion shutdown broadcast failed: %s", exc)

        await orch.stop()
        if failure:
            print(f"Training stopped early: {failure}")
            raise SystemExit(1)
        print("Training complete.")

    asyncio.run(_main())
