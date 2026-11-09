"""Capture orchestrator: run training while snapshotting weights + messages
every N steps, for offline replay through a different model node.

A thin subclass of :class:`TrainingOrchestrator` that, at every ``capture_every``
training step, persists:

- ``weights_step{step}/``  — the HF-consolidated weights at the START of the
  step (== the weights the rollout used to generate this step's completions,
  == the weights the model node's forward_accum_grad runs on). Exported via
  the model node's ``export_weights`` endpoint to a PERSISTENT dir (not the
  throwaway tempdir the normal weight-sync uses, which the rollout deletes).
- ``round{step}.pkl``       — the full inputs to the model node's
  ``forward_accum_grad`` (the TrainStepRequest dict, + rollouts/reward/metadata)
  so an offline replay can feed the exact same batch.
- ``round{step}_hf_metrics.json`` — the HF (or whichever model node is in use)
  reported ``loss`` (from forward_accum_grad) and ``grad_norm`` (from
  model_update_step) for this step — the metrics the replay must match.

The capture is driven entirely from the orchestrator (CPU side); the model node
is unmodified. The extra weight export (one consolidation + disk write) runs
only every ``capture_every`` steps, so overhead is negligible.

Activated by passing ``capture_every > 0`` (the driver wires this via env vars
read in ``training_grpo._main``).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any

from whirlwind.messages import (
    ExportWeightsRequest,
    RewardResult,
    RolloutResult,
    TrainStepRequest,
    TrainStepResult,
)
from whirlwind.training.training_grpo import TrainingOrchestrator, _to_dict, _from_dict

logger = logging.getLogger(__name__)


class CaptureOrchestrator(TrainingOrchestrator):
    """TrainingOrchestrator that snapshots weights + messages every N steps."""

    def __init__(
        self,
        config: Any,
        capture_every: int = 16,
        capture_dir: str | None = None,
    ) -> None:
        super().__init__(config)
        self.capture_every = max(int(capture_every), 0)
        if capture_dir is None:
            rid = os.environ.get("WW_RUN_ID", "nocapture")
            capture_dir = os.path.join(getattr(config, "workdir", None) or ".", f"captures-{rid}")
        self.capture_dir = capture_dir
        if self.capture_every > 0:
            os.makedirs(self.capture_dir, exist_ok=True)
            logger.info(
                "CaptureOrchestrator: capture every %d steps -> %s",
                self.capture_every, self.capture_dir,
            )
        # path to the per-step metrics json currently being filled (set in
        # _call_train_accum, appended in _call_train_update).
        self._cap_metrics_path: str | None = None

    def _is_capture_step(self, step: int) -> bool:
        return self.capture_every > 0 and step > 0 and step % self.capture_every == 0

    async def _export_weights_to(self, path: str) -> None:
        """Export the model's current weights to a PERSISTENT path (all ranks).

        Uses the model node's ``export_weights`` endpoint (same one the normal
        weight-sync uses), but to our own dir so the rollout's post-sync
        cleanup cannot delete it.
        """
        assert self._node is not None
        os.makedirs(path, exist_ok=True)
        req = _to_dict(ExportWeightsRequest(path=path))
        await self._node.call("model", "export_weights", req)
        logger.info("CAPTURE: exported start-of-step weights -> %s", path)

    async def _call_train_accum(
        self,
        rollouts: RolloutResult,
        reward_result: RewardResult,
        step: int,
    ) -> dict[str, Any]:
        cap = self._is_capture_step(step)
        if cap:
            # 1. Export W_s (start-of-step weights) BEFORE the forward. The model
            #    is awake (the loop woke it just before train_accum); export only
            #    reads params, so it is safe ahead of backward.
            wpath = os.path.join(self.capture_dir, f"weights_step{step}")
            try:
                await self._export_weights_to(wpath)
            except Exception:
                logger.exception("CAPTURE: weight export failed at step %d", step)

        # Build the request exactly as the parent does (so we capture the real
        # inputs), then dispatch. We reconstruct here to access the dict for
        # dumping without duplicating logic.
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
                accum_steps=self.config.model.gradient_accumulation_steps,
            )
        )

        if cap:
            # 2. Dump the full inputs (the TrainStepRequest dict the model node
            #    receives + the rollouts/reward + config metadata for replay).
            try:
                cap_payload = {
                    "step": step,
                    "accum_steps": self.config.model.gradient_accumulation_steps,
                    "lr": self._compute_lr(step),
                    "temperature": getattr(self.config.model, "temperature", None),
                    "engine_is_correction": getattr(self.config.model, "engine_is_correction", None),
                    "engine_is_mode": getattr(self.config.model, "engine_is_mode", None),
                    "engine_is_cap": getattr(self.config.model, "engine_is_cap", None),
                    "loss_reduction": getattr(self.config.model, "loss_reduction", None),
                    "max_grad_norm": getattr(self.config.model, "max_grad_norm", None),
                    "train_req": req,  # already a dict (_to_dict)
                    "rewards": reward_result.rewards,
                    "advantages": reward_result.advantages,
                    "sampling_logprobs": rollouts.logprobs,
                    "completions": rollouts.completions,
                    "prompt_token_ids": rollouts.prompt_token_ids,
                    "token_ids": rollouts.token_ids,
                    "finish_reasons": rollouts.finish_reasons,
                }
                cap_path = os.path.join(self.capture_dir, f"round{step}.pkl")
                with open(cap_path, "wb") as f:
                    pickle.dump(cap_payload, f)
                logger.info(
                    "CAPTURE: saved %d completions -> %s", len(input_ids), cap_path
                )
            except Exception:
                logger.exception("CAPTURE: message dump failed at step %d", step)

        # 3. Dispatch to the model node (HF or manual — whichever is running).
        fwd_data = await self._node.call("model", "forward_accum_grad", req)
        result = _to_dict(_from_dict(TrainStepResult, fwd_data))

        if cap:
            # 4. Record the HF/node loss (grad_norm comes from _call_train_update).
            self._cap_metrics_path = os.path.join(
                self.capture_dir, f"round{step}_hf_metrics.json"
            )
            try:
                with open(self._cap_metrics_path, "w") as f:
                    json.dump(
                        {
                            "step": step,
                            "loss": result.get("loss"),
                            "grad_norm": None,  # filled by _call_train_update
                            "lr": self._compute_lr(step),
                        },
                        f, indent=2,
                    )
            except Exception:
                logger.exception("CAPTURE: metrics write failed at step %d", step)

        return result

    async def _call_train_update(self, step: int) -> dict[str, Any]:
        result = await super()._call_train_update(step)
        if self._is_capture_step(step) and self._cap_metrics_path is not None:
            try:
                # Re-read the json (has loss) and add grad_norm.
                with open(self._cap_metrics_path) as f:
                    metrics = json.load(f)
                metrics["grad_norm"] = result.get("grad_norm")
                with open(self._cap_metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                logger.info(
                    "CAPTURE: step %d HF metrics: loss=%s grad_norm=%s",
                    step, metrics.get("loss"), metrics.get("grad_norm"),
                )
            except Exception:
                logger.exception("CAPTURE: grad_norm write failed at step %d", step)
            self._cap_metrics_path = None
        return result
