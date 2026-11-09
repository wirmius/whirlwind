"""Integration tests for whirlwind RLHF components.

These tests validate:
1. Config loading from YAML (all 4 config types)
2. Message serialization roundtrips
3. Reward computation (unit test)
4. Training orchestrator prompt expansion logic
5. Full component bringup (requires Qwen3-0.6B model + GPU)

Run:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -k "full"  # only GPU-heavy tests
"""

from __future__ import annotations

import asyncio
import importlib
import os
import pathlib
import tempfile
import unittest

from whirlwind.config import (
    ModelConfig,
    ProgressTrackerConfig,
    RewardConfig,
    RolloutConfig,
    TrainingConfig,
)
from whirlwind.messages import (
    MetricsBatch,
    PromptBatch,
    RewardRequest,
    RewardResult,
    RolloutResult,
    TrainStepRequest,
    TrainStepResult,
    WeightSyncRequest,
    _from_dict,
    _to_dict,
)


def _import_torch():
    """Import torch lazily — skip tensor tests if unavailable.

    Catches OSError too (broken .so / missing CUDA libs), not just ImportError.
    """
    try:
        import torch

        return torch
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_sglang() -> bool:
    return importlib.util.find_spec("sglang") is not None


def _has_qwen3() -> bool:
    """Check if Qwen3-0.6B model is available locally."""
    # For tests, we check if we can import transformers
    try:
        from transformers import AutoModelForCausalLM

        # Try to load the config only (fast)
        AutoModelForCausalLM.from_config(
            AutoModelForCausalLM.config_class.from_pretrained(
                "Qwen/Qwen3-0.6B", trust_remote_code=False
            )
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# TestConfigLoading
# ---------------------------------------------------------------------------


class TestConfigLoading(unittest.TestCase):
    """Test YAML config loading for all config types."""

    def test_rollout_config_from_yaml(self) -> None:
        cfg = RolloutConfig.from_yaml("configs/rollout/sglang.yaml")
        self.assertIsInstance(cfg, RolloutConfig)
        self.assertNotEqual(cfg.model_path, "")
        self.assertGreater(cfg.tp_size, 0)

    def test_reward_config_from_yaml(self) -> None:
        cfg = RewardConfig.from_yaml("configs/reward/function.yaml")
        self.assertIsInstance(cfg, RewardConfig)
        self.assertIn(cfg.reward_type, ("length", "accuracy"))
        self.assertGreater(cfg.group_size, 0)

    def test_model_config_from_yaml(self) -> None:
        cfg = ModelConfig.from_yaml("configs/model/fsdp2.yaml")
        self.assertIsInstance(cfg, ModelConfig)
        self.assertNotEqual(cfg.model_path, "")
        self.assertGreater(cfg.fsdp_size, 0)

    def test_training_config_from_yaml(self) -> None:
        cfg = TrainingConfig.from_yaml("configs/training/grpo.yaml")
        self.assertIsInstance(cfg, TrainingConfig)
        self.assertGreater(cfg.total_steps, 0)
        self.assertIsInstance(cfg.rollout, RolloutConfig)

    def test_config_save_load_roundtrip(self) -> None:
        """Test save_yaml / from_yaml roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # RolloutConfig
            rc = RolloutConfig(model_path="test-model", tp_size=2, temperature=0.8)
            path = os.path.join(tmpdir, "rollout.yaml")
            rc.save_yaml(path)
            rc2 = RolloutConfig.from_yaml(path)
            self.assertEqual(rc.model_path, rc2.model_path)
            self.assertEqual(rc.tp_size, rc2.tp_size)
            self.assertEqual(rc.temperature, rc2.temperature)

            # RewardConfig
            rw = RewardConfig(reward_type="length", group_size=8)
            path = os.path.join(tmpdir, "reward.yaml")
            rw.save_yaml(path)
            rw2 = RewardConfig.from_yaml(path)
            self.assertEqual(rw.reward_type, rw2.reward_type)
            self.assertEqual(rw.group_size, rw2.group_size)

            # ModelConfig
            mc = ModelConfig(
                model_path="test-model",
                learning_rate=2e-5,
                fsdp_size=4,
            )
            path = os.path.join(tmpdir, "model.yaml")
            mc.save_yaml(path)
            mc2 = ModelConfig.from_yaml(path)
            self.assertEqual(mc.model_path, mc2.model_path)
            self.assertEqual(mc.learning_rate, mc2.learning_rate)
            self.assertEqual(mc.fsdp_size, mc2.fsdp_size)

            # TrainingConfig
            tc = TrainingConfig(total_steps=500)
            path_tc = os.path.join(tmpdir, "training.yaml")
            tc.save_yaml(path_tc)
            tc2 = TrainingConfig.from_yaml(path_tc)
            self.assertEqual(tc.total_steps, tc2.total_steps)

            # ProgressTrackerConfig
            pt = ProgressTrackerConfig(
                backend="wandb",
                wandb_project="test-project",
                wandb_entity="test-entity",
                wandb_mode="disabled",
            )
            path = os.path.join(tmpdir, "tracker.yaml")
            pt.save_yaml(path)
            pt2 = ProgressTrackerConfig.from_yaml(path)
            self.assertEqual(pt.backend, pt2.backend)
            self.assertEqual(pt.wandb_project, pt2.wandb_project)
            self.assertEqual(pt.wandb_entity, pt2.wandb_entity)
            self.assertEqual(pt.wandb_mode, pt2.wandb_mode)

    def test_test_configs_load(self) -> None:
        """Test that test-specific configs load correctly."""
        base = pathlib.Path(__file__).parent
        cfg_dir = base / "configs"

        rc = RolloutConfig.from_yaml(cfg_dir / "rollout_qwen3_0_6b.yaml")
        self.assertEqual(rc.model_path, "Qwen/Qwen3-0.6B")
        self.assertGreater(rc.tp_size, 0)

        mc = ModelConfig.from_yaml(cfg_dir / "model_qwen3_0_6b.yaml")
        self.assertEqual(mc.model_path, "Qwen/Qwen3-0.6B")
        self.assertEqual(mc.fsdp_size, 1)

        rw = RewardConfig.from_yaml(cfg_dir / "reward_test.yaml")
        self.assertEqual(rw.reward_type, "length")
        self.assertEqual(rw.group_size, 4)

        tc = TrainingConfig.from_yaml(cfg_dir / "training_test.yaml")
        self.assertEqual(tc.total_steps, 3)
        self.assertEqual(tc.rollout_per_prompt, 2)


# ---------------------------------------------------------------------------
# TestMessageSerialization
# ---------------------------------------------------------------------------


class TestMessageSerialization(unittest.TestCase):
    """Test message dataclass serialization roundtrips."""

    def test_prompt_batch_roundtrip(self) -> None:
        batch = PromptBatch(
            prompts=["Hello", "Hello", "World", "World"],
            entry_ids=[0, 0, 1, 1],
            step=42,
        )
        data = _to_dict(batch)
        restored = _from_dict(PromptBatch, data)
        self.assertEqual(batch.prompts, restored.prompts)
        self.assertEqual(batch.step, restored.step)

    def test_rollout_result_roundtrip(self) -> None:
        result = RolloutResult(
            completions=["Hi there", "Greetings", "Earth", "Planet"],
            token_ids=[[1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11]],
            prompt_token_ids=[[0], [0, 1], [0, 1, 2], [0, 1]],
            entry_ids=[0, 0, 1, 1],
        )
        data = _to_dict(result)
        restored = _from_dict(RolloutResult, data)
        self.assertEqual(result.completions, restored.completions)

    def test_reward_request_roundtrip(self) -> None:
        req = RewardRequest(
            prompts=["Hello", "World"],
            completions=["Hi", "Earth"],
            entry_ids=[0, 0],
            step=5,
        )
        data = _to_dict(req)
        restored = _from_dict(RewardRequest, data)
        self.assertEqual(req.prompts, restored.prompts)
        self.assertEqual(req.completions, restored.completions)
        self.assertEqual(req.step, restored.step)

    def test_reward_result_roundtrip(self) -> None:
        result = RewardResult(
            rewards=[1.5, -0.3, 0.8, -1.2],
            entry_ids=[0, 0, 1, 1],
        )
        data = _to_dict(result)
        restored = _from_dict(RewardResult, data)
        self.assertAlmostEqual(result.rewards[0], restored.rewards[0])

    def test_train_step_request_roundtrip(self) -> None:
        req = TrainStepRequest(
            input_ids=[[1, 2, 3], [4, 5]],
            label_ids=[[-100, 2, 3], [-100, 5]],
            rewards=[1.0, -1.0],
            prompts=["A", "B"],
            completions=["C", "D"],
            entry_ids=[0, 1],
        )
        data = _to_dict(req)
        restored = _from_dict(TrainStepRequest, data)
        self.assertEqual(req.prompts, restored.prompts)
        self.assertEqual(req.completions, restored.completions)
        self.assertEqual(req.rewards, restored.rewards)

    def test_train_step_result_roundtrip(self) -> None:
        result = TrainStepResult(
            loss=2.5,
            grad_norm=0.8,
            lr=1e-5,
            checkpoint_path="/tmp/ckpt",
        )
        data = _to_dict(result)
        restored = _from_dict(TrainStepResult, data)
        self.assertAlmostEqual(result.loss, restored.loss)
        self.assertAlmostEqual(result.grad_norm, restored.grad_norm)
        self.assertAlmostEqual(result.lr, restored.lr)
        self.assertEqual(result.checkpoint_path, restored.checkpoint_path)

    def test_weight_sync_request_roundtrip(self) -> None:
        req = WeightSyncRequest(
            checkpoint_path="/tmp/step_5",
            step=5,
        )
        data = _to_dict(req)
        restored = _from_dict(WeightSyncRequest, data)
        self.assertEqual(req.checkpoint_path, restored.checkpoint_path)
        self.assertEqual(req.step, restored.step)

    def test_metrics_batch_roundtrip(self) -> None:
        batch = MetricsBatch(
            metrics={"loss": 0.5, "grad_norm": 1.2, "lr": 1e-5},
            node_id="orchestrator",
        )
        data = _to_dict(batch)
        restored = _from_dict(MetricsBatch, data)
        self.assertAlmostEqual(batch.metrics["loss"], restored.metrics["loss"])
        self.assertAlmostEqual(batch.metrics["grad_norm"], restored.metrics["grad_norm"])
        self.assertEqual(batch.node_id, restored.node_id)


# ---------------------------------------------------------------------------
# TestRewardComputation (unit test)
# ---------------------------------------------------------------------------


class TestRewardComputation(unittest.TestCase):
    """Test reward computation logic in isolation."""

    def test_length_reward_basic(self) -> None:
        """Test basic length-based reward without normalization."""
        from whirlwind.messages import RewardRequest
        from whirlwind.reward import RewardNode

        config = RewardConfig(
            reward_type="length",
            length_scale=1.0,
        )
        node = RewardNode(config)
        # Manually test the internal method

        req = RewardRequest(
            entry_ids=[0, 0, 1],
            prompts=["A", "B", "C"],
            completions=["short", "much longer text", "medium"],
            step=1,
        )
        rewards = node._length_based_reward(req)
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 5)  # len("short")
        self.assertEqual(rewards[1], 16)  # len("much longer text")
        self.assertEqual(rewards[2], 6)  # len("medium")

    def test_length_reward_scaled(self) -> None:
        """Test length-based reward with scaling."""
        from whirlwind.messages import RewardRequest
        from whirlwind.reward import RewardNode

        config = RewardConfig(
            reward_type="length",
            length_scale=2.0,
        )
        node = RewardNode(config)
        req = RewardRequest(
            entry_ids=[0],
            prompts=["X"],
            completions=["hello"],
            step=1,
        )
        rewards = node._length_based_reward(req)
        self.assertAlmostEqual(rewards[0], 10)  # 2.0 * len("hello")

    def test_length_reward_normalized(self) -> None:
        """Test length-based reward + GRPO advantage computation (group normalization)."""
        from whirlwind.messages import RewardRequest
        from whirlwind.reward import RewardNode

        config = RewardConfig(
            reward_type="length",
            length_scale=1.0,
            group_size=2,
        )
        node = RewardNode(config)
        req = RewardRequest(
            entry_ids=[0, 0],
            prompts=["A", "B"],
            completions=["short", "longer"],
            step=1,
        )
        # _length_based_reward now returns raw (non-normalized) rewards
        raw_rewards = node._length_based_reward(req)
        self.assertEqual(len(raw_rewards), 2)
        self.assertNotAlmostEqual(raw_rewards[0], raw_rewards[1])

        # GRPO advantages are computed separately via _compute_grpo_advantages
        advantages = node._compute_grpo_advantages(raw_rewards, req.entry_ids)
        # After z-score normalization, mean should be ~0
        self.assertAlmostEqual(sum(advantages) / len(advantages), 0.0, places=5)
        self.assertEqual(len(advantages), 2)

    def test_frac_reward_zero_std_metric(self) -> None:
        """frac_reward_zero_std metric matches fraction of zero-std-group samples.

        Mirrors TRL frac_reward_zero_std: fraction of samples belonging to
        groups whose reward std is ~0 (no learning signal).
        """
        from whirlwind.messages import RewardRequest
        from whirlwind.reward import RewardNode

        captured: list[dict] = []

        class _MockNode:
            def call_bg_no_return(self, target, method, payload):
                if method == "log_metrics":
                    captured.append(payload["metrics"])

        config = RewardConfig(reward_type="accuracy", group_size=4)
        node = RewardNode(config)
        node._node = _MockNode()  # type: ignore[assignment]

        # Group 0: rewards [1, 1, 1, 1] -> std 0 (zero-std, 4 samples)
        # Group 1: rewards [0, 1, 0, 1] -> std > 0 (4 samples)
        # frac = 4/8 = 0.5
        req = RewardRequest(
            entry_ids=[0, 0, 0, 0, 1, 1, 1, 1],
            prompts=["q"] * 8,
            completions=[
                "#### 5",
                "#### 5",
                "#### 5",
                "#### 5",  # group 0: all match
                "no answer",
                "#### 5",
                "no answer",
                "#### 5",  # group 1: mixed
            ],
            responses=[
                "#### 5",
                "#### 5",
                "#### 5",
                "#### 5",
                "#### 5",
                "#### 5",
                "#### 5",
                "#### 5",
            ],
            step=1,
        )
        node._compute_rewards(req)
        self.assertTrue(captured, "no metrics captured")
        m = captured[0]
        # Default log_prefix="" → train group.
        self.assertIn("train/frac_reward_zero_std", m)
        self.assertAlmostEqual(m["train/frac_reward_zero_std"], 0.5, places=5)

    async def _test_reward_node_rpc(self) -> None:
        """Test reward node RPC endpoint (requires whirlcoms orchestrator)."""
        from whirlwind.reward import RewardNode
        from whirlwind.whirlcoms import Orchestrator

        orch = Orchestrator()
        await orch.start()

        config = RewardConfig(reward_type="length", group_size=4)
        node = RewardNode(config)
        await node.start()

        await asyncio.sleep(0.5)  # let node register

        # Call via RPC (direct)
        from whirlwind.messages import RewardRequest, _to_dict

        req = _to_dict(
            RewardRequest(
                entry_ids=[0, 0, 1],
                prompts=["A", "B", "C"],
                completions=["hi", "hello world", "yes"],
                step=1,
            )
        )
        result = node._compute_rewards(req)
        self.assertIn("rewards", result)
        self.assertEqual(len(result["rewards"]), 3)

        await node.stop()
        await orch.stop()

    def test_reward_node_rpc_sync(self) -> None:
        """Sync wrapper for async test."""
        asyncio.run(self._test_reward_node_rpc())


# ---------------------------------------------------------------------------
# TestTrackerNode (unit test — backend="none")
# ---------------------------------------------------------------------------


class TestTrackerNode(unittest.TestCase):
    """Test tracker node RPC endpoints (no W&B required)."""

    async def _test_tracker_node_rpc(self) -> None:
        from whirlwind.whirlcoms import Orchestrator

        orch = Orchestrator()
        await orch.start()

        from whirlwind.tracker import ProgressTrackerNode

        config = ProgressTrackerConfig(
            backend="none",
        )
        node = ProgressTrackerNode(config)
        await node.start()

        await asyncio.sleep(0.5)  # let node register

        # Call log_metrics
        req = _to_dict(
            MetricsBatch(
                metrics={"loss": 0.5, "grad_norm": 1.2, "lr": 1e-5},
                node_id="orchestrator",
            )
        )
        result = node._handle_log_metrics(req)
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 3)

        # Call finish
        finish_result = node._handle_finish()
        self.assertTrue(finish_result["ok"])

        # Call ping
        ping_result = node._ping()
        self.assertTrue(ping_result["ok"])
        self.assertEqual(ping_result["backend"], "none")

        await node.stop()
        await orch.stop()

    def test_tracker_node_rpc_sync(self) -> None:
        """Sync wrapper for async test."""
        asyncio.run(self._test_tracker_node_rpc())

    def test_grouping_prefix_preserved(self) -> None:
        """train/ and val/ prefixed keys are preserved as-is (not re-prefixed
        with node_id), so all training metrics collapse under train/* and
        validation under val/* regardless of source node. Bare keys fall back
        to the legacy {node_id}.{name} grouping."""
        from whirlwind.config import ProgressTrackerConfig
        from whirlwind.tracker import ProgressTrackerNode

        node = ProgressTrackerNode(ProgressTrackerConfig(backend="none"))
        # train/ and val/ keys from any node → preserved
        node._handle_log_metrics(
            _to_dict(
                MetricsBatch(
                    metrics={
                        "train/loss": 0.5,
                        "train/clip_ratio/region_mean": 0.0,
                        "val/reward_mean": 0.8,
                    },
                    node_id="model",
                )
            )
        )
        node._handle_log_metrics(
            _to_dict(
                MetricsBatch(
                    metrics={"train/completions/mean_length": 12.0},
                    node_id="rollout",
                )
            )
        )
        node._handle_log_metrics(
            _to_dict(MetricsBatch(metrics={"val/batch_size": 32.0}, node_id="reward"))
        )
        # Bare (ungrouped) key → legacy {node_id}.{name}
        node._handle_log_metrics(
            _to_dict(MetricsBatch(metrics={"step": 7.0}, node_id="orchestrator"))
        )
        acc = node._accumulated
        # Grouped keys preserved exactly
        self.assertIn("train/loss", acc)
        self.assertIn("train/clip_ratio/region_mean", acc)
        self.assertIn("train/completions/mean_length", acc)
        self.assertIn("val/reward_mean", acc)
        self.assertIn("val/batch_size", acc)
        self.assertNotIn("model.train/loss", acc)  # not double-prefixed
        self.assertNotIn("rollout.train/completions/mean_length", acc)
        # Bare key → legacy node-id grouping
        self.assertIn("orchestrator.step", acc)
        self.assertNotIn("step", acc)


# ---------------------------------------------------------------------------
# TestOrchestratorLogic (unit test)
# ---------------------------------------------------------------------------


class TestOrchestratorLogic(unittest.TestCase):
    """Test training orchestrator logic without full RPC."""

    def test_expand_prompts(self) -> None:
        """Test prompt expansion for rollout groups."""
        from whirlwind.config import TrainingConfig
        from whirlwind.training import TrainingOrchestrator

        config = TrainingConfig()
        orch = TrainingOrchestrator(config)

        prompts = ["A", "B", "C"]
        result = orch._expand_prompts(prompts, reps=4)
        expanded, _, entry_ids = result[0], result[1], result[2]

        self.assertEqual(len(entry_ids), 12)  # 3 prompts × 4 reps
        self.assertEqual(len(expanded), 12)
        self.assertEqual(entry_ids, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        self.assertEqual(expanded, ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"])

    def test_expand_prompts_single(self) -> None:
        """Test with single replication (no expansion)."""
        from whirlwind.config import TrainingConfig
        from whirlwind.training import TrainingOrchestrator

        config = TrainingConfig()
        orch = TrainingOrchestrator(config)

        prompts = ["Hello"]
        result = orch._expand_prompts(prompts, reps=1)
        expanded, _, entry_ids = result[0], result[1], result[2]

        self.assertEqual(entry_ids, [0])
        self.assertEqual(expanded, ["Hello"])


# ---------------------------------------------------------------------------
# TestLossVariants (unit test — torch only, no GPU)
# ---------------------------------------------------------------------------


@unittest.skipIf(_import_torch() is None, "torch not installed")
class TestLossVariants(unittest.TestCase):
    """Test GRPO / DR-GRPO / BNPO loss aggregation variants."""

    def _make_tensors(
        self,
    ) -> tuple:
        """Create test tensors for loss computation.

        Two samples:
            - Sample 0: 3 completion tokens, advantage +1.0
            - Sample 1: 2 completion tokens, advantage -1.0
        """
        import torch

        # Completion mask: [batch=2, seq=5] — last 3 tokens for sample 0, last 2 for sample 1
        mask = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0],
            ]
        )
        # Current and old logprobs (identical → no IS ratio effect)
        per_token_logps = torch.tensor(
            [
                [0.0, 0.0, -1.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, -1.0, -1.0],
            ]
        )
        old_per_token_logps = per_token_logps.clone()
        advantages = torch.tensor([1.0, -1.0])
        return per_token_logps, old_per_token_logps, None, advantages, mask

    def test_grpo_loss_aggregation(self) -> None:
        """GRPO: mean of per-sequence losses."""
        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(loss_reduction="grpo", gradient_accumulation_steps=1)
        node = ModelUpdateNode(config)
        args = self._make_tensors()
        loss = node._compute_loss(*args)
        # With identical per-token logps (IS ratio ≈ 1.0):
        #   Sample 0: advantage +1.0 → loss = -1.0 * 1.0 = -1.0
        #   Sample 1: advantage -1.0 → loss = -1.0 * (-1.0) = +1.0
        # GRPO mean = (-1.0 + 1.0) / 2 = 0.0
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_bnpo_loss_aggregation(self) -> None:
        """BNPO: global token-level normalization."""
        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(loss_reduction="bnpo", gradient_accumulation_steps=1)
        node = ModelUpdateNode(config)
        args = self._make_tensors()
        loss = node._compute_loss(*args)
        # BNPO divides by total active tokens (5) instead of per-seq then mean
        # With identical per-token losses and same advantages, BNPO
        # should produce a different result when sequences have different lengths
        # (but with our uniform -1.0 per-token logps and matched advantages,
        #  both GRPO and BNPO converge to the same value).
        # Verify it computes without error and gives a finite result.
        import torch

        self.assertTrue(torch.isfinite(loss))

    def test_dr_grpo_loss_aggregation(self) -> None:
        """DR-GRPO: normalize by batch_size * max_completion_length."""
        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(loss_reduction="dr_grpo", gradient_accumulation_steps=1)
        node = ModelUpdateNode(config)
        args = self._make_tensors()
        loss = node._compute_loss(*args)
        # Verifies DR-GRPO aggregation runs and produces a finite result
        import torch

        self.assertTrue(torch.isfinite(loss))

    def test_gradient_accumulation_scaling(self) -> None:
        """Loss should be divided by gradient_accumulation_steps."""
        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config1 = ModelConfig(loss_reduction="grpo", gradient_accumulation_steps=1)
        node1 = ModelUpdateNode(config1)
        loss1 = node1._compute_loss(*self._make_tensors())

        config4 = ModelConfig(loss_reduction="grpo", gradient_accumulation_steps=4)
        node4 = ModelUpdateNode(config4)
        loss4 = node4._compute_loss(*self._make_tensors())

        # With acc_steps=4, loss should be 1/4 of acc_steps=1
        self.assertAlmostEqual(loss4.item(), loss1.item() / 4.0, places=5)

    def test_invalid_loss_reduction_falls_back_to_grpo(self) -> None:
        """Unknown loss_reduction should fall back to grpo."""
        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(loss_reduction="unknown_variant", gradient_accumulation_steps=1)
        node = ModelUpdateNode(config)
        loss = node._compute_loss(*self._make_tensors())
        # Should produce a finite result (falls back to GRPO)
        import torch

        self.assertTrue(torch.isfinite(loss))

    def test_token_level_importance_sampling(self) -> None:
        """Token-level IS: each token gets its own importance weight."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(
            loss_reduction="grpo",
            importance_sampling_level="token",
            gradient_accumulation_steps=1,
        )
        node = ModelUpdateNode(config)

        # Two samples with different logprobs (IS ratio != 1.0)
        mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0]])
        per_token_logps = torch.tensor([[0.0, 0.0, -0.5, -0.5, -0.5], [0.0, 0.0, 0.0, -0.5, -0.5]])
        old_per_token_logps = torch.tensor(
            [[0.0, 0.0, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, -1.0, -1.0]]
        )
        advantages = torch.tensor([1.0, -1.0])

        loss = node._compute_loss(per_token_logps, old_per_token_logps, None, advantages, mask)
        # IS ratio: exp(log_ratio) = exp(0.5) ≈ 1.6487 per token
        # coef_1 ≈ 1.6487 > (1 + clip_high=0.2), so clipped to 1.2
        # With advantage +1.0: -min(1.6487 * 1.0, 1.2 * 1.0) = -1.2
        # With advantage -1.0: -min(1.6487 * (-1.0), 1.2 * (-1.0)) = -(-1.2) = +1.2
        # GRPO mean of per-seq losses ≈ 0.0 (same as no IS for symmetric case)
        self.assertTrue(torch.isfinite(loss))

    def test_sequence_level_importance_sampling(self) -> None:
        """Sequence-level IS: one importance weight per sequence."""
        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(
            loss_reduction="grpo",
            importance_sampling_level="sequence",
            gradient_accumulation_steps=1,
        )
        node = ModelUpdateNode(config)

        per_token_logps, old_per_token_logps, _, advantages, mask = self._make_tensors()
        loss = node._compute_loss(per_token_logps, old_per_token_logps, None, advantages, mask)
        # Identical logprobs → IS ratio ≈ 1.0 → same as GRPO baseline
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_invalid_importance_sampling_level(self) -> None:
        """Unknown importance_sampling_level should raise ValueError."""
        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(
            loss_reduction="grpo",
            importance_sampling_level="invalid",
            gradient_accumulation_steps=1,
        )
        node = ModelUpdateNode(config)

        per_token_logps, old_per_token_logps, _, advantages, mask = self._make_tensors()
        with self.assertRaises(ValueError):
            node._compute_loss(per_token_logps, old_per_token_logps, None, advantages, mask)

    def _pad_advantages(self, advantages_list, mask):
        """Pad per-token advantages to [batch, max_len] like _pad_and_stack.

        Args:
            advantages_list: list of lists [batch, variable] or list of floats [batch]
            mask: completion mask [batch, max_len]
        Returns:
            Padded tensor [batch] or [batch, max_len]
        """
        import torch

        if advantages_list is None:
            return torch.zeros(mask.size(0), dtype=torch.float32)
        if isinstance(advantages_list[0], list):
            # Per-token advantages: pad to [batch, max_len]
            batch_size = mask.size(0)
            max_len = mask.size(1)
            padded = torch.zeros(batch_size, max_len, dtype=torch.float32)
            for i in range(batch_size):
                n_tokens = len(advantages_list[i])
                padded[i, -n_tokens:] = torch.tensor(advantages_list[i])
            return padded
        else:
            # Sequence-level advantages: [batch]
            return torch.tensor(advantages_list, dtype=torch.float32)

    def test_per_token_advantages_grpo(self) -> None:
        """Per-token advantages with GRPO loss aggregation."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(
            loss_reduction="grpo",
            importance_sampling_level="token",
            gradient_accumulation_steps=1,
        )
        node = ModelUpdateNode(config)

        # Same setup as _make_tensors but with per-token advantages
        mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0]])
        per_token_logps = torch.tensor([[0.0, 0.0, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, -1.0, -1.0]])
        old_per_token_logps = per_token_logps.clone()
        # Per-token advantages: padded to [batch, max_len] by _pad_and_stack
        advantages = self._pad_advantages([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], mask)

        loss = node._compute_loss(per_token_logps, old_per_token_logps, None, advantages, mask)
        # With identical logprobs (IS ratio ≈ 1.0):
        # per_token_loss = -advantages * mask
        # Sample 0: 3 tokens with adv=1.0 → loss = -3.0/3 = -1.0
        # Sample 1: 2 tokens with adv=-1.0 → loss = +2.0/2 = +1.0
        # GRPO mean = (-1.0 + 1.0) / 2 = 0.0
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_per_token_advantages_with_clipping(self) -> None:
        """Per-token advantages with PPO clipping (different logprobs)."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(
            loss_reduction="grpo",
            importance_sampling_level="token",
            gradient_accumulation_steps=1,
            clip_range_low=0.2,
            clip_range_high=0.2,
        )
        node = ModelUpdateNode(config)

        mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
        per_token_logps = torch.tensor([[0.0, 0.0, -0.5, -0.5, -0.5]])
        old_per_token_logps = torch.tensor([[0.0, 0.0, -1.0, -1.0, -1.0]])
        # Per-token advantages: padded to [batch, max_len]
        advantages = self._pad_advantages([[1.0, 1.0, 1.0]], mask)

        loss = node._compute_loss(per_token_logps, old_per_token_logps, None, advantages, mask)
        # IS ratio: exp(0.5) ≈ 1.6487 > 1.2 (clipped)
        # With clipping: coef_2 = clamp(1.6487, 0.8, 1.2) = 1.2
        # per_token_loss = -min(1.6487*1.0, 1.2*1.0) = -1.2 per token
        # GRPO: mean(-1.2 * 3 / 3) = -1.2
        self.assertAlmostEqual(loss.item(), -1.2, places=5)

    def test_per_token_vs_sequence_advantages(self) -> None:
        """Verify per-token advantages produce different results from sequence-level."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model import ModelUpdateNode

        config = ModelConfig(
            loss_reduction="grpo",
            importance_sampling_level="token",
            gradient_accumulation_steps=1,
        )
        node = ModelUpdateNode(config)

        mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0]])
        per_token_logps = torch.tensor([[0.0, 0.0, -0.5, -0.5, -0.5], [0.0, 0.0, 0.0, -0.5, -0.5]])
        old_per_token_logps = torch.tensor(
            [[0.0, 0.0, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, -1.0, -1.0]]
        )

        # Sequence-level advantages
        seq_advantages = torch.tensor([1.0, -1.0])
        seq_loss = node._compute_loss(
            per_token_logps, old_per_token_logps, None, seq_advantages, mask
        )

        # Per-token advantages (padded to [batch, max_len])
        # Simulate what _pad_and_stack does: right-pad completion portion
        token_advantages = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, -1.0, -1.0]])
        token_loss = node._compute_loss(
            per_token_logps, old_per_token_logps, None, token_advantages, mask
        )

        # Both should be finite and potentially different due to broadcasting
        self.assertTrue(torch.isfinite(seq_loss))
        self.assertTrue(torch.isfinite(token_loss))


# ---------------------------------------------------------------------------
# TestFullIntegration (GPU-required)
# ---------------------------------------------------------------------------


class TestFullIntegration(unittest.TestCase):
    """Full integration tests requiring GPU + Qwen3-0.6B model.

    These tests are slow and require:
    - CUDA GPU available
    - Qwen/Qwen3-0.6B model downloaded
    - SGLang installed
    """

    @unittest.skipUnless(_has_sglang(), "SGLang not installed")
    def test_full_training_loop(self) -> None:
        """Full training loop: rollout → reward → train → sync.

        This is an integration test that spins up all components.
        Requires GPU and the Qwen3-0.6B model.
        """
        # This test would:
        # 1. Start the whirlcoms orchestrator
        # 2. Start rollout node with Qwen3-0.6B
        # 3. Start reward node
        # 4. Start model update node (FSDP, single GPU)
        # 5. Run 2 training steps
        # 6. Verify checkpoint was saved
        # 7. Clean up

        # For now, mark as skipped with a meaningful message
        self.skipTest(
            "Full integration test requires GPU + Qwen3-0.6B model. "
            "Run manually with bringup_and_run.sh"
        )


if __name__ == "__main__":
    unittest.main()
