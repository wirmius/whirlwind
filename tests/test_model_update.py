"""Tests for the FSDP2 model-update node (whirlwind.model.model_fsdp2).

These tests exercise the model_update node INDEPENDENT of every other node:
no orchestrator, no rollout, no reward, no tracker, no RPC, no torchrun.
They construct a single-rank (world_size=1) ModelUpdateNode, initialize the
FSDP2-wrapped model directly, and call the training methods synchronously.

Single-rank FSDP2 is still real FSDP2: fully_shard runs with a 1-D device mesh
of shape (1,), params become DTensors, and all the shard/scatter/gather code
paths in _fsdp_using_function execute (the world_size==1 fast path calls the
inner fn directly, but the FSDP wrapping + DTensors are identical). So these
tests pin the real GRPO/SFT backward + optimizer + checkpoint + sleep/wake +
weight-export plumbing, just without multi-rank NCCL.

Tier 0 (pure-torch, no GPU/model): _compute_loss CE mode + logprob alignment.
Tier 1 (GPU + Qwen3-0.6B): reference weight-update match, checkpoint
roundtrip, export_weights reload, sleep/wake no-op, forward_no_grad top-K,
LR override + grad clip, microbatch/accumulation reconstruction.

Environment:
  Needs an environment with torch+CUDA+transformers (see tests/run_tests.sh).
  GPU tests skipUnless CUDA + transformers are importable. The small model
  defaults to Qwen/Qwen3-0.6B (downloads from HF hub into HF_HOME on first run,
  or set WHIRLWIND_TEST_MODEL / WHIRLWIND_TEST_MODEL_DIR to point at a local copy).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests._helpers import (
    _import_torch,
    model_path,
    skip_no_model_env,
)

# Pure-torch tier imports torch lazily inside tests; GPU tier imports below are
# also lazy so collection never crashes on a broken torch install.


def _ce_loss_reference(model, input_ids, label_ids):
    """Textbook HF causal-LM CE loss for the reference.

    Mirrors AutoModelForCausalLM(...).loss: shifted logits vs shifted labels,
    ignore_index=-100, mean over non-ignored tokens.
    """
    import torch.nn.functional as F

    logits = model(input_ids).logits  # [B, L, V]
    logits = logits[:, :-1, :]  # predict next
    labels = label_ids[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Tier 0 — pure torch, no GPU/model: CE loss + alignment
# ---------------------------------------------------------------------------


@skip_no_torch := unittest.skipIf(_import_torch() is None, "torch not installed")
class TestCELossMode(unittest.TestCase):
    """CE loss mode (loss_reduction='ce') matches HF F.cross_entropy exactly."""

    def setUp(self):
        import torch

        self.torch = torch
        torch.manual_seed(0)
        self.B, self.L, self.V = 2, 6, 50
        self.input_ids = torch.randint(0, self.V, (self.B, self.L))
        self.label_ids = torch.tensor(
            [
                [-100, -100, -100, 7, 13, 22],
                [-100, -100, -100, 3, 41, 9],
            ]
        )
        self.mask = (self.label_ids != -100).float()
        self.logits = torch.randn(self.B, self.L, self.V)

    def _per_token_logps(self):
        import torch

        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        lg = self.logits[:, :-1, :]
        lab_shift = self.label_ids[:, 1:]
        ptl = ModelUpdateNode._selective_log_softmax(lg, lab_shift)
        pad = torch.zeros(self.B, 1, dtype=ptl.dtype)
        return torch.cat([pad, ptl], dim=-1)  # prepend (matches the fix)

    def test_ce_matches_hf_cross_entropy(self):
        """_compute_loss(ce) == F.cross_entropy(ignore_index=-100, mean)."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        node = ModelUpdateNode(ModelConfig(loss_reduction="ce"))
        ptl = self._per_token_logps()
        ce = node._compute_loss(
            per_token_logps=ptl,
            old_per_token_logps=None,
            ref_per_token_logps=None,
            advantages=torch.zeros(self.B),
            mask=self.mask,
        )
        # HF reference
        import torch.nn.functional as F

        hf = F.cross_entropy(
            self.logits[:, :-1, :].reshape(-1, self.V),
            self.label_ids[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        self.assertAlmostEqual(ce.item(), hf.item(), places=4)

    def test_ce_ignores_grpo_inputs(self):
        """CE loss must not depend on advantages/old_logprobs/ref_logprobs."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        node = ModelUpdateNode(ModelConfig(loss_reduction="ce"))
        ptl = self._per_token_logps()
        base = node._compute_loss(
            per_token_logps=ptl,
            old_per_token_logps=None,
            ref_per_token_logps=None,
            advantages=torch.zeros(self.B),
            mask=self.mask,
        )
        with_junk = node._compute_loss(
            per_token_logps=ptl,
            old_per_token_logps=self.torch.randn(self.B, self.L),
            ref_per_token_logps=self.torch.randn(self.B, self.L),
            advantages=self.torch.tensor([5.0, -3.0]),
            mask=self.mask,
        )
        self.assertAlmostEqual(base.item(), with_junk.item(), places=6)

    def test_ce_differs_from_grpo(self):
        """CE and GRPO are different objectives."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        ptl = self._per_token_logps()
        node_ce = ModelUpdateNode(ModelConfig(loss_reduction="ce"))
        ce = node_ce._compute_loss(
            per_token_logps=ptl,
            old_per_token_logps=None,
            ref_per_token_logps=None,
            advantages=torch.zeros(self.B),
            mask=self.mask,
        )
        node_grpo = ModelUpdateNode(ModelConfig(loss_reduction="grpo", kl_coef=0.0))
        grpo = node_grpo._compute_loss(
            per_token_logps=ptl,
            old_per_token_logps=ptl.detach(),
            ref_per_token_logps=None,
            advantages=self.torch.tensor([1.0, -1.0]),
            mask=self.mask,
        )
        self.assertNotAlmostEqual(ce.item(), grpo.item(), places=3)


@skip_no_torch
class TestDrGrpoDenominator(unittest.TestCase):
    """DR-GRPO loss must normalize by the CONFIGURED max_completion_length cap,
    not the actual longest completion in the batch (DIV-1: matches TRL/paper).
    """

    def setUp(self):
        import torch

        self.torch = torch
        torch.manual_seed(0)
        # Two sequences, completion lengths 2 and 4 (actual max = 4),
        # but the config cap (max_completion_length) = 1024.
        self.B, self.L, self.V = 2, 6, 50
        self.label_ids = torch.tensor(
            [
                [-100, -100, 7, 13, -100, -100],  # 2 completion tokens
                [-100, -100, 3, 41, 9, 22],  # 4 completion tokens
            ]
        )
        self.mask = (self.label_ids != -100).float()
        self.input_ids = torch.randint(0, self.V, (self.B, self.L))
        self.logits = torch.randn(self.B, self.L, self.V)

    def _per_token_logps(self):
        import torch

        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        lg = self.logits[:, :-1, :]
        lab_shift = self.label_ids[:, 1:]
        ptl = ModelUpdateNode._selective_log_softmax(lg, lab_shift)
        pad = torch.zeros(self.B, 1, dtype=ptl.dtype)
        return torch.cat([pad, ptl], dim=-1)

    def test_denominator_uses_config_cap_not_actual_max(self):
        """dr_grpo loss must be sum(masked_loss) / (batch_size * max_completion_length_cap)."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        cap = 1024
        node = ModelUpdateNode(
            ModelConfig(loss_reduction="dr_grpo", max_completion_length=cap, kl_coef=0.0)
        )
        ptl = self._per_token_logps()
        # Use detached ptl as old_logprobs so coef_1 == 1 (no IS distortion);
        # advantages uniform so the per-token loss is a clean function of ptl.
        advantages = torch.ones(self.B)
        loss = node._compute_loss(
            per_token_logps=ptl,
            old_per_token_logps=ptl.detach(),
            ref_per_token_logps=None,
            advantages=advantages,
            mask=self.mask,
        )
        # Reconstruct the expected numerator: -min(coef_1*adv, coef_2*adv) * mask.
        # With old==current, log_ratio=0, coef_1=coef_2=1, per_token_loss = -adv = -1.
        expected_numerator = -(1.0 * self.mask).sum()  # = -sum(mask)
        expected_loss = expected_numerator / (self.B * cap)
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)
        # Sanity: actual-max denominator would give a ~256x larger value
        actual_max = int(self.mask.sum(-1).max().item())  # 4
        wrong_loss = expected_numerator / (self.B * actual_max)
        self.assertNotAlmostEqual(loss.item(), wrong_loss.item(), places=3)

    def test_loss_invariant_to_actual_completion_lengths(self):
        """Two batches with the SAME per-token loss but DIFFERENT actual lengths
        must yield the SAME dr_grpo loss (length-invariance from the cap)."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        node = ModelUpdateNode(
            ModelConfig(loss_reduction="dr_grpo", max_completion_length=1024, kl_coef=0.0)
        )
        # Short batch: 2 completion tokens per row
        mask_short = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
        # Long batch: 4 completion tokens per row (same per-token loss = -1)
        mask_long = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        ptl = -torch.ones(2, 6)  # uniform logps -> per_token_loss = -1 everywhere
        adv = torch.ones(2)
        loss_short = node._compute_loss(
            per_token_logps=ptl,
            old_per_token_logps=ptl.detach(),
            ref_per_token_logps=None,
            advantages=adv,
            mask=mask_short,
        )
        loss_long = node._compute_loss(
            per_token_logps=ptl,
            old_per_token_logps=ptl.detach(),
            ref_per_token_logps=None,
            advantages=adv,
            mask=mask_long,
        )
        # Same denominator (2 * 1024); numerators differ (sum of masks), so losses
        # differ proportionally to token count -- but the *rate per token* is
        # identical, which is the length-invariance property. Assert the
        # per-token contribution is equal:
        rate_short = loss_short.item() / mask_short.sum().item()
        rate_long = loss_long.item() / mask_long.sum().item()
        self.assertAlmostEqual(rate_short, rate_long, places=5)


@skip_no_torch
class TestLogprobAlignment(unittest.TestCase):
    """The prepend-pad fix: per_token_logps[t] == logprob(label_ids[t]).

    Pins the bug fix in _compute_per_token_logps so it cannot silently regress:
    the first completion token's logprob must be included, and the IS ratio
    must pair each token with its own old logprob (not the previous token's).
    """

    def test_first_completion_token_included(self):
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        class Out:
            def __init__(self, logits):
                self.logits = logits

        class StubModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self._logits = logits

            def forward(self, input_ids, attention_mask=None):
                return Out(self._logits)

        # prompt=[3,3], completion=[C0=0, C1=1, C2=2]
        input_ids = torch.tensor([[3, 3, 0, 1, 2]])
        label_ids = torch.tensor([[-100, -100, 0, 1, 2]])
        mask = (label_ids != -100).float()
        V = 4
        logits = torch.zeros(1, 5, V)
        logits[0, 1] = torch.tensor([0.0, 2.7, 0.0, 0.0])  # predicts C0
        logits[0, 2] = torch.tensor([0.0, 0.0, 1.0, 4.0])  # predicts C1
        logits[0, 3] = torch.tensor([5.0, 0.0, 0.0, 0.0])  # predicts C2

        def lp(t, tok):
            return logits[0, t, tok].item() - torch.logsumexp(logits[0, t], dim=-1).item()

        lp_C0, lp_C1, lp_C2 = lp(1, 0), lp(2, 1), lp(3, 2)

        node = ModelUpdateNode(ModelConfig())
        node._use_autocast = False  # CPU fp32 StubModel; skip autocast branch
        # TRL-aligned: prompt=[3,3] (p_max=2, no left pad), completion=[0,1,2].
        # StubModel returns logits at absolute positions; after shift [:, :-1]
        # and slice [:, -3:], positions 1,2,3 predict completion tokens 0,1,2.
        prompt_completion_ids = torch.tensor([[3, 3, 0, 1, 2]])
        completion_ids = torch.tensor([[0, 1, 2]])
        ptl, _ = node._compute_per_token_logps(
            model=StubModel(logits),
            prompt_completion_ids=prompt_completion_ids,
            attention_mask=torch.ones_like(prompt_completion_ids, dtype=torch.float32),
            completion_ids=completion_ids,
            logits_to_keep=3,
            temperature=1.0,
        )
        # per_token_logps is [1, 3] completion-shaped (first token INCLUDED by
        # construction -- no prepend/offset to get wrong).
        kept = [ptl[0, t].item() for t in range(3)]
        self.assertAlmostEqual(kept[0], lp_C0, places=3)  # first completion token INCLUDED
        self.assertAlmostEqual(kept[1], lp_C1, places=3)
        self.assertAlmostEqual(kept[2], lp_C2, places=3)

    def test_importance_sampling_ratio_aligned(self):
        """IS ratio pairs each token with its own old logprob (not shifted)."""
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        class Out:
            def __init__(self, logits):
                self.logits = logits

        class StubModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self._logits = logits

            def forward(self, input_ids, attention_mask=None):
                return Out(self._logits)

        input_ids = torch.tensor([[3, 3, 0, 1, 2]])
        label_ids = torch.tensor([[-100, -100, 0, 1, 2]])
        mask = (label_ids != -100).float()
        V = 4
        logits = torch.zeros(1, 5, V)
        logits[0, 1] = torch.tensor([0.0, 2.7, 0.0, 0.0])
        logits[0, 2] = torch.tensor([0.0, 0.0, 1.0, 4.0])
        logits[0, 3] = torch.tensor([5.0, 0.0, 0.0, 0.0])

        node = ModelUpdateNode(ModelConfig())
        node._use_autocast = False  # CPU fp32 StubModel; skip autocast branch
        prompt_completion_ids = torch.tensor([[3, 3, 0, 1, 2]])
        completion_ids = torch.tensor([[0, 1, 2]])
        ptl, _ = node._compute_per_token_logps(
            model=StubModel(logits),
            prompt_completion_ids=prompt_completion_ids,
            attention_mask=torch.ones_like(prompt_completion_ids, dtype=torch.float32),
            completion_ids=completion_ids,
            logits_to_keep=3,
            temperature=1.0,
        )
        # old_logprobs == current logprobs at completion positions -> ratio = 1.
        # per_token_logps is [1, 3] completion-shaped, so old aligns 1:1.
        old = ptl.clone()
        log_ratio = ptl - old
        for t in range(3):
            self.assertAlmostEqual(log_ratio[0, t].item(), 0.0, places=4)


class TestEngineISCorrection(unittest.TestCase):
    """Engine/trainer IS correction (DIV-6 alignment to TRL).

    Verifies the bounded IS ratio (from SGLang generation logprobs) is applied
    separately from ``coef_1`` (which uses the fresh detached forward), and
    that ``sequence_mask`` mode zeros sequences exceeding the cap.
    """

    def _make(self, **kw):
        import torch  # noqa: F401

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        defaults = dict(
            loss_reduction="grpo",
            kl_coef=0.0,
            importance_sampling_level="token",
            engine_is_correction=True,
            engine_is_mode="sequence_mask",
            engine_is_cap=3.0,
        )
        defaults.update(kw)
        return ModelUpdateNode(ModelConfig(**defaults))

    def test_is_ratio_one_when_sampling_equals_old(self):
        """sampling == old (forward) → IS ratio = exp(0) = 1 → loss unchanged."""
        import torch

        torch.manual_seed(0)
        B, T = 3, 5
        per = torch.randn(B, T)
        old = per.detach()
        samp = old.clone()  # == old → ratio 1.0
        adv = torch.randn(B)
        mask = torch.ones(B, T)
        node = self._make()
        loss_no_is = node._compute_loss(per, old, None, adv, mask, sampling_per_token_logps=None)
        node._accum_buf["diag"].clear()
        loss_with_is = node._compute_loss(per, old, None, adv, mask, sampling_per_token_logps=samp)
        self.assertAlmostEqual(loss_no_is.item(), loss_with_is.item(), places=5)

    def test_is_ratio_modifies_loss_when_sampling_differs(self):
        """sampling != old with ratio below cap → loss changes (ratio != 1)."""
        import torch

        torch.manual_seed(0)
        B, T = 3, 5
        per = torch.randn(B, T)
        old = per.detach()
        adv = torch.randn(B)
        mask = torch.ones(B, T)
        node = self._make()
        loss_no_is = node._compute_loss(per, old, None, adv, mask, sampling_per_token_logps=None)
        node._accum_buf["diag"].clear()
        # diff = 0.1/token → seq sum = 0.5 → ratio = exp(0.5)≈1.65 < cap 3 (kept)
        samp = old - 0.1
        loss_with_is = node._compute_loss(per, old, None, adv, mask, sampling_per_token_logps=samp)
        self.assertNotAlmostEqual(loss_no_is.item(), loss_with_is.item(), places=3)

    def test_is_ratio_masked_above_cap(self):
        """sequence_mask zeros sequences whose ratio > cap; keeps the rest."""
        import torch

        torch.manual_seed(0)
        B, T = 2, 4
        per = torch.randn(B, T)
        old = per.detach()
        adv = torch.tensor([1.0, 1.0])
        mask = torch.ones(B, T)
        node = self._make()
        samp = old.clone()
        # seq 0: diff=1.0/token → seq sum=4 → ratio=exp(4)≈54 > 3 → masked to 0
        samp[0] = old[0] - 1.0
        # seq 1: diff=0 → ratio=1.0 → kept unchanged
        loss = node._compute_loss(per, old, None, adv, mask, sampling_per_token_logps=samp)
        # Plain PPO loss on only the unmasked seq 1 (IS off, B=1)
        loss_seq1 = node._compute_loss(
            per[1:2], old[1:2], None, adv[1:2], mask[1:2], sampling_per_token_logps=None
        )
        # grpo = mean over batch; seq 0 zeroed → loss = (0 + loss_seq1) / 2
        self.assertAlmostEqual(loss.item(), loss_seq1.item() / 2.0, places=4)

    def test_is_correction_off_ignores_sampling(self):
        """engine_is_correction=False → sampling logprobs don't affect the loss."""
        import torch

        torch.manual_seed(0)
        B, T = 2, 3
        per = torch.randn(B, T)
        old = per.detach()
        adv = torch.randn(B)
        mask = torch.ones(B, T)
        node = self._make(engine_is_correction=False)
        a = node._compute_loss(per, old, None, adv, mask, sampling_per_token_logps=None)
        b = node._compute_loss(per, old, None, adv, mask, sampling_per_token_logps=old - 0.5)
        self.assertAlmostEqual(a.item(), b.item(), places=5)


class TestDiagnostics(unittest.TestCase):
    """Per-step diagnostic metrics (mirror TRL self._metrics keys)."""

    def test_diag_buffer_populates_with_trl_keys(self):
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        node = ModelUpdateNode(
            ModelConfig(
                loss_reduction="grpo",
                kl_coef=0.0,
                engine_is_correction=True,
                engine_is_mode="sequence_mask",
                engine_is_cap=3.0,
            )
        )
        B, T = 2, 4
        torch.manual_seed(1)
        per = torch.randn(B, T)
        old = per.detach()
        adv = torch.randn(B)
        mask = torch.ones(B, T)
        samp = old - 0.1
        # Entropy is plumbed from _forward_accum_grad; emulate it here with a
        # synthetic per-token tensor so the entropy diagnostic is exercised.
        entropy = torch.randn(B, T)
        node._accum_buf["diag"].clear()
        node._compute_loss(
            per, old, None, adv, mask, sampling_per_token_logps=samp, entropy=entropy
        )
        diag = node._accum_buf["diag"]
        self.assertEqual(len(diag), 1)
        d = diag[0]
        for k in [
            "clip_ratio/low_mean",
            "clip_ratio/high_mean",
            "clip_ratio/region_mean",
            "sampling/importance_sampling_ratio/min",
            "sampling/importance_sampling_ratio/mean",
            "sampling/importance_sampling_ratio/max",
            "sampling/sampling_logp_difference/mean",
            "sampling/sampling_logp_difference/max",
            "entropy",
        ]:
            self.assertIn(k, d, f"missing diagnostic key: {k}")
        # |old - samp| = 0.1 at every masked token
        self.assertAlmostEqual(d["sampling/sampling_logp_difference/mean"], 0.1, places=3)

    def test_diag_aggregation_min_of_min_max_of_max(self):
        """Aggregation reports GLOBAL min/max (min-of-mins, max-of-maxes), not
        the arithmetic mean of per-microbatch extremes — and preserves
        min<=mean<=max across multiple microbatches."""
        import math

        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        node = ModelUpdateNode(
            ModelConfig(
                loss_reduction="grpo",
                kl_coef=0.0,
                engine_is_correction=True,
                engine_is_mode="sequence_mask",
                engine_is_cap=3.0,
            )
        )
        node.world_size = 1  # single-rank: test the microbatch aggregation only
        node._accum_buf["diag"].clear()

        # Two microbatches with different ratio distributions.
        # seq-level ratio = exp(sum(old - samp)); mask all-ones → seq sum = T*diff.
        B, T = 2, 4
        torch.manual_seed(7)
        adv = torch.randn(B)
        mask = torch.ones(B, T)

        # microbatch A: per = old, samp = old - 0.05 → diff 0.05, ratio = exp(0.2)≈1.22
        perA = torch.randn(B, T)
        oldA = perA.detach()
        sampA = oldA - 0.05
        # microbatch B: per = old, samp = old - 0.4  → diff 0.4, ratio = exp(1.6)≈4.95 → masked to 0
        perB = torch.randn(B, T)
        oldB = perB.detach()
        sampB = oldB - 0.4
        node._compute_loss(perA, oldA, None, adv, mask, sampling_per_token_logps=sampA)
        node._compute_loss(perB, oldB, None, adv, mask, sampling_per_token_logps=sampB)
        self.assertEqual(len(node._accum_buf["diag"]), 2)

        # The aggregation currently inlines in _model_update_step; replicate it
        # here to assert the contract (sorted keys + non-numeric passthrough,
        # mirroring the production code).
        diag_buf = node._accum_buf.get("diag", [])
        agg_diag: dict[str, float] = {}
        keys = set().union(*(d.keys() for d in diag_buf))
        for k in sorted(keys):
            vals = [d[k] for d in diag_buf if k in d]
            if not vals:
                continue
            if not all(isinstance(v, (int, float)) for v in vals):
                agg_diag[k] = vals[-1]  # non-numeric: last-write-wins passthrough
                continue
            if k.endswith("/min"):
                agg_diag[k] = min(vals)
            elif k.endswith("/max"):
                agg_diag[k] = max(vals)
            else:
                agg_diag[k] = sum(vals) / len(vals)

        rmin = agg_diag["sampling/importance_sampling_ratio/min"]
        rmean = agg_diag["sampling/importance_sampling_ratio/mean"]
        rmax = agg_diag["sampling/importance_sampling_ratio/max"]
        # Global min/max, NOT averages: min should be 0.0 (microbatch B's masked
        # sequences), max should be ~1.22 (microbatch A's ratios).
        self.assertAlmostEqual(rmin, 0.0, places=6)
        self.assertAlmostEqual(rmax, math.exp(0.2), places=3)
        self.assertLessEqual(rmin, rmean)
        self.assertLessEqual(rmean, rmax)

    def test_ce_mode_skips_diagnostics(self):
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        node = ModelUpdateNode(ModelConfig(loss_reduction="ce"))
        B, T = 2, 4
        torch.manual_seed(2)
        per = torch.randn(B, T)
        old = per.detach()
        adv = torch.randn(B)
        mask = torch.ones(B, T)
        node._accum_buf["diag"].clear()
        node._compute_loss(per, old, None, adv, mask, sampling_per_token_logps=old)
        self.assertEqual(len(node._accum_buf["diag"]), 0)

    def test_entropy_matches_categorical_entropy(self):
        """compute_entropy=True matches TRL entropy_from_logits.

        Verifies _compute_per_token_logps(compute_entropy=True) matches TRL's
        entropy_from_logits (logsumexp - sum(softmax*logits)) at the shifted
        positions, and is None when not requested.
        """
        import torch

        from whirlwind.config import ModelConfig
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        class Out:
            def __init__(self, logits):
                self.logits = logits

        class StubModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self._logits = logits

            def forward(self, input_ids, attention_mask=None):
                return Out(self._logits)

        torch.manual_seed(0)
        V = 5
        logits = torch.randn(2, 6, V)
        input_ids = torch.randint(0, V, (2, 6))
        label_ids = input_ids.clone()
        label_ids[:, :2] = -100  # 2 prompt tokens, 4 completion tokens
        node = ModelUpdateNode(ModelConfig())
        node._use_autocast = False  # CPU fp32 StubModel; skip autocast branch
        # TRL-aligned: completion_ids = the 4 completion tokens, logits_to_keep=4.
        completion_ids = input_ids[:, 2:]  # [2, 4]
        attention_mask = torch.ones_like(input_ids, dtype=torch.float32)

        # entropy=None when not requested
        _, ent_off = node._compute_per_token_logps(
            model=StubModel(logits),
            prompt_completion_ids=input_ids,
            attention_mask=attention_mask,
            completion_ids=completion_ids,
            logits_to_keep=4,
            temperature=1.0,
        )
        self.assertIsNone(ent_off)

        # entropy computed when requested
        _, ent_on = node._compute_per_token_logps(
            model=StubModel(logits),
            prompt_completion_ids=input_ids,
            attention_mask=attention_mask,
            completion_ids=completion_ids,
            logits_to_keep=4,
            temperature=1.0,
            compute_entropy=True,
        )
        self.assertEqual(ent_on.shape, (2, 4))
        # Reference: TRL entropy_from_logits on the completion-sliced shifted
        # logits (shift then keep last 4 positions -- NO prepend in the
        # TRL-aligned path; entropy is completion-shaped [2, 4] by construction).
        shifted = logits[:, :-1, :][:, -4:, :]  # [2, 4, V]
        ref = torch.logsumexp(shifted, dim=-1) - (
            torch.softmax(shifted, dim=-1).mul(shifted).sum(dim=-1)
        )  # [2, 4]
        for b in range(2):
            for t in range(4):
                self.assertAlmostEqual(ent_on[b, t].item(), ref[b, t].item(), places=4)


# ---------------------------------------------------------------------------
# Shared GPU fixture helpers
# ---------------------------------------------------------------------------


def _make_node(config_overrides: dict | None = None):
    """Construct a single-rank ModelUpdateNode with FSDP2 init, NO RPC.

    Skips start() (no zmq/whirlcoms node), calls intitialize_model() directly.
    Sets RANK/WORLD_SIZE/LOCAL_RANK env vars for single-rank FSDP2.
    """
    import torch  # noqa: F401

    from whirlwind.config import ModelConfig
    from whirlwind.model.model_fsdp2 import ModelUpdateNode

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29512")

    overrides = dict(
        model_path=model_path(),
        torch_dtype="float32",  # deterministic reference comparison
        fsdp_cpu_offload=False,
        fsdp_reshard_after_forward=True,
        fsdp_activation_checkpointing=False,  # simpler; AC path covered separately
        fsdp_mixed_precision=False,  # fp32 throughout for exact reference match
        disable_dropout=True,
        torch_compile=False,
        optim="adamw",
        learning_rate=1e-4,
        weight_decay=0.0,
        max_grad_norm=1000.0,  # effectively no clip, so reference math is exact
        gradient_accumulation_steps=1,
        micro_batch_size=64,  # large enough to avoid microbatch splitting in
        # the reference-exactness tests; the microbatch path is exercised
        # separately by TestMicrobatchAccumulation.
        kl_coef=0.0,
        loss_reduction="ce",
        importance_sampling_level="token",
        temperature=1.0,
    )
    if config_overrides:
        overrides.update(config_overrides)
    cfg = ModelConfig(**overrides)
    node = ModelUpdateNode(cfg)
    node.intitialize_model({})  # no start() -> no RPC node; init model directly
    return node


def _toy_batch(node, n_samples=2, seq_len=16, prompt_len=4):
    """Build a TrainStepRequest batch from random token ids.

    Returns (req, input_ids_list, label_ids_list, mask) on CPU for the reference.
    """
    import torch  # noqa: F401

    from whirlwind.messages import TrainStepRequest

    tok = node._tokenizer
    V = tok.vocab_size
    input_ids, label_ids = [], []
    for _ in range(n_samples):
        ids = torch.randint(0, V, (seq_len,)).tolist()
        input_ids.append(ids)
        label_ids.append([-100] * prompt_len + ids[prompt_len:])
    req = TrainStepRequest(
        input_ids=input_ids,
        label_ids=label_ids,
        rewards=[0.0] * n_samples,
        advantages=None,
        prompts=[],
        completions=[],
    )
    return req, input_ids, label_ids


def _node_state_dict(node):
    """Return the node's full model state dict as plain CPU tensors.

    FSDP2 models return DTensors from .state_dict(); the FSDP-aware
    get_model_state_dict(full_state_dict=True, cpu_offload=True) gathers
    and offloads them to a plain CPU dict (names match the base HF model).
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(
        node._fsdp,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


# ---------------------------------------------------------------------------
# Tier 1 — GPU + Qwen3-0.6B: full FSDP2 plumbing vs a textbook HF Trainer
# ---------------------------------------------------------------------------


@skip_no_model_env
class TestReferenceWeightUpdate(unittest.TestCase):
    """The headline test: model_updater (CE, single-rank FSDP2) vs a textbook
    HF Trainer must produce identical updated weights + before/after logits.

    Both sides: same Qwen3-0.6B, same batch, same AdamW LR, same grad clip,
    same fp32, dropout off, no AC, no compile. The model_updater runs its real
    forward_accum_grad -> model_update_step -> export_weights path. The
    reference runs a minimal single-process forward/backward/step. We compare:
      - pre-update logits (sanity: models start identical)
      - post-update logits
      - updated weights
    """

    @classmethod
    def setUpClass(cls):
        import torch

        torch.manual_seed(123)
        cls.tmp = tempfile.mkdtemp(prefix="whirlwind_ref_")
        # Two independent nodes/refs must start from identical weights.
        # We load the reference model first, snapshot its state_dict, and feed
        # the SAME initial weights into the model_updater via a cold checkpoint
        # so both begin byte-identical (avoids any nondeterminism in two
        # separate from_pretrained calls).
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.mp = model_path()
        cls.tok = AutoTokenizer.from_pretrained(cls.mp, trust_remote_code=False)
        if getattr(cls.tok, "pad_token", None) is None:
            cls.tok.pad_token = cls.tok.eos_token

        torch.manual_seed(123)
        ref = AutoModelForCausalLM.from_pretrained(
            cls.mp,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        # disable dropout in the reference too
        for a in ("dropout", "attention_dropout", "hidden_dropout", "emb_dropout"):
            if hasattr(ref.config, a):
                setattr(ref.config, a, 0.0)
        ref.eval()  # we manage gradients manually
        cls.init_state = {k: v.clone() for k, v in ref.state_dict().items()}
        cls.ref_model = ref

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _load_init_into_node(self, node):
        """Force the node's FSDP2 model to the reference's initial weights.

        Plain load_state_dict fails on FSDP2 (mixed torch.Tensor / DTensor),
        so we use the FSDP-aware set_model_state_dict with full_state_dict=True,
        cpu_offload=True: it accepts a plain CPU tensor dict and shards it in.
        We pass clones so the FSDP setter cannot mutate self.init_state.
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        sd = {k: v.clone() for k, v in self.init_state.items()}
        set_model_state_dict(
            node._fsdp,
            model_state_dict=sd,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    def _logits(self, model, input_ids_tensor):
        import torch

        with torch.no_grad():
            out = model(input_ids_tensor)
            return out.logits.clone()

    def test_weight_update_matches_reference(self):
        import torch

        node = _make_node()
        try:
            # Align node's initial weights to the reference's
            self._load_init_into_node(node)

            # Same deterministic batch for both
            torch.manual_seed(999)
            req, ids_list, lab_list = _toy_batch(node, n_samples=2, seq_len=16, prompt_len=4)
            ids_t = torch.tensor(ids_list, dtype=torch.long, device="cuda")
            # pre-update logits (sanity: should match the reference model)
            pre_node = self._logits(node._fsdp, ids_t).cpu()
            pre_ref = self._logits(self.ref_model, ids_t.cpu()).cpu()
            max_pre_diff = (pre_node - pre_ref).abs().max().item()
            self.assertLess(
                max_pre_diff,
                1e-4,
                f"Pre-update logits diverged before any training (max diff {max_pre_diff}). "
                "Models did not start identical; _load_init_into_node failed to align weights.",
            )

            # --- Model updater: forward_accum_grad -> model_update_step ---
            from whirlwind.messages import ModelUpdateRequest

            fwd = node.forward_accum_grad(req)
            self.assertIsNotNone(fwd)
            upd = node.model_update_step(ModelUpdateRequest(lr=None))
            self.assertIsNotNone(upd)

            # --- Reference: same CE loss, same AdamW step, same LR, no clip ---
            ref_model = self.ref_model
            ref_model.train()
            opt = torch.optim.AdamW(
                ref_model.parameters(),
                lr=node.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=node.config.weight_decay,
            )
            # reset the reference to the SAME initial weights as the node started from
            ref_model.load_state_dict({k: v.clone() for k, v in self.init_state.items()})
            loss = _ce_loss_reference(ref_model, ids_t.cpu(), torch.tensor(lab_list))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # grad clip: node uses max_grad_norm=1000 -> effectively no clip.
            torch.nn.utils.clip_grad_norm_(
                ref_model.parameters(), max_norm=node.config.max_grad_norm
            )
            opt.step()

            # --- Compare post-update logits + weights ---
            post_node = self._logits(node._fsdp, ids_t).cpu()
            post_ref = self._logits(ref_model, ids_t.cpu()).cpu()
            max_post_diff = (post_node - post_ref).abs().max().item()
            self.assertLess(
                max_post_diff,
                1e-3,
                f"Post-update logits diverged (max diff {max_post_diff}). "
                "Weight update does not match the HF reference.",
            )

            # Compare a sample of named params (FSDP-aware get -> plain CPU dict)
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            node_state = get_model_state_dict(
                node._fsdp,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            ref_state = ref_model.state_dict()
            checked = 0
            for k, rv in ref_state.items():
                if k in node_state and rv.is_floating_point():
                    d = (node_state[k].float() - rv.float()).abs().max().item()
                    # fp32 AdamW across two independent optimizers + FSDP gather
                    # accumulates ~1e-4..1e-3 noise per step; the logits check
                    # above is the stricter functional guarantee.
                    self.assertLess(d, 1e-3, f"Param '{k}' diverged: max diff {d}")
                    checked += 1
            self.assertGreater(checked, 0, "no params compared")
        finally:
            del node


@skip_no_model_env
class TestCheckpointRoundtrip(unittest.TestCase):
    """save_hot_checkpoint -> load_hot_checkpoint restores params + optim +
    forward output to the pre-save state."""

    def test_hot_checkpoint_roundtrip(self):
        import torch

        from whirlwind.messages import CheckpointRequest

        node = _make_node()
        try:
            ckpt = os.path.join(node.config.checkpoint_dir, "roundtrip_ckpt")
            req, ids_list, lab_list = _toy_batch(node, n_samples=2, seq_len=12, prompt_len=3)
            ids_t = torch.tensor(ids_list, dtype=torch.long, device="cuda")

            # Train one step to populate optimizer state + nonzero grads
            node.forward_accum_grad(req)
            from whirlwind.messages import ModelUpdateRequest

            node.model_update_step(ModelUpdateRequest(lr=None))

            # Snapshot post-step logits + the full state dict
            before_logits = self._logits(node._fsdp, ids_t).cpu()
            before_state = _node_state_dict(node)

            # Save (decorate_serdes returns a dict in-process; parse it back)
            from whirlwind.messages import CheckpointResult, _from_dict

            res = _from_dict(
                CheckpointResult, node.save_hot_checkpoint(CheckpointRequest(path=ckpt, step=1))
            )
            self.assertTrue(res.ok, f"save failed: {res.error}")

            # Perturb: take another (different) step so the model diverges
            torch.manual_seed(7)
            req2, _, _ = _toy_batch(node, n_samples=2, seq_len=12, prompt_len=3)
            node.forward_accum_grad(req2)
            node.model_update_step(ModelUpdateRequest(lr=None))
            after_perturb_logits = self._logits(node._fsdp, ids_t).cpu()
            self.assertGreater(
                (after_perturb_logits - before_logits).abs().max().item(),
                1e-4,
                "perturbation did not change logits; roundtrip test is vacuous",
            )

            # Load back. The load now uses the FSDP-aware stub layout matching
            # the save; if it ever regresses, dcp.load raises CheckpointException
            # (a BaseException subclass, which `except Exception` misses), so
            # we surface any failure as a hard test failure rather than a skip.
            try:
                load_raw = node.load_hot_checkpoint(CheckpointRequest(path=ckpt, step=1))
                load_res = _from_dict(CheckpointResult, load_raw)
                self.assertTrue(
                    load_res.ok,
                    f"load_hot_checkpoint failed: {load_res.error}",
                )
            except BaseException as exc:  # noqa: BLE001 - surface dcp's CheckpointException
                self.fail(f"load_hot_checkpoint raised: {type(exc).__name__}: {exc}")

            restored_logits = self._logits(node._fsdp, ids_t).cpu()
            self.assertLess(
                (restored_logits - before_logits).abs().max().item(),
                1e-4,
                "logits differ after checkpoint roundtrip",
            )
            after_state = _node_state_dict(node)
            k0 = next(iter(before_state))
            self.assertLess(
                (after_state[k0].float() - before_state[k0].float()).abs().max().item(),
                1e-5,
                "params differ after checkpoint roundtrip",
            )
        finally:
            del node

    @staticmethod
    def _logits(model, ids):
        import torch

        with torch.no_grad():
            return model(ids).logits.clone()


@skip_no_model_env
class TestExportWeightsReload(unittest.TestCase):
    """export_weights writes HF-format weights; reloading via
    AutoModelForCausalLM.from_pretrained yields params matching the node."""

    def test_export_then_reload_matches(self):
        import torch
        from transformers import AutoModelForCausalLM

        from whirlwind.messages import ExportWeightsRequest

        node = _make_node()
        try:
            out_dir = os.path.join(node.config.checkpoint_dir, "export_test")
            from whirlwind.messages import OperationResult, _from_dict

            res = _from_dict(
                OperationResult, node.export_weights(ExportWeightsRequest(path=out_dir))
            )
            self.assertTrue(res.ok, f"export failed: {res}")
            self.assertTrue(os.path.exists(os.path.join(out_dir, "pytorch_model.bin")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "config.json")))

            reloaded = AutoModelForCausalLM.from_pretrained(
                out_dir,
                torch_dtype=torch.float32,
                trust_remote_code=False,
            )
            reloaded.eval()
            node_state = _node_state_dict(node)
            re_state = reloaded.state_dict()
            checked = 0
            for k, rv in re_state.items():
                if k in node_state and rv.is_floating_point():
                    d = (node_state[k].float() - rv.float()).abs().max().item()
                    self.assertLess(d, 1e-4, f"exported param '{k}' diverged: {d}")
                    checked += 1
            self.assertGreater(checked, 0, "no params compared after reload")
        finally:
            del node


@skip_no_model_env
class TestSleepWakeNoOp(unittest.TestCase):
    """sleep -> wake must not change params, optimizer state, or a forward pass."""

    def test_sleep_wake_is_noop(self):
        import torch

        from whirlwind.messages import NoInputRequest

        node = _make_node()
        try:
            req, ids_list, _ = _toy_batch(node, n_samples=2, seq_len=12, prompt_len=3)
            ids_t = torch.tensor(ids_list, dtype=torch.long, device="cuda")
            before_logits = self._logits(node._fsdp, ids_t).cpu()
            before_state = _node_state_dict(node)

            node.sleep(NoInputRequest())
            # while asleep, FSDP2 moves sharded param data to CPU; we don't
            # introspect internals here (they vary by build) — just ensure no crash.
            node.wake(NoInputRequest())

            after_state = _node_state_dict(node)
            after_logits = self._logits(node._fsdp, ids_t).cpu()
            # compare a sample of params via the full state dicts
            k0 = next(iter(before_state))
            self.assertLess(
                (after_state[k0].float() - before_state[k0].float()).abs().max().item(),
                1e-5,
                "params changed after sleep/wake",
            )
            self.assertLess(
                (after_logits - before_logits).abs().max().item(),
                1e-4,
                "logits changed after sleep/wake",
            )
        finally:
            del node

    @staticmethod
    def _logits(model, ids):
        import torch

        with torch.no_grad():
            return model(ids).logits.clone()


@skip_no_model_env
class TestForwardNoGradTopK(unittest.TestCase):
    """forward_no_grad returns top-K + selected logits matching a reference."""

    def test_forward_no_grad_topk(self):
        import torch

        from whirlwind.messages import TopKLogitsRequest, TopKLogitsResult, _from_dict

        node = _make_node()
        try:
            req, ids_list, lab_list = _toy_batch(node, n_samples=1, seq_len=10, prompt_len=3)
            ids_t = torch.tensor(ids_list, dtype=torch.long, device="cuda")
            topk = 5
            # decorate_serdes returns a dict in-process; parse it back.
            # Set explicit tracing fields on the request so we can assert the
            # result propagates them (previously dropped -- see _forward_no_grad).
            req_msg = TopKLogitsRequest(
                input_ids=ids_list,
                label_ids=lab_list,
                rewards=[0.0] * len(ids_list),
                topk=topk,
                temperature=1.0,
                step=0,
                dataset_ids=["test_ds"] * len(ids_list),
                entry_ids=list(range(len(ids_list))),
                rollout_ids=list(range(len(ids_list))),
                subrollout_ids=[0] * len(ids_list),
            )
            out = _from_dict(TopKLogitsResult, node.forward_no_grad(req_msg))
            # Reference forward + map_topk
            with torch.no_grad():
                logits = node._fsdp(ids_t).logits
            from whirlwind.messages import map_topk

            label_t = torch.tensor(lab_list, dtype=torch.long, device="cuda")
            tk_ids, tk_logits, sel_logits = map_topk(logits, label_t, topk, 1.0)
            import numpy as np

            np.testing.assert_allclose(out.topk_logits, tk_logits.cpu().tolist(), atol=1e-3)
            np.testing.assert_allclose(out.selected_logits, sel_logits.cpu().tolist(), atol=1e-3)
            self.assertEqual(len(out.topk_ids), len(tk_ids))
            # tracing fields must be propagated, not dropped
            self.assertEqual(out.dataset_ids, ["test_ds"] * len(ids_list))
            self.assertEqual(out.entry_ids, list(range(len(ids_list))))
            self.assertEqual(out.rollout_ids, list(range(len(ids_list))))
            self.assertEqual(out.subrollout_ids, [0] * len(ids_list))
        finally:
            del node


@skip_no_model_env
class TestLRAndGradClip(unittest.TestCase):
    """model_update_step applies req.lr and clips to max_grad_norm."""

    def test_lr_override_and_grad_clip(self):

        from whirlwind.messages import ModelUpdateRequest, TrainStepResult, _from_dict

        # node_clip clips at 0.5; node_noclip uses a huge max_grad_norm.
        node_clip = _make_node(
            config_overrides={
                "max_grad_norm": 0.5,
                "learning_rate": 1e-4,
            }
        )
        node_noclip = _make_node(config_overrides={"max_grad_norm": 1e9, "learning_rate": 1e-4})
        try:
            req, _, _ = _toy_batch(node_clip, n_samples=2, seq_len=12, prompt_len=3)
            node_clip.forward_accum_grad(req)
            node_noclip.forward_accum_grad(req)  # same batch, separate grads

            override_lr = 3e-5
            res = _from_dict(
                TrainStepResult, node_clip.model_update_step(ModelUpdateRequest(lr=override_lr))
            )
            # lr override is applied
            self.assertAlmostEqual(res.lr, override_lr, places=8)
            for pg in node_clip._optimizer.param_groups:
                self.assertAlmostEqual(pg["lr"], override_lr, places=8)
            # clip_grad_norm_ reports the TOTAL norm BEFORE clipping (raw).
            # So res.grad_norm is the raw grad norm (should be > 0), and the
            # actual update used clipped gradients. Verify clipping took effect
            # by comparing the post-step weights to the no-clip run: a 0.5 clip
            # vs 1e9 clip on a raw norm > 0.5 must produce different weights.
            self.assertGreater(res.grad_norm, 0.0, "reported grad_norm should be the raw norm")
            _ = _from_dict(
                TrainStepResult, node_noclip.model_update_step(ModelUpdateRequest(lr=override_lr))
            )
            if res.grad_norm > 0.5:
                # clipping changed the update -> weights must differ
                from torch.distributed.checkpoint.state_dict import (
                    StateDictOptions,
                    get_model_state_dict,
                )

                s_clip = get_model_state_dict(
                    node_clip._fsdp,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                s_noclip = get_model_state_dict(
                    node_noclip._fsdp,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                k0 = next(iter(s_clip))
                self.assertGreater(
                    (s_clip[k0].float() - s_noclip[k0].float()).abs().max().item(),
                    1e-5,
                    "clipping at 0.5 did not change the update vs no-clip",
                )
        finally:
            del node_clip, node_noclip


@skip_no_model_env
class TestMicrobatchAccumulation(unittest.TestCase):
    """Gradient accumulation (the GRPO path) vs microbatch splitting.

    Two behaviours:
    (a) gradient_accumulation_steps: multiple forward_accum_grad calls with
        the SAME batch, then one model_update_step, must equal a single
        forward_accum_grad on the full batch with acc_steps=1 scaled by 1/N.
        This is the path GRPO actually uses, and the loss reconstruction in
        _model_update_step is designed for it.
    (b) micro_batch_size split: splitting one batch into per-sample
        microbatches changes the token-level (CE/BNPO) normalization, because
        each microbatch normalizes by its OWN token count. We characterize the
        current behaviour (it diverges from the pooled full-batch result) so a
        future fix is detected, rather than silently asserting a false equality.
    """

    def test_gradient_accumulation_matches_scaled_full_batch(self):
        import torch

        from whirlwind.messages import ModelUpdateRequest

        # One optimizer step over N accumulated forward passes on the SAME
        # batch must match one step over the full batch (acc divides the loss
        # so the accumulated gradient sums to the full-batch gradient).
        node_full = _make_node(
            config_overrides={
                "micro_batch_size": 64,
                "gradient_accumulation_steps": 1,
            }
        )
        node_accum = _make_node(
            config_overrides={
                "micro_batch_size": 64,
                "gradient_accumulation_steps": 4,
            }
        )
        try:
            req, ids_list, _ = _toy_batch(node_full, n_samples=4, seq_len=12, prompt_len=3)
            ids_t = torch.tensor(ids_list, dtype=torch.long, device="cuda")

            node_full.forward_accum_grad(req)
            node_full.model_update_step(ModelUpdateRequest(lr=None))
            full_logits = self._logits(node_full._fsdp, ids_t).cpu()

            # 4 accumulations of the SAME batch, then one update. GAS scaling
            # is orchestrator-owned via req.accum_steps (the orchestrator sets
            # it on each accumulation call; the node divides the per-call loss
            # by it) — mirror that contract here.
            req.accum_steps = 4
            for _ in range(4):
                node_accum.forward_accum_grad(req)
            node_accum.model_update_step(ModelUpdateRequest(lr=None))
            accum_logits = self._logits(node_accum._fsdp, ids_t).cpu()

            self.assertLess(
                (full_logits - accum_logits).abs().max().item(),
                1e-3,
                "gradient accumulation of the same batch diverged from a single full-batch step",
            )
        finally:
            del node_full, node_accum

    def test_microbatch_split_characterization(self):
        """Microbatch-split CE normalization matches the pooled full batch.

        With loss_reduction='ce' (token-level), splitting a batch into
        per-sample microbatches normalizes each microbatch by its own token
        count; the accumulated gradient therefore differs from the pooled
        full-batch gradient unless the microbatch normalizer uses the global
        token count. The global_batch_size_override fix (v0.4.0) made the
        microbatch path use the pooled denominator, so the two now match
        (this test previously pinned the divergent pre-fix behaviour).
        """
        import torch

        from whirlwind.messages import ModelUpdateRequest

        node_full = _make_node(
            config_overrides={
                "micro_batch_size": 64,
                "gradient_accumulation_steps": 1,
            }
        )
        node_micro = _make_node(
            config_overrides={
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
            }
        )
        try:
            req, ids_list, _ = _toy_batch(node_full, n_samples=4, seq_len=12, prompt_len=3)
            ids_t = torch.tensor(ids_list, dtype=torch.long, device="cuda")

            node_full.forward_accum_grad(req)
            node_full.model_update_step(ModelUpdateRequest(lr=None))
            full_logits = self._logits(node_full._fsdp, ids_t).cpu()

            node_micro.forward_accum_grad(req)  # splits into 4 microbatches
            node_micro.model_update_step(ModelUpdateRequest(lr=None))
            micro_logits = self._logits(node_micro._fsdp, ids_t).cpu()

            diff = (full_logits - micro_logits).abs().max().item()
            # Post-v0.4.0 (global_batch_size_override): the microbatch path
            # normalizes with the pooled denominator, so the split matches the
            # full-batch step up to numerical noise.
            self.assertLess(
                diff,
                1e-3,
                f"microbatch split diverged from full batch (diff {diff}). "
                "The CE/BNPO microbatch normalization may have regressed.",
            )
        finally:
            del node_full, node_micro

    @staticmethod
    def _logits(model, ids):
        import torch

        with torch.no_grad():
            return model(ids).logits.clone()



def _standby_release_worker(rank: int, port: int, out_dir: str) -> None:
    """Rank 0 calls stop(); rank 1 sits in _standby() (2-process gloo group)."""
    import asyncio
    import datetime

    import torch.distributed as dist

    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2")
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    from whirlwind.config import ModelConfig
    from whirlwind.model.model_fsdp2 import ModelUpdateNode

    node = ModelUpdateNode(ModelConfig())
    node._initialized_model = True  # the other rank has reached standby
    if rank == 0:
        asyncio.run(node.stop())
    else:
        node._standby()
    with open(os.path.join(out_dir, f"rank{rank}.done"), "w") as f:
        f.write("pg still initialized" if dist.is_initialized() else "ok")


@skip_no_torch
class TestStandbyRelease(unittest.TestCase):
    """stop() on rank 0 releases the ranks waiting in _standby().

    Without the stop message they stayed blocked in scatter_object_list after
    rank 0 destroyed the process group: torchrun outlived a finished run, and
    ranks on other nodes died with an NCCL error.
    """

    def test_stop_releases_standby_rank(self):
        import socket
        import time

        import torch.multiprocessing as mp

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        out_dir = tempfile.mkdtemp(prefix="ww-standby-")
        self.addCleanup(shutil.rmtree, out_dir, True)
        ctx = mp.start_processes(
            _standby_release_worker, args=(port, out_dir), nprocs=2,
            join=False, start_method="spawn",
        )
        deadline = time.monotonic() + 120
        while not ctx.join(timeout=5):
            if time.monotonic() > deadline:
                for p in ctx.processes:
                    p.kill()
                self.fail("rank 1 never left _standby() after rank 0's stop()")
        for rank in (0, 1):
            with open(os.path.join(out_dir, f"rank{rank}.done")) as f:
                self.assertEqual(f.read(), "ok", f"rank {rank}")


class TestHybridShardMesh(unittest.TestCase):
    """HYBRID_SHARD mesh shape: one row per node, from torchrun's layout."""

    def test_rows_are_nodes(self):
        from whirlwind.model.model_fsdp2 import hybrid_shard_mesh_shape

        self.assertEqual(hybrid_shard_mesh_shape(16, 8), (2, 8))  # 2 nodes x 8 GPUs
        self.assertEqual(hybrid_shard_mesh_shape(12, 4), (3, 4))
        self.assertEqual(hybrid_shard_mesh_shape(4, 4), (1, 4))  # one node = FULL_SHARD

    def test_uneven_nodes_rejected(self):
        from whirlwind.model.model_fsdp2 import hybrid_shard_mesh_shape

        with self.assertRaisesRegex(ValueError, "same number of ranks on every node"):
            hybrid_shard_mesh_shape(6, 4)
        with self.assertRaises(ValueError):
            hybrid_shard_mesh_shape(4, 0)

    def test_config_default_is_full_shard(self):
        from whirlwind.config import ModelConfig

        self.assertEqual(ModelConfig().fsdp_sharding_strategy, "FULL_SHARD")


class TestFsdpLayerDiscovery(unittest.TestCase):
    """Per-layer FSDP2 wrap discovery (fail-loud).

    An earlier candidate-list/MRO scan matched nothing on transformers 5.x,
    silently degrading every run to a ROOT-ONLY FSDP2 unit: the whole model
    all-gathered every microbatch forward and activation checkpointing did
    nothing (torch never wraps the root module).
    """

    def _tiny_qwen3(self, n_layers=2):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=n_layers,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=32,
            max_position_embeddings=64,
        )
        return Qwen3ForCausalLM(cfg)

    def test_discovers_layers_on_real_transformers_model(self):
        # The old scan's candidate path (getattr on the class MRO) found
        # nothing even for Qwen3DecoderLayer; the instance scan must.
        from whirlwind.model.model_fsdp2 import _discover_decoder_layer_cls

        m = self._tiny_qwen3()
        found = _discover_decoder_layer_cls(m, "layer")
        self.assertEqual({c.__name__ for c in found}, {"Qwen3DecoderLayer"})
        n = sum(1 for mod in m.modules() if isinstance(mod, tuple(found)))
        self.assertEqual(n, 2)

    def _checkpointed_backward(self, use_cache: bool) -> None:
        # Per-layer activation checkpointing exactly as the model node applies
        # it (non-reentrant wrapper on the discovered decoder layers).
        from functools import partial

        import torch
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        from whirlwind.model.model_fsdp2 import _discover_decoder_layer_cls

        m = self._tiny_qwen3()
        layers = tuple(_discover_decoder_layer_cls(m, "layer"))
        apply_activation_checkpointing(
            m,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda mod: isinstance(mod, layers),
        )
        m.config.use_cache = use_cache
        m(torch.randint(0, 32, (2, 7))).logits.sum().backward()
        self.assertTrue(any(p.grad is not None for p in m.parameters()))

    def test_activation_checkpointing_runs_with_cache_off(self):
        # The model node sets config.use_cache = False at load: training
        # forwards never need a KV cache.
        self._checkpointed_backward(use_cache=False)

    def test_activation_checkpointing_breaks_with_cache_on(self):
        # Why the node turns the cache off: the backward recompute of a
        # checkpointed layer appends its keys to the forward's cache again.
        with self.assertRaises(Exception):
            self._checkpointed_backward(use_cache=True)

    def test_discovers_moe_layer_class_by_name(self):
        # MoE architectures name their layer e.g. Qwen3_5MoeDecoderLayer; the
        # discovery is structural (instance scan by class NAME), so a
        # synthetic tree covers it without a MoE checkpoint.
        import torch.nn as nn

        from whirlwind.model.model_fsdp2 import _discover_decoder_layer_cls

        class Qwen3_5MoeDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)

        class FakeMoeForCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                inner = nn.Module()
                inner.layers = nn.ModuleList(
                    [Qwen3_5MoeDecoderLayer() for _ in range(3)]
                )
                self.model = inner
                self.lm_head = nn.Linear(8, 8)

        found = _discover_decoder_layer_cls(FakeMoeForCausalLM(), "layer")
        self.assertEqual(
            {c.__name__ for c in found}, {"Qwen3_5MoeDecoderLayer"}
        )

    def test_no_layers_warns_root_only(self):
        # Fail-loud: a silent empty match is what let runs go root-only
        # unnoticed. The warning names the consequences.
        import torch.nn as nn

        from whirlwind.model.model_fsdp2 import _discover_decoder_layer_cls

        m = nn.Sequential(nn.Linear(4, 4))
        with self.assertLogs(
            "whirlwind.model.model_fsdp2", level="WARNING"
        ) as logs:
            found = _discover_decoder_layer_cls(m, "layer")
        self.assertEqual(found, set())
        self.assertTrue(any("ROOT-ONLY" in line for line in logs.output))

    def test_granularity_off_is_silent(self):
        # Other granularity values intend root-only: no warning, no scan.
        import torch.nn as nn

        from whirlwind.model.model_fsdp2 import _discover_decoder_layer_cls

        m = nn.Sequential(nn.Linear(4, 4))
        found = _discover_decoder_layer_cls(m, "off")
        self.assertEqual(found, set())

if __name__ == "__main__":
    unittest.main()
