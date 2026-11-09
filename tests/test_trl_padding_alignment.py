"""Regression test: TRL-aligned _pad_and_stack + _compute_per_token_logps.

Pins the TRL-alignment refactor (grpo_trainer.py:1873-1931, 1108-1124):
  - _pad_and_stack splits pre-joined input_ids/label_ids at the -100 boundary,
    LEFT-pads prompts, RIGHT-pads completions, concats -> prompt_completion_ids,
    and right-pads engine logprobs/advantages to C_max (completion-aligned by
    construction -- no per-sample prompt_len offset, no prepend hack).
  - _compute_per_token_logps slices logits to the completion span and returns
    [B, C] completion-shaped logps.

Verified by calling the REAL ModelUpdateNode methods via a stub on a tiny RoPE
Llama model (RoPE + attention_mask-derived position_ids, same family as Qwen3),
and comparing the completion-span logprobs against a no-padding reference.
Also pins the engine-logprob length assert (the previously-unchecked contract)
and that _compute_loss runs end-to-end with engine IS correction on [B, C] tensors.
"""
from __future__ import annotations

import unittest

from tests._helpers import skip_no_gpu, skip_no_transformers


@skip_no_gpu
@skip_no_transformers
class TestTRLPaddingAlignment(unittest.TestCase):
    def test_completion_logprobs_match_no_pad_reference(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        torch.manual_seed(0)
        cfg = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
            rope_theta=10000.0, tie_word_embeddings=True,
        )
        model = LlamaForCausalLM(cfg).eval().cuda()
        PAD = 0

        class Stub:
            local_rank = 0
            _tokenizer = type("T", (), {"pad_token_id": PAD})()
            _use_autocast = False
            _logits_fp32 = False
            _is_fp32 = False
            _loss_fp32 = False
            _compute_entropy = True
            _selective_log_softmax = staticmethod(ModelUpdateNode._selective_log_softmax)

            class config:
                kl_coef = 0.0
                clip_range_low = 0.2
                clip_range_high = 0.2
                loss_reduction = "grpo"
                importance_sampling_level = "token"
                engine_is_correction = True
                engine_is_mode = "token_truncate"
                engine_is_cap = 2.0
                max_completion_length = 3
            _accum_buf = {"diag": []}
        stub = Stub()

        # two samples, different prompt & completion lengths; tokens avoid PAD(0)
        input_ids = [[5, 6, 7] + [11, 12], [8, 9] + [13, 14, 15]]
        label_ids = [[-100, -100, -100, 11, 12], [-100, -100, 13, 14, 15]]
        sampling_lp = [[0.1, 0.2], [0.3, 0.4, 0.5]]

        # no-padding reference: forward each sample alone, gather completion logps
        refs = []
        for inp, lab in zip(input_ids, label_ids):
            ids = torch.tensor([inp], dtype=torch.long).cuda()
            with torch.no_grad():
                lg = model(ids).logits.float()[:, :-1, :]
            pl = sum(1 for x in lab if x == -100)
            comp = inp[pl:]
            pos = torch.tensor([pl + j - 1 for j in range(len(comp))]).cuda()
            comp_t = torch.tensor(comp, dtype=torch.long).cuda()
            chosen = lg[0, pos, :].gather(-1, comp_t.unsqueeze(-1)).squeeze(-1)
            refs.append((chosen - torch.logsumexp(lg[0, pos, :], dim=-1)).tolist())

        # REAL _pad_and_stack
        t = ModelUpdateNode._pad_and_stack(
            stub, input_ids, label_ids, advantages=[1.0, -1.0],
            sampling_logprobs=sampling_lp, ref_logprobs=None, old_logprobs=None,
        )
        pc, attn, comp_ids, cmask, cmax = (t["prompt_completion_ids"], t["attention_mask"],
                                           t["completion_ids"], t["completion_mask"], t["logits_to_keep"])
        self.assertEqual(pc.shape, (2, 6))
        self.assertEqual(comp_ids.shape, (2, 3))
        self.assertEqual(cmask.shape, (2, 3))
        # left-pad prompt: sample1 prompt len 2 < p_max 3 -> [0,1,1,1,1,1]
        self.assertEqual(attn[1].tolist(), [0.0, 1, 1, 1, 1, 1])
        self.assertEqual(cmask[0].tolist(), [1.0, 1.0, 0.0])  # right-pad completion
        self.assertEqual(comp_ids[0].tolist(), [11, 12, PAD])
        # engine logprobs right-aligned at completion span
        self.assertEqual([round(x, 3) for x in t["sampling_logprobs"][0].tolist()], [0.1, 0.2, 0.0])

        # REAL _compute_per_token_logps
        with torch.no_grad():
            ptl, ent = ModelUpdateNode._compute_per_token_logps(
                stub, model=model, prompt_completion_ids=pc, attention_mask=attn,
                completion_ids=comp_ids, logits_to_keep=cmax, temperature=1.0, compute_entropy=True,
            )
        self.assertEqual(ptl.shape, (2, 3))
        self.assertEqual(ent.shape, (2, 3))

        # completion logprobs must EXACTLY match the no-pad reference
        max_abs = 0.0
        for i in range(2):
            c = int(cmask[i].sum().item())
            for g, r in zip(ptl[i, :c].tolist(), refs[i]):
                max_abs = max(max_abs, abs(g - r))
        self.assertLess(max_abs, 1e-4, f"completion logprobs misaligned (max diff {max_abs})")

        # _compute_loss end-to-end with engine IS on, [B, C] tensors
        loss = ModelUpdateNode._compute_loss(
            stub, per_token_logps=ptl, old_per_token_logps=ptl.detach(),
            ref_per_token_logps=None, advantages=torch.tensor([1.0, -1.0]).cuda(),
            mask=cmask, global_batch_size_override=0, global_max_completion_len_override=0,
            completion_len=t["completion_len"], prompt_len=t["prompt_len"],
            sampling_per_token_logps=t["sampling_logprobs"], entropy=ent,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("sampling/importance_sampling_ratio/mean", stub._accum_buf["diag"][0])

    def test_nan_pads_engine_logprob_length_mismatch(self):
        """A short engine-logprob list is NaN-padded to comp_len (no crash).

        The rollout intermittently returns a logprob count != completion length
        (vLLM: 0 logprobs for a non-empty completion). The model node defensively
        NaN-pads shortfalls / truncates extras so the IS ratio is neutralized to
        1 (no annihilation) instead of crashing or silently zeroing.
        """
        import torch  # noqa: F401
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        class Stub:
            local_rank = 0
            _tokenizer = type("T", (), {"pad_token_id": 0})()
            _use_autocast = False
            _selective_log_softmax = staticmethod(ModelUpdateNode._selective_log_softmax)
        stub = Stub()
        input_ids = [[5, 6, 7, 11, 12], [8, 9, 13, 14, 15]]
        label_ids = [[-100, -100, -100, 11, 12], [-100, -100, 13, 14, 15]]
        # sample0: completion len 2 but only 1 logprob; sample1: correct (3).
        # Must NOT crash; sample0's shortfall NaN-padded to len 2.
        t = ModelUpdateNode._pad_and_stack(
            stub, input_ids, label_ids, [1.0, -1.0],
            sampling_logprobs=[[0.1], [0.3, 0.4, 0.5]],
            ref_logprobs=None, old_logprobs=None,
        )
        samp = t["sampling_logprobs"]
        self.assertEqual(samp.shape, (2, 3))  # c_max = 3
        # sample0: [0.1, NaN, 0.0] (1 real + 1 NaN-pad + 1 right-pad)
        self.assertAlmostEqual(samp[0, 0].item(), 0.1, places=3)
        self.assertTrue(torch.isnan(samp[0, 1]).item())
        self.assertEqual(samp[0, 2].item(), 0.0)
        # sample1: [0.3, 0.4, 0.5] (correct, no padding)
        self.assertEqual([round(x, 3) for x in samp[1].tolist()], [0.3, 0.4, 0.5])


if __name__ == "__main__":
    unittest.main()
