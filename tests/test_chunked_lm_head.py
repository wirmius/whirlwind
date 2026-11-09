"""Correctness test: chunked lm_head forward == non-chunked forward.

Pins the design where chunking lives in the FORWARD only (the lm_head runs in
token chunks so the full [B, C, V] logits are never materialized) and the
STANDARD loss is called once on the assembled [B, C] per-token logps, unmodified.

Asserts that, for a tiny real model:
  1. _compute_per_token_logps_chunked produces the SAME per-token logps and
     entropy as the non-chunked _compute_per_token_logps (chunking is
     numerically transparent).
  2. The standard compute_grpo_loss_core (via _compute_loss) yields the SAME
     scalar loss for grpo AND dr_grpo whether the logps came from the chunked
     or non-chunked forward -- i.e. the loss is chunk-agnostic and the global
     normalizers from the waterfall make the reductions correct.

No FSDP is needed here: the chunked path just calls model.model (body) and
model.lm_head directly, which works on a plain (non-sharded) model.
"""
from __future__ import annotations

import unittest

from tests._helpers import skip_no_gpu, skip_no_transformers


@skip_no_gpu
@skip_no_transformers
class TestChunkedLmHeadForward(unittest.TestCase):
    def _make_model_and_inputs(self):
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
            _chunked = False
            _body_module = None
            _lm_head_module = None
            _selective_log_softmax = staticmethod(ModelUpdateNode._selective_log_softmax)
            _compute_per_token_logps_chunked = ModelUpdateNode._compute_per_token_logps_chunked

            class config:
                kl_coef = 0.0
                clip_range_low = 0.2
                clip_range_high = 0.2
                loss_reduction = "grpo"
                importance_sampling_level = "token"
                engine_is_correction = False
                engine_is_mode = "token_truncate"
                engine_is_cap = 2.0
                max_completion_length = 3
                temperature = 1.0
                logits_chunk_size = 2

            _accum_buf = {"diag": []}

        stub = Stub()
        # two samples, different prompt & completion lengths; tokens avoid PAD(0)
        input_ids = [[5, 6, 7] + [11, 12], [8, 9] + [13, 14, 15]]
        label_ids = [[-100, -100, -100, 11, 12], [-100, -100, 13, 14, 15]]
        sampling_lp = [[0.1, 0.2], [0.3, 0.4, 0.5]]
        t = ModelUpdateNode._pad_and_stack(
            stub, input_ids, label_ids, advantages=[1.0, -1.0],
            sampling_logprobs=sampling_lp, ref_logprobs=None, old_logprobs=None,
        )
        return model, stub, t

    def test_chunked_logps_match_nonchunked(self):
        import torch
        from whirlwind.model.model_fsdp2 import ModelUpdateNode

        model, stub, t = self._make_model_and_inputs()
        pc, attn, comp_ids, cmask, cmax = (
            t["prompt_completion_ids"], t["attention_mask"], t["completion_ids"],
            t["completion_mask"], t["logits_to_keep"],
        )

        # --- non-chunked forward ---
        stub._chunked = False
        with torch.no_grad():
            ptl_nc, ent_nc = ModelUpdateNode._compute_per_token_logps(
                stub, model=model, prompt_completion_ids=pc, attention_mask=attn,
                completion_ids=comp_ids, logits_to_keep=cmax, temperature=1.0,
                compute_entropy=True,
            )

        # --- chunked forward (body + lm_head separately, token chunks of 2) ---
        stub._chunked = True
        stub._body_module = model.model
        stub._lm_head_module = model.lm_head
        with torch.no_grad():
            ptl_ch, ent_ch = ModelUpdateNode._compute_per_token_logps(
                stub, model=model, prompt_completion_ids=pc, attention_mask=attn,
                completion_ids=comp_ids, logits_to_keep=cmax, temperature=1.0,
                compute_entropy=True,
            )

        # 1) chunking is numerically transparent on the per-token logps + entropy
        self.assertEqual(ptl_ch.shape, ptl_nc.shape)
        self.assertTrue(torch.allclose(ptl_ch, ptl_nc, atol=1e-6),
                        f"chunked vs non-chunked logps differ: max "
                        f"{(ptl_ch - ptl_nc).abs().max().item()}")
        self.assertTrue(torch.allclose(ent_ch, ent_nc, atol=1e-6),
                        f"chunked vs non-chunked entropy differ: max "
                        f"{(ent_ch - ent_nc).abs().max().item()}")

        # 2) the standard loss is chunk-agnostic for grpo AND dr_grpo
        comp_len = t["completion_len"]
        total_tokens = int(comp_len.sum().item())
        adv = torch.tensor([1.0, -1.0]).cuda()
        for reduction in ("grpo", "dr_grpo"):
            stub.config.loss_reduction = reduction
            stub._accum_buf = {"diag": []}
            loss_nc = ModelUpdateNode._compute_loss(
                stub, per_token_logps=ptl_nc, old_per_token_logps=ptl_nc.detach(),
                ref_per_token_logps=None, advantages=adv, mask=cmask,
                global_batch_size_override=2, global_max_completion_len_override=3,
                completion_len=comp_len, prompt_len=t["prompt_len"],
                total_tokens=total_tokens,
                sampling_per_token_logps=t["sampling_logprobs"], entropy=ent_nc,
            )
            stub._accum_buf = {"diag": []}
            loss_ch = ModelUpdateNode._compute_loss(
                stub, per_token_logps=ptl_ch, old_per_token_logps=ptl_ch.detach(),
                ref_per_token_logps=None, advantages=adv, mask=cmask,
                global_batch_size_override=2, global_max_completion_len_override=3,
                completion_len=comp_len, prompt_len=t["prompt_len"],
                total_tokens=total_tokens,
                sampling_per_token_logps=t["sampling_logprobs"], entropy=ent_ch,
            )
            self.assertTrue(torch.isfinite(loss_nc), f"{reduction}: non-chunked loss not finite")
            self.assertTrue(torch.isfinite(loss_ch), f"{reduction}: chunked loss not finite")
            self.assertTrue(torch.allclose(loss_ch, loss_nc, atol=1e-6),
                            f"{reduction}: chunked loss {loss_ch.item():.6f} != "
                            f"non-chunked {loss_nc.item():.6f}")


if __name__ == "__main__":
    unittest.main()
