"""E2E: real ModelUpdateNode (Qwen3-0.6B + single-rank FSDP2) — chunked vs
non-chunked lm_head forward, for grpo AND dr_grpo.

For each reduction: build two real ModelUpdateNodes (one with logits_chunk_size=0,
one with logits_chunk_size>0), align them to identical initial weights, run the
same TrainStepRequest through forward_accum_grad + model_update_step, and assert
the loss and post-update weights match. This validates the REAL config-driven
FSDP2 init (body/lm_head wrapped separately when logits_chunk_size>0) + the real
_forward_accum_grad -> _compute_per_token_logps chunked dispatch + backward +
optimizer, for both reductions. Chunking is forward-only; the standard loss is
called once (unmodified) with global normalizers.

Tier 1 (GPU + Qwen3-0.6B). Single-rank FSDP2 is still real FSDP2 (DTensors,
shard/gather paths). The post-update weight diff is fp32 AdamW noise from two
independent optimizer states (same 1e-3 tolerance as TestReferenceWeightUpdate).
"""
from __future__ import annotations

import os
import unittest

from tests._helpers import skip_no_model_env


@skip_no_model_env
class TestChunkedE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29577")
        import torch
        from transformers import AutoModelForCausalLM
        from tests._helpers import model_path

        torch.manual_seed(123)
        ref = AutoModelForCausalLM.from_pretrained(
            model_path(), torch_dtype=torch.float32, attn_implementation="sdpa",
            trust_remote_code=False,
        )
        for a in ("dropout", "attention_dropout", "hidden_dropout", "emb_dropout"):
            if hasattr(ref.config, a):
                setattr(ref.config, a, 0.0)
        cls.init_state = {k: v.clone() for k, v in ref.state_dict().items()}
        del ref

    def _make(self, reduction, chunk_size):
        from tests.test_model_update import _make_node
        from tests._helpers import model_path
        base = dict(
            model_path=model_path(), torch_dtype="float32",
            fsdp_cpu_offload=False, fsdp_reshard_after_forward=True,
            fsdp_activation_checkpointing=False, fsdp_mixed_precision=False,
            disable_dropout=True, torch_compile=False, optim="adamw",
            learning_rate=1e-4, weight_decay=0.0, max_grad_norm=1000.0,
            gradient_accumulation_steps=1, micro_batch_size=64, kl_coef=0.0,
            importance_sampling_level="token", engine_is_correction=False,
            temperature=1.0, max_completion_length=16,
        )
        return _make_node(
            config_overrides=dict(base, loss_reduction=reduction, logits_chunk_size=chunk_size)
        )

    def _align(self, node):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, set_model_state_dict,
        )
        sd = {k: v.clone() for k, v in self.init_state.items()}
        set_model_state_dict(
            node._fsdp, model_state_dict=sd,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    def _batch(self):
        import torch
        from whirlwind.messages import TrainStepRequest
        torch.manual_seed(999)
        ids_list, lab_list = [], []
        for _ in range(2):
            ids = torch.randint(2, 2000, (16,)).tolist()
            ids_list.append(ids)
            lab_list.append([-100] * 4 + ids[4:])
        req = TrainStepRequest(
            input_ids=ids_list, label_ids=lab_list, rewards=[0.0, 0.0],
            advantages=[1.0, -0.5], prompts=[], completions=[],
        )
        req.global_batch_size_override = 2
        req.global_max_completion_len_override = 16
        return req

    def _run(self, reduction):
        import torch
        from whirlwind.messages import ModelUpdateRequest
        from tests.test_model_update import _node_state_dict

        node_nc = self._make(reduction, 0)
        node_ch = self._make(reduction, 4)  # completion len 12 -> chunks 4,4,4
        try:
            self._align(node_nc); self._align(node_ch)
            req = self._batch()
            l_nc = node_nc.forward_accum_grad(req)["loss"]
            l_ch = node_ch.forward_accum_grad(req)["loss"]
            self.assertTrue(torch.isfinite(torch.tensor(l_nc)), f"{reduction}: nc loss not finite")
            self.assertTrue(torch.isfinite(torch.tensor(l_ch)), f"{reduction}: ch loss not finite")
            self.assertLess(abs(l_nc - l_ch), 1e-4,
                            f"{reduction}: chunked loss {l_ch} != non-chunked {l_nc}")
            node_nc.model_update_step(ModelUpdateRequest(lr=None))
            node_ch.model_update_step(ModelUpdateRequest(lr=None))
            s_nc = _node_state_dict(node_nc)
            s_ch = _node_state_dict(node_ch)
            max_post = max(
                (s_nc[k].float() - s_ch[k].float()).abs().max().item()
                for k in s_nc if s_nc[k].is_floating_point()
            )
            self.assertLess(max_post, 1e-3,
                            f"{reduction}: post-update weights diverged (max diff {max_post})")
        finally:
            del node_nc, node_ch
            torch.cuda.empty_cache()

    def test_grpo_chunked_matches_nonchunked(self):
        self._run("grpo")

    def test_dr_grpo_chunked_matches_nonchunked(self):
        self._run("dr_grpo")


if __name__ == "__main__":
    unittest.main()
