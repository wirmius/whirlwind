"""FSDP2 smoke test for chunked lm_head with TIED embeddings (2 GPUs).

Qwen3-0.6B ties lm_head.weight to embed_tokens.weight — they CANNOT be in
separate FSDP2 units. The tied path keeps the STANDARD root wrap (no body/lm_head
split) and monkey-patches the lm_head to chunk + selective_log_softmax INSIDE
the root's FSDP forward (where the DTensor weight is usable).

Validates: finite loss, non-zero grads (backward through the tied weight), and
FSDP state API compatibility (_get_fsdp_state, .parameters) for the full
colocate pipeline (sleep/wake, grad clip, export).

Run:
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 tests/test_chunked_fsdp2_tied_smoke.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    from transformers import LlamaConfig, LlamaForCausalLM
    from torch.distributed.fsdp import fully_shard
    from whirlwind.model.model_fsdp2 import ModelUpdateNode

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
        rope_theta=10000.0, tie_word_embeddings=True,  # <-- TIED
    )
    model = LlamaForCausalLM(cfg).to(dev)
    assert model.lm_head.weight is model.model.embed_tokens.weight, "weights not tied!"

    mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world,))
    # Standard root wrap (no body/lm_head split for tied)
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    for child in list(model.model.children()):
        if isinstance(child, LlamaDecoderLayer):
            fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)  # root wrap (owns embed_tokens + lm_head, tied)

    # --- validate FSDP state API (sleep/wake, grad clip, export compatibility) ---
    fsdp_state = model._get_fsdp_state()
    assert fsdp_state is not None, "root _get_fsdp_state() failed"
    n_params = sum(1 for _ in model.parameters())
    assert n_params > 0, "root .parameters() returned nothing"

    B, P, C = 4, 5, 8
    torch.manual_seed(100 + rank)
    ids = torch.randint(2, 64, (B, P + C), device=dev)
    comp = ids[:, P:]
    attn = torch.ones(B, P + C, device=dev)
    cmask = torch.ones(B, C, device=dev)
    cmask[0, -2:] = 0.0
    adv = torch.tensor([1.0, -0.5, 0.3, 0.8], device=dev)
    samp = (torch.randn(B, C, device=dev) * 0.1).float()
    prompt_len = torch.full((B,), float(P), device=dev)

    class Stub:
        local_rank = rank
        _use_autocast = False
        _logits_fp32 = False
        _is_fp32 = False
        _loss_fp32 = False
        _compute_entropy = False
        _chunked = True
        _chunked_tied = True
        _body_module = model.model
        _lm_head_module = model.lm_head
        _mesh = mesh
        _selective_log_softmax = staticmethod(ModelUpdateNode._selective_log_softmax)
        _compute_per_token_logps_chunked = ModelUpdateNode._compute_per_token_logps_chunked
        _compute_per_token_logps_chunked_tied = ModelUpdateNode._compute_per_token_logps_chunked_tied

        class config:
            kl_coef = 0.0
            clip_range_low = 0.2
            clip_range_high = 0.2
            loss_reduction = "grpo"
            importance_sampling_level = "token"
            engine_is_correction = False
            engine_is_mode = "token_truncate"
            engine_is_cap = 2.0
            max_completion_length = C
            temperature = 1.0
            logits_chunk_size = 3

        _accum_buf = {"diag": []}

    stub = Stub()

    for reduction in ("grpo", "dr_grpo"):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        stub.config.loss_reduction = reduction
        # Call through the real dispatch (model = root FSDP module)
        ptl, _ = ModelUpdateNode._compute_per_token_logps(
            stub, model=model, prompt_completion_ids=ids, attention_mask=attn,
            completion_ids=comp, logits_to_keep=comp.size(1), temperature=1.0,
            compute_entropy=False,
        )
        comp_len = cmask.sum(-1)
        total_tokens = int(comp_len.sum().item())
        stub._accum_buf = {"diag": []}
        loss = ModelUpdateNode._compute_loss(
            stub, per_token_logps=ptl, old_per_token_logps=ptl.detach(),
            ref_per_token_logps=None, advantages=adv, mask=cmask,
            global_batch_size_override=B, global_max_completion_len_override=C,
            completion_len=comp_len, prompt_len=prompt_len,
            total_tokens=total_tokens, sampling_per_token_logps=samp, entropy=None,
        )
        loss.backward()

        assert torch.isfinite(loss).all(), f"[{reduction}] rank {rank}: loss not finite"
        # grads flowed through chunked lm_head -> tied weight -> body
        nz = sum(1 for p in model.parameters()
                 if p.grad is not None and p.grad.abs().sum().item() > 0)
        assert nz > 0, f"[{reduction}] rank {rank}: no non-zero grads (got {nz})"
        print(f"[{reduction}] rank {rank}: loss={loss.item():.6f}  nonzero_grad_params={nz}",
              flush=True)

    dist.barrier()
    if rank == 0:
        print("FSDP2 chunked TIED smoke: PASS (grpo + dr_grpo, state API OK, grads OK)",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
