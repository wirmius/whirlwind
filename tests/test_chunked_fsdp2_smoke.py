"""FSDP2 smoke test for the chunked lm_head forward (2 GPUs).

Replicates ModelUpdateNode's chunked-mode init (FSDP2: shard each decoder block,
then wrap the body and lm_head as SEPARATE units so the lm_head can run in token
chunks with a grad boundary at the body's last hidden state), then runs the
chunked forward + the STANDARD loss (called once, unmodified) + backward, for
BOTH grpo and dr_grpo reductions.

Asserts finite loss and non-zero parameter gradients on both ranks. This
validates the body/lm_head FSDP2 split + backward through the chunked lm_head
that the plain-model equivalence test (test_chunked_lm_head.py) cannot cover.

Run:
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 tests/test_chunked_fsdp2_smoke.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def _run_reduction(reduction: str, stub, body, lm_head, tensors, dev) -> torch.Tensor:
    from whirlwind.model.model_fsdp2 import ModelUpdateNode

    stub.config.loss_reduction = reduction
    pc, attn, comp_ids, cmask = (
        tensors["pc"], tensors["attn"], tensors["comp_ids"], tensors["cmask"],
    )
    comp_len = cmask.sum(-1)
    total_tokens = int(comp_len.sum().item())
    adv = tensors["adv"]

    # chunked forward: body -> hidden -> chunked lm_head -> assembled [B, C] logps
    stub._chunked = True
    stub._body_module = body
    stub._lm_head_module = lm_head
    ptl, _ent = ModelUpdateNode._compute_per_token_logps(
        stub, model=None, prompt_completion_ids=pc, attention_mask=attn,
        completion_ids=comp_ids, logits_to_keep=comp_ids.size(1), temperature=1.0,
        compute_entropy=False,
    )
    stub._accum_buf = {"diag": []}
    loss = ModelUpdateNode._compute_loss(
        stub, per_token_logps=ptl, old_per_token_logps=ptl.detach(),
        ref_per_token_logps=None, advantages=adv, mask=cmask,
        global_batch_size_override=tensors["B"], global_max_completion_len_override=tensors["C"],
        completion_len=comp_len, prompt_len=tensors["prompt_len"],
        total_tokens=total_tokens, sampling_per_token_logps=tensors["samp"],
        entropy=None,
    )
    return loss


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
        rope_theta=10000.0, tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).to(dev)
    body = model.model
    lm_head = model.lm_head

    mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world,))
    # shard each decoder block, then body + lm_head separately (chunked-mode wrap)
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    for child in list(body.children()):
        if isinstance(child, LlamaDecoderLayer):
            fully_shard(child, mesh=mesh)
    fully_shard(body, mesh=mesh)
    fully_shard(lm_head, mesh=mesh)

    B, P, C = 4, 5, 8
    torch.manual_seed(100 + rank)
    ids = torch.randint(2, 64, (B, P + C), device=dev)
    comp = ids[:, P:]
    attn = torch.ones(B, P + C, device=dev)
    cmask = torch.ones(B, C, device=dev)
    cmask[0, -2:] = 0.0  # sample 0 has a shorter completion (right-pad)
    adv = torch.tensor([1.0, -0.5, 0.3, 0.8], device=dev)
    samp = (torch.randn(B, C, device=dev) * 0.1).float()
    prompt_len = torch.full((B,), float(P), device=dev)

    tensors = dict(
        pc=ids, attn=attn, comp_ids=comp, cmask=cmask, adv=adv, samp=samp,
        prompt_len=prompt_len, B=B, C=C,
    )

    class Stub:
        local_rank = rank
        _use_autocast = False
        _logits_fp32 = False
        _is_fp32 = False
        _loss_fp32 = False
        _compute_entropy = False
        _chunked = True
        _body_module = body
        _lm_head_module = lm_head
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
            max_completion_length = C
            temperature = 1.0
            logits_chunk_size = 3  # C=8 -> chunks of 3,3,2

        _accum_buf = {"diag": []}

    stub = Stub()

    for reduction in ("grpo", "dr_grpo"):
        # zero grads on the body params we will inspect
        for p in body.parameters():
            if p.grad is not None:
                p.grad = None

        loss = _run_reduction(reduction, stub, body, lm_head, tensors, dev)
        loss.backward()

        assert torch.isfinite(loss).all(), f"[{reduction}] rank {rank}: loss not finite"
        # at least one body parameter received a non-zero gradient
        nz = sum(
            1 for p in body.parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )
        assert nz > 0, f"[{reduction}] rank {rank}: no non-zero body grads (got {nz})"
        print(f"[{reduction}] rank {rank}: loss={loss.item():.6f}  nonzero_grad_params={nz}",
              flush=True)

    dist.barrier()
    if rank == 0:
        print("FSDP2 chunked lm_head smoke: PASS (grpo + dr_grpo, finite loss + grads)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
