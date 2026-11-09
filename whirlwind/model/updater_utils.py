"""Updater utilities — GRPO loss math + tensor assembly, infrastructure-independent.

This module holds the loss-computation and tensor-assembly code that is shared
across the updater implementations (``model_fsdp2.ModelUpdateNode`` and its
subclasses ``model_updater_hf_trainer.HFTrainerModelUpdateNode`` and
``model_accelerate_pure.AcceleratePureModelUpdateNode``). These functions depend
only on tensors + config scalars/flags — NOT on FSDP, the RPC node, the
tokenizer (beyond ``pad_token_id``), or the diagnostics accumulator. The
updaters keep thin method wrappers that pull device/pad_token_id/config from
``self`` and delegate here (so call sites, inheritance, and the
``tests/test_trl_padding_alignment.py`` regression that calls the unbound
methods are unchanged).

Contents
--------
- :func:`selective_log_softmax`   — memory-efficient chosen-token logprob (TRL-aligned).
- :func:`pad_and_stack`           — TRL-aligned left-pad prompt / right-pad completion assembly.
- :func:`pad_full_sequence`       — right-pad pre-joined seqs (eval / full-seq path).
- :func:`compute_grpo_loss_core`  — PPO clipped surrogate + engine IS + KL + reduction.
- :func:`compute_engine_is_ratio` — detached engine IS ratio (needs only [B,C] logprobs).

The chunked lm_head forward (token-axis logits chunking via a grad boundary
at the body's last hidden state) is wired in
``ModelUpdateNode._compute_per_token_logps_chunked`` and uses
:func:`selective_log_softmax` per token chunk. The standard loss is called
once on the assembled [B, C] logps (unmodified) with global normalizers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

# Warn-once state for pad_and_stack's logprob-length-mismatch warning, keyed by
# an arbitrary caller-supplied identity (e.g. id(node)). Keeps the util
# stateless w.r.t. the node while preserving the once-per-process warning.
_LP_MISMATCH_WARNED: set[int] = set()


def selective_log_softmax(logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    """Compute log-probability of chosen tokens (per-token, memory-efficient).

    Only computes log-softmax at the positions of the chosen tokens, avoiding a
    full softmax over the entire vocabulary.

    Follows TRL's dtype-aware strategy:
    - float32/float64: uses logsumexp (numerically accurate)
    - bfloat16/float16: uses max-subtraction (logsumexp is unstable in low precision)

    Args:
        logits:    [batch, seq_len, vocab_size]
        label_ids: [batch, seq_len] chosen token ids

    Returns:
        per_token_logps: [batch, seq_len]
    """
    # Gather chosen logits first (cheap gather, no softmax needed)
    chosen_logits = logits.gather(dim=-1, index=label_ids.unsqueeze(-1).clamp(min=0)).squeeze(
        -1
    )  # [batch, seq_len]

    if logits.dtype in (torch.float32, torch.float64):
        # logsumexp is accurate in high precision.
        # Loop over batch to limit peak memory (one logsumexp at a time).
        logsumexp_values = torch.stack(
            [torch.logsumexp(lg, dim=-1) for lg in logits]
        )  # [batch, seq_len]
        return chosen_logits - logsumexp_values
    else:
        # bfloat16/float16: match TRL's selective_log_softmax bf16 path
        # EXACTLY (trl/trainer/utils.py). torch.log_softmax is a fused
        # kernel that accumulates in fp32 internally even for bf16 input,
        # so it is ~10x more accurate than the manual
        # (x-max).exp().sum().log()+max done entirely in bf16 it replaces
        # (measured: per-token MAE 0.0018 vs 0.016 vs an fp32 reference;
        # the manual path also injected a +0.36 spurious shift into the
        # sequence IS-ratio vs +0.06 for this path). The pre-gathered
        # `chosen_logits` above is left for the float32 branch only.
        per_token_logps = []
        for row_logits, row_labels in zip(logits, label_ids):
            row_logps = torch.log_softmax(row_logits, dim=-1)
            # -100 prompt slots clamp to 0 (masked out downstream by
            # completion_mask, same as the float32 branch's gather).
            idx = row_labels.unsqueeze(-1).clamp(min=0)
            per_token_logps.append(row_logps.gather(dim=-1, index=idx).squeeze(-1))
        return torch.stack(per_token_logps)


def pad_and_stack(
    input_ids: list[list[int]],
    label_ids: list[list[int]],
    advantages: list[float] | list[list[float]] | None,
    sampling_logprobs: list[list[float]] | None,
    ref_logprobs: list[list[float]] | None,
    old_logprobs: list[list[float]] | None,
    device: torch.device,
    pad_token_id: int,
    warn_key: int | None = None,
    mask_ids: int = -100,
) -> dict[str, torch.Tensor | None]:
    """TRL-aligned tensor assembly (grpo_trainer.py:1873-1931).

    input_ids / label_ids arrive PRE-JOINED per sample:
        input_ids[i] = prompt[i] + completion[i]
        label_ids[i] = [-100]*prompt_len + completion_tokens
    We split them at the -100 boundary, then follow TRL's structure:

    1. LEFT-pad prompts to P_max (pad_token_id); prompt_mask = 0 there.
    2. RIGHT-pad completions to C_max (pad_token_id); completion_mask = 0 there.
    3. RIGHT-pad engine logprobs (sampling/old/ref) + per-token advantages
       to the SAME C_max -- completion-aligned BY CONSTRUCTION.
    4. concat -> prompt_completion_ids [B, P_max+C_max],
       attention_mask = cat(prompt_mask, completion_mask).

    Returns:
        Dict of tensors on ``device`` (completion-shaped: [B, C_max]).
    """
    # 1. Split into prompt + completion at the -100 boundary in label_ids.
    prompt_ids_list: list[list[int]] = []
    completion_ids_list: list[list[int]] = []
    for inp, lab in zip(input_ids, label_ids, strict=False):
        prompt_len = sum(1 for x in lab if x == mask_ids)
        prompt_ids_list.append(list(inp[:prompt_len]))
        completion_ids_list.append(list(inp[prompt_len:]))

    batch = len(input_ids)
    p_max = max((len(p) for p in prompt_ids_list), default=0)
    c_max = max((len(c) for c in completion_ids_list), default=0)

    # 2. LEFT-pad prompts to p_max; prompt_mask 0 for left-pad, 1 for real.
    padded_prompt: list[list[int]] = []
    prompt_mask: list[list[float]] = []
    for p in prompt_ids_list:
        npad = p_max - len(p)
        padded_prompt.append([pad_token_id] * npad + p)
        prompt_mask.append([0.0] * npad + [1.0] * len(p))

    # 3. RIGHT-pad completions to c_max; completion_mask 1 for real, 0 for right-pad.
    padded_completion: list[list[int]] = []
    completion_mask: list[list[float]] = []
    for c in completion_ids_list:
        npad = c_max - len(c)
        padded_completion.append(c + [pad_token_id] * npad)
        completion_mask.append([1.0] * len(c) + [0.0] * npad)

    prompt_ids_t = torch.tensor(padded_prompt, dtype=torch.long, device=device)
    completion_ids_t = torch.tensor(padded_completion, dtype=torch.long, device=device)
    prompt_mask_t = torch.tensor(prompt_mask, dtype=torch.float32, device=device)
    completion_mask_t = torch.tensor(completion_mask, dtype=torch.float32, device=device)

    # 4. concat -> [B, p_max + c_max].
    prompt_completion_ids = torch.cat([prompt_ids_t, completion_ids_t], dim=1)
    attention_mask = torch.cat([prompt_mask_t, completion_mask_t], dim=1)

    # Advantages: [B] (sequence-level) or [B, c_max] (per-token, right-padded).
    if advantages is None or len(advantages) == 0:
        adv_t = torch.zeros(batch, dtype=torch.float32, device=device)
    elif isinstance(advantages[0], list):
        per_token_adv: list[list[float]] = advantages  # type: ignore[assignment]
        if c_max > 0:
            adv_padded = []
            for i, comp_adv in enumerate(per_token_adv):
                comp_adv = list(comp_adv) if i < len(per_token_adv) else []
                assert len(comp_adv) <= c_max, (
                    f"per-token advantages len {len(comp_adv)} > c_max {c_max}"
                )
                adv_padded.append(comp_adv + [0.0] * (c_max - len(comp_adv)))
            adv_t = torch.tensor(adv_padded, dtype=torch.float32, device=device)
        else:
            adv_t = torch.zeros(batch, dtype=torch.float32, device=device)
    else:
        seq_adv: list[float] = advantages  # type: ignore[assignment]
        adv_t = torch.tensor(seq_adv, dtype=torch.float32, device=device)

    sampling_logprobs_t = _pad_comp_lp(
        sampling_logprobs, completion_ids_list, c_max, device, warn_key
    )
    old_logprobs_t = _pad_comp_lp(old_logprobs, completion_ids_list, c_max, device, warn_key)
    ref_logprobs_t = _pad_comp_lp(ref_logprobs, completion_ids_list, c_max, device, warn_key)

    # Ground-truth per-sample lengths (the waterfall source). The loss builds
    # its internal mask from completion_len; prompt_len is passed through.
    completion_len_t = torch.tensor(
        [len(c) for c in completion_ids_list], dtype=torch.long, device=device
    )
    prompt_len_t = torch.tensor(
        [len(p) for p in prompt_ids_list], dtype=torch.long, device=device
    )

    return {
        "prompt_completion_ids": prompt_completion_ids,
        "attention_mask": attention_mask,
        "completion_ids": completion_ids_t,
        "completion_mask": completion_mask_t,
        "completion_len": completion_len_t,
        "prompt_len": prompt_len_t,
        "logits_to_keep": c_max,
        "advantages": adv_t,
        "sampling_logprobs": sampling_logprobs_t,
        "old_logprobs": old_logprobs_t,
        "ref_logprobs": ref_logprobs_t,
    }


def _pad_comp_lp(
    lp_lists: list[list[float]] | None,
    completion_ids_list: list[list[int]],
    c_max: int,
    device: torch.device,
    warn_key: int | None,
) -> torch.Tensor | None:
    """Right-pad engine logprobs to c_max (completion-aligned), NaN-pad mismatches."""
    if not lp_lists or c_max <= 0:
        return None
    padded = []
    for i, comp_lp in enumerate(lp_lists):
        comp_lp = list(comp_lp) if i < len(lp_lists) else []
        comp_len = len(completion_ids_list[i])
        if len(comp_lp) != comp_len:
            # Defensive: rollout returned a logprob count != completion length.
            # NaN-pad shortfalls / truncate extras so the IS ratio is neutralized
            # to 1 at these positions (compute_grpo_loss_core's isnan guard) --
            # NO annihilation, NO crash. Warn once per caller.
            if len(comp_lp) > comp_len:
                comp_lp = comp_lp[:comp_len]
            else:
                comp_lp = comp_lp + [float("nan")] * (comp_len - len(comp_lp))
            if warn_key is not None and warn_key not in _LP_MISMATCH_WARNED:
                _LP_MISMATCH_WARNED.add(warn_key)
                logger.warning(
                    "rollout logprob count != completion len (sample "
                    "%s: %d vs %d) -- NaN-padded to neutralize IS ratio; "
                    "root-cause the rollout (vLLM empty-logprob case).",
                    i, len(lp_lists[i]) if i < len(lp_lists) else 0, comp_len,
                )
        padded.append(comp_lp + [0.0] * (c_max - len(comp_lp)))
    return torch.tensor(padded, dtype=torch.float32, device=device)


def pad_full_sequence(
    input_ids: list[list[int]],
    label_ids: list[list[int]],
    device: torch.device,
    pad_token_id: int,
    mask_ids: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad pre-joined prompt+completion sequences (eval / full-seq path).

    Used by the eval ``_forward_no_grad`` path (full-seq top-K forward, no
    attention_mask). Prompts stay left-aligned, right-pad to max_len, label_ids
    padded with mask_ids.
    """
    max_len = max(len(seq) for seq in input_ids)
    padded_input = [seq + [pad_token_id] * (max_len - len(seq)) for seq in input_ids]
    padded_label = [lab + [mask_ids] * (max_len - len(lab)) for lab in label_ids]
    input_ids_t = torch.tensor(padded_input, dtype=torch.long, device=device)
    label_ids_t = torch.tensor(padded_label, dtype=torch.long, device=device)
    return input_ids_t, label_ids_t


def compute_engine_is_ratio(
    old_per_token_logps: torch.Tensor,
    sampling_per_token_logps: torch.Tensor,
    mask: torch.Tensor,
    is_mode: str,
    is_cap: float | None,
    is_fp32: bool = False,
) -> dict[str, torch.Tensor]:
    """Detached engine IS ratio (TRL ``vllm_importance_sampling`` correction).

    ``ratio = exp(old - sampling)`` where ``old`` (detached trainer forward) and
    ``sampling`` (engine generation-time logprobs) are both CONSTANTS — so this
    ratio carries NO gradient. It is a per-sequence (or per-token) multiplier on
    the PPO surrogate. Because it needs only the small ``[B, C]`` logprobs (NOT
    the logits), it can be precomputed once and reused across logits chunks.

    Uses ``mask`` to zero right-pad positions (layout masking, not a reduction).
    The sequence-level sum is a SUM (numerator), not a per-length normalization,
    so no provided length is needed here.

    Returns a dict with:
        ``ratio``:      the (possibly capped/masked) ratio applied to the loss.
        ``ratio_uncapped``: the ratio BEFORE the cap/mask (for diagnostics).
        ``logps_diff``:    the pre-exp diff (seq-sum for sequence modes, [B,C] for token).
    """
    # NaN sentinel -> old (so old - sampling = 0 -> ratio 1, neutral).
    if torch.isnan(sampling_per_token_logps).any():
        sampling_per_token_logps = torch.where(
            torch.isnan(sampling_per_token_logps),
            old_per_token_logps,
            sampling_per_token_logps,
        )
    per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
    if is_fp32 and per_token_logps_diff.dtype != torch.float32:
        per_token_logps_diff = per_token_logps_diff.float()
    sequence_level = is_mode in ("sequence_mask", "sequence_truncate")
    if sequence_level:
        logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)  # [B, 1]
    else:
        logps_diff = per_token_logps_diff  # [B, C]
    # Sanitize before exp: a single -inf sampling logprob (engine quirk,
    # observed on H100 SXM / flashinfer sampling) makes the diff +inf;
    # exp(+inf)=inf and the downstream masked_fill zeroes the FORWARD
    # value, but the BACKWARD of exp at that position computes
    # 0 x inf = NaN — poisoning every gradient while the printed loss
    # stays finite. Clamping to +-60 (exp ~ 1e26, finite in fp32) keeps
    # the masked_fill semantics (out-of-cap ratios still zeroed/masked)
    # while making the backward well-defined. NaN diffs (if any input
    # escapes the isnan guard) are neutralized to 0 -> ratio 1.
    logps_diff = torch.nan_to_num(logps_diff, nan=0.0, posinf=60.0, neginf=-60.0)
    logps_diff = logps_diff.clamp(min=-60.0, max=60.0)
    ratio_uncapped = torch.exp(logps_diff)
    ratio = ratio_uncapped
    clip_disabled = is_cap is None or is_cap <= 0
    if not clip_disabled:
        if is_mode in ("sequence_truncate", "token_truncate"):
            ratio = torch.clamp(ratio, max=is_cap)
        elif is_mode in ("sequence_mask", "token_mask"):
            ratio = ratio.masked_fill(ratio > is_cap, 0.0)
        else:
            raise ValueError(
                f"Unknown engine_is_mode: {is_mode!r}. Must be one of "
                "'sequence_mask', 'sequence_truncate', 'token_mask', 'token_truncate'."
            )
    return {"ratio": ratio, "ratio_uncapped": ratio_uncapped, "logps_diff": logps_diff}


def compute_grpo_loss_core(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor | None,
    ref_per_token_logps: torch.Tensor | None,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    kl_coef: float,
    clip_low: float,
    clip_high: float,
    loss_reduction: str,
    importance_sampling_level: str,
    engine_is_correction: bool,
    engine_is_mode: str,
    engine_is_cap: float | None,
    max_allowed_len: int,
    global_batch_size_override: int = 0,
    completion_len: torch.Tensor | None = None,
    prompt_len: torch.Tensor | None = None,
    total_tokens: int = 0,
    sampling_per_token_logps: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    loss_fp32: bool = False,
    is_fp32: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """Core GRPO loss (PPO clipped surrogate + engine IS + KL + reduction).

    Pure tensor function: no ``self``, no diagnostics accumulator.

    ``mask`` is used for LAYOUT (zeroing right-pad: ``per_token_loss * mask``,
    the sequence-level sums). ``completion_len`` / ``max_allowed_len`` /
    ``global_batch_size_override`` are used for REDUCTION denominators — never
    ``mask.sum()`` when the lengths are provided. When ``completion_len`` is
    None, the function falls back to ``mask.sum()`` (backward-compatible).

    Args:
        per_token_logps: [B, C] current policy logprobs (differentiable)
        old_per_token_logps: [B, C] detached old-policy logprobs
        ref_per_token_logps: [B, C] reference logprobs (for KL)
        advantages: [B] or [B, C] per-sample / per-token advantages
        mask: [B, C] completion mask (layout — zeros right-pad)
        max_allowed_len: int — the configured completion cap (dr_grpo denominator)
        global_batch_size_override: int — global batch size for dr_grpo under
            sharding/GAS (0 = use local batch_size)
        completion_len: [B] real completion-token count per sample. When
            provided, used for per-sequence denominators (grpo, sequence-level
            coef_1). None = fall back to mask.sum().
        prompt_len: [B] real prompt-token count per sample (passed through to
            intermediates; not used by any reduction yet).
        total_tokens: int — the GLOBAL real-token count for the bnpo/ce
            denominator. When >0, used directly (the loss does NOT aggregate
            completion_len — important when masks are sliced by
            chunking/sharding, where the local sum would be wrong). 0 = fall
            back to completion_len.sum() (if provided) then mask.sum().

    Returns the scalar loss plus a dict of intermediates the caller may use for
    diagnostics (clip_ratio stats, IS-ratio stats, per-token KL, pre/post-IS
    loss, etc.).
    """
    if loss_fp32:
        per_token_logps = per_token_logps.float()
        if old_per_token_logps is not None:
            old_per_token_logps = old_per_token_logps.float()
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps.float()
        if sampling_per_token_logps is not None:
            sampling_per_token_logps = sampling_per_token_logps.float()
        advantages = advantages.float()
        mask = mask.float()
        if completion_len is not None:
            completion_len = completion_len.float()
        if entropy is not None:
            entropy = entropy.float()

    # advantages: [batch] -> [batch, 1] for broadcasting (per-token stays [B, C]).
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    if old_per_token_logps is None:
        old_per_token_logps = per_token_logps.detach()

    log_ratio = per_token_logps - old_per_token_logps

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio  # [B, C]
        coef_1 = torch.exp(log_importance_weights)
    elif importance_sampling_level == "sequence":
        # Per-sequence normalization: sum via mask (layout), divide by
        # completion_len (provided length) when available, else mask.sum().
        seq_sum = (log_ratio * mask).sum(-1)
        if completion_len is not None:
            log_importance_weights = seq_sum / completion_len.clamp(min=1)
        else:
            log_importance_weights = seq_sum / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
        coef_1 = torch.exp(log_importance_weights)
    else:
        raise ValueError(
            f"Unknown importance_sampling_level: {importance_sampling_level!r}. "
            "Must be 'token' or 'sequence'."
        )

    coef_2 = torch.clamp(coef_1, 1.0 - clip_low, 1.0 + clip_high)
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    loss_pre_is = per_token_loss

    importance_sampling_ratio: torch.Tensor | None = None
    per_token_kl: torch.Tensor | None = None
    if (
        sampling_per_token_logps is not None
        and old_per_token_logps is not None
        and engine_is_correction
        and loss_reduction != "ce"
    ):
        _is = compute_engine_is_ratio(
            old_per_token_logps,
            sampling_per_token_logps,
            mask,
            engine_is_mode,
            engine_is_cap,
            is_fp32=is_fp32,
        )
        importance_sampling_ratio = _is["ratio"]
        is_logps_diff = _is["logps_diff"]
        is_ratio_uncapped = _is["ratio_uncapped"]
        per_token_loss = per_token_loss * importance_sampling_ratio
    else:
        is_logps_diff = None
        is_ratio_uncapped = None
    loss_post_is = per_token_loss

    if kl_coef > 0.0 and ref_per_token_logps is not None:
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1.0
        )
        per_token_loss = per_token_loss + kl_coef * per_token_kl

    # ------------------------------------------------------------------
    # Reduction: uses provided lengths for denominators (not mask.sum()).
    # ------------------------------------------------------------------
    masked_loss = per_token_loss * mask  # [B, C]
    batch_size = masked_loss.size(0)

    # Total tokens: provided global scalar (no aggregation), else fall back.
    # Under chunking/sharding the local completion_len.sum() / mask.sum() would
    # be wrong, so the caller should pass the global total_tokens.
    if total_tokens > 0:
        total_tokens_t = torch.as_tensor(
            float(total_tokens), dtype=torch.float32, device=per_token_logps.device
        ).clamp(min=1.0)
    elif completion_len is not None:
        total_tokens_t = completion_len.sum().clamp(min=1.0)
    else:
        total_tokens_t = mask.sum().clamp(min=1.0)

    if loss_reduction == "ce":
        loss = -(per_token_logps * mask).sum() / total_tokens_t
    elif loss_reduction == "bnpo":
        loss = masked_loss.sum() / total_tokens_t
    elif loss_reduction == "dr_grpo":
        effective_batch_size = (
            global_batch_size_override if global_batch_size_override > 0 else batch_size
        )
        loss = masked_loss.sum() / (effective_batch_size * max_allowed_len)
    else:  # "grpo" — per-sequence normalization then mean over batch.
        if completion_len is not None:
            per_seq_loss = masked_loss.sum(-1) / completion_len.clamp(min=1)
        else:
            per_seq_loss = masked_loss.sum(-1) / mask.sum(-1).clamp(min=1.0)
        # Mean over the GLOBAL batch via global_batch_size_override (mirrors
        # dr_grpo's denominator mechanism). per_seq_loss.sum() / global_batch.
        # When the override is 0 (single forward, no microbatching) this is
        # bit-identical to per_seq_loss.mean(). Under microbatching the override
        # is the global per-rank batch, injected into EVERY microbatch, so
        # Σ per-microbatch (sum/ global) reassembles the true mean (additive
        # split) — avoiding the n_microbatches× scale error that per-microbatch
        # .mean() produced (each microbatch divided by its local count).
        effective_batch_size = (
            global_batch_size_override if global_batch_size_override > 0 else batch_size
        )
        loss = per_seq_loss.sum() / effective_batch_size

    intermediates = {
        # post-(possible-)upcast inputs (for the WW_IS_DIAG dump)
        "per_token_logps": per_token_logps,
        "old_per_token_logps": old_per_token_logps,
        "sampling_per_token_logps": sampling_per_token_logps,
        "ref_per_token_logps": ref_per_token_logps,
        "advantages": advantages,
        "mask": mask,
        "completion_len": completion_len,
        "prompt_len": prompt_len,
        "entropy": entropy,
        # PPO surrogate stage
        "log_ratio": log_ratio,
        "coef_1": coef_1,
        "coef_2": coef_2,
        "per_token_loss1": per_token_loss1,
        "per_token_loss2": per_token_loss2,
        "per_token_loss_pre_is": loss_pre_is,
        # engine IS-correction stage
        "is_logps_diff": is_logps_diff,
        "is_ratio_uncapped": is_ratio_uncapped,
        "importance_sampling_ratio": importance_sampling_ratio,
        "per_token_loss_post_is": loss_post_is,
        # KL stage
        "per_token_kl": per_token_kl,
        "per_token_loss_final": per_token_loss,
        # aggregation stage
        "masked_loss": masked_loss,
        "per_seq_loss": per_seq_loss if loss_reduction == "grpo" else None,
        "batch_size": batch_size,
        "total_tokens": total_tokens_t,
        "loss": loss,
    }
    return loss, intermediates
