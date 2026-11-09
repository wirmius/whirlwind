# Extension Guide — adding variants to Whirlwind

How to extend Whirlwind with new machinery: reward functions, rollout engines,
model updaters, orchestrators, loss variants, and model-forward features.
This is the official extension path.

## Principles

1. **New machinery goes in new sibling files**, in the same module directory as
   the original — the pattern `model_fsdp2.py` / `model_updater_hf_trainer.py` /
   `model_accelerate_pure.py` and `rollout_sglang.py` / `rollout_vllm.py` /
   `sglang_rollout_pool.py` already establish.
2. **Reuse before re-implementing.** Subclass the original; override only what
   differs; import its utils. If two siblings need the same helper, move the
   helper into the module's shared file (`updater_utils.py`, `utils.py`) as a
   minimal, behavior-preserving change.
3. **Edit existing files only if unavoidable**, and such edits stay minimal +
   generic: a dispatch entry, a hook point, a config-gated branch defaulting to
   base behavior — never variant-specific logic inline.
4. **Config selects the sibling** (YAML tree per node kind, `configs/<kind>/`).
5. **Base regression tests stay green** — originals keep working unchanged.
6. **No long-lived per-variant branches.** Siblings import the originals, so
   base fixes flow into variants automatically; the only merge surface is the
   rare minimal edit. Use short-lived experiment branches and tags instead.

## Selection mechanisms (how the system picks implementations)

| What | Mechanism | Example |
|---|---|---|
| Rollout engine | `--rollout-module MOD` (bringup) | `--rollout-module whirlwind.rollout.rollout_vllm` |
| Model updater | `--model-module MOD` (bringup, torchrun `-m`) | `--model-module whirlwind.model.model_updater_hf_trainer` |
| Node configs | `configs/<kind>/<variant>.yaml` via `WHIRLWIND_*_CONFIG` env | `WHIRLWIND_MODEL_CONFIG=.../chunked_dr_grpo.yaml` |
| Loss / IS / precision | flags in the model config | `loss_reduction: dr_grpo`, `engine_is_correction: true` |
| Orchestrator | which module you launch | `python -m whirlwind.training.training_grpo_capture` |

Bringup (`bringup_and_run.sh`) is the composition root: it launches nodes with
`CUDA_VISIBLE_DEVICES` isolation, `torchrun` for the model updater, and the
`WHIRLWIND_*_CONFIG` env vars. A new sibling that keeps the same RPC contract
needs **zero orchestrator changes** — the orchestrator talks to node IDs
(`rollout`, `reward`, `model`, `dataloader`), not to modules.

## Recipes

### 1. New reward function

Subclass `RewardNode` (`reward_function.py`) and override `_raw_rewards(req)`
— the per-sample reward list. Group-normalized advantages, truncation masking
and metrics/table logging are inherited. End the module with
`main(MyRewardNode)` so bringup can run it with `python -m`. A complete example
is [`examples/length_penalty_reward.py`](../examples/length_penalty_reward.py).

Wire it up with `--reward-module <dotted.module>` and, if it needs its own
knobs, a `configs/reward/<variant>.yaml`. To change the advantage estimator
too, override `_compute_grpo_advantages`.

### 2. New rollout engine

Create `whirlwind/rollout/rollout_<variant>.py` next to the existing engines.
**RPC contract to satisfy** (see `rollout_sglang.py` / `rollout_vllm.py`):

- `generate(PromptBatch) -> RolloutResult` — completions, `token_ids`,
  per-token `logprobs` (NaN sentinel for missing — never 0.0: 0.0 annihilates
  the IS ratio), `finish_reasons`
- `update_weights(checkpoint_path)` — hot-reload; raise on failure (no silent
  stale weights)
- `sleep()` / `wake()` — colocate memory management (engine-specific)
- `ping`

Engine logprobs must be **temperature-consistent** with sampling (vLLM's
`processed_logprobs` lesson) and length-aligned to the completion tokens
(SGLang truncates/NaN-pads to `len(comp_toks)`). The model node asserts this.

Select with `--rollout-module whirlwind.rollout.rollout_<variant>` and a
`configs/rollout/<variant>.yaml`. For multi-engine setups, see
`sglang_rollout_pool.py` (pool of engines behind one node ID) and
`docs/rollout_pool_design.md`.

### 3. New model updater

Create `whirlwind/model/model_<variant>.py`. Subclass `ModelUpdateNode`
(`model_fsdp2.py`) — the HF-Trainer-backed node
(`model_updater_hf_trainer.py`) is the worked example: it inherits
`_pad_and_stack`, `_compute_per_token_logps`, `_compute_loss`, and swaps only
the training-step machinery (HF `Trainer.training_step`, `accelerator.backward`).

Know the invariants before overriding:

- **FP32 master**: `torch_dtype="float32"` (bf16 master freezes GRPO updates
  below BF16 ULP — the flat-learning bug). Keep it in your config.
- **FSDP2 gradient reduction is `ReduceOp.AVG`** for fp32 reduce with
  world_size>1 (`torch/.../_fsdp_collectives.py`, `_get_gradient_divide_factors`).
  Denominators normalize per-rank; cross-rank averaging is FSDP's job.
- **The dr_grpo/grpo denominators** in the microbatch path are
  `data.batch_size // world_size` and the token total `// world_size`: the
  microbatch split runs *before* `_fsdp_using_function` scatters the batch,
  so `data` there is the global batch, and each rank must normalize by its
  own share. Without the division the microbatched gradient is
  `world_size`× smaller than the `micro_batch_size: -1` path. If you move the
  decorator, re-derive this (TIPS_AND_TRICKS.md §2.1).
- **Rank-0 RPC + `_standby`**: non-lead ranks block in a scatter/handler/
  all-gather loop; all FSDP ranks must execute the same collectives in the
  same order.
- Feature knobs to inherit/extend: `loss_reduction` (`grpo|dr_grpo|bnpo|ce`),
  `engine_is_correction` + `engine_is_mode` + `engine_is_cap`,
  `logits_chunk_size` (chunked lm_head), `micro_batch_size` (-1 = bypass the
  microbatch loop), `fsdp_grad_reduce_dtype`, `gradient_accumulation_steps`.

Select with `--model-module whirlwind.model.model_<variant>` and
`configs/model/<variant>.yaml`.

### 4. Orchestrator variant

Create `whirlwind/training/training_<variant>.py` next to
`training_grpo.py`; subclass `TrainingOrchestrator` and override the phases
you need. Existing precedent: `training_grpo_capture.py` (adds a
diagnostic-capture mode). The RPC layer (`whirlcoms/`) and `utils.py` are
shared as-is.

### 4b. Reward node variant

Create `whirlwind/reward/reward_<variant>.py` next to
`reward_function.py`; subclass `RewardNode` and override the
`_compute_rewards` handler (async overrides work — the node dispatcher
awaits coroutine handlers). Use this when the variant needs more than a
different per-sample reward: its own advantage estimator, per-token
advantages (`RewardResult.advantages` may be `[batch, completion_len]`; the
training loop and the model node consume it unmodified), or a call to another
node (e.g. a scoring endpoint you add to the model node via node→node RPC).
Selected via `--reward-module`; if it needs its own knobs, give the reward
YAML a variant-specific top-level section, add its name to
`_EXTRA_SECTIONS` in `config.py` (unknown keys are rejected otherwise), and
read the section in the node.

### 5. New loss reduction

The loss math lives in `updater_utils.compute_grpo_loss_core` — shared by all
model updater siblings. Add the new reduction as a branch there keyed by
`loss_reduction` (a legitimate dispatch edit), or compute it in your own
updater sibling on top of the returned intermediates (the function returns
every stage: log_ratio, coefs, per-token losses, KL, masked loss — see
`_update_diag.py` for how diagnostics consume them).

### 6. Feature flag / config knob

Add the field to the relevant dataclass in `whirlwind/config.py` (default =
base behavior), read it in the node, and gate the new path behind it. Default
**off**. This is the pattern for model-forward extensions too — e.g. mixing
`inputs_embeds` with `input_ids` would land as a config-gated seam
(placeholder-token splice or an embedding hook) that no-ops when the flag is
unset; a large enough feature justifies a new updater sibling instead.

### 7. Dataloader variant

`whirlwind/data/` holds the RPC dataloader node (`dataloader_node.py`) and the
legacy in-process loader (`data_hf.py`). A new source (non-GSM8K dataset,
different prompt builder) is a sibling file + `configs/dataloader/*.yaml`;
the message contract (`BatchRequest`/`BatchResult`, `entry_id` tracking for
GRPO grouping) must be preserved.

## Run harness (the repro pattern)

Every variant gets the existing run pattern, not a new one:

1. `configs/training/<variant>.yaml` — training config (references the
   model/rollout/reward YAMLs, steps, validation cadence, seed, workdir).
2. A driver script modelled on `examples/gsm8k_dr_grpo_colocate.sh` — exports
   `WHIRLWIND_*_CONFIG`, sets `WW_RUN_ID`, launches bringup, tees a log.
3. Record the config tree + seed + commit next to the result; root causes of
   surprises belong in `TIPS_AND_TRICKS.md`.

A run is reproducible from: the git tag/SHA + driver + config tree + seed.

## Testing

- `pytest tests/` must stay green with your variant *not selected*.
- Add `tests/test_<variant>_*.py` for variant-specific behavior; GPU-free unit
  tests preferred (see `test_chunked_lm_head.py`,
  `test_trl_padding_alignment.py` for the style: exact-reference comparisons,
  not smoke checks).
- Before a long run: a 65-step driver (val@64 sanity), then the full 200-step.

## Anti-patterns

- Variant-specific logic inline in an original file.
- Copy-pasting helpers between siblings (move them to shared utils instead).
- Long-lived per-variant branches (port tax, drift, weaker repro).
- Changing numerics of the base path while adding a variant (keep the base
  trajectory bit-stable; if a fix is needed, it's a separate commit first).
