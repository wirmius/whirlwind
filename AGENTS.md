# Working on Whirlwind with an AI coding assistant

Whirlwind is a small GRPO/RLHF post-training framework for one multi-GPU
machine: separate processes (dataloader, rollout engine, reward, FSDP2 model
updater, tracker) talk over ZeroMQ peer-to-peer RPC, orchestrated by one
training loop. This file is the entry point for coding agents; the human entry
point is [README.md](README.md).

## Load the right skill first

Skill files live under `.claude/skills/<name>/SKILL.md` (Agent Skills format:
YAML front matter with `name` and `description`, then the body).

| Task | Load |
|---|---|
| Understand the architecture, find where something lives, add a variant | `whirlwind-codebase` |
| Touch precision, loss scaling, microbatching, sleep/wake, engine logprobs, importance sampling, FSDP collectives, RPC timeouts | `whirlwind-training-pitfalls` (and read `TIPS_AND_TRICKS.md`) |
| Work on the experimental async proxy-rollout mode (`rollout_proxy.py`, `replay_buffer.py`, `training_async.py`) | `whirlwind-codebase` (its experimental section) and `docs/async_proxy_rollout.md` |
| Work on the experimental multi-turn agentic mode (`*_agent.py`, `training_agentic.py`, `config_agentic.py`) | `whirlwind-codebase` (its experimental section) and `docs/agentic_multiturn.md` |

## Repo map

```
whirlwind/config.py        config dataclasses (authoritative field list; unknown YAML keys raise)
whirlwind/config_agentic.py  config subclasses for the experimental agentic mode
whirlwind/messages.py      typed message contracts, split/join, serialization
whirlwind/whirlcoms/       ZeroMQ RPC: node, registry (orchestrator.py), transport (SHM for >64 KB)
whirlwind/training/        training_grpo.py (the loop), training_grpo_capture.py (batch capture sibling),
                           training_async.py + sim_external_agent.py (experimental async mode),
                           training_agentic.py + sim_os_agent.py, sim_reflect_agent.py, sim_swe_agent.py
                           (experimental agentic mode)
whirlwind/rollout/         rollout_sglang.py, rollout_vllm.py, sglang_rollout_pool.py,
                           rollout_proxy.py + replay_buffer.py (experimental async mode),
                           rollout_sglang_agent.py + rollout_proxy_agent.py (experimental agentic mode)
whirlwind/model/           model_fsdp2.py (manual FSDP2 updater), model_updater_hf_trainer.py,
                           model_accelerate_pure.py, updater_utils.py (shared loss math), _update_diag.py,
                           fp8_loader.py (experimental FP8 checkpoint ingest)
whirlwind/reward/          reward_function.py (base node), advantages.py (shared GRPO advantages)
whirlwind/data/            dataloader_node.py (HF datasets, few-shot prompts, entry_id tracking)
whirlwind/tracker/         tracker_wandb.py
configs/<stage>/           one YAML per node kind; configs/training/*.yaml points at the others
examples/                  runnable recipes (gsm8k_dr_grpo_colocate.sh, experimental_async_proxy.sh,
                           experimental_agentic.sh)
tests/                     pytest suite; GPU tests skip without CUDA
bringup_and_run.sh         composition root: launches every node with its GPUs and configs
docs/                      reference docs (docs/INDEX.md), TECHNICAL_README.md, TIPS_AND_TRICKS.md
```

## Conventions

- **New machinery = a new sibling file** in the node's directory, selected by
  `--rollout-module` / `--model-module` / `--reward-module` /
  `--training-module` or a config file. Edit an original only for a minimal,
  generic hook (a dispatch entry, a config-gated branch defaulting to the old
  behavior). Full rules and recipes: `docs/extension_guide.md`.
- **Config knobs default to the current behavior.** Add the field to the
  dataclass in `whirlwind/config.py`, read it in the node, gate the new path
  behind it. Unknown YAML keys are rejected on purpose; keep it that way.
- **Keep the base trajectory bit-stable.** A change to the numerics of the
  default path is its own commit with an exact-reference test, never a side
  effect of adding a variant.
- **Engines report missing logprobs as NaN, never 0.0.** 0.0 would annihilate
  the importance-sampling ratio.
- **Every FSDP rank runs the same collectives in the same order.** Rank 0
  serves RPC; the others wait in `_standby`. Rank-dependent branches deadlock.
- **Loss denominators are per-rank**; FSDP2 averages gradients across ranks.
  The microbatch split runs before the rank scatter, so the microbatch
  overrides divide by `world_size` (TIPS_AND_TRICKS.md §2.1).
- Style: ruff (`pyproject.toml`), line length 100, type hints, dataclasses for
  messages and configs. Comments explain *why*; keep run names, dates and
  machine names out of the code.

## Test gate

```bash
python3 -m pytest tests/ -q                  # whole suite; GPU tests skip without CUDA
python3 -m pytest tests/test_async_proxy.py -q   # experimental async mode (CPU, fake engine)
python3 -m pytest tests/test_agentic.py -q       # experimental agentic mode (CPU)
./bringup_and_run.sh --dry-run               # validate the default config tree
```

Tests are unittest-style and pytest-discoverable. Prefer exact-reference
comparisons over smoke checks (see `tests/test_trl_padding_alignment.py`,
`tests/test_chunked_lm_head.py`). Validate any
numerics change on at least two ranks: world-size bugs are invisible on one GPU.

## Things that bite

The short list; the reasoning and the checks are in `TIPS_AND_TRICKS.md`.

1. FSDP2 keeps the sharded master weights in the load dtype: GRPO needs
   `torch_dtype="float32"` or the updates round to zero.
2. Never reassign FSDP2 internals (`_sharded_param_data`) to move them across
   devices; offload only optimizer state, or use `CPUOffloadPolicy`.
3. Engine logprobs must be computed after temperature scaling (vLLM
   `processed_logprobs`) and must align 1:1 with completion tokens.
4. `node.call` on a node without `fail_on_timeout=True` returns `None` on
   timeout; use `call_typed` or guard the result.
5. The reported loss is a noisy reporter under bf16; judge training by the
   gradient norm and the validation reward.
