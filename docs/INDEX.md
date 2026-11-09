# Documentation Index

Entry point: [`README.md`](../README.md).

## Reference

| Doc | What | Status |
|---|---|---|
| `README.md` | What it is, install, crash course | current |
| `TECHNICAL_README.md` | Architecture, step lifecycle, configuration, runtime knobs, numerics, validation | current |
| `TIPS_AND_TRICKS.md` | The rules learned the hard way: precision, loss scaling, sleep/wake, engine logprobs, RPC, methodology, debugging toolbox | current |
| `docs/extension_guide.md` | How to add variants (sibling files, recipes, run pattern, rules) | current |
| `docs/whirlcoms.md` | RPC layer: framing, transport, discovery, the v2 node seam | current |
| `docs/messages.md` | Message types, dimensionality, split/join contract | reference — verify against `whirlwind/messages.py` |
| `docs/configuration.md` | Every config field with type, default and notes, generated from the dataclasses | current |
| `docs/launch.md` | Launching: example recipes, `bringup_and_run.sh`, manual launch, GPUs, checkpoints, common issues | current |
| `docs/debugging.md` | Running one node under a debugger inside a live run: launch, timeouts, per-node commands, VS Code configs | current |
| `CHANGELOG.md` | Notable changes | current |

## Experimental

| Doc | What |
|---|---|
| `docs/async_proxy_rollout.md` | The async proxy-rollout (actor-learner) mode for agent post-training: architecture, running it, knobs, staleness, limitations |
| `docs/agentic_multiturn.md` | Multi-turn agents on top of the async mode: episode protocol, episode grouping, serving, shared-GPU layout, example agents |
| `docs/agent_connector_spec.md` | The protocol an external agent uses to talk to the rollout proxy |

## Design docs

| Doc | What |
|---|---|
| `docs/rollout_pool_design.md` | N rollout engines behind one node ID (pool) |
| `docs/lora_design.md` | LoRA path: FSDP2 + SGLang/vLLM LoRA rollout |

## For AI coding assistants

| File | What |
|---|---|
| `AGENTS.md` | Orientation: repo map, conventions, test gate, which skill to load |
| `.claude/skills/whirlwind-codebase/SKILL.md` | Architecture, module map, selection mechanisms, extension rules |
| `.claude/skills/whirlwind-training-pitfalls/SKILL.md` | Checklist before touching precision, loss scaling, sleep/wake, engine logprobs or collectives |

When a doc goes stale, update it or mark its status here.
