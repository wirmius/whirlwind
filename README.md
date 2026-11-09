# 🌪️ Whirlwind

**A human-in-control, extensible and debuggable post-training package for
experimenting with novel algorithms.**

New post-training ideas (a reward, an advantage estimator, a sampling scheme,
a different update) rarely fit the shape of a large training stack. Trying one
usually means changing several layers you did not write, under a scheduler
that decides what runs where, in processes you cannot stop and look inside.
Whirlwind is built the other way around: it is small enough to read
completely, and it is made of plain processes that you start, stop and step
through yourself.

- **Human in control.** Every stage of the loop (data, rollout, reward, model
  update, tracking, and the training loop itself) is an ordinary Python
  process with its own GPUs, config and log. You decide where each one runs,
  and nothing schedules, hides or restarts them behind your back. The training
  loop is one file you can read top to bottom. Configs are plain YAML and
  reject unknown keys, so a typo fails at startup instead of silently running
  the default.
- **Extensible.** Stages talk only through typed messages, addressed by node
  name. To try an idea, write one sibling module and pass one flag
  (`--reward-module`, `--rollout-module`, `--model-module` or
  `--training-module`); the rest of the loop does not change and does not
  need to know. A custom reward is a subclass that overrides one method.
- **Debuggable.** Launch the whole loop for real with one node under your
  debugger, and stop at a breakpoint inside a real training step
  ([docs/debugging.md](docs/debugging.md)). Every RPC can be printed as it
  happens, and training steps can be captured and replayed offline. The
  numerics are checked against TRL's `GRPOTrainer`, and the lessons are
  written down in [`TIPS_AND_TRICKS.md`](TIPS_AND_TRICKS.md).

Rollout runs on SGLang or vLLM, and the model update on PyTorch FSDP2 (with
HF-Trainer and accelerate alternatives). The rollout engine and the trainer
can share the same GPUs through sleep/wake. The whole system is about 22k
lines of Python, with no Ray and no cluster scheduler.

## How the stages talk: whirlcoms

The entire communication layer is `whirlwind/whirlcoms/`, about 2,000 lines
on top of ZeroMQ:

- A **node** is a name plus the handlers it exposes. It binds one socket,
  `ipc:///tmp/whirlwind-<name>`, and calls other nodes directly by name.
- A **registry**, inside the training-loop process, records which nodes are
  up: their addresses, endpoints and heartbeats (every 5 s). It never relays
  data.
- A **message** is a dataclass sent as msgpack. Tensors over 64 KB go
  through a file in `/dev/shm` instead of the socket.

A complete two-node example:

```python
import asyncio

from whirlwind.whirlcoms import Node, Orchestrator


async def main():
    registry = Orchestrator()            # in a real run the training loop hosts it
    await registry.start()

    reward = Node("reward", auto_process=True)

    @reward.expose("score")
    def score(completion: str) -> float:
        return float(completion.strip().endswith("#### 42"))

    await reward.start()                 # binds ipc:///tmp/whirlwind-reward, registers

    loop = Node("loop")
    await loop.start()
    while "reward" not in loop.topology:  # the registry broadcasts who is up
        await asyncio.sleep(0.1)

    print(await loop.call("reward", "score", "6 * 7 = 42 #### 42"))   # 1.0

    for n in (loop, reward, registry):
        await n.stop()


asyncio.run(main())
```

That is the whole contract between stages. Because a node is found by its
name, any process started on the same host with the right config joins a
running loop. This is what lets you start only some of the nodes, run one by
hand or under a debugger, or swap an implementation without the others
noticing. Internals: [docs/whirlcoms.md](docs/whirlcoms.md).

## Install

Python ≥ 3.11, Linux, CUDA GPUs on one machine (the model update can add
ranks on more machines: [docs/launch.md](docs/launch.md#multi-node-model-update)):

```bash
pip install -e ".[sglang]"       # core + the SGLang rollout engine (or ".[vllm]")
pip install -e ".[lora,dev]"     # optional: PEFT for LoRA, pytest
```

## Crash course

**1. Run the reference recipe.** Qwen3-0.6B-Base learns GSM8K, with the
rollout engine and the trainer colocated on the same GPUs:

```bash
GPUS=0,1 bash examples/gsm8k_dr_grpo_colocate.sh    # two GPUs (the reference setup)
GPUS=0   bash examples/gsm8k_dr_grpo_colocate.sh    # one GPU
```

Metrics stream to W&B; `export WANDB_MODE=offline` to run without an account.
Logs and checkpoints land under `runs/`.

**2. Check a configuration without launching anything:**

```bash
./bringup_and_run.sh --dry-run
```

**3. Swap a stage.** The same loop with different engines:

```bash
./bringup_and_run.sh --rollout-gpus 0 --model-gpus 1,2 \
    --rollout-module whirlwind.rollout.rollout_vllm --rollout-config configs/rollout/vllm.yaml \
    --model-module whirlwind.model.model_updater_hf_trainer
```

**4. Tune through YAML.** Each stage reads its own file under `configs/<stage>/`
(pass `--model-config`, `--rollout-config`, …); every field is listed in
[docs/configuration.md](docs/configuration.md).

**5. Bring your own idea.** A reward that prefers short correct answers:

```python
# examples/length_penalty_reward.py
from whirlwind.reward.reward_function import RewardNode, main

class LengthPenaltyReward(RewardNode):
    def _raw_rewards(self, req):
        correct = super()._raw_rewards(req)          # 1.0 / 0.0 accuracy
        return [r - 1e-4 * len(c) for r, c in zip(correct, req.completions)]

if __name__ == "__main__":
    main(LengthPenaltyReward)
```

```bash
./bringup_and_run.sh --reward-module examples.length_penalty_reward
```

New rollout engines, model updaters and training loops follow the same
pattern; see the [extension guide](docs/extension_guide.md).

## Example launches on a 4-GPU machine

**Baseline and variant side by side.** The reference recipe on GPUs 0 and 1,
and the same recipe with your reward on GPUs 2 and 3:

```bash
GPUS=0,1 bash examples/gsm8k_dr_grpo_colocate.sh
GPUS=2,3 LOG_DIR=runs/length_penalty bash examples/gsm8k_dr_grpo_colocate.sh \
    --reward-module examples.length_penalty_reward
```

Each launch gets its own `WW_RUN_ID`, which namespaces its sockets, torchrun
port and weight-export directory, so the two runs never see each other. If
your training config writes checkpoints, give each run its own `workdir`.

**Cross-check the model update.** The same recipe with the HF-Trainer-backed
model node, next to the baseline above:

```bash
GPUS=2,3 LOG_DIR=runs/hftrainer \
MODEL_CONFIG=configs/model/fsdp2_base_colocate_gas1_fp32master_hftrainer.yaml \
    bash examples/gsm8k_dr_grpo_colocate.sh \
    --model-module whirlwind.model.model_updater_hf_trainer
```

**Rollout and training on separate GPUs.** No sleep/wake: SGLang keeps GPU 0
and FSDP2 trains on GPUs 1–3, with the default configs (Qwen3-0.6B on GSM8K):

```bash
./bringup_and_run.sh --rollout-gpus 0 --model-gpus 1,2,3
```

**Agent post-training.** Both experimental modes at once (see
[below](#experimental-agent-post-training)):

```bash
ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_async_proxy.sh   # simulated GSM8K agent
ENGINE_GPU=2 MODEL_GPU=3 bash examples/experimental_agentic.sh       # multi-turn tool use
```

More layouts, including a model update that spans several machines, are in
[docs/launch.md](docs/launch.md).

## Debug one node in a live run

Start every node except one, then start that one under your debugger. It
joins the running loop like any other node. Here the reward node runs under
`debugpy`, next to the reference recipe:

```bash
# in both terminals: one run id, and timeouts that let a node sit at a breakpoint
export WW_RUN_ID=debug WHIRLWIND_HEARTBEAT_TIMEOUT=86400 \
       WHIRLWIND_RPC_TIMEOUT=86400 WHIRLWIND_HEARTBEAT_FATAL_MISSES=100000

# terminal 1: everything except the reward node
GPUS=0,1 bash examples/gsm8k_dr_grpo_colocate.sh \
    --nodes orchestrator,rollout,dataloader,tracker,model_update

# terminal 2: the reward node, waiting for your debugger on port 5678
python -m debugpy --listen 5678 --wait-for-client \
    -m whirlwind.reward.reward_function --config configs/reward/function.yaml
```

Attach your editor's debugger to port 5678 and set a breakpoint in
`RewardNode._compute_rewards`. [docs/debugging.md](docs/debugging.md)
covers every node (including the model node and the training loop itself),
why each timeout matters, and ready-made VS Code launch configurations.

## What's in the box

| stage | implementations |
|---|---|
| rollout | SGLang, vLLM, a pool of engines behind one name; LoRA adapters |
| model update | manual FSDP2 (chunked lm_head, fp32 master weights), HF-Trainer-backed, accelerate |
| objective | GRPO-family losses (`grpo`, `dr_grpo`, `bnpo`) with engine importance-sampling correction |
| reward | accuracy (regex), length; anything you subclass |
| data / tracking | Hugging Face datasets, Weights & Biases |

## Experimental: agent post-training

An actor-learner mode for training on the traffic of an external agentic
system. An always-on rollout engine sits behind a proxy that serves the
agent, logs every sample with the policy version that produced it, and takes
rewards back whenever the agent reports them. A separate LoRA updater trains
on whole, fresh-enough groups from the proxy's replay buffer and hot-swaps
the adapter into the engine.

```bash
ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_async_proxy.sh     # simulated GSM8K agent
ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_agentic.sh       # multi-turn tool-using agent
```

On GSM8K with a simulated single-turn agent, a 65-step run raised validation
accuracy from 0.146 to 0.609, against 0.636 for the synchronous LoRA recipe.
**Multi-turn agents** tag each request with its episode; a sibling training
loop regroups the turns into episodes and trains every turn with its
episode's advantage. Three example agents ship with it: procedurally
generated file-system and clock questions solved with tool calls that run in
a podman container (format enforced by constrained decoding; Qwen3-0.6B-Base
goes from 72% to 100% exact answers on held-out tasks in 100 steps),
reflect-and-retry GSM8K, and containerized SWE-Gym bash episodes.

Both are **experimental**: LoRA updates only, and the engine and the updater
each need a GPU (a shared-GPU layout exists for the multi-turn mode). Design,
API and limitations: [docs/async_proxy_rollout.md](docs/async_proxy_rollout.md),
[docs/agentic_multiturn.md](docs/agentic_multiturn.md); the agent-side
protocol: [docs/agent_connector_spec.md](docs/agent_connector_spec.md).

## Upcoming releases

- **NVFP4 native training.** The model update computes in NVIDIA's 4-bit
  floating-point format (NVFP4) on Blackwell GPUs.
- **Message Bag recording.** Record the typed messages that cross whirlcoms
  during a run, then replay them into a single node: reproduce a step
  offline, or step through a node's handling of real traffic without
  starting the rest of the loop.

## Learn more

- **[TECHNICAL_README.md](TECHNICAL_README.md)**: architecture, the step
  lifecycle, configuration, runtime knobs, numerics, validation results
- **[TIPS_AND_TRICKS.md](TIPS_AND_TRICKS.md)**: the rules learned the hard
  way (precision, loss scaling, sleep/wake, engine logprobs, RPC,
  methodology) and the debugging toolbox
- **[docs/launch.md](docs/launch.md)**: every launch option, GPU layouts,
  multi-node, checkpoints, common issues
- **[docs/debugging.md](docs/debugging.md)**: one node under a debugger in a
  live run
- **[docs/extension_guide.md](docs/extension_guide.md)**: adding your own stages
- **[AGENTS.md](AGENTS.md)**: orientation for AI coding assistants, with
  skill files under [`.claude/skills/`](.claude/skills/)
- **[docs/INDEX.md](docs/INDEX.md)**: every other doc

## License

MIT; see [LICENSE](LICENSE).
