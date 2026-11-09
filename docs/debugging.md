# Debugging one node in a live run

Every stage of the loop is its own process, and nodes find each other by name
through IPC sockets on the host. So you can launch the whole loop for real and
start one node yourself, under a debugger. It joins the run as if bringup had
started it, and you stop at a breakpoint inside a real training step: real
batches, real weights, and real peers waiting for its answer.

## The short version

The reference recipe, with the reward node in the debugger. You need
`pip install debugpy` (or the debugger of your editor, see [VS Code](#vs-code)).

```bash
# in both terminals
export WW_RUN_ID=debug                          # the same run id everywhere
export WHIRLWIND_HEARTBEAT_TIMEOUT=86400        # see "Timeouts" below
export WHIRLWIND_RPC_TIMEOUT=86400
export WHIRLWIND_HEARTBEAT_FATAL_MISSES=100000

# terminal 1: every node except the reward node
GPUS=0,1 bash examples/gsm8k_dr_grpo_colocate.sh \
    --nodes orchestrator,rollout,dataloader,tracker,model_update

# terminal 2: the reward node, waiting for a debugger on port 5678
python -m debugpy --listen 5678 --wait-for-client \
    -m whirlwind.reward.reward_function --config configs/reward/function.yaml
```

Attach to `127.0.0.1:5678`, set a breakpoint in `RewardNode._compute_rewards`
(`whirlwind/reward/reward_function.py`), and the first training step stops
there. Start terminal 2 within a few minutes of terminal 1: at bringup the
training loop waits at most 5 minutes for each node it needs (10 minutes for
a colocated model node).

## How it works

`--nodes` lists the nodes bringup starts. For every other node it prints the
exact command, with the configs and GPUs it would have used:

```
# --- reward ---
PYTHONUNBUFFERED=1 python3 -m "whirlwind.reward.reward_function" --config "/path/to/whirlwind/configs/reward/function.yaml" --log-level "INFO"
```

Run that command yourself with the debugger in front of `-m`. A node is found
by its name (`ipc:///tmp/whirlwind-<WW_RUN_ID>-<node>`), so no other node
needs to know it was started differently, and the training loop simply waits
for it to register.

The run id has to match. The example recipes pick a new `WW_RUN_ID` for every
launch unless one is set, so export it in both terminals. `bringup_and_run.sh`
used directly sets none, which also matches a second terminal that sets none.

## Timeouts: keep the run alive while you are paused

A node stopped at a breakpoint answers nothing, neither heartbeats nor calls.
With the default settings the rest of the run gives up on it:

| while a node is paused | default | variable (read by) |
|---|---|---|
| the registry drops the node, and the training loop ends the run with "required node '…' lost" | 15 s | `WHIRLWIND_HEARTBEAT_TIMEOUT` (the training loop, which hosts the registry) |
| a call into the paused node times out, which ends the run | 300 s (600 s in the example recipes) | `WHIRLWIND_RPC_TIMEOUT` (the caller) |
| when the paused process is the training loop: every other node shuts itself down | 12 missed heartbeats, about 85 s | `WHIRLWIND_HEARTBEAT_FATAL_MISSES` (every node) |
| model node with several ranks: ranks waiting for a paused rank time out | 600 s | `WHIRLWIND_NCCL_TIMEOUT` (the model node) |

The exports in the short version raise the first three for every node, in
both terminals. With the default settings, a run with the reward node held at
a breakpoint ended 30 s in; with these settings it waited, the call returned
as soon as the debugger continued, and the other nodes also outlived a
100-second pause of the training loop.

With these settings the run no longer notices a node that has really died.
Stop it with Ctrl+C in the bringup terminal, and end the debug session
yourself.

## Which node, which command

| node to debug | `--nodes` for bringup | breakpoints worth setting |
|---|---|---|
| reward | `orchestrator,rollout,dataloader,tracker,model_update` | `RewardNode._compute_rewards`, `_raw_rewards` |
| dataloader | `orchestrator,rollout,reward,tracker,model_update` | `_handle_next_batch` |
| tracker | `orchestrator,rollout,reward,dataloader,model_update` | |
| rollout | `orchestrator,reward,dataloader,tracker,model_update` | `_generate` |
| model_update | `orchestrator,rollout,reward,dataloader,tracker` | `_model_update_step`, `_compute_loss`, `_compute_per_token_logps` |
| orchestrator (the training loop) | `rollout,reward,dataloader,tracker,model_update` | `run`, `_call_rollout`, `_call_reward`, `_call_train_accum` |

Run each node with the command bringup printed for it. For the rollout node,
keep its `CUDA_VISIBLE_DEVICES`; you debug the node's Python side, while the
engine's scheduler processes belong to SGLang or vLLM. The reward,
dataloader and tracker nodes need no GPU.

### The model node

`torchrun` starts one process per GPU. Debug one rank: use a one-GPU layout
and start rank 0 yourself, with the variables torchrun would set.

```bash
# terminal 1: the one-GPU recipe, without the model node
GPUS=0 bash examples/gsm8k_dr_grpo_colocate.sh \
    --nodes orchestrator,rollout,reward,dataloader,tracker

# terminal 2: rank 0 of 1 under the debugger
CUDA_VISIBLE_DEVICES=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29555 \
python -m debugpy --listen 5678 --wait-for-client -m whirlwind.model.model_fsdp2 \
    --config configs/model/fsdp2_base_colocate_gas1_fp32master_chunked_dr_grpo_world1.yaml
```

In VS Code you can instead launch `torch.distributed.run` as the module (see
[below](#vs-code)); the debugger follows the rank processes. With several
ranks each one stops at its own breakpoints, and the ranks that are not
paused wait in the next collective: raise `WHIRLWIND_NCCL_TIMEOUT`.

### The training loop

The training loop hosts the registry, so start it first and let the others
join it:

```bash
# terminal 1: the loop under the debugger (attach before starting terminal 2)
CUDA_VISIBLE_DEVICES= python -m debugpy --listen 5678 --wait-for-client \
    -m whirlwind.training.training_grpo --config configs/training/gsm8k_dr_grpo_colocate.yaml

# terminal 2: everything else
GPUS=0,1 bash examples/gsm8k_dr_grpo_colocate.sh \
    --nodes rollout,reward,dataloader,tracker,model_update
```

The other nodes retry registration for about two minutes, so start terminal 2
soon after attaching. The loop reads the rollout, model and reward configs
from `WHIRLWIND_ROLLOUT_CONFIG`, `WHIRLWIND_MODEL_CONFIG` and
`WHIRLWIND_REWARD_CONFIG`, and falls back to `rollout_yaml`, `model_yaml` and
`reward_yaml` in the training config. It takes the learning-rate schedule
from the model config. Bringup exports these variables only for the nodes it
starts, so export them in terminal 1 when they differ from the fallbacks. For
the two-GPU recipe above they are the same.

## VS Code

Put these in `.vscode/launch.json`, then run bringup with `--nodes` in a
terminal that has the exports from the short version.

```jsonc
{
  "version": "0.2.0",
  "configurations": [
    {
      // attach to a node started with `python -m debugpy --listen 5678 ...`
      "name": "whirlwind: attach (5678)",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "127.0.0.1", "port": 5678 },
      "justMyCode": false
    },
    {
      // or let VS Code start the node itself
      "name": "whirlwind: reward node (joins the live run)",
      "type": "debugpy",
      "request": "launch",
      "module": "whirlwind.reward.reward_function",
      "args": ["--config", "configs/reward/function.yaml"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "WW_RUN_ID": "debug",
        "WHIRLWIND_HEARTBEAT_TIMEOUT": "86400",
        "WHIRLWIND_RPC_TIMEOUT": "86400",
        "WHIRLWIND_HEARTBEAT_FATAL_MISSES": "100000"
      }
    },
    {
      // the model node, one rank under torchrun (one-GPU layout)
      "name": "whirlwind: model node (joins the live run)",
      "type": "debugpy",
      "request": "launch",
      "module": "torch.distributed.run",
      "args": [
        "--nproc_per_node", "1", "--master_port", "29555",
        "-m", "whirlwind.model.model_fsdp2",
        "--config", "configs/model/fsdp2_base_colocate_gas1_fp32master_chunked_dr_grpo_world1.yaml"
      ],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "CUDA_VISIBLE_DEVICES": "0",
        "WW_RUN_ID": "debug",
        "WHIRLWIND_HEARTBEAT_TIMEOUT": "86400",
        "WHIRLWIND_RPC_TIMEOUT": "86400",
        "WHIRLWIND_HEARTBEAT_FATAL_MISSES": "100000"
      }
    }
  ]
}
```

VS Code runs the node with the interpreter selected for the workspace; pick
the environment the rest of the run uses.

Without an IDE, `python -m pdb -m whirlwind.reward.reward_function --config ...`
works for any single-process node you start in your own terminal, and so does
a `breakpoint()` in its code. Nodes that bringup starts have no terminal
input, so use a breakpoint only in the node you launch yourself.

## Other tools

- `NODE_COMMS_DBG_STDOUT=1` prints every RPC as it happens, and
  `NODE_COMMS_TIMING=1` records call latencies.
- `--training-module whirlwind.training.training_grpo_capture` with
  `WW_CAPTURE_EVERY=N` and `WW_CAPTURE_DIR` saves a step's starting weights
  and its full `TrainStepRequest`, so the model update can be replayed
  offline.
- The numerics switches (`WW_PRINT_STEP1_DIAG`, `WW_IS_DIAG_DIR`,
  `WW_UPDATE_DIAG`, …) are listed in
  [TIPS_AND_TRICKS.md §8](../TIPS_AND_TRICKS.md#8-debugging-toolbox).
