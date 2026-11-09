# Launch Guide

Every stage of the loop is its own process. `bringup_and_run.sh` starts them
with the right GPUs and configs, waits for training to finish, tears
everything down, and exits with the training loop's status. The recipes
under `examples/` wrap it.

## Quick start

```bash
GPUS=0,1 bash examples/gsm8k_dr_grpo_colocate.sh   # two GPUs, rollout and trainer colocated
GPUS=0   bash examples/gsm8k_dr_grpo_colocate.sh   # one GPU
./bringup_and_run.sh --dry-run                     # validate the default configs only
```

The example recipes pick the shipped 1- or 2-GPU rollout and model configs
from `GPUS`; for other GPU counts set `ROLLOUT_CONFIG` and `MODEL_CONFIG`.
Extra arguments go to `bringup_and_run.sh` and override the recipe's flags,
e.g. `--training-config my_run.yaml`. Logs go to `runs/<recipe>/driver.log`;
set `WANDB_MODE=offline` (or use a tracker config with `backend: "none"`) to
run without a W&B account.

The experimental async proxy-rollout and agentic modes (agent
post-training) do not go through `bringup_and_run.sh`; they have their own
launchers, `examples/experimental_async_proxy.sh` (see
[async_proxy_rollout.md](async_proxy_rollout.md)) and
`examples/experimental_agentic.sh` (see
[agentic_multiturn.md](agentic_multiturn.md)).

## `bringup_and_run.sh`

```bash
# rollout engine and trainer on separate GPUs
./bringup_and_run.sh --rollout-gpus 0 --model-gpus 1,2

# colocated: the same GPUs for both, with colocate configs
# (enable_memory_saver in the rollout config, memory_management_enabled in
# the training config)
./bringup_and_run.sh --rollout-gpus 0,1 --model-gpus 0,1 \
    --rollout-config configs/rollout/sglang_colocate_tp2.yaml \
    --model-config configs/model/fsdp2_base_colocate_gas1_fp32master_chunked_dr_grpo.yaml \
    --training-config configs/training/gsm8k_dr_grpo_colocate.yaml \
    --dataloader-config configs/dataloader/gsm8k.yaml
```

What it does:

1. Starts the training loop (orchestrator), dataloader, rollout, reward and
   tracker nodes, and the model node under `torchrun --nproc_per_node=N`
   (N = number of `--model-gpus`). Each node runs in its own process group
   with its own `CUDA_VISIBLE_DEVICES`.
2. Exports `WHIRLWIND_*_CONFIG` so the training config's `$ENV:fallback`
   paths resolve to the files passed on the command line.
3. Waits for the training loop. After it finishes, service nodes get a
   grace period (`WW_BRINGUP_GRACE_S`, default 120 s) to exit on their own.
4. Tears down every process group (SIGTERM, then SIGKILL after
   `WW_BRINGUP_TERM_WAIT_S`, default 20 s) and sweeps orphaned rank workers
   on the run's GPUs.
5. Exits with the training loop's status: non-zero if it crashed or stopped
   early because a required node was lost.

Module flags select implementations without code changes:
`--rollout-module`, `--model-module`, `--reward-module`, `--training-module`.
Config flags: `--rollout-config`, `--model-config`, `--reward-config`,
`--training-config`, `--tracker-config`, `--dataloader-config`. Run
`./bringup_and_run.sh --help` for the full list.

Environment:

| variable | default | effect |
|---|---|---|
| `WHIRLWIND_PYTHON` | `python3` on PATH | interpreter for every node (point it at a venv or conda env) |
| `WHIRLWIND_TORCHRUN` | `torchrun` on PATH | launcher for the model node |
| `WHIRLWIND_TORCHRUN_ARGS` | unset | extra torchrun arguments for the model node, e.g. a multi-node group ([below](#multi-node-model-update)) |
| `WW_RUN_ID` | unset | namespaces sockets, checkpoints and the torchrun port so several runs can share a host |
| `WHIRLWIND_RPC_TIMEOUT` | 300 | RPC timeout in seconds |

Everything else in the environment (`CUDA_*`, `NCCL_*`, `HF_*`, `WANDB_*`,
`WHIRLWIND_*`) is passed through to the nodes.

**Partial launch** — start some nodes, print the commands for the rest:

```bash
./bringup_and_run.sh --rollout-gpus 0 --model-gpus 1,2 --nodes orchestrator,rollout
```

This is also how you run one node under a debugger while the rest of the
run is live: [debugging.md](debugging.md).

## Manual launch

Each node is a module with `--config` and `--log-level`; start the training
loop first.

```bash
python3 -m whirlwind.training.training_grpo  --config configs/training/grpo.yaml
python3 -m whirlwind.data.dataloader_node    --config configs/dataloader/datasets.yaml
CUDA_VISIBLE_DEVICES=0 python3 -m whirlwind.rollout.rollout_sglang --config configs/rollout/sglang.yaml
python3 -m whirlwind.reward.reward_function  --config configs/reward/function.yaml
python3 -m whirlwind.tracker.tracker_wandb   --config configs/tracker/wandb.yaml
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 \
    -m whirlwind.model.model_fsdp2 --config configs/model/fsdp2.yaml
```

Export `NODE_COMMS_ENSURE_REGISTER=1` so nodes started before the training
loop retry registration instead of starting unregistered.

## GPU assignment

| component | GPU | how |
|---|---|---|
| training loop, dataloader, reward, tracker | no | — |
| rollout | yes | `--rollout-gpus` (tensor- or data-parallel size in the rollout config) |
| model update | yes, N | `--model-gpus`, one torchrun rank per GPU |

Separate GPU sets need no special configs. Shared GPUs (colocate) need the
colocate configs: the engine sleeps during the update and the trainer
offloads optimizer state during rollout.

## Multi-node model update

Only the model node spans machines. Node-to-node RPC runs over local IPC
sockets, so the training loop, dataloader, rollout, reward and tracker nodes,
and model rank 0 (the rank that answers RPC), all run on one head node. The
model ranks on other machines get their work from rank 0 over NCCL.

```bash
# head node: every node, plus model ranks 0-3
WHIRLWIND_TORCHRUN_ARGS="--nnodes=2 --node_rank=0 --master_addr=$HEAD" \
./bringup_and_run.sh --rollout-gpus 0,1,2,3 --model-gpus 0,1,2,3 \
    --master-port 29500 --model-config my_model.yaml ...

# each other node (node_rank 1, 2, ...): model ranks only
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=2 --node_rank=1 \
    --master_addr=$HEAD --master_port=29500 --nproc_per_node=4 \
    -m whirlwind.model.model_fsdp2 --config my_model.yaml
```

- `HEAD` is an address of the head node that the other nodes can reach.
  Pass the same port everywhere: `--master-port` on the head (otherwise
  bringup derives one from `WW_RUN_ID`), `--master_port` on the others.
- Set `fsdp_sharding_strategy: HYBRID_SHARD` in the model config. It shards
  each parameter over one node's GPUs and replicates it across nodes, so
  parameter all-gathers and gradient reduce-scatters stay inside a node and
  only one gradient all-reduce per step crosses the network. `FULL_SHARD`
  also runs across nodes, but shards over every rank, so all of that traffic
  crosses the network.
- Every node runs the same number of model ranks with the same model config
  and needs the weights at `model_path`. Set `fsdp_size` to the total number
  of ranks.
- `workdir` must be on a filesystem that every node can see: each rank
  writes its own hot-checkpoint shards.
- Each step's batch is split over all model ranks.

To try it on one machine, start the two torchrun groups on disjoint GPUs
(`CUDA_VISIBLE_DEVICES`), with `--master_addr=127.0.0.1` and a different
`NCCL_HOSTID` for each. Without distinct host IDs NCCL sees a single host,
tries to reach the other group's GPUs directly, and fails with "invalid
device ordinal".

## Datasets

The dataloader node serves prompts from Hugging Face datasets configured in
`configs/dataloader/*.yaml` (`configs/dataloader/gsm8k.yaml` is the reference
recipe). It loads every configured dataset at startup, keeps independent
shuffling, batching and position per dataset, builds the system prompt from
the template plus few-shot examples, and saves its position in checkpoints.

## Checkpoints and resumption

With `workdir` set in the training config:

- **Hot checkpoints** (`hot_checkpoint_every_n_steps`) resume a killed run.
  If `workdir/hot_cp/` exists at startup, the training loop resumes from it
  automatically; started by hand, it also takes `--load-hot-checkpoint PATH`.
  Restored: the step counter, the dataloader position, model weights and
  optimizer state (DCP shards), and the rollout weights. The LR schedule is
  recomputed from the step.
- **Cold checkpoints** (`cold_checkpoint_steps`) are consolidated HF weights
  (`cold_cp_step_N/`) for evaluation and inference, not for resumption.

```
workdir/
├── config.json          # every node's config
├── state.json           # step counter + node states (written after the shards)
├── hot_cp/              # DCP shards: model + optimizer
└── cold_cp_step_N/      # HF weights
```

## Common issues

**"Not all nodes discovered within 300s"** — a node failed during startup;
its traceback is in the log above the message. Check GPU memory on the
assigned GPUs and the node's config.

**SIGSEGV in `ncclCommInitRank`** (the rollout never registers, or the model
node dies right after its first weight export) — the host's InfiniBand
stack. Set `NCCL_IB_DISABLE=1`.

**The rollout node never registers, with no error** — on a multi-socket host,
a tensor-parallel engine (or an FSDP group) spread over GPUs on different
sockets can hang in NCCL setup. Pick GPUs that share a link (`NV#`, `PIX` or
`PXB` in `nvidia-smi topo -m`).

**ZMQ "ipc path ... is too long"** — IPC socket paths are limited to about
107 characters and the engine places its sockets under `TMPDIR`; use a short
`TMPDIR`.

**"RPC call timed out"** — raise `WHIRLWIND_RPC_TIMEOUT` (long generations,
long-context validation).

**"scatter_object_list mismatch" or hung collectives** — all FSDP ranks must
reach the same collectives in the same order; a handler that runs on rank 0
without being dispatched through the standby loop desynchronizes them.

**`/dev/shm` full** — weight exports and large RPC payloads use `/dev/shm`.
Check `df -h /dev/shm`, remove stale `/dev/shm/whirlwind-shm-*` files after a
crash, or point the model config's `checkpoint_dir` elsewhere.

More in [TIPS_AND_TRICKS.md](../TIPS_AND_TRICKS.md).
