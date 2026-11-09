#!/usr/bin/env bash
# EXPERIMENTAL — async proxy-rollout (actor-learner) GRPO: an always-on
# rollout engine behind a proxy with a replay buffer serves an external agent,
# and a separate FSDP2 LoRA updater trains on what the agent was served. Here
# a simulated agent plays the external system on GSM8K. See
# docs/async_proxy_rollout.md for the design, the API and the limitations.
#
#   ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_async_proxy.sh
#   STEPS=3 ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_async_proxy.sh   # smoke
#
# Non-colocate by design: the engine and the updater need one GPU each.
#
# Nodes (one process group each):
#   rollout-engine   SGLang on ENGINE_GPU, never sleeps, LoRA hot-swap
#   rollout          the proxy (CPU): generate_external, report_reward,
#                    pop_ready_batch, weight-version counter + replay buffer
#   model            FSDP2 LoRA updater on MODEL_GPU (torchrun, 1 process)
#   reward, dataloader, tracker   (CPU) validation and the agent's prompts
#   orchestrator     the async training loop (CPU)
#   sim-external-agent  drives generate_external and reports rewards
#
# Environment: WHIRLWIND_PYTHON / WHIRLWIND_TORCHRUN pick the interpreter;
# TRAINING_CONFIG, TRACKER_CONFIG, DATALOADER_CONFIG override the configs;
# LOG_DIR (default runs/async_proxy_gsm8k). Set WANDB_MODE=offline to train
# without a W&B account. Exits with the training loop's status.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${WHIRLWIND_PYTHON:-python3}"
TORCHRUN="${WHIRLWIND_TORCHRUN:-torchrun}"
ENGINE_GPU="${ENGINE_GPU:-0}"
MODEL_GPU="${MODEL_GPU:-1}"
STEPS="${STEPS:-65}"
SIM_BATCHES="${SIM_BATCHES:-0}"   # agent generate calls; 0 = until the run ends
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training/async_proxy_gsm8k.yaml}"
TRACKER_CONFIG="${TRACKER_CONFIG:-configs/tracker/wandb.yaml}"
DATALOADER_CONFIG="${DATALOADER_CONFIG:-configs/dataloader/gsm8k.yaml}"
ENGINE_CONFIG=configs/rollout/async_proxy_engine.yaml
PROXY_CONFIG=configs/rollout/async_proxy_1x.yaml
REWARD_CONFIG=configs/reward/function.yaml
MODEL_CONFIG=configs/model/fsdp2_async_lora_1gpu.yaml

if [[ "$ENGINE_GPU" == "$MODEL_GPU" ]]; then
	echo "ENGINE_GPU and MODEL_GPU must differ: the async mode does not colocate." >&2
	exit 2
fi

# The engine claims 85% of its GPU (mem_fraction_static in the engine config)
# and expects the GPU to itself; sharing it with another job crashed the
# engine (CUDA illegal memory access) in testing. Warn if either GPU is busy.
# (nvidia-smi numbers GPUs in PCI order; set CUDA_DEVICE_ORDER=PCI_BUS_ID so
# CUDA agrees.)
if command -v nvidia-smi >/dev/null 2>&1; then
	for g in "$ENGINE_GPU" "$MODEL_GPU"; do
		used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
		if [[ "${used:-0}" =~ ^[0-9]+$ ]] && ((used > 1024)); then
			echo "WARNING: GPU $g already has ${used} MiB in use; the async mode expects exclusive GPUs." >&2
		fi
	done
fi

export WW_RUN_ID="${WW_RUN_ID:-async-proxy-$$}"
export NODE_COMMS_ENSURE_REGISTER=1
export WHIRLWIND_RPC_TIMEOUT="${WHIRLWIND_RPC_TIMEOUT:-600}"
export WW_ASYNC_TOTAL_STEPS="$STEPS"
export WHIRLWIND_ROLLOUT_CONFIG="$PWD/$ENGINE_CONFIG"
export WHIRLWIND_REWARD_CONFIG="$PWD/$REWARD_CONFIG"
export WHIRLWIND_MODEL_CONFIG="$PWD/$MODEL_CONFIG"
MASTER_PORT=$((10000 + $(printf '%s' "$WW_RUN_ID" | cksum | awk '{print $1}') % 1000))

LOG_DIR="${LOG_DIR:-runs/async_proxy_gsm8k}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1
echo "=== EXPERIMENTAL async proxy GRPO: engine GPU $ENGINE_GPU, model GPU $MODEL_GPU, $STEPS steps — $(date) ==="

PIDS=()
cleanup() {
	local code=$?
	echo "=== shutting down ==="
	for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
	for _ in $(seq 1 20); do
		local alive=0
		for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
		((alive == 0)) && break
		sleep 1
	done
	for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
	# Reap only the nodes: a bare `wait` would also wait for the log's tee,
	# which cannot finish while this script still holds its pipe.
	for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
	exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start() { # name, then the command (run in its own process group)
	local name=$1
	shift
	setsid env PYTHONUNBUFFERED=1 "$@" &
	PIDS+=($!)
	echo ">>> $name: pid $!"
}

start rollout-engine env CUDA_VISIBLE_DEVICES="$ENGINE_GPU" \
	"$PYTHON" -m whirlwind.rollout.rollout_sglang --config "$ENGINE_CONFIG"
start rollout-proxy env CUDA_VISIBLE_DEVICES="" \
	"$PYTHON" -m whirlwind.rollout.rollout_proxy --config "$PROXY_CONFIG"
start reward "$PYTHON" -m whirlwind.reward.reward_function --config "$REWARD_CONFIG"
start dataloader "$PYTHON" -m whirlwind.data.dataloader_node --config "$DATALOADER_CONFIG"
start tracker "$PYTHON" -m whirlwind.tracker.tracker_wandb --config "$TRACKER_CONFIG"
start model env CUDA_VISIBLE_DEVICES="$MODEL_GPU" \
	"$TORCHRUN" --nproc_per_node=1 --master_port="$MASTER_PORT" \
	-m whirlwind.model.model_fsdp2 --config "$MODEL_CONFIG"
start orchestrator env CUDA_VISIBLE_DEVICES="" \
	"$PYTHON" -m whirlwind.training.training_async --config "$TRAINING_CONFIG"
ORCH_PID=$!
# The agent registers with the cluster's registry (hosted by the orchestrator);
# give the cluster a head start, then let it drive generation until the run ends.
start sim-external-agent bash -c "sleep 30 && exec env CUDA_VISIBLE_DEVICES='' '$PYTHON' -m whirlwind.training.sim_external_agent \
	--group-size 4 --batches '$SIM_BATCHES' --reward-config '$REWARD_CONFIG' --client-id sim"

rc=0
wait "$ORCH_PID" || rc=$?
echo "=== training loop exited with status $rc — $(date) ==="
exit "$rc"
