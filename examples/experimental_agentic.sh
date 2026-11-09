#!/usr/bin/env bash
# EXPERIMENTAL — agentic (multi-turn) GRPO. An always-on engine behind the
# agent proxy serves a multi-turn agent; the agentic training loop regroups the
# served turns into episodes and trains a FSDP2 LoRA adapter on them. See
# docs/agentic_multiturn.md. Two tasks:
#
#   TASK=os_tools (default)  whirlwind.training.sim_os_agent: procedurally
#       generated file-system / clock questions answered by calling tools
#       (list_dir, find_file, read_file, get_time); 128 training tasks,
#       held-out evaluation logged by the agent (os_eval/*). The tools run
#       in a podman container (OS_BACKEND=podman, image OS_IMAGE);
#       OS_BACKEND=memory simulates them in Python instead. OS_CONSTRAIN=1
#       (default) enforces the <tool_call> format by constrained decoding.
#   TASK=reflect             whirlwind.training.sim_reflect_agent: GSM8K with
#       up to MAX_TURNS tries and a "reflect and try again" message after
#       each wrong answer; GSM8K test accuracy as validation.
#
#   ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_agentic.sh
#   TASK=reflect ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_agentic.sh
#   STEPS=3 ENGINE_GPU=0 MODEL_GPU=1 bash examples/experimental_agentic.sh   # smoke
#   ENGINE_GPU=0 MODEL_GPU=0 bash examples/experimental_agentic.sh   # one GPU (os_tools)
#
# With ENGINE_GPU == MODEL_GPU the os_tools task runs in the shared-GPU
# layout: the engine is drained and put to sleep for every optimizer step.
#
# Nodes (one process group each):
#   rollout-engine   SGLang on ENGINE_GPU (rollout_sglang_agent), LoRA hot-swap
#   rollout          the agent proxy (CPU): generate_external, report_reward,
#                    touch, pop_ready_batch + replay buffer
#   model            FSDP2 LoRA updater on MODEL_GPU (torchrun, 1 process)
#   tracker          (CPU); TASK=reflect also starts reward + dataloader
#   orchestrator     the agentic training loop (CPU)
#   the agent        sim-os-agent or sim-reflect-agent
#
# Environment: WHIRLWIND_PYTHON / WHIRLWIND_TORCHRUN pick the interpreter;
# TRAINING_CONFIG, TRACKER_CONFIG, DATALOADER_CONFIG override the configs;
# GROUP_SIZE and MAX_TURNS shape either agent, CARRIER (flat|messages) the
# reflect agent, EVIDENCE_REWARD (0.2) the OS agent; LOG_DIR (default
# runs/agentic_<task>). Set WANDB_MODE=offline to train without a W&B account.
# Exits with the training loop's status.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${WHIRLWIND_PYTHON:-python3}"
TORCHRUN="${WHIRLWIND_TORCHRUN:-torchrun}"
ENGINE_GPU="${ENGINE_GPU:-0}"
MODEL_GPU="${MODEL_GPU:-1}"
TASK="${TASK:-os_tools}"
TRACKER_CONFIG="${TRACKER_CONFIG:-configs/tracker/wandb.yaml}"
DATALOADER_CONFIG="${DATALOADER_CONFIG:-configs/dataloader/gsm8k.yaml}"
PROXY_CONFIG=configs/rollout/agentic_proxy.yaml
REWARD_CONFIG=configs/reward/function.yaml
case "$TASK" in
os_tools)
	STEPS="${STEPS:-100}"
	GROUP_SIZE="${GROUP_SIZE:-8}"
	MAX_TURNS="${MAX_TURNS:-4}"
	EVIDENCE_REWARD="${EVIDENCE_REWARD:-0.2}"
	OS_BACKEND="${OS_BACKEND:-podman}"
	OS_IMAGE="${OS_IMAGE:-docker.io/library/python:3.12-slim}"
	OS_CONSTRAIN="${OS_CONSTRAIN:-1}" # 1: grammar-enforced <tool_call> turns + answer tool
	if [[ "$OS_BACKEND" == podman ]] && ! command -v podman >/dev/null 2>&1; then
		echo "OS_BACKEND=podman needs podman on PATH (or set OS_BACKEND=memory)." >&2
		exit 2
	fi
	if [[ "$ENGINE_GPU" == "$MODEL_GPU" ]]; then # shared-GPU layout
		TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training/agentic_os_tools_shared.yaml}"
		ENGINE_CONFIG=configs/rollout/agentic_engine_tools_shared.yaml
	else
		TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training/agentic_os_tools.yaml}"
		ENGINE_CONFIG=configs/rollout/agentic_engine_tools.yaml
	fi
	MODEL_CONFIG=configs/model/fsdp2_agentic_tools_lora_1gpu.yaml
	;;
reflect)
	STEPS="${STEPS:-40}"
	GROUP_SIZE="${GROUP_SIZE:-4}"
	MAX_TURNS="${MAX_TURNS:-3}"
	CARRIER="${CARRIER:-flat}"
	TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training/agentic_reflect_gsm8k.yaml}"
	ENGINE_CONFIG=configs/rollout/agentic_engine.yaml
	MODEL_CONFIG=configs/model/fsdp2_async_lora_1gpu.yaml
	;;
*)
	echo "TASK must be os_tools or reflect (got '$TASK')." >&2
	exit 2
	;;
esac

if [[ "$ENGINE_GPU" == "$MODEL_GPU" && "$TASK" != os_tools ]]; then
	echo "TASK=$TASK needs ENGINE_GPU != MODEL_GPU (only os_tools ships shared-GPU configs; see docs/agentic_multiturn.md)." >&2
	exit 2
fi

# The engine claims 85% of its GPU and expects it to itself; sharing a GPU
# with another job crashed the engine in testing. Warn if either GPU is busy.
# (nvidia-smi numbers GPUs in PCI order; set CUDA_DEVICE_ORDER=PCI_BUS_ID so
# CUDA agrees.)
if command -v nvidia-smi >/dev/null 2>&1; then
	for g in "$ENGINE_GPU" "$MODEL_GPU"; do
		used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
		if [[ "${used:-0}" =~ ^[0-9]+$ ]] && ((used > 1024)); then
			echo "WARNING: GPU $g already has ${used} MiB in use; the agentic mode expects exclusive GPUs." >&2
		fi
	done
fi

export WW_RUN_ID="${WW_RUN_ID:-agentic-$TASK-$$}"
export NODE_COMMS_ENSURE_REGISTER=1
export WHIRLWIND_RPC_TIMEOUT="${WHIRLWIND_RPC_TIMEOUT:-900}"
export WW_ASYNC_TOTAL_STEPS="$STEPS"
export WHIRLWIND_ROLLOUT_CONFIG="$PWD/$ENGINE_CONFIG"
export WHIRLWIND_REWARD_CONFIG="$PWD/$REWARD_CONFIG"
export WHIRLWIND_MODEL_CONFIG="$PWD/$MODEL_CONFIG"
MASTER_PORT=$((10000 + $(printf '%s' "$WW_RUN_ID" | cksum | awk '{print $1}') % 1000))

LOG_DIR="${LOG_DIR:-runs/agentic_$TASK}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1
echo "=== EXPERIMENTAL agentic GRPO ($TASK): engine GPU $ENGINE_GPU, model GPU $MODEL_GPU, $STEPS steps, K=$GROUP_SIZE, $MAX_TURNS turns — $(date) ==="

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
	if [[ "$TASK" == os_tools && "${OS_BACKEND:-}" == podman ]]; then # the agent's task containers
		podman ps -aq --filter "label=whirlwind.sim_os_agent=$WW_RUN_ID" | xargs -r podman rm -f -t 0 >/dev/null 2>&1 || true
	fi
	# Reap only the nodes: a bare `wait` would also wait for the log's tee.
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
	"$PYTHON" -m whirlwind.rollout.rollout_sglang_agent --config "$ENGINE_CONFIG"
start rollout-proxy env CUDA_VISIBLE_DEVICES="" \
	"$PYTHON" -m whirlwind.rollout.rollout_proxy_agent --config "$PROXY_CONFIG"
if [[ "$TASK" == reflect ]]; then # validation + the agent's problems
	start reward "$PYTHON" -m whirlwind.reward.reward_function --config "$REWARD_CONFIG"
	start dataloader "$PYTHON" -m whirlwind.data.dataloader_node --config "$DATALOADER_CONFIG"
fi
start tracker "$PYTHON" -m whirlwind.tracker.tracker_wandb --config "$TRACKER_CONFIG"
start model env CUDA_VISIBLE_DEVICES="$MODEL_GPU" \
	"$TORCHRUN" --nproc_per_node=1 --master_port="$MASTER_PORT" \
	-m whirlwind.model.model_fsdp2 --config "$MODEL_CONFIG"
start orchestrator env CUDA_VISIBLE_DEVICES="" \
	"$PYTHON" -m whirlwind.training.training_agentic --config "$TRAINING_CONFIG"
ORCH_PID=$!
# The agent registers with the cluster's registry (hosted by the orchestrator);
# give the cluster a head start, then let it play episodes until the run ends.
if [[ "$TASK" == os_tools ]]; then
	start sim-os-agent bash -c "sleep 30 && exec env CUDA_VISIBLE_DEVICES='' '$PYTHON' -m whirlwind.training.sim_os_agent \
	--group-size '$GROUP_SIZE' --max-turns '$MAX_TURNS' --evidence-reward '$EVIDENCE_REWARD' \
	--backend '$OS_BACKEND' --image '$OS_IMAGE' --container-label '$WW_RUN_ID' \
	--trajectory-log '$LOG_DIR/os_trajectories.jsonl' --client-id sim-os \
	$([[ "$OS_CONSTRAIN" == 1 ]] && echo --constrain-calls)"
else
	start sim-reflect-agent bash -c "sleep 30 && exec env CUDA_VISIBLE_DEVICES='' '$PYTHON' -m whirlwind.training.sim_reflect_agent \
	--group-size '$GROUP_SIZE' --max-turns '$MAX_TURNS' --carrier '$CARRIER' --reward-config '$REWARD_CONFIG' --client-id sim-reflect"
fi

rc=0
wait "$ORCH_PID" || rc=$?
echo "=== training loop exited with status $rc — $(date) ==="
exit "$rc"
