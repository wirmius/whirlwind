#!/usr/bin/env bash
# bringup_and_run.sh — start whirlwind components and run training.
#
# Usage (start everything, the default):
#   ./bringup_and_run.sh --rollout-gpus 0 --model-gpus 1,2
#
# Colocated (rollout engine and trainer share the same GPUs via sleep/wake;
# use colocate configs, see examples/):
#   ./bringup_and_run.sh --rollout-gpus 0,1 --model-gpus 0,1 ...
#
# Usage (start only some nodes; print commands for the rest):
#   ./bringup_and_run.sh \
#       --rollout-gpus 0 \
#       --model-gpus 1,2 \
#       --nodes orchestrator,rollout
#
#   This starts orchestrator + rollout, then prints the shell commands
#   you need to run reward, tracker, and model_update in other terminals.
#
# Available nodes:
#   orchestrator   — binds ZMQ sockets, drives the training loop
#   rollout        — rollout engine (SGLang or vLLM), handles generation
#   reward         — computes rewards / advantages
#   dataloader     — loads HF datasets, serves batches
#   tracker        — logs metrics (W&B or none; CPU)
#   model_update   — FSDP training via torchrun
#
# When orchestrator is started by this script (default if in --nodes),
# the script waits for it to finish training before cleaning up, and exits
# with the training loop's status (non-zero if it crashed or stopped early
# because a required node was lost).
#
# Environment:
#   WHIRLWIND_PYTHON     interpreter for every node (default: python3 on PATH)
#   WHIRLWIND_TORCHRUN   torchrun launcher for the model node (default: torchrun on PATH)
#   WHIRLWIND_TORCHRUN_ARGS  extra torchrun arguments for the model node, e.g.
#                        "--nnodes=2 --node_rank=0 --master_addr=HEAD" to put
#                        model ranks on more machines (docs/launch.md)
#   WW_RUN_ID            namespaces sockets, checkpoints and the torchrun port,
#                        so several runs can share one host
#   WW_BRINGUP_GRACE_S, WW_BRINGUP_TERM_WAIT_S   teardown timing (see below)
#   Everything else (CUDA_*, NCCL_*, HF_*, WANDB_*, WHIRLWIND_*) is passed
#   through to the nodes unchanged.

set -euo pipefail

# Fail-fast node registration: nodes retry until the orchestrator's ctrl
# socket is up (12 attempts, backoff) instead of silently starting
# unregistered. Without this, a start-up race (nodes up before the
# orchestrator binds its socket) can leave the model node undiscovered: the
# training loop waits forever and the non-lead torchrun ranks die at the
# init barrier after the NCCL timeout.
export NODE_COMMS_ENSURE_REGISTER="${NODE_COMMS_ENSURE_REGISTER:-1}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ROLLOUT_GPUS=""
MODEL_GPUS=""
ROLLOUT_CONFIG="configs/rollout/sglang.yaml"
ROLLOUT_MODULE="whirlwind.rollout.rollout_sglang"  # python -m module for the rollout node (sglang | vllm)
# python -m module for the model-update node (default: manual FSDP2). Override with
# --model-module to swap in a sibling, e.g. the HF-Trainer-backed node
# (whirlwind.model.model_updater_hf_trainer).
MODEL_MODULE="whirlwind.model.model_fsdp2"
REWARD_CONFIG="configs/reward/function.yaml"
# python -m module for the reward node. Override with --reward-module to swap
# in a variant (e.g. examples.length_penalty_reward).
REWARD_MODULE="whirlwind.reward.reward_function"
MODEL_CONFIG="configs/model/fsdp2.yaml"
TRAINING_CONFIG="configs/training/grpo.yaml"
# python -m module for the training orchestrator loop. Override with
# --training-module to swap in a variant (e.g. the diagnostic-capture loop:
# whirlwind.training.training_grpo_capture).
TRAINING_MODULE="whirlwind.training.training_grpo"
TRACKER_CONFIG="configs/tracker/wandb.yaml"
DATA_LOADER_CONFIG="configs/dataloader/datasets.yaml"
LOG_LEVEL="INFO"
DRY_RUN=false
# torchrun master port (default 29500 collides if another torchrun is running;
# override via --master-port, or auto-derived from WW_RUN_ID via cksum when
# WW_RUN_ID is set and --master-port is not given — mapped to [10000,10999]
# so concurrent WW runs (different WW_RUN_IDs) get distinct ports). Passed to
# the model_update node's torchrun.
MASTER_PORT="29500"
_MASTER_PORT_SET=false
# Comma-separated node list; empty means "all"
NODE_SPEC=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Interpreter and torchrun used for every node (a venv or conda env: point
# these at its binaries, or activate it before calling this script).
PYTHON="${WHIRLWIND_PYTHON:-python3}"
TORCHRUN="${WHIRLWIND_TORCHRUN:-torchrun}"
# Extra torchrun arguments for the model node (split on whitespace).
read -ra TORCHRUN_ARGS <<<"${WHIRLWIND_TORCHRUN_ARGS:-}"

ALL_NODES=(orchestrator rollout reward dataloader tracker model_update)

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
	case "$1" in
	--rollout-gpus)
		ROLLOUT_GPUS="$2"
		shift 2
		;;
	--model-gpus)
		MODEL_GPUS="$2"
		shift 2
		;;
	--rollout-config)
		ROLLOUT_CONFIG="$2"
		shift 2
		;;
	--rollout-module)
		ROLLOUT_MODULE="$2"
		shift 2
		;;
	--reward-config)
		REWARD_CONFIG="$2"
		shift 2
		;;
	--reward-module)
		REWARD_MODULE="$2"
		shift 2
		;;
	--model-config)
		MODEL_CONFIG="$2"
		shift 2
		;;
	--model-module)
		MODEL_MODULE="$2"
		shift 2
		;;
	--tracker-config)
		TRACKER_CONFIG="$2"
		shift 2
		;;
	--dataloader-config)
		DATA_LOADER_CONFIG="$2"
		shift 2
		;;
	--training-config)
		TRAINING_CONFIG="$2"
		shift 2
		;;
	--training-module)
		TRAINING_MODULE="$2"
		shift 2
		;;
	--log-level)
		LOG_LEVEL="$2"
		shift 2
		;;
	--master-port)
		MASTER_PORT="$2"
		_MASTER_PORT_SET=true
		shift 2
		;;

	--nodes)
		NODE_SPEC="$2"
		shift 2
		;;
	--dry-run)
		DRY_RUN=true
		shift
		;;
	--help | -h)
		echo "Usage: $0 [options]"
		echo ""
		echo "Options:"
		echo "  --nodes NODES           Comma-separated node names to start"
		echo "                          (default: all).  Available:"
		echo "                          orchestrator, rollout, reward, dataloader, tracker, model_update"
		echo "                          Remaining nodes are printed as shell commands."
		echo ""
		echo "  Hot checkpoint auto-detected from config's workdir on startup."
		echo ""
		echo "  --rollout-gpus GPUs     GPU(s) for the rollout engine (e.g. 0 or 0,1)"
		echo "  --model-gpus GPUs       GPU(s) for FSDP training (e.g. 1,2); the same list as"
		echo "                          --rollout-gpus for colocated runs"
		echo "  --model-module MODULE   python -m module for the model-update node"
		echo "                          (default: whirlwind.model.model_fsdp2;"
		echo "                           HF-Trainer-backed: whirlwind.model.model_updater_hf_trainer)"
		echo "  --rollout-config PATH   YAML config for rollout node"
		echo "  --rollout-module MOD    python -m module for rollout"
		echo "                          (whirlwind.rollout.rollout_sglang | whirlwind.rollout.rollout_vllm)"
		echo "  --reward-config PATH    YAML config for reward node"
		echo "  --reward-module MODULE  python -m module for the reward node"
		echo "                          (default: whirlwind.reward.reward_function;"
		echo "                           e.g. examples.length_penalty_reward)"
		echo "  --model-config PATH     YAML config for model update node"
		echo "  --tracker-config PATH   YAML config for progress tracker node"
		echo "  --dataloader-config PATH  YAML config for dataloader node"
		echo "  --training-config PATH    YAML config for training orchestrator"
		echo "  --training-module MODULE  python -m module for the training loop"
		echo "                          (default: whirlwind.training.training_grpo)"
		echo "  --log-level LEVEL       Logging level (default: INFO)"
		echo "  --master-port PORT      torchrun master port (default: 29500, or derived from"
		echo "                          WW_RUN_ID; override if another torchrun is running)"
		echo "  --dry-run               Validate configs only, don't start"
		echo ""
		echo "Environment: WHIRLWIND_PYTHON, WHIRLWIND_TORCHRUN, WW_RUN_ID (see the script header)."
		echo ""
		echo "Examples:"
		echo "  # Start everything, rollout and trainer on separate GPUs:"
		echo "  $0 --rollout-gpus 0 --model-gpus 1,2"
		echo ""
		echo "  # Start orchestrator + rollout only; print commands for the rest:"
		echo "  $0 --rollout-gpus 0 --model-gpus 1,2 --nodes orchestrator,rollout"
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		exit 1
		;;
	esac
done

# Auto-derive the torchrun master port from WW_RUN_ID when not explicitly given
# (stable: cksum is deterministic; range [10000,10999]). Enables multiple WW
# runs on one host from a single env var, alongside the socket/checkpoint
# namespacing in whirlcoms/_runid.py + config.py.
if [ -n "${WW_RUN_ID:-}" ] && [ "$_MASTER_PORT_SET" = false ]; then
	_CRC=$(printf '%s' "$WW_RUN_ID" | cksum | awk '{print $1}')
	MASTER_PORT=$(( 10000 + _CRC % 1000 ))
	echo "[bringup] WW_RUN_ID=$WW_RUN_ID -> derived torchrun master port $MASTER_PORT"
fi

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
ROLLOUT_CONFIG="$(cd "$SCRIPT_DIR" && realpath -m "$ROLLOUT_CONFIG" 2>/dev/null || echo "$SCRIPT_DIR/$ROLLOUT_CONFIG")"
REWARD_CONFIG="$(cd "$SCRIPT_DIR" && realpath -m "$REWARD_CONFIG" 2>/dev/null || echo "$SCRIPT_DIR/$REWARD_CONFIG")"
MODEL_CONFIG="$(cd "$SCRIPT_DIR" && realpath -m "$MODEL_CONFIG" 2>/dev/null || echo "$SCRIPT_DIR/$MODEL_CONFIG")"
TRAINING_CONFIG="$(cd "$SCRIPT_DIR" && realpath -m "$TRAINING_CONFIG" 2>/dev/null || echo "$SCRIPT_DIR/$TRAINING_CONFIG")"
TRACKER_CONFIG="$(cd "$SCRIPT_DIR" && realpath -m "$TRACKER_CONFIG" 2>/dev/null || echo "$SCRIPT_DIR/$TRACKER_CONFIG")"
DATA_LOADER_CONFIG="$(cd "$SCRIPT_DIR" && realpath -m "$DATA_LOADER_CONFIG" 2>/dev/null || echo "$SCRIPT_DIR/$DATA_LOADER_CONFIG")"

# Export as env vars so configs can reference them via $WHIRLWIND_*
# Only set if not already defined externally.
export WHIRLWIND_ROLLOUT_CONFIG="${WHIRLWIND_ROLLOUT_CONFIG:-$ROLLOUT_CONFIG}"
export WHIRLWIND_REWARD_CONFIG="${WHIRLWIND_REWARD_CONFIG:-$REWARD_CONFIG}"
export WHIRLWIND_MODEL_CONFIG="${WHIRLWIND_MODEL_CONFIG:-$MODEL_CONFIG}"
export WHIRLWIND_TRAINING_CONFIG="${WHIRLWIND_TRAINING_CONFIG:-$TRAINING_CONFIG}"
export WHIRLWIND_TRACKER_CONFIG="${WHIRLWIND_TRACKER_CONFIG:-$TRACKER_CONFIG}"
export WHIRLWIND_DATA_LOADER_CONFIG="${WHIRLWIND_DATA_LOADER_CONFIG:-$DATA_LOADER_CONFIG}"

# ---------------------------------------------------------------------------
# Parse node spec into arrays: managed[] (started by script) and remaining[] (printed)
# ---------------------------------------------------------------------------
declare -a MANAGED=()
declare -a REMAINING=()

if [[ -n "$NODE_SPEC" ]]; then
	IFS=',' read -ra REQUESTED <<<"$NODE_SPEC"
	for node in "${ALL_NODES[@]}"; do
		found=false
		for req in "${REQUESTED[@]}"; do
			req="$(echo "$req" | xargs)" # trim whitespace
			if [[ "$req" == "$node" ]]; then
				found=true
				break
			fi
		done
		if $found; then
			MANAGED+=("$node")
		else
			REMAINING+=("$node")
		fi
	done
else
	# No --nodes specified: manage all
	MANAGED=("${ALL_NODES[@]}")
fi

# ---------------------------------------------------------------------------
# PID tracking and cleanup
# ---------------------------------------------------------------------------
# Every managed node is launched under `setsid`: it becomes a session/
# process-group leader with PGID == PID, so `kill -TERM -- -$pid` reaches the
# WHOLE process tree (torchrun rank workers, sglang scheduler/detokenizer
# subprocesses) — not just the top-level python. Without this, killing the
# node PID orphans the engine children (the engine's scheduler processes
# would outlive the run and keep their GPU memory).
PIDS=()
TRAINER_PID=""
# Set when the orchestrator node is managed by this script. Used to bound
# the rank-teardown wedge: torchrun can outlive a COMPLETED run when a rank
# spins in NCCL teardown — the wait loop must not block forever on it.
ORCHESTRATOR_PID=""

# TERM the whole process group of every tracked PID; escalate to KILL after
# TERM_WAIT seconds; reap. Safe to call multiple times (traps + grace path).
cleanup() {
	local code=$?
	echo ""
	echo "=== Shutting down ==="
	for pid in "${PIDS[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then
			echo "  SIGTERM -> process group $pid"
			kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
		fi
	done
	# Bounded graceful window (torchrun/sglang install signal handlers that
	# need a few seconds to flush).
	local waited=0 alive=0
	local term_wait="${WW_BRINGUP_TERM_WAIT_S:-20}"
	while (( waited < term_wait )); do
		alive=0
		for pid in "${PIDS[@]}"; do
			kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
		done
		(( alive == 0 )) && break
		sleep 1
		waited=$((waited + 1))
	done
	# Escalate: SIGKILL any surviving groups (wedged NCCL / CUDA teardown).
	for pid in "${PIDS[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then
			echo "  SIGKILL -> process group $pid (still alive after ${term_wait}s)"
			kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
		fi
	done
	# Sweep for orphaned rank workers: torchrun's elastic agent spawns each
	# rank in its OWN session (PGID == PID), so the group kills above cannot
	# reach them once torchrun itself exits — a rank wedged in NCCL teardown
	# can spin and hold its GPU memory indefinitely after the launcher exits. Debris = OUR user's model-update workers still holding
	# memory on THIS run's GPUs (exclusive per run, so a parallel arm's
	# workers on other GPUs are never touched).
	local _gpus="${MODEL_GPUS},${ROLLOUT_GPUS}"
	if [[ -n "${_gpus//,/}" ]]; then
		local _debris
		_debris=$(nvidia-smi --id="$_gpus" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u || true)
		for pid in $_debris; do
			if [[ "$(ps -o user= -p "$pid" 2>/dev/null)" == "$(id -un)" ]] \
				&& ps -o cmd= -p "$pid" 2>/dev/null | grep -q "whirlwind\.model\.model_"; then
				echo "  SIGKILL -> orphaned rank worker $pid (own pgid; NCCL debris sweep)"
				kill -KILL "$pid" 2>/dev/null || true
			fi
		done
	fi
	wait 2>/dev/null
	echo "=== Cleanup complete ==="
	exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------------------
# Helper: generate the launch command for each node
# ---------------------------------------------------------------------------
# Count GPUs for torchrun (needed before any model_print/start)
if [[ -n "$MODEL_GPUS" ]]; then
	IFS=',' read -ra MODEL_GPU_ARRAY <<<"$MODEL_GPUS"
	NPROC=${#MODEL_GPU_ARRAY[@]}
else
	NPROC=1
fi

# Shell functions that echo the command string for each node
cmd_orchestrator() {
	echo "PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='' ${PYTHON} -m \"${TRAINING_MODULE}\" --config \"${TRAINING_CONFIG}\" --log-level \"${LOG_LEVEL}\""
}

cmd_rollout() {
	echo "CUDA_VISIBLE_DEVICES=\"${ROLLOUT_GPUS}\" PYTHONUNBUFFERED=1 ${PYTHON} -m \"${ROLLOUT_MODULE}\" --config \"${ROLLOUT_CONFIG}\" --log-level \"${LOG_LEVEL}\""
}

cmd_reward() {
	echo "PYTHONUNBUFFERED=1 ${PYTHON} -m \"${REWARD_MODULE}\" --config \"${REWARD_CONFIG}\" --log-level \"${LOG_LEVEL}\""
}

cmd_dataloader() {
	echo "PYTHONUNBUFFERED=1 ${PYTHON} -m whirlwind.data.dataloader_node --config \"${DATA_LOADER_CONFIG}\" --log-level \"${LOG_LEVEL}\""
}

cmd_tracker() {
	echo "PYTHONUNBUFFERED=1 ${PYTHON} -m whirlwind.tracker.tracker_wandb --config \"${TRACKER_CONFIG}\" --log-level \"${LOG_LEVEL}\""
}

cmd_model_update() {
	echo "CUDA_VISIBLE_DEVICES=\"${MODEL_GPUS}\" PYTHONUNBUFFERED=1 ${TORCHRUN} --nproc_per_node=\"${NPROC}\" --master_port=\"${MASTER_PORT}\"${TORCHRUN_ARGS[*]:+ ${TORCHRUN_ARGS[*]}} -m \"${MODEL_MODULE}\" --config \"${MODEL_CONFIG}\" --log-level \"${LOG_LEVEL}\""
}

# Dispatch: given a node name, run the corresponding cmd_<name> function
get_cmd() {
	local node="$1"
	cmd_"$node"
}

# Dispatch: given a node name, start the node in background
start_node() {
	local node="$1"
	case "$node" in
	orchestrator)
		echo ">>> Starting orchestrator..."
		cd "$SCRIPT_DIR"
		setsid env PYTHONUNBUFFERED=1 \
			CUDA_VISIBLE_DEVICES="" \
			"$PYTHON" -m "${TRAINING_MODULE}" \
			--config "$TRAINING_CONFIG" \
			--log-level "$LOG_LEVEL" &
		echo "    Orchestrator PID: $! (process group $!)"
		PIDS+=($!)
		TRAINER_PID=$!
		ORCHESTRATOR_PID=$!
		;;
	rollout)
		echo ">>> Starting rollout node..."
		setsid env CUDA_VISIBLE_DEVICES="$ROLLOUT_GPUS" \
			PYTHONUNBUFFERED=1 \
			"$PYTHON" -m "$ROLLOUT_MODULE" \
			--config "$ROLLOUT_CONFIG" \
			--log-level "$LOG_LEVEL" &
		echo "    Rollout PID: $! (process group $!)"
		PIDS+=($!)
		;;
	reward)
		echo ">>> Starting reward node..."
		setsid env PYTHONUNBUFFERED=1 \
			"$PYTHON" -m "${REWARD_MODULE}" \
			--config "$REWARD_CONFIG" \
			--log-level "$LOG_LEVEL" &
		echo "    Reward PID: $! (process group $!)"
		PIDS+=($!)
		;;
	dataloader)
		echo ">>> Starting dataloader node..."
		setsid env PYTHONUNBUFFERED=1 \
			"$PYTHON" -m whirlwind.data.dataloader_node \
			--config "$DATA_LOADER_CONFIG" \
			--log-level "$LOG_LEVEL" &
		echo "    Dataloader PID: $! (process group $!)"
		PIDS+=($!)
		;;
	tracker)
		echo ">>> Starting progress tracker..."
		setsid env PYTHONUNBUFFERED=1 \
			"$PYTHON" -m whirlwind.tracker.tracker_wandb \
			--config "$TRACKER_CONFIG" \
			--log-level "$LOG_LEVEL" &
		echo "    Tracker PID: $! (process group $!)"
		PIDS+=($!)
		;;
	model_update)
		echo ">>> Starting model update node (torchrun, ${NPROC} GPUs)..."
		setsid env CUDA_VISIBLE_DEVICES="$MODEL_GPUS" \
			PYTHONUNBUFFERED=1 \
			"$TORCHRUN" \
			--nproc_per_node="$NPROC" \
			--master_port="$MASTER_PORT" \
			"${TORCHRUN_ARGS[@]}" \
			-m "${MODEL_MODULE}" \
			--config "$MODEL_CONFIG" \
			--log-level "$LOG_LEVEL" &
		echo "    Model update PID: $! (torchrun master, process group $!)"
		PIDS+=($!)
		TRAINER_PID=$!
		;;
	*)
		echo "ERROR: Unknown node: $node" >&2
		exit 1
		;;
	esac
}

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
if $DRY_RUN; then
	echo "=== Dry run ==="
	echo "  Rollout GPUs:   ${ROLLOUT_GPUS:-<default>}"
	echo "  Model GPUs:     ${MODEL_GPUS:-<default>}"
	echo "  Rollout config: $ROLLOUT_CONFIG"
	echo "  Rollout module: $ROLLOUT_MODULE"
	echo "  Reward config:  $REWARD_CONFIG"
	echo "  Model config:   $MODEL_CONFIG"
	echo "  Tracker config: $TRACKER_CONFIG"
	echo "  Dataloader config: $DATA_LOADER_CONFIG"
	echo "  Training config: $TRAINING_CONFIG"
	echo "  Python:         $PYTHON"
	echo "  Log level:      $LOG_LEVEL"
	echo "  Nodes to start: ${MANAGED[*]}"
	echo "  Remaining:      ${REMAINING[*]:-<none>}"
	echo ""
	echo "Validating configs..."
	"$PYTHON" "$SCRIPT_DIR/validate_configs.py" \
		"$ROLLOUT_CONFIG" "$REWARD_CONFIG" "$MODEL_CONFIG" "$TRACKER_CONFIG" "$TRAINING_CONFIG" \
		"$DATA_LOADER_CONFIG"
	exit 0
fi

# ---------------------------------------------------------------------------
# Validate GPU assignment (only when rollout or model_update is managed)
# ---------------------------------------------------------------------------
for node in "${MANAGED[@]}"; do
	if [[ "$node" == "rollout" && -z "$ROLLOUT_GPUS" ]]; then
		echo "ERROR: --rollout-gpus is required (rollout node is in --nodes)"
		echo "  Example: --rollout-gpus 0"
		exit 1
	fi
	if [[ "$node" == "model_update" && -z "$MODEL_GPUS" ]]; then
		echo "ERROR: --model-gpus is required (model_update node is in --nodes)"
		echo "  Example: --model-gpus 1,2"
		exit 1
	fi
done

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "=== Whirlwind RLHF Bringup ==="
echo "  Rollout GPUs:  ${ROLLOUT_GPUS:-<not managed>}"
echo "  Model GPUs:    ${MODEL_GPUS:-<not managed>} (${NPROC} GPU(s))"
echo "  Tracker:       $(grep -m1 '^backend:' "$TRACKER_CONFIG" 2>/dev/null | tr -d '"' || echo 'N/A')"
echo "  Log level:     $LOG_LEVEL"
echo "  Managed nodes: ${MANAGED[*]}"
if [[ ${#REMAINING[@]} -gt 0 ]]; then
	echo "  Remaining:     ${REMAINING[*]}"
fi
echo ""

# ---------------------------------------------------------------------------
# Start managed nodes
# ---------------------------------------------------------------------------
for node in "${MANAGED[@]}"; do
	start_node "$node"
done
echo ""

# ---------------------------------------------------------------------------
# Print commands for remaining nodes
# ---------------------------------------------------------------------------
if [[ ${#REMAINING[@]} -gt 0 ]]; then
	echo "=========================================="
	echo " Commands for remaining ${#REMAINING[@]} node(s) (run in separate terminal(s)):"
	echo "=========================================="
	echo ""
	for node in "${REMAINING[@]}"; do
		echo "# --- $node ---"
		get_cmd "$node"
		echo ""
	done
	echo "=========================================="
fi

# ---------------------------------------------------------------------------
# Wait
# ---------------------------------------------------------------------------
if [[ ${#REMAINING[@]} -gt 0 ]]; then
	echo ""
	echo "${#REMAINING[@]} node(s) not started by this script.  Copy the commands above"
	echo "and run them in separate terminal(s) before training will proceed."
	echo ""
	echo "Script will block waiting for the managed node(s) to finish."
	echo "Press Ctrl+C to abort."
	echo ""
else
	echo "=== All components started ==="
	echo "Waiting for training to complete..."
	echo ""
fi

# Wait for the run to finish.
#
# When the trainer (orchestrator or model_update/torchrun) is managed, wait
# trainer-driven: service nodes (rollout/reward/dataloader/tracker) are
# long-running and only exit on the trainer's cluster-shutdown broadcast
# (normal completion), the nodes' own heartbeat fuse (crashed trainer), or
# our cleanup signal. If the trainer has exited but service nodes linger
# past WW_BRINGUP_GRACE_S (default 120s > the ~80s heartbeat fuse), stop
# waiting and fall through to cleanup instead of blocking forever.
#
# Without a managed trainer, keep the plain per-PID wait (partial-node
# setups where this script manages e.g. only the rollout node).
RUN_RC=0
if [[ ${#PIDS[@]} -gt 0 ]]; then
	if [[ -n "$TRAINER_PID" ]]; then
		GRACE="${WW_BRINGUP_GRACE_S:-120}"
		trainer_done_at=""
		orch_done_at=""
		while :; do
			alive=0
			for pid in "${PIDS[@]}"; do
				kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
			done
			(( alive == 0 )) && break
			if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
				if [[ -z "$trainer_done_at" ]]; then
					trainer_done_at=$SECONDS
					echo "Trainer (pid $TRAINER_PID) exited; waiting up to ${GRACE}s for service nodes to stop..."
				fi
				if (( SECONDS - trainer_done_at > GRACE )); then
					echo "Service nodes still alive ${GRACE}s after trainer exit - proceeding to cleanup."
					break
				fi
			elif [[ -n "$ORCHESTRATOR_PID" ]] && ! kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
				# Orchestrator is gone (normal completion OR crash) but the trainer
				# (torchrun) is still alive: the known rank-teardown wedge — a rank
				# spinning in NCCL teardown after a successful run never exits, and
				# torchrun never reaps. Bound it: past GRACE seconds after the
				# orchestrator exit, fall through to cleanup (TERM/KILL pass) instead
				# of waiting forever. In the crashed-orchestrator scenario the model
				# node's own heartbeat fuse (~80s) exits torchrun well inside this
				# window, so the wedge bound never cuts a live run short.
				if [[ -z "$orch_done_at" ]]; then
					orch_done_at=$SECONDS
					echo "Orchestrator exited but trainer (pid $TRAINER_PID) still alive; waiting up to ${GRACE}s (rank-teardown wedge bound)..."
				fi
				if (( SECONDS - orch_done_at > GRACE )); then
					echo "Trainer still alive ${GRACE}s after orchestrator exit (rank-teardown wedge) - proceeding to cleanup."
					break
				fi
			fi
			sleep 2
		done
		# NOTE: no per-PID `wait` here — bash auto-reaps async children, and a
		# blocking wait on a still-live PID (wedged rollout/tracker) would hang
		# bringup BEFORE the EXIT trap fires. cleanup() reaps after its KILL
		# pass, when everything is guaranteed dead.
		# Exit status: the training loop's own exit code when this script
		# started it (non-zero if it crashed or stopped early because a
		# required node was lost); otherwise the trainer's.
		_status_pid="${ORCHESTRATOR_PID:-$TRAINER_PID}"
		if kill -0 "$_status_pid" 2>/dev/null; then
			echo "Training process (pid $_status_pid) still running at teardown - reporting failure."
			RUN_RC=1
		else
			wait "$_status_pid" 2>/dev/null || RUN_RC=$?
		fi
	else
		for pid in "${PIDS[@]}"; do
			wait "$pid" || { _rc=$?; (( RUN_RC == 0 )) && RUN_RC=$_rc; }
		done
	fi
fi

echo ""
echo "=== Done (exit status $RUN_RC) ==="
exit "$RUN_RC"
