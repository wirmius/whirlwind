#!/usr/bin/env bash
# The mainline recipe: Qwen3-0.6B-Base on GSM8K with dr_grpo, the SGLang
# rollout engine and the FSDP2 trainer colocated on the same GPUs via
# sleep/wake.
#
#   GPUS=0,1 bash examples/gsm8k_dr_grpo_colocate.sh
#   GPUS=0   bash examples/gsm8k_dr_grpo_colocate.sh
#
# GPUs: GPUS=0 runs everything on one GPU, GPUS=0,1 colocates on two (the
# configs used for the reference numbers). For other GPU counts, point
# ROLLOUT_CONFIG and MODEL_CONFIG at configs sized for them (rollout
# tp_size or dp_size, model fsdp_size). Extra arguments are passed to
# bringup_and_run.sh, and later flags win, e.g. --training-config my.yaml.
#
# Run from any directory inside an environment with the dependencies
# installed (see README); WHIRLWIND_PYTHON selects the interpreter. Logs to
# runs/gsm8k_dr_grpo_colocate/driver.log. Set WANDB_MODE=offline to train without a W&B account.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPUS="${GPUS:-0,1}"
IFS=',' read -ra _gpu_list <<<"$GPUS"
case ${#_gpu_list[@]} in
1)
	_rollout=configs/rollout/sglang_colocate_g1.yaml
	_model=configs/model/fsdp2_base_colocate_gas1_fp32master_chunked_dr_grpo_world1.yaml
	;;
2)
	_rollout=configs/rollout/sglang_colocate_tp2.yaml
	_model=configs/model/fsdp2_base_colocate_gas1_fp32master_chunked_dr_grpo.yaml
	;;
*)
	_rollout=""
	_model=""
	;;
esac
ROLLOUT_CONFIG="${ROLLOUT_CONFIG:-$_rollout}"
MODEL_CONFIG="${MODEL_CONFIG:-$_model}"
if [[ -z "$ROLLOUT_CONFIG" || -z "$MODEL_CONFIG" ]]; then
	echo "GPUS=$GPUS: configs are shipped for 1 or 2 GPUs; set ROLLOUT_CONFIG and MODEL_CONFIG for ${#_gpu_list[@]}." >&2
	exit 2
fi

export WW_RUN_ID="${WW_RUN_ID:-gsm8k-dr-grpo-$$}"
export NODE_COMMS_ENSURE_REGISTER=1
export WHIRLWIND_RPC_TIMEOUT="${WHIRLWIND_RPC_TIMEOUT:-600}"

LOG_DIR="${LOG_DIR:-runs/gsm8k_dr_grpo_colocate}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1
echo "=== gsm8k dr_grpo colocate on GPUs $GPUS — $(date) ==="

bash bringup_and_run.sh \
	--rollout-gpus "$GPUS" --model-gpus "$GPUS" \
	--rollout-module whirlwind.rollout.rollout_sglang \
	--rollout-config "$ROLLOUT_CONFIG" \
	--model-config "$MODEL_CONFIG" \
	--training-config configs/training/gsm8k_dr_grpo_colocate.yaml \
	--tracker-config configs/tracker/wandb.yaml \
	--dataloader-config configs/dataloader/gsm8k.yaml \
	--log-level INFO \
	"$@"
rc=$?
echo "=== done rc=$rc — $(date) ==="
exit $rc
