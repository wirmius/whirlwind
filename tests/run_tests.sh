#!/usr/bin/env bash
# Run the whirlwind test suite.
#
# The suite needs a torch+transformers+zmq+msgpack environment (CUDA for the
# GPU tests, which skip otherwise). The venv's own lib/ is put first on
# LD_LIBRARY_PATH: a conda-built pyzmq can need a newer libstdc++ (e.g.
# GLIBCXX_3.4.30) than an older system /lib64 provides.
#
# Usage:
#   ./tests/run_tests.sh                   # run everything
#   ./tests/run_tests.sh tests.test_model_update
#   ./tests/run_tests.sh -k TestReferenceWeightUpdate
#   CUDA_VISIBLE_DEVICES=0 ./tests/run_tests.sh
#
# Environment overrides:
#   WHIRLWIND_TEST_PYTHON   interpreter to use (default: python3 on PATH)
#   WHIRLWIND_TEST_LD_LIB   extra LD_LIBRARY_PATH entries
#   HF_HOME                 HuggingFace cache (default: the HF default, ~/.cache/huggingface)
#   WHIRLWIND_TEST_MODEL / WHIRLWIND_TEST_MODEL_DIR  small model override
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PY="${WHIRLWIND_TEST_PYTHON:-$(command -v python3)}"
if [ ! -x "$PY" ]; then
	echo "ERROR: python interpreter not found: $PY" >&2
	echo "Set WHIRLWIND_TEST_PYTHON to a venv with torch+CUDA+transformers+zmq+msgpack." >&2
	exit 1
fi

# Prefer the venv's bundled libraries (libstdc++) over the system ones.
VENV_LIB="$(dirname "$(dirname "$PY")")/lib"
export LD_LIBRARY_PATH="${VENV_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"  # empty = CPU only; unset = GPU 0

if [ "$#" -gt 0 ]; then
	# If the first arg looks like a pytest flag, use pytest; else unittest path.
	case "$1" in
	-*) "$PY" -m pytest "$@" ;;
	*) "$PY" -m unittest "$@" -v ;;
	esac
else
	# Run all test modules. unittest first (no pytest dependency required).
	"$PY" -m unittest discover -s tests -p "test_*.py" -v
fi
