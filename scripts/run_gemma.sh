#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
python "$ROOT/code/experiments/main/run_matrix.py" \
  --model google/gemma-4-31B-it --effort off --backend vllm \
  --base-url "$BASE_URL" --vllm-max-output-tokens 2048 \
  --sample-size 100 --workers "${WORKERS:-2}" --parallel-jobs "${PARALLEL_JOBS:-1}" \
  --execute
