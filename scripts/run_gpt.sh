#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the environment}"
python "$ROOT/code/experiments/main/run_matrix.py" \
  --model gpt-5.4-nano --effort low --backend openai \
  --sample-size 100 --workers "${WORKERS:-30}" --parallel-jobs "${PARALLEL_JOBS:-1}" \
  --execute
