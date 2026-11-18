#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the environment}"
MODEL_SLUG="${MODEL_SLUG:-gpt_5_4_nano}"
RUN_ID="seed42_n100_s10_k3_r2"
python "$ROOT/code/experiments/transcript_audit/run_audit_matrix.py" \
  --shared-root "$ROOT/data/results/main_v2/shared/$RUN_ID/$MODEL_SLUG" \
  --gmemory-root "$ROOT/data/results/main_v2/gmemory/${RUN_ID}_t1_i3_embed/$MODEL_SLUG" \
  --output-root "$ROOT/data/results/transcript_audit_v2/$RUN_ID/$MODEL_SLUG" \
  --model "${AUDITOR_MODEL:-gpt-5.4-mini}" --effort "${AUDITOR_EFFORT:-high}" \
  --workers "${WORKERS:-30}" --execute
