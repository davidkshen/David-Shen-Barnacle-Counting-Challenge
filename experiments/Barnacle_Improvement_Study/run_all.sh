#!/usr/bin/env bash
# Rebuild every result in results/ from the saved checkpoints (CPU, no PyTorch needed).
# Expects Barnacle_Three_Track_Study/ and Barnacle_Assistant/ next to this folder,
# or set BARNACLE_THREE_TRACK / BARNACLE_ASSISTANT.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
PY=${PYTHON:-python3}
for step in validate_np diagnose test_time_fixes adapt_budget adapt_diagnostics review_sim audit_sim combined_workflow_sim unseen_counts; do
  echo "== $step"
  "$PY" -W ignore experiments/$step.py
done
"$PY" -W ignore make_figures.py
echo "All results rebuilt."
