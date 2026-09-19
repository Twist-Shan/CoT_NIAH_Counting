#!/usr/bin/env bash
set -euo pipefail

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
QUEUE_ROOT=$RUN_BASE/queue_logs
OUTPUT_ROOT=$RUN_BASE/final_audit
REPRESENTATION_ROOT=$RUN_BASE/native_aligned_representation
CPU_THREADS=${V6_CPU_THREADS:-8}
mkdir -p "$QUEUE_ROOT" "$OUTPUT_ROOT"
exec > >(tee -a "$QUEUE_ROOT/final_audit.log") 2>&1

wait_for_pass() {
  local marker=$1
  local label=$2
  echo "[$(date --iso-8601=seconds)] WAIT $label marker=$marker"
  while [[ ! -s "$marker" ]]; do sleep 30; done
  grep -qx PASS "$marker" || { echo "$label is not PASS" >&2; exit 1; }
  echo "[$(date --iso-8601=seconds)] READY $label"
}

wait_for_pass "$QUEUE_ROOT/Qwen3-8B_confirmation.COMPLETE" \
  "Qwen3-8B full index+bullet confirmation"
wait_for_pass "$QUEUE_ROOT/Gemma4-E4B_confirmation.COMPLETE" \
  "Gemma4-E4B full index+bullet confirmation"

cd "$ROOT"
echo "[$(date --iso-8601=seconds)] START four-cell Native-aligned representation threads=$CPU_THREADS"
env OMP_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
  MKL_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" \
  "$PYTHON" scripts/analyze_realistic_niah_v6_native_aligned_representation.py \
    --run-root "$RUN_BASE" --output "$REPRESENTATION_ROOT" \
    --alignment-contract \
      "$ROOT/configs/realistic_niah_v6_native_analysis_alignment_v1.json"
grep -qx PASS "$REPRESENTATION_ROOT/COMPLETE"
echo "[$(date --iso-8601=seconds)] PASS four-cell Native-aligned representation"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" \
  "$PYTHON" scripts/audit_realistic_niah_v6_suite_completion.py \
    --run-root "$RUN_BASE" --output "$OUTPUT_ROOT"
grep -qx PASS "$OUTPUT_ROOT/suite_completion.COMPLETE"
printf 'PASS\n' >"$QUEUE_ROOT/final_audit.COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE V6 four-cell final audit"
