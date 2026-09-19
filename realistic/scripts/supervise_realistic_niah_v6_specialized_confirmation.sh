#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <index|bullet> <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi
MODE=$1
MODEL=$2
GPU_INDEX=$3
case "$MODE" in
  index) PROMPT_MODE=enumeration_index; TIMING=rank_before_city; ANCHOR_ROLE=post_marker ;;
  bullet) PROMPT_MODE=enumeration_bullet; TIMING=structural_item_end; ANCHOR_ROLE=p0_item_end ;;
  *) echo "mode must be index or bullet" >&2; exit 2 ;;
esac
case "$MODEL" in
  Qwen3-8B) SOURCE_LAYER=19 ;;
  Gemma4-E4B) SOURCE_LAYER=16 ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac
if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  echo "gpu-index must be non-negative" >&2
  exit 2
fi

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
MODEL_ROOT=$RUN_BASE/$PROMPT_MODE/$MODEL
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
MECHANISM=$MODEL_ROOT/freeze/mechanism_frozen_confirmation.json
FREEZE=$MODEL_ROOT/freeze/confirmation_freeze.json
GENERATIONS=$MODEL_ROOT/generation/generations.jsonl
COHORT_REGISTRY=$MODEL_ROOT/replacement/confirmation/selected_cells.jsonl
DISCOVERY_REGISTRY=$MODEL_ROOT/replacement/discovery/selected_cells.jsonl
TARGET_DISCOVERY=$MODEL_ROOT/causal/targeted_retrieval/discovery_formal
TARGET_CONFIRMATION=$MODEL_ROOT/causal/targeted_retrieval/confirmation_formal
SELECTION=$TARGET_DISCOVERY/analysis/selection.json
PANEL_ROOT=$TARGET_CONFIRMATION/final_transition_panel
ANCHORS=$PANEL_ROOT/mode_panel.jsonl
TARGETED=$PANEL_ROOT/targeted_registry.jsonl
OUTPUT_ROOT=$MODEL_ROOT/causal/specialized/confirmation_formal
DISCOVERY_SPECIALIZED=$MODEL_ROOT/causal/specialized/discovery_formal
ANALYSIS_ROOT=$MODEL_ROOT/causal/specialized/confirmation_analysis
LOG_ROOT=$OUTPUT_ROOT/logs

for path in "$PYTHON" "$CONFIG" "$MECHANISM" "$FREEZE" "$GENERATIONS" \
  "$COHORT_REGISTRY" "$DISCOVERY_REGISTRY" "$SELECTION" "$ANCHORS" \
  "$TARGETED" "$DISCOVERY_SPECIALIZED/discovery.COMPLETE"; do
  [[ -s "$path" ]] || { echo "missing specialized confirmation input: $path" >&2; exit 4; }
done
SELECTED_K=$($PYTHON -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_k"]))' \
  "$SELECTION")
BANK_PLAN=$DISCOVERY_SPECIALIZED/bank_plan/retrieval_anchor_bank_plan.csv
SPECIALIZED_BANK_AUDIT=$DISCOVERY_SPECIALIZED/bank_plan/specialized_bank_plan_audit.json
for path in "$BANK_PLAN" "$SPECIALIZED_BANK_AUDIT" \
  "$DISCOVERY_SPECIALIZED/bank_plan/specialized_bank_plan.COMPLETE"; do
  [[ -s "$path" ]] || { echo "missing frozen specialized bank plan: $path" >&2; exit 4; }
done
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/locks"
cd "$ROOT"
exec 9>"$OUTPUT_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another $MODE/$MODEL specialized confirmation owns the lock" >&2
  exit 75
fi

run_logged() {
  local name=$1
  shift
  {
    echo "[$(date --iso-8601=seconds)] START $name"
    printf 'COMMAND env CUDA_VISIBLE_DEVICES=%q' "$GPU_INDEX"
    printf ' %q' "$@"
    printf '\n'
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 "$@"
    echo "[$(date --iso-8601=seconds)] PASS $name"
  } 2>&1 | tee "$LOG_ROOT/$name.log"
}

kernel() {
  local target=$1
  shift
  run_logged "$target" "$PYTHON" scripts/run_realistic_niah_v6_kernel.py \
    --target "$target" --v6-config "$CONFIG" --phase confirmation \
    --confirmation-freeze "$FREEZE" --cohort-registry "$COHORT_REGISTRY" \
    --bank-selection "$SELECTION" -- "$@"
}

common=(
  --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE"
  --device-map auto --torch-dtype bfloat16 --attention-backend sdpa
  --generations "$GENERATIONS" --seed-role confirmation
)

kernel targeted-counter-write "${common[@]}" \
  --anchor-registry "$ANCHORS" --targeted-registry "$TARGETED" \
  --bank-plan "$BANK_PLAN" --source-layer "$SOURCE_LAYER" --resume \
  --output "$OUTPUT_ROOT/targeted_counter_write"

kernel stratified-targeted-counter-ncc "${common[@]}" \
  --timing "$TIMING" --panel "$ANCHORS" --bank-plan "$BANK_PLAN" --resume \
  --output "$OUTPUT_ROOT/stratified_ncc"

kernel targeted-counter-logit-margin "${common[@]}" \
  --timing "$TIMING" --panel "$ANCHORS" --bank-plan "$BANK_PLAN" --resume \
  --output "$OUTPUT_ROOT/direct_count_logit_margin"

kernel targeted-counter-ncc "${common[@]}" \
  --anchor-registry "$ANCHORS" --targeted-registry "$TARGETED" \
  --bank-plan "$BANK_PLAN" --source-layer "$SOURCE_LAYER" --resume \
  --output "$OUTPUT_ROOT/count_geometry_ncc"

run_logged terminal-token-state-bridge \
  "$PYTHON" scripts/run_realistic_niah_v6_kernel.py \
  --target terminal-token-state-bridge --v6-config "$CONFIG" \
  --phase confirmation --confirmation-freeze "$FREEZE" \
  --cohort-registry "$COHORT_REGISTRY" -- \
  "${common[@]}" --anchor-registry "$ANCHORS" --layer "$SOURCE_LAYER" \
  --max-new-tokens 32 --resume --output "$OUTPUT_ROOT/terminal_state_bridge"

kernel token-level-ablation \
  --mode answer --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
  --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --output "$OUTPUT_ROOT/token_ablation_answer" \
  --bank-plan "$BANK_PLAN" --bank-size "$SELECTED_K" \
  --anchor-registry "$TARGETED" --split confirmation \
  --conditions clean prompt_all_blank prompt_records_blank trace_all_blank \
    prompt_and_trace_blank --run-greedy --max-new-tokens 32

kernel token-level-ablation \
  --mode targeting --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
  --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --output "$OUTPUT_ROOT/token_ablation_targeting" \
  --bank-plan "$BANK_PLAN" --bank-size "$SELECTED_K" \
  --anchor-role "$ANCHOR_ROLE" --anchor-registry "$TARGETED" \
  --split confirmation --matched-control-repeats 3 \
  --conditions clean early_half_trace_blank cumulative_trace_blank \
    recent_transition_blank full_trace_blank early_half_trace_matched_control \
    cumulative_trace_matched_control recent_transition_matched_control \
    full_trace_matched_control --run-greedy --max-new-tokens 32

run_logged specialized-confirmation-analysis \
  "$PYTHON" scripts/analyze_realistic_niah_v6_specialized_confirmation.py \
  --v6-config "$CONFIG" --confirmation-freeze "$FREEZE" \
  --model "$MODEL" --timing "$TIMING" \
  --discovery-root "$DISCOVERY_SPECIALIZED" \
  --confirmation-root "$OUTPUT_ROOT" \
  --discovery-registry "$DISCOVERY_REGISTRY" \
  --confirmation-registry "$COHORT_REGISTRY" \
  --output "$ANALYSIS_ROOT"

"$PYTHON" - "$OUTPUT_ROOT" "$MODEL" "$PROMPT_MODE" "$FREEZE" \
  "$COHORT_REGISTRY" "$ANALYSIS_ROOT" "$BANK_PLAN" \
  "$SPECIALIZED_BANK_AUDIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model, prompt_mode = sys.argv[2:4]
freeze, cohort, analysis_root, bank, bank_audit = map(pathlib.Path, sys.argv[4:9])
required = {
    "targeted_counter_write": root / "targeted_counter_write/manifest.json",
    "stratified_ncc": root / "stratified_ncc/manifest.json",
    "direct_count_logit_margin": root / "direct_count_logit_margin/manifest.json",
    "count_geometry_ncc": root / "count_geometry_ncc/manifest.json",
    "terminal_state_bridge": root / "terminal_state_bridge/manifest.json",
    "token_ablation_answer": root / "token_ablation_answer/worker_00_manifest.json",
    "token_ablation_targeting": root / "token_ablation_targeting/worker_00_manifest.json",
}
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
analysis_manifest = analysis_root / "analysis_manifest.json"
analysis_marker = analysis_root / "confirmation_analysis.COMPLETE"
if not analysis_manifest.is_file() or analysis_marker.read_text().strip() != "PASS":
    raise FileNotFoundError("specialized confirmation analysis did not seal")
adapter_paths = [
    root / name / "v6_adapter_manifest.json"
    for name in (
        "targeted_counter_write", "stratified_ncc", "direct_count_logit_margin",
        "count_geometry_ncc", "terminal_state_bridge", "token_ablation_answer",
        "token_ablation_targeting",
    )
]
for path in adapter_paths:
    value = json.loads(path.read_text())
    identity = value.get("specialized_slot_identity", {})
    if identity.get("status") != "PASS_FIXED_SLOT_TRUE_SOURCE_IDENTITY":
        raise ValueError(f"specialized slot/source audit failed: {path}")
    if int(identity.get("analysis_slot_count", -1)) != 10:
        raise ValueError(f"specialized confirmation slot count changed: {path}")
payload = {
    "schema_version": "realistic_niah_v6_specialized_confirmation_complete_v2",
    "status": "CONFIRMATION_COMPLETE",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "confirmation_used_for_selection": False,
    "panel_membership_identity": "analysis_slot_seed",
    "statistical_identity": "true_source_seed",
    "seed_aliasing": False,
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "cohort_registry_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
    "bank_plan": {
        "path": str(bank.resolve()),
        "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
    },
    "specialized_bank_plan_adapter": {
        "path": str(bank_audit.resolve()),
        "sha256": hashlib.sha256(bank_audit.read_bytes()).hexdigest(),
    },
    "artifacts": {
        "specialized_bank_plan": {
            "path": str(bank.resolve()),
            "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        },
        "specialized_bank_plan_adapter": {
            "path": str(bank_audit.resolve()),
            "sha256": hashlib.sha256(bank_audit.read_bytes()).hexdigest(),
        },
    },
    "outputs": {
        name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in required.items()
    },
    "adapter_manifests": {
        path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in adapter_paths
    },
    "analysis_manifest": {
        "path": str(analysis_manifest.resolve()),
        "sha256": hashlib.sha256(analysis_manifest.read_bytes()).hexdigest(),
    },
}
path = root / "specialized_confirmation_complete.json"
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
print(json.dumps(payload, sort_keys=True))
PY

printf 'PASS\n' >"$OUTPUT_ROOT/confirmation.COMPLETE"
