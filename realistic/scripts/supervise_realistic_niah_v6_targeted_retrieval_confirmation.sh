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
  index) PROMPT_MODE=enumeration_index; ANCHOR_ROLE=post_marker ;;
  bullet) PROMPT_MODE=enumeration_bullet; ANCHOR_ROLE=p0_item_end ;;
  *) echo "mode must be index or bullet" >&2; exit 2 ;;
esac
case "$MODEL" in
  Qwen3-8B|Gemma4-E4B) ;;
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
REPORT_CONTRACT=$ROOT/configs/realistic_niah_v6_targeted_retrieval_report_contract.json
GENERATIONS=$MODEL_ROOT/generation/generations.jsonl
DISCOVERY_REGISTRY=$MODEL_ROOT/replacement/discovery/selected_cells.jsonl
CONFIRMATION_REGISTRY=$MODEL_ROOT/replacement/confirmation/selected_cells.jsonl
FREEZE=$MODEL_ROOT/freeze/confirmation_freeze.json
DISCOVERY_ROOT=$MODEL_ROOT/causal/targeted_retrieval/discovery_formal
SELECTION=$DISCOVERY_ROOT/analysis/selection.json
OUTPUT_ROOT=$MODEL_ROOT/causal/targeted_retrieval/confirmation_formal
SOURCE_ROOT=$OUTPUT_ROOT/source_writes/$ANCHOR_ROLE
PANEL_ROOT=$OUTPUT_ROOT/final_transition_panel
BEHAVIOR_REGISTRY=$PANEL_ROOT/behavior_anchor_registry.jsonl
LOG_ROOT=$OUTPUT_ROOT/logs

for path in "$PYTHON" "$CONFIG" "$REPORT_CONTRACT" "$GENERATIONS" "$DISCOVERY_REGISTRY" \
  "$CONFIRMATION_REGISTRY" "$FREEZE" "$SELECTION"; do
  [[ -s "$path" ]] || { echo "missing targeted confirmation input: $path" >&2; exit 4; }
done
SELECTED_K=$($PYTHON -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_k"]))' \
  "$SELECTION")
RANDOM_CONDITION=$($PYTHON -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["selected_random_condition"]))' \
  "$SELECTION")
case "$RANDOM_CONDITION" in
  layer_matched_random|global_random) ;;
  *) echo "invalid frozen random condition: $RANDOM_CONDITION" >&2; exit 4 ;;
esac
PLAN=$DISCOVERY_ROOT/plans/k$SELECTED_K/retrieval_anchor_bank_plan.csv
[[ -s "$PLAN" ]] || { echo "missing discovery-frozen plan: $PLAN" >&2; exit 4; }
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/locks"
cd "$ROOT"
exec 9>"$OUTPUT_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another $MODE/$MODEL targeted confirmation owns the lock" >&2
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

causal() {
  run_logged "$1" "$PYTHON" scripts/run_realistic_niah_v6_causal.py \
    --v6-config "$CONFIG" --model-label "$MODEL" --phase confirmation \
    --confirmation-freeze "$FREEZE" \
    --cohort-registry "$CONFIRMATION_REGISTRY" \
    --causal-membership-registry "$DISCOVERY_REGISTRY" -- "${@:2}"
}

causal source_writes causal-source-writes \
  --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
  --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --output "$SOURCE_ROOT" \
  --anchor-role "$ANCHOR_ROLE" --include-secondary

run_logged freeze_final_transition_panel \
  "$PYTHON" scripts/build_realistic_niah_v6_final_transition_panel.py \
  --v6-config "$CONFIG" --model "$MODEL" --generations "$GENERATIONS" \
  --cohort-registry "$CONFIRMATION_REGISTRY" --source-writes "$SOURCE_ROOT" \
  --seed-role confirmation --output "$PANEL_ROOT"

causal frozen_k_behavior causal-heads-behavior \
  --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
  --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --plan "$PLAN" \
  --output "$OUTPUT_ROOT/behavior/k$SELECTED_K" \
  --anchor-role "$ANCHOR_ROLE" --include-secondary \
  --anchor-registry-input "$BEHAVIOR_REGISTRY" \
  --allow-selection-scope-bank-transfer --evaluation-split confirmation \
  --counts 1 2 3 4 5 6 7 8 9 10 --conditions clean selected_bank \
    "$RANDOM_CONDITION" --limit 10 --anchor-sampling prompt_final_transition \
  --max-new-tokens 256 --decode-head-ablation-steps -1

run_logged analyze_frozen_k_confirmation \
  "$PYTHON" scripts/analyze_realistic_niah_v6_targeted_retrieval_confirmation.py \
  --model "$MODEL" --prompt-mode "$PROMPT_MODE" \
  --behavior "$OUTPUT_ROOT/behavior/k$SELECTED_K" --selection "$SELECTION" \
  --report-contract "$REPORT_CONTRACT" \
  --confirmation-freeze "$FREEZE" --expected-seeds 10 \
  --bootstrap-samples 10000 --random-seed 20260828 \
  --output "$OUTPUT_ROOT/analysis.json"

"$PYTHON" - "$OUTPUT_ROOT" "$MODEL" "$PROMPT_MODE" "$SELECTED_K" \
  "$FREEZE" "$CONFIRMATION_REGISTRY" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model, prompt_mode = sys.argv[2:4]
selected_k = int(sys.argv[4])
freeze, registry = map(pathlib.Path, sys.argv[5:7])
required = {
    "source": root / ("source_writes/post_marker/manifest.json" if prompt_mode == "enumeration_index" else "source_writes/p0_item_end/manifest.json"),
    "panel": root / "final_transition_panel/manifest.json",
    "behavior": root / f"behavior/k{selected_k}/manifest.json",
    "behavior_adapter": root / f"behavior/k{selected_k}/v6_adapter_manifest.json",
    "analysis": root / "analysis.json",
}
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
analysis = json.loads(required["analysis"].read_text())
if analysis["status"] != "CONFIRMATION_EVALUATED_FROZEN_K":
    raise ValueError("targeted confirmation analysis is incomplete")
if int(analysis["selected_k"]) != selected_k or analysis["bank_size_reselected"]:
    raise ValueError("targeted confirmation changed discovery K")
payload = {
    "schema_version": "realistic_niah_v6_targeted_retrieval_confirmation_complete_v1",
    "status": "CONFIRMATION_COMPLETE",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "selected_k": selected_k,
    "confirmation_used_for_selection": False,
    "statistical_identity": "true_source_seed",
    "seed_aliasing": False,
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "cohort_registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    "outputs": {
        name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in required.items()
    },
}
path = root / "confirmation_complete.json"
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
print(json.dumps(payload, sort_keys=True))
PY

printf 'PASS\n' >"$OUTPUT_ROOT/confirmation.COMPLETE"
