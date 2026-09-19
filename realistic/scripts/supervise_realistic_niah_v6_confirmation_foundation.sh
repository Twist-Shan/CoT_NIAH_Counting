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
  index|bullet) PROMPT_MODE=enumeration_$MODE ;;
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
STIMULI=${V6_STIMULI:-$ROOT/work/nonthinking_report_filestream_stage3/stimuli.jsonl}
REPLACEMENT_POOL=${V6_REPLACEMENT_POOL:-$RUN_BASE/replacement_seed_pool}
REPLACEMENT_POLICY=${V6_REPLACEMENT_POLICY:-$ROOT/configs/realistic_niah_v6_replacement_policy.json}
RESUME_FROM=${V6_CONFIRMATION_FOUNDATION_RESUME_FROM:-start}
MODEL_ROOT=$RUN_BASE/$PROMPT_MODE/$MODEL
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
MECHANISM=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}_count_stream_dev.json
FREEZE_ROOT=$MODEL_ROOT/freeze
FREEZE=$FREEZE_ROOT/confirmation_freeze.json
BASE_REGISTRY=$MODEL_ROOT/replacement/confirmation/selected_cells.jsonl
COHERENT_OUTPUT=$MODEL_ROOT/replacement/confirmation_broad
LOG_ROOT=$MODEL_ROOT/logs/confirmation_foundation

case "$RESUME_FROM" in
  start|coherent-broad) ;;
  *) echo "V6_CONFIRMATION_FOUNDATION_RESUME_FROM must be start or coherent-broad" >&2; exit 2 ;;
esac

for path in "$PYTHON" "$CONFIG" "$MECHANISM" "$STIMULI" \
  "$REPLACEMENT_POLICY" "$REPLACEMENT_POOL/stimuli.jsonl" "$FREEZE"; do
  [[ -s "$path" ]] || { echo "missing confirmation input: $path" >&2; exit 4; }
done
if [[ "$RESUME_FROM" == coherent-broad ]]; then
  [[ -s "$BASE_REGISTRY" ]] || {
    echo "missing resolved confirmation registry for coherent-broad resume: $BASE_REGISTRY" >&2
    exit 4
  }
  echo "[$(date --iso-8601=seconds)] RESUME confirmation foundation from coherent-broad"
fi
mkdir -p "$LOG_ROOT" "$FREEZE_ROOT"
cd "$ROOT"
exec 9>"$FREEZE_ROOT/confirmation_foundation.lock"
if ! flock -n 9; then
  echo "another $MODE/$MODEL confirmation foundation owns the lock" >&2
  exit 75
fi

"$PYTHON" scripts/run_realistic_niah_v6.py validate-confirmation-freeze \
  --config "$CONFIG" --model "$MODEL" --confirmation-freeze "$FREEZE" \
  --output "$FREEZE_ROOT/validation_before_confirmation_foundation.json"

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

foundation_phase() {
  local phase=$1
  run_logged "$phase" env \
    V6_ROOT="$ROOT" V6_PYTHON="$PYTHON" V6_CACHE="$CACHE" \
    V6_STIMULI="$STIMULI" V6_RUN_ROOT="$MODEL_ROOT" \
    V6_REPLACEMENT_POOL="$REPLACEMENT_POOL" \
    V6_REPLACEMENT_POLICY="$REPLACEMENT_POLICY" \
    V6_CONFIRMATION_FREEZE="$FREEZE" \
    bash "$ROOT/scripts/supervise_realistic_niah_v6_enumeration.sh" \
      "$MODE" "$MODEL" "$phase"
}

if [[ "$RESUME_FROM" == start ]]; then
  foundation_phase confirmation-generate
  foundation_phase confirmation-supplement
fi

run_logged coherent_broad_confirmation \
  "$PYTHON" scripts/run_realistic_niah_v6_broad_panel_replacement.py \
  --v6-config "$CONFIG" --mechanism-config "$MECHANISM" \
  --replacement-policy "$REPLACEMENT_POLICY" \
  --coherent-broad-policy \
    "$ROOT/configs/realistic_niah_v6_coherent_broad_replacement_policy.json" \
  --replacement-stimuli "$REPLACEMENT_POOL/stimuli.jsonl" \
  --base-cohort-registry "$BASE_REGISTRY" --model "$MODEL" \
  --phase confirmation --generation-root "$MODEL_ROOT/generation" \
  --output "$COHERENT_OUTPUT" --cache-dir "$CACHE" --device-map auto \
  --torch-dtype bfloat16 --attention-backend sdpa \
  --confirmation-freeze "$FREEZE"

foundation_phase confirmation-foundation-resolved

"$PYTHON" - "$MODEL_ROOT" "$MODEL" "$PROMPT_MODE" "$FREEZE" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model, prompt_mode = sys.argv[2:4]
freeze = pathlib.Path(sys.argv[4])
required = {
    "generation_manifest": root / "generation/manifest_confirmation.json",
    "cell_registry": root / "replacement/confirmation/selected_cells.jsonl",
    "cell_mapping": root / "replacement/confirmation/replacement_mapping.jsonl",
    "cell_manifest": root / "replacement/confirmation/manifest.json",
    "broad_registry": root / "replacement/confirmation_broad/selected_cells.jsonl",
    "broad_mapping": root / "replacement/confirmation_broad/coherent_mapping.jsonl",
    "broad_manifest": root / "replacement/confirmation_broad/manifest.json",
    "formal_capture": root / "capture/confirmation_formal/v6_adapter_manifest.json",
    "all_capture": root / "capture/confirmation_all_sample/v6_adapter_manifest.json",
    "confirmation_attention": root / "attention/confirmation_formal.manifest.json",
    "confirmation_answer_query": root / "attention/confirmation_answer_query_formal.manifest.json",
}
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
cell_rows = [json.loads(line) for line in required["cell_registry"].read_text().splitlines() if line.strip()]
broad_rows = [json.loads(line) for line in required["broad_registry"].read_text().splitlines() if line.strip()]
if len(cell_rows) != 100 or len(broad_rows) != 100:
    raise ValueError(f"confirmation registry sizes changed: cell={len(cell_rows)}, broad={len(broad_rows)}")
expected_slots = {(count, seed) for count in range(1, 11) for seed in range(1254, 1264)}
for name, rows in (("cell", cell_rows), ("broad", broad_rows)):
    observed = {(int(row["gold_count"]), int(row["analysis_slot_seed"])) for row in rows}
    if observed != expected_slots or len(observed) != len(rows):
        raise ValueError(f"{name} confirmation slots changed")
attention = json.loads(required["confirmation_attention"].read_text())
answer = json.loads(required["confirmation_answer_query"].read_text())
if int(attention["requests"]) != 100 or int(answer["requests"]) != 100:
    raise ValueError("resolved confirmation attention request count changed")
payload = {
    "schema_version": "realistic_niah_v6_confirmation_foundation_complete_v2_native_aligned",
    "status": "CONFIRMATION_FOUNDATION_COMPLETE",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "confirmation_seed_count": 10,
    "cell_count": 100,
    "panel_membership_identity": "analysis_slot_seed",
    "statistical_identity": "true_source_seed",
    "seed_aliasing": False,
    "representation_analysis": {
        "status": "DEFERRED_UNTIL_ALL_FOUR_ORIGINAL_CAPTURES_EXIST",
        "population": "original_registered_all_sample_panel",
        "running_endpoint": "item_end exact four-cell common support",
        "final_endpoint": "answer_query_v3 exact full 300-trajectory panel",
        "legacy_generic_confirmation_scan_required": False,
    },
    "freeze": str(freeze.resolve()),
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "artifacts": {
        name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in required.items()
    },
}
path = root / "freeze/confirmation_foundation_complete.json"
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
print(json.dumps(payload, sort_keys=True))
PY

"$PYTHON" scripts/run_realistic_niah_v6.py validate-confirmation-freeze \
  --config "$CONFIG" --model "$MODEL" --confirmation-freeze "$FREEZE" \
  --output "$FREEZE_ROOT/validation_after_confirmation_foundation.json"

printf 'PASS\n' >"$FREEZE_ROOT/confirmation-foundation.COMPLETE"
