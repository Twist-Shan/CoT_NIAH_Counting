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
REPLACEMENT_POOL=${V6_REPLACEMENT_POOL:-$RUN_BASE/replacement_seed_pool}
REPLACEMENT_POLICY=${V6_REPLACEMENT_POLICY:-$ROOT/configs/realistic_niah_v6_replacement_policy.json}
MODEL_ROOT=$RUN_BASE/$PROMPT_MODE/$MODEL
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
MECHANISM=$MODEL_ROOT/freeze/mechanism_frozen_confirmation.json
FREEZE=$MODEL_ROOT/freeze/confirmation_freeze.json
GENERATIONS=$MODEL_ROOT/generation/generations.jsonl
COHORT_REGISTRY=$MODEL_ROOT/replacement/confirmation/selected_cells.jsonl
TARGET_DISCOVERY=$MODEL_ROOT/causal/targeted_retrieval/discovery_formal
SELECTION=$TARGET_DISCOVERY/analysis/selection.json
BANK_K=$($PYTHON -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_k"]))' \
  "$SELECTION")
BANK_PLAN=$TARGET_DISCOVERY/plans/k$BANK_K/retrieval_anchor_bank_plan.csv
COUNT_DISCOVERY=$MODEL_ROOT/count_stream/discovery_formal
BASIS=$COUNT_DISCOVERY/running_basis.npz
DISCOVERY_ROOT=$MODEL_ROOT/causal/report_tail/discovery_formal
NATURAL_SELECTION=$DISCOVERY_ROOT/natural_layer_sweep/layer_sweep_analysis.json
CONTRACT=$DISCOVERY_ROOT/native_loop/contract
OUTPUT_ROOT=$MODEL_ROOT/causal/report_tail/confirmation_formal
NATURAL_ROOT=$OUTPUT_ROOT/natural_selected
NATIVE_ROOT=$OUTPUT_ROOT/native_loop
NATIVE_COHORT_ROOT=$MODEL_ROOT/replacement/confirmation_native_loop
NATIVE_COHORT_REGISTRY=$NATIVE_COHORT_ROOT/selected_cells.jsonl
DISCOVERY_NATIVE_MANIFEST=$MODEL_ROOT/replacement/discovery_native_loop/manifest.json
NATIVE_POLICY=${V6_COHERENT_NATIVE_LOOP_POLICY:-$ROOT/configs/realistic_niah_v6_coherent_native_loop_replacement_policy.json}
RESTORE_ROOT=$OUTPUT_ROOT/restoration
WALK_ROOT=$OUTPUT_ROOT/single_seed_walkthrough
LOG_ROOT=$OUTPUT_ROOT/logs
RESUME_FROM=${V6_REPORT_TAIL_RESUME_FROM:-start}
case "$RESUME_FROM" in
  start|native-loop|walkthrough) ;;
  *) echo "V6_REPORT_TAIL_RESUME_FROM must be start, native-loop, or walkthrough" >&2; exit 2 ;;
esac

for path in "$PYTHON" "$CONFIG" "$MECHANISM" "$FREEZE" "$GENERATIONS" \
  "$COHORT_REGISTRY" "$SELECTION" "$BANK_PLAN" "$BASIS" \
  "$NATURAL_SELECTION" "$CONTRACT/targeted_selection_compat.json" \
  "$CONTRACT/anchor_routing_compat.json" \
  "$REPLACEMENT_POOL/stimuli.jsonl" "$REPLACEMENT_POLICY" \
  "$DISCOVERY_NATIVE_MANIFEST" \
  "$NATIVE_POLICY"; do
  [[ -s "$path" ]] || { echo "missing report-tail confirmation input: $path" >&2; exit 4; }
done
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/locks"
cd "$ROOT"
exec 9>"$OUTPUT_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another $MODE/$MODEL report-tail confirmation owns the lock" >&2
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

v6_count() {
  local name=$1
  shift
  run_logged "$name" "$PYTHON" scripts/run_realistic_niah_v6_count_stream.py \
    --v6-config "$CONFIG" --confirmation-freeze "$FREEZE" \
    --cohort-registry "$COHORT_REGISTRY" "$@"
}

if [[ "$RESUME_FROM" == start ]]; then
natural_kernel() {
  local name=$1
  shift
  run_logged "$name" "$PYTHON" scripts/run_realistic_niah_v6_kernel.py \
    --target natural-aligned-progress --v6-config "$CONFIG" \
    --phase confirmation --confirmation-freeze "$FREEZE" \
    --cohort-registry "$COHORT_REGISTRY" -- \
    --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
    --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --gold-count 10 --donor-occurrence 6 \
    --tail-offset 0 "$@"
}

mkdir -p "$NATURAL_ROOT/baseline"
for direction in forward_skip backward_rewind; do
  receiver=5
  [[ "$direction" == backward_rewind ]] && receiver=7
  natural_kernel "natural_baseline_$direction" \
    --receiver-occurrence "$receiver" --patch-scope fixed_suffix \
    --patch-width 1 --layers 0 --conditions receiver_self native_donor \
    --output "$NATURAL_ROOT/baseline/${direction}_k6"
done

selected_layer() {
  local scope=$1
  "$PYTHON" -c \
    'import json,sys; rows=json.load(open(sys.argv[1]))["scopes"]; row=next(x for x in rows if x["scope"]==sys.argv[2]); print("NONE" if row["selected_layer"] is None else int(row["selected_layer"]))' \
    "$NATURAL_SELECTION" "$scope"
}

for scope in item_end_w1 event_tail_w4 item_span; do
  layer=$(selected_layer "$scope")
  if [[ "$layer" == NONE ]]; then
    mkdir -p "$NATURAL_ROOT/$scope"
    "$PYTHON" - "$scope" "$NATURAL_SELECTION" "$FREEZE" \
      "$NATURAL_ROOT/$scope/negative_skip.json" <<'PY'
import hashlib, json, os, pathlib, sys
scope, selection_raw, freeze_raw, output_raw = sys.argv[1:]
selection, freeze, output = map(pathlib.Path, (selection_raw, freeze_raw, output_raw))
row = next(value for value in json.loads(selection.read_text())["scopes"] if value["scope"] == scope)
if row["status"] != "NEGATIVE_FROZEN" or row["selected_layer"] is not None:
    raise ValueError("natural confirmation skip is not a frozen negative")
payload = {
    "schema_version": "realistic_niah_v6_natural_confirmation_negative_skip_v1",
    "status": "DISCOVERY_NEGATIVE_RETAINED_NO_CONFIRMATION_LAYER",
    "scope": scope,
    "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "confirmation_outcomes_read": False,
    "negative_result_retained": True,
}
tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(output)
print(json.dumps(payload, sort_keys=True))
PY
    continue
  fi
  case "$scope" in
    item_end_w1) patch_scope=fixed_suffix; patch_width=1 ;;
    event_tail_w4) patch_scope=fixed_suffix; patch_width=4 ;;
    item_span) patch_scope=item_span; patch_width= ;;
  esac
  for direction in forward_skip backward_rewind; do
    receiver=5
    [[ "$direction" == backward_rewind ]] && receiver=7
    scope_args=(--receiver-occurrence "$receiver" --patch-scope "$patch_scope" \
      --layers "$layer" --conditions donor_to_receiver \
      --output "$NATURAL_ROOT/$scope/${direction}_k6")
    if [[ -n "$patch_width" ]]; then
      scope_args+=(--patch-width "$patch_width")
    fi
    natural_kernel "natural_${scope}_${direction}_L${layer}" "${scope_args[@]}"
  done
done

run_logged natural_confirmation_analysis \
  "$PYTHON" scripts/analyze_realistic_niah_v6_natural_confirmation.py \
  --root "$NATURAL_ROOT" --discovery-selection "$NATURAL_SELECTION" \
  --confirmation-freeze "$FREEZE" --prompt-mode "$PROMPT_MODE" \
  --model "$MODEL" --output "$NATURAL_ROOT/confirmation_analysis.json"
elif [[ "$RESUME_FROM" == native-loop ]]; then
  "$PYTHON" - "$NATURAL_ROOT/confirmation_analysis.json" "$MODEL" \
    "$PROMPT_MODE" "$FREEZE" "$OUTPUT_ROOT/resume_native_loop.audit.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

analysis_path, model, prompt_mode, freeze_path, output_path = sys.argv[1:]
analysis = pathlib.Path(analysis_path)
freeze = pathlib.Path(freeze_path)
output = pathlib.Path(output_path)
if not analysis.is_file() or not freeze.is_file():
    raise FileNotFoundError([str(path) for path in (analysis, freeze) if not path.is_file()])
value = json.loads(analysis.read_text(encoding="utf-8"))
freeze_sha256 = hashlib.sha256(freeze.read_bytes()).hexdigest()
if value.get("status") != "CONFIRMATION_COMPLETE":
    raise ValueError("natural confirmation analysis is not complete")
if value.get("model_label") != model or value.get("prompt_mode") != prompt_mode:
    raise ValueError("natural confirmation analysis has the wrong cell identity")
if value.get("confirmation_freeze_sha256") != freeze_sha256:
    raise ValueError("natural confirmation analysis used a different freeze")
if value.get("confirmation_used_for_selection") is not False:
    raise ValueError("natural confirmation analysis changed the selection firewall")
if value.get("layer_reselected") is not False:
    raise ValueError("natural confirmation analysis reselected a frozen layer")
payload = {
    "schema_version": "realistic_niah_v6_report_tail_resume_v1",
    "status": "PASS_REUSE_COMPLETED_NATURAL_CONFIRMATION",
    "resume_from": "native_loop_coherent_panel",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "natural_analysis_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
    "confirmation_freeze_sha256": freeze_sha256,
    "completed_model_trials_recomputed": False,
    "confirmation_outcomes_used_for_selection": False,
    "seed_selection_changed": False,
    "frozen_k_changed": False,
}
tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(output)
print(json.dumps(payload, sort_keys=True))
PY
fi

if [[ "$RESUME_FROM" != walkthrough ]]; then

"$PYTHON" - "$DISCOVERY_NATIVE_MANIFEST" "$NATIVE_POLICY" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path, policy_path = map(pathlib.Path, sys.argv[1:3])
manifest = json.loads(manifest_path.read_text())
policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
if manifest.get("status") != "PASS_TRUE_SOURCE_COHERENT_NATIVE_LOOP_PANEL":
    raise ValueError("discovery native-loop coherent panel is not frozen PASS")
if manifest.get("coherent_native_loop_policy_sha256") != policy_sha:
    raise ValueError("native-loop replacement policy changed after discovery")
print(json.dumps({"native_loop_policy_freeze": "PASS", "sha256": policy_sha}))
PY

run_logged native_loop_coherent_panel \
  "$PYTHON" scripts/run_realistic_niah_v6_broad_panel_replacement.py \
  --panel-kind native_loop --v6-config "$CONFIG" \
  --mechanism-config "$MECHANISM" \
  --replacement-policy "$REPLACEMENT_POLICY" \
  --coherent-native-loop-policy "$NATIVE_POLICY" \
  --replacement-stimuli "$REPLACEMENT_POOL/stimuli.jsonl" \
  --base-cohort-registry "$COHORT_REGISTRY" --model "$MODEL" \
  --phase native_loop_confirmation --generation-root "$MODEL_ROOT/generation" \
  --output "$NATIVE_COHORT_ROOT" --cache-dir "$CACHE" --device-map auto \
  --torch-dtype bfloat16 --attention-backend sdpa \
  --confirmation-freeze "$FREEZE"

v6_count_native() {
  local name=$1
  shift
  run_logged "$name" "$PYTHON" scripts/run_realistic_niah_v6_count_stream.py \
    --v6-config "$CONFIG" --confirmation-freeze "$FREEZE" \
    --cohort-registry "$NATIVE_COHORT_REGISTRY" "$@"
}

COMMON_ROWS=(
  --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE"
  --device-map auto --torch-dtype bfloat16 --attention-backend sdpa
  --generations "$GENERATIONS" --seed-role confirmation --cohort one_to_one
  --donor-offsets -3 -2 -1 1 2 3 --random-seed 20260821
)
PLAN_DIR=$NATIVE_ROOT/plan_confirmation_offsets123
PLAN=$PLAN_DIR/native_loop_plan.csv
v6_count_native native_loop_plan plan-native-loop "${COMMON_ROWS[@]}" --output "$PLAN_DIR"
v6_count_native native_loop_p0 p0-native-loop "${COMMON_ROWS[@]}" \
  --plan "$PLAN" --basis "$BASIS" --layer "$SOURCE_LAYER" \
  --targeted-selection "$CONTRACT/targeted_selection_compat.json" \
  --anchor-routing "$CONTRACT/anchor_routing_compat.json" \
  --conditions clean self_patch full_donor_patch count_subspace_transplant \
    norm_matched_orthogonal_patch count_component_removed \
    count_component_restored --max-new-tokens 48 --output "$NATIVE_ROOT/p0"
v6_count_native native_loop_boundary boundary-native-loop "${COMMON_ROWS[@]}" \
  --plan "$PLAN" --basis "$BASIS" --layer "$SOURCE_LAYER" \
  --conditions clean self_patch full_donor_patch count_subspace_transplant \
    norm_matched_orthogonal_patch --max-new-tokens 64 \
  --output "$NATIVE_ROOT/boundary"
run_logged native_loop_analysis \
  "$PYTHON" scripts/analyze_realistic_niah_v6_native_loop.py \
  --v6-config "$CONFIG" --cohort-registry "$NATIVE_COHORT_REGISTRY" \
  --model "$MODEL" --confirmation-freeze "$FREEZE" \
  --trials "$NATIVE_ROOT/p0" "$NATIVE_ROOT/boundary" \
  --phase confirmation --output "$NATIVE_ROOT/analysis"

v6_count restoration restoration \
  --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE" \
  --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --seed-role confirmation --cohort one_to_one \
  --layer "$SOURCE_LAYER" --conditions clean trace_token_corrupt \
    ordinary_token_corrupt trace_corrupt_full_span_restore \
    trace_corrupt_endpoint_restore trace_corrupt_marker_restore \
    trace_corrupt_ordinary_state_patch ordinary_corrupt_ordinary_state_restore \
  --output "$RESTORE_ROOT/trials"
v6_count restoration_analysis analyze \
  --mechanism-config "$MECHANISM" --trials "$RESTORE_ROOT/trials" \
  --experiment-ids trace_source_restoration --outcome correct_count_margin \
  --strata layer --random-seed 20260820 --output "$RESTORE_ROOT/analysis"
else
  "$PYTHON" - "$OUTPUT_ROOT" "$MODEL" "$PROMPT_MODE" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model, prompt_mode = sys.argv[2:4]
required = {
    "natural_analysis": root / "natural_selected/confirmation_analysis.json",
    "native_coherent_panel": root.parent.parent.parent / "replacement/confirmation_native_loop/manifest.json",
    "native_coherent_mapping": root.parent.parent.parent / "replacement/confirmation_native_loop/coherent_mapping.jsonl",
    "native_plan": root / "native_loop/plan_confirmation_offsets123/manifest.json",
    "native_p0": root / "native_loop/p0/manifest.json",
    "native_boundary": root / "native_loop/boundary/manifest.json",
    "native_analysis": root / "native_loop/analysis/claim_gates.json",
    "native_analysis_adapter": root / "native_loop/analysis/v6_analysis_manifest.json",
    "restoration": root / "restoration/trials/manifest.json",
    "restoration_analysis": root / "restoration/analysis/manifest.json",
}
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
for name, path in required.items():
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{name} is not a JSON object: {path}")
print(
    json.dumps(
        {
            "status": "PASS_REUSE_PRE_WALKTHROUGH_OUTPUTS",
            "model_label": model,
            "prompt_mode": prompt_mode,
            "completed_model_trials_recomputed": False,
            "required_outputs": {name: str(path) for name, path in required.items()},
        },
        sort_keys=True,
    )
)
PY
fi

WALK_CONFIG=$WALK_ROOT/walkthrough_config.json
mkdir -p "$WALK_ROOT"
"$PYTHON" - "$GENERATIONS" "$COHORT_REGISTRY" "$MODEL" \
  "$SOURCE_LAYER" "$WALK_CONFIG" <<'PY'
import json, os, pathlib, sys


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows

generations_raw, registry_raw, model, layer_raw, output_raw = sys.argv[1:]
generations, registry, output = map(pathlib.Path, (generations_raw, registry_raw, output_raw))
rows = read_jsonl(generations)
by_id = {str(row["request_id"]): row for row in rows}
cells = read_jsonl(registry)
candidates = []
for cell in cells:
    if int(cell["gold_count"]) != 10:
        continue
    request = by_id[str(cell["source_request_id"])]
    if request.get("trace_parse", {}).get("strict_causal_eligible") is True:
        candidates.append(
            {
                "request": request,
                "analysis_slot_seed": int(cell["analysis_slot_seed"]),
                "replacement_applied": bool(cell["replacement_applied"]),
            }
        )
if not candidates:
    raise ValueError("No resolved strict N=10 confirmation case for walkthrough")
chosen_cell = min(
    candidates,
    key=lambda row: (int(row["analysis_slot_seed"]), int(row["request"]["seed"])),
)
chosen = chosen_cell["request"]
payload = {
    "schema_version": "realistic_niah_v6_single_seed_walkthrough_protocol_v1",
    "status": "FROZEN_FORMAT_ONLY_CASE_STUDY",
    "purpose": "V6 enumeration analogue of the Native-thinking one-case restoration walkthrough",
    "case_study_not_inferential": True,
    "case_selected_by_outcome": False,
    "selection_rule": "lowest fixed confirmation analysis slot in the resolved fresh-strict N=10 panel; no intervention outcome read",
    "intervention": {
        "answer_query_patched": False,
        "candidate_counts": list(range(1, 11)),
        "layer_mode": "cumulative clamp from frozen source layer",
    },
    "models": {
        model: {
            "seed": int(chosen["seed"]),
            "analysis_slot_seed": int(chosen_cell["analysis_slot_seed"]),
            "replacement_applied": bool(chosen_cell["replacement_applied"]),
            "gold_count": 10,
            "request_id": str(chosen["request_id"]),
            "source_layer": int(layer_raw),
        }
    },
    "selection_inputs": ["split", "analysis_slot_seed", "true_source_seed", "gold_count", "fresh_strict_parser_status"],
    "intervention_outcomes_read": False,
    "reporting": "illustrative paths only; no population inference",
}
tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(output)
print(json.dumps(payload, sort_keys=True))
PY
WALK_SEED=$($PYTHON -c \
  'import json,sys; v=json.load(open(sys.argv[1])); print(v["models"][sys.argv[2]]["seed"])' \
  "$WALK_CONFIG" "$MODEL")
WALK_REQUEST=$($PYTHON -c \
  'import json,sys; v=json.load(open(sys.argv[1])); print(v["models"][sys.argv[2]]["request_id"])' \
  "$WALK_CONFIG" "$MODEL")
run_logged single_seed_walkthrough \
  "$PYTHON" scripts/run_realistic_niah_v6_kernel.py \
  --target single-seed-walkthrough --v6-config "$CONFIG" \
  --phase confirmation --confirmation-freeze "$FREEZE" \
  --cohort-registry "$COHORT_REGISTRY" -- \
  --mechanism-config "$MECHANISM" --walkthrough-config "$WALK_CONFIG" \
  --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
  --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --request-id "$WALK_REQUEST" \
  --seed "$WALK_SEED" --expected-count 10 --source-layer "$SOURCE_LAYER" \
  --max-new-tokens 16 --output "$WALK_ROOT/trials"
run_logged single_seed_walkthrough_analysis \
  "$PYTHON" scripts/analyze_realistic_niah_v5_single_seed_walkthrough.py \
  --input "$WALK_ROOT/trials/walkthrough_rows.jsonl" \
  --output "$WALK_ROOT/analysis"

"$PYTHON" - "$OUTPUT_ROOT" "$MODEL" "$PROMPT_MODE" "$FREEZE" \
  "$COHORT_REGISTRY" <<'PY'
import hashlib, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
model, prompt_mode = sys.argv[2:4]
freeze, cohort = map(pathlib.Path, sys.argv[4:6])
required = {
    "natural_analysis": root / "natural_selected/confirmation_analysis.json",
    "native_coherent_panel": root.parent.parent.parent / "replacement/confirmation_native_loop/manifest.json",
    "native_coherent_mapping": root.parent.parent.parent / "replacement/confirmation_native_loop/coherent_mapping.jsonl",
    "native_plan": root / "native_loop/plan_confirmation_offsets123/manifest.json",
    "native_p0": root / "native_loop/p0/manifest.json",
    "native_boundary": root / "native_loop/boundary/manifest.json",
    "native_analysis": root / "native_loop/analysis/claim_gates.json",
    "native_analysis_adapter": root / "native_loop/analysis/v6_analysis_manifest.json",
    "restoration": root / "restoration/trials/manifest.json",
    "restoration_analysis": root / "restoration/analysis/manifest.json",
    "walkthrough": root / "single_seed_walkthrough/analysis/walkthrough_complete.json",
    "walkthrough_config": root / "single_seed_walkthrough/walkthrough_config.json",
}
for scope in ("item_end_w1", "event_tail_w4", "item_span"):
    directory = root / "natural_selected" / scope
    negative = directory / "negative_skip.json"
    if negative.is_file():
        required[f"natural_{scope}_negative"] = negative
        continue
    for direction in ("forward_skip", "backward_rewind"):
        manifest = directory / f"{direction}_k6/manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        required[f"natural_{scope}_{direction}"] = manifest
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
walkthrough = json.loads(required["walkthrough"].read_text())
if walkthrough["status"] != "PASS" or not walkthrough["case_study_not_inferential"]:
    raise ValueError("single-seed walkthrough contract failed")
payload = {
    "schema_version": "realistic_niah_v6_report_tail_confirmation_complete_v1",
    "status": "CONFIRMATION_COMPLETE",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "confirmation_used_for_selection": False,
    "negative_results_retained": True,
    "single_seed_walkthrough_inferential": False,
    "panel_membership_identity": "analysis_slot_seed",
    "statistical_identity": "true_source_seed",
    "seed_aliasing": False,
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "cohort_registry_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
    "outputs": {
        name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in required.items()
    },
}
path = root / "report_tail_confirmation_complete.json"
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
print(json.dumps(payload, sort_keys=True))
PY

printf 'PASS\n' >"$OUTPUT_ROOT/confirmation.COMPLETE"
