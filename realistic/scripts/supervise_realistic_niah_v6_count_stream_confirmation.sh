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
  Qwen3-8B) SOURCE_LAYER=18; READOUT_LAYER=19 ;;
  Gemma4-E4B) SOURCE_LAYER=16; READOUT_LAYER=17 ;;
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
DISCOVERY_ROOT=$MODEL_ROOT/count_stream/discovery_formal
RUN_ROOT=$MODEL_ROOT/count_stream/confirmation_formal
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
MECHANISM=$MODEL_ROOT/freeze/mechanism_frozen_confirmation.json
FREEZE=$MODEL_ROOT/freeze/confirmation_freeze.json
GENERATIONS=$MODEL_ROOT/generation/generations.jsonl
COHORT_REGISTRY=$MODEL_ROOT/replacement/confirmation/selected_cells.jsonl
COHERENT_REGISTRY=$MODEL_ROOT/replacement/confirmation_broad/selected_cells.jsonl
BASIS=$DISCOVERY_ROOT/running_basis.npz
STAGE1=$DISCOVERY_ROOT/stage1_complete.json
LOG=$RUN_ROOT/logs/confirmation.log

for path in "$PYTHON" "$CONFIG" "$MECHANISM" "$FREEZE" \
  "$GENERATIONS" "$COHORT_REGISTRY" "$COHERENT_REGISTRY" "$BASIS" "$STAGE1"; do
  [[ -s "$path" ]] || { echo "missing count confirmation input: $path" >&2; exit 4; }
done
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$RUN_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another $MODE/$MODEL count confirmation owns the lock" >&2
  exit 75
fi

run_logged() {
  local name=$1
  shift
  echo "[$(date --iso-8601=seconds)] START $name"
  "$@"
  echo "[$(date --iso-8601=seconds)] PASS $name"
}

v6() {
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" scripts/run_realistic_niah_v6_count_stream.py \
    --v6-config "$CONFIG" --confirmation-freeze "$FREEZE" \
    --cohort-registry "$COHORT_REGISTRY" "$@"
}

v6_broad() {
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" scripts/run_realistic_niah_v6_count_stream.py \
    --v6-config "$CONFIG" --confirmation-freeze "$FREEZE" \
    --cohort-registry "$COHERENT_REGISTRY" "$@"
}

selection_dir() {
  local key=$1
  "$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "$STAGE1" "$key"
}

reusable_broad_confirmation() {
  local source=$1
  local decision=$2
  local output=$3
  "$PYTHON" - "$source" "$decision" "$FREEZE" "$output" \
    "$MODEL" "$PROMPT_MODE" <<'PY'
import hashlib
import json
import pathlib
import sys

source, decision_raw, freeze_raw, output_raw, model, prompt_mode = sys.argv[1:]
decision = pathlib.Path(decision_raw)
freeze = pathlib.Path(freeze_raw)
output = pathlib.Path(output_raw)

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

negative = output / "negative_skip.json"
if negative.is_file():
    try:
        value = json.loads(negative.read_text())
        if value.get("status") != "DISCOVERY_NEGATIVE_RETAINED_NO_CONFIRMATION_BANK":
            raise ValueError("broad negative-skip status changed")
        if value.get("discovery_decision_sha256") != sha256(decision):
            raise ValueError("broad negative-skip decision hash changed")
        if value.get("confirmation_freeze_sha256") != sha256(freeze):
            raise ValueError("broad negative-skip freeze hash changed")
    except Exception as exc:
        print(f"INVALID_REUSE {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print("REUSE_VALIDATED_NEGATIVE")
    raise SystemExit(0)

manifest_path = output / "manifest.json"
analysis_path = output / "confirmation_analysis.json"
adapter_path = output / "v6_adapter_manifest.json"
if not all(path.is_file() for path in (manifest_path, analysis_path, adapter_path)):
    raise SystemExit(3)
try:
    manifest = json.loads(manifest_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    adapter = json.loads(adapter_path.read_text())
    shards = list((output / "shards").glob("*.jsonl"))
    if int(manifest.get("completed_shards", -1)) != len(shards) or not shards:
        raise ValueError("broad confirmation shard manifest is incomplete")
    if adapter.get("run_status") != "COMPLETE":
        raise ValueError("broad confirmation adapter did not complete")
    if analysis.get("status") != "CONFIRMATION_EVALUATED_FROZEN_K":
        raise ValueError("broad confirmation analysis status changed")
    if analysis.get("model_label") != model or analysis.get("prompt_mode") != prompt_mode:
        raise ValueError("broad confirmation analysis cell changed")
    if analysis.get("source_group") != source:
        raise ValueError("broad confirmation source group changed")
    if analysis.get("confirmation_used_for_selection") is not False:
        raise ValueError("confirmation leaked into broad-K selection")
    if analysis.get("bank_size_reselected") is not False:
        raise ValueError("confirmation reselected broad K")
    if analysis.get("seed_aliasing") is not False:
        raise ValueError("broad confirmation aliases a source seed")
    if analysis.get("trials_manifest_sha256") != sha256(manifest_path):
        raise ValueError("broad confirmation trial-manifest hash changed")
    if analysis.get("discovery_decision_sha256") != sha256(decision):
        raise ValueError("broad confirmation discovery-decision hash changed")
    if analysis.get("confirmation_freeze_sha256") != sha256(freeze):
        raise ValueError("broad confirmation freeze hash changed")
    plan = pathlib.Path(analysis["frozen_plan"])
    if analysis.get("frozen_plan_sha256") != sha256(plan):
        raise ValueError("broad confirmation frozen-plan hash changed")
except Exception as exc:
    print(f"INVALID_REUSE {exc}", file=sys.stderr)
    raise SystemExit(2) from exc
print(f"REUSE_VALIDATED_COMPLETE shards={len(shards)}")
PY
}

run_broad_confirmation() {
  local source=$1
  local selection_key=$2
  local output=$3
  local selected_dir decision status selected_k plan
  selected_dir=$(selection_dir "$selection_key")
  decision=$selected_dir/k_selection_decision.json
  [[ -s "$decision" ]] || { echo "missing broad decision: $decision" >&2; exit 4; }
  local reuse_status
  if reusable_broad_confirmation "$source" "$decision" "$output"; then
    echo "[$(date --iso-8601=seconds)] REUSE validated broad $source confirmation"
    return
  else
    reuse_status=$?
    if [[ "$reuse_status" -ne 3 ]]; then
      echo "invalid existing broad $source confirmation; fail closed" >&2
      exit "$reuse_status"
    fi
  fi
  status=$($PYTHON -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$decision")
  mkdir -p "$output"
  if [[ "$status" == no_positive_discovery_bank ]]; then
    "$PYTHON" - "$source" "$decision" "$FREEZE" "$output/negative_skip.json" <<'PY'
import hashlib, json, os, pathlib, sys
source, decision_raw, freeze_raw, output_raw = sys.argv[1:]
decision, freeze, output = map(pathlib.Path, (decision_raw, freeze_raw, output_raw))
value = json.loads(decision.read_text())
if value["status"] != "no_positive_discovery_bank":
    raise ValueError("negative confirmation skip received a nonnegative decision")
payload = {
    "schema_version": "realistic_niah_v6_broad_confirmation_negative_skip_v1",
    "status": "DISCOVERY_NEGATIVE_RETAINED_NO_CONFIRMATION_BANK",
    "source_group": source,
    "discovery_decision": str(decision.resolve()),
    "discovery_decision_sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
    "confirmation_freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "confirmation_outcomes_read": False,
    "negative_result_retained": True,
}
tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(output)
print(json.dumps(payload, sort_keys=True))
PY
    return
  fi
  [[ "$status" == frozen_for_confirmation ]] || {
    echo "unfrozen broad decision: $decision status=$status" >&2
    exit 5
  }
  selected_k=$($PYTHON -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_bank_size"]))' \
    "$decision")
  plan=$selected_dir/frozen_answer_broad_head_plan.csv
  [[ -s "$plan" ]] || { echo "missing frozen broad plan: $plan" >&2; exit 4; }
  run_logged "broad_${source}_k${selected_k}" v6_broad broad-heads \
    --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE" \
    --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --seed-role confirmation \
    --cohort parser_hit --row-panel broad_confirmation --plan "$plan" \
    --bank-size "$selected_k" --skip-greedy --output "$output"
  run_logged "analyze_broad_${source}_k${selected_k}" \
    "$PYTHON" scripts/analyze_realistic_niah_v6_broad_confirmation.py \
    --v6-config "$CONFIG" --mechanism-config "$MECHANISM" \
    --model "$MODEL" --source-group "$source" --trials "$output" \
    --discovery-decision "$decision" --frozen-plan "$plan" \
    --confirmation-freeze "$FREEZE" --random-seed 20260820 \
    --output "$output/confirmation_analysis.json"
}

run_broad_confirmation trace_items trace_selection_dir \
  "$RUN_ROOT/broad_trace_confirmation"
run_broad_confirmation prompt_records prompt_selection_dir \
  "$RUN_ROOT/broad_prompt_confirmation"

TRACE_PLAN_DIR=$RUN_ROOT/trace_pair_plan
TRACE_PLAN=$TRACE_PLAN_DIR/trace_patch_pair_plan.csv
TRACE_PATCH=$RUN_ROOT/trace_patch
RECOVERY=$RUN_ROOT/trace_pair_plan_confirmation_role_guard.recovery.json
if [[ -s "$LOG" ]] && \
    grep -Fq "Trace-patch pair sampling is frozen on development rows" "$LOG" && \
    [[ ! -s "$RECOVERY" ]]; then
  trace_shards=0
  if [[ -d "$TRACE_PATCH/shards" ]]; then
    trace_shards=$(find "$TRACE_PATCH/shards" -maxdepth 1 -type f \
      -name '*.jsonl' | wc -l)
  fi
  [[ "$trace_shards" -eq 0 ]] || {
    echo "development-role guard appeared after trace shards existed" >&2
    exit 6
  }
  [[ ! -s "$TRACE_PLAN_DIR/manifest.json" ]] || {
    echo "development-role guard appeared despite a completed trace plan" >&2
    exit 6
  }
  "$PYTHON" - "$RECOVERY" "$LOG" \
    "$TRACE_PLAN_DIR/v6_adapter_manifest.json" \
    "$ROOT/scripts/run_realistic_niah_v6_count_stream.py" \
    "$MODEL" "$PROMPT_MODE" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
failure_log = pathlib.Path(sys.argv[2])
adapter = pathlib.Path(sys.argv[3])
runner = pathlib.Path(sys.argv[4])
model, prompt_mode = sys.argv[5:7]

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

payload = {
    "schema_version": "realistic_niah_v6_structural_recovery_v1",
    "status": "PASS_CONFIRMATION_TRACE_PLAN_ROLE_ADAPTER_RECOVERY",
    "failure_class": "structural_legacy_development_only_dispatch_guard",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "recovered_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "legacy_error": "Trace-patch pair sampling is frozen on development rows",
    "recovery": "outcome_blind_v6_confirmation_plan_adapter",
    "confirmation_outcomes_read_for_pair_selection": False,
    "trace_patch_model_loaded_before_failure": False,
    "completed_trace_patch_shards_before_recovery": 0,
    "completed_model_trials_recomputed": False,
    "completed_broad_confirmation_outputs_reused": True,
    "sample_failure": False,
    "seed_replacement_triggered": False,
    "failure_log": str(failure_log.resolve()),
    "failure_log_sha256_at_recovery": sha256(failure_log),
    "prior_adapter_manifest": str(adapter.resolve()),
    "prior_adapter_manifest_sha256": sha256(adapter),
    "recovery_runner": str(runner.resolve()),
    "recovery_runner_sha256": sha256(runner),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(payload, sort_keys=True))
PY
fi
RESUME_FLAG_RECOVERY=$RUN_ROOT/trace_patch_resume_flag.recovery.json
if [[ -s "$LOG" ]] && \
    grep -Fq "unrecognized arguments: --resume" "$LOG" && \
    [[ ! -s "$RESUME_FLAG_RECOVERY" ]]; then
  trace_shards=0
  if [[ -d "$TRACE_PATCH/shards" ]]; then
    trace_shards=$(find "$TRACE_PATCH/shards" -maxdepth 1 -type f \
      -name '*.jsonl' | wc -l)
  fi
  [[ "$trace_shards" -eq 0 ]] || {
    echo "explicit resume-flag failure appeared after trace shards existed" >&2
    exit 6
  }
  "$PYTHON" - "$RESUME_FLAG_RECOVERY" "$LOG" "$MODEL" \
    "$PROMPT_MODE" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
failure_log = pathlib.Path(sys.argv[2])
model, prompt_mode = sys.argv[3:5]
payload = {
    "schema_version": "realistic_niah_v6_structural_recovery_v1",
    "status": "PASS_IMPLICIT_DEFAULT_RESUME_FLAG_RECOVERY",
    "failure_class": "structural_cli_flag_mismatch",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "recovered_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "legacy_error": "unrecognized arguments: --resume",
    "recovery": "remove_explicit_flag_and_use_registered_default_resume_true",
    "default_resume_policy_changed": False,
    "trace_patch_model_loaded_before_failure": False,
    "completed_trace_patch_shards_before_recovery": 0,
    "completed_model_trials_recomputed": False,
    "sample_failure": False,
    "seed_replacement_triggered": False,
    "failure_log": str(failure_log.resolve()),
    "failure_log_sha256_at_recovery": hashlib.sha256(
        failure_log.read_bytes()
    ).hexdigest(),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(payload, sort_keys=True))
PY
fi
run_logged plan_trace_patch v6 plan-trace-patch \
  --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE" \
  --generations "$GENERATIONS" --seed-role confirmation --cohort one_to_one \
  --output "$TRACE_PLAN_DIR"

run_logged trace_patch v6 trace-patch \
  --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE" \
  --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --seed-role confirmation --cohort one_to_one \
  --row-panel trace_patch --pair-plan "$TRACE_PLAN" --basis "$BASIS" \
  --layer "$SOURCE_LAYER" --readout-layers "$READOUT_LAYER" \
  --conditions clean self_patch full_donor_patch progress_projected_patch \
    norm_matched_orthogonal_patch --random-seed 20260820 --skip-greedy \
  --output "$TRACE_PATCH"

run_logged analyze_trace_patch v6 analyze \
  --mechanism-config "$MECHANISM" --trials "$TRACE_PATCH" \
  --experiment-ids trace_intermediate_state_patching \
    trace_terminal_state_patching \
  --outcome donor_vs_receiver_city_log_odds --strata donor_direction \
  --random-seed 20260820 --output "$RUN_ROOT/trace_patch_analysis"

"$PYTHON" - "$RUN_ROOT" "$MODEL" "$PROMPT_MODE" "$FREEZE" \
  "$COHORT_REGISTRY" "$COHERENT_REGISTRY" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model, prompt_mode = sys.argv[2:4]
freeze, cell_registry, broad_registry = map(pathlib.Path, sys.argv[4:7])
required = {
    "trace_plan": root / "trace_pair_plan/manifest.json",
    "trace_trials": root / "trace_patch/manifest.json",
    "trace_analysis": root / "trace_patch_analysis/manifest.json",
}
for name in ("broad_trace_confirmation", "broad_prompt_confirmation"):
    trial = root / name / "confirmation_analysis.json"
    negative = root / name / "negative_skip.json"
    if trial.is_file():
        required[name] = trial
    elif negative.is_file():
        required[name] = negative
    else:
        raise FileNotFoundError(f"neither broad confirmation nor negative skip exists: {name}")
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
pair = json.loads(required["trace_plan"].read_text())
trials = json.loads(required["trace_trials"].read_text())
if int(trials["completed_shards"]) != int(pair["pair_count"]):
    raise ValueError("confirmation trace shards do not match the frozen pair plan")
payload = {
    "schema_version": "realistic_niah_v6_count_stream_confirmation_complete_v1",
    "status": "CONFIRMATION_COMPLETE",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "trace_pair_count": int(pair["pair_count"]),
    "confirmation_used_for_selection": False,
    "negative_results_retained": True,
    "panel_membership_identity": "analysis_slot_seed",
    "statistical_identity": "true_source_seed",
    "seed_aliasing": False,
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "cell_registry_sha256": hashlib.sha256(cell_registry.read_bytes()).hexdigest(),
    "coherent_registry_sha256": hashlib.sha256(broad_registry.read_bytes()).hexdigest(),
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

printf 'PASS\n' >"$RUN_ROOT/confirmation.COMPLETE"
