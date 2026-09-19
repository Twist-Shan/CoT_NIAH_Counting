#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi
MODEL=$1
GPU_INDEX=$2
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
COHERENT_NATIVE_LOOP_POLICY=${V6_COHERENT_NATIVE_LOOP_POLICY:-$ROOT/configs/realistic_niah_v6_coherent_native_loop_replacement_policy.json}
CONFIRMATION_FOUNDATION_RESUME_FROM=${V6_CONFIRMATION_FOUNDATION_RESUME_FROM:-start}
QUEUE_ROOT=$RUN_BASE/queue_logs
mkdir -p "$QUEUE_ROOT"
QUEUE_LOG=$QUEUE_ROOT/${MODEL}_confirmation.log
exec > >(tee -a "$QUEUE_LOG") 2>&1

wait_for_pass() {
  local marker=$1
  local label=$2
  echo "[$(date --iso-8601=seconds)] WAIT $label marker=$marker"
  while [[ ! -s "$marker" ]]; do sleep 30; done
  grep -qx PASS "$marker" || { echo "$label is not PASS" >&2; exit 1; }
  echo "[$(date --iso-8601=seconds)] READY $label"
}

reuse_or_run() {
  local marker=$1
  local label=$2
  shift 2
  if [[ -s "$marker" ]] && grep -qx PASS "$marker"; then
    echo "[$(date --iso-8601=seconds)] REUSE $label marker=$marker"
    return 0
  fi
  echo "[$(date --iso-8601=seconds)] START $label"
  "$@"
  if [[ ! -s "$marker" ]] || ! grep -qx PASS "$marker"; then
    echo "$label did not write a PASS marker: $marker" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] PASS $label"
}

wait_for_pass "$QUEUE_ROOT/${MODEL}_report_tail_discovery.COMPLETE" \
  "$MODEL complete discovery suite"

# A queue may be resumed from an arbitrary launcher working directory.  Older
# V6 launchers resolved the freeze script relative to that directory and could
# therefore fail before opening confirmation data.  Record that structural
# recovery explicitly; it is never a sample failure and never replaces a seed.
if grep -Fq "can't open file 'scripts/freeze_realistic_niah_v6_confirmation.py'" \
    "$QUEUE_LOG"; then
  printf '%s\n' \
    '{"status":"PASS_CWD_INDEPENDENT_FREEZE_PATH_RESUME","failure_class":"structural_launcher_path","confirmation_data_opened_before_failure":false,"sample_failure":false,"seed_replacement_triggered":false,"recovery":"absolute_ROOT_freeze_script_path"}' \
    >"$QUEUE_ROOT/${MODEL}_confirmation_path.recovery.json"
  echo "[$(date --iso-8601=seconds)] RECOVERY $MODEL absolute freeze-script path"
fi
if grep -Fq "missing confirmation input: $STIMULI" "$QUEUE_LOG" && \
    [[ -s "$STIMULI" ]]; then
  stimuli_sha256=$(sha256sum "$STIMULI" | awk '{print $1}')
  printf \
    '{"status":"PASS_AUTHORIZED_STIMULI_PRESENT_ON_RESUME","failure_class":"structural_missing_input_path","confirmation_data_opened_before_failure":false,"sample_failure":false,"seed_replacement_triggered":false,"stimuli_path":"%s","stimuli_sha256":"%s"}\n' \
    "$STIMULI" "$stimuli_sha256" \
    >"$QUEUE_ROOT/${MODEL}_confirmation_stimuli_path.recovery.json"
  echo "[$(date --iso-8601=seconds)] RECOVERY $MODEL authorized stimuli present sha256=$stimuli_sha256"
fi

for mode in index bullet; do
  prompt_mode=enumeration_$mode
  model_root=$RUN_BASE/$prompt_mode/$MODEL
  config=$ROOT/configs/realistic_niah_v6_${prompt_mode}.json
  mechanism=$ROOT/configs/realistic_niah_v6_${prompt_mode}_count_stream_dev.json
  freeze_root=$model_root/freeze
  if [[ -s "$freeze_root/full-confirmation.COMPLETE" ]] && \
      grep -qx PASS "$freeze_root/full-confirmation.COMPLETE"; then
    echo "[$(date --iso-8601=seconds)] REUSE $MODEL $mode full confirmation suite"
    continue
  fi
  if [[ -s "$freeze_root/freeze.COMPLETE" ]] && \
      grep -qx PASS "$freeze_root/freeze.COMPLETE" && \
      [[ -s "$freeze_root/confirmation_freeze.json" ]] && \
      [[ -s "$freeze_root/mechanism_frozen_confirmation.json" ]]; then
    "$PYTHON" "$ROOT/scripts/run_realistic_niah_v6.py" \
      validate-confirmation-freeze \
      --config "$config" --model "$MODEL" \
      --confirmation-freeze "$freeze_root/confirmation_freeze.json" \
      --output "$freeze_root/validation_before_queue_resume.json"
    echo "[$(date --iso-8601=seconds)] REUSE validated $MODEL $mode discovery freeze"
  else
    echo "[$(date --iso-8601=seconds)] START $MODEL $mode discovery freeze"
    "$PYTHON" "$ROOT/scripts/freeze_realistic_niah_v6_confirmation.py" \
      --v6-config "$config" --mechanism-config "$mechanism" \
      --model "$MODEL" --model-root "$model_root" --output "$freeze_root"
    grep -qx PASS "$freeze_root/freeze.COMPLETE"
    echo "[$(date --iso-8601=seconds)] PASS $MODEL $mode discovery freeze"
  fi

  common_env=(env V6_ROOT="$ROOT" V6_PYTHON="$PYTHON" V6_CACHE="$CACHE"
    V6_RUN_BASE="$RUN_BASE" V6_STIMULI="$STIMULI"
    V6_REPLACEMENT_POOL="$REPLACEMENT_POOL"
    V6_REPLACEMENT_POLICY="$REPLACEMENT_POLICY"
    V6_COHERENT_NATIVE_LOOP_POLICY="$COHERENT_NATIVE_LOOP_POLICY"
    V6_CONFIRMATION_FOUNDATION_RESUME_FROM="$CONFIRMATION_FOUNDATION_RESUME_FROM"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1)

  reuse_or_run "$freeze_root/confirmation-foundation.COMPLETE" \
    "$MODEL $mode confirmation foundation" \
    "${common_env[@]}" bash \
    "$ROOT/scripts/supervise_realistic_niah_v6_confirmation_foundation.sh" \
    "$mode" "$MODEL" "$GPU_INDEX"

  reuse_or_run \
    "$model_root/causal/targeted_retrieval/confirmation_formal/confirmation.COMPLETE" \
    "$MODEL $mode targeted confirmation" \
    "${common_env[@]}" bash \
    "$ROOT/scripts/supervise_realistic_niah_v6_targeted_retrieval_confirmation.sh" \
    "$mode" "$MODEL" "$GPU_INDEX"

  reuse_or_run \
    "$model_root/count_stream/confirmation_formal/confirmation.COMPLETE" \
    "$MODEL $mode count confirmation" \
    "${common_env[@]}" bash \
    "$ROOT/scripts/supervise_realistic_niah_v6_count_stream_confirmation.sh" \
    "$mode" "$MODEL" "$GPU_INDEX"

  reuse_or_run \
    "$model_root/causal/specialized/confirmation_formal/confirmation.COMPLETE" \
    "$MODEL $mode specialized confirmation" \
    "${common_env[@]}" bash \
    "$ROOT/scripts/supervise_realistic_niah_v6_specialized_confirmation.sh" \
    "$mode" "$MODEL" "$GPU_INDEX"

  reuse_or_run \
    "$model_root/causal/report_tail/confirmation_formal/confirmation.COMPLETE" \
    "$MODEL $mode report-tail confirmation" \
    "${common_env[@]}" bash \
    "$ROOT/scripts/supervise_realistic_niah_v6_report_tail_confirmation.sh" \
    "$mode" "$MODEL" "$GPU_INDEX"

  printf 'PASS\n' >"$model_root/freeze/full-confirmation.COMPLETE"
done

printf 'PASS\n' >"$QUEUE_ROOT/${MODEL}_confirmation.COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE $MODEL V6 confirmation suite"
