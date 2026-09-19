#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <index|bullet> <Qwen3-8B|Gemma4-E4B> <phase>" >&2
  echo "phases: preflight discovery-generate discovery-foundation discovery-supplement discovery-foundation-resolved confirmation-generate confirmation-supplement confirmation-foundation confirmation-foundation-resolved suite-audit" >&2
  exit 2
fi

MODE=$1
MODEL=$2
PHASE=$3

case "$MODE" in
  index) PROMPT_MODE=enumeration_index ;;
  bullet) PROMPT_MODE=enumeration_bullet ;;
  *) echo "mode must be index or bullet" >&2; exit 2 ;;
esac

case "$MODEL" in
  Qwen3-8B|Gemma4-E4B) ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
STIMULI=${V6_STIMULI:-$ROOT/work/nonthinking_report_filestream_stage3/stimuli.jsonl}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_ROOT=${V6_RUN_ROOT:-$ROOT/work/realistic_niah_v6/$PROMPT_MODE/$MODEL}
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
REPLACEMENT_POLICY=${V6_REPLACEMENT_POLICY:-$ROOT/configs/realistic_niah_v6_replacement_policy.json}
REPLACEMENT_POOL=${V6_REPLACEMENT_POOL:-$ROOT/work/realistic_niah_v6/replacement_seed_pool}
GEN_ROOT=$RUN_ROOT/generation
GENERATIONS=$GEN_ROOT/generations.jsonl
RESOLVED_DISCOVERY=$RUN_ROOT/replacement/discovery
DISCOVERY_REGISTRY=$RESOLVED_DISCOVERY/selected_cells.jsonl
RESOLVED_CONFIRMATION=$RUN_ROOT/replacement/confirmation
CONFIRMATION_REGISTRY=$RESOLVED_CONFIRMATION/selected_cells.jsonl
FORMAL_CAPTURE=$RUN_ROOT/capture/formal
ALL_CAPTURE=$RUN_ROOT/capture/all_sample
CONFIRMATION_FORMAL_CAPTURE=$RUN_ROOT/capture/confirmation_formal
CONFIRMATION_ALL_CAPTURE=$RUN_ROOT/capture/confirmation_all_sample

if [[ ! -x "$PYTHON" ]]; then
  echo "V6_PYTHON is not executable: $PYTHON" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$GEN_ROOT"
cd "$ROOT"

gpu_prefix=()
if [[ -n "${V6_CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_prefix=(env CUDA_VISIBLE_DEVICES="$V6_CUDA_VISIBLE_DEVICES")
fi

run_logged() {
  local name=$1
  shift
  {
    echo "[$(date --iso-8601=seconds)] START $name"
    printf 'COMMAND'
    printf ' %q' "${gpu_prefix[@]}" "$@"
    printf '\n'
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
        --format=csv,noheader || true
    fi
    "${gpu_prefix[@]}" "$@"
    echo "[$(date --iso-8601=seconds)] PASS $name"
  } 2>&1 | tee "$RUN_ROOT/logs/$name.log"
}

preflight() {
  run_logged preflight \
    "$PYTHON" scripts/run_realistic_niah_v6.py preflight \
    --config "$CONFIG" \
    --stimuli "$STIMULI" \
    --output "$RUN_ROOT/preflight.json"
}

generate_discovery() {
  run_logged discovery_generate \
    "$PYTHON" scripts/run_realistic_niah_v6.py generate \
    --config "$CONFIG" \
    --model "$MODEL" \
    --cache-dir "$CACHE" \
    --stimuli "$STIMULI" \
    --seed-role discovery \
    --output "$GEN_ROOT"
}

generate_confirmation() {
  if [[ -z "${V6_CONFIRMATION_FREEZE:-}" ]]; then
    echo "V6_CONFIRMATION_FREEZE is required for confirmation" >&2
    exit 2
  fi
  run_logged confirmation_generate \
    "$PYTHON" scripts/run_realistic_niah_v6.py generate \
    --config "$CONFIG" \
    --model "$MODEL" \
    --cache-dir "$CACHE" \
    --stimuli "$STIMULI" \
    --seed-role confirmation \
    --confirmation-freeze "$V6_CONFIRMATION_FREEZE" \
    --output "$GEN_ROOT"
}

foundation_discovery() {
  if [[ ! -f "$GENERATIONS" ]]; then
    echo "missing discovery generations: $GENERATIONS" >&2
    exit 2
  fi
  run_logged discovery_capture_formal \
    "$PYTHON" scripts/run_realistic_niah_v6.py capture \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role discovery \
    --output "$FORMAL_CAPTURE"
  run_logged discovery_capture_all_sample \
    "$PYTHON" scripts/run_realistic_niah_v6.py capture \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role discovery --include-nonstrict \
    --output "$ALL_CAPTURE"
  run_logged discovery_attention_formal \
    "$PYTHON" scripts/run_realistic_niah_v6.py attention \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role discovery \
    --output "$RUN_ROOT/attention/discovery_formal.csv"
  run_logged discovery_attention_all_sample \
    "$PYTHON" scripts/run_realistic_niah_v6.py attention \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role discovery --include-nonstrict \
    --output "$RUN_ROOT/attention/discovery_all_sample.csv"
  run_logged discovery_answer_query_formal \
    "$PYTHON" scripts/run_realistic_niah_v6.py attention-answer-query \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role discovery \
    --output "$RUN_ROOT/attention/discovery_answer_query_formal.csv"
}

supplement_discovery() {
  if [[ ! -s "$REPLACEMENT_POOL/stimuli.jsonl" ]]; then
    echo "missing frozen-amendment replacement pool: $REPLACEMENT_POOL/stimuli.jsonl" >&2
    exit 2
  fi
  run_logged discovery_supplement \
    "$PYTHON" scripts/run_realistic_niah_v6_replacement_generation.py \
    --v6-config "$CONFIG" --replacement-policy "$REPLACEMENT_POLICY" \
    --replacement-stimuli "$REPLACEMENT_POOL/stimuli.jsonl" \
    --model "$MODEL" --seed-role discovery --generation-root "$GEN_ROOT" \
    --output "$RESOLVED_DISCOVERY" --cache-dir "$CACHE" \
    --device-map auto --torch-dtype bfloat16 --attention-backend sdpa
}

supplement_confirmation() {
  if [[ -z "${V6_CONFIRMATION_FREEZE:-}" ]]; then
    echo "V6_CONFIRMATION_FREEZE is required for confirmation" >&2
    exit 2
  fi
  if [[ ! -s "$REPLACEMENT_POOL/stimuli.jsonl" ]]; then
    echo "missing frozen-amendment replacement pool: $REPLACEMENT_POOL/stimuli.jsonl" >&2
    exit 2
  fi
  run_logged confirmation_supplement \
    "$PYTHON" scripts/run_realistic_niah_v6_replacement_generation.py \
    --v6-config "$CONFIG" --replacement-policy "$REPLACEMENT_POLICY" \
    --replacement-stimuli "$REPLACEMENT_POOL/stimuli.jsonl" \
    --model "$MODEL" --seed-role confirmation --generation-root "$GEN_ROOT" \
    --output "$RESOLVED_CONFIRMATION" --cache-dir "$CACHE" \
    --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
    --confirmation-freeze "$V6_CONFIRMATION_FREEZE"
}

foundation_discovery_resolved() {
  if [[ ! -s "$DISCOVERY_REGISTRY" ]]; then
    echo "missing resolved discovery registry: $DISCOVERY_REGISTRY" >&2
    exit 2
  fi
  run_logged discovery_capture_formal_resolved \
    "$PYTHON" scripts/run_realistic_niah_v6.py capture \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role discovery \
    --cohort-registry "$DISCOVERY_REGISTRY" --output "$FORMAL_CAPTURE"
  run_logged discovery_representation_formal_resolved \
    "$PYTHON" scripts/run_realistic_niah_v6.py representation \
    --config "$CONFIG" --capture-index "$FORMAL_CAPTURE/capture_index.jsonl" \
    --output "$RUN_ROOT/representation/formal"
  run_logged discovery_attention_formal_resolved \
    "$PYTHON" scripts/run_realistic_niah_v6.py attention \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role discovery \
    --cohort-registry "$DISCOVERY_REGISTRY" \
    --output "$RUN_ROOT/attention/discovery_formal.csv"
  run_logged discovery_answer_query_formal_resolved \
    "$PYTHON" scripts/run_realistic_niah_v6.py attention-answer-query \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role discovery \
    --cohort-registry "$DISCOVERY_REGISTRY" \
    --output "$RUN_ROOT/attention/discovery_answer_query_formal.csv"
}

foundation_confirmation() {
  if [[ -z "${V6_CONFIRMATION_FREEZE:-}" ]]; then
    echo "V6_CONFIRMATION_FREEZE is required for confirmation" >&2
    exit 2
  fi
  if [[ ! -f "$GENERATIONS" ]]; then
    echo "missing accumulated generations: $GENERATIONS" >&2
    exit 2
  fi
  # Rebuild the indexes over discovery+confirmation. Existing discovery shards
  # are reused; only the fresh confirmation rows need model forwards.
  run_logged all_capture_formal \
    "$PYTHON" scripts/run_realistic_niah_v6.py capture \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role all \
    --output "$CONFIRMATION_FORMAL_CAPTURE"
  run_logged all_capture_all_sample \
    "$PYTHON" scripts/run_realistic_niah_v6.py capture \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role all --include-nonstrict \
    --output "$CONFIRMATION_ALL_CAPTURE"
  # Representation is intentionally not analysed cell-by-cell here.  The
  # Native-thinking primary path needs all four original all-sample captures
  # before it can freeze exact item_end common support and the full 300-row
  # answer_query_v3 panel.  queue_realistic_niah_v6_final_audit.sh runs that
  # single shared CPU analysis after both model queues complete.
  run_logged confirmation_attention_formal \
    "$PYTHON" scripts/run_realistic_niah_v6.py attention \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role confirmation \
    --output "$RUN_ROOT/attention/confirmation_formal.csv"
  run_logged confirmation_answer_query_formal \
    "$PYTHON" scripts/run_realistic_niah_v6.py attention-answer-query \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role confirmation \
    --output "$RUN_ROOT/attention/confirmation_answer_query_formal.csv"
}

foundation_confirmation_resolved() {
  if [[ -z "${V6_CONFIRMATION_FREEZE:-}" ]]; then
    echo "V6_CONFIRMATION_FREEZE is required for confirmation" >&2
    exit 2
  fi
  for path in "$DISCOVERY_REGISTRY" "$CONFIRMATION_REGISTRY"; do
    if [[ ! -s "$path" ]]; then
      echo "missing resolved formal cohort registry: $path" >&2
      exit 2
    fi
  done
  if [[ ! -f "$GENERATIONS" ]]; then
    echo "missing accumulated generations: $GENERATIONS" >&2
    exit 2
  fi
  run_logged all_capture_formal_resolved \
    "$PYTHON" scripts/run_realistic_niah_v6.py capture \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role all \
    --cohort-registry "$DISCOVERY_REGISTRY" \
    --additional-cohort-registry "$CONFIRMATION_REGISTRY" \
    --output "$CONFIRMATION_FORMAL_CAPTURE"
  run_logged all_capture_all_sample \
    "$PYTHON" scripts/run_realistic_niah_v6.py capture \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role all --include-nonstrict \
    --output "$CONFIRMATION_ALL_CAPTURE"
  # See foundation_confirmation above: confirmation representation is a
  # four-cell Native-aligned analysis, not a ten-site generic per-cell sweep.
  run_logged confirmation_attention_formal_resolved \
    "$PYTHON" scripts/run_realistic_niah_v6.py attention \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role confirmation \
    --cohort-registry "$CONFIRMATION_REGISTRY" \
    --output "$RUN_ROOT/attention/confirmation_formal.csv"
  run_logged confirmation_answer_query_formal_resolved \
    "$PYTHON" scripts/run_realistic_niah_v6.py attention-answer-query \
    --config "$CONFIG" --model "$MODEL" --cache-dir "$CACHE" \
    --generations "$GENERATIONS" --seed-role confirmation \
    --cohort-registry "$CONFIRMATION_REGISTRY" \
    --output "$RUN_ROOT/attention/confirmation_answer_query_formal.csv"
}

case "$PHASE" in
  preflight) preflight ;;
  discovery-generate) preflight; generate_discovery ;;
  discovery-foundation) foundation_discovery ;;
  discovery-supplement) supplement_discovery ;;
  discovery-foundation-resolved) foundation_discovery_resolved ;;
  confirmation-generate) generate_confirmation ;;
  confirmation-supplement) supplement_confirmation ;;
  confirmation-foundation) foundation_confirmation ;;
  confirmation-foundation-resolved) foundation_confirmation_resolved ;;
  suite-audit)
    run_logged suite_audit \
      "$PYTHON" scripts/run_realistic_niah_v6.py suite-audit \
      --config "$CONFIG" --output "$RUN_ROOT/suite_audit.json"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac

printf 'PASS\n' >"$RUN_ROOT/${PHASE}.COMPLETE"
