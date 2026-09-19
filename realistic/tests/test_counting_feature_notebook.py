import json
from pathlib import Path


def _notebook_source(path: str) -> str:
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_counting_feature_notebook_regenerates_when_config_missing() -> None:
    source = _notebook_source("notebooks/counting_feature_analysis.ipynb")

    assert "dataset-generation-main-v12" not in source
    assert "REPO_DIR = Path('/content/drive/MyDrive/Colab Notebooks/compression/dataset-generation-main-v14')" in source
    assert "REPO_DIR = Path.cwd()" in source
    assert "CONFIG_PATH = GENERATE_DATA_DIR / 'config.used.json'" in source
    assert "if not DATASET_PATH.exists() or not CONFIG_PATH.exists():" in source
    assert "Missing generation config" in source
    assert "--config {CONFIG_PATH}" in source
    assert "CONFIG_OVERRIDES = {" in source
    assert "COUNTING_ANALYSIS_CONFIG_PATH = REPO_DIR / 'configs' / 'counting_analysis.json'" in source
    assert "load_counting_feature_config_file(COUNTING_ANALYSIS_CONFIG_PATH)" in source
    assert "build_counting_feature_run_config({**CONFIG_FILE_OVERRIDES, **CONFIG_OVERRIDES})" in source
    assert "'LAYERS':" in source
    assert "'PROMPT_STYLE': 'vanilla'" in source
    assert "'COUNTING_NEEDLE_KIND':" in source
    assert "'MARKER_TEXT': '[dolphin]'" in source
    assert "'UID_TOKEN_LENGTH':" in source
    assert "'USE_KV_CACHE_FOR_NONTHINKG': True" in source
    assert "if not USE_THINKING and not USE_KV_CACHE_FOR_NONTHINKG:" in source
    assert "--no-use-kv-cache-for-nonthinkg" in source
    assert "'WORD_LEVEL_INSERTION':" in source
    assert "'FEATURE_CALC_POS': 'needle-last'" in source
    assert "'STEERING_POSITION_MODE': 'needle_span'" in source
    assert "'RUN_COUNTING_FEATURE_CALC':" in source
    assert "'RUN_NEEDLE_SENSITIVITY':" in source
    assert "'NUM_REMOVAL':" in source
    assert "'TARGET_SENSITIVITY_POSITION': 'last-token'" in source
    assert "'NEEDLE_SENSITIVITY_SEED':" in source
    assert "'MAX_SENSITIVITY_EXAMPLES':" in source
    assert "'FILTER_EXAMPLE':" in source
    assert "filter_example=FILTER_EXAMPLE" in source
    assert "if not RUN_COUNTING_FEATURE_CALC:" in source
    assert "Skipping Block 8 counting-feature calculation" in source
    assert "COUNTING_FEATURE_CALC_METHOD == 'ridge' and RUN_COUNTING_FEATURE_CALC" in source
    assert "from counting.feature_analysis import load_scored_rows, write_json" in source
    assert "'NUM_EXAMPLES':" in source
    assert "'NUM_MAX_NEEDLES':" in source
    assert "positive int samples 1..NUM_MAX_NEEDLES per example" in source
    assert "'COUNTING_PROBE_MODE':" in source
    assert "'COUNTING_PROBE_MODE': 'all_diagnostics'" in source
    assert "count-like diagnostics exclude argmax" in source
    assert "Probe Mode Guide" in source
    assert "occurrence_index_probe" in source
    assert "A prototype is the mean hidden vector" in source
    assert "--num-max-needles {NUM_MAX_NEEDLES}" in source
    assert "run_counting_probe_diagnostics" in source
    assert "run_needle_sensitivity_analysis" in source
    assert "timing.stage('needle_sensitivity_analysis')" in source
    assert "RUN_NEEDLE_SENSITIVITY" in source
    assert "COUNTING_PROBE_MODE != 'direct'" in source
    assert "PYTHONUNBUFFERED=1 PYTHONPATH=src python -u scripts/gen_responses.py" in source
    assert "Launching response-generation subprocess" in source
    assert "print_counting_feature_run_config(RUN_CONFIG)" in source
    assert "StageTimer" in source
    assert "TIMING_SUMMARY_JSON = TABLES_DIR / 'timing_summary.json'" in source
    assert "TIMING_SUMMARY_CSV = TABLES_DIR / 'timing_summary.csv'" in source
    assert "timing.stage('dataset_generation_or_restore')" in source
    assert "timing.stage('baseline_response_generation')" in source
    assert "timing.stage('cache_cleanup_and_archive')" in source
    assert "subprocess.run(cmd, shell=True, check=True)" in source
    assert "counting_feature_run_metadata.json" in source
    assert "Updated run metadata with generation config" in source
    assert "FACT_TEMPLATES_PATH" in source
    assert "--fact-templates-path {FACT_TEMPLATES_PATH}" in source
    assert "--counting-needle-kind {COUNTING_NEEDLE_KIND}" in source
    assert "--marker-text {shlex.quote(MARKER_TEXT)}" in source
    assert "--uid-token-length {UID_TOKEN_LENGTH}" in source
    assert "--word-level-insertion" in source


def test_counting_feature_notebook_has_steering_block() -> None:
    source = _notebook_source("notebooks/counting_feature_analysis.ipynb")

    assert "## 9. Steering counting feature" in source
    assert "MAX_NUM_STEERING_EXAMPLES" in source
    assert "STEERING_COEFF" in source
    assert "MAX_NEW_TOKEN_STEERING" in source
    assert "run_counting_steering_sweep" in source
    assert "steering_summary.csv" in source
    assert "FEATURE_CALC_POS" in source
    assert "counterfactual_count_layer_{layer}.pt" in source
    assert "dynamic_niah_v2_counterfactual.jsonl" in source
    assert "validate_counting_dataset_count" in source
    assert "COUNTERFACTUAL_SCRIPT_RUN_DIR" in source
    assert "--run-dir {COUNTERFACTUAL_SCRIPT_RUN_DIR}" in source
    assert "## 10. New test examples and steering eval" in source
    assert "STEERING_TEST_EVAL" in source
    assert "NUM_MAX_NEEDLES_STEERING_EVAL" in source
    assert "NUM_EXAMPLES_STEERING_EVAL" in source
    assert "STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION" in source
    assert "run_steering_test_eval" in source
    assert "steering_test_eval_summary.csv" in source
    assert "steering_test_eval_predictions.csv" in source


def test_counting_feature_notebook_has_feature_cache() -> None:
    source = _notebook_source("notebooks/counting_feature_analysis.ipynb")

    assert "REUSE_COUNTING_FEATURE_CACHE" in source
    assert "SAVE_COUNTING_FEATURE_CACHE" in source
    assert "COUNTING_FEATURE_CACHE_DIR" in source
    assert "validate_counting_feature_cache" in source
    assert "restore_counting_feature_cache" in source
    assert "save_counting_feature_cache" in source
    assert "Reusing existing counting-feature probe outputs" in source
