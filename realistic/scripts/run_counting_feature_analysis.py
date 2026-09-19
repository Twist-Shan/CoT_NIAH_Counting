#!/usr/bin/env python3
"""Run the full counting-feature analysis pipeline as a script.

This is a script version of notebooks/counting_feature_analysis.ipynb. It keeps
notebook-equivalent output folders and defaults to the 2026-06-16 broad steering
eval configuration requested for the next experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


def _json_override_path(value: str | None) -> dict:
    if not value:
        return {}
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Override JSON must contain an object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Override RUN_ROOT, e.g. /content for Colab or a local output root.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Override RESULTS_PATH for the final zip output directory.",
    )
    parser.add_argument(
        "--user-run-name",
        type=str,
        default=None,
        help="Override USER_RUN_NAME for reproducible/custom run names.",
    )
    parser.add_argument(
        "--override-json",
        type=str,
        default=None,
        help="Optional flat JSON object merged after the script's default overrides.",
    )
    parser.add_argument(
        "--ignore-reusable-dataset-cache",
        action="store_true",
        help="Force regeneration instead of copying data/niah-example cached datasets.",
    )
    return parser.parse_args()


ARGS = parse_args()
REPO_DIR = ARGS.repo_dir.resolve()
os.chdir(REPO_DIR)
if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))
print("Repo:", REPO_DIR)


# %% Notebook cell 4
from pathlib import Path
from datetime import datetime
import json
import shutil
import subprocess

from counting.analysis import (
    build_counting_run_name,
    build_counting_setting_name,
    cleanup_counting_archive_artifacts,
    format_counting_cleanup_summary,
    save_counting_dataset_cache,
    validate_counting_dataset_cache,
)
from counting.feature_analysis import (
    StageTimer,
    build_counting_feature_cache_config,
    build_counting_feature_run_config,
    load_counting_feature_config_file,
    print_counting_feature_run_config,
    restore_counting_feature_cache,
    save_counting_feature_cache,
    save_counting_feature_run_metadata,
    write_response_generation_checkpoint_archive,
    validate_counting_feature_cache,
)
from dataset_generation.run_utils import archive_directory

# Config resolution order:
# 1. Python defaults in counting.feature_analysis
# 2. configs/counting_analysis.json
# 3. CONFIG_OVERRIDES below
# Keep frequently adjusted values visible here so experiments can be changed
# without opening the JSON file. These notebook values win over the JSON file.
COUNTING_ANALYSIS_CONFIG_PATH = REPO_DIR / 'configs' / 'counting_analysis.json'
CONFIG_FILE_OVERRIDES = load_counting_feature_config_file(COUNTING_ANALYSIS_CONFIG_PATH)
CONFIG_OVERRIDES = {
    'NUM_EXAMPLES': 100,
    'TARGET_HAYSTACK_TOKENS': 2000,
    'LAYERS': [16, 24],
    'PROMPT_STYLE': 'vanilla',
    'COUNTING_NEEDLE_KIND': 'city_score',
    'MARKER_TEXT': '[dolphin]',
    'UID_TOKEN_LENGTH': 4,
    'WORD_LEVEL_INSERTION': False,
    'FEATURE_CALC_POS': 'needle-last',
    'STEERING_POSITION_MODE': 'needle_span',
    'RUN_COUNTING_FEATURE_CALC': True,
    'FILTER_EXAMPLE': True,
    'STEERING_COEFF': [0.5, 1, 2, 3, 4],
    'STEERING_TEST_EVAL': True,
    'NUM_MAX_NEEDLES_STEERING_EVAL': 10,
    'NUM_EXAMPLES_STEERING_EVAL': 10,
}
if ARGS.run_root is not None:
    CONFIG_OVERRIDES['RUN_ROOT'] = ARGS.run_root
if ARGS.results_path is not None:
    CONFIG_OVERRIDES['RESULTS_PATH'] = ARGS.results_path
if ARGS.user_run_name is not None:
    CONFIG_OVERRIDES['USER_RUN_NAME'] = ARGS.user_run_name
CONFIG_OVERRIDES.update(_json_override_path(ARGS.override_json))


RUN_CONFIG = build_counting_feature_run_config({**CONFIG_FILE_OVERRIDES, **CONFIG_OVERRIDES})
globals().update(RUN_CONFIG)
RUN_ROOT = Path(RUN_ROOT)
if RESULTS_PATH is not None:
    RESULTS_PATH = Path(RESULTS_PATH)

SETTING_NAME = build_counting_setting_name(
    model_name=MODEL_NAME,
    task_type=TASK_TYPE,
    prompt_style=PROMPT_STYLE,
    target_haystack_tokens=TARGET_HAYSTACK_TOKENS,
    num_examples=NUM_EXAMPLES,
    insertion_positions=INSERTION_POSITIONS,
    global_random_seed=GLOBAL_RANDOM_SEED,
    haystack_seed=HAYSTACK_SEED,
    needle_seed=NEEDLE_SEED,
    thinking_mode=USE_THINKING,
    fact_templates_path=FACT_TEMPLATES_PATH,
    randomize_needle_insertion=RANDOMIZE_NEEDLE_INSERTION,
    randomize_needle_seed=RANDOMIZE_NEEDLE_SEED,
    sentence_level_insertion=SENTENCE_LEVEL_INSERTION,
    word_level_insertion=WORD_LEVEL_INSERTION,
    counting_needle_kind=COUNTING_NEEDLE_KIND,
    uid_token_length=UID_TOKEN_LENGTH,
)
SAVED_DATASET_DIR = REPO_DIR / 'data' / 'niah-example' / SETTING_NAME
EXPECTED_CACHE_CONFIG = {
    'task_type': TASK_TYPE,
    'tokenizer_name': MODEL_NAME,
    'num_examples': NUM_EXAMPLES,
    'target_haystack_tokens': TARGET_HAYSTACK_TOKENS,
    'num_needles': NUM_NEEDLES,
    'insertion_positions': INSERTION_POSITIONS,
    'randomize_needle_insertion': RANDOMIZE_NEEDLE_INSERTION,
    'randomize_needle_seed': RANDOMIZE_NEEDLE_SEED,
    'sentence_level_insertion': SENTENCE_LEVEL_INSERTION,
    'word_level_insertion': WORD_LEVEL_INSERTION,
    'prompt_style': PROMPT_STYLE,
    'counting_needle_kind': COUNTING_NEEDLE_KIND,
    'marker_text': MARKER_TEXT,
    'uid_token_length': UID_TOKEN_LENGTH,
    'thinking_mode': USE_THINKING,
    'global_random_seed': GLOBAL_RANDOM_SEED,
    'haystack_seed': HAYSTACK_SEED,
    'needle_seed': NEEDLE_SEED,
    'fact_templates_path': FACT_TEMPLATES_PATH,
}

RUN_NAME = USER_RUN_NAME or build_counting_run_name(
    model_name=MODEL_NAME,
    task_type=TASK_TYPE,
    prompt_style=PROMPT_STYLE,
    target_haystack_tokens=TARGET_HAYSTACK_TOKENS,
    insertion_positions=INSERTION_POSITIONS,
    start_time=datetime.now(),
    randomize_needle_insertion=RANDOMIZE_NEEDLE_INSERTION,
    randomize_needle_seed=RANDOMIZE_NEEDLE_SEED,
    sentence_level_insertion=SENTENCE_LEVEL_INSERTION,
    word_level_insertion=WORD_LEVEL_INSERTION,
    fact_templates_path=FACT_TEMPLATES_PATH,
    counting_needle_kind=COUNTING_NEEDLE_KIND,
    uid_token_length=UID_TOKEN_LENGTH,
)
RUN_DIR = RUN_ROOT / RUN_NAME
GENERATE_DATA_DIR = RUN_DIR / 'generate_data'
TABLES_DIR = RUN_DIR / 'tables'
TENSORS_DIR = RUN_DIR / 'tensors'
FIGURES_DIR = RUN_DIR / 'figures'
FEATURE_TENSORS_DIR = TENSORS_DIR / 'counting_features'
FEATURE_TABLES_DIR = TABLES_DIR / 'counting_features'
FEATURE_FIGURES_DIR = FIGURES_DIR / 'counting_features'
TIMING_SUMMARY_JSON = TABLES_DIR / 'timing_summary.json'
TIMING_SUMMARY_CSV = TABLES_DIR / 'timing_summary.csv'
for path in [RUN_DIR, GENERATE_DATA_DIR, TABLES_DIR, TENSORS_DIR, FIGURES_DIR, FEATURE_TENSORS_DIR, FEATURE_TABLES_DIR, FEATURE_FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)
timing = StageTimer(json_path=TIMING_SUMMARY_JSON, csv_path=TIMING_SUMMARY_CSV)
timing.mark_skipped('config_resolution_and_path_setup', reason='completed before timing output paths were initialized')
if RESULTS_PATH is None:
    RESULTS_PATH = Path('results/counting_features') / RUN_NAME
FEATURE_CACHE_ROOT = Path(FEATURE_CACHE_ROOT)
if not FEATURE_CACHE_ROOT.is_absolute():
    FEATURE_CACHE_ROOT = REPO_DIR / FEATURE_CACHE_ROOT
feature_cache_leaf = 'counting_features_counterfactual' if COUNTING_FEATURE_CALC_METHOD == 'counterfactual' else 'counting_features'
COUNTING_FEATURE_CACHE_DIR = FEATURE_CACHE_ROOT / SETTING_NAME / feature_cache_leaf
COUNTING_FEATURE_CACHE_CONFIG = build_counting_feature_cache_config(
    RUN_CONFIG,
    setting_name=SETTING_NAME,
)
FEATURE_CACHE_RESTORED = False
if REUSE_COUNTING_FEATURE_CACHE and COUNTING_FEATURE_CACHE_DIR.exists():
    try:
        validated_feature_cache = validate_counting_feature_cache(
            COUNTING_FEATURE_CACHE_DIR,
            COUNTING_FEATURE_CACHE_CONFIG,
            layers=LAYERS,
            run_classification=RUN_CLASSIFICATION,
        )
        restored_feature_cache = restore_counting_feature_cache(
            COUNTING_FEATURE_CACHE_DIR,
            feature_tensors_dir=FEATURE_TENSORS_DIR,
            feature_tables_dir=FEATURE_TABLES_DIR,
            feature_figures_dir=FEATURE_FIGURES_DIR,
        )
        FEATURE_CACHE_RESTORED = True
        print('Validated and restored counting-feature cache:', json.dumps(validated_feature_cache, indent=2))
        print('Restored counting-feature artifacts:', json.dumps(restored_feature_cache, indent=2))
    except Exception as exc:
        print('Existing counting-feature cache did not validate; recomputing. Reason:', repr(exc))
elif REUSE_COUNTING_FEATURE_CACHE:
    print('No counting-feature cache found for current setting:', COUNTING_FEATURE_CACHE_DIR)

# Keep Hugging Face model downloads on Drive in Colab when available.
# Outside Colab, respect existing HF_HOME/TRANSFORMERS_CACHE or the HF defaults.
if 'HF_HOME' in os.environ:
    print('Using existing HF_HOME:', os.environ['HF_HOME'])
else:
    colab_hf_cache_dir = Path('/content/drive/MyDrive/Colab Notebooks/huggingface_models')
    if colab_hf_cache_dir.parent.exists():
        colab_hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(colab_hf_cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(colab_hf_cache_dir)
        print('Using Colab HF cache:', colab_hf_cache_dir)
    else:
        print('HF_HOME is not set; using Hugging Face default cache directory.')
DATASET_PATH = GENERATE_DATA_DIR / 'dynamic_niah_v2.jsonl'
CONFIG_PATH = GENERATE_DATA_DIR / 'config.used.json'
RUN_METADATA_PATH = TABLES_DIR / 'counting_feature_run_metadata.json'
PREDICTIONS_PATH = TABLES_DIR / 'predictions.jsonl'
METRICS_PATH = TABLES_DIR / 'metrics.json'
RESPONSE_GENERATION_CHECKPOINT_ZIP = RUN_DIR / f'{RUN_NAME}_response_generation_checkpoint.zip'
CACHE_INFO = None
if ARGS.ignore_reusable_dataset_cache:
    print('Ignoring reusable dataset cache by request:', SAVED_DATASET_DIR)
elif SAVED_DATASET_DIR.exists():
    try:
        CACHE_INFO = validate_counting_dataset_cache(SAVED_DATASET_DIR, EXPECTED_CACHE_CONFIG)
        print('Validated cached dataset:', SAVED_DATASET_DIR)
        shutil.copyfile(CACHE_INFO['dataset_path'], DATASET_PATH)
        shutil.copyfile(CACHE_INFO['config_path'], CONFIG_PATH)
        if CACHE_INFO.get('predictions_path') is not None:
            shutil.copyfile(CACHE_INFO['predictions_path'], PREDICTIONS_PATH)
            print('Validated cached scored predictions; Block 5 can be skipped.')
        if CACHE_INFO.get('metrics_path') is not None:
            shutil.copyfile(CACHE_INFO['metrics_path'], METRICS_PATH)
    except Exception as exc:
        CACHE_INFO = None
        print('Existing cached dataset did not validate; regenerating. Reason:', repr(exc))

save_counting_feature_run_metadata(
    RUN_METADATA_PATH,
    resolved_config=RUN_CONFIG,
    generation_config_path=CONFIG_PATH,
    extra={
        'run_name': RUN_NAME,
        'setting_name': SETTING_NAME,
        'counting_analysis_config_path': str(COUNTING_ANALYSIS_CONFIG_PATH),
        'counting_analysis_config_values': CONFIG_FILE_OVERRIDES,
        'notebook_config_overrides': CONFIG_OVERRIDES,
        'run_dir': str(RUN_DIR),
        'dataset_path': str(DATASET_PATH),
        'predictions_path': str(PREDICTIONS_PATH),
        'metrics_path': str(METRICS_PATH),
        'saved_dataset_dir': str(SAVED_DATASET_DIR),
    },
)
print('Counting analysis config file:', COUNTING_ANALYSIS_CONFIG_PATH)
print('Config values loaded from file:', json.dumps(CONFIG_FILE_OVERRIDES, indent=2, sort_keys=True))
print('Notebook config overrides:', json.dumps(CONFIG_OVERRIDES, indent=2, sort_keys=True))
print('Resolved counting-feature run config:')
print_counting_feature_run_config(RUN_CONFIG)
print('Run metadata path:', RUN_METADATA_PATH)
print('Run name:', RUN_NAME)
print('Setting name:', SETTING_NAME)
print('Reusable dataset cache:', SAVED_DATASET_DIR)
print('Runtime run dir:', RUN_DIR)
print('Counting-feature cache dir (optional):', COUNTING_FEATURE_CACHE_DIR)
print('Counting-feature cache restored:', FEATURE_CACHE_RESTORED)
print('Dataset path:', DATASET_PATH)
print('Config path:', CONFIG_PATH)
print('Predictions path:', PREDICTIONS_PATH)
print('Response-generation checkpoint zip:', RESPONSE_GENERATION_CHECKPOINT_ZIP)
print('Layers:', LAYERS)
print('Target count type:', TARGET_COUNT_TYPE)
print('Classification enabled:', RUN_CLASSIFICATION)
print('Randomized insertion:', RANDOMIZE_NEEDLE_INSERTION)
print('Sentence-level insertion:', SENTENCE_LEVEL_INSERTION)
print('Word-level insertion:', WORD_LEVEL_INSERTION)
print('Fact templates path:', FACT_TEMPLATES_PATH)
print('Counting needle kind:', COUNTING_NEEDLE_KIND)
print('Marker text:', MARKER_TEXT)
print('Counting feature calc method:', COUNTING_FEATURE_CALC_METHOD)
print('Feature calc position:', FEATURE_CALC_POS)
if COUNTING_FEATURE_CALC_METHOD in {'ridge', 'contrastive-success'}:
    print('WARNING: COUNTERFACTUAL_REMOVED_NEEDLE_INDEX is ignored unless COUNTING_FEATURE_CALC_METHOD=\'counterfactual\'.')
print('Steering position mode:', STEERING_POSITION_MODE)


# %% Notebook cell 8
with timing.stage('dataset_generation_or_restore'):
    if not DATASET_PATH.exists() or not CONFIG_PATH.exists():
        positions = ' '.join('null' if p is None else str(p) for p in INSERTION_POSITIONS)
        cmd = (
            f"PYTHONPATH=src python scripts/generate_dynamic_niah_v2.py "
            f"--config configs/niah_dynamic.json "
            f"--task-type {TASK_TYPE} "
            f"--tokenizer {TOKENIZER_NAME} "
            f"--num-examples {NUM_EXAMPLES} "
            f"--target-haystack-tokens {TARGET_HAYSTACK_TOKENS} "
            f"--num-needles {NUM_NEEDLES} "
            f"--positions {positions} "
            f"--prompt-style {PROMPT_STYLE} "
            f"--counting-needle-kind {COUNTING_NEEDLE_KIND} "
            f"--marker-text {shlex.quote(MARKER_TEXT)} "
            f"--uid-token-length {UID_TOKEN_LENGTH} "
            f"--global-random-seed {GLOBAL_RANDOM_SEED} "
            f"--haystack-seed {HAYSTACK_SEED} "
            f"--needle-seed {NEEDLE_SEED} "
            f"--fact-templates-path {FACT_TEMPLATES_PATH} "
            f"--run-dir {RUN_DIR} "
            f"--output-dir {GENERATE_DATA_DIR} "
            f"--data-save-path {DATASET_PATH}"
        )
        if RANDOMIZE_NEEDLE_INSERTION:
            cmd += f' --randomize-needle-insertion --randomize-needle-seed {RANDOMIZE_NEEDLE_SEED}'
        if SENTENCE_LEVEL_INSERTION:
            cmd += ' --sentence-level-insertion'
        if WORD_LEVEL_INSERTION:
            cmd += ' --word-level-insertion'
        if USE_THINKING:
            cmd += ' --thinking-mode'
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    else:
        print('Using existing or cached dataset:', DATASET_PATH)
        print('Using existing config:', CONFIG_PATH)
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f'Dataset generation did not write expected dataset: {DATASET_PATH}')
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f'Dataset generation did not write expected config: {CONFIG_PATH}')

    save_counting_feature_run_metadata(
        RUN_METADATA_PATH,
        resolved_config=RUN_CONFIG,
        generation_config_path=CONFIG_PATH,
        extra={
            'run_name': RUN_NAME,
            'setting_name': SETTING_NAME,
            'run_dir': str(RUN_DIR),
            'dataset_path': str(DATASET_PATH),
            'predictions_path': str(PREDICTIONS_PATH),
            'metrics_path': str(METRICS_PATH),
            'saved_dataset_dir': str(SAVED_DATASET_DIR),
        },
    )
    print('Updated run metadata with generation config:', RUN_METADATA_PATH)

# %% Notebook cell 10
with timing.stage('baseline_response_generation'):
    if RUN_GENERATION_EVAL and not PREDICTIONS_PATH.exists():
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f'Missing generation config: {CONFIG_PATH}. Re-run the dataset generation cell; '
                'it now regenerates whenever either the dataset JSONL or config.used.json is missing.'
            )
        cmd = (
            f"PYTHONUNBUFFERED=1 PYTHONPATH=src python -u scripts/gen_responses.py "
            f"--config {CONFIG_PATH} "
            f"--tokenizer {TOKENIZER_NAME} "
            f"--data-save-path {DATASET_PATH} "
            f"--output-pred-jsonl {PREDICTIONS_PATH} "
            f"--output-metrics-json {METRICS_PATH} "
        )
        if not USE_THINKING and not USE_KV_CACHE_FOR_NONTHINKG:
            cmd += ' --no-use-kv-cache-for-nonthinkg'
        print(cmd)
        print('Launching response-generation subprocess; model loading may take a few minutes before tqdm starts.', flush=True)
        subprocess.run(cmd, shell=True, check=True)
    elif PREDICTIONS_PATH.exists():
        print('Skipping response generation/eval because scored predictions are available:', PREDICTIONS_PATH)
    else:
        print('Skipping response generation/eval by RUN_GENERATION_EVAL=False. Existing predictions:', PREDICTIONS_PATH.exists())

with timing.stage('response_generation_checkpoint_archive'):
    if PREDICTIONS_PATH.exists() and METRICS_PATH.exists():
        timing.save_json(TIMING_SUMMARY_JSON)
        timing.save_csv(TIMING_SUMMARY_CSV)
        checkpoint_path = write_response_generation_checkpoint_archive(
            run_dir=RUN_DIR,
            archive_path=RESPONSE_GENERATION_CHECKPOINT_ZIP,
            dataset_path=DATASET_PATH,
            config_path=CONFIG_PATH,
            predictions_path=PREDICTIONS_PATH,
            metrics_path=METRICS_PATH,
            metadata_path=RUN_METADATA_PATH,
            timing_json_path=TIMING_SUMMARY_JSON,
            timing_csv_path=TIMING_SUMMARY_CSV,
        )
        print('Response-generation checkpoint archive written to:', checkpoint_path)
    else:
        print(
            'Skipping response-generation checkpoint archive because predictions/metrics are incomplete:',
            PREDICTIONS_PATH.exists(),
            METRICS_PATH.exists(),
        )

# %% Notebook cell 12
with timing.stage('target_filter_tokenize_and_model_load'):
    import json
    import torch

    from counting.feature_analysis import (
        build_needle_token_mask,
        build_target_count_matrix,
        filter_successful_rows,
        load_jsonl,
        load_scored_rows,
        save_target_artifacts,
        tokenize_counting_examples,
        train_test_split_indices,
        validate_counting_dataset_count,
        write_json,
    )
    from dataset_generation.dynamic_niah_v2 import DynamicNiahV2Config
    from single_example import load_model_and_tokenizer

    rows = load_jsonl(DATASET_PATH)
    original_count_summary = validate_counting_dataset_count(
        rows,
        expected_count=sum(pos is not None for pos in INSERTION_POSITIONS),
        label='original counting-feature dataset',
    )
    print('Original count summary:', json.dumps(original_count_summary, indent=2))
    scored_rows = load_scored_rows(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else None
    successful_rows, filter_summary = filter_successful_rows(rows, scored_rows, filter_example=FILTER_EXAMPLE)
    print('Filter summary:', json.dumps(filter_summary, indent=2))
    if not successful_rows:
        raise RuntimeError('No selected examples available for counting-feature calculation.')

    cfg = DynamicNiahV2Config.from_config_file(GENERATE_DATA_DIR / 'config.used.json')
    model, tokenizer = load_model_and_tokenizer(cfg, model_name=MODEL_NAME)
    if getattr(cfg, 'randomize_needle_insertion', False):
        expected_inserted_needles = [
            sum(1 for needle in row.get('needles', []) if needle.get('is_inserted', True))
            for row in successful_rows
        ]
        print('Expected inserted needle spans per row:', expected_inserted_needles[:10])
    else:
        expected_inserted_needles = sum(pos is not None for pos in cfg.insertion_positions)
        print('Expected inserted needle spans:', expected_inserted_needles)
    examples, length_stats = tokenize_counting_examples(
        successful_rows,
        tokenizer,
        thinking_mode=USE_THINKING,
        expected_num_needles=expected_inserted_needles,
    )
    print('Tokenized sequence length stats:', json.dumps(length_stats, indent=2))
    split = train_test_split_indices(len(examples), test_fraction=TEST_FRACTION, seed=SPLIT_SEED)
    print('Train examples:', split.train_indices)
    print('Test examples:', split.test_indices)

    target = build_target_count_matrix(examples, target_count_type=TARGET_COUNT_TYPE)
    needle_mask = build_needle_token_mask(examples, max_sequence_length=target.shape[1], matching_only=True)
    save_target_artifacts(target, examples, FEATURE_TENSORS_DIR, target_count_type=TARGET_COUNT_TYPE)
    torch.save(needle_mask, FEATURE_TENSORS_DIR / 'matching_needle_token_mask.pt')
    write_json(
        FEATURE_TABLES_DIR / 'split_and_filter_summary.json',
        {
            'filter_summary': filter_summary,
            'length_stats': length_stats,
            'split': split.__dict__,
            'filter_example': FILTER_EXAMPLE,
            'target_count_type': TARGET_COUNT_TYPE,
            'max_train_tokens_per_layer': MAX_TRAIN_TOKENS_PER_LAYER,
        },
    )
    print('Target shape:', tuple(target.shape))

# %% Notebook cell 14
with timing.stage('ridge_hidden_state_extraction'):
    from counting.feature_analysis import extract_hidden_states_by_layer

    hidden_paths = {}
    if COUNTING_FEATURE_CALC_METHOD == 'ridge':
        missing_layers = [layer for layer in LAYERS if not (FEATURE_TENSORS_DIR / f'hidden_layer_{layer}.pt').exists()]
        if missing_layers:
            hidden_paths = extract_hidden_states_by_layer(
                model,
                examples,
                missing_layers,
                FEATURE_TENSORS_DIR,
                dtype=HIDDEN_STATE_DTYPE,
            )
        else:
            print('All requested hidden-state files already exist.')
        hidden_paths = {layer: FEATURE_TENSORS_DIR / f'hidden_layer_{layer}.pt' for layer in LAYERS}
    else:
        print(
            'Skipping ridge hidden-state extraction because '
            f'COUNTING_FEATURE_CALC_METHOD={COUNTING_FEATURE_CALC_METHOD!r}.'
        )
    print(hidden_paths)

# %% Notebook cell 16
with timing.stage('feature_vector_calculation'):
    if not RUN_COUNTING_FEATURE_CALC:
        print('Skipping Block 8 counting-feature calculation because RUN_COUNTING_FEATURE_CALC=False. Existing feature-vector files will be required for steering/eval blocks.')
    else:
        import pandas as pd

        from counting.feature_analysis import (
            counterfactual_insertion_positions,
            evaluate_classification_probe,
            evaluate_ridge_probe,
            extract_hidden_states_by_layer,
            filter_successful_rows,
            fit_classification_probe,
            fit_contrastive_success_direction,
            fit_counterfactual_count_direction,
            fit_ridge_probe,
            load_jsonl,
            load_scored_rows,
            plot_probe_2d_projection,
            plot_ridge_line_fit,
            prepare_contrastive_examples_for_position,
            prepare_feature_examples_for_position,
            release_torch_memory,
            save_classification_probe,
            save_contrastive_success_direction,
            save_counterfactual_count_direction,
            save_ridge_probe,
            select_contrastive_success_examples,
            tokenize_counting_examples,
            tokenize_prompt_rows,
            validate_counting_dataset_count,
            write_json,
        )

        if COUNTING_FEATURE_CALC_METHOD not in {'ridge', 'contrastive-success', 'counterfactual'}:
            raise ValueError(
                "COUNTING_FEATURE_CALC_METHOD must be 'ridge', 'contrastive-success', or 'counterfactual', "
                f"got {COUNTING_FEATURE_CALC_METHOD!r}"
            )
        if FEATURE_CALC_POS not in {'last', 'needle-last'}:
            raise ValueError(
                "FEATURE_CALC_POS must be 'last' or 'needle-last', "
                f"got {FEATURE_CALC_POS!r}"
            )

        def row_id_for_notebook(row, fallback_index):
            return str(row.get('id', fallback_index))

        def dataset_indices_for_selected_rows(all_rows, selected_rows):
            lookup = {row_id_for_notebook(row, idx): idx for idx, row in enumerate(all_rows)}
            return [int(lookup.get(row_id_for_notebook(row, local_idx), local_idx)) for local_idx, row in enumerate(selected_rows)]

        def retag_examples_with_dataset_indices(candidate_examples, dataset_indices):
            return [
                type(example)(
                    example_index=int(dataset_index),
                    row_id=example.row_id,
                    input_ids=example.input_ids,
                    needle_segments=example.needle_segments,
                    matching_needle_ids=example.matching_needle_ids,
                )
                for example, dataset_index in zip(candidate_examples, dataset_indices)
            ]

        def tokenized_feature_examples(selected_rows, selected_dataset_indices, *, position_mode, config_obj):
            if position_mode == 'last':
                return tokenize_prompt_rows(
                    selected_rows,
                    tokenizer,
                    thinking_mode=USE_THINKING,
                    dataset_indices=selected_dataset_indices,
                )
            if getattr(config_obj, 'randomize_needle_insertion', False):
                expected_inserted = [
                    sum(1 for needle in row.get('needles', []) if needle.get('is_inserted', True))
                    for row in selected_rows
                ]
            else:
                expected_inserted = sum(pos is not None for pos in config_obj.insertion_positions)
            candidate_examples, feature_length_stats = tokenize_counting_examples(
                selected_rows,
                tokenizer,
                thinking_mode=USE_THINKING,
                expected_num_needles=expected_inserted,
            )
            return retag_examples_with_dataset_indices(candidate_examples, selected_dataset_indices), feature_length_stats

        def unique_gold_count_or_none(selected_rows):
            counts = sorted({int(row.get('gold_answer', {}).get('count')) for row in selected_rows if row.get('gold_answer', {}).get('count') is not None})
            return counts[0] if len(counts) == 1 else None

        if COUNTING_FEATURE_CALC_METHOD == 'ridge':
            PROBE_SUMMARY_PATH = FEATURE_TABLES_DIR / 'probe_summary.csv'
            required_probe_outputs = [PROBE_SUMMARY_PATH]
            for layer in LAYERS:
                required_probe_outputs.extend([
                    FEATURE_TENSORS_DIR / f'ridge_probe_layer_{layer}.pt',
                    FEATURE_TABLES_DIR / f'ridge_layer_{layer}_eval.json',
                    FEATURE_FIGURES_DIR / f'ridge_layer_{layer}_train_line.png',
                    FEATURE_FIGURES_DIR / f'ridge_layer_{layer}_test_line.png',
                    FEATURE_FIGURES_DIR / f'ridge_layer_{layer}_train_2d.png',
                    FEATURE_FIGURES_DIR / f'ridge_layer_{layer}_test_2d.png',
                ])
                if RUN_CLASSIFICATION:
                    required_probe_outputs.extend([
                        FEATURE_TENSORS_DIR / f'classification_probe_layer_{layer}.pt',
                        FEATURE_TABLES_DIR / f'classification_layer_{layer}_eval.json',
                        FEATURE_FIGURES_DIR / f'classification_layer_{layer}_train_2d.png',
                        FEATURE_FIGURES_DIR / f'classification_layer_{layer}_test_2d.png',
                    ])

            if all(path.exists() for path in required_probe_outputs):
                print('Reusing existing counting-feature probe outputs; skipping Block 8 fitting/plotting.')
                summary_df = pd.read_csv(PROBE_SUMMARY_PATH)
            else:
                summary_rows = []
                for layer in LAYERS:
                    print(f'=== Layer {layer} ===')
                    hidden = torch.load(FEATURE_TENSORS_DIR / f'hidden_layer_{layer}.pt', map_location='cpu')
                    ridge = fit_ridge_probe(
                        hidden,
                        target,
                        train_example_indices=split.train_indices,
                        needle_mask=needle_mask,
                        alpha=RIDGE_ALPHA,
                        max_train_tokens=MAX_TRAIN_TOKENS_PER_LAYER,
                        seed=GLOBAL_RANDOM_SEED + layer,
                        standardize=STANDARDIZE_FEATURES,
                    )
                    ridge_train = evaluate_ridge_probe(ridge, hidden, target, example_indices=split.train_indices, max_tokens=MAX_EVAL_TOKENS_PER_LAYER)
                    ridge_test = evaluate_ridge_probe(ridge, hidden, target, example_indices=split.test_indices, max_tokens=MAX_EVAL_TOKENS_PER_LAYER)
                    save_ridge_probe(ridge, FEATURE_TENSORS_DIR, layer=layer)
                    write_json(FEATURE_TABLES_DIR / f'ridge_layer_{layer}_eval.json', {'train': ridge_train, 'test': ridge_test})
                    plot_ridge_line_fit(ridge, hidden, target, FEATURE_FIGURES_DIR / f'ridge_layer_{layer}_train_line.png', example_indices=split.train_indices, max_points=MAX_PLOT_POINTS, title=f'Layer {layer} ridge train')
                    plot_ridge_line_fit(ridge, hidden, target, FEATURE_FIGURES_DIR / f'ridge_layer_{layer}_test_line.png', example_indices=split.test_indices, max_points=MAX_PLOT_POINTS, title=f'Layer {layer} ridge test')
                    plot_probe_2d_projection(ridge, hidden, target, FEATURE_FIGURES_DIR / f'ridge_layer_{layer}_train_2d.png', example_indices=split.train_indices, max_points=MAX_PLOT_POINTS, title=f'Layer {layer} ridge train 2D')
                    plot_probe_2d_projection(ridge, hidden, target, FEATURE_FIGURES_DIR / f'ridge_layer_{layer}_test_2d.png', example_indices=split.test_indices, max_points=MAX_PLOT_POINTS, title=f'Layer {layer} ridge test 2D')
                    summary_rows.append({'layer': layer, 'probe': 'ridge', 'split': 'train', **ridge_train})
                    summary_rows.append({'layer': layer, 'probe': 'ridge', 'split': 'test', **ridge_test})

                    if RUN_CLASSIFICATION:
                        clf = fit_classification_probe(
                            hidden,
                            target,
                            train_example_indices=split.train_indices,
                            needle_mask=needle_mask,
                            max_train_tokens=MAX_TRAIN_TOKENS_PER_LAYER,
                            seed=GLOBAL_RANDOM_SEED + 10_000 + layer,
                            standardize=STANDARDIZE_FEATURES,
                            lr=CLASSIFIER_LR,
                            epochs=CLASSIFIER_EPOCHS,
                            l2_penalty=CLASSIFIER_L2,
                        )
                        clf_train = evaluate_classification_probe(clf, hidden, target, example_indices=split.train_indices, max_tokens=MAX_EVAL_TOKENS_PER_LAYER)
                        clf_test = evaluate_classification_probe(clf, hidden, target, example_indices=split.test_indices, max_tokens=MAX_EVAL_TOKENS_PER_LAYER)
                        save_classification_probe(clf, FEATURE_TENSORS_DIR, layer=layer)
                        write_json(FEATURE_TABLES_DIR / f'classification_layer_{layer}_eval.json', {'train': clf_train, 'test': clf_test})
                        plot_probe_2d_projection(clf, hidden, target, FEATURE_FIGURES_DIR / f'classification_layer_{layer}_train_2d.png', example_indices=split.train_indices, classifier=clf, max_points=MAX_PLOT_POINTS, title=f'Layer {layer} classification train 2D')
                        plot_probe_2d_projection(clf, hidden, target, FEATURE_FIGURES_DIR / f'classification_layer_{layer}_test_2d.png', example_indices=split.test_indices, classifier=clf, max_points=MAX_PLOT_POINTS, title=f'Layer {layer} classification test 2D')
                        summary_rows.append({'layer': layer, 'probe': 'classification', 'split': 'train', **{k: v for k, v in clf_train.items() if k != 'confusion_matrix'}})
                        summary_rows.append({'layer': layer, 'probe': 'classification', 'split': 'test', **{k: v for k, v in clf_test.items() if k != 'confusion_matrix'}})
                    else:
                        print('Classification skipped because interpolation targets are fractional.')
                    del hidden
                    release_torch_memory(collect_garbage=True)

                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(PROBE_SUMMARY_PATH, index=False)
                summary_df
        elif COUNTING_FEATURE_CALC_METHOD == 'contrastive-success':
            if scored_rows is None:
                raise FileNotFoundError(
                    f'Contrastive success direction requires scored predictions: {PREDICTIONS_PATH}'
                )
            CONTRASTIVE_ROOT_TENSORS_DIR = FEATURE_TENSORS_DIR / 'contrastive_success'
            CONTRASTIVE_ROOT_TABLES_DIR = FEATURE_TABLES_DIR / 'contrastive_success'
            CONTRASTIVE_TENSORS_DIR = CONTRASTIVE_ROOT_TENSORS_DIR / FEATURE_CALC_POS
            CONTRASTIVE_TABLES_DIR = CONTRASTIVE_ROOT_TABLES_DIR / FEATURE_CALC_POS
            CONTRASTIVE_TENSORS_DIR.mkdir(parents=True, exist_ok=True)
            CONTRASTIVE_TABLES_DIR.mkdir(parents=True, exist_ok=True)
            CONTRASTIVE_SUMMARY_PATH = CONTRASTIVE_TABLES_DIR / 'contrastive_success_summary.csv'
            CONTRASTIVE_METADATA_PATH = CONTRASTIVE_TABLES_DIR / 'contrastive_success_metadata.json'
            required_contrastive_outputs = [
                CONTRASTIVE_SUMMARY_PATH,
                CONTRASTIVE_METADATA_PATH,
                *(CONTRASTIVE_TENSORS_DIR / f'contrastive_success_layer_{layer}.pt' for layer in LAYERS),
            ]
            if all(path.exists() for path in required_contrastive_outputs):
                print('Reusing existing contrastive success directions; skipping Block 8 calculation.')
                summary_df = pd.read_csv(CONTRASTIVE_SUMMARY_PATH)
            else:
                contrastive_selection = select_contrastive_success_examples(rows, scored_rows)
                print('Contrastive selection summary:', json.dumps(contrastive_selection.summary, indent=2))
                for warning_message in contrastive_selection.summary.get('warnings', []):
                    print('WARNING:', warning_message)
                if contrastive_selection.summary['num_selected_per_group'] == 0:
                    raise RuntimeError('No balanced successful/unsuccessful examples available for contrastive direction.')
                candidate_examples, contrastive_length_stats = tokenized_feature_examples(
                    contrastive_selection.selected_rows,
                    contrastive_selection.selected_dataset_indices,
                    position_mode=FEATURE_CALC_POS,
                    config_obj=cfg,
                )
                prepared_contrastive = prepare_contrastive_examples_for_position(
                    rows=contrastive_selection.selected_rows,
                    examples=candidate_examples,
                    labels=contrastive_selection.labels,
                    selected_dataset_indices=contrastive_selection.selected_dataset_indices,
                    selected_row_ids=contrastive_selection.selected_row_ids,
                    position_mode=FEATURE_CALC_POS,
                )
                print('Feature position:', FEATURE_CALC_POS)
                print('Prepared contrastive summary:', json.dumps(prepared_contrastive.summary, indent=2))
                if prepared_contrastive.summary['num_skipped']:
                    print('WARNING: skipped contrastive examples:', prepared_contrastive.summary['num_skipped'])
                if prepared_contrastive.summary['num_successful_after_skip'] < 10 or prepared_contrastive.summary['num_unsuccessful_after_skip'] < 10:
                    print(
                        'WARNING: contrastive direction has fewer than 10 examples in at least one group after position filtering.'
                    )
                print('Contrastive tokenized sequence length stats:', json.dumps(contrastive_length_stats, indent=2))
                missing_layers = [
                    layer for layer in LAYERS
                    if not (CONTRASTIVE_TENSORS_DIR / f'hidden_layer_{layer}.pt').exists()
                ]
                if missing_layers:
                    extract_hidden_states_by_layer(
                        model,
                        prepared_contrastive.examples,
                        missing_layers,
                        CONTRASTIVE_TENSORS_DIR,
                        dtype=HIDDEN_STATE_DTYPE,
                    )
                else:
                    print('All requested contrastive hidden-state files already exist.')
                summary_rows = []
                for layer in LAYERS:
                    hidden = torch.load(CONTRASTIVE_TENSORS_DIR / f'hidden_layer_{layer}.pt', map_location='cpu')
                    result = fit_contrastive_success_direction(
                        hidden,
                        prepared_contrastive.examples,
                        prepared_contrastive.labels,
                        layer=layer,
                        position_mode=FEATURE_CALC_POS,
                    )
                    save_contrastive_success_direction(
                        result,
                        CONTRASTIVE_TENSORS_DIR,
                        selected_dataset_indices=prepared_contrastive.selected_dataset_indices,
                        selected_row_ids=prepared_contrastive.selected_row_ids,
                        labels=prepared_contrastive.labels,
                        position_indices=prepared_contrastive.position_indices,
                    )
                    summary_rows.append(result.metrics)
                    del hidden
                    release_torch_memory(collect_garbage=True)
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(CONTRASTIVE_SUMMARY_PATH, index=False)
                write_json(
                    CONTRASTIVE_METADATA_PATH,
                    {
                        'method': 'contrastive-success',
                        'position': FEATURE_CALC_POS,
                        'initial_selection_summary': contrastive_selection.summary,
                        'prepared_selection_summary': prepared_contrastive.summary,
                        'length_stats': contrastive_length_stats,
                        'layers': LAYERS,
                        'summary_path': str(CONTRASTIVE_SUMMARY_PATH),
                        'tensor_dir': str(CONTRASTIVE_TENSORS_DIR),
                    },
                )
            summary_df
        else:
            if scored_rows is None:
                raise FileNotFoundError(
                    f'Counterfactual count direction requires scored original predictions: {PREDICTIONS_PATH}'
                )
            counterfactual_positions = counterfactual_insertion_positions(
                INSERTION_POSITIONS,
                removed_needle_index=COUNTERFACTUAL_REMOVED_NEEDLE_INDEX,
            )
            COUNTERFACTUAL_GENERATE_DATA_DIR = GENERATE_DATA_DIR / 'counterfactual'
            COUNTERFACTUAL_GENERATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
            COUNTERFACTUAL_SCRIPT_RUN_DIR = COUNTERFACTUAL_GENERATE_DATA_DIR / 'script_run'
            COUNTERFACTUAL_DATASET_PATH = GENERATE_DATA_DIR / 'dynamic_niah_v2_counterfactual.jsonl'
            COUNTERFACTUAL_CONFIG_PATH = COUNTERFACTUAL_GENERATE_DATA_DIR / 'config.used.json'
            COUNTERFACTUAL_PREDICTIONS_PATH = TABLES_DIR / 'predictions_counterfactual.jsonl'
            COUNTERFACTUAL_METRICS_PATH = TABLES_DIR / 'metrics_counterfactual.json'
            COUNTERFACTUAL_SAVED_DATASET_DIR = Path(str(SAVED_DATASET_DIR) + '_counterfactual')
            COUNTERFACTUAL_EXPECTED_CACHE_CONFIG = dict(EXPECTED_CACHE_CONFIG)
            COUNTERFACTUAL_EXPECTED_CACHE_CONFIG['insertion_positions'] = counterfactual_positions

            if COUNTERFACTUAL_SAVED_DATASET_DIR.exists():
                try:
                    counterfactual_cache_info = validate_counting_dataset_cache(
                        COUNTERFACTUAL_SAVED_DATASET_DIR,
                        COUNTERFACTUAL_EXPECTED_CACHE_CONFIG,
                    )
                    print('Validated cached counterfactual dataset:', COUNTERFACTUAL_SAVED_DATASET_DIR)
                    shutil.copyfile(counterfactual_cache_info['dataset_path'], COUNTERFACTUAL_DATASET_PATH)
                    shutil.copyfile(counterfactual_cache_info['config_path'], COUNTERFACTUAL_CONFIG_PATH)
                    if counterfactual_cache_info.get('predictions_path') is not None:
                        shutil.copyfile(counterfactual_cache_info['predictions_path'], COUNTERFACTUAL_PREDICTIONS_PATH)
                        print('Validated cached counterfactual scored predictions.')
                    if counterfactual_cache_info.get('metrics_path') is not None:
                        shutil.copyfile(counterfactual_cache_info['metrics_path'], COUNTERFACTUAL_METRICS_PATH)
                except Exception as exc:
                    print('Existing cached counterfactual dataset did not validate; regenerating. Reason:', repr(exc))

            if not COUNTERFACTUAL_DATASET_PATH.exists() or not COUNTERFACTUAL_CONFIG_PATH.exists():
                positions = ' '.join('null' if p is None else str(p) for p in counterfactual_positions)
                cmd = (
                    f"PYTHONPATH=src python scripts/generate_dynamic_niah_v2.py "
                    f"--config configs/niah_dynamic.json "
                    f"--task-type {TASK_TYPE} "
                    f"--tokenizer {TOKENIZER_NAME} "
                    f"--num-examples {NUM_EXAMPLES} "
                    f"--target-haystack-tokens {TARGET_HAYSTACK_TOKENS} "
                    f"--num-needles {NUM_NEEDLES} "
                    f"--positions {positions} "
                    f"--prompt-style {PROMPT_STYLE} "
                    f"--counting-needle-kind {COUNTING_NEEDLE_KIND} "
                    f"--marker-text {shlex.quote(MARKER_TEXT)} "
                    f"--uid-token-length {UID_TOKEN_LENGTH} "
                    f"--global-random-seed {GLOBAL_RANDOM_SEED} "
                    f"--haystack-seed {HAYSTACK_SEED} "
                    f"--needle-seed {NEEDLE_SEED} "
                    f"--fact-templates-path {FACT_TEMPLATES_PATH} "
                    f"--run-dir {COUNTERFACTUAL_SCRIPT_RUN_DIR} "
                    f"--output-dir {COUNTERFACTUAL_GENERATE_DATA_DIR} "
                    f"--data-save-path {COUNTERFACTUAL_DATASET_PATH}"
                )
                if RANDOMIZE_NEEDLE_INSERTION:
                    cmd += f' --randomize-needle-insertion --randomize-needle-seed {RANDOMIZE_NEEDLE_SEED}'
                if SENTENCE_LEVEL_INSERTION:
                    cmd += ' --sentence-level-insertion'
                if WORD_LEVEL_INSERTION:
                    cmd += ' --word-level-insertion'
                if USE_THINKING:
                    cmd += ' --thinking-mode'
                print(cmd)
                subprocess.run(cmd, shell=True, check=True)
            else:
                print('Using existing counterfactual dataset:', COUNTERFACTUAL_DATASET_PATH)
                print('Using existing counterfactual config:', COUNTERFACTUAL_CONFIG_PATH)
            if not COUNTERFACTUAL_DATASET_PATH.exists():
                raise FileNotFoundError(f'Counterfactual dataset generation did not write expected dataset: {COUNTERFACTUAL_DATASET_PATH}')
            if not COUNTERFACTUAL_CONFIG_PATH.exists():
                raise FileNotFoundError(f'Counterfactual dataset generation did not write expected config: {COUNTERFACTUAL_CONFIG_PATH}')

            if RUN_GENERATION_EVAL and not COUNTERFACTUAL_PREDICTIONS_PATH.exists():
                cmd = (
                    f"PYTHONUNBUFFERED=1 PYTHONPATH=src python -u scripts/gen_responses.py "
                    f"--config {COUNTERFACTUAL_CONFIG_PATH} "
                    f"--tokenizer {TOKENIZER_NAME} "
                    f"--data-save-path {COUNTERFACTUAL_DATASET_PATH} "
                    f"--output-pred-jsonl {COUNTERFACTUAL_PREDICTIONS_PATH} "
                    f"--output-metrics-json {COUNTERFACTUAL_METRICS_PATH} "
                )
                if not USE_THINKING and not USE_KV_CACHE_FOR_NONTHINKG:
                    cmd += ' --no-use-kv-cache-for-nonthinkg'
                print(cmd)
                print('Launching response-generation subprocess; model loading may take a few minutes before tqdm starts.', flush=True)
                subprocess.run(cmd, shell=True, check=True)
            elif COUNTERFACTUAL_PREDICTIONS_PATH.exists():
                print('Skipping counterfactual response generation/eval because scored predictions are available:', COUNTERFACTUAL_PREDICTIONS_PATH)
            else:
                raise RuntimeError('Counterfactual direction requires scored counterfactual predictions; set RUN_GENERATION_EVAL=True or provide predictions_counterfactual.jsonl.')

            original_disk_rows = load_jsonl(DATASET_PATH)
            original_disk_count_summary = validate_counting_dataset_count(
                original_disk_rows,
                expected_count=sum(pos is not None for pos in INSERTION_POSITIONS),
                label='original counting-feature dataset on disk after counterfactual generation',
            )
            print('Original on-disk count summary after counterfactual generation:', json.dumps(original_disk_count_summary, indent=2))
            counterfactual_rows = load_jsonl(COUNTERFACTUAL_DATASET_PATH)
            counterfactual_count_summary = validate_counting_dataset_count(
                counterfactual_rows,
                expected_count=sum(pos is not None for pos in counterfactual_positions),
                label='counterfactual counting-feature dataset',
            )
            print('Counterfactual count summary:', json.dumps(counterfactual_count_summary, indent=2))
            counterfactual_scored_rows = load_scored_rows(COUNTERFACTUAL_PREDICTIONS_PATH)
            counterfactual_successful_rows, counterfactual_filter_summary = filter_successful_rows(
                counterfactual_rows,
                counterfactual_scored_rows,
                filter_example=FILTER_EXAMPLE,
            )
            print('Counterfactual filter summary:', json.dumps(counterfactual_filter_summary, indent=2))
            if not counterfactual_successful_rows:
                raise RuntimeError('No selected counterfactual examples available.')

            COUNTERFACTUAL_ROOT_TENSORS_DIR = FEATURE_TENSORS_DIR / 'counterfactual'
            COUNTERFACTUAL_ROOT_TABLES_DIR = FEATURE_TABLES_DIR / 'counterfactual'
            COUNTERFACTUAL_TENSORS_DIR = COUNTERFACTUAL_ROOT_TENSORS_DIR / FEATURE_CALC_POS
            COUNTERFACTUAL_TABLES_DIR = COUNTERFACTUAL_ROOT_TABLES_DIR / FEATURE_CALC_POS
            COUNTERFACTUAL_ORIGINAL_HIDDEN_DIR = COUNTERFACTUAL_TENSORS_DIR / 'original'
            COUNTERFACTUAL_COUNTERFACTUAL_HIDDEN_DIR = COUNTERFACTUAL_TENSORS_DIR / 'counterfactual'
            for path in [COUNTERFACTUAL_TENSORS_DIR, COUNTERFACTUAL_TABLES_DIR, COUNTERFACTUAL_ORIGINAL_HIDDEN_DIR, COUNTERFACTUAL_COUNTERFACTUAL_HIDDEN_DIR]:
                path.mkdir(parents=True, exist_ok=True)
            COUNTERFACTUAL_SUMMARY_PATH = COUNTERFACTUAL_TABLES_DIR / 'counterfactual_count_summary.csv'
            COUNTERFACTUAL_METADATA_PATH = COUNTERFACTUAL_TABLES_DIR / 'counterfactual_count_metadata.json'
            required_counterfactual_outputs = [
                COUNTERFACTUAL_SUMMARY_PATH,
                COUNTERFACTUAL_METADATA_PATH,
                *(COUNTERFACTUAL_TENSORS_DIR / f'counterfactual_count_layer_{layer}.pt' for layer in LAYERS),
            ]
            if all(path.exists() for path in required_counterfactual_outputs):
                print('Reusing existing counterfactual count directions; skipping Block 8 calculation.')
                summary_df = pd.read_csv(COUNTERFACTUAL_SUMMARY_PATH)
            else:
                original_successful_dataset_indices = dataset_indices_for_selected_rows(rows, successful_rows)
                original_successful_row_ids = [row_id_for_notebook(row, idx) for idx, row in zip(original_successful_dataset_indices, successful_rows)]
                original_candidate_examples, original_feature_length_stats = tokenized_feature_examples(
                    successful_rows,
                    original_successful_dataset_indices,
                    position_mode=FEATURE_CALC_POS,
                    config_obj=cfg,
                )
                prepared_original = prepare_feature_examples_for_position(
                    rows=successful_rows,
                    examples=original_candidate_examples,
                    selected_dataset_indices=original_successful_dataset_indices,
                    selected_row_ids=original_successful_row_ids,
                    position_mode=FEATURE_CALC_POS,
                    label='original_successful',
                )

                counterfactual_cfg = DynamicNiahV2Config.from_config_file(COUNTERFACTUAL_CONFIG_PATH)
                counterfactual_successful_dataset_indices = dataset_indices_for_selected_rows(counterfactual_rows, counterfactual_successful_rows)
                counterfactual_successful_row_ids = [row_id_for_notebook(row, idx) for idx, row in zip(counterfactual_successful_dataset_indices, counterfactual_successful_rows)]
                counterfactual_candidate_examples, counterfactual_feature_length_stats = tokenized_feature_examples(
                    counterfactual_successful_rows,
                    counterfactual_successful_dataset_indices,
                    position_mode=FEATURE_CALC_POS,
                    config_obj=counterfactual_cfg,
                )
                prepared_counterfactual = prepare_feature_examples_for_position(
                    rows=counterfactual_successful_rows,
                    examples=counterfactual_candidate_examples,
                    selected_dataset_indices=counterfactual_successful_dataset_indices,
                    selected_row_ids=counterfactual_successful_row_ids,
                    position_mode=FEATURE_CALC_POS,
                    label='counterfactual_successful',
                )
                print('Feature position:', FEATURE_CALC_POS)
                print('Prepared original summary:', json.dumps(prepared_original.summary, indent=2))
                print('Prepared counterfactual summary:', json.dumps(prepared_counterfactual.summary, indent=2))
                if prepared_original.summary['num_selected'] < 10 or prepared_counterfactual.summary['num_selected'] < 10:
                    print('WARNING: counterfactual direction has fewer than 10 successful examples in at least one dataset after position filtering.')
                print('Original feature tokenized sequence length stats:', json.dumps(original_feature_length_stats, indent=2))
                print('Counterfactual feature tokenized sequence length stats:', json.dumps(counterfactual_feature_length_stats, indent=2))

                missing_original_layers = [
                    layer for layer in LAYERS
                    if not (COUNTERFACTUAL_ORIGINAL_HIDDEN_DIR / f'hidden_layer_{layer}.pt').exists()
                ]
                if missing_original_layers:
                    extract_hidden_states_by_layer(
                        model,
                        prepared_original.examples,
                        missing_original_layers,
                        COUNTERFACTUAL_ORIGINAL_HIDDEN_DIR,
                        dtype=HIDDEN_STATE_DTYPE,
                    )
                else:
                    print('All requested original counterfactual hidden-state files already exist.')

                missing_counterfactual_layers = [
                    layer for layer in LAYERS
                    if not (COUNTERFACTUAL_COUNTERFACTUAL_HIDDEN_DIR / f'hidden_layer_{layer}.pt').exists()
                ]
                if missing_counterfactual_layers:
                    extract_hidden_states_by_layer(
                        model,
                        prepared_counterfactual.examples,
                        missing_counterfactual_layers,
                        COUNTERFACTUAL_COUNTERFACTUAL_HIDDEN_DIR,
                        dtype=HIDDEN_STATE_DTYPE,
                    )
                else:
                    print('All requested counterfactual hidden-state files already exist.')

                original_gold_count = unique_gold_count_or_none(prepared_original.rows)
                counterfactual_gold_count = unique_gold_count_or_none(prepared_counterfactual.rows)
                summary_rows = []
                for layer in LAYERS:
                    original_hidden = torch.load(COUNTERFACTUAL_ORIGINAL_HIDDEN_DIR / f'hidden_layer_{layer}.pt', map_location='cpu')
                    counterfactual_hidden = torch.load(COUNTERFACTUAL_COUNTERFACTUAL_HIDDEN_DIR / f'hidden_layer_{layer}.pt', map_location='cpu')
                    result = fit_counterfactual_count_direction(
                        original_hidden,
                        prepared_original.examples,
                        counterfactual_hidden,
                        prepared_counterfactual.examples,
                        layer=layer,
                        position_mode=FEATURE_CALC_POS,
                    )
                    save_counterfactual_count_direction(
                        result,
                        COUNTERFACTUAL_TENSORS_DIR,
                        original_dataset_indices=prepared_original.selected_dataset_indices,
                        original_row_ids=prepared_original.selected_row_ids,
                        original_position_indices=prepared_original.position_indices,
                        counterfactual_dataset_indices=prepared_counterfactual.selected_dataset_indices,
                        counterfactual_row_ids=prepared_counterfactual.selected_row_ids,
                        counterfactual_position_indices=prepared_counterfactual.position_indices,
                        original_gold_count=original_gold_count,
                        counterfactual_gold_count=counterfactual_gold_count,
                    )
                    summary_rows.append(result.metrics)
                    del original_hidden, counterfactual_hidden
                    release_torch_memory(collect_garbage=True)
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(COUNTERFACTUAL_SUMMARY_PATH, index=False)
                write_json(
                    COUNTERFACTUAL_METADATA_PATH,
                    {
                        'method': 'counterfactual',
                        'position': FEATURE_CALC_POS,
                        'counterfactual_removed_needle_index': COUNTERFACTUAL_REMOVED_NEEDLE_INDEX,
                        'original_insertion_positions': INSERTION_POSITIONS,
                        'counterfactual_insertion_positions': counterfactual_positions,
                        'original_dataset_path': str(DATASET_PATH),
                        'counterfactual_dataset_path': str(COUNTERFACTUAL_DATASET_PATH),
                        'original_predictions_path': str(PREDICTIONS_PATH),
                        'counterfactual_predictions_path': str(COUNTERFACTUAL_PREDICTIONS_PATH),
                        'original_filter_summary': filter_summary,
                        'counterfactual_filter_summary': counterfactual_filter_summary,
                        'prepared_original_summary': prepared_original.summary,
                        'prepared_counterfactual_summary': prepared_counterfactual.summary,
                        'original_length_stats': original_feature_length_stats,
                        'counterfactual_length_stats': counterfactual_feature_length_stats,
                        'original_gold_count': original_gold_count,
                        'counterfactual_gold_count': counterfactual_gold_count,
                        'layers': LAYERS,
                        'summary_path': str(COUNTERFACTUAL_SUMMARY_PATH),
                        'tensor_dir': str(COUNTERFACTUAL_TENSORS_DIR),
                        'original_hidden_dir': str(COUNTERFACTUAL_ORIGINAL_HIDDEN_DIR),
                        'counterfactual_hidden_dir': str(COUNTERFACTUAL_COUNTERFACTUAL_HIDDEN_DIR),
                    },
                )
            summary_df

# %% Notebook cell 18
with timing.stage('regular_steering_sweep'):
    import pandas as pd

    from counting.feature_analysis import load_scored_rows, write_json
    from counting.steering import (
        run_counting_steering_sweep,
        run_needle_span_counting_steering_sweep,
        select_steering_examples,
        write_csv as write_steering_csv,
        write_jsonl as write_steering_jsonl,
    )

    if STEERING_POSITION_MODE not in {'last_token', 'needle_span'}:
        raise ValueError(
            "STEERING_POSITION_MODE must be 'last_token' or 'needle_span', "
            f"got {STEERING_POSITION_MODE!r}"
        )
    if COUNTING_FEATURE_CALC_METHOD not in {'ridge', 'contrastive-success', 'counterfactual'}:
        raise ValueError(
            "COUNTING_FEATURE_CALC_METHOD must be 'ridge', 'contrastive-success', or 'counterfactual', "
            f"got {COUNTING_FEATURE_CALC_METHOD!r}"
        )
    if FEATURE_CALC_POS not in {'last', 'needle-last'}:
        raise ValueError(
            "FEATURE_CALC_POS must be 'last' or 'needle-last', "
            f"got {FEATURE_CALC_POS!r}"
        )
    if COUNTING_FEATURE_CALC_METHOD in {'ridge', 'contrastive-success'}:
        print('WARNING: COUNTERFACTUAL_REMOVED_NEEDLE_INDEX is ignored for ridge and contrastive-success steering.')

    if COUNTING_FEATURE_CALC_METHOD == 'ridge':
        steering_dir_name = 'steering' if STEERING_POSITION_MODE == 'last_token' else 'steering_needle_span'
        VECTOR_DIR = FEATURE_TENSORS_DIR
        VECTOR_LABEL = 'ridge probe'
    elif COUNTING_FEATURE_CALC_METHOD == 'contrastive-success':
        steering_dir_name = (
            'steering_contrastive_success'
            if STEERING_POSITION_MODE == 'last_token'
            else 'steering_contrastive_success_needle_span'
        )
        VECTOR_DIR = FEATURE_TENSORS_DIR / 'contrastive_success' / FEATURE_CALC_POS
        VECTOR_LABEL = 'contrastive success vector'
    else:
        steering_dir_name = (
            'steering_counterfactual'
            if STEERING_POSITION_MODE == 'last_token'
            else 'steering_counterfactual_needle_span'
        )
        VECTOR_DIR = FEATURE_TENSORS_DIR / 'counterfactual' / FEATURE_CALC_POS
        VECTOR_LABEL = 'counterfactual count vector'
    STEERING_TABLES_DIR = FEATURE_TABLES_DIR / steering_dir_name
    STEERING_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    STEERING_DETAILS_PATH = STEERING_TABLES_DIR / 'steering_results.jsonl'
    STEERING_SUMMARY_PATH = STEERING_TABLES_DIR / 'steering_summary.csv'
    STEERING_METADATA_PATH = STEERING_TABLES_DIR / 'steering_metadata.json'

    if RUN_STEERING:
        if not PREDICTIONS_PATH.exists():
            raise FileNotFoundError(
                f'Steering requires scored predictions for successful/unsuccessful grouping: {PREDICTIONS_PATH}'
            )
        steering_scored_rows = load_scored_rows(PREDICTIONS_PATH)
        steering_examples, steering_selection_summary = select_steering_examples(
            rows,
            steering_scored_rows,
            max_total=MAX_NUM_STEERING_EXAMPLES,
        )
        print('Steering selection summary:', json.dumps(steering_selection_summary, indent=2))
        if not steering_examples:
            raise RuntimeError('No scored examples available for steering.')

        if COUNTING_FEATURE_CALC_METHOD == 'ridge':
            missing_vector_layers = [
                layer for layer in LAYERS
                if not (VECTOR_DIR / f'ridge_probe_layer_{layer}.pt').exists()
            ]
        elif COUNTING_FEATURE_CALC_METHOD == 'contrastive-success':
            missing_vector_layers = [
                layer for layer in LAYERS
                if not (VECTOR_DIR / f'contrastive_success_layer_{layer}.pt').exists()
            ]
        else:
            missing_vector_layers = [
                layer for layer in LAYERS
                if not (VECTOR_DIR / f'counterfactual_count_layer_{layer}.pt').exists()
            ]
        if missing_vector_layers:
            raise FileNotFoundError(
                f'Missing {VECTOR_LABEL} files for steering layers: '
                + ', '.join(str(layer) for layer in missing_vector_layers)
            )

        if STEERING_POSITION_MODE == 'last_token':
            steering_details, steering_summary, steering_metadata = run_counting_steering_sweep(
                model=model,
                tokenizer=tokenizer,
                examples=steering_examples,
                layers=LAYERS,
                probe_dir=VECTOR_DIR,
                betas=STEERING_COEFF,
                max_new_tokens=MAX_NEW_TOKEN_STEERING,
                thinking_mode=USE_THINKING,
                vector_source=COUNTING_FEATURE_CALC_METHOD,
            )
            steering_metadata['steering_position_mode'] = 'last_token'
        else:
            steering_details, steering_summary, steering_metadata = run_needle_span_counting_steering_sweep(
                model=model,
                tokenizer=tokenizer,
                examples=steering_examples,
                layers=LAYERS,
                probe_dir=VECTOR_DIR,
                betas=STEERING_COEFF,
                max_new_tokens=MAX_NEW_TOKEN_STEERING,
                thinking_mode=USE_THINKING,
                vector_source=COUNTING_FEATURE_CALC_METHOD,
            )
        steering_metadata['counting_feature_calc_method'] = COUNTING_FEATURE_CALC_METHOD
        steering_metadata['feature_calc_pos'] = FEATURE_CALC_POS
        steering_metadata['counterfactual_removed_needle_index'] = COUNTERFACTUAL_REMOVED_NEEDLE_INDEX
        steering_metadata['selection_summary'] = steering_selection_summary
        steering_metadata['details_path'] = str(STEERING_DETAILS_PATH)
        steering_metadata['summary_path'] = str(STEERING_SUMMARY_PATH)

        write_steering_jsonl(steering_details, STEERING_DETAILS_PATH)
        write_steering_csv(steering_summary, STEERING_SUMMARY_PATH)
        write_json(STEERING_METADATA_PATH, steering_metadata)
        print('Counting feature calc method:', COUNTING_FEATURE_CALC_METHOD)
        print('Feature calc position:', FEATURE_CALC_POS)
        print('Vector dir:', VECTOR_DIR)
        print('Steering mode:', STEERING_POSITION_MODE)
        print('Steering details:', STEERING_DETAILS_PATH)
        print('Steering summary:', STEERING_SUMMARY_PATH)
        print('Steering metadata:', STEERING_METADATA_PATH)
        pd.DataFrame(steering_summary)
    else:
        print('Skipping counting-feature steering because RUN_STEERING=False.')

# %% Notebook cell 20
with timing.stage('steering_test_eval'):
    from counting.steering_eval import (
        generate_steering_eval_datasets,
        run_steering_test_eval,
        write_json as write_steering_eval_json,
        write_steering_test_eval_outputs,
    )

    STEERING_TEST_EVAL_DATA_DIR = GENERATE_DATA_DIR / 'steering_test_eval'
    STEERING_TEST_EVAL_TABLES_DIR = FEATURE_TABLES_DIR / 'steering_test_eval'
    STEERING_TEST_EVAL_TENSORS_DIR = FEATURE_TENSORS_DIR / 'steering_test_eval'
    STEERING_TEST_EVAL_SUMMARY_PATH = STEERING_TEST_EVAL_TABLES_DIR / 'steering_test_eval_summary.csv'
    STEERING_TEST_EVAL_PREDICTIONS_PATH = STEERING_TEST_EVAL_TABLES_DIR / 'steering_test_eval_predictions.csv'
    STEERING_TEST_EVAL_METADATA_PATH = STEERING_TEST_EVAL_TABLES_DIR / 'steering_test_eval_metadata.json'

    if STEERING_TEST_EVAL:
        if COUNTING_FEATURE_CALC_METHOD == 'ridge':
            missing_eval_vector_layers = [
                layer for layer in LAYERS
                if not (VECTOR_DIR / f'ridge_probe_layer_{layer}.pt').exists()
            ]
        elif COUNTING_FEATURE_CALC_METHOD == 'contrastive-success':
            missing_eval_vector_layers = [
                layer for layer in LAYERS
                if not (VECTOR_DIR / f'contrastive_success_layer_{layer}.pt').exists()
            ]
        else:
            missing_eval_vector_layers = [
                layer for layer in LAYERS
                if not (VECTOR_DIR / f'counterfactual_count_layer_{layer}.pt').exists()
            ]
        if missing_eval_vector_layers:
            raise FileNotFoundError(
                f'Missing {VECTOR_LABEL} files for steering-test eval layers: '
                + ', '.join(str(layer) for layer in missing_eval_vector_layers)
            )

        BASE_STEERING_EVAL_CONFIG = DynamicNiahV2Config.from_config_file(CONFIG_PATH)
        eval_datasets, eval_dataset_metadata = generate_steering_eval_datasets(
            base_config=BASE_STEERING_EVAL_CONFIG,
            output_root=STEERING_TEST_EVAL_DATA_DIR,
            max_needles=NUM_MAX_NEEDLES_STEERING_EVAL,
            num_examples=NUM_EXAMPLES_STEERING_EVAL,
            randomize_needle_min_separation=STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION,
        )
        steering_eval_details, steering_eval_summary, steering_eval_metadata = run_steering_test_eval(
            model=model,
            tokenizer=tokenizer,
            datasets=eval_datasets,
            layers=LAYERS,
            probe_dir=VECTOR_DIR,
            betas=STEERING_COEFF,
            max_new_tokens=MAX_NEW_TOKEN_STEERING,
            thinking_mode=USE_THINKING,
            vector_source=COUNTING_FEATURE_CALC_METHOD,
            steering_position_mode=STEERING_POSITION_MODE,
        )
        steering_eval_metadata['counting_feature_calc_method'] = COUNTING_FEATURE_CALC_METHOD
        steering_eval_metadata['feature_calc_pos'] = FEATURE_CALC_POS
        steering_eval_metadata['counterfactual_removed_needle_index'] = COUNTERFACTUAL_REMOVED_NEEDLE_INDEX
        steering_eval_metadata['num_max_needles_steering_eval'] = NUM_MAX_NEEDLES_STEERING_EVAL
        steering_eval_metadata['num_examples_steering_eval'] = NUM_EXAMPLES_STEERING_EVAL
        steering_eval_metadata['steering_eval_randomize_needle_min_separation'] = STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION
        steering_eval_metadata['dataset_metadata'] = eval_dataset_metadata

        steering_eval_paths = write_steering_test_eval_outputs(
            detail_rows=steering_eval_details,
            summary_rows=steering_eval_summary,
            metadata=steering_eval_metadata,
            tables_dir=STEERING_TEST_EVAL_TABLES_DIR,
            tensors_dir=STEERING_TEST_EVAL_TENSORS_DIR,
        )
        write_steering_eval_json(
            STEERING_TEST_EVAL_DATA_DIR / 'steering_test_eval_datasets_metadata.json',
            {'datasets': eval_dataset_metadata},
        )
        print('Steering test eval datasets:', STEERING_TEST_EVAL_DATA_DIR)
        print('Steering test eval outputs:', json.dumps(steering_eval_paths, indent=2))
        pd.DataFrame(steering_eval_summary)
    else:
        print('Skipping steering test eval because STEERING_TEST_EVAL=False.')

# %% Notebook cell 22
with timing.stage('cache_cleanup_and_archive'):
    if SAVE_COUNTING_FEATURE_CACHE:
        try:
            saved_feature_cache = save_counting_feature_cache(
                COUNTING_FEATURE_CACHE_DIR,
                feature_tensors_dir=FEATURE_TENSORS_DIR,
                feature_tables_dir=FEATURE_TABLES_DIR,
                feature_figures_dir=FEATURE_FIGURES_DIR,
                cache_config=COUNTING_FEATURE_CACHE_CONFIG,
                overwrite=False,
            )
            print('Saved reusable counting-feature cache:', json.dumps(saved_feature_cache, indent=2))
        except FileExistsError:
            try:
                validated_feature_cache = validate_counting_feature_cache(
                    COUNTING_FEATURE_CACHE_DIR,
                    COUNTING_FEATURE_CACHE_CONFIG,
                    layers=LAYERS,
                    run_classification=RUN_CLASSIFICATION,
                )
                print('Reusable counting-feature cache already exists and validates:', validated_feature_cache['cache_dir'])
            except Exception as exc:
                print('Counting-feature cache already exists but does not validate; not overwriting. Reason:', repr(exc))

    if SAVE_GENERATED_DATA:
        try:
            copied_cache = save_counting_dataset_cache(
                cache_dir=SAVED_DATASET_DIR,
                dataset_path=DATASET_PATH,
                config_path=CONFIG_PATH,
                predictions_path=PREDICTIONS_PATH if PREDICTIONS_PATH.exists() else None,
                metrics_path=METRICS_PATH if METRICS_PATH.exists() else None,
                overwrite=False,
            )
            print('Saved reusable generated/scored dataset cache:', json.dumps(copied_cache, indent=2))
        except FileExistsError:
            validated = validate_counting_dataset_cache(SAVED_DATASET_DIR, EXPECTED_CACHE_CONFIG)
            print('Reusable dataset cache already exists and validates:', validated['cache_dir'])

        if COUNTING_FEATURE_CALC_METHOD == 'counterfactual' and 'COUNTERFACTUAL_DATASET_PATH' in globals():
            try:
                copied_counterfactual_cache = save_counting_dataset_cache(
                    cache_dir=COUNTERFACTUAL_SAVED_DATASET_DIR,
                    dataset_path=COUNTERFACTUAL_DATASET_PATH,
                    config_path=COUNTERFACTUAL_CONFIG_PATH,
                    predictions_path=COUNTERFACTUAL_PREDICTIONS_PATH if COUNTERFACTUAL_PREDICTIONS_PATH.exists() else None,
                    metrics_path=COUNTERFACTUAL_METRICS_PATH if COUNTERFACTUAL_METRICS_PATH.exists() else None,
                    overwrite=False,
                )
                print('Saved reusable counterfactual generated/scored dataset cache:', json.dumps(copied_counterfactual_cache, indent=2))
            except FileExistsError:
                validated_counterfactual = validate_counting_dataset_cache(
                    COUNTERFACTUAL_SAVED_DATASET_DIR,
                    COUNTERFACTUAL_EXPECTED_CACHE_CONFIG,
                )
                print('Reusable counterfactual dataset cache already exists and validates:', validated_counterfactual['cache_dir'])

    cleanup_report = cleanup_counting_archive_artifacts(
        RUN_DIR,
        delete_large_pt=DELETE_LARGE_PT_WHEN_DONE,
        max_pt_bytes=100 * 1024 * 1024,
    )
    print(format_counting_cleanup_summary(cleanup_report))
    if not DELETE_LARGE_PT_WHEN_DONE:
        print('Large .pt deletion disabled; qk_cache, attention_stats/input_*, and corrupted_needle_tokens.jsonl were still removed.')

    archive_path = archive_directory(
        RUN_DIR,
        RESULTS_PATH,
        archive_name=RUN_NAME,
        include_source_dir=False,
    )
    print('Archive written to:', archive_path)

timing.save_json(TIMING_SUMMARY_JSON)
timing.save_csv(TIMING_SUMMARY_CSV)
print('Timing summary written to:', TIMING_SUMMARY_JSON, TIMING_SUMMARY_CSV)
