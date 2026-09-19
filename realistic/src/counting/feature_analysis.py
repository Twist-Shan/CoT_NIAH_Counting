from __future__ import annotations

import json
import shutil
import math
import csv
import gc
import time
import warnings
import zipfile
import collections
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from dataset_generation.response_eval import score_prediction
from single_example.single_example_analysis import (
    locate_uncontrolled_needle_segments,
    render_and_tokenize_messages,
)

TargetCountType = Literal["left_jump", "right_jump", "interpolation"]
DEFAULT_MAX_TRAIN_TOKENS = 200_000
DEFAULT_LARGE_FILE_WARNING_BYTES = 1_000_000_000
COUNTING_FEATURE_CACHE_METADATA_FILENAME = "counting_feature_cache_metadata.json"
COUNTING_PROBE_MODES = {
    "direct",
    "occurrence_index_probe",
    "mean_across_needles_span_last",
    "mean_across_needles_span_mean",
    "final_token",
    "all_diagnostics",
}
COUNTING_PROBE_MODE_ALIASES = {
    "mean_across_examples": "occurrence_index_probe",
    "mean_final_token": "final_token",
}


def normalize_counting_probe_mode(mode: str) -> str:
    return COUNTING_PROBE_MODE_ALIASES.get(str(mode), str(mode))


COUNTING_FEATURE_CONFIG_DEFAULTS: dict[str, Any] = {
    # Dataset/generation settings.
    "TASK_TYPE": "match_count",
    "MODEL_NAME": "Qwen/Qwen3-8B",
    "TOKENIZER_NAME": None,
    "SMOKE_TEST": False,
    "NUM_EXAMPLES": None,
    "TARGET_HAYSTACK_TOKENS": 1000,
    "NUM_NEEDLES": 3,
    "NUM_MAX_NEEDLES": None,
    "INSERTION_POSITIONS": [100, 200, 400],
    "RANDOMIZE_NEEDLE_INSERTION": True,
    "RANDOMIZE_NEEDLE_SEED": 42,
    "SENTENCE_LEVEL_INSERTION": True,
    "WORD_LEVEL_INSERTION": False,
    "LAYERS": [16, 20, 24, 28],
    "GLOBAL_RANDOM_SEED": 42,
    "HAYSTACK_SEED": 123,
    "NEEDLE_SEED": 456,
    "FACT_TEMPLATES_PATH": "data/templates/niah_fact_single_template.txt",
    "COUNTING_NEEDLE_KIND": "city_score",
    "MARKER_TEXT": "[dolphin]",
    "UID_TOKEN_LENGTH": 4,
    "PROMPT_STYLE": "vanilla",
    "USE_THINKING": False,
    "USE_KV_CACHE_FOR_NONTHINKG": True,
    # Runtime/output settings.
    "USER_RUN_NAME": None,
    "RUN_ROOT": "/content",
    "RUN_GENERATION_EVAL": True,
    "DELETE_LARGE_PT_WHEN_DONE": True,
    "RESULTS_PATH": None,
    "SAVE_GENERATED_DATA": True,
    "REUSE_COUNTING_FEATURE_CACHE": True,
    "SAVE_COUNTING_FEATURE_CACHE": False,
    "FEATURE_CACHE_ROOT": "results/counting_feature_cache",
    # Counting-feature settings.
    "TARGET_COUNT_TYPE": "interpolation",
    "TEST_FRACTION": 0.25,
    "SPLIT_SEED": 42,
    "HIDDEN_STATE_DTYPE": "bfloat16",
    "RIDGE_ALPHA": 1.0,
    "STANDARDIZE_FEATURES": True,
    "MAX_TRAIN_TOKENS_PER_LAYER": 50_000,
    "MAX_EVAL_TOKENS_PER_LAYER": 50_000,
    "RUN_CLASSIFICATION": None,
    "CLASSIFIER_EPOCHS": None,
    "CLASSIFIER_LR": 0.05,
    "CLASSIFIER_L2": 1e-4,
    "MAX_PLOT_POINTS": 20_000,
    "RUN_COUNTING_FEATURE_CALC": True,
    "FILTER_EXAMPLE": True,
    "COUNTING_FEATURE_CALC_METHOD": "counterfactual",
    "COUNTING_PROBE_MODE": "direct",
    "RUN_COUNTING_PROBE_BASELINE": True,
    "COUNTING_PROBE_BASELINE_MIN_DISTANCE": 5,
    "FEATURE_CALC_POS": "needle-last",
    "CONTRASTIVE_FEATURE_CALC_POS": None,
    "COUNTERFACTUAL_REMOVED_NEEDLE_INDEX": 0,
    # Needle-removal sensitivity diagnostics.
    "RUN_NEEDLE_SENSITIVITY": False,
    "NUM_REMOVAL": 3,
    "TARGET_SENSITIVITY_POSITION": "last-token",
    "NEEDLE_SENSITIVITY_SEED": 42,
    "MAX_SENSITIVITY_EXAMPLES": None,
    # Counting-feature steering settings.
    "RUN_STEERING": True,
    "STEERING_POSITION_MODE": "needle_span",
    "MAX_NUM_STEERING_EXAMPLES": 10,
    "STEERING_COEFF": [-2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
    "MAX_NEW_TOKEN_STEERING": 20,
    # Additional held-out-style steering evaluation settings.
    "STEERING_TEST_EVAL": False,
    "NUM_MAX_NEEDLES_STEERING_EVAL": 5,
    "NUM_EXAMPLES_STEERING_EVAL": 20,
    "STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION": 20,
}


def load_counting_feature_config_file(path: str | Path) -> dict[str, Any]:
    """Load notebook/script overrides from a valid JSON config file.

    Preferred file shape:

        {"config": {...}, "notes": {...}}

    Only the "config" object affects behavior. Notes and metadata fields are
    intentionally ignored so the JSON can remain human-readable without using
    non-standard JSON comments.
    """

    config_path = _as_path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Counting-feature config file must contain a JSON object: {config_path}"
        )

    if "config" in payload:
        config_payload = payload["config"]
        if not isinstance(config_payload, Mapping):
            raise ValueError(
                "Counting-feature config file field 'config' must be a JSON object: "
                f"{config_path}"
            )
        return dict(config_payload)

    ignored_keys = {"notes", "comments"}
    return {
        str(key): value
        for key, value in payload.items()
        if not str(key).startswith("_") and str(key) not in ignored_keys
    }


def build_counting_feature_run_config(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete notebook/runtime config after applying overrides.

    The counting-feature notebook intentionally keeps user-editable settings in
    one explicit dictionary. Derived defaults are resolved here so the printed
    and saved run metadata contains every parameter that affected the run.
    """

    cfg = dict(COUNTING_FEATURE_CONFIG_DEFAULTS)
    unknown = sorted(set((overrides or {})) - set(cfg))
    if unknown:
        raise ValueError(f"Unknown counting-feature config override(s): {unknown}")
    cfg.update(overrides or {})

    smoke_test = bool(cfg["SMOKE_TEST"])
    if cfg["TOKENIZER_NAME"] is None:
        cfg["TOKENIZER_NAME"] = cfg["MODEL_NAME"]
    if cfg["NUM_EXAMPLES"] is None:
        cfg["NUM_EXAMPLES"] = 20 if smoke_test else 100
    if cfg["LAYERS"] is None:
        cfg["LAYERS"] = [12] if smoke_test else [4, 8, 12, 16, 20, 24, 28]
    if cfg["RUN_CLASSIFICATION"] is None:
        cfg["RUN_CLASSIFICATION"] = cfg["TARGET_COUNT_TYPE"] != "interpolation"
    if cfg["CLASSIFIER_EPOCHS"] is None:
        cfg["CLASSIFIER_EPOCHS"] = 80 if smoke_test else 200

    cfg["RUN_ROOT"] = str(cfg["RUN_ROOT"])
    if cfg["RESULTS_PATH"] is not None:
        cfg["RESULTS_PATH"] = str(cfg["RESULTS_PATH"])
    cfg["FEATURE_CACHE_ROOT"] = str(cfg["FEATURE_CACHE_ROOT"])
    cfg["FACT_TEMPLATES_PATH"] = str(cfg["FACT_TEMPLATES_PATH"])
    cfg["INSERTION_POSITIONS"] = list(cfg["INSERTION_POSITIONS"])
    cfg["LAYERS"] = [int(layer) for layer in cfg["LAYERS"]]
    if cfg["NUM_MAX_NEEDLES"] is not None:
        cfg["NUM_MAX_NEEDLES"] = int(cfg["NUM_MAX_NEEDLES"])
        if cfg["NUM_MAX_NEEDLES"] <= 0:
            raise ValueError(
                "NUM_MAX_NEEDLES must be None or a positive integer, "
                f"got {cfg['NUM_MAX_NEEDLES']}"
            )
    legacy_feature_calc_pos = cfg.get("CONTRASTIVE_FEATURE_CALC_POS")
    if legacy_feature_calc_pos is not None:
        if (overrides or {}).get("FEATURE_CALC_POS") is not None:
            raise ValueError(
                "Use FEATURE_CALC_POS instead of CONTRASTIVE_FEATURE_CALC_POS; "
                "do not set both names."
            )
        warnings.warn(
            "CONTRASTIVE_FEATURE_CALC_POS is deprecated; use FEATURE_CALC_POS.",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg["FEATURE_CALC_POS"] = legacy_feature_calc_pos
    cfg["CONTRASTIVE_FEATURE_CALC_POS"] = None
    if cfg["COUNTING_FEATURE_CALC_METHOD"] not in {
        "ridge",
        "contrastive-success",
        "counterfactual",
    }:
        raise ValueError(
            "COUNTING_FEATURE_CALC_METHOD must be 'ridge', 'contrastive-success', "
            "or 'counterfactual', "
            f"got {cfg['COUNTING_FEATURE_CALC_METHOD']!r}"
        )
    cfg["COUNTING_PROBE_MODE"] = normalize_counting_probe_mode(
        cfg["COUNTING_PROBE_MODE"]
    )
    if cfg["COUNTING_PROBE_MODE"] not in COUNTING_PROBE_MODES:
        raise ValueError(
            "COUNTING_PROBE_MODE must be one of "
            f"{sorted(COUNTING_PROBE_MODES)}, got {cfg['COUNTING_PROBE_MODE']!r}"
        )
    cfg["RUN_COUNTING_PROBE_BASELINE"] = bool(cfg["RUN_COUNTING_PROBE_BASELINE"])
    cfg["COUNTING_PROBE_BASELINE_MIN_DISTANCE"] = int(
        cfg["COUNTING_PROBE_BASELINE_MIN_DISTANCE"]
    )
    if cfg["COUNTING_PROBE_BASELINE_MIN_DISTANCE"] < 0:
        raise ValueError(
            "COUNTING_PROBE_BASELINE_MIN_DISTANCE must be non-negative, "
            f"got {cfg['COUNTING_PROBE_BASELINE_MIN_DISTANCE']}"
        )
    if (
        cfg["COUNTING_PROBE_MODE"] != "direct"
        and cfg["COUNTING_FEATURE_CALC_METHOD"] != "ridge"
    ):
        raise ValueError(
            "Non-direct COUNTING_PROBE_MODE values require "
            "COUNTING_FEATURE_CALC_METHOD='ridge'; got "
            f"{cfg['COUNTING_FEATURE_CALC_METHOD']!r}"
        )
    if cfg["FEATURE_CALC_POS"] not in {"last", "needle-last"}:
        raise ValueError(
            "FEATURE_CALC_POS must be 'last' or 'needle-last', "
            f"got {cfg['FEATURE_CALC_POS']!r}"
        )
    cfg["COUNTERFACTUAL_REMOVED_NEEDLE_INDEX"] = int(
        cfg["COUNTERFACTUAL_REMOVED_NEEDLE_INDEX"]
    )
    cfg["RUN_COUNTING_FEATURE_CALC"] = bool(cfg["RUN_COUNTING_FEATURE_CALC"])
    cfg["FILTER_EXAMPLE"] = bool(cfg["FILTER_EXAMPLE"])
    cfg["RUN_NEEDLE_SENSITIVITY"] = bool(cfg["RUN_NEEDLE_SENSITIVITY"])
    cfg["NUM_REMOVAL"] = int(cfg["NUM_REMOVAL"])
    if cfg["NUM_REMOVAL"] <= 0:
        raise ValueError(f"NUM_REMOVAL must be positive, got {cfg['NUM_REMOVAL']}")
    cfg["TARGET_SENSITIVITY_POSITION"] = str(cfg["TARGET_SENSITIVITY_POSITION"])
    if cfg["TARGET_SENSITIVITY_POSITION"] != "last-token":
        raise ValueError(
            "TARGET_SENSITIVITY_POSITION currently supports only 'last-token', "
            f"got {cfg['TARGET_SENSITIVITY_POSITION']!r}"
        )
    cfg["NEEDLE_SENSITIVITY_SEED"] = int(cfg["NEEDLE_SENSITIVITY_SEED"])
    if cfg["MAX_SENSITIVITY_EXAMPLES"] is not None:
        cfg["MAX_SENSITIVITY_EXAMPLES"] = int(cfg["MAX_SENSITIVITY_EXAMPLES"])
        if cfg["MAX_SENSITIVITY_EXAMPLES"] <= 0:
            raise ValueError(
                "MAX_SENSITIVITY_EXAMPLES must be None or positive, "
                f"got {cfg['MAX_SENSITIVITY_EXAMPLES']}"
            )
    cfg["STEERING_TEST_EVAL"] = bool(cfg["STEERING_TEST_EVAL"])
    cfg["SENTENCE_LEVEL_INSERTION"] = bool(cfg["SENTENCE_LEVEL_INSERTION"])
    cfg["WORD_LEVEL_INSERTION"] = bool(cfg["WORD_LEVEL_INSERTION"])
    if cfg["SENTENCE_LEVEL_INSERTION"] and cfg["WORD_LEVEL_INSERTION"]:
        raise ValueError(
            "SENTENCE_LEVEL_INSERTION and WORD_LEVEL_INSERTION are mutually exclusive"
        )
    cfg["COUNTING_NEEDLE_KIND"] = str(cfg["COUNTING_NEEDLE_KIND"]).strip().lower()
    if cfg["COUNTING_NEEDLE_KIND"] not in {"city_score", "marker"}:
        raise ValueError(
            "COUNTING_NEEDLE_KIND must be 'city_score' or 'marker', "
            f"got {cfg['COUNTING_NEEDLE_KIND']!r}"
        )
    cfg["MARKER_TEXT"] = str(cfg["MARKER_TEXT"])
    cfg["UID_TOKEN_LENGTH"] = int(cfg["UID_TOKEN_LENGTH"])
    if cfg["UID_TOKEN_LENGTH"] <= 0:
        raise ValueError(
            "UID_TOKEN_LENGTH must be positive, "
            f"got {cfg['UID_TOKEN_LENGTH']}"
        )
    cfg["NUM_MAX_NEEDLES_STEERING_EVAL"] = int(
        cfg["NUM_MAX_NEEDLES_STEERING_EVAL"]
    )
    cfg["NUM_EXAMPLES_STEERING_EVAL"] = int(cfg["NUM_EXAMPLES_STEERING_EVAL"])
    cfg["STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION"] = int(
        cfg["STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION"]
    )
    if cfg["NUM_MAX_NEEDLES_STEERING_EVAL"] <= 0:
        raise ValueError(
            "NUM_MAX_NEEDLES_STEERING_EVAL must be positive, "
            f"got {cfg['NUM_MAX_NEEDLES_STEERING_EVAL']}"
        )
    if cfg["NUM_EXAMPLES_STEERING_EVAL"] <= 0:
        raise ValueError(
            "NUM_EXAMPLES_STEERING_EVAL must be positive, "
            f"got {cfg['NUM_EXAMPLES_STEERING_EVAL']}"
        )
    if cfg["STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION"] < 0:
        raise ValueError(
            "STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION must be non-negative, "
            f"got {cfg['STEERING_EVAL_RANDOMIZE_NEEDLE_MIN_SEPARATION']}"
        )
    return cfg


def save_counting_feature_run_metadata(
    path: str | Path,
    *,
    resolved_config: dict[str, Any],
    generation_config_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save auditable counting-feature run metadata including all configs."""

    generation_config: dict[str, Any] | None = None
    if generation_config_path is not None:
        gen_path = _as_path(generation_config_path)
        if gen_path.exists():
            generation_config = json.loads(gen_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_notebook_config": resolved_config,
        "generation_config_path": (
            None if generation_config_path is None else str(generation_config_path)
        ),
        "resolved_generation_config": generation_config,
    }
    if extra:
        payload.update(extra)
    return write_json(path, payload)


def print_counting_feature_run_config(config: dict[str, Any]) -> None:
    """Print the complete resolved config in a deterministic JSON format."""

    print(
        json.dumps(
            config, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default
        )
    )


_FEATURE_CACHE_CONFIG_KEYS = (
    "TASK_TYPE",
    "MODEL_NAME",
    "TOKENIZER_NAME",
    "SMOKE_TEST",
    "NUM_EXAMPLES",
    "TARGET_HAYSTACK_TOKENS",
    "NUM_NEEDLES",
    "NUM_MAX_NEEDLES",
    "INSERTION_POSITIONS",
    "RANDOMIZE_NEEDLE_INSERTION",
    "RANDOMIZE_NEEDLE_SEED",
    "SENTENCE_LEVEL_INSERTION",
    "WORD_LEVEL_INSERTION",
    "LAYERS",
    "GLOBAL_RANDOM_SEED",
    "HAYSTACK_SEED",
    "NEEDLE_SEED",
    "FACT_TEMPLATES_PATH",
    "COUNTING_NEEDLE_KIND",
    "MARKER_TEXT",
    "UID_TOKEN_LENGTH",
    "PROMPT_STYLE",
    "USE_THINKING",
    "TARGET_COUNT_TYPE",
    "TEST_FRACTION",
    "SPLIT_SEED",
    "HIDDEN_STATE_DTYPE",
    "RIDGE_ALPHA",
    "STANDARDIZE_FEATURES",
    "MAX_TRAIN_TOKENS_PER_LAYER",
    "MAX_EVAL_TOKENS_PER_LAYER",
    "RUN_CLASSIFICATION",
    "CLASSIFIER_EPOCHS",
    "CLASSIFIER_LR",
    "CLASSIFIER_L2",
    "RUN_COUNTING_FEATURE_CALC",
    "FILTER_EXAMPLE",
    "COUNTING_FEATURE_CALC_METHOD",
    "COUNTING_PROBE_MODE",
    "RUN_COUNTING_PROBE_BASELINE",
    "COUNTING_PROBE_BASELINE_MIN_DISTANCE",
    "FEATURE_CALC_POS",
    "COUNTERFACTUAL_REMOVED_NEEDLE_INDEX",
)


def build_counting_feature_cache_config(
    resolved_config: Mapping[str, Any], *, setting_name: str
) -> dict[str, Any]:
    """Return the stable settings that define reusable counting-feature artifacts."""

    payload = {
        key: resolved_config.get(key)
        for key in _FEATURE_CACHE_CONFIG_KEYS
        if key in resolved_config
    }
    payload["setting_name"] = str(setting_name)
    return json.loads(json.dumps(payload, default=_json_default))


def _copy_dir_contents(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Counting-feature cache source does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _required_counting_feature_cache_files(
    *,
    layers: Sequence[int],
    run_classification: bool,
    counting_feature_calc_method: str = "ridge",
    feature_calc_pos: str = "last",
) -> list[Path]:
    required = [
        Path("tensors/target_count_y_t.pt"),
        Path("tensors/target_count_metadata.json"),
        Path("tensors/matching_needle_token_mask.pt"),
        Path("tables/split_and_filter_summary.json"),
    ]
    method = str(counting_feature_calc_method)
    if method == "ridge":
        required.extend(
            [
                Path("tensors/hidden_state_metadata.json"),
                Path("tables/probe_summary.csv"),
            ]
        )
        for layer in [int(x) for x in layers]:
            required.extend(
                [
                    Path(f"tensors/hidden_layer_{layer}.pt"),
                    Path(f"tensors/ridge_probe_layer_{layer}.pt"),
                    Path(f"tensors/ridge_probe_layer_{layer}_metrics.json"),
                    Path(f"tables/ridge_layer_{layer}_eval.json"),
                    Path(f"figures/ridge_layer_{layer}_train_line.png"),
                    Path(f"figures/ridge_layer_{layer}_test_line.png"),
                    Path(f"figures/ridge_layer_{layer}_train_2d.png"),
                    Path(f"figures/ridge_layer_{layer}_test_2d.png"),
                ]
            )
            if run_classification:
                required.extend(
                    [
                        Path(f"tensors/classification_probe_layer_{layer}.pt"),
                        Path(f"tensors/classification_probe_layer_{layer}_metrics.json"),
                        Path(f"tables/classification_layer_{layer}_eval.json"),
                        Path(f"figures/classification_layer_{layer}_train_2d.png"),
                        Path(f"figures/classification_layer_{layer}_test_2d.png"),
                    ]
                )
    elif method == "contrastive-success":
        position = str(feature_calc_pos)
        required.extend(
            [
                Path(f"tensors/contrastive_success/{position}/hidden_state_metadata.json"),
                Path(
                    f"tables/contrastive_success/{position}/"
                    "contrastive_success_summary.csv"
                ),
                Path(
                    f"tables/contrastive_success/{position}/"
                    "contrastive_success_metadata.json"
                ),
            ]
        )
        for layer in [int(x) for x in layers]:
            required.extend(
                [
                    Path(f"tensors/contrastive_success/{position}/hidden_layer_{layer}.pt"),
                    Path(
                        f"tensors/contrastive_success/{position}/"
                        f"contrastive_success_layer_{layer}.pt"
                    ),
                ]
            )
    elif method == "counterfactual":
        position = str(feature_calc_pos)
        required.extend(
            [
                Path(
                    f"tables/counterfactual/{position}/"
                    "counterfactual_count_summary.csv"
                ),
                Path(
                    f"tables/counterfactual/{position}/"
                    "counterfactual_count_metadata.json"
                ),
                Path(
                    f"tensors/counterfactual/{position}/original/"
                    "hidden_state_metadata.json"
                ),
                Path(
                    f"tensors/counterfactual/{position}/counterfactual/"
                    "hidden_state_metadata.json"
                ),
            ]
        )
        for layer in [int(x) for x in layers]:
            required.extend(
                [
                    Path(
                        f"tensors/counterfactual/{position}/"
                        f"counterfactual_count_layer_{layer}.pt"
                    ),
                    Path(
                        f"tensors/counterfactual/{position}/original/"
                        f"hidden_layer_{layer}.pt"
                    ),
                    Path(
                        f"tensors/counterfactual/{position}/counterfactual/"
                        f"hidden_layer_{layer}.pt"
                    ),
                ]
            )
    else:
        raise ValueError(
            "counting_feature_calc_method must be 'ridge', 'contrastive-success', "
            "or 'counterfactual', "
            f"got {counting_feature_calc_method!r}"
        )
    return required


def validate_counting_feature_cache(
    cache_dir: str | Path,
    expected_config: Mapping[str, Any],
    *,
    layers: Sequence[int],
    run_classification: bool,
) -> dict[str, Any]:
    """Validate a reusable counting-feature cache against exact run settings."""

    cache = _as_path(cache_dir)
    metadata_path = cache / COUNTING_FEATURE_CACHE_METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing counting-feature cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_config = metadata.get("cache_config")
    expected = json.loads(json.dumps(dict(expected_config), default=_json_default))
    if actual_config != expected:
        raise ValueError(
            "Counting-feature cache settings do not match current run settings"
        )
    missing = [
        str(path)
        for path in _required_counting_feature_cache_files(
            layers=layers,
            run_classification=run_classification,
            counting_feature_calc_method=str(
                expected_config.get("COUNTING_FEATURE_CALC_METHOD", "ridge")
            ),
            feature_calc_pos=str(expected_config.get("FEATURE_CALC_POS", "last")),
        )
        if not (cache / path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Counting-feature cache is incomplete; missing files: " + ", ".join(missing[:20])
        )
    return {
        "cache_dir": str(cache),
        "metadata_path": str(metadata_path),
        "cache_config": actual_config,
        "num_required_files": len(
            _required_counting_feature_cache_files(
                layers=layers,
                run_classification=run_classification,
                counting_feature_calc_method=str(
                    expected_config.get("COUNTING_FEATURE_CALC_METHOD", "ridge")
                ),
                feature_calc_pos=str(expected_config.get("FEATURE_CALC_POS", "last")),
            )
        ),
    }


def restore_counting_feature_cache(
    cache_dir: str | Path,
    *,
    feature_tensors_dir: str | Path,
    feature_tables_dir: str | Path,
    feature_figures_dir: str | Path,
) -> dict[str, str]:
    """Copy cached counting-feature artifacts into the active run directory."""

    cache = _as_path(cache_dir)
    tensors = _as_path(feature_tensors_dir)
    tables = _as_path(feature_tables_dir)
    figures = _as_path(feature_figures_dir)
    _copy_dir_contents(cache / "tensors", tensors)
    _copy_dir_contents(cache / "tables", tables)
    _copy_dir_contents(cache / "figures", figures)
    return {
        "cache_dir": str(cache),
        "feature_tensors_dir": str(tensors),
        "feature_tables_dir": str(tables),
        "feature_figures_dir": str(figures),
    }


def save_counting_feature_cache(
    cache_dir: str | Path,
    *,
    feature_tensors_dir: str | Path,
    feature_tables_dir: str | Path,
    feature_figures_dir: str | Path,
    cache_config: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, str]:
    """Persist reusable counting-feature artifacts under a stable cache folder."""

    cache = _as_path(cache_dir)
    if cache.exists() and not overwrite:
        raise FileExistsError(f"Counting-feature cache already exists: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    _copy_dir_contents(_as_path(feature_tensors_dir), cache / "tensors")
    _copy_dir_contents(_as_path(feature_tables_dir), cache / "tables")
    _copy_dir_contents(_as_path(feature_figures_dir), cache / "figures")
    metadata_path = cache / COUNTING_FEATURE_CACHE_METADATA_FILENAME
    write_json(
        metadata_path,
        {
            "schema_version": "counting_feature_cache_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cache_config": json.loads(json.dumps(dict(cache_config), default=_json_default)),
        },
    )
    return {
        "cache_dir": str(cache),
        "metadata_path": str(metadata_path),
    }


@dataclass(frozen=True)
class TokenizedCountingExample:
    """Tokenized uncontrolled prompt and matching-needle spans for one row."""

    example_index: int
    row_id: str | None
    input_ids: list[int]
    needle_segments: list[dict[str, Any]]
    matching_needle_ids: list[str]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class SplitResult:
    train_indices: list[int]
    test_indices: list[int]
    seed: int
    test_fraction: float


@dataclass(frozen=True)
class ProbeSample:
    flat_indices: np.ndarray
    example_indices: np.ndarray
    token_indices: np.ndarray
    num_available: int
    num_sampled: int
    max_tokens: int | None
    seed: int


@dataclass(frozen=True)
class ContrastiveSelectionResult:
    selected_rows: list[dict[str, Any]]
    selected_dataset_indices: list[int]
    selected_row_ids: list[str]
    labels: list[str]
    successful_dataset_indices: list[int]
    unsuccessful_dataset_indices: list[int]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ContrastiveSuccessDirection:
    layer: int
    direction: torch.Tensor
    raw_direction: torch.Tensor
    raw_norm: float
    successful_mean: torch.Tensor
    unsuccessful_mean: torch.Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PreparedContrastiveExamples:
    rows: list[dict[str, Any]]
    examples: list[TokenizedCountingExample]
    selected_dataset_indices: list[int]
    selected_row_ids: list[str]
    labels: list[str]
    position_indices: list[int]
    summary: dict[str, Any]


@dataclass(frozen=True)
class PreparedFeatureExamples:
    rows: list[dict[str, Any]]
    examples: list[TokenizedCountingExample]
    selected_dataset_indices: list[int]
    selected_row_ids: list[str]
    position_indices: list[int]
    summary: dict[str, Any]


@dataclass(frozen=True)
class CounterfactualCountDirection:
    layer: int
    direction: torch.Tensor
    raw_direction: torch.Tensor
    raw_norm: float
    original_mean: torch.Tensor
    counterfactual_mean: torch.Tensor
    metrics: dict[str, Any]


def _row_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("id", index))


def select_contrastive_success_examples(
    dataset_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    min_group_warning: int = 10,
) -> ContrastiveSelectionResult:
    """Select a balanced success/failure pool for contrastive directions."""

    scored_by_id = {
        _row_id(row, idx): row for idx, row in enumerate(scored_rows)
    }
    successes: list[tuple[int, dict[str, Any]]] = []
    failures: list[tuple[int, dict[str, Any]]] = []
    missing_scores: list[str] = []
    for idx, row in enumerate(dataset_rows):
        row_id = _row_id(row, idx)
        scored = scored_by_id.get(row_id)
        if scored is None or "exact_match" not in scored:
            missing_scores.append(row_id)
            continue
        if bool(scored.get("exact_match")):
            successes.append((idx, dict(row)))
        else:
            failures.append((idx, dict(row)))

    n = min(len(successes), len(failures))
    selected_successes = successes[:n]
    selected_failures = failures[:n]
    selected = selected_successes + selected_failures
    labels = ["successful"] * len(selected_successes) + ["unsuccessful"] * len(
        selected_failures
    )
    selected_indices = [idx for idx, _row in selected]
    selected_rows = [row for _idx, row in selected]
    selected_row_ids = [_row_id(row, idx) for idx, row in selected]
    warnings_out: list[str] = []
    if len(successes) < int(min_group_warning) or len(failures) < int(min_group_warning):
        warnings_out.append(
            "Contrastive success direction has fewer than "
            f"{int(min_group_warning)} examples in at least one group: "
            f"successful={len(successes)}, unsuccessful={len(failures)}"
        )
    if n == 0:
        warnings_out.append(
            "No balanced contrastive examples are available because one group is empty"
        )

    summary = {
        "num_dataset_rows": len(dataset_rows),
        "num_scored_rows": len(scored_rows),
        "num_successful_available": len(successes),
        "num_unsuccessful_available": len(failures),
        "num_missing_scores": len(missing_scores),
        "missing_score_ids": missing_scores[:50],
        "num_selected_per_group": n,
        "num_selected_total": len(selected_rows),
        "selected_dataset_indices": selected_indices,
        "selected_row_ids": selected_row_ids,
        "selected_labels": labels,
        "warnings": warnings_out,
    }
    return ContrastiveSelectionResult(
        selected_rows=selected_rows,
        selected_dataset_indices=selected_indices,
        selected_row_ids=selected_row_ids,
        labels=labels,
        successful_dataset_indices=[idx for idx, _row in selected_successes],
        unsuccessful_dataset_indices=[idx for idx, _row in selected_failures],
        summary=summary,
    )


@dataclass(frozen=True)
class RidgeProbeResult:
    coef: np.ndarray
    intercept: float
    feature_mean: np.ndarray | None
    feature_scale: np.ndarray | None
    alpha: float
    standardize: bool
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class ClassificationProbeResult:
    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    feature_mean: np.ndarray | None
    feature_scale: np.ndarray | None
    l2_penalty: float
    standardize: bool
    metrics: dict[str, float | int | list[list[int]]]
    loss_history: list[float]


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = _as_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return out


def _bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024**3)


def torch_gpu_memory_snapshot() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of current CUDA memory use."""

    snapshot: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if not torch.cuda.is_available():
        return snapshot
    for device_idx in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{device_idx}")
        props = torch.cuda.get_device_properties(device)
        free_bytes = None
        total_bytes = int(props.total_memory)
        try:
            free_bytes, total_from_info = torch.cuda.mem_get_info(device)
            total_bytes = int(total_from_info)
        except Exception:
            free_bytes = None
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        max_allocated = int(torch.cuda.max_memory_allocated(device))
        max_reserved = int(torch.cuda.max_memory_reserved(device))
        snapshot["devices"].append(
            {
                "index": int(device_idx),
                "name": str(props.name),
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "max_allocated_bytes": max_allocated,
                "max_reserved_bytes": max_reserved,
                "free_bytes": None if free_bytes is None else int(free_bytes),
                "total_bytes": total_bytes,
                "allocated_gib": _bytes_to_gib(allocated),
                "reserved_gib": _bytes_to_gib(reserved),
                "max_allocated_gib": _bytes_to_gib(max_allocated),
                "max_reserved_gib": _bytes_to_gib(max_reserved),
                "free_gib": _bytes_to_gib(free_bytes),
                "total_gib": _bytes_to_gib(total_bytes),
            }
        )
    return snapshot


def print_gpu_memory_snapshot(
    label: str, snapshot: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Print and return a compact CUDA memory snapshot for logs/notebook output."""

    payload = torch_gpu_memory_snapshot() if snapshot is None else dict(snapshot)
    if not payload.get("cuda_available"):
        print(f"[gpu-memory] {label}: CUDA unavailable")
        return payload
    parts = []
    for device in payload.get("devices", []):
        free = device.get("free_gib")
        total = device.get("total_gib")
        allocated = device.get("allocated_gib")
        reserved = device.get("reserved_gib")
        max_allocated = device.get("max_allocated_gib")
        parts.append(
            "cuda:{index} {name}: allocated={allocated:.2f}GiB "
            "reserved={reserved:.2f}GiB max_allocated={max_allocated:.2f}GiB "
            "free={free:.2f}GiB total={total:.2f}GiB".format(
                index=device.get("index"),
                name=device.get("name"),
                allocated=allocated or 0.0,
                reserved=reserved or 0.0,
                max_allocated=max_allocated or 0.0,
                free=free or 0.0,
                total=total or 0.0,
            )
        )
    print(f"[gpu-memory] {label}: " + " | ".join(parts), flush=True)
    return payload


def write_response_generation_checkpoint_archive(
    *,
    run_dir: str | Path,
    archive_path: str | Path,
    dataset_path: str | Path,
    config_path: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
    metadata_path: str | Path | None = None,
    timing_json_path: str | Path | None = None,
    timing_csv_path: str | Path | None = None,
) -> Path:
    """Zip the expensive generation/scoring artifacts after they are available."""

    run_root = _as_path(run_dir)
    output_path = _as_path(archive_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {
        "dataset": _as_path(dataset_path),
        "config": _as_path(config_path),
        "predictions": _as_path(predictions_path),
        "metrics": _as_path(metrics_path),
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot write response-generation checkpoint; missing required file(s): "
            + "; ".join(missing)
        )
    optional = [
        _as_path(metadata_path) if metadata_path is not None else None,
        _as_path(timing_json_path) if timing_json_path is not None else None,
        _as_path(timing_csv_path) if timing_csv_path is not None else None,
    ]
    files = list(required.values()) + [path for path in optional if path is not None and path.exists()]
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            try:
                arcname = file_path.relative_to(run_root)
            except ValueError:
                arcname = file_path.name
            zf.write(file_path, arcname.as_posix())
    return output_path


class StageTimer:
    """Collect auditable wall-clock timings for notebook/script stages."""

    def __init__(
        self,
        *,
        json_path: str | Path | None = None,
        csv_path: str | Path | None = None,
    ) -> None:
        self.records: list[dict[str, Any]] = []
        self.json_path = None if json_path is None else _as_path(json_path)
        self.csv_path = None if csv_path is None else _as_path(csv_path)

    @contextmanager
    def stage(self, name: str, **metadata: Any):
        start_perf = time.perf_counter()
        start_utc = datetime.now(timezone.utc)
        gpu_memory_start = torch_gpu_memory_snapshot()
        record: dict[str, Any] = {
            "stage": str(name),
            "status": "completed",
            "start_utc": start_utc.isoformat(),
            "end_utc": None,
            "elapsed_seconds": None,
            "gpu_memory_start": gpu_memory_start,
            "gpu_memory_end": None,
        }
        if metadata:
            record["metadata"] = metadata
        print_gpu_memory_snapshot(f"{name} start", gpu_memory_start)
        try:
            yield record
        except Exception as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)
            raise
        finally:
            end_utc = datetime.now(timezone.utc)
            gpu_memory_end = torch_gpu_memory_snapshot()
            record["end_utc"] = end_utc.isoformat()
            record["elapsed_seconds"] = time.perf_counter() - start_perf
            record["gpu_memory_end"] = gpu_memory_end
            print_gpu_memory_snapshot(f"{name} end", gpu_memory_end)
            self.records.append(record)
            self.autosave()

    def mark_skipped(self, name: str, **metadata: Any) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {
            "stage": str(name),
            "status": "skipped",
            "start_utc": now,
            "end_utc": now,
            "elapsed_seconds": 0.0,
        }
        if metadata:
            record["metadata"] = metadata
        self.records.append(record)
        self.autosave()
        return record

    def autosave(self) -> None:
        if self.json_path is not None:
            self.save_json(self.json_path)
        if self.csv_path is not None:
            self.save_csv(self.csv_path)

    def to_payload(self) -> dict[str, Any]:
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "records": self.records,
        }

    def save_json(self, path: str | Path) -> Path:
        return write_json(path, self.to_payload())

    def save_csv(self, path: str | Path) -> Path:
        output_path = _as_path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "stage",
            "status",
            "start_utc",
            "end_utc",
            "elapsed_seconds",
            "error_type",
            "error_message",
            "gpu_memory_start",
            "gpu_memory_end",
            "metadata",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                row = {key: record.get(key, "") for key in fieldnames}
                if isinstance(row.get("metadata"), dict):
                    row["metadata"] = json.dumps(
                        row["metadata"],
                        sort_keys=True,
                        ensure_ascii=False,
                        default=_json_default,
                    )
                if isinstance(row.get("gpu_memory_start"), dict):
                    row["gpu_memory_start"] = json.dumps(
                        row["gpu_memory_start"],
                        sort_keys=True,
                        ensure_ascii=False,
                        default=_json_default,
                    )
                if isinstance(row.get("gpu_memory_end"), dict):
                    row["gpu_memory_end"] = json.dumps(
                        row["gpu_memory_end"],
                        sort_keys=True,
                        ensure_ascii=False,
                        default=_json_default,
                    )
                writer.writerow(row)
        return output_path


def warn_if_large_file(
    path: str | Path, *, threshold_bytes: int = DEFAULT_LARGE_FILE_WARNING_BYTES
) -> bool:
    """Warn and return True when a file exceeds the configured byte threshold."""

    file_path = _as_path(path)
    if not file_path.exists() or not file_path.is_file():
        return False
    size = file_path.stat().st_size
    if size > int(threshold_bytes):
        warnings.warn(
            f"Large counting-feature artifact: {file_path} is {size / (1024 ** 3):.2f} GiB; "
            f"consider reducing layers/examples or deleting intermediates before archiving.",
            RuntimeWarning,
            stacklevel=2,
        )
        return True
    return False


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = _as_path(path)
    rows: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_scored_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load response rows containing an `exact_match` field."""

    rows = load_jsonl(path)
    missing = [i for i, row in enumerate(rows) if "exact_match" not in row]
    if missing:
        raise ValueError(
            f"Scored rows at {path} are missing exact_match at rows {missing[:10]}"
        )
    return rows


def counting_dataset_count_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize gold and inserted counts for generated counting rows."""

    from collections import Counter

    gold_counts: Counter[int | None] = Counter()
    relevant_counts: Counter[int] = Counter()
    realized_counts: Counter[int] = Counter()
    inserted_counts: Counter[int] = Counter()
    for row in rows:
        raw_count = row.get("gold_answer", {}).get("count")
        try:
            gold_count = int(raw_count) if raw_count is not None else None
        except (TypeError, ValueError):
            gold_count = None
        gold_counts[gold_count] += 1
        relevant_counts[len(row.get("relevant_records", []) or [])] += 1
        realized_counts[len(row.get("realized_insertions", []) or [])] += 1
        inserted_counts[
            sum(
                1
                for needle in row.get("needles", []) or []
                if needle.get("is_inserted", True)
            )
        ] += 1
    return {
        "num_rows": len(rows),
        "gold_count_distribution": dict(gold_counts),
        "relevant_record_count_distribution": dict(relevant_counts),
        "realized_insertion_count_distribution": dict(realized_counts),
        "inserted_needle_count_distribution": dict(inserted_counts),
    }


def validate_counting_dataset_count(
    rows: Sequence[dict[str, Any]],
    *,
    expected_count: int,
    label: str,
) -> dict[str, Any]:
    """Raise if a counting dataset does not uniformly contain expected_count needles."""

    summary = counting_dataset_count_summary(rows)
    expected = {int(expected_count): int(summary["num_rows"])}
    checked_keys = [
        "gold_count_distribution",
        "relevant_record_count_distribution",
        "realized_insertion_count_distribution",
        "inserted_needle_count_distribution",
    ]
    mismatches = {key: summary[key] for key in checked_keys if summary[key] != expected}
    if mismatches:
        raise ValueError(
            f"{label} does not uniformly contain expected count {int(expected_count)}. "
            f"Summary: {json.dumps(summary)}"
        )
    return summary


def filter_successful_rows(
    dataset_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]] | None = None,
    *,
    filter_example: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select rows for counting-feature calculation.

    When `scored_rows` is omitted, this accepts dataset rows that already include
    `exact_match=True`. The notebook can generate scored rows first when needed.
    Set `filter_example=False` to keep all dataset rows while still reporting
    success/failure/missing-score counts for auditability.
    """

    if scored_rows is None:
        scored_by_id = {
            str(row.get("id", idx)): row for idx, row in enumerate(dataset_rows)
        }
    else:
        scored_by_id = {
            str(row.get("id", idx)): row for idx, row in enumerate(scored_rows)
        }
    kept: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    missing_scores: list[str] = []
    failed: list[str] = []
    for idx, row in enumerate(dataset_rows):
        row_id = str(row.get("id", idx))
        scored = scored_by_id.get(row_id)
        if scored is None or "exact_match" not in scored:
            missing_scores.append(row_id)
            if not filter_example:
                kept.append(dict(row))
            continue
        if bool(scored.get("exact_match")):
            row_copy = dict(row)
            successes.append(row_copy)
            kept.append(row_copy)
        else:
            failed.append(row_id)
            if not filter_example:
                kept.append(dict(row))
    summary = {
        "filter_example": bool(filter_example),
        "num_dataset_rows": len(dataset_rows),
        "num_scored_rows": (
            len(scored_rows) if scored_rows is not None else len(dataset_rows)
        ),
        "num_successful": len(successes),
        "num_failed": len(failed),
        "num_missing_scores": len(missing_scores),
        "num_rows_used": len(kept),
        "included_unsuccessful": not bool(filter_example) and bool(failed),
        "included_missing_scores": not bool(filter_example) and bool(missing_scores),
        "failed_ids": failed[:50],
        "missing_score_ids": missing_scores[:50],
    }
    return kept, summary


def exact_match_from_prediction(
    row: dict[str, Any], prediction: dict[str, Any]
) -> bool:
    return bool(
        score_prediction(prediction, row.get("gold_answer", {}), row["task_type"])
    )


def train_test_split_indices(
    n: int,
    *,
    test_fraction: float = 0.25,
    seed: int = 42,
    min_test: int = 1,
) -> SplitResult:
    if n <= 0:
        raise ValueError("Cannot split zero examples")
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1), got {test_fraction}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    test_n = max(int(min_test), int(round(n * float(test_fraction))))
    if n > 1:
        test_n = min(test_n, n - 1)
    else:
        test_n = 0
    test = sorted(int(x) for x in perm[:test_n])
    train = sorted(int(x) for x in perm[test_n:])
    return SplitResult(
        train_indices=train,
        test_indices=test,
        seed=int(seed),
        test_fraction=float(test_fraction),
    )


def sequence_length_stats(lengths: Sequence[int]) -> dict[str, float | int]:
    if not lengths:
        raise ValueError("No sequence lengths to summarize")
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
    }


def matching_needle_ids_for_row(row: dict[str, Any]) -> list[str]:
    """Return needle IDs that contribute to the row's gold answer."""

    ids: list[str] = []
    for record in row.get("relevant_records", []) or []:
        needle_id = record.get("needle_id")
        if needle_id is not None:
            ids.append(str(needle_id))
    if ids:
        return ids

    gold_count = row.get("gold_answer", {}).get("count")
    try:
        count = int(gold_count)
    except (TypeError, ValueError):
        return []
    inserted = []
    for needle in row.get("needles", []) or []:
        if needle.get("is_inserted", True):
            needle_id = needle.get("needle_id")
            if needle_id is not None:
                inserted.append(str(needle_id))
    return inserted[:count]


def _matching_segments(
    segments: Sequence[dict[str, Any]],
    matching_needle_ids: Sequence[str] | None,
    gold_count: int | None = None,
) -> list[dict[str, Any]]:
    if matching_needle_ids:
        wanted = {str(x) for x in matching_needle_ids}
        matched = [seg for seg in segments if str(seg.get("needle_id")) in wanted]
        if matched:
            return sorted(matched, key=lambda item: int(item["start"]))
    sorted_segments = sorted(segments, key=lambda item: int(item["start"]))
    if gold_count is None:
        return sorted_segments
    return sorted_segments[: max(0, int(gold_count))]


def build_target_count_vector(
    sequence_length: int,
    needle_segments: Sequence[dict[str, Any]],
    *,
    target_count_type: TargetCountType,
    matching_needle_ids: Sequence[str] | None = None,
    gold_count: int | None = None,
) -> torch.Tensor:
    """Build y_t for one tokenized uncontrolled prompt.

    The vector follows the plan's convention that hidden state h_t predicts the
    count before token position t, with explicit jump behavior inside needle spans.
    """

    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if target_count_type not in {"left_jump", "right_jump", "interpolation"}:
        raise ValueError(f"Unsupported target_count_type: {target_count_type}")
    y = torch.zeros(int(sequence_length), dtype=torch.float32)
    segments = _matching_segments(needle_segments, matching_needle_ids, gold_count)
    complete = 0.0
    cursor = 0
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if start < 0 or end > sequence_length or end <= start:
            raise ValueError(
                f"Invalid needle span {segment} for sequence_length={sequence_length}"
            )
        if start < cursor:
            raise ValueError(f"Needle spans overlap or are out of order: {segments}")
        if cursor < start:
            y[cursor:start] = complete
        length = end - start
        if target_count_type == "left_jump":
            y[start:end] = complete + 1.0
            complete += 1.0
            cursor = end
        elif target_count_type == "right_jump":
            y[start:end] = complete
            complete += 1.0
            cursor = end
        else:
            interp = torch.arange(length, dtype=torch.float32) / float(length)
            y[start:end] = complete + interp
            complete += 1.0
            cursor = end
    if cursor < sequence_length:
        y[cursor:] = complete
    return y


def build_target_count_matrix(
    examples: Sequence[TokenizedCountingExample],
    *,
    target_count_type: TargetCountType,
    max_sequence_length: int | None = None,
) -> torch.Tensor:
    if not examples:
        raise ValueError("No examples supplied")
    max_len = int(max_sequence_length or max(ex.sequence_length for ex in examples))
    target = torch.full((len(examples), max_len), float("nan"), dtype=torch.float32)
    for row_idx, example in enumerate(examples):
        y = build_target_count_vector(
            example.sequence_length,
            example.needle_segments,
            target_count_type=target_count_type,
            matching_needle_ids=example.matching_needle_ids,
        )
        target[row_idx, : example.sequence_length] = y
    return target


def build_needle_token_mask(
    examples: Sequence[TokenizedCountingExample],
    *,
    max_sequence_length: int | None = None,
    matching_only: bool = True,
) -> torch.Tensor:
    max_len = int(max_sequence_length or max(ex.sequence_length for ex in examples))
    mask = torch.zeros((len(examples), max_len), dtype=torch.bool)
    for row_idx, example in enumerate(examples):
        segments = _matching_segments(
            example.needle_segments,
            example.matching_needle_ids if matching_only else None,
        )
        for segment in segments:
            mask[row_idx, int(segment["start"]) : int(segment["end"])] = True
    return mask


def tokenize_counting_examples(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    thinking_mode: bool = False,
    expected_num_needles: int | Sequence[int] | None = None,
) -> tuple[list[TokenizedCountingExample], dict[str, Any]]:
    examples: list[TokenizedCountingExample] = []
    for idx, row in enumerate(rows):
        messages = row.get("uncontrolled_messages") or row.get("messages")
        if messages is None:
            raise ValueError(
                f"Row {idx} has neither uncontrolled_messages nor messages"
            )
        tokenized = render_and_tokenize_messages(
            tokenizer, messages, thinking_mode=thinking_mode
        )
        row_expected_num_needles = expected_num_needles
        if isinstance(expected_num_needles, Sequence) and not isinstance(
            expected_num_needles, (str, bytes)
        ):
            row_expected_num_needles = expected_num_needles[idx]
        segments = locate_uncontrolled_needle_segments(
            row=row,
            uncontrolled_input_ids=tokenized.input_ids,
            prompt_text=tokenized.prompt_text,
            token_offsets=tokenized.token_offsets,
            expected_num_needles=row_expected_num_needles,
        )
        examples.append(
            TokenizedCountingExample(
                example_index=idx,
                row_id=None if row.get("id") is None else str(row.get("id")),
                input_ids=[
                    int(x)
                    for x in tokenized.input_ids.detach().cpu().reshape(-1).tolist()
                ],
                needle_segments=segments,
                matching_needle_ids=matching_needle_ids_for_row(row),
            )
        )
    stats = sequence_length_stats([ex.sequence_length for ex in examples])
    return examples, stats


def tokenize_prompt_rows(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    thinking_mode: bool = False,
    dataset_indices: Sequence[int] | None = None,
) -> tuple[list[TokenizedCountingExample], dict[str, Any]]:
    """Tokenize uncontrolled prompts without requiring needle-span localization."""

    examples: list[TokenizedCountingExample] = []
    if dataset_indices is not None and len(dataset_indices) != len(rows):
        raise ValueError(
            "dataset_indices length must match rows length, got "
            f"{len(dataset_indices)} and {len(rows)}"
        )
    for local_idx, row in enumerate(rows):
        messages = row.get("uncontrolled_messages") or row.get("messages")
        if messages is None:
            raise ValueError(
                f"Row {local_idx} has neither uncontrolled_messages nor messages"
            )
        tokenized = render_and_tokenize_messages(
            tokenizer, messages, thinking_mode=thinking_mode
        )
        example_index = (
            int(dataset_indices[local_idx])
            if dataset_indices is not None
            else int(local_idx)
        )
        examples.append(
            TokenizedCountingExample(
                example_index=example_index,
                row_id=None if row.get("id") is None else str(row.get("id")),
                input_ids=[
                    int(x)
                    for x in tokenized.input_ids.detach().cpu().reshape(-1).tolist()
                ],
                needle_segments=[],
                matching_needle_ids=[],
            )
        )
    stats = sequence_length_stats([ex.sequence_length for ex in examples])
    return examples, stats


def contrastive_feature_position_index(
    example: TokenizedCountingExample, *, position_mode: str
) -> int:
    """Return the token position used for contrastive success features."""

    mode = str(position_mode)
    if mode == "last":
        pos = int(example.sequence_length) - 1
    elif mode == "needle-last":
        segments = _matching_segments(example.needle_segments, example.matching_needle_ids)
        if not segments:
            raise ValueError("no matching needle span found")
        last_segment = sorted(segments, key=lambda item: int(item["start"]))[-1]
        pos = int(last_segment["end"]) - 1
    else:
        raise ValueError(
            "position_mode must be 'last' or 'needle-last', "
            f"got {position_mode!r}"
        )
    if pos < 0 or pos >= int(example.sequence_length):
        raise ValueError(
            f"resolved position {pos} is outside sequence length {example.sequence_length}"
        )
    return pos


def prepare_contrastive_examples_for_position(
    *,
    rows: Sequence[dict[str, Any]],
    examples: Sequence[TokenizedCountingExample],
    labels: Sequence[str],
    selected_dataset_indices: Sequence[int],
    selected_row_ids: Sequence[str],
    position_mode: str,
) -> PreparedContrastiveExamples:
    """Skip unusable rows for the requested position and rebalance labels."""

    lengths = {
        "rows": len(rows),
        "examples": len(examples),
        "labels": len(labels),
        "selected_dataset_indices": len(selected_dataset_indices),
        "selected_row_ids": len(selected_row_ids),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Contrastive preparation inputs have mismatched lengths: {lengths}")

    skipped: list[dict[str, Any]] = []
    successful: list[tuple[dict[str, Any], TokenizedCountingExample, int, str, str, int]] = []
    unsuccessful: list[tuple[dict[str, Any], TokenizedCountingExample, int, str, str, int]] = []
    for row, example, label, dataset_index, row_id in zip(
        rows, examples, labels, selected_dataset_indices, selected_row_ids
    ):
        try:
            pos = contrastive_feature_position_index(
                example, position_mode=position_mode
            )
        except ValueError as exc:
            skipped.append(
                {
                    "dataset_index": int(dataset_index),
                    "row_id": str(row_id),
                    "label": str(label),
                    "reason": str(exc),
                }
            )
            continue
        item = (dict(row), example, int(dataset_index), str(row_id), str(label), int(pos))
        if str(label) == "successful":
            successful.append(item)
        elif str(label) == "unsuccessful":
            unsuccessful.append(item)
        else:
            skipped.append(
                {
                    "dataset_index": int(dataset_index),
                    "row_id": str(row_id),
                    "label": str(label),
                    "reason": "label is neither successful nor unsuccessful",
                }
            )

    n = min(len(successful), len(unsuccessful))
    balanced = successful[:n] + unsuccessful[:n]
    if n == 0:
        raise ValueError(
            "No balanced contrastive examples remain after resolving feature positions"
        )

    out_rows = [item[0] for item in balanced]
    out_examples = [item[1] for item in balanced]
    out_dataset_indices = [item[2] for item in balanced]
    out_row_ids = [item[3] for item in balanced]
    out_labels = [item[4] for item in balanced]
    out_positions = [item[5] for item in balanced]
    summary = {
        "position_mode": str(position_mode),
        "num_input_examples": len(rows),
        "num_skipped": len(skipped),
        "skipped": skipped[:50],
        "num_successful_after_skip": len(successful),
        "num_unsuccessful_after_skip": len(unsuccessful),
        "num_selected_per_group_after_rebalance": n,
        "num_selected_total_after_rebalance": len(balanced),
        "selected_dataset_indices": out_dataset_indices,
        "selected_row_ids": out_row_ids,
        "selected_labels": out_labels,
        "selected_position_indices": out_positions,
    }
    return PreparedContrastiveExamples(
        rows=out_rows,
        examples=out_examples,
        selected_dataset_indices=out_dataset_indices,
        selected_row_ids=out_row_ids,
        labels=out_labels,
        position_indices=out_positions,
        summary=summary,
    )


def counterfactual_insertion_positions(
    insertion_positions: Sequence[Any],
    *,
    removed_needle_index: int = 0,
) -> list[Any]:
    """Return insertion positions with one non-final needle slot removed."""

    positions = list(insertion_positions)
    if not positions:
        raise ValueError("INSERTION_POSITIONS must be non-empty for counterfactual runs")
    if any(pos is None for pos in positions):
        raise ValueError(
            "Counterfactual count direction requires original INSERTION_POSITIONS "
            "to contain no None values"
        )
    index = int(removed_needle_index)
    last_valid_removal = len(positions) - 2
    if index < 0 or index > last_valid_removal:
        raise ValueError(
            "COUNTERFACTUAL_REMOVED_NEEDLE_INDEX must be a valid non-last index; "
            f"got {index} for INSERTION_POSITIONS length {len(positions)}"
        )
    positions[index] = None
    return positions


def prepare_feature_examples_for_position(
    *,
    rows: Sequence[dict[str, Any]],
    examples: Sequence[TokenizedCountingExample],
    selected_dataset_indices: Sequence[int],
    selected_row_ids: Sequence[str],
    position_mode: str,
    label: str,
) -> PreparedFeatureExamples:
    """Skip rows that cannot provide the requested hidden-state feature position."""

    lengths = {
        "rows": len(rows),
        "examples": len(examples),
        "selected_dataset_indices": len(selected_dataset_indices),
        "selected_row_ids": len(selected_row_ids),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Feature preparation inputs have mismatched lengths: {lengths}")

    kept: list[tuple[dict[str, Any], TokenizedCountingExample, int, str, int]] = []
    skipped: list[dict[str, Any]] = []
    for row, example, dataset_index, row_id in zip(
        rows, examples, selected_dataset_indices, selected_row_ids
    ):
        try:
            pos = contrastive_feature_position_index(
                example, position_mode=position_mode
            )
        except ValueError as exc:
            skipped.append(
                {
                    "dataset_index": int(dataset_index),
                    "row_id": str(row_id),
                    "label": str(label),
                    "reason": str(exc),
                }
            )
            continue
        kept.append((dict(row), example, int(dataset_index), str(row_id), int(pos)))

    if not kept:
        raise ValueError(
            "No examples remain after resolving feature positions for "
            f"{label!r} examples"
        )

    out_rows = [item[0] for item in kept]
    out_examples = [item[1] for item in kept]
    out_dataset_indices = [item[2] for item in kept]
    out_row_ids = [item[3] for item in kept]
    out_positions = [item[4] for item in kept]
    summary = {
        "position_mode": str(position_mode),
        "label": str(label),
        "num_input_examples": len(rows),
        "num_selected": len(kept),
        "num_skipped": len(skipped),
        "skipped": skipped[:50],
        "selected_dataset_indices": out_dataset_indices,
        "selected_row_ids": out_row_ids,
        "selected_position_indices": out_positions,
    }
    return PreparedFeatureExamples(
        rows=out_rows,
        examples=out_examples,
        selected_dataset_indices=out_dataset_indices,
        selected_row_ids=out_row_ids,
        position_indices=out_positions,
        summary=summary,
    )


def save_target_artifacts(
    target: torch.Tensor,
    examples: Sequence[TokenizedCountingExample],
    output_dir: str | Path,
    *,
    target_count_type: TargetCountType,
    large_file_warning_bytes: int = DEFAULT_LARGE_FILE_WARNING_BYTES,
) -> dict[str, Path]:
    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "target_count_y_t.pt"
    metadata_path = out_dir / "target_count_metadata.json"
    torch.save(target, target_path)
    warn_if_large_file(target_path, threshold_bytes=large_file_warning_bytes)
    write_json(
        metadata_path,
        {
            "target_count_type": target_count_type,
            "shape": list(target.shape),
            "num_examples": len(examples),
            "examples": [
                {
                    "example_index": ex.example_index,
                    "row_id": ex.row_id,
                    "sequence_length": ex.sequence_length,
                    "matching_needle_ids": ex.matching_needle_ids,
                    "needle_segments": ex.needle_segments,
                }
                for ex in examples
            ],
        },
    )
    return {"target": target_path, "metadata": metadata_path}


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).lower().replace("torch.", "")
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def release_torch_memory(*, collect_garbage: bool = False) -> None:
    """Release unused Python/CUDA memory after large transient tensors are dropped."""

    if collect_garbage:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _forward_with_hidden_states_no_cache(model: Any, inputs: torch.Tensor) -> Any:
    try:
        return model(inputs, output_hidden_states=True, use_cache=False)
    except TypeError:
        return model(inputs, output_hidden_states=True)


def extract_hidden_states_by_layer(
    model: Any,
    examples: Sequence[TokenizedCountingExample],
    layers: Sequence[int],
    output_dir: str | Path,
    *,
    dtype: str | torch.dtype = "bfloat16",
    large_file_warning_bytes: int = DEFAULT_LARGE_FILE_WARNING_BYTES,
) -> dict[int, Path]:
    """Run one forward pass per example and save padded [n, T, d] tensors per layer."""

    if not examples:
        raise ValueError("No examples supplied")
    selected_layers = [int(layer) for layer in layers]
    if not selected_layers:
        raise ValueError("At least one layer is required")
    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_dtype = _torch_dtype(dtype)
    max_len = max(ex.sequence_length for ex in examples)
    device = model.get_input_embeddings().weight.device
    layer_tensors: dict[int, torch.Tensor] | None = None
    for row_idx, example in enumerate(examples):
        release_torch_memory(collect_garbage=False)
        inputs = torch.tensor([example.input_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = _forward_with_hidden_states_no_cache(model, inputs)
        hidden_states = list(out.hidden_states)
        if layer_tensors is None:
            hidden_dim = int(hidden_states[selected_layers[0]].shape[-1])
            layer_tensors = {
                layer: torch.full(
                    (len(examples), max_len, hidden_dim),
                    float("nan"),
                    dtype=storage_dtype,
                    device="cpu",
                )
                for layer in selected_layers
            }
        seq_len = example.sequence_length
        for layer in selected_layers:
            hidden = (
                hidden_states[layer]
                .detach()
                .squeeze(0)
                .to(device="cpu", dtype=storage_dtype)
            )
            layer_tensors[layer][row_idx, :seq_len, :] = hidden
            del hidden
        del hidden_states
        del out
        del inputs
        release_torch_memory(collect_garbage=True)
    assert layer_tensors is not None
    paths: dict[int, Path] = {}
    for layer, tensor in layer_tensors.items():
        path = out_dir / f"hidden_layer_{layer}.pt"
        torch.save(tensor, path)
        warn_if_large_file(path, threshold_bytes=large_file_warning_bytes)
        paths[layer] = path
    write_json(
        out_dir / "hidden_state_metadata.json",
        {
            "layers": selected_layers,
            "dtype": str(storage_dtype).replace("torch.", ""),
            "shape_by_layer": {
                str(layer): list(tensor.shape)
                for layer, tensor in layer_tensors.items()
            },
            "sequence_length_stats": sequence_length_stats(
                [ex.sequence_length for ex in examples]
            ),
        },
    )
    return paths


def _stable_seed_component(value: Any) -> int:
    text = str(value)
    total = 0
    for char in text:
        total = (total * 131 + ord(char)) % 2_147_483_647
    return total


def _non_needle_replacement_candidates(
    sequence_length: int,
    needle_segments: Sequence[Mapping[str, Any]],
    span_length: int,
) -> list[tuple[int, int]]:
    if span_length <= 0 or span_length > sequence_length:
        return []
    blocked = np.zeros(int(sequence_length), dtype=bool)
    for segment in needle_segments:
        start = max(0, int(segment.get("start", 0)))
        end = min(int(sequence_length), int(segment.get("end", 0)))
        if end > start:
            blocked[start:end] = True
    candidates: list[tuple[int, int]] = []
    for start in range(0, int(sequence_length) - int(span_length) + 1):
        end = start + int(span_length)
        if not bool(np.any(blocked[start:end])):
            candidates.append((start, end))
    return candidates


def _extract_target_vectors(
    model: Any,
    input_ids_batch: Sequence[Sequence[int]],
    *,
    layers: Sequence[int],
    target_position: int,
) -> dict[int, np.ndarray]:
    if not input_ids_batch:
        raise ValueError("input_ids_batch must be non-empty")
    lengths = {len(ids) for ids in input_ids_batch}
    if len(lengths) != 1:
        raise ValueError("Needle-sensitivity batches must have equal sequence lengths")
    seq_len = next(iter(lengths))
    if target_position < 0 or target_position >= seq_len:
        raise ValueError(
            f"target_position={target_position} is outside sequence length {seq_len}"
        )
    device = model.get_input_embeddings().weight.device
    inputs = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
    with torch.inference_mode():
        out = _forward_with_hidden_states_no_cache(model, inputs)
    hidden_states = list(out.hidden_states)
    vectors = {
        int(layer): (
            hidden_states[int(layer)][:, int(target_position), :]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        for layer in layers
    }
    del hidden_states
    del out
    del inputs
    release_torch_memory(collect_garbage=True)
    return vectors


def _normalize_vector(x: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, float]:
    arr = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= eps or not math.isfinite(norm):
        return np.zeros_like(arr, dtype=np.float32), norm
    return (arr / norm).astype(np.float32), norm


def _memory_summary_row(stage: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage,
        "cuda_available": bool(snapshot.get("cuda_available")),
        "device_count": int(snapshot.get("device_count", 0) or 0),
    }
    devices = list(snapshot.get("devices", []) or [])
    if devices:
        first = dict(devices[0])
        row.update(
            {
                "device_name": first.get("name"),
                "allocated_bytes": first.get("allocated_bytes"),
                "reserved_bytes": first.get("reserved_bytes"),
                "max_allocated_bytes": first.get("max_allocated_bytes"),
                "max_reserved_bytes": first.get("max_reserved_bytes"),
                "free_bytes": first.get("free_bytes"),
                "total_bytes": first.get("total_bytes"),
                "allocated_gib": first.get("allocated_gib"),
                "reserved_gib": first.get("reserved_gib"),
                "max_allocated_gib": first.get("max_allocated_gib"),
                "max_reserved_gib": first.get("max_reserved_gib"),
                "free_gib": first.get("free_gib"),
                "total_gib": first.get("total_gib"),
            }
        )
    else:
        row["device_name"] = "cpu"
    return row


def _plot_needle_sensitivity_summaries(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    figure_dir: Path,
    layer: int,
) -> list[str]:
    rows = [row for row in summary_rows if int(row.get("layer", layer)) == int(layer)]
    if not rows:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths: list[str] = []
    styles = {
        True: {"label": "successful", "marker": "o", "color": "#1f77b4"},
        False: {"label": "failed", "marker": "^", "color": "#d62728"},
    }
    for metric, filename, ylabel in [
        (
            "dist_sensitivity",
            f"dist_sensitivity_by_count_layer_{int(layer)}.png",
            "distance sensitivity",
        ),
        (
            "mean_cosine_drop",
            f"cosine_drop_by_count_layer_{int(layer)}.png",
            "mean cosine drop",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        for exact in (True, False):
            subset = [
                row for row in rows if _is_true(row.get("model_exact_match")) is exact
            ]
            if not subset:
                continue
            style = styles[exact]
            ax.scatter(
                [float(row["gold_count"]) for row in subset],
                [float(row[metric]) for row in subset],
                s=28,
                alpha=0.78,
                marker=style["marker"],
                color=style["color"],
                label=f"{style['label']} (n={len(subset)})",
            )
        ax.set_xlabel("gold count")
        ax.set_ylabel(ylabel)
        ax.set_title(f"needle sensitivity L{int(layer)}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = figure_dir / filename
        fig.savefig(out)
        plt.close(fig)
        paths.append(str(out))

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    data = []
    labels = []
    for exact in (True, False):
        subset = [
            float(row["dist_sensitivity"])
            for row in rows
            if _is_true(row.get("model_exact_match")) is exact
        ]
        if subset:
            data.append(subset)
            labels.append("successful" if exact else "failed")
    if data:
        ax.boxplot(data)
        ax.set_xticklabels(labels)
        ax.set_ylabel("distance sensitivity")
        ax.set_title(f"success/failure sensitivity L{int(layer)}")
        fig.tight_layout()
        out = figure_dir / f"success_failure_sensitivity_layer_{int(layer)}.png"
        fig.savefig(out)
        paths.append(str(out))
    plt.close(fig)
    return paths


def run_needle_sensitivity_analysis(
    *,
    model: Any,
    rows: Sequence[dict[str, Any]],
    examples: Sequence[TokenizedCountingExample],
    layers: Sequence[int],
    output_tables_dir: str | Path,
    output_tensors_dir: str | Path,
    output_figures_dir: str | Path | None = None,
    num_removal: int = 3,
    target_position: str = "last-token",
    seed: int = 42,
    max_examples: int | None = None,
    scored_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure target-position hidden-state sensitivity to single-needle removal.

    The implementation streams one example at a time and saves only selected
    layer/position vectors or scalar summaries, not full sequence hidden states.
    """

    selected_layers = [int(layer) for layer in layers]
    if not selected_layers:
        raise ValueError("At least one layer is required for needle sensitivity")
    num_removal = int(num_removal)
    if num_removal <= 0:
        raise ValueError(f"num_removal must be positive, got {num_removal}")
    if target_position != "last-token":
        raise ValueError(
            "run_needle_sensitivity_analysis currently supports only "
            f"target_position='last-token', got {target_position!r}"
        )
    if len(rows) != len(examples):
        raise ValueError(
            f"rows/examples length mismatch: {len(rows)} rows vs {len(examples)} examples"
        )

    table_dir = _as_path(output_tables_dir) / "needle_sensitivity"
    tensor_dir = _as_path(output_tensors_dir) / "needle_sensitivity"
    figure_dir = (
        None
        if output_figures_dir is None
        else _as_path(output_figures_dir) / "needle_sensitivity"
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)

    timing = StageTimer(
        json_path=table_dir / "sensitivity_timing_summary.json",
        csv_path=table_dir / "sensitivity_timing_summary.csv",
    )
    memory_rows: list[dict[str, Any]] = []

    def record_memory(stage: str) -> None:
        snapshot = torch_gpu_memory_snapshot()
        memory_rows.append(_memory_summary_row(stage, snapshot))
        print_gpu_memory_snapshot(f"needle_sensitivity:{stage}", snapshot)

    scored_by_id = {
        _row_id(row, idx): row for idx, row in enumerate(scored_rows or [])
    }
    example_rows: list[dict[str, Any]] = []
    removal_rows_by_layer: dict[int, list[dict[str, Any]]] = {
        layer: [] for layer in selected_layers
    }
    summary_rows_by_layer: dict[int, list[dict[str, Any]]] = {
        layer: [] for layer in selected_layers
    }
    mean_vectors_by_layer: dict[int, list[np.ndarray]] = {
        layer: [] for layer in selected_layers
    }
    tensor_indices_by_layer: dict[int, list[int]] = {
        layer: [] for layer in selected_layers
    }
    tensor_row_ids_by_layer: dict[int, list[str]] = {
        layer: [] for layer in selected_layers
    }
    processed = 0
    skipped_reasons: Counter[str] = Counter()
    fallback_single_forward_count = 0
    metadata_path = table_dir / "sensitivity_metadata.json"

    with timing.stage("needle_sensitivity_total"):
        record_memory("start")
        with timing.stage("example_selection_and_perturbation"):
            candidates: list[dict[str, Any]] = []
            for local_idx, (row, example) in enumerate(zip(rows, examples)):
                row_id = _row_id(row, local_idx)
                scored = scored_by_id.get(row_id)
                exact_match = (
                    row.get("exact_match")
                    if "exact_match" in row
                    else (None if scored is None else scored.get("exact_match"))
                )
                gold_count = count_probe_target_for_row(row)
                segments = _matching_segments(
                    example.needle_segments,
                    example.matching_needle_ids,
                    gold_count,
                )
                exclude_reason = ""
                if gold_count is None:
                    exclude_reason = "missing_gold_count"
                elif len(segments) < num_removal:
                    exclude_reason = "fewer_than_num_removal_needles"
                else:
                    for segment in segments:
                        start = int(segment["start"])
                        end = int(segment["end"])
                        if start < 0 or end > example.sequence_length or end <= start:
                            exclude_reason = "invalid_span"
                            break
                audit = {
                    "row_id": row_id,
                    "local_example_index": int(local_idx),
                    "gold_count": gold_count,
                    "model_exact_match": exact_match,
                    "num_matching_spans": len(segments),
                    "num_requested_removals": int(num_removal),
                    "num_valid_removals": 0,
                    "selected_needle_ids": [],
                    "selected_occurrence_indices": [],
                    "included": not bool(exclude_reason),
                    "exclude_reason": exclude_reason,
                }
                if exclude_reason:
                    skipped_reasons[exclude_reason] += 1
                    example_rows.append(audit)
                    continue
                rng = np.random.default_rng(
                    int(seed) + int(local_idx) * 100_003 + _stable_seed_component(row_id)
                )
                selected_indices = sorted(
                    rng.choice(len(segments), size=num_removal, replace=False).tolist()
                )
                removals: list[dict[str, Any]] = []
                for removal_idx, seg_idx in enumerate(selected_indices):
                    segment = dict(segments[int(seg_idx)])
                    span_start = int(segment["start"])
                    span_end = int(segment["end"])
                    span_len = span_end - span_start
                    candidates_for_span = _non_needle_replacement_candidates(
                        example.sequence_length,
                        example.needle_segments,
                        span_len,
                    )
                    if not candidates_for_span:
                        continue
                    repl_start, repl_end = candidates_for_span[
                        int(rng.integers(0, len(candidates_for_span)))
                    ]
                    perturbed_ids = list(example.input_ids)
                    perturbed_ids[span_start:span_end] = example.input_ids[
                        repl_start:repl_end
                    ]
                    if len(perturbed_ids) != example.sequence_length:
                        raise AssertionError("Needle sensitivity replacement changed length")
                    removals.append(
                        {
                            "removal_index": int(removal_idx),
                            "segment_index": int(seg_idx),
                            "segment": segment,
                            "replacement_start": int(repl_start),
                            "replacement_end": int(repl_end),
                            "input_ids": perturbed_ids,
                        }
                    )
                if not removals:
                    audit["included"] = False
                    audit["exclude_reason"] = "no_valid_replacement_segments"
                    skipped_reasons["no_valid_replacement_segments"] += 1
                    example_rows.append(audit)
                    continue
                audit["num_valid_removals"] = len(removals)
                audit["selected_needle_ids"] = [
                    removal["segment"].get("needle_id", "") for removal in removals
                ]
                audit["selected_occurrence_indices"] = [
                    int(removal["segment_index"]) + 1 for removal in removals
                ]
                example_rows.append(audit)
                candidates.append(
                    {
                        "local_example_index": int(local_idx),
                        "row": row,
                        "row_id": row_id,
                        "example": example,
                        "gold_count": int(gold_count),
                        "model_exact_match": exact_match,
                        "removals": removals,
                    }
                )
            if max_examples is not None:
                candidates = candidates[: int(max_examples)]
                allowed = {
                    (item["row_id"], int(item["local_example_index"]))
                    for item in candidates
                }
                for audit in example_rows:
                    key = (str(audit["row_id"]), int(audit["local_example_index"]))
                    if audit["included"] and key not in allowed:
                        audit["included"] = False
                        audit["exclude_reason"] = "max_sensitivity_examples_cap"
                        skipped_reasons["max_sensitivity_examples_cap"] += 1
            record_memory("after_example_selection")

        for item in candidates:
            processed += 1
            local_idx = int(item["local_example_index"])
            row_id = str(item["row_id"])
            example = item["example"]
            target_idx = int(example.sequence_length) - 1
            with timing.stage("original_forward", row_id=row_id, local_example_index=local_idx):
                original_by_layer = _extract_target_vectors(
                    model,
                    [example.input_ids],
                    layers=selected_layers,
                    target_position=target_idx,
                )
                record_memory(f"after_original_forward_{local_idx}")
            perturbed_inputs = [removal["input_ids"] for removal in item["removals"]]
            try:
                with timing.stage(
                    "perturbed_forward_batch",
                    row_id=row_id,
                    local_example_index=local_idx,
                    batch_size=len(perturbed_inputs),
                ):
                    perturbed_by_layer = _extract_target_vectors(
                        model,
                        perturbed_inputs,
                        layers=selected_layers,
                        target_position=target_idx,
                    )
                    record_memory(f"after_perturbed_batch_{local_idx}")
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                fallback_single_forward_count += 1
                release_torch_memory(collect_garbage=True)
                arrays = {layer: [] for layer in selected_layers}
                with timing.stage(
                    "perturbed_forward_single_fallback",
                    row_id=row_id,
                    local_example_index=local_idx,
                    num_removals=len(perturbed_inputs),
                ):
                    for perturbed_ids in perturbed_inputs:
                        one = _extract_target_vectors(
                            model,
                            [perturbed_ids],
                            layers=selected_layers,
                            target_position=target_idx,
                        )
                        for layer in selected_layers:
                            arrays[layer].append(one[layer][0])
                        del one
                        release_torch_memory(collect_garbage=True)
                    perturbed_by_layer = {
                        layer: np.stack(arrays[layer], axis=0).astype(np.float32)
                        for layer in selected_layers
                    }
                    record_memory(f"after_perturbed_single_fallback_{local_idx}")

            with timing.stage("metric_calculation", row_id=row_id, local_example_index=local_idx):
                for layer in selected_layers:
                    h0_raw = original_by_layer[layer][0]
                    h0, original_norm = _normalize_vector(h0_raw)
                    perturbed_normed = []
                    perturbed_norms = []
                    distances = []
                    cosine_similarities = []
                    deltas = []
                    for removal_idx, h_raw in enumerate(perturbed_by_layer[layer]):
                        hr, perturbed_norm = _normalize_vector(h_raw)
                        delta = h0 - hr
                        distance = float(np.linalg.norm(delta))
                        cosine = float(np.dot(h0, hr))
                        perturbed_normed.append(hr)
                        perturbed_norms.append(float(perturbed_norm))
                        distances.append(distance)
                        cosine_similarities.append(cosine)
                        deltas.append(delta)
                        removal = item["removals"][removal_idx]
                        segment = removal["segment"]
                        removal_rows_by_layer[layer].append(
                            {
                                "layer": int(layer),
                                "row_id": row_id,
                                "local_example_index": local_idx,
                                "gold_count": int(item["gold_count"]),
                                "model_exact_match": item.get("model_exact_match"),
                                "target_position": int(target_idx),
                                "removal_index": int(removal["removal_index"]),
                                "removed_needle_id": segment.get("needle_id", ""),
                                "removed_occurrence_index": int(removal["segment_index"]) + 1,
                                "removed_span_start": int(segment["start"]),
                                "removed_span_end": int(segment["end"]),
                                "replacement_start": int(removal["replacement_start"]),
                                "replacement_end": int(removal["replacement_end"]),
                                "original_norm": float(original_norm),
                                "perturbed_norm": float(perturbed_norm),
                                "l2_distance": distance,
                                "cosine_similarity": cosine,
                                "cosine_drop": float(1.0 - cosine),
                            }
                        )
                    perturbed_arr = np.stack(perturbed_normed, axis=0).astype(np.float32)
                    mean_sensitivity_vector = h0 - perturbed_arr.mean(axis=0)
                    mean_delta_norm = float(np.linalg.norm(mean_sensitivity_vector))
                    mean_distance = float(np.mean(distances))
                    directional_consistency = (
                        mean_delta_norm / mean_distance
                        if mean_distance > 1e-12
                        else float("nan")
                    )
                    summary_rows_by_layer[layer].append(
                        {
                            "layer": int(layer),
                            "row_id": row_id,
                            "local_example_index": local_idx,
                            "gold_count": int(item["gold_count"]),
                            "model_exact_match": item.get("model_exact_match"),
                            "target_position": int(target_idx),
                            "num_valid_removals": int(len(distances)),
                            "mean_sensitivity_norm": mean_delta_norm,
                            "dist_sensitivity": float(
                                math.sqrt(float(np.mean(np.square(distances))))
                            ),
                            "mean_cosine_drop": float(
                                np.mean([1.0 - v for v in cosine_similarities])
                            ),
                            "max_removal_distance": float(np.max(distances)),
                            "min_removal_distance": float(np.min(distances)),
                            "removal_distance_std": float(np.std(distances)),
                            "directional_consistency": float(directional_consistency),
                            "original_norm": float(original_norm),
                            "mean_perturbed_norm": float(np.mean(perturbed_norms)),
                        }
                    )
                    mean_vectors_by_layer[layer].append(
                        mean_sensitivity_vector.astype(np.float32)
                    )
                    tensor_indices_by_layer[layer].append(local_idx)
                    tensor_row_ids_by_layer[layer].append(row_id)
                del original_by_layer
                del perturbed_by_layer
                release_torch_memory(collect_garbage=True)
                record_memory(f"after_metric_calculation_{local_idx}")

        with timing.stage("serialization"):
            examples_path = _write_dict_rows_csv(
                table_dir / "sensitivity_examples.csv", example_rows
            )
            summary_paths: dict[int, str] = {}
            removal_paths: dict[int, str] = {}
            tensor_paths: dict[int, str] = {}
            figure_paths: dict[int, list[str]] = {}
            for layer in selected_layers:
                removal_paths[layer] = str(
                    _write_dict_rows_csv(
                        table_dir / f"sensitivity_removals_layer_{int(layer)}.csv",
                        removal_rows_by_layer[layer],
                    )
                )
                summary_paths[layer] = str(
                    _write_dict_rows_csv(
                        table_dir / f"sensitivity_summary_layer_{int(layer)}.csv",
                        summary_rows_by_layer[layer],
                    )
                )
                if mean_vectors_by_layer[layer]:
                    payload = {
                        "layer": int(layer),
                        "row_ids": tensor_row_ids_by_layer[layer],
                        "local_example_indices": tensor_indices_by_layer[layer],
                        "mean_sensitivity_vectors": torch.from_numpy(
                            np.stack(mean_vectors_by_layer[layer], axis=0).astype(np.float32)
                        ),
                        "dist_sensitivity": torch.tensor(
                            [
                                float(row["dist_sensitivity"])
                                for row in summary_rows_by_layer[layer]
                            ],
                            dtype=torch.float32,
                        ),
                        "target_position": target_position,
                        "normalized": True,
                    }
                    tensor_path = tensor_dir / f"mean_sensitivity_layer_{int(layer)}.pt"
                    torch.save(payload, tensor_path)
                    tensor_paths[layer] = str(tensor_path)
                if figure_dir is not None:
                    figure_paths[layer] = _plot_needle_sensitivity_summaries(
                        summary_rows=summary_rows_by_layer[layer],
                        figure_dir=figure_dir,
                        layer=layer,
                    )
            memory_path = _write_dict_rows_csv(
                table_dir / "sensitivity_memory_summary.csv", memory_rows
            )
            metadata = {
                "schema_version": "needle_sensitivity_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "layers": selected_layers,
                "num_removal": int(num_removal),
                "target_position": target_position,
                "seed": int(seed),
                "max_examples": max_examples,
                "num_input_rows": len(rows),
                "num_processed_examples": int(processed),
                "num_excluded_examples": sum(
                    1 for row in example_rows if not row["included"]
                ),
                "exclude_reason_counts": dict(skipped_reasons),
                "fallback_single_forward_count": int(fallback_single_forward_count),
                "examples_path": str(examples_path),
                "summary_paths": {str(k): v for k, v in summary_paths.items()},
                "removal_paths": {str(k): v for k, v in removal_paths.items()},
                "tensor_paths": {str(k): v for k, v in tensor_paths.items()},
                "figure_paths": {str(k): v for k, v in figure_paths.items()},
                "timing_json_path": str(table_dir / "sensitivity_timing_summary.json"),
                "timing_csv_path": str(table_dir / "sensitivity_timing_summary.csv"),
                "memory_path": str(memory_path),
            }
            metadata_path = write_json(table_dir / "sensitivity_metadata.json", metadata)
            record_memory("after_serialization")

    timing.save_json(table_dir / "sensitivity_timing_summary.json")
    timing.save_csv(table_dir / "sensitivity_timing_summary.csv")
    _write_dict_rows_csv(table_dir / "sensitivity_memory_summary.csv", memory_rows)
    return {
        "table_dir": str(table_dir),
        "tensor_dir": str(tensor_dir),
        "figure_dir": None if figure_dir is None else str(figure_dir),
        "metadata_path": str(metadata_path),
        "num_processed_examples": int(processed),
        "exclude_reason_counts": dict(skipped_reasons),
    }


def _contrastive_hidden_matrix(
    hidden: torch.Tensor,
    examples: Sequence[TokenizedCountingExample],
    *,
    position_mode: str,
) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden tensor [n, T, d], got {tuple(hidden.shape)}")
    if hidden.shape[0] != len(examples):
        raise ValueError(
            f"Hidden tensor has {hidden.shape[0]} examples but metadata has {len(examples)}"
    )
    rows: list[torch.Tensor] = []
    for row_idx, example in enumerate(examples):
        pos = contrastive_feature_position_index(example, position_mode=position_mode)
        if pos < 0 or pos >= hidden.shape[1]:
            raise ValueError(
                f"Invalid contrastive position {pos} for example {row_idx} "
                f"with hidden sequence length {hidden.shape[1]}"
            )
        row = hidden[row_idx, pos, :].detach().to(dtype=torch.float32)
        if not torch.isfinite(row).all():
            raise ValueError(
                f"Contrastive hidden state contains non-finite values for example {row_idx}"
            )
        rows.append(row)
    return torch.stack(rows, dim=0)


def fit_contrastive_success_direction(
    hidden: torch.Tensor,
    examples: Sequence[TokenizedCountingExample],
    labels: Sequence[str],
    *,
    layer: int,
    position_mode: str = "last",
) -> ContrastiveSuccessDirection:
    """Compute raw mean(success) - mean(failure) at the selected token position."""

    if len(labels) != len(examples):
        raise ValueError(
            f"labels length must match examples length, got {len(labels)} and {len(examples)}"
        )
    selected_hidden = _contrastive_hidden_matrix(
        hidden, examples, position_mode=position_mode
    )
    success_mask = torch.tensor(
        [str(label) == "successful" for label in labels], dtype=torch.bool
    )
    failure_mask = torch.tensor(
        [str(label) == "unsuccessful" for label in labels], dtype=torch.bool
    )
    if int(success_mask.sum()) == 0 or int(failure_mask.sum()) == 0:
        raise ValueError("Contrastive direction requires both successful and unsuccessful examples")
    successful = selected_hidden[success_mask]
    unsuccessful = selected_hidden[failure_mask]
    successful_mean = successful.mean(dim=0)
    unsuccessful_mean = unsuccessful.mean(dim=0)
    raw_direction = successful_mean - unsuccessful_mean
    raw_norm = float(torch.linalg.vector_norm(raw_direction).item())
    if not torch.isfinite(torch.tensor(raw_norm)) or raw_norm <= 0.0:
        raise ValueError(f"Contrastive success direction has invalid norm: {raw_norm}")
    direction = raw_direction / raw_norm
    projections = selected_hidden @ direction
    success_proj = projections[success_mask]
    failure_proj = projections[failure_mask]
    margin = float(success_proj.mean().item() - failure_proj.mean().item())
    pooled_std = float(projections.std(unbiased=False).item()) if projections.numel() > 1 else 0.0
    metrics = {
        "layer": int(layer),
        "method": "raw_mean_difference",
        "position": str(position_mode),
        "num_successful": int(success_mask.sum().item()),
        "num_unsuccessful": int(failure_mask.sum().item()),
        "raw_norm": raw_norm,
        "successful_projection_mean": float(success_proj.mean().item()),
        "successful_projection_std": (
            float(success_proj.std(unbiased=False).item())
            if success_proj.numel() > 1
            else 0.0
        ),
        "unsuccessful_projection_mean": float(failure_proj.mean().item()),
        "unsuccessful_projection_std": (
            float(failure_proj.std(unbiased=False).item())
            if failure_proj.numel() > 1
            else 0.0
        ),
        "projection_margin": margin,
        "projection_pooled_std": pooled_std,
        "projection_margin_over_pooled_std": (
            margin / pooled_std if pooled_std > 0 else float("nan")
        ),
    }
    return ContrastiveSuccessDirection(
        layer=int(layer),
        direction=direction.detach().cpu(),
        raw_direction=raw_direction.detach().cpu(),
        raw_norm=raw_norm,
        successful_mean=successful_mean.detach().cpu(),
        unsuccessful_mean=unsuccessful_mean.detach().cpu(),
        metrics=metrics,
    )


def save_contrastive_success_direction(
    result: ContrastiveSuccessDirection,
    output_dir: str | Path,
    *,
    selected_dataset_indices: Sequence[int],
    selected_row_ids: Sequence[str],
    labels: Sequence[str],
    position_indices: Sequence[int] | None = None,
    prefix: str = "contrastive_success",
) -> Path:
    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_layer_{int(result.layer)}.pt"
    torch.save(
        {
            "layer": int(result.layer),
            "direction": result.direction,
            "raw_direction": result.raw_direction,
            "raw_norm": result.raw_norm,
            "successful_mean": result.successful_mean,
            "unsuccessful_mean": result.unsuccessful_mean,
            "position": result.metrics.get("position", "last"),
            "method": "raw_mean_difference",
            "selected_dataset_indices": [int(x) for x in selected_dataset_indices],
            "selected_row_ids": [str(x) for x in selected_row_ids],
            "selected_labels": [str(x) for x in labels],
            "selected_position_indices": (
                None
                if position_indices is None
                else [int(x) for x in position_indices]
            ),
            "metrics": result.metrics,
        },
        path,
    )
    return path


def fit_counterfactual_count_direction(
    original_hidden: torch.Tensor,
    original_examples: Sequence[TokenizedCountingExample],
    counterfactual_hidden: torch.Tensor,
    counterfactual_examples: Sequence[TokenizedCountingExample],
    *,
    layer: int,
    position_mode: str = "last",
) -> CounterfactualCountDirection:
    """Compute mean(original success) - mean(counterfactual success)."""

    original_selected = _contrastive_hidden_matrix(
        original_hidden, original_examples, position_mode=position_mode
    )
    counterfactual_selected = _contrastive_hidden_matrix(
        counterfactual_hidden, counterfactual_examples, position_mode=position_mode
    )
    if original_selected.numel() == 0 or counterfactual_selected.numel() == 0:
        raise ValueError("Counterfactual direction requires non-empty hidden matrices")
    if original_selected.shape[1] != counterfactual_selected.shape[1]:
        raise ValueError(
            "Original and counterfactual hidden dimensions do not match: "
            f"{original_selected.shape[1]} vs {counterfactual_selected.shape[1]}"
        )

    original_mean = original_selected.mean(dim=0)
    counterfactual_mean = counterfactual_selected.mean(dim=0)
    raw_direction = original_mean - counterfactual_mean
    raw_norm = float(torch.linalg.vector_norm(raw_direction).item())
    if not torch.isfinite(torch.tensor(raw_norm)) or raw_norm <= 0.0:
        raise ValueError(f"Counterfactual count direction has invalid norm: {raw_norm}")
    direction = raw_direction / raw_norm
    original_proj = original_selected @ direction
    counterfactual_proj = counterfactual_selected @ direction
    margin = float(original_proj.mean().item() - counterfactual_proj.mean().item())
    pooled = torch.cat([original_proj, counterfactual_proj], dim=0)
    pooled_std = float(pooled.std(unbiased=False).item()) if pooled.numel() > 1 else 0.0
    metrics = {
        "layer": int(layer),
        "method": "counterfactual_count_difference",
        "position": str(position_mode),
        "num_original_successful": int(original_selected.shape[0]),
        "num_counterfactual_successful": int(counterfactual_selected.shape[0]),
        "raw_norm": raw_norm,
        "original_projection_mean": float(original_proj.mean().item()),
        "original_projection_std": (
            float(original_proj.std(unbiased=False).item())
            if original_proj.numel() > 1
            else 0.0
        ),
        "counterfactual_projection_mean": float(counterfactual_proj.mean().item()),
        "counterfactual_projection_std": (
            float(counterfactual_proj.std(unbiased=False).item())
            if counterfactual_proj.numel() > 1
            else 0.0
        ),
        "projection_margin": margin,
        "projection_pooled_std": pooled_std,
        "projection_margin_over_pooled_std": (
            margin / pooled_std if pooled_std > 0 else float("nan")
        ),
    }
    return CounterfactualCountDirection(
        layer=int(layer),
        direction=direction.detach().cpu(),
        raw_direction=raw_direction.detach().cpu(),
        raw_norm=raw_norm,
        original_mean=original_mean.detach().cpu(),
        counterfactual_mean=counterfactual_mean.detach().cpu(),
        metrics=metrics,
    )


def save_counterfactual_count_direction(
    result: CounterfactualCountDirection,
    output_dir: str | Path,
    *,
    original_dataset_indices: Sequence[int],
    original_row_ids: Sequence[str],
    original_position_indices: Sequence[int],
    counterfactual_dataset_indices: Sequence[int],
    counterfactual_row_ids: Sequence[str],
    counterfactual_position_indices: Sequence[int],
    original_gold_count: int | None = None,
    counterfactual_gold_count: int | None = None,
    prefix: str = "counterfactual_count",
) -> Path:
    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_layer_{int(result.layer)}.pt"
    torch.save(
        {
            "layer": int(result.layer),
            "direction": result.direction,
            "raw_direction": result.raw_direction,
            "raw_norm": result.raw_norm,
            "original_mean": result.original_mean,
            "counterfactual_mean": result.counterfactual_mean,
            "position": result.metrics.get("position", "last"),
            "method": "counterfactual_count_difference",
            "original_gold_count": original_gold_count,
            "counterfactual_gold_count": counterfactual_gold_count,
            "original_dataset_indices": [int(x) for x in original_dataset_indices],
            "original_row_ids": [str(x) for x in original_row_ids],
            "original_position_indices": [int(x) for x in original_position_indices],
            "counterfactual_dataset_indices": [
                int(x) for x in counterfactual_dataset_indices
            ],
            "counterfactual_row_ids": [str(x) for x in counterfactual_row_ids],
            "counterfactual_position_indices": [
                int(x) for x in counterfactual_position_indices
            ],
            "metrics": result.metrics,
        },
        path,
    )
    return path


def _valid_mask_for_examples(
    target: torch.Tensor, example_indices: Sequence[int] | None = None
) -> np.ndarray:
    mask = torch.isfinite(target)
    if example_indices is not None:
        ex_mask = torch.zeros(target.shape[0], dtype=torch.bool)
        ex_mask[list(example_indices)] = True
        mask &= ex_mask[:, None]
    return mask.cpu().numpy().reshape(-1)


def sample_probe_positions(
    target: torch.Tensor,
    *,
    example_indices: Sequence[int] | None = None,
    needle_mask: torch.Tensor | None = None,
    max_tokens: int | None = DEFAULT_MAX_TRAIN_TOKENS,
    seed: int = 42,
    balanced: bool = True,
) -> ProbeSample:
    """Subsample valid token positions, keeping needle tokens when possible.

    The default cap of 200k token positions bounds memory/runtime for Colab A100
    runs while preserving all matching-needle spans and balancing off-needle
    samples across integer count values when possible.
    """

    valid = _valid_mask_for_examples(target, example_indices)
    all_indices = np.flatnonzero(valid)
    if max_tokens is None or len(all_indices) <= int(max_tokens):
        chosen = all_indices
    else:
        rng = np.random.default_rng(int(seed))
        keep: np.ndarray
        if needle_mask is not None:
            needle_flat = needle_mask.cpu().numpy().reshape(-1) & valid
            keep = np.flatnonzero(needle_flat)
        else:
            keep = np.asarray([], dtype=np.int64)
        max_tokens_int = int(max_tokens)
        if keep.size > max_tokens_int:
            chosen = np.sort(rng.choice(keep, size=max_tokens_int, replace=False))
        else:
            remaining_budget = max_tokens_int - keep.size
            keep_set = set(int(x) for x in keep.tolist())
            pool = np.asarray(
                [idx for idx in all_indices if int(idx) not in keep_set], dtype=np.int64
            )
            if balanced and pool.size:
                y = target.cpu().numpy().reshape(-1)[pool]
                labels = np.floor(y + 1e-6).astype(int)
                classes = np.unique(labels)
                per_class = max(1, math.ceil(remaining_budget / max(1, len(classes))))
                pieces: list[np.ndarray] = []
                for cls in classes:
                    cls_pool = pool[labels == cls]
                    take = min(per_class, cls_pool.size)
                    if take:
                        pieces.append(rng.choice(cls_pool, size=take, replace=False))
                sampled = (
                    np.concatenate(pieces) if pieces else np.asarray([], dtype=np.int64)
                )
                if sampled.size > remaining_budget:
                    sampled = rng.choice(sampled, size=remaining_budget, replace=False)
                elif sampled.size < remaining_budget:
                    sampled_set = set(int(x) for x in sampled.tolist())
                    leftover = np.asarray(
                        [idx for idx in pool if int(idx) not in sampled_set],
                        dtype=np.int64,
                    )
                    if leftover.size:
                        extra = rng.choice(
                            leftover,
                            size=min(remaining_budget - sampled.size, leftover.size),
                            replace=False,
                        )
                        sampled = np.concatenate([sampled, extra])
            else:
                sampled = rng.choice(
                    pool, size=min(remaining_budget, pool.size), replace=False
                )
            chosen = np.sort(np.concatenate([keep, sampled]))
    n_cols = target.shape[1]
    ex_idx = chosen // n_cols
    tok_idx = chosen % n_cols
    return ProbeSample(
        flat_indices=chosen.astype(np.int64),
        example_indices=ex_idx.astype(np.int64),
        token_indices=tok_idx.astype(np.int64),
        num_available=int(all_indices.size),
        num_sampled=int(chosen.size),
        max_tokens=max_tokens,
        seed=int(seed),
    )


def gather_flat_features(
    hidden: torch.Tensor, target: torch.Tensor, sample: ProbeSample
) -> tuple[np.ndarray, np.ndarray]:
    x = (
        hidden[sample.example_indices, sample.token_indices, :]
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )
    y = (
        target[sample.example_indices, sample.token_indices]
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[finite], y[finite]


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return ((x - mean) / scale).astype(np.float32), mean, scale


def _standardize_apply(
    x: np.ndarray, mean: np.ndarray | None, scale: np.ndarray | None
) -> np.ndarray:
    if mean is None or scale is None:
        return x.astype(np.float32, copy=False)
    return ((x - mean) / scale).astype(np.float32)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered**2))
    corr = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if y_true.size > 1 and np.std(y_pred) > 0
        else float("nan")
    )
    return {
        "num_tokens": int(y_true.size),
        "mse": float(np.mean(err**2)) if y_true.size else float("nan"),
        "mae": float(np.mean(np.abs(err))) if y_true.size else float("nan"),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "pearson_corr": corr,
    }


def fit_ridge_probe(
    hidden: torch.Tensor,
    target: torch.Tensor,
    *,
    train_example_indices: Sequence[int] | None = None,
    needle_mask: torch.Tensor | None = None,
    alpha: float = 1.0,
    max_train_tokens: int | None = DEFAULT_MAX_TRAIN_TOKENS,
    seed: int = 42,
    standardize: bool = True,
) -> RidgeProbeResult:
    sample = sample_probe_positions(
        target,
        example_indices=train_example_indices,
        needle_mask=needle_mask,
        max_tokens=max_train_tokens,
        seed=seed,
    )
    x, y = gather_flat_features(hidden, target, sample)
    if x.size == 0:
        raise ValueError(
            "No finite hidden-state/target rows available for ridge fitting"
        )
    if standardize:
        x_fit, mean, scale = _standardize_fit(x)
    else:
        x_fit, mean, scale = x.astype(np.float32, copy=False), None, None
    y_mean = float(y.mean())
    yc = (y - y_mean).astype(np.float32)
    xtx = x_fit.T @ x_fit
    xtx.flat[:: xtx.shape[0] + 1] += float(alpha)
    xty = x_fit.T @ yc
    coef = np.linalg.solve(xtx.astype(np.float64), xty.astype(np.float64)).astype(
        np.float32
    )
    pred = x_fit @ coef + y_mean
    metrics = regression_metrics(y, pred) | {
        "sample_num_available": sample.num_available,
        "sample_num_used": sample.num_sampled,
        "max_train_tokens": -1 if max_train_tokens is None else int(max_train_tokens),
    }
    return RidgeProbeResult(
        coef=coef,
        intercept=y_mean,
        feature_mean=mean,
        feature_scale=scale,
        alpha=float(alpha),
        standardize=bool(standardize),
        metrics=metrics,
    )


def predict_ridge(result: RidgeProbeResult, x: np.ndarray) -> np.ndarray:
    x_eval = _standardize_apply(
        x.astype(np.float32, copy=False), result.feature_mean, result.feature_scale
    )
    return x_eval @ result.coef + float(result.intercept)


def evaluate_ridge_probe(
    result: RidgeProbeResult,
    hidden: torch.Tensor,
    target: torch.Tensor,
    *,
    example_indices: Sequence[int] | None = None,
    max_tokens: int | None = DEFAULT_MAX_TRAIN_TOKENS,
    seed: int = 123,
) -> dict[str, float | int]:
    sample = sample_probe_positions(
        target,
        example_indices=example_indices,
        max_tokens=max_tokens,
        seed=seed,
        balanced=False,
    )
    x, y = gather_flat_features(hidden, target, sample)
    pred = predict_ridge(result, x)
    return regression_metrics(y, pred) | {
        "sample_num_available": sample.num_available,
        "sample_num_used": sample.num_sampled,
    }


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    class_to_idx = {int(c): i for i, c in enumerate(classes.astype(int).tolist())}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if int(yt) in class_to_idx and int(yp) in class_to_idx:
            cm[class_to_idx[int(yt)], class_to_idx[int(yp)]] += 1
    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else float("nan")
    f1s: list[float] = []
    for i in range(len(classes)):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    return {
        "num_tokens": int(y_true.size),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "confusion_matrix": cm.tolist(),
    }


def fit_classification_probe(
    hidden: torch.Tensor,
    target: torch.Tensor,
    *,
    train_example_indices: Sequence[int] | None = None,
    needle_mask: torch.Tensor | None = None,
    max_train_tokens: int | None = DEFAULT_MAX_TRAIN_TOKENS,
    seed: int = 42,
    standardize: bool = True,
    lr: float = 0.1,
    epochs: int = 200,
    l2_penalty: float = 1e-4,
) -> ClassificationProbeResult:
    sample = sample_probe_positions(
        target,
        example_indices=train_example_indices,
        needle_mask=needle_mask,
        max_tokens=max_train_tokens,
        seed=seed,
    )
    x, y_float = gather_flat_features(hidden, target, sample)
    if not np.allclose(y_float, np.round(y_float), atol=1e-5):
        raise ValueError(
            "Classification requires integer targets; disable it for interpolation targets"
        )
    y = np.round(y_float).astype(np.int64)
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(
            "Classification requires at least two classes in the training sample"
        )
    if standardize:
        x_fit, mean, scale = _standardize_fit(x)
    else:
        x_fit, mean, scale = x.astype(np.float32, copy=False), None, None
    class_to_idx = {int(cls): i for i, cls in enumerate(classes.tolist())}
    y_idx = np.asarray([class_to_idx[int(v)] for v in y], dtype=np.int64)
    xt = torch.from_numpy(x_fit)
    yt = torch.from_numpy(y_idx)
    torch.manual_seed(int(seed))
    model = torch.nn.Linear(xt.shape[1], len(classes))
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(l2_penalty)
    )
    losses: list[float] = []
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = model(xt)
        loss = torch.nn.functional.cross_entropy(logits, yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        pred_idx = model(xt).argmax(dim=1).cpu().numpy()
    pred = classes[pred_idx]
    metrics = classification_metrics(y, pred, classes) | {
        "sample_num_available": sample.num_available,
        "sample_num_used": sample.num_sampled,
        "max_train_tokens": -1 if max_train_tokens is None else int(max_train_tokens),
    }
    coef = model.weight.detach().cpu().numpy().astype(np.float32)
    intercept = model.bias.detach().cpu().numpy().astype(np.float32)
    return ClassificationProbeResult(
        coef=coef,
        intercept=intercept,
        classes=classes,
        feature_mean=mean,
        feature_scale=scale,
        l2_penalty=float(l2_penalty),
        standardize=bool(standardize),
        metrics=metrics,
        loss_history=losses,
    )


def predict_classification(
    result: ClassificationProbeResult, x: np.ndarray
) -> np.ndarray:
    x_eval = _standardize_apply(
        x.astype(np.float32, copy=False), result.feature_mean, result.feature_scale
    )
    logits = x_eval @ result.coef.T + result.intercept[None, :]
    return result.classes[np.argmax(logits, axis=1)]


def evaluate_classification_probe(
    result: ClassificationProbeResult,
    hidden: torch.Tensor,
    target: torch.Tensor,
    *,
    example_indices: Sequence[int] | None = None,
    max_tokens: int | None = DEFAULT_MAX_TRAIN_TOKENS,
    seed: int = 123,
) -> dict[str, Any]:
    sample = sample_probe_positions(
        target,
        example_indices=example_indices,
        max_tokens=max_tokens,
        seed=seed,
        balanced=False,
    )
    x, y_float = gather_flat_features(hidden, target, sample)
    if not np.allclose(y_float, np.round(y_float), atol=1e-5):
        raise ValueError("Classification evaluation requires integer targets")
    y = np.round(y_float).astype(np.int64)
    pred = predict_classification(result, x).astype(np.int64)
    return classification_metrics(y, pred, result.classes) | {
        "sample_num_available": sample.num_available,
        "sample_num_used": sample.num_sampled,
    }


def save_ridge_probe(
    result: RidgeProbeResult,
    output_dir: str | Path,
    *,
    layer: int,
    prefix: str = "ridge_probe",
) -> dict[str, Path]:
    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{prefix}_layer_{int(layer)}.pt"
    json_path = out_dir / f"{prefix}_layer_{int(layer)}_metrics.json"
    torch.save(
        {
            "coef": torch.from_numpy(result.coef),
            "intercept": result.intercept,
            "feature_mean": (
                None
                if result.feature_mean is None
                else torch.from_numpy(result.feature_mean)
            ),
            "feature_scale": (
                None
                if result.feature_scale is None
                else torch.from_numpy(result.feature_scale)
            ),
            "alpha": result.alpha,
            "standardize": result.standardize,
            "metrics": result.metrics,
        },
        pt_path,
    )
    write_json(
        json_path,
        {
            "layer": int(layer),
            "kind": "ridge",
            "alpha": result.alpha,
            "standardize": result.standardize,
            "coef_shape": list(result.coef.shape),
            "has_feature_mean": result.feature_mean is not None,
            "has_feature_scale": result.feature_scale is not None,
            "metrics": result.metrics,
        },
    )
    return {"probe": pt_path, "metrics": json_path}


def save_classification_probe(
    result: ClassificationProbeResult,
    output_dir: str | Path,
    *,
    layer: int,
    prefix: str = "classification_probe",
) -> dict[str, Path]:
    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{prefix}_layer_{int(layer)}.pt"
    json_path = out_dir / f"{prefix}_layer_{int(layer)}_metrics.json"
    torch.save(
        {
            "coef": torch.from_numpy(result.coef),
            "intercept": torch.from_numpy(result.intercept),
            "classes": torch.from_numpy(result.classes),
            "feature_mean": (
                None
                if result.feature_mean is None
                else torch.from_numpy(result.feature_mean)
            ),
            "feature_scale": (
                None
                if result.feature_scale is None
                else torch.from_numpy(result.feature_scale)
            ),
            "l2_penalty": result.l2_penalty,
            "standardize": result.standardize,
            "metrics": result.metrics,
            "loss_history": result.loss_history,
        },
        pt_path,
    )
    write_json(
        json_path,
        {
            "layer": int(layer),
            "kind": "classification",
            "classes": result.classes,
            "coef_shape": list(result.coef.shape),
            "intercept_shape": list(result.intercept.shape),
            "l2_penalty": result.l2_penalty,
            "standardize": result.standardize,
            "has_feature_mean": result.feature_mean is not None,
            "has_feature_scale": result.feature_scale is not None,
            "metrics": result.metrics,
            "loss_history": result.loss_history,
        },
    )
    return {"probe": pt_path, "metrics": json_path}


def count_probe_target_for_row(row: Mapping[str, Any]) -> int | None:
    """Return the count target for count-like Dynamic NIAH rows."""

    raw = (row.get("gold_answer") or {}).get("count")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _write_dict_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    csv_path = _as_path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return csv_path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, default=_json_default)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return csv_path


def _count_distribution(values: Sequence[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for value in values:
        out[int(value)] = out.get(int(value), 0) + 1
    return out


def _ridge_from_matrix(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    standardize: bool,
) -> RidgeProbeResult:
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D features, got shape {x.shape}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"Expected y shape {(x.shape[0],)}, got {y.shape}")
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[finite].astype(np.float32, copy=False)
    y = y[finite].astype(np.float32, copy=False)
    if x.shape[0] == 0:
        raise ValueError("No finite rows available for ridge fitting")
    if standardize:
        x_fit, mean, scale = _standardize_fit(x)
    else:
        x_fit, mean, scale = x, None, None
    y_mean = float(y.mean())
    yc = (y - y_mean).astype(np.float32)
    xtx = x_fit.T @ x_fit
    xtx.flat[:: xtx.shape[0] + 1] += float(alpha)
    xty = x_fit.T @ yc
    coef = np.linalg.solve(xtx.astype(np.float64), xty.astype(np.float64)).astype(
        np.float32
    )
    pred = x_fit @ coef + y_mean
    return RidgeProbeResult(
        coef=coef,
        intercept=y_mean,
        feature_mean=mean,
        feature_scale=scale,
        alpha=float(alpha),
        standardize=bool(standardize),
        metrics=regression_metrics(y, pred) | {"num_examples": int(y.size)},
    )


def _evaluate_matrix_ridge(
    result: RidgeProbeResult, x: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x_eval = x[finite].astype(np.float32, copy=False)
    y_eval = y[finite].astype(np.float32, copy=False)
    if x_eval.shape[0] == 0:
        return {
            "num_examples": 0,
            "mse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson_corr": float("nan"),
            "rounded_count_accuracy": float("nan"),
        }
    pred = predict_ridge(result, x_eval)
    return regression_metrics(y_eval, pred) | {
        "num_examples": int(y_eval.size),
        "rounded_count_accuracy": float(np.mean(np.rint(pred) == np.rint(y_eval))),
    }


def _example_probe_rows(
    rows: Sequence[dict[str, Any]],
    examples: Sequence[TokenizedCountingExample],
    scored_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    for local_idx, (row, example) in enumerate(zip(rows, examples)):
        row_id = str(row.get("id", example.row_id or local_idx))
        scored = None if scored_by_id is None else scored_by_id.get(row_id)
        exact_match = (
            row.get("exact_match")
            if "exact_match" in row
            else (None if scored is None else scored.get("exact_match"))
        )
        gold_count = count_probe_target_for_row(row)
        segments = _matching_segments(
            example.needle_segments,
            example.matching_needle_ids,
            gold_count,
        )
        exclude_reason = ""
        if gold_count is None:
            exclude_reason = "missing_gold_count"
        elif not segments:
            exclude_reason = "no_matching_spans"
        elif len(segments) != int(gold_count):
            exclude_reason = "matching_span_count_mismatch"
        else:
            for segment in segments:
                start = int(segment["start"])
                end = int(segment["end"])
                if start < 0 or end > example.sequence_length or end <= start:
                    exclude_reason = "invalid_span"
                    break
        final_pos = int(example.sequence_length) - 1
        audit = {
            "local_example_index": int(local_idx),
            "row_id": row_id,
            "gold_count": gold_count,
            "num_matching_spans": len(segments),
            "span_starts": [int(seg["start"]) for seg in segments],
            "span_ends": [int(seg["end"]) for seg in segments],
            "final_pos": final_pos,
            "model_exact_match": exact_match,
            "included": not bool(exclude_reason),
            "exclude_reason": exclude_reason,
        }
        audit_rows.append(audit)
        if not exclude_reason:
            included.append(
                {
                    "local_example_index": int(local_idx),
                    "row": row,
                    "row_id": row_id,
                    "model_exact_match": exact_match,
                    "example": example,
                    "gold_count": int(gold_count),
                    "segments": segments,
                    "final_pos": final_pos,
                }
            )
    return audit_rows, included


def _hidden_vector(hidden: torch.Tensor, example_index: int, position: int) -> np.ndarray:
    return (
        hidden[int(example_index), int(position)]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _span_mean_vector(
    hidden: torch.Tensor, example_index: int, start: int, end: int
) -> np.ndarray:
    return (
        hidden[int(example_index), int(start) : int(end)]
        .detach()
        .float()
        .mean(dim=0)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _example_feature_matrix(
    hidden: torch.Tensor, included: Sequence[dict[str, Any]], *, mode: str
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    local_indices: list[int] = []
    for item in included:
        local_idx = int(item["local_example_index"])
        if mode == "final_token":
            vec = _hidden_vector(hidden, local_idx, int(item["final_pos"]))
        elif mode == "mean_across_needles_span_last":
            span_vecs = [
                _hidden_vector(hidden, local_idx, int(seg["end"]) - 1)
                for seg in item["segments"]
            ]
            vec = np.mean(np.stack(span_vecs, axis=0), axis=0).astype(np.float32)
        elif mode == "mean_across_needles_span_mean":
            span_vecs = [
                _span_mean_vector(
                    hidden, local_idx, int(seg["start"]), int(seg["end"])
                )
                for seg in item["segments"]
            ]
            vec = np.mean(np.stack(span_vecs, axis=0), axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported example-level probe mode: {mode}")
        x_rows.append(vec)
        y_rows.append(int(item["gold_count"]))
        local_indices.append(local_idx)
    if not x_rows:
        raise ValueError(f"No included examples available for {mode}")
    return (
        np.stack(x_rows, axis=0).astype(np.float32),
        np.asarray(y_rows, dtype=np.float32),
        local_indices,
    )


def _split_label(local_idx: int, split: SplitResult) -> str:
    train = set(int(idx) for idx in split.train_indices)
    test = set(int(idx) for idx in split.test_indices)
    idx = int(local_idx)
    if idx in train:
        return "train"
    if idx in test:
        return "test"
    return "unused"


def _metrics_prefix(metrics: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _baseline_metrics_prefix(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    return {f"baseline_{key}": value for key, value in metrics.items()}


def _baseline_r2_title(metrics: Mapping[str, Any] | None) -> str:
    value = None if metrics is None else metrics.get("test_r2")
    try:
        if value is not None and math.isfinite(float(value)):
            return f"baseline R2: {float(value):.3f}"
    except (TypeError, ValueError):
        pass
    return "baseline R2: n/a"


def _plot_probe_predictions(
    *,
    rows: Sequence[Mapping[str, Any]],
    figure_dir: Path | None,
    mode: str,
    layer: int,
    target_column: str,
    baseline_metrics: Mapping[str, Any] | None = None,
) -> None:
    if figure_dir is None or not rows:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    for split_name in ("train", "test"):
        split_rows = [row for row in rows if row.get("split") == split_name]
        if not split_rows:
            continue
        target = np.asarray([float(row[target_column]) for row in split_rows])
        pred = np.asarray([float(row["prediction"]) for row in split_rows])
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.scatter(target, pred, s=18, alpha=0.75)
        lo = float(min(target.min(), pred.min()))
        hi = float(max(target.max(), pred.max()))
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.6)
        ax.set_xlabel(target_column)
        ax.set_ylabel("prediction")
        ax.set_title(
            f"{mode} layer {int(layer)} {split_name} "
            f"({_baseline_r2_title(baseline_metrics)})"
        )
        fig.tight_layout()
        fig.savefig(figure_dir / f"pred_vs_target_layer_{int(layer)}_{split_name}.png")
        plt.close(fig)

    test_rows = [row for row in rows if row.get("split") == "test"]
    if test_rows:
        target = np.asarray([float(row[target_column]) for row in test_rows])
        pred = np.asarray([float(row["prediction"]) for row in test_rows])
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.scatter(target, pred - target, s=18, alpha=0.75)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_xlabel(target_column)
        ax.set_ylabel("prediction - target")
        ax.set_title(
            f"{mode} layer {int(layer)} residuals "
            f"({_baseline_r2_title(baseline_metrics)})"
        )
        fig.tight_layout()
        fig.savefig(figure_dir / f"residual_vs_target_layer_{int(layer)}_test.png")
        plt.close(fig)


def _is_true(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _scored_rows_by_id(
    scored_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if scored_rows is None:
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for idx, row in enumerate(scored_rows):
        out[str(row.get("id", idx))] = row
    return out


def _safe_metric_value(metrics: Mapping[str, Any], key: str) -> str:
    value = metrics.get(key)
    try:
        if value is not None and math.isfinite(float(value)):
            return f"{float(value):.3f}"
    except (TypeError, ValueError):
        pass
    return "n/a"


def _plot_extra_eval_unified(
    *,
    rows: Sequence[Mapping[str, Any]],
    metrics_by_eval_set: Mapping[str, Mapping[str, Any]],
    figure_dir: Path | None,
    mode: str,
    layer: int,
    target_column: str,
) -> dict[str, str]:
    if figure_dir is None or not rows:
        return {}
    figure_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    all_rows = [row for row in rows if row.get("eval_set") == "unfiltered_all"]
    if not all_rows:
        all_rows = list(rows)
    outputs: dict[str, str] = {}
    styles = {
        True: {"label": "successful", "marker": "o", "color": "#1f77b4"},
        False: {"label": "failed", "marker": "^", "color": "#d62728"},
    }
    title_metrics = []
    for eval_set, label in [
        ("filtered_test", "filtered"),
        ("unfiltered_all", "all"),
        ("failed_only", "failed"),
    ]:
        metrics = metrics_by_eval_set.get(eval_set)
        if metrics:
            title_metrics.append(
                f"{label} R2={_safe_metric_value(metrics, 'r2')} "
                f"MAE={_safe_metric_value(metrics, 'mae')}"
            )
    title_suffix = " | ".join(title_metrics)

    def _scatter(ax: Any, *, residual: bool) -> None:
        for exact_value in (True, False):
            subset = [
                row
                for row in all_rows
                if _is_true(row.get("model_exact_match")) is exact_value
            ]
            if not subset:
                continue
            target = np.asarray([float(row[target_column]) for row in subset])
            pred = np.asarray([float(row["prediction"]) for row in subset])
            y_value = pred - target if residual else pred
            style = styles[exact_value]
            ax.scatter(
                target,
                y_value,
                s=24,
                alpha=0.75,
                marker=style["marker"],
                color=style["color"],
                label=f"{style['label']} (n={len(subset)})",
            )
        ax.legend(loc="best", fontsize=8)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    _scatter(ax, residual=False)
    target_values = np.asarray([float(row[target_column]) for row in all_rows])
    pred_values = np.asarray([float(row["prediction"]) for row in all_rows])
    if target_values.size and pred_values.size:
        lo = float(min(target_values.min(), pred_values.min()))
        hi = float(max(target_values.max(), pred_values.max()))
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.55)
    ax.set_xlabel(target_column)
    ax.set_ylabel("prediction")
    ax.set_title(f"{mode} L{int(layer)} | trained on successful examples")
    if title_suffix:
        ax.text(
            0.02,
            0.98,
            title_suffix,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    fig.tight_layout()
    path = figure_dir / f"extra_eval_unified_layer_{int(layer)}_prediction_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    outputs["prediction_scatter"] = str(path)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    _scatter(ax, residual=True)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.55)
    ax.set_xlabel(target_column)
    ax.set_ylabel("prediction - target")
    ax.set_title(f"{mode} L{int(layer)} residuals | trained on successful examples")
    if title_suffix:
        ax.text(
            0.02,
            0.98,
            title_suffix,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    fig.tight_layout()
    path = figure_dir / f"extra_eval_unified_layer_{int(layer)}_residual_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    outputs["residual_scatter"] = str(path)
    return outputs


def _needle_token_metadata(
    example: TokenizedCountingExample,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    seq_len = int(example.sequence_length)
    any_mask = np.zeros(seq_len, dtype=bool)
    matching_mask = np.zeros(seq_len, dtype=bool)
    needle_ids = [""] * seq_len
    matching_ids = {str(needle_id) for needle_id in example.matching_needle_ids}
    for segment in example.needle_segments:
        start = max(0, int(segment.get("start", 0)))
        end = min(seq_len, int(segment.get("end", 0)))
        if end <= start:
            continue
        needle_id = str(segment.get("needle_id", ""))
        any_mask[start:end] = True
        if needle_id in matching_ids:
            matching_mask[start:end] = True
        for pos in range(start, end):
            needle_ids[pos] = needle_id
    return any_mask, matching_mask, needle_ids


def _count_buckets_for_selection(max_count: int) -> list[tuple[str, int, int]]:
    if max_count <= 12:
        return [
            ("low", 1, min(3, max_count)),
            ("medium", 4, min(8, max_count)),
            ("high", 9, max_count),
        ]
    return [
        ("low", 1, min(10, max_count)),
        ("medium", 11, min(20, max_count)),
        ("high", 21, max_count),
    ]


def _select_sequence_projection_examples(
    included: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [item for item in included if int(item.get("gold_count", 0)) > 0]
    if not valid:
        return [], []
    max_count = max(int(item["gold_count"]) for item in valid)
    rng = np.random.default_rng(int(seed))
    keyed: list[tuple[float, str, int, dict[str, Any]]] = [
        (
            float(rng.random()),
            str(item.get("row_id", item.get("local_example_index", ""))),
            int(item["local_example_index"]),
            item,
        )
        for item in valid
    ]
    keyed.sort(key=lambda row: (row[0], row[1], row[2]))
    selected: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()

    def _candidate_key(item: Mapping[str, Any]) -> tuple[str, int]:
        return (
            str(item.get("row_id", item.get("local_example_index", ""))),
            int(item["local_example_index"]),
        )

    for bucket_name, lo, hi in _count_buckets_for_selection(max_count):
        bucket_items = [
            item
            for _rand, _row_id, _local_idx, item in keyed
            if lo <= int(item["gold_count"]) <= hi
        ]
        if not bucket_items:
            metadata_rows.append(
                {
                    "bucket": bucket_name,
                    "bucket_min_count": int(lo),
                    "bucket_max_count": int(hi),
                    "selection_status": "skipped",
                    "selection_reason": "no_examples_in_bucket",
                }
            )
            continue
        for desired_success in (True, False):
            candidates = [
                item
                for item in bucket_items
                if _is_true(item.get("model_exact_match")) is desired_success
                and _candidate_key(item) not in selected_keys
            ]
            reason = "preferred_status"
            if not candidates:
                candidates = [
                    item
                    for item in bucket_items
                    if _candidate_key(item) not in selected_keys
                ]
                reason = (
                    "substituted_available_status"
                    if candidates
                    else "no_unused_examples_in_bucket"
                )
            if not candidates:
                metadata_rows.append(
                    {
                        "bucket": bucket_name,
                        "bucket_min_count": int(lo),
                        "bucket_max_count": int(hi),
                        "desired_success_status": bool(desired_success),
                        "selection_status": "skipped",
                        "selection_reason": reason,
                    }
                )
                continue
            item = candidates[0]
            selected_keys.add(_candidate_key(item))
            selected.append(item)
            metadata_rows.append(
                {
                    "bucket": bucket_name,
                    "bucket_min_count": int(lo),
                    "bucket_max_count": int(hi),
                    "desired_success_status": bool(desired_success),
                    "selection_status": "selected",
                    "selection_reason": reason,
                    "row_id": item.get("row_id", item["local_example_index"]),
                    "local_example_index": int(item["local_example_index"]),
                    "gold_count": int(item["gold_count"]),
                    "model_exact_match": item.get("model_exact_match"),
                    "sequence_length": int(item["example"].sequence_length),
                }
            )
    return selected, metadata_rows


def _save_sequence_projection_diagnostics(
    *,
    mode: str,
    layer: int,
    result: RidgeProbeResult,
    hidden: torch.Tensor,
    included: Sequence[dict[str, Any]],
    table_dir: Path,
    figure_dir: Path | None,
    seed: int,
) -> dict[str, Any]:
    selected, selection_rows = _select_sequence_projection_examples(
        included,
        seed=int(seed) + int(layer) * 1_009,
    )
    table_out = table_dir / "sequence_projection"
    figure_out = None if figure_dir is None else figure_dir / "sequence_projection"
    table_out.mkdir(parents=True, exist_ok=True)
    if figure_out is not None:
        figure_out.mkdir(parents=True, exist_ok=True)

    examples_path = _write_dict_rows_csv(
        table_out / f"sequence_projection_examples_layer_{int(layer)}.csv",
        [
            {"mode": mode, "layer": int(layer)} | row
            for row in selection_rows
        ],
    )
    value_rows: list[dict[str, Any]] = []
    figure_paths: list[str] = []
    if not selected:
        values_path = _write_dict_rows_csv(
            table_out / f"sequence_projection_values_layer_{int(layer)}.csv",
            value_rows,
        )
        return {
            "status": "skipped",
            "skip_reason": "no_selected_examples",
            "examples_path": str(examples_path),
            "values_path": str(values_path),
            "figure_paths": figure_paths,
            "num_selected_examples": 0,
        }

    if figure_out is not None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    else:
        plt = None

    for item in selected:
        local_idx = int(item["local_example_index"])
        example = item["example"]
        seq_len = int(example.sequence_length)
        seq_hidden = (
            hidden[local_idx, :seq_len]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        projection = predict_ridge(result, seq_hidden)
        positions = np.arange(seq_len, dtype=np.int64)
        denom = max(seq_len - 1, 1)
        normalized = positions.astype(np.float32) / float(denom)
        any_mask, matching_mask, needle_ids = _needle_token_metadata(example)
        non_matching_mask = any_mask & ~matching_mask
        for pos, norm_pos, value in zip(positions, normalized, projection):
            value_rows.append(
                {
                    "mode": mode,
                    "layer": int(layer),
                    "row_id": item.get("row_id", local_idx),
                    "local_example_index": local_idx,
                    "gold_count": int(item["gold_count"]),
                    "model_exact_match": item.get("model_exact_match"),
                    "token_position": int(pos),
                    "normalized_position": float(norm_pos),
                    "projection": float(value),
                    "is_any_needle_token": bool(any_mask[int(pos)]),
                    "is_matching_needle_token": bool(matching_mask[int(pos)]),
                    "needle_id_if_any": needle_ids[int(pos)],
                }
            )
        if plt is None:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        finite_projection = projection[np.isfinite(projection)]
        linthresh = 1.0
        if finite_projection.size:
            linthresh = max(
                1.0,
                float(np.nanpercentile(np.abs(finite_projection), 99)),
            )
        ax.plot(positions, projection, color="#555555", linewidth=0.7, alpha=0.35)
        base = ax.scatter(
            positions[~any_mask],
            projection[~any_mask],
            c=normalized[~any_mask],
            cmap="Blues",
            s=11,
            alpha=0.78,
            linewidths=0,
            label="non-needle tokens",
        )
        if np.any(non_matching_mask):
            ax.scatter(
                positions[non_matching_mask],
                projection[non_matching_mask],
                color="#f39c12",
                marker="x",
                s=28,
                linewidths=1.2,
                label="non-matching needle tokens",
            )
        if np.any(matching_mask):
            ax.scatter(
                positions[matching_mask],
                projection[matching_mask],
                color="#d62728",
                marker="D",
                s=24,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.35,
                label="matching needle tokens",
            )
        ax.set_yscale("symlog", linthresh=linthresh)
        cbar = fig.colorbar(base, ax=ax)
        cbar.set_label("normalized token position")
        exact_label = (
            "unknown"
            if item.get("model_exact_match") is None
            else ("success" if _is_true(item.get("model_exact_match")) else "failed")
        )
        ax.set_xlabel("token position")
        ax.set_ylabel("ridge prediction (symlog)")
        ax.set_title(
            f"{mode} L{int(layer)} row={item.get('row_id', local_idx)} "
            f"gold={int(item['gold_count'])} {exact_label} "
            f"(linthresh={linthresh:.2g})"
        )
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = (
            figure_out
            / f"sequence_projection_layer_{int(layer)}_example_{local_idx}.png"
        )
        fig.savefig(path)
        plt.close(fig)
        figure_paths.append(str(path))

    values_path = _write_dict_rows_csv(
        table_out / f"sequence_projection_values_layer_{int(layer)}.csv",
        value_rows,
    )
    return {
        "status": "saved",
        "examples_path": str(examples_path),
        "values_path": str(values_path),
        "figure_paths": figure_paths,
        "num_selected_examples": len(selected),
        "num_projection_rows": len(value_rows),
    }


def _extra_eval_metrics(
    *,
    y: np.ndarray,
    pred: np.ndarray,
    num_parent_examples: int,
    num_eval_rows: int,
    target_distribution: Mapping[int, int],
    eval_set: str,
    layer: int,
    mode: str,
) -> dict[str, Any]:
    metrics = regression_metrics(y, pred) | {
        "rounded_count_accuracy": (
            float(np.mean(np.rint(pred) == np.rint(y))) if y.size else float("nan")
        )
    }
    return {
        "mode": mode,
        "layer": int(layer),
        "eval_set": eval_set,
        "training_source": "filtered_train",
        "status": "fit" if y.size else "skipped",
        "num_parent_examples": int(num_parent_examples),
        "num_eval_rows": int(num_eval_rows),
        "target_distribution": json.dumps(
            {str(k): int(v) for k, v in sorted(target_distribution.items())},
            sort_keys=True,
        ),
        **metrics,
    }


def _write_failed_only_count_bin_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    target_column: str,
) -> str | None:
    failed_rows = [row for row in rows if row.get("eval_set") == "failed_only"]
    if not failed_rows:
        return None
    bins = [(1, 3), (1, 6), (1, 10), (11, 20), (21, 30)]
    out_rows: list[dict[str, Any]] = []
    for lo, hi in bins:
        subset = [
            row
            for row in failed_rows
            if lo <= int(round(float(row[target_column]))) <= hi
        ]
        if not subset:
            continue
        target = np.asarray([float(row[target_column]) for row in subset], dtype=np.float32)
        pred = np.asarray([float(row["prediction"]) for row in subset], dtype=np.float32)
        residual = pred - target
        out_rows.append(
            {
                "bin": f"{lo}-{hi}",
                "num_rows": int(len(subset)),
                "mean_gold_target": float(target.mean()),
                "mean_prediction": float(pred.mean()),
                "mean_residual": float(residual.mean()),
                "mae": float(np.mean(np.abs(residual))),
                "rounded_accuracy": float(np.mean(np.rint(pred) == np.rint(target))),
            }
        )
    if not out_rows:
        return None
    return str(_write_dict_rows_csv(path, out_rows))


def _outside_needle_distance(position: int, spans: Sequence[Mapping[str, Any]]) -> int:
    pos = int(position)
    distances: list[int] = []
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if start <= pos < end:
            return 0
        if pos < start:
            distances.append(start - pos)
        else:
            distances.append(pos - end + 1)
    return min(distances) if distances else math.inf


def _baseline_candidate_positions(
    item: Mapping[str, Any], *, min_distance: int
) -> tuple[list[int], list[int]]:
    example = item["example"]
    all_spans = [
        {"start": int(span["start"]), "end": int(span["end"])}
        for span in getattr(example, "needle_segments", [])
    ]
    strict: list[int] = []
    outside: list[int] = []
    for position in range(int(example.sequence_length)):
        distance = _outside_needle_distance(position, all_spans)
        if distance > 0:
            outside.append(position)
        if distance >= int(min_distance):
            strict.append(position)
    return strict, outside


def _requested_baseline_positions(
    included: Sequence[Mapping[str, Any]], resolved_config: Mapping[str, Any] | None
) -> int:
    if resolved_config is not None and resolved_config.get("NUM_MAX_NEEDLES") is not None:
        return int(resolved_config["NUM_MAX_NEEDLES"])
    if resolved_config is not None and resolved_config.get("NUM_NEEDLES") is not None:
        return int(resolved_config["NUM_NEEDLES"])
    counts = [int(item["gold_count"]) for item in included]
    return max(counts) if counts else 1


def _sample_baseline_positions(
    included: Sequence[Mapping[str, Any]],
    *,
    requested_positions: int,
    min_distance: int,
    seed: int,
    layer: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    shortage_count = 0
    relaxed_count = 0
    for item in included:
        local_idx = int(item["local_example_index"])
        strict, outside = _baseline_candidate_positions(item, min_distance=min_distance)
        candidates = strict
        relaxed = False
        if not candidates:
            candidates = outside
            relaxed = True
            relaxed_count += 1
        if len(candidates) < int(requested_positions):
            shortage_count += 1
        if not candidates:
            records.append(
                {
                    "local_example_index": local_idx,
                    "gold_count": int(item["gold_count"]),
                    "fake_positions": [],
                    "num_candidates": 0,
                    "used_relaxed_distance": relaxed,
                }
            )
            continue
        rng = np.random.default_rng(
            int(seed) + int(layer) * 100_003 + local_idx * 1_009
        )
        sample_size = min(int(requested_positions), len(candidates))
        fake_positions = sorted(
            int(pos)
            for pos in rng.choice(
                np.asarray(candidates, dtype=np.int64),
                size=sample_size,
                replace=False,
            ).tolist()
        )
        records.append(
            {
                "local_example_index": local_idx,
                "gold_count": int(item["gold_count"]),
                "fake_positions": fake_positions,
                "num_candidates": int(len(candidates)),
                "used_relaxed_distance": relaxed,
            }
        )
    if shortage_count:
        warnings.warn(
            "Counting-probe baseline could not sample the requested number of "
            f"positions at distance >= {int(min_distance)} for {shortage_count} "
            "example(s); using all available positions for those examples.",
            RuntimeWarning,
            stacklevel=2,
        )
    if relaxed_count:
        warnings.warn(
            "Counting-probe baseline found no positions satisfying the requested "
            f"distance margin for {relaxed_count} example(s); relaxing to positions "
            "outside needle spans.",
            RuntimeWarning,
            stacklevel=2,
        )
    return records


def _save_baseline_probe(
    *,
    mode: str,
    layer: int,
    hidden: torch.Tensor,
    included: Sequence[dict[str, Any]],
    split: SplitResult,
    tensor_dir: Path,
    table_dir: Path,
    alpha: float,
    standardize: bool,
    resolved_config: Mapping[str, Any] | None,
    min_distance: int,
    seed: int,
) -> dict[str, Any]:
    requested = _requested_baseline_positions(included, resolved_config)
    records = _sample_baseline_positions(
        included,
        requested_positions=requested,
        min_distance=int(min_distance),
        seed=int(seed),
        layer=int(layer),
    )
    _write_dict_rows_csv(
        table_dir / f"baseline_positions_layer_{int(layer)}.csv",
        [
            {
                "local_example_index": record["local_example_index"],
                "gold_count": record["gold_count"],
                "fake_positions": record["fake_positions"],
                "num_fake_positions": len(record["fake_positions"]),
                "num_candidates": record["num_candidates"],
                "used_relaxed_distance": record["used_relaxed_distance"],
            }
            for record in records
        ],
    )

    vectors: list[np.ndarray] = []
    targets: list[float] = []
    local_indices: list[int] = []
    occurrence_indices: list[int] = []
    positions_for_rows: list[int | list[int]] = []
    for record in records:
        local_idx = int(record["local_example_index"])
        positions = [int(pos) for pos in record["fake_positions"]]
        if not positions:
            continue
        if mode == "occurrence_index_probe":
            for occurrence_idx, position in enumerate(positions, start=1):
                vectors.append(_hidden_vector(hidden, local_idx, position))
                targets.append(float(occurrence_idx))
                local_indices.append(local_idx)
                occurrence_indices.append(int(occurrence_idx))
                positions_for_rows.append(position)
        elif mode == "final_token":
            rng = np.random.default_rng(
                int(seed) + int(layer) * 100_003 + local_idx * 1_009 + 7_919
            )
            position = int(rng.choice(np.asarray(positions, dtype=np.int64)))
            vectors.append(_hidden_vector(hidden, local_idx, position))
            targets.append(float(record["gold_count"]))
            local_indices.append(local_idx)
            positions_for_rows.append(position)
        elif mode in {"mean_across_needles_span_last", "mean_across_needles_span_mean"}:
            fake_vecs = [_hidden_vector(hidden, local_idx, position) for position in positions]
            vectors.append(np.mean(np.stack(fake_vecs, axis=0), axis=0).astype(np.float32))
            targets.append(float(record["gold_count"]))
            local_indices.append(local_idx)
            positions_for_rows.append(positions)
        else:
            raise ValueError(f"Unsupported baseline probe mode: {mode}")

    base = {
        "status": "skipped",
        "skip_reason": "",
        "num_baseline_rows": int(len(vectors)),
        "requested_positions_per_example": int(requested),
        "baseline_min_distance": int(min_distance),
    }
    if not vectors:
        return base | {"skip_reason": "no_baseline_positions"}
    x = np.stack(vectors, axis=0).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    unique_targets = sorted(int(v) for v in np.unique(np.rint(y).astype(np.int64)).tolist())
    if len(unique_targets) < 2:
        return base | {
            "skip_reason": "fewer_than_two_targets",
            "unique_targets": unique_targets,
        }

    train_mask = np.asarray([idx in set(split.train_indices) for idx in local_indices])
    test_mask = np.asarray([idx in set(split.test_indices) for idx in local_indices])
    if not np.any(train_mask) or not np.any(test_mask):
        return base | {"skip_reason": "empty_train_or_test"}

    result = _ridge_from_matrix(
        x[train_mask],
        y[train_mask],
        alpha=alpha,
        standardize=standardize,
    )
    save_ridge_probe(result, tensor_dir, layer=layer, prefix="baseline_ridge_probe")
    train_metrics = _evaluate_matrix_ridge(result, x[train_mask], y[train_mask])
    test_metrics = _evaluate_matrix_ridge(result, x[test_mask], y[test_mask])
    pred = predict_ridge(result, x)
    target_column = (
        "target_occurrence_index" if mode == "occurrence_index_probe" else "gold_count"
    )
    prediction_rows: list[dict[str, Any]] = []
    for row_idx, (local_idx, target, prediction, fake_pos) in enumerate(
        zip(local_indices, y, pred, positions_for_rows)
    ):
        row = {
            "local_example_index": int(local_idx),
            "split": _split_label(int(local_idx), split),
            target_column: float(target),
            "prediction": float(prediction),
            "rounded_prediction": int(np.rint(prediction)),
            "fake_position": fake_pos,
        }
        if mode == "occurrence_index_probe":
            row["occurrence_index"] = int(occurrence_indices[row_idx])
        prediction_rows.append(row)
    _write_dict_rows_csv(
        table_dir / f"baseline_predictions_layer_{int(layer)}.csv", prediction_rows
    )
    metrics = {
        "status": "fit",
        "alpha": float(alpha),
        "standardize": bool(standardize),
        "num_baseline_rows": int(y.size),
        "unique_targets": unique_targets,
        "requested_positions_per_example": int(requested),
        "baseline_min_distance": int(min_distance),
    }
    metrics |= _metrics_prefix(train_metrics, "train")
    metrics |= _metrics_prefix(test_metrics, "test")
    return metrics


def _extra_eval_feature_rows(
    *,
    mode: str,
    hidden: torch.Tensor,
    included: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if mode == "occurrence_index_probe":
        vectors: list[np.ndarray] = []
        targets: list[float] = []
        metadata: list[dict[str, Any]] = []
        for item in included:
            local_idx = int(item["local_example_index"])
            for occurrence_idx, segment in enumerate(item["segments"], start=1):
                vectors.append(_hidden_vector(hidden, local_idx, int(segment["end"]) - 1))
                targets.append(float(occurrence_idx))
                metadata.append(
                    {
                        "local_example_index": local_idx,
                        "row_id": item.get("row_id", local_idx),
                        "gold_count": int(item["gold_count"]),
                        "occurrence_index": int(occurrence_idx),
                        "model_exact_match": item.get("model_exact_match"),
                    }
                )
        if not vectors:
            return (
                np.empty((0, int(hidden.shape[-1])), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                [],
            )
        return (
            np.stack(vectors, axis=0).astype(np.float32),
            np.asarray(targets, dtype=np.float32),
            metadata,
        )

    x, y, local_indices = _example_feature_matrix(hidden, included, mode=mode)
    by_local = {int(item["local_example_index"]): item for item in included}
    metadata = []
    for local_idx in local_indices:
        item = by_local[int(local_idx)]
        metadata.append(
            {
                "local_example_index": int(local_idx),
                "row_id": item.get("row_id", local_idx),
                "gold_count": int(item["gold_count"]),
                "occurrence_index": "",
                "model_exact_match": item.get("model_exact_match"),
            }
        )
    return x, y, metadata


def _save_extra_probe_evaluations(
    *,
    mode: str,
    layer: int,
    result: RidgeProbeResult,
    extra_hidden: torch.Tensor,
    extra_included: Sequence[dict[str, Any]],
    filtered_prediction_rows: Sequence[Mapping[str, Any]],
    table_dir: Path,
    figure_dir: Path | None,
) -> dict[str, Any]:
    if not extra_included:
        return {"status": "skipped", "skip_reason": "no_extra_included_examples"}

    eval_sets = {
        "unfiltered_all": list(extra_included),
        "failed_only": [
            item for item in extra_included if not _is_true(item.get("model_exact_match"))
        ],
    }
    target_column = (
        "target_occurrence_index" if mode == "occurrence_index_probe" else "gold_count"
    )
    metrics_rows: list[dict[str, Any]] = []
    unified_rows: list[dict[str, Any]] = []
    prediction_paths: dict[str, str] = {}
    examples_paths: dict[str, str] = {}
    count_bin_paths: dict[str, str] = {}

    filtered_test = [
        row for row in filtered_prediction_rows if row.get("split") == "test"
    ]
    if filtered_test:
        y_test = np.asarray(
            [float(row[target_column]) for row in filtered_test], dtype=np.float32
        )
        pred_test = np.asarray(
            [float(row["prediction"]) for row in filtered_test], dtype=np.float32
        )
        metrics_rows.append(
            _extra_eval_metrics(
                y=y_test,
                pred=pred_test,
                num_parent_examples=len(
                    {int(row["local_example_index"]) for row in filtered_test}
                ),
                num_eval_rows=len(filtered_test),
                target_distribution=dict(
                    collections.Counter(int(round(float(v))) for v in y_test)
                ),
                eval_set="filtered_test",
                layer=layer,
                mode=mode,
            )
        )

    for eval_set, eval_included in eval_sets.items():
        audit_rows = [
            {
                "eval_set": eval_set,
                "row_id": item.get("row_id", item["local_example_index"]),
                "local_example_index": int(item["local_example_index"]),
                "model_exact_match": item.get("model_exact_match"),
                "gold_count": int(item["gold_count"]),
                "num_matching_spans": len(item["segments"]),
                "span_starts": [int(seg["start"]) for seg in item["segments"]],
                "span_ends": [int(seg["end"]) for seg in item["segments"]],
                "include_status": "included",
                "exclude_reason": "",
            }
            for item in eval_included
        ]
        examples_paths[eval_set] = str(
            _write_dict_rows_csv(table_dir / f"extra_eval_examples_{eval_set}.csv", audit_rows)
        )
        if not eval_included:
            metrics_rows.append(
                {
                    "mode": mode,
                    "layer": int(layer),
                    "eval_set": eval_set,
                    "training_source": "filtered_train",
                    "status": "skipped",
                    "skip_reason": "no_valid_rows",
                    "num_parent_examples": 0,
                    "num_eval_rows": 0,
                }
            )
            continue

        x, y, metadata = _extra_eval_feature_rows(
            mode=mode,
            hidden=extra_hidden,
            included=eval_included,
        )
        if y.size == 0:
            metrics_rows.append(
                {
                    "mode": mode,
                    "layer": int(layer),
                    "eval_set": eval_set,
                    "training_source": "filtered_train",
                    "status": "skipped",
                    "skip_reason": "no_valid_rows",
                    "num_parent_examples": len(eval_included),
                    "num_eval_rows": 0,
                }
            )
            continue
        pred = predict_ridge(result, x)
        prediction_rows: list[dict[str, Any]] = []
        for item_meta, target, prediction in zip(metadata, y, pred):
            row = {
                "eval_set": eval_set,
                "layer": int(layer),
                "row_id": item_meta["row_id"],
                "local_example_index": int(item_meta["local_example_index"]),
                "occurrence_index_if_any": item_meta.get("occurrence_index", ""),
                target_column: float(target),
                "gold_target": float(target),
                "prediction": float(prediction),
                "predicted_value": float(prediction),
                "rounded_prediction": int(np.rint(prediction)),
                "residual": float(prediction - target),
                "model_exact_match": item_meta.get("model_exact_match"),
            }
            prediction_rows.append(row)
            unified_rows.append(row)
        prediction_paths[eval_set] = str(
            _write_dict_rows_csv(
                table_dir / f"extra_eval_predictions_{eval_set}_layer_{int(layer)}.csv",
                prediction_rows,
            )
        )
        metrics_rows.append(
            _extra_eval_metrics(
                y=y,
                pred=pred,
                num_parent_examples=len(
                    {int(meta["local_example_index"]) for meta in metadata}
                ),
                num_eval_rows=int(y.size),
                target_distribution=dict(
                    collections.Counter(int(round(float(v))) for v in y)
                ),
                eval_set=eval_set,
                layer=layer,
                mode=mode,
            )
        )
        if eval_set == "failed_only" and mode != "occurrence_index_probe":
            path = _write_failed_only_count_bin_metrics(
                rows=prediction_rows,
                path=table_dir / f"failed_only_count_bin_metrics_layer_{int(layer)}.csv",
                target_column=target_column,
            )
            if path is not None:
                count_bin_paths[str(layer)] = path

    metrics_file = table_dir / "extra_eval_metrics.csv"
    existing_metric_rows: list[dict[str, Any]] = []
    if metrics_file.exists() and metrics_file.read_text(encoding="utf-8").strip():
        with metrics_file.open(newline="", encoding="utf-8") as handle:
            existing_metric_rows = [
                dict(row)
                for row in csv.DictReader(handle)
                if str(row.get("layer")) != str(int(layer))
            ]
    metrics_path = _write_dict_rows_csv(
        metrics_file,
        [*existing_metric_rows, *metrics_rows],
    )
    metric_lookup = {str(row.get("eval_set")): row for row in metrics_rows}
    figure_paths = _plot_extra_eval_unified(
        rows=unified_rows,
        metrics_by_eval_set=metric_lookup,
        figure_dir=figure_dir,
        mode=mode,
        layer=layer,
        target_column=target_column,
    )
    return {
        "status": "saved",
        "metrics_path": str(metrics_path),
        "prediction_paths": prediction_paths,
        "examples_paths": examples_paths,
        "figure_paths": figure_paths,
        "count_bin_paths": count_bin_paths,
        "summary": metrics_rows,
    }


def _save_example_level_probe(
    *,
    mode: str,
    layer: int,
    hidden: torch.Tensor,
    included: Sequence[dict[str, Any]],
    split: SplitResult,
    tensor_dir: Path,
    table_dir: Path,
    figure_dir: Path | None,
    alpha: float,
    standardize: bool,
    baseline_metrics: Mapping[str, Any] | None = None,
    extra_hidden: torch.Tensor | None = None,
    extra_included: Sequence[dict[str, Any]] | None = None,
    sequence_hidden: torch.Tensor | None = None,
    sequence_included: Sequence[dict[str, Any]] | None = None,
    sequence_projection_seed: int = 42,
) -> dict[str, Any]:
    x, y, local_indices = _example_feature_matrix(hidden, included, mode=mode)
    torch.save(torch.from_numpy(x), tensor_dir / f"layer_{int(layer)}_X.pt")
    torch.save(torch.from_numpy(y), tensor_dir / "y.pt")
    unique_counts = sorted(int(v) for v in np.unique(y.astype(np.int64)).tolist())
    base = {
        "mode": mode,
        "layer": int(layer),
        "num_examples": int(y.size),
        "unique_gold_counts": unique_counts,
    }
    if len(unique_counts) < 2:
        return base | {"status": "skipped", "skip_reason": "fewer_than_two_counts"}

    local_to_row = {idx: pos for pos, idx in enumerate(local_indices)}
    train_rows = [local_to_row[i] for i in split.train_indices if i in local_to_row]
    test_rows = [local_to_row[i] for i in split.test_indices if i in local_to_row]
    if not train_rows or not test_rows:
        return base | {"status": "skipped", "skip_reason": "empty_train_or_test"}

    result = _ridge_from_matrix(
        x[np.asarray(train_rows)],
        y[np.asarray(train_rows)],
        alpha=alpha,
        standardize=standardize,
    )
    save_ridge_probe(result, tensor_dir, layer=layer)
    train_metrics = _evaluate_matrix_ridge(
        result, x[np.asarray(train_rows)], y[np.asarray(train_rows)]
    )
    test_metrics = _evaluate_matrix_ridge(
        result, x[np.asarray(test_rows)], y[np.asarray(test_rows)]
    )
    pred = predict_ridge(result, x)
    prediction_rows = [
        {
            "local_example_index": int(local_idx),
            "split": _split_label(int(local_idx), split),
            "gold_count": float(gold),
            "prediction": float(prediction),
            "rounded_prediction": int(np.rint(prediction)),
        }
        for local_idx, gold, prediction in zip(local_indices, y, pred)
    ]
    _write_dict_rows_csv(
        table_dir / f"predictions_layer_{int(layer)}.csv", prediction_rows
    )
    _plot_probe_predictions(
        rows=prediction_rows,
        figure_dir=figure_dir,
        mode=mode,
        layer=layer,
        target_column="gold_count",
        baseline_metrics=baseline_metrics,
    )
    metrics_out = base | {
        "status": "fit",
        "alpha": float(alpha),
        "standardize": bool(standardize),
    } | _metrics_prefix(train_metrics, "train") | _metrics_prefix(
        test_metrics, "test"
    ) | _baseline_metrics_prefix(
        baseline_metrics
    )
    if extra_hidden is not None and extra_included is not None:
        metrics_out["extra_eval"] = _save_extra_probe_evaluations(
            mode=mode,
            layer=layer,
            result=result,
            extra_hidden=extra_hidden,
            extra_included=extra_included,
            filtered_prediction_rows=prediction_rows,
            table_dir=table_dir,
            figure_dir=figure_dir,
        )
    seq_hidden = hidden if sequence_hidden is None else sequence_hidden
    seq_included = included if sequence_included is None else sequence_included
    metrics_out["sequence_projection"] = _save_sequence_projection_diagnostics(
        mode=mode,
        layer=layer,
        result=result,
        hidden=seq_hidden,
        included=seq_included,
        table_dir=table_dir,
        figure_dir=figure_dir,
        seed=int(sequence_projection_seed),
    )
    return metrics_out


def _save_occurrence_index_probe(
    *,
    layer: int,
    hidden: torch.Tensor,
    included: Sequence[dict[str, Any]],
    split: SplitResult,
    tensor_dir: Path,
    table_dir: Path,
    figure_dir: Path | None,
    alpha: float,
    standardize: bool,
    baseline_metrics: Mapping[str, Any] | None = None,
    extra_hidden: torch.Tensor | None = None,
    extra_included: Sequence[dict[str, Any]] | None = None,
    sequence_hidden: torch.Tensor | None = None,
    sequence_included: Sequence[dict[str, Any]] | None = None,
    sequence_projection_seed: int = 42,
) -> dict[str, Any]:
    vectors: list[np.ndarray] = []
    targets: list[int] = []
    local_indices: list[int] = []
    occurrence_indices: list[int] = []
    for item in included:
        local_idx = int(item["local_example_index"])
        for occurrence_idx, segment in enumerate(item["segments"], start=1):
            vectors.append(_hidden_vector(hidden, local_idx, int(segment["end"]) - 1))
            targets.append(int(occurrence_idx))
            local_indices.append(local_idx)
            occurrence_indices.append(int(occurrence_idx))
    if not vectors:
        raise ValueError("No occurrence vectors available for occurrence_index_probe")
    x = np.stack(vectors, axis=0).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    unique_occurrences = sorted(int(v) for v in np.unique(y.astype(np.int64)).tolist())

    prototypes = []
    metadata_rows = []
    geometry_rows = []
    gold_by_local_index = {
        int(item["local_example_index"]): int(item["gold_count"]) for item in included
    }
    for occurrence_idx in unique_occurrences:
        rows = np.where(y.astype(np.int64) == occurrence_idx)[0]
        proto = x[rows].mean(axis=0).astype(np.float32)
        prototypes.append(proto)
        contributing_counts = [gold_by_local_index[int(local_indices[row])] for row in rows]
        metadata_rows.append(
            {
                "layer": int(layer),
                "occurrence_index": int(occurrence_idx),
                "num_contributing_examples": int(len(rows)),
                "contributing_count_distribution": _count_distribution(
                    contributing_counts
                ),
            }
        )
    proto_arr = np.stack(prototypes, axis=0).astype(np.float32)
    torch.save(torch.from_numpy(proto_arr), tensor_dir / f"layer_{int(layer)}_prototypes.pt")
    _write_dict_rows_csv(
        table_dir / f"prototype_metadata_layer_{int(layer)}.csv", metadata_rows
    )
    for idx, occurrence_idx in enumerate(unique_occurrences):
        row = {
            "layer": int(layer),
            "occurrence_index": int(occurrence_idx),
            "prototype_norm": float(np.linalg.norm(proto_arr[idx])),
        }
        if idx > 0:
            a = proto_arr[idx - 1]
            b = proto_arr[idx]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            row["adjacent_cosine"] = float(a @ b / denom) if denom > 0 else float("nan")
            row["adjacent_delta_norm"] = float(np.linalg.norm(b - a))
        else:
            row["adjacent_cosine"] = float("nan")
            row["adjacent_delta_norm"] = float("nan")
        geometry_rows.append(row)
    _write_dict_rows_csv(
        table_dir / f"prototype_geometry_layer_{int(layer)}.csv", geometry_rows
    )

    base = {
        "mode": "occurrence_index_probe",
        "layer": int(layer),
        "num_occurrence_rows": int(y.size),
        "unique_occurrence_indices": unique_occurrences,
    }
    if len(unique_occurrences) < 2:
        return base | {"status": "skipped", "skip_reason": "fewer_than_two_occurrences"}

    train_mask = np.asarray([idx in set(split.train_indices) for idx in local_indices])
    test_mask = np.asarray([idx in set(split.test_indices) for idx in local_indices])
    if not np.any(train_mask) or not np.any(test_mask):
        return base | {"status": "skipped", "skip_reason": "empty_train_or_test"}
    result = _ridge_from_matrix(
        x[train_mask],
        y[train_mask],
        alpha=alpha,
        standardize=standardize,
    )
    save_ridge_probe(result, tensor_dir, layer=layer)
    train_metrics = _evaluate_matrix_ridge(result, x[train_mask], y[train_mask])
    test_metrics = _evaluate_matrix_ridge(result, x[test_mask], y[test_mask])
    pred = predict_ridge(result, x)
    prediction_rows = [
        {
            "local_example_index": int(local_idx),
            "occurrence_index": int(occ_idx),
            "split": _split_label(int(local_idx), split),
            "target_occurrence_index": int(target),
            "prediction": float(prediction),
            "rounded_prediction": int(np.rint(prediction)),
        }
        for local_idx, occ_idx, target, prediction in zip(
            local_indices, occurrence_indices, y, pred
        )
    ]
    _write_dict_rows_csv(
        table_dir / f"predictions_layer_{int(layer)}.csv", prediction_rows
    )
    _plot_probe_predictions(
        rows=prediction_rows,
        figure_dir=figure_dir,
        mode="occurrence_index_probe",
        layer=layer,
        target_column="target_occurrence_index",
        baseline_metrics=baseline_metrics,
    )
    metrics_out = base | {
        "status": "fit",
        "alpha": float(alpha),
        "standardize": bool(standardize),
    } | _metrics_prefix(train_metrics, "train") | _metrics_prefix(
        test_metrics, "test"
    ) | _baseline_metrics_prefix(
        baseline_metrics
    )
    if extra_hidden is not None and extra_included is not None:
        metrics_out["extra_eval"] = _save_extra_probe_evaluations(
            mode="occurrence_index_probe",
            layer=layer,
            result=result,
            extra_hidden=extra_hidden,
            extra_included=extra_included,
            filtered_prediction_rows=prediction_rows,
            table_dir=table_dir,
            figure_dir=figure_dir,
        )
    seq_hidden = hidden if sequence_hidden is None else sequence_hidden
    seq_included = included if sequence_included is None else sequence_included
    metrics_out["sequence_projection"] = _save_sequence_projection_diagnostics(
        mode="occurrence_index_probe",
        layer=layer,
        result=result,
        hidden=seq_hidden,
        included=seq_included,
        table_dir=table_dir,
        figure_dir=figure_dir,
        seed=int(sequence_projection_seed),
    )
    return metrics_out


def _ridge_direction_for_similarity(probe_path: Path) -> np.ndarray:
    payload = torch.load(probe_path, map_location="cpu")
    coef = payload["coef"].detach().float().cpu().numpy().astype(np.float32)
    scale = payload.get("feature_scale")
    if scale is not None:
        scale_arr = scale.detach().float().cpu().numpy().astype(np.float32)
        safe_scale = np.where(np.abs(scale_arr) < 1e-12, 1.0, scale_arr)
        coef = (coef / safe_scale).astype(np.float32)
    return coef


def _cosine_similarity_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.stack([np.asarray(vec, dtype=np.float32) for vec in vectors], axis=0)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms <= 0, 1.0, norms)
    normalized = arr / safe
    out = normalized @ normalized.T
    zero = (norms[:, 0] <= 0)
    if np.any(zero):
        out[zero, :] = np.nan
        out[:, zero] = np.nan
    return out.astype(np.float32)


def _plot_similarity_heatmap(
    *,
    matrix: np.ndarray,
    modes: Sequence[str],
    layer: int,
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(modes)))
    ax.set_yticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=35, ha="right")
    ax.set_yticklabels(modes)
    ax.set_title(f"Ridge-vector cosine similarity, layer {int(layer)}")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            label = "nan" if not math.isfinite(float(value)) else f"{float(value):.2f}"
            ax.text(col, row, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def _save_ridge_vector_similarity(
    *,
    modes: Sequence[str],
    layers: Sequence[int],
    tensor_root: Path,
    table_root: Path,
    figure_root: Path | None,
) -> dict[str, Any]:
    table_dir = table_root / "ridge_vector_similarity"
    figure_dir = None if figure_root is None else figure_root / "ridge_vector_similarity"
    table_dir.mkdir(parents=True, exist_ok=True)
    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {}
    for layer in [int(x) for x in layers]:
        present_modes: list[str] = []
        directions: list[np.ndarray] = []
        skipped: dict[str, str] = {}
        for probe_mode in modes:
            probe_path = tensor_root / probe_mode / f"ridge_probe_layer_{layer}.pt"
            if not probe_path.exists():
                skipped[probe_mode] = "missing_ridge_probe"
                continue
            direction = _ridge_direction_for_similarity(probe_path)
            if directions and direction.shape != directions[0].shape:
                skipped[probe_mode] = (
                    f"shape_mismatch:{tuple(direction.shape)}!="
                    f"{tuple(directions[0].shape)}"
                )
                continue
            present_modes.append(probe_mode)
            directions.append(direction)
        if len(directions) < 2:
            outputs[str(layer)] = {
                "status": "skipped",
                "skip_reason": "fewer_than_two_vectors",
                "skipped_modes": skipped,
            }
            continue
        matrix = _cosine_similarity_matrix(directions)
        rows = []
        for row_idx, row_mode in enumerate(present_modes):
            row = {"mode": row_mode}
            for col_idx, col_mode in enumerate(present_modes):
                row[col_mode] = float(matrix[row_idx, col_idx])
            rows.append(row)
        csv_path = _write_dict_rows_csv(
            table_dir / f"layer_{layer}_cosine_similarity.csv", rows
        )
        heatmap_path = None
        if figure_dir is not None:
            heatmap_path = figure_dir / f"layer_{layer}_cosine_similarity_heatmap.png"
            _plot_similarity_heatmap(
                matrix=matrix,
                modes=present_modes,
                layer=layer,
                figure_path=heatmap_path,
            )
        outputs[str(layer)] = {
            "status": "saved",
            "csv": str(csv_path),
            "heatmap": None if heatmap_path is None else str(heatmap_path),
            "modes": list(present_modes),
            "skipped_modes": skipped,
            "direction": "input_space_adjusted_coef_when_standardized",
        }
    write_json(
        table_dir / "ridge_vector_similarity_metadata.json",
        {"layers": [int(x) for x in layers], "modes": list(modes), "outputs": outputs},
    )
    return {
        "table_dir": str(table_dir),
        "figure_dir": None if figure_dir is None else str(figure_dir),
        "layers": outputs,
    }


def run_counting_probe_diagnostics(
    *,
    mode: str,
    rows: Sequence[dict[str, Any]],
    examples: Sequence[TokenizedCountingExample],
    split: SplitResult,
    layers: Sequence[int],
    feature_tensors_dir: str | Path,
    feature_tables_dir: str | Path,
    feature_figures_dir: str | Path | None = None,
    filter_summary: Mapping[str, Any] | None = None,
    resolved_config: Mapping[str, Any] | None = None,
    ridge_alpha: float = 1.0,
    standardize_features: bool = True,
    run_baseline: bool = True,
    baseline_min_distance: int = 5,
    split_seed: int = 42,
    scored_rows: Sequence[dict[str, Any]] | None = None,
    extra_eval_rows: Sequence[dict[str, Any]] | None = None,
    extra_eval_examples: Sequence[TokenizedCountingExample] | None = None,
    extra_eval_feature_tensors_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run non-direct counting probe diagnostics and save artifacts."""

    mode = normalize_counting_probe_mode(mode)
    if mode not in COUNTING_PROBE_MODES - {"direct"}:
        raise ValueError(f"Unsupported non-direct COUNTING_PROBE_MODE: {mode!r}")
    selected_modes = (
        [
            "occurrence_index_probe",
            "mean_across_needles_span_last",
            "mean_across_needles_span_mean",
            "final_token",
        ]
        if mode == "all_diagnostics"
        else [mode]
    )
    tensor_root = _as_path(feature_tensors_dir) / "probe_diagnostics"
    table_root = _as_path(feature_tables_dir) / "probe_diagnostics"
    figure_root = (
        None
        if feature_figures_dir is None
        else _as_path(feature_figures_dir) / "probe_diagnostics"
    )
    if feature_figures_dir is not None:
        figure_root.mkdir(parents=True, exist_ok=True)
    scored_by_id = _scored_rows_by_id(scored_rows)
    audit_rows, included = _example_probe_rows(
        rows,
        examples,
        scored_by_id=scored_by_id,
    )
    extra_audit_rows: list[dict[str, Any]] = []
    extra_included: list[dict[str, Any]] = []
    extra_tensor_root = (
        None
        if extra_eval_feature_tensors_dir is None
        else _as_path(extra_eval_feature_tensors_dir)
    )
    extra_eval_enabled = (
        extra_eval_rows is not None
        and extra_eval_examples is not None
        and extra_tensor_root is not None
    )
    if extra_eval_enabled:
        extra_audit_rows, extra_included = _example_probe_rows(
            extra_eval_rows,
            extra_eval_examples,
            scored_by_id=scored_by_id,
        )
    gold_counts = [int(item["gold_count"]) for item in included]
    outputs: dict[str, Any] = {}
    for probe_mode in selected_modes:
        tensor_dir = tensor_root / probe_mode
        table_dir = table_root / probe_mode
        figure_dir = None if figure_root is None else figure_root / probe_mode
        tensor_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        if figure_dir is not None:
            figure_dir.mkdir(parents=True, exist_ok=True)
        _write_dict_rows_csv(table_dir / "probe_examples.csv", audit_rows)
        summary_rows: list[dict[str, Any]] = []
        baseline_rows: list[dict[str, Any]] = []
        for layer in [int(x) for x in layers]:
            hidden_path = _as_path(feature_tensors_dir) / f"hidden_layer_{layer}.pt"
            if not hidden_path.exists():
                raise FileNotFoundError(
                    f"Missing hidden-state tensor for diagnostic probe: {hidden_path}"
                )
            hidden = torch.load(hidden_path, map_location="cpu")
            extra_hidden = None
            if extra_eval_enabled:
                assert extra_tensor_root is not None
                extra_hidden_path = extra_tensor_root / f"hidden_layer_{layer}.pt"
                if not extra_hidden_path.exists():
                    raise FileNotFoundError(
                        "Missing extra-evaluation hidden-state tensor for "
                        f"diagnostic probe: {extra_hidden_path}"
                    )
                extra_hidden = torch.load(extra_hidden_path, map_location="cpu")
            baseline_metrics: dict[str, Any] | None = None
            if run_baseline:
                baseline_metrics = _save_baseline_probe(
                    mode=probe_mode,
                    layer=layer,
                    hidden=hidden,
                    included=included,
                    split=split,
                    tensor_dir=tensor_dir,
                    table_dir=table_dir,
                    alpha=float(ridge_alpha),
                    standardize=bool(standardize_features),
                    resolved_config=resolved_config,
                    min_distance=int(baseline_min_distance),
                    seed=int(split_seed),
                )
                baseline_rows.append(
                    {"mode": probe_mode, "layer": int(layer)} | baseline_metrics
                )
            if probe_mode == "occurrence_index_probe":
                metrics = _save_occurrence_index_probe(
                    layer=layer,
                    hidden=hidden,
                    included=included,
                    split=split,
                    tensor_dir=tensor_dir,
                    table_dir=table_dir,
                    figure_dir=figure_dir,
                    alpha=float(ridge_alpha),
                    standardize=bool(standardize_features),
                    baseline_metrics=baseline_metrics,
                    extra_hidden=extra_hidden,
                    extra_included=extra_included if extra_eval_enabled else None,
                    sequence_hidden=extra_hidden if extra_hidden is not None else hidden,
                    sequence_included=(
                        extra_included if extra_eval_enabled else included
                    ),
                    sequence_projection_seed=int(split_seed),
                )
            else:
                metrics = _save_example_level_probe(
                    mode=probe_mode,
                    layer=layer,
                    hidden=hidden,
                    included=included,
                    split=split,
                    tensor_dir=tensor_dir,
                    table_dir=table_dir,
                    figure_dir=figure_dir,
                    alpha=float(ridge_alpha),
                    standardize=bool(standardize_features),
                    baseline_metrics=baseline_metrics,
                    extra_hidden=extra_hidden,
                    extra_included=extra_included if extra_eval_enabled else None,
                    sequence_hidden=extra_hidden if extra_hidden is not None else hidden,
                    sequence_included=(
                        extra_included if extra_eval_enabled else included
                    ),
                    sequence_projection_seed=int(split_seed),
                )
            summary_rows.append(metrics)
            del hidden
            if extra_hidden is not None:
                del extra_hidden
        _write_dict_rows_csv(table_dir / "summary.csv", summary_rows)
        _write_dict_rows_csv(table_dir / "ridge_metrics.csv", summary_rows)
        _write_dict_rows_csv(table_dir / "baseline_ridge_metrics.csv", baseline_rows)
        metadata = {
            "COUNTING_PROBE_MODE": mode,
            "probe_mode": probe_mode,
            "COUNTING_FEATURE_CALC_METHOD": (
                None
                if resolved_config is None
                else resolved_config.get("COUNTING_FEATURE_CALC_METHOD")
            ),
            "FILTER_EXAMPLE": (
                None if resolved_config is None else resolved_config.get("FILTER_EXAMPLE")
            ),
            "NUM_NEEDLES": (
                None if resolved_config is None else resolved_config.get("NUM_NEEDLES")
            ),
            "NUM_MAX_NEEDLES": (
                None if resolved_config is None else resolved_config.get("NUM_MAX_NEEDLES")
            ),
            "RUN_COUNTING_PROBE_BASELINE": bool(run_baseline),
            "COUNTING_PROBE_BASELINE_MIN_DISTANCE": int(baseline_min_distance),
            "num_rows": len(rows),
            "num_included_examples": len(included),
            "count_distribution": _count_distribution(gold_counts),
            "layers": [int(x) for x in layers],
            "split": split.__dict__,
            "filter_summary": dict(filter_summary or {}),
            "extra_eval": {
                "attempted": bool(extra_eval_enabled),
                "num_rows": 0 if extra_eval_rows is None else len(extra_eval_rows),
                "num_included_examples": len(extra_included),
                "num_failed_examples": sum(
                    1
                    for item in extra_included
                    if not _is_true(item.get("model_exact_match"))
                ),
                "num_successful_examples": sum(
                    1
                    for item in extra_included
                    if _is_true(item.get("model_exact_match"))
                ),
                "audit_rows_path": str(
                    _write_dict_rows_csv(
                        table_dir / "extra_eval_all_examples_audit.csv",
                        extra_audit_rows,
                    )
                )
                if extra_eval_enabled
                else None,
            },
            "summary": summary_rows,
            "baseline_summary": baseline_rows,
        }
        write_json(table_dir / "diagnostic_metadata.json", metadata)
        outputs[probe_mode] = {
            "tensor_dir": str(tensor_dir),
            "table_dir": str(table_dir),
            "figure_dir": None if figure_dir is None else str(figure_dir),
            "summary": summary_rows,
            "baseline_summary": baseline_rows,
        }
    if mode == "all_diagnostics":
        similarity = _save_ridge_vector_similarity(
            modes=selected_modes,
            layers=layers,
            tensor_root=tensor_root,
            table_root=table_root,
            figure_root=figure_root,
        )
        outputs["ridge_vector_similarity"] = similarity
    return outputs


def top_pca_direction(
    x: np.ndarray,
    *,
    max_rows: int = 50_000,
    seed: int = 42,
    include_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Return the top right-singular direction of a row-feature matrix."""

    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D feature matrix, got {x.shape}")
    if include_rows is not None:
        include = np.asarray(include_rows, dtype=bool)
        if include.shape != (x.shape[0],):
            raise ValueError(
                f"include_rows must have shape {(x.shape[0],)}, got {include.shape}"
            )
        if np.any(include):
            x = x[include]
    rng = np.random.default_rng(int(seed))
    if x.shape[0] > int(max_rows):
        x = x[rng.choice(x.shape[0], size=int(max_rows), replace=False)]
    centered = x.astype(np.float32, copy=False) - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    v = vh[0].astype(np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _raw_norm_inlier_mask_by_example(
    x_raw: np.ndarray,
    example_indices: np.ndarray,
    *,
    norm_median_multiplier: float = 5.0,
) -> np.ndarray:
    """Keep rows whose raw hidden-state norm is not an example-level outlier."""

    if x_raw.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    norms = np.linalg.norm(x_raw.astype(np.float32, copy=False), axis=1)
    inlier = np.ones(norms.shape, dtype=bool)
    for example_index in np.unique(example_indices):
        ex_rows = example_indices == example_index
        ex_norms = norms[ex_rows]
        finite_norms = ex_norms[np.isfinite(ex_norms)]
        if finite_norms.size == 0:
            inlier[ex_rows] = False
            continue
        median = float(np.median(finite_norms))
        if median <= 0.0:
            inlier[ex_rows] = np.isfinite(ex_norms)
        else:
            inlier[ex_rows] = np.isfinite(ex_norms) & (
                ex_norms <= float(norm_median_multiplier) * median
            )
    return inlier


def _robust_limits(
    values: np.ndarray, percentiles: tuple[float, float] | None
) -> tuple[float, float]:
    if values.size == 0:
        return -1.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    if percentiles is None:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    else:
        lo_p, hi_p = percentiles
        lo, hi = (float(v) for v in np.percentile(finite, [lo_p, hi_p]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if lo == hi:
        pad = max(1.0, abs(lo) * 0.05)
    else:
        pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def classification_count_direction(result: ClassificationProbeResult) -> np.ndarray:
    """Return a low-to-high count direction for a multi-class classifier."""

    if result.coef.ndim != 2 or result.coef.shape[0] == 0:
        raise ValueError(
            f"Expected 2D classifier coefficients, got {result.coef.shape}"
        )
    order = np.argsort(result.classes.astype(float))
    if order.size >= 2:
        return (result.coef[order[-1]] - result.coef[order[0]]).astype(np.float32)
    centered = result.coef - result.coef.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    return vh[0].astype(np.float32)


def _classification_logits_from_standardized(
    result: ClassificationProbeResult, x_standardized: np.ndarray
) -> np.ndarray:
    return (
        x_standardized.astype(np.float32, copy=False) @ result.coef.T
        + result.intercept[None, :]
    )


def _plot_sample(
    hidden: torch.Tensor,
    target: torch.Tensor,
    *,
    example_indices: Sequence[int] | None,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, _, _ = _plot_sample_with_positions(
        hidden,
        target,
        example_indices=example_indices,
        max_points=max_points,
        seed=seed,
    )
    return x, y


def _plot_sample_with_positions(
    hidden: torch.Tensor,
    target: torch.Tensor,
    *,
    example_indices: Sequence[int] | None,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = sample_probe_positions(
        target,
        example_indices=example_indices,
        max_tokens=max_points,
        seed=seed,
        balanced=True,
    )
    x = (
        hidden[sample.example_indices, sample.token_indices, :]
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )
    y = (
        target[sample.example_indices, sample.token_indices]
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return (
        x[finite],
        y[finite],
        sample.example_indices[finite],
        sample.token_indices[finite],
    )


def plot_ridge_line_fit(
    result: RidgeProbeResult,
    hidden: torch.Tensor,
    target: torch.Tensor,
    output_path: str | Path,
    *,
    example_indices: Sequence[int] | None = None,
    max_points: int = 20_000,
    seed: int = 42,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    x, y, _, token_positions = _plot_sample_with_positions(
        hidden,
        target,
        example_indices=example_indices,
        max_points=max_points,
        seed=seed,
    )
    pred = predict_ridge(result, x)
    out = _as_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap_name = "viridis"
    if _is_integer_count_values(y):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D

        labels = np.array(sorted(int(round(v)) for v in np.unique(y[np.isfinite(y)])))
        position_norm = Normalize(
            vmin=float(np.min(token_positions)) if token_positions.size else 0.0,
            vmax=float(np.max(token_positions)) if token_positions.size else 1.0,
        )
        marker_by_label = {
            int(label): _marker_for_label_index(idx)
            for idx, label in enumerate(labels.tolist())
        }
        rounded_y = np.round(y).astype(int)
        for label in labels.tolist():
            label_mask = rounded_y == int(label)
            if not np.any(label_mask):
                continue
            ax.scatter(
                pred[label_mask],
                y[label_mask],
                c=token_positions[label_mask],
                s=16,
                alpha=0.85,
                cmap=cmap_name,
                norm=position_norm,
                marker=marker_by_label[int(label)],
                edgecolors="none",
            )
        fig.colorbar(
            ScalarMappable(norm=position_norm, cmap=cmap_name),
            ax=ax,
            label="token position",
        )
        label_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_by_label[int(label)],
                linestyle="",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=6,
                label=str(int(label)),
            )
            for label in labels.tolist()
        ]
        if label_handles:
            label_legend = ax.legend(
                handles=label_handles,
                title="target count",
                loc="upper left",
                frameon=True,
            )
            ax.add_artist(label_legend)
    else:
        sc = ax.scatter(
            pred,
            y,
            c=token_positions,
            s=10,
            alpha=0.5,
            cmap=cmap_name,
            edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label="token position")
    if pred.size:
        lo = float(min(np.min(pred), np.min(y)))
        hi = float(max(np.max(pred), np.max(y)))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, label="y=x")
        if pred.size > 1:
            slope, intercept = np.polyfit(pred, y, deg=1)
            xs = np.linspace(float(np.min(pred)), float(np.max(pred)), 100)
            ax.plot(
                xs,
                slope * xs + intercept,
                color="tab:red",
                linewidth=1.2,
                label="linear fit",
            )
    ax.set_xlabel(r"$h_t^T u$")
    ax.set_ylabel(r"target count $y_t$")
    ax.set_title(title or "Ridge counting probe line fit")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _is_integer_count_values(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.all(np.isclose(finite, np.round(finite), atol=1e-6)))


def _discrete_count_marker_metadata(
    values: np.ndarray, marker_by_label: dict[int, str]
) -> dict[str, Any]:
    unique = sorted(float(v) for v in np.unique(values[np.isfinite(values)]))
    return {
        "color_mode": "continuous_token_positions",
        "marker_mode": "discrete_integer_counts",
        "unique_target_values": unique,
        "num_unique_target_values": len(unique),
        "scatter_alpha": 0.85,
        "marker_by_label": {str(label): marker for label, marker in marker_by_label.items()},
        "colorbar_label": "token position",
        "legend_title": "target count",
        "note": (
            "Point color encodes the sampled hidden state's token position with a continuous "
            "gradient, while marker shape encodes the integer target-count label category."
        ),
    }


def _continuous_position_color_metadata(
    values: np.ndarray, positions: np.ndarray, cmap_name: str, alpha: float
) -> dict[str, Any]:
    finite_values = values[np.isfinite(values)]
    finite_positions = positions[np.isfinite(positions)]
    return {
        "color_mode": "continuous_token_positions",
        "marker_mode": "single_marker_continuous_targets",
        "colormap": cmap_name,
        "unique_target_values": sorted(float(v) for v in np.unique(finite_values)),
        "num_unique_target_values": int(np.unique(finite_values).size),
        "scatter_alpha": float(alpha),
        "target_min": None if finite_values.size == 0 else float(np.min(finite_values)),
        "target_max": None if finite_values.size == 0 else float(np.max(finite_values)),
        "position_min": None if finite_positions.size == 0 else int(np.min(finite_positions)),
        "position_max": None if finite_positions.size == 0 else int(np.max(finite_positions)),
        "colorbar_label": "token position",
        "note": (
            "Point color encodes the sampled hidden state's token position with a continuous "
            "gradient. Continuous target values are not shown as discrete label markers."
        ),
    }


def _marker_for_label_index(index: int) -> str:
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    return markers[index % len(markers)]


def plot_probe_2d_projection(
    probe_vector: np.ndarray | RidgeProbeResult | ClassificationProbeResult,
    hidden: torch.Tensor,
    target: torch.Tensor,
    output_path: str | Path,
    *,
    example_indices: Sequence[int] | None = None,
    classifier: ClassificationProbeResult | None = None,
    feature_mean: np.ndarray | None = None,
    feature_scale: np.ndarray | None = None,
    max_points: int = 20_000,
    seed: int = 42,
    title: str | None = None,
    pca_norm_median_multiplier: float | None = 5.0,
    axis_percentiles: tuple[float, float] | None = (1.0, 99.0),
) -> Path:
    """Plot hidden states in the probe/PCA span in the probe feature space.

    Probe coefficients learned with ``standardize=True`` live in standardized
    hidden-state coordinates. Passing a probe result (or explicit
    ``feature_mean``/``feature_scale``) makes the scatter, PCA direction, and
    optional classifier boundary use that same coordinate system. The PCA fit
    excludes per-example raw hidden-state norm outliers by default.
    """

    import matplotlib.pyplot as plt

    inferred_classifier = classifier
    if isinstance(probe_vector, RidgeProbeResult):
        feature_mean = probe_vector.feature_mean
        feature_scale = probe_vector.feature_scale
        vector = probe_vector.coef
    elif isinstance(probe_vector, ClassificationProbeResult):
        inferred_classifier = (
            probe_vector if inferred_classifier is None else inferred_classifier
        )
        feature_mean = probe_vector.feature_mean
        feature_scale = probe_vector.feature_scale
        vector = classification_count_direction(probe_vector)
    else:
        vector = probe_vector
        if (
            inferred_classifier is not None
            and feature_mean is None
            and feature_scale is None
        ):
            feature_mean = inferred_classifier.feature_mean
            feature_scale = inferred_classifier.feature_scale

    x_raw, y, sample_examples, token_positions = _plot_sample_with_positions(
        hidden,
        target,
        example_indices=example_indices,
        max_points=max_points,
        seed=seed,
    )
    x = _standardize_apply(x_raw, feature_mean, feature_scale)
    u = np.asarray(vector, dtype=np.float32)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 0:
        raise ValueError("Probe vector has zero norm")
    u = u / u_norm
    pca_inliers = None
    if pca_norm_median_multiplier is not None:
        pca_inliers = _raw_norm_inlier_mask_by_example(
            x_raw,
            sample_examples,
            norm_median_multiplier=float(pca_norm_median_multiplier),
        )
    v = top_pca_direction(x, seed=seed, include_rows=pca_inliers)
    v = v - float(np.dot(v, u)) * u
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-8:
        v = top_pca_direction(
            x + np.random.default_rng(seed).normal(0, 1e-6, x.shape).astype(np.float32),
            seed=seed,
            include_rows=pca_inliers,
        )
        v = v - float(np.dot(v, u)) * u
        v_norm = float(np.linalg.norm(v))
    v = v / max(v_norm, 1e-8)
    coords = np.column_stack([x @ u, x @ v])
    x_lim = _robust_limits(coords[:, 0], axis_percentiles)
    y_lim = _robust_limits(coords[:, 1], axis_percentiles)

    out = _as_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap_name = "viridis"
    scatter_alpha = 0.85
    if _is_integer_count_values(y):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D

        labels = np.array(sorted(int(round(v)) for v in np.unique(y[np.isfinite(y)])))
        position_norm = Normalize(
            vmin=float(np.min(token_positions)) if token_positions.size else 0.0,
            vmax=float(np.max(token_positions)) if token_positions.size else 1.0,
        )
        marker_by_label = {
            int(label): _marker_for_label_index(idx)
            for idx, label in enumerate(labels.tolist())
        }
        rounded_y = np.round(y).astype(int)
        for label in labels.tolist():
            label_mask = rounded_y == int(label)
            if not np.any(label_mask):
                continue
            ax.scatter(
                coords[label_mask, 0],
                coords[label_mask, 1],
                c=token_positions[label_mask],
                s=16,
                alpha=scatter_alpha,
                cmap=cmap_name,
                norm=position_norm,
                marker=marker_by_label[int(label)],
                edgecolors="none",
            )
        colorbar = fig.colorbar(
            ScalarMappable(norm=position_norm, cmap=cmap_name),
            ax=ax,
            label="token position",
        )
        colorbar.ax.set_ylabel("token position")
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_by_label[int(label)],
                linestyle="",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=6,
                label=str(int(label)),
            )
            for label in labels.tolist()
        ]
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title="target count",
                loc="best",
                frameon=True,
            )
        plot_color_metadata = _discrete_count_marker_metadata(y, marker_by_label)
        plot_color_metadata["colormap"] = cmap_name
        plot_color_metadata["position_min"] = (
            None if token_positions.size == 0 else int(np.min(token_positions))
        )
        plot_color_metadata["position_max"] = (
            None if token_positions.size == 0 else int(np.max(token_positions))
        )
    else:
        scatter_alpha = 0.5
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=token_positions,
            s=10,
            alpha=scatter_alpha,
            cmap=cmap_name,
            marker="o",
            edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label="token position")
        plot_color_metadata = _continuous_position_color_metadata(
            y, token_positions, cmap_name, scatter_alpha
        )
    if inferred_classifier is not None and coords.shape[0] > 0:
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], 150),
            np.linspace(y_lim[0], y_lim[1], 150),
        )
        # Evaluate the classifier directly in the plotted feature space. When a
        # classifier was trained with standardization, ``x``/``u``/``v`` are
        # standardized coordinates, so applying standardization a second time
        # would put the contour in the wrong space.
        basis_points = xx.reshape(-1, 1) * u[None, :] + yy.reshape(-1, 1) * v[None, :]
        logits = _classification_logits_from_standardized(
            inferred_classifier, basis_points
        )
        pred = inferred_classifier.classes[np.argmax(logits, axis=1)].reshape(xx.shape)
        ax.contour(xx, yy, pred, colors="black", linewidths=0.6, alpha=0.8)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("probe direction")
    ax.set_ylabel("top PCA direction orthogonalized to probe")
    ax.set_title(title or "Counting probe 2D projection")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    metadata_path = out.with_suffix(out.suffix + ".metadata.json")
    write_json(
        metadata_path,
        {
            "figure_path": str(out),
            "num_plotted_points": int(coords.shape[0]),
            **plot_color_metadata,
        },
    )
    return out
