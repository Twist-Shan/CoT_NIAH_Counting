"""Native-thinking-aligned representation analysis for the V6 grammar contrast.

This module deliberately has only two primary endpoints:

* ``item_end`` running-count states on exact four-cell common support; and
* ``answer_query_v3`` final-count states on the exact registered 300 trajectories.

The original all-sample captures are used for both endpoints.  Resolved
replacement cohorts are appropriate for cellwise causal replication, but they
cannot manufacture source identity for a direct index-versus-bullet contrast.
All preprocessing and layer selection use discovery rows only.  Confirmation
rows are evaluated at every layer for transparent curves, but never enter the
programmed winner rule.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from realistic_niah_v5.covariance_geometry import (
    evaluate_covariance_geometry_layer,
)
from realistic_niah_v5.cross_mode_geometry import (
    CLASSES,
    ModeDataset,
    load_native_thinking_capture,
)
from realistic_niah_v5.dual_endpoint_geometry import (
    load_native_thinking_final_count,
)
from realistic_niah_v5.trace_stratified_geometry import (
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)

from .spec import (
    CONFIRMATION_SEEDS,
    COUNTS,
    DISCOVERY_SEEDS,
    MODEL_LABELS,
    PROMPT_MODES,
)


SCHEMA_VERSION = "realistic_niah_v6_native_aligned_representation_v1"
CONTRACT_SCHEMA_VERSION = "realistic_niah_v6_native_analysis_alignment_v1"
RUNNING_SITE = "item_end"
FINAL_SITE = "answer_query_v3"
RUNNING_ALIGNMENT_COLUMNS = ("split", "seed", "gold_count", "occurrence")
FINAL_ALIGNMENT_COLUMNS = ("split", "seed", "gold_count")
PCA_DIM = 16
CV_FOLDS = 5
PCA_WHITEN = True
RANDOM_STATE = 0
RELATIVE_RIDGE = 1e-6


Cell = tuple[str, str]
RunningKey = tuple[str, int, int, int]
TrajectoryKey = tuple[str, int, int]


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            values.append(value)
    return values


def _cell_slug(cell: Cell) -> str:
    return f"{cell[0]}|{cell[1]}"


def _capture_index(run_root: Path, cell: Cell) -> Path:
    prompt_mode, model_label = cell
    return (
        run_root
        / prompt_mode
        / model_label
        / "capture"
        / "confirmation_all_sample"
        / "capture_index.jsonl"
    )


def _cell_output_root(run_root: Path, cell: Cell) -> Path:
    prompt_mode, model_label = cell
    return run_root / prompt_mode / model_label / "representation" / "native_aligned"


def _expected_trajectory_keys() -> set[TrajectoryKey]:
    return {
        (
            "discovery" if seed in DISCOVERY_SEEDS else "confirmation",
            int(seed),
            int(gold_count),
        )
        for seed in (*DISCOVERY_SEEDS, *CONFIRMATION_SEEDS)
        for gold_count in COUNTS
    }


def _key_tuples(
    metadata: pd.DataFrame, columns: Sequence[str]
) -> list[tuple[Any, ...]]:
    missing = sorted(set(columns) - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata lacks alignment columns {missing}")
    result: list[tuple[Any, ...]] = []
    for values in metadata[list(columns)].itertuples(index=False, name=None):
        result.append(
            tuple(
                str(value) if column == "split" else int(value)
                for column, value in zip(columns, values)
            )
        )
    return result


def _key_digest(keys: Iterable[tuple[Any, ...]]) -> str:
    canonical = sorted(set(keys))
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _identity_map(
    metadata: pd.DataFrame, columns: Sequence[str]
) -> dict[tuple[Any, ...], str]:
    if "stimulus_id" not in metadata:
        raise ValueError("Metadata has no stimulus_id for source-identity validation")
    keys = _key_tuples(metadata, columns)
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate alignment keys on {list(columns)}")
    result: dict[tuple[Any, ...], str] = {}
    for key, stimulus_id in zip(keys, metadata["stimulus_id"].astype(str)):
        result[key] = str(stimulus_id)
    return result


def _subset_dataset(
    dataset: ModeDataset,
    keys: set[tuple[Any, ...]],
    *,
    columns: Sequence[str],
) -> ModeDataset:
    observed = _key_tuples(dataset.metadata, columns)
    mask = np.fromiter(
        (key in keys for key in observed), dtype=bool, count=len(observed)
    )
    selected = dataset.metadata.loc[mask].copy()
    selected["_source_row"] = np.flatnonzero(mask)
    order_columns = list(columns)
    selected = selected.sort_values(order_columns, kind="mergesort")
    source_rows = selected.pop("_source_row").to_numpy(dtype=int)
    result = ModeDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=selected.reset_index(drop=True),
        states_by_layer={
            int(layer): np.asarray(states)[source_rows]
            for layer, states in dataset.states_by_layer.items()
        },
    )
    result.validate()
    result_keys = _key_tuples(result.metadata, columns)
    if set(result_keys) != keys or len(result_keys) != len(keys):
        raise ValueError("Exact common-support filtering changed the key set")
    return result


def _support_by_split_and_label(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    support = (
        metadata.groupby(["split", "occurrence"], sort=True)
        .size()
        .rename("state_rows")
        .reset_index()
    )
    return [
        {
            "split": str(row.split),
            "occurrence": int(row.occurrence),
            "state_rows": int(row.state_rows),
        }
        for row in support.itertuples(index=False)
    ]


def _select_one(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ValueError("Cannot select a layer from an empty candidate table")
    return frame.sort_values(
        [
            "discovery_selection_score",
            "discovery_oof_ncc_balanced_accuracy",
            "discovery_oof_logistic_balanced_accuracy",
            "layer",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def _validate_contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path)
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("Native-alignment contract schema changed")
    if contract.get("status") != "FROZEN_ANALYSIS_PATH_CORRECTION":
        raise ValueError("Native-alignment contract is not frozen")
    direct = contract.get("direct_grammar_contrast", {})
    expected_direct = {
        "modes": list(PROMPT_MODES),
        "models": list(MODEL_LABELS),
        "population": "original_registered_all_sample_panel",
        "replacement_rows_allowed": False,
        "identity_fields": ["split", "seed", "gold_count", "stimulus_id"],
        "within_model_only": True,
    }
    for key, expected in expected_direct.items():
        if direct.get(key) != expected:
            raise ValueError(f"Native-alignment contract changed {key}")
    representation = contract.get("representation", {})
    running = representation.get("running_index", {})
    final = representation.get("final_count", {})
    if (
        running.get("site_kind") != RUNNING_SITE
        or running.get("cohort") != "parser_hit"
        or tuple(running.get("alignment_key", ())) != RUNNING_ALIGNMENT_COLUMNS
        or tuple(running.get("labels", ())) != CLASSES
    ):
        raise ValueError("Running-index Native analysis path changed")
    if (
        final.get("site_kind") != FINAL_SITE
        or tuple(final.get("alignment_key", ())) != FINAL_ALIGNMENT_COLUMNS
        or tuple(final.get("labels", ())) != CLASSES
    ):
        raise ValueError("Final-count Native analysis path changed")
    if (
        int(representation.get("pca_dim", -1)) != PCA_DIM
        or int(representation.get("grouped_discovery_cv_folds", -1)) != CV_FOLDS
    ):
        raise ValueError("Native preprocessing dimensions changed")
    return contract


def _validate_original_capture(path: Path, cell: Cell) -> dict[str, Any]:
    prompt_mode, model_label = cell
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _read_jsonl(path)
    observed_keys: list[TrajectoryKey] = []
    request_ids: list[str] = []
    for row in rows:
        seed = int(row["seed"])
        gold_count = int(row["gold_count"])
        split = str(row["split"])
        expected_split = (
            "discovery" if seed in DISCOVERY_SEEDS else "confirmation"
        )
        if split != expected_split:
            raise ValueError(f"Capture split changed for seed {seed}: {split}")
        if str(row.get("model_label")) != model_label:
            raise ValueError(f"Capture model mismatch in {path}")
        row_mode = row.get("prompt_mode")
        if row_mode is not None and str(row_mode) != prompt_mode:
            raise ValueError(f"Capture grammar mismatch in {path}")
        for alias in ("analysis_slot_seed", "v6_analysis_slot_seed"):
            if row.get(alias) is not None and int(row[alias]) != seed:
                raise ValueError(
                    f"Direct grammar contrast contains replacement identity: {alias}"
                )
        if bool(row.get("replacement_applied", False)):
            raise ValueError("Direct grammar contrast contains a replacement row")
        observed_keys.append((split, seed, gold_count))
        request_ids.append(str(row["request_id"]))
    expected = _expected_trajectory_keys()
    if len(rows) != 300 or set(observed_keys) != expected:
        raise ValueError(
            f"{_cell_slug(cell)} all-sample capture is not the exact 300 panel: "
            f"rows={len(rows)}, keys={len(set(observed_keys))}"
        )
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(f"{_cell_slug(cell)} capture reuses request IDs")
    adapter = path.parent / "v6_adapter_manifest.json"
    adapter_value = _read_json(adapter)
    if adapter_value.get("formal_cohort") is not False:
        raise ValueError(f"{_cell_slug(cell)} capture is not the original all-sample view")
    if adapter_value.get("prompt_mode") != prompt_mode:
        raise ValueError(f"{_cell_slug(cell)} adapter grammar changed")
    if adapter_value.get("model_label") != model_label:
        raise ValueError(f"{_cell_slug(cell)} adapter model changed")
    return {
        "capture_index": _artifact(path),
        "capture_adapter": _artifact(adapter),
        "trajectory_rows": len(rows),
        "trajectory_key_sha256": _key_digest(observed_keys),
        "replacement_rows": 0,
    }


def _load_running_datasets(
    run_root: Path, cells: Sequence[Cell]
) -> tuple[dict[Cell, ModeDataset], dict[str, Any]]:
    datasets: dict[Cell, ModeDataset] = {}
    inputs: dict[str, Any] = {}
    for cell in cells:
        index = _capture_index(run_root, cell)
        inputs[_cell_slug(cell)] = _validate_original_capture(index, cell)
        print(
            f"[native-aligned] load running {_cell_slug(cell)} site={RUNNING_SITE}",
            flush=True,
        )
        dataset = load_native_thinking_capture(
            index,
            site_kind=RUNNING_SITE,
            site_policy="uniform",
            cohort="parser_hit",
        )
        if dataset.model_label != cell[1]:
            raise ValueError(f"Loaded running model mismatch for {_cell_slug(cell)}")
        datasets[cell] = dataset
    return datasets, inputs


def _validate_cross_cell_identity(
    datasets: Mapping[Cell, ModeDataset],
    *,
    columns: Sequence[str],
    required_keys: set[tuple[Any, ...]],
) -> dict[str, Any]:
    maps = {
        cell: _identity_map(dataset.metadata, columns)
        for cell, dataset in datasets.items()
    }
    reference_cell = next(iter(maps))
    reference = maps[reference_cell]
    for cell, identities in maps.items():
        if set(identities) != required_keys:
            raise ValueError(f"{_cell_slug(cell)} identity keys are not exact")
        mismatched = [
            key
            for key in sorted(required_keys)
            if identities[key] != reference[key]
        ]
        if mismatched:
            raise ValueError(
                f"Stimulus identity differs between {_cell_slug(reference_cell)} "
                f"and {_cell_slug(cell)} at {mismatched[:5]}"
            )
    digest_payload = [
        [*key, reference[key]] for key in sorted(required_keys)
    ]
    return {
        "status": "PASS_EXACT_SOURCE_IDENTITY",
        "alignment_columns": list(columns),
        "key_count": len(required_keys),
        "key_sha256": _key_digest(required_keys),
        "key_and_stimulus_sha256": hashlib.sha256(
            json.dumps(digest_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "reference_cell": _cell_slug(reference_cell),
        "mismatch_count": 0,
    }


def _analyze_endpoint_layers(
    *,
    endpoint: str,
    cell: Cell,
    dataset: ModeDataset,
    token_site: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prompt_mode, model_label = cell
    rows: list[dict[str, Any]] = []
    layers = sorted(dataset.states_by_layer)
    endpoint_started = time.perf_counter()
    for layer_index, layer in enumerate(layers, 1):
        layer_started = time.perf_counter()
        states = np.asarray(dataset.states_by_layer[layer])
        discovery = grouped_discovery_cv_metrics(
            states,
            dataset.metadata,
            CLASSES,
            pca_dim=PCA_DIM,
            random_state=RANDOM_STATE,
            folds=CV_FOLDS,
            pca_whiten=PCA_WHITEN,
        )
        confirmation = confirmation_metrics(
            states,
            dataset.metadata,
            CLASSES,
            pca_dim=PCA_DIM,
            random_state=RANDOM_STATE,
            pca_whiten=PCA_WHITEN,
        )
        rows.append(
            {
                "endpoint": endpoint,
                "prompt_mode": prompt_mode,
                "grammar": prompt_mode.removeprefix("enumeration_"),
                "model_label": model_label,
                "analysis_group": "all_traces",
                "selector": f"fixed_{token_site}_discovery_only_layer",
                "token_site": token_site,
                "layer": int(layer),
                "retained_labels": " ".join(map(str, CLASSES)),
                "retained_class_count": len(CLASSES),
                "pca_dim": PCA_DIM,
                "pca_whiten": PCA_WHITEN,
                "exact_four_cell_sample_alignment": True,
                **discovery,
                **confirmation,
            }
        )
        print(
            f"[native-aligned] {endpoint} {_cell_slug(cell)} "
            f"layer={layer} ({layer_index}/{len(layers)}) "
            f"seconds={time.perf_counter() - layer_started:.2f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    selected = _select_one(frame).to_dict()
    return frame, {
        **selected,
        "endpoint_elapsed_seconds": time.perf_counter() - endpoint_started,
    }


def _selected_mode_contrasts(
    running_selected: pd.DataFrame, final_selected: pd.DataFrame
) -> pd.DataFrame:
    metrics = (
        "discovery_selection_score",
        "confirmation_logistic_balanced_accuracy",
        "confirmation_ncc_balanced_accuracy",
        "confirmation_class_balanced_snr_db",
    )
    rows: list[dict[str, Any]] = []
    for endpoint, selected in (
        ("running_index", running_selected),
        ("final_count", final_selected),
    ):
        for model_label in MODEL_LABELS:
            model_rows = selected.loc[
                selected["model_label"].astype(str).eq(model_label)
            ]
            by_mode = {
                str(row["prompt_mode"]): row
                for row in model_rows.to_dict(orient="records")
            }
            if set(by_mode) != set(PROMPT_MODES):
                raise ValueError(f"Missing grammar winner for {endpoint}/{model_label}")
            index_row = by_mode["enumeration_index"]
            bullet_row = by_mode["enumeration_bullet"]
            contrast: dict[str, Any] = {
                "endpoint": endpoint,
                "model_label": model_label,
                "contrast": "enumeration_bullet_minus_enumeration_index",
                "exact_original_sample_alignment": True,
                "independent_within_grammar_layer_selection": True,
                "index_selected_layer": int(index_row["layer"]),
                "bullet_selected_layer": int(bullet_row["layer"]),
            }
            for metric in metrics:
                contrast[f"index_{metric}"] = float(index_row[metric])
                contrast[f"bullet_{metric}"] = float(bullet_row[metric])
                contrast[f"bullet_minus_index_{metric}"] = float(
                    bullet_row[metric] - index_row[metric]
                )
            rows.append(contrast)
    return pd.DataFrame(rows)


def analyze_native_aligned_representation(
    *,
    run_root: str | Path,
    output_dir: str | Path,
    contract_path: str | Path,
    project_root: str | Path | None = None,
    command: str | None = None,
) -> dict[str, Path]:
    """Run the frozen two-endpoint Native-thinking representation path."""

    started = time.perf_counter()
    run_root = Path(run_root).resolve()
    output = Path(output_dir).resolve()
    contract_path = Path(contract_path).resolve()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    complete_marker = output / "COMPLETE"
    complete_marker.unlink(missing_ok=True)
    contract = _validate_contract(contract_path)
    cells: tuple[Cell, ...] = tuple(
        (prompt_mode, model_label)
        for prompt_mode in PROMPT_MODES
        for model_label in MODEL_LABELS
    )

    phase_started = time.perf_counter()
    running_raw, input_audits = _load_running_datasets(run_root, cells)
    running_key_sets = {
        cell: set(_key_tuples(dataset.metadata, RUNNING_ALIGNMENT_COLUMNS))
        for cell, dataset in running_raw.items()
    }
    common_running: set[RunningKey] = set.intersection(
        *(set(values) for values in running_key_sets.values())
    )
    if not common_running:
        raise ValueError("Four-cell item_end common support is empty")
    if {key[3] for key in common_running} != set(CLASSES):
        raise ValueError("Four-cell item_end common support lacks count classes 1..10")
    support_counts: dict[tuple[str, int], int] = {}
    for split, _seed, _gold_count, occurrence in common_running:
        key = (str(split), int(occurrence))
        support_counts[key] = support_counts.get(key, 0) + 1
    confirmation_min = min(
        support_counts.get(("confirmation", label), 0) for label in CLASSES
    )
    if confirmation_min < 3:
        raise ValueError(
            "Four-cell item_end confirmation common support has fewer than "
            f"three rows for a class (minimum={confirmation_min})"
        )
    running_aligned = {
        cell: _subset_dataset(
            dataset, common_running, columns=RUNNING_ALIGNMENT_COLUMNS
        )
        for cell, dataset in running_raw.items()
    }
    running_identity = _validate_cross_cell_identity(
        running_aligned,
        columns=RUNNING_ALIGNMENT_COLUMNS,
        required_keys=set(common_running),
    )
    for model_label in MODEL_LABELS:
        layers_by_mode = {
            cell[0]: set(running_aligned[cell].states_by_layer)
            for cell in cells
            if cell[1] == model_label
        }
        if len({tuple(sorted(layers)) for layers in layers_by_mode.values()}) != 1:
            raise ValueError(f"Layer grid differs by grammar for {model_label}")

    running_candidates: list[pd.DataFrame] = []
    running_winners: list[dict[str, Any]] = []
    running_covariance: list[dict[str, Any]] = []
    running_support: dict[str, Any] = {}
    for cell in cells:
        dataset = running_aligned[cell]
        candidates, winner = _analyze_endpoint_layers(
            endpoint="running_index",
            cell=cell,
            dataset=dataset,
            token_site=RUNNING_SITE,
        )
        winner_layer = int(winner["layer"])
        covariance = evaluate_covariance_geometry_layer(
            np.asarray(dataset.states_by_layer[winner_layer]),
            dataset.metadata,
            CLASSES,
            pca_dim=PCA_DIM,
            random_state=RANDOM_STATE,
            relative_ridge=RELATIVE_RIDGE,
            discovery_cv_folds=CV_FOLDS,
        )
        covariance.pop("metric_definitions", None)
        covariance_row = {
            "endpoint": "running_index",
            "prompt_mode": cell[0],
            "model_label": cell[1],
            "token_site": RUNNING_SITE,
            "layer": winner_layer,
            **{f"cov_{key}": value for key, value in covariance.items()},
        }
        running_candidates.append(candidates)
        running_winners.append(winner)
        running_covariance.append(covariance_row)
        running_support[_cell_slug(cell)] = {
            "state_rows": int(len(dataset.metadata)),
            "trajectory_cells": int(
                dataset.metadata[["split", "seed", "gold_count"]]
                .drop_duplicates()
                .shape[0]
            ),
            "layers": sorted(map(int, dataset.states_by_layer)),
            "support": _support_by_split_and_label(dataset.metadata),
            "common_key_sha256": _key_digest(common_running),
        }
    running_frame = pd.concat(running_candidates, ignore_index=True).sort_values(
        ["prompt_mode", "model_label", "layer"], kind="mergesort"
    )
    running_selected = pd.DataFrame(running_winners).sort_values(
        ["prompt_mode", "model_label"], kind="mergesort"
    )
    running_covariance_frame = pd.DataFrame(running_covariance).sort_values(
        ["prompt_mode", "model_label"], kind="mergesort"
    )
    running_seconds = time.perf_counter() - phase_started
    del running_aligned, running_raw
    gc.collect()

    phase_started = time.perf_counter()
    final_raw: dict[Cell, ModeDataset] = {}
    expected_trajectories = _expected_trajectory_keys()
    for cell in cells:
        print(
            f"[native-aligned] load final {_cell_slug(cell)} site={FINAL_SITE}",
            flush=True,
        )
        dataset = load_native_thinking_final_count(_capture_index(run_root, cell))
        observed = _key_tuples(dataset.metadata, FINAL_ALIGNMENT_COLUMNS)
        if len(observed) != 300 or set(observed) != expected_trajectories:
            raise ValueError(
                f"{_cell_slug(cell)} answer_query_v3 is not the exact full panel"
            )
        final_raw[cell] = dataset
    final_identity = _validate_cross_cell_identity(
        final_raw,
        columns=FINAL_ALIGNMENT_COLUMNS,
        required_keys=set(expected_trajectories),
    )
    final_candidates: list[pd.DataFrame] = []
    final_winners: list[dict[str, Any]] = []
    final_support: dict[str, Any] = {}
    for cell in cells:
        dataset = final_raw[cell]
        if set(dataset.states_by_layer) != set(running_support[_cell_slug(cell)]["layers"]):
            raise ValueError(f"Running/final layer grid differs for {_cell_slug(cell)}")
        candidates, winner = _analyze_endpoint_layers(
            endpoint="final_count",
            cell=cell,
            dataset=dataset,
            token_site=FINAL_SITE,
        )
        final_candidates.append(candidates)
        final_winners.append(winner)
        final_support[_cell_slug(cell)] = {
            "trajectory_rows": int(len(dataset.metadata)),
            "discovery_rows": int(
                dataset.metadata["split"].astype(str).eq("discovery").sum()
            ),
            "confirmation_rows": int(
                dataset.metadata["split"].astype(str).eq("confirmation").sum()
            ),
            "trajectory_key_sha256": _key_digest(expected_trajectories),
            "layers": sorted(map(int, dataset.states_by_layer)),
        }
    final_frame = pd.concat(final_candidates, ignore_index=True).sort_values(
        ["prompt_mode", "model_label", "layer"], kind="mergesort"
    )
    final_selected = pd.DataFrame(final_winners).sort_values(
        ["prompt_mode", "model_label"], kind="mergesort"
    )
    final_seconds = time.perf_counter() - phase_started
    del final_raw
    gc.collect()

    contrasts = _selected_mode_contrasts(running_selected, final_selected)
    paths = {
        "running_candidates": output / "running_index_candidate_metrics.csv",
        "running_selected": output / "running_index_selected.csv",
        "running_covariance": output / "running_index_covariance_selected.csv",
        "final_candidates": output / "final_count_candidate_metrics.csv",
        "final_selected": output / "final_count_selected.csv",
        "grammar_contrasts": output / "grammar_contrasts.csv",
        "audit": output / "alignment_audit.json",
        "manifest": output / "analysis_manifest.json",
        "complete": complete_marker,
    }
    _atomic_csv(paths["running_candidates"], running_frame)
    _atomic_csv(paths["running_selected"], running_selected)
    _atomic_csv(paths["running_covariance"], running_covariance_frame)
    _atomic_csv(paths["final_candidates"], final_frame)
    _atomic_csv(paths["final_selected"], final_selected)
    _atomic_csv(paths["grammar_contrasts"], contrasts)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_NATIVE_ANALYSIS_PATH_ALIGNED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_question": contract["scientific_question"],
        "analysis_population": "original_registered_all_sample_panel",
        "replacement_rows_allowed": False,
        "direct_grammar_contrast": "within model only",
        "running_index": {
            "site_kind": RUNNING_SITE,
            "cohort": "parser_hit",
            "alignment_key": list(RUNNING_ALIGNMENT_COLUMNS),
            "exact_four_cell_common_support": True,
            "common_state_rows": len(common_running),
            "common_trajectory_cells": len({key[:3] for key in common_running}),
            "common_key_sha256": _key_digest(common_running),
            "confirmation_support_min": confirmation_min,
            "source_identity": running_identity,
            "support": running_support,
        },
        "final_count": {
            "site_kind": FINAL_SITE,
            "alignment_key": list(FINAL_ALIGNMENT_COLUMNS),
            "exact_full_registered_panel": True,
            "trajectory_rows_per_cell": 300,
            "discovery_rows_per_cell": 200,
            "confirmation_rows_per_cell": 100,
            "source_identity": final_identity,
            "support": final_support,
        },
        "selection_rule": contract["representation"]["layer_selection"],
        "confirmation_policy": contract["representation"]["confirmation_policy"],
        "independent_layer_selection": contract["representation"][
            "independent_layer_selection"
        ],
        "preprocessing": {
            "pca_dim": PCA_DIM,
            "pca_whiten": PCA_WHITEN,
            "grouped_discovery_cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "relative_ridge_covariance_diagnostic": RELATIVE_RIDGE,
        },
        "legacy_generic_ten_site_scan": (
            "historical frozen discovery provenance only; excluded from primary "
            "evidence and not rerun on confirmation"
        ),
    }
    _atomic_json(paths["audit"], audit)

    cell_manifests: dict[str, dict[str, Any]] = {}
    for cell in cells:
        cell_root = _cell_output_root(run_root, cell)
        running_cell = running_frame.loc[
            running_frame["prompt_mode"].astype(str).eq(cell[0])
            & running_frame["model_label"].astype(str).eq(cell[1])
        ].reset_index(drop=True)
        running_selected_cell = running_selected.loc[
            running_selected["prompt_mode"].astype(str).eq(cell[0])
            & running_selected["model_label"].astype(str).eq(cell[1])
        ].reset_index(drop=True)
        final_cell = final_frame.loc[
            final_frame["prompt_mode"].astype(str).eq(cell[0])
            & final_frame["model_label"].astype(str).eq(cell[1])
        ].reset_index(drop=True)
        final_selected_cell = final_selected.loc[
            final_selected["prompt_mode"].astype(str).eq(cell[0])
            & final_selected["model_label"].astype(str).eq(cell[1])
        ].reset_index(drop=True)
        cell_paths = {
            "running_candidates": cell_root / "running_index_candidate_metrics.csv",
            "running_selected": cell_root / "running_index_selected.csv",
            "final_candidates": cell_root / "final_count_candidate_metrics.csv",
            "final_selected": cell_root / "final_count_selected.csv",
        }
        _atomic_csv(cell_paths["running_candidates"], running_cell)
        _atomic_csv(cell_paths["running_selected"], running_selected_cell)
        _atomic_csv(cell_paths["final_candidates"], final_cell)
        _atomic_csv(cell_paths["final_selected"], final_selected_cell)
        cell_manifest_path = cell_root / "cell_manifest.json"
        cell_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS_NATIVE_ANALYSIS_PATH_ALIGNED",
            "prompt_mode": cell[0],
            "model_label": cell[1],
            "analysis_population": "original_registered_all_sample_panel",
            "exact_four_cell_sample_alignment": True,
            "confirmation_used_for_selection": False,
            "replacement_rows": 0,
            "alignment_contract": _artifact(contract_path),
            "capture": input_audits[_cell_slug(cell)],
            "artifacts": {
                name: _artifact(path) for name, path in cell_paths.items()
            },
            "global_alignment_audit": _artifact(paths["audit"]),
        }
        _atomic_json(cell_manifest_path, cell_manifest)
        cell_manifests[_cell_slug(cell)] = _artifact(cell_manifest_path)

    kernel_paths = {
        "cross_mode_loader": root / "src/realistic_niah_v5/cross_mode_geometry.py",
        "dual_endpoint_loader": root / "src/realistic_niah_v5/dual_endpoint_geometry.py",
        "discovery_confirmation_metrics": (
            root / "src/realistic_niah_v5/trace_stratified_geometry.py"
        ),
        "covariance_diagnostic": root / "src/realistic_niah_v5/covariance_geometry.py",
    }
    total_seconds = time.perf_counter() - started
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_NATIVE_ANALYSIS_PATH_ALIGNED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "run_root": str(run_root),
        "output_dir": str(output),
        "alignment_contract": _artifact(contract_path),
        "inputs": input_audits,
        "outputs": {
            name: _artifact(path)
            for name, path in paths.items()
            if name not in {"manifest", "complete"}
        },
        "cell_manifests": cell_manifests,
        "native_numerical_kernels_reused_unchanged": {
            name: _artifact(path) for name, path in kernel_paths.items()
        },
        "selection_split": "discovery",
        "evaluation_split": "confirmation",
        "confirmation_used_for_selection": False,
        "timings_seconds": {
            "running_index": running_seconds,
            "final_count": final_seconds,
            "total": total_seconds,
        },
        "runtime": {
            "device": "CPU",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "package_versions": {
                package: _package_version(package)
                for package in ("numpy", "pandas", "scikit-learn", "scipy")
            },
        },
    }
    _atomic_json(paths["manifest"], manifest)
    paths["complete"].write_text("PASS\n", encoding="utf-8")
    return paths

