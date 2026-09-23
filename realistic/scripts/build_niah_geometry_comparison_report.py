"""Discovery-fitted coordinate exports and data readers for the paper geometry figures."""

from __future__ import annotations

import csv

import gc

import hashlib

import json

import sys

from pathlib import Path

from typing import Any, Iterable, Mapping

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    ModeDataset,
    TRACE_AWARE_SITE_BY_MARKER_KIND,
    load_native_thinking_capture,
    load_non_thinking_capture,
)

from realistic_niah_v5.dual_endpoint_geometry import (  # noqa: E402
    PCA_WHITEN,
    SCHEMA_VERSION as DUAL_ENDPOINT_SCHEMA_VERSION,
    load_native_thinking_final_count,
    load_non_thinking_final_count,
)

MODELS = ("Qwen3-8B", "Gemma4-E4B")

DUAL_ENDPOINT_DIRECTORY = "pca16_whiten"

EXPECTED_FULL_PANEL = {
    "discovery": list(range(1234, 1254)),
    "confirmation": list(range(1254, 1264)),
}

EXPECTED_COUNTS = tuple(range(1, 11))

def expected_trajectory_keys() -> set[tuple[str, int, int]]:
    return {
        (split, int(seed), int(gold_count))
        for split, seeds in EXPECTED_FULL_PANEL.items()
        for seed in seeds
        for gold_count in EXPECTED_COUNTS
    }

def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"empty CSV: {path}")
    return rows

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_dual_endpoint_results(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Load independently selected running-index and final-count results."""

    results: dict[str, dict[str, Any]] = {}
    inputs: list[Path] = []
    for model in MODELS:
        directory = root / model / DUAL_ENDPOINT_DIRECTORY
        paths = {
            "running_candidates": directory / "running_index_candidate_metrics.csv",
            "running_selected": directory / "running_index_selected.csv",
            "eligibility": directory / "running_index_group_eligibility.csv",
            "final_candidates": directory / "final_count_candidate_metrics.csv",
            "final_selected": directory / "final_count_selected.csv",
            "audit": directory / "dual_endpoint_geometry_audit.json",
            "runtime": directory / "runtime_log.json",
        }
        audit = read_json(paths["audit"])
        require(
            audit.get("schema_version") == DUAL_ENDPOINT_SCHEMA_VERSION,
            f"dual-endpoint schema mismatch for {model}",
        )
        require(audit.get("model_label") == model, f"dual-endpoint model mismatch for {model}")
        require(bool(audit.get("pca_whiten")) == PCA_WHITEN, "dual PCA whitening mismatch")
        payload = {
            "audit": audit,
            "runtime": read_json(paths["runtime"]),
            "running_candidates": read_csv(paths["running_candidates"]),
            "running_selected": read_csv(paths["running_selected"]),
            "eligibility": read_csv(paths["eligibility"]),
            "final_candidates": read_csv(paths["final_candidates"]),
            "final_selected": read_csv(paths["final_selected"]),
        }
        for key in ("running_selected", "final_selected"):
            require(
                all(row.get("model_label") == model for row in payload[key]),
                f"dual-endpoint row model mismatch for {model}/{key}",
            )
        results[model] = payload
        inputs.extend(paths.values())
    return results, inputs

def _one_row(rows: Iterable[Mapping[str, Any]], **criteria: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    require(len(matches) == 1, f"expected one row for {criteria}; found {len(matches)}")
    return matches[0]

def fit_dual_display_coordinates(dataset: ModeDataset) -> dict[str, Any]:
    """Fit a separate discovery-only PCA3 display for every available layer."""

    metadata = dataset.metadata.reset_index(drop=True)
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy()
    require(discovery.sum() >= 3, f"{dataset.mode}: too few dual-endpoint discovery rows")
    result: dict[str, Any] = {}
    for layer, values in sorted(dataset.states_by_layer.items()):
        states = np.asarray(values, dtype=np.float32)
        scaler = StandardScaler().fit(states[discovery])
        scaled_discovery = scaler.transform(states[discovery])
        pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(
            scaled_discovery
        )
        coordinates = pca.transform(scaler.transform(states))
        points = [
            [
                str(row.split),
                int(row.seed),
                int(row.occurrence),
                round(float(coordinates[index, 0]), 5),
                round(float(coordinates[index, 1]), 5),
                round(float(coordinates[index, 2]), 5),
                int(row.gold_count),
            ]
            for index, row in enumerate(metadata.itertuples(index=False))
        ]
        result[str(layer)] = {
            "evr": [round(float(value), 6) for value in pca.explained_variance_ratio_],
            "points": points,
        }
    return result

def _dual_metric_curve(
    candidates: Iterable[Mapping[str, Any]], selected: Mapping[str, Any]
) -> dict[str, Any]:
    criteria = {
        key: str(selected[key])
        for key in ("mode", "analysis_group", "selector", "token_site")
    }
    rows = [
        row
        for row in candidates
        if all(str(row.get(key)) == value for key, value in criteria.items())
    ]
    require(rows, f"no dual candidate curve for {criteria}")
    result = {}
    for row in rows:
        result[str(int(row["layer"]))] = {
            "discovery_logistic": float(
                row["discovery_oof_logistic_balanced_accuracy"]
            ),
            "discovery_ncc": float(row["discovery_oof_ncc_balanced_accuracy"]),
            "discovery_score": float(row["discovery_selection_score"]),
            "confirmation_logistic": float(
                row["confirmation_logistic_balanced_accuracy"]
            ),
            "confirmation_ncc": float(row["confirmation_ncc_balanced_accuracy"]),
            "confirmation_snr_db": float(
                row["confirmation_class_balanced_snr_db"]
            ),
        }
    return result

def build_dual_visual_data(
    export_root: Path,
    native_running_root: Path,
    native_final_root: Path,
    dual_results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[Path]]:
    visual: dict[str, Any] = {}
    inputs: list[Path] = []
    for model in MODELS:
        payload = dual_results[model]
        running_selected = payload["running_selected"]
        final_selected = payload["final_selected"]
        running_non_row = _one_row(
            running_selected, mode="non_thinking", analysis_group="all_traces"
        )
        running_native_row = _one_row(
            running_selected,
            mode="native_thinking",
            analysis_group="all_traces",
        )
        final_non_row = _one_row(final_selected, mode="non_thinking")
        final_native_row = _one_row(final_selected, mode="native_thinking")

        non_running_index = (
            export_root
            / model
            / "numeric"
            / "representation"
            / "capture"
            / "capture_index.jsonl"
        )
        native_running_index = native_running_root / model / "capture_index.jsonl"
        non_final_index = (
            export_root
            / model
            / "numeric"
            / "representation"
            / "answer_query_all_layers_v1"
            / "capture_index.jsonl"
        )
        native_final_candidates = (
            native_final_root / model / "capture_index.jsonl",
            native_final_root
            / model
            / "representation"
            / "capture_primary"
            / "capture_index.jsonl",
        )
        native_final_index = next(
            (path for path in native_final_candidates if path.is_file()),
            native_final_candidates[0],
        )
        datasets = {
            "running_non": load_non_thinking_capture(
                non_running_index,
                design_variant="v4.4",
                pooling=str(running_non_row["token_site"]),
            ),
            "running_native": load_native_thinking_capture(
                native_running_index,
                site_kind=str(running_native_row["token_site"]),
                cohort="parser_hit",
            ),
            "final_non": load_non_thinking_final_count(non_final_index),
            "final_native": load_native_thinking_final_count(native_final_index),
        }
        selected_rows = {
            "running_non": running_non_row,
            "running_native": running_native_row,
            "final_non": final_non_row,
            "final_native": final_native_row,
        }
        candidate_rows = {
            "running_non": payload["running_candidates"],
            "running_native": payload["running_candidates"],
            "final_non": payload["final_candidates"],
            "final_native": payload["final_candidates"],
        }
        panels: dict[str, Any] = {}
        for key, dataset in datasets.items():
            selected = selected_rows[key]
            panels[key] = {
                "endpoint": str(selected["endpoint"]),
                "mode": str(selected["mode"]),
                "token_site": str(selected["token_site"]),
                "default_layer": int(selected["layer"]),
                "layers": sorted(dataset.states_by_layer),
                "coordinates": fit_dual_display_coordinates(dataset),
                "metrics": _dual_metric_curve(candidate_rows[key], selected),
            }
        visual[model] = {"panels": panels}
        inputs.extend(
            [
                non_running_index,
                native_running_index,
                non_final_index,
                native_final_index,
            ]
        )
        del datasets
        gc.collect()
    return visual, inputs


def export_geometry_data(*, non_thinking_export_root: Path,
                         native_running_root: Path, native_final_root: Path,
                         dual_endpoint_root: Path, output: Path,
                         manifest_path: Path) -> dict[str, Any]:
    """Export the unchanged DUAL plotting payload directly as JSON."""
    if output.suffix.lower() != ".json":
        raise ValueError("Geometry data output must use a .json suffix")
    results, result_inputs = load_dual_endpoint_results(dual_endpoint_root.resolve())
    visual, capture_inputs = build_dual_visual_data(
        non_thinking_export_root.resolve(), native_running_root.resolve(),
        native_final_root.resolve(), results)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": "niah_geometry_plot_data_v1",
                                 "DUAL": visual}, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    manifest = {"schema_version": "niah_geometry_plot_manifest_v1",
                "inputs": {str(p): sha256(p) for p in sorted(set(result_inputs + capture_inputs))},
                "output": str(output), "output_sha256": sha256(output)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
