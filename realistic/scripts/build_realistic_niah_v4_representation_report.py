from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from realistic_niah_v4.causal_audit import (  # noqa: E402
    audit_screen_8h,
    find_screen_designs,
)
from realistic_niah_v4.answer_query_patching import (  # noqa: E402
    audit_answer_query_patching,
)
from realistic_niah_v4.partitioned_attention import (  # noqa: E402
    partition_sample_metrics,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
POOLINGS = ("span_end", "span_mean")
ATTENTION_POOLINGS = ("span_end", "span_mean", "span_sum")
VARIANTS = ("v4.1", "v4.2", "v4.3", "v4.4")
VARIANT_DESCRIPTIONS = {
    "v4.1": "position, city-score order, and city-score content fixed",
    "v4.2": "position released; order and content fixed",
    "v4.3": "position and city-score order released; content fixed",
    "v4.4": "position, order, and city-score content all released",
}

# Aurora is the project-wide plotting system for V4 and later reports.  The
# chromatic anchors come from the user-provided Aurora reference; intermediate
# count colors are restrained blends of those anchors so that the ordered
# 1--10 trajectory remains legible without introducing a second palette.
AURORA = {
    "midnight_indigo": "#23165C",
    "polar_violet": "#6750E8",
    "ice_cyan": "#00C2FF",
    "aurora_yellow": "#F6E36A",
    "aurora_teal": "#00D4B4",
    "aurora_green": "#39E58C",
    "polar_magenta": "#C04DFF",
    "sunset_pink": "#FF5FA2",
    "night_black": "#161923",
    "snow_white": "#F8FBFF",
    "frost_gray": "#8190A5",
    "warm_brown": "#765347",
}
MODEL_COLORS = {
    "Qwen3-8B": AURORA["polar_violet"],
    "Gemma4-E4B": AURORA["aurora_teal"],
}
VARIANT_COLORS = {
    "v4.1": AURORA["midnight_indigo"],
    "v4.2": AURORA["polar_violet"],
    "v4.3": AURORA["ice_cyan"],
    "v4.4": AURORA["sunset_pink"],
}
POOLING_COLORS = {
    "span_end": AURORA["ice_cyan"],
    "span_mean": AURORA["sunset_pink"],
    "span_sum": AURORA["aurora_yellow"],
}
COUNT_COLORS = (
    "#23165C",
    "#4430A2",
    "#6750E8",
    "#9950F4",
    "#C04DFF",
    "#FF5FA2",
    "#F6E36A",
    "#39E58C",
    "#00D4B4",
    "#00C2FF",
)
PHENOTYPE_COLORS = {
    "global_endpoint_aggregator": AURORA["aurora_green"],
    "partition_local_endpoint_aggregator": AURORA["ice_cyan"],
    "first_needle_locator": AURORA["aurora_yellow"],
    "targeted_occurrence_retriever": AURORA["sunset_pink"],
    "occurrence_endpoint_selector": AURORA["sunset_pink"],
    "other": AURORA["frost_gray"],
}
MODEL_HEAD_GRIDS = {
    "Qwen3-8B": (36, 32),
    "Gemma4-E4B": (42, 8),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _number(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    prefix = "+" if signed and numeric > 0 else ""
    return f"{prefix}{numeric:.{digits}f}"


def _image_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _primary_layers(model_root: Path) -> dict[str, int]:
    payload = _read_json(
        model_root / "representation" / "analysis" / "representation_summary.json"
    )
    return {
        str(pooling): int(layer)
        for pooling, layer in payload["primary_layer_selection"]["layers"].items()
    }


def _grouped_pca_ridge_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
) -> tuple[float, float]:
    """Discovery-only grouped-seed ridge CV in an already fitted PCA space.

    PCA is deliberately fitted once on all discovery states because this metric
    diagnoses how much of the discovery count signal is visible in the plotted
    coordinates.  It is not reported as a held-out confirmation estimate.
    """

    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    scores: list[tuple[float, float]] = []
    for alpha in alphas:
        fold_scores: list[float] = []
        for train, validation in splitter.split(x, y, groups):
            estimator = make_pipeline(
                StandardScaler(), Ridge(alpha=float(alpha), solver="lsqr")
            )
            estimator.fit(x[train], y[train])
            fold_scores.append(
                float(r2_score(y[validation], estimator.predict(x[validation])))
            )
        scores.append((float(alpha), float(np.mean(fold_scores))))
    return max(scores, key=lambda item: item[1])


def _select_manifold_layer(
    rows: list[dict[str, Any]],
    *,
    cv_tolerance: float = 0.02,
) -> int:
    """Choose a discovery-only 3D display layer without confirmation leakage.

    First retain layers whose full-space grouped-seed CV-R2 is within
    ``cv_tolerance`` of the probe optimum.  Among those layers, maximize the
    multiplicative 3D manifold-fidelity score

        M3 = EVR3 * signal_capture3 * 1 / (1 + LOO_noise_to_signal).

    The gate prevents a visually compact but weakly count-decodable layer from
    winning.  The product requires all three descriptive properties rather than
    allowing one large term to compensate additively for a failed term.
    """

    if not rows:
        raise ValueError("Cannot select a manifold layer from an empty sweep")
    maximum_cv = max(float(row["full_space_discovery_cv_r2"]) for row in rows)
    candidates = [
        row
        for row in rows
        if float(row["full_space_discovery_cv_r2"])
        >= maximum_cv - float(cv_tolerance)
    ]
    selected = sorted(
        candidates,
        key=lambda row: (-float(row["manifold_fidelity_m3"]), int(row["layer"])),
    )[0]
    return int(selected["layer"])


def _layer_sweep(
    model_root: Path,
    *,
    model: str,
    probe_layers: dict[str, int],
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    dict[str, dict[int, PCA]],
]:
    """Compute discovery-only PCA/manifold diagnostics for every decoder block."""

    capture_root = model_root / "representation" / "capture"
    records = [
        record
        for record in _read_jsonl(capture_root / "capture_index.jsonl")
        if str(record["design_variant"]) == "v4.1"
        and str(record["split"]) == "discovery"
    ]
    records.sort(key=lambda record: int(record["seed"]))
    if len(records) != 20:
        raise RuntimeError(
            f"{model}: expected 20 v4.1 discovery representation shards"
        )
    metrics = pd.read_csv(model_root / "representation" / "analysis" /
                          "representation_layer_metrics.csv")
    all_rows: list[dict[str, Any]] = []
    display_layers: dict[str, int] = {}
    pca_models: dict[str, dict[int, PCA]] = {pooling: {} for pooling in POOLINGS}
    for pooling in POOLINGS:
        arrays: list[np.ndarray] = []
        layer_indices: np.ndarray | None = None
        seeds: list[int] = []
        for record in records:
            shard = capture_root / str(record["shard_path"])
            with np.load(shard, allow_pickle=False) as payload:
                current_layers = np.asarray(payload["layer_indices"], dtype=int)
                if layer_indices is None:
                    layer_indices = current_layers
                elif not np.array_equal(layer_indices, current_layers):
                    raise RuntimeError(f"{model}/{pooling}: inconsistent layer grid")
                arrays.append(np.asarray(payload[pooling]))
            seeds.append(int(record["seed"]))
        assert layer_indices is not None
        states = np.stack(arrays, axis=0)
        count_labels = np.tile(np.arange(1, 11, dtype=float), len(states))
        seed_labels = np.repeat(np.asarray(seeds, dtype=int), 10)
        pooling_rows: list[dict[str, Any]] = []
        for layer_axis, layer in enumerate(layer_indices):
            layer_states = np.asarray(states[:, layer_axis], dtype=np.float32)
            flat = layer_states.reshape(-1, layer_states.shape[-1])
            pca = PCA(n_components=6, svd_solver="randomized", random_state=0)
            projected = pca.fit_transform(flat)
            pca_models[pooling][int(layer)] = pca
            pca_alpha, pca3_cv_r2 = _grouped_pca_ridge_cv(
                projected[:, :3], count_labels, seed_labels
            )
            centroids = np.stack(
                [flat[count_labels == count].mean(axis=0) for count in range(1, 11)]
            )
            grand = centroids.mean(axis=0, keepdims=True)
            centered_centroids = centroids - grand
            full_signal = float(np.sum(centered_centroids * centered_centroids))
            projected_signal = float(
                np.sum((centered_centroids @ pca.components_[:3].T) ** 2)
            )
            signal_capture = (
                projected_signal / full_signal if full_signal > 0 else math.nan
            )
            seed_sum = layer_states.sum(axis=0, dtype=np.float64)
            leave_one_out_centroids = (
                seed_sum[None, :, :] - layer_states.astype(np.float64)
            ) / (len(layer_states) - 1)
            leave_one_out_residual = (
                layer_states.astype(np.float64) - leave_one_out_centroids
            )
            noise_rms = float(
                np.sqrt(np.mean(np.sum(leave_one_out_residual**2, axis=-1)))
            )
            signal_rms = float(
                np.sqrt(np.mean(np.sum(centered_centroids**2, axis=-1)))
            )
            noise_to_signal = (
                noise_rms / signal_rms if signal_rms > 0 else math.nan
            )
            compactness = (
                1.0 / (1.0 + noise_to_signal)
                if math.isfinite(noise_to_signal)
                else math.nan
            )
            metric = metrics[
                (metrics["design_variant"] == "v4.1")
                & (metrics["pooling"] == pooling)
                & (pd.to_numeric(metrics["layer"]).astype(int) == int(layer))
            ]
            if len(metric) != 1:
                raise RuntimeError(
                    f"{model}/{pooling}/L{layer}: missing unique layer metric"
                )
            metric_row = metric.iloc[0]
            evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
            row = {
                "model": model,
                "pooling": pooling,
                "layer": int(layer),
                "full_space_discovery_cv_r2": float(
                    metric_row["discovery_group_cv_r2"]
                ),
                "pca3_discovery_cv_r2": float(pca3_cv_r2),
                "pca3_ridge_alpha": float(pca_alpha),
                "pca_evr_pc1": float(evr[0]),
                "pca_evr_pc1_2": float(evr[:2].sum()),
                "pca_evr_pc1_3": float(evr[:3].sum()),
                "pca_evr_pc1_6": float(evr[:6].sum()),
                "count_signal_capture_pc1_3": float(signal_capture),
                "discovery_loo_noise_to_signal": float(noise_to_signal),
                "discovery_compactness": float(compactness),
                "centroid_step_cv": float(
                    metric_row["centroid_adjacent_step_cv"]
                ),
                "centroid_path_to_chord": float(
                    metric_row["centroid_path_to_chord_ratio"]
                ),
                "centroid_successive_cosine": float(
                    metric_row["centroid_adjacent_direction_cosine"]
                ),
            }
            row["manifold_fidelity_m3"] = float(
                row["pca_evr_pc1_3"]
                * row["count_signal_capture_pc1_3"]
                * row["discovery_compactness"]
            )
            pooling_rows.append(row)
        selected_layer = _select_manifold_layer(pooling_rows)
        display_layers[pooling] = selected_layer
        maximum_cv = max(
            float(row["full_space_discovery_cv_r2"]) for row in pooling_rows
        )
        for row in pooling_rows:
            row["within_decodability_gate"] = bool(
                float(row["full_space_discovery_cv_r2"]) >= maximum_cv - 0.02
            )
            row["probe_optimal"] = bool(
                int(row["layer"]) == int(probe_layers[pooling])
            )
            row["manifold_display"] = bool(
                int(row["layer"]) == int(selected_layer)
            )
        all_rows.extend(pooling_rows)
        del states
    return all_rows, display_layers, pca_models


def _n10_labels(
    model_root: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], pd.DataFrame]:
    labels = pd.read_csv(model_root / "behavior" / "capture" / "generation_labels.csv")
    labels = labels[labels["gold_count"].astype(int) == 10].copy()
    if labels.duplicated(["design_variant", "seed"]).any():
        raise ValueError("N=10 labels are not unique by variant and seed")
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in labels.to_dict("records"):
        outcome = str(row.get("outcome_group", "wrong"))
        if _bool(row.get("is_correct")):
            outcome = "correct"
        elif not _bool(row.get("format_valid")):
            outcome = "invalid"
        else:
            outcome = "wrong"
        lookup[(str(row["design_variant"]), int(row["seed"]))] = {
            "outcome": outcome,
            "parsed_count": (
                None if pd.isna(row.get("parsed_count")) else int(row["parsed_count"])
            ),
            "count_error": (
                None if pd.isna(row.get("count_error")) else int(row["count_error"])
            ),
        }
    return lookup, labels


def _load_projection(
    model_root: Path,
    *,
    model: str,
    pooling: str,
    layer: int,
    labels: dict[tuple[str, int], dict[str, Any]],
    components: int = 6,
) -> dict[str, Any]:
    capture_root = model_root / "representation" / "capture"
    records = _read_jsonl(capture_root / "capture_index.jsonl")
    tensors: dict[str, list[tuple[int, str, np.ndarray]]] = {
        variant: [] for variant in VARIANTS
    }
    for record in records:
        variant = str(record["design_variant"])
        if variant not in tensors:
            continue
        shard = capture_root / str(record["shard_path"])
        with np.load(shard, allow_pickle=False) as payload:
            layer_indices = np.asarray(payload["layer_indices"], dtype=int)
            match = np.flatnonzero(layer_indices == int(layer))
            if len(match) != 1:
                raise RuntimeError(
                    f"{model}/{pooling}: layer {layer} absent in {shard}"
                )
            states = np.asarray(payload[pooling][int(match[0])], dtype=np.float32)
        if states.shape[0] != 10:
            raise RuntimeError(f"Expected ten occurrence states, got {states.shape}")
        tensors[variant].append((int(record["seed"]), str(record["split"]), states))
    for variant in VARIANTS:
        tensors[variant].sort(key=lambda item: item[0])
        if len(tensors[variant]) != 30:
            raise RuntimeError(
                f"{model}/{pooling}/{variant}: expected 30 seed captures"
            )

    reference = np.stack(
        [states for _seed, split, states in tensors["v4.1"] if split == "discovery"],
        axis=0,
    )
    fit = reference.reshape(-1, reference.shape[-1])
    pca = PCA(n_components=int(components), svd_solver="randomized", random_state=0)
    pca.fit(fit)

    rows: list[list[Any]] = []
    for variant in VARIANTS:
        for seed, split, states in tensors[variant]:
            projected = pca.transform(states)
            label = labels.get((variant, seed))
            if label is None:
                raise RuntimeError(
                    f"Missing final-output label for {model}/{variant}/seed{seed}"
                )
            for count_index, point in enumerate(projected, start=1):
                rows.append(
                    [
                        variant,
                        int(seed),
                        split,
                        label["outcome"],
                        label["parsed_count"],
                        label["count_error"],
                        int(count_index),
                        *[round(float(value), 6) for value in point],
                    ]
                )
    return {
        "model": model,
        "pooling": pooling,
        "layer": int(layer),
        "fit_variant": "v4.1",
        "fit_split": "discovery",
        "explained_variance_ratio": [
            round(float(value), 8) for value in pca.explained_variance_ratio_
        ],
        "rows": rows,
    }


def _load_prompt_projection_layers(
    model_root: Path,
    *,
    model: str,
    labels: dict[tuple[str, int], dict[str, Any]],
    pca_models: dict[str, dict[int, PCA]],
    layer_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Project every captured post-block layer for the interactive viewer."""

    capture_root = model_root / "representation" / "capture"
    records = _read_jsonl(capture_root / "capture_index.jsonl")
    projections: dict[str, dict[str, Any]] = {}
    for pooling in POOLINGS:
        tensors: dict[str, list[tuple[int, str, np.ndarray]]] = {
            variant: [] for variant in VARIANTS
        }
        layer_indices: np.ndarray | None = None
        for record in records:
            variant = str(record["design_variant"])
            if variant not in tensors:
                continue
            shard = capture_root / str(record["shard_path"])
            with np.load(shard, allow_pickle=False) as payload:
                current_layers = np.asarray(payload["layer_indices"], dtype=int)
                if layer_indices is None:
                    layer_indices = current_layers
                elif not np.array_equal(layer_indices, current_layers):
                    raise RuntimeError(f"{model}/{pooling}: inconsistent layer grid")
                states = np.asarray(payload[pooling])
            tensors[variant].append(
                (int(record["seed"]), str(record["split"]), states)
            )
        assert layer_indices is not None
        for variant in VARIANTS:
            tensors[variant].sort(key=lambda item: item[0])
            if len(tensors[variant]) != 30:
                raise RuntimeError(
                    f"{model}/{pooling}/{variant}: expected 30 seed captures"
                )
        layer_lookup = {
            int(row["layer"]): row
            for row in layer_rows
            if row["model"] == model and row["pooling"] == pooling
        }
        for layer_axis, layer in enumerate(layer_indices):
            pca = pca_models[pooling][int(layer)]
            rows: list[list[Any]] = []
            for variant in VARIANTS:
                for seed, split, states_by_layer in tensors[variant]:
                    states = np.asarray(
                        states_by_layer[int(layer_axis)], dtype=np.float32
                    )
                    projected = pca.transform(states)
                    label = labels.get((variant, seed))
                    if label is None:
                        raise RuntimeError(
                            f"Missing final-output label for {model}/{variant}/seed{seed}"
                        )
                    for count_index, point in enumerate(projected, start=1):
                        rows.append(
                            [
                                variant,
                                int(seed),
                                split,
                                label["outcome"],
                                label["parsed_count"],
                                label["count_error"],
                                int(count_index),
                                *[round(float(value), 6) for value in point],
                            ]
                        )
            diagnostic = layer_lookup[int(layer)]
            key = f"{model}|{pooling}|{int(layer)}"
            projections[key] = {
                "model": model,
                "pooling": pooling,
                "layer": int(layer),
                "fit_variant": "v4.1",
                "fit_split": "discovery",
                "probe_optimal": bool(diagnostic["probe_optimal"]),
                "manifold_display": bool(diagnostic["manifold_display"]),
                "manifold_fidelity_m3": round(
                    float(diagnostic["manifold_fidelity_m3"]), 8
                ),
                "explained_variance_ratio": [
                    round(float(value), 8)
                    for value in pca.explained_variance_ratio_
                ],
                "rows": rows,
            }
        del tensors
    return projections


def _answer_query_projection_data(
    run_root: Path,
) -> dict[str, dict[str, Any]]:
    """PCA sensitivity views of saved discovery answer-query states.

    Two bases are fit per saved layer.  ``all`` uses every V4.1 discovery
    prompt; ``correct_only`` uses only V4.1 discovery prompts whose actual
    greedy continuation is strictly correct.  Both bases project every saved
    prompt so that fit-cohort sensitivity is separable from the displayed
    correct/wrong/invalid outcome filter.
    """

    result: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        all_layer_capture = (
            run_root
            / model
            / "numeric"
            / "representation"
            / "answer_query_all_layers_v1"
        )
        if (all_layer_capture / "capture_index.jsonl").exists():
            capture_root = all_layer_capture
            capture_source = "answer_query_all_layers_v1"
        else:
            family_root = (
                run_root
                / model
                / "numeric"
                / "causal"
                / "geometric_steering_v1"
            )
            candidates: list[Path] = []
            for candidate in sorted(family_root.glob("design_*")):
                design_path = candidate / "design.json"
                if not design_path.exists():
                    continue
                design = _read_json(design_path)
                if design.get("methods") == ["centroid_delta"]:
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise RuntimeError(
                    f"{model}: expected one completed centroid-delta steering design"
                )
            capture_root = candidates[0] / "discovery_capture"
            capture_source = "geometric_steering_v1_discovery_capture"
        records = _read_jsonl(capture_root / "capture_index.jsonl")
        labels = pd.read_csv(
            run_root
            / model
            / "numeric"
            / "behavior"
            / "capture"
            / "generation_labels.csv"
        )
        label_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
        for label in labels.to_dict("records"):
            key = (
                str(label["design_variant"]),
                int(label["seed"]),
                int(label["gold_count"]),
            )
            outcome = "correct" if _bool(label["is_correct"]) else (
                "wrong" if _bool(label["format_valid"]) else "invalid"
            )
            label_lookup[key] = {
                "outcome": outcome,
                "predicted_count": (
                    None
                    if pd.isna(label.get("parsed_count"))
                    else int(label["parsed_count"])
                ),
                "count_error": (
                    None
                    if pd.isna(label.get("count_error"))
                    else int(label["count_error"])
                ),
            }
        metadata: list[tuple[str, int, int, str, int | None, int | None]] = []
        state_rows: list[np.ndarray] = []
        layer_indices: np.ndarray | None = None
        for record in records:
            shard = capture_root / str(record["shard_path"])
            with np.load(_native_open_path(shard), allow_pickle=False) as payload:
                current_layers = np.asarray(payload["layer_indices"], dtype=int)
                if layer_indices is None:
                    layer_indices = current_layers
                elif not np.array_equal(layer_indices, current_layers):
                    raise RuntimeError(f"{model}: query-state layer grid changed")
                states = np.asarray(payload["query_states"], dtype=np.float32)
            state_rows.append(states)
            key = (
                str(record["design_variant"]),
                int(record["seed"]),
                int(record["count"]),
            )
            if key not in label_lookup:
                raise RuntimeError(f"{model}: missing answer-query outcome {key}")
            label = label_lookup[key]
            metadata.append(
                (
                    *key,
                    str(label["outcome"]),
                    label["predicted_count"],
                    label["count_error"],
                )
            )
        assert layer_indices is not None
        states = np.stack(state_rows, axis=0)
        if states.shape[:2] != (800, len(layer_indices)):
            raise RuntimeError(
                f"{model}: unexpected answer-query discovery shape {states.shape}"
            )
        all_reference = np.asarray(
            [item[0] == "v4.1" for item in metadata], dtype=bool
        )
        correct_reference = np.asarray(
            [item[0] == "v4.1" and item[3] == "correct" for item in metadata],
            dtype=bool,
        )
        all_fit_diagnostics: list[dict[str, Any]] = []
        model_result_keys: list[str] = []
        for layer_axis, layer in enumerate(layer_indices):
            for fit_cohort, reference_mask in (
                ("all", all_reference),
                ("correct_only", correct_reference),
            ):
                fit_count = int(reference_mask.sum())
                if fit_count < 6:
                    raise RuntimeError(
                        f"{model}/L{layer}: only {fit_count} {fit_cohort} rows"
                    )
                pca = PCA(
                    n_components=6,
                    svd_solver="randomized",
                    random_state=0,
                )
                pca.fit(states[reference_mask, int(layer_axis)])
                projected = pca.transform(states[:, int(layer_axis)])
                common_eval = states[all_reference, int(layer_axis)]
                common_projected = projected[all_reference]
                common_total_variance = float(
                    np.var(common_eval, axis=0, ddof=1).sum()
                )
                common_capture = [
                    float(
                        np.var(common_projected[:, :components], axis=0, ddof=1).sum()
                        / max(common_total_variance, 1e-12)
                    )
                    for components in range(1, 7)
                ]
                common_counts = np.asarray(
                    [int(item[2]) for item, keep in zip(metadata, all_reference) if keep],
                    dtype=int,
                )
                common_seeds = np.asarray(
                    [int(item[1]) for item, keep in zip(metadata, all_reference) if keep],
                    dtype=int,
                )
                full_centroids = np.stack(
                    [
                        common_eval[common_counts == count].mean(axis=0)
                        for count in range(1, 11)
                    ]
                )
                centered_full_centroids = full_centroids - full_centroids.mean(
                    axis=0, keepdims=True
                )
                full_signal = float(np.sum(centered_full_centroids**2))
                projected_signal = float(
                    np.sum(
                        (
                            centered_full_centroids
                            @ pca.components_[:3].T
                        )
                        ** 2
                    )
                )
                signal_capture = (
                    projected_signal / full_signal if full_signal > 0 else math.nan
                )
                noise_residuals: list[np.ndarray] = []
                for count in range(1, 11):
                    group = common_eval[common_counts == count]
                    group_sum = group.sum(axis=0, dtype=np.float64)
                    noise_residuals.append(
                        group.astype(np.float64)
                        - (group_sum[None, :] - group.astype(np.float64))
                        / (len(group) - 1)
                    )
                noise_rms = float(
                    np.sqrt(
                        np.mean(
                            np.sum(np.concatenate(noise_residuals, axis=0) ** 2, axis=1)
                        )
                    )
                )
                signal_rms = float(
                    np.sqrt(np.mean(np.sum(centered_full_centroids**2, axis=1)))
                )
                noise_to_signal = (
                    noise_rms / signal_rms if signal_rms > 0 else math.nan
                )
                compactness = (
                    1.0 / (1.0 + noise_to_signal)
                    if math.isfinite(noise_to_signal)
                    else math.nan
                )
                pca3_alpha, pca3_cv_r2 = _grouped_pca_ridge_cv(
                    common_projected[:, :3],
                    common_counts.astype(float),
                    common_seeds,
                )
                rows: list[list[Any]] = []
                for (
                    variant,
                    seed,
                    count,
                    outcome,
                    predicted_count,
                    count_error,
                ), point in zip(metadata, projected):
                    rows.append(
                        [
                            variant,
                            int(seed),
                            int(count),
                            outcome,
                            predicted_count,
                            count_error,
                            *[round(float(value), 6) for value in point],
                        ]
                    )
                fit_count_support = {
                    str(count): int(
                        sum(
                            bool(keep) and int(item[2]) == count
                            for keep, item in zip(reference_mask, metadata)
                        )
                    )
                    for count in range(1, 11)
                }
                result_key = f"{model}|{int(layer)}|{fit_cohort}"
                model_result_keys.append(result_key)
                result[result_key] = {
                    "model": model,
                    "layer": int(layer),
                    "position": "answer_query",
                    "capture_source": capture_source,
                    "captured_layer_count": int(len(layer_indices)),
                    "fit_variant": "v4.1",
                    "fit_split": "discovery",
                    "fit_cohort": fit_cohort,
                    "fit_rows": fit_count,
                    "fit_count_support": fit_count_support,
                    "fit_outcome_counts": {
                        outcome: int(
                            sum(
                                bool(keep) and item[3] == outcome
                                for keep, item in zip(reference_mask, metadata)
                            )
                        )
                        for outcome in ("correct", "wrong", "invalid")
                    },
                    "explained_variance_ratio": [
                        round(float(value), 8)
                        for value in pca.explained_variance_ratio_
                    ],
                    "common_v41_variance_capture": [
                        round(value, 8) for value in common_capture
                    ],
                    "pca3_discovery_cv_r2": round(float(pca3_cv_r2), 8),
                    "pca3_ridge_alpha": float(pca3_alpha),
                    "count_signal_capture_pc1_3": round(float(signal_capture), 8),
                    "discovery_loo_noise_to_signal": round(
                        float(noise_to_signal), 8
                    ),
                    "discovery_compactness": round(float(compactness), 8),
                    "manifold_fidelity_m3": round(
                        float(
                            np.sum(pca.explained_variance_ratio_[:3])
                            * signal_capture
                            * compactness
                        ),
                        8,
                    ),
                    "rows": rows,
                }
                if fit_cohort == "all":
                    all_fit_diagnostics.append(
                        {
                            "layer": int(layer),
                            "pca3_discovery_cv_r2": float(pca3_cv_r2),
                            "manifold_fidelity_m3": float(
                                np.sum(pca.explained_variance_ratio_[:3])
                                * signal_capture
                                * compactness
                            ),
                        }
                    )
        maximum_cv = max(
            float(row["pca3_discovery_cv_r2"])
            for row in all_fit_diagnostics
        )
        gated = [
            row
            for row in all_fit_diagnostics
            if float(row["pca3_discovery_cv_r2"]) >= maximum_cv - 0.02
        ]
        manifold_layer = int(
            sorted(
                gated,
                key=lambda row: (
                    -float(row["manifold_fidelity_m3"]),
                    int(row["layer"]),
                ),
            )[0]["layer"]
        )
        probe_layer = int(
            sorted(
                all_fit_diagnostics,
                key=lambda row: (
                    -float(row["pca3_discovery_cv_r2"]),
                    int(row["layer"]),
                ),
            )[0]["layer"]
        )
        for key in model_result_keys:
            result[key]["probe_optimal"] = bool(result[key]["layer"] == probe_layer)
            result[key]["manifold_display"] = bool(
                result[key]["layer"] == manifold_layer
            )
        del states
    return result


def _answer_query_pca_sensitivity_rows(
    projections: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize answer-query PCA fit-cohort sensitivity on one evaluation set.

    Every geometry diagnostic is evaluated on all V4.1 discovery prompts,
    irrespective of the PCA fitting cohort.  This keeps the plotted states
    fixed while changing only the basis used to represent them.
    """

    rows: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, int, str], np.ndarray] = {}
    for data in projections.values():
        selected = [row for row in data["rows"] if row[0] == "v4.1"]
        coordinates = np.asarray(
            [[float(value) for value in row[6:12]] for row in selected],
            dtype=float,
        )
        counts = np.asarray([int(row[2]) for row in selected], dtype=int)
        centroids = np.stack(
            [coordinates[counts == count].mean(axis=0) for count in range(1, 11)]
        )
        trajectory = centroids[:, :3]
        trajectories[(data["model"], int(data["layer"]), data["fit_cohort"])] = (
            trajectory
        )
        steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        chord = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
        centroid_by_row = np.stack([centroids[count - 1, :3] for count in counts])
        within_rms = float(
            np.sqrt(np.mean(np.sum((coordinates[:, :3] - centroid_by_row) ** 2, axis=1)))
        )
        grand_centroid = trajectory.mean(axis=0)
        between_rms = float(
            np.sqrt(np.mean(np.sum((trajectory - grand_centroid) ** 2, axis=1)))
        )
        support = [int(data["fit_count_support"][str(count)]) for count in range(1, 11)]
        evr = [float(value) for value in data["explained_variance_ratio"]]
        common_capture = [
            float(value) for value in data["common_v41_variance_capture"]
        ]
        rows.append(
            {
                "model": data["model"],
                "layer": int(data["layer"]),
                "fit_cohort": data["fit_cohort"],
                "fit_rows": int(data["fit_rows"]),
                "fit_count_support_min": min(support),
                "fit_count_support_max": max(support),
                "fit_evr_pc1_3": float(sum(evr[:3])),
                "fit_evr_pc1_6": float(sum(evr[:6])),
                "common_v41_capture_pc1_3": common_capture[2],
                "common_v41_capture_pc1_6": common_capture[5],
                "step_cv": float(np.std(steps) / max(np.mean(steps), 1e-12)),
                "path_chord": float(steps.sum() / max(chord, 1e-12)),
                "within_count_seed_rms": within_rms,
                "between_count_centroid_rms": between_rms,
                "seed_noise_to_count_signal": float(
                    within_rms / max(between_rms, 1e-12)
                ),
            }
        )

    for row in rows:
        key = (row["model"], int(row["layer"]))
        candidate = trajectories[(*key, row["fit_cohort"])]
        baseline = trajectories[(*key, "all")]
        candidate_distances = np.linalg.norm(
            candidate[:, None, :] - candidate[None, :, :], axis=2
        )[np.triu_indices(10, k=1)]
        baseline_distances = np.linalg.norm(
            baseline[:, None, :] - baseline[None, :, :], axis=2
        )[np.triu_indices(10, k=1)]
        if np.std(candidate_distances) <= 1e-12 or np.std(baseline_distances) <= 1e-12:
            correlation = math.nan
        else:
            correlation = float(
                np.corrcoef(candidate_distances, baseline_distances)[0, 1]
            )
        row["centroid_distance_corr_to_all"] = correlation
    return sorted(
        rows,
        key=lambda row: (
            MODELS.index(str(row["model"])),
            int(row["layer"]),
            0 if row["fit_cohort"] == "all" else 1,
        ),
    )


def _metric_rows(
    run_root: Path, primary: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        analysis = run_root / model / "numeric" / "representation" / "analysis"
        metrics = pd.read_csv(analysis / "representation_layer_metrics.csv")
        for pooling in POOLINGS:
            layer = primary[model][pooling]
            selected = metrics[
                (metrics["pooling"] == pooling)
                & (metrics["layer"].astype(int) == layer)
            ]
            for row in selected.to_dict("records"):
                rows.append(row)
    return rows


def _behavior_rows(labels_by_model: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, labels in labels_by_model.items():
        for (variant, split), frame in labels.groupby(
            ["design_variant", "split"], sort=True
        ):
            parsed = pd.to_numeric(frame["parsed_count"], errors="coerce")
            rows.append(
                {
                    "model": model,
                    "variant": str(variant),
                    "split": str(split),
                    "n": int(len(frame)),
                    "correct": int(frame["is_correct"].map(_bool).sum()),
                    "accuracy": float(frame["is_correct"].map(_bool).mean()),
                    "mean_prediction": float(parsed.mean()),
                    "mae": float(
                        pd.to_numeric(frame["count_error"], errors="coerce")
                        .abs()
                        .mean()
                    ),
                }
            )
    return rows


def _sensitivity_rows(run_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            run_root
            / model
            / "numeric"
            / "representation"
            / "analysis"
            / "seed_sensitivity_paired_bootstrap.csv"
        )
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            result.append({"model": model, **row})
    return result


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _seed_bootstrap(
    values: np.ndarray,
    *,
    label: str,
    iterations: int = 20_000,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    estimate = float(values.mean())
    rng = np.random.default_rng(_stable_seed(label))
    sampled = values[
        rng.integers(0, values.size, size=(int(iterations), values.size))
    ].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return estimate, float(low), float(high)


def _exact_sign_flip_p(values: np.ndarray) -> float:
    """Two-sided exact sign-flip p-value for one seed-level paired contrast."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    if values.size > 20:
        raise ValueError("Exact sign-flip enumeration is capped at 20 clusters")
    observed = abs(float(values.mean()))
    assignments = np.arange(1 << values.size, dtype=np.uint64)[:, None]
    bits = (assignments >> np.arange(values.size, dtype=np.uint64)) & 1
    signs = bits.astype(float) * 2.0 - 1.0
    permuted = np.abs(signs @ values / values.size)
    return float(np.mean(permuted >= observed - 1e-12))


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    finite = [index for index, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(finite, key=lambda index: p_values[index])
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _p_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    if numeric < 0.001:
        return "&lt;0.001"
    return f"{numeric:.3f}"


def _span_end_undercount_frame(
    run_root: Path, model: str, *, pooling: str = "span_end"
) -> pd.DataFrame:
    path = _attention_analysis_root(run_root, model) / "tables" / "omission_diagnostics.csv"
    frame = pd.read_csv(path)
    frame = frame[
        (frame["split"] == "confirmation")
        & (frame["pooling"] == pooling)
        & (pd.to_numeric(frame["omission_count"], errors="coerce") > 0)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No confirmation {pooling} undercounts in {path}")
    selected_counts = pd.to_numeric(
        frame["selected_head_count"], errors="raise"
    ).astype(int)
    if not (selected_counts == 8).all():
        raise RuntimeError(f"Expected an eight-head discovery ensemble in {path}")
    frame["count"] = pd.to_numeric(frame["count"], errors="raise").astype(int)
    frame["omission_count"] = pd.to_numeric(
        frame["omission_count"], errors="raise"
    ).astype(int)
    invalid_k = (frame["omission_count"] <= 0) | (
        frame["omission_count"] > frame["count"]
    )
    if invalid_k.any():
        raise RuntimeError(f"Invalid undercount magnitude in {path}")
    frame["overlap"] = pd.to_numeric(
        frame["bottom_k_tail_overlap_fraction"], errors="raise"
    )
    frame["overlap_count"] = pd.to_numeric(
        frame["bottom_k_tail_overlap"], errors="raise"
    ).astype(int)
    frame["chance"] = frame["omission_count"] / frame["count"]
    frame["delta"] = frame["overlap"] - frame["chance"]
    frame["exact"] = (frame["overlap_count"] == frame["omission_count"]).astype(float)
    frame["exact_chance"] = [
        1.0 / math.comb(int(count), int(k))
        for count, k in zip(frame["count"], frame["omission_count"])
    ]
    frame["exact_delta"] = frame["exact"] - frame["exact_chance"]
    frame["tail_prefix_ratio"] = pd.to_numeric(
        frame["undercount_tail_to_prefix_ratio"], errors="coerce"
    )
    return frame


def _span_end_alignment_rows(run_root: Path) -> list[dict[str, Any]]:
    """Variant-level confirmation tail alignment with seed-cluster inference."""
    results: list[dict[str, Any]] = []
    for model in MODELS:
        frame = _span_end_undercount_frame(run_root, model)

        for variant in VARIANTS:
            selected = frame[frame["design_variant"] == variant].copy()
            if selected.empty:
                raise RuntimeError(f"No {model}/{variant} span-end undercounts")
            by_seed = selected.groupby("seed", sort=True)[
                [
                    "overlap",
                    "chance",
                    "delta",
                    "exact",
                    "exact_chance",
                    "exact_delta",
                    "tail_prefix_ratio",
                ]
            ].mean()
            delta_est, delta_low, delta_high = _seed_bootstrap(
                by_seed["delta"].to_numpy(),
                label=f"tail-delta|{model}|{variant}",
            )
            exact_est, exact_low, exact_high = _seed_bootstrap(
                by_seed["exact_delta"].to_numpy(),
                label=f"tail-exact-delta|{model}|{variant}",
            )
            results.append(
                {
                    "model": model,
                    "variant": variant,
                    "prompts": int(len(selected)),
                    "seeds": int(by_seed.shape[0]),
                    "mean_k": float(selected["omission_count"].mean()),
                    "overlap": float(by_seed["overlap"].mean()),
                    "chance": float(by_seed["chance"].mean()),
                    "delta": delta_est,
                    "delta_low": delta_low,
                    "delta_high": delta_high,
                    "p_raw": _exact_sign_flip_p(by_seed["delta"].to_numpy()),
                    "exact": float(by_seed["exact"].mean()),
                    "exact_chance": float(by_seed["exact_chance"].mean()),
                    "exact_delta": exact_est,
                    "exact_delta_low": exact_low,
                    "exact_delta_high": exact_high,
                    "tail_prefix_ratio": float(by_seed["tail_prefix_ratio"].mean()),
                }
            )

    adjusted = _holm_adjust([float(row["p_raw"]) for row in results])
    for row, value in zip(results, adjusted):
        row["p_holm"] = value
    return results


def _span_end_pooled_rows(run_root: Path) -> list[dict[str, Any]]:
    """Equal-variant-weight model-level summary; every seed contributes once."""
    results: list[dict[str, Any]] = []
    metrics = [
        "overlap",
        "chance",
        "delta",
        "exact",
        "exact_chance",
        "exact_delta",
        "tail_prefix_ratio",
    ]
    for model in MODELS:
        frame = _span_end_undercount_frame(run_root, model)
        seed_variant = frame.groupby(["seed", "design_variant"], sort=True)[
            metrics
        ].mean()
        variant_counts = (
            seed_variant.reset_index().groupby("seed")["design_variant"].nunique()
        )
        if not (variant_counts == len(VARIANTS)).all():
            raise RuntimeError(f"Incomplete pooled span-end variants for {model}")
        by_seed = seed_variant.groupby("seed", sort=True)[metrics].mean()
        delta_est, delta_low, delta_high = _seed_bootstrap(
            by_seed["delta"].to_numpy(),
            label=f"tail-pooled-delta|{model}",
        )
        exact_est, exact_low, exact_high = _seed_bootstrap(
            by_seed["exact_delta"].to_numpy(),
            label=f"tail-pooled-exact-delta|{model}",
        )
        results.append(
            {
                "model": model,
                "seeds": int(len(by_seed)),
                "overlap": float(by_seed["overlap"].mean()),
                "chance": float(by_seed["chance"].mean()),
                "delta": delta_est,
                "delta_low": delta_low,
                "delta_high": delta_high,
                "p_raw": _exact_sign_flip_p(by_seed["delta"].to_numpy()),
                "exact": float(by_seed["exact"].mean()),
                "exact_chance": float(by_seed["exact_chance"].mean()),
                "exact_delta": exact_est,
                "exact_delta_low": exact_low,
                "exact_delta_high": exact_high,
                "tail_prefix_ratio": float(by_seed["tail_prefix_ratio"].mean()),
            }
        )
    adjusted = _holm_adjust([float(row["p_raw"]) for row in results])
    for row, value in zip(results, adjusted):
        row["p_holm"] = value
    return results


def _attention_omission_pooling_sensitivity_rows(
    run_root: Path,
) -> list[dict[str, Any]]:
    """Pooled end-versus-sum omission diagnostics on identical undercounts."""

    results: list[dict[str, Any]] = []
    metrics = [
        "overlap",
        "chance",
        "delta",
        "exact",
        "exact_chance",
        "exact_delta",
        "tail_prefix_ratio",
    ]
    for model in MODELS:
        for pooling in ("span_end", "span_sum"):
            frame = _span_end_undercount_frame(
                run_root, model, pooling=pooling
            )
            seed_variant = frame.groupby(
                ["seed", "design_variant"], sort=True
            )[metrics].mean()
            by_seed = seed_variant.groupby("seed", sort=True)[metrics].mean()
            delta, low, high = _seed_bootstrap(
                by_seed["delta"].to_numpy(),
                label=f"omission-pooling|{model}|{pooling}",
            )
            results.append(
                {
                    "model": model,
                    "pooling": pooling,
                    "prompts": int(len(frame)),
                    "seeds": int(len(by_seed)),
                    "overlap": float(by_seed["overlap"].mean()),
                    "chance": float(by_seed["chance"].mean()),
                    "delta": delta,
                    "delta_low": low,
                    "delta_high": high,
                    "exact": float(by_seed["exact"].mean()),
                    "exact_chance": float(by_seed["exact_chance"].mean()),
                    "tail_prefix_ratio": float(
                        by_seed["tail_prefix_ratio"].mean()
                    ),
                }
            )
    return results


def _span_end_nested_rows(run_root: Path) -> list[dict[str, Any]]:
    """Exact new-needle diagnostic on undercount-ending nested transitions."""
    results: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            _attention_analysis_root(run_root, model)
            / "tables"
            / "nested_increment_diagnostics.csv"
        )
        frame = pd.read_csv(path)
        frame = frame[
            (frame["split"] == "confirmation")
            & (frame["pooling"] == "span_end")
            & (pd.to_numeric(frame["count"], errors="coerce") >= 2)
            & (pd.to_numeric(frame["omission_count"], errors="coerce") > 0)
            & frame["increment_status"].isin(
                ["failed_to_increment", "registered_plus_one"]
            )
        ].copy()
        frame["omission_count"] = pd.to_numeric(
            frame["omission_count"], errors="raise"
        ).astype(int)
        frame["new_needle_low_attention_rank"] = pd.to_numeric(
            frame["new_needle_low_attention_rank"], errors="raise"
        ).astype(int)
        frame["new_needle_normalized_share"] = pd.to_numeric(
            frame["new_needle_normalized_share"], errors="raise"
        )
        frame["new_in_bottom_k"] = (
            frame["new_needle_low_attention_rank"] <= frame["omission_count"]
        ).astype(float)
        status_names = {
            "failed_to_increment": "failed",
            "registered_plus_one": "registered",
        }
        frame["status"] = frame["increment_status"].map(status_names)

        block_status = frame.groupby(["seed", "design_variant", "status"], sort=True)[
            ["new_in_bottom_k", "new_needle_normalized_share"]
        ].mean()
        wide_bottom = block_status["new_in_bottom_k"].unstack("status").dropna()
        wide_share = (
            block_status["new_needle_normalized_share"].unstack("status").dropna()
        )
        if not {"failed", "registered"}.issubset(wide_bottom.columns):
            raise RuntimeError(f"Missing paired nested statuses in {path}")
        common_blocks = wide_bottom.index.intersection(wide_share.index)
        wide_bottom = wide_bottom.loc[common_blocks]
        wide_share = wide_share.loc[common_blocks]
        paired_mask = frame.set_index(["seed", "design_variant"]).index.isin(
            common_blocks
        )
        paired_frame = frame.loc[paired_mask]
        seed_bottom = wide_bottom.groupby(level="seed").mean()
        seed_share = wide_share.groupby(level="seed").mean()
        bottom_difference = (
            (wide_bottom["failed"] - wide_bottom["registered"])
            .groupby(level="seed")
            .mean()
            .to_numpy()
        )
        share_difference = (
            (wide_share["registered"] - wide_share["failed"])
            .groupby(level="seed")
            .mean()
            .to_numpy()
        )
        bottom_est, bottom_low, bottom_high = _seed_bootstrap(
            bottom_difference,
            label=f"nested-bottom-difference|{model}",
        )
        share_est, share_low, share_high = _seed_bootstrap(
            share_difference,
            label=f"nested-share-difference|{model}",
        )
        failed_rate, failed_low, failed_high = _seed_bootstrap(
            seed_bottom["failed"].to_numpy(),
            label=f"nested-failed-rate|{model}",
        )
        registered_rate, registered_low, registered_high = _seed_bootstrap(
            seed_bottom["registered"].to_numpy(),
            label=f"nested-registered-rate|{model}",
        )
        results.append(
            {
                "model": model,
                "paired_seeds": int(seed_bottom.shape[0]),
                "paired_blocks": int(len(common_blocks)),
                "failed_n": int((paired_frame["status"] == "failed").sum()),
                "registered_n": int((paired_frame["status"] == "registered").sum()),
                "failed_bottom": failed_rate,
                "failed_bottom_low": failed_low,
                "failed_bottom_high": failed_high,
                "registered_bottom": registered_rate,
                "registered_bottom_low": registered_low,
                "registered_bottom_high": registered_high,
                "bottom_difference": bottom_est,
                "bottom_difference_low": bottom_low,
                "bottom_difference_high": bottom_high,
                "bottom_p_raw": _exact_sign_flip_p(bottom_difference),
                "failed_share": float(seed_share["failed"].mean()),
                "registered_share": float(seed_share["registered"].mean()),
                "share_difference": share_est,
                "share_difference_low": share_low,
                "share_difference_high": share_high,
                "share_p_raw": _exact_sign_flip_p(share_difference),
            }
        )

    for field in ("bottom", "share"):
        adjusted = _holm_adjust([float(row[f"{field}_p_raw"]) for row in results])
        for row, value in zip(results, adjusted):
            row[f"{field}_p_holm"] = value
    return results


def _table_metric_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model_label']))}</td>"
            f"<td><code>{html.escape(str(row['pooling']))}</code></td>"
            f"<td>L{int(row['layer'])}</td>"
            f"<td>{html.escape(str(row['design_variant']))}</td>"
            f"<td>{_number(row['confirmation_r2'])}</td>"
            f"<td>{_number(row['confirmation_mae'])}</td>"
            f"<td>{_number(row['noise_to_signal_ratio'])}</td>"
            f"<td>{_number(row['discovery_confirmation_linear_cka'])}</td>"
            f"<td>{_number(row['discovery_confirmation_distance_correlation'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _representation_r2_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1080, 420
    panel_lefts = (84, 604)
    plot_width, top, bottom = 390, 72, 322
    y_min, y_max = -0.25, 1.05

    def x_position(variant: str, left: float) -> float:
        return left + VARIANTS.index(variant) / (len(VARIANTS) - 1) * plot_width

    def y_position(value: float) -> float:
        bounded = max(y_min, min(y_max, float(value)))
        return bottom - (bounded - y_min) / (y_max - y_min) * (bottom - top)

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="representation-r2-title representation-r2-desc">',
        '<title id="representation-r2-title">Held-out ridge count decoding across the V4 relaxation ladder</title>',
        '<desc id="representation-r2-desc">Span-end decoding remains positive through v4.4, while span-mean decoding collapses after city-score order is released, especially in Gemma.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for index, pooling in enumerate(POOLINGS):
        x = 392 + index * 170
        color = POOLING_COLORS[pooling]
        dash = "" if pooling == "span_end" else ' stroke-dasharray="7 5"'
        parts.extend(
            [
                f'<line x1="{x}" y1="28" x2="{x+28}" y2="28" stroke="{color}" stroke-width="4"{dash}/>',
                f'<circle cx="{x+14}" cy="28" r="4" fill="{color}"/>',
                f'<text x="{x+36}" y="32" font-size="12">{pooling.replace("_", "-")}</text>',
            ]
        )
    for panel_index, model in enumerate(MODELS):
        left = panel_lefts[panel_index]
        parts.append(
            f'<text x="{left}" y="54" font-size="17" font-weight="700">{html.escape(model)}</text>'
        )
        for tick in (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
            y = y_position(tick)
            line_color = AURORA["warm_brown"] if tick == 0 else AURORA["frost_gray"]
            line_width = 1.5 if tick == 0 else 1
            opacity = 0.65 if tick == 0 else 0.28
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" '
                    f'stroke="{line_color}" stroke-width="{line_width}" opacity="{opacity}"/>',
                    f'<text x="{left-11}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                    f'fill="{AURORA["frost_gray"]}">{tick:.2f}</text>',
                ]
            )
            if tick == 0:
                parts.append(
                    f'<text x="{left+plot_width-4}" y="{y-6:.1f}" text-anchor="end" '
                    f'font-size="10" fill="{AURORA["warm_brown"]}">R²=0</text>'
                )
        for variant in VARIANTS:
            x = x_position(variant, left)
            parts.append(
                f'<text x="{x:.1f}" y="344" text-anchor="middle" font-size="11" '
                f'fill="{AURORA["frost_gray"]}">{variant}</text>'
            )
        for pooling in POOLINGS:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model_label"] == model and row["pooling"] == pooling
                ],
                key=lambda row: VARIANTS.index(str(row["design_variant"])),
            )
            points = [
                (
                    x_position(str(row["design_variant"]), left),
                    y_position(float(row["confirmation_r2"])),
                )
                for row in selected
            ]
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
                for index, (x, y) in enumerate(points)
            )
            color = POOLING_COLORS[pooling]
            dash = "" if pooling == "span_end" else ' stroke-dasharray="7 5"'
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"{dash}/>'
            )
            for x, y in points:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" '
                    f'stroke="{AURORA["snow_white"]}" stroke-width="1.5"/>'
                )
        parts.extend(
            [
                f'<text x="{left + plot_width/2:.1f}" y="388" text-anchor="middle" font-size="12">controlled relaxation panel</text>',
                f'<text transform="translate({left-58} {(top+bottom)/2:.1f}) rotate(-90)" '
                'text-anchor="middle" font-size="12">confirmation R²</text>',
            ]
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _layer_row(
    rows: list[dict[str, Any]],
    *,
    model: str,
    pooling: str,
    selection: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["model"] == model
        and row["pooling"] == pooling
        and bool(row[selection])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {selection} row for {model}/{pooling}")
    return matches[0]


def _table_layer_selection_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for model in MODELS:
        for pooling in POOLINGS:
            probe = _layer_row(
                rows,
                model=model,
                pooling=pooling,
                selection="probe_optimal",
            )
            display = _layer_row(
                rows,
                model=model,
                pooling=pooling,
                selection="manifold_display",
            )
            rendered.append(
                "<tr>"
                f"<td>{html.escape(model)}</td>"
                f"<td><code>{html.escape(pooling)}</code></td>"
                f"<td>L{int(probe['layer'])}</td>"
                f"<td>{_number(probe['full_space_discovery_cv_r2'])}</td>"
                f"<td>{_number(probe['pca_evr_pc1_3'])}</td>"
                f"<td>{_number(probe['count_signal_capture_pc1_3'])}</td>"
                f"<td>{_number(probe['discovery_loo_noise_to_signal'])}</td>"
                f"<td>L{int(display['layer'])}</td>"
                f"<td>{_number(display['full_space_discovery_cv_r2'])}</td>"
                f"<td>{_number(display['pca_evr_pc1_3'])}</td>"
                f"<td>{_number(display['count_signal_capture_pc1_3'])}</td>"
                f"<td>{_number(display['pca3_discovery_cv_r2'])}</td>"
                f"<td>{_number(display['discovery_loo_noise_to_signal'])}</td>"
                f"<td>{_number(display['manifold_fidelity_m3'])}</td>"
                "</tr>"
            )
    return "".join(rendered)


def _layer_sweep_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1180, 850
    panel_width, panel_height = 420, 235
    positions = ((105, 155), (660, 155), (105, 520), (660, 520))
    metrics = (
        ("full_space_discovery_cv_r2", "full-space CV R²", AURORA["polar_violet"]),
        ("pca_evr_pc1_3", "PC1–3 total EVR", AURORA["ice_cyan"]),
        (
            "count_signal_capture_pc1_3",
            "PC1–3 count-signal F₃",
            AURORA["aurora_green"],
        ),
        ("discovery_compactness", "seed compactness C", AURORA["sunset_pink"]),
    )
    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="layer-sweep-title layer-sweep-desc">',
        '<title id="layer-sweep-title">Discovery-only decoder-layer sweep for PCA manifold selection</title>',
        '<desc id="layer-sweep-desc">Purple is full-space grouped-seed count-probe CV R squared. Cyan is the fraction of total variance explained by PC1 through PC3. Green is the fraction of between-count centroid signal retained by those PCs. Pink is seed compactness C equals one over one plus the leave-one-seed-out noise-to-signal ratio. Dashed brown marks the probe-optimal layer P; solid indigo marks the manifold-display layer M. Higher is better for all four curves.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    legend_x = 72
    for _key, label, color in metrics:
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="32" x2="{legend_x+30}" y2="32" stroke="{color}" stroke-width="4"/>',
                f'<text x="{legend_x+38}" y="36" font-size="11">{html.escape(label)}</text>',
            ]
        )
        legend_x += 275
    parts.extend(
        [
            f'<line x1="72" y1="70" x2="102" y2="70" stroke="{AURORA["warm_brown"]}" stroke-width="2" stroke-dasharray="6 4"/>',
            '<text x="110" y="74" font-size="11">P · probe-optimal: maximum full-space CV R²</text>',
            f'<line x1="392" y1="70" x2="422" y2="70" stroke="{AURORA["midnight_indigo"]}" stroke-width="3"/>',
            '<text x="430" y="74" font-size="11">M · manifold-display: maximum EVR₃ × F₃ × C inside the R² gate</text>',
        ]
    )
    panel_index = 0
    for model in MODELS:
        for pooling in POOLINGS:
            left, top = positions[panel_index]
            panel_index += 1
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model"] == model and row["pooling"] == pooling
                ],
                key=lambda row: int(row["layer"]),
            )
            maximum_layer = max(int(row["layer"]) for row in selected)

            def x_position(layer: int) -> float:
                return left + int(layer) / max(maximum_layer, 1) * panel_width

            def y_position(value: float) -> float:
                return top + panel_height - max(0.0, min(1.0, float(value))) * panel_height

            parts.append(
                f'<text x="{left}" y="{top-45}" font-size="15" font-weight="700">'
                f'{html.escape(model)} · {html.escape(pooling.replace("_", "-"))}</text>'
            )
            for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
                y = y_position(tick)
                parts.extend(
                    [
                        f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_width}" y2="{y:.1f}" stroke="{AURORA["frost_gray"]}" opacity=".25"/>',
                        f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="{AURORA["frost_gray"]}">{tick:.2f}</text>',
                    ]
                )
            for layer_tick in sorted({0, maximum_layer // 2, maximum_layer}):
                x = x_position(layer_tick)
                parts.extend(
                    [
                        f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+panel_height}" stroke="{AURORA["frost_gray"]}" opacity=".16"/>',
                        f'<text x="{x:.1f}" y="{top+panel_height+20}" text-anchor="middle" font-size="10" fill="{AURORA["frost_gray"]}">L{layer_tick}</text>',
                    ]
                )
            probe = _layer_row(
                rows,
                model=model,
                pooling=pooling,
                selection="probe_optimal",
            )
            display = _layer_row(
                rows,
                model=model,
                pooling=pooling,
                selection="manifold_display",
            )
            parts.extend(
                [
                    f'<text x="{left}" y="{top-22}" font-size="10" font-weight="700" fill="{AURORA["warm_brown"]}">P: L{int(probe["layer"])}</text>',
                    f'<text x="{left+62}" y="{top-22}" font-size="10" font-weight="700" fill="{AURORA["midnight_indigo"]}">M: L{int(display["layer"])}</text>',
                ]
            )
            for layer, color, dash, label in (
                (
                    int(probe["layer"]),
                    AURORA["warm_brown"],
                    ' stroke-dasharray="6 4"',
                    "P",
                ),
                (
                    int(display["layer"]),
                    AURORA["midnight_indigo"],
                    "",
                    "M",
                ),
            ):
                x = x_position(layer)
                parts.extend(
                    [
                        f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+panel_height}" stroke="{color}" stroke-width="{3 if label == "M" else 2}"{dash}/>',
                    ]
                )
            for key, _label, color in metrics:
                points = [
                    (x_position(int(row["layer"])), y_position(float(row[key])))
                    for row in selected
                ]
                path = " ".join(
                    ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
                    for index, (x, y) in enumerate(points)
                )
                parts.append(
                    f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" opacity=".9"/>'
                )
            parts.extend(
                [
                    f'<text x="{left+panel_width/2:.1f}" y="{top+panel_height+42}" text-anchor="middle" font-size="11">post-block decoder layer index</text>',
                    f'<text transform="translate({left-50} {top+panel_height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="11">discovery-only score</text>',
                ]
            )
    parts.append("</g></svg>")
    return "".join(parts)


def _layer_selection_conclusion_html(rows: list[dict[str, Any]]) -> str:
    statements: list[str] = []
    for model in MODELS:
        selections: list[str] = []
        for pooling in POOLINGS:
            probe = _layer_row(
                rows,
                model=model,
                pooling=pooling,
                selection="probe_optimal",
            )
            display = _layer_row(
                rows,
                model=model,
                pooling=pooling,
                selection="manifold_display",
            )
            selections.append(
                f"{pooling.replace('_', '-')} L{int(probe['layer'])}→L{int(display['layer'])}"
            )
        statements.append(f"{html.escape(model)}：" + "；".join(selections))
    return (
        '<div class="section-conclusion"><span>本节结论 · 层选择已拆分</span><p>'
        + "。".join(statements)
        + "。旧层只回答‘full residual space 中哪层最容易线性解码’，不能保证前三个 PC 能完整展示 manifold。新的 3D/PC1–PC2 图使用 discovery-only manifold-display 层；原 probe-optimal 层仍保留用于预注册的 held-out Ridge 结果，两者不再混称为同一个 primary layer。这个 post-hoc display 规则没有查看 confirmation 标签，因此不会把测试集外观反向用于选层。</p></div>"
    )


def _answer_query_counter_svg(
    projections: dict[str, dict[str, Any]],
) -> str:
    """Six-panel landmark audit; the interactive view retains every layer."""

    width, height = 1120, 810
    panel_width, panel_height = 270, 225
    lefts = (95, 430, 765)
    tops = (135, 485)
    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="answer-query-counter-title answer-query-counter-desc">',
        '<title id="answer-query-counter-title">Answer-query count manifolds at three decoder-layer landmarks</title>',
        '<desc id="answer-query-counter-desc">The interactive figure contains every captured post-block answer-query layer. This printable audit chooses three landmarks per model: the first layer, the discovery-selected manifold-display layer when distinct, and the last layer. PCA is fit to all V4.1 discovery prompts. Point and node color encodes gold count from one, indigo, to ten, cyan. Point outline encodes the final greedy outcome: white correct, dark wrong, and pink invalid. Dashed gray paths connect v4.1 count centroids and solid black paths connect v4.4 centroids.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
        f'<line x1="675" y1="30" x2="710" y2="30" stroke="{AURORA["frost_gray"]}" stroke-width="2" stroke-dasharray="6 4"/>',
        '<text x="718" y="34" font-size="11">v4.1 centroid path</text>',
        f'<line x1="895" y1="30" x2="930" y2="30" stroke="{AURORA["night_black"]}" stroke-width="3"/>',
        '<text x="938" y="34" font-size="11">v4.4 centroid path</text>',
    ]
    for count in range(1, 11):
        x = 180 + (count - 1) * 76
        parts.extend(
            [
                f'<circle cx="{x}" cy="74" r="5" fill="{COUNT_COLORS[count-1]}" stroke="{AURORA["night_black"]}" stroke-width=".5"/>',
                f'<text x="{x+10}" y="78" font-size="10">N={count}</text>',
            ]
        )
    parts.append(
        f'<text x="115" y="78" text-anchor="end" font-size="10" fill="{AURORA["frost_gray"]}">gold count</text>'
    )
    for model_index, model in enumerate(MODELS):
        model_layers = sorted(
            int(key.split("|")[1])
            for key in projections
            if key.startswith(model + "|") and key.endswith("|all")
        )
        if len(model_layers) < 3:
            raise RuntimeError(f"{model}: expected at least three answer-query layers")
        display_layers = [
            int(data["layer"])
            for key, data in projections.items()
            if key.startswith(model + "|")
            and key.endswith("|all")
            and bool(data.get("manifold_display"))
        ]
        preferred = [model_layers[0], *display_layers, model_layers[-1]]
        landmarks: list[int] = []
        for layer in preferred:
            if int(layer) not in landmarks:
                landmarks.append(int(layer))
        if len(landmarks) < 3:
            middle = model_layers[len(model_layers) // 2]
            if middle not in landmarks:
                landmarks.insert(1, middle)
        if len(landmarks) < 3:
            for layer in model_layers:
                if layer not in landmarks:
                    landmarks.insert(-1, layer)
                if len(landmarks) == 3:
                    break
        model_layers = sorted(landmarks[:3])
        top = tops[model_index]
        parts.append(
            f'<text x="30" y="{top+panel_height/2:.1f}" transform="rotate(-90 30 {top+panel_height/2:.1f})" text-anchor="middle" font-size="16" font-weight="700">{html.escape(model)}</text>'
        )
        for column, layer in enumerate(model_layers):
            left = lefts[column]
            data = projections[f"{model}|{layer}|all"]
            rows = [row for row in data["rows"] if row[0] in {"v4.1", "v4.4"}]
            x_values = np.asarray([float(row[6]) for row in rows])
            y_values = np.asarray([float(row[7]) for row in rows])
            x_low, x_high = np.quantile(x_values, [0.005, 0.995])
            y_low, y_high = np.quantile(y_values, [0.005, 0.995])
            x_margin = max(1e-8, float(x_high - x_low) * 0.08)
            y_margin = max(1e-8, float(y_high - y_low) * 0.08)
            x_low, x_high = float(x_low - x_margin), float(x_high + x_margin)
            y_low, y_high = float(y_low - y_margin), float(y_high + y_margin)

            def project(row: list[Any]) -> tuple[float, float]:
                x = left + (float(row[6]) - x_low) / (x_high - x_low) * panel_width
                y = top + panel_height - (
                    (float(row[7]) - y_low) / (y_high - y_low) * panel_height
                )
                return x, y

            evr = data["explained_variance_ratio"]
            parts.extend(
                [
                    f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="{AURORA["snow_white"]}" stroke="{AURORA["frost_gray"]}" stroke-opacity=".42"/>',
                    f'<text x="{left}" y="{top-18}" font-size="14" font-weight="700">L{layer}</text>',
                    f'<text x="{left+panel_width}" y="{top-18}" text-anchor="end" font-size="10" fill="{AURORA["frost_gray"]}">PC1+PC2 EVR {100*(float(evr[0])+float(evr[1])):.1f}%</text>',
                ]
            )
            for fraction in (0.25, 0.5, 0.75):
                x = left + fraction * panel_width
                y = top + fraction * panel_height
                parts.extend(
                    [
                        f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+panel_height}" stroke="{AURORA["frost_gray"]}" opacity=".14"/>',
                        f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_width}" y2="{y:.1f}" stroke="{AURORA["frost_gray"]}" opacity=".14"/>',
                    ]
                )
            for row in rows:
                if row[0] != "v4.4":
                    continue
                x, y = project(row)
                stroke = {
                    "correct": AURORA["snow_white"],
                    "wrong": AURORA["night_black"],
                    "invalid": AURORA["sunset_pink"],
                }[str(row[3])]
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.1" fill="{COUNT_COLORS[int(row[2])-1]}" stroke="{stroke}" stroke-width=".7" opacity=".38"/>'
                )
            for variant, color, dash, opacity in (
                ("v4.1", AURORA["frost_gray"], ' stroke-dasharray="6 4"', 0.8),
                ("v4.4", AURORA["night_black"], "", 1.0),
            ):
                selected = [row for row in rows if row[0] == variant]
                centroids: list[list[Any]] = []
                for count in range(1, 11):
                    group = [row for row in selected if int(row[2]) == count]
                    if not group:
                        raise RuntimeError(
                            f"{model}/L{layer}/{variant}: incomplete count grid"
                        )
                    centroid = group[0].copy()
                    centroid[6] = float(np.mean([float(row[6]) for row in group]))
                    centroid[7] = float(np.mean([float(row[7]) for row in group]))
                    centroids.append(centroid)
                points = [project(row) for row in centroids]
                path = " ".join(
                    ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
                    for index, (x, y) in enumerate(points)
                )
                parts.append(
                    f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{2.6 if variant == "v4.4" else 1.8}" opacity="{opacity}"{dash}/>'
                )
                if variant == "v4.4":
                    for count, (x, y) in enumerate(points, start=1):
                        parts.append(
                            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.3" fill="{COUNT_COLORS[count-1]}" stroke="{AURORA["night_black"]}" stroke-width=".8"/>'
                        )
                    endpoint_labels = (
                        (1, points[0][0] - 8, points[0][1] - 10, "end"),
                        (10, points[-1][0] + 8, points[-1][1] + 14, "start"),
                    )
                    for count, label_x, label_y, anchor in endpoint_labels:
                        label_x = max(left + 6, min(left + panel_width - 6, label_x))
                        label_y = max(top + 12, min(top + panel_height - 5, label_y))
                        parts.append(
                            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" font-size="9" font-weight="700" paint-order="stroke" stroke="{AURORA["snow_white"]}" stroke-width="3" stroke-linejoin="round">N={count}</text>'
                        )
            parts.extend(
                [
                    f'<text x="{left+panel_width/2:.1f}" y="{top+panel_height+24}" text-anchor="middle" font-size="10">PC1 score</text>',
                    f'<text transform="translate({left-34} {top+panel_height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="10">PC2 score</text>',
                ]
            )
    parts.append("</g></svg>")
    return "".join(parts)


def _answer_query_layer_sweep_svg(
    projections: dict[str, dict[str, Any]],
) -> str:
    """Discovery-only all-layer diagnostics for the answer-query state."""

    width, height = 1120, 470
    panel_width, panel_height = 430, 250
    lefts = (105, 635)
    top = 125
    line_specs = (
        ("pca3_discovery_cv_r2", AURORA["polar_violet"], "PCA3 grouped-seed CV R²"),
        ("evr3", AURORA["ice_cyan"], "PC1–3 total EVR"),
        (
            "count_signal_capture_pc1_3",
            AURORA["aurora_green"],
            "PC1–3 count-signal capture",
        ),
        (
            "discovery_compactness",
            AURORA["sunset_pink"],
            "seed compactness",
        ),
    )
    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="aq-layer-sweep-title aq-layer-sweep-desc">',
        '<title id="aq-layer-sweep-title">All-layer answer-query manifold diagnostics</title>',
        '<desc id="aq-layer-sweep-desc">Two panels show Qwen and Gemma. The horizontal axis is zero-based post-block decoder layer. Four discovery-only curves report PCA-three grouped-seed count-probe R squared, cumulative explained variance of the first three principal components, the fraction of count-centroid signal retained by those components, and seed compactness. P and M vertical markers identify the PCA-three probe and manifold-display layers.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for legend_index, (_metric, color, label) in enumerate(line_specs):
        x = 150 + (legend_index % 2) * 430
        y = 35 + (legend_index // 2) * 32
        parts.extend(
            [
                f'<line x1="{x}" y1="{y}" x2="{x+35}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x+45}" y="{y+4}" font-size="11">{html.escape(label)}</text>',
            ]
        )
    for model_index, model in enumerate(MODELS):
        left = lefts[model_index]
        data_rows = sorted(
            [
                data
                for key, data in projections.items()
                if key.startswith(model + "|") and key.endswith("|all")
            ],
            key=lambda item: int(item["layer"]),
        )
        layers = [int(item["layer"]) for item in data_rows]
        if len(layers) < 3:
            raise RuntimeError(f"{model}: insufficient answer-query layer sweep")
        values_by_metric = {
            "pca3_discovery_cv_r2": [
                float(item["pca3_discovery_cv_r2"]) for item in data_rows
            ],
            "evr3": [
                float(np.sum(item["explained_variance_ratio"][:3]))
                for item in data_rows
            ],
            "count_signal_capture_pc1_3": [
                float(item["count_signal_capture_pc1_3"]) for item in data_rows
            ],
            "discovery_compactness": [
                float(item["discovery_compactness"]) for item in data_rows
            ],
        }
        minimum = min(
            -0.05,
            min(values_by_metric["pca3_discovery_cv_r2"]),
        )
        maximum = 1.0

        def project_x(layer: int) -> float:
            return left + (layer - layers[0]) / max(layers[-1] - layers[0], 1) * panel_width

        def project_y(value: float) -> float:
            return top + (maximum - value) / (maximum - minimum) * panel_height

        parts.extend(
            [
                f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="{AURORA["snow_white"]}" stroke="{AURORA["frost_gray"]}" stroke-opacity=".45"/>',
                f'<text x="{left}" y="{top-18}" font-size="15" font-weight="700">{model}</text>',
            ]
        )
        for tick in np.linspace(minimum, maximum, 5):
            y = project_y(float(tick))
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.2f}" x2="{left+panel_width}" y2="{y:.2f}" stroke="{AURORA["frost_gray"]}" opacity=".14"/>',
                    f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="10" fill="{AURORA["frost_gray"]}">{tick:.2f}</text>',
                ]
            )
        for metric, color, _label in line_specs:
            points = [
                (project_x(layer), project_y(value))
                for layer, value in zip(layers, values_by_metric[metric])
            ]
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
                for index, (x, y) in enumerate(points)
            )
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
        for flag, label, color, dash in (
            ("probe_optimal", "P", AURORA["warm_brown"], "5 4"),
            ("manifold_display", "M", AURORA["midnight_indigo"], ""),
        ):
            selected = [item for item in data_rows if bool(item.get(flag))]
            if len(selected) != 1:
                raise RuntimeError(f"{model}: expected one answer-query {flag} layer")
            layer = int(selected[0]["layer"])
            x = project_x(layer)
            label_x = x - 5 if layer >= layers[0] + 0.75 * (layers[-1] - layers[0]) else x + 5
            label_anchor = "end" if label_x < x else "start"
            label_y = top + (15 if flag == "probe_optimal" else 31)
            parts.extend(
                [
                    f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+panel_height}" stroke="{color}" stroke-width="2.2" stroke-dasharray="{dash}" opacity=".85"/>',
                    f'<text x="{label_x:.2f}" y="{label_y}" text-anchor="{label_anchor}" font-size="11" font-weight="700" paint-order="stroke" stroke="{AURORA["snow_white"]}" stroke-width="3" fill="{color}">{label}:L{layer}</text>',
                ]
            )
        parts.extend(
            [
                f'<text x="{left+panel_width/2}" y="{top+panel_height+38}" text-anchor="middle" font-size="11">post-block decoder layer index</text>',
                f'<text x="{left-62}" y="{top+panel_height/2}" transform="rotate(-90 {left-62} {top+panel_height/2})" text-anchor="middle" font-size="11">discovery-only score</text>',
            ]
        )
    parts.extend(["</g>", "</svg>"])
    return "".join(parts)


def _representation_conclusion_html(
    rows: list[dict[str, Any]], sensitivity_rows: list[dict[str, Any]]
) -> str:
    summaries: list[str] = []
    for model in MODELS:
        end = next(
            row
            for row in rows
            if row["model_label"] == model
            and row["pooling"] == "span_end"
            and row["design_variant"] == "v4.4"
        )
        mean = next(
            row
            for row in rows
            if row["model_label"] == model
            and row["pooling"] == "span_mean"
            and row["design_variant"] == "v4.4"
        )
        summaries.append(
            f"{html.escape(model)} 在 v4.4 的 span-end R²={_number(end['confirmation_r2'])}，"
            f"span-mean R²={_number(mean['confirmation_r2'])}"
        )
    first_noise: list[str] = []
    for model in MODELS:
        selected = [
            row
            for row in sensitivity_rows
            if row["model"] == model
            and row["pooling"] == "span_end"
            and row["metric"] == "curve_residual_to_signal"
            and _bool(row["increase_ci_excludes_zero"])
        ]
        if selected:
            row = selected[0]
            first_noise.append(
                f"{html.escape(model)} 在 {html.escape(str(row['left_variant']))}→{html.escape(str(row['right_variant']))}"
            )
    return (
        '<div class="section-conclusion"><span>本节结论</span><p>'
        + "；".join(summaries)
        + "。因此 needle 末端一直保留可解码的 count-related signal，但它不是低噪声、等间距的标量计数器；span-mean 在释放 city-score 顺序后明显崩溃，说明其早期高分主要依赖固定记录结构。span-end 的 seed-noise 首次显著上升分别出现在 "
        + "、".join(first_noise)
        + "。这些证据证明信息可用性，不证明生成必然读取该方向。</p></div>"
    )


def _attention_top_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            _attention_analysis_root(run_root, model)
            / "tables"
            / "discovery_head_summary.csv"
        )
        frame = pd.read_csv(path)
        rank = pd.to_numeric(frame["candidate_rank"], errors="coerce")
        selected = frame[rank == 1].copy()
        expected = len(VARIANTS) * len(ATTENTION_POOLINGS)
        if len(selected) != expected:
            raise RuntimeError(f"Expected {expected} rank-1 rows in {path}")
        for row in selected.to_dict("records"):
            rows.append(
                {
                    "model": model,
                    "variant": str(row["design_variant"]),
                    "pooling": str(row["pooling"]),
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "coverage": float(row["pool_coverage"]),
                    "effective_number": float(row["pool_effective_number"]),
                    "primary": float(row["pool_primary"]),
                    "total_mass": float(row["pool_sum"]),
                    "enrichment": float(row["pool_enrichment"]),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            MODELS.index(str(row["model"])),
            ATTENTION_POOLINGS.index(str(row["pooling"])),
            VARIANTS.index(str(row["variant"])),
        ),
    )


def _attention_head_atlas_rows(run_root: Path) -> list[dict[str, Any]]:
    """Return every discovery full-attention head used by the atlas."""

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            _attention_analysis_root(run_root, model)
            / "tables"
            / "discovery_head_summary.csv"
        )
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            rows.append(
                {
                    "model": model,
                    "variant": str(row["design_variant"]),
                    "pooling": str(row["pooling"]),
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "layer_type": str(row["layer_type"]),
                    "examples": int(row["examples"]),
                    "seeds": int(row["seeds"]),
                    "pool_sum": float(row["pool_sum"]),
                    "pool_coverage": float(row["pool_coverage"]),
                    "pool_primary": float(row["pool_primary"]),
                    "pool_enrichment": float(row["pool_enrichment"]),
                    "pool_effective_number": float(
                        row["pool_effective_number"]
                    ),
                    "positive_contrast": _bool(
                        row["positive_needle_control_contrast"]
                    ),
                    "density_enrichment_gt_one": _bool(
                        row["needle_density_enrichment_gt_one"]
                    ),
                    "is_broad_candidate": _bool(row["is_broad_candidate"]),
                    "candidate_rank": (
                        None
                        if pd.isna(row["candidate_rank"])
                        else int(float(row["candidate_rank"]))
                    ),
                }
            )
    return rows


def _trajectory_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear CKA between two count-centroid trajectories."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    denominator = math.sqrt(
        float(np.sum(left_gram**2)) * float(np.sum(right_gram**2))
    )
    if denominator <= 0:
        return math.nan
    return float(np.sum(left_gram * right_gram) / denominator)


def _trajectory_distance_correlation(
    left: np.ndarray, right: np.ndarray
) -> float:
    """Correlation of all 45 pairwise distances among ten centroids."""

    indices = np.triu_indices(10, k=1)
    left_distances = np.linalg.norm(
        np.asarray(left)[:, None, :] - np.asarray(left)[None, :, :], axis=-1
    )[indices]
    right_distances = np.linalg.norm(
        np.asarray(right)[:, None, :] - np.asarray(right)[None, :, :], axis=-1
    )[indices]
    if np.std(left_distances) <= 0 or np.std(right_distances) <= 0:
        return math.nan
    return float(np.corrcoef(left_distances, right_distances)[0, 1])


def _successive_step_alignment(left: np.ndarray, right: np.ndarray) -> float:
    left_steps = np.diff(np.asarray(left, dtype=np.float64), axis=0)
    right_steps = np.diff(np.asarray(right, dtype=np.float64), axis=0)
    denominators = np.linalg.norm(left_steps, axis=1) * np.linalg.norm(
        right_steps, axis=1
    )
    valid = denominators > 0
    if not np.any(valid):
        return math.nan
    return float(
        np.mean(
            np.sum(left_steps[valid] * right_steps[valid], axis=1)
            / denominators[valid]
        )
    )


def _joint_counter_projection_data(
    run_root: Path,
    answer_query_projections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Place prompt-occurrence and answer-query centroids in one PCA basis.

    Each pair has the same model, panel, discovery seed, and count ``k``.  The
    prompt member is occurrence ``k`` from the N=10 prompt; the answer member
    is the prompt-final ``Total:`` state from the N=k prompt.  ``raw`` PCA uses
    the states directly.  ``role_centered`` subtracts separate V4.1 discovery
    means for the two token roles before fitting the shared PCA.
    """

    result: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        model_root = run_root / model / "numeric"
        answer_root = (
            model_root / "representation" / "answer_query_all_layers_v1"
        )
        answer_index = answer_root / "capture_index.jsonl"
        if not answer_index.exists():
            continue
        answer_records = _read_jsonl(answer_index)
        answer_states: list[np.ndarray] = []
        answer_lookup: dict[tuple[str, int, int], int] = {}
        layer_indices: np.ndarray | None = None
        for record in answer_records:
            key = (
                str(record["design_variant"]),
                int(record["seed"]),
                int(record["count"]),
            )
            if key in answer_lookup:
                raise RuntimeError(f"{model}: duplicate answer-query pair {key}")
            with np.load(
                answer_root / str(record["shard_path"]), allow_pickle=False
            ) as payload:
                current_layers = np.asarray(payload["layer_indices"], dtype=int)
                if layer_indices is None:
                    layer_indices = current_layers
                elif not np.array_equal(layer_indices, current_layers):
                    raise RuntimeError(f"{model}: answer-query layer grid changed")
                answer_states.append(np.asarray(payload["query_states"]))
            answer_lookup[key] = len(answer_states) - 1
        if len(answer_states) != 800 or layer_indices is None:
            raise RuntimeError(
                f"{model}: expected 800 all-layer answer-query states"
            )
        answer_tensor = np.stack(answer_states, axis=0)

        prompt_root = model_root / "representation" / "capture"
        prompt_records = [
            record
            for record in _read_jsonl(prompt_root / "capture_index.jsonl")
            if str(record["split"]) == "discovery"
        ]
        prompt_lookup: dict[tuple[str, int], int] = {}
        prompt_states: dict[str, list[np.ndarray]] = {
            pooling: [] for pooling in POOLINGS
        }
        for record in prompt_records:
            key = (str(record["design_variant"]), int(record["seed"]))
            if key in prompt_lookup:
                raise RuntimeError(f"{model}: duplicate prompt counter pair {key}")
            with np.load(
                prompt_root / str(record["shard_path"]), allow_pickle=False
            ) as payload:
                current_layers = np.asarray(payload["layer_indices"], dtype=int)
                if not np.array_equal(layer_indices, current_layers):
                    raise RuntimeError(
                        f"{model}: prompt/answer post-block layer grids differ"
                    )
                for pooling in POOLINGS:
                    prompt_states[pooling].append(np.asarray(payload[pooling]))
            prompt_lookup[key] = len(prompt_lookup)
        if len(prompt_lookup) != 80:
            raise RuntimeError(f"{model}: expected 80 discovery N=10 prompts")

        pair_keys = sorted(
            answer_lookup,
            key=lambda item: (VARIANTS.index(item[0]), item[1], item[2]),
        )
        if len(pair_keys) != 800:
            raise RuntimeError(f"{model}: incomplete answer-query pairing grid")
        variants = np.asarray([key[0] for key in pair_keys])
        counts = np.asarray([key[2] for key in pair_keys], dtype=int)
        v41 = variants == "v4.1"
        answer_order = np.asarray(
            [answer_lookup[key] for key in pair_keys], dtype=int
        )
        for pooling in POOLINGS:
            prompt_tensor = np.stack(prompt_states[pooling], axis=0)
            prompt_order = np.asarray(
                [prompt_lookup[(key[0], key[1])] for key in pair_keys], dtype=int
            )
            for layer_axis, layer in enumerate(layer_indices):
                prompt_layer = np.stack(
                    [
                        prompt_tensor[prompt_index, int(layer_axis), count - 1]
                        for prompt_index, count in zip(prompt_order, counts)
                    ],
                    axis=0,
                ).astype(np.float32)
                answer_layer = answer_tensor[
                    answer_order, int(layer_axis)
                ].astype(np.float32)
                prompt_mean = prompt_layer[v41].mean(axis=0)
                answer_mean = answer_layer[v41].mean(axis=0)
                prompt_centroids_full = np.stack(
                    [prompt_layer[v41 & (counts == count)].mean(axis=0)
                     for count in range(1, 11)]
                )
                answer_centroids_full = np.stack(
                    [answer_layer[v41 & (counts == count)].mean(axis=0)
                     for count in range(1, 11)]
                )
                prompt_centered = prompt_centroids_full - prompt_mean
                answer_centered = answer_centroids_full - answer_mean
                prompt_scale = float(
                    np.sqrt(np.mean(np.sum(prompt_centered**2, axis=1)))
                )
                answer_scale = float(
                    np.sqrt(np.mean(np.sum(answer_centered**2, axis=1)))
                )
                mean_signal_scale = 0.5 * (prompt_scale + answer_scale)
                metrics = {
                    "trajectory_linear_cka": _trajectory_linear_cka(
                        prompt_centered, answer_centered
                    ),
                    "trajectory_distance_correlation": (
                        _trajectory_distance_correlation(
                            prompt_centered, answer_centered
                        )
                    ),
                    "successive_step_cosine": _successive_step_alignment(
                        prompt_centered, answer_centered
                    ),
                    "prompt_centroid_rms": prompt_scale,
                    "answer_centroid_rms": answer_scale,
                    "answer_to_prompt_scale_ratio": (
                        answer_scale / prompt_scale if prompt_scale > 0 else math.nan
                    ),
                    "role_offset_norm": float(
                        np.linalg.norm(answer_mean - prompt_mean)
                    ),
                    "role_offset_to_count_signal": (
                        float(np.linalg.norm(answer_mean - prompt_mean))
                        / mean_signal_scale
                        if mean_signal_scale > 0
                        else math.nan
                    ),
                }
                for mode in ("raw", "role_centered"):
                    if mode == "raw":
                        prompt_input = prompt_layer
                        answer_input = answer_layer
                    else:
                        prompt_input = prompt_layer - prompt_mean
                        answer_input = answer_layer - answer_mean
                    fit = np.concatenate(
                        [prompt_input[v41], answer_input[v41]], axis=0
                    )
                    pca = PCA(
                        n_components=6,
                        svd_solver="randomized",
                        random_state=0,
                    )
                    pca.fit(fit)
                    prompt_projected = pca.transform(prompt_input)
                    answer_projected = pca.transform(answer_input)
                    rows: list[list[Any]] = []
                    for variant in VARIANTS:
                        for role, projected in (
                            ("prompt_occurrence", prompt_projected),
                            ("answer_query", answer_projected),
                        ):
                            for count in range(1, 11):
                                mask = (variants == variant) & (counts == count)
                                centroid = projected[mask].mean(axis=0)
                                rows.append(
                                    [
                                        variant,
                                        role,
                                        int(count),
                                        int(mask.sum()),
                                        *[
                                            round(float(value), 6)
                                            for value in centroid
                                        ],
                                    ]
                                )
                    answer_key = f"{model}|{int(layer)}|all"
                    result[f"{model}|{pooling}|{int(layer)}|{mode}"] = {
                        "model": model,
                        "pooling": pooling,
                        "layer": int(layer),
                        "mode": mode,
                        "fit_variant": "v4.1",
                        "fit_split": "discovery",
                        "fit_rows": int(len(fit)),
                        "manifold_display": bool(
                            answer_query_projections.get(answer_key, {}).get(
                                "manifold_display", False
                            )
                        ),
                        "explained_variance_ratio": [
                            round(float(value), 8)
                            for value in pca.explained_variance_ratio_
                        ],
                        **{
                            key: round(float(value), 8)
                            for key, value in metrics.items()
                        },
                        "rows": rows,
                    }
            del prompt_tensor
        del answer_tensor
    return result


def _prompt_counter_dynamics_frames(run_root: Path) -> dict[str, pd.DataFrame]:
    root = run_root / "analysis" / "prompt_counter_dynamics_v1"
    paths = {
        "banks": root / "selected_head_bank.csv",
        "samples": root / "attention_bank_by_sample.csv.gz",
        "noise": root / "hidden_counter_noise_by_sample.csv.gz",
        "slopes": root / "occurrence_slope_summary.csv",
        "associations": root / "attention_noise_association.csv",
        "all_head_samples": root / "all_head_attention_by_sample.csv.gz",
        "all_head_slopes": root / "all_head_occurrence_slope_summary.csv",
        "all_head_associations": root
        / "all_head_attention_noise_association.csv",
        "profiles": root / "profile_maps.csv.gz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Prompt-counter dynamics analysis is incomplete: " + ", ".join(missing)
        )
    return {key: pd.read_csv(path) for key, path in paths.items()}


def _prompt_counter_profile_data(
    frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    profiles = frames["profiles"]
    samples = frames["samples"]
    noise = frames["noise"]
    result: dict[str, dict[str, Any]] = {}
    keys = ["model", "hidden_pooling", "layer", "design_variant"]
    curve = samples[samples["split"] == "confirmation"].merge(
        noise,
        on=[
            "model",
            "design_variant",
            "seed",
            "split",
            "hidden_pooling",
            "layer",
            "query_occurrence",
        ],
        how="inner",
        validate="one_to_one",
    )
    curve_summary = (
        curve.groupby([*keys, "query_occurrence"], as_index=False)[
            [
                "row_effective_fraction",
                "row_effective_tokens",
                "needle_effective_number",
                "needle_relative_coverage",
                "counter_noise",
            ]
        ]
        .mean()
        .sort_values([*keys, "query_occurrence"])
    )
    curve_lookup = {
        tuple(group_key): group
        for group_key, group in curve_summary.groupby(keys, sort=False)
    }
    for group_key, group in profiles.groupby(keys, sort=True):
        matrix_mass = np.zeros((10, 10), dtype=float)
        matrix_share = np.zeros((10, 10), dtype=float)
        for row in group.to_dict("records"):
            query = int(row["query_occurrence"]) - 1
            key = int(row["key_occurrence"]) - 1
            matrix_mass[query, key] = float(row["mean_attention_mass"])
            matrix_share[query, key] = float(row["within_needle_share"])
        curves = curve_lookup[tuple(group_key)]
        result["|".join(str(value) for value in group_key)] = {
            "model": str(group_key[0]),
            "pooling": str(group_key[1]),
            "layer": int(group_key[2]),
            "variant": str(group_key[3]),
            "key_pooling": str(group["key_pooling"].iloc[0]),
            "confirmation_seed_count": int(
                group["confirmation_seed_count"].iloc[0]
            ),
            "mean_attention_mass": [
                [round(float(value), 8) for value in row]
                for row in matrix_mass
            ],
            "within_needle_share": [
                [round(float(value), 8) for value in row]
                for row in matrix_share
            ],
            "curves": {
                metric: [round(float(value), 8) for value in curves[metric]]
                for metric in (
                    "row_effective_fraction",
                    "row_effective_tokens",
                    "needle_effective_number",
                    "needle_relative_coverage",
                    "counter_noise",
                )
            },
        }
    return result


def _prompt_counter_selected_rows_for_scope(
    slopes: pd.DataFrame,
    associations: pd.DataFrame,
    layer_sweep_rows: list[dict[str, Any]],
    *,
    head_scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for pooling in POOLINGS:
            layer_rows = [
                row
                for row in layer_sweep_rows
                if row["model"] == model and row["pooling"] == pooling
            ]
            selections = {
                "probe_optimal": int(
                    next(row["layer"] for row in layer_rows if row["probe_optimal"])
                ),
                "manifold_display": int(
                    next(
                        row["layer"]
                        for row in layer_rows
                        if row["manifold_display"]
                    )
                ),
            }
            for selection, layer in selections.items():
                for variant in VARIANTS:
                    shared = (
                        (slopes["model"] == model)
                        & (slopes["hidden_pooling"] == pooling)
                        & (pd.to_numeric(slopes["layer"]).astype(int) == layer)
                        & (slopes["design_variant"] == variant)
                    )

                    def slope(quantity: str, metric: str) -> pd.Series:
                        selected = slopes[
                            shared
                            & (slopes["quantity"] == quantity)
                            & (slopes["metric"] == metric)
                        ]
                        if len(selected) != 1:
                            raise RuntimeError(
                                f"Missing dynamics slope {model}/{pooling}/"
                                f"{variant}/L{layer}/{quantity}/{metric}"
                            )
                        return selected.iloc[0]

                    assoc_mask = (
                        (associations["model"] == model)
                        & (associations["hidden_pooling"] == pooling)
                        & (
                            pd.to_numeric(associations["layer"]).astype(int)
                            == layer
                        )
                        & (associations["design_variant"] == variant)
                    )

                    def association(metric: str) -> pd.Series:
                        selected = associations[
                            assoc_mask
                            & (associations["attention_metric"] == metric)
                        ]
                        if len(selected) != 1:
                            raise RuntimeError(
                                f"Missing dynamics association {model}/{pooling}/"
                                f"{variant}/L{layer}/{metric}"
                            )
                        return selected.iloc[0]

                    row_fraction = slope(
                        "attention", "row_effective_fraction"
                    )
                    row_effective = slope(
                        "attention", "row_effective_tokens"
                    )
                    effective = slope(
                        "attention", "needle_effective_number"
                    )
                    coverage = slope(
                        "attention", "needle_relative_coverage"
                    )
                    counter_noise = slope("hidden_counter", "counter_noise")
                    coverage_association = association(
                        "needle_relative_coverage"
                    )
                    row_association = association("row_effective_fraction")
                    rows.append(
                        {
                            "model": model,
                            "pooling": pooling,
                            "head_scope": head_scope,
                            "selection": selection,
                            "layer": int(layer),
                            "variant": variant,
                            "row_effective_fraction_slope": float(
                                row_fraction["mean_slope_per_full_1_to_10_range"]
                            ),
                            "row_effective_tokens_slope": float(
                                row_effective[
                                    "mean_slope_per_full_1_to_10_range"
                                ]
                            ),
                            "needle_effective_number_slope": float(
                                effective["mean_slope_per_full_1_to_10_range"]
                            ),
                            "needle_relative_coverage_slope": float(
                                coverage["mean_slope_per_full_1_to_10_range"]
                            ),
                            "counter_noise_slope": float(
                                counter_noise[
                                    "mean_slope_per_full_1_to_10_range"
                                ]
                            ),
                            "counter_noise_slope_low": float(
                                counter_noise["ci95_low"]
                            ),
                            "counter_noise_slope_high": float(
                                counter_noise["ci95_high"]
                            ),
                            "coverage_noise_correlation": float(
                                coverage_association["correlation"]
                            ),
                            "coverage_noise_correlation_low": float(
                                coverage_association["ci95_low"]
                            ),
                            "coverage_noise_correlation_high": float(
                                coverage_association["ci95_high"]
                            ),
                            "row_dispersion_noise_correlation": float(
                                row_association["correlation"]
                            ),
                            "row_dispersion_noise_correlation_low": float(
                                row_association["ci95_low"]
                            ),
                            "row_dispersion_noise_correlation_high": float(
                                row_association["ci95_high"]
                            ),
                        }
                    )
    return rows


def _prompt_counter_selected_rows(
    frames: dict[str, pd.DataFrame],
    layer_sweep_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        *_prompt_counter_selected_rows_for_scope(
            frames["slopes"],
            frames["associations"],
            layer_sweep_rows,
            head_scope="discovery_top8_broad",
        ),
        *_prompt_counter_selected_rows_for_scope(
            frames["all_head_slopes"],
            frames["all_head_associations"],
            layer_sweep_rows,
            head_scope="all_heads_control",
        ),
    ]


def _table_prompt_counter_dynamics_html(
    rows: list[dict[str, Any]],
) -> str:
    return "".join(
        "<tr>"
        f"<td>{html.escape(row['model'])}</td>"
        f"<td>{html.escape(row['pooling'].replace('_', '-'))}</td>"
        f"<td>{html.escape(row['head_scope'].replace('_', '-'))}</td>"
        f"<td>{html.escape(row['selection'].replace('_', '-'))}</td>"
        f"<td>L{int(row['layer'])}</td>"
        f"<td>{html.escape(row['variant'])}</td>"
        f"<td>{_number(row['row_effective_tokens_slope'], 2, signed=True)}</td>"
        f"<td>{_number(row['row_effective_fraction_slope'], 4, signed=True)}</td>"
        f"<td>{_number(row['needle_effective_number_slope'], 3, signed=True)}</td>"
        f"<td>{_number(row['needle_relative_coverage_slope'], 3, signed=True)}</td>"
        f"<td>{_number(row['counter_noise_slope'], 3, signed=True)} "
        f"[{_number(row['counter_noise_slope_low'], 3, signed=True)}, "
        f"{_number(row['counter_noise_slope_high'], 3, signed=True)}]</td>"
        f"<td>{_number(row['coverage_noise_correlation'], 3, signed=True)} "
        f"[{_number(row['coverage_noise_correlation_low'], 3, signed=True)}, "
        f"{_number(row['coverage_noise_correlation_high'], 3, signed=True)}]</td>"
        f"<td>{_number(row['row_dispersion_noise_correlation'], 3, signed=True)} "
        f"[{_number(row['row_dispersion_noise_correlation_low'], 3, signed=True)}, "
        f"{_number(row['row_dispersion_noise_correlation_high'], 3, signed=True)}]</td>"
        "</tr>"
        for row in rows
    )


def _prompt_counter_association_svg(
    associations: pd.DataFrame,
    layer_sweep_rows: list[dict[str, Any]],
) -> str:
    """All-layer occurrence-adjusted correlation for relative coverage."""

    width, height = 1120, 720
    panel_width, panel_height = 430, 225
    lefts = (105, 635)
    tops = (120, 435)
    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="prompt-dynamics-title prompt-dynamics-desc">',
        '<title id="prompt-dynamics-title">Occurrence-adjusted association between needle coverage and counter noise across layers</title>',
        '<desc id="prompt-dynamics-desc">Four panels show Qwen and Gemma by span-end and span-mean hidden pooling. The horizontal axis is zero-based post-block layer. The vertical axis is the confirmation correlation between top-eight-bank relative needle coverage and full-space counter noise after subtracting each occurrence mean. One line is shown for each V4 panel. Vertical dotted and solid markers identify the prompt probe-optimal and manifold-display layers.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for variant_index, variant in enumerate(VARIANTS):
        x = 300 + variant_index * 145
        parts.extend(
            [
                f'<line x1="{x}" y1="42" x2="{x+28}" y2="42" stroke="{VARIANT_COLORS[variant]}" stroke-width="3"/>',
                f'<text x="{x+36}" y="46" font-size="11">{variant}</text>',
            ]
        )
    for model_index, model in enumerate(MODELS):
        for pooling_index, pooling in enumerate(POOLINGS):
            left = lefts[pooling_index]
            top = tops[model_index]
            subset = associations[
                (associations["model"] == model)
                & (associations["hidden_pooling"] == pooling)
                & (
                    associations["attention_metric"]
                    == "needle_relative_coverage"
                )
            ].copy()
            layers = sorted(pd.to_numeric(subset["layer"]).astype(int).unique())
            if not layers:
                raise RuntimeError(f"No dynamics association rows for {model}/{pooling}")

            def project_x(layer: int) -> float:
                return left + (layer - layers[0]) / max(layers[-1] - layers[0], 1) * panel_width

            def project_y(value: float) -> float:
                return top + (1.0 - (value + 1.0) / 2.0) * panel_height

            parts.extend(
                [
                    f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="{AURORA["snow_white"]}" stroke="{AURORA["frost_gray"]}" stroke-opacity=".45"/>',
                    f'<line x1="{left}" y1="{project_y(0):.2f}" x2="{left+panel_width}" y2="{project_y(0):.2f}" stroke="{AURORA["warm_brown"]}" stroke-width="1" opacity=".65"/>',
                    f'<text x="{left}" y="{top-18}" font-size="15" font-weight="700">{model} · {pooling.replace("_", "-")}</text>',
                ]
            )
            for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
                y = project_y(tick)
                parts.append(
                    f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="10" fill="{AURORA["frost_gray"]}">{tick:g}</text>'
                )
            selection_rows = [
                row
                for row in layer_sweep_rows
                if row["model"] == model and row["pooling"] == pooling
            ]
            for flag, dash, color in (
                ("probe_optimal", "5 4", AURORA["warm_brown"]),
                ("manifold_display", "", AURORA["midnight_indigo"]),
            ):
                layer = int(next(row["layer"] for row in selection_rows if row[flag]))
                x = project_x(layer)
                parts.append(
                    f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+panel_height}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}" opacity=".7"/>'
                )
            for variant in VARIANTS:
                group = subset[subset["design_variant"] == variant].sort_values(
                    "layer"
                )
                points = [
                    (project_x(int(row.layer)), project_y(float(row.correlation)))
                    for row in group.itertuples()
                ]
                path = " ".join(
                    ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
                    for index, (x, y) in enumerate(points)
                )
                parts.append(
                    f'<path d="{path}" fill="none" stroke="{VARIANT_COLORS[variant]}" stroke-width="2" opacity=".88"/>'
                )
            parts.extend(
                [
                    f'<text x="{left+panel_width/2}" y="{top+panel_height+34}" text-anchor="middle" font-size="11">post-block decoder layer</text>',
                    f'<text x="{left-58}" y="{top+panel_height/2}" transform="rotate(-90 {left-58} {top+panel_height/2})" text-anchor="middle" font-size="11">occurrence-adjusted correlation</text>',
                ]
            )
    parts.extend(["</g>", "</svg>"])
    return "".join(parts)


def _prompt_counter_dynamics_conclusion_html(
    rows: list[dict[str, Any]],
) -> str:
    statements: list[str] = []
    for model in MODELS:
        for pooling in POOLINGS:
            selected = [
                row
                for row in rows
                if row["model"] == model
                and row["pooling"] == pooling
                and row["selection"] == "manifold_display"
                and row["head_scope"] == "discovery_top8_broad"
            ]
            controls = [
                row
                for row in rows
                if row["model"] == model
                and row["pooling"] == pooling
                and row["selection"] == "manifold_display"
                and row["head_scope"] == "all_heads_control"
            ]
            if len(selected) != 4:
                raise RuntimeError(
                    f"Incomplete manifold dynamics rows for {model}/{pooling}"
                )
            if len(controls) != 4:
                raise RuntimeError(
                    f"Incomplete all-head dynamics rows for {model}/{pooling}"
                )
            noise_positive = sum(
                float(row["counter_noise_slope_low"]) > 0 for row in selected
            )
            noise_negative = sum(
                float(row["counter_noise_slope_high"]) < 0 for row in selected
            )
            association_positive = sum(
                float(row["coverage_noise_correlation_low"]) > 0
                for row in selected
            )
            association_negative = sum(
                float(row["coverage_noise_correlation_high"]) < 0
                for row in selected
            )
            median_coverage_slope = float(
                np.median(
                    [row["needle_relative_coverage_slope"] for row in selected]
                )
            )
            median_row_tokens_slope = float(
                np.median([row["row_effective_tokens_slope"] for row in selected])
            )
            median_row_fraction_slope = float(
                np.median([row["row_effective_fraction_slope"] for row in selected])
            )
            median_noise_slope = float(
                np.median([row["counter_noise_slope"] for row in selected])
            )
            median_correlation = float(
                np.median(
                    [row["coverage_noise_correlation"] for row in selected]
                )
            )
            control_coverage_slope = float(
                np.median(
                    [row["needle_relative_coverage_slope"] for row in controls]
                )
            )
            control_correlation = float(
                np.median(
                    [row["coverage_noise_correlation"] for row in controls]
                )
            )
            statements.append(
                f"<strong>{html.escape(model)} · {html.escape(pooling.replace('_', '-'))}</strong> "
                f"manifold-display layer：row effective-token count / visible-key fraction 的 "
                f"N=1→10 变化中位数分别为 "
                f"{_number(median_row_tokens_slope, 2, signed=True)} / "
                f"{_number(median_row_fraction_slope, 4, signed=True)}；"
                f"relative-coverage slope 中位数 "
                f"{_number(median_coverage_slope, 3, signed=True)}，counter-noise slope "
                f"中位数 {_number(median_noise_slope, 3, signed=True)}；四个 panels 中 noise "
                f"CI 明确为正/负分别 {noise_positive}/{noise_negative}，coverage–noise 的 "
                f"occurrence-adjusted correlation 中位数 "
                f"{_number(median_correlation, 3, signed=True)}，CI 明确为正/负分别 "
                f"{association_positive}/{association_negative}。作为选择偏差控制，all-heads 的 "
                f"relative-coverage slope / occurrence-adjusted correlation 中位数分别为 "
                f"{_number(control_coverage_slope, 3, signed=True)} / "
                f"{_number(control_correlation, 3, signed=True)}。"
            )
    return (
        '<div class="section-conclusion"><span>2.2b 当前结论 · 分开趋势与同-n关联</span><p>'
        + " ".join(statements)
        + " <strong>综合结论：</strong>绝对有效 token/needle 数通常随 n 增长，但四个 model×pooling "
        "组合的 relative coverage 都下降，说明有效覆盖增长慢于可用 needle 数。关键是没有任何组合出现 "
        "CI 明确为正的 hidden-noise 趋势：Qwen 基本持平，Gemma 在若干 panels 反而下降。因此当前数据不支持 "
        "‘n 增大 → retrieval 更 diffuse → counter 更 noisy’这一普遍链条。Gemma span-end 的 3/4 panels "
        "存在有限的同-n正相关，但方向是相对覆盖更广的样本更 noisy；它既不是 across-n 趋势，也未干预 "
        "attention。证明因果仍需对 frozen bank 做写入位置的定向 ablation/patching。</p></div>"
    )


def _attention_pooling_alignment_rows(
    run_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            _attention_analysis_root(run_root, model)
            / "tables"
            / "span_end_alignment_heads.csv"
        )
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            rows.append({"model": model, **row})
    return sorted(
        rows,
        key=lambda row: (
            MODELS.index(str(row["model"])),
            VARIANTS.index(str(row["design_variant"])),
            ATTENTION_POOLINGS.index(str(row["right_pooling"])),
        ),
    )


def _attention_analysis_root(run_root: Path, model: str) -> Path:
    """Prefer the non-destructive three-pooling analysis when complete."""

    attention_root = run_root / model / "numeric" / "attention"
    expanded = attention_root / "analysis_span_sum_v3"
    if (expanded / "attention_analysis_manifest.json").exists():
        return expanded
    return attention_root / "analysis"


def _normalized_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values)
    return values / total


def _classify_attention_phenotype(
    *,
    effective_number: float,
    dominant_share: float,
    winner_frequency: float,
    winner_mode: int,
    first_share: float,
    winner_is_first: float,
    local_count: float,
    local_effective_fraction: float,
    dominant_quarter_mass: float,
    span_mean_effective_number: float,
    span_mean_dominant_share: float,
) -> str:
    """Apply the frozen discovery-only retrieval phenotype thresholds.

    The order is part of the definition: global breadth has priority over
    local breadth, local breadth over endpoint selection, and endpoint
    phenotypes over the span-mean-only fallback.
    """

    global_broad = effective_number >= 6.0 and dominant_share <= 0.25
    selector = effective_number <= 2.0 and winner_frequency >= 0.80
    local_broad = (
        not global_broad
        and local_count >= 2.0
        and local_effective_fraction >= 0.80
        and dominant_quarter_mass >= 0.50
    )
    first_locator = (
        selector
        and winner_mode == 1
        and first_share >= 0.80
        and winner_is_first >= 0.90
    )
    span_mean_broad = (
        span_mean_effective_number >= 6.0
        and span_mean_dominant_share <= 0.25
    )
    if global_broad:
        return "global_endpoint_aggregator"
    if local_broad:
        return "partition_local_endpoint_aggregator"
    if first_locator:
        return "first_needle_locator"
    if selector:
        return "targeted_occurrence_retriever"
    if span_mean_broad:
        return "broad_span_mean_only"
    return "mixed"


def _attention_head_phenotypes(run_root: Path) -> list[dict[str, Any]]:
    """Classify discovery-selected heads from raw N=10 answer-query rows.

    Selection and classification use only discovery seeds.  The raw row is
    needed because the standard summary stores breadth but not which exact
    occurrence a selector retrieves.  The result therefore separates global
    breadth, depth-partition-local breadth, a strict first-needle locator, and
    other stable occurrence-targeted retrieval.
    """

    results: list[dict[str, Any]] = []
    for model in MODELS:
        model_root = run_root / model / "numeric"
        analysis_root = _attention_analysis_root(run_root, model)
        summary = pd.read_csv(
            analysis_root / "tables" / "discovery_head_summary.csv"
        )
        candidates = summary[
            (summary["pooling"] == "span_end")
            & summary["is_broad_candidate"].map(_bool)
        ].copy()
        candidate_lookup = {
            (str(row.design_variant), int(row.layer), int(row.head)): row
            for row in candidates.itertuples()
        }
        occurrence = pd.read_csv(
            analysis_root / "tables" / "occurrence_attention.csv.gz"
        )
        occurrence = occurrence[
            (occurrence["split"] == "discovery")
            & (occurrence["count"].astype(int) == 10)
            & (occurrence["pooling"] == "span_end")
        ].copy()
        spans: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for stimulus_id, frame in occurrence.groupby("stimulus_id", sort=False):
            frame = frame.sort_values("occurrence_index")
            if len(frame) != 10:
                raise RuntimeError(f"{model}/{stimulus_id}: expected ten spans")
            spans[str(stimulus_id)] = (
                frame["span_start"].to_numpy(dtype=int),
                frame["span_end"].to_numpy(dtype=int),
                frame["normalized_depth"].to_numpy(dtype=float),
            )
        index_path = (
            model_root
            / "attention"
            / "capture"
            / "attention_capture_index.jsonl"
        )
        index_records = [
            record
            for record in _read_jsonl(index_path)
            if str(record["split"]) == "discovery"
            and int(record["count"]) == 10
        ]
        if len(index_records) != len(VARIANTS) * 20:
            raise RuntimeError(
                f"{model}: expected 80 N=10 discovery attention rows"
            )
        capture_root = index_path.parent
        accumulators: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(
            lambda: {
                "samples": 0,
                "endpoint_profile": np.zeros(10, dtype=np.float64),
                "span_mean_profile": np.zeros(10, dtype=np.float64),
                "span_sum_profile": np.zeros(10, dtype=np.float64),
                "winner_counts": np.zeros(10, dtype=np.int64),
                "effective_number": 0.0,
                "span_mean_effective_number": 0.0,
                "span_sum_effective_number": 0.0,
                "end_sum_profile_cosine": 0.0,
                "end_sum_winner_match": 0.0,
                "first_share": 0.0,
                "winner_is_first": 0.0,
                "local_count": 0.0,
                "local_effective_fraction": 0.0,
                "quarter_mass": np.zeros(4, dtype=np.float64),
            }
        )
        for record in index_records:
            stimulus_id = str(record["stimulus_id"])
            variant = str(record["design_variant"])
            starts, ends, depths = spans[stimulus_id]
            selected = candidates[candidates["design_variant"] == variant]
            by_layer = {
                int(layer): frame.sort_values("head")
                for layer, frame in selected.groupby("layer", sort=True)
            }
            raw_path = capture_root / str(record["raw_attention_shard_path"])
            with np.load(_native_open_path(raw_path), allow_pickle=False) as raw:
                key_starts = np.asarray(raw["key_starts"], dtype=int)
                for layer, layer_candidates in by_layer.items():
                    rows = np.asarray(raw[f"layer_{layer:03d}"], dtype=np.float32)
                    head_indices = layer_candidates["head"].to_numpy(dtype=int)
                    key_start = int(key_starts[layer])
                    local_starts = starts - key_start
                    local_ends = ends - key_start
                    if (
                        np.any(local_starts < 0)
                        or np.any(local_ends > rows.shape[1])
                        or np.any(local_ends <= local_starts)
                    ):
                        raise RuntimeError(
                            f"{model}/{stimulus_id}/L{layer}: needle not visible"
                        )
                    head_rows = rows[head_indices]
                    endpoint_values = head_rows[:, local_ends - 1]
                    span_values = np.stack(
                        [
                            head_rows[:, start:end].mean(axis=1)
                            for start, end in zip(local_starts, local_ends)
                        ],
                        axis=1,
                    )
                    span_sum_values = np.stack(
                        [
                            head_rows[:, start:end].sum(axis=1)
                            for start, end in zip(local_starts, local_ends)
                        ],
                        axis=1,
                    )
                    edges = np.linspace(0, rows.shape[1], 5, dtype=int)
                    quarter_values = np.stack(
                        [
                            head_rows[:, edges[index] : edges[index + 1]].sum(
                                axis=1, dtype=np.float64
                            )
                            for index in range(4)
                        ],
                        axis=1,
                    )
                    quarter_totals = quarter_values.sum(axis=1, keepdims=True)
                    quarter_profiles = np.divide(
                        quarter_values,
                        quarter_totals,
                        out=np.zeros_like(quarter_values),
                        where=quarter_totals > 0,
                    )
                    for axis, head in enumerate(head_indices):
                        key = (variant, layer, int(head))
                        endpoint = np.asarray(endpoint_values[axis], dtype=float)
                        span_mean = np.asarray(span_values[axis], dtype=float)
                        span_sum = np.asarray(span_sum_values[axis], dtype=float)
                        metrics = partition_sample_metrics(
                            endpoint,
                            depths,
                            partitions=4,
                        )
                        endpoint_profile = _normalized_profile(endpoint)
                        span_mean_profile = _normalized_profile(span_mean)
                        span_sum_profile = _normalized_profile(span_sum)
                        span_denominator = float(np.square(span_mean_profile).sum())
                        span_sum_denominator = float(np.square(span_sum_profile).sum())
                        alignment_denominator = float(
                            np.linalg.norm(endpoint_profile)
                            * np.linalg.norm(span_sum_profile)
                        )
                        accumulator = accumulators[key]
                        accumulator["samples"] += 1
                        accumulator["endpoint_profile"] += endpoint_profile
                        accumulator["span_mean_profile"] += span_mean_profile
                        accumulator["span_sum_profile"] += span_sum_profile
                        winner = int(metrics["winner_occurrence_index"]) - 1
                        accumulator["winner_counts"][winner] += 1
                        accumulator["effective_number"] += float(
                            metrics["effective_number"]
                        )
                        accumulator["span_mean_effective_number"] += (
                            1.0 / span_denominator
                            if span_denominator > 0
                            else 0.0
                        )
                        accumulator["span_sum_effective_number"] += (
                            1.0 / span_sum_denominator
                            if span_sum_denominator > 0
                            else 0.0
                        )
                        accumulator["end_sum_profile_cosine"] += (
                            float(np.dot(endpoint_profile, span_sum_profile))
                            / alignment_denominator
                            if alignment_denominator > 0
                            else 0.0
                        )
                        accumulator["end_sum_winner_match"] += float(
                            int(np.argmax(endpoint_profile))
                            == int(np.argmax(span_sum_profile))
                        )
                        accumulator["first_share"] += float(
                            metrics["first_occurrence_share"]
                        )
                        accumulator["winner_is_first"] += float(
                            metrics["winner_is_first"]
                        )
                        accumulator["local_count"] += float(
                            metrics["local_needle_count"]
                        )
                        accumulator["local_effective_fraction"] += float(
                            metrics["local_effective_fraction"]
                        )
                        accumulator["quarter_mass"] += quarter_profiles[axis]
        for key, source in sorted(accumulators.items()):
            variant, layer, head = key
            samples = int(source["samples"])
            if samples != 20:
                raise RuntimeError(
                    f"{model}/{variant}/L{layer}H{head}: {samples} discovery rows"
                )
            endpoint_profile = source["endpoint_profile"] / samples
            span_mean_profile = source["span_mean_profile"] / samples
            span_sum_profile = source["span_sum_profile"] / samples
            winner_counts = np.asarray(source["winner_counts"], dtype=int)
            winner_mode = int(np.argmax(winner_counts)) + 1
            winner_frequency = float(winner_counts.max() / samples)
            effective = float(source["effective_number"] / samples)
            dominant_share = float(endpoint_profile.max())
            first_share = float(source["first_share"] / samples)
            winner_is_first = float(source["winner_is_first"] / samples)
            local_count = float(source["local_count"] / samples)
            local_fraction = float(
                source["local_effective_fraction"] / samples
            )
            quarter_profile = source["quarter_mass"] / samples
            dominant_quarter_mass = float(quarter_profile.max())
            span_effective = float(
                source["span_mean_effective_number"] / samples
            )
            span_dominant_share = float(span_mean_profile.max())
            span_sum_effective = float(
                source["span_sum_effective_number"] / samples
            )
            span_sum_dominant_share = float(span_sum_profile.max())
            selector = bool(
                effective <= 2.0 and winner_frequency >= 0.80
            )
            phenotype = _classify_attention_phenotype(
                effective_number=effective,
                dominant_share=dominant_share,
                winner_frequency=winner_frequency,
                winner_mode=winner_mode,
                first_share=first_share,
                winner_is_first=winner_is_first,
                local_count=local_count,
                local_effective_fraction=local_fraction,
                dominant_quarter_mass=dominant_quarter_mass,
                span_mean_effective_number=span_effective,
                span_mean_dominant_share=span_dominant_share,
            )
            candidate = candidate_lookup[key]
            results.append(
                {
                    "model": model,
                    "variant": variant,
                    "layer": layer,
                    "head": head,
                    "candidate_rank": int(float(candidate.candidate_rank)),
                    "pool_primary": float(candidate.pool_primary),
                    "pool_sum": float(candidate.pool_sum),
                    "pool_coverage": float(candidate.pool_coverage),
                    "pool_enrichment": float(candidate.pool_enrichment),
                    "samples": samples,
                    "effective_number_mean": effective,
                    "dominant_occurrence": int(np.argmax(endpoint_profile)) + 1,
                    "dominant_occurrence_mean_share": dominant_share,
                    "winner_occurrence_mode": winner_mode,
                    "winner_occurrence_mode_frequency": winner_frequency,
                    "first_occurrence_share_mean": first_share,
                    "winner_is_first_mean": winner_is_first,
                    "local_needle_count_mean": local_count,
                    "local_effective_fraction_mean": local_fraction,
                    "row_dominant_quarter": int(np.argmax(quarter_profile)) + 1,
                    "row_dominant_quarter_mass": dominant_quarter_mass,
                    "span_mean_effective_number_mean": span_effective,
                    "span_mean_dominant_occurrence_mean_share": span_dominant_share,
                    "span_sum_effective_number_mean": span_sum_effective,
                    "span_sum_dominant_occurrence": int(
                        np.argmax(span_sum_profile)
                    )
                    + 1,
                    "span_sum_dominant_occurrence_mean_share": (
                        span_sum_dominant_share
                    ),
                    "end_sum_profile_cosine_mean": float(
                        source["end_sum_profile_cosine"] / samples
                    ),
                    "end_sum_winner_match_rate": float(
                        source["end_sum_winner_match"] / samples
                    ),
                    "phenotype": phenotype,
                    "target_occurrence": winner_mode if selector else None,
                    "endpoint_profile": [
                        float(value) for value in endpoint_profile
                    ],
                    "span_mean_profile": [
                        float(value) for value in span_mean_profile
                    ],
                    "span_sum_profile": [
                        float(value) for value in span_sum_profile
                    ],
                }
            )

        # The earlier Qwen partition analysis used the same discovery rules.
        # Requiring agreement for the two broad classes guards the generic
        # two-model implementation against silent indexing errors.
        if model == "Qwen3-8B":
            existing_path = (
                model_root
                / "attention"
                / "analysis"
                / "partitioning"
                / "all_candidate_head_phenotypes_by_split.csv"
            )
            existing = pd.read_csv(existing_path)
            existing = existing[existing["split"] == "discovery"]
            current = [row for row in results if row["model"] == model]
            for phenotype in (
                "global_endpoint_aggregator",
                "partition_local_endpoint_aggregator",
            ):
                old_set = {
                    (str(row.design_variant), int(row.layer), int(row.head))
                    for row in existing[existing["phenotype"] == phenotype].itertuples()
                }
                new_set = {
                    (str(row["variant"]), int(row["layer"]), int(row["head"]))
                    for row in current
                    if row["phenotype"] == phenotype
                }
                if old_set != new_set:
                    raise RuntimeError(
                        f"Qwen {phenotype} audit mismatch: "
                        f"old={len(old_set)} new={len(new_set)}"
                    )
    return sorted(
        results,
        key=lambda row: (
            MODELS.index(str(row["model"])),
            VARIANTS.index(str(row["variant"])),
            int(row["candidate_rank"]),
        ),
    )


def _attention_outcome_effect_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            _attention_analysis_root(run_root, model)
            / "tables"
            / "confirmation_wrong_minus_correct_effects.csv"
        )
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            rows.append({"model": model, **row})
    return rows


def _qwen_partition_summary(run_root: Path) -> dict[str, Any]:
    root = (
        run_root
        / "Qwen3-8B"
        / "numeric"
        / "attention"
        / "analysis"
        / "partitioning"
    )
    counts = pd.read_csv(root / "all_candidate_phenotype_counts.csv")
    bank = pd.read_csv(root / "phenotype_bank_coverage.csv")
    by_split = pd.read_csv(root / "all_candidate_head_phenotypes_by_split.csv")
    assessment = _read_json(root / "partition_hypothesis_assessment.json")
    manifest = _read_json(root / "partition_analysis_manifest.json")

    key_phenotypes = (
        "global_endpoint_aggregator",
        "partition_local_endpoint_aggregator",
        "occurrence_endpoint_selector",
    )
    count_lookup = {
        (str(row.design_variant), str(row.phenotype)): int(row.heads)
        for row in counts.itertuples()
    }
    total_lookup = {
        variant: int(sum(count_lookup[(variant, phenotype)] for phenotype in counts[counts["design_variant"] == variant]["phenotype"].astype(str)))
        for variant in VARIANTS
    }
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for phenotype in key_phenotypes:
            selected = bank[
                (bank["design_variant"] == variant)
                & (bank["phenotype"] == phenotype)
            ]
            if len(selected) != 1:
                raise RuntimeError(f"Missing Qwen partition bank {variant}/{phenotype}")
            item = selected.iloc[0]
            rows.append(
                {
                    "variant": variant,
                    "phenotype": phenotype,
                    "heads": count_lookup[(variant, phenotype)],
                    "equal_effective_number": float(
                        item["equal_head_profile_effective_number"]
                    ),
                    "raw_effective_number": float(
                        item["raw_attention_ensemble_effective_number"]
                    ),
                    "mean_bank_mass": float(item["mean_summed_bank_endpoint_mass"]),
                }
            )

    global_rows = by_split[
        by_split["phenotype"] == "global_endpoint_aggregator"
    ].copy()
    global_rows["cell"] = (
        global_rows["design_variant"].astype(str)
        + "|"
        + global_rows["split"].astype(str)
    )
    stable = (
        global_rows.groupby(["layer", "head"])["cell"]
        .nunique()
        .loc[lambda values: values == len(VARIANTS) * 2]
        .index.tolist()
    )
    stable_labels = [f"L{int(layer)}H{int(head)}" for layer, head in stable]
    return {
        "rows": rows,
        "counts": count_lookup,
        "totals": total_lookup,
        "stable_global_heads": stable_labels,
        "assessment": assessment,
        "candidate_counts": {
            str(key): int(value)
            for key, value in manifest["candidate_counts"].items()
        },
    }


def _table_attention_top_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td><code>{html.escape(str(row['pooling']))}</code></td>"
            f"<td>L{int(row['layer'])}H{int(row['head'])}</td>"
            f"<td>{_number(row['total_mass'], 6)}</td>"
            f"<td>{_number(row['coverage'])}</td>"
            f"<td>{_number(row['effective_number'], 2)}</td>"
            f"<td>{_number(row['primary'], 6)}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_attention_pooling_alignment_html(
    rows: list[dict[str, Any]],
) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['design_variant']))}</td>"
            f"<td><code>{html.escape(str(row['left_pooling']))}</code> ↔ <code>{html.escape(str(row['right_pooling']))}</code></td>"
            f"<td>{int(row['heads_compared'])}</td>"
            f"<td>{_number(row['spearman_primary'])}</td>"
            f"<td>{int(row['top_k_intersection'])}/{int(row['top_k'])}</td>"
            f"<td>{_number(row['top_k_jaccard'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _attention_pooling_alignment_conclusion_html(
    rows: list[dict[str, Any]],
) -> str:
    summaries: list[str] = []
    for model in MODELS:
        comparisons: list[str] = []
        for right_pooling in ("span_mean", "span_sum"):
            selected = [
                row
                for row in rows
                if row["model"] == model
                and row["right_pooling"] == right_pooling
            ]
            if len(selected) != len(VARIANTS):
                raise RuntimeError(
                    f"{model}/{right_pooling}: incomplete pooling alignment rows"
                )
            correlations = [float(row["spearman_primary"]) for row in selected]
            overlaps = [int(row["top_k_intersection"]) for row in selected]
            comparisons.append(
                f"end↔{right_pooling.removeprefix('span_')} Spearman "
                f"{min(correlations):.3f}–{max(correlations):.3f}, "
                f"top-8 overlap {min(overlaps)}–{max(overlaps)}/8"
            )
        summaries.append(
            f"<strong>{html.escape(model)}</strong>：" + "；".join(comparisons)
        )
    return (
        '<div class="section-conclusion"><span>3.1 pooling 对齐结论 · Endpoint 不是完整 span 的同义词</span><p>'
        + " ".join(summaries)
        + "。Spearman 衡量所有可见 heads 的固定-head排序，而 top-8 overlap 衡量各 pooling 实际会锁定多少相同候选；"
        "即使全局相关较高，只要 top-8 不完全重合，就不能把 endpoint atlas 直接当作 full-span atlas。"
        "Span-mean 与 span-sum 在记录 token 长度相同的 panel 中可产生相同 occurrence profile，"
        "但二者的总量语义仍不同：前者是 density，后者才是 query-row mass。</p></div>"
    )


def _table_partition_bank_html(rows: list[dict[str, Any]]) -> str:
    labels = {
        "global_endpoint_aggregator": "global endpoint aggregator",
        "partition_local_endpoint_aggregator": "partition-local aggregator",
        "occurrence_endpoint_selector": "occurrence selector",
    }
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td>{html.escape(labels[str(row['phenotype'])])}</td>"
            f"<td>{int(row['heads'])}</td>"
            f"<td>{_number(row['equal_effective_number'], 2)}</td>"
            f"<td>{_number(row['raw_effective_number'], 2)}</td>"
            f"<td>{_number(row['mean_bank_mass'], 5)}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _mix_hex(left: str, right: str, fraction: float) -> str:
    fraction = max(0.0, min(1.0, float(fraction)))
    left_rgb = tuple(int(left[index : index + 2], 16) for index in (1, 3, 5))
    right_rgb = tuple(int(right[index : index + 2], 16) for index in (1, 3, 5))
    values = tuple(
        round(a + fraction * (b - a)) for a, b in zip(left_rgb, right_rgb)
    )
    return "#" + "".join(f"{value:02X}" for value in values)


def _aurora_sequential(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    anchors = (
        AURORA["midnight_indigo"],
        AURORA["polar_violet"],
        AURORA["ice_cyan"],
        AURORA["aurora_yellow"],
    )
    scaled = value * (len(anchors) - 1)
    index = min(int(scaled), len(anchors) - 2)
    return _mix_hex(anchors[index], anchors[index + 1], scaled - index)


def _attention_head_atlas_svg(
    atlas_rows: list[dict[str, Any]],
    phenotypes: list[dict[str, Any]],
    *,
    variant: str | None = None,
    pooling: str = "span_end",
) -> str:
    """Layer-by-head atlas for one registered attention pooling."""

    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"Unknown attention-atlas variant: {variant}")
    if pooling not in ATTENTION_POOLINGS:
        raise ValueError(f"Unknown attention-atlas pooling: {pooling}")
    display_variants = VARIANTS if variant is None else (variant,)
    width, height = 1260, 900
    panel_width = 230 if variant is None else 900
    panel_height = 300
    lefts = (115, 405, 695, 985) if variant is None else (180,)
    tops = (130, 535)
    id_suffix = (
        ("all" if variant is None else variant.replace(".", "-"))
        + "-"
        + pooling.replace("_", "-")
    )
    phenotype_lookup = {
        (
            str(row["model"]),
            str(row["variant"]),
            int(row["layer"]),
            int(row["head"]),
        ): row
        for row in phenotypes
    }
    selected_pooling = [row for row in atlas_rows if row["pooling"] == pooling]
    scales: dict[str, tuple[float, float]] = {}
    for model in MODELS:
        values = np.asarray(
            [
                float(row["pool_primary"])
                for row in selected_pooling
                if row["model"] == model and float(row["pool_primary"]) > 0
            ],
            dtype=float,
        )
        logs = np.log10(values)
        low, high = np.quantile(logs, [0.01, 0.995])
        scales[model] = (float(low), float(max(high, low + 1e-9)))
    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="head-atlas-title-{id_suffix} head-atlas-desc-{id_suffix}">',
        f'<title id="head-atlas-title-{id_suffix}">All-head answer-query {html.escape(pooling)} attention atlas: {html.escape(variant or "all panels")}</title>',
        f'<desc id="head-atlas-desc-{id_suffix}">Every captured full-attention head is placed by decoder layer and head index for {html.escape(variant or "all V4 panels")}. Color is the discovery log primary score under {html.escape(pooling)}. Endpoint phenotype symbols are shown only in the span-end view.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
        '<text x="115" y="22" font-size="11" font-weight="700">color = log₁₀(primary score)</text>',
        f'<text x="115" y="39" font-size="9">primary = {html.escape(pooling)} pooled needle evidence × entropy coverage</text>',
    ]
    for index in range(120):
        x = 500 + index * 3.0
        parts.append(
            f'<rect x="{x:.1f}" y="16" width="3.5" height="16" fill="{_aurora_sequential(index/119)}"/>'
        )
    parts.extend(
        [
            '<text x="500" y="53" text-anchor="start" font-size="10">low within model</text>',
            '<text x="860" y="53" text-anchor="end" font-size="10">99.5% clipped</text>',
            (
                f'<circle cx="875" cy="24" r="6" fill="none" stroke="{PHENOTYPE_COLORS["global_endpoint_aggregator"]}" stroke-width="2"/>'
                '<text x="886" y="28" font-size="10">global broad</text>'
                f'<rect x="966" y="18" width="12" height="12" fill="none" stroke="{PHENOTYPE_COLORS["partition_local_endpoint_aggregator"]}" stroke-width="2"/>'
                '<text x="984" y="28" font-size="10">local broad</text>'
                f'<circle cx="1065" cy="24" r="4" fill="{PHENOTYPE_COLORS["first_needle_locator"]}" stroke="{AURORA["night_black"]}" stroke-width=".7"/>'
                '<text x="1074" y="28" font-size="10">first locator</text>'
                f'<circle cx="1152" cy="24" r="4" fill="{PHENOTYPE_COLORS["targeted_occurrence_retriever"]}" stroke="{AURORA["night_black"]}" stroke-width=".7"/>'
                '<text x="1161" y="28" font-size="10">weak first-focused</text>'
                if pooling == "span_end"
                else '<text x="875" y="28" font-size="10">endpoint phenotype overlay hidden for this pooling</text>'
            ),
        ]
    )
    for model_index, model in enumerate(MODELS):
        max_layers, max_heads = MODEL_HEAD_GRIDS[model]
        top = tops[model_index]
        low, high = scales[model]
        parts.append(
            f'<text x="26" y="{top+panel_height/2:.1f}" transform="rotate(-90 26 {top+panel_height/2:.1f})" text-anchor="middle" font-size="17" font-weight="700">{html.escape(model)}</text>'
        )
        for variant_index, displayed_variant in enumerate(display_variants):
            left = lefts[variant_index]
            cell_width = panel_width / max_heads
            cell_height = panel_height / max_layers
            parts.extend(
                [
                    f'<text x="{left+panel_width/2:.1f}" y="{top-22}" text-anchor="middle" font-size="14" font-weight="700">{displayed_variant}</text>',
                    f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="{AURORA["frost_gray"]}" opacity=".12" stroke="{AURORA["frost_gray"]}"/>',
                ]
            )
            panel = [
                row
                for row in selected_pooling
                if row["model"] == model and row["variant"] == displayed_variant
            ]
            for row in panel:
                layer, head = int(row["layer"]), int(row["head"])
                value = max(float(row["pool_primary"]), 10**low)
                score = (math.log10(value) - low) / (high - low)
                x = left + head * cell_width
                y = top + layer * cell_height
                key = (model, displayed_variant, layer, head)
                phenotype = phenotype_lookup.get(key) if pooling == "span_end" else None
                tooltip = (
                    f"{model} {displayed_variant} L{layer}H{head}; primary="
                    f"{float(row['pool_primary']):.6g}; N_eff="
                    f"{float(row['pool_effective_number']):.3f}"
                )
                parts.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width+.08:.2f}" height="{cell_height+.08:.2f}" fill="{_aurora_sequential(score)}"><title>{html.escape(tooltip)}</title></rect>'
                )
                if phenotype is None:
                    continue
                label = str(phenotype["phenotype"])
                center_x, center_y = x + cell_width / 2, y + cell_height / 2
                radius = max(1.4, min(cell_width, cell_height) * 0.34)
                if label == "global_endpoint_aggregator":
                    parts.append(
                        f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{radius:.2f}" fill="none" stroke="{PHENOTYPE_COLORS[label]}" stroke-width="1.25"/>'
                    )
                elif label == "partition_local_endpoint_aggregator":
                    parts.append(
                        f'<rect x="{x+.8:.2f}" y="{y+.8:.2f}" width="{max(0.5,cell_width-1.6):.2f}" height="{max(0.5,cell_height-1.6):.2f}" fill="none" stroke="{PHENOTYPE_COLORS[label]}" stroke-width="1.25"/>'
                    )
                elif label in {"first_needle_locator", "targeted_occurrence_retriever"}:
                    color = PHENOTYPE_COLORS[label]
                    parts.append(
                        f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{max(1.2,radius*.62):.2f}" fill="{color}" stroke="{AURORA["night_black"]}" stroke-width=".45"/>'
                    )
            for layer_tick in sorted({0, max_layers // 2, max_layers - 1}):
                y = top + (layer_tick + 0.5) * cell_height
                parts.append(
                    f'<text x="{left-7}" y="{y+3:.1f}" text-anchor="end" font-size="9" fill="{AURORA["frost_gray"]}">L{layer_tick}</text>'
                )
            for head_tick in sorted({0, max_heads // 2, max_heads - 1}):
                x = left + (head_tick + 0.5) * cell_width
                parts.append(
                    f'<text x="{x:.1f}" y="{top+panel_height+16}" text-anchor="middle" font-size="9" fill="{AURORA["frost_gray"]}">H{head_tick}</text>'
                )
            parts.extend(
                [
                    f'<text x="{left+panel_width/2:.1f}" y="{top+panel_height+38}" text-anchor="middle" font-size="10">attention head index</text>',
                    f'<text transform="translate({left-43} {top+panel_height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="10">post-block layer</text>',
                ]
            )
    parts.append("</g></svg>")
    return "".join(parts)


def _attention_head_atlas_interactive_html(
    atlas_rows: list[dict[str, Any]],
    phenotypes: list[dict[str, Any]],
) -> str:
    variant_buttons = "".join(
        (
            f'<button type="button" class="atlas-button" data-atlas-variant="{variant}" '
            f'aria-pressed="{"true" if index == 0 else "false"}">{variant}</button>'
        )
        for index, variant in enumerate(VARIANTS)
    )
    pooling_buttons = "".join(
        (
            f'<button type="button" class="atlas-pooling-button" data-atlas-pooling="{pooling}" '
            f'aria-pressed="{"true" if index == 0 else "false"}">{pooling}</button>'
        )
        for index, pooling in enumerate(ATTENTION_POOLINGS)
    )
    panels = "".join(
        (
            f'<div class="atlas-panel" data-atlas-variant="{variant}" data-atlas-pooling="{pooling}"'
            + ("" if variant_index == 0 and pooling_index == 0 else " hidden")
            + ">"
            + _attention_head_atlas_svg(
                atlas_rows, phenotypes, variant=variant, pooling=pooling
            )
            + "</div>"
        )
        for variant_index, variant in enumerate(VARIANTS)
        for pooling_index, pooling in enumerate(ATTENTION_POOLINGS)
    )
    return (
        '<div class="atlas-interactive" id="attention-atlas-interactive">'
        '<div class="atlas-controls" role="group" aria-label="V4 panel">'
        '<span>V4 panel</span>'
        + variant_buttons
        + '<span>Pooling</span>'
        + pooling_buttons
        + "</div>"
        + panels
        + "</div>"
    )


def _table_head_phenotype_counts_html(rows: list[dict[str, Any]]) -> str:
    order = (
        "global_endpoint_aggregator",
        "partition_local_endpoint_aggregator",
        "first_needle_locator",
        "targeted_occurrence_retriever",
        "broad_span_mean_only",
        "mixed",
    )
    rendered: list[str] = []
    for model in MODELS:
        for variant in VARIANTS:
            selected = [
                row
                for row in rows
                if row["model"] == model and row["variant"] == variant
            ]
            counts = {
                phenotype: sum(
                    row["phenotype"] == phenotype for row in selected
                )
                for phenotype in order
            }
            rendered.append(
                "<tr>"
                f"<td>{html.escape(model)}</td><td>{variant}</td>"
                + "".join(f"<td>{counts[key]}</td>" for key in order)
                + f"<td>{len(selected)}</td></tr>"
            )
    return "".join(rendered)


def _table_head_representatives_html(rows: list[dict[str, Any]]) -> str:
    categories = (
        "global_endpoint_aggregator",
        "partition_local_endpoint_aggregator",
        "first_needle_locator",
        "targeted_occurrence_retriever",
    )
    labels = {
        "global_endpoint_aggregator": "global broad",
        "partition_local_endpoint_aggregator": "local broad",
        "first_needle_locator": "first-needle locator",
        "targeted_occurrence_retriever": "targeted retrieval",
    }
    rendered: list[str] = []
    for model in MODELS:
        for variant in VARIANTS:
            for category in categories:
                selected = sorted(
                    [
                        row
                        for row in rows
                        if row["model"] == model
                        and row["variant"] == variant
                        and row["phenotype"] == category
                    ],
                    key=lambda row: (
                        -float(row["pool_primary"]),
                        int(row["candidate_rank"]),
                    ),
                )
                if not selected:
                    continue
                row = selected[0]
                target = (
                    f"needle {int(row['target_occurrence'])}"
                    if row["target_occurrence"] is not None
                    else "—"
                )
                rendered.append(
                    "<tr>"
                    f"<td>{html.escape(model)}</td><td>{variant}</td>"
                    f"<td>{html.escape(labels[category])}</td>"
                    f"<td>L{int(row['layer'])}H{int(row['head'])}</td>"
                    f"<td>{target}</td>"
                    f"<td>{_number(row['effective_number_mean'], 2)}</td>"
                    f"<td>{_number(row['pool_sum'], 6)}</td>"
                    f"<td>{_number(row['pool_coverage'], 3)}</td>"
                    f"<td>{_number(row['row_dominant_quarter_mass'], 3)}</td>"
                    "</tr>"
                )
    return "".join(rendered)


def _representative_head_profiles_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1220, 660
    panel_width, panel_height = 225, 205
    lefts = (105, 395, 685, 975)
    tops = (105, 405)
    categories = (
        "global_endpoint_aggregator",
        "partition_local_endpoint_aggregator",
        "first_needle_locator",
        "targeted_occurrence_retriever",
    )
    labels = ("global broad", "partition-local broad", "strict first locator", "weak first-focused")
    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="profile-title profile-desc">',
        '<title id="profile-title">Representative discovery attention profiles by head phenotype</title>',
        '<desc id="profile-desc">For v4.1, each panel shows the highest-primary discovery candidate in one phenotype. Green is global broad, cyan is partition-local broad, yellow is a strict first-needle locator, and pink is the best weaker first-focused head. The x axis is needle occurrence one through ten. The y axis is mean normalized endpoint attention share, so each prompt profile sums to one before averaging.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for column, label in enumerate(labels):
        parts.append(
            f'<text x="{lefts[column]+panel_width/2:.1f}" y="35" text-anchor="middle" font-size="13" font-weight="700">{label}</text>'
        )
    for model_index, model in enumerate(MODELS):
        top = tops[model_index]
        parts.append(
            f'<text x="25" y="{top+panel_height/2:.1f}" transform="rotate(-90 25 {top+panel_height/2:.1f})" text-anchor="middle" font-size="16" font-weight="700">{html.escape(model)}</text>'
        )
        for column, category in enumerate(categories):
            left = lefts[column]
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["variant"] == "v4.1"
                    and row["phenotype"] == category
                ],
                key=lambda row: -float(row["pool_primary"]),
            )
            parts.append(
                f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="{AURORA["snow_white"]}" stroke="{AURORA["frost_gray"]}" stroke-opacity=".45"/>'
            )
            for tick in (0.0, 0.5, 1.0):
                y = top + panel_height - tick * panel_height
                parts.extend(
                    [
                        f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_width}" y2="{y:.1f}" stroke="{AURORA["frost_gray"]}" opacity=".18"/>',
                        f'<text x="{left-7}" y="{y+4:.1f}" text-anchor="end" font-size="9" fill="{AURORA["frost_gray"]}">{tick:.1f}</text>',
                    ]
                )
            if not selected:
                parts.append(
                    f'<text x="{left+panel_width/2:.1f}" y="{top+panel_height/2:.1f}" text-anchor="middle" font-size="12" fill="{AURORA["frost_gray"]}">no discovery candidate</text>'
                )
                continue
            row = selected[0]
            profile = np.asarray(row["endpoint_profile"], dtype=float)
            color = PHENOTYPE_COLORS[category]
            points: list[tuple[float, float]] = []
            for index, value in enumerate(profile):
                x = left + (index + 0.5) / 10 * panel_width
                y = top + panel_height - float(value) * panel_height
                points.append((x, y))
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
                for index, (x, y) in enumerate(points)
            )
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.8"/>'
            )
            for occurrence, (x, y) in enumerate(points, start=1):
                parts.extend(
                    [
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="{color}" stroke="{AURORA["night_black"]}" stroke-width=".6"/>',
                        f'<text x="{x:.2f}" y="{top+panel_height+16}" text-anchor="middle" font-size="8">{occurrence}</text>',
                    ]
                )
            target_text = (
                f" · target {int(row['target_occurrence'])}"
                if row["target_occurrence"] is not None
                else ""
            )
            parts.append(
                f'<text x="{left+panel_width/2:.1f}" y="{top-11}" text-anchor="middle" font-size="10">L{int(row["layer"])}H{int(row["head"])} · N_eff,2 {float(row["effective_number_mean"]):.2f}{target_text}</text>'
            )
            parts.extend(
                [
                    f'<text x="{left+panel_width/2:.1f}" y="{top+panel_height+35}" text-anchor="middle" font-size="10">needle occurrence index</text>',
                    f'<text transform="translate({left-43} {top+panel_height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="10">mean endpoint share</text>',
                ]
            )
    parts.append("</g></svg>")
    return "".join(parts)


def _attention_outcome_effect_svg(
    rows: list[dict[str, Any]], *, pooling: str = "span_end"
) -> str:
    selected = [
        row
        for row in rows
        if row["pooling"] == pooling
        and row["metric"] == "pool_coverage"
    ]
    if pooling not in ATTENTION_POOLINGS:
        raise ValueError(f"Unknown outcome-effect pooling: {pooling}")
    width, height = 1220, 690
    panel_width, panel_height = 220, 215
    lefts = (110, 400, 690, 980)
    tops = (105, 420)
    maximum = max(
        0.05,
        float(
            np.quantile(
                np.abs(
                    [
                        float(row["wrong_minus_correct_count_adjusted"])
                        for row in selected
                        if math.isfinite(
                            float(row["wrong_minus_correct_count_adjusted"])
                        )
                    ]
                ),
                0.98,
            )
        ),
    )
    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="outcome-head-title-{pooling} outcome-head-desc-{pooling}">',
        f'<title id="outcome-head-title-{pooling}">Count-adjusted correct versus wrong {pooling} attention breadth for ranked heads</title>',
        f'<desc id="outcome-head-desc-{pooling}">Each cell is count-adjusted wrong minus correct entropy coverage for one discovery-ranked {pooling} head. Pink is a negative effect, green is positive, white is zero, and a dark outline means the seed-cluster bootstrap interval excludes zero without family-wise correction.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
        f'<rect x="470" y="19" width="35" height="15" fill="{AURORA["sunset_pink"]}"/><text x="512" y="31" font-size="10">wrong narrower</text>',
        f'<rect x="635" y="19" width="35" height="15" fill="{AURORA["snow_white"]}" stroke="{AURORA["frost_gray"]}"/><text x="677" y="31" font-size="10">wrong−correct = 0</text>',
        f'<rect x="790" y="19" width="35" height="15" fill="{AURORA["aurora_green"]}"/><text x="832" y="31" font-size="10">wrong broader</text>',
    ]
    for model_index, model in enumerate(MODELS):
        top = tops[model_index]
        parts.append(
            f'<text x="27" y="{top+panel_height/2:.1f}" transform="rotate(-90 27 {top+panel_height/2:.1f})" text-anchor="middle" font-size="16" font-weight="700">{html.escape(model)}</text>'
        )
        for variant_index, variant in enumerate(VARIANTS):
            left = lefts[variant_index]
            panel = sorted(
                [
                    row
                    for row in selected
                    if row["model"] == model and row["design_variant"] == variant
                ],
                key=lambda row: int(row["head_rank"]),
            )
            if len(panel) != 8:
                raise RuntimeError(
                    f"{model}/{variant}: expected eight {pooling} outcome heads"
                )
            cell_height = panel_height / 8
            parts.extend(
                [
                    f'<text x="{left+panel_width/2:.1f}" y="{top-16}" text-anchor="middle" font-size="14" font-weight="700">{variant}</text>',
                    f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="{AURORA["snow_white"]}" stroke="{AURORA["frost_gray"]}" stroke-opacity=".45"/>',
                ]
            )
            for index, row in enumerate(panel):
                value = float(row["wrong_minus_correct_count_adjusted"])
                scaled = max(-1.0, min(1.0, value / maximum))
                color = (
                    _mix_hex(AURORA["snow_white"], AURORA["aurora_green"], scaled)
                    if scaled >= 0
                    else _mix_hex(AURORA["snow_white"], AURORA["sunset_pink"], -scaled)
                )
                y = top + index * cell_height
                significant = bool(
                    float(row["bootstrap_ci_low"]) > 0
                    or float(row["bootstrap_ci_high"]) < 0
                )
                parts.extend(
                    [
                        f'<rect x="{left}" y="{y:.2f}" width="{panel_width}" height="{cell_height:.2f}" fill="{color}" stroke="{AURORA["night_black"] if significant else AURORA["snow_white"]}" stroke-width="{1.8 if significant else .7}"><title>{html.escape(model)} {variant} rank {int(row["head_rank"])} L{int(row["layer"])}H{int(row["head"])}; wrong-correct coverage={value:.4f}; CI [{float(row["bootstrap_ci_low"]):.4f},{float(row["bootstrap_ci_high"]):.4f}]</title></rect>',
                        f'<text x="{left+7}" y="{y+cell_height*.68:.2f}" font-size="9">#{int(row["head_rank"])} · L{int(row["layer"])}H{int(row["head"])}</text>',
                        f'<text x="{left+panel_width-7}" y="{y+cell_height*.68:.2f}" text-anchor="end" font-size="9">{value:+.3f}</text>',
                    ]
                )
            parts.extend(
                [
                    f'<text x="{left+panel_width/2:.1f}" y="{top+panel_height+22}" text-anchor="middle" font-size="10">discovery rank 1→8</text>',
                    f'<text transform="translate({left-39} {top+panel_height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="10">wrong − correct coverage</text>',
                ]
            )
    parts.append("</g></svg>")
    return "".join(parts)


def _attention_outcome_effect_interactive_html(
    rows: list[dict[str, Any]],
) -> str:
    buttons = "".join(
        f'<button type="button" class="outcome-pooling-button" data-outcome-pooling="{pooling}" aria-pressed="{"true" if index == 0 else "false"}">{pooling}</button>'
        for index, pooling in enumerate(ATTENTION_POOLINGS)
    )
    panels = "".join(
        f'<div class="outcome-pooling-panel" data-outcome-pooling="{pooling}"{"" if index == 0 else " hidden"}>{_attention_outcome_effect_svg(rows, pooling=pooling)}</div>'
        for index, pooling in enumerate(ATTENTION_POOLINGS)
    )
    return (
        '<div class="atlas-interactive" id="attention-outcome-interactive">'
        '<div class="atlas-controls" role="group" aria-label="attention pooling">'
        '<span>Pooling</span>'
        + buttons
        + "</div>"
        + panels
        + "</div>"
    )


def _table_attention_outcome_summary_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for model in MODELS:
        for pooling in ATTENTION_POOLINGS:
            for metric in (
                "pool_primary",
                "pool_coverage",
                "pool_enrichment",
                "pool_min_to_mean",
            ):
                selected = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["pooling"] == pooling
                    and row["metric"] == metric
                    and math.isfinite(float(row["wrong_minus_correct_count_adjusted"]))
                ]
                negative = sum(float(row["bootstrap_ci_high"]) < 0 for row in selected)
                positive = sum(float(row["bootstrap_ci_low"]) > 0 for row in selected)
                values = [float(row["wrong_minus_correct_count_adjusted"]) for row in selected]
                rendered.append(
                    "<tr>"
                    f"<td>{html.escape(model)}</td><td><code>{pooling}</code></td><td><code>{metric}</code></td>"
                    f"<td>{len(selected)}</td><td>{negative}</td><td>{positive}</td>"
                    f"<td>{_number(np.median(values), 4, signed=True)}</td>"
                    f"<td>[{_number(min(values), 4, signed=True)}, {_number(max(values), 4, signed=True)}]</td>"
                    "</tr>"
                )
    return "".join(rendered)


def _attention_outcome_conclusion_html(rows: list[dict[str, Any]]) -> str:
    """Report sparse wrong-correct associations for every attention pooling."""

    def significant_counts(model: str, pooling: str, metric: str) -> tuple[int, int]:
        selected = [
            row
            for row in rows
            if row["model"] == model
            and row["pooling"] == pooling
            and row["metric"] == metric
            and math.isfinite(float(row["wrong_minus_correct_count_adjusted"]))
        ]
        return (
            sum(float(row["bootstrap_ci_high"]) < 0 for row in selected),
            sum(float(row["bootstrap_ci_low"]) > 0 for row in selected),
        )

    def model_summary(model: str, metric: str) -> str:
        values = [
            significant_counts(model, pooling, metric)
            for pooling in ATTENTION_POOLINGS
        ]
        return " / ".join(
            f"{negative} negative, {positive} positive"
            for negative, positive in values
        )

    return (
        '<div class="section-conclusion"><span>3.6a 结论 · 正误差异稀疏，而非全局 shutoff</span><p>'
        "以下三项顺序均为 span-end / span-mean / span-sum；每项分别列出 CI&lt;0 与 CI&gt;0 的 cell 数。"
        f"Coverage：Qwen {model_summary('Qwen3-8B', 'pool_coverage')}，Gemma "
        f"{model_summary('Gemma4-E4B', 'pool_coverage')}；primary score：Qwen "
        f"{model_summary('Qwen3-8B', 'pool_primary')}，Gemma "
        f"{model_summary('Gemma4-E4B', 'pool_primary')}。"
        "每个 pooling 都有 32 个 head×panel cells，而且这些单-cell intervals 未做 family-wise correction；"
        "严谨结论只是少数 discovery-ranked channels 在 wrong prompts 中显示更窄或更弱的关联信号，"
        "不是 attention 整体下降。三种 pooling 还会各自选择不同 top-8 bank，因此跨 pooling 的计数不是固定-head paired test。"
        "是否与行为上少算的具体 needles 对齐，需看下一节的 occurrence-level omission diagnostics。</p></div>"
    )


def _attention_breadth_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1080, 430
    panel_lefts = (80, 600)
    plot_width, top, bottom = 400, 74, 330
    bar_width = 22

    def y_position(value: float) -> float:
        return bottom - max(0.0, min(10.0, float(value))) / 10.0 * (bottom - top)

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="breadth-title breadth-desc">',
        '<title id="breadth-title">Effective number of needles covered by each discovery rank-1 attention head</title>',
        '<desc id="breadth-desc">For each model and V4 panel, three bars compare the occurrence breadth of the rank-one head under endpoint mass, mean per-token span density, and literal full-span attention sum.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for index, pooling in enumerate(ATTENTION_POOLINGS):
        x = 305 + index * 180
        color = POOLING_COLORS[pooling]
        parts.extend(
            [
                f'<rect x="{x}" y="19" width="15" height="15" fill="{color}"/>',
                f'<text x="{x+23}" y="32" font-size="12">{pooling.replace("_", "-")}</text>',
            ]
        )
    for panel_index, model in enumerate(MODELS):
        left = panel_lefts[panel_index]
        parts.append(
            f'<text x="{left}" y="54" font-size="17" font-weight="700">{html.escape(model)}</text>'
        )
        for tick in range(0, 11, 2):
            y = y_position(tick)
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" '
                    f'stroke="{AURORA["frost_gray"]}" opacity=".27"/>',
                    f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                    f'fill="{AURORA["frost_gray"]}">{tick}</text>',
                ]
            )
        group_width = plot_width / len(VARIANTS)
        for variant_index, variant in enumerate(VARIANTS):
            center = left + (variant_index + 0.5) * group_width
            parts.append(
                f'<text x="{center:.1f}" y="352" text-anchor="middle" font-size="11" '
                f'fill="{AURORA["frost_gray"]}">{variant}</text>'
            )
            for pooling_index, pooling in enumerate(ATTENTION_POOLINGS):
                row = next(
                    item
                    for item in rows
                    if item["model"] == model
                    and item["variant"] == variant
                    and item["pooling"] == pooling
                )
                x = center + (pooling_index - 1.0) * 29 - bar_width / 2
                y = y_position(row["effective_number"])
                color = POOLING_COLORS[pooling]
                parts.extend(
                    [
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bottom-y:.1f}" '
                        f'fill="{color}" opacity=".88"/>',
                        f'<text x="{x+bar_width/2:.1f}" y="{y-7:.1f}" text-anchor="middle" '
                        f'font-size="10" fill="{AURORA["night_black"]}">{float(row["effective_number"]):.1f}</text>',
                    ]
                )
        parts.extend(
            [
                f'<text x="{left+plot_width/2:.1f}" y="399" text-anchor="middle" font-size="12">V4 panel</text>',
                f'<text transform="translate({left-50} {(top+bottom)/2:.1f}) rotate(-90)" '
                'text-anchor="middle" font-size="12">effective number N_eff (max 10)</text>',
            ]
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _partition_phenotype_svg(summary: dict[str, Any]) -> str:
    width, height = 900, 475
    left, top, bottom, plot_width = 105, 70, 335, 690
    totals = summary["totals"]
    y_max = max(totals.values()) * 1.05
    order = (
        "global_endpoint_aggregator",
        "partition_local_endpoint_aggregator",
        "occurrence_endpoint_selector",
        "other",
    )
    labels = {
        "global_endpoint_aggregator": "global aggregator",
        "partition_local_endpoint_aggregator": "partition-local",
        "occurrence_endpoint_selector": "selector",
        "other": "other phenotypes",
    }

    def y_position(value: float) -> float:
        return bottom - float(value) / y_max * (bottom - top)

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="phenotype-title phenotype-desc">',
        '<title id="phenotype-title">Qwen span-end attention head phenotypes</title>',
        '<desc id="phenotype-desc">Each stacked bar counts Qwen discovery-eligible span-end heads. Green is global aggregation, cyan is partition-local aggregation, pink is an occurrence selector, and gray is every other phenotype. The compact G, L, S, O line below each bar reports every segment count, including segments too small to label inside the bar.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    legend_x = 125
    for index, phenotype in enumerate(order):
        x = legend_x + index * 180
        color = PHENOTYPE_COLORS[phenotype]
        parts.extend(
            [
                f'<rect x="{x}" y="22" width="14" height="14" fill="{color}"/>',
                f'<text x="{x+21}" y="34" font-size="11">{labels[phenotype]}</text>',
            ]
        )
    for tick in range(0, 251, 50):
        y = y_position(tick)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" '
                f'stroke="{AURORA["frost_gray"]}" opacity=".27"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                f'fill="{AURORA["frost_gray"]}">{tick}</text>',
            ]
        )
    group_width = plot_width / len(VARIANTS)
    for index, variant in enumerate(VARIANTS):
        x = left + index * group_width + 38
        bar_width = group_width - 76
        cumulative = 0
        key_total = 0
        for phenotype in order[:-1]:
            key_total += int(summary["counts"][(variant, phenotype)])
        values = {
            phenotype: int(summary["counts"][(variant, phenotype)])
            for phenotype in order[:-1]
        }
        values["other"] = int(totals[variant]) - key_total
        for phenotype in order:
            value = values[phenotype]
            y_top = y_position(cumulative + value)
            y_bottom = y_position(cumulative)
            parts.append(
                f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_width:.1f}" '
                f'height="{y_bottom-y_top:.1f}" fill="{PHENOTYPE_COLORS[phenotype]}"/>'
            )
            if value >= 18:
                parts.append(
                    f'<text x="{x+bar_width/2:.1f}" y="{(y_top+y_bottom)/2+4:.1f}" '
                    f'text-anchor="middle" font-size="11" font-weight="700" '
                    f'fill="{AURORA["night_black"]}">{value}</text>'
                )
            cumulative += value
        parts.extend(
            [
                f'<text x="{x+bar_width/2:.1f}" y="{y_position(totals[variant])-8:.1f}" '
                f'text-anchor="middle" font-size="11">n={totals[variant]}</text>',
                f'<text x="{x+bar_width/2:.1f}" y="360" text-anchor="middle" font-size="12">{variant}</text>',
                f'<text x="{x+bar_width/2:.1f}" y="386" text-anchor="middle" font-size="10" fill="{AURORA["night_black"]}">G {values["global_endpoint_aggregator"]} · L {values["partition_local_endpoint_aggregator"]} · S {values["occurrence_endpoint_selector"]} · O {values["other"]}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{left+plot_width/2:.1f}" y="448" text-anchor="middle" font-size="12">V4 panel</text>',
            f'<text transform="translate(35 {(top+bottom)/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12">discovery-eligible head count</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts)


def _attention_conclusion_html(
    top_rows: list[dict[str, Any]], partition: dict[str, Any]
) -> str:
    qwen_end = [
        row
        for row in top_rows
        if row["model"] == "Qwen3-8B" and row["pooling"] == "span_end"
    ]
    qwen_mean = [
        row
        for row in top_rows
        if row["model"] == "Qwen3-8B" and row["pooling"] == "span_mean"
    ]
    qwen_sum = [
        row
        for row in top_rows
        if row["model"] == "Qwen3-8B" and row["pooling"] == "span_sum"
    ]
    gemma_end = [
        row
        for row in top_rows
        if row["model"] == "Gemma4-E4B" and row["pooling"] == "span_end"
    ]
    gemma_mean = [
        row
        for row in top_rows
        if row["model"] == "Gemma4-E4B" and row["pooling"] == "span_mean"
    ]
    gemma_sum = [
        row
        for row in top_rows
        if row["model"] == "Gemma4-E4B" and row["pooling"] == "span_sum"
    ]
    assessment = partition["assessment"]["assessments"]
    first_share = float(np.mean([row["endpoint_first_occurrence_share"] for row in assessment]))
    global_counts = [
        partition["counts"][(variant, "global_endpoint_aggregator")]
        for variant in VARIANTS
    ]
    local_counts = [
        partition["counts"][(variant, "partition_local_endpoint_aggregator")]
        for variant in VARIANTS
    ]
    return (
        '<div class="section-conclusion"><span>本节结论</span><p>'
        f"Qwen 的 rank-1 span-end head 在四个 panel 的平均 N_eff={np.mean([row['effective_number'] for row in qwen_end]):.2f}，"
        f"约 {100*first_share:.1f}% 的 endpoint share 都落在第一个 occurrence；它是 selector，不是 broad aggregator。"
        f"作为完整 span 敏感性对照，Qwen 的 span-mean/span-sum rank-1 平均 N_eff 分别为 "
        f"{np.mean([row['effective_number'] for row in qwen_mean]):.2f}/{np.mean([row['effective_number'] for row in qwen_sum]):.2f}；"
        f"Gemma 的 span-end/span-mean/span-sum 分别为 "
        f"{np.mean([row['effective_number'] for row in gemma_end]):.2f}/"
        f"{np.mean([row['effective_number'] for row in gemma_mean]):.2f}/"
        f"{np.mean([row['effective_number'] for row in gemma_sum]):.2f}。"
        "这些 rank-1 数字来自各 pooling 各自选出的 head，不能解释为同一 head 的 paired 变化；固定-head 的 end–mean/end–sum 对齐另见 3.1 表。"
        f"Qwen 全候选分析仍找到每个 panel {min(global_counts)}–{max(global_counts)} 个 global aggregators，且有 {len(partition['stable_global_heads'])} 个在全部 panel×split cells 中保持该 phenotype；partition-local heads 从 v4.1 的 {local_counts[0]} 个降到其余 panel 的 {min(local_counts[1:])} 个。"
        "因此 broad aggregation 是多 head 分布式机制，不能用最高排名的单个 head 代表；固定、seed-invariant 的分区电路目前证据不足。</p></div>"
    )


def _table_behavior_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['model'])}</td>"
            f"<td>{html.escape(row['variant'])}</td>"
            f"<td>{html.escape(row['split'])}</td>"
            f"<td>{row['correct']}/{row['n']}</td>"
            f"<td>{_number(row['accuracy'], 2)}</td>"
            f"<td>{_number(row['mean_prediction'], 2)}</td>"
            f"<td>{_number(row['mae'], 2)}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _table_sensitivity_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td><code>{html.escape(str(row['pooling']))}</code></td>"
            f"<td>L{int(row['primary_layer'])}</td>"
            f"<td>{html.escape(str(row['metric']))}</td>"
            f"<td>{html.escape(str(row['left_variant']))} → {html.escape(str(row['right_variant']))}</td>"
            f"<td>{_number(row['delta_mean'], signed=True)}</td>"
            f"<td>[{_number(row['ci95_low'])}, {_number(row['ci95_high'])}]</td>"
            f"<td>{'yes' if _bool(row['increase_ci_excludes_zero']) else 'no'}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _table_answer_query_pca_sensitivity_html(
    rows: list[dict[str, Any]],
) -> str:
    body: list[str] = []
    for row in rows:
        correlation = row["centroid_distance_corr_to_all"]
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>L{int(row['layer'])}</td>"
            f"<td><code>{html.escape(str(row['fit_cohort']))}</code></td>"
            f"<td>{int(row['fit_rows'])}</td>"
            f"<td>{int(row['fit_count_support_min'])}–{int(row['fit_count_support_max'])}</td>"
            f"<td>{_number(row['fit_evr_pc1_3'])}</td>"
            f"<td>{_number(row['fit_evr_pc1_6'])}</td>"
            f"<td>{_number(row['common_v41_capture_pc1_3'])}</td>"
            f"<td>{_number(row['common_v41_capture_pc1_6'])}</td>"
            f"<td>{_number(row['step_cv'])}</td>"
            f"<td>{_number(row['path_chord'])}</td>"
            f"<td>{_number(row['within_count_seed_rms'])}</td>"
            f"<td>{_number(row['between_count_centroid_rms'])}</td>"
            f"<td>{_number(row['seed_noise_to_count_signal'])}</td>"
            f"<td>{'NA' if not np.isfinite(correlation) else _number(correlation)}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _answer_query_pca_conclusion_html(rows: list[dict[str, Any]]) -> str:
    """Summarize the all-fit versus correct-only answer-query PCA audit."""

    summaries: list[str] = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        by_layer = {
            int(layer): {
                str(row["fit_cohort"]): row
                for row in model_rows
                if int(row["layer"]) == int(layer)
            }
            for layer in sorted({int(row["layer"]) for row in model_rows})
        }
        correct_rows = [
            pair["correct_only"]
            for pair in by_layer.values()
            if "correct_only" in pair
        ]
        if not correct_rows or any(
            set(pair) != {"all", "correct_only"} for pair in by_layer.values()
        ):
            raise RuntimeError(f"{model}: incomplete answer-query PCA sensitivity rows")
        layer_bits: list[str] = []
        for layer, pair in by_layer.items():
            all_fit = pair["all"]
            correct_fit = pair["correct_only"]
            layer_bits.append(
                f"L{layer}: common PC1–3 capture "
                f"{float(all_fit['common_v41_capture_pc1_3']):.3f}→"
                f"{float(correct_fit['common_v41_capture_pc1_3']):.3f}, "
                f"noise/signal {float(all_fit['seed_noise_to_count_signal']):.3f}→"
                f"{float(correct_fit['seed_noise_to_count_signal']):.3f}, "
                f"centroid-distance r={float(correct_fit['centroid_distance_corr_to_all']):.3f}"
            )
        representative = correct_rows[0]
        summaries.append(
            f"<strong>{html.escape(model)}</strong> 的 correct-only fit 使用 "
            f"{int(representative['fit_rows'])} 条 V4.1 discovery prompts，"
            f"每个 count 支持范围为 {int(representative['fit_count_support_min'])}–"
            f"{int(representative['fit_count_support_max'])}；"
            + "；".join(layer_bits)
            + "。"
        )
    return (
        '<div class="section-conclusion"><span>2.3 敏感性结论 · 正确样本并未被偷偷等同于全部样本</span><p>'
        + " ".join(summaries)
        + "这里的 capture 与 scatter 都在同一批 all-V4.1 evaluation states 上计算，"
        "因此变化来自 PCA fit cohort，而不是换了评估点。若 correct-only 的某个 count 支持为 0，"
        "该 basis 对该 count 只能解释为外推；PCA 结果仍是 representation sensitivity，不是正确性的因果检验。</p></div>"
    )


def _table_span_end_alignment_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td>{int(row['prompts'])} / {int(row['seeds'])}</td>"
            f"<td>{_number(row['mean_k'], 2)}</td>"
            f"<td>{_number(row['overlap'])}</td>"
            f"<td>{_number(row['chance'])}</td>"
            f"<td>{_number(row['delta'], signed=True)} "
            f"[{_number(row['delta_low'])}, {_number(row['delta_high'])}]</td>"
            f"<td>{_p_value(row['p_holm'])}</td>"
            f"<td>{_number(row['exact'])} / {_number(row['exact_chance'])}</td>"
            f"<td>{_number(row['exact_delta'], signed=True)} "
            f"[{_number(row['exact_delta_low'])}, {_number(row['exact_delta_high'])}]</td>"
            f"<td>{_number(row['tail_prefix_ratio'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _table_span_end_pooled_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{int(row['seeds'])}</td>"
            f"<td>{_number(row['overlap'])} / {_number(row['chance'])}</td>"
            f"<td>{_number(row['delta'], signed=True)} "
            f"[{_number(row['delta_low'])}, {_number(row['delta_high'])}]</td>"
            f"<td>{_p_value(row['p_holm'])}</td>"
            f"<td>{_number(row['exact'])} / {_number(row['exact_chance'])}</td>"
            f"<td>{_number(row['exact_delta'], signed=True)} "
            f"[{_number(row['exact_delta_low'])}, {_number(row['exact_delta_high'])}]</td>"
            f"<td>{_number(row['tail_prefix_ratio'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _table_attention_omission_pooling_sensitivity_html(
    rows: list[dict[str, Any]],
) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td><code>{html.escape(str(row['pooling']))}</code></td>"
            f"<td>{int(row['prompts'])} / {int(row['seeds'])}</td>"
            f"<td>{_number(row['overlap'])}</td>"
            f"<td>{_number(row['chance'])}</td>"
            f"<td>{_number(row['delta'], signed=True)} [{_number(row['delta_low'])}, {_number(row['delta_high'])}]</td>"
            f"<td>{_number(row['exact'])} / {_number(row['exact_chance'])}</td>"
            f"<td>{_number(row['tail_prefix_ratio'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _attention_omission_pooling_conclusion_html(
    rows: list[dict[str, Any]],
) -> str:
    summaries: list[str] = []
    for model in MODELS:
        selected = {
            str(row["pooling"]): row for row in rows if row["model"] == model
        }
        if set(selected) != {"span_end", "span_sum"}:
            raise RuntimeError(f"{model}: incomplete end-versus-sum omission audit")
        endpoint = selected["span_end"]
        full_span = selected["span_sum"]
        if int(endpoint["prompts"]) != int(full_span["prompts"]):
            raise RuntimeError(f"{model}: end/sum omission prompt sets differ")
        summaries.append(
            f"<strong>{html.escape(model)}</strong> 在同一 {int(endpoint['prompts'])} 条 undercount prompts 上，"
            f"overlap−chance 为 {_number(endpoint['delta'], signed=True)}→"
            f"{_number(full_span['delta'], signed=True)}，tail/prefix 为 "
            f"{_number(endpoint['tail_prefix_ratio'])}→"
            f"{_number(full_span['tail_prefix_ratio'])}（span-end→span-sum）"
        )
    return (
        '<div class="section-conclusion"><span>3.6b pooling 敏感性结论 · Endpoint 是否代表完整 span</span><p>'
        + "；".join(summaries)
        + "。两种 pooling 使用各自 discovery-ranked top-8 bank，所以这些箭头是同提示集合上的描述性敏感性比较，"
        "不是固定-head或 end−sum 的随机化显著性检验。只有当方向和量级都接近时，才把 endpoint omission 当作完整 span omission 的稳健代理。</p></div>"
    )


def _table_span_end_nested_html(rows: list[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{int(row['failed_n'])} / {int(row['registered_n'])}</td>"
            f"<td>{int(row['paired_blocks'])} / {int(row['paired_seeds'])}</td>"
            f"<td>{_number(row['failed_bottom'])} / "
            f"{_number(row['registered_bottom'])}</td>"
            f"<td>{_number(row['bottom_difference'], signed=True)} "
            f"[{_number(row['bottom_difference_low'])}, "
            f"{_number(row['bottom_difference_high'])}]</td>"
            f"<td>{_p_value(row['bottom_p_holm'])}</td>"
            f"<td>{_number(row['failed_share'])} / "
            f"{_number(row['registered_share'])}</td>"
            f"<td>{_number(row['share_difference'], signed=True)} "
            f"[{_number(row['share_difference_low'])}, "
            f"{_number(row['share_difference_high'])}]</td>"
            f"<td>{_p_value(row['share_p_holm'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def _span_end_alignment_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1040, 390
    panel_lefts = [70, 570]
    panel_width = 390
    x_max = 0.70

    def x_position(value: float, panel_left: float) -> float:
        return panel_left + max(0.0, min(x_max, value)) / x_max * panel_width

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="tail-plot-title tail-plot-desc">',
        '<title id="tail-plot-title">Span-end tail and bottom-k overlap versus chance</title>',
        '<desc id="tail-plot-desc">Observed overlap is above the hypergeometric chance baseline in most model and variant panels.</desc>',
        f'<rect width="1040" height="390" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for panel_index, model in enumerate(MODELS):
        left = panel_lefts[panel_index]
        parts.append(
            f'<text x="{left}" y="34" font-size="17" font-weight="700">'
            f"{html.escape(model)}</text>"
        )
        for tick in np.arange(0.0, x_max + 0.001, 0.1):
            x = x_position(float(tick), left)
            parts.append(
                f'<line x1="{x:.1f}" y1="55" x2="{x:.1f}" y2="315" '
                f'stroke="{AURORA["frost_gray"]}" stroke-width="1" opacity=".28"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="338" text-anchor="middle" '
                f'font-size="11" fill="{AURORA["frost_gray"]}">{tick:.1f}</text>'
            )
        model_rows = [row for row in rows if row["model"] == model]
        for row_index, row in enumerate(model_rows):
            y = 88 + row_index * 58
            chance_x = x_position(float(row["chance"]), left)
            observed_x = x_position(float(row["overlap"]), left)
            ci_low_x = x_position(float(row["chance"]) + float(row["delta_low"]), left)
            ci_high_x = x_position(
                float(row["chance"]) + float(row["delta_high"]), left
            )
            parts.extend(
                [
                    f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" '
                    f'font-size="12" font-weight="650">{html.escape(str(row["variant"]))}</text>',
                    f'<line x1="{chance_x:.1f}" y1="{y}" x2="{observed_x:.1f}" '
                    f'y2="{y}" stroke="{AURORA["frost_gray"]}" stroke-width="2"/>',
                    f'<line x1="{ci_low_x:.1f}" y1="{y}" x2="{ci_high_x:.1f}" '
                    f'y2="{y}" stroke="{MODEL_COLORS[model]}" stroke-width="5" stroke-linecap="round" opacity=".42"/>',
                    f'<path d="M {chance_x:.1f} {y - 6} L {chance_x + 6:.1f} {y} '
                    f'L {chance_x:.1f} {y + 6} L {chance_x - 6:.1f} {y} Z" fill="{AURORA["aurora_yellow"]}"/>',
                    f'<circle cx="{observed_x:.1f}" cy="{y}" r="6" fill="{MODEL_COLORS[model]}" stroke="{AURORA["snow_white"]}" stroke-width="1.5"/>',
                    f'<text x="{left + panel_width + 9}" y="{y + 4}" font-size="11" fill="{AURORA["night_black"]}">'
                    f'Δ {_number(row["delta"], signed=True)}</text>',
                ]
            )
        parts.append(
            f'<text x="{left + panel_width / 2:.1f}" y="365" text-anchor="middle" '
            f'font-size="12" fill="{AURORA["night_black"]}">tail overlap fraction</text>'
        )
    parts.extend(
        [
            f'<path d="M 761 28 L 767 34 L 761 40 L 755 34 Z" fill="{AURORA["aurora_yellow"]}"/>',
            f'<text x="774" y="38" font-size="11" fill="{AURORA["frost_gray"]}">hypergeometric chance</text>',
            f'<circle cx="910" cy="34" r="6" fill="{AURORA["polar_violet"]}"/>',
            f'<text x="920" y="38" font-size="11" fill="{AURORA["frost_gray"]}">observed (95% seed CI)</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts)


def _span_end_nested_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 920, 290
    left, plot_width, x_max = 180, 570, 0.70

    def x_position(value: float) -> float:
        return left + max(0.0, min(x_max, value)) / x_max * plot_width

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="nested-plot-title nested-plot-desc">',
        '<title id="nested-plot-title">New needle bottom-k attention risk by increment status</title>',
        '<desc id="nested-plot-desc">The newly introduced needle is more often in the lowest-attention set when the output fails to increment.</desc>',
        f'<rect width="920" height="290" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    for tick in np.arange(0.0, x_max + 0.001, 0.1):
        x = x_position(float(tick))
        parts.append(
            f'<line x1="{x:.1f}" y1="48" x2="{x:.1f}" y2="222" '
            f'stroke="{AURORA["frost_gray"]}" stroke-width="1" opacity=".28"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="244" text-anchor="middle" font-size="11" '
            f'fill="{AURORA["frost_gray"]}">{tick:.1f}</text>'
        )
    for row_index, row in enumerate(rows):
        center_y = 92 + row_index * 88
        parts.append(
            f'<text x="{left - 18}" y="{center_y + 4}" text-anchor="end" '
            f'font-size="14" font-weight="700">{html.escape(str(row["model"]))}</text>'
        )
        for status, color, offset in (
            ("failed", AURORA["sunset_pink"], -12),
            ("registered", AURORA["aurora_green"], 12),
        ):
            value = float(row[f"{status}_bottom"])
            low = float(row[f"{status}_bottom_low"])
            high = float(row[f"{status}_bottom_high"])
            y = center_y + offset
            parts.extend(
                [
                    f'<line x1="{x_position(low):.1f}" y1="{y}" '
                    f'x2="{x_position(high):.1f}" y2="{y}" stroke="{color}" '
                    'stroke-width="4" stroke-linecap="round" opacity=".42"/>',
                    f'<circle cx="{x_position(value):.1f}" cy="{y}" r="6" '
                    f'fill="{color}" stroke="{AURORA["snow_white"]}" stroke-width="1.5"/>',
                ]
            )
        parts.append(
            f'<text x="770" y="{center_y + 4}" font-size="11" fill="{AURORA["night_black"]}">'
            f'RD {_number(row["bottom_difference"], signed=True)} '
            f'[{_number(row["bottom_difference_low"])}, {_number(row["bottom_difference_high"])}]'
            "</text>"
        )
    parts.extend(
        [
            f'<circle cx="590" cy="25" r="6" fill="{AURORA["sunset_pink"]}"/><text x="601" y="29" font-size="11" fill="{AURORA["frost_gray"]}">failed to increment</text>',
            f'<circle cx="720" cy="25" r="6" fill="{AURORA["aurora_green"]}"/><text x="731" y="29" font-size="11" fill="{AURORA["frost_gray"]}">registered +1</text>',
            f'<text x="465" y="273" text-anchor="middle" font-size="12" fill="{AURORA["night_black"]}">P(new needle is in current bottom-k attention set)</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts)


def _variant_list(values: list[str]) -> str:
    if not values:
        return "无"
    if len(values) == 1:
        return values[0]
    return "、".join(values)


def _span_end_conclusion_html(
    pooled_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    nested_rows: list[dict[str, Any]],
) -> str:
    cards: list[str] = []
    for model in MODELS:
        selected = [row for row in alignment_rows if row["model"] == model]
        pooled = next(row for row in pooled_rows if row["model"] == model)
        positive_ci = [
            str(row["variant"]) for row in selected if float(row["delta_low"]) > 0
        ]
        holm = [str(row["variant"]) for row in selected if float(row["p_holm"]) < 0.05]
        cards.append(
            '<div class="note"><strong>'
            + html.escape(model)
            + " 尾部对齐。</strong><p>四个 panel 等权 pooling 后，tail-alignment contrast 为 "
            + _number(pooled["delta"], signed=True)
            + " ["
            + _number(pooled["delta_low"])
            + ", "
            + _number(pooled["delta_high"])
            + "]，Holm p="
            + _p_value(pooled["p_holm"])
            + "。四个 panel 的点估计都高于机会水平；95% seed-cluster 区间排除 0 的 panel 为 "
            + html.escape(_variant_list(positive_ci))
            + "；exact sign-flip 检验经 Holm 校正后仍小于 0.05 的 panel 为 "
            + html.escape(_variant_list(holm))
            + "。</p></div>"
        )
    nested_sentences: list[str] = []
    for row in nested_rows:
        nested_sentences.append(
            f"{html.escape(str(row['model']))}：risk difference "
            f"{_number(row['bottom_difference'], signed=True)} "
            f"[{_number(row['bottom_difference_low'])}, "
            f"{_number(row['bottom_difference_high'])}], "
            f"Holm p={_p_value(row['bottom_p_holm'])}"
        )
    cards.append(
        '<div class="note"><strong>新增 needle 的 exact 配对检查。</strong><p>'
        + "；".join(nested_sentences)
        + "。正值表示：当输出没有随 gold count 增加时，新加入的 needle 更常落入 "
        + "bottom-k attention set。</p></div>"
    )
    return "".join(cards)


def _causal_frames(
    run_root: Path,
) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, Any]],
]:
    designs = find_screen_designs(run_root)
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    paths: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        frames[model] = {}
        paths[model] = {}
        for stage in ("ablation", "patching", "steering"):
            selected = designs[model][stage]
            frames[model][stage] = pd.read_csv(
                selected.root / "detail.csv.gz", compression="gzip"
            )
            paths[model][stage] = selected
        frames[model]["geometry"] = pd.read_csv(
            designs[model]["steering"].root / "centroid_geometry_summary.csv"
        )
    return frames, paths


def _completed_steering_v2_root(
    run_root: Path,
    *,
    model: str,
    phase: str,
) -> Path:
    family_root = (
        run_root
        / model
        / "numeric"
        / "causal"
        / "geometric_steering_v2"
    )
    candidates: list[Path] = []
    for root in sorted(family_root.glob(f"{phase}_*")):
        complete = root / "complete.json"
        if not complete.is_file():
            continue
        payload = _read_json(complete)
        if payload.get("status") == "complete" and payload.get("phase") == phase:
            candidates.append(root)
    if len(candidates) != 1:
        raise RuntimeError(
            f"{model}: expected one completed steering-v2 {phase} root, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _steering_v2_paired_effects(detail: pd.DataFrame) -> pd.DataFrame:
    """Pair geometric and norm-matched controls with invalid rows set to failure."""

    identifiers = [
        "model_label",
        "design_variant",
        "seed",
        "receiver_stimulus_id",
        "target_stimulus_id",
        "receiver_count",
        "target_count",
        "target_direction",
        "steering_protocol",
        "layer_set",
        "alpha",
    ]
    required = set(identifiers) | {
        "condition",
        "patched_format_valid",
        "direction_aligned_generated_count_shift",
        "moved_toward_donor_gold",
        "follows_donor_gold",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise RuntimeError(f"Steering-v2 detail is missing columns: {missing}")
    work = detail.copy()
    valid = work["patched_format_valid"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    aligned = pd.to_numeric(
        work["direction_aligned_generated_count_shift"], errors="coerce"
    )
    work["strict_aligned_shift"] = np.where(valid & aligned.notna(), aligned, 0.0)
    work["strict_moved"] = np.where(
        valid,
        work["moved_toward_donor_gold"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        ),
        False,
    ).astype(float)
    work["strict_target_hit"] = np.where(
        valid,
        work["follows_donor_gold"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        ),
        False,
    ).astype(float)
    work["strict_valid"] = valid.astype(float)
    metrics = (
        "strict_aligned_shift",
        "strict_moved",
        "strict_target_hit",
        "strict_valid",
    )
    geometric = work[work["condition"] == "geometric"][identifiers + list(metrics)]
    if geometric.duplicated(identifiers).any():
        raise RuntimeError("Steering-v2 geometric rows are not unique")
    random = work[work["condition"] == "orthogonal_norm_matched_random"]
    if random.empty:
        raise RuntimeError("Steering-v2 matched random rows are missing")
    random_mean = random.groupby(identifiers, as_index=False)[list(metrics)].mean()
    random_mean = random_mean.rename(
        columns={metric: f"{metric}_random" for metric in metrics}
    )
    paired = geometric.merge(random_mean, on=identifiers, how="inner", validate="1:1")
    if len(paired) != len(geometric):
        raise RuntimeError("Steering-v2 geometric/control pairing is incomplete")
    for metric in metrics:
        paired[f"{metric}_effect"] = paired[metric] - paired[f"{metric}_random"]
    return paired


def _steering_v2_rows(
    run_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Load the discovery lock and analyze held-out single/multi-layer steering."""

    selection_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for model in MODELS:
        screen_root = _completed_steering_v2_root(
            run_root, model=model, phase="screen"
        )
        confirmation_root = _completed_steering_v2_root(
            run_root, model=model, phase="confirmation"
        )
        screen_design = _read_json(screen_root / "design.json")
        confirmation_design = _read_json(confirmation_root / "design.json")
        selection = _read_json(screen_root / "selection.json")
        scores = pd.read_csv(screen_root / "plan_scores.csv")
        summary = pd.read_csv(confirmation_root / "summary.csv")
        detail = pd.read_csv(confirmation_root / "detail.csv.gz", compression="gzip")
        if screen_design.get("evaluation_split") != "discovery":
            raise RuntimeError(f"{model}: steering-v2 screen split changed")
        if confirmation_design.get("evaluation_split") != "confirmation":
            raise RuntimeError(f"{model}: steering-v2 confirmation split changed")
        if tuple(confirmation_design.get("variants", ())) != VARIANTS:
            raise RuntimeError(f"{model}: steering-v2 confirmation panels changed")
        if tuple(confirmation_design.get("seeds", ())) != tuple(range(1254, 1264)):
            raise RuntimeError(f"{model}: steering-v2 confirmation seeds changed")
        if len(detail) != 960 or set(detail["steering_protocol"].astype(str)) != {
            "single_layer",
            "multi_layer",
        }:
            raise RuntimeError(f"{model}: unexpected steering-v2 confirmation detail")
        paired = _steering_v2_paired_effects(detail)
        model_overall_start = len(overall_rows)
        model_panel_start = len(panel_rows)
        for protocol in ("single_layer", "multi_layer"):
            locked = selection["selected"][protocol]
            layer_set = str(locked["layer_set"])
            alpha = float(locked["alpha"])
            score_row = scores[
                (scores["steering_protocol"].astype(str) == protocol)
                & (scores["layer_set"].astype(str) == layer_set)
                & np.isclose(pd.to_numeric(scores["alpha"]), alpha)
            ]
            summary_row = summary[
                (summary["steering_protocol"].astype(str) == protocol)
                & (summary["layer_set"].astype(str) == layer_set)
                & np.isclose(pd.to_numeric(summary["alpha"]), alpha)
            ]
            if len(score_row) != 1 or len(summary_row) != 1:
                raise RuntimeError(f"{model}: locked steering-v2 plan is not unique")
            score_payload = score_row.iloc[0].to_dict()
            summary_payload = summary_row.iloc[0].to_dict()
            selection_rows.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "layer_set": layer_set,
                    "alpha": alpha,
                    "candidate_plans": int(selection["candidate_plan_count"]),
                    "screen_seeds": int(score_payload["screen_seeds"]),
                    "mean_screen_effect": float(
                        score_payload["mean_aligned_shift_effect"]
                    ),
                    "worst_panel_screen_effect": float(
                        score_payload["worst_variant_aligned_shift_effect"]
                    ),
                    "positive_screen_panels": int(
                        score_payload["positive_variant_count"]
                    ),
                    "screen_valid_rate": float(
                        score_payload["geometric_valid_rate"]
                    ),
                    "robust_selection_score": float(
                        score_payload["robust_selection_score"]
                    ),
                }
            )
            selected = paired[
                (paired["steering_protocol"].astype(str) == protocol)
                & (paired["layer_set"].astype(str) == layer_set)
                & np.isclose(pd.to_numeric(paired["alpha"]), alpha)
            ]
            seed_values = selected.groupby("seed", sort=True)[
                "strict_aligned_shift_effect"
            ].mean()
            if len(seed_values) != 10:
                raise RuntimeError(f"{model}/{protocol}: expected ten confirmation seeds")
            estimate, low, high = _seed_bootstrap(
                seed_values.to_numpy(dtype=float),
                label=f"steering-v2-overall:{model}:{protocol}:{layer_set}:{alpha}",
            )
            overall_rows.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "layer_set": layer_set,
                    "alpha": alpha,
                    "paired_rows": int(len(selected)),
                    "seeds": 10,
                    "geometric_valid_rate": float(
                        summary_payload["geometric_valid_rate"]
                    ),
                    "random_valid_rate": float(summary_payload["random_valid_rate"]),
                    "geometric_aligned_shift": float(
                        summary_payload["geometric_mean_aligned_shift"]
                    ),
                    "random_aligned_shift": float(
                        summary_payload["random_mean_aligned_shift"]
                    ),
                    "aligned_effect": estimate,
                    "aligned_effect_low": low,
                    "aligned_effect_high": high,
                    "moved_effect": float(summary_payload["moved_rate_effect"]),
                    "moved_effect_low": float(
                        summary_payload["moved_rate_ci95_low"]
                    ),
                    "moved_effect_high": float(
                        summary_payload["moved_rate_ci95_high"]
                    ),
                    "target_hit_effect": float(
                        summary_payload["target_hit_rate_effect"]
                    ),
                    "p_raw": _exact_sign_flip_p(seed_values.to_numpy(dtype=float)),
                }
            )
            for variant in VARIANTS:
                cell = selected[selected["design_variant"].astype(str) == variant]
                variant_seed = cell.groupby("seed", sort=True)[
                    "strict_aligned_shift_effect"
                ].mean()
                if len(variant_seed) != 10:
                    raise RuntimeError(
                        f"{model}/{protocol}/{variant}: incomplete confirmation seeds"
                    )
                panel_estimate, panel_low, panel_high = _seed_bootstrap(
                    variant_seed.to_numpy(dtype=float),
                    label=(
                        f"steering-v2-panel:{model}:{protocol}:{layer_set}:"
                        f"{alpha}:{variant}"
                    ),
                )
                panel_rows.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "layer_set": layer_set,
                        "alpha": alpha,
                        "variant": variant,
                        "paired_rows": int(len(cell)),
                        "aligned_effect": panel_estimate,
                        "aligned_effect_low": panel_low,
                        "aligned_effect_high": panel_high,
                        "p_raw": _exact_sign_flip_p(
                            variant_seed.to_numpy(dtype=float)
                        ),
                    }
                )
        model_overall = overall_rows[model_overall_start:]
        for row, p_holm in zip(
            model_overall,
            _holm_adjust([float(row["p_raw"]) for row in model_overall]),
        ):
            row["p_holm"] = p_holm
        model_panels = panel_rows[model_panel_start:]
        for row, p_holm in zip(
            model_panels,
            _holm_adjust([float(row["p_raw"]) for row in model_panels]),
        ):
            row["p_holm"] = p_holm
        audit_rows.append(
            {
                "model": model,
                "screen_root": screen_root.name,
                "screen_rows": int(pd.read_csv(screen_root / "detail.csv.gz").shape[0]),
                "screen_shards": len(
                    _read_jsonl(screen_root / "capture" / "capture_index.jsonl")
                ),
                "confirmation_root": confirmation_root.name,
                "confirmation_rows": int(len(detail)),
                "confirmation_shards": len(
                    _read_jsonl(
                        confirmation_root / "capture" / "capture_index.jsonl"
                    )
                ),
                "selection_rule": str(selection["selection_rule"]),
            }
        )
    return selection_rows, overall_rows, panel_rows, audit_rows


def _answer_query_frames(
    run_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    analysis_root = run_root / "analysis" / "answer_query_patching_dense_v1"
    required = (
        "layer_summary",
        "pair_summary",
        "variant_summary",
        "outcome_summary",
        "stratum_summary",
        "invalid_rows",
    )
    frames: dict[str, pd.DataFrame] = {}
    for name in required:
        path = analysis_root / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing answer-query analysis table {path}; run "
                "scripts/analyze_realistic_niah_v4_answer_query_patching.py first"
            )
        frames[name] = pd.read_csv(path)
    return frames, audit_answer_query_patching(run_root)


def _answer_query_final_rows(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    selected: list[pd.DataFrame] = []
    for model in MODELS:
        model_rows = frame[frame["model"] == model]
        selected.append(model_rows[model_rows["layer"] == model_rows["layer"].max()])
    return pd.concat(selected, ignore_index=True).to_dict("records")


def _answer_query_onset_rows(
    layer_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        selected = layer_frame[layer_frame["model"] == model].sort_values("layer")
        candidates = selected[
            (selected["layer"] > selected["layer"].min())
            & (selected["eligible_donor_adoption_rate"] >= 0.5)
            & (selected["eligible_donor_adoption_vs_layer0_p_holm"] < 0.05)
        ]
        if candidates.empty:
            raise RuntimeError(f"No significant answer-query transport onset for {model}")
        rows.append(candidates.iloc[0].to_dict())
    return rows


def _native_open_path(path: Path) -> str | Path:
    """Return a Windows extended-length path when an artifact path is long."""

    resolved = path.resolve()
    rendered = str(resolved)
    if sys.platform == "win32" and len(rendered) >= 240:
        return "\\\\?\\" + rendered
    return resolved


def _all_generation_labels(model_root: Path) -> pd.DataFrame:
    """Load the complete 4 panels x 30 seeds x 10 counts behavior grid."""
    path = model_root / "behavior" / "capture" / "generation_labels.csv"
    labels = pd.read_csv(path)
    expected_rows = len(VARIANTS) * 30 * 10
    if len(labels) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} behavior rows in {path}")
    if labels.duplicated(["design_variant", "seed", "gold_count"]).any():
        raise RuntimeError(f"Behavior rows are not unique in {path}")
    labels = labels.copy()
    for field in ("gold_count", "parsed_count", "count_error"):
        labels[field] = pd.to_numeric(labels[field], errors="coerce")
    labels["is_correct_bool"] = labels["is_correct"].map(_bool)
    labels["format_valid_bool"] = labels["format_valid"].map(_bool)
    return labels


def _behavior_panel_rows(
    labels_by_model: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, labels in labels_by_model.items():
        selected = labels[labels["split"] == "confirmation"].copy()
        for variant, frame in selected.groupby("design_variant", sort=True):
            error = pd.to_numeric(frame["count_error"], errors="coerce")
            prediction = pd.to_numeric(frame["parsed_count"], errors="coerce")
            rows.append(
                {
                    "model": model,
                    "variant": str(variant),
                    "n": int(len(frame)),
                    "correct": int(frame["is_correct_bool"].sum()),
                    "accuracy": float(frame["is_correct_bool"].mean()),
                    "format_valid": float(frame["format_valid_bool"].mean()),
                    "mean_prediction": float(prediction.mean()),
                    "mae": float(error.abs().mean()),
                    "undercount_rate": float((error < 0).mean()),
                    "overcount_rate": float((error > 0).mean()),
                }
            )
    return rows


def _behavior_count_rows(
    labels_by_model: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, labels in labels_by_model.items():
        selected = labels[labels["split"] == "confirmation"].copy()
        for (variant, count), frame in selected.groupby(
            ["design_variant", "gold_count"], sort=True
        ):
            error = pd.to_numeric(frame["count_error"], errors="coerce")
            rows.append(
                {
                    "model": model,
                    "variant": str(variant),
                    "count": int(count),
                    "n": int(len(frame)),
                    "accuracy": float(frame["is_correct_bool"].mean()),
                    "undercount_rate": float((error < 0).mean()),
                    "mean_prediction": float(
                        pd.to_numeric(frame["parsed_count"], errors="coerce").mean()
                    ),
                }
            )
    return rows


def _behavior_count_pooled_rows(
    labels_by_model: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, labels in labels_by_model.items():
        selected = labels[labels["split"] == "confirmation"].copy()
        for count, frame in selected.groupby("gold_count", sort=True):
            error = pd.to_numeric(frame["count_error"], errors="coerce")
            rows.append(
                {
                    "model": model,
                    "count": int(count),
                    "n": int(len(frame)),
                    "correct": int(frame["is_correct_bool"].sum()),
                    "accuracy": float(frame["is_correct_bool"].mean()),
                    "mean_prediction": float(
                        pd.to_numeric(frame["parsed_count"], errors="coerce").mean()
                    ),
                    "undercount_rate": float((error < 0).mean()),
                }
            )
    return rows


def _table_behavior_panel_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td>{int(row['correct'])}/{int(row['n'])}</td>"
            f"<td>{100 * float(row['accuracy']):.1f}%</td>"
            f"<td>{_number(row['mean_prediction'], 2)}</td>"
            f"<td>{_number(row['mae'], 2)}</td>"
            f"<td>{100 * float(row['undercount_rate']):.1f}%</td>"
            f"<td>{100 * float(row['format_valid']):.1f}%</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_behavior_count_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{int(row['count'])}</td>"
            f"<td>{int(row['correct'])}/{int(row['n'])}</td>"
            f"<td>{100 * float(row['accuracy']):.1f}%</td>"
            f"<td>{_number(row['mean_prediction'], 2)}</td>"
            f"<td>{100 * float(row['undercount_rate']):.1f}%</td>"
            "</tr>"
        )
    return "".join(rendered)


def _behavior_accuracy_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 1080, 450
    panel_lefts = (74, 604)
    plot_width, top, bottom = 390, 76, 350

    def x_position(count: int, left: float) -> float:
        return left + (int(count) - 1) / 9 * plot_width

    def y_position(value: float) -> float:
        return bottom - max(0.0, min(1.0, float(value))) * (bottom - top)

    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="behavior-plot-title behavior-plot-desc">',
        '<title id="behavior-plot-title">Confirmation accuracy by true count and V4 panel</title>',
        '<desc id="behavior-plot-desc">Accuracy falls sharply at medium and high counts in both models; each line is one controlled-relaxation panel.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
    ]
    legend_x = 292
    for index, variant in enumerate(VARIANTS):
        x = legend_x + index * 125
        color = VARIANT_COLORS[variant]
        parts.extend(
            [
                f'<line x1="{x}" y1="30" x2="{x+24}" y2="30" stroke="{color}" stroke-width="4"/>',
                f'<circle cx="{x+12}" cy="30" r="4" fill="{color}"/>',
                f'<text x="{x+31}" y="34" font-size="12">{variant}</text>',
            ]
        )
    for panel_index, model in enumerate(MODELS):
        left = panel_lefts[panel_index]
        high_count_x = x_position(5, left) - 18
        parts.append(
            f'<rect x="{high_count_x:.1f}" y="{top}" width="{left + plot_width - high_count_x:.1f}" '
            f'height="{bottom-top}" fill="{AURORA["aurora_yellow"]}" opacity=".12"/>'
        )
        parts.append(
            f'<text x="{left}" y="58" font-size="17" font-weight="700">{html.escape(model)}</text>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y_position(tick)
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" '
                    f'stroke="{AURORA["frost_gray"]}" stroke-width="1" opacity=".28"/>',
                    f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                    f'fill="{AURORA["frost_gray"]}">{tick:.2f}</text>',
                ]
            )
        for count in range(1, 11):
            x = x_position(count, left)
            parts.append(
                f'<text x="{x:.1f}" y="371" text-anchor="middle" font-size="11" '
                f'fill="{AURORA["frost_gray"]}">{count}</text>'
            )
        for variant in VARIANTS:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model"] == model and row["variant"] == variant
                ],
                key=lambda row: int(row["count"]),
            )
            points = [
                (x_position(int(row["count"]), left), y_position(row["accuracy"]))
                for row in selected
            ]
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
                for index, (x, y) in enumerate(points)
            )
            color = VARIANT_COLORS[variant]
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for x, y in points:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{color}" '
                    f'stroke="{AURORA["snow_white"]}" stroke-width="1.2"/>'
                )
        parts.extend(
            [
                f'<text x="{left + plot_width/2:.1f}" y="414" text-anchor="middle" font-size="12">true needle count N</text>',
                f'<text transform="translate({left-53} {(top+bottom)/2:.1f}) rotate(-90)" '
                'text-anchor="middle" font-size="12">greedy exact-match accuracy</text>',
            ]
        )
    parts.append("</g></svg>")
    return "".join(parts)


def _behavior_conclusion_html(
    panel_rows: list[dict[str, Any]], pooled_rows: list[dict[str, Any]]
) -> str:
    sentences: list[str] = []
    for model in MODELS:
        model_panels = [row for row in panel_rows if row["model"] == model]
        model_counts = [row for row in pooled_rows if row["model"] == model]
        first_below_half = next(
            int(row["count"])
            for row in sorted(model_counts, key=lambda item: int(item["count"]))
            if float(row["accuracy"]) < 0.5
        )
        sentences.append(
            f"{html.escape(model)} 的 confirmation panel accuracy 为 "
            + "/".join(f"{100*float(row['accuracy']):.0f}%" for row in model_panels)
            + f"（v4.1→v4.4），首次低于 50% 出现在 N={first_below_half}"
        )
    return (
        '<div class="section-conclusion"><span>本节结论</span><p>'
        + "；".join(sentences)
        + "。主要行为边界由 count 大小而不是 V4 panel 决定；高 count 的错误几乎都是 undercount，因此后续机制分析应解释为什么证据未被完整聚合，而不能只比较总体 accuracy。</p></div>"
    )


def _paired_seed_contrast(
    frame: pd.DataFrame,
    *,
    metric: str,
    condition_column: str,
    treatment: str,
    control: str,
    identity_columns: list[str],
    label: str,
) -> dict[str, Any]:
    pivot = frame.pivot(
        index=identity_columns,
        columns=condition_column,
        values=metric,
    )
    missing = sorted({treatment, control} - set(pivot.columns))
    if missing or pivot[[treatment, control]].isna().any().any():
        raise RuntimeError(f"{label}: incomplete paired conditions {missing}")
    differences = (pivot[treatment] - pivot[control]).rename("difference").reset_index()
    seed_values = differences.groupby("seed", sort=True)["difference"].mean().to_numpy()
    if len(seed_values) != 10:
        raise RuntimeError(f"{label}: expected ten paired confirmation seeds")
    estimate, low, high = _seed_bootstrap(seed_values, label=label)
    return {
        "estimate": estimate,
        "low": low,
        "high": high,
        "p_raw": _exact_sign_flip_p(seed_values),
        "seed_values": seed_values,
    }


def _one_sample_seed_estimate(
    frame: pd.DataFrame,
    *,
    metric: str,
    label: str,
) -> dict[str, Any]:
    seed_values = frame.groupby("seed", sort=True)[metric].mean().to_numpy()
    if len(seed_values) != 10:
        raise RuntimeError(f"{label}: expected ten confirmation seeds")
    estimate, low, high = _seed_bootstrap(seed_values, label=label)
    return {
        "estimate": estimate,
        "low": low,
        "high": high,
        "p_raw": _exact_sign_flip_p(seed_values),
        "seed_values": seed_values,
    }


def _causal_ablation_rows(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        detail = frames[model]["ablation"].copy()
        detail["prediction_changed_numeric"] = detail["prediction_changed"].astype(
            float
        )
        for top_n in (4, 8):
            selected = detail[detail["top_n"].astype(int) == top_n].copy()
            identity = ["design_variant", "seed", "stimulus_id", "top_n"]
            changed = _paired_seed_contrast(
                selected,
                metric="prediction_changed_numeric",
                condition_column="condition",
                treatment="ranked",
                control="layer_matched_random",
                identity_columns=identity,
                label=f"causal-ablation-changed-{model}-top{top_n}",
            )
            count_shift = _paired_seed_contrast(
                selected,
                metric="generated_count_shift",
                condition_column="condition",
                treatment="ranked",
                control="layer_matched_random",
                identity_columns=identity,
                label=f"causal-ablation-shift-{model}-top{top_n}",
            )
            error = _paired_seed_contrast(
                selected,
                metric="absolute_error_delta",
                condition_column="condition",
                treatment="ranked",
                control="layer_matched_random",
                identity_columns=identity,
                label=f"causal-ablation-error-{model}-top{top_n}",
            )
            ranked = selected[selected["condition"] == "ranked"]
            control = selected[selected["condition"] == "layer_matched_random"]
            baseline = selected.drop_duplicates("stimulus_id")
            rows.append(
                {
                    "model": model,
                    "top_n": top_n,
                    "prompts": int(baseline["stimulus_id"].nunique()),
                    "baseline_correct": int(
                        baseline["baseline_is_correct"].astype(bool).sum()
                    ),
                    "ranked_changed": float(
                        ranked["prediction_changed_numeric"].mean()
                    ),
                    "random_changed": float(
                        control["prediction_changed_numeric"].mean()
                    ),
                    "changed_difference": changed["estimate"],
                    "changed_difference_low": changed["low"],
                    "changed_difference_high": changed["high"],
                    "ranked_count_shift": float(ranked["generated_count_shift"].mean()),
                    "random_count_shift": float(
                        control["generated_count_shift"].mean()
                    ),
                    "count_shift_difference": count_shift["estimate"],
                    "count_shift_difference_low": count_shift["low"],
                    "count_shift_difference_high": count_shift["high"],
                    "count_shift_p_raw": count_shift["p_raw"],
                    "error_difference": error["estimate"],
                    "error_difference_low": error["low"],
                    "error_difference_high": error["high"],
                }
            )
    adjusted = _holm_adjust([float(row["count_shift_p_raw"]) for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["count_shift_p_holm"] = p_holm
    return rows


def _causal_patching_rows(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        detail = frames[model]["patching"].copy()
        detail["prediction_changed_numeric"] = detail["prediction_changed"].astype(
            float
        )
        detail["moved_numeric"] = detail["moved_toward_donor_gold"].astype(float)
        detail["direction_aligned_shift"] = pd.to_numeric(
            detail["generated_count_shift"]
        ) * np.sign(pd.to_numeric(detail["gold_count_offset"]))
        for layer in sorted(pd.to_numeric(detail["start_layer"]).astype(int).unique()):
            selected = detail[pd.to_numeric(detail["start_layer"]).astype(int) == layer]
            aligned = _one_sample_seed_estimate(
                selected,
                metric="direction_aligned_shift",
                label=f"causal-patching-aligned-{model}-L{layer}",
            )
            moved = _one_sample_seed_estimate(
                selected,
                metric="moved_numeric",
                label=f"causal-patching-moved-{model}-L{layer}",
            )
            insertion = selected[selected["direction"] == "needle_insertion"]
            removal = selected[selected["direction"] == "needle_removal"]
            rows.append(
                {
                    "model": model,
                    "layer": int(layer),
                    "rows": int(len(selected)),
                    "changed_rate": float(
                        selected["prediction_changed_numeric"].mean()
                    ),
                    "moved_rate": moved["estimate"],
                    "moved_rate_low": moved["low"],
                    "moved_rate_high": moved["high"],
                    "insertion_shift": float(insertion["generated_count_shift"].mean()),
                    "removal_shift": float(removal["generated_count_shift"].mean()),
                    "aligned_shift": aligned["estimate"],
                    "aligned_shift_low": aligned["low"],
                    "aligned_shift_high": aligned["high"],
                    "aligned_shift_p_raw": aligned["p_raw"],
                }
            )
    adjusted = _holm_adjust([float(row["aligned_shift_p_raw"]) for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["aligned_shift_p_holm"] = p_holm
    return rows


def _causal_steering_rows(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        detail = frames[model]["steering"].copy()
        detail["prediction_changed_numeric"] = detail["prediction_changed"].astype(
            float
        )
        detail["moved_numeric"] = detail["moved_toward_path_count"].astype(float)
        detail["target_hit_numeric"] = detail["nearest_path_count_hit"].astype(float)
        detail["direction_aligned_shift"] = pd.to_numeric(
            detail["generated_count_shift"]
        ) * np.sign(pd.to_numeric(detail["intended_count_shift"]))
        for layer in sorted(pd.to_numeric(detail["layer"]).astype(int).unique()):
            selected = detail[pd.to_numeric(detail["layer"]).astype(int) == layer]
            identity = [
                "design_variant",
                "seed",
                "receiver_stimulus_id",
                "target_stimulus_id",
                "layer",
                "steering_method",
                "alpha",
            ]
            moved = _paired_seed_contrast(
                selected,
                metric="moved_numeric",
                condition_column="condition",
                treatment="geometric",
                control="orthogonal_norm_matched_random",
                identity_columns=identity,
                label=f"causal-steering-moved-{model}-L{layer}",
            )
            aligned = _paired_seed_contrast(
                selected,
                metric="direction_aligned_shift",
                condition_column="condition",
                treatment="geometric",
                control="orthogonal_norm_matched_random",
                identity_columns=identity,
                label=f"causal-steering-aligned-{model}-L{layer}",
            )
            geometric = selected[selected["condition"] == "geometric"]
            control = selected[
                selected["condition"] == "orthogonal_norm_matched_random"
            ]
            baseline = selected.drop_duplicates("receiver_stimulus_id")
            rows.append(
                {
                    "model": model,
                    "layer": int(layer),
                    "pairs_per_condition": int(len(geometric)),
                    "baseline_correct": int(
                        baseline["baseline_is_correct"].astype(bool).sum()
                    ),
                    "geometric_changed": float(
                        geometric["prediction_changed_numeric"].mean()
                    ),
                    "random_changed": float(
                        control["prediction_changed_numeric"].mean()
                    ),
                    "geometric_moved": float(geometric["moved_numeric"].mean()),
                    "random_moved": float(control["moved_numeric"].mean()),
                    "moved_difference": moved["estimate"],
                    "moved_difference_low": moved["low"],
                    "moved_difference_high": moved["high"],
                    "geometric_target_hit": float(
                        geometric["target_hit_numeric"].mean()
                    ),
                    "random_target_hit": float(control["target_hit_numeric"].mean()),
                    "geometric_aligned_shift": float(
                        geometric["direction_aligned_shift"].mean()
                    ),
                    "random_aligned_shift": float(
                        control["direction_aligned_shift"].mean()
                    ),
                    "aligned_difference": aligned["estimate"],
                    "aligned_difference_low": aligned["low"],
                    "aligned_difference_high": aligned["high"],
                    "aligned_difference_p_raw": aligned["p_raw"],
                }
            )
    adjusted = _holm_adjust([float(row["aligned_difference_p_raw"]) for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["aligned_difference_p_holm"] = p_holm
    return rows


def _causal_geometry_rows(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        geometry = frames[model]["geometry"].copy()
        for layer, selected in geometry.groupby("layer", sort=True):
            rows.append(
                {
                    "model": model,
                    "layer": int(layer),
                    "variants": int(selected["design_variant"].nunique()),
                    "projection_correlation_mean": float(
                        selected["endpoint_projection_count_correlation"].mean()
                    ),
                    "projection_correlation_min": float(
                        selected["endpoint_projection_count_correlation"].min()
                    ),
                    "monotone_fraction_min": float(
                        selected["endpoint_projection_monotone_fraction"].min()
                    ),
                    "step_cv_mean": float(selected["adjacent_step_cv"].mean()),
                    "successive_cosine_mean": float(
                        selected["mean_successive_step_cosine"].mean()
                    ),
                    "tortuosity_mean": float(selected["path_tortuosity"].mean()),
                }
            )
    return rows


def _table_answer_query_layer_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{int(row['layer'])}</td>"
            f"<td>{int(row['rows'])} / 10</td>"
            f"<td>{100*float(row['patched_valid_rate']):.2f}%</td>"
            f"<td>{int(row['eligible_donor_prediction_rows'])}</td>"
            f"<td>{100*float(row['eligible_donor_adoption_rate']):.1f}% "
            f"[{100*float(row['eligible_donor_adoption_rate_ci95_low']):.1f}, "
            f"{100*float(row['eligible_donor_adoption_rate_ci95_high']):.1f}] "
            f"(n={int(row['eligible_donor_prediction_rows'])}, "
            f"{int(row['eligible_donor_adoption_rate_seed_clusters'])} seeds)</td>"
            f"<td>{100*float(row['changed_rate']):.1f}%</td>"
            f"<td>{100*float(row['moved_toward_donor_gold_rate']):.1f}%</td>"
            f"<td>{100*float(row['follows_donor_prediction_rate']):.1f}%</td>"
            f"<td>{_number(row['mean_direction_aligned_shift'], signed=True)} "
            f"[{_number(row['mean_direction_aligned_shift_ci95_low'], signed=True)}, "
            f"{_number(row['mean_direction_aligned_shift_ci95_high'], signed=True)}]</td>"
            f"<td>{_p_value(row['eligible_donor_adoption_vs_layer0_p_holm'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_answer_query_variant_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{int(row['layer'])}</td>"
            f"<td>{html.escape(str(row['design_variant']))}</td>"
            f"<td>{int(row['rows'])} / {int(row['seed_clusters'])}</td>"
            f"<td>{100*float(row['patched_valid_rate']):.1f}%</td>"
            f"<td>{100*float(row['eligible_donor_adoption_rate']):.1f}% "
            f"[{100*float(row['eligible_donor_adoption_rate_ci95_low']):.1f}, "
            f"{100*float(row['eligible_donor_adoption_rate_ci95_high']):.1f}] "
            f"(n={int(row['eligible_donor_prediction_rows'])}, "
            f"{int(row['eligible_donor_adoption_rate_seed_clusters'])} seeds)</td>"
            f"<td>{_number(row['mean_direction_aligned_shift'], signed=True)} "
            f"[{_number(row['mean_direction_aligned_shift_ci95_low'], signed=True)}, "
            f"{_number(row['mean_direction_aligned_shift_ci95_high'], signed=True)}]</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_answer_query_pair_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{int(row['layer'])}</td>"
            f"<td>{int(row['receiver_count'])}→{int(row['donor_count'])}</td>"
            f"<td>{int(row['rows'])} / {int(row['seed_clusters'])}</td>"
            f"<td>{100*float(row['patched_valid_rate']):.1f}%</td>"
            f"<td>{int(row['eligible_donor_prediction_rows'])} / "
            f"{int(row['eligible_donor_adoption_rate_seed_clusters'])} seeds</td>"
            f"<td>{100*float(row['eligible_donor_adoption_rate']):.1f}% "
            f"[{100*float(row['eligible_donor_adoption_rate_ci95_low']):.1f}, "
            f"{100*float(row['eligible_donor_adoption_rate_ci95_high']):.1f}]</td>"
            f"<td>{100*float(row['follows_donor_prediction_rate']):.1f}%</td>"
            f"<td>{_number(row['mean_direction_aligned_shift'], signed=True)} "
            f"[{_number(row['mean_direction_aligned_shift_ci95_low'], signed=True)}, "
            f"{_number(row['mean_direction_aligned_shift_ci95_high'], signed=True)}]</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_answer_query_audit_html(audit: dict[str, Any]) -> str:
    rendered: list[str] = []
    for model in MODELS:
        row = audit["models"][model]
        rendered.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{row['shards']} / {row['detail_rows']}</td>"
            f"<td>{row['successful_rows']} / {row['skipped_rows']}</td>"
            f"<td>{row['patched_valid_rows']} / {row['patched_invalid_rows']}</td>"
            f"<td>{row['eligible_donor_prediction_rows']}</td>"
            "<td>verified</td>"
            "</tr>"
        )
    return "".join(rendered)


def _answer_query_invalid_html(invalid: pd.DataFrame) -> str:
    if invalid.empty:
        return (
            '<div class="callout"><strong>严格格式审计。</strong> '
            "所有 patched continuation 都是 1–10 内的合法整数。</div>"
        )
    models = ", ".join(sorted(invalid["model"].astype(str).unique()))
    layers = ", ".join(
        f"L{int(value)}" for value in sorted(pd.to_numeric(invalid["start_layer"]).unique())
    )
    first = invalid.iloc[0]
    return (
        '<div class="callout"><strong>严格格式审计：'
        f"{len(invalid)} 个非法输出。</strong>它们全部来自 {html.escape(models)}、"
        f"{html.escape(str(first['design_variant']))}、seed {int(first['seed'])}、"
        f"receiver {int(first['receiver_count'])} ← donor {int(first['donor_count'])}，"
        f"位于 {layers}。receiver baseline 为 <code>"
        f"{html.escape(str(first['receiver_baseline_completion_text_raw']))}</code> "
        f"（token IDs <code>{html.escape(str(first['receiver_baseline_generated_token_ids']))}</code>）；"
        f"donor baseline 为 <code>{html.escape(str(first['donor_baseline_completion_text_raw']))}</code> "
        f"（IDs <code>{html.escape(str(first['donor_baseline_generated_token_ids']))}</code>）；"
        f"patch 后生成 <code>{html.escape(str(first['patched_completion_text_raw']))}</code> "
        f"（IDs <code>{html.escape(str(first['patched_generated_token_ids']))}</code>）。"
        "这支持“前缀被 transport，但随后发生未 patch 的自回归续写错误”：answer-query state "
        "决定首位数字，而下一生成步已超出 single-position patch。所有 eligible invalid rows "
        "均按 donor-adoption failure 计入主分析，没有被删除。</div>"
    )


def _table_causal_ablation_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>top-{row['top_n']}</td>"
            f"<td>{row['prompts']} ({row['baseline_correct']} correct)</td>"
            f"<td>{100*row['ranked_changed']:.1f}% / {100*row['random_changed']:.1f}%</td>"
            f"<td>{100*row['changed_difference']:+.1f} pp "
            f"[{100*row['changed_difference_low']:+.1f}, {100*row['changed_difference_high']:+.1f}]</td>"
            f"<td>{_number(row['ranked_count_shift'], signed=True)} / "
            f"{_number(row['random_count_shift'], signed=True)}</td>"
            f"<td>{_number(row['count_shift_difference'], signed=True)} "
            f"[{_number(row['count_shift_difference_low'], signed=True)}, "
            f"{_number(row['count_shift_difference_high'], signed=True)}]</td>"
            f"<td>{_p_value(row['count_shift_p_holm'])}</td>"
            f"<td>{_number(row['error_difference'], signed=True)} "
            f"[{_number(row['error_difference_low'], signed=True)}, "
            f"{_number(row['error_difference_high'], signed=True)}]</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_causal_patching_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{row['layer']}</td>"
            f"<td>{row['rows']} / 10</td><td>{100*row['changed_rate']:.1f}%</td>"
            f"<td>{100*row['moved_rate']:.1f}% "
            f"[{100*row['moved_rate_low']:.1f}, {100*row['moved_rate_high']:.1f}]</td>"
            f"<td>{_number(row['insertion_shift'], signed=True)}</td>"
            f"<td>{_number(row['removal_shift'], signed=True)}</td>"
            f"<td>{_number(row['aligned_shift'], signed=True)} "
            f"[{_number(row['aligned_shift_low'], signed=True)}, "
            f"{_number(row['aligned_shift_high'], signed=True)}]</td>"
            f"<td>{_p_value(row['aligned_shift_p_holm'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_causal_steering_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{row['layer']}</td>"
            f"<td>{row['pairs_per_condition']} / 10</td>"
            f"<td>{100*row['geometric_changed']:.1f}% / {100*row['random_changed']:.1f}%</td>"
            f"<td>{100*row['geometric_moved']:.1f}% / {100*row['random_moved']:.1f}%</td>"
            f"<td>{100*row['moved_difference']:+.1f} pp "
            f"[{100*row['moved_difference_low']:+.1f}, {100*row['moved_difference_high']:+.1f}]</td>"
            f"<td>{100*row['geometric_target_hit']:.1f}% / "
            f"{100*row['random_target_hit']:.1f}%</td>"
            f"<td>{_number(row['geometric_aligned_shift'], signed=True)} / "
            f"{_number(row['random_aligned_shift'], signed=True)}</td>"
            f"<td>{_number(row['aligned_difference'], signed=True)} "
            f"[{_number(row['aligned_difference_low'], signed=True)}, "
            f"{_number(row['aligned_difference_high'], signed=True)}]</td>"
            f"<td>{_p_value(row['aligned_difference_p_holm'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_steering_v2_selection_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['protocol']))}</td>"
            f"<td>{html.escape(str(row['layer_set']))}</td>"
            f"<td>{_number(row['alpha'], 2)}</td>"
            f"<td>{row['candidate_plans']} / {row['screen_seeds']}</td>"
            f"<td>{_number(row['mean_screen_effect'], signed=True)}</td>"
            f"<td>{_number(row['worst_panel_screen_effect'], signed=True)}</td>"
            f"<td>{row['positive_screen_panels']}/4</td>"
            f"<td>{100*float(row['screen_valid_rate']):.1f}%</td>"
            f"<td>{_number(row['robust_selection_score'], signed=True)}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_steering_v2_summary_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['protocol']))}</td>"
            f"<td>{html.escape(str(row['layer_set']))}</td>"
            f"<td>{_number(row['alpha'], 2)}</td>"
            f"<td>{row['paired_rows']} / {row['seeds']}</td>"
            f"<td>{100*float(row['geometric_valid_rate']):.1f}% / "
            f"{100*float(row['random_valid_rate']):.1f}%</td>"
            f"<td>{_number(row['geometric_aligned_shift'], signed=True)} / "
            f"{_number(row['random_aligned_shift'], signed=True)}</td>"
            f"<td>{_number(row['aligned_effect'], signed=True)} "
            f"[{_number(row['aligned_effect_low'], signed=True)}, "
            f"{_number(row['aligned_effect_high'], signed=True)}]</td>"
            f"<td>{100*float(row['moved_effect']):+.1f} pp "
            f"[{100*float(row['moved_effect_low']):+.1f}, "
            f"{100*float(row['moved_effect_high']):+.1f}]</td>"
            f"<td>{100*float(row['target_hit_effect']):+.1f} pp</td>"
            f"<td>{_p_value(row['p_holm'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_steering_v2_panel_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['protocol']))}</td>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td>{row['paired_rows']} / 10</td>"
            f"<td>{_number(row['aligned_effect'], signed=True)} "
            f"[{_number(row['aligned_effect_low'], signed=True)}, "
            f"{_number(row['aligned_effect_high'], signed=True)}]</td>"
            f"<td>{_p_value(row['p_holm'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_steering_v2_audit_html(rows: list[dict[str, Any]]) -> str:
    return "".join(
        "<tr>"
        f"<td>{html.escape(str(row['model']))}</td>"
        f"<td>{row['screen_shards']} / {row['screen_rows']}</td>"
        f"<td>{row['confirmation_shards']} / {row['confirmation_rows']}</td>"
        f"<td><code>{html.escape(str(row['screen_root']))}</code></td>"
        f"<td><code>{html.escape(str(row['confirmation_root']))}</code></td>"
        "</tr>"
        for row in rows
    )


def _steering_v2_conclusion_html(
    overall_rows: list[dict[str, Any]],
    panel_rows: list[dict[str, Any]],
) -> str:
    cards: list[str] = []
    for model in MODELS:
        sentences: list[str] = []
        for protocol in ("single_layer", "multi_layer"):
            row = next(
                item
                for item in overall_rows
                if item["model"] == model and item["protocol"] == protocol
            )
            panels = [
                item
                for item in panel_rows
                if item["model"] == model and item["protocol"] == protocol
            ]
            positive = sum(float(item["aligned_effect"]) > 0 for item in panels)
            positive_ci = sum(float(item["aligned_effect_low"]) > 0 for item in panels)
            sentences.append(
                f"{protocol} {row['layer_set']} (α={float(row['alpha']):g})："
                f"Δ={_number(row['aligned_effect'], signed=True)} "
                f"[{_number(row['aligned_effect_low'], signed=True)}, "
                f"{_number(row['aligned_effect_high'], signed=True)}]，"
                f"向 target 移动率差={100*float(row['moved_effect']):+.1f} pp "
                f"[{100*float(row['moved_effect_low']):+.1f}, "
                f"{100*float(row['moved_effect_high']):+.1f}]，"
                f"exact target-hit 差={100*float(row['target_hit_effect']):+.1f} pp，"
                f"Holm p={_p_value(row['p_holm'])}；"
                f"panel 点估计为正 {positive}/4，CI 下界>0 {positive_ci}/4"
            )
        cards.append(
            '<div class="note"><strong>'
            + html.escape(model)
            + " held-out steering。</strong><p>"
            + "；".join(sentences)
            + "。</p></div>"
        )
    return "".join(cards)


def _table_causal_geometry_html(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td><td>L{row['layer']}</td>"
            f"<td>{row['variants']}</td>"
            f"<td>{_number(row['projection_correlation_mean'])} "
            f"(min {_number(row['projection_correlation_min'])})</td>"
            f"<td>{_number(row['monotone_fraction_min'])}</td>"
            f"<td>{_number(row['step_cv_mean'])}</td>"
            f"<td>{_number(row['successive_cosine_mean'])}</td>"
            f"<td>{_number(row['tortuosity_mean'])}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _table_causal_audit_html(audit: dict[str, Any]) -> str:
    rendered: list[str] = []
    for model in MODELS:
        stages = audit["models"][model]
        ablation = stages["ablation"]
        patching = stages["patching"]
        steering = stages["steering"]
        rendered.append(
            "<tr>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{ablation['shards']} / {ablation['detail_rows']}</td>"
            f"<td>{patching['shards']} / {patching['detail_rows']} "
            f"({patching['skipped_rows']} skipped)</td>"
            f"<td>{steering['discovery']['npz_shards']} / "
            f"{steering['shards']} / {steering['detail_rows']}</td>"
            "<td>verified</td>"
            "</tr>"
        )
    return "".join(rendered)


def _forest_svg(
    rows: list[dict[str, Any]],
    *,
    estimate_key: str,
    low_key: str,
    high_key: str,
    title: str,
    axis_label: str,
    label: Any,
) -> str:
    width = 1280
    left = 400
    right = 270
    top = 62
    row_height = 38
    height = top + row_height * len(rows) + 68
    lows = [float(row[low_key]) for row in rows]
    highs = [float(row[high_key]) for row in rows]
    minimum = min([0.0, *lows])
    maximum = max([0.0, *highs])
    span = max(maximum - minimum, 1e-6)
    minimum -= 0.08 * span
    maximum += 0.08 * span

    def x_position(value: float) -> float:
        return left + (float(value) - minimum) / (maximum - minimum) * (
            width - left - right
        )

    id_suffix = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="forest-title-{id_suffix} forest-desc-{id_suffix}">',
        f'<title id="forest-title-{id_suffix}">{html.escape(title)}</title>',
        f'<desc id="forest-desc-{id_suffix}">Purple marks Qwen and teal marks Gemma. Each circle is the seed-equal point estimate; the thick translucent horizontal segment is its seed-cluster bootstrap 95 percent confidence interval. The brown vertical line is zero. Right-side text repeats estimate and confidence interval.</desc>',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        '<g font-family="Aptos,Segoe UI,system-ui,sans-serif">',
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="{AURORA["night_black"]}">'
        f"{html.escape(title)}</text>",
        f'<circle cx="{width-250}" cy="43" r="5" fill="{MODEL_COLORS["Qwen3-8B"]}"/><text x="{width-239}" y="47" font-size="10" fill="{AURORA["frost_gray"]}">Qwen3-8B</text>',
        f'<circle cx="{width-145}" cy="43" r="5" fill="{MODEL_COLORS["Gemma4-E4B"]}"/><text x="{width-134}" y="47" font-size="10" fill="{AURORA["frost_gray"]}">Gemma4-E4B</text>',
    ]
    ticks = np.linspace(minimum, maximum, 6)
    for tick in ticks:
        x = x_position(float(tick))
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" '
                f'y2="{height-42}" stroke="{AURORA["frost_gray"]}" stroke-width="1" opacity=".28"/>',
                f'<text x="{x:.1f}" y="{height-24}" text-anchor="middle" '
                f'font-size="10" fill="{AURORA["frost_gray"]}">{tick:.2f}</text>',
            ]
        )
    zero_x = x_position(0.0)
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{top-14}" x2="{zero_x:.1f}" '
        f'y2="{height-40}" stroke="{AURORA["warm_brown"]}" stroke-width="1.6"/>'
    )
    colors = MODEL_COLORS
    for index, row in enumerate(rows):
        y = top + index * row_height
        estimate = float(row[estimate_key])
        low = float(row[low_key])
        high = float(row[high_key])
        color = colors.get(str(row.get("model")), AURORA["aurora_green"])
        parts.extend(
            [
                f'<text x="{left-12}" y="{y+4}" text-anchor="end" font-size="11" '
                f'fill="{AURORA["night_black"]}">{html.escape(str(label(row)))}</text>',
                f'<line x1="{x_position(low):.1f}" y1="{y}" '
                f'x2="{x_position(high):.1f}" y2="{y}" stroke="{color}" '
                'stroke-width="5" stroke-linecap="round" opacity=".38"/>',
                f'<circle cx="{x_position(estimate):.1f}" cy="{y}" r="6" '
                f'fill="{color}" stroke="{AURORA["snow_white"]}" stroke-width="1.5"/>',
                f'<text x="{width-18}" y="{y+4}" text-anchor="end" font-size="10" fill="{AURORA["frost_gray"]}">'
                f"{estimate:+.3f} [{low:+.3f}, {high:+.3f}]</text>",
            ]
        )
    parts.extend(
        [
            f'<text x="{(left + width-right)/2:.1f}" y="{height-5}" text-anchor="middle" '
            f'font-size="11" fill="{AURORA["night_black"]}">{html.escape(axis_label)}</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts)


def _causal_conclusion_html(
    ablation_rows: list[dict[str, Any]],
    patching_rows: list[dict[str, Any]],
    steering_rows: list[dict[str, Any]],
    answer_query_layer_rows: list[dict[str, Any]],
) -> str:
    cards: list[str] = []
    for model in MODELS:
        ablation = next(
            row for row in ablation_rows if row["model"] == model and row["top_n"] == 8
        )
        cards.append(
            '<div class="note"><strong>'
            + html.escape(model)
            + " mixed-bank 必要性。</strong><p>Top-8 ranked ablation 改变 "
            + f"{100*ablation['ranked_changed']:.1f}% 的输出；"
            + f"layer-matched random heads 仅改变 {100*ablation['random_changed']:.1f}%。"
            + "配对 count-shift contrast 为 "
            + _number(ablation["count_shift_difference"], signed=True)
            + " ["
            + _number(ablation["count_shift_difference_low"], signed=True)
            + ", "
            + _number(ablation["count_shift_difference_high"], signed=True)
            + "]，Holm p="
            + _p_value(ablation["count_shift_p_holm"])
            + "。负值表示 ablate ranked heads 后 undercount 更强。</p></div>"
        )
    steering_sentences: list[str] = []
    for model in MODELS:
        selected = [row for row in steering_rows if row["model"] == model]
        final = max(selected, key=lambda row: int(row["layer"]))
        steering_sentences.append(
            f"{model} L{final['layer']}：aligned geometric-minus-random shift "
            f"{_number(final['aligned_difference'], signed=True)} "
            f"[{_number(final['aligned_difference_low'], signed=True)}, "
            f"{_number(final['aligned_difference_high'], signed=True)}], "
            f"Holm p={_p_value(final['aligned_difference_p_holm'])}"
        )
    max_moved = max(float(row["moved_rate"]) for row in patching_rows)
    cards.append(
        '<div class="note"><strong>Transport 与 manipulability 的分离。</strong><p>'
        + "；".join(steering_sentences)
        + f"。相反，exact needle-end residual patching 在所有测试层中最多只有 "
        f"{100*max_moved:.1f}% 的 rows 朝 donor gold 移动。这说明晚层 answer-query "
        "count geometry 可以被因果操纵，但被测试的单个 endpoint 不是充分 transport channel。</p></div>"
    )
    onset_sentences: list[str] = []
    final_sentences: list[str] = []
    for row in _answer_query_onset_rows(pd.DataFrame(answer_query_layer_rows)):
        onset_sentences.append(
            f"{row['model']} L{int(row['layer'])}："
            f"{100*float(row['eligible_donor_adoption_rate']):.1f}% "
            f"[{100*float(row['eligible_donor_adoption_rate_ci95_low']):.1f}, "
            f"{100*float(row['eligible_donor_adoption_rate_ci95_high']):.1f}]"
        )
    for row in _answer_query_final_rows(pd.DataFrame(answer_query_layer_rows)):
        final_sentences.append(
            f"{row['model']} L{int(row['layer'])}："
            f"{100*float(row['eligible_donor_adoption_rate']):.2f}%"
        )
    cards.append(
        '<div class="note"><strong>Exact query-state transport。</strong><p>'
        + "首次出现显著且 ≥50% 的 donor-prediction adoption 位于 "
        + "；".join(onset_sentences)
        + "。最终 block 的保守 eligible-row adoption rates 为 "
        + "；".join(final_sentences)
        + "。因此，晚层 answer-query state 是模型已计算 prediction 的高度充分载体，"
        "与 needle-end transplant 的近零结果形成鲜明对照。</p></div>"
    )
    return "".join(cards)


def _projection_2d_svg(projection: dict[str, Any]) -> str:
    """Aurora PC1/PC2 audit panels on one shared scale per model/pooling."""
    width, height = 960, 820
    panel_width, panel_height = 370, 245
    panel_positions = ((90, 86), (535, 86), (90, 455), (535, 455))
    rows = projection["rows"]
    x_values = np.asarray([float(row[7]) for row in rows], dtype=float)
    y_values = np.asarray([float(row[8]) for row in rows], dtype=float)
    x_low, x_high = np.quantile(x_values, [0.005, 0.995])
    y_low, y_high = np.quantile(y_values, [0.005, 0.995])
    x_margin = max(1e-6, float(x_high - x_low) * 0.08)
    y_margin = max(1e-6, float(y_high - y_low) * 0.08)
    x_low, x_high = float(x_low - x_margin), float(x_high + x_margin)
    y_low, y_high = float(y_low - y_margin), float(y_high + y_margin)

    def project(row: list[Any], left: float, top: float) -> tuple[float, float]:
        x = left + (float(row[7]) - x_low) / (x_high - x_low) * panel_width
        y = top + panel_height - (float(row[8]) - y_low) / (y_high - y_low) * panel_height
        return x, y

    parts = [
        f'<svg class="projection-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="PC1 and PC2 projections for {html.escape(str(projection["model"]))} {html.escape(str(projection["pooling"]))}. Pale points are individual seed-by-occurrence states. Dashed black connects discovery centroids and solid black connects confirmation centroids. Color encodes occurrence index one through ten.">',
        f'<rect width="{width}" height="{height}" fill="{AURORA["snow_white"]}"/>',
        f'<g font-family="Aptos,Segoe UI,system-ui,sans-serif" fill="{AURORA["night_black"]}">',
        f'<circle cx="180" cy="28" r="3" fill="{AURORA["polar_violet"]}" opacity=".35"/><text x="190" y="32" font-size="10">individual seed × occurrence state</text>',
        f'<line x1="405" y1="28" x2="440" y2="28" stroke="{AURORA["night_black"]}" stroke-width="1.5" stroke-dasharray="6 5" opacity=".6"/><text x="448" y="32" font-size="10">discovery centroid path</text>',
        f'<line x1="650" y1="28" x2="685" y2="28" stroke="{AURORA["night_black"]}" stroke-width="2.4"/><text x="693" y="32" font-size="10">confirmation centroid path</text>',
    ]
    for panel_index, variant in enumerate(VARIANTS):
        left, top = panel_positions[panel_index]
        selected = [row for row in rows if row[0] == variant]
        parts.extend(
            [
                f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" '
                f'fill="{AURORA["snow_white"]}" stroke="{AURORA["frost_gray"]}" stroke-opacity=".42"/>',
                f'<text x="{left}" y="{top-16}" font-size="15" font-weight="700">{variant}</text>',
            ]
        )
        for fraction in (0.25, 0.5, 0.75):
            x = left + fraction * panel_width
            y = top + fraction * panel_height
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+panel_height}" '
                    f'stroke="{AURORA["frost_gray"]}" opacity=".16"/>',
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_width}" y2="{y:.1f}" '
                    f'stroke="{AURORA["frost_gray"]}" opacity=".16"/>',
                ]
            )
        for row in sorted(selected, key=lambda item: item[2] == "confirmation"):
            x, y = project(row, left, top)
            opacity = 0.50 if row[2] == "confirmation" else 0.16
            radius = 2.4 if row[2] == "confirmation" else 1.7
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{COUNT_COLORS[int(row[6])-1]}" opacity="{opacity}"/>'
            )
        for split, dash, opacity in (
            ("discovery", ' stroke-dasharray="6 5"', 0.60),
            ("confirmation", "", 0.95),
        ):
            split_rows = [row for row in selected if row[2] == split]
            centroid_rows: list[list[Any]] = []
            for count in range(1, 11):
                group = [row for row in split_rows if int(row[6]) == count]
                if not group:
                    continue
                centroid = group[0].copy()
                centroid[7] = float(np.mean([float(row[7]) for row in group]))
                centroid[8] = float(np.mean([float(row[8]) for row in group]))
                centroid_rows.append(centroid)
            points = [project(row, left, top) for row in centroid_rows]
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
                for index, (x, y) in enumerate(points)
            )
            parts.append(
                f'<path d="{path}" fill="none" stroke="{AURORA["night_black"]}" '
                f'stroke-width="{2.4 if split == "confirmation" else 1.5}" opacity="{opacity}"{dash}/>'
            )
            for count, (x, y) in enumerate(points, start=1):
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{COUNT_COLORS[count-1]}" '
                    f'stroke="{AURORA["night_black"]}" stroke-width=".8" opacity="{opacity}"/>'
                )
        parts.extend(
            [
                f'<text x="{left+panel_width/2:.1f}" y="{top+panel_height+28}" text-anchor="middle" font-size="11">PC1 score</text>',
                f'<text transform="translate({left-43} {top+panel_height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="11">PC2 score</text>',
                f'<text x="{left}" y="{top+panel_height+49}" font-size="10" fill="{AURORA["frost_gray"]}">shared axes: PC1 [{x_low:.2f}, {x_high:.2f}], PC2 [{y_low:.2f}, {y_high:.2f}]</text>',
            ]
        )
    legend_y = 800
    for count in range(1, 11):
        x = 135 + (count - 1) * 72
        parts.extend(
            [
                f'<circle cx="{x}" cy="{legend_y}" r="5" fill="{COUNT_COLORS[count-1]}"/>',
                f'<text x="{x+10}" y="{legend_y+4}" font-size="10">{count}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="80" y="{legend_y+4}" text-anchor="end" font-size="10" fill="{AURORA["frost_gray"]}">occurrence index</text>',
            "</g></svg>",
        ]
    )
    return "".join(parts)


def _static_figure_html(projections: dict[str, dict[str, Any]]) -> str:
    cards: list[str] = []
    for model in MODELS:
        for pooling in POOLINGS:
            projection = projections[f"{model}|{pooling}"]
            evr = projection["explained_variance_ratio"]
            cards.append(
                '<article class="figure-card">'
                f'<div class="figure-kicker">{html.escape(model)} · {html.escape(pooling.replace("_", "-"))} · L{int(projection["layer"])}</div>'
                '<div class="figure-intro"><p><strong>画什么：</strong>同一模型与 pooling 在 V4.1–V4.4 中的 occurrence-index PC1–PC2 轨迹及跨 seed 散布。</p>'
                '<p><strong>如何得到：</strong>只用 V4.1 discovery states 拟合该层 PCA，再把四个 panel 的 discovery/confirmation states 投到同一 basis；淡点是 seed×occurrence，折线是 1→10 的 split centroids。</p>'
                '<p><strong>能说明什么：</strong>可比较控制项释放后轨迹是否仍有序、散点是否变宽；不能把二维分离度等同于模型实际使用该坐标。</p></div>'
                f"{_projection_2d_svg(projection)}"
                '<p class="figure-caption"><strong>图 B2-F3b · Prompt-reading PC1–PC2 audit。</strong>'
                "横轴/纵轴是该卡片所示 model×pooling×layer 在 v4.1 discovery occurrence states 上拟合的 PC1/PC2 score；卡片内 V4.1–V4.4 四格共用同一 PCA basis 与坐标范围。"
                "颜色编码 occurrence index/count：N=1 为靛蓝，依次过渡到 N=10 的青色；淡小点是单个 seed×occurrence state，其中 discovery 更淡更小、confirmation 更深更大。"
                "黑色虚线及其节点连接 discovery 的 N=1→10 centroids，黑色实线及其节点连接 confirmation centroids；折线只连接离散均值，不是回归或平滑拟合。"
                f"PC1/PC2 分别解释 {100*float(evr[0]):.1f}%/{100*float(evr[1]):.1f}% 的 v4.1 discovery 总方差。PCA 正负号任意，只应比较卡片内的顺序、间距与跨 seed 散布，不能跨 model、pooling 或 layer 比较绝对坐标。</p>"
                "</article>"
            )
    return "\n".join(cards)


REPORT_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Realistic NIAH V4：从表征到因果机制</title>
<style>
:root {
  --midnight:#23165C; --violet:#6750E8; --cyan:#00C2FF; --yellow:#F6E36A;
  --teal:#00D4B4; --green:#39E58C; --magenta:#C04DFF; --pink:#FF5FA2;
  --ink:#171717; --paper:#F3EEE4; --muted:#5F6368; --frost:#8190A5;
  --brown:#765347; --line:rgba(118,83,71,.24); --surface:#FFFDF8;
  --surface-soft:#EEE6DA; --surface-strong:#FBF7EF;
  --soft-violet:rgba(103,80,232,.08); --soft-cyan:rgba(0,194,255,.09);
  --soft-yellow:rgba(246,227,106,.18); --soft-green:rgba(57,229,140,.10);
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; color:var(--ink); background:var(--paper); font:15px/1.68 "Aptos","Segoe UI Variable Text","Segoe UI",system-ui,sans-serif; }
header { padding:48px max(24px,calc((100vw - 1240px)/2)) 38px; color:var(--ink); background:var(--surface-strong); border-bottom:1px solid var(--line); }
header::after { display:none; }
header .eyebrow { color:var(--muted); text-transform:uppercase; letter-spacing:.12em; font-size:11px; font-weight:700; }
h1 { max-width:930px; margin:11px 0 15px; font:720 clamp(32px,4.5vw,54px)/1.08 Georgia,"Times New Roman",serif; letter-spacing:-.02em; }
header p { max-width:920px; margin:0; color:#444746; font-size:16px; }
.meta { display:flex; flex-wrap:wrap; gap:9px; margin-top:26px; }
.pill { border:1px solid var(--line); padding:6px 10px; color:#3C4043; font:12px/1.3 "Cascadia Mono","SFMono-Regular",Consolas,monospace; background:var(--surface-soft); }
nav { position:sticky; top:0; z-index:20; display:flex; gap:21px; overflow:auto; padding:11px max(24px,calc((100vw - 1240px)/2)); background:rgba(251,247,239,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
nav a { color:#3C4043; text-decoration:none; white-space:nowrap; font-weight:650; font-size:13px; }
nav a:hover { color:#000000; text-decoration:underline; }
main { max-width:1240px; margin:auto; padding:38px 24px 88px; }
section { margin:0 0 64px; padding-top:4px; scroll-margin-top:62px; }
.report-preamble,.report-appendix { margin:0 0 56px; padding-top:4px; scroll-margin-top:62px; }
.report-preamble { padding-bottom:34px; border-bottom:1px solid var(--line); }
.report-appendix { padding:30px 0 8px; border-top:1px solid var(--line); }
main>section { padding-top:34px; border-top:1px solid var(--line); }
.section-kicker { display:block; margin-bottom:8px; color:var(--muted); font:700 11px/1.2 "Cascadia Mono",Consolas,monospace; letter-spacing:.11em; text-transform:uppercase; }
h2 { max-width:980px; margin:0 0 11px; font:700 clamp(26px,3vw,36px)/1.16 Georgia,"Times New Roman",serif; letter-spacing:-.015em; }
h3 { margin:28px 0 9px; font-size:18px; line-height:1.3; }
h4 { margin:22px 0 7px; font-size:14px; color:var(--ink); }
p { max-width:980px; }
.lede { max-width:980px; color:#39445A; font-size:16px; }
.callout { margin:20px 0; padding:16px 18px; border-left:3px solid var(--brown); background:var(--surface-soft); }
.callout strong { color:var(--ink); }
.grid4 { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; margin:24px 0; border:1px solid var(--line); background:var(--line); }
.step { min-height:164px; padding:18px; background:var(--surface); border-top:2px solid #80868B; }
.step strong { display:block; color:var(--ink); font:780 23px/1.15 "Aptos Display","Segoe UI",sans-serif; }
.step small { color:var(--muted); font-family:"Cascadia Mono",Consolas,monospace; }
.table-wrap { overflow:auto; border:1px solid var(--line); background:var(--surface); }
.table-disclosure { margin:14px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
.table-disclosure > summary { display:list-item; padding:11px 2px; color:var(--ink); font-weight:720; }
.table-disclosure[open] > summary { margin-bottom:10px; }
.table-disclosure > .table-wrap:last-child { margin-bottom:12px; }
table { width:100%; border-collapse:collapse; font-size:12.5px; font-variant-numeric:tabular-nums; }
caption { padding:10px 12px; text-align:left; color:var(--muted); }
th,td { padding:9px 10px; text-align:right; border-bottom:1px solid rgba(129,144,165,.19); white-space:nowrap; }
th { position:sticky; top:0; background:var(--surface-soft); color:#3C4043; font:720 10.5px/1.3 "Cascadia Mono",Consolas,monospace; letter-spacing:.035em; text-transform:uppercase; }
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
tbody tr:hover { background:var(--soft-cyan); }
tr:last-child td { border-bottom:0; }
code { color:#202124; background:var(--surface-soft); padding:2px 5px; border:1px solid var(--line); font-family:"Cascadia Mono",Consolas,monospace; }
details { margin:14px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
summary { cursor:pointer; padding:11px 2px; color:var(--ink); font-weight:720; }
summary:hover { color:#000000; text-decoration:underline; }
.viz-shell { margin-top:22px; padding:18px; color:var(--paper); background:#161923; border:1px solid #3C4043; }
.controls { display:grid; grid-template-columns:repeat(6,minmax(110px,1fr)); gap:10px; margin-bottom:12px; }
label { display:flex; flex-direction:column; gap:4px; color:rgba(248,251,255,.72); font:720 10.5px/1.3 "Cascadia Mono",Consolas,monospace; letter-spacing:.04em; text-transform:uppercase; }
select,button { width:100%; border:1px solid rgba(0,194,255,.42); background:rgba(22,25,35,.54); color:var(--paper); padding:8px 9px; font:inherit; }
button { cursor:pointer; font-weight:720; transition:transform .16s ease,background .16s ease; }
button:hover { background:rgba(103,80,232,.42); } button:active { transform:translateY(1px); }
.canvas-wrap { position:relative; min-height:610px; background:#120D31; border:1px solid rgba(0,194,255,.32); }
#counter3d,#answer-counter3d,#joint-counter3d { display:block; width:100%; height:610px; cursor:grab; }
#counter3d.dragging,#answer-counter3d.dragging,#joint-counter3d.dragging { cursor:grabbing; }
#prompt-counter-attention-map { display:block; width:100%; height:650px; }
#tooltip,#answer-tooltip,#pcd-tooltip { position:absolute; display:none; pointer-events:none; max-width:340px; padding:9px 11px; border:1px solid var(--cyan); background:rgba(22,25,35,.95); color:var(--paper); font-size:12px; line-height:1.55; box-shadow:0 10px 24px rgba(22,25,35,.35); z-index:4; }
.viz-foot { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:12px; color:rgba(248,251,255,.72); font-size:12px; }
#geometry-stats { color:var(--yellow); text-align:right; }
.legend { display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; color:rgba(248,251,255,.76); font-size:12px; }
.legend i { display:inline-block; width:10px; height:10px; margin-right:5px; }
.figures { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
.figure-card { padding:13px; background:var(--surface); border:1px solid var(--line); }
.figure-kicker { margin:2px 4px 10px; color:#3C4043; font:720 12px/1.3 "Cascadia Mono",Consolas,monospace; }
.projection-svg,.stat-svg { display:block; width:100%; height:auto; background:var(--surface); }
.figure-caption,.stat-figure figcaption { margin:10px 5px 3px; color:var(--muted); font-size:12px; line-height:1.55; }
.figure-caption strong,.stat-figure figcaption strong { color:var(--ink); }
.stat-grid { display:grid; grid-template-columns:1fr; gap:16px; margin:19px 0; }
.stat-figure { margin:0; padding:14px; background:var(--surface); border:1px solid var(--line); }
.figure-intro { max-width:980px; margin:18px 0 10px; padding:13px 15px; background:var(--surface-strong); border:1px solid var(--line); color:#3C4043; font-size:13px; }
.figure-intro p { margin:4px 0; }
.figure-intro strong { color:#202124; }
.atlas-controls { display:flex; gap:8px; margin:0 0 12px; }
.atlas-controls button { width:auto; min-width:74px; padding:7px 13px; color:#3C4043; background:var(--surface); border:1px solid var(--line); }
.atlas-controls button[aria-pressed="true"] { color:#FFFFFF; background:#3C4043; border-color:#3C4043; }
.atlas-panel[hidden] { display:none; }
.atlas-panel .stat-svg { max-height:900px; }
.concept-box { max-width:980px; margin:18px 0; padding:15px 17px; background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--brown); }
.concept-box p { margin:6px 0 0; color:#4E463E; }
.concept-label { display:block; margin-bottom:4px; color:var(--brown); font:760 11px/1.3 "Cascadia Mono",Consolas,monospace; letter-spacing:.08em; text-transform:uppercase; }
.formula { max-width:980px; margin:18px 0; padding:16px 18px; background:var(--surface-strong); border:1px solid var(--line); border-left:4px solid var(--brown); overflow-x:auto; }
.formula-title { margin-bottom:8px; color:var(--brown); font:760 11px/1.3 "Cascadia Mono",Consolas,monospace; letter-spacing:.08em; text-transform:uppercase; }
.equation-grid { display:grid; gap:0; }
.equation-row { display:grid; grid-template-columns:minmax(300px,1fr) minmax(320px,1.45fr); gap:22px; align-items:start; padding:11px 0; border-top:1px solid var(--line); }
.equation-row:first-child { border-top:0; }
.equation-expression { color:var(--midnight); font:600 18px/1.55 "Cambria Math","STIX Two Math","Times New Roman",serif; white-space:nowrap; }
.equation-explain { color:#4E463E; font-size:13px; line-height:1.58; }
.equation-expression sub,.equation-expression sup { font-size:.72em; }
.formula-note { margin:10px 0 0; padding-top:10px; border-top:1px solid var(--line); color:var(--muted); font-size:12.5px; }
.command-block { max-width:980px; margin:16px 0; padding:14px 16px; color:#2F2A24; background:var(--surface-soft); border:1px solid var(--line); font:13px/1.6 "Cascadia Mono",Consolas,monospace; overflow:auto; white-space:nowrap; }
.method-strip { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; margin:17px 0; border:1px solid var(--line); background:var(--line); }
.method-strip div { padding:13px; background:var(--surface); font-size:12px; }
.method-strip strong { display:block; margin-bottom:4px; color:var(--ink); font-size:13px; }
.metric-defs { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:20px 0; }
.definition { padding:15px 17px; border-top:2px solid #80868B; background:var(--surface); }
.definition strong { color:var(--ink); }
.definition p { margin:5px 0 0; color:var(--muted); font-size:12.5px; }
.notes { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
.note { padding:17px; background:var(--surface); border-top:2px solid #80868B; }
.note strong { color:var(--ink); }
.section-conclusion { margin:24px 0 0; padding:17px 19px 18px; color:var(--ink); background:var(--surface-soft); border-left:4px solid var(--brown); }
.section-conclusion span { display:block; margin-bottom:5px; color:#3C4043; font:760 11px/1.2 "Cascadia Mono",Consolas,monospace; letter-spacing:.11em; text-transform:uppercase; }
.section-conclusion p { margin:0; color:#202124; }
.mechanism-flow { display:grid; grid-template-columns:1fr 36px 1fr 36px 1fr 36px 1fr; align-items:stretch; margin:25px 0; }
.flow-node { padding:17px; background:var(--surface); border-top:2px solid #80868B; }
.flow-node b { display:block; color:var(--ink); }
.flow-node small { color:var(--muted); }
.flow-arrow { display:grid; place-items:center; color:#5F6368; font-size:24px; }
.evidence-ledger { display:grid; grid-template-columns:1.2fr 1fr; gap:1px; margin:20px 0; border:1px solid var(--line); background:var(--line); }
.ledger-row { display:contents; }
.ledger-row>div { padding:13px 15px; background:var(--surface); }
.evidence-tag { display:inline-block; margin-right:8px; padding:2px 7px; color:#3C4043; background:var(--surface-soft); border:1px solid var(--line); font:700 10px/1.5 "Cascadia Mono",Consolas,monospace; text-transform:uppercase; }
.next-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.next-item { padding:18px 0; border-top:1px solid #80868B; }
.next-item strong { display:block; color:var(--ink); }
footer { padding:25px; color:var(--muted); text-align:center; border-top:1px solid var(--line); font-size:12px; }
@media (max-width:960px) { .grid4,.notes,.method-strip,.metric-defs,.next-grid { grid-template-columns:repeat(2,1fr); } .controls { grid-template-columns:repeat(3,1fr); } .figures { grid-template-columns:1fr; } .mechanism-flow { grid-template-columns:1fr; gap:0; } .flow-arrow { transform:rotate(90deg); min-height:34px; } }
@media (max-width:720px) { .equation-row { grid-template-columns:1fr; gap:5px; } .equation-expression { white-space:normal; } }
@media (max-width:600px) { main { padding-inline:16px; } header { padding-inline:18px; } .grid4,.notes,.method-strip,.metric-defs,.next-grid,.viz-foot,.evidence-ledger { grid-template-columns:1fr; } .ledger-row { display:block; } .controls { grid-template-columns:repeat(2,1fr); } #counter3d,#answer-counter3d,#joint-counter3d { height:500px; } .canvas-wrap { min-height:500px; } .stat-figure,.figure-card { overflow-x:auto; } .stat-svg,.projection-svg { min-width:720px; } }
@media (prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } button { transition:none; } }
</style>
</head>
<body>
<header>
  <div class="eyebrow">Realistic NIAH · Non-thinking · V4.1–V4.4</div>
  <h1>从可解码的 count signal，到生成真正使用的因果状态</h1>
  <p>本报告把完整 V4 结果组织成一条逐级收紧的证据链：先定位 prompt-reading representation，再检查 answer-query attention 如何聚合 needles，最后用 ablation、residual patching 与 geometric steering 区分“信息存在”“机制必要”“状态充分”和“方向可操纵”四种不同主张。</p>
  <div class="meta">
    <span class="pill">Qwen3-8B + Gemma4-E4B</span><span class="pill">10,000 canonical passage tokens</span><span class="pill">numeric counts 1–10</span><span class="pill">30 paired seeds / panel</span><span class="pill">commit @@COMMIT@@</span>
  </div>
</header>
<nav><a href="#behavior">1 · Behavior</a><a href="#counter-representation">2 · Counter representation</a><a href="#attention-representation">3 · Attention representation</a><a href="#head-ablation">4 · Head ablation</a><a href="#geometry-steering">5 · Geometry steering</a><a href="#appendix">综合与复现</a></nav>
<main>
<div class="report-preamble" id="overview">
  <span class="section-kicker">Executive synthesis · 不计入五个证据块</span>
  <h2>当前最小机制：分布式 retrieval bank，在后层写入可执行的 answer-query count state</h2>
  <p class="lede">最符合全部结果的工作模型不是“每个 needle 末尾独立保存一个可直接搬运的整数”，也不是“一个最强 head 均匀数完所有 needles”。更窄、也更可检验的模型是：多个 attention heads 在 <code>Total:</code> query 聚合 occurrence evidence；被测试的 mixed ranked bank 对维持最终 count magnitude 有因果贡献；随后在模型后段形成一个能够决定首个答案 token 的 query residual state。global broad heads 是否作为一个 phenotype 单独必要，目前仍未被 ablation 分离。</p>
  <div class="mechanism-flow" aria-label="V4 mechanism summary">
    <div class="flow-node"><b>Needle-local states</b><small>span-end 中 count index 可解码，但单 endpoint state 跨 prompt patch 不足以搬运 count。</small></div><div class="flow-arrow">→</div>
    <div class="flow-node"><b>Distributed head bank</b><small>多个 broad / selector / local heads 并存；ranked top-8 bank ablation 比 layer-matched random 更强地导致 undercount。</small></div><div class="flow-arrow">→</div>
    <div class="flow-node"><b>Late answer-query state</b><small>Qwen L26、Gemma L31 开始出现强 donor-prediction transport；末层对合法 eligible rows 达到 100%。</small></div><div class="flow-arrow">→</div>
    <div class="flow-node"><b>Greedy numeric output</b><small>count-related geometry可被 steering，但 exact target hit 仍低；多 token 的“10”还需要后续自回归计算。</small></div>
  </div>
  <div class="evidence-ledger">
    <div class="ledger-row"><div><span class="evidence-tag">Descriptive</span>Needle-end count information persists</div><div>v4.4 confirmation R²：Qwen 0.866；Gemma 0.916。</div></div>
    <div class="ledger-row"><div><span class="evidence-tag">Correlational</span>Undercount aligns with low span-end attention</div><div>Omitted-tail overlap exceeds its combinatorial baseline; nested failures more often place the new needle in bottom-k.</div></div>
    <div class="ledger-row"><div><span class="evidence-tag">Necessary</span>Discovery-ranked mixed head bank preserves count</div><div>Top-8 ranked-minus-random count shift：Qwen −0.331；Gemma −2.156；不归因于单一 phenotype。</div></div>
    <div class="ledger-row"><div><span class="evidence-tag">Sufficient</span>Late query state carries computed prediction</div><div>Final-layer valid eligible rows全部复制 donor prediction；不是 donor gold，也不保证多 token realization。</div></div>
    <div class="ledger-row"><div><span class="evidence-tag">Manipulable</span>Late query geometry moves output</div><div>Geometric-minus-random aligned shift：Qwen L26 +0.958；Gemma L31 +1.388。</div></div>
  </div>
  <div class="section-conclusion"><span>本节结论</span><p>目前可以主张“distributed retrieval/aggregation → late executable query state”这条 bank-level 机制链；还不能把因果必要性专门归给 global broad heads，也不能主张存在唯一 scalar counter、单一 broad head、固定 partition circuit，或能够精确设定任意 target count 的线性控制方向。</p></div>
</div>
<div class="report-preamble" id="design">
  <span class="section-kicker">Experimental design · 全部五块共享</span>
  <h2>实验设定：用四级 controlled relaxation 定位 seed sensitivity 的来源</h2>
  <p class="lede">两个模型都以 non-thinking mode 直接回答数字。每个 panel 包含 30 个 paired seeds × 10 个 gold counts；seed 1234–1253 仅用于 discovery、模型/层/head/方向选择，seed 1254–1263 仅用于 confirmation。所有 correctness、wrong/undercount 与 causal effect 标签都来自 <code>Total:</code> 后完整 deterministic greedy continuation；数字 10 按完整多-token sequence 解析，不使用 first-token probability。</p>
  <div class="concept-box"><span class="concept-label">此处定义 · Discovery / confirmation</span><p><strong>Discovery</strong> 是 seeds 1234–1253，只用于选择 layer/head、拟合 PCA basis、Ridge probe、count centroids 与 steering plan；<strong>confirmation</strong> 是完全不相交的 seeds 1254–1263，只用于最终 held-out 估计。任何用 confirmation 观察到的结果都不会回头改变 discovery 阶段的选择。</p></div>
  <div class="grid4">
    <div class="step"><strong>V4.1</strong><small>all fixed</small><p>Needle position、city-score 顺序与具体内容跨 seed 固定，只改变 count 和 haystack。</p></div>
    <div class="step"><strong>V4.2</strong><small>release position</small><p>释放 needle position；city-score 顺序与内容仍固定。</p></div>
    <div class="step"><strong>V4.3</strong><small>release order</small><p>同时释放 position 与 city-score 顺序；内容仍固定。</p></div>
    <div class="step"><strong>V4.4</strong><small>release content</small><p>position、顺序、city-score 内容全部跨 seed 变化。</p></div>
  </div>
  <div class="method-strip">
    <div><strong>Stimulus</strong>10,000 canonical passage tokens；gold count N∈{1,…,10}；同一 family 采用 nested N−1→N construction，新增 occurrence 可精确定位。</div>
    <div><strong>Prompt-reading capture</strong>每个 active needle 保存两种 state：最后 token 的 <code>span_end</code>，以及整个 needle span 的 tokenwise mean <code>span_mean</code>。</div>
    <div><strong>Answer-query capture</strong>在 prompt-final <code>Total:</code> query 保存 hidden state 与每层每 head 对原 prompt key positions 的原始 attention row。</div>
    <div><strong>Models and output</strong>Qwen3-8B 与 Gemma4-E4B；greedy、numeric-only、最多 16 new tokens；所有 2,400 个 baseline answers 均 format-valid。</div>
  </div>
  <h3>完成性与原始数据审计</h3>
  <div class="table-wrap"><table><thead><tr><th>artifact / model</th><th>Qwen3-8B</th><th>Gemma4-E4B</th><th>用于什么</th></tr></thead><tbody>
    <tr><td>behavior rows</td><td>1,200</td><td>1,200</td><td>完整 greedy output label</td></tr>
    <tr><td>representation capture shards</td><td>120</td><td>120</td><td>span-end / span-mean hidden states</td></tr>
    <tr><td>raw answer-query attention tensors</td><td>1,200</td><td>1,200</td><td>head ranking、omission、partitioning</td></tr>
    <tr><td>raw attention bytes</td><td>28.36 GB</td><td>1.78 GB</td><td>保留可复算的 query rows</td></tr>
    <tr><td>causal detail rows / model</td><td>9,200</td><td>9,200</td><td>640 ablation + 720 endpoint patch + 2,560 query patch + 1,440 initial steering + 3,840 steering v2</td></tr>
  </tbody></table></div>
  <div class="callout"><strong>解释规则。</strong>v4.1 的干净曲线仍可能只是固定 identity、位置或记录顺序编码。只有 signal 在 v4.3/v4.4 仍可跨 seed 解码，才支持相对 content-independent 的 count representation；即便如此，representation 仍不等于机制。</div>
  <div class="section-conclusion"><span>本节结论</span><p>V4.1→V4.4 是逐项释放 nuisance factors 的 paired ladder，而不是四个无关数据集。后续所有 discovery selection 与 confirmation inference 严格分离；实验数据、raw attention、hidden states 和 causal detail rows 均完整，因此报告中的差异可以解释为控制项释放与干预效应，而不是样本缺失。</p></div>
</div>

<section id="behavior">
  <span class="section-kicker">Block 1 / 5 · Behavior analysis</span>
  <h2>Behavior：主要失败边界随 count 增大出现，而不是由某一个 V4 panel 单独触发</h2>
  <p class="lede">图中横轴是真实 needle count N，纵轴是完整 greedy numeric sequence 的 exact-match accuracy；每条线对应一个 V4 panel，每个点包含 10 个 confirmation seeds。黄色背景从 N=5 开始，仅用于帮助观察中高 count 区间，不参与统计。</p>
  <div class="figure-intro"><p><strong>画什么：</strong>两个模型在 count 1–10 上的完整 greedy 数字答案准确率，以及四种控制释放条件的差异。</p><p><strong>如何得到：</strong>每个点汇总一个 model×panel×gold-count 下 10 个 confirmation seeds；只有最终 continuation 能被严格解析为正确的 1–10 整数序列才记为正确。</p><p><strong>能说明什么：</strong>它定位行为失效从哪个 count 开始、是否随 V4.1→V4.4 系统变化；它本身不解释失败发生在哪个内部计算阶段。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@BEHAVIOR_ACCURACY_SVG@@<figcaption><strong>图 B1-F1 · Confirmation accuracy by count。</strong>左图为 Qwen3-8B，右图为 Gemma4-E4B；横轴是真实 needle count N=1–10，纵轴是完整 greedy continuation 经严格 1–10 数字解析后的 exact-match accuracy（0–1）。靛蓝、紫、青、粉四条线依次表示 V4.1、V4.2、V4.3、V4.4；每个圆点汇总该 model×panel×count 的 10 个 confirmation seeds，线段只连接相邻 count 的点，不是拟合曲线。N≥5 的淡黄色底纹只是视觉分区，不进入统计。两模型均在 N≈4–6 开始快速下降，N=9–10 的错误几乎都是 undercount。</figcaption></figure></div>
  <h3>Panel-level confirmation summary</h3>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>correct / 100</th><th>accuracy</th><th>mean prediction</th><th>MAE</th><th>undercount</th><th>format valid</th></tr></thead><tbody>@@BEHAVIOR_PANEL_ROWS@@</tbody></table></div>
  <details><summary>展开：跨 panel pooling 后每个 count 的完整数值</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>gold N</th><th>correct / 40</th><th>accuracy</th><th>mean prediction</th><th>undercount</th></tr></thead><tbody>@@BEHAVIOR_COUNT_ROWS@@</tbody></table></div></details>
  @@BEHAVIOR_CONCLUSION@@
</section>

<section id="counter-representation">
  <span class="section-kicker">Block 2 / 5 · Counter representation</span>
  <h2>Counter representation：prompt 读取过程与 <code>Total:</code> query 的状态必须分开定位</h2>
  <div class="concept-box"><span class="concept-label">本节先定义 · 状态位置与层号</span><p><strong>L0 不是 embedding。</strong>Qwen 的 L0–L35、Gemma 的 L0–L41 都表示对应 decoder block 输出之后的 zero-based post-block residual。Prompt-reading state 在 needle 的最后一个 token（<code>span_end</code>）或全 span token 均值（<code>span_mean</code>）处捕获；answer-query state 则在 prompt 末尾 <code>Total:</code> query、首个答案 token 尚未生成时捕获。</p></div>
  <h3>2.1 哪一层最可解码，哪一层最适合显示 manifold？</h3>
  <p class="lede">原分析的 probe-optimal layer 只用 v4.1 discovery grouped-seed full-space CV-R² 选择：Qwen span-end L1、span-mean L0；Gemma span-end L22、span-mean L0。它回答“哪层在完整 residual space 中最容易线性解码”，不回答“哪层的前三个 PC 最完整展示 count manifold”。因此本报告保留原 probe 结果，同时新增逐层 PCA/manifold sweep，并把 3D 展示层单独命名为 manifold-display layer。</p>
  <div class="formula">
    <div class="formula-title">Ridge count probe 与 held-out 拟合度</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">ŷ = w<sup>T</sup>z(h) + b</div><div class="equation-explain"><em>h</em> 是捕获的 residual，<em>z(h)</em> 是标准化后的完整 hidden state；Ridge 的 α 只在 discovery seeds 上用 grouped 5-fold CV 选择。</div></div>
      <div class="equation-row"><div class="equation-expression">R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²</div><div class="equation-explain">R²=1 表示完美预测；R²=0 等同于只预测 confirmation 标签均值；R²&lt;0 表示比该均值基线更差。</div></div>
    </div>
    <p class="formula-note">每个 V4 panel 都在自己的 discovery seeds 上拟合，再在同 panel 的 10 个 confirmation seeds 上评估；confirmation 不参与 layer 或 α 的选择。</p>
  </div>
  <div class="figure-intro"><p><strong>画什么：</strong>在预先选定的 probe-optimal layer 上，span-end 与 span-mean 对 occurrence index/count 的 held-out 线性解码强度。</p><p><strong>如何得到：</strong>每个 panel 只用 discovery seeds 选择 ridge 正则并拟合 full-space probe，再在不相交的 10 个 confirmation seeds 上计算 R²；四个 panel 各自拟合、各自验证。</p><p><strong>能说明什么：</strong>正 R² 说明该位置的完整 residual 含可泛化的线性 count signal；它不说明信号低维、因果必要或可被单点运输。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@REPRESENTATION_R2_SVG@@<figcaption><strong>图 B2-F1 · Held-out count decoding。</strong>左图为 Qwen3-8B，右图为 Gemma4-E4B；横轴从 V4.1 到 V4.4 依次释放 position、city-score order 与 content，纵轴是在 10 个未参与拟合的 confirmation seeds 上得到的 Ridge count-probe R²。青色实线/圆点是 span-end（Qwen L1、Gemma L22），粉色虚线/圆点是 span-mean（两模型均 L0）；线段只连接四个 panel 的离散估计。棕色水平线是 R²=0：位于其上优于用 confirmation 标签均值预测，位于其下则更差。span-end 到 V4.4 仍为正；span-mean 在 V4.3 释放 city-score order 后明显退化，说明其早期可解码性强依赖固定记录结构。</figcaption></figure></div>
  <div class="formula">
    <div class="formula-title">选择 3D manifold-display layer 的四个量</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">EVR₃ = (λ₁ + λ₂ + λ₃) / Σ<sub>k</sub>λ<sub>k</sub></div><div class="equation-explain">前三个主成分解释的<strong>全部样本方差</strong>比例；高值也可能来自位置或内容等 nuisance variation。</div></div>
      <div class="equation-row"><div class="equation-expression">F₃ = Σ<sub>i</sub>‖P₃(μ<sub>i</sub>−μ̄)‖² / Σ<sub>i</sub>‖μ<sub>i</sub>−μ̄‖²</div><div class="equation-explain">前三个 PC 保留的<strong>count-centroid between-count signal</strong>比例；μ<sub>i</sub> 是 count/index i 的 discovery centroid。</div></div>
      <div class="equation-row"><div class="equation-expression">C = 1 / (1 + R<sub>LOO</sub>)</div><div class="equation-explain">跨 seed 紧致度；R<sub>LOO</sub> 是 leave-one-seed-out noise RMS 与 count-centroid signal RMS 的比值，所以 C 越高越紧。</div></div>
      <div class="equation-row"><div class="equation-expression">M₃ = EVR₃ × F₃ × C</div><div class="equation-explain">先保留 full-space CV-R² 距最优值不超过 0.02 的层，再用 M₃ 最大者作为 3D 展示层。它只选择展示，不替代 probe-optimal layer。</div></div>
    </div>
  </div>
  <div class="figure-intro"><p><strong>画什么：</strong>每个已捕获 decoder layer 的 full-space 可解码度、前三个 PC 的总方差解释度、前三个 PC 对 count-centroid signal 的保留率，以及跨 seed 紧致度。</p><p><strong>如何得到：</strong>所有曲线只使用 V4.1 discovery states；P 标记 full-space CV-R² 最大层，M 在“距最佳 R²≤0.02”的层中再按 M₃=EVR₃×F₃×compactness 选择。</p><p><strong>能说明什么：</strong>它把“最容易线性读出”与“最适合用 3D 展示”分开，避免仅凭 PCA explained variance 选到主要解释 nuisance 的层。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@LAYER_SWEEP_SVG@@<figcaption><strong>图 B2-F2 · Discovery-only layer sweep。</strong>四格顺序为：上排 Qwen、下排 Gemma；左列 span-end、右列 span-mean。横轴是 zero-based post-block decoder layer，纵轴统一为 0–1；每条折线连接相邻已捕获层的 discovery-only 数值，不做平滑。紫线是完整 residual space 的 grouped-seed count-probe CV-R²。青线是 EVR₃=(λ₁+λ₂+λ₃)/Σλ，即 PC1–PC3 对全部样本方差的解释比例。绿线是 F₃=Σᵢ||P₃(μᵢ−μ̄)||²/Σᵢ||μᵢ−μ̄||²，即前三个 PC 保留的 between-count centroid signal 比例。<strong>粉线是 seed compactness C=1/(1+R<sub>LOO</sub>)</strong>，其中 R<sub>LOO</sub>=跨 seed noise RMS/count-centroid signal RMS，因此粉线越高表示同一 count 的不同 seeds 相对 count 间距越集中。四条线均为越高越好。棕色虚线 P 标出 full-space CV-R² 最大的 probe-optimal 层；靛蓝实线 M 标出先要求 R² 距最优≤0.02、再最大化 M₃=EVR₃×F₃×C 的 manifold-display 层；每格标题下的 P:Lx/M:Ly 给出层号。该图显示高 EVR 本身不足以保证低维图忠实呈现 count manifold。</figcaption></figure></div>
  <details><summary>Probe-optimal 与 manifold-display 层的逐项比较</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>probe L</th><th>probe full CV R²</th><th>probe EVR₃</th><th>probe F₃</th><th>probe LOO noise/signal</th><th>display L</th><th>display full CV R²</th><th>display EVR₃</th><th>display F₃</th><th>display PCA3 CV R²</th><th>display LOO noise/signal</th><th>M₃</th></tr></thead><tbody>@@LAYER_SELECTION_ROWS@@</tbody></table></div></details>
  <p class="artifact-link">完整逐层数值：<a href="realistic_niah_v4_layer_sweep.csv">realistic_niah_v4_layer_sweep.csv</a>。</p>
  @@LAYER_SELECTION_CONCLUSION@@
  <div class="concept-box"><span class="concept-label">下表使用 · Seed 散点与跨 split 几何</span><p><strong>Noise / signal</strong> 是 confirmation 样本到其 discovery count centroid 的 RMS 距离，除以十个 discovery centroids 相对 grand mean 的 RMS 距离；越小表示同 count 的跨 seed 散点相对 count 间距越紧。<strong>Linear CKA</strong> 比较 discovery/confirmation 的 centered centroid Gram geometry；<strong>distance correlation</strong> 比较两套 centroid 两两距离。后二者越接近 1，跨 split 的相对几何越稳定。</p></div>
  <details><summary>Primary-layer confirmation metrics</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>layer</th><th>panel</th><th>confirm R²</th><th>confirm MAE</th><th>noise / signal</th><th>linear CKA</th><th>distance corr.</th></tr></thead><tbody>@@METRIC_ROWS@@</tbody></table></div></details>
  <details><summary>Paired confirmation-seed sensitivity：相邻 relaxation 在哪里首次显著变差</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>layer</th><th>metric</th><th>step</th><th>Δ mean</th><th>95% seed CI</th><th>CI &gt; 0</th></tr></thead><tbody>@@SENSITIVITY_ROWS@@</tbody></table></div></details>
  @@REPRESENTATION_CONCLUSION@@
  <div id="counter">
  <span class="section-kicker">Prompt-reading geometry + answer-query geometry</span>
  <h3>2.2 Prompt-reading counter：随 prompt 读取的 needle occurrence states</h3>
  <p class="lede">下面的 3D view 可选择模型、pooling、<strong>任意已捕获 decoder layer</strong>、V4 panel、split 与输出标签。每个 model×pooling×layer 都只用 v4.1 discovery occurrence states 拟合自己的 PC1–PC6，再把同一 basis 应用于 v4.1–v4.4；layer 下拉框默认落在 manifold-display 层。淡点是单 seed×occurrence state，彩色节点/连线是 index 1→10 centroids。不同层分别拟合 PCA，因此只能比较每层内部的顺序、散布与解释度，不能把 PC 坐标值跨层直接相减。</p>
  <div class="figure-intro"><p><strong>画什么：</strong>模型逐个读入第 1→10 个 needle 时，needle-end 或整段均值 residual 在任意已捕获层形成的三维 trajectory。</p><p><strong>如何得到：</strong>对每个 model×pooling×layer 单独在 V4.1 discovery 拟合 PC1–PC6；交互控件仅切换投影数据、panel、split 和最终行为标签，不重新拟合 PCA。</p><p><strong>能说明什么：</strong>可检查 index trajectory 是否连续、是否接近一维、到哪一层最清楚以及释放 position/order/content 后是否对 seed 敏感；三维外仍可能存在重要 count signal。</p></div>
  <div class="viz-shell">
    <div class="controls">
      <label>Model<select id="model-select"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
      <label>Pooling<select id="pooling-select"><option value="span_end">span-end</option><option value="span_mean">span-mean</option></select></label>
      <label>Post-block layer<select id="layer-select"></select></label>
      <label>Variant<select id="variant-select"><option>v4.1</option><option>v4.2</option><option>v4.3</option><option>v4.4</option></select></label>
      <label>Split<select id="split-select"><option value="all">all</option><option value="discovery">discovery</option><option value="confirmation">confirmation</option></select></label>
      <label>Final output<select id="outcome-select"><option value="all">all</option><option value="correct">correct</option><option value="wrong">wrong</option><option value="invalid">invalid</option></select></label>
      <label>View<button id="reset-view" type="button">reset rotation</button></label>
      <label>X axis<select id="x-axis"></select></label>
      <label>Y axis<select id="y-axis"></select></label>
      <label>Z axis<select id="z-axis"></select></label>
      <label>Points<select id="points-select"><option value="all">all seed points</option><option value="confirmation">confirmation only</option><option value="centroids">centroids only</option></select></label>
      <label>Scale<select id="scale-select"><option value="metric">equal metric scale</option><option value="normalized">normalize each axis</option></select></label>
      <label>Preset<select id="axis-preset"><option value="0,1,2">PC1 / PC2 / PC3</option><option value="0,2,3">PC1 / PC3 / PC4</option><option value="1,2,3">PC2 / PC3 / PC4</option><option value="3,4,5">PC4 / PC5 / PC6</option></select></label>
    </div>
    <div class="canvas-wrap"><canvas id="counter3d" aria-label="Interactive 3D PCA counter trajectory"></canvas><div id="tooltip"></div></div>
    <div class="viz-foot"><div id="pca-stats"></div><div id="geometry-stats"></div></div>
    <div class="legend" id="count-legend"></div>
  </div>
  <p class="figure-caption"><strong>图 B2-F3a · Interactive prompt-reading counter trajectory。</strong>交互图追踪一个 N=10 prompt 在读到第 1→10 个 needle 时的 occurrence state；Model、Pooling、Post-block layer、V4 panel、split 与最终 greedy output 标签均可切换。X/Y/Z 下拉框选择该 model×pooling×layer 的 PC1–PC6，默认显示 PC1/PC2/PC3；颜色从靛蓝 N=1 依次过渡到青色 N=10。淡点是当前筛选条件下的单个 seed×occurrence state，彩色大节点和连线是 occurrence 1→10 的 centroids；连线只表示顺序，不是拟合曲线。数字标签会在屏幕坐标中自动避让；过密时省略部分数字，但上方 1–10 色标始终给出完整映射。右下统计给出所选 PC 的 discovery EVR、step CV 与 path/chord。每个 layer 都在 V4.1 discovery 上单独拟合 PCA，因此只可比较同一 model×pooling×layer 内的 panel/split/标签变化，不可跨层比较 PC 绝对坐标；“correct/wrong”是整条 N=10 prompt 的最终输出标签，十个 occurrence 点共享该标签。</p>
  <div class="formula">
    <div class="formula-title">3D trajectory 的形状诊断</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">step CV = SD(d<sub>i</sub>) / mean(d<sub>i</sub>)</div><div class="equation-explain">其中 d<sub>i</sub>=‖μ<sub>i+1</sub>−μ<sub>i</sub>‖。越接近 0，表示相邻 count 的 centroid 步长越等距。</div></div>
      <div class="equation-row"><div class="equation-expression">path / chord = Σ<sub>i=1..9</sub>d<sub>i</sub> / ‖μ<sub>10</sub>−μ<sub>1</sub>‖</div><div class="equation-explain">越接近 1，centroid path 越接近直线；明显大于 1 表示弯折或回绕。</div></div>
    </div>
    <p class="formula-note">理想的等距直线计数轴应同时满足 step CV≈0 与 path/chord≈1；两项只描述几何，不证明该轴被生成机制使用。</p>
  </div>
  <div class="callout"><strong>坐标可比性。</strong>同一 model×pooling 内四个 panel 共享 PCA basis；不同模型或不同 pooling 分别拟合，因此 PC 坐标绝对值不可跨 panel group 直接比较。PCA component 的正负号没有语义。</div>
  <h3>Aurora PC1–PC2 audit panels</h3>
  <p>以下四张图与 3D view 使用相同隐藏状态与 v4.1 discovery basis，但固定展示 PC1–PC2，便于比较跨 seed 散点宽度。它们替代旧配色 PNG 作为主报告图；原始 CSV/PNG 仍保留在 run artifact 中。</p>
  <div class="figures">@@STATIC_FIGURES@@</div>
  <h3>2.2b Prompt counter 写入时的 attention dispersion 与 hidden-state noise</h3>
  <div class="concept-box"><span class="concept-label">先固定 query · 两种模式都从 needle end 发出 attention</span><p><strong>无论下拉框选哪一种模式，query 都固定在模型刚读完第 <em>n</em> 个 needle 的最后一个 token。</strong>变化的不是 query，而是横轴第 <em>j</em> 个 needle 用哪些 key tokens 接收这条 attention row：<code>endpoint-key</code> 只取第 <em>j</em> 个 needle 的最后一个 token；<code>full-span-key</code> 把第 <em>j</em> 个 needle 的全部 token 权重求和。图中第 <em>n</em> 行因此回答“刚读完第 <em>n</em> 个 needle 时，这个末 token 回看第 1…<em>n</em> 个 needles 的哪一些位置？”</p></div>
  <div class="formula">
    <div class="formula-title">热图一个格子的严格定义</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">A<sup>end</sup><sub>n,j</sub>=mean<sub>s,h</sub> α<sup>ℓ,h</sup><sub>end(S<sub>n</sub>)→end(S<sub>j</sub>)</sub></div><div class="equation-explain">Endpoint-key 模式：query 是第 n 个 needle 的末 token；key 只取第 j 个 needle 的末 token。s 遍历 10 个 confirmation seeds，h 遍历只用 discovery 冻结的 top-8 heads。</div></div>
      <div class="equation-row"><div class="equation-expression">A<sup>span</sup><sub>n,j</sub>=mean<sub>s,h</sub> Σ<sub>t∈S<sub>j</sub></sub> α<sup>ℓ,h</sup><sub>end(S<sub>n</sub>)→t</sub></div><div class="equation-explain">Full-span-key 模式：query 仍是第 n 个 needle 的末 token；只是把第 j 个 needle span 内所有 key-token attention 相加。因此它衡量整条 record 收到的质量。</div></div>
      <div class="equation-row"><div class="equation-expression">P<sub>n,j</sub>=A<sub>n,j</sub>/Σ<sub>r≤n</sub>A<sub>n,r</sub></div><div class="equation-explain"><code>share among visible needle pools</code> 的格值。它只比较同一行已经可见的 needles 之间如何分配；<code>absolute mass</code> 则直接显示 A<sub>n,j</sub> 在整条 prompt attention row 中占多少。</div></div>
    </div>
    <p class="formula-note"><strong>具体例子：</strong>第 4 行、第 2 列表示“刚读完 needle 4 时，它的末 token 对 needle 2 的 endpoint 或完整 span 分配了多少 attention”。第 4 行、第 4 列是当前 needle 自身，包含 causal attention 的对角线；第 4 行、第 7 列属于未来 needle，因 causal mask 不可见。这里的 10×10 图是按 needle 聚合后的摘要，不是原始 token×token attention matrix。</p>
  </div>
  <div class="formula">
    <div class="formula-title">分散度与 counter noise</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">N<sub>eff</sub>=exp[−Σ<sub>j≤n</sub> p<sub>j</sub> log p<sub>j</sub>]</div><div class="equation-explain">先令 m<sub>j</sub> 为该 head 对第 j 个历史 needle key pool 的 attention mass，再令 p<sub>j</sub>=m<sub>j</sub>/Σm。N<sub>eff</sub> 是等效覆盖的 needle 数；若均匀覆盖全部 n 个 needles，则 N<sub>eff</sub>=n。</div></div>
      <div class="equation-row"><div class="equation-expression">relative coverage=N<sub>eff</sub>/n</div><div class="equation-explain">把“可见 needle 数随 n 机械增加”除掉；接近 1 表示在当前 n 个 needles 间近似均匀，接近 1/n 表示集中在一个 needle。</div></div>
      <div class="equation-row"><div class="equation-expression">row fraction=exp[H(a)]/K</div><div class="equation-explain">a 是该 head 在整条可见 prompt key axis 上归一化的 attention row，K 是可见 key 数；它衡量对整个 prompt 的扩散，而不是只在 needle pools 内的均匀度。</div></div>
      <div class="equation-row"><div class="equation-expression">counter noise=‖h<sub>s,n</sub>−μ<sub>disc,n</sub>‖₂ / RMS<sub>n</sub>(μ<sub>disc,n</sub>−μ̄)</div><div class="equation-explain">分子是 confirmation seed 的完整 residual 到相同 panel、occurrence 的 discovery centroid 的距离；分母是 discovery 十个 occurrence centroids 的 count-signal RMS。它在 full hidden space 计算，不是 PCA 散点距离。</div></div>
    </div>
  </div>
  <div class="figure-intro"><p><strong>画什么：</strong>纵向选一个 query 时刻 n，横向查看它把 attention 分给哪些 needle key pools j。第 n 行只允许 j≤n，所以矩阵天然是下三角；对角线 j=n 是当前 needle，不是未来位置。</p><p><strong>如何得到：</strong>每个 model×panel×layer×key-pooling 先在 20 个 discovery seeds 中按 mean(total needle mass×relative coverage) 冻结 top-8 heads；热图再先对这 8 个 heads、再对 10 个 confirmation seeds 等权平均。head 选择与作图 seed 严格分开。</p><p><strong>怎样读：</strong>先用 <code>absolute mass</code> 看这些 needle pools 实际拿到整条 attention row 的多少质量，再用 <code>share among visible needle pools</code> 看拿到的这部分质量在 needles 之间是否均匀。每个切换后的视图都用自己的最大格值自动缩放颜色，因此<strong>不能跨模型、层或 pooling 直接比较“谁更亮”</strong>；跨视图比较必须看 hover 数字与右下统计。</p><p><strong>能说明什么：</strong>下方行出现更多非零格只说明绝对覆盖对象增加。只有 needle N<sub>eff</sub> 增加且 relative coverage=N<sub>eff</sub>/n 不下降，才支持“随 n 相对更均匀”。该图仍是 attention 的描述性摘要，不证明它在更新 hidden counter。</p></div>
  <div class="viz-shell">
    <div class="controls">
      <label>Model<select id="pcd-model-select"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
      <label>Hidden-noise / key-mass pairing<select id="pcd-pooling-select"><option value="span_end">span-end hidden + endpoint-key mass</option><option value="span_mean">span-mean hidden + full-span-key mass</option></select></label>
      <label>Post-block layer<select id="pcd-layer-select"></select></label>
      <label>Variant<select id="pcd-variant-select"><option>v4.1</option><option>v4.2</option><option>v4.3</option><option>v4.4</option></select></label>
      <label>Cell value<select id="pcd-metric-select"><option value="mean_attention_mass">absolute mass in full row</option><option value="within_needle_share">share among visible needle pools</option></select></label>
    </div>
    <div class="canvas-wrap"><canvas id="prompt-counter-attention-map" aria-label="Needle-end write-side attention map across prompt occurrences"></canvas><div id="pcd-tooltip"></div></div>
    <div class="viz-foot"><div id="pcd-map-stats"></div><div id="pcd-curve-stats"></div></div>
  </div>
  <p class="figure-caption"><strong>图 B2-F3b · Needle-end query 的 write-side attention map。</strong><strong>纵轴</strong> n=1…10 表示 query 位于“第 n 个 needle 的最后一个 token”；<strong>横轴</strong> j=1…10 表示被读取的第 j 个 needle key pool。格 (n,j) 是 discovery-frozen top-8 bank 与 10 个 confirmation seeds 的等权平均；上三角 j&gt;n 是未来 needle，以灰色叉线标识；对角线 j=n 是当前 needle 自身。<code>absolute mass in full row</code> 保留该 endpoint/full-span key pool 在整条 attention row 中的绝对质量；<code>share among visible needle pools</code> 在平均后再令每行 j≤n 的格值之和为 1，只比较 needle pools 内部的相对分配。Endpoint-key 模式每列只取一个末 token；full-span-key 模式对该列 needle 的全部 tokens 求和。颜色条始终是“0→当前视图最大格值”的自动刻度，跨视图不可直接比颜色。Hover 同时报告 absolute mass、needle-pool share 和该行所有可见 needle pools 的总质量。右下统计给出相同条件下 n=1→10 的 row effective fraction、needle N<sub>eff</sub>、relative coverage 与 full-space counter noise 变化；这些 confirmation 数据不参与 head 选择。</p>
  <div class="figure-intro"><p><strong>画什么：</strong>每层上，top-8 bank 的 relative needle coverage 与 full-space counter noise 在控制 occurrence n 后还剩多少 confirmation-seed 相关。</p><p><strong>如何得到：</strong>在每个 n 内分别减去十个 confirmation seeds 的均值，再把 100 个 seed×occurrence residual pairs 求相关；四条线对应 V4.1–V4.4。棕色虚线为 prompt probe-optimal layer，靛蓝实线为 manifold-display layer。该逐层图是 exploratory localization；表中的两个预定义 layer 才是重点审计位置。</p><p><strong>能说明什么：</strong>正相关表示同一个 n 下 attention 比平均更均匀覆盖 needles 的 seed 也往往有更大的 counter residual；负相关表示更均匀覆盖反而伴随更稳定 counter。即使 CI 不含 0，它仍是同层观察关联，不证明 attention dispersion 导致 hidden noise。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@PROMPT_COUNTER_ASSOCIATION_SVG@@<figcaption><strong>图 B2-F3c · Occurrence-adjusted coverage/noise association across layers。</strong>四格为 Qwen/Gemma×span-end/span-mean；横轴是 zero-based post-block layer，纵轴是 confirmation 中先按 occurrence n 去均值后的 Pearson correlation，范围 −1 到 1，棕色水平线为 0。V4.1–V4.4 分别用靛蓝、紫、青、粉线连接相邻层的点估计；图中不画逐层 CI 以避免遮挡，probe/manifold 两类重点层的 seed-cluster bootstrap 95% CI 在下表给出。正值只表示 seed-level 共变，不是因果方向。</figcaption></figure></div>
  <details><summary>Probe-optimal / manifold-display 层的 dispersion、noise 与关联</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>head scope</th><th>selection</th><th>layer</th><th>panel</th><th>Δ row N<sub>eff</sub></th><th>Δ row effective fraction</th><th>Δ needle N<sub>eff</sub></th><th>Δ relative coverage</th><th>Δ counter noise [95% CI]</th><th>corr(coverage, noise) [95% CI]</th><th>corr(row fraction, noise) [95% CI]</th></tr></thead><tbody>@@PROMPT_COUNTER_DYNAMICS_ROWS@@</tbody></table></div><p class="formula-note"><code>discovery-top8-broad</code> 是每个 model×panel×layer×pooling 仅用 discovery 冻结的 broad bank；<code>all-heads-control</code> 对该层所有 heads 等权平均，检查趋势是否只是 broad-head 选择规则造成。所有 Δ 都是先在每个 confirmation seed 内对 x=(n−1)/9 做线性回归，再等权平均十个 seed slope，因此表示从 n=1 到 n=10 的完整范围变化，而不是“每增加一个 needle”的变化。Row N<sub>eff</sub> 是整条可见 prompt row 的绝对有效 token 数，row effective fraction 再除以可见 key 数；前者随序列变长可机械增加，后者才衡量相对扩散。相关的 bootstrap 以 seed 为 cluster，并在每次重采样后重新做 occurrence 去均值。</p></details>
  @@PROMPT_COUNTER_DYNAMICS_CONCLUSION@@
  <h3>2.3 Answer-query counter：<code>Total:</code> 位置的聚合状态</h3>
  <p class="lede">本轮新增的 <code>answer_query_all_layers_v1</code> capture 对每条 variant×discovery seed×gold count prompt，在生成第一个答案 token 之前、prompt-final <code>Total:</code> query 位置保存<strong>每一个 post-block decoder layer</strong>的完整 residual；Qwen 为 L0–L35，Gemma 为 L0–L41。它不是 needle token 的均值，也不是首个答案 token 生成后的状态。每层同时拟合两个 V4.1 discovery PCA basis：<code>all</code> 使用全部 200 条 prompts；<code>correct_only</code> 只使用最终 greedy 数字严格正确的 prompts。两套 basis 都投影同一批 800 条保存状态。</p>
  <div class="callout"><strong>逐层完整性。</strong>交互图的 layer 下拉框来自新 capture 的实际 <code>layer_indices</code>，不是报告端插值：Qwen 36 层、Gemma 42 层均逐层前向保存。下方静态 PC1–PC2 图为避免 78 个 panel 挤在一页，只显示每个模型的 first / discovery-selected manifold-display / last 三个 landmark；逐层结果必须以上方交互图和导出的 sensitivity CSV 为准。</div>
  <div class="figure-intro"><p><strong>画什么：</strong>先沿 decoder depth 总览每一层的 answer-query count geometry，再进入逐层三维图。四条曲线都只使用 V4.1 discovery prompts；横轴是 zero-based post-block layer，纵轴是定义在该层上的 discovery score（交叉验证 R² 偶尔可低于 0）。</p><p><strong>如何得到：</strong>每层在全部 20 seeds×10 counts 上独立拟合六维 PCA。紫线是在 PC1–3 上用 seed-grouped 五折 Ridge 预测 gold count 的 held-out R²；青线是 PC1–3 对该层全部 prompt 方差的累计解释率；绿线是十个 count centroids 的 full-space 离均差平方和中被 PC1–3 保留的比例；粉线是 <em>C</em>=1/(1+noise/signal)，其中 noise 是同 count leave-one-seed-out residual RMS，signal 是十个 count centroids 相对总 centroid 的 RMS。棕色虚线 P 标记紫线最高层；深靛实线 M 先保留紫线距最高值不超过 0.02 的层，再最大化 M₃=EVR₃×count-signal capture×C。</p><p><strong>能说明什么：</strong>P 回答“三维坐标在哪层最能跨 seed 解码 count”，M 回答“在几乎同等可解码的层里，哪层兼顾方差覆盖、count-signal 覆盖和 seed 紧致度，最适合展示 manifold”。任何单条曲线都不能独立证明因果 counter；它们只是 discovery-only 层选择诊断。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@ANSWER_QUERY_LAYER_SWEEP_SVG@@<figcaption><strong>图 B2-F4-overview · Answer-query 逐层 discovery diagnostics。</strong>左图为 Qwen3-8B L0–L35，右图为 Gemma4-E4B L0–L41；横轴均为 zero-based post-block decoder layer，纵轴为该层的 discovery-only score。紫线=PCA3 seed-grouped CV R²；青线=PC1–3 cumulative EVR；绿线=PC1–3 count-centroid signal capture；粉线=seed compactness C=1/(1+leave-one-seed-out noise-to-signal)。棕色虚线 P 是三维 count probe 的最高-CV层；深靛实线 M 是先通过“CV 距最优不超过 0.02”门槛、再最大化 M₃ 的展示层。线只连接相邻保存层以帮助读取，不表示跨层连续拟合；四个指标共用纵轴但定义不同，必须按上述图注分别解释。</figcaption></figure></div>
  <div class="figure-intro"><p><strong>画什么：</strong>生成答案前，prompt-final <code>Total:</code> query 的完整 residual 在三维 PCA 空间中如何随 gold count 1–10 组织；可切换模型、保存层、V4 panel、PCA 拟合 cohort、最终输出标签与任意 PC1–PC6 轴组合。</p><p><strong>如何得到：</strong><code>all-fit</code> 用 V4.1 的 20 discovery seeds×10 counts 拟合；<code>correct-only-fit</code> 只用其中 strict-correct rows 拟合。切换 panel 或 outcome 时只筛选投影点，不重新拟合。淡点是单 prompt，彩色节点与线是当前筛选后十个 gold-count centroids。</p><p><strong>能说明什么：</strong>它检查 count-conditioned query manifold 是否由错误样本驱动，以及正确、错误、非法输出在同一 basis 中是否分离。PCA 仍是描述性证据；2.5 的 exact donor patch 与第 5 章 steering 才检验该状态是否驱动输出。</p></div>
  <div class="viz-shell">
    <div class="controls">
      <label>Model<select id="aq-model-select"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
      <label>Post-block layer<select id="aq-layer-select"></select></label>
      <label>Variant<select id="aq-variant-select"><option>v4.1</option><option>v4.2</option><option>v4.3</option><option>v4.4</option></select></label>
      <label>PCA fit<select id="aq-fit-select"><option value="all">all V4.1 discovery</option><option value="correct_only">correct-only sensitivity</option></select></label>
      <label>Final output<select id="aq-outcome-select"><option value="all">all outcomes</option><option value="correct">correct only</option><option value="wrong">wrong only</option><option value="invalid">invalid only</option></select></label>
      <label>View<button id="aq-reset-view" type="button">reset rotation</button></label>
      <label>X axis<select id="aq-x-axis"></select></label>
      <label>Y axis<select id="aq-y-axis"></select></label>
      <label>Z axis<select id="aq-z-axis"></select></label>
      <label>Points<select id="aq-points-select"><option value="all">all discovery prompts</option><option value="centroids">centroids only</option></select></label>
      <label>Scale<select id="aq-scale-select"><option value="metric">equal metric scale</option><option value="normalized">normalize each axis</option></select></label>
      <label>Preset<select id="aq-axis-preset"><option value="0,1,2">PC1 / PC2 / PC3</option><option value="0,2,3">PC1 / PC3 / PC4</option><option value="1,2,3">PC2 / PC3 / PC4</option><option value="3,4,5">PC4 / PC5 / PC6</option></select></label>
    </div>
    <div class="canvas-wrap"><canvas id="answer-counter3d" aria-label="Interactive 3D PCA of the answer-query count manifold"></canvas><div id="answer-tooltip"></div></div>
    <div class="viz-foot"><div id="aq-pca-stats"></div><div id="aq-geometry-stats"></div></div>
    <div class="legend" id="aq-count-legend"></div>
  </div>
  <p class="figure-caption"><strong>图 B2-F4a · Interactive answer-query counter manifold。</strong>每个淡点是一条 model×V4 panel×discovery seed×gold count prompt 在首个答案 token 生成前、<code>Total:</code> query 位置的完整 post-block residual；填充色从靛蓝 N=1 过渡到青色 N=10，轮廓色编码最终 greedy outcome：白色=correct、深色=wrong、粉色=invalid。彩色大节点与白线是当前筛选后 N=1→10 centroids；若某个 count 在筛选后没有点，该节点缺失，连线只连接仍存在的相邻显示节点，不是回归曲线。<code>PCA fit</code> 切换 all-fit 与 correct-only-fit；<code>Final output</code> 只切换显示点，不重拟合 PCA。左下角同时列出 fit-cohort EVR 与共同 V4.1 全样本上的 variance capture；右下角给出当前三轴 centroid step CV、path/chord、within-count seed RMS、between-count centroid RMS 及二者比值。每层与每个 fit cohort 独立定轴，故跨 basis 比较应使用这些无坐标符号依赖的摘要，不能直接比较 PC 坐标方向。</p>
  <div class="figure-intro"><p><strong>画什么：</strong>把同一 answer-query 数据固定到 PC1–PC2，形成两个模型×三个保存层的静态审计图，便于不操作 3D 控件也能直接比较 V4.1 与 V4.4。</p><p><strong>如何得到：</strong>每层沿用上方交互图的 V4.1 discovery PCA basis；灰色虚线路径是 V4.1 centroids，黑色实线路径与半透明散点是 V4.4。</p><p><strong>能说明什么：</strong>它提供可打印、固定视角的 cross-layer audit；只显示 PC1–PC2，不能替代上方可切换 PC3–PC6 的三维检查。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@ANSWER_QUERY_COUNTER_SVG@@<figcaption><strong>图 B2-F4b · Static all-fit answer-query PC1–PC2 landmark audit。</strong>每行一个模型，每列依次为 first、discovery-selected manifold-display 与 last layer（实际层号写在各 panel 标题中）；每格横轴是 PC1 score、纵轴是 PC2 score，固定使用该层由全部 V4.1 discovery prompts 拟合的 basis。点填充色编码 gold count，点轮廓为白=correct、深色=wrong、粉=invalid；半透明小点是 V4.4 的 20 seeds×10 counts。灰色虚线连接 V4.1 centroids，黑色实线和彩色节点连接 V4.4 centroids。该图只提供可打印 landmark 审计，不代表只捕获了三层。</figcaption></figure></div>
  <details><summary>All-fit 与 correct-only-fit PCA 敏感性：@@ANSWER_QUERY_SENSITIVITY_ROW_COUNT@@ 个 model×layer×fit rows</summary><p class="lede">下表把两种 basis 都固定评估在同一批 V4.1 全部 200 条 prompts 上。<code>fit EVR</code> 的分母是各自拟合 cohort 的方差，仅描述该 cohort；<code>common capture</code> 才是在共同 V4.1 全样本方差分母下可直接比较的 PC1–3/PC1–6 捕获率。Centroid trajectory 与 seed scatter 也都使用共同评估集。<code>per-count fit n range</code> 明示 correct-only cohort 的类别不平衡；若下界为 0，则该 basis 对缺失 count 的投影是敏感性外推，不是由该 count 的正确样本直接拟合。</p><div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>PCA fit</th><th>fit n</th><th>per-count fit n range</th><th>fit EVR PC1–3</th><th>fit EVR PC1–6</th><th>common capture PC1–3</th><th>common capture PC1–6</th><th>step CV</th><th>path/chord</th><th>within-count seed RMS</th><th>between-centroid RMS</th><th>noise/signal</th><th>centroid distance corr vs all-fit</th></tr></thead><tbody>@@ANSWER_QUERY_PCA_SENSITIVITY_ROWS@@</tbody></table></div><p class="formula-note">Within-count seed RMS 是每个 prompt 到其 gold-count centroid 的三维均方根距离；between-centroid RMS 是十个 count centroids 到其总 centroid 的均方根距离；noise/signal 为前者除以后者，越大表示跨 seed 散布相对 count 间分离越强。最后一列比较十个 centroids 的 45 个两两距离，因此不受 PCA 轴正负号或旋转影响。</p></details>
  @@ANSWER_QUERY_PCA_CONCLUSION@@
  <h3>2.3b Prompt counter 与 answer-query counter 的共同坐标图</h3>
  <div class="concept-box"><span class="concept-label">此处定义 · paired joint state</span><p>对同一 model、V4 panel、discovery seed 与 count <em>k</em>，prompt 端取 N=10 序列读到第 <em>k</em> 个 needle 后的 <code>span_end</code> 或 <code>span_mean</code> state；answer 端取 N=<em>k</em> 序列末尾 <code>Total:</code> query state。两者来自同一个 post-block layer 和同一 residual 坐标系，因此可以联合拟合 PCA；若分别拟合 PCA 后再叠图，轴没有共同含义，本报告不采用那种做法。</p></div>
  <div class="figure-intro"><p><strong>画什么：</strong>同一层中，prompt occurrence 的 1→10 centroid path 与 answer-query 的 1→10 centroid path 是否具有相似的距离结构与递增方向。</p><p><strong>如何得到：</strong>每个 centroid 等权平均 20 个 discovery seeds；<code>raw joint PCA</code> 直接拼接两种 token-role states 拟合，<code>role-centered joint PCA</code> 先分别减去两种 role 在 V4.1 discovery 的 grand mean，再在同一 basis 中拟合。四个 panel 只投影，不重新拟合。圆点/实线为 prompt，方点/虚线为 answer；同色细线连接相同 count。</p><p><strong>能说明什么：</strong>若 role-centered 后两条 trajectory 的 CKA、两两距离相关与相邻 step cosine 都高，说明两处 count geometry 在形状和递增方向上相容；这仍不证明 prompt state 被逐字运输到 answer state，因果运输要由 patching 检验。</p></div>
  <div class="viz-shell">
    <div class="controls">
      <label>Model<select id="joint-model-select"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
      <label>Prompt pooling<select id="joint-pooling-select"><option value="span_end">span-end</option><option value="span_mean">span-mean</option></select></label>
      <label>Post-block layer<select id="joint-layer-select"></select></label>
      <label>Variant<select id="joint-variant-select"><option>v4.1</option><option>v4.2</option><option>v4.3</option><option>v4.4</option></select></label>
      <label>Joint basis<select id="joint-mode-select"><option value="role_centered">role-centered</option><option value="raw">raw</option></select></label>
      <label>View<button id="joint-reset-view" type="button">reset rotation</button></label>
      <label>X axis<select id="joint-x-axis"></select></label>
      <label>Y axis<select id="joint-y-axis"></select></label>
      <label>Z axis<select id="joint-z-axis"></select></label>
    </div>
    <div class="canvas-wrap"><canvas id="joint-counter3d" aria-label="Joint prompt-occurrence and answer-query PCA trajectories"></canvas></div>
    <div class="viz-foot"><div id="joint-pca-stats"></div><div id="joint-geometry-stats"></div></div>
    <div class="legend" id="joint-count-legend"></div>
  </div>
  <p class="figure-caption"><strong>图 B2-F4c · Joint prompt/answer counter geometry。</strong>颜色映射 gold count 1–10；圆点和实线只表示 prompt-reading centroids，方点和虚线只表示 answer-query centroids，黄色半透明细线连接同一 count 的两个 role centroid。所有点都是 20 个 discovery seeds 的均值，图中没有 individual-seed scatter。X/Y/Z 是当前 model×pooling×layer×joint-mode 在 V4.1 discovery 联合拟合的 PC score；每层与两种 joint mode 都独立定轴，不能跨图比较绝对 PC 坐标。Raw 模式可能主要显示固定 token-role offset；role-centered 模式去掉该常量位移后再比较 count trajectory。右下角的 linear CKA 比较 centered Gram geometry，distance corr 比较十个 centroids 的 45 个两两距离，step cosine 比较九个相邻 count 增量在原始 full residual space 中的方向；role-offset/signal 量化 raw token-role 差异相对两条 count trajectory 平均 RMS 尺度有多大。</p>
  <div class="section-conclusion"><span>2.3b 结论边界</span><p>共同 PCA 只回答“两类状态能否在同一个同层坐标系中呈现相容的 count geometry”。Role centering 是预先声明的可视化敏感性处理，不会把 answer state 替换成 prompt state，也不会删除随 count 变化的 role-specific component；若 raw 与 role-centered 结论不同，应解释为固定 token-role offset 对 PCA 方差分配的影响，而不是任选更好看的图。</p></div>
  <div class="section-conclusion"><span>当前结论 · 两种表示不能混称</span><p>Prompt-reading 图追踪同一个 N=10 prompt 内第 1→10 个 needle occurrence 的局部状态；answer-query 图比较十个不同 gold-count prompts 在 <code>Total:</code> 位置的聚合状态。前者说明读入过程中哪些 layer 出现可视的 index trajectory，后者说明生成前哪些 layer 已形成 count-conditioned query geometry。只有后者与 late answer-query donor patching/steering 位于同一干预位置，因此不能用 prompt occurrence PCA 直接替代 answer-query counter 的机制证据。</p></div>
  <details><summary>N=10 trajectory 的实际 greedy outcome strata</summary><p class="lede">一条 N=10 trajectory 的十个 occurrence vectors 共同继承该 prompt 的最终输出标签；不是按单 occurrence 重新分类。Qwen confirmation 在四个 panel 都没有正确 N=10 trajectory；Gemma 只有 v4.1 的 1 条，因此 correct/wrong 几何只能作 audit，不能作有 power 的组间比较。</p><div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>split</th><th>correct / n</th><th>accuracy</th><th>mean prediction</th><th>MAE</th></tr></thead><tbody>@@BEHAVIOR_ROWS@@</tbody></table></div></details>
  <div class="section-conclusion"><span>本节结论</span><p>可切换 PCA 中的 centroid trajectory 证明 count-related geometry 具有低维可视结构，但 individual seed scatter、step CV 与 path/chord 显示它既不完全等距，也不总是笔直。PCA 只能说明 representation 的组织方式；是否进入生成读出，需要后面的 attention 与 causal intervention。</p></div>

  <h3>2.4 Exact needle-end residual patching：可解码 endpoint 是否是可运输的 count carrier？</h3>
  <p>这不是把一个 count prompt 随意贴到另一个 prompt。Stimuli 采用 nested N−1/N pair：两条序列在十个预留 slot 上等长、同位置；第 N 个 slot 在 N prompt 中是 active needle，在 N−1 prompt 中由 canonical token-length-matched haystack control 占位。Insertion 以 N−1 为 receiver，把 N donor 在<strong>第 N 个 slot 的最后一个 token</strong>上的完整 d<sub>model</sub> post-block residual 复制到 receiver 的同位置；removal 反向把 inactive-control endpoint state 复制到 active N receiver。实验只 patch <code>span_end</code>，既不是 span mean，也不是整段 needle tokens。</p>
  <div class="callout"><strong>“有 needle → 没 needle”的精确定义。</strong>这里的“没 needle”指<strong>同一个 toggled slot</strong>在 N−1 prompt 中是等长 inactive haystack control，不是整条 prompt 的 gold count=0；本数据注册范围是 1–10，没有 zero-needle baseline。这个 nested contrast 隔离的是新增一条 evidence 的状态是否充分。</div>
  <p>完成的 protocol 是 <code>cumulative_from_layer</code>：若 start=L18，就在 L18、L19、…、最后一层，每一层都用 donor 在该层保存的 endpoint vector 覆盖 receiver 的同一 endpoint；随后从 receiver prompt 执行完整 greedy generation。这样检验的是“即使该单点状态从某深度起被持续夹持为 donor 值，它是否足以改变答案”。正 aligned shift 要求 insertion 增大生成 count、removal 减小生成 count；<em>moved</em> 还要求最终输出到 donor gold 的距离严格缩短。</p>
  <div class="formula">
    <div class="formula-title">Needle-end cumulative patch 的实际操作</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">h′<sub>r,ℓ,e</sub> = h<sub>d,ℓ,e</sub>, &nbsp; ℓ ≥ L<sub>start</sub></div><div class="equation-explain"><em>e</em> 是 toggled slot 的最后一个 token；从 start layer 到末层，每层都把 receiver endpoint 的完整 d<sub>model</sub> residual 替换为 paired donor 在同层同位置的 residual。</div></div>
      <div class="equation-row"><div class="equation-expression">aligned shift = s · (ŷ′<sub>r</sub> − ŷ<sub>r</sub>)</div><div class="equation-explain">Insertion 时 s=+1，removal 时 s=−1；正值表示最终 strict parsed count 朝 donor gold 的方向移动。</div></div>
    </div>
    <p class="formula-note">没有被替换的是 needle span 的其他 tokens、其他 prompt positions 与 answer-token positions；因此该实验只检验“单 endpoint 状态”的充分性。</p>
  </div>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>start</th><th>rows / seeds</th><th>changed</th><th>moved [95% CI]</th><th>insertion shift</th><th>removal shift</th><th>aligned shift [95% CI]</th><th>Holm p</th></tr></thead><tbody>@@CAUSAL_PATCHING_ROWS@@</tbody></table></div>
  <div class="figure-intro"><p><strong>画什么：</strong>从不同 start layer 起持续替换单个 toggled needle endpoint 后，最终生成数字沿 donor count 方向移动了多少。</p><p><strong>如何得到：</strong>每个 nested pair 同时做 insertion/removal；先在 confirmation seed 内平均 panel、pair 与方向，再对 10 个 seeds bootstrap 95% CI，表中 Holm p 来自同一模型多个 start layers 的 exact sign-flip family。</p><p><strong>能说明什么：</strong>若 CI 明显大于零，单 endpoint residual 可作为可运输的充分 carrier；接近零只否定这一单点、累计夹持 protocol，不否定整段或多 endpoint 的分布式状态。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@CAUSAL_PATCHING_SVG@@<figcaption><strong>图 B2-F5 · Exact needle-end state transport。</strong>纵向每行是一种 model×cumulative start layer；从该层到最后一层，receiver 在 toggled slot 最后一个 token 的完整 residual 都被逐层替换为 paired donor 的同层 residual。横轴是 direction-aligned generated-count shift：insertion 的生成变化取正方向、removal 取反方向，正值表示朝 donor gold 移动，0 表示无方向性运输。紫色圆点为 Qwen、青绿色圆点为 Gemma；圆点是先在每个 confirmation seed 内平均 panel/pair/direction、再对 10 个 seeds 等权得到的估计，粗半透明横线是 seed-cluster bootstrap 95% CI，棕色竖线是零，右侧文字重复 estimate [CI]。所有 CI 跨 0，说明这个单 endpoint、累计夹持 protocol 没有建立充分运输。</figcaption></figure></div>
  <div class="section-conclusion"><span>本小节结论 · Endpoint insufficiency</span><p>所有 tested depths 的 aligned-shift CI 都包含 0，严格 moved rate 最高仅 2.1%，且两个方向都没有一致效应。因此即使从中层/后层起逐层夹持，单个 toggled needle-end vector 仍不足以跨 prompt 搬运一个 +1/−1 count。它虽然高度可解码，却更可能只是局部记录的一部分；该 null 不排除完整 needle token sequence、多个 endpoints 的协调状态，或必须在 <code>Total:</code> query 重新聚合后才成为可执行状态。</p></div>

  <h3>2.5 Exact answer-query residual patching：聚合后的 query state 是否足以搬运模型已经算出的 prediction？</h3>
  <p>这里的 site 与 2.4 完全不同。对同一 panel×seed 的 count pairs 5↔6、7↔8、9↔10、5↔10，先分别保存 donor 与 receiver 在 prompt-final <code>Total:</code> query 的完整 post-block residual。对每个测试层只做一次 <code>single_layer</code> replacement：在 receiver prefill 到达该层后令 h<sub>receiver,query</sub>′=h<sub>donor,query</sub>，其余 prompt positions、其他层和随后生成的 answer-token positions都不 patch；然后从 receiver context 执行最多 16 tokens 的完整 deterministic greedy continuation。这是 sample-wise 全 d<sub>model</sub> 状态替换，不是 PCA coordinate、均值向量或概率比较。</p>
  <p>Primary estimand 只在 receiver 与 donor baseline predictions 不同的 eligible rows 中计算：patched output 是否等于 <em>donor model prediction</em>。它有意不等同于 donor-gold accuracy，因为一个完美 transport 也可以忠实复制 donor 已经算错的数字。越界或不可严格解析的 continuation 留在分母并记为 transport failure。</p>
  <div class="formula">
    <div class="formula-title">Answer-query single-layer donor-state replacement</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">h′<sub>r,ℓ,q</sub> = h<sub>d,ℓ,q</sub></div><div class="equation-explain">只在测试层 ℓ、prompt-final query 位置 q，把 donor 的<strong>完整 sample-wise hidden state</strong>复制给 receiver；其他层和位置保持 receiver 原值。</div></div>
      <div class="equation-row"><div class="equation-expression">adoption = 1[ŷ′<sub>r</sub> = ŷ<sub>d</sub>]</div><div class="equation-explain">只在 receiver 与 donor baseline predictions 不同的 rows 中定义。它检验 patched 输出是否跟随 donor 的<strong>实际模型预测</strong>，而不是 donor gold label。</div></div>
    </div>
    <p class="formula-note">不可解析或超出 1–10 的输出保留在分母并记为 adoption=0；因此 Gemma 的 <code>11</code> 不会被裁剪或当成成功。</p>
  </div>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>rows / seeds</th><th>valid</th><th>eligible n</th><th>adopts donor prediction [95% CI]</th><th>changed (valid)</th><th>moved to donor gold (valid)</th><th>matches donor prediction (all valid)</th><th>aligned shift (valid) [95% CI]</th><th>adoption vs L0 Holm p</th></tr></thead><tbody>@@ANSWER_QUERY_LAYER_ROWS@@</tbody></table></div>
  <div class="figure-intro"><p><strong>画什么：</strong>在不同单层替换 answer-query residual 后，receiver 的最终完整数字答案采用 donor baseline prediction 的比例。</p><p><strong>如何得到：</strong>每模型选择 8 个从早到末层的 post-block layers；每层覆盖四个 panels、10 个 confirmation seeds 与四组双向 count pairs。CI 以 seed 聚类，later-layer adoption 与 L0 做配对检验并 Holm 校正。</p><p><strong>能说明什么：</strong>曲线的跃迁定位“已经算出的 prediction”何时写入可运输 query state；它不证明状态是一维 counter，也不保证只 patch prefill query 就能约束多-token continuation 的每一个后续 token。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@ANSWER_QUERY_ADOPTION_SVG@@<figcaption><strong>图 B2-F6 · Layerwise answer-query donor-prediction transport。</strong>纵向每行是一种 model×single patched layer；该层只在 receiver 的 prompt-final <code>Total:</code> query 位置把完整 residual 替换为 paired donor state，随后从 receiver context 完整 greedy 生成。横轴是在 receiver 与 donor baseline predictions 不同的 eligible rows 中，patched 最终数字严格等于 donor baseline prediction 的比例（0–1）；它衡量复制模型已算出的 prediction，而非 donor-gold accuracy。紫色为 Qwen、青绿色为 Gemma；圆点是 10 个 confirmation seeds 等权估计，粗半透明横线是 seed-cluster bootstrap 95% CI，棕色竖线是 0，右侧文字重复 estimate [CI]。无法解析或生成 1–10 之外数字的 rows 留在分母并按 adoption=0；later-layer 与同模型 L0 的显著性另以 Holm 校正。中后层 adoption 的跃迁表明 prediction 在 late query state 中成为可运输状态。</figcaption></figure></div>
  <details><summary>展开：末层按 V4 panel 与 directed count pair 的稳健性</summary>
    <h4>By V4 panel</h4><div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>panel</th><th>rows / seeds</th><th>valid</th><th>eligible adoption [95% CI]</th><th>aligned shift [95% CI]</th></tr></thead><tbody>@@ANSWER_QUERY_VARIANT_ROWS@@</tbody></table></div>
    <h4>By directed count pair</h4><div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>receiver→donor</th><th>rows / seeds</th><th>valid</th><th>eligible n</th><th>eligible adoption [95% CI]</th><th>follows donor prediction (valid)</th><th>aligned shift [95% CI]</th></tr></thead><tbody>@@ANSWER_QUERY_PAIR_ROWS@@</tbody></table></div>
  </details>
  @@ANSWER_QUERY_INVALID@@
  <div class="callout"><strong>Gemma 的五个 <code>11</code> 是什么？</strong>它们全部来自同一个 V4.1 confirmation family（seed 1263，receiver baseline=5，donor baseline=10），分别出现在 L31/L35/L38/L40/L41。保存的 continuation token IDs 是 receiver <code>5</code>=[236810,106]、donor <code>10</code>=[236770,236771,106]、patched <code>11</code>=[236770,236770,106]。<code>11</code> 能解析为整数，但超出注册答案集合 1–10，因此按 strict rule 记为 invalid/failure，而不是裁剪为 10。一个合理但仍属机制推断的解释是：late query patch 已把 donor <code>10</code> 的首个数字 token 搬入 receiver readout；后续 answer-token positions 没有被 patch，Gemma 在自回归第二步重复了 <code>1</code> 而没有生成 <code>0</code>。这说明单个 prefill query state 对首个 numeric decision 极强，但完整多-token realization 仍包含后续计算。</div>
  <div class="section-conclusion"><span>本小节结论 · Query-state sufficiency</span><p>Transport 在 Qwen L18→L26 与 Gemma L20→L31 之间突然开启；末层所有合法 eligible rows 都等于 donor prediction。把 Gemma 五个生成 <code>11</code> 的 strict-invalid rows 作为 failure 后，保守 adoption 仍为 Qwen 100%、Gemma 99.58%。这证明 late query state 对“模型已经算出的 prediction”高度充分，但不证明它是单维 scalar counter，也不保证多-token answer 的后续 token 都由同一次 patch 决定。</p></div>
  </div>
  <div class="section-conclusion"><span>Block 2 结论</span><p>Prompt 中 needle-end states 保留稳定、可解码的 index/count information；但单 endpoint patch 近乎为零，说明该信息不是一个可单点搬运的运行计数器。相反，<code>Total:</code> 位置的后层完整 residual 可以近确定性搬运 donor prediction，支持“分布式 evidence 先被聚合，再在 answer-query 侧形成 executable count state”。</p></div>
</section>

<section id="attention-representation">
  <span class="section-kicker">Block 3 / 5 · Attention-map representation</span>
  <h2>Answer-query attention：从全 head 图谱到 retrieval phenotype，再到错误关联</h2>
  <p class="lede">分析位置固定为 prompt-final <code>Total:</code> query，所有 attention 都是模型原始 query→prompt rows。Discovery seeds 只用于选 head、定义 phenotype 与排序；correct/wrong、omission 与 nested-increment 诊断只使用 confirmation seeds。主 phenotype 分析采用 <code>span_end</code>，因为它与 prompt-reading counter 的 endpoint 位置及 omission 问题直接对齐；同时完整报告 <code>span_sum</code>（整段真实 attention mass）与 <code>span_mean</code>（按 token 长度归一化的 density），并用固定-head ranking alignment 与 omission sensitivity 检查 endpoint 结论是否依赖 pooling。</p>
  <div class="concept-box"><span class="concept-label">本节先定义 · Full-attention visibility</span><p>只有 key range 能覆盖全部 N=10 needle spans 的 head 才进入全局 atlas 与 phenotype 分析。Qwen 的 36 层均为 full attention；Gemma 只有 L5、L11、L17、L23、L29、L35、L41 是 global-attention layers。Gemma 其余灰色 atlas rows 表示该全局 estimand<strong>不可定义</strong>，不是 attention=0。Layer/head 均 zero-based，例如 L29H3 是第 30 个 block 的第 4 个 head。</p></div>

  <h3>3.1 全 head attention atlas：每层每 head 都放在同一个坐标系中</h3>
  <p>设某个 layer×head 在 prompt-final answer-query 上的 attention row 为 <em>a<sub>t</sub></em>，它对该 head 可见的全部 prompt key tokens 求和约为 1。第 i 个 needle 的 model-token span 是半开区间 S<sub>i</sub>=[s<sub>i</sub>,e<sub>i</sub>)，长度 L<sub>i</sub>=e<sub>i</sub>−s<sub>i</sub>。本报告现在同时保留 endpoint、完整 span 总量与按长度归一化的 span density，三者不能混称。</p>
  <div class="formula">
    <div class="formula-title">每个 occurrence 的三种 attention pooling</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">m<sub>i</sub><sup>end</sup> = a<sub>eᵢ−1</sub></div><div class="equation-explain"><code>span_end</code>：只取该 needle 最后一个 model token 的单个 attention weight。Σm<sup>end</sup> 只是全部 endpoints 的 mass，不是完整 needle spans 的总 mass。</div></div>
      <div class="equation-row"><div class="equation-expression">m<sub>i</sub><sup>sum</sup> = Σ<sub>t=sᵢ..eᵢ−1</sub>a<sub>t</sub></div><div class="equation-explain"><code>span_sum</code>：把该 needle span 的全部 tokens 相加。Needle spans 不重叠时，Σm<sup>sum</sup> 才是整条 query row 落在所有 needle-span tokens 上的真实 attention fraction。</div></div>
      <div class="equation-row"><div class="equation-expression">m<sub>i</sub><sup>mean</sup> = m<sub>i</sub><sup>sum</sup>/L<sub>i</sub></div><div class="equation-explain"><code>span_mean</code>：每 token 平均 attention density，用来控制 tokenizer span 长度；Σm<sup>mean</sup> 是密度之和，不能解释为 query-row mass。</div></div>
      <div class="equation-row"><div class="equation-expression">M<sup>r</sup> = Σ<sub>i=1..N</sub>m<sub>i</sub><sup>r</sup>, &nbsp; p<sub>i</sub><sup>r</sup> = m<sub>i</sub><sup>r</sup>/M<sup>r</sup></div><div class="equation-explain">对选定 pooling r∈{end, mean, sum}，先得到十个 occurrence values，再归一化成 occurrence profile p。不同 r 的 M 语义不同，只有 span-sum 是完整 needle-token mass。</div></div>
      <div class="equation-row"><div class="equation-expression">N<sub>eff,H</sub> = exp(−Σ<sub>i</sub>p<sub>i</sub>log p<sub>i</sub>)</div><div class="equation-explain">Entropy effective number：1 表示近乎单点，N 表示完全均匀覆盖 N 个 needles。</div></div>
      <div class="equation-row"><div class="equation-expression">C<sub>H</sub> = N<sub>eff,H</sub>/N, &nbsp; S = M × C<sub>H</sub></div><div class="equation-explain">C<sub>H</sub> 是 0–1 的 entropy coverage；S 同时奖励总 mass 与覆盖广度。Atlas 显示 log₁₀(S)，不把 S 当成 causal importance。</div></div>
    </div>
    <p class="formula-note">若 M 数值上为 0，则代码约定 C<sub>H</sub>=N<sub>eff,H</sub>=S=0。对 <code>span_sum</code>，pool-sum/coverage/primary/occurrence diagnostics 使用 literal token sum；候选 gate 中与 hard-negative 比较的 contrast/enrichment 仍使用 per-token span-mean density，避免长 span 仅因 token 多而自动获益。</p>
  </div>
  <div class="figure-intro"><p><strong>画什么：</strong>在选定 V4 panel 与 pooling 中，把每个可观测 attention head 放到 layer×head 网格，直接比较 endpoint、per-token density 与 full-span mass 的 retrieval atlas。</p><p><strong>如何得到：</strong>只用 discovery N=10 prompts；颜色为当前 pooling 的 broad-primary score，在每个 model×pooling 内取 log₁₀ 并按 99.5% 分位截断。Phenotype 阈值原本定义在 endpoint profile 上，所以符号只在 <code>span_end</code> 视图叠加。</p><p><strong>能说明什么：</strong>若 span-end 与 span-sum 在同一 layer/head bank 上同时变亮，endpoint 是 full-span retrieval 的良好代理；若分离，则 endpoint 只代表记录边界而非完整 span。跨 pooling 颜色各自缩放，精确对齐应看图下 Spearman/top-8 表。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@ATTENTION_HEAD_ATLAS_HTML@@<figcaption><strong>图 B3-F1 · Switchable all-head retrieval atlas。</strong>顶部第一组按钮切换 V4.1–V4.4，第二组切换 <code>span_end</code>、<code>span_mean</code>、<code>span_sum</code>；上图为 Qwen、下图为 Gemma。横轴是 zero-based attention-head index H，纵轴是 zero-based post-block layer L，每格是一个 LxHy。格底色为当前 pooling 的 log₁₀(S)，深靛低、黄色高；每个 model×pooling 单独缩放，因此可比较同一模型同一 pooling 的 panels，不能按色深直接比较 end 与 sum 的绝对大小。Endpoint 视图中的绿色空心圆=global broad、青色方框=partition-local broad、黄色点=strict first locator、粉色点=weak first-focused；mean/sum 视图不叠加这些 endpoint-defined symbols。Gemma 灰行表示 sliding-local layer 看不见全部远距 needles，estimand 不可定义，并非 attention=0。</figcaption></figure></div>
  <details><summary>Span-end 与 full-span pooling 的全-head对齐</summary><p class="lede"><code>Spearman primary</code> 在所有 full-visibility heads 上比较两种 pooling 的 S 排名；<code>top-8 intersection/Jaccard</code> 比较各自 discovery 排名前八的 head bank。前者衡量全局排序是否对齐，后者衡量 causal screen 会选到多少相同 heads。</p><div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>pooling pair</th><th>heads compared</th><th>Spearman primary</th><th>top-8 overlap</th><th>top-8 Jaccard</th></tr></thead><tbody>@@ATTENTION_POOLING_ALIGNMENT_ROWS@@</tbody></table></div></details>
  @@ATTENTION_POOLING_ALIGNMENT_CONCLUSION@@
  <div class="section-conclusion"><span>本小节结论 · 全局分布</span><p>高 retrieval score 不是单一孤立 head，而是在多个层形成稀疏但重复出现的 bank；同时，颜色高也不等于 broad，因为高 mass 的 selector 也可能排名靠前。因此后续必须把“强度”与“覆盖形状”分开分类。</p></div>

  <h3>3.2 Global broad retrieval heads：哪些 head 同时覆盖多数 needles？</h3>
  <p>这一步不从“最亮的一个 head”出发，而是扫描全部可观察 heads。首先只保留同时满足两项 discovery gate 的候选：needle endpoint attention 高于 matched hard-negative positions，且相对 token-density 的 enrichment&gt;1。然后在 20 个 N=10 discovery prompts 上，计算覆盖宽度、最大单 occurrence share、winner occurrence、四个 normalized-depth quarters 的 row mass 及 full-span profile。阈值在读取 confirmation outcomes 前冻结；这些名称是可复算的 attention-shape 定义，不是先验神经模块标签。</p>
  <div class="formula">
    <div class="formula-title">Phenotype 分类使用的 participation breadth</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">q<sub>i</sub> = m<sub>i</sub>/Σ<sub>j</sub>m<sub>j</sub></div><div class="equation-explain">对每个 prompt，把十个 occurrence masses 归一化为和为 1 的 profile。</div></div>
      <div class="equation-row"><div class="equation-expression">N<sub>eff,2</sub> = 1 / Σ<sub>i</sub>q<sub>i</sub>²</div><div class="equation-explain">Participation effective number 对 dominant occurrence 更敏感：均匀覆盖 k 个 occurrences 时等于 k，集中单点时接近 1。</div></div>
    </div>
    <p class="formula-note">N<sub>eff,2</sub> 用于 global/local/selector phenotype 阈值；3.1 的 N<sub>eff,H</sub> 用于 entropy coverage。二者范围相似但公式和阈值不同，不能互换。</p>
  </div>
  <div class="metric-defs">
    <div class="definition"><strong>Global broad retrieval</strong><p>mean N<sub>eff,2</sub>≥6，且任何单 occurrence 的 mean normalized share≤0.25。含义：至少约六个 endpoints 有实质贡献，且无单点支配。</p></div>
    <div class="definition"><strong>Partition-local broad retrieval</strong><p>不满足 global；dominant depth quarter 平均至少包含 2 个 needles；quarter 内 local effective fraction≥0.80；该 quarter 占整个 query row attention mass≥0.50。含义：head 在局部深度区间内广泛聚合，而非全局覆盖。</p></div>
    <div class="definition"><strong>First-needle locator</strong><p>先满足 selector gate：mean N<sub>eff,2</sub>≤2 且同一 winner occurrence 的频率≥0.80；再要求 winner mode=1、occurrence 1 mean share≥0.80、每 prompt winner 为 first 的比例≥0.90。</p></div>
    <div class="definition"><strong>Weak targeted-selector bucket</strong><p>满足 selector gate 但不满足 strict first-locator；<code>target occurrence</code> 是 20 个 discovery prompts 中 winner 的众数。这个 residual bucket 原本允许 target=2–10，但实际所有入选 rows 的 target 都是 occurrence 1，因此它最终表示“较弱的 first-focused selector”，而不是其他单一 needle retriever。</p></div>
    <div class="definition"><strong>Broad span-mean only</strong><p>Endpoint 上不属于上述类别，但整个 needle-span token mean 的 N<sub>eff,2</sub>≥6 且最大 occurrence share≤0.25。它说明记录内部有广覆盖，不说明 endpoint counter 被广泛读取。</p></div>
    <div class="definition"><strong>Mixed</strong><p>通过候选 gate，但不满足任何强 phenotype 阈值。保留而不强行归类，避免把连续 profile 人为离散化后遗漏。</p></div>
  </div>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>global broad</th><th>local broad</th><th>strict first locator</th><th>weak first-focused selector</th><th>broad span-mean only</th><th>mixed</th><th>all gated candidates</th></tr></thead><tbody>@@ATTENTION_PHENOTYPE_COUNT_ROWS@@</tbody></table></div>
  <p class="artifact-link">机器可读明细：<a href="realistic_niah_v4_head_atlas.csv">all-head atlas（@@ATTENTION_ATLAS_ROW_COUNT@@ model×panel×pooling×layer×head rows）</a>；<a href="realistic_niah_v4_head_phenotypes.csv">gated phenotype profiles（@@ATTENTION_PHENOTYPE_ROW_COUNT@@ rows）</a>。CSV 保留每个 head 的原始层号、排名、mass、coverage、enrichment、target occurrence 与十维 endpoint/span profiles。</p>
  <div class="figure-intro"><p><strong>画什么：</strong>V4.1 中 global broad、local broad、strict first locator 和 weak first-focused bucket 各自最高-primary代表 head 的十维 endpoint profile。</p><p><strong>如何得到：</strong>每个 prompt 内先把十个 endpoint masses 归一化，再跨 20 个 discovery prompts 平均；代表 head 只在对应冻结 phenotype 内按 primary score 选择。</p><p><strong>能说明什么：</strong>线形让“跨多数 needles”“只在一个深度分区内覆盖”和“集中于第一个 needle”可直接区分；代表图用于解释形状，不代表该单 head 最必要。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@ATTENTION_HEAD_PROFILE_SVG@@<figcaption><strong>图 B3-F2 · Representative retrieval profiles（V4.1）。</strong>上排为 Qwen3-8B、下排为 Gemma4-E4B；四列依次为 global broad（绿）、partition-local broad（青）、strict first locator（黄）与 weak first-focused（粉）。每格从该 phenotype 的 V4.1 discovery candidates 中选择 primary score S 最高的一个 head；格顶给出 zero-based LxHy、participation effective number N<sub>eff,2</sub>=1/Σqᵢ²，以及适用时的 selector target。横轴是 needle occurrence index 1–10；纵轴是 answer-query→endpoint attention share：先对每个 N=10 prompt 令 qᵢ=mᵢ/Σmᵢ，再在 20 个 discovery prompts 上平均，因此每条 profile 的十个均值近似和为 1。点是十个 occurrence 的均值，连线只帮助读取相邻位置。最后一列不是 occurrence 2–10 的稳定 targeted retriever，而是未通过 strict first-locator 附加阈值的较弱 first-focused 反例。</figcaption></figure></div>
  <details><summary>展开：每个模型×panel×phenotype 的最高-primary代表 head</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>phenotype</th><th>head</th><th>target</th><th>mean N_eff,2</th><th>endpoint mass</th><th>entropy coverage</th><th>dominant row-quarter mass</th></tr></thead><tbody>@@ATTENTION_REPRESENTATIVE_ROWS@@</tbody></table></div></details>
  <div class="section-conclusion"><span>3.2 结论 · Global broad bank</span><p>Qwen 的 global broad 数量为 34/23/24/22（V4.1→V4.4），其中 14 个 heads 在四个 discovery panels 都保持 global；Gemma 为 16/10/9/11，其中 8 个跨四 panels 稳定。global broad 因此是多个层、多个 heads 组成的 bank，而不是一个孤立 head。释放 position/order/content 后数量下降，但没有消失；这支持稳定的全局 retrieval capacity，尚不等于这些 global heads 作为一类已被单独证明必要。</p></div>

  <h3>3.3 Partition-local broad retrieval heads：是否先分区、再在区内聚合？</h3>
  <p>Local broad head 不要求覆盖全部十个 needles，而要求注意力在一个 normalized-depth quarter 内同时覆盖至少两个 occurrences，并且该 quarter 占完整 query row mass 至少 50%。这是对“Qwen 是否先按深度分区、再在区内 aggregation”的直接操作化。我们仍扫描每个 panel 的全部 discovery-eligible candidates（Qwen 212/226/226/225 heads），而不是只看 top-1/top-8；下面的旧 Qwen high-resolution analysis 与统一 raw-row 重算逐 head 对齐，作为索引、position gating 与分类实现的 replication check。</p>
  <div class="metric-defs">
    <div class="definition"><strong>Global endpoint aggregator</strong><p>endpoint N<sub>eff</sub>≥6，且任何单 occurrence 的 mean normalized share≤0.25。</p></div>
    <div class="definition"><strong>Partition-local endpoint aggregator</strong><p>不是 global；winning depth quartile 内至少含 2 个 needles；local effective fraction≥0.8；整个 query row 至少 50% mass 落在同一 depth quartile。</p></div>
    <div class="definition"><strong>Occurrence endpoint selector</strong><p>endpoint N<sub>eff</sub>≤2，且至少 80% examples 选择相同 occurrence。</p></div>
    <div class="definition"><strong>证据边界</strong><p>这些 phenotype 是行为描述，不是模块标签。一个 head 可有 broad span-mean profile，却在 endpoint 上是 selector；attention profile 也没有包含 value vector 与 output projection。</p></div>
  </div>
  <div class="figure-intro"><p><strong>画什么：</strong>Qwen 在四个 V4 panels 中通过 discovery gate 的 heads 被分为 global broad、partition-local broad、occurrence selector 与其他 profile 后的数量。</p><p><strong>如何得到：</strong>对每个候选使用相同十维 span-end profile、normalized-depth quarter 规则和固定阈值；堆叠高度是 head 数，不按 attention mass 加权。</p><p><strong>能说明什么：</strong>可检验 local phenotype 是否随 position 被释放而稳定存在；数量变化支持/反对固定 partition circuit，但不能说明某类 head 的因果贡献大小。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@PARTITION_PHENOTYPE_SVG@@<figcaption><strong>图 B3-F3 · Qwen discovery-eligible span-end head taxonomy replication。</strong>横轴是 V4.1–V4.4，纵轴是满足 full-attention visibility 与 discovery eligibility 的 Qwen span-end head 数；每根堆叠柱的总数写为 n。绿色 G=global aggregator，青色 L=partition-local aggregator，粉色 S=occurrence selector（含 first-focused），灰色 O=未进入前三类的其他 phenotype；柱下的 “G x · L y · S z · O w” 给出全部四段的精确计数，避免小色块因空间不足而漏标。堆叠高度是计数，不表示效应强度。V4.1 的 local bank 较大，释放 position 后减少，而 global bank 在四个 panels 均存在；该分类只描述 discovery attention pattern，不证明任一类别因果必要。</figcaption></figure></div>
  <details><summary>Phenotype bank coverage：等权覆盖潜力与 raw attention 实际权重</summary><div class="table-wrap"><table><thead><tr><th>panel</th><th>phenotype</th><th>heads</th><th>equal-head N_eff</th><th>raw-mass N_eff</th><th>mean summed endpoint mass</th></tr></thead><tbody>@@PARTITION_BANK_ROWS@@</tbody></table></div></details>
  <div class="callout"><strong>跨 panel×split 稳定的 Qwen global aggregators（13 个）：</strong><code>@@STABLE_GLOBAL_HEADS@@</code>。其中 L6H12 的 endpoint mass 较高；L13H16、L17H22 也是下一轮 bank-specific ablation 的优先候选。这个更严格的 13-head replication set 与只要求四个 discovery panels 稳定的 14-head set 不是同一 estimand。</div>
  @@ATTENTION_CONCLUSION@@

  <div class="section-conclusion"><span>3.3 结论 · Local broad 较弱且对控制释放敏感</span><p>Qwen local broad 数量为 15/7/8/9，四个 discovery panels 间没有同一个 head 始终保持 local；Gemma 为 5/3/2/1，仅 1 个 head 跨四 panels 稳定。Qwen 在全固定 V4.1 中确实有较多“分区后区内聚合”profile，但位置一旦释放就明显减少、转类或换 head。因此目前只能主张局部 aggregation 存在，不能主张一个固定 partition-local circuit 是跨 seed 的核心机制；global broad bank 的稳定性更强。</p></div>

  <h3>3.4 First-needle locator heads：强 selector 主要在找序列的起点</h3>
  <p>Strict first locator 先要求 selector gate（mean N<sub>eff,2</sub>≤2，且至少 80% discovery prompts 的 winner occurrence 相同），再同时要求 winner mode=1、occurrence 1 的 mean normalized share≥0.80、逐 prompt first-winner rate≥0.90。Qwen 在 V4.1→V4.4 分别找到 68/76/79/75 个，61 个 heads 在四个 discovery panels 都保持 strict first locator；Gemma 为 4/2/3/2，其中 2 个稳定。Qwen L29H3 是最清楚的例子：四个 panels 中约 99% endpoint share 都给 occurrence 1；position 被释放后它仍跟随“最早的 needle”，而不是固定绝对 token-depth bin。</p>
  <p>Primary score 排名本身不能识别 broad aggregation，因为它同时奖励总 needle mass 与 entropy coverage。以下 rank-1 audit 专门展示这一反例：Qwen 的最高-primary span-end head 往往就是 first locator，而不是 global aggregator。这里 N<sub>eff,H</sub>=exp(H(p))=10×C<sub>H</sub> 是 entropy effective number；它与分类阈值使用的 participation N<sub>eff,2</sub> 公式不同，不能混用同一阈值。</p>
  <div class="figure-intro"><p><strong>画什么：</strong>每个 model×panel×pooling 的 discovery rank-1 head 实际覆盖多少个 needles。</p><p><strong>如何得到：</strong>先按 primary score S=M×C<sub>H</sub> 选唯一 rank-1，再以 entropy effective number N<sub>eff,H</sub>=exp(H(p)) 汇总十个 occurrences 的覆盖宽度。</p><p><strong>能说明什么：</strong>Qwen span-end rank-1 接近 1，直接否定“最高分 head 必然是 broad aggregator”；这张图是 ranking 反例检查，不估计单 head 必要性。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@ATTENTION_BREADTH_SVG@@<figcaption><strong>图 B3-F4 · Rank-1 head breadth。</strong>左图为 Qwen3-8B、右图为 Gemma4-E4B；横轴是 V4 panel，纵轴是按各 pooling discovery primary score S 排名第一的 head 的 entropy effective number N<sub>eff,H</sub>=exp[−Σpᵢlog pᵢ]（N=10 时范围 1–10）。青色柱=<code>span_end</code>，粉色柱=<code>span_mean</code>，黄色柱=<code>span_sum</code>；柱顶数字是精确 N<sub>eff,H</sub>。1 表示几乎集中于单个 occurrence，10 表示在十个 occurrences 间完全均匀。因为三种 pooling 可选中不同 rank-1 head，柱高差同时包含 pooling profile 与 head selection 差异；“排名第一”也不等于“最 broad”，所以此图不能替代全候选 phenotype 或 end–sum alignment 表。</figcaption></figure></div>
  <details><summary>展开：每个 model×panel×pooling 的 rank-1 head 与指标</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>pooling</th><th>rank-1 head</th><th>pooled M</th><th>coverage</th><th>N_eff,H</th><th>primary</th></tr></thead><tbody>@@ATTENTION_TOP_ROWS@@</tbody></table></div><p class="formula-note"><code>pooled M</code> 的含义随 pooling 而变：end=全部 endpoints mass；sum=全部 needle-span tokens 的真实 row mass；mean=各 span per-token density 之和。</p></details>
  <div class="section-conclusion"><span>3.4 结论 · 起点定位是强而稳定的 phenotype</span><p>尤其在 Qwen 中，first-locator heads 的数量和跨 panel 稳定性都高于 local broad heads。它们可能提供序列边界、扫描起点或归一化锚点，但 attention shape 本身不能区分这些功能，也不表示它们在做逐项加法；其因果角色需要与 global/local banks 分开 ablate。</p></div>

  <h3>3.5 Targeted retrieval heads：除第一个 needle 外，几乎没有稳定的单-occurrence retrieval</h3>
  <p>我们原先保留了一个宽松的 <code>targeted_occurrence_retriever</code> residual bucket：满足 selector gate，但未同时达到 strict first-locator 的 first-share 与 first-win 阈值。该定义允许 target occurrence 为 1–10；然而实际 Qwen 的 85 个 model×panel rows、Gemma 的 6 个 rows，winner 众数<strong>全部是 occurrence 1</strong>。跨四个 discovery panels 稳定的 Qwen 5 个、Gemma 1 个也全部 target=1。换言之，数据没有发现稳定关注 occurrence 2–10 中某一个的 targeted head；所谓 “targeted” 只是较弱、较不一致的 first-focused selector。</p>
  <p>这批 profile 的绝对 endpoint mass 也不强：其 median endpoint pool-sum 约为 Qwen 0.00116、Gemma 0.000536。它们可以在归一化 profile 上显得尖锐，但原始 query row 给 needle endpoints 的总 mass 很小。因此报告不再把它们与 global/local broad 并列解释为独立 retrieval mechanism；保留该 bucket 只是为了完整记录 classifier residual 与负结果。</p>
  <div class="section-conclusion"><span>3.5 结论 · Targeted-other 是负结果</span><p>除 first needle 外，endpoint-defined selector scan 没有证据支持“每个特定 needle 都由某个专属 head 定位”。当前观察到的是 broad banks 与大量 first-focused selectors，而不是十个 occurrence-specific pointers。这个负结果严格限于已捕获 prompts、单-head answer-query attention-weight profiles 与冻结的 endpoint selector gate；span-sum atlas 用于检查完整记录覆盖，但本报告没有另行把它扩张成一套 occurrence-specific selector taxonomy。它也不排除多个 heads 的组合编码，或 value vectors 在相似 attention weight 下携带不同内容。</p></div>

  <h3>3.6 Correct versus wrong 与 undercount omission：错误时究竟差在哪里？</h3>
  <h4>3.6a 同一 gold count 下，wrong prompts 的 retrieval 是否整体更差？</h4>
  <p>该比较只使用 confirmation prompts 和 discovery-ranked top-8 heads，并对 gold count 做显式调整，避免“错误样本本来就集中在大 count”造成伪差异。只有同一 count 内同时存在 correct 与 wrong 样本时才形成 contrast；95% CI 以 confirmation seed 为 cluster 重采样。负的 coverage/min-to-mean 表示错误时注意力在 needles 间更窄或尾部更弱；这仍是 outcome association，不是 causal effect。</p>
  <div class="formula">
    <div class="formula-title">Count-adjusted wrong − correct effect</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">Δ<sub>c</sub> = mean(x | wrong,c) − mean(x | correct,c)</div><div class="equation-explain">先在每个 gold count c 内比较 attention metric x，消除 correct/wrong 两组 count composition 的一阶差异。</div></div>
      <div class="equation-row"><div class="equation-expression">w<sub>c</sub> = 2n<sub>w,c</sub>n<sub>r,c</sub> / (n<sub>w,c</sub>+n<sub>r,c</sub>)</div><div class="equation-explain">Harmonic-overlap 权重；任一组样本很少时自动降权，任一组缺失时该 count 不进入估计。</div></div>
      <div class="equation-row"><div class="equation-expression">Δ = Σ<sub>c</sub>w<sub>c</sub>Δ<sub>c</sub> / Σ<sub>c</sub>w<sub>c</sub></div><div class="equation-explain">最终 cell effect。Δ&lt;0 表示同 count 下 wrong prompts 的该 attention metric 更低。</div></div>
    </div>
  </div>
  <div class="figure-intro"><p><strong>画什么：</strong>在相同 gold-count strata 内，wrong 减 correct 的 entropy coverage，逐 model×panel×discovery-ranked head 展开，并可切换 endpoint、span density 与 full-span sum。</p><p><strong>如何得到：</strong>每种 pooling 独立按 discovery primary 排 top-8；先在每个 count 内计算 wrong−correct，再用两组样本量的 harmonic-overlap 权重合并 counts；95% CI 以 confirmation seed 为 cluster。深框只是单 cell CI，不含 family-wise correction。</p><p><strong>能说明什么：</strong>end 与 sum 同时出现负 cell，支持错误与完整 needle-span retrieval 变窄一致；只有 end 变差则更像 endpoint/boundary channel。它仍是 association，不是 attention 导致错误的因果证据。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@ATTENTION_OUTCOME_EFFECT_SVG@@<figcaption><strong>图 B3-F5 · Count-adjusted wrong−correct attention breadth。</strong>按钮切换 <code>span_end</code>、<code>span_mean</code>、<code>span_sum</code>；横轴四列为 V4.1–V4.4，纵向上半排为 Qwen、下半排为 Gemma，每列从上到下是当前 pooling 的 discovery-ranked heads #1–#8。Cell 值是同一 gold count 内 wrong−correct entropy coverage，再按 correct/wrong harmonic-overlap 样本量合并。粉色为负值（wrong 更窄），绿色为正值（wrong 更广），白色约为 0；黑框表示该 cell 的 seed-cluster bootstrap 95% CI 不含 0，但未作多重比较校正。三种 pooling 的 top-8 bank 可能不同，因此跨 pooling 比较的是“各自最强 bank 的 outcome association”，不是固定 head 的 paired contrast。</figcaption></figure></div>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>metric</th><th>head×panel cells</th><th>CI entirely &lt;0</th><th>CI entirely &gt;0</th><th>median wrong−correct</th><th>range</th></tr></thead><tbody>@@ATTENTION_OUTCOME_SUMMARY_ROWS@@</tbody></table></div>
  <p class="artifact-link">全部 768 个 model×panel×pooling×head×metric effects：<a href="realistic_niah_v4_attention_outcome_effects.csv">realistic_niah_v4_attention_outcome_effects.csv</a>。</p>
  <div class="callout"><strong>Multiplicity boundary。</strong>深色边框只表示单个 head×panel 的 seed-bootstrap CI 不含 0，没有对 32 个 cells 做 family-wise correction；因此这里用于定位错误相关 channels，不把任一单格宣称为预注册 confirmatory discovery。</div>
  @@ATTENTION_OUTCOME_CONCLUSION@@

  <div id="span-end-attention">
  <h4>3.6b Undercount omission：低 attention 是否恰好落在少算的 needles？</h4>
  <p class="lede">预先完成的主分析使用 <code>span_end</code>；现在另加 <code>span_sum</code> 敏感性分析，在完全相同的 undercount prompts 上检验“只看 endpoint”与“看完整 needle span”是否得到一致 omission 结论。每种 pooling 都使用自己在 discovery 排名的 top-8 ensemble；每个 head 先归一化到 occurrence mean share=1 后再等权平均，避免高-mass selector完全淹没 broad heads。</p>
  <div class="method-strip">
    <div><strong>Behavior label</strong>完整 greedy integer output <em>N̂</em>；sequence probability 与 candidate score 均不参与。无法解析或非-undercount rows 不属于该 estimand。</div>
    <div><strong>Held-out unit</strong>10 个 confirmation seeds（1254–1263）；同一 seed 内全部 prompts 保留在同一 resampling cluster。</div>
    <div><strong>Uncertainty</strong>对 seed-level means 做 20,000 次 percentile bootstrap，报告 95% interval。</div>
    <div><strong>Testing</strong>two-sided exact seed sign-flip；两个 pooled model tests 为一个 Holm family，八个 panel-level tests 为另一个。</div>
  </div>

  <h4>3.6b-1 Behavior-implied omitted tail 与 lowest-attention occurrences</h4>
  <p>对 gold count <em>N</em> 和 undercount <em>N̂</em>，令 <em>k=N−N̂</em>。行为上“少算”的尾部集合为 <em>T<sub>k</sub>={N−k+1,…,N}</em>；attention-implied 集合 <em>B<sub>k</sub></em> 是 ensemble attention 最低的 k 个 occurrence endpoints。主分数是两个集合的 overlap fraction。</p>
  <div class="formula">
    <div class="formula-title">Omitted-tail overlap 与组合随机基线</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">S = |B<sub>k</sub> ∩ T<sub>k</sub>| / k</div><div class="equation-explain">B<sub>k</sub> 是 attention 最低的 k 个 endpoints，T<sub>k</sub> 是 behavior 假设下被少算的最后 k 个 needles；S 是两集合的重合比例。</div></div>
      <div class="equation-row"><div class="equation-expression">E[S<sub>chance</sub>] = k/N</div><div class="equation-explain">若 B<sub>k</sub> 是从 N 个 occurrences 中均匀随机选择的 k-subset，期望 overlap fraction 为 k/N。</div></div>
      <div class="equation-row"><div class="equation-expression">P(B<sub>k</sub>=T<sub>k</sub>) = 1 / C(N,k)</div><div class="equation-explain">随机条件下两个 k-subsets 完全相同的概率；报告中的 primary effect 是 seed-equal mean 的 S−k/N。</div></div>
    </div>
  </div>
  <p class="lede">Primary estimand 是 seed-equal mean 的 <em>S−k/N</em>。Cross-panel aggregate 先在每个 seed 内给四个 panel 等权，再跨 seed 推断，避免选择最有利的 relaxation。<em>Tail/prefix</em> 是 omitted tail 的 mean normalized attention 除以 retained prefix；小于 1 表示 tail evidence 被相对抑制。该 omission 分析为 post-hoc inferential audit，并非 preregistered confirmatory test。</p>
  <h4>Cross-panel aggregate</h4>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>seeds</th><th>overlap / chance</th><th>Δ [95% seed CI]</th><th>Holm p</th><th>exact / chance</th><th>exact Δ [95% CI]</th><th>tail / prefix</th></tr></thead><tbody>@@SPAN_END_POOLED_ROWS@@</tbody></table></div>
  <details><summary>Span-end 与 full-span-sum omission sensitivity</summary><p class="lede">下表固定同一批 confirmation undercount prompts，分别按 endpoint value 与完整 span token sum 构造各自 discovery top-8 ensemble。若两行的 overlap−chance 与 tail/prefix 接近，endpoint 是 full-span omission 的可靠代理；若明显分离，则遗漏结论依赖 span 内 attention 的取法。</p><div class="table-wrap"><table><thead><tr><th>model</th><th>pooling</th><th>prompts / seeds</th><th>overlap</th><th>chance</th><th>Δ [95% seed CI]</th><th>exact / chance</th><th>tail / prefix</th></tr></thead><tbody>@@ATTENTION_OMISSION_POOLING_ROWS@@</tbody></table></div></details>
  @@ATTENTION_OMISSION_POOLING_CONCLUSION@@
  <details><summary>Panel-level heterogeneity</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>panel</th><th>prompts / seeds</th><th>mean k</th><th>overlap</th><th>chance k/N</th><th>Δ [95% seed CI]</th><th>Holm p</th><th>exact / chance</th><th>exact Δ [95% CI]</th><th>tail / prefix</th></tr></thead><tbody>@@SPAN_END_ALIGNMENT_ROWS@@</tbody></table></div></details>
  <div class="figure-intro"><p><strong>画什么：</strong>每个 undercount prompt 中，attention 最低的 k=N−N̂ 个 endpoints 与行为假设下“被漏掉的最后 k 个 needles”重合多少，并与相同 N,k 的组合随机基线比较。</p><p><strong>如何得到：</strong>top-8 heads 先各自归一化到 occurrence mean share=1，再等权组成 ensemble；四个 panels 先在 seed 内等权，CI/bootstrap 与 exact sign-flip 都以 10 个 confirmation seeds 为推断单位。</p><p><strong>能说明什么：</strong>observed 明显高于 k/N 表示低-attention 集合并非随机，且与 undercount 的尾部结构对齐；它依赖“顺序尾部就是遗漏集合”的行为假设，因此还不是精确 forgotten-item 读出。</p></div>
  <div class="stat-grid">
    <figure class="stat-figure">@@SPAN_END_ALIGNMENT_SVG@@<figcaption><strong>图 B3-F6 · Omitted-tail overlap。</strong>左图为 Qwen3-8B（紫），右图为 Gemma4-E4B（青绿）；纵向每行依次对应 V4.1–V4.4。对一个 undercount prompt，令 k=gold−prediction，并比较 span-end ensemble attention 最低的 k 个 occurrences 与行为上被少算的最后 k 个 occurrences；横轴是两集合的 overlap/k（0–1，图示范围 0–0.7）。彩色圆点是先在 seed 内平均、再等权跨 confirmation seeds 的 observed overlap；黄色菱形是相同 k/N 下的 hypergeometric chance；灰线只连接 chance 与 observed；粗半透明彩线是 observed−chance 的 seed-cluster bootstrap 95% CI 平移回 overlap 坐标。右侧 Δ=observed−chance。圆点与整个 CI 位于菱形右侧表示低-attention set 对行为上漏掉的尾部有超机会对齐；这是 occurrence-level 关联而非因果。</figcaption></figure>
  </div>

  <p>跨 panel pooled 结果为：Qwen overlap 0.4565、chance 0.2598，Δ=0.1968 [0.1049, 0.2860]；Gemma overlap 0.3636、chance 0.2366，Δ=0.1270 [0.0451, 0.2213]，两个模型的 Holm p 均为 0.015625。Qwen/Gemma 的 omitted-tail attention 相对 retained-prefix 分别只有 0.3348/0.7800。四个单 panel 的点估计方向都为正，但 Holm 校正后各模型仅 V4.1 单 panel 保持显著；因此最稳健的 estimand 是预先声明的 cross-panel aggregate，而不是挑选某个 panel。</p>

  <h4>3.6b-2 Nested N−1→N 中精确新增的 needle</h4>
  <p>Tail 分析把“少算”解释为遗漏后 k 个 occurrences，仍依赖顺序假设。Nested construction 提供更强的 paired check：从 N−1 到 N 新增的 occurrence 精确已知。我们只比较两组最终都仍 undercount 的 transitions：(i) 新 endpoint 是否落在当前 bottom-k attention set；(ii) 新 endpoint 的 normalized share，其中 1 表示在所有 occurrences 间均匀。<em>Failed</em> 表示 output 未增加；<em>registered</em> 表示恰好 +1。先在 seed×panel 内配对，再在 seed 内平均 panels。</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>failed / registered n</th><th>paired blocks / seeds</th><th>bottom-k failed / reg.</th><th>risk Δ F−R [95% CI]</th><th>Holm p</th><th>share failed / reg.</th><th>share Δ R−F [95% CI]</th><th>Holm p</th></tr></thead><tbody>@@SPAN_END_NESTED_ROWS@@</tbody></table></div>
  <div class="figure-intro"><p><strong>画什么：</strong>已知 N−1→N 精确新增的那个 endpoint，在模型未把输出增加 1 与成功增加 1 时，进入 attention bottom-k 的概率。</p><p><strong>如何得到：</strong>只比较两端最终仍 undercount 的 nested transitions，避免简单把 correct 与 wrong 混在一起；先在 seed×panel 内配对 failed/registered，再在 seed 内平均并 bootstrap。</p><p><strong>能说明什么：</strong>failed 组更常把新 needle 放入 bottom-k，直接对齐“哪一个新增 evidence 没有被注册”；但校正后的 p 值决定它目前是 confirmatory 还是方向性支持。</p></div>
  <div class="stat-grid">
    <figure class="stat-figure">@@SPAN_END_NESTED_SVG@@<figcaption><strong>图 B3-F7 · Newly added needle in bottom-k。</strong>纵向两组为 Qwen3-8B 与 Gemma4-E4B；横轴是从 nested N−1→N prompt 新增的 needle endpoint 落入当前 bottom-k attention set 的概率，其中 k=N−最终 prediction。粉色圆点表示最终输出没有随新增 needle 增加 1（failed to increment），绿色圆点表示成功增加 1（registered +1）；每个圆点为 seed-equal风险估计，粗半透明横线为完整 confirmation-seed bootstrap 95% CI。右侧 RD=粉色风险−绿色风险及其 paired 95% CI；RD>0 表示失败增量时新增 needle 更常是低-attention occurrence。两组最终都可能 undercount，所以这是 nested increment-status 比较，不是简单 correct/wrong 对比。</figcaption></figure>
  </div>
  <div class="notes">@@SPAN_END_CONCLUSION@@</div>
  <div class="callout"><strong>Nested 的统计强度。</strong>Qwen bottom-k risk difference 为 0.1582 [−0.0010, 0.3088]，Gemma 为 0.1704 [0.0147, 0.3206]；但两者 exact test 经 Holm 后均为 p=0.1406。Normalized-share contrast 对 Qwen 为 0.0388 [−0.0302, 0.1111]，对 Gemma 为 0.1033 [0.0020, 0.2011]，相应 Holm p 为 0.1758。也就是说，Gemma 的 bootstrap CI 给出方向性较强的支持，但预设 family correction 后仍不能称为显著 confirmatory result。</div>
  <div class="callout"><strong>推断边界。</strong>Tail set 是由 behavior 推断出的“可能遗漏集合”，不是模型内部忘记项的直接记录；nested check 虽然精确知道新增 occurrence，但 attention 与 output 仍是相关。它们为 causal ablation/patching 提供靶点，不能替代干预。</div>
  <div class="section-conclusion"><span>3.6b 结论 · Pooled tail 对齐稳健，exact-new-needle 为方向性支持</span><p>在 confirmation undercounts 中，span-end ensemble 的最低-attention occurrences 与行为上少算的尾部在两个模型都显著高于组合随机基线；这是目前“attention 差在被漏掉 needles 上”的最强 pooled 证据。Nested comparison 不依赖尾部假设，并同样指向 failed increment 时新增 endpoint attention 更差，但 family-wise 校正后未显著。最准确的结论是：evidence omission 与 undercount 在 occurrence level 同步出现，关联既不是全 head shutoff，也尚未达到单独证明因果的程度。</p></div>
  </div>
  <div class="section-conclusion"><span>Block 3 结论</span><p>Answer-query retrieval 由 global broad bank、较不稳定的 partition-local heads 与大量 first-needle locators 共同构成；没有发现 occurrence 2–10 的稳定单-needle targeted heads。错误不是统一关闭全部 attention，而是少数 channels 变窄，并在 pooled undercounts 中把最低 attention 更常分配给行为上少算的 tail；exact-new-needle nested check 方向一致但校正后证据较弱。因而 attention 提供了具体 omission mechanism 的关联靶点，必要性仍需按 phenotype/dose 的 ablation 才能分离。</p></div>
</section>

<section id="head-ablation">
  <span class="section-kicker">Block 4 / 5 · Head ablation</span>
  <h2>Head-bank necessity：删掉 discovery-ranked retrieval heads 是否比同层随机 heads 更容易造成 undercount？</h2>
  <p class="lede">干预只作用于 prompt-final <code>Total:</code> query row，并在 intervention 后执行完整 deterministic greedy generation。每模型使用 160 个 confirmation prompt shards（4 panels×10 seeds×counts 7–10），每 shard 同时保留 baseline、discovery-ranked bank ablation 与 layer-matched random control；所有标签都来自最终 parsed continuation，不使用 candidate probability。</p>
  <div class="formula">
    <div class="formula-title">Ablation 的 paired necessity estimand</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">shift<sub>b</sub> = ŷ<sub>ablated,b</sub> − ŷ<sub>baseline</sub></div><div class="equation-explain">b 是 ranked bank 或 layer-matched random bank；ŷ 是完整 greedy continuation 的 strict parsed count。</div></div>
      <div class="equation-row"><div class="equation-expression">effect = shift<sub>ranked</sub> − shift<sub>random</sub></div><div class="equation-explain">负值表示删 ranked bank 比删相同层、相同数量 random heads 造成更强 undercount，因此支持该<strong>整组 bank</strong>的必要贡献。</div></div>
    </div>
    <p class="formula-note"><strong>Changed</strong> 只表示 ablated prediction 与 baseline 不同；它不区分方向。主推断先在每个 confirmation seed 内平均，再对 10 个 paired seeds bootstrap，并用 exact sign-flip + Holm correction。</p>
  </div>
  <div class="method-strip">
    <div><strong>Selected bank</strong>按 discovery span-end primary score 取 top-4/top-8；不使用 confirmation outcome 重新排序。</div>
    <div><strong>Matched control</strong>随机 heads 与 ranked bank 的 head 数量、所在 layers 完全匹配，用于排除“只要删若干 heads 就会下降”。</div>
    <div><strong>Primary estimand</strong>count shift=(ablated prediction−baseline prediction)；报告 ranked−random paired contrast，负值表示 ranked bank 额外导致 undercount。</div>
    <div><strong>Scope</strong>该 bank 混合 global/local/selector phenotypes；实验检验 bank-level necessity，不识别其中哪一类单独必要。</div>
    <div><strong>Inference unit</strong>point estimate 等权 prompts；seed 内先平均 panels/pairs，再 bootstrap 10 个 paired seeds；primary tests 为 exact sign-flip + family-wise Holm。</div>
  </div>

  <h3>4.1 Discovery-ranked head-bank ablation</h3>
  <p>Primary contrast 是 ranked minus layer-matched random。生成 count shift 定义为 ablated prediction−baseline prediction，因此负 contrast 表示删掉 ranked bank 比删掉同层 random heads 造成更强 undercount。由于高 count baseline-correct prompts 极少，主表按 preregistered screen estimand 合并 correct/wrong；原始 summary/control tables 仍保留 outcome strata。</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>set</th><th>prompts (correct)</th><th>changed ranked / random</th><th>Δ changed [95% CI]</th><th>count shift ranked / random</th><th>Δ count shift [95% CI]</th><th>Holm p</th><th>Δ MAE [95% CI]</th></tr></thead><tbody>@@CAUSAL_ABLATION_ROWS@@</tbody></table></div>
  <div class="figure-intro"><p><strong>画什么：</strong>同时消融 discovery-primary 排名前 k 的 heads，相对消融相同层、相同数量随机 heads 后，生成 count 额外下降多少。</p><p><strong>如何得到：</strong>已完成 k=4 与 k=8；干预只清零这些 heads 在 prompt-final answer-query row 的贡献，随后执行完整 greedy generation。每个 prompt 的 ranked−random contrast 先在 confirmation seed 内平均，再以 10 个 seeds bootstrap 和 exact sign-flip 推断。</p><p><strong>能说明什么：</strong>负效应证明“这个被选中的混合 head bank”对维持输出 magnitude 有因果贡献；因为 bank 同时含 global/local/first-selector，且没有 k=1/2 dose points，它不能证明任一单 head或任一 phenotype 单独必要。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@CAUSAL_ABLATION_SVG@@<figcaption><strong>图 B4-F1 · Ranked mixed head-bank ablation。</strong>纵向每行是一个 model×bank size（top-4 或 top-8）；干预在 prompt-final answer-query row 清零 discovery primary-score 排名靠前的 mixed bank，并与清零相同层、相同 head 数量的随机 bank 配对。横轴是 (ranked ablation 的 generated-count shift)−(layer-matched random ablation 的 shift)；负值表示 ranked bank 相对 control 造成额外 undercount，0 表示两种 bank 无差别。紫色为 Qwen、青绿色为 Gemma；圆点是 10 个 confirmation seeds 等权估计，粗半透明横线是 seed-cluster bootstrap 95% CI，棕色竖线是零，右侧文字重复 estimate [CI]。CI 全在 0 左侧支持所选 mixed bank 的 bank-level necessity，但不能推出 bank 内任一单 head 或单一 phenotype 必要。</figcaption></figure></div>
  <h3>4.2 当前能谈哪一种“必要性”？</h3>
  <p>Top-8 ranked bank 的 ranked−random count-shift contrast 在 Qwen 为 −0.331、Gemma 为 −2.156，两个模型的 Holm p 都为 0.0078。这说明在当前高-count confirmation distribution 和 answer-query-row intervention 下，<strong>被共同删掉的 top-8 mixed bank</strong> 对维持 count magnitude 有可重复的必要贡献。它不等价于以下更强主张：(i) top-8 中每个 head 单独必要；(ii) global broad、local broad 或 first locator 任一 phenotype 单独必要；(iii) top-8 是唯一能完成该功能的 bank；(iv) attention weight 本身而非其 value/output contribution 是因果载体。</p>
  <p>要得到 dose-resolved necessity，下一轮应冻结同一 discovery ranking，运行 nested k=1,2,…,8（最好延伸到 top-16）并为每个 k 构造逐层数量匹配的多组 random controls；同一 prompts 上画 cumulative effect curve 和 marginal Δ(k)=effect(k)−effect(k−1)。再分别对 stable-global、local 与 first-locator banks 做相同扫描，并加入 leave-one-head-out。若某个 k=1 head 的效应稳定超出同层 random 才能讨论单-head necessity；若只在较大 k 出现效应，则证据支持冗余/分布式 bank。</p>
  <div class="section-conclusion"><span>Block 4 结论 · 目前只建立 mixed-bank necessity</span><p>现有 ablation 已足以说明 discovery-ranked top-8 mixed bank 对两模型的 count magnitude 有因果贡献，但不足以把必要性归因给 global broad heads、partition-local heads、first locators 或任何单 head。报告因此不再使用“某种 broad head 必要”这一表述；top1→topk dose scan 与 phenotype-specific matched ablation 是把 bank-level 结论分解为机制结论的必要下一步。</p></div>

  </section>

<section id="geometry-steering">
  <span class="section-kicker">Block 5 / 5 · Geometry steering</span>
  <h2>Answer-query geometry steering：单层或多层 count-centroid 方向能否稳定推动生成 readout？</h2>
  <p class="lede">该 block 检验 full-dimensional <em>directional manipulability</em>，不是 full-state replacement。对每个被干预层，方向都是该层 discovery centroids 的 Δ<sub>ℓ</sub>=μ<sub>ℓ,target</sub>−μ<sub>ℓ,receiver</sub>；它修改全部 d<sub>model</sub> 维，不只修改 PCA coordinates。Single-layer 只在一个 post-block answer-query state 上加 Δ<sub>ℓ</sub>；multi-layer 在同一次 prefill 的多个层分别加各自的 Δ<sub>ℓ</sub>。每次干预后都执行完整 greedy generation，并以最终 strict parsed count 计算 effect。</p>
  <div class="formula">
    <div class="formula-title">三种 full-dimensional residual intervention：本报告不要混称</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">h′<sub>r</sub> = h<sub>d</sub></div><div class="equation-explain"><strong>Donor-state replacement：</strong>搬运某个 donor prompt 的完整 sample state。已在 2.5 的 answer-query patching 中执行。</div></div>
      <div class="equation-row"><div class="equation-expression">h′<sub>r</sub> = μ<sub>target</sub></div><div class="equation-explain"><strong>Centroid transplant：</strong>把 receiver 整体替换为 discovery target-count 的均值状态。本轮<strong>没有运行</strong>，所以不能据此声称“均值完整状态充分”。</div></div>
      <div class="equation-row"><div class="equation-expression">h′<sub>r,ℓ</sub> = h<sub>r,ℓ</sub> + α(μ<sub>ℓ,target</sub>−μ<sub>ℓ,receiver</sub>)</div><div class="equation-explain"><strong>Centroid delta / geometry steering：</strong>保留 receiver 相对自身 centroid 的 residual，只沿 full-dimensional count-centroid 方向平移。本 Block 实际执行的是这一种。</div></div>
    </div>
    <p class="formula-note">Single-layer 只改一个 ℓ；multi-layer 对每个锁定层分别使用该层自己的 μ<sub>ℓ,target</sub>−μ<sub>ℓ,receiver</sub>，不是把同一个向量重复贴到多层。</p>
  </div>
  <div class="formula">
    <div class="formula-title">Steering 的 primary outcome</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">aligned shift = (ŷ′−ŷ<sub>r</sub>) · sign(y<sub>target</sub>−y<sub>receiver</sub>)</div><div class="equation-explain">正值表示最终输出沿 target 方向移动；越界或不可解析输出按 0 记入，而不是删除。</div></div>
      <div class="equation-row"><div class="equation-expression">paired effect = aligned<sub>geometry</sub> − aligned<sub>random</sub></div><div class="equation-explain">Random arm 在同一层使用与 geometry delta 正交、L2 norm 相同的向量；正 effect 才表示 count geometry 优于等范数任意扰动。</div></div>
    </div>
  </div>
  <div class="method-strip">
    <div><strong>Geometry arm</strong><code>centroid_delta</code>：h′<sub>ℓ</sub>=h<sub>ℓ</sub>+α(μ<sub>ℓ,target</sub>−μ<sub>ℓ,receiver</sub>)；每层使用自己的 full-dimensional discovery delta。</div>
    <div><strong>Matched control</strong>每个被干预层使用与该层几何 delta 正交且 L2 norm 相同的 random vector；layer set、α、prompt 与 directed pair 完全匹配。</div>
    <div><strong>Primary estimand</strong>aligned shift=(patched−receiver baseline)×sign(target−receiver)；报告 geometry−random paired difference。</div>
    <div><strong>Not tested</strong><code>centroid_transplant</code> h′=μ<sub>target</sub> 未运行；因此本 block 不回答“均值完整状态是否充分”。</div>
    <div><strong>Strict outcome</strong>越界或无法解析的 generation 不被删除：aligned shift、moved 与 target hit 都记为 0；因此有效率下降会直接惩罚 steering。</div>
  </div>

  <h3>5.1 Initial fixed-layer screen：α=1 的单层方向在哪里开始有效？</h3>
  <p>第一轮 <code>screen_8h_v1</code> 在预先选定的 Qwen L9/L18/L26 与 Gemma L10/L20/L31 上分别运行单层 centroid delta、固定 α=1。Discovery seeds 仅拟合 centroids；四个 panels、10 个 confirmation seeds 与 7↔8/9↔10/5↔10 的双向 pairs 用于干预。它回答“哪一层的方向进入生成 readout”，但因为 layer 与 α 的报告选择可能参考这批 confirmation 结果，所以这里定位为 initial causal screen，不作为最终方案选择的独立验证。</p>
  <div class="callout"><strong>设计边界。</strong>Centroid-delta 保留 receiver 相对自身 count centroid 的 nuisance residual，因此适合检验“count direction 是否与 readout 对齐”；但它不是把 target hidden state 整体搬来。完整 sample donor replacement 已由 2.5 的 answer-query patching 检验；仍未运行的第三臂是 <code>centroid_transplant</code>（h′=μ<sub>target</sub>）。</div>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>pairs / seeds</th><th>changed geom. / random</th><th>moved geom. / random</th><th>Δ moved [95% CI]</th><th>target hit geom. / random</th><th>aligned shift geom. / random</th><th>Δ aligned [95% CI]</th><th>Holm p</th></tr></thead><tbody>@@CAUSAL_STEERING_ROWS@@</tbody></table></div>
  <div class="figure-intro"><p><strong>画什么：</strong>三个固定 depths 中，单层 centroid delta 相对同层等范数正交 random control 的方向对齐 count-shift effect。</p><p><strong>如何得到：</strong>对每个 prompt/pair 先算 geometric−random，再在 confirmation seed 内平均 panel/pair；圆点和 CI 来自 10 个 seeds，模型内三层 exact sign-flip p 值做 Holm 校正。</p><p><strong>能说明什么：</strong>它定位 readout-sensitive depth；后层显著而早层近零说明“可解码”与“被后续生成使用”并不相同，但不能证明方案跨新 seeds 的选择稳定性。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@CAUSAL_STEERING_SVG@@<figcaption><strong>图 B5-F1 · Initial single-layer centroid-delta screen。</strong>纵向每行是一个 model×single patched answer-query layer；geometry arm 在完整 d<sub>model</sub> residual 上加 discovery count-centroid delta μ<sub>target</sub>−μ<sub>receiver</sub>（α=1），control arm 加同层、等 L2 norm 且正交的随机向量。横轴是两臂的 direction-aligned generated-count shift 之差 geometry−random；正值表示几何方向比 matched random 更能把最终 strict parsed count 推向 target，0 表示无额外方向性。紫色为 Qwen、青绿色为 Gemma；圆点为 10 个 confirmation seeds 等权估计，粗半透明横线为 seed-cluster bootstrap 95% CI，棕色竖线为零，右侧文字重复 estimate [CI]。Qwen L26 与 Gemma L31 为明显正值；这是 directional manipulability，不是完整 target hidden-state replacement。</figcaption></figure></div>
  <div class="section-conclusion"><span>5.1 结论 · 初始 screen 支持 late directional manipulability</span><p>Qwen L26 的 geometric−random aligned shift 为 +0.958，Gemma L31 为 +1.388（均 Holm p=0.0117），而 early/middle layers 近零；但 exact target hit 只有 Qwen 8.75%、Gemma 7.5%。所以第一轮只支持“late full-dimensional count direction 能推动输出”，不能支持“α=1 精确设置整数”或“完整 target state 已搬运”。</p></div>

  <h3>5.2 Discovery-locked confirmation：Single-layer 与 Multi-layer steering</h3>
  <p>为避免从第一轮结果中挑 layer 后仍在同一批 confirmation data 上报告，我们新运行两阶段 v2。<strong>Discovery screen</strong> 只用 seeds 1234–1237：Qwen 候选 layer sets 为 9、18、26、18+26、9+18+26；Gemma 为 10、20、31、20+31、10+20+31；每个 set 扫 α∈{0.25,0.5,1}，共 15 个 plans。Single-layer 候选是三个 singleton sets，multi-layer 候选是两个复合集合。</p>
  <p>每种 protocol 只锁定一个 plan。选择分数为四个 V4 panels 中最差的 mean paired strict aligned-shift effect，减去 2×geometry invalid rate；若并列，再按总体 mean effect 较大、α 较小决定。锁定后不再改动，使用完全不相交的 seeds 1254–1263 做 <strong>confirmation</strong>：4 panels×10 seeds×6 directed pairs×2 protocols×2 conditions=960 rows/model。Multi-layer 不是把一个向量重复贴到多层，而是在每个 selected layer 分别施加该层自己的 μ<sub>ℓ,target</sub>−μ<sub>ℓ,receiver</sub>。</p>
  <div class="formula">
    <div class="formula-title">Discovery-only robust plan selection</div>
    <div class="equation-grid">
      <div class="equation-row"><div class="equation-expression">score(plan) = min<sub>v∈V4.1..V4.4</sub>Δ<sub>v</sub> − 2·invalid rate</div><div class="equation-explain">Δ<sub>v</sub> 是该 panel 的 discovery paired strict aligned-shift effect。取最差 panel 防止只在一个 relaxation 上表现好；invalid penalty 防止靠破坏输出格式得到表面位移。</div></div>
    </div>
    <p class="formula-note">每种 protocol 只按该分数锁定一个 plan；若并列，依次选择总体 mean effect 更大、α 更小者。之后不再调整，并在 seeds 1254–1263 上做 held-out confirmation。</p>
  </div>
  <h4>Discovery 选择记录</h4>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>protocol</th><th>locked layers</th><th>α</th><th>candidate plans / seeds</th><th>mean screen Δ</th><th>worst-panel screen Δ</th><th>positive panels</th><th>valid</th><th>robust score</th></tr></thead><tbody>@@STEERING_V2_SELECTION_ROWS@@</tbody></table></div>
  <h4>Held-out confirmation aggregate</h4>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>protocol</th><th>layers</th><th>α</th><th>pairs / seeds</th><th>valid geom. / random</th><th>aligned geom. / random</th><th>paired Δ [95% CI]</th><th>Δ moved [95% CI]</th><th>Δ target hit</th><th>Holm p</th></tr></thead><tbody>@@STEERING_V2_SUMMARY_ROWS@@</tbody></table></div>
  <div class="figure-intro"><p><strong>画什么：</strong>discovery 阶段锁定的一个 single-layer plan 与一个 multi-layer plan，在 10 个独立 confirmation seeds 上相对 matched random control 的 strict aligned-count shift。</p><p><strong>如何得到：</strong>越界输出按零效应保留；每个 seed 内平均四个 panels 与六个 directions，再对 10 个 seeds bootstrap 95% CI。每模型 single/multi 两个 primary tests 做 exact sign-flip 与 Holm correction。</p><p><strong>能说明什么：</strong>CI 与 Holm p 检验方向能否跨新 seeds 复现；single 与 multi 的相对大小说明协调多层干预是否比单层更稳定，但不是对两个 protocols 的直接随机化差异检验。</p></div>
  <div class="stat-grid"><figure class="stat-figure">@@STEERING_V2_SVG@@<figcaption><strong>图 B5-F2 · Discovery-locked single-layer versus multi-layer steering。</strong>纵向每行写明 model、protocol、仅由 discovery screen 锁定的 layer set 与 α；single-layer 只在一个 answer-query layer 加该层的 full-dimensional centroid delta，multi-layer 在同一次 prefill 的多个锁定层分别加各层自己的 delta。横轴是在完全独立的 confirmation seeds 上，strict direction-aligned count shift 的 paired geometry−norm-matched-orthogonal-random effect；正值偏向 target，0 表示 geometry 不优于 control。紫色为 Qwen、青绿色为 Gemma；圆点是四个 panels×六个 directed pairs 先在每个 seed 内平均、再对 10 个 seeds 等权得到的估计，粗半透明横线是 seed-cluster bootstrap 95% CI，棕色竖线是零，右侧文字重复 estimate [CI]。无法解析或超出 1–10 的 generation 不删除，而在两个 primary outcomes 中记为零效应；single 与 multi 的点估计可描述性比较，但不是两 protocol 的直接随机化 contrast。</figcaption></figure></div>
  <details><summary>四个 V4 panels 的 held-out heterogeneity</summary><div class="table-wrap"><table><thead><tr><th>model</th><th>protocol</th><th>panel</th><th>paired rows / seeds</th><th>paired Δ [95% CI]</th><th>panel-family Holm p</th></tr></thead><tbody>@@STEERING_V2_PANEL_ROWS@@</tbody></table></div></details>
  <p class="artifact-link">机器可读表：<a href="realistic_niah_v4_steering_v2_selection.csv">discovery plan selection</a>；<a href="realistic_niah_v4_steering_v2_confirmation.csv">held-out confirmation summary</a>；<a href="realistic_niah_v4_steering_v2_panels.csv">panel heterogeneity</a>。</p>
  <div class="notes">@@STEERING_V2_CONCLUSION@@</div>
  <div class="section-conclusion"><span>5.2 结论 · Single 与 multi 都跨 held-out seeds 复现，但没有 multi 优势证据</span><p>四个 discovery-locked plans 在独立 confirmation seeds 上都保持正 effect：Qwen single L26 为 +1.000、multi L9+18+26 为 +0.992；Gemma single L31 为 +1.371、multi L10+20+31 为 +1.387；四个 overall CI 与各自四个 panel CI 的下界都高于 0，overall Holm p 均为 0.0039。Multi 相对 single 的描述性差只有 Qwen −0.008、Gemma +0.017 count unit，而且本设计没有对 protocol 差做直接随机化检验，因此不能声称 multi-layer 优于 single-layer。Exact target-hit 的净增益仍只有 +5.4 至 +7.5 pp；结论是 late count-centroid 方向可稳定操纵，而不是 target count 被精确设置。完整 donor-state sufficiency 继续由 2.5 的 sample-wise replacement 提供。</p></div>

  <h3>5.3 为什么 early decoding 很强，steering 却可能无效？</h3>
  <p>下表诊断 steering 使用的 10 个 discovery centroids。Endpoint correlation 是 count 与 centroid 在 1→10 chord 上投影的相关；monotonicity 检查该投影是否随 count 单调。Step CV 高表示相邻 count 步长不等；path/chord 高表示曲线弯折。一个方向可以高度可解码，却不一定是后续 readout 使用的 causal coordinate。</p>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>layer</th><th>variants</th><th>endpoint corr. mean (min)</th><th>minimum monotone fraction</th><th>mean step CV</th><th>mean successive-step cosine</th><th>mean tortuosity</th></tr></thead><tbody>@@CAUSAL_GEOMETRY_ROWS@@</tbody></table></div>

  <div class="section-conclusion"><span>当前结论 · Availability versus usage</span><p>Early/middle layers 的 centroid path 可接近单调、R² 很高，却在 steering 下近乎 inert；late path 更弯、更不等距，反而强烈改变 output。信息“可被线性读出”与模型“实际沿该方向生成”必须分开验证。</p></div>

  <h3>5.4 Artifact and label audit</h3>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>ablation shards / rows</th><th>patch families / rows</th><th>steering discovery / families / rows</th><th>greedy-label alignment</th></tr></thead><tbody>@@CAUSAL_AUDIT_ROWS@@</tbody></table></div>
  <div class="callout"><strong>Audit result。</strong>所有 expected shards、detail/summary/control tables 均存在；每个 patch row 成功；discovery NPZ shapes 与 finite values 已核验；causal baseline 与保存的 greedy behavior labels 完全一致；patched correctness 由最终 parsed continuation 重算；logs 中无 Traceback、OOM 或 FAILED marker。</div>
  <h4>Answer-query dense-patching audit</h4>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>family shards / rows</th><th>successful / skipped</th><th>valid / invalid</th><th>eligible donor-prediction rows</th><th>greedy-label alignment</th></tr></thead><tbody>@@ANSWER_QUERY_AUDIT_ROWS@@</tbody></table></div>
  <h4>Steering v2 design and row audit</h4>
  <div class="table-wrap"><table><thead><tr><th>model</th><th>screen shards / rows</th><th>confirmation shards / rows</th><th>screen design</th><th>confirmation design</th></tr></thead><tbody>@@STEERING_V2_AUDIT_ROWS@@</tbody></table></div>
  <div class="section-conclusion"><span>Block 5 结论 · 稳定 directional manipulability 与 state sufficiency 已被分开</span><p>Initial screen 定位 late readout-sensitive layers；v2 再以 discovery-only robust rule 锁定 single/multi plans，并证实四个方案都在独立 confirmation seeds 与全部 V4 panels 上复现。Single 与 multi 几乎同效，当前没有多层干预更优的证据；低 exact-hit 又排除了“均值差方向可精确设定 target”这一强解释。完整 donor state 的 sufficiency 来自 Block 2 的 sample-wise answer-query replacement，centroid transplant 仍是缺失的第三臂。</p></div>
</section>

<div class="report-appendix" id="appendix">
  <span class="section-kicker">Appendix A · Mechanistic synthesis</span>
  <h2>综合机制：哪些解释被支持，哪些解释已经不够，哪些仍无法区分？</h2>
  <div class="evidence-ledger">
    <div class="ledger-row"><div><strong>H1 · Needle-end 存有独立、可直接运输的 running count</strong></div><div><span class="evidence-tag">Not sufficient</span>span-end probe 很强，但 exact endpoint patch 全部接近 null；单 endpoint transport 解释不足。</div></div>
    <div class="ledger-row"><div><strong>H2 · 一个最高排名 broad head 统一汇总所有 needles</strong></div><div><span class="evidence-tag">Rejected for Qwen</span>Qwen L29H3 是 first-occurrence selector；真正 broad coverage 分散在每 panel 22–34 个 global heads。</div></div>
    <div class="ledger-row"><div><strong>H3 · Discovery-ranked mixed head bank 对 count magnitude 必要</strong></div><div><span class="evidence-tag">Supported</span>ranked top-8 bank ablation 相对 layer-matched random 在两个模型都额外造成 undercount；phenotype-specific necessity 仍未分解。</div></div>
    <div class="ledger-row"><div><strong>H4 · Late answer-query residual 携带模型已完成的 count decision</strong></div><div><span class="evidence-tag">Strongly supported</span>single-layer donor patch 在 late layers 近确定性复制 donor prediction，并跨全部 V4 panels/pairs 稳健。</div></div>
    <div class="ledger-row"><div><strong>H5 · Late state 是精确、等距、单维 scalar counter</strong></div><div><span class="evidence-tag">Not established</span>path 弯曲、步长不等；steering exact hit 低；Gemma “11”说明多-token realization 仍依赖后续 computation。</div></div>
    <div class="ledger-row"><div><strong>H6 · Qwen 使用固定的 partition-local aggregation circuit</strong></div><div><span class="evidence-tag">Open</span>存在 local heads，但 phenotype 跨 split/panel 稳定性弱；global bank 更稳定。</div></div>
  </div>
  <div class="section-conclusion"><span>本节结论</span><p>目前最小且不超出证据的机制是：prompt-reading 阶段在 needle spans 保留 count-related local states；多个 answer-query heads 以不同 breadth/selection profiles 聚合这些证据；被测试的 mixed ranked bank 对避免 undercount 有必要贡献；聚合结果在后层 query residual 中变成可直接驱动首个 numeric decision 的 executable state。该链条仍缺少 phenotype-specific ablation，以及从具体 V/O head contributions 到 query residual 的逐层写入分解。</p></div>
</div>

<div class="report-appendix" id="limits">
  <span class="section-kicker">Appendix B · Limits and next discriminating tests</span>
  <h2>目前仍缺什么：下一轮应优先做能区分机制的实验，而不是简单扩大同一 grid</h2>
  <div class="next-grid">
    <div class="next-item"><strong>1. Stable-global bank vs selector bank 的因果分离</strong><p>分别 ablate 13 个跨 panel×split 稳定 global aggregators、L29H3-like selectors、partition-local heads，并使用多组 layer-matched random controls；增加 leave-one-head-out 与 cumulative dose curve，判断必要性来自 broad heads 还是 rank bank 中的混合 phenotype。</p></div>
    <div class="next-item"><strong>2. 从 attention weight 到写入 residual 的路径分解</strong><p>对 priority heads 保存/重构 V、head output 与 O-projection contribution，在 <code>Total:</code> query 做 direct logit/readout alignment 和 sublayer causal tracing，定位 aggregation evidence 在哪一层被 MLP/residual 转成 count decision。</p></div>
    <div class="next-item"><strong>3. Full-needle 与 coordinated multi-endpoint patch</strong><p>单 endpoint null 不能排除分布式 source state。下一步用 exact tokenwise full-needle patch、position-aligned patch，以及多个新增 endpoints 的 coordinated patch；仍避免大规模 head-output patching。</p></div>
    <div class="next-item"><strong>4. Full-state transport 与 steering 的三臂分离</strong><p>在同一 answer-query layer、同一 receiver/donor pairs 上直接比较：sample donor replacement h′=h<sub>d</sub>、centroid transplant h′=μ<sub>target</sub>、centroid delta h′=h+μ<sub>target</sub>−μ<sub>receiver</sub>，并为后两者加入 norm/off-manifold matched controls。随后再对 α、adjacent/non-local pairs、chord/polyline/local tangent 做 dose sweep；这样才能区分“完整状态充分”“均值状态充分”“方向可操纵”和“精确 target-setting”。</p></div>
    <div class="next-item"><strong>5. 两模型 phenotype 的 confirmation stability</strong><p>本报告已用统一 raw-row 规则完成 Qwen 与 Gemma 的 discovery taxonomy；但为了避免 outcome leakage，类别定义没有用 confirmation 重选阈值。下一步应冻结规则后在 confirmation raw rows 上测同一 head 的 phenotype stability，并报告跨 panel×split 的 Jaccard/transition matrix，再决定 phenotype-specific ablation bank。</p></div>
    <div class="next-item"><strong>6. Error-correction 与 thinking-mode generalization</strong><p>高 count correct baselines 太少，correct/wrong causal strata power 不足。可通过调节 length/count 难度获得 matched correct/wrong prompts，再检验 patch 是否纠错；最后扩展到 thinking mode，比较 query state 与 CoT progress state 是否分离。</p></div>
  </div>
  <div class="callout"><strong>不要过度外推。</strong>初始 causal screen 使用 selected pairs、三个 steering depths、α=1、一个 matched random replicate，以及八个 query-patch layers；steering v2 增加 discovery-only layer-set/α selection 与 held-out confirmation，但仍只有一个 matched random replicate。它支持上述具体因果主张，不替代更大、预注册且多-control-replicate 的 full sweep。</div>
  <div class="section-conclusion"><span>本节结论</span><p>优先级最高的是“stable global aggregator bank 的 phenotype-specific ablation”与“head V/O contribution → query residual 的写入路径”，因为它们直接补上当前机制链中唯一缺失的 causal edge；扩大 PCA 或重复更多同类 attention heatmaps 的信息增益较低。</p></div>
</div>

<div class="report-appendix" id="reproducibility">
  <span class="section-kicker">Appendix C · Reproducibility</span>
  <h2>复现、归档与报告 provenance</h2>
  <p>Source run：<code>@@RUN_NAME@@</code>。本地与 Lambda filesystem 均保留完整 run；最终 answer-query bundle 的 SHA-256 为 <code>93776fdea92a07e358d52594969a7ab0d97ad9ef9107ed543d4a7daaa6567920</code>。报告由保存的 behavior labels、representation NPZ、raw answer-query attention rows、causal detail/summary/control tables 重新生成，不依赖服务器内存状态。</p>
  <div class="command-block">PYTHONPATH=src python scripts/build_realistic_niah_v4_representation_report.py --run-root &lt;run-root&gt; --output reports/realistic_niah_v4_representation_report.html --repo-root .</div>
  <p>图表视觉系统固定为 Aurora：Midnight Indigo <code>#23165C</code>、Polar Violet <code>#6750E8</code>、Ice Cyan <code>#00C2FF</code>、Aurora Yellow <code>#F6E36A</code>、Aurora Teal <code>#00D4B4</code>、Aurora Green <code>#39E58C</code>、Polar Magenta <code>#C04DFF</code>、Sunset Pink <code>#FF5FA2</code>；图内背景/网格使用 Snow White 与 Frost Gray。报告正文改用低饱和米白页面、象牙白内容面板与 Warm Brown 边界，使文字层级更清楚；后续 V4+ plots 仍应复用 Aurora palette 和语义映射。</p>
  <div class="section-conclusion"><span>本节结论</span><p>所有图表的数值来源、坐标含义、计算公式、selection split 与 inference unit 都在报告中显式记录；HTML 为 self-contained artifact，可离线打开并复查交互式 3D geometry。</p></div>
</div>
</main>
<footer>生成时间 @@GENERATED@@ · source run <code>@@RUN_NAME@@</code> · commit <code>@@COMMIT@@</code> · neutral academic layout / Aurora figures</footer>
<script>
function tableMetadata(wrap) {
  const table=wrap.querySelector('table');
  if (!table) return {rows:0,columns:[]};
  const rows=table.querySelectorAll('tbody tr').length;
  const columns=Array.from(table.querySelectorAll('thead th')).map(th=>th.textContent.trim()).filter(Boolean);
  return {rows,columns};
}
function makeTablesCollapsible() {
  const groupedDetails=new Set();
  document.querySelectorAll('.table-wrap').forEach((wrap,index)=>{
    const existing=wrap.closest('details');
    if (existing) { groupedDetails.add(existing); return; }
    const meta=tableMetadata(wrap);
    const details=document.createElement('details');
    details.className='table-disclosure';
    details.dataset.tableIndex=String(index+1);
    const summary=document.createElement('summary');
    const columns=meta.columns.slice(0,3).join(' / ')+(meta.columns.length>3?' / …':'');
    summary.textContent=`展开数据表 · ${meta.rows} 行${columns?` · ${columns}`:''}`;
    wrap.parentNode.insertBefore(details,wrap);
    details.append(summary,wrap);
  });
  groupedDetails.forEach(details=>{
    details.removeAttribute('open');
    details.classList.add('table-disclosure');
    const wraps=Array.from(details.querySelectorAll('.table-wrap'));
    const totalRows=wraps.reduce((sum,wrap)=>sum+tableMetadata(wrap).rows,0);
    const summary=Array.from(details.children).find(child=>child.tagName==='SUMMARY');
    if (summary && !summary.dataset.tableMetadata) {
      const suffix=wraps.length>1?` · ${wraps.length} 个表 / ${totalRows} 行`:` · ${totalRows} 行`;
      summary.append(document.createTextNode(suffix));
      summary.dataset.tableMetadata='true';
    }
  });
}
makeTablesCollapsible();
const REP_DATA = @@REP_DATA@@;
const AQ_DATA = @@ANSWER_QUERY_DATA@@;
const JOINT_DATA = @@JOINT_COUNTER_DATA@@;
const PCD_DATA = @@PROMPT_COUNTER_PROFILE_DATA@@;
const COLORS = ['#23165C','#4430A2','#6750E8','#9950F4','#C04DFF','#FF5FA2','#F6E36A','#39E58C','#00D4B4','#00C2FF'];
function drawCountLabels(renderCtx,items,width,height){
  const occupied=[];
  const points=items.map(item=>({x:item.x,y:item.y}));
  const ordered=[...items].sort((a,b)=>((a.count===1||a.count===10)?0:1)-((b.count===1||b.count===10)?0:1)||a.count-b.count);
  const candidates=[[9,-13],[9,13],[-9,-13],[-9,13],[0,-18],[0,18],[16,0],[-16,0]];
  renderCtx.save();renderCtx.font='600 10px system-ui';renderCtx.textBaseline='middle';
  for(const item of ordered){
    const text=String(item.count),textWidth=Math.ceil(renderCtx.measureText(text).width),boxWidth=textWidth+8,boxHeight=16;
    let selected=null;
    for(const [dx,dy] of candidates){
      const box={x:item.x+dx-(dx<0?boxWidth:(dx===0?boxWidth/2:0)),y:item.y+dy-boxHeight/2,w:boxWidth,h:boxHeight};
      const inside=box.x>=3&&box.y>=3&&box.x+box.w<=width-3&&box.y+box.h<=height-3;
      const overlaps=occupied.some(other=>box.x<other.x+other.w+3&&box.x+box.w+3>other.x&&box.y<other.y+other.h+3&&box.y+box.h+3>other.y);
      const coversNode=points.some(point=>(point.x!==item.x||point.y!==item.y)&&point.x>=box.x-3&&point.x<=box.x+box.w+3&&point.y>=box.y-3&&point.y<=box.y+box.h+3);
      if(inside&&!overlaps&&!coversNode){selected=box;break;}
    }
    if(!selected)continue;
    occupied.push(selected);renderCtx.fillStyle='rgba(18,13,49,.84)';renderCtx.fillRect(selected.x,selected.y,selected.w,selected.h);
    renderCtx.fillStyle='#F8FBFF';renderCtx.textAlign='center';renderCtx.fillText(text,selected.x+selected.w/2,selected.y+selected.h/2+.5);
  }
  renderCtx.restore();
}

const pcdCanvas=document.getElementById('prompt-counter-attention-map');
const pcdCtx=pcdCanvas.getContext('2d');
const pcdTooltip=document.getElementById('pcd-tooltip');
const pcdControls={
  model:document.getElementById('pcd-model-select'),pooling:document.getElementById('pcd-pooling-select'),
  layer:document.getElementById('pcd-layer-select'),variant:document.getElementById('pcd-variant-select'),
  metric:document.getElementById('pcd-metric-select')
};
let pcdCells=[];
function pcdAvailableLayers(){
  const prefix=`${pcdControls.model.value}|${pcdControls.pooling.value}|`;
  return [...new Set(Object.keys(PCD_DATA).filter(key=>key.startsWith(prefix)).map(key=>+key.split('|')[2]))].sort((a,b)=>a-b);
}
function pcdRefreshLayers(){
  const layers=pcdAvailableLayers();pcdControls.layer.innerHTML='';
  for(const layer of layers){const rep=REP_DATA[`${pcdControls.model.value}|${pcdControls.pooling.value}|${layer}`],option=document.createElement('option');option.value=String(layer);const role=rep?.manifold_display?' · manifold-display':(rep?.probe_optimal?' · probe-optimal':'');option.textContent=`L${layer}${role}`;pcdControls.layer.appendChild(option);}
  const preferred=layers.find(layer=>REP_DATA[`${pcdControls.model.value}|${pcdControls.pooling.value}|${layer}`]?.manifold_display);if(layers.length)pcdControls.layer.value=String(preferred??layers[0]);
}
function pcdActiveData(){return PCD_DATA[`${pcdControls.model.value}|${pcdControls.pooling.value}|${pcdControls.layer.value}|${pcdControls.variant.value}`];}
function pcdColor(value,maximum){const t=Math.max(0,Math.min(1,value/Math.max(maximum,1e-12))),start=[35,22,92],middle=[103,80,232],end=[0,194,255];let left,right,u;if(t<.5){left=start;right=middle;u=t*2;}else{left=middle;right=end;u=(t-.5)*2;}const rgb=left.map((channel,index)=>Math.round(channel+(right[index]-channel)*u));return`rgb(${rgb.join(',')})`;}
function pcdDraw(){
  const rect=pcdCanvas.getBoundingClientRect(),width=rect.width,height=rect.height;pcdCtx.clearRect(0,0,width,height);pcdCtx.fillStyle='#FFFDF8';pcdCtx.fillRect(0,0,width,height);const data=pcdActiveData();pcdCells=[];
  if(!data){pcdCtx.fillStyle='#765347';pcdCtx.font='16px system-ui';pcdCtx.textAlign='center';pcdCtx.fillText('Prompt-counter dynamics data are unavailable.',width/2,height/2);return;}
  const rawMatrix=data.mean_attention_mass,shareMatrix=data.within_needle_share,matrix=data[pcdControls.metric.value],visible=[];for(let query=0;query<10;query++)for(let key=0;key<=query;key++)visible.push(matrix[query][key]);const maximum=Math.max(...visible,1e-12),cell=Math.min(45,(width-210)/10,(height-170)/10),gridWidth=cell*10,left=Math.max(105,(width-gridWidth)/2),top=88;
  pcdCtx.font='12px system-ui';pcdCtx.textAlign='center';pcdCtx.fillStyle='#161923';pcdCtx.fillText('key needle j · 被当前 query 看向的 needle',left+gridWidth/2,40);pcdCtx.save();pcdCtx.translate(34,top+gridWidth/2);pcdCtx.rotate(-Math.PI/2);pcdCtx.fillText('query at end of needle n · 当前读到第 n 个 needle',0,0);pcdCtx.restore();
  for(let query=0;query<10;query++){
    pcdCtx.fillStyle='#161923';pcdCtx.textAlign='right';pcdCtx.fillText(String(query+1),left-12,top+(query+.66)*cell);
    for(let key=0;key<10;key++){
      const x=left+key*cell,y=top+query*cell,value=matrix[query][key],visibleCell=key<=query,rowNeedleMass=rawMatrix[query].slice(0,query+1).reduce((sum,item)=>sum+item,0);pcdCtx.fillStyle=visibleCell?pcdColor(value,maximum):'#E8E3DA';pcdCtx.fillRect(x,y,cell,cell);pcdCtx.strokeStyle='rgba(129,144,165,.38)';pcdCtx.lineWidth=.7;pcdCtx.strokeRect(x,y,cell,cell);if(!visibleCell){pcdCtx.strokeStyle='rgba(118,83,71,.28)';pcdCtx.beginPath();pcdCtx.moveTo(x+3,y+3);pcdCtx.lineTo(x+cell-3,y+cell-3);pcdCtx.stroke();}pcdCells.push({x,y,w:cell,h:cell,query:query+1,key:key+1,value,rawMass:rawMatrix[query][key],normalizedShare:shareMatrix[query][key],rowNeedleMass,visible:visibleCell});
    }
  }
  pcdCtx.textAlign='center';pcdCtx.fillStyle='#161923';for(let key=0;key<10;key++)pcdCtx.fillText(String(key+1),left+(key+.5)*cell,top-12);
  const legendY=top+gridWidth+46,legendWidth=Math.min(320,gridWidth);for(let index=0;index<100;index++){pcdCtx.fillStyle=pcdColor(maximum*index/99,maximum);pcdCtx.fillRect(left+legendWidth*index/100,legendY,legendWidth/100+1,12);}pcdCtx.fillStyle='#161923';pcdCtx.font='11px system-ui';pcdCtx.textAlign='left';pcdCtx.fillText('0',left,legendY+30);pcdCtx.textAlign='right';pcdCtx.fillText(maximum.toPrecision(3),left+legendWidth,legendY+30);pcdCtx.textAlign='left';pcdCtx.fillText(pcdControls.metric.value==='within_needle_share'?'share among visible needle pools · 当前视图自动刻度':'absolute mass in full attention row · 当前视图自动刻度',left,legendY-8);
  const curves=data.curves,delta=values=>values[9]-values[0];document.getElementById('pcd-map-stats').innerHTML=`<strong>${data.model} · L${data.layer} · ${data.variant}</strong><br>query site = needle end（两种模式固定不变）<br>hidden-noise pairing = ${data.pooling} · key mass = ${data.key_pooling}<br>top-8 bank frozen on discovery · ${data.confirmation_seed_count} confirmation seeds · color autoscaled per view`;
  document.getElementById('pcd-curve-stats').innerHTML=`confirmation n=1→10 mean change:<br>row N<sub>eff</sub> ${delta(curves.row_effective_tokens).toFixed(2)} · row effective fraction ${delta(curves.row_effective_fraction).toFixed(4)} · needle N<sub>eff</sub> ${delta(curves.needle_effective_number).toFixed(3)} · relative coverage ${delta(curves.needle_relative_coverage).toFixed(3)} · counter noise ${delta(curves.counter_noise).toFixed(3)}`;
}
function pcdResize(){const rect=pcdCanvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);pcdCanvas.width=Math.max(1,Math.round(rect.width*dpr));pcdCanvas.height=Math.max(1,Math.round(rect.height*dpr));pcdCtx.setTransform(dpr,0,0,dpr,0,0);pcdDraw();}
[pcdControls.model,pcdControls.pooling].forEach(control=>control.addEventListener('change',()=>{pcdRefreshLayers();pcdDraw();}));[pcdControls.layer,pcdControls.variant,pcdControls.metric].forEach(control=>control.addEventListener('change',pcdDraw));
pcdCanvas.addEventListener('pointermove',event=>{const rect=pcdCanvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,cell=pcdCells.find(item=>x>=item.x&&x<=item.x+item.w&&y>=item.y&&y<=item.y+item.h);if(cell){pcdTooltip.style.display='block';pcdTooltip.style.left=`${Math.max(8,Math.min(rect.width-370,x+14))}px`;pcdTooltip.style.top=`${Math.max(8,y-10)}px`;const keyLabel=pcdActiveData()?.key_pooling==='needle_end'?'needle endpoint token':'all tokens in the needle span';pcdTooltip.innerHTML=cell.visible?`<strong>第 ${cell.query} 个 needle 末 token → 第 ${cell.key} 个 needle</strong><br>key pool: ${keyLabel}<br>absolute mass in full row: ${cell.rawMass.toPrecision(5)}<br>share among visible needle pools: ${cell.normalizedShare.toPrecision(5)}<br>all visible needle pools in this row: ${cell.rowNeedleMass.toPrecision(5)}<br>${cell.query===cell.key?'diagonal = current needle itself':'off-diagonal = an earlier needle'}`:`<strong>query n=${cell.query}, key j=${cell.key}</strong><br>future needle：被 causal mask 屏蔽`;}else pcdTooltip.style.display='none';});pcdCanvas.addEventListener('mouseleave',()=>{pcdTooltip.style.display='none';});
const canvas = document.getElementById('counter3d');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const controls = {
  model: document.getElementById('model-select'), pooling: document.getElementById('pooling-select'),
  layer: document.getElementById('layer-select'),
  variant: document.getElementById('variant-select'), split: document.getElementById('split-select'),
  outcome: document.getElementById('outcome-select'), points: document.getElementById('points-select'),
  scale: document.getElementById('scale-select'), x: document.getElementById('x-axis'),
  y: document.getElementById('y-axis'), z: document.getElementById('z-axis'),
  preset: document.getElementById('axis-preset')
};
for (const select of [controls.x, controls.y, controls.z]) {
  for (let i=0;i<6;i++) { const o=document.createElement('option'); o.value=i; o.textContent=`PC${i+1}`; select.appendChild(o); }
}
controls.x.value='0'; controls.y.value='1'; controls.z.value='2';
let yaw=-0.72, pitch=0.44, zoom=1.0, dragging=false, lastX=0, lastY=0, projectedPoints=[];

function availableLayers() {
  const prefix=`${controls.model.value}|${controls.pooling.value}|`;
  return Object.keys(REP_DATA).filter(key=>key.startsWith(prefix)).map(key=>+key.slice(prefix.length)).sort((a,b)=>a-b);
}
function refreshLayerOptions() {
  const layers=availableLayers(); controls.layer.innerHTML='';
  for (const layer of layers) {
    const data=REP_DATA[`${controls.model.value}|${controls.pooling.value}|${layer}`];
    const option=document.createElement('option'); option.value=String(layer);
    const role=data.manifold_display?' · manifold-display':(data.probe_optimal?' · probe-optimal':'');
    option.textContent=`L${layer}${role}`; controls.layer.appendChild(option);
  }
  const defaultLayer=layers.find(layer=>REP_DATA[`${controls.model.value}|${controls.pooling.value}|${layer}`].manifold_display);
  controls.layer.value=String(defaultLayer??layers[0]);
}
function activeData() { return REP_DATA[`${controls.model.value}|${controls.pooling.value}|${controls.layer.value}`]; }
function filteredRows() {
  const data=activeData(); if (!data) return [];
  return data.rows.filter(r => r[0]===controls.variant.value && (controls.split.value==='all'||r[2]===controls.split.value) && (controls.outcome.value==='all'||r[3]===controls.outcome.value));
}
function resizeCanvas() {
  const rect=canvas.getBoundingClientRect(), dpr=Math.min(window.devicePixelRatio||1,2);
  canvas.width=Math.max(1,Math.round(rect.width*dpr)); canvas.height=Math.max(1,Math.round(rect.height*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0); draw();
}
function statsFor(rows, axes) {
  if (!rows.length) return null;
  const vals=axes.map(a=>rows.map(r=>r[7+a]));
  const mins=vals.map(v=>Math.min(...v)), maxs=vals.map(v=>Math.max(...v));
  const centers=mins.map((m,i)=>(m+maxs[i])/2), ranges=mins.map((m,i)=>Math.max(maxs[i]-m,1e-8));
  return {mins,maxs,centers,ranges};
}
function makeTransform(rows, axes, width, height) {
  const s=statsFor(rows,axes); if (!s) return null;
  const perAxis=controls.scale.value==='normalized';
  const common=Math.max(...s.ranges); const scales=s.ranges.map(r=>perAxis?1/r:1/common);
  const radius=Math.min(width,height)*0.36*zoom;
  return p=>{
    let x=(p[0]-s.centers[0])*scales[0]*2, y=(p[1]-s.centers[1])*scales[1]*2, z=(p[2]-s.centers[2])*scales[2]*2;
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    const x1=cy*x+sy*z, z1=-sy*x+cy*z, y1=cp*y-sp*z1, z2=sp*y+cp*z1;
    return {x:width/2+x1*radius,y:height/2-y1*radius,z:z2,raw:p};
  };
}
function centroids(rows) {
  const groups=new Map();
  for (const r of rows) { const key=r[2]; if (!groups.has(key)) groups.set(key,new Map()); const byCount=groups.get(key); if (!byCount.has(r[6])) byCount.set(r[6],[]); byCount.get(r[6]).push(r); }
  const result=[];
  for (const [split,byCount] of groups.entries()) {
    const path=[];
    for (let count=1;count<=10;count++) { const rs=byCount.get(count)||[]; if (!rs.length) continue; const p=[]; for(let pc=0;pc<6;pc++) p.push(rs.reduce((a,r)=>a+r[7+pc],0)/rs.length); path.push({count,p,n:rs.length}); }
    result.push({split,path});
  }
  return result;
}
function geometryText(paths, axes) {
  if (!paths.length) return 'No centroid path for this filter.';
  return paths.map(group=>{
    const p=group.path.map(d=>axes.map(a=>d.p[a])); if(p.length<2) return `${group.split}: insufficient points`;
    const steps=[]; for(let i=1;i<p.length;i++) steps.push(Math.hypot(...p[i].map((v,j)=>v-p[i-1][j])));
    const mean=steps.reduce((a,b)=>a+b,0)/steps.length; const sd=Math.sqrt(steps.reduce((a,b)=>a+(b-mean)**2,0)/steps.length); const chord=Math.hypot(...p[p.length-1].map((v,j)=>v-p[0][j]));
    const path=steps.reduce((a,b)=>a+b,0); return `${group.split}: step CV ${(sd/Math.max(mean,1e-9)).toFixed(2)} · path/chord ${(path/Math.max(chord,1e-9)).toFixed(2)}`;
  }).join('<br>');
}
function drawAxes(transform, stats, axes, width, height) {
  const origin=[stats.mins[0],stats.mins[1],stats.mins[2]], ends=[[stats.maxs[0],origin[1],origin[2]],[origin[0],stats.maxs[1],origin[2]],[origin[0],origin[1],stats.maxs[2]]];
  const o=transform(origin); ctx.lineWidth=1; ctx.font='11px system-ui';
  ends.forEach((end,i)=>{ const e=transform(end); ctx.strokeStyle=['#00C2FF','#39E58C','#FF5FA2'][i]; ctx.beginPath();ctx.moveTo(o.x,o.y);ctx.lineTo(e.x,e.y);ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText(`PC${axes[i]+1}`,e.x+4,e.y-4); });
}
function draw() {
  const rect=canvas.getBoundingClientRect(), width=rect.width, height=rect.height;
  ctx.clearRect(0,0,width,height); ctx.fillStyle='#120D31'; ctx.fillRect(0,0,width,height);
  const rows=filteredRows(), axes=[+controls.x.value,+controls.y.value,+controls.z.value];
  const data=activeData(); const role=data?(data.manifold_display?'manifold-display':(data.probe_optimal?'probe-optimal':'layer sweep')):''; document.getElementById('pca-stats').innerHTML=data?`<strong>${data.model} · ${data.pooling} · L${data.layer} · ${role}</strong><br>PCA fit: v4.1 discovery · EVR ${data.explained_variance_ratio.slice(0,6).map((v,i)=>`PC${i+1} ${(100*v).toFixed(1)}%`).join(' · ')} · M₃ ${data.manifold_fidelity_m3.toFixed(3)}`:'';
  const stats=statsFor(rows,axes), transform=makeTransform(rows,axes,width,height); projectedPoints=[];
  if (!rows.length || !stats || !transform) { ctx.fillStyle='#F6E36A';ctx.font='16px system-ui';ctx.textAlign='center';ctx.fillText('No trajectories match this filter.',width/2,height/2);document.getElementById('geometry-stats').textContent='No data';return; }
  drawAxes(transform,stats,axes,width,height);
  const paths=centroids(rows); const pointMode=controls.points.value;
  const labelGroup=paths.find(group=>group.split==='confirmation')||paths[0]; let centroidLabels=[];
  for (const group of paths) {
    const pts=group.path.map(d=>({...d,q:transform(axes.map(a=>d.p[a]))}));
    ctx.strokeStyle=group.split==='confirmation'?'#F8FBFF':'#8190A5'; ctx.lineWidth=group.split==='confirmation'?2.5:1.5; ctx.setLineDash(group.split==='confirmation'?[]:[6,5]);
    ctx.beginPath(); pts.forEach((d,i)=>i?ctx.lineTo(d.q.x,d.q.y):ctx.moveTo(d.q.x,d.q.y));ctx.stroke();ctx.setLineDash([]);
    for (const d of pts) { ctx.fillStyle=COLORS[d.count-1];ctx.strokeStyle='#161923';ctx.lineWidth=1;ctx.beginPath();ctx.arc(d.q.x,d.q.y,5.6,0,Math.PI*2);ctx.fill();ctx.stroke(); }
    if(group===labelGroup)centroidLabels=pts.map(d=>({count:d.count,x:d.q.x,y:d.q.y}));
  }
  drawCountLabels(ctx,centroidLabels,width,height);
  if (pointMode!=='centroids') {
    let pointRows=rows; if(pointMode==='confirmation') pointRows=rows.filter(r=>r[2]==='confirmation');
    const pts=pointRows.map(r=>({r,q:transform(axes.map(a=>r[7+a]))})).sort((a,b)=>a.q.z-b.q.z);
    for(const item of pts){const r=item.r,q=item.q;ctx.globalAlpha=r[2]==='confirmation'?.56:.18;ctx.fillStyle=COLORS[r[6]-1];ctx.strokeStyle=r[3]==='correct'?'#F8FBFF':(r[3]==='invalid'?'#FF5FA2':'#161923');ctx.lineWidth=r[3]==='correct'?1.8:.7;ctx.beginPath();ctx.arc(q.x,q.y,r[2]==='confirmation'?3.0:2.2,0,Math.PI*2);ctx.fill();ctx.stroke();projectedPoints.push({x:q.x,y:q.y,r});} ctx.globalAlpha=1;
  }
  ctx.fillStyle='#8190A5';ctx.font='11px system-ui';ctx.textAlign='left';ctx.fillText(`${rows.length} occurrence points · ${new Set(rows.map(r=>r[1])).size} seeds`,12,height-12);
  document.getElementById('geometry-stats').innerHTML=geometryText(paths,axes);
}
function reset(){yaw=-0.72;pitch=.44;zoom=1;draw();}
[controls.model,controls.pooling].forEach(el=>el.addEventListener('change',()=>{refreshLayerOptions();draw();}));
[controls.layer,controls.variant,controls.split,controls.outcome,controls.points,controls.scale,controls.x,controls.y,controls.z].forEach(el=>el.addEventListener('change',draw));
controls.preset.addEventListener('change',()=>{const a=controls.preset.value.split(',');controls.x.value=a[0];controls.y.value=a[1];controls.z.value=a[2];draw();});
document.getElementById('reset-view').addEventListener('click',reset);
canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(dragging){yaw+=(e.clientX-lastX)*.008;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-lastY)*.008));lastX=e.clientX;lastY=e.clientY;draw();return;} const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=null,dist=Infinity;for(const p of projectedPoints){const d=(p.x-x)**2+(p.y-y)**2;if(d<dist){dist=d;best=p;}}if(best&&dist<80){const r=best.r;tooltip.style.display='block';tooltip.style.left=`${Math.min(rect.width-250,x+14)}px`;tooltip.style.top=`${Math.max(8,y-10)}px`;tooltip.innerHTML=`<strong>${r[0]} · seed ${r[1]} · index ${r[6]}</strong><br>${r[2]} · output ${r[3]} · predicted ${r[4]??'invalid'} · error ${r[5]??'—'}`;}else tooltip.style.display='none';});
canvas.addEventListener('pointerup',()=>{dragging=false;canvas.classList.remove('dragging');}); canvas.addEventListener('pointercancel',()=>{dragging=false;canvas.classList.remove('dragging');}); canvas.addEventListener('mouseleave',()=>{tooltip.style.display='none';});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.45,Math.min(2.8,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
document.getElementById('count-legend').innerHTML=COLORS.map((c,i)=>`<span><i style="background:${c}"></i>${i+1}</span>`).join('');

const aqCanvas=document.getElementById('answer-counter3d');
const aqCtx=aqCanvas.getContext('2d');
const aqTooltip=document.getElementById('answer-tooltip');
const aqControls={
  model:document.getElementById('aq-model-select'), layer:document.getElementById('aq-layer-select'),
  variant:document.getElementById('aq-variant-select'), fit:document.getElementById('aq-fit-select'),
  outcome:document.getElementById('aq-outcome-select'), points:document.getElementById('aq-points-select'),
  scale:document.getElementById('aq-scale-select'), x:document.getElementById('aq-x-axis'),
  y:document.getElementById('aq-y-axis'), z:document.getElementById('aq-z-axis'),
  preset:document.getElementById('aq-axis-preset')
};
for(const select of [aqControls.x,aqControls.y,aqControls.z]){
  for(let i=0;i<6;i++){const option=document.createElement('option');option.value=String(i);option.textContent=`PC${i+1}`;select.appendChild(option);}
}
aqControls.x.value='0';aqControls.y.value='1';aqControls.z.value='2';
let aqYaw=-.72,aqPitch=.44,aqZoom=1,aqDragging=false,aqLastX=0,aqLastY=0,aqProjectedPoints=[];
function aqAvailableLayers(){
  const prefix=`${aqControls.model.value}|`;
  return [...new Set(Object.keys(AQ_DATA).filter(key=>key.startsWith(prefix)).map(key=>+key.split('|')[1]))].sort((a,b)=>a-b);
}
function aqRefreshLayerOptions(){
  const layers=aqAvailableLayers();aqControls.layer.innerHTML='';
  layers.forEach(layer=>{const data=AQ_DATA[`${aqControls.model.value}|${layer}|all`],option=document.createElement('option');option.value=String(layer);const role=data?.manifold_display?' · manifold-display':(data?.probe_optimal?' · PCA3-probe':'');option.textContent=`L${layer}${role}`;aqControls.layer.appendChild(option);});
  const preferred=layers.find(layer=>AQ_DATA[`${aqControls.model.value}|${layer}|all`]?.manifold_display);
  if(layers.length)aqControls.layer.value=String(preferred??layers[layers.length-1]);
}
function aqActiveData(){return AQ_DATA[`${aqControls.model.value}|${aqControls.layer.value}|${aqControls.fit.value}`];}
function aqFilteredRows(){const data=aqActiveData();return data?data.rows.filter(row=>row[0]===aqControls.variant.value&&(aqControls.outcome.value==='all'||row[3]===aqControls.outcome.value)):[];}
function aqStatsFor(rows,axes){
  if(!rows.length)return null;
  const values=axes.map(axis=>rows.map(row=>row[6+axis]));
  const mins=values.map(v=>Math.min(...v)),maxs=values.map(v=>Math.max(...v));
  return {mins,maxs,centers:mins.map((v,i)=>(v+maxs[i])/2),ranges:mins.map((v,i)=>Math.max(maxs[i]-v,1e-8))};
}
function aqMakeTransform(rows,axes,width,height){
  const stats=aqStatsFor(rows,axes);if(!stats)return null;
  const perAxis=aqControls.scale.value==='normalized',common=Math.max(...stats.ranges),scales=stats.ranges.map(range=>perAxis?1/range:1/common),radius=Math.min(width,height)*.36*aqZoom;
  return point=>{let x=(point[0]-stats.centers[0])*scales[0]*2,y=(point[1]-stats.centers[1])*scales[1]*2,z=(point[2]-stats.centers[2])*scales[2]*2;const cy=Math.cos(aqYaw),sy=Math.sin(aqYaw),cp=Math.cos(aqPitch),sp=Math.sin(aqPitch);const x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;return{x:width/2+x1*radius,y:height/2-y1*radius,z:z2};};
}
function aqCentroids(rows){
  const byCount=new Map();for(const row of rows){if(!byCount.has(row[2]))byCount.set(row[2],[]);byCount.get(row[2]).push(row);}
  const path=[];for(let count=1;count<=10;count++){const group=byCount.get(count)||[];if(!group.length)continue;const point=[];for(let pc=0;pc<6;pc++)point.push(group.reduce((sum,row)=>sum+row[6+pc],0)/group.length);path.push({count,point,n:group.length});}return path;
}
function aqGeometryText(path,rows,axes){
  const points=path.map(item=>axes.map(axis=>item.point[axis]));if(points.length<2)return 'centroid path: insufficient points';
  const steps=[];for(let i=1;i<points.length;i++)steps.push(Math.hypot(...points[i].map((value,j)=>value-points[i-1][j])));
  const mean=steps.reduce((a,b)=>a+b,0)/steps.length,sd=Math.sqrt(steps.reduce((a,b)=>a+(b-mean)**2,0)/steps.length),chord=Math.hypot(...points[points.length-1].map((value,j)=>value-points[0][j])),total=steps.reduce((a,b)=>a+b,0);
  const lookup=new Map(path.map(item=>[item.count,axes.map(axis=>item.point[axis])])),residual2=[];
  for(const row of rows){const center=lookup.get(row[2]);if(!center)continue;residual2.push(axes.reduce((sum,axis,j)=>sum+(row[6+axis]-center[j])**2,0));}
  const within=Math.sqrt(residual2.reduce((a,b)=>a+b,0)/Math.max(residual2.length,1)),grand=points[0].map((_,j)=>points.reduce((sum,point)=>sum+point[j],0)/points.length),between=Math.sqrt(points.reduce((sum,point)=>sum+point.reduce((inner,value,j)=>inner+(value-grand[j])**2,0),0)/points.length);
  return `current 3D: step CV ${(sd/Math.max(mean,1e-9)).toFixed(2)} · path/chord ${(total/Math.max(chord,1e-9)).toFixed(2)} · seed RMS ${within.toFixed(2)} · centroid RMS ${between.toFixed(2)} · noise/signal ${(within/Math.max(between,1e-9)).toFixed(2)}`;
}
function aqDrawAxes(transform,stats,axes,width,height){
  const origin=[stats.mins[0],stats.mins[1],stats.mins[2]],ends=[[stats.maxs[0],origin[1],origin[2]],[origin[0],stats.maxs[1],origin[2]],[origin[0],origin[1],stats.maxs[2]]],start=transform(origin);aqCtx.lineWidth=1;aqCtx.font='11px system-ui';aqCtx.textAlign='left';
  ends.forEach((end,index)=>{const finish=transform(end);aqCtx.strokeStyle=['#00C2FF','#39E58C','#FF5FA2'][index];aqCtx.beginPath();aqCtx.moveTo(start.x,start.y);aqCtx.lineTo(finish.x,finish.y);aqCtx.stroke();aqCtx.fillStyle=aqCtx.strokeStyle;aqCtx.fillText(`PC${axes[index]+1}`,finish.x+4,finish.y-4);});
}
function aqDraw(){
  const rect=aqCanvas.getBoundingClientRect(),width=rect.width,height=rect.height;aqCtx.clearRect(0,0,width,height);aqCtx.fillStyle='#120D31';aqCtx.fillRect(0,0,width,height);
  const rows=aqFilteredRows(),axes=[+aqControls.x.value,+aqControls.y.value,+aqControls.z.value],data=aqActiveData(),stats=aqStatsFor(rows,axes),transform=aqMakeTransform(rows,axes,width,height);aqProjectedPoints=[];
  document.getElementById('aq-pca-stats').innerHTML=data?`<strong>${data.model} · answer-query · L${data.layer} · ${aqControls.variant.value} · ${aqControls.outcome.value}</strong><br>PCA fit: ${data.fit_cohort} V4.1 discovery (n=${data.fit_rows}; per-count ${Math.min(...Object.values(data.fit_count_support))}–${Math.max(...Object.values(data.fit_count_support))})<br>fit-cohort EVR ${data.explained_variance_ratio.slice(0,6).map((value,index)=>`PC${index+1} ${(100*value).toFixed(1)}%`).join(' · ')}<br>common V4.1 cumulative capture: PC1–3 ${(100*data.common_v41_variance_capture[2]).toFixed(1)}% · PC1–6 ${(100*data.common_v41_variance_capture[5]).toFixed(1)}%`:'';
  if(!rows.length||!stats||!transform){aqCtx.fillStyle='#F6E36A';aqCtx.font='16px system-ui';aqCtx.textAlign='center';aqCtx.fillText('No answer-query states match this filter.',width/2,height/2);document.getElementById('aq-geometry-stats').textContent='No data';return;}
  aqDrawAxes(transform,stats,axes,width,height);
  if(aqControls.points.value!=='centroids'){
    const points=rows.map(row=>({row,projected:transform(axes.map(axis=>row[6+axis]))})).sort((a,b)=>a.projected.z-b.projected.z);
    for(const item of points){const row=item.row,point=item.projected;aqCtx.globalAlpha=row[3]==='correct'?.60:.34;aqCtx.fillStyle=COLORS[row[2]-1];aqCtx.strokeStyle=row[3]==='correct'?'#F8FBFF':(row[3]==='invalid'?'#FF5FA2':'#161923');aqCtx.lineWidth=row[3]==='correct'?1.7:(row[3]==='invalid'?1.3:.7);aqCtx.beginPath();aqCtx.arc(point.x,point.y,row[3]==='correct'?3.1:2.7,0,Math.PI*2);aqCtx.fill();aqCtx.stroke();aqProjectedPoints.push({x:point.x,y:point.y,row});}aqCtx.globalAlpha=1;
  }
  const path=aqCentroids(rows),projectedPath=path.map(item=>({...item,projected:transform(axes.map(axis=>item.point[axis]))}));
  aqCtx.strokeStyle='#F8FBFF';aqCtx.lineWidth=2.5;aqCtx.beginPath();projectedPath.forEach((item,index)=>index?aqCtx.lineTo(item.projected.x,item.projected.y):aqCtx.moveTo(item.projected.x,item.projected.y));aqCtx.stroke();
  for(const item of projectedPath){aqCtx.fillStyle=COLORS[item.count-1];aqCtx.strokeStyle='#161923';aqCtx.lineWidth=1;aqCtx.beginPath();aqCtx.arc(item.projected.x,item.projected.y,5.8,0,Math.PI*2);aqCtx.fill();aqCtx.stroke();}
  drawCountLabels(aqCtx,projectedPath.map(item=>({count:item.count,x:item.projected.x,y:item.projected.y})),width,height);
  aqCtx.fillStyle='#8190A5';aqCtx.font='11px system-ui';aqCtx.textAlign='left';aqCtx.fillText(`${rows.length} prompts · ${new Set(rows.map(row=>row[1])).size} seeds · ${aqControls.outcome.value}`,12,height-12);
  document.getElementById('aq-geometry-stats').textContent=aqGeometryText(path,rows,axes);
}
function aqResizeCanvas(){const rect=aqCanvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);aqCanvas.width=Math.max(1,Math.round(rect.width*dpr));aqCanvas.height=Math.max(1,Math.round(rect.height*dpr));aqCtx.setTransform(dpr,0,0,dpr,0,0);aqDraw();}
function aqReset(){aqYaw=-.72;aqPitch=.44;aqZoom=1;aqDraw();}
aqControls.model.addEventListener('change',()=>{aqRefreshLayerOptions();aqDraw();});
[aqControls.layer,aqControls.variant,aqControls.fit,aqControls.outcome,aqControls.points,aqControls.scale,aqControls.x,aqControls.y,aqControls.z].forEach(element=>element.addEventListener('change',aqDraw));
aqControls.preset.addEventListener('change',()=>{const axes=aqControls.preset.value.split(',');aqControls.x.value=axes[0];aqControls.y.value=axes[1];aqControls.z.value=axes[2];aqDraw();});
document.getElementById('aq-reset-view').addEventListener('click',aqReset);
aqCanvas.addEventListener('pointerdown',event=>{aqDragging=true;aqLastX=event.clientX;aqLastY=event.clientY;aqCanvas.classList.add('dragging');aqCanvas.setPointerCapture(event.pointerId);});
aqCanvas.addEventListener('pointermove',event=>{if(aqDragging){aqYaw+=(event.clientX-aqLastX)*.008;aqPitch=Math.max(-1.45,Math.min(1.45,aqPitch+(event.clientY-aqLastY)*.008));aqLastX=event.clientX;aqLastY=event.clientY;aqDraw();return;}const rect=aqCanvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let best=null,distance=Infinity;for(const point of aqProjectedPoints){const current=(point.x-x)**2+(point.y-y)**2;if(current<distance){distance=current;best=point;}}if(best&&distance<80){const row=best.row;aqTooltip.style.display='block';aqTooltip.style.left=`${Math.min(rect.width-250,x+14)}px`;aqTooltip.style.top=`${Math.max(8,y-10)}px`;aqTooltip.innerHTML=`<strong>${row[0]} · seed ${row[1]} · gold ${row[2]}</strong><br>outcome ${row[3]} · predicted ${row[4]??'invalid'} · error ${row[5]??'NA'}<br>answer-query · L${aqControls.layer.value} · ${aqControls.fit.value} PCA`;}else aqTooltip.style.display='none';});
aqCanvas.addEventListener('pointerup',()=>{aqDragging=false;aqCanvas.classList.remove('dragging');});aqCanvas.addEventListener('pointercancel',()=>{aqDragging=false;aqCanvas.classList.remove('dragging');});aqCanvas.addEventListener('mouseleave',()=>{aqTooltip.style.display='none';});
aqCanvas.addEventListener('wheel',event=>{event.preventDefault();aqZoom=Math.max(.45,Math.min(2.8,aqZoom*Math.exp(-event.deltaY*.001)));aqDraw();},{passive:false});
document.getElementById('aq-count-legend').innerHTML=COLORS.map((color,index)=>`<span><i style="background:${color}"></i>${index+1}</span>`).join('');

const jointCanvas=document.getElementById('joint-counter3d');
const jointCtx=jointCanvas.getContext('2d');
const jointControls={
  model:document.getElementById('joint-model-select'),pooling:document.getElementById('joint-pooling-select'),
  layer:document.getElementById('joint-layer-select'),variant:document.getElementById('joint-variant-select'),
  mode:document.getElementById('joint-mode-select'),x:document.getElementById('joint-x-axis'),
  y:document.getElementById('joint-y-axis'),z:document.getElementById('joint-z-axis')
};
for(const select of [jointControls.x,jointControls.y,jointControls.z]){
  for(let i=0;i<6;i++){const option=document.createElement('option');option.value=String(i);option.textContent=`PC${i+1}`;select.appendChild(option);}
}
jointControls.x.value='0';jointControls.y.value='1';jointControls.z.value='2';
let jointYaw=-.72,jointPitch=.44,jointZoom=1,jointDragging=false,jointLastX=0,jointLastY=0;
function jointAvailableLayers(){
  const prefix=`${jointControls.model.value}|${jointControls.pooling.value}|`;
  return [...new Set(Object.keys(JOINT_DATA).filter(key=>key.startsWith(prefix)).map(key=>+key.split('|')[2]))].sort((a,b)=>a-b);
}
function jointRefreshLayers(){
  const layers=jointAvailableLayers();jointControls.layer.innerHTML='';
  for(const layer of layers){
    const data=JOINT_DATA[`${jointControls.model.value}|${jointControls.pooling.value}|${layer}|role_centered`];
    const option=document.createElement('option');option.value=String(layer);option.textContent=`L${layer}${data&&data.manifold_display?' · answer manifold-display':''}`;jointControls.layer.appendChild(option);
  }
  const preferred=layers.find(layer=>JOINT_DATA[`${jointControls.model.value}|${jointControls.pooling.value}|${layer}|role_centered`]?.manifold_display);
  if(layers.length)jointControls.layer.value=String(preferred??layers[layers.length-1]);
}
function jointActiveData(){return JOINT_DATA[`${jointControls.model.value}|${jointControls.pooling.value}|${jointControls.layer.value}|${jointControls.mode.value}`];}
function jointRows(){const data=jointActiveData();return data?data.rows.filter(row=>row[0]===jointControls.variant.value):[];}
function jointStats(rows,axes){
  if(!rows.length)return null;const values=axes.map(axis=>rows.map(row=>row[4+axis]));const mins=values.map(v=>Math.min(...v)),maxs=values.map(v=>Math.max(...v));return{mins,maxs,centers:mins.map((v,i)=>(v+maxs[i])/2),ranges:mins.map((v,i)=>Math.max(maxs[i]-v,1e-8))};
}
function jointTransform(rows,axes,width,height){
  const stats=jointStats(rows,axes);if(!stats)return null;const common=Math.max(...stats.ranges),radius=Math.min(width,height)*.36*jointZoom;
  return point=>{let x=(point[0]-stats.centers[0])*2/common,y=(point[1]-stats.centers[1])*2/common,z=(point[2]-stats.centers[2])*2/common;const cy=Math.cos(jointYaw),sy=Math.sin(jointYaw),cp=Math.cos(jointPitch),sp=Math.sin(jointPitch),x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;return{x:width/2+x1*radius,y:height/2-y1*radius,z:z2};};
}
function jointDraw(){
  const rect=jointCanvas.getBoundingClientRect(),width=rect.width,height=rect.height;jointCtx.clearRect(0,0,width,height);jointCtx.fillStyle='#120D31';jointCtx.fillRect(0,0,width,height);
  const data=jointActiveData(),rows=jointRows(),axes=[+jointControls.x.value,+jointControls.y.value,+jointControls.z.value],stats=jointStats(rows,axes),transform=jointTransform(rows,axes,width,height);
  if(!data||!rows.length||!stats||!transform){jointCtx.fillStyle='#F6E36A';jointCtx.font='16px system-ui';jointCtx.textAlign='center';jointCtx.fillText('Joint all-layer capture is not available.',width/2,height/2);document.getElementById('joint-pca-stats').textContent='No joint data';document.getElementById('joint-geometry-stats').textContent='No joint data';return;}
  const origin=[stats.mins[0],stats.mins[1],stats.mins[2]],ends=[[stats.maxs[0],origin[1],origin[2]],[origin[0],stats.maxs[1],origin[2]],[origin[0],origin[1],stats.maxs[2]]],start=transform(origin);jointCtx.font='11px system-ui';jointCtx.textAlign='left';ends.forEach((end,index)=>{const finish=transform(end);jointCtx.strokeStyle=['#00C2FF','#39E58C','#FF5FA2'][index];jointCtx.lineWidth=1;jointCtx.beginPath();jointCtx.moveTo(start.x,start.y);jointCtx.lineTo(finish.x,finish.y);jointCtx.stroke();jointCtx.fillStyle=jointCtx.strokeStyle;jointCtx.fillText(`PC${axes[index]+1}`,finish.x+4,finish.y-4);});
  const roles={prompt_occurrence:[],answer_query:[]};for(const row of rows){roles[row[1]].push({row,count:row[2],point:transform(axes.map(axis=>row[4+axis]))});}for(const role of Object.keys(roles))roles[role].sort((a,b)=>a.count-b.count);
  const answerByCount=new Map(roles.answer_query.map(item=>[item.count,item]));jointCtx.strokeStyle='rgba(246,227,106,.42)';jointCtx.lineWidth=1;for(const prompt of roles.prompt_occurrence){const answer=answerByCount.get(prompt.count);if(!answer)continue;jointCtx.beginPath();jointCtx.moveTo(prompt.point.x,prompt.point.y);jointCtx.lineTo(answer.point.x,answer.point.y);jointCtx.stroke();}
  for(const [role,items] of Object.entries(roles)){jointCtx.strokeStyle=role==='prompt_occurrence'?'#F8FBFF':'#F6E36A';jointCtx.lineWidth=role==='prompt_occurrence'?2.7:2.2;jointCtx.setLineDash(role==='prompt_occurrence'?[]:[7,5]);jointCtx.beginPath();items.forEach((item,index)=>index?jointCtx.lineTo(item.point.x,item.point.y):jointCtx.moveTo(item.point.x,item.point.y));jointCtx.stroke();jointCtx.setLineDash([]);for(const item of items){jointCtx.fillStyle=COLORS[item.count-1];jointCtx.strokeStyle=role==='prompt_occurrence'?'#F8FBFF':'#F6E36A';jointCtx.lineWidth=1.4;if(role==='prompt_occurrence'){jointCtx.beginPath();jointCtx.arc(item.point.x,item.point.y,5.5,0,Math.PI*2);jointCtx.fill();jointCtx.stroke();}else{jointCtx.fillRect(item.point.x-4.8,item.point.y-4.8,9.6,9.6);jointCtx.strokeRect(item.point.x-4.8,item.point.y-4.8,9.6,9.6);}}}
  drawCountLabels(jointCtx,roles.prompt_occurrence.map(item=>({count:item.count,x:item.point.x,y:item.point.y})),width,height);
  jointCtx.fillStyle='#8190A5';jointCtx.font='11px system-ui';jointCtx.textAlign='left';jointCtx.fillText('circles/solid = prompt · squares/dashed = answer · connectors = same count',12,height-12);
  document.getElementById('joint-pca-stats').innerHTML=`<strong>${data.model} · ${data.pooling} · L${data.layer} · ${data.mode}</strong><br>shared PCA fit: V4.1 discovery, ${data.fit_rows} role-states · EVR ${data.explained_variance_ratio.slice(0,6).map((value,index)=>`PC${index+1} ${(100*value).toFixed(1)}%`).join(' · ')}`;
  document.getElementById('joint-geometry-stats').innerHTML=`full-space trajectory agreement: linear CKA ${data.trajectory_linear_cka.toFixed(3)} · distance corr ${data.trajectory_distance_correlation.toFixed(3)} · successive-step cosine ${data.successive_step_cosine.toFixed(3)}<br>answer/prompt trajectory scale ${data.answer_to_prompt_scale_ratio.toFixed(3)} · raw role-offset / count-signal ${data.role_offset_to_count_signal.toFixed(2)}`;
}
function jointResize(){const rect=jointCanvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);jointCanvas.width=Math.max(1,Math.round(rect.width*dpr));jointCanvas.height=Math.max(1,Math.round(rect.height*dpr));jointCtx.setTransform(dpr,0,0,dpr,0,0);jointDraw();}
function jointReset(){jointYaw=-.72;jointPitch=.44;jointZoom=1;jointDraw();}
[jointControls.model,jointControls.pooling].forEach(control=>control.addEventListener('change',()=>{jointRefreshLayers();jointDraw();}));
[jointControls.layer,jointControls.variant,jointControls.mode,jointControls.x,jointControls.y,jointControls.z].forEach(control=>control.addEventListener('change',jointDraw));
document.getElementById('joint-reset-view').addEventListener('click',jointReset);
jointCanvas.addEventListener('pointerdown',event=>{jointDragging=true;jointLastX=event.clientX;jointLastY=event.clientY;jointCanvas.classList.add('dragging');jointCanvas.setPointerCapture(event.pointerId);});
jointCanvas.addEventListener('pointermove',event=>{if(!jointDragging)return;jointYaw+=(event.clientX-jointLastX)*.008;jointPitch=Math.max(-1.45,Math.min(1.45,jointPitch+(event.clientY-jointLastY)*.008));jointLastX=event.clientX;jointLastY=event.clientY;jointDraw();});
jointCanvas.addEventListener('pointerup',()=>{jointDragging=false;jointCanvas.classList.remove('dragging');});jointCanvas.addEventListener('pointercancel',()=>{jointDragging=false;jointCanvas.classList.remove('dragging');});
jointCanvas.addEventListener('wheel',event=>{event.preventDefault();jointZoom=Math.max(.45,Math.min(2.8,jointZoom*Math.exp(-event.deltaY*.001)));jointDraw();},{passive:false});
document.getElementById('joint-count-legend').innerHTML=COLORS.map((color,index)=>`<span><i style="background:${color}"></i>${index+1}</span>`).join('');

function refreshAttentionAtlas(){
  const variant=document.querySelector('.atlas-button[aria-pressed="true"]').dataset.atlasVariant;
  const pooling=document.querySelector('.atlas-pooling-button[aria-pressed="true"]').dataset.atlasPooling;
  document.querySelectorAll('.atlas-panel').forEach(panel=>{panel.hidden=panel.dataset.atlasVariant!==variant||panel.dataset.atlasPooling!==pooling;});
}
document.querySelectorAll('.atlas-button').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('.atlas-button').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));refreshAttentionAtlas();}));
document.querySelectorAll('.atlas-pooling-button').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('.atlas-pooling-button').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));refreshAttentionAtlas();}));
document.querySelectorAll('.outcome-pooling-button').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('.outcome-pooling-button').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));document.querySelectorAll('.outcome-pooling-panel').forEach(panel=>{panel.hidden=panel.dataset.outcomePooling!==button.dataset.outcomePooling;});}));
refreshLayerOptions();new ResizeObserver(resizeCanvas).observe(canvas);resizeCanvas();
aqRefreshLayerOptions();new ResizeObserver(aqResizeCanvas).observe(aqCanvas);aqResizeCanvas();
jointRefreshLayers();new ResizeObserver(jointResize).observe(jointCanvas);jointResize();
pcdRefreshLayers();new ResizeObserver(pcdResize).observe(pcdCanvas);pcdResize();
</script>
</body>
</html>"""


def build_report(run_root: Path, output: Path, repo_root: Path) -> None:
    run_root = run_root.resolve()
    probe_layers: dict[str, dict[str, int]] = {}
    display_layers: dict[str, dict[str, int]] = {}
    layer_sweep_rows: list[dict[str, Any]] = []
    labels_lookup: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    labels_frames: dict[str, pd.DataFrame] = {}
    all_labels_frames: dict[str, pd.DataFrame] = {}
    projections: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        model_root = run_root / model / "numeric"
        probe_layers[model] = _primary_layers(model_root)
        labels_lookup[model], labels_frames[model] = _n10_labels(model_root)
        all_labels_frames[model] = _all_generation_labels(model_root)
        model_sweep, model_display, pca_models = _layer_sweep(
            model_root,
            model=model,
            probe_layers=probe_layers[model],
        )
        layer_sweep_rows.extend(model_sweep)
        display_layers[model] = model_display
        projections.update(
            _load_prompt_projection_layers(
                model_root,
                model=model,
                labels=labels_lookup[model],
                pca_models=pca_models,
                layer_rows=model_sweep,
            )
        )

    display_projections = {
        f"{model}|{pooling}": projections[
            f"{model}|{pooling}|{display_layers[model][pooling]}"
        ]
        for model in MODELS
        for pooling in POOLINGS
    }
    answer_query_projections = _answer_query_projection_data(run_root)
    answer_query_pca_sensitivity_rows = _answer_query_pca_sensitivity_rows(
        answer_query_projections
    )
    joint_counter_projections = _joint_counter_projection_data(
        run_root, answer_query_projections
    )
    prompt_counter_dynamics = _prompt_counter_dynamics_frames(run_root)
    prompt_counter_profile_data = _prompt_counter_profile_data(
        prompt_counter_dynamics
    )
    prompt_counter_selected_rows = _prompt_counter_selected_rows(
        prompt_counter_dynamics, layer_sweep_rows
    )
    metric_rows = _metric_rows(run_root, probe_layers)
    behavior_rows = _behavior_rows(labels_frames)
    behavior_panel_rows = _behavior_panel_rows(all_labels_frames)
    behavior_count_rows = _behavior_count_rows(all_labels_frames)
    behavior_count_pooled_rows = _behavior_count_pooled_rows(all_labels_frames)
    sensitivity_rows = _sensitivity_rows(run_root)
    attention_top_rows = _attention_top_rows(run_root)
    attention_atlas_rows = _attention_head_atlas_rows(run_root)
    attention_pooling_alignment_rows = _attention_pooling_alignment_rows(run_root)
    attention_phenotypes = _attention_head_phenotypes(run_root)
    attention_outcome_effects = _attention_outcome_effect_rows(run_root)
    partition_summary = _qwen_partition_summary(run_root)
    span_end_alignment_rows = _span_end_alignment_rows(run_root)
    span_end_pooled_rows = _span_end_pooled_rows(run_root)
    span_end_nested_rows = _span_end_nested_rows(run_root)
    attention_omission_pooling_rows = (
        _attention_omission_pooling_sensitivity_rows(run_root)
    )
    causal_audit = audit_screen_8h(run_root)
    causal_frames, _causal_paths = _causal_frames(run_root)
    causal_ablation_rows = _causal_ablation_rows(causal_frames)
    causal_patching_rows = _causal_patching_rows(causal_frames)
    causal_steering_rows = _causal_steering_rows(causal_frames)
    causal_geometry_rows = _causal_geometry_rows(causal_frames)
    (
        steering_v2_selection_rows,
        steering_v2_summary_rows,
        steering_v2_panel_rows,
        steering_v2_audit_rows,
    ) = _steering_v2_rows(run_root)
    answer_query_frames, answer_query_audit = _answer_query_frames(run_root)
    answer_query_layer_rows = answer_query_frames["layer_summary"].to_dict("records")
    answer_query_variant_rows = _answer_query_final_rows(
        answer_query_frames["variant_summary"]
    )
    answer_query_pair_rows = _answer_query_final_rows(
        answer_query_frames["pair_summary"]
    )
    commit = _git_commit(repo_root)
    replacements = {
        "@@COMMIT@@": html.escape(commit[:12]),
        "@@GENERATED@@": html.escape(
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ),
        "@@RUN_NAME@@": html.escape(run_root.name),
        "@@METRIC_ROWS@@": _table_metric_html(metric_rows),
        "@@BEHAVIOR_ROWS@@": _table_behavior_html(behavior_rows),
        "@@BEHAVIOR_ACCURACY_SVG@@": _behavior_accuracy_svg(behavior_count_rows),
        "@@BEHAVIOR_PANEL_ROWS@@": _table_behavior_panel_html(
            behavior_panel_rows
        ),
        "@@BEHAVIOR_COUNT_ROWS@@": _table_behavior_count_html(
            behavior_count_pooled_rows
        ),
        "@@BEHAVIOR_CONCLUSION@@": _behavior_conclusion_html(
            behavior_panel_rows, behavior_count_pooled_rows
        ),
        "@@REPRESENTATION_R2_SVG@@": _representation_r2_svg(metric_rows),
        "@@LAYER_SWEEP_SVG@@": _layer_sweep_svg(layer_sweep_rows),
        "@@LAYER_SELECTION_ROWS@@": _table_layer_selection_html(
            layer_sweep_rows
        ),
        "@@LAYER_SELECTION_CONCLUSION@@": _layer_selection_conclusion_html(
            layer_sweep_rows
        ),
        "@@REPRESENTATION_CONCLUSION@@": _representation_conclusion_html(
            metric_rows, sensitivity_rows
        ),
        "@@ANSWER_QUERY_COUNTER_SVG@@": _answer_query_counter_svg(
            answer_query_projections
        ),
        "@@ANSWER_QUERY_LAYER_SWEEP_SVG@@": _answer_query_layer_sweep_svg(
            answer_query_projections
        ),
        "@@ANSWER_QUERY_DATA@@": json.dumps(
            answer_query_projections, ensure_ascii=False, separators=(",", ":")
        ),
        "@@JOINT_COUNTER_DATA@@": json.dumps(
            joint_counter_projections, ensure_ascii=False, separators=(",", ":")
        ),
        "@@PROMPT_COUNTER_PROFILE_DATA@@": json.dumps(
            prompt_counter_profile_data,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "@@PROMPT_COUNTER_ASSOCIATION_SVG@@": _prompt_counter_association_svg(
            prompt_counter_dynamics["associations"], layer_sweep_rows
        ),
        "@@PROMPT_COUNTER_DYNAMICS_ROWS@@": _table_prompt_counter_dynamics_html(
            prompt_counter_selected_rows
        ),
        "@@PROMPT_COUNTER_DYNAMICS_CONCLUSION@@": (
            _prompt_counter_dynamics_conclusion_html(
                prompt_counter_selected_rows
            )
        ),
        "@@ANSWER_QUERY_SENSITIVITY_ROW_COUNT@@": f"{len(answer_query_pca_sensitivity_rows):,}",
        "@@ANSWER_QUERY_PCA_SENSITIVITY_ROWS@@": (
            _table_answer_query_pca_sensitivity_html(
                answer_query_pca_sensitivity_rows
            )
        ),
        "@@ANSWER_QUERY_PCA_CONCLUSION@@": _answer_query_pca_conclusion_html(
            answer_query_pca_sensitivity_rows
        ),
        "@@SENSITIVITY_ROWS@@": _table_sensitivity_html(sensitivity_rows),
        "@@ATTENTION_BREADTH_SVG@@": _attention_breadth_svg(attention_top_rows),
        "@@ATTENTION_HEAD_ATLAS_HTML@@": _attention_head_atlas_interactive_html(
            attention_atlas_rows, attention_phenotypes
        ),
        "@@ATTENTION_ATLAS_ROW_COUNT@@": f"{len(attention_atlas_rows):,}",
        "@@ATTENTION_PHENOTYPE_ROW_COUNT@@": f"{len(attention_phenotypes):,}",
        "@@ATTENTION_POOLING_ALIGNMENT_ROWS@@": (
            _table_attention_pooling_alignment_html(
                attention_pooling_alignment_rows
            )
        ),
        "@@ATTENTION_POOLING_ALIGNMENT_CONCLUSION@@": (
            _attention_pooling_alignment_conclusion_html(
                attention_pooling_alignment_rows
            )
        ),
        "@@ATTENTION_HEAD_PROFILE_SVG@@": _representative_head_profiles_svg(
            attention_phenotypes
        ),
        "@@ATTENTION_PHENOTYPE_COUNT_ROWS@@": _table_head_phenotype_counts_html(
            attention_phenotypes
        ),
        "@@ATTENTION_REPRESENTATIVE_ROWS@@": _table_head_representatives_html(
            attention_phenotypes
        ),
        "@@ATTENTION_OUTCOME_EFFECT_SVG@@": _attention_outcome_effect_interactive_html(
            attention_outcome_effects
        ),
        "@@ATTENTION_OUTCOME_SUMMARY_ROWS@@": _table_attention_outcome_summary_html(
            attention_outcome_effects
        ),
        "@@ATTENTION_OUTCOME_CONCLUSION@@": _attention_outcome_conclusion_html(
            attention_outcome_effects
        ),
        "@@ATTENTION_TOP_ROWS@@": _table_attention_top_html(attention_top_rows),
        "@@PARTITION_PHENOTYPE_SVG@@": _partition_phenotype_svg(
            partition_summary
        ),
        "@@PARTITION_BANK_ROWS@@": _table_partition_bank_html(
            partition_summary["rows"]
        ),
        "@@STABLE_GLOBAL_HEADS@@": html.escape(
            ", ".join(partition_summary["stable_global_heads"])
        ),
        "@@ATTENTION_CONCLUSION@@": _attention_conclusion_html(
            attention_top_rows, partition_summary
        ),
        "@@SPAN_END_ALIGNMENT_ROWS@@": _table_span_end_alignment_html(
            span_end_alignment_rows
        ),
        "@@SPAN_END_POOLED_ROWS@@": _table_span_end_pooled_html(span_end_pooled_rows),
        "@@ATTENTION_OMISSION_POOLING_ROWS@@": (
            _table_attention_omission_pooling_sensitivity_html(
                attention_omission_pooling_rows
            )
        ),
        "@@ATTENTION_OMISSION_POOLING_CONCLUSION@@": (
            _attention_omission_pooling_conclusion_html(
                attention_omission_pooling_rows
            )
        ),
        "@@SPAN_END_ALIGNMENT_SVG@@": _span_end_alignment_svg(span_end_alignment_rows),
        "@@SPAN_END_NESTED_ROWS@@": _table_span_end_nested_html(span_end_nested_rows),
        "@@SPAN_END_NESTED_SVG@@": _span_end_nested_svg(span_end_nested_rows),
        "@@SPAN_END_CONCLUSION@@": _span_end_conclusion_html(
            span_end_pooled_rows, span_end_alignment_rows, span_end_nested_rows
        ),
        "@@CAUSAL_ABLATION_ROWS@@": _table_causal_ablation_html(causal_ablation_rows),
        "@@CAUSAL_PATCHING_ROWS@@": _table_causal_patching_html(causal_patching_rows),
        "@@CAUSAL_STEERING_ROWS@@": _table_causal_steering_html(causal_steering_rows),
        "@@STEERING_V2_SELECTION_ROWS@@": _table_steering_v2_selection_html(
            steering_v2_selection_rows
        ),
        "@@STEERING_V2_SUMMARY_ROWS@@": _table_steering_v2_summary_html(
            steering_v2_summary_rows
        ),
        "@@STEERING_V2_PANEL_ROWS@@": _table_steering_v2_panel_html(
            steering_v2_panel_rows
        ),
        "@@STEERING_V2_AUDIT_ROWS@@": _table_steering_v2_audit_html(
            steering_v2_audit_rows
        ),
        "@@STEERING_V2_CONCLUSION@@": _steering_v2_conclusion_html(
            steering_v2_summary_rows, steering_v2_panel_rows
        ),
        "@@CAUSAL_GEOMETRY_ROWS@@": _table_causal_geometry_html(causal_geometry_rows),
        "@@CAUSAL_AUDIT_ROWS@@": _table_causal_audit_html(causal_audit),
        "@@ANSWER_QUERY_LAYER_ROWS@@": _table_answer_query_layer_html(
            answer_query_layer_rows
        ),
        "@@ANSWER_QUERY_VARIANT_ROWS@@": _table_answer_query_variant_html(
            answer_query_variant_rows
        ),
        "@@ANSWER_QUERY_PAIR_ROWS@@": _table_answer_query_pair_html(
            answer_query_pair_rows
        ),
        "@@ANSWER_QUERY_INVALID@@": _answer_query_invalid_html(
            answer_query_frames["invalid_rows"]
        ),
        "@@ANSWER_QUERY_AUDIT_ROWS@@": _table_answer_query_audit_html(
            answer_query_audit
        ),
        "@@CAUSAL_ABLATION_SVG@@": _forest_svg(
            causal_ablation_rows,
            estimate_key="count_shift_difference",
            low_key="count_shift_difference_low",
            high_key="count_shift_difference_high",
            title="Discovery-ranked mixed head-bank ablation versus layer-matched random",
            axis_label="paired mean count shift: ranked minus random (negative = stronger undercount)",
            label=lambda row: f"{row['model']} top-{row['top_n']}",
        ),
        "@@CAUSAL_PATCHING_SVG@@": _forest_svg(
            causal_patching_rows,
            estimate_key="aligned_shift",
            low_key="aligned_shift_low",
            high_key="aligned_shift_high",
            title="Exact needle-end residual transport",
            axis_label="mean direction-aligned generated-count shift",
            label=lambda row: f"{row['model']} L{row['layer']}",
        ),
        "@@CAUSAL_STEERING_SVG@@": _forest_svg(
            causal_steering_rows,
            estimate_key="aligned_difference",
            low_key="aligned_difference_low",
            high_key="aligned_difference_high",
            title="Centroid-delta steering versus norm-matched orthogonal random",
            axis_label="paired direction-aligned count shift: geometric minus random",
            label=lambda row: f"{row['model']} L{row['layer']}",
        ),
        "@@STEERING_V2_SVG@@": _forest_svg(
            steering_v2_summary_rows,
            estimate_key="aligned_effect",
            low_key="aligned_effect_low",
            high_key="aligned_effect_high",
            title="Discovery-locked single-layer and multi-layer steering",
            axis_label=(
                "held-out paired strict aligned-count shift: geometric minus random"
            ),
            label=lambda row: (
                f"{row['model']} {row['protocol'].replace('_', '-')} "
                f"{row['layer_set']} α={float(row['alpha']):g}"
            ),
        ),
        "@@ANSWER_QUERY_ADOPTION_SVG@@": _forest_svg(
            answer_query_layer_rows,
            estimate_key="eligible_donor_adoption_rate",
            low_key="eligible_donor_adoption_rate_ci95_low",
            high_key="eligible_donor_adoption_rate_ci95_high",
            title="Exact answer-query donor-prediction transport",
            axis_label="eligible rows adopting donor baseline prediction",
            label=lambda row: f"{row['model']} L{int(row['layer'])}",
        ),
        "@@CAUSAL_CONCLUSION@@": _causal_conclusion_html(
            causal_ablation_rows,
            causal_patching_rows,
            causal_steering_rows,
            answer_query_layer_rows,
        ),
        "@@STATIC_FIGURES@@": _static_figure_html(display_projections),
        "@@REP_DATA@@": json.dumps(
            projections, ensure_ascii=False, separators=(",", ":")
        ),
    }
    rendered = REPORT_TEMPLATE
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    layer_sweep_path = output.with_name("realistic_niah_v4_layer_sweep.csv")
    pd.DataFrame(layer_sweep_rows).to_csv(layer_sweep_path, index=False)
    atlas_path = output.with_name("realistic_niah_v4_head_atlas.csv")
    pd.DataFrame(attention_atlas_rows).to_csv(atlas_path, index=False)
    phenotype_path = output.with_name("realistic_niah_v4_head_phenotypes.csv")
    phenotype_frame = pd.DataFrame(attention_phenotypes).copy()
    for column in ("endpoint_profile", "span_mean_profile", "span_sum_profile"):
        phenotype_frame[column] = phenotype_frame[column].map(
            lambda values: json.dumps(values, separators=(",", ":"))
        )
    phenotype_frame.to_csv(phenotype_path, index=False)
    outcome_effect_path = output.with_name(
        "realistic_niah_v4_attention_outcome_effects.csv"
    )
    pd.DataFrame(attention_outcome_effects).to_csv(
        outcome_effect_path, index=False
    )
    answer_query_pca_sensitivity_path = output.with_name(
        "realistic_niah_v4_answer_query_pca_sensitivity.csv"
    )
    pd.DataFrame(answer_query_pca_sensitivity_rows).to_csv(
        answer_query_pca_sensitivity_path, index=False
    )
    prompt_counter_selected_path = output.with_name(
        "realistic_niah_v4_prompt_counter_dynamics_selected_layers.csv"
    )
    pd.DataFrame(prompt_counter_selected_rows).to_csv(
        prompt_counter_selected_path, index=False
    )
    joint_counter_metrics_path = output.with_name(
        "realistic_niah_v4_joint_counter_layer_metrics.csv"
    )
    pd.DataFrame(
        [
            {key: value for key, value in item.items() if key != "rows"}
            for item in joint_counter_projections.values()
        ]
    ).to_csv(joint_counter_metrics_path, index=False)
    steering_v2_selection_path = output.with_name(
        "realistic_niah_v4_steering_v2_selection.csv"
    )
    pd.DataFrame(steering_v2_selection_rows).to_csv(
        steering_v2_selection_path, index=False
    )
    steering_v2_confirmation_path = output.with_name(
        "realistic_niah_v4_steering_v2_confirmation.csv"
    )
    pd.DataFrame(steering_v2_summary_rows).to_csv(
        steering_v2_confirmation_path, index=False
    )
    steering_v2_panel_path = output.with_name(
        "realistic_niah_v4_steering_v2_panels.csv"
    )
    pd.DataFrame(steering_v2_panel_rows).to_csv(
        steering_v2_panel_path, index=False
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "projection_panels": len(projections),
                "projection_rows": sum(
                    len(item["rows"]) for item in projections.values()
                ),
                "probe_optimal_layers": probe_layers,
                "manifold_display_layers": display_layers,
                "layer_sweep_csv": str(layer_sweep_path.resolve()),
                "head_atlas_csv": str(atlas_path.resolve()),
                "head_phenotypes_csv": str(phenotype_path.resolve()),
                "attention_outcome_effects_csv": str(
                    outcome_effect_path.resolve()
                ),
                "answer_query_pca_sensitivity_csv": str(
                    answer_query_pca_sensitivity_path.resolve()
                ),
                "joint_counter_layer_metrics_csv": str(
                    joint_counter_metrics_path.resolve()
                ),
                "prompt_counter_dynamics_selected_layers_csv": str(
                    prompt_counter_selected_path.resolve()
                ),
                "joint_counter_projection_panels": len(
                    joint_counter_projections
                ),
                "steering_v2_selection_csv": str(
                    steering_v2_selection_path.resolve()
                ),
                "steering_v2_confirmation_csv": str(
                    steering_v2_confirmation_path.resolve()
                ),
                "steering_v2_panels_csv": str(
                    steering_v2_panel_path.resolve()
                ),
                "attention_phenotype_rows": len(attention_phenotypes),
                "behavior_confirmation_rows": sum(
                    int(row["n"]) for row in behavior_panel_rows
                ),
                "attention_top_rows": len(attention_top_rows),
                "qwen_stable_global_heads": len(
                    partition_summary["stable_global_heads"]
                ),
                "causal_audit_validated": bool(causal_audit["validated"]),
                "causal_summary_rows": {
                    "ablation": len(causal_ablation_rows),
                    "endpoint_patching": len(causal_patching_rows),
                    "steering": len(causal_steering_rows),
                    "centroid_geometry": len(causal_geometry_rows),
                    "steering_v2_selection": len(steering_v2_selection_rows),
                    "steering_v2_confirmation": len(steering_v2_summary_rows),
                    "steering_v2_panels": len(steering_v2_panel_rows),
                },
                "answer_query_audit_validated": bool(answer_query_audit["validated"]),
                "answer_query_summary_rows": {
                    "layers": len(answer_query_layer_rows),
                    "final_variants": len(answer_query_variant_rows),
                    "final_pairs": len(answer_query_pair_rows),
                },
                "answer_query_invalid_rows": int(
                    len(answer_query_frames["invalid_rows"])
                ),
                "commit": commit,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a self-contained V4 representation HTML report."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    build_report(args.run_root, args.output, args.repo_root)


if __name__ == "__main__":
    main()
