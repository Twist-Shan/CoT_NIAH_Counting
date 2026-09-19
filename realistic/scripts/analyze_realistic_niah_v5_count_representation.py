#!/usr/bin/env python3
"""Validate a native-thinking count state without using confirmation labels.

The legacy counting report showed that count could be linearly decoded from
hidden states.  This script makes that claim harder to satisfy:

* the decoder and the reported count bases are fitted on discovery seeds only;
* the layer is selected by grouped (seed-held-out) discovery cross-validation;
* confirmation seeds are evaluated exactly once at the frozen layer;
* a token-position-only baseline and a position-residualized hidden-state probe
  expose the strongest ordinal/absolute-position confound; and
* ``city_end`` can be restricted to ``rank_after_city`` events, so the current
  explicit ordinal/count marker has not appeared yet.

The output is descriptive/representational evidence.  It is deliberately not
called a mechanism proof; the saved discovery-only bases are intended for a
separate causal removal/patching experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from realistic_niah_v5.causal import fit_centroid_subspace  # noqa: E402
from realistic_niah_v5.count_stream import (  # noqa: E402
    deterministic_control_basis,
)


SCHEMA_VERSION = "realistic_niah_v5_count_representation_v1"
DEFAULT_SITE_KINDS = ("city_end", "item_end", "post_boundary")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


@dataclass(frozen=True)
class SiteCapture:
    metadata: pd.DataFrame
    states: np.ndarray
    layers: np.ndarray


def _event_by_occurrence(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episode = manifest.get("episode_parse") or {}
    selected = episode.get("events") or []
    output: dict[int, dict[str, Any]] = {}
    for event in selected:
        rank = event.get("rank")
        if rank is None:
            continue
        output[int(rank)] = dict(event)
    return output


def load_site_capture(
    capture_index: Path,
    *,
    site_kind: str,
    cohort: str,
) -> SiteCapture:
    """Load one site kind while retaining states as float16 to cap RAM."""

    if cohort not in {"parser_hit", "one_to_one", "one_to_one_correct"}:
        raise ValueError(f"Unknown cohort: {cohort}")
    root = capture_index.parent
    index_rows = [
        json.loads(line)
        for line in capture_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    state_rows: list[np.ndarray] = []
    expected_layers: np.ndarray | None = None
    expected_width: int | None = None
    for index_row in index_rows:
        if cohort in {"one_to_one", "one_to_one_correct"} and not bool(
            index_row.get("trace_one_to_one")
        ):
            continue
        if cohort == "one_to_one_correct" and not bool(index_row.get("exact_count")):
            continue
        manifest_path = root / str(index_row["manifest_path"])
        states_path = root / str(index_row["states_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parser = manifest.get("parser") or {}
        if cohort == "parser_hit" and not bool(parser.get("detected")):
            continue
        site_rows = list(manifest["site_rows"])
        selected_axes = [
            axis
            for axis, site in enumerate(site_rows)
            if str(site.get("site_kind")) == str(site_kind)
            and site.get("occurrence") is not None
            and bool(site.get("alignment_eligible", True))
        ]
        if not selected_axes:
            continue
        events = _event_by_occurrence(manifest)
        with np.load(states_path, allow_pickle=False) as archive:
            layers = np.asarray(archive["layer_indices"], dtype=np.int64)
            selected_states = np.asarray(
                archive["site_states"][selected_axes], dtype=np.float16
            )
        if selected_states.ndim != 3:
            raise ValueError(f"Unexpected state shape in {states_path}")
        if expected_layers is None:
            expected_layers = layers
            expected_width = int(selected_states.shape[-1])
        elif not np.array_equal(expected_layers, layers):
            raise ValueError("Capture shards disagree on layer indices")
        elif expected_width != int(selected_states.shape[-1]):
            raise ValueError("Capture shards disagree on hidden width")
        for local_axis, site_axis in enumerate(selected_axes):
            site = site_rows[site_axis]
            occurrence = int(site["occurrence"])
            event = events.get(occurrence, {})
            endpoint = int(site.get("endpoint_token", site.get("prefix_token_count", 1) - 1))
            literal_start = site.get("literal_token_start")
            literal_end = site.get("literal_token_end")
            literal_span = (
                int(literal_end) - int(literal_start)
                if literal_start is not None and literal_end is not None
                else 0
            )
            total_tokens = int(manifest.get("prompt_token_count", 0)) + int(
                manifest.get("output_token_count", 0)
            )
            rows.append(
                {
                    "request_id": str(manifest["request_id"]),
                    "model_label": str(manifest["model_label"]),
                    "seed": int(manifest["seed"]),
                    "split": str(manifest.get("split", index_row.get("split", ""))),
                    "gold_count": int(manifest["gold_count"]),
                    "occurrence": occurrence,
                    "site_id": str(site["site_id"]),
                    "site_kind": str(site["site_kind"]),
                    "endpoint_token": endpoint,
                    "relative_endpoint": endpoint / max(total_tokens - 1, 1),
                    "literal_span_tokens": literal_span,
                    "output_token_count": int(manifest.get("output_token_count", 0)),
                    "prompt_token_count": int(manifest.get("prompt_token_count", 0)),
                    "association": str(event.get("association", "unknown")),
                    "evidence_family": str(event.get("evidence_family", "unknown")),
                    "evidence_kind": str(event.get("evidence_kind", "unknown")),
                    "exact_count": bool(manifest.get("exact_count")),
                    "trace_one_to_one": bool(parser.get("trace_one_to_one")),
                }
            )
            state_rows.append(selected_states[local_axis])
    if not state_rows or expected_layers is None:
        raise ValueError(f"No {site_kind!r} states matched {capture_index}")
    states = np.stack(state_rows, axis=0)
    metadata = pd.DataFrame(rows)
    if states.shape[:2] != (len(metadata), len(expected_layers)):
        raise RuntimeError("Metadata/state shape mismatch")
    return SiteCapture(metadata=metadata, states=states, layers=expected_layers)


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("Metric vectors must be aligned and one-dimensional")
    residual = float(np.sum((truth - predicted) ** 2))
    centered = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "r2": 1.0 - residual / centered if centered > 0 else math.nan,
        "mae": float(np.mean(np.abs(truth - predicted))),
        "rmse": float(np.sqrt(np.mean((truth - predicted) ** 2))),
        "rounded_exact": float(np.mean(np.rint(predicted) == truth)),
        "signed_bias": float(np.mean(predicted - truth)),
    }


def _scaled_design(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train_array = np.asarray(train, dtype=np.float64)
    test_array = np.asarray(test, dtype=np.float64)
    mean = train_array.mean(axis=0, keepdims=True)
    scale = train_array.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    train_scaled = (train_array - mean) / scale
    test_scaled = (test_array - mean) / scale
    return (
        np.column_stack([np.ones(len(train_scaled)), train_scaled]),
        np.column_stack([np.ones(len(test_scaled)), test_scaled]),
    )


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    x_train, x_test = _scaled_design(train_x, test_x)
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        x_train.T @ x_train + penalty,
        x_train.T @ np.asarray(train_y, dtype=np.float64),
    )
    return x_test @ coefficients


def _fit_hidden_probe(
    train_states: np.ndarray,
    train_y: np.ndarray,
    test_states: np.ndarray,
    *,
    rank: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, basis = fit_centroid_subspace(
        np.asarray(train_states, dtype=np.float32),
        np.asarray(train_y, dtype=np.int64),
        rank=int(rank),
    )
    train_projection = (np.asarray(train_states, dtype=np.float32) - center) @ basis
    test_projection = (np.asarray(test_states, dtype=np.float32) - center) @ basis
    prediction = _ridge_predict(
        train_projection,
        train_y,
        test_projection,
        alpha=float(alpha),
    )
    return prediction, center, basis


def _position_features(metadata: pd.DataFrame, *, include_gold: bool) -> np.ndarray:
    endpoint = metadata["endpoint_token"].to_numpy(dtype=np.float64)
    relative = metadata["relative_endpoint"].to_numpy(dtype=np.float64)
    span = metadata["literal_span_tokens"].to_numpy(dtype=np.float64)
    output_length = metadata["output_token_count"].to_numpy(dtype=np.float64)
    columns = [endpoint, endpoint**2, relative, relative**2, span, output_length]
    if include_gold:
        gold = metadata["gold_count"].to_numpy(dtype=np.float64)
        columns.extend([gold, gold**2])
    return np.column_stack(columns)


def _position_residualize(
    train_states: np.ndarray,
    train_metadata: pd.DataFrame,
    test_states: np.ndarray,
    test_metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_design, test_design = _scaled_design(
        _position_features(train_metadata, include_gold=False),
        _position_features(test_metadata, include_gold=False),
    )
    train_float = np.asarray(train_states, dtype=np.float32)
    test_float = np.asarray(test_states, dtype=np.float32)
    coefficients, *_ = np.linalg.lstsq(train_design, train_float, rcond=None)
    return (
        train_float - train_design @ coefficients,
        test_float - test_design @ coefficients,
    )


def _seed_folds(metadata: pd.DataFrame, folds: int) -> list[np.ndarray]:
    seeds = sorted(int(value) for value in metadata["seed"].unique())
    if len(seeds) < int(folds):
        raise ValueError("There are fewer discovery seeds than requested folds")
    output = []
    seed_values = metadata["seed"].to_numpy(dtype=int)
    for fold in range(int(folds)):
        selected = {seed for index, seed in enumerate(seeds) if index % folds == fold}
        output.append(np.asarray([value in selected for value in seed_values], dtype=bool))
    if not all(mask.any() and (~mask).any() for mask in output):
        raise RuntimeError("A grouped discovery fold is empty")
    return output


def _cross_validated_hidden_predictions(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    rank: int,
    alpha: float,
    folds: int,
) -> np.ndarray:
    labels = metadata["occurrence"].to_numpy(dtype=int)
    predictions = np.full(len(labels), np.nan, dtype=np.float64)
    for test_mask in _seed_folds(metadata, folds):
        train_mask = ~test_mask
        prediction, _center, _basis = _fit_hidden_probe(
            states[train_mask],
            labels[train_mask],
            states[test_mask],
            rank=rank,
            alpha=alpha,
        )
        predictions[test_mask] = prediction
    if not np.isfinite(predictions).all():
        raise RuntimeError("Discovery cross-validation left missing predictions")
    return predictions


def _cohort_masks(site_kind: str, metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = {"all_one_to_one": np.ones(len(metadata), dtype=bool)}
    if site_kind == "city_end":
        masks["rank_after_city_pre_marker"] = metadata["association"].eq(
            "rank_after_city"
        ).to_numpy()
    return masks


def analyze_site(
    capture: SiteCapture,
    *,
    rank: int,
    alpha: float,
    folds: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    metadata = capture.metadata.reset_index(drop=True)
    site_kind = str(metadata["site_kind"].iloc[0])
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    basis_arrays: dict[str, np.ndarray] = {}
    for cohort_name, cohort_mask in _cohort_masks(site_kind, metadata).items():
        discovery_mask = cohort_mask & metadata["split"].eq("discovery").to_numpy()
        confirmation_mask = cohort_mask & metadata["split"].eq("confirmation").to_numpy()
        discovery_meta = metadata.loc[discovery_mask].reset_index(drop=True)
        confirmation_meta = metadata.loc[confirmation_mask].reset_index(drop=True)
        if len(discovery_meta) < 20 or len(confirmation_meta) < 10:
            continue
        discovery_y = discovery_meta["occurrence"].to_numpy(dtype=int)
        confirmation_y = confirmation_meta["occurrence"].to_numpy(dtype=int)
        position_prediction = _ridge_predict(
            _position_features(discovery_meta, include_gold=True),
            discovery_y,
            _position_features(confirmation_meta, include_gold=True),
            alpha=alpha,
        )
        metric_rows.append(
            {
                "site_kind": site_kind,
                "cohort": cohort_name,
                "layer": -1,
                "variant": "position_only",
                "discovery_observations": len(discovery_meta),
                "confirmation_observations": len(confirmation_meta),
                "discovery_seed_count": discovery_meta["seed"].nunique(),
                "confirmation_seed_count": confirmation_meta["seed"].nunique(),
                "discovery_cv_r2": math.nan,
                **{
                    f"confirmation_{key}": value
                    for key, value in _metric_row(
                        confirmation_y, position_prediction
                    ).items()
                },
            }
        )
        layer_cv: dict[str, dict[int, dict[str, float]]] = {
            "hidden_raw": {},
            "hidden_position_residualized": {},
        }
        for layer_axis, layer_value in enumerate(capture.layers):
            discovery_states = capture.states[discovery_mask, layer_axis]
            confirmation_states = capture.states[confirmation_mask, layer_axis]
            cv_prediction = _cross_validated_hidden_predictions(
                discovery_states,
                discovery_meta,
                rank=rank,
                alpha=alpha,
                folds=folds,
            )
            cv_metrics = _metric_row(discovery_y, cv_prediction)
            confirmation_prediction, center, basis = _fit_hidden_probe(
                discovery_states,
                discovery_y,
                confirmation_states,
                rank=rank,
                alpha=alpha,
            )
            confirmation_metrics = _metric_row(
                confirmation_y, confirmation_prediction
            )
            residual_discovery, residual_confirmation = _position_residualize(
                discovery_states,
                discovery_meta,
                confirmation_states,
                confirmation_meta,
            )
            residual_cv_prediction = _cross_validated_hidden_predictions(
                residual_discovery,
                discovery_meta,
                rank=rank,
                alpha=alpha,
                folds=folds,
            )
            residual_cv_metrics = _metric_row(
                discovery_y, residual_cv_prediction
            )
            residual_prediction, _residual_center, _residual_basis = _fit_hidden_probe(
                residual_discovery,
                discovery_y,
                residual_confirmation,
                rank=rank,
                alpha=alpha,
            )
            residual_metrics = _metric_row(confirmation_y, residual_prediction)
            layer = int(layer_value)
            layer_cv["hidden_raw"][layer] = cv_metrics
            layer_cv["hidden_position_residualized"][layer] = residual_cv_metrics
            common = {
                "site_kind": site_kind,
                "cohort": cohort_name,
                "layer": layer,
                "discovery_observations": len(discovery_meta),
                "confirmation_observations": len(confirmation_meta),
                "discovery_seed_count": discovery_meta["seed"].nunique(),
                "confirmation_seed_count": confirmation_meta["seed"].nunique(),
            }
            metric_rows.append(
                {
                    **common,
                    "variant": "hidden_raw",
                    **{f"discovery_cv_{key}": value for key, value in cv_metrics.items()},
                    **{
                        f"confirmation_{key}": value
                        for key, value in confirmation_metrics.items()
                    },
                }
            )
            metric_rows.append(
                {
                    **common,
                    "variant": "hidden_position_residualized",
                    **{
                        f"discovery_cv_{key}": value
                        for key, value in residual_cv_metrics.items()
                    },
                    **{
                        f"confirmation_{key}": value
                        for key, value in residual_metrics.items()
                    },
                }
            )
            prefix = f"{site_kind}__{cohort_name}"
            basis_arrays[f"center__{prefix}__L{layer}"] = center.astype(np.float32)
            basis_arrays[f"basis__{prefix}__L{layer}"] = basis.astype(np.float32)
            basis_arrays[f"control__{prefix}__L{layer}"] = deterministic_control_basis(
                basis,
                seed=int(random_seed) + layer,
            )
        position_row = next(
            row
            for row in metric_rows
            if row["site_kind"] == site_kind
            and row["cohort"] == cohort_name
            and row["variant"] == "position_only"
        )
        for variant in ("hidden_raw", "hidden_position_residualized"):
            selected_layer = max(
                layer_cv[variant],
                key=lambda value: (layer_cv[variant][value]["r2"], -int(value)),
            )
            selected = next(
                row
                for row in metric_rows
                if row["site_kind"] == site_kind
                and row["cohort"] == cohort_name
                and row["variant"] == variant
                and int(row["layer"]) == selected_layer
            )
            selection_rows.append(
                {
                    "site_kind": site_kind,
                    "cohort": cohort_name,
                    "selection_variant": variant,
                    "selection_rule": "max_grouped_discovery_seed_cv_r2_tie_smallest_layer",
                    "selected_layer": selected_layer,
                    "discovery_cv_r2": selected["discovery_cv_r2"],
                    "confirmation_r2": selected["confirmation_r2"],
                    "confirmation_mae": selected["confirmation_mae"],
                    "confirmation_rounded_exact": selected[
                        "confirmation_rounded_exact"
                    ],
                    "confirmation_position_only_r2": position_row[
                        "confirmation_r2"
                    ],
                    "confirmation_observations": selected[
                        "confirmation_observations"
                    ],
                    "confirmation_seed_count": selected["confirmation_seed_count"],
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(selection_rows), basis_arrays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--site-kinds", nargs="+", default=list(DEFAULT_SITE_KINDS)
    )
    parser.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260820)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_metrics: list[pd.DataFrame] = []
    all_selections: list[pd.DataFrame] = []
    basis_arrays: dict[str, np.ndarray] = {}
    site_audits = []
    for site_kind in args.site_kinds:
        site_started = time.perf_counter()
        capture = load_site_capture(
            args.capture_index.resolve(),
            site_kind=str(site_kind),
            cohort=args.cohort,
        )
        metrics, selections, arrays = analyze_site(
            capture,
            rank=args.rank,
            alpha=args.ridge_alpha,
            folds=args.folds,
            random_seed=args.random_seed,
        )
        all_metrics.append(metrics)
        all_selections.append(selections)
        overlap = sorted(set(basis_arrays) & set(arrays))
        if overlap:
            raise RuntimeError(f"Duplicate saved basis keys: {overlap[:3]}")
        basis_arrays.update(arrays)
        site_audits.append(
            {
                "site_kind": str(site_kind),
                "observations": len(capture.metadata),
                "discovery_observations": int(
                    capture.metadata["split"].eq("discovery").sum()
                ),
                "confirmation_observations": int(
                    capture.metadata["split"].eq("confirmation").sum()
                ),
                "layers": [int(value) for value in capture.layers],
                "hidden_width": int(capture.states.shape[-1]),
                "elapsed_seconds": time.perf_counter() - site_started,
            }
        )
        print(
            f"[count-representation] {site_kind}: {len(capture.metadata)} states, "
            f"{len(capture.layers)} layers",
            flush=True,
        )
    metric_frame = pd.concat(all_metrics, ignore_index=True)
    selection_frame = pd.concat(all_selections, ignore_index=True)
    metrics_path = output / "layer_metrics.csv"
    selections_path = output / "frozen_layer_confirmation.csv"
    basis_path = output / "discovery_count_bases.npz"
    _atomic_csv(metrics_path, metric_frame)
    _atomic_csv(selections_path, selection_frame)
    _atomic_npz(basis_path, **basis_arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scope": "representation_validation_not_mechanism_proof",
        "capture_index": str(args.capture_index.resolve()),
        "capture_index_sha256": _sha256(args.capture_index.resolve()),
        "cohort": args.cohort,
        "site_kinds": [str(value) for value in args.site_kinds],
        "rank": int(args.rank),
        "ridge_alpha": float(args.ridge_alpha),
        "discovery_grouping": "seed",
        "discovery_folds": int(args.folds),
        "layer_selection_uses_confirmation": False,
        "basis_fit_uses_confirmation": False,
        "confirmation_is_read_once_after_layer_freeze": True,
        "position_residualization_fit": (
            "all discovery observations only; grouped seed CV is applied to "
            "the residualized discovery states without confirmation access"
        ),
        "position_features": [
            "endpoint_token",
            "endpoint_token_squared",
            "relative_endpoint",
            "relative_endpoint_squared",
            "literal_span_tokens",
            "output_token_count",
        ],
        "position_only_additional_features": ["gold_count", "gold_count_squared"],
        "site_audits": site_audits,
        "artifacts": {
            "layer_metrics": metrics_path.name,
            "frozen_layer_confirmation": selections_path.name,
            "discovery_count_bases": basis_path.name,
            "discovery_count_bases_sha256": _sha256(basis_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    _atomic_json(output / "manifest.json", manifest)
    print(selection_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
