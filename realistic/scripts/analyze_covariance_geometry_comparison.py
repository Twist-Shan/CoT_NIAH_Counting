#!/usr/bin/env python3
"""Compare count geometry with discovery-selected covariance-aware metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.covariance_geometry import (  # noqa: E402
    SELECTION_METRICS,
    evaluate_covariance_geometry_layer,
    select_discovery_winners,
)
from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    CLASSES,
    ModeDataset,
    load_native_thinking_capture,
    load_non_thinking_capture,
)
from realistic_niah_v5.dual_endpoint_geometry import (  # noqa: E402
    load_native_thinking_final_count,
    load_non_thinking_final_count,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-thinking-root",
        type=Path,
        default=ROOT / "work" / "nonthinking_v44_geometry_300_150_136_166_78",
    )
    parser.add_argument(
        "--native-root",
        type=Path,
        default=ROOT / "work" / "v5_geometry_full_panel",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "covariance_geometry_comparison",
    )
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--relative-ridge", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--discovery-cv-folds", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def dataset_specs(
    non_thinking_root: Path, native_root: Path, model_label: str
) -> list[tuple[str, str, Path, Any]]:
    non_model = non_thinking_root / model_label / "numeric" / "representation"
    native_running = native_root / "running" / model_label / "capture_index.jsonl"
    native_final = native_root / "final" / model_label / "capture_index.jsonl"
    non_running = non_model / "capture" / "capture_index.jsonl"
    non_final = non_model / "answer_query_all_layers_v1" / "capture_index.jsonl"
    return [
        (
            "running_index",
            "non_thinking",
            non_running,
            lambda: load_non_thinking_capture(non_running, pooling="span_end"),
        ),
        (
            "running_index",
            "native_thinking",
            native_running,
            lambda: load_native_thinking_capture(
                native_running, site_kind="item_end", cohort="parser_hit"
            ),
        ),
        (
            "final_count",
            "non_thinking",
            non_final,
            lambda: load_non_thinking_final_count(non_final),
        ),
        (
            "final_count",
            "native_thinking",
            native_final,
            lambda: load_native_thinking_final_count(native_final),
        ),
    ]


def analyze_dataset(
    dataset: ModeDataset,
    *,
    endpoint: str,
    pca_dim: int,
    random_state: int,
    relative_ridge: float,
    discovery_cv_folds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in sorted(dataset.states_by_layer):
        metrics = evaluate_covariance_geometry_layer(
            dataset.states_by_layer[layer],
            dataset.metadata,
            CLASSES,
            pca_dim=pca_dim,
            random_state=random_state,
            relative_ridge=relative_ridge,
            discovery_cv_folds=discovery_cv_folds,
        )
        metrics.pop("metric_definitions", None)
        rows.append(
            {
                "model_label": dataset.model_label,
                "endpoint": endpoint,
                "mode": dataset.mode,
                "layer": int(layer),
                **metrics,
            }
        )
    return rows


def comparison_rows(selected: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, endpoint, selector), frame in selected.groupby(
        ["model_label", "endpoint", "selector"], sort=True
    ):
        by_mode = frame.set_index("mode")
        non = by_mode.loc["non_thinking"]
        native = by_mode.loc["native_thinking"]
        rows.append(
            {
                "model_label": model,
                "endpoint": endpoint,
                "selector": selector,
                "non_thinking_layer": int(non["selected_layer"]),
                "native_thinking_layer": int(native["selected_layer"]),
                "non_thinking_confirmation": float(non["confirmation_value"]),
                "native_thinking_confirmation": float(native["confirmation_value"]),
                "native_minus_non": float(
                    native["confirmation_value"] - non["confirmation_value"]
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for model_label in MODELS:
        for endpoint, _mode, capture_index, loader in dataset_specs(
            args.non_thinking_root, args.native_root, model_label
        ):
            if not capture_index.exists():
                raise FileNotFoundError(capture_index)
            inputs[str(capture_index.resolve())] = sha256(capture_index)
            dataset = loader()
            all_rows.extend(
                analyze_dataset(
                    dataset,
                    endpoint=endpoint,
                    pca_dim=args.pca_dim,
                    random_state=args.random_state,
                    relative_ridge=args.relative_ridge,
                    discovery_cv_folds=args.discovery_cv_folds,
                )
            )
    per_layer = pd.DataFrame(all_rows).sort_values(
        ["model_label", "endpoint", "mode", "layer"]
    )
    selected = select_discovery_winners(per_layer).sort_values(
        ["model_label", "endpoint", "selector", "mode"]
    )
    comparisons = pd.DataFrame(comparison_rows(selected)).sort_values(
        ["model_label", "endpoint", "selector"]
    )
    per_layer_path = args.output / "per_layer_metrics.csv"
    selected_path = args.output / "discovery_selected_metrics.csv"
    comparisons_path = args.output / "confirmation_mode_comparisons.csv"
    atomic_csv(per_layer_path, per_layer)
    atomic_csv(selected_path, selected)
    atomic_csv(comparisons_path, comparisons)
    audit = {
        "schema_version": "niah_covariance_geometry_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS),
        "classes": list(CLASSES),
        "pca_dim": int(args.pca_dim),
        "relative_covariance_ridge": float(args.relative_ridge),
        "discovery_cv_folds": int(args.discovery_cv_folds),
        "selection_rule": (
            "each model x endpoint x mode x metric selects its layer by a "
            "seed-grouped discovery OOF value; PCA is fitted once on all "
            "discovery rows per layer; confirmation never enters ranking and "
            "only the frozen-layer confirmation value is interpreted"
        ),
        "metric_definitions": {
            "isotropic_snr": "trace(Sigma_B) / trace(Sigma_W) after discovery-fitted PCA16 whitening",
            "fisher_trace": "held-out Sigma_B measured by a within-covariance precision fitted on separate discovery rows after discovery-fitted PCA16",
            "mahalanobis_silhouette": "class-balanced silhouette after discovery within-covariance whitening",
            "ordinal_rsa": "Spearman rho between 45 centroid distances and absolute count gaps in the frozen Mahalanobis metric",
        },
        "selection_metrics": {
            key: {"discovery": value[0], "confirmation": value[1]}
            for key, value in SELECTION_METRICS.items()
        },
        "inputs": inputs,
        "outputs": {
            str(per_layer_path.resolve()): sha256(per_layer_path),
            str(selected_path.resolve()): sha256(selected_path),
            str(comparisons_path.resolve()): sha256(comparisons_path),
        },
    }
    atomic_json(args.output / "audit.json", audit)
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
