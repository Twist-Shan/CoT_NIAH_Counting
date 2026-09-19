#!/usr/bin/env python3
"""Pair native grammar-filtered running states with non-thinking states.

The native grammar and native layer are frozen from the existing discovery-only
clean-grammar selection.  The same split/seed/gold-N/running-k cells are then
selected from non-thinking, whose layer is selected independently on those
paired discovery rows.  Confirmation labels never select a grammar or layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v5.causal_aligned_geometry import (  # noqa: E402
    load_causal_aligned_native_capture,
)
from realistic_niah_v5.covariance_geometry import (  # noqa: E402
    class_balanced_silhouette,
    ordinal_centroid_rsa,
)
from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    CLASSES,
    ModeDataset,
    load_non_thinking_capture,
)
from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
KEY_COLUMNS = ("split", "seed", "gold_count", "occurrence")
SCHEMA = "realistic_niah_v5_grammar_filtered_cross_mode_geometry_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subset(dataset: ModeDataset, indices: np.ndarray) -> ModeDataset:
    result = ModeDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=dataset.metadata.iloc[indices].reset_index(drop=True),
        states_by_layer={layer: values[indices] for layer, values in dataset.states_by_layer.items()},
    )
    result.validate()
    return result


def _sort_indices(frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.reset_index().sort_values(list(KEY_COLUMNS), kind="mergesort")
    return ordered["index"].to_numpy(dtype=int)


def paired_datasets(
    native: ModeDataset,
    non_thinking: ModeDataset,
    grammar: str,
) -> tuple[ModeDataset, ModeDataset]:
    native_mask = native.metadata["grammar_class"].astype(str).eq(grammar).to_numpy()
    native_filtered = subset(native, np.flatnonzero(native_mask))
    native_filtered = subset(native_filtered, _sort_indices(native_filtered.metadata))
    native_keys = {
        tuple(row)
        for row in native_filtered.metadata[list(KEY_COLUMNS)].itertuples(index=False, name=None)
    }
    if len(native_keys) != len(native_filtered.metadata):
        raise ValueError(f"Native filtered keys are not unique for {native.model_label}/{grammar}")
    non_keys = non_thinking.metadata[list(KEY_COLUMNS)].apply(tuple, axis=1)
    non_mask = non_keys.isin(native_keys).to_numpy()
    non_filtered = subset(non_thinking, np.flatnonzero(non_mask))
    non_filtered = subset(non_filtered, _sort_indices(non_filtered.metadata))
    left = list(native_filtered.metadata[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    right = list(non_filtered.metadata[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    if left != right:
        raise ValueError(
            f"Paired event cells disagree for {native.model_label}/{grammar}: "
            f"native={len(left)}, non-thinking={len(right)}"
        )
    non_filtered.metadata["request_id"] = native_filtered.metadata["request_id"].astype(str)
    non_filtered.metadata["grammar_class"] = grammar
    return native_filtered, non_filtered


def layer_metrics(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    pca_dim: int,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    return {
        **grouped_discovery_cv_metrics(
            states,
            metadata,
            CLASSES,
            pca_dim=pca_dim,
            folds=folds,
            random_state=seed,
            pca_whiten=True,
        ),
        **confirmation_metrics(
            states,
            metadata,
            CLASSES,
            pca_dim=pca_dim,
            random_state=seed,
            pca_whiten=True,
        ),
    }


def choose_layer(
    dataset: ModeDataset,
    *,
    pca_dim: int,
    folds: int,
    seed: int,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for layer, states in sorted(dataset.states_by_layer.items()):
        candidates.append(
            {
                "layer": int(layer),
                **layer_metrics(
                    states,
                    dataset.metadata,
                    pca_dim=pca_dim,
                    folds=folds,
                    seed=seed,
                ),
            }
        )
    winner = sorted(
        candidates,
        key=lambda row: (
            -float(row["discovery_selection_score"]),
            -float(row["discovery_oof_ncc_balanced_accuracy"]),
            -float(row["discovery_oof_logistic_balanced_accuracy"]),
            int(row["layer"]),
        ),
    )[0]
    return int(winner["layer"]), winner, candidates


def geometry_payload(
    dataset: ModeDataset,
    layer: int,
    metrics: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    discovery = dataset.metadata["split"].astype(str).eq("discovery").to_numpy()
    scaler = StandardScaler().fit(dataset.states_by_layer[layer][discovery].astype(np.float32))
    scaled = scaler.transform(dataset.states_by_layer[layer].astype(np.float32))
    pca = PCA(n_components=3, random_state=seed).fit(scaled[discovery])
    xyz = pca.transform(scaled)
    confirmation = dataset.metadata["split"].astype(str).eq("confirmation").to_numpy()
    labels = dataset.metadata.loc[confirmation, "occurrence"].to_numpy(dtype=int)
    silhouette = class_balanced_silhouette(xyz[confirmation], labels, CLASSES)
    ordinal_rsa = ordinal_centroid_rsa(xyz[confirmation], labels, CLASSES)
    points = []
    for value, row in zip(xyz, dataset.metadata.itertuples(index=False)):
        points.append(
            {
                "x": float(value[0]),
                "y": float(value[1]),
                "z": float(value[2]),
                "split": str(row.split),
                "seed": int(row.seed),
                "gold_count": int(row.gold_count),
                "occurrence": int(row.occurrence),
                "request_id": str(row.request_id),
            }
        )
    return {
        "layer": int(layer),
        "pca3_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "confirmation_pca3_class_balanced_silhouette": float(silhouette),
        "confirmation_pca3_ordinal_rsa": float(ordinal_rsa),
        "metrics": metrics,
        "points": points,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-thinking-root",
        type=Path,
        default=ROOT / "work/nonthinking_v44_geometry_300_150_136_166_78",
    )
    parser.add_argument(
        "--native-running-root",
        type=Path,
        default=ROOT / "work/v5_geometry_full_panel/running",
    )
    parser.add_argument(
        "--event-registry",
        type=Path,
        default=ROOT / "reports/v5_native_causal_site_review/event_registry.csv",
    )
    parser.add_argument(
        "--native-selection",
        type=Path,
        default=ROOT / "reports/v5_native_clean_grammar_geometry/selected_clean_grammar.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/v5_grammar_filtered_cross_mode_geometry",
    )
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = pd.read_csv(args.native_selection)
    if set(selection["model_label"].astype(str)) != set(MODELS):
        raise ValueError("Native clean-grammar selection must contain exactly both models")
    payload: dict[str, Any] = {"schema_version": SCHEMA, "models": {}}
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for model in MODELS:
        selected = selection.loc[selection["model_label"].astype(str).eq(model)]
        if len(selected) != 1:
            raise ValueError(f"Expected one frozen native winner for {model}")
        selected_row = selected.iloc[0]
        grammar = str(selected_row["grammar_class"])
        native_layer = int(selected_row["layer"])
        native, native_audit = load_causal_aligned_native_capture(
            args.native_running_root / model / "capture_index.jsonl",
            args.event_registry,
            site_kind="item_end",
        )
        non = load_non_thinking_capture(
            args.non_thinking_root
            / model
            / "numeric/representation/capture/capture_index.jsonl",
            design_variant="v4.4",
            pooling="span_end",
        )
        native, non = paired_datasets(native, non, grammar)
        native_metrics = layer_metrics(
            native.states_by_layer[native_layer],
            native.metadata,
            pca_dim=args.pca_dim,
            folds=args.folds,
            seed=args.seed,
        )
        non_layer, non_metrics, non_candidates = choose_layer(
            non,
            pca_dim=args.pca_dim,
            folds=args.folds,
            seed=args.seed,
        )
        native_discovery_score = float(selected_row["discovery_selection_score"])
        if not np.isclose(
            native_discovery_score,
            float(native_metrics["discovery_selection_score"]),
            atol=1e-8,
        ):
            raise ValueError(
                f"Frozen native winner changed after exact pairing for {model}: "
                f"{native_discovery_score} vs {native_metrics['discovery_selection_score']}"
            )
        model_payload = {
            "grammar_class": grammar,
            "pairing_key": list(KEY_COLUMNS),
            "native_thinking": geometry_payload(
                native, native_layer, native_metrics, seed=args.seed
            ),
            "non_thinking": geometry_payload(non, non_layer, non_metrics, seed=args.seed),
            "native_loader_audit": native_audit,
        }
        payload["models"][model] = model_payload
        for mode, dataset, layer, metrics in (
            ("non_thinking", non, non_layer, non_metrics),
            ("native_thinking", native, native_layer, native_metrics),
        ):
            support = (
                dataset.metadata.groupby(["split", "occurrence"]).size().rename("n").reset_index()
            )
            metric_rows.append(
                {
                    "model_label": model,
                    "mode": mode,
                    "grammar_class": grammar,
                    "layer": int(layer),
                    "states": int(len(dataset.metadata)),
                    "trajectories": int(
                        dataset.metadata[list(KEY_COLUMNS[:-1])].drop_duplicates().shape[0]
                    ),
                    "confirmation_min_per_class": int(
                        support.loc[support["split"].astype(str).eq("confirmation"), "n"].min()
                    ),
                    **metrics,
                    "confirmation_pca3_class_balanced_silhouette": model_payload[mode][
                        "confirmation_pca3_class_balanced_silhouette"
                    ],
                    "confirmation_pca3_ordinal_rsa": model_payload[mode][
                        "confirmation_pca3_ordinal_rsa"
                    ],
                }
            )
        for row in non_candidates:
            candidate_rows.append({"model_label": model, "mode": "non_thinking", **row})
        print(
            model,
            grammar,
            f"non L{non_layer}={non_metrics['confirmation_logistic_balanced_accuracy']:.3f}/"
            f"{non_metrics['confirmation_ncc_balanced_accuracy']:.3f}",
            f"native L{native_layer}={native_metrics['confirmation_logistic_balanced_accuracy']:.3f}/"
            f"{native_metrics['confirmation_ncc_balanced_accuracy']:.3f}",
        )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "paired_metrics.csv"
    candidates_path = args.output / "nonthinking_layer_candidates.csv"
    payload_path = args.output / "geometry_payload.json"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_path, index=False)
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    audit = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filter_selection": (
            "native grammar and native layer frozen from grouped discovery OOF; "
            "non-thinking uses exactly paired event cells and independently selects its layer "
            "with grouped discovery OOF; confirmation evaluates only"
        ),
        "pairing_key": list(KEY_COLUMNS),
        "inputs": {
            str(args.event_registry.resolve()): sha256(args.event_registry),
            str(args.native_selection.resolve()): sha256(args.native_selection),
        },
        "outputs": {
            str(path.resolve()): sha256(path)
            for path in (metrics_path, candidates_path, payload_path)
        },
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
