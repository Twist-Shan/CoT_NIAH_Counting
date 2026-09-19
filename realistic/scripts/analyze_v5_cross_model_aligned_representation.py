#!/usr/bin/env python3
"""Reanalyse Native item-end count geometry on exact cross-model common support."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal_aligned_geometry import (  # noqa: E402
    load_causal_aligned_native_capture,
)
from realistic_niah_v5.covariance_geometry import (  # noqa: E402
    evaluate_covariance_geometry_layer,
)
from realistic_niah_v5.cross_mode_geometry import CLASSES  # noqa: E402
from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
SITE = "item_end"
SCHEMA = "realistic_niah_v5_cross_model_aligned_representation_v1"


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def key_series(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: (
            str(row["split"]),
            int(row["seed"]),
            int(row["gold_count"]),
            int(row["occurrence"]),
        ),
        axis=1,
    )


def key_digest(keys: set[tuple[str, int, int, int]]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(keys), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def select_one(frame: pd.DataFrame) -> pd.Series:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--legacy-root",
        type=Path,
        default=ROOT / "reports/v5_native_causal_aligned_geometry",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/v5_native_cross_model_aligned_geometry",
    )
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--discovery-cv-folds", type=int, default=5)
    parser.add_argument("--relative-ridge", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    datasets: dict[str, Any] = {}
    loader_audits: dict[str, Any] = {}
    key_maps: dict[str, pd.Series] = {}
    for model in MODELS:
        index = args.native_running_root / model / "capture_index.jsonl"
        dataset, audit = load_causal_aligned_native_capture(
            index, args.event_registry, site_kind=SITE
        )
        keys = key_series(dataset.metadata)
        duplicates = keys[keys.duplicated(keep=False)]
        if not duplicates.empty:
            raise ValueError(f"{model} has duplicate aligned state keys")
        datasets[model] = dataset
        loader_audits[model] = audit
        key_maps[model] = keys

    common = set(key_maps[MODELS[0]]) & set(key_maps[MODELS[1]])
    if not common:
        raise ValueError("Cross-model causal item-end support is empty")
    phase_counts = collections.Counter(key[0] for key in common)
    occurrence_counts = collections.Counter((key[0], key[3]) for key in common)
    if set(key[3] for key in common) != set(CLASSES):
        raise ValueError("Common support does not cover all ten count classes")
    if min(
        count for (split, _occurrence), count in occurrence_counts.items()
        if split == "confirmation"
    ) < 3:
        raise ValueError("Common confirmation support has fewer than three rows/class")

    candidates: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    aligned_support: dict[str, Any] = {}
    for model in MODELS:
        dataset = datasets[model]
        mask = key_maps[model].isin(common).to_numpy(dtype=bool)
        metadata = dataset.metadata.loc[mask].reset_index(drop=True)
        observed_keys = set(key_series(metadata))
        if observed_keys != common or len(metadata) != len(common):
            raise ValueError(f"{model} failed exact common-support filtering")
        support = (
            metadata.groupby(["split", "occurrence"], sort=True)
            .size()
            .rename("count")
            .reset_index()
        )
        aligned_support[model] = {
            "rows": int(len(metadata)),
            "trajectories": int(
                metadata[["split", "seed", "gold_count"]]
                .drop_duplicates()
                .shape[0]
            ),
            "support": support.to_dict(orient="records"),
            "key_multiset_sha256": key_digest(observed_keys),
        }
        model_rows: list[dict[str, Any]] = []
        for layer, full_states in sorted(dataset.states_by_layer.items()):
            states = np.asarray(full_states)[mask]
            discovery = grouped_discovery_cv_metrics(
                states,
                metadata,
                CLASSES,
                pca_dim=int(args.pca_dim),
                random_state=int(args.random_state),
                folds=int(args.discovery_cv_folds),
                pca_whiten=True,
            )
            confirmation = confirmation_metrics(
                states,
                metadata,
                CLASSES,
                pca_dim=int(args.pca_dim),
                random_state=int(args.random_state),
                pca_whiten=True,
            )
            row = {
                "model_label": model,
                "site_kind": SITE,
                "layer": int(layer),
                "pca_dim": int(args.pca_dim),
                "exact_cross_model_sample_alignment": True,
                "common_key_sha256": key_digest(common),
                **discovery,
                **confirmation,
            }
            candidates.append(row)
            model_rows.append(row)
        winner = select_one(pd.DataFrame(model_rows)).to_dict()
        layer = int(winner["layer"])
        covariance = evaluate_covariance_geometry_layer(
            np.asarray(dataset.states_by_layer[layer])[mask],
            metadata,
            CLASSES,
            pca_dim=int(args.pca_dim),
            random_state=int(args.random_state),
            relative_ridge=float(args.relative_ridge),
            discovery_cv_folds=int(args.discovery_cv_folds),
        )
        covariance.pop("metric_definitions", None)
        selected.append(
            {**winner, **{f"cov_{key}": value for key, value in covariance.items()}}
        )
        print(
            f"[aligned-representation] {model} L{layer} "
            f"rows={len(metadata)} "
            f"conf-log={float(winner['confirmation_logistic_balanced_accuracy']):.3f}",
            flush=True,
        )
        del datasets[model]
        gc.collect()

    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["model_label", "layer"], kind="mergesort"
    )
    selected_frame = pd.DataFrame(selected).sort_values(
        "model_label", kind="mergesort"
    )
    legacy = pd.read_csv(args.legacy_root / "site_selected.csv")
    comparison_rows: list[dict[str, Any]] = []
    for model in MODELS:
        old = legacy.loc[
            legacy["model_label"].astype(str).eq(model)
            & legacy["site_kind"].astype(str).eq(SITE)
        ].iloc[0]
        current = selected_frame.loc[
            selected_frame["model_label"].astype(str).eq(model)
        ].iloc[0]
        comparison_rows.append(
            {
                "model_label": model,
                "legacy_scope": "model-specific causal progress support",
                "causal_scope": "exact cross-model common (phase,seed,N,k) support",
                "legacy_layer": int(old["layer"]),
                "causal_layer": int(current["layer"]),
                "legacy_confirmation_logistic_balanced_accuracy": float(
                    old["confirmation_logistic_balanced_accuracy"]
                ),
                "causal_confirmation_logistic_balanced_accuracy": float(
                    current["confirmation_logistic_balanced_accuracy"]
                ),
                "causal_minus_legacy_logistic": float(
                    current["confirmation_logistic_balanced_accuracy"]
                    - old["confirmation_logistic_balanced_accuracy"]
                ),
                "legacy_confirmation_ncc_balanced_accuracy": float(
                    old["confirmation_ncc_balanced_accuracy"]
                ),
                "causal_confirmation_ncc_balanced_accuracy": float(
                    current["confirmation_ncc_balanced_accuracy"]
                ),
                "causal_minus_legacy_ncc": float(
                    current["confirmation_ncc_balanced_accuracy"]
                    - old["confirmation_ncc_balanced_accuracy"]
                ),
                "legacy_confirmation_snr_db": float(
                    old["confirmation_class_balanced_snr_db"]
                ),
                "causal_confirmation_snr_db": float(
                    current["confirmation_class_balanced_snr_db"]
                ),
                "causal_minus_legacy_snr_db": float(
                    current["confirmation_class_balanced_snr_db"]
                    - old["confirmation_class_balanced_snr_db"]
                ),
                "legacy_confirmation_noise_power": float(old["confirmation_noise_power"]),
                "causal_confirmation_noise_power": float(current["confirmation_noise_power"]),
            }
        )

    output = args.output.resolve()
    atomic_csv(output / "site_layer_candidates.csv", candidate_frame)
    atomic_csv(output / "site_selected.csv", selected_frame)
    atomic_csv(output / "model_site_winners.csv", selected_frame)
    atomic_csv(
        output / "legacy_vs_causal_item_end.csv", pd.DataFrame(comparison_rows)
    )
    audit = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "site_kind": SITE,
        "alignment_key": ["split", "seed", "gold_count", "occurrence"],
        "exact_cross_model_sample_alignment": True,
        "common_key_sha256": key_digest(common),
        "common_state_rows": len(common),
        "phase_state_rows": dict(sorted(phase_counts.items())),
        "common_trajectory_cells": len({key[:3] for key in common}),
        "support": aligned_support,
        "loader_audits_before_common_filter": loader_audits,
        "selection_rule": (
            "within each architecture, select layer using only grouped-CV "
            "discovery rows after freezing the exact cross-model common support"
        ),
    }
    atomic_json(output / "audit.json", audit)
    print(json.dumps({"status": "PASS", "audit": audit}, sort_keys=True))


if __name__ == "__main__":
    main()
