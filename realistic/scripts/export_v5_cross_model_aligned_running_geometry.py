#!/usr/bin/env python3
"""Export PCA3 display coordinates for exact common-support item-end states."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal_aligned_geometry import (  # noqa: E402
    load_causal_aligned_native_capture,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def keys(frame: pd.DataFrame) -> list[tuple[str, int, int, int]]:
    return [
        (str(row.split), int(row.seed), int(row.gold_count), int(row.occurrence))
        for row in frame.itertuples(index=False)
    ]


def digest(values: set[tuple[str, int, int, int]]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        "--aligned-analysis-root",
        type=Path,
        default=ROOT / "reports/v5_native_cross_model_aligned_geometry",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    datasets: dict[str, Any] = {}
    key_lists: dict[str, list[tuple[str, int, int, int]]] = {}
    for model in MODELS:
        dataset, _audit = load_causal_aligned_native_capture(
            args.native_running_root / model / "capture_index.jsonl",
            args.event_registry,
            site_kind="item_end",
        )
        datasets[model] = dataset
        key_lists[model] = keys(dataset.metadata)
    common = set(key_lists[MODELS[0]]) & set(key_lists[MODELS[1]])
    selected = pd.read_csv(args.aligned_analysis_root / "site_selected.csv")
    payload: dict[str, Any] = {
        "schema_version": "realistic_niah_v5_cross_model_aligned_pca3_v1",
        "status": "PASS",
        "exact_cross_model_sample_alignment": True,
        "alignment_key": ["split", "seed", "gold_count", "occurrence"],
        "common_key_sha256": digest(common),
        "models": {},
    }
    for model in MODELS:
        dataset = datasets[model]
        mask = np.asarray([key in common for key in key_lists[model]], dtype=bool)
        metadata = dataset.metadata.loc[mask].reset_index(drop=True)
        discovery = metadata["split"].astype(str).eq("discovery").to_numpy()
        confirmation = metadata["split"].astype(str).eq("confirmation").to_numpy()
        layers: dict[str, Any] = {}
        for layer, full_states in sorted(dataset.states_by_layer.items()):
            states = np.asarray(full_states, dtype=np.float32)[mask]
            scaler = StandardScaler().fit(states[discovery])
            scaled = scaler.transform(states)
            pca = PCA(n_components=3, random_state=0).fit(scaled[discovery])
            coordinates = pca.transform(scaled[confirmation])
            meta = metadata.loc[confirmation].reset_index(drop=True)
            rows = [
                [
                    int(row.seed),
                    int(row.occurrence),
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(row.gold_count),
                ]
                for row, point in zip(meta.itertuples(index=False), coordinates)
            ]
            layers[str(int(layer))] = {
                "evr": [float(value) for value in pca.explained_variance_ratio_],
                "rows": rows,
            }
        winner = selected.loc[selected["model_label"].astype(str).eq(model)]
        if len(winner) != 1:
            raise ValueError(f"Expected one aligned representation winner for {model}")
        payload["models"][model] = {
            "default_layer": int(winner.iloc[0]["layer"]),
            "layers": layers,
            "discovery_rows": int(discovery.sum()),
            "confirmation_rows": int(confirmation.sum()),
            "token_site": "causal item_end / p0_item_end exact commit token",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "common_keys": len(common),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
