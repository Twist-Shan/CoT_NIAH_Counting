from __future__ import annotations

"""Evaluate whether the prompt running-index curve is endpoint-gated.

Discovery endpoint states define a frozen PCA space and count-centroid curve.
The same map is then applied to confirmation needle interiors, hard negatives,
and depth-stratified ordinary passage tokens from the same N=10 prompts.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


EPS = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def prompt_layer(packed_root: Path, model: str, layer: int) -> dict[str, np.ndarray]:
    return load_npz(packed_root / "layers" / f"{model}__prompt_running__L{layer:02d}.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    index = read_jsonl(args.capture / "capture_index.jsonl")
    metric_rows: list[dict[str, Any]] = []
    formula_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for model in sorted({str(row["model_label"]) for row in index}):
        model_rows = [row for row in index if str(row["model_label"]) == model]
        layers = list(map(int, model_rows[0]["layers"]))
        loaded = [(row, load_npz(args.capture / row["path"])) for row in model_rows]
        for layer_index, layer in enumerate(layers):
            prompt = prompt_layer(args.packed_root, model, layer)
            discovery = prompt["split"].astype(str) == "discovery"
            x_train = prompt["states"][discovery].astype(np.float32)
            y_train = prompt["count"][discovery].astype(int)
            pca = PCA(n_components=min(32, len(x_train) - 1), svd_solver="randomized", random_state=20260806)
            pca.fit(x_train)
            endpoint_centroids = {count: x_train[y_train == count].mean(axis=0).astype(np.float64) for count in range(1, 11)}
            endpoint_mean = np.stack(list(endpoint_centroids.values())).mean(axis=0)
            gamma = {count: endpoint_centroids[count] - endpoint_mean for count in range(1, 11)}
            endpoint_centroids_z = {count: pca.transform(endpoint_centroids[count][None].astype(np.float32))[0] for count in range(1, 11)}

            discovery_tokens: dict[str, list[np.ndarray]] = {}
            confirmation_records: list[dict[str, Any]] = []
            for row, payload in loaded:
                states = payload["states"][layer_index].astype(np.float32)
                categories = payload["categories"].astype(str)
                if str(row["split"]) == "discovery":
                    for category in np.unique(categories):
                        discovery_tokens.setdefault(category, []).append(states[categories == category])
                else:
                    z = pca.transform(states)
                    for i in range(len(states)):
                        confirmation_records.append(
                            {
                                "seed": int(row["seed"]),
                                "category": str(categories[i]),
                                "state": states[i].astype(np.float64),
                                "z": z[i],
                                "prefix_count": int(payload["prefix_count"][i]),
                                "occurrence_index": int(payload["occurrence_index"][i]),
                                "position": int(payload["positions"][i]),
                            }
                        )
            category_baseline = {category: np.concatenate(chunks, axis=0).astype(np.float64).mean(axis=0) for category, chunks in discovery_tokens.items()}

            by_category: dict[str, list[dict[str, Any]]] = {}
            for record in confirmation_records:
                by_category.setdefault(record["category"], []).append(record)
            for category, records in sorted(by_category.items()):
                valid = [record for record in records if 1 <= record["prefix_count"] <= 10]
                if valid:
                    gold = np.asarray([record["prefix_count"] for record in valid])
                    projected = np.stack([record["z"] for record in valid])
                    distances = np.stack(
                        [np.square(projected - endpoint_centroids_z[count]).sum(axis=1) for count in range(1, 11)],
                        axis=1,
                    )
                    predicted = 1 + np.argmin(distances, axis=1)
                    accuracy = float(np.mean(predicted == gold))
                    mae = float(np.mean(np.abs(predicted - gold)))
                else:
                    accuracy = float("nan")
                    mae = float("nan")
                energy = np.asarray([float(np.square(record["z"][:3]).sum()) for record in records])
                metric_rows.append(
                    {
                        "model_label": model,
                        "layer": layer,
                        "category": category,
                        "rows": len(records),
                        "seeds": len({record["seed"] for record in records}),
                        "nearest_endpoint_count_accuracy": accuracy,
                        "nearest_endpoint_count_mae": mae,
                        "rank3_projection_energy_mean": float(energy.mean()),
                        "rank3_projection_energy_median": float(np.median(energy)),
                    }
                )

            # Full-space held-out comparison of baseline, endpoint-gated, and ungated curves.
            for category, records in sorted(by_category.items()):
                baseline = category_baseline[category]
                observed = np.stack([record["state"] for record in records])
                base_prediction = np.broadcast_to(baseline, observed.shape)
                gated_prediction = base_prediction.copy()
                ungated_prediction = base_prediction.copy()
                span_gated_prediction = base_prediction.copy()
                for i, record in enumerate(records):
                    prefix = record["prefix_count"]
                    occurrence = record["occurrence_index"]
                    if category == "needle_endpoint" and 1 <= occurrence <= 10:
                        gated_prediction[i] += gamma[occurrence]
                    if category in {"needle_endpoint", "needle_interior"} and 1 <= occurrence <= 10:
                        span_gated_prediction[i] += gamma[occurrence]
                    if 1 <= prefix <= 10:
                        ungated_prediction[i] += gamma[prefix]
                sse_base = float(np.square(observed - base_prediction).sum())
                for name, prediction in (
                    ("category_baseline", base_prediction),
                    ("endpoint_gated_curve", gated_prediction),
                    ("needle_span_gated_curve", span_gated_prediction),
                    ("ungated_prefix_curve", ungated_prediction),
                ):
                    sse = float(np.square(observed - prediction).sum())
                    formula_rows.append(
                        {
                            "model_label": model,
                            "layer": layer,
                            "category": category,
                            "model": name,
                            "rows": len(records),
                            "sse": sse,
                            "incremental_r2_vs_category_baseline": 1.0 - sse / max(sse_base, EPS),
                        }
                    )

            # Compact confirmation projection table for report scatter plots.
            for record in confirmation_records:
                projection_rows.append(
                    {
                        "model_label": model,
                        "layer": layer,
                        "seed": record["seed"],
                        "category": record["category"],
                        "prefix_count": record["prefix_count"],
                        "occurrence_index": record["occurrence_index"],
                        "position": record["position"],
                        "pc1": float(record["z"][0]),
                        "pc2": float(record["z"][1]),
                        "pc3": float(record["z"][2]),
                    }
                )
            print(f"[all-token-analysis] {model} L{layer}", flush=True)

    pd.DataFrame(metric_rows).to_csv(args.output / "all_token_category_metrics.csv", index=False)
    pd.DataFrame(formula_rows).to_csv(args.output / "gated_curve_formula_tests.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(args.output / "all_token_frozen_pca_projections.csv.gz", index=False, compression="gzip")
    audit = {
        "schema_version": "realistic_niah_v4_4_all_token_control_analysis_v1",
        "fit_split": "discovery",
        "evaluation_split": "confirmation",
        "basis": "PCA32 fit to prompt needle-end states only",
        "curve": "ten discovery needle-end count centroids, centered by their mean",
        "formula_models": {
            "category_baseline": "discovery mean separately for each token category",
            "endpoint_gated_curve": "baseline + gamma(occurrence) only at needle endpoints",
            "needle_span_gated_curve": "baseline + gamma(occurrence) at every needle-span token",
            "ungated_prefix_curve": "baseline + gamma(completed-prefix-count) at all token categories",
        },
        "metric_rows": len(metric_rows),
        "formula_rows": len(formula_rows),
        "projection_rows": len(projection_rows),
        "status": "PASS",
    }
    (args.output / "all_token_analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
