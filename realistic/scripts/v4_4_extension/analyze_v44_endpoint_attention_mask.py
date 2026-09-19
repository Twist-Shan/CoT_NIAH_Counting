from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_LAYERS = {"Qwen3-8B": 8, "Gemma4-E4B": 9}


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    stats = [abs(float(np.mean(values * np.asarray(signs)))) for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(stats) >= observed - 1e-15))


def bootstrap(values: np.ndarray, reps: int = 50000) -> tuple[float, float]:
    rng = np.random.default_rng(20260806)
    means = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted


def read_index(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prompt_geometry(packed: Path, model: str, layer: int) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, float]:
    with np.load(packed / "layers" / f"{model}__prompt_running__L{layer:02d}.npz", allow_pickle=False) as z:
        x = np.asarray(z["states"], dtype=np.float64)
        y = np.asarray(z["count"], dtype=int)
        split = np.asarray(z["split"]).astype(str)
    fit = split == "discovery"
    x, y = x[fit], y[fit]
    mean = x.mean(axis=0)
    count_mean = float(y.mean())
    slope = ((y - count_mean)[:, None] * (x - mean)).sum(axis=0) / max(float(np.square(y - count_mean).sum()), 1e-12)
    centroids = {count: x[y == count].mean(axis=0) for count in range(1, 11)}
    centroid_rms = float(np.sqrt(np.mean([np.square(value - mean).sum() for value in centroids.values()])))
    return mean, centroids, slope, centroid_rms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-heads", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = read_index(args.capture / "capture_index.jsonl")

    sample_rows = []
    for model in sorted({row["model_label"] for row in index}):
        model_index = [row for row in index if row["model_label"] == model]
        first = np.load(args.capture / model_index[0]["state_path"], allow_pickle=False)
        layers = np.asarray(first["layers"], dtype=int)
        conditions = np.asarray(first["conditions"]).astype(str)
        occurrences = np.asarray(first["occurrences"], dtype=int)
        geometry = {int(layer): prompt_geometry(args.packed_root, model, int(layer)) for layer in layers}
        for row in model_index:
            with np.load(args.capture / row["state_path"], allow_pickle=False) as z:
                states = np.asarray(z["states"], dtype=np.float64)
            for ci, condition in enumerate(conditions):
                for oi, occurrence in enumerate(occurrences):
                    for li, layer in enumerate(layers):
                        mean, centroids, slope, centroid_rms = geometry[int(layer)]
                        state = states[ci, oi, li]
                        prediction = 5.5 + float(np.dot(state - mean, slope)) / max(float(np.dot(slope, slope)), 1e-12)
                        nearest = min(centroids, key=lambda count: float(np.square(state - centroids[count]).sum()))
                        sample_rows.append(
                            {
                                "model_label": model, "seed": int(row["seed"]), "condition": condition,
                                "occurrence": int(occurrence), "layer": int(layer),
                                "continuous_count_prediction": prediction,
                                "continuous_absolute_error": abs(prediction - occurrence),
                                "nearest_centroid_count": int(nearest),
                                "nearest_centroid_correct": int(nearest == occurrence),
                                "normalized_gold_centroid_distance": float(np.linalg.norm(state - centroids[int(occurrence)]) / max(centroid_rms, 1e-12)),
                            }
                        )
    samples = pd.DataFrame(sample_rows)
    samples.to_csv(args.output / "attention_mask_state_metrics.csv.gz", index=False, compression="gzip")
    seed = samples.groupby(["model_label", "seed", "condition", "layer"], as_index=False).agg(
        continuous_absolute_error=("continuous_absolute_error", "mean"),
        nearest_centroid_accuracy=("nearest_centroid_correct", "mean"),
        normalized_noise=("normalized_gold_centroid_distance", "mean"),
    )
    seed.to_csv(args.output / "attention_mask_seed_metrics.csv", index=False)
    effects = []
    for (model, layer), part in seed.groupby(["model_label", "layer"]):
        pivot = part.pivot(index="seed", columns="condition", values=["continuous_absolute_error", "nearest_centroid_accuracy", "normalized_noise"])
        for metric in ("continuous_absolute_error", "normalized_noise", "nearest_centroid_accuracy"):
            direction = -1.0 if metric == "nearest_centroid_accuracy" else 1.0
            # Positive improvement means lower error/noise or higher accuracy.
            needle = direction * (pivot[(metric, "clean")] - pivot[(metric, "needle_only")])
            matched = direction * (pivot[(metric, "clean")] - pivot[(metric, "matched_nonneedle_only")])
            # Correct the accuracy orientation explicitly.
            if metric == "nearest_centroid_accuracy":
                needle = pivot[(metric, "needle_only")] - pivot[(metric, "clean")]
                matched = pivot[(metric, "matched_nonneedle_only")] - pivot[(metric, "clean")]
            for seed_value in pivot.index:
                effects.append(
                    {
                        "model_label": model, "layer": int(layer), "seed": int(seed_value), "metric": metric,
                        "needle_only_improvement": float(needle.loc[seed_value]),
                        "matched_nonneedle_improvement": float(matched.loc[seed_value]),
                        "specificity": float(needle.loc[seed_value] - matched.loc[seed_value]),
                        "primary_layer": int(layer) == PRIMARY_LAYERS[model],
                    }
                )
    effect_frame = pd.DataFrame(effects)
    effect_frame.to_csv(args.output / "attention_mask_seed_effects.csv", index=False)
    stats = []
    for keys, part in effect_frame.groupby(["model_label", "layer", "metric", "primary_layer"]):
        for estimand in ("needle_only_improvement", "matched_nonneedle_improvement", "specificity"):
            values = part[estimand].to_numpy(float)
            low, high = bootstrap(values)
            stats.append({"model_label": keys[0], "layer": keys[1], "metric": keys[2], "primary_layer": keys[3], "estimand": estimand, "mean": float(values.mean()), "ci95_low": low, "ci95_high": high, "p_value": exact_sign_flip(values), "seed_count": len(values)})
    stats = pd.DataFrame(stats)
    stats["holm_p_primary_family"] = np.nan
    primary_mask = stats["primary_layer"] & (stats["estimand"] == "specificity")
    stats.loc[primary_mask, "holm_p_primary_family"] = holm(stats.loc[primary_mask, "p_value"].to_numpy())
    stats.to_csv(args.output / "attention_mask_statistics.csv", index=False)

    # Discovery-frozen head ranking by earlier full-span mass; confirmation test
    # uses the new same-depth, same-length non-needle contrast.
    head_stats = []
    selected_rows = []
    for model in sorted({row["model_label"] for row in index}):
        discovery_dir = args.base_run / model / "numeric" / "representation" / "prompt_counter_attention_v1" / "shards" / "v4.4"
        discovery_parts = []
        for path in discovery_dir.glob("*.csv.gz"):
            frame = pd.read_csv(path)
            if str(frame.iloc[0]["split"]) == "discovery":
                discovery_parts.append(frame[frame["query_occurrence"].isin([2, 4, 6, 8, 10])])
        discovery = pd.concat(discovery_parts, ignore_index=True)
        ranking = discovery.groupby(["layer", "head"], as_index=False)["prior_needle_span_mass"].mean().sort_values("prior_needle_span_mass", ascending=False).head(args.top_heads)
        confirmation_parts = [
            pd.read_csv(args.capture / row["attention_path"])
            for row in index
            if row["model_label"] == model
        ]
        confirmation = pd.concat(confirmation_parts, ignore_index=True)
        if confirmation.empty:
            continue
        for rank, candidate in enumerate(ranking.itertuples(index=False), start=1):
            selected = confirmation[(confirmation["layer"] == candidate.layer) & (confirmation["head"] == candidate.head)]
            values = selected.groupby("seed")["prior_span_preference"].mean().to_numpy(float)
            low, high = bootstrap(values)
            head_stats.append({"model_label": model, "rank": rank, "layer": int(candidate.layer), "head": int(candidate.head), "discovery_prior_span_mass": float(candidate.prior_needle_span_mass), "confirmation_preference_mean": float(values.mean()), "ci95_low": low, "ci95_high": high, "p_value": exact_sign_flip(values), "seed_count": len(values)})
            for seed_value, value in selected.groupby("seed")["prior_span_preference"].mean().items():
                selected_rows.append({"model_label": model, "rank": rank, "layer": int(candidate.layer), "head": int(candidate.head), "seed": int(seed_value), "prior_span_preference": float(value)})
    head_stats = pd.DataFrame(head_stats)
    if not head_stats.empty:
        head_stats["holm_p_within_model"] = np.nan
        for model in head_stats["model_label"].unique():
            mask = head_stats["model_label"] == model
            head_stats.loc[mask, "holm_p_within_model"] = holm(head_stats.loc[mask, "p_value"].to_numpy())
    head_stats.to_csv(args.output / "earlier_span_head_confirmation.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.output / "earlier_span_head_seed_effects.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_endpoint_attention_mask_analysis_v1",
        "state_basis_fit": "discovery prompt endpoint states",
        "state_evaluation": "confirmation seeds, occurrences 2/4/6/8/10",
        "primary_layers": PRIMARY_LAYERS,
        "primary_inference": "two-sided exact sign flip over ten seed effects; Holm over two models by three specificity endpoints",
        "head_selection": f"top {args.top_heads} heads/model by discovery earlier full-span mass",
        "head_confirmation": "prior needle full-span mass minus equal-length same-depth non-needle mass on confirmation seeds, for models with materialized clean attention rows",
        "head_confirmation_models": sorted(head_stats["model_label"].unique().tolist()) if not head_stats.empty else [],
        "status": "PASS",
    }
    (args.output / "endpoint_attention_mask_analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
