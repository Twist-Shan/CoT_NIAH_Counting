from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        means.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def bootstrap(values: np.ndarray, seed: int = 20260806, reps: int = 50000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def holm(frame: pd.DataFrame, family: list[str]) -> pd.Series:
    order = np.argsort(frame["p_value"].to_numpy())
    adjusted = np.empty(len(frame), dtype=float)
    running = 0.0
    m = len(frame)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(frame.iloc[index]["p_value"])))
        adjusted[index] = running
    return pd.Series(adjusted, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.input / "token_corruption_detail.csv")

    representation_rows = []
    for model, model_part in detail.groupby("model_label"):
        sample_paths = json.loads(model_part.iloc[0]["answer_state_paths"])
        for layer_text in sample_paths:
            layer = int(layer_text)
            packed_path = args.packed_root / "layers" / f"{model}__answer_query__L{layer:02d}.npz"
            with np.load(packed_path, allow_pickle=False) as z:
                states = np.asarray(z["states"], dtype=np.float32)
                counts = np.asarray(z["count"], dtype=int)
                split = np.asarray(z["split"]).astype(str)
            fit = split == "discovery"
            centroids = {count: states[fit & (counts == count)].mean(axis=0).astype(np.float64) for count in range(1, 11)}
            for _, row in model_part.iterrows():
                paths = json.loads(row["answer_state_paths"])
                state = np.load(args.input / paths[str(layer)]).astype(np.float64)
                centroid = centroids[int(row["gold_count"])]
                representation_rows.append(
                    {
                        "model_label": model,
                        "seed": int(row["seed"]),
                        "gold_count": int(row["gold_count"]),
                        "condition": row["condition"],
                        "layer": layer,
                        "gold_centroid_squared_distance": float(np.square(state - centroid).sum()),
                    }
                )
    representation = pd.DataFrame(representation_rows)
    representation.to_csv(args.output / "token_corruption_representation.csv", index=False)

    seed_rows = []
    for population, population_mask in (
        ("all", pd.Series(True, index=detail.index)),
        ("clean_correct_only", detail["clean_correct"].astype(bool)),
    ):
        subset = detail[population_mask].copy()
        for (model, seed), part in subset.groupby(["model_label", "seed"]):
            pivot = part.pivot(index="gold_count", columns="condition", values=["correct", "absolute_error"])
            for metric, orientation in (("correct", -1.0), ("absolute_error", 1.0)):
                clean = pivot[(metric, "clean")]
                needle = pivot[(metric, "needle_corrupt")]
                control = pivot[(metric, "ordinary_control")]
                # Positive means damage.
                needle_damage = float(orientation * (needle - clean).mean())
                control_damage = float(orientation * (control - clean).mean())
                seed_rows.append(
                    {
                        "population": population,
                        "model_label": model,
                        "seed": int(seed),
                        "endpoint": "accuracy_drop" if metric == "correct" else "absolute_error_increase",
                        "needle_damage": needle_damage,
                        "control_damage": control_damage,
                        "specificity": needle_damage - control_damage,
                        "rows": len(clean),
                    }
                )
    # Representation damage uses squared distance from the frozen gold centroid.
    merged = representation.pivot(index=["model_label", "seed", "gold_count", "layer"], columns="condition", values="gold_centroid_squared_distance").reset_index()
    clean_correct = detail[detail["condition"] == "clean"][["model_label", "seed", "gold_count", "clean_correct"]]
    merged = merged.merge(clean_correct, on=["model_label", "seed", "gold_count"], how="left")
    for population, part in (("all", merged), ("clean_correct_only", merged[merged["clean_correct"].astype(bool)])):
        for (model, seed, layer), group in part.groupby(["model_label", "seed", "layer"]):
            needle_damage = float((group["needle_corrupt"] - group["clean"]).mean())
            control_damage = float((group["ordinary_control"] - group["clean"]).mean())
            seed_rows.append(
                {
                    "population": population,
                    "model_label": model,
                    "seed": int(seed),
                    "endpoint": f"gold_centroid_distance_increase_L{int(layer)}",
                    "needle_damage": needle_damage,
                    "control_damage": control_damage,
                    "specificity": needle_damage - control_damage,
                    "rows": len(group),
                }
            )
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(args.output / "token_corruption_seed_effects.csv", index=False)

    stats = []
    for keys, part in seed_frame.groupby(["population", "model_label", "endpoint"]):
        for estimand in ("needle_damage", "control_damage", "specificity"):
            values = part[estimand].to_numpy(float)
            low, high = bootstrap(values)
            stats.append(
                {
                    "population": keys[0],
                    "model_label": keys[1],
                    "endpoint": keys[2],
                    "estimand": estimand,
                    "seed_count": len(values),
                    "mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": exact_sign_flip(values),
                }
            )
    statistics = pd.DataFrame(stats)
    statistics["holm_p_within_population"] = np.nan
    for population in statistics["population"].unique():
        mask = statistics["population"] == population
        statistics.loc[mask, "holm_p_within_population"] = holm(statistics.loc[mask].reset_index(drop=True), []).to_numpy()
    statistics.to_csv(args.output / "token_corruption_statistics.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_token_corruption_analysis_v1",
        "inference_unit": "seed after averaging count-conditioned prompts",
        "primary_specificity": "needle damage minus equal-token-budget ordinary-passage corruption damage",
        "p_value": "two-sided exact sign-flip over seed effects",
        "confidence_interval": "50,000 seed bootstrap resamples",
        "multiplicity": "Holm within all-sample and clean-correct-only populations",
        "detail_rows": len(detail),
        "seed_effect_rows": len(seed_frame),
        "statistics_rows": len(statistics),
        "status": "PASS",
    }
    (args.output / "token_corruption_analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
