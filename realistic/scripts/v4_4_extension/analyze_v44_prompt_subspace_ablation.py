from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pvalue(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = [abs(float(np.mean(values * np.asarray(signs)))) for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260806)
    means = values[rng.integers(0, len(values), size=(50000, len(values)))].mean(axis=1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values); result = np.empty(len(values)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[index]))); result[index] = running
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--packed-root",
        type=Path,
        default=Path("runs/v4_4_counter_channel_20260806/packed"),
    )
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.input / "prompt_subspace_ablation_detail.csv")
    seed_rows = []
    for population, subset in (("all", detail), ("clean_correct_only", detail[detail["clean_correct"].astype(bool)])):
        for (model, seed), part in subset.groupby(["model_label", "seed"]):
            pivot = part.pivot(index="gold_count", columns="condition", values=["correct", "absolute_error"])
            for condition in ("actual_rank3_remove", "centroid_curve_remove", "normmatched_orthogonal"):
                accuracy_drop = float((pivot[("correct", "clean")] - pivot[("correct", condition)]).mean())
                error_increase = float((pivot[("absolute_error", condition)] - pivot[("absolute_error", "clean")]).mean())
                seed_rows.extend([
                    {"population": population, "model_label": model, "seed": int(seed), "condition": condition, "endpoint": "accuracy_drop", "effect": accuracy_drop, "rows": len(pivot)},
                    {"population": population, "model_label": model, "seed": int(seed), "condition": condition, "endpoint": "absolute_error_increase", "effect": error_increase, "rows": len(pivot)},
                ])
    seed = pd.DataFrame(seed_rows)
    # Specificity of both count removals relative to the norm-matched orthogonal control.
    specific = []
    for keys, part in seed.groupby(["population", "model_label", "seed", "endpoint"]):
        values = dict(zip(part["condition"], part["effect"]))
        for condition in ("actual_rank3_remove", "centroid_curve_remove"):
            specific.append({"population": keys[0], "model_label": keys[1], "seed": keys[2], "condition": condition, "endpoint": keys[3], "effect": values[condition] - values["normmatched_orthogonal"], "rows": int(part["rows"].max())})
    specific = pd.DataFrame(specific)
    representation_rows = []
    if "answer_state_paths" in detail.columns:
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
                centroids = {
                    count: states[fit & (counts == count)].mean(axis=0).astype(np.float64)
                    for count in range(1, 11)
                }
                for _, row in model_part.iterrows():
                    state_paths = json.loads(row["answer_state_paths"])
                    state = np.load(args.input / state_paths[str(layer)]).astype(np.float64)
                    representation_rows.append(
                        {
                            "model_label": model,
                            "seed": int(row["seed"]),
                            "gold_count": int(row["gold_count"]),
                            "condition": row["condition"],
                            "layer": layer,
                            "gold_centroid_squared_distance": float(
                                np.square(state - centroids[int(row["gold_count"])]).sum()
                            ),
                        }
                    )
        representation = pd.DataFrame(representation_rows)
        representation.to_csv(
            args.output / "subspace_ablation_answer_representation.csv", index=False
        )
        wide = representation.pivot(
            index=["model_label", "seed", "gold_count", "layer"],
            columns="condition",
            values="gold_centroid_squared_distance",
        ).reset_index()
        clean_labels = (
            detail[detail["condition"] == "clean"]
            [["model_label", "seed", "gold_count", "clean_correct"]]
            .drop_duplicates()
        )
        wide = wide.merge(
            clean_labels, on=["model_label", "seed", "gold_count"], how="left"
        )
        representation_specific = []
        for population, subset in (
            ("all", wide),
            ("clean_correct_only", wide[wide["clean_correct"].astype(bool)]),
        ):
            for (model, seed_value, layer), part in subset.groupby(
                ["model_label", "seed", "layer"]
            ):
                control_damage = float(
                    (part["normmatched_orthogonal"] - part["clean"]).mean()
                )
                for condition in ("actual_rank3_remove", "centroid_curve_remove"):
                    damage = float((part[condition] - part["clean"]).mean())
                    representation_specific.append(
                        {
                            "population": population,
                            "model_label": model,
                            "seed": int(seed_value),
                            "condition": condition,
                            "endpoint": f"gold_centroid_distance_increase_L{int(layer)}",
                            "effect": damage - control_damage,
                            "rows": len(part),
                        }
                    )
        if representation_specific:
            specific = pd.concat(
                [specific, pd.DataFrame(representation_specific)], ignore_index=True
            )
    seed.to_csv(args.output / "subspace_ablation_seed_effects.csv", index=False)
    specific.to_csv(args.output / "subspace_ablation_specificity_seed_effects.csv", index=False)
    stats = []
    for keys, part in specific.groupby(["population", "model_label", "condition", "endpoint"]):
        values = part["effect"].to_numpy(float); low, high = ci(values)
        stats.append({"population": keys[0], "model_label": keys[1], "condition": keys[2], "endpoint": keys[3], "mean": float(values.mean()), "ci95_low": low, "ci95_high": high, "p_value": pvalue(values), "seed_count": len(values), "sample_rows": int(part["rows"].sum())})
    stats = pd.DataFrame(stats); stats["holm_p_within_population"] = np.nan
    for population in stats["population"].unique():
        mask = stats["population"] == population
        stats.loc[mask, "holm_p_within_population"] = holm(stats.loc[mask, "p_value"].to_numpy())
    stats.to_csv(args.output / "subspace_ablation_statistics.csv", index=False)
    (args.output / "analysis_audit.json").write_text(json.dumps({
        "schema_version": "realistic_niah_v4_4_prompt_subspace_ablation_analysis_v1",
        "inference_unit": "confirmation seed after averaging counts 2-10",
        "specificity": "count-subspace damage minus equal-norm orthogonal-subspace damage",
        "answer_representation": "squared distance to discovery-frozen gold-count centroid at two downstream answer-query layers",
        "p_value": "two-sided exact sign-flip", "ci": "50,000 seed bootstrap",
        "multiplicity": "Holm within all and clean-correct-only populations",
        "status": "PASS",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
