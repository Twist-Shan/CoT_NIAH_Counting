from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METRICS = (
    "needle_raw_rms_change",
    "ordinary_raw_rms_change",
    "raw_specificity",
    "needle_relative_rms_change",
    "ordinary_relative_rms_change",
    "relative_specificity",
    "needle_mean_cosine_distance",
    "ordinary_mean_cosine_distance",
    "cosine_specificity",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def exact_sign_flip(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    null = np.asarray(
        [float(np.mean(array * np.asarray(signs))) for signs in itertools.product((-1, 1), repeat=len(array))],
        dtype=np.float64,
    )
    return float(np.mean(np.abs(null) >= observed - 1e-15))


def bootstrap_ci(
    values: Sequence[float], *, draws: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(draws), len(array)))
    samples = array[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    values: Sequence[float], *, draws: int, seed: int
) -> dict[str, float]:
    low, high = bootstrap_ci(values, draws=draws, seed=seed)
    return {
        "mean": float(np.mean(values)),
        "ci95_low": low,
        "ci95_high": high,
        "exact_sign_flip_p": exact_sign_flip(values),
        "seed_units": int(len(values)),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze canonical full-vector span-state deformation curves."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_span_state_deformation.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
    draws = int(config["bootstrap_draws"])
    bootstrap_seed = int(config["bootstrap_seed"])
    seeds = tuple(int(value) for value in config["confirmation_seeds"])
    counts = tuple(int(value) for value in config["counts"])
    model_order = ("Qwen3-8B", "Gemma4-E4B")
    layerwise_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_4_5_span_state_deformation_analysis_v1",
        "models": {},
    }
    audit: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_4_5_span_state_deformation_analysis_audit_v1",
        "status": "PASS",
        "models": {},
    }

    for model_index, model in enumerate(model_order):
        run_audit_path = root / model / "complete.json"
        run_audit = json.loads(run_audit_path.read_text(encoding="utf-8"))
        details = read_jsonl(root / model / "detail.jsonl")
        layers = tuple(int(value) for value in config["layers"][model])
        expected_keys = {
            (int(seed), int(count), int(layer))
            for seed in seeds
            for count in counts
            for layer in layers
        }
        keys = [
            (int(row["seed"]), int(row["gold_count"]), int(row["layer"]))
            for row in details
        ]
        finite = all(
            math.isfinite(float(row[metric])) for row in details for metric in METRICS
        )
        model_audit = {
            "run_audit_status": run_audit.get("status"),
            "rows": len(details),
            "expected_rows": len(expected_keys),
            "unique_keys": len(set(keys)),
            "exact_key_coverage": set(keys) == expected_keys,
            "finite_metrics": finite,
        }
        if not (
            model_audit["run_audit_status"] == "PASS"
            and model_audit["rows"] == model_audit["expected_rows"]
            and model_audit["unique_keys"] == model_audit["expected_rows"]
            and model_audit["exact_key_coverage"]
            and model_audit["finite_metrics"]
        ):
            audit["status"] = "FAIL"
        audit["models"][model] = model_audit

        by_layer_seed: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in details:
            key = (int(row["layer"]), int(row["seed"]))
            for metric in METRICS:
                by_layer_seed[key][metric].append(float(row[metric]))
        seed_layer_means: dict[tuple[int, int, str], float] = {}
        for (layer, seed), metric_values in by_layer_seed.items():
            for metric, values in metric_values.items():
                if len(values) != len(counts):
                    raise RuntimeError(
                        f"{model} L{layer} seed {seed} has {len(values)} {metric} rows"
                    )
                seed_layer_means[(layer, seed, metric)] = float(np.mean(values))

        for layer in layers:
            result: dict[str, Any] = {"model_label": model, "layer": int(layer)}
            for metric_index, metric in enumerate(METRICS):
                values = [seed_layer_means[(layer, seed, metric)] for seed in seeds]
                stats = summarize(
                    values,
                    draws=draws,
                    seed=bootstrap_seed + model_index * 10000 + layer * 100 + metric_index,
                )
                for name, value in stats.items():
                    result[f"{metric}_{name}"] = value
            layerwise_rows.append(result)

        start, end = (
            int(value) for value in config["independently_frozen_reusable_windows"][model]
        )
        window_layers = tuple(layer for layer in layers if start <= layer <= end)
        model_summary: dict[str, Any] = {
            "reusable_window": [start, end],
            "reusable_window_layers": len(window_layers),
            "pointwise_nominal_positive_layers": [],
        }
        for metric_index, metric in enumerate(
            ("raw_specificity", "relative_specificity", "cosine_specificity")
        ):
            values = [
                float(
                    np.mean(
                        [seed_layer_means[(layer, seed, metric)] for layer in window_layers]
                    )
                )
                for seed in seeds
            ]
            model_summary[f"window_{metric}"] = summarize(
                values,
                draws=draws,
                seed=bootstrap_seed + model_index * 10000 + 9000 + metric_index,
            )
        for row in layerwise_rows:
            if row["model_label"] != model:
                continue
            if (
                float(row["raw_specificity_mean"]) > 0
                and float(row["raw_specificity_exact_sign_flip_p"]) < 0.05
            ):
                model_summary["pointwise_nominal_positive_layers"].append(int(row["layer"]))
        summary["models"][model] = model_summary

    if audit["status"] != "PASS":
        (output / "analysis_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError("Span-state deformation analysis audit failed")
    write_csv(output / "layerwise_state_deformation.csv", layerwise_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit["layerwise_rows"] = len(layerwise_rows)
    audit["expected_layerwise_rows"] = sum(
        len(config["layers"][model]) for model in model_order
    )
    if audit["layerwise_rows"] != audit["expected_layerwise_rows"]:
        audit["status"] = "FAIL"
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise RuntimeError("Span-state deformation analysis output audit failed")


if __name__ == "__main__":
    main()
