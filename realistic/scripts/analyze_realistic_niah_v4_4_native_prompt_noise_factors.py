from __future__ import annotations

"""Grouped factor attribution for prompt running-counter noise."""

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.counter_channel import benchmark_noise_models  # noqa: E402


def correct_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "correct"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--correct-only", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["elastic_net", "random_forest", "extra_trees", "hist_gradient_boosting"],
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["noise_total_rms", "noise_orthogonal_rms", "count_axis_deviation_abs"],
    )
    args = parser.parse_args()
    started = time.perf_counter()
    frames = [pd.read_csv(path.resolve(), low_memory=False) for path in args.inputs]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["split"].astype(str) == args.split].copy()
    if args.correct_only:
        frame = frame.loc[correct_mask(frame["baseline_correct"])].copy()
    if frame.empty:
        raise RuntimeError(f"no prompt noise rows for split={args.split}")
    frame["count_axis_deviation_abs"] = frame["count_axis_deviation"].abs()
    frame["seed_group"] = frame["model_label"].astype(str) + "/" + frame["seed"].astype(str)

    feature_sets = {
        "count_only": (["running_index"], []),
        "plus_endpoint_position": (
            [
                "running_index",
                "count_progress",
                "token_start",
                "token_end",
                "token_span",
                "prompt_progress",
                "previous_endpoint_gap",
            ],
            [],
        ),
        "plus_content_context": (
            [
                "running_index",
                "count_progress",
                "token_start",
                "token_end",
                "token_span",
                "prompt_progress",
                "previous_endpoint_gap",
                "input_tokens",
                "score",
                "realization_seed",
                "content_seed",
            ],
            ["city", "haystack_source_mode", "haystack_source_files"],
        ),
        "plus_outcome_diagnostic": (
            [
                "running_index",
                "count_progress",
                "token_start",
                "token_end",
                "token_span",
                "prompt_progress",
                "previous_endpoint_gap",
                "input_tokens",
                "score",
                "realization_seed",
                "content_seed",
                "absolute_deviation",
            ],
            [
                "city",
                "haystack_source_mode",
                "haystack_source_files",
                "baseline_correct",
                "predicted_count",
            ],
        ),
    }
    results = []
    for model, scoped in frame.groupby("model_label", sort=True):
        for target in args.targets:
            for feature_set, (numeric, categorical) in feature_sets.items():
                result = benchmark_noise_models(
                    scoped,
                    target=target,
                    group_column="seed_group",
                    numeric=numeric,
                    categorical=categorical,
                    folds=args.folds,
                    n_jobs=args.n_jobs,
                    algorithms=args.algorithms,
                )
                result.insert(0, "model_label", model)
                result.insert(1, "feature_set", feature_set)
                result["post_outcome_diagnostic"] = feature_set == "plus_outcome_diagnostic"
                results.append(result)
                print(f"[prompt noise factors] {model} {target} {feature_set}", flush=True)
    metrics = pd.concat(results, ignore_index=True)
    ordering = {name: index for index, name in enumerate(feature_sets)}
    metrics["feature_order"] = metrics["feature_set"].map(ordering)
    metrics = metrics.sort_values(["model_label", "target", "model", "feature_order"])
    metrics["incremental_r2"] = metrics.groupby(
        ["model_label", "target", "model"], sort=False
    )["heldout_r2_log1p"].diff()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "prompt_noise_factor_models.csv", index=False)
    conditional = (
        frame.groupby(["model_label", "running_index", "city"], dropna=False)[
            ["noise_total_rms", "noise_orthogonal_rms", "count_axis_deviation_abs"]
        ]
        .agg(["count", "mean", "std", "median"])
    )
    conditional.columns = ["__".join(column) for column in conditional.columns]
    conditional.reset_index().to_csv(output / "prompt_noise_conditional_variance.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_native_prompt_noise_factors_v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "split": args.split,
        "correct_only": args.correct_only,
        "rows": len(frame),
        "seed_groups": int(frame["seed_group"].nunique()),
        "algorithms": args.algorithms,
        "targets": args.targets,
        "post_outcome_features_are_diagnostic_only": True,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    (output / "prompt_noise_factor_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
