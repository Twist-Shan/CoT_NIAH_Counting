from __future__ import annotations

"""Grouped predictive audit of factors associated with trace counter noise.

The analysis is explanatory/predictive, not causal.  Primary feature blocks
contain only pre-outcome trace covariates; correctness is added in a separately
labelled diagnostic block so downstream outcomes cannot be mistaken for causes.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.counter_channel import benchmark_noise_models  # noqa: E402


BASE_NUMERIC = ["running_index", "gold_count"]
POSITION_NUMERIC = [
    *BASE_NUMERIC,
    "count_progress",
    "trace_position",
    "trace_progress",
    "previous_same_site_gap",
    "token_span",
]
TRACE_NUMERIC = [
    *POSITION_NUMERIC,
    "input_tokens",
    "output_tokens",
    "sequence_length",
    "item_count",
    "duplicate_gold_city_items",
    "city_occurrences_in_item",
]
TRACE_CATEGORICAL = [
    "state_kind",
    "marker_kind",
    "termination_kind",
    "trace_order_class",
    "trace_one_to_one",
]
OUTCOME_CATEGORICAL = ["baseline_correct", "cutoff_correct"]


def _correct_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "correct"})


def load_sampled(
    paths: list[Path], *, split: str, fraction: float, max_rows: int, correct_only: bool
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    rows = 0
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
            chunk = chunk.loc[chunk["split"].astype(str) == split].copy()
            if correct_only:
                chunk = chunk.loc[_correct_mask(chunk["baseline_correct"])].copy()
            if fraction < 1.0 and len(chunk):
                hashed = pd.util.hash_pandas_object(
                    chunk[["request_id", "site", "running_index"]], index=False
                ).to_numpy(dtype=np.uint64)
                threshold = int(fraction * 10_000)
                chunk = chunk.loc[(hashed % 10_000) < threshold]
            if len(chunk):
                remaining = max_rows - rows
                frames.append(chunk.iloc[:remaining])
                rows += min(len(chunk), remaining)
            if rows >= max_rows:
                break
        if rows >= max_rows:
            break
    if not frames:
        raise RuntimeError("no scalar noise rows matched the requested split/sample")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--correct-only", action="store_true")
    parser.add_argument("--sample-fraction", type=float, default=0.25)
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["elastic_net", "hist_gradient_boosting"],
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["noise_total_rms", "noise_orthogonal_rms", "count_axis_deviation_abs"],
    )
    args = parser.parse_args()
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("sample fraction must be in (0,1]")

    started = time.perf_counter()
    frame = load_sampled(
        [path.resolve() for path in args.inputs],
        split=args.split,
        fraction=args.sample_fraction,
        max_rows=args.max_rows,
        correct_only=args.correct_only,
    )
    frame["count_axis_deviation_abs"] = frame["count_axis_deviation"].abs()
    # Seed is the grouping unit: all counts and stochastic realizations from the
    # same underlying prompt family remain on one side of a fold.
    frame["seed_group"] = frame["model_label"].astype(str) + "/" + frame["seed"].astype(str)

    feature_sets = {
        "count_only": (BASE_NUMERIC, []),
        "plus_trace_position": (POSITION_NUMERIC, []),
        "plus_trace_form": (TRACE_NUMERIC, TRACE_CATEGORICAL),
        "plus_outcome_diagnostic": (
            TRACE_NUMERIC,
            [*TRACE_CATEGORICAL, *OUTCOME_CATEGORICAL],
        ),
    }
    scopes: list[tuple[str, pd.DataFrame]] = []
    for model, model_frame in frame.groupby("model_label", sort=True):
        pooled = model_frame.copy()
        pooled["site_scope"] = pooled["site"].astype(str)
        scopes.append((f"{model}/all_sites", pooled))
        for site in ("city_end", "item_end", "marker_end"):
            site_frame = model_frame.loc[model_frame["site"] == site].copy()
            if len(site_frame):
                scopes.append((f"{model}/{site}", site_frame))

    results: list[pd.DataFrame] = []
    for scope, scoped in scopes:
        pooled = scope.endswith("/all_sites")
        for target in args.targets:
            for feature_set, (numeric, categorical) in feature_sets.items():
                categorical = [*categorical, *(["site_scope"] if pooled else [])]
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
                result.insert(0, "scope", scope)
                result.insert(1, "feature_set", feature_set)
                result["post_outcome_diagnostic"] = feature_set == "plus_outcome_diagnostic"
                results.append(result)
                print(f"[noise factors] {scope} {target} {feature_set}", flush=True)
    metrics = pd.concat(results, ignore_index=True)
    ordering = {name: index for index, name in enumerate(feature_sets)}
    metrics["feature_order"] = metrics["feature_set"].map(ordering)
    metrics = metrics.sort_values(["scope", "target", "model", "feature_order"])
    metrics["incremental_r2"] = metrics.groupby(
        ["scope", "target", "model"], sort=False
    )["heldout_r2_log1p"].diff()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "trace_noise_factor_models.csv", index=False)
    conditional = (
        frame.groupby(
            ["model_label", "site", "running_index", "marker_kind", "trace_order_class"],
            dropna=False,
        )[["noise_total_rms", "noise_orthogonal_rms", "count_axis_deviation_abs"]]
        .agg(["count", "mean", "std", "median"])
    )
    conditional.columns = ["__".join(column) for column in conditional.columns]
    conditional.reset_index().to_csv(output / "trace_noise_conditional_variance.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_native_noise_factors_v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "split": args.split,
        "correct_only": args.correct_only,
        "sample_fraction": args.sample_fraction,
        "max_rows": args.max_rows,
        "analysis_rows": len(frame),
        "seed_groups": int(frame["seed_group"].nunique()),
        "targets": args.targets,
        "algorithms": args.algorithms,
        "feature_sets": feature_sets,
        "post_outcome_features_are_diagnostic_only": True,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    (output / "noise_factor_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
