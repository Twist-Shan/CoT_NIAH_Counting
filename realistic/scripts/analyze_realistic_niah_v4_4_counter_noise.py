from __future__ import annotations

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

from realistic_niah_v4.counter_channel import (  # noqa: E402
    benchmark_noise_models,
    leave_group_out_noise,
    load_layer_dataset,
    read_layer_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain prompt/trace running-index conditional variance"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--factor-config", type=Path, required=True)
    parser.add_argument("--covariates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=["prompt_running", "trace_running"])
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    factor_config = json.loads(args.factor_config.read_text(encoding="utf-8"))
    covariates = (
        pd.read_csv(args.covariates)
        if args.covariates is not None
        else None
    )
    merge_keys = list(factor_config.get("merge_keys", ["sample_id", "layer"]))
    numeric = list(factor_config.get("numeric", []))
    categorical = list(factor_config.get("categorical", []))
    targets = list(
        factor_config.get(
            "targets", ["noise_total", "noise_parallel", "noise_orthogonal"]
        )
    )
    feature_groups = dict(factor_config.get("feature_groups", {}))

    datasets = [
        load_layer_dataset(manifest, row)
        for row in read_layer_manifest(manifest)
        if str(row["role"]) in set(args.roles)
        and (args.layers is None or int(row["layer"]) in set(args.layers))
    ]
    if not datasets:
        raise RuntimeError(f"No datasets matched roles={args.roles}")
    noise_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        frame = leave_group_out_noise(dataset, rank=args.rank)
        if covariates is not None:
            missing_keys = set(merge_keys) - set(frame.columns) | set(merge_keys) - set(covariates.columns)
            if missing_keys:
                raise ValueError(f"noise/covariate merge is missing keys {sorted(missing_keys)}")
            before = len(frame)
            frame = frame.merge(covariates, on=merge_keys, how="left", validate="many_to_one")
            if len(frame) != before:
                raise RuntimeError("covariate merge changed noise row count")
        noise_frames.append(frame)
        print(f"[noise] {dataset.model_label} {dataset.role} L{dataset.layer}", flush=True)
    noise = pd.concat(noise_frames, ignore_index=True)
    noise.to_csv(output / "counter_noise_rows.csv.gz", index=False, compression="gzip")

    model_frames: list[pd.DataFrame] = []
    factor_sets: list[tuple[str, list[str], list[str]]] = [("all", numeric, categorical)]
    for group_name, columns in feature_groups.items():
        columns = list(columns)
        factor_sets.append(
            (
                f"without_{group_name}",
                [column for column in numeric if column not in columns],
                [column for column in categorical if column not in columns],
            )
        )
    scopes: list[tuple[str, str, str, pd.DataFrame]] = [
        ("pooled", "all", "all", noise)
    ]
    for (model_label, role), scoped in noise.groupby(
        ["model_label", "role"], sort=True, dropna=False
    ):
        scopes.append(("model_role", str(model_label), str(role), scoped))
    for scope, scope_model, scope_role, scoped_noise in scopes:
        for target in targets:
            for factor_set, current_numeric, current_categorical in factor_sets:
                result = benchmark_noise_models(
                    scoped_noise,
                    target=target,
                    group_column=str(factor_config.get("group_column", "seed")),
                    numeric=current_numeric,
                    categorical=current_categorical,
                    folds=args.folds,
                    n_jobs=args.n_jobs,
                )
                result.insert(0, "factor_set", factor_set)
                result.insert(0, "scope_role", scope_role)
                result.insert(0, "scope_model_label", scope_model)
                result.insert(0, "analysis_scope", scope)
                model_frames.append(result)
    model_results = pd.concat(model_frames, ignore_index=True)
    model_results.to_csv(output / "counter_noise_model_comparison.csv", index=False)

    importance_index = [
        "analysis_scope",
        "scope_model_label",
        "scope_role",
        "target",
        "model",
    ]
    full = model_results[model_results["factor_set"] == "all"].set_index(
        importance_index
    )
    importance_rows: list[dict[str, object]] = []
    for row in model_results.itertuples(index=False):
        if row.factor_set == "all":
            continue
        key = (
            row.analysis_scope,
            row.scope_model_label,
            row.scope_role,
            row.target,
            row.model,
        )
        if key not in full.index:
            continue
        full_row = full.loc[key]
        importance_rows.append(
            {
                "analysis_scope": row.analysis_scope,
                "scope_model_label": row.scope_model_label,
                "scope_role": row.scope_role,
                "target": row.target,
                "model": row.model,
                "feature_group": str(row.factor_set).removeprefix("without_"),
                "full_r2": float(full_row["heldout_r2_log1p"]),
                "without_group_r2": float(row.heldout_r2_log1p),
                "delta_r2_removed_group": float(
                    full_row["heldout_r2_log1p"] - row.heldout_r2_log1p
                ),
            }
        )
    pd.DataFrame(importance_rows).to_csv(
        output / "counter_noise_feature_group_importance.csv", index=False
    )

    conditional = (
        noise.groupby(["model_label", "role", "layer", "count"], dropna=False)[targets]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    conditional.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in conditional.columns
    ]
    conditional.to_csv(output / "counter_noise_conditional_variance.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_counter_noise_analysis_v1",
        "manifest": str(manifest),
        "factor_config": str(args.factor_config.resolve()),
        "covariates": None if args.covariates is None else str(args.covariates.resolve()),
        "roles": args.roles,
        "layers": args.layers,
        "rank": args.rank,
        "datasets": len(datasets),
        "noise_rows": len(noise),
        "targets": targets,
        "numeric_factors": numeric,
        "categorical_factors": categorical,
        "feature_groups": feature_groups,
        "split_unit": str(factor_config.get("group_column", "seed")),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    (output / "counter_noise_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
