from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .report_followups import (
    centroid_mean_pca,
    load_rankings,
    load_run,
    run_position_local_ablation_suite,
    run_strict_ablation_suite,
)
from .training import load_final_model


COUNT_BINS = ("1-10", "11-20", "21-30")


def count_bin_from_value(value: int | float) -> str:
    count = int(value)
    if count <= 10:
        return "1-10"
    if count <= 20:
        return "11-20"
    return "21-30"


def _with_count_bin(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result["count_bin"] = result[column].map(count_bin_from_value)
    return result


def _with_all(frame: pd.DataFrame, group_columns: Iterable[str]) -> pd.DataFrame:
    """Return bin-level rows plus an independently recomputed all-count row."""

    groups = list(group_columns)
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column not in groups]
    all_counts = frame.groupby(groups, as_index=False)[numeric].mean()
    all_counts["count_bin"] = "all"
    return pd.concat([frame, all_counts], ignore_index=True, sort=False)


def _linear_summary(
    frame: pd.DataFrame,
    groups: Iterable[str],
    *,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = list(groups)
    for keys, part in frame.groupby(group_columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        x = part[x_column].to_numpy(dtype=float)
        y = part[y_column].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) < 2 or np.allclose(x, x[0]):
            slope = intercept = r2 = math.nan
        else:
            design = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            prediction = design @ beta
            denominator = float(((y - y.mean()) ** 2).sum())
            slope = float(beta[1])
            intercept = float(beta[0])
            r2 = (
                1.0 - float(((y - prediction) ** 2).sum()) / denominator
                if denominator > 1e-12
                else math.nan
            )
        row = {
            **dict(zip(group_columns, key_values)),
            "n_rows": int(len(part)),
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
        }
        for metric in (
            "follows_donor",
            "follows_receiver",
            "accuracy",
            "causal_expected_shift",
            "normalized_recovery",
        ):
            if metric in part.columns:
                row[f"mean_{metric}"] = float(part[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_nested_patching(source: Path) -> dict[str, pd.DataFrame]:
    detail = pd.read_csv(source)
    detail = _with_count_bin(detail, "receiver_count")
    summary_groups = ["mode", "family", "replicate", "count_bin", "donor_offset", "top_n"]
    summary = detail.groupby(summary_groups, as_index=False).mean(numeric_only=True)
    all_summary = detail.groupby(
        ["mode", "family", "replicate", "donor_offset", "top_n"], as_index=False
    ).mean(numeric_only=True)
    all_summary["count_bin"] = "all"
    summary = pd.concat([summary, all_summary], ignore_index=True, sort=False)

    regression_groups = ["mode", "family", "replicate", "count_bin", "top_n"]
    regression = _linear_summary(
        detail,
        regression_groups,
        x_column="donor_offset",
        y_column="causal_expected_shift",
    )
    all_detail = detail.copy()
    all_detail["count_bin"] = "all"
    regression = pd.concat(
        [
            regression,
            _linear_summary(
                all_detail,
                regression_groups,
                x_column="donor_offset",
                y_column="causal_expected_shift",
            ),
        ],
        ignore_index=True,
    )
    return {
        "nested_head_patching_by_bin": summary,
        "nested_head_patching_regression_by_bin": regression,
    }


def _aggregate_retrieval_patching(source: Path) -> dict[str, pd.DataFrame]:
    detail = pd.read_csv(source)
    detail = _with_count_bin(detail, "count")
    groups = ["family", "replicate", "query_role", "count_bin", "top_n"]
    summary = detail.groupby(groups, as_index=False).mean(numeric_only=True)
    all_detail = detail.copy()
    all_detail["count_bin"] = "all"
    summary = pd.concat(
        [summary, all_detail.groupby(groups, as_index=False).mean(numeric_only=True)],
        ignore_index=True,
    )
    return {"retrieval_control_patching_by_bin": summary}


def _aggregate_steering(source: Path) -> dict[str, pd.DataFrame]:
    detail = pd.read_csv(source)
    detail = _with_count_bin(detail, "count")
    groups = ["site", "mode", "count_bin", "layer", "alpha"]
    summary = detail.groupby(groups, as_index=False).agg(
        n_examples=("example_idx", "size"),
        mean_baseline_expected=("baseline_expected", "mean"),
        mean_steered_expected=("steered_expected", "mean"),
        mean_causal_expected_shift=("causal_expected_shift", "mean"),
        std_causal_expected_shift=("causal_expected_shift", "std"),
        accuracy=("accuracy", "mean"),
    )
    all_detail = detail.copy()
    all_detail["count_bin"] = "all"
    summary = pd.concat(
        [
            summary,
            all_detail.groupby(groups, as_index=False).agg(
                n_examples=("example_idx", "size"),
                mean_baseline_expected=("baseline_expected", "mean"),
                mean_steered_expected=("steered_expected", "mean"),
                mean_causal_expected_shift=("causal_expected_shift", "mean"),
                std_causal_expected_shift=("causal_expected_shift", "std"),
                accuracy=("accuracy", "mean"),
            ),
        ],
        ignore_index=True,
    )
    gain = _linear_summary(
        detail,
        ["site", "mode", "count_bin", "layer"],
        x_column="alpha",
        y_column="causal_expected_shift",
    )
    all_detail["count_bin"] = "all"
    gain = pd.concat(
        [
            gain,
            _linear_summary(
                all_detail,
                ["site", "mode", "count_bin", "layer"],
                x_column="alpha",
                y_column="causal_expected_shift",
            ),
        ],
        ignore_index=True,
    )
    return {
        "geometry_steering_by_bin": summary,
        "geometry_steering_gain_by_bin": gain,
    }


def _aggregate_state_transplant(source: Path, prefix: str) -> dict[str, pd.DataFrame]:
    detail = pd.read_csv(source)
    detail = _with_count_bin(detail, "receiver_count")
    summary = detail.groupby(
        ["site", "mode", "count_bin", "donor_offset", "layer"], as_index=False
    ).mean(numeric_only=True)
    all_detail = detail.copy()
    all_detail["count_bin"] = "all"
    summary = pd.concat(
        [
            summary,
            all_detail.groupby(
                ["site", "mode", "count_bin", "donor_offset", "layer"], as_index=False
            ).mean(numeric_only=True),
        ],
        ignore_index=True,
    )
    regression = _linear_summary(
        detail,
        ["site", "mode", "count_bin", "layer"],
        x_column="donor_offset",
        y_column="causal_expected_shift",
    )
    regression = pd.concat(
        [
            regression,
            _linear_summary(
                all_detail,
                ["site", "mode", "count_bin", "layer"],
                x_column="donor_offset",
                y_column="causal_expected_shift",
            ),
        ],
        ignore_index=True,
    )
    return {
        f"{prefix}_by_bin": summary,
        f"{prefix}_regression_by_bin": regression,
    }


def _aggregate_trace_progress(source: Path) -> dict[str, pd.DataFrame]:
    detail = pd.read_csv(source)
    detail = _with_count_bin(detail, "gold_count")
    summary = detail.groupby(
        ["count_bin", "donor_offset", "layer"], as_index=False
    ).mean(numeric_only=True)
    all_detail = detail.copy()
    all_detail["count_bin"] = "all"
    summary = pd.concat(
        [
            summary,
            all_detail.groupby(
                ["count_bin", "donor_offset", "layer"], as_index=False
            ).mean(numeric_only=True),
        ],
        ignore_index=True,
    )
    return {"trace_progress_transplant_by_bin": summary}


def build_stratified_tables(
    run_dir: str | Path,
    *,
    device: str | None = None,
    rerun_ablation: bool = True,
    rerun_position_local_ablation: bool = True,
    ablation_examples_per_count: int = 8,
    random_replicates: int = 8,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis" / "report_stratified"
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    followup = run_dir / "analysis" / "report_followups" / "tables"
    state = run_dir / "analysis" / "state_causal" / "tables"
    outputs: dict[str, pd.DataFrame] = {}

    cfg = vocab = rankings = models = None
    if rerun_ablation or rerun_position_local_ablation:
        cfg, vocab = load_run(run_dir, device=device)
        rankings = load_rankings(run_dir)
        models = {mode: load_final_model(cfg, vocab, run_dir, mode) for mode in cfg.modes}

    if rerun_ablation:
        assert cfg is not None and vocab is not None and rankings is not None and models is not None
        single, cumulative = run_strict_ablation_suite(
            models,
            cfg,
            vocab,
            rankings,
            examples_per_count=int(ablation_examples_per_count),
            random_replicates=int(random_replicates),
        )
        outputs["single_head_ablation_by_bin"] = single
        outputs["cumulative_head_ablation_by_bin"] = cumulative
    else:
        outputs["single_head_ablation_by_bin"] = pd.read_csv(
            followup / "single_head_ablation.csv"
        )
        outputs["cumulative_head_ablation_by_bin"] = pd.read_csv(
            followup / "strict_cumulative_ablation.csv"
        )

    if rerun_position_local_ablation:
        assert cfg is not None and vocab is not None and rankings is not None and models is not None
        outputs["position_local_ablation_by_bin"] = run_position_local_ablation_suite(
            models,
            cfg,
            vocab,
            rankings,
            examples_per_count=int(ablation_examples_per_count),
            random_replicates=int(random_replicates),
        )
    else:
        outputs["position_local_ablation_by_bin"] = pd.read_csv(
            table_dir / "position_local_ablation_by_bin.csv"
        )

    if models is not None:
        for model in models.values():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    outputs.update(_aggregate_nested_patching(followup / "nested_head_patching.csv"))
    outputs.update(_aggregate_retrieval_patching(followup / "retrieval_control_patching.csv"))
    outputs.update(_aggregate_steering(state / "steering_rows.csv"))
    outputs.update(
        _aggregate_state_transplant(
            state / "final_state_transplant_rows.csv", "final_state_transplant"
        )
    )
    outputs.update(
        _aggregate_state_transplant(
            followup / "centroid_transplant.csv", "centroid_transplant"
        )
    )
    outputs.update(_aggregate_trace_progress(state / "trace_progress_transplant_rows.csv"))
    coordinates, geometry = centroid_mean_pca(run_dir)
    outputs["centroid_mean_pca_coordinates"] = coordinates
    outputs["centroid_mean_geometry"] = geometry

    for name, frame in outputs.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)

    manifest = {
        "run_dir": str(run_dir.resolve()),
        "count_bins": {
            "1-10": "gold or receiver count in [1, 10]",
            "11-20": "gold or receiver count in [11, 20]",
            "21-30": "gold or receiver count in [21, 30]",
        },
        "fresh_forward_passes": {
            "single_head_ablation_by_bin": bool(rerun_ablation),
            "cumulative_head_ablation_by_bin": bool(rerun_ablation),
            "position_local_ablation_by_bin": bool(rerun_position_local_ablation),
        },
        "ablation_examples_per_exact_count": int(ablation_examples_per_count),
        "ablation_random_orders": int(random_replicates),
        "position_local_ablation": {
            "direct_broad": "zero selected pre-c_proj head slices only at the non-thinking <Ans> query",
            "targeted_retrieval": "zero selected pre-c_proj head slices only at every CoT trace <k> query",
            "trace_readout": "zero selected pre-c_proj head slices only at the CoT <Ans> query",
            "layer_matched_controls": (
                "each cumulative control preserves the ranked-top layer sequence while replacing "
                "the selected head within that layer"
            ),
        },
        "reaggregated_from_existing_per_example_interventions": {
            "nested_head_patching": "analysis/report_followups/tables/nested_head_patching.csv",
            "retrieval_control_patching": "analysis/report_followups/tables/retrieval_control_patching.csv",
            "geometry_steering": "analysis/state_causal/tables/steering_rows.csv",
            "final_state_transplant": "analysis/state_causal/tables/final_state_transplant_rows.csv",
            "centroid_transplant": "analysis/report_followups/tables/centroid_transplant.csv",
            "trace_progress_transplant": "analysis/state_causal/tables/trace_progress_transplant_rows.csv",
        },
        "aggregation_rule": (
            "Every bin statistic is recomputed from rows whose gold/receiver count falls in that bin. "
            "The all-count row is recomputed from all raw rows, not averaged from the three bin means."
        ),
        "pca_rule": (
            "For each semantic site and layer, hidden states are averaged by exact count first. "
            "PCA is then fitted to the 30 count centroids; PC variance ratios therefore describe "
            "between-count centroid geometry rather than within-count sample variance."
        ),
        "row_counts": {name: int(len(frame)) for name, frame in outputs.items()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs
