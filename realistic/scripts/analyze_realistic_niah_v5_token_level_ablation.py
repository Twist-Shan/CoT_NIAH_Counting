#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


TARGETING_METRICS = (
    "bank_target_attention_share_of_gold_mass",
    "bank_mean_head_target_relative_mass",
    "bank_target_top1_fraction",
    "bank_mean_target_minus_max_wrong_mass",
    "bank_source_specific_ov_write_norm_sum",
    "target_city_log_probability",
    "target_mean_token_logit_margin",
    "target_city_teacher_forced_exact",
    "target_city_retrieved",
)


ANSWER_METRICS = (
    "bank_prompt_records_mass_sum",
    "bank_prompt_records_broad_score_mean",
    "bank_trace_context_mass_sum",
    "bank_trace_context_broad_score_mean",
    "bank_trace_items_mass_sum",
    "bank_trace_items_broad_score_mean",
    "gold_first_answer_token_log_probability",
    "exact_count",
    "absolute_error",
)


def read_shards(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "shards").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["shard_path"] = str(path.relative_to(root))
                row["shard_line"] = int(line_number)
                rows.append(row)
    if not rows:
        raise ValueError(f"No token-level shards found under {root}")
    frame = pd.DataFrame(rows)
    if frame["experiment_id"].nunique() != 1:
        raise ValueError("One analysis root cannot mix token-level experiment modes")
    return frame


def _available_metrics(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    metrics = []
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            frame[column] = values
            metrics.append(column)
    return metrics


def paired_condition_effects(
    frame: pd.DataFrame, *, metrics: Sequence[str]
) -> pd.DataFrame:
    identifiers = ["model_label", "request_id", "seed", "condition"]
    optional = [
        "dataset_split",
        "target_grammar_class",
        "anchor_equivalence_id",
        "from_occurrence",
        "to_occurrence",
        "bank_sha256",
    ]
    identifiers.extend(column for column in optional if column in frame.columns)
    unit = [
        column
        for column in identifiers
        if column != "condition"
    ]
    collapsed = frame.groupby(identifiers, as_index=False, dropna=False).agg(
        **{metric: (metric, "mean") for metric in metrics},
        repeat_count=("condition", "size"),
    )
    clean = collapsed.loc[collapsed["condition"].eq("clean")]
    if clean.empty:
        raise ValueError("Every analysis needs clean rows")
    treatment = collapsed.loc[~collapsed["condition"].eq("clean")]
    merged = treatment.merge(
        clean[[*unit, *metrics]],
        on=unit,
        how="inner",
        validate="many_to_one",
        suffixes=("", "_clean"),
    )
    if len(merged) != len(treatment):
        raise ValueError("Some treatment units lack a clean pair")
    for metric in metrics:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_clean"]
    return merged


def factorial_effects(frame: pd.DataFrame, *, metrics: Sequence[str]) -> pd.DataFrame:
    experiment = str(frame["experiment_id"].iloc[0])
    if experiment == "targeting_trace_token_blank":
        arms = {
            "clean": "clean",
            "first_blank": "cumulative_trace_blank",
            "second_blank": "recent_transition_blank",
            "both_blank": "full_trace_blank",
        }
        factor_names = ("cumulative_trace", "recent_transition")
    else:
        arms = {
            "clean": "clean",
            "first_blank": "prompt_all_blank",
            "second_blank": "trace_all_blank",
            "both_blank": "prompt_and_trace_blank",
        }
        factor_names = ("prompt", "trace")
    selected = frame.loc[frame["condition"].isin(arms.values())].copy()
    unit = ["model_label", "request_id", "seed"]
    for column in (
        "dataset_split",
        "target_grammar_class",
        "anchor_equivalence_id",
        "from_occurrence",
        "to_occurrence",
        "bank_sha256",
    ):
        if column in selected.columns:
            unit.append(column)
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(unit, dropna=False):
        by_condition = {
            condition: values.iloc[0]
            for condition, values in group.groupby("condition")
        }
        if set(arms.values()) - set(by_condition):
            continue
        payload = dict(zip(unit, key if isinstance(key, tuple) else (key,)))
        for metric in metrics:
            clean = float(by_condition[arms["clean"]][metric])
            first = float(by_condition[arms["first_blank"]][metric])
            second = float(by_condition[arms["second_blank"]][metric])
            both = float(by_condition[arms["both_blank"]][metric])
            payload[f"{factor_names[0]}_blank_effect__{metric}"] = first - clean
            payload[f"{factor_names[1]}_blank_effect__{metric}"] = second - clean
            payload[f"joint_blank_effect__{metric}"] = both - clean
            payload[f"factorial_interaction__{metric}"] = (
                both - first - second + clean
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def summarize_seed_equal(
    effects: pd.DataFrame, *, metrics: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_columns = [f"delta_{metric}" for metric in metrics]
    group = ["model_label", "condition", "seed"]
    for column in ("dataset_split", "target_grammar_class"):
        if column in effects.columns:
            group.insert(-1, column)
    seed = effects.groupby(group, as_index=False, dropna=False).agg(
        **{column: (column, "mean") for column in delta_columns},
        paired_units=("request_id", "size"),
        requests=("request_id", "nunique"),
    )
    summary_group = [column for column in group if column != "seed"]
    rows: list[dict[str, Any]] = []
    for key, frame in seed.groupby(summary_group, dropna=False):
        payload = dict(
            zip(summary_group, key if isinstance(key, tuple) else (key,))
        )
        payload["seed_count"] = int(frame["seed"].nunique())
        payload["paired_units"] = int(frame["paired_units"].sum())
        for column in delta_columns:
            values = frame[column].to_numpy(float)
            finite = values[np.isfinite(values)]
            payload[f"mean_{column}"] = (
                float(np.mean(finite)) if len(finite) else math.nan
            )
            payload[f"se_{column}"] = (
                float(np.std(finite, ddof=1) / math.sqrt(len(finite)))
                if len(finite) > 1
                else math.nan
            )
        rows.append(payload)
    return seed, pd.DataFrame(rows)


def matched_control_specificity(
    effects: pd.DataFrame, *, metrics: Sequence[str]
) -> pd.DataFrame:
    pairs = {
        "early_half_trace_blank": "early_half_trace_matched_control",
        "cumulative_trace_blank": "cumulative_trace_matched_control",
        "recent_transition_blank": "recent_transition_matched_control",
        "full_trace_blank": "full_trace_matched_control",
    }
    rows: list[dict[str, Any]] = []
    unit = ["model_label", "request_id", "seed"]
    for column in (
        "dataset_split",
        "target_grammar_class",
        "anchor_equivalence_id",
        "from_occurrence",
        "to_occurrence",
        "bank_sha256",
    ):
        if column in effects.columns:
            unit.append(column)
    for treatment, control in pairs.items():
        left = effects.loc[effects["condition"].eq(treatment)]
        right = effects.loc[effects["condition"].eq(control)]
        if left.empty or right.empty:
            continue
        merged = left.merge(
            right[[*unit, *[f"delta_{metric}" for metric in metrics]]],
            on=unit,
            how="inner",
            validate="one_to_one",
            suffixes=("_treatment", "_control"),
        )
        for row in merged.to_dict("records"):
            payload = {column: row[column] for column in unit}
            payload.update(
                {
                    "treatment": treatment,
                    "matched_control": control,
                }
            )
            for metric in metrics:
                payload[f"specificity__{metric}"] = float(
                    row[f"delta_{metric}_treatment"]
                    - row[f"delta_{metric}_control"]
                )
            rows.append(payload)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = read_shards(args.input)
    status = (
        frame["status"].astype(str)
        if "status" in frame.columns
        else pd.Series("ok", index=frame.index)
    )
    completed = frame.loc[status.eq("ok")].copy()
    excluded = frame.loc[~frame.index.isin(completed.index)].copy()
    experiment = str(completed["experiment_id"].iloc[0])
    candidates = (
        TARGETING_METRICS
        if experiment == "targeting_trace_token_blank"
        else ANSWER_METRICS
    )
    metrics = _available_metrics(completed, candidates)
    if not metrics:
        raise ValueError("No registered outcome metric is available")
    effects = paired_condition_effects(completed, metrics=metrics)
    factorial = factorial_effects(completed, metrics=metrics)
    seed, summary = summarize_seed_equal(effects, metrics=metrics)
    specificity = (
        matched_control_specificity(effects, metrics=metrics)
        if experiment == "targeting_trace_token_blank"
        else pd.DataFrame()
    )

    args.output.mkdir(parents=True, exist_ok=True)
    flat = completed.drop(
        columns=[
            column
            for column in (
                "head_metrics",
                "attention_blank_hook_audit",
                "score_blank_hook_audit",
                "generation_blank_hook_audit",
            )
            if column in completed.columns
        ]
    )
    flat.to_csv(args.output / "token_level_detail.csv", index=False)
    effects.to_csv(args.output / "paired_condition_effects.csv", index=False)
    seed.to_csv(args.output / "seed_equal_effects.csv", index=False)
    summary.to_csv(args.output / "seed_equal_summary.csv", index=False)
    factorial.to_csv(args.output / "factorial_effects.csv", index=False)
    specificity.to_csv(args.output / "matched_control_specificity.csv", index=False)
    excluded.to_csv(args.output / "excluded_rows.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v5_token_level_ablation_analysis_v1",
        "status": "PASS",
        "experiment_id": experiment,
        "input": str(args.input.resolve()),
        "input_shard_count": len(list((args.input / "shards").glob("*.jsonl"))),
        "completed_rows": int(len(completed)),
        "excluded_rows": int(len(excluded)),
        "request_count": int(completed["request_id"].nunique()),
        "seed_count": int(completed["seed"].nunique()),
        "conditions": sorted(completed["condition"].astype(str).unique()),
        "metrics": metrics,
        "estimand": "request/anchor paired to clean, events averaged within seed, seeds equal weighted",
        "factorial_definition": (
            "joint - first_blank - second_blank + clean"
        ),
        "token_deletion_used": False,
        "analysis_files_sha256": {},
    }
    for path in sorted(args.output.glob("*.csv")):
        audit["analysis_files_sha256"][path.name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
