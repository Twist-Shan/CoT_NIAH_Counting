#!/usr/bin/env python3
"""Summarize small-sample marker-scrubbed restoration diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for path in sorted((args.input / "shards").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("No diagnostic shard rows found")
    frame = pd.DataFrame(rows)
    key = ["seed", "target_occurrence"]
    source = (
        frame[frame["condition"].eq("source_reference")]
        .set_index(key)[
            [
                "running_target_margin",
                "running_target_exact",
                "next_token_target_margin",
                "next_token_running_target_exact",
            ]
        ]
        .rename(
            columns={
                "running_target_margin": "source_margin",
                "running_target_exact": "source_exact",
                "next_token_target_margin": "source_next_token_margin",
                "next_token_running_target_exact": "source_next_token_exact",
            }
        )
    )
    blank = (
        frame[frame["condition"].eq("blank_reference")]
        .set_index(key)[
            [
                "running_target_margin",
                "running_target_exact",
                "next_token_target_margin",
                "next_token_running_target_exact",
            ]
        ]
        .rename(
            columns={
                "running_target_margin": "blank_margin",
                "running_target_exact": "blank_exact",
                "next_token_target_margin": "blank_next_token_margin",
                "next_token_running_target_exact": "blank_next_token_exact",
            }
        )
    )
    baselines = source.join(blank, validate="one_to_one")
    conditions = [
        "source_reference",
        "blank_reference",
        "single_item_once",
        "single_item_once_cache_aligned",
        "all_items_once",
        "single_item_cumulative_clamp",
        "full_prefix_identity",
    ]
    summary_rows = []
    for condition in conditions:
        selected = frame[frame["condition"].eq(condition)].set_index(key)
        if len(selected) != len(baselines):
            raise ValueError(f"Condition {condition} is missing diagnostic cells")
        joined = selected.join(baselines, validate="one_to_one")
        margin = float(joined["running_target_margin"].mean())
        exact = float(joined["running_target_exact"].astype(float).mean())
        source_margin = float(joined["source_margin"].mean())
        blank_margin = float(joined["blank_margin"].mean())
        source_exact = float(joined["source_exact"].astype(float).mean())
        blank_exact = float(joined["blank_exact"].astype(float).mean())
        margin_denom = source_margin - blank_margin
        exact_denom = source_exact - blank_exact
        next_margin = float(joined["next_token_target_margin"].mean())
        source_next_margin = float(joined["source_next_token_margin"].mean())
        blank_next_margin = float(joined["blank_next_token_margin"].mean())
        next_margin_denom = source_next_margin - blank_next_margin
        next_exact = float(
            joined["next_token_running_target_exact"].astype(float).mean()
        )
        source_next_exact = float(joined["source_next_token_exact"].astype(float).mean())
        blank_next_exact = float(joined["blank_next_token_exact"].astype(float).mean())
        next_exact_denom = source_next_exact - blank_next_exact
        summary_rows.append(
            {
                "condition": condition,
                "cells": len(joined),
                "mean_target_margin": margin,
                "exact_accuracy": exact,
                "delta_margin_vs_blank": margin - blank_margin,
                "margin_gap_closure": (
                    (margin - blank_margin) / margin_denom
                    if margin_denom != 0
                    else float("nan")
                ),
                "exact_gap_closure": (
                    (exact - blank_exact) / exact_denom
                    if exact_denom != 0
                    else float("nan")
                ),
                "mean_next_token_target_margin": next_margin,
                "next_token_exact_accuracy": next_exact,
                "delta_next_token_margin_vs_blank": next_margin - blank_next_margin,
                "next_token_margin_gap_closure": (
                    (next_margin - blank_next_margin) / next_margin_denom
                    if next_margin_denom != 0
                    else float("nan")
                ),
                "next_token_exact_gap_closure": (
                    (next_exact - blank_next_exact) / next_exact_denom
                    if next_exact_denom != 0
                    else float("nan")
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    identity = frame[frame["condition"].eq("full_prefix_identity")]
    identity_max_abs_delta = float(
        identity["source_candidate_log_score_max_abs_delta"].astype(float).max()
    )
    identity_next_token_max_abs_delta = float(
        identity["source_next_token_candidate_logit_max_abs_delta"].astype(float).max()
    )
    identity_prefill_max_abs_delta = float(
        identity["source_prefill_last_logit_max_abs_delta"].astype(float).max()
    )
    payload = {
        "schema_version": "realistic_niah_v5_bullet_restore_diagnostic_analysis_v1",
        "status": (
            "PASS_NEXT_TOKEN_IDENTITY"
            if identity_prefill_max_abs_delta <= 1e-6
            else "FAIL_NEXT_TOKEN_IDENTITY"
        ),
        "model_label": str(frame["model_label"].iloc[0]),
        "seed_count": int(frame["seed"].nunique()),
        "target_occurrences": sorted(
            int(value) for value in frame["target_occurrence"].unique()
        ),
        "source_layer": int(frame["source_layer"].iloc[0]),
        "identity_candidate_log_score_max_abs_delta": identity_max_abs_delta,
        "identity_next_token_candidate_logit_max_abs_delta": (
            identity_next_token_max_abs_delta
        ),
        "identity_prefill_last_logit_max_abs_delta": identity_prefill_max_abs_delta,
        "conditions": summary.to_dict(orient="records"),
        "diagnostic_only": True,
        "formal_confirmation_unchanged": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "condition_summary.csv", index=False)
    (args.output / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
