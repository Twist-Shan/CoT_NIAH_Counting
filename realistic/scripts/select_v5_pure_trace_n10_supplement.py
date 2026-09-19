#!/usr/bin/env python3
"""Apply the frozen whole-trace-pure N=10 grammar filter to a supplement registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGETS = {
    "Qwen3-8B": ("adjacent_rank_after_city", "inline_count"),
    "Gemma4-E4B": ("same_unit_rank_before_city", "inline_count"),
}
CLASSES = tuple(range(1, 11))
SCHEMA = "realistic_niah_v5_pure_trace_n10_supplement_selection_v1"


def _truth(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(False).astype(str).str.lower().eq("true")


def select(registry_path: Path, output_dir: Path) -> dict[str, object]:
    registry = pd.read_csv(registry_path)
    eligible = registry.loc[
        _truth(registry, "primary_full_chain_event")
        & _truth(registry, "progress_commit_eligible")
        & _truth(registry, "progress_commit_site_resolved")
        & _truth(registry, "exact_count")
        & registry["trace_category"].astype(str).eq("one_to_one")
        & registry["gold_count"].astype(int).eq(10)
        & registry["parsed_count"].astype(int).eq(10)
    ].copy()
    pure_rows = []
    for (model, request_id), group in eligible.groupby(
        ["model_label", "request_id"], sort=True
    ):
        occurrences = group["occurrence"].astype(int).tolist()
        ranks = group["rank"].astype(int).tolist()
        grammars = sorted(set(group["grammar_class"].astype(str)))
        markers = sorted(set(group["marker_kind"].astype(str)))
        if len(group) != 10:
            continue
        if sorted(occurrences) != list(CLASSES) or sorted(ranks) != list(CLASSES):
            continue
        if any(rank != occurrence for rank, occurrence in zip(ranks, occurrences)):
            continue
        if len(grammars) != 1 or len(markers) != 1:
            continue
        first = group.iloc[0]
        pure_rows.append(
            {
                "model_label": str(model),
                "request_id": str(request_id),
                "split": str(first["split"]),
                "seed": int(first["seed"]),
                "gold_count": 10,
                "grammar_class": grammars[0],
                "marker_kind": markers[0],
            }
        )
    pure = pd.DataFrame(pure_rows)
    if pure.empty:
        raise ValueError("No whole-trace-pure N=10 supplement trajectories")
    selected_parts = []
    support_rows = []
    for model, (grammar, marker) in TARGETS.items():
        model_pure = pure.loc[pure["model_label"].astype(str).eq(model)]
        for (observed_grammar, observed_marker), group in model_pure.groupby(
            ["grammar_class", "marker_kind"], sort=True
        ):
            support_rows.append(
                {
                    "model_label": model,
                    "grammar_class": str(observed_grammar),
                    "marker_kind": str(observed_marker),
                    "discovery_trajectories": int(
                        group["split"].astype(str).eq("discovery").sum()
                    ),
                    "confirmation_trajectories": int(
                        group["split"].astype(str).eq("confirmation").sum()
                    ),
                    "total_trajectories": int(len(group)),
                }
            )
        selected_parts.append(
            model_pure.loc[
                model_pure["grammar_class"].astype(str).eq(grammar)
                & model_pure["marker_kind"].astype(str).eq(marker)
            ]
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(
        ["model_label", "split", "seed"], kind="mergesort"
    ).reset_index(drop=True)
    support = pd.DataFrame(support_rows).sort_values(
        ["model_label", "discovery_trajectories", "grammar_class"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pure.to_csv(output_dir / "all_pure_trace_trajectories.csv", index=False)
    support.to_csv(output_dir / "pure_trace_grammar_support.csv", index=False)
    selected.to_csv(output_dir / "selected_frozen_grammar_trajectories.csv", index=False)
    counts = {
        model: {
            split: int(
                (
                    selected["model_label"].astype(str).eq(model)
                    & selected["split"].astype(str).eq(split)
                ).sum()
            )
            for split in ("discovery", "confirmation")
        }
        for model in TARGETS
    }
    result = {
        "schema_version": SCHEMA,
        "frozen_targets": {
            model: {"grammar_class": grammar, "marker_kind": marker}
            for model, (grammar, marker) in TARGETS.items()
        },
        "selected_counts": counts,
        "selection_rule": (
            "whole-trace exact one-to-one N=10 with ranks=occurrences=1..10; "
            "one grammar and marker kind; grammar targets frozen before supplement"
        ),
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    select(args.event_registry, args.output_dir)


if __name__ == "__main__":
    main()
