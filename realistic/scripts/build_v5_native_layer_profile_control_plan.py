#!/usr/bin/env python3
"""Freeze a P0-ranked bank with the exact layer profile of a P2 bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import (  # noqa: E402
    global_random_controls,
    layer_matched_random_controls,
    ranked_bank_with_layer_profile,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _heads(value: str) -> list[tuple[int, int]]:
    return [(int(layer), int(head)) for layer, head in json.loads(value)]


def _bank_sha(heads: list[tuple[int, int]]) -> str:
    return hashlib.sha256(json.dumps(heads).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    p0_source = parser.add_mutually_exclusive_group(required=True)
    p0_source.add_argument("--p0-plan-dir", type=Path)
    p0_source.add_argument(
        "--p0-ranking",
        type=Path,
        help="Reuse an already frozen P0 grammar ranking without recapturing P0.",
    )
    parser.add_argument("--p2-plan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--random-control-matching",
        choices=["layer_matched", "global"],
        default="layer_matched",
    )
    parser.add_argument("--random-repeats", type=int, default=3)
    args = parser.parse_args()

    if args.random_repeats < 1:
        raise ValueError("random-repeats must be positive")
    p0_ranking_path = (
        args.p0_ranking
        if args.p0_ranking is not None
        else args.p0_plan_dir / "crossfit_source_specific_head_ranking.csv"
    )
    p0_plan_path = (
        None
        if args.p0_plan_dir is None
        else args.p0_plan_dir / "retrieval_anchor_bank_plan.csv"
    )
    p2_plan_path = args.p2_plan_dir / "retrieval_anchor_bank_plan.csv"
    p0_ranking = pd.read_csv(p0_ranking_path)
    p2_plan = pd.read_csv(
        p2_plan_path, engine="python", dtype=str, keep_default_na=False
    )
    p2_selected = p2_plan.loc[p2_plan["condition"].eq("selected_bank")]
    if len(p2_selected) != 1:
        raise ValueError("Expected one frozen selected bank in the P2 reference plan")
    p2_row = p2_selected.iloc[0].to_dict()
    if p0_plan_path is not None:
        p0_plan = pd.read_csv(
            p0_plan_path, engine="python", dtype=str, keep_default_na=False
        )
        p0_selected = p0_plan.loc[p0_plan["condition"].eq("selected_bank")]
        if len(p0_selected) != 1:
            raise ValueError("Expected one frozen selected bank in the P0 plan")
        p0_row = p0_selected.iloc[0].to_dict()
        if p0_row["model_label"] != p2_row["model_label"]:
            raise ValueError("P0 and P2 plans must use the same model")
        if (
            p0_row["selection_target_grammar_class"]
            != p2_row["selection_target_grammar_class"]
        ):
            raise ValueError("P0 and P2 plans must use the same target grammar")
        if p0_row["validation_seeds"] != p2_row["validation_seeds"]:
            raise ValueError("P0 and P2 plans must share validation seeds")
    else:
        p0_row = dict(p2_row)
        p0_row["selection_anchor_role"] = "p0_item_end"
        p0_row["query_site_kind"] = (
            "grammar_aware_transition_anchor:p0_item_end"
        )
        p0_row["selection_metric"] = (
            "seed_event_mean_target_source_attention_mass"
        )
        p0_row["selection_metric_column"] = "target_source_attention_mass"
        if "grammar" in p0_ranking.columns:
            observed_grammars = set(p0_ranking["grammar"].astype(str))
            expected_grammar = p2_row["selection_target_grammar_class"]
            if observed_grammars != {expected_grammar}:
                raise ValueError(
                    "Direct P0 ranking grammar differs from the P2 reference"
                )

    p2_heads = _heads(p2_row["heads"])
    layer_profile = dict(sorted(Counter(layer for layer, _head in p2_heads).items()))
    selected = ranked_bank_with_layer_profile(
        p0_ranking, layer_profile=layer_profile
    )
    if len(selected) != len(p2_heads):
        raise AssertionError((len(selected), len(p2_heads)))
    control_builder = (
        layer_matched_random_controls
        if args.random_control_matching == "layer_matched"
        else global_random_controls
    )
    controls = control_builder(
        p0_ranking,
        selected,
        repeats=args.random_repeats,
        seed_text=(
            "v5-hybrid-layer-profile:"
            f"{p0_row['model_label']}:"
            f"{p0_row['selection_target_grammar_class']}:"
            f"K{len(selected)}:{_bank_sha(p2_heads)}"
        ),
    )

    common = dict(p0_row)
    for key in ("condition", "repeat", "heads", "bank_sha256"):
        common.pop(key, None)
    common["bank_size"] = str(len(selected))
    common["bank_selection_policy"] = (
        "p0_discovery_ranking_constrained_to_p2_selected_layer_profile"
    )
    common["random_control_matching"] = args.random_control_matching
    common["layer_profile_reference_selection_anchor_role"] = p2_row[
        "selection_anchor_role"
    ]
    common["layer_profile_reference_bank_sha256"] = p2_row["bank_sha256"]
    common["layer_profile_reference_plan_sha256"] = _sha256(p2_plan_path)
    common["p0_ranking_source_sha256"] = _sha256(p0_ranking_path)
    rows = [
        {
            **common,
            "condition": "selected_bank",
            "repeat": "0",
            "heads": json.dumps(selected),
            "bank_sha256": _bank_sha(selected),
        }
    ]
    random_condition = (
        "layer_matched_random"
        if args.random_control_matching == "layer_matched"
        else "global_random"
    )
    rows.extend(
        {
            **common,
            "condition": random_condition,
            "repeat": str(repeat),
            "heads": json.dumps(control),
            "bank_sha256": _bank_sha(control),
        }
        for repeat, control in enumerate(controls, start=1)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    output_plan = args.output / "retrieval_anchor_bank_plan.csv"
    output_ranking = args.output / "crossfit_source_specific_head_ranking.csv"
    output_audit = args.output / "causal_plan_audit.json"
    pd.DataFrame(rows).to_csv(output_plan, index=False)
    ranked = p0_ranking.copy()
    selected_set = set(selected)
    ranked["selected_by_layer_profile_control"] = [
        (int(layer), int(head)) in selected_set
        for layer, head in zip(ranked["layer"], ranked["head"])
    ]
    ranked.to_csv(output_ranking, index=False)
    audit = {
        "schema_version": "realistic_niah_v5_layer_profile_control_plan_v1",
        "status": "PASS",
        "model_label": p0_row["model_label"],
        "target_grammar_class": p0_row["selection_target_grammar_class"],
        "bank_size": len(selected),
        "selection_anchor_role": p0_row["selection_anchor_role"],
        "selection_metric": p0_row["selection_metric"],
        "selection_aggregation": p0_row["selection_aggregation"],
        "layer_profile_reference_selection_anchor_role": p2_row[
            "selection_anchor_role"
        ],
        "layer_profile": {str(layer): count for layer, count in layer_profile.items()},
        "selected_bank_sha256": _bank_sha(selected),
        "reference_bank_sha256": p2_row["bank_sha256"],
        "random_control_matching": args.random_control_matching,
        "random_repeats": args.random_repeats,
        "p0_plan_sha256": (
            _sha256(p0_plan_path) if p0_plan_path is not None else None
        ),
        "p0_ranking_sha256": _sha256(p0_ranking_path),
        "p2_plan_sha256": _sha256(p2_plan_path),
        "output_plan_sha256": _sha256(output_plan),
    }
    output_audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
