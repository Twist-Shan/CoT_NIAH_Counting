#!/usr/bin/env python3
"""Build a frozen selected-vs-layer-matched plan for count-endpoint trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value).encode("utf-8")).hexdigest()


def build_plan(
    selection: dict[str, Any],
    *,
    heads_per_layer: int,
    random_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    model = str(selection["model_label"])
    panel = selection["sample_panel"]
    development = selection["development_selection"]
    selected = [
        [int(layer), int(head)]
        for layer, head in development["primary_bank_heads"]
    ]
    if not selected or len(selected) != len({tuple(value) for value in selected}):
        raise ValueError("primary_bank_heads must be non-empty and unique")
    if any(not 0 <= head < int(heads_per_layer) for _layer, head in selected):
        raise ValueError("A selected head falls outside --heads-per-layer")
    expected_selected_sha = str(development["primary_bank_sha256"])
    if _sha256_json(selected) != expected_selected_sha:
        raise ValueError("primary_bank_heads disagree with primary_bank_sha256")

    per_layer: dict[int, int] = {}
    for layer, _head in selected:
        per_layer[layer] = per_layer.get(layer, 0) + 1
    selected_set = {tuple(value) for value in selected}
    rng = np.random.default_rng(int(random_seed))
    controls: list[list[list[int]]] = []
    for _repeat in range(int(random_repeats)):
        control: list[list[int]] = []
        for layer, needed in sorted(per_layer.items()):
            candidates = [
                head
                for head in range(int(heads_per_layer))
                if (layer, head) not in selected_set
            ]
            if len(candidates) < needed:
                raise ValueError(
                    f"L{layer} has {len(candidates)} selected-excluded controls "
                    f"for {needed} requested heads"
                )
            chosen = rng.choice(candidates, size=needed, replace=False)
            control.extend([[layer, int(head)] for head in sorted(chosen.tolist())])
        controls.append(control)

    validation = [int(value) for value in panel["seeds"]]
    training = [int(value) for value in panel["discovery_seeds"]]
    common = {
        "model_label": model,
        "mechanism": "retrieval_anchor_localization",
        "query_site_kind": (
            "grammar_aware_transition_anchor:"
            + str(development["head_ranking_source_anchor"])
        ),
        "experiment_id": "targeted_retrieval_final_count_endpoint",
        "fold": 0,
        "training_seeds": json.dumps(training),
        "validation_seeds": json.dumps(validation),
        "bank_size": len(selected),
        "bank_selection_policy": (
            "previously_frozen_discovery_bank_with_new_outcome_blind_controls"
        ),
        "selection_metric": str(development["head_ranking_metric"]),
        "selection_metric_column": "source_attention_mass",
        "selection_anchor_role": str(development["head_ranking_source_anchor"]),
        "selection_target_grammar_class": str(
            development["head_ranking_source_grammar"]
        ),
        "selection_site_scope": "exact_single_query_anchor",
        "selection_eligibility_scope": "local",
        "selection_aggregation": "seed_first_equal_anchor_mean",
        "random_control_matching": "layer_matched_selected_excluded",
    }
    rows = [
        {
            **common,
            "condition": "selected_bank",
            "repeat": 0,
            "heads": json.dumps(selected),
            "bank_sha256": _sha256_json(selected),
        }
    ]
    rows.extend(
        {
            **common,
            "condition": "layer_matched_random",
            "repeat": repeat,
            "heads": json.dumps(control),
            "bank_sha256": _sha256_json(control),
        }
        for repeat, control in enumerate(controls, start=1)
    )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads-per-layer", type=int, required=True)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20260821)
    args = parser.parse_args(argv)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    frame = build_plan(
        selection,
        heads_per_layer=int(args.heads_per_layer),
        random_repeats=int(args.random_repeats),
        random_seed=int(args.random_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(args.output)
    audit = {
        "status": "PASS",
        "plan_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "model_label": str(selection["model_label"]),
        "selected_bank_sha256": str(frame.iloc[0]["bank_sha256"]),
        "conditions": frame["condition"].value_counts().to_dict(),
        "random_seed": int(args.random_seed),
        "control_rule": "layer-matched, selected-excluded, without replacement",
        "endpoint_blind": True,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
