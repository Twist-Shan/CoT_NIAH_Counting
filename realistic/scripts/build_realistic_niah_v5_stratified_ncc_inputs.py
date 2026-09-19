#!/usr/bin/env python3
"""Freeze timing-specific panels and retrieval-head banks for NCC 5.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import layer_matched_random_controls  # noqa: E402
from realistic_niah_v5.count_stream import (  # noqa: E402
    COUNT_STREAM_CONFIRMATION_SEEDS,
    COUNT_STREAM_DISCOVERY_SEEDS,
)


MODEL_CONTRACTS = {
    "Qwen3-8B": {
        "bank_size": 128,
        "after_grammar": "adjacent_rank_after_city",
        "before_grammar_preferences": (
            "adjacent_rank_before_city",
            "same_unit_rank_before_city",
        ),
    },
    "Gemma4-E4B": {
        "bank_size": 6,
        "after_grammar": "adjacent_rank_after_city",
        "before_grammar_preferences": (
            "same_unit_rank_before_city",
            "adjacent_rank_before_city",
        ),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _timing(grammar_class: str) -> str | None:
    if "rank_after_city" in grammar_class:
        return "rank_after_city"
    if "rank_before_city" in grammar_class:
        return "rank_before_city"
    return None


def _grammar_preferences(model_label: str, timing: str) -> tuple[str, ...]:
    contract = MODEL_CONTRACTS[model_label]
    if timing == "rank_after_city":
        return (str(contract["after_grammar"]),)
    return tuple(str(value) for value in contract["before_grammar_preferences"])


def _select_panel(
    rows: Sequence[dict[str, Any]], *, model_label: str, timing: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preferences = _grammar_preferences(model_label, timing)
    preference = {grammar: index for index, grammar in enumerate(preferences)}
    selected: list[dict[str, Any]] = []
    phase_rows: dict[str, list[dict[str, Any]]] = {}
    for phase, phase_seeds in (
        ("development", COUNT_STREAM_DISCOVERY_SEEDS),
        ("confirmation", COUNT_STREAM_CONFIRMATION_SEEDS),
    ):
        active: list[dict[str, Any]] = []
        for seed in phase_seeds:
            candidates = [
                row
                for row in rows
                if int(row["seed"]) == int(seed)
                and _timing(str(row["target_grammar_class"])) == timing
            ]
            if not candidates:
                continue
            # Count support is chosen without model outcomes; grammar preference
            # only breaks a gold-count tie and matches the branch's bank source.
            chosen = min(
                candidates,
                key=lambda row: (
                    -int(row["gold_count"]),
                    preference.get(str(row["target_grammar_class"]), len(preference)),
                    str(row["request_id"]),
                ),
            )
            frozen = {
                **chosen,
                "grammar_span_timing_stratum": timing,
                "stratified_ncc_seed_role": phase,
                "stratified_ncc_outcome_blind": True,
                "stratified_ncc_selection_rank_used": False,
                "stratified_ncc_selection_rule": (
                    "within_timing_highest_gold_count_then_model_grammar_preference_"
                    "then_request_id"
                ),
            }
            active.append(frozen)
        active.sort(key=lambda row: int(row["seed"]))
        if len(active) != len({int(row["seed"]) for row in active}):
            raise ValueError(f"{model_label} {timing} {phase} has duplicate seeds")
        phase_rows[phase] = active
        selected.extend(active)
    if any("selection_rank" in row for row in selected):
        raise ValueError("Stratified NCC panel must not contain selection_rank")
    selected.sort(key=lambda row: int(row["seed"]))
    audit = {
        phase: {
            "seed_count": len(active),
            "seeds": [int(row["seed"]) for row in active],
            "missing_fixed_seeds": sorted(
                set(
                    COUNT_STREAM_DISCOVERY_SEEDS
                    if phase == "development"
                    else COUNT_STREAM_CONFIRMATION_SEEDS
                )
                - {int(row["seed"]) for row in active}
            ),
            "grammar_counts": {
                grammar: sum(
                    str(row["target_grammar_class"]) == grammar for row in active
                )
                for grammar in sorted(
                    {str(row["target_grammar_class"]) for row in active}
                )
            },
        }
        for phase, active in phase_rows.items()
    }
    return selected, audit


def _heads(raw: Any) -> list[tuple[int, int]]:
    values = json.loads(str(raw)) if isinstance(raw, str) else raw
    return [(int(layer), int(head)) for layer, head in values]


def _after_bank(
    head_scores: pd.DataFrame, *, model_label: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = MODEL_CONTRACTS[model_label]
    grammar = str(contract["after_grammar"])
    bank_size = int(contract["bank_size"])
    ranking = head_scores.loc[
        head_scores["model_label"].astype(str).eq(model_label)
        & head_scores["grammar"].astype(str).eq(grammar)
    ].copy()
    if ranking.empty:
        raise ValueError(f"No discovery P0 ranking for {model_label} {grammar}")
    if int(ranking["n_seeds"].min()) != len(COUNT_STREAM_DISCOVERY_SEEDS):
        raise ValueError("Rank-after bank is not based on all discovery seeds")
    ranking = ranking.sort_values(["rank", "layer", "head"]).reset_index(drop=True)
    selected = [
        (int(row.layer), int(row.head))
        for row in ranking.head(bank_size).itertuples(index=False)
    ]
    if len(selected) != bank_size or len(set(selected)) != bank_size:
        raise ValueError("Rank-after selected bank is incomplete or duplicated")
    controls = layer_matched_random_controls(
        ranking,
        selected,
        repeats=3,
        seed_text=f"v5-stratified-ncc:{model_label}:{grammar}:P0:Top{bank_size}",
    )
    common = {
        "model_label": model_label,
        "mechanism": "retrieval_anchor_localization",
        "query_site_kind": "grammar_aware_transition_anchor:p0_item_end",
        "experiment_id": "stratified_targeted_retrieval_counter_ncc",
        "fold": 0,
        "training_seeds": json.dumps(list(COUNT_STREAM_DISCOVERY_SEEDS)),
        "validation_seeds": json.dumps(
            list(COUNT_STREAM_DISCOVERY_SEEDS + COUNT_STREAM_CONFIRMATION_SEEDS)
        ),
        "bank_size": bank_size,
        "bank_selection_policy": "frozen_discovery_p0_grammar_specific_top_k",
        "selection_metric": "seed_event_mean_target_source_attention_mass",
        "selection_metric_column": "score",
        "selection_anchor_role": "p0_item_end",
        "selection_target_grammar_class": grammar,
        "selection_site_scope": "exact_single_query_anchor",
        "selection_eligibility_scope": "local",
        "selection_aggregation": "seed_event_mean_over_discovery",
        "random_control_matching": "layer_matched_selected_excluded",
        "timing_branch": "rank_after_city",
        "selection_rank_used_for_ncc_panel": False,
    }

    def make_row(
        heads: Sequence[tuple[int, int]], condition: str, repeat: int
    ) -> dict[str, Any]:
        normalized = [[int(layer), int(head)] for layer, head in heads]
        return {
            **common,
            "condition": condition,
            "repeat": int(repeat),
            "heads": json.dumps(normalized),
            "bank_sha256": _sha256_json(normalized),
        }

    rows = [make_row(selected, "selected_bank", 0)] + [
        make_row(control, "layer_matched_random", repeat)
        for repeat, control in enumerate(controls, start=1)
    ]
    layer_counts: dict[int, int] = {}
    for layer, _head in selected:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    return pd.DataFrame(rows), {
        "source_grammar": grammar,
        "bank_size": bank_size,
        "selected_heads": [list(value) for value in selected],
        "selected_bank_sha256": rows[0]["bank_sha256"],
        "selected_layer_counts": {
            str(layer): count for layer, count in sorted(layer_counts.items())
        },
        "maximum_head_layer": max(layer for layer, _head in selected),
        "capture_start_layer": max(layer for layer, _head in selected) + 1,
        "discovery_seed_count": int(ranking["n_seeds"].min()),
    }


def _before_bank(
    current_bank: pd.DataFrame, *, model_label: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = current_bank.loc[
        current_bank["model_label"].astype(str).eq(model_label)
    ].copy()
    counts = frame["condition"].astype(str).value_counts().to_dict()
    if counts != {"layer_matched_random": 3, "selected_bank": 1}:
        raise ValueError(f"Frozen rank-before bank factorial changed: {counts}")
    if "selection_rank" in frame.columns:
        raise ValueError("Frozen rank-before plan contains forbidden selection_rank")
    selected_row = frame.loc[frame["condition"].astype(str).eq("selected_bank")].iloc[0]
    selected = _heads(selected_row["heads"])
    expected_size = int(MODEL_CONTRACTS[model_label]["bank_size"])
    if len(selected) != expected_size:
        raise ValueError("Frozen rank-before selected-bank size changed")
    selected_layers: dict[int, int] = {}
    for layer, _head in selected:
        selected_layers[layer] = selected_layers.get(layer, 0) + 1
    selected_set = set(selected)
    for row in frame.loc[
        frame["condition"].astype(str).eq("layer_matched_random")
    ].itertuples(index=False):
        active = _heads(row.heads)
        counts_by_layer: dict[int, int] = {}
        for layer, _head in active:
            counts_by_layer[layer] = counts_by_layer.get(layer, 0) + 1
        if counts_by_layer != selected_layers or selected_set & set(active):
            raise ValueError("Frozen rank-before control is not disjoint layer matched")
    frame["timing_branch"] = "rank_before_city"
    maximum = max(selected_layers)
    return frame, {
        "source_grammar": str(selected_row["selection_target_grammar_class"]),
        "bank_size": expected_size,
        "selected_heads": [list(value) for value in selected],
        "selected_bank_sha256": str(selected_row["bank_sha256"]),
        "selected_layer_counts": {
            str(layer): count for layer, count in sorted(selected_layers.items())
        },
        "maximum_head_layer": maximum,
        "capture_start_layer": maximum + 1,
        "discovery_seed_count": len(COUNT_STREAM_DISCOVERY_SEEDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CONTRACTS), required=True)
    parser.add_argument("--targeted-registry", type=Path, required=True)
    parser.add_argument("--current-bank-plan", type=Path, required=True)
    parser.add_argument("--head-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_label = str(args.model)
    registry = _read_jsonl(args.targeted_registry)
    if not registry or any(str(row.get("request_id", "")) == "" for row in registry):
        raise ValueError("Targeted registry is empty or malformed")
    if len(registry) != len({str(row["request_id"]) for row in registry}):
        raise ValueError("Targeted registry contains duplicate request IDs")
    if {str(row.get("request_id", "")).split("/")[0] for row in registry} != {
        model_label
    }:
        raise ValueError("Targeted registry belongs to another model")

    head_scores = pd.read_csv(args.head_scores)
    current_bank = pd.read_csv(args.current_bank_plan)
    panels: dict[str, dict[str, Any]] = {}
    banks: dict[str, dict[str, Any]] = {}
    panel_paths: dict[str, Path] = {}
    bank_paths: dict[str, Path] = {}
    for timing in ("rank_after_city", "rank_before_city"):
        panel, panel_audit = _select_panel(
            registry, model_label=model_label, timing=timing
        )
        panel_path = args.output / f"{timing}_panel.jsonl"
        _atomic_jsonl(panel_path, panel)
        panel_paths[timing] = panel_path
        panels[timing] = panel_audit

        if timing == "rank_after_city":
            bank, bank_audit = _after_bank(head_scores, model_label=model_label)
        else:
            bank, bank_audit = _before_bank(current_bank, model_label=model_label)
        bank_path = args.output / f"{timing}_bank_plan.csv"
        _atomic_text(bank_path, bank.to_csv(index=False, lineterminator="\n"))
        bank_paths[timing] = bank_path
        banks[timing] = bank_audit

    before_set = {
        tuple(value) for value in banks["rank_before_city"]["selected_heads"]
    }
    after_set = {
        tuple(value) for value in banks["rank_after_city"]["selected_heads"]
    }
    overlap = before_set & after_set
    manifest = {
        "schema_version": "realistic_niah_v5_stratified_targeted_counter_ncc_inputs_v1",
        "status": "FROZEN",
        "model_label": model_label,
        "design": "two_independent_timing_specific_estimands_then_standardized_synthesis",
        "cohort_policy": "maximal_eligible_seed_panel_within_fixed_phase_and_timing",
        "cohorts_are_unpaired_across_timing_branches": True,
        "panel_selection_uses_model_outcomes": False,
        "confirmation_used_for_bank_selection": False,
        "inputs": {
            "targeted_registry": {
                "path": str(args.targeted_registry.resolve()),
                "sha256": _sha256(args.targeted_registry),
            },
            "current_rank_before_bank_plan": {
                "path": str(args.current_bank_plan.resolve()),
                "sha256": _sha256(args.current_bank_plan),
            },
            "discovery_p0_head_scores": {
                "path": str(args.head_scores.resolve()),
                "sha256": _sha256(args.head_scores),
            },
        },
        "panels": {
            timing: {
                **panels[timing],
                "path": str(panel_paths[timing].resolve()),
                "sha256": _sha256(panel_paths[timing]),
            }
            for timing in panels
        },
        "banks": {
            timing: {
                **banks[timing],
                "path": str(bank_paths[timing].resolve()),
                "sha256": _sha256(bank_paths[timing]),
            }
            for timing in banks
        },
        "selected_bank_overlap": {
            "head_count": len(overlap),
            "fraction_of_bank": len(overlap)
            / int(MODEL_CONTRACTS[model_label]["bank_size"]),
            "heads": [list(value) for value in sorted(overlap)],
        },
    }
    _atomic_json(args.output / "stratified_ncc_input_manifest.json", manifest)
    print(
        json.dumps(
            {
                "model_label": model_label,
                "status": "FROZEN",
                "panel_counts": {
                    timing: {
                        phase: panels[timing][phase]["seed_count"]
                        for phase in ("development", "confirmation")
                    }
                    for timing in panels
                },
                "capture_start_layers": {
                    timing: banks[timing]["capture_start_layer"] for timing in banks
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
