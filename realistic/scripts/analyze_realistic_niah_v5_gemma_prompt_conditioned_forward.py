#!/usr/bin/env python3
"""Summarize the frozen Gemma prompt-conditioned forward transplant panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


PHASE_SIZES = {"discovery": 20, "confirmation": 10}
DONORS = (4, 6, 8)
CONDITIONS = ("receiver_self", "native_donor", "donor_to_receiver")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _float_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": mean(finite) if finite else None,
        "median": median(finite) if finite else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def _rate(values: Iterable[bool]) -> dict[str, float | int]:
    items = [bool(value) for value in values]
    hits = sum(items)
    return {"hits": hits, "n": len(items), "rate": hits / len(items)}


def _condition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    generation_rows = [
        row for row in rows if "greedy_donor_successor_adoption" in row
    ]
    return {
        "seed_count": len(rows),
        "donor_vs_receiver_sum_logodds": _float_summary(
            row["donor_vs_receiver_sum_logodds"] for row in rows
        ),
        "donor_successor_argmax": _rate(
            row["donor_successor_argmax"] for row in rows
        ),
        "receiver_successor_argmax": _rate(
            row["receiver_successor_argmax"] for row in rows
        ),
        "donor_vs_receiver_attention_log_ratio": _float_summary(
            row["donor_vs_receiver_attention_log_ratio"] for row in rows
        ),
        "donor_attention_exceeds_receiver": _rate(
            float(row["donor_successor_attention_mass"])
            > float(row["receiver_successor_attention_mass"])
            for row in rows
        ),
        "greedy_donor_successor_adoption": (
            None
            if not generation_rows
            else _rate(
                row["greedy_donor_successor_adoption"] for row in generation_rows
            )
        ),
        "greedy_receiver_successor_retention": (
            None
            if not generation_rows
            else _rate(
                row["greedy_receiver_successor_retention"] for row in generation_rows
            )
        ),
        "equal_length_complete_item": _rate(
            row["equal_length_complete_item"] for row in rows
        ),
        "effective_patch_width": _float_summary(row["patch_width"] for row in rows),
        "receiver_item_coverage": _float_summary(
            row["receiver_item_coverage"] for row in rows
        ),
        "donor_item_coverage": _float_summary(
            row["donor_item_coverage"] for row in rows
        ),
    }


def _paired_summary(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_pair: dict[
        tuple[int, int, int], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["receiver_occurrence_j"]),
            int(row["donor_occurrence_k"]),
        )
        by_pair[key][str(row["condition"])] = row
    effects: list[dict[str, Any]] = []
    for seed, receiver, donor in sorted(by_pair):
        conditions = by_pair[(seed, receiver, donor)]
        if set(conditions) != set(CONDITIONS):
            raise ValueError(
                f"Seed {seed}, receiver {receiver}, donor {donor} lacks "
                "the registered condition triplet"
            )
        self_row = conditions["receiver_self"]
        patch_row = conditions["donor_to_receiver"]
        effects.append(
            {
                "seed": seed,
                "donor_occurrence": donor,
                "receiver_occurrence": receiver,
                "logodds_gain": float(patch_row["donor_vs_receiver_sum_logodds"])
                - float(self_row["donor_vs_receiver_sum_logodds"]),
                "attention_log_ratio_gain": float(
                    patch_row["donor_vs_receiver_attention_log_ratio"]
                )
                - float(self_row["donor_vs_receiver_attention_log_ratio"]),
                "patch_donor_argmax": bool(patch_row["donor_successor_argmax"]),
                "self_donor_argmax": bool(self_row["donor_successor_argmax"]),
                "patch_greedy_donor_adoption": bool(
                    patch_row["greedy_donor_successor_adoption"]
                ),
                "self_greedy_donor_adoption": bool(
                    self_row["greedy_donor_successor_adoption"]
                ),
                "patch_greedy_receiver_retention": bool(
                    patch_row["greedy_receiver_successor_retention"]
                ),
                "self_greedy_receiver_retention": bool(
                    self_row["greedy_receiver_successor_retention"]
                ),
                "greedy_adoption_gain": int(
                    bool(patch_row["greedy_donor_successor_adoption"])
                )
                - int(bool(self_row["greedy_donor_successor_adoption"])),
                "attention_redirected": (
                    float(patch_row["donor_successor_attention_mass"])
                    > float(patch_row["receiver_successor_attention_mass"])
                    and float(self_row["donor_successor_attention_mass"])
                    <= float(self_row["receiver_successor_attention_mass"])
                ),
            }
        )
    return {
        "pair_count": len(effects),
        "logodds_gain_patch_minus_self": _float_summary(
            row["logodds_gain"] for row in effects
        ),
        "positive_logodds_gain": _rate(row["logodds_gain"] > 0 for row in effects),
        "attention_log_ratio_gain_patch_minus_self": _float_summary(
            row["attention_log_ratio_gain"] for row in effects
        ),
        "positive_attention_gain": _rate(
            row["attention_log_ratio_gain"] > 0 for row in effects
        ),
        "donor_argmax_patch": _rate(row["patch_donor_argmax"] for row in effects),
        "donor_argmax_self": _rate(row["self_donor_argmax"] for row in effects),
        "greedy_donor_adoption_patch": _rate(
            row["patch_greedy_donor_adoption"] for row in effects
        ),
        "greedy_donor_adoption_self": _rate(
            row["self_greedy_donor_adoption"] for row in effects
        ),
        "greedy_receiver_retention_patch": _rate(
            row["patch_greedy_receiver_retention"] for row in effects
        ),
        "greedy_receiver_retention_self": _rate(
            row["self_greedy_receiver_retention"] for row in effects
        ),
        "greedy_adoption_net_gain": mean(
            row["greedy_adoption_gain"] for row in effects
        ),
        "self_no_to_patch_yes": _rate(
            row["patch_greedy_donor_adoption"]
            and not row["self_greedy_donor_adoption"]
            for row in effects
        ),
        "self_yes_to_patch_no": _rate(
            row["self_greedy_donor_adoption"]
            and not row["patch_greedy_donor_adoption"]
            for row in effects
        ),
        "attention_redirected_self_to_patch": _rate(
            row["attention_redirected"] for row in effects
        ),
    }, effects


def analyze(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "gemma_prompt_conditioned_forward_analysis_v1",
        "status": "PASS",
        "claim_scope": "prompt-conditioned no-index auxiliary only",
        "model_label": "Gemma4-E4B",
        "layer": 16,
        "patch_scope": "item_span",
        "direction": "forward_only_k_to_k_plus_one",
        "phases": {},
    }
    for phase, expected_seed_count in PHASE_SIZES.items():
        phase_rows: list[dict[str, Any]] = []
        groups: dict[str, Any] = {}
        phase_effects: list[dict[str, Any]] = []
        for donor in DONORS:
            path = root / phase / f"forward_k{donor}" / "trials.jsonl"
            rows = _read_jsonl(path)
            if len(rows) != expected_seed_count * len(CONDITIONS):
                raise ValueError(f"Unexpected row count in {path}: {len(rows)}")
            if {str(row["cohort_mode"]) for row in rows} != {
                "prompt_conditioned_noindex"
            }:
                raise ValueError(f"Cohort-mode leak in {path}")
            if {int(row["layer"]) for row in rows} != {16}:
                raise ValueError(f"Layer drift in {path}")
            by_condition = {
                condition: [
                    row for row in rows if str(row["condition"]) == condition
                ]
                for condition in CONDITIONS
            }
            if any(len(values) != expected_seed_count for values in by_condition.values()):
                raise ValueError(f"Condition panel is incomplete in {path}")
            paired, effects = _paired_summary(rows)
            groups[f"receiver_{donor - 1}_donor_{donor}"] = {
                "receiver_occurrence": donor - 1,
                "donor_occurrence": donor,
                "expected_successor": donor + 1,
                "conditions": {
                    condition: _condition_summary(values)
                    for condition, values in by_condition.items()
                },
                "paired_patch_minus_self": paired,
            }
            phase_rows.extend(rows)
            phase_effects.extend(effects)

        pooled_pairs, _ = _paired_summary(phase_rows)
        by_seed_effects: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for effect in phase_effects:
            by_seed_effects[int(effect["seed"])].append(effect)
        if any(len(values) != len(DONORS) for values in by_seed_effects.values()):
            raise ValueError(f"{phase} does not contain all donor groups per seed")
        result["phases"][phase] = {
            "expected_seed_count": expected_seed_count,
            "seed_count": len(by_seed_effects),
            "pair_count_across_k": len(phase_effects),
            "groups": groups,
            "pooled_across_k": pooled_pairs,
            "seed_level_robustness": {
                "any_greedy_donor_adoption": _rate(
                    any(row["patch_greedy_donor_adoption"] for row in values)
                    for values in by_seed_effects.values()
                ),
                "all_three_greedy_donor_adoption": _rate(
                    all(row["patch_greedy_donor_adoption"] for row in values)
                    for values in by_seed_effects.values()
                ),
                "all_three_donor_argmax": _rate(
                    all(row["patch_donor_argmax"] for row in values)
                    for values in by_seed_effects.values()
                ),
                "all_three_positive_logodds_gain": _rate(
                    all(row["logodds_gain"] > 0 for row in values)
                    for values in by_seed_effects.values()
                ),
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
