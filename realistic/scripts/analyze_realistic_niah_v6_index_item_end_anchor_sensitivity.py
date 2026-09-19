#!/usr/bin/env python3
"""Analyze the frozen V6 index post-marker/item-end 2x2 sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

try:
    from build_realistic_niah_v6_index_item_end_sensitivity_panel import (
        _validate_generation_container,
    )
except ModuleNotFoundError:  # package import used by the regression tests
    from scripts.build_realistic_niah_v6_index_item_end_sensitivity_panel import (
        _validate_generation_container,
    )


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA = "realistic_niah_v6_index_item_end_anchor_sensitivity_v1"
CONTRACT_STATUS = "FROZEN_EXPLORATORY_BEFORE_NEW_ARM_OUTCOMES"
CELL_SPECS = {
    "p2bank_at_p2": ("post_marker", "post_marker", True),
    "p2bank_at_p0": ("post_marker", "p0_item_end", False),
    "p0bank_at_p2": ("p0_item_end", "post_marker", False),
    "p0bank_at_p0": ("p0_item_end", "p0_item_end", True),
}
CONTRASTS = {
    "overall_item_end_minus_primary": ("p0bank_at_p0", "p2bank_at_p2"),
    "site_effect_for_p2_bank": ("p2bank_at_p0", "p2bank_at_p2"),
    "site_effect_for_p0_bank": ("p0bank_at_p0", "p0bank_at_p2"),
    "bank_effect_at_p0": ("p0bank_at_p0", "p2bank_at_p0"),
    "bank_effect_at_p2": ("p0bank_at_p2", "p2bank_at_p2"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Non-object JSONL row in {path}")
                rows.append(value)
    return rows


def read_trials(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.rglob("trial_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No behavior shards under {root}")
    rows = [row for path in paths for row in read_jsonl(path)]
    trial_ids = [str(row.get("trial_id", "")) for row in rows]
    if not all(trial_ids) or len(trial_ids) != len(set(trial_ids)):
        raise ValueError(f"Behavior trials are missing or duplicate under {root}")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError(f"Non-ok behavior trial under {root}")
    if any(row.get("trial_complete") is not True for row in rows):
        raise ValueError(f"Incomplete behavior trial under {root}")
    return rows


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty vector")
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_summary(
    values: list[float], *, samples: int, seed: int
) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty seed vector")
    rng = random.Random(seed)
    n = len(values)
    boot = sorted(
        mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)
    )
    return {
        "estimate": mean(values),
        "ci95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "n_analysis_slot_seeds": n,
        "bootstrap_samples": samples,
        "bootstrap_random_seed": seed,
    }


def load_registry(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    result = {str(row["request_id"]): int(row["analysis_slot_seed"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate request id in {path}")
    return result


def failure(row: Mapping[str, Any]) -> float:
    direct = row.get("correct_next_needle")
    if direct is not None:
        return float(not bool(direct))
    return float(str(row.get("behavior_outcome")) != "correct_next_needle")


def analyze_cell(
    *,
    name: str,
    root: Path,
    registry: Mapping[str, int],
    random_condition: str,
    expected_slots: set[int],
) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    bank_role, site_role, has_clean = CELL_SPECS[name]
    rows = read_trials(root)
    allowed = {"selected_bank", random_condition}
    if has_clean:
        allowed.add("clean")
    conditions = {str(row["condition"]) for row in rows}
    if conditions != allowed:
        raise ValueError(f"{name} conditions {conditions} != {allowed}")

    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        request_id = str(row["request_id"])
        if request_id not in registry:
            raise ValueError(f"{name} request is absent from its target-site registry")
        slot = int(registry[request_id])
        condition = str(row["condition"])
        grouped[slot][condition].append(failure(row))
        if condition != "clean":
            if int(row.get("head_ablation_decode_steps_requested", 0)) != -1:
                raise ValueError(f"{name} did not retain persistent decode ablation")
            if float(row.get("head_ablation_selected_post_zero_max_abs", 1.0)) != 0.0:
                raise ValueError(f"{name} did not exactly zero selected head slices")
            roles = {str(value) for value in row.get("intervention_anchor_roles", ())}
            if site_role not in roles:
                raise ValueError(f"{name} was not intervened at {site_role}")
            if str(row.get("head_selection_anchor_role")) != bank_role:
                raise ValueError(f"{name} did not use its frozen bank anchor")

    if set(grouped) != expected_slots:
        raise ValueError(f"{name} changed the frozen analysis-slot set")
    slot_rows: dict[int, dict[str, float]] = {}
    for slot in sorted(grouped):
        selected = grouped[slot].get("selected_bank", [])
        random_rows = grouped[slot].get(random_condition, [])
        clean = grouped[slot].get("clean", [])
        if len(selected) != 1 or len(random_rows) != 3:
            raise ValueError(
                f"{name} slot {slot} expected 1 selected and 3 random trials"
            )
        if has_clean and len(clean) != 1:
            raise ValueError(f"{name} slot {slot} expected one clean trial")
        if not has_clean and clean:
            raise ValueError(f"{name} slot {slot} unexpectedly reran clean")
        selected_failure = selected[0]
        random_failure = mean(random_rows)
        slot_rows[slot] = {
            "selected_failure": selected_failure,
            "random_failure_mean": random_failure,
            "selected_minus_random_failure": selected_failure - random_failure,
            **({"clean_failure": clean[0]} if clean else {}),
        }

    effects = [slot_rows[slot]["selected_minus_random_failure"] for slot in sorted(slot_rows)]
    return (
        {
            "cell": name,
            "bank_selection_anchor_role": bank_role,
            "intervention_start_anchor_role": site_role,
            "behavior_root": str(root.resolve()),
            "behavior_manifest_sha256": sha256_file(root / "manifest.json"),
            "selected_failure_rate": mean(
                slot_rows[slot]["selected_failure"] for slot in sorted(slot_rows)
            ),
            "registered_random_failure_rate": mean(
                slot_rows[slot]["random_failure_mean"] for slot in sorted(slot_rows)
            ),
            "clean_failure_rate": (
                mean(slot_rows[slot]["clean_failure"] for slot in sorted(slot_rows))
                if has_clean
                else None
            ),
        },
        slot_rows,
    )


def plan_heads(path: Path) -> dict[str, set[tuple[int, int]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty plan: {path}")

    # Full-panel plan files contain the frozen selected bank plus registered
    # random-control repeats for the same crossfit fold.  Bank overlap is a
    # property of the selected banks only; treating the control rows as
    # additional banks both changes the estimand and creates false duplicate
    # fold failures.
    selected_rows = [row for row in rows if row.get("condition") == "selected_bank"]
    if not selected_rows:
        raise ValueError(f"Plan has no selected_bank rows: {path}")

    result: dict[str, set[tuple[int, int]]] = {}
    for index, row in enumerate(selected_rows):
        fold = str(row.get("fold", index))
        heads = {
            (int(value[0]), int(value[1])) for value in json.loads(row["heads"])
        }
        if fold in result or not heads:
            raise ValueError(f"Invalid fold/head membership in {path}")
        result[fold] = heads
    return result


def bank_overlap(p0_plan: Path, p2_plan: Path) -> dict[str, Any]:
    p0 = plan_heads(p0_plan)
    p2 = plan_heads(p2_plan)
    if set(p0) != set(p2):
        raise ValueError("P0/P2 plans changed the crossfit fold registry")
    folds = []
    for fold in sorted(p0):
        intersection = len(p0[fold] & p2[fold])
        union = len(p0[fold] | p2[fold])
        folds.append(
            {
                "fold": fold,
                "p0_head_count": len(p0[fold]),
                "p2_head_count": len(p2[fold]),
                "intersection": intersection,
                "jaccard": intersection / union,
            }
        )
    return {
        "by_fold": folds,
        "mean_intersection": mean(value["intersection"] for value in folds),
        "mean_jaccard": mean(value["jaccard"] for value in folds),
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"Refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True
    )
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cohort-registry", type=Path, required=True)
    parser.add_argument("--generation-container-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("Unexpected sensitivity contract schema")
    if contract.get("status") != CONTRACT_STATUS:
        raise ValueError("Sensitivity contract is not frozen")
    if contract.get("confirmation_authorized") is not False:
        raise ValueError("Sensitivity analysis cannot open confirmation")
    model_contract = contract["models"][args.model]
    k = int(model_contract["fixed_k"])
    random_condition = str(model_contract["random_condition"])
    expected_slots = {int(value) for value in contract["analysis_slot_seeds"]}
    samples = int(contract["fixed_design"]["bootstrap_samples"])
    bootstrap_seed = int(contract["fixed_design"]["bootstrap_random_seed"])

    primary_plan = args.primary_root / "plans" / f"k{k}" / "retrieval_anchor_bank_plan.csv"
    primary_panel = args.primary_root / "final_transition_panel" / "behavior_anchor_registry.jsonl"
    primary_selection = args.primary_root / "analysis" / "selection.json"
    frozen_hashes = {
        primary_plan: model_contract["frozen_primary_plan_sha256"],
        primary_panel: model_contract["frozen_primary_panel_sha256"],
        primary_selection: model_contract["frozen_primary_selection_sha256"],
        args.cohort_registry: model_contract["frozen_cohort_registry_sha256"],
    }
    for path, expected in frozen_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen dependency changed: {path}")
    amendment_path = args.generation_container_amendment
    if amendment_path is None and args.model == "Gemma4-E4B":
        registered_amendment = (
            ROOT
            / "configs"
            / "realistic_niah_v6_index_item_end_generation_container_amendment1.json"
        )
        if registered_amendment.is_file():
            amendment_path = registered_amendment
    generation_container_audit = _validate_generation_container(
        generations_path=args.generations.resolve(),
        cohort_registry_path=args.cohort_registry.resolve(),
        model=args.model,
        model_contract=model_contract,
        amendment_path=amendment_path,
    )

    p0_panel = args.sensitivity_root / "panels" / "p0_item_end"
    p0_registry_path = p0_panel / "behavior_anchor_registry.jsonl"
    p0_panel_manifest = json.loads((p0_panel / "manifest.json").read_text(encoding="utf-8"))
    if p0_panel_manifest.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("P0 panel was not built under this frozen contract")
    registries = {
        "post_marker": load_registry(primary_panel),
        "p0_item_end": load_registry(p0_registry_path),
    }
    if set(registries["post_marker"].values()) != expected_slots:
        raise ValueError("Primary panel changed the analysis-slot set")
    if set(registries["p0_item_end"].values()) != expected_slots:
        raise ValueError("Item-end panel changed the analysis-slot set")

    roots = {
        "p2bank_at_p2": args.primary_root / "behavior" / f"k{k}",
        "p2bank_at_p0": args.sensitivity_root / "behavior" / "p2bank_at_p0",
        "p0bank_at_p2": args.sensitivity_root / "behavior" / "p0bank_at_p2",
        "p0bank_at_p0": args.sensitivity_root / "behavior" / "p0bank_at_p0",
    }
    cell_summaries: dict[str, dict[str, Any]] = {}
    cell_slots: dict[str, dict[int, dict[str, float]]] = {}
    for index, name in enumerate(CELL_SPECS):
        site_role = CELL_SPECS[name][1]
        summary, slots = analyze_cell(
            name=name,
            root=roots[name],
            registry=registries[site_role],
            random_condition=random_condition,
            expected_slots=expected_slots,
        )
        vector = [slots[slot]["selected_minus_random_failure"] for slot in sorted(slots)]
        summary["selected_minus_random_failure"] = bootstrap_summary(
            vector, samples=samples, seed=bootstrap_seed + index
        )
        cell_summaries[name] = summary
        cell_slots[name] = slots

    contrast_summaries: dict[str, dict[str, Any]] = {}
    for index, (name, (left, right)) in enumerate(CONTRASTS.items(), start=100):
        values = [
            cell_slots[left][slot]["selected_minus_random_failure"]
            - cell_slots[right][slot]["selected_minus_random_failure"]
            for slot in sorted(expected_slots)
        ]
        contrast_summaries[name] = {
            "left": left,
            "right": right,
            **bootstrap_summary(values, samples=samples, seed=bootstrap_seed + index),
        }

    primary_contrast = contrast_summaries["overall_item_end_minus_primary"]
    if primary_contrast["ci95"][0] > 0:
        decision = "SUPPORTS_ANCHOR_SENSITIVITY"
    elif primary_contrast["estimate"] > 0:
        decision = "DESCRIPTIVE_ITEM_END_IMPROVEMENT_UNCERTAIN"
    else:
        decision = "NO_ITEM_END_IMPROVEMENT"

    p0_plan = (
        args.sensitivity_root
        / "plans"
        / "p0_item_end"
        / f"k{k}"
        / "retrieval_anchor_bank_plan.csv"
    )
    payload = {
        "schema_version": "realistic_niah_v6_index_item_end_anchor_sensitivity_analysis_v1",
        "status": "PASS",
        "scientific_scope": "post_hoc_motivated_prospectively_frozen_sensitivity",
        "model_label": args.model,
        "prompt_mode": "enumeration_index",
        "fixed_k": k,
        "random_condition": random_condition,
        "decision": decision,
        "decision_is_exploratory": True,
        "may_replace_primary_result": False,
        "may_reselect_k": False,
        "confirmation_authorized": False,
        "cells": cell_summaries,
        "contrasts": contrast_summaries,
        "bank_overlap": bank_overlap(p0_plan, primary_plan),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "primary_plan_sha256": sha256_file(primary_plan),
        "p0_plan_sha256": sha256_file(p0_plan),
        "p0_panel_manifest_sha256": sha256_file(p0_panel / "manifest.json"),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "generation_container_audit": generation_container_audit,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "analysis.json", payload)
    write_csv(
        args.output / "cell_effects.csv",
        [
            {
                "cell": name,
                "bank_selection_anchor_role": value["bank_selection_anchor_role"],
                "intervention_start_anchor_role": value[
                    "intervention_start_anchor_role"
                ],
                "selected_failure_rate": value["selected_failure_rate"],
                "registered_random_failure_rate": value[
                    "registered_random_failure_rate"
                ],
                "clean_failure_rate": value["clean_failure_rate"],
                "selected_minus_random_failure": value[
                    "selected_minus_random_failure"
                ]["estimate"],
                "ci95_low": value["selected_minus_random_failure"]["ci95"][0],
                "ci95_high": value["selected_minus_random_failure"]["ci95"][1],
            }
            for name, value in cell_summaries.items()
        ],
    )
    slot_rows = []
    for cell in CELL_SPECS:
        for slot in sorted(expected_slots):
            slot_rows.append(
                {
                    "cell": cell,
                    "analysis_slot_seed": slot,
                    **cell_slots[cell][slot],
                }
            )
    write_csv(args.output / "slot_effects.csv", slot_rows)
    print(json.dumps({"status": "PASS", "decision": decision, "model": args.model}))


if __name__ == "__main__":
    main()
