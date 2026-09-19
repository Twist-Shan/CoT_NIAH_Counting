#!/usr/bin/env python3
"""Analyze and freeze a report-matched V6 targeted-retrieval dose response.

The independent unit is the seed.  Each anchor first contributes a paired
selected-bank minus mean(registered-random) failure contrast; anchors are then
averaged within seed, and seeds receive equal weight.  The full report-matched
dose grid is run, then each model x enumeration mode freezes the K with the
largest discovery contrast (exact tie: smaller K).  Confirmation cannot
reselect K.  Negative results are retained rather than replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "realistic_niah_v6_targeted_retrieval_selection_v1"
REPORT_CONTRACT_SCHEMA_VERSION = (
    "realistic_niah_v6_targeted_retrieval_report_contract_v1"
)
DEFAULT_REPORT_CONTRACT = (
    ROOT / "configs/realistic_niah_v6_targeted_retrieval_report_contract.json"
)
EXPECTED_REPORT_GRIDS = {
    "Qwen3-8B": (32, 64, 80, 96, 112, 128),
    "Gemma4-E4B": (1, 2, 4, 6, 8),
}
EXPECTED_REPORT_REFERENCE_K = {"Qwen3-8B": 128, "Gemma4-E4B": 6}
RANDOM_CONDITIONS = {"layer_matched_random", "global_random"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def load_report_contract(path: Path) -> dict[str, Any]:
    """Load the post-preflight, outcome-blind targeted subprotocol."""

    source = path.resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Targeted-retrieval report contract is not one object")
    expected_header = {
        "schema_version": REPORT_CONTRACT_SCHEMA_VERSION,
        "status": "FROZEN_OUTCOME_BLIND_PROTOCOL_CORRECTION",
        "scope": "targeted_retrieval_discovery_and_confirmation_only",
    }
    for key, expected in expected_header.items():
        if value.get(key) != expected:
            raise ValueError(
                f"Targeted report contract {key} changed: {value.get(key)!r}"
            )
    if set(value.get("applies_to_prompt_modes", ())) != {
        "enumeration_index",
        "enumeration_bullet",
    }:
        raise ValueError("Targeted report contract prompt-mode scope changed")
    models = value.get("models")
    if not isinstance(models, dict) or set(models) != set(EXPECTED_REPORT_GRIDS):
        raise ValueError("Targeted report contract model registry changed")
    for model, expected_grid in EXPECTED_REPORT_GRIDS.items():
        entry = models.get(model)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing targeted report contract for {model}")
        grid = tuple(int(item) for item in entry.get("bank_grid", ()))
        if grid != expected_grid:
            raise ValueError(f"{model} report-matched bank grid changed: {grid}")
        keys = {str(item) for item in grid}
        matching = entry.get("control_matching_by_k")
        conditions = entry.get("random_condition_by_k")
        if not isinstance(matching, dict) or set(matching) != keys:
            raise ValueError(f"{model} control-matching map is incomplete")
        if not isinstance(conditions, dict) or set(conditions) != keys:
            raise ValueError(f"{model} random-condition map is incomplete")
        for bank_size in grid:
            match = str(matching[str(bank_size)])
            condition = str(conditions[str(bank_size)])
            expected_condition = (
                "layer_matched_random" if match == "layer_matched" else "global_random"
            )
            if match not in {"layer_matched", "global"}:
                raise ValueError(f"{model} K={bank_size} has invalid control matching")
            if condition != expected_condition:
                raise ValueError(
                    f"{model} K={bank_size} control matching/condition disagree"
                )
        if int(entry.get("report_reference_k", -1)) != int(
            EXPECTED_REPORT_REFERENCE_K[model]
        ):
            raise ValueError(f"{model} report-reference K changed")
    correction = value.get("protocol_correction")
    if not isinstance(correction, dict):
        raise ValueError("Targeted report contract lacks correction provenance")
    required_false = (
        "trigger_is_sample_failure",
        "targeted_behavior_outcomes_observed_before_freeze",
        "confirmation_outcomes_observed_before_freeze",
        "selection_outcomes_used_to_choose_grid_or_control_family",
        "frozen_base_config_files_mutated",
    )
    if any(correction.get(key) is not False for key in required_false):
        raise ValueError("Targeted protocol correction is not outcome-blind/hash-stable")
    selection = value.get("downstream_bank_selection")
    expected_selection = {
        "selection_split": "v6_discovery",
        "primary_estimand": (
            "seed_equal_selected_minus_mean_of_three_registered_random_next_city_failure"
        ),
        "rule": "maximize_primary_estimand_exact_tie_smaller_k",
        "selected_independently_by_prompt_mode_and_model": True,
        "report_reference_k_is_audit_reference_not_forced_choice": True,
        "confirmation_may_not_reselect_k": True,
        "negative_discovery_result_is_retained": True,
    }
    if not isinstance(selection, dict) or any(
        selection.get(key) != expected
        for key, expected in expected_selection.items()
    ):
        raise ValueError("Targeted downstream bank-selection rule changed")
    return {**value, "_path": str(source), "_sha256": _sha256(source)}


def model_report_contract(
    contract: Mapping[str, Any], model: str
) -> dict[str, Any]:
    entry = contract["models"][model]
    return {
        "bank_grid": tuple(int(value) for value in entry["bank_grid"]),
        "control_matching_by_k": {
            int(key): str(value)
            for key, value in entry["control_matching_by_k"].items()
        },
        "random_condition_by_k": {
            int(key): str(value)
            for key, value in entry["random_condition_by_k"].items()
        },
        "report_reference_k": int(entry["report_reference_k"]),
    }


def expected_conditions(random_condition: str, *, clean: bool = True) -> dict[str, int]:
    if random_condition not in RANDOM_CONDITIONS:
        raise ValueError(f"Unsupported registered random condition: {random_condition}")
    result = {"selected_bank": 1, random_condition: 3}
    return ({"clean": 1, **result} if clean else result)


def _read_shards(path: Path) -> list[dict[str, Any]]:
    shards = sorted((path / "shards").glob("*.jsonl"))
    if not shards:
        raise ValueError(f"No targeted-retrieval shards found at {path}")
    rows: list[dict[str, Any]] = []
    for shard in shards:
        with shard.open("r", encoding="utf-8") as handle:
            values = [json.loads(line) for line in handle if line.strip()]
        if len(values) != 1:
            raise ValueError(f"Expected one row in behavioral shard {shard}")
        value = values[0]
        if value.get("status") != "ok" or value.get("trial_complete") is not True:
            raise ValueError(f"Incomplete behavioral shard {shard}")
        rows.append(value)
    return rows


def _validate_frozen_plan(
    plan: Path,
    audit: Path,
    *,
    model: str,
    bank_size: int,
    random_condition: str,
) -> dict[str, Any]:
    if not plan.is_file() or not audit.is_file():
        raise FileNotFoundError(f"K={bank_size} plan is incomplete")
    with plan.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("model_label")) == model
        ]
    condition_counts = defaultdict(int)
    for row in rows:
        condition_counts[str(row.get("condition"))] += 1
        if int(row["bank_size"]) != bank_size:
            raise ValueError(f"K={bank_size} plan contains another bank size")
        heads = json.loads(str(row["heads"]))
        normalized = [(int(layer), int(head)) for layer, head in heads]
        if len(normalized) != bank_size or len(set(normalized)) != bank_size:
            raise ValueError(f"K={bank_size} plan has an invalid head bank")
        if hashlib.sha256(str(row["heads"]).encode("utf-8")).hexdigest() != str(
            row["bank_sha256"]
        ):
            raise ValueError(f"K={bank_size} plan bank hash mismatch")
    observed = dict(condition_counts)
    expected = expected_conditions(random_condition, clean=False)
    if observed != expected:
        raise ValueError(f"K={bank_size} full-panel plan arms changed: {observed}")
    audit_value = json.loads(audit.read_text(encoding="utf-8"))
    if audit_value.get("full_panel_plan") is not True:
        raise ValueError(f"K={bank_size} is not a frozen full-panel plan")
    if audit_value.get("confirmation_used_for_selection") is not False:
        raise ValueError(f"K={bank_size} plan read confirmation outcomes")
    if int(audit_value.get("registered_bank_size", -1)) != bank_size:
        raise ValueError(f"K={bank_size} plan audit bank size mismatch")
    return {
        "plan": str(plan.resolve()),
        "plan_sha256": _sha256(plan),
        "audit": str(audit.resolve()),
        "audit_sha256": _sha256(audit),
        "condition_counts": expected,
        "random_condition": random_condition,
    }


def _anchor_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["seed"]),
        str(row.get("request_id", "")),
        str(
            row.get(
                "branch_anchor_equivalence_id",
                row.get("anchor_equivalence_id", row.get("query_site_id", "")),
            )
        ),
    )


def _failure(row: Mapping[str, Any]) -> float:
    correct = row.get("correct_next_needle")
    if not isinstance(correct, bool):
        correct = str(row.get("behavior_outcome", "")) == "correct_next_needle"
    return float(not correct)


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_interval(
    seed_effects: Mapping[int, float], *, draws: int, random_seed: int
) -> tuple[float, float]:
    seeds = sorted(seed_effects)
    if not seeds:
        raise ValueError("No seed effects are available for bootstrap")
    rng = random.Random(random_seed)
    values = []
    for _ in range(draws):
        sampled = [seed_effects[rng.choice(seeds)] for _ in seeds]
        values.append(sum(sampled) / len(sampled))
    return _quantile(values, 0.025), _quantile(values, 0.975)


def analyze_dose(
    path: Path,
    *,
    bank_size: int,
    expected_seeds: int,
    bootstrap_samples: int,
    random_seed: int,
    random_condition: str,
    split: str = "discovery",
) -> dict[str, Any]:
    registered_conditions = expected_conditions(random_condition)
    rows = _read_shards(path)
    groups: dict[tuple[int, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if str(row.get("split")) != str(split):
            raise ValueError(
                f"Non-{split} row found in {path}: {row.get('split')!r}"
            )
        observed_k = int(row.get("planned_bank_size", row.get("bank_size", -1)))
        if observed_k != bank_size:
            raise ValueError(f"K mismatch in {path}: expected {bank_size}, got {observed_k}")
        condition = str(row.get("condition"))
        if condition not in registered_conditions:
            raise ValueError(f"Unexpected condition {condition!r} in {path}")
        groups[_anchor_key(row)][condition].append(row)

    seed_anchor_effects: dict[int, list[float]] = defaultdict(list)
    selected_failures: list[float] = []
    random_failures: list[float] = []
    clean_failures: list[float] = []
    for key, conditions in groups.items():
        observed = {name: len(values) for name, values in conditions.items()}
        if observed != registered_conditions:
            raise ValueError(
                f"Incomplete paired arms for K={bank_size}, anchor={key}: {observed}"
            )
        selected = _failure(conditions["selected_bank"][0])
        random_values = [_failure(row) for row in conditions[random_condition]]
        clean = _failure(conditions["clean"][0])
        effect = selected - sum(random_values) / len(random_values)
        seed_anchor_effects[key[0]].append(effect)
        selected_failures.append(selected)
        random_failures.extend(random_values)
        clean_failures.append(clean)

    if len(seed_anchor_effects) != expected_seeds:
        raise ValueError(
            f"K={bank_size} has {len(seed_anchor_effects)} seeds; expected {expected_seeds}"
        )
    seed_effects = {
        seed: sum(values) / len(values) for seed, values in seed_anchor_effects.items()
    }
    effect = sum(seed_effects.values()) / len(seed_effects)
    lower, upper = _bootstrap_interval(
        seed_effects,
        draws=bootstrap_samples,
        random_seed=random_seed + bank_size,
    )
    return {
        "bank_size": bank_size,
        "random_condition": random_condition,
        "split": str(split),
        "seed_count": len(seed_effects),
        "anchor_count": len(groups),
        "condition_rows": len(rows),
        "clean_failure_rate": sum(clean_failures) / len(clean_failures),
        "selected_failure_rate": sum(selected_failures) / len(selected_failures),
        "random_failure_rate": sum(random_failures) / len(random_failures),
        "seed_equal_selected_minus_random_failure": effect,
        "seed_bootstrap_95_lo": lower,
        "seed_bootstrap_95_hi": upper,
        "directional_positive": effect > 0.0,
        "interval_strictly_positive": lower > 0.0,
        "behavior_root": str(path.resolve()),
        "manifest_sha256": _sha256(path / "manifest.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["Qwen3-8B", "Gemma4-E4B"], required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=["enumeration_index", "enumeration_bullet"],
        required=True,
    )
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument(
        "--report-contract", type=Path, default=DEFAULT_REPORT_CONTRACT
    )
    parser.add_argument("--bank-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260828)
    parser.add_argument("--report-reference-k", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    contract = load_report_contract(args.report_contract)
    registered = model_report_contract(contract, args.model)
    sizes = sorted(set(args.bank_sizes))
    if sizes != list(args.bank_sizes):
        raise ValueError("--bank-sizes must be unique and increasing")
    expected_grid = list(registered["bank_grid"])
    if sizes != expected_grid:
        raise ValueError(
            f"{args.model} discovery dose grid must be exactly {expected_grid}"
        )
    if int(args.report_reference_k) != int(registered["report_reference_k"]):
        raise ValueError(
            f"{args.model} report-reference K must be "
            f"{registered['report_reference_k']}"
        )
    dose_rows = [
        analyze_dose(
            args.causal_root / "behavior" / f"k{bank_size}",
            bank_size=bank_size,
            expected_seeds=args.expected_seeds,
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.random_seed,
            random_condition=registered["random_condition_by_k"][bank_size],
        )
        for bank_size in sizes
    ]
    dose_argmax = min(
        dose_rows,
        key=lambda row: (
            -float(row["seed_equal_selected_minus_random_failure"]),
            int(row["bank_size"]),
        ),
    )
    chosen = dose_argmax
    selected_k = int(chosen["bank_size"])
    selected_random_condition = str(
        registered["random_condition_by_k"][selected_k]
    )
    plan_dir = args.causal_root / "plans" / f"k{selected_k}"
    plan = plan_dir / "retrieval_anchor_bank_plan.csv"
    audit = plan_dir / "causal_plan_audit.json"
    selected_plan = _validate_frozen_plan(
        plan,
        audit,
        model=args.model,
        bank_size=selected_k,
        random_condition=selected_random_condition,
    )
    reference_dir = args.causal_root / "plans" / f"k{args.report_reference_k}"
    reference_plan = _validate_frozen_plan(
        reference_dir / "retrieval_anchor_bank_plan.csv",
        reference_dir / "causal_plan_audit.json",
        model=args.model,
        bank_size=int(args.report_reference_k),
        random_condition=registered["random_condition_by_k"][
            int(args.report_reference_k)
        ],
    )

    args.output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dose_rows[0])
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(dose_rows)
    dose_response_path = args.output / "dose_response.csv"
    _atomic_text(dose_response_path, buffer.getvalue())
    dose_response_sha256 = _sha256(dose_response_path)

    selection = {
        "schema_version": SCHEMA_VERSION,
        "status": "DISCOVERY_FROZEN_CHOICE",
        "model_label": args.model,
        "prompt_mode": args.prompt_mode,
        "selection_split": "discovery",
        "selection_rule": (
            "maximize discovery seed-equal selected-bank minus mean of three "
            "registered-random next-city failure contrasts; exact tie -> smaller K"
        ),
        "negative_result_retained": not bool(chosen["directional_positive"]),
        "selected_k": selected_k,
        "selected_random_condition": selected_random_condition,
        "selected_control_matching": registered["control_matching_by_k"][selected_k],
        "selected_effect": float(chosen["seed_equal_selected_minus_random_failure"]),
        "selected_interval": [
            float(chosen["seed_bootstrap_95_lo"]),
            float(chosen["seed_bootstrap_95_hi"]),
        ],
        "report_reference_k": int(args.report_reference_k),
        "report_reference_k_is_audit_reference_not_forced_choice": True,
        "selected_by_v6_discovery_dose_rule": True,
        "selected_matches_report_reference_k": selected_k
        == int(args.report_reference_k),
        "dose_argmax_k": int(dose_argmax["bank_size"]),
        "dose_argmax_effect": float(
            dose_argmax["seed_equal_selected_minus_random_failure"]
        ),
        "dose_argmax_used_for_downstream_bank": True,
        "followup_kernel_bank_policy": (
            "use the V6 discovery-selected K through a hash-validated process-local "
            "contract; retain every dose outcome and forbid confirmation reselection"
        ),
        "bank_grid": sizes,
        "random_condition_by_k": {
            str(key): value
            for key, value in registered["random_condition_by_k"].items()
        },
        "report_contract": str(Path(contract["_path"])),
        "report_contract_sha256": str(contract["_sha256"]),
        "frozen_plan": selected_plan["plan"],
        "frozen_plan_sha256": selected_plan["plan_sha256"],
        "frozen_plan_audit": selected_plan["audit"],
        "frozen_plan_audit_sha256": selected_plan["audit_sha256"],
        "report_reference_plan": reference_plan["plan"],
        "report_reference_plan_sha256": reference_plan["plan_sha256"],
        "report_reference_plan_audit": reference_plan["audit"],
        "report_reference_plan_audit_sha256": reference_plan["audit_sha256"],
        "dose_response": str(dose_response_path.resolve()),
        "dose_response_sha256": dose_response_sha256,
    }
    _atomic_json(args.output / "selection.json", selection)
    _atomic_json(
        args.output / "analysis_audit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "model_label": args.model,
            "prompt_mode": args.prompt_mode,
            "expected_seed_count": args.expected_seeds,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.random_seed,
            "conditions_per_anchor_by_k": {
                str(bank_size): expected_conditions(
                    registered["random_condition_by_k"][bank_size]
                )
                for bank_size in sizes
            },
            "bank_grid": sizes,
            "selected_k": selected_k,
            "selected_by_v6_discovery_dose_rule": True,
            "report_reference_k": int(args.report_reference_k),
            "report_reference_k_is_audit_reference_not_forced_choice": True,
            "dose_argmax_used_for_downstream_bank": True,
            "report_contract": str(Path(contract["_path"])),
            "report_contract_sha256": str(contract["_sha256"]),
            "dose_response": str(dose_response_path.resolve()),
            "dose_response_sha256": dose_response_sha256,
            "dose_rows": dose_rows,
        },
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
