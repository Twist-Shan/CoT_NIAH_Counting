#!/usr/bin/env python3
"""Analyze direct count-logit margins for one model x timing branch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import bootstrap_seed_mean_ci, sign_flip_pvalue  # noqa: E402
from realistic_niah_v5.targeted_counter_logit_margin import (  # noqa: E402
    LOGIT_MARGIN_ENDPOINTS,
)
from realistic_niah_v5.targeted_counter_ncc import NCC_CONDITIONS  # noqa: E402


ENDPOINT_FIELDS = {
    "final_answer_sequence_margin": "correct_count_margin",
    "local_rank_adjacent_sequence_margin": "local_rank_adjacent_sequence_margin",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_path = root / "frozen_row_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"No frozen logit-margin plan under {root}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    files = sorted((root / "shards").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No logit-margin shards under {root}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    return rows, plan


def _audit(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    timing: str,
    phase: str,
) -> None:
    if (
        plan.get("schema_version")
        != "realistic_niah_v5_targeted_counter_logit_margin_plan_v2"
        or str(plan.get("timing_branch")) != timing
        or str(plan.get("seed_role")) != phase
    ):
        raise ValueError("Logit-margin frozen plan belongs to another protocol")
    expected_seeds = {int(value) for value in plan["seeds"]}
    seeds = {int(row["seed"]) for row in rows}
    if seeds != expected_seeds or len(rows) != len(seeds):
        raise ValueError(f"{phase} logit-margin seed contract changed")
    expected_endpoints = LOGIT_MARGIN_ENDPOINTS[timing]
    model_labels = {str(row["model_label"]) for row in rows}
    if model_labels != {str(plan["model_label"])}:
        raise ValueError("Logit-margin model label changed")
    for row in rows:
        if (
            row.get("schema_version")
            != "realistic_niah_v5_targeted_counter_logit_margin_capture_v2"
            or row.get("experiment_id")
            != "targeted_retrieval_query_to_direct_count_logit_margin"
            or str(row.get("timing_branch")) != timing
        ):
            raise ValueError("Logit-margin shard schema or timing changed")
        if tuple(row["endpoint_names"]) != expected_endpoints:
            raise ValueError("Logit-margin endpoint registry changed")
        if (
            not row["answer_query_is_downstream_of_targeted_query"]
            or not row["candidate_answer_tokens_run_without_head_hooks"]
            or not row["outcome_blind_panel"]
            or row["selection_rank_used"]
            or not row["no_decoder_fit_or_layer_selection"]
        ):
            raise ValueError("Logit-margin causal or outcome-blind contract changed")
        conditions = list(row["conditions"])
        if tuple(item["condition"] for item in conditions) != NCC_CONDITIONS:
            raise ValueError("Logit-margin condition order changed")
        if str(conditions[1]["receiver_bank_sha256"]) != str(
            plan["selected_bank_sha256"]
        ):
            raise ValueError("Logit-margin selected bank changed")
        for condition in conditions:
            expected_applications = int(condition["receiver_head_count"]) > 0
            applications = dict(condition["head_ablation_layer_applications"])
            if bool(applications) != expected_applications:
                raise ValueError("Logit-margin hook-application audit changed")
            if any(int(value) != 1 for value in applications.values()):
                raise ValueError("A logit-margin hook did not apply exactly once")
            if float(condition["head_ablation_selected_post_zero_max_abs"]) != 0.0:
                raise ValueError("A logit-margin selected head slice was not zero")
            local_audit = condition.get("local_rank_prefill_hook_audit")
            if timing == "rank_after_city" and local_audit is not None:
                local_applications = dict(
                    local_audit["head_ablation_layer_applications"]
                )
                if bool(local_applications) != expected_applications:
                    raise ValueError("Local marker hook-application audit changed")
                if any(int(value) != 1 for value in local_applications.values()):
                    raise ValueError("A local marker hook did not apply exactly once")
                if (
                    float(local_audit["head_ablation_selected_post_zero_max_abs"])
                    != 0.0
                ):
                    raise ValueError("A local marker selected head slice was not zero")


def _summary(values: Iterable[float], name: str, random_seed: int) -> dict[str, Any]:
    active = np.asarray(list(values), dtype=float)
    if not len(active) or not np.isfinite(active).all():
        raise ValueError(f"Cannot summarize empty/non-finite {name}")
    result = bootstrap_seed_mean_ci(active, samples=10_000, seed=random_seed)
    result.update(
        {
            "estimand": name,
            "p_value_two_sided_sign_flip": sign_flip_pvalue(active),
            "higher_is_supportive": True,
        }
    )
    return result


def _endpoint_rows(
    rows: list[dict[str, Any]], endpoint: str, *, phase: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    field = ENDPOINT_FIELDS[endpoint]
    long_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for row in rows:
        by_condition = {item["condition"]: item for item in row["conditions"]}
        values = {name: by_condition[name].get(field) for name in NCC_CONDITIONS}
        availability = [value is not None for value in values.values()]
        if not any(availability):
            continue
        if not all(availability):
            raise ValueError("Logit-margin endpoint is missing only some conditions")
        margins = {name: float(value) for name, value in values.items()}
        if not all(np.isfinite(value) for value in margins.values()):
            raise ValueError("Logit-margin endpoint contains non-finite values")
        seed = int(row["seed"])
        for condition, margin in margins.items():
            long_rows.append(
                {
                    "phase": phase,
                    "seed": seed,
                    "request_id": str(row["request_id"]),
                    "gold_count": int(row["gold_count"]),
                    "endpoint": endpoint,
                    "condition": condition,
                    "margin": margin,
                }
            )
        clean = margins["clean"]
        selected_loss = clean - margins["selected_mask"]
        random_losses = [
            clean - margins[f"random_mask_r{repeat}"] for repeat in (1, 2, 3)
        ]
        clean_correct = (
            int(by_condition["clean"]["predicted_count_among_candidates"])
            == int(row["gold_count"])
            if endpoint == "final_answer_sequence_margin"
            else clean > 0
        )
        effect_rows.append(
            {
                "phase": phase,
                "seed": seed,
                "request_id": str(row["request_id"]),
                "gold_count": int(row["gold_count"]),
                "endpoint": endpoint,
                "clean_margin": clean,
                "clean_correct": bool(clean_correct),
                "selected_margin": margins["selected_mask"],
                "selected_margin_loss": selected_loss,
                "random_r1_margin_loss": random_losses[0],
                "random_r2_margin_loss": random_losses[1],
                "random_r3_margin_loss": random_losses[2],
                "random_mean_margin_loss": float(np.mean(random_losses)),
                "selected_vs_random_specificity": float(
                    selected_loss - np.mean(random_losses)
                ),
            }
        )
    return pd.DataFrame(long_rows), pd.DataFrame(effect_rows)


def _phase_summary(frame: pd.DataFrame, *, endpoint: str, phase: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"No {phase} rows for endpoint {endpoint}")
    seed_offset = 100 if phase == "development" else 200
    return {
        "phase": phase,
        "seed_count": int(len(frame)),
        "seeds": sorted(int(value) for value in frame["seed"]),
        "clean_mean_margin": float(frame["clean_margin"].mean()),
        "clean_accuracy": float(frame["clean_correct"].astype(float).mean()),
        "selected_margin_loss": _summary(
            frame["selected_margin_loss"],
            "clean_margin_minus_selected_margin",
            seed_offset + 1,
        ),
        "selected_vs_random_specificity": _summary(
            frame["selected_vs_random_specificity"],
            "selected_loss_minus_three_random_mean_loss",
            seed_offset + 2,
        ),
    }


def analyze(
    discovery: list[dict[str, Any]],
    discovery_plan: dict[str, Any],
    confirmation: list[dict[str, Any]],
    confirmation_plan: dict[str, Any],
    *,
    timing: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _audit(discovery, discovery_plan, timing=timing, phase="development")
    _audit(confirmation, confirmation_plan, timing=timing, phase="confirmation")
    all_long: list[pd.DataFrame] = []
    all_effects: list[pd.DataFrame] = []
    endpoint_results: dict[str, Any] = {}
    for endpoint in LOGIT_MARGIN_ENDPOINTS[timing]:
        discovery_long, discovery_effects = _endpoint_rows(
            discovery, endpoint, phase="development"
        )
        confirmation_long, confirmation_effects = _endpoint_rows(
            confirmation, endpoint, phase="confirmation"
        )
        if discovery_effects.empty or confirmation_effects.empty:
            if endpoint == "local_rank_adjacent_sequence_margin":
                continue
            raise ValueError(f"Primary logit-margin endpoint {endpoint} is incomplete")
        all_long.extend((discovery_long, confirmation_long))
        all_effects.extend((discovery_effects, confirmation_effects))
        development = _phase_summary(
            discovery_effects, endpoint=endpoint, phase="development"
        )
        heldout = _phase_summary(
            confirmation_effects, endpoint=endpoint, phase="confirmation"
        )
        chance = 0.1 if endpoint == "final_answer_sequence_margin" else 0.5
        validity = bool(
            heldout["clean_mean_margin"] > 0
            and heldout["clean_accuracy"] > chance
        )
        directional = bool(heldout["selected_margin_loss"]["mean_effect"] > 0)
        specific = bool(
            heldout["selected_vs_random_specificity"]["mean_effect"] > 0
        )
        interval_confirmed = bool(
            heldout["selected_margin_loss"]["ci_low"] > 0
            and heldout["selected_vs_random_specificity"]["ci_low"] > 0
        )
        if not validity:
            status = "UNINTERPRETABLE_MARGIN_SHIFT_READOUT_VALIDITY_FAILURE"
        elif interval_confirmed:
            status = "INTERVAL_CONFIRMED_DIRECTIONAL_SPECIFIC_SUPPORT"
        elif directional and specific:
            status = "VALID_READOUT_DIRECTIONAL_SPECIFIC_EVIDENCE"
        else:
            status = "NO_DIRECTIONAL_SPECIFIC_EVIDENCE"
        endpoint_results[endpoint] = {
            "endpoint": endpoint,
            "is_primary_endpoint": endpoint == "final_answer_sequence_margin",
            "development": development,
            "confirmation": heldout,
            "readout_validity": {
                "pass": validity,
                "chance_accuracy": chance,
                "requires_clean_mean_margin_positive": True,
                "requires_clean_accuracy_strictly_above_chance": True,
            },
            "selected_mask_changes_margin_directionally": directional,
            "selected_mask_more_damaging_than_random": specific,
            "bootstrap_interval_excludes_zero_for_both_gates": interval_confirmed,
            "effect_status": status,
        }

    if "final_answer_sequence_margin" not in endpoint_results:
        raise RuntimeError("Logit-margin analysis lost its primary endpoint")
    long = pd.concat(all_long, ignore_index=True)
    effects = pd.concat(all_effects, ignore_index=True)
    primary = endpoint_results["final_answer_sequence_margin"]
    claim = {
        "schema_version": "realistic_niah_v5_targeted_counter_logit_margin_analysis_v1",
        "status": "PASS",
        "model_label": str(discovery_plan["model_label"]),
        "timing_branch": timing,
        "primary_endpoint": "final_answer_sequence_margin",
        "primary_endpoint_result": primary,
        "endpoint_results": endpoint_results,
        "conditions": list(NCC_CONDITIONS),
        "no_decoder_fit_or_layer_selection": True,
        "raw_ncc_centroids_used": False,
        "candidate_answer_scoring": "full_autoregressive_sequence_log_probability_1_to_10",
        "confirmation_status": "registered_existing_split_after_ncc_inspection",
        "margin_gate_registered_before_logit_outcome_inspection": True,
        "confirmation_used_for_registration": False,
        "outcome_blind_panel": True,
        "selection_rank_used": False,
        "raw_margins_pooled_across_branches": False,
        "restrictions": [
            "The confirmation rows reuse the fixed split after NCC outcomes were inspected; this is not pristine prospective confirmation.",
            "The two timing branches use maximal eligible, unpaired cohorts.",
            "The mask is applied only at the frozen retrieval query; answer candidate tokens run without hooks.",
            "A positive output-margin effect supports persistence to the answer distribution, not exclusive use of one hidden-state code.",
        ],
        "allowed_claim": (
            "The frozen targeted-query bank changed direct count-output margins "
            f"according to the registered gates ({primary['effect_status']})."
        ),
    }
    return long, effects, claim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--timing", choices=tuple(LOGIT_MARGIN_ENDPOINTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    discovery, discovery_plan = _load(args.discovery)
    confirmation, confirmation_plan = _load(args.confirmation)
    long, effects, claim = analyze(
        discovery,
        discovery_plan,
        confirmation,
        confirmation_plan,
        timing=str(args.timing),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output / "condition_margins.csv", index=False)
    effects.to_csv(args.output / "seed_effects.csv", index=False)
    _atomic_json(args.output / "claim_gates.json", claim)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": claim["model_label"],
                "timing": claim["timing_branch"],
                "effect_status": claim["primary_endpoint_result"]["effect_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
