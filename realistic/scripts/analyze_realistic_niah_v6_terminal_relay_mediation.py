#!/usr/bin/env python3
"""Run the Native relay estimator and audit V6 source/slot provenance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_realistic_niah_v5_terminal_relay_mediation as native_analyzer  # noqa: E402
from realistic_niah_v6.answer_trace_extension import (  # noqa: E402
    load_contract,
    load_relay_geometry_amendment,
    model_contract,
    sha256_file,
)
from realistic_niah_v6.spec import V6Config  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _geometry_reason(geometry: str) -> str:
    return (
        "not applicable: a trace item is shorter than the requested "
        f"{geometry} geometry"
    )


def _not_estimable_gate(rule: str, *, reason: str) -> dict[str, Any]:
    return {
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "seed_count": 0,
        "pass": False,
        "estimable": False,
        "not_estimable_reason": reason,
        "rule": rule,
    }


def _all_geometry_not_applicable_artifacts(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    expected_seed_count: int,
    geometry: str = "suffix8",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Represent a fully registered but zero-support relay assay honestly."""

    reason = _geometry_reason(geometry)
    if not rows or {str(row.get("status")) for row in rows} != {
        "not_applicable"
    }:
        raise ValueError("All-N/A relay audit received an estimable or missing row")
    if {str(row.get("exclusion_reason")) for row in rows} != {reason}:
        raise ValueError("All-N/A relay audit received another exclusion reason")
    expected_split = "development" if phase == "discovery" else "confirmation"
    if {str(row.get("mechanism_split")) for row in rows} != {expected_split}:
        raise ValueError("All-N/A relay audit received the wrong split")

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_sha256"]), []).append(row)
    if not by_pair or any(len(values) != 6 for values in by_pair.values()):
        raise ValueError("An all-N/A relay pair must contain all six factorial cells")
    pair_to_seed: dict[str, int] = {}
    for pair, values in by_pair.items():
        seeds = {int(row["seed"]) for row in values}
        if len(seeds) != 1:
            raise ValueError("An all-N/A relay pair crosses true source seeds")
        pair_to_seed[pair] = next(iter(seeds))
    planned_seeds = sorted(set(pair_to_seed.values()))
    if len(planned_seeds) != int(expected_seed_count):
        raise ValueError("All-N/A relay audit lost a preregistered seed")
    pairs_per_seed = {
        seed: sum(value == seed for value in pair_to_seed.values())
        for seed in planned_seeds
    }

    gates = {
        "terminal_state_patch_effect": _not_estimable_gate(
            "natural terminal donor patch damage CI low > 0", reason=reason
        ),
        "post_terminal_suffix_specific_mediation": _not_estimable_gate(
            "patch-by-clean-suffix-reset interaction CI low > 0", reason=reason
        ),
        "post_terminal_suffix_residual_equivalence": {
            **_not_estimable_gate("residual ratio CI high < 0.20", reason=reason),
            "relative_equivalence_bound": native_analyzer.RELATIVE_EQUIVALENCE_BOUND,
        },
        "self_reset_is_nondamaging": {
            **_not_estimable_gate(
                "self-reset damage ratio CI high < 0.20", reason=reason
            ),
            "relative_equivalence_bound": native_analyzer.RELATIVE_EQUIVALENCE_BOUND,
        },
        "answer_query_only_mediation": _not_estimable_gate(
            "secondary: query-only reset interaction CI low > 0", reason=reason
        ),
    }
    claim_gates = {
        "phase": phase,
        "confirmatory": phase == "confirmation",
        "estimable": False,
        "not_estimable_reason": reason,
        "primary_gate_ids": [
            "terminal_state_patch_effect",
            "post_terminal_suffix_specific_mediation",
            "post_terminal_suffix_residual_equivalence",
            "self_reset_is_nondamaging",
        ],
        "residual_relay_pass": False,
        "supplementary_greedy_gate_ids": [],
        "greedy_exact_count_support_pass": None,
        "gates": gates,
        "allowed_claim_if_confirmation_passes": (
            "A terminal trace count state propagates through the post-terminal "
            "residual suffix before determining the answer count."
        ),
        "restriction": (
            "No numerical relay claim is estimable because every preregistered "
            f"trace item is shorter than {geometry}; this is not a gate failure."
        ),
    }
    audit = {
        "status": "PASS_EXECUTION_NOT_ESTIMABLE_GEOMETRY",
        "phase": phase,
        "estimable": False,
        "not_estimable_reason": reason,
        "trial_rows": 0,
        "planned_trial_rows": len(rows),
        "planned_pair_count": len(by_pair),
        "eligible_pair_count": 0,
        "geometry_not_applicable_pair_count": len(by_pair),
        "geometry": geometry,
        "geometry_not_applicable_reason": reason,
        "planned_seed_count": len(planned_seeds),
        "seed_count": 0,
        "geometry_not_applicable_full_seed_count": len(planned_seeds),
        "geometry_not_applicable_full_seeds": planned_seeds,
        "planned_pairs_per_seed_min": min(pairs_per_seed.values()),
        "planned_pairs_per_seed_max": max(pairs_per_seed.values()),
        "source_patch_stops_before_relay": True,
        "selection_rank_used": False,
        "residual_relay_pass": False,
    }
    return claim_gates, audit


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _native_estimator_artifacts(
    rows: list[dict[str, Any]],
    *,
    output: Path,
    phase: str,
    geometry: str,
    source_layer: int,
    relay_layer: int,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the unchanged Native numerical functions with dynamic geometry audit."""

    reason = _geometry_reason(geometry)
    excluded = [row for row in rows if str(row.get("status")) == "not_applicable"]
    if excluded and {str(row.get("exclusion_reason")) for row in excluded} != {
        reason
    }:
        raise ValueError("Relay analysis found an unregistered exclusion reason")
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_sha256"]), []).append(row)
    if any(len(values) != 6 for values in by_pair.values()):
        raise ValueError("A relay pair must contain all six factorial cells")
    if any(len({str(row["status"]) for row in values}) != 1 for values in by_pair.values()):
        raise ValueError("Relay analysis found a mixed pair status")

    planned_seeds = sorted({int(row["seed"]) for row in rows})
    eligible_rows = [row for row in rows if str(row.get("status")) == "ok"]
    trials = pd.DataFrame(eligible_rows)
    if trials.empty:
        raise ValueError("Relay analysis has no geometry-eligible seed")
    expected_split = "development" if phase == "discovery" else "confirmation"
    if set(trials["mechanism_split"].astype(str)) != {expected_split}:
        raise ValueError("Relay analysis received the wrong split")
    if "selection_rank" in trials.columns:
        raise ValueError("Formal relay trials must not use selection_rank")
    if set(trials["source_layer"].astype(int)) != {int(source_layer)}:
        raise ValueError("Unexpected terminal source layer")
    if set(trials["relay_layer"].astype(int)) != {int(relay_layer)}:
        raise ValueError("Unexpected frozen relay layer")

    effects = native_analyzer.relay_pair_effects(trials)
    claims = native_analyzer.relay_claim_gates(
        effects,
        phase=phase,
        bootstrap_samples=int(bootstrap_samples),
        random_seed=int(random_seed),
    )
    _atomic_csv(output / "pair_effects.csv", effects)
    _atomic_json(output / "claim_gates.json", claims)
    eligible_seeds = sorted({int(seed) for seed in trials["seed"]})
    full_na_seeds = sorted(set(planned_seeds) - set(eligible_seeds))
    per_seed = effects.groupby("seed")["pair_sha256"].nunique()
    native_audit = {
        "status": "PASS",
        "phase": phase,
        "estimable": True,
        "not_estimable_reason": None,
        "geometry": geometry,
        "trial_rows": int(len(trials)),
        "planned_trial_rows": len(rows),
        "planned_pair_count": len(by_pair),
        "eligible_pair_count": int(effects["pair_sha256"].nunique()),
        "geometry_not_applicable_pair_count": sum(
            1
            for values in by_pair.values()
            if str(values[0].get("status")) == "not_applicable"
        ),
        "geometry_not_applicable_reason": reason,
        "planned_seed_count": len(planned_seeds),
        "seed_count": len(eligible_seeds),
        "geometry_not_applicable_full_seed_count": len(full_na_seeds),
        "geometry_not_applicable_full_seeds": full_na_seeds,
        "pairs_per_seed_min": int(per_seed.min()),
        "pairs_per_seed_max": int(per_seed.max()),
        "source_patch_stops_before_relay": True,
        "selection_rank_used": False,
        "residual_relay_pass": bool(claims["residual_relay_pass"]),
    }
    _atomic_json(output / "audit.json", native_audit)
    return claims, native_audit


def _read_shards(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in sorted((path / "shards").glob("*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("No V6 terminal relay shards found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--relay-geometry-amendment", type=Path)
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument("--cohort-registry", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    args = parser.parse_args()

    config = V6Config.load(args.config)
    contract = load_contract(args.extension_contract)
    relay_geometry_amendment = (
        load_relay_geometry_amendment(
            args.relay_geometry_amendment,
            extension_contract_path=args.extension_contract,
        )
        if args.relay_geometry_amendment is not None
        else None
    )
    frozen = model_contract(
        contract,
        prompt_mode=config.prompt_mode,
        model_label=args.model,
        relay_geometry_amendment=relay_geometry_amendment,
    )
    validate_confirmation_freeze(
        args.confirmation_freeze,
        prompt_mode=config.prompt_mode,
        model_label=args.model,
    )
    rows = _read_shards(args.trials)
    if {str(row.get("prompt_mode")) for row in rows} != {config.prompt_mode}:
        raise ValueError("Terminal relay trials lost their V6 prompt mode")
    if {str(row.get("patch_geometry")) for row in rows} != {
        frozen["relay_geometry"]
    }:
        raise ValueError("Terminal relay trials used another patch geometry")
    expected_amendment_sha = (
        sha256_file(args.relay_geometry_amendment)
        if args.relay_geometry_amendment is not None
        else None
    )
    if expected_amendment_sha is not None and {
        row.get("relay_geometry_amendment_sha256") for row in rows
    } != {expected_amendment_sha}:
        raise ValueError("Terminal relay trial amendment hash changed")
    geometry_preflight: dict[str, Any] | None = None
    if args.relay_geometry_amendment is not None:
        run_manifest_path = args.trials / "manifest.json"
        geometry_preflight_path = (
            args.trials / "terminal_relay_geometry_eligibility_audit.json"
        )
        if not run_manifest_path.is_file() or not geometry_preflight_path.is_file():
            raise ValueError("Task-adapted relay lost its outcome-blind geometry audit")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        geometry_preflight = json.loads(
            geometry_preflight_path.read_text(encoding="utf-8")
        )
        if run_manifest.get("geometry_eligibility_audit_sha256") != sha256_file(
            geometry_preflight_path
        ):
            raise ValueError("Task-adapted relay geometry audit hash changed")
        if geometry_preflight.get("status") != (
            "PASS_OUTCOME_BLIND_GEOMETRY_AUDIT"
        ):
            raise ValueError("Task-adapted relay geometry audit did not pass")
        if (
            geometry_preflight.get("patch_geometry") != frozen["relay_geometry"]
            or geometry_preflight.get("relay_geometry_amendment_sha256")
            != expected_amendment_sha
            or geometry_preflight.get("intervention_loop_started_before_audit")
            is not False
            or geometry_preflight.get("intervention_outcomes_used_for_geometry_choice")
            is not False
            or geometry_preflight.get("per_pair_geometry_adaptation") is not False
        ):
            raise ValueError("Task-adapted relay geometry audit crossed its firewall")
    if any(bool(row.get("seed_aliasing", True)) for row in rows):
        raise ValueError("Terminal relay trials alias true source seeds")
    slot_to_sources: dict[int, set[int]] = {}
    for row in rows:
        slot_to_sources.setdefault(int(row["v6_analysis_slot_seed"]), set()).add(
            int(row["v6_source_seed"])
        )
    if set(slot_to_sources) != {int(v) for v in config.confirmation_seeds}:
        raise ValueError("Terminal relay trials lost a frozen confirmation slot")
    if any(len(values) != 1 for values in slot_to_sources.values()):
        raise ValueError("Terminal relay slot/source mapping is not coherent")
    sources = [next(iter(slot_to_sources[slot])) for slot in sorted(slot_to_sources)]
    if len(sources) != len(set(sources)):
        raise ValueError("Two terminal relay slots share one true source seed")
    statuses = {str(row.get("status")) for row in rows}
    if not statuses <= {"ok", "not_applicable"}:
        raise ValueError("Terminal relay trials contain an invalid status")
    status_by_pair: dict[str, set[str]] = {}
    for row in rows:
        status_by_pair.setdefault(str(row["pair_sha256"]), set()).add(
            str(row["status"])
        )
    if any(len(values) != 1 for values in status_by_pair.values()):
        raise ValueError("Terminal relay pair mixes eligible and N/A cells")
    if geometry_preflight is not None:
        preflight_planned = int(geometry_preflight.get("planned_pair_count", -1))
        preflight_eligible = int(geometry_preflight.get("eligible_pair_count", -1))
        observed_eligible = sum(values == {"ok"} for values in status_by_pair.values())
        if preflight_planned != len(status_by_pair) or preflight_eligible != observed_eligible:
            raise ValueError(
                "Task-adapted relay interventions disagree with geometry preflight"
            )

    if statuses == {"not_applicable"}:
        gates, native_audit = _all_geometry_not_applicable_artifacts(
            rows,
            phase="confirmation",
            expected_seed_count=len(config.confirmation_seeds),
            geometry=frozen["relay_geometry"],
        )
        _atomic_json(args.output / "claim_gates.json", gates)
        _atomic_json(args.output / "audit.json", native_audit)
        _atomic_text(
            args.output / "pair_effects.csv",
            "model_label,seed,request_id,gold_count,mechanism_split,pair_sha256,"
            "donor_offset,source_layer,relay_layer\n",
        )
    else:
        gates, native_audit = _native_estimator_artifacts(
            rows,
            output=args.output,
            phase="confirmation",
            geometry=frozen["relay_geometry"],
            source_layer=frozen["relay_source_layer"],
            relay_layer=frozen["relay_layer"],
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.random_seed,
        )
    gates_path = args.output / "claim_gates.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    if int(native_audit.get("planned_seed_count", -1)) != len(
        config.confirmation_seeds
    ):
        raise ValueError("Terminal relay native audit lost a planned V6 seed")
    eligible_seed_count = int(native_audit.get("seed_count", -1))
    relay_estimable = bool(native_audit.get("estimable", eligible_seed_count > 0))
    if relay_estimable:
        if not 0 < eligible_seed_count <= len(config.confirmation_seeds):
            raise ValueError(
                "Terminal relay native audit has invalid eligible seed count"
            )
    elif (
        eligible_seed_count != 0
        or int(native_audit.get("geometry_not_applicable_full_seed_count", -1))
        != len(config.confirmation_seeds)
        or gates.get("estimable") is not False
    ):
        raise ValueError("Non-estimable relay audit has inconsistent zero support")
    gate_values = gates["gates"]
    primary_ids = (
        "terminal_state_patch_effect",
        "post_terminal_suffix_specific_mediation",
    )
    primary_pass = relay_estimable and all(
        bool(gate_values[name]["pass"]) for name in primary_ids
    )
    audit = {
        "schema_version": "realistic_niah_v6_terminal_relay_analysis_v1",
        "status": "PASS_EXECUTION_COMPLETE",
        "scientific_result": (
            "POSITIVE"
            if primary_pass
            else "NEGATIVE"
            if relay_estimable
            else "NOT_ESTIMABLE_GEOMETRY"
        ),
        "prompt_mode": config.prompt_mode,
        "model_label": args.model,
        "protocol_relation": contract["protocol_relation"],
        "relay_geometry": frozen["relay_geometry"],
        "relay_original_geometry": frozen["relay_original_geometry"],
        "relay_scientific_label": frozen["relay_scientific_label"],
        "native_numerical_estimator": (
            "scripts/analyze_realistic_niah_v5_terminal_relay_mediation.py; "
            "numerical estimands unchanged, with planned-versus-geometry-eligible "
            "seed accounting"
        ),
        "primary_gate_ids": list(primary_ids),
        "partial_mediation_primary_pass": primary_pass,
        "answer_query_only_secondary_pass": relay_estimable
        and bool(gate_values["answer_query_only_mediation"]["pass"]),
        "relay_estimable": relay_estimable,
        "not_estimable_reason": native_audit.get("not_estimable_reason"),
        "complete_mediation_not_claimed": True,
        "analysis_slot_to_true_source_seed": {
            str(slot): next(iter(values))
            for slot, values in sorted(slot_to_sources.items())
        },
        "seed_aliasing": False,
        "bootstrap_unit": "true source seed; one-to-one with frozen analysis slot",
        "planned_seed_count": int(native_audit["planned_seed_count"]),
        "eligible_seed_count": eligible_seed_count,
        "geometry_not_applicable_full_seed_count": int(
            native_audit.get("geometry_not_applicable_full_seed_count", 0)
        ),
        "geometry_not_applicable_full_seeds": list(
            native_audit.get("geometry_not_applicable_full_seeds", [])
        ),
        "intervention_outcomes_used_for_selection": False,
        "suffix4_intervention_outcomes_used_for_selection": False,
        "v6_config_sha256": sha256_file(args.config),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "relay_geometry_amendment_sha256": expected_amendment_sha,
        "geometry_eligibility_audit_sha256": (
            sha256_file(
                args.trials / "terminal_relay_geometry_eligibility_audit.json"
            )
            if geometry_preflight is not None
            else None
        ),
        "confirmation_freeze_sha256": sha256_file(args.confirmation_freeze),
        "cohort_registry_sha256": sha256_file(args.cohort_registry),
        "claim_gates_sha256": sha256_file(gates_path),
    }
    _atomic_json(args.output / "v6_extension_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
