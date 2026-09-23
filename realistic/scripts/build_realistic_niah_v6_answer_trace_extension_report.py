"""Collect audited answer/trace extension results as JSON; no HTML rendering."""

from __future__ import annotations

import argparse

import csv

import datetime as dt

import hashlib

import json

import os

import sys

from pathlib import Path

from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v6.answer_trace_extension import (  # noqa: E402
    load_relay_geometry_amendment,
)

PROMPT_MODES = ("enumeration_index", "enumeration_bullet")

MODE_LABELS = {
    "enumeration_index": "Index enumeration",
    "enumeration_bullet": "Bullet enumeration",
}

MODELS = ("Qwen3-8B", "Gemma4-E4B")

def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)

def _gate(gates: dict[str, Any], name: str) -> dict[str, Any]:
    value = gates["gates"][name]
    estimate = value.get("estimate", value.get("mean", value.get("value")))
    low = value.get("ci95_low", value.get("ci_low"))
    high = value.get("ci95_high", value.get("ci_high"))
    result = {
        "pass": bool(value["pass"]),
        "estimate": estimate,
        "low": low,
        "high": high,
    }
    for field in ("rule", "seed_count", "relative_equivalence_bound", "role"):
        if field in value:
            result[field] = value[field]
    return result

def _relay_support(audit: dict[str, Any], *, geometry: str) -> dict[str, Any]:
    eligible = int(audit.get("eligible_seed_count", -1))
    return {
        "geometry": geometry,
        "estimable": bool(audit.get("relay_estimable", eligible > 0)),
        "planned_seed_count": int(audit.get("planned_seed_count", -1)),
        "eligible_seed_count": eligible,
        "geometry_not_applicable_full_seed_count": int(
            audit.get("geometry_not_applicable_full_seed_count", -1)
        ),
        "geometry_not_applicable_full_seeds": [
            int(seed)
            for seed in audit.get("geometry_not_applicable_full_seeds", [])
        ],
        "not_estimable_reason": audit.get("not_estimable_reason"),
        "scientific_result": audit.get("scientific_result"),
        "partial_mediation_primary_pass": bool(
            audit.get("partial_mediation_primary_pass", False)
        ),
    }

def build(
    run_root: Path,
    output_root: Path,
    *,
    relay_geometry_amendment_path: Path | None = None,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    contract_hashes: set[str] = set()
    extension_contract_path = (
        ROOT / "configs" / "realistic_niah_v6_answer_trace_extension_v1.json"
    )
    relay_geometry_amendment = (
        load_relay_geometry_amendment(
            relay_geometry_amendment_path,
            extension_contract_path=extension_contract_path,
        )
        if relay_geometry_amendment_path is not None
        else None
    )
    relay_geometry_amendment_sha256 = (
        _sha(relay_geometry_amendment_path)
        if relay_geometry_amendment_path is not None
        else None
    )
    for prompt_mode in PROMPT_MODES:
        for model in MODELS:
            root = (
                run_root
                / prompt_mode
                / model
                / "causal"
                / "answer_trace_extension_v1"
            )
            complete_path = root / "extension_complete.json"
            answer_audit_path = (
                root
                / "answer_query_layer_sweep"
                / "analysis"
                / "v6_extension_audit.json"
            )
            layer_path = (
                root
                / "answer_query_layer_sweep"
                / "analysis"
                / "layer_effects.csv"
            )
            original_relay_audit_path = (
                root
                / "terminal_relay_partial_confirmation"
                / "relay_analysis_confirmation"
                / "v6_extension_audit.json"
            )
            original_gates_path = (
                root
                / "terminal_relay_partial_confirmation"
                / "relay_analysis_confirmation"
                / "claim_gates.json"
            )
            use_task_adapted_relay = (
                relay_geometry_amendment is not None
                and prompt_mode == "enumeration_bullet"
            )
            relay_directory = (
                str(relay_geometry_amendment["adaptation"]["output_directory_name"])
                if use_task_adapted_relay
                else "terminal_relay_partial_confirmation"
            )
            relay_audit_path = (
                root
                / relay_directory
                / "relay_analysis_confirmation"
                / "v6_extension_audit.json"
            )
            gates_path = (
                root
                / relay_directory
                / "relay_analysis_confirmation"
                / "claim_gates.json"
            )
            for path in (
                complete_path,
                answer_audit_path,
                layer_path,
                original_relay_audit_path,
                original_gates_path,
                relay_audit_path,
                gates_path,
            ):
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Missing V6 answer/trace artifact: {path}")
            complete = _json(complete_path)
            answer_audit = _json(answer_audit_path)
            original_relay_audit = _json(original_relay_audit_path)
            relay_audit = _json(relay_audit_path)
            gates = _json(gates_path)
            if complete.get("status") != "PASS_EXECUTION_COMPLETE":
                raise ValueError(f"Extension cell is incomplete: {prompt_mode}/{model}")
            contract_hashes.add(str(complete["extension_contract_sha256"]))
            if str(relay_audit.get("extension_contract_sha256")) != str(
                complete["extension_contract_sha256"]
            ):
                raise ValueError(
                    "Terminal relay and answer extension contract hashes disagree: "
                    f"{prompt_mode}/{model}"
                )
            relay_geometry = str(
                relay_audit.get(
                    "relay_geometry", "suffix4" if use_task_adapted_relay else "suffix8"
                )
            )
            expected_geometry = "suffix4" if use_task_adapted_relay else "suffix8"
            if relay_geometry != expected_geometry:
                raise ValueError(
                    "Terminal relay report selected the wrong geometry: "
                    f"{prompt_mode}/{model}"
                )
            if use_task_adapted_relay and relay_audit.get(
                "relay_geometry_amendment_sha256"
            ) != relay_geometry_amendment_sha256:
                raise ValueError(
                    "Task-adapted Bullet relay amendment hash changed: "
                    f"{prompt_mode}/{model}"
                )
            layer_rows = _rows(layer_path)
            if not layer_rows:
                raise ValueError(f"No answer layer rows: {prompt_mode}/{model}")
            terminal = max(layer_rows, key=lambda row: int(row["layer"]))
            onset = answer_audit["native_analysis_audit"][
                "descriptive_onset_layer"
            ].get(model)
            relay_planned_seed_count = int(
                relay_audit.get("planned_seed_count", -1)
            )
            relay_eligible_seed_count = int(
                relay_audit.get("eligible_seed_count", -1)
            )
            relay_estimable = bool(
                relay_audit.get(
                    "relay_estimable", relay_eligible_seed_count > 0
                )
            )
            relay_full_na_seeds = [
                int(seed)
                for seed in relay_audit.get(
                    "geometry_not_applicable_full_seeds", []
                )
            ]
            relay_full_na_count = int(
                relay_audit.get(
                    "geometry_not_applicable_full_seed_count", -1
                )
            )
            if relay_planned_seed_count != 10:
                raise ValueError(
                    "Terminal relay must retain all 10 preregistered seeds: "
                    f"{prompt_mode}/{model}"
                )
            if relay_estimable and not (
                0 < relay_eligible_seed_count <= relay_planned_seed_count
            ):
                raise ValueError(
                    "Terminal relay has an invalid geometry-eligible seed count: "
                    f"{prompt_mode}/{model}"
                )
            if not relay_estimable and relay_eligible_seed_count != 0:
                raise ValueError(
                    "Non-estimable terminal relay has nonzero support: "
                    f"{prompt_mode}/{model}"
                )
            if (
                relay_full_na_count != len(relay_full_na_seeds)
                or len(set(relay_full_na_seeds)) != len(relay_full_na_seeds)
                or relay_eligible_seed_count + relay_full_na_count
                != relay_planned_seed_count
            ):
                raise ValueError(
                    "Terminal relay planned/eligible/full-NA seed accounting is "
                    f"inconsistent: {prompt_mode}/{model}"
                )
            answer_layer_effects = [
                {
                    "layer": int(row["layer"]),
                    "seed_clusters": int(row["seed_clusters"]),
                    "pairs": int(row["pairs"]),
                    "full_donor_adoption": float(row["full_donor_adoption"]),
                    "full_donor_adoption_ci95_low": float(
                        row["full_donor_adoption_ci95_low"]
                    ),
                    "full_donor_adoption_ci95_high": float(
                        row["full_donor_adoption_ci95_high"]
                    ),
                    "adoption_specificity": float(row["adoption_specificity"]),
                    "adoption_specificity_ci95_low": float(
                        row["adoption_specificity_ci95_low"]
                    ),
                    "adoption_specificity_ci95_high": float(
                        row["adoption_specificity_ci95_high"]
                    ),
                    "registered_numeric_valid": float(
                        row["registered_numeric_valid"]
                    ),
                }
                for row in sorted(layer_rows, key=lambda row: int(row["layer"]))
            ]
            relay_gate_ids = (
                "terminal_state_patch_effect",
                "post_terminal_suffix_specific_mediation",
                "post_terminal_suffix_residual_equivalence",
                "self_reset_is_nondamaging",
                "answer_query_only_mediation",
            )
            relay_gates = {
                gate_id: _gate(gates, gate_id) for gate_id in relay_gate_ids
            }
            original_suffix8_support = _relay_support(
                original_relay_audit, geometry="suffix8"
            )
            cells.append(
                {
                    "prompt_mode": prompt_mode,
                    "mode_label": MODE_LABELS[prompt_mode],
                    "model_label": model,
                    "answer_registered_pairs": int(
                        answer_audit["pair_registry_audit"]["registered_pairs"]
                    ),
                    "answer_seed_clusters": int(terminal["seed_clusters"]),
                    "answer_terminal_layer": int(terminal["layer"]),
                    "answer_terminal_adoption": float(
                        terminal["full_donor_adoption"]
                    ),
                    "answer_terminal_ci_low": float(
                        terminal["full_donor_adoption_ci95_low"]
                    ),
                    "answer_terminal_ci_high": float(
                        terminal["full_donor_adoption_ci95_high"]
                    ),
                    "answer_descriptive_onset": onset,
                    "answer_layer_effects": answer_layer_effects,
                    "relay_gates": relay_gates,
                    "relay_geometry": relay_geometry,
                    "relay_original_geometry": "suffix8",
                    "relay_evidence_label": (
                        "post_hoc_task_adapted_bullet_relay_replication"
                        if use_task_adapted_relay
                        else "original_registered_suffix8"
                    ),
                    "relay_geometry_amendment_sha256": (
                        relay_geometry_amendment_sha256
                        if use_task_adapted_relay
                        else None
                    ),
                    "original_suffix8_relay": original_suffix8_support,
                    "terminal_patch": relay_gates["terminal_state_patch_effect"],
                    "suffix_mediation": relay_gates[
                        "post_terminal_suffix_specific_mediation"
                    ],
                    "suffix_residual_ratio": relay_gates[
                        "post_terminal_suffix_residual_equivalence"
                    ],
                    "self_reset_ratio": relay_gates["self_reset_is_nondamaging"],
                    "query_mediation": relay_gates["answer_query_only_mediation"],
                    "relay_planned_seed_count": relay_planned_seed_count,
                    "relay_eligible_seed_count": relay_eligible_seed_count,
                    "relay_estimable": relay_estimable,
                    "relay_not_estimable_reason": relay_audit.get(
                        "not_estimable_reason"
                    ),
                    "relay_geometry_not_applicable_full_seed_count": (
                        relay_full_na_count
                    ),
                    "relay_geometry_not_applicable_full_seeds": (
                        relay_full_na_seeds
                    ),
                    "partial_mediation_pass": bool(
                        relay_audit["partial_mediation_primary_pass"]
                    ),
                    "complete_mediation_not_claimed": True,
                    "seed_aliasing": False,
                    "artifact_hashes": {
                        "completion": _sha(complete_path),
                        "answer_audit": _sha(answer_audit_path),
                        "layer_effects": _sha(layer_path),
                        "relay_audit": _sha(relay_audit_path),
                        "claim_gates": _sha(gates_path),
                    },
                    "original_suffix8_artifact_hashes": {
                        "relay_audit": _sha(original_relay_audit_path),
                        "claim_gates": _sha(original_gates_path),
                    },
                }
            )
    if len(contract_hashes) != 1:
        raise ValueError("Four V6 extension cells do not share one frozen contract")

    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    task_adapted_enabled = relay_geometry_amendment is not None
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "realistic_niah_v6_answer_trace_extension_report_v2",
        "status": "PASS_COMPLETE",
        "generated_utc": generated,
        "extension_contract_sha256": next(iter(contract_hashes)),
        "relay_geometry_amendment_sha256": relay_geometry_amendment_sha256,
        "relay_geometry_policy": (
            "index_suffix8_bullet_suffix4_task_adapted"
            if task_adapted_enabled
            else "all_cells_original_suffix8"
        ),
        "original_suffix8_artifacts_preserved": True,
        "cells": cells,
    }
    _atomic_text(
        output_root / "report_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _atomic_text(output_root / "evidence.COMPLETE", "PASS\n")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--relay-geometry-amendment", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.run_root,
                args.output_root,
                relay_geometry_amendment_path=args.relay_geometry_amendment,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
