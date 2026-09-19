#!/usr/bin/env python3
"""Assemble sealed Native-thinking terminal-chain evidence for the report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _estimand(
    claims: dict[str, Any], estimand: str, outcome: str
) -> dict[str, Any]:
    matches = [
        row
        for row in claims["all_estimands"]
        if row["estimand"] == estimand and row["outcome"] == outcome
    ]
    if len(matches) != 1:
        raise ValueError(f"Missing unique estimand {estimand}/{outcome}")
    return matches[0]


def _geometry_summary(
    discovery: dict[str, Any], confirmation: dict[str, Any]
) -> dict[str, Any]:
    if discovery["seed_count"] != 20 or confirmation["seed_count"] != 10:
        raise ValueError("Native terminal-chain seed contract changed")
    if discovery["selection_rank_used"] or confirmation["selection_rank_used"]:
        raise ValueError("Native terminal-chain used selection_rank")
    if not discovery["outcome_blind"] or not confirmation["outcome_blind"]:
        raise ValueError("Native terminal-chain is not outcome blind")
    if discovery["state_patch_geometry"] != confirmation["state_patch_geometry"]:
        raise ValueError("Native terminal-chain mixed geometries")

    def phase_payload(claims: dict[str, Any]) -> dict[str, Any]:
        return {
            "registered_probability_utility_gate_pass": bool(
                claims["generated_suffix_state_bridge_pass"]
            ),
            "clean_replay_exact": _estimand(
                claims, "clean_reference_suffix_exact", "exact_suffix"
            ),
            "targeted_terminal_nonmarker_damage": _estimand(
                claims, "targeted_terminal_nonmarker_damage", "token_accuracy"
            ),
            "targeted_answer_damage_margin": _estimand(
                claims, "targeted_answer_damage", "correct_count_margin"
            ),
            "clean_state_restoration_margin": _estimand(
                claims, "selected_clean_state_restoration", "correct_count_margin"
            ),
            "restoration_specificity_margin": _estimand(
                claims, "restoration_specificity", "correct_count_margin"
            ),
            "selected_state_occlusion_margin": _estimand(
                claims, "selected_state_occlusion", "correct_count_margin"
            ),
            "count_strata": claims["count_strata"],
        }

    return {
        "state_patch_geometry": discovery["state_patch_geometry"],
        "discovery": phase_payload(discovery),
        "confirmation": phase_payload(confirmation),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "qwen_legacy_midcount": args.qwen_legacy_midcount,
        "qwen_terminal_discovery": args.qwen_terminal_discovery,
        "qwen_terminal_confirmation": args.qwen_terminal_confirmation,
        "qwen_generated_discovery": args.qwen_generated_discovery,
        "qwen_generated_confirmation": args.qwen_generated_confirmation,
        "qwen_prefix_discovery": args.qwen_prefix_discovery,
        "qwen_prefix_confirmation": args.qwen_prefix_confirmation,
        "qwen_high_discovery": args.qwen_high_discovery,
        "qwen_high_confirmation": args.qwen_high_confirmation,
        "gemma_complete": args.gemma_complete,
    }
    evidence = {name: _read(path) for name, path in paths.items()}
    legacy = evidence["qwen_legacy_midcount"]["confirmation"]["gates"]
    if evidence["qwen_legacy_midcount"]["complete_write_edge_formal_pass"]:
        raise ValueError("Legacy Qwen weak bridge unexpectedly became a full PASS")
    gemma = evidence["gemma_complete"]
    if not gemma["complete_generated_suffix_state_bridge_pass"]:
        raise ValueError("Gemma generated-suffix bridge is not a complete PASS")
    gemma_confirmation = gemma["confirmation"]
    if gemma_confirmation["seed_count"] != 10:
        raise ValueError("Gemma confirmation seed contract changed")

    qwen_geometries = {
        "terminal_span": _geometry_summary(
            evidence["qwen_terminal_discovery"],
            evidence["qwen_terminal_confirmation"],
        ),
        "generated_suffix_span": _geometry_summary(
            evidence["qwen_generated_discovery"],
            evidence["qwen_generated_confirmation"],
        ),
        "terminal_prefix_span": _geometry_summary(
            evidence["qwen_prefix_discovery"],
            evidence["qwen_prefix_confirmation"],
        ),
    }
    for name, row in qwen_geometries.items():
        if row["state_patch_geometry"] != name:
            raise ValueError(f"Qwen geometry label changed for {name}")
    high = _geometry_summary(
        evidence["qwen_high_discovery"], evidence["qwen_high_confirmation"]
    )
    high_counts = {
        int(row["gold_count"]): row for row in high["confirmation"]["count_strata"]
    }
    if set(high_counts) != {9, 10}:
        raise ValueError("Qwen high-count diagnostic is not exactly count 9/10")

    qwen_confirmation = {
        name: row["confirmation"] for name, row in qwen_geometries.items()
    }
    generated_restore = qwen_confirmation["generated_suffix_span"][
        "clean_state_restoration_margin"
    ]["mean_effect"]
    terminal_restore = qwen_confirmation["terminal_span"][
        "clean_state_restoration_margin"
    ]["mean_effect"]
    prefix_restore = qwen_confirmation["terminal_prefix_span"][
        "clean_state_restoration_margin"
    ]["mean_effect"]
    if not (generated_restore > terminal_restore and generated_restore > prefix_restore):
        raise ValueError("Qwen generated-suffix restoration is not the strongest geometry")

    payload = {
        "schema_version": "realistic_niah_v5_native_terminal_chain_evidence_v1",
        "status": "PASS",
        "protocol": {
            "discovery_seed_count": 20,
            "confirmation_seed_count": 10,
            "outcome_blind": True,
            "selection_rank_used": False,
            "qwen_targeted_bank_size": 128,
            "gemma_targeted_bank_size": 6,
        },
        "qwen": {
            "overall_status": "PARTIAL_MARGIN_SUPPORTED_REGISTERED_UTILITY_GATE_FAIL",
            "legacy_teacher_forced_bridge": {
                "targeted_receiver_damage_margin": legacy["targeted_receiver_damage"],
                "clean_state_restoration_margin": legacy[
                    "clean_state_restores_selected_receiver"
                ],
                "restoration_specificity_margin": legacy[
                    "restoration_is_targeted_specific"
                ],
                "complete_bridge_pass": False,
            },
            "balanced_count2_6_geometries": qwen_geometries,
            "count9_10_terminal_span": high,
            "count10_confirmation": high_counts[10],
            "count9_confirmation": high_counts[9],
            "descriptive_best_geometry": "generated_suffix_span",
            "descriptive_best_geometry_selection_is_posthoc": True,
            "allowed_claim": (
                "Qwen Top-128 targeted retrieval changes the generated terminal "
                "suffix; restoring the multi-token post-query suffix state produces "
                "a moderate, matched-control-adjusted repair of count margin. The "
                "registered probability-utility serial-chain gate did not pass, so "
                "this is a partial distributed-state pathway, not a complete or "
                "exclusive terminal-counter claim."
            ),
        },
        "gemma": {
            "overall_status": "CONFIRMED_GENERATED_SUFFIX_SERIAL_BRIDGE",
            "complete_bridge_pass": True,
            "confirmation": {
                "clean_replay_exact": _estimand(
                    gemma_confirmation, "clean_reference_suffix_exact", "exact_suffix"
                ),
                "targeted_terminal_nonmarker_damage": _estimand(
                    gemma_confirmation,
                    "targeted_terminal_nonmarker_damage",
                    "token_accuracy",
                ),
                "targeted_answer_damage_utility": _estimand(
                    gemma_confirmation, "targeted_answer_damage", "expected_count_utility"
                ),
                "clean_state_restoration_utility": _estimand(
                    gemma_confirmation,
                    "selected_clean_state_restoration",
                    "expected_count_utility",
                ),
                "restoration_specificity_utility": _estimand(
                    gemma_confirmation, "restoration_specificity", "expected_count_utility"
                ),
                "selected_state_occlusion_utility": _estimand(
                    gemma_confirmation,
                    "selected_state_occlusion",
                    "expected_count_utility",
                ),
                "targeted_answer_damage_margin": _estimand(
                    gemma_confirmation, "targeted_answer_damage", "correct_count_margin"
                ),
                "clean_state_restoration_margin": _estimand(
                    gemma_confirmation,
                    "selected_clean_state_restoration",
                    "correct_count_margin",
                ),
            },
            "allowed_claim": gemma_confirmation[
                "allowed_claim_if_confirmation_passes"
            ],
        },
        "cross_model_claim": (
            "Both models support targeted retrieval followed by a distributed "
            "trace-state pathway to count readout, but only Gemma satisfies the "
            "complete preregistered generated-suffix serial bridge. Qwen supports "
            "a weaker count-dependent margin pathway."
        ),
        "inputs_sha256": {str(path): _sha256(path) for path in paths.values()},
    }
    return payload


def parse_args() -> argparse.Namespace:
    root = Path("work_remote_snapshots")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-legacy-midcount",
        type=Path,
        default=root / "qwen_write_edge_midcount58_v2_complete.json",
    )
    for flag, filename in (
        ("qwen-terminal-discovery", "qwen_count2_6_terminal_discovery_claim_gates.json"),
        ("qwen-terminal-confirmation", "qwen_count2_6_terminal_confirmation_claim_gates.json"),
        ("qwen-generated-discovery", "qwen_count2_6_generated_suffix_discovery_claim_gates.json"),
        ("qwen-generated-confirmation", "qwen_count2_6_generated_suffix_confirmation_claim_gates.json"),
        ("qwen-prefix-discovery", "qwen_count2_6_terminal_prefix_discovery_claim_gates.json"),
        ("qwen-prefix-confirmation", "qwen_count2_6_terminal_prefix_confirmation_claim_gates.json"),
        ("qwen-high-discovery", "qwen_count9_10_terminal_discovery_claim_gates.json"),
        ("qwen-high-confirmation", "qwen_count9_10_terminal_confirmation_claim_gates.json"),
    ):
        parser.add_argument(f"--{flag}", type=Path, default=root / filename)
    parser.add_argument(
        "--gemma-complete",
        type=Path,
        default=root / "gemma_generated_suffix_state_top6_v2_complete.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "native_terminal_chain_evidence_20d10c_20260821.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
