#!/usr/bin/env python3
"""Build the immutable behavior-job ledger from frozen plans and registries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _control_condition(plan: Path) -> str:
    with plan.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    conditions = {row["condition"] for row in rows}
    if "selected_bank" not in conditions:
        raise AssertionError(f"Plan has no selected bank: {plan}")
    controls = conditions - {"selected_bank"}
    if len(controls) != 1:
        raise AssertionError(f"Plan must have one control family: {plan}: {controls}")
    control = next(iter(controls))
    control_rows = [row for row in rows if row["condition"] == control]
    if len(control_rows) != 3:
        raise AssertionError(f"Plan must have three random repeats: {plan}")
    return control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sidecar-output",
        type=Path,
        help="Also register the frozen layer-profile diagnostic job.",
    )
    parser.add_argument(
        "--supplement-only",
        action="store_true",
        help="Register only P2-ranked grammars; reuse completed P0 grammar cells.",
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    confirmation_seeds = set(map(int, config["causal_confirmation_seeds"]))
    contract = spec["scientific_contract"]
    jobs = []
    for model, model_spec in spec["models"].items():
        model_root = args.run_root / model
        primary_k = int(model_spec["primary_bank_size"])
        for grammar, selection_role in model_spec["grammars"].items():
            if args.supplement_only and grammar not in set(
                model_spec["supplement_execution"]["run_grammars"]
            ):
                continue
            registry = model_root / "registries" / "by_grammar" / f"{grammar}.jsonl"
            if not registry.is_file():
                raise FileNotFoundError(registry)
            registry_rows = [
                json.loads(line)
                for line in registry.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for k_value in model_spec["doses"]:
                k = int(k_value)
                plan = (
                    model_root
                    / "plans"
                    / grammar
                    / selection_role
                    / f"k{k}"
                    / "retrieval_anchor_bank_plan.csv"
                )
                if not plan.is_file():
                    raise FileNotFoundError(plan)
                split = "all" if k == primary_k else "confirmation"
                expected_anchors = (
                    len(registry_rows)
                    if split == "all"
                    else sum(
                        int(row["seed"]) in confirmation_seeds
                        for row in registry_rows
                    )
                )
                output = (
                    model_root
                    / "behaviors"
                    / grammar
                    / f"k{k}"
                    / ("full" if split == "all" else "confirmation")
                )
                jobs.append(
                    {
                        "job_index": len(jobs),
                        "model_label": model,
                        "grammar": grammar,
                        "bank_size": k,
                        "selection_anchor_role": selection_role,
                        "intervention_start_anchor_role": contract[
                            "intervention_start_anchor_role"
                        ],
                        "selection_intervention_site_decoupled": selection_role
                        != contract["intervention_start_anchor_role"],
                        "decode_head_ablation_steps": int(
                            contract["decode_head_ablation_steps"]
                        ),
                        "evaluation_split": split,
                        "expected_anchors": expected_anchors,
                        "execution_status": (
                            "registered" if expected_anchors else "skipped_empty_split"
                        ),
                        "random_condition": _control_condition(plan),
                        "plan": str(plan.resolve()),
                        "registry": str(registry.resolve()),
                        "output": str(output.resolve()),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    sidecar_jobs = []
    if args.sidecar_output is not None:
        sidecar = spec["registered_sidecar"]
        model = str(sidecar["model_label"])
        grammar = str(sidecar["target_grammar_class"])
        k = int(sidecar["bank_size"])
        model_root = args.run_root / model
        diagnostic_root = (
            model_root
            / "diagnostics"
            / f"{grammar}_p0_score_post_marker_layer_k{k}"
        )
        plan = diagnostic_root / "plan" / "retrieval_anchor_bank_plan.csv"
        if not plan.is_file():
            raise FileNotFoundError(plan)
        registry = model_root / "registries" / "by_grammar" / f"{grammar}.jsonl"
        registry_rows = [
            json.loads(line)
            for line in registry.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_anchors = sum(
            int(row["seed"]) in confirmation_seeds for row in registry_rows
        )
        sidecar_jobs.append(
            {
                "job_index": 0,
                "model_label": model,
                "grammar": grammar,
                "bank_size": k,
                "selection_anchor_role": sidecar["selection_anchor_role"],
                "intervention_start_anchor_role": sidecar[
                    "intervention_start_anchor_role"
                ],
                "selection_intervention_site_decoupled": False,
                "decode_head_ablation_steps": int(
                    contract["decode_head_ablation_steps"]
                ),
                "evaluation_split": "confirmation",
                "expected_anchors": expected_anchors,
                "execution_status": (
                    "registered" if expected_anchors else "skipped_empty_split"
                ),
                "random_condition": _control_condition(plan),
                "plan": str(plan.resolve()),
                "registry": str(registry.resolve()),
                "output": str((diagnostic_root / "behavior_confirmation").resolve()),
                "diagnostic": "p0_ranking_constrained_to_p2_layer_profile",
            }
        )
        args.sidecar_output.parent.mkdir(parents=True, exist_ok=True)
        args.sidecar_output.write_text(
            "".join(json.dumps(job, sort_keys=True) + "\n" for job in sidecar_jobs),
            encoding="utf-8",
        )
    print(json.dumps({
        "status": "PASS",
        "jobs": len(jobs),
        "sidecar_jobs": len(sidecar_jobs),
        "models": sorted(spec["models"]),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
