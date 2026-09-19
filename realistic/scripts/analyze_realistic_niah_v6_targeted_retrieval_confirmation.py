#!/usr/bin/env python3
"""Evaluate one discovery-frozen targeted bank on fresh V6 confirmation seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from analyze_realistic_niah_v6_targeted_retrieval import (  # noqa: E402
    DEFAULT_REPORT_CONTRACT,
    analyze_dose,
    load_report_contract,
    model_report_contract,
    _sha256,
)
from realistic_niah_v6.pipeline import sha256_file  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402


SCHEMA_VERSION = "realistic_niah_v6_targeted_retrieval_confirmation_v1"


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("enumeration_index", "enumeration_bullet"),
        required=True,
    )
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--report-contract", type=Path, default=DEFAULT_REPORT_CONTRACT
    )
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = validate_confirmation_freeze(
        args.confirmation_freeze,
        prompt_mode=str(args.prompt_mode),
        model_label=str(args.model),
    )
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    contract = load_report_contract(args.report_contract)
    registered = model_report_contract(contract, args.model)
    required = {
        "status": "DISCOVERY_FROZEN_CHOICE",
        "model_label": str(args.model),
        "prompt_mode": str(args.prompt_mode),
        "selection_split": "discovery",
    }
    for key, expected in required.items():
        if selection.get(key) != expected:
            raise ValueError(
                f"Discovery selection {key} mismatch: "
                f"expected {expected!r}, got {selection.get(key)!r}"
            )
    selected_k = int(selection["selected_k"])
    if selected_k not in registered["bank_grid"]:
        raise ValueError("Discovery selection lies outside the frozen report grid")
    if selection.get("selected_by_v6_discovery_dose_rule") is not True:
        raise ValueError("Discovery K was not selected by the registered dose rule")
    if selection.get("dose_argmax_used_for_downstream_bank") is not True:
        raise ValueError("Discovery dose argmax was not frozen downstream")
    if int(selection.get("dose_argmax_k", -1)) != selected_k:
        raise ValueError("Discovery selected K and registered dose argmax disagree")
    if str(selection.get("report_contract_sha256")) != str(contract["_sha256"]):
        raise ValueError("Targeted report contract changed after discovery")
    dose_response = Path(str(selection.get("dose_response", "")))
    if not dose_response.is_file() or _sha256(dose_response) != str(
        selection.get("dose_response_sha256")
    ):
        raise ValueError("Discovery dose-response table changed after K selection")
    random_condition = str(registered["random_condition_by_k"][selected_k])
    if selection.get("selected_random_condition") != random_condition:
        raise ValueError("Discovery selected random-control family changed")
    frozen_plan = Path(str(selection["frozen_plan"]))
    if not frozen_plan.is_file() or sha256_file(frozen_plan) != str(
        selection["frozen_plan_sha256"]
    ):
        raise ValueError("Discovery-frozen targeted bank plan changed")

    result = analyze_dose(
        args.behavior,
        bank_size=selected_k,
        expected_seeds=int(args.expected_seeds),
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
        random_condition=random_condition,
        split="confirmation",
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "CONFIRMATION_EVALUATED_FROZEN_K",
        "model_label": str(args.model),
        "prompt_mode": str(args.prompt_mode),
        "selected_k": selected_k,
        "selected_random_condition": random_condition,
        "result": result,
        "selection": str(args.selection.resolve()),
        "selection_sha256": _sha256(args.selection),
        "frozen_plan": str(frozen_plan.resolve()),
        "frozen_plan_sha256": sha256_file(frozen_plan),
        "confirmation_freeze": str(args.confirmation_freeze.resolve()),
        "confirmation_freeze_sha256": sha256_file(args.confirmation_freeze),
        "freeze_contract_sha256": str(freeze["freeze_sha256"]),
        "confirmation_used_for_selection": False,
        "bank_size_reselected": False,
        "selected_by_v6_discovery_dose_rule": True,
        "confirmation_reselected_k": False,
        "report_contract": str(Path(contract["_path"])),
        "report_contract_sha256": str(contract["_sha256"]),
        "discovery_dose_response": str(dose_response.resolve()),
        "discovery_dose_response_sha256": _sha256(dose_response),
        "negative_result_retained": True,
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
