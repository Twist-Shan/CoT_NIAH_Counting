#!/usr/bin/env python3
"""Freeze V6 answer-query donor pairs before reading patch outcomes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v6.answer_trace_extension import (  # noqa: E402
    PAIR_SCHEMA,
    coherent_slot_to_source,
    load_contract,
    model_contract,
    select_low_mid_high_edges,
    sha256_file,
    validate_pair_registry,
)
from realistic_niah_v6.parsing import parse_trace_record  # noqa: E402
from realistic_niah_v6.pipeline import (  # noqa: E402
    read_jsonl,
    validate_generation_contracts,
    write_jsonl,
)
from realistic_niah_v6.replacement import resolved_generation_records  # noqa: E402
from realistic_niah_v6.spec import V6Config  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402


PLAN_SCHEMA = "realistic_niah_v6_answer_query_layer_sweep_plan_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_plan(
    *,
    config_path: Path,
    extension_contract_path: Path,
    confirmation_freeze_path: Path,
    generations_path: Path,
    cohort_registry_path: Path,
    model_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    config = V6Config.load(config_path)
    contract = load_contract(extension_contract_path)
    frozen = model_contract(
        contract, prompt_mode=config.prompt_mode, model_label=model_label
    )
    validate_confirmation_freeze(
        confirmation_freeze_path,
        prompt_mode=config.prompt_mode,
        model_label=model_label,
    )
    source_rows = read_jsonl(generations_path)
    validate_generation_contracts(
        source_rows,
        config,
        model_label=model_label,
        config_sha256=sha256_file(config_path),
    )
    rows = resolved_generation_records(
        source_rows,
        config,
        registry_path=cohort_registry_path,
        model_label=model_label,
    )
    if {str(row["split"]) for row in rows} != {"confirmation"}:
        raise ValueError("Answer-query extension requires confirmation rows only")
    slots = [int(value) for value in config.confirmation_seeds]
    source_by_slot = coherent_slot_to_source(rows, expected_slots=slots)

    by_slot: dict[int, dict[int, tuple[dict[str, Any], dict[str, Any]]]] = {}
    excluded_not_exact: list[str] = []
    excluded_not_strict: list[str] = []
    for row in rows:
        parsed = parse_trace_record(row)
        request_id = str(row["request_id"])
        if not bool(parsed.get("strict_causal_eligible")):
            excluded_not_strict.append(request_id)
            continue
        if not bool(parsed.get("exact_count")):
            excluded_not_exact.append(request_id)
            continue
        slot = int(row.get("v6_analysis_slot_seed", row["seed"]))
        count = int(parsed["gold_count"])
        if count in by_slot.setdefault(slot, {}):
            raise ValueError(f"Duplicate clean-correct slot/count: {slot}/N{count}")
        by_slot[slot][count] = (dict(row), parsed)

    config_hash = sha256_file(config_path)
    contract_hash = sha256_file(extension_contract_path)
    freeze_hash = sha256_file(confirmation_freeze_path)
    registry_hash = sha256_file(cohort_registry_path)
    generation_hash = sha256_file(generations_path)
    pairs: list[dict[str, Any]] = []
    slots_without_edge: list[int] = []
    for slot in slots:
        by_count = by_slot.get(slot, {})
        selected_edges = select_low_mid_high_edges(sorted(by_count), cap=3)
        if not selected_edges:
            slots_without_edge.append(slot)
            continue
        for lower_count, higher_count in selected_edges:
            for receiver_count, donor_count, donor_role in (
                (lower_count, higher_count, "same_slot_adjacent_available_higher"),
                (higher_count, lower_count, "same_slot_adjacent_available_lower"),
            ):
                receiver_row, receiver_parsed = by_count[receiver_count]
                donor_row, donor_parsed = by_count[donor_count]
                receiver_source = int(receiver_row["seed"])
                donor_source = int(donor_row["seed"])
                expected_source = int(source_by_slot[slot])
                if receiver_source != expected_source or donor_source != expected_source:
                    raise ValueError("Answer-query pair crosses true source trajectories")
                pairs.append(
                    {
                        "schema_version": PAIR_SCHEMA,
                        "pair_id": (
                            f"{config.prompt_mode}__{model_label}__slot{slot}__"
                            f"R{receiver_count}__D{donor_count}"
                        ),
                        "prompt_mode": config.prompt_mode,
                        "model_label": model_label,
                        "seed": expected_source,
                        "v6_source_seed": expected_source,
                        "v6_analysis_slot_seed": slot,
                        "seed_aliasing": False,
                        "split": "confirmation",
                        "receiver_request_id": str(receiver_row["request_id"]),
                        "donor_request_id": str(donor_row["request_id"]),
                        "receiver_count": receiver_count,
                        "donor_count": donor_count,
                        "receiver_site_id": frozen["answer_site_id"],
                        "donor_site_id": frozen["answer_site_id"],
                        "layers": list(frozen["answer_layers"]),
                        "donor_role": donor_role,
                        "pair_direction": (
                            "higher_to_lower"
                            if donor_count > receiver_count
                            else "lower_to_higher"
                        ),
                        "receiver_exact_count": bool(receiver_parsed["exact_count"]),
                        "donor_exact_count": bool(donor_parsed["exact_count"]),
                        "pair_eligibility": (
                            "strict_one_to_one_and_receiver_and_donor_"
                            "baseline_final_answer_exact"
                        ),
                        "pair_selection_uses_patch_outcome": False,
                        "v6_config_sha256": config_hash,
                        "extension_contract_sha256": contract_hash,
                        "confirmation_freeze_sha256": freeze_hash,
                        "cohort_registry_sha256": registry_hash,
                        "generations_sha256": generation_hash,
                    }
                )
    registry_audit = validate_pair_registry(
        pairs,
        prompt_mode=config.prompt_mode,
        model_label=model_label,
        expected_layers=frozen["answer_layers"],
        expected_slots=slots,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "pairs.jsonl"
    write_jsonl(pair_path, pairs)
    exact_counts_by_slot = {
        str(slot): sorted(by_slot.get(slot, {})) for slot in slots
    }
    audit = {
        "schema_version": PLAN_SCHEMA,
        "status": "PASS_FROZEN_BEFORE_PATCH_OUTCOMES",
        "prompt_mode": config.prompt_mode,
        "model_label": model_label,
        "protocol_relation": contract["protocol_relation"],
        "confirmation_slots": slots,
        "slot_to_true_source_seed": {
            str(slot): int(source) for slot, source in source_by_slot.items()
        },
        "seed_aliasing": False,
        "site_id": frozen["answer_site_id"],
        "layers": list(frozen["answer_layers"]),
        "conditions": list(frozen["answer_conditions"]),
        "pair_selection": contract["answer_query_full_state_patching"][
            "pair_selection"
        ],
        "pair_selection_uses_patch_outcome": False,
        "eligible_exact_counts_by_slot": exact_counts_by_slot,
        "slots_without_eligible_edge": slots_without_edge,
        "excluded_not_strict_rows": len(excluded_not_strict),
        "excluded_not_exact_rows": len(excluded_not_exact),
        "registered_pairs": len(pairs),
        "registry_audit": registry_audit,
        "config": str(config_path.resolve()),
        "v6_config_sha256": config_hash,
        "extension_contract": str(extension_contract_path.resolve()),
        "extension_contract_sha256": contract_hash,
        "confirmation_freeze": str(confirmation_freeze_path.resolve()),
        "confirmation_freeze_sha256": freeze_hash,
        "cohort_registry": str(cohort_registry_path.resolve()),
        "cohort_registry_sha256": registry_hash,
        "generations": str(generations_path.resolve()),
        "generations_sha256": generation_hash,
        "pairs": str(pair_path.resolve()),
        "pairs_sha256": sha256_file(pair_path),
    }
    _atomic_json(output_dir / "plan_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cohort-registry", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(
                config_path=args.config,
                extension_contract_path=args.extension_contract,
                confirmation_freeze_path=args.confirmation_freeze,
                generations_path=args.generations,
                cohort_registry_path=args.cohort_registry,
                model_label=args.model,
                output_dir=args.output_dir,
            ),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
