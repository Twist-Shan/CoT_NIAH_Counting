#!/usr/bin/env python3
"""Audit V6 provenance around the frozen Native answer-query estimator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_v5_answer_query_layer_sweep as native_analyzer  # noqa: E402
from realistic_niah_v6.answer_trace_extension import (  # noqa: E402
    load_contract,
    model_contract,
    sha256_file,
    validate_pair_registry,
)
from realistic_niah_v6.pipeline import read_jsonl  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument("--cohort-registry", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = V6Config.load(args.config)
    contract = load_contract(args.extension_contract)
    frozen = model_contract(
        contract, prompt_mode=config.prompt_mode, model_label=args.model
    )
    validate_confirmation_freeze(
        args.confirmation_freeze,
        prompt_mode=config.prompt_mode,
        model_label=args.model,
    )
    pairs = read_jsonl(args.pairs)
    pair_audit = validate_pair_registry(
        pairs,
        prompt_mode=config.prompt_mode,
        model_label=args.model,
        expected_layers=frozen["answer_layers"],
        expected_slots=config.confirmation_seeds,
    )
    trials = read_jsonl(args.trials)
    if {str(row.get("prompt_mode")) for row in trials} != {config.prompt_mode}:
        raise ValueError("Answer-query trials lost their V6 prompt mode")
    if any(bool(row.get("seed_aliasing", True)) for row in trials):
        raise ValueError("Answer-query trials alias true source seeds")
    slots = sorted({int(row["v6_analysis_slot_seed"]) for row in trials})
    sources = sorted({int(row["v6_source_seed"]) for row in trials})
    if len(slots) != len(sources):
        raise ValueError("Answer-query slot/source clustering is not one-to-one")

    native_audit = native_analyzer.analyze(
        args.trials,
        args.pairs,
        args.output_dir,
        expected_layers=frozen["answer_layers"],
    )
    audit = {
        "schema_version": "realistic_niah_v6_answer_query_layer_sweep_analysis_v1",
        "status": "PASS_COMPLETE",
        "prompt_mode": config.prompt_mode,
        "model_label": args.model,
        "protocol_relation": contract["protocol_relation"],
        "native_numerical_estimator": (
            "scripts/analyze_v5_answer_query_layer_sweep.py unchanged"
        ),
        "native_analysis_audit": native_audit,
        "pair_registry_audit": pair_audit,
        "analysis_slots": slots,
        "true_source_seeds": sources,
        "seed_aliasing": False,
        "bootstrap_unit": "true source seed; one-to-one with frozen analysis slot",
        "intervention_outcomes_used_for_pair_selection": False,
        "v6_config_sha256": sha256_file(args.config),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "confirmation_freeze_sha256": sha256_file(args.confirmation_freeze),
        "cohort_registry_sha256": sha256_file(args.cohort_registry),
        "pairs_sha256": sha256_file(args.pairs),
        "trials_sha256": sha256_file(args.trials),
    }
    _atomic_json(args.output_dir / "v6_extension_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
