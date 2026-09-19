#!/usr/bin/env python3
"""Audit strict V6 cell deficits before loading a reserve-generation model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v6.pipeline import (  # noqa: E402
    read_jsonl,
    sha256_file,
    validate_generation_contracts,
)
from realistic_niah_v6.replacement import (  # noqa: E402
    load_replacement_policy,
    resolve_replacement_panel,
)
from realistic_niah_v6.spec import MODEL_LABELS, V6Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--replacement-policy", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_LABELS, required=True)
    parser.add_argument(
        "--seed-role", choices=("discovery", "confirmation"), required=True
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = V6Config.load(args.v6_config)
    policy = load_replacement_policy(args.replacement_policy, config)
    rows = read_jsonl(args.generations)
    validate_generation_contracts(
        rows,
        config,
        model_label=args.model,
        config_sha256=sha256_file(args.v6_config),
    )
    resolution = resolve_replacement_panel(
        rows,
        config=config,
        model_label=args.model,
        seed_role=args.seed_role,
        policy=policy,
    )
    failures_by_count: dict[str, Counter[str]] = {
        str(count): Counter() for count in config.counts
    }
    for row in resolution["attempt_ledger"]:
        if row["candidate_kind"] != "original" or row["eligible"]:
            continue
        failures_by_count[str(row["gold_count"])].update(row["failure_reasons"])
    result = {
        "schema_version": "realistic_niah_v6_replacement_readiness_audit_v1",
        "status": "PASS_OUTCOME_BLIND_DEFICIT_AUDIT",
        "model_label": args.model,
        "prompt_mode": config.prompt_mode,
        "seed_role": args.seed_role,
        "quota_per_count": resolution["quota_per_count"],
        "currently_complete": resolution["complete"],
        "strict_rows_per_count": resolution["selected_per_count"],
        "shortfalls": resolution["shortfalls"],
        "first_wave_candidate_rows": len(resolution["next_candidates"]),
        "original_failure_reasons_by_count": {
            count: dict(reasons) for count, reasons in failures_by_count.items()
        },
        "selection_inputs": [
            "generation_presence",
            "fresh_v6_parse.strict_causal_eligible",
            "ascending_frozen_amendment_seed_order",
        ],
        "intervention_outcomes_read": False,
        "hidden_states_read": False,
        "attention_scores_read": False,
        "head_ranks_read": False,
        "generations_sha256": sha256_file(args.generations),
        "replacement_policy_sha256": sha256_file(args.replacement_policy),
        "v6_config_sha256": sha256_file(args.v6_config),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
