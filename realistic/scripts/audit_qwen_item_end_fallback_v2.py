#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.spec import resolve_model_spec
from realistic_niah_v5.causal import mechanism_continuations
from realistic_niah_v5.parsing import parse_trace_record
from realistic_niah_v5.pipeline import read_jsonl, registered_records
from realistic_niah_v5.spec import V5Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    spec = resolve_model_spec(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    rows = registered_records(
        read_jsonl(args.generations), V5Config(), model_label=args.model
    )
    rows = [
        row
        for row in rows
        if row.get("split") == "confirmation"
        and parse_trace_record(row)["parser"].get("trace_one_to_one")
    ]

    audit: dict[str, object] = {
        "schema_version": "realistic_niah_v5_qwen_boundary_fallback_audit_v2",
        "model_label": args.model,
        "evaluation_split": "confirmation",
        "cohort": "one_to_one",
        "request_count": len(rows),
        "policies": {},
    }
    for policy in ("strict_registered", "item_end_fallback_v2"):
        eligible_counter: Counter[str] = Counter()
        excluded_counter: Counter[str] = Counter()
        exclusion_scope_counter: Counter[str] = Counter()
        fallback_counter: Counter[str] = Counter()
        request_ids_with_fallback: set[str] = set()
        for row in rows:
            for mechanism in ("targeted_retrieval", "progress_transition"):
                eligible, excluded = mechanism_continuations(
                    row,
                    tokenizer,
                    mechanism=mechanism,
                    boundary_policy=policy,
                )
                for result in eligible:
                    key = f"{mechanism}:{result['transition_phase']}"
                    eligible_counter[key] += 1
                    if result["target_site_fallback"]:
                        fallback_counter[key] += 1
                        request_ids_with_fallback.add(
                            str(row.get("request_id", row.get("stimulus_id")))
                        )
                for result in excluded:
                    key = (
                        f"{mechanism}:{result['transition_phase']}:"
                        f"{result['status']}"
                    )
                    excluded_counter[key] += 1
                    scope = (
                        "target_boundary_candidates"
                        if result.get("target_candidate_audit")
                        else "query_boundary"
                    )
                    exclusion_scope_counter[
                        f"{mechanism}:{result['transition_phase']}:{scope}"
                    ] += 1
        audit["policies"][policy] = {
            "eligible_phase_rows": dict(sorted(eligible_counter.items())),
            "excluded_phase_rows": dict(sorted(excluded_counter.items())),
            "exclusion_scope_rows": dict(
                sorted(exclusion_scope_counter.items())
            ),
            "item_end_fallback_phase_rows": dict(
                sorted(fallback_counter.items())
            ),
            "requests_using_item_end_fallback": len(request_ids_with_fallback),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
