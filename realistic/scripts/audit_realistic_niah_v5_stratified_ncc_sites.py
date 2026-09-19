#!/usr/bin/env python3
"""Tokenizer-only audit of all frozen timing-specific NCC endpoints."""

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

from realistic_niah_v4.modeling import load_registered_tokenizer  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.causal_sites import compile_causal_site_plan  # noqa: E402
from realistic_niah_v5.count_stream import build_answer_source_registry  # noqa: E402
from realistic_niah_v5.integrated_bridge import _final_post_marker_position  # noqa: E402
from realistic_niah_v5.pipeline import read_jsonl, registered_records  # noqa: E402
from realistic_niah_v5.spec import V5Config  # noqa: E402
from realistic_niah_v5.stratified_targeted_counter_ncc import (  # noqa: E402
    grammar_timing,
    stratified_endpoint_positions,
)
from realistic_niah_v5.terminal_token_state import _site_positions  # noqa: E402


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _decode(tokenizer: Any, input_ids: tuple[int, ...], positions: tuple[int, ...]) -> str:
    return tokenizer.decode(
        [int(input_ids[position]) for position in positions],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--v5-config", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--timing", choices=("rank_after_city", "rank_before_city"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = load_registered_tokenizer(
        resolve_model_spec(str(args.model)), cache_dir=args.cache_dir
    )
    rows = registered_records(
        read_jsonl(args.generations),
        V5Config.load(args.v5_config),
        model_label=str(args.model),
    )
    by_id = {str(row["request_id"]): row for row in rows}
    panel = read_jsonl(args.panel)
    audits: list[dict[str, Any]] = []
    matching_event_count = 0
    for targeted_site in panel:
        request_id = str(targeted_site["request_id"])
        row = by_id.get(request_id)
        if row is None:
            raise ValueError(f"No registered generation for {request_id}")
        encoding, registry = build_answer_source_registry(row, tokenizer)
        plan = compile_causal_site_plan(row, tokenizer)
        events = list(plan["events"])
        targeted_query, _specification = _final_post_marker_position(
            row,
            gold_count=int(encoding.count),
            targeted_site=targeted_site,
        )
        event_audits = []
        for occurrence, event in enumerate(events, start=1):
            if grammar_timing(event) != str(args.timing):
                continue
            endpoints = stratified_endpoint_positions(
                registry,
                event,
                occurrence=occurrence,
                timing=str(args.timing),
            )
            matching_event_count += 1
            event_audits.append(
                {
                    "occurrence": occurrence,
                    "grammar_class": str(event["grammar_class"]),
                    "endpoints": {
                        name: {
                            "positions": list(positions),
                            "decoded": _decode(tokenizer, encoding.input_ids, positions),
                        }
                        for name, positions in endpoints.items()
                    },
                }
            )
        if not event_audits or event_audits[-1]["occurrence"] != int(encoding.count):
            raise ValueError("Panel row lacks the required final timing event")
        final_event = events[-1]
        final_sites = final_event["sites"]
        final_endpoints = stratified_endpoint_positions(
            registry,
            final_event,
            occurrence=int(encoding.count),
            timing=str(args.timing),
        )
        if min(position for span in final_endpoints.values() for position in span) <= int(
            targeted_query
        ):
            raise ValueError("Final audited endpoint is not downstream of retrieval")
        site_context = {}
        for role in (
            "city_target_span",
            "pre_marker_state",
            "rank_evidence_core_span",
            "post_update_commit_state",
        ):
            if role not in final_sites:
                continue
            positions = _site_positions(final_sites[role], role=role)
            site_context[role] = {
                "positions": list(positions),
                "decoded": _decode(tokenizer, encoding.input_ids, positions),
            }
        audits.append(
            {
                "request_id": request_id,
                "seed": int(encoding.seed),
                "gold_count": int(encoding.count),
                "targeted_query_position": int(targeted_query),
                "final_grammar_class": str(final_event["grammar_class"]),
                "final_sites": site_context,
                "matching_events": event_audits,
            }
        )

    result = {
        "schema_version": "realistic_niah_v5_stratified_ncc_site_audit_v1",
        "status": "PASS",
        "model_label": str(args.model),
        "timing_branch": str(args.timing),
        "panel_row_count": len(panel),
        "matching_event_count": matching_event_count,
        "all_final_endpoints_downstream_of_targeted_query": True,
        "city_to_rank_marker_tokens_excluded": str(args.timing) == "rank_after_city",
        "rows": audits,
    }
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_label": str(args.model),
                "timing_branch": str(args.timing),
                "panel_row_count": len(panel),
                "matching_event_count": matching_event_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
