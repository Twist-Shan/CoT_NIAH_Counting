#!/usr/bin/env python3
"""Audit routed targeted-query and terminal-state geometry without outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.count_stream import (  # noqa: E402
    build_answer_source_registry,
    trace_patch_geometry_positions,
)
from realistic_niah_v5.integrated_bridge import (  # noqa: E402
    _final_post_marker_position,
    _post_query_receiver_positions,
)
from realistic_niah_v5.parsing import (  # noqa: E402
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    raw_output_text,
)
from realistic_niah_v5.pipeline import read_jsonl, registered_records  # noqa: E402
from realistic_niah_v5.spec import V5Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["Qwen3-8B", "Gemma4-E4B"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--anchor-registry", type=Path, required=True)
    parser.add_argument(
        "--geometry", choices=["suffix8", "full_span"], default="suffix8"
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    config = V5Config.load(ROOT / "configs" / "realistic_niah_v5.json")
    registry = {str(row["request_id"]): row for row in read_jsonl(args.anchor_registry)}
    rows = []
    for row in registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    ):
        request_id = str(row["request_id"])
        if request_id not in registry:
            continue
        parsed = find_trace_count_sequence(
            raw_output_text(row),
            model_family=infer_model_family(row),
            gold_records=gold_records(row),
        )
        if parsed.trace_one_to_one:
            rows.append(row)
    model, tokenizer, adapter = load_registered_model(
        resolve_model_spec(args.model),
        cache_dir=args.cache_dir,
        device_map="auto",
        torch_dtype="bfloat16",
        attention_backend="sdpa",
    )
    del model, adapter
    audits = []
    for row in rows:
        encoding, source_registry = build_answer_source_registry(row, tokenizer)
        count = int(encoding.count)
        query, site = _final_post_marker_position(
            row,
            gold_count=count,
            targeted_site=registry[str(row["request_id"])],
        )
        receiver: tuple[int, ...] = ()
        try:
            if args.geometry == "full_span":
                terminal = source_registry.trace_items[-1]
                receiver = tuple(range(int(terminal[0]), int(terminal[1])))
            else:
                receiver, _donor, _audit = trace_patch_geometry_positions(
                    source_registry,
                    receiver_occurrence=count,
                    donor_occurrence=count - 1,
                    geometry=str(args.geometry),
                )
            post_query = list(_post_query_receiver_positions(query, receiver))
        except ValueError as exc:
            if "not applicable" not in str(exc).lower():
                raise
            post_query = [position for position in receiver if position > query]
            audits.append(
                {
                    "request_id": row["request_id"],
                    "seed": int(row["seed"]),
                    "gold_count": count,
                    "anchor_equivalence_id": site["anchor_equivalence_id"],
                    "query_position": query,
                    "receiver_positions": list(receiver),
                    "query_before_receiver": (
                        query < min(receiver) if receiver else None
                    ),
                    "query_in_receiver": query in set(receiver),
                    "post_query_receiver_count": len(post_query),
                    "post_query_receiver_positions": post_query,
                    "geometry_not_applicable": True,
                    "exclusion_reason": str(exc),
                }
            )
            continue
        audits.append(
            {
                "request_id": row["request_id"],
                "seed": int(row["seed"]),
                "gold_count": count,
                "anchor_equivalence_id": site["anchor_equivalence_id"],
                "query_position": query,
                "receiver_positions": list(receiver),
                "query_before_receiver": query < min(receiver),
                "query_in_receiver": query in set(receiver),
                "post_query_receiver_count": len(post_query),
                "post_query_receiver_positions": post_query,
            }
        )
    summary = {
        "model_label": args.model,
        "geometry": str(args.geometry),
        "row_count": len(audits),
        "geometry_not_applicable_count": sum(
            bool(value.get("geometry_not_applicable")) for value in audits
        ),
        "strictly_after_count": sum(value["query_before_receiver"] is True for value in audits),
        "overlap_count": sum(value["query_in_receiver"] is True for value in audits),
        "zero_post_query_count": sum(
            value["post_query_receiver_count"] == 0 for value in audits
        ),
        "post_query_widths": {
            str(width): sum(value["post_query_receiver_count"] == width for value in audits)
            for width in sorted(
                {
                    value["post_query_receiver_count"]
                    for value in audits
                    if value["post_query_receiver_count"] is not None
                }
            )
        },
    }
    violations = [
        value for value in audits if value["query_before_receiver"] is False
    ]
    if args.summary_only:
        summary["violation_request_ids"] = [
            str(value["request_id"]) for value in violations
        ]
    else:
        summary["violations"] = violations
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
