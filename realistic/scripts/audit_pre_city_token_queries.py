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
from realistic_niah_v5.parsing import parse_trace_record
from realistic_niah_v5.pipeline import read_jsonl, registered_records
from realistic_niah_v5.pre_city import pre_city_token_queries
from realistic_niah_v5.spec import V5Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        if parse_trace_record(row)["parser"].get("trace_one_to_one")
    ]
    variants: Counter[str] = Counter()
    anchors: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    duplicate_position_groups = 0
    occurrence_rows = 0
    requests_with_queries: set[str] = set()
    split_requests: Counter[str] = Counter()
    for row in rows:
        queries, excluded = pre_city_token_queries(
            row, tokenizer, depths=(1, 2), include_anchor=True
        )
        request_id = str(row.get("request_id", row.get("stimulus_id")))
        if queries:
            requests_with_queries.add(request_id)
            split_requests[str(row.get("split"))] += 1
        variants.update(query.query_variant for query in queries)
        anchors.update(query.anchor_kind for query in queries)
        exclusions.update(str(value.get("status")) for value in excluded)
        by_occurrence: dict[int, list[int]] = {}
        for query in queries:
            by_occurrence.setdefault(query.occurrence, []).append(
                query.query_output_token_count
            )
        occurrence_rows += len(by_occurrence)
        duplicate_position_groups += sum(
            len(values) != len(set(values)) for values in by_occurrence.values()
        )
    payload = {
        "schema_version": "realistic_niah_v5_pre_city_token_query_audit_v1",
        "model_label": args.model,
        "cohort": "one_to_one",
        "input_requests": len(rows),
        "requests_with_queries": len(requests_with_queries),
        "requests_with_queries_by_split": dict(sorted(split_requests.items())),
        "occurrence_rows": occurrence_rows,
        "query_variant_rows": dict(sorted(variants.items())),
        "anchor_kind_rows": dict(sorted(anchors.items())),
        "exclusion_status_rows": dict(sorted(exclusions.items())),
        "occurrences_with_variant_position_aliasing": duplicate_position_groups,
        "depths": [1, 2],
        "anchor_policy": (
            "nearest literal baseline token boundary at-or-left of marker_end; "
            "item_start when no marker is registered"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
