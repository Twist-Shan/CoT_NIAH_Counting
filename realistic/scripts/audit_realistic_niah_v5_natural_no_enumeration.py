#!/usr/bin/env python3
"""Audit archived natural traces for count-label-free item states."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_count_stream import _atomic_json, _atomic_jsonl  # noqa: E402
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_no_count_enumeration_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label")) == str(args.model)
    ]
    if len(rows) != 300:
        raise ValueError(f"Expected the frozen 300-row archive, found {len(rows)}")
    audited = []
    eligible = []
    for row in rows:
        audit = audit_no_count_enumeration_trace(row)
        record = {
            "request_id": str(row["request_id"]),
            "model_label": str(row["model_label"]),
            "seed": int(row["seed"]),
            "gold_count": int(row["gold_count"]),
            "marker_kind": str(row["trace_parse"]["parser"]["marker_kind"]),
            "eligible": bool(audit["eligible"]),
            "reasons": list(audit["reasons"]),
        }
        audited.append(record)
        if audit["eligible"]:
            selected = dict(row)
            selected["no_count_enumeration_audit"] = audit
            eligible.append(selected)

    marker_counts = Counter(record["marker_kind"] for record in audited)
    eligible_marker_counts = Counter(
        record["marker_kind"] for record in audited if record["eligible"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output / "eligible_generations.jsonl", eligible)
    _atomic_jsonl(args.output / "row_audit.jsonl", audited)
    _atomic_json(
        args.output / "audit.json",
        {
            "schema_version": "realistic_niah_v5_natural_no_enumeration_audit_v1",
            "status": "PASS",
            "model_label": str(args.model),
            "archive_row_count": len(rows),
            "eligible_row_count": len(eligible),
            "eligible_seed_count": len({int(row["seed"]) for row in eligible}),
            "eligible_discovery_seed_count": len(
                {int(row["seed"]) for row in eligible if 1234 <= int(row["seed"]) <= 1253}
            ),
            "eligible_confirmation_seed_count": len(
                {int(row["seed"]) for row in eligible if 1254 <= int(row["seed"]) <= 1263}
            ),
            "marker_counts": dict(sorted(marker_counts.items())),
            "eligible_marker_counts": dict(sorted(eligible_marker_counts.items())),
            "plain_bullets_allowed": True,
            "future_text_after_item_not_used_for_exclusion": True,
            "outcome_fields_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
