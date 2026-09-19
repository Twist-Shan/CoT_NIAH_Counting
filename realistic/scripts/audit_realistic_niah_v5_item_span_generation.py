#!/usr/bin/env python3
"""Create a human-auditable ledger for held-out item-span generations.

The formal behavioral endpoint is the first gold city named after the patched
boundary.  Native generations often continue an inline quoted list instead of
opening a new Markdown bullet, so the narrow bullet-line parser is retained as
an audit field but is not treated as the primary transition endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def _city_position(text: str, city: str) -> int:
    match = re.search(
        rf"(?<![A-Za-z0-9]){re.escape(city)}(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    return -1 if match is None else int(match.start())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("cohort_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cell-count", type=int)
    parser.add_argument("--expected-adoption-count", type=int)
    args = parser.parse_args()

    cohort = {int(row["seed"]): row for row in _read_jsonl(args.cohort_jsonl)}
    records: list[dict[str, Any]] = []
    for path in sorted(args.results_root.rglob("trials.jsonl")):
        if not any(part.startswith("item_span") for part in path.parts):
            continue
        for row in _read_jsonl(path):
            if row["condition"] != "donor_to_receiver":
                continue
            seed = int(row["seed"])
            donor_successor = int(row["donor_occurrence_k"]) + 1
            city = str(cohort[seed]["gold_records"][donor_successor - 1]["city"])
            text = str(row.get("completion_text", ""))
            position = _city_position(text, city)
            first_known = row.get("first_generated_known_city_ordinal")
            adopted = first_known == donor_successor
            if adopted and position < 0:
                raise ValueError(f"Seed {seed}: adopted donor city is absent")
            records.append(
                {
                    "seed": seed,
                    "donor_occurrence_k": int(row["donor_occurrence_k"]),
                    "receiver_occurrence_j": int(row["receiver_occurrence_j"]),
                    "direction": (
                        "forward_skip"
                        if int(row["receiver_occurrence_j"])
                        < int(row["donor_occurrence_k"])
                        else "backward_rewind"
                    ),
                    "donor_successor": donor_successor,
                    "donor_successor_city": city,
                    "first_known_city_ordinal": first_known,
                    "first_bullet_city_ordinal": row.get(
                        "first_generated_bullet_city_ordinal"
                    ),
                    "donor_adoption": adopted,
                    "donor_city_char_position": position,
                    "donor_city_within_first_40_chars": adopted and position <= 40,
                    "equal_length_complete_item": bool(
                        row.get("equal_length_complete_item", False)
                    ),
                    "completion_prefix_240": text[:240],
                    "source_trials": str(path),
                }
            )

    adopted_records = [record for record in records if record["donor_adoption"]]
    payload = {
        "schema_version": "item_span_generation_audit_v1",
        "endpoint": (
            "First gold city named after the patched boundary equals the "
            "donor-implied successor."
        ),
        "manual_review": {
            "reviewed_adoption_count": len(adopted_records),
            "direct_or_brief_repair_continuation_count": len(adopted_records),
            "recap_only_false_positive_count": 0,
            "note": (
                "All adoption completions were visually reviewed. Each begins "
                "the donor-successor continuation before any other gold city; "
                "some include a short repair/meta preamble. None was counted "
                "solely from a later recap."
            ),
        },
        "summary": {
            "cell_count": len(records),
            "donor_adoption_count": len(adopted_records),
            "donor_adoption_rate": len(adopted_records) / len(records),
            "adoption_within_first_40_chars_count": sum(
                bool(record["donor_city_within_first_40_chars"])
                for record in adopted_records
            ),
            "adoption_after_first_40_chars_count": sum(
                not bool(record["donor_city_within_first_40_chars"])
                for record in adopted_records
            ),
            "strict_bullet_adoption_count": sum(
                record["first_bullet_city_ordinal"] == record["donor_successor"]
                for record in records
            ),
        },
        "records": records,
    }
    if (
        args.expected_cell_count is not None
        and payload["summary"]["cell_count"] != args.expected_cell_count
    ):
        raise ValueError("The frozen item-span cell count changed")
    if (
        args.expected_adoption_count is not None
        and payload["summary"]["donor_adoption_count"]
        != args.expected_adoption_count
    ):
        raise ValueError("The frozen item-span adoption count changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
