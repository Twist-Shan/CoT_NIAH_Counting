#!/usr/bin/env python3
"""Read-only health diagnostic for an incomplete targeted-count run.

This script is deliberately not a formal analyzer.  It reports only anchors for
which all five frozen arms (clean, selected, and three random controls) already
exist, and it never writes a claim gate or an analysis artifact.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


TOTAL_RE = re.compile(r"(?i)(?:^|\b)total\s*:\s*([0-9]+)\b")
EXPECTED_ARMS = Counter(
    {"clean": 1, "selected_bank": 1, "layer_matched_random": 3}
)


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in sorted((path / "shards").glob("*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed JSONL at {shard}:{line_number}: {error}"
                    ) from error
    return rows


def diagnose(path: Path) -> dict[str, Any]:
    rows = _rows(path)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["request_id"],
            int(row["seed"]),
            int(row["gold_count"]),
            int(row["from_occurrence"]),
            int(row["to_occurrence"]),
        )
        groups[key].append(row)

    complete: list[dict[str, Any]] = []
    for group in groups.values():
        if Counter(str(row["condition"]) for row in group) == EXPECTED_ARMS:
            complete.extend(group)

    stats: dict[str, Any] = {}
    for condition in ("clean", "selected_bank", "layer_matched_random"):
        arm = [row for row in complete if row["condition"] == condition]
        parsed: list[int | None] = []
        for row in arm:
            matches = TOTAL_RE.findall(str(row.get("completion_text", "")))
            parsed.append(int(matches[-1]) if matches else None)
        denominator = len(arm)
        stats[condition] = {
            "rows": denominator,
            "parsed_rate": (
                sum(value is not None for value in parsed) / denominator
                if denominator
                else None
            ),
            "count_accuracy": (
                sum(
                    value == int(row["gold_count"])
                    for value, row in zip(parsed, arm, strict=True)
                )
                / denominator
                if denominator
                else None
            ),
            "next_city_accuracy": (
                sum(bool(row.get("correct_next_needle")) for row in arm) / denominator
                if denominator
                else None
            ),
            "truncated_rate": (
                sum(bool(row.get("generation_truncated")) for row in arm) / denominator
                if denominator
                else None
            ),
        }
    return {
        "diagnostic_only": True,
        "formal_claim_permitted": False,
        "raw_rows": len(rows),
        "complete_anchors": len(complete) // 5,
        "seed_count_seen": len({int(row["seed"]) for row in complete}),
        "stats": stats,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for run in args.runs:
        print(json.dumps({"run": str(run), **diagnose(run)}, sort_keys=True))


if __name__ == "__main__":
    main()
