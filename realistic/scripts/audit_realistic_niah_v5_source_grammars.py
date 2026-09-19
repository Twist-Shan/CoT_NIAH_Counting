#!/usr/bin/env python3
"""Audit grammar and seed coverage in a v5 causal source-write run.

Each source-write shard contains one row per attention head for one retrieval
event.  Reading the first non-empty row is therefore sufficient to recover the
event-level metadata without loading the full head table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _target_grammar(row: dict[str, Any]) -> str:
    direct = row.get("target_grammar_class")
    if direct not in (None, ""):
        return str(direct)
    pair = str(row.get("grammar_pair", ""))
    return pair.rsplit(" -> ", 1)[-1] if pair else "unknown"


def _first_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"Empty shard: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Source-write output directory containing shards/trial_*.jsonl",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    shard_dir = args.source_dir / "shards"
    shards = sorted(shard_dir.glob("trial_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No source-write shards found in {shard_dir}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard in shards:
        row = _first_row(shard)
        row["_shard"] = shard.name
        grouped[_target_grammar(row)].append(row)

    grammars: list[dict[str, Any]] = []
    for grammar, rows in sorted(grouped.items()):
        seeds = sorted({int(row["seed"]) for row in rows})
        requests = {str(row["request_id"]) for row in rows}
        grammars.append(
            {
                "grammar": grammar,
                "events": len(rows),
                "seed_count": len(seeds),
                "seed_ids": seeds,
                "request_count": len(requests),
                "local_eligible_events": sum(
                    bool(row.get("local_anchor_eligible")) for row in rows
                ),
                "primary_eligible_events": sum(
                    bool(row.get("primary_anchor_eligible")) for row in rows
                ),
                "occurrence_min": min(int(row["occurrence"]) for row in rows),
                "occurrence_max": max(int(row["occurrence"]) for row in rows),
            }
        )

    payload = {
        "schema_version": "realistic_niah_v5_source_grammar_audit_v1",
        "source_dir": str(args.source_dir),
        "event_count": len(shards),
        "grammar_count": len(grammars),
        "grammars": grammars,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
