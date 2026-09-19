#!/usr/bin/env python3
"""Audit frozen marker-pair coverage for requested split/transition strata."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "realistic_niah_v5_marker_pair_coverage_audit_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transitions(value: str) -> tuple[tuple[int, int], ...]:
    result = tuple(tuple(int(part) for part in item.split(":")) for item in value.split(","))
    if any(len(pair) != 2 or pair[1] != pair[0] + 1 for pair in result):
        raise argparse.ArgumentTypeError("adjacent low:high transitions required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=transitions, default=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))
    parser.add_argument("--minimum-per-stratum", type=int, default=1)
    args = parser.parse_args()

    rows = read_jsonl(args.pairs)
    counts = Counter(
        (str(row["split"]), int(row["counterfactual_count"]), int(row["full_count"]))
        for row in rows
    )
    coverage = {}
    missing = []
    for split in ("discovery", "confirmation"):
        for low, high in args.transitions:
            label = f"{split}:N{low}_to_N{high}"
            count = counts[(split, low, high)]
            coverage[label] = count
            if count < args.minimum_per_stratum:
                missing.append(label)
    audit = {
        "schema_version": SCHEMA,
        "status": "passed" if not missing else "incomplete",
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256(args.pairs),
        "minimum_per_stratum": args.minimum_per_stratum,
        "coverage": coverage,
        "missing": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
