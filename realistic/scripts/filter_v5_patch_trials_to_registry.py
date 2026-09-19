#!/usr/bin/env python3
"""Filter immutable patch trials to a smaller frozen pair registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "realistic_niah_v5_filtered_patch_trials_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions-per-pair", type=int, default=4)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Frozen pair registry contains duplicate pair_id values")
    wanted = set(pair_ids)
    rows = [row for row in read_jsonl(args.trials) if str(row.get("pair_id")) in wanted]
    counts = Counter(str(row["pair_id"]) for row in rows)
    missing = sorted(wanted - set(counts))
    bad = {pair_id: count for pair_id, count in counts.items() if count != args.conditions_per_pair}
    if missing or bad:
        raise ValueError(f"Filtered trial coverage mismatch: missing={missing}, bad={bad}")
    rows.sort(key=lambda row: (pair_ids.index(str(row["pair_id"])), str(row["condition"])))
    atomic_jsonl(args.output, rows)
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "source_trials": str(args.trials.resolve()),
        "source_trials_sha256": sha256(args.trials),
        "frozen_pairs": str(args.pairs.resolve()),
        "frozen_pairs_sha256": sha256(args.pairs),
        "pairs": len(pair_ids),
        "conditions_per_pair": args.conditions_per_pair,
        "rows": len(rows),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
