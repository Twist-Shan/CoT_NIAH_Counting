#!/usr/bin/env python3
"""Independently prove that completed JSONL shards never contain selection_rank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _files(path: Path) -> list[Path]:
    files = sorted(path.glob("shards/*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No JSONL shards under {path}")
    return files


def audit(path: Path, *, expected_seeds: int) -> dict[str, Any]:
    files = _files(path)
    seeds: set[int] = set()
    row_count = 0
    file_hashes: dict[str, str] = {}
    violations: list[dict[str, Any]] = []
    for shard in files:
        payload = shard.read_bytes()
        file_hashes[shard.name] = hashlib.sha256(payload).hexdigest()
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_count += 1
            if "seed" in row:
                seeds.add(int(row["seed"]))
            if "selection_rank" in row:
                violations.append(
                    {"file": shard.name, "line_number": line_number}
                )
    if violations:
        raise ValueError(f"selection_rank found in completed shards: {violations[:5]}")
    observed_seeds = sorted(seeds)
    if len(observed_seeds) != int(expected_seeds):
        raise ValueError(
            f"Expected {expected_seeds} seeds, observed {observed_seeds}"
        )
    hash_payload = json.dumps(file_hashes, sort_keys=True).encode("utf-8")
    return {
        "schema_version": "realistic_niah_v5_selection_rank_absence_audit_v1",
        "status": "PASS",
        "selection_rank_used": False,
        "expected_seed_count": int(expected_seeds),
        "seed_count": len(observed_seeds),
        "seeds": observed_seeds,
        "shard_count": len(files),
        "row_count": row_count,
        "source_file_sha256": file_hashes,
        "source_file_hash_ledger_sha256": hashlib.sha256(hash_payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.trials.resolve(), expected_seeds=int(args.expected_seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "source_file_sha256"}, sort_keys=True))


if __name__ == "__main__":
    main()
