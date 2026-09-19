#!/usr/bin/env python3
"""Create a frozen generation view containing exactly one aligned row per seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_view(
    generations: list[dict[str, Any]], anchors: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_request = {str(row["request_id"]): row for row in generations}
    if len(by_request) != len(generations):
        raise ValueError("Generation input has duplicate request IDs")
    if any("selection_rank" in row for row in anchors):
        raise ValueError("Aligned anchors contain forbidden selection_rank")
    selected: list[dict[str, Any]] = []
    for anchor in anchors:
        request_id = str(anchor["request_id"])
        if request_id not in by_request:
            raise KeyError(f"Aligned request is missing from generations: {request_id}")
        row = dict(by_request[request_id])
        if int(row["seed"]) != int(anchor["seed"]):
            raise ValueError(f"Seed mismatch for {request_id}")
        count = int(row.get("gold_count", len(row.get("gold_records", row.get("gold_pairs", [])))))
        if count != int(anchor["gold_count"]):
            raise ValueError(f"Count mismatch for {request_id}")
        row.update(
            {
                "alignment_pair_id": str(anchor["alignment_pair_id"]),
                "alignment_key": list(anchor["alignment_key"]),
                "cross_model_exact_sample_alignment": True,
                "alignment_selection_rank_used": False,
                "alignment_outcome_blind": True,
            }
        )
        selected.append(row)
    selected.sort(key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in selected]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Aligned generation view must have one row per seed")
    core = {
        "schema_version": "realistic_niah_v5_aligned_generation_view_v1",
        "status": "FROZEN_EXACT_SAMPLE_ALIGNMENT",
        "seeds": seeds,
        "row_count": len(selected),
        "alignment_keys": [row["alignment_key"] for row in selected],
        "outcome_blind": True,
        "selection_rank_used": False,
    }
    return selected, {**core, "view_sha256": _sha256_json(selected), "manifest_sha256": _sha256_json(core)}


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Existing frozen output changed: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = build_view(
        _read_jsonl(args.generations), _read_jsonl(args.anchors)
    )
    _atomic(
        args.output,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    _atomic(args.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

