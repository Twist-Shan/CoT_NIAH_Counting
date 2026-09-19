#!/usr/bin/env python3
"""Audit token-span eligibility without reading query-mediation outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAIR_KEYS = ("seed", "receiver_occurrence", "donor_occurrence")


def _rows(root: Path) -> list[dict[str, Any]]:
    shards = sorted((root / "shards").glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No query-mediation shards under {root}")
    pairs: dict[tuple[int, int, int], dict[str, Any]] = {}
    for shard in shards:
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            # The audit deliberately selects only position metadata.
            key = tuple(int(raw[name]) for name in PAIR_KEYS)
            value = {
                "model_label": str(raw["model_label"]),
                "seed": int(raw["seed"]),
                "receiver_occurrence": int(raw["receiver_occurrence"]),
                "donor_occurrence": int(raw["donor_occurrence"]),
                "donor_offset": int(raw["donor_offset"]),
                "receiver_span_token_count": int(
                    raw["receiver_span_token_count"]
                ),
                "donor_span_token_count": int(raw["donor_span_token_count"]),
            }
            prior = pairs.setdefault(key, value)
            if prior != value:
                raise ValueError(f"Span metadata changed within pair {key}")
    return list(pairs.values())


def audit(root: Path) -> dict[str, Any]:
    rows = _rows(root)
    models = sorted({row["model_label"] for row in rows})
    if len(models) != 1:
        raise ValueError(f"Expected one model, found {models}")
    seeds = sorted({row["seed"] for row in rows})
    offsets = sorted({row["donor_offset"] for row in rows})
    per_seed: dict[int, dict[str, int]] = defaultdict(
        lambda: {"pairs": 0, "suffix4": 0, "suffix8": 0, "full_span": 0}
    )
    per_offset: dict[int, dict[str, int]] = defaultdict(
        lambda: {"pairs": 0, "suffix4": 0, "suffix8": 0, "full_span": 0}
    )
    minimum_lengths: Counter[int] = Counter()
    for row in rows:
        receiver_length = int(row["receiver_span_token_count"])
        donor_length = int(row["donor_span_token_count"])
        minimum = min(receiver_length, donor_length)
        minimum_lengths[minimum] += 1
        eligibility = {
            "pairs": True,
            "suffix4": minimum >= 4,
            "suffix8": minimum >= 8,
            "full_span": receiver_length == donor_length,
        }
        for label, eligible in eligibility.items():
            if eligible:
                per_seed[int(row["seed"])][label] += 1
                per_offset[int(row["donor_offset"])][label] += 1
    summary = {}
    for label in ("suffix4", "suffix8", "full_span"):
        eligible = sum(per_seed[seed][label] for seed in seeds)
        summary[label] = {
            "eligible_pair_count": eligible,
            "eligible_pair_fraction": eligible / len(rows),
            "seeds_with_all_offsets": sum(
                per_seed[seed][label] == len(offsets) for seed in seeds
            ),
            "seeds_with_any_pair": sum(
                per_seed[seed][label] > 0 for seed in seeds
            ),
            "per_offset_eligible": {
                str(offset): per_offset[offset][label] for offset in offsets
            },
        }
    return {
        "schema_version": "realistic_niah_v5_query_mediation_span_eligibility_v1",
        "model_label": models[0],
        "source_root": str(root.resolve()),
        "outcome_fields_accessed": False,
        "pair_count": len(rows),
        "seed_count": len(seeds),
        "seeds": seeds,
        "offsets": offsets,
        "minimum_span_length_distribution": {
            str(length): count for length, count in sorted(minimum_lengths.items())
        },
        "eligibility": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
