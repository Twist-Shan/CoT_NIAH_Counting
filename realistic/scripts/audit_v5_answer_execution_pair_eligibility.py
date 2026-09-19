#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.parsing import parse_trace_record
from realistic_niah_v5.pipeline import read_jsonl


def audit(path: Path, model: str) -> dict[str, object]:
    by_seed: dict[int, set[int]] = {}
    eligible = []
    excluded_incorrect = []
    for row in read_jsonl(path):
        if str(row.get("model_label", row.get("model"))) != model:
            continue
        parsed = parse_trace_record(row)
        if str(row.get("split")) != "confirmation":
            continue
        if not bool(parsed["parser"].get("trace_one_to_one")):
            continue
        if not bool(parsed.get("exact_count")):
            excluded_incorrect.append(
                str(row.get("request_id", row.get("stimulus_id")))
            )
            continue
        seed = int(row["seed"])
        count = int(parsed["gold_count"])
        by_seed.setdefault(seed, set()).add(count)
        eligible.append((seed, count))
    pairs = []
    for seed, values in sorted(by_seed.items()):
        counts = sorted(values)
        for receiver in counts:
            lower = [value for value in counts if value < receiver]
            higher = [value for value in counts if value > receiver]
            if lower:
                pairs.append((seed, receiver, max(lower)))
            if higher:
                pairs.append((seed, receiver, min(higher)))
    return {
        "model_label": model,
        "generations": str(path.resolve()),
        "eligible_rows": len(eligible),
        "eligibility": (
            "strict one-to-one and baseline final numeric answer equals gold"
        ),
        "excluded_incorrect_one_to_one_rows": len(excluded_incorrect),
        "eligible_seeds": len(by_seed),
        "seeds_with_two_or_more_counts": sum(len(values) >= 2 for values in by_seed.values()),
        "directed_nearest_neighbor_pairs": len(pairs),
        "counts_by_seed": {str(seed): sorted(values) for seed, values in sorted(by_seed.items())},
        "pairs": [
            {"seed": seed, "receiver_count": receiver, "donor_count": donor}
            for seed, receiver, donor in pairs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.generations, args.model), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
