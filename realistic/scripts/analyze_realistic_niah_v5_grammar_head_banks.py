#!/usr/bin/env python3
"""Build and compare grammar-specific v5 retrieval-head rankings.

The implementation streams source-write shards and directly implements the
registered seed/event weighting rule:

    score_g(h) = mean_seed(mean_event(attention_mass_h))

This avoids repeatedly loading the million-row source table once per grammar.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


Head = tuple[int, int]


def _target_grammar(row: dict[str, Any]) -> str:
    direct = row.get("target_grammar_class")
    if direct not in (None, ""):
        return str(direct)
    pair = str(row.get("grammar_pair", ""))
    return pair.rsplit(" -> ", 1)[-1] if pair else "unknown"


def _has_anchor_role(row: dict[str, Any], role: str) -> bool:
    if str(row.get("anchor_role")) == role:
        return True
    roles = row.get("anchor_roles", [])
    return isinstance(roles, list) and role in {str(value) for value in roles}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1"}


def _pearson(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = list(zip(left, right))
    if not pairs:
        return float("nan")
    left_mean = sum(x for x, _ in pairs) / len(pairs)
    right_mean = sum(y for _, y in pairs) / len(pairs)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in pairs
    )
    left_ss = sum((x - left_mean) ** 2 for x, _ in pairs)
    right_ss = sum((y - right_mean) ** 2 for _, y in pairs)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", default="Qwen3-8B")
    parser.add_argument("--anchor-role", default="p0_item_end")
    parser.add_argument("--metric", default="target_source_attention_mass")
    parser.add_argument("--top-k", type=int, nargs="+", default=[32, 64, 96, 128])
    args = parser.parse_args()

    shards = sorted((args.source_dir / "shards").glob("trial_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No source-write shards in {args.source_dir}")

    sums: dict[tuple[str, int, Head], float] = defaultdict(float)
    counts: dict[tuple[str, int, Head], int] = defaultdict(int)
    pooled_sums: dict[tuple[int, Head], float] = defaultdict(float)
    pooled_counts: dict[tuple[int, Head], int] = defaultdict(int)
    event_counts: Counter[str] = Counter()
    event_seeds: dict[str, set[int]] = defaultdict(set)
    event_requests: dict[str, set[str]] = defaultdict(set)
    candidate_heads: set[Head] = set()

    for shard in shards:
        event_recorded = False
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("model_label")) != args.model_label:
                    continue
                if str(row.get("status", "ok")) != "ok":
                    continue
                if not _truthy(row.get("event_specific", True)):
                    continue
                if not _truthy(row.get("local_anchor_eligible", False)):
                    continue
                if not _has_anchor_role(row, args.anchor_role):
                    continue
                try:
                    value = float(row[args.metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                grammar = _target_grammar(row)
                seed = int(row["seed"])
                head = (int(row["layer"]), int(row["head"]))
                candidate_heads.add(head)
                sums[(grammar, seed, head)] += value
                counts[(grammar, seed, head)] += 1
                pooled_sums[(seed, head)] += value
                pooled_counts[(seed, head)] += 1
                if not event_recorded:
                    event_counts[grammar] += 1
                    event_seeds[grammar].add(seed)
                    event_requests[grammar].add(str(row["request_id"]))
                    event_recorded = True

    if not candidate_heads:
        raise ValueError("No eligible source-write rows remain")
    candidate_heads = set(sorted(candidate_heads))
    grammar_names = sorted(event_counts)

    rankings: dict[str, list[dict[str, Any]]] = {}
    for grammar in grammar_names:
        rows: list[dict[str, Any]] = []
        for head in sorted(candidate_heads):
            seed_values = [
                sums[(grammar, seed, head)] / counts[(grammar, seed, head)]
                for seed in sorted(event_seeds[grammar])
                if counts[(grammar, seed, head)]
            ]
            if not seed_values:
                continue
            rows.append(
                {
                    "grammar": grammar,
                    "layer": head[0],
                    "head": head[1],
                    "discovery_selection_value": sum(seed_values)
                    / len(seed_values),
                    "n_seeds": len(seed_values),
                    "event_count": event_counts[grammar],
                    "request_count": len(event_requests[grammar]),
                    "selection_anchor_role": args.anchor_role,
                    "selection_metric": args.metric,
                    "selection_aggregation": "seed_event_mean",
                    "selection_eligibility_scope": "local",
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row["discovery_selection_value"]),
                int(row["layer"]),
                int(row["head"]),
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["discovery_rank"] = rank
        rankings[grammar] = rows
        _write_csv(args.output / f"ranking_{grammar}.csv", rows)

    pooled_rows: list[dict[str, Any]] = []
    pooled_seeds = sorted({seed for seed, _ in pooled_counts})
    for head in sorted(candidate_heads):
        seed_values = [
            pooled_sums[(seed, head)] / pooled_counts[(seed, head)]
            for seed in pooled_seeds
            if pooled_counts[(seed, head)]
        ]
        pooled_rows.append(
            {
                "grammar": "pooled_all_grammars",
                "layer": head[0],
                "head": head[1],
                "discovery_selection_value": sum(seed_values) / len(seed_values),
                "n_seeds": len(seed_values),
                "event_count": sum(event_counts.values()),
                "request_count": len(
                    {
                        request
                        for requests in event_requests.values()
                        for request in requests
                    }
                ),
                "selection_anchor_role": args.anchor_role,
                "selection_metric": args.metric,
                "selection_aggregation": "seed_event_mean",
                "selection_eligibility_scope": "local",
            }
        )
    pooled_rows.sort(
        key=lambda row: (
            -float(row["discovery_selection_value"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    for rank, row in enumerate(pooled_rows, start=1):
        row["discovery_rank"] = rank
    rankings["pooled_all_grammars"] = pooled_rows
    _write_csv(args.output / "ranking_pooled_all_grammars.csv", pooled_rows)

    rank_maps = {
        name: {
            (int(row["layer"]), int(row["head"])): int(
                row["discovery_rank"]
            )
            for row in rows
        }
        for name, rows in rankings.items()
    }
    overlap_rows: list[dict[str, Any]] = []
    bank_names = grammar_names + ["pooled_all_grammars"]
    universe_size = len(candidate_heads)
    for k in sorted(set(args.top_k)):
        if k < 1 or k > universe_size:
            raise ValueError(f"Invalid top-k {k} for {universe_size} heads")
        top_sets = {
            name: {
                (int(row["layer"]), int(row["head"]))
                for row in rankings[name][:k]
            }
            for name in bank_names
        }
        for left_index, left in enumerate(bank_names):
            for right in bank_names[left_index + 1 :]:
                intersection = len(top_sets[left] & top_sets[right])
                union = len(top_sets[left] | top_sets[right])
                chance = (k * k) / universe_size
                common_heads = sorted(top_sets[left] & top_sets[right])
                overlap_rows.append(
                    {
                        "top_k": k,
                        "left": left,
                        "right": right,
                        "left_seed_count": len(event_seeds.get(left, pooled_seeds)),
                        "right_seed_count": len(event_seeds.get(right, pooled_seeds)),
                        "intersection": intersection,
                        "overlap_fraction": intersection / k,
                        "jaccard": intersection / union,
                        "chance_intersection": chance,
                        "intersection_over_chance": (
                            intersection / chance if chance else float("nan")
                        ),
                        "full_rank_spearman": _pearson(
                            (
                                rank_maps[left][head]
                                for head in sorted(candidate_heads)
                            ),
                            (
                                rank_maps[right][head]
                                for head in sorted(candidate_heads)
                            ),
                        ),
                        "common_heads": json.dumps(common_heads),
                    }
                )
    _write_csv(args.output / "pairwise_topk_overlap.csv", overlap_rows)

    main_grammars = [
        grammar for grammar in grammar_names if len(event_seeds[grammar]) >= 10
    ]
    consensus = []
    for k in sorted(set(args.top_k)):
        membership: Counter[Head] = Counter()
        for grammar in main_grammars:
            membership.update(
                (int(row["layer"]), int(row["head"]))
                for row in rankings[grammar][:k]
            )
        histogram = Counter(membership.values())
        consensus.append(
            {
                "top_k": k,
                "main_grammars": main_grammars,
                "grammar_count": len(main_grammars),
                "membership_histogram_nonzero": {
                    str(count): histogram[count]
                    for count in sorted(histogram)
                },
                "heads_in_at_least_two": sum(
                    count >= 2 for count in membership.values()
                ),
                "heads_in_at_least_three": sum(
                    count >= 3 for count in membership.values()
                ),
                "heads_in_every_main_grammar": sorted(
                    head
                    for head, count in membership.items()
                    if count == len(main_grammars)
                ),
            }
        )

    summary = {
        "schema_version": "realistic_niah_v5_grammar_head_bank_overlap_v1",
        "source_dir": str(args.source_dir),
        "model_label": args.model_label,
        "anchor_role": args.anchor_role,
        "selection_metric": args.metric,
        "selection_eligibility_scope": "local",
        "selection_aggregation": (
            "equal_seed_mean_of_within_seed_event_means"
        ),
        "candidate_head_count": universe_size,
        "event_count": sum(event_counts.values()),
        "grammar_count": len(grammar_names),
        "grammars": [
            {
                "grammar": grammar,
                "events": event_counts[grammar],
                "seed_count": len(event_seeds[grammar]),
                "seed_ids": sorted(event_seeds[grammar]),
                "request_count": len(event_requests[grammar]),
                "top_heads": [
                    [int(row["layer"]), int(row["head"])]
                    for row in rankings[grammar][:10]
                ],
            }
            for grammar in grammar_names
        ],
        "main_grammars": main_grammars,
        "top_k_values": sorted(set(args.top_k)),
        "pairwise_overlap_csv": str(
            args.output / "pairwise_topk_overlap.csv"
        ),
        "consensus": consensus,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grammar_head_bank_overlap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
