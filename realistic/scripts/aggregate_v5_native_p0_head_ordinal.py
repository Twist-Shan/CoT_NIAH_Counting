#!/usr/bin/env python3
"""Aggregate exact-P0 attention by grammar, head, and target needle ordinal.

This is a streaming post-processing pass over frozen source-attention shards;
it does not load a model.  The registered aggregation is preserved: events
are averaged within seed first, then seeds receive equal weight.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true"}


def _target_grammar(value: Any) -> str:
    return str(value).rsplit(" -> ", 1)[-1]


def _has_p0(row: dict[str, Any]) -> bool:
    if str(row.get("anchor_role")) == "p0_item_end":
        return True
    return "p0_item_end" in {str(value) for value in row.get("anchor_roles", [])}


def _mean_by_seed(
    totals: dict[tuple[Any, ...], list[float]],
    *,
    key_prefix_length: int,
) -> dict[tuple[Any, ...], dict[str, float | int]]:
    pooled: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    event_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for key, (total, count) in totals.items():
        group = key[:key_prefix_length]
        pooled[group].append(total / count)
        event_counts[group] += int(count)
    return {
        key: {
            "value": sum(values) / len(values),
            "n_seeds": len(values),
            "n_events": event_counts[key],
        }
        for key, values in pooled.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-writes", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ranking_seed: dict[tuple[str, int, int, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    ordinal_seed: dict[tuple[str, int, int, int, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    shard_paths = sorted((args.source_writes / "shards").glob("*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No source-attention shards under {args.source_writes}")
    eligible_rows = 0
    for shard_index, path in enumerate(shard_paths, start=1):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") != "ok" or row.get("split") != "discovery":
                    continue
                if not _truthy(row.get("event_specific")):
                    continue
                if not _truthy(row.get("local_anchor_eligible")):
                    continue
                if not _has_p0(row):
                    continue
                value = float(row["target_source_attention_mass"])
                seed = int(row["seed"])
                layer = int(row["layer"])
                head = int(row["head"])
                ordinal = int(row["to_occurrence"])
                grammar = _target_grammar(row.get("grammar_pair"))
                for scope in ("all", grammar):
                    rank_key = (scope, layer, head, seed)
                    ranking_seed[rank_key][0] += value
                    ranking_seed[rank_key][1] += 1
                    ordinal_key = (scope, ordinal, layer, head, seed)
                    ordinal_seed[ordinal_key][0] += value
                    ordinal_seed[ordinal_key][1] += 1
                eligible_rows += 1
        if shard_index % 100 == 0:
            print(f"processed {shard_index}/{len(shard_paths)} shards", flush=True)

    ranking = _mean_by_seed(ranking_seed, key_prefix_length=3)
    ordinal = _mean_by_seed(ordinal_seed, key_prefix_length=4)
    scopes = sorted({key[0] for key in ranking}, key=lambda value: (value != "all", value))
    result_scopes: dict[str, Any] = {}
    for scope in scopes:
        ranked = sorted(
            (
                {
                    "layer": int(layer),
                    "head": int(head),
                    "score": float(summary["value"]),
                    "n_seeds": int(summary["n_seeds"]),
                }
                for (active_scope, layer, head), summary in ranking.items()
                if active_scope == scope
            ),
            key=lambda row: (-row["score"], row["layer"], row["head"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        selected = ranked[: int(args.top_k)]
        matrix_rows = []
        for rank_row in selected:
            layer = int(rank_row["layer"])
            head = int(rank_row["head"])
            for target_ordinal in range(2, 11):
                summary = ordinal.get((scope, target_ordinal, layer, head))
                matrix_rows.append(
                    {
                        "target_ordinal": target_ordinal,
                        "layer": layer,
                        "head": head,
                        "rank": int(rank_row["rank"]),
                        "value": None if summary is None else float(summary["value"]),
                        "n_seeds": 0 if summary is None else int(summary["n_seeds"]),
                        "n_events": 0 if summary is None else int(summary["n_events"]),
                    }
                )
        result_scopes[scope] = {
            "n_seeds": max((int(row["n_seeds"]) for row in ranked), default=0),
            "ranking": ranked if scope == "all" else selected,
            "selected_heads": selected,
            "ordinal_rows": matrix_rows,
        }

    payload = {
        "schema_version": "realistic_niah_v5_p0_head_ordinal_v1",
        "model_label": args.model,
        "query_site": "p0_item_end",
        "selection_split": "discovery",
        "selection_metric": "target_source_attention_mass",
        "selection_aggregation": "equal_seed_mean_of_within_seed_event_means",
        "top_k": int(args.top_k),
        "source_writes": str(args.source_writes),
        "source_shards": len(shard_paths),
        "eligible_head_rows": eligible_rows,
        "scopes": result_scopes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
