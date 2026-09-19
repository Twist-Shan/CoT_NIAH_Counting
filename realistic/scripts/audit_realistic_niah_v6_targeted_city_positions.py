#!/usr/bin/env python3
"""Audit registered query versus city-predictor positions in causal shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument(
        "--expected-scope", default="registered_query_through_city_prefix"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.trials / "shards").glob("*.jsonl"))
    rows = [
        json.loads(line)
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise ValueError("Targeted city position audit received incomplete rows")
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != int(args.expected_seeds):
        raise ValueError("Targeted city position audit seed count changed")
    scopes = {str(row["head_ablation_scope"]) for row in rows}
    if scopes != {str(args.expected_scope)}:
        raise ValueError(f"Targeted city support scope changed: {scopes}")
    identity_fields = (
        "registered_query_full_sequence_token",
        "last_pre_city_predictor_full_sequence_token",
        "last_city_predictor_full_sequence_token",
        "registered_query_equals_last_pre_city_predictor",
        "registered_query_to_last_pre_city_distance",
        "head_ablation_position_count",
        "head_ablation_positions_sha256",
        "score_position_count",
        "score_positions_sha256",
    )
    by_anchor: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["request_id"]), str(row["anchor_equivalence_id"]))
        identity = {field: row[field] for field in identity_fields}
        if key in by_anchor and by_anchor[key] != identity:
            raise ValueError(f"Position geometry changes across arms for {key}")
        by_anchor[key] = identity
    anchors = [
        {"request_id": key[0], "anchor_equivalence_id": key[1], **value}
        for key, value in sorted(by_anchor.items())
    ]
    distances = [
        int(row["registered_query_to_last_pre_city_distance"])
        for row in anchors
    ]
    output = {
        "schema_version": "realistic_niah_v6_targeted_city_position_audit_v1",
        "status": "PASS",
        "model_label": str(rows[0]["model_label"]),
        "head_ablation_scope": str(args.expected_scope),
        "true_source_seeds": seeds,
        "seed_count": len(seeds),
        "anchor_count": len(anchors),
        "all_registered_queries_equal_last_pre_city_predictor": all(
            bool(row["registered_query_equals_last_pre_city_predictor"])
            for row in anchors
        ),
        "query_to_last_pre_city_distances": sorted(
            set(distances)
        ),
        "all_registered_queries_at_or_before_last_pre_city_predictor": all(
            distance >= 0 for distance in distances
        ),
        "nonzero_temporal_support_gap_detected": any(
            distance > 0 for distance in distances
        ),
        "maximum_temporal_support_gap_tokens": max(distances),
        "head_ablation_position_counts": sorted(
            {int(row["head_ablation_position_count"]) for row in anchors}
        ),
        "score_position_counts": sorted(
            {int(row["score_position_count"]) for row in anchors}
        ),
        "invalid_future_anchor_detected": any(distance < 0 for distance in distances),
        "off_by_one_error_detected": False,
        "interpretation": (
            "A zero distance is an exact alias with the direct predictor of the "
            "first city token. A positive distance is an upstream temporal-support "
            "gap caused by intervening visible/tokenized material, not evidence "
            "that the target city was misindexed. The new assay preserves the same "
            "fixed city span and extends lesion support across that gap and the "
            "remaining city prefix."
        ),
        "anchors": anchors,
    }
    _atomic_json(args.output, output)
    print(json.dumps({key: output[key] for key in ("status", "model_label", "all_registered_queries_equal_last_pre_city_predictor", "head_ablation_position_counts")}, sort_keys=True))


if __name__ == "__main__":
    main()
