#!/usr/bin/env python3
"""Freeze one geometry-eligible, outcome-blind write-edge anchor per seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import load_registered_tokenizer
from realistic_niah_v4.spec import resolve_model_spec
from realistic_niah_v5.count_stream import build_answer_source_registry
from realistic_niah_v5.integrated_bridge import (
    _final_post_marker_position,
    _post_query_receiver_positions,
)

from scripts.build_realistic_niah_v5_write_edge_anchor_subset import (
    CONFIRMATION_SEEDS,
    DISCOVERY_SEEDS,
    REGISTERED_SEEDS,
    _atomic_json,
    _atomic_jsonl,
    _canonical_sha256,
    _read_jsonl,
)


SELECTION_RULE = (
    "fullspan_post_query_geometry_eligible_then_highest_gold_count_"
    "then_request_id_per_seed"
)


def select_geometry_eligible_anchor_subset(
    rows: Iterable[dict[str, Any]],
    *,
    eligible_request_ids: Iterable[str],
    seeds: Sequence[int] = REGISTERED_SEEDS,
) -> list[dict[str, Any]]:
    """Select the highest-count geometry-eligible row without using outcomes."""

    registered = tuple(int(value) for value in seeds)
    if len(set(registered)) != len(registered):
        raise ValueError("Registered write-edge seeds must be unique")
    eligible = {str(value) for value in eligible_request_ids}
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in registered}
    for raw in rows:
        if "selection_rank" in raw:
            raise ValueError("Write-edge anchor selection must not use selection_rank")
        seed = int(raw["seed"])
        if seed not in by_seed or str(raw["request_id"]) not in eligible:
            continue
        count = int(raw["gold_count"])
        if int(raw["from_occurrence"]) != count - 1:
            raise ValueError("A source anchor is not the final N-1 transition")
        if int(raw["to_occurrence"]) != count:
            raise ValueError("A source anchor does not retrieve terminal item N")
        by_seed[seed].append(dict(raw))
    missing = [seed for seed, values in by_seed.items() if not values]
    if missing:
        raise ValueError(f"No full-span post-query eligible anchor for seeds {missing}")
    selected = []
    for seed in registered:
        candidates = sorted(
            by_seed[seed],
            key=lambda row: (-int(row["gold_count"]), str(row["request_id"])),
        )
        chosen = candidates[0]
        chosen["write_edge_row_selection_rule"] = SELECTION_RULE
        chosen["write_edge_geometry_eligible"] = True
        chosen["write_edge_outcome_blind"] = True
        chosen["write_edge_selection_rank_used"] = False
        selected.append(chosen)
    if [int(row["seed"]) for row in selected] != list(registered):
        raise RuntimeError("Write-edge anchor subset changed registered seed order")
    if len({str(row["request_id"]) for row in selected}) != len(selected):
        raise ValueError("Write-edge anchor subset contains duplicate requests")
    return selected


def _generation_index(
    generations: Iterable[Mapping[str, Any]], request_ids: set[str]
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in generations:
        request_id = str(raw.get("request_id", raw.get("stimulus_id", "")))
        if request_id not in request_ids:
            continue
        if request_id in indexed:
            raise ValueError(f"Duplicate generation row for {request_id}")
        indexed[request_id] = dict(raw)
    missing = sorted(request_ids - set(indexed))
    if missing:
        raise ValueError(f"Missing generation rows for {missing}")
    return indexed


def fullspan_post_query_eligibility(
    anchors: Sequence[dict[str, Any]],
    generations: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    *,
    answer_site_id: str,
) -> tuple[set[str], dict[str, str]]:
    """Compute causal geometry eligibility from text/token positions only."""

    eligible: set[str] = set()
    excluded: dict[str, str] = {}
    for anchor in anchors:
        request_id = str(anchor["request_id"])
        row = generations[request_id]
        encoding, registry = build_answer_source_registry(
            row, tokenizer, answer_site_id=answer_site_id
        )
        count = int(anchor["gold_count"])
        if int(encoding.count) != count or len(registry.trace_items) != count:
            raise ValueError(f"Incomplete registered trace for {request_id}")
        receiver_span = registry.trace_items[-1]
        receiver_positions = tuple(range(int(receiver_span[0]), int(receiver_span[1])))
        targeted_query_position, _specification = _final_post_marker_position(
            row, gold_count=count, targeted_site=anchor
        )
        try:
            _post_query_receiver_positions(targeted_query_position, receiver_positions)
        except ValueError as exc:
            if "not applicable" not in str(exc).lower():
                raise
            excluded[request_id] = str(exc)
        else:
            eligible.add(request_id)
    return eligible, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--answer-site-id", default="answer_query_v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = _read_jsonl(args.input)
    request_ids = {str(row["request_id"]) for row in source}
    generations = _generation_index(_read_jsonl(args.generations), request_ids)
    tokenizer = load_registered_tokenizer(
        resolve_model_spec(str(args.model)), cache_dir=args.cache_dir
    )
    eligible, excluded = fullspan_post_query_eligibility(
        source,
        generations,
        tokenizer,
        answer_site_id=str(args.answer_site_id),
    )
    selected = select_geometry_eligible_anchor_subset(
        source, eligible_request_ids=eligible
    )
    canonical_sha = _canonical_sha256(selected)
    excluded_reason_counts = Counter(excluded.values())
    audit = {
        "schema_version": "realistic_niah_v5_write_edge_fullspan_anchor_subset_v2",
        "status": "PASS",
        "source_registry": str(args.input.resolve()),
        "source_registry_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "generations": str(args.generations.resolve()),
        "generations_sha256": hashlib.sha256(args.generations.read_bytes()).hexdigest(),
        "model_label": str(args.model),
        "answer_site_id": str(args.answer_site_id),
        "selection_rule": SELECTION_RULE,
        "eligibility_rule": (
            "terminal full-span contains at least one strictly post-targeted-query "
            "token and the targeted query does not follow the terminal span"
        ),
        "eligibility_uses_outcome": False,
        "outcome_blind": True,
        "selection_rank_used": False,
        "registered_seeds": list(REGISTERED_SEEDS),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "source_row_count": len(source),
        "eligible_source_row_count": len(eligible),
        "excluded_source_row_count": len(excluded),
        "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "row_count": len(selected),
        "selected_count_by_seed": {
            str(row["seed"]): int(row["gold_count"]) for row in selected
        },
        "selected_request_by_seed": {
            str(row["seed"]): str(row["request_id"]) for row in selected
        },
        "canonical_rows_sha256": canonical_sha,
    }
    if args.output.exists():
        existing = _read_jsonl(args.output)
        if _canonical_sha256(existing) != canonical_sha:
            raise ValueError("Existing frozen full-span anchor subset changed")
    else:
        _atomic_jsonl(args.output, selected)
    audit_path = args.output.with_suffix(".audit.json")
    if audit_path.exists():
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing_audit != audit:
            raise ValueError("Existing full-span anchor audit changed")
    else:
        _atomic_json(audit_path, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
