#!/usr/bin/env python3
"""Freeze outcome-blind geometry-eligible Qwen anchors in count band 5--8."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import load_registered_tokenizer
from realistic_niah_v4.spec import resolve_model_spec

from scripts.build_realistic_niah_v5_write_edge_anchor_subset import (
    CONFIRMATION_SEEDS,
    DISCOVERY_SEEDS,
    REGISTERED_SEEDS,
    _atomic_json,
    _atomic_jsonl,
    _canonical_sha256,
    _read_jsonl,
)
from scripts.build_realistic_niah_v5_write_edge_fullspan_anchor_subset import (
    _generation_index,
    fullspan_post_query_eligibility,
    select_geometry_eligible_anchor_subset,
)
from scripts.run_realistic_niah_v5_count_stream import _cohort_exclusion_reason


MIN_COUNT = 5
MAX_COUNT = 8
SELECTION_RULE = (
    "parser_one_to_one_and_fullspan_post_query_geometry_eligible_in_"
    "fixed_count_band_5_8_"
    "then_highest_gold_count_then_request_id_per_seed"
)


def filter_one_to_one_band(
    rows: Iterable[dict[str, Any]],
    generations: Mapping[str, Mapping[str, Any]],
    *,
    exclusion_fn: Callable[[Mapping[str, Any], str], str | None] = (
        _cohort_exclusion_reason
    ),
) -> list[dict[str, Any]]:
    """Apply the formal parser cohort before geometry/count selection."""

    return [
        row
        for row in rows
        if exclusion_fn(generations[str(row["request_id"])], "one_to_one")
        is None
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model", choices=["Qwen3-8B"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--answer-site-id", default="answer_query_v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = _read_jsonl(args.input)
    band = [
        row
        for row in source
        if MIN_COUNT <= int(row["gold_count"]) <= MAX_COUNT
    ]
    request_ids = {str(row["request_id"]) for row in band}
    generations = _generation_index(_read_jsonl(args.generations), request_ids)
    cohort_band = filter_one_to_one_band(band, generations)
    tokenizer = load_registered_tokenizer(
        resolve_model_spec(str(args.model)), cache_dir=args.cache_dir
    )
    eligible, excluded = fullspan_post_query_eligibility(
        cohort_band,
        generations,
        tokenizer,
        answer_site_id=str(args.answer_site_id),
    )
    selected = select_geometry_eligible_anchor_subset(
        band,
        eligible_request_ids=eligible,
    )
    for row in selected:
        row["write_edge_row_selection_rule"] = SELECTION_RULE
        row["write_edge_fixed_count_band"] = [MIN_COUNT, MAX_COUNT]
    canonical_sha = _canonical_sha256(selected)
    audit = {
        "schema_version": "realistic_niah_v5_write_edge_midcount_anchor_subset_v2",
        "status": "PASS",
        "source_registry": str(args.input.resolve()),
        "source_registry_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "generations": str(args.generations.resolve()),
        "generations_sha256": hashlib.sha256(args.generations.read_bytes()).hexdigest(),
        "model_label": str(args.model),
        "answer_site_id": str(args.answer_site_id),
        "selection_rule": SELECTION_RULE,
        "fixed_count_band": [MIN_COUNT, MAX_COUNT],
        "count_band_rationale": (
            "avoid terminal high-count saturation while preserving a fixed, "
            "outcome-blind mid-count carrier"
        ),
        "eligibility_uses_outcome": False,
        "outcome_blind": True,
        "selection_rank_used": False,
        "registered_seeds": list(REGISTERED_SEEDS),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "band_source_row_count": len(band),
        "one_to_one_band_row_count": len(cohort_band),
        "non_one_to_one_band_row_count": len(band) - len(cohort_band),
        "eligible_band_row_count": len(eligible),
        "excluded_band_row_count": len(excluded),
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
            raise ValueError("Existing frozen mid-count anchor subset changed")
    else:
        _atomic_jsonl(args.output, selected)
    audit_path = args.output.with_suffix(".audit.json")
    if audit_path.exists():
        if json.loads(audit_path.read_text(encoding="utf-8")) != audit:
            raise ValueError("Existing frozen mid-count anchor audit changed")
    else:
        _atomic_json(audit_path, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
