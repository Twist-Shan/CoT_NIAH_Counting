#!/usr/bin/env python3
"""Audit position- and token-matched natural progress sites across a k grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import load_registered_tokenizer  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.counting_mechanism_transfer import (  # noqa: E402
    build_first_pass_tstar_answer_source_registry,
)
from realistic_niah_v5.natural_aligned_progress import (  # noqa: E402
    SITE_POLICIES,
    align_natural_donor_prompt,
    matched_post_item_site_candidates,
    matched_post_item_sites,
)
from scripts.run_realistic_niah_v5_cross_seed_counter_recurrence import (  # noqa: E402
    prefix_through_boundary,
)
from scripts.run_realistic_niah_v5_same_site_progress_transplant import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _read_rows,
)


SCHEMA_VERSION = "natural_aligned_k_grid_audit_v1"


def _audit_pair(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    gold_count: int,
    receiver_occurrence: int,
    donor_occurrence: int,
    tail_window: int,
    site_policy: str,
    include_site_candidates: bool,
) -> dict[str, Any]:
    encoding, registry = build_first_pass_tstar_answer_source_registry(
        row,
        tokenizer,
        candidate_counts=tuple(range(1, int(gold_count) + 1)),
    )
    receiver_site, donor_site, site_audit = matched_post_item_sites(
        encoding,
        registry.trace_items,
        receiver_occurrence=receiver_occurrence,
        donor_occurrence=donor_occurrence,
        tokenizer=tokenizer,
        tail_window=tail_window,
        site_policy=site_policy,
    )
    aligned_donor, alignment_audit = align_natural_donor_prompt(
        encoding,
        registry,
        receiver_site=receiver_site,
        donor_site=donor_site,
        tokenizer=tokenizer,
    )
    receiver_prefix = prefix_through_boundary(encoding, receiver_site)
    donor_prefix = prefix_through_boundary(aligned_donor, receiver_site)
    if receiver_prefix.sequence_length != donor_prefix.sequence_length:
        raise RuntimeError("Aligned natural prefixes have different lengths")
    if int(receiver_prefix.input_ids[-1]) != int(donor_prefix.input_ids[-1]):
        raise RuntimeError("Aligned natural prefixes have different commit tokens")
    outcome = {
        "schema_version": SCHEMA_VERSION,
        "request_id": str(row["request_id"]),
        "seed": int(row["seed"]),
        "receiver_occurrence_j": int(receiver_occurrence),
        "donor_occurrence_k": int(donor_occurrence),
        "receiver_successor": int(receiver_occurrence) + 1,
        "donor_successor": int(donor_occurrence) + 1,
        "status": "PASS",
        "aligned_prefix_token_count": int(receiver_prefix.sequence_length),
        **site_audit,
        **alignment_audit,
    }
    if include_site_candidates:
        outcome["site_candidates"] = matched_post_item_site_candidates(
            encoding,
            registry.trace_items,
            receiver_occurrence=receiver_occurrence,
            donor_occurrence=donor_occurrence,
            tokenizer=tokenizer,
            tail_window=tail_window,
        )
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--gold-count", type=int, required=True)
    parser.add_argument("--donor-occurrences", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--tail-window", type=int, default=4)
    parser.add_argument("--site-policy", choices=SITE_POLICIES, default="latest_structural")
    parser.add_argument("--include-site-candidates", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    donors = tuple(
        sorted(
            {
                int(value)
                for value in (
                    args.donor_occurrences
                    if args.donor_occurrences is not None
                    else range(2, int(args.gold_count))
                )
            }
        )
    )
    invalid = [value for value in donors if not 2 <= value < int(args.gold_count)]
    if invalid:
        raise ValueError(f"Donor occurrences must satisfy 2 <= k < N: {invalid}")
    rows = _read_rows(
        args.generations,
        gold_count=int(args.gold_count),
        seeds=args.seeds,
        max_seeds=args.max_seeds,
    )
    tokenizer = load_registered_tokenizer(
        resolve_model_spec(str(args.model)),
        cache_dir=args.cache_dir,
    )

    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        for donor in donors:
            receiver = donor - 1
            try:
                audit_rows.append(
                    _audit_pair(
                        row,
                        tokenizer,
                        gold_count=int(args.gold_count),
                        receiver_occurrence=receiver,
                        donor_occurrence=donor,
                        tail_window=int(args.tail_window),
                        site_policy=str(args.site_policy),
                        include_site_candidates=bool(args.include_site_candidates),
                    )
                )
            except Exception as error:  # Preserve every pre-registered cell.
                audit_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "request_id": str(row["request_id"]),
                        "seed": int(row["seed"]),
                        "receiver_occurrence_j": receiver,
                        "donor_occurrence_k": donor,
                        "receiver_successor": donor,
                        "donor_successor": donor + 1,
                        "status": "FAIL",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

    summary = []
    for donor in donors:
        group = [row for row in audit_rows if int(row["donor_occurrence_k"]) == donor]
        passed = [row for row in group if row["status"] == "PASS"]
        summary.append(
            {
                "donor_occurrence_k": donor,
                "receiver_occurrence_j": donor - 1,
                "registered_seed_count": len(group),
                "pass_count": len(passed),
                "pass_rate": len(passed) / len(group) if group else None,
                "shared_commit_token_texts": sorted(
                    {str(row["shared_commit_token_text"]) for row in passed}
                ),
                "alignment_token_deltas": sorted(
                    {int(row["alignment_token_delta"]) for row in passed}
                ),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output / "audit.jsonl", audit_rows)
    _atomic_json(args.output / "summary.json", summary)
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if all(row["status"] == "PASS" for row in audit_rows) else "FAIL",
            "model": str(args.model),
            "gold_count": int(args.gold_count),
            "donor_occurrences": list(donors),
            "tail_window": int(args.tail_window),
            "site_policy": str(args.site_policy),
            "includes_site_candidates": bool(args.include_site_candidates),
            "seed_count": len(rows),
            "registered_cell_count": len(audit_rows),
            "passed_cell_count": sum(row["status"] == "PASS" for row in audit_rows),
        },
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
