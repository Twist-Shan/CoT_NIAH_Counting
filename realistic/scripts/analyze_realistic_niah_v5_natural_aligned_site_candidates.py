#!/usr/bin/env python3
"""Summarize outcome-blind natural item-tail site candidates by grammar."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _grammar(candidate: dict[str, Any]) -> str:
    if bool(candidate["is_exact_period"]):
        return "plain_period"
    if bool(candidate["is_whitespace"]):
        return "whitespace"
    if bool(candidate["is_period_like"]):
        return "period_like_nonplain"
    if bool(candidate["is_structural"]):
        return "other_structural"
    return "lexical"


def _latest(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda row: (
            int(bool(row["is_structural"])),
            int(row["receiver_site"]),
            int(row["donor_site_before_alignment"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [row for row in _read_jsonl(args.audit) if row["status"] == "PASS"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cells: list[dict[str, Any]] = []
    for row in rows:
        candidates = list(row["site_candidates"])
        latest = _latest(candidates)
        grammar = _grammar(latest)
        selected_text = str(row["shared_commit_token_text"])
        cell = {
            "seed": int(row["seed"]),
            "donor_occurrence_k": int(row["donor_occurrence_k"]),
            "original_site_grammar": grammar,
            "original_token_text": str(latest["shared_commit_token_text"]),
            "period_preferred_token_text": selected_text,
            "period_preferred_receiver_tail_offset": int(row["receiver_tail_offset"]),
            "period_preferred_donor_tail_offset": int(row["donor_tail_offset"]),
            "exact_period_available": any(bool(value["is_exact_period"]) for value in candidates),
            "period_like_available": any(bool(value["is_period_like"]) for value in candidates),
            "non_whitespace_structural_available": any(
                bool(value["is_structural"]) and not bool(value["is_whitespace"])
                for value in candidates
            ),
            "candidate_count": len(candidates),
        }
        cells.append(cell)
        groups[grammar].append(cell)

    summary = []
    for grammar, group in sorted(groups.items()):
        summary.append(
            {
                "original_site_grammar": grammar,
                "cell_count": len(group),
                "seed_count": len({row["seed"] for row in group}),
                "exact_period_available_count": sum(row["exact_period_available"] for row in group),
                "period_like_available_count": sum(row["period_like_available"] for row in group),
                "non_whitespace_structural_available_count": sum(
                    row["non_whitespace_structural_available"] for row in group
                ),
                "period_preferred_token_text_counts": dict(
                    Counter(row["period_preferred_token_text"] for row in group)
                ),
                "period_preferred_receiver_tail_offset_counts": dict(
                    Counter(row["period_preferred_receiver_tail_offset"] for row in group)
                ),
            }
        )

    seed_summary = []
    for seed in sorted({row["seed"] for row in cells}):
        group = [row for row in cells if row["seed"] == seed]
        seed_summary.append(
            {
                "seed": seed,
                "cell_count": len(group),
                "original_site_grammar_counts": dict(
                    Counter(row["original_site_grammar"] for row in group)
                ),
                "period_preferred_token_text_counts": dict(
                    Counter(row["period_preferred_token_text"] for row in group)
                ),
                "all_cells_have_period_like_candidate": all(
                    row["period_like_available"] for row in group
                ),
            }
        )

    payload = {"summary": summary, "seed_summary": seed_summary, "cells": cells}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
