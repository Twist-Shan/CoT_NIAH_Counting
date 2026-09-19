#!/usr/bin/env python3
"""Combine completed natural-aligned k-grid cell files descriptively."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_realistic_niah_v5_natural_aligned_k_grid import (  # noqa: E402
    _read_jsonl,
    _seed_cluster_bootstrap,
    _summarize,
    _write_json,
    _write_jsonl,
)


SCHEMA_VERSION = "natural_aligned_k_grid_combined_v1"


def _grammar_label(token: str) -> str:
    if token == ".":
        return "plain_period"
    if token.startswith('."'):
        return "quote_closing_period"
    if token.isspace():
        return "whitespace"
    return "other"


def _group_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize(cells)
    summary["seeds"] = sorted({int(cell["seed"]) for cell in cells})
    summary["seed_count"] = len(summary["seeds"])
    summary["seed_cluster_bootstrap_95ci_first_successor_skip_rate"] = (
        _seed_cluster_bootstrap(cells)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for path in args.cell_files:
        for source in _read_jsonl(path):
            key = (int(source["seed"]), int(source["donor_occurrence_k"]))
            if key in seen:
                raise ValueError(f"Duplicate seed/k cell: {key}")
            seen.add(key)
            cells.append(
                {
                    **source,
                    "schema_version": SCHEMA_VERSION,
                    "commit_grammar": _grammar_label(
                        str(source["shared_commit_token_text"])
                    ),
                }
            )
    cells.sort(key=lambda row: (int(row["seed"]), int(row["donor_occurrence_k"])))
    if any(int(cell["donor_occurrence_k"]) not in range(2, 10) for cell in cells):
        raise ValueError("Combined file contains a donor k outside 2..9")

    per_k = [
        {
            "donor_occurrence_k": donor,
            **_group_summary(
                [cell for cell in cells if int(cell["donor_occurrence_k"]) == donor]
            ),
        }
        for donor in range(2, 10)
    ]
    per_seed = [
        {
            "seed": seed,
            **_group_summary([cell for cell in cells if int(cell["seed"]) == seed]),
        }
        for seed in sorted({int(cell["seed"]) for cell in cells})
    ]
    per_grammar = [
        {
            "commit_grammar": grammar,
            **_group_summary(
                [cell for cell in cells if str(cell["commit_grammar"]) == grammar]
            ),
        }
        for grammar in sorted({str(cell["commit_grammar"]) for cell in cells})
    ]
    overall = _group_summary(cells)
    overall["successful_cells"] = [
        {
            "seed": int(cell["seed"]),
            "donor_occurrence_k": int(cell["donor_occurrence_k"]),
            "donor_successor": int(cell["donor_successor"]),
            "commit_grammar": str(cell["commit_grammar"]),
        }
        for cell in cells
        if cell["eligible"] and cell["first_successor_skip"]
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output / "cells.jsonl", cells)
    _write_json(args.output / "overall.json", overall)
    _write_json(args.output / "per_k.json", per_k)
    _write_json(args.output / "per_seed.json", per_seed)
    _write_json(args.output / "per_commit_grammar.json", per_grammar)
    _write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "cell_files": [str(path) for path in args.cell_files],
            "registered_cell_count": len(cells),
            "seeds": sorted({int(cell["seed"]) for cell in cells}),
            "donor_occurrences": list(range(2, 10)),
            "interpretation": (
                "descriptive combination; seed1791 was grammar-selected after the "
                "initial five-seed panel"
            ),
        },
    )
    print(
        json.dumps(
            {"overall": overall, "per_k": per_k, "per_grammar": per_grammar},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
