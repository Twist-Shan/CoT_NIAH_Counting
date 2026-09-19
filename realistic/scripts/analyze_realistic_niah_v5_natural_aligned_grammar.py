#!/usr/bin/env python3
"""Stratify a natural-aligned transplant run by matched commit grammar."""

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
    _write_json,
    _write_jsonl,
)
from scripts.combine_realistic_niah_v5_natural_aligned_k_grid import (  # noqa: E402
    _grammar_label,
)


SCHEMA_VERSION = "natural_aligned_grammar_stratification_v1"
CONDITIONS = ("receiver_self", "native_donor", "donor_to_receiver")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    geometry = {
        int(row["seed"]): row
        for row in _read_jsonl(args.input / "geometry_audit.jsonl")
    }
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl(args.input / "trials.jsonl"):
        if int(row["layer"]) != int(args.layer):
            continue
        condition = str(row["condition"])
        if condition in CONDITIONS:
            grouped.setdefault(int(row["seed"]), {})[condition] = row

    cells: list[dict[str, Any]] = []
    for seed in sorted(grouped):
        conditions = grouped[seed]
        if set(conditions) != set(CONDITIONS):
            raise RuntimeError(f"seed={seed} does not have all three conditions")
        receiver = conditions["receiver_self"]
        native_donor = conditions["native_donor"]
        transplant = conditions["donor_to_receiver"]
        geo = geometry[seed]
        receiver_successor = int(receiver["receiver_successor"])
        donor_successor = int(receiver["donor_successor"])
        receiver_first = receiver.get("first_generated_known_city_ordinal")
        transplant_first = transplant.get("first_generated_known_city_ordinal")
        eligible = bool(
            int(geo["aligned_donor_site"]) == int(geo["receiver_site"])
            and receiver["receiver_successor_argmax"]
            and native_donor["donor_successor_argmax"]
            and receiver_first == receiver_successor
        )
        token = str(geo["shared_commit_token_text"])
        cells.append(
            {
                "schema_version": SCHEMA_VERSION,
                "seed": seed,
                "receiver_occurrence_j": int(receiver["receiver_occurrence_j"]),
                "donor_occurrence_k": int(receiver["donor_occurrence_k"]),
                "receiver_successor": receiver_successor,
                "donor_successor": donor_successor,
                "shared_commit_token_text": token,
                "commit_grammar": _grammar_label(token),
                "eligible": eligible,
                "receiver_first_generated_known_city_ordinal": receiver_first,
                "transplant_first_generated_known_city_ordinal": transplant_first,
                "first_successor_skip": transplant_first == donor_successor,
                "transplant_donor_candidate_argmax": bool(
                    transplant["donor_successor_argmax"]
                ),
            }
        )

    by_grammar = []
    for grammar in sorted({str(cell["commit_grammar"]) for cell in cells}):
        group = [cell for cell in cells if cell["commit_grammar"] == grammar]
        eligible = [cell for cell in group if cell["eligible"]]
        by_grammar.append(
            {
                "commit_grammar": grammar,
                "registered_cell_count": len(group),
                "eligible_cell_count": len(eligible),
                "first_successor_skip_count": sum(
                    bool(cell["first_successor_skip"]) for cell in eligible
                ),
                "donor_candidate_argmax_count": sum(
                    bool(cell["transplant_donor_candidate_argmax"])
                    for cell in eligible
                ),
                "seeds": [int(cell["seed"]) for cell in group],
                "token_surfaces_json": sorted(
                    {
                        json.dumps(
                            cell["shared_commit_token_text"], ensure_ascii=False
                        )
                        for cell in group
                    }
                ),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output / "cells.jsonl", cells)
    _write_json(args.output / "by_commit_grammar.json", by_grammar)
    _write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "input": str(args.input),
            "layer": int(args.layer),
            "registered_cell_count": len(cells),
            "eligible_cell_count": sum(bool(cell["eligible"]) for cell in cells),
            "interpretation": "post-hoc grammar stratification",
        },
    )
    print(json.dumps(by_grammar, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
