#!/usr/bin/env python3
"""Freeze clean-correct Native answer-query donor/receiver pairs and layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.parsing import parse_trace_record


SCHEMA = "realistic_niah_v5_answer_query_layer_sweep_plan_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    generations: Path,
    model_label: str,
    output_dir: Path,
    *,
    layers: list[int],
    num_layers: int,
    site_id: str,
    unordered_pairs_per_seed: int,
) -> dict[str, Any]:
    frozen_layers = sorted(set(int(layer) for layer in layers))
    if len(frozen_layers) != len(layers):
        raise ValueError("Layer grid must not contain duplicates")
    if not frozen_layers or frozen_layers[0] < 0 or frozen_layers[-1] >= num_layers:
        raise ValueError(
            f"Layer grid must fall in [0, {num_layers - 1}]: {frozen_layers}"
        )

    model_rows = [
        row
        for row in read_jsonl(generations)
        if str(row.get("model_label", row.get("model"))) == model_label
    ]
    eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    excluded_incorrect: list[str] = []
    excluded_non_one_to_one: list[str] = []
    for row in model_rows:
        if str(row.get("split")) != "confirmation":
            continue
        parsed = parse_trace_record(row)
        request_id = str(row.get("request_id", row.get("stimulus_id")))
        if not bool(parsed["parser"].get("trace_one_to_one")):
            excluded_non_one_to_one.append(request_id)
            continue
        if not bool(parsed.get("exact_count")):
            excluded_incorrect.append(request_id)
            continue
        eligible.append((row, parsed))

    by_seed: dict[int, dict[int, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row, parsed in eligible:
        seed = int(row["seed"])
        count = int(parsed["gold_count"])
        if count in by_seed.setdefault(seed, {}):
            raise ValueError(f"Duplicate confirmation seed/count: {seed}/{count}")
        by_seed[seed][count] = (row, parsed)

    if unordered_pairs_per_seed < 1 or unordered_pairs_per_seed > 3:
        raise ValueError("unordered_pairs_per_seed must be in [1, 3]")
    pairs: list[dict[str, Any]] = []
    for seed, by_count in sorted(by_seed.items()):
        counts = sorted(by_count)
        available_edges = list(zip(counts[:-1], counts[1:]))
        edge_count = min(unordered_pairs_per_seed, len(available_edges))
        targets = {
            0: (),
            1: (5.5,),
            2: (1.5, 9.5),
            3: (1.5, 5.5, 9.5),
        }[edge_count]
        selected_edges: list[tuple[int, int]] = []
        for target in targets:
            remaining = [
                edge for edge in available_edges if edge not in selected_edges
            ]
            selected_edges.append(
                min(
                    remaining,
                    key=lambda edge: (
                        abs(((edge[0] + edge[1]) / 2.0) - target),
                        edge[1] - edge[0],
                        edge[0],
                    ),
                )
            )
        for lower_count, higher_count in sorted(selected_edges):
            for receiver_count, donor_count, donor_role in (
                (
                    lower_count,
                    higher_count,
                    "same_seed_adjacent_available_higher",
                ),
                (
                    higher_count,
                    lower_count,
                    "same_seed_adjacent_available_lower",
                ),
            ):
                receiver_row, receiver_parsed = by_count[receiver_count]
                receiver_id = str(
                    receiver_row.get(
                        "request_id", receiver_row.get("stimulus_id")
                    )
                )
                donor_row, donor_parsed = by_count[donor_count]
                donor_id = str(
                    donor_row.get("request_id", donor_row.get("stimulus_id"))
                )
                pairs.append(
                    {
                        "schema_version": SCHEMA,
                        "pair_id": (
                            f"{model_label}__seed{seed}__"
                            f"R{receiver_count}__D{donor_count}"
                        ),
                        "model_label": model_label,
                        "seed": seed,
                        "split": "confirmation",
                        "receiver_request_id": receiver_id,
                        "donor_request_id": donor_id,
                        "receiver_count": receiver_count,
                        "donor_count": donor_count,
                        "receiver_site_id": site_id,
                        "donor_site_id": site_id,
                        "donor_role": donor_role,
                        "pair_direction": (
                            "higher_to_lower"
                            if donor_count > receiver_count
                            else "lower_to_higher"
                        ),
                        "receiver_exact_count": bool(
                            receiver_parsed["exact_count"]
                        ),
                        "donor_exact_count": bool(donor_parsed["exact_count"]),
                        "pair_eligibility": (
                            "strict_one_to_one_and_receiver_and_donor_"
                            "baseline_final_answer_exact"
                        ),
                        "pair_selection_uses_patch_outcome": False,
                    }
                )

    if not pairs:
        raise ValueError("No eligible confirmation donor/receiver pairs")
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / f"{model_label}__answer_query_layer_pairs.jsonl"
    write_jsonl(pair_path, pairs)
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "model_label": model_label,
        "generations": str(generations.resolve()),
        "generations_sha256": sha256(generations),
        "split": "confirmation",
        "site_id": site_id,
        "query_definition": (
            "answer_query_v3: last literal trace token immediately before the "
            "first numeric answer token"
        ),
        "surface_site_policy": (
            "receiver and donor use the same registered semantic/literal query "
            "site; absolute token index is not required to match"
        ),
        "num_model_layers": int(num_layers),
        "layers": frozen_layers,
        "normalized_layer_depths": [
            float(layer / max(1, num_layers - 1)) for layer in frozen_layers
        ],
        "layer_grid_policy": (
            "architecture-fixed approximately uniform post-block depths; no "
            "patch outcome or confirmation label used to select a layer"
        ),
        "conditions": ["self_patch", "full_donor_patch"],
        "behavioral_endpoint": (
            "actual deterministic greedy continuation parsed as an integer; "
            "invalid or out-of-range text counts as donor-adoption failure"
        ),
        "pair_policy": (
            "within each seed, choose up to three adjacent-in-available-count-grid "
            "undirected pairs nearest low/middle/high count anchors, then run both "
            "directions; all rows are strict one-to-one and clean-correct"
        ),
        "unordered_pairs_per_seed_cap": int(unordered_pairs_per_seed),
        "pair_selection_uses_patch_outcome": False,
        "eligible_clean_correct_rows": len(eligible),
        "eligible_seeds": sorted(by_seed),
        "registered_pairs": len(pairs),
        "directed_gap_histogram": {
            str(gap): sum(
                int(pair["donor_count"]) - int(pair["receiver_count"]) == gap
                for pair in pairs
            )
            for gap in sorted(
                {
                    int(pair["donor_count"]) - int(pair["receiver_count"])
                    for pair in pairs
                }
            )
        },
        "excluded_incorrect_one_to_one_rows": len(excluded_incorrect),
        "excluded_incorrect_request_ids": sorted(excluded_incorrect),
        "excluded_non_one_to_one_rows": len(excluded_non_one_to_one),
        "pairs": str(pair_path.resolve()),
        "pairs_sha256": sha256(pair_path),
    }
    audit_path = output_dir / f"{model_label}__answer_query_layer_plan_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--site-id", default="answer_query_v3")
    parser.add_argument("--unordered-pairs-per-seed", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.generations,
                args.model,
                args.output_dir,
                layers=args.layers,
                num_layers=args.num_layers,
                site_id=args.site_id,
                unordered_pairs_per_seed=args.unordered_pairs_per_seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
