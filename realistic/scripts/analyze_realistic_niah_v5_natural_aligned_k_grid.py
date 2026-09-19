#!/usr/bin/env python3
"""Pool first-successor outcomes from the natural aligned N=10 k grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
from typing import Any


SCHEMA_VERSION = "natural_aligned_k_grid_analysis_v1"
CONDITIONS = ("receiver_self", "native_donor", "donor_to_receiver")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of an empty collection")
    position = probability * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def _seed_cluster_bootstrap(
    cells: list[dict[str, Any]], *, draws: int = 20_000, seed: int = 20260827
) -> list[float] | None:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        if cell["eligible"]:
            grouped.setdefault(int(cell["seed"]), []).append(cell)
    seeds = sorted(grouped)
    if not seeds:
        return None
    generator = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [grouped[generator.choice(seeds)] for _ in seeds]
        flattened = [cell for cluster in sample for cell in cluster]
        estimates.append(
            sum(bool(cell["first_successor_skip"]) for cell in flattened)
            / len(flattened)
        )
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [cell for cell in cells if cell["eligible"]]
    successes = [cell for cell in eligible if cell["first_successor_skip"]]
    retentions = [cell for cell in eligible if cell["receiver_successor_retained"]]
    other = [
        cell
        for cell in eligible
        if not cell["first_successor_skip"] and not cell["receiver_successor_retained"]
    ]
    shifts = [float(cell["donor_vs_receiver_logodds_shift"]) for cell in eligible]
    receiver_margins = [
        float(cell["receiver_donor_vs_receiver_sum_logodds"]) for cell in eligible
    ]
    transplant_margins = [
        float(cell["transplant_donor_vs_receiver_sum_logodds"]) for cell in eligible
    ]
    patch_norms = [float(cell["transplant_patch_delta_norm"]) for cell in eligible]
    return {
        "registered_cell_count": len(cells),
        "eligible_cell_count": len(eligible),
        "first_successor_skip_count": len(successes),
        "first_successor_skip_rate": len(successes) / len(eligible) if eligible else None,
        "receiver_successor_retention_count": len(retentions),
        "other_or_unparsed_first_count": len(other),
        "receiver_accidental_donor_successor_count": sum(
            bool(cell["receiver_accidental_donor_successor"]) for cell in eligible
        ),
        "transplant_donor_candidate_argmax_count": sum(
            bool(cell["transplant_donor_candidate_argmax"]) for cell in eligible
        ),
        "mean_donor_vs_receiver_logodds_shift": statistics.fmean(shifts)
        if shifts
        else None,
        "median_donor_vs_receiver_logodds_shift": statistics.median(shifts)
        if shifts
        else None,
        "mean_receiver_donor_vs_receiver_sum_logodds": statistics.fmean(
            receiver_margins
        )
        if receiver_margins
        else None,
        "mean_transplant_donor_vs_receiver_sum_logodds": statistics.fmean(
            transplant_margins
        )
        if transplant_margins
        else None,
        "mean_transplant_patch_delta_norm": statistics.fmean(patch_norms)
        if patch_norms
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--k6-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells: list[dict[str, Any]] = []
    expected_seeds: tuple[int, ...] | None = None
    for donor in range(2, 10):
        source = args.k6_source if donor == 6 else args.grid_root / f"k{donor}"
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"k={donor} manifest is not PASS")
        if int(manifest["donor_occurrence_k"]) != donor:
            raise RuntimeError(f"k={donor} source has a different donor occurrence")
        if list(manifest["layers"]) != [31]:
            raise RuntimeError(f"k={donor} was not evaluated at frozen L31")
        geometry = {
            int(row["seed"]): row for row in _read_jsonl(source / "geometry_audit.jsonl")
        }
        trials = _read_jsonl(source / "trials.jsonl")
        by_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for row in trials:
            if int(row["layer"]) != 31:
                continue
            condition = str(row["condition"])
            if condition not in CONDITIONS:
                continue
            seed = int(row["seed"])
            if condition in by_seed.setdefault(seed, {}):
                raise RuntimeError(f"Duplicate seed={seed} k={donor} condition={condition}")
            by_seed[seed][condition] = row
        seeds = tuple(sorted(by_seed))
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise RuntimeError(f"k={donor} uses non-uniform seeds: {seeds} != {expected_seeds}")
        for seed in seeds:
            conditions = by_seed[seed]
            missing = sorted(set(CONDITIONS) - set(conditions))
            if missing:
                raise RuntimeError(f"seed={seed} k={donor} missing conditions {missing}")
            receiver = conditions["receiver_self"]
            native_donor = conditions["native_donor"]
            transplant = conditions["donor_to_receiver"]
            receiver_successor = donor
            donor_successor = donor + 1
            geometry_gate = (
                seed in geometry
                and int(geometry[seed]["aligned_donor_site"])
                == int(geometry[seed]["receiver_site"])
            )
            receiver_readout_gate = bool(receiver["receiver_successor_argmax"])
            donor_readout_gate = bool(native_donor["donor_successor_argmax"])
            receiver_first = receiver.get("first_generated_known_city_ordinal")
            transplant_first = transplant.get("first_generated_known_city_ordinal")
            receiver_generation_gate = receiver_first == receiver_successor
            eligible = bool(
                geometry_gate
                and receiver_readout_gate
                and donor_readout_gate
                and receiver_generation_gate
            )
            cells.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "seed": seed,
                    "receiver_occurrence_j": donor - 1,
                    "donor_occurrence_k": donor,
                    "receiver_successor": receiver_successor,
                    "donor_successor": donor_successor,
                    "geometry_gate": geometry_gate,
                    "receiver_readout_gate": receiver_readout_gate,
                    "native_donor_readout_gate": donor_readout_gate,
                    "receiver_generation_gate": receiver_generation_gate,
                    "eligible": eligible,
                    "receiver_first_generated_known_city_ordinal": receiver_first,
                    "transplant_first_generated_known_city_ordinal": transplant_first,
                    "first_successor_skip": transplant_first == donor_successor,
                    "receiver_successor_retained": transplant_first == receiver_successor,
                    "receiver_accidental_donor_successor": receiver_first == donor_successor,
                    "transplant_donor_candidate_argmax": bool(
                        transplant["donor_successor_argmax"]
                    ),
                    "receiver_donor_vs_receiver_sum_logodds": float(
                        receiver["donor_vs_receiver_sum_logodds"]
                    ),
                    "transplant_donor_vs_receiver_sum_logodds": float(
                        transplant["donor_vs_receiver_sum_logodds"]
                    ),
                    "donor_vs_receiver_logodds_shift": float(
                        transplant["donor_vs_receiver_sum_logodds"]
                        - receiver["donor_vs_receiver_sum_logodds"]
                    ),
                    "transplant_patch_delta_norm": float(
                        transplant["realized_patch_delta_norm"]
                    ),
                    "receiver_transition_token_count": int(
                        receiver["receiver_transition_token_count"]
                    ),
                    "donor_transition_token_count": int(
                        transplant["donor_transition_token_count"]
                    ),
                    "shared_commit_token_id": int(
                        geometry[seed]["shared_commit_token_id"]
                    ),
                    "shared_commit_token_text": str(
                        geometry[seed]["shared_commit_token_text"]
                    ),
                    "alignment_token_delta": int(
                        geometry[seed]["alignment_token_delta"]
                    ),
                    "transplant_completion_text": str(transplant["completion_text"]),
                }
            )

    per_k = []
    for donor in range(2, 10):
        group = [cell for cell in cells if int(cell["donor_occurrence_k"]) == donor]
        per_k.append(
            {
                "receiver_occurrence_j": donor - 1,
                "donor_occurrence_k": donor,
                **_summarize(group),
                "successful_seeds": [
                    int(cell["seed"])
                    for cell in group
                    if cell["eligible"] and cell["first_successor_skip"]
                ],
            }
        )
    per_seed = []
    for seed in expected_seeds or ():
        group = [cell for cell in cells if int(cell["seed"]) == seed]
        per_seed.append({"seed": seed, **_summarize(group)})

    overall = _summarize(cells)
    overall["seed_cluster_bootstrap_95ci_first_successor_skip_rate"] = (
        _seed_cluster_bootstrap(cells)
    )
    overall["successful_cells"] = [
        {
            "seed": int(cell["seed"]),
            "receiver_occurrence_j": int(cell["receiver_occurrence_j"]),
            "donor_occurrence_k": int(cell["donor_occurrence_k"]),
            "donor_successor": int(cell["donor_successor"]),
        }
        for cell in cells
        if cell["eligible"] and cell["first_successor_skip"]
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output / "cells.jsonl", cells)
    _write_json(args.output / "per_k.json", per_k)
    _write_json(args.output / "per_seed.json", per_seed)
    _write_json(args.output / "overall.json", overall)
    _write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "gold_count": 10,
            "frozen_layer": 31,
            "donor_occurrences": list(range(2, 10)),
            "seeds": list(expected_seeds or ()),
            "primary_endpoint": (
                "first generated known city after donor_to_receiver equals N[k+1]"
            ),
            "recap_and_stopping_ignored": True,
            "eligibility_gates": [
                "matched natural geometry",
                "receiver_self candidate argmax is N[j+1]",
                "native_donor candidate argmax is N[k+1]",
                "receiver_self first generated city is N[j+1]",
            ],
        },
    )
    print(json.dumps({"overall": overall, "per_k": per_k}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
