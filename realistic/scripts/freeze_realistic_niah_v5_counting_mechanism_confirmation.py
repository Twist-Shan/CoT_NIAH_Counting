#!/usr/bin/env python3
"""Freeze one Geometry layer band and emit untouched-confirmation configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


CONDITIONS = (
    "position_difference",
    "opposite_position_difference",
    "norm_matched_orthogonal",
)
SELECTION_METRICS = (
    "paper_ci",
    "donor_aligned_expected_shift",
    "target_is_candidate_argmax",
    "greedy_target_adoption",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _read_trials(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    linear = [row for row in rows if row.get("experiment") == "linear_additivity"]
    if not linear:
        raise ValueError("Discovery trials contain no linear-additivity rows")
    return linear


def _as_number(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    return float(value)


def score_geometry_bands(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_eval_seeds: Sequence[int],
) -> tuple[str, list[dict[str, Any]]]:
    expected_seeds = {int(value) for value in expected_eval_seeds}
    observed_seeds = {int(row["seed"]) for row in rows}
    if observed_seeds != expected_seeds:
        raise ValueError(
            "Geometry discovery seeds differ from the frozen heldout panel: "
            f"observed={sorted(observed_seeds)} expected={sorted(expected_seeds)}"
        )
    grouped: dict[str, dict[tuple[int, int, int], dict[str, Mapping[str, Any]]]] = {}
    for row in rows:
        band = str(row.get("steering_band", ""))
        if not band:
            raise ValueError("Geometry row lacks steering_band")
        cell = (
            int(row["seed"]),
            int(row["receiver_occurrence"]),
            int(row["position_difference"]),
        )
        condition = str(row["condition"])
        by_condition = grouped.setdefault(band, {}).setdefault(cell, {})
        if condition in by_condition:
            raise ValueError(f"Duplicate geometry condition: {band} {cell} {condition}")
        by_condition[condition] = row
    if not grouped:
        raise ValueError("No Geometry layer bands were observed")
    reference_cells: set[tuple[int, int, int]] | None = None
    scores: list[dict[str, Any]] = []
    for band in sorted(grouped):
        cells = grouped[band]
        active_cells = set(cells)
        if reference_cells is None:
            reference_cells = active_cells
        elif active_cells != reference_cells:
            raise ValueError("Geometry bands do not cover identical heldout cells")
        for cell, by_condition in cells.items():
            if set(by_condition) != set(CONDITIONS):
                raise ValueError(
                    f"Geometry cell lacks registered controls: {band} {cell}"
                )
        metric_rows: dict[str, Any] = {}
        rank_values: list[float] = []
        for metric in SELECTION_METRICS:
            real_values: list[float] = []
            opposite_values: list[float] = []
            orthogonal_values: list[float] = []
            contrasts: list[float] = []
            for by_condition in cells.values():
                real = _as_number(by_condition["position_difference"][metric])
                opposite = _as_number(
                    by_condition["opposite_position_difference"][metric]
                )
                orthogonal = _as_number(
                    by_condition["norm_matched_orthogonal"][metric]
                )
                real_values.append(real)
                opposite_values.append(opposite)
                orthogonal_values.append(orthogonal)
                contrasts.append(real - 0.5 * (opposite + orthogonal))
            contrast_mean = float(mean(contrasts))
            metric_rows[metric] = {
                "directional_contrast_mean": contrast_mean,
                "real_mean": float(mean(real_values)),
                "opposite_mean": float(mean(opposite_values)),
                "orthogonal_mean": float(mean(orthogonal_values)),
            }
            rank_values.append(contrast_mean)
        scores.append(
            {
                "steering_band": band,
                "cell_count": len(cells),
                "seed_count": len({cell[0] for cell in cells}),
                "metrics": metric_rows,
                "lexicographic_rank_values": rank_values,
            }
        )
    winner = max(
        scores,
        key=lambda value: (
            *tuple(float(item) for item in value["lexicographic_rank_values"]),
            str(value["steering_band"]),
        ),
    )["steering_band"]
    return str(winner), scores


def freeze_confirmation_config(
    discovery_config: Mapping[str, Any],
    *,
    selected_band: str,
    selection_audit: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(discovery_config))
    if not str(result.get("phase", "")).endswith("_discovery"):
        raise ValueError("Input config is not a discovery plan")
    discovery_seeds = tuple(int(value) for value in result["seeds"])
    contract = result["cohort_contract"]
    confirmation_seeds = tuple(
        int(value) for value in contract["confirmation_seeds_reserved"]
    )
    if set(discovery_seeds) & set(confirmation_seeds):
        raise ValueError("Discovery and confirmation seed panels overlap")
    geometry = result["experiments"]["linear_additivity"]
    bands = geometry["layer_bands"]
    if selected_band not in bands:
        raise ValueError("Selected layer band is absent from discovery config")
    result["phase"] = str(result["phase"]).removesuffix("_discovery") + "_confirmation"
    result["status"] = "geometry_band_frozen_before_confirmation_outcomes"
    result["seeds"] = list(confirmation_seeds)
    for name, spec in result["experiments"].items():
        if name == "linear_additivity":
            spec["fit_seeds"] = list(discovery_seeds)
            spec["eval_seeds"] = list(confirmation_seeds)
            spec["layer_bands"] = {selected_band: list(bands[selected_band])}
        else:
            spec["seeds"] = list(confirmation_seeds)
    contract["confirmation_seed_count"] = len(confirmation_seeds)
    contract["discovery_seeds_used_for_confirmation_geometry_fit"] = list(
        discovery_seeds
    )
    contract["confirmation_outcomes_accessed_during_freeze"] = False
    result["geometry_band_freeze"] = dict(selection_audit)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--discovery-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    discovery = _read_json(args.discovery_config)
    geometry = discovery["experiments"]["linear_additivity"]
    rows = _read_trials(args.trials)
    selected, scores = score_geometry_bands(
        rows,
        expected_eval_seeds=tuple(int(value) for value in geometry["eval_seeds"]),
    )
    trials_sha = hashlib.sha256(args.trials.read_bytes()).hexdigest()
    audit = {
        "status": "PASS",
        "selection_population": "heldout_discovery_geometry_cells",
        "selection_uses_confirmation_outcomes": False,
        "selection_rule": (
            "lexicographically maximize across-band means of paired real minus "
            "one-half(opposite+orthogonal) for paper_ci, donor-aligned expected "
            "shift, candidate argmax adoption, then greedy adoption"
        ),
        "metric_priority": list(SELECTION_METRICS),
        "selected_band": selected,
        "band_scores": scores,
        "discovery_trials_sha256": trials_sha,
    }
    frozen = freeze_confirmation_config(
        discovery,
        selected_band=selected,
        selection_audit=audit,
    )
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected_band": selected, "status": "PASS"}))


if __name__ == "__main__":
    main()
