#!/usr/bin/env python3
"""Analyze unified count-carrier trials with a seed-cluster bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PRIMARY_METRICS = (
    "mean_signed_current_shift",
    "mean_signed_next_shift",
    "current_to_next_slope",
    "mean_donor_vs_receiver_mean_logodds_change",
)

CONTRASTS = {
    "distributed_minus_residual": (
        "residual_count_plus_kv",
        "residual_count_subspace",
    ),
    "residual_minus_residual_orthogonal": (
        "residual_count_subspace",
        "residual_count_subspace_orthogonal",
    ),
    "distributed_minus_distributed_orthogonal": (
        "residual_count_plus_kv",
        "residual_count_plus_kv_orthogonal",
    ),
    "whole_minus_residual": (
        "whole_state",
        "residual_count_subspace",
    ),
}


def read_trials(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Trial line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError("No unified-carrier trials were found")
    return rows


def validate_balanced_trials(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require one complete carrier-by-dose panel for every seed/receiver unit."""

    carriers = tuple(sorted({str(row["carrier"]) for row in rows}))
    doses = tuple(sorted({int(row["dose"]) for row in rows}))
    if doses != (-1, 1):
        raise ValueError("Unified transition analysis requires doses -1 and +1")
    expected = {(dose, carrier) for dose in doses for carrier in carriers}
    units: dict[tuple[int, int], set[tuple[int, str]]] = {}
    keys: set[tuple[int, int, int, str]] = set()
    for row in rows:
        seed = int(row["seed"])
        receiver = int(row["receiver"])
        dose = int(row["dose"])
        carrier = str(row["carrier"])
        key = (seed, receiver, dose, carrier)
        if key in keys:
            raise ValueError(f"Duplicate trial cell {key}")
        keys.add(key)
        units.setdefault((seed, receiver), set()).add((dose, carrier))
    incomplete = {unit: sorted(expected - cells) for unit, cells in units.items() if cells != expected}
    if incomplete:
        raise ValueError(f"Incomplete carrier panels: {incomplete}")
    return {
        "row_count": len(rows),
        "seed_count": len({int(row["seed"]) for row in rows}),
        "unit_count": len(units),
        "doses": list(doses),
        "carriers": list(carriers),
    }


def _through_origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(x @ x)
    if denominator <= 1e-12:
        return float("nan")
    return float((x @ y) / denominator)


def carrier_metrics(
    rows: Sequence[Mapping[str, Any]], carrier: str
) -> dict[str, float]:
    active = [row for row in rows if str(row["carrier"]) == str(carrier)]
    if not active:
        raise ValueError(f"No rows for carrier {carrier}")
    dose = np.asarray([float(row["dose"]) for row in active], dtype=np.float64)
    current = np.asarray(
        [float(row["current_shift"]) for row in active], dtype=np.float64
    )
    later = np.asarray(
        [float(row["next_shift"]) for row in active], dtype=np.float64
    )
    pair_logodds = np.asarray(
        [
            float(row["donor_vs_receiver_mean_logodds_change"])
            for row in active
        ],
        dtype=np.float64,
    )
    return {
        "trial_count": float(len(active)),
        "mean_abs_current_shift": float(np.mean(np.abs(current))),
        "mean_abs_next_shift": float(np.mean(np.abs(later))),
        "mean_signed_current_shift": float(np.mean(dose * current)),
        "mean_signed_next_shift": float(np.mean(dose * later)),
        "current_to_next_slope": _through_origin_slope(current, later),
        "current_shift_rms": float(np.sqrt(np.mean(current * current))),
        "current_sign_rate": float(np.mean(dose * current > 0.0)),
        "next_sign_rate": float(np.mean(dose * later > 0.0)),
        "current_exact_rate": float(
            np.mean([bool(row["current_exact"]) for row in active])
        ),
        "next_exact_rate": float(
            np.mean([bool(row["next_exact"]) for row in active])
        ),
        "mean_donor_vs_receiver_mean_logodds_change": float(
            np.mean(pair_logodds)
        ),
    }


def _interval(values: Sequence[float]) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return [None, None]
    low, high = np.quantile(finite, (0.025, 0.975))
    return [float(low), float(high)]


def cluster_bootstrap_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Bootstrap complete seed clusters and return carrier/contrast intervals."""

    design = validate_balanced_trials(rows)
    if int(draws) < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    by_seed = {
        seed: [row for row in rows if int(row["seed"]) == seed] for seed in seeds
    }
    carriers = tuple(str(value) for value in design["carriers"])
    point = {carrier: carrier_metrics(rows, carrier) for carrier in carriers}
    carrier_draws = {
        carrier: {metric: [] for metric in PRIMARY_METRICS} for carrier in carriers
    }
    active_contrasts = {
        label: pair
        for label, pair in CONTRASTS.items()
        if pair[0] in carriers and pair[1] in carriers
    }
    contrast_draws = {
        label: {metric: [] for metric in PRIMARY_METRICS}
        for label in active_contrasts
    }
    rng = np.random.default_rng(int(bootstrap_seed))
    for _draw in range(int(draws)):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled = [row for seed in sampled_seeds for row in by_seed[int(seed)]]
        sampled_metrics = {
            carrier: carrier_metrics(sampled, carrier) for carrier in carriers
        }
        for carrier in carriers:
            for metric in PRIMARY_METRICS:
                carrier_draws[carrier][metric].append(
                    sampled_metrics[carrier][metric]
                )
        for label, (left, right) in active_contrasts.items():
            for metric in PRIMARY_METRICS:
                contrast_draws[label][metric].append(
                    sampled_metrics[left][metric] - sampled_metrics[right][metric]
                )

    carrier_output: dict[str, Any] = {}
    for carrier in carriers:
        carrier_output[carrier] = {
            **point[carrier],
            "cluster_bootstrap_95ci": {
                metric: _interval(carrier_draws[carrier][metric])
                for metric in PRIMARY_METRICS
            },
        }
    contrast_output: dict[str, Any] = {}
    for label, (left, right) in active_contrasts.items():
        contrast_output[label] = {
            "left": left,
            "right": right,
            "difference": {
                metric: float(point[left][metric] - point[right][metric])
                for metric in PRIMARY_METRICS
            },
            "cluster_bootstrap_95ci": {
                metric: _interval(contrast_draws[label][metric])
                for metric in PRIMARY_METRICS
            },
        }
    return {
        "schema_version": "unified_carrier_transition_analysis_v1",
        "design": design,
        "bootstrap": {
            "cluster": "seed",
            "draws": int(draws),
            "seed": int(bootstrap_seed),
            "interval": "percentile_95",
        },
        "primary_metrics": list(PRIMARY_METRICS),
        "carriers": carrier_output,
        "contrasts": contrast_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()
    analysis = cluster_bootstrap_analysis(
        read_trials(args.trials),
        draws=int(args.bootstrap_draws),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(analysis["carriers"], sort_keys=True))


if __name__ == "__main__":
    main()
