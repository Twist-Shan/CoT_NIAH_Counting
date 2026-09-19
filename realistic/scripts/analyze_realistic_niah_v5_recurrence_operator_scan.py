#!/usr/bin/env python3
"""Rank recurrence-operator families on causal transition-scan outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BOOTSTRAP_METRICS = (
    "current_target_rate",
    "next_target_rate",
    "reset_accuracy",
    "plus_one_from_realized_accuracy",
    "plus_one_given_current_target_accuracy",
    "identity_accuracy",
    "ols_retention",
    "iv_retention",
)


def read_trials(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError("No recurrence-operator trials were found")
    return rows


def validate_trials(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys: set[tuple[int, int, int, str]] = set()
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["receiver"]),
            int(row["donor"]),
            str(row["carrier"]),
        )
        if key in keys:
            raise ValueError(f"Duplicate operator-scan cell {key}")
        keys.add(key)
    carriers = sorted({str(row["carrier"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    incomplete: dict[str, Any] = {}
    for carrier in carriers:
        active = [row for row in rows if str(row["carrier"]) == carrier]
        cells_by_seed = {
            seed: {
                (int(row["receiver"]), int(row["donor"]))
                for row in active
                if int(row["seed"]) == seed
            }
            for seed in seeds
        }
        expected = cells_by_seed[seeds[0]]
        changed = {
            seed: sorted(cells ^ expected)
            for seed, cells in cells_by_seed.items()
            if cells != expected
        }
        if changed:
            incomplete[carrier] = changed
    if incomplete:
        raise ValueError(f"Unbalanced seed panels: {incomplete}")
    return {
        "row_count": len(rows),
        "seed_count": len(seeds),
        "carriers": carriers,
        "receivers": sorted({int(row["receiver"]) for row in rows}),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def carrier_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Carrier metrics require at least one trial")
    current_prediction = np.asarray(
        [int(row["current_prediction"]) for row in rows], dtype=np.int64
    )
    next_prediction = np.asarray(
        [int(row["next_prediction"]) for row in rows], dtype=np.int64
    )
    clean_next_prediction = np.asarray(
        [int(row["clean_next_prediction"]) for row in rows], dtype=np.int64
    )
    donor = np.asarray([int(row["donor"]) for row in rows], dtype=np.int64)
    dose = np.asarray([float(row["dose"]) for row in rows], dtype=np.float64)
    delta_current = np.asarray(
        [float(row["current_shift"]) for row in rows], dtype=np.float64
    )
    delta_next = np.asarray(
        [float(row["next_shift"]) for row in rows], dtype=np.float64
    )
    plus_one_prediction = np.minimum(current_prediction + 1, 10)
    identity_prediction = current_prediction
    current_target_mask = current_prediction == donor
    ols = _safe_ratio(
        float(delta_current @ delta_next), float(delta_current @ delta_current)
    )
    iv = _safe_ratio(float(dose @ delta_next), float(dose @ delta_current))
    return {
        "trial_count": float(len(rows)),
        "current_target_rate": float(np.mean(current_prediction == donor)),
        "next_target_rate": float(np.mean(next_prediction == donor + 1)),
        "reset_accuracy": float(np.mean(next_prediction == clean_next_prediction)),
        "plus_one_from_realized_accuracy": float(
            np.mean(next_prediction == plus_one_prediction)
        ),
        "plus_one_given_current_target_accuracy": float(
            np.mean(next_prediction[current_target_mask] == donor[current_target_mask] + 1)
        )
        if np.any(current_target_mask)
        else float("nan"),
        "identity_accuracy": float(np.mean(next_prediction == identity_prediction)),
        "ols_retention": ols,
        "iv_retention": iv,
        "mean_abs_current_shift": float(np.mean(np.abs(delta_current))),
        "mean_abs_next_shift": float(np.mean(np.abs(delta_next))),
    }


def _fit_linear(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("Linear operator fit received inconsistent arrays")
    gram = x.T @ x + 1e-8 * np.eye(x.shape[1], dtype=np.float64)
    return np.linalg.solve(gram, x.T @ y)


def _model_predictions(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    train_dx = np.asarray(
        [float(row["current_shift"]) for row in train], dtype=np.float64
    )
    train_dy = np.asarray(
        [float(row["next_shift"]) for row in train], dtype=np.float64
    )
    test_dx = np.asarray(
        [float(row["current_shift"]) for row in test], dtype=np.float64
    )
    test_clean_next = np.asarray(
        [float(row["clean_next_soft"]) for row in test], dtype=np.float64
    )
    test_current = np.asarray(
        [float(row["current_soft"]) for row in test], dtype=np.float64
    )
    train_receiver = np.asarray(
        [int(row["receiver"]) for row in train], dtype=np.int64
    )
    test_receiver = np.asarray(
        [int(row["receiver"]) for row in test], dtype=np.int64
    )
    train_dose = np.asarray(
        [float(row["dose"]) for row in train], dtype=np.float64
    )
    test_dose = np.asarray(
        [float(row["dose"]) for row in test], dtype=np.float64
    )

    rho = _safe_ratio(float(train_dx @ train_dy), float(train_dx @ train_dx))
    if not np.isfinite(rho):
        rho = 0.0
    position_prediction = np.empty_like(test_dx)
    for receiver in sorted(set(int(value) for value in train_receiver)):
        train_mask = train_receiver == receiver
        test_mask = test_receiver == receiver
        local_rho = _safe_ratio(
            float(train_dx[train_mask] @ train_dy[train_mask]),
            float(train_dx[train_mask] @ train_dx[train_mask]),
        )
        if not np.isfinite(local_rho):
            local_rho = 0.0
        position_prediction[test_mask] = test_clean_next[test_mask] + local_rho * test_dx[
            test_mask
        ]

    quadratic_beta = _fit_linear(
        np.column_stack((train_dx, train_dx * train_dx)), train_dy
    )
    dose_beta = _fit_linear(
        np.column_stack((train_dx, train_dose)), train_dy
    )
    lookup = {
        (int(receiver), int(donor)): float(
            np.mean(
                [
                    float(row["next_shift"])
                    for row in train
                    if int(row["receiver"]) == int(receiver)
                    and int(row["donor"]) == int(donor)
                ]
            )
        )
        for receiver in sorted({int(row["receiver"]) for row in train})
        for donor in sorted(
            {
                int(row["donor"])
                for row in train
                if int(row["receiver"]) == int(receiver)
            }
        )
    }
    lookup_delta = np.asarray(
        [lookup[(int(row["receiver"]), int(row["donor"]))] for row in test],
        dtype=np.float64,
    )
    return {
        "reset_to_clean_next": test_clean_next,
        "plus_one": np.clip(test_current + 1.0, 1.0, 10.0),
        "identity": test_current,
        "global_leaky_reset": test_clean_next + rho * test_dx,
        "position_leaky_reset": position_prediction,
        "quadratic_reset": test_clean_next
        + np.column_stack((test_dx, test_dx * test_dx)) @ quadratic_beta,
        "dose_conditioned_affine_reset": test_clean_next
        + np.column_stack((test_dx, test_dose)) @ dose_beta,
        "donor_lookup_reset": test_clean_next + lookup_delta,
    }


def leave_one_seed_out_rmse(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    predictions: dict[str, list[float]] = {}
    targets: list[float] = []
    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        train = [row for row in rows if int(row["seed"]) != seed]
        test = [row for row in rows if int(row["seed"]) == seed]
        target = [float(row["next_soft"]) for row in test]
        targets.extend(target)
        for label, values in _model_predictions(train, test).items():
            predictions.setdefault(label, []).extend(values.tolist())
    observed = np.asarray(targets, dtype=np.float64)
    return {
        label: float(
            np.sqrt(np.mean((observed - np.asarray(values, dtype=np.float64)) ** 2))
        )
        for label, values in predictions.items()
    }


def _interval(values: Sequence[float]) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return [None, None]
    low, high = np.quantile(finite, (0.025, 0.975))
    return [float(low), float(high)]


def _transition_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cells = sorted({(int(row["receiver"]), int(row["donor"])) for row in rows})
    for receiver, donor in cells:
        active = [
            row
            for row in rows
            if int(row["receiver"]) == receiver and int(row["donor"]) == donor
        ]
        counts: dict[int, int] = {}
        for row in active:
            value = int(row["next_prediction"])
            counts[value] = counts.get(value, 0) + 1
        mode = max(sorted(counts), key=lambda value: counts[value])
        output.append(
            {
                "receiver": receiver,
                "donor": donor,
                "trial_count": len(active),
                "modal_next_prediction": mode,
                "modal_next_count": counts[mode],
                "next_prediction_counts": {
                    str(key): counts[key] for key in sorted(counts)
                },
                "current_target_rate": float(
                    np.mean(
                        [int(row["current_prediction"]) == donor for row in active]
                    )
                ),
                "reset_rate": float(
                    np.mean(
                        [
                            int(row["next_prediction"])
                            == int(row["clean_next_prediction"])
                            for row in active
                        ]
                    )
                ),
                "plus_one_target_rate": float(
                    np.mean(
                        [int(row["next_prediction"]) == donor + 1 for row in active]
                    )
                ),
                "mean_current_shift": float(
                    np.mean([float(row["current_shift"]) for row in active])
                ),
                "mean_next_shift": float(
                    np.mean([float(row["next_shift"]) for row in active])
                ),
            }
        )
    return output


def analyze_operator_scan(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    design = validate_trials(rows)
    if int(bootstrap_draws) < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    rng = np.random.default_rng(int(bootstrap_seed))
    output: dict[str, Any] = {}
    for carrier in design["carriers"]:
        active = [row for row in rows if str(row["carrier"]) == carrier]
        point = carrier_metrics(active)
        seeds = tuple(sorted({int(row["seed"]) for row in active}))
        by_seed = {
            seed: [row for row in active if int(row["seed"]) == seed]
            for seed in seeds
        }
        draws = {metric: [] for metric in BOOTSTRAP_METRICS}
        for _draw in range(int(bootstrap_draws)):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            sampled = [row for seed in sampled_seeds for row in by_seed[int(seed)]]
            metrics = carrier_metrics(sampled)
            for metric in BOOTSTRAP_METRICS:
                draws[metric].append(metrics[metric])
        discrete = {
            "reset_to_clean_next": point["reset_accuracy"],
            "plus_one_target": point["next_target_rate"],
            "plus_one_from_realized": point[
                "plus_one_from_realized_accuracy"
            ],
            "plus_one_given_current_target": point[
                "plus_one_given_current_target_accuracy"
            ],
            "identity": point["identity_accuracy"],
        }
        for shift in range(-4, 5):
            accuracy = float(
                np.mean(
                    [
                        int(row["next_prediction"])
                        == min(max(int(row["current_prediction"]) + shift, 1), 10)
                        for row in active
                    ]
                )
            )
            discrete[f"clipped_shift_{shift:+d}"] = accuracy
        output[carrier] = {
            "metrics": point,
            "strata": {
                "by_receiver": {
                    str(receiver): carrier_metrics(
                        [
                            row
                            for row in active
                            if int(row["receiver"]) == int(receiver)
                        ]
                    )
                    for receiver in sorted(
                        {int(row["receiver"]) for row in active}
                    )
                },
                "by_absolute_dose": {
                    str(absolute_dose): carrier_metrics(
                        [
                            row
                            for row in active
                            if abs(int(row["dose"])) == int(absolute_dose)
                        ]
                    )
                    for absolute_dose in sorted(
                        {abs(int(row["dose"])) for row in active}
                    )
                },
            },
            "cluster_bootstrap_95ci": {
                metric: _interval(draws[metric]) for metric in BOOTSTRAP_METRICS
            },
            "discrete_operator_accuracy": dict(
                sorted(discrete.items(), key=lambda item: (-item[1], item[0]))
            ),
            "leave_one_seed_out_soft_rmse": dict(
                sorted(
                    leave_one_seed_out_rmse(active).items(),
                    key=lambda item: (item[1], item[0]),
                )
            ),
            "transition_table": _transition_table(active),
        }
    return {
        "schema_version": "recurrence_operator_scan_analysis_v1",
        "design": design,
        "bootstrap": {
            "cluster": "seed",
            "draws": int(bootstrap_draws),
            "seed": int(bootstrap_seed),
            "interval": "percentile_95",
        },
        "carriers": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()
    analysis = analyze_operator_scan(
        read_trials(args.trials),
        bootstrap_draws=int(args.bootstrap_draws),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                carrier: {
                    "metrics": payload["metrics"],
                    "discrete_operator_accuracy": payload[
                        "discrete_operator_accuracy"
                    ],
                    "leave_one_seed_out_soft_rmse": payload[
                        "leave_one_seed_out_soft_rmse"
                    ],
                }
                for carrier, payload in analysis["carriers"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
