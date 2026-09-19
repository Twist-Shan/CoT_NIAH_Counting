#!/usr/bin/env python3
"""Summarize paper-aligned counting-mechanism transfer trials."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_realistic_niah_v5_count_stream import _atomic_json  # noqa: E402


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows:
        raise ValueError("Counting-mechanism trial file is empty")
    required = {"experiment", "condition", "seed"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Trial row lacks fields {sorted(missing)}")
    return rows


def _seed_mean_values(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> np.ndarray:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            by_seed[int(row["seed"])].append(number)
    values = np.asarray(
        [float(np.mean(items)) for _seed, items in sorted(by_seed.items())],
        dtype=float,
    )
    return values


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    values = _seed_mean_values(rows, metric)
    if values.size == 0:
        return {
            "metric": metric,
            "seed_count": 0,
            "mean": None,
            "ci95": [None, None],
            "positive_seed_fraction": None,
        }
    generator = np.random.default_rng(int(random_seed))
    if values.size == 1:
        low = high = float(values[0])
    else:
        indices = generator.integers(
            0, values.size, size=(int(bootstrap_samples), values.size)
        )
        means = values[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "metric": metric,
        "seed_count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [float(low), float(high)],
        "positive_seed_fraction": float(np.mean(values > 0)),
        "seed_values": [float(value) for value in values],
    }


def _group(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Iterable[tuple[tuple[Any, ...], list[Mapping[str, Any]]]]:
    values: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        values[tuple(row.get(key) for key in keys)].append(row)
    for key in sorted(values, key=lambda item: tuple(str(value) for value in item)):
        yield key, values[key]


def _summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    metrics: Sequence[str],
    bootstrap_samples: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group_index, (key, group_rows) in enumerate(_group(rows, keys)):
        result.append(
            {
                **{name: value for name, value in zip(keys, key)},
                "trial_count": len(group_rows),
                "seed_count": len({int(row["seed"]) for row in group_rows}),
                "metrics": {
                    metric: _metric_summary(
                        group_rows,
                        metric,
                        bootstrap_samples=bootstrap_samples,
                        random_seed=int(random_seed) + group_index * 1009 + offset,
                    )
                    for offset, metric in enumerate(metrics)
                },
            }
        )
    return result


def _flatten_continued_hops(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for readout in row.get("boundary_countscope_readouts", ()):
            result.append(
                {
                    "seed": int(row["seed"]),
                    "k": int(row["k"]),
                    "region": str(row["region"]),
                    "source_end_occurrence": int(row["source_end_occurrence"]),
                    **readout,
                }
            )
    return result


def _flatten_continued_early_stops(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for readout in row.get("early_stop_count_readouts", ()):
            result.append(
                {
                    "seed": int(row["seed"]),
                    "k": int(row["k"]),
                    "region": str(row["region"]),
                    "source_end_occurrence": int(row["source_end_occurrence"]),
                    **readout,
                }
            )
    return result


def _flatten_successors(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        readout = dict(row.get("successor_readout", {}))
        if not readout or not bool(readout.get("available", True)):
            continue
        result.append(
            {
                "seed": int(row["seed"]),
                "k": int(row["k"]),
                "region": str(row["region"]),
                "source_end_occurrence": int(row["source_end_occurrence"]),
                "candidate_donor_successor_adoption": float(
                    int(readout.get("predicted_occurrence_mean_logprob", -1))
                    == int(readout["donor_successor_occurrence"])
                ),
                "greedy_donor_successor_adoption": float(
                    bool(readout.get("greedy_donor_successor_adoption", False))
                ),
                "target_delta_mean_logprob_margin": readout.get(
                    "target_delta_mean_logprob_margin"
                ),
            }
        )
    return result


def analyze(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    by_experiment: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_experiment[str(row["experiment"])].append(row)

    countscope = by_experiment.get("countscope", [])
    continued = by_experiment.get("continued_counting", [])
    linear = by_experiment.get("linear_additivity", [])
    separator = by_experiment.get("separator_collapse", [])
    maximum = by_experiment.get("maximum_count", [])
    hop_rows = _flatten_continued_hops(continued)
    early_stop_rows = _flatten_continued_early_stops(continued)
    successor_rows = _flatten_successors(continued)

    return {
        "schema_version": "counting_mechanism_transfer_summary_v1",
        "trial_count": len(rows),
        "experiments": sorted(by_experiment),
        "countscope": _summaries(
            countscope,
            keys=("region", "donor_occurrence"),
            metrics=(
                "paper_ci",
                "target_probability",
                "target_is_candidate_argmax",
                "greedy_target_adoption",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        ),
        "countscope_by_receiver": _summaries(
            countscope,
            keys=("receiver_mode", "region", "donor_occurrence"),
            metrics=(
                "paper_ci",
                "target_probability",
                "target_is_candidate_argmax",
                "greedy_target_adoption",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 5000,
        ),
        "continued_final": _summaries(
            continued,
            keys=("region", "k", "source_end_occurrence"),
            metrics=(
                "paper_ci",
                "target_probability",
                "target_is_candidate_argmax",
                "greedy_target_adoption",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 10000,
        ),
        "continued_boundary_countscope": _summaries(
            hop_rows,
            keys=("region", "k", "source_end_occurrence", "hop_after_patch"),
            metrics=(
                "paper_ci",
                "target_probability",
                "target_is_candidate_argmax",
                "greedy_target_adoption",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 20000,
        ),
        "continued_early_stop": _summaries(
            early_stop_rows,
            keys=("region", "k", "source_end_occurrence", "hop_after_patch"),
            metrics=(
                "paper_ci",
                "target_probability",
                "target_is_candidate_argmax",
                "greedy_target_adoption",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 25000,
        ),
        "continued_successor": _summaries(
            successor_rows,
            keys=("region", "k", "source_end_occurrence"),
            metrics=(
                "candidate_donor_successor_adoption",
                "greedy_donor_successor_adoption",
                "target_delta_mean_logprob_margin",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 30000,
        ),
        "linear_additivity": _summaries(
            linear,
            keys=(
                "steering_band",
                "condition",
                "receiver_occurrence",
                "position_difference",
            ),
            metrics=(
                "paper_ci",
                "expected_count_shift_from_baseline",
                "donor_aligned_expected_shift",
                "target_is_candidate_argmax",
                "greedy_target_adoption",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 40000,
        ),
        "separator_collapse": _summaries(
            separator,
            keys=("region",),
            metrics=(
                "correct_probability_drop",
                "expected_count_shift",
                "target_is_candidate_argmax",
                "greedy_target_adoption",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 50000,
        ),
        "separator_collapse_by_dose": _summaries(
            separator,
            keys=("region", "separator_target_group_size"),
            metrics=(
                "correct_probability_drop",
                "expected_count_shift",
                "target_is_candidate_argmax",
                "greedy_target_adoption",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 55000,
        ),
        "maximum_count": _summaries(
            maximum,
            keys=("region", "k", "source_end_occurrence", "target_end_occurrence"),
            metrics=(
                "paper_ci",
                "target_probability",
                "target_is_candidate_argmax",
                "greedy_target_adoption",
            ),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + 60000,
        ),
        "interpretation_contract": {
            "strong_countscope": (
                "requires argmax or greedy donor-count adoption, not CI alone"
            ),
            "single_state_recurrence": (
                "requires k=1 boundary and final adoption plus hop-2 persistence"
            ),
            "short_trajectory_only": (
                "k=2/3 success with k=1 failure is a trajectory/ledger result"
            ),
            "linear_geometry": "steering does not establish a naturally used addition operator",
            "separator_shortcut": "marker damage must exceed payload and closing controls",
            "maximum_operator": (
                "requires max-hypothesis argmax/greedy adoption across both "
                "source<target and source>target"
            ),
        },
    }


def _mean(row: Mapping[str, Any], metric: str) -> str:
    value = row["metrics"][metric]["mean"]
    return "NA" if value is None else f"{float(value):.3f}"


def markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Native-thinking counting-mechanism transfer",
        "",
        f"Trials: {summary['trial_count']}",
        "",
        "This is exploratory. CI is reported for paper comparability; candidate "
        "argmax and greedy adoption determine causal sufficiency.",
        "",
    ]

    def table(
        title: str,
        rows: Sequence[Mapping[str, Any]],
        keys: Sequence[str],
        metrics: Sequence[str],
    ) -> None:
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend(["Not run.", ""])
            return
        headers = list(keys) + list(metrics) + ["seeds"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            values = [str(row.get(key)) for key in keys]
            values.extend(_mean(row, metric) for metric in metrics)
            values.append(str(row["seed_count"]))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    table(
        "CountScope",
        summary["countscope"],
        ("region", "donor_occurrence"),
        ("paper_ci", "target_is_candidate_argmax", "greedy_target_adoption"),
    )
    if any(row.get("receiver_mode") is not None for row in summary["countscope_by_receiver"]):
        table(
            "CountScope by receiver mode",
            summary["countscope_by_receiver"],
            ("receiver_mode", "region", "donor_occurrence"),
            ("paper_ci", "target_is_candidate_argmax", "greedy_target_adoption"),
        )
    table(
        "Continued counting: final answer",
        summary["continued_final"],
        ("region", "k", "source_end_occurrence"),
        ("paper_ci", "target_is_candidate_argmax", "greedy_target_adoption"),
    )
    table(
        "Continued counting: boundary CountScope",
        summary["continued_boundary_countscope"],
        ("region", "k", "source_end_occurrence", "hop_after_patch"),
        ("paper_ci", "target_is_candidate_argmax", "greedy_target_adoption"),
    )
    if summary["continued_early_stop"]:
        table(
            "Continued counting: immediate early-stop answer",
            summary["continued_early_stop"],
            ("region", "k", "source_end_occurrence", "hop_after_patch"),
            ("paper_ci", "target_is_candidate_argmax", "greedy_target_adoption"),
        )
    table(
        "Continued counting: native successor",
        summary["continued_successor"],
        ("region", "k", "source_end_occurrence"),
        ("candidate_donor_successor_adoption", "greedy_donor_successor_adoption"),
    )
    table(
        "Linear additivity",
        summary["linear_additivity"],
        (
            "steering_band",
            "condition",
            "receiver_occurrence",
            "position_difference",
        ),
        ("paper_ci", "donor_aligned_expected_shift", "greedy_target_adoption"),
    )
    table(
        "Separator collapse",
        summary["separator_collapse"],
        ("region",),
        ("correct_probability_drop", "expected_count_shift", "greedy_target_adoption"),
    )
    if any(
        row.get("separator_target_group_size") is not None
        for row in summary["separator_collapse_by_dose"]
    ):
        table(
            "Separator collapse by dose",
            summary["separator_collapse_by_dose"],
            ("region", "separator_target_group_size"),
            (
                "correct_probability_drop",
                "expected_count_shift",
                "greedy_target_adoption",
            ),
        )
    table(
        "Maximum latent count",
        summary["maximum_count"],
        ("region", "k", "source_end_occurrence", "target_end_occurrence"),
        ("paper_ci", "target_is_candidate_argmax", "greedy_target_adoption"),
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260826)
    args = parser.parse_args()
    rows = _read(args.trials)
    summary = analyze(
        rows,
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "summary.json", summary)
    (args.output / "SUMMARY.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"trial_count": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
