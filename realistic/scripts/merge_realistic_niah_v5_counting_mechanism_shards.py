#!/usr/bin/env python3
"""Merge seed-disjoint counting-mechanism shards with config-derived audits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


EXPERIMENTS = (
    "countscope",
    "continued_counting",
    "linear_additivity",
    "separator_collapse",
    "maximum_count",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as source:
        return [json.loads(line) for line in source if line.strip()]


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _expected_counts(config: Mapping[str, Any], seed: int) -> dict[str, int]:
    experiments = config["experiments"]
    fixed_count = int(config["cohort_contract"]["fixed_count"])
    result: dict[str, int] = {}
    countscope = experiments["countscope"]
    result["countscope"] = len(countscope["donor_occurrences"]) * len(
        countscope["regions"]
    )
    continued = experiments["continued_counting"]
    result["continued_counting"] = len(continued["regions"]) * sum(
        int(source) >= int(width) and int(width) < fixed_count
        for source in continued["source_end_occurrences"]
        for width in continued["k_values"]
    )
    geometry = experiments["linear_additivity"]
    if int(seed) in {int(value) for value in geometry["eval_seeds"]}:
        valid_shifts = sum(
            1 <= int(receiver) + int(shift) <= fixed_count
            for receiver in geometry["receiver_occurrences"]
            for shift in geometry["shifts"]
        )
        result["linear_additivity"] = (
            valid_shifts
            * len(geometry["layer_bands"])
            * len(geometry["conditions"])
        )
    else:
        result["linear_additivity"] = 0
    result["separator_collapse"] = len(
        experiments["separator_collapse"]["regions"]
    )
    maximum = experiments["maximum_count"]
    result["maximum_count"] = len(maximum["regions"]) * sum(
        int(source) >= int(width) and int(target) >= int(width)
        for source, target in maximum["source_target_pairs"]
        for width in maximum["k_values"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--tail-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_seeds = {int(value) for value in config["seeds"]}
    tail_seeds = {int(value) for value in args.tail_seeds}
    if not tail_seeds < expected_seeds:
        raise ValueError("Tail seeds must be a proper subset of the config panel")
    base_trials = [
        row
        for row in _read_jsonl(args.base / "trials.jsonl")
        if int(row["seed"]) not in tail_seeds
    ]
    tail_trials = [
        row
        for row in _read_jsonl(args.tail / "trials.jsonl")
        if int(row["seed"]) in tail_seeds
    ]
    base_skips = [
        row
        for row in _read_jsonl(args.base / "skipped_trials.jsonl")
        if int(row["seed"]) not in tail_seeds
    ]
    tail_skips = [
        row
        for row in _read_jsonl(args.tail / "skipped_trials.jsonl")
        if int(row["seed"]) in tail_seeds
    ]
    trials = sorted(
        base_trials + tail_trials,
        key=lambda row: (
            int(row["seed"]),
            str(row["experiment"]),
            str(row.get("condition", "")),
            str(row.get("region", "")),
            int(row.get("source_end_occurrence", -1)),
            int(row.get("target_end_occurrence", -1)),
            int(row.get("receiver_occurrence", -1)),
            int(row.get("position_difference", 0)),
            int(row.get("k", -1)),
        ),
    )
    skips = sorted(
        base_skips + tail_skips,
        key=lambda row: (int(row["seed"]), str(row["experiment"])),
    )
    observed_seeds = {int(row["seed"]) for row in trials + skips}
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"Merged seed panel differs: {sorted(observed_seeds)}"
        )
    trial_counts = Counter(
        (int(row["seed"]), str(row["experiment"])) for row in trials
    )
    skip_counts = Counter(
        (int(row["seed"]), str(row["experiment"])) for row in skips
    )
    for seed in sorted(expected_seeds):
        expected = _expected_counts(config, seed)
        for experiment in EXPERIMENTS:
            total = trial_counts[(seed, experiment)] + skip_counts[(seed, experiment)]
            if total != expected[experiment]:
                raise ValueError(
                    "Merged cell-count mismatch: "
                    f"seed={seed} experiment={experiment} "
                    f"observed={total} expected={expected[experiment]}"
                )
    _atomic_jsonl(args.output / "trials.jsonl", trials)
    _atomic_jsonl(args.output / "skipped_trials.jsonl", skips)
    manifest = {
        "status": "PASS",
        "schema_version": "counting_mechanism_transfer_shard_merge_v1",
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "base_path": str(args.base.resolve()),
        "tail_path": str(args.tail.resolve()),
        "tail_seeds": sorted(tail_seeds),
        "seed_count": len(expected_seeds),
        "trial_count": len(trials),
        "skipped_trial_count": len(skips),
        "experiment_row_counts": {
            experiment: sum(row["experiment"] == experiment for row in trials)
            for experiment in EXPERIMENTS
        },
        "all_config_derived_seed_cell_counts_match": True,
        "overlap_policy": "tail seeds exclusively sourced from tail shard",
    }
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
