#!/usr/bin/env python3
"""Seed-level inference for the unindexed full-commit successor assay."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_count_stream import _atomic_json  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    statistics = [
        abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def _bootstrap_ci(
    values: np.ndarray, *, samples: int, random_seed: int
) -> tuple[float, float]:
    generator = np.random.default_rng(int(random_seed))
    indices = generator.integers(0, len(values), size=(int(samples), len(values)))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.DataFrame(_read_jsonl(args.input))
    required = {"estimand", "seed", "effect", "pair_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pair contrasts lack columns {missing}")
    results: list[dict[str, Any]] = []
    for index, (estimand, group) in enumerate(
        frame.groupby("estimand", sort=True), start=1
    ):
        seed_effects = group.groupby("seed", sort=True)["effect"].mean()
        values = seed_effects.to_numpy(float)
        if len(values) != 10:
            raise ValueError(f"{estimand} has {len(values)} seeds, expected 10")
        low, high = _bootstrap_ci(
            values,
            samples=int(args.bootstrap_samples),
            random_seed=int(args.random_seed) + index,
        )
        results.append(
            {
                "estimand": str(estimand),
                "mean_effect": float(np.mean(values)),
                "ci_low": low,
                "ci_high": high,
                "exact_sign_flip_p_two_sided": _exact_sign_flip(values),
                "seed_count": int(len(values)),
                "pair_count": int(group["pair_sha256"].nunique()),
                "positive_seed_fraction": float(np.mean(values > 0)),
                "seed_effects": {
                    str(seed): float(value) for seed, value in seed_effects.items()
                },
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "unindexed_full_commit_successor_analysis_v1",
        "status": "PASS",
        "bootstrap_samples": int(args.bootstrap_samples),
        "random_seed": int(args.random_seed),
        "independent_unit": "seed_mean_over_two_donor_offsets",
        "estimands": results,
    }
    _atomic_json(args.output / "estimands.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
