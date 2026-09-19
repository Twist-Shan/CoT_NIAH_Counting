#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "realistic_niah_v5_greedy_head_damage_aggregate_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm(values: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    adjusted = np.full(len(raw), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(raw))
    order = finite[np.argsort(raw[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, raw[index] * (len(order) - rank)))
        adjusted[index] = running
    return adjusted


def aggregate(paths: list[Path], output_dir: Path) -> dict[str, object]:
    frames = []
    inputs = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_statistics"] = str(path.resolve())
        frames.append(frame)
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path)})
    combined = pd.concat(frames, ignore_index=True)
    identity = ["model_label", "family", "bank_size", "metric"]
    if combined.duplicated(identity).any():
        duplicates = combined.loc[combined.duplicated(identity, keep=False), identity]
        raise ValueError(f"Duplicate aggregate tests:\n{duplicates.to_string(index=False)}")
    combined["holm_family_size"] = 0
    combined["holm_p_within_family_endpoint"] = np.nan
    for (_family, _metric), indices in combined.groupby(
        ["family", "metric"], sort=True
    ).groups.items():
        combined.loc[indices, "holm_family_size"] = int(len(indices))
        combined.loc[indices, "holm_p_within_family_endpoint"] = holm(
            combined.loc[indices, "sign_flip_p"]
        )
    combined["pointwise_ci_excludes_zero_harmful"] = combined["ci95_low"] > 0
    combined["pointwise_p_le_0_05"] = combined["sign_flip_p"] <= 0.05
    combined["holm_p_le_0_05"] = combined["holm_p_within_family_endpoint"] <= 0.05
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "combined_statistics_holm.csv"
    combined.sort_values(identity).to_csv(output, index=False)
    summary = {
        "schema_version": SCHEMA,
        "status": "passed",
        "inputs": inputs,
        "multiplicity": (
            "Holm correction separately within each mechanism/query-variant "
            "and endpoint across all constructible model x registered-K tests"
        ),
        "tests": int(len(combined)),
        "families": {
            f"{family}::{metric}": int(len(frame))
            for (family, metric), frame in combined.groupby(["family", "metric"])
        },
        "holm_significant": int(combined["holm_p_le_0_05"].sum()),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.statistics, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
