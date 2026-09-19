#!/usr/bin/env python3
"""Emit a compact finite-value and coverage audit for V5 representation CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    paths = {
        "geometry": args.output_dir / "geometry_summary.csv",
        "regression": args.output_dir / "regression_confirmation.csv",
        "classification": args.output_dir / "classification_confirmation.csv",
    }
    payload: dict[str, object] = {"output_dir": str(args.output_dir.resolve())}
    all_sites: set[str] = set()
    all_cohorts: set[str] = set()
    for name, path in paths.items():
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include="number")
        nonfinite_by_column = {
            column: int((~np.isfinite(numeric[column].to_numpy(dtype=float))).sum())
            for column in numeric.columns
            if (~np.isfinite(numeric[column].to_numpy(dtype=float))).any()
        }
        sites = sorted(frame["site_kind"].astype(str).unique())
        cohorts = sorted(frame["cohort"].astype(str).unique())
        all_sites.update(sites)
        all_cohorts.update(cohorts)
        payload[name] = {
            "rows": int(len(frame)),
            "sites": sites,
            "cohorts": cohorts,
            "layer_min": int(frame["layer"].min()),
            "layer_max": int(frame["layer"].max()),
            "layer_count": int(frame["layer"].nunique()),
            "nonfinite_numeric_by_column": nonfinite_by_column,
        }
    payload["all_sites"] = sorted(all_sites)
    payload["all_cohorts"] = sorted(all_cohorts)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
