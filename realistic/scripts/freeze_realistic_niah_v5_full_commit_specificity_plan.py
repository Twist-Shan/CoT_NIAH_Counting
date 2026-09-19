#!/usr/bin/env python3
"""Freeze outcome-blind natural-donor controls for the 5.3 extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.native_loop import (  # noqa: E402
    choose_shuffled_commit_donor_occurrence,
)


SCHEMA_VERSION = "realistic_niah_v5_full_commit_specificity_plan_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def freeze_plan(
    frame: pd.DataFrame,
    *,
    donor_offsets: tuple[int, ...],
    random_seed: int,
) -> pd.DataFrame:
    required = {
        "panel_kind",
        "pair_sha256",
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "selection_rank_used",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Native-loop plan lacks columns: {missing}")
    used = frame["selection_rank_used"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes"}
    )
    if used.any() or "selection_rank" in frame.columns:
        raise ValueError("Specificity plan cannot use selection rank")
    work = frame.loc[
        frame["panel_kind"].astype(str).eq("p0_local")
        & pd.to_numeric(frame["donor_offset"], errors="raise")
        .astype(int)
        .isin(donor_offsets)
    ].copy()
    if work.empty:
        raise ValueError("No registered P0 pairs remain for specificity")
    if work["pair_sha256"].duplicated().any():
        raise ValueError("Native-loop plan contains duplicate pair hashes")

    shuffled_occurrences: list[int] = []
    specificity_hashes: list[str] = []
    for row in work.itertuples(index=False):
        seed = int(row.seed)
        shuffled = choose_shuffled_commit_donor_occurrence(
            gold_count=int(row.gold_count),
            receiver_occurrence=int(row.receiver_occurrence),
            donor_occurrence=int(row.donor_occurrence),
            random_seed=int(random_seed) + seed * 1009,
        )
        shuffled_occurrences.append(shuffled)
        specificity_hashes.append(
            _sha256_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "native_pair_sha256": str(row.pair_sha256),
                    "shuffled_donor_occurrence": shuffled,
                    "random_seed": int(random_seed),
                }
            )
        )
    work["shuffled_donor_occurrence"] = shuffled_occurrences
    work["shuffled_donor_offset"] = (
        work["shuffled_donor_occurrence"].astype(int)
        - work["receiver_occurrence"].astype(int)
    )
    work["shuffled_absolute_distance_matched"] = (
        work["shuffled_donor_offset"].abs()
        == work["donor_offset"].astype(int).abs()
    )
    work["specificity_schema_version"] = SCHEMA_VERSION
    work["specificity_pair_sha256"] = specificity_hashes
    work["specificity_outcome_blind"] = True
    return work.sort_values(
        ["donor_offset", "seed", "gold_count"], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-loop-plan", type=Path, required=True)
    parser.add_argument("--donor-offsets", type=int, nargs="+", default=[-1, 1])
    parser.add_argument("--random-seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = pd.read_csv(args.native_loop_plan)
    frozen = freeze_plan(
        original,
        donor_offsets=tuple(int(value) for value in args.donor_offsets),
        random_seed=int(args.random_seed),
    )
    _atomic_csv(args.output, frozen)
    manifest_path = args.output.with_suffix(".manifest.json")
    _atomic_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "native_loop_plan": str(args.native_loop_plan.resolve()),
            "native_loop_plan_sha256": _sha256(args.native_loop_plan),
            "frozen_plan": str(args.output.resolve()),
            "frozen_plan_sha256": _sha256(args.output),
            "donor_offsets": [int(value) for value in args.donor_offsets],
            "random_seed": int(args.random_seed),
            "pair_count": int(len(frozen)),
            "seed_count": int(frozen["seed"].nunique()),
            "all_shuffled_absolute_distances_matched": bool(
                frozen["shuffled_absolute_distance_matched"].all()
            ),
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )
    print(
        json.dumps(
            {
                "pair_count": int(len(frozen)),
                "seed_count": int(frozen["seed"].nunique()),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
