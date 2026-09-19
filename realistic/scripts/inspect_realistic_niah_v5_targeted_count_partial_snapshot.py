#!/usr/bin/env python3
"""Inspect a paused, incomplete endpoint without creating a formal claim gate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER_PATH = SCRIPT_DIR / "analyze_realistic_niah_v5_targeted_count_endpoint.py"
SPEC = importlib.util.spec_from_file_location("targeted_endpoint_analyzer", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


STATUS = "EXPLORATORY_PARTIAL_SNAPSHOT_NOT_A_FORMAL_DISCOVERY_GATE"
KEY = ["request_id", "seed", "gold_count", "from_occurrence", "to_occurrence"]
EXPECTED_ARMS = {"clean": 1, "selected_bank": 1, "layer_matched_random": 3}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _shard_ledger(path: Path) -> tuple[int, str]:
    files = sorted((path / "shards").glob("*.jsonl"))
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return len(files), digest.hexdigest()


def inspect(
    trials: pd.DataFrame,
    *,
    bootstrap_samples: int,
    random_seed: int,
    expected_anchor_count_by_seed: dict[int, int] | None = None,
) -> dict[str, Any]:
    if set(trials["status"].astype(str)) != {"ok"}:
        raise ValueError("Partial snapshot permits no failed-trial exclusions")
    if set(trials["split"].astype(str)) != {"discovery"}:
        raise ValueError("Partial snapshot requires discovery rows only")
    if set(trials["head_ablation_decode_steps_requested"].astype(int)) != {-1}:
        raise ValueError("Targeted bank must remain ablated through decode")
    if "selection_rank" in trials.columns:
        raise ValueError("Partial snapshot must not contain selection_rank")
    observed_conditions = set(trials["condition"].astype(str))
    if observed_conditions - set(EXPECTED_ARMS):
        raise ValueError(f"Unexpected conditions: {sorted(observed_conditions)}")

    arm_counts = (
        trials.groupby(KEY + ["condition"]).size().unstack(fill_value=0)
    ).reindex(columns=list(EXPECTED_ARMS), fill_value=0)
    for condition, expected in EXPECTED_ARMS.items():
        if (arm_counts[condition] > expected).any():
            raise ValueError(f"An anchor has too many {condition} rows")
    complete_mask = np.logical_and.reduce(
        [arm_counts[name].eq(expected) for name, expected in EXPECTED_ARMS.items()]
    )
    raw_complete_keys = arm_counts.loc[complete_mask].reset_index()[KEY]
    if raw_complete_keys.empty:
        raise ValueError("No complete five-arm anchors in partial snapshot")
    raw_seed_coverage = (
        raw_complete_keys.groupby("seed").size().sort_index().astype(int).to_dict()
    )
    partially_observed_seeds: dict[str, dict[str, int]] = {}
    complete_keys = raw_complete_keys
    if expected_anchor_count_by_seed is not None:
        fully_completed_seeds = []
        for raw_seed, observed in raw_seed_coverage.items():
            seed = int(raw_seed)
            expected = int(expected_anchor_count_by_seed.get(seed, -1))
            if expected <= 0:
                raise ValueError(f"Seed {seed} is absent from the frozen registry")
            if int(observed) > expected:
                raise ValueError(f"Seed {seed} exceeds its frozen anchor count")
            if int(observed) == expected:
                fully_completed_seeds.append(seed)
            else:
                partially_observed_seeds[str(seed)] = {
                    "observed_complete_anchors": int(observed),
                    "expected_frozen_anchors": expected,
                }
        complete_keys = raw_complete_keys.loc[
            raw_complete_keys["seed"].astype(int).isin(fully_completed_seeds)
        ].copy()
        if complete_keys.empty:
            raise ValueError("No fully completed seed in partial snapshot")
    raw_complete = trials.merge(
        raw_complete_keys, on=KEY, how="inner", validate="many_to_one"
    )
    complete = trials.merge(complete_keys, on=KEY, how="inner", validate="many_to_one")
    if not (
        complete["to_occurrence"].astype(int).eq(complete["gold_count"].astype(int))
        & complete["from_occurrence"].astype(int).eq(
            complete["gold_count"].astype(int) - 1
        )
    ).all():
        raise ValueError("Snapshot contains a non-final N-1 -> N transition")

    scored = complete.copy()
    scored["parsed_final_count"] = scored["completion_text"].map(
        ANALYZER.parse_final_total
    )
    scored["final_count_correct"] = scored["parsed_final_count"].eq(
        scored["gold_count"]
    ).astype(float)
    scored["final_count_failure"] = 1.0 - scored["final_count_correct"]
    scored["next_city_failure"] = 1.0 - scored["correct_next_needle"].astype(float)
    scored["joint_retrieval_and_count_failure"] = (
        scored["next_city_failure"].eq(1.0)
        & scored["final_count_failure"].eq(1.0)
    ).astype(float)
    scored["final_undercount"] = (
        scored["parsed_final_count"].notna()
        & scored["parsed_final_count"].lt(scored["gold_count"])
    ).astype(float)
    scored["exact_minus_one"] = (
        scored["parsed_final_count"].notna()
        & scored["parsed_final_count"].eq(scored["gold_count"] - 1)
    ).astype(float)

    outcomes = (
        "final_count_failure",
        "joint_retrieval_and_count_failure",
        "next_city_failure",
        "final_undercount",
        "exact_minus_one",
    )
    summaries: dict[str, dict[str, float]] = {}
    seed_effect_rows: list[dict[str, Any]] = []
    for offset, outcome in enumerate(outcomes):
        frame, summary = ANALYZER._seed_contrast(
            scored,
            outcome,
            samples=bootstrap_samples,
            seed=random_seed + offset,
        )
        summaries[outcome] = summary
        seed_effect_rows.extend(frame.to_dict(orient="records"))

    clean = scored.loc[scored["condition"].eq("clean")]
    clean_seed_accuracy = clean.groupby("seed")["final_count_correct"].mean()
    clean_accuracy = ANALYZER._bootstrap(
        clean_seed_accuracy.to_numpy(dtype=float),
        samples=bootstrap_samples,
        seed=random_seed + 100,
    )
    gates = {
        "clean_endpoint_adequacy": {
            **clean_accuracy,
            "snapshot_pass": clean_accuracy["ci_low"] >= 0.50,
            "formal_rule": "clean final-count accuracy CI low >= 0.50",
        },
        "targeted_bank_changes_final_count": {
            **summaries["final_count_failure"],
            "snapshot_pass": summaries["final_count_failure"]["ci_low"] > 0,
            "formal_rule": "selected-minus-random final-count failure CI low > 0",
        },
        "retrieval_failure_propagates_to_count": {
            **summaries["joint_retrieval_and_count_failure"],
            "snapshot_pass": summaries["joint_retrieval_and_count_failure"][
                "ci_low"
            ]
            > 0,
            "formal_rule": (
                "selected-minus-random joint retrieval+count failure CI low > 0"
            ),
        },
    }
    seed_coverage = (
        complete_keys.groupby("seed").size().sort_index().astype(int).to_dict()
    )
    observed_seeds = sorted(int(value) for value in complete["seed"].unique())
    return {
        "schema_version": "realistic_niah_v5_targeted_partial_snapshot_v1",
        "status": STATUS,
        "snapshot_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "formal_discovery_gate_evaluated": False,
        "formal_discovery_seed_contract_met": observed_seeds
        == list(range(1234, 1254)),
        "observed_seed_count": len(observed_seeds),
        "observed_seeds": observed_seeds,
        "complete_anchor_count": int(len(complete_keys)),
        "complete_trial_row_count": int(len(complete)),
        "discarded_incomplete_anchor_trial_row_count": int(
            len(trials) - len(raw_complete)
        ),
        "excluded_partial_seed_trial_row_count": int(
            len(raw_complete) - len(complete)
        ),
        "partially_observed_seeds": partially_observed_seeds,
        "complete_anchor_count_by_seed": {
            str(seed): count for seed, count in seed_coverage.items()
        },
        "selection_rank_used": False,
        "bootstrap_unit": "seed",
        "bootstrap_samples": int(bootstrap_samples),
        "snapshot_primary_gates": gates,
        "snapshot_all_primary_pass": all(
            bool(value["snapshot_pass"]) for value in gates.values()
        ),
        "secondary_seed_contrasts": {
            name: summaries[name]
            for name in ("next_city_failure", "final_undercount", "exact_minus_one")
        },
        "restriction": (
            "This is an optional, incomplete-run snapshot. It is not the frozen "
            "20-seed discovery analysis and cannot open confirmation or a bridge."
        ),
        "seed_effect_rows": seed_effect_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-registry", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    args = parser.parse_args()
    trials_path = args.trials.resolve()
    trials = ANALYZER._read_shards(trials_path)
    registry_rows = [
        json.loads(line)
        for line in args.anchor_registry.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_anchor_count_by_seed: dict[int, int] = {}
    for row in registry_rows:
        seed = int(row["seed"])
        if seed in range(1234, 1254):
            expected_anchor_count_by_seed[seed] = (
                expected_anchor_count_by_seed.get(seed, 0) + 1
            )
    if set(expected_anchor_count_by_seed) != set(range(1234, 1254)):
        raise ValueError("Frozen registry does not cover every discovery seed")
    result = inspect(
        trials,
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
        expected_anchor_count_by_seed=expected_anchor_count_by_seed,
    )
    shard_count, shard_ledger_sha = _shard_ledger(trials_path)
    result["trials_root"] = str(trials_path)
    result["observed_shard_count"] = shard_count
    result["observed_shard_ledger_sha256"] = shard_ledger_sha
    result["anchor_registry"] = str(args.anchor_registry.resolve())
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
