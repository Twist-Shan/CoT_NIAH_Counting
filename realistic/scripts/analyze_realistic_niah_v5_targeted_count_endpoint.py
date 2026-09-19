#!/usr/bin/env python3
"""Analyze persistent targeted-retrieval ablation through the final count."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


TOTAL_RE = re.compile(r"(?i)(?:^|\b)total\s*:\s*([0-9]+)\b")


def parse_final_total(text: str) -> int | None:
    matches = TOTAL_RE.findall(str(text))
    return int(matches[-1]) if matches else None


def _read_shards(path: Path) -> pd.DataFrame:
    files = sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL shards under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        # TextIO iteration follows physical CR/LF boundaries.  str.splitlines()
        # also splits on U+2028/U+2029, which are legal inside a JSON string and
        # occur verbatim in some model completions.
        with file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed JSONL at {file}:{line_number}: {error}"
                    ) from error
    return pd.DataFrame(rows)


def _bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    draws = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "seed_count": int(len(values)),
    }


def _seed_contrast(
    anchor: pd.DataFrame,
    outcome: str,
    *,
    samples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for seed_value, group in anchor.groupby("seed", sort=True):
        selected = group.loc[group["condition"].eq("selected_bank"), outcome]
        random = group.loc[group["condition"].eq("layer_matched_random"), outcome]
        if selected.empty or random.empty:
            raise ValueError(f"Seed {seed_value} lacks selected or random arms")
        rows.append(
            {
                "seed": int(seed_value),
                "outcome": outcome,
                "selected": float(selected.mean()),
                "random_mean": float(random.mean()),
                "selected_minus_random": float(selected.mean() - random.mean()),
            }
        )
    frame = pd.DataFrame(rows)
    summary = _bootstrap(
        frame["selected_minus_random"].to_numpy(dtype=float),
        samples=samples,
        seed=seed,
    )
    return frame, summary


def analyze(
    trials: pd.DataFrame,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if set(trials["status"].astype(str)) != {"ok"}:
        raise ValueError("Formal endpoint analysis permits no failed-trial exclusions")
    expected_split = "discovery" if phase == "discovery" else "confirmation"
    if set(trials["split"].astype(str)) != {expected_split}:
        raise ValueError(f"Expected only {expected_split} rows")
    if set(trials["head_ablation_decode_steps_requested"].astype(int)) != {-1}:
        raise ValueError("Targeted bank must remain ablated through all decode steps")
    expected_seeds = 20 if phase == "discovery" else 10
    seeds = sorted(int(value) for value in trials["seed"].unique())
    if len(seeds) != expected_seeds:
        raise ValueError(f"{phase} requires {expected_seeds} seeds, observed {seeds}")
    if set(trials["condition"].astype(str)) != {
        "clean", "selected_bank", "layer_matched_random"
    }:
        raise ValueError("Formal endpoint requires clean, selected, and random arms")
    if "selection_rank" in trials.columns:
        raise ValueError("Formal endpoint trials must not use selection_rank")

    key = ["request_id", "seed", "gold_count", "from_occurrence", "to_occurrence"]
    trial_counts = trials.groupby(key + ["condition"]).size().unstack(fill_value=0)
    if not (trial_counts["clean"] == 1).all():
        raise ValueError("Each anchor must have exactly one clean arm")
    if not (trial_counts["selected_bank"] == 1).all():
        raise ValueError("Each anchor must have exactly one selected arm")
    if not (trial_counts["layer_matched_random"] == 3).all():
        raise ValueError("Each anchor must have exactly three random repeats")
    # The preregistered direct endpoint is the final retrieval transition N-1 -> N.
    if not (
        trials["to_occurrence"].astype(int).eq(trials["gold_count"].astype(int))
        & trials["from_occurrence"].astype(int).eq(
            trials["gold_count"].astype(int) - 1
        )
    ).all():
        raise ValueError("All anchors must be the outcome-blind final N-1 -> N transition")

    scored = trials.copy()
    scored["parsed_final_count"] = scored["completion_text"].map(parse_final_total)
    scored["final_count_parsed"] = scored["parsed_final_count"].notna().astype(float)
    scored["final_count_correct"] = (
        scored["parsed_final_count"].eq(scored["gold_count"])
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

    outcomes = [
        "final_count_failure",
        "joint_retrieval_and_count_failure",
        "next_city_failure",
        "final_undercount",
        "exact_minus_one",
    ]
    seed_frames: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, float]] = {}
    for index, outcome in enumerate(outcomes):
        frame, summary = _seed_contrast(
            scored,
            outcome,
            samples=bootstrap_samples,
            seed=random_seed + index,
        )
        seed_frames.append(frame)
        summaries[outcome] = summary

    clean = scored.loc[scored["condition"].eq("clean")]
    clean_seed_accuracy = clean.groupby("seed")["final_count_correct"].mean()
    clean_accuracy = _bootstrap(
        clean_seed_accuracy.to_numpy(dtype=float),
        samples=bootstrap_samples,
        seed=random_seed + 100,
    )
    final_gate = summaries["final_count_failure"]
    joint_gate = summaries["joint_retrieval_and_count_failure"]
    gates = {
        "clean_endpoint_adequacy": {
            **clean_accuracy,
            "pass": clean_accuracy["ci_low"] >= 0.50,
            "rule": "clean final-count accuracy CI low >= 0.50",
        },
        "targeted_bank_changes_final_count": {
            **final_gate,
            "pass": final_gate["ci_low"] > 0,
            "rule": "selected-minus-random final-count failure CI low > 0",
        },
        "retrieval_failure_propagates_to_count": {
            **joint_gate,
            "pass": joint_gate["ci_low"] > 0,
            "rule": "selected-minus-random joint retrieval+count failure CI low > 0",
        },
        "directional_undercount": {
            **summaries["final_undercount"],
            "pass": summaries["final_undercount"]["ci_low"] > 0,
            "rule": "secondary: selected-minus-random undercount CI low > 0",
        },
        "exact_minus_one": {
            **summaries["exact_minus_one"],
            "pass": summaries["exact_minus_one"]["ci_low"] > 0,
            "rule": "secondary: selected-minus-random exact N-1 CI low > 0",
        },
    }
    primary = (
        "clean_endpoint_adequacy",
        "targeted_bank_changes_final_count",
        "retrieval_failure_propagates_to_count",
    )
    claims = {
        "phase": phase,
        "confirmatory": phase == "confirmation",
        "primary_gate_ids": list(primary),
        "targeted_to_count_pass": all(bool(gates[name]["pass"]) for name in primary),
        "gates": gates,
        "restriction": (
            "The joint endpoint establishes intervention propagation, not by "
            "itself a formal natural-effects mediation estimate."
        ),
    }
    return scored, pd.concat(seed_frames, ignore_index=True), claims


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation"], required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    args = parser.parse_args(argv)
    trials = _read_shards(args.trials.resolve())
    scored, seed_effects, claims = analyze(
        trials,
        phase=str(args.phase),
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    output = args.output.resolve()
    _atomic_csv(output / "anchor_arms.csv", scored)
    _atomic_csv(output / "seed_effects.csv", seed_effects)
    _atomic_json(output / "claim_gates.json", claims)
    _atomic_json(
        output / "audit.json",
        {
            "status": "PASS",
            "phase": str(args.phase),
            "trial_rows": int(len(scored)),
            "anchor_count": int(
                scored[["request_id", "from_occurrence", "to_occurrence"]]
                .drop_duplicates().shape[0]
            ),
            "seed_count": int(scored["seed"].nunique()),
            "seeds": sorted(int(value) for value in scored["seed"].unique()),
            "generation_truncated_rows": int(scored["generation_truncated"].sum()),
            "unparsed_rows_counted_as_failure": int(
                scored["parsed_final_count"].isna().sum()
            ),
            "decode_head_ablation_steps": -1,
            "selection_rank_used": False,
            "targeted_to_count_pass": bool(claims["targeted_to_count_pass"]),
        },
    )


if __name__ == "__main__":
    main()
