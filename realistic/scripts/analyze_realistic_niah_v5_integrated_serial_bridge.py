#!/usr/bin/env python3
"""Analyze the frozen targeted-write -> state -> readout bridge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EQUIVALENCE_BOUND = 0.20


def _trial_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("shards/*.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No integrated bridge shards in {path}")
    return files


def _read_jsonl(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        # Iterate physical LF-delimited records. str.splitlines() also splits
        # legal U+2028/U+2029 characters embedded in JSON strings.
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError("Integrated bridge has no rows")
    return pd.DataFrame(rows)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Integrated bridge bootstrap requires finite seed effects")
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, len(values), size=(int(samples), len(values)))].mean(
        axis=1
    )
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "seed_count": int(len(values)),
    }


def _seed_summary(
    effects: pd.DataFrame, column: str, *, samples: int, seed: int
) -> dict[str, float]:
    values = (
        effects.groupby("seed", as_index=False)[column]
        .mean()[column]
        .to_numpy(dtype=float)
    )
    return _bootstrap(values, samples=samples, seed=seed)


def _ratio_gate(
    effects: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, Any]:
    seed_frame = effects.groupby("seed", as_index=False)[
        ["damage__matched_control", "damage__cut"]
    ].mean()
    values = seed_frame[["damage__matched_control", "damage__cut"]].to_numpy(
        dtype=float
    )
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Integrated residual ratio requires finite paired seed effects")
    denominator = abs(float(values[:, 0].mean()))
    if denominator <= 1e-8:
        raise ValueError("Integrated matched-control damage is too small for a ratio")
    estimate = abs(float(values[:, 1].mean())) / denominator
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    sampled = values[indices]
    matched = np.abs(sampled[:, :, 0].mean(axis=1))
    cut = np.abs(sampled[:, :, 1].mean(axis=1))
    ratios = cut / np.maximum(matched, 1e-8)
    low = float(np.quantile(ratios, 0.025))
    high = float(np.quantile(ratios, 0.975))
    return {
        "estimate": float(estimate),
        "ci_low": low,
        "ci_high": high,
        "pass": high < EQUIVALENCE_BOUND,
        "relative_equivalence_bound": EQUIVALENCE_BOUND,
        "rule": "selected-vs-random cut residual ratio CI high < 0.20",
    }


def analyze(
    trials: pd.DataFrame,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    required = {
        "request_id",
        "model_label",
        "seed",
        "gold_count",
        "mechanism_split",
        "write_condition",
        "write_repeat",
        "readout_condition",
        "status",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Integrated bridge rows lack {missing}")
    if "selection_rank" in trials.columns:
        raise ValueError("Formal integrated bridge must not use selection_rank")
    expected_role = "development" if phase == "discovery" else "confirmation"
    if set(trials["mechanism_split"].astype(str)) != {expected_role}:
        raise ValueError(f"Expected only {expected_role} rows")
    expected_seeds = 20 if phase == "discovery" else 10
    observed_seeds = sorted(int(value) for value in trials["seed"].unique())
    if len(observed_seeds) != expected_seeds:
        raise ValueError(f"{phase} requires {expected_seeds} seeds: {observed_seeds}")
    allowed_status = {"ok", "not_applicable"}
    if not set(trials["status"].astype(str)).issubset(allowed_status):
        raise ValueError("Integrated bridge contains a failed status")
    expected_writes = {"clean": 1, "selected_bank": 1, "layer_matched_random": 3}
    expected_readouts = {"natural", "matched_control", "cut"}
    key = ["request_id", "seed", "gold_count"]
    for _sample, group in trials.groupby(key, sort=False):
        if len(group) != 15:
            raise ValueError("Every integrated sample must contain exactly 15 rows")
        for readout, readout_group in group.groupby("readout_condition"):
            if str(readout) not in expected_readouts:
                raise ValueError("Unknown integrated readout condition")
            counts = readout_group["write_condition"].astype(str).value_counts().to_dict()
            if counts != expected_writes:
                raise ValueError(f"Integrated write arms changed: {counts}")
    usable = trials.loc[trials["status"].eq("ok")].copy()
    if not len(usable):
        raise ValueError("No applicable integrated bridge rows")
    if set(usable["readout_condition"].astype(str)) != expected_readouts:
        raise ValueError("Applicable integrated rows lost a readout arm")

    effects: list[dict[str, Any]] = []
    for sample_key, group in usable.groupby(key, sort=False):
        effect: dict[str, Any] = dict(zip(key, sample_key, strict=True))
        for readout in ("natural", "matched_control", "cut"):
            cell = group.loc[group["readout_condition"].eq(readout)]
            clean = cell.loc[cell["write_condition"].eq("clean")]
            selected = cell.loc[cell["write_condition"].eq("selected_bank")]
            random = cell.loc[cell["write_condition"].eq("layer_matched_random")]
            if len(clean) != 1 or len(selected) != 1 or len(random) != 3:
                raise ValueError("An applicable integrated cell is incomplete")
            effect[f"damage__{readout}"] = float(
                random["correct_count_margin"].astype(float).mean()
                - selected.iloc[0]["correct_count_margin"]
            )
            effect[f"clean_exact__{readout}"] = float(clean.iloc[0].get("exact_count", 0.0))
            if "exact_count" in random and "exact_count" in selected:
                effect[f"exact_damage__{readout}"] = float(
                    random["exact_count"].astype(float).mean()
                    - float(selected.iloc[0]["exact_count"])
                )
        effect["readout_occlusion"] = (
            effect["damage__matched_control"] - effect["damage__cut"]
        )
        effects.append(effect)
    effect_frame = pd.DataFrame(effects)
    summaries = {
        name: _seed_summary(
            effect_frame,
            column,
            samples=bootstrap_samples,
            seed=random_seed + index,
        )
        for index, (name, column) in enumerate(
            (
                ("targeted_state_damage", "damage__natural"),
                ("matched_control_damage", "damage__matched_control"),
                ("readout_occlusion", "readout_occlusion"),
            )
        )
    }
    clean_summary = _seed_summary(
        effect_frame,
        "clean_exact__natural",
        samples=bootstrap_samples,
        seed=random_seed + 50,
    )
    gates: dict[str, Any] = {
        "clean_state_endpoint_adequacy": {
            **clean_summary,
            "pass": clean_summary["ci_low"] >= 0.50,
            "rule": "clean-state greedy exact-count accuracy CI low >= 0.50",
        },
        "targeted_bank_changes_terminal_state_readout": {
            **summaries["targeted_state_damage"],
            "pass": summaries["targeted_state_damage"]["ci_low"] > 0,
            "rule": "selected-minus-mean-random margin damage CI low > 0",
        },
        "matched_readout_control_preserves_damage": {
            **summaries["matched_control_damage"],
            "pass": summaries["matched_control_damage"]["ci_low"] > 0,
            "rule": "selected-minus-mean-random matched-control damage CI low > 0",
        },
        "readout_cut_occludes_targeted_state_effect": {
            **summaries["readout_occlusion"],
            "pass": summaries["readout_occlusion"]["ci_low"] > 0,
            "rule": "matched damage minus cut damage CI low > 0",
        },
        "cut_residual_equivalence": _ratio_gate(
            effect_frame,
            samples=bootstrap_samples,
            seed=random_seed + 100,
        ),
    }
    primary = tuple(gates)
    greedy = None
    if "exact_damage__natural" in effect_frame:
        greedy = _seed_summary(
            effect_frame,
            "exact_damage__natural",
            samples=bootstrap_samples,
            seed=random_seed + 200,
        )
        gates["supplementary_greedy_exact_count_damage"] = {
            **greedy,
            "pass": greedy["ci_low"] > 0,
            "role": "supplementary_final_output",
            "rule": "selected-minus-random exact-count damage CI low > 0",
        }
    claims = {
        "phase": phase,
        "confirmatory": phase == "confirmation",
        "primary_gate_ids": list(primary),
        "integrated_serial_bridge_pass": all(bool(gates[name]["pass"]) for name in primary),
        "gates": gates,
        "allowed_claim_if_confirmation_passes": (
            "The frozen targeted-retrieval bank causally changes a terminal trace "
            "state that affects count readout through the model-specific confirmed "
            "answer route."
        ),
        "restriction": (
            "This teacher-forced bridge must be paired with the separately frozen "
            "free-generation targeted-count endpoint; it is not a natural-effects "
            "mediation estimate or a minimal circuit."
        ),
    }
    audit = {
        "status": "PASS",
        "phase": phase,
        "seed_count": len(observed_seeds),
        "seeds": observed_seeds,
        "planned_sample_count": int(trials[key].drop_duplicates().shape[0]),
        "applicable_sample_count": int(effect_frame.shape[0]),
        "not_applicable_sample_count": int(
            trials.loc[trials["status"].eq("not_applicable"), key]
            .drop_duplicates()
            .shape[0]
        ),
        "trial_row_count": int(len(trials)),
        "applicable_trial_row_count": int(len(usable)),
        "selection_rank_used": False,
        "teacher_forced_trace": True,
        "integrated_serial_bridge_pass": bool(claims["integrated_serial_bridge_pass"]),
    }
    return effect_frame, claims, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation"], required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    args = parser.parse_args()
    trials = _read_jsonl(_trial_files(args.trials))
    effects, claims, audit = analyze(
        trials,
        phase=str(args.phase),
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    output = Path(args.output)
    _atomic_csv(output / "sample_effects.csv", effects)
    _atomic_json(output / "claim_gates.json", claims)
    _atomic_json(output / "audit.json", audit)
    print(json.dumps(claims, sort_keys=True))


if __name__ == "__main__":
    main()
