#!/usr/bin/env python3
"""Analyze the frozen within-example targeted-write mediator restoration assay."""

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
    files = sorted(path.glob("shards/*.jsonl")) or sorted(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No restoration shards in {path}")
    return files


def _read_jsonl(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError("Restoration assay has no rows")
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
        raise ValueError("Restoration bootstrap requires finite seed effects")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    draws = values[indices].mean(axis=1)
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
    paired = effects.groupby("seed", as_index=False)[
        ["selected_restore__matched_control", "selected_restore__cut"]
    ].mean()
    values = paired[
        ["selected_restore__matched_control", "selected_restore__cut"]
    ].to_numpy(dtype=float)
    denominator = abs(float(values[:, 0].mean()))
    if denominator <= 1e-8:
        raise ValueError("Matched-control restoration is too small for a ratio")
    estimate = abs(float(values[:, 1].mean())) / denominator
    rng = np.random.default_rng(int(seed))
    sampled = values[
        rng.integers(0, len(values), size=(int(samples), len(values)))
    ]
    ratios = np.abs(sampled[:, :, 1].mean(axis=1)) / np.maximum(
        np.abs(sampled[:, :, 0].mean(axis=1)), 1e-8
    )
    high = float(np.quantile(ratios, 0.975))
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(ratios, 0.025)),
        "ci_high": high,
        "pass": high < EQUIVALENCE_BOUND,
        "relative_equivalence_bound": EQUIVALENCE_BOUND,
        "rule": "absolute cut/matched-control restoration ratio CI high < 0.20",
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
        "bridge_design",
        "receiver_write_condition",
        "receiver_write_repeat",
        "mediator_condition",
        "mediator_state_source",
        "greedy_generation_run",
        "readout_condition",
        "status",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Restoration rows lack {missing}")
    if "selection_rank" in trials.columns:
        raise ValueError("Formal restoration must not use selection_rank")
    if set(trials["bridge_design"].astype(str)) != {"restoration"}:
        raise ValueError("Restoration analyzer received another bridge design")
    expected_role = "development" if phase == "discovery" else "confirmation"
    if set(trials["mechanism_split"].astype(str)) != {expected_role}:
        raise ValueError(f"Expected only {expected_role} rows")
    expected_seed_count = 20 if phase == "discovery" else 10
    observed_seeds = sorted(int(value) for value in trials["seed"].unique())
    if len(observed_seeds) != expected_seed_count:
        raise ValueError(
            f"{phase} requires {expected_seed_count} seeds: {observed_seeds}"
        )
    if not set(trials["status"].astype(str)).issubset({"ok", "not_applicable"}):
        raise ValueError("Restoration assay contains a failed status")

    key = ["request_id", "seed", "gold_count"]
    expected_receivers = {"clean": 1, "selected_bank": 1, "layer_matched_random": 3}
    expected_mediators = {"self_state", "clean_state_restore"}
    expected_readouts = {"natural", "matched_control", "cut"}
    for _sample, group in trials.groupby(key, sort=False):
        if len(group) != 30:
            raise ValueError("Every restoration sample must contain exactly 30 rows")
        statuses = set(group["status"].astype(str))
        if statuses not in ({"ok"}, {"not_applicable"}):
            raise ValueError("A restoration sample mixes applicable and excluded rows")
        if statuses == {"ok"}:
            greedy_rows = group.loc[group["greedy_generation_run"].astype(bool)]
            if len(greedy_rows) != 1:
                raise ValueError(
                    "An applicable restoration sample must run greedy exactly once"
                )
            greedy_row = greedy_rows.iloc[0]
            if not (
                str(greedy_row["receiver_write_condition"]) == "clean"
                and str(greedy_row["mediator_condition"]) == "self_state"
                and str(greedy_row["readout_condition"]) == "natural"
            ):
                raise ValueError("Restoration greedy generation used the wrong cell")
        self_rows = group.loc[
            group["mediator_condition"].astype(str).eq("self_state")
        ]
        if not (
            self_rows["mediator_state_source"].astype(str).to_numpy()
            == self_rows["receiver_write_condition"].astype(str).to_numpy()
        ).all():
            raise ValueError("A self-state mediator is not paired to its receiver")
        restore_rows = group.loc[
            group["mediator_condition"].astype(str).eq("clean_state_restore")
        ]
        if set(restore_rows["mediator_state_source"].astype(str)) != {"clean"}:
            raise ValueError("A clean-state restoration used a non-clean mediator")
        for mediator in expected_mediators:
            for readout in expected_readouts:
                cell = group.loc[
                    group["mediator_condition"].astype(str).eq(mediator)
                    & group["readout_condition"].astype(str).eq(readout)
                ]
                counts = (
                    cell["receiver_write_condition"].astype(str).value_counts().to_dict()
                )
                if counts != expected_receivers:
                    raise ValueError(f"Restoration receiver arms changed: {counts}")
    usable = trials.loc[trials["status"].eq("ok")].copy()
    if not len(usable):
        raise ValueError("No applicable restoration rows")

    effects: list[dict[str, Any]] = []
    for sample_key, group in usable.groupby(key, sort=False):
        effect: dict[str, Any] = dict(zip(key, sample_key, strict=True))
        for readout in ("natural", "matched_control", "cut"):
            cell = group.loc[group["readout_condition"].astype(str).eq(readout)]
            clean_self = cell.loc[
                cell["receiver_write_condition"].astype(str).eq("clean")
                & cell["mediator_condition"].astype(str).eq("self_state")
            ]
            selected_self = cell.loc[
                cell["receiver_write_condition"].astype(str).eq("selected_bank")
                & cell["mediator_condition"].astype(str).eq("self_state")
            ]
            selected_restore = cell.loc[
                cell["receiver_write_condition"].astype(str).eq("selected_bank")
                & cell["mediator_condition"].astype(str).eq("clean_state_restore")
            ]
            random_self = cell.loc[
                cell["receiver_write_condition"].astype(str).eq(
                    "layer_matched_random"
                )
                & cell["mediator_condition"].astype(str).eq("self_state")
            ].sort_values("receiver_write_repeat")
            random_restore = cell.loc[
                cell["receiver_write_condition"].astype(str).eq(
                    "layer_matched_random"
                )
                & cell["mediator_condition"].astype(str).eq("clean_state_restore")
            ].sort_values("receiver_write_repeat")
            if (
                len(clean_self) != 1
                or len(selected_self) != 1
                or len(selected_restore) != 1
                or len(random_self) != 3
                or len(random_restore) != 3
            ):
                raise ValueError("An applicable restoration cell is incomplete")
            if random_self["receiver_write_repeat"].tolist() != random_restore[
                "receiver_write_repeat"
            ].tolist():
                raise ValueError("Random restoration repeats are not paired")
            selected_self_margin = float(
                selected_self.iloc[0]["correct_count_margin"]
            )
            selected_restore_margin = float(
                selected_restore.iloc[0]["correct_count_margin"]
            )
            random_self_margins = random_self[
                "correct_count_margin"
            ].to_numpy(dtype=float)
            random_restore_margins = random_restore[
                "correct_count_margin"
            ].to_numpy(dtype=float)
            selected_restoration = selected_restore_margin - selected_self_margin
            random_restoration = float(
                np.mean(random_restore_margins - random_self_margins)
            )
            effect[f"receiver_damage__{readout}"] = float(
                random_self_margins.mean() - selected_self_margin
            )
            effect[f"selected_restore__{readout}"] = selected_restoration
            effect[f"random_restore__{readout}"] = random_restoration
            effect[f"restore_specificity__{readout}"] = (
                selected_restoration - random_restoration
            )
            effect[f"clean_exact__{readout}"] = float(
                clean_self.iloc[0].get("exact_count", 0.0)
            )
        effect["restoration_occlusion"] = (
            effect["selected_restore__matched_control"]
            - effect["selected_restore__cut"]
        )
        effects.append(effect)
    effect_frame = pd.DataFrame(effects)
    applicable_seeds = sorted(int(value) for value in effect_frame["seed"].unique())
    if applicable_seeds != observed_seeds:
        missing_applicable = sorted(set(observed_seeds) - set(applicable_seeds))
        raise ValueError(
            f"{phase} has no applicable restoration sample for seeds "
            f"{missing_applicable}; the effective seed count would violate the "
            f"fixed {expected_seed_count}-seed contract"
        )

    summaries = {
        name: _seed_summary(
            effect_frame,
            column,
            samples=bootstrap_samples,
            seed=random_seed + offset,
        )
        for offset, (name, column) in enumerate(
            (
                ("receiver_damage", "receiver_damage__matched_control"),
                ("selected_restore", "selected_restore__matched_control"),
                ("restore_specificity", "restore_specificity__matched_control"),
                ("restoration_occlusion", "restoration_occlusion"),
            )
        )
    }
    clean = _seed_summary(
        effect_frame,
        "clean_exact__natural",
        samples=bootstrap_samples,
        seed=random_seed + 50,
    )
    gates: dict[str, Any] = {
        "clean_endpoint_adequacy": {
            **clean,
            "pass": clean["ci_low"] >= 0.50,
            "rule": "clean self-state natural exact-count accuracy CI low >= 0.50",
        },
        "targeted_receiver_damage": {
            **summaries["receiver_damage"],
            "pass": summaries["receiver_damage"]["ci_low"] > 0,
            "rule": "mean-random minus selected self-state margin CI low > 0",
        },
        "clean_state_restores_selected_receiver": {
            **summaries["selected_restore"],
            "pass": summaries["selected_restore"]["ci_low"] > 0,
            "rule": "selected clean-restore minus self-state margin CI low > 0",
        },
        "restoration_is_targeted_specific": {
            **summaries["restore_specificity"],
            "pass": summaries["restore_specificity"]["ci_low"] > 0,
            "rule": "selected minus mean-random restoration CI low > 0",
        },
        "readout_cut_occludes_restoration": {
            **summaries["restoration_occlusion"],
            "pass": summaries["restoration_occlusion"]["ci_low"] > 0,
            "rule": "matched-control restoration minus cut restoration CI low > 0",
        },
        "cut_restoration_residual_equivalence": _ratio_gate(
            effect_frame,
            samples=bootstrap_samples,
            seed=random_seed + 100,
        ),
    }
    primary = tuple(gates)
    passed = all(bool(gates[name]["pass"]) for name in primary)
    claims = {
        "phase": phase,
        "confirmatory": phase == "confirmation",
        "primary_gate_ids": list(primary),
        "integrated_mediator_restoration_pass": passed,
        "gates": gates,
        "allowed_claim_if_confirmation_passes": (
            "The frozen targeted-retrieval intervention damages a terminal count "
            "state; restoring the same example's clean state rescues the answer "
            "readout, and the model-specific readout cut removes that rescue."
        ),
        "restriction": (
            "This controlled teacher-forced mediator restoration must be paired "
            "with the separately frozen natural free-generation endpoint."
        ),
    }
    audit = {
        "status": "PASS",
        "phase": phase,
        "seed_count": len(observed_seeds),
        "seeds": observed_seeds,
        "applicable_seed_count": len(applicable_seeds),
        "applicable_seeds": applicable_seeds,
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
        "integrated_mediator_restoration_pass": passed,
    }
    return effect_frame, claims, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["discovery", "confirmation"], required=True
    )
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
