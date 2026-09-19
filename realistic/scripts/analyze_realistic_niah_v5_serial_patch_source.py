#!/usr/bin/env python3
"""Analyze terminal-state patches crossed with persistent all-head source masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_realistic_niah_v5_full_state_patch_source import (
    full_state_patch_source_effects,
    summarize_effects,
)


RELATIVE_EQUIVALENCE_BOUND = 0.20
GEOMETRY_NOT_APPLICABLE_REASON = (
    "not applicable: a trace item is shorter than the requested suffix8 geometry"
)
def _read_shards(path: Path) -> pd.DataFrame:
    files = sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL shards under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(
            json.loads(line)
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return pd.DataFrame(rows)


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


def _summary_row(
    summary: pd.DataFrame,
    *,
    outcome: str,
    source_family: str,
    estimand: str,
) -> dict[str, float]:
    selected = summary.loc[
        summary["grouping"].eq("overall")
        & summary["outcome"].eq(outcome)
        & summary["source_family"].eq(source_family)
        & summary["estimand"].eq(estimand)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one summary row for {outcome}/{source_family}/{estimand}, "
            f"observed {len(selected)}"
        )
    row = selected.iloc[0]
    return {
        "estimate": float(row["mean_seed_equal"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
    }


def _optional_summary_row(
    summary: pd.DataFrame,
    *,
    outcome: str,
    source_family: str,
    estimand: str,
) -> dict[str, float] | None:
    selected = summary.loc[
        summary["grouping"].eq("overall")
        & summary["outcome"].eq(outcome)
        & summary["source_family"].eq(source_family)
        & summary["estimand"].eq(estimand)
    ]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise ValueError(
            f"Expected at most one summary row for "
            f"{outcome}/{source_family}/{estimand}, observed {len(selected)}"
        )
    row = selected.iloc[0]
    return {
        "estimate": float(row["mean_seed_equal"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
    }


def _residual_ratio_gate(
    effects: pd.DataFrame, *, bootstrap_samples: int, random_seed: int
) -> dict[str, Any]:
    clean_column = "correct_count_margin__patch_damage_clean"
    masked_column = (
        "correct_count_margin__trace_items__patch_damage_true_mask"
    )
    seed_values = effects.groupby("seed", as_index=False)[
        [clean_column, masked_column]
    ].mean()
    values = seed_values[[clean_column, masked_column]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Serial source seed effects contain non-finite values")
    rng = np.random.default_rng(int(random_seed))
    indices = rng.integers(
        0, len(seed_values), size=(int(bootstrap_samples), len(seed_values))
    )
    draws = values[indices].mean(axis=1)
    valid = np.abs(draws[:, 0]) > 1e-8
    if valid.mean() < 0.99:
        raise ValueError("Clean patch effect is too close to zero for ratio gate")
    ratios = np.abs(draws[valid, 1]) / np.abs(draws[valid, 0])
    point = abs(float(values[:, 1].mean())) / abs(float(values[:, 0].mean()))
    low = float(np.quantile(ratios, 0.025))
    high = float(np.quantile(ratios, 0.975))
    return {
        "estimate": point,
        "ci_low": low,
        "ci_high": high,
        "relative_equivalence_bound": RELATIVE_EQUIVALENCE_BOUND,
        "pass": high < RELATIVE_EQUIVALENCE_BOUND,
        "rule": "persistent_trace_mask_residual_ratio_ci_high < 0.20",
    }


def serial_source_claim_gates(
    effects: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    phase: str,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    storage = _summary_row(
        summary,
        outcome="correct_count_margin",
        source_family="none",
        estimand="patch_damage_clean",
    )
    matched = _summary_row(
        summary,
        outcome="correct_count_margin",
        source_family="trace_items",
        estimand="patch_damage_matched_mask",
    )
    interaction = _summary_row(
        summary,
        outcome="correct_count_margin",
        source_family="trace_items",
        estimand="specific_interaction",
    )
    prompt = _summary_row(
        summary,
        outcome="correct_count_margin",
        source_family="prompt_records",
        estimand="specific_interaction",
    )
    adoption = _summary_row(
        summary,
        outcome="expected_count",
        source_family="none",
        estimand="adoption_clean",
    )
    adoption_interaction = _summary_row(
        summary,
        outcome="expected_count",
        source_family="trace_items",
        estimand="adoption_interaction",
    )
    residual = _residual_ratio_gate(
        effects,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed + 20_000,
    )
    greedy_storage = _optional_summary_row(
        summary,
        outcome="exact_count",
        source_family="none",
        estimand="patch_damage_clean",
    )
    greedy_matched = _optional_summary_row(
        summary,
        outcome="exact_count",
        source_family="trace_items",
        estimand="patch_damage_matched_mask",
    )
    greedy_interaction = _optional_summary_row(
        summary,
        outcome="exact_count",
        source_family="trace_items",
        estimand="specific_interaction",
    )
    gates: dict[str, Any] = {
        "storage_main_effect": {
            **storage,
            "pass": storage["ci_low"] > 0,
            "rule": "ci_low > 0",
        },
        "matched_source_control_preserves_patch": {
            **matched,
            "pass": matched["ci_low"] > 0,
            "rule": "ci_low > 0",
        },
        "trace_source_specific_occlusion": {
            **interaction,
            "pass": interaction["ci_high"] < 0,
            "rule": "specific_interaction_ci_high < 0",
        },
        "trace_mask_residual_equivalence": residual,
        "prompt_source_interaction": {
            **prompt,
            "pass": prompt["ci_low"] >= 0,
            "rule": "ci_low >= 0",
            "role": "secondary_route_diagnostic",
        },
        "donor_count_adoption": {
            **adoption,
            "pass": adoption["ci_low"] > 0,
            "rule": "ci_low > 0",
            "role": "secondary",
        },
        "donor_adoption_trace_occlusion": {
            **adoption_interaction,
            "pass": adoption_interaction["ci_high"] < 0,
            "rule": "ci_high < 0",
            "role": "secondary",
        },
    }
    greedy_gate_ids: list[str] = []
    if (
        greedy_storage is not None
        and greedy_matched is not None
        and greedy_interaction is not None
    ):
        gates.update(
            {
                "greedy_exact_count_patch_effect": {
                    **greedy_storage,
                    "pass": greedy_storage["ci_low"] > 0,
                    "rule": "supplementary exact-count patch damage ci_low > 0",
                    "role": "supplementary_final_output",
                },
                "greedy_exact_count_matched_control": {
                    **greedy_matched,
                    "pass": greedy_matched["ci_low"] > 0,
                    "rule": "supplementary matched-mask exact-count damage ci_low > 0",
                    "role": "supplementary_final_output",
                },
                "greedy_exact_count_specific_occlusion": {
                    **greedy_interaction,
                    "pass": greedy_interaction["ci_high"] < 0,
                    "rule": "supplementary exact-count interaction ci_high < 0",
                    "role": "supplementary_final_output",
                },
            }
        )
        greedy_gate_ids = [
            "greedy_exact_count_patch_effect",
            "greedy_exact_count_matched_control",
            "greedy_exact_count_specific_occlusion",
        ]
    primary = (
        "storage_main_effect",
        "matched_source_control_preserves_patch",
        "trace_source_specific_occlusion",
        "trace_mask_residual_equivalence",
    )
    passed = all(bool(gates[name]["pass"]) for name in primary)
    return {
        "phase": phase,
        "confirmatory": phase == "confirmation",
        "primary_gate_ids": list(primary),
        "distributed_serial_readout_pass": bool(passed),
        "supplementary_greedy_gate_ids": greedy_gate_ids,
        "greedy_exact_count_support_pass": (
            all(bool(gates[name]["pass"]) for name in greedy_gate_ids)
            if greedy_gate_ids
            else None
        ),
        "allowed_claim_if_confirmation_passes": (
            "A terminal trace count state causally affects the answer and is "
            "read out through distributed trace-source attention across the "
            "answer query and numeric answer tokens."
        ),
        "restriction": (
            "All-head source masking identifies a distributed route, not a "
            "minimal or uniquely necessary sparse head set."
        ),
        "gates": gates,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["discovery", "confirmation"], required=True
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260820)
    args = parser.parse_args(argv)

    trials = _read_shards(args.trials.resolve())
    status_by_pair = trials.groupby("pair_sha256")["status"].agg(
        lambda values: tuple(sorted(set(str(value) for value in values)))
    )
    mixed_or_invalid = status_by_pair.loc[
        ~status_by_pair.isin([("ok",), ("not_applicable",)])
    ]
    if not mixed_or_invalid.empty:
        raise ValueError(
            "Formal serial source analysis permits no mixed-status pairs: "
            f"{mixed_or_invalid.to_dict()}"
        )
    not_applicable = trials.loc[trials["status"].astype(str).eq("not_applicable")]
    if not not_applicable.empty:
        reasons = set(not_applicable["exclusion_reason"].astype(str))
        if reasons != {GEOMETRY_NOT_APPLICABLE_REASON}:
            raise ValueError(
                "Only the frozen suffix8 pre-intervention geometry exclusion "
                f"is permitted, observed {sorted(reasons)}"
            )
        counts = not_applicable.groupby("pair_sha256").size()
        if not (counts == 10).all():
            raise ValueError(
                "A geometry-ineligible pair must carry all ten factorial "
                f"N/A cells: {counts.loc[counts.ne(10)].to_dict()}"
            )
    planned_pair_count = int(trials["pair_sha256"].nunique())
    excluded_pair_count = int(not_applicable["pair_sha256"].nunique())
    trials = trials.loc[trials["status"].astype(str).eq("ok")].copy()
    if trials.empty:
        raise ValueError("No geometry-eligible formal serial source trials remain")
    expected_split = "development" if args.phase == "discovery" else "confirmation"
    if set(trials["mechanism_split"].astype(str)) != {expected_split}:
        raise ValueError("Serial source analysis received the wrong mechanism split")
    if set(trials["mask_scope"].astype(str)) != {
        "answer_query_and_answer_tokens"
    }:
        raise ValueError("Formal distributed readout requires persistent source masks")
    if "selection_rank" in trials.columns:
        raise ValueError("Formal serial source trials must not contain selection_rank")
    expected_seed_count = 20 if args.phase == "discovery" else 10
    observed_seeds = sorted(int(value) for value in trials["seed"].unique())
    if len(observed_seeds) != expected_seed_count:
        raise ValueError(
            f"{args.phase} requires {expected_seed_count} seeds, observed {observed_seeds}"
        )
    pair_counts = trials.groupby("seed")["pair_sha256"].nunique()
    if (pair_counts < 1).any():
        raise ValueError("Every canonical seed must contribute an eligible pair")

    effects = full_state_patch_source_effects(trials)
    summary = summarize_effects(
        effects,
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    claims = serial_source_claim_gates(
        effects,
        summary,
        phase=args.phase,
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    output = args.output.resolve()
    _atomic_csv(output / "pair_effects.csv", effects)
    _atomic_csv(output / "seed_equal_summary.csv", summary)
    _atomic_json(output / "claim_gates.json", claims)
    _atomic_json(
        output / "audit.json",
        {
            "status": "PASS",
            "phase": args.phase,
            "trial_rows": int(len(trials)),
            "pair_count": int(effects["pair_sha256"].nunique()),
            "planned_pair_count": planned_pair_count,
            "geometry_not_applicable_pair_count": excluded_pair_count,
            "geometry_not_applicable_reason": GEOMETRY_NOT_APPLICABLE_REASON,
            "pairs_per_seed_min": int(pair_counts.min()),
            "pairs_per_seed_max": int(pair_counts.max()),
            "seed_count": len(observed_seeds),
            "seeds": observed_seeds,
            "mask_scope": "answer_query_and_answer_tokens",
            "bootstrap_samples": int(args.bootstrap_samples),
            "relative_equivalence_bound": RELATIVE_EQUIVALENCE_BOUND,
            "distributed_serial_readout_pass": bool(
                claims["distributed_serial_readout_pass"]
            ),
        },
    )


if __name__ == "__main__":
    main()
