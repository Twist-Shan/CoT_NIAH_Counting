#!/usr/bin/env python3
"""Analyze prospective P0 -> retrieval-query targeted-head mediation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import (  # noqa: E402
    bootstrap_seed_mean_ci,
    sign_flip_pvalue,
)
from realistic_niah_v5.count_stream import (  # noqa: E402
    COUNT_STREAM_CONFIRMATION_SEEDS,
    COUNT_STREAM_DISCOVERY_SEEDS,
)


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


def _read_trials(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        files = [path] if path.is_file() else sorted((path / "shards").glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No query-mediation shards under {path}")
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise ValueError("Query-mediation analysis received no rows")
    frame = pd.DataFrame(rows)
    if "selection_rank" in frame.columns:
        raise ValueError("Formal query-mediation outcomes contain selection_rank")
    used = frame.get("selection_rank_used", pd.Series(False, index=frame.index))
    if used.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"}).any():
        raise ValueError("Formal query mediation used selection_rank")
    return frame


def _seed_linear_contrast(
    frame: pd.DataFrame,
    *,
    estimand: str,
    outcome: str,
    coefficients: Mapping[tuple[str, str], float],
    offsets: tuple[int, ...],
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = frame.loc[
        pd.to_numeric(frame["donor_offset"], errors="raise")
        .astype(int)
        .isin(offsets)
    ].copy()
    selected[outcome] = pd.to_numeric(selected[outcome], errors="coerce")
    selected = selected.loc[selected[outcome].notna()]
    keys = ["model_label", "seed", "pair_sha256"]
    arm_keys = ["state_condition", "head_condition"]
    if selected.duplicated(keys + arm_keys).any():
        raise ValueError(f"Estimand {estimand} has duplicate pair/arm rows")
    wide = selected.pivot(index=keys, columns=arm_keys, values=outcome)
    required = list(coefficients)
    missing = [arm for arm in required if arm not in wide.columns]
    if missing:
        raise ValueError(f"Estimand {estimand} is missing arms {missing}")
    wide = wide.dropna(subset=required).copy()
    pair_effect = pd.Series(0.0, index=wide.index, dtype=float)
    for arm, coefficient in coefficients.items():
        pair_effect += float(coefficient) * wide[arm].astype(float)
    effect_frame = pair_effect.rename("pair_effect").reset_index()
    seed_effects = (
        effect_frame
        .groupby(["model_label", "seed"], as_index=False)
        .agg(effect=("pair_effect", "mean"), pair_count=("pair_sha256", "nunique"))
    )
    values = seed_effects["effect"].to_numpy(dtype=float)
    summary = bootstrap_seed_mean_ci(
        values, samples=int(bootstrap_samples), seed=int(random_seed)
    )
    summary.update(
        {
            "estimand": estimand,
            "outcome": outcome,
            "coefficients": json.dumps(
                {f"{state}|{head}": value for (state, head), value in coefficients.items()},
                sort_keys=True,
            ),
            "offsets": list(offsets),
            "n_seeds": int(len(seed_effects)),
            "pair_count": int(len(effect_frame)),
            "p_value": sign_flip_pvalue(values),
            "higher_is_supportive": True,
            "gate_pass": bool(summary["ci_low"] > 0.0),
        }
    )
    seed_effects.insert(0, "estimand", estimand)
    return summary, seed_effects


def _effect_coefficients(
    treatment: str, control: str, head_condition: str
) -> dict[tuple[str, str], float]:
    return {
        (treatment, head_condition): 1.0,
        (control, head_condition): -1.0,
    }


def _interaction_coefficients(
    treatment: str, control: str, masked: str
) -> dict[tuple[str, str], float]:
    return {
        (treatment, "intact"): 1.0,
        (control, "intact"): -1.0,
        (treatment, masked): -1.0,
        (control, masked): 1.0,
    }


def _specificity_coefficients(
    treatment: str, control: str
) -> dict[tuple[str, str], float]:
    return {
        (treatment, "selected_mask"): -1.0,
        (control, "selected_mask"): 1.0,
        (treatment, "layer_matched_random_mask"): 1.0,
        (control, "layer_matched_random_mask"): -1.0,
    }


def _capped_width_diagnostics(
    frame: pd.DataFrame,
    *,
    geometry: str,
    outcome: str,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any] | None:
    """Describe realized widths without changing the registered pooled gate."""

    name = str(geometry)
    if not name.startswith("suffix_cap"):
        return None
    requested = int(name.removeprefix("suffix_cap"))
    required = {
        "pair_sha256",
        "patch_token_count",
        "requested_patch_token_count",
        "patch_token_count_capped_by_shorter_span",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Capped geometry lacks width audit fields: {missing}")
    metadata = frame[
        [
            "pair_sha256",
            "seed",
            "donor_offset",
            "patch_token_count",
            "requested_patch_token_count",
            "patch_token_count_capped_by_shorter_span",
        ]
    ].drop_duplicates()
    if metadata["pair_sha256"].duplicated().any():
        raise ValueError("Capped width metadata changed across registered arms")
    observed_requested = set(
        pd.to_numeric(
            metadata["requested_patch_token_count"], errors="raise"
        ).astype(int)
    )
    if observed_requested != {requested}:
        raise ValueError("Capped geometry requested-width audit changed")
    widths = pd.to_numeric(metadata["patch_token_count"], errors="raise").astype(int)
    if (widths < 1).any() or (widths > requested).any():
        raise ValueError("Capped geometry realized an invalid patch width")
    expected_capped = widths.lt(requested)
    observed_capped = metadata[
        "patch_token_count_capped_by_shorter_span"
    ].map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})
    if not observed_capped.reset_index(drop=True).equals(
        expected_capped.reset_index(drop=True)
    ):
        raise ValueError("Capped-width flag disagrees with realized width")

    exact_pair_ids = set(
        metadata.loc[widths.eq(requested), "pair_sha256"].astype(str)
    )
    exact = frame.loc[frame["pair_sha256"].astype(str).isin(exact_pair_ids)].copy()
    specifications = [
        (
            "full_state_effect_intact",
            _effect_coefficients("full_donor_patch", "self_patch", "intact"),
        ),
        (
            "full_selected_mask_interaction",
            _interaction_coefficients(
                "full_donor_patch", "self_patch", "selected_mask"
            ),
        ),
        (
            "full_head_output_restore",
            {
                (
                    "full_donor_patch_heads_into_self_patch",
                    "selected_restore",
                ): 1.0,
                ("self_patch", "intact"): -1.0,
            },
        ),
        (
            "full_random_mask_interaction",
            _interaction_coefficients(
                "full_donor_patch", "self_patch", "layer_matched_random_mask"
            ),
        ),
        (
            "full_selected_vs_random_specificity",
            _specificity_coefficients("full_donor_patch", "self_patch"),
        ),
    ]
    exact_summaries: list[dict[str, Any]] = []
    for index, (estimand, coefficients) in enumerate(specifications):
        summary, _seed_effects = _seed_linear_contrast(
            exact,
            estimand=f"{estimand}_exact_width_{requested}",
            outcome=outcome,
            coefficients=coefficients,
            offsets=(-1, 1),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + index,
        )
        exact_summaries.append(summary)
    width_counts = widths.value_counts().sort_index()
    return {
        "requested_width": requested,
        "pair_count": int(len(metadata)),
        "realized_width_pair_counts": {
            str(int(width)): int(count) for width, count in width_counts.items()
        },
        "capped_pair_count": int(expected_capped.sum()),
        "exact_width_pair_count": int(widths.eq(requested).sum()),
        "exact_width_seed_count": int(
            metadata.loc[widths.eq(requested), "seed"].nunique()
        ),
        "exact_width_primary_offset_pair_count": int(
            metadata.loc[
                widths.eq(requested)
                & pd.to_numeric(metadata["donor_offset"], errors="raise")
                .astype(int)
                .isin((-1, 1))
            ].shape[0]
        ),
        "secondary_only": True,
        "changes_registered_pooled_gate": False,
        "exact_width_estimands": exact_summaries,
    }


def analyze(
    frame: pd.DataFrame,
    *,
    phase: str,
    geometry: str,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected_seeds = (
        set(COUNT_STREAM_DISCOVERY_SEEDS)
        if phase == "discovery"
        else set(COUNT_STREAM_CONFIRMATION_SEEDS)
    )
    observed_seeds = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"{phase} query-mediation seed mismatch: expected="
            f"{sorted(expected_seeds)} observed={sorted(observed_seeds)}"
        )
    expected_split = "development" if phase == "discovery" else "confirmation"
    if set(frame["mechanism_split"].astype(str)) != {expected_split}:
        raise ValueError("Query-mediation mechanism split disagrees with phase")
    if set(frame["patch_geometry"].astype(str)) != {str(geometry)}:
        raise ValueError("Query-mediation analysis mixed geometries")
    if set(frame["experiment_id"].astype(str)) != {
        "p0_same_trajectory_query_mediation"
    }:
        raise ValueError("Query-mediation analysis received another experiment")
    if frame["head_plan_file_sha256"].astype(str).nunique() != 1:
        raise ValueError("Query-mediation outcomes mix frozen head plans")
    if frame["targeted_bank_sha256"].astype(str).nunique() != 1:
        raise ValueError("Query-mediation outcomes mix targeted banks")

    expected_rows_per_pair = 14 if str(geometry) == "endpoint" else 7
    rows_per_pair = frame.groupby("pair_sha256").size()
    if not rows_per_pair.eq(expected_rows_per_pair).all():
        raise ValueError("A query-mediation pair lacks a registered arm")
    offsets_observed = set(
        pd.to_numeric(frame["donor_offset"], errors="raise").astype(int)
    )
    if offsets_observed != {-3, -2, -1, 1, 2, 3}:
        raise ValueError("Query-mediation signed-offset contract changed")

    outcome = "donor_vs_receiver_query_city_log_odds"
    specifications: list[dict[str, Any]] = []
    comparisons = [("full", "full_donor_patch", "self_patch")]
    if str(geometry) == "endpoint":
        comparisons.append(
            (
                "count",
                "count_subspace_transplant",
                "norm_matched_orthogonal_patch",
            )
        )
    for label, treatment, control in comparisons:
        restore_state = f"{treatment}_heads_into_{control}"
        specifications.extend(
            [
                {
                    "estimand": f"{label}_state_effect_intact",
                    "coefficients": _effect_coefficients(
                        treatment, control, "intact"
                    ),
                    "primary": True,
                    "role": f"{label}_sufficiency",
                },
                {
                    "estimand": f"{label}_selected_mask_interaction",
                    "coefficients": _interaction_coefficients(
                        treatment, control, "selected_mask"
                    ),
                    "primary": True,
                    "role": f"{label}_mediation",
                },
                {
                    "estimand": f"{label}_head_output_restore",
                    "coefficients": {
                        (restore_state, "selected_restore"): 1.0,
                        (control, "intact"): -1.0,
                    },
                    "primary": True,
                    "role": f"{label}_restoration",
                },
                {
                    "estimand": f"{label}_random_mask_interaction",
                    "coefficients": _interaction_coefficients(
                        treatment, control, "layer_matched_random_mask"
                    ),
                    "primary": False,
                    "role": f"{label}_random_control",
                },
                {
                    "estimand": f"{label}_selected_vs_random_specificity",
                    "coefficients": _specificity_coefficients(treatment, control),
                    "primary": False,
                    "role": f"{label}_specificity",
                },
            ]
        )

    summaries: list[dict[str, Any]] = []
    effects: list[pd.DataFrame] = []
    for index, specification in enumerate(specifications):
        summary, seed_effects = _seed_linear_contrast(
            frame,
            estimand=specification["estimand"],
            outcome=outcome,
            coefficients=specification["coefficients"],
            offsets=(-1, 1),
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + index,
        )
        summary["primary"] = bool(specification["primary"])
        summary["role"] = str(specification["role"])
        summaries.append(summary)
        effects.append(seed_effects)

    for distance in (2, 3):
        for label, treatment, control in comparisons:
            for suffix, coefficients in (
                (
                    "state_effect_intact",
                    _effect_coefficients(treatment, control, "intact"),
                ),
                (
                    "selected_mask_interaction",
                    _interaction_coefficients(
                        treatment, control, "selected_mask"
                    ),
                ),
            ):
                estimand = f"{label}_{suffix}_distance_{distance}"
                summary, seed_effects = _seed_linear_contrast(
                    frame,
                    estimand=estimand,
                    outcome=outcome,
                    coefficients=coefficients,
                    offsets=(-distance, distance),
                    bootstrap_samples=bootstrap_samples,
                    random_seed=random_seed + len(summaries),
                )
                summary["primary"] = False
                summary["role"] = f"{label}_dose_robustness"
                summaries.append(summary)
                effects.append(seed_effects)

    by_name = {summary["estimand"]: summary for summary in summaries}
    full_primary = [
        by_name["full_state_effect_intact"]["gate_pass"],
        by_name["full_selected_mask_interaction"]["gate_pass"],
        by_name["full_head_output_restore"]["gate_pass"],
    ]
    full_mediation_pass = bool(all(full_primary))
    count_mediation_pass = False
    if str(geometry) == "endpoint":
        count_primary = [
            by_name["count_state_effect_intact"]["gate_pass"],
            by_name["count_selected_mask_interaction"]["gate_pass"],
            by_name["count_head_output_restore"]["gate_pass"],
        ]
        count_mediation_pass = bool(all(count_primary))
    gates = {
        "schema_version": "realistic_niah_v5_query_mediation_analysis_v1",
        "phase": phase,
        "geometry": str(geometry),
        "registered_seeds": sorted(expected_seeds),
        "seed_count": len(expected_seeds),
        "selection_rank_used": False,
        "targeted_bank_sha256": str(frame["targeted_bank_sha256"].iloc[0]),
        "head_plan_file_sha256": str(frame["head_plan_file_sha256"].iloc[0]),
        "full_state_mediation_pass": full_mediation_pass,
        "count_specific_mediation_pass": count_mediation_pass,
        "geometry_pass": full_mediation_pass,
        "confirmation_eligible": bool(
            phase == "discovery" and full_mediation_pass
        ),
        "primary_gate_rule": (
            "positive_95pct_ci_for_intact_state_effect_selected_mask_"
            "interaction_and_selected_head_output_restoration"
        ),
        "random_specificity_is_secondary": True,
        "estimands": summaries,
    }
    width_diagnostics = _capped_width_diagnostics(
        frame,
        geometry=str(geometry),
        outcome=outcome,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed + len(summaries) + 100,
    )
    if width_diagnostics is not None:
        gates["capped_width_diagnostics"] = width_diagnostics
    return (
        pd.DataFrame(summaries),
        pd.concat(effects, ignore_index=True),
        gates,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--phase", choices=["discovery", "confirmation"], required=True)
    parser.add_argument(
        "--geometry",
        choices=[
            "endpoint",
            "suffix4",
            "suffix8",
            "suffix_cap4",
            "suffix_cap8",
        ],
        required=True,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = _read_trials(args.trials)
    estimands, seed_effects, gates = analyze(
        frame,
        phase=args.phase,
        geometry=args.geometry,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    _atomic_csv(args.output / "estimands.csv", estimands)
    _atomic_csv(args.output / "seed_effects.csv", seed_effects)
    _atomic_json(args.output / "claim_gates.json", gates)
    print(
        json.dumps(
            {
                "geometry": gates["geometry"],
                "full_state_mediation_pass": gates["full_state_mediation_pass"],
                "count_specific_mediation_pass": gates[
                    "count_specific_mediation_pass"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
