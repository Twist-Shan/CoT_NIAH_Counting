from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from realistic_niah_v4_4_3.io import atomic_csv_gzip, atomic_json, stage_root
from realistic_niah_v4_4_4.upstream_path_pipeline import BASE_STAGE, EXPANDED_STAGE

from .upstream_confirmation_spec import V444UpstreamConfirmationConfig


ANALYSIS_STAGE = "upstream_confirmation_analysis"


def exact_sign_flip_p(values: Iterable[float]) -> float:
    """Two-sided exact paired sign-flip p-value, vectorized in bounded chunks."""

    samples = np.asarray(tuple(values), dtype=np.float64)
    if samples.ndim != 1 or len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError("Sign-flip values must be a finite nonempty vector")
    if len(samples) > 24:
        raise ValueError("Exact enumeration is intentionally limited to 24 pairs")
    observed = abs(float(samples.mean()))
    total = 1 << len(samples)
    exceed = 0
    columns = np.arange(len(samples), dtype=np.uint64)
    for start in range(0, total, 65_536):
        stop = min(total, start + 65_536)
        identifiers = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = (((identifiers >> columns) & 1).astype(np.float64) * 2.0) - 1.0
        statistics = (signs @ samples) / len(samples)
        exceed += int(np.count_nonzero(np.abs(statistics) >= observed - 1e-15))
    return exceed / total


def bootstrap_mean_ci(
    values: Iterable[float], *, repetitions: int, seed: int
) -> tuple[float, float]:
    samples = np.asarray(tuple(values), dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(samples), size=(int(repetitions), len(samples)))
    means = samples[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(tuple(p_values), dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _metric_record(
    values: Iterable[float], *, config: V444UpstreamConfirmationConfig, seed: int
) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    low, high = bootstrap_mean_ci(
        array, repetitions=config.bootstrap_repetitions, seed=seed
    )
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "ci_low": low,
        "ci_high": high,
        "exact_two_sided_p": exact_sign_flip_p(array),
    }


def _load_stage(run_root: Path, model_label: str, stage: str) -> pd.DataFrame:
    path = stage_root(run_root, model_label, stage) / "effects.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing confirmation effect table: {path}")
    frame = pd.read_csv(path)
    frame["source_stage"] = stage
    return frame


def _load_effects(
    run_root: Path, config: V444UpstreamConfirmationConfig
) -> pd.DataFrame:
    return pd.concat(
        [
            _load_stage(run_root, config.model_label, BASE_STAGE),
            _load_stage(run_root, config.model_label, EXPANDED_STAGE),
        ],
        ignore_index=True,
    )


def build_seed_metrics(effects: pd.DataFrame) -> pd.DataFrame:
    return (
        effects.groupby(["seed", "late_set"], as_index=False)
        .agg(
            early_donor_log_odds_gain=("early_donor_log_odds_gain", "mean"),
            block_donor_log_odds_gain=("late_block_donor_log_odds_gain", "mean"),
            control_donor_log_odds_gain=("late_control_donor_log_odds_gain", "mean"),
            mediation_specificity=(
                "donor_log_odds_mediation_specificity",
                "mean",
            ),
            early_expected_count_transport=("early_transport", "mean"),
            expected_count_mediation_specificity=("mediation_specificity", "mean"),
            pair_count=("receiver_count", "size"),
        )
        .sort_values(["late_set", "seed"])
        .reset_index(drop=True)
    )


def audit_campaign(
    run_root: Path,
    *,
    config: V444UpstreamConfirmationConfig,
    effects: pd.DataFrame,
    seed_metrics: pd.DataFrame,
) -> dict[str, Any]:
    primary_rows = len(config.evaluation_seeds) * len(config.donor_pairs)
    secondary_rows = primary_rows * (len(config.late_head_sets) - 1)
    base = effects[effects["source_stage"] == BASE_STAGE]
    expanded = effects[effects["source_stage"] == EXPANDED_STAGE]
    closure_max = float(effects["late_block_closure_relative_l2"].abs().max())
    orthogonality_max = float(
        effects["late_control_output_cosine_to_induced"].abs().max()
    )
    reproducibility_max = float(
        effects["late_prefill_reproducibility_relative_l2"].abs().max()
    )
    early_spread = float(
        seed_metrics.groupby("seed")["early_donor_log_odds_gain"]
        .agg(lambda values: float(values.max() - values.min()))
        .max()
    )
    forbidden: list[str] = []
    for stage in (BASE_STAGE, EXPANDED_STAGE):
        directory = stage_root(run_root, config.model_label, stage)
        for path in directory.rglob("*"):
            if path.is_file() and (
                path.suffix.lower() in {".pt", ".pth", ".npy", ".npz"}
                or "raw_attention" in path.name.lower()
                or "full_state" in path.name.lower()
            ):
                forbidden.append(str(path.relative_to(run_root)))
    expected_sets = {name for name, _heads in config.late_head_sets}
    checks = [
        {
            "name": "primary_row_count",
            "passed": len(base) == primary_rows,
            "detail": {"observed": len(base), "expected": primary_rows},
        },
        {
            "name": "leave_one_out_row_count",
            "passed": len(expanded) == secondary_rows,
            "detail": {"observed": len(expanded), "expected": secondary_rows},
        },
        {
            "name": "independent_seed_registry",
            "passed": set(effects["seed"].astype(int))
            == set(config.evaluation_seeds),
            "detail": sorted(set(effects["seed"].astype(int))),
        },
        {
            "name": "frozen_route_and_early_set",
            "passed": set(effects["route"].astype(str)) == {"slot_state"}
            and set(effects["early_set"].astype(str)) == {"top4"},
            "detail": {
                "routes": sorted(set(effects["route"].astype(str))),
                "early_sets": sorted(set(effects["early_set"].astype(str))),
            },
        },
        {
            "name": "frozen_late_sets",
            "passed": set(effects["late_set"].astype(str)) == expected_sets,
            "detail": sorted(set(effects["late_set"].astype(str))),
        },
        {
            "name": "exact_l28_block_closure",
            "passed": math.isfinite(closure_max)
            and closure_max <= config.block_closure_relative_tolerance,
            "detail": closure_max,
        },
        {
            "name": "same_span_orthogonal_control",
            "passed": math.isfinite(orthogonality_max)
            and orthogonality_max <= config.control_orthogonality_tolerance,
            "detail": orthogonality_max,
        },
        {
            "name": "deterministic_prefill_reproducibility",
            "passed": math.isfinite(reproducibility_max) and reproducibility_max <= 1e-5,
            "detail": reproducibility_max,
        },
        {
            "name": "early_intervention_identical_across_late_comparisons",
            "passed": math.isfinite(early_spread) and early_spread <= 1e-7,
            "detail": early_spread,
        },
        {
            "name": "no_persisted_raw_states_or_attention",
            "passed": not forbidden,
            "detail": forbidden,
        },
    ]
    return {
        "all_checks_pass": all(item["passed"] for item in checks),
        "checks": checks,
        "effect_rows": len(effects),
        "seed_metric_rows": len(seed_metrics),
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"




def analyze_campaign(
    run_root: str | Path, *, config: V444UpstreamConfirmationConfig
) -> dict[str, Any]:
    root = Path(run_root)
    effects = _load_effects(root, config)
    seed_metrics = build_seed_metrics(effects)
    primary = seed_metrics[
        seed_metrics["late_set"] == config.primary_late_set
    ].sort_values("seed")
    if len(primary) != len(config.evaluation_seeds):
        raise RuntimeError("Primary set does not contain one paired metric per seed")

    early_record = _metric_record(
        primary["early_donor_log_odds_gain"], config=config, seed=44_410
    )
    mediation_record = _metric_record(
        primary["mediation_specificity"], config=config, seed=44_411
    )
    expected_count_record = _metric_record(
        primary["expected_count_mediation_specificity"],
        config=config,
        seed=44_412,
    )
    iut_p = max(
        early_record["exact_two_sided_p"],
        mediation_record["exact_two_sided_p"],
    )
    serial_confirmed = bool(
        early_record["mean"] > 0
        and mediation_record["mean"] > 0
        and iut_p < config.primary_alpha
    )

    full_by_seed = primary.set_index("seed")["mediation_specificity"]
    loo_rows: list[dict[str, Any]] = []
    for index, (name, _heads) in enumerate(config.late_head_sets[1:]):
        loo = seed_metrics[seed_metrics["late_set"] == name].sort_values("seed")
        loo_by_seed = loo.set_index("seed")["mediation_specificity"]
        if list(loo_by_seed.index) != list(full_by_seed.index):
            raise RuntimeError(f"Leave-one-out seed alignment failed for {name}")
        decrement = full_by_seed.to_numpy() - loo_by_seed.to_numpy()
        removed_head = int(name.removeprefix("minus_h"))
        loo_record = _metric_record(
            loo_by_seed.to_numpy(), config=config, seed=44_420 + index
        )
        decrement_record = _metric_record(
            decrement, config=config, seed=44_430 + index
        )
        loo_rows.append(
            {
                "late_set": name,
                "removed_head": f"H{removed_head}",
                "loo_mediation": loo_record,
                "decrement": decrement_record,
            }
        )
    adjusted = holm_adjust(
        row["decrement"]["exact_two_sided_p"] for row in loo_rows
    )
    for row, adjusted_p in zip(loo_rows, adjusted):
        row["decrement_holm_p"] = adjusted_p
        decrement_supported = bool(
            row["decrement"]["mean"] > 0 and adjusted_p < config.primary_alpha
        )
        loo_supported = bool(
            row["loo_mediation"]["mean"] > 0
            and row["loo_mediation"]["exact_two_sided_p"] < config.primary_alpha
        )
        if decrement_supported and loo_supported:
            interpretation = "incremental but not individually necessary"
        elif decrement_supported:
            interpretation = "necessary within the tested set"
        else:
            interpretation = "no unique decrement resolved"
        row["incremental_contribution_supported"] = decrement_supported
        row["loo_mediation_still_supported"] = loo_supported
        row["interpretation"] = interpretation

    audit = audit_campaign(
        root, config=config, effects=effects, seed_metrics=seed_metrics
    )
    serial_confirmed = bool(serial_confirmed and audit["all_checks_pass"])
    payload: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_4_4_upstream_confirmation_analysis_v1",
        "config": config.to_dict(),
        "primary_decision": {
            "classification": (
                "independent_serial_chain_confirmed"
                if serial_confirmed
                else "independent_serial_chain_not_confirmed"
            ),
            "serial_chain_confirmed": serial_confirmed,
            "early_effect": early_record,
            "mediation": mediation_record,
            "intersection_union_p": iut_p,
            "expected_count_mediation_secondary": expected_count_record,
            "decision_rule": (
                "early donor-log-odds gain > 0 and L28 natural-vs-orthogonal "
                "mediation specificity > 0, with max(two-sided exact p) < 0.05, "
                "and all causal audits passing"
            ),
            "inferential_status": (
                "independent confirmation on seeds 1294--1313; head sets, route, "
                "endpoint, and controls frozen from the exploratory campaign"
            ),
        },
        "leave_one_out": loo_rows,
        "audit": audit,
    }
    output = stage_root(root, config.model_label, ANALYSIS_STAGE)
    atomic_csv_gzip(seed_metrics, output / "seed_metrics.csv.gz")
    atomic_json(
        output / "realistic_niah_v4_4_4_upstream_confirmation_analysis.json",
        payload,
    )
    atomic_json(
        output / "complete.json",
        {
            "schema_version": "realistic_niah_v4_4_4_upstream_confirmation_complete_v1",
            "primary_decision": payload["primary_decision"],
            "audit": audit,
        },
    )
    return payload
