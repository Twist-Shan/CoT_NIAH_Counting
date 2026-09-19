from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from realistic_niah_v4_4_3.io import atomic_csv_gzip, atomic_json, stage_root

from .upstream_path_pipeline import BASE_STAGE, EXPANDED_STAGE, SMOKE_STAGE
from .upstream_path_spec import V444UpstreamPathConfig


ANALYSIS_STAGE = "upstream_path_analysis"


def _require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def exact_sign_flip_p(values: np.ndarray) -> float:
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError("Sign-flip input must be a finite nonempty vector")
    observed = abs(float(samples.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(samples)):
        statistic = abs(float(np.mean(samples * np.asarray(signs))))
        exceed += int(statistic >= observed - 1e-15)
        total += 1
    return exceed / total


def bootstrap_mean_ci(
    values: np.ndarray, *, repetitions: int, seed: int
) -> tuple[float, float]:
    samples = np.asarray(values, dtype=float)
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(samples), size=(int(repetitions), len(samples)))
    means = samples[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _load_effects(run_root: Path, config: V444UpstreamPathConfig) -> pd.DataFrame:
    frames = []
    for stage in (SMOKE_STAGE, BASE_STAGE, EXPANDED_STAGE):
        path = stage_root(run_root, config.model_label, stage) / "effects.csv.gz"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["source_stage"] = stage
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No upstream-path effect table is available")
    result = pd.concat(frames, ignore_index=True)
    _require_columns(
        result,
        (
            "seed", "receiver_count", "donor_count", "early_set", "route",
            "late_set", "early_transport", "late_block_transport",
            "late_control_transport", "mediation_specificity",
            "early_donor_log_odds_gain",
            "late_block_donor_log_odds_gain",
            "late_control_donor_log_odds_gain",
            "donor_log_odds_mediation_specificity",
            "late_block_closure_relative_l2",
            "late_control_output_cosine_to_induced",
        ),
        "upstream effects",
    )
    return result


def build_seed_metrics(effects: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        effects.groupby(["seed", "early_set", "route", "late_set"], as_index=False)
        .agg(
            early_transport=("early_transport", "mean"),
            late_block_transport=("late_block_transport", "mean"),
            late_control_transport=("late_control_transport", "mean"),
            block_suppression=("block_suppression", "mean"),
            control_suppression=("control_suppression", "mean"),
            mediation_specificity=("mediation_specificity", "mean"),
            early_donor_log_odds_gain=("early_donor_log_odds_gain", "mean"),
            late_block_donor_log_odds_gain=("late_block_donor_log_odds_gain", "mean"),
            late_control_donor_log_odds_gain=("late_control_donor_log_odds_gain", "mean"),
            donor_log_odds_block_suppression=("donor_log_odds_block_suppression", "mean"),
            donor_log_odds_control_suppression=("donor_log_odds_control_suppression", "mean"),
            donor_log_odds_mediation_specificity=("donor_log_odds_mediation_specificity", "mean"),
            late_induced_output_norm=("late_induced_output_norm", "mean"),
            pair_count=("receiver_count", "size"),
        )
    )
    return metrics


def summarize_seed_metrics(
    seed_metrics: pd.DataFrame, *, config: V444UpstreamPathConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (early_set, route, late_set), group in seed_metrics.groupby(
        ["early_set", "route", "late_set"], sort=True
    ):
        row: dict[str, Any] = {
            "early_set": str(early_set),
            "route": str(route),
            "late_set": str(late_set),
            "seed_count": int(group["seed"].nunique()),
            "pair_count_per_seed": int(group["pair_count"].min()),
        }
        for metric_index, metric in enumerate(
            (
                "early_donor_log_odds_gain",
                "donor_log_odds_mediation_specificity",
                "donor_log_odds_block_suppression",
                "early_transport",
                "mediation_specificity",
                "block_suppression",
            )
        ):
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(
                values,
                repetitions=config.bootstrap_repetitions,
                seed=44_400 + metric_index * 1000 + sum(map(ord, str(early_set) + str(route) + str(late_set))),
            )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_p"] = exact_sign_flip_p(values)
        row["late_block_transport_mean"] = float(group["late_block_transport"].mean())
        row["late_control_transport_mean"] = float(group["late_control_transport"].mean())
        row["late_block_donor_log_odds_gain_mean"] = float(
            group["late_block_donor_log_odds_gain"].mean()
        )
        row["late_control_donor_log_odds_gain_mean"] = float(
            group["late_control_donor_log_odds_gain"].mean()
        )
        row["late_induced_output_norm_mean"] = float(group["late_induced_output_norm"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    for metric in (
        "early_donor_log_odds_gain",
        "donor_log_odds_mediation_specificity",
        "early_transport",
        "mediation_specificity",
    ):
        adjusted = np.empty(len(summary), dtype=float)
        for _late_set, indices in summary.groupby("late_set").groups.items():
            positions = list(indices)
            values = holm_adjust(summary.loc[positions, f"{metric}_p"].tolist())
            adjusted[positions] = values
        summary[f"{metric}_holm_p"] = adjusted
    summary["early_supported"] = (
        (summary["early_donor_log_odds_gain_mean"] > 0)
        & (summary["early_donor_log_odds_gain_holm_p"] < config.primary_alpha)
    )
    summary["mediation_supported"] = (
        (summary["donor_log_odds_mediation_specificity_mean"] > 0)
        & (
            summary["donor_log_odds_mediation_specificity_holm_p"]
            < config.primary_alpha
        )
    )
    summary["expected_count_early_supported_secondary"] = (
        (summary["early_transport_mean"] > 0)
        & (summary["early_transport_holm_p"] < config.primary_alpha)
    )
    summary["expected_count_mediation_supported_secondary"] = (
        (summary["mediation_specificity_mean"] > 0)
        & (summary["mediation_specificity_holm_p"] < config.primary_alpha)
    )
    summary["serial_path_supported"] = (
        summary["early_supported"] & summary["mediation_supported"]
    )
    return summary


def _natural_summary(run_root: Path, config: V444UpstreamPathConfig) -> pd.DataFrame:
    path = stage_root(run_root, config.model_label, BASE_STAGE) / "natural.csv.gz"
    if not path.is_file():
        raise FileNotFoundError("Upstream base natural table is missing")
    frame = pd.read_csv(path)
    return (
        frame.groupby(
            ["candidate_rank", "layer", "head", "v442_stable_score"], as_index=False
        )
        .agg(
            observed_broad_score_mean=("broad_retrieval_score", "mean"),
            observed_broad_score_sd=("broad_retrieval_score", "std"),
            needle_attention_mass_mean=("needle_attention_mass", "mean"),
            occurrence_coverage_mean=("occurrence_coverage", "mean"),
            baseline_correct_rate=("baseline_correct", "mean"),
            sample_count=("seed", "size"),
        )
        .sort_values("candidate_rank")
    )


def _decision(summary: pd.DataFrame, config: V444UpstreamPathConfig) -> dict[str, Any]:
    base = summary[summary["late_set"] == config.primary_late_set]
    route_support = {
        route: bool(base.loc[base["route"] == route, "serial_path_supported"].any())
        for route in config.routes
    }
    mechanistic_support = bool(
        route_support["slot_edge_qk"] or route_support["slot_state"]
    )
    expanded_available = bool((summary["late_set"] != config.primary_late_set).any())
    expanded_supported = bool(
        summary.loc[
            (summary["late_set"] != config.primary_late_set)
            & summary["route"].isin(["slot_edge_qk", "slot_state"]),
            "serial_path_supported",
        ].any()
    )
    if mechanistic_support:
        classification = "upstream_read_to_l28_write_supported_exploratory"
    elif expanded_supported:
        classification = (
            "upstream_read_to_expanded_l28_write_supported_exploratory"
        )
    elif route_support["answer_query_full"]:
        classification = "full_output_chain_only"
    else:
        classification = "upstream_to_l28_chain_not_supported"
    return {
        "classification": classification,
        "route_support_with_base_h16_h19": route_support,
        "base_h16_h19_sufficient": mechanistic_support,
        "expanded_stage_available": expanded_available,
        "expanded_l28_set_support": expanded_supported,
        "overall_mechanistic_support": bool(
            mechanistic_support or expanded_supported
        ),
        "requires_expanded_stage": bool(
            config.expand_late_if_base_insufficient
            and not mechanistic_support
            and not expanded_available
        ),
        "inferential_status": (
            "exploratory causal supplement on reused V4.4.4 confirmation seeds; "
            "not an independent confirmation"
        ),
    }


def audit_campaign(
    run_root: str | Path,
    *,
    config: V444UpstreamPathConfig,
    effects: pd.DataFrame | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    frame = _load_effects(root, config) if effects is None else effects
    expected_base = (
        len(config.evaluation_seeds)
        * len(config.donor_pairs)
        * len(config.early_set_sizes)
        * len(config.routes)
    )
    base_rows = int((frame["late_set"] == config.primary_late_set).sum())
    closure_max = float(frame["late_block_closure_relative_l2"].abs().max())
    orthogonality_max = float(
        frame["late_control_output_cosine_to_induced"].abs().max()
    )
    reproducibility_max = float(
        frame["late_prefill_reproducibility_relative_l2"].abs().max()
    )
    forbidden = []
    for stage in (BASE_STAGE, EXPANDED_STAGE):
        directory = root / "models" / config.model_label / stage
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and (
                path.suffix.lower() in {".pt", ".pth", ".npy", ".npz"}
                or "raw_attention" in path.name.lower()
                or "full_state" in path.name.lower()
            ):
                forbidden.append(str(path.relative_to(root)))
    checks = [
        {
            "name": "base_row_count",
            "passed": base_rows == expected_base,
            "detail": {"observed": base_rows, "expected": expected_base},
        },
        {
            "name": "seed_registry",
            "passed": set(frame["seed"].astype(int)) == set(config.evaluation_seeds),
            "detail": sorted(set(frame["seed"].astype(int))),
        },
        {
            "name": "exact_l28_block_closure",
            "passed": math.isfinite(closure_max)
            and closure_max <= config.block_closure_relative_tolerance,
            "detail": closure_max,
        },
        {
            "name": "late_control_orthogonality",
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
            "name": "no_persisted_raw_states_or_attention",
            "passed": not forbidden,
            "detail": forbidden,
        },
    ]
    return {
        "all_checks_pass": all(item["passed"] for item in checks),
        "checks": checks,
        "observed_effect_rows": len(frame),
    }


def analyze_campaign(
    run_root: str | Path, *, config: V444UpstreamPathConfig
) -> dict[str, Any]:
    root = Path(run_root)
    effects = _load_effects(root, config)
    seed_metrics = build_seed_metrics(effects)
    summary = summarize_seed_metrics(seed_metrics, config=config)
    natural = _natural_summary(root, config)
    decision = _decision(summary, config)
    audit = audit_campaign(root, config=config, effects=effects)
    output = stage_root(root, config.model_label, ANALYSIS_STAGE)
    atomic_csv_gzip(seed_metrics, output / "seed_metrics.csv.gz")
    atomic_csv_gzip(summary, output / "summary.csv.gz")
    atomic_csv_gzip(natural, output / "natural_head_summary.csv.gz")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_upstream_analysis_v1",
        "decision": decision,
        "audit": audit,
        "config": config.to_dict(),
        "summary": json.loads(summary.to_json(orient="records")),
        "natural_head_summary": json.loads(natural.to_json(orient="records")),
    }
    atomic_json(output / "realistic_niah_v4_4_4_upstream_path_analysis.json", payload)
    atomic_json(
        output / "complete.json",
        {
            "schema_version": "realistic_niah_v4_4_4_upstream_analysis_complete_v1",
            "decision": decision,
            "audit": audit,
            "effect_rows": len(effects),
            "seed_metric_rows": len(seed_metrics),
            "summary_rows": len(summary),
        },
    )
    return payload
