from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from realistic_niah_v4_4_3.io import atomic_csv_gzip, atomic_json, atomic_text

from .spec import V444Config


def exact_sign_flip_p(values: Sequence[float], *, alternative: str) -> float:
    """Exact paired randomization p-value without materializing a sign matrix."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan
    sums = np.asarray([0.0], dtype=np.float64)
    for value in array:
        sums = np.concatenate((sums - value, sums + value))
    observed_sum = float(array.sum())
    tolerance = 1e-12 * max(1.0, abs(observed_sum))
    if alternative == "greater":
        return float(np.mean(sums >= observed_sum - tolerance))
    if alternative == "less":
        return float(np.mean(sums <= observed_sum + tolerance))
    raise ValueError("alternative must be greater or less")


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    if len(x_array) < 2:
        return math.nan
    centered = x_array - float(x_array.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return math.nan
    return float(np.dot(centered, y_array - float(y_array.mean())) / denominator)


def _bootstrap_ci(
    values: Sequence[float], *, repetitions: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, math.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(int(seed))
    draw_means = np.empty(int(repetitions), dtype=np.float64)
    chunk = 1_000
    for start in range(0, int(repetitions), chunk):
        stop = min(start + chunk, int(repetitions))
        indices = generator.integers(0, len(array), size=(stop - start, len(array)))
        draw_means[start:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(draw_means, [0.025, 0.975])
    return mean, float(low), float(high)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    output = np.full(len(values), np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return output.tolist()
    order = finite_indices[np.argsort(values[finite_indices], kind="mergesort")]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * float(values[index]))
        running = max(running, adjusted)
        output[index] = running
    return output.tolist()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _append_metric(
    rows: list[dict[str, Any]],
    *,
    family: str,
    endpoint: str,
    set_id: str,
    set_role: str,
    seed: int,
    value: float,
    alternative: str,
) -> None:
    rows.append(
        {
            "family": family,
            "endpoint": endpoint,
            "set_id": str(set_id),
            "set_role": str(set_role),
            "seed": int(seed),
            "value": float(value),
            "alternative": alternative,
        }
    )


def build_seed_metrics(
    natural: pd.DataFrame,
    directed: pd.DataFrame,
    mediation: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        natural,
        (
            "seed",
            "gold_count",
            "set_id",
            "set_role",
            "natural_carrier_coefficient",
        ),
        "natural detail",
    )
    _require_columns(
        directed,
        (
            "seed",
            "gold_count",
            "set_id",
            "set_role",
            "intervention",
            "beta",
            "delta_expected_count",
            "delta_expected_count_absolute_error",
            "delta_correct_margin",
        ),
        "directed detail",
    )
    _require_columns(
        mediation,
        (
            "seed",
            "set_id",
            "set_role",
            "intervention",
            "continuous_normalized_transport",
        ),
        "mediation detail",
    )
    rows: list[dict[str, Any]] = []
    group_keys = ["set_id", "set_role", "seed"]
    for (set_id, set_role, seed), group in natural.groupby(group_keys, sort=True):
        _append_metric(
            rows,
            family="natural_signal",
            endpoint="natural_carrier_count_slope",
            set_id=set_id,
            set_role=set_role,
            seed=seed,
            value=_ols_slope(
                group["gold_count"], group["natural_carrier_coefficient"]
            ),
            alternative="greater",
        )
    injection = directed[directed["intervention"].str.startswith(
        "natural_ov_z_injection_beta_", na=False
    )]
    for (set_id, set_role, seed), group in injection.groupby(group_keys, sort=True):
        per_count = [
            _ols_slope(count_group["beta"], count_group["delta_expected_count"])
            for _count, count_group in group.groupby("gold_count", sort=True)
        ]
        _append_metric(
            rows,
            family="pre_o_injection",
            endpoint="injection_dose_slope",
            set_id=set_id,
            set_role=set_role,
            seed=seed,
            value=float(np.mean(per_count)),
            alternative="greater",
        )
    removal_names = {
        "natural_ov_count_axis_removal": "axis",
        "equal_output_norm_set_span_orthogonal_removal": "control",
    }
    removal = directed[directed["intervention"].isin(removal_names)].copy()
    removal["condition"] = removal["intervention"].map(removal_names)
    for (set_id, set_role, seed), group in removal.groupby(group_keys, sort=True):
        mean_values = group.groupby("condition", sort=True)[
            ["delta_expected_count_absolute_error", "delta_correct_margin"]
        ].mean()
        if set(mean_values.index) != {"axis", "control"}:
            raise RuntimeError("A removal seed lacks axis or in-span control rows")
        _append_metric(
            rows,
            family="centered_removal",
            endpoint="removal_error_axis_minus_control",
            set_id=set_id,
            set_role=set_role,
            seed=seed,
            value=(
                mean_values.loc["axis", "delta_expected_count_absolute_error"]
                - mean_values.loc[
                    "control", "delta_expected_count_absolute_error"
                ]
            ),
            alternative="greater",
        )
        _append_metric(
            rows,
            family="centered_removal",
            endpoint="removal_margin_axis_minus_control",
            set_id=set_id,
            set_role=set_role,
            seed=seed,
            value=(
                mean_values.loc["axis", "delta_correct_margin"]
                - mean_values.loc["control", "delta_correct_margin"]
            ),
            alternative="less",
        )
    mediation_names = {
        "donor_z_patch": "patch",
        "donor_z_patch_natural_axis_block": "block",
        "donor_z_patch_orthogonal_control": "control",
    }
    selected = mediation[mediation["intervention"].isin(mediation_names)].copy()
    selected["condition"] = selected["intervention"].map(mediation_names)
    for (set_id, set_role, seed), group in selected.groupby(group_keys, sort=True):
        values = group.groupby("condition", sort=True)[
            "continuous_normalized_transport"
        ].mean()
        if set(values.index) != {"patch", "block", "control"}:
            raise RuntimeError("A mediation seed lacks one registered condition")
        endpoints = (
            ("donor_patch_transport", values["patch"], "greater"),
            (
                "mediation_control_minus_axis_block",
                values["control"] - values["block"],
                "greater",
            ),
            (
                "mediation_patch_minus_axis_block",
                values["patch"] - values["block"],
                "greater",
            ),
        )
        for endpoint, value, alternative in endpoints:
            _append_metric(
                rows,
                family="path_mediation",
                endpoint=endpoint,
                set_id=set_id,
                set_role=set_role,
                seed=seed,
                value=value,
                alternative=alternative,
            )
    result = pd.DataFrame(rows)
    if result["value"].isna().any() or not np.isfinite(result["value"]).all():
        raise RuntimeError("Seed-metric table contains non-finite values")
    return result


def add_candidate_specificity(
    seed_metrics: pd.DataFrame, selection: Mapping[str, Any]
) -> pd.DataFrame:
    candidate_id = str(selection["candidate"]["set_id"])
    control_ids = [str(item["set_id"]) for item in selection["matched_controls"]]
    candidate = seed_metrics[seed_metrics["set_id"].eq(candidate_id)]
    controls = seed_metrics[seed_metrics["set_id"].isin(control_ids)]
    control_mean = controls.groupby(
        ["family", "endpoint", "seed", "alternative"], as_index=False
    )["value"].mean()
    merged = candidate.merge(
        control_mean,
        on=["family", "endpoint", "seed", "alternative"],
        how="inner",
        suffixes=("_candidate", "_control_mean"),
        validate="one_to_one",
    )
    expected = candidate[["family", "endpoint", "seed"]].drop_duplicates()
    if len(merged) != len(expected):
        raise RuntimeError("Candidate/matched-control specificity grid is incomplete")
    rows = []
    for row in merged.itertuples(index=False):
        rows.append(
            {
                "family": row.family,
                "endpoint": f"{row.endpoint}__candidate_minus_control_mean",
                "set_id": candidate_id,
                "set_role": "candidate_specificity",
                "seed": int(row.seed),
                "value": float(row.value_candidate - row.value_control_mean),
                "alternative": row.alternative,
            }
        )
    return pd.concat([seed_metrics, pd.DataFrame(rows)], ignore_index=True)


def summarize_seed_metrics(
    seed_metrics: pd.DataFrame, config: V444Config
) -> pd.DataFrame:
    rows = []
    keys = ["family", "endpoint", "set_id", "set_role", "alternative"]
    for key, group in seed_metrics.groupby(keys, sort=True):
        family, endpoint, set_id, set_role, alternative = key
        values = group.sort_values("seed")["value"].to_numpy(float)
        mean, low, high = _bootstrap_ci(
            values,
            repetitions=config.bootstrap_repetitions,
            seed=444_000 + sum(ord(char) for char in f"{endpoint}:{set_id}"),
        )
        rows.append(
            {
                "family": family,
                "endpoint": endpoint,
                "set_id": set_id,
                "set_role": set_role,
                "alternative": alternative,
                "seeds": int(group["seed"].nunique()),
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_seeds": int(np.sum(values > 0)),
                "negative_seeds": int(np.sum(values < 0)),
                "one_sided_exact_sign_flip_p": exact_sign_flip_p(
                    values, alternative=alternative
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_row(
    summary: pd.DataFrame, *, endpoint: str, role: str
) -> pd.Series:
    selected = summary[
        summary["endpoint"].eq(endpoint) & summary["set_role"].eq(role)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {role}/{endpoint} row, found {len(selected)}")
    return selected.iloc[0]


def primary_decision(summary: pd.DataFrame, config: V444Config) -> dict[str, Any]:
    family_endpoints = {
        "natural_signal": ("natural_carrier_count_slope",),
        "pre_o_injection": ("injection_dose_slope",),
        "centered_removal": (
            "removal_error_axis_minus_control",
            "removal_margin_axis_minus_control",
        ),
        "path_mediation": (
            "donor_patch_transport",
            "mediation_control_minus_axis_block",
        ),
    }
    families: dict[str, Any] = {}
    for family, endpoints in family_endpoints.items():
        components = []
        for endpoint in endpoints:
            for role, suffix in (
                ("candidate_core", ""),
                ("candidate_specificity", "__candidate_minus_control_mean"),
            ):
                row = _summary_row(summary, endpoint=endpoint + suffix, role=role)
                components.append(
                    {
                        "endpoint": str(row.endpoint),
                        "role": role,
                        "alternative": str(row.alternative),
                        "mean": float(row["mean"]),
                        "ci95_low": float(row.ci95_low),
                        "ci95_high": float(row.ci95_high),
                        "p": float(row.one_sided_exact_sign_flip_p),
                    }
                )
        family_p = max(item["p"] for item in components)
        families[family] = {
            "intersection_union_p": family_p,
            "passes_alpha": bool(family_p <= config.primary_alpha),
            "components": components,
        }
    global_p = max(item["intersection_union_p"] for item in families.values())
    return {
        "decision_rule": config.primary_decision_rule,
        "alpha": config.primary_alpha,
        "families": families,
        "global_intersection_union_p": global_p,
        "full_natural_ov_transporter_support": bool(global_p <= config.primary_alpha),
        "interpretation": (
            "All four pre-registered evidence families pass their conjunction."
            if global_p <= config.primary_alpha
            else "At least one required evidence family fails; full natural transporter use is not confirmed."
        ),
    }


def nested_k_analysis(
    seed_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    selection: Mapping[str, Any],
    config: V444Config,
) -> list[dict[str, Any]]:
    definitions = [selection["candidate"], *selection.get("registered_nested_sets", [])]
    family_endpoints = {
        "natural_signal": ("natural_carrier_count_slope",),
        "pre_o_injection": ("injection_dose_slope",),
        "centered_removal": (
            "removal_error_axis_minus_control",
            "removal_margin_axis_minus_control",
        ),
    }
    records: list[dict[str, Any]] = []
    for definition in definitions:
        set_id = str(definition["set_id"])
        role = str(definition["set_role"])
        record: dict[str, Any] = {
            "set_id": set_id,
            "heads": list(definition["heads"]),
            "k": len(definition["heads"]),
            "families": {},
        }
        for family, endpoints in family_endpoints.items():
            components = []
            for endpoint in endpoints:
                selected = summary[
                    summary["endpoint"].eq(endpoint)
                    & summary["set_id"].eq(set_id)
                    & summary["set_role"].eq(role)
                ]
                if len(selected) != 1:
                    raise RuntimeError(f"Nested-K summary lacks {set_id}/{endpoint}")
                row = selected.iloc[0]
                components.append(
                    {
                        "endpoint": endpoint,
                        "mean": float(row["mean"]),
                        "p": float(row.one_sided_exact_sign_flip_p),
                    }
                )
            record["families"][family] = {
                "raw_intersection_union_p": max(item["p"] for item in components),
                "components": components,
            }
        records.append(record)
    for family in family_endpoints:
        adjusted = _holm_adjust(
            [item["families"][family]["raw_intersection_union_p"] for item in records]
        )
        for item, value in zip(records, adjusted):
            item["families"][family]["holm_p_across_k"] = float(value)
            item["families"][family]["holm_passes_alpha"] = bool(
                value <= config.primary_alpha
            )
    return records


def factorial_analysis(
    seed_metrics: pd.DataFrame, selection: Mapping[str, Any], config: V444Config
) -> list[dict[str, Any]]:
    candidate_id = str(selection["candidate"]["set_id"])
    components = [str(item["set_id"]) for item in selection["factorial_components"]]
    if len(components) != 2:
        return []
    allowed = seed_metrics[
        seed_metrics["set_id"].isin([candidate_id, *components])
        & ~seed_metrics["family"].eq("path_mediation")
    ]
    rows = []
    for (family, endpoint, alternative), group in allowed.groupby(
        ["family", "endpoint", "alternative"], sort=True
    ):
        pivot = group.pivot(index="seed", columns="set_id", values="value").dropna()
        if not set([candidate_id, *components]).issubset(pivot.columns):
            continue
        synergy = (
            pivot[candidate_id] - pivot[components[0]] - pivot[components[1]]
        ).to_numpy(float)
        mean, low, high = _bootstrap_ci(
            synergy,
            repetitions=config.bootstrap_repetitions,
            seed=444_900 + sum(ord(char) for char in endpoint),
        )
        greater = exact_sign_flip_p(synergy, alternative="greater")
        less = exact_sign_flip_p(synergy, alternative="less")
        rows.append(
            {
                "family": family,
                "endpoint": endpoint,
                "definition": "joint_K2_minus_H16_minus_H19",
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "two_sided_exact_sign_flip_p": min(1.0, 2.0 * min(greater, less)),
                "seeds": int(len(pivot)),
            }
        )
    return rows


def baseline_summary(natural: pd.DataFrame) -> dict[str, Any]:
    baseline = natural[
        [
            "seed",
            "gold_count",
            "baseline_expected_count",
            "baseline_correct_margin",
            "baseline_predicted_count",
        ]
    ].drop_duplicates()
    expected = baseline["baseline_expected_count"].to_numpy(float)
    gold = baseline["gold_count"].to_numpy(float)
    predicted = baseline["baseline_predicted_count"].to_numpy(float)
    return {
        "rows": int(len(baseline)),
        "seeds": int(baseline["seed"].nunique()),
        "candidate_count_accuracy": float(np.mean(predicted == gold)),
        "expected_count_mae": float(np.mean(np.abs(expected - gold))),
        "mean_correct_margin": float(baseline["baseline_correct_margin"].mean()),
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        row = {}
        for key, value in raw.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            row[key] = value
        output.append(row)
    return output


def audit_campaign(run_root: str | Path, config: V444Config) -> dict[str, Any]:
    root = Path(run_root)
    model_root = root / "models" / config.model_label
    confirmation = model_root / "confirmation"
    selection = json.loads(
        (model_root / "center_controls" / "selection.json").read_text(encoding="utf-8")
    )
    entry_count = 1 + config.matched_control_count + 2 + len(
        config.secondary_nested_head_sets
    )
    expected = {
        "dataset_rows": len(config.dataset_seeds) * len(config.counts),
        "confirmation_seed_shards": len(config.confirmation_seeds),
        "natural_rows": len(config.confirmation_seeds)
        * entry_count
        * len(config.counts),
        "directed_rows": len(config.confirmation_seeds)
        * entry_count
        * len(config.causal_counts)
        * (2 + len(config.injection_betas)),
        "mediation_rows": len(config.confirmation_seeds)
        * (1 + config.matched_control_count)
        * len(config.mediation_pairs)
        * 3,
    }
    dataset_rows = sum(
        1 for _line in (root / "dataset" / "stimuli.jsonl").open("rb")
    )
    natural = pd.read_csv(confirmation / "natural_activation_detail.csv.gz")
    directed = pd.read_csv(confirmation / "directed_detail.csv.gz")
    mediation = pd.read_csv(confirmation / "mediation_detail.csv.gz")
    observed = {
        "dataset_rows": dataset_rows,
        "confirmation_seed_shards": len(
            list((confirmation / "natural_shards").glob("seed*.csv.gz"))
        ),
        "natural_rows": len(natural),
        "directed_rows": len(directed),
        "mediation_rows": len(mediation),
    }
    checks = {
        key: observed[key] == value for key, value in expected.items()
    }
    checks.update(
        {
            "four_matched_controls": len(selection["matched_controls"])
            == config.matched_control_count,
            "selection_precedes_causal_outcomes": not bool(
                selection["selection_uses_causal_outcomes"]
            ),
            "smoke_complete": (model_root / "smoke" / "complete.json").is_file(),
            "model_complete": (model_root / "complete.json").is_file(),
            "single_design_hash_natural": natural["design_hash"].nunique() == 1,
            "single_design_hash_directed": directed["design_hash"].nunique() == 1,
            "single_design_hash_mediation": mediation["design_hash"].nunique() == 1,
            "all_confirmation_seeds_present": set(natural["seed"].astype(int))
            == set(config.confirmation_seeds),
            "no_raw_attention_artifact": not any(root.rglob("*raw_attention*")),
            "no_full_hidden_state_artifact": not any(root.rglob("*full_hidden*")),
        }
    )
    return {
        "schema_version": "realistic_niah_v4_4_4_audit_v1",
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def _format_p(value: float) -> str:
    return f"{value:.6g}" if math.isfinite(value) else "NA"




def analyze_campaign(
    *, run_root: str | Path, config: V444Config
) -> dict[str, Any]:
    root = Path(run_root)
    model_root = root / "models" / config.model_label
    confirmation = model_root / "confirmation"
    natural = pd.read_csv(confirmation / "natural_activation_detail.csv.gz")
    directed = pd.read_csv(confirmation / "directed_detail.csv.gz")
    mediation = pd.read_csv(confirmation / "mediation_detail.csv.gz")
    selection = json.loads(
        (model_root / "center_controls" / "selection.json").read_text(encoding="utf-8")
    )
    seed_metrics = build_seed_metrics(natural, directed, mediation)
    seed_metrics = add_candidate_specificity(seed_metrics, selection)
    summary = summarize_seed_metrics(seed_metrics, config)
    decision = primary_decision(summary, config)
    audit = audit_campaign(root, config)
    if not audit["all_checks_pass"]:
        raise RuntimeError(f"V4.4.4 campaign audit failed: {audit['checks']}")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_analysis_v1",
        "config": config.to_dict(),
        "baseline": baseline_summary(natural),
        "primary_decision": decision,
        "nested_k": nested_k_analysis(seed_metrics, summary, selection, config),
        "factorial": factorial_analysis(seed_metrics, selection, config),
        "audit": audit,
        "summary": _json_records(summary),
    }
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv_gzip(seed_metrics, output / "seed_metrics.csv.gz")
    atomic_csv_gzip(summary, output / "endpoint_summary.csv.gz")
    atomic_json(output / "realistic_niah_v4_4_4_analysis.json", payload)
    atomic_json(output / "audit.json", audit)
    atomic_json(
        output / "complete.json",
        {
            "schema_version": "realistic_niah_v4_4_4_analysis_complete_v1",
            "full_natural_ov_transporter_support": decision[
                "full_natural_ov_transporter_support"
            ],
            "global_intersection_union_p": decision[
                "global_intersection_union_p"
            ],
            "audit_all_checks_pass": audit["all_checks_pass"],
        },
    )
    return payload
