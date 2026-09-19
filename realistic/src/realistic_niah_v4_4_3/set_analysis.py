from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .io import atomic_csv_gzip, atomic_json, atomic_text
from .set_spec import V443SetConfig


def _bootstrap(values: Sequence[float], *, seed: int, repetitions: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, math.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(int(seed))
    draws = generator.choice(array, size=(int(repetitions), len(array)), replace=True)
    low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
    return mean, float(low), float(high)


def _exact_sign_flip(values: Sequence[float], *, alternative: str) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan
    observed = float(array.mean())
    null = np.asarray(
        [
            np.mean(array * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=len(array))
        ],
        dtype=float,
    )
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-12))
    if alternative == "less":
        return float(np.mean(null <= observed + 1e-12))
    raise ValueError("alternative must be greater or less")


def _benjamini_hochberg(values: Sequence[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg q values in the input order."""
    array = np.asarray(values, dtype=float)
    output = np.full(array.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(array))
    if len(finite) == 0:
        return output.tolist()
    order = finite[np.argsort(array[finite], kind="stable")]
    adjusted = np.empty(len(order), dtype=float)
    running = 1.0
    count = len(order)
    for reverse_rank in range(count - 1, -1, -1):
        rank = reverse_rank + 1
        index = order[reverse_rank]
        running = min(running, float(array[index]) * count / rank)
        adjusted[reverse_rank] = min(1.0, running)
    output[order] = adjusted
    return output.tolist()


def _summarize_seed_effects(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str,
    alternative: str,
    config: V443SetConfig,
    seed_offset: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_columns), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        per_seed = group.groupby("seed")[value_column].mean().to_numpy(float)
        mean, low, high = _bootstrap(
            per_seed,
            seed=443700 + int(seed_offset),
            repetitions=config.mapping_null_repetitions,
        )
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "metric": value_column,
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "alternative": alternative,
                "one_sided_exact_sign_flip_p": _exact_sign_flip(
                    per_seed, alternative=alternative
                ),
                "seed_count": int(len(per_seed)),
            }
        )
    return pd.DataFrame(rows)


def summarize_patch(
    detail: pd.DataFrame, config: V443SetConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit = [
        "model_label",
        "seed",
        "receiver_count",
        "donor_count",
        "set_id",
        "set_role",
        "layer",
        "set_size",
        "heads",
    ]
    pivot = detail.pivot(
        index=unit,
        columns="intervention",
        values="continuous_normalized_transport",
    ).reset_index()
    pivot["z_transport_over_norm"] = pivot["z_donor"] - pivot["output_norm_control"]
    pivot["alpha_over_scramble"] = (
        pivot["alpha_receiver_v"] - pivot["alpha_position_scramble"]
    )
    role_rows = []
    for metric, alternative, offset in (
        ("z_transport_over_norm", "greater", 1),
        ("alpha_over_scramble", "greater", 2),
    ):
        role_rows.append(
            _summarize_seed_effects(
                pivot,
                group_columns=[
                    "model_label",
                    "set_id",
                    "set_role",
                    "layer",
                    "set_size",
                    "heads",
                ],
                value_column=metric,
                alternative=alternative,
                config=config,
                seed_offset=offset,
            )
        )
    role_summary = pd.concat(role_rows, ignore_index=True)
    pair_index = [column for column in unit if column not in {"set_role", "heads"}]
    specificity_rows = []
    for metric, alternative, offset in (
        ("z_transport_over_norm", "greater", 3),
        ("alpha_over_scramble", "greater", 4),
    ):
        paired = pivot.pivot(
            index=pair_index,
            columns="set_role",
            values=metric,
        ).reset_index()
        paired = paired.dropna(subset=["candidate_set", "matched_set"])
        paired[f"candidate_minus_matched_{metric}"] = (
            paired["candidate_set"] - paired["matched_set"]
        )
        specificity_rows.append(
            _summarize_seed_effects(
                paired,
                group_columns=[
                    "model_label",
                    "set_id",
                    "layer",
                    "set_size",
                ],
                value_column=f"candidate_minus_matched_{metric}",
                alternative=alternative,
                config=config,
                seed_offset=offset,
            )
        )
    return pivot, role_summary, pd.concat(specificity_rows, ignore_index=True)


def summarize_injection(
    detail: pd.DataFrame, config: V443SetConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = detail[
        detail["intervention"].astype(str).str.startswith(
            "signed_answer_direction_injection_beta_"
        )
    ].copy()
    seed_rows = []
    group_columns = [
        "model_label",
        "seed",
        "set_id",
        "set_role",
        "layer",
        "set_size",
        "heads",
    ]
    for key, group in frame.groupby(group_columns, sort=True):
        beta = group["beta"].to_numpy(float)
        outcome = group["delta_expected_count"].to_numpy(float)
        denominator = float(np.dot(beta, beta))
        slope = float(np.dot(beta, outcome) / denominator)
        seed_rows.append(
            {
                **dict(zip(group_columns, key)),
                "injection_slope": slope,
                "reachable_answer_cosine": float(
                    group["reachable_answer_cosine"].iloc[0]
                ),
                "mean_non_count_token_kl": float(group["non_count_token_kl"].mean()),
            }
        )
    seed_frame = pd.DataFrame(seed_rows)
    role_summary = _summarize_seed_effects(
        seed_frame,
        group_columns=[
            "model_label",
            "set_id",
            "set_role",
            "layer",
            "set_size",
            "heads",
        ],
        value_column="injection_slope",
        alternative="greater",
        config=config,
        seed_offset=5,
    )
    pair_index = [
        "model_label",
        "seed",
        "set_id",
        "layer",
        "set_size",
    ]
    paired = seed_frame.pivot(
        index=pair_index,
        columns="set_role",
        values="injection_slope",
    ).reset_index()
    paired["candidate_minus_matched_injection_slope"] = (
        paired["candidate_set"] - paired["matched_set"]
    )
    specificity = _summarize_seed_effects(
        paired,
        group_columns=["model_label", "set_id", "layer", "set_size"],
        value_column="candidate_minus_matched_injection_slope",
        alternative="greater",
        config=config,
        seed_offset=6,
    )
    return pd.concat([role_summary, specificity], ignore_index=True), seed_frame


def summarize_removal(
    detail: pd.DataFrame, config: V443SetConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = detail[
        detail["intervention"].isin(
            ["answer_direction_removal", "equal_norm_orthogonal_removal"]
        )
    ].copy()
    unit = [
        "model_label",
        "seed",
        "gold_count",
        "set_id",
        "set_role",
        "layer",
        "set_size",
        "heads",
    ]
    rows = []
    for metric, name in (
        ("delta_expected_count_absolute_error", "removal_error_over_orthogonal"),
        ("delta_correct_margin", "removal_margin_over_orthogonal"),
    ):
        pivot = frame.pivot(
            index=unit,
            columns="intervention",
            values=metric,
        ).reset_index()
        pivot[name] = (
            pivot["answer_direction_removal"]
            - pivot["equal_norm_orthogonal_removal"]
        )
        rows.append(pivot[unit + [name]])
    merged = rows[0].merge(rows[1], on=unit, validate="one_to_one")
    role_summaries = []
    specificity_summaries = []
    pair_index = [column for column in unit if column not in {"set_role", "heads"}]
    for metric, alternative, offset in (
        ("removal_error_over_orthogonal", "greater", 7),
        ("removal_margin_over_orthogonal", "less", 8),
    ):
        role_summaries.append(
            _summarize_seed_effects(
                merged,
                group_columns=[
                    "model_label",
                    "set_id",
                    "set_role",
                    "layer",
                    "set_size",
                    "heads",
                ],
                value_column=metric,
                alternative=alternative,
                config=config,
                seed_offset=offset,
            )
        )
        paired = merged.pivot(
            index=pair_index,
            columns="set_role",
            values=metric,
        ).reset_index()
        name = f"candidate_minus_matched_{metric}"
        paired[name] = paired["candidate_set"] - paired["matched_set"]
        specificity_summaries.append(
            _summarize_seed_effects(
                paired,
                group_columns=["model_label", "set_id", "layer", "set_size"],
                value_column=name,
                alternative=alternative,
                config=config,
                seed_offset=offset + 10,
            )
        )
    return (
        merged,
        pd.concat(role_summaries, ignore_index=True),
        pd.concat(specificity_summaries, ignore_index=True),
    )


def _nested_role_and_specificity(
    frame: pd.DataFrame,
    *,
    unit_columns: Sequence[str],
    metrics: Sequence[tuple[str, str, int]],
    config: V443SetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_rows: list[pd.DataFrame] = []
    specificity_rows: list[pd.DataFrame] = []
    role_groups = [
        "model_label",
        "set_id",
        "set_role",
        "layer",
        "set_size",
        "heads",
    ]
    pair_index = [
        column for column in unit_columns if column not in {"set_role", "heads"}
    ]
    for metric, alternative, offset in metrics:
        role_rows.append(
            _summarize_seed_effects(
                frame,
                group_columns=role_groups,
                value_column=metric,
                alternative=alternative,
                config=config,
                seed_offset=100 + offset,
            )
        )
        paired = frame.pivot(
            index=pair_index,
            columns="set_role",
            values=metric,
        ).reset_index()
        paired = paired.dropna(subset=["candidate_set", "matched_set"])
        contrast = f"candidate_minus_matched_{metric}"
        paired[contrast] = paired["candidate_set"] - paired["matched_set"]
        specificity_rows.append(
            _summarize_seed_effects(
                paired,
                group_columns=[
                    "model_label",
                    "set_id",
                    "layer",
                    "set_size",
                ],
                value_column=contrast,
                alternative=alternative,
                config=config,
                seed_offset=120 + offset,
            )
        )
    return pd.concat(role_rows, ignore_index=True), pd.concat(
        specificity_rows, ignore_index=True
    )


def _nested_incremental_summary(
    frame: pd.DataFrame,
    *,
    pairing_columns: Sequence[str],
    metrics: Sequence[tuple[str, str, int]],
    config: V443SetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = frame[frame["set_role"].eq("candidate_set")].copy()
    seed_rows: list[pd.DataFrame] = []
    for (model_label, layer), group in candidate.groupby(
        ["model_label", "layer"], sort=True
    ):
        sizes = sorted(int(value) for value in group["set_size"].unique())
        for previous, current in zip(sizes[:-1], sizes[1:]):
            left = group[group["set_size"].eq(previous)][
                list(pairing_columns) + [name for name, _alternative, _offset in metrics]
            ]
            right = group[group["set_size"].eq(current)][
                list(pairing_columns) + [name for name, _alternative, _offset in metrics]
            ]
            merged = left.merge(
                right,
                on=list(pairing_columns),
                suffixes=("_previous", "_current"),
                validate="one_to_one",
            )
            merged["model_label"] = model_label
            merged["layer"] = int(layer)
            merged["from_k"] = int(previous)
            merged["to_k"] = int(current)
            for metric, _alternative, _offset in metrics:
                merged[f"increment_{metric}"] = (
                    merged[f"{metric}_current"] - merged[f"{metric}_previous"]
                )
            seed_rows.append(merged)
    units = pd.concat(seed_rows, ignore_index=True) if seed_rows else pd.DataFrame()
    summaries = []
    for metric, alternative, offset in metrics:
        summaries.append(
            _summarize_seed_effects(
                units,
                group_columns=["model_label", "layer", "from_k", "to_k"],
                value_column=f"increment_{metric}",
                alternative=alternative,
                config=config,
                seed_offset=140 + offset,
            )
        )
    return units, pd.concat(summaries, ignore_index=True)


def build_nested_k_tables(
    model_root: Path,
    *,
    model_label: str,
    config: V443SetConfig,
    set_mapping: pd.DataFrame,
    set_patch_units: pd.DataFrame,
    set_removal_units: pd.DataFrame,
) -> dict[str, Any]:
    """Join the frozen K=1 run to the K>1 nested-set experiment.

    K=1 is comparable for mapping, staged patch, and removal.  It is deliberately
    excluded from injection curves because the earlier intervention was applied to
    the full layer post-O residual instead of the selected set's output span.
    """
    baseline_root = Path(config.single_head_baseline_run_root) / "models" / model_label
    selection = json.loads(
        (baseline_root / "mapping" / "head_selection.json").read_text(
            encoding="utf-8"
        )
    )
    head_scores = pd.read_csv(
        baseline_root / "mapping" / "head_mapping_scores.csv.gz"
    )
    mapping_rows: list[dict[str, Any]] = []
    k1_heads: dict[int, int] = {}
    for candidate in selection["candidate_heads"]:
        layer = int(candidate["layer"])
        head = int(candidate["head"])
        k1_heads[layer] = head
        control = selection["matched_control_heads"][f"L{layer}H{head}"]
        for role, selected_head in (
            ("candidate_set", head),
            ("matched_set", int(control["head"])),
        ):
            row = head_scores[
                head_scores["layer"].astype(int).eq(layer)
                & head_scores["head"].astype(int).eq(selected_head)
            ].iloc[0]
            mapping_rows.append(
                {
                    "model_label": model_label,
                    "set_id": f"L{layer}K1",
                    "set_role": role,
                    "layer": layer,
                    "set_size": 1,
                    "heads": str(selected_head),
                    "fit_mapping_cosine": float(row["fit_mapping_cosine"]),
                    "heldout_count_mapping_cosine": float(
                        row["heldout_count_mapping_cosine"]
                    ),
                    "fit_mapped_norm": float(row["fit_mapped_norm"]),
                    "heldout_mapped_norm": float(row["heldout_mapped_norm"]),
                    "source_experiment": "V4.4.3 single-head baseline",
                }
            )
    mapping_new = set_mapping.copy()
    mapping_new["source_experiment"] = "V4.4.3-Set"
    mapping_curve = pd.concat(
        [pd.DataFrame(mapping_rows), mapping_new], ignore_index=True, sort=False
    )
    anchor_checks = []
    for layer, head in k1_heads.items():
        first_size = min(config.set_sizes_for(model_label))
        row = set_mapping[
            set_mapping["set_role"].eq("candidate_set")
            & set_mapping["layer"].astype(int).eq(layer)
            & set_mapping["set_size"].astype(int).eq(first_size)
        ].iloc[0]
        members = {int(value) for value in str(row["heads"]).split(",")}
        anchor_checks.append(
            {
                "layer": layer,
                "k1_head": head,
                "first_set_size": int(first_size),
                "first_set_heads": sorted(members),
                "k1_is_nested_anchor": bool(head in members),
            }
        )
    if not all(item["k1_is_nested_anchor"] for item in anchor_checks):
        raise RuntimeError("The frozen K=1 head is not the anchor of a nested set")

    old_patch = pd.read_csv(baseline_root / "staged_patch" / "detail.csv.gz")
    patch_index = [
        "model_label",
        "seed",
        "receiver_count",
        "donor_count",
        "parent_candidate",
        "head_role",
        "layer",
        "head",
    ]
    old_patch = old_patch.pivot(
        index=patch_index,
        columns="intervention",
        values="continuous_normalized_transport",
    ).reset_index()
    old_patch["z_transport_over_norm"] = (
        old_patch["z_donor"] - old_patch["output_norm_control"]
    )
    old_patch["alpha_over_scramble"] = (
        old_patch["alpha_receiver_v"] - old_patch["alpha_position_scramble"]
    )
    old_patch["set_role"] = old_patch["head_role"].map(
        {"candidate": "candidate_set", "matched_control": "matched_set"}
    )
    old_patch["set_size"] = 1
    old_patch["set_id"] = old_patch["layer"].map(
        lambda value: f"L{int(value)}K1"
    )
    old_patch["heads"] = old_patch["head"].astype(int).astype(str)
    patch_unit_columns = [
        "model_label",
        "seed",
        "receiver_count",
        "donor_count",
        "set_id",
        "set_role",
        "layer",
        "set_size",
        "heads",
    ]
    nested_patch = pd.concat(
        [
            old_patch[
                patch_unit_columns
                + ["z_transport_over_norm", "alpha_over_scramble"]
            ],
            set_patch_units[
                patch_unit_columns
                + ["z_transport_over_norm", "alpha_over_scramble"]
            ],
        ],
        ignore_index=True,
    )
    patch_metrics = (
        ("z_transport_over_norm", "greater", 1),
        ("alpha_over_scramble", "greater", 2),
    )
    nested_patch_roles, nested_patch_specificity = _nested_role_and_specificity(
        nested_patch,
        unit_columns=patch_unit_columns,
        metrics=patch_metrics,
        config=config,
    )
    patch_increment_units, patch_increment_summary = _nested_incremental_summary(
        nested_patch,
        pairing_columns=["seed", "receiver_count", "donor_count"],
        metrics=patch_metrics,
        config=config,
    )

    old_directed = pd.read_csv(baseline_root / "directed" / "detail.csv.gz")
    old_directed = old_directed[
        old_directed["intervention"].isin(
            ["answer_direction_removal", "equal_norm_orthogonal_removal"]
        )
    ].copy()
    removal_index = [
        "model_label",
        "seed",
        "gold_count",
        "parent_candidate",
        "head_role",
        "layer",
        "head",
    ]
    old_removal_parts = []
    for source_metric, result_metric in (
        ("delta_expected_count_absolute_error", "removal_error_over_orthogonal"),
        ("delta_correct_margin", "removal_margin_over_orthogonal"),
    ):
        pivot = old_directed.pivot(
            index=removal_index,
            columns="intervention",
            values=source_metric,
        ).reset_index()
        pivot[result_metric] = (
            pivot["answer_direction_removal"]
            - pivot["equal_norm_orthogonal_removal"]
        )
        old_removal_parts.append(pivot[removal_index + [result_metric]])
    old_removal = old_removal_parts[0].merge(
        old_removal_parts[1], on=removal_index, validate="one_to_one"
    )
    old_removal["set_role"] = old_removal["head_role"].map(
        {"candidate": "candidate_set", "matched_control": "matched_set"}
    )
    old_removal["set_size"] = 1
    old_removal["set_id"] = old_removal["layer"].map(
        lambda value: f"L{int(value)}K1"
    )
    old_removal["heads"] = old_removal["head"].astype(int).astype(str)
    removal_unit_columns = [
        "model_label",
        "seed",
        "gold_count",
        "set_id",
        "set_role",
        "layer",
        "set_size",
        "heads",
    ]
    removal_metric_names = [
        "removal_error_over_orthogonal",
        "removal_margin_over_orthogonal",
    ]
    nested_removal = pd.concat(
        [
            old_removal[removal_unit_columns + removal_metric_names],
            set_removal_units[removal_unit_columns + removal_metric_names],
        ],
        ignore_index=True,
    )
    removal_metrics = (
        ("removal_error_over_orthogonal", "greater", 3),
        ("removal_margin_over_orthogonal", "less", 4),
    )
    nested_removal_roles, nested_removal_specificity = (
        _nested_role_and_specificity(
            nested_removal,
            unit_columns=removal_unit_columns,
            metrics=removal_metrics,
            config=config,
        )
    )
    removal_increment_units, removal_increment_summary = (
        _nested_incremental_summary(
            nested_removal,
            pairing_columns=["seed", "gold_count"],
            metrics=removal_metrics,
            config=config,
        )
    )

    output = model_root / "analysis"
    tables = {
        "nested_mapping_curve.csv.gz": mapping_curve,
        "nested_patch_units.csv.gz": nested_patch,
        "nested_patch_role_summary.csv.gz": nested_patch_roles,
        "nested_patch_specificity.csv.gz": nested_patch_specificity,
        "nested_patch_increment_units.csv.gz": patch_increment_units,
        "nested_patch_increment_summary.csv.gz": patch_increment_summary,
        "nested_removal_units.csv.gz": nested_removal,
        "nested_removal_role_summary.csv.gz": nested_removal_roles,
        "nested_removal_specificity.csv.gz": nested_removal_specificity,
        "nested_removal_increment_units.csv.gz": removal_increment_units,
        "nested_removal_increment_summary.csv.gz": removal_increment_summary,
    }
    for filename, frame in tables.items():
        atomic_csv_gzip(frame, output / filename)
    return {
        "single_head_baseline_run_root": str(config.single_head_baseline_run_root),
        "k1_anchor_checks": anchor_checks,
        "injection_k1_comparable": False,
        "injection_k1_exclusion_reason": (
            "the K=1 baseline injected at the full layer post-O residual; "
            "K>1 injection is constrained to the selected set output span"
        ),
        "table_files": sorted(tables),
    }


def _summary_lookup(frame: pd.DataFrame, set_id: str, metric_contains: str) -> dict[str, Any]:
    selected = frame[
        frame["set_id"].astype(str).eq(set_id)
        & frame["metric"].astype(str).str.contains(metric_contains, regex=False)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {set_id}/{metric_contains} summary row")
    output: dict[str, Any] = {}
    for key, value in selected.iloc[0].to_dict().items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        output[key] = value
    return output


def analyze_set_model(
    run_root: str | Path,
    *,
    model_label: str,
    config: V443SetConfig,
) -> dict[str, Any]:
    model_root = Path(run_root) / "models" / model_label
    mapping = pd.read_csv(model_root / "mapping" / "set_mapping_scores.csv.gz")
    patch = pd.read_csv(model_root / "staged_patch" / "detail.csv.gz")
    directed = pd.read_csv(model_root / "directed" / "detail.csv.gz")
    patch_units, patch_roles, patch_specificity = summarize_patch(patch, config)
    injection_summary, injection_seed = summarize_injection(directed, config)
    removal_units, removal_roles, removal_specificity = summarize_removal(
        directed, config
    )
    patch_selectivity = patch[patch["intervention"].eq("z_donor")].copy()
    patch_selectivity["intervention_family"] = "z_donor"
    directed_selectivity = directed.copy()
    directed_selectivity["intervention_family"] = directed_selectivity[
        "intervention"
    ]
    injection_mask = directed_selectivity["intervention"].astype(str).str.startswith(
        "signed_answer_direction_injection_beta_"
    ) & directed_selectivity["beta"].fillna(0).ne(0)
    directed_selectivity.loc[
        injection_mask, "intervention_family"
    ] = "signed_injection_nonzero_beta"
    directed_selectivity = directed_selectivity[
        directed_selectivity["intervention_family"].isin(
            [
                "signed_injection_nonzero_beta",
                "answer_direction_removal",
                "equal_norm_orthogonal_removal",
            ]
        )
    ]
    local_selectivity = pd.concat(
        [patch_selectivity, directed_selectivity], ignore_index=True, sort=False
    )
    local_selectivity = (
        local_selectivity.groupby(
            [
                "model_label",
                "set_id",
                "set_role",
                "layer",
                "set_size",
                "heads",
                "intervention_family",
            ],
            sort=True,
            as_index=False,
        )[
            [
                "non_count_token_kl",
                "count_subspace_delta_fraction",
                "count_token_logit_delta_l2",
                "full_vocab_logit_delta_l2",
            ]
        ]
        .mean()
    )
    output = model_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for frame, filename in (
        (patch_units, "patch_units.csv.gz"),
        (patch_roles, "patch_role_summary.csv.gz"),
        (patch_specificity, "patch_specificity.csv.gz"),
        (injection_seed, "injection_seed_slopes.csv.gz"),
        (injection_summary, "injection_summary.csv.gz"),
        (removal_units, "removal_units.csv.gz"),
        (removal_roles, "removal_role_summary.csv.gz"),
        (removal_specificity, "removal_specificity.csv.gz"),
        (local_selectivity, "local_selectivity_summary.csv.gz"),
    ):
        atomic_csv_gzip(frame, output / filename)
    nested_k = build_nested_k_tables(
        model_root,
        model_label=model_label,
        config=config,
        set_mapping=mapping,
        set_patch_units=patch_units,
        set_removal_units=removal_units,
    )
    decisions = []
    candidate_mapping = mapping[mapping["set_role"].eq("candidate_set")]
    for row in candidate_mapping.itertuples():
        set_id = str(row.set_id)
        z = _summary_lookup(
            patch_specificity,
            set_id,
            "candidate_minus_matched_z_transport_over_norm",
        )
        alpha = _summary_lookup(
            patch_specificity,
            set_id,
            "candidate_minus_matched_alpha_over_scramble",
        )
        injection = _summary_lookup(
            injection_summary,
            set_id,
            "candidate_minus_matched_injection_slope",
        )
        removal_error = _summary_lookup(
            removal_specificity,
            set_id,
            "candidate_minus_matched_removal_error_over_orthogonal",
        )
        removal_margin = _summary_lookup(
            removal_specificity,
            set_id,
            "candidate_minus_matched_removal_margin_over_orthogonal",
        )
        mapping_generalizes = bool(
            float(row.fit_mapping_cosine) > 0
            and float(row.heldout_count_mapping_cosine) > 0
        )
        family_passes = {
            "z_transport": bool(
                float(z["mean"]) > 0
                and float(z["one_sided_exact_sign_flip_p"]) <= 0.05
            ),
            "set_reachable_injection": bool(
                float(injection["mean"]) > 0
                and float(injection["one_sided_exact_sign_flip_p"]) <= 0.05
            ),
            "answer_direction_removal": bool(
                float(removal_error["mean"]) > 0
                and float(removal_margin["mean"]) < 0
                and float(removal_error["one_sided_exact_sign_flip_p"]) <= 0.05
                and float(removal_margin["one_sided_exact_sign_flip_p"]) <= 0.05
            ),
        }
        decisions.append(
            {
                "model_label": model_label,
                "set_id": set_id,
                "layer": int(row.layer),
                "set_size": int(row.set_size),
                "heads": str(row.heads),
                "fit_mapping_cosine": float(row.fit_mapping_cosine),
                "heldout_count_mapping_cosine": float(
                    row.heldout_count_mapping_cosine
                ),
                "mapping_generalizes": mapping_generalizes,
                "z_specificity": z,
                "alpha_specificity": alpha,
                "injection_specificity": injection,
                "removal_error_specificity": removal_error,
                "removal_margin_specificity": removal_margin,
                "causal_family_passes": family_passes,
                "causal_family_pass_count": int(sum(family_passes.values())),
                "triangulated_set_support": bool(
                    mapping_generalizes and sum(family_passes.values()) >= 2
                ),
            }
        )
    # The enlarged K grid creates multiple candidate sets per model.  Preserve
    # the frozen raw exact-test decision, but use within-family BH correction for
    # the report's inferential headline.
    family_p_values: dict[str, list[float]] = {
        "z_transport": [
            float(item["z_specificity"]["one_sided_exact_sign_flip_p"])
            for item in decisions
        ],
        "set_reachable_injection": [
            float(item["injection_specificity"]["one_sided_exact_sign_flip_p"])
            for item in decisions
        ],
        "answer_direction_removal": [
            max(
                float(item["removal_error_specificity"]["one_sided_exact_sign_flip_p"]),
                float(item["removal_margin_specificity"]["one_sided_exact_sign_flip_p"]),
            )
            for item in decisions
        ],
    }
    family_q_values = {
        family: _benjamini_hochberg(values)
        for family, values in family_p_values.items()
    }
    for index, item in enumerate(decisions):
        raw_passes = dict(item["causal_family_passes"])
        q_values = {
            family: float(values[index])
            for family, values in family_q_values.items()
        }
        fdr_passes = {
            family: bool(raw_passes[family] and q_values[family] <= 0.05)
            for family in raw_passes
        }
        item["causal_family_q_values_bh"] = q_values
        item["causal_family_passes_fdr"] = fdr_passes
        item["causal_family_pass_count_fdr"] = int(sum(fdr_passes.values()))
        item["triangulated_set_support_raw"] = bool(
            item["triangulated_set_support"]
        )
        item["triangulated_set_support_fdr"] = bool(
            item["mapping_generalizes"] and sum(fdr_passes.values()) >= 2
        )
        item["triangulated_set_support"] = item[
            "triangulated_set_support_fdr"
        ]
    payload = {
        "schema_version": "realistic_niah_v4_4_3_set_model_analysis_v1",
        "model_label": model_label,
        "nested_k_analysis": nested_k,
        "candidate_set_decisions": decisions,
        "any_triangulated_set_support": any(
            item["triangulated_set_support"] for item in decisions
        ),
        "any_triangulated_set_support_raw": any(
            item["triangulated_set_support_raw"] for item in decisions
        ),
        "definition_boundary": {
            "triangulated_set_support": (
                "held-out mapping direction is positive and at least two of three "
                "distinct intervention families have candidate-vs-matched BH q<=0.05"
            ),
            "raw_screening_support": (
                "the same rule using unadjusted one-sided exact p<=0.05"
            ),
            "multiple_testing": (
                "Benjamini-Hochberg correction is applied within model and causal family "
                "across the enlarged nested-K grid"
            ),
            "irreducibility": (
                "not implied; a sufficient set can contain redundant members"
            ),
        },
    }
    atomic_json(output / "analysis.json", payload)
    return payload


def analyze_set_campaign(
    run_root: str | Path,
    *,
    config: V443SetConfig,
) -> dict[str, Any]:
    analyses = {
        model: analyze_set_model(
            run_root, model_label=model, config=config
        )
        for model in config.model_labels
    }
    supported = {
        model: bool(payload["any_triangulated_set_support"])
        for model, payload in analyses.items()
    }
    raw_supported = {
        model: bool(payload["any_triangulated_set_support_raw"])
        for model, payload in analyses.items()
    }
    if any(supported.values()):
        conclusion = (
            "至少一个预冻结 head set 在 held-out mapping 与三类不同干预家族中的至少两类上获得 "
            "BH-FDR 校正后的 matched-set 特异支持；"
            "这支持小型 set 的因果充分性，但不自动证明每个成员不可替代。"
        )
    elif any(raw_supported.values()):
        conclusion = (
            "至少一个 nested head set 通过未校正的 2-of-3 exact-p 筛选，但在模型内、证据族内跨全部 "
            "layer×K 做 BH 校正后没有集合保留 2-of-3 支持；因此目前只能把这些 set 视为后续确认候选，"
            "不能宣称已确认小型 circuit。"
        )
    else:
        conclusion = (
            "当前预冻结的 nested 层内 head sets 即使在未校正筛选下，也未形成跨至少两类因果检验的 "
            "matched-set 特异支持；"
            "这不排除跨层 set、MLP 或更大分布式 circuit。"
        )
    lines = [
        "# Realistic NIAH V4.4.3-Set：OV vertical geometry 因果检验",
        "",
        "## 1. 猜想",
        "",
        "prompt running-index 与 answer-count 表示同一计数变量，但使用近似正交的 residual 载体方向；一个小型 attention-head set 通过 QK 定位和 OV 写回共同完成重编码。",
        "",
        "可证伪预测包括：held-out OV mapping 保持同号；Z transport 超过等范数输出 control；donor-α 超过位置打乱 α；set-output span 内的 signed injection 产生正 dose response；answer-direction removal 比等范数正交 removal 更损害计数。",
        "",
        "**本节结论：** 单头失败不能否定该猜想；合适的下一检验单位是预先冻结、并和同规模 matched set 比较的小型 head set。",
        "",
        "## 2. 实验设计与定义",
        "",
        "- discovery seeds 1234--1253；fit counts 1/3/5/7/9 选择 nested sets（Qwen K=1/2/3/4/6/8；Gemma K=1/2/3/4）；held-out counts 2/4/6/8/10 只评估。K=1 复用旧单头 run。",
        "- screen seeds 1254--1258 做 alpha/Z/O 分阶段 patch；confirmation seeds 1259--1263 做 removal 与 set-reachable injection。",
        "- set mapping：`m_S = sum_h M_OV^h u_prompt`，`r_S = cos(m_S, u_answer)`。",
        "- transport：`T = (E[N|patch] - E[N|base]) / (donor_count - receiver_count)`。",
        "- injection direction：`u_answer,S = normalize(P_col([W_O^h]_{h in S}) u_answer)`；斜率 `b = sum beta*DeltaE[N] / sum beta^2`。",
        "- removal：error contrast 的正值与 correct-margin contrast 的负值共同支持方向性损伤。",
        "- 每个 candidate set 配一个同层、同 K、成员不重叠且 OV 输出范数匹配的 control set。",
        "- 因果统计以 seed 为单位做单侧 exact sign-flip；扩大 K 后，在每个模型、每类 causal family 内跨全部 layer×K 做 Benjamini-Hochberg 校正。",
        "",
        "**本节结论：** set 选择、held-out 几何、screen 与 confirmation 完全分离；注入已限制到 set 输出子空间；主结论采用 BH q<=0.05，而不是从多个 K 中挑 raw p<=0.05。",
        "",
        "## 3. 具体结果",
        "",
    ]
    for model, payload in analyses.items():
        lines.extend(
            [
                f"### {model}",
                "",
                "| set | heads | fit map | held-out map | raw families (Z/I/R) | BH q (Z/I/R) | raw 2/3 | FDR 2/3 |",
                "|---|---|---:|---:|---|---|---|---|",
            ]
        )
        for item in payload["candidate_set_decisions"]:
            raw = item["causal_family_passes"]
            q = item["causal_family_q_values_bh"]
            lines.append(
                "| {set_id} | {heads} | {fit:.4g} | {held:.4g} | {z}/{inj}/{rem} | "
                "{zq:.5f}/{iq:.5f}/{rq:.5f} | {raw_joint} | {fdr_joint} |".format(
                    set_id=item["set_id"],
                    heads=item["heads"],
                    fit=item["fit_mapping_cosine"],
                    held=item["heldout_count_mapping_cosine"],
                    z=raw["z_transport"],
                    inj=raw["set_reachable_injection"],
                    rem=raw["answer_direction_removal"],
                    zq=q["z_transport"],
                    iq=q["set_reachable_injection"],
                    rq=q["answer_direction_removal"],
                    raw_joint=item["triangulated_set_support_raw"],
                    fdr_joint=item["triangulated_set_support_fdr"],
                )
            )
        lines.extend(
            [
                "",
                f"**本节结论：** {model} 的未校正 2-of-3 support={payload['any_triangulated_set_support_raw']}；BH-FDR 2-of-3 support={payload['any_triangulated_set_support']}。",
                "",
            ]
        )
    lines.extend(
        [
            "## 4. 综合分析",
            "",
            conclusion,
            "",
            "set sufficiency、set specificity 与 member irreducibility 是三个不同命题。本实验用 candidate-vs-matched 检验前两者，但没有做 leave-one-head-out，因此不检验成员不可替代性。另有四个边界：预选层来自既有 geometry；每个 causal split 只有 5 seeds；只搜索同层 sets；更大的 K 可能只增加可干预子空间维数，因此必须结合 matched specificity 与边际增益解释。",
            "",
            "**本节结论：** 只有 held-out mapping 与至少两类 BH-FDR matched-set 因果证据汇合，才支持小型 set 的因果充分性；raw 显著或单一 family 只作为确认线索。",
            "",
        ]
    )
    analysis_root = Path(run_root) / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    report_path = analysis_root / "realistic_niah_v4_4_3_ov_set_causal_report.md"
    atomic_text(report_path, "\n".join(lines))
    payload = {
        "schema_version": "realistic_niah_v4_4_3_set_campaign_analysis_v1",
        "model_support": supported,
        "model_support_raw": raw_supported,
        "multiple_testing": (
            "Benjamini-Hochberg within each model and causal family across layer-by-K sets"
        ),
        "conclusion": conclusion,
        "model_analyses": analyses,
        "report_path": str(report_path),
    }
    atomic_json(analysis_root / "analysis.json", payload)
    return payload


def audit_set_campaign(
    run_root: str | Path,
    *,
    config: V443SetConfig,
) -> dict[str, Any]:
    root = Path(run_root)
    models: dict[str, Any] = {}
    complete = True
    for model in config.model_labels:
        model_root = root / "models" / model
        entry_count = (
            len(config.target_layers(model))
            * len(config.set_sizes_for(model))
            * 2
        )
        expected_patch_shards = (
            len(config.screen_seeds)
            * len(config.directed_patch_pairs)
            * entry_count
        )
        expected_directed_shards = (
            len(config.confirmation_seeds)
            * len(config.injection_counts)
            * entry_count
        )
        patch_shards = list((model_root / "staged_patch" / "shards").glob("*.csv.gz"))
        directed_shards = list((model_root / "directed" / "shards").glob("*.csv.gz"))
        patch_rows = sum(len(pd.read_csv(path)) for path in patch_shards)
        directed_rows = sum(len(pd.read_csv(path)) for path in directed_shards)
        model_complete = bool(
            len(patch_shards) == expected_patch_shards
            and patch_rows == expected_patch_shards * len(config.patch_interventions)
            and len(directed_shards) == expected_directed_shards
            and directed_rows
            == expected_directed_shards * (2 + len(config.injection_betas))
            and (model_root / "analysis" / "analysis.json").is_file()
            and (model_root / "complete.json").is_file()
        )
        complete = complete and model_complete
        models[model] = {
            "expected_patch_shards": expected_patch_shards,
            "actual_patch_shards": len(patch_shards),
            "patch_rows": patch_rows,
            "expected_directed_shards": expected_directed_shards,
            "actual_directed_shards": len(directed_shards),
            "directed_rows": directed_rows,
            "complete": model_complete,
        }
    payload = {
        "schema_version": "realistic_niah_v4_4_3_set_audit_v1",
        "complete": bool(
            complete
            and (root / "analysis" / "analysis.json").is_file()
            and (root / "analysis" / "realistic_niah_v4_4_3_ov_set_causal_report.md").is_file()
        ),
        "models": models,
    }
    atomic_json(root / "audit.json", payload)
    if payload["complete"]:
        atomic_json(
            root / "complete.json",
            {
                "schema_version": "realistic_niah_v4_4_3_set_complete_v1",
                "state": "COMPLETE",
                "audit": "audit.json",
                "report": "analysis/realistic_niah_v4_4_3_ov_set_causal_report.md",
            },
        )
    return payload
