from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .io import atomic_csv_gzip, atomic_json, atomic_text
from .spec import V443Config


def _bootstrap_seed_mean(
    frame: pd.DataFrame,
    column: str,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    per_seed = frame.groupby("seed", as_index=False)[column].mean()
    values = per_seed[column].to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(int(repetitions), len(values)), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def _exact_sign_flip_p(values: Sequence[float], *, alternative: str) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan
    observed = float(array.mean())
    null = np.asarray(
        [
            np.mean(array * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=len(array))
        ]
    )
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-12))
    if alternative == "less":
        return float(np.mean(null <= observed + 1e-12))
    raise ValueError("alternative must be greater or less")


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert dataframe records without emitting non-standard JSON NaN."""

    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                record[key] = None
            elif isinstance(value, np.generic):
                record[key] = value.item()
            else:
                record[key] = value
        records.append(record)
    return records


def summarize_patch(detail: pd.DataFrame, config: V443Config) -> pd.DataFrame:
    required = {
        "model_label",
        "seed",
        "parent_candidate",
        "head_role",
        "layer",
        "head",
        "intervention",
        "continuous_normalized_transport",
        "delta_expected_count",
        "non_count_token_kl",
        "output_delta_norm",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Patch detail is missing columns: {missing}")
    rows = []
    group_columns = [
        "model_label",
        "parent_candidate",
        "head_role",
        "layer",
        "head",
        "intervention",
    ]
    for key, group in detail.groupby(group_columns, sort=True, dropna=False):
        mean, low, high = _bootstrap_seed_mean(
            group,
            "continuous_normalized_transport",
            repetitions=config.mapping_null_repetitions,
            seed=443001 + int(group["layer"].iloc[0]) * 101 + int(group["head"].iloc[0]),
        )
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "rows": int(len(group)),
                "seeds": int(group["seed"].nunique()),
                "mean_continuous_normalized_transport": mean,
                "transport_ci95_low": low,
                "transport_ci95_high": high,
                "mean_delta_expected_count": float(
                    group["delta_expected_count"].mean()
                ),
                "mean_non_count_token_kl": float(
                    group["non_count_token_kl"].mean()
                ),
                "mean_output_delta_norm": float(group["output_delta_norm"].mean()),
                "max_z_o_candidate_logit_abs_delta": float(
                    group["z_o_candidate_logit_max_abs_delta"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_intervention_contrast(
    detail: pd.DataFrame,
    *,
    left: str,
    right: str,
    value: str,
    contrast_name: str,
) -> pd.DataFrame:
    index = [
        "model_label",
        "seed",
        "receiver_count",
        "donor_count",
        "parent_candidate",
        "head_role",
        "layer",
        "head",
    ]
    selected = detail[detail["intervention"].isin([left, right])]
    pivot = selected.pivot(index=index, columns="intervention", values=value).reset_index()
    pivot = pivot.dropna(subset=[left, right])
    pivot[contrast_name] = pivot[left] - pivot[right]
    return pivot


def patch_contrasts(detail: pd.DataFrame, config: V443Config) -> pd.DataFrame:
    definitions = (
        (
            "z_donor",
            "output_norm_control",
            "z_minus_norm_control_transport",
        ),
        (
            "alpha_receiver_v",
            "alpha_position_scramble",
            "alpha_minus_scramble_transport",
        ),
    )
    rows = []
    for left, right, name in definitions:
        paired = _paired_intervention_contrast(
            detail,
            left=left,
            right=right,
            value="continuous_normalized_transport",
            contrast_name="contrast",
        )
        group_columns = [
            "model_label",
            "parent_candidate",
            "head_role",
            "layer",
            "head",
        ]
        for key, group in paired.groupby(group_columns, sort=True):
            mean, low, high = _bootstrap_seed_mean(
                group,
                "contrast",
                repetitions=config.mapping_null_repetitions,
                seed=443101 + int(group["layer"].iloc[0]) * 101 + int(group["head"].iloc[0]),
            )
            per_seed = group.groupby("seed")["contrast"].mean().to_numpy(float)
            rows.append(
                {
                    **dict(zip(group_columns, key)),
                    "contrast_name": name,
                    "mean_contrast": mean,
                    "contrast_ci95_low": low,
                    "contrast_ci95_high": high,
                    "one_sided_sign_flip_p": _exact_sign_flip_p(
                        per_seed, alternative="greater"
                    ),
                    "seeds": int(group["seed"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def summarize_injection(detail: pd.DataFrame, config: V443Config) -> pd.DataFrame:
    frame = detail[
        detail["intervention"].astype(str).str.startswith(
            "signed_answer_direction_injection_beta_"
        )
    ].copy()
    frame = frame[np.isfinite(frame["beta"].astype(float))]
    group_columns = [
        "model_label",
        "parent_candidate",
        "head_role",
        "layer",
        "head",
    ]
    rows = []
    for key, group in frame.groupby(group_columns, sort=True):
        beta = group["beta"].to_numpy(float)
        outcome = group["delta_expected_count"].to_numpy(float)
        denominator = float(np.dot(beta, beta))
        slope = float(np.dot(beta, outcome) / denominator) if denominator else math.nan
        rho = float(spearmanr(beta, outcome).statistic)
        seed_slopes = []
        for _seed, seed_group in group.groupby("seed", sort=True):
            x = seed_group["beta"].to_numpy(float)
            y = seed_group["delta_expected_count"].to_numpy(float)
            denom = float(np.dot(x, x))
            seed_slopes.append(float(np.dot(x, y) / denom) if denom else math.nan)
        generator = np.random.default_rng(
            443201 + int(group["layer"].iloc[0]) * 101 + int(group["head"].iloc[0])
        )
        finite = np.asarray(seed_slopes, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) > 1:
            boot = generator.choice(
                finite,
                size=(config.mapping_null_repetitions, len(finite)),
                replace=True,
            ).mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
        elif len(finite) == 1:
            low = high = finite[0]
        else:
            low = high = math.nan
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "rows": int(len(group)),
                "seeds": int(group["seed"].nunique()),
                "counts": int(group["gold_count"].nunique()),
                "injection_slope_expected_count_per_beta": slope,
                "slope_ci95_low": float(low),
                "slope_ci95_high": float(high),
                "beta_outcome_spearman": rho,
                "positive_seed_slopes": int(np.sum(finite > 0)),
                "one_sided_sign_flip_p": _exact_sign_flip_p(
                    finite, alternative="greater"
                ),
                "mean_non_count_token_kl": float(
                    group["non_count_token_kl"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_removal(detail: pd.DataFrame, config: V443Config) -> pd.DataFrame:
    left = "answer_direction_removal"
    right = "equal_norm_orthogonal_removal"
    index = [
        "model_label",
        "seed",
        "gold_count",
        "parent_candidate",
        "head_role",
        "layer",
        "head",
    ]
    selected = detail[detail["intervention"].isin([left, right])]
    frames = []
    for metric in (
        "delta_expected_count_absolute_error",
        "delta_correct_margin",
        "non_count_token_kl",
    ):
        pivot = selected.pivot(
            index=index, columns="intervention", values=metric
        ).reset_index()
        pivot[f"{metric}_answer_minus_orthogonal"] = pivot[left] - pivot[right]
        frames.append((metric, pivot))
    merged = frames[0][1]
    for metric, frame in frames[1:]:
        keep = index + [f"{metric}_answer_minus_orthogonal"]
        merged = merged.merge(frame[keep], on=index, how="inner")
    group_columns = [
        "model_label",
        "parent_candidate",
        "head_role",
        "layer",
        "head",
    ]
    rows = []
    for key, group in merged.groupby(group_columns, sort=True):
        row = {**dict(zip(group_columns, key)), "seeds": int(group["seed"].nunique())}
        for metric, _frame in frames:
            name = f"{metric}_answer_minus_orthogonal"
            mean, low, high = _bootstrap_seed_mean(
                group,
                name,
                repetitions=config.mapping_null_repetitions,
                seed=443301
                + int(group["layer"].iloc[0]) * 101
                + int(group["head"].iloc[0]),
            )
            row[f"mean_{name}"] = mean
            row[f"{name}_ci95_low"] = low
            row[f"{name}_ci95_high"] = high
            per_seed = group.groupby("seed")[name].mean().to_numpy(float)
            alternative = "less" if metric == "delta_correct_margin" else "greater"
            row[f"{name}_one_sided_sign_flip_p"] = _exact_sign_flip_p(
                per_seed, alternative=alternative
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_head_specificity(
    patch: pd.DataFrame,
    directed: pd.DataFrame,
    config: V443Config,
) -> pd.DataFrame:
    """Pair every selected-head effect with its registered matched head."""

    rows: list[dict[str, Any]] = []

    def append_summary(
        frame: pd.DataFrame,
        *,
        contrast_name: str,
        alternative: str | None,
        seed_offset: int,
    ) -> None:
        for (model, parent), group in frame.groupby(
            ["model_label", "parent_candidate"], sort=True
        ):
            mean, low, high = _bootstrap_seed_mean(
                group,
                "candidate_minus_matched",
                repetitions=config.mapping_null_repetitions,
                seed=443401 + int(seed_offset),
            )
            per_seed = (
                group.groupby("seed")["candidate_minus_matched"]
                .mean()
                .to_numpy(float)
            )
            rows.append(
                {
                    "model_label": model,
                    "parent_candidate": parent,
                    "contrast_name": contrast_name,
                    "support_direction": alternative or "zero_is_expected",
                    "mean_candidate_minus_matched": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "one_sided_sign_flip_p": (
                        _exact_sign_flip_p(per_seed, alternative=alternative)
                        if alternative is not None
                        else math.nan
                    ),
                    "max_abs_unit_difference": float(
                        np.max(np.abs(group["candidate_minus_matched"].to_numpy(float)))
                    ),
                    "seeds": int(group["seed"].nunique()),
                }
            )

    patch_index = [
        "model_label",
        "seed",
        "receiver_count",
        "donor_count",
        "parent_candidate",
        "head_role",
    ]
    for left, right, name, offset in (
        ("z_donor", "output_norm_control", "z_transport", 1),
        ("alpha_receiver_v", "alpha_position_scramble", "alpha_localization", 2),
    ):
        selected = patch[patch["intervention"].isin([left, right])]
        within = selected.pivot(
            index=patch_index,
            columns="intervention",
            values="continuous_normalized_transport",
        ).reset_index()
        within["within_role"] = within[left] - within[right]
        unit_index = [column for column in patch_index if column != "head_role"]
        paired = within.pivot(
            index=unit_index, columns="head_role", values="within_role"
        ).reset_index()
        paired = paired.dropna(subset=["candidate", "matched_control"])
        paired["candidate_minus_matched"] = (
            paired["candidate"] - paired["matched_control"]
        )
        append_summary(
            paired,
            contrast_name=name,
            alternative="greater",
            seed_offset=offset,
        )

    directed_index = [
        "model_label",
        "seed",
        "gold_count",
        "parent_candidate",
        "head_role",
    ]
    removal_rows = directed[
        directed["intervention"].isin(
            ["answer_direction_removal", "equal_norm_orthogonal_removal"]
        )
    ]
    for metric, name, alternative, offset in (
        (
            "delta_expected_count_absolute_error",
            "removal_absolute_error",
            "greater",
            3,
        ),
        ("delta_correct_margin", "removal_correct_margin", "less", 4),
    ):
        within = removal_rows.pivot(
            index=directed_index,
            columns="intervention",
            values=metric,
        ).reset_index()
        within["within_role"] = (
            within["answer_direction_removal"]
            - within["equal_norm_orthogonal_removal"]
        )
        unit_index = [column for column in directed_index if column != "head_role"]
        paired = within.pivot(
            index=unit_index, columns="head_role", values="within_role"
        ).reset_index()
        paired = paired.dropna(subset=["candidate", "matched_control"])
        paired["candidate_minus_matched"] = (
            paired["candidate"] - paired["matched_control"]
        )
        append_summary(
            paired,
            contrast_name=name,
            alternative=alternative,
            seed_offset=offset,
        )

    injection = directed[
        directed["intervention"].astype(str).str.startswith(
            "signed_answer_direction_injection_beta_"
        )
    ]
    injection_index = [
        "model_label",
        "seed",
        "gold_count",
        "beta",
        "parent_candidate",
    ]
    paired_injection = injection.pivot(
        index=injection_index,
        columns="head_role",
        values="delta_expected_count",
    ).reset_index()
    paired_injection = paired_injection.dropna(
        subset=["candidate", "matched_control"]
    )
    paired_injection["candidate_minus_matched"] = (
        paired_injection["candidate"] - paired_injection["matched_control"]
    )
    append_summary(
        paired_injection,
        contrast_name="layer_injection_nonlocalization_audit",
        alternative=None,
        seed_offset=5,
    )
    return pd.DataFrame(rows)


def _candidate_decisions(
    mapping: pd.DataFrame,
    patch_contrast: pd.DataFrame,
    injection: pd.DataFrame,
    removal: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    candidates = injection[injection["head_role"].eq("candidate")]
    for item in candidates.itertuples():
        mask = (
            mapping["layer"].astype(int).eq(int(item.layer))
            & mapping["head"].astype(int).eq(int(item.head))
        )
        map_row = mapping[mask].iloc[0]
        contrasts = patch_contrast[
            patch_contrast["parent_candidate"].astype(str).eq(item.parent_candidate)
            & patch_contrast["head_role"].eq("candidate")
        ]
        z = contrasts[
            contrasts["contrast_name"].eq("z_minus_norm_control_transport")
        ]
        alpha = contrasts[
            contrasts["contrast_name"].eq("alpha_minus_scramble_transport")
        ]
        removal_row = removal[
            removal["parent_candidate"].astype(str).eq(item.parent_candidate)
            & removal["head_role"].eq("candidate")
        ].iloc[0]
        control_injection = injection[
            injection["parent_candidate"].astype(str).eq(item.parent_candidate)
            & injection["head_role"].eq("matched_control")
        ]
        control_removal = removal[
            removal["parent_candidate"].astype(str).eq(item.parent_candidate)
            & removal["head_role"].eq("matched_control")
        ]
        control_contrasts = patch_contrast[
            patch_contrast["parent_candidate"].astype(str).eq(item.parent_candidate)
            & patch_contrast["head_role"].eq("matched_control")
        ]
        mapping_pass = bool(
            float(map_row["fit_mapping_cosine"]) > 0
            and float(map_row["heldout_count_mapping_cosine"]) > 0
        )
        z_pass = bool(not z.empty and float(z.iloc[0]["mean_contrast"]) > 0)
        alpha_pass = bool(
            not alpha.empty and float(alpha.iloc[0]["mean_contrast"]) > 0
        )
        injection_pass = bool(
            float(item.injection_slope_expected_count_per_beta) > 0
            and float(item.beta_outcome_spearman) > 0.5
        )
        removal_error = float(
            removal_row[
                "mean_delta_expected_count_absolute_error_answer_minus_orthogonal"
            ]
        )
        removal_margin = float(
            removal_row["mean_delta_correct_margin_answer_minus_orthogonal"]
        )
        removal_pass = bool(removal_error > 0 and removal_margin < 0)
        candidate_exceeds_matched_z = False
        candidate_exceeds_matched_injection = False
        candidate_exceeds_matched_removal_error = False
        candidate_exceeds_matched_removal_margin = False
        if not control_injection.empty and not control_removal.empty:
            control_z = control_contrasts[
                control_contrasts["contrast_name"].eq(
                    "z_minus_norm_control_transport"
                )
            ]
            if not z.empty and not control_z.empty:
                candidate_exceeds_matched_z = bool(
                    float(z.iloc[0]["mean_contrast"])
                    > float(control_z.iloc[0]["mean_contrast"])
                )
            candidate_exceeds_matched_injection = bool(
                float(item.injection_slope_expected_count_per_beta)
                > float(
                    control_injection.iloc[0][
                        "injection_slope_expected_count_per_beta"
                    ]
                )
            )
            candidate_exceeds_matched_removal_error = bool(
                removal_error
                > float(
                    control_removal.iloc[0][
                        "mean_delta_expected_count_absolute_error_answer_minus_orthogonal"
                    ]
                )
            )
            candidate_exceeds_matched_removal_margin = bool(
                removal_margin
                < float(
                    control_removal.iloc[0][
                        "mean_delta_correct_margin_answer_minus_orthogonal"
                    ]
                )
            )
        rows.append(
            {
                "model_label": item.model_label,
                "parent_candidate": item.parent_candidate,
                "layer": int(item.layer),
                "head": int(item.head),
                "mapping_pass": mapping_pass,
                "z_transport_vs_norm_control_pass": z_pass,
                "alpha_vs_scramble_pass": alpha_pass,
                "signed_injection_pass": injection_pass,
                "direction_removal_vs_orthogonal_pass": removal_pass,
                "joint_ov_head_support": bool(
                    mapping_pass and z_pass and injection_pass and removal_pass
                ),
                "alpha_localization_support": alpha_pass,
                "candidate_exceeds_matched_on_z_transport": candidate_exceeds_matched_z,
                "candidate_exceeds_matched_on_injection_slope": candidate_exceeds_matched_injection,
                "candidate_exceeds_matched_on_removal_error": candidate_exceeds_matched_removal_error,
                "candidate_exceeds_matched_on_removal_margin": candidate_exceeds_matched_removal_margin,
                "matched_head_specificity_support": bool(
                    candidate_exceeds_matched_z
                    and candidate_exceeds_matched_removal_error
                    and candidate_exceeds_matched_removal_margin
                ),
            }
        )
    return rows


def analyze_model(
    run_root: str | Path,
    *,
    model_label: str,
    config: V443Config,
) -> dict[str, Any]:
    model_root = Path(run_root) / "models" / model_label
    mapping = pd.read_csv(model_root / "mapping" / "head_mapping_scores.csv.gz")
    patch = pd.read_csv(model_root / "staged_patch" / "detail.csv.gz")
    directed = pd.read_csv(model_root / "directed" / "detail.csv.gz")
    output = model_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    patch_summary = summarize_patch(patch, config)
    contrasts = patch_contrasts(patch, config)
    injection = summarize_injection(directed, config)
    removal = summarize_removal(directed, config)
    specificity = summarize_head_specificity(patch, directed, config)
    decisions = _candidate_decisions(
        mapping, contrasts, injection, removal
    )
    for decision in decisions:
        head_rows = specificity[
            specificity["parent_candidate"].astype(str).eq(
                decision["parent_candidate"]
            )
        ]
        by_name = {
            str(row.contrast_name): row for row in head_rows.itertuples()
        }
        z_row = by_name["z_transport"]
        error_row = by_name["removal_absolute_error"]
        margin_row = by_name["removal_correct_margin"]
        alpha_row = by_name["alpha_localization"]
        injection_row = by_name["layer_injection_nonlocalization_audit"]
        decision["head_specific_directional_support"] = bool(
            float(z_row.mean_candidate_minus_matched) > 0
            and float(error_row.mean_candidate_minus_matched) > 0
            and float(margin_row.mean_candidate_minus_matched) < 0
        )
        decision["head_specific_exact_p_le_0_05"] = bool(
            float(z_row.one_sided_sign_flip_p) <= 0.05
            and float(error_row.one_sided_sign_flip_p) <= 0.05
            and float(margin_row.one_sided_sign_flip_p) <= 0.05
        )
        decision["alpha_selected_minus_matched_exact_p"] = float(
            alpha_row.one_sided_sign_flip_p
        )
        decision["layer_injection_selected_matched_max_abs_difference"] = float(
            injection_row.max_abs_unit_difference
        )
    smoke = json.loads(
        (model_root / "smoke" / "complete.json").read_text(encoding="utf-8")
    )
    if "attention_cache_candidate_logit_max_abs_delta" in patch.columns:
        cache_raw_max = float(
            patch["attention_cache_candidate_logit_max_abs_delta"].max()
        )
        cache_centered_max = float(
            patch[
                "attention_cache_candidate_centered_logit_max_abs_delta"
            ].max()
        )
        cache_scope = "formal staged-patch prompts"
    else:
        cache_raw_max = float(
            smoke.get("attention_cache_candidate_logit_max_abs_delta", math.nan)
        )
        cache_centered_max = float(
            smoke.get(
                "attention_cache_candidate_centered_logit_max_abs_delta",
                math.nan,
            )
        )
        cache_scope = "actual-model smoke (formal shards predate centered audit column)"
    if "value_path_estimand" in mapping.columns:
        mapping_estimands = (
            mapping[["layer", "value_source_layer", "value_path_estimand"]]
            .drop_duplicates()
            .sort_values("layer")
            .to_dict(orient="records")
        )
    else:
        mapping_estimands = [
            {
                "layer": int(layer),
                "value_source_layer": int(layer),
                "value_path_estimand": "linear_w_o_w_v_prompt_direction",
            }
            for layer in sorted(mapping["layer"].astype(int).unique())
        ]
    atomic_csv_gzip(patch_summary, output / "patch_summary.csv.gz")
    atomic_csv_gzip(contrasts, output / "patch_contrasts.csv.gz")
    atomic_csv_gzip(injection, output / "injection_summary.csv.gz")
    atomic_csv_gzip(removal, output / "removal_summary.csv.gz")
    atomic_csv_gzip(specificity, output / "head_specificity_contrasts.csv.gz")
    payload = {
        "schema_version": "realistic_niah_v4_4_3_model_analysis_v2",
        "model_label": model_label,
        "evidence_scope": {
            "mapping": "discovery seeds; fit-count ranking and held-out-count check",
            "staged_patch": "five screen seeds",
            "directed": "five disjoint confirmation seeds",
        },
        "candidate_decisions": decisions,
        "any_single_head_joint_support": any(
            item["joint_ov_head_support"] for item in decisions
        ),
        "z_o_equivalence_max_candidate_logit_delta": float(
            patch["z_o_candidate_logit_max_abs_delta"].max()
        ),
        "mapping_estimands": mapping_estimands,
        "attention_cache_audit": {
            "scope": cache_scope,
            "raw_candidate_logit_max_abs_delta": cache_raw_max,
            "centered_candidate_logit_max_abs_delta": (
                cache_centered_max if math.isfinite(cache_centered_max) else None
            ),
            "centered_tolerance": float(config.attention_cache_logit_tolerance),
        },
        "head_specificity_contrasts": _json_safe_records(specificity),
        "signed_injection_scope": (
            "layer-level post-O answer-direction steering; identical for the "
            "selected and matched head within a layer, so it is not a head-localizing test"
        ),
        "local_selectivity_boundary": (
            "non-count next-token KL is a local specificity control, not a full "
            "generic-language benchmark"
        ),
    }
    atomic_json(output / "analysis.json", payload)
    return payload


def analyze_campaign(
    run_root: str | Path,
    *,
    config: V443Config,
) -> dict[str, Any]:
    analyses = {
        model: analyze_model(run_root, model_label=model, config=config)
        for model in config.model_labels
    }
    supported = {
        model: bool(payload["any_single_head_joint_support"])
        for model, payload in analyses.items()
    }
    statistically_specific = {
        model: any(
            bool(item["joint_ov_head_support"])
            and bool(item["head_specific_exact_p_le_0_05"])
            for item in payload["candidate_decisions"]
        )
        for model, payload in analyses.items()
    }
    if any(statistically_specific.values()):
        conclusion = (
            "至少一个单头同时通过冻结方向规则、matched-head 特异性与全部 head-specific exact p≤0.05。"
        )
    elif any(supported.values()):
        conclusion = (
            "冻结方向规则出现联合通过候选，但没有单头同时通过全部 matched-head 特异性 exact p≤0.05；"
            "结果是部分/提示性支持，尚不能确认单头 OV circuit。"
        )
    else:
        conclusion = (
            "当前候选中没有单头同时通过冻结方向规则；应优先考虑分布式 multi-head/MLP/跨层重编码。"
        )
    lines = [
        "# Realistic NIAH V4.4.3：OV vertical geometry 因果检验",
        "",
        f"**结论：** {conclusion}",
        "",
        "## 设计边界",
        "",
        "- 输入仅来自冻结的 numeric V4.4；原始 V4.4 filestream 只读。",
        "- head 排序只使用 discovery seeds 与 fit counts；even held-out counts 不参与排序。",
        "- staged patch 使用 5 个 screen seeds；定向 removal/injection 使用另外 5 个 confirmation seeds。",
        "- `z_h` 与 `o_h=W_O z_h` patch 在同一线性头内理论上等价；报告把它们的数值一致性作为实现审计，而不是两份独立因果证据。",
        "- signed injection 在 layer post-O residual 上直接加入 answer direction；同层 selected/matched 条件相同，因此它只证明层级可控性，不定位 head。",
        "- non-count next-token KL 只控制 answer-query 局部分布，不等同于完整通用语言能力评测。",
        "",
        "## 模型结果",
        "",
    ]
    for model, payload in analyses.items():
        lines.extend([f"### {model}", ""])
        for item in payload["candidate_decisions"]:
            lines.append(
                "- {head}: mapping={mapping}, z-control={z}, alpha-scramble={alpha}, "
                "injection(layer-level)={injection}, removal={removal}, joint={joint}, "
                "selected>matched(head-specific directional)={specificity}, "
                "all head-specific exact p<=.05={specific_p}.".format(
                    head=item["parent_candidate"],
                    mapping=item["mapping_pass"],
                    z=item["z_transport_vs_norm_control_pass"],
                    alpha=item["alpha_vs_scramble_pass"],
                    injection=item["signed_injection_pass"],
                    removal=item["direction_removal_vs_orthogonal_pass"],
                    joint=item["joint_ov_head_support"],
                    specificity=item["matched_head_specificity_support"],
                    specific_p=item["head_specific_exact_p_le_0_05"],
                )
            )
        lines.extend(
            [
                "",
                f"Z/O 数值等价审计最大 count-logit 差：{payload['z_o_equivalence_max_candidate_logit_delta']:.6g}",
                "",
                "Value-path estimand: "
                + "; ".join(
                    "L{layer} uses V source L{source} ({kind})".format(
                        layer=int(row["layer"]),
                        source=int(row["value_source_layer"]),
                        kind=row["value_path_estimand"],
                    )
                    for row in payload["mapping_estimands"]
                ),
                "",
                "Attention-cache audit: raw max={raw:.6g}; centered max={centered}; "
                "centered tolerance={tolerance:.6g}; scope={scope}.".format(
                    raw=payload["attention_cache_audit"][
                        "raw_candidate_logit_max_abs_delta"
                    ],
                    centered=(
                        "not recorded"
                        if payload["attention_cache_audit"][
                            "centered_candidate_logit_max_abs_delta"
                        ]
                        is None
                        else f"{payload['attention_cache_audit']['centered_candidate_logit_max_abs_delta']:.6g}"
                    ),
                    tolerance=payload["attention_cache_audit"][
                        "centered_tolerance"
                    ],
                    scope=payload["attention_cache_audit"]["scope"],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 判定规则",
            "",
            "单头 OV 支持要求：fit/held-out-count 映射方向同为正；donor-Z transport 优于等范数输出控制；signed injection 对 expected count 呈正斜率且 Spearman > 0.5；answer-direction removal 比等范数正交 removal 更损害 count margin/误差。Alpha-vs-scramble 单独作为 QK 定位证据，不计入 OV 必要条件。",
            "",
            "Matched-head specificity 是额外二级审计，不改写冻结的联合方向规则；若 selected head 未超过同层 matched head，则不能把通过方向规则写成该 head 的特异定位。",
            "",
            "如果严格单头特异性证据不足，本实验不能据此断言 OV 不参与；它只说明当前预注册层与单头候选尚不足以解释该几何，下一步应检验小头集合、MLP 与跨层路径。",
            "",
        ]
    )
    analysis_root = Path(run_root) / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    report_path = analysis_root / "realistic_niah_v4_4_3_ov_causal_report.md"
    atomic_text(report_path, "\n".join(lines))
    payload = {
        "schema_version": "realistic_niah_v4_4_3_campaign_analysis_v2",
        "model_support": supported,
        "statistically_specific_model_support": statistically_specific,
        "conclusion": conclusion,
        "model_analyses": analyses,
        "report_path": str(report_path),
    }
    atomic_json(analysis_root / "analysis.json", payload)
    return payload


def audit_campaign(
    run_root: str | Path,
    *,
    config: V443Config,
) -> dict[str, Any]:
    root = Path(run_root)
    model_results = {}
    complete = True
    for model in config.model_labels:
        model_root = root / "models" / model
        selection_path = model_root / "mapping" / "head_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        candidate_count = len(selection["candidate_heads"])
        entry_count = 2 * candidate_count
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
        model_results[model] = {
            "candidate_count": candidate_count,
            "expected_patch_shards": expected_patch_shards,
            "actual_patch_shards": len(patch_shards),
            "patch_rows": patch_rows,
            "expected_directed_shards": expected_directed_shards,
            "actual_directed_shards": len(directed_shards),
            "directed_rows": directed_rows,
            "complete": model_complete,
        }
    payload = {
        "schema_version": "realistic_niah_v4_4_3_audit_v1",
        "complete": complete,
        "models": model_results,
        "required_campaign_files": {
            "resolved_config": (root / "resolved_config.json").is_file(),
            "owner": (root / "owner.json").is_file(),
            "input_manifest": (root / "input_manifest.json").is_file(),
            "report": (
                root / "analysis" / "realistic_niah_v4_4_3_ov_causal_report.md"
            ).is_file(),
        },
    }
    payload["complete"] = bool(
        payload["complete"] and all(payload["required_campaign_files"].values())
    )
    atomic_json(root / "audit.json", payload)
    if payload["complete"]:
        atomic_json(
            root / "complete.json",
            {
                "schema_version": "realistic_niah_v4_4_3_complete_v1",
                "state": "COMPLETE",
                "audit": "audit.json",
                "report": "analysis/realistic_niah_v4_4_3_ov_causal_report.md",
            },
        )
    return payload
