from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from realistic_niah_v4_4_3.io import atomic_csv_gzip, atomic_json, atomic_text, stage_root
from realistic_niah_v4_4_4.analysis import (
    _bootstrap_ci,
    _holm_adjust,
    exact_sign_flip_p,
)

from .readwrite_spec import V444ReadWriteConfig


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _append_metric(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    family: str,
    metric: str,
    value: float,
    alternative: str = "greater",
    layer: int | None = None,
    stratum: str = "all",
) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"Non-finite read/write seed metric: {metric}")
    rows.append(
        {
            "seed": int(seed),
            "family": family,
            "metric": metric,
            "layer": None if layer is None else int(layer),
            "stratum": str(stratum),
            "value": float(value),
            "alternative": alternative,
        }
    )


def build_seed_metrics(
    natural: pd.DataFrame,
    read: pd.DataFrame,
    read_trace: pd.DataFrame,
    write: pd.DataFrame,
    *,
    config: V444ReadWriteConfig,
) -> pd.DataFrame:
    _require_columns(
        read,
        (
            "seed",
            "receiver_count",
            "donor_count",
            "component",
            "intervention",
            "continuous_normalized_transport",
            "component_mechanical_transport",
            "baseline_predicted_count",
        ),
        "V4.4.4 read/write detail",
    )
    _require_columns(
        read_trace,
        (
            "seed",
            "receiver_count",
            "donor_count",
            "component",
            "layer",
            "downstream_count_axis_coefficient",
        ),
        "V4.4.4 read/write trace detail",
    )
    _require_columns(
        write,
        (
            "seed",
            "gold_count",
            "intervention",
            "layer",
            "signed_beta",
            "delta_expected_count",
            "downstream_count_axis_coefficient",
        ),
        "V4.4.4 read/write intervention detail",
    )
    _require_columns(
        natural,
        ("seed", "gold_count", "layer", "baseline_correct"),
        "V4.4.4 read/write natural detail",
    )
    rows: list[dict[str, Any]] = []
    patch = read[read["intervention"].eq("component_patch")].copy()
    patch["baseline_stratum"] = np.where(
        patch["baseline_predicted_count"].astype(int)
        == patch["receiver_count"].astype(int),
        "baseline_correct",
        "baseline_wrong",
    )
    for (seed, component), group in patch.groupby(["seed", "component"], sort=True):
        _append_metric(
            rows,
            seed=int(seed),
            family="read_transport",
            metric=f"read_{component}_behavior_transport",
            value=float(group["continuous_normalized_transport"].mean()),
        )
        _append_metric(
            rows,
            seed=int(seed),
            family="read_transport",
            metric=f"read_{component}_mechanical_transport",
            value=float(group["component_mechanical_transport"].mean()),
        )
    for (seed, component, stratum), group in patch.groupby(
        ["seed", "component", "baseline_stratum"], sort=True
    ):
        _append_metric(
            rows,
            seed=int(seed),
            family="read_outcome_sensitivity",
            metric=f"read_{component}_behavior_transport",
            value=float(group["continuous_normalized_transport"].mean()),
            stratum=str(stratum),
        )
    patch_seed = (
        patch.groupby(["seed", "component"], sort=True)[
            "continuous_normalized_transport"
        ]
        .mean()
        .unstack("component")
    )
    if {"value", "routing"}.issubset(patch_seed.columns):
        for seed, row in patch_seed.iterrows():
            _append_metric(
                rows,
                seed=int(seed),
                family="read_mode_contrast",
                metric="read_value_minus_routing_transport",
                value=float(row["value"] - row["routing"]),
            )
            _append_metric(
                rows,
                seed=int(seed),
                family="read_mode_contrast",
                metric="read_routing_minus_value_transport",
                value=float(row["routing"] - row["value"]),
            )
    for component in ("value", "routing"):
        subset = read[read["component"].eq(component)].copy()
        pivot = subset.pivot_table(
            index=["seed", "receiver_count", "donor_count"],
            columns="intervention",
            values="continuous_normalized_transport",
            aggfunc="first",
        )
        required = {
            "component_patch_plus_natural_axis_block",
            "component_patch_plus_orthogonal_control",
        }
        if not required.issubset(pivot.columns):
            raise RuntimeError(f"Read/write {component} mediation grid is incomplete")
        contrast = (
            pivot["component_patch_plus_orthogonal_control"]
            - pivot["component_patch_plus_natural_axis_block"]
        )
        for seed, values in contrast.groupby(level="seed", sort=True):
            _append_metric(
                rows,
                seed=int(seed),
                family="read_mediation",
                metric=f"read_{component}_ov_mediation_specificity",
                value=float(values.mean()),
            )
    traced = read_trace.copy()
    traced["normalized_downstream_transport"] = (
        traced["downstream_count_axis_coefficient"]
        / (traced["donor_count"] - traced["receiver_count"])
    )
    for (seed, component, layer), group in traced.groupby(
        ["seed", "component", "layer"], sort=True
    ):
        _append_metric(
            rows,
            seed=int(seed),
            family="read_downstream_trace",
            metric=f"read_{component}_downstream_transport",
            value=float(group["normalized_downstream_transport"].mean()),
            layer=int(layer),
        )

    behavior = write.drop_duplicates(
        ["seed", "gold_count", "intervention"]
    ).copy()
    behavior_pivot = behavior.pivot_table(
        index=["seed", "gold_count"],
        columns="intervention",
        values="delta_expected_count",
        aggfunc="first",
    )
    expected_interventions = {
        "natural_plus",
        "natural_minus",
        "orthogonal_plus",
        "orthogonal_minus",
    }
    if not expected_interventions.issubset(behavior_pivot.columns):
        raise RuntimeError("Read/write behavior grid is incomplete")
    natural_behavior = (
        behavior_pivot["natural_plus"] - behavior_pivot["natural_minus"]
    ) / (2.0 * config.write_beta)
    control_behavior = (
        behavior_pivot["orthogonal_plus"] - behavior_pivot["orthogonal_minus"]
    ) / (2.0 * config.write_beta)
    for seed in sorted(set(behavior_pivot.index.get_level_values("seed"))):
        natural_seed = float(natural_behavior.xs(seed, level="seed").mean())
        control_seed = float(control_behavior.xs(seed, level="seed").mean())
        _append_metric(
            rows,
            seed=int(seed),
            family="write_behavior",
            metric="write_natural_behavior_slope",
            value=natural_seed,
        )
        _append_metric(
            rows,
            seed=int(seed),
            family="write_behavior",
            metric="write_orthogonal_behavior_slope",
            value=control_seed,
        )
        _append_metric(
            rows,
            seed=int(seed),
            family="write_behavior",
            metric="write_behavior_specificity",
            value=natural_seed - control_seed,
        )

    trace_pivot = write.pivot_table(
        index=["seed", "gold_count", "layer"],
        columns="intervention",
        values="downstream_count_axis_coefficient",
        aggfunc="first",
    )
    if not expected_interventions.issubset(trace_pivot.columns):
        raise RuntimeError("Read/write residual grid is incomplete")
    natural_trace = (
        trace_pivot["natural_plus"] - trace_pivot["natural_minus"]
    ) / (2.0 * config.write_beta)
    control_trace = (
        trace_pivot["orthogonal_plus"] - trace_pivot["orthogonal_minus"]
    ) / (2.0 * config.write_beta)
    for seed in sorted(set(trace_pivot.index.get_level_values("seed"))):
        for layer in config.downstream_layers:
            natural_seed = float(
                natural_trace.xs(
                    (seed, layer), level=("seed", "layer")
                ).mean()
            )
            control_seed = float(
                control_trace.xs(
                    (seed, layer), level=("seed", "layer")
                ).mean()
            )
            _append_metric(
                rows,
                seed=int(seed),
                family="write_residual_trace",
                metric="write_natural_residual_slope",
                value=natural_seed,
                layer=int(layer),
            )
            _append_metric(
                rows,
                seed=int(seed),
                family="write_residual_trace",
                metric="write_orthogonal_residual_slope",
                value=control_seed,
                layer=int(layer),
            )
            _append_metric(
                rows,
                seed=int(seed),
                family="write_residual_trace",
                metric="write_residual_specificity",
                value=natural_seed - control_seed,
                layer=int(layer),
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("Read/write supplement produced no seed metrics")
    return result.sort_values(
        ["family", "metric", "stratum", "layer", "seed"],
        na_position="first",
    ).reset_index(drop=True)


def summarize_seed_metrics(
    seed_metrics: pd.DataFrame, *, config: V444ReadWriteConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = seed_metrics.groupby(
        ["family", "metric", "stratum", "layer", "alternative"],
        dropna=False,
        sort=True,
    )
    for offset, (key, group) in enumerate(grouped, start=1):
        family, metric, stratum, layer, alternative = key
        values = group["value"].to_numpy(float)
        mean, low, high = _bootstrap_ci(
            values,
            repetitions=config.bootstrap_repetitions,
            seed=445_000 + offset,
        )
        rows.append(
            {
                "family": str(family),
                "metric": str(metric),
                "stratum": str(stratum),
                "layer": None if pd.isna(layer) else int(layer),
                "alternative": str(alternative),
                "seed_count": len(values),
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "exact_sign_flip_p": exact_sign_flip_p(
                    values, alternative=str(alternative)
                ),
                "positive_seed_fraction": float(np.mean(values > 0)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["holm_p_within_family_metric"] = summary["exact_sign_flip_p"]
    write_mask = (
        summary["metric"].eq("write_residual_specificity")
        & summary["stratum"].eq("all")
    )
    if write_mask.any():
        summary.loc[write_mask, "holm_p_within_family_metric"] = _holm_adjust(
            summary.loc[write_mask, "exact_sign_flip_p"].tolist()
        )
    return summary.sort_values(
        ["family", "metric", "stratum", "layer"], na_position="first"
    ).reset_index(drop=True)


def _row(
    summary: pd.DataFrame,
    metric: str,
    *,
    layer: int | None = None,
    stratum: str = "all",
) -> pd.Series:
    selected = summary[
        summary["metric"].eq(metric) & summary["stratum"].eq(stratum)
    ]
    selected = selected[
        selected["layer"].isna()
        if layer is None
        else selected["layer"].fillna(-1).astype(int).eq(int(layer))
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one read/write summary row for {metric}/L{layer}/{stratum}"
        )
    return selected.iloc[0]


def primary_decision(
    summary: pd.DataFrame, *, config: V444ReadWriteConfig
) -> dict[str, Any]:
    alpha = float(config.primary_alpha)
    full = _row(summary, "read_full_behavior_transport")
    value_behavior = _row(summary, "read_value_behavior_transport")
    value_mediation = _row(summary, "read_value_ov_mediation_specificity")
    route_behavior = _row(summary, "read_routing_behavior_transport")
    route_mediation = _row(summary, "read_routing_ov_mediation_specificity")
    value_dominance = _row(summary, "read_value_minus_routing_transport")
    route_dominance = _row(summary, "read_routing_minus_value_transport")
    value_family_p = max(
        float(value_behavior["exact_sign_flip_p"]),
        float(value_mediation["exact_sign_flip_p"]),
    )
    route_family_p = max(
        float(route_behavior["exact_sign_flip_p"]),
        float(route_mediation["exact_sign_flip_p"]),
    )
    full_pass = float(full["exact_sign_flip_p"]) <= alpha and float(full["mean"]) > 0
    value_pass = value_family_p <= alpha
    route_pass = route_family_p <= alpha
    value_dom = float(value_dominance["exact_sign_flip_p"]) <= alpha
    route_dom = float(route_dominance["exact_sign_flip_p"]) <= alpha
    if not full_pass:
        classification = "no_transport"
    elif value_pass and value_dom and not route_dom:
        classification = "value_dominant"
    elif route_pass and route_dom and not value_dom:
        classification = "routing_dominant"
    elif value_pass and route_pass:
        classification = "mixed"
    else:
        classification = "unresolved"

    write_behavior = _row(summary, "write_behavior_specificity")
    final_layer = int(config.downstream_layers[-1])
    write_final = _row(
        summary, "write_residual_specificity", layer=final_layer
    )
    write_supported = bool(
        float(write_behavior["exact_sign_flip_p"]) <= alpha
        and float(write_behavior["mean"]) > 0
        and float(write_final["holm_p_within_family_metric"]) <= alpha
        and float(write_final["mean"]) > 0
    )
    return {
        "alpha": alpha,
        "read_mode": {
            "classification": classification,
            "full_patch_p": float(full["exact_sign_flip_p"]),
            "full_patch_mean": float(full["mean"]),
            "value_family_p": value_family_p,
            "routing_family_p": route_family_p,
            "value_minus_routing_p": float(
                value_dominance["exact_sign_flip_p"]
            ),
            "routing_minus_value_p": float(
                route_dominance["exact_sign_flip_p"]
            ),
        },
        "write_propagation": {
            "supported": write_supported,
            "behavior_specificity_mean": float(write_behavior["mean"]),
            "behavior_specificity_p": float(
                write_behavior["exact_sign_flip_p"]
            ),
            "final_layer": final_layer,
            "final_residual_specificity_mean": float(write_final["mean"]),
            "final_residual_specificity_holm_p": float(
                write_final["holm_p_within_family_metric"]
            ),
        },
        "serial_read_write_supported": bool(
            classification in {"value_dominant", "routing_dominant", "mixed"}
            and write_supported
        ),
        "interpretation": (
            "Mechanistic extension on parent-campaign evaluation seeds; "
            "requires a new-seed replication for an independent confirmation."
        ),
    }


def audit_campaign(
    run_root: str | Path, *, config: V444ReadWriteConfig
) -> dict[str, Any]:
    root = Path(run_root)
    discovery = stage_root(root, config.model_label, "read_write_discovery")
    evaluation = stage_root(root, config.model_label, "read_write_evaluation")
    required = [
        root / "v4_4_4_read_write_base_snapshot.json",
        discovery / "complete.json",
        discovery / "artifacts.pt",
        discovery / "natural_partition_detail.csv.gz",
        discovery / "downstream_axis_detail.csv.gz",
        stage_root(root, config.model_label, "read_write_smoke") / "complete.json",
        evaluation / "design.json",
        evaluation / "complete.json",
        evaluation / "natural_detail.csv.gz",
        evaluation / "read_mechanical_detail.csv.gz",
        evaluation / "read_causal_detail.csv.gz",
        evaluation / "read_trace_detail.csv.gz",
        evaluation / "write_trace_detail.csv.gz",
    ]
    checks: list[dict[str, Any]] = []
    missing = [str(path) for path in required if not path.is_file()]
    checks.append({"name": "required_artifacts", "passed": not missing, "detail": missing})
    if missing:
        return {
            "all_checks_pass": False,
            "check_count": len(checks),
            "checks": checks,
        }
    snapshot_path = root / "v4_4_4_read_write_base_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_errors: list[str] = []
    if snapshot.get("schema_version") != (
        "realistic_niah_v4_4_4_read_write_base_snapshot_v1"
    ):
        snapshot_errors.append("unexpected snapshot schema")
    for relative, registered in snapshot.get("files", {}).items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            snapshot_errors.append(f"unsafe protected path: {relative}")
            continue
        protected = root / relative_path
        if not protected.is_file():
            snapshot_errors.append(f"missing protected file: {relative}")
            continue
        observed_bytes = protected.stat().st_size
        observed_hash = hashlib.sha256(protected.read_bytes()).hexdigest()
        if observed_bytes != int(registered["bytes"]):
            snapshot_errors.append(f"size changed: {relative}")
        if observed_hash != str(registered["sha256"]):
            snapshot_errors.append(f"sha256 changed: {relative}")
    checks.append(
        {
            "name": "frozen_v4_4_4_base_unchanged",
            "passed": not snapshot_errors,
            "detail": snapshot_errors,
        }
    )
    natural = pd.read_csv(evaluation / "natural_detail.csv.gz")
    evaluation_design = json.loads(
        (evaluation / "design.json").read_text(encoding="utf-8")
    )
    cache_reference_tolerance = float(
        evaluation_design.get("attention_cache_policy", {}).get(
            "reference_tolerance", math.nan
        )
    )
    mechanical = pd.read_csv(evaluation / "read_mechanical_detail.csv.gz")
    read = pd.read_csv(evaluation / "read_causal_detail.csv.gz")
    read_trace = pd.read_csv(evaluation / "read_trace_detail.csv.gz")
    write = pd.read_csv(evaluation / "write_trace_detail.csv.gz")
    expected = {
        "natural_rows": len(config.evaluation_seeds)
        * len(config.counts)
        * len(config.downstream_layers),
        "mechanical_rows": len(config.evaluation_seeds)
        * len(config.donor_pairs)
        * 5,
        "read_rows": len(config.evaluation_seeds) * len(config.donor_pairs) * 7,
        "read_trace_rows": len(config.evaluation_seeds)
        * len(config.donor_pairs)
        * 3
        * len(config.downstream_layers),
        "write_rows": len(config.evaluation_seeds)
        * len(config.write_counts)
        * 4
        * len(config.downstream_layers),
    }
    observed = {
        "natural_rows": len(natural),
        "mechanical_rows": len(mechanical),
        "read_rows": len(read),
        "read_trace_rows": len(read_trace),
        "write_rows": len(write),
    }
    checks.append(
        {
            "name": "merged_row_counts",
            "passed": observed == expected,
            "detail": {"expected": expected, "observed": observed},
        }
    )
    closure_max = float(mechanical["closure_relative_l2"].max())
    checks.append(
        {
            "name": "shapley_closure",
            "passed": closure_max <= config.closure_relative_tolerance,
            "detail": closure_max,
        }
    )
    anchored = mechanical[mechanical["position_group"].eq("all_positions")]
    endpoint_columns = {
        "receiver_endpoint_reconstruction_relative_l2",
        "donor_endpoint_reconstruction_relative_l2",
    }
    endpoint_columns_present = endpoint_columns.issubset(anchored.columns)
    endpoint_max = (
        float(anchored[list(endpoint_columns)].max().max())
        if endpoint_columns_present and not anchored.empty
        else math.inf
    )
    checks.append(
        {
            "name": "eager_endpoint_reconstruction",
            "passed": (
                endpoint_columns_present
                and math.isfinite(endpoint_max)
                and endpoint_max <= config.edge_reconstruction_relative_tolerance
            ),
            "detail": endpoint_max,
        }
    )
    cache_columns = (
        "attention_cache_candidate_logit_max_abs_delta",
        "attention_cache_candidate_centered_logit_max_abs_delta",
        "attention_cache_reference_tolerance_exceeded",
    )
    missing_cache_columns = [name for name in cache_columns if name not in natural]
    checks.append(
        {
            "name": "attention_cache_diagnostics_recorded",
            "passed": not missing_cache_columns,
            "detail": missing_cache_columns,
        }
    )
    if not missing_cache_columns:
        cache_samples = natural.drop_duplicates(["seed", "gold_count"])
        raw_cache = cache_samples[cache_columns[0]].astype(float)
        centered_cache = cache_samples[cache_columns[1]].astype(float)
        checks.append(
            {
                "name": "attention_cache_diagnostics_finite",
                "passed": bool(
                    math.isfinite(cache_reference_tolerance)
                    and np.isfinite(raw_cache).all()
                    and np.isfinite(centered_cache).all()
                ),
                "detail": {
                    "policy": (
                        "diagnostic_only; eager endpoint reconstruction is the hard gate"
                    ),
                    "reference_tolerance": cache_reference_tolerance,
                    "max_raw_candidate_logit_delta": float(raw_cache.max()),
                    "max_centered_candidate_logit_delta": float(centered_cache.max()),
                    "reference_tolerance_exceedance_samples": int(
                        cache_samples[cache_columns[2]].astype(bool).sum()
                    ),
                    "sample_count": int(len(cache_samples)),
                },
            }
        )
    control_cosine = float(write["orthogonal_control_axis_cosine"].abs().max())
    checks.append(
        {
            "name": "write_control_orthogonality",
            "passed": control_cosine <= 1e-4,
            "detail": control_cosine,
        }
    )
    seeds = sorted(int(value) for value in natural["seed"].unique())
    checks.append(
        {
            "name": "evaluation_seed_registry",
            "passed": seeds == list(config.evaluation_seeds),
            "detail": seeds,
        }
    )
    campaign_roots = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("read_write_")
    ]
    forbidden = [
        str(path)
        for campaign_root in campaign_roots
        for path in campaign_root.rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in ("raw_attention", "full_hidden", "full_value")
        )
    ]
    checks.append(
        {
            "name": "no_raw_attention_or_full_state_artifacts",
            "passed": not forbidden,
            "detail": forbidden,
        }
    )
    return {
        "all_checks_pass": all(bool(check["passed"]) for check in checks),
        "check_count": len(checks),
        "checks": checks,
        "observed_rows": observed,
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = frame.copy()
    result = result.where(pd.notna(result), None)
    return result.to_dict("records")


def _markdown_report(
    payload: Mapping[str, Any], *, config: V444ReadWriteConfig
) -> str:
    decision = payload["primary_decision"]
    read = decision["read_mode"]
    write = decision["write_propagation"]
    summary = pd.DataFrame(payload["summary"])
    lines = [
        "# Realistic NIAH V4.4.4 补充：state 读取分解与 OV 写入报告",
        "",
        "## 总结论",
        "",
        f"读取模式判定为 **{read['classification']}**；下游写入传播支持为 "
        f"**{write['supported']}**；联合 read→write 路径支持为 "
        f"**{decision['serial_read_write_supported']}**。",
        "",
        "本实验复用父 V4.4.4 的 evaluation seeds，属于冻结候选后的机制扩展，"
        "不是全新 seed 的独立复现。*本段结论：当前统计量可检验机制分解，"
        "但最终发表级确认仍需 1294–1313 新 seed 复现。*",
        "",
        "## 1. 猜想与可证伪预测",
        "",
        "V4.4.4 已支持 L28 H16/H19 是自然 OV transporter，但没有说明它们的 "
        "pre-O state 主要来自 V-content 改变还是 alpha-routing 改变。本实验预测："
        "真实 donor-Z transport 应能被二者分解，且真正的读取分量既应推动 donor "
        "count，也应被 frozen natural OV-axis block 特异性削弱。",
        "",
        "```text",
        "Δz_value = 1/2[(z_RD-z_RR)+(z_DD-z_DR)]",
        "Δz_route = 1/2[(z_DR-z_RR)+(z_DD-z_RD)]",
        "Δz_full  = Δz_value + Δz_route",
        "```",
        "",
        "其中 RR/DD 使用模型实际 fused pre-O 端点，RD/DR 使用 crossed "
        "alpha-V 计算。*本节结论：这是对同一次真实 donor-Z movement 的因果"
        "分账，不把 QK head 与 OV head 预设为同一组。*",
        "",
        "## 2. 实验设定",
        "",
        f"- 模型/候选：{config.model_label}, L{config.mediator_layer}, "
        f"heads={list(config.heads)}。",
        f"- axis discovery seeds：{config.discovery_seeds[0]}–"
        f"{config.discovery_seeds[-1]}；evaluation seeds："
        f"{config.evaluation_seeds[0]}–{config.evaluation_seeds[-1]}。",
        f"- counts={list(config.counts)}；directed donor pairs="
        f"{list(config.donor_pairs)}；write counts={list(config.write_counts)}。",
        f"- post-block trace layers={list(config.downstream_layers)}；"
        f"pre-O intervention beta={config.write_beta}。",
        "- 主分析使用全部 paired trials；baseline-correct/wrong 仅在 discovery "
        "固定的 count axes 上分层，不重新拟合 PCA/count axis。",
        "",
        "*本节结论：方向估计、机制评价和 outcome 分层之间没有按结果重新选轴。*",
        "",
        "## 3. 读取分解结果",
        "",
        "| metric | mean | 95% CI | p |",
        "|---|---:|---:|---:|",
    ]
    read_metrics = summary[
        summary["metric"].isin(
            [
                "read_full_behavior_transport",
                "read_value_behavior_transport",
                "read_routing_behavior_transport",
                "read_value_ov_mediation_specificity",
                "read_routing_ov_mediation_specificity",
                "read_value_minus_routing_transport",
            ]
        )
        & summary["stratum"].eq("all")
    ]
    for row in read_metrics.to_dict("records"):
        lines.append(
            f"| {row['metric']} | {row['mean']:.6g} | "
            f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] | "
            f"{row['exact_sign_flip_p']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"value family p={read['value_family_p']:.6g}；routing family "
            f"p={read['routing_family_p']:.6g}；value-minus-routing "
            f"p={read['value_minus_routing_p']:.6g}。",
            "",
            f"*本节结论：按冻结判据，读取模式为 **{read['classification']}**。"
            "只有 component transport 与 natural-OV mediation 同时成立才计为"
            "自然读取证据。*",
            "",
            "## 4. OV 写入与层间传播",
            "",
            "在 L28 真实 pre-O 边界施加 ±β natural z-step，并与同一 H16/H19 "
            "W_O span 内、等 post-O 范数的正交方向比较。纵向 estimand 为",
            "",
            "```text",
            "coefficient_l = <[h_l(+β)-h_l(-β)]/(2β), s_l> / ||s_l||²",
            "```",
            "",
            "其中 s_l 是 discovery seeds 上拟合的该层自然 answer-query count step。",
            "",
            "| layer | natural slope | orthogonal slope | specificity | Holm p |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in config.downstream_layers:
        layer_rows = summary[
            summary["layer"].fillna(-1).astype(int).eq(int(layer))
            & summary["stratum"].eq("all")
        ].set_index("metric")
        natural_row = layer_rows.loc["write_natural_residual_slope"]
        control_row = layer_rows.loc["write_orthogonal_residual_slope"]
        specificity_row = layer_rows.loc["write_residual_specificity"]
        lines.append(
            f"| L{layer} | {natural_row['mean']:.6g} | "
            f"{control_row['mean']:.6g} | {specificity_row['mean']:.6g} | "
            f"{specificity_row['holm_p_within_family_metric']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"答案分布上的 natural-minus-orthogonal specificity mean="
            f"{write['behavior_specificity_mean']:.6g}, "
            f"p={write['behavior_specificity_p']:.6g}；最终 L{write['final_layer']} "
            f"residual specificity={write['final_residual_specificity_mean']:.6g}, "
            f"Holm p={write['final_residual_specificity_holm_p']:.6g}。",
            "",
            f"*本节结论：下游写入传播支持为 **{write['supported']}**。只有答案分布"
            "与最终层固定自然 count axis 同时优于正交控制，才判定写入存活。*",
            "",
            "## 5. 正确/错误基线与证据边界",
            "",
            "正确/错误分层只用于敏感性分析；由于所有轴均在分层前冻结，答错样本"
            "不会通过重新拟合改变 geometry。若某一层样本过少，其区间应视为描述性"
            "而不是主检验。",
            "",
            f"审计通过：**{payload['audit']['all_checks_pass']}**，共 "
            f"{payload['audit']['check_count']} 项。未持久化 full hidden state、"
            "full V tensor 或 raw attention map。",
            "eager/cache candidate-logit 差异仅作为数值诊断记录；硬门槛是"
            " all-key eager alpha-V 对真实 pre-O endpoint 的相对 L2 重建误差。",
            "",
            "即使联合检验为正，也只定位 terminal state-component → H16/H19 "
            "pre-O Z → natural OV axis → downstream count state → count distribution。"
            "它不定位构造 V-state 的更早 heads/MLP；该问题需要按本轮 read-mode "
            "结果选择 upstream residual/V 或 Q/K path patch。",
            "",
            "*本节结论：本报告可以判断模型在 terminal attention 中读了哪一类"
            "state 并如何写入，但不能把上游 state builder 一并宣称为已识别。*",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_campaign(
    run_root: str | Path, *, config: V444ReadWriteConfig
) -> dict[str, Any]:
    root = Path(run_root)
    evaluation = stage_root(root, config.model_label, "read_write_evaluation")
    natural = pd.read_csv(evaluation / "natural_detail.csv.gz")
    read = pd.read_csv(evaluation / "read_causal_detail.csv.gz")
    read_trace = pd.read_csv(evaluation / "read_trace_detail.csv.gz")
    write = pd.read_csv(evaluation / "write_trace_detail.csv.gz")
    seed_metrics = build_seed_metrics(
        natural, read, read_trace, write, config=config
    )
    summary = summarize_seed_metrics(seed_metrics, config=config)
    decision = primary_decision(summary, config=config)
    audit = audit_campaign(root, config=config)
    analysis_root = stage_root(root, config.model_label, "read_write_analysis")
    atomic_csv_gzip(seed_metrics, analysis_root / "seed_metrics.csv.gz")
    atomic_csv_gzip(summary, analysis_root / "metric_summary.csv.gz")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_read_write_analysis_v1",
        "primary_decision": decision,
        "audit": audit,
        "summary": _json_records(summary),
    }
    atomic_json(
        analysis_root / "realistic_niah_v4_4_4_read_write_analysis.json", payload
    )
    atomic_text(
        analysis_root / "realistic_niah_v4_4_4_read_write_report.md",
        _markdown_report(payload, config=config),
    )
    atomic_json(
        analysis_root / "complete.json",
        {
            "schema_version": "realistic_niah_v4_4_4_read_write_analysis_complete_v1",
            "audit_all_checks_pass": audit["all_checks_pass"],
            "primary_decision": decision,
        },
    )
    return payload
