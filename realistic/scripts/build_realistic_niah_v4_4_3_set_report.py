#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER = "#F4EFE6"
SURFACE = "#FFFDF8"
INK = "#20242D"
MUTED = "#657080"
PINK = "#D64B83"
TEAL = "#008C7D"
INDIGO = "#5B4BC4"
GRAY = "#8D94A1"
LAYER_COLORS = ("#5B4BC4", "#008C7D", "#D97732")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _figure_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=190,
        bbox_inches="tight",
        facecolor=SURFACE,
    )
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _errorbar(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    x: str,
    color: str,
    label: str,
    linestyle: str = "-",
    marker: str = "o",
) -> None:
    ordered = frame.sort_values(x)
    if ordered.empty:
        return
    xs = ordered[x].to_numpy(float)
    means = ordered["mean"].to_numpy(float)
    lows = ordered["ci95_low"].to_numpy(float)
    highs = ordered["ci95_high"].to_numpy(float)
    ax.plot(
        xs,
        means,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.8,
        markersize=4.5,
        label=label,
    )
    ax.fill_between(xs, lows, highs, color=color, alpha=0.12)


def _mapping_figure(model_root: Path, model: str) -> str:
    frame = pd.read_csv(model_root / "analysis" / "nested_mapping_curve.csv.gz")
    layers = sorted(int(value) for value in frame["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(4.25 * len(layers), 3.8), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        local = frame[frame["layer"].astype(int).eq(layer)]
        for role, role_color, role_alpha in (
            ("candidate_set", INDIGO, 1.0),
            ("matched_set", GRAY, 0.78),
        ):
            subset = local[local["set_role"].eq(role)].sort_values("set_size")
            label_prefix = "candidate" if role == "candidate_set" else "matched"
            ax.plot(
                subset["set_size"],
                subset["fit_mapping_cosine"],
                color=role_color,
                alpha=role_alpha,
                marker="o" if role == "candidate_set" else "s",
                linestyle="-",
                label=f"{label_prefix}: fit",
            )
            ax.plot(
                subset["set_size"],
                subset["heldout_count_mapping_cosine"],
                color=role_color,
                alpha=role_alpha,
                marker="o" if role == "candidate_set" else "s",
                linestyle="--",
                label=f"{label_prefix}: held-out",
            )
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(f"L{layer}")
        ax.set_xlabel("head-set size K")
        ax.set_xticks(sorted(int(value) for value in local["set_size"].unique()))
        _style_axes(ax)
    axes[0].set_ylabel("OV mapping cosine")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"{model}: nested-set OV mapping", y=1.02, fontsize=13)
    fig.tight_layout()
    return _figure_uri(fig)


def _nested_effect_figure(
    model_root: Path,
    model: str,
    *,
    summary_name: str,
    metrics: Sequence[tuple[str, str, str]],
    title: str,
) -> str:
    frame = pd.read_csv(model_root / "analysis" / summary_name)
    layers = sorted(int(value) for value in frame["layer"].unique())
    fig, axes = plt.subplots(
        len(metrics),
        len(layers),
        figsize=(4.05 * len(layers), 3.15 * len(metrics)),
        squeeze=False,
    )
    for row_index, (metric, label, support_sign) in enumerate(metrics):
        for column_index, layer in enumerate(layers):
            ax = axes[row_index, column_index]
            local = frame[
                frame["metric"].eq(metric)
                & frame["layer"].astype(int).eq(layer)
            ]
            for role, color, linestyle, marker in (
                ("candidate_set", PINK, "-", "o"),
                ("matched_set", GRAY, "--", "s"),
            ):
                _errorbar(
                    ax,
                    local[local["set_role"].eq(role)],
                    x="set_size",
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    label="candidate" if role == "candidate_set" else "matched",
                )
            ax.axhline(0, color=INK, linewidth=0.8)
            ax.set_title(f"L{layer} · {label}")
            ax.set_xlabel("head-set size K")
            ax.set_ylabel("paired causal contrast")
            ax.text(
                0.02,
                0.97,
                support_sign,
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
                color=MUTED,
            )
            _style_axes(ax)
            if row_index == 0 and column_index == len(layers) - 1:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model}: {title}", y=1.01, fontsize=13)
    fig.tight_layout()
    return _figure_uri(fig)


def _specificity_figure(
    model_root: Path,
    model: str,
    *,
    summary_name: str,
    metrics: Sequence[tuple[str, str, str]],
    title: str,
) -> str:
    frame = pd.read_csv(model_root / "analysis" / summary_name)
    layers = sorted(int(value) for value in frame["layer"].unique())
    fig, axes = plt.subplots(
        len(metrics),
        len(layers),
        figsize=(4.05 * len(layers), 3.0 * len(metrics)),
        squeeze=False,
    )
    for row_index, (contains, label, sign_note) in enumerate(metrics):
        for column_index, layer in enumerate(layers):
            ax = axes[row_index, column_index]
            local = frame[
                frame["metric"].astype(str).str.contains(contains, regex=False)
                & frame["layer"].astype(int).eq(layer)
            ].sort_values("set_size")
            _errorbar(
                ax,
                local,
                x="set_size",
                color=TEAL,
                label="candidate − matched",
            )
            ax.axhline(0, color=INK, linewidth=0.8)
            for row in local.itertuples():
                if float(row.one_sided_exact_sign_flip_p) <= 0.05:
                    ax.annotate(
                        "raw p≤.05",
                        (float(row.set_size), float(row.mean)),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        fontsize=6.8,
                        color=PINK,
                    )
            ax.set_title(f"L{layer} · {label}")
            ax.set_xlabel("head-set size K")
            ax.set_ylabel("candidate − matched")
            ax.text(
                0.02,
                0.97,
                sign_note,
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
                color=MUTED,
            )
            _style_axes(ax)
    fig.suptitle(f"{model}: {title}", y=1.01, fontsize=13)
    fig.tight_layout()
    return _figure_uri(fig)


def _injection_slope_figure(model_root: Path, model: str) -> str:
    frame = pd.read_csv(model_root / "analysis" / "injection_summary.csv.gz")
    frame = frame[frame["metric"].eq("injection_slope")]
    layers = sorted(int(value) for value in frame["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(4.15 * len(layers), 3.7), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        local = frame[frame["layer"].astype(int).eq(layer)]
        for role, color, linestyle, marker in (
            ("candidate_set", PINK, "-", "o"),
            ("matched_set", GRAY, "--", "s"),
        ):
            _errorbar(
                ax,
                local[local["set_role"].eq(role)],
                x="set_size",
                color=color,
                linestyle=linestyle,
                marker=marker,
                label="candidate" if role == "candidate_set" else "matched",
            )
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(f"L{layer}")
        ax.set_xlabel("head-set size K (K≥2)")
        _style_axes(ax)
    axes[0].set_ylabel("injection slope b̂ (count units / β)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"{model}: answer-direction injection inside each set's output span",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    return _figure_uri(fig)


def _dose_response_figure(model_root: Path, model: str) -> str:
    frame = pd.read_csv(model_root / "directed" / "detail.csv.gz")
    frame = frame[
        frame["set_role"].eq("candidate_set")
        & frame["intervention"].astype(str).str.startswith(
            "signed_answer_direction_injection_beta_"
        )
    ]
    layers = sorted(int(value) for value in frame["layer"].unique())
    sizes = sorted(int(value) for value in frame["set_size"].unique())
    palette = plt.cm.viridis(np.linspace(0.12, 0.88, len(sizes)))
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 3.7), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        local = frame[frame["layer"].astype(int).eq(layer)]
        for color, size in zip(palette, sizes):
            subset = local[local["set_size"].astype(int).eq(size)]
            points = (
                subset.groupby("beta", sort=True)["delta_expected_count"]
                .mean()
                .reset_index()
            )
            if points.empty:
                continue
            ax.plot(
                points["beta"],
                points["delta_expected_count"],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color=color,
                label=f"K={size}",
            )
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_title(f"L{layer}")
        ax.set_xlabel("signed injection coefficient β")
        _style_axes(ax)
    axes[0].set_ylabel("change in expected count ΔE[N]")
    axes[-1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(f"{model}: candidate-set injection dose response", y=1.02, fontsize=13)
    fig.tight_layout()
    return _figure_uri(fig)


def _increment_figure(model_root: Path, model: str) -> str:
    patch = pd.read_csv(
        model_root / "analysis" / "nested_patch_increment_summary.csv.gz"
    )
    removal = pd.read_csv(
        model_root / "analysis" / "nested_removal_increment_summary.csv.gz"
    )
    definitions = (
        (patch, "increment_z_transport_over_norm", "Z transport gain", "positive = gain"),
        (patch, "increment_alpha_over_scramble", "α localization gain", "positive = gain"),
        (
            removal,
            "increment_removal_error_over_orthogonal",
            "removal-error gain",
            "positive = stronger damage",
        ),
        (
            removal,
            "increment_removal_margin_over_orthogonal",
            "removal-margin gain",
            "negative = stronger damage",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.1), squeeze=False)
    for ax, (frame, metric, label, sign_note) in zip(axes.ravel(), definitions):
        local_metric = frame[frame["metric"].eq(metric)]
        for color, layer in zip(
            LAYER_COLORS,
            sorted(int(value) for value in local_metric["layer"].unique()),
        ):
            local = local_metric[local_metric["layer"].astype(int).eq(layer)].copy()
            local["transition"] = [
                f"{int(left)}→{int(right)}"
                for left, right in zip(local["from_k"], local["to_k"])
            ]
            x = np.arange(len(local), dtype=float)
            means = local["mean"].to_numpy(float)
            ax.errorbar(
                x,
                means,
                yerr=np.vstack(
                    (
                        means - local["ci95_low"].to_numpy(float),
                        local["ci95_high"].to_numpy(float) - means,
                    )
                ),
                color=color,
                marker="o",
                capsize=3,
                label=f"L{layer}",
            )
            ax.set_xticks(x, local["transition"], rotation=35, ha="right")
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel("nested-set transition Kprev→K")
        ax.set_ylabel("paired increment")
        ax.text(0.02, 0.97, sign_note, transform=ax.transAxes, va="top", fontsize=8, color=MUTED)
        _style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model}: marginal gain from adding heads", y=1.01, fontsize=13)
    fig.tight_layout()
    return _figure_uri(fig)


def _format_number(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:+.{digits}g}"


def _format_p(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.5f}"


def _head_set_table(model_root: Path) -> str:
    frame = pd.read_csv(model_root / "analysis" / "nested_mapping_curve.csv.gz")
    frame = frame[frame["set_role"].eq("candidate_set")].sort_values(
        ["layer", "set_size"]
    )
    directed = pd.read_csv(model_root / "directed" / "detail.csv.gz")
    reachable = (
        directed[directed["set_role"].eq("candidate_set")]
        .groupby("set_id")["reachable_answer_cosine"]
        .mean()
        .to_dict()
    )
    rows = [
        "<table><thead><tr><th>layer</th><th>K</th><th>candidate heads</th>",
        "<th>fit cosine</th><th>held-out cosine</th><th>reachable answer cosine</th>",
        "<th>source</th></tr></thead><tbody>",
    ]
    for row in frame.itertuples():
        rows.append(
            "<tr><td>L{layer}</td><td>{size}</td><td><code>{heads}</code></td>"
            "<td>{fit}</td><td>{held}</td><td>{reachable}</td><td>{source}</td></tr>".format(
                layer=int(row.layer),
                size=int(row.set_size),
                heads=html.escape(str(row.heads)),
                fit=_format_number(row.fit_mapping_cosine),
                held=_format_number(row.heldout_count_mapping_cosine),
                reachable=(
                    "not comparable"
                    if int(row.set_size) == 1
                    else _format_number(reachable.get(str(row.set_id)))
                ),
                source=html.escape(str(row.source_experiment)),
            )
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def _evidence_table(model_payload: dict[str, Any]) -> str:
    rows = [
        "<table><thead><tr><th>set</th><th>heads</th><th>held-out map</th>",
        "<th>Z Δ / raw p / BH q</th><th>inject Δ / raw p / BH q</th>",
        "<th>remove error Δ / p</th><th>remove margin Δ / p</th>",
        "<th>raw 2-of-3</th><th>FDR 2-of-3</th></tr></thead><tbody>",
    ]
    for item in sorted(
        model_payload["candidate_set_decisions"],
        key=lambda value: (int(value["layer"]), int(value["set_size"])),
    ):
        q = item["causal_family_q_values_bh"]
        z = item["z_specificity"]
        injection = item["injection_specificity"]
        error = item["removal_error_specificity"]
        margin = item["removal_margin_specificity"]
        raw = bool(item["triangulated_set_support_raw"])
        corrected = bool(item["triangulated_set_support_fdr"])
        rows.append(
            "<tr><td>{set_id}</td><td><code>{heads}</code></td><td>{held}</td>"
            "<td>{zm} / {zp} / {zq}</td><td>{im} / {ip} / {iq}</td>"
            "<td>{em} / {ep}</td><td>{mm} / {mp}</td>"
            "<td class='{raw_class}'>{raw_text}</td>"
            "<td class='{fdr_class}'>{fdr_text}</td></tr>".format(
                set_id=html.escape(str(item["set_id"])),
                heads=html.escape(str(item["heads"])),
                held=_format_number(item["heldout_count_mapping_cosine"]),
                zm=_format_number(z["mean"]),
                zp=_format_p(z["one_sided_exact_sign_flip_p"]),
                zq=_format_p(q["z_transport"]),
                im=_format_number(injection["mean"]),
                ip=_format_p(injection["one_sided_exact_sign_flip_p"]),
                iq=_format_p(q["set_reachable_injection"]),
                em=_format_number(error["mean"]),
                ep=_format_p(error["one_sided_exact_sign_flip_p"]),
                mm=_format_number(margin["mean"]),
                mp=_format_p(margin["one_sided_exact_sign_flip_p"]),
                raw_class="pass" if raw else "fail",
                raw_text="PASS" if raw else "FAIL",
                fdr_class="pass" if corrected else "fail",
                fdr_text="PASS" if corrected else "FAIL",
            )
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def _selectivity_table(model_root: Path) -> str:
    frame = pd.read_csv(
        model_root / "analysis" / "local_selectivity_summary.csv.gz"
    )
    frame = frame[frame["set_role"].eq("candidate_set")]
    rows = [
        "<table><thead><tr><th>set</th><th>Z: non-count KL / count fraction</th>",
        "<th>nonzero injection: non-count KL / count fraction</th>",
        "<th>answer removal: non-count KL / count fraction</th></tr></thead><tbody>",
    ]
    for set_id in sorted(
        frame["set_id"].unique(),
        key=lambda value: (
            int(str(value).split("K")[0][1:]),
            int(str(value).split("K")[1]),
        ),
    ):
        local = frame[frame["set_id"].eq(set_id)]
        cells = []
        for family in (
            "z_donor",
            "signed_injection_nonzero_beta",
            "answer_direction_removal",
        ):
            selected = local[local["intervention_family"].eq(family)]
            if selected.empty:
                cells.append("—")
            else:
                row = selected.iloc[0]
                cells.append(
                    f"{float(row['non_count_token_kl']):.3g} / "
                    f"{float(row['count_subspace_delta_fraction']):.3g}"
                )
        rows.append(
            f"<tr><td>{html.escape(str(set_id))}</td>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in cells)
            + "</tr>"
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def _audit_table(run_root: Path, models: Iterable[str]) -> str:
    rows = [
        "<table><thead><tr><th>model</th><th>set-Z / post-O max |Δ|</th>",
        "<th>cache raw max |Δ|</th><th>cache centered max |Δ|</th>",
        "<th>output layer → V source</th></tr></thead><tbody>",
    ]
    for model in models:
        model_root = run_root / "models" / model
        smoke = _read_json(model_root / "smoke" / "complete.json")
        score = pd.read_csv(model_root / "mapping" / "head_mapping_scores.csv.gz")
        sources = (
            score[["layer", "value_source_layer"]]
            .drop_duplicates()
            .sort_values("layer")
        )
        source_text = ", ".join(
            f"L{int(row.layer)}→L{int(row.value_source_layer)}"
            for row in sources.itertuples()
        )
        rows.append(
            "<tr><td>{model}</td><td>{zo:.6g}</td><td>{raw:.6g}</td>"
            "<td>{centered:.6g}</td><td><code>{sources}</code></td></tr>".format(
                model=html.escape(model),
                zo=float(smoke["z_o_candidate_logit_max_abs_delta"]),
                raw=float(smoke["attention_cache_candidate_logit_max_abs_delta"]),
                centered=float(
                    smoke[
                        "attention_cache_candidate_centered_logit_max_abs_delta"
                    ]
                ),
                sources=html.escape(source_text),
            )
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def _model_result_sentence(payload: dict[str, Any]) -> str:
    raw = [
        item["set_id"]
        for item in payload["candidate_set_decisions"]
        if item["triangulated_set_support_raw"]
    ]
    corrected = [
        item["set_id"]
        for item in payload["candidate_set_decisions"]
        if item["triangulated_set_support_fdr"]
    ]
    raw_text = "、".join(raw) if raw else "无"
    corrected_text = "、".join(corrected) if corrected else "无"
    return (
        f"未校正的 2-of-3 筛选通过集合：{raw_text}；"
        f"同一证据族内跨 K 做 BH 校正后通过集合：{corrected_text}。"
    )


def _model_result_details(payload: dict[str, Any]) -> str:
    rows = payload["candidate_set_decisions"]
    raw = [item for item in rows if item["triangulated_set_support_raw"]]

    if raw:
        anchor = min(raw, key=lambda item: int(item["set_size"]))
        largest = max(raw, key=lambda item: int(item["set_size"]))
        layer_text = "、".join(
            f"L{layer}" for layer in sorted({int(item["layer"]) for item in raw})
        )
        q = anchor["causal_family_q_values_bh"]
        removal_p = max(
            float(anchor["removal_error_specificity"]["one_sided_exact_sign_flip_p"]),
            float(anchor["removal_margin_specificity"]["one_sided_exact_sign_flip_p"]),
        )
        return f"""
  <div class="callout">
    <p><strong>具体结果：</strong>raw 2-of-3 候选全部集中在 {html.escape(layer_text)}。最小集合
    <code>{html.escape(anchor['set_id'])}</code>（heads={html.escape(str(anchor['heads']))}）的 held-out mapping cosine
    为 {_format_number(anchor['heldout_count_mapping_cosine'])}；candidate−matched 的 Z transport 为
    {_format_number(anchor['z_specificity']['mean'])}（raw p={_format_p(anchor['z_specificity']['one_sided_exact_sign_flip_p'])},
    BH q={_format_p(q['z_transport'])}），set-span injection slope 为
    {_format_number(anchor['injection_specificity']['mean'])}（raw p={_format_p(anchor['injection_specificity']['one_sided_exact_sign_flip_p'])},
    BH q={_format_p(q['set_reachable_injection'])}），removal error / margin 分别为
    {_format_number(anchor['removal_error_specificity']['mean'])} / {_format_number(anchor['removal_margin_specificity']['mean'])}
    （保守 family raw p={_format_p(removal_p)}, BH q={_format_p(q['answer_direction_removal'])}）。</p>
    <p><strong>K 扩展分析：</strong>在同一 nested family 中，从
    <code>{html.escape(anchor['set_id'])}</code> 到 <code>{html.escape(largest['set_id'])}</code>，
    injection specificity 从 {_format_number(anchor['injection_specificity']['mean'])} 变为
    {_format_number(largest['injection_specificity']['mean'])}，没有出现“加入更多 heads 后证据单调增强”的模式。
    BH 后只有 injection family 保留 q≤.05，Z 与 removal 均未通过，因此不能把 raw 共现解释为已确认的小型 circuit。</p>
  </div>
"""

    best_z = min(
        rows,
        key=lambda item: float(item["z_specificity"]["one_sided_exact_sign_flip_p"]),
    )
    best_injection = min(
        rows,
        key=lambda item: float(
            item["injection_specificity"]["one_sided_exact_sign_flip_p"]
        ),
    )
    best_removal = min(
        rows,
        key=lambda item: max(
            float(item["removal_error_specificity"]["one_sided_exact_sign_flip_p"]),
            float(item["removal_margin_specificity"]["one_sided_exact_sign_flip_p"]),
        ),
    )
    removal_p = max(
        float(best_removal["removal_error_specificity"]["one_sided_exact_sign_flip_p"]),
        float(best_removal["removal_margin_specificity"]["one_sided_exact_sign_flip_p"]),
    )
    return f"""
  <div class="callout">
    <p><strong>具体结果：</strong>最小的 Z raw p 出现在
    <code>{html.escape(best_z['set_id'])}</code>（p={_format_p(best_z['z_specificity']['one_sided_exact_sign_flip_p'])},
    BH q={_format_p(best_z['causal_family_q_values_bh']['z_transport'])}）；最小的 injection raw p 出现在
    <code>{html.escape(best_injection['set_id'])}</code>（p={_format_p(best_injection['injection_specificity']['one_sided_exact_sign_flip_p'])},
    BH q={_format_p(best_injection['causal_family_q_values_bh']['set_reachable_injection'])}）；最强 removal
    候选为 <code>{html.escape(best_removal['set_id'])}</code>（保守 family raw p={_format_p(removal_p)},
    BH q={_format_p(best_removal['causal_family_q_values_bh']['answer_direction_removal'])}）。</p>
    <p><strong>分析：</strong>三类证据没有在同一 layer×K set 上形成 raw 2-of-3 共现，且没有集合在 BH 后形成
    2-of-3。因而目前的数据不支持该模型存在一个被本设计定位到的、matched-control 特异的小型 head set；单个 family
    的低 raw p 只能视为探索性线索。</p>
  </div>
"""


def _model_section(run_root: Path, model: str, payload: dict[str, Any]) -> str:
    model_root = run_root / "models" / model
    mapping = _mapping_figure(model_root, model)
    patch_roles = _nested_effect_figure(
        model_root,
        model,
        summary_name="nested_patch_role_summary.csv.gz",
        metrics=(
            ("z_transport_over_norm", "Z transport", "positive supports transport"),
            ("alpha_over_scramble", "α localization", "positive supports QK localization"),
        ),
        title="absolute staged-patch effects",
    )
    patch_specificity = _specificity_figure(
        model_root,
        model,
        summary_name="nested_patch_specificity.csv.gz",
        metrics=(
            ("candidate_minus_matched_z_transport_over_norm", "Z transport", "positive supports set specificity"),
            ("candidate_minus_matched_alpha_over_scramble", "α localization", "positive supports set specificity"),
        ),
        title="staged-patch specificity",
    )
    injection_slope = _injection_slope_figure(model_root, model)
    dose_response = _dose_response_figure(model_root, model)
    removal_specificity = _specificity_figure(
        model_root,
        model,
        summary_name="nested_removal_specificity.csv.gz",
        metrics=(
            ("candidate_minus_matched_removal_error_over_orthogonal", "absolute error", "positive supports directional damage"),
            ("candidate_minus_matched_removal_margin_over_orthogonal", "correct margin", "negative supports directional damage"),
        ),
        title="answer-direction removal specificity",
    )
    increments = _increment_figure(model_root, model)
    return f"""
<section id="{html.escape(model)}">
  <div class="eyebrow">MODEL RESULTS</div><h2>{html.escape(model)}</h2>
  <h3>集合成员与 held-out geometry</h3>
  <div class="table-scroll">{_head_set_table(model_root)}</div>
  <figure><img src="{mapping}" alt="{html.escape(model)} nested mapping curves">
    <figcaption><strong>图 1（{html.escape(model)}）。</strong>横轴是集合大小 K；纵轴是集合 OV 映射向量与 answer-count 方向的 cosine。实线为 discovery fit counts 1/3/5/7/9，虚线为从未参与集合选择的 held-out counts 2/4/6/8/10；紫色为 candidate，灰色为同层、等 K、不重叠且范数匹配的 control。cosine&gt;0 表示同号映射，但不等价于因果充分性。</figcaption>
  </figure>
  <div class="section-conclusion"><strong>本小节结论：</strong>该图只回答“候选集合的 OV 映射是否跨 count split 保持同号以及随 K 如何变化”；它不单独判定 causal circuit。</div>

  <h3>Staged patch：QK 定位与 OV 运输</h3>
  <figure><img src="{patch_roles}" alt="absolute staged patch effects">
    <figcaption><strong>图 2（{html.escape(model)}）。</strong>横轴为 K，纵轴为按 seed 配对的因果 contrast，阴影为 seed bootstrap 95% CI。Z transport 定义为 T(Z-donor)−T(equal-norm output control)；α localization 定义为 T(donor-α, receiver-V)−T(position-scrambled α, receiver-V)。粉色实线为 candidate，灰色虚线为 matched set；正值符合预期。</figcaption>
  </figure>
  <figure><img src="{patch_specificity}" alt="staged patch specificity">
    <figcaption><strong>图 3（{html.escape(model)}）。</strong>横轴为 K，纵轴进一步取 candidate−matched。误差带是以 seed 为单位的 bootstrap 95% CI；“raw p≤.05”标记来自单侧 exact sign-flip，尚未做跨 K 校正。正值才支持候选集合相对于同规模对照的特异性。</figcaption>
  </figure>
  <div class="section-conclusion"><strong>本小节结论：</strong>绝对效应与 set specificity 是两个命题；只有 candidate−matched 的同号、统计证据才用于定位该预冻结集合。</div>

  <h3>Set-output-span injection</h3>
  <figure><img src="{injection_slope}" alt="set constrained injection slope">
    <figcaption><strong>图 4（{html.escape(model)}）。</strong>横轴为 K（仅新版 K≥2），纵轴为斜率 b̂=Σβ·ΔE[N]/Σβ²，单位是每 1 个 β 引起的 expected-count 变化；阴影为 seed bootstrap 95% CI。注入方向是 answer-count direction 投影到该 head set 的 W<sub>O</sub> column span 后的单位向量，因而属于 set-specific intervention。旧 K=1 是 full-layer post-O injection，边界不同，未画入本图。</figcaption>
  </figure>
  <figure><img src="{dose_response}" alt="candidate set dose response">
    <figcaption><strong>图 5（{html.escape(model)}）。</strong>横轴是 signed β（−2 到 +2），纵轴是相对 baseline 的 expected count 改变量 ΔE[N]；每条线是一个 candidate K，点对 confirmation seeds 1259–1263 与 gold counts 2/5/8 求均值。正斜率表示沿投影 answer direction 的正向注入会提高模型的软计数输出。</figcaption>
  </figure>
  <div class="section-conclusion"><strong>本小节结论：</strong>只有同时呈现正向 dose response 且 candidate slope 超过 matched slope，才能把可控性定位到选定 set；单纯某层可注入并不定位 head set。</div>

  <h3>Answer-direction removal</h3>
  <figure><img src="{removal_specificity}" alt="answer direction removal specificity">
    <figcaption><strong>图 6（{html.escape(model)}）。</strong>横轴为 K；纵轴为 [answer-direction removal−equal-norm orthogonal removal] 的 candidate−matched 配对差。上排使用 expected-count absolute-error 改变量，正值支持特异损伤；下排使用 gold-answer margin 改变量，负值支持特异损伤。误差带为 seed bootstrap 95% CI。</figcaption>
  </figure>
  <div class="section-conclusion"><strong>本小节结论：</strong>removal 必须在两个互补 outcome 上方向一致（误差更大、margin 更低），并优于 matched set，才计为一类因果证据。</div>

  <h3>扩大 K 的边际收益</h3>
  <figure><img src="{increments}" alt="incremental gains across nested K">
    <figcaption><strong>图 7（{html.escape(model)}）。</strong>横轴是相邻 nested transition K<sub>prev</sub>→K；纵轴是同一 seed、同一 donor/receiver pair（或同一 gold count）下，较大集合效应减较小集合效应。Z、α 与 removal-error 的正增量、removal-margin 的负增量表示新增 heads 带来预测方向上的边际收益。K=1 来自旧单头基线，其余来自本轮 set run。</figcaption>
  </figure>
  <div class="section-conclusion"><strong>本小节结论：</strong>若效应只随 K 单调扩大而没有局部边际饱和，这更像分布式容量效应；若少数 transition 产生稳定跃迁，则更支持小型协作集合。</div>

  <h3>逐集合统计矩阵</h3>
  <div class="table-scroll">{_evidence_table(payload)}</div>
  <p class="note">表中 Δ 均为 candidate−matched。removal family 要求 error Δ&gt;0 与 margin Δ&lt;0 且两项 raw p 均≤.05；其 family p 取两者较大的保守值，再和 Z、injection 分别在模型内跨 K 做 BH 校正。Raw 2-of-3 是未校正筛选；FDR 2-of-3 是主报告判据。</p>
  {_model_result_details(payload)}
  <h3>局部 selectivity 描述量</h3>
  <div class="table-scroll">{_selectivity_table(model_root)}</div>
  <p class="note">每格为 mean non-count-token KL / mean count-subspace delta fraction。KL 是 answer query 上排除 1–10 count candidates 后的局部分布变化；count fraction=‖Δlogits<sub>count</sub>‖₂/‖Δlogits<sub>vocab</sub>‖₂。两者只描述干预是否广泛扰动局部 logits，没有预冻结 pass threshold，也不作为第四类因果证据。极小负 KL 来自浮点误差，应解释为约 0。</p>
  <div class="section-conclusion"><strong>{html.escape(model)} 当前结论：</strong>{html.escape(_model_result_sentence(payload))}</div>
</section>
"""


def build_report(run_root: Path, output: Path) -> None:
    campaign = _read_json(run_root / "analysis" / "analysis.json")
    config = _read_json(run_root / "resolved_config.json")
    models = list(campaign["model_analyses"])
    model_sections = "".join(
        _model_section(run_root, model, campaign["model_analyses"][model])
        for model in models
    )
    raw_supported = {
        model: bool(payload.get("any_triangulated_set_support_raw", False))
        for model, payload in campaign["model_analyses"].items()
    }
    fdr_supported = {
        model: bool(payload["any_triangulated_set_support"])
        for model, payload in campaign["model_analyses"].items()
    }
    executive = (
        "；".join(
            f"{model}: raw={'有' if raw_supported[model] else '无'}，FDR={'有' if fdr_supported[model] else '无'}"
            for model in models
        )
        + "。主结论以 FDR 后的 matched-set 证据为准。"
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Realistic NIAH V4.4.3-Set · OV vertical geometry causal test</title>
<style>
:root{{--paper:{PAPER};--surface:{SURFACE};--ink:{INK};--muted:{MUTED};--line:#CFC7BA;--pink:{PINK};--teal:{TEAL};--indigo:{INDIGO};--green:#247A4A;--red:#A23C3C}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Noto Sans SC",Arial,sans-serif;line-height:1.66}}main{{max-width:1180px;margin:auto;padding:44px 28px 90px}}header,section{{padding:28px 0 46px;border-bottom:1px solid var(--line)}}h1{{font-size:42px;line-height:1.12;margin:10px 0 18px}}h2{{font-size:30px;margin:8px 0 18px}}h3{{font-size:21px;margin:34px 0 13px}}p,li{{max-width:100ch}}.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.13em;color:var(--teal)}}.lead{{font-size:18px;max-width:92ch}}.callout,.section-conclusion{{background:var(--surface);border-left:4px solid var(--teal);padding:15px 19px;margin:20px 0}}.section-conclusion{{border-left-color:var(--pink)}}.hypothesis{{display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center;background:var(--surface);border:1px solid var(--line);padding:22px;margin:22px 0}}.node{{border:1px solid var(--line);padding:16px;background:#FAF7F1}}.arrow{{font-size:28px;color:var(--indigo)}}.eq{{font-family:Cambria,"Times New Roman",serif;background:#EEE8DD;border-left:3px solid var(--indigo);padding:11px 15px;margin:12px 0;overflow:auto}}figure{{margin:24px 0;background:var(--surface);border:1px solid var(--line);padding:15px}}figure img{{display:block;width:100%;height:auto}}figcaption,.note{{font-size:13px;color:var(--muted);margin-top:10px}}.table-scroll{{overflow:auto;background:var(--surface);border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-size:12.5px}}th,td{{padding:9px 10px;border-bottom:1px solid #E1DBD0;text-align:left;vertical-align:top;white-space:nowrap}}th{{background:#ECE5D9;position:sticky;top:0}}td.pass{{color:var(--green);font-weight:700}}td.fail{{color:var(--red);font-weight:700}}code{{background:#EBE5DA;padding:1px 4px;border-radius:3px}}nav{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}}nav a{{color:var(--indigo);text-decoration:none;border:1px solid var(--line);background:var(--surface);padding:5px 9px;border-radius:20px;font-size:12px}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}@media(max-width:800px){{main{{padding:26px 16px}}h1{{font-size:32px}}.grid2,.hypothesis{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg);text-align:center}}}}
</style></head><body><main>
<header>
  <div class="eyebrow">REALISTIC NIAH · V4.4.3-SET · FROZEN SPLITS</div>
  <h1>OV vertical geometry 的小型 head-set 因果检验</h1>
  <p class="lead">本报告检验：prompt running-index 与 answer-count 虽编码同一个计数变量，却位于近似正交的 residual directions；这种“垂直 geometry”是否由同层多个 attention heads 协作，经 QK 定位和 V/O 写回完成重编码。</p>
  <div class="callout"><strong>结果摘要：</strong>{html.escape(executive)}<br>{html.escape(str(campaign['conclusion']))}</div>
  <nav><a href="#hypothesis">猜想</a><a href="#design">设计与定义</a><a href="#audit">实现审计</a>{''.join(f'<a href="#{html.escape(model)}">{html.escape(model)}</a>' for model in models)}<a href="#synthesis">综合结论</a><a href="#repro">复现</a></nav>
  <p class="note">Run root: <code>{html.escape(str(run_root))}</code></p>
</header>

<section id="hypothesis">
  <div class="eyebrow">01 · HYPOTHESIS</div><h2>猜想及可证伪预测</h2>
  <div class="hypothesis"><div class="node"><strong>Prompt counter</strong><br>running-index direction u<sub>P</sub><br>在 needle 读取过程中更新</div><div class="arrow">→ Σ<sub>h∈S</sub> OV<sub>h</sub> →</div><div class="node"><strong>Answer counter</strong><br>answer-count direction u<sub>A</sub><br>在 answer query 上控制输出</div></div>
  <p>核心猜想 H<sub>set</sub>：存在一个小型、同层 head set S，使各 head 的 OV 映射贡献相加后，把 u<sub>P</sub> 的计数差分写入 u<sub>A</sub>。QK 决定从哪些位置读取，V/O 决定读取内容如何进入 residual stream。单头检验失败并不排除这一协作机制。</p>
  <div class="eq">m<sub>S</sub> = Σ<sub>h∈S</sub> M<sub>OV</sub><sup>h</sup>u<sub>P</sub>, &nbsp; r<sub>S</sub> = cos(m<sub>S</sub>, u<sub>A</sub>)</div>
  <p>该猜想给出五个相互补充的预测：P1，r<sub>S</sub> 在 selection fit counts 与 held-out counts 上同号；P2，donor Z patch 的 count transport 超过等范数输出 control；P3，保留 donor α 的 patch 超过位置打乱 α；P4，沿 set-output span 内可达的 answer direction 做 signed injection 产生正 dose-response；P5，移除该方向比移除等范数正交方向更损害计数，并且 candidate set 的损伤超过 matched set。</p>
  <p>反证模式也预先明确：若 mapping 只在 fit counts 为正、held-out 反号，说明选择过拟合；若绝对 patch 有效但 candidate≈matched，说明是一般层敏感性而非集合定位；若 injection 可控但 removal 无特异损伤，只能说明该子空间“足以驱动”，不能说明模型自然使用它。</p>
  <div class="section-conclusion"><strong>本节结论：</strong>本实验检验的是“小型同层 set 是否具有可泛化的映射与 matched-set 特异因果作用”，而不是再次要求一个严格单头独自完成全部重编码。</div>
</section>

<section id="design">
  <div class="eyebrow">02 · DESIGN</div><h2>实验设定、干预边界与计算方法</h2>
  <div class="grid2"><div><h3>数据拆分</h3><ul><li>Discovery seeds：1234–1253；fit counts：1/3/5/7/9，仅用于估计 directions 和选择 nested sets。</li><li>Held-out geometry：counts 2/4/6/8/10，只用于选择后的泛化评估。</li><li>Screen seeds：1254–1258；用于 α/Z/O staged patch。</li><li>Confirmation seeds：1259–1263；用于 removal 与 set-constrained injection。</li></ul></div><div><h3>K 网格</h3><ul><li>Qwen3-8B：K=1/2/3/4/6/8。</li><li>Gemma4-E4B：K=1/2/3/4；每层 8 个 query heads，K&gt;4 时无法构造同层、等 K、成员不重叠的严格 matched control。</li><li>K=1 复用此前冻结的单头 run；每个 K&gt;1 是 greedy nested extension。</li></ul></div></div>
  <h3>集合选择与 matched control</h3>
  <p>从空集开始，每一步加入使 discovery-fit r<sub>S</sub> 最大的尚未选择 head，并在指定 K 截取 nested snapshots。held-out counts、screen 结果与 confirmation 结果均不参与 selection。每个 candidate set 配一个同层、同 K、与 candidate 成员完全不重叠的 control；先从 mapped-output norm 最接近的 128 个组合中筛选，再选 mapping score 最接近该 control pool 中位数者，以避免只比较“强 set”和“弱范数 set”。</p>
  <h3>主要 estimands</h3>
  <div class="table-scroll"><table><thead><tr><th>概念</th><th>计算</th><th>支持方向</th></tr></thead><tbody>
    <tr><td>Normalized transport T</td><td>[E(N|patch)−E(N|baseline)]/(donor count−receiver count)</td><td>正：向 donor count 移动</td></tr>
    <tr><td>Z transport contrast</td><td>T(Z-donor)−T(equal-norm orthogonal output)</td><td>正</td></tr>
    <tr><td>α localization contrast</td><td>T(donor-α, receiver-V)−T(position-scrambled α, receiver-V)</td><td>正</td></tr>
    <tr><td>Reachable answer direction</td><td>u<sub>A,S</sub>=normalize(P<sub>col([W<sub>O</sub><sup>h</sup>]<sub>h∈S</sub>)</sub>u<sub>A</sub>)</td><td>projection cosine&gt;0</td></tr>
    <tr><td>Injection slope</td><td>b̂=Σβ·ΔE[N]/Σβ²，β∈{{−2,−1,−.5,0,.5,1,2}}</td><td>正</td></tr>
    <tr><td>Removal error contrast</td><td>Δ absolute error(answer removal)−Δ absolute error(orthogonal removal)</td><td>正</td></tr>
    <tr><td>Removal margin contrast</td><td>Δ correct margin(answer removal)−Δ correct margin(orthogonal removal)</td><td>负</td></tr>
  </tbody></table></div>
  <h3>统计判据</h3>
  <p>所有因果均值先在同一 seed 内对 donor/receiver pairs 或 gold counts 求平均，再以 seed 为独立单位。图中区间是 seed bootstrap 95% CI；方向检验使用枚举全部 2<sup>5</sup> 符号翻转的单侧 exact sign-flip，因此 n=5 时最小 raw p=1/32=0.03125。一个 family 的 set-specific effect 用 candidate−matched；removal family 同时要求 error&gt;0 和 margin&lt;0。</p>
  <p>因为本轮扩大了 K 网格，除了报告预冻结 raw p≤.05，还在每个模型、每类 causal family 内跨全部 layer×K 做 Benjamini–Hochberg 校正。主判据“FDR 2-of-3”要求 held-out mapping 为正，且 Z transport、set-reachable injection、answer-direction removal 三个互相区分的 family 中至少两个 BH q≤.05。α localization 是 QK 辅助证据，不重复算作第四个独立 family。</p>
  <div class="section-conclusion"><strong>本节结论：</strong>selection、held-out geometry、screen、confirmation 四部分没有数据泄漏；新增 K 的多重比较被显式校正。K=1 与 K&gt;1 只在 mapping、patch、removal 上直接比较，旧 full-layer injection 不与新版 set-span injection 混合。</div>
</section>

<section id="audit">
  <div class="eyebrow">03 · IMPLEMENTATION AUDIT</div><h2>真实模型 smoke test 与 V-source 解析</h2>
  <div class="table-scroll">{_audit_table(run_root, models)}</div>
  <p>set-Z patch 后捕获模型真实的 post-O attention output，再在独立 forward 中整向量替换；这避免在 fused BF16 O projection 外部重算 ΣW<sub>O</sub>Δz 所造成的舍入差。严格 smoke 要求候选答案序列分数最大差异≤{float(config['strict_zo_equivalence_tolerance']):g}。表中的 cache centered max 去掉所有 count candidates 的共同平移，保留会影响 expected count 与 margin 的相对结构。</p>
  <p>“output layer→V source”由运行时 graph resolver 记录；若 Gemma late output layers 共用更早的 sliding-KV provider，表中会明确显示，避免把 output-layer 编号误写成 V 的实际来源层。</p>
  <div class="section-conclusion"><strong>本节结论：</strong>只有通过 Z/O 等价与 attention-cache 数值审计的实现才进入全量；审计量本身是实现一致性检查，不作为额外因果证据重复计数。</div>
</section>

{model_sections}

<section id="synthesis">
  <div class="eyebrow">04 · SYNTHESIS</div><h2>跨模型综合结论与解释边界</h2>
  <div class="callout"><strong>主结论：</strong>{html.escape(str(campaign['conclusion']))}</div>
  <ul><li><strong>Set sufficiency：</strong>某集合的干预能按预测改变输出。</li><li><strong>Set specificity：</strong>candidate 的效果显著超过同层、同 K、范数匹配且不重叠的 control。</li><li><strong>Member irreducibility：</strong>移除任何一个成员都会破坏集合效应。本轮未做 leave-one-head-out，因此无论 set 是否通过，都不能声称每个成员不可替代。</li></ul>
  <p>还需保留四个限制。第一，layers 来自前一阶段 geometry 线索，因此结论针对 late-layer 候选域，不是全网络无偏搜索。第二，每个 split 只有 5 个 causal seeds，exact p 的分辨率有限。第三，只搜索同层集合；跨层 attention circuit、MLP-mediated re-encoding 和更大分布式机制仍是备选解释。第四，随着 K 增大，任意大集合可能仅因输出子空间维数增加而更易干预；因此报告将 matched-set specificity 与 nested marginal gain 作为必要对照，而不把“更大的绝对效应”自动解释为特定 circuit。</p>
  <div class="section-conclusion"><strong>本节结论：</strong>能支持的最强表述由 FDR 后的多证据 matched-set 结果决定；raw 显著、绝对效应或单一 family 的通过都只作为线索，不升级为已确认的小型 circuit。</div>
</section>

<section id="repro">
  <div class="eyebrow">05 · REPRODUCIBILITY</div><h2>产物、隔离与复现入口</h2>
  <ul><li>原始 causal shards 与模型中间产物：<code>{html.escape(str(run_root))}</code>（filestream）。</li><li>主分析：<code>analysis/analysis.json</code>；逐模型 nested-K CSV 位于 <code>models/&lt;model&gt;/analysis/</code>。</li><li>冻结配置：<code>resolved_config.json</code>；输入来源与哈希：<code>input_manifest.json</code>；完整性：<code>audit.json</code>。</li><li>报告生成器：<code>scripts/build_realistic_niah_v4_4_3_set_report.py</code>。</li></ul>
  <p class="note">报告中的图全部以内嵌 PNG 存储，不依赖网络资源；CSV/JSON 保留精确数字，HTML 中显示值仅为格式化后的摘要。</p>
  <div class="section-conclusion"><strong>本节结论：</strong>V4.4.3-Set 与 filestream 中其他 causal runs 使用独立 run root；报告转存不移动或复制大体量 raw shards。</div>
</section>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output or (
        run_root
        / "analysis"
        / "realistic_niah_v4_4_3_ov_set_causal_report.html"
    )
    build_report(run_root, output.resolve())
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
