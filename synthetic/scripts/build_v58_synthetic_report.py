#!/usr/bin/env python
"""Build the self-contained v58 synthetic NiaH report.

The report consumes archived v58 CSV/JSON artifacts only.  It does not run
model inference.  The primary comparison is between two independently
initialized and independently trained models with matched architecture/data:
separator/no-index Thinking and Non-thinking.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from build_v58_commit_query_section import build_section as build_commit_query_section
from build_v58_native_continuation_section import build_section as build_native_continuation_section


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "v58_final"
ANALYSIS = DATA / "analysis"
REPORT_DIR = ROOT / "reports"
ASSET_DIR = REPORT_DIR / "NiaH_Synthetic_report_assets"

BLUE = "#2563A6"
BLUE_DARK = "#173B63"
ORANGE = "#D97706"
PURPLE = "#7158A6"
GREEN = "#23856D"
RED = "#C64E4E"
GREY = "#7B8794"
LIGHT_GREY = "#D7DEE7"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D8DEE7", linewidth=0.65, alpha=0.72)
    ax.spines[["top", "right"]].set_visible(False)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.titlesize": 14,
        }
    )


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def figure(path: Path, caption: str, alt: str) -> str:
    return (
        f'<figure><img src="{data_uri(path)}" alt="{html.escape(alt)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def pct(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{100 * number:.{digits}f}%"


def num(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def html_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{item}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def plot_behavior(by_count: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.25))
    colors = {"nonthinking": ORANGE, "thinking": PURPLE}
    labels = {"nonthinking": "Non-thinking", "thinking": "Thinking"}

    ax = axes[0]
    for mode in ("nonthinking", "thinking"):
        line = by_count.loc[by_count["mode"].eq(mode)].sort_values("count")
        y = line["ar_final_accuracy"].to_numpy(float)
        lo = line["ar_final_accuracy_wilson95_low"].to_numpy(float)
        hi = line["ar_final_accuracy_wilson95_high"].to_numpy(float)
        ax.errorbar(
            line["count"],
            y,
            yerr=np.vstack((y - lo, hi - y)),
            marker="o",
            linewidth=2,
            capsize=2.5,
            color=colors[mode],
            label=labels[mode],
        )
    ax.set(
        title="A · Independent confirmation by count",
        xlabel="Gold count",
        ylabel="Free-running exact accuracy",
        xticks=range(1, 11),
        ylim=(-0.03, 1.04),
    )
    style_axis(ax)
    ax.legend(loc="center left")

    ax = axes[1]
    ordered = summary.set_index("mode").loc[["nonthinking", "thinking"]]
    y = ordered["ar_final_accuracy"].to_numpy(float)
    lo = ordered["ar_final_accuracy_wilson95_low"].to_numpy(float)
    hi = ordered["ar_final_accuracy_wilson95_high"].to_numpy(float)
    ax.bar([0, 1], y, color=[ORANGE, PURPLE], width=0.62)
    ax.errorbar([0, 1], y, yerr=np.vstack((y - lo, hi - y)), fmt="none", color="#222", capsize=4)
    for x, value in enumerate(y):
        ax.text(x, value + 0.045, f"{100 * value:.2f}%", ha="center", fontweight="bold")
    ax.set(
        title="B · Overall behavior (2,000/mode)",
        ylabel="Free-running exact accuracy",
        xticks=[0, 1],
        xticklabels=["Non-thinking", "Thinking"],
        ylim=(0, 1.08),
    )
    style_axis(ax)

    ax = axes[2]
    thinking = summary.loc[summary["mode"].eq("thinking")].iloc[0]
    metrics = ["trace_exact", "trace_ordered_marker_accuracy", "trace_marker_count_accuracy"]
    values = [float(thinking[item]) for item in metrics]
    names = ["Exact trace", "Ordered marker", "Marker count"]
    bars = ax.bar(range(3), values, color=[BLUE, GREEN, PURPLE], width=0.62)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{100 * value:.1f}%", ha="center")
    ax.set(
        title="C · Thinking trace diagnostics",
        ylabel="Rate",
        xticks=range(3),
        xticklabels=names,
        ylim=(0, 1.06),
    )
    style_axis(ax)
    fig.suptitle("v58 behavior: uniform Thinking advantage on a disjoint test cohort", y=1.04)
    fig.tight_layout()
    savefig(fig, path)


def plot_geometry(selected: pd.DataFrame, path: Path) -> None:
    order = [
        ("nonthinking", "nonthinking_prompt_occurrence", "NT running"),
        ("thinking", "thinking_item_end", "T running"),
        ("nonthinking", "nonthinking_answer_query", "NT final"),
        ("thinking", "thinking_answer_query", "T final"),
    ]
    rows = []
    for mode, endpoint, label in order:
        match = selected.loc[
            selected["comparison_mode"].eq(mode) & selected["endpoint"].eq(endpoint)
        ]
        if len(match) != 1:
            raise RuntimeError(f"geometry row mismatch for {mode}/{endpoint}: {len(match)}")
        row = match.iloc[0].copy()
        row["display"] = label
        rows.append(row)
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.25))
    ax = axes[0]
    log_values = frame["confirmation_logistic_balanced_accuracy"].to_numpy(float)
    ncc_values = frame["confirmation_ncc_balanced_accuracy"].to_numpy(float)
    ax.bar(x - width / 2, log_values, width, color=BLUE, label="L2 Logistic")
    ax.bar(x + width / 2, ncc_values, width, color=GREEN, hatch="//", alpha=0.82, label="NCC")
    ax.axhline(0.1, color=RED, linestyle=":", linewidth=1.4, label="10-class chance")
    for index, row in frame.reset_index(drop=True).iterrows():
        height = max(float(row["confirmation_logistic_balanced_accuracy"]), float(row["confirmation_ncc_balanced_accuracy"]))
        ax.text(index, height + 0.045, f"L{int(row['layer'])}", ha="center", fontweight="bold")
    ax.set(
        title="A · Frozen held-out decodability",
        ylabel="Balanced accuracy",
        xticks=x,
        xticklabels=frame["display"],
        ylim=(0, 1.10),
    )
    style_axis(ax)
    ax.legend(loc="upper left")

    ax = axes[1]
    rsa = frame["confirmation_ordinal_rsa"].to_numpy(float)
    bars = ax.bar(x, rsa, color=[ORANGE, PURPLE, ORANGE, PURPLE], alpha=0.88)
    for bar, value in zip(bars, rsa, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    ax.axhline(0, color="#222", linewidth=1)
    ax.set(
        title="B · Held-out ordinal RSA",
        ylabel="Spearman correlation with count gap",
        xticks=x,
        xticklabels=frame["display"],
        ylim=(-0.08, 1.02),
    )
    style_axis(ax)
    fig.suptitle("Clean geometry: running states and answer-query states are separate endpoints", y=1.04)
    fig.tight_layout()
    savefig(fig, path)


def _head_matrix(frame: pd.DataFrame, value: str = "score") -> np.ndarray:
    matrix = np.full((4, 8), np.nan)
    for row in frame.itertuples(index=False):
        matrix[int(row.layer) - 1, int(row.head)] = float(getattr(row, value))
    if np.isnan(matrix).any():
        raise RuntimeError("head matrix is incomplete")
    return matrix


def plot_role_specialization(dynamics: pd.DataFrame, path: Path) -> None:
    final = dynamics.loc[dynamics["step"].eq(dynamics["step"].max())]
    broad = final.loc[final["role"].eq("nonthinking_broad")]
    targeted = final.loc[final["role"].eq("targeted_retrieval")]
    role_names = ["thinking_broad", "marker_successor", "targeted_retrieval"]

    fig = plt.figure(figsize=(15.2, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], hspace=0.34, wspace=0.24)
    for slot, frame, title, cmap in (
        (grid[0, 0], broad, "A · Non-thinking broad score", "YlOrBr"),
        (grid[0, 1], targeted, "B · Thinking targeted mass", "PuBuGn"),
    ):
        ax = fig.add_subplot(slot)
        matrix = _head_matrix(frame)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0)
        ax.set(title=title, xlabel="Head", ylabel="Layer", xticks=range(8), yticks=range(4), yticklabels=["L1", "L2", "L3", "L4"])
        for layer in range(4):
            for head in range(8):
                color = "white" if matrix[layer, head] > 0.62 * float(np.nanmax(matrix)) else "#18212B"
                ax.text(head, layer, f"{matrix[layer, head]:.3f}", ha="center", va="center", fontsize=7.3, color=color)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Role score")

    ax = fig.add_subplot(grid[1, :])
    normalized_rows = []
    for role in role_names:
        line = final.loc[final["role"].eq(role)].sort_values(["layer", "head"])
        values = line["score"].to_numpy(float)
        normalized_rows.append(values / max(float(values.max()), 1e-12))
    matrix = np.vstack(normalized_rows)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    labels = [f"L{layer}H{head}" for layer in range(1, 5) for head in range(8)]
    ax.set(
        title="C · Within-Thinking role fingerprint (row-normalized)",
        xlabel="Attention head",
        yticks=range(3),
        yticklabels=["Broad", "Marker successor", "Targeted retrieval"],
        xticks=range(32),
        xticklabels=labels,
    )
    ax.tick_params(axis="x", labelrotation=75, labelsize=7.2)
    fig.colorbar(image, ax=ax, fraction=0.018, pad=0.015, label="Score / row maximum")
    fig.suptitle("Final-checkpoint role specialization and head-bank differentiation", y=0.99)
    savefig(fig, path)


def _arm_curve(
    frame: pd.DataFrame,
    *,
    mode: str,
    scope: str,
    metric: str,
    endpoint: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subset = frame.loc[
        frame["comparison_mode"].eq(mode) & frame["scope"].eq(scope)
    ].copy()
    if endpoint is not None:
        subset = subset.loc[subset["endpoint"].eq(endpoint)]
    clean = subset.loc[subset["path_kind"].eq("clean")]
    if len(clean) != 1:
        raise RuntimeError(f"clean arm mismatch: {mode}/{scope}/{metric}/{endpoint}")
    ks = np.array([0, 1, 2, 4], dtype=int)
    ranked = [float(clean.iloc[0][metric])]
    control_mid = [np.nan]
    control_lo = [np.nan]
    control_hi = [np.nan]
    for k in ks[1:]:
        rank = subset.loc[subset["path_kind"].eq("ranked") & subset["top_k"].eq(k)]
        controls = subset.loc[
            subset["path_kind"].eq("layer_matched_control") & subset["top_k"].eq(k), metric
        ].dropna()
        if len(rank) != 1 or controls.empty:
            raise RuntimeError(f"arm mismatch: {mode}/{scope}/K{k}/{metric}/{endpoint}")
        ranked.append(float(rank.iloc[0][metric]))
        control_mid.append(float(controls.median()))
        control_lo.append(float(controls.min()))
        control_hi.append(float(controls.max()))
    return ks, np.asarray(ranked), np.asarray(control_mid), np.asarray(control_lo), np.asarray(control_hi)


def _plot_ranked_control(
    ax: plt.Axes,
    curve: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    title: str,
    ylabel: str,
    color: str,
    ylim: tuple[float, float] = (0, 1.05),
) -> None:
    ks, ranked, control, low, high = curve
    ax.plot(ks, ranked, marker="o", color=color, linewidth=2.2, label="Ranked bank")
    valid = ~np.isnan(control)
    ax.plot(ks[valid], control[valid], marker="s", color=GREY, linestyle="--", label="Control median")
    ax.fill_between(ks[valid], low[valid], high[valid], color=LIGHT_GREY, alpha=0.65, label="Control range")
    ax.set(title=title, xlabel="Cumulative Top-K removed", ylabel=ylabel, xticks=ks, ylim=ylim)
    style_axis(ax)


def plot_topk(
    tf_behavior: pd.DataFrame,
    tf_ncc: pd.DataFrame,
    targeted_free_running: pd.DataFrame,
    broad_free_running: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2))
    _plot_ranked_control(
        axes[0, 0],
        _arm_curve(tf_behavior, mode="thinking", scope="role_query_local", metric="teacher_forced_trace_accuracy"),
        title="A · Teacher-forced trace-token accuracy",
        ylabel="Accuracy",
        color=PURPLE,
    )
    _plot_ranked_control(
        axes[0, 1],
        _arm_curve(
            targeted_free_running,
            mode="thinking",
            scope="confirmation_role_query_local_free_running",
            metric="trace_exact",
        ),
        title="B · Free-running exact trace",
        ylabel="Exact rate",
        color=PURPLE,
    )
    _plot_ranked_control(
        axes[0, 2],
        _arm_curve(
            targeted_free_running,
            mode="thinking",
            scope="confirmation_role_query_local_free_running",
            metric="ar_final_accuracy",
        ),
        title="C · Thinking free-running final answer",
        ylabel="Exact accuracy",
        color=PURPLE,
    )
    _plot_ranked_control(
        axes[1, 0],
        _arm_curve(
            tf_ncc,
            mode="thinking",
            scope="role_query_local",
            endpoint="thinking_item_end",
            metric="confirmation_ncc_balanced_accuracy",
        ),
        title="D · Post-ablation running NCC",
        ylabel="Frozen NCC balanced accuracy",
        color=BLUE,
        ylim=(0, 0.36),
    )
    axes[1, 0].axhline(0.1, color=RED, linestyle=":", linewidth=1.2)
    _plot_ranked_control(
        axes[1, 1],
        _arm_curve(
            tf_ncc,
            mode="thinking",
            scope="role_query_local",
            endpoint="thinking_answer_query",
            metric="confirmation_ncc_balanced_accuracy",
        ),
        title="E · Post-ablation final NCC",
        ylabel="Frozen NCC balanced accuracy",
        color=GREEN,
    )
    _plot_ranked_control(
        axes[1, 2],
        _arm_curve(
            broad_free_running,
            mode="nonthinking",
            scope="confirmation_role_query_local_free_running",
            metric="ar_final_accuracy",
        ),
        title="F · Non-thinking broad-bank test",
        ylabel="Exact accuracy",
        color=ORANGE,
        ylim=(0, 0.42),
    )
    for ax in axes.flat:
        ax.legend(loc="best", fontsize=7.5)
    fig.suptitle("Ranked Top-K necessity and clean-frozen post-ablation representation", y=1.01)
    fig.tight_layout()
    savefig(fig, path)


def _paired_damage_ci(
    detail: pd.DataFrame,
    arm: str,
    metric: str,
    *,
    bootstraps: int = 10_000,
) -> tuple[float, float, float]:
    clean = detail.loc[detail["arm"].eq("clean"), ["row_id", metric]].set_index("row_id")
    damaged = detail.loc[detail["arm"].eq(arm), ["row_id", metric]].set_index("row_id")
    paired = clean.join(damaged, lsuffix="_clean", rsuffix="_damaged", how="inner")
    if len(paired) != len(clean) or len(paired) != len(damaged):
        raise RuntimeError(f"unpaired successor rows: {arm}/{metric}")
    differences = (
        paired[f"{metric}_clean"].to_numpy(dtype=float)
        - paired[f"{metric}_damaged"].to_numpy(dtype=float)
    )
    rng = np.random.default_rng(20260901)
    indices = rng.integers(0, len(differences), size=(bootstraps, len(differences)))
    samples = differences[indices].mean(axis=1)
    return float(differences.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def plot_role_causal_separation(
    successor_summary: pd.DataFrame,
    successor_detail: pd.DataFrame,
    path: Path,
) -> None:
    arms = [
        "clean",
        "targeted_top2",
        "successor_top1",
        "successor_same_layer_low_score",
        "targeted_top2_plus_successor_top1",
    ]
    labels = ["Clean", "Targeted T2", "Successor T1", "L2 low-score\ncontrol", "Joint"]
    colors = [GREY, PURPLE, ORANGE, BLUE, RED]
    metrics = [
        ("trace_ordered_marker_accuracy", "trace_ordered_marker_accuracy", "Ordered marker"),
        ("trace_marker_count_accuracy", "trace_marker_count_accuracy", "Marker count"),
        ("ar_final_accuracy", "ar_accuracy", "Final count"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.5))
    summary = successor_summary.set_index("arm").loc[arms]

    ax = axes[0, 0]
    x = np.arange(len(arms))
    width = 0.23
    for index, (summary_metric, _detail_metric, label) in enumerate(metrics):
        values = summary[summary_metric].to_numpy(dtype=float)
        ax.bar(x + (index - 1) * width, values, width, label=label)
    ax.set(
        title="A · Clean and intervened behavior",
        xlabel="Intervention arm",
        ylabel="Exact / token-weighted rate",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1.06),
    )
    ax.legend(loc="lower left")
    style_axis(ax)

    ax = axes[0, 1]
    intervention_arms = arms[1:]
    intervention_labels = labels[1:]
    x = np.arange(len(intervention_arms))
    for index, (_summary_metric, detail_metric, label) in enumerate(metrics):
        estimates = []
        lower = []
        upper = []
        for arm in intervention_arms:
            estimate, low, high = _paired_damage_ci(successor_detail, arm, detail_metric)
            estimates.append(estimate)
            lower.append(estimate - low)
            upper.append(high - estimate)
        ax.bar(
            x + (index - 1) * width,
            estimates,
            width,
            yerr=np.vstack([lower, upper]),
            capsize=2.5,
            label=label,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set(
        title="B · Paired damage from clean (bootstrap 95% CI)",
        xlabel="Intervention arm",
        ylabel="Clean − intervened rate",
        xticks=x,
        xticklabels=intervention_labels,
        ylim=(-0.05, 0.62),
    )
    ax.legend(loc="upper left")
    style_axis(ax)

    ax = axes[1, 0]
    length_match = []
    for arm in arms:
        rows = successor_detail.loc[successor_detail["arm"].eq(arm)].copy()
        valid = rows["ar_pred_count"].notna()
        length_match.append(
            float(
                (
                    rows.loc[valid, "ar_pred_count"].astype(float)
                    == rows.loc[valid, "trace_generated_marker_count"].astype(float)
                ).mean()
            )
        )
    bars = ax.bar(np.arange(len(arms)), length_match, color=colors)
    for bar, value in zip(bars, length_match, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{100*value:.1f}%", ha="center", fontsize=8)
    ax.set(
        title="C · Final answer follows generated trace length",
        xlabel="Intervention arm",
        ylabel="P(answer = generated marker count | answered)",
        xticks=np.arange(len(arms)),
        xticklabels=labels,
        ylim=(0.9, 1.012),
    )
    style_axis(ax)

    ax = axes[1, 1]
    clean_by_count = (
        successor_detail.loc[successor_detail["arm"].eq("clean")]
        .groupby("count")["ar_accuracy"]
        .mean()
    )
    line_arms = [
        "targeted_top2",
        "successor_top1",
        "successor_same_layer_low_score",
        "targeted_top2_plus_successor_top1",
    ]
    line_labels = ["Targeted T2", "Successor T1", "L2 low-score control", "Joint"]
    line_colors = [PURPLE, ORANGE, BLUE, RED]
    for arm, label, color in zip(line_arms, line_labels, line_colors, strict=True):
        arm_by_count = successor_detail.loc[successor_detail["arm"].eq(arm)].groupby("count")["ar_accuracy"].mean()
        damage = clean_by_count - arm_by_count
        ax.plot(damage.index, damage.values, marker="o", linewidth=2, color=color, label=label)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set(
        title="D · Final-count damage by gold count",
        xlabel="Gold count",
        ylabel="Clean − intervened accuracy",
        xticks=np.arange(1, 11),
        ylim=(-0.08, 0.96),
    )
    ax.legend(loc="upper left")
    style_axis(ax)

    fig.suptitle("Causal role separation: targeted content vs successor/cardinality", y=1.01)
    fig.tight_layout()
    savefig(fig, path)


def geometry_projection_widget(cloud: pd.DataFrame) -> str:
    """Build the v20-style 2x2 endpoint comparison in both 2D and 3D.

    The preferred input is the discovery-fitted, confirmation-only export with
    an explicit ``endpoint`` column.  The older phase-transition cloud is
    accepted as a transparent fallback; it lacks the Non-thinking running
    endpoint, which is rendered as unavailable rather than fabricated.
    """

    if "endpoint" in cloud.columns:
        columns = ["endpoint", "layer", "sample", "k", "pc1", "pc2", "pc3"]
        columns.extend(
            column
            for column in ("pc1_variance_ratio", "pc2_variance_ratio", "pc3_variance_ratio")
            if column in cloud.columns
        )
        payload = cloud.loc[:, columns].copy()
        source_label = "discovery-fitted PCA; held-out confirmation states"
    else:
        endpoint_map = {
            ("nonthinking", "final_answer"): "nonthinking_answer_query",
            ("thinking", "trace_marker"): "thinking_item_end",
            ("thinking", "final_answer"): "thinking_answer_query",
        }
        final_step = int(cloud["step"].max())
        payload = cloud.loc[cloud["step"].eq(final_step)].copy()
        payload["endpoint"] = [
            endpoint_map.get((str(mode), str(site)), "")
            for mode, site in zip(payload["mode"], payload["site"], strict=True)
        ]
        payload = payload.loc[payload["endpoint"].ne(""), ["endpoint", "layer", "sample", "k", "pc1", "pc2", "pc3"]]
        source_label = (
            "phase-transition milestone fallback; Non-thinking running cloud "
            "was not archived"
        )
    for column in ("pc1", "pc2", "pc3"):
        payload[column] = payload[column].astype(float).round(6)
    for column in ("layer", "sample", "k"):
        payload[column] = payload[column].astype(int)
    records = json.dumps(payload.to_dict(orient="records"), ensure_ascii=False, separators=(",", ":"))
    plotly_runtime = "\n".join(line.rstrip() for line in get_plotlyjs().splitlines())
    layer_options = (
        '<option value="0">L0 · embedding output</option>'
        '<option value="1">L1</option><option value="2" selected>L2</option>'
        '<option value="3">L3</option><option value="4">L4</option>'
    )
    return (
        '<div class="geometry-widget">'
        '<div class="geometry-controls">'
        f'<span class="geometry-source">{html.escape(source_label)}</span>'
        '</div>'
        '<h4 class="geometry-view-title"><span>2D comparison · PC1–PC2</span>'
        f'<label class="geometry-layer-control">2D layer <select id="geometry-layer-2d">{layer_options}</select></label></h4>'
        '<div id="geometry-2d" class="geometry-panel geometry-panel-2d" aria-label="Layer-selectable 2D PCA geometry comparison"></div>'
        '<h4 class="geometry-view-title"><span>3D comparison · PC1–PC3</span>'
        f'<label class="geometry-layer-control">3D layer <select id="geometry-layer-3d">{layer_options}</select></label>'
        '<span class="geometry-interaction-hint">拖拽旋转 · 滚轮缩放 · 双击复位</span></h4>'
        '<div id="geometry-3d" class="geometry-panel geometry-panel-3d" aria-label="Layer-selectable 3D PCA geometry comparison"></div>'
        '</div>'
        f'<script>{plotly_runtime}</script>'
        f'<script>const V58_GEOMETRY_DATA={records};</script>'
        + r"""
<script>
(function(){
  const layer2d = document.getElementById('geometry-layer-2d');
  const layer3d = document.getElementById('geometry-layer-3d');
  const specs=[
    {endpoint:'nonthinking_prompt_occurrence',title:'Non-thinking · running index',row:0,col:0},
    {endpoint:'thinking_item_end',title:'Thinking · running index',row:0,col:1},
    {endpoint:'nonthinking_answer_query',title:'Non-thinking · final count',row:1,col:0},
    {endpoint:'thinking_answer_query',title:'Thinking · final count',row:1,col:1}
  ];
  function endpointRows(endpoint,layerValue){return V58_GEOMETRY_DATA.filter(d=>d.endpoint===endpoint&&d.layer===layerValue);}
  function summarize(rows){
    const grouped=new Map();
    for(const row of rows){if(!grouped.has(row.k))grouped.set(row.k,[]);grouped.get(row.k).push(row);}
    const ks=[...grouped.keys()].sort((a,b)=>a-b);
    return ks.map(k=>{const g=grouped.get(k);return{k:k,x:g.reduce((s,d)=>s+d.pc1,0)/g.length,y:g.reduce((s,d)=>s+d.pc2,0)/g.length,z:g.reduce((s,d)=>s+d.pc3,0)/g.length};});
  }
  function axisTitle(rows,index,includeVariance=true){
    const key=`pc${index}_variance_ratio`,value=rows.length?Number(rows[0][key]):NaN;
    return includeVariance&&Number.isFinite(value)?`PC${index} (${(100*value).toFixed(1)}% discovery variance)`:`PC${index}`;
  }
  function axisName(prefix,index){return index===0?prefix:prefix+(index+1);}
  function renderGeometry(){
    const traces2d=[],traces3d=[],annotations2d=[],annotations3d=[];
    const selected2d=Number(layer2d.value),selected3d=Number(layer3d.value);
    const layout2d={title:{text:`Four aligned endpoints · L${selected2d}`,font:{size:16}},margin:{l:64,r:40,t:86,b:55},paper_bgcolor:'#fff',plot_bgcolor:'#fff',showlegend:true,legend:{orientation:'h',y:1.08,x:0}};
    const layout3d={title:{text:`Four aligned endpoints · L${selected3d}`,font:{size:16}},margin:{l:0,r:0,t:86,b:0},paper_bgcolor:'#fff',showlegend:true,dragmode:'orbit',legend:{orientation:'h',y:1.08,x:0}};
    const xdomains=[[0,.46],[.54,1],[0,.46],[.54,1]],ydomains=[[.56,1],[.56,1],[0,.44],[0,.44]];
    specs.forEach((spec,index)=>{
      const rows2d=endpointRows(spec.endpoint,selected2d),centers2d=summarize(rows2d);
      const rows3d=endpointRows(spec.endpoint,selected3d),centers3d=summarize(rows3d);
      const hover2d=rows2d.map(d=>`count/progress k=${d.k}<br>sample=${d.sample}<br>endpoint=${spec.title}<br>layer=L${selected2d}`);
      const hover3d=rows3d.map(d=>`count/progress k=${d.k}<br>sample=${d.sample}<br>endpoint=${spec.title}<br>layer=L${selected3d}`);
      const xref=axisName('x',index),yref=axisName('y',index),scene=index===0?'scene':'scene'+(index+1);
      if(rows2d.length){
        traces2d.push({type:'scattergl',mode:'markers',name:'held-out states',legendgroup:'states',showlegend:index===0,xaxis:xref,yaxis:yref,x:rows2d.map(d=>d.pc1),y:rows2d.map(d=>d.pc2),text:hover2d,hoverinfo:'text+x+y',marker:{size:6,opacity:.46,color:rows2d.map(d=>d.k),cmin:1,cmax:10,colorscale:'Viridis',showscale:index===3,colorbar:index===3?{title:'k',len:.40,y:.22}:undefined}});
        traces2d.push({type:'scatter',mode:'lines+markers+text',name:'class centroids',legendgroup:'centroids',showlegend:index===0,xaxis:xref,yaxis:yref,x:centers2d.map(d=>d.x),y:centers2d.map(d=>d.y),text:centers2d.map(d=>String(d.k)),textposition:'top center',line:{color:'#c64e4e',width:2},marker:{size:8,color:centers2d.map(d=>d.k),cmin:1,cmax:10,colorscale:'Viridis',line:{color:'#17202a',width:1},showscale:false}});
      }else{
        annotations2d.push({xref:'paper',yref:'paper',x:(xdomains[index][0]+xdomains[index][1])/2,y:(ydomains[index][0]+ydomains[index][1])/2,text:'Not archived<br>requires final-checkpoint re-export',showarrow:false,font:{color:'#b94444',size:14}});
      }
      if(rows3d.length){
        traces3d.push({type:'scatter3d',mode:'markers',name:'held-out states',legendgroup:'states3d',showlegend:index===0,scene:scene,x:rows3d.map(d=>d.pc1),y:rows3d.map(d=>d.pc2),z:rows3d.map(d=>d.pc3),text:hover3d,hoverinfo:'text+x+y+z',marker:{size:3.2,opacity:.46,color:rows3d.map(d=>d.k),cmin:1,cmax:10,colorscale:'Viridis',showscale:index===3,colorbar:index===3?{title:'k',len:.40,y:.22}:undefined}});
        traces3d.push({type:'scatter3d',mode:'lines+markers+text',name:'class centroids',legendgroup:'centroids3d',showlegend:index===0,scene:scene,x:centers3d.map(d=>d.x),y:centers3d.map(d=>d.y),z:centers3d.map(d=>d.z),text:centers3d.map(d=>String(d.k)),textposition:'top center',line:{color:'#c64e4e',width:5},marker:{size:4.5,color:centers3d.map(d=>d.k),cmin:1,cmax:10,colorscale:'Viridis',line:{color:'#17202a',width:1},showscale:false}});
      }else{
        annotations3d.push({xref:'paper',yref:'paper',x:(xdomains[index][0]+xdomains[index][1])/2,y:(ydomains[index][0]+ydomains[index][1])/2,text:'Not archived<br>requires final-checkpoint re-export',showarrow:false,font:{color:'#b94444',size:14}});
      }
      const xlayout=index===0?'xaxis':'xaxis'+(index+1),ylayout=index===0?'yaxis':'yaxis'+(index+1);
      layout2d[xlayout]={domain:xdomains[index],anchor:yref,title:{text:axisTitle(rows2d,1)},zeroline:false};
      layout2d[ylayout]={domain:ydomains[index],anchor:xref,title:{text:axisTitle(rows2d,2)},zeroline:false};
      layout3d[scene]={domain:{x:xdomains[index],y:ydomains[index]},xaxis:{title:{text:axisTitle(rows3d,1,false)}},yaxis:{title:{text:axisTitle(rows3d,2,false)}},zaxis:{title:{text:axisTitle(rows3d,3,false)}},aspectmode:'data'};
      const titleY=spec.row===0?1.025:.465;
      annotations2d.push({xref:'paper',yref:'paper',x:(xdomains[index][0]+xdomains[index][1])/2,y:titleY,text:`<b>${spec.title}</b>`,showarrow:false,font:{size:13}});
      annotations3d.push({xref:'paper',yref:'paper',x:(xdomains[index][0]+xdomains[index][1])/2,y:titleY,text:`<b>${spec.title}</b>`,showarrow:false,font:{size:13}});
    });
    layout2d.annotations=annotations2d;layout3d.annotations=annotations3d;
    Plotly.react('geometry-2d',traces2d,layout2d,{responsive:true,displaylogo:false});
    Plotly.react('geometry-3d',traces3d,layout3d,{responsive:true,displaylogo:false,scrollZoom:true});
  }
  layer2d.addEventListener('change',renderGeometry);
  layer3d.addEventListener('change',renderGeometry);
  renderGeometry();
})();
</script>
"""
    )


def plot_role_differentiation_heatmaps(
    attention: pd.DataFrame,
    broad_rankings: pd.DataFrame,
    thinking_rankings: pd.DataFrame,
    path: Path,
) -> dict[str, float]:
    ranking_by_role = {
        "nonthinking_broad": broad_rankings,
        "marker_successor": thinking_rankings,
        "targeted_retrieval": thinking_rankings,
    }
    specs = [
        ("nonthinking_broad", "Non-thinking broad retrieval"),
        ("marker_successor", "Thinking marker successor"),
        ("targeted_retrieval", "Thinking targeted retrieval"),
    ]
    steps = np.asarray(sorted(attention["step"].astype(int).unique()), dtype=float)
    head_labels = [f"L{layer}H{head}" for layer in range(1, 5) for head in range(8)]
    fig, axes = plt.subplots(3, 2, figsize=(15.2, 12.2), sharey=True)
    statistics: dict[str, float] = {}
    matrices: dict[str, np.ndarray] = {}
    frozen_by_role: dict[str, list[tuple[int, int]]] = {}
    for role, _title in specs:
        role_rows = attention.loc[attention["role"].eq(role)].copy()
        role_rankings = ranking_by_role[role].loc[ranking_by_role[role]["role"].eq(role)].sort_values("rank")
        first = role_rankings.iloc[0]
        second = role_rankings.iloc[1]
        frozen_by_role[role] = [
            (int(first["layer"]), int(first["head"])),
            (int(second["layer"]), int(second["head"])),
        ]
        matrix_columns = []
        for step in steps:
            snapshot = role_rows.loc[role_rows["step"].eq(int(step))].sort_values(["layer", "head"])
            values = snapshot["score"].to_numpy(float)
            if len(values) != 32:
                raise RuntimeError(f"{role} step {int(step)} has {len(values)} heads")
            matrix_columns.append(values / max(float(values.sum()), 1e-12))
        matrices[role] = np.column_stack(matrix_columns)
        for rank_row, label_prefix in ((first, "rank1"), (second, "rank2")):
            line = role_rows.loc[
                role_rows["layer"].eq(int(rank_row["layer"]))
                & role_rows["head"].eq(int(rank_row["head"]))
            ].sort_values("step")
            if label_prefix == "rank1":
                final_score = float(line.iloc[-1]["score"])
                statistics[f"{role}_rank1_final_score"] = final_score
                statistics[f"{role}_rank1_first_50pct_step"] = float(
                    line.loc[line["score"].ge(0.5 * final_score), "step"].iloc[0]
                )
                statistics[f"{role}_rank1_first_80pct_step"] = float(
                    line.loc[line["score"].ge(0.8 * final_score), "step"].iloc[0]
                )

    vmax = max(float(matrix.max()) for matrix in matrices.values())
    meshes = []
    for row, (role, title) in enumerate(specs):
        matrix = matrices[role]
        frozen_labels = ", ".join(f"L{layer}H{head}" for layer, head in frozen_by_role[role])
        for column, scale in enumerate(("linear", "log")):
            ax = axes[row, column]
            if scale == "linear":
                centers = steps
                edges = np.concatenate(([-50.0], (centers[:-1] + centers[1:]) / 2, [10050.0]))
                switch = 1500.0
                ax.set_xlim(-50, 10050)
                ax.set_xticks([0, 1500, 3000, 5000, 7000, 10000], ["0", "1.5k", "3k", "5k", "7k", "10k"])
                scale_title = "linear steps"
            else:
                centers = steps + 100.0
                inner = np.sqrt(centers[:-1] * centers[1:])
                edges = np.concatenate(([centers[0] ** 2 / inner[0]], inner, [centers[-1] ** 2 / inner[-1]]))
                switch = 1600.0
                ax.set_xscale("log")
                ax.set_xlim(edges[0], edges[-1])
                ticks = np.asarray([100, 200, 600, 1600, 3100, 5100, 10100], dtype=float)
                ax.set_xticks(ticks, ["0", "100", "500", "1.5k", "3k", "5k", "10k"])
                ax.minorticks_off()
                scale_title = "log(step + 100)"
            y_edges = np.arange(33) - 0.5
            mesh = ax.pcolormesh(edges, y_edges, matrix, cmap="magma", vmin=0.0, vmax=vmax, shading="flat")
            meshes.append(mesh)
            ax.axvline(switch, color="white", linestyle=":", linewidth=1.1)
            for boundary in (7.5, 15.5, 23.5):
                ax.axhline(boundary, color="white", linewidth=0.7, alpha=0.72)
            for rank, (head_layer, head) in enumerate(frozen_by_role[role], start=1):
                y = (head_layer - 1) * 8 + head
                ax.axhline(y, color="#56D7FF", linewidth=0.85, linestyle="-" if rank == 1 else "--", alpha=0.95)
            ax.set_title(f"{title} · {scale_title}\nfrozen rank-1/2: {frozen_labels}", fontsize=10.2)
            ax.set_xlabel("optimizer step" if scale == "linear" else "optimizer step + 100 (log scale)")
            ax.set_yticks(np.arange(32), head_labels, fontsize=6.8)
            ax.set_ylabel("attention head")
    fig.suptitle("Head-bank differentiation over training: identical role-share heatmaps on linear and log axes", y=1.005)
    fig.text(0.5, 0.982, "Each column within a panel sums to 1 across the same 32 heads; cyan lines mark frozen discovery rank-1/rank-2 heads", ha="center", va="top", fontsize=8.8, color="#444")
    fig.tight_layout(rect=(0, 0, 0.94, 0.965), h_pad=1.7, w_pad=1.2)
    colorbar_axis = fig.add_axes((0.955, 0.18, 0.012, 0.66))
    colorbar = fig.colorbar(meshes[-1], cax=colorbar_axis)
    colorbar.set_label("share of total role score")
    savefig(fig, path)
    return statistics


def plot_geometry_emergence(
    geometry_dynamics: pd.DataFrame,
    selected_layers: dict,
    path: Path,
) -> dict[str, float]:
    endpoints = [
        (
            "nonthinking",
            "final_answer",
            int(selected_layers["nonthinking"]["nonthinking_answer_query"]),
            "Non-thinking final",
            ORANGE,
        ),
        (
            "thinking",
            "trace_marker",
            int(selected_layers["thinking"]["thinking_item_end"]),
            "Thinking running",
            BLUE,
        ),
        (
            "thinking",
            "final_answer",
            int(selected_layers["thinking"]["thinking_answer_query"]),
            "Thinking final",
            PURPLE,
        ),
    ]
    metrics = [
        ("centroid_pc1_to_pc3_variance_fraction", "A · Variance captured by centroid PC1–PC3", "Variance fraction"),
        ("centroid_effective_dimension", "B · Centroid effective dimension", "Effective dimension"),
        ("path_straightness_chord_over_arc", "C · Count/progress path straightness", "Chord / arc length"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))
    statistics: dict[str, float] = {}
    for ax, (metric, title, ylabel) in zip(axes, metrics, strict=True):
        for mode, site, selected_layer, label, color in endpoints:
            line = geometry_dynamics.loc[
                geometry_dynamics["mode"].eq(mode)
                & geometry_dynamics["site"].eq(site)
                & geometry_dynamics["layer"].eq(selected_layer)
            ].sort_values("step")
            display_label = f"{label} · L{selected_layer}"
            ax.plot(line["step"], line[metric], color=color, linewidth=2.1, marker="o", markersize=3.4, label=display_label)
            key = label.lower().replace("-", "").replace(" ", "_")
            statistics[f"{key}_{metric}_final"] = float(line.iloc[-1][metric])
        ax.axvline(1500, color=RED, linestyle=":", linewidth=1.1)
        ax.set(title=title, xlabel="Training step", ylabel=ylabel)
        ax.ticklabel_format(style="plain", axis="x")
        style_axis(ax)
        ax.legend(loc="best", fontsize=7.5)
    fig.suptitle("Representation geometry over training (fixed final-selected physical layers)", y=1.02)
    fig.tight_layout()
    savefig(fig, path)
    return statistics


def plot_sufficiency(answer: pd.DataFrame, progress: pd.DataFrame, path: Path) -> None:
    answer = answer.loc[answer["is_discovery_selected_layer"].eq(1.0)].copy()
    answer_summary = answer.groupby(["mode", "condition"], as_index=False).agg(
        donor_adoption=("donor_adoption", "mean"),
        receiver_retention=("receiver_retention", "mean"),
        donor_margin_shift=("donor_margin_shift", "mean"),
        observations=("donor_adoption", "size"),
    )
    progress_summary = progress.groupby("condition", as_index=False).agg(
        donor_first_adoption=("donor_first_adoption", "mean"),
        natural_first_retention=("natural_first_retention", "mean"),
        generated_final_correct=("generated_final_correct", "mean"),
        observations=("donor_first_adoption", "size"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    conditions = ["adjacent_count_donor", "same_count_context_control"]
    x = np.arange(2)
    width = 0.34
    for index, mode in enumerate(("nonthinking", "thinking")):
        values = []
        for condition in conditions:
            row = answer_summary.loc[
                answer_summary["mode"].eq(mode) & answer_summary["condition"].eq(condition)
            ]
            values.append(float(row.iloc[0]["donor_adoption"]))
        bars = ax.bar(x + (index - 0.5) * width, values, width, color=ORANGE if mode == "nonthinking" else PURPLE, label="Non-thinking" if mode == "nonthinking" else "Thinking")
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{100 * value:.1f}%", ha="center", fontsize=8.5)
    ax.set(
        title="A · Answer-query full-state transplant",
        ylabel="Greedy donor-count adoption",
        xticks=x,
        xticklabels=["Adjacent-count donor", "Same-count context control"],
        ylim=(0, 1.08),
    )
    style_axis(ax)
    ax.legend()

    ax = axes[1]
    order = [
        "clean",
        "centroid_shift",
        "orthogonal_control",
        "natural_marker_cross_position",
        "natural_item_span_cross_position",
    ]
    labels = ["Clean", "Centroid shift", "Orthogonal", "Natural marker†", "Natural item†"]
    values = [
        float(progress_summary.loc[progress_summary["condition"].eq(condition), "donor_first_adoption"].iloc[0])
        for condition in order
    ]
    colors = [GREY, BLUE, LIGHT_GREY, ORANGE, ORANGE]
    bars = ax.bar(range(len(order)), values, color=colors)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{100 * value:.1f}%", ha="center", fontsize=8)
    ax.set(
        title="B · Free-running progress-state sufficiency",
        ylabel="Donor next-marker adoption",
        xticks=range(len(order)),
        xticklabels=labels,
        ylim=(0, 0.58),
    )
    ax.tick_params(axis="x", labelrotation=18)
    style_axis(ax)
    fig.suptitle("Terminal answer state is executable; same-position progress state remains weak", y=1.03)
    fig.tight_layout()
    savefig(fig, path)


def _weighted_routing(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for step, group in frame.groupby("step"):
        weights = group["observations"].to_numpy(float)
        total = float(weights.sum())
        row: dict[str, float] = {"step": float(step), "observations": total}
        for metric in ("targeted_mass", "qk_margin", "correct_occurrence_top1"):
            row[metric] = float(np.average(group[metric].to_numpy(float), weights=weights))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step")


def _bank_dynamics(
    frame: pd.DataFrame,
    selected_heads: set[tuple[int, int]],
) -> pd.DataFrame:
    rows = []
    for step, group in frame.groupby("step"):
        group = group.copy()
        group["selected"] = [
            (int(layer), int(head)) in selected_heads
            for layer, head in zip(group["layer"], group["head"], strict=True)
        ]
        values = group["score"].to_numpy(float)
        selected = group.loc[group["selected"], "score"].to_numpy(float)
        other = group.loc[~group["selected"], "score"].to_numpy(float)
        current = set(
            (int(row.layer), int(row.head))
            for row in group.nlargest(len(selected_heads), "score").itertuples(index=False)
        )
        union = selected_heads | current
        rows.append(
            {
                "step": int(step),
                "selected_mean": float(selected.mean()),
                "other_mean": float(other.mean()),
                "bank_share": float(selected.sum() / max(values.sum(), 1e-12)),
                "jaccard": float(len(selected_heads & current) / len(union)),
            }
        )
    return pd.DataFrame(rows).sort_values("step")


def plot_training_dynamics(
    high_power: pd.DataFrame,
    routing: pd.DataFrame,
    attention: pd.DataFrame,
    rankings: pd.DataFrame,
    patching: pd.DataFrame,
    causal: pd.DataFrame,
    path: Path,
) -> dict[str, float]:
    routed = _weighted_routing(routing)
    targeted_heads = set(
        (int(row.layer), int(row.head))
        for row in rankings.loc[
            rankings["role"].eq("targeted_retrieval") & rankings["rank"].le(4)
        ].itertuples(index=False)
    )
    broad_heads = set(
        (int(row.layer), int(row.head))
        for row in rankings.loc[
            rankings["role"].eq("nonthinking_broad") & rankings["rank"].le(4)
        ].itertuples(index=False)
    )
    targeted = _bank_dynamics(
        attention.loc[attention["role"].eq("targeted_retrieval")], targeted_heads
    )
    broad = _bank_dynamics(
        attention.loc[attention["role"].eq("nonthinking_broad")], broad_heads
    )
    top2_patch = patching.loc[
        patching["intervention"].eq("value_only_at_target_source")
        & patching["top_n"].eq(2)
    ].sort_values("step")

    causal_fixed = causal.loc[causal["intervention"].eq("fixed_head_zero")].copy()
    causal_control = causal.loc[causal["intervention"].eq("same_layer_control_zero")].copy()
    causal_specific = causal_fixed.merge(
        causal_control[["step", "role", "causal_damage"]],
        on=["step", "role"],
        suffixes=("_fixed", "_control"),
    )
    causal_specific["specificity"] = (
        causal_specific["causal_damage_fixed"] - causal_specific["causal_damage_control"]
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.4), sharex=False)
    ax = axes[0, 0]
    for mode, color, label in (
        ("nonthinking", ORANGE, "Non-thinking final"),
        ("thinking", PURPLE, "Thinking final"),
    ):
        line = high_power.loc[high_power["mode"].eq(mode)].sort_values("step")
        ax.plot(line["step"], line["ar_accuracy"], marker="o", markersize=3.5, linewidth=2, color=color, label=label)
    trace = high_power.loc[high_power["mode"].eq("thinking")].sort_values("step")
    ax.plot(trace["step"], trace["trace_exact"], color=BLUE, linestyle="--", linewidth=1.7, label="Thinking trace exact")
    ax.set(title="A · High-power free-running behavior", xlabel="Training step", ylabel="Rate", ylim=(-0.03, 1.03))
    style_axis(ax)
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    ax.plot(routed["step"], routed["targeted_mass"], color=GREEN, linewidth=2.2, label="Targeted attention mass")
    ax.set(title="B · Targeted routing formation", xlabel="Training step", ylabel="Attention mass", ylim=(0, 0.48))
    style_axis(ax)
    twin = ax.twinx()
    twin.plot(routed["step"], routed["qk_margin"], color=PURPLE, linewidth=1.5, alpha=0.82, label="QK correct-vs-best-wrong")
    twin.axhline(0, color=GREY, linestyle=":", linewidth=1)
    twin.set_ylabel("Scaled QK margin")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right")

    ax = axes[0, 2]
    ax.plot(targeted["step"], targeted["bank_share"], color=PURPLE, linewidth=2.2, label="Targeted Top-4 share")
    ax.plot(broad["step"], broad["bank_share"], color=ORANGE, linestyle="--", linewidth=1.8, label="NT broad Top-4 share")
    ax.axhline(4 / 32, color=GREY, linestyle=":", linewidth=1, label="Uniform 4/32")
    ax.set(title="C · Bank concentration", xlabel="Training step", ylabel="Top-4 share of all-head score", ylim=(0, 1.0))
    style_axis(ax)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(targeted["step"], targeted["jaccard"], color=PURPLE, linewidth=2.2, label="Targeted identity")
    ax.plot(broad["step"], broad["jaccard"], color=ORANGE, linestyle="--", linewidth=1.8, label="NT broad identity")
    ax.set(title="D · Final-bank identity stabilization", xlabel="Training step", ylabel="Top-4 Jaccard vs final", ylim=(-0.03, 1.03))
    style_axis(ax)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(top2_patch["step"], top2_patch["normalized_recovery_mean"], color=GREEN, marker="o", linewidth=2, label="Top-2 value recovery")
    ax.plot(top2_patch["step"], top2_patch["patched_correct"], color=BLUE, marker="s", linewidth=1.7, label="Patched marker accuracy")
    ax.plot(top2_patch["step"], top2_patch["corrupt_correct"], color=GREY, linestyle="--", linewidth=1.5, label="Corrupt baseline accuracy")
    ax.set(title="E · Retrieval transport emerges", xlabel="Training step", ylabel="Recovery / accuracy", ylim=(-0.06, 1.03))
    style_axis(ax)
    ax.legend(loc="upper left")

    ax = axes[1, 2]
    for role, color, label in (
        ("marker_successor", BLUE, "Marker-successor head"),
        ("targeted_retrieval", PURPLE, "Targeted head"),
    ):
        line = causal_specific.loc[causal_specific["role"].eq(role)].sort_values("step")
        ax.plot(line["step"], line["specificity"], color=color, marker="o", markersize=3.5, linewidth=2, label=label)
    ax.axhline(0, color=GREY, linestyle=":", linewidth=1)
    ax.set(title="F · Position-local causal specificity", xlabel="Training step", ylabel="Damage(selected) - damage(control)")
    style_axis(ax)
    ax.legend(loc="upper left")

    for ax in axes.flat:
        ax.axvline(1500, color=RED, linestyle=":", linewidth=1.0, alpha=0.78)
        ax.ticklabel_format(style="plain", axis="x")
    fig.suptitle("Synchronized training dynamics (linear step axis; 500 free-running examples/mode/checkpoint)", y=1.01)
    fig.tight_layout()
    savefig(fig, path)

    final_mass = float(routed.iloc[-1]["targeted_mass"])
    first_50 = int(routed.loc[routed["targeted_mass"].ge(0.5 * final_mass), "step"].iloc[0])
    first_80 = int(routed.loc[routed["targeted_mass"].ge(0.8 * final_mass), "step"].iloc[0])
    return {
        "final_targeted_mass": final_mass,
        "first_targeted_50pct_step": first_50,
        "first_targeted_80pct_step": first_80,
        "final_targeted_top4_share": float(targeted.iloc[-1]["bank_share"]),
        "final_broad_top4_share": float(broad.iloc[-1]["bank_share"]),
        "final_targeted_jaccard": float(targeted.iloc[-1]["jaccard"]),
        "final_broad_jaccard": float(broad.iloc[-1]["jaccard"]),
    }


def row_value(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    result = frame.loc[mask]
    if len(result) != 1:
        raise RuntimeError(f"expected one row for {conditions}, found {len(result)}")
    return result.iloc[0]


def median_control(
    frame: pd.DataFrame,
    metric: str,
    *,
    mode: str,
    scope: str,
    top_k: int,
) -> float:
    rows = frame.loc[
        frame["comparison_mode"].eq(mode)
        & frame["scope"].eq(scope)
        & frame["path_kind"].eq("layer_matched_control")
        & frame["top_k"].eq(top_k),
        metric,
    ].dropna()
    if rows.empty:
        raise RuntimeError("missing matched controls")
    return float(rows.median())


def build_report(output: Path, *, run_root: Path | None = None,
                 analysis_root: Path | None = None, asset_dir: Path | None = None) -> dict:
    global DATA, ANALYSIS, ASSET_DIR
    DATA = Path(run_root).resolve() if run_root is not None else DATA
    ANALYSIS = Path(analysis_root).resolve() if analysis_root is not None else DATA / "analysis"
    ASSET_DIR = Path(asset_dir).resolve() if asset_dir is not None else output.parent / "NiaH_Synthetic_report_assets"
    setup_style()
    cfg = read_json(DATA / "config.json")
    run_manifest = read_json(DATA / "manifest.json")
    confirmation = read_csv(ANALYSIS / "behavior_confirmation_v58" / "autoregressive_summary.csv")
    confirmation_by_count = read_csv(ANALYSIS / "behavior_confirmation_v58" / "autoregressive_by_count.csv")
    behavior_gate = read_json(ANALYSIS / "behavior_confirmation_v58" / "behavior_gate.json")
    canonical = read_csv(DATA / "tables" / "final_autoregressive_summary.csv")
    sampling = read_csv(DATA / "tables" / "training_sampling_distribution.csv")
    train_metrics = read_csv(DATA / "tables" / "train_metrics.csv")
    geometry = read_csv(ANALYSIS / "v58_clean_ncc" / "selected_confirmation_summary.csv")
    selected_layers = read_json(ANALYSIS / "v58_clean_ncc" / "selected_layers.json")
    clean_projection_path = ANALYSIS / "v58_clean_ncc" / "geometry_confirmation_pca_all_layers.csv"
    clean_projection_manifest_path = (
        ANALYSIS / "v58_clean_ncc" / "geometry_confirmation_pca_all_layers_manifest.json"
    )
    geometry_cloud = read_csv(
        clean_projection_path
        if clean_projection_path.exists()
        else ANALYSIS / "phase_transition" / "tables" / "milestone_manifold_cloud_3d.csv"
    )
    geometry_dynamics = read_csv(ANALYSIS / "phase_transition" / "tables" / "dense_manifold_geometry.csv")
    rankings = read_csv(ANALYSIS / "v58_topk_ncc" / "head_rankings.csv")
    thinking_rankings = read_csv(ANALYSIS / "phase_transition" / "tables" / "fixed_head_rankings.csv")
    tf_behavior = read_csv(ANALYSIS / "v58_topk_ncc" / "post_ablation_behavior.csv")
    tf_ncc = read_csv(ANALYSIS / "v58_topk_ncc" / "post_ablation_ncc_selected.csv")
    targeted_hp = read_csv(ANALYSIS / "v58_confirmation_topk_hp500" / "free_running_summary.csv")
    targeted_hp_detail = read_csv(ANALYSIS / "v58_confirmation_topk_hp500" / "free_running_detail.csv")
    broad_hp = read_csv(ANALYSIS / "v58_confirmation_broad_topk_hp500" / "free_running_summary.csv")
    factorial_summary = read_csv(ANALYSIS / "v58_thinking_factorial_hp500" / "factorial_summary.csv")
    factorial_detail = read_csv(ANALYSIS / "v58_thinking_factorial_hp500" / "factorial_detail.csv")
    successor_summary = read_csv(ANALYSIS / "v58_thinking_successor_hp500" / "successor_summary.csv")
    successor_detail = read_csv(ANALYSIS / "v58_thinking_successor_hp500" / "successor_detail.csv")
    targeted_hp_manifest = read_json(ANALYSIS / "v58_confirmation_topk_hp500" / "manifest.json")
    broad_hp_manifest = read_json(ANALYSIS / "v58_confirmation_broad_topk_hp500" / "manifest.json")
    factorial_manifest = read_json(ANALYSIS / "v58_thinking_factorial_hp500" / "manifest.json")
    successor_manifest = read_json(ANALYSIS / "v58_thinking_successor_hp500" / "manifest.json")
    answer = read_csv(ANALYSIS / "v58_free_running_sufficiency" / "answer_transplant_confirmation.csv")
    progress = read_csv(ANALYSIS / "v58_free_running_sufficiency" / "progress_rollout_confirmation.csv")
    attention = read_csv(ANALYSIS / "extended" / "tables" / "attention_role_dynamics.csv")
    routing = read_csv(ANALYSIS / "phase_transition_audit" / "tables" / "routing_qk_by_k.csv")
    patching = read_csv(ANALYSIS / "phase_transition_audit" / "tables" / "retrieval_transport_recovery.csv")
    causal = read_csv(ANALYSIS / "phase_transition_audit" / "tables" / "local_head_causal_damage.csv")
    high_power = read_csv(ANALYSIS / "phase_transition_audit" / "tables" / "high_power_ar_summary.csv")

    final_hp_t = row_value(high_power, step=10000, mode="thinking")
    final_hp_nt = row_value(high_power, step=10000, mode="nonthinking")
    canonical_t = row_value(canonical, mode="thinking")
    canonical_nt = row_value(canonical, mode="nonthinking")
    if not math.isclose(float(final_hp_t["ar_accuracy"]), float(canonical_t["ar_final_accuracy"]), abs_tol=1e-12):
        raise RuntimeError("high-power Thinking endpoint does not reproduce canonical evaluation")
    if not math.isclose(float(final_hp_nt["ar_accuracy"]), float(canonical_nt["ar_final_accuracy"]), abs_tol=1e-12):
        raise RuntimeError("high-power Non-thinking endpoint does not reproduce canonical evaluation")
    if any(
        overlap != 0
        for overlap in (
            targeted_hp_manifest["selection_overlap_by_mode"]["thinking"],
            broad_hp_manifest["selection_overlap_by_mode"]["nonthinking"],
            factorial_manifest["selection_overlap"],
            successor_manifest["selection_overlap"],
        )
    ):
        raise RuntimeError("high-power causal confirmation overlaps role-selection examples")
    factorial_compare_columns = [
        "row_id",
        "ar_pred_count",
        "ar_accuracy",
        "trace_exact",
        "trace_ordered_marker_accuracy",
        "trace_generated_marker_count",
        "generated_tokens",
    ]
    factorial_rows = {
        arm: factorial_detail.loc[factorial_detail["arm"].eq(arm), factorial_compare_columns]
        .sort_values("row_id")
        .reset_index(drop=True)
        for arm in ("clean", "targeted_only", "broad_only", "joint")
    }
    broad_factorial_rowwise_null = factorial_rows["clean"].equals(factorial_rows["broad_only"])
    broad_factorial_interaction_null = factorial_rows["targeted_only"].equals(factorial_rows["joint"])
    if not broad_factorial_rowwise_null or not broad_factorial_interaction_null:
        raise RuntimeError("Thinking broad factorial arm is not the expected rowwise null")

    figures = {
        "behavior": ASSET_DIR / "v58_behavior_confirmation.png",
        "geometry": ASSET_DIR / "v58_clean_geometry.png",
        "roles": ASSET_DIR / "v58_role_specialization.png",
        "topk": ASSET_DIR / "v58_topk_causal.png",
        "role_causal": ASSET_DIR / "v58_role_causal_separation.png",
        "sufficiency": ASSET_DIR / "v58_free_running_sufficiency.png",
        "commit_query": ASSET_DIR / "v58_commit_query_alignment.png",
        "native_layer_selection": ASSET_DIR / "v58_native_layer_selection.png",
        "native_continuation": ASSET_DIR / "v58_native_continuation.png",
        "dynamics": ASSET_DIR / "v58_training_dynamics.png",
        "role_dynamics_heatmap": ASSET_DIR / "v58_role_differentiation_heatmaps.png",
        "geometry_dynamics": ASSET_DIR / "v58_geometry_emergence.png",
    }
    plot_behavior(confirmation_by_count, confirmation, figures["behavior"])
    plot_geometry(geometry, figures["geometry"])
    plot_role_specialization(attention, figures["roles"])
    plot_topk(tf_behavior, tf_ncc, targeted_hp, broad_hp, figures["topk"])
    plot_role_causal_separation(successor_summary, successor_detail, figures["role_causal"])
    plot_sufficiency(answer, progress, figures["sufficiency"])
    dynamics_stats = plot_training_dynamics(
        high_power, routing, attention, rankings, patching, causal, figures["dynamics"]
    )
    dynamics_stats.update(
        plot_role_differentiation_heatmaps(
            attention,
            rankings,
            thinking_rankings,
            figures["role_dynamics_heatmap"],
        )
    )
    dynamics_stats.update(
        plot_geometry_emergence(geometry_dynamics, selected_layers, figures["geometry_dynamics"])
    )
    geometry_widget = geometry_projection_widget(geometry_cloud)
    commit_query_section, commit_query_provenance = build_commit_query_section(ANALYSIS, figures["commit_query"])
    native_continuation_section, native_continuation_provenance = build_native_continuation_section(
        ANALYSIS, figures["native_layer_selection"], figures["native_continuation"]
    )

    c_t = row_value(confirmation, mode="thinking")
    c_nt = row_value(confirmation, mode="nonthinking")
    g_nt_run = row_value(geometry, comparison_mode="nonthinking", endpoint="nonthinking_prompt_occurrence")
    g_nt_final = row_value(geometry, comparison_mode="nonthinking", endpoint="nonthinking_answer_query")
    g_t_run = row_value(geometry, comparison_mode="thinking", endpoint="thinking_item_end")
    g_t_final = row_value(geometry, comparison_mode="thinking", endpoint="thinking_answer_query")
    geometry_initial_nt_final = row_value(
        geometry_dynamics,
        step=0,
        mode="nonthinking",
        site="final_answer",
        layer=int(selected_layers["nonthinking"]["nonthinking_answer_query"]),
    )
    geometry_initial_t_run = row_value(
        geometry_dynamics,
        step=0,
        mode="thinking",
        site="trace_marker",
        layer=int(selected_layers["thinking"]["thinking_item_end"]),
    )
    geometry_initial_t_final = row_value(
        geometry_dynamics,
        step=0,
        mode="thinking",
        site="final_answer",
        layer=int(selected_layers["thinking"]["thinking_answer_query"]),
    )

    tf_clean = row_value(tf_behavior, comparison_mode="thinking", scope="role_query_local", path_kind="clean", top_k=0)
    tf_k2 = row_value(tf_behavior, comparison_mode="thinking", scope="role_query_local", path_kind="ranked", top_k=2)
    hp_scope = "confirmation_role_query_local_free_running"
    fr_clean = row_value(targeted_hp, comparison_mode="thinking", scope=hp_scope, path_kind="clean", top_k=0)
    fr_k2 = row_value(targeted_hp, comparison_mode="thinking", scope=hp_scope, path_kind="ranked", top_k=2)
    fr_k4 = row_value(targeted_hp, comparison_mode="thinking", scope=hp_scope, path_kind="ranked", top_k=4)
    fr_k2_control_trace = median_control(
        targeted_hp, "trace_exact", mode="thinking", scope=hp_scope, top_k=2
    )
    fr_k4_control_trace = median_control(
        targeted_hp, "trace_exact", mode="thinking", scope=hp_scope, top_k=4
    )
    broad_clean = row_value(broad_hp, comparison_mode="nonthinking", scope=hp_scope, path_kind="clean", top_k=0)
    broad_k4 = row_value(broad_hp, comparison_mode="nonthinking", scope=hp_scope, path_kind="ranked", top_k=4)
    broad_k4_control = median_control(
        broad_hp, "ar_final_accuracy", mode="nonthinking", scope=hp_scope, top_k=4
    )

    factorial_clean = row_value(factorial_summary, arm="clean")
    factorial_targeted = row_value(factorial_summary, arm="targeted_only")
    factorial_broad = row_value(factorial_summary, arm="broad_only")
    factorial_joint = row_value(factorial_summary, arm="joint")
    successor_clean_hp = row_value(successor_summary, arm="clean")
    successor_targeted_hp = row_value(successor_summary, arm="targeted_top2")
    successor_top1_hp = row_value(successor_summary, arm="successor_top1")
    successor_low_hp = row_value(successor_summary, arm="successor_same_layer_low_score")
    successor_joint_hp = row_value(successor_summary, arm="targeted_top2_plus_successor_top1")

    targeted_k2_detail = targeted_hp_detail.loc[
        targeted_hp_detail["path_kind"].eq("ranked") & targeted_hp_detail["top_k"].eq(2)
    ].copy()
    targeted_k2_valid = targeted_k2_detail["ar_pred_count"].notna()
    targeted_k2_answer_length_match = float(
        (
            targeted_k2_detail.loc[targeted_k2_valid, "ar_pred_count"].astype(float)
            == targeted_k2_detail.loc[targeted_k2_valid, "trace_generated_marker_count"].astype(float)
        ).mean()
    )
    targeted_content_wrong_length_right = targeted_k2_detail.loc[
        targeted_k2_detail["trace_exact"].eq(0)
        & targeted_k2_detail["trace_marker_count_accuracy"].eq(1)
    ]
    targeted_length_wrong = targeted_k2_detail.loc[
        targeted_k2_detail["trace_marker_count_accuracy"].eq(0)
    ]

    successor_final_damage, successor_final_ci_low, successor_final_ci_high = _paired_damage_ci(
        successor_detail, "successor_top1", "ar_accuracy"
    )
    successor_marker_damage, successor_marker_ci_low, successor_marker_ci_high = _paired_damage_ci(
        successor_detail, "successor_top1", "trace_marker_count_accuracy"
    )

    selected_answer = answer.loc[answer["is_discovery_selected_layer"].eq(1.0)]
    t_donor = selected_answer.loc[
        selected_answer["mode"].eq("thinking")
        & selected_answer["condition"].eq("adjacent_count_donor")
    ]
    t_context = selected_answer.loc[
        selected_answer["mode"].eq("thinking")
        & selected_answer["condition"].eq("same_count_context_control")
    ]
    nt_donor = selected_answer.loc[
        selected_answer["mode"].eq("nonthinking")
        & selected_answer["condition"].eq("adjacent_count_donor")
    ]
    nt_context = selected_answer.loc[
        selected_answer["mode"].eq("nonthinking")
        & selected_answer["condition"].eq("same_count_context_control")
    ]

    patch_final = row_value(
        patching,
        step=10000,
        intervention="value_only_at_target_source",
        top_n=2,
    )
    successor_final = row_value(
        causal,
        step=10000,
        role="marker_successor",
        intervention="fixed_head_zero",
    )
    successor_control = row_value(
        causal,
        step=10000,
        role="marker_successor",
        intervention="same_layer_control_zero",
    )
    target_final = row_value(
        causal,
        step=10000,
        role="targeted_retrieval",
        intervention="fixed_head_zero",
    )
    target_control = row_value(
        causal,
        step=10000,
        role="targeted_retrieval",
        intervention="same_layer_control_zero",
    )

    sample_counts = sampling.loc[sampling["dimension"].eq("accepted_counts")].copy()
    sample_counts["share"] = sample_counts["examples"] / sample_counts["total_training_examples"]
    sample_min = float(sample_counts["share"].min())
    sample_max = float(sample_counts["share"].max())
    late_loss = train_metrics.loc[train_metrics["step"].eq(10000)].set_index("mode")

    top_t = rankings.loc[rankings["role"].eq("targeted_retrieval")].sort_values("rank").head(4)
    top_nt = rankings.loc[rankings["role"].eq("nonthinking_broad")].sort_values("rank").head(4)
    top_t_text = ", ".join(f"L{int(row.layer)}H{int(row.head)}" for row in top_t.itertuples(index=False))
    top_nt_text = ", ".join(f"L{int(row.layer)}H{int(row.head)}" for row in top_nt.itertuples(index=False))

    setup_table = html_table(
        ["Component", "v58 setting", "Control / interpretation"],
        [
            ["Task", "Shakespeare character stream; 3-character target set; query first; 256-character data; count 1–10", "Same corpus split, needle pool, sampler, tokenization and context permutation policy"],
            ["Trace", "Thinking: &lt;Think&gt; (&lt;Sep&gt; marker)<sup>N</sup> &lt;/Think&gt; &lt;Ans&gt; count; Non-thinking: &lt;Ans&gt; count", "No explicit numeric running index; marker sequence itself is unchanged across the v35→v58 search"],
            ["Models", "Two independent 12,658,176-parameter transformers; 4 layers × 8 heads; d=512; MLP=2048; RoPE (base 10,000); 384 model positions; atomic answer tokens", "Not a shared mode-switch model; exact head IDs are not compared across independently initialized models"],
            ["Optimization", "10,000 steps × batch 128 = 1.28M task examples/model; AdamW; lr 3e-4; warmup 500; weight decay .01; grad clip 1; BF16", "Single seed 1234; snapshots every 100 steps; full recovery state every 500"],
            ["Sampler", f"Max-entropy set×count sampler; realized count shares {pct(sample_min,2)}–{pct(sample_max,2)} in both modes", "Balances gold count while respecting which character sets can realize each count"],
            ["Loss 1–1500", "Teacher-forced next-token cross-entropy on every non-padding token", "Language-model warm start; final count has only its natural one-token frequency"],
            ["Loss 1501–10000", "Task output only; component-normalized coefficients count/trace/structure = 8/8/16", "Thinking region shares 25%/25%/50%; Non-thinking count/structure = 33.3%/66.7%; no scheduled sampling or contrastive loss"],
            ["Final readout", "Atomic count rows 1–10 use independent output vectors; all other vocabulary rows share input/output weights (tied)", "The hybrid readout is identical across modes, but a fully tied-count replication remains an external-validity check"],
        ],
    )

    setting_review_table = html_table(
        ["Review question", "Current v58 setting", "Assessment for the claim"],
        [
            ["Task and class balance", f"Same count-1–10 task, corpus split, sampler and architecture; realized class shares {pct(sample_min,2)}–{pct(sample_max,2)}", '<span class="status yes">clean matched comparison</span>'],
            ["Mode separation", "Two independently initialized and independently optimized models; no joint mode training or TTT", '<span class="status yes">supports separate dynamics</span>'],
            ["Trace content", "Fixed no-index grammar: (&lt;Sep&gt; marker)<sup>N</sup>; no numeric running labels", '<span class="status yes">preserves the intended trace</span>'],
            ["Position / trace-length leakage", "Data/query lengths are fixed and every trace item is exactly two tokens, so the Thinking answer-query position obeys p<sub>Ans</sub>=C+2N", '<span class="status no">major confound for final NCC/readout</span>'],
            ["Output parameterization", "Non-count vocabulary is tied; count rows 1–10 are untied for both modes", '<span class="status partial">fair across modes; tied replication open</span>'],
            ["Supervision parity", "Examples and optimizer steps are matched, but Thinking receives N marker labels plus the answer whereas Non-thinking receives only the answer", '<span class="status partial">protocol advantage, not equal-label comparison</span>'],
            ["Seed robustness", "One paired experiment seed (1234), with independently initialized mode models", '<span class="status no">multi-seed replication open</span>'],
            ["Broad-vs-targeted symmetry", "Thinking is competent; Non-thinking is at an 18% behavioral floor", '<span class="status no">NT broad mechanism not established</span>'],
        ],
    )

    behavior_table = html_table(
        ["Evaluation", "Non-thinking", "Thinking", "Difference / uniformity"],
        [
            ["Canonical test (50/count)", pct(canonical_nt["ar_final_accuracy"]), pct(canonical_t["ar_final_accuracy"]), f"+{100*(float(canonical_t['ar_final_accuracy'])-float(canonical_nt['ar_final_accuracy'])):.1f} pp"],
            ["Independent confirmation (200/count)", f"{pct(c_nt['ar_final_accuracy'],2)} [{pct(c_nt['ar_final_accuracy_wilson95_low'],2)}, {pct(c_nt['ar_final_accuracy_wilson95_high'],2)}]", f"{pct(c_t['ar_final_accuracy'],2)} [{pct(c_t['ar_final_accuracy_wilson95_low'],2)}, {pct(c_t['ar_final_accuracy_wilson95_high'],2)}]", f"+{100*(float(c_t['ar_final_accuracy'])-float(c_nt['ar_final_accuracy'])):.2f} pp"],
            ["Thinking per-count range", "—", f"{pct(behavior_gate['metrics']['thinking_min_count_accuracy'])}–100.0%", f"spread {100*float(behavior_gate['metrics']['thinking_count_spread']):.1f} pp"],
            ["Thinking trace diagnostics", "—", f"exact {pct(c_t['trace_exact'])}; ordered {pct(c_t['trace_ordered_marker_accuracy'])}; marker-count {pct(c_t['trace_marker_count_accuracy'])}", "Diagnostic only; not a behavior gate"],
        ],
    )

    geometry_table = html_table(
        ["Mode / endpoint", "Selected layer", "Logistic BA", "NCC BA", "Ordinal RSA"],
        [
            ["Non-thinking running: kth prompt occurrence end", f"L{int(g_nt_run['layer'])}", pct(g_nt_run["confirmation_logistic_balanced_accuracy"]), pct(g_nt_run["confirmation_ncc_balanced_accuracy"]), num(g_nt_run["confirmation_ordinal_rsa"])],
            ["Thinking running: kth trace item end", f"L{int(g_t_run['layer'])}", pct(g_t_run["confirmation_logistic_balanced_accuracy"]), pct(g_t_run["confirmation_ncc_balanced_accuracy"]), num(g_t_run["confirmation_ordinal_rsa"])],
            ["Non-thinking final: answer query", f"L{int(g_nt_final['layer'])}", pct(g_nt_final["confirmation_logistic_balanced_accuracy"]), pct(g_nt_final["confirmation_ncc_balanced_accuracy"]), num(g_nt_final["confirmation_ordinal_rsa"])],
            ["Thinking final: answer query", f"L{int(g_t_final['layer'])}", pct(g_t_final["confirmation_logistic_balanced_accuracy"]), pct(g_t_final["confirmation_ncc_balanced_accuracy"]), num(g_t_final["confirmation_ordinal_rsa"])],
        ],
    )

    causal_table = html_table(
        ["Experiment", "Clean / damaged baseline", "Intervention result", "Matched-control interpretation"],
        [
            ["Teacher-forced targeted Top-2", f"trace token acc {pct(tf_clean['teacher_forced_trace_accuracy'])}", pct(tf_k2["teacher_forced_trace_accuracy"]), f"control median {pct(median_control(tf_behavior,'teacher_forced_trace_accuracy',mode='thinking',scope='role_query_local',top_k=2))}"],
            ["500-sample targeted Top-2", f"trace exact {pct(fr_clean['trace_exact'])}; final {pct(fr_clean['ar_final_accuracy'])}", f"trace {pct(fr_k2['trace_exact'])}; ordered marker {pct(fr_k2['trace_ordered_marker_accuracy'])}; final {pct(fr_k2['ar_final_accuracy'])}", f"control trace median {pct(fr_k2_control_trace)}; trace damage is selective, count damage is modest"],
            ["500-sample targeted Top-4", f"trace exact {pct(fr_clean['trace_exact'])}", f"trace {pct(fr_k4['trace_exact'])}; ordered marker {pct(fr_k4['trace_ordered_marker_accuracy'])}; final {pct(fr_k4['ar_final_accuracy'])}", f"sole complement trace {pct(fr_k4_control_trace)}; it contains targeted ranks 5/6/7/10, so it is not a null"],
            ["500-sample successor L2H3", f"marker-count/final {pct(successor_clean_hp['trace_marker_count_accuracy'])}/{pct(successor_clean_hp['ar_final_accuracy'])}", f"marker-count {pct(successor_top1_hp['trace_marker_count_accuracy'])}; final {pct(successor_top1_hp['ar_final_accuracy'])}", f"low-score L2H5 gives {pct(successor_low_hp['trace_marker_count_accuracy'])}/{pct(successor_low_hp['ar_final_accuracy'])}; role is distributed but L2H3 is selectively stronger"],
            ["Targeted Top-2 + successor L2H3", f"targeted-only final {pct(successor_targeted_hp['ar_final_accuracy'])}; successor-only {pct(successor_top1_hp['ar_final_accuracy'])}", f"joint final {pct(successor_joint_hp['ar_final_accuracy'])}; ordered marker {pct(successor_joint_hp['trace_ordered_marker_accuracy'])}", "targeted adds ordering/content damage, but essentially no additional count damage beyond successor"],
            ["Top-2 targeted value patch", f"corrupt marker acc {pct(patch_final['corrupt_correct'])}", f"patched {pct(patch_final['patched_correct'])}; normalized recovery {pct(patch_final['normalized_recovery_mean'])}", f"restores +{num(patch_final['margin_restoration'],2)} correct-marker logit margin"],
            ["Thinking broad Top-2 factorial", f"clean final/trace {pct(factorial_clean['ar_final_accuracy'])}/{pct(factorial_clean['trace_exact'])}", f"broad-only {pct(factorial_broad['ar_final_accuracy'])}/{pct(factorial_broad['trace_exact'])}; joint exactly equals targeted-only", "zero broad main effect and zero broad×targeted interaction on all 500 rows"],
        ],
    )

    alignment_table = html_table(
        ["Large-model experiment / claim", "v58 aligned result", "Status"],
        [
            ["Separate running and final geometry", f"Running NCC NT/T={pct(g_nt_run['confirmation_ncc_balanced_accuracy'])}/{pct(g_t_run['confirmation_ncc_balanced_accuracy'])}; final={pct(g_nt_final['confirmation_ncc_balanced_accuracy'])}/{pct(g_t_final['confirmation_ncc_balanced_accuracy'])}", '<span class="status partial">final strongly aligned; running weak</span>'],
            ["Targeted retrieval bank and matched Top-K lesions", f"Thinking Top-4={top_t_text}; 500-sample trace exact {pct(fr_clean['trace_exact'])}→{pct(fr_k4['trace_exact'])}, final {pct(fr_clean['ar_final_accuracy'])}→{pct(fr_k4['ar_final_accuracy'])}", '<span class="status yes">content path is causal; final mediation partial</span>'],
            ["Successor / trace-cardinality mechanism", f"L2H3 lesion changes marker-count {pct(successor_clean_hp['trace_marker_count_accuracy'])}→{pct(successor_top1_hp['trace_marker_count_accuracy'])} and final {pct(successor_clean_hp['ar_final_accuracy'])}→{pct(successor_top1_hp['ar_final_accuracy'])}", '<span class="status yes">strong role separation from targeted bank</span>'],
            ["Post-ablation representation readout", f"Thinking running/final NCC remain {pct(g_t_run['confirmation_ncc_balanced_accuracy'])}/{pct(g_t_final['confirmation_ncc_balanced_accuracy'])} after local Top-K", '<span class="status partial">measured; no mediation at earlier layers</span>'],
            ["Free-running answer-state sufficiency", f"Thinking adjacent donor adoption {pct(t_donor['donor_adoption'].mean())} vs context control {pct(t_context['donor_adoption'].mean())}; NT {pct(nt_donor['donor_adoption'].mean())} vs {pct(nt_context['donor_adoption'].mean())}", '<span class="status yes">strong Thinking terminal readout</span>'],
            ["Natural no-index progress / recurrence", "Fresh discovery-selected L2 confirmation supports donor-directed free continuation over full-norm controls; multi-step transfer decays", '<span class="status partial">local continuation sufficiency supported; sustained recurrence limited</span>'],
            ["Non-thinking broad retrieval/aggregation", f"Descriptive Top-4={top_nt_text}; 500-sample clean/ranked-K4/control-complement final={pct(broad_clean['ar_final_accuracy'])}/{pct(broad_k4['ar_final_accuracy'])}/{pct(broad_k4_control)}", '<span class="status no">floor and contaminated controls; not selective</span>'],
            ["Universal Thinking final broad aggregator", "Not required: the trace-derived answer-query state itself is executable", '<span class="status partial">not claimed and not needed</span>'],
            ["One-arm serial mediation", "Targeted and successor lesions now share one free-running baseline, but targeted→successor→answer rescue is not closed", '<span class="status partial">role dissociation established; rescue chain open</span>'],
        ],
    )

    css = """
:root{--ink:#16202A;--muted:#52606D;--line:#D6DEE8;--paper:#FFFFFF;--wash:#F3F6F9;--blue:#2563A6;--orange:#D97706;--purple:#7158A6;--green:#23856D;--red:#B94444}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#E9EEF3;color:var(--ink);font-family:Inter,"Segoe UI","Noto Sans SC",Arial,sans-serif;line-height:1.64}
main{max-width:1180px;margin:28px auto;background:var(--paper);padding:55px 66px 72px;box-shadow:0 12px 38px rgba(24,39,56,.12)}
h1{font-size:2.25rem;line-height:1.16;margin:0 0 12px;letter-spacing:-.025em}h2{font-size:1.55rem;margin:52px 0 14px;padding-top:10px;border-top:2px solid var(--ink)}h3{font-size:1.14rem;margin:30px 0 9px}h4{margin:22px 0 6px}p{margin:9px 0 13px}.dek{font-size:1.08rem;color:var(--muted);max-width:940px}.meta{font-size:.88rem;color:var(--muted);margin-bottom:25px}
.abstract,.conclusion,.warning,.example,.formula,.audit,.purpose{padding:15px 18px;margin:16px 0;border-left:4px solid var(--blue);background:#F4F8FC}.conclusion{border-left-color:var(--green);background:#F2F8F6}.warning{border-left-color:var(--red);background:#FFF5F4}.example{border-left-color:var(--orange);background:#FFF8EC}.formula{border-left-color:var(--purple);background:#F7F5FB}.audit{border-left-color:#607D8B;background:#F3F6F8}.label{font-weight:750;margin-right:5px}
.toc{background:var(--wash);padding:17px 22px;border:1px solid var(--line);margin:24px 0}.toc ol{columns:2;margin:8px 0 0;padding-left:23px}.toc a{color:var(--blue);text-decoration:none}
.chain{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.chain>div{padding:14px;border:1px solid var(--line);background:#FAFBFC}.chain b{display:block;color:var(--blue);margin-bottom:4px}
figure{margin:24px 0 31px}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);background:white}figcaption{font-size:.9rem;color:#46525E;margin-top:8px;line-height:1.52}
.geometry-widget{margin:20px 0 8px;border:1px solid var(--line);background:#FAFBFC;padding:14px}.geometry-controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:0 0 12px}.geometry-source{font-size:.82rem;color:var(--muted)}.geometry-view-title{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0 7px;color:#334155}.geometry-layer-control{font-size:.78rem;font-weight:700;color:#334155}.geometry-layer-control select{margin-left:5px;padding:5px 8px;border:1px solid #BCC7D3;background:white;color:var(--ink)}.geometry-interaction-hint{font-size:.75rem;font-weight:500;color:#64748B}.geometry-panel{min-width:0;border:1px solid #E2E8F0;background:white}.geometry-panel-2d{height:850px}.geometry-panel-3d{height:980px}.interactive-caption{font-size:.9rem;color:#46525E;margin:7px 0 28px;line-height:1.52}
.table-wrap{overflow-x:auto;margin:15px 0 24px;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;font-size:.88rem}th{background:#EDF2F7;text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}td{padding:9px 10px;border-bottom:1px solid #E8EDF2;vertical-align:top}tr:last-child td{border-bottom:0}
code{font-family:"Cascadia Mono",Consolas,monospace;font-size:.88em;background:#EEF2F5;padding:2px 5px;border-radius:3px}.status{display:inline-block;padding:2px 7px;border-radius:12px;font-size:.78rem;font-weight:700;white-space:nowrap}.status.yes{color:#12664F;background:#DDF3EA}.status.partial{color:#855600;background:#FFF0C9}.status.no{color:#8F2F2F;background:#FBE2E2}
ul{padding-left:22px}.small{font-size:.86rem;color:var(--muted)}a{color:var(--blue)}
@media(max-width:800px){main{margin:0;padding:30px 20px}.chain{grid-template-columns:1fr}.geometry-panel-2d{height:760px}.geometry-panel-3d{height:900px}.toc ol{columns:1}h1{font-size:1.8rem}}
@media print{body{background:white}main{box-shadow:none;margin:0;max-width:none}figure{break-inside:avoid}}
"""

    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,"><title>NiaH Synthetic v58: geometry, causal experiments, and training dynamics</title><style>{css}</style></head><body><main>
<h1>NiaH Synthetic v58：No-index Thinking vs Independent Non-thinking</h1>
<p class="dek">完整训练设定、可选层的2D/3D Geometry Comparison、Top-K ablation / patching、free-running causal sufficiency，以及与v20和大模型实验对齐的training dynamics</p>
<p class="meta">生成日期：2026-09-05 · count classes：1–10 · seed：1234 · 两个 independently initialized models · training dynamics 同时报告 linear 与 log(step+100) 横轴</p>

<div class="abstract"><span class="label">核心结论。</span>v58 在独立 confirmation 上 Thinking 为 <strong>{pct(c_t['ar_final_accuracy'],2)}</strong>，Non-thinking 为 <strong>{pct(c_nt['ar_final_accuracy'],2)}</strong>，差 {100*(float(c_t['ar_final_accuracy'])-float(c_nt['ar_final_accuracy'])):.2f} pp；Thinking 十个 count 均在 {pct(behavior_gate['metrics']['thinking_min_count_accuracy'])}–100%，没有类别塌缩。新的每臂500条free-running干预给出更具体的角色分化：删除L4 targeted Top-2使trace exact从{pct(fr_clean['trace_exact'])}降到{pct(fr_k2['trace_exact'])}，但final count仅从{pct(fr_clean['ar_final_accuracy'])}降到{pct(fr_k2['ar_final_accuracy'])}；删除L2 successor L2H3则使marker-count从{pct(successor_clean_hp['trace_marker_count_accuracy'])}降到{pct(successor_top1_hp['trace_marker_count_accuracy'])}、final count降到{pct(successor_top1_hp['ar_final_accuracy'])}。这说明<strong>L4 targeted bank主要控制检索内容/顺序，L2 successor bank主要控制trace cardinality/termination，而final answer几乎读取已生成trace长度</strong>。因此targeted retrieval是因果有效的内容通路，却尚未被证明是Thinking最终accuracy优势的主要中介。Thinking answer-query NCC仍为{pct(g_t_final['confirmation_ncc_balanced_accuracy'])}，但固定两-token trace令answer position与count确定相关；Non-thinking broad-bank necessity也仍未建立。</div>

<div class="toc"><b>目录</b><ol><li><a href="#logic">问题与证据层级</a></li><li><a href="#setup">完整训练设定</a></li><li><a href="#behavior">行为与均匀性</a></li><li><a href="#geometry">Geometry Comparison / NCC</a></li><li><a href="#roles">Role specialization</a></li><li><a href="#causal">Ablation 与 patching</a></li><li><a href="#sufficiency">Free-running sufficiency</a></li><li><a href="#dynamics">Training dynamics</a></li><li><a href="#alignment">与大模型对齐及 gaps</a></li><li><a href="#limits">结论与限制</a></li><li><a href="#artifacts">复现产物</a></li></ol></div>

<h2 id="logic">1. 研究问题与证据层级</h2>
<p>核心问题不是“Thinking 有没有更好看的 attention map”，而是同一个 counting task 在两种输出协议下是否形成不同计算：Non-thinking 候选机制是 answer query 对 prompt evidence 的 broad retrieval；Thinking 候选机制是 trace 内逐项 targeted retrieval，再把 trace 压缩为最终 count state。报告把证据分成四层，避免把可读性、attention 与 causality 混为一谈。</p>
<div class="chain"><div><b>Representation</b>Logistic / NCC / ordinal RSA：count 是否可读、是否形成原型几何。</div><div><b>Routing</b>Broad score / targeted mass：query 看向哪些 evidence positions。</div><div><b>Necessity</b>Ranked Top-K removal 相对 matched controls 是否更伤自然计算。</div><div><b>Sufficiency</b>把 donor state 写入 receiver 后，free-running 输出是否采用 donor 信息。</div></div>
<div class="example"><span class="label">简单例子。</span>目标字符是 {{a,b,c}}，正文中按顺序出现 a、x、b、a、c，因此 gold count=4。Non-thinking 直接生成 <code>&lt;Ans&gt; 4</code>；Thinking 生成 <code>&lt;Think&gt; &lt;Sep&gt;a &lt;Sep&gt;b &lt;Sep&gt;a &lt;Sep&gt;c &lt;/Think&gt; &lt;Ans&gt; 4</code>。trace 没有“1,2,3,4”数字 index；若第 3 个 separator query 的 attention 指向正文第 3 个目标 occurrence，它就是 targeted retrieval 候选。</div>
<div class="conclusion"><span class="label">本节结论。</span>高 NCC 只说明 count cloud 可由简单原型读取；ranked ablation 才问模型是否自然依赖候选 bank；donor adoption 才问 state 是否足以驱动后续输出。新的factorial intervention进一步把候选链拆成targeted content与successor/cardinality两条作用不同的通路，因而不能再把“targeted trace damage”直接等同于“final-count mediation”。</div>

<h2 id="setup">2. 完整训练设定：模型到底怎么训</h2>
{setup_table}
<h3>2.1 为什么 Thinking 的 final-count token weight share 仍约 6%</h3>
<p>v58 在 task-output 阶段使用 <strong>component-normalized</strong> loss。Thinking 的 count、marker content、structure 三个区域先各自求 mean loss，再乘 8/8/16，因此 count 区域的目标系数份额是 8/(8+8+16)=25%。日志中的 <code>batch_final_count_token_weight_share</code> 则把系数重新摊回所有有效 token；因为 Thinking 有多个 marker 与 separator，单个 count token 最终约占 {pct(late_loss.loc['thinking','batch_final_count_token_weight_share'])}，而 Non-thinking 为 {pct(late_loss.loc['nonthinking','batch_final_count_token_weight_share'])}。前者不是“count 只得到 6% 的 component loss”，而是单 token 在更长 continuation 中的 token-level 份额。</p>
<p>step 1–1500 与 step 1501–10000 的 total loss 也不能直接连成一条物理同尺度曲线：前者平均整个序列，后者把多个区域的 mean loss 加权相加。报告因此用 free-running accuracy、attention mass、patch recovery 与 causal damage 做 dynamics 主指标，而不把 loss schedule 切换误画成 grokking 突变。</p>
<h3>2.2 RoPE、tied readout 与 position shortcut</h3>
<p><strong>当前位置编码确实是 RoPE</strong>：<code>rope_base=10000</code>，模型最多使用384个位置；256-character data与query长度固定。RoPE通过旋转query/key使attention感知相对位移，它对“第k个separator应检索正文中第k个目标occurrence”的有序targeted retrieval是合理选择，不能简单因为存在shortcut就移除。</p>
<p><strong>Tied</strong>表示input embedding与output classifier共享token向量：对普通token v，<code>logit(v)=h·e_v</code>。v58只对atomic count 1–10使用独立输出向量<code>w_v</code>，即<code>logit(v)=h·w_v</code>且<code>w_v≠e_v</code>；其他token仍tied。该hybrid readout在Thinking/Non-thinking两侧完全相同，所以行为比较是matched的，但它可能让最终类别边界更容易形成；与自然大模型对齐时应补fully-tied count replication。</p>
<div class="warning"><span class="label">最重要的setting风险。</span>Thinking的每个trace item固定为两个token <code>&lt;Sep&gt; marker</code>，而其余prefix长度固定，因此答案query的位置满足 <code>p_Ans=C+2N</code>。例如count 7的<code>&lt;Ans&gt;</code>必然比count 4晚6个位置。RoPE可以利用该距离；任何能感知位置的编码也可能利用它。这不否定发生在trace内部的targeted retrieval证据，但使final NCC=100%和full-state donor adoption无法单独区分“语义count code”与“trace-length/position code”。</div>
<div class="example"><span class="label">建议的最小position control。</span>在不改变marker身份与顺序的辅助replication中，让trace item之间带有随机、无监督的neutral gap，使总trace长度不再等于2N；或在固定最终位置的counterfactual intervention中删去一个marker item并用masked control slots补齐位置。若accuracy、final NCC和donor adoption仍保持，semantic compression的解释会明显更强。</div>
<h3>2.3 给合作者检查的setting审计</h3>
{setting_review_table}
<p>建议合作者重点决定四件事：(1) 主张是否只需要“supervised Thinking形成targeted retrieval和terminal readout”，还是必须包含对称的broad-vs-targeted机制；(2) 是否接受当前operational compression定义，还是要求position-jittered replication；(3) 是否要求count rows fully tied；(4) training-dynamics是否需要至少2–3个seed和paired initialization。</p>
<div class="audit"><span class="label">训练审计。</span>两种 mode 各看到 1,280,000 个 task examples；十个 count 的 realized shares 为 {pct(sample_min,2)}–{pct(sample_max,2)}。训练、phase、extended stages 均在 run manifest 中标记 complete；101 个 scientific snapshots 覆盖 step 0–10,000。训练总时长记录为 {float(read_csv(DATA/'tables'/'runtime_events.csv').loc[lambda d:d['block'].eq('train'),'duration_seconds'].iloc[0])/60:.1f} 分钟（CUDA runtime；manifest 未持久化 GPU SKU）。</div>
<div class="conclusion"><span class="label">本节结论。</span>v58 没有改 trace 内容、没有联合训练 mode、没有 scheduled sampling 或额外 contrastive objective。性能差来自两个独立模型在同一任务和均衡 sampler 上学习到的差异；主要设计变化是4×8、d=512的并行容量和component-normalized task-output loss。RoPE本身适合有序检索，但固定trace长度映射和hybrid count readout必须作为final-representation解释的边界公开报告。</div>

<h2 id="behavior">3. 行为结果：均匀且远超 Non-thinking</h2>
<h3>目的</h3><p>先确认 Thinking 优势不是只由 count=1 或 count=10 的边界类别贡献，也不是 canonical 500 条样本的偶然波动。</p>
<h3>设置</h3><p>Canonical test 每个 count 50 条；独立 confirmation 每个 count 200 条、每种 mode 共 2,000 条。Confirmation 明确排除 canonical 的 <code>(set_id, corpus_start)</code>，overlap=0，并用 Wilson 95% interval 报 overall 与 per-count accuracy。预注册 behavior gate 为 Thinking≥75%、最差 count≥70%、spread≤20 pp、Thinking−NT≥30 pp；trace exact 仅作诊断。</p>
{behavior_table}
{figure(figures['behavior'], "图 1｜行为与均匀性。Panel A 横轴是 gold count 1–10，纵轴是独立 confirmation 的 free-running exact-answer accuracy；点为每类 200 条样本的均值，误差线为逐类 Wilson 95% interval。Panel B 是每种 mode 的 2,000 条总体准确率与 Wilson interval。Panel C 仅对 Thinking 报 trace exact、逐位置 ordered-marker accuracy 与 marker-count accuracy。", "v58 behavior accuracy and count uniformity")}
<p>Non-thinking 在 N=1 和 N=10 有明显边界偏好，但总体只有 {pct(c_nt['ar_final_accuracy'],2)}；Thinking 每个 count 都高于或等于 90%，最低 N=9 为 90.0%。Canonical 与独立 confirmation 的总体结果分别是 {pct(canonical_t['ar_final_accuracy'])}/{pct(canonical_nt['ar_final_accuracy'])} 和 {pct(c_t['ar_final_accuracy'],2)}/{pct(c_nt['ar_final_accuracy'],2)}，方向与数量级一致。</p>
<div class="example"><span class="label">简单例子。</span>如果一个模型只会在所有输入都答“1”，均匀测试仍可能让 count=1 看起来 100%，但其余九类会接近 0，spread 接近 100 pp。v58 Thinking 的最差类仍为 90%，排除了这种 collapse。</div>
<div class="conclusion"><span class="label">本节结论。</span>按你最新提出的标准，“Native-thinking 不必追求满分，只需均匀且远超 Non-thinking”，v58 不只是勉强通过，而是以 78.25 pp 独立确认差距和 10 pp Thinking spread 明确通过；没有继续为 accuracy 搜索 v59 的必要。</div>

<h2 id="geometry">4. Geometry Comparison：running 与 final 分开</h2>
<h3>目的</h3><p>对齐大模型报告的两个语义站点：running label 在第 k 个 evidence/event 完成边界读取；final label 在生成首个答案数字前的 answer query 读取。分别问逐步 progress 与最终 count 是否形成紧致、可泛化的表示。</p>
<h3>计算方法</h3><div class="formula"><span class="label">Frozen NCC。</span>每个 endpoint 用 discovery 10 states/class 与 disjoint confirmation 8/class。Discovery 内按 grouped OOF 的 mean(Logistic BA, NCC BA) 选择一个 common decoder layer，并拟合 <code>StandardScaler → whitened PCA≤16</code>。对 class c 的 discovery states 求 centroid μ<sub>c</sub>；confirmation state z 的预测为 <code>argmin_c ||z−μ_c||²</code>。Balanced accuracy 是 10 类 recall 的平均，chance=10%。Ordinal RSA 是 centroid pair distance 与 |c−c′| 的 held-out Spearman correlation。本文把异质输入在 answer query 汇聚为可由少量 count centroids 泛化读取的几何称为 <em>representation compression</em>；这是操作性定义，不等同于信息论压缩或低维性的证明。</div>
{geometry_table}
{figure(figures['geometry'], "图 2｜Clean geometry。Panel A 横轴依次为 Non-thinking prompt-occurrence running、Thinking trace-item running、两种 mode 的 answer-query final endpoint；纵轴为未参与层选择的 confirmation balanced accuracy。实心柱为 L2 Logistic，斜线柱为 NCC，红色点线为 10-class chance；柱顶 Lx 是 discovery-only 选层。Panel B 使用相同四个 endpoint，纵轴为 held-out ordinal RSA。", "v58 clean running and final count geometry")}
<p>Thinking running NCC 从 NT 的 {pct(g_nt_run['confirmation_ncc_balanced_accuracy'])} 提升到 {pct(g_t_run['confirmation_ncc_balanced_accuracy'])}，但绝对值仍低；与此同时 ordinal RSA 达 {num(g_t_run['confirmation_ordinal_rsa'])}，说明平均 centroid path 随 count gap 有序，却不是每个 count 都形成紧致球状簇。真正显著的 compression 在 answer query：Thinking Logistic/NCC 都是 100%，而 NT 只有 {pct(g_nt_final['confirmation_logistic_balanced_accuracy'])}/{pct(g_nt_final['confirmation_ncc_balanced_accuracy'])}。</p>
<h3>4.3 可选择层的二维与三维主成分对比</h3>
<div class="purpose"><span class="label">实验目的。</span>沿用v20的四宫格：列固定比较Non-thinking与Thinking，行固定比较running index/progress与final count。二维与三维各有独立层选择器；每个视图内部的四个endpoint始终处于同一个residual depth，同时允许两个视图显示不同层。</div>
<div class="formula"><span class="label">定义与计算。</span>对每个endpoint × layer，StandardScaler与PC1–PC3只在discovery states上拟合，再把disjoint confirmation states投影到同一坐标系。二维图显示PC1–PC2，三维图显示PC1–PC3；红线按k=1→10连接confirmation class centroids。每个panel独立拟合PCA，因此PC方向只在该panel内部解释；跨mode比较簇的分离、有序性与重叠，不能比较PC1的正负号或绝对坐标。</div>
{geometry_widget}
<p class="interactive-caption">图 3｜可选择层的2D与3D Geometry Comparison。两张图均为2×2：左列Non-thinking、右列Thinking；上行running index/progress、下行answer-query final count。2D与3D层选择器相互独立，各自同时切换对应视图的四个panel；3D图支持拖拽旋转、滚轮缩放和双击复位。横纵轴为discovery-fitted PC1/PC2，三维图再加入PC3；颜色和数字为k=1–10，半透明点是held-out confirmation state，红线连接class centroid。PCA图用于描述形状，正式可读性仍以图2的discovery-selected、confirmation-evaluated Logistic/NCC为准。</p>
<div class="example"><span class="label">说明性示例。</span>如果count 4与count 5在PC1–PC2上重叠、但沿PC3分开，二维图会隐藏该差异，三维图可以显示。若切换layer后形状改变，只能说明该层的低维投影不同；是否更可读仍由held-out decoder指标判断。</div>
<div class="warning"><span class="label">Position-aware interpretation。</span>由于Thinking的answer-query位置本身是count的确定函数，100% final NCC应表述为“在当前固定grammar下，terminal state形成完美可读的count/trace-length code”。在position-jitter或fixed-position control完成前，不把它单独当作content-invariant semantic compression的证明。</div>
<div class="example"><span class="label">简单例子。</span>十个班级的平均身高可以按年级递增，因此 ordinal RSA 很高；但每个班的学生身高大量重叠，最近 centroid 分类仍会很差。Thinking running state 正是“平均轨迹有序、单例 cloud 仍松散”的情况。</div>
<div class="conclusion"><span class="label">本节结论。</span>Figure 2确认Thinking final state在冻结的held-out指标上可完全读出count；Figure 3让同一层的四个endpoint在2D与3D下直接比较。PCA图不增加因果证据。固定trace grammar仍使final representation与trace length/answer position相关，position-invariant semantic compression的结论待定。</div>

<h2 id="roles">5. Attention role specialization 与 head-bank differentiation</h2>
<h3>目的与定义</h3><p><strong>Targeted mass</strong> 是 Thinking 第 k 个 separator query 对正文中 matching 第 k 个目标 occurrence 的 attention mass。<strong>Broad score</strong> 用于 Non-thinking answer query：先求所有 active target occurrences 的总 attention mass M，再以有效覆盖率 C=exp(H)/N 惩罚只盯一个 occurrence 的 head，最终 B=M×C。所有 final roles 只在 held-out selection split 排名，图和数值在 disjoint reporting split 读取。</p>
{figure(figures['roles'], "图 4｜Role specialization。Panel A 为 Non-thinking final broad score 的 4 layers × 8 heads 热图；横轴=head，纵轴=layer，格内是 reporting-split raw score。Panel B 对 Thinking targeted mass 作同样展示，两图各有独立色标，不能跨图按颜色深浅比较。Panel C 只在同一个 Thinking 模型内比较 broad、marker-successor、targeted 三种 role；横轴列出 32 个 heads，纵轴是 role，每行除以该 role 的最大值，因此用于看 head identity 分工而非 raw effect size。", "v58 attention role specialization heatmaps")}
<p>Targeted ranking 的 Top-4 为 <strong>{top_t_text}</strong>，全部位于 L4；最终 reporting split 中前八名也全部来自 L4。Thinking broad 的 Top-4 与 targeted Top-4 零重合；marker-successor Top-4 与 targeted 只重合 1/4。这是同一 Thinking 模型内可解释的 role differentiation。相反，Non-thinking broad 的 discovery Top-4 为 {top_nt_text}，但 reporting split 的最高 head 变为 L1H3，说明小样本下 exact identity 不稳定。</p>
<div class="warning"><span class="label">重要边界。</span>Thinking 与 Non-thinking 是独立初始化模型，L1H3 在两边没有 neuron identity 对齐意义；跨 mode 只能比较 role 在 layer/head-bank 层面的分布。Non-thinking broad 的 selection/reporting 每个 count 仅 2/1 条，且其行为接近 floor，因此本报告不把 descriptive broad map 升级为稳定 causal bank。</div>
<div class="conclusion"><span class="label">本节结论。</span>v58 的清楚分化发生在 Thinking 内部：L4 targeted bank 与较早的 marker-successor、较分散的 broad role 使用不同 head subsets。Targeted bank 很集中；Non-thinking broad role 目前只有描述性信号，没有同等强度的稳定性。</div>

<h2 id="causal">6. Causal experiments：Top-K ablation、post-ablation NCC 与 patching</h2>
<h3>6.1 Ranked Top-K necessity：500-sample confirmation</h3><p>Head ranking只用discovery split；confirmation每个count 50条、每个arm共500条，且与selection样本overlap=0。Role-query-local removal只在Non-thinking answer query或Thinking每个separator query删除head output。Teacher-forced读即时token accuracy；free-running允许错误累积到trace与最终答案。Layer-matched controls是同层、同K的竞争head集合，不保证是功能为零的negative control。</p>
{figure(figures['topk'], "图 5｜Top-K causal suite。六个panel的横轴都是累计删除K=0/1/2/4；实线为discovery-ranked bank，灰虚线为layer-matched controls中位数，灰带为controls的min–max（不是置信区间）。A、D、E来自原teacher-forced/frozen-NCC protocol；B、C、F替换为新的500-sample confirmation free rollout。B/C是Thinking trace exact/final count，F是Non-thinking broad-bank final count。K4仅有一个disjoint complement，因此该点不是稳定的control distribution。", "v58 high-power Top-K ablation, post-ablation NCC, and free-running behavior")}
{causal_table}
<p>Targeted Top-2的paired trace-exact damage为22.4 pp（bootstrap 95% CI 18.4–26.4 pp），而final-count damage只有6.6 pp（4.2–9.2 pp）；Top-4 trace damage为33.2 pp、final damage10.4 pp。K4唯一control是剩余半个L4 bank，其中仍含targeted ranking第5/6/7/10名，故它是“complement”而不是null。Non-thinking clean只有{pct(broad_clean['ar_final_accuracy'])}，ranked K4为{pct(broad_k4['ar_final_accuracy'])}，但唯一control直接降到{pct(broad_k4_control)}；检查发现其中L1H2会让模型几乎不输出答案。因此这不是broad bank的选择性因果证据。</p>

<h3>6.2 为什么targeted retrieval对count伤害不大：content与cardinality分工</h3>
{figure(figures['role_causal'], "图 6｜Targeted content与successor/cardinality的因果分工；每个干预方案500条。A横轴是clean、targeted Top-2、successor L2H3、同层低successor-score L2H5 control及joint intervention；纵轴依次报告ordered-marker、marker-count与final-count rate。B纵轴为相对同一clean row的paired damage，误差线是10,000次row bootstrap 95% CI。C纵轴为模型有答案时P(final answer=实际生成的marker数)。D横轴是gold count 1–10，纵轴是逐类clean accuracy减intervened accuracy；count 10受上界/termination边界影响，应单独解释。", "v58 causal role separation between targeted content and trace cardinality")}
<p>证据直接回答“为什么对count伤害不大”。Targeted Top-2后，500条中final answer与实际生成marker数一致{pct(targeted_k2_answer_length_match)}。共有{len(targeted_content_wrong_length_right)}条出现“trace content或顺序不完全正确，但marker总数仍正确”；这些样本的final answer是{int(targeted_content_wrong_length_right['ar_accuracy'].sum())}/{len(targeted_content_wrong_length_right)}正确。相反，在{len(targeted_length_wrong)}条marker数量错误的样本中，final answer只有{int(targeted_length_wrong['ar_accuracy'].sum())}/{len(targeted_length_wrong)}正确。因此L4 targeted lesion常把某个marker身份/顺序弄错，却仍生成正确数量的items；terminal readout于是仍能答对count。</p>
<p>L2H3 successor lesion给出互补证据：marker-count damage为{100*successor_marker_damage:.1f} pp（95% CI {100*successor_marker_ci_low:.1f}–{100*successor_marker_ci_high:.1f}），final-count damage为{100*successor_final_damage:.1f} pp（{100*successor_final_ci_low:.1f}–{100*successor_final_ci_high:.1f}），但ordered-marker只从{pct(successor_clean_hp['trace_ordered_marker_accuracy'])}降到{pct(successor_top1_hp['trace_ordered_marker_accuracy'])}。低successor-score同层head L2H5也有损害，说明cardinality support是分布式的；然而L2H3的final损害比L2H5大{100*(float(successor_low_hp['ar_final_accuracy'])-float(successor_top1_hp['ar_final_accuracy'])):.1f} pp。Joint intervention的final为{pct(successor_joint_hp['ar_final_accuracy'])}，与successor-only的{pct(successor_top1_hp['ar_final_accuracy'])}几乎相同，却把ordered-marker进一步降到{pct(successor_joint_hp['trace_ordered_marker_accuracy'])}：targeted路径增加content/order damage，没有再增加明显count damage。</p>
<p>最后，在同一500条上删除Thinking broad Top-2（L3H6/L1H5）与clean逐行完全相同；joint也逐行等于targeted-only。故targeted后count仍正确不是因为这个final-query broad bank接管了计算，而是因为生成trace的cardinality仍由successor/termination通路维持。</p>
<div class="example"><span class="label">简单例子。</span>gold trace应为 <code>a,b,a,c</code>。Targeted lesion可能生成<code>a,b,b,c</code>：内容错一项但仍有4个marker，最终仍答4。Successor lesion则可能生成<code>a,b,a</code>后提前结束：内容prefix大体正确，却只有3项，final readout随之答3。前者伤“what/order”，后者伤“how many/when to stop”。</div>

<h3>6.3 为什么 Top-K 后 NCC 不变</h3><p>Thinking clean running/final representations由 discovery 选在 L3/L2；targeted heads 全在 L4。删除 L4 head output 不可能逆向改变已经记录的 L2/L3 state。Teacher forcing又把正确 gold marker作为下一 token 输入，使后续 answer query看到修复后的正确 trace。因此 post-ablation NCC 维持 {pct(g_t_run['confirmation_ncc_balanced_accuracy'])}/{pct(g_t_final['confirmation_ncc_balanced_accuracy'])} 并不否定 targeted heads；它说明当前 bank 直接参与 next-marker readout，而没有被证明是 earlier NCC compression 的 mediator。</p>
<h3>6.4 Retrieval transport patching</h3><p>构造 clean/corrupt 对：在第 k 个 targeted query 破坏应检索的 prompt occurrence，使正确 marker margin由 clean 的 {num(patch_final['clean_margin'],2)} 降到 {num(patch_final['corrupt_margin'],2)}。随后只从 clean run 恢复 ranked Top-2 heads 在 target source position 的 value contribution；最终恢复 {pct(patch_final['normalized_recovery_mean'])} 的 clean-corrupt margin gap，marker accuracy从 {pct(patch_final['corrupt_correct'])} 升到 {pct(patch_final['patched_correct'])}。这比只看 attention mass 更接近 causal retrieval transport。</p>
<div class="example"><span class="label">简单例子。</span>把第 4 个应检索的正文 occurrence 换成错误 evidence，模型的正确 marker logit落后；只把 clean L4 targeted heads 从正确 source 取出的 value write补回，正确 marker margin恢复约三分之一。这个实验不直接 patch整层 residual，因此比“把 clean final state 全部复制回来”更局部。</div>
<div class="conclusion"><span class="label">本节结论。</span>Targeted bank有attention concentration、即时损害、free-running content/order damage与value-only recovery四类一致证据；successor/termination bank则对trace cardinality和final count更必要。这是明确的role specialization，但也缩小了可写主张：targeted retrieval是自然计算中的辅助内容路径，目前没有证据表明它单独中介Thinking的主要accuracy优势。Non-thinking broad-bank necessity仍未建立。</div>

<h2 id="sufficiency">7. Free-running causal sufficiency：terminal state 足够吗？</h2>
<h3>Experiment A · Answer-query state transplant</h3><p>对相邻 count receiver/donor，在 discovery 选择一个 layer；Thinking 先自由生成 trace直到 <code>&lt;Ans&gt;</code>，然后把 donor 的完整 answer-query residual写入 receiver并继续 greedy生成。Primary outcome 是 receiver 是否采用 donor count；same-count context donor控制“复制另一个上下文 state本身”是否导致 adoption。</p>
{figure(figures['sufficiency'], "图 7｜Free-running sufficiency。Panel A 横轴是相邻-count full-state donor 与 same-count context control，纵轴是最终 greedy donor-count adoption；每个方案16个confirmation pairs，橙/紫分别为Non-thinking/Thinking。Panel B在Thinking trace中比较clean、same-position centroid shift、等范数orthogonal control，以及两个跨absolute-position natural donor upper bounds；纵轴是下一marker采用donor successor的比例，†表示位置混杂。", "v58 free-running terminal and progress-state sufficiency")}
<p>Thinking 在 L4 的相邻 donor adoption 是 {pct(t_donor['donor_adoption'].mean())}（15/16），same-count context control 为 {pct(t_context['donor_adoption'].mean())}，平均 donor-vs-receiver margin shift +{num(t_donor['donor_margin_shift'].mean(),2)}。Non-thinking 在 L3 为 {pct(nt_donor['donor_adoption'].mean())}，但 context control 也是 {pct(nt_context['donor_adoption'].mean())}，因此不是 count-specific categorical sufficiency。</p>
<h3>Experiment B · Progress-state transplant</h3><p>Same-position centroid shift在17个 eligible cells 中把 donor next-marker adoption从 clean的 {pct(progress.loc[progress['condition'].eq('clean'),'donor_first_adoption'].mean())} 提到 {pct(progress.loc[progress['condition'].eq('centroid_shift'),'donor_first_adoption'].mean())}；orthogonal control仍为 {pct(progress.loc[progress['condition'].eq('orthogonal_control'),'donor_first_adoption'].mean())}。Natural marker/item donor达到 {pct(progress.loc[progress['condition'].eq('natural_marker_cross_position'),'donor_first_adoption'].mean())}，但 donor与receiver位于不同 absolute positions；固定两-token item格式无法消除RoPE/trace-length混杂。所有 progress arms 的最终 count accuracy均为100%，因此没有形成“改写 progress→改写 final answer”的干净链。</p>
<div class="conclusion"><span class="label">本节结论。</span>不需要证明一个universal final broad aggregator。v58已证明free-generated trace后的answer-query state可执行地驱动count；结合Figure 6中final answer几乎逐行等于generated marker count，当前证据支持trace-cardinality/answer-position readout。该结果确认terminal bridge存在；position-invariant semantic aggregation与低维progress state的行为充分性仍为结论待定。</div>
{commit_query_section}
{native_continuation_section}

<h2 id="dynamics">8. Training dynamics：何时形成 specialization</h2>
<h3>8.1 实验目的、横轴与冻结选择</h3><div class="purpose"><span class="label">实验目的。</span>检验行为准确率、representation geometry、attention role、QK routing、value transport和局部因果效应的形成顺序，并与v20使用相同的“固定最终选中对象后向前追踪”原则比较。</div>
<p>Figure 8与Figure 10使用线性optimizer step；Figure 9把完全相同的head×checkpoint矩阵同时画在线性step与log(step+100)横轴上。v58覆盖0–10,000 steps且每100 steps保存scientific snapshot：线性轴保留3,000–8,000的真实形成宽度，log轴展开0–1,500的早期分化。<a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html">Olsson et al.</a>在跨数量级token范围时使用log elapsed tokens并分析loss导数；坐标变换只改变视觉分配，不会单独构成突变证据。v58未进行跨seed change-point model比较，因此本节报告形成区间和时间顺序，不报告已确认的phase transition。</p>
<div class="formula"><span class="label">冻结选择。</span>Head identity与geometry layer均在step 10,000的discovery split确定。所有早期checkpoint追踪相同LxHy或相同residual layer；没有在每个checkpoint重新选择最高score head或最优decoder layer。红色竖虚线标记step 1,500从全序列loss切换到task-output component-normalized loss。</div>

<h3>8.2 行为、routing、bank concentration与因果使用</h3>
{figure(figures['dynamics'], "图 8｜同步training dynamics。所有panel横轴为optimizer step，红色竖虚线为step 1,500 loss-scope切换。A纵轴为每个checkpoint、每种mode各500条free-running样本的final-answer accuracy，并显示Thinking trace exact。B左纵轴为固定L4H5的targeted attention mass，右纵轴为correct occurrence减best wrong occurrence的scaled QK margin。C纵轴为最终Top-4 bank占全部32 heads同一role score总和的比例，水平点线4/32表示均匀份额。D纵轴为即时Top-4与最终Top-4的Jaccard。E纵轴为corrupt baseline下Top-2 value patch recovery与marker accuracy。F纵轴为selected-head damage减same-layer control damage，单位为correct-token logit margin。", "v58 synchronized behavior, routing, bank, patching, and causal training dynamics")}
<p>Thinking final accuracy从step 3,000的41.4%提高到4,500的59.6%、6,000的79.4%、8,000的91.4%和10,000的95.4%。固定L4H5 targeted mass在step {int(dynamics_stats['first_targeted_50pct_step']):,}首次达到最终值的50%，在step {int(dynamics_stats['first_targeted_80pct_step']):,}达到80%；QK margin约step 4,800后转为正值。Top-2 value recovery从step 3,000附近开始增加，step 10,000为{pct(patch_final['normalized_recovery_mean'])}。Marker-successor L2H3的最终matched causal specificity为{num(float(successor_final['causal_damage'])-float(successor_control['causal_damage']))} logit-margin units；targeted L4H5为{num(float(target_final['causal_damage'])-float(target_control['causal_damage']))}。</p>

<h3>8.3 Head分化热图：linear steps与log steps</h3>
{figure(figures['role_dynamics_heatmap'], "图 9｜Head-bank differentiation heatmaps。三行依次为Non-thinking broad、Thinking marker-successor与Thinking targeted retrieval；纵轴列出32个attention heads，横轴为101个checkpoint。每个checkpoint内把同一role的32个非负score归一化为和1，颜色表示该head占全部role score的份额，因此观察的是bank内部集中与迁移，不能跨role解释raw effect size。左列用线性optimizer step，右列显示完全相同的数据但使用log(step+100)横轴；白色竖虚线是step 1,500 objective switch，青色实/虚横线标出final discovery rank-1/rank-2 head。", "v58 linear and log step role differentiation heatmaps")}
<p>Linear与log两列的数据完全相同。Log轴把step 0–1,500展开，因此更容易看到successor bank在objective switch附近迅速集中；linear轴保留step 3,000–8,000的真实宽度，更清楚显示targeted bank是持续数千steps逐渐迁移到L4，而不是单checkpoint突变。热图的列归一化会抹去role总量的增长，所以必须与Figure 8B的raw targeted mass、Figure 8C的Top-4 share以及Figure 8F的matched causal specificity共同解释。</p>
<p>Thinking successor discovery rank-1 L2H3在step {int(dynamics_stats['marker_successor_rank1_first_50pct_step']):,}达到最终reporting score的50%，在step {int(dynamics_stats['marker_successor_rank1_first_80pct_step']):,}达到80%；targeted discovery rank-1 L4H5的对应时间为step {int(dynamics_stats['targeted_retrieval_rank1_first_50pct_step']):,}与{int(dynamics_stats['targeted_retrieval_rank1_first_80pct_step']):,}。该时间顺序表明successor role较早形成。Successor rank-2 L4H0在reporting曲线末端高于L2H3，因此精确head排序存在split差异；高功效ablation仍显示L2H3的cardinality effect更强。Non-thinking broad rank-1在初始化时已高于最终score的一半，后续轨迹非单调，且最终selection与reporting的最高head不一致；当前数据不支持为Non-thinking broad bank指定稳定形成时间。</p>

<h3>8.4 Representation geometry emergence</h3>
{figure(figures['geometry_dynamics'], "图 10｜固定最终选层的representation geometry dynamics。横轴为optimizer step；红色竖虚线为step 1,500 loss-scope切换。橙、蓝、紫曲线分别固定追踪Non-thinking final L4、Thinking running L3和Thinking final L2。A纵轴为count/progress centroids前三个PC解释的方差比例；B为centroid effective dimension，数值越低表示centroid variance集中于更少方向；C为按k顺序连接centroid所得path straightness，即首尾弦长除以相邻路径总长。所有量均为描述性几何，不等同于held-out decodability。", "v58 fixed-layer representation geometry emergence")}
<p>Thinking running L3在step 0时前三个centroid PC已解释{pct(geometry_initial_t_run['centroid_pc1_to_pc3_variance_fraction'])}方差，最终为{pct(dynamics_stats['thinking_running_centroid_pc1_to_pc3_variance_fraction_final'])}；该低维结构在训练前已存在，可能来自共享token与位置结构，不能解释为已经学会count。Thinking final L2的effective dimension从{num(geometry_initial_t_final['centroid_effective_dimension'],2)}降到{num(dynamics_stats['thinking_final_centroid_effective_dimension_final'],2)}。Non-thinking final L4也从{num(geometry_initial_nt_final['centroid_effective_dimension'],2)}降到{num(dynamics_stats['nonthinking_final_centroid_effective_dimension_final'],2)}，但其最终held-out NCC只有{pct(g_nt_final['confirmation_ncc_balanced_accuracy'])}。因此低effective dimension与高count可读性不是等价指标；Non-thinking centroids可以集中在少数方向，同时仍缺少按count稳定分离的class geometry。</p>
<div class="example"><span class="label">说明性示例。</span>十个centroid都落在同一条弯曲曲线上时，effective dimension可以接近1；如果不同count沿曲线反复折返或sample cloud跨类重叠，NCC仍可能接近chance。Figure 10描述centroid形状，Figure 2衡量held-out class readout，两者回答不同问题。</div>

<h3>8.5 综合形成顺序</h3>
<p>v58的单seed结果支持四个连续阶段：(1) step 0–1,500期间，trace token/position相关的低维几何已经可见；(2) step 1,500–3,000期间，L2 successor score明显增加，Thinking在step 3,000达到41.4% final accuracy；(3) step 3,000–6,000期间，L4 targeted mass、QK selectivity和value recovery同步增加，final accuracy达到79.4%；(4) step 6,000–10,000期间，targeted mass逐渐接近平台，final accuracy继续提高到95.4%，局部causal specificity与patch recovery保持正值。</p>
<div class="warning"><span class="label">分析边界。</span>上述顺序来自单一seed 1234。时间上先出现属于temporal-order evidence；训练因果关系需要多个seed、受控改变形成时间以及behavior onset的同步变化。当前曲线支持3,000–8,000的连续形成区间，瞬时phase transition的结论待定。</div>
<div class="conclusion"><span class="label">本节结论。</span>v58复现了v20的主要形成顺序：trace相关结构和successor role较早出现，targeted retrieval bank随后在L4分化，Thinking行为在targeted routing增强期间持续改善。v58进一步显示最终count主要依赖successor/cardinality路径；targeted role与accuracy的时间共现尚未证明targeted路径中介主要accuracy增益。</div>

<h2 id="alignment">9. 与大模型 Non-thinking / Native-thinking 实验对齐及 gaps</h2>
{alignment_table}
<h3>9.1 已经对齐的部分</h3><ul><li><b>站点：</b>Non-thinking running取k-th prompt occurrence end；Thinking running取k-th trace item end；final均取答案数字前query，与Geometry Comparison的语义边界一致。</li><li><b>Representation：</b>discovery-only选层与frozen confirmation Logistic/NCC，分开running和final，不用PCA图替代held-out指标。</li><li><b>Necessity：</b>discovery-ranked Top-K与layer/head-count matched controls，同时报告teacher-forced immediate damage、post-ablation frozen NCC和free-running rollout。</li><li><b>Sufficiency：</b>相邻count donor、same-count context control、free-running continuation，直接读取greedy donor adoption。</li><li><b>Dynamics：</b>固定final-selected heads追踪全部checkpoint，不在每一步重新选一个最好看的head。</li></ul>
<h3>9.2 当前最重要的 gaps</h3><ul><li><b>Targeted-to-count mediation gap：</b>L4 targeted lesion强烈损害content/order，却仅小幅损害final count；joint lesion在successor-only基础上几乎不增加count damage。故targeted retrieval存在不等于它解释了Thinking accuracy优势。</li><li><b>Non-thinking broad retrieval gap：</b>大模型已有broad-bank matched causal effect、retrieval subspace和late executable state；v58 Non-thinking在16–18% floor附近。500-sample ranked K4虽降低accuracy，但control complement更严重地破坏answer production，仍无法给出选择性。</li><li><b>Running representation gap：</b>Thinking running NCC只有{pct(g_t_run['confirmation_ncc_balanced_accuracy'])}；高ordinal RSA说明有序，不代表紧致或content-free counter。</li><li><b>Progress continuation gap：</b>Experiment D在新机制cohort、discovery-selected L2上确认contextual item state对donor-directed自由continuation的局部因果充分性；多步延续弱于Native报告。单字符重复、内容混合与有限discovery覆盖仍限制解释，纯算术counter及稳定长程递推尚未建立；final-count不变不否定局部充分性。</li><li><b>Control contamination：</b>L4 K4 complement仍含targeted ranks 5/6/7/10；L2 low-score control仍有cardinality作用；Non-thinking control含一般readout head。raw attention score不能保证因果null。</li><li><b>Serial mediation gap：</b>虽然targeted与successor lesion已在同一free-running baseline中比较，但还没有restore targeted output→successor/termination→answer state并闭合最终答案。</li><li><b>Final-state confound：</b>fixed grammar使p<sub>Ans</sub>=C+2N；full answer residual transplant可能携带count、trace length、position和其他上下文，尚无absolute-position matched不同count control。</li><li><b>Scale与监督：</b>12.66M字符模型、固定separator grammar、atomic one-token answer、256-char context和单seed，不能与4B/8B自然语言模型比较绝对head ID、formation step或accuracy。</li></ul>
<div class="warning"><span class="label">关于“两个mechanism都显著”。</span>v58足以作为强Thinking targeted-content / successor-cardinality specialization与operational final-compression setting；它不支持把Non-thinking broad retrieval写成同等强的已复现机制，也不支持说targeted bank本身解释了大部分final-count accuracy。如果论文中心句必须是“两侧都由强因果证据确认，只是broad vs targeted不同”，还需要一个行为不在floor、但明显弱于Thinking的Non-thinking setting或额外训练seed，不能靠换图隐藏当前null。</div>
<div class="conclusion"><span class="label">本节结论。</span>v58与大模型在geometry、matched/free-running ablation、patching、sufficiency和dynamics的实验层级上已经对齐。已有Thinking targeted content path、successor/cardinality path、terminal readout及contextual-state continuation的局部证据；主要未对齐处是targeted-to-final-count mediation、Non-thinking broad aggregation、position-invariant final compression与稳定长程progress recurrence。</div>

<h2 id="limits">10. 最终结论、限制与建议写法</h2>
<div class="abstract"><span class="label">可以写进正文的结论。</span>(1) 在同一balanced count-1–10任务上，独立训练的no-index Thinking以{pct(c_t['ar_final_accuracy'],2)}显著超过Non-thinking的{pct(c_nt['ar_final_accuracy'],2)}，且Thinking十类均≥{pct(behavior_gate['metrics']['thinking_min_count_accuracy'])}；(2) answer-query NCC为100%，但固定grammar使其更准确地称为operational count/trace-length compression；(3) L4 targeted bank因果控制trace content/order，Top-2使exact trace下降22.4 pp而final count仅下降6.6 pp；L2 successor L2H3因果控制trace cardinality/termination，使marker-count下降53.4 pp、final下降51.6 pp；(4) final answer在targeted-lesion后仍有{pct(targeted_k2_answer_length_match)}等于实际生成marker数，说明terminal readout主要追踪trace cardinality；(5) target-source value patch与answer-state donor adoption分别证明局部retrieval transport和terminal executability，但尚未闭合targeted→count mediation；(6)不需要也没有证据支持universal final broad aggregator。</div>
<ul><li>所有模型与training dynamics来自单一seed 1234；checkpoint rows不是独立训练replicates。</li><li>高功效free-running confirmation为50/count、500/arm且与selection overlap=0；但K4只有一个disjoint complement，不能当作control distribution。</li><li>Attention role score按routing定义排名，不含V/W<sub>O</sub>写入强度；单head attention score与causal damage未必一致。低score同层head仍可能承担一般trace/cardinality支持。</li><li>Targeted bank位于L4，而running/final NCC选层在L3/L2；因此当前Top-K不能检验earlier NCC的中介。</li><li>Targeted content damage与successor cardinality damage已在同一baseline中分离，但尚无position-controlled serial rescue。</li><li>高功效AR cached evaluator已与reference逐token一致；修复前错误使用tied embedding的派生CSV已隔离，不进入报告。</li><li><code>plan.tex</code>和<code>LLM_Compression.pdf</code>本轮已不在此前Downloads路径，未重新核验；大模型对齐直接依据三份仍可用HTML报告与旧report保存的crosswalk。</li></ul>
<p><b>下一步优先级：</b>第一优先不是继续调accuracy，而是做position-controlled trace-length replication，并在同一damaged rollout中restore targeted output→successor/termination→answer state，检验serial mediation。第二优先是2–3个固定v58 seed和行为高于floor的matched Non-thinking baseline。不要把targeted trace damage写成已解释final accuracy，也不要用降低Non-thinking到chance替代broad mechanism证据。</p>
<div class="conclusion"><span class="label">最终判断。</span>保留v58。当前证据支持功能分工：targeted bank参与内容/顺序生成，successor bank参与数量/终止控制，terminal state用于count readout。新增独立机制confirmation支持contextual item state对donor-directed continuation的局部因果充分性。Non-thinking broad-bank necessity、Thinking targeted-to-count完整mediation、稳定长程progress recurrence和position-invariant final compression仍未建立。</div>

<h2 id="artifacts">11. 复现产物与 provenance</h2>
{html_table(["Artifact","Path / role"],[
    ["Frozen v58 run archive", html.escape(str(DATA))],
    ["Independent behavior confirmation", html.escape(str(ANALYSIS/'behavior_confirmation_v58'))],
    ["Clean aligned NCC", html.escape(str(ANALYSIS/'v58_clean_ncc'))],
    ["Clean NCC and layer-selectable confirmation PCA clouds", html.escape(str(ANALYSIS/'v58_clean_ncc'))],
    ["Milestone geometry dynamics", html.escape(str(ANALYSIS/'phase_transition'))],
    ["Teacher-forced Top-K + post-ablation NCC", html.escape(str(ANALYSIS/'v58_topk_ncc'))],
    ["High-power targeted Top-K (500/arm)", html.escape(str(ANALYSIS/'v58_confirmation_topk_hp500'))],
    ["High-power Non-thinking broad Top-K", html.escape(str(ANALYSIS/'v58_confirmation_broad_topk_hp500'))],
    ["Thinking broad×targeted factorial", html.escape(str(ANALYSIS/'v58_thinking_factorial_hp500'))],
    ["Thinking targeted×successor factorial", html.escape(str(ANALYSIS/'v58_thinking_successor_hp500'))],
    ["Free-running sufficiency", html.escape(str(ANALYSIS/'v58_free_running_sufficiency'))],
    ["Commit→query primary assay", html.escape(str(ANALYSIS/'v58_commit_query_20260905'))],
    ["Position-aligned item-scope sensitivity", html.escape(str(ANALYSIS/'v58_aligned_item_query_20260905'))],
    ["Commit→query protocol", html.escape(str(ROOT/'docs'/'v58_commit_query_protocol.md'))],
    ["Native-aligned fresh continuation confirmation", html.escape(str(ANALYSIS/'v58_native_continuation_20260905'))],
    ["Frozen Native-aligned continuation protocol", html.escape(str(ROOT/'docs'/'v58_native_continuation_protocol.md'))],
    ["101-snapshot roles", html.escape(str(ANALYSIS/'extended'))],
    ["Routing/patching/causal/high-power dynamics", html.escape(str(ANALYSIS/'phase_transition_audit'))],
    ["Four-endpoint geometry cloud exporter", html.escape(str(ROOT/'scripts'/'export_v58_geometry_cloud.py'))],
    ["Report builder", html.escape(str(Path(__file__).resolve()))],
    ["Self-contained report", html.escape(str(output.resolve()))],
])}
<p class="small">Run created {html.escape(str(run_manifest.get('created_at_utc','unknown')))}; code commit at report build: <code>{git_commit()}</code>. Static figures are regenerated from archived tables and embedded as base64; the selectable geometry view embeds its data and Plotly runtime inline. The final HTML has no external image or JavaScript dependency. External paper links are references only; all v58 numerical claims come from local frozen artifacts.</p>
</main></body></html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    geometry_record_count = (
        len(geometry_cloud)
        if "endpoint" in geometry_cloud.columns
        else len(
            geometry_cloud.loc[
                geometry_cloud["step"].eq(geometry_cloud["step"].max())
                & geometry_cloud["site"].isin(["trace_marker", "final_answer"])
            ]
        )
    )
    manifest = {
        "schema_version": "v58_synthetic_report_v6",
        "report": str(output.resolve()),
        "report_sha256": sha256(output),
        "primary_comparison": "independently trained v58 separator/no-index Thinking vs v58 Non-thinking",
        "run": DATA.name,
        "code_commit": git_commit(),
        "behavior_gate": behavior_gate,
        "dynamics_statistics": dynamics_stats,
        "commit_query_alignment": commit_query_provenance,
        "native_continuation_alignment": native_continuation_provenance,
        "figures": {
            key: {"path": str(value.resolve()), "sha256": sha256(value)}
            for key, value in figures.items()
        },
        "source_roots": {
            "run_archive": str(DATA.resolve()),
            "analysis": str(ANALYSIS.resolve()),
        },
        "input_hashes": {
            input_label(path): sha256(path)
            for path in (
                DATA / "config.json",
                ANALYSIS / "behavior_confirmation_v58" / "autoregressive_summary.csv",
                ANALYSIS / "v58_clean_ncc" / "selected_confirmation_summary.csv",
                ANALYSIS / "v58_clean_ncc" / "selected_layers.json",
                *((clean_projection_path,) if clean_projection_path.exists() else ()),
                *((clean_projection_manifest_path,) if clean_projection_manifest_path.exists() else ()),
                ANALYSIS / "phase_transition" / "tables" / "milestone_manifold_cloud_3d.csv",
                ANALYSIS / "phase_transition" / "tables" / "dense_manifold_geometry.csv",
                ANALYSIS / "phase_transition" / "tables" / "fixed_head_rankings.csv",
                ANALYSIS / "extended" / "tables" / "attention_role_dynamics.csv",
                ANALYSIS / "v58_topk_ncc" / "post_ablation_behavior.csv",
                ANALYSIS / "v58_confirmation_topk_hp500" / "free_running_summary.csv",
                ANALYSIS / "v58_confirmation_topk_hp500" / "free_running_detail.csv",
                ANALYSIS / "v58_confirmation_broad_topk_hp500" / "free_running_summary.csv",
                ANALYSIS / "v58_thinking_factorial_hp500" / "factorial_summary.csv",
                ANALYSIS / "v58_thinking_factorial_hp500" / "factorial_detail.csv",
                ANALYSIS / "v58_thinking_successor_hp500" / "successor_summary.csv",
                ANALYSIS / "v58_thinking_successor_hp500" / "successor_detail.csv",
                ANALYSIS / "v58_free_running_sufficiency" / "answer_transplant_confirmation.csv",
                ANALYSIS / "phase_transition_audit" / "tables" / "high_power_ar_summary.csv",
            )
        },
        "validation": {
            "high_power_endpoint_matches_canonical": True,
            "cached_vs_reference_generation": "10/10 exact for each mode at step 5000",
            "causal_confirmation_examples_per_arm": 500,
            "causal_selection_overlap": 0,
            "thinking_broad_factorial_clean_equals_broad_rowwise": broad_factorial_rowwise_null,
            "thinking_broad_factorial_targeted_equals_joint_rowwise": broad_factorial_interaction_null,
            "targeted_top2_answer_equals_generated_marker_count": targeted_k2_answer_length_match,
            "interactive_geometry_records": int(geometry_record_count),
            "interactive_geometry_endpoints": sorted(
                str(value)
                for value in (
                    geometry_cloud["endpoint"].unique()
                    if "endpoint" in geometry_cloud.columns
                    else ["nonthinking_answer_query", "thinking_item_end", "thinking_answer_query"]
                )
            ),
            "interactive_geometry_layers": sorted(int(value) for value in geometry_cloud["layer"].unique()),
            "interactive_geometry_steps": (
                sorted(int(value) for value in geometry_cloud["step"].unique())
                if "step" in geometry_cloud.columns
                else [int(cfg["train_steps"])]
            ),
            "inline_plotly_runtime": '<script src="https://cdn.plot.ly' not in report,
            "self_contained_images": all(f'data:image/png;base64,' in report for _ in [0]),
            "missing_reference_files": [
                "paper/reference.pdf",
                "paper/reference.pdf",
            ],
        },
    }
    manifest_path = output.with_name("NiaH_Synthetic_report_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def input_label(path: Path) -> str:
    """Keep input hashes usable when run artifacts live outside the checkout."""
    path = path.resolve()
    for label, parent in (("analysis", ANALYSIS), ("run", DATA), ("source", ROOT)):
        if path.is_relative_to(parent.resolve()):
            return label + "/" + path.relative_to(parent.resolve()).as_posix()
    raise ValueError(f"Input is outside the configured report roots: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DATA,
                        help="Completed v58 run containing config.json, manifest.json, and tables/")
    parser.add_argument("--analysis-root", type=Path,
                        help="Supplemental analysis directory; defaults to <run-root>/analysis")
    parser.add_argument("--asset-dir", type=Path,
                        help="Figure directory; defaults next to the output report")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "NiaH_Synthetic_report.html",
    )
    args = parser.parse_args()
    manifest = build_report(args.output.resolve(), run_root=args.run_root,
                            analysis_root=args.analysis_root, asset_dir=args.asset_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
