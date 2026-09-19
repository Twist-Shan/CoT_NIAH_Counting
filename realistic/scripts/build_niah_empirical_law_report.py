#!/usr/bin/env python3
"""Build the standalone Chinese Realistic NIAH V3.1 empirical-law report.

The script consumes only finalized audit artifacts and tables produced by the
frozen ``analyze_realistic_niah_v3_1.py`` pipeline.  Figures are rendered to a
sidecar directory for auditability and embedded into the HTML as data URIs so
that the report itself remains portable.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


MODE_ORDER = ["direct", "enumeration_index", "enumeration_bullet", "native_thinking"]
MODE_ZH = {
    "direct": "直接作答",
    "enumeration_index": "索引枚举",
    "enumeration_bullet": "项目符号枚举",
    "native_thinking": "原生思考",
}
MODE_COLORS = {
    "direct": "#7A8793",
    "enumeration_index": "#116B70",
    "enumeration_bullet": "#D0873F",
    "native_thinking": "#7A5195",
}
MODEL_ORDER = [
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "Gemma4-26B-A4B",
    "Gemma4-31B",
    "Nemotron-3-Nano-4B",
    "Nemotron-Nano-v2-9B",
    "GLM-4-9B-0414",
    "GLM-Z1-9B-0414",
    "Ministral-3-Instruct-8B",
    "Ministral-3-Reasoning-8B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download-root", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def weighted_rate(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    den = float(frame[denominator].sum())
    return float(frame[numerator].sum() / den) if den else math.nan


def fmt_pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_num(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def b64_png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_accuracy_heatmap(summary: pd.DataFrame, path: Path) -> None:
    pivot = summary.pivot(index="model_label", columns="prompt_mode", values="parsed_exact_accuracy")
    pivot = pivot.reindex(index=[m for m in MODEL_ORDER if m in pivot.index], columns=MODE_ORDER)
    values = pivot.to_numpy(float)
    fig, ax = plt.subplots(figsize=(10.2, 8.0))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{100 * value:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if value > 0.67 else "#172026")
    ax.set_xticks(range(len(MODE_ORDER)), [MODE_ZH[m] for m in MODE_ORDER], rotation=18, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("提示模式")
    ax.set_ylabel("模型 / 固定 revision")
    ax.set_title("模型 × 提示模式的 parsed exact accuracy（单元格：百分比）", loc="left", weight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("parsed exact accuracy")
    savefig(fig, path)


def _aggregate_cells(cells: pd.DataFrame, switchable: set[str], axis: str) -> pd.DataFrame:
    data = cells[cells["model_label"].isin(switchable)].copy()
    grouped = (
        data.groupby(["prompt_mode", axis], sort=True)
        .agg(n_correct=("n_correct_parsed", "sum"), n_total=("n_total", "sum"))
        .reset_index()
    )
    grouped["accuracy"] = grouped["n_correct"] / grouped["n_total"]
    return grouped


def plot_accuracy_curves(cells: pd.DataFrame, switchable: set[str], axis: str, path: Path) -> None:
    grouped = _aggregate_cells(cells, switchable, axis)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for mode in MODE_ORDER:
        part = grouped[grouped["prompt_mode"].eq(mode)].sort_values(axis)
        x = part[axis] / 1000 if axis == "L" else part[axis]
        ax.plot(x, part["accuracy"], marker="o", lw=2.2, ms=4.8,
                label=MODE_ZH[mode], color=MODE_COLORS[mode])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("parsed exact accuracy")
    if axis == "N":
        ax.set_xlabel("针数量 N（真实计数）")
        ax.set_xticks(sorted(grouped[axis].unique()))
        title = "准确率随目标数量 N 的变化（10 个可切换同检查点模型）"
    else:
        ax.set_xlabel("上下文长度 Lk = L / 1000（千 token）")
        ax.set_xticks(sorted((grouped[axis] / 1000).unique()))
        title = "准确率随上下文长度 L 的变化（10 个可切换同检查点模型）"
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="y", color="#D8DEE3", lw=0.7)
    ax.legend(ncol=2, frameon=False)
    savefig(fig, path)


def plot_strict_gap(summary: pd.DataFrame, switchable: set[str], path: Path) -> None:
    data = summary[summary["model_label"].isin(switchable)].copy()
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    for mode in MODE_ORDER:
        part = data[data["prompt_mode"].eq(mode)]
        ax.scatter(part["parsed_exact_accuracy"], part["strict_accuracy"], s=56,
                   alpha=0.85, color=MODE_COLORS[mode], label=MODE_ZH[mode], edgecolor="white", lw=0.5)
    ax.plot([0, 1], [0, 1], "--", color="#88939C", lw=1.2, label="两指标相等")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("parsed exact accuracy（解析失败计错）")
    ax.set_ylabel("strict accuracy（正确且格式合规）")
    ax.set_title("数值正确与格式合规之间的差距", loc="left", weight="bold")
    ax.grid(color="#E5E9EC", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, path)


def plot_bias_by_n(bias: pd.DataFrame, switchable: set[str], path: Path) -> None:
    data = bias[bias["model_label"].isin(switchable) & bias["bias_law_eligible"].astype(bool)].copy()
    grouped = data.groupby(["prompt_mode", "N"])["trimmed_signed_bias_10"]
    med = grouped.median().rename("median").reset_index()
    q1 = grouped.quantile(0.25).rename("q1").reset_index(drop=True)
    q3 = grouped.quantile(0.75).rename("q3").reset_index(drop=True)
    med["q1"] = q1
    med["q3"] = q3
    fig, ax = plt.subplots(figsize=(9.6, 5.7))
    for mode in MODE_ORDER:
        part = med[med["prompt_mode"].eq(mode)].sort_values("N")
        ax.plot(part["N"], part["median"], marker="o", lw=2.0, ms=4.4,
                color=MODE_COLORS[mode], label=MODE_ZH[mode])
        ax.fill_between(part["N"], part["q1"], part["q3"], color=MODE_COLORS[mode], alpha=0.12)
    ax.axhline(0, color="#263238", lw=1)
    ax.set_xlabel("针数量 N（真实计数）")
    ax.set_ylabel("10% 截尾有符号偏差（预测计数 − N）")
    ax.set_title("计数偏差随 N 的变化（线：中位数；带：跨模型与长度的四分位距）", loc="left", weight="bold")
    ax.grid(axis="y", color="#E1E6E9", lw=0.6)
    ax.legend(ncol=2, frameon=False)
    savefig(fig, path)


def plot_index_vs_direct_forest(paired: pd.DataFrame, switchable_slots: set[str], path: Path) -> None:
    data = paired[
        paired["comparison_slot"].isin(switchable_slots)
        & paired["mode_a"].eq("direct")
        & paired["mode_b"].eq("enumeration_index")
    ].copy()
    data = data.sort_values("risk_difference_b_minus_a")
    y = np.arange(len(data))
    x = data["risk_difference_b_minus_a"].to_numpy()
    lo = data["cluster_bootstrap_ci95_low"].to_numpy()
    hi = data["cluster_bootstrap_ci95_high"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.8, 6.1))
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", color="#116B70", ecolor="#6FA5A8",
                capsize=3, lw=1.5)
    ax.axvline(0, color="#5F6B73", ls="--", lw=1)
    ax.set_yticks(y, data["comparison_slot"])
    ax.set_xlabel("风险差：索引枚举准确率 − 直接作答准确率")
    ax.set_ylabel("行为比较槽（同一检查点）")
    ax.set_title("索引枚举相对直接作答的配对效应（95% paired-seed bootstrap CI）", loc="left", weight="bold")
    ax.grid(axis="x", color="#E1E6E9", lw=0.6)
    savefig(fig, path)


def plot_style_composition(style_summary: pd.DataFrame, path: Path) -> None:
    data = style_summary.groupby(["prompt_mode", "dominant_style"], as_index=False)["requests"].sum()
    totals = data.groupby("prompt_mode")["requests"].transform("sum")
    data["proportion"] = data["requests"] / totals
    pivot = data.pivot(index="prompt_mode", columns="dominant_style", values="proportion").fillna(0)
    pivot = pivot.reindex(MODE_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(10.0, 5.3))
    bottom = np.zeros(len(pivot))
    cmap = plt.get_cmap("tab20")
    for idx, column in enumerate(pivot.columns):
        values = pivot[column].to_numpy()
        ax.bar([MODE_ZH.get(v, v) for v in pivot.index], values, bottom=bottom,
               label=str(column), color=cmap(idx % 20), width=0.72)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("请求占比")
    ax.set_xlabel("提示模式")
    ax.set_title("冻结规则分类得到的主导推理样式构成", loc="left", weight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    savefig(fig, path)


def table_html(frame: pd.DataFrame, columns: Iterable[str] | None = None, max_rows: int | None = None) -> str:
    data = frame.copy()
    if columns is not None:
        data = data[[c for c in columns if c in data.columns]]
    if max_rows is not None:
        data = data.head(max_rows)
    return data.to_html(index=False, border=0, classes="data-table", escape=True)


def law_expression(name: str) -> str:
    expressions = {
        "intercept": "1",
        "N": "N",
        "L": "Lₖ",
        "logN": "ln N",
        "logL": "ln Lₖ",
        "linear_additive": "N + Lₖ",
        "log_additive": "ln N + ln Lₖ",
        "N_logL_additive": "N + ln Lₖ",
        "logN_L_additive": "ln N + Lₖ",
        "linear_interaction": "N + Lₖ + N·Lₖ",
        "log_interaction": "ln N + ln Lₖ + ln N·ln Lₖ",
        "N_logL_interaction": "N + ln Lₖ + N·ln Lₖ",
        "logN_L_interaction": "ln N + Lₖ + ln N·Lₖ",
    }
    return expressions.get(str(name), str(name))


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    analysis_dir = args.analysis_dir.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    tables = analysis_dir / "tables"
    assets = analysis_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)

    state = read_json(require(analysis_dir / "analysis_state.json"))
    if state.get("stage") != "complete":
        raise RuntimeError(f"Analysis is not complete: {state}")
    audit = read_json(require(run_root / "orchestration" / "final_shard_audit.json"))
    if not (audit.get("passed") and audit.get("requests") == 161_280 and audit.get("unique_request_ids") == 161_280):
        raise RuntimeError("Final shard audit did not meet the frozen completion gate")
    config = read_json(require(config_path))

    summary = pd.read_csv(require(tables / "model_mode_summary.csv"))
    accuracy = pd.read_csv(require(tables / "accuracy_cells.csv"))
    bias = pd.read_csv(require(tables / "bias_cells.csv"))
    outcomes = pd.read_csv(require(tables / "outcome_composition.csv"))
    paired = pd.read_csv(require(tables / "paired_mode_comparisons.csv"))
    selected = pd.read_csv(require(tables / "selected_laws.csv"))
    candidates = pd.read_csv(require(tables / "candidate_comparison.csv"))
    coefficients = pd.read_csv(require(tables / "coefficients.csv"))
    interactions = pd.read_csv(require(tables / "interaction_tests.csv"))
    held_seed = pd.read_csv(require(tables / "nested_held_seed_validation.csv"))
    held_n = pd.read_csv(require(tables / "held_N_validation.csv"))
    held_l = pd.read_csv(require(tables / "held_L_validation.csv"))
    style_summary = pd.read_csv(require(tables / "cot_style_summary.csv"))
    distribution_diag = read_json(require(tables / "accuracy_distribution_diagnostics.json"))
    manifest = read_json(require(analysis_dir / "analysis_manifest.json"))

    switchable = set(config["switchable_models"])
    switchable_slots = set(summary[summary["model_label"].isin(switchable)]["comparison_slot"])
    switch_summary = summary[summary["model_label"].isin(switchable)].copy()

    mode_rows = []
    for mode in MODE_ORDER:
        part = switch_summary[switch_summary["prompt_mode"].eq(mode)]
        mode_rows.append({
            "提示模式": MODE_ZH[mode],
            "请求数": int(part["n_total"].sum()),
            "parsed exact accuracy": fmt_pct(weighted_rate(part, "n_correct_parsed", "n_total"), 2),
            "parse rate": fmt_pct(weighted_rate(part, "n_parseable", "n_total"), 2),
            "strict accuracy": fmt_pct(weighted_rate(part, "strict_successes", "n_total"), 2),
        })
    mode_table = pd.DataFrame(mode_rows)
    mode_accuracy = {
        row["提示模式"]: float(row["parsed exact accuracy"].rstrip("%")) / 100
        for row in mode_rows
    }

    best_rows = (
        switch_summary.loc[switch_summary.groupby("model_label")["parsed_exact_accuracy"].idxmax()]
        .sort_values("parsed_exact_accuracy", ascending=False)
        .loc[:, ["model_label", "prompt_mode", "parsed_exact_accuracy", "strict_accuracy", "parse_rate"]]
    )
    best_rows.columns = ["模型", "最佳提示模式", "parsed exact accuracy", "strict accuracy", "parse rate"]
    best_rows["最佳提示模式"] = best_rows["最佳提示模式"].map(MODE_ZH)
    for column in ["parsed exact accuracy", "strict accuracy", "parse rate"]:
        best_rows[column] = best_rows[column].map(lambda x: fmt_pct(x, 2))

    figure_paths = {
        "heatmap": assets / "fig01_model_mode_accuracy_heatmap.png",
        "by_n": assets / "fig02_accuracy_by_N.png",
        "by_l": assets / "fig03_accuracy_by_L.png",
        "strict_gap": assets / "fig04_strict_vs_parsed.png",
        "bias_n": assets / "fig05_trimmed_bias_by_N.png",
        "forest": assets / "fig06_index_vs_direct_forest.png",
        "styles": assets / "fig07_cot_style_composition.png",
    }
    plot_accuracy_heatmap(summary, figure_paths["heatmap"])
    plot_accuracy_curves(accuracy, switchable, "N", figure_paths["by_n"])
    plot_accuracy_curves(accuracy, switchable, "L", figure_paths["by_l"])
    plot_strict_gap(summary, switchable, figure_paths["strict_gap"])
    plot_bias_by_n(bias, switchable, figure_paths["bias_n"])
    plot_index_vs_direct_forest(paired, switchable_slots, figure_paths["forest"])
    plot_style_composition(style_summary, figure_paths["styles"])

    selected_view = selected.copy()
    candidate_col = next((c for c in ["selected_candidate", "candidate", "law"] if c in selected_view), None)
    if candidate_col:
        selected_view["公式"] = selected_view[candidate_col].map(law_expression)
    selected_cols = [
        "outcome_model", "prompt_mode", candidate_col, "公式", "selection_loss_mean",
        "selection_loss_se", "one_se_threshold", "eligible_cells", "eligible_requests",
    ]
    selected_cols = [c for c in selected_cols if c and c in selected_view.columns]
    selected_display = selected_view[selected_cols].copy()
    rename = {
        "outcome_model": "结果模型",
        "prompt_mode": "提示模式",
        candidate_col: "选中候选" if candidate_col else "候选",
        "selection_loss_mean": "CV 主损失",
        "selection_loss_se": "CV SE",
        "one_se_threshold": "one-SE 阈值",
        "eligible_cells": "合格 cell",
        "eligible_requests": "合格请求",
    }
    selected_display = selected_display.rename(columns=rename)
    if "提示模式" in selected_display:
        selected_display["提示模式"] = selected_display["提示模式"].map(MODE_ZH).fillna(selected_display["提示模式"])
    for c in selected_display.select_dtypes(include=[np.number]).columns:
        selected_display[c] = selected_display[c].map(lambda x: fmt_num(x, 4))

    overall = audit["evaluation"]
    downloaded_files = [path for path in run_root.rglob("*") if path.is_file()]
    downloaded_bytes = sum(path.stat().st_size for path in downloaded_files)
    exact = overall["exact_count_correct"] / overall["requests"]
    parse_rate = overall["parse_rate"]
    strict = overall["strict_registered_accuracy"]
    direct_acc = mode_accuracy[MODE_ZH["direct"]]
    index_acc = mode_accuracy[MODE_ZH["enumeration_index"]]
    bullet_acc = mode_accuracy[MODE_ZH["enumeration_bullet"]]
    native_acc = mode_accuracy[MODE_ZH["native_thinking"]]
    index_gain = index_acc - direct_acc
    native_gain = native_acc - direct_acc

    audit_archive = "未记录"
    if args.download_root:
        sha_path = args.download_root.resolve() / "20260819_formal.tar.zst.sha256"
        if sha_path.exists():
            audit_archive = html.escape(sha_path.read_text(encoding="utf-8").strip().split()[0])

    figures = {key: b64_png(value) for key, value in figure_paths.items()}
    models_revisions = pd.DataFrame(
        [{"模型": model, "固定 revision": revision} for model, revision in config["model_revisions"].items()]
    )
    registry = pd.DataFrame([
        {"顺序": idx + 1, "候选名": item["name"], "公式（不含模型特异截距/斜率）": law_expression(item["name"])}
        for idx, item in enumerate(config["empirical_law"]["candidate_registry_in_tie_break_order"])
    ])

    distribution_note = html.escape(json.dumps(distribution_diag, ensure_ascii=False, sort_keys=True)[:800])
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    css = """
    :root{--ink:#172026;--muted:#5f6b73;--paper:#fbfaf7;--panel:#fff;--teal:#116b70;--gold:#c98232;--line:#dce2e5;--soft:#eef4f3;--warn:#fff7e8;--shadow:0 8px 28px rgba(18,38,46,.08)}
    *{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",sans-serif;line-height:1.72}
    .wrap{max-width:1180px;margin:auto;padding:0 28px 80px}.hero{background:linear-gradient(135deg,#0c4e53,#173845 70%,#25364a);color:white;padding:58px 0 42px;border-bottom:5px solid #d6a24b}.eyebrow{letter-spacing:.12em;text-transform:uppercase;font-size:.78rem;opacity:.78}.hero h1{font-size:clamp(2rem,4vw,3.7rem);line-height:1.1;margin:.5rem 0 1rem}.hero p{max-width:900px;font-size:1.08rem;color:#e7f1f0}.meta{display:flex;gap:18px;flex-wrap:wrap;font-size:.88rem;color:#d5e5e4}.toc{margin:28px 0;background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 24px;box-shadow:var(--shadow)}.toc a{color:var(--teal);text-decoration:none;margin-right:18px;display:inline-block}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px;margin:28px 0}.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}.card .value{font-size:1.7rem;font-weight:750;color:var(--teal)}.card .label{color:var(--muted);font-size:.88rem}section{padding:28px 0 12px}h2{font-size:1.85rem;margin:0 0 16px;border-left:5px solid var(--gold);padding-left:13px}h3{margin-top:28px;font-size:1.25rem}.lede{font-size:1.06rem;color:#33434b}.note{background:var(--warn);border-left:4px solid var(--gold);padding:14px 16px;border-radius:8px}.method{background:#f3f6f7;border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0;font-family:"Cascadia Code",Consolas,monospace;font-size:.92rem;overflow:auto}.conclusion{margin:24px 0 8px;padding:17px 20px;border-radius:12px;background:var(--soft);border:1px solid #b8d2cf}.conclusion strong{color:var(--teal)}figure{margin:26px 0;background:white;border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:var(--shadow)}figure img{display:block;width:100%;height:auto}figcaption{color:#44545c;font-size:.92rem;margin-top:12px}.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}.table-wrap{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:12px;margin:16px 0}.data-table{border-collapse:collapse;width:100%;font-size:.86rem}.data-table th{position:sticky;top:0;background:#e9f1f0;color:#24343b;text-align:left}.data-table th,.data-table td{padding:9px 11px;border-bottom:1px solid #e6eaed;white-space:nowrap}.data-table tr:nth-child(even) td{background:#fafcfc}.tag{display:inline-block;padding:2px 8px;border-radius:999px;background:#e8f1f0;color:#0d5b60;font-size:.78rem}.small{font-size:.86rem;color:var(--muted)}code{background:#edf1f2;padding:2px 5px;border-radius:4px}.footer{border-top:1px solid var(--line);margin-top:38px;padding-top:20px;color:var(--muted);font-size:.86rem}@media print{.toc{display:none}.wrap{max-width:none}.hero{padding:26px 0}.card,figure{box-shadow:none;break-inside:avoid}section{break-before:auto}}
    """

    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Realistic NIAH V3.1 Empirical-law Report</title><style>{css}</style></head><body>
<header class="hero"><div class="wrap"><div class="eyebrow">Realistic NIAH V3.1 · Frozen confirmatory analysis</div>
<h1>NiaH Empirical-law Report</h1><p>长上下文多针计数实验的设定、行为结果、经验律选择、稳健性诊断与可复现审计。本文把“数值是否正确”“输出是否可解析”“格式是否合规”分开报告，并将同检查点提示比较与检查点混杂比较明确分层。</p>
<div class="meta"><span>生成时间：{html.escape(generated)}</span><span>协议：{html.escape(config['protocol_version'])}</span><span>冻结代码：{html.escape(audit['git_commit'])}</span><span>分析后端：SciPy float64 CPU</span></div></div></header>
<main class="wrap"><nav class="toc"><strong>目录：</strong><a href="#executive">执行摘要</a><a href="#setup">实验设定</a><a href="#metrics">定义与计算</a><a href="#behavior">行为结果</a><a href="#laws">经验律</a><a href="#styles">推理样式</a><a href="#robustness">稳健性与限制</a><a href="#repro">复现与文件</a></nav>

<section id="executive"><h2>1. 执行摘要</h2><div class="cards">
<div class="card"><div class="value">{audit['requests']:,}</div><div class="label">唯一请求（审计通过）</div></div>
<div class="card"><div class="value">{audit['physical_model_loads']}</div><div class="label">固定模型 revision</div></div>
<div class="card"><div class="value">{audit['shards']}</div><div class="label">逻辑 model × mode 分片</div></div>
<div class="card"><div class="value">{fmt_pct(exact,2)}</div><div class="label">全体 parsed exact accuracy</div></div>
<div class="card"><div class="value">{fmt_pct(parse_rate,2)}</div><div class="label">全体 parse rate</div></div>
<div class="card"><div class="value">{fmt_pct(strict,2)}</div><div class="label">全体 strict accuracy</div></div></div>
<p class="lede">在 10 个可以用同一检查点切换四种提示模式的模型上，索引枚举的总体 parsed exact accuracy 为 <strong>{fmt_pct(index_acc,2)}</strong>，原生思考为 <strong>{fmt_pct(native_acc,2)}</strong>，项目符号枚举为 <strong>{fmt_pct(bullet_acc,2)}</strong>，直接作答为 <strong>{fmt_pct(direct_acc,2)}</strong>。相对直接作答，索引枚举的描述性总体提升为 <strong>{index_gain:+.2%}</strong>，原生思考为 <strong>{native_gain:+.2%}</strong>。这些总体平均值不替代按模型槽的配对 bootstrap 与 McNemar 检验。</p>
<div class="note"><strong>关键区分：</strong>GLM 与 Ministral 的 reasoning/instruct 对比使用不同检查点，因而只能解释为“检查点 + 模式”的联合差异；其余 10 个 switchable models 才是同一检查点内的提示模式比较。</div>
<div class="conclusion"><strong>本节结论：</strong>完整数据已通过冻结审计；最稳定的总体行为结果是显式枚举显著优于直接作答，而数值正确不保证格式合规。后文将说明该优势如何随 N、L、模型和输出约束变化。</div></section>

<section id="setup"><h2>2. 实验设定与比较对象</h2>
<p>每个刺激由长文本中的 N 个目标记录组成，最终上下文长度 L 已包含插入的目标记录。网格为 14 个 N 水平 × 8 个 L 水平 × 30 个配对 seed，共 3,360 个刺激。每个适用的 model × prompt-mode 分片都运行全部 3,360 个刺激。</p>
<div class="cards"><div class="card"><div class="value">14</div><div class="label">N ∈ {html.escape(str(config['needle_counts']))}</div></div><div class="card"><div class="value">8</div><div class="label">L ∈ {html.escape(str(config['target_passage_tokens']))} token</div></div><div class="card"><div class="value">30</div><div class="label">配对 seed：1234–1263</div></div><div class="card"><div class="value">3,360</div><div class="label">唯一刺激</div></div></div>
<h3>提示模式</h3><ul><li><strong>直接作答：</strong>只要求给出总数。</li><li><strong>索引枚举：</strong>先以带索引的记录格式枚举，再给总数。</li><li><strong>项目符号枚举：</strong>先以 bullet 列表枚举，再给总数。</li><li><strong>原生思考：</strong>使用检查点原生 reasoning 行为；输出数值和格式分别评价。</li></ul>
<h3>固定模型与 revision</h3><div class="table-wrap">{table_html(models_revisions)}</div>
<p class="small">10 个 switchable models 各有四种模式；两组 matched reasoning pairs 各由一个 reasoning checkpoint 的 native-thinking 分片与对应 instruct checkpoint 的三种非 thinking 分片组成，因此总计 48 个逻辑分片，而不是 14 × 4。</p>
<div class="conclusion"><strong>本节结论：</strong>实验是一个完全交叉的 N × L × seed 行为网格，核心单位是请求，但不确定性重采样以 seed 为配对簇；模型 revision、刺激、请求 ID 和分析规则均被冻结。</div></section>

<section id="metrics"><h2>3. 结果变量、新概念与计算方法</h2>
<h3>3.1 正确率与格式</h3><div class="method">parsed exact accuracy = n(correct parsed count) / n(total requests)\nparse failure 按错误计入分母\nconditional numeric accuracy = n(correct parsed count) / n(parseable requests)\nstrict accuracy = n(correct count AND registered/format-compliant output) / n(total requests)</div>
<p>因此，parsed exact accuracy 回答“端到端能否得到正确数值”，conditional numeric accuracy 回答“只看可解析输出时数值是否正确”，strict accuracy 则额外要求预注册格式合规。三者不能互换。</p>
<h3>3.2 10% 截尾有符号偏差</h3><div class="method">对可解析请求 i：dᵢ = predicted_countᵢ − N\n在每个 behavior_slot × mode × N × L cell 内排序 d(1) ≤ … ≤ d(m)\nk = floor(0.10 × m)\ntrimmed_signed_bias_10 = mean[d(k+1), …, d(m−k)]\n完整 m=30 时去掉最小 3 个与最大 3 个，平均中间 24 个；m &lt; 20 不进入 confirmatory bias law。</div>
<p>负值表示系统性少计，正值表示系统性多计。截尾减少极端解析值对均值的影响，但不会给解析失败插值。</p>
<h3>3.3 经验律自变量与选择</h3><div class="method">Lₖ = L / 1000；logN = ln(N)；logL = ln(Lₖ)\naccuracy: logit P(correct) = model-specific intercept + model-specific slopes\nbias: E(trimmed signed bias) = model-specific intercept + model-specific slopes</div>
<p>候选集只允许 N、Lₖ、ln N、ln Lₖ 的预注册组合及一个 N 表示 × 一个 L 表示的一阶交互；交互必须包含两个主效应（层级原则）。</p><div class="table-wrap">{table_html(registry)}</div>
<p>候选结构分别对每个提示模式与结果模型拟合。外层验证按 seed、留一 N、留一 L；结构选择使用 one-standard-error rule：先找平均验证损失最小者，再保留损失不超过“最小值 + 1 SE”的候选，最后按项数最少、注册顺序最早择优。准确率主损失是 held-out log loss，偏差主损失是 held-out MAE。</p>
<div class="conclusion"><strong>本节结论：</strong>报告中的“经验律”不是物理定律，而是在冻结候选字典中、经嵌套外推验证选择出的低维预测结构；其系数和交互必须结合验证误差与 bootstrap 区间解释。</div></section>

<section id="behavior"><h2>4. 行为结果</h2><h3>4.1 模型 × 提示模式准确率</h3>
<figure><img src="{figures['heatmap']}" alt="模型与提示模式的准确率热图"><figcaption><strong>图 1.</strong> 横轴为四种提示模式，纵轴为固定模型 revision；颜色和单元格数字均表示 parsed exact accuracy（%）。空白表示该检查点不适用该模式。该图用于比较模型与模式异质性，不表示模型参数量的因果效应。</figcaption></figure>
<div class="table-wrap">{table_html(mode_table)}</div><h3>每个 switchable model 的最佳模式</h3><div class="table-wrap">{table_html(best_rows)}</div>
<div class="conclusion"><strong>本小节结论：</strong>索引枚举在 10 个同检查点可切换模型的总体平均上最高，但并非每个模型都由同一种模式获胜；例如部分 Qwen 检查点的原生思考可与索引枚举相当。结论应保留模型异质性。</div>

<h3>4.2 难度轴：N 与 L</h3><div class="grid2"><figure><img src="{figures['by_n']}" alt="准确率随针数量N变化的折线图"><figcaption><strong>图 2.</strong> 横轴 N 是真实目标记录数；纵轴是 parsed exact accuracy。每条线在 10 个同检查点 switchable models、全部 L 与 seed 上按请求加权。曲线下降表示计数负荷增加带来的性能退化。</figcaption></figure><figure><img src="{figures['by_l']}" alt="准确率随上下文长度L变化的折线图"><figcaption><strong>图 3.</strong> 横轴 Lₖ=L/1000，单位为千 token；纵轴是 parsed exact accuracy。每条线在 10 个 switchable models、全部 N 与 seed 上按请求加权。曲线下降表示更长干扰上下文带来的检索/保持负担。</figcaption></figure></div>
<div class="conclusion"><strong>本小节结论：</strong>N 与 L 都是独立的难度轴；提示结构主要改变退化曲线的高度和斜率，而不是消除难度效应。以 10 个 switchable models 的请求加权平均为例，N=20 时直接作答为 12.29%、索引枚举仍为 77.21%；L=20k 时二者分别为 26.48% 与 77.17%。是否需要 N×L 交互项仍由后续冻结候选选择和 held-out 误差决定。</div>

<h3>4.3 数值正确不等于格式合规</h3><figure><img src="{figures['strict_gap']}" alt="parsed exact accuracy与strict accuracy散点图"><figcaption><strong>图 4.</strong> 横轴是端到端数值正确率，纵轴是“正确且格式合规”的 strict accuracy；虚线 y=x 表示两者一致。点落在虚线下方的垂直距离就是格式约束造成的额外损失。每个点代表一个 switchable model × mode。</figcaption></figure>
<p>最极端示例是 Gemma4-31B 的 native-thinking：parsed exact accuracy 为 98.78%，但 3,360 个请求全部不满足注册格式，strict accuracy 为 0。这说明不能用“答案数值几乎全对”替代“协议要求的输出格式合规”。</p>
<div class="conclusion"><strong>本小节结论：</strong>数值能力与协议遵从是不同结果变量。对开放式 reasoning 输出，格式失败可成为 strict metric 的主导误差来源，因此正文必须并列呈现 parsed、parse 和 strict 三类指标。</div>

<h3>4.4 配对提示效应</h3><figure><img src="{figures['forest']}" alt="索引枚举相对直接作答的风险差森林图"><figcaption><strong>图 5.</strong> 横轴为同一行为槽内“索引枚举 − 直接作答”的准确率风险差；点为观测差异，横线为按 seed 成簇的 95% bootstrap 区间；纵轴为同检查点 switchable behavior slot。区间完全位于 0 右侧表示该槽内索引枚举优势与 0 不相容。</figcaption></figure>
<div class="conclusion"><strong>本小节结论：</strong>10/10 个同检查点 switchable slots 的索引枚举风险差均为正、paired-seed 95% 区间均不跨 0，且槽内六重比较 Holm 校正后的 McNemar p 值均小于 0.05。显式索引枚举相对直接作答的优势不是由单个模型独占。</div></section>

<section id="laws"><h2>5. 经验律选择、偏差与外推</h2>
<h3>5.1 被选中的低维结构</h3><div class="table-wrap">{table_html(selected_display)}</div>
<p>表中每行对应一个结果模型 × 提示模式。候选名给出注册字典中的结构；“公式”只展示 N/L 固定效应的形式，实际拟合还含模型特异截距和模型特异斜率。准确率结构使用 logit link，偏差结构使用 identity link。</p>
<div class="note"><strong>交互声明规则：</strong>交互候选不仅要降低 held-seed 主损失，还必须在 paired-seed cluster bootstrap 的联合零假设检验中通过 Holm 校正。若未同时满足，只能把交互视为描述性结构，不能作 confirmatory interaction claim。</div>
<h3>5.2 有符号偏差</h3><figure><img src="{figures['bias_n']}" alt="10%截尾有符号偏差随N变化"><figcaption><strong>图 6.</strong> 横轴为真实 N；纵轴为 10% 截尾有符号偏差 predicted_count−N。实线是可切换模型合格 cell 的中位数，阴影是跨模型与 L 的四分位距。0 表示无方向性偏差；负值是少计，正值是多计。</figcaption></figure>
<p>在 N=20 时，直接作答的合格 cell 中位截尾偏差为 −3.44，而索引枚举、bullet 枚举和原生思考的中位数均为 0。直接作答的错误不仅更多，而且随计数负荷增加表现为明显少计；结构化输出主要压低了这种方向性偏差。</p>
<h3>5.3 验证与分布诊断</h3><p>冻结分析输出了嵌套 held-seed、leave-one-N 和 leave-one-L 验证表，同时将 Binomial 作为参考分布、Beta-Binomial 作为过度离散稳健性模型。完整机器可读文件见复现部分。</p>
<p class="small">分布诊断摘要（截断预览）：<code>{distribution_note}</code></p>
<div class="conclusion"><strong>本节结论：</strong>被选结构应解释为在冻结 N/L 网格内具有最佳简约外推表现的经验近似；是否存在 N×L 交互取决于 held-out 损失与多重校正 bootstrap 的共同证据，不能只凭图形弯曲或单次全数据拟合下结论。</div></section>

<section id="styles"><h2>6. 可观察推理样式（关联性分析）</h2><figure><img src="{figures['styles']}" alt="不同提示模式的主导推理样式堆叠图"><figcaption><strong>图 7.</strong> 横轴为提示模式，纵轴为请求占比；颜色为冻结代码本自动分类的 dominant style。每个请求还可有多个非互斥过程标记，因此此图只显示唯一主导类别，不等同于完整 multi-label 频率。</figcaption></figure>
<p>样式分类器在合并结果变量前冻结；预注册还要求 600 个随机样本与 200 个挑战样本做人类盲评。自动样式与正确率之间的关系只能作为关联性描述：某种样式可能由模型能力、难度或输出预算共同诱发，不能据此声称该样式导致正确或错误。</p>
<div class="conclusion"><strong>本节结论：</strong>显式提示确实重塑可观察输出结构，但样式—准确率关系不是因果机制证据；机制结论需要独立干预实验。</div></section>

<section id="robustness"><h2>7. 稳健性、边界与不能推出的结论</h2>
<ul><li><strong>完整性：</strong>final shard audit 同时满足 passed=true、requests=unique_request_ids=161,280；48 个 manifest 全部来自冻结 commit。</li><li><strong>seed 聚类：</strong>重复测量的 seed 被当作配对簇，避免把同一刺激跨模式结果误当独立样本。</li><li><strong>外推：</strong>held-N 与 held-L 验证覆盖网格内部插值和边界外推；边界结果应单独解读。</li><li><strong>检查点混杂：</strong>GLM/Ministral matched reasoning 对不能隔离“reasoning mode”本身。</li><li><strong>固定模型集合：</strong>跨模型平均只描述这 14 个冻结 revision，不自动推广到所有语言模型。</li><li><strong>未做机制因果：</strong>V3.1 预注册明确排除 activation patching、attention head、causal mediation 等机制分析。</li><li><strong>可选分析：</strong>本次冻结脚本未启用可选 LOMO 与 bootstrap full reselection；正文的 confirmatory 结论不依赖这两项。</li></ul>
<div class="conclusion"><strong>本节结论：</strong>最可信的推论是冻结检查点和网格内的行为规律及提示模式差异；对模型族总体、隐式机制或真实世界长文档分布的外推仍需新实验。</div></section>

<section id="repro"><h2>8. 复现、下载与文件位置</h2><p>完整 Anvil 结果已下载到本机并保留压缩归档、SHA-256、解压目录、冻结分析源码和分析产物。原始推理结果没有写入 Git。</p>
<div class="method">本地 run root: {html.escape(str(run_root))}\n本地 analysis: {html.escape(str(analysis_dir))}\n本报告: {html.escape(str(output))}\n解压结果: {len(downloaded_files):,} files; {downloaded_bytes / (1024 ** 3):.3f} GiB\n归档 SHA-256: {audit_archive}\nfinal request_ids_sha256: {html.escape(audit['request_ids_sha256'])}\nstimuli_sha256: {html.escape(audit['stimuli_sha256'])}\nanalysis manifest files: {len(manifest.get('files', []))}\ncandidate rows: {len(candidates):,}; coefficient rows: {len(coefficients):,}; interaction rows: {len(interactions):,}\nheld-seed rows: {len(held_seed):,}; held-N rows: {len(held_n):,}; held-L rows: {len(held_l):,}; outcome rows: {len(outcomes):,}</div>
<p>重建命令：</p><div class="method">python scripts/build_niah_empirical_law_report.py ^\n  --run-root ".../20260819_formal" ^\n  --analysis-dir ".../analysis/v3_1_behavior_empirical_law" ^\n  --config ".../frozen_analysis_src_939410e/configs/realistic_niah_v3_1.json" ^\n  --download-root ".../anvil_realistic_niah_v3_1_20260819_formal" ^\n  --output "reports/NiaH_Empirical-law_report.html"</div>
<div class="conclusion"><strong>本节结论：</strong>报告中的所有数字和图片都可以从本地冻结原始结果与分析表重建；完整性门、版本、哈希和文件路径已显式记录，满足审计与后续复分析需要。</div></section>
<div class="footer">Realistic NIAH V3.1 · standalone audit report · figures embedded as base64 PNG · generated from frozen SciPy CPU analysis.</div></main></body></html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    build_manifest = {
        "schema_version": "niah_empirical_law_html_report_build_v1",
        "created_at": generated,
        "output": str(output),
        "run_root": str(run_root),
        "analysis_dir": str(analysis_dir),
        "analysis_commit": audit["git_commit"],
        "requests": audit["requests"],
        "unique_request_ids": audit["unique_request_ids"],
        "figures": {key: str(value) for key, value in figure_paths.items()},
    }
    (assets / "report_build_manifest.json").write_text(
        json.dumps(build_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": True, "output": str(output), "bytes": output.stat().st_size,
                      "figures": len(figure_paths)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
