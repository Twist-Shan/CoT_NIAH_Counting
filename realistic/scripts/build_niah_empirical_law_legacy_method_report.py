#!/usr/bin/env python3
"""Build the Chinese V3.1 report using the archived focused-law method."""

from __future__ import annotations

import argparse
import base64
import html
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_ORDER = [
    "direct",
    "enumeration_index",
    "enumeration_bullet",
    "native_thinking",
]
MODE_ZH = {
    "direct": "直接作答",
    "enumeration_index": "索引枚举",
    "enumeration_bullet": "项目符号枚举",
    "native_thinking": "原生思考",
}
MODE_COLORS = {
    "direct": "#73808A",
    "enumeration_index": "#0F6B70",
    "enumeration_bullet": "#C98335",
    "native_thinking": "#76558E",
}
FAMILY_ORDER = [
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
    "GLM 9B pair",
    "Ministral 8B pair",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--behavior-analysis-dir", required=True, type=Path)
    parser.add_argument("--law-analysis-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--support-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def fmt_pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_num(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def weighted_rate(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    denominator_value = float(frame[denominator].sum())
    return (
        float(frame[numerator].sum() / denominator_value)
        if denominator_value
        else math.nan
    )


def b64_png(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:image/png;base64," + encoded


def table_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="data-table", escape=True)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def predictor_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = frame["N"].to_numpy(dtype=float)
    lk = frame["L"].to_numpy(dtype=float) / 1000.0
    lnn = np.log(n)
    lnl = np.log(lk)
    return {
        "N": n,
        "L": lk,
        "lnN": lnn,
        "lnL": lnl,
        "density": n / lk,
        "N_x_L": n * lk,
        "N_x_lnL": n * lnl,
        "lnN_x_L": lnn * lk,
        "lnN_x_lnL": lnn * lnl,
    }


def predict_selected(
    grid: pd.DataFrame,
    coefficient_block: pd.DataFrame,
) -> np.ndarray:
    values = predictor_columns(grid)
    intercept = float(
        coefficient_block.loc[
            coefficient_block["term"].eq("intercept"), "coefficient"
        ].iloc[0]
    )
    prediction = np.full(len(grid), intercept)
    for row in coefficient_block.itertuples(index=False):
        if row.term != "intercept":
            prediction += float(row.coefficient) * values[row.term]
    return prediction


def plot_selected_mode(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    selected_fits: pd.DataFrame,
    mode: str,
    path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    levels = sorted(cells["L"].unique())
    color_values = plt.get_cmap("viridis")(
        np.linspace(0.06, 0.94, len(levels))
    )
    colors = dict(zip(levels, color_values))
    n_grid = np.linspace(float(cells["N"].min()), float(cells["N"].max()), 240)
    fig, axes = plt.subplots(3, 4, figsize=(16.8, 12.0), sharex=True)
    for index, family in enumerate(FAMILY_ORDER):
        axis = axes.flat[index]
        cell_block = cells[
            cells["analysis_family"].eq(family) & cells["mode"].eq(mode)
        ]
        coefficient_block = coefficients[
            coefficients["analysis_family"].eq(family)
            & coefficients["mode"].eq(mode)
        ]
        for level in levels:
            observed = cell_block[cell_block["L"].eq(level)].sort_values("N")
            grid = pd.DataFrame({"N": n_grid, "L": level})
            axis.scatter(
                observed["N"],
                observed["signed_mean_deviation"],
                s=18,
                alpha=0.78,
                color=colors[level],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            axis.plot(
                n_grid,
                predict_selected(grid, coefficient_block),
                color=colors[level],
                lw=1.35,
            )
        fit = selected_fits[
            selected_fits["analysis_family"].eq(family)
            & selected_fits["mode"].eq(mode)
        ].iloc[0]
        axis.axhline(0, color="#4F5B62", lw=0.8)
        axis.set_title(
            f"{family}\nCV R²={fit.cv_r2:.2f}, MAE={fit.cv_mae:.2f}",
            fontsize=9.2,
        )
        axis.grid(color="#E2E7E9", lw=0.55)
        if index // 4 == 2:
            axis.set_xlabel("真实针数量 N")
        if index % 4 == 0:
            axis.set_ylabel("条件平均有符号偏差")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=colors[level],
            lw=1.3,
            label=f"L={level // 1000}k",
            markersize=5,
        )
        for level in levels
    ]
    fig.legend(handles=handles, loc="upper center", ncol=8, frameon=False)
    fig.suptitle(
        f"{MODE_ZH[mode]}：观测条件均值与所选共享结构的模型特异拟合",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, path)


def plot_lomo(choices: pd.DataFrame, path: Path) -> None:
    data = choices.set_index("mode").reindex(MODE_ORDER)
    values = data["lomo_formula_stability"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.bar(
        [MODE_ZH[item] for item in MODE_ORDER],
        values,
        color=[MODE_COLORS[item] for item in MODE_ORDER],
        width=0.66,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            weight="bold",
        )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("LOMO 结构一致率")
    ax.set_xlabel("提示模式")
    ax.set_title("留一模型后是否仍选择同一共享结构", loc="left", weight="bold")
    ax.grid(axis="y", color="#E1E6E9", lw=0.6)
    savefig(fig, path)


def plot_interaction_effect(coefficients: pd.DataFrame, path: Path) -> None:
    data = coefficients[coefficients["term"].eq("N_x_L")]
    pivot = data.pivot(
        index="analysis_family",
        columns="mode",
        values="standardized_effect",
    ).reindex(index=FAMILY_ORDER, columns=MODE_ORDER)
    values = pivot.to_numpy(dtype=float)
    bound = max(float(np.nanquantile(np.abs(values), 0.95)), 0.25)
    fig, ax = plt.subplots(figsize=(8.6, 7.3))
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        aspect="auto",
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    ax.set_xticks(
        range(len(MODE_ORDER)),
        [MODE_ZH[item] for item in MODE_ORDER],
        rotation=18,
        ha="right",
    )
    ax.set_yticks(range(len(FAMILY_ORDER)), FAMILY_ORDER)
    ax.set_xlabel("提示模式")
    ax.set_ylabel("分析模型族")
    ax.set_title("所选 N×Lk 交互项的标准化效应", loc="left", weight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("标准化交互效应（带符号）")
    savefig(fig, path)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    behavior_dir = args.behavior_analysis_dir.resolve()
    law_dir = args.law_analysis_dir.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    support = load_module(
        "niah_report_support",
        args.support_script.resolve(),
    )

    audit = read_json(
        require(run_root / "orchestration" / "final_shard_audit.json")
    )
    if not (
        audit.get("passed")
        and audit.get("requests") == 161_280
        and audit.get("unique_request_ids") == 161_280
    ):
        raise RuntimeError("Frozen result audit failed")
    law_state = read_json(require(law_dir / "analysis_state.json"))
    if law_state.get("stage") != "complete":
        raise RuntimeError(f"Focused-law analysis is incomplete: {law_state}")
    law_manifest = read_json(require(law_dir / "analysis_manifest.json"))
    config = read_json(require(config_path))

    behavior_tables = behavior_dir / "tables"
    law_tables = law_dir / "tables"
    summary = pd.read_csv(require(behavior_tables / "model_mode_summary.csv"))
    accuracy = pd.read_csv(require(behavior_tables / "accuracy_cells.csv"))
    bias = pd.read_csv(require(behavior_tables / "bias_cells.csv"))
    paired = pd.read_csv(require(behavior_tables / "paired_mode_comparisons.csv"))
    cells = pd.read_csv(require(law_tables / "condition_signed_bias.csv"))
    choices = pd.read_csv(require(law_tables / "selected_mode_laws.csv"))
    selected_fits = pd.read_csv(
        require(law_tables / "selected_model_fit_metrics.csv")
    )
    coefficients = pd.read_csv(
        require(law_tables / "selected_model_coefficients.csv")
    )
    mapping = pd.read_csv(require(law_tables / "model_mode_mapping.csv"))
    lomo = pd.read_csv(require(law_tables / "lomo_structure_selection.csv"))

    switchable = set(config["switchable_models"])
    switch_summary = summary[summary["model_label"].isin(switchable)]
    switchable_slots = set(switch_summary["comparison_slot"])

    mode_rows: list[dict[str, object]] = []
    for mode in MODE_ORDER:
        part = switch_summary[switch_summary["prompt_mode"].eq(mode)]
        mode_rows.append(
            {
                "提示模式": MODE_ZH[mode],
                "请求数": int(part["n_total"].sum()),
                "parsed exact accuracy": fmt_pct(
                    weighted_rate(part, "n_correct_parsed", "n_total"),
                    2,
                ),
                "parse rate": fmt_pct(
                    weighted_rate(part, "n_parseable", "n_total"),
                    2,
                ),
                "strict accuracy": fmt_pct(
                    weighted_rate(part, "strict_successes", "n_total"),
                    2,
                ),
            }
        )
    mode_table = pd.DataFrame(mode_rows)

    law_table = choices[
        [
            "mode",
            "selected_formula_label",
            "median_cv_r2",
            "q25_cv_r2",
            "median_cv_mae",
            "median_cv_r2_gain_over_parent",
            "special_term_significant_fraction",
            "special_cv_gain_q",
            "lomo_formula_stability",
            "evidence_reading",
        ]
    ].copy()
    law_table["mode"] = law_table["mode"].map(MODE_ZH)
    law_table["evidence_reading"] = law_table["evidence_reading"].map(
        {
            "Strong cross-model support": "强跨模型支持",
            "Tentative cross-model support": "暂定跨模型支持",
            "No reliable shared law": "无可靠共享律",
        }
    ).fillna(law_table["evidence_reading"])
    law_table.columns = [
        "提示模式",
        "选中共享结构",
        "中位 CV R²",
        "Q25 CV R²",
        "中位 CV MAE",
        "相对主效应中位 ΔCV R²",
        "交互项显著模型比例",
        "交互增益 BH q",
        "LOMO 结构一致率",
        "证据分级",
    ]
    numeric_columns = [
        "中位 CV R²",
        "Q25 CV R²",
        "中位 CV MAE",
        "相对主效应中位 ΔCV R²",
        "交互增益 BH q",
    ]
    for column in numeric_columns:
        law_table[column] = law_table[column].map(
            lambda value: fmt_num(value, 3)
        )
    for column in ["交互项显著模型比例", "LOMO 结构一致率"]:
        law_table[column] = law_table[column].map(
            lambda value: fmt_pct(value, 1)
        )

    revisions = (
        mapping[["source_model", "source_version"]]
        .drop_duplicates()
        .sort_values("source_model")
    )
    revisions.columns = ["物理模型", "固定 revision"]

    assets = law_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "accuracy_heatmap": assets / "fig01_model_mode_accuracy.png",
        "accuracy_n": assets / "fig02_accuracy_by_N.png",
        "accuracy_l": assets / "fig03_accuracy_by_L.png",
        "strict_gap": assets / "fig04_strict_gap.png",
        "paired_forest": assets / "fig05_index_direct_forest.png",
        "bias_n": assets / "fig06_bias_by_N.png",
        "lomo": assets / "fig07_lomo_stability.png",
        "interaction": assets / "fig08_interaction_effect.png",
    }
    support.plot_accuracy_heatmap(summary, figure_paths["accuracy_heatmap"])
    support.plot_accuracy_curves(
        accuracy, switchable, "N", figure_paths["accuracy_n"]
    )
    support.plot_accuracy_curves(
        accuracy, switchable, "L", figure_paths["accuracy_l"]
    )
    support.plot_strict_gap(
        summary, switchable, figure_paths["strict_gap"]
    )
    support.plot_index_vs_direct_forest(
        paired, switchable_slots, figure_paths["paired_forest"]
    )
    support.plot_bias_by_n(bias, switchable, figure_paths["bias_n"])
    plot_lomo(choices, figure_paths["lomo"])
    plot_interaction_effect(coefficients, figure_paths["interaction"])
    for mode in MODE_ORDER:
        path = assets / f"fig_selected_{mode}.png"
        figure_paths[f"selected_{mode}"] = path
        plot_selected_mode(
            cells,
            coefficients,
            selected_fits,
            mode,
            path,
        )
    figures = {key: b64_png(path) for key, path in figure_paths.items()}

    overall = audit["evaluation"]
    overall_exact = overall["exact_count_correct"] / overall["requests"]
    generated = (
        datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    )
    css = """
    :root{--ink:#172026;--muted:#5f6b73;--paper:#fbfaf7;--teal:#116b70;--gold:#c98232;--line:#dce2e5;--soft:#eef4f3;--warn:#fff7e8;--shadow:0 8px 28px rgba(18,38,46,.08)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",sans-serif;line-height:1.72}.wrap{max-width:1180px;margin:auto;padding:0 28px 80px}.hero{background:linear-gradient(135deg,#0c4e53,#173845 70%,#25364a);color:white;padding:58px 0 42px;border-bottom:5px solid #d6a24b}.eyebrow{letter-spacing:.12em;text-transform:uppercase;font-size:.78rem;opacity:.78}.hero h1{font-size:clamp(2rem,4vw,3.7rem);line-height:1.1;margin:.5rem 0 1rem}.hero p{max-width:920px;font-size:1.08rem;color:#e7f1f0}.meta{display:flex;gap:18px;flex-wrap:wrap;font-size:.88rem;color:#d5e5e4}.toc{margin:28px 0;background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 24px;box-shadow:var(--shadow)}.toc a{color:var(--teal);text-decoration:none;margin-right:18px;display:inline-block}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px;margin:28px 0}.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}.card .value{font-size:1.7rem;font-weight:750;color:var(--teal)}.card .label{color:var(--muted);font-size:.88rem}section{padding:28px 0 12px}h2{font-size:1.85rem;margin:0 0 16px;border-left:5px solid var(--gold);padding-left:13px}h3{margin-top:28px;font-size:1.25rem}.lede{font-size:1.06rem;color:#33434b}.note{background:var(--warn);border-left:4px solid var(--gold);padding:14px 16px;border-radius:8px}.method{background:#f3f6f7;border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0;font-family:"Cascadia Code",Consolas,monospace;font-size:.9rem;white-space:pre-wrap;overflow:auto}.conclusion{margin:24px 0 8px;padding:17px 20px;border-radius:12px;background:var(--soft);border:1px solid #b8d2cf}.conclusion strong{color:var(--teal)}figure{margin:26px 0;background:white;border:1px solid var(--line);border-radius:15px;padding:16px;box-shadow:var(--shadow)}figure img{display:block;width:100%;height:auto}figcaption{color:#44545c;font-size:.92rem;margin-top:12px}.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}.table-wrap{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:12px;margin:16px 0}.data-table{border-collapse:collapse;width:100%;font-size:.84rem}.data-table th{position:sticky;top:0;background:#e9f1f0;color:#24343b;text-align:left}.data-table th,.data-table td{padding:9px 11px;border-bottom:1px solid #e6eaed;white-space:nowrap}.data-table tr:nth-child(even) td{background:#fafcfc}.small{font-size:.86rem;color:var(--muted)}code{background:#edf1f2;padding:2px 5px;border-radius:4px}.footer{border-top:1px solid var(--line);margin-top:38px;padding-top:20px;color:var(--muted);font-size:.86rem}@media print{.toc{display:none}.wrap{max-width:none}.hero{padding:26px 0}.card,figure{box-shadow:none;break-inside:avoid}}
    """

    selected_figures = "".join(
        f"""<figure><img src="{figures[f'selected_{mode}']}" alt="{MODE_ZH[mode]}的经验律拟合图"><figcaption><strong>图 {9 + index}.</strong> {MODE_ZH[mode]}。横轴为真实针数量 N，纵轴为该 N×L 条件中可解析请求的平均有符号偏差（预测数−N）；点为观测条件均值，线为模型特异系数下的所选共享结构预测，不同颜色代表八个 L 水平。每个面板对应一个分析模型族；标题同时报告五折 held-condition CV R² 与 MAE。</figcaption></figure>"""
        for index, mode in enumerate(MODE_ORDER)
    )
    direct_r2 = float(
        choices.loc[choices["mode"].eq("direct"), "median_cv_r2"].iloc[0]
    )
    native_r2 = float(
        choices.loc[
            choices["mode"].eq("native_thinking"), "median_cv_r2"
        ].iloc[0]
    )
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NiaH Empirical-law Report</title><style>{css}</style></head><body>
<header class="hero"><div class="wrap"><div class="eyebrow">Realistic NIAH V3.1 · audited behavior grid</div><h1>NiaH Empirical-law Report</h1><p>完整长上下文多针计数实验的设定、行为结果与经验律分析。经验律部分严格复用既有 focused empirical-law 方法，只把模型集合与 N/L 网格更新为 V3.1；未使用后来加入且计算昂贵的嵌套 bootstrap 设计。</p><div class="meta"><span>生成时间：{html.escape(generated)}</span><span>协议：{html.escape(config['protocol_version'])}</span><span>冻结推理 commit：{html.escape(audit['git_commit'])}</span><span>经验律后端：NumPy/SciPy CPU</span></div></div></header>
<main class="wrap"><nav class="toc"><strong>目录：</strong><a href="#summary">摘要</a><a href="#setup">实验设定</a><a href="#metrics">定义与方法</a><a href="#behavior">行为结果</a><a href="#laws">经验律</a><a href="#limits">稳健性与限制</a><a href="#repro">复现</a></nav>

<section id="summary"><h2>1. 执行摘要</h2><div class="cards"><div class="card"><div class="value">{audit['requests']:,}</div><div class="label">唯一请求</div></div><div class="card"><div class="value">14</div><div class="label">固定物理模型 revision</div></div><div class="card"><div class="value">48</div><div class="label">逻辑 model-mode 槽</div></div><div class="card"><div class="value">{fmt_pct(overall_exact,2)}</div><div class="label">全体 parsed exact accuracy</div></div><div class="card"><div class="value">26</div><div class="label">候选经验律</div></div><div class="card"><div class="value">1,248</div><div class="label">候选拟合</div></div></div><p class="lede">四种模式均选择层级交互结构 <strong>N + Lₖ + N×Lₖ</strong>。但证据强度不相同：直接作答与原生思考的 LOMO 结构一致率为 100%，索引枚举为 83.3%，项目符号枚举仅为 50%。因此，“同一结构被选中”不等于“结构同样稳定”。</p><div class="conclusion"><strong>本节结论：</strong>完整推理数据与旧式经验律分析均已通过审计；最稳妥的总括是 N 与 L 的联合效应广泛存在，但 bullet 模式的共享结构仍明显依赖具体模型集合。</div></section>

<section id="setup"><h2>2. 实验设定与比较单位</h2><p>实验使用 14 个 N 水平、8 个 L 水平与 30 个配对 seed。每个适用的物理模型×提示模式都覆盖 14×8×30=3,360 个请求，共 161,280 个唯一请求。L 是包含目标记录后的最终上下文长度；经验律中使用 Lₖ=L/1000。</p><div class="cards"><div class="card"><div class="value">14</div><div class="label">N={html.escape(str(config['needle_counts']))}</div></div><div class="card"><div class="value">8</div><div class="label">L={html.escape(str(config['target_passage_tokens']))}</div></div><div class="card"><div class="value">30</div><div class="label">每个 N×L 单元的 seed</div></div><div class="card"><div class="value">5,376</div><div class="label">model-family×mode×N×L 单元</div></div></div><p>10 个模型的同一 checkpoint 支持四种提示模式。GLM 与 Ministral 分别合成为一个“分析模型族”：三种非 thinking 模式来自 instruct checkpoint，native-thinking 来自配对 reasoning checkpoint。这与旧报告对 GLM 的处理一致，但这种跨 checkpoint 比较不能解释成纯提示因果效应。</p><div class="table-wrap">{table_html(revisions)}</div><div class="conclusion"><strong>本节结论：</strong>网格在每个逻辑槽内完整平衡；同 checkpoint 的模式比较可作提示效应描述，而 GLM/Ministral 的 native 对比同时包含 checkpoint 差异。</div></section>

<section id="metrics"><h2>3. 指标、经验律与计算方法</h2><h3>3.1 行为指标</h3><div class="method">parsed exact accuracy = 正确解析计数 / 全部请求（解析失败计错）
parse rate = 可解析请求 / 全部请求
strict accuracy = 数值正确且满足注册格式 / 全部请求
有符号偏差 d = predicted_count − N（只在可解析请求上定义）</div><h3>3.2 经验律结果变量与候选</h3><p>每个分析模型族×模式×N×L 单元先计算可解析请求的平均有符号偏差。26 个冻结候选由 15 个 N、Lₖ、lnN、lnLₖ 的非空加性组合、4 个满足层级原则的一阶交互结构和 7 个密度 ρ=N/Lₖ 结构组成。每个候选允许模型特异系数；共享的是“包含哪些项”。</p><div class="method">例：y = β₀ + βN·N + βL·Lₖ + βNL·N·Lₖ
∂y/∂N = βN + βNL·Lₖ
∂y/∂Lₖ = βL + βNL·N</div><p>因此交互项 βNL 表示一个难度轴会改变另一个难度轴的边际效应，而不是简单地“再加一个乘积”。</p><h3>3.3 与旧报告完全一致的验证/选择</h3><ol><li>对每个模型族×模式×候选做 OLS；系数使用 HC3 稳健标准误与置信区间。</li><li>按 <code>(index(N)+index(L)) mod 5</code> 做五折 held-condition CV；整组 N×L 条件被留出，但 seed 不单独分折。</li><li>跨模型汇总 CV R²/MAE、标准化效应和交互相对主效应模型的 ΔCV R²；Wilcoxon 与系数检验使用 BH 校正。</li><li>在容差内优先选择项数更少、CV MAE 更小的结构；再做 leave-one-model-out（LOMO）结构稳定性。</li></ol><div class="note"><strong>明确删除的重计算：</strong>没有 2000 次 interaction bootstrap，也没有 nested held-seed、leave-one-N、leave-one-L 的候选重选择。它们不是旧 focused empirical-law 方法的一部分，且在当前网格上造成乘法级计算量。配对提示比较表中已完成的 paired-seed bootstrap 不受影响，因为它回答的是另一个问题。</div><div class="conclusion"><strong>本节结论：</strong>本报告恢复的是旧报告真正使用的方法，而不是较晚设计但未完成的重型方案；验证单位是 N×L 条件，模型集合稳健性由 LOMO 检查。</div></section>

<section id="behavior"><h2>4. 行为结果</h2><div class="table-wrap">{table_html(mode_table)}</div><figure><img src="{figures['accuracy_heatmap']}" alt="模型和模式准确率热图"><figcaption><strong>图 1.</strong> 横轴为提示模式，纵轴为固定物理模型；颜色与单元格数字表示 parsed exact accuracy（%）。空白是该物理模型不承担的模式，而非缺失请求。</figcaption></figure><div class="grid2"><figure><img src="{figures['accuracy_n']}" alt="准确率随N变化"><figcaption><strong>图 2.</strong> 横轴为真实针数量 N，纵轴为 parsed exact accuracy；线在 10 个同 checkpoint 可切换模型、全部 L 与 seed 上按请求加权。</figcaption></figure><figure><img src="{figures['accuracy_l']}" alt="准确率随L变化"><figcaption><strong>图 3.</strong> 横轴为 Lₖ=L/1000（千 token），纵轴为 parsed exact accuracy；线的聚合范围与图2相同。</figcaption></figure></div><figure><img src="{figures['strict_gap']}" alt="数值准确率与严格准确率"><figcaption><strong>图 4.</strong> 横轴为端到端数值正确率，纵轴为正确且格式合规的 strict accuracy；点落在 y=x 下方表示存在额外格式损失。</figcaption></figure><figure><img src="{figures['paired_forest']}" alt="索引枚举相对直接作答森林图"><figcaption><strong>图 5.</strong> 横轴为同 checkpoint 槽内索引枚举−直接作答的准确率风险差，横线为 paired-seed cluster bootstrap 95% 区间；纵轴为比较槽。该 bootstrap 已在行为预处理阶段完成，与被删除的经验律 interaction bootstrap 不同。</figcaption></figure><figure><img src="{figures['bias_n']}" alt="偏差随N变化"><figcaption><strong>图 6.</strong> 横轴 N，纵轴为合格单元的 10% 截尾有符号偏差；线为跨模型/L 中位数，带为四分位距。负值表示少计。</figcaption></figure><div class="conclusion"><strong>本节结论：</strong>显式枚举总体上提高数值正确率并压低高 N 下的少计，但格式合规仍是独立问题；N 与 L 都形成可见难度梯度。</div></section>

<section id="laws"><h2>5. 共享经验律结果</h2><div class="table-wrap">{table_html(law_table)}</div><p>四个模式都通过注册门选择 N+Lₖ+N×Lₖ。direct 的中位 CV R² 为 {fmt_num(direct_r2,3)}，native-thinking 为 {fmt_num(native_r2,3)}；index 与 bullet 的中位 CV R² 较低，表示共享结构只解释部分模型内的条件偏差变化。</p><figure><img src="{figures['lomo']}" alt="LOMO结构稳定性"><figcaption><strong>图 7.</strong> 横轴为提示模式，纵轴为 LOMO 结构一致率：每次删去一个分析模型族、重新做跨模型结构选择后，仍选中全样本结构的比例。100% 表示 12 次留一均一致；50% 表示仅 6/12 一致。</figcaption></figure><figure><img src="{figures['interaction']}" alt="标准化交互效应"><figcaption><strong>图 8.</strong> 横轴为提示模式，纵轴为分析模型族；颜色和数字是所选 N×Lₖ 项的带符号标准化效应。正负号反映交互方向，绝对值反映相对结果变量标准差的效应强度；不能只看跨模型中位数忽略符号异质性。</figcaption></figure>{selected_figures}<div class="conclusion"><strong>本节结论：</strong>direct 与 native-thinking 对 N×L 联合结构有强且模型集合稳定的支持；index 的证据为暂定但较稳定；bullet 虽通过全样本门限，LOMO 仅 50%，应报告为模型依赖、不可作强共享律结论。</div></section>

<section id="limits"><h2>6. 稳健性、边界与不能推出的结论</h2><ul><li><strong>解析条件化：</strong>经验律结果变量只在可解析输出上定义；解析失败率必须与偏差一起看。</li><li><strong>held-condition 不是 held-seed：</strong>CV 检查对未参与拟合的 N×L 条件的预测，但不会测量换 seed 后的完整重选择不确定性。</li><li><strong>网格内证据：</strong>14 个 N 与 8 个 L 之外属于外推；低维经验律不是物理定律。</li><li><strong>模型固定：</strong>LOMO 只检查这 12 个分析模型族，不能代表所有语言模型。</li><li><strong>检查点混杂：</strong>GLM/Ministral pair 的 native 与非 native 来自不同 checkpoint。</li><li><strong>选择后推断：</strong>HC3/BH 和 LOMO 降低过拟合风险，但不是选择后置信区间的完整解决方案。</li><li><strong>未做机制因果：</strong>回归交互描述行为曲面，不识别内部神经机制。</li></ul><div class="conclusion"><strong>本节结论：</strong>最可信的是冻结模型集合与 N/L 网格内的共享偏差结构；对新 seed、新模型或网格外长度的泛化仍需独立确认。</div></section>

<section id="repro"><h2>7. 复现与文件</h2><div class="method">run root: {html.escape(str(run_root))}
behavior tables: {html.escape(str(behavior_dir))}
focused-law analysis: {html.escape(str(law_dir))}
report: {html.escape(str(output))}
request_ids_sha256: {html.escape(audit['request_ids_sha256'])}
stimuli_sha256: {html.escape(audit['stimuli_sha256'])}
law input SHA-256: {html.escape(law_manifest['input_sha256'])}
legacy method SHA-256: {html.escape(law_manifest['legacy_method_script_sha256'])}
candidate fits: {len(selected_fits) * 26:,}; selected model-mode fits: {len(selected_fits)}; LOMO rows: {len(lomo)}</div><p>重建经验律：</p><div class="method">python scripts/analyze_realistic_niah_v3_1_legacy_method.py --input ".../request_level.csv.gz" --legacy-script ".../build_v2_focused_empirical_law_report.py" --output ".../v3_1_focused_empirical_law_legacy_method"</div><div class="conclusion"><strong>本节结论：</strong>输入哈希、旧方法源码哈希、模型映射、全部中间表和图均已本地保存；结果可在普通 CPU 上约数分钟重建，无需 GPU。</div></section>
<div class="footer">Realistic NIAH V3.1 · V2 focused empirical-law method applied to the current model set · figures embedded as base64 PNG.</div></main></body></html>"""

    if "�" in report:
        raise ValueError("Report contains a Unicode replacement character")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    build_manifest = {
        "schema_version": "niah_empirical_law_legacy_method_report_v1",
        "created_at": generated,
        "output": str(output),
        "requests": audit["requests"],
        "analysis_families": len(FAMILY_ORDER),
        "candidate_fits": len(selected_fits) * 26,
        "selected_laws": choices["mode"].tolist(),
        "figures": {key: str(value) for key, value in figure_paths.items()},
    }
    (assets / "report_build_manifest.json").write_text(
        json.dumps(build_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(output),
                "bytes": output.stat().st_size,
                "figures": len(figure_paths),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
