#!/usr/bin/env python3
"""Build the compact all-count NiaH geometry comparison report.

The main text contains only the two registered endpoint comparisons.  Trace
format proportions and trajectory-band diagnostics are appendices; there is no
format-stratified site x layer winner search in the primary estimand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.augment_niah_geometry_comparison_report import (  # noqa: E402
    FULL_LEGACY_MARKERS,
    HYBRID_MARKERS,
    TRACE_CATEGORIES,
    hybrid_marker_summary,
    legacy_compatible_marker_summary,
    marker_definitions_html,
    table,
    trace_category_summary,
)
from scripts.build_niah_geometry_comparison_report import (  # noqa: E402
    MODELS,
    build_dual_visual_data,
    dual_endpoint_section,
    esc,
    expected_trajectory_keys,
    load_dual_endpoint_results,
    read_csv,
    read_json,
    read_jsonl,
    sha256,
)


REPORT_SCHEMA_VERSION = "niah_geometry_comparison_v31_explicit_grammar_endpoints"


GRAMMAR_ITEM_END_LOCATIONS = {
    "adjacent_rank_after_city": (
        "city-bearing unit 后还有相邻 rank/commit；取整个合并 item 的末 token，"
        "通常是 rank 末尾数字、单词或标点。"
    ),
    "adjacent_rank_before_city": (
        "rank-only unit 在前、city unit 在后；取后一个 city/score unit 的末 token，"
        "不是前置 rank token。"
    ),
    "same_unit_rank_after_city": (
        "city 与后置 rank 在同一 unit；取后置 rank/Count suffix 的末 token。"
    ),
    "same_unit_rank_before_city": (
        "rank 与 city 在同一 unit 且 rank 在前；取 city/score 与闭合格式全部完成后的末 token。"
    ),
    "structural_explicit_rank_before_city": (
        "structural recap 含显式 ordinal；取完整 recap item 的末 token，不停在 marker。"
    ),
    "structural_invariant_bullet": (
        "bullet 本身不携带 count；取该 bullet item 的 city/score 内容全部完成后的末 token。"
    ),
    "structural_unmarked": (
        "没有显式 rank；取 parser 注册的完整 city-bearing item 的末 token。"
    ),
    "evidence_sequence_unranked": (
        "按 score-supported evidence sequence 恢复 item；取当前 city+score evidence 的末 token。"
    ),
}


def _short_text(value: Any, limit: int = 105) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _selected_by_mode(
    dual_results: Mapping[str, Mapping[str, Any]], model: str, endpoint: str
) -> dict[str, Mapping[str, Any]]:
    rows = dual_results[model][f"{endpoint}_selected"]
    selected = {str(row["mode"]): row for row in rows}
    expected = {"non_thinking", "native_thinking"}
    if set(selected) != expected:
        raise ValueError(
            f"Expected exactly {sorted(expected)} for {model}/{endpoint}; "
            f"got {sorted(selected)}"
        )
    return selected


def _accuracy_svg(
    model: str,
    dual_results: Mapping[str, Mapping[str, Any]],
) -> str:
    rows: list[tuple[str, float, float]] = []
    for endpoint, endpoint_label in (("running", "Running"), ("final", "Final")):
        selected = _selected_by_mode(dual_results, model, endpoint)
        for field, metric_label in (
            ("confirmation_logistic_balanced_accuracy", "Logistic"),
            ("confirmation_ncc_balanced_accuracy", "Nearest centroid"),
        ):
            rows.append(
                (
                    f"{endpoint_label} · {metric_label}",
                    float(selected["non_thinking"][field]),
                    float(selected["native_thinking"][field]),
                )
            )
    width, height = 700, 260
    x0, x1 = 170.0, 520.0
    scale = lambda value: x0 + float(value) * (x1 - x0)
    grid = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = scale(tick)
        grid.append(
            f'<line class="metric-gridline" x1="{x:.1f}" y1="31" '
            f'x2="{x:.1f}" y2="224"/><text class="metric-tick" '
            f'x="{x:.1f}" y="19" text-anchor="middle">{100*tick:.0f}%</text>'
        )
    marks = []
    for index, (label, non, native) in enumerate(rows):
        y = 56 + 47 * index
        non_x, native_x = scale(non), scale(native)
        marks.append(
            f'<text class="metric-label" x="8" y="{y+4}">{esc(label)}</text>'
            f'<line class="metric-link" x1="{non_x:.1f}" y1="{y}" '
            f'x2="{native_x:.1f}" y2="{y}"/>'
            f'<circle class="metric-dot metric-non" cx="{non_x:.1f}" cy="{y}" r="6"/>'
            f'<circle class="metric-dot metric-native" cx="{native_x:.1f}" cy="{y}" r="6"/>'
            f'<text class="metric-value" x="542" y="{y+4}">'
            f'{100*non:.1f} → {100*native:.1f}%</text>'
        )
    return (
        f'<figure class="metric-figure"><h3>{esc(model)}</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(model)} confirmation balanced accuracy, non-thinking versus native-thinking">'
        f'<title>{esc(model)} confirmation balanced accuracy</title>'
        f'<desc>Four held-out comparisons. Dark circles are non-thinking and teal circles are native-thinking.</desc>'
        + "".join(grid)
        + "".join(marks)
        + "</svg></figure>"
    )


def empirical_claims(
    dual_results: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render frozen Confirmation-100 probe evidence and a direct visual comparison."""

    verdicts = []
    charts = []
    total_wins = 0
    total_comparisons = 0
    for model in MODELS:
        wins = []
        deltas = []
        for endpoint in ("running", "final"):
            rows = _selected_by_mode(dual_results, model, endpoint)
            for field in (
                "confirmation_logistic_balanced_accuracy",
                "confirmation_ncc_balanced_accuracy",
            ):
                non = float(rows["non_thinking"][field])
                native = float(rows["native_thinking"][field])
                wins.append(native > non)
                deltas.append(native - non)
        total_wins += sum(wins)
        total_comparisons += len(wins)
        verdict = (
            "四个 endpoint × probe 对比均为 native-thinking 更高"
            if all(wins)
            else f"四个对比中有 {sum(wins)}/4 个为 native-thinking 更高"
        )
        verdicts.append(
            f"<li><strong>{esc(model)}：</strong>{verdict}；balanced-accuracy "
            f"差值范围为 {100*min(deltas):+.1f}–{100*max(deltas):+.1f} 个百分点。</li>"
        )
        charts.append(_accuracy_svg(model, dual_results))
    qwen_final = _selected_by_mode(dual_results, "Qwen3-8B", "final")
    gemma_running = _selected_by_mode(dual_results, "Gemma4-E4B", "running")

    def delta_pair(rows: Mapping[str, Mapping[str, Any]]) -> tuple[float, float]:
        return (
            100
            * (
                float(rows["native_thinking"]["confirmation_logistic_balanced_accuracy"])
                - float(rows["non_thinking"]["confirmation_logistic_balanced_accuracy"])
            ),
            100
            * (
                float(rows["native_thinking"]["confirmation_ncc_balanced_accuracy"])
                - float(rows["non_thinking"]["confirmation_ncc_balanced_accuracy"])
            ),
        )

    qwen_delta = delta_pair(qwen_final)
    gemma_delta = delta_pair(gemma_running)
    return f"""
<section id="claims"><h2>核心结果：native-thinking 的稳定优势是 count decodability</h2>
<div class="callout"><strong>最稳健的主结论：</strong>在两种 mode 各自由 discovery 选择最佳层、并冻结到相同的 N=1…10 held-out panel 后，native-thinking 在 {total_wins}/{total_comparisons} 个 model × endpoint × probe 比较中都更高。这里主张的是 <em>count information is more decodable</em>，不把它自动等同于所有 PCA 投影都更紧。</div>
<div class="definitions"><div><h3>{total_wins}/{total_comparisons} held-out wins</h3><p>Qwen 与 Gemma、running 与 final、Logistic 与 nearest centroid 全部同向。它是跨模型、跨 endpoint 最稳定的现象。</p></div><div><h3>Qwen final count</h3><p>Logistic / NCC 相对 non-thinking 提高 {qwen_delta[0]:+.1f} / {qwen_delta[1]:+.1f} 个百分点，是 Qwen 中最显著的 confirmation 差值。</p></div><div><h3>Gemma running index</h3><p>Logistic / NCC 提高 {gemma_delta[0]:+.1f} / {gemma_delta[1]:+.1f} 个百分点，是 Gemma 中最显著的 confirmation 差值。</p></div></div>
<div class="metric-legend"><span><i class="legend-non"></i>non-thinking</span><span><i class="legend-native"></i>native-thinking</span><span>右侧数值：non → native</span></div>
<div class="metric-grid">{''.join(charts)}</div>
<ul>{''.join(verdicts)}</ul>
<div class="callout warning"><strong>怎样理解图与 probe 的差别：</strong>PCA3 最大化总体方差，不以 count 类别分离为目标；因此视觉差异可以弱于全维/PCA16 probe 的 held-out 差异。报告保留全部四组 PCA 对比，但主 claim 以冻结的 confirmation 指标为准，不通过挑选“最好看”的视角定义优势。</div>
</section>"""


def new_native_geometry_section(
    clean_root: Path, marker_root: Path
) -> tuple[str, list[Path], dict[str, Any]]:
    """Render discovery-selected single-grammar and paired marker results."""
    clean_paths = [clean_root / name for name in (
        "selected_clean_grammar.csv", "grammar_support.csv", "geometry_payload.json", "audit.json"
    )]
    marker_paths = [marker_root / name for name in (
        "site_selected.csv", "geometry_payload.json", "audit.json"
    )]
    for path in clean_paths + marker_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    clean = {row["model_label"]: row for row in read_csv(clean_paths[0])}
    clean_payload = read_json(clean_paths[2])
    marker = {(row["scope"], row["site"]): row for row in read_csv(marker_paths[0])}
    pooled = {
        row["model_label"]: row
        for row in read_csv(ROOT / "reports/v5_native_causal_aligned_geometry/site_selected.csv")
        if row["site_kind"] == "item_end"
    }
    clean_rows = []
    clean_visual: dict[str, Any] = {}
    for model in MODELS:
        row, base = clean[model], pooled[model]
        selected_payload = dict(clean_payload[model]["selected"])
        selected_payload["metrics"] = {
            "confirmation_logistic": float(
                row["confirmation_logistic_balanced_accuracy"]
            ),
            "confirmation_ncc": float(
                row["confirmation_ncc_balanced_accuracy"]
            ),
            "confirmation_snr_db": float(
                row["confirmation_class_balanced_snr_db"]
            ),
        }
        clean_visual[model] = selected_payload
        clean_rows.append((
            esc(model), f"<code>{esc(row['grammar_class'])}</code>", f"L{int(float(row['layer']))}",
            str(int(float(row["rows"]))),
            f"{_pct(float(base['confirmation_logistic_balanced_accuracy']))} / {_pct(float(base['confirmation_ncc_balanced_accuracy']))}",
            f"{_pct(float(row['confirmation_logistic_balanced_accuracy']))} / {_pct(float(row['confirmation_ncc_balanced_accuracy']))}",
            f"{float(row['confirmation_class_balanced_snr_db']):+.2f} dB",
        ))
    marker_rows = []
    for scope in ("all_rank_before", "adjacent_rank_before_city"):
        for site in ("pre_marker", "post_marker"):
            row = marker[(scope, site)]
            marker_rows.append((
                f"<code>{scope}</code>", f"<code>{site}</code>", f"L{int(float(row['layer']))}",
                str(int(float(row["rows"]))),
                f"{_pct(float(row['confirmation_logistic_balanced_accuracy']))} / {_pct(float(row['confirmation_ncc_balanced_accuracy']))}",
                f"{float(row['confirmation_class_balanced_snr_db']):+.2f} dB",
            ))
    html = f"""
<section id="appendix-clean-grammar"><h2>Appendix E · 单一 trace grammar 与 Qwen marker 补采</h2>
<div class="callout"><strong>结论先行：</strong>限制为一种 grammar 只让 Qwen 的 counter geometry 明显变干净；Gemma 反而下降。因此 pooled native-thinking 图的混乱，Qwen 很大一部分来自 trace-format mixture，Gemma 则不能用这一解释概括。新补的 Qwen <code>post_marker</code> 也没有优于 <code>pre_marker</code>：marker 出现前 count 已经高度可读。</div>
<h3>固定 <code>item_end</code>，只在 discovery 选择 grammar × layer</h3>
<p class="small">候选必须在 discovery 与 confirmation 都覆盖 k=1…10。Qwen winner 的 confirmation 最小支持更充足；Gemma winner 的 k=10 只有 n=2，故仅作 exploratory。Pooled 与 single-grammar 均各自在 discovery 选自己的层，confirmation 不参与选择。</p>
{table(('model','discovery winner grammar','layer','states','pooled Log/NCC','single-grammar Log/NCC','single-grammar SNR'), clean_rows)}
<h3>同一 grammar 内的 running-index geometry</h3>
<p class="small">下图固定使用上表由 discovery-only 选择的 grammar 与 layer。PCA3 也只在该 grammar 的 discovery states 上拟合；下拉框只切换显示 full retained states 或 confirmation states，不重新拟合 PCA、不重新选层。</p>
<div class="dual-grid">
<article class="geometry-card"><h3>Qwen3-8B · <code>adjacent_rank_after_city</code></h3><div class="controls"><label>Rows<select id="clean-qwen-cohort"><option value="all">full retained states</option><option value="confirmation">confirmation only</option></select></label></div><canvas id="clean-qwen" role="img" aria-label="Qwen single grammar counter geometry"></canvas><div class="rotate-hint">drag to rotate · discovery-fitted PCA3 · fixed grammar and layer</div><div class="panel-stats" id="clean-qwen-stats"></div></article>
<article class="geometry-card"><h3>Gemma4-E4B · <code>same_unit_rank_before_city</code></h3><div class="controls"><label>Rows<select id="clean-gemma-cohort"><option value="all">full retained states</option><option value="confirmation">confirmation only</option></select></label></div><canvas id="clean-gemma" role="img" aria-label="Gemma single grammar counter geometry"></canvas><div class="rotate-hint">drag to rotate · discovery-fitted PCA3 · fixed grammar and layer</div><div class="panel-stats" id="clean-gemma-stats"></div></article>
</div>
<div class="callout warning"><strong>图的边界：</strong>两张图的 PCA basis 各自在自己的 model × grammar × layer 上拟合，因此只能比较各自的有序性、重叠和相对紧密程度，不能比较左右坐标轴的绝对尺度。Gemma confirmation 的 k=10 只有 2 个 states，视觉上的末端 centroid 尤其不稳定。</div>
<div class="definitions two"><div><h3>Qwen</h3><p><code>adjacent_rank_after_city</code> 在 L6 得到 95.7% / 90.3%，相对 pooled causal-aligned <code>item_end</code> 的 69.8% / 60.1% 是实质改善。支持“多种 trace grammar 的 nuisance variation 会遮蔽 counter”。</p></div><div><h3>Gemma</h3><p><code>same_unit_rank_before_city</code> 在 L16 只有 65.8% / 62.4%，低于 pooled 80.8% / 72.6%。单 grammar 并未净化其 counter；更可能是减少数据后丢失了共享信号，且 k=10 很稀。</p></div></div>
<h3>新 GPU capture：显式 rank marker 前后</h3>
<p class="small">642 个 retrieval-eligible rank-before events，teacher-forced 重放冻结 completion，不重新采样、不干预。<code>pre_marker</code> 是 rank surface 开始前一个 token；<code>post_marker</code> 是 rank surface 结束、city 开始前的 compiler query token。两侧使用同一批 event，各自仅由 discovery 选层。</p>
{table(('scope','site','layer','states','confirmation Log/NCC','SNR'), marker_rows)}
<div class="callout warning"><strong>反证式解释：</strong><code>pre_marker</code> 在 pooled 与 adjacent scope 都达到 100% / 100%，而 <code>post_marker</code> 分别为 93.7% / 89.9% 与 98.0% / 96.7%。所以 count 信息不是由显式 ordinal marker 才“写入”的；marker 更像改变局部表示坐标、措辞或边界状态。由于 pre-marker 的上下文本身已包含此前计数进度，这也不是“纯净隐式计数器”的充分证明。</div>
</section>"""
    return html, clean_paths + marker_paths, clean_visual


def index_city_geometry_appendix(
    analysis_root: Path,
) -> tuple[str, list[Path], dict[str, Any]]:
    """Render the a-priori fixed explicit-index-plus-city slice."""

    paths = [
        analysis_root / "site_selected.csv",
        analysis_root / "site_support.csv",
        analysis_root / "geometry_payload.json",
        analysis_root / "audit.json",
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    selected = {
        (str(row["model_label"]), str(row["site"])): row
        for row in read_csv(paths[0])
    }
    payload = read_json(paths[2])
    if str(payload.get("schema_version")) != (
        "realistic_niah_v5_native_index_city_geometry_v1"
    ):
        raise ValueError(f"Unexpected index+city payload: {paths[2]}")
    models = payload.get("models", {})
    if set(models) != set(MODELS):
        raise ValueError(f"Index+city payload has models {sorted(models)}")

    rows = []
    verdicts = []
    for model in MODELS:
        city = selected[(model, "city_end")]
        item = selected[(model, "item_end")]
        city_mean = (
            float(city["confirmation_logistic_balanced_accuracy"])
            + float(city["confirmation_ncc_balanced_accuracy"])
        ) / 2
        item_mean = (
            float(item["confirmation_logistic_balanced_accuracy"])
            + float(item["confirmation_ncc_balanced_accuracy"])
        ) / 2
        delta = 100 * (city_mean - item_mean)
        direction = "更高" if delta > 0 else "更低"
        verdicts.append(
            f"<li><strong>{esc(model)}：</strong><code>city_end</code> 的两 probe "
            f"均值比同 grammar <code>item_end</code> {direction} {abs(delta):.1f} 个百分点。</li>"
        )
        for site, row in (("city_end", city), ("item_end", item)):
            rows.append(
                (
                    esc(model),
                    f"<code>{esc(row['fixed_grammar'])}</code>",
                    f"<code>{site}</code>",
                    f"L{int(float(row['layer']))}",
                    str(int(float(row["states"]))),
                    f"{_pct(row['confirmation_logistic_balanced_accuracy'])} / "
                    f"{_pct(row['confirmation_ncc_balanced_accuracy'])}",
                    f"{float(row['confirmation_class_balanced_snr_db']):+.2f} dB",
                )
            )

    html = f"""
<section id="appendix-index-city"><h2>Appendix E.2 · 固定的显式 index + city grammar</h2>
<div class="callout"><strong>这里的“最纯净”是格式控制，不是隐式计数器：</strong>Qwen 固定为 <code>adjacent_rank_before_city</code>，Gemma 固定为 <code>same_unit_rank_before_city</code>；不再让 discovery 从多种 grammar 中挑赢家。主图取 <code>city_end</code>，即模型已经看完当前显式 index 和当前 city 的最后一个 token。每个位点只独立选择 layer，confirmation 不参与选择。</div>
{table(('model','a-priori grammar','site','layer','states','confirmation Log/NCC','SNR'), rows)}
<ul>{''.join(verdicts)}</ul>
<div class="dual-grid">
<article class="geometry-card"><h3>Qwen3-8B · index + city at <code>city_end</code></h3><div class="controls"><label>Rows<select id="index-city-qwen-cohort"><option value="all">full retained states</option><option value="confirmation">confirmation only</option></select></label></div><canvas id="index-city-qwen" role="img" aria-label="Qwen fixed index plus city geometry"></canvas><div class="rotate-hint">drag to rotate · fixed grammar · discovery-fitted PCA3</div><div class="panel-stats" id="index-city-qwen-stats"></div></article>
<article class="geometry-card"><h3>Gemma4-E4B · index + city at <code>city_end</code></h3><div class="controls"><label>Rows<select id="index-city-gemma-cohort"><option value="all">full retained states</option><option value="confirmation">confirmation only</option></select></label></div><canvas id="index-city-gemma" role="img" aria-label="Gemma fixed index plus city geometry"></canvas><div class="rotate-hint">drag to rotate · fixed grammar · discovery-fitted PCA3</div><div class="panel-stats" id="index-city-gemma-stats"></div></article>
</div>
<div class="callout warning"><strong>判读边界：</strong>显式 rank 已直接出现在上下文中，所以图更清楚最多说明“固定格式后，index 信息在读完 city 时更可读”；它不能单独证明模型维护了不依赖 surface marker 的内部 counter。两模型的 layer 与 PCA basis 也各自独立，左右只能比较有序性和重叠，不能比较坐标尺度。</div>
</section>"""
    return html, paths, models


def _phase_silhouette_svg(
    model: str, rows: list[tuple[str, float, str]]
) -> str:
    """Draw an honest compactness comparison with lexical controls labelled."""

    width = 760
    height = 58 + 42 * len(rows)
    x0, x1 = 220.0, 650.0
    lower, upper = -0.10, 1.00
    scale = lambda value: x0 + (float(value) - lower) / (upper - lower) * (x1 - x0)
    grid = []
    for tick in (-0.1, 0.0, 0.25, 0.5, 0.75, 1.0):
        x = scale(tick)
        cls = "metric-zero" if tick == 0 else "metric-gridline"
        grid.append(
            f'<line class="{cls}" x1="{x:.1f}" y1="28" x2="{x:.1f}" '
            f'y2="{height-24}"/><text class="metric-tick" x="{x:.1f}" y="18" '
            f'text-anchor="middle">{tick:.2f}</text>'
        )
    color = {
        "non": "#20242D",
        "native": "#00A88F",
        "semantic": "#E76F51",
        "control": "#6750E8",
        "ablation": "#D6B52C",
    }
    marks = []
    for index, (label, value, kind) in enumerate(rows):
        y = 48 + 42 * index
        x = scale(value)
        marks.append(
            f'<text class="metric-label" x="8" y="{y+4}">{esc(label)}</text>'
            f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="{color[kind]}" '
            f'stroke="#FFFDF8" stroke-width="2"/>'
            f'<text class="metric-value" x="{x+12:.1f}" y="{y+4}">{value:+.3f}</text>'
        )
    return (
        f'<figure class="metric-figure"><h3>{esc(model)} · confirmation Mahalanobis silhouette</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(model)} compactness comparison"><title>{esc(model)} compactness comparison</title>'
        + "".join(grid)
        + "".join(marks)
        + "</svg></figure>"
    )


def phase_grammar_ablation_appendix(
    phase_root: Path,
    covariance_root: Path,
    grammar_root: Path | None = None,
    standard_grammar_root: Path | None = None,
) -> tuple[str, list[Path], dict[str, Any]]:
    """Render strict phase sites and discovery-only grammar ablation."""

    phase_paths = [
        phase_root / "site_selected.csv",
        phase_root / "site_support.csv",
        phase_root / "site_surface_composition.csv",
        phase_root / "geometry_payload.json",
        phase_root / "audit.json",
    ]
    capture_audits = [
        phase_root / "capture_audits" / model / "capture_audit.json"
        for model in MODELS
    ]
    covariance_path = covariance_root / "discovery_selected_metrics.csv"
    for path in phase_paths + capture_audits + [covariance_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    selected_rows = read_csv(phase_paths[0])
    selected = {
        (str(row["model_label"]), str(row["site"])): row
        for row in selected_rows
    }
    phase_payload = read_json(phase_paths[3])
    if str(phase_payload.get("schema_version")) != (
        "realistic_niah_v5_native_phase_geometry_analysis_v1"
    ):
        raise ValueError(f"Unexpected phase payload schema: {phase_paths[3]}")
    covariance = read_csv(covariance_path)
    silhouette_baseline = {
        (str(row["model_label"]), str(row["mode"])): row
        for row in covariance
        if str(row["endpoint"]) == "running_index"
        and str(row["selector"]) == "mahalanobis_silhouette"
    }
    surface_rows = read_csv(phase_paths[2])

    grammar_paths: list[Path] = []
    grammar_selected: list[dict[str, Any]] = []
    phase_grammar_detailed: list[dict[str, Any]] = []
    grammar_payload: dict[str, Any] = {"models": {}}
    grammar_candidates: list[dict[str, Any]] = []
    if grammar_root is not None:
        grammar_paths = [
            grammar_root / "grammar_selected.csv",
            grammar_root / "grammar_site_eligibility.csv",
            grammar_root / "shared_layer_selected_sites.csv",
            grammar_root / "grammar_geometry_payload.json",
            grammar_root / "audit.json",
            grammar_root / "grammar_site_layer_candidates.csv",
        ]
        for path in grammar_paths:
            if not path.exists():
                raise FileNotFoundError(path)
        phase_grammar_detailed = read_csv(grammar_paths[0])
        grammar_selected = list(phase_grammar_detailed)
        grammar_payload = read_json(grammar_paths[3])
        grammar_candidates = read_csv(grammar_paths[5])
        if str(grammar_payload.get("schema_version")) != (
            "realistic_niah_v5_native_phase_grammar_geometry_v1"
        ):
            raise ValueError(f"Unexpected grammar payload schema: {grammar_paths[3]}")

    standard_paths: list[Path] = []
    standard_payload: dict[str, Any] = {"models": {}}
    standard_candidates: list[dict[str, Any]] = []
    standard_grammar_detailed: list[dict[str, Any]] = []
    if standard_grammar_root is not None:
        standard_paths = [
            standard_grammar_root / "standard_grammar_selected.csv",
            standard_grammar_root / "standard_grammar_site_eligibility.csv",
            standard_grammar_root / "standard_grammar_site_layer_candidates.csv",
            standard_grammar_root / "standard_grammar_geometry_payload.json",
            standard_grammar_root / "audit.json",
        ]
        for path in standard_paths:
            if not path.exists():
                raise FileNotFoundError(path)
        standard_grammar_detailed = read_csv(standard_paths[0])
        standard_candidates = read_csv(standard_paths[2])
        standard_payload = read_json(standard_paths[3])
        if str(standard_payload.get("schema_version")) != (
            "realistic_niah_v5_native_standard_grammar_geometry_v1"
        ):
            raise ValueError(
                f"Unexpected standard grammar payload schema: {standard_paths[3]}"
            )

    def pick_site_layer(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("Cannot select from an empty grammar candidate set")
        return sorted(
            rows,
            key=lambda row: (
                -float(row["discovery_selection_score"]),
                -float(row["discovery_oof_ncc_balanced_accuracy"]),
                -float(row["discovery_oof_logistic_balanced_accuracy"]),
                int(float(row["layer"])),
                str(row["site"]),
            ),
        )[0]

    if grammar_candidates or standard_candidates:
        combined = grammar_candidates + standard_candidates
        grammar_selected = []
        keys = sorted(
            {
                (str(row["model_label"]), str(row["grammar_class"]))
                for row in combined
            }
        )
        for model, grammar in keys:
            candidate = pick_site_layer(
                [
                    row
                    for row in combined
                    if str(row["model_label"]) == model
                    and str(row["grammar_class"]) == grammar
                ]
            )
            detailed = [
                row
                for row in phase_grammar_detailed + standard_grammar_detailed
                if str(row["model_label"]) == model
                and str(row["grammar_class"]) == grammar
                and str(row["site"]) == str(candidate["site"])
                and int(float(row["layer"])) == int(float(candidate["layer"]))
            ]
            if len(detailed) != 1:
                raise ValueError(
                    f"Expected one detailed winner for {model}/{grammar}/"
                    f"{candidate['site']}/L{candidate['layer']}; got {len(detailed)}"
                )
            grammar_selected.append(detailed[0])

    role = {
        "post_city": "strict semantic site · first original token after city token",
        "post_marker": "explicit-marker query control · before city",
        "marker_end": "explicit marker surface control",
    }
    phase_table_rows = []
    surface_table_rows = []
    visual: dict[str, Any] = {}
    verdicts = []
    charts = []
    for model in MODELS:
        model_visual: dict[str, Any] = {"phase_views": {}, "ablation_views": {}}
        for site in ("post_city", "post_marker", "marker_end"):
            row = selected[(model, site)]
            metrics = phase_payload["models"][model]["sites"][site]["metrics"]
            model_visual["phase_views"][site] = {
                "label": f"{site} · L{int(float(row['layer']))}",
                "site": site,
                "layer": int(float(row["layer"])),
                "role": role[site],
                "metrics": metrics,
                "pca3": phase_payload["models"][model]["sites"][site]["pca3"],
            }
            phase_table_rows.append(
                (
                    esc(model),
                    f"<code>{site}</code>",
                    esc(role[site]),
                    f"L{int(float(row['layer']))}",
                    str(int(float(row["states"]))),
                    f"{_pct(row['confirmation_logistic_balanced_accuracy'])} / "
                    f"{_pct(row['confirmation_ncc_balanced_accuracy'])}",
                    f"{float(row['cov_confirmation_mahalanobis_silhouette']):+.3f}",
                    f"{float(row['cov_confirmation_fisher_trace_frozen']):.2f}",
                    f"{float(row['cov_confirmation_ordinal_rsa']):+.3f}",
                    f"{float(row['confirmation_radius_gap_ratio']):.3f}",
                )
            )
        for site in ("post_city", "post_marker", "marker_end"):
            site_surface = [
                row
                for row in surface_rows
                if str(row["model_label"]) == model and str(row["site"]) == site
            ]
            total = sum(int(float(row["states"])) for row in site_surface)
            composition: dict[str, int] = {}
            for row in site_surface:
                key = str(row["token_surface_class"])
                composition[key] = composition.get(key, 0) + int(float(row["states"]))
            summary = ", ".join(
                f"{esc(key)} {100*value/total:.1f}%"
                for key, value in sorted(composition.items())
            )
            surface_table_rows.append(
                (esc(model), f"<code>{site}</code>", str(total), summary)
            )

        non = float(silhouette_baseline[(model, "non_thinking")]["confirmation_value"])
        native = float(silhouette_baseline[(model, "native_thinking")]["confirmation_value"])
        post_city = float(selected[(model, "post_city")]["cov_confirmation_mahalanobis_silhouette"])
        marker_end = float(selected[(model, "marker_end")]["cov_confirmation_mahalanobis_silhouette"])
        chart_rows: list[tuple[str, float, str]] = [
            ("non-thinking span_end · independent best layer", non, "non"),
            ("native item_end · independent best layer", native, "native"),
            ("native post_city · classification-selected layer", post_city, "semantic"),
            ("native marker_end · lexical control", marker_end, "control"),
        ]
        comparison = "高于" if post_city > non else "低于"
        verdicts.append(
            f"<li><strong>{esc(model)}：</strong>严格 <code>post_city</code> silhouette "
            f"{post_city:+.3f}，{comparison} non-thinking 最佳层 {non:+.3f}；"
            f"显式 <code>marker_end</code> 则为 {marker_end:+.3f}。</li>"
        )

        grammar_model = grammar_payload.get("models", {}).get(model, {})
        standard_model = standard_payload.get("models", {}).get(model, {})
        model_grammar_winners = [
            row for row in grammar_selected if str(row["model_label"]) == model
        ]
        for winner in model_grammar_winners:
            grammar = str(winner["grammar_class"])
            phase_item = grammar_model.get("grammars", {}).get(grammar, {})
            standard_item = standard_model.get("grammars", {}).get(grammar, {})
            item = phase_item
            if str(winner["site"]) in {
                "pre_city",
                "city_end",
                "city_unit_end",
                "item_end",
                "post_boundary",
            }:
                item = standard_item
            if "pca3" not in item:
                raise ValueError(
                    f"Missing PCA3 payload for full-site winner {model}/{grammar}/"
                    f"{winner['site']}"
                )
            key = f"grammar:{grammar}"
            model_visual["ablation_views"][key] = {
                "label": f"{grammar} · {winner['site']} · L{int(float(winner['layer']))}",
                "site": str(winner["site"]),
                "layer": int(float(winner["layer"])),
                "grammar": grammar,
                "metrics": winner,
                "pca3": item["pca3"],
                "warning": "grammar-specific site and layer; discovery-selected upper-bound ablation",
            }
        shared = grammar_model.get("shared_layer", {})
        shared_scope = "phase-only 3-site candidate pool"
        if shared and standard_candidates:
            full_model_candidates = [
                row
                for row in grammar_candidates + standard_candidates
                if str(row["model_label"]) == model
                and str(row["grammar_class"])
                in {str(winner["grammar_class"]) for winner in model_grammar_winners}
            ]
            full_layer_options: list[tuple[float, int, dict[str, str]]] = []
            for layer in sorted({int(float(row["layer"])) for row in full_model_candidates}):
                chosen = []
                sites_at_layer: dict[str, str] = {}
                for winner in model_grammar_winners:
                    grammar = str(winner["grammar_class"])
                    rows_at_layer = [
                        row
                        for row in full_model_candidates
                        if str(row["grammar_class"]) == grammar
                        and int(float(row["layer"])) == layer
                    ]
                    if not rows_at_layer:
                        break
                    now = pick_site_layer(rows_at_layer)
                    chosen.append(now)
                    sites_at_layer[grammar] = str(now["site"])
                if len(chosen) == len(model_grammar_winners) and chosen:
                    full_layer_options.append(
                        (
                            sum(float(row["discovery_selection_score"]) for row in chosen)
                            / len(chosen),
                            layer,
                            sites_at_layer,
                        )
                    )
            full_layer_options.sort(key=lambda row: (-row[0], row[1]))
            if full_layer_options:
                _score, full_shared_layer, full_shared_sites = full_layer_options[0]
                if (
                    int(shared["layer"]) == full_shared_layer
                    and {
                        str(key): str(value)
                        for key, value in shared["selected_sites"].items()
                    }
                    == full_shared_sites
                ):
                    shared_scope = "full 8-site candidate pool"
        if shared:
            for view_key, payload_key, label in (
                ("shared:raw", "raw_pca3", "shared layer · raw pooled grammars"),
                (
                    "shared:centered",
                    "grammar_centered_pca3",
                    "shared layer · discovery grammar-centered sensitivity",
                ),
            ):
                model_visual["ablation_views"][view_key] = {
                    "label": f"{label} · L{int(shared['layer'])} · {shared_scope}",
                    "site": "grammar-specific",
                    "layer": int(shared["layer"]),
                    "grammar": "combined",
                    "metrics": shared["metrics"],
                    "pca3": shared[payload_key],
                    "warning": (
                        f"{shared_scope}; "
                        + shared[payload_key].get("warning", shared["selection"])
                    ),
                }
            chart_rows.append(
                (
                    "native grammar ablation · shared layer",
                    float(shared["metrics"]["cov_confirmation_mahalanobis_silhouette"]),
                    "ablation",
                )
            )
        visual[model] = model_visual
        charts.append(_phase_silhouette_svg(model, chart_rows))

    grammar_rows = []
    for row in grammar_selected:
        grammar_rows.append(
            (
                esc(row["model_label"]),
                f"<code>{esc(row['grammar_class'])}</code>",
                f"<code>{esc(row['site'])}</code>",
                f"L{int(float(row['layer']))}",
                str(int(float(row["states"]))),
                f"{_pct(row['confirmation_logistic_balanced_accuracy'])} / "
                f"{_pct(row['confirmation_ncc_balanced_accuracy'])}",
                f"{float(row['cov_confirmation_mahalanobis_silhouette']):+.3f}",
                f"{float(row['confirmation_radius_gap_ratio']):.3f}",
            )
        )

    def options(model: str, family: str, default: str | None = None) -> str:
        items = visual[model][family]
        ordered = list(items)
        if default in ordered:
            ordered.remove(default)
            ordered.insert(0, default)
        return "".join(
            f'<option value="{esc(key)}">{esc(items[key]["label"])}</option>'
            for key in ordered
        )

    phase_cards = []
    ablation_cards = []
    for model in MODELS:
        slug_model = "qwen" if model.startswith("Qwen") else "gemma"
        phase_cards.append(
            f'<article class="geometry-card"><h3>{esc(model)} · phase site</h3>'
            f'<div class="controls"><label>Site<select id="phase-{slug_model}-view">'
            f'{options(model, "phase_views", "post_city")}</select></label>'
            f'<label>Rows<select id="phase-{slug_model}-cohort"><option value="confirmation_raw">confirmation raw states</option><option value="all_raw">full raw states</option><option value="confirmation_means">confirmation seed×count means</option><option value="all_means">full seed×count means</option></select></label></div>'
            f'<canvas id="phase-{slug_model}" role="img" aria-label="{esc(model)} phase-site geometry"></canvas>'
            f'<div class="rotate-hint">drag to rotate · each view has its own discovery-fitted PCA3</div>'
            f'<div class="panel-stats" id="phase-{slug_model}-stats"></div></article>'
        )
        if visual[model]["ablation_views"]:
            ablation_cards.append(
                f'<article class="geometry-card"><h3>{esc(model)} · grammar ablation</h3>'
                f'<div class="controls"><label>Discovery-selected view<select id="ablation-{slug_model}-view">'
                f'{options(model, "ablation_views", "shared:centered")}</select></label>'
                f'<label>Rows<select id="ablation-{slug_model}-cohort"><option value="confirmation_raw">confirmation raw states</option><option value="all_raw">full raw states</option><option value="confirmation_means">confirmation seed×count means</option><option value="all_means">full seed×count means</option></select></label></div>'
                f'<canvas id="ablation-{slug_model}" role="img" aria-label="{esc(model)} grammar ablation geometry"></canvas>'
                f'<div class="rotate-hint">drag to rotate · shared:centered is an explicit nuisance-removal sensitivity view</div>'
                f'<div class="panel-stats" id="ablation-{slug_model}-stats"></div></article>'
            )

    grammar_block = ""
    if grammar_rows:
        every_marker_end = all(str(row["site"]) == "marker_end" for row in grammar_selected)
        if every_marker_end:
            winner_warning = (
                "所有满足完整 1…10 支持的 grammar 都把 <code>marker_end</code> 选成 winner。"
                "若 winner 位于 L0，主要反映当前 token embedding/局部 surface identity；"
                "这属于 lexical positive control，不是深层 consolidation 的证据。"
            )
        else:
            semantic_winners = [
                row for row in grammar_selected if str(row["site"]) != "marker_end"
            ]
            marker_winners = [
                row for row in grammar_selected if str(row["site"]) == "marker_end"
            ]
            semantic_text = "；".join(
                f"{esc(row['model_label'])} <code>{esc(row['grammar_class'])}</code> "
                f"选择 <code>{esc(row['site'])}</code> L{int(float(row['layer']))}，"
                f"confirmation Log/NCC={_pct(row['confirmation_logistic_balanced_accuracy'])}/"
                f"{_pct(row['confirmation_ncc_balanced_accuracy'])}，silhouette="
                f"{float(row['cov_confirmation_mahalanobis_silhouette']):+.3f}"
                for row in semantic_winners
            )
            marker_text = "；".join(
                f"{esc(row['model_label'])} <code>{esc(row['grammar_class'])}</code> "
                f"仍选择 <code>marker_end</code> L{int(float(row['layer']))}"
                for row in marker_winners
            )
            winner_warning = (
                f"8-site 搜索不是清一色 marker：{semantic_text}。这支持该 grammar 在读完 "
                f"city 所在语义单元后仍有紧致 count state。其余 winner 为：{marker_text}；"
                "其中 L0 结果仍应按显式 token identity 解释。"
            )
        grammar_block = f"""
<h3>Grammar-specific discovery winner</h3>
<p class="small">完整候选池包含 8 个 sites：旧 capture 的 <code>pre_city / city_end / city_unit_end / item_end / post_boundary</code>，以及新补采的 <code>post_marker / marker_end / post_city</code>。只有 discovery 与 confirmation 都覆盖 k=1…10、且每个 k 至少由两个 discovery seeds 支持的 grammar×site 才可入选。先在 discovery 用 grouped-CV 选择 site×layer，confirmation 完全冻结；这是 ablation upper bound，不替换 pooled 主结果。</p>
{table(('model','grammar','winner site','layer','states','confirmation Log/NCC','silhouette','radius/gap ↓'), grammar_rows)}
<div class="callout warning"><strong>Winner 审计：</strong>{winner_warning}</div>
<div class="dual-grid">{''.join(ablation_cards)}</div>
<div class="callout warning"><strong>“再放在一起”的严格做法：</strong>不同 grammar 若各自选择不同 layer，原 hidden vectors 不可直接塞进一个 PCA。合并图因此另加一个共享层约束：每个 grammar 可在该层选自己的 site，但所有 state 都来自同一 layer；图的下拉标签会注明共享层是否仍是完整 8-site 候选池的 winner。<code>shared:raw</code> 是原 states；<code>shared:centered</code> 只减去 discovery 估计的 grammar mean，用于检验格式 offset，不能作为原始 geometry 的主证据。</div>
"""

    html = f"""
<section id="appendix-phase-grammar"><h2>Appendix F · Phase-site 与 grammar-stratified ablation</h2>
<div class="callout"><strong>结论先行：</strong>这次补采没有支持“离开显式 marker 后，native-thinking 普遍比 non-thinking 更紧”。两个模型的 <code>marker_end</code> 都极清晰，但严格 <code>post_city</code> 都明显变差。最稳妥的解释是：显式 rank surface 提供了强的 count-readable state；是否存在跨 grammar、跨 surface 的紧致抽象 counter，必须看 <code>post_city</code> 与 grammar-frozen confirmation，而不能只挑最漂亮的 marker 图。</div>
<div class="definitions"><div><h3><code>post_marker</code></h3><p>causal compiler 注册的显式 rank marker 之后、city 之前的 query token。它是 lexical positive control；上下文已经直接出现 k。</p></div><div><h3><code>marker_end</code></h3><p>显式 rank core 的最后一个原始输出 token。Qwen 多为词法/数字混合，Gemma 几乎全是 numeric；高可解码性预期含直接 surface leakage。</p></div><div><h3><code>post_city</code></h3><p>city-containing token 之后的第一个原始 baseline output token，且不越过该 item 的 commit。它可能是 lexical、syntax 或 whitespace，不把字符串截断后重新 tokenization。</p></div></div>
{table(('model','site','role','layer','states','confirmation Log/NCC','Mahalanobis silhouette','frozen Fisher','ordinal RSA','radius/gap ↓'), phase_table_rows)}
<div class="callout warning"><strong>“紧”和“有序”必须分开：</strong>Gemma <code>marker_end</code> 的 silhouette 为 {float(selected[('Gemma4-E4B','marker_end')]['cov_confirmation_mahalanobis_silhouette']):+.3f}，但 ordinal RSA 为 {float(selected[('Gemma4-E4B','marker_end')]['cov_confirmation_ordinal_rsa']):+.3f}：它形成非常紧的十类 token clusters，却几乎不沿数值差排列。相反，Qwen/Gemma <code>post_city</code> 的 silhouette 只有 {float(selected[('Qwen3-8B','post_city')]['cov_confirmation_mahalanobis_silhouette']):+.3f}/{float(selected[('Gemma4-E4B','post_city')]['cov_confirmation_mahalanobis_silhouette']):+.3f}，但 RSA 仍有 {float(selected[('Qwen3-8B','post_city')]['cov_confirmation_ordinal_rsa']):+.3f}/{float(selected[('Gemma4-E4B','post_city')]['cov_confirmation_ordinal_rsa']):+.3f}。因此 post-city 更像“有序 centroid trend 淹没在较大类内变异中”，不是紧簇。</div>
<h3>Token surface 组成</h3>{table(('model','site','states','surface composition'), surface_table_rows)}
<h3>与 non-thinking 的紧致性对照</h3>
<p class="small">下图比较 confirmation Mahalanobis silhouette；每种方法的层都只由 discovery 选择。non-thinking/native item_end 使用各自 silhouette 最佳层；新增 phase sites 使用 classification-selected layer，因此不把 confirmation silhouette 用来挑图。越高表示单个 state 更靠近自己 count cloud。紫色 marker 是显式 lexical control，不与 semantic sites 混作机制证据。</p>
<div class="metric-grid">{''.join(charts)}</div><ul>{''.join(verdicts)}</ul>
<h3>原始 phase-site 3D 与去噪显示</h3>
<p class="small">默认显示最严格的 <code>post_city</code> confirmation raw states。可切换到 marker controls，也可切换 seed×count means；means 只为降低 overplotting，所有表格指标仍使用 raw states。</p>
<div class="dual-grid">{''.join(phase_cards)}</div>
{grammar_block}
</section>"""
    return (
        html,
        phase_paths
        + capture_audits
        + [covariance_path]
        + grammar_paths
        + standard_paths,
        visual,
    )


def metric_guide_section() -> str:
    """Explain every primary probe/geometry metric with examples and limits."""

    return """
<section id="metric-guide"><h2>指标字典：六个数分别回答什么</h2>
<div class="callout"><strong>先统一符号：</strong>对每个 count 类别 k，<code>μₖ</code> 是该类 hidden states 的 centroid，<code>μ̄</code> 是十个类别 centroid 的等权平均。<code>Σ<sub>B</sub></code> 是十个 centroid 围绕 <code>μ̄</code> 的 class-balanced between-count covariance；<code>Σ<sub>W</sub></code> 是先在每类内部求 residual covariance、再对十类等权平均的 within-count covariance。因此 running 中样本较多的低 k 不会自动获得更大权重。</div>
<div class="metric-guide-grid">
<article class="metric-guide-card"><h3>Logistic balanced accuracy</h3><p class="formula">BAcc = (1/10) Σₖ recallₖ</p><p><strong>怎么算：</strong>在 discovery 拟合 StandardScaler、whitened PCA16 和带 class balancing 的十分类线性 Logistic；层也只由 seed-grouped discovery OOF 选择。冻结后在 confirmation 预测 k/N，最后等权平均十类 recall。</p><p><strong>现实意义：</strong>回答“一个简单线性下游读出器能否恢复 count”。它允许学习十个线性 decision regions，因此是 <em>linear decodability</em> 指标。</p><p><strong>例：</strong>若十个 count 的 recall 分别约为 80%，BAcc 就约为 80%；即使 running 的 k=1 states 更多，永远猜 1 也只接近 10% chance，而不会因样本量膨胀。</p><p class="small"><strong>不能说明：</strong>可解码不等于模型因果使用该变量，也不保证每类形成紧密球状簇。</p></article>
<article class="metric-guide-card"><h3>Nearest-centroid balanced accuracy (NCC)</h3><p class="formula">ŷ = arg minₖ ‖z − μₖ<sup>disc</sup>‖²；BAcc = meanₖ recallₖ</p><p><strong>怎么算：</strong>在与 Logistic 相同的 discovery-fitted whitened PCA16 中求每个 count 的 discovery centroid；confirmation state 直接分给欧氏距离最近的 centroid，不再学习额外 decision weights。</p><p><strong>现实意义：</strong>回答“count 是否已被组织成可由简单原型读取的几何”。NCC 高而 Logistic 也高时，证据不只依赖一个灵活的线性边界。</p><p><strong>例：</strong>若某 state 到 count-4 centroid 的距离为 0.8，到 count-3/5 centroid 为 1.4/1.2，则预测 4；十类分别算 recall 后再平均。</p><p class="small"><strong>不能说明：</strong>弯曲流形或强各向异性簇可能线性可分却不接近自己的欧氏 centroid，所以 NCC 与 Logistic 不必同步。</p></article>
<article class="metric-guide-card"><h3>Isotropic SNR</h3><p class="formula">SNR = tr(Σ<sub>B</sub>) / tr(Σ<sub>W</sub>)；SNR<sub>dB</sub> = 10 log₁₀(SNR)</p><p><strong>怎么算：</strong>先用 discovery-fitted StandardScaler 与 PCA16 whitening 变换 states，再在 confirmation 上按上面的 class-balanced 定义计算 centroid energy 与类内 residual energy。</p><p><strong>现实意义：</strong>回答“总的 between-count signal 相对总的 within-count variation 有多大”。0 dB 表示两者相等；负值表示类内总变异更大；数值越高越清楚。</p><p><strong>例：</strong>若 <code>tr(Σ<sub>B</sub>)=4</code>、<code>tr(Σ<sub>W</sub>)=1</code>，SNR=4，即 6.02 dB。由 −4.04 dB 升到 −1.78 dB 表示比值提高约 1.68 倍，而不是说 noise 本身一定缩小。</p><p class="small"><strong>不能说明：</strong>trace 把 16 个方向直接相加，不区分某个方向本来就很 noisy；升高既可能来自 centroid 拉开，也可能来自类内残差减小。</p></article>
<article class="metric-guide-card"><h3>Fisher trace</h3><p class="formula">F = tr[(Σ<sub>W</sub><sup>disc</sup> + λI)<sup>−1</sup> Σ<sub>B</sub><sup>conf</sup>]</p><p><strong>怎么算：</strong>在 discovery-fitted PCA16 空间估计 class-balanced within covariance，加一个按 covariance scale 缩放的 ridge 后求 precision；该 precision 完全冻结，只用 confirmation 的 between-count covariance 评价。每个 metric 的层仍由 discovery OOF Fisher 选择。</p><p><strong>现实意义：</strong>回答“confirmation 的 count centroids 是否沿 discovery 中稳定、低噪声的方向分开”。它是 covariance-aware 的 separation：同样的 centroid 差，落在低噪声方向会得到更高权重。</p><p><strong>例：</strong>二维玩具例中，若 <code>Σ<sub>B</sub>=diag(4,1)</code>、<code>Σ<sub>W</sub>=diag(1,9)</code>，则 F≈4/1+1/9=4.11；第二维虽然有 signal=1，但 noise=9，所以贡献很小。这个例子说明 Fisher 为什么能处理各向异性 noise。</p><p class="small"><strong>不能说明：</strong>F 没有 0–1 上界，也没有通用“及格线”；它依赖 discovery covariance 的稳定性与 ridge。若 discovery/confirmation noise 分布漂移，需同时检查 frozen precision 的 noise calibration，不能只看 F。</p></article>
<article class="metric-guide-card"><h3>Mahalanobis silhouette</h3><p class="formula">sᵢ = (bᵢ − aᵢ) / max(aᵢ,bᵢ)，范围 [−1,1]</p><p><strong>怎么算：</strong>先用 discovery 的 <code>Σ<sub>W</sub><sup>−1/2</sup></code> whitening PCA16 states；对 confirmation 点 i，<code>aᵢ</code> 是它到同 count 其他点的平均距离，<code>bᵢ</code> 是它到“最近的另一个 count 类”的平均距离。先在每个 count 内平均 <code>sᵢ</code>，再对十类等权平均。</p><p><strong>现实意义：</strong>回答“单个 state 是否更像自己 count 的 cloud，而不是最近的其他 count cloud”。接近 1 表示 pointwise cluster membership 清楚，接近 0 表示边界重叠，负值表示平均更靠近别类。</p><p><strong>例：</strong>若某点的 <code>a=1</code>、<code>b=3</code>，则 silhouette=(3−1)/3=0.67；若 <code>a=3</code>、<code>b=1</code>，则为 −0.67。</p><p class="small"><strong>不能说明：</strong>它不关心 count 顺序；十个彼此隔离但按 1,7,2,9… 排列的簇仍可有很高 silhouette。</p></article>
<article class="metric-guide-card"><h3>Held-out ordinal RSA</h3><p class="formula">ρ = Spearman({|k−ℓ|}, {d<sub>Mahalanobis</sub>(μₖ,μℓ)})，共 C(10,2)=45 对</p><p><strong>怎么算：</strong>用与 silhouette 相同的 discovery-frozen Mahalanobis metric，在 confirmation 求十个 count centroids；枚举 45 个 centroid pair，将几何距离与数字差 <code>|k−ℓ|</code> 做 Spearman rank correlation。</p><p><strong>现实意义：</strong>回答“表征是否近似一条有序 count axis”：数字相差越大，centroid 是否通常也越远。ρ 接近 1 表示高度单调有序，接近 0 表示没有稳定的 ordinal relation。</p><p><strong>例：</strong>对 counts 1,2,3，若 d(1,2)=1、d(2,3)=1、d(1,3)=2，distance rank 与 gap rank 一致，ρ 接近 1；若 1 与 10 反而相邻，ρ 会下降。</p><p class="small"><strong>不能说明：</strong>centroid 可以排得很有序，但各类 cloud 仍大量重叠；因此 RSA 高不等于 silhouette 高或分类准确率高。</p></article>
</div>
<details><summary>统一的防泄漏与比较规则</summary><div class="callout"><p>所有 label 都使用 gold running index/final N，模型答错的 trajectory 也保留。StandardScaler、PCA、Logistic、centroid、within-covariance metric 与 layer selection 都只从 discovery 得到；confirmation 不参与选层，只用于冻结评价。</p><p>Logistic/NCC 与主 SNR 使用 discovery-fitted whitened PCA16。Fisher、Mahalanobis silhouette 与 ordinal RSA 先使用 discovery-fitted、未 PCA-whiten 的 16-D scores，再只按 discovery 的 within-count covariance 做 noise whitening。每个 model × mode × endpoint × metric 各选自己的最佳层，因此表中的绝对值回答各指标自己的最佳可读性，不是假设所有指标共享同一层。</p></div></details>
</section>"""


def load_causal_aligned_results(
    causal_root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Load and validate the CPU-only causal-event-aligned geometry sweep."""

    causal_root = causal_root.resolve()
    paths = {
        "selected": causal_root / "site_selected.csv",
        "winners": causal_root / "model_site_winners.csv",
        "legacy": causal_root / "legacy_vs_causal_item_end.csv",
        "audit": causal_root / "audit.json",
    }
    selected = read_csv(paths["selected"])
    winners = read_csv(paths["winners"])
    legacy = read_csv(paths["legacy"])
    audit = read_json(paths["audit"])
    if str(audit.get("schema_version")) != (
        "realistic_niah_v5_native_causal_aligned_geometry_v1"
    ):
        raise ValueError(
            f"Unexpected causal-aligned schema in {paths['audit']}: "
            f"{audit.get('schema_version')!r}"
        )
    if str(audit.get("primary_counter_site")) != "item_end":
        raise ValueError("Causal-aligned primary counter site must remain item_end")
    expected_sites = {
        "pre_city",
        "city_end",
        "city_unit_end",
        "item_end",
        "post_boundary",
    }
    observed = {
        (str(row["model_label"]), str(row["site_kind"])) for row in selected
    }
    expected = {(model, site) for model in MODELS for site in expected_sites}
    if observed != expected:
        raise ValueError(
            "Causal-aligned selected-site panel is incomplete: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if len(winners) != len(MODELS) or len(legacy) != len(MODELS):
        raise ValueError("Expected one causal site winner and one legacy comparison per model")
    paired = audit.get("paired_site_metadata_sha256", {})
    if set(paired) != set(MODELS):
        raise ValueError("Causal-aligned audit does not cover both model panels")
    for name in ("selected", "winners", "legacy"):
        expected_sha = audit.get("outputs", {}).get(str(paths[name].resolve()))
        if expected_sha is not None and str(expected_sha) != sha256(paths[name]):
            raise ValueError(f"Causal-aligned output hash mismatch: {paths[name]}")
    return {
        "selected": selected,
        "winners": winners,
        "legacy": legacy,
        "audit": audit,
    }, list(paths.values())


def causal_aligned_progress_section(results: Mapping[str, Any]) -> str:
    """Render the frozen P0 progress-site reanalysis without replacing the primary panel."""

    selected = list(results["selected"])
    audit = results["audit"]
    winner_by_model = {
        str(row["model_label"]): str(row["site_kind"])
        for row in results["winners"]
    }
    site_order = {
        "pre_city": 0,
        "city_end": 1,
        "city_unit_end": 2,
        "item_end": 3,
        "post_boundary": 4,
    }
    role = {
        "pre_city": "paired control · retrieval onset",
        "city_end": "paired control · city complete",
        "city_unit_end": "paired control · semantic unit complete",
        "item_end": "primary · P0 progress commit",
        "post_boundary": "exploratory · downstream boundary",
    }
    rows = []
    for row in sorted(
        selected,
        key=lambda value: (
            MODELS.index(str(value["model_label"])),
            site_order[str(value["site_kind"])],
        ),
    ):
        model = str(row["model_label"])
        site = str(row["site_kind"])
        flags = []
        if site == "item_end":
            flags.append('<span class="site-badge primary-badge">PRIMARY</span>')
        if winner_by_model.get(model) == site:
            flags.append('<span class="site-badge winner-badge">DISCOVERY WINNER</span>')
        rows.append(
            (
                esc(model),
                f"<code>{esc(site)}</code><br><span class=\"small\">{esc(role[site])}</span>"
                + "".join(flags),
                f"L{int(float(row['layer']))}",
                _pct(row["discovery_selection_score"]),
                f"{_pct(row['confirmation_logistic_balanced_accuracy'])} / "
                f"{_pct(row['confirmation_ncc_balanced_accuracy'])}",
                f"{float(row['confirmation_class_balanced_snr_db']):+.2f} dB",
                f"{float(row['cov_confirmation_mahalanobis_silhouette']):.3f}",
                f"{float(row['cov_confirmation_ordinal_rsa']):.3f}",
            )
        )

    legacy_rows = []
    for row in sorted(
        results["legacy"], key=lambda value: MODELS.index(str(value["model_label"]))
    ):
        legacy_rows.append(
            (
                esc(row["model_label"]),
                f"L{int(float(row['legacy_layer']))} → L{int(float(row['causal_layer']))}",
                f"{_pct(row['legacy_confirmation_logistic_balanced_accuracy'])} → "
                f"{_pct(row['causal_confirmation_logistic_balanced_accuracy'])} "
                f"({100*float(row['causal_minus_legacy_logistic']):+.2f} pp)",
                f"{_pct(row['legacy_confirmation_ncc_balanced_accuracy'])} → "
                f"{_pct(row['causal_confirmation_ncc_balanced_accuracy'])} "
                f"({100*float(row['causal_minus_legacy_ncc']):+.2f} pp)",
                f"{float(row['legacy_confirmation_snr_db']):+.2f} → "
                f"{float(row['causal_confirmation_snr_db']):+.2f} dB "
                f"({float(row['causal_minus_legacy_snr_db']):+.2f})",
            )
        )

    exact_rows = []
    for model in MODELS:
        loader = audit["loader_audits"][f"{model}/item_end"]
        total = int(loader["causal_registry_event_count"])
        matched = int(loader["matched_state_count"])
        exact_rows.append(
            (
                esc(model),
                f"{matched} / {total}",
                _pct(matched / total),
                str(int(loader["matched_trajectory_count"])),
                f"<code>{esc(str(loader['metadata_key_sha256'])[:12])}…</code>",
            )
        )

    return f"""
<section id="causal-aligned"><h2>Causal compiler 对齐后的 progress geometry</h2>
<div class="callout"><strong>结论：</strong>更精确的 causal event 筛选让主位点 <code>item_end = P0 progress commit</code> 的 held-out decodability 小幅上升，但没有让 isotropic SNR 同步上升；因此目前不能概括成“counter 普遍更紧致”。更强的结果是 <em>site sensitivity</em>：Qwen 的 downstream <code>post_boundary</code> 与 <code>city_unit_end</code> 更易读，而 Gemma 的 discovery winner 正是统一的 <code>item_end</code>。</div>
<div class="definitions two"><div><h3>这次本地重分析做了什么</h3><p>不重新运行模型。先从冻结 causal registry 保留 <code>primary_full_chain_event ∧ progress_commit_eligible ∧ progress_commit_site_resolved</code>，再只接受旧 capture 的 <code>item_end.endpoint_token</code> 与新 <code>commit_output_token</code> 完全相等的 event。五个位点在每个模型内使用同一批 event，逐层以 seed-grouped discovery OOF 选择层，confirmation 只做一次冻结评价。</p></div><div><h3>与主 representation 的关系</h3><p>本节只评价 P0 progress/update state 及其 paired neighbours；旧 capture 中不存在的 marker 后位点不混入主比较。主 representation 始终固定为逐 grammar 的完整 <code>item_end</code>，其具体 token 位置与支持数已在“Token 提取”表中列出。</p></div></div>
<h3>完全相同 token 的保留率</h3>
{table(('model', 'exact events', 'retained', 'trajectories', 'paired-event SHA'), exact_rows)}
<h3>同事件、不同 token site 的冻结结果</h3>
<p class="small">每个位点先各自在 discovery 选择 decoder layer；<code>DISCOVERY WINNER</code> 也只按 discovery OOF 的 Logistic/NCC 均值选择。表中 Log/NCC、SNR、Mahalanobis silhouette 与 ordinal RSA 全部来自 held-out confirmation。粗体标签不把 downstream boundary 重新定义成 primary counter。</p>
{table(('model', 'site / role', 'layer', 'discovery OOF', 'confirmation Log / NCC', 'SNR', 'Mah. silhouette', 'ordinal RSA'), rows)}
<h3>Primary <code>item_end</code>：旧 parser cohort → causal-aligned cohort</h3>
{table(('model', 'layer', 'Logistic BAcc', 'NCC BAcc', 'SNR'), legacy_rows)}
<div class="callout warning"><strong>如何解释：</strong>Qwen primary 的 Log/NCC 分别提高 1.68/1.24 pp，但 SNR 下降 0.09 dB；Gemma 分别提高 2.43/3.37 pp，SNR 下降 0.72 dB。这里复用的是同一批 archived activations，变化同时包含更严格的 event/cohort 筛选与重新选层，不能表述为一次新的 forward pass 让 hidden state 本身“变紧”。Qwen <code>post_boundary</code> 的 92.8% / 76.1% 很强，却位于 commit 之后，可能混入后续结构或 ordinal 信息，应视为机制定位线索而非更纯的 counter 证据。</div>
</section>"""


def load_nonthinking_internal_metrics(
    covariance_root: Path,
) -> tuple[dict[str, dict[str, dict[str, Mapping[str, str]]]], list[Path]]:
    """Load the discovery-selected, confirmation-frozen non-thinking metrics."""

    covariance_root = covariance_root.resolve()
    selected_path = covariance_root / "discovery_selected_metrics.csv"
    audit_path = covariance_root / "audit.json"
    rows = read_csv(selected_path)
    audit = read_json(audit_path)
    if str(audit.get("schema_version")) != "niah_covariance_geometry_v1":
        raise ValueError(
            f"Unexpected covariance audit schema in {audit_path}: "
            f"{audit.get('schema_version')!r}"
        )
    if int(audit.get("pca_dim", -1)) != 16:
        raise ValueError(f"Expected PCA16 covariance audit in {audit_path}")

    endpoints = ("running_index", "final_count")
    selectors = (
        "isotropic_snr",
        "ordinal_rsa",
        "mahalanobis_silhouette",
        "fisher_trace",
    )
    indexed: dict[str, dict[str, dict[str, Mapping[str, str]]]] = {
        model: {endpoint: {} for endpoint in endpoints} for model in MODELS
    }
    for row in rows:
        if str(row.get("mode")) != "non_thinking":
            continue
        model = str(row.get("model_label"))
        endpoint = str(row.get("endpoint"))
        selector = str(row.get("selector"))
        if model not in indexed or endpoint not in endpoints or selector not in selectors:
            continue
        if selector in indexed[model][endpoint]:
            raise ValueError(
                f"Duplicate covariance winner for {model}/{endpoint}/{selector}"
            )
        if int(float(row["pca_components"])) != 16:
            raise ValueError(
                f"Expected PCA16 winner for {model}/{endpoint}/{selector}"
            )
        expected_confirmation_rows = 550 if endpoint == "running_index" else 100
        if int(float(row["confirmation_rows"])) != expected_confirmation_rows:
            raise ValueError(
                f"Unexpected confirmation support for {model}/{endpoint}/{selector}: "
                f"{row['confirmation_rows']}"
            )
        indexed[model][endpoint][selector] = row

    missing = [
        f"{model}/{endpoint}/{selector}"
        for model in MODELS
        for endpoint in endpoints
        for selector in selectors
        if selector not in indexed[model][endpoint]
    ]
    if missing:
        raise ValueError(
            "Missing non-thinking covariance winners: " + ", ".join(missing)
        )
    return indexed, [selected_path, audit_path]


def _internal_endpoint_metric_svg(
    metrics: Mapping[str, Mapping[str, Mapping[str, Mapping[str, str]]]],
    *,
    models: Iterable[str],
    selector: str,
    title: str,
    axis_title: str,
    domain: tuple[float, float],
    ticks: Iterable[float],
    decimals: int,
) -> str:
    """Draw a running-to-pre-answer comparison for one metric."""

    models = tuple(models)
    if not models:
        raise ValueError("Internal endpoint metric figure requires a model")
    width = 700
    plot_bottom = 82 + 57 * (len(models) - 1)
    height = plot_bottom + 45
    x0, x1 = 155.0, 520.0
    low, high = domain
    scale = lambda value: x0 + (float(value) - low) / (high - low) * (x1 - x0)
    grid = []
    for tick in ticks:
        x = scale(tick)
        label = f"{tick:.{decimals}f}"
        grid.append(
            f'<line class="metric-gridline" x1="{x:.1f}" y1="31" '
            f'x2="{x:.1f}" y2="{plot_bottom}"/><text class="metric-tick" '
            f'x="{x:.1f}" y="19" text-anchor="middle">{esc(label)}</text>'
        )
    marks = []
    for index, model in enumerate(models):
        y = 62 + 57 * index
        running = metrics[model]["running_index"][selector]
        answer = metrics[model]["final_count"][selector]
        running_value = float(running["confirmation_value"])
        answer_value = float(answer["confirmation_value"])
        running_layer = int(float(running["selected_layer"]))
        answer_layer = int(float(answer["selected_layer"]))
        marks.append(
            f'<text class="metric-label" x="8" y="{y+4}">{esc(model)}</text>'
            f'<line class="metric-link" x1="{scale(running_value):.1f}" y1="{y}" '
            f'x2="{scale(answer_value):.1f}" y2="{y}"/>'
            f'<circle class="metric-dot metric-running" '
            f'cx="{scale(running_value):.1f}" cy="{y}" r="6"/>'
            f'<circle class="metric-dot metric-answer" '
            f'cx="{scale(answer_value):.1f}" cy="{y}" r="6"/>'
            f'<text class="metric-value" x="536" y="{y+4}">'
            f'{running_value:.{decimals}f} (L{running_layer}) → '
            f'{answer_value:.{decimals}f} (L{answer_layer})</text>'
        )
    return (
        f'<figure class="metric-figure"><h3>{esc(title)}</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}, non-thinking running needle end versus pre-answer query">'
        f'<title>{esc(title)}</title>'
        '<desc>Discovery-selected layers evaluated on frozen confirmation data. '
        'Purple is the running needle-end endpoint and teal is the pre-answer query endpoint.</desc>'
        + "".join(grid)
        + "".join(marks)
        + f'<text class="metric-axis-title" x="337" y="{height-12}" '
        f'text-anchor="middle">{esc(axis_title)}</text>'
        + "</svg></figure>"
    )


def _metric_pair_cell(
    metrics: Mapping[str, Mapping[str, Mapping[str, str]]],
    selector: str,
    *,
    decimals: int,
    suffix: str = "",
) -> str:
    running = metrics["running_index"][selector]
    answer = metrics["final_count"][selector]
    running_value = float(running["confirmation_value"])
    answer_value = float(answer["confirmation_value"])
    running_layer = int(float(running["selected_layer"]))
    answer_layer = int(float(answer["selected_layer"]))
    return (
        f"{running_value:.{decimals}f}{suffix} @ L{running_layer} → "
        f"{answer_value:.{decimals}f}{suffix} @ L{answer_layer}"
    )


def _internal_geometry_blocks(
    metrics: Mapping[str, Mapping[str, Mapping[str, Mapping[str, str]]]],
    dual_visual: Mapping[str, Any],
    *,
    models: Iterable[str],
) -> str:
    """Render Appendix-D controls/canvases using the existing all-layer states."""

    blocks = []
    for model in models:
        slug = "qwen" if model.startswith("Qwen") else "gemma"
        running_layer = int(
            float(metrics[model]["running_index"]["isotropic_snr"]["selected_layer"])
        )
        answer_layer = int(
            float(metrics[model]["final_count"]["isotropic_snr"]["selected_layer"])
        )
        panels = dual_visual[model]["panels"]
        running_layers = list(map(int, panels["running_non"]["layers"]))
        answer_layers = list(map(int, panels["final_non"]["layers"]))
        if running_layer not in running_layers or answer_layer not in answer_layers:
            raise ValueError(
                f"Appendix-D SNR layer missing from visual states for {model}: "
                f"running L{running_layer}, answer L{answer_layer}"
            )
        running_options = "".join(
            f'<option value="{layer}"'
            f'{(" selected" if layer == running_layer else "")}>L{layer}</option>'
            for layer in running_layers
        )
        answer_options = "".join(
            f'<option value="{layer}"'
            f'{(" selected" if layer == answer_layer else "")}>L{layer}</option>'
            for layer in answer_layers
        )
        blocks.append(
            f"""<article class="appendix-model internal-model"><h3>{esc(model)}</h3>
<div class="controls internal-controls"><label>Rows<select id="internal-{slug}-cohort"><option value="all">full panel · 300 source trajectories · N=1…10</option><option value="confirmation" selected>confirmation · 100 source trajectories · N=1…10</option></select></label></div>
<div class="band-grid"><figure class="band-figure"><h4>Running index · prompt needle-end</h4>
<div class="controls"><label>Layer <span class="small">(default: discovery-selected SNR)</span><select id="internal-{slug}-running-layer">{running_options}</select></label></div>
<canvas id="internal-{slug}-running" role="img" aria-label="{esc(model)} non-thinking running-index geometry in three dimensions"></canvas>
<p class="rotate-hint">drag to rotate · color/centroid label = running k · discovery-fitted PCA3</p><p class="panel-stats" id="internal-{slug}-running-stats"></p></figure>
<figure class="band-figure"><h4>Pre-answer query · final count</h4>
<div class="controls"><label>Layer <span class="small">(default: discovery-selected SNR)</span><select id="internal-{slug}-answer-layer">{answer_options}</select></label></div>
<canvas id="internal-{slug}-answer" role="img" aria-label="{esc(model)} non-thinking pre-answer final-count geometry in three dimensions"></canvas>
<p class="rotate-hint">drag to rotate · color/centroid label = gold N · discovery-fitted PCA3</p><p class="panel-stats" id="internal-{slug}-answer-stats"></p></figure></div></article>"""
        )
    return "".join(blocks)


def nonthinking_internal_section(
    metrics: Mapping[str, Mapping[str, Mapping[str, Mapping[str, str]]]],
    *,
    dual_visual: Mapping[str, Any],
    domain_evidence_included: bool,
) -> str:
    """Render the exploratory two-model non-thinking comparison as Appendix D."""

    models = MODELS
    charts = [
        _internal_endpoint_metric_svg(
            metrics,
            models=models,
            selector="isotropic_snr",
            title="Confirmation SNR",
            axis_title="PCA16 isotropic SNR (dB; higher is clearer)",
            domain=(-8.0, 0.0),
            ticks=(-8.0, -6.0, -4.0, -2.0, 0.0),
            decimals=2,
        ),
        _internal_endpoint_metric_svg(
            metrics,
            models=models,
            selector="ordinal_rsa",
            title="Held-out ordinal RSA",
            axis_title="Spearman ρ (higher means count distances are more ordinal)",
            domain=(0.7, 1.0),
            ticks=(0.7, 0.8, 0.9, 1.0),
            decimals=3,
        ),
    ]
    rows = []
    verdicts = []
    case_cards = []
    for model in models:
        model_metrics = metrics[model]
        running_snr = float(
            model_metrics["running_index"]["isotropic_snr"]["confirmation_value"]
        )
        answer_snr = float(
            model_metrics["final_count"]["isotropic_snr"]["confirmation_value"]
        )
        running_rsa = float(
            model_metrics["running_index"]["ordinal_rsa"]["confirmation_value"]
        )
        answer_rsa = float(
            model_metrics["final_count"]["ordinal_rsa"]["confirmation_value"]
        )
        running_silhouette = float(
            model_metrics["running_index"]["mahalanobis_silhouette"][
                "confirmation_value"
            ]
        )
        answer_silhouette = float(
            model_metrics["final_count"]["mahalanobis_silhouette"][
                "confirmation_value"
            ]
        )
        running_fisher = float(
            model_metrics["running_index"]["fisher_trace"]["confirmation_value"]
        )
        answer_fisher = float(
            model_metrics["final_count"]["fisher_trace"]["confirmation_value"]
        )
        rows.append(
            (
                esc(model),
                esc(_metric_pair_cell(model_metrics, "isotropic_snr", decimals=2, suffix=" dB")),
                esc(_metric_pair_cell(model_metrics, "ordinal_rsa", decimals=3)),
                esc(_metric_pair_cell(model_metrics, "mahalanobis_silhouette", decimals=3)),
                esc(_metric_pair_cell(model_metrics, "fisher_trace", decimals=1)),
            )
        )
        verdicts.append(
            f"<li><strong>{esc(model)}：</strong>pre-answer 相对 running 的 SNR "
            f"提高 {answer_snr-running_snr:+.2f} dB，ordinal RSA 提高 "
            f"{answer_rsa-running_rsa:+.3f}。</li>"
        )
        if answer_silhouette > running_silhouette and answer_fisher > running_fisher:
            case_text = (
                "四项指标都上升：除了 count ordering 与 global signal/noise ratio，"
                "pointwise cluster membership 和 discovery-low-noise directions 上的"
                " centroid separation 也同向变清楚。"
            )
        else:
            silhouette_direction = (
                "上升" if answer_silhouette > running_silhouette else "下降"
            )
            fisher_direction = "上升" if answer_fisher > running_fisher else "下降"
            case_text = (
                f"这是指标分歧的实际案例：SNR/RSA 上升，但 silhouette {silhouette_direction}、"
                f"Fisher {fisher_direction}。它表示 count axis 的总体相对强度与顺序更清楚，"
                "却不能推出每个 state 更贴近自己簇，或 centroids 在 discovery-low-noise "
                "directions 上也更分离。"
            )
        case_cards.append(
            f'<div class="callout"><strong>{esc(model)} · 四项合并判读：</strong>'
            f'{esc(case_text)}</div>'
        )
    domain_note = (
        "同时，Appendix C 的 entity-domain probe 显示 non-thinking pre-answer state "
        "仍保留明显的实体域信息；因此这里的 consolidation 更接近表征重组，"
        "不是把 prompt semantics 完全过滤掉。"
        if domain_evidence_included
        else "当前报告未载入实体域迁移结果，因此不能据此判断 prompt semantics 是否被过滤。"
    )
    geometry_blocks = _internal_geometry_blocks(
        metrics, dual_visual, models=models
    )
    return f"""
<section id="appendix-nonthinking-internal"><h2>Appendix D · Non-thinking 内部：running index → pre-answer query</h2>
<div class="callout warning"><strong>结论等级：exploratory、支持性。</strong>Qwen 与 Gemma 的 pre-answer final-count 表征都有更高的 confirmation SNR 与 ordinal RSA。但 Qwen 的 Mahalanobis silhouette 和 Fisher trace 同时下降，Gemma 则四项都上升。因此 Gemma 的“答案前表征更清楚”证据更广；Qwen 只支持 count axis 更有序、global between/within ratio 更高，不支持 universal cluster tightening。两者都不构成严格的 consolidation effect。</div>
<div class="definitions two"><div><h3>Running endpoint</h3><p>在 prompt 的第 k 个 needle span 末 token 取 hidden state，并以 k=1…10 标注。每条 trajectory 可贡献多个 ragged states。</p></div><div><h3>Pre-answer endpoint</h3><p>取 <code>answer_query_v3</code>：prompt-final <code>Total:</code> 的冒号 hidden state，并以最终 N=1…10 标注。它位于生成数字之前，因此没有读取答案 digit。</p></div></div>
<div class="callout"><strong>3-D 图如何对齐定量结果：</strong>左右图默认分别使用该 endpoint 由 discovery SNR 选出的 layer，也可查看所有层和 full/confirmation 两种 cohort。每张图的 StandardScaler/PCA3 只在该 endpoint、该层的 discovery states 上独立拟合；因此可比较 count 顺序和相对散度，不能直接比较左右坐标的绝对尺度。</div>
{geometry_blocks}
<div class="metric-legend"><span><i class="legend-running"></i>running needle-end</span><span><i class="legend-answer"></i>pre-answer query</span><span>每个 endpoint/metric 各自由 discovery 选择最佳层</span></div>
<div class="metric-grid">{''.join(charts)}</div>
{table(['模型','SNR: running → pre-answer','Ordinal RSA: running → pre-answer','Mahalanobis silhouette','Fisher trace'], rows)}
<ul>{''.join(verdicts)}</ul>
{''.join(case_cards)}
<p>{esc(domain_note)}</p>
<div class="callout warning"><strong>为什么不是严格比较：</strong>两端共享同一套 trajectory panel，但统计单位与标签语义不同：running 是每条 trajectory 的多个中间 k，confirmation 共 550 states，且 k 的 support 呈三角形；pre-answer 是每条 trajectory 一个最终 N，共 100 states且每类 10 条。class-balanced 指标和 discovery-frozen 选择减轻了 support 不均衡，却不能把两端变成严格的一一配对 contraction test。</div>
<p class="muted"><strong>新增对应实验：</strong>native-thinking 的 running → answer 与 city/flower/animal 对照现已放在 Appendix C.2。Qwen 的 answer endpoint 明显更跨域可读，Gemma 则没有同方向提升；因此本节的 non-thinking 结果不再外推为统一的 native-thinking consolidation law。</p>
</section>"""


def _retained_label_text(value: Mapping[str, Any]) -> str:
    labels = list(map(int, value["retained_labels"]))
    if labels == list(range(1, 11)):
        return "k=1…10"
    if labels and labels == list(range(labels[0], labels[-1] + 1)):
        return f"k={labels[0]}…{labels[-1]}"
    return "k=" + ",".join(map(str, labels))


def _support_range(value: Mapping[str, Any]) -> str:
    retained = set(map(int, value["retained_labels"]))
    supports = [
        int(count)
        for label, count in value["support"].items()
        if int(label) in retained
    ]
    return f"nₖ={min(supports)}–{max(supports)}" if supports else "no retained k"


def _snr_svg(
    model: str,
    dual_results: Mapping[str, Mapping[str, Any]],
    audit: Mapping[str, Any],
    *,
    domain: tuple[float, float],
) -> str:
    running = _selected_by_mode(dual_results, model, "running")
    final = _selected_by_mode(dual_results, model, "final")
    conditioned = audit["ordinal_decodability"]["band_conditioned_confirmation_snr"]
    upper = conditioned["per_band"]["upper"]
    lower = conditioned["per_band"]["lower"]
    rows: list[tuple[str, list[tuple[str, float]]]] = [
        (
            "Running · global",
            [
                ("non", float(running["non_thinking"]["confirmation_class_balanced_snr_db"])),
                ("native", float(running["native_thinking"]["confirmation_class_balanced_snr_db"])),
            ],
        ),
        ("Native upper · within-band", [("upper", float(upper["snr_db"]))]),
        ("Native lower · within-band", [("lower", float(lower["snr_db"]))]),
        (
            "Final · global",
            [
                ("non", float(final["non_thinking"]["confirmation_class_balanced_snr_db"])),
                ("native", float(final["native_thinking"]["confirmation_class_balanced_snr_db"])),
            ],
        ),
    ]
    width, height = 700, 260
    x0, x1 = 190.0, 520.0
    low, high = domain
    scale = lambda value: x0 + (float(value) - low) / (high - low) * (x1 - x0)
    ticks = list(range(math.ceil(low), math.floor(high) + 1, 2))
    if 0 not in ticks and low <= 0 <= high:
        ticks.append(0)
        ticks.sort()
    grid = []
    for tick in ticks:
        x = scale(tick)
        zero_class = " metric-zero" if tick == 0 else ""
        tick_label = "0" if tick == 0 else f"{tick:+d}"
        grid.append(
            f'<line class="metric-gridline{zero_class}" x1="{x:.1f}" y1="31" '
            f'x2="{x:.1f}" y2="224"/><text class="metric-tick" '
            f'x="{x:.1f}" y="19" text-anchor="middle">{tick_label}</text>'
        )
    marks = []
    for index, (label, values) in enumerate(rows):
        y = 56 + 47 * index
        marks.append(f'<text class="metric-label" x="8" y="{y+4}">{esc(label)}</text>')
        if len(values) == 2:
            marks.append(
                f'<line class="metric-link" x1="{scale(values[0][1]):.1f}" y1="{y}" '
                f'x2="{scale(values[1][1]):.1f}" y2="{y}"/>'
            )
        for kind, value in values:
            marks.append(
                f'<circle class="metric-dot snr-{kind}" cx="{scale(value):.1f}" '
                f'cy="{y}" r="6"/>'
            )
        summary = " / ".join(f"{kind} {value:+.2f}" for kind, value in values)
        marks.append(
            f'<text class="metric-value" x="538" y="{y+4}">{esc(summary)}</text>'
        )
    return (
        f'<figure class="metric-figure"><h3>{esc(model)}</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(model)} confirmation signal to noise ratio in decibels">'
        f'<title>{esc(model)} confirmation SNR</title>'
        f'<desc>Global non-thinking and native-thinking SNR plus native-thinking upper and lower discovery-fitted within-band SNR.</desc>'
        + "".join(grid)
        + "".join(marks)
        + '<text class="metric-axis-title" x="355" y="250" text-anchor="middle">SNR (dB)</text>'
        + "</svg></figure>"
    )


def snr_section(
    dual_results: Mapping[str, Mapping[str, Any]],
    band_audits: Mapping[str, Mapping[str, Any]],
) -> str:
    values: list[float] = []
    for model in MODELS:
        for endpoint in ("running", "final"):
            selected = _selected_by_mode(dual_results, model, endpoint)
            values.extend(
                float(selected[mode]["confirmation_class_balanced_snr_db"])
                for mode in ("non_thinking", "native_thinking")
            )
        conditioned = band_audits[model]["ordinal_decodability"][
            "band_conditioned_confirmation_snr"
        ]
        values.extend(
            float(conditioned["per_band"][name]["snr_db"])
            for name in ("upper", "lower")
            if math.isfinite(float(conditioned["per_band"][name]["snr_db"]))
        )
    domain = (float(math.floor(min(values)) - 1), float(math.ceil(max(values)) + 1))
    charts = [
        _snr_svg(model, dual_results, band_audits[model], domain=domain)
        for model in MODELS
    ]
    rows = []
    interpretations = []
    for model in MODELS:
        audit = band_audits[model]
        global_snr = float(
            audit["ordinal_decodability"]["raw"][
                "confirmation_class_balanced_snr_db"
            ]
        )
        conditioned = audit["ordinal_decodability"][
            "band_conditioned_confirmation_snr"
        ]
        upper = conditioned["per_band"]["upper"]
        lower = conditioned["per_band"]["lower"]
        macro = conditioned["macro_within_band"]
        complete = (
            upper["retained_labels"] == list(range(1, 11))
            and lower["retained_labels"] == list(range(1, 11))
        )
        macro_text = (
            f"{float(macro['snr_db']):+.2f} dB"
            if complete
            else "不汇总：两带的 k 范围不同"
        )
        rows.append(
            (
                esc(model),
                f"{global_snr:+.2f} dB",
                f"{float(upper['snr_db']):+.2f} dB<br><span class='small'>{esc(_retained_label_text(upper))}; {esc(_support_range(upper))}</span>",
                f"{float(lower['snr_db']):+.2f} dB<br><span class='small'>{esc(_retained_label_text(lower))}; {esc(_support_range(lower))}</span>",
                macro_text,
            )
        )
        if complete:
            direction = "上升" if float(macro["snr_db"]) > global_snr else "下降"
            interpretations.append(
                f"<li><strong>{esc(model)}：</strong>upper 与 lower 都覆盖 k=1…10；"
                f"从 global {global_snr:+.2f} dB 到 equal-band macro "
                f"{float(macro['snr_db']):+.2f} dB（{direction}）。这说明 band offset "
                "确实贡献了同一 k 的跨-template scatter，但该条件化结果仍是 post-hoc nuisance diagnostic。</li>"
            )
        else:
            interpretations.append(
                f"<li><strong>{esc(model)}：</strong>upper 只保留 {_retained_label_text(upper)}，"
                f"lower 保留 {_retained_label_text(lower)}。上下分层本身与 running position "
                "纠缠，因此两条 SNR 不是同一个 10-class estimand；不能用 macro 改写主结论。</li>"
            )
    return f"""
<section id="snr"><h2>SNR：global 与 band-conditioned 要回答两个不同问题</h2>
<div class="definitions two"><div><h3>Global SNR（主指标）</h3><p>完整公式、数值例子和解释边界见上面的 <a href="#metric-guide">指标字典</a>。本节强调其 estimand：同一个 k 若落在不同 trace-template band，band offset 会进入 global within-count noise；因此它衡量的是模型未先验知道 band 时所面对的总变异。</p></div><div><h3>Within-band SNR（混杂诊断）</h3><p>先只用 discovery PCA3 拟合两个 K-means band 并冻结，再把 confirmation 指派到 upper/lower。在每条 band 内以该 band 自己的 grand centroid 计算 signal，并以 (band,k) centroid 计算 residual；因此上下 band 的均值差既不算 signal，也不算 noise。每个 k 在该 band 至少需要 2 states，否则剔除并公开 support。</p></div></div>
<div class="metric-legend"><span><i class="legend-non"></i>non global</span><span><i class="legend-native"></i>native global</span><span><i class="legend-upper"></i>native upper</span><span><i class="legend-lower"></i>native lower</span></div>
<div class="metric-grid">{''.join(charts)}</div>
{table(['模型','Native running · global','Upper · conditional','Lower · conditional','Equal-band macro'], rows)}
<ul>{''.join(interpretations)}</ul>
<div class="callout warning"><strong>解释边界：</strong>条件化能回答“每条 band 内的 count geometry 是否清楚”，但不能取代跨 mode 的 primary global comparison，因为 non-thinking 没有经过同样的 hidden-state-derived 分带。Probe accuracy 与 SNR 也不等价：判别器可利用少数方向，而 SNR 把 PCA16 中的残差能量各向同性相加。</div>
</section>"""


def _capture_manifest_example(
    root: Path,
    model: str,
    site_kind: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    index_path = root / model / "capture_index.jsonl"
    rows = read_jsonl(index_path)
    if not rows:
        raise ValueError(f"Empty capture index: {index_path}")
    row = sorted(rows, key=lambda value: (int(value["seed"]), int(value["gold_count"])))[0]
    manifest_path = index_path.parent / str(row["manifest_path"])
    manifest = read_json(manifest_path)
    sites = [
        value
        for value in manifest["site_rows"]
        if str(value.get("site_kind")) == site_kind
    ]
    if site_kind == "item_end":
        sites = [value for value in sites if int(value.get("occurrence") or 0) == 1]
    if len(sites) != 1:
        raise ValueError(
            f"Expected one {site_kind} example in {manifest_path}; got {len(sites)}"
        )
    return row, sites[0], index_path, manifest_path


def _native_item_end_grammar_table(grammar_registry: Path) -> str:
    rows = read_csv(grammar_registry)
    required = {
        "model_label",
        "request_id",
        "occurrence",
        "grammar_class",
        "item_text",
        "commit_token_text",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Grammar registry is missing required columns: {grammar_registry}")
    keys = [
        (str(row["model_label"]), str(row["request_id"]), int(row["occurrence"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Grammar registry event keys are not unique: {grammar_registry}")
    table_rows = []
    for model in MODELS:
        model_rows = [row for row in rows if str(row["model_label"]) == model]
        grammars = sorted({str(row["grammar_class"]) for row in model_rows})
        for grammar in grammars:
            grammar_rows = [
                row for row in model_rows if str(row["grammar_class"]) == grammar
            ]
            if grammar not in GRAMMAR_ITEM_END_LOCATIONS:
                raise ValueError(f"Unregistered grammar class: {model}/{grammar}")
            example = grammar_rows[0]
            token = json.dumps(
                str(example["commit_token_text"]), ensure_ascii=False
            )
            table_rows.append(
                (
                    esc(model),
                    f"<code>{esc(grammar)}</code>",
                    str(len(grammar_rows)),
                    esc(GRAMMAR_ITEM_END_LOCATIONS[grammar]),
                    (
                        f"<span class=\"small\">{esc(_short_text(example['item_text']))}</span>"
                        f"<br><code>endpoint token = {esc(token)}</code>"
                    ),
                )
            )
    return table(
        ("model", "新版 parser grammar", "states", "item_end 具体位置", "registry 实例"),
        table_rows,
    )


def token_extraction_section(
    native_running_root: Path,
    native_final_root: Path,
    grammar_registry: Path,
) -> tuple[str, list[Path]]:
    running_row, running_site, running_index, running_manifest_path = (
        _capture_manifest_example(
            native_running_root, "Qwen3-8B", "item_end"
        )
    )
    final_row, final_site, final_index, final_manifest_path = (
        _capture_manifest_example(
            native_final_root, "Qwen3-8B", "answer_query_v3"
        )
    )
    if str(running_row["request_id"]) != str(final_row["request_id"]):
        raise ValueError("Running and final token examples do not refer to one trace")
    running_manifest = read_json(running_manifest_path)
    event = running_manifest["episode_parse"]["events"][0]
    prompt_tokens = int(running_manifest["prompt_token_count"])
    running_prefix = int(running_site["prefix_token_count"])
    running_endpoint = int(running_site["endpoint_token"])
    running_global = prompt_tokens + running_endpoint
    final_prefix = int(final_site["prefix_token_count"])
    final_endpoint = int(final_site["endpoint_token"])
    final_global = prompt_tokens + final_endpoint
    if running_endpoint != running_prefix - 1 or final_endpoint != final_prefix - 1:
        raise ValueError("Example endpoint is not the last token of the saved prefix")
    html = f"""
<section id="tokens"><h2>Token 提取：边界、索引与实际例子</h2>
<div class="callout"><strong>统一约定：</strong>所有 token index 都从 0 开始；区间 <code>[start,end)</code> 的 <code>end</code> 不包含在 span 内。报告读取的是所选 decoder block 输出中<strong>边界前最后一个 token</strong>的 post-block hidden state，始终是单 token endpoint，不做 span mean。</div>
<div class="token-flow">
<article><h3>1 · Non-thinking prompt：<code>span_end</code></h3><p>第 k 个 needle 在 prompt tokenizer 下已有精确 token span <code>[sₖ,eₖ)</code>。固定读取 <code>h<sup>(ℓ)</sup>[eₖ−1]</code>；既不取城市前一个 token，也不把整个 needle 平均。下面是索引示意，数字只用于说明 exclusive-end：</p><div class="token-strip"><span>…</span><span data-pos="8421">The</span><span data-pos="8422">Chicago</span><span data-pos="8423">entry</span><span data-pos="8424">scored</span><span data-pos="8425">72</span><span class="picked" data-pos="8426">.</span><b>eₖ=8427</b></div><p class="small">span=<code>[8421,8427)</code>，所以 endpoint=<code>8426</code>。实际 capture 直接由每条 stimulus 的 <code>needle_spans</code> 生成这些位置，并为每个 k 各保存一个 state。</p></article>
<article><h3>2 · Native running：<code>item_end</code></h3><p>Parser 先按新版 surface grammar 在原始 response 中注册第 k 个<strong>完整语义 item</strong>，再对 <code>response[:char_end]</code> 做 exact-prefix token alignment；endpoint 永远是这个完整 item prefix 的最后一个真实 output token。它不是统一的句点，也不是一看到 rank/city 就停止。</p><div class="boundary-example"><span>… {esc(event['city'])} … {esc(event['evidence_surface'])}</span><i></i><span>&lt;/think&gt; …</span></div><p>实际 Qwen 例（seed {int(running_row['seed'])}, gold N={int(running_row['gold_count'])}, k=1）：字符 item=<code>[{int(running_site['char_start'])},{int(running_site['char_end'])})</code>；prefix 有 {running_prefix} 个 output tokens，因此 output endpoint=<code>{running_endpoint}</code>。拼回 prompt 后，global hidden index=<code>{prompt_tokens}+{running_prefix}−1={running_global}</code>。本例边界跨 tokenizer piece，使用 <code>{esc(running_site['alignment_strategy'])}</code>，不是近邻猜测。</p></article>
<article><h3>3 · Native final：<code>answer_query_v3</code></h3><p>独立 parser 寻找最后一个 literal <code>Total: &lt;integer&gt;</code>，边界停在数字首字符之前。因此读取的是“模型即将写出最终数字”时的 prefix endpoint，而不是数字 token 本身。</p><div class="boundary-example"><span>… &lt;/think&gt; Total: </span><i></i><span class="answer-token">1</span></div><p>同一条 Qwen trace：字符 query=<code>[{int(final_site['char_start'])},{int(final_site['char_end'])})</code>；prefix={final_prefix} output tokens，output endpoint=<code>{final_endpoint}</code>，global hidden index=<code>{prompt_tokens}+{final_prefix}−1={final_global}</code>。Final filestream 只保留这一站点，每条 trajectory 恰好一个 state。</p></article>
<article><h3>为什么两侧可以比较</h3><p>Running 比较的是“第 k 个计数单元完成后”的单-token state：prompt needle 完成边界 <code>span_end</code> 对 thinking item 完成边界 <code>item_end</code>。Final 比较的是两侧各自即将输出总数的 query state。语义角色对齐，但 token 字面值、绝对层号与坐标系不要求相同；每种 mode 只在 discovery 内选自己的 layer。</p></article>
</div>
<h3>Native <code>item_end</code> 在每种新版 grammar 中具体落在哪里</h3>
<p class="small">下表直接来自新版 causal-site event registry，共覆盖 Qwen 1,651 与 Gemma 1,564 个 parser-observed states；这些 event 可与 Appendix A 的 geometry states 一一配对。<code>states</code> 是 event 数，不是 trajectory 数。</p>
{_native_item_end_grammar_table(grammar_registry)}
<div class="callout warning"><strong>Representation site 与其他 causal control 不混用。</strong>当前主比较对所有 grammar 都固定读取完整 item 的 <code>item_end</code>；grammar 只决定“完整 item”在表面文本中止于哪里，不参与 layer selector。marker 前后等其他 causal query/control 不替换本报告的 representation site。</div></section>"""
    return html, [
        running_index.resolve(),
        running_manifest_path.resolve(),
        final_index.resolve(),
        final_manifest_path.resolve(),
        grammar_registry.resolve(),
    ]


def _cohort_table(
    summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    categories: Iterable[str],
) -> str:
    rows = []
    categories = tuple(categories)
    for model in MODELS:
        for split in ("all", "confirmation"):
            payload = summary[model][split]
            total = int(payload["total"])
            values = []
            for category in categories:
                count = int(payload["counts"].get(category, 0))
                values.append(
                    f"{count} <span class=\"muted\">({100*count/total:.1f}%)</span>"
                )
            rows.append((esc(model), esc(split), str(total), *values))
    return table(("模型", "cohort", "trajectories", *categories), rows)


def marker_appendix(parser_rows: list[dict[str, Any]]) -> str:
    for model in MODELS:
        model_rows = [
            row for row in parser_rows if str(row.get("model_label")) == model
        ]
        keys = {
            (str(row["split"]), int(row["seed"]), int(row["gold_count"]))
            for row in model_rows
        }
        if len(model_rows) != 300 or keys != expected_trajectory_keys():
            raise ValueError(
                f"{model} parser audit is not the registered 10 x 30 panel: "
                f"rows={len(model_rows)}, unique_keys={len(keys)}"
            )
    legacy = legacy_compatible_marker_summary(parser_rows)
    hybrid = hybrid_marker_summary(parser_rows)
    categories = trace_category_summary(parser_rows)
    return f"""
<section id="appendix-markers"><h2>Appendix A · Trace marker 与 completeness 构成</h2>
<p>比例单位始终是 trajectory，不是 hidden-state。<code>full</code> 的分母为每模型 300，<code>confirmation</code> 的分母为每模型 100；两者都覆盖 N=1…10。旧五类用于与早期 parser/图例兼容，<code>unresolved</code> 是审计状态，不是第六类 marker。</p>
<h3>Legacy-compatible 五类 marker</h3>
{_cohort_table(legacy, FULL_LEGACY_MARKERS)}
{marker_definitions_html()}
<details><summary>Current hybrid parser marker</summary>
<p class="small"><code>inline_count</code> 合并 Count:k、文字 cardinal/ordinal 等连续显式进度事件；<code>evidence_sequence</code> 是只有 city+score 表面顺序、没有显式 ordinal 的保守兜底。该分类用于混杂诊断，不改变固定的 <code>item_end</code> 主站点，也不参与 layer selector。</p>
{_cohort_table(hybrid, HYBRID_MARKERS)}</details>
<details><summary>Trace completeness category</summary>
{_cohort_table(categories, TRACE_CATEGORIES)}</details>
</section>"""


def _association(audit: Mapping[str, Any], column: str, *, full: bool) -> float:
    key = "full_panel_categorical_associations" if full else "categorical_associations"
    rows = {str(row["column"]): row for row in audit[key]}
    return float(rows[column]["nmi"])


def _normalized_mutual_information(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("NMI inputs must be non-empty and equally sized")
    total = float(len(left))

    def entropy(values: list[str]) -> float:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return -sum(
            (count / total) * math.log(count / total) for count in counts.values()
        )

    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    joint: dict[tuple[str, str], int] = {}
    for one, two in zip(left, right):
        left_counts[one] = left_counts.get(one, 0) + 1
        right_counts[two] = right_counts.get(two, 0) + 1
        joint[(one, two)] = joint.get((one, two), 0) + 1
    mutual_information = 0.0
    for (one, two), count in joint.items():
        probability = count / total
        mutual_information += probability * math.log(
            probability
            / ((left_counts[one] / total) * (right_counts[two] / total))
        )
    denominator = (entropy(left) + entropy(right)) / 2.0
    return 1.0 if denominator == 0.0 else mutual_information / denominator


def _band_verdict(
    audit: Mapping[str, Any], grammar_nmi_full: float, grammar_nmi_confirmation: float
) -> str:
    occurrence = _association(audit, "occurrence", full=False)
    if grammar_nmi_confirmation >= 0.30:
        return (
            "新版 grammar 与 frozen upper/lower band 明显对应："
            f"NMI full={grammar_nmi_full:.3f}、confirmation={grammar_nmi_confirmation:.3f}。"
            "因此上下分层主要是多种 trace grammar 的混合，而不是两套 count state；"
            "具体哪类落入哪一层见下表。"
        )
    return (
        "新版 grammar 不能解释主要分层："
        f"NMI full={grammar_nmi_full:.3f}、confirmation={grammar_nmi_confirmation:.3f}，"
        f"而 confirmation running-k NMI={occurrence:.3f}。除少量结构 bullet 外，"
        "同一 grammar 内仍跨越上下 band，因此这里不能复用 Qwen 的 format-mixture 解释。"
    )


def _band_model_block(
    model: str,
    audit: Mapping[str, Any],
    grammar_nmi_full: float,
    grammar_nmi_confirmation: float,
    grammar_rows: list[tuple[str, ...]],
) -> str:
    slug = "qwen" if model.startswith("Qwen") else "gemma"
    scope = audit["scope"]
    raw_full = audit["display_pca3"]["full_panel_two_band"]
    raw_confirmation = audit["display_pca3"]["confirmation_two_band"]
    centered = audit["within_trajectory_centered_pca3"]
    raw_metrics = audit["ordinal_decodability"]["raw"]
    centered_metrics = audit["ordinal_decodability"][
        "within_trajectory_centered_diagnostic"
    ]
    association_rows = [
        (
            "<code>new_parser_grammar</code>",
            f"{grammar_nmi_full:.3f}",
            f"{grammar_nmi_confirmation:.3f}",
        )
    ]
    for column in ("seed", "occurrence", "boundary_kind"):
        association_rows.append(
            (
                f"<code>{esc(column)}</code>",
                f"{_association(audit, column, full=True):.3f}",
                f"{_association(audit, column, full=False):.3f}",
            )
        )
    return f"""
<article class="appendix-model"><h3>{esc(model)}</h3>
<div class="callout"><strong>判读：</strong>{esc(_band_verdict(audit, grammar_nmi_full, grammar_nmi_confirmation))}</div>
<p>固定分析站点为 <code>{esc(audit['site_kind'])}</code> @ L{int(audit['layer'])}。新版 event registry 按 <code>model + request_id + occurrence</code> 与每个 geometry state 一一连接；默认颜色就是新版 grammar。原始图问“上下层是否由 grammar 构成”，centered 图问去掉整条 trajectory offset 后 grammar cloud 是否仍分离。</p>
<div class="controls band-controls"><label>Cohort<select id="band-{slug}-cohort"><option value="all">完整 300 trajectories</option><option value="confirmation">Confirmation 100</option></select></label><label>Color<select id="band-{slug}-color"><option value="grammar">New parser grammar</option><option value="band">Frozen upper/lower band</option><option value="occurrence">Running k</option></select></label></div>
<div class="band-grid"><figure class="band-figure"><h4>Raw · discovery-fitted PCA3</h4><canvas id="band-{slug}-raw" role="img" aria-label="{esc(model)} raw native-thinking geometry in three dimensions"></canvas><p class="rotate-hint">drag to rotate · band centers fitted on discovery only</p><p class="panel-stats" id="band-{slug}-raw-stats"></p></figure><figure class="band-figure"><h4>Trajectory-centered · discovery-fitted PCA3</h4><canvas id="band-{slug}-centered" role="img" aria-label="{esc(model)} trajectory-centered native-thinking geometry in three dimensions"></canvas><p class="rotate-hint">drag to rotate · colors keep the raw frozen-band identity</p><p class="panel-stats" id="band-{slug}-centered-stats"></p></figure></div>
<div class="band-dynamic-legend" id="band-{slug}-legend"></div>
<div class="definitions two"><div><h3>Full 300 view</h3><p>{int(scope['full_trajectories'])} trajectories / {int(scope['full_states'])} states；raw frozen-band silhouette={float(raw_full['silhouette']):.3f}，upper/lower={int(raw_full['cluster_sizes']['upper'])}/{int(raw_full['cluster_sizes']['lower'])}。这里包含 discovery 与 confirmation，只用于描述。</p></div><div><h3>Confirmation 100 view</h3><p>{int(scope['confirmation_trajectories'])} trajectories / {int(scope['confirmation_states'])} states；raw frozen-band silhouette={float(raw_confirmation['silhouette']):.3f}，upper/lower={int(raw_confirmation['cluster_sizes']['upper'])}/{int(raw_confirmation['cluster_sizes']['lower'])}。Band centers 和“upper”命名都已经由 discovery 冻结。</p></div></div>
{table(['candidate nuisance','NMI · full','NMI · confirmation'], association_rows)}
<h4>每种新版 grammar 在 frozen band 中的构成</h4>
{table(('grammar','full states','full lower / upper','confirmation states','confirmation lower / upper'), grammar_rows)}
<div class="definitions two"><div><h3>Band 是否主要是 trace offset</h3><p>Confirmation 中 raw-band 与 centered-space 新 band 的 NMI={float(centered['raw_vs_centered_band_nmi']):.3f}。越接近 0，越说明原上下标签在去 trajectory mean 后不再对应同一分层；越接近 1，越说明分层不是一个简单的 trajectory-level translation。</p></div><div><h3>去 offset 后 count signal 是否还在</h3><p>confirmation Logistic {_pct(raw_metrics['confirmation_logistic_balanced_accuracy'])} → {_pct(centered_metrics['confirmation_logistic_balanced_accuracy'])}；NCC {_pct(raw_metrics['confirmation_ncc_balanced_accuracy'])} → {_pct(centered_metrics['confirmation_ncc_balanced_accuracy'])}；SNR {float(raw_metrics['confirmation_class_balanced_snr_db']):.2f} → {float(centered_metrics['confirmation_class_balanced_snr_db']):.2f} dB。Centering 使用整条 trajectory，只是混杂诊断，不是在线 estimator 或因果干预。</p></div></div>
</article>"""


def _compact_band_points(rows: list[dict[str, str]]) -> list[list[Any]]:
    return [
        [
            str(row["split"]),
            int(row["seed"]),
            int(row["gold_count"]),
            int(row["occurrence"]),
            str(row["marker_kind"]),
            str(row["band"]),
            round(float(row["pc1"]), 5),
            round(float(row["pc2"]), 5),
            round(float(row["pc3"]), 5),
            round(float(row["centered_pc1"]), 5),
            round(float(row["centered_pc2"]), 5),
            round(float(row["centered_pc3"]), 5),
            str(row["grammar_class"]),
        ]
        for row in rows
    ]


def band_appendix(
    band_root: Path,
    grammar_registry: Path,
) -> tuple[str, list[Path], dict[str, Any], dict[str, Mapping[str, Any]]]:
    blocks = []
    inputs: list[Path] = [grammar_registry]
    visual: dict[str, Any] = {}
    audits: dict[str, Mapping[str, Any]] = {}
    registry_rows = read_csv(grammar_registry)
    registry = {
        (
            str(row["model_label"]),
            str(row["request_id"]),
            int(row["occurrence"]),
        ): str(row["grammar_class"])
        for row in registry_rows
    }
    if len(registry) != len(registry_rows):
        raise ValueError(f"Duplicate event keys in grammar registry: {grammar_registry}")
    for model in MODELS:
        directory = band_root / model
        audit_path = directory / "band_diagnostic.json"
        all_points_path = directory / "all_points.csv"
        confirmation_points_path = directory / "confirmation_points.csv"
        audit = read_json(audit_path)
        if str(audit.get("schema_version")) != "native_geometry_band_diagnostic_v2_frozen_band":
            raise ValueError(f"Band diagnostic is not discovery-frozen v2: {audit_path}")
        if str(audit.get("model_label")) != model:
            raise ValueError(f"Band diagnostic model mismatch: {audit_path}")
        if sorted(map(int, audit["scope"]["gold_counts"])) != list(range(1, 11)):
            raise ValueError(f"Band diagnostic is not all-count: {audit_path}")
        if int(audit["scope"]["full_trajectories"]) != 300:
            raise ValueError(f"Band diagnostic is not full-300: {audit_path}")
        if int(audit["scope"]["discovery_trajectories"]) != 200:
            raise ValueError(f"Band diagnostic is not discovery-200: {audit_path}")
        if int(audit["scope"]["confirmation_trajectories"]) != 100:
            raise ValueError(f"Band diagnostic is not confirmation-100: {audit_path}")
        all_points = []
        for source_row in read_csv(all_points_path):
            row = dict(source_row)
            key = (model, str(row["request_id"]), int(row["occurrence"]))
            grammar = registry.get(key)
            if grammar is None:
                raise ValueError(f"Band state missing from grammar registry: {key}")
            row["grammar_class"] = grammar
            all_points.append(row)
        confirmation_points = read_csv(confirmation_points_path)
        filtered_confirmation = [
            row for row in all_points if str(row["split"]) == "confirmation"
        ]
        if len(filtered_confirmation) != len(confirmation_points):
            raise ValueError(f"Band point cohorts disagree: {directory}")
        confirmation_rows = [
            row for row in all_points if str(row["split"]) == "confirmation"
        ]
        grammar_nmi_full = _normalized_mutual_information(
            [str(row["band"]) for row in all_points],
            [str(row["grammar_class"]) for row in all_points],
        )
        grammar_nmi_confirmation = _normalized_mutual_information(
            [str(row["band"]) for row in confirmation_rows],
            [str(row["grammar_class"]) for row in confirmation_rows],
        )
        grammar_table_rows = []
        for grammar in sorted({str(row["grammar_class"]) for row in all_points}):
            full_group = [
                row for row in all_points if str(row["grammar_class"]) == grammar
            ]
            confirmation_group = [
                row
                for row in confirmation_rows
                if str(row["grammar_class"]) == grammar
            ]

            def band_mix(group: list[dict[str, str]]) -> str:
                if not group:
                    return "n/a"
                lower = sum(str(row["band"]) == "lower" for row in group)
                upper = len(group) - lower
                return (
                    f"{lower} ({100*lower/len(group):.1f}%) / "
                    f"{upper} ({100*upper/len(group):.1f}%)"
                )

            grammar_table_rows.append(
                (
                    f"<code>{esc(grammar)}</code>",
                    str(len(full_group)),
                    band_mix(full_group),
                    str(len(confirmation_group)),
                    band_mix(confirmation_group),
                )
            )
        blocks.append(
            _band_model_block(
                model,
                audit,
                grammar_nmi_full,
                grammar_nmi_confirmation,
                grammar_table_rows,
            )
        )
        audits[model] = audit
        visual[model] = {
            "layer": int(audit["layer"]),
            "site": str(audit["site_kind"]),
            "raw_evr": [
                round(float(value), 7)
                for value in audit["display_pca3"]["explained_variance_ratio"]
            ],
            "centered_evr": [
                round(float(value), 7)
                for value in audit["within_trajectory_centered_pca3"][
                    "explained_variance_ratio"
                ]
            ],
            "points": _compact_band_points(all_points),
        }
        inputs.extend((audit_path, all_points_path, confirmation_points_path))
    return f"""
<section id="appendix-bands"><h2>Appendix A · Native-thinking 上下分层是否由新版 grammar 造成</h2>
<div class="callout warning"><strong>结论先说：</strong>Qwen 的上下层很大程度由新版 grammar 分开（full/confirmation NMI 约 0.60/0.61）；Gemma 的对应 NMI 只有约 0.03/0.06，除 invariant bullet 外，同一 grammar 仍跨越两层。因此“上下两层=grammar mixture”只适用于 Qwen，不能外推成跨模型机制。</div>
<div class="definitions"><div><h3>Step 1 · 冻结分带</h3><p>StandardScaler、PCA3 与两类 K-means center 都只在 discovery states 上拟合，再冻结分配 full/confirmation 的 upper/lower。“上下”只是显示名。</p></div><div><h3>Step 2 · 连接新版 grammar</h3><p>用新版 event registry 按每个 occurrence 精确连接 grammar，而不是沿用 trajectory-level 旧 marker。报告 grammar–band NMI 与逐 grammar 的 lower/upper 构成。</p></div><div><h3>Step 3 · 去 trajectory offset</h3><p>逐 trajectory 在原 hidden space 减去自己的 mean，再重拟合 discovery PCA3，用来判断 grammar cloud 是否主要是整条 trace 的平移。</p></div></div>
<div class="callout warning"><strong>不要把两团直接叫两个计数器：</strong>PCA3 只是总方差的低维显示，K-means 又强制给出两组。必须同时看 frozen confirmation silhouette、NMI、support、trajectory centering 与 within-band SNR；任何单张图都不足以识别机制。</div>
{''.join(blocks)}
</section>""", inputs, visual, audits


def _grammar_filter_accuracy_svg(
    indexed: Mapping[tuple[str, str], Mapping[str, str]],
) -> str:
    width, height = 920, 315
    x0, x1 = 255.0, 735.0

    def sx(value: float) -> float:
        return x0 + (float(value) - 0.1) / 0.9 * (x1 - x0)

    elements = []
    for tick in (0.1, 0.3, 0.5, 0.7, 0.9):
        x = sx(tick)
        elements.append(
            f'<line class="metric-gridline" x1="{x:.1f}" y1="38" '
            f'x2="{x:.1f}" y2="270"/><text class="metric-tick" '
            f'x="{x:.1f}" y="27" text-anchor="middle">{100*tick:.0f}%</text>'
        )
    row_y = 65
    for model in MODELS:
        elements.append(
            f'<text class="metric-label" x="12" y="{row_y+18}">{esc(model)}</text>'
        )
        for label, field in (
            ("Logistic", "confirmation_logistic_balanced_accuracy"),
            ("Nearest centroid", "confirmation_ncc_balanced_accuracy"),
        ):
            y = row_y if label == "Logistic" else row_y + 42
            non = float(indexed[(model, "non_thinking")][field])
            native = float(indexed[(model, "native_thinking")][field])
            elements.append(
                f'<text class="metric-label" x="118" y="{y+4}">{esc(label)}</text>'
                f'<line class="metric-link" x1="{sx(non):.1f}" y1="{y}" '
                f'x2="{sx(native):.1f}" y2="{y}"/>'
                f'<circle class="metric-dot metric-non" cx="{sx(non):.1f}" cy="{y}" r="6"/>'
                f'<circle class="metric-dot metric-native" cx="{sx(native):.1f}" cy="{y}" r="6"/>'
                f'<text class="metric-value" x="755" y="{y+4}">'
                f'{100*non:.1f}% → {100*native:.1f}%</text>'
            )
        row_y += 116
    return (
        f'<figure class="metric-figure"><h3>Grammar-filtered paired confirmation BAcc</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Held-out balanced accuracy after native grammar filtering and exact event pairing">'
        '<title>Grammar-filtered paired confirmation balanced accuracy</title>'
        '<desc>Dark circles are non-thinking and teal circles are native-thinking. '
        'The event cells are paired and each mode uses its own discovery-selected layer.</desc>'
        + "".join(elements)
        + '<text class="metric-axis-title" x="495" y="298" text-anchor="middle">'
        'frozen confirmation balanced accuracy</text></svg>'
        '<div class="metric-legend"><span><i class="legend-non"></i>non-thinking</span>'
        '<span><i class="legend-native"></i>native-thinking</span></div></figure>'
    )


def grammar_filtered_comparison(
    root: Path,
) -> tuple[str, list[Path], dict[str, Any]]:
    metrics_path = root / "paired_metrics.csv"
    candidates_path = root / "nonthinking_layer_candidates.csv"
    payload_path = root / "geometry_payload.json"
    audit_path = root / "audit.json"
    metrics = read_csv(metrics_path)
    payload = read_json(payload_path)
    audit = read_json(audit_path)
    expected = "realistic_niah_v5_grammar_filtered_cross_mode_geometry_v1"
    if str(payload.get("schema_version")) != expected or str(audit.get("schema_version")) != expected:
        raise ValueError(f"Unexpected grammar-filtered comparison schema in {root}")
    indexed = {
        (str(row["model_label"]), str(row["mode"])): row for row in metrics
    }
    if set(indexed) != {
        (model, mode)
        for model in MODELS
        for mode in ("non_thinking", "native_thinking")
    }:
        raise ValueError(f"Grammar-filtered metric panel is incomplete: {metrics_path}")
    rows = []
    verdicts = []
    cards = []
    for model in MODELS:
        model_payload = payload["models"][model]
        grammar = str(model_payload["grammar_class"])
        non = indexed[(model, "non_thinking")]
        native = indexed[(model, "native_thinking")]
        log_gap = 100 * (
            float(native["confirmation_logistic_balanced_accuracy"])
            - float(non["confirmation_logistic_balanced_accuracy"])
        )
        ncc_gap = 100 * (
            float(native["confirmation_ncc_balanced_accuracy"])
            - float(non["confirmation_ncc_balanced_accuracy"])
        )
        sil_gap = (
            float(native["confirmation_pca3_class_balanced_silhouette"])
            - float(non["confirmation_pca3_class_balanced_silhouette"])
        )
        rsa_gap = (
            float(native["confirmation_pca3_ordinal_rsa"])
            - float(non["confirmation_pca3_ordinal_rsa"])
        )
        verdicts.append(
            f"{model}: Log/NCC +{log_gap:.1f}/+{ncc_gap:.1f} pp；"
            f"PCA3 silhouette Δ={sil_gap:+.3f}，ordinal RSA Δ={rsa_gap:+.3f}"
        )
        for mode, label in (("non_thinking", "non-thinking"), ("native_thinking", "native-thinking")):
            row = indexed[(model, mode)]
            rows.append(
                (
                    esc(model),
                    label,
                    f"<code>{esc(grammar)}</code>",
                    f"L{int(float(row['layer']))}",
                    str(int(float(row["states"]))),
                    str(int(float(row["confirmation_min_per_class"]))),
                    f"{_pct(row['confirmation_logistic_balanced_accuracy'])} / "
                    f"{_pct(row['confirmation_ncc_balanced_accuracy'])}",
                    f"{float(row['confirmation_class_balanced_snr_db']):.2f} dB",
                    f"{float(row['confirmation_pca3_class_balanced_silhouette']):.3f}",
                    f"{float(row['confirmation_pca3_ordinal_rsa']):.3f}",
                )
            )
        slug = "qwen" if model.startswith("Qwen") else "gemma"
        cards.append(
            f"""
<article class="appendix-model"><h3>{esc(model)} · <code>{esc(grammar)}</code></h3>
<div class="controls"><label>Rows<select id="grammar-filter-{slug}-cohort"><option value="confirmation">Confirmation only</option><option value="all">Discovery + confirmation</option></select></label></div>
<div class="dual-grid">
<figure class="geometry-card"><h3>Paired non-thinking</h3><canvas id="grammar-filter-{slug}-non" role="img" aria-label="{esc(model)} grammar-filter paired non-thinking PCA3"></canvas><p class="rotate-hint">drag to rotate · independently discovery-selected layer and PCA3</p><p class="panel-stats" id="grammar-filter-{slug}-non-stats"></p></figure>
<figure class="geometry-card"><h3>Grammar-filtered native-thinking</h3><canvas id="grammar-filter-{slug}-native" role="img" aria-label="{esc(model)} grammar-filtered native-thinking PCA3"></canvas><p class="rotate-hint">drag to rotate · native-only discovery filter, frozen confirmation display</p><p class="panel-stats" id="grammar-filter-{slug}-native-stats"></p></figure>
</div></article>"""
        )
    return f"""
<section id="appendix-grammar-filter"><h2>Appendix C · 单一 thinking-trace grammar 的 3D PCA</h2>
<div class="callout"><strong>结论：</strong>filter 能把 native-thinking 的<strong>线性可读性优势</strong>显著放大，但不能把它改写成两个模型都具有更高 PCA3 cluster tightness。{'；'.join(verdicts)}。因此这组图适合说明“控制 native grammar 后 count path 更可读”，不宜单独 claim“所有簇都更紧”。</div>
<div class="definitions"><div><h3>Filter 如何冻结</h3><p>每个模型只用 native discovery 的 grouped-CV 选择一个覆盖 k=1…10 的 grammar × native layer；confirmation 不参与选择。</p></div><div><h3>左右如何配对</h3><p>筛中 native event 后，按 <code>split + seed + gold N + running k</code> 从 non-thinking 取完全相同的任务单元；左右 state 数、类别支持完全相同。</p></div><div><h3>Non-thinking 如何选层</h3><p>在这批 paired discovery rows 上独立选择自己的最佳层；没有把 native 的层强加给 non-thinking，也没有用 confirmation 挑视角。</p></div></div>
{table(('model','mode','native filter grammar','layer','states','C min nₖ','C Log / NCC','C SNR','C PCA3 silhouette','C PCA3 ordinal RSA'), rows)}
{_grammar_filter_accuracy_svg(indexed)}
{''.join(cards)}
<div class="callout warning"><strong>Estimand 边界：</strong>grammar 是 native trace 的结果变量，non-thinking 本身没有该 grammar。这是条件于“native 生成了 grammar G”的 paired subgroup analysis，不是随机化的 grammar intervention，也不替代上面的全样本主比较。Gemma confirmation 的 k=10 只有 2 个 states，尾部 centroid 仍属低支持。</div>
</section>""", [metrics_path, candidates_path, payload_path, audit_path], payload


def pure_trace_n10_comparison(
    root: Path,
) -> tuple[str, list[Path], dict[str, Any]]:
    metrics_path = root / "paired_metrics.csv"
    selected_path = root / "selected_pure_trace_grammar.csv"
    support_path = root / "pure_trace_grammar_support.csv"
    candidates_path = root / "layer_candidates.csv"
    payload_path = root / "geometry_payload.json"
    audit_path = root / "audit.json"
    metrics = read_csv(metrics_path)
    selected = read_csv(selected_path)
    payload = read_json(payload_path)
    audit = read_json(audit_path)
    expected = {
        "realistic_niah_v5_pure_trace_n10_cross_mode_geometry_v1",
        "realistic_niah_v5_pure_trace_n10_cross_mode_geometry_v2",
    }
    if (
        str(payload.get("schema_version")) not in expected
        or str(audit.get("schema_version")) not in expected
    ):
        raise ValueError(f"Unexpected pure-trace N=10 schema in {root}")
    supplemented = str(payload.get("schema_version")).endswith("_v2")
    indexed = {
        (str(row["model_label"]), str(row["mode"])): row for row in metrics
    }
    expected_keys = {
        (model, mode)
        for model in MODELS
        for mode in ("non_thinking", "native_thinking")
    }
    if set(indexed) != expected_keys:
        raise ValueError(f"Pure-trace N=10 metric panel is incomplete: {metrics_path}")
    selected_by_model = {str(row["model_label"]): row for row in selected}
    if set(selected_by_model) != set(MODELS):
        raise ValueError(f"Pure-trace N=10 selections are incomplete: {selected_path}")

    rows = []
    cards = []
    summaries = []
    for model in MODELS:
        chosen = selected_by_model[model]
        grammar = str(chosen["grammar_class"])
        discovery_n = int(float(chosen["discovery_trajectories"]))
        confirmation_n = int(float(chosen["confirmation_trajectories"]))
        non = indexed[(model, "non_thinking")]
        native = indexed[(model, "native_thinking")]
        log_gap = 100 * (
            float(native["confirmation_logistic_balanced_accuracy"])
            - float(non["confirmation_logistic_balanced_accuracy"])
        )
        ncc_gap = 100 * (
            float(native["confirmation_ncc_balanced_accuracy"])
            - float(non["confirmation_ncc_balanced_accuracy"])
        )
        summaries.append(
            f"{model}: D/C={discovery_n}/{confirmation_n} 条，"
            f"Log/NCC gap={log_gap:+.1f}/{ncc_gap:+.1f} pp"
        )
        for mode, label in (
            ("non_thinking", "non-thinking"),
            ("native_thinking", "native-thinking"),
        ):
            row = indexed[(model, mode)]
            rows.append(
                (
                    esc(model),
                    label,
                    f"<code>{esc(grammar)}</code>",
                    f"L{int(float(row['layer']))}",
                    f"{discovery_n} / {confirmation_n}",
                    str(int(float(row["confirmation_rows"]))),
                    f"{_pct(row['confirmation_logistic_balanced_accuracy'])} / "
                    f"{_pct(row['confirmation_ncc_balanced_accuracy'])}",
                    f"{float(row['confirmation_class_balanced_snr_db']):.2f} dB",
                    f"{float(row['confirmation_pca3_class_balanced_silhouette']):.3f}",
                    f"{float(row['confirmation_pca3_ordinal_rsa']):.3f}",
                )
            )
        slug = "qwen" if model.startswith("Qwen") else "gemma"
        cards.append(
            f"""
<article class="appendix-model"><h3>{esc(model)} · N=10 · <code>{esc(grammar)}</code></h3>
<div class="controls"><label>Rows<select id="pure-trace-n10-{slug}-cohort"><option value="confirmation">Confirmation only</option><option value="all">Discovery + confirmation</option></select></label></div>
<div class="dual-grid">
<figure class="geometry-card"><h3>Paired non-thinking</h3><canvas id="pure-trace-n10-{slug}-non" role="img" aria-label="{esc(model)} N=10 pure-trace paired non-thinking PCA3"></canvas><p class="rotate-hint">drag to rotate · faint lines are individual k=1→10 trajectories</p><p class="panel-stats" id="pure-trace-n10-{slug}-non-stats"></p></figure>
<figure class="geometry-card"><h3>Whole-trace-pure native-thinking</h3><canvas id="pure-trace-n10-{slug}-native" role="img" aria-label="{esc(model)} N=10 whole-trace-pure native-thinking PCA3"></canvas><p class="rotate-hint">drag to rotate · same seeds and k cells; independent discovery-selected layer/PCA basis</p><p class="panel-stats" id="pure-trace-n10-{slug}-native-stats"></p></figure>
</div></article>"""
        )

    return f"""
<section id="appendix-pure-trace-n10"><h2>Appendix D · N=10 整条 trace 单一 grammar</h2>
<div class="callout"><strong>{'连续 seed 补样结果' if supplemented else '当前探索性结果'}：</strong>{'；'.join(summaries)}。Qwen 的线性可读性差距很大，但 raw PCA3 silhouette 是否更紧仍须按数值如实判断；Gemma 若支持量仍小则不解释方向。{'Grammar/marker target 在补样生成前已由原 discovery panel 冻结；新增 discovery 只拟合 layer/PCA，新增 confirmation 只评价。' if supplemented else '新增连续 seed 的独立补样正在单独进行，下面先如实展示已注册 30-seed panel。'}</div>
<div class="definitions"><div><h3>为什么只取 N=10</h3><p>每条轨迹都必须真实提供 k=1…10，因而左右都能画完整 counter path；不把较小 N 的短轨迹混入，也不补齐缺失 item。</p></div><div><h3>“纯 grammar”如何定义</h3><p>整条 native trace 必须 exact-count、one-to-one，含十个唯一 progress commits，rank 与 occurrence 都严格为 1…10，而且十个 item 共用同一个 <code>grammar_class</code> 与 <code>marker_kind</code>。</p></div><div><h3>如何避免挑图</h3><p>grammar 只按 native discovery 中合格的完整轨迹数最大化；两个 mode 再各自用 discovery grouped-CV 选层。Confirmation 不参与 grammar、layer 或 PCA basis 的选择。</p></div></div>
{table(('model','mode','whole-trace grammar','layer','D / C traces','C states','C Log / NCC','C SNR','C PCA3 silhouette','C PCA3 ordinal RSA'), rows)}
{''.join(cards)}
<div class="callout warning"><strong>解释边界：</strong>这是条件于 native 生成一整条纯 grammar 的 subgroup analysis，不是 grammar intervention。每个 retained seed 在 non-thinking 侧按 <code>split + seed + N=10 + k</code> 精确配对。独立 PCA 坐标只允许比较形状、轨迹一致性与冻结指标，不能比较左右的绝对坐标距离。</div>
</section>""", [
        metrics_path,
        selected_path,
        support_path,
        candidates_path,
        payload_path,
        audit_path,
    ], payload


def indexed_numeric_n10_comparison(
    root: Path,
    gemma_root: Path | None = None,
    gemma_premarker_root: Path | None = None,
) -> tuple[str, list[Path], dict[str, Any]]:
    """Render strict single-surface N=10 secondary panels as Appendix C."""

    metrics_path = root / "paired_metrics.csv"
    selected_path = root / "selected_strict_dash_20_10_trajectories.csv"
    candidates_path = root / "layer_candidates.csv"
    payload_path = root / "geometry_payload.json"
    audit_path = root / "audit.json"
    metrics = read_csv(metrics_path)
    selected = read_csv(selected_path)
    payload = read_json(payload_path)
    audit = read_json(audit_path)
    expected = (
        "realistic_niah_v5_indexed_numeric_n10_strict_dash_20_10_geometry_v1"
    )
    if (
        str(payload.get("schema_version")) != expected
        or str(audit.get("schema_version")) != expected
    ):
        raise ValueError(f"Unexpected indexed-numeric N=10 schema in {root}")
    model = "Qwen3-8B"
    model_payload = payload.get("models", {}).get(model)
    if not isinstance(model_payload, Mapping):
        raise ValueError(f"Indexed-numeric payload lacks {model}: {payload_path}")
    indexed = {str(row["mode"]): row for row in metrics}
    if set(indexed) != {"non_thinking", "native_thinking"}:
        raise ValueError(f"Indexed-numeric metrics are incomplete: {metrics_path}")
    discovery_seeds = sorted(
        int(float(row["seed"]))
        for row in selected
        if str(row["split"]) == "discovery"
    )
    confirmation_seeds = sorted(
        int(float(row["seed"]))
        for row in selected
        if str(row["split"]) == "confirmation"
    )
    if len(discovery_seeds) != 20 or len(confirmation_seeds) != 10:
        raise ValueError(
            "Indexed-numeric seed panel changed unexpectedly: "
            f"D/C={len(discovery_seeds)}/{len(confirmation_seeds)}"
        )
    rows = []
    layer_selects: dict[str, str] = {}
    for mode, label, short in (
        ("non_thinking", "non-thinking", "non"),
        ("native_thinking", "native-thinking", "native"),
    ):
        row = indexed[mode]
        mode_payload = model_payload[mode]
        selected_layer = int(mode_payload["selected_layer"])
        layers = sorted(int(value) for value in mode_payload["layers"])
        if layers != list(range(36)):
            raise ValueError(f"Indexed-numeric {mode} layer grid is incomplete")
        layer_selects[short] = "".join(
            f'<option value="{layer}"'
            f'{" selected" if layer == selected_layer else ""}>L{layer}'
            f'{" · discovery best" if layer == selected_layer else ""}</option>'
            for layer in layers
        )
        rows.append(
            (
                label,
                "span_end" if mode == "non_thinking" else "item_end = score digit",
                f"L{selected_layer}",
                f"{len(discovery_seeds)} / {len(confirmation_seeds)}",
                str(int(float(row["confirmation_rows"]))),
                f"{_pct(row['confirmation_logistic_balanced_accuracy'])} / "
                f"{_pct(row['confirmation_ncc_balanced_accuracy'])}",
                f"{float(row['confirmation_class_balanced_snr_db']):.2f} dB",
                f"{float(row['confirmation_pca3_class_balanced_silhouette']):.3f}",
                f"{float(row['confirmation_pca3_ordinal_rsa']):.3f}",
            )
        )
    non = indexed["non_thinking"]
    native = indexed["native_thinking"]
    log_gap = 100 * (
        float(native["confirmation_logistic_balanced_accuracy"])
        - float(non["confirmation_logistic_balanced_accuracy"])
    )
    ncc_gap = 100 * (
        float(native["confirmation_ncc_balanced_accuracy"])
        - float(non["confirmation_ncc_balanced_accuracy"])
    )
    discovery_text = ", ".join(map(str, discovery_seeds))
    confirmation_text = ", ".join(map(str, confirmation_seeds))
    html = f"""
<section id="appendix-indexed-numeric-n10"><h2>Appendix C · N=10 单一 surface 主簇</h2>
<div class="callout"><strong>Qwen 结果：</strong>排除 <code>city received a score of …</code> 长句支簇，只保留整条十项均为 <code>k. city - score</code>、且 <code>item_end</code> 落在 score digit 的轨迹后，secondary D/C={len(discovery_seeds)}/{len(confirmation_seeds)}；冻结 confirmation 的 Log/NCC gap 为 {log_gap:+.1f}/{ncc_gap:+.1f} pp。</div>
<div class="definitions"><div><h3>Qwen grammar</h3><p>每条 native trace 必须 exact、one-to-one、N=10；十个 item 都精确匹配 <code>k. city - score</code>，例如 <code>7. Seattle - 84</code>。显式 running index <code>7.</code> 已在同一个 item 开头；<code>item_end</code> 读取 score 的末 token <code>84</code>。</p></div><div><h3>Hidden state 在哪里</h3><p>若 output endpoint token 为 <code>t</code>，图中 Lℓ 使用该 token 经过 decoder block ℓ 后的 residual state <code>h^(ℓ)[prompt_tokens + t]</code>。它是 post-block、single-token state，能够看到截至 endpoint 的全部自回归上下文；不是 token embedding、span mean 或下一 token state。</p></div><div><h3>20/10 与选层</h3><p>该 surface rule 是诊断旧分簇后的 exploratory analysis。30 个 text-eligible seeds 只按固定 seed hash 分成 20 discovery / 10 confirmation；层和每层 PCA3 均只用 discovery 拟合，confirmation 冻结评价。图不画 per-seed 折线。</p></div></div>
{table(('mode','endpoint','default layer','D / C traces','C states','C Log / NCC','C SNR','C PCA3 silhouette','C PCA3 ordinal RSA'), rows)}
<div class="callout"><strong>Discovery seeds：</strong><code class="seed-list">{esc(discovery_text)}</code><br><strong>Confirmation seeds：</strong><code class="seed-list">{esc(confirmation_text)}</code></div>
<article class="appendix-model"><h3>Qwen3-8B · N=10 · strict <code>k. city - score</code></h3>
<div class="controls"><label>Rows<select id="indexed-numeric-n10-qwen-cohort"><option value="confirmation">Confirmation only</option><option value="all">Discovery + confirmation</option></select></label></div>
<div class="dual-grid">
<figure class="geometry-card"><h3>Paired non-thinking</h3><label class="layer-control">Layer<select id="indexed-numeric-n10-qwen-non-layer">{layer_selects['non']}</select></label><canvas id="indexed-numeric-n10-qwen-non" role="img" aria-label="Qwen strict city-dash-score paired non-thinking PCA3"></canvas><p class="rotate-hint">drag to rotate · scatter plus count-centroid path; no per-seed lines</p><p class="panel-stats" id="indexed-numeric-n10-qwen-non-stats"></p></figure>
<figure class="geometry-card"><h3>Strict main-cluster native-thinking</h3><label class="layer-control">Layer<select id="indexed-numeric-n10-qwen-native-layer">{layer_selects['native']}</select></label><canvas id="indexed-numeric-n10-qwen-native" role="img" aria-label="Qwen strict city-dash-score native-thinking PCA3"></canvas><p class="rotate-hint">drag to rotate · independent layer and discovery-fitted PCA basis</p><p class="panel-stats" id="indexed-numeric-n10-qwen-native-stats"></p></figure>
</div></article>
<div class="callout warning"><strong>共同解释边界：</strong>Qwen 与 Gemma 的主图都把显式 running index 当作 grammar 的一部分，而不是试图构造 label-free latent-counter test。因而可以比较“固定显式 grammar 后的 representation geometry / decodability”，不能把高准确率单独解释成模型形成了抽象离散计数器。两组都是诊断旧分簇后定义的 exploratory paired subgroup，不是新的预注册 confirmation。</div>
</section>"""
    inputs = [metrics_path, selected_path, candidates_path, payload_path, audit_path]
    if gemma_root is None:
        return html, inputs, payload

    gemma_metrics_path = gemma_root / "paired_metrics.csv"
    gemma_selected_path = gemma_root / "selected_trajectories.csv"
    gemma_candidates_path = gemma_root / "layer_candidates.csv"
    gemma_payload_path = gemma_root / "geometry_payload.json"
    gemma_audit_path = gemma_root / "audit.json"
    gemma_metrics = read_csv(gemma_metrics_path)
    gemma_selected = read_csv(gemma_selected_path)
    gemma_candidates = read_csv(gemma_candidates_path)
    gemma_payload = read_json(gemma_payload_path)
    gemma_audit = read_json(gemma_audit_path)
    gemma_expected = "realistic_niah_v5_gemma_inline_count_n10_geometry_v2"
    if (
        str(gemma_payload.get("schema_version")) != gemma_expected
        or str(gemma_audit.get("schema_version")) != gemma_expected
    ):
        raise ValueError(f"Unexpected Gemma count-colon schema in {gemma_root}")
    gemma_model = "Gemma4-E4B"
    gemma_model_payload = gemma_payload.get("models", {}).get(gemma_model)
    if not isinstance(gemma_model_payload, Mapping):
        raise ValueError(f"Gemma count-colon payload lacks {gemma_model}")
    gemma_native_site = str(gemma_payload.get("native_site_kind", "item_end"))
    if gemma_native_site != "item_end":
        raise ValueError(
            "The report's explicit-grammar Gemma panel must use item_end; "
            f"got {gemma_native_site!r}"
        )
    gemma_family = str(gemma_payload.get("surface_family", "count_colon"))
    gemma_prefix_record = gemma_family == "controlled_prefix_record"
    if gemma_family not in {"count_colon", "controlled_prefix_record"}:
        raise ValueError(f"Unsupported Gemma Appendix C surface family: {gemma_family}")
    gemma_model_payload["surface_label"] = (
        "controlled Record-k prefix core; item-end after city/score"
        if gemma_prefix_record
        else "single-episode (Count: k), item-end read"
    )
    gemma_model_payload["native_site_kind"] = gemma_native_site
    gemma_indexed = {str(row["mode"]): row for row in gemma_metrics}
    if set(gemma_indexed) != {"non_thinking", "native_thinking"}:
        raise ValueError(f"Gemma count-colon metrics are incomplete: {gemma_metrics_path}")
    gemma_discovery_seeds = sorted(
        int(float(row["seed"]))
        for row in gemma_selected
        if str(row["split"]) == "discovery"
    )
    gemma_confirmation_seeds = sorted(
        int(float(row["seed"]))
        for row in gemma_selected
        if str(row["split"]) == "confirmation"
    )
    if len(gemma_discovery_seeds) != 20 or len(gemma_confirmation_seeds) != 10:
        raise ValueError(
            "Gemma count-colon seed panel changed unexpectedly: "
            f"D/C={len(gemma_discovery_seeds)}/{len(gemma_confirmation_seeds)}"
        )
    gemma_rows = []
    gemma_layer_selects: dict[str, str] = {}
    for mode, label, short in (
        ("non_thinking", "non-thinking", "non"),
        ("native_thinking", "native-thinking", "native"),
    ):
        row = gemma_indexed[mode]
        mode_payload = gemma_model_payload[mode]
        selected_layer = int(mode_payload["selected_layer"])
        layers = sorted(int(value) for value in mode_payload["layers"])
        if layers != list(range(max(layers) + 1)):
            raise ValueError(f"Gemma count-colon {mode} layer grid is incomplete")
        gemma_layer_selects[short] = "".join(
            f'<option value="{layer}"'
            f'{" selected" if layer == selected_layer else ""}>L{layer}'
            f'{" · discovery best" if layer == selected_layer else ""}</option>'
            for layer in layers
        )
        gemma_rows.append(
            (
                label,
                (
                    "span_end"
                    if mode == "non_thinking"
                    else (
                        "item_end after city/score"
                        if gemma_prefix_record
                        else "item_end = closing )"
                    )
                ),
                f"L{selected_layer}",
                f"{len(gemma_discovery_seeds)} / {len(gemma_confirmation_seeds)}",
                str(int(float(row["confirmation_rows"]))),
                f"{_pct(row['confirmation_logistic_balanced_accuracy'])} / "
                f"{_pct(row['confirmation_ncc_balanced_accuracy'])}",
                f"{float(row['confirmation_class_balanced_snr_db']):.2f} dB",
                f"{float(row['confirmation_pca3_class_balanced_silhouette']):.3f}",
                f"{float(row['confirmation_pca3_ordinal_rsa']):.3f}",
            )
        )
    gemma_non = gemma_indexed["non_thinking"]
    gemma_native = gemma_indexed["native_thinking"]
    gemma_native_candidates = [
        row
        for row in gemma_candidates
        if str(row.get("mode")) == "native_thinking"
    ]
    gemma_best_score = max(
        float(row["discovery_selection_score"])
        for row in gemma_native_candidates
    )
    gemma_tied_best_layers = sorted(
        int(float(row["layer"]))
        for row in gemma_native_candidates
        if math.isclose(
            float(row["discovery_selection_score"]),
            gemma_best_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    gemma_tie_text = ", ".join(f"L{layer}" for layer in gemma_tied_best_layers)
    gemma_log_gap = 100 * (
        float(gemma_native["confirmation_logistic_balanced_accuracy"])
        - float(gemma_non["confirmation_logistic_balanced_accuracy"])
    )
    gemma_ncc_gap = 100 * (
        float(gemma_native["confirmation_ncc_balanced_accuracy"])
        - float(gemma_non["confirmation_ncc_balanced_accuracy"])
    )
    filter_audit = gemma_payload.get("generation_filter_audit", [])
    filter_text = "；".join(
        f"{int(row['retained_trajectories'])}/{int(row['raw_trajectories'])}"
        for row in filter_audit
    ) or "未提供"
    gemma_generated_candidates = sum(
        int(row.get("raw_trajectories", 0)) for row in filter_audit
    )
    prem_rows: list[dict[str, str]] = []
    prem_inputs: list[Path] = []
    if gemma_premarker_root is not None and not gemma_prefix_record:
        prem_metrics_path = gemma_premarker_root / "paired_metrics.csv"
        prem_payload_path = gemma_premarker_root / "geometry_payload.json"
        prem_candidates_path = gemma_premarker_root / "layer_candidates.csv"
        prem_selected_path = gemma_premarker_root / "selected_trajectories.csv"
        prem_audit_path = gemma_premarker_root / "audit.json"
        prem_payload = read_json(prem_payload_path)
        prem_audit = read_json(prem_audit_path)
        if (
            str(prem_payload.get("schema_version")) != gemma_expected
            or str(prem_audit.get("schema_version")) != gemma_expected
            or str(prem_payload.get("native_site_kind")) != "pre_marker"
        ):
            raise ValueError(f"Unexpected Gemma pre-marker control in {gemma_premarker_root}")
        prem_rows = read_csv(prem_metrics_path)
        prem_inputs = [
            prem_metrics_path,
            prem_payload_path,
            prem_candidates_path,
            prem_selected_path,
            prem_audit_path,
        ]
    prem_native = next(
        (row for row in prem_rows if str(row.get("mode")) == "native_thinking"),
        None,
    )
    prem_control_html = (
        f'<div class="callout"><strong>去显式标签 sensitivity control：</strong>'
        f'把同一批 Gemma trajectories 的 endpoint 前移到 <code>pre_marker</code>（city/score evidence 已结束，'
        f'<code>Count: k</code> 尚未开始）后，discovery 最佳层为 L{int(float(prem_native["layer"]))}，'
        f'冻结 confirmation Log/NCC 为 {_pct(prem_native["confirmation_logistic_balanced_accuracy"])} / '
        f'{_pct(prem_native["confirmation_ncc_balanced_accuracy"])}。这说明主图的 L1/100% 含有显式 surface cue，'
        f'但移除该 cue 后仍有较强的 retrieval-complete count decodability。</div>'
        if prem_native is not None
        else ""
    )
    if gemma_prefix_record:
        gemma_title = (
            "Gemma4-E4B · N=10 · controlled "
            "<code>Record k … city … score</code> prefix core"
        )
        gemma_result_text = (
            "用单独 controlled prompt 诱导结构性不同的 prefix grammar；整条轨迹必须 "
            "exact、one-to-one，并且十个 selected events 都由 <code>Record k</code> "
            "先给出 running index，随后在同一完整 item 中出现 city 与 score。"
            "Markdown bullet/bold 仅记为外壳，不改变 parser core。"
        )
        gemma_definitions_html = f"""
<div class="definitions"><div><h3>Gemma grammar</h3><p>冻结 parser core 与本地 Markdown shell 为 <code>same_unit_rank_before_city / inline_count</code>：每个 item 必须精确是 <code>*&nbsp;&nbsp;&nbsp;Record 7: (Seattle, 84)</code>。<code>Record 7</code> 在 Seattle 与 84 之前；plain-line、粗体/斜体、<code>(Count: 7)</code> suffix、city-before-rank 与漏数轨迹全部排除。</p></div><div><h3>Hidden state 在哪里</h3><p><code>item_end</code> 固定为 city/score 之后的裸 closing parenthesis <code>)</code>；带 Markdown 尾缀的 <code>)*</code>/<code>)**</code> 或句号/引号 endpoint 也被排除。它与显式 <code>k</code> 之间隔着完整 city/score，因此不会像旧 <code>(Count: k)</code> 的 closing parenthesis 那样形成紧邻数字的浅层 shortcut。</p></div><div><h3>为什么选 L{int(float(gemma_native['layer']))}</h3><p>每层仅用 20 个 discovery seeds 的 grouped OOF Logistic/NCC 选取；最高 selection score 的层为 <code>{esc(gemma_tie_text)}</code>，固定 tie-break 后默认 L{int(float(gemma_native['layer']))}。10 个 confirmation seeds 完全不参与选层或 PCA 拟合。</p></div><div><h3>筛选与 split</h3><p>共生成 <code>{gemma_generated_candidates}</code> 个 controlled candidates，严格保留 30 个完整 N=10、同一 nested-bullet shell 与同一裸 <code>)</code> endpoint trajectories；再用独立固定 salt 哈希分为 20 discovery / 10 confirmation。筛选只读取文本/parser，不查看 hidden states、PCA 或 probe 指标。</p></div></div>"""
        gemma_aria = "Gemma controlled Record-prefix"
    else:
        gemma_title = "Gemma4-E4B · N=10 · single episode <code>(Count: k)</code>"
        gemma_result_text = (
            "要求整条 reasoning 只有一个十项 counting episode，且每项均以 "
            "<code>(Count: k)</code> 结束。主图固定读取右括号上的 "
            "<code>item_end</code>；此时该项 targeted city/score evidence 与显式 "
            "<code>Count: k</code> 都已经在自回归上下文中。"
        )
        gemma_definitions_html = f"""
<div class="definitions"><div><h3>Gemma grammar</h3><p>Parser 名称为 <code>adjacent_rank_after_city / inline_count / count_colon</code>。可读 surface 例如 <code>… Ljubljana received a score of 64.\" (Count: 1)</code>；十项必须依次写成 <code>Count: 1…10</code>，整条 reasoning 不得出现第二个 counting episode。</p></div><div><h3>Hidden state 在哪里</h3><p><code>item_end</code> 是 closing parenthesis <code>)</code> 的 post-block state：Lℓ 读取 <code>h^(ℓ)[prompt_tokens + t_)]</code>。它位于该项 retrieval 文本与 count marker 之后，并能看到当前 <code>k</code>；L1 是第一个 decoder block 的输出，不是“retrieval 之前”的时间点。</p></div><div><h3>为什么默认是 L1</h3><p>Discovery 上 <code>{esc(gemma_tie_text)}</code> 的 selection score 并列最高；固定 tie-break 取最浅层，所以默认显示 L1。这里数字 <code>k</code> 紧邻 endpoint，浅层即可读取显式 surface cue；L1 不能解释为模型在第一层完成了 counting 或 consolidation。</p></div><div><h3>筛选与 split</h3><p>两个独立 raw seed pools 的 retained/raw 分别为 <code>{esc(filter_text)}</code>。30 个文本合格 seeds 使用独立固定 salt 做 20 discovery / 10 confirmation；hidden states 不参与 seed 筛选，层和 PCA 只由 discovery 拟合。</p></div></div>"""
        gemma_aria = "Gemma strict count-colon"
    gemma_html = f"""
<article class="appendix-model"><h3>{gemma_title}</h3>
<div class="callout"><strong>结果：</strong>{gemma_result_text} secondary D/C=20/10；冻结 confirmation 的 Log/NCC gap 为 {gemma_log_gap:+.1f}/{gemma_ncc_gap:+.1f} pp。</div>
{gemma_definitions_html}
{table(('mode','endpoint','default layer','D / C traces','C states','C Log / NCC','C SNR','C PCA3 silhouette','C PCA3 ordinal RSA'), gemma_rows)}
{prem_control_html}
<div class="callout"><strong>Discovery seeds：</strong><code class="seed-list">{esc(', '.join(map(str, gemma_discovery_seeds)))}</code><br><strong>Confirmation seeds：</strong><code class="seed-list">{esc(', '.join(map(str, gemma_confirmation_seeds)))}</code></div>
<div class="controls"><label>Rows<select id="indexed-numeric-n10-gemma-cohort"><option value="confirmation">Confirmation only</option><option value="all">Discovery + confirmation</option></select></label></div>
<div class="dual-grid">
<figure class="geometry-card"><h3>Paired non-thinking</h3><label class="layer-control">Layer<select id="indexed-numeric-n10-gemma-non-layer">{gemma_layer_selects['non']}</select></label><canvas id="indexed-numeric-n10-gemma-non" role="img" aria-label="{gemma_aria} paired non-thinking PCA3"></canvas><p class="rotate-hint">drag to rotate · scatter plus count-centroid path; no per-seed lines</p><p class="panel-stats" id="indexed-numeric-n10-gemma-non-stats"></p></figure>
<figure class="geometry-card"><h3>Strict main-cluster native-thinking</h3><label class="layer-control">Layer<select id="indexed-numeric-n10-gemma-native-layer">{gemma_layer_selects['native']}</select></label><canvas id="indexed-numeric-n10-gemma-native" role="img" aria-label="{gemma_aria} native-thinking PCA3"></canvas><p class="rotate-hint">drag to rotate · independent layer and discovery-fitted PCA basis</p><p class="panel-stats" id="indexed-numeric-n10-gemma-native-stats"></p></figure>
</div></article>"""
    html = html.replace(
        '<div class="callout warning"><strong>共同解释边界：</strong>',
        gemma_html + '\n<div class="callout warning"><strong>共同解释边界：</strong>',
        1,
    )
    payload["models"].update(gemma_payload["models"])
    payload["gemma_surface_family"] = gemma_family
    inputs.extend(
        [
            gemma_metrics_path,
            gemma_selected_path,
            gemma_candidates_path,
            gemma_payload_path,
            gemma_audit_path,
        ]
    )
    inputs.extend(prem_inputs)
    return html, inputs, payload


def _domain_dimension_svg(model: str, payload: Mapping[str, Any]) -> str:
    """Render PCA-dimension sensitivity for the frozen answer layer."""

    modes = ("non_thinking", "native_thinking")
    dimensions = [
        int(row["dimensions"])
        for row in payload[modes[0]]["metrics"]["dimension_sweep"]
    ]
    if dimensions != [1, 2, 4, 8, 16, 32]:
        raise ValueError(f"Unexpected domain-transfer PCA dimensions for {model}")
    width, height = 760, 330
    panels = (
        (
            "City confirmation",
            "city_logistic_balanced_accuracy",
            52.0,
            350.0,
        ),
        (
            "City → flower/animal",
            "cross_domain_logistic_balanced_accuracy",
            422.0,
            720.0,
        ),
    )
    y_top, y_bottom = 54.0, 268.0

    def sx(dimension: int, left: float, right: float) -> float:
        return left + math.log2(dimension) / 5.0 * (right - left)

    def sy(value: float) -> float:
        return y_bottom - float(value) * (y_bottom - y_top)

    elements = []
    for title, metric, left, right in panels:
        elements.append(
            f'<text class="metric-label" x="{(left+right)/2:.1f}" y="27" '
            f'text-anchor="middle">{esc(title)}</text>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = sy(tick)
            elements.append(
                f'<line class="metric-gridline" x1="{left:.1f}" y1="{y:.1f}" '
                f'x2="{right:.1f}" y2="{y:.1f}"/>'
            )
            if left == panels[0][2]:
                elements.append(
                    f'<text class="metric-tick" x="{left-8:.1f}" y="{y+4:.1f}" '
                    f'text-anchor="end">{100*tick:.0f}%</text>'
                )
        chance_y = sy(0.1)
        elements.append(
            f'<line class="domain-chance" x1="{left:.1f}" y1="{chance_y:.1f}" '
            f'x2="{right:.1f}" y2="{chance_y:.1f}"/>'
        )
        for dimension in dimensions:
            x = sx(dimension, left, right)
            elements.append(
                f'<text class="metric-tick" x="{x:.1f}" y="288" '
                f'text-anchor="middle">{dimension}</text>'
            )
        for mode, css_class in (
            ("non_thinking", "domain-line-non"),
            ("native_thinking", "domain-line-native"),
        ):
            rows = payload[mode]["metrics"]["dimension_sweep"]
            points = [
                (sx(int(row["dimensions"]), left, right), sy(float(row[metric])))
                for row in rows
            ]
            elements.append(
                f'<polyline class="domain-dim-line {css_class}" style="fill:none" points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                + '"/>'
            )
            for x, y in points:
                shape = (
                    f'<circle class="domain-dim-mark {css_class}" cx="{x:.1f}" '
                    f'cy="{y:.1f}" r="4.5"/>'
                    if mode == "non_thinking"
                    else f'<rect class="domain-dim-mark {css_class}" x="{x-4.5:.1f}" '
                    f'y="{y-4.5:.1f}" width="9" height="9"/>'
                )
                elements.append(shape)
    elements.append(
        '<text class="metric-axis-title" x="380" y="319" '
        'text-anchor="middle">PCA dimensions (fit on layer-selection seeds only)</text>'
    )
    return (
        f'<figure class="metric-figure domain-dim-figure"><h3>{esc(model)} · '
        f'Logistic dimension sweep</h3><svg viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{esc(model)} held-out count accuracy across PCA dimensions">'
        f'<title>{esc(model)} PCA dimension sensitivity</title><desc>Non-thinking '
        f'is the dark circle line and native-thinking is the teal square line. The '
        f'dashed reference is ten-class chance.</desc>{"".join(elements)}</svg></figure>'
    )


def _domain_verdict(model_payload: Mapping[str, Any]) -> str:
    non = model_payload["non_thinking"]["metrics"]
    native = model_payload["native_thinking"]["metrics"]
    transfer_fields = (
        "logistic_balanced_accuracy",
        "ncc_balanced_accuracy",
    )
    transfer_wins = [
        float(native["cross_domain_count_mean"][field])
        > float(non["cross_domain_count_mean"][field])
        for field in transfer_fields
    ]
    leakage_wins = [
        float(native["count_residual_domain_leakage"][field])
        < float(non["count_residual_domain_leakage"][field])
        for field in transfer_fields
    ]
    if all(transfer_wins) and all(leakage_wins):
        return (
            "两个 count probe 的跨实体迁移都更高，同时两个 residual-domain "
            "probe 都更低；这一组 exploratory 指标与 native-thinking 在 answer "
            "endpoint 更稳健地保留 count、并保留较少实体类别 nuisance 的解释一致。"
        )
    if any(transfer_wins) and any(leakage_wins):
        return (
            "证据是混合的：至少一个跨实体 count probe 与一个 residual-domain "
            "probe 有利于 native-thinking，但没有在两种 probe 上同时复现。"
        )
    return (
        "当前指标不支持“native-thinking 更过滤实体类别 nuisance”的简单结论；"
        "3-D 视觉分离不能替代 held-out transfer 与 residual-domain probe。"
    )


def _domain_metric_definitions() -> str:
    """Give one plain-language guide, with reproducibility details collapsed."""

    rows = (
        (
            "独立最佳层",
            "这个 model × mode 在哪一层最容易读出 count？",
            "只用 city discovery 200 按 seed 交叉验证；各模式各选各的层。",
            "仅确定读取位置",
        ),
        (
            "三域 pooled count BAcc",
            "在 city 上学到的 count 表征，放到三种实体上一共还能读多准？",
            "冻结后一起测试 city 100 + flower 100 + animal 100。",
            "越高越好；chance 10%",
        ),
        (
            "city → flower/animal count BAcc",
            "不看熟悉的 city，只换成新实体后还能读多准？",
            "分别测试 flower 100、animal 100，再平均两者。",
            "越高越好；chance 10%",
        ),
        (
            "count-residual domain BAcc",
            "先减掉每个 count 的平均表征后，还能不能猜出实体是 city、flower 还是 animal？",
            "在 confirmation 300 上按 seed 留一验证，预测三种实体域。",
            "越低越好；chance 33.3%",
        ),
    )
    return f"""
<h3>如何读下面两张表</h3>
<div class="callout"><strong>统一流程：</strong>只用原始 <code>city discovery 200</code> 选层并训练 count 读出器 → 全部冻结 → 再看 <code>confirmation</code>。两个 count accuracy 问“count 还读不读得出来”（越高越好）；最后一个问“去掉 count 后实体类型还剩多少”（越低越好）。</div>
{table(('表头', '它实际在问什么', '怎么算', '如何判读'), rows)}
<p class="small"><strong>Log / NCC / BAcc：</strong>统一计算与例子见 <a href="#metric-guide">指标字典</a>；本 appendix 的单元格始终按 <code>Log / NCC</code> 排列。这里的三域类别平衡，所以 BAcc 也等于普通 accuracy。</p>
<details><summary>本 appendix 特有的复现细节</summary><div class="callout"><p>选层时，city discovery 200 按 seed 做 5-fold grouped CV；每层分数是 Log 与 NCC 的跨折均值再取平均，同分取较早层。</p><p>Residual-domain 每折用 9 个 confirmation seeds 估计各 count centroid <code>μ_N</code>，对训练和 held-out seed 都计算 <code>r_i=h_i−μ_{{N_i}}</code>，再用 residual 预测 city/flower/animal，最后平均 10 折。它只移除训练折可估计的加性 count centroid，不保证移除非线性 domain information。</p></div></details>
"""


def _domain_model_block(model: str, payload: Mapping[str, Any]) -> str:
    slug = "qwen" if model.startswith("Qwen") else "gemma"
    metrics_rows = []
    audit_rows = []
    for mode, label in (
        ("non_thinking", "non-thinking"),
        ("native_thinking", "native-thinking"),
    ):
        mode_payload = payload[mode]
        metrics = mode_payload["metrics"]
        audit = mode_payload["audit"]
        overall = metrics["overall_count"]
        transfer = metrics["cross_domain_count_mean"]
        leakage = metrics["count_residual_domain_leakage"]
        metrics_rows.append(
            (
                label,
                f"L{int(mode_payload['selected_layer'])}",
                f"{_pct(overall['logistic_balanced_accuracy'])} / "
                f"{_pct(overall['ncc_balanced_accuracy'])}",
                f"{_pct(transfer['logistic_balanced_accuracy'])} / "
                f"{_pct(transfer['ncc_balanced_accuracy'])}",
                f"{_pct(leakage['logistic_balanced_accuracy'])} / "
                f"{_pct(leakage['ncc_balanced_accuracy'])}",
            )
        )
        exact_denominator = int(audit["transfer_exact_count_denominator"])
        exact_text = (
            f"{int(audit['transfer_exact_count_rows'])}/{exact_denominator}"
            if exact_denominator
            else "n/a"
        )
        audit_rows.append(
            (
                label,
                str(int(audit["transfer_answer_states"])),
                str(int(audit["transfer_running_states"])),
                exact_text,
                str(int(audit.get("transfer_generation_rescue_rows", 0))),
                esc(json.dumps(audit["trace_category_counts"], ensure_ascii=False)),
            )
        )
    return f"""
<article class="appendix-model"><h3>{esc(model)}</h3>
<div class="callout"><strong>判读：</strong>{esc(_domain_verdict(payload))}</div>
<div class="dual-grid domain-grid">
<figure class="geometry-card"><h3>Answer endpoint · non-thinking</h3><div class="controls"><label>Cohort<select id="domain-{slug}-non-cohort"><option value="all">全部 300</option><option value="city">city 100</option><option value="flower">flower 100</option><option value="animal">animal 100</option></select></label><label>Color<select id="domain-{slug}-non-color"><option value="count">gold count</option><option value="domain">entity domain</option></select></label></div><canvas id="domain-{slug}-non" role="img" aria-label="{esc(model)} non-thinking city flower animal answer geometry"></canvas><p class="rotate-hint">drag to rotate · circle city · triangle flower · square animal</p><p class="panel-stats" id="domain-{slug}-non-stats"></p></figure>
<figure class="geometry-card"><h3>Answer endpoint · native-thinking</h3><div class="controls"><label>Cohort<select id="domain-{slug}-native-cohort"><option value="all">全部 300</option><option value="city">city 100</option><option value="flower">flower 100</option><option value="animal">animal 100</option></select></label><label>Color<select id="domain-{slug}-native-color"><option value="count">gold count</option><option value="domain">entity domain</option></select></label></div><canvas id="domain-{slug}-native" role="img" aria-label="{esc(model)} native-thinking city flower animal answer geometry"></canvas><p class="rotate-hint">drag to rotate · each mode uses its own selected layer and PCA basis</p><p class="panel-stats" id="domain-{slug}-native-stats"></p></figure>
</div><div class="domain-legend"><span><i class="domain-city"></i>city</span><span><i class="domain-flower"></i>flower</span><span><i class="domain-animal"></i>animal</span><span>颜色默认表示 N=1…10</span></div>
{table(('mode', '独立最佳层', '三域 pooled count BAcc（PCA16；Log/NCC）', 'city → flower/animal count BAcc（PCA16；Log/NCC）', 'count-residual domain BAcc（PCA16；Log/NCC）↓'), metrics_rows)}
<details><summary>PCA dimension sweep（仅作敏感性诊断）</summary>{_domain_dimension_svg(model, payload)}<p class="small">保留逐维结果供检查，但不再跨维度汇总，也不把它用于上面的判读。</p></details>
<details><summary>Capture 完整性与已保存的 running index</summary><p class="small">Flower 与 animal 每种各 100 trajectories。Answer 图每条只用一个 answer-query state；running states 没有被丢弃：non-thinking 保存 prompt 中每个 active-record span end，native-thinking 保存 parser 实际观察到的每个 item end，允许少数、重复与 ragged path，并保存全部 decoder layers。每个 model × mode 另有顶层 <code>site_index.jsonl</code>：一行对应一个 site，直接记录 shard <code>states.npz</code>、<code>site_states</code> 的 <code>state_axis</code>、token endpoint、边界语义与层注册表，后续无需逐个扫描 shard manifest。若 greedy generation 达到旧 token ceiling，只有 censored row 会在保持 prompt 与 greedy rule 不变时提高 ceiling 重试；数量单列审计。</p>{table(('mode', 'transfer answer states', 'transfer running states', 'final exact / audited', 'ceiling rescues', 'native trace categories'), audit_rows)}</details>
</article>"""


def domain_transfer_appendix(
    analysis_root: Path,
) -> tuple[str, list[Path], dict[str, Any]]:
    payload_path = analysis_root / "report_payload.json"
    manifest_path = analysis_root / "analysis_manifest.json"
    layer_path = analysis_root / "layer_selection_sweep.csv"
    dimension_path = analysis_root / "pca_dimension_sweep.csv"
    capture_audit_path = analysis_root / "capture_audit.json"
    payload = read_json(payload_path)
    if str(payload.get("schema_version")) != "realistic_niah_domain_transfer_geometry_v2_city_discovery":
        raise ValueError(f"Unexpected domain-transfer payload: {payload_path}")
    models = payload.get("models", {})
    if set(models) != set(MODELS):
        raise ValueError(f"Domain-transfer payload has models {sorted(models)}")
    capture_audit = read_json(capture_audit_path)
    if str(capture_audit.get("schema_version")) != (
        "realistic_niah_domain_transfer_capture_audit_v1"
    ):
        raise ValueError(f"Unexpected domain-transfer capture audit: {capture_audit_path}")
    strict_audits = {
        (str(row["model_label"]), str(row["mode"])): row
        for row in capture_audit.get("audits", ())
    }
    expected_audits = {
        (model, mode)
        for model in MODELS
        for mode in ("non_thinking", "native_thinking")
    }
    if set(strict_audits) != expected_audits:
        raise ValueError("Domain-transfer capture audit is not the complete 2 x 2 grid")
    for model in MODELS:
        model_payload = models[model]
        if set(model_payload) != {"non_thinking", "native_thinking"}:
            raise ValueError(f"Domain-transfer modes are incomplete for {model}")
        for mode in ("non_thinking", "native_thinking"):
            value = model_payload[mode]
            strict = strict_audits[(model, mode)]
            if not (
                bool(strict.get("registered_panel_complete"))
                and bool(strict.get("all_states_finite"))
                and bool(strict.get("site_index_audited"))
                and int(strict.get("answer_states", -1))
                == int(value["audit"]["transfer_answer_states"])
                and int(strict.get("running_states", -1))
                == int(value["audit"]["transfer_running_states"])
            ):
                raise ValueError(f"Strict capture audit disagrees for {model}/{mode}")
    return f"""
<section id="appendix-domain-transfer"><h2>Appendix B · Entity-domain transfer 实验设计</h2>
<div class="definitions"><div><h3>配对刺激</h3><p>同一套 V4.4 confirmation 10 counts × 10 seeds 被逐 cell 改写为 city、flower、animal；haystack、active-record score、N 与 seed 保持不变，只改实体词表及对应 prompt。每个 model × mode 共 300 answer trajectories；按要求保留错误输出，count 标签始终是 gold N。</p></div><div><h3>比较位置</h3><p>Non-thinking 取回答前 prompt-final colon；native-thinking 取完整 reasoning 后、数字答案前的 <code>answer_query_v3</code>。两个 mode 独立选层，比较的是各自最清楚的 answer endpoint，而不是同层绝对距离。</p></div><div><h3>3-D 图的基准</h3><p>每个 panel 的 StandardScaler/PCA3 只在该 mode 的 city discovery 200 上拟合，再原样 transform 三个 domain 的 confirmation 300。形状表示 domain、颜色表示 count；左右坐标轴不可直接比较绝对距离。</p></div></div>
<div class="callout"><strong>当前只保留设计与 B.2：</strong>原先的 answer-only transfer 大表、PCA 图和 dimension sweep 暂时移除，避免与 B.2 的 running→answer 对比重复。下节 B.2 是当前唯一保留的结果展示。</div>
<details><summary>Flower / animal prompt 如何分别改写</summary><div class="definitions two"><div><h3>Non-thinking</h3><p><code>How many flower-score audit records are in the passage?</code>（animal 版本只把 flower 改为 animal。）随后仍是 <code>Do not explain, reason aloud, quote, or list any records.</code>，并强制整段 response 为 <code>Total:&lt;integer&gt;</code>。保存 prompt-final colon 的 hidden state。</p></div><div><h3>Native-thinking</h3><p>定义句分别写成 <code>A flower-score audit record names one flower…</code> 与 <code>An animal-score audit record names one animal…</code>；问题后要求 concise reasoning、不要 repeating/restarting，并以 <code>Total: &lt;integer&gt;</code> 结束。保存 thinking item ends 与最终数字前的 <code>answer_query_v3</code>。</p></div></div></details>
<p class="small"><strong>冻结规则：</strong>city discovery 负责选层、拟合 StandardScaler/PCA/probe；flower/animal 只做 frozen confirmation transform/evaluation。模型答错的 trajectories 仍保留，running path 仍按 parser-observed 1…M，不补齐到 gold N。</p>
</section>""", [payload_path, manifest_path, layer_path, dimension_path, capture_audit_path], {}


def domain_endpoint_comparison_appendix(
    analysis_root: Path,
) -> tuple[str, list[Path], dict[str, Any]]:
    """Render running-index versus answer-token domain geometry within each mode."""

    paths = [
        analysis_root / "domain_endpoint_metrics.csv",
        analysis_root / "geometry_payload.json",
        analysis_root / "audit.json",
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    payload = read_json(paths[1])
    if str(payload.get("schema_version")) != (
        "realistic_niah_domain_endpoint_comparison_v1"
    ):
        raise ValueError(f"Unexpected domain endpoint payload: {paths[1]}")
    models = payload.get("models", {})
    if set(models) != set(MODELS):
        raise ValueError(f"Domain endpoint payload has models {sorted(models)}")

    table_rows = []
    verdicts = []
    blocks = []
    for model in MODELS:
        model_payload = models[model]
        if set(model_payload) != {"non_thinking", "native_thinking"}:
            raise ValueError(f"Incomplete endpoint comparison for {model}")
        slug = "qwen" if model.startswith("Qwen") else "gemma"
        mode_blocks = []
        for mode, mode_short, mode_label in (
            ("non_thinking", "non", "non-thinking"),
            ("native_thinking", "native", "native-thinking"),
        ):
            endpoints = model_payload[mode]
            if set(endpoints) != {"running_index", "answer_token"}:
                raise ValueError(f"Incomplete endpoints for {model}/{mode}")
            transfer_means: dict[str, float] = {}
            for endpoint, endpoint_label in (
                ("running_index", "running index"),
                ("answer_token", "answer token"),
            ):
                value = endpoints[endpoint]
                metrics = value["metrics"]
                transfer_means[endpoint] = sum(
                    (
                        float(metrics[domain]["logistic_balanced_accuracy"])
                        + float(metrics[domain]["ncc_balanced_accuracy"])
                    )
                    / 2
                    for domain in ("flower", "animal")
                ) / 2
                table_rows.append(
                    (
                        esc(model),
                        esc(mode_label),
                        esc(endpoint_label),
                        f"L{int(value['layer'])}",
                        f"<code>{esc(value['city_site'])}</code>",
                        f"{_pct(metrics['city']['logistic_balanced_accuracy'])} / {_pct(metrics['city']['ncc_balanced_accuracy'])}",
                        f"{_pct(metrics['flower']['logistic_balanced_accuracy'])} / {_pct(metrics['flower']['ncc_balanced_accuracy'])}",
                        f"{_pct(metrics['animal']['logistic_balanced_accuracy'])} / {_pct(metrics['animal']['ncc_balanced_accuracy'])}",
                    )
                )
            delta = 100 * (
                transfer_means["answer_token"] - transfer_means["running_index"]
            )
            direction = "高" if delta > 0 else "低"
            verdicts.append(
                f"<li><strong>{esc(model)} · {esc(mode_label)}：</strong>answer 的 flower/animal "
                f"平均 Log/NCC 比 running {direction} {abs(delta):.1f} 个百分点。</li>"
            )
            base = f"endpoint-domain-{slug}-{mode_short}"
            mode_blocks.append(
                f"""
<h4>{esc(mode_label)}</h4>
<div class="dual-grid">
<figure class="geometry-card"><h3>Running index · occurrence k</h3><div class="controls"><label>Domain<select id="{base}-running-domain"><option value="all">city + flower + animal</option><option value="city">city</option><option value="flower">flower</option><option value="animal">animal</option></select></label><label>Color<select id="{base}-running-color"><option value="count">running k</option><option value="domain">entity domain</option></select></label></div><canvas id="{base}-running" role="img" aria-label="{esc(model)} {esc(mode_label)} running index domain geometry"></canvas><p class="rotate-hint">drag to rotate · domain-specific centroid paths · confirmation only</p><p class="panel-stats" id="{base}-running-stats"></p></figure>
<figure class="geometry-card"><h3>Answer token · gold N</h3><div class="controls"><label>Domain<select id="{base}-answer-domain"><option value="all">city + flower + animal</option><option value="city">city</option><option value="flower">flower</option><option value="animal">animal</option></select></label><label>Color<select id="{base}-answer-color"><option value="count">gold N</option><option value="domain">entity domain</option></select></label></div><canvas id="{base}-answer" role="img" aria-label="{esc(model)} {esc(mode_label)} answer token domain geometry"></canvas><p class="rotate-hint">drag to rotate · independent endpoint layer/PCA · confirmation only</p><p class="panel-stats" id="{base}-answer-stats"></p></figure>
</div>"""
            )
        blocks.append(
            f"<article class=\"appendix-model\"><h3>{esc(model)}</h3>"
            + "".join(mode_blocks)
            + "</article>"
        )

    html = f"""
<section id="appendix-domain-endpoints"><h2>Appendix B.2 · 三个 entity domain 内：running index → answer token</h2>
<div class="callout"><strong>核心比较：</strong>在 non-thinking 内部比较 prompt needle-end 的 running k 与回答前 answer state，在 native-thinking 内部比较 thinking item-end 的 running k 与数字答案前 state。每个 model × mode × endpoint 使用自己的 discovery-selected layer，也各自在 city discovery 上拟合 StandardScaler/PCA；因此左右比较的是 count ordering、domain separation 与 frozen probe，而不是同一坐标系中的点距。</div>
<div class="definitions two"><div><h3>Running index</h3><p>每个 parser 实际观察到的 occurrence k 是一个 state；低 k 样本更多，native trace 允许 ragged，绝不按 gold N 补齐。图中 centroid path 按 domain 分开计算，避免把 city/flower/animal 三个 cloud 的均值画在任一 cloud 之外。</p></div><div><h3>Answer token</h3><p>每条 trajectory 只贡献一个 state，标签是 gold final N；模型答错的样本仍保留。Flower/animal 只用于 frozen confirmation transfer，既不选层也不拟合 PCA/probe。</p></div></div>
<div class="domain-legend"><span><i class="domain-city"></i>city</span><span><i class="domain-flower"></i>flower</span><span><i class="domain-animal"></i>animal</span><span>默认颜色表示 k/N=1…10；线与大点始终是各 domain 内 centroid</span></div>
<ul>{''.join(verdicts)}</ul>
{table(('model','mode','endpoint','layer','city site','city Log/NCC','flower Log/NCC','animal Log/NCC'), table_rows)}
{''.join(blocks)}
<div class="callout warning"><strong>不能直接叫作 consolidation 的充分证据：</strong>running 与 answer 的 observation unit 和标签语义不同（多个 occurrence k 对一个 gold N），因此 answer 更高只能作为“最终 count 更可读/更跨域稳定”的证据。要把差值解释为 retrieval 后过滤 prompt semantics，还需要 native-thinking 的同一轨迹、同一目标变量的 running→answer longitudinal capture。</div>
</section>"""
    return html, paths, models


def fisher_lda_section(
    root: Path,
) -> tuple[str, list[Path], dict[str, Any]]:
    """Load and render the frozen discovery-fitted Fisher/LDA3 diagnostic."""

    paths = [root / "geometry_payload.json", root / "summary.csv", root / "audit.json"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    payload = read_json(paths[0])
    audit = read_json(paths[2])
    expected_schema = "niah_discovery_fitted_fisher_lda3_v1"
    if str(payload.get("schema_version")) != expected_schema:
        raise ValueError(f"Unexpected Fisher/LDA payload schema: {paths[0]}")
    if str(audit.get("schema_version")) != expected_schema:
        raise ValueError(f"Unexpected Fisher/LDA audit schema: {paths[2]}")
    models = payload.get("models", {})
    if set(models) != set(MODELS):
        raise ValueError(f"Fisher/LDA payload has models {sorted(models)}")

    rows = []
    verdicts = []
    blocks = []
    for model in MODELS:
        model_payload = models[model]
        if set(model_payload) != {"running_index", "final_count"}:
            raise ValueError(f"Incomplete Fisher/LDA endpoints for {model}")
        slug = "qwen" if model.startswith("Qwen") else "gemma"
        for endpoint, endpoint_label in (
            ("running_index", "running index"),
            ("final_count", "final count"),
        ):
            endpoint_payload = model_payload[endpoint]
            if set(endpoint_payload) != {"non_thinking", "native_thinking"}:
                raise ValueError(f"Incomplete Fisher/LDA modes for {model}/{endpoint}")
            for mode, mode_label in (
                ("non_thinking", "non-thinking"),
                ("native_thinking", "native-thinking"),
            ):
                value = endpoint_payload[mode]
                metrics = value["metrics"]
                held_out = value["held_out"]
                rows.append(
                    (
                        esc(model),
                        esc(endpoint_label),
                        esc(mode_label),
                        f"<code>{esc(value['token_site'])}</code> @ L{int(value['selected_layer'])}",
                        f"{_pct(held_out['logistic_balanced_accuracy'])} / "
                        f"{_pct(held_out['ncc_balanced_accuracy'])}",
                        f"{100*float(value['fit']['top3_fisher_trace_fraction']):.1f}%",
                        f"{float(metrics['discovery_lda3_class_balanced_silhouette']):+.3f}",
                        f"{float(metrics['confirmation_lda3_class_balanced_silhouette']):+.3f}",
                        f"{float(metrics['confirmation_lda3_radius_gap_ratio']):.3f}",
                    )
                )
            non = endpoint_payload["non_thinking"]
            native = endpoint_payload["native_thinking"]
            non_sil = float(
                non["metrics"]["confirmation_lda3_class_balanced_silhouette"]
            )
            native_sil = float(
                native["metrics"]["confirmation_lda3_class_balanced_silhouette"]
            )
            non_ratio = float(
                non["metrics"]["confirmation_lda3_radius_gap_ratio"]
            )
            native_ratio = float(
                native["metrics"]["confirmation_lda3_radius_gap_ratio"]
            )
            if native_sil > non_sil and native_ratio < non_ratio:
                geometry_claim = (
                    "两个紧致度指标同向：native 的 silhouette 更高且 radius/gap 更低，"
                    "支持 held-out 判别空间中相对更紧。"
                )
            elif native_sil <= non_sil and native_ratio >= non_ratio:
                geometry_claim = (
                    "两个紧致度指标都不支持 native 更紧；分类提升应写成更可解码，"
                    "不能改写成 cluster contraction。"
                )
            else:
                geometry_claim = (
                    "silhouette 与 radius/gap 分歧，只支持更可解码，不支持稳健的"
                    " universal tightening。"
                )
            verdicts.append(
                f"<li><strong>{esc(model)} · {esc(endpoint_label)}：</strong>"
                f"confirmation silhouette {non_sil:+.3f} → {native_sil:+.3f}；"
                f"radius/gap {non_ratio:.3f} → {native_ratio:.3f}。{geometry_claim}</li>"
            )

        blocks.append(
            f"""
<article class="appendix-model"><div class="section-title"><h3>{esc(model)}</h3>
<div class="controls"><label>Endpoint<select id="fisher-{slug}-endpoint"><option value="running_index">running index</option><option value="final_count">final count</option></select></label>
<label>Rows<select id="fisher-{slug}-cohort"><option value="confirmation">confirmation only</option><option value="all">discovery + confirmation</option></select></label></div></div>
<div class="dual-grid">
<figure class="geometry-card"><h3>Non-thinking</h3><canvas id="fisher-{slug}-non" role="img" aria-label="{esc(model)} non-thinking discovery-fitted Fisher LDA geometry"></canvas><p class="rotate-hint">drag to rotate · F1/F2/F3 fitted on discovery · confirmation default</p><p class="panel-stats" id="fisher-{slug}-non-stats"></p></figure>
<figure class="geometry-card"><h3>Native-thinking</h3><canvas id="fisher-{slug}-native" role="img" aria-label="{esc(model)} native-thinking discovery-fitted Fisher LDA geometry"></canvas><p class="rotate-hint">drag to rotate · independent layer and Fisher basis per mode</p><p class="panel-stats" id="fisher-{slug}-native-stats"></p></figure>
</div></article>"""
        )

    html = f"""
<section id="fisher-lda"><h2>Discovery-fitted Fisher/LDA3：让图与判别问题对齐</h2>
<div class="callout"><strong>这是一张 supervised diagnostic，不是新的无监督主图。</strong>每个 model × mode × endpoint 先沿用 grouped discovery CV 选择的 layer；随后只用 discovery 拟合 StandardScaler、未 whiten 的 PCA16、class-balanced within-count covariance whitening，以及前三个 between-count eigenvectors。整个 map 冻结后才投影 confirmation。默认只显示 confirmation；切到全量会把用于拟合的 discovery 点也画出，因此视觉上必然更乐观。</div>
<div class="definitions two"><div><h3>图中的两条 centroid path</h3><p>虚线与空心大点是 frozen discovery centroids；实线与实心大点是 confirmation centroids。二者贴近表示判别轴跨 seed split 稳定。彩色小点默认全是 held-out states。</p></div><div><h3>怎样读“更紧”</h3><p>不能只看 Log/NCC。这里同时报告 confirmation class-balanced silhouette（越高越好）与 radius/gap（点到 discovery 自类中心的半径 ÷ 最近相邻中心间隔，越低越好）。两项同向才写相对更紧。</p></div></div>
<div class="metric-legend"><span><i style="background:#FFFDF8;border:2px dashed #626A74"></i>discovery centroid path</span><span><i style="background:#00A88F;border:1px solid #20242D"></i>confirmation centroid path</span><span>颜色 = k/N=1…10</span></div>
{table(('model','endpoint','mode','site / selected layer','held-out Log/NCC','top-3 Fisher trace','D LDA3 silhouette','C LDA3 silhouette','C radius/gap ↓'), rows)}
<ul>{''.join(verdicts)}</ul>
{''.join(blocks)}
<div class="callout warning"><strong>比较边界：</strong>左右使用各自 discovery-selected layer 与各自 Fisher basis，所以可比较 held-out silhouette、radius/gap 和 centroid 稳定性，不能比较坐标的绝对长度。LDA3 主动最大化 discovery 类间分离；若 confirmation 没有复现，就不能用漂亮的 discovery cloud 支持结论。原始 discovery-fitted PCA3 仍保留在上面的主结果中。</div>
</section>"""
    return html, paths, payload


def _fisher_lda_script(fisher_visual: Mapping[str, Any]) -> str:
    payload = json.dumps(
        fisher_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const FISHER=__FISHER__;
const FISHER_VIEWS={};
function fisherIds(model,mode){const s=slug(model),short=mode==='non_thinking'?'non':'native',base='fisher-'+s+'-'+short;return {endpoint:document.getElementById('fisher-'+s+'-endpoint'),cohort:document.getElementById('fisher-'+s+'-cohort'),canvas:document.getElementById(base),stats:document.getElementById(base+'-stats')}}
function fisherPath(c,points,sx,sy,view,{dashed=false,width=2,alpha=1}={}){const rotated=points.map(p=>({p,r:rotate3(p[1],p[2],p[3],view)}));c.save();c.globalAlpha=alpha;c.strokeStyle='#4B5563';c.lineWidth=width;c.setLineDash(dashed?[6,5]:[]);c.beginPath();rotated.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();c.restore();return rotated}
function drawFisher3D(model,mode){
  const ids=fisherIds(model,mode);if(!ids.canvas)return;const endpoint=ids.endpoint.value,cohort=ids.cohort.value,payload=FISHER.models[model][endpoint][mode],points=payload.points.filter(p=>cohort==='all'||p[0]==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No states in this cohort.',20,30);return}
  const view=FISHER_VIEWS[canvas.id]||(FISHER_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),disc=payload.discovery_centroids,conf=payload.confirmation_centroids,rotated=points.map(p=>({p,r:rotate3(p[3],p[4],p[5],view)})),rdisc=disc.map(p=>rotate3(p[1],p[2],p[3],view)),rconf=conf.map(p=>rotate3(p[1],p[2],p[3],view));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[3]),Math.abs(p[4]),Math.abs(p[5])]),...disc.flatMap(p=>[Math.abs(p[1]),Math.abs(p[2]),Math.abs(p[3])]),...conf.flatMap(p=>[Math.abs(p[1]),Math.abs(p[2]),Math.abs(p[3])]),1e-6),axisLen=maxAbs*.66,labels=payload.axis_labels||['F1','F2','F3'],axes=[[labels[0],'#D14B4B',rotate3(axisLen,0,0,view)],[labels[1],'#008E7B',rotate3(0,axisLen,0,view)],[labels[2],'#6750E8',rotate3(0,0,axisLen,view)]],xy=rotated.map(o=>o.r).concat(rdisc,rconf,axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.12;xmax+=dx*.12;ymin-=dy*.12;ymax+=dy*.12;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  fisherPath(c,disc,sx,sy,view,{dashed:true,width:1.7,alpha:.72});fisherPath(c,conf,sx,sy,view,{width:2.4,alpha:.88});const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);for(const o of rotated){const p=o.p,depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.38+.50*depth;c.fillStyle=COLORS[p[2]-1];c.strokeStyle=p[0]==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=p[0]==='confirmation'?2.0:.8;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.5+1.15*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(let i=0;i<disc.length;i++){const p=disc[i],r=rdisc[i];c.fillStyle='#FFFDF8';c.strokeStyle=COLORS[p[0]-1];c.lineWidth=2;c.setLineDash([3,2]);c.beginPath();c.arc(sx(r[0]),sy(r[1]),5.1,0,Math.PI*2);c.fill();c.stroke()}c.setLineDash([]);for(let i=0;i<conf.length;i++){const p=conf[i],r=rconf[i];c.fillStyle=COLORS[p[0]-1];c.strokeStyle='#20242D';c.lineWidth=1.3;c.beginPath();c.arc(sx(r[0]),sy(r[1]),6.0,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(p[0]),sx(r[0])+7,sy(r[1])-6)}
  const trajectories=new Set(points.map(p=>p[0]+':'+p[1]+':'+p[6])).size,seeds=new Set(points.map(p=>p[0]+':'+p[1])).size,support=conf.map(p=>p[4]),fit=payload.fit,m=payload.metrics,held=payload.held_out;ids.stats.textContent=`${payload.token_site} · L${payload.selected_layer} · ${cohort==='confirmation'?'frozen confirmation':'discovery + confirmation'} · ${trajectories} trajectories / ${points.length} states / ${seeds} seeds · C nₖ ${Math.min(...support)}–${Math.max(...support)} · top-3 Fisher trace ${(100*fit.top3_fisher_trace_fraction).toFixed(1)}% · C Log/NCC ${(100*held.logistic_balanced_accuracy).toFixed(1)}%/${(100*held.ncc_balanced_accuracy).toFixed(1)}% · C silhouette ${m.confirmation_lda3_class_balanced_silhouette.toFixed(3)} · C radius/gap ${m.confirmation_lda3_radius_gap_ratio.toFixed(3)}`;
}
function setupFisher(model){const non=fisherIds(model,'non_thinking'),redraw=()=>{drawFisher3D(model,'non_thinking');drawFisher3D(model,'native_thinking')};non.endpoint.addEventListener('change',redraw);non.cohort.addEventListener('change',redraw);for(const mode of ['non_thinking','native_thinking']){const ids=fisherIds(model,mode);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=FISHER_VIEWS[ids.canvas.id]||(FISHER_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawFisher3D(model,mode)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop)}redraw()}
for(const model of Object.keys(FISHER.models)){try{setupFisher(model)}catch(error){console.error('Fisher/LDA panel failed',model,error)}}
window.addEventListener('resize',()=>{for(const model of Object.keys(FISHER.models))for(const mode of ['non_thinking','native_thinking']){try{drawFisher3D(model,mode)}catch(error){console.error('Fisher/LDA resize failed',model,mode,error)}}});
""".replace("__FISHER__", payload)


def _dual_script(dual_visual: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dual_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const DUAL=__DUAL__;
const COLORS=['#6750E8','#00A9D8','#00A88F','#2DBE77','#A7C957','#D6B52C','#F29E4C','#E76F51','#D94B86','#8E5DB7'];
const VIEWS={};
function slug(model){return model.startsWith('Qwen')?'qwen':'gemma'}
function rotate3(x,y,z,view){const cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z;return [x1,cp*y-sp*z1,sp*y+cp*z1]}
function dualIds(model,panel){const base='dual-'+slug(model)+'-'+panel;return {base,layer:document.getElementById(base+'-layer'),split:document.getElementById(base+'-split'),canvas:document.getElementById(base)}}
function drawDual3D(model,panel){
  const ids=dualIds(model,panel),payload=DUAL[model].panels[panel],layer=+ids.layer.value,cohort=ids.split.value,block=payload.coordinates[String(layer)],points=block.points.filter(p=>cohort==='all'||p[0]==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height,stat=document.getElementById(ids.base+'-stats');c.clearRect(0,0,w,h);
  if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No states in this cohort.',20,30);return}
  const view=VIEWS[canvas.id]||(VIEWS[canvas.id]={yaw:-.72,pitch:.46}),groups=new Map();for(const p of points){if(!groups.has(p[2]))groups.set(p[2],[]);groups.get(p[2]).push(p)}
  const cent=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ps])=>[k,ps.reduce((s,p)=>s+p[3],0)/ps.length,ps.reduce((s,p)=>s+p[4],0)/ps.length,ps.reduce((s,p)=>s+p[5],0)/ps.length,ps.length]);
  const rotated=points.map(p=>({p,r:rotate3(p[3],p[4],p[5],view)})),rcent=cent.map(p=>({p,r:rotate3(p[1],p[2],p[3],view)})),maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[3]),Math.abs(p[4]),Math.abs(p[5])]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]];
  const xy=rotated.map(o=>o.r).concat(axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:23,r:23,t:18,b:22},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.25;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  c.strokeStyle='#2C3440';c.globalAlpha=.8;c.lineWidth=2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);
  for(const o of rotated){const p=o.p,depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.40+.48*depth;c.fillStyle=COLORS[p[2]-1];c.strokeStyle=p[0]==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=p[0]==='confirmation'?2.1:1;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.6+1.2*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const o of rcent){const p=o.p;c.fillStyle=COLORS[p[0]-1];c.strokeStyle='#20242D';c.lineWidth=1.3;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(p[0]),sx(o.r[0])+7,sy(o.r[1])-6)}
  const trajectories=new Set(points.map(p=>p[0]+':'+p[1]+':'+p[6])).size,seeds=new Set(points.map(p=>p[0]+':'+p[1])).size,support=cent.map(p=>p[4]),metric=payload.metrics[String(layer)],evr=100*block.evr.reduce((a,b)=>a+b,0),source=cohort==='all'?300:100;
  stat.textContent=`${payload.token_site} · L${layer} · source ${source} trajectories across N=1…10 · displayed ${trajectories} trajectories / ${points.length} states / ${seeds} seeds · nₖ ${Math.min(...support)}–${Math.max(...support)} · EVR3 ${evr.toFixed(1)}% · C-held-out Log/NCC ${(100*metric.confirmation_logistic).toFixed(1)}%/${(100*metric.confirmation_ncc).toFixed(1)}% · SNR ${metric.confirmation_snr_db.toFixed(2)} dB`;
}
function setup(model,panel){const ids=dualIds(model,panel);ids.layer.addEventListener('change',()=>drawDual3D(model,panel));ids.split.addEventListener('change',()=>drawDual3D(model,panel));let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=VIEWS[ids.canvas.id]||(VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawDual3D(model,panel)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);drawDual3D(model,panel)}
for(const model of Object.keys(DUAL))for(const panel of Object.keys(DUAL[model].panels)){try{setup(model,panel)}catch(error){console.error('dual panel failed',model,panel,error)}}
const INTERNAL_VIEWS={};
function internalIds(model,endpoint){const s=slug(model),base='internal-'+s+'-'+endpoint;return {base,cohort:document.getElementById('internal-'+s+'-cohort'),layer:document.getElementById(base+'-layer'),canvas:document.getElementById(base),stats:document.getElementById(base+'-stats')}}
function internalLabel(point,endpoint){return endpoint==='running'?point[2]:point[6]}
function drawInternal3D(model,endpoint){
  const ids=internalIds(model,endpoint);if(!ids.canvas)return;const panel=endpoint==='running'?'running_non':'final_non',payload=DUAL[model]?.panels?.[panel],canvas=ids.canvas;if(!payload){const c=canvas.getContext('2d'),rect=canvas.getBoundingClientRect();c.clearRect(0,0,rect.width,rect.height);if(ids.stats)ids.stats.textContent='Panel data unavailable; report build is incomplete.';return}const layer=+ids.layer.value,cohort=ids.cohort.value,block=payload.coordinates[String(layer)];if(!block){if(ids.stats)ids.stats.textContent=`L${layer} coordinates unavailable.`;return}const points=block.points.filter(p=>cohort==='all'||p[0]==='confirmation'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No states in this cohort.',20,30);return}
  const view=INTERNAL_VIEWS[canvas.id]||(INTERNAL_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),groups=new Map();for(const p of points){const label=internalLabel(p,endpoint);if(!groups.has(label))groups.set(label,[]);groups.get(label).push(p)}
  const cent=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([label,ps])=>[label,ps.reduce((s,p)=>s+p[3],0)/ps.length,ps.reduce((s,p)=>s+p[4],0)/ps.length,ps.reduce((s,p)=>s+p[5],0)/ps.length,ps.length]),rotated=points.map(p=>({p,r:rotate3(p[3],p[4],p[5],view)})),rcent=cent.map(p=>({p,r:rotate3(p[1],p[2],p[3],view)}));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[3]),Math.abs(p[4]),Math.abs(p[5])]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]],xy=rotated.map(o=>o.r).concat(rcent.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  c.strokeStyle='#4B5563';c.globalAlpha=.75;c.lineWidth=2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);
  for(const o of rotated){const label=internalLabel(o.p,endpoint),depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.38+.50*depth;c.fillStyle=COLORS[Math.max(0,Math.min(9,label-1))];c.strokeStyle=o.p[0]==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=o.p[0]==='confirmation'?1.9:.8;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.4+1.1*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const o of rcent){const label=o.p[0];c.fillStyle=COLORS[Math.max(0,Math.min(9,label-1))];c.strokeStyle='#20242D';c.lineWidth=1.4;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(label),sx(o.r[0])+7,sy(o.r[1])-6)}
  const trajectories=new Set(points.map(p=>p[0]+':'+p[1]+':'+p[6])).size,seeds=new Set(points.map(p=>p[0]+':'+p[1])).size,support=cent.map(p=>p[4]),evr=100*block.evr.reduce((a,b)=>a+b,0),source=cohort==='all'?300:100,labelName=endpoint==='running'?'running k':'gold N';ids.stats.textContent=`${payload.token_site} · L${layer} · ${cohort==='all'?'full 300-source':'confirmation 100-source'} · ${trajectories}/${source} trajectories · ${points.length} states / ${seeds} seeds · ${labelName} support nₖ ${Math.min(...support)}–${Math.max(...support)} · EVR3 ${evr.toFixed(1)}%`;
}
function setupInternal(model,endpoint){const ids=internalIds(model,endpoint);if(!ids.canvas)return;const redraw=()=>drawInternal3D(model,endpoint);ids.layer.addEventListener('change',redraw);ids.cohort.addEventListener('change',redraw);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=INTERNAL_VIEWS[ids.canvas.id]||(INTERNAL_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;redraw()});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);redraw()}
for(const model of Object.keys(DUAL))for(const endpoint of ['running','answer']){try{setupInternal(model,endpoint)}catch(error){console.error('internal panel failed',model,endpoint,error)}}
let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(()=>{for(const model of Object.keys(DUAL)){for(const panel of Object.keys(DUAL[model].panels)){try{drawDual3D(model,panel)}catch(error){console.error('dual resize failed',model,panel,error)}}for(const endpoint of ['running','answer']){try{drawInternal3D(model,endpoint)}catch(error){console.error('internal resize failed',model,endpoint,error)}}}},100)});
""".replace("__DUAL__", payload)


def _band_script(band_visual: Mapping[str, Any]) -> str:
    payload = json.dumps(
        band_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const BAND=__BAND__;
const BAND_VIEWS={};
const BAND_COLORS={upper:'#00A88F',lower:'#6750E8'};
const MARKER_COLORS={inline_count:'#A7C957',indexed:'#00A88F',ordinal:'#D6B52C',bullet:'#00A9D8',audit_sentence:'#D94B86',completion_recap:'#6750E8',evidence_sequence:'#E76F51',unresolved:'#8A838E'};
const GRAMMAR_COLORS={adjacent_rank_after_city:'#087F72',adjacent_rank_before_city:'#E76F51',same_unit_rank_after_city:'#00A9D8',same_unit_rank_before_city:'#8E5DB7',structural_explicit_rank_before_city:'#D6B52C',structural_invariant_bullet:'#6750E8',structural_unmarked:'#626A74',evidence_sequence_unranked:'#D94B86'};
function bandSlug(model){return model.startsWith('Qwen')?'qwen':'gemma'}
function bandIds(model,space){const s=bandSlug(model);return {cohort:document.getElementById('band-'+s+'-cohort'),color:document.getElementById('band-'+s+'-color'),canvas:document.getElementById('band-'+s+'-'+space),stats:document.getElementById('band-'+s+'-'+space+'-stats'),legend:document.getElementById('band-'+s+'-legend')}}
function bandColor(point,mode){if(mode==='band')return BAND_COLORS[point[5]];if(mode==='occurrence')return COLORS[Math.max(0,Math.min(9,point[3]-1))];if(mode==='grammar')return GRAMMAR_COLORS[point[12]]||'#8A838E';return MARKER_COLORS[point[4]]||'#8A838E'}
function drawBandMark(c,x,y,r,point,mode){c.beginPath();if(mode==='band'&&point[5]==='lower')c.rect(x-r,y-r,2*r,2*r);else c.arc(x,y,r,0,Math.PI*2);c.fill();c.stroke()}
function renderBandLegend(model){const ids=bandIds(model,'raw'),mode=ids.color.value,points=BAND[model].points,values=mode==='band'?['upper','lower']:mode==='occurrence'?[1,2,3,4,5,6,7,8,9,10]:mode==='grammar'?[...new Set(points.map(p=>p[12]))].sort():[...new Set(points.map(p=>p[4]))].sort();ids.legend.replaceChildren();for(const value of values){const span=document.createElement('span'),swatch=document.createElement('i'),label=document.createElement('b'),probe=mode==='band'?['',0,0,0,'',value]:mode==='occurrence'?['',0,0,value,'','upper']:mode==='grammar'?['',0,0,0,'','upper',0,0,0,0,0,0,value]:['',0,0,0,value,'upper'];swatch.style.background=bandColor(probe,mode);if(mode==='band'&&value==='lower')swatch.className='square';label.textContent=mode==='occurrence'?'k='+value:String(value);span.append(swatch,label);ids.legend.append(span)}}
function drawBand3D(model,space){
  const ids=bandIds(model,space),payload=BAND[model],cohort=ids.cohort.value,mode=ids.color.value,start=space==='raw'?6:9,points=payload.points.filter(p=>cohort==='all'||p[0]==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length)return;
  const view=BAND_VIEWS[canvas.id]||(BAND_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),rotated=points.map(p=>({p,r:rotate3(p[start],p[start+1],p[start+2],view)})),maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[start]),Math.abs(p[start+1]),Math.abs(p[start+2])]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]];
  const centerGroups=new Map();for(const p of points){if(!centerGroups.has(p[5]))centerGroups.set(p[5],[]);centerGroups.get(p[5]).push(p)}const centers=[...centerGroups.entries()].map(([name,ps])=>({name,r:rotate3(ps.reduce((s,p)=>s+p[start],0)/ps.length,ps.reduce((s,p)=>s+p[start+1],0)/ps.length,ps.reduce((s,p)=>s+p[start+2],0)/ps.length,view)}));
  const xy=rotated.map(o=>o.r).concat(axes.map(a=>a[2]),centers.map(o=>o.r),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.38+.48*depth;c.fillStyle=bandColor(o.p,mode);c.strokeStyle=o.p[0]==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=o.p[0]==='confirmation'?1.8:.65;drawBandMark(c,sx(o.r[0]),sy(o.r[1]),2.2+1.1*depth,o.p,mode)}
  c.globalAlpha=1;for(const center of centers){const x=sx(center.r[0]),y=sy(center.r[1]);c.fillStyle=BAND_COLORS[center.name];c.strokeStyle='#20242D';c.lineWidth=2;c.beginPath();if(center.name==='lower')c.rect(x-6,y-6,12,12);else c.arc(x,y,6,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='11px Consolas';c.fillText(center.name,x+9,y-7)}
  const trajectories=new Set(points.map(p=>p[0]+':'+p[1]+':'+p[2])).size,upper=points.filter(p=>p[5]==='upper').length,lower=points.length-upper,evr=100*(space==='raw'?payload.raw_evr:payload.centered_evr).reduce((a,b)=>a+b,0);ids.stats.textContent=`${payload.site} · L${payload.layer} · ${cohort==='all'?'full 300-source':'confirmation 100-source'} · ${trajectories} trajectories / ${points.length} states · raw-band upper/lower ${upper}/${lower} · EVR3 ${evr.toFixed(1)}%`;
}
function setupBand(model){const raw=bandIds(model,'raw'),redraw=()=>{drawBand3D(model,'raw');drawBand3D(model,'centered');renderBandLegend(model)};raw.cohort.addEventListener('change',redraw);raw.color.addEventListener('change',redraw);for(const space of ['raw','centered']){const ids=bandIds(model,space);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=BAND_VIEWS[ids.canvas.id]||(BAND_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawBand3D(model,space)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop)}redraw()}
for(const model of Object.keys(BAND)){try{setupBand(model)}catch(error){console.error('band panel failed',model,error)}}
window.addEventListener('resize',()=>{for(const model of Object.keys(BAND)){try{drawBand3D(model,'raw');drawBand3D(model,'centered')}catch(error){console.error('band resize failed',model,error)}}});
""".replace("__BAND__", payload)


def _domain_script(domain_visual: Mapping[str, Any]) -> str:
    if not domain_visual:
        return ""
    payload = json.dumps(
        domain_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const DOMAIN_TRANSFER=__DOMAIN_TRANSFER__;
const DOMAIN_VIEWS={};
const DOMAIN_COLORS={city:'#20242D',flower:'#00A88F',animal:'#E76F51'};
function domainIds(model,mode){const s=model.startsWith('Qwen')?'qwen':'gemma',base='domain-'+s+'-'+mode;return {canvas:document.getElementById(base),cohort:document.getElementById(base+'-cohort'),color:document.getElementById(base+'-color'),stats:document.getElementById(base+'-stats')}}
function domainMark(c,domain,x,y,r){c.beginPath();if(domain==='animal'){c.rect(x-r,y-r,2*r,2*r)}else if(domain==='flower'){c.moveTo(x,y-r*1.18);c.lineTo(x+r*1.08,y+r*.88);c.lineTo(x-r*1.08,y+r*.88);c.closePath()}else{c.arc(x,y,r,0,Math.PI*2)}c.fill();c.stroke()}
function domainPointColor(point,mode){return mode==='domain'?DOMAIN_COLORS[point[0]]:COLORS[Math.max(0,Math.min(9,point[2]-1))]}
function drawDomain3D(model,mode){
  const ids=domainIds(model,mode),payload=DOMAIN_TRANSFER[model][mode],cohort=ids.cohort.value,colorMode=ids.color.value,points=payload.points.filter(p=>cohort==='all'||p[0]===cohort),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length)return;
  const view=DOMAIN_VIEWS[canvas.id]||(DOMAIN_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),rotated=points.map(p=>({p,r:rotate3(p[4],p[5],p[6],view)})),maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[4]),Math.abs(p[5]),Math.abs(p[6])]),1e-6),axisLen=maxAbs*.70,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]];
  const centerGroups=new Map();for(const p of points){if(!centerGroups.has(p[2]))centerGroups.set(p[2],[]);centerGroups.get(p[2]).push(p)}const centers=[...centerGroups.entries()].sort((a,b)=>a[0]-b[0]).map(([count,rows])=>[count,rows.reduce((s,p)=>s+p[4],0)/rows.length,rows.reduce((s,p)=>s+p[5],0)/rows.length,rows.reduce((s,p)=>s+p[6],0)/rows.length]);const rotatedCenters=centers.map(p=>({p,r:rotate3(p[1],p[2],p[3],view)}));
  const xy=rotated.map(o=>o.r).concat(rotatedCenters.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  if(colorMode==='count'){c.strokeStyle='#4B5563';c.globalAlpha=.72;c.lineWidth=2;c.beginPath();rotatedCenters.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke()}
  const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.35+.52*depth;c.fillStyle=domainPointColor(o.p,colorMode);c.strokeStyle='#FFFDF8';c.lineWidth=1.8;domainMark(c,o.p[0],sx(o.r[0]),sy(o.r[1]),2.5+1.1*depth)}
  if(colorMode==='count'){c.globalAlpha=1;for(const o of rotatedCenters){c.fillStyle=COLORS[o.p[0]-1];c.strokeStyle='#20242D';c.lineWidth=1.4;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.5,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(o.p[0]),sx(o.r[0])+7,sy(o.r[1])-6)}}
  c.globalAlpha=1;const evr=100*payload.evr.reduce((a,b)=>a+b,0),metric=payload.metrics.cross_domain_count_mean,leak=payload.metrics.count_residual_domain_leakage;ids.stats.textContent=`answer query · L${payload.layer} · ${points.length} confirmation trajectories · city-discovery PCA3 EVR ${evr.toFixed(1)}% · city→flower/animal Log/NCC ${(100*metric.logistic_balanced_accuracy).toFixed(1)}%/${(100*metric.ncc_balanced_accuracy).toFixed(1)}% · residual-domain Log/NCC ${(100*leak.logistic_balanced_accuracy).toFixed(1)}%/${(100*leak.ncc_balanced_accuracy).toFixed(1)}%`;
}
function setupDomain(model,mode){const ids=domainIds(model,mode),redraw=()=>drawDomain3D(model,mode);ids.cohort.addEventListener('change',redraw);ids.color.addEventListener('change',redraw);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=DOMAIN_VIEWS[ids.canvas.id]||(DOMAIN_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawDomain3D(model,mode)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);redraw()}
for(const model of Object.keys(DOMAIN_TRANSFER))for(const mode of ['non','native']){try{setupDomain(model,mode)}catch(error){console.error('domain panel failed',model,mode,error)}}
window.addEventListener('resize',()=>{for(const model of Object.keys(DOMAIN_TRANSFER))for(const mode of ['non','native']){try{drawDomain3D(model,mode)}catch(error){console.error('domain resize failed',model,mode,error)}}});
""".replace("__DOMAIN_TRANSFER__", payload)


def _domain_endpoint_script(endpoint_visual: Mapping[str, Any]) -> str:
    if not endpoint_visual:
        return ""
    payload = json.dumps(
        endpoint_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const DOMAIN_ENDPOINT=__DOMAIN_ENDPOINT__;
const DOMAIN_ENDPOINT_VIEWS={};
const ENDPOINT_DOMAIN_COLORS={city:'#20242D',flower:'#00A88F',animal:'#E76F51'};
function endpointDomainIds(model,mode,endpoint){const s=slug(model),ms=mode==='non_thinking'?'non':'native',es=endpoint==='running_index'?'running':'answer',base='endpoint-domain-'+s+'-'+ms+'-'+es;return {canvas:document.getElementById(base),domain:document.getElementById(base+'-domain'),color:document.getElementById(base+'-color'),stats:document.getElementById(base+'-stats')}}
function endpointDomainMark(c,domain,x,y,r){c.beginPath();if(domain==='animal'){c.rect(x-r,y-r,2*r,2*r)}else if(domain==='flower'){c.moveTo(x,y-r*1.18);c.lineTo(x+r*1.08,y+r*.88);c.lineTo(x-r*1.08,y+r*.88);c.closePath()}else{c.arc(x,y,r,0,Math.PI*2)}c.fill();c.stroke()}
function endpointDomainColor(point,mode){return mode==='domain'?ENDPOINT_DOMAIN_COLORS[point.domain]:COLORS[Math.max(0,Math.min(9,point.count-1))]}
function drawEndpointDomain3D(model,mode,endpoint){
  const ids=endpointDomainIds(model,mode,endpoint),payload=DOMAIN_ENDPOINT[model][mode][endpoint];if(!ids.canvas||!payload)return;const selected=ids.domain.value,colorMode=ids.color.value,points=payload.points.filter(p=>selected==='all'||p.domain===selected),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No states in this domain.',20,30);ids.stats.textContent='No states.';return}
  const view=DOMAIN_ENDPOINT_VIEWS[canvas.id]||(DOMAIN_ENDPOINT_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),groups=new Map();for(const p of points){const key=p.domain+'|'+p.count;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(p)}
  const centers=[...groups.entries()].map(([key,rows])=>{const [domain,count]=key.split('|');return {domain,count:+count,n:rows.length,x:rows.reduce((s,p)=>s+p.x,0)/rows.length,y:rows.reduce((s,p)=>s+p.y,0)/rows.length,z:rows.reduce((s,p)=>s+p.z,0)/rows.length}}).sort((a,b)=>a.domain.localeCompare(b.domain)||a.count-b.count),rotated=points.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)})),rcent=centers.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)}));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p.x),Math.abs(p.y),Math.abs(p.z)]),1e-6),axisLen=maxAbs*.70,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]],xy=rotated.map(o=>o.r).concat(rcent.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  const domains=[...new Set(centers.map(p=>p.domain))];for(const domain of domains){const path=rcent.filter(o=>o.p.domain===domain).sort((a,b)=>a.p.count-b.p.count);c.strokeStyle=ENDPOINT_DOMAIN_COLORS[domain];c.globalAlpha=.68;c.lineWidth=2;c.beginPath();path.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke()}
  const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.34+.52*depth;c.fillStyle=endpointDomainColor(o.p,colorMode);c.strokeStyle='#FFFDF8';c.lineWidth=1.6;endpointDomainMark(c,o.p.domain,sx(o.r[0]),sy(o.r[1]),2.35+1.0*depth)}
  c.globalAlpha=1;for(const o of rcent){const p=o.p;c.fillStyle=colorMode==='domain'?ENDPOINT_DOMAIN_COLORS[p.domain]:COLORS[p.count-1];c.strokeStyle='#20242D';c.lineWidth=1.4;endpointDomainMark(c,p.domain,sx(o.r[0]),sy(o.r[1]),5.4);if(selected!=='all'){c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(p.count),sx(o.r[0])+7,sy(o.r[1])-6)}}
  const trajectories=new Set(points.map(p=>p.domain+':'+p.trajectory_id)).size,support=[...new Map([...Array(10)].map((_,i)=>[i+1,points.filter(p=>p.count===i+1).length])).values()],evr=100*payload.pca3_explained_variance_ratio.reduce((a,b)=>a+b,0),domainsShown=selected==='all'?['city','flower','animal']:[selected],metricText=domainsShown.map(domain=>{const m=payload.metrics[domain];return `${domain} ${(100*m.logistic_balanced_accuracy).toFixed(1)}/${(100*m.ncc_balanced_accuracy).toFixed(1)}%`}).join(' · ');ids.stats.textContent=`${payload.city_site} · L${payload.layer} · ${points.length} states / ${trajectories} domain-trajectories · nₖ ${Math.min(...support)}–${Math.max(...support)} · city-discovery PCA3 EVR ${evr.toFixed(1)}% · frozen Log/NCC: ${metricText}`;
}
function setupEndpointDomain(model,mode,endpoint){const ids=endpointDomainIds(model,mode,endpoint);if(!ids.canvas)return;const redraw=()=>drawEndpointDomain3D(model,mode,endpoint);ids.domain.addEventListener('change',redraw);ids.color.addEventListener('change',redraw);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=DOMAIN_ENDPOINT_VIEWS[ids.canvas.id]||(DOMAIN_ENDPOINT_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;redraw()});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);redraw()}
for(const model of Object.keys(DOMAIN_ENDPOINT))for(const mode of ['non_thinking','native_thinking'])for(const endpoint of ['running_index','answer_token']){try{setupEndpointDomain(model,mode,endpoint)}catch(error){console.error('domain endpoint panel failed',model,mode,endpoint,error)}}
window.addEventListener('resize',()=>{for(const model of Object.keys(DOMAIN_ENDPOINT))for(const mode of ['non_thinking','native_thinking'])for(const endpoint of ['running_index','answer_token']){try{drawEndpointDomain3D(model,mode,endpoint)}catch(error){console.error('domain endpoint resize failed',model,mode,endpoint,error)}}});
""".replace("__DOMAIN_ENDPOINT__", payload)


def _index_city_script(index_city_visual: Mapping[str, Any]) -> str:
    if not index_city_visual:
        return ""
    payload = json.dumps(
        index_city_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const INDEX_CITY=__INDEX_CITY__;
const INDEX_CITY_VIEWS={};
function indexCityIds(model){const s=slug(model),base='index-city-'+s;return {cohort:document.getElementById(base+'-cohort'),canvas:document.getElementById(base),stats:document.getElementById(base+'-stats')}}
function drawIndexCity3D(model){
  const ids=indexCityIds(model),modelPayload=INDEX_CITY[model],payload=modelPayload?.sites?.city_end;if(!ids.canvas||!payload)return;const cohort=ids.cohort.value,points=payload.points.filter(p=>cohort==='all'||p.split==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No retained states.',20,30);return}
  const view=INDEX_CITY_VIEWS[canvas.id]||(INDEX_CITY_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),groups=new Map();for(const p of points){if(!groups.has(p.occurrence))groups.set(p.occurrence,[]);groups.get(p.occurrence).push(p)}const centers=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,rows])=>({k,n:rows.length,x:rows.reduce((s,p)=>s+p.x,0)/rows.length,y:rows.reduce((s,p)=>s+p.y,0)/rows.length,z:rows.reduce((s,p)=>s+p.z,0)/rows.length})),rotated=points.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)})),rcent=centers.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)}));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p.x),Math.abs(p.y),Math.abs(p.z)]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]],xy=rotated.map(o=>o.r).concat(rcent.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}c.strokeStyle='#4B5563';c.globalAlpha=.78;c.lineWidth=2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();
  const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.38+.50*depth;c.fillStyle=COLORS[o.p.occurrence-1];c.strokeStyle=o.p.split==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=o.p.split==='confirmation'?1.9:.8;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.4+1.1*depth,0,Math.PI*2);c.fill();c.stroke()}c.globalAlpha=1;for(const o of rcent){c.fillStyle=COLORS[o.p.k-1];c.strokeStyle='#20242D';c.lineWidth=1.4;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(o.p.k),sx(o.r[0])+7,sy(o.r[1])-6)}
  const trajectories=new Set(points.map(p=>p.request_id)).size,seeds=new Set(points.map(p=>p.seed)).size,support=centers.map(p=>p.n),evr=100*payload.pca3_explained_variance_ratio.reduce((a,b)=>a+b,0),m=payload.metrics;ids.stats.textContent=`${modelPayload.fixed_grammar} · city_end · L${payload.layer} · ${cohort==='all'?'full retained':'confirmation'} · ${trajectories} trajectories / ${points.length} states / ${seeds} seeds · nₖ ${Math.min(...support)}–${Math.max(...support)} · EVR3 ${evr.toFixed(1)}% · C-held-out Log/NCC ${(100*m.confirmation_logistic).toFixed(1)}%/${(100*m.confirmation_ncc).toFixed(1)}% · SNR ${m.confirmation_snr_db.toFixed(2)} dB`;
}
function setupIndexCity(model){const ids=indexCityIds(model);if(!ids.canvas)return;const redraw=()=>drawIndexCity3D(model);ids.cohort.addEventListener('change',redraw);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=INDEX_CITY_VIEWS[ids.canvas.id]||(INDEX_CITY_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;redraw()});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);redraw()}
for(const model of Object.keys(INDEX_CITY)){try{setupIndexCity(model)}catch(error){console.error('index city panel failed',model,error)}}window.addEventListener('resize',()=>{for(const model of Object.keys(INDEX_CITY)){try{drawIndexCity3D(model)}catch(error){console.error('index city resize failed',model,error)}}});
""".replace("__INDEX_CITY__", payload)


def _phase_grammar_script(phase_visual: Mapping[str, Any]) -> str:
    if not phase_visual:
        return ""
    payload = json.dumps(
        phase_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const PHASE_GRAMMAR=__PHASE_GRAMMAR__;
const PHASE_GRAMMAR_VIEWS={};
function phaseGrammarIds(model,family){const s=slug(model),m=family==='phase'?'phase':'ablation',base=m+'-'+s;return {view:document.getElementById(base+'-view'),cohort:document.getElementById(base+'-cohort'),canvas:document.getElementById(base),stats:document.getElementById(base+'-stats')}}
function drawPhaseGrammar3D(model,family){
  const ids=phaseGrammarIds(model,family);if(!ids.canvas||!ids.view)return;const views=PHASE_GRAMMAR[model][family+'_views'],entry=views[ids.view.value];if(!entry)return;const mode=ids.cohort.value,useMeans=mode.endsWith('_means'),confirmation=mode.startsWith('confirmation'),source=useMeans?entry.pca3.seed_count_means:entry.pca3.raw_points,points=source.filter(p=>!confirmation||p.split==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No states in this view.',20,30);ids.stats.textContent='No states.';return}
  const view=PHASE_GRAMMAR_VIEWS[canvas.id]||(PHASE_GRAMMAR_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),groups=new Map();for(const p of points){if(!groups.has(p.occurrence))groups.set(p.occurrence,[]);groups.get(p.occurrence).push(p)}const centers=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,rows])=>({k,n:rows.length,x:rows.reduce((s,p)=>s+p.x,0)/rows.length,y:rows.reduce((s,p)=>s+p.y,0)/rows.length,z:rows.reduce((s,p)=>s+p.z,0)/rows.length})),rotated=points.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)})),rcent=centers.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)}));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p.x),Math.abs(p.y),Math.abs(p.z)]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]],xy=rotated.map(o=>o.r).concat(rcent.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}c.strokeStyle='#4B5563';c.globalAlpha=.78;c.lineWidth=2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();
  const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.36+.52*depth;c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.occurrence-1))];c.strokeStyle=o.p.split==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=o.p.split==='confirmation'?1.8:.8;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),(useMeans?3.2:2.25)+.9*depth,0,Math.PI*2);c.fill();c.stroke()}c.globalAlpha=1;for(const o of rcent){c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.k-1))];c.strokeStyle='#20242D';c.lineWidth=1.4;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(o.p.k),sx(o.r[0])+7,sy(o.r[1])-6)}
  const m=entry.metrics,evr=100*entry.pca3.evr.reduce((a,b)=>a+b,0),support=centers.map(p=>p.n),log=100*Number(m.confirmation_logistic_balanced_accuracy),ncc=100*Number(m.confirmation_ncc_balanced_accuracy),sil=Number(m.cov_confirmation_mahalanobis_silhouette),ratio=Number(m.confirmation_radius_gap_ratio);ids.stats.textContent=`${entry.label} · ${mode.replaceAll('_',' ')} · ${points.length} ${useMeans?'means':'states'} · nₖ ${Math.min(...support)}–${Math.max(...support)} · EVR3 ${evr.toFixed(1)}% · C Log/NCC ${log.toFixed(1)}/${ncc.toFixed(1)}% · silhouette ${sil.toFixed(3)} · radius/gap ${ratio.toFixed(3)} · ${entry.warning||entry.role||''}`;
}
function setupPhaseGrammar(model,family){const ids=phaseGrammarIds(model,family);if(!ids.canvas||!ids.view)return;const redraw=()=>drawPhaseGrammar3D(model,family);ids.view.addEventListener('change',redraw);ids.cohort.addEventListener('change',redraw);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=PHASE_GRAMMAR_VIEWS[ids.canvas.id]||(PHASE_GRAMMAR_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;redraw()});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);redraw()}
for(const model of Object.keys(PHASE_GRAMMAR)){for(const family of ['phase','ablation']){try{setupPhaseGrammar(model,family)}catch(error){console.error('phase grammar panel failed',model,family,error)}}}window.addEventListener('resize',()=>{for(const model of Object.keys(PHASE_GRAMMAR)){for(const family of ['phase','ablation']){try{drawPhaseGrammar3D(model,family)}catch(error){console.error('phase grammar resize failed',model,family,error)}}}});
""".replace("__PHASE_GRAMMAR__", payload)


def _clean_grammar_script(clean_visual: Mapping[str, Any]) -> str:
    if not clean_visual:
        return ""
    payload = json.dumps(
        clean_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const CLEAN_GRAMMAR=__CLEAN_GRAMMAR__;
const CLEAN_VIEWS={};
function cleanIds(model){const s=slug(model);return {cohort:document.getElementById('clean-'+s+'-cohort'),canvas:document.getElementById('clean-'+s),stats:document.getElementById('clean-'+s+'-stats')}}
function drawCleanGrammar3D(model){
  const payload=CLEAN_GRAMMAR[model],ids=cleanIds(model);if(!payload||!ids.canvas)return;const cohort=ids.cohort.value,points=payload.points.filter(p=>cohort==='all'||p.split==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No retained states in this cohort.',20,30);ids.stats.textContent='No retained states.';return}
  const view=CLEAN_VIEWS[canvas.id]||(CLEAN_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),groups=new Map();for(const p of points){if(!groups.has(p.occurrence))groups.set(p.occurrence,[]);groups.get(p.occurrence).push(p)}
  const cent=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ps])=>({k,n:ps.length,x:ps.reduce((s,p)=>s+p.x,0)/ps.length,y:ps.reduce((s,p)=>s+p.y,0)/ps.length,z:ps.reduce((s,p)=>s+p.z,0)/ps.length})),rotated=points.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)})),rcent=cent.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)}));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p.x),Math.abs(p.y),Math.abs(p.z)]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]],xy=rotated.map(o=>o.r).concat(rcent.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  c.strokeStyle='#4B5563';c.globalAlpha=.78;c.lineWidth=2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);
  for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.38+.50*depth;c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.occurrence-1))];c.strokeStyle=o.p.split==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=o.p.split==='confirmation'?1.9:.8;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.4+1.1*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const o of rcent){c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.k-1))];c.strokeStyle='#20242D';c.lineWidth=1.4;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(o.p.k),sx(o.r[0])+7,sy(o.r[1])-6)}
  const trajectories=new Set(points.map(p=>p.request_id)).size,seeds=new Set(points.map(p=>p.seed)).size,support=cent.map(p=>p.n),evr=100*payload.pca3_explained_variance_ratio.reduce((a,b)=>a+b,0),m=payload.metrics;ids.stats.textContent=`${payload.grammar_class} · item_end · L${payload.layer} · ${cohort==='all'?'full retained':'confirmation'} · ${trajectories} trajectories / ${points.length} states / ${seeds} seeds · nₖ ${Math.min(...support)}–${Math.max(...support)} · EVR3 ${evr.toFixed(1)}% · C-held-out Log/NCC ${(100*m.confirmation_logistic).toFixed(1)}%/${(100*m.confirmation_ncc).toFixed(1)}% · SNR ${m.confirmation_snr_db.toFixed(2)} dB`;
}
function setupCleanGrammar(model){const ids=cleanIds(model);if(!ids.canvas)return;const redraw=()=>drawCleanGrammar3D(model);ids.cohort.addEventListener('change',redraw);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=CLEAN_VIEWS[ids.canvas.id]||(CLEAN_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;redraw()});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);redraw()}
for(const model of Object.keys(CLEAN_GRAMMAR)){try{setupCleanGrammar(model)}catch(error){console.error('clean grammar panel failed',model,error)}}
window.addEventListener('resize',()=>{for(const model of Object.keys(CLEAN_GRAMMAR)){try{drawCleanGrammar3D(model)}catch(error){console.error('clean grammar resize failed',model,error)}}});
""".replace("__CLEAN_GRAMMAR__", payload)


def _grammar_filter_script(visual: Mapping[str, Any]) -> str:
    if not visual:
        return ""
    payload = json.dumps(
        visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const GRAMMAR_FILTER=__GRAMMAR_FILTER__;
const GRAMMAR_FILTER_VIEWS={};
function grammarFilterIds(model,mode){const s=slug(model),short=mode==='non_thinking'?'non':'native',base='grammar-filter-'+s+'-'+short;return {cohort:document.getElementById('grammar-filter-'+s+'-cohort'),canvas:document.getElementById(base),stats:document.getElementById(base+'-stats')}}
function drawGrammarFilter3D(model,mode){
  const modelPayload=GRAMMAR_FILTER.models[model],payload=modelPayload[mode],ids=grammarFilterIds(model,mode);if(!payload||!ids.canvas)return;const cohort=ids.cohort.value,points=payload.points.filter(p=>cohort==='all'||p.split==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No retained states in this cohort.',20,30);return}
  const view=GRAMMAR_FILTER_VIEWS[canvas.id]||(GRAMMAR_FILTER_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),groups=new Map();for(const p of points){if(!groups.has(p.occurrence))groups.set(p.occurrence,[]);groups.get(p.occurrence).push(p)}
  const cent=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ps])=>({k,n:ps.length,x:ps.reduce((s,p)=>s+p.x,0)/ps.length,y:ps.reduce((s,p)=>s+p.y,0)/ps.length,z:ps.reduce((s,p)=>s+p.z,0)/ps.length})),rotated=points.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)})),rcent=cent.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)}));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p.x),Math.abs(p.y),Math.abs(p.z)]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]],xy=rotated.map(o=>o.r).concat(rcent.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  c.strokeStyle='#4B5563';c.globalAlpha=.78;c.lineWidth=2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);
  for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.38+.50*depth;c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.occurrence-1))];c.strokeStyle=o.p.split==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=o.p.split==='confirmation'?1.9:.8;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.4+1.1*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const o of rcent){c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.k-1))];c.strokeStyle='#20242D';c.lineWidth=1.4;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(o.p.k),sx(o.r[0])+7,sy(o.r[1])-6)}
  const trajectories=new Set(points.map(p=>p.request_id)).size,seeds=new Set(points.map(p=>p.seed)).size,support=cent.map(p=>p.n),evr=100*payload.pca3_explained_variance_ratio.reduce((a,b)=>a+b,0),m=payload.metrics,label=mode==='non_thinking'?'span_end':'item_end';ids.stats.textContent=`${modelPayload.grammar_class} filter · ${label} · L${payload.layer} · ${cohort==='all'?'paired discovery + confirmation':'frozen confirmation'} · ${trajectories} trajectories / ${points.length} states / ${seeds} seeds · nₖ ${Math.min(...support)}–${Math.max(...support)} · EVR3 ${evr.toFixed(1)}% · C Log/NCC ${(100*m.confirmation_logistic_balanced_accuracy).toFixed(1)}%/${(100*m.confirmation_ncc_balanced_accuracy).toFixed(1)}% · SNR ${m.confirmation_class_balanced_snr_db.toFixed(2)} dB · PCA3 sil ${payload.confirmation_pca3_class_balanced_silhouette.toFixed(3)} · RSA ${payload.confirmation_pca3_ordinal_rsa.toFixed(3)}`;
}
function setupGrammarFilter(model){const non=grammarFilterIds(model,'non_thinking'),redraw=()=>{drawGrammarFilter3D(model,'non_thinking');drawGrammarFilter3D(model,'native_thinking')};non.cohort.addEventListener('change',redraw);for(const mode of ['non_thinking','native_thinking']){const ids=grammarFilterIds(model,mode);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=GRAMMAR_FILTER_VIEWS[ids.canvas.id]||(GRAMMAR_FILTER_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawGrammarFilter3D(model,mode)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop)}redraw()}
for(const model of Object.keys(GRAMMAR_FILTER.models)){try{setupGrammarFilter(model)}catch(error){console.error('grammar-filter panel failed',model,error)}}
window.addEventListener('resize',()=>{for(const model of Object.keys(GRAMMAR_FILTER.models))for(const mode of ['non_thinking','native_thinking']){try{drawGrammarFilter3D(model,mode)}catch(error){console.error('grammar-filter resize failed',model,mode,error)}}});
""".replace("__GRAMMAR_FILTER__", payload)


def _pure_trace_n10_script(visual: Mapping[str, Any]) -> str:
    if not visual:
        return ""
    payload = json.dumps(
        visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const PURE_TRACE_N10=__PURE_TRACE_N10__;
const PURE_TRACE_N10_VIEWS={};
function pureTraceN10Ids(model,mode){const s=slug(model),short=mode==='non_thinking'?'non':'native',base='pure-trace-n10-'+s+'-'+short;return {cohort:document.getElementById('pure-trace-n10-'+s+'-cohort'),canvas:document.getElementById(base),stats:document.getElementById(base+'-stats')}}
function drawPureTraceN10(model,mode){
  const modelPayload=PURE_TRACE_N10.models[model],payload=modelPayload[mode],ids=pureTraceN10Ids(model,mode);if(!payload||!ids.canvas)return;const cohort=ids.cohort.value,points=payload.points.filter(p=>cohort==='all'||p.split==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No retained states in this cohort.',20,30);return}
  const view=PURE_TRACE_N10_VIEWS[canvas.id]||(PURE_TRACE_N10_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),classGroups=new Map(),traceGroups=new Map();for(const p of points){if(!classGroups.has(p.occurrence))classGroups.set(p.occurrence,[]);classGroups.get(p.occurrence).push(p);if(!traceGroups.has(p.request_id))traceGroups.set(p.request_id,[]);traceGroups.get(p.request_id).push(p)}
  const cent=[...classGroups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ps])=>({k,n:ps.length,x:ps.reduce((s,p)=>s+p.x,0)/ps.length,y:ps.reduce((s,p)=>s+p.y,0)/ps.length,z:ps.reduce((s,p)=>s+p.z,0)/ps.length})),rotated=points.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)})),rcent=cent.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)}));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p.x),Math.abs(p.y),Math.abs(p.z)]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]],xy=rotated.map(o=>o.r).concat(rcent.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  c.strokeStyle='#7A7270';c.lineWidth=.75;c.globalAlpha=.20;for(const ps of traceGroups.values()){const path=ps.slice().sort((a,b)=>a.occurrence-b.occurrence).map(p=>rotate3(p.x,p.y,p.z,view));c.beginPath();path.forEach((r,i)=>i?c.lineTo(sx(r[0]),sy(r[1])):c.moveTo(sx(r[0]),sy(r[1])));c.stroke()}
  c.strokeStyle='#303744';c.globalAlpha=.86;c.lineWidth=2.2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);
  for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.36+.50*depth;c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.occurrence-1))];c.strokeStyle=o.p.split==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=o.p.split==='confirmation'?1.9:.8;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.5+1.1*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const o of rcent){c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.k-1))];c.strokeStyle='#20242D';c.lineWidth=1.4;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(o.p.k),sx(o.r[0])+7,sy(o.r[1])-6)}
  const trajectories=traceGroups.size,seeds=new Set(points.map(p=>p.seed)).size,support=cent.map(p=>p.n),evr=100*payload.pca3_explained_variance_ratio.reduce((a,b)=>a+b,0),m=payload.metrics,label=mode==='non_thinking'?'span_end':'item_end';ids.stats.textContent=`${modelPayload.grammar_class} · ${label} · L${payload.layer} · ${cohort==='all'?'discovery + confirmation':'frozen confirmation'} · ${trajectories} complete trajectories / ${points.length} states / ${seeds} seeds · nₖ ${Math.min(...support)}–${Math.max(...support)} · EVR3 ${evr.toFixed(1)}% · C Log/NCC ${(100*m.confirmation_logistic_balanced_accuracy).toFixed(1)}%/${(100*m.confirmation_ncc_balanced_accuracy).toFixed(1)}% · SNR ${m.confirmation_class_balanced_snr_db.toFixed(2)} dB · PCA3 sil ${payload.confirmation_pca3_class_balanced_silhouette.toFixed(3)} · RSA ${payload.confirmation_pca3_ordinal_rsa.toFixed(3)}`;
}
function setupPureTraceN10(model){const non=pureTraceN10Ids(model,'non_thinking'),redraw=()=>{drawPureTraceN10(model,'non_thinking');drawPureTraceN10(model,'native_thinking')};non.cohort.addEventListener('change',redraw);for(const mode of ['non_thinking','native_thinking']){const ids=pureTraceN10Ids(model,mode);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=PURE_TRACE_N10_VIEWS[ids.canvas.id]||(PURE_TRACE_N10_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawPureTraceN10(model,mode)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop)}redraw()}
for(const model of Object.keys(PURE_TRACE_N10.models)){try{setupPureTraceN10(model)}catch(error){console.error('pure-trace N10 panel failed',model,error)}}
window.addEventListener('resize',()=>{for(const model of Object.keys(PURE_TRACE_N10.models))for(const mode of ['non_thinking','native_thinking']){try{drawPureTraceN10(model,mode)}catch(error){console.error('pure-trace N10 resize failed',model,mode,error)}}});
""".replace("__PURE_TRACE_N10__", payload)


def _indexed_numeric_n10_script(visual: Mapping[str, Any]) -> str:
    if not visual:
        return ""
    payload = json.dumps(
        visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return r"""
const INDEXED_NUMERIC_N10=__INDEXED_NUMERIC_N10__;
const INDEXED_NUMERIC_N10_VIEWS={};
function indexedNumericN10Ids(model,mode){const s=slug(model),short=mode==='non_thinking'?'non':'native',base='indexed-numeric-n10-'+s+'-'+short;return {cohort:document.getElementById('indexed-numeric-n10-'+s+'-cohort'),layer:document.getElementById(base+'-layer'),canvas:document.getElementById(base),stats:document.getElementById(base+'-stats')}}
function drawIndexedNumericN10(model,mode){
  const modelPayload=INDEXED_NUMERIC_N10.models[model],modePayload=modelPayload[mode],ids=indexedNumericN10Ids(model,mode);if(!modePayload||!ids.canvas||!ids.layer)return;const layer=String(ids.layer.value),payload=modePayload.layers[layer];if(!payload)return;const cohort=ids.cohort.value,points=payload.points.filter(p=>cohort==='all'||p.split==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height;c.clearRect(0,0,w,h);if(!points.length){c.fillStyle='#626A74';c.font='14px Segoe UI';c.fillText('No retained states in this cohort.',20,30);return}
  const view=INDEXED_NUMERIC_N10_VIEWS[canvas.id]||(INDEXED_NUMERIC_N10_VIEWS[canvas.id]={yaw:-.72,pitch:.46}),classGroups=new Map();for(const p of points){if(!classGroups.has(p.occurrence))classGroups.set(p.occurrence,[]);classGroups.get(p.occurrence).push(p)}
  const cent=[...classGroups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ps])=>({k,n:ps.length,x:ps.reduce((s,p)=>s+p.x,0)/ps.length,y:ps.reduce((s,p)=>s+p.y,0)/ps.length,z:ps.reduce((s,p)=>s+p.z,0)/ps.length})),rotated=points.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)})),rcent=cent.map(p=>({p,r:rotate3(p.x,p.y,p.z,view)}));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p.x),Math.abs(p.y),Math.abs(p.z)]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]],xy=rotated.map(o=>o.r).concat(rcent.map(o=>o.r),axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:24,r:24,t:18,b:23},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.2;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  c.strokeStyle='#303744';c.globalAlpha=.86;c.lineWidth=2.2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);
  for(const o of rotated){const depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.36+.50*depth;c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.occurrence-1))];c.strokeStyle=o.p.split==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=o.p.split==='confirmation'?1.9:.8;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.5+1.1*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const o of rcent){c.fillStyle=COLORS[Math.max(0,Math.min(9,o.p.k-1))];c.strokeStyle='#20242D';c.lineWidth=1.4;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(o.p.k),sx(o.r[0])+7,sy(o.r[1])-6)}
  const trajectories=new Set(points.map(p=>p.request_id)).size,seeds=new Set(points.map(p=>p.seed)).size,support=cent.map(p=>p.n),evr=100*payload.pca3_explained_variance_ratio.reduce((a,b)=>a+b,0),m=payload.metrics,label=mode==='non_thinking'?'span_end':(modelPayload.native_site_kind||'item_end'),best=Number(layer)===Number(modePayload.selected_layer)?' · discovery best':'',surface=modelPayload.surface_label||'strict k. city - score';ids.stats.textContent=`${surface} · ${label} · L${layer}${best} · ${cohort==='all'?'discovery + confirmation':'secondary frozen confirmation'} · ${trajectories} complete trajectories / ${points.length} states / ${seeds} seeds · nₖ ${Math.min(...support)}–${Math.max(...support)} · EVR3 ${evr.toFixed(1)}% · C Log/NCC ${(100*m.confirmation_logistic_balanced_accuracy).toFixed(1)}%/${(100*m.confirmation_ncc_balanced_accuracy).toFixed(1)}% · SNR ${m.confirmation_class_balanced_snr_db.toFixed(2)} dB · PCA3 sil ${payload.confirmation_pca3_class_balanced_silhouette.toFixed(3)} · RSA ${payload.confirmation_pca3_ordinal_rsa.toFixed(3)}`;
}
function setupIndexedNumericN10(model){const non=indexedNumericN10Ids(model,'non_thinking'),native=indexedNumericN10Ids(model,'native_thinking'),redraw=()=>{drawIndexedNumericN10(model,'non_thinking');drawIndexedNumericN10(model,'native_thinking')};non.cohort.addEventListener('change',redraw);non.layer.addEventListener('change',()=>drawIndexedNumericN10(model,'non_thinking'));native.layer.addEventListener('change',()=>drawIndexedNumericN10(model,'native_thinking'));for(const mode of ['non_thinking','native_thinking']){const ids=indexedNumericN10Ids(model,mode);let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=INDEXED_NUMERIC_N10_VIEWS[ids.canvas.id]||(INDEXED_NUMERIC_N10_VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawIndexedNumericN10(model,mode)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop)}redraw()}
for(const model of Object.keys(INDEXED_NUMERIC_N10.models)){try{setupIndexedNumericN10(model)}catch(error){console.error('indexed-numeric N10 panel failed',model,error)}}
window.addEventListener('resize',()=>{for(const model of Object.keys(INDEXED_NUMERIC_N10.models))for(const mode of ['non_thinking','native_thinking']){try{drawIndexedNumericN10(model,mode)}catch(error){console.error('indexed-numeric N10 resize failed',model,mode,error)}}});
""".replace("__INDEXED_NUMERIC_N10__", payload)


def build_html(
    *,
    dual_results: Mapping[str, Mapping[str, Any]],
    dual_visual: Mapping[str, Any],
    fisher_lda_html: str,
    fisher_lda_visual: Mapping[str, Any],
    token_html: str,
    causal_html: str,
    new_native_html: str,
    new_native_visual: Mapping[str, Any],
    index_city_html: str,
    index_city_visual: Mapping[str, Any],
    phase_grammar_html: str,
    phase_grammar_visual: Mapping[str, Any],
    marker_html: str,
    band_html: str,
    band_visual: Mapping[str, Any],
    band_audits: Mapping[str, Mapping[str, Any]],
    grammar_filter_html: str = "",
    grammar_filter_visual: Mapping[str, Any] | None = None,
    pure_trace_n10_html: str = "",
    pure_trace_n10_visual: Mapping[str, Any] | None = None,
    indexed_numeric_n10_html: str = "",
    indexed_numeric_n10_visual: Mapping[str, Any] | None = None,
    nonthinking_internal_html: str = "",
    domain_html: str = "",
    domain_visual: Mapping[str, Any] | None = None,
    domain_endpoint_html: str = "",
    domain_endpoint_visual: Mapping[str, Any] | None = None,
) -> str:
    css = """
:root{--paper:#F3EEE4;--surface:#FFFDF8;--ink:#20242D;--muted:#626A74;--line:#C9C2B6;--indigo:#23165C;--teal:#00A88F;--yellow:#D6B52C}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;overflow-x:hidden;background:var(--paper);color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;line-height:1.62}a{color:var(--indigo)}nav{position:sticky;top:0;z-index:5;display:flex;gap:18px;padding:10px 22px;background:rgba(243,238,228,.96);border-bottom:1px solid var(--line);overflow-x:auto}nav a{color:var(--indigo);font-size:13px;font-weight:750;text-decoration:none;white-space:nowrap}main{max-width:1480px;margin:auto;padding:38px 28px 80px}header{max-width:1080px;border-bottom:2px solid var(--ink);padding-bottom:28px}.eyebrow{font:700 12px/1.2 Consolas,monospace;letter-spacing:.12em;color:var(--teal)}h1{font-size:44px;line-height:1.08;margin:10px 0 16px;letter-spacing:-.035em}h2{font-size:29px;margin:0 0 12px}h4{color:var(--indigo)}.lead{font-size:18px;color:#404852;max-width:92ch}section{padding:46px 0;border-bottom:1px solid var(--line)}.callout{max-width:1120px;background:var(--surface);border-left:4px solid var(--teal);padding:15px 19px;margin:20px 0}.warning{border-left-color:var(--yellow)}.definitions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}.definitions.two{grid-template-columns:repeat(2,minmax(0,1fr))}.definitions>div,.geometry-card,.appendix-model{min-width:0;background:var(--surface);border:1px solid var(--line);padding:17px}.definitions h3,.geometry-card h3{color:var(--indigo);margin:0 0 8px;font-size:17px}.definitions p,.geometry-card p{font-size:13px;color:var(--muted);margin:0 0 12px}.controls{display:flex;gap:12px;flex-wrap:wrap}.controls label,.layer-control{font-size:12px;font-weight:700;color:var(--muted)}.layer-control{display:inline-block;margin:0 0 10px}select{display:block;margin-top:4px;border:1px solid var(--line);background:var(--surface);padding:7px 28px 7px 9px;color:var(--ink)}.seed-list{white-space:normal;word-break:break-word}.dual-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:28px}.geometry-card canvas{display:block;width:100%;height:390px;background:#F8F4EC;border:1px solid #DDD5C9;touch-action:none;cursor:grab}.geometry-card canvas:active{cursor:grabbing}.rotate-hint{margin-top:5px;color:#7A7270;font:10px/1.4 Consolas,monospace}.panel-stats{min-height:70px;margin-top:7px;color:var(--muted);font:12px/1.5 Consolas,monospace}.table-scroll{overflow:auto;background:var(--surface);border:1px solid var(--line);margin:16px 0 22px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #DED8CE}th{background:#ECE6DA;color:#303744}.site-badge{display:block;width:max-content;margin-top:5px;padding:2px 6px;border:1px solid currentColor;border-radius:2px;font:700 9px/1.3 Consolas,monospace;letter-spacing:.04em}.primary-badge{color:var(--teal)}.winner-badge{color:var(--indigo)}.muted,.small{color:var(--muted);font-size:12px}.metric-grid,.metric-guide-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:16px 0 20px}.metric-guide-card{min-width:0;background:var(--surface);border:1px solid var(--line);padding:17px}.metric-guide-card h3{color:var(--indigo);margin:0 0 9px;font-size:17px}.metric-guide-card p{font-size:13px;margin:8px 0}.metric-guide-card .formula{background:#ECE6DA;color:#303744;padding:8px 10px;font:12px/1.5 Consolas,monospace}.metric-figure{min-width:0;margin:0;background:var(--surface);border:1px solid var(--line);padding:13px}.metric-figure h3{margin:0;color:var(--indigo);font-size:17px}.metric-figure svg{display:block;width:100%;height:auto}.metric-gridline{stroke:#D9D2C7;stroke-width:1}.metric-zero{stroke:#756E68;stroke-width:1.5}.metric-tick,.metric-label,.metric-value,.metric-axis-title{fill:#303744;font:12px Consolas,monospace}.metric-tick{fill:var(--muted);font-size:11px}.metric-link{stroke:#8A838E;stroke-width:2}.metric-dot{stroke:#FFFDF8;stroke-width:2}.metric-non,.snr-non{fill:#20242D}.metric-native,.snr-native,.metric-answer{fill:#00A88F}.metric-running{fill:#6750E8}.snr-upper{fill:#E76F51}.snr-lower{fill:#6750E8}.metric-legend,.band-dynamic-legend{display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:10px 0}.metric-legend span,.band-dynamic-legend span{display:inline-flex;align-items:center;gap:6px}.metric-legend i,.band-dynamic-legend i{display:inline-block;width:11px;height:11px;border-radius:50%;background:#8A838E}.metric-legend .legend-non{background:#20242D}.metric-legend .legend-native,.metric-legend .legend-answer{background:#00A88F}.metric-legend .legend-running{background:#6750E8}.metric-legend .legend-upper{background:#E76F51}.metric-legend .legend-lower{background:#6750E8}.band-dynamic-legend i.square{border-radius:0}.band-dynamic-legend b{font-weight:500}.token-flow{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:20px 0}.token-flow article{min-width:0;background:var(--surface);border:1px solid var(--line);padding:17px}.token-flow h3{color:var(--indigo);margin:0 0 8px;font-size:17px}.token-flow p{font-size:13px;color:var(--muted)}.token-strip{display:flex;gap:5px;align-items:flex-start;flex-wrap:wrap;margin:17px 0 26px}.token-strip span{position:relative;background:#ECE6DA;padding:5px 7px;font:12px Consolas,monospace}.token-strip span[data-pos]::after{content:attr(data-pos);position:absolute;left:50%;top:100%;transform:translateX(-50%);font:9px Consolas,monospace;color:#7A7270}.token-strip .picked{background:#00A88F;color:#FFFDF8}.token-strip b{font:11px Consolas,monospace;color:var(--indigo);padding:5px}.boundary-example{display:flex;align-items:stretch;margin:15px 0;font:12px/1.5 Consolas,monospace}.boundary-example span{background:#ECE6DA;padding:8px 10px}.boundary-example i{display:block;width:4px;background:#E76F51}.boundary-example .answer-token{background:#D9F1EA}.band-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:18px 0}.band-figure{min-width:0;margin:0;background:#FFFDF8;border:1px solid var(--line);padding:12px}.band-figure h4{font-size:15px;color:var(--indigo);margin:0 0 8px}.band-figure canvas{display:block;width:100%;height:380px;background:#F8F4EC;border:1px solid #DDD5C9;touch-action:none;cursor:grab}.band-figure canvas:active{cursor:grabbing}.band-controls{margin-top:15px}.appendix-model{margin:22px 0}.domain-legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:-14px 0 18px}.domain-legend span{display:inline-flex;align-items:center;gap:6px}.domain-legend i{display:inline-block;width:11px;height:11px;background:#20242D}.domain-legend .domain-city{border-radius:50%}.domain-legend .domain-flower{background:#00A88F;clip-path:polygon(50% 0,100% 100%,0 100%)}.domain-legend .domain-animal{background:#E76F51}.domain-dim-figure{margin:18px 0}.domain-dim-line{fill:none;stroke-width:2.3}.domain-line-non{stroke:#20242D;fill:#20242D}.domain-line-native{stroke:#00A88F;fill:#00A88F}.domain-dim-mark{stroke:#FFFDF8;stroke-width:1.4}.domain-chance{stroke:#8A838E;stroke-width:1.3;stroke-dasharray:5 4}.provenance{font:11px/1.6 Consolas,monospace;color:var(--muted)}details{background:var(--surface);border:1px solid var(--line);margin:18px 0}summary{cursor:pointer;padding:12px 15px;font-weight:750;color:var(--indigo)}@media(max-width:1000px){.dual-grid,.definitions,.definitions.two,.metric-grid,.metric-guide-grid,.token-flow,.band-grid{grid-template-columns:1fr}}@media(max-width:650px){main{padding:25px 13px 60px}h1{font-size:34px}.geometry-card canvas,.band-figure canvas{height:330px}.metric-value{font-size:10px}}
"""
    script = (
        _dual_script(dual_visual)
        + _band_script(band_visual)
        + _grammar_filter_script(grammar_filter_visual or {})
        + _pure_trace_n10_script(pure_trace_n10_visual or {})
        + _indexed_numeric_n10_script(indexed_numeric_n10_visual or {})
        + _domain_script(domain_visual or {})
        + _domain_endpoint_script(domain_endpoint_visual or {})
        + _clean_grammar_script(new_native_visual)
        + _index_city_script(index_city_visual)
        + _phase_grammar_script(phase_grammar_visual)
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NiaH Geometry Comparison</title><style>{css}</style></head><body>
<nav><a href="#scope">口径</a><a href="#tokens">Token 提取</a><a href="#causal-aligned">Causal 对齐</a><a href="#metric-guide">指标定义</a><a href="#claims">核心结果</a><a href="#dual">PCA 对比</a><a href="#snr">SNR</a><a href="#appendix-bands">Appendix A</a>{'<a href="#appendix-domain-transfer">Appendix B</a>' if domain_html else ''}{'<a href="#appendix-domain-endpoints">B.2 endpoints</a>' if domain_endpoint_html else ''}{'<a href="#appendix-indexed-numeric-n10">Appendix C</a>' if indexed_numeric_n10_html else ''}{'<a href="#appendix-nonthinking-internal">Appendix D</a>' if nonthinking_internal_html else ''}{'<a href="#appendix-clean-grammar">Appendix E</a>' if new_native_html else ''}{'<a href="#appendix-index-city">E.2 index+city</a>' if index_city_html else ''}{'<a href="#appendix-phase-grammar">Appendix F</a>' if phase_grammar_html else ''}</nav><main>
<header><div class="eyebrow">REALISTIC NIAH · ALL-COUNT GEOMETRY</div><h1>NiaH Geometry Comparison</h1><p class="lead">Running index 与 final count 的全部四组正式 PCA 对比均保留，并覆盖 N=1…10 的完整 300 trajectories 与 confirmation 100 trajectories。最稳定的跨模型现象不是“每张 PCA 都更紧”，而是 native-thinking 在 Logistic 与 nearest-centroid 的全部 8 个 frozen held-out 比较中都更可解码；报告据此先给结论，再完整展示 PCA 与负/混合视觉结果。</p></header>
<section id="scope"><h2>严格比较口径</h2><div class="definitions"><div><h3>Full 300</h3><p>10 个 gold N × 30 seeds。它是 descriptive geometry view；PCA3 仍只由 discovery 200 拟合，避免 confirmation 反向选显示 basis。</p></div><div><h3>Confirmation 100</h3><p>10 个 gold N × 10 held-out seeds。主表的 Logistic、nearest-centroid 与 SNR 都是 discovery-frozen 后在这里评价。</p></div><div><h3>Native running 的 ragged rule</h3><p>每条 trace 只贡献 parser 实际观察到的 1…M。数到 8 就贡献八个 states；不按 gold N 或最终 Total 补到 9/10。</p></div></div></section>
{token_html}
{causal_html}
{metric_guide_section()}
{empirical_claims(dual_results)}
{dual_endpoint_section(dual_results, dual_visual)}
{snr_section(dual_results, band_audits)}
{marker_html}{band_html}{domain_html}{domain_endpoint_html}{grammar_filter_html}{pure_trace_n10_html}{indexed_numeric_n10_html}
{nonthinking_internal_html}
{new_native_html}{index_city_html}{phase_grammar_html}
<section><h2>解释边界</h2><p>这些图和 probes 证明的是 within-task decodability/geometry，不单独证明离散计数器、逐步加一算法或因果使用。两个 mode 的 end token 语义和最佳层仍不同，因此比较的是各自语义对齐的 single-token 边界上同一任务变量的可读性，而不是共享坐标系中的绝对距离。Appendix C 主图明确包含显式 running-index surface；Gemma 另报 retrieval-complete、label-not-yet-visible 的 <code>pre_marker</code> sensitivity control。</p><p class="provenance">Report schema: {REPORT_SCHEMA_VERSION} · pooled 10 counts × 30 seeds · full/confirmation views: 300/100 trajectories · pooled running sites fixed: span_end/item_end · Appendix C main sites: Qwen score-digit item_end, Gemma closing-parenthesis item_end · Gemma sensitivity site: pre_marker · layer selector: discovery only · new-parser grammar diagnosis: Appendix A · entity-domain diagnostics: Appendix B/B.2 · Qwen/Gemma strict single-surface N=10 secondary 20/10 layer sweeps: Appendix C</p></section>
</main><script>{script}</script></body></html>"""


def build_report(
    *,
    non_thinking_export_root: Path,
    native_running_root: Path,
    native_final_root: Path,
    dual_endpoint_root: Path,
    parser_audit: Path,
    band_root: Path,
    grammar_registry: Path,
    output: Path,
    manifest_path: Path,
    fisher_lda_root: Path | None = None,
    domain_transfer_root: Path | None = None,
    covariance_root: Path | None = None,
    causal_aligned_root: Path | None = None,
    grammar_filter_root: Path | None = None,
    pure_trace_n10_root: Path | None = None,
    indexed_numeric_n10_root: Path | None = None,
    gemma_count_colon_n10_root: Path | None = None,
    gemma_premarker_n10_root: Path | None = None,
    clean_grammar_root: Path | None = None,
    post_marker_root: Path | None = None,
    index_city_root: Path | None = None,
    domain_endpoint_root: Path | None = None,
    phase_geometry_root: Path | None = None,
    phase_grammar_root: Path | None = None,
    standard_grammar_root: Path | None = None,
) -> dict[str, Any]:
    dual_results, dual_inputs = load_dual_endpoint_results(
        dual_endpoint_root.resolve()
    )
    dual_visual, visual_inputs = build_dual_visual_data(
        non_thinking_export_root.resolve(),
        native_running_root.resolve(),
        native_final_root.resolve(),
        dual_results,
    )
    fisher_lda_html = ""
    fisher_lda_inputs: list[Path] = []
    fisher_lda_visual: dict[str, Any] = {"models": {}}
    if fisher_lda_root is not None:
        (
            fisher_lda_html,
            fisher_lda_inputs,
            fisher_lda_visual,
        ) = fisher_lda_section(fisher_lda_root.resolve())
    token_html, token_inputs = token_extraction_section(
        native_running_root.resolve(),
        native_final_root.resolve(),
        grammar_registry.resolve(),
    )
    causal_html = ""
    causal_inputs: list[Path] = []
    if causal_aligned_root is not None:
        causal_results, causal_inputs = load_causal_aligned_results(
            causal_aligned_root.resolve()
        )
        causal_html = causal_aligned_progress_section(causal_results)
    new_native_html = ""
    new_native_inputs: list[Path] = []
    new_native_visual: dict[str, Any] = {}
    if clean_grammar_root is not None and post_marker_root is not None:
        (
            new_native_html,
            new_native_inputs,
            new_native_visual,
        ) = new_native_geometry_section(
            clean_grammar_root.resolve(), post_marker_root.resolve()
        )
    index_city_html = ""
    index_city_inputs: list[Path] = []
    index_city_visual: dict[str, Any] = {}
    if index_city_root is not None:
        (
            index_city_html,
            index_city_inputs,
            index_city_visual,
        ) = index_city_geometry_appendix(index_city_root.resolve())
    phase_grammar_html = ""
    phase_grammar_inputs: list[Path] = []
    phase_grammar_visual: dict[str, Any] = {}
    if phase_geometry_root is not None:
        if covariance_root is None:
            raise ValueError("--phase-geometry-root requires --covariance-root")
        (
            phase_grammar_html,
            phase_grammar_inputs,
            phase_grammar_visual,
        ) = phase_grammar_ablation_appendix(
            phase_geometry_root.resolve(),
            covariance_root.resolve(),
            phase_grammar_root.resolve() if phase_grammar_root is not None else None,
            standard_grammar_root.resolve()
            if standard_grammar_root is not None
            else None,
        )
    band_html, band_inputs, band_visual, band_audits = band_appendix(
        band_root.resolve(), grammar_registry.resolve()
    )
    grammar_filter_html = ""
    grammar_filter_inputs: list[Path] = []
    grammar_filter_visual: dict[str, Any] = {}
    if grammar_filter_root is not None:
        (
            grammar_filter_html,
            grammar_filter_inputs,
            grammar_filter_visual,
        ) = grammar_filtered_comparison(grammar_filter_root.resolve())
    pure_trace_n10_html = ""
    pure_trace_n10_inputs: list[Path] = []
    pure_trace_n10_visual: dict[str, Any] = {}
    if pure_trace_n10_root is not None:
        (
            pure_trace_n10_html,
            pure_trace_n10_inputs,
            pure_trace_n10_visual,
        ) = pure_trace_n10_comparison(pure_trace_n10_root.resolve())
    indexed_numeric_n10_html = ""
    indexed_numeric_n10_inputs: list[Path] = []
    indexed_numeric_n10_visual: dict[str, Any] = {}
    if indexed_numeric_n10_root is not None:
        (
            indexed_numeric_n10_html,
            indexed_numeric_n10_inputs,
            indexed_numeric_n10_visual,
        ) = indexed_numeric_n10_comparison(
            indexed_numeric_n10_root.resolve(),
            gemma_count_colon_n10_root.resolve()
            if gemma_count_colon_n10_root is not None
            else None,
            gemma_premarker_n10_root.resolve()
            if gemma_premarker_n10_root is not None
            else None,
        )
    domain_html = ""
    domain_inputs: list[Path] = []
    domain_visual: dict[str, Any] = {}
    if domain_transfer_root is not None:
        domain_html, domain_inputs, domain_visual = domain_transfer_appendix(
            domain_transfer_root.resolve()
        )
    domain_endpoint_html = ""
    domain_endpoint_inputs: list[Path] = []
    domain_endpoint_visual: dict[str, Any] = {}
    if domain_endpoint_root is not None:
        (
            domain_endpoint_html,
            domain_endpoint_inputs,
            domain_endpoint_visual,
        ) = domain_endpoint_comparison_appendix(domain_endpoint_root.resolve())
    nonthinking_internal_html = ""
    covariance_inputs: list[Path] = []
    if covariance_root is not None:
        internal_metrics, covariance_inputs = load_nonthinking_internal_metrics(
            covariance_root.resolve()
        )
        nonthinking_internal_html = nonthinking_internal_section(
            internal_metrics,
            dual_visual=dual_visual,
            domain_evidence_included=bool(domain_html),
        )
    document = build_html(
        dual_results=dual_results,
        dual_visual=dual_visual,
        fisher_lda_html=fisher_lda_html,
        fisher_lda_visual=fisher_lda_visual,
        token_html=token_html,
        causal_html=causal_html,
        new_native_html=new_native_html,
        new_native_visual=new_native_visual,
        index_city_html=index_city_html,
        index_city_visual=index_city_visual,
        phase_grammar_html=phase_grammar_html,
        phase_grammar_visual=phase_grammar_visual,
        marker_html="",
        band_html=band_html,
        band_visual=band_visual,
        band_audits=band_audits,
        grammar_filter_html=grammar_filter_html,
        grammar_filter_visual=grammar_filter_visual,
        pure_trace_n10_html=pure_trace_n10_html,
        pure_trace_n10_visual=pure_trace_n10_visual,
        indexed_numeric_n10_html=indexed_numeric_n10_html,
        indexed_numeric_n10_visual=indexed_numeric_n10_visual,
        nonthinking_internal_html=nonthinking_internal_html,
        domain_html=domain_html,
        domain_visual=domain_visual,
        domain_endpoint_html=domain_endpoint_html,
        domain_endpoint_visual=domain_endpoint_visual,
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    inputs = sorted(
        set(
            dual_inputs
            + visual_inputs
            + fisher_lda_inputs
            + band_inputs
            + grammar_filter_inputs
            + pure_trace_n10_inputs
            + indexed_numeric_n10_inputs
            + token_inputs
            + domain_inputs
            + covariance_inputs
            + causal_inputs
            + new_native_inputs
            + index_city_inputs
            + phase_grammar_inputs
            + domain_endpoint_inputs
        ),
        key=str,
    )
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": {
            "full": "10 gold counts x 30 seeds = 300 trajectories",
            "confirmation": "10 gold counts x 10 seeds = 100 trajectories",
        },
        "running_state_rule": "parser-observed 1..M only; never pad to gold N or final Total",
        "answer_query_v3": "independent final-answer extraction; exactly one site per trajectory",
        "primary_analysis": "running sites fixed to span_end/item_end; pooled discovery-only layer selection; frozen confirmation evaluation",
        "fisher_lda3_diagnostic": (
            "each model x endpoint x mode reuses its discovery-classification-selected "
            "layer; StandardScaler, PCA16, class-balanced within covariance whitening, "
            "and top-3 between-count Fisher axes fit on discovery only; confirmation "
            "is the default frozen display; raw PCA3 remains the unsupervised primary view"
            if fisher_lda_html
            else "not included"
        ),
        "causal_aligned_progress_geometry": (
            "CPU-only archived-state reanalysis on exact causal primary progress "
            "commits; item_end is the fixed P0 primary, four paired token sites are "
            "controls/exploratory; site and layer selection use discovery only; "
            "post_marker was absent from this archived cohort and is evaluated separately by a new paired replay capture"
            if causal_aligned_root is not None
            else "not included"
        ),
        "native_item_end_grammar_registry": (
            "new causal-site event registry joined one-to-one to all 3215 native "
            "running states by model/request_id/occurrence; item_end remains fixed "
            "while each surface grammar defines the complete-item boundary"
        ),
        "legacy_marker_appendix": "removed; superseded by the new per-event grammar registry in the token section and Appendix A",
        "trace_format_site_layer_sweep": "omitted; Appendix A diagnoses frozen-band association with the new parser grammar",
        "grammar_filtered_cross_mode": (
            "Appendix C; native grammar and native layer selected on discovery only; "
            "non-thinking paired one-to-one by split/seed/gold-N/running-k and selects "
            "its own layer on paired discovery; confirmation frozen"
            if grammar_filter_html
            else "not included"
        ),
        "pure_trace_n10_cross_mode": (
            "Appendix D; N=10 only; whole native trace must be exact one-to-one "
            "with ten commits from one grammar and marker kind; grammar selected by "
            "qualified discovery trajectory count only; exact non-thinking pairing; "
            "each mode independently selects layer on discovery; confirmation frozen"
            if pure_trace_n10_html
            else "not included"
        ),
        "indexed_numeric_n10_cross_mode": (
            "Appendix C exploratory secondary panels; Qwen requires exact k. city - "
            "score items ending on the score digit; "
            + (
                "Gemma requires exact ten-item nested bullets of the form "
                "Record k: (city, score), with item_end fixed to the bare closing "
                "parenthesis after city/score. "
                if indexed_numeric_n10_visual.get("gemma_surface_family")
                == "controlled_prefix_record"
                else
                "Gemma requires one and only one ten-item reasoning episode whose "
                "items end in (Count: k), with item_end fixed to the closing "
                "parenthesis after the explicit count label. "
            )
            + "Each model has exactly 30 text-eligible "
            "paired seeds and an independent seed-hash 20/10 split; hidden states do "
            "not select seeds. Every layer has its own discovery-fitted PCA3 and "
            "secondary frozen confirmation metrics; all retained seeds and Gemma raw "
            "generation filter rates are printed in the report"
            if indexed_numeric_n10_html
            else "not included"
        ),
        "single_grammar_counter_geometry": (
            "item_end fixed; grammar and layer selected on grouped discovery OOF only; confirmation frozen; Gemma k=10 has n=2"
            if new_native_html else "not included"
        ),
        "fixed_index_city_geometry": (
            "a-priori model-specific rank-before-city grammar; city_end primary and item_end control; layers selected separately by grouped discovery OOF; confirmation frozen"
            if index_city_html else "not included"
        ),
        "native_phase_grammar_ablation": (
            "teacher-forced replay of frozen completions; strict original-token post_city, "
            "post_marker and marker_end controls; full grammar-specific 8-site x layer selection "
            "uses grouped discovery CV and confirmation is frozen; combined view uses one "
            "shared layer and labels discovery grammar-centering as sensitivity-only"
            if phase_grammar_html
            else "not included"
        ),
        "qwen_post_marker_capture": (
            "642 paired replayed rank-before events; pre_marker versus compiler post_marker; no resampling or intervention"
            if new_native_html else "not included"
        ),
        "native_band_snr": "bands and PCA16 frozen on discovery; per-band confirmation SNR requires at least two states per retained k",
        "native_band_conclusion": (
            "new-parser grammar versus frozen band: Qwen full/confirmation NMI about "
            "0.60/0.61, Gemma about 0.03/0.06; grammar-mixture explanation is model-specific"
        ),
        "nonthinking_internal_comparison": (
            "Appendix D exploratory Qwen and Gemma running span_end versus pre-answer "
            "answer_query_v3; interactive PCA3 uses each endpoint's discovery-SNR layer "
            "by default with all layers and full/confirmation cohorts available; unequal "
            "state units, independent display bases, and mixed metric directions are disclosed"
            if covariance_root is not None
            else "not included"
        ),
        "metric_guide": (
            "Logistic/NCC balanced accuracy, isotropic SNR, frozen Fisher trace, "
            "Mahalanobis silhouette, and held-out ordinal RSA each include an "
            "exact calculation, practical interpretation, worked example, and boundary"
        ),
        "entity_domain_transfer": (
            "design only in Appendix B: paired city/flower/animal panels with "
            "city-discovery fitting and frozen transfer; answer-only result figures "
            "and dimension sweep omitted in favor of B.2 running-versus-answer"
            if domain_transfer_root is not None
            else "not included"
        ),
        "entity_domain_endpoint_comparison": (
            "within each model/mode: running occurrence-k versus answer gold-N; independent discovery-selected layers and independent city-discovery PCA/probe bases; city/flower/animal confirmation only"
            if domain_endpoint_html
            else "not included"
        ),
        "inputs": {str(path): sha256(path) for path in inputs},
        "output": str(output),
        "output_sha256": sha256(output),
    }
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-thinking-export-root", type=Path, required=True)
    parser.add_argument("--native-running-root", type=Path, required=True)
    parser.add_argument("--native-final-root", type=Path, required=True)
    parser.add_argument("--dual-endpoint-root", type=Path, required=True)
    parser.add_argument("--parser-audit", type=Path, required=True)
    parser.add_argument("--band-root", type=Path, required=True)
    parser.add_argument("--grammar-registry", type=Path, required=True)
    parser.add_argument("--fisher-lda-root", type=Path)
    parser.add_argument("--domain-transfer-root", type=Path)
    parser.add_argument("--covariance-root", type=Path)
    parser.add_argument("--causal-aligned-root", type=Path)
    parser.add_argument("--grammar-filter-root", type=Path)
    parser.add_argument("--pure-trace-n10-root", type=Path)
    parser.add_argument("--indexed-numeric-n10-root", type=Path)
    parser.add_argument("--gemma-count-colon-n10-root", type=Path)
    parser.add_argument("--gemma-premarker-n10-root", type=Path)
    parser.add_argument("--clean-grammar-root", type=Path)
    parser.add_argument("--post-marker-root", type=Path)
    parser.add_argument("--index-city-root", type=Path)
    parser.add_argument("--domain-endpoint-root", type=Path)
    parser.add_argument("--phase-geometry-root", type=Path)
    parser.add_argument("--phase-grammar-root", type=Path)
    parser.add_argument("--standard-grammar-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_report(
        non_thinking_export_root=args.non_thinking_export_root,
        native_running_root=args.native_running_root,
        native_final_root=args.native_final_root,
        dual_endpoint_root=args.dual_endpoint_root,
        parser_audit=args.parser_audit,
        band_root=args.band_root,
        grammar_registry=args.grammar_registry,
        fisher_lda_root=args.fisher_lda_root,
        domain_transfer_root=args.domain_transfer_root,
        covariance_root=args.covariance_root,
        causal_aligned_root=args.causal_aligned_root,
        grammar_filter_root=args.grammar_filter_root,
        pure_trace_n10_root=args.pure_trace_n10_root,
        indexed_numeric_n10_root=args.indexed_numeric_n10_root,
        gemma_count_colon_n10_root=args.gemma_count_colon_n10_root,
        gemma_premarker_n10_root=args.gemma_premarker_n10_root,
        clean_grammar_root=args.clean_grammar_root,
        post_marker_root=args.post_marker_root,
        index_city_root=args.index_city_root,
        domain_endpoint_root=args.domain_endpoint_root,
        phase_geometry_root=args.phase_geometry_root,
        phase_grammar_root=args.phase_grammar_root,
        standard_grammar_root=args.standard_grammar_root,
        output=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
