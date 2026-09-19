from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ARM_LABELS = {
    "clean": "Clean",
    "selected_bank": "Ranked K",
    "layer_matched_random_mean": "3× layer-matched random mean",
    "layer_matched_random_repeat_1": "Random repeat 1",
    "layer_matched_random_repeat_2": "Random repeat 2",
    "layer_matched_random_repeat_3": "Random repeat 3",
}
GRAMMAR_LABELS = {
    "pooled": "Natural-frequency pooled",
    "macro_primary_grammars": "Equal-primary-grammar macro",
    "adjacent_rank_after_city": "adjacent rank-after-city",
    "adjacent_rank_before_city": "adjacent rank-before-city",
    "same_unit_rank_after_city": "same-unit rank-after-city",
    "same_unit_rank_before_city": "same-unit rank-before-city",
    "structural_unmarked": "structural unmarked",
    "structural_invariant_bullet": "invariant bullet",
    "evidence_sequence_unranked": "unranked evidence sequence",
    "structural_explicit_rank_before_city": "explicit structural rank",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(number):
        return "—"
    return f"{100 * number:.{digits}f}%"


def _p(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(number):
        return "—"
    return "<0.0001" if number < 0.0001 else f"{number:.4f}"


def _i(value: Any) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "—"


def _one(frame: pd.DataFrame, **filters: Any) -> pd.Series | None:
    selected = frame
    for column, value in filters.items():
        if column not in selected:
            return None
        selected = selected.loc[selected[column].astype(str).eq(str(value))]
    if selected.empty:
        return None
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{value}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _effect_svg(rows: list[pd.Series], *, label_key: str, title: str) -> str:
    if not rows:
        return '<div class="empty">尚无可绘制结果。</div>'
    width, left, right = 880, 250, 36
    plot_width = width - left - right
    row_height = 44
    height = 55 + row_height * len(rows)
    low = min(-0.05, *(float(row["ci95_low"]) for row in rows))
    high = max(0.05, *(float(row["ci95_high"]) for row in rows))
    span = max(high - low, 0.05)

    def x(value: float) -> float:
        return left + (value - low) / span * plot_width

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<line x1="{x(0):.1f}" y1="28" x2="{x(0):.1f}" y2="{height-20}" class="zero"/>',
    ]
    for index, row in enumerate(rows):
        y = 48 + index * row_height
        label = GRAMMAR_LABELS.get(str(row[label_key]), str(row[label_key]))
        mean = float(row["mean"])
        ci_low = float(row["ci95_low"])
        ci_high = float(row["ci95_high"])
        parts.extend(
            [
                f'<text x="8" y="{y+5}" class="axis-label">{html.escape(label)}</text>',
                f'<line x1="{x(ci_low):.1f}" y1="{y}" x2="{x(ci_high):.1f}" y2="{y}" class="ci"/>',
                f'<circle cx="{x(mean):.1f}" cy="{y}" r="6" class="point"/>',
                f'<text x="{min(width-60, x(ci_high)+9):.1f}" y="{y+5}" class="value-label">{_pct(mean)}</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def _dose_svg(rows: list[pd.Series]) -> str:
    if not rows:
        return '<div class="empty">剂量网格尚未完成。</div>'
    rows = sorted(rows, key=lambda row: int(row["registered_bank_size"]))
    width, height, left, right, top, bottom = 880, 320, 70, 35, 32, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    ks = [int(row["registered_bank_size"]) for row in rows]
    low = min(-0.05, *(float(row["ci95_low"]) for row in rows))
    high = max(0.10, *(float(row["ci95_high"]) for row in rows))
    pad = 0.08 * (high - low)
    low, high = low - pad, high + pad

    def x(index: int) -> float:
        return left + index * plot_width / max(1, len(rows) - 1)

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="K dose response">',
        f'<line x1="{left}" y1="{y(0):.1f}" x2="{width-right}" y2="{y(0):.1f}" class="zero"/>',
    ]
    points = []
    for index, row in enumerate(rows):
        cx = x(index)
        mean = float(row["mean"])
        points.append(f"{cx:.1f},{y(mean):.1f}")
        parts.extend(
            [
                f'<line x1="{cx:.1f}" y1="{y(float(row["ci95_low"])):.1f}" x2="{cx:.1f}" y2="{y(float(row["ci95_high"])):.1f}" class="ci"/>',
                f'<text x="{cx:.1f}" y="{height-23}" text-anchor="middle" class="axis-label">K{ks[index]}</text>',
                f'<text x="{cx:.1f}" y="{y(mean)-12:.1f}" text-anchor="middle" class="value-label">{_pct(mean)}</text>',
            ]
        )
    parts.append(f'<polyline points="{" ".join(points)}" class="dose-line"/>')
    for index, row in enumerate(rows):
        parts.append(f'<circle cx="{x(index):.1f}" cy="{y(float(row["mean"])):.1f}" r="6" class="point"/>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--selection-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.analysis / "analysis_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(args.selection_config.read_text(encoding="utf-8"))
    model_label = str(selection["model_label"])
    primary_k = int(selection["development_selection"]["primary_bank_size"])
    dose_grid = [
        int(value)
        for value in selection["development_selection"]["registered_nested_dose_grid"]
    ]
    if manifest["analysis_status"] != "complete" and not args.allow_provisional:
        raise ValueError("Refusing to build a final report from provisional analysis")
    estimands = pd.read_csv(args.analysis / "estimands.csv")
    arms = pd.read_csv(args.analysis / "raw_arm_rates.csv")
    flow = pd.read_csv(args.analysis / "sample_flow.csv")
    failure_modes = pd.read_csv(args.analysis / "failure_modes.csv")
    count_estimands_path = args.analysis / "count_estimands.csv"
    count_estimands = (
        pd.read_csv(count_estimands_path)
        if count_estimands_path.exists()
        else pd.DataFrame()
    )

    primary = _one(
        estimands,
        registered_bank_size=primary_k,
        evaluation_scope="confirmation",
        analysis_population="all_examples",
        grammar_class="pooled",
    )
    displayed_scope = "confirmation"
    for fallback_scope in ("full_panel", "discovery"):
        if primary is not None:
            break
        primary = _one(
            estimands,
            registered_bank_size=primary_k,
            evaluation_scope=fallback_scope,
            analysis_population="all_examples",
            grammar_class="pooled",
        )
        displayed_scope = f"{fallback_scope} provisional"
    if primary is None:
        raise ValueError(f"No K{primary_k} pooled result is available")

    raw_rows = []
    for arm in ARM_LABELS:
        row = _one(
            arms,
            registered_bank_size=primary_k,
            evaluation_scope=displayed_scope.split()[0],
            analysis_population="all_examples",
            grammar_class="pooled",
            arm=arm,
        )
        if row is not None:
            raw_rows.append(
                (
                    html.escape(ARM_LABELS[arm]),
                    _pct(row["mean"]),
                    f'{_pct(row["ci95_low"])} – {_pct(row["ci95_high"])}',
                    _i(row["n_seeds"]),
                    _i(row["n_anchor_units"]),
                )
            )

    dose_rows = [
        row
        for _, row in estimands.loc[
            estimands["evaluation_scope"].eq("confirmation")
            & estimands["analysis_population"].eq("all_examples")
            & estimands["grammar_class"].eq("pooled")
        ].iterrows()
    ]
    observed_dose_grid = sorted(
        {int(row["registered_bank_size"]) for row in dose_rows}
    )
    dose_grid_complete = observed_dose_grid == sorted(dose_grid)
    dose_table_rows = [
        (
            f'K{int(row["registered_bank_size"])}',
            _pct(row["mean"]),
            f'{_pct(row["ci95_low"])} – {_pct(row["ci95_high"])}',
            _p(row["sign_flip_p"]),
            _p(row["holm_p"]),
            _i(row["n_seeds"]),
            _i(row["n_anchor_units"]),
        )
        for row in sorted(dose_rows, key=lambda value: int(value["registered_bank_size"]))
    ]

    grammar_rows = [
        row
        for _, row in estimands.loc[
            estimands["registered_bank_size"].eq(primary_k)
            & estimands["evaluation_scope"].eq("confirmation")
            & estimands["analysis_population"].eq("all_examples")
            & ~estimands["grammar_class"].eq("pooled")
        ].iterrows()
    ]
    grammar_table_rows = [
        (
            html.escape(GRAMMAR_LABELS.get(str(row["grammar_class"]), str(row["grammar_class"]))),
            _pct(row["mean"]),
            f'{_pct(row["ci95_low"])} – {_pct(row["ci95_high"])}',
            _p(row["sign_flip_p"]),
            _p(row["holm_p"]),
            _i(row["n_seeds"]),
            _i(row["n_anchor_units"]),
        )
        for row in grammar_rows
    ]

    clean_correct = _one(
        estimands,
        registered_bank_size=primary_k,
        evaluation_scope=displayed_scope.split()[0],
        analysis_population="clean_correct_only",
        grammar_class="pooled",
    )
    count_rows = []
    if not count_estimands.empty:
        selected_counts = count_estimands.loc[
            count_estimands["registered_bank_size"].eq(primary_k)
            & count_estimands["evaluation_scope"].eq("confirmation")
            & count_estimands["analysis_population"].eq("all_examples")
        ]
        count_rows = [
            (
                f'N={int(row["gold_count"])}',
                _pct(row["mean"]),
                f'{_pct(row["ci95_low"])} – {_pct(row["ci95_high"])}',
                _i(row["n_seeds"]),
                _i(row["n_anchor_units"]),
            )
            for _, row in selected_counts.sort_values("gold_count").iterrows()
        ]

    panel_flow = flow.loc[flow["stage"].eq("registered_prompt_panel")]
    flow_rows = [
        (
            html.escape(str(row["evaluation_scope"])),
            html.escape(str(row["gold_count"])),
            _i(row["n_prompt_units"]),
            _i(row["n_eligible_anchor_units"]),
            _i(row["n_ineligible_prompt_units"]),
        )
        for _, row in panel_flow.iterrows()
    ]

    mode_rows = []
    mode_scope = "confirmation" if displayed_scope == "confirmation" else "full_panel"
    selected_modes = failure_modes.loc[
        failure_modes["registered_bank_size"].eq(primary_k)
        & failure_modes["evaluation_scope"].eq(mode_scope)
        & failure_modes["arm"].isin(["selected_bank", "layer_matched_random_pooled_repeats"])
    ]
    for _, row in selected_modes.sort_values(
        ["arm", "grammar_class", "behavior_outcome"]
    ).iterrows():
        mode_rows.append(
            (
                html.escape(ARM_LABELS.get(str(row["arm"]), str(row["arm"]))),
                html.escape(GRAMMAR_LABELS.get(str(row["grammar_class"]), str(row["grammar_class"]))),
                html.escape(str(row["behavior_outcome"])),
                _i(row["n_trials"]),
                _pct(row.get("fraction_within_arm_grammar")),
            )
        )

    status_label = (
        "FINAL / COMPLETE"
        if manifest["analysis_status"] == "complete" and dose_grid_complete
        else "PROVISIONAL / DOSE GRID RUNNING"
    )
    status_class = (
        "ok"
        if manifest["analysis_status"] == "complete" and dose_grid_complete
        else "warn"
    )
    ranking = selection["development_selection"]
    routing = selection["routing"]
    selection_seed_count = len(ranking["ranking_scope_seed_coverage"])
    missing_seed_count = len(ranking["ranking_scope_missing_discovery_seeds"])
    if model_label == "Qwen3-8B":
        route_note = (
            "rank-before-city grammar 在 post_marker 开始干预；rank-after-city "
            "与 marker-neutral grammar 在 p0_item_end 开始干预。"
        )
    else:
        route_note = (
            "全部可注册 grammar 都在 p0_item_end 开始干预；该位置是 item k "
            "已经提交、item k+1 尚未生成的最早 event-specific 边界。"
        )
    css = """
:root{--ink:#18252d;--muted:#617079;--line:#d8e0df;--paper:#f3f5f2;--card:#fbfcfa;--accent:#176d68;--accent-soft:#e7f1ee;--amber:#a86718;--amber-soft:#f8edda;--red:#9c463e}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.64 "Segoe UI Variable","Aptos","Noto Sans SC",system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:34px 28px 84px}header{padding:36px 40px;background:#20383a;color:#f8fbfa;border:1px solid #29484a;border-radius:14px;box-shadow:0 18px 45px #263c3920}h1{font-size:38px;line-height:1.14;letter-spacing:-.035em;margin:8px 0 14px}h2{font-size:27px;line-height:1.2;letter-spacing:-.025em;margin:0 0 18px}h3{margin:27px 0 10px}.eyebrow{letter-spacing:.12em;font-weight:800;font-size:12px;color:#b7d3ce}.lead{font-size:18px;max-width:900px}.meta{color:#c4d3d0}.status{display:inline-block;margin-top:8px;padding:5px 10px;border-radius:999px;font-weight:800;font-size:12px}.status.ok{background:#d7efe6;color:#145d4e}.status.warn{background:#f8e8c8;color:#765012}section{margin-top:22px;padding:29px 32px;background:var(--card);border:1px solid var(--line);border-radius:12px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.card{padding:18px;border-right:1px solid var(--line)}.card:last-child{border-right:0}.card strong{display:block;font-size:27px;color:var(--accent)}.card span{color:var(--muted);font-size:13px}.formula{padding:17px 20px;background:var(--accent-soft);border-left:4px solid var(--accent);border-radius:6px}.callout{padding:16px 20px;background:var(--amber-soft);border-left:4px solid var(--amber);border-radius:6px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#edf1ef;font-weight:800}tr:last-child td{border-bottom:0}figure{margin:20px 0;padding:15px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:#f8faf7}figcaption{color:var(--muted);font-size:13px;margin-top:9px}.zero{stroke:#9aa7a5;stroke-width:1;stroke-dasharray:4 4}.ci{stroke:#4f7775;stroke-width:4;stroke-linecap:round}.point{fill:var(--amber);stroke:white;stroke-width:2}.dose-line{fill:none;stroke:var(--accent);stroke-width:3}.axis-label{fill:var(--ink);font-size:13px}.value-label{fill:var(--muted);font-size:12px;font-weight:700}.audit{font-family:"Cascadia Mono","SFMono-Regular",Consolas,monospace;font-size:12px;word-break:break-all}.empty{color:var(--muted);padding:24px;text-align:center}code{font-family:"Cascadia Mono",Consolas,monospace;background:#edf1ef;padding:.08em .32em;border-radius:4px}@media(max-width:800px){main{padding:14px}.cards{grid-template-columns:1fr 1fr}.card:nth-child(2){border-right:0}section,header{padding:22px}h1{font-size:30px}}"""

    primary_ci = f'{_pct(primary["ci95_low"])} – {_pct(primary["ci95_high"])}'
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V5 Native Thinking · Targeted Retrieval</title><style>{css}</style></head><body><main>
<header><div class="eyebrow">REALISTIC NIAH · NATIVE THINKING V5 · TARGETED RETRIEVAL</div><h1>Grammar-routed targeted retrieval causal report</h1><p class="lead">在每次第 k→k+1 个 needle 的冻结 retrieval 位点，持续关闭 discovery-only attention-head bank，并与三组逐层数量完全匹配的随机 head bank 比较。</p><div class="status {status_class}">{status_label}</div><p class="meta">{html.escape(model_label)} · old 300-prompt panel · discovery 1234–1253 · registered confirmation 1254–1263 · N=1…10</p></header>
<section><h2>1 · 结论先行</h2><div class="cards"><div class="card"><span>K{primary_k} selected−random failure</span><strong>{_pct(primary['mean'])}</strong><span>95% CI {primary_ci}</span></div><div class="card"><span>Seed sign-flip / Holm</span><strong>{_p(primary['sign_flip_p'])}</strong><span>Holm {_p(primary['holm_p'])}</span></div><div class="card"><span>支持规模</span><strong>{_i(primary['n_anchor_units'])}</strong><span>{_i(primary['n_seeds'])} seeds · {html.escape(displayed_scope)}</span></div><div class="card"><span>Clean-correct sensitivity</span><strong>{_pct(clean_correct['mean']) if clean_correct is not None else '—'}</strong><span>同一 contrast，先筛 clean-correct</span></div></div><div class="callout"><strong>解释边界。</strong>主效应是 selected bank 比同层随机 bank 多造成多少“未取回注册 next needle”的失败；它证明这组 heads 对当前 retrieval transition 的集合必要性，不证明单个 head 独立编码一个离散计数器。registered confirmation 与 discovery 在正式 head 排名上严格分离，但早期 parser/site smoke 曾接触部分旧 prompts，因此不称 pristine held-out。</div></section>
<section><h2>2 · 冻结设计与 estimand</h2><p>Head 排名只读取 discovery source writes；实际覆盖 {selection_seed_count} 个 discovery seeds，另有 {missing_seed_count} 个 seed 因没有符合 selection grammar 的 transition 而不进入排名。主 metric 是 <code>{html.escape(str(ranking['head_ranking_metric']))}</code>，primary bank 为 K{primary_k}，并跨注册 grammar 复用。{html.escape(route_note)}</p><div class="formula"><strong>Primary anchor effect</strong><br>Δᵢ = 𝟙[selected 未正确取回 next needle] − (1/3)Σᵣ𝟙[randomᵣ 未正确取回 next needle]<br>先在同一 seed 内平均 prompts，再让 seeds 等权；CI 为 seed-cluster bootstrap，p 为 two-sided seed sign-flip。</div><h3>300-prompt sample flow</h3>{_table(['Scope','N','Prompt units','Eligible anchors','Not applicable'], flow_rows)}</section>
<section><h2>3 · Raw arms 先于 contrast</h2>{_table(['Arm','Seed-equal failure','95% CI','Seeds','Anchors'], raw_rows)}<p>Random mean 在每个 prompt-anchor 内先平均三个预冻结重复，再进入 seed 聚合；因此不会把 3× random 当作三倍独立样本。</p></section>
<section><h2>4 · Registered-confirmation dose response</h2><figure>{_dose_svg(dose_rows)}<figcaption>同一 discovery-only 排名的严格嵌套 {html.escape(', '.join(f'K{k}' for k in dose_grid))}。K{primary_k} clean arm 被所有次级 K 复用；每个 K 的 selected 与三组 matched random 均重新执行。</figcaption></figure>{_table(['K','Selected−random failure','95% CI','Raw p','Holm p','Seeds','Anchors'], dose_table_rows)}</section>
<section><h2>5 · 一个统一 bank 跨 grammar 是否成立？</h2><figure>{_effect_svg(grammar_rows, label_key='grammar_class', title=f'K{primary_k} grammar effects')}<figcaption>Natural pooled 保留真实 grammar 频率；macro 只对预定义且有支持的 primary grammar 等权。稀有 grammar 支持量不足时只作描述，不升级为稳定机制结论。</figcaption></figure>{_table(['Grammar / summary','Effect','95% CI','Raw p','Holm p','Seeds','Anchors'], grammar_table_rows)}</section>
<section><h2>6 · N=1…10 与失败形态</h2><h3>按 gold count</h3>{_table(['Count','Effect','95% CI','Seeds','Anchors'], count_rows)}<h3>Exclusive behavior outcomes</h3>{_table(['Arm','Grammar','Outcome','Trials','Within arm×grammar'], mode_rows)}</section>
<section><h2>7 · 审计、边界与可复现性</h2><p class="audit">Selection schema: {html.escape(selection['schema_version'])}<br>Selection status: {html.escape(selection['selection_status'])}<br>Model: {html.escape(model_label)}<br>Discovery-only K{primary_k} bank: {html.escape(ranking['primary_bank_sha256'])}<br>Anchor registry: {html.escape(manifest['clean_reference_anchor_registry_sha256'])}<br>Routing policy: {html.escape(routing['policy_id'])}<br>Routing SHA256: {html.escape(routing['sha256'])}<br>Analysis manifest SHA256: {_sha256(args.analysis / 'analysis_manifest.json')}<br>Selection config SHA256: {_sha256(args.selection_config)}</p><p>正式分析器逐 K 核验 plan、selected-bank、routing 与 registry hashes。Representation geometry 不进入本 causal estimand；parser 与 token-site 的构造规则由独立技术报告说明。</p></section>
</main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": _sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
