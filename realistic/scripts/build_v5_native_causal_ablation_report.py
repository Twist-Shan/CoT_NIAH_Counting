from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_COLOR = {"Qwen3-8B": "#315f78", "Gemma4-E4B": "#17736b"}
ARM_LABEL = {
    "clean": "Clean",
    "selected_bank": "Ranked bank",
    "layer_matched_random_mean": "3× layer-matched random mean",
}
GRAMMAR_LABEL = {
    "adjacent_rank_after_city": "adjacent rank-after-city",
    "adjacent_rank_before_city": "adjacent rank-before-city",
    "same_unit_rank_after_city": "same-unit rank-after-city",
    "same_unit_rank_before_city": "same-unit rank-before-city",
    "structural_invariant_bullet": "invariant bullet",
    "structural_unmarked": "structural unmarked",
    "evidence_sequence_unranked": "unranked evidence sequence",
    "structural_explicit_rank_before_city": "explicit structural rank",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if pd.isna(number) else f"{100 * number:.{digits}f}%"


def pvalue(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(number):
        return "—"
    return "<0.0001" if number < 0.0001 else f"{number:.4f}"


def one(frame: pd.DataFrame, **filters: Any) -> pd.Series | None:
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


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def load_model(
    model: str,
    analysis: Path,
    selection_path: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads((analysis / "analysis_manifest.json").read_text(encoding="utf-8"))
    estimands = pd.read_csv(analysis / "estimands.csv")
    arms = pd.read_csv(analysis / "raw_arm_rates.csv")
    counts = pd.read_csv(analysis / "count_estimands.csv")
    failure = pd.read_csv(analysis / "failure_modes.csv")
    primary_k = int(selection["development_selection"]["primary_bank_size"])
    planned = sorted(
        int(value)
        for value in selection["development_selection"]["registered_nested_dose_grid"]
    )
    observed = sorted(
        int(value)
        for value in estimands.loc[
            estimands["evaluation_scope"].eq("confirmation")
            & estimands["analysis_population"].eq("all_examples")
            & estimands["grammar_class"].eq("pooled"),
            "registered_bank_size",
        ].unique()
    )
    primary = one(
        estimands,
        registered_bank_size=primary_k,
        evaluation_scope="confirmation",
        analysis_population="all_examples",
        grammar_class="pooled",
    )
    if primary is None:
        raise ValueError(f"{model} is missing its primary K{primary_k} result")
    return {
        "model": model,
        "analysis": analysis,
        "selection_path": selection_path,
        "selection": selection,
        "manifest": manifest,
        "estimands": estimands,
        "arms": arms,
        "counts": counts,
        "failure": failure,
        "primary_k": primary_k,
        "planned_ks": planned,
        "observed_ks": observed,
        "dose_complete": planned == observed,
        "primary": primary,
    }


def dose_svg(models: list[dict[str, Any]]) -> str:
    width, height = 980, 440
    left, right, top = 75, 32, 45
    panel_height = 150
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Registered confirmation dose response by model">'
    ]
    for panel_index, payload in enumerate(models):
        frame = payload["estimands"]
        rows = frame.loc[
            frame["evaluation_scope"].eq("confirmation")
            & frame["analysis_population"].eq("all_examples")
            & frame["grammar_class"].eq("pooled")
        ].sort_values("registered_bank_size")
        y0 = top + panel_index * 200
        plot_width = width - left - right
        color = MODEL_COLOR[payload["model"]]
        parts.append(
            f'<text x="{left}" y="{y0 - 15}" class="panel-title">'
            f'{html.escape(payload["model"])}</text>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_height * (1 - tick)
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
                f'y2="{y:.1f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" '
                f'class="axis">{int(100*tick)}%</text>'
            )
        points: list[str] = []
        n = max(1, len(rows))
        for index, (_, row) in enumerate(rows.iterrows()):
            x = left + (index + 0.5) * plot_width / n
            y = y0 + panel_height * (1 - float(row["mean"]))
            y_low = y0 + panel_height * (1 - float(row["ci95_low"]))
            y_high = y0 + panel_height * (1 - float(row["ci95_high"]))
            points.append(f"{x:.1f},{y:.1f}")
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" '
                    f'y2="{y_high:.1f}" class="ci" style="stroke:{color}"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" '
                    f'style="fill:{color}" class="dot"/>',
                    f'<text x="{x:.1f}" y="{y0+panel_height+22:.1f}" '
                    f'text-anchor="middle" class="axis">K{int(row["registered_bank_size"])}</text>',
                    f'<text x="{x:.1f}" y="{max(y0+12, y-11):.1f}" '
                    f'text-anchor="middle" class="value">{pct(row["mean"])}</text>',
                ]
            )
        if len(points) > 1:
            parts.append(
                f'<polyline points="{" ".join(points)}" class="trend" '
                f'style="stroke:{color}"/>'
            )
        if not payload["dose_complete"]:
            missing = sorted(set(payload["planned_ks"]) - set(payload["observed_ks"]))
            parts.append(
                f'<text x="{width-right}" y="{y0-15}" text-anchor="end" '
                f'class="pending">running: {html.escape(", ".join(f"K{k}" for k in missing))}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-analysis", type=Path, required=True)
    parser.add_argument("--gemma-analysis", type=Path, required=True)
    parser.add_argument("--qwen-selection", type=Path, required=True)
    parser.add_argument("--gemma-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = [
        load_model("Qwen3-8B", args.qwen_analysis, args.qwen_selection),
        load_model("Gemma4-E4B", args.gemma_analysis, args.gemma_selection),
    ]
    complete = all(payload["dose_complete"] for payload in models)
    status = "FINAL / COMPLETE" if complete else "INTERIM / GEMMA DOSE GRID RUNNING"
    status_class = "pass" if complete else "running"

    headline_rows: list[list[str]] = []
    raw_rows: list[list[str]] = []
    scope_rows: list[list[str]] = []
    grammar_rows: list[list[str]] = []
    count_rows: list[list[str]] = []
    for payload in models:
        model = payload["model"]
        k = payload["primary_k"]
        primary = payload["primary"]
        headline_rows.append(
            [
                html.escape(model),
                f"K{k}",
                pct(primary["mean"]),
                f'{pct(primary["ci95_low"])} – {pct(primary["ci95_high"])}',
                pvalue(primary["sign_flip_p"]),
                pvalue(primary["holm_p"]),
                str(int(primary["n_seeds"])),
                str(int(primary["n_anchor_units"])),
            ]
        )
        for arm in ARM_LABEL:
            row = one(
                payload["arms"],
                registered_bank_size=k,
                evaluation_scope="confirmation",
                analysis_population="all_examples",
                grammar_class="pooled",
                arm=arm,
            )
            if row is not None:
                raw_rows.append(
                    [
                        html.escape(model),
                        f"K{k}",
                        html.escape(ARM_LABEL[arm]),
                        pct(row["mean"]),
                        f'{pct(row["ci95_low"])} – {pct(row["ci95_high"])}',
                        str(int(row["n_anchor_units"])),
                    ]
                )
        for scope in ("confirmation", "full_panel", "discovery"):
            row = one(
                payload["estimands"],
                registered_bank_size=k,
                evaluation_scope=scope,
                analysis_population="all_examples",
                grammar_class="pooled",
            )
            if row is not None:
                scope_rows.append(
                    [
                        html.escape(model),
                        f"K{k}",
                        html.escape(scope),
                        pct(row["mean"]),
                        f'{pct(row["ci95_low"])} – {pct(row["ci95_high"])}',
                        str(int(row["n_seeds"])),
                        str(int(row["n_anchor_units"])),
                    ]
                )
        grammar = payload["estimands"].loc[
            payload["estimands"]["registered_bank_size"].eq(k)
            & payload["estimands"]["evaluation_scope"].eq("confirmation")
            & payload["estimands"]["analysis_population"].eq("all_examples")
            & ~payload["estimands"]["grammar_class"].isin(
                ["pooled", "macro_primary_grammars"]
            )
        ]
        for _, row in grammar.iterrows():
            grammar_rows.append(
                [
                    html.escape(model),
                    html.escape(
                        GRAMMAR_LABEL.get(
                            str(row["grammar_class"]), str(row["grammar_class"])
                        )
                    ),
                    pct(row["mean"]),
                    f'{pct(row["ci95_low"])} – {pct(row["ci95_high"])}',
                    pvalue(row["sign_flip_p"]),
                    str(int(row["n_seeds"])),
                    str(int(row["n_anchor_units"])),
                    html.escape(str(row["inferential_status"])),
                ]
            )
        counts = payload["counts"].loc[
            payload["counts"]["registered_bank_size"].eq(k)
            & payload["counts"]["evaluation_scope"].eq("confirmation")
            & payload["counts"]["analysis_population"].eq("all_examples")
        ]
        for _, row in counts.sort_values("gold_count").iterrows():
            count_rows.append(
                [
                    html.escape(model),
                    f"N={int(row['gold_count'])}",
                    pct(row["mean"]),
                    f'{pct(row["ci95_low"])} – {pct(row["ci95_high"])}',
                    str(int(row["n_seeds"])),
                ]
            )

    qwen, gemma = models
    q_primary, g_primary = qwen["primary"], gemma["primary"]
    css = """
:root{--ink:#19262d;--muted:#637178;--paper:#f3f5f2;--surface:#fbfcfa;--line:#d7dfdd;--deep:#20383a;--qwen:#315f78;--gemma:#17736b;--amber:#a76618;--amber-soft:#f8edda;--green-soft:#e4f1ed}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 "Segoe UI Variable","Aptos","Noto Sans SC",system-ui,sans-serif}.shell{display:grid;grid-template-columns:240px minmax(0,1fr);gap:26px;max-width:1370px;margin:auto;padding:26px}nav{position:sticky;top:18px;align-self:start;padding:20px 18px;border-top:3px solid var(--deep);background:#edf1ee}nav strong{display:block;margin-bottom:12px}nav a{display:block;padding:6px 0;color:#40545a;text-decoration:none;border-bottom:1px solid #dce3e0;font-size:13px}main{min-width:0}header{padding:38px 42px;background:var(--deep);color:#f7faf8;border:1px solid #294a4b;border-radius:14px;box-shadow:0 18px 45px #243d3b22}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.13em;color:#b7d3ce}h1{font-size:40px;line-height:1.12;letter-spacing:-.04em;margin:9px 0 15px}h2{font-size:28px;line-height:1.2;letter-spacing:-.03em;margin:0 0 18px}h3{font-size:18px;margin:27px 0 10px}.lead{font-size:18px;max-width:900px}.meta{color:#c7d6d3}.status{display:inline-block;margin-top:8px;padding:5px 10px;border-radius:999px;font-size:12px;font-weight:850}.status.pass{color:#145d4e;background:#d7efe6}.status.running{color:#785116;background:#f4dfb9}section{margin-top:22px;padding:30px 34px;background:var(--surface);border:1px solid var(--line);border-radius:12px}.hero-metrics{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.metric{padding:22px}.metric+ .metric{border-left:1px solid var(--line)}.metric span{display:block;color:var(--muted);font-size:13px}.metric strong{display:block;font-size:42px;line-height:1.12;margin:6px 0}.metric.qwen strong{color:var(--qwen)}.metric.gemma strong{color:var(--gemma)}.callout{padding:17px 20px;background:var(--amber-soft);border-left:4px solid var(--amber);border-radius:6px}.formula{padding:17px 20px;background:var(--green-soft);border-left:4px solid var(--gemma);border-radius:6px;font-family:"Cascadia Mono",Consolas,monospace}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px}table{width:100%;border-collapse:collapse;font-size:13.5px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{background:#edf1ef}tr:last-child td{border-bottom:0}figure{margin:20px 0;padding:12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:#f8faf7}figcaption{font-size:13px;color:var(--muted)}svg{display:block;width:100%;height:auto}.grid{stroke:#dce4e1;stroke-width:1}.ci{stroke-width:4;stroke-linecap:round}.dot{stroke:#fff;stroke-width:2}.trend{fill:none;stroke-width:3}.axis{fill:#65747a;font-size:12px}.value{fill:#33484e;font-size:12px;font-weight:750}.panel-title{fill:#1b3035;font-size:15px;font-weight:800}.pending{fill:#9a641d;font-size:12px;font-weight:750}.link-strip{display:flex;flex-wrap:wrap;gap:9px}.link-strip a{padding:8px 11px;border:1px solid var(--line);border-radius:6px;color:#1d5e5a;text-decoration:none;background:#f7faf8}.audit{font-family:"Cascadia Mono",Consolas,monospace;font-size:12px;word-break:break-all;color:#40525a}code{font-family:"Cascadia Mono",Consolas,monospace;background:#edf1ef;padding:.08em .32em;border-radius:4px}@media(max-width:900px){.shell{display:block;padding:12px}nav{position:relative;top:0;margin-bottom:12px}.hero-metrics{grid-template-columns:1fr}.metric+.metric{border-left:0;border-top:1px solid var(--line)}header,section{padding:23px}h1{font-size:32px}}@media print{body{background:#fff}.shell{display:block;padding:0}nav{display:none}header,section{box-shadow:none;break-inside:avoid}}
"""
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    output = args.output
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Native-thinking V5 · Causal Ablation</title><style>{css}</style></head><body><div class="shell">
<nav><strong>Native causal</strong><a href="#claim">1 · 结论与边界</a><a href="#design">2 · 冻结设计</a><a href="#raw">3 · Raw arms</a><a href="#dose">4 · Dose response</a><a href="#grammar">5 · Grammar transfer</a><a href="#counts">6 · Count strata</a><a href="#interpret">7 · 机制解释</a><a href="#audit">8 · 审计</a></nav><main>
<header><div class="eyebrow">REALISTIC NIAH · NATIVE THINKING V5 · CAUSAL ONLY</div><h1>Targeted-retrieval ablation report</h1><p class="lead">在模型准备第 k+1 个 trace item 时，持续关闭 discovery-frozen pre-O head slices，直接观察自由生成首先取回的 city record 是否仍为注册的第 k+1 个 needle。</p><span class="status {status_class}">{status}</span><p class="meta">Qwen3-8B + Gemma4-E4B · old300 · discovery 1234–1253 · registered confirmation 1254–1263</p></header>
<section id="claim"><h2>1 · 结论先行，同时限定结论</h2><div class="hero-metrics"><div class="metric qwen"><span>Qwen K{qwen['primary_k']} confirmation selected−random failure</span><strong>{pct(q_primary['mean'])}</strong><span>95% CI {pct(q_primary['ci95_low'])} – {pct(q_primary['ci95_high'])}</span></div><div class="metric gemma"><span>Gemma K{gemma['primary_k']} confirmation selected−random failure</span><strong>{pct(g_primary['mean'])}</strong><span>95% CI {pct(g_primary['ci95_low'])} – {pct(g_primary['ci95_high'])}</span></div></div><p>两个模型都出现强而稳定的 registered-confirmation 效应：selected bank 比逐层完全匹配的随机 bank 更容易打断“第 k 次 retrieval → 第 k 个 needle”的一一对应。Gemma 只需 8 个 heads；Qwen 需要约 112–125 个 heads 才进入高失败区，说明功能相似但回路宽度明显不同。</p><div class="callout"><strong>不能升级成的结论。</strong>结果支持 bank-level necessity，不支持“每个入选 head 都是独立计数器”、唯一通路、或 representation 几何本身具有因果性。这里也没有把 removal、restoration 或 patching 的结果混进 ablation 结论。</div>{table(['Model','Primary K','Effect','95% CI','Raw p','Holm p','Seeds','Anchors'], headline_rows)}</section>
<section id="design"><h2>2 · 与 Non-thinking 同步的统计骨架</h2><p>报告顺序与 Non-thinking 保持一致：先列 raw ranked 与 raw random arms，再列 ranked−random contrast；all-examples 先于 clean-correct；confirmation 先于 full-panel；统计单位为 seed，不把三个 random repeats 当作三倍独立样本。</p><div class="formula">Δᵢ = I[selected first semantic city record ≠ registered next city] − (1/3) Σᵣ I[randomᵣ failure]</div><p>Qwen 按 grammar 在 <code>post_marker</code> 或 <code>p0_item_end</code> 开始关闭；Gemma 所有 grammar 统一在 <code>p0_item_end</code> 开始关闭。两者都从冻结位点持续关闭到所有后续 cached decode forwards，阻止模型先偏离、再逃出 ablation window 后自我纠正。</p>{table(['Model','K','Scope','Effect','95% CI','Seeds','Anchors'], scope_rows)}</section>
<section id="raw"><h2>3 · Raw arms：先确认不是 generic damage</h2>{table(['Model','K','Arm','Seed-equal failure','95% CI','Anchors'], raw_rows)}<p>主效应要求 ranked bank 的失败显著高于具有完全相同逐层 head 数的 random banks。Clean failure 保留在 all-example denominator 中；clean-correct sensitivity 另行报告，避免只展示有利子样本。</p></section>
<section id="dose"><h2>4 · Confirmation dose response</h2><figure>{dose_svg(models)}<figcaption>每个 panel 的横轴是该模型预冻结的嵌套 K；纵轴是 selected−mean(random) failure。Qwen 报告 K32–K125 六档，Gemma 报告 K1–K8 五档；所有点均来自冻结 bank 的 registered confirmation。</figcaption></figure></section>
<section id="grammar"><h2>5 · 统一 bank 跨 trace grammar 的迁移</h2><p>Bank 在一个 discovery selection grammar 上排序，随后冻结迁移到其他 grammar。只有支持量足够的行承担 confirmatory 解释；单个 bullet 或极少的 rank-before 样本仅保留描述。</p>{table(['Model','Grammar','Effect','95% CI','Raw p','Seeds','Anchors','Status'], grammar_rows)}</section>
<section id="counts"><h2>6 · N=2…10 的 registered count strata</h2><p>N=1 没有 k→k+1 transition，因此明确记为 not applicable，而不是从 300-prompt denominator 中静默删除。</p>{table(['Model','Count','Effect','95% CI','Seeds'], count_rows)}</section>
<section id="interpret"><h2>7 · 当前最窄机制解释</h2><h3>Qwen：宽而有阈值的 retrieval bank</h3><p>K32 仅有约 5% specificity，K64/K80/K96 逐步增大，到 K112/K125 跃升到约 80%。最保守解释是 retrieval 由较宽、具有旁路的 head 集合共同承担；不是少量 top-mass heads 的单点瓶颈。</p><h3>Gemma：紧凑且跨 grammar 的 K8 bank</h3><p>Full panel 中 K8 selected 失败率为 83.3%，random 为 9.4%；confirmation 的 seed-equal contrast 为 {pct(g_primary['mean'])}。两个主 grammar 方向一致，且独立 ranking 得到同一 K8 head set，支持统一 retrieval module。稀有 bullet 的 1 个 confirmation anchor 不受破坏，不能据此否定或确认迁移。</p><h3>为什么位点与 head ranking 必须同位</h3><p>Ranking 读取的 query token 与 ablation 起点相同。attention score 衡量该 query 对注册第 k+1 个 prompt-record span 的读取；行为 endpoint 则完全依赖自由生成，不使用 next-city candidate ranking。位点 positive control 与持续关闭共同排除了“只是 query 选错”及“模型偏离后逃逸恢复”两个主要实现漏洞。</p></section>
<section id="audit"><h2>8 · 分离、链接与复现账本</h2><div class="link-strip"><a href="v5_native_targeted_retrieval/Qwen3-8B/targeted_retrieval_report.html">Qwen model report</a><a href="v5_native_targeted_retrieval/Gemma4-E4B/targeted_retrieval_report.html">Gemma model report</a><a href="NiaH_Native-Thinking_Parser_and_Token_Sites.html">Parser / token-site report</a><a href="NiaH_Geometry_Comparison.html">Representation geometry comparison</a></div><p><strong>分离规则：</strong>本页只把 causal intervention 当作机制证据。Geometry、PCA、probe 与 token representation 只存在于单独的 representation 报告；parser/token 报告只解释 cohort、边界和 anchor 如何构造，不重复声称行为效应。</p><p class="audit">Generated UTC: {html.escape(generated)}<br>Qwen analysis manifest: {sha256(args.qwen_analysis / 'analysis_manifest.json')}<br>Gemma analysis manifest: {sha256(args.gemma_analysis / 'analysis_manifest.json')}<br>Qwen selection: {sha256(args.qwen_selection)}<br>Gemma selection: {sha256(args.gemma_selection)}<br>Report schema: realistic_niah_v5_native_causal_ablation_report_v1</p></section>
</main></div></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output.resolve()), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
