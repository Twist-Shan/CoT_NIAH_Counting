from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from realistic_niah_v4_4_3.io import atomic_text, stage_root

from .relay_spec import V444RelayConfig


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _discovery_svg(frame: pd.DataFrame, selected: str) -> str:
    rows = frame.sort_values("mean_count_slope", ascending=False).reset_index(drop=True)
    width, left, right, top, row_h = 900, 245, 45, 34, 34
    height = top + row_h * len(rows) + 48
    values = rows["mean_count_slope"].to_numpy(float)
    limit = max(float(abs(values).max()), 1e-8)
    center = left + (width - left - right) / 2
    scale = (width - left - right) / (2 * limit)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Discovery relay position-set slopes">',
        f'<line x1="{center:.1f}" y1="20" x2="{center:.1f}" y2="{height-28}" class="zero"/>',
    ]
    for index, row in rows.iterrows():
        y = top + index * row_h
        value = float(row["mean_count_slope"])
        x = center + value * scale
        color = "#c05b34" if row["position_set"] == selected else (
            "#247a78" if row["set_role"] == "relay_candidate" else "#899397"
        )
        parts.append(
            f'<text x="{left-10}" y="{y+5}" text-anchor="end">{html.escape(str(row["position_set"]))}</text>'
        )
        parts.append(
            f'<line x1="{center:.1f}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{color}" stroke-width="8"/>'
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y}" r="5" fill="{color}"/><text x="{x + (8 if value >= 0 else -8):.1f}" y="{y+5}" text-anchor="{("start" if value >= 0 else "end")}">{value:.3g}</text>'
        )
    parts.append(
        f'<text x="{center:.1f}" y="{height-8}" text-anchor="middle">mean within-seed slope of contribution coefficient per count</text></svg>'
    )
    return "".join(parts)


def _pvalue_svg(metrics: Sequence[Mapping[str, Any]], alpha: float) -> str:
    width, left, right, top, row_h = 900, 270, 55, 34, 38
    height = top + row_h * len(metrics) + 48
    max_x = max(
        -math.log10(max(float(row["exact_sign_flip_p"]), 1e-12))
        for row in metrics
    )
    max_x = max(max_x, -math.log10(alpha)) * 1.15
    scale = (width - left - right) / max_x
    threshold = left + (-math.log10(alpha)) * scale
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Relay exact sign-flip p-values">',
        f'<line x1="{threshold:.1f}" y1="18" x2="{threshold:.1f}" y2="{height-30}" class="threshold"/>',
    ]
    for index, row in enumerate(metrics):
        y = top + index * row_h
        score = -math.log10(max(float(row["exact_sign_flip_p"]), 1e-12))
        x = left + score * scale
        passed = float(row["exact_sign_flip_p"]) <= alpha
        color = "#247a78" if passed else "#b7a58b"
        parts.append(
            f'<text x="{left-10}" y="{y+5}" text-anchor="end">{html.escape(str(row["metric"]))}</text>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{color}" stroke-width="8"/><circle cx="{x:.1f}" cy="{y}" r="5" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x+8:.1f}" y="{y+5}">p={float(row["exact_sign_flip_p"]):.3g}</text>'
        )
    parts.append(
        f'<text x="{(left+width-right)/2:.1f}" y="{height-8}" text-anchor="middle">−log10(exact sign-flip p); dashed line = α={alpha:g}</text></svg>'
    )
    return "".join(parts)


def _template(
    analysis: Mapping[str, Any], discovery: pd.DataFrame, *, relay_config: V444RelayConfig
) -> str:
    decision = analysis["primary_decision"]
    selection = analysis["selection"]
    metrics = analysis["metric_summary"]
    supported = bool(decision["supported"])
    status = "SUPPORTED" if supported else "NOT SUPPORTED"
    metric_rows = [
        [
            row["metric"],
            row["alternative"],
            _fmt(row["mean"], 6),
            f"[{_fmt(row['ci95_low'], 6)}, {_fmt(row['ci95_high'], 6)}]",
            _fmt(row["positive_seed_fraction"], 4),
            _fmt(row["exact_sign_flip_p"], 6),
        ]
        for row in metrics
    ]
    family_rows = [
        [name, _fmt(value, 6), "pass" if value <= relay_config.primary_alpha else "fail"]
        for name, value in decision["family_p_values"].items()
    ]
    metric_by_name = {row["metric"]: row for row in metrics}
    natural = metric_by_name["natural_relay_slope"]
    first_stage = metric_by_name["edge_patch_first_stage_transport"]
    behavior = metric_by_name["edge_patch_behavior_transport"]
    mediation = metric_by_name["ov_mediation_specificity"]
    removal_error = metric_by_name["relay_removal_error_specificity"]
    removal_margin = metric_by_name["relay_removal_margin_specificity"]
    result_readout = f"""<div class="callout"><strong>分层判读。</strong>
自然 relay carrier 随 count 稳定变化（mean={_fmt(natural['mean'],6)},
p={_fmt(natural['exact_sign_flip_p'],6)}），而 receiver-α/donor-V patch
也确实在冻结 OV 轴上产生有符号的机械 first stage
（mean={_fmt(first_stage['mean'],6)}, p={_fmt(first_stage['exact_sign_flip_p'],6)}）。
但是答案层 transport 的区间跨 0（mean={_fmt(behavior['mean'],6)},
95% CI=[{_fmt(behavior['ci95_low'],6)}, {_fmt(behavior['ci95_high'],6)}],
p={_fmt(behavior['exact_sign_flip_p'],6)}）；natural-axis block 相对同 span
正交控制没有特异中介效应（mean={_fmt(mediation['mean'],6)},
p={_fmt(mediation['exact_sign_flip_p'],6)}）。两项 removal estimand 也都朝注册预测的
反方向（error mean={_fmt(removal_error['mean'],6)},
margin mean={_fmt(removal_margin['mean'],6)}）。因此数据支持“末端 value state
携带 count 且可沿 H16/H19 的 OV 子空间推动”，但不支持“模型自然依赖该
terminal relay→OV channel 产生答案”。</div>"""
    audit_rows = [
        [row["name"], "PASS" if row["passed"] else "FAIL", json.dumps(row["detail"], ensure_ascii=False)]
        for row in analysis["audit"]["checks"]
    ]
    css = """
:root{--ink:#1f2b30;--muted:#66767b;--paper:#f6f3ed;--card:#fff;--line:#d9d6cf;--teal:#247a78;--rust:#c05b34;--soft:#edf3f1}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.66}header{padding:58px 24px 42px;background:linear-gradient(135deg,#143b3a,#263940 62%,#4b352a);color:white}.hero,main{max-width:1120px;margin:auto}.eyebrow{letter-spacing:.14em;text-transform:uppercase;color:#b8ddd8;font-size:.8rem}.hero h1{font-size:clamp(2rem,5vw,4.4rem);line-height:1.04;margin:.4rem 0 1rem}.lede{max-width:900px;color:#e1ecea}.badge{display:inline-block;border:1px solid #ffffff55;border-radius:999px;padding:5px 10px;margin:6px 7px 0 0;font-size:.82rem}main{padding:34px 24px 80px}section{margin:0 0 52px}h2{font-size:1.9rem;margin:0 0 15px}h3{color:#18494a;margin-top:30px}.callout{background:#fff3e8;border-left:4px solid var(--rust);padding:15px 18px;margin:18px 0}.conclusion{border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:13px 0;margin-top:22px;font-weight:650}.formula{background:var(--soft);border:1px solid #d7e1df;padding:12px 15px;font-family:"Cambria Math","Times New Roman",serif;overflow:auto}.table-wrap{overflow:auto;border:1px solid var(--line);background:white;margin:15px 0 22px}table{width:100%;border-collapse:collapse;font-size:.88rem}th,td{padding:9px 11px;border-bottom:1px solid #ece8e0;text-align:left;vertical-align:top}th{background:#eaf0ee}figure{background:white;border:1px solid var(--line);padding:14px;margin:22px 0}svg{display:block;width:100%;height:auto;font-family:inherit;font-size:12px}.zero{stroke:#5b6a6f;stroke-width:1.2}.threshold{stroke:var(--rust);stroke-width:1.4;stroke-dasharray:5 5}figcaption{border-top:1px solid #e6e2db;margin-top:8px;padding-top:10px;color:var(--muted);font-size:.88rem}.status{font-size:1.35rem;font-weight:800;color:#bfe9df}.foot{font-size:.85rem;color:var(--muted)}@media print{body{background:white}header{background:white;color:black;border-bottom:2px solid black}.lede,.status,.eyebrow{color:black}section,figure,.table-wrap{break-inside:avoid}}
"""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4.4.4 relay-to-OV causal supplement</title><style>{css}</style></head><body>
<header><div class="hero"><div class="eyebrow">Realistic NIAH · V4.4.4 · Relay Extension</div><h1>Relay value state → L28 H16/H19 → answer</h1><p class="lede">本补充实验不要求 H16/H19 自己通过 QK 定位原始 needle。它固定 receiver 的 Q、K 与 attention 权重，只替换候选位置集合的 donor V 内容，并检验该效应是否经冻结的自然 OV 轴到达答案。</p><div class="status">{status}</div><span class="badge">selected: {html.escape(selection['selected_position_set'])}</span><span class="badge">n={len(relay_config.confirmation_seeds)} seeds</span><span class="badge">L28 H16/H19</span><span class="badge">global IUT p={_fmt(decision['global_intersection_union_p'],6)}</span></div></header>
<main>
<section><h2>1. 假说与因果边界</h2><div class="formula">R<sub>P</sub> → Σ<sub>j∈P</sub> α<sub>h</sub>(q,j)V(j) → z<sub>28,h</sub> → W<sub>O</sub><sup>h</sup>z<sub>28,h</sub> → answer</div><p>这里的 relay 是位置集合 P 上的 value-state，而不是预设的单 token 或单 head。receiver-α/donor-V edge patch 由 receiver 的自然 α 读取 donor V，因此没有改动 QK routing；H16/H19 共享 GQA KV source，但干预只写入二者各自计算得到的 pre-O z slice，不连带修改同 GQA group 的其他 query heads。</p><div class="conclusion">本节结论：若 edge patch、自然 removal 与 downstream OV block 同时成立，可以支持“模型自然读取 relay 内容并通过该 OV set 影响答案”；它仍不识别建立 relay 的上游 QK heads。</div></section>
<section><h2>2. Relay 位置集合发现</h2><p>Discovery 只使用 {len(relay_config.discovery_seeds)} 个 center seeds 的自然 activation。每个 key position 的纵轴贡献定义为 Σ<sub>h</sub> α<sub>h</sub>(q,j)〈W<sub>O</sub><sup>h</sup>v(j),u<sub>m</sub>〉；图中纵轴是该集合总贡献系数相对 count 的 within-seed slope。橙色为冻结候选，青色为可选 relay 集合，灰色为原始 needle source controls。</p><figure>{_discovery_svg(discovery, selection['selected_position_set'])}<figcaption>横向条从 0 指向 mean count slope；正值表示该位置集合沿冻结 L28 natural-OV 轴提供的贡献随 gold count 增加。集合大小不同，因此这是总运输贡献，不是 per-token attention。</figcaption></figure><div class="conclusion">本节结论：候选选择只依据自然 source contribution，selection qualified={str(selection['selection_qualified']).lower()}；没有读取 confirmation 的行为干预结果。</div></section>
<section><h2>3. 确认性因果结果</h2>{_table(['metric','alternative','seed mean','bootstrap 95% CI','positive seed fraction','exact p'],metric_rows)}<figure>{_pvalue_svg(metrics, relay_config.primary_alpha)}<figcaption>横轴为 −log10(exact sign-flip p)，越右证据越强；虚线是 α={relay_config.primary_alpha:g}。这是逐 estimand p，不替代四 family 的 intersection-union 判据。</figcaption></figure>{_table(['family','family p','status'],family_rows)}{result_readout}<div class="conclusion">本节结论：四 family 联合判据为 {status}；global intersection-union p={_fmt(decision['global_intersection_union_p'],6)}。Edge-patch family 同时要求机械 first stage 与答案 transport，removal family 同时要求误差恶化和正确 margin 降低。</div></section>
<section><h2>4. 干预定义与控制</h2><h3>4.1 V-only edge patch</h3><div class="formula">Δz<sub>h,P</sub>=Σ<sub>j∈P</sub>α<sub>h</sub><sup>receiver</sup>(q,j)[v<sup>donor</sup>(j)−v<sup>receiver</sup>(j)]</div><p>donor/receiver 使用同 seed 的严格位置对齐 prompts；所有注册 pair 双向执行。正 transport 定义为 Δexpected-count/(donor−receiver)&gt;0。</p><h3>4.2 Serial OV block</h3><p>将 edge patch 的 post-O 变化投影到冻结 V4.4.4 natural axis，只在 H16/H19 的 pre-O z-space 加入反向分量。控制位于同一 W<sub>O</sub> column span、post-O 范数相同且与 natural axis 正交。</p><h3>4.3 Relay removal</h3><p>relay count step 与 count-zero center 只在 discovery fit counts 上拟合。confirmation 中移除自然 edge output 在该 relay step 上的 centered component，并与同范数正交控制比较。</p><div class="conclusion">本节结论：edge patch 检验自然 routing 下的内容充分性，removal 检验自然必要性，serial block 检验该内容效应是否由冻结的 downstream OV channel 中介；三者不能相互替代。</div></section>
<section><h2>5. 审计与限制</h2>{_table(['check','status','detail'],audit_rows)}<ul><li>Attention row 由 selected module 的 eager/cache 重跑取得；该重跑与原 forward 的最终 candidate-logit 差异逐样本保存为数值诊断，不作为机制有效性的硬门槛。硬门槛是更直接的 L28 pre-O 重建：原 forward 的 z 必须能由所记录的 ΣαV 在 relative L2 ≤ {relay_config.contribution_reconstruction_relative_tolerance:g} 下重建。</li><li>Discovery 与 confirmation seeds 分离，但 confirmation seeds 与已完成的 V4.4.4 downstream campaign 相同；它们对 relay selection 是未使用的，却不是整个研究中从未观察过的全新 seeds。</li><li>动态 top-K 集合由每个 receiver 的自然 contribution 按预注册规则选择；若它胜出，结论适用于该动态 set rule，不代表固定绝对 token indices。</li><li>source-control 集合只能说明原始 span 携带内容，不能单独称为 relay；primary selection 明确排除了这些集合。</li><li>不保存 raw per-token contribution maps 或 full V tensors；所有定位统计在线聚合。</li></ul><div class="conclusion">本节结论：审计 all_checks_pass={str(analysis['audit']['all_checks_pass']).lower()}。本补充只验证 terminal relay→OV→answer，不识别 relay 的上游构造电路。</div><p class="foot">schema: {html.escape(str(analysis['schema_version']))}</p></section>
</main></body></html>"""


def build_relay_html_report(
    *, run_root: str | Path, relay_config: V444RelayConfig
) -> Path:
    root = Path(run_root)
    analysis_path = (
        root / "analysis" / "relay" / "realistic_niah_v4_4_4_relay_analysis.json"
    )
    if not analysis_path.is_file():
        raise FileNotFoundError("Relay analysis JSON is missing")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    discovery_root = stage_root(root, relay_config.model_label, "relay_discovery")
    discovery = pd.read_csv(
        discovery_root / "position_set_selection_summary.csv.gz"
    )
    output = (
        root / "analysis" / "relay" / "realistic_niah_v4_4_4_relay_report.html"
    )
    atomic_text(output, _template(analysis, discovery, relay_config=relay_config))
    return output
