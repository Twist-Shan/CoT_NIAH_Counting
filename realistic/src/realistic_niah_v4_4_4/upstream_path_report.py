from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from realistic_niah_v4_4_3.io import atomic_text, stage_root

from .upstream_path_analysis import ANALYSIS_STAGE
from .upstream_path_spec import V444UpstreamPathConfig


ROUTE_LABELS = {
    "slot_edge_qk": "answer-query 的 slot-edge QK-only",
    "answer_query_full": "answer-query full-Z（上界）",
    "slot_state": "slot-position state-builder",
}


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if abs(number) < 1e-3 and number != 0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _table(frame: pd.DataFrame, columns: Iterable[tuple[str, str]]) -> str:
    specs = list(columns)
    head = "".join(f"<th>{html.escape(label)}</th>" for _name, label in specs)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for name, _label in specs:
            value = row[name]
            if isinstance(value, bool):
                rendered = "是" if value else "否"
            elif isinstance(value, (int, float)):
                rendered = _fmt(value)
            else:
                rendered = html.escape(str(value))
            cells.append(f"<td>{rendered}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _interval_svg(
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    x_label: str,
    label_column: str | None = None,
) -> str:
    if frame.empty:
        return "<p>无可绘制结果。</p>"
    rows = frame.reset_index(drop=True)
    lows = rows[f"{metric}_ci_low"].astype(float)
    highs = rows[f"{metric}_ci_high"].astype(float)
    limit = max(abs(float(lows.min())), abs(float(highs.max())), 1e-4) * 1.15
    width = 1050
    left = 310
    right = 35
    row_height = 32
    top = 55
    bottom = 55
    height = top + bottom + row_height * len(rows)

    def x(value: float) -> float:
        return left + (value + limit) / (2 * limit) * (width - left - right)

    palette = {
        "slot_edge_qk": "#157f75",
        "answer_query_full": "#b56a26",
        "slot_state": "#5b5da8",
    }
    pieces = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
        f"<text x='{left}' y='25' class='svg-title'>{html.escape(title)}</text>",
        f"<line x1='{x(0):.2f}' x2='{x(0):.2f}' y1='{top-12}' y2='{height-bottom+8}' class='zero'/>",
    ]
    for index, row in rows.iterrows():
        y = top + index * row_height
        route = str(row["route"])
        label = (
            str(row[label_column])
            if label_column is not None
            else f"{row['early_set']} · {ROUTE_LABELS.get(route, route)}"
        )
        mean = float(row[f"{metric}_mean"])
        low = float(row[f"{metric}_ci_low"])
        high = float(row[f"{metric}_ci_high"])
        color = palette.get(route, "#444")
        pieces.extend(
            [
                f"<text x='{left-12}' y='{y+5}' text-anchor='end' class='svg-label'>{html.escape(label)}</text>",
                f"<line x1='{x(low):.2f}' x2='{x(high):.2f}' y1='{y}' y2='{y}' stroke='{color}' stroke-width='3'/>",
                f"<line x1='{x(low):.2f}' x2='{x(low):.2f}' y1='{y-5}' y2='{y+5}' stroke='{color}' stroke-width='2'/>",
                f"<line x1='{x(high):.2f}' x2='{x(high):.2f}' y1='{y-5}' y2='{y+5}' stroke='{color}' stroke-width='2'/>",
                f"<circle cx='{x(mean):.2f}' cy='{y}' r='5.5' fill='{color}'/>",
            ]
        )
    for tick in (-limit, -limit / 2, 0.0, limit / 2, limit):
        pieces.append(
            f"<text x='{x(tick):.2f}' y='{height-25}' text-anchor='middle' class='svg-tick'>{tick:.3f}</text>"
        )
    pieces.append(
        f"<text x='{(left+width-right)/2:.2f}' y='{height-5}' text-anchor='middle' class='svg-axis'>{html.escape(x_label)}</text>"
    )
    pieces.append("</svg>")
    return "".join(pieces)


def _natural_svg(frame: pd.DataFrame) -> str:
    rows = frame.sort_values("candidate_rank").reset_index(drop=True)
    width, height = 1050, 390
    left, right, top, bottom = 75, 30, 40, 85
    maximum = max(
        float(rows["v442_stable_score"].max()),
        float(rows["observed_broad_score_mean"].max()),
        1e-6,
    ) * 1.15

    def y(value: float) -> float:
        return top + (maximum - value) / maximum * (height - top - bottom)

    spacing = (width - left - right) / len(rows)
    pieces = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='冻结与复测 broad retrieval score'>",
        "<text x='75' y='24' class='svg-title'>冻结 V4.4.2 排名与本轮自然运行复测</text>",
        f"<line x1='{left}' x2='{left}' y1='{top}' y2='{height-bottom}' class='axis'/>",
        f"<line x1='{left}' x2='{width-right}' y1='{height-bottom}' y2='{height-bottom}' class='axis'/>",
    ]
    for index, row in rows.iterrows():
        center = left + (index + 0.5) * spacing
        label = f"L{int(row['layer'])}H{int(row['head'])}"
        frozen = float(row["v442_stable_score"])
        observed = float(row["observed_broad_score_mean"])
        base = height - bottom
        pieces.extend(
            [
                f"<rect x='{center-14:.2f}' y='{y(frozen):.2f}' width='12' height='{base-y(frozen):.2f}' fill='#c68a38'/>",
                f"<rect x='{center+2:.2f}' y='{y(observed):.2f}' width='12' height='{base-y(observed):.2f}' fill='#247b78'/>",
                f"<text x='{center:.2f}' y='{base+18}' text-anchor='middle' class='svg-tick'>{label}</text>",
                f"<text x='{center:.2f}' y='{base+34}' text-anchor='middle' class='svg-tick'>rank {int(row['candidate_rank'])}</text>",
            ]
        )
    pieces.extend(
        [
            "<rect x='720' y='12' width='12' height='12' fill='#c68a38'/><text x='738' y='23' class='svg-tick'>V4.4.2 cue-stable</text>",
            "<rect x='875' y='12' width='12' height='12' fill='#247b78'/><text x='893' y='23' class='svg-tick'>本轮均值</text>",
            f"<text x='18' y='{(top+height-bottom)/2:.2f}' transform='rotate(-90 18 {(top+height-bottom)/2:.2f})' text-anchor='middle' class='svg-axis'>broad retrieval score</text>",
            "</svg>",
        ]
    )
    return "".join(pieces)


def build_html_report(
    *, run_root: str | Path, config: V444UpstreamPathConfig
) -> Path:
    root = Path(run_root)
    analysis_root = stage_root(root, config.model_label, ANALYSIS_STAGE)
    analysis_path = analysis_root / "realistic_niah_v4_4_4_upstream_path_analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError("Run upstream-path analysis before building the report")
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    summary = pd.DataFrame(payload["summary"])
    natural = pd.DataFrame(payload["natural_head_summary"])
    decision = payload["decision"]
    audit = payload["audit"]
    base = summary[summary["late_set"] == config.primary_late_set].copy()
    base["route"] = pd.Categorical(base["route"], categories=list(config.routes), ordered=True)
    base = base.sort_values(["route", "early_set"])
    expanded = summary[summary["late_set"] != config.primary_late_set].copy()
    relay_comparison = summary[
        (summary["early_set"] == "top4")
        & (summary["route"] == "slot_state")
    ].copy()
    relay_comparison["late_set"] = pd.Categorical(
        relay_comparison["late_set"],
        categories=list(config.late_sets),
        ordered=True,
    )
    relay_comparison = relay_comparison.sort_values("late_set")
    supported = summary[summary["serial_path_supported"]].copy()
    classification_cn = {
        "upstream_read_to_l28_write_supported_exploratory": "至少一条受约束的上游读取路径得到探索性串联支持",
        "upstream_read_to_expanded_l28_write_supported_exploratory": "至少一条上游读取/relay 路径得到扩展 L28 头集的探索性串联支持",
        "full_output_chain_only": "只有 full-Z 上界成立，尚不能定位到 QK 或 slot-state 路径",
        "upstream_to_l28_chain_not_supported": "本轮没有建立上游读取到 L28 写入的串联路径",
    }.get(decision["classification"], decision["classification"])
    candidate_lines = "；".join(
        f"rank {index+1}: L{item.layer}H{item.head} (min={item.stable_score:.4f})"
        for index, item in enumerate(config.early_candidates)
    )
    audit_rows = pd.DataFrame(
        [
            {"检查": item["name"], "通过": bool(item["passed"]), "细节": str(item["detail"])}
            for item in audit["checks"]
        ]
    )
    support_text = (
        "没有组合同时通过 early donor log-odds gain 与 L28-specific mediation 的 Holm 校正门槛。"
        if supported.empty
        else "；".join(
            f"{row.early_set}/{ROUTE_LABELS.get(str(row.route), row.route)}/{row.late_set}"
            for row in supported.itertuples()
        )
    )
    expanded_section = ""
    if not expanded.empty:
        expanded_section = f"""
        <section>
          <h2>6. 扩展 L28 头集的结果</h2>
          <p>扩展阶段只在基准 H16/H19 未能建立受约束路径时触发。它比较同一早期干预在 <code>H16–H19 GQA group</code>、V4.4.2 broad top-4 与 broad top-8 L28 头集上的精确恢复阻断。扩展结果仍使用同一批 10 个 seed，因此只能说明哪个集合更能截获该路径，不能当作独立复现。</p>
          <figure>{_interval_svg(relay_comparison, metric='donor_log_odds_mediation_specificity', title='top-4 slot-state relay 在不同 L28 头集中的中介特异性', x_label='control gain − exact-block gain；正值 = 精确恢复抑制更多', label_column='late_set')}<figcaption>图 4｜纵轴列出四个预先冻结的 L28 接收集合；横轴是 top-4 早层 slot-state 干预的 donor log-odds mediation specificity。圆点为跨 seed 均值，线段为 seed bootstrap 95% CI。显著性不由 CI 是否跨 0 判定，而使用表中的双侧 exact sign-flip Holm p。<code>base_h16_h19</code> 是 H16/H19；<code>gqa_h16_h19</code> 是完整 H16–H19；broad top-4 为 H19/H16/H17/H2；broad top-8 再包含 H18/H23/H1/H11。</figcaption></figure>
          {_table(relay_comparison, [("late_set","L28 集合"),("early_donor_log_odds_gain_mean","early gain"),("early_donor_log_odds_gain_holm_p","early Holm p"),("late_block_donor_log_odds_gain_mean","exact-block gain"),("late_control_donor_log_odds_gain_mean","orthogonal-control gain"),("donor_log_odds_mediation_specificity_mean","mediation specificity"),("donor_log_odds_mediation_specificity_holm_p","mediation Holm p"),("serial_path_supported","串联支持")])}
          <p>通过双重门槛的组合只有 <code>top4 / slot_state / H16–H19</code> 与 <code>top4 / slot_state / broad top-8</code>。H16/H19 单独不通过；缺少 H18、以 H2 替代的 broad top-4 也不通过。两个阳性集合都包含 L28 H18，而两个阴性较小集合都不包含它，因此 H18 是下一步最优先的必要性候选；但本轮是 set intervention，尚不能把集合效应归因给单个 H18。</p>
          <details><summary>查看全部扩展组合</summary>{_table(expanded, [("late_set","L28 集合"),("early_set","早期集合"),("route","路径"),("early_donor_log_odds_gain_mean","donor log-odds gain"),("early_donor_log_odds_gain_holm_p","early Holm p"),("donor_log_odds_mediation_specificity_mean","mediation specificity"),("donor_log_odds_mediation_specificity_holm_p","mediation Holm p"),("serial_path_supported","串联支持")])}</details>
          <p><strong>本节结论：</strong>{'早层 top-4 在 slot positions 写下的 donor state 产生稳定的 answer shift；该 shift 被完整 L28 H16–H19 或包含它的 broad top-8 精确恢复特异地消除，支持“slot relay → L28 set → answer”的探索性串联路径。' if decision['expanded_l28_set_support'] else '扩大 L28 头集后仍未得到通过双重门槛的受约束串联证据。'}</p>
        </section>
        """
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>V4.4.4 上游读取 → L28 OV 写入路径报告</title>
<style>
:root{{--ink:#24231f;--muted:#6d6a61;--paper:#f5f1e8;--panel:#fffdf8;--line:#d8d0c1;--teal:#157f75;--orange:#b56a26;--violet:#5b5da8;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans SC","Microsoft YaHei",sans-serif;line-height:1.7}}
main{{max-width:1220px;margin:0 auto;padding:42px 34px 80px}} h1{{font-family:Georgia,"Noto Serif SC",serif;font-size:40px;line-height:1.2;margin:0 0 12px}} h2{{font-family:Georgia,"Noto Serif SC",serif;font-size:27px;margin:0 0 18px;border-bottom:1px solid var(--line);padding-bottom:8px}} h3{{font-size:19px;margin-top:26px}} p{{margin:10px 0 16px}} section{{background:var(--panel);border:1px solid var(--line);padding:28px 30px;margin-top:22px;box-shadow:0 2px 12px #4b423312}} .lede{{font-size:18px;color:#3d3a34;max-width:1000px}} .verdict{{border-left:6px solid var(--teal);background:#edf7f4;padding:18px 22px;margin:22px 0}} .caveat{{border-left:6px solid var(--orange);background:#fbf2e7;padding:15px 20px}} code{{background:#eee8dc;padding:2px 5px;border-radius:3px}} .formula{{font-family:"Cambria Math",Georgia,serif;background:#f2ede3;border:1px solid var(--line);padding:14px 18px;overflow:auto}} .table-wrap{{overflow:auto;border:1px solid var(--line);margin:16px 0}} table{{border-collapse:collapse;width:100%;font-size:13px;background:white}} th{{position:sticky;top:0;background:#ede7db;text-align:left}} th,td{{padding:9px 10px;border-bottom:1px solid #e3ddd1;white-space:nowrap}} tr:hover td{{background:#faf6ed}} figure{{margin:22px 0;border:1px solid var(--line);background:#fff;padding:14px}} figcaption{{color:var(--muted);font-size:13px;margin-top:8px}} svg{{width:100%;height:auto}} .svg-title{{font-size:17px;font-weight:650;fill:#2d2a25}} .svg-label{{font-size:12px;fill:#35322d}} .svg-tick{{font-size:11px;fill:#6d6a61}} .svg-axis{{font-size:12px;fill:#4c4942}} .axis{{stroke:#817a6f;stroke-width:1}} .zero{{stroke:#8d877c;stroke-width:1;stroke-dasharray:4 4}} .meta{{font-size:13px;color:var(--muted)}}
</style></head><body><main>
<header><div class="meta">Realistic NiaH Counting · V4.4.4 append-only supplement · Qwen3-8B</div><h1>前层 broad retrieval 如何接到 L28 OV 写入？</h1><p class="lede">本报告检验一个严格的串联猜想：V4.4.2 冻结的前层 broad-retrieval heads 先形成或搬运 count state，随后 L28 的 H16/H19（必要时扩展同层集合）读取该变化并通过自己的 O projection 写入 answer-count 方向。</p></header>
<div class="verdict"><strong>当前总判断：</strong>{html.escape(classification_cn)}。<br/><strong>通过组合：</strong>{html.escape(support_text)}</div>
<div class="caveat"><strong>证据等级：</strong>这是对既有 V4.4.4 confirmation seeds（1284–1293）的 10-seed 探索性因果补充，不是新 seed 的独立 confirmation。显著性使用 seed 作为独立单位、双侧 exact sign-flip，并在每个 L28 集合内做 Holm 校正。</div>

<section><h2>1. 猜想、可区分的路径与判据</h2>
<p><strong>路径 A（QK/answer-query）：</strong>前层 head 在最终 answer query 上改变对所有注册 slot edges 的权重；干预使用 donor 的 attention routing 与 receiver 的 V，因此只替换 <code>α</code> 引起的 slot-edge pre-O 贡献。它检验“前层 broad retrieval 的路由结果是否沿 query residual 送到 L28”。</p>
<p><strong>路径 B（slot-state/KV precursor）：</strong>只在 slot token 的 query positions，把候选前层 heads 的自然 pre-O <code>z_h</code> 换成 donor state；answer query 本身不在早期被直接修改。若影响到 L28，这说明这些早层输出能先改写 slot residual，继而成为更后层 K/V 或 relay state 的前体。</p>
<p><strong>上界：</strong><code>answer_query_full</code> 把早层候选 head 的完整 answer-query Z 换成 donor 值。它能说明该 head set 的自然输出可以搬运 count，但不能把作用拆成 QK 与 V，因此只作为正上界。</p>
<div class="formula">donor log-odds gain: G<sub>E</sub> = [(s<sub>D</sub>−s<sub>R</sub>) | early patch] − [(s<sub>D</sub>−s<sub>R</sub>) | receiver]<br/>mediation specificity: M<sub>G</sub> = G<sub>early + orthogonal L28 control</sub> − G<sub>early + exact L28 restoration</sub></div>
<p>主终点使用完整 candidate sequence score 的 donor-vs-receiver log-odds gain；它在 clean receiver 概率接近 1 时仍有分辨率。expected-count transport 继续作为可解释的次要效应量。对每次早期干预，先记录它诱发的 L28 selected-head pre-O 变化 Δz。精确阻断加入 −Δz，使这些 heads 回到 receiver 的自然状态；对照位于同一 selected-head W<sub>O</sub> span、post-O 范数相同，并与 Δz 的 post-O 方向正交。只有 <code>G_E &gt; 0</code> 与 <code>M_G &gt; 0</code> 都通过 Holm 校正，才称该串联路径得到支持。</p>
<p><strong>本节结论：</strong>本设计把“前层能影响答案”“变化到达 L28”“答案效应特异地由 L28 selected heads 中介”分成三个可单独失败的命题；单独 injection 成功不再被解释成自然 transporter。</p></section>

<section><h2>2. 冻结头集与实验设定</h2>
<p>候选完全来自 V4.4.2 的 non-thinking broad-retrieval atlas，并在看到本轮因果结果前按 <code>min(cue-present score, cue-absent score)</code> 降序冻结。使用嵌套 top-2、top-4、top-8，具体为：{html.escape(candidate_lines)}。</p>
<p>模型为 Qwen3-8B；mediator 固定在 L28。主 L28 set 是 H16/H19。donor/receiver 对为 1↔6、3↔8、5↔10；每个 seed 先在 6 个有向 pair 内平均，再跨 10 个 seed 推断。所有干预发生在真正 pre-O 边界，最终变化均通过模型自己的 W<sub>O</sub> 产生。没有持久化 raw attention、QK cache 或 full hidden state，只保存标量摘要。</p>
<figure>{_natural_svg(natural)}<figcaption>图 1｜横轴是按 V4.4.2 cue-stable broad score 冻结的候选 head/rank；纵轴是 broad retrieval score。橙色为 V4.4.2 两种 cue 条件中的较小值，绿色为本轮 10 seeds × 10 counts 的自然运行均值。两者的数据分布不同，因此本图用于检查候选仍有同类信号，不把绝对高度相等作为门槛。</figcaption></figure>
{_table(natural, [("candidate_rank","rank"),("layer","layer"),("head","head"),("v442_stable_score","V4.4.2 stable"),("observed_broad_score_mean","本轮 broad mean"),("occurrence_coverage_mean","coverage"),("baseline_correct_rate","baseline correct")])}
<p><strong>本节结论：</strong>候选与 top-k 边界在本轮结果之前已固定；本报告中的扩头比较不会使用 outcome 重新挑选早期 heads。</p></section>

<section><h2>3. 前层干预是否产生 donor-directed answer shift？</h2>
<figure>{_interval_svg(base, metric='early_donor_log_odds_gain', title='前层干预的 donor-vs-receiver sequence log-odds gain', x_label='log-score units；正值 = donor 相对 receiver 更受偏好')}<figcaption>图 2｜每行是一个早期 top-k × 路径。圆点为跨 seed 均值，线段为 seed bootstrap 95% CI；先在每个 seed 内平均 6 个有向 donor pair。横轴为干预前后 donor sequence score 减 receiver sequence score 的变化，0 表示没有 donor-directed 偏好变化。</figcaption></figure>
{_table(base, [("early_set","早期集合"),("route","路径"),("early_donor_log_odds_gain_mean","log-odds gain"),("early_donor_log_odds_gain_ci_low","CI low"),("early_donor_log_odds_gain_ci_high","CI high"),("early_donor_log_odds_gain_holm_p","Holm p"),("early_transport_mean","expected-count transport (secondary)"),("early_supported","通过")])}
<p><strong>本节结论：</strong>前层干预是否具有 donor-directed 行为效应，必须看表中 <code>early_supported</code>；full-Z 成立而两个受约束路径不成立时，只能得到“整个早期 head output 可搬运”的较弱结论。</p></section>

<section><h2>4. 该效应是否特异地经过 L28 H16/H19？</h2>
<figure>{_interval_svg(base, metric='donor_log_odds_mediation_specificity', title='L28 精确恢复相对等范数正交对照的 log-odds 中介特异性', x_label='control gain − exact-block gain；正值 = L28 精确恢复抑制更多')}<figcaption>图 3｜纵向每行仍是同一早期 top-k × 路径；横轴为 donor log-odds mediation specificity。若精确恢复 L28 H16/H19 比同范数、同 span、正交方向的对照更强地消除 donor-directed log-odds gain，则数值为正。误差线定义同图 2。</figcaption></figure>
{_table(base, [("early_set","早期集合"),("route","路径"),("donor_log_odds_mediation_specificity_mean","specificity mean"),("donor_log_odds_mediation_specificity_ci_low","CI low"),("donor_log_odds_mediation_specificity_ci_high","CI high"),("donor_log_odds_mediation_specificity_holm_p","Holm p"),("mediation_specificity_mean","expected-count specificity (secondary)"),("serial_path_supported","串联支持")])}
<p><strong>本节结论：</strong>{html.escape(support_text)} 只有同时满足 early donor log-odds gain 与 log-odds mediation specificity 的组合，才支持“前层读/建 state → L28 selected heads → answer”的串联解释。</p></section>

<section><h2>5. 数值与实现审计</h2>
{_table(audit_rows, [("检查","检查"),("通过","通过"),("细节","细节")])}
<p>精确阻断的 closure 比较阻断后的 L28 selected-head Z 与 receiver 自然 Z；正交性在真实 W<sub>O</sub> 输出空间计算。deterministic-prefill audit 则验证为 block/control 重跑早期干预时，进入 L28 前的状态与单独 early run 一致。</p>
<p><strong>本节结论：</strong>{'所有预注册数值与存储审计均通过。' if audit['all_checks_pass'] else '至少一项审计失败；在修复前不应解释因果结果。'}</p></section>

{expanded_section}

<section><h2>7. 可以声称什么、还不能声称什么</h2>
<p>若 QK-only 串联成立，可声称：在 answer query 上，冻结前层 heads 对 slot edges 的路由变化能产生 donor-directed count shift，且该效应中特异的一部分通过 L28 selected heads。若 slot-state 串联成立，可声称：早层 heads 在 slot positions 写下的 state 能成为下游读取/写入路径的因果前体。两者都不要求同一个 head 同时承担原始 needle 定位与最终 OV 写入。</p>
<p>本实验尚不能证明一个唯一的逐 token 电路，也不能把 slot-state route 直接等同于某个具体 L28 K head；slot-position Z patch 允许 L22–L27 的任意中间计算参与。进一步定位需要在已支持路径内做 layer-by-layer edge patch、leave-one-head-out，以及新 seeds 的冻结 confirmation。</p>
<p><strong>本节结论：</strong>当前最强可用表述由首页总判断给出；所有结果都应保留“reused-seed exploratory causal supplement”的限定。</p></section>
</main></body></html>"""
    destination = analysis_root / "realistic_niah_v4_4_4_upstream_path_report.html"
    atomic_text(destination, document)
    return destination
