from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


GRAMMAR_ORDER = [
    "adjacent_rank_after_city",
    "adjacent_rank_before_city",
    "same_unit_rank_after_city",
    "same_unit_rank_before_city",
    "structural_explicit_rank_before_city",
    "structural_invariant_bullet",
    "structural_unmarked",
    "evidence_sequence_unranked",
]
GRAMMAR_RULES = {
    "adjacent_rank_after_city": (
        "rank 与 city 分属相邻语义单元；city 在前",
        "先完成 city retrieval，再生成后置 rank commit",
        '<span class="city">Riga, 95.</span> <span class="rank">Fifth.</span>',
    ),
    "adjacent_rank_before_city": (
        "rank-only 单元紧邻并位于 city 单元之前",
        "rank state 可先形成，再读取当前 city target",
        '<span class="rank">8.</span> <span class="city">Osaka</span>',
    ),
    "same_unit_rank_after_city": (
        "同一语义单元内，city 字符早于 rank 字符",
        "city target 与后置 update 在同一 unit 内分开",
        '(City: <span class="city">Baku</span>, Score: 98) → <span class="rank">Count = 2</span>',
    ),
    "same_unit_rank_before_city": (
        "同一语义单元内，rank 字符早于 city 字符",
        "marker state 后继续生成 city",
        '(<span class="rank">Record 2:</span> <span class="city">Riga</span>, 60)',
    ),
    "structural_explicit_rank_before_city": (
        "structural recap 中存在 indexed/ordinal marker",
        "只作 secondary structural control",
        '<span class="rank">That’s one record:</span> <span class="city">Wuhan with 59.</span>',
    ),
    "structural_invariant_bullet": (
        "每项复用相同 bullet；bullet 不表达 running count",
        "marker-neutral structural control",
        '<span class="rank">−</span> <span class="city">Baku: 72</span>',
    ),
    "structural_unmarked": (
        "没有可解释 rank marker",
        "只依赖 city/item/transition sites",
        'In the 2024 city score audit, <span class="city">Porto received a score of 51.</span>',
    ),
    "evidence_sequence_unranked": (
        "按正文首次出现顺序恢复 score-supported city evidence",
        "exploratory retrieval；不允许 rank/accumulator 解释",
        '<span class="city">Vancouver with a score of 62.</span> Then <span class="city">Geneva with 57</span>, …',
    ),
}
FINAL_SITE_RULES = {
    "p0_item_end": (
        "P0 · previous-item commit",
        "在 transition k→k+1 中，选择完整 item k 的 endpoint output token；"
        "此时 target item k+1 的 marker 与 city 都还没有出现。",
    ),
    "post_marker": (
        "P2 · marker-conditioned query",
        "选择 target item k+1 的 rank semantic core 最后一个 output token；"
        "rank 信息已经进入 residual stream，但 city k+1 的首 token 尚未出现。",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def route_for(config: dict[str, Any], grammar: str) -> str:
    route = config["routes"].get(grammar)
    if route is None:
        return "N/A"
    required = route.get("required", [])
    optional = route.get("optional", [])
    value = " + ".join(str(item) for item in required)
    if optional:
        value += " (optional: " + ", ".join(str(item) for item in optional) + ")"
    return value


def model_grammar_rows(
    *,
    model: str,
    routing: dict[str, Any],
    event_counts: dict[tuple[str, str], int],
    anchor_support: dict[str, int],
) -> list[list[str]]:
    """Render only grammar classes that are actually observed for one model."""

    rows: list[list[str]] = []
    for grammar in GRAMMAR_ORDER:
        count = int(event_counts.get((model, grammar), 0))
        if count == 0:
            continue
        rule, _meaning, example = GRAMMAR_RULES[grammar]
        site = route_for(routing, grammar)
        if site not in FINAL_SITE_RULES:
            raise ValueError(f"Unexpected final route for {model}/{grammar}: {site}")
        site_label, site_position = FINAL_SITE_RULES[site]
        rows.append(
            [
                f"<code>{html.escape(grammar)}</code>",
                f'<div class="trace">{example}</div><div class="small">{html.escape(rule)}</div>',
                f'<span class="route-badge route-{html.escape(site)}">'
                f"{html.escape(site)}</span><div class=\"small\">{html.escape(site_label)}</div>",
                html.escape(site_position),
                f"{count:,}",
                f"{int(anchor_support.get(grammar, 0)):,}",
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-review-dir", type=Path, required=True)
    parser.add_argument("--qwen-routing", type=Path, required=True)
    parser.add_argument("--gemma-routing", type=Path, required=True)
    parser.add_argument("--qwen-selection", type=Path, required=True)
    parser.add_argument("--gemma-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    review = args.site_review_dir
    manifest = json.loads((review / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((review / "audit.json").read_text(encoding="utf-8"))
    events = pd.read_csv(review / "event_registry.csv")
    qwen_route = json.loads(args.qwen_routing.read_text(encoding="utf-8"))
    gemma_route = json.loads(args.gemma_routing.read_text(encoding="utf-8"))
    qwen_selection = json.loads(args.qwen_selection.read_text(encoding="utf-8"))
    gemma_selection = json.loads(args.gemma_selection.read_text(encoding="utf-8"))

    event_counts = (
        events.groupby(["model_label", "grammar_class"], dropna=False)
        .size()
        .to_dict()
    )
    q_registry = qwen_selection["anchor_registry"]
    g_registry = gemma_selection["anchor_registry"]
    qwen_grammar_rows = model_grammar_rows(
        model="Qwen3-8B",
        routing=qwen_route,
        event_counts=event_counts,
        anchor_support={
            str(key): int(value) for key, value in q_registry["grammar_support"].items()
        },
    )
    gemma_grammar_rows = model_grammar_rows(
        model="Gemma4-E4B",
        routing=gemma_route,
        event_counts=event_counts,
        anchor_support={
            str(key): int(value) for key, value in g_registry["grammar_support"].items()
        },
    )

    q_heads = qwen_selection["development_selection"]
    g_heads = gemma_selection["development_selection"]
    head_rows = [
        [
            "Qwen3-8B",
            f"<code>{html.escape(str(q_heads['head_ranking_source_grammar']))}</code>",
            f"<code>{html.escape(str(q_heads['head_ranking_source_anchor']))}</code>",
            "query token → registered target prompt record 的 attention mass",
            f"K{q_heads['primary_bank_size']}",
            "按 Qwen grammar route 应用于 <code>post_marker</code> 或 <code>p0_item_end</code>",
        ],
        [
            "Gemma4-E4B",
            f"<code>{html.escape(str(g_heads['head_ranking_source_grammar']))}</code>",
            f"<code>{html.escape(str(g_heads['head_ranking_source_anchor']))}</code>",
            "query token → registered target prompt record 的 attention mass",
            f"K{g_heads['primary_bank_size']}",
            "统一应用于所有 Gemma grammar 的 <code>p0_item_end</code>",
        ],
    ]
    qwen_event_total = sum(
        int(value)
        for (model, _grammar), value in event_counts.items()
        if model == "Qwen3-8B"
    )
    gemma_event_total = sum(
        int(value)
        for (model, _grammar), value in event_counts.items()
        if model == "Gemma4-E4B"
    )
    qwen_post_grammars = {
        grammar
        for grammar in GRAMMAR_ORDER
        if int(event_counts.get(("Qwen3-8B", grammar), 0)) > 0
        and route_for(qwen_route, grammar) == "post_marker"
    }
    qwen_post_events = sum(
        int(event_counts.get(("Qwen3-8B", grammar), 0))
        for grammar in qwen_post_grammars
    )
    qwen_post_anchors = sum(
        int(q_registry["grammar_support"].get(grammar, 0))
        for grammar in qwen_post_grammars
    )

    css = """
:root{--paper:#f3eee4;--surface:#fffdf8;--ink:#20242d;--muted:#626a74;--line:#c9c2b6;--deep:#24223e;--indigo:#23165c;--teal:#087f72;--teal-soft:#e2f0ec;--amber:#9a641d;--amber-soft:#faeed8;--city:#087f72;--rank:#a35b12}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:14.5px/1.67 "Segoe UI Variable","Aptos","Noto Sans SC",system-ui,sans-serif}.layout{display:grid;grid-template-columns:252px minmax(0,1150px);gap:28px;max-width:1480px;margin:auto;padding:27px}nav{position:sticky;top:18px;align-self:start;max-height:calc(100dvh - 36px);overflow:auto;padding:19px 18px;background:#ece6da;border-top:3px solid var(--indigo)}nav strong{display:block;margin-bottom:11px;color:var(--indigo)}nav a{display:block;padding:7px 0;color:#4c4a5d;text-decoration:none;border-bottom:1px solid #d7d0c5;font-size:13px}main{min-width:0}header{padding:40px 43px;background:var(--deep);color:#fffdf8;border-radius:12px}.eyebrow{letter-spacing:.13em;font-size:12px;font-weight:850;color:#9bd3c8}h1{font-size:40px;line-height:1.12;letter-spacing:-.04em;margin:8px 0 14px}h2{font-size:28px;line-height:1.2;letter-spacing:-.03em;margin:0 0 17px;color:var(--indigo)}h3{font-size:18px;margin:25px 0 10px;color:var(--indigo)}.lead{font-size:18px;max-width:920px}.meta{color:#cdc9da}.small{font-size:12px;color:var(--muted);margin-top:6px}section{scroll-margin-top:18px;margin-top:22px;padding:31px 34px;background:var(--surface);border:1px solid var(--line);border-radius:10px}.split,.site-grid{display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.split>div,.site-grid>article{min-width:0;padding:20px}.split>div+div,.site-grid>article+article{border-left:1px solid var(--line)}.callout{padding:16px 19px;background:var(--amber-soft);border-left:4px solid var(--amber)}.decision{padding:16px 19px;background:var(--teal-soft);border-left:4px solid var(--teal)}.flow{font:14px/1.75 "Cascadia Mono",Consolas,monospace;padding:17px 20px;background:#2d2b45;color:#f6f2e9;overflow:auto}.token-line{max-width:100%;margin:14px 0;padding:13px 15px;background:#f0ebe1;border:1px solid #d7d0c5;font:13px/1.8 "Cascadia Mono",Consolas,monospace;overflow:auto;white-space:nowrap}.token-line .query{background:var(--teal);color:white;padding:4px 6px}.token-line .pending{color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);margin:14px 0 21px}table{width:100%;border-collapse:collapse;font-size:13px}#qwen table,#gemma table{table-layout:fixed;min-width:1000px}#qwen th:nth-child(1),#gemma th:nth-child(1){width:19%}#qwen th:nth-child(2),#gemma th:nth-child(2){width:27%}#qwen th:nth-child(3),#gemma th:nth-child(3){width:14%}#qwen th:nth-child(4),#gemma th:nth-child(4){width:26%}#qwen th:nth-child(5),#gemma th:nth-child(5),#qwen th:nth-child(6),#gemma th:nth-child(6){width:7%}#qwen td:first-child code,#gemma td:first-child code{white-space:normal;overflow-wrap:anywhere;word-break:break-word}th,td{padding:10px 11px;border-bottom:1px solid #ded8ce;text-align:left;vertical-align:top}th{background:#ece6da;white-space:nowrap;color:#303744}tr:last-child td{border-bottom:0}code{font-family:"Cascadia Mono",Consolas,monospace;background:#efebe3;padding:.08em .31em;border-radius:3px}.trace{font-family:"Cascadia Mono",Consolas,monospace;white-space:normal;min-width:0;background:#edf3f0;padding:8px 10px}.trace .city,.city{color:var(--city);font-weight:800}.trace .rank,.rank{color:var(--rank);font-weight:800}.route-badge{display:inline-block;padding:3px 7px;border:1px solid currentColor;font:700 10px/1.3 "Cascadia Mono",Consolas,monospace;white-space:nowrap}.route-p0_item_end{color:var(--teal)}.route-post_marker{color:var(--indigo)}.model-summary{display:flex;gap:18px;flex-wrap:wrap;padding:12px 0 16px;color:var(--muted)}.model-summary strong{color:var(--ink);font:700 17px "Cascadia Mono",Consolas,monospace}.audit{font-family:"Cascadia Mono",Consolas,monospace;font-size:11px;word-break:break-all;color:#67616c}.links{display:flex;gap:9px;flex-wrap:wrap}.links a{padding:8px 11px;border:1px solid var(--line);color:var(--indigo);text-decoration:none;background:#f7f3eb}@media(max-width:940px){.layout{display:block;padding:12px}nav{position:relative;top:0;max-height:none;margin-bottom:12px}.split,.site-grid{grid-template-columns:1fr}.split>div+div,.site-grid>article+article{border-left:0;border-top:1px solid var(--line)}header,section{padding:24px}h1{font-size:32px}}@media(max-width:560px){header,section{padding:21px 17px}.table-wrap{margin-left:-17px;margin-right:-17px;border-left:0;border-right:0}h2{font-size:24px}}@media print{body{background:#fff}.layout{display:block;padding:0}nav{display:none}header,section{break-inside:avoid}}
"""
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Native-thinking V5 · Parser and Token Sites</title><style>{css}</style></head><body><div class="layout">
<nav><strong>Parser / token sites</strong><a href="#logic">1 · 核心逻辑</a><a href="#sites">2 · 两个最终位点</a><a href="#qwen">3 · Qwen grammars</a><a href="#gemma">4 · Gemma grammars</a><a href="#alignment">5 · Token 对齐</a><a href="#causal-link">6 · 与 causal 实验衔接</a><a href="#scope">7 · 报告边界</a></nav><main>
<header><div class="eyebrow">REALISTIC NIAH · NATIVE THINKING V5</div><h1>Grammar-aware parser and token sites</h1><p class="lead">这份报告只解释最终使用的逻辑：thinking trace 如何被拆成逐个 event，每个 target event 属于哪种 surface grammar，以及 Qwen、Gemma 分别在哪一个精确 output token 开始 targeted-retrieval intervention。</p><p class="meta">Qwen3-8B + Gemma4-E4B · 600 frozen traces · representation 与 causal effect 分页报告</p></header>

<section id="logic"><h2>1 · 从 trace 到 transition：grammar 是逐 event 判定的</h2><div class="decision"><strong>核心单位是 transition k→k+1。</strong>Parser 先按文本中的真实 city、rank evidence 与 semantic-unit 边界恢复 event 1…M；然后对<strong>目标 event k+1</strong>判定 grammar，最后由 <code>(model, target grammar)</code> 决定 query token。</div><div class="flow">raw thinking trace
  → ordered events: (city, rank evidence, semantic unit)
  → transition k → k+1
  → classify the surface grammar of target event k+1
  → choose one frozen query token
  → retrieve the registered prompt record for needle k+1</div><p>Grammar 不是给整条 trace 贴一个总标签。相邻两个 event 可以采用不同写法，因此 routing 始终读取 target item 本身的字符顺序与 parser association。Gold N 和最终 <code>Total</code> 不参与 event 顺序或 token 位置的决定。</p><div class="model-summary"><span><strong>8</strong><br>union grammar classes</span><span><strong>7</strong><br>Qwen observed</span><span><strong>5</strong><br>Gemma observed</span><span><strong>2</strong><br>final query roles</span></div></section>

<section id="sites"><h2>2 · 最终只保留两个 query roles</h2><div class="site-grid"><article><h3><code>p0_item_end</code></h3><p><strong>P0：previous-item commit。</strong>在 transition k→k+1 中，选择完整 item k 的 endpoint output token。这个 state 已经提交了 running progress k，但 target item k+1 的 marker 与 city 都尚未出现。</p><div class="token-line">[ item k … <span class="query">endpoint token</span> ] <span class="pending">[ target item k+1 尚未生成 ]</span></div><p class="small">如果 item k 以句点、右括号或格式 token 结束，query 就是 exact-prefix alignment 得到的那个 endpoint token；不是固定寻找某个标点字符串。</p></article><article><h3><code>post_marker</code></h3><p><strong>P2：marker-conditioned query。</strong>先进入 target item k+1，选择 rank semantic core 的最后一个 output token。此时 ordinal/index marker 已进入 residual stream，但 target city 的首 token 尚未生成。</p><div class="token-line">[ item k ] [ marker k+1 … <span class="query">rank-core end</span> ] <span class="pending">[ city k+1 … ]</span></div><p class="small"><code>post_marker</code> 指“在 marker core 已读入后的 state”，不是再向后偏移一个 token。边缘句点、冒号或括号属于 surface shell，不用字符串长度猜位置。</p></article></div><div class="callout"><strong>重要：</strong>token 的绝对编号随 trace 改变；固定的是语义角色。所有 head ranking 与 ablation 都读取同一个 registry 中的精确 output-token index。</div></section>

<section id="qwen"><h2>3 · Qwen3-8B：7 类 grammar，两类路由</h2><div class="model-summary"><span><strong>{qwen_event_total:,}</strong><br>parsed events</span><span><strong>{len(qwen_grammar_rows)}</strong><br>observed grammars</span><span><strong>{int(q_registry['eligible_selected_prompts'])}</strong><br>formal anchors</span><span><strong>{qwen_post_anchors}</strong><br>post-marker anchors</span></div><div class="decision"><strong>Qwen routing rule。</strong>若 target item 在 city 之前给出可解释的 ordinal/index marker，则在 <code>post_marker</code> query；其余 rank-after 或 marker-neutral item 都在上一 item 的 <code>p0_item_end</code> query。三个 rank-before grammar 共 {qwen_post_events:,} events / {qwen_post_anchors} formal anchors；其余四类共 {qwen_event_total-qwen_post_events:,} events / {int(q_registry['eligible_selected_prompts'])-qwen_post_anchors} anchors。</div>{table(['Target grammar','Surface form / 判定','Final query','精确 token 位置','Events','Anchors'], qwen_grammar_rows)}</section>

<section id="gemma"><h2>4 · Gemma4-E4B：5 类 grammar，统一 P0</h2><div class="model-summary"><span><strong>{gemma_event_total:,}</strong><br>parsed events</span><span><strong>{len(gemma_grammar_rows)}</strong><br>observed grammars</span><span><strong>{int(g_registry['eligible_selected_prompts'])}</strong><br>formal anchors</span><span><strong>1</strong><br>shared query role</span></div><div class="decision"><strong>Gemma routing rule。</strong>所有 observed target grammars 都在 <code>p0_item_end</code> 开始 query。即使 surface 上是 rank-before-city，也不把 Qwen 的 <code>post_marker</code> 规则外推给 Gemma；Gemma 的统一解释是：完成 item k 后就开始组织 needle k+1。</div>{table(['Target grammar','Surface form / 判定','Final query','精确 token 位置','Events','Anchors'], gemma_grammar_rows)}</section>

<section id="alignment"><h2>5 · 字符 grammar 如何落到精确 token</h2><p>Parser 先在原始 reasoning 文本中确定 semantic spans，再映射回模型实际生成的 output token IDs。这里不假设“一个字符片段等于一个 token”，也不把 Qwen 与 Gemma 的 tokenizer 当成相同。</p><ol><li><strong>Event span：</strong>city、rank evidence 与完整 item 都保留原始字符起止位置。</li><li><strong>Covering tokens：</strong>semantic span 对应所有与其字符发生重叠的 output tokens，保持单调顺序。</li><li><strong>Rank core：</strong>从 rank surface 中去掉只负责格式的边缘标点；<code>post_marker</code> 取最后一个覆盖 core 的 token。若 tokenizer 把 core 与标点融合，保留整个真实 token，不制造不存在的 sub-token。</li><li><strong>Item endpoint：</strong><code>p0_item_end</code> 取 <code>reasoning[:item_char_end]</code> exact prefix 的最后一个 output token。</li><li><strong>Target source：</strong>needle k+1 对应 prompt 中已注册 record 的完整 source-token span；所有 heads 都在同一个 query token 上计算对该 span 的 attention mass。</li></ol>{table(['Role','实际选择','语义保证'], [["<code>p0_item_end</code>","完整 item k 的 exact-prefix 最后一个 output token","target marker/city k+1 尚未出现"],["<code>post_marker</code>","target rank core 的最后一个 covering output token","rank k+1 已出现，city k+1 尚未出现"],["<code>target_prompt_record</code>","prompt 中 needle k+1 注册 record 的 source-token span","不同 query site 始终比较同一个 retrieval target"],["<code>generated_city_record</code>","自由生成后第一个语义 city record","先答错再纠正仍是 retrieval failure"]])}</section>

<section id="causal-link"><h2>6 · Token site、head ranking 与 ablation 使用同一位置</h2>{table(['Model','Ranking source grammar','Ranking query','Head score','Frozen bank','Applied route'], head_rows)}<div class="flow">teacher-force to the frozen query token
  → rank/freeze heads by attention to prompt needle k+1
  → zero the selected pre-O head slices at that same query
  → keep the same heads off in every later decode step
  → score the first generated semantic city record</div><p>因此，本页的 token route 不只是一个 parser annotation：它同时规定 head bank 在哪里被定义、causal intervention 从哪里开始，以及后续自由生成要回答哪个 registered needle。一个模型内部使用统一 ordered head bank，再按 grammar 把该 bank 放到对应 query role；不会为 confirmation grammar 重新选 heads。</p><div class="callout"><strong>分工：</strong>本页只定义 event、grammar 与 token site。实际 failure rate、selected-vs-random contrast 和 K-dose 结果见 causal ablation report；PCA、probe 与 counter geometry 见 Geometry Comparison。</div></section>

<section id="scope"><h2>7 · 报告边界与阅读入口</h2><div class="split"><div><h3>可以从本页得出的结论</h3><p>Qwen 与 Gemma 的 surface grammar 构成不同；Qwen 最终需要 <code>post_marker</code> / <code>p0_item_end</code> 两类 grammar route，而 Gemma 统一使用 P0。每个 route 都对应一个可复现的 output token，而不是模糊窗口。</p></div><div><h3>本页不单独证明的内容</h3><p>Parser coverage 不等于机制正确；attention mass 不等于 causal necessity；token-site 定义也不等于 counter 已被证明。相应证据分别由 causal intervention 和 representation analysis 提供。</p></div></div><p>冻结 compiler 覆盖 {manifest['plan_count']} 条 trajectories、{manifest['event_count']} 个 events 与 {manifest['transition_count']} 个 k→k+1 transitions；compile failures={audit['compile_failure_count']}，audit status=<strong>{html.escape(str(audit['status']).upper())}</strong>。N=1 没有 k→k+1 transition，因此保留在 300-prompt denominator 中但不产生 retrieval anchor。</p><div class="links"><a href="NiaH_Native-Thinking_Causal_Ablation_report.html">Causal ablation report</a><a href="NiaH_Geometry_Comparison.html">Representation geometry comparison</a><a href="v5_native_causal_site_review/causal_site_review.html">Full event registry review</a></div><details><summary>Reproducibility ledger</summary><p class="audit">Generated UTC: {html.escape(generated)}<br>Compiler manifest: {sha256(review / 'manifest.json')}<br>Compiler audit: {sha256(review / 'audit.json')}<br>Qwen routing: {sha256(args.qwen_routing)}<br>Gemma routing: {sha256(args.gemma_routing)}<br>Qwen selection: {sha256(args.qwen_selection)}<br>Gemma selection: {sha256(args.gemma_selection)}<br>Report schema: realistic_niah_v5_native_parser_token_sites_v2</p></details></section>
</main></div></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
