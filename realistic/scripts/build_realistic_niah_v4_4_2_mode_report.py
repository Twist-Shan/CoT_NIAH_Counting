from __future__ import annotations

import argparse
from pathlib import Path


EXTRA_CSS = r"""
.mode-note{background:var(--surface);border-left:4px solid var(--pink);padding:16px 20px;margin:20px 0}
.single-plot{max-width:900px;margin:0 auto}.single-plot .plot-shell canvas{height:590px}
.head-atlas-card canvas{height:440px}.head-atlas-controls{margin-top:16px}
.head-axis-note{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font:12px Consolas,monospace;margin:5px 0 0}
.attention-metric{font:12px/1.55 Consolas,monospace;color:var(--muted);margin:8px 0 14px}
"""


NONTHINKING_SECTION = r"""
<section id="nonthinking">
<h2>3 · Non-thinking：有提示 vs 无提示</h2>
<p>这里补回此前遗漏的 non-thinking 对照。User prompt 在两种条件下保持同一任务文本，唯一差别是是否保留开头的计数提示；模型不生成原生 thinking trace，因此 hidden geometry 只比较 <code>Total:</code> 处的 answer-query state。</p>
<div class="controls">
  <label>model<select id="nt-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
  <label>layer<select id="nt-layer"></select></label>
  <label>PCA view<select id="nt-basis"><option value="centered">cue-centered（比较轨迹形状）</option><option value="raw">raw pooled（包含全局 cue shift）</option></select></label>
  <label>points<select id="nt-points"><option value="all">seeds + centroids</option><option value="centroids">centroids only</option></select></label>
  <label>x<select id="nt-axis-x"><option value="0">PC1</option><option value="1">PC2</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
  <label>y<select id="nt-axis-y"><option value="1">PC2</option><option value="0">PC1</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
  <label>z<select id="nt-axis-z"><option value="2">PC3</option><option value="0">PC1</option><option value="1">PC2</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
  <label class="check"><input id="nt-pairs" type="checkbox">paired seed links</label>
  <button id="nt-reset" type="button">reset camera</button>
</div>
<div class="legend"><span><i class="shape circle"></i>cue-present · circle / solid</span><span><i class="shape square"></i>cue-absent · square / dashed</span><span>颜色 = gold count 1–10</span></div>
<figure class="plot-card single-plot"><h3>Non-thinking answer-query counter</h3><div class="plot-shell"><canvas id="nt-answer-canvas" aria-label="Non-thinking answer-query interactive 3D PCA"></canvas><div class="plot-tooltip" id="nt-answer-tooltip"></div></div><div class="plot-stats" id="nt-answer-stats"></div><figcaption>两种 prompt 在同一个 pooled PCA basis 中投影；拖拽旋转，滚轮缩放。PCA 只用于显示，显著性来自 full hidden space 的配对检验。</figcaption></figure>
<div class="conclusion" id="nt-selected-conclusion"><strong>当前层结论</strong><span></span></div>

<h3>Non-thinking 逐层 geometry 与显著性</h3>
<div class="sweep-shell"><canvas id="nt-sweep-canvas" aria-label="Non-thinking layer sweep for cue effects"></canvas><div class="sweep-note">依次显示 cue-present/absent count-centroid CKA、Δ count η²（95% seed-cluster bootstrap CI）和 count×cue interaction 的 −log10(q)。实心点表示相应逐层 BH-FDR q&lt;.05；点击 layer 会更新上方 3D 图。</div></div>
<details class="data-table" open><summary>Non-thinking 当前模型的 FDR-significant layer ranges</summary><div class="table-scroll"><table><thead><tr><th>counter</th><th>count-strength q&lt;.05</th><th>count×cue interaction q&lt;.05</th></tr></thead><tbody id="nt-sig-table"></tbody></table></div></details>
<div class="mode-note"><strong>读图边界。</strong>这部分回答“去掉前置提示怎样改变 non-thinking 的最终聚合状态”。它不能回答 trace 内部变化，因为 non-thinking 本身没有 trace token；两种 mode 的 behavior 差异见报告末尾。</div>
</section>

"""


HEAD_ATLAS_SECTION = r"""
<section id="attention-answer">
<h2>5 · Retrieval attention：按 mode 固定分数的 layer × head 大表</h2>
<p>每个格子固定为一个 decoder layer / attention head。<strong>non-thinking 按 broad retrieval score 标色</strong>；<strong>native thinking 按 trace targeted retrieval score 标色</strong>。同一 model × mode 下，cue-present 与 cue-absent 共用正值色标；第三张图为 <code>cue-absent − cue-present</code>。</p>
<div class="controls head-atlas-controls">
  <label>model<select id="head-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
  <label>mode<select id="head-mode"><option value="nonthinking">non-thinking</option><option value="native_thinking">native thinking</option></select></label>
</div>
<div class="attention-metric" id="head-atlas-caption"></div>
<div class="attention-triptych">
  <figure class="heat-card head-atlas-card"><h3>cue-present</h3><canvas id="head-atlas-present" aria-label="Cue-present layer by head attention atlas"></canvas><div class="colorbar" style="background:linear-gradient(90deg,#F7F3EA,#588BD2,#23165C)"></div><div class="bar-labels"><span>0</span><span class="head-actual-max">max</span></div><div class="head-axis-note"><span>y = head</span><span>x = layer</span></div></figure>
  <figure class="heat-card head-atlas-card"><h3>cue-absent</h3><canvas id="head-atlas-absent" aria-label="Cue-absent layer by head attention atlas"></canvas><div class="colorbar" style="background:linear-gradient(90deg,#F7F3EA,#588BD2,#23165C)"></div><div class="bar-labels"><span>0</span><span class="head-actual-max">max</span></div><div class="head-axis-note"><span>y = head</span><span>x = layer</span></div></figure>
  <figure class="heat-card head-atlas-card"><h3>absent − present</h3><canvas id="head-atlas-delta" aria-label="Cue attention difference layer by head atlas"></canvas><div class="colorbar div"></div><div class="bar-labels"><span class="head-delta-min">−max</span><span>0</span><span class="head-delta-max">+max</span></div><div class="head-axis-note"><span>y = head</span><span>x = layer</span></div></figure>
</div>
<div class="attention-hover" id="head-atlas-hover">移动鼠标查看 layer、head 和实际 retrieval score。</div>
<details class="data-table" open><summary>|absent − present| 最大的 12 个 layer / head cells</summary><div class="table-scroll"><table><thead><tr><th>layer / head</th><th>cue-present</th><th>cue-absent</th><th>absent − present</th></tr></thead><tbody id="head-change-table"></tbody></table></div></details>
<div class="callout warning"><strong>读图边界。</strong>同一 mode 的左右两图可直接比较，因为使用同一色标；non-thinking 与 native thinking 使用不同分数定义，颜色绝对值不可跨 mode 等同比较。Atlas 每格是跨 100 个样本的描述性均值，未对数百个 layer×head cells 分别作显著性声明；颜色深浅也不代表因果重要性。</div>
<div hidden aria-hidden="true"><canvas id="answer-attn-present"></canvas><canvas id="answer-attn-absent"></canvas><canvas id="answer-attn-delta"></canvas><span class="actual-max"></span><span class="delta-min"></span><span class="delta-max"></span><div id="answer-attn-hover"></div></div>
</section>

"""


def build(
    template_path: Path,
    nonthinking_geometry_path: Path,
    retrieval_score_path: Path,
    extension_script_path: Path,
    output_path: Path,
) -> None:
    html = template_path.read_text(encoding="utf-8")
    nonthinking_geometry = nonthinking_geometry_path.read_text(encoding="utf-8")
    retrieval_score = retrieval_score_path.read_text(encoding="utf-8")
    extension_script = extension_script_path.read_text(encoding="utf-8")
    extension_script = extension_script.replace(
        "@@NT_GEOMETRY_DATA@@", nonthinking_geometry
    ).replace("@@RETRIEVAL_SCORE_DATA@@", retrieval_score)

    required_markers = [
        "</style>",
        '<section id="joint">',
        '<section id="attention-answer">',
        '<section id="attention-trace">',
        "</script>",
    ]
    missing = [marker for marker in required_markers if marker not in html]
    if missing:
        raise RuntimeError(f"Template markers missing: {missing}")

    html = html.replace("</style>", EXTRA_CSS + "\n</style>", 1)
    html = html.replace(
        '<nav><a href="#scope">结论</a><a href="#geometry">Trace / Answer 3D</a><a href="#layers">逐层显著性</a><a href="#joint">Joint geometry</a><a href="#attention-answer">Answer attention</a><a href="#attention-trace">Trace attention</a><a href="#methods">统计定义</a></nav>',
        '<nav><a href="#scope">结论</a><a href="#geometry">Native 3D</a><a href="#nonthinking">Non-thinking 3D</a><a href="#joint">Joint geometry</a><a href="#attention-answer">Head attention atlas</a><a href="#attention-trace">Trace attention</a><a href="#mode-summary">Thinking vs non-thinking</a><a href="#methods">统计定义</a></nav>',
        1,
    )
    html = html.replace(
        '<div class="eyebrow">Realistic NIAH · V4.4.2 · native thinking</div>',
        '<div class="eyebrow">Realistic NIAH · V4.4.2 · two modes × two prompts</div>',
        1,
    )
    html = html.replace(
        '<h1>Trace 与 Answer Counter：有无前置提示的 3D Geometry 与 Attention</h1>',
        '<h1>有无前置提示如何改变 Non-thinking 与 Native Thinking</h1>',
        1,
    )
    html = html.replace(
        '<p class="lead">在同一共享 PCA 坐标中叠加 cue-present 与 cue-absent，并用全空间配对检验判断“看起来不同”是否对应可复现的 count-geometry 变化。Attention 同时给出两种 prompt 的原始 map 和 absent−present 差值。</p>',
        '<p class="lead">分别在 non-thinking 与 native thinking 内比较 cue-present 和 cue-absent：hidden state 用共享 PCA 显示并用 full-space 配对检验判断显著性；attention 恢复为旧版易读的 layer × head atlas，一次只看一个分数。</p>',
        1,
    )
    html = html.replace(
        '<section id="geometry">\n<h2>1 · Trace 与 Answer counter 的共享 3D PCA</h2>',
        '<section id="geometry">\n<h2>1 · Native thinking：Trace 与 Answer counter 的共享 3D PCA</h2>',
        1,
    )
    html = html.replace(
        '<div class="callout warning"><strong>如何判断显著性。</strong>',
        '<div class="mode-note"><strong>Non-thinking 已补齐。</strong>Answer-query 的 count×cue interaction 在 Qwen 32/36 层、Gemma 34/42 层通过逐层 FDR；count-strength 改变分别在 11/36 与 15/42 层显著。下面给出独立的 3D 与逐层检验，而不是把 non-thinking 混入 native thinking 的 PCA。</div>\n<div class="callout warning"><strong>如何判断显著性。</strong>',
        1,
    )
    html = html.replace(
        '<section id="layers">\n<h2>2 · 逐层 geometry 与显著性</h2>',
        '<section id="layers">\n<h2>2 · Native thinking 逐层 geometry 与显著性</h2>',
        1,
    )
    html = html.replace(
        '<section id="joint">',
        NONTHINKING_SECTION + '<section id="joint">',
        1,
    )
    html = html.replace(
        '<section id="joint">\n<h2>3 · Role-centered joint geometry</h2>',
        '<section id="joint">\n<h2>4 · Native trace / answer role-centered joint geometry</h2>',
        1,
    )
    attention_start = html.index('<section id="attention-answer">')
    trace_start = html.index('<section id="attention-trace">')
    html = html[:attention_start] + HEAD_ATLAS_SECTION + html[trace_start:]
    html = html.replace(
        '<section id="attention-trace">\n<h2>5 · Trace attention：当前 layer 的时间结构</h2>',
        '<section id="attention-trace">\n<h2>6 · Native trace attention：当前 layer 的时间结构</h2>',
        1,
    )
    html = html.replace(
        '<section id="methods">\n<h2>6 · 统计定义与解释边界</h2>',
        '<section id="methods">\n<h2>8 · 统计定义与解释边界</h2>',
        1,
    )

    mode_summary = r"""
<section id="mode-summary">
<h2>7 · Thinking vs non-thinking：行为层总结</h2>
<div class="summary-grid">
  <div class="summary"><strong>Qwen3-8B · cue-present</strong><span>non-thinking 55% → native thinking 96%，thinking gain +41 pp。</span></div>
  <div class="summary"><strong>Qwen3-8B · cue-absent</strong><span>non-thinking 37% → native thinking 86%，thinking gain +49 pp。</span></div>
  <div class="summary"><strong>Gemma4-E4B · cue-present</strong><span>non-thinking 39% → native thinking 82%，thinking gain +43 pp。</span></div>
  <div class="summary"><strong>Gemma4-E4B · cue-absent</strong><span>non-thinking 42% → native thinking 75%，thinking gain +33 pp。</span></div>
</div>
<p>提示移除效应必须在 mode 内解释：Qwen non-thinking 为 −18 pp，native thinking 为 −10 pp；Gemma non-thinking 为 +3 pp，native thinking 为 −7 pp。前者说明 Qwen 的原生 thinking 对提示缺失有部分补偿；Gemma 的两个 cue effect 置信区间都跨零，因此不能把符号差异当成稳定交互。</p>
<div class="callout warning"><strong>跨 mode 的 geometry 边界。</strong>Non-thinking 与 native thinking 的 answer-query 图分别拟合 PCA，所以两张 3D 图的绝对坐标不能直接相减。可靠比较应优先使用各自 mode 内的 cue effect、full-space layer statistics，以及同一 attention head / score 的 atlas。</div>
</section>

"""
    html = html.replace('<section id="methods">', mode_summary + '<section id="methods">', 1)
    html = html.replace("</script>", "\n" + extension_script + "\n</script>", 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output_path)
    print(output_path)
    print(output_path.stat().st_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--nonthinking-geometry", type=Path, required=True)
    parser.add_argument("--retrieval-score-atlas", type=Path, required=True)
    parser.add_argument("--extension-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(
        arguments.template,
        arguments.nonthinking_geometry,
        arguments.retrieval_score_atlas,
        arguments.extension_script,
        arguments.output,
    )
