from __future__ import annotations

import argparse
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Realistic NIAH V4.4.2 · Counter Geometry & Attention</title>
<style>
:root{--paper:#F3EEE4;--surface:#FFFDF8;--ink:#20242D;--muted:#5E6672;--line:#C9C2B6;--indigo:#23165C;--violet:#6750E8;--cyan:#00A9D8;--teal:#00A88F;--green:#2DBE77;--pink:#D94B86;--yellow:#D6B52C;--red:#B53D66}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;line-height:1.62}
nav{position:sticky;top:0;z-index:5;background:rgba(243,238,228,.96);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:18px;flex-wrap:wrap}
nav a{color:var(--indigo);text-decoration:none;font-size:14px;font-weight:650}
main{max-width:1220px;margin:0 auto;padding:36px 28px 80px}header{max-width:980px;padding:24px 0 28px;border-bottom:2px solid var(--ink)}
.eyebrow{font:700 12px/1.2 Consolas,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}
h1{font-size:40px;line-height:1.08;letter-spacing:-.035em;margin:10px 0 18px}h2{font-size:28px;line-height:1.22;letter-spacing:-.02em;margin:0 0 14px}h3{font-size:20px;line-height:1.3;margin:28px 0 8px}
p{max-width:92ch;margin:10px 0 16px}.lead{font-size:18px;color:#3E4651}.meta{font:12px/1.6 Consolas,monospace;color:var(--muted)}
section{padding:46px 0;border-bottom:1px solid var(--line)}.callout,.conclusion{background:var(--surface);border-left:4px solid var(--teal);padding:16px 20px;margin:20px 0}.warning{border-left-color:var(--yellow)}.conclusion strong:first-child{display:block;color:var(--indigo);margin-bottom:5px}
.summary-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:20px 0}.summary{background:var(--surface);border:1px solid var(--line);padding:15px}.summary strong{display:block;color:var(--indigo);margin-bottom:5px}.summary span{font-size:13px;color:var(--muted)}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 12px}.controls label{font-size:12px;color:var(--muted);display:grid;gap:3px}.controls select,.controls button{font:13px "Segoe UI",sans-serif;color:var(--ink);background:#FAF7F0;border:1px solid #AAA195;border-radius:4px;padding:6px 9px}.controls button:active{transform:translateY(1px)}
.check{display:flex!important;align-items:center;gap:6px!important;align-self:end;padding:6px 2px}.check input{accent-color:var(--indigo)}
.two-plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.plot-card{margin:0;background:var(--surface);border:1px solid var(--line);padding:12px}.plot-card h3{margin:0 0 8px}
.plot-shell{position:relative;background:#15112B;border-radius:4px;overflow:hidden}.plot-shell canvas{display:block;width:100%;height:550px;touch-action:none}.plot-tooltip{position:absolute;display:none;pointer-events:none;background:rgba(255,253,248,.97);color:var(--ink);border:1px solid var(--line);padding:8px 10px;font-size:12px;max-width:280px;z-index:3}
.plot-stats{font:12px/1.55 Consolas,monospace;color:var(--muted);margin-top:8px;min-height:76px}.sig{color:#007F71;font-weight:700}.nonsig{color:var(--muted)}figcaption{font-size:13px;color:var(--muted);margin-top:10px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin:10px 0}.legend span{display:inline-flex;align-items:center;gap:5px}.shape{display:inline-block;width:10px;height:10px;border:2px solid}.circle{border-radius:50%;border-color:#FFFDF8;background:var(--violet)}.square{border-color:#F6E36A;background:var(--violet)}
.sweep-shell{position:relative;background:var(--surface);border:1px solid var(--line);padding:10px}.sweep-shell canvas{display:block;width:100%;height:430px}.sweep-note{font:12px/1.55 Consolas,monospace;color:var(--muted);margin-top:7px}
.joint-shell canvas{height:590px}
.attention-triptych{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.heat-card{margin:0;background:var(--surface);border:1px solid var(--line);padding:10px}.heat-card h3{font-size:16px;margin:0 0 7px}.heat-card canvas{display:block;width:100%;height:410px;background:#15112B}.heat-card.compact canvas{height:340px}.heat-card.square-map canvas{height:auto;aspect-ratio:1/1}
.colorbar{height:10px;margin-top:8px;border:1px solid var(--line)}.seq{background:linear-gradient(90deg,#15112B,#2546A8,#00A9D8,#F6E36A)}.div{background:linear-gradient(90deg,#315BC7,#F3EEE4,#D94B86)}.bar-labels{display:flex;justify-content:space-between;font:11px Consolas,monospace;color:var(--muted);margin-top:2px}
.attention-hover{font:12px/1.55 Consolas,monospace;color:var(--muted);min-height:20px;margin:8px 0}
details.data-table{background:var(--surface);border:1px solid var(--line);margin:16px 0}details.data-table summary{cursor:pointer;padding:12px 14px;font-weight:650;color:var(--indigo)}.table-scroll{overflow:auto;border-top:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 11px;border-bottom:1px solid #DED8CE;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#ECE6DA;color:#303744}tbody tr:hover{background:#FAF6EE}
code{font-family:Consolas,monospace;background:#EAE4D8;padding:1px 4px}.small{font-size:13px;color:var(--muted)}
@media(max-width:900px){.two-plots,.attention-triptych,.summary-grid{grid-template-columns:1fr}.plot-shell canvas{height:500px}.heat-card canvas{height:460px}}
@media(max-width:620px){main{padding:24px 12px 60px}h1{font-size:31px}nav{gap:10px}.plot-shell canvas{height:440px}}
</style>
</head>
<body>
<nav><a href="#scope">结论</a><a href="#geometry">Trace / Answer 3D</a><a href="#layers">逐层显著性</a><a href="#joint">Joint geometry</a><a href="#attention-answer">Answer attention</a><a href="#attention-trace">Trace attention</a><a href="#methods">统计定义</a></nav>
<main>
<header>
<div class="eyebrow">Realistic NIAH · V4.4.2 · native thinking</div>
<h1>Trace 与 Answer Counter：有无前置提示的 3D Geometry 与 Attention</h1>
<p class="lead">在同一共享 PCA 坐标中叠加 cue-present 与 cue-absent，并用全空间配对检验判断“看起来不同”是否对应可复现的 count-geometry 变化。Attention 同时给出两种 prompt 的原始 map 和 absent−present 差值。</p>
<p class="meta">2 models · 10 count buckets · 10 paired seeds · Qwen 36 layers · Gemma 42 layers · 999 permutations · 1,000 seed-cluster bootstraps</p>
</header>

<section id="scope">
<h2>先看统计结论，再旋转 3D 图</h2>
<div class="summary-grid">
  <div class="summary"><strong>Answer counter 对提示敏感</strong><span>count×cue interaction：Qwen 31/36 层、Gemma 13/42 层在逐层 FDR 后显著。</span></div>
  <div class="summary"><strong>Trace mean 保留稳定 counter</strong><span>Gemma centroid CKA 0.94–0.98；Qwen 多数层也保留结构，但提示会改变其 count-specific arrangement。</span></div>
  <div class="summary"><strong>Trace last 不是稳定的主指标</strong><span>其 centroid CKA 较低且 seed variance 大；两个模型均无逐层 FDR 显著 interaction，故报告同时提供 trace mean sensitivity。</span></div>
</div>
<div class="callout warning"><strong>如何判断显著性。</strong>3D PCA 只用于显示。主检验在完整 hidden dimension 中进行：一是 count-strength η² 是否改变；二是配对 hidden delta 是否随 count 系统变化（count×cue interaction）。两者都对 layer sweep 做 BH-FDR。Centroid CKA 描述轨迹保持程度，但不单独提供显著性结论。</div>
</section>

<section id="geometry">
<h2>1 · Trace 与 Answer counter 的共享 3D PCA</h2>
<p>Trace 与 answer 各自使用独立但 cue-shared 的 PCA basis：同一 counter 的 cue-present 和 cue-absent 能直接比较；左右图的绝对 PC 方向不可互相比。默认层来自先前 V4.1 answer-manifold landmark，避免以当前 cue effect 挑选最好看的层。</p>
<div class="controls">
  <label>model<select id="model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
  <label>layer<select id="layer"></select></label>
  <label>trace representation<select id="trace-site"><option value="trace_mean">trace mean（主 sensitivity）</option><option value="trace_last">last trace token</option></select></label>
  <label>PCA view<select id="basis"><option value="centered">cue-centered（比较轨迹形状）</option><option value="raw">raw pooled（包含全局 cue shift）</option></select></label>
  <label>points<select id="points"><option value="all">seeds + centroids</option><option value="centroids">centroids only</option></select></label>
  <label>x<select id="axis-x"><option value="0">PC1</option><option value="1">PC2</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
  <label>y<select id="axis-y"><option value="1">PC2</option><option value="0">PC1</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
  <label>z<select id="axis-z"><option value="2">PC3</option><option value="0">PC1</option><option value="1">PC2</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
  <label class="check"><input id="pairs" type="checkbox">paired seed links</label>
  <button id="reset" type="button">reset cameras</button>
</div>
<div class="legend"><span><i class="shape circle"></i>cue-present · circle / solid</span><span><i class="shape square"></i>cue-absent · square / dashed</span><span>颜色 = gold count 1…10</span></div>
<div class="two-plots">
  <figure class="plot-card"><h3 id="trace-title">Trace counter</h3><div class="plot-shell"><canvas id="trace-canvas" aria-label="Trace counter interactive 3D PCA"></canvas><div class="plot-tooltip" id="trace-tooltip"></div></div><div class="plot-stats" id="trace-stats"></div><figcaption>拖拽旋转，滚轮缩放。Trace mean 对所有保存的 trace query states 取均值；trace last 使用最后一个保存的 trace query state。</figcaption></figure>
  <figure class="plot-card"><h3>Answer-query counter</h3><div class="plot-shell"><canvas id="answer-canvas" aria-label="Answer query counter interactive 3D PCA"></canvas><div class="plot-tooltip" id="answer-tooltip"></div></div><div class="plot-stats" id="answer-stats"></div><figcaption>Answer state 是生成 <code>Total:</code> 时最后一个 answer-query hidden state；不是 answer token，也不是最后一层。</figcaption></figure>
</div>
<div class="conclusion" id="selected-conclusion"><strong>当前层结论</strong><span></span></div>
</section>

<section id="layers">
<h2>2 · 逐层 geometry 与显著性</h2>
<p>三个 panel 同时显示 trace（当前选择为 trace mean 或 trace last）与 answer。点击任意 layer 可同步上方 3D 和下方 attention。</p>
<div class="sweep-shell"><canvas id="sweep-canvas" aria-label="Layer sweep for CKA, count strength and interaction significance"></canvas><div class="sweep-note">Panel 1：cue-present/absent count centroids 的 full-space linear CKA。Panel 2：Δ count η² = absent−present，误差线为 seed-cluster bootstrap 95% CI，实心点表示 strength q&lt;.05。Panel 3：count×cue interaction 的 −log10(q)，虚线为 q=.05。</div></div>
<details class="data-table"><summary>当前模型的 FDR-significant layer ranges</summary><div class="table-scroll"><table><thead><tr><th>counter</th><th>count-strength q&lt;.05</th><th>count×cue interaction q&lt;.05</th></tr></thead><tbody id="sig-table"></tbody></table></div></details>
</section>

<section id="joint">
<h2>3 · Role-centered joint geometry</h2>
<p>仿照之前 V4.4 HTML 的 prompt/answer joint view：先分别减去 trace/answer × cue 的 role mean，再拟合一个 shared PCA，只画四条 count-centroid trajectories。这样能比较两种 counter 的形状，但刻意去除了 token-role 与 cue 的固定全局位移。</p>
<div class="controls"><label>x<select id="joint-x"><option value="0">PC1</option><option value="1">PC2</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label><label>y<select id="joint-y"><option value="1">PC2</option><option value="0">PC1</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label><label>z<select id="joint-z"><option value="2">PC3</option><option value="0">PC1</option><option value="1">PC2</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label><button id="joint-reset" type="button">reset camera</button></div>
<div class="legend"><span>trace present ○ solid white</span><span>trace absent □ dashed yellow</span><span>answer present △ solid cyan</span><span>answer absent ◇ dashed pink</span></div>
<figure class="plot-card"><div class="plot-shell joint-shell"><canvas id="joint-canvas" aria-label="Joint trace and answer counter geometry"></canvas></div><div class="plot-stats" id="joint-stats"></div><figcaption>颜色仍表示 gold count；线型/形状区分 counter site 与 prompt variant。</figcaption></figure>
</section>

<section id="attention-answer">
<h2>4 · Answer-query attention：左右原始图 + 差值</h2>
<p>每个 heatmap 的纵轴是全部 decoder layers，横轴是 key region。cue-present 与 cue-absent 使用相同正值色标；差值为 absent−present，使用对称色标。</p>
<div class="attention-triptych">
  <figure class="heat-card"><h3>cue-present</h3><canvas id="answer-attn-present" aria-label="Answer attention cue present"></canvas><div class="colorbar seq"></div><div class="bar-labels"><span>0</span><span class="actual-max">max</span></div></figure>
  <figure class="heat-card"><h3>cue-absent</h3><canvas id="answer-attn-absent" aria-label="Answer attention cue absent"></canvas><div class="colorbar seq"></div><div class="bar-labels"><span>0</span><span class="actual-max">max</span></div></figure>
  <figure class="heat-card"><h3>absent − present</h3><canvas id="answer-attn-delta" aria-label="Answer attention difference"></canvas><div class="colorbar div"></div><div class="bar-labels"><span class="delta-min">−max</span><span>0</span><span class="delta-max">+max</span></div></figure>
</div>
<div class="attention-hover" id="answer-attn-hover">移动鼠标查看 layer、region 和 attention mass。</div>
</section>

<section id="attention-trace">
<h2>5 · Trace attention：当前 layer 的时间结构</h2>
<h3>5.1 Trace query time × key region</h3>
<p>纵轴是 64 个归一化 trace-query time bins，横轴是 key region。该图与上方 model/layer 控件联动。</p>
<div class="attention-triptych">
  <figure class="heat-card compact"><h3>cue-present</h3><canvas id="trace-attn-present" aria-label="Trace attention cue present"></canvas><div class="colorbar seq"></div><div class="bar-labels"><span>0</span><span class="trace-actual-max">max</span></div></figure>
  <figure class="heat-card compact"><h3>cue-absent</h3><canvas id="trace-attn-absent" aria-label="Trace attention cue absent"></canvas><div class="colorbar seq"></div><div class="bar-labels"><span>0</span><span class="trace-actual-max">max</span></div></figure>
  <figure class="heat-card compact"><h3>absent − present</h3><canvas id="trace-attn-delta" aria-label="Trace attention difference"></canvas><div class="colorbar div"></div><div class="bar-labels"><span class="trace-delta-min">−max</span><span>0</span><span class="trace-delta-max">+max</span></div></figure>
</div>
<div class="attention-hover" id="trace-attn-hover">移动鼠标查看 trace time、region 和 attention mass。</div>

<h3>5.2 Trace-to-trace attention</h3>
<p>纵轴是 trace query time，横轴是被回看的 trace key time；为控制 HTML 体积，原始 128×128 map 按计数加权降采样为 32×32。</p>
<div class="attention-triptych">
  <figure class="heat-card square-map"><h3>cue-present</h3><canvas id="ttt-present" aria-label="Trace to trace attention cue present"></canvas><div class="colorbar seq"></div><div class="bar-labels"><span>0</span><span class="ttt-actual-max">max</span></div></figure>
  <figure class="heat-card square-map"><h3>cue-absent</h3><canvas id="ttt-absent" aria-label="Trace to trace attention cue absent"></canvas><div class="colorbar seq"></div><div class="bar-labels"><span>0</span><span class="ttt-actual-max">max</span></div></figure>
  <figure class="heat-card square-map"><h3>absent − present</h3><canvas id="ttt-delta" aria-label="Trace to trace attention difference"></canvas><div class="colorbar div"></div><div class="bar-labels"><span class="ttt-delta-min">−max</span><span>0</span><span class="ttt-delta-max">+max</span></div></figure>
</div>
<div class="attention-hover" id="ttt-hover">移动鼠标查看 trace query/key time 和 attention mass。</div>
<div class="callout warning"><strong>Gemma 的零 attention 需要按架构解释。</strong>Gemma 全层使用 512-token sliding window；后期 query 对远处 cue/needle 的质量严格为零。这是可见性限制，不等于 cue 没有通过 earlier hidden states 产生间接影响。</div>
</section>

<section id="methods">
<h2>6 · 统计定义与解释边界</h2>
<details class="data-table" open><summary>三个 geometry 指标</summary><div class="table-scroll"><table><thead><tr><th>指标</th><th>含义</th><th>推断</th></tr></thead><tbody>
<tr><td>Centroid linear CKA</td><td>比较 cue-present 与 cue-absent 的 10 个 count centroids 的 pairwise geometry；1 表示高度保持。</td><td>描述性；不单独声称显著。</td></tr>
<tr><td>Count strength η²</td><td>full hidden space 中由 count bucket 解释的总变异比例。报告 absent−present。</td><td>paired cue-label permutation p；seed-cluster bootstrap CI；按 model/site 跨层 BH-FDR。</td></tr>
<tr><td>Count×cue interaction</td><td>检验配对 hidden delta 是否依赖 count；排除“所有 count 只发生同一个全局平移”的解释。</td><td>count labels 在每个 seed 内置换；999 permutations；按 model/site 跨层 BH-FDR。</td></tr>
</tbody></table></div></details>
<p class="small">PCA 由每个 model/site/layer 的 200 个状态（100 paired stimuli × 2 cue conditions）共同拟合。cue-centered view 先分别减去两个 condition mean；raw view 保留全局 shift。因为 PCA fit 使用当前数据，图是探索性 display，显著性只来自 full-space tests。Attention weight 描述信息路由机会，不等同于 causal importance。</p>
</section>
</main>

<script>
const GEOM=@@GEOMETRY_DATA@@;
const ATTN=@@ATTENTION_DATA@@;
const COUNT_COLORS=['#23165C','#3D2A90','#5140C8','#6750E8','#4B7BE5','#00A9D8','#00A88F','#2DBE77','#D6B52C','#D94B86'];
const controls={model:document.getElementById('model'),layer:document.getElementById('layer'),traceSite:document.getElementById('trace-site'),basis:document.getElementById('basis'),points:document.getElementById('points'),pairs:document.getElementById('pairs'),x:document.getElementById('axis-x'),y:document.getElementById('axis-y'),z:document.getElementById('axis-z')};
const cameras={trace:{yaw:-.72,pitch:.42,zoom:1},answer:{yaw:-.72,pitch:.42,zoom:1},joint:{yaw:-.72,pitch:.42,zoom:1}};
const charts={};
function clamp(v,a,b){return Math.max(a,Math.min(b,v))}
function format(v,d=3){return Number.isFinite(+v)?(+v).toFixed(d):'NA'}
function pformat(v){if(!Number.isFinite(+v))return'NA';if(+v<.001)return'<.001';return(+v).toFixed(3)}
function ranges(values){if(!values.length)return'none';const xs=[...new Set(values.map(Number))].sort((a,b)=>a-b),out=[];let a=xs[0],b=a;for(let i=1;i<xs.length;i++){if(xs[i]===b+1)b=xs[i];else{out.push(a===b?'L'+a:'L'+a+'–'+b);a=b=xs[i]}}out.push(a===b?'L'+a:'L'+a+'–'+b);return out.join(', ')}
function activeKey(site){return controls.model.value+'|'+site+'|'+controls.layer.value}
function rowCoords(row,condition,basis){let start;if(basis==='raw')start=condition==='cue_present'?4:10;else start=condition==='cue_present'?16:22;return row.slice(start,start+6)}
function resizeCanvas(canvas){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(320,Math.round(rect.width)),h=Math.max(260,Math.round(rect.height));if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return{ctx,w,h}}
function transformFactory(points,axes,w,h,camera){const vals=axes.map((axis)=>points.map(p=>p[axis]));const mins=vals.map(v=>Math.min(...v)),maxs=vals.map(v=>Math.max(...v));const center=mins.map((v,i)=>(v+maxs[i])/2),range=Math.max(...mins.map((v,i)=>Math.max(maxs[i]-v,1e-8))),radius=Math.min(w,h)*.38*camera.zoom;return p=>{let x=(p[axes[0]]-center[0])*2/range,y=(p[axes[1]]-center[1])*2/range,z=(p[axes[2]]-center[2])*2/range;const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;return{x:w/2+x1*radius,y:h/2-y1*radius,z:z2}}}
function centroids(rows,condition,basis){const result=[];for(let count=1;count<=10;count++){const chosen=rows.filter(r=>r[1]===count).map(r=>rowCoords(r,condition,basis));result.push(chosen[0].map((_,axis)=>chosen.reduce((s,p)=>s+p[axis],0)/chosen.length))}return result}
function statHtml(stat,evr){const strength=+stat.count_eta_q<.05?'sig':'nonsig',interaction=+stat.interaction_q<.05?'sig':'nonsig';return '<strong>EVR PC1–3 '+format(100*evr.slice(0,3).reduce((a,b)=>a+b,0),1)+'%</strong> · centroid CKA '+format(stat.centroid_cka)+'<br>count η² '+format(stat.count_eta_present)+' → '+format(stat.count_eta_absent)+' · Δ '+format(stat.count_eta_delta)+' ['+format(stat.count_eta_delta_ci_low)+', '+format(stat.count_eta_delta_ci_high)+'] · <span class="'+strength+'">q '+pformat(stat.count_eta_q)+'</span><br>count×cue interaction η² '+format(stat.interaction_eta_sq)+' · <span class="'+interaction+'">q '+pformat(stat.interaction_q)+'</span>'}
function drawGeometry(name,site){const canvas=document.getElementById(name+'-canvas'),tooltip=document.getElementById(name+'-tooltip'),dataset=GEOM.datasets[activeKey(site)],stat=GEOM.statistics[activeKey(site)];if(!dataset)return;const {ctx,w,h}=resizeCanvas(canvas),basis=controls.basis.value,axes=[+controls.x.value,+controls.y.value,+controls.z.value],rows=dataset.rows,all=[];for(const row of rows){all.push(rowCoords(row,'cue_present',basis),rowCoords(row,'cue_absent',basis))}const tf=transformFactory(all,axes,w,h,cameras[name]);ctx.fillStyle='#15112B';ctx.fillRect(0,0,w,h);ctx.strokeStyle='rgba(255,255,255,.10)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(w*.08,h/2);ctx.lineTo(w*.92,h/2);ctx.moveTo(w/2,h*.08);ctx.lineTo(w/2,h*.92);ctx.stroke();const screen=[];if(controls.pairs.checked){ctx.strokeStyle='rgba(255,255,255,.10)';ctx.lineWidth=.8;for(const row of rows){const p=tf(rowCoords(row,'cue_present',basis)),a=tf(rowCoords(row,'cue_absent',basis));ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(a.x,a.y);ctx.stroke()}}if(controls.points.value==='all'){const items=[];for(const row of rows){for(const condition of ['cue_present','cue_absent']){items.push({row,condition,q:tf(rowCoords(row,condition,basis))})}}items.sort((a,b)=>a.q.z-b.q.z);for(const item of items){const {row,condition,q}=item,count=row[1],correct=condition==='cue_present'?row[2]:row[3];ctx.globalAlpha=correct?.58:.22;ctx.fillStyle=COUNT_COLORS[count-1];ctx.strokeStyle=condition==='cue_present'?'#FFFDF8':'#F6E36A';ctx.lineWidth=condition==='cue_present'?1.1:1.4;if(condition==='cue_present'){ctx.beginPath();ctx.arc(q.x,q.y,3,0,Math.PI*2);ctx.fill();ctx.stroke()}else{ctx.fillRect(q.x-3,q.y-3,6,6);ctx.strokeRect(q.x-3,q.y-3,6,6)}screen.push({x:q.x,y:q.y,row,condition})}ctx.globalAlpha=1}
for(const condition of ['cue_present','cue_absent']){const cs=centroids(rows,condition,basis).map((p,i)=>({count:i+1,p,q:tf(p)}));ctx.strokeStyle=condition==='cue_present'?'#FFFDF8':'#F6E36A';ctx.lineWidth=2.6;ctx.setLineDash(condition==='cue_present'?[]:[7,5]);ctx.beginPath();cs.forEach((item,i)=>i?ctx.lineTo(item.q.x,item.q.y):ctx.moveTo(item.q.x,item.q.y));ctx.stroke();ctx.setLineDash([]);for(const item of cs){ctx.fillStyle=COUNT_COLORS[item.count-1];ctx.strokeStyle=condition==='cue_present'?'#FFFDF8':'#F6E36A';ctx.lineWidth=1.4;if(condition==='cue_present'){ctx.beginPath();ctx.arc(item.q.x,item.q.y,6,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#FFFDF8';ctx.font='11px Consolas';ctx.fillText(String(item.count),item.q.x+7,item.q.y-7)}else{ctx.fillRect(item.q.x-5,item.q.y-5,10,10);ctx.strokeRect(item.q.x-5,item.q.y-5,10,10)}}}
charts[name]={canvas,tooltip,screen,site};const evr=basis==='raw'?dataset.evr_raw:dataset.evr_cue_centered;document.getElementById(name+'-stats').innerHTML=statHtml(stat,evr)}
function attachGeometry(name){const canvas=document.getElementById(name+'-canvas'),tooltip=document.getElementById(name+'-tooltip');let drag=false,lx=0,ly=0;canvas.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointermove',e=>{if(drag){cameras[name].yaw+=(e.clientX-lx)*.008;cameras[name].pitch=clamp(cameras[name].pitch+(e.clientY-ly)*.008,-1.45,1.45);lx=e.clientX;ly=e.clientY;drawAll();return}const chart=charts[name],rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=null,dist=Infinity;for(const p of chart.screen||[]){const d=(p.x-x)**2+(p.y-y)**2;if(d<dist){dist=d;best=p}}if(best&&dist<90){tooltip.style.display='block';tooltip.style.left=Math.min(rect.width-260,x+13)+'px';tooltip.style.top=Math.max(8,y-12)+'px';const correct=best.condition==='cue_present'?best.row[2]:best.row[3];tooltip.innerHTML='<strong>count '+best.row[1]+' · seed '+best.row[0]+'</strong><br>'+best.condition+' · '+(correct?'correct':'wrong')+'<br>'+chart.site+' · L'+controls.layer.value}else tooltip.style.display='none'});canvas.addEventListener('wheel',e=>{e.preventDefault();cameras[name].zoom=clamp(cameras[name].zoom*Math.exp(-e.deltaY*.001),.55,2.8);drawAll()},{passive:false})}
function drawJoint(){const d=GEOM.joint[controls.model.value+'|'+controls.layer.value];if(!d)return;const canvas=document.getElementById('joint-canvas'),{ctx,w,h}=resizeCanvas(canvas),axes=[+document.getElementById('joint-x').value,+document.getElementById('joint-y').value,+document.getElementById('joint-z').value],points=d.rows.map(r=>r.slice(3,9)),tf=transformFactory(points,axes,w,h,cameras.joint);ctx.fillStyle='#15112B';ctx.fillRect(0,0,w,h);const specs=[['trace_last','cue_present','#FFFDF8',[],'circle'],['trace_last','cue_absent','#F6E36A',[7,5],'square'],['answer_query','cue_present','#00A9D8',[],'triangle'],['answer_query','cue_absent','#D94B86',[7,5],'diamond']];for(const [site,condition,color,dash,shape] of specs){const rows=d.rows.filter(r=>r[0]===site&&r[1]===condition).sort((a,b)=>a[2]-b[2]).map(r=>({r,q:tf(r.slice(3,9))}));ctx.strokeStyle=color;ctx.lineWidth=2.4;ctx.setLineDash(dash);ctx.beginPath();rows.forEach((x,i)=>i?ctx.lineTo(x.q.x,x.q.y):ctx.moveTo(x.q.x,x.q.y));ctx.stroke();ctx.setLineDash([]);for(const item of rows){const x=item.q.x,y=item.q.y,count=item.r[2];ctx.fillStyle=COUNT_COLORS[count-1];ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.beginPath();if(shape==='circle')ctx.arc(x,y,6,0,Math.PI*2);else if(shape==='square')ctx.rect(x-5,y-5,10,10);else if(shape==='triangle'){ctx.moveTo(x,y-6);ctx.lineTo(x+6,y+5);ctx.lineTo(x-6,y+5);ctx.closePath()}else{ctx.moveTo(x,y-7);ctx.lineTo(x+7,y);ctx.lineTo(x,y+7);ctx.lineTo(x-7,y);ctx.closePath()}ctx.fill();ctx.stroke()}}document.getElementById('joint-stats').innerHTML='<strong>'+d.model+' · L'+d.layer+'</strong> · shared role-centered EVR PC1–3 '+format(100*d.evr.slice(0,3).reduce((a,b)=>a+b,0),1)+'%<br>trace↔answer centroid CKA · cue-present '+format(d.trace_answer_cka_present)+' · cue-absent '+format(d.trace_answer_cka_absent)}
function attachJoint(){const canvas=document.getElementById('joint-canvas');let drag=false,lx=0,ly=0;canvas.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointermove',e=>{if(!drag)return;cameras.joint.yaw+=(e.clientX-lx)*.008;cameras.joint.pitch=clamp(cameras.joint.pitch+(e.clientY-ly)*.008,-1.45,1.45);lx=e.clientX;ly=e.clientY;drawJoint()});canvas.addEventListener('wheel',e=>{e.preventDefault();cameras.joint.zoom=clamp(cameras.joint.zoom*Math.exp(-e.deltaY*.001),.55,2.8);drawJoint()},{passive:false})}
function drawSweep(){const canvas=document.getElementById('sweep-canvas'),{ctx,w,h}=resizeCanvas(canvas),model=controls.model.value,sites=[controls.traceSite.value,'answer_query'],colors=['#00A88F','#D94B86'],layers=ATTN.models[model].layers,pad={l:54,r:22,t:25,b:34},gap=24,panelH=(h-pad.t-pad.b-gap*2)/3,x=l=>pad.l+(w-pad.l-pad.r)*(l-layers[0])/(layers[layers.length-1]-layers[0]);ctx.fillStyle='#FFFDF8';ctx.fillRect(0,0,w,h);const panels=[{title:'Centroid CKA',min:0,max:1,value:s=>s.centroid_cka},{title:'Δ count η² (absent−present)',min:-.25,max:.25,value:s=>s.count_eta_delta},{title:'−log10 interaction q',min:0,max:4,value:s=>Math.min(4,-Math.log10(Math.max(+s.interaction_q,1e-4)))}];panels.forEach((panel,pi)=>{const top=pad.t+pi*(panelH+gap),bottom=top+panelH,y=v=>bottom-(bottom-top)*(v-panel.min)/(panel.max-panel.min);ctx.strokeStyle='#C9C2B6';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.l,top);ctx.lineTo(pad.l,bottom);ctx.lineTo(w-pad.r,bottom);ctx.stroke();ctx.fillStyle='#20242D';ctx.font='12px Segoe UI';ctx.fillText(panel.title,pad.l,top-8);for(let t=0;t<=4;t++){const value=panel.min+(panel.max-panel.min)*t/4,yy=y(value);ctx.strokeStyle='rgba(94,102,114,.18)';ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillStyle='#5E6672';ctx.font='10px Consolas';ctx.fillText(format(value,2),5,yy+3)}if(pi===2){const yy=y(-Math.log10(.05));ctx.strokeStyle='#D6B52C';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.setLineDash([])}sites.forEach((site,si)=>{const vals=layers.map(layer=>GEOM.statistics[model+'|'+site+'|'+layer]);ctx.strokeStyle=colors[si];ctx.lineWidth=2;ctx.beginPath();vals.forEach((s,i)=>{const xx=x(layers[i]),yy=y(clamp(panel.value(s),panel.min,panel.max));i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)});ctx.stroke();for(let i=0;i<vals.length;i++){const s=vals[i],xx=x(layers[i]),yy=y(clamp(panel.value(s),panel.min,panel.max)),sig=pi===1?+s.count_eta_q<.05:pi===2?+s.interaction_q<.05:false;ctx.fillStyle=sig?colors[si]:'#FFFDF8';ctx.strokeStyle=colors[si];ctx.beginPath();ctx.arc(xx,yy,sig?3.4:2.2,0,Math.PI*2);ctx.fill();ctx.stroke();if(pi===1){const lo=y(clamp(+s.count_eta_delta_ci_low,panel.min,panel.max)),hi=y(clamp(+s.count_eta_delta_ci_high,panel.min,panel.max));ctx.globalAlpha=.22;ctx.beginPath();ctx.moveTo(xx,lo);ctx.lineTo(xx,hi);ctx.stroke();ctx.globalAlpha=1}}});const selected=x(+controls.layer.value);ctx.strokeStyle='#23165C';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(selected,top);ctx.lineTo(selected,bottom);ctx.stroke();ctx.setLineDash([])});ctx.fillStyle='#00A88F';ctx.fillRect(w-214,8,12,3);ctx.fillStyle='#20242D';ctx.font='11px Segoe UI';ctx.fillText('trace',w-197,12);ctx.fillStyle='#D94B86';ctx.fillRect(w-132,8,12,3);ctx.fillStyle='#20242D';ctx.fillText('answer',w-115,12);ctx.fillStyle='#5E6672';ctx.font='10px Consolas';for(let i=0;i<layers.length;i+=5)ctx.fillText('L'+layers[i],x(layers[i])-7,h-8)}
function matrixDelta(a,b){return a.map((row,i)=>row.map((v,j)=>(v==null||b[i][j]==null)?null:b[i][j]-v))}
function finiteValues(matrix){return matrix.flat(Infinity).filter(v=>v!=null&&Number.isFinite(+v)).map(Number)}
function seqColor(value,max){const t=clamp(value/Math.max(max,1e-12),0,1),stops=[[21,17,43],[37,70,168],[0,169,216],[246,227,106]],p=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(p)),f=p-i,a=stops[i],b=stops[i+1];return'rgb('+a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(',')+')'}
function divColor(value,max){const t=clamp(value/Math.max(max,1e-12),-1,1),zero=[243,238,228],end=t<0?[49,91,199]:[217,75,134],f=Math.abs(t);return'rgb('+zero.map((v,k)=>Math.round(v+(end[k]-v)*f)).join(',')+')'}
function drawHeat(id,matrix,options){const canvas=document.getElementById(id),{ctx,w,h}=resizeCanvas(canvas),rows=matrix.length,cols=matrix[0].length,left=options.left||42,bottom=options.bottom||52,top=10,right=8,plotW=w-left-right,plotH=h-top-bottom,cw=plotW/cols,ch=plotH/rows;ctx.fillStyle='#15112B';ctx.fillRect(0,0,w,h);for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const v=matrix[r][c];ctx.fillStyle=v==null?'#2A2638':options.diverging?divColor(v,options.max):seqColor(v,options.max);ctx.fillRect(left+c*cw,top+r*ch,Math.ceil(cw+.3),Math.ceil(ch+.3))}ctx.fillStyle='#5E6672';ctx.font='10px Segoe UI';if(options.xLabels)for(let c=0;c<cols;c++){ctx.save();ctx.translate(left+(c+.5)*cw,h-bottom+5);ctx.rotate(-Math.PI/3);ctx.fillText(options.xLabels[c],0,0);ctx.restore()}ctx.fillText(options.yTop||'0',3,top+8);ctx.fillText(options.yBottom||String(rows-1),3,top+plotH);canvas.onmousemove=e=>{const rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*w/rect.width,y=(e.clientY-rect.top)*h/rect.height,c=Math.floor((x-left)/cw),r=Math.floor((y-top)/ch);if(r>=0&&r<rows&&c>=0&&c<cols)options.hover(r,c,matrix[r][c])};return options.max}
function drawAttention(){const model=controls.model.value,layer=+controls.layer.value,d=ATTN.models[model],li=d.layers.indexOf(layer),p=d.conditions.cue_present,a=d.conditions.cue_absent,regions=ATTN.region_names;const answerP=p.answer_layer_region,answerA=a.answer_layer_region,answerD=matrixDelta(answerP,answerA),answerMax=Math.max(...finiteValues(answerP),...finiteValues(answerA)),answerDelta=Math.max(...finiteValues(answerD).map(Math.abs),1e-12),answerHover=(r,c,v)=>document.getElementById('answer-attn-hover').textContent='L'+d.layers[r]+' · '+regions[c]+' · mass '+format(v,5);drawHeat('answer-attn-present',answerP,{max:answerMax,xLabels:regions,yTop:'L0',yBottom:'L'+d.layers[d.layers.length-1],hover:answerHover});drawHeat('answer-attn-absent',answerA,{max:answerMax,xLabels:regions,yTop:'L0',yBottom:'L'+d.layers[d.layers.length-1],hover:answerHover});drawHeat('answer-attn-delta',answerD,{max:answerDelta,diverging:true,xLabels:regions,yTop:'L0',yBottom:'L'+d.layers[d.layers.length-1],hover:answerHover});document.querySelectorAll('.actual-max').forEach(x=>x.textContent=format(answerMax,3));document.querySelectorAll('.delta-min').forEach(x=>x.textContent='−'+format(answerDelta,3));document.querySelectorAll('.delta-max').forEach(x=>x.textContent='+'+format(answerDelta,3));
const traceP=p.trace_time_region[li],traceA=a.trace_time_region[li],traceD=matrixDelta(traceP,traceA),traceMax=Math.max(...finiteValues(traceP),...finiteValues(traceA)),traceDelta=Math.max(...finiteValues(traceD).map(Math.abs),1e-12),traceHover=(r,c,v)=>document.getElementById('trace-attn-hover').textContent='L'+layer+' · trace time '+format(r/(traceP.length-1),2)+' · '+regions[c]+' · mass '+format(v,5);drawHeat('trace-attn-present',traceP,{max:traceMax,xLabels:regions,yTop:'early',yBottom:'late',hover:traceHover});drawHeat('trace-attn-absent',traceA,{max:traceMax,xLabels:regions,yTop:'early',yBottom:'late',hover:traceHover});drawHeat('trace-attn-delta',traceD,{max:traceDelta,diverging:true,xLabels:regions,yTop:'early',yBottom:'late',hover:traceHover});document.querySelectorAll('.trace-actual-max').forEach(x=>x.textContent=format(traceMax,3));document.querySelectorAll('.trace-delta-min').forEach(x=>x.textContent='−'+format(traceDelta,3));document.querySelectorAll('.trace-delta-max').forEach(x=>x.textContent='+'+format(traceDelta,3));
const tttP=p.trace_to_trace[li],tttA=a.trace_to_trace[li],tttD=matrixDelta(tttP,tttA),tttMax=Math.max(...finiteValues(tttP),...finiteValues(tttA)),tttDelta=Math.max(...finiteValues(tttD).map(Math.abs),1e-12),tttHover=(r,c,v)=>document.getElementById('ttt-hover').textContent='L'+layer+' · query time '+format(r/(tttP.length-1),2)+' · key time '+format(c/(tttP[0].length-1),2)+' · mass '+format(v,6);drawHeat('ttt-present',tttP,{max:tttMax,bottom:25,yTop:'early',yBottom:'late',hover:tttHover});drawHeat('ttt-absent',tttA,{max:tttMax,bottom:25,yTop:'early',yBottom:'late',hover:tttHover});drawHeat('ttt-delta',tttD,{max:tttDelta,diverging:true,bottom:25,yTop:'early',yBottom:'late',hover:tttHover});document.querySelectorAll('.ttt-actual-max').forEach(x=>x.textContent=format(tttMax,4));document.querySelectorAll('.ttt-delta-min').forEach(x=>x.textContent='−'+format(tttDelta,4));document.querySelectorAll('.ttt-delta-max').forEach(x=>x.textContent='+'+format(tttDelta,4))}
function updateSigTable(){const model=controls.model.value,sites=[[controls.traceSite.value,controls.traceSite.value==='trace_mean'?'trace mean':'trace last'],['answer_query','answer query']],layers=ATTN.models[model].layers,body=document.getElementById('sig-table');body.innerHTML=sites.map(([site,label])=>{const stats=layers.map(l=>GEOM.statistics[model+'|'+site+'|'+l]);return'<tr><td>'+label+'</td><td>'+ranges(layers.filter((l,i)=>+stats[i].count_eta_q<.05))+'</td><td>'+ranges(layers.filter((l,i)=>+stats[i].interaction_q<.05))+'</td></tr>'}).join('')}
function updateConclusion(){const model=controls.model.value,layer=controls.layer.value,trace=GEOM.statistics[activeKey(controls.traceSite.value)],answer=GEOM.statistics[activeKey('answer_query')],describe=(name,s)=>name+': strength '+(+s.count_eta_q<.05?'显著':'不显著')+' (q='+pformat(s.count_eta_q)+'), interaction '+(+s.interaction_q<.05?'显著':'不显著')+' (q='+pformat(s.interaction_q)+')';document.querySelector('#selected-conclusion span').textContent=model+' · L'+layer+' · '+describe('trace',trace)+'；'+describe('answer',answer)+'。'}
function refreshLayers(){const model=controls.model.value,layers=ATTN.models[model].layers,hadOptions=controls.layer.options.length>0,old=+controls.layer.value,landmark=GEOM.landmarks[model].display;controls.layer.innerHTML='';for(const layer of layers){const o=document.createElement('option');o.value=String(layer);let suffix='';if(layer===GEOM.landmarks[model].display)suffix=' · prior display landmark';if(layer===GEOM.landmarks[model].probe)suffix=' · prior probe landmark';o.textContent='L'+layer+suffix;controls.layer.appendChild(o)}controls.layer.value=String(hadOptions&&layers.includes(old)?old:landmark)}
function drawAll(){drawGeometry('trace',controls.traceSite.value);drawGeometry('answer','answer_query');drawJoint();drawSweep();drawAttention();updateSigTable();updateConclusion();document.getElementById('trace-title').textContent=controls.traceSite.value==='trace_mean'?'Trace-mean counter':'Last-trace-token counter'}
for(const key of ['layer','traceSite','basis','points','pairs','x','y','z'])controls[key].addEventListener('change',drawAll);controls.model.addEventListener('change',()=>{refreshLayers();drawAll()});document.getElementById('reset').addEventListener('click',()=>{cameras.trace={yaw:-.72,pitch:.42,zoom:1};cameras.answer={yaw:-.72,pitch:.42,zoom:1};drawAll()});for(const id of ['joint-x','joint-y','joint-z'])document.getElementById(id).addEventListener('change',drawJoint);document.getElementById('joint-reset').addEventListener('click',()=>{cameras.joint={yaw:-.72,pitch:.42,zoom:1};drawJoint()});
document.getElementById('sweep-canvas').addEventListener('click',e=>{const canvas=e.currentTarget,rect=canvas.getBoundingClientRect(),w=rect.width,model=controls.model.value,layers=ATTN.models[model].layers,x=(e.clientX-rect.left),left=54,right=22,index=Math.round((x-left)/(w-left-right)*(layers.length-1));controls.layer.value=String(layers[clamp(index,0,layers.length-1)]);drawAll()});
attachGeometry('trace');attachGeometry('answer');attachJoint();refreshLayers();drawAll();if(location.hash)setTimeout(()=>{const target=document.querySelector(location.hash);if(target)target.scrollIntoView()},250);window.addEventListener('resize',drawAll);
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-data", type=Path, required=True)
    parser.add_argument("--attention-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geometry = args.geometry_data.read_text(encoding="utf-8")
    attention = args.attention_data.read_text(encoding="utf-8")
    rendered = HTML.replace("@@GEOMETRY_DATA@@", geometry).replace(
        "@@ATTENTION_DATA@@", attention
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)
    print(args.output.stat().st_size)


if __name__ == "__main__":
    main()
