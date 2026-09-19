from __future__ import annotations

import argparse
from pathlib import Path


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Realistic NIAH V4.4.2 · Four counter comparisons</title>
<style>
:root{--bg:#f4f1ea;--paper:#fffdf8;--ink:#20242d;--muted:#646b76;--line:#d9d1c5;--navy:#15112b;--violet:#34226d;--pink:#d94b86;--cyan:#00a9d8;--gold:#d9b72f;--green:#26866f;--red:#b63f5d}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 Inter,"Segoe UI","Microsoft YaHei",sans-serif}header{background:linear-gradient(135deg,#17122f,#36256d 62%,#244d70);color:#fff;padding:48px max(22px,calc((100vw - 1450px)/2)) 42px}header .eyebrow{font:12px/1.4 Consolas,monospace;letter-spacing:.12em;text-transform:uppercase;color:#bbdff4}h1{font-size:clamp(32px,4vw,54px);line-height:1.12;margin:10px 0 16px;max-width:1100px}header p{max-width:1050px;margin:0;color:#e6e2f4;font-size:17px}nav{display:flex;flex-wrap:wrap;gap:9px;margin-top:22px}nav a{color:#fff;text-decoration:none;border:1px solid rgba(255,255,255,.32);padding:6px 10px;border-radius:3px;font-size:13px}main{max-width:1480px;margin:auto;padding:34px 22px 80px}section{margin:0 0 54px}h2{font-size:28px;line-height:1.25;margin:0 0 10px}h3{font-size:18px;line-height:1.35;margin:0}.lede{color:var(--muted);max-width:1120px;margin:0 0 18px}.controls{display:flex;flex-wrap:wrap;align-items:end;gap:12px;background:var(--paper);border:1px solid var(--line);padding:13px 14px;margin:18px 0}.controls label{display:grid;gap:4px;font:12px/1.3 Consolas,monospace;color:var(--muted)}select,button{font:14px/1.25 "Segoe UI","Microsoft YaHei",sans-serif;color:var(--ink);background:#fff;border:1px solid #bbb2a5;border-radius:3px;padding:7px 9px}button{cursor:pointer}.check{display:flex!important;align-items:center;gap:7px!important;padding:7px 4px}.check input{width:16px;height:16px}.legend{display:flex;flex-wrap:wrap;gap:11px 17px;color:var(--muted);font-size:13px;margin:10px 0}.shape{display:inline-block;width:10px;height:10px;margin-right:6px;border:2px solid}.circle{border-radius:50%;border-color:#fff;background:var(--violet);box-shadow:0 0 0 1px #777}.square{border-color:#d8b326;background:var(--violet)}.count-key{display:inline-flex;gap:3px;align-items:center}.count-key i{width:10px;height:10px;border-radius:50%;display:inline-block}.geometry-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.plot-card{margin:0;background:var(--paper);border:1px solid var(--line);padding:12px}.plot-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:8px}.site-note{font:12px/1.4 Consolas,monospace;color:var(--muted)}.plot-shell{position:relative;background:var(--navy);overflow:hidden}.plot-shell canvas{display:block;width:100%;height:480px;touch-action:none}.tooltip{position:absolute;display:none;pointer-events:none;z-index:3;max-width:260px;background:rgba(255,253,248,.97);border:1px solid var(--line);padding:7px 9px;color:var(--ink);font-size:12px}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin-top:9px}.metric{background:#f3efe7;padding:7px 8px;min-height:57px}.metric span{display:block;color:var(--muted);font-size:11px;line-height:1.3}.metric strong{display:block;font:13px/1.35 Consolas,monospace;margin-top:4px}.metric.sig{box-shadow:inset 3px 0 var(--pink)}.metric.ns{box-shadow:inset 3px 0 #bbb2a5}figcaption{color:var(--muted);font-size:12px;margin-top:8px}.boundary{background:#fff7dd;border-left:4px solid var(--gold);padding:12px 15px;margin:16px 0;color:#504923}.attention-block{margin-top:26px}.attention-block+.attention-block{padding-top:32px;border-top:1px solid var(--line)}.attention-title{display:flex;justify-content:space-between;gap:16px;align-items:baseline;flex-wrap:wrap}.formula{font:12px/1.55 Consolas,monospace;color:var(--muted);margin:5px 0 12px}.atlas-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.heat-card{margin:0;background:var(--paper);border:1px solid var(--line);padding:9px}.heat-card h3{font-size:15px;margin-bottom:6px}.heat-card canvas{display:block;width:100%;height:390px;background:var(--paper)}.colorbar{height:9px;margin:8px 3px 2px;background:linear-gradient(90deg,#f7f3ea,#588bd2,#23165c)}.colorbar.div{background:linear-gradient(90deg,#315bc7,#f7f3ea,#b53d66)}.bar-labels{display:flex;justify-content:space-between;color:var(--muted);font:10px/1.3 Consolas,monospace}.atlas-hover{min-height:26px;margin-top:8px;padding:5px 8px;background:var(--paper);border:1px solid var(--line);font:12px/1.4 Consolas,monospace;color:var(--muted)}.method-note{color:var(--muted);font-size:12px;margin:8px 0 0}.footer-note{border-top:1px solid var(--line);padding-top:16px;color:var(--muted);font-size:12px}
@media(max-width:1050px){.geometry-grid,.atlas-grid{grid-template-columns:1fr}.plot-shell canvas{height:500px}.heat-card canvas{height:470px}}@media(max-width:650px){header{padding-top:34px}main{padding-left:10px;padding-right:10px}.plot-shell canvas{height:410px}.heat-card canvas{height:380px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
</head>
<body>
<header>
  <div class="eyebrow">Realistic NIAH · V4.4.2 · 2 models × 2 modes × 2 prompts</div>
  <h1>有无开头提示：四个 counter geometry 对比</h1>
  <p>只保留四组 hidden-state 对比：non-thinking 的 prompt / answer counter，以及 native thinking 的 trace-last / answer counter。每个 mode 后面紧跟自己的 attention score 大表，左右直接比较 cue-present 与 cue-absent。</p>
  <nav><a href="#geometry">四组 geometry</a><a href="#attention">两组 attention maps</a><a href="#definitions">定义与边界</a></nav>
</header>
<main>
<section id="geometry">
  <h2>1 · Hidden-state geometry</h2>
  <p class="lede">每张图都在该 counter 自己的 cue-present / cue-absent 共享 PCA basis 中投影；圆点/实线是有提示，方点/虚线是无提示，颜色表示读取进度或 gold count 1–10。拖拽任一图会同步旋转四张图，滚轮同步缩放。</p>
  <div class="controls">
    <label>model<select id="geom-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
    <label>layer<select id="geom-layer"></select></label>
    <label>PCA view<select id="geom-basis"><option value="raw">raw pooled（含整体 cue shift）</option><option value="centered">cue-centered（比较轨迹形状）</option></select></label>
    <label>points<select id="geom-points"><option value="all">10 seeds + centroids</option><option value="centroids">centroids only</option></select></label>
    <label>x<select id="axis-x"><option value="0">PC1</option><option value="1">PC2</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
    <label>y<select id="axis-y"><option value="1">PC2</option><option value="0">PC1</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
    <label>z<select id="axis-z"><option value="2">PC3</option><option value="0">PC1</option><option value="1">PC2</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
    <label class="check"><input id="geom-pairs" type="checkbox" checked>paired cue links</label>
    <button id="geom-reset" type="button">reset camera</button>
  </div>
  <div class="legend">
    <span><i class="shape circle"></i>cue-present · circle / solid</span>
    <span><i class="shape square"></i>cue-absent · square / dashed</span>
    <span class="count-key" id="count-key"></span>
  </div>
  <div class="geometry-grid" id="geometry-grid"></div>
  <div class="boundary"><strong>显著性读法：</strong>每张图下方的 q 值来自原始 full hidden space，不来自 3D 投影。<code>count×cue q</code> 检验提示是否改变 counter 的计数形状；<code>Δ strength q</code> 检验 count η² 是否随提示发生变化。两者均逐层 BH-FDR 校正，seed 是独立 cluster。</div>
</section>

<section id="attention">
  <h2>2 · Attention score maps</h2>
  <p class="lede">每个格子固定对应一个 decoder layer / head。同一 mode 内 cue-present 与 cue-absent 共用色标；第三张图是 cue-absent − cue-present。non-thinking 与 native thinking 的分数定义不同，颜色绝对值不能跨 mode 比较。</p>
  <div class="controls"><label>model<select id="attn-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label></div>
  <div id="attention-panels"></div>
</section>

<section id="definitions">
  <h2>3 · Counter 与 score 定义</h2>
  <div class="boundary"><strong>Non-thinking prompt counter：</strong>对每个 seed 只使用同一个 N=10 prompt，依次取第 1–10 个 active needle 最后 token 处的 post-block residual。它表示“随着 prompt 被读入”的 counter；不是十个独立 N=k prompt 的最终状态，也不是 answer state。</div>
  <div class="boundary"><strong>Native trace counter：</strong>使用最后一个保存的 thinking-trace query state（trace last），因此四图中不再混入 trace mean。Answer counter 在两个 mode 中都使用最后一个 <code>Total:</code> answer-query state。</div>
  <p class="footer-note">PCA 每个 model × counter × layer 独立拟合，所以只比较同一张图内部“有提示 vs 无提示”；不要直接相减不同图的 PC 坐标。Prompt endpoint 的补算使用 prompt-only causal forward：在 causal decoder 中，某个 needle endpoint 的 state 不受后续 prompt suffix 或生成 continuation 影响。</p>
</section>
</main>

<script>
const NATIVE_GEOM=@@NATIVE@@;
const NT_GEOM=@@NONTHINKING@@;
const PROMPT_GEOM=@@PROMPT@@;
const ATLAS=@@ATLAS@@;

const COUNT_COLORS=['#3db7d6','#3f8fd2','#5565c9','#794fbd','#a146aa','#c54a8c','#df5b6d','#ee7950','#eda33f','#d4c63b'];
const PANELS=[
 {id:'nt-prompt',title:'Non-thinking · prompt counter',note:'N=10 prompt · needle endpoints 1–10',data:PROMPT_GEOM,site:'prompt_counter',unit:'occurrence'},
 {id:'nt-answer',title:'Non-thinking · answer counter',note:'Total: answer-query · gold count 1–10',data:NT_GEOM,site:'answer_query',unit:'gold count'},
 {id:'native-trace',title:'Native thinking · trace counter',note:'trace last · gold count 1–10',data:NATIVE_GEOM,site:'trace_last',unit:'gold count'},
 {id:'native-answer',title:'Native thinking · answer counter',note:'Total: answer-query · gold count 1–10',data:NATIVE_GEOM,site:'answer_query',unit:'gold count'}
];
const controls={
 model:document.getElementById('geom-model'),layer:document.getElementById('geom-layer'),basis:document.getElementById('geom-basis'),points:document.getElementById('geom-points'),pairs:document.getElementById('geom-pairs'),
 x:document.getElementById('axis-x'),y:document.getElementById('axis-y'),z:document.getElementById('axis-z')
};
const camera={yaw:-.72,pitch:.42,zoom:1};
const charts={};

function clamp(v,a,b){return Math.max(a,Math.min(b,v))}
function fmt(v,d=3){return Number.isFinite(+v)?(+v).toFixed(d):'N/A'}
function pformat(v){if(!Number.isFinite(+v))return 'N/A';if(+v<.001)return (+v).toExponential(1);return (+v).toFixed(3)}
function resizeCanvas(canvas,minH=260){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(300,Math.round(rect.width)),h=Math.max(minH,Math.round(rect.height));if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return{ctx,w,h}}
function key(panel){return controls.model.value+'|'+panel.site+'|'+controls.layer.value}
function rowCoords(row,condition,basis){let start;if(basis==='raw')start=condition==='cue_present'?4:10;else start=condition==='cue_present'?16:22;return row.slice(start,start+6).map(Number)}
function meanPoint(points){const out=Array(points[0].length).fill(0);for(const p of points)for(let i=0;i<out.length;i++)out[i]+=p[i]/points.length;return out}
function centroids(rows,condition,basis){const out=[];for(let count=1;count<=10;count++){const points=rows.filter(r=>+r[1]===count).map(r=>rowCoords(r,condition,basis));out.push(meanPoint(points))}return out}
function rotated(point,center,axes){let x=point[axes[0]]-center[axes[0]],y=point[axes[1]]-center[axes[1]],z=point[axes[2]]-center[axes[2]];const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);const x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;return{x:x1,y:y1,z:z2}}
function transformFactory(points,axes,w,h){const center=meanPoint(points),rot=points.map(p=>rotated(p,center,axes));let span=1;for(const q of rot)span=Math.max(span,Math.abs(q.x),Math.abs(q.y));const scale=.39*Math.min(w,h)/span*camera.zoom;return p=>{const q=rotated(p,center,axes);return{x:w/2+q.x*scale,y:h/2-q.y*scale,z:q.z}}}
function panelMarkup(panel){return '<figure class="plot-card"><div class="plot-head"><h3>'+panel.title+'</h3><span class="site-note">'+panel.note+'</span></div><div class="plot-shell"><canvas id="'+panel.id+'-canvas" aria-label="'+panel.title+' interactive 3D PCA"></canvas><div class="tooltip" id="'+panel.id+'-tooltip"></div></div><div class="stats" id="'+panel.id+'-stats"></div><figcaption>共享 PCA 仅覆盖本图的 cue-present / cue-absent；seed n=10。圆/方形中心轨迹分别连接 count 1→10。</figcaption></figure>'}
document.getElementById('geometry-grid').innerHTML=PANELS.map(panelMarkup).join('');
document.getElementById('count-key').innerHTML='count '+COUNT_COLORS.map((c,i)=>'<i style="background:'+c+'" aria-label="count '+(i+1)+'"></i>').join('')+' 1→10';

function statsHtml(panel,dataset,stat){const evr=controls.basis.value==='raw'?dataset.evr_raw:dataset.evr_cue_centered;const interaction=+stat.interaction_q,strength=+stat.count_eta_q;return [
 '<div class="metric '+(interaction<.05?'sig':'ns')+'"><span>count×cue q</span><strong>'+pformat(interaction)+(interaction<.05?' · FDR sig.':' · n.s.')+'</strong></div>',
 '<div class="metric '+(strength<.05?'sig':'ns')+'"><span>Δ strength q</span><strong>'+pformat(strength)+(strength<.05?' · FDR sig.':' · n.s.')+'</strong></div>',
 '<div class="metric"><span>count η² · present → absent</span><strong>'+fmt(stat.count_eta_present,4)+' → '+fmt(stat.count_eta_absent,4)+'</strong></div>',
 '<div class="metric"><span>centroid CKA · EVR PC1–3</span><strong>'+fmt(stat.centroid_cka,3)+' · '+fmt(100*evr.slice(0,3).reduce((a,b)=>a+b,0),1)+'%</strong></div>'
 ].join('')}
function drawMarker(ctx,x,y,count,condition,size){ctx.fillStyle=COUNT_COLORS[count-1];ctx.strokeStyle=condition==='cue_present'?'#fffdf8':'#f1cf45';ctx.lineWidth=condition==='cue_present'?1.2:1.6;ctx.beginPath();if(condition==='cue_present')ctx.arc(x,y,size,0,Math.PI*2);else ctx.rect(x-size,y-size,size*2,size*2);ctx.fill();ctx.stroke()}
function drawPanel(panel){const dataset=panel.data.datasets[key(panel)],stat=panel.data.statistics[key(panel)];if(!dataset||!stat)return;const canvas=document.getElementById(panel.id+'-canvas'),tooltip=document.getElementById(panel.id+'-tooltip'),{ctx,w,h}=resizeCanvas(canvas),basis=controls.basis.value,axes=[+controls.x.value,+controls.y.value,+controls.z.value],rows=dataset.rows,all=[];for(const row of rows){all.push(rowCoords(row,'cue_present',basis),rowCoords(row,'cue_absent',basis))}const tf=transformFactory(all,axes,w,h);ctx.fillStyle='#15112b';ctx.fillRect(0,0,w,h);ctx.strokeStyle='rgba(255,255,255,.12)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(w*.07,h/2);ctx.lineTo(w*.93,h/2);ctx.moveTo(w/2,h*.07);ctx.lineTo(w/2,h*.93);ctx.stroke();if(controls.pairs.checked){ctx.strokeStyle='rgba(255,255,255,.12)';ctx.lineWidth=.7;for(const row of rows){const p=tf(rowCoords(row,'cue_present',basis)),a=tf(rowCoords(row,'cue_absent',basis));ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(a.x,a.y);ctx.stroke()}}const screen=[];if(controls.points.value==='all'){const items=[];for(const row of rows)for(const condition of ['cue_present','cue_absent'])items.push({row,condition,q:tf(rowCoords(row,condition,basis))});items.sort((a,b)=>a.q.z-b.q.z);ctx.globalAlpha=.55;for(const item of items){drawMarker(ctx,item.q.x,item.q.y,+item.row[1],item.condition,2.7);screen.push({x:item.q.x,y:item.q.y,row:item.row,condition:item.condition})}ctx.globalAlpha=1}for(const condition of ['cue_present','cue_absent']){const centers=centroids(rows,condition,basis).map((point,i)=>({count:i+1,q:tf(point)}));ctx.strokeStyle=condition==='cue_present'?'#fffdf8':'#f1cf45';ctx.lineWidth=2.6;ctx.setLineDash(condition==='cue_present'?[]:[7,5]);ctx.beginPath();centers.forEach((item,i)=>i?ctx.lineTo(item.q.x,item.q.y):ctx.moveTo(item.q.x,item.q.y));ctx.stroke();ctx.setLineDash([]);for(const item of centers)drawMarker(ctx,item.q.x,item.q.y,item.count,condition,condition==='cue_present'?6:5)}charts[panel.id]={panel,canvas,tooltip,screen};document.getElementById(panel.id+'-stats').innerHTML=statsHtml(panel,dataset,stat)}
function drawGeometry(){PANELS.forEach(drawPanel)}
function attachPanel(panel){const canvas=document.getElementById(panel.id+'-canvas'),tooltip=document.getElementById(panel.id+'-tooltip');let drag=false,lx=0,ly=0;canvas.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointercancel',()=>drag=false);canvas.addEventListener('pointermove',e=>{if(drag){camera.yaw+=(e.clientX-lx)*.008;camera.pitch=clamp(camera.pitch+(e.clientY-ly)*.008,-1.45,1.45);lx=e.clientX;ly=e.clientY;drawGeometry();return}const chart=charts[panel.id],rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=null,dist=Infinity;for(const p of chart.screen||[]){const d=(p.x-x)**2+(p.y-y)**2;if(d<dist){dist=d;best=p}}if(best&&dist<90){tooltip.style.display='block';tooltip.style.left=Math.min(rect.width-225,x+12)+'px';tooltip.style.top=Math.max(8,y-12)+'px';tooltip.innerHTML='<strong>'+panel.unit+' '+best.row[1]+' · seed '+best.row[0]+'</strong><br>'+best.condition+'<br>'+panel.site+' · L'+controls.layer.value}else tooltip.style.display='none'});canvas.addEventListener('mouseleave',()=>tooltip.style.display='none');canvas.addEventListener('wheel',e=>{e.preventDefault();camera.zoom=clamp(camera.zoom*Math.exp(-e.deltaY*.001),.55,2.8);drawGeometry()},{passive:false})}
function refreshLayers(){const model=controls.model.value,layers=Object.values(NATIVE_GEOM.datasets).filter(d=>d.model===model&&d.site==='trace_last').map(d=>+d.layer).sort((a,b)=>a-b),old=+controls.layer.value;controls.layer.innerHTML=layers.map(layer=>'<option value="'+layer+'">L'+layer+(layer===NATIVE_GEOM.landmarks[model].display?' · prior display landmark':'')+'</option>').join('');controls.layer.value=String(layers.includes(old)?old:NATIVE_GEOM.landmarks[model].display)}
for(const keyName of ['layer','basis','points','pairs','x','y','z'])controls[keyName].addEventListener('change',drawGeometry);controls.model.addEventListener('change',()=>{refreshLayers();document.getElementById('attn-model').value=controls.model.value;drawGeometry();drawAllAttention()});document.getElementById('geom-reset').addEventListener('click',()=>{camera.yaw=-.72;camera.pitch=.42;camera.zoom=1;drawGeometry()});PANELS.forEach(attachPanel);

const ATTENTION_SPECS=[
 {mode:'nonthinking',id:'attn-nt',title:'Non-thinking · broad retrieval score'},
 {mode:'native_thinking',id:'attn-native',title:'Native thinking · targeted retrieval score'}
];
document.getElementById('attention-panels').innerHTML=ATTENTION_SPECS.map(spec=>'<div class="attention-block" id="'+spec.id+'"><div class="attention-title"><h3>'+spec.title+'</h3><span class="site-note" id="'+spec.id+'-sample"></span></div><div class="formula" id="'+spec.id+'-formula"></div><div class="atlas-grid"><figure class="heat-card"><h3>cue-present</h3><canvas id="'+spec.id+'-present"></canvas><div class="colorbar"></div><div class="bar-labels"><span>0</span><span class="actual-max">cap</span></div></figure><figure class="heat-card"><h3>cue-absent</h3><canvas id="'+spec.id+'-absent"></canvas><div class="colorbar"></div><div class="bar-labels"><span>0</span><span class="actual-max">cap</span></div></figure><figure class="heat-card"><h3>cue-absent − cue-present</h3><canvas id="'+spec.id+'-delta"></canvas><div class="colorbar div"></div><div class="bar-labels"><span class="delta-min">−cap</span><span>0</span><span class="delta-max">+cap</span></div></figure></div><div class="atlas-hover" id="'+spec.id+'-hover">移动鼠标查看 layer / head / score。</div><p class="method-note" id="'+spec.id+'-note"></p></div>').join('');
const attnModel=document.getElementById('attn-model');
function finite(values){return values.flat(Infinity).filter(Number.isFinite)}
function quantile(values,p){const x=values.filter(Number.isFinite).sort((a,b)=>a-b);if(!x.length)return 0;const pos=(x.length-1)*p,lo=Math.floor(pos),hi=Math.ceil(pos);return lo===hi?x[lo]:x[lo]+(x[hi]-x[lo])*(pos-lo)}
function sequential(v,max){const t=clamp(v/Math.max(max,1e-12),0,1),a=t<.55?t/.55:(t-.55)/.45,l=t<.55?[247,243,234]:[88,139,210],r=t<.55?[88,139,210]:[35,22,92];return 'rgb('+l.map((x,i)=>Math.round(x+(r[i]-x)*a)).join(',')+')'}
function diverging(v,max){const t=clamp(v/Math.max(max,1e-12),-1,1),a=Math.abs(t),l=[247,243,234],r=t<0?[49,91,199]:[181,61,102];return 'rgb('+l.map((x,i)=>Math.round(x+(r[i]-x)*a)).join(',')+')'}
function headMatrix(modeData,condition){const source=modeData.conditions[condition].layer_head_score;return Array.from({length:modeData.heads},(_,head)=>modeData.layers.map((_,li)=>source[li][head]))}
function matrixDelta(present,absent){return present.map((row,r)=>row.map((v,c)=>Number.isFinite(v)&&Number.isFinite(absent[r][c])?absent[r][c]-v:null))}
function drawHeat(canvasId,matrix,{layers,max,label,div=false,hoverId}){const canvas=document.getElementById(canvasId),{ctx,w,h}=resizeCanvas(canvas),rows=matrix.length,cols=matrix[0].length,left=42,right=9,top=10,bottom=42,pw=w-left-right,ph=h-top-bottom,cw=pw/cols,ch=ph/rows;ctx.fillStyle='#fffdf8';ctx.fillRect(0,0,w,h);for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const v=matrix[r][c];ctx.fillStyle=v==null?'#d8d3ca':div?diverging(v,max):sequential(v,max);ctx.fillRect(left+c*cw,top+r*ch,Math.ceil(cw+.2),Math.ceil(ch+.2))}ctx.strokeStyle='#aaa296';ctx.strokeRect(left,top,pw,ph);ctx.fillStyle='#646b76';ctx.font='10px Consolas';for(let c=0;c<cols;c+=5){ctx.save();ctx.translate(left+(c+.5)*cw,h-bottom+7);ctx.rotate(-Math.PI/3);ctx.fillText('L'+layers[c],0,0);ctx.restore()}const step=rows<=12?1:4;for(let r=0;r<rows;r+=step)ctx.fillText('H'+r,7,top+(r+.68)*ch);canvas.onmousemove=e=>{const rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*w/rect.width,y=(e.clientY-rect.top)*h/rect.height,c=Math.floor((x-left)/cw),r=Math.floor((y-top)/ch);if(r>=0&&r<rows&&c>=0&&c<cols){const v=matrix[r][c];document.getElementById(hoverId).textContent=label+' · L'+layers[c]+' H'+r+' · score '+(v==null?'N/A':fmt(v,7))}}}
function drawAttention(spec){const modeData=ATLAS.models[attnModel.value].modes[spec.mode],present=headMatrix(modeData,'cue_present'),absent=headMatrix(modeData,'cue_absent'),delta=matrixDelta(present,absent),actual=finite([present,absent]),diff=finite(delta).map(Math.abs),actualMax=Math.max(quantile(actual,.995),1e-12),deltaMax=Math.max(quantile(diff,.995),1e-12);drawHeat(spec.id+'-present',present,{layers:modeData.layers,max:actualMax,label:'cue-present',hoverId:spec.id+'-hover'});drawHeat(spec.id+'-absent',absent,{layers:modeData.layers,max:actualMax,label:'cue-absent',hoverId:spec.id+'-hover'});drawHeat(spec.id+'-delta',delta,{layers:modeData.layers,max:deltaMax,label:'absent−present',div:true,hoverId:spec.id+'-hover'});document.querySelectorAll('#'+spec.id+' .actual-max').forEach(x=>x.textContent='p99.5 '+fmt(actualMax,6));document.querySelector('#'+spec.id+' .delta-min').textContent='−'+fmt(deltaMax,6);document.querySelector('#'+spec.id+' .delta-max').textContent='+'+fmt(deltaMax,6);document.getElementById(spec.id+'-formula').textContent=modeData.score_definition.name+' · site: '+modeData.score_definition.site+' · '+modeData.score_definition.formula;const pp=modeData.conditions.cue_present.valid_samples_by_layer,aa=modeData.conditions.cue_absent.valid_samples_by_layer,unavailable=Math.max(...pp,...aa)===0,zero=actual.length&&Math.max(...actual.map(Math.abs))<=1e-12;document.getElementById(spec.id+'-sample').textContent=attnModel.value+' · scheduled n='+modeData.conditions.cue_present.samples;document.getElementById(spec.id+'-note').textContent='finite n/layer: present '+Math.min(...pp)+'–'+Math.max(...pp)+', absent '+Math.min(...aa)+'–'+Math.max(...aa)+'. '+(unavailable?'N/A 是 architecture window 对原始 needle 不可见，属于结构性不可用。':zero?'所有 direct retrieval scores 在捕获 mask 下为 0；这是结构性结果，不是缺失数据。':'正值图使用 mode 内共享 p99.5 色标，差值图独立使用对称 p99.5 色标。')}
function drawAllAttention(){ATTENTION_SPECS.forEach(drawAttention)}
attnModel.addEventListener('change',()=>{controls.model.value=attnModel.value;refreshLayers();drawGeometry();drawAllAttention()});

refreshLayers();drawGeometry();drawAllAttention();window.addEventListener('resize',()=>{drawGeometry();drawAllAttention()});
</script>
</body>
</html>
'''


def js_payload(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip().replace("</", "<\\/")


def build(
    *,
    native_geometry: Path,
    nonthinking_geometry: Path,
    prompt_geometry: Path,
    retrieval_atlas: Path,
    output: Path,
) -> None:
    html = (
        HTML.replace("@@NATIVE@@", js_payload(native_geometry))
        .replace("@@NONTHINKING@@", js_payload(nonthinking_geometry))
        .replace("@@PROMPT@@", js_payload(prompt_geometry))
        .replace("@@ATLAS@@", js_payload(retrieval_atlas))
    )
    if "@@" in html:
        raise RuntimeError("Unresolved report template placeholder")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8", newline="\n")
    temporary.replace(output)
    print(output)
    print(output.stat().st_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-geometry", type=Path, required=True)
    parser.add_argument("--nonthinking-geometry", type=Path, required=True)
    parser.add_argument("--prompt-geometry", type=Path, required=True)
    parser.add_argument("--retrieval-atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        native_geometry=args.native_geometry,
        nonthinking_geometry=args.nonthinking_geometry,
        prompt_geometry=args.prompt_geometry,
        retrieval_atlas=args.retrieval_atlas,
        output=args.output,
    )
