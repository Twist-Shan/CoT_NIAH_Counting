"""Build the editable Non-thinking section figure from verified measurements.

Run with python. Then export nonthinking_section_v1.drawio.
Requires the exact confirmation full-span subset in data/reverse_layer_summary.csv.
The source CSV SHA256 is retained in data/reverse_figure_source.md.
"""
from pathlib import Path
from collections import defaultdict
import csv, gzip, json, hashlib, base64, copy
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1]
REPO=WORK/'realistic'
REPORTS=REPO/'reports/v4_non-thinking_causal'
DATA=OUT/'data'; DATA.mkdir(exist_ok=True)
MODELS=['Qwen3-8B','Gemma4-E4B']
COLORS=['#527D96','#B5835C']
INK='#30312E'; GRAY='#8190A5'; GRID='#E7E8EE'
sources={}
def record(p):
    sources[str(p.relative_to(WORK))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def readcsv(p):
    record(p)
    with (gzip.open(p,'rt',encoding='utf-8-sig') if p.suffix=='.gz' else p.open(encoding='utf-8-sig',newline='')) as f:
        return list(csv.DictReader(f))
def writecsv(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

exact=DATA/'reverse_layer_summary.csv'
assert exact.exists(), 'Exact corruption data are required; do not substitute rounded values.'
is_draft=False
reverse=readcsv(exact)
reverse=[r for r in reverse if r['population']=='confirmation' and r['condition']=='reverse_needle_full' and r['metric']=='expected_error_increase']
assert {m:sum(r['model']==m for r in reverse) for m in MODELS}==dict(zip(MODELS,[36,42]))
writecsv(DATA/'corruption_plot.csv',reverse)

# Existing ablation estimand: absolute generated-count shift, 20 seed clusters.
ab=readcsv(REPORTS/'v4_4_report_additions/full_span_topk_raw_arms.csv')
ab+=readcsv(REPORTS/'v4_4_causal_v2/full_span_topk_k6_extension/top6_raw_arms.csv')
ab=[r for r in ab if r['metric']=='absolute_count_shift' and int(r['top_n'])==dict(zip(MODELS,[32,6]))[r['model_label']]]
assert len(ab)==4
writecsv(DATA/'ablation_plot.csv',ab)

# Match the existing all-layer single-layer answer-state screen, then cluster by seed.
answer=[]; rng=np.random.default_rng(20260909)
for model,short in zip(MODELS,['qwen','gemma']):
    p=REPO/f'exports/realistic_20260803_v4_4_causal_v2_{short}/run'/model/'numeric/causal_v2/analysis/tables/answer_patching_paired_effects.csv.gz'
    raw=readcsv(p)
    raw=[r for r in raw if r['patch_protocol']=='single_layer' and r['phase']=='screen' and r['site']=='answer_query']
    by=defaultdict(list)
    for r in raw:
        v=float(r['control_adjusted_transport'])
        assert np.isfinite(v)
        by[int(r['start_layer']),int(r['seed'])].append(v)
    layers=sorted({l for l,s in by}); seeds=sorted({s for l,s in by})
    draws=rng.integers(0,len(seeds),(10000,len(seeds)))
    for l in layers:
        values=np.array([np.mean(by[l,s]) for s in seeds]); lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975])
        answer.append(dict(model=model,layer=l,mean=float(values.mean()),ci95_low=float(lo),ci95_high=float(hi),seed_count=len(seeds)))
writecsv(DATA/'answer_patching_plot.csv',answer)

plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'mathtext.fontset':'stix','axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.7,'lines.linewidth':1.7,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'legend.frameon':False})
fig=plt.figure(figsize=(10,3.50),facecolor='white')
axes=[fig.add_axes(p) for p in [[.067,.21,.306,.60],[.477,.21,.163,.60],[.744,.21,.249,.60]]]
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    for s in ['left','bottom']:ax.spines[s].set_color(GRAY)
    ax.tick_params(color=GRAY,labelcolor=INK,length=3,pad=3)
    ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
    ax.axhline(0,color=GRAY,lw=.6)
for x,letter,title in [(0.01,'B','Source-state corruption'),(.415,'C','Head ablation'),(.677,'D','Answer-state patching')]:
    fig.text(x,.967,letter,fontweight='bold',fontsize=14,va='top')
    fig.text(x+.026,.967,title,fontweight='bold',fontsize=12,va='top')
ax=axes[0]
for m,c in zip(MODELS,COLORS):
    rr=sorted([r for r in reverse if r['model']==m],key=lambda r:int(r['layer']))
    x=[int(r['layer']) for r in rr];y=[float(r['mean']) for r in rr]
    ax.plot(x,y,color=c)
    if not is_draft:ax.fill_between(x,[float(r['ci95_low']) for r in rr],[float(r['ci95_high']) for r in rr],color=c,alpha=.16,lw=0)
ax.set(xlim=(-.5,41.5),ylim=(-.45,3.35),xticks=[0,10,20,30,40],yticks=[0,1,2,3],xlabel='Intervention layer',ylabel='Expected-count error increase')
ax.legend([Line2D([],[],color=c) for c in COLORS],['Qwen','Gemma'],loc='upper right',fontsize=9,handlelength=1.6)
ax=axes[1]
for i,(m,c) in enumerate(zip(MODELS,COLORS)):
    for cond,dx,marker in [('ranked',.11,'o'),('layer_matched_random',-.11,'D')]:
        r=next(r for r in ab if r['model_label']==m and r['condition']==cond)
        y=float(r['mean']);lo=float(r['ci95_low']);hi=float(r['ci95_high'])
        ax.errorbar(i+dx,y,yerr=[[y-lo],[hi-y]],fmt=marker,color=c,ms=4.8,capsize=2.5,lw=1.1,mfc=c if cond=='ranked' else 'white',mew=1.1)
ax.set(xlim=(-.5,1.5),ylim=(-.08,2.5),xticks=[0,1],xticklabels=['Qwen\n$K=32$','Gemma\n$K=6$'],yticks=[0,1,2],ylabel='Absolute count shift')
ax.legend([Line2D([],[],marker='o',color=INK,ls='',ms=4),Line2D([],[],marker='D',color=INK,ls='',mfc='white',ms=4)],['Top-$K$','Random'],loc='lower left',bbox_to_anchor=(-.05,1.04),ncol=2,fontsize=8.5,handletextpad=.35,columnspacing=.8,borderpad=0)
ax=axes[2]
for m,c in zip(MODELS,COLORS):
    rr=[r for r in answer if r['model']==m]; x=[r['layer'] for r in rr]
    ax.plot(x,[r['mean'] for r in rr],color=c)
    ax.fill_between(x,[r['ci95_low'] for r in rr],[r['ci95_high'] for r in rr],color=c,alpha=.16,lw=0)
ax.set(xlim=(-.5,41.5),ylim=(-.04,.94),xticks=[0,10,20,30,40],yticks=[0,.2,.4,.6,.8],xlabel='Intervention layer',ylabel='State-transfer effect')
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
fig.text(.067,.02,'B: rounded confirmation means; confidence bands pending exact source CSV.' if is_draft else 'Shading and error bars: 95% pointwise seed-bootstrap intervals.',fontsize=8,color='#64665F')
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'result_panels_v1.{ext}',dpi=220)
plt.close(fig)

# Preserve every supplied mechanism cell. Recolor semantic roles; compose a copy.
source=record(OUT/'transformer_mechanism_polished.drawio')
tree=ET.parse(source); model=tree.find('.//mxGraphModel'); root=model.find('root')
mapping={'#33332F':'#30312E','#C77A53':'#B87652','#B5B5AC':'#B1B0A7','#6A6C63':'#64665F','#0072B2':'#64665F','#159CC6':'#969182'}
for cell in root.findall('mxCell'):
    style=cell.get('style','')
    for a,b in mapping.items():style=style.replace(a,b)
    if '#969182' in style:style=style.replace('fillColor=#FFFFFF','fillColor=#E7E3D7')
    cell.set('style',style)
model.set('background','#FFFFFF')
tree.write(OUT/'mechanism_matched_v1.drawio',encoding='utf-8',xml_declaration=True)
# Combined 1700-unit-wide page. Mechanism occupies upper half; quantitative panels lower half.
for cell in root.findall('mxCell'):
    geo=cell.find('mxGeometry')
    if geo is not None:
        if cell.get('vertex')=='1':geo.set('y',str(float(geo.get('y','0'))+40))
        if cell.get('edge')=='1':
            for point in geo.iter('mxPoint'):
                if 'y' in point.attrib:point.set('y',str(float(point.get('y'))+40))
model.set('pageWidth','1700');model.set('pageHeight','1325')
def vertex(id,value,style,x,y,w,h):
    c=ET.SubElement(root,'mxCell',id=id,value=value,style=style,vertex='1',parent='1')
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
vertex('section-panel-a','A   Non-thinking mechanism','text;html=1;strokeColor=none;fillColor=none;fontFamily=Times New Roman;fontSize=30;fontStyle=1;fontColor=#30312E;align=left;',14,0,900,50)
svg=(OUT/'result_panels_v1.svg').read_bytes()
uri='data:image/svg+xml,'+base64.b64encode(svg).decode()
vertex('section-results','',f'shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image={uri};',0,710,1700,595)
tree.write(OUT/'nonthinking_section_v1.drawio',encoding='utf-8',xml_declaration=True)
manifest={'status':'LAYOUT_DRAFT_PENDING_EXACT_CORRUPTION_CSV' if is_draft else 'COMPLETE','sources':sources,'palette':dict(zip(MODELS,COLORS)),'mechanism_palette':mapping,'corruption':{'population':'confirmation 10 seeds x counts 1-10','layers':[36,42],'precision':'3 decimal places; no reconstructed CI' if is_draft else 'exact source CSV'},'ablation':{'seed_clusters':20,'metric':'absolute generated-count shift','gemma_top6':'post-hoc dose extension'},'answer':{'seed_counts':sorted({r['seed_count'] for r in answer}),'phase':'screen; single_layer','metric':'control_adjusted_transport','bootstrap':10000},'editable':'mechanism native draw.io cells; quantitative panel SVG, regenerated by this script'}
(OUT/'nonthinking_section_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':manifest['status'],'answer_seed_counts':manifest['answer']['seed_counts'],'answer_endpoints':{m:next(r['mean'] for r in reversed(answer) if r['model']==m) for m in MODELS}}))
