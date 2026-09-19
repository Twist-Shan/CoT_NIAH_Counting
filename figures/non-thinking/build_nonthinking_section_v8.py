"""Build the editable Non-thinking section figure from verified measurements.

Run with python. Then export nonthinking_section_v8.drawio.
Requires the exact confirmation full-span subset in data/reverse_layer_summary.csv.
The source CSV SHA256 is retained in data/reverse_figure_source.md.
"""
from pathlib import Path
from collections import defaultdict
import csv, gzip, json, hashlib, base64, copy, re
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import Bbox

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1]
REPO=WORK/'realistic'
REPORTS=REPO/'reports/v4_non-thinking_causal'
DATA=OUT/'data'; DATA.mkdir(exist_ok=True)
MODELS=['Qwen3-8B','Gemma4-E4B']
COLORS=['#168DCA','#E87824']
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
bidirectional=readcsv(DATA/'bidirectional_patch_plot.csv')
patch_audit=json.loads(record(DATA/'bidirectional_patch_audit.json').read_text())
assert patch_audit['status']=='PASS' and patch_audit['measurement_cells']==15600
assert patch_audit['seeds']==list(range(1254,1264))
for m,nlayers in zip(MODELS,[36,42]):
    for direction in ['restore','corrupt']:
        rr=[r for r in bidirectional if r['model']==m and r['direction']==direction]
        assert len(rr)==nlayers and {int(r['layer']) for r in rr}==set(range(nlayers))
for filename in ['restoration_confirmation.csv','corruption_confirmation.csv',
                 'restoration_confirmation_source_audit.json','corruption_confirmation_source_audit.json']:
    record(DATA/filename)

# Relative ablation effect: |generated_patch - generated_clean| / true count N.
# Average valid random replicates within prompt, then prompts within seed.
ab=[]; ab_seed=[]; ab_missing={}
ab_abs_reference=readcsv(REPORTS/'v4_4_report_additions/full_span_topk_raw_arms.csv')
ab_abs_reference+=readcsv(REPORTS/'v4_4_causal_v2/full_span_topk_k6_extension/top6_raw_arms.csv')
ab_abs_checks=[]
ab_paths=[REPO/'work/nonthinking_report_filestream_stage1/qwen_topk_detail.csv.gz',
          REPORTS/'v4_4_causal_v2/full_span_topk_k6_extension/provenance/detail.csv.gz']
ab_rng=np.random.default_rng(20260909)
for m,k,p in zip(MODELS,[32,6],ab_paths):
    raw=[r for r in readcsv(p) if int(r['top_n'])==k and r['condition'] in ['ranked','layer_matched_random']]
    assert len(raw)==400 and {r['model_label'] for r in raw}=={m}
    seeds=sorted({int(r['seed']) for r in raw})
    assert seeds==list(range(1316,1336))
    assert {int(r['gold_count']) for r in raw}==set(range(1,6))
    draws=ab_rng.integers(0,len(seeds),(10000,len(seeds)))
    for condition in ['original','layer_matched_random','ranked']:
        by=defaultdict(list)
        arm=[r for r in raw if r['condition']==('ranked' if condition=='original' else condition)]
        missing=0
        for r in arm:
            if condition!='original' and not r['generated_count_shift'].strip():
                missing+=1
                continue
            n=int(r['gold_count']);assert n>0
            value=0.0 if condition=='original' else abs(float(r['generated_count_shift']))/n
            assert np.isfinite(value)
            by[int(r['seed']),r['stimulus_id']].append(value)
        assert len(by)==100
        if condition!='original':
            gold_by_id={r['stimulus_id']:int(r['gold_count']) for r in arm}
            raw_absolute=float(np.mean([np.mean(v)*gold_by_id[stimulus] for (_,stimulus),v in by.items()]))
            reference=next(r for r in ab_abs_reference if r['model_label']==m and int(r['top_n'])==k and r['condition']==condition and r['metric']=='absolute_count_shift')
            assert abs(raw_absolute-float(reference['mean']))<1e-12
            ab_abs_checks.append(dict(model=m,condition=condition,recomputed=raw_absolute,audited=float(reference['mean'])))
        values=[]
        for seed in seeds:
            prompt_values=[np.mean(v) for (s,_),v in by.items() if s==seed]
            assert len(prompt_values)==5
            value=float(np.mean(prompt_values));values.append(value)
            ab_seed.append(dict(model_label=m,top_n=k,condition=condition,seed=seed,effect=value,prompts=5))
        values=np.asarray(values)
        lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975])
        ab.append(dict(model_label=m,top_n=k,metric='relative_absolute_count_shift',condition=condition,seed_clusters=20,prompts=100,mean=float(values.mean()),ci95_low=float(lo),ci95_high=float(hi),missing_generation_rows=missing))
        ab_missing[m+' / '+condition]=missing
writecsv(DATA/'ablation_plot.csv',ab)
writecsv(DATA/'ablation_relative_seed_effects.csv',ab_seed)
record(WORK/'figures/additional_tasks_aurora/build_figures.py')

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

plt.rcParams.update({'font.family':'Times New Roman','font.size':11.5,'mathtext.fontset':'stix','axes.labelsize':12,'axes.titlesize':13,'xtick.labelsize':11.5,'ytick.labelsize':11.5,'axes.linewidth':.8,'lines.linewidth':1.9,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'legend.frameon':False})
fig=plt.figure(figsize=(10,2.65),facecolor='white')
# Equal visual gutters include each panel's y-label, ticks, legend and title.
panel_widths=[.353,.225,.353]
panel_gutter=.021
panel_starts=[.012,.012+panel_widths[0]+panel_gutter,.012+sum(panel_widths[:2])+2*panel_gutter]
axis_rects=[[panel_starts[0]+.068,.210,.285,.57],[panel_starts[1]+.060,.210,.165,.57],[panel_starts[2]+.068,.210,.285,.57]]
axes=[fig.add_axes(p) for p in axis_rects]
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    for s in ['left','bottom']:ax.spines[s].set_color(GRAY)
    ax.tick_params(color=GRAY,labelcolor=INK,length=3,pad=3)
    ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
    ax.axhline(0,color=GRAY,lw=.6)
panel_headers=[]
for x,letter,title in zip(panel_starts,'BCD',['Source-state patching','Head ablation','Answer-state patching']):
    panel_headers.append([
        fig.text(x,.967,letter,fontweight='bold',fontsize=14,va='top'),
        fig.text(x+.026,.967,title,fontweight='bold',fontsize=13,va='top')])
ax=axes[0]
for m,c in zip(MODELS,COLORS):
    for direction,linestyle in [('restore','-'),('corrupt',(0,(3.5,2.2)))]:
        rr=sorted([r for r in bidirectional if r['model']==m and r['direction']==direction],key=lambda r:int(r['layer']))
        x=[int(r['layer']) for r in rr];y=[float(r['mean']) for r in rr]
        ax.plot(x,y,color=c,ls=linestyle,lw=1.9 if direction=='restore' else 1.65)
        ax.fill_between(x,[float(r['ci95_low']) for r in rr],[float(r['ci95_high']) for r in rr],color=c,alpha=.10,lw=0)
ax.set(xlim=(-.5,41.5),ylim=(-.09,.69),xticks=[0,10,20,30,40],yticks=[0,.2,.4,.6],xlabel='Intervention layer',ylabel='Effect')
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
model_handles=[Line2D([],[],color=c) for c in COLORS]
model_legend=ax.legend(model_handles,['Qwen','Gemma'],loc='lower center',bbox_to_anchor=(.5,1.025),ncol=2,fontsize=11.5,handlelength=1.6,borderpad=0,borderaxespad=0)
ax.add_artist(model_legend)
ax.legend([Line2D([],[],color=INK,ls='-'),Line2D([],[],color=INK,ls=(0,(3.5,2.2)))],
          ['Restore','Corrupt'],loc='upper right',bbox_to_anchor=(.99,.99),fontsize=10.5,
          handlelength=1.8,borderpad=.15,labelspacing=.3,handletextpad=.55)
ax=axes[1]
for i,(m,c) in enumerate(zip(MODELS,COLORS)):
    top=next(r['mean'] for r in ab if r['model_label']==m and r['condition']=='ranked')
    ax.vlines(i,0,top,color=c,lw=.8,alpha=.55,zorder=2)
    for cond,marker in [('original','o'),('layer_matched_random','D'),('ranked','o')]:
        r=next(r for r in ab if r['model_label']==m and r['condition']==cond)
        y=float(r['mean']);lo=float(r['ci95_low']);hi=float(r['ci95_high'])
        ax.errorbar(i,y,yerr=None if cond=='original' else [[y-lo],[hi-y]],fmt=marker,color=c,ms={'original':4.2,'layer_matched_random':3.8,'ranked':5.2}[cond],capsize=2.5,lw=1.0,mfc='white' if cond=='original' else c,mew=1.0,zorder=4)
        if cond!='original':
            dx,align=(-7,'right') if i==0 else (7,'left')
            ax.annotate(f'{100*y:.0f}%',(i,y),xytext=(dx,5 if cond=='ranked' else 2),textcoords='offset points',color=c,ha=align,va='bottom',fontsize=10)
ax.set(xlim=(-.65,1.65),ylim=(-.025,.78),xticks=[0,1],xticklabels=['Qwen','Gemma'],yticks=[0,.2,.4,.6],ylabel='Ablation effect')
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
for i,k in enumerate([32,6]):
    ax.text(i,-.22,f'$K={k}$',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=12)
ax.legend([Line2D([],[],marker='o',color=INK,ls='',mfc='white',ms=4),Line2D([],[],marker='D',color=INK,ls='',ms=4),Line2D([],[],marker='o',color=INK,ls='',ms=4)],['Original','Random','Top-$K$'],loc='lower center',bbox_to_anchor=(panel_starts[1]+panel_widths[1]/2,.210+.57*1.025),bbox_transform=fig.transFigure,ncol=3,fontsize=11.5,handlelength=.4,handletextpad=.45,columnspacing=.65,borderpad=0,borderaxespad=0)
ax=axes[2]
for m,c in zip(MODELS,COLORS):
    rr=[r for r in answer if r['model']==m]; x=[r['layer'] for r in rr]
    ax.plot(x,[r['mean'] for r in rr],color=c)
    ax.fill_between(x,[r['ci95_low'] for r in rr],[r['ci95_high'] for r in rr],color=c,alpha=.16,lw=0)
ax.set(xlim=(-.5,41.5),ylim=(-.04,.94),xticks=[0,10,20,30,40],yticks=[0,.2,.4,.6,.8],xlabel='Intervention layer',ylabel='State-transfer effect')
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax.legend(model_handles,['Qwen','Gemma'],loc='lower center',bbox_to_anchor=(.5,1.025),ncol=2,fontsize=11.5,handlelength=1.6,borderpad=0,borderaxespad=0)
for ax in axes:
    ax.xaxis.set_label_coords(.5,-.22)
    ax.yaxis.set_label_coords([-.15,-.22,-.19][axes.index(ax)],.5)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for ax,left in zip(axes,panel_starts):
    label=ax.yaxis.label
    shift=(left*fig.bbox.width-label.get_window_extent(renderer).x0)/ax.bbox.width
    ax.yaxis.set_label_coords(label.get_position()[0]+shift,.5)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
panel_boxes=[Bbox.union([ax.get_tightbbox(renderer)]+[t.get_window_extent(renderer) for t in header]) for ax,header in zip(axes,panel_headers)]
visible_gaps=[(panel_boxes[i+1].x0-panel_boxes[i].x1)/fig.bbox.width for i in range(2)]
assert abs(visible_gaps[0]-visible_gaps[1])*fig.bbox.width < 1, visible_gaps
text_boxes=[(a.get_text(),a.get_window_extent(renderer)) for a in fig.findobj(matplotlib.text.Text) if a.get_visible() and a.get_text()]
outside=[s for s,b in text_boxes if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height]
overlaps=[]
for i,(s,b) in enumerate(text_boxes):
    for t,c in text_boxes[i+1:]:
        if min(b.x1,c.x1)-max(b.x0,c.x0)>1 and min(b.y1,c.y1)-max(b.y0,c.y0)>1:
            overlaps.append([s,t])
assert not outside, outside
assert not overlaps, overlaps
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'result_panels_v8.{ext}',dpi=220)
# draw.io's SVG image renderer cannot resolve Matplotlib's local STIX font.
# Keep the text-editable SVG above, and embed identical outlined glyphs in the composite.
with plt.rc_context({'svg.fonttype':'path'}):
    fig.savefig(OUT/'result_panels_v8_outlined.svg')
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
tree.write(OUT/'mechanism_matched_v8.drawio',encoding='utf-8',xml_declaration=True)
# Equal column pitch; consistent row spacing and room around the answer-state chain.
cells={c.get('id'):c for c in root.findall('mxCell')}
columns=[330,566,802,1038,1274]
rows_y={'high':70,'mid':140,'low':210,'tokens':290,'positions':344}
def place(id,cx,cy,w,h,size,bold=False):
    cell=cells[id];g=cell.find('mxGeometry')
    g.set('x',str(cx-w/2));g.set('y',str(cy-h/2));g.set('width',str(w));g.set('height',str(h))
    style=re.sub(r'fontSize=[^;]+;',f'fontSize={size};',cell.get('style',''))
    style=re.sub(r'fontStyle=[^;]+;',f'fontStyle={1 if bold else 0};',style)
    cell.set('style',style)
for row in rows_y:
    place('row-'+row,134,rows_y[row],200,36,28,True)
position_ids=['position-1','XbEThH6K2D68lSNjHB_t-1','XbEThH6K2D68lSNjHB_t-2','XbEThH6K2D68lSNjHB_t-4']
for index,n in enumerate([1,2,3,10]):
    cx=columns[index]
    place('token-'+str(n),cx,rows_y['tokens'],144,44,32)
    place('low-'+str(n),cx,rows_y['low'],144,44,32)
    place(position_ids[index],cx,rows_y['positions'],144,38,30)
    p=cells['retrieve-branch-'+str(n)].find('.//mxPoint')
    p.set('x',str(cx));p.set('y',str(rows_y['mid']))
for i in range(4):
    place('ellipsis-'+str(i),(columns[i]+columns[i+1])/2,rows_y['tokens'],44,36,30)
    cells['ellipsis-'+str(i)].set('value',r'\(\cdots\)')
place('answer-token',columns[-1],rows_y['tokens'],144,44,32)
place('position-T',columns[-1],rows_y['positions'],144,38,30)
for row in ['low','mid','high']:place('answer-'+row,columns[-1],rows_y[row],144,44,32)
place('total',1510,rows_y['high'],200,44,32)
bank_center=(columns[0]+columns[3])/2
place('form-label',bank_center,(rows_y['tokens']+rows_y['low'])/2,180,34,28)
stage_label_y=(rows_y['high']+rows_y['mid'])/2
place('retrieve-label',bank_center,stage_label_y,200,34,28)
place('consolidate-label',1110,stage_label_y,200,34,28)
for label_id in ['form-label','retrieve-label','consolidate-label']:
    cell=cells[label_id]
    cell.set('style',re.sub(r'fontColor=[^;]+;', 'fontColor=#000000;',cell.get('style')))
p=cells['retrieval-path'].find('.//mxPoint');p.set('x',str(columns[0]));p.set('y',str(rows_y['mid']))
p=cells['readout'].find('.//mxPoint');p.set('x',str(columns[-1]+72));p.set('y',str(rows_y['high']))
for c in cells.values():
    if c.get('edge')=='1':c.set('style',c.get('style').replace('strokeWidth=3.75','strokeWidth=2.5').replace('endSize=12','endSize=6.5'))
tree.write(OUT/'mechanism_matched_v8.drawio',encoding='utf-8',xml_declaration=True)
model.set('pageWidth','1700');model.set('pageHeight','831')
def vertex(id,value,style,x,y,w,h):
    c=ET.SubElement(root,'mxCell',id=id,value=value,style=style,vertex='1',parent='1')
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
vertex('section-panel-a','A','text;html=1;strokeColor=none;fillColor=none;fontFamily=Times New Roman;fontSize=33;fontStyle=1;fontColor=#30312E;align=left;',20,0,35,44)
vertex('section-title-a','Non-thinking mechanism','text;html=1;strokeColor=none;fillColor=none;fontFamily=Times New Roman;fontSize=30.69;fontStyle=1;fontColor=#30312E;align=left;',64,0,820,44)
svg=(OUT/'result_panels_v8_outlined.svg').read_bytes()
uri='data:image/svg+xml,'+base64.b64encode(svg).decode()
vertex('section-results','',f'shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image={uri};',0,380,1700,450.5)
tree.write(OUT/'nonthinking_section_v8.drawio',encoding='utf-8',xml_declaration=True)
manifest={'status':'LAYOUT_DRAFT_PENDING_EXACT_CORRUPTION_CSV' if is_draft else 'COMPLETE','sources':sources,'palette':dict(zip(MODELS,COLORS)),'mechanism_palette':mapping,'corruption':{'population':'confirmation 10 seeds x counts 1-10','layers':[36,42],'precision':'3 decimal places; no reconstructed CI' if is_draft else 'exact source CSV'},'ablation':{'seed_clusters':20,'metric':'absolute generated-count shift','gemma_top6':'post-hoc dose extension'},'answer':{'seed_counts':sorted({r['seed_count'] for r in answer}),'phase':'screen; single_layer','metric':'control_adjusted_transport','bootstrap':10000},'editable':'mechanism native draw.io cells; quantitative panel SVG, regenerated by this script'}
manifest['layout']={'version':6,'result_axes':axis_rects,'axis_font_pt':12,'tick_font_pt':11.5,'legend_font_pt':11.5,'title_font_pt':13,'font_reference_width_inches':10,'at_7_inch_width':{'axis_pt':8.4,'ticks_and_legend_pt':8.05,'title_pt':9.1},'mechanism_column_centers':columns,'mechanism_row_centers':rows_y,'mechanism_text_size':28,'mechanism_math_size':32,'composition':{'width':1700,'mechanism_height':380,'results_height':450.5},'change_from_v5':{'upper_row_pitch':[50,70],'answer_arrow_clear_length':[14,26],'low_to_token_pitch':[82,80],'token_to_position_pitch':[50,54],'mechanism_text_size':[24,28],'mechanism_math_size':[26,32],'results_layout':'same plot widths/heights; full panel bounds aligned and equally spaced'}}
manifest['layout']['font_export']='Times New Roman text + STIX math; outlined result SVG in composite prevents font fallback; editable text SVG retained separately'
manifest['layout']['statistics_note']='Removed from artwork; retained in README caption'
manifest['layout']['render_check']={'text_outside_canvas':outside,'text_overlaps':overlaps,'visible_panel_gaps':visible_gaps,'panel_bounds':[[b.x0/fig.bbox.width,b.x1/fig.bbox.width] for b in panel_boxes]}
manifest['layout']['version']=8
manifest['layout']['mechanism_stage_label_color']='#000000'
manifest['ablation']={'seed_clusters':20,'prompts_per_model':100,'counts':[1,2,3,4,5],'metric':'abs(generated_patch - generated_clean) / gold_count','aggregation':'valid random replicates within prompt, five prompts within seed, then equal seed average','bootstrap':10000,'original':'zero by definition, relative to the same clean prediction','missing_generation_rows':ab_missing,'gemma_top6':'post-hoc dose extension','audited_absolute_mean_checks':ab_abs_checks,'ci_provenance':'recomputed from raw records, not copied from the reference image'}
manifest['palette_reference']='figures/additional_tasks_aurora/build_figures.py; synchronized 2026-09-09 after brighter-color update'
manifest['corruption']['metric']='(|E_patched-N|-|E_clean-N|)/N, evaluated per input before seed aggregation'
manifest['source_state_patching']={
    'directions':['restore','corrupt'],'line_styles':{'restore':'solid','corrupt':'dashed'},
    'cohort':'matched historical confirmation seeds 1254-1263, counts 1-10',
    'layers':{'Qwen3-8B':36,'Gemma4-E4B':42},'measurement_cells':15600,
    'restore_metric':'(|E_corrupt-N|-|E_restored-N|)/N',
    'corrupt_metric':'(|E_patched-N|-|E_clean-N|)/N',
    'aggregation':'per-input ratio, then equal counts within seed, then equal seeds',
    'bootstrap':10000,'bootstrap_seed':20260909,'intervals':'pointwise seed percentile bootstrap',
    'negative_effects':'retained','contemporary_restore_validation':'four layers per model; excluded from full-layer curves',
    'backend_caveat':patch_audit['backend_caveat']}
(OUT/'nonthinking_section_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':manifest['status'],'answer_seed_counts':manifest['answer']['seed_counts'],'answer_endpoints':{m:next(r['mean'] for r in reversed(answer) if r['model']==m) for m in MODELS}}))
