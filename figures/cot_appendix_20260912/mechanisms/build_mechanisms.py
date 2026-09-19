"""Create CoT appendix mechanism figures from fixed local measurements only.

Run from the workspace: python -s figures/cot_appendix_20260912/mechanisms/build_mechanisms.py
All outputs stay in this directory. Existing figure and manuscript assets are untouched.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import csv, hashlib, json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np

START=time.perf_counter()
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
LR=Path('\\\\?\\'+str(ROOT))
REPO=LR/'realistic'
DATA=LR/'figures/cot-reasoning/data'
SOURCES={}; QA={}; OUTPUTS={}; METHODS={}
MODELS=['Qwen3-8B','Gemma4-E4B']; SHORT=['Qwen','Gemma']
COLORS=['#168DCA','#E87824']; GRAY='#737373'; GRID='#E5E8EC'
(OUT/'data').mkdir(parents=True,exist_ok=True)
(OUT/'qa').mkdir(exist_ok=True)
for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts')/name))
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.titlesize':11,'axes.titleweight':'normal',
 'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
 'axes.edgecolor':'#788495','axes.labelcolor':'#252525','text.color':'#252525',
 'xtick.color':'#252525','ytick.color':'#252525','axes.linewidth':.6,
 'lines.linewidth':1.5,'legend.frameon':False,'savefig.facecolor':'white'})

def source(p):
    SOURCES[p.relative_to(LR).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def js(p):return json.loads(source(p).read_text(encoding='utf-8'))
def csvread(p):return list(csv.DictReader(source(p).open(encoding='utf-8-sig',newline='')))
def jsonl(p):return [json.loads(x) for x in source(p).read_text(encoding='utf-8').split('\n') if x.strip()]
def flag(v):return str(v).lower() in ['1','1.0','true']
def export(name,rows):
    assert rows
    with (OUT/'data'/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(a,seed=20260912):
    a=np.asarray(a,float);assert a.ndim==1 and np.isfinite(a).all()
    draw=np.random.default_rng(seed).integers(0,len(a),(10000,len(a)))
    low,high=np.quantile(a[draw].mean(axis=1),[.025,.975])
    return float(a.mean()),float(low),float(high)
def axis(ax,title,ylabel=None,xlabel=None):
    ax.set_title(title,loc='left',pad=8)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color=GRID,lw=.5,zorder=0);ax.set_axisbelow(True)
    ax.tick_params(length=2.5,width=.6)
    if ylabel:ax.set_ylabel(ylabel)
    if xlabel:ax.set_xlabel(xlabel)
def err(ax,x,val,color,marker='o',ms=4):
    m,lo,hi=val
    ax.errorbar(x,m,yerr=[[m-lo],[hi-m]],color=color,marker=marker,ms=ms,capsize=2.5,lw=1,zorder=4)
def model_ticks(ax):ax.set_xticks([0,1],SHORT);ax.set_xlim(-.45,1.45)
def save(fig,name,title):
    assert abs(fig.get_figwidth()-6.5)<1e-12
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    boxes=[(t.get_text(),matplotlib.text.Text.get_window_extent(t,renderer)) for t in texts]
    outside=[s for s,b in boxes if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width+.5 or b.y1 > fig.bbox.height+.5]
    overlap=[]
    for i,(s,b) in enumerate(boxes):
        for t,c in boxes[i+1:]:
            if min(b.x1,c.x1)-max(b.x0,c.x0)>1.5 and min(b.y1,c.y1)-max(b.y0,c.y0)>1.5:
                overlap.append([s,t])
    assert not outside,(name,outside)
    assert not overlap,(name,overlap)
    QA[name]={'width_inches':6.5,'height_inches':fig.get_figheight(),'outside_text':outside,'text_overlaps':overlap,'title_pt':11,'axis_pt':10,'tick_legend_pt':9}
    for ext in ['pdf','svg','png']:
        p=OUT/f'{name}.{ext}'
        kw={'dpi':220} if ext=='png' else {}
        if ext=='pdf':kw['metadata']={'Title':title,'Author':'Anonymous Authors','CreationDate':None}
        fig.savefig(p,**kw)
        OUTPUTS[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    plt.close(fig)

def progress():
    p=REPO/'work/same_site_progress_transplant_20260827/n10_patch_scope_layer_sweep_v2/layer_sweep_analysis.json'
    j=js(p);cells=j['cells'];assert len(cells)==4320
    scopes=['item_end_w1','event_tail_w4','item_span'];labels=['Endpoint','Four-token tail','Item span'];styles=[':', '--','-']
    seeds=sorted({r['seed'] for r in cells});assert len(seeds)==20
    plot=[]
    for direction in ['forward_skip','backward_rewind']:
        for scope in scopes:
            for layer in range(36):
                rr=[r for r in cells if r['direction']==direction and r['scope']==scope and r['layer']==layer]
                assert len(rr)==20 and {r['seed'] for r in rr}==set(seeds)
                values=[next(r['paired_logodds_shift'] for r in rr if r['seed']==s) for s in seeds]
                m,lo,hi=ci(values)
                plot.append(dict(direction=direction,scope=scope,layer=layer+1,mean=m,ci95_low=lo,ci95_high=hi,seeds=20,patches=20))
    export('progress_discovery_layers.csv',plot)
    confirmation=csvread(DATA/'progress_state_cells.csv')
    saved=csvread(DATA/'progress_state_plot.csv');summary=[]
    for scope in scopes:
        rr=[r for r in confirmation if r['scope']==scope]
        assert len(rr)==60 and Counter(int(r['seed']) for r in rr)==dict.fromkeys(sorted({int(r['seed']) for r in rr}),6)
        target=next(r for r in saved if r['scope']==scope)
        assert sum(int(r['patched_adoption']) for r in rr)==int(target['successes'])
        summary.append(dict(scope=scope,layer=int(target['layer'])+1,mean=float(target['mean']),ci95_low=float(target['ci95_low']),ci95_high=float(target['ci95_high']),successes=int(target['successes']),patches=60,seeds=10,self_adoption=0.0))
    export('progress_confirmation_scopes.csv',summary)
    fig=plt.figure(figsize=(6.5,4.35))
    axs=[fig.add_axes([.105,.575,.365,.31]),fig.add_axes([.605,.575,.365,.31]),fig.add_axes([.20,.135,.72,.265])]
    ymax=max(r['ci95_high'] for r in plot);ymin=min(0,min(r['ci95_low'] for r in plot))
    for ax,direction,title in zip(axs[:2],['forward_skip','backward_rewind'],['A. Forward: discovery','B. Backward: discovery']):
        axis(ax,title,'City log-odds change','Intervention layer')
        for scope,label,style in zip(scopes,labels,styles):
            rr=[r for r in plot if r['scope']==scope and r['direction']==direction]
            x=[r['layer'] for r in rr];y=[r['mean'] for r in rr]
            ax.plot(x,y,color=COLORS[0],ls=style,label=label)
            ax.fill_between(x,[r['ci95_low'] for r in rr],[r['ci95_high'] for r in rr],color=COLORS[0],alpha=.06,lw=0)
        ax.axhline(0,color=GRAY,lw=.6);ax.set_xlim(1,36);ax.set_xticks([1,10,20,30,36]);ax.set_ylim(ymin-2,ymax+2);ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.legend([Line2D([],[],color=COLORS[0],ls=s) for s in styles],labels,loc='upper center',bbox_to_anchor=(.54,1.0),ncol=3,handlelength=2.1)
    ax=axs[2];axis(ax,'C. Scope controls: confirmation','Target-successor adoption')
    for x,r in enumerate(summary):
        err(ax,x,(r['mean'],r['ci95_low'],r['ci95_high']),COLORS[0])
        ax.plot(x,0,'o',mfc='white',mec=GRAY,ms=4,zorder=5)
        ax.text(x,r['ci95_high']+.07,f"{r['successes']}/60",ha='center',fontsize=9)
    ax.set_xticks([0,1,2],['Endpoint\nL27','Four-token tail\nL1','Item span\nL1']);ax.set_xlim(-.45,2.45);ax.set_ylim(-.055,1.06);ax.set_yticks([0,.5,1]);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.text(.01,.93,'Hollow: self patch',transform=ax.transAxes,ha='left',fontsize=9,color=GRAY)
    METHODS['progress']={'discovery_seeds':seeds,'discovery_scope_layer_cells':4320,'discovery_curve':'mean paired city log-odds shift; new descriptive pointwise seed-bootstrap intervals (10000,RNG20260912)','selection':'Original discovery median-of-seed-means 95% plateau rule retained; selected layers unchanged; plotted mean curve is not used to reselect layers.','confirmation':'three separately discovery-frozen geometries, same 60 cells and ten seeds; preserves original interval RNG20260911','limit':'Endpoint vs span changes both layer and geometry. Full-layer scans are discovery data, not confirmation.'}
    save(fig,'cot_progress_controls','CoT progress-state intervention scope and layer controls')

def carrier():
    damage=[];restore=[];rawplot=[];seedrows=[]
    for model in MODELS:
        p=REPO/f'work/v5_native_sample_aligned_20260829/runs/{model}/targeted_counter_write'
        gates=js(p/'analysis_confirmation/claim_gates.json');se=csvread(p/'analysis_confirmation/seed_effects.csv');assert len(se)==10
        for metric in ['selected_carrier_deformation','selected_carrier_deformation_specificity','clean_carrier_restoration']:
            r=next(r for r in gates['all_estimands'] if r['estimand']==metric)
            assert abs(np.mean([float(s[metric]) for s in se])-r['mean_effect'])<1e-12
            row=dict(model=model,estimand=metric,mean=r['mean_effect'],ci95_low=r['ci_low'],ci95_high=r['ci_high'],seeds=10)
            (restore if metric=='clean_carrier_restoration' else damage).append(row)
        raw=sum([jsonl(f) for f in sorted((p/'confirmation/shards').glob('*.jsonl'))],[]);assert len(raw)==70
        for cond in ['selected_mask','selected_mask_clean_carrier_restore','selected_mask_matched_position_state_control']:
            rr=sorted([r for r in raw if r['condition']==cond],key=lambda r:r['seed']);assert len(rr)==10
            a=[r['boundary_state_rms_distance_to_clean_final'] for r in rr];m,lo,hi=ci(a)
            rawplot.append(dict(model=model,condition=cond,mean=m,ci95_low=lo,ci95_high=hi,seeds=10))
            seedrows += [dict(model=model,condition=cond,seed=r['seed'],item_end_rms=r['boundary_state_rms_distance_to_clean_final']) for r in rr]
    export('carrier_deformation.csv',damage);export('carrier_restoration.csv',restore);export('carrier_raw_conditions.csv',rawplot);export('carrier_raw_seeds.csv',seedrows)
    fig,axs=plt.subplots(2,2,figsize=(6.5,4.55));fig.subplots_adjust(left=.105,right=.975,bottom=.13,top=.89,wspace=.42,hspace=.80)
    ax=axs[0,0];axis(ax,'A. Carrier deformation','Carrier RMS change');model_ticks(ax)
    for model,x,color in zip(MODELS,[0,1],COLORS):
        for metric,dx,marker in [('selected_carrier_deformation',-.12,'o'),('selected_carrier_deformation_specificity',.12,'D')]:
            r=next(r for r in damage if r['model']==model and r['estimand']==metric);err(ax,x+dx,(r['mean'],r['ci95_low'],r['ci95_high']),color,marker)
    ax.set_ylim(0,.225);ax.set_yticks([0,.1,.2])
    fig.legend([Line2D([],[],color=GRAY,marker='o',lw=0),Line2D([],[],color=GRAY,marker='D',lw=0)],['Selected mask','Selected minus random'],loc='upper center',bbox_to_anchor=(.53,1.0),ncol=2)
    ax=axs[0,1];axis(ax,'B. Item-end restoration','Item-end RMS reduction');model_ticks(ax)
    for model,x,color in zip(MODELS,[0,1],COLORS):
        r=next(r for r in restore if r['model']==model);err(ax,x,(r['mean'],r['ci95_low'],r['ci95_high']),color)
    ax.set_ylim(0,.39);ax.set_yticks([0,.1,.2,.3])
    for model,color,ax,letter in zip(MODELS,COLORS,axs[1],['C','D']):
        axis(ax,f'{letter}. {SHORT[MODELS.index(model)]}: position control','Item-end RMS distance')
        rr=[r for r in rawplot if r['model']==model]
        for x,r in enumerate(rr):err(ax,x,(r['mean'],r['ci95_low'],r['ci95_high']),color)
        ax.set_xticks([0,1,2],['Masked','Carrier\nrestored','Position\ncontrol']);ax.set_xlim(-.4,2.4);ax.set_ylim(0,max(r['ci95_high'] for r in rr)*1.12);ax.yaxis.set_major_locator(MaxNLocator(4))
    METHODS['carrier']={'population':'same ten canonical confirmation final transitions/model as head ablation; seven conditions and ten seeds/model','A':'Selected carrier deformation and selected-minus-mean-random deformation; original registered seed-bootstrap intervals','B':'Masked item-end RMS minus clean-carrier-restored RMS; original registered intervals','C_D':'Absolute item-end RMS from clean for three raw conditions; pointwise 10000 seed-bootstrap intervals,RNG20260912; each axis begins at zero and retains the full position-control magnitude','limits':'Position control matches token count and nearby registered position/depth, NOT the realized perturbation norm. It strongly perturbs the state. Carrier assay uses fixed tokens and does not show free-generation repair.'}
    save(fig,'cot_carrier_controls','CoT carrier deformation, restoration, and position controls')

def answer_readout():
    answer=[];control=[];details=csvread(DATA/'answer_state_detail.csv');saved=csvread(DATA/'answer_state_plot.csv')
    for model,L in zip(MODELS,[36,42]):
        rr=[r for r in details if r['model_label']==model];assert len(rr)==80*L
        for layer in range(L):
            selfrows=[r for r in rr if int(r['layer'])==layer and r['condition']=='self_patch'];assert len(selfrows)==40
            assert all(flag(r['receiver_retention']) and not flag(r['donor_adoption']) for r in selfrows)
            s=next(r for r in saved if r['model']==model and int(r['layer'])==layer)
            patch=[r for r in rr if int(r['layer'])==layer and r['condition']=='full_donor_patch'];assert len(patch)==40
            seedvals=[np.mean([flag(r['donor_adoption']) for r in patch if int(r['seed'])==seed]) for seed in range(1254,1264)]
            assert abs(np.mean(seedvals)-float(s['mean']))<1e-12
            answer.append(dict(model=model,layer=layer+1,condition='full_donor_patch',mean=float(s['mean']),ci95_low=float(s['ci95_low']),ci95_high=float(s['ci95_high']),pairs=40,seeds=10))
            control.append(dict(model=model,layer=layer+1,condition='self_patch',mean=0,ci95_low=0,ci95_high=0,pairs=40,seeds=10))
    blank=csvread(DATA/'answer_sources_plot.csv');bc=csvread(DATA/'answer_sources_cells.csv')
    for r in blank:
        rr=[s for s in bc if s['model']==r['model'] and s['condition']==r['condition']];assert len(rr)==100
        assert abs(np.mean([int(s['exact']) for s in rr])-float(r['mean']))<1e-12
    relay=js(REPO/'work/cot_completion_20260912/terminal_intervals/summary.json')
    assert relay['status']=='PASS_RECOMPUTED_INTERVALS'
    source(REPO/'work/cot_completion_20260912/terminal_intervals/audit.json')
    pairs=csvread(REPO/'work/cot_completion_20260912/terminal_intervals/common_support_pairs.csv');assert len(pairs)==86
    metrics=['patch_damage_natural','patch_damage__post_terminal_suffix','specific_mediation__post_terminal_suffix','specific_mediation__answer_query'];rp=[]
    for model,suffix in zip(MODELS,['qwen','gemma']):
        for metric in metrics:
            r=relay['models'][model]['effects'][metric]
            value=np.mean([np.mean([float(p[metric+'__'+suffix]) for p in pairs if int(p['seed'])==s]) for s in range(1254,1264)])
            assert abs(value-r['estimate'])<1e-12
            rp.append(dict(model=model,estimand=metric,mean=value,ci95_low=r['ci_low'],ci95_high=r['ci_high'],pairs=86,seeds=10))
    export('answer_layer_controls.csv',answer+control);export('answer_blanking.csv',blank);export('terminal_decomposition.csv',rp)
    fig,axs=plt.subplots(2,2,figsize=(6.5,4.8));fig.subplots_adjust(left=.18,right=.975,bottom=.13,top=.92,wspace=.67,hspace=.85)
    for model,color,ax,title in zip(MODELS,COLORS,axs[0],['A. Qwen: answer sources','B. Gemma: answer sources']):
        axis(ax,title,'Exact-count accuracy')
        for x,condition in enumerate(['clean','prompt_records_blank','trace_all_blank']):
            r=next(r for r in blank if r['model']==model and r['condition']==condition);err(ax,x,tuple(float(r[k]) for k in ['mean','ci95_low','ci95_high']),color)
        ax.set_xticks([0,1,2],['Original','Prompt\nblank','Trace\nblank']);ax.set_xlim(-.4,2.4);ax.set_ylim(-.04,1.06);ax.set_yticks([0,.5,1]);ax.yaxis.set_major_formatter(PercentFormatter(1))
    for model,color,ax,title in zip(MODELS,COLORS,axs[1],['C. Qwen: terminal pathway','D. Gemma: terminal pathway']):
        axis(ax,title,None,'Count-score margin change');ax.grid(False);ax.grid(axis='x',color=GRID,lw=.5);ax.axvline(0,color=GRAY,lw=.6)
        for y,metric in enumerate(metrics):
            r=next(r for r in rp if r['model']==model and r['estimand']==metric)
            ax.errorbar(r['mean'],3-y,xerr=[[r['mean']-r['ci95_low']],[r['ci95_high']-r['mean']]],color=color,marker='o',ms=4,capsize=2.5,lw=1)
        ax.set_yticks([3,2,1,0],['Natural\ndamage','Suffix\nresidual','Suffix\nmediated','Query\nmediated']);ax.set_ylim(-.5,3.5);ax.set_xlim(0,6.0);ax.set_xticks([0,2,4,6])
    METHODS['answer_readout']={'answer_patch_validation':'Full answer-layer data and self controls are still verified/exported; duplicated curves are omitted here because the main mechanism figure shows all layers.','A_B':'100 prompts/model/condition;10 seeds x N1..10; frozen original/record blank/whole trace blank; corrected Gemma replay; 10000 seed-bootstrap draws, RNG20260912','C_D':'86 common-support within-trace pairs/model;49 prompts,10 seeds; average pairs inside seed,then equal seeds; recomputed 10000-draw intervals,RNG20260912; query-only and suffix effects are nested, NOT additive','backend_validation':'Complete paired Gemma replay preserves original inputs and banks; corrected trace-blank accuracy15%, original12%. Main plotted values use corrected data.','terminal_intervals':'Point estimates reproduce to 1e-12; all displayed intervals recomputed with10000draws,RNG20260912,from unchanged common-support cells; archived intervals preserved separately.'}
    save(fig,'cot_answer_readout_controls','CoT source blanking and terminal pathway controls')

def incomplete_mediation():
    metrics=['full_state_effect_intact','full_selected_mask_interaction','full_head_output_restore','full_selected_vs_random_specificity'];rows=[]
    for geometry in ['endpoint','suffix4','suffix8']:
        j=js(REPO/f'work_remote_snapshots/gemma_query_mediation_{geometry}_discovery_claim_gates.json')
        assert j['seed_count']==20 and not j['geometry_pass'] and not j['confirmation_eligible']
        for metric in metrics:
            r=next(r for r in j['estimands'] if r['estimand']==metric);assert r['n_seeds']==20 and r['pair_count']==40
            rows.append(dict(geometry=geometry,estimand=metric,mean=r['mean_effect'],ci95_low=r['ci_low'],ci95_high=r['ci_high'],seeds=20,pairs=40,phase='discovery',confirmation_opened=False))
    export('gemma_incomplete_mediation.csv',rows)
    fig,axs=plt.subplots(2,2,figsize=(6.5,4.3));fig.subplots_adjust(left=.15,right=.975,bottom=.13,top=.92,wspace=.56,hspace=.68)
    for metric,ax,title in zip(metrics,axs.flat,['A. Full-state effect','B. Selected-mask interaction','C. Head-output restoration','D. Selected minus random']):
        axis(ax,title,None,'City log-odds effect');ax.grid(False);ax.grid(axis='x',color=GRID,lw=.5);ax.axvline(0,color=GRAY,lw=.7)
        rr=[r for r in rows if r['estimand']==metric]
        lo=min(0,min(r['ci95_low'] for r in rr));hi=max(r['ci95_high'] for r in rr);pad=(hi-lo)*.12
        for y,r in enumerate(rr):
            ax.errorbar(r['mean'],2-y,xerr=[[r['mean']-r['ci95_low']],[r['ci95_high']-r['mean']]],marker='o',color=COLORS[1],ms=4,capsize=2.5,lw=1)
        ax.set_yticks([2,1,0],['Endpoint','Suffix 4','Suffix 8']);ax.set_ylim(-.45,2.45);ax.set_xlim(lo-pad,hi+pad)
        ticks=MaxNLocator(4).tick_values(lo-pad,hi+pad)
        ax.set_xticks([v for v in ticks if lo-pad<=v<=hi+pad])
    METHODS['incomplete_mediation']={'model':'Gemma4-E4B','population':'20 discovery seeds;40 +/-1 directed pairs per geometry;endpoint/suffix4/suffix8; frozen Top6','estimands':metrics,'intervals':'original 95% seed-bootstrap intervals','gate':'All three geometries fail the joint full-state effect / selected-mask interaction / selected-head-output restoration positivity gate; confirmation NOT opened. Head-output restoration intervals span zero.','interpretation':'This tested pre-output restoration procedure does not complete the narrow serial path. It does not negate the separately established full-state effect or bank necessity. Independent horizontal scales for the four estimands.'}
    save(fig,'cot_gemma_mediation_limits','Gemma discovery mediation assay and its limits')

def summary_controls():
    restoration=csvread(DATA/'carrier_restore_plot.csv')
    blank=csvread(DATA/'answer_sources_plot.csv')
    fraction=csvread(DATA/'terminal_mediation_plot.csv')
    fig=plt.figure(figsize=(6.5,4.15))
    axs=[fig.add_axes([.11,.56,.35,.30]),fig.add_axes([.61,.56,.35,.30]),fig.add_axes([.21,.14,.72,.24])]
    ax=axs[0];axis(ax,'A. Item-end restoration','Item-end RMS reduction');model_ticks(ax)
    for model,x,color in zip(MODELS,[0,1],COLORS):
        r=next(r for r in restoration if r['model']==model)
        err(ax,x,tuple(float(r[k]) for k in ['mean','ci95_low','ci95_high']),color)
    ax.set_ylim(0,.39);ax.set_yticks([0,.1,.2,.3])
    ax=axs[1];axis(ax,'B. Answer sources','Exact-count accuracy')
    for model,offset,color in zip(MODELS,[-.085,.085],COLORS):
        for x,condition in enumerate(['clean','prompt_records_blank','trace_all_blank']):
            r=next(r for r in blank if r['model']==model and r['condition']==condition)
            err(ax,x+offset,tuple(float(r[k]) for k in ['mean','ci95_low','ci95_high']),color,ms=3.5)
    ax.set_xticks([0,1,2],['Original','Prompt\nblank','Trace\nblank']);ax.set_xlim(-.45,2.45);ax.set_ylim(-.045,1.065);ax.set_yticks([0,.5,1]);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax=axs[2];axis(ax,'C. Partial terminal mediation',None,'Suffix-mediated fraction')
    ax.grid(False);ax.grid(axis='x',color=GRID,lw=.5)
    for model,y,color in zip(MODELS,[1,0],COLORS):
        r=next(r for r in fraction if r['model']==model);m,lo,hi=[float(r[k]) for k in ['mean','ci95_low','ci95_high']]
        ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],color=color,marker='o',ms=4,capsize=2.5,lw=1)
    ax.set_yticks([1,0],SHORT);ax.set_ylim(-.6,1.6);ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.xaxis.set_major_formatter(PercentFormatter(1))
    fig.legend([Line2D([],[],color=c,marker='o',lw=1) for c in COLORS],SHORT,loc='upper center',bbox_to_anchor=(.54,1.0),ncol=2)
    METHODS['summary_controls']={'purpose':'Drop-in replacement for old figure native-counter-readout-controls; preserves A restoration / B blanking / C fraction and all existing panel references','intervals':'Points and95%seed-bootstrap intervals validated from audited plotting CSVs. A retains archived intervals; B/C use10000draws,RNG20260912.','population':'A ten matched transitions/model; B 100 prompts/condition/model; C86 common directed pairs and10 seeds/model','limits':'Fixed-token item-end RMS restoration is not free-generation rescue. Terminal mediation partial; suffix ratio intervals reproducibly recomputed. Gemma blanking uses corrected-backend data.'}
    save(fig,'cot_summary_controls','CoT state restoration, answer sources, and partial terminal mediation')

if __name__=='__main__':
    for fn in [progress,carrier,answer_readout,incomplete_mediation,summary_controls]:fn()
    manifest={'status':'PASS_DATA_AND_TEXT_LAYOUT','source_sha256':SOURCES,'output_sha256':OUTPUTS,'layout':QA,'methods':METHODS,'elapsed_seconds':time.perf_counter()-START,'render_visual_review':'PENDING','source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Read-only existing data; no inference, no original figure/data/manuscript overwrite.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'pdfs':[k for k in OUTPUTS if k.endswith('.pdf')],'elapsed_seconds':manifest['elapsed_seconds']}))
