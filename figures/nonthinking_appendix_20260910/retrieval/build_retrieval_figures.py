"""Rebuild non-thinking retrieval appendix figures from immutable local CSVs.

Run: python figures/nonthinking_appendix_20260910/retrieval/build_retrieval_figures.py
No model execution or raw-data modification. The JSON audit records every input hash.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv, gzip, hashlib, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
REPO = ROOT / 'realistic'
REPORT = REPO / 'reports/v4_non-thinking_causal'
MODELS = ['Qwen3-8B', 'Gemma4-E4B']
DISPLAY_NAMES = {'Qwen3-8B': 'Qwen', 'Gemma4-E4B': 'Gemma'}
COLORS = ['#168DCA', '#E87824']
GRAY = '#7E8791'
INPUTS = {}

def read(path):
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    opener = gzip.open(path, 'rt', encoding='utf-8-sig') if path.suffix == '.gz' else path.open(encoding='utf-8-sig', newline='')
    with opener as stream:
        return list(csv.DictReader(stream))

def savecsv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#8D99A5', 'axes.linewidth': .65, 'text.color': '#30312E',
    'axes.labelcolor': '#30312E', 'xtick.color': '#30312E', 'ytick.color': '#30312E',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def style(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', pad=9)
    ax.grid(axis='y', color='#E7E8EE', linewidth=.6)
    ax.set_axisbelow(True)

def export(fig, name):
    # Fixed physical size; no tight bbox that silently rescales font sizes.
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=260, facecolor='white')
    plt.close(fig)

atlas = [r for r in read(REPORT / 'v4_4/realistic_niah_v4_head_atlas.csv') if r['variant'] == 'v4.4']
profiles = [r for r in read(REPORT / 'v4_4/realistic_niah_v4_head_phenotypes.csv') if r['variant'] == 'v4.4']
members = read(REPORT / 'v4_4_causal_v2/full_span_topk/full_span_topk_membership.csv')
bank_audit = {}
fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.35))
fig.subplots_adjust(left=.085, right=.985, bottom=.10, top=.93, wspace=.29, hspace=.55)
for j, (model, color, k) in enumerate(zip(MODELS, COLORS, [32, 6])):
    selected = [r for r in members if r['model_label'] == model and int(r['top_n']) == 32 and int(r['rank']) <= k]
    keys = {(r['layer'], r['head']) for r in selected}
    rows = [r for r in atlas if r['model'] == model and r['pooling'] == 'span_sum']
    bank = [r for r in rows if (r['layer'], r['head']) in keys]
    assert len(bank) == k and all(int(r['examples']) == 180 and int(r['seeds']) == 20 for r in rows)
    bank_audit[model] = {'head_count': k, 'discovery_seeds': list(range(1234,1254)), 'counts': list(range(2,11)),
        'mean_literal_mass': float(np.mean([float(r['pool_sum']) for r in bank])),
        'mean_effective_coverage': float(np.mean([float(r['pool_coverage']) for r in bank])),
        'layer_counts': dict(Counter(int(r['layer']) for r in bank)),
        'scope': 'all saved full-attention heads; Gemma sliding-attention layers excluded',
        'atlas_warning': 'descriptive atlas scores; frozen causal membership used without reranking'}
    ax = axs[0,j]; style(ax, f'{"AB"[j]}  {DISPLAY_NAMES[model]}: retrieval by layer')
    # Small deterministic head offsets prevent all heads at a layer being hidden.
    offsets = lambda rr: [int(r['layer']) + (int(r['head']) / (32 if j == 0 else 8) - .5) * .55 for r in rr]
    ax.scatter(offsets(rows), [float(r['pool_primary']) for r in rows], s=6, color=color, alpha=.20, edgecolor='none')
    ax.scatter(offsets(bank), [float(r['pool_primary']) for r in bank], s=17, color=color, edgecolor='white', linewidth=.25, zorder=4)
    end = [r for r in atlas if r['model'] == model and r['pooling'] == 'span_end']
    layers = sorted({int(r['layer']) for r in end})
    ax.plot(layers, [max(float(r['pool_primary']) for r in end if int(r['layer']) == l) for l in layers], '--', color=GRAY, lw=1.1)
    ax.set(xlim=(-1,35.9 if j == 0 else 42), ylim=(-.015,.61), xlabel='Layer', ylabel='Retrieval score')
    ax.set_xticks([0,10,20,30] if j == 0 else [5,17,29,41])
    ax.legend(handles=[Line2D([],[],marker='o',ls='',ms=3.5,color=color,label=f'Frozen top-{k}'),
        Line2D([],[],ls='--',color=GRAY,lw=1.1,label='Endpoint max.')], loc='upper left',frameon=False,handlelength=1.3)
    ax = axs[1,j]; style(ax, f'{"CD"[j]}  {DISPLAY_NAMES[model]}: attention across records')
    for q, chosen in enumerate(selected[:3]):
        p = next(r for r in profiles if r['model'] == model and r['layer'] == chosen['layer'] and r['head'] == chosen['head'])
        assert int(p['samples']) == 20
        vals = np.asarray(json.loads(p['span_sum_profile']))
        assert vals.shape == (10,) and abs(vals.sum() - 1) < 1e-6
        ax.plot(range(1,11), vals, color=color, ls=['-','--',':'][q], lw=1.5,
            marker=['o','s','^'][q], markersize=2.7, label=chosen['head_label'])
    ax.axhline(.1, color=GRAY, lw=.75, alpha=.75)
    ax.set(xlim=(.8,10.2), ylim=(0,.25), xlabel='Record index ($N=10$)', ylabel='Share of record attention')
    ax.set_xticks([1,2,4,6,8,10]); ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.legend(loc='upper right',frameon=False,ncol=1,handlelength=1.8,labelspacing=.15)
export(fig, 'retrieval_attention')

raw = []
for name in ['qwen','gemma']:
    raw += read(REPO / f'work/nonthinking_report_filestream_stage1/{name}_topk_detail.csv.gz')
raw += read(REPORT / 'v4_4_causal_v2/full_span_topk_k6_extension/provenance/detail.csv.gz')
assert len(raw) == 5200
assert {int(r['seed']) for r in raw} == set(range(1316,1336))
assert {int(r['gold_count']) for r in raw} == set(range(1,6))
ablation=[]; seed_rows=[]; rng=np.random.default_rng(20260910)
draws=rng.integers(0,20,size=(10000,20))
for model in MODELS:
    for k in sorted({int(r['top_n']) for r in raw if r['model_label'] == model}):
        for condition in ['ranked','layer_matched_random']:
            rr=[r for r in raw if r['model_label']==model and int(r['top_n'])==k and r['condition']==condition]
            assert len(rr)==(100 if condition=='ranked' else 300)
            for metric in ['relative_count_shift','absolute_count_shift','correct_to_wrong']:
                by=defaultdict(list); missing=0
                for r in rr:
                    if metric=='correct_to_wrong':
                        if r['baseline_is_correct']!='True' or r['baseline_format_valid']!='True': continue
                        val=float(r['patched_is_correct']!='True')
                    else:
                        if not r['generated_count_shift'].strip(): missing+=1; continue
                        val=abs(float(r['generated_count_shift']))
                        if metric=='relative_count_shift': val/=int(r['gold_count'])
                    by[int(r['seed']),r['stimulus_id']].append(val)
                seed_values=[]
                for seed in range(1316,1336):
                    pv=[np.mean(v) for (s,_),v in by.items() if s==seed]
                    assert len(pv)==5 if metric!='correct_to_wrong' else len(pv)>0
                    seed_values.append(float(np.mean(pv)))
                    seed_rows.append(dict(model=model,k=k,condition=condition,metric=metric,seed=seed,value=seed_values[-1],prompts=len(pv)))
                arr=np.asarray(seed_values);lo,hi=np.quantile(arr[draws].mean(axis=1),[.025,.975])
                ablation.append(dict(model=model,k=k,condition=condition,metric=metric,mean=float(arr.mean()),ci95_low=float(lo),ci95_high=float(hi),seeds=20,prompts=len(by),missing_numeric_rows=missing))

# Validate absolute-arm means against the existing independent report audit.
reference=read(REPORT/'v4_4_report_additions/full_span_topk_raw_arms.csv')+read(REPORT/'v4_4_causal_v2/full_span_topk_k6_extension/top6_raw_arms.csv')
for r in ablation:
    if r['metric']!='absolute_count_shift':continue
    ref=next(x for x in reference if x['model_label']==r['model'] and int(x['top_n'])==r['k'] and x['condition']==r['condition'] and x['metric']=='absolute_count_shift')
    assert abs(r['mean']-float(ref['mean']))<1e-12
savecsv('ablation_summary.csv',ablation);savecsv('ablation_seed_effects.csv',seed_rows)

rwroot=REPORT/'v4_4_4/read_write'
rwseed=read(rwroot/'seed_metrics.csv.gz');rwsummary=read(rwroot/'metric_summary.csv.gz')
for row in rwsummary:
    matching=[r for r in rwseed if all(r[key]==row[key] for key in ['metric','stratum','layer'])]
    assert len(matching)==20 and abs(np.mean([float(r['value']) for r in matching])-float(row['mean']))<1e-12

fig,axs=plt.subplots(2,2,figsize=(6.5,4.55))
fig.subplots_adjust(left=.085,right=.985,bottom=.10,top=.93,wspace=.29,hspace=.60)
for j,(model,color) in enumerate(zip(MODELS,COLORS)):
    ax=axs[0,j];style(ax,f'{"AB"[j]}  {DISPLAY_NAMES[model]}: ablation dose')
    for condition,ls,marker in [('ranked','-','o'),('layer_matched_random','--','D')]:
        rows=[r for r in ablation if r['model']==model and r['metric']=='relative_count_shift' and r['condition']==condition and r['k']!=6]
        x=np.array([r['k'] for r in rows]);y=np.array([r['mean'] for r in rows]);low=np.array([r['ci95_low'] for r in rows]);high=np.array([r['ci95_high'] for r in rows])
        ax.plot(x,y,ls=ls,color=color,marker=marker,markersize=3,lw=1.5,label='Ranked' if condition=='ranked' else 'Random')
        ax.fill_between(x,low,high,color=color,alpha=.12 if condition=='ranked' else .06,lw=0)
    if j==1:
        for condition in ['ranked','layer_matched_random']:
            row=next(r for r in ablation if r['model']==model and r['k']==6 and r['metric']=='relative_count_shift' and r['condition']==condition)
            ax.errorbar(6,row['mean'],yerr=[[row['mean']-row['ci95_low']],[row['ci95_high']-row['mean']]],marker='*',ms=7,color=color,mfc='white',mew=.9,lw=.8,capsize=2)
    ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);ax.set(xlim=(.8,38),ylim=(-.015,.76),xlabel='Number of ablated heads',ylabel='Ablation effect')
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.legend(frameon=False,loc='upper left',ncol=2,handlelength=1.7,columnspacing=.8)
    if j==1:ax.text(.035,.66,r'$\star$  $k=6$ extension',transform=ax.transAxes,fontsize=8)

ax=axs[1,0];style(ax,'C  Qwen: reading components')
for i,(name,label) in enumerate([('read_full_behavior_transport','Full state'),('read_routing_behavior_transport','Routing'),('read_value_behavior_transport','Value')]):
    r=next(r for r in rwsummary if r['metric']==name and r['stratum']=='all')
    y,lo,hi=[float(r[key]) for key in ['mean','ci95_low','ci95_high']]
    ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],marker='o',ms=4,color=COLORS[0],capsize=3,lw=1.25)
ax.set(xlim=(-.5,2.5),ylim=(0,.15),ylabel='Source-directed transfer')
ax.set_xticks(range(3),['Full state','Routing','Value']);ax.set_yticks([0,.05,.10,.15]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax=axs[1,1];style(ax,'D  Qwen: downstream OV write')
for metric,color,ls,label in [('write_natural_residual_slope',COLORS[0],'-','Natural'),('write_orthogonal_residual_slope',GRAY,'--','Orthogonal')]:
    rr=sorted([r for r in rwsummary if r['metric']==metric and r['stratum']=='all'],key=lambda r:float(r['layer']))
    x=[int(float(r['layer'])) for r in rr];y=[float(r['mean']) for r in rr]
    ax.plot(x,y,color=color,ls=ls,lw=1.5,label=label)
    ax.fill_between(x,[float(r['ci95_low']) for r in rr],[float(r['ci95_high']) for r in rr],color=color,alpha=.13,lw=0)
ax.set(xlim=(27.7,35.3),ylim=(0,.068),xlabel='Readout layer',ylabel='Count-axis response');ax.set_xticks([28,30,32,35]);ax.legend(frameon=False,loc='upper right',ncol=2,columnspacing=.7,handlelength=1.5)
export(fig,'retrieval_ablation_readwrite')

audit={'status':'PASS','inputs_sha256':INPUTS,'palette':dict(zip(MODELS,COLORS)),'figure_inches':{'retrieval_attention':[6.5,4.35],'retrieval_ablation_readwrite':[6.5,4.55]},
    'fonts_pt':{'axes':8.5,'ticks':8,'legend':8,'titles':10},'bank_summary':bank_audit,
    'ablation':{'rows':len(raw),'seeds':list(range(1316,1336)),'counts':list(range(1,6)), 'bootstrap_draws':10000,'bootstrap_seed':20260910,'unit':'seed after promptwise average of valid random replicates','correct_only_prompts':{'Qwen3-8B':87,'Gemma4-E4B':63},'gemma_top6':'post-hoc dose extension; not part of original 12-test Holm family'},
    'readwrite':{'model':'Qwen3-8B','heads':'L28H16/L28H19','seeds':list(range(1274,1294)),'statistics':'saved seed-bootstrap CIs, all means verified against seed_metrics.csv.gz; reused parent evaluation seeds'},
    'elapsed_seconds':time.perf_counter()-START}
(OUT/'retrieval_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','figures':['retrieval_attention','retrieval_ablation_readwrite'],'elapsed_seconds':audit['elapsed_seconds']}))
