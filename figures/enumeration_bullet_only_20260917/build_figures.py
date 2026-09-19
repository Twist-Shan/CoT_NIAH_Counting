"""Render Bullet-only Enumeration figures from frozen, audited results.

No inference, selection, refitting, or outcome filtering. The already verified
Bullet PCA figure is copied without changing its coordinates or rendering.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / 'realistic'
PAPER = ROOT / 'runs/paper_figures/figures/enumeration_bullet'
PRIOR = ROOT / 'figures/enumeration_midlayer_20260917'
ARCHIVE = REPO / 'outputs/enumeration_replay_midlayer_20260917/pca3_randomized_v2'
MODE = 'enumeration_bullet'
MODELS = [('Qwen3-8B', 'Qwen', '#168DCA', 36), ('Gemma4-E4B', 'Gemma', '#E87824', 42)]
SOURCES, FIGURES = {}, {}


def read(path):
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')


def read_json(path):
    return json.loads(read(path))


def module(name, path):
    read(path)
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def cell(report, model):
    selected = [c for c in report['cells'] if c['model'] == model and c['mode'] == MODE]
    assert len(selected) == 1
    return selected[0]


def two_panels(height=2.65, bottom=.29, top=.76):
    fig = plt.figure(figsize=(6.5, height))
    return fig, [fig.add_axes([x, bottom, .365, top-bottom]) for x in [.10, .61]]


def save(fig, name, rows, extra=None):
    assert rows and all(r.get('mode', r.get('format')) == MODE for r in rows)
    style.save(fig, name, rows, extra)
    FIGURES[name] = style.FIGURES[name]
    (OUT / 'data' / (name + '.json')).write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(OUT / (name + '.pdf'), PAPER / (name + '.pdf'))


def legend(fig, labels, styles):
    fig.legend([Line2D([], [], color='.25', ls=s) for s in styles], labels,
               loc='center', bbox_to_anchor=(.5, .04), ncol=len(labels))


def head_scores(report):
    fig, axes = two_panels(2.45, .20, .74)
    archive = REPO / 'work/et2/src'
    inventory = read_json(archive / 'backup_manifest.json')
    assert SOURCES[(archive / 'backup_manifest.json').relative_to(ROOT).as_posix()] == report['source_inventory_sha256']
    rows = []
    for j, (model, short, _, _) in enumerate(MODELS):
        rel = f'fresh_retrieve_gpu_v2/jobs/localize/{model}/{MODE}/ranking.json'
        path = archive / rel
        ranking = read_json(path)
        assert SOURCES[path.relative_to(ROOT).as_posix()] == inventory['files'][rel]['sha256']
        layers = sorted({r['layer'] for r in ranking}); heads = sorted({r['head'] for r in ranking})
        assert len(ranking) == len(layers)*len(heads) == (1152 if j == 0 else 56)
        matrix = np.full((len(heads), len(layers)), np.nan)
        for r in ranking: matrix[heads.index(r['head']), layers.index(r['layer'])] = r['score']
        assert np.isfinite(matrix).all() and 0 <= matrix.min() <= matrix.max() <= 1
        ax = axes[j]
        im = ax.imshow(matrix, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
        for r in ranking[:128 if j == 0 else 6]:
            ax.add_patch(Rectangle((layers.index(r['layer'])-.5, heads.index(r['head'])-.5), 1, 1,
                                  fill=False, edgecolor='black', linewidth=.35 if j == 0 else .8))
        ix = [0, 11, 23, 35] if j == 0 else list(range(7)); iy = [0, 7, 15, 23, 31] if j == 0 else [0, 3, 7]
        ax.set(xticks=ix, xticklabels=[layers[x]+1 for x in ix], yticks=iy, yticklabels=[heads[y]+1 for y in iy])
        style.panel(ax, f'{"AB"[j]}. {short}', 'Head', 'Layer'); ax.grid(False)
        rows.extend(dict(model=model, mode=MODE, **r) for r in ranking)
    cax = fig.add_axes([.54, .93, .36, .023])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    cb.ax.tick_params(labelsize=9, pad=2, length=2)
    fig.text(.10, .944, 'Target-record attention', fontsize=9, va='center')
    save(fig, 'enumeration_head_scores', rows)


def readouts():
    fig, axes = two_panels()
    selection = read_json(ARCHIVE / 'selection.json')
    rows, chosen_rows = [], []
    for i, endpoint in enumerate(['running_index', 'final_count']):
        path = ARCHIVE / f'{endpoint}_candidate_metrics.csv'
        candidates = list(csv.DictReader(read(path).splitlines()))
        assert SOURCES[path.relative_to(ROOT).as_posix()] == selection['source_sha256'][path.name]
        ax = axes[i]
        style.panel(ax, ['A. Running index', 'B. Final count'][i], 'Balanced accuracy', 'Layer')
        for model, _, color, depth in MODELS:
            rr = sorted([r for r in candidates if r['model_label'] == model and r['prompt_mode'] == MODE], key=lambda r: int(r['layer']))
            assert [int(r['layer']) for r in rr] == list(range(depth))
            chosen, = [r for r in selection['selected'][endpoint] if r['model_label'] == model and r['prompt_mode'] == MODE]
            expected = max(rr, key=lambda r: (round(float(r['discovery_oof_ncc_balanced_accuracy']), 12), round(float(r['discovery_oof_logistic_balanced_accuracy']), 12), -int(r['layer'])))
            assert int(chosen['layer']) == int(expected['layer'])
            for method, ls in [('ncc', '-'), ('logistic', '--')]:
                values = [float(r[f'discovery_oof_{method}_balanced_accuracy']) for r in rr]
                ax.plot([int(r['layer'])+1 for r in rr], values, color=color, ls=ls)
                rows.extend(dict(format=MODE, model=model, endpoint=endpoint, split='discovery_oof', method=method,
                                 layer_display_one_based=int(r['layer'])+1, balanced_accuracy=v, states=int(r['discovery_oof_rows'])) for r, v in zip(rr, values))
            layer, value = int(chosen['layer'])+1, float(chosen['discovery_oof_ncc_balanced_accuracy'])
            ax.scatter(layer, value, s=27, color=color, edgecolor='white', linewidth=.6, zorder=4)
            ax.annotate(f'L{layer}', (layer, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9, color=color)
            chosen_rows.append(dict(chosen, layer_display_one_based=layer))
        ax.set(xlim=(.5, 42.5), xticks=[1, 10, 20, 30, 42], ylim=(0, 1.15), yticks=[0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.axhline(.1, color=style.GRAY, ls=':', lw=.9)
    fig.legend([Line2D([], [], color=m[2]) for m in MODELS] + [Line2D([], [], color=style.INK, ls=s) for s in ['-', '--']],
               ['Qwen', 'Gemma', 'Nearest centroid', 'Logistic'], loc='upper center', bbox_to_anchor=(.535, .995), ncol=4,
               handlelength=1.7, handletextpad=.45, columnspacing=1.4)
    assert len(rows) == 312
    save(fig, 'enumeration_representations', rows, dict(selected_layers=chosen_rows, displayed_split='discovery_oof', confirmation_curves_plotted=False))


def pca():
    name = 'enumeration_pca_bullet'
    prior = read_json(PRIOR / 'representation_manifest.json')['figures'][name]
    coordinates = list(csv.DictReader(read(ARCHIVE / 'data/enumeration_pca_coordinates.csv').splitlines()))
    rows = [dict(r, display_condition='enumeration') for r in coordinates if r['format'] == MODE]
    assert len(rows) == prior['source_rows'] == 1278
    assert sorted(len([r for r in rows if r['model']==m and r['endpoint']==e]) for m, _, _, _ in MODELS for e in ['running', 'final']) == [100,100,528,550]
    for ext in ['pdf', 'svg', 'png']:
        path = PRIOR / f'{name}.{ext}'
        raw = path.read_bytes(); h = hashlib.sha256(raw).hexdigest()
        assert h == prior['artifacts_sha256'][path.name]
        SOURCES[path.relative_to(ROOT).as_posix()] = h
        shutil.copy2(path, OUT / path.name)
    shutil.copy2(OUT / f'{name}.pdf', PAPER / f'{name}.pdf')
    (OUT / 'data' / f'{name}.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    FIGURES[name] = dict(prior, coordinates_refit=False, rendering='Unchanged verified Bullet PCA; discovery-only randomized SVD seed 0, native signs.')
    print(name, 'PASS (unchanged verified rendering)', flush=True)


def retrieve(report):
    fig, axes = two_panels()
    rows=[]
    for i, metric in enumerate(['next_city_failure', 'final_exact_count_failure']):
        ax=axes[i]
        for model, _, color, _ in MODELS:
            doses=cell(report, model)['dose_response']
            for key, ls in [('selected_minus_clean','-'),('random_minus_clean','--')]:
                stats=[d['metrics'][metric][key] for d in doses]
                causal.curve(ax,[d['dose_k'] for d in doses],stats,color,linestyle=ls,marker='o' if ls=='-' else None)
                rows.extend(dict(model=model,mode=MODE,metric=metric,condition=key,k=d['dose_k'],estimate=causal.value(s),ci95=s['ci95'],random_control=d['random_control']) for d,s in zip(doses,stats))
        ax.set_xscale('log',base=2); ax.set(xticks=[1,2,4,8,32,128],xlim=(.85,153),ylim=(-.05,1.08),yticks=[0,.5,1])
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        style.panel(ax,['A. Next city','B. Final count'][i],'Failure increase','Ablated heads')
    causal.models_legend(fig);legend(fig,['Selected','Random'],['-','--'])
    assert sum(cell(report,m)['coverage']['formal']['truncated'] for m,_,_,_ in MODELS)==3
    save(fig,'enumeration_retrieve',rows)


def update(report):
    fig,axes=two_panels();rows=[]
    for mi,(model,_,color,_) in enumerate(MODELS):
        c=cell(report,model);groups=c['groups'];assert c['layer_one_based']==[19,21][mi]
        for di,direction in enumerate(['forward','backward']):
            marker,ls=[('o','-'),('D','--')][di]
            for j,scope in enumerate(['endpoint','four_token_tail','item_span']):
                g=next(g for g in groups if g['scope']==scope and g['direction']==direction and g['donor_k']=='all')
                stat=g['paired_target_minus_self_adoption'];causal.point(axes[0],j+[-.18,-.06,.06,.18][2*mi+di],stat,color,marker)
                rows.append(dict(model=model,mode=MODE,layer_one_based=c['layer_one_based'],direction=direction,scope=scope,metric='paired_adoption',estimate=causal.value(stat),ci95=stat['ci95']))
            g=next(g for g in groups if g['scope']=='item_span' and g['direction']==direction and g['donor_k']=='all')
            hops=[g['conditions']['donor_to_receiver']['continuation'][str(h)] for h in range(1,5)]
            assert all(h['total_trials']==30 for h in hops)
            causal.curve(axes[1],list(range(1,5)),[h['conditional'] for h in hops],color,linestyle=ls,marker=marker)
            rows.extend(dict(model=model,mode=MODE,layer_one_based=c['layer_one_based'],direction=direction,scope='item_span',metric='conditional_next_step',hop=h['hop'],numerator=h['successes'],denominator=h['conditional_eligible'],horizon_eligible=h['horizon_eligible'],**h['conditional']) for h in hops)
    axes[0].set(xticks=[0,1,2],xticklabels=['Endpoint','4 tokens','Item span'],xlim=(-.45,2.45),ylim=(-.10,1.08),yticks=[0,.5,1]);axes[0].axhline(0,color='.55',lw=.6)
    axes[1].set(xticks=[1,2,3,4],xlim=(.85,4.15),ylim=(-.05,1.08),yticks=[0,.5,1])
    for ax in axes:ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    style.panel(axes[0],'A. Patch scope','Target minus self')
    style.panel(axes[1],'B. Continued prefix','Conditional success','Continuation step')
    causal.models_legend(fig)
    fig.legend([Line2D([],[],color='.25',ls=ls,marker=m,ms=4) for ls,m in [('-','o'),('--','D')]],['Forward','Backward'],loc='center',bbox_to_anchor=(.5,.04),ncol=2)
    save(fig,'enumeration_update',rows)


def readout(answer,blanking):
    fig,axes=two_panels();rows=[]
    for model,_,color,_ in MODELS:
        layers=cell(answer,model)['layerwise']
        for condition,ls in [('full_donor_patch','-'),('self_patch','--')]:
            stats=[l['conditions'][condition]['donor_adoption'] for l in layers]
            causal.curve(axes[0],[l['layer_one_based'] for l in layers],stats,color,linestyle=ls)
            rows.extend(dict(model=model,mode=MODE,layer=l['layer_one_based'],condition=condition,metric='answer_donor_adoption',estimate=causal.value(s),ci95=s['ci95']) for l,s in zip(layers,stats))
    for mi,(model,_,color,_) in enumerate(MODELS):
        for ci,condition in enumerate(['clean','prompt_records_blank','trace_all_blank']):
            stat=cell(blanking,model)['conditions'][condition]['accuracy'];causal.point(axes[1],ci+(mi-.5)*.17,stat,color)
            rows.append(dict(model=model,mode=MODE,condition=condition,metric='blanking_accuracy',estimate=causal.value(stat),ci95=stat['ci95'],numerator=stat['numerator'],denominator=stat['denominator']))
    axes[0].set(xlim=(1,42),xticks=[1,10,20,30,40],ylim=(-.05,1.08),yticks=[0,.5,1])
    axes[1].set(xticks=[0,1,2],xticklabels=['Clean','Record blank','Trace blank'],xlim=(-.4,2.4),ylim=(-.05,1.08),yticks=[0,.5,1])
    for ax in axes:ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    style.panel(axes[0],'A. Answer state','Target-count adoption','Layer')
    style.panel(axes[1],'B. Trace blanking','Exact-count accuracy')
    causal.models_legend(fig);legend(fig,['Target state','Self patch'],['-','--'])
    save(fig,'enumeration_readout',rows)


if __name__=='__main__':
    start=time.monotonic();PAPER.mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    style=module('thinking_appendix_style',ROOT/'figures/thinking_appendix_style_20260913/build_figures.py');style.OUT=OUT;style.setup()
    causal=module('enumeration_verified_helpers',PRIOR/'build_causal.py')
    base=REPO/'outputs/enumeration_alignment_20260916'
    reports={name:read_json(base/directory/'audit.json') for name,directory in [('retrieve','retrieve_audit_1550'),('read','read_audit_v1')]}
    reports['answer']=read_json(REPO/'outputs/enumeration_followup_20260917/audit_v1/answer_audit.json')
    reports['update']=read_json(REPO/'outputs/enumeration_replay_midlayer_20260917/midlayer_audit/combined_update_audit.json')
    assert all(r['status']=='PASS' for r in reports.values())
    head_scores(reports['retrieve']);readouts();pca();retrieve(reports['retrieve']);update(reports['update']);readout(reports['answer'],reports['read'])
    read(Path(__file__))
    manifest=dict(status='PASS',scope=MODE,figures=FIGURES,source_sha256=SOURCES,inference_or_refitting=False,
                  selection_or_cohort_changes=False,font_profile=dict(family='Times New Roman',titles_pt=11,axes_pt=10,ticks_legend_pt=9,width_inches=6.5),seconds=time.monotonic()-start)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('PASS: six Bullet-only figures; all original Bullet observations retained.',flush=True)
