"""Regenerate appendix H figures from accepted, frozen additional-task results.

No model inference. Uses the Synthetic appendix typography and Aurora blue/orange.
Run with the existing Anaconda NumPy/Matplotlib environment; see manifest.json.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
PAPER = WORK / 'runs/paper_figures/figures/additional_tasks'
RUNS = WORK / 'realistic/additional_experiments/runs'
V3 = RUNS / 'task_local_disjoint_first_20260908_v3/downloaded'
FULL = RUNS / 'native_broad_full_span_20260909_v1/downloaded'
CAT = RUNS / 'category_target_broad_20260909_v1/downloaded'
NAT = RUNS / 'report_refresh_20260908_v1'
MODELS = ['Qwen3-8B', 'Gemma4-E4B']
TASKS = ['kth', 'category']
INK, GRAY, GRID = '#161923', '#8190A5', '#E7E8EE'
COLORS = {'Qwen3-8B': '#168DCA', 'Gemma4-E4B': '#E87824'}
LIGHT = {'Qwen3-8B': '#59AADA', 'Gemma4-E4B': '#ED9B58'}
CMAPS = {
    'Qwen3-8B': LinearSegmentedColormap.from_list('aurora_blue',
        ['#161923', '#244F79', '#198CC9', '#55B6DA', '#A2DBE7', '#E4F4F0']),
    'Gemma4-E4B': LinearSegmentedColormap.from_list('aurora_orange',
        ['#161923', '#663C2D', '#BF6428', '#EF8734', '#F4BD79', '#FFF0D3']),
}
SOURCES, FIGURES, CHECKS = {}, [], []
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 10, 'mathtext.fontset': 'stix',
    'axes.titlesize': 10.5, 'axes.labelsize': 10, 'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5, 'axes.linewidth': .7, 'lines.linewidth': 1.5,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK,
    'axes.labelcolor': INK, 'axes.titlecolor': INK, 'legend.frameon': False,
})

def source(p):
    SOURCES[str(p.relative_to(WORK))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def rows(p):
    with source(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def read(p):
    return json.loads(source(p).read_text(encoding='utf-8'))

def export(name, data):
    with (OUT / 'plot_data' / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, text_boxes = [], []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text(): continue
        # Matplotlib keeps off-axis major/minor tick artists; inspect visible text.
        if artist in [a.xaxis.get_offset_text() for a in fig.axes] and not artist.get_text(): continue
        box = artist.get_window_extent(renderer)
        text_boxes.append((artist.get_text(), box))
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            outside.append(artist.get_text())
    overlaps = []
    for i, (label, box) in enumerate(text_boxes):
        for other, other_box in text_boxes[i+1:]:
            width = min(box.x1, other_box.x1) - max(box.x0, other_box.x0)
            height = min(box.y1, other_box.y1) - max(box.y0, other_box.y0)
            if width > 1 and height > 1:
                overlaps.append(dict(labels=[label, other],
                    bounds=[list(box.bounds),list(other_box.bounds)]))
    CHECKS.append(dict(figure=stem, outside_canvas=outside, text_overlaps=overlaps,
        minimum_font_points=min(a.get_fontsize() for a in fig.findobj(matplotlib.text.Text)
            if a.get_visible() and a.get_text())))
    assert not outside, (stem, outside)
    if overlaps:
        fig.savefig(OUT/f'{stem}_layout_debug.png', dpi=160, facecolor='white')
    assert not overlaps, (stem, overlaps)
    for ext in ['pdf', 'svg', 'png']:
        target = OUT / f'{stem}.{ext}'
        fig.savefig(target, dpi=300, facecolor='white', metadata={'Creator': 'Matplotlib; appendix H'})
    (PAPER / f'{stem}.pdf').write_bytes((OUT / f'{stem}.pdf').read_bytes())
    FIGURES.append(dict(stem=stem, width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(),
        sha256=hashlib.sha256((PAPER / f'{stem}.pdf').read_bytes()).hexdigest()))
    plt.close(fig)

def panel(ax, title, xlabel, ylabel):
    ax.set_title(title, loc='left', pad=7)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, lw=.6)
    ax.tick_params(length=3, pad=2)

def footer(fig, items, ncol=3, show_ci=False):
    handles = [Line2D([], [], **kw) for _, kw in items]
    labels = [name for name, _ in items]
    if show_ci:
        handles.append(Patch(facecolor=GRAY, alpha=.18, edgecolor='none'))
        labels.append('95% CI')
        ncol += 1
    fig.legend(handles, labels,
        loc='lower center', bbox_to_anchor=(.52, .005), ncol=ncol, fontsize=9.5,
        handlelength=2.0, columnspacing=1.2, handletextpad=.5)

def natural_figure():
    data = rows(NAT / 'natural_per_case.csv')
    summary = rows(NAT / 'natural_summary.csv')
    assert len(data) == 2400
    for r in summary:
        subset = [x for x in data if all(x[k] == r[k] for k in ['model', 'task', 'mode'])
                  and (r['split'] == 'all' or x['split'] == r['split'])]
        assert len(subset) == int(r['n']) and sum(int(x['correct']) for x in subset) == int(r['correct'])
        assert abs(np.mean([int(x['correct']) for x in subset]) - float(r['mean'])) < 1e-12
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    fig.subplots_adjust(left=.09, right=.985, bottom=.36, top=.86, wspace=.30)
    points = []
    for i, task in enumerate(TASKS):
        levels = list(range(1, 11)) if task == 'kth' else [1, 3, 5, 7, 9]
        for j, model in enumerate(MODELS):
            ax = axes[i]
            for mode, ls, marker, color in [('nonthinking', '--', 's', LIGHT[model]),
                                            ('native_thinking', '-', 'o', COLORS[model])]:
                values = []
                for level in levels:
                    subset = [r for r in data if (r['task'], r['model'], r['mode'], int(r['level'])) ==
                              (task, model, mode, level)]
                    assert len(subset) == (30 if task == 'kth' else 60)
                    mean = np.mean([int(r['correct']) for r in subset])
                    values.append(100 * mean)
                    points.append(dict(task=task, model=model, mode=mode, level=level, n=len(subset), accuracy=mean))
                ax.plot(levels, values, color=color, ls=ls, marker=marker, ms=3.7)
        panel(ax, f'{chr(65+i)}. '+('Kth record' if task=='kth' else 'Category count'),
            'Requested position $k$' if task == 'kth' else 'Target-category count', 'Answer accuracy (%)')
        ax.set(xticks=levels, ylim=(-3, 103), yticks=[0,25,50,75,100])
    footer(fig, [(model.split('-')[0]+' / '+label, dict(color=color, ls=ls, marker=marker, ms=3.5))
        for model in MODELS for label,color,ls,marker in [
            ('Non-thinking', LIGHT[model], '--', 's'), ('Thinking', COLORS[model], '-', 'o')]], 2)
    export('01_behavior.csv', points); save(fig, '01_behavior')

def score_figures():
    banks, all_scores = {}, []
    for task in TASKS:
        for model in MODELS:
            for assay, mode in [('broad', 'nonthinking'), ('targeted', 'native_thinking')]:
                p = V3 / f'banks/{task}_{model}_{mode}_{assay}.json'
                b = read(p); ranking = b['ranking']
                banks[task, model, mode, assay] = ranking
                assert len(ranking) == (1152 if model == MODELS[0] else 336)
                assert len({(l, h) for l, h, s in ranking}) == len(ranking)
                assert all(0 <= s <= 1 + 1e-10 for l, h, s in ranking)
                for l, h, s in ranking:
                    all_scores.append(dict(task=task, model=model, mode=mode, assay=assay, layer=l, head=h, score=s))
    specs = [('broad', 'nonthinking', 'Broad', r'$B_h$'),
             ('targeted', 'native_thinking', 'Targeted', r'$T_h$')]
    scales = {}
    fig = plt.figure(figsize=(6.5, 4.4))
    for t, task in enumerate(TASKS):
        fig.text(.283 + .462*t, .978,
                 'Kth record' if task == 'kth' else 'Category count',
                 ha='center', va='top', fontsize=11, fontweight='bold')
    for i, model in enumerate(MODELS):
        fig.text(.075, .920 if i == 0 else .473, model,
                 color=COLORS[model], fontsize=10.5, va='center')
        for t, task in enumerate(TASKS):
            for a, (assay, mode, title, label) in enumerate(specs):
                j = 2*t+a
                x, y = .075+j*.231, .620 if i==0 else .175
                ax = fig.add_axes([x, y, .195, .235])
                ranking = banks[task,model,mode,assay]
                vmax = float(np.ceil(max(s for _,_,s in ranking) / .05)*.05)
                scales[f'{task}/{model}/{assay}'] = dict(vmin=0, vmax=vmax,
                    normalization='linear', clipping=False)
                layers, heads = (36,32) if model == MODELS[0] else (42,8)
                matrix = np.full((layers,heads), np.nan)
                for l,h,s in ranking: matrix[l,h] = s
                assert np.isfinite(matrix).all()
                im = ax.imshow(matrix, aspect='auto', interpolation='nearest', origin='upper',
                    cmap=CMAPS[model], vmin=0, vmax=vmax)
                ax.set_title(f'{chr(65+i*4+j)}. {title} {label}', loc='left', fontsize=9.5, pad=5)
                ax.set(xlabel='Head' if i==1 else '', ylabel='Layer' if j == 0 else '',
                       xticks=[0,15,31] if heads==32 else [0,3,7],
                       yticks=[0,12,24,35] if layers==36 else [0,14,28,41])
                ax.set_yticklabels([1,13,25,36] if layers==36 else [1,15,29,42])
                ax.set_xticklabels([1,16,32] if heads==32 else [1,4,8])
                ax.tick_params(labelsize=9, length=2, pad=2)
                cax = fig.add_axes([x, y-(.090 if i==0 else .108), .195, .020])
                cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0,vmax])
                cb.ax.tick_params(labelsize=9, length=2, pad=2)
                cb.ax.set_xticklabels(['0', f'{vmax:.2f}'.rstrip('0').rstrip('.')])
                cb.outline.set_linewidth(.5)
    save(fig, '02_head_scores')
    export('02_head_scores.csv', all_scores)
    return scales

SUMMARY = rows(V3 / 'analysis/summary.csv')

def ablation_figure():
    fig, axes = plt.subplots(4, 2, figsize=(6.5, 5.6))
    fig.subplots_adjust(left=.10, right=.985, bottom=.15, top=.915, wspace=.30, hspace=.65)
    for j, model in enumerate(MODELS):
        box = axes[0,j].get_position()
        fig.text((box.x0+box.x1)/2, .985, model, ha='center', va='top',
                 color=COLORS[model], fontsize=11, fontweight='bold')
    for start, label in [(0, 'Next-entity accuracy (%)'), (2, 'Final-answer accuracy (%)')]:
        y = (axes[start,0].get_position().y1 + axes[start+1,0].get_position().y0)/2
        fig.text(.025, y, label, rotation=90, ha='center', va='center', fontsize=10)
    table = []
    for i, (assay, task) in enumerate([('targeted','kth'), ('targeted','category'),
                                     ('broad','kth'), ('broad','category')]):
        mode = 'native_thinking' if assay == 'targeted' else 'nonthinking'
        for j, model in enumerate(MODELS):
            ax = axes[i,j]
            subset = [r for r in SUMMARY if (r['task'],r['model'],r['mode'],r['assay']) ==
                      (task,model,mode,assay)]
            for metric, color, ls, marker in [('selected',COLORS[model],'-','o'),
                                             ('random',LIGHT[model],'--','s')]:
                rs = sorted([r for r in subset if r['metric']==metric], key=lambda r:int(r['k']))
                ks = [int(r['k']) for r in rs]
                assert len(ks) == len(set(ks))
                y,lo,hi = [100*np.array([float(r[key]) for r in rs]) for key in ['mean','lower','upper']]
                assert np.all(0 <= lo) and np.all(lo <= y) and np.all(y <= hi) and np.all(hi <= 100)
                if assay == 'broad': assert np.max(hi) <= 40
                ax.fill_between(ks,lo,hi,color=color,alpha=.14 if metric=='selected' else .11,lw=0)
                ax.plot(ks,y,color=color,ls=ls,marker=marker,ms=3.7,
                        markerfacecolor=color if metric=='selected' else 'white',markeredgewidth=1)
                table.extend(rs)
            task_label = 'kth record' if task=='kth' else 'category count'
            panel(ax,f'{chr(65+i*2+j)}. {assay.capitalize()}: {task_label}',
                  r'Ablated heads $K$' if i==3 else '', '')
            ax.set_title(ax.get_title(loc='left'),loc='left',fontsize=10)
            if assay=='broad' and model==MODELS[0]: ax.set_xscale('log',base=2)
            ax.set_xticks(ks, [str(k) for k in ks])
            ax.set(ylim=(-3,103) if assay=='targeted' else (-1.2,41.2),
                   yticks=[0,50,100] if assay=='targeted' else [0,20,40])
    footer(fig,[('Selected ablation',dict(color=INK,ls='-',marker='o',ms=3.7)),
                ('Random control',dict(color=GRAY,ls='--',marker='s',ms=3.7,markerfacecolor='white'))],2,show_ci=True)
    export('03_ablation.csv',table); save(fig,'03_ablation')

def category_figure():
    cat = rows(CAT / 'analysis/summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.3))
    fig.subplots_adjust(left=.09,right=.985,bottom=.30,top=.86,wspace=.30)
    data=[]
    for j,model in enumerate(MODELS):
        ax=axes[j]
        for metric,color,ls,marker in [('selected',COLORS[model],'-','o'),('random',LIGHT[model],'--','s')]:
            rs=[r for r in cat if r['model']==model and r['metric']==metric and r['population']=='all_examples']
            rs=sorted(rs,key=lambda r:int(r['k']))
            ks=[int(r['k']) for r in rs]
            y,lo,hi=[100*np.array([float(r[key]) for r in rs]) for key in ['mean','lower','upper']]
            assert np.all(0 <= lo) and np.all(lo <= y) and np.all(y <= hi) and np.max(hi) <= 40
            ax.fill_between(ks,lo,hi,color=color,alpha=.14 if metric=='selected' else .11,lw=0)
            ax.plot(ks,y,color=color,ls=ls,marker=marker,ms=3.7,
                    markerfacecolor=color if metric=='selected' else 'white',markeredgewidth=1)
            data.extend(rs)
        panel(ax,f'{chr(65+j)}. {model}',r'Ablated heads $K$','Final-answer accuracy (%)')
        ax.set_title(ax.get_title(loc='left'),loc='left',color=COLORS[model])
        if j==0: ax.set_xscale('log',base=2)
        ax.set_xticks(ks,[str(k) for k in ks]);ax.set(ylim=(-1.2,41.2),yticks=[0,20,40])
    footer(fig,[('Selected ablation',dict(color=INK,ls='-',marker='o',ms=3.7)),
                ('Random control',dict(color=GRAY,ls='--',marker='s',ms=3.7,markerfacecolor='white'))],2,show_ci=True)
    export('04_category_scope.csv',data);save(fig,'04_category_scope')

def main():
    (OUT/'plot_data').mkdir(exist_ok=True);PAPER.mkdir(exist_ok=True)
    natural_figure();scales=score_figures()
    ablation_figure()
    category_figure()
    manifest=dict(source_hashes=SOURCES,figures=FIGURES,checks=CHECKS,score_scales=scales,
        style=dict(reference='figures/synthetic_appendix_paper_checked_20260908',font='Times New Roman',
            qwen=COLORS[MODELS[0]],gemma=COLORS[MODELS[1]],control_shading='pointwise 95% CI',
            interval_method='percentile bootstrap clustered by data-generation seed'),
        versions=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
        elapsed_seconds=time.perf_counter()-START)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(figures=len(FIGURES),scales=scales,checks=CHECKS,elapsed_seconds=manifest['elapsed_seconds']),indent=2))

if __name__=='__main__': main()
