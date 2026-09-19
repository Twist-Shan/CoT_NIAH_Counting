"""Plot complete supplementary grids at the manuscript's appendix font sizes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, FixedLocator, FixedFormatter

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
LR=Path('\\\\?\\'+str(ROOT)) if os.name=='nt' else ROOT
DATA=LR/'realistic/work/cot_completion_20260912'
MODELS=['Qwen3-8B','Gemma4-E4B']
COLORS=['#168DCA','#E87824']
SOURCES={}
QA={}


def read(path):
    SOURCES[str(path).removeprefix('\\\\?\\')]=hashlib.sha256(path.read_bytes()).hexdigest()
    text=path.read_text()
    return [json.loads(text)] if path.suffix=='.json' else [json.loads(s) for s in text.split('\n') if s.strip()]


def ci(values):
    values=np.asarray(values,float)
    assert values.shape==(10,) and np.isfinite(values).all()
    rng=np.random.default_rng(20260912)
    boot=values[rng.integers(0,10,(10000,10))].mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return float(values.mean()),float(lo),float(hi)


def outcome(row,name):
    if name=='next_item':return bool(row['correct_next_needle'])
    values=re.findall(r'(?i)(?:^|\b)total\s*:\s*([0-9]+)\b',row['completion_text'])
    return bool(values and int(values[-1])==int(row['gold_count']))


def load(model,task):
    folder=DATA/'generation/runs'/model/f'{task}_full'
    assert read(folder/'process_status.json')[0]['status']=='COMPLETE'
    rows=[r for p in sorted((folder/'results').glob('*.jsonl')) for r in read(p)]
    expected=80 if task=='recovery' else 330 if model=='Qwen3-8B' else 170
    assert len(rows)==expected
    return rows


def setup():
    for font in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts')/font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.titlesize':11,
        'axes.titleweight':'normal','axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'legend.fontsize':9,'axes.linewidth':.6,'axes.edgecolor':'#788495','legend.frameon':False})


def axis(ax,title,ylabel):
    ax.set_title(title,loc='left',pad=9)
    ax.set_ylabel(ylabel)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E5E8EC',lw=.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5,width=.6)
    ax.yaxis.set_major_formatter(PercentFormatter(1))


def save(fig,name,rows):
    assert fig.get_figwidth()==6.5
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    boxes=[(t.get_text(),t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    outside=[s for s,b in boxes if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5]
    overlaps=[]
    for i,(s,b) in enumerate(boxes):
        for t,c in boxes[i+1:]:
            if min(b.x1,c.x1)-max(b.x0,c.x0)>1.5 and min(b.y1,c.y1)-max(b.y0,c.y0)>1.5:
                overlaps.append([s,t])
    assert not outside,(name,outside)
    assert not overlaps,(name,overlaps)
    for ext in ['pdf','png','svg']:
        kwargs={'dpi':200} if ext=='png' else {'metadata':{'CreationDate':None}} if ext=='pdf' else {}
        fig.savefig(OUT/f'{name}.{ext}',**kwargs)
    with (OUT/f'{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    QA[name]={'outside_text':outside,'text_overlaps':overlaps,'width_inches':6.5,'height_inches':fig.get_figheight(),
              'title_pt':11,'axis_pt':10,'tick_legend_pt':9}
    plt.close(fig)


def dose():
    fig,axs=plt.subplots(2,2,figsize=(6.5,4.95))
    fig.subplots_adjust(left=.12,right=.985,bottom=.10,top=.87,wspace=.42,hspace=.65)
    plots=[]
    for i,(model,color) in enumerate(zip(MODELS,COLORS)):
        rows=load(model,'dose')
        ks=sorted({r['dose_k'] for r in rows if r['dose_k']})
        for j,measure in enumerate(['next_item','final_count']):
            ax=axs[i,j]
            label='Next item' if measure=='next_item' else 'Final count'
            axis(ax,f'{"ABCD"[2*i+j]}. {["Qwen","Gemma"][i]}: {label.lower()}','Ablation effect')
            for condition,style,marker in [('selected_bank','-','o'),('layer_matched_random','--','D')]:
                curve=[]
                for k in ks:
                    values=[]
                    for seed in range(1254,1264):
                        clean=next(r for r in rows if r['seed']==seed and r['dose_k']==0)
                        cells=[r for r in rows if r['seed']==seed and r['dose_k']==k and r['condition']==condition]
                        assert len(cells)==(1 if condition=='selected_bank' else 3)
                        values.append(float(outcome(clean,measure))-np.mean([outcome(r,measure) for r in cells]))
                    m,lo,hi=ci(values);curve.append((m,lo,hi))
                    plots.append(dict(model=model,outcome=measure,k=k,condition=condition,mean=m,ci95_low=lo,ci95_high=hi,seeds=10))
                arr=np.array(curve)
                ax.plot(ks,arr[:,0],color=color,ls=style,marker=marker,ms=3.5,lw=1.2)
                ax.fill_between(ks,arr[:,1],arr[:,2],color=color,alpha=.07 if condition=='layer_matched_random' else .14,lw=0)
            ax.axhline(0,color='#777777',lw=.6)
            ax.set_xscale('log',base=2)
            ax.xaxis.set_major_locator(FixedLocator(ks));ax.xaxis.set_major_formatter(FixedFormatter([str(k) for k in ks]));ax.minorticks_off()
            ax.set_xlabel('Number of ablated heads, $K$')
            ax.set_xlim(ks[0]/1.12,ks[-1]*1.12)
    ymin=min(-.04,min(r['ci95_low'] for r in plots)-.04)
    for ax in axs.flat:ax.set_ylim(ymin,1.04);ax.set_yticks([0,.5,1])
    fig.legend([Line2D([],[],color='#555555',marker='o',lw=1.2),Line2D([],[],color='#555555',marker='D',ls='--',lw=1.2)],
               ['Frozen Top-$K$ prefix','Layer-matched random'],loc='upper center',bbox_to_anchor=(.55,.995),ncol=2)
    save(fig,'cot_current_bank_dose',plots)


def recovery():
    model_rows={model:load(model,'recovery') for model in MODELS}
    for measure in ['final_count','next_item']:
        fig,axs=plt.subplots(2,2,figsize=(6.5,5.05))
        fig.subplots_adjust(left=.12,right=.985,bottom=.11,top=.925,wspace=.38,hspace=.62)
        plots=[]
        for i,(model,color) in enumerate(zip(MODELS,COLORS)):
            rows=model_rows[model]
            for j,scope in enumerate(['local','persistent']):
                ax=axs[i,j]
                axis(ax,f'{"ABCD"[2*i+j]}. {["Qwen","Gemma"][i]}: {scope} lesion',
                     'Exact-count accuracy' if measure=='final_count' else 'Next-item accuracy')
                arms=['clean','clean_restore',scope+'_lesion',scope+'_restore',scope+'_matched']
                for x,arm in enumerate(arms):
                    cells=[r for r in rows if r['condition']==arm]
                    assert len(cells)==10 and {r['seed'] for r in cells}==set(range(1254,1264))
                    m,lo,hi=ci([outcome(r,measure) for r in sorted(cells,key=lambda r:r['seed'])])
                    ax.errorbar(x,m,yerr=[[m-lo],[hi-m]],color=color,marker='o',mfc='white' if x in [1,4] else color,
                                ms=4.3,capsize=2.5,lw=1)
                    ax.text(x,hi+.055,f'{round(m*10)}/10',ha='center',fontsize=9,color=color)
                    plots.append(dict(model=model,scope=scope,outcome=measure,condition=arm,mean=m,ci95_low=lo,ci95_high=hi,seeds=10))
                ax.set_xticks(range(5),['Clean','Self\nrestore','Lesion','Carrier\nrestore','Matched\nrestore'])
                ax.set_xlim(-.45,4.45);ax.set_ylim(-.045,1.22);ax.set_yticks([0,.5,1])
        save(fig,'cot_free_'+measure+'_recovery',plots)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['dose','recovery'],required=True)
    args=p.parse_args()
    setup()
    dose() if args.stage=='dose' else recovery()
    manifest={'source_sha256':SOURCES,'layout_audit':QA,'bootstrap_seed':20260912,'bootstrap_draws':10000,
              'font_family':'Times New Roman / STIX','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/f'{args.stage}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
