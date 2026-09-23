"""Reproduce protected-control summaries and the diagnostic report."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    args=p.parse_args()
    root=args.input
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['files'].items():
        assert digest(root/name)==h,name
    records=[]
    metrics=['ar_accuracy','ar_answered','eos_reached','answer_and_eos_correct',
             'normalized_count_shift','absolute_count_shift']
    for path in sorted((root/'nonthinking/arms').glob('*.csv')):
        d=pd.read_csv(path)
        assert len(d)==100 and d.key.nunique()==100
        r={c:d[c].iloc[0] for c in ['arm','top_k','repeat','heads','overlap']}
        assert 'L1H2' not in r['heads'].split(';')
        r.update({m:d[m].mean() for m in metrics})
        r.update({m+'_n':int(d[m].notna().sum()) for m in metrics})
        records.append(r)
    d=pd.DataFrame(records)
    assert len(d)==352
    assert d.ar_answered.eq(1).all()
    d.to_csv(root/'condition_summary.csv',index=False)
    rows=[]
    for k,g in d.groupby('top_k'):
        s=g[g.arm=='selected'].iloc[0]
        c=g[g.arm=='control']
        for m in metrics:
            rows.append(dict(top_k=k,metric=m,selected=s[m],control_mean=c[m].mean(),
                             control_min=c[m].min(),control_max=c[m].max(),
                             controls=len(c),overlap=int(c.overlap.iloc[0]),inputs=100))
    curves=pd.DataFrame(rows)
    curves.to_csv(root/'selected_vs_controls.csv',index=False)
    accuracy=curves[curves.metric=='ar_accuracy']
    shift=curves[curves.metric=='normalized_count_shift']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':14,
                         'axes.titlesize':16,'xtick.labelsize':12,'ytick.labelsize':12,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#161923',
                         'text.color':'#161923','axes.labelcolor':'#161923'})
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.15))
    for ax,g,title,label,limits,baseline in [
        (axes[0],accuracy,'A. Count accuracy','Accuracy (%)',(-1,31),21),
        (axes[1],shift,'B. Count displacement','Normalized count shift (%)',(-4,114),0)]:
        ax.fill_between(g.top_k,100*g.control_min,100*g.control_max,color='#8190A5',alpha=.18,linewidth=0)
        ax.plot(g.top_k,100*g.selected,'o-',color='#B52F6B',linewidth=2.2,markersize=5,label='Selected')
        ax.plot(g.top_k,100*g.control_mean,'s--',color='#8190A5',linewidth=2,markersize=4,label='Control mean')
        ax.axhline(baseline,color='#161923',linestyle=':',linewidth=1.6,label='Clean')
        ax.set(title=title,xlabel='Top-k',ylabel=label,ylim=limits,xticks=range(1,9))
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#E3E4EA',linewidth=.8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
    fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.85,wspace=.35)
    for ext in ('pdf','svg','png'):
        fig.savefig(root/f'protected_L1H2_ablation.{ext}',dpi=190,facecolor='white')
    plt.close(fig)
    table='\n'.join(f'| {int(r.top_k)} | {r.selected*100:.2f} | {r.control_mean*100:.2f} | {r.control_min*100:.0f}–{r.control_max*100:.0f} | {int(r.controls)} | {int(r.overlap)} |' for r in accuracy.itertuples())
    shift_table='\n'.join(f'| {int(r.top_k)} | {r.selected*100:.2f} | {r.control_mean*100:.2f} |' for r in shift.itertuples())
    eos=curves[curves.metric=='eos_reached']
    eos_table='\n'.join(f'| {int(r.top_k)} | {r.selected*100:.2f} | {r.control_mean*100:.2f} |' for r in eos.itertuples())
    derived=['condition_summary.csv','selected_vs_controls.csv']+[f'protected_L1H2_ablation.{e}' for e in ['pdf','png','svg']]
    (root/'summary_manifest.json').write_text(json.dumps({'script_sha256':digest(Path(__file__)),
        'raw_manifest_sha256':digest(root/'manifest.json'),
        'files':{f:digest(root/f) for f in derived}},indent=2),encoding='utf-8')
    print('PROTECTED CONTROL SUMMARY COMPLETE')
    print(accuracy.to_string(index=False))


if __name__=='__main__':
    main()
