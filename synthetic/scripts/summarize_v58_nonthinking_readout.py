"""Summarize frozen NT diagnostic without filtering intervention failures."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--prior',required=True,type=Path)
    parser.add_argument('--sites',required=True,type=Path)
    args=parser.parse_args()
    root=args.input
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete'
    for name,expected in manifest['files'].items():
        assert digest(root/name)==expected,name
    metrics=['full_vocab_accuracy','first_is_number','p_numeric_1to10','p_ans',
             'count_restricted_accuracy','count_restricted_abs_error',
             'count_restricted_nll','count_restricted_margin','numeric_vs_ans_margin']
    records=[]
    for p in sorted((root/'readouts').glob('*.csv')):
        if p.stem=='clean':
            continue
        d=pd.read_csv(p)
        assert len(d)==100 and d.key.nunique()==100
        record={c:d[c].iloc[0] for c in ['intervention','arm','top_k','repeat','heads','overlap']}
        record['has_L1H2']='L1H2' in record['heads'].split(';')
        record.update({m:d[m].mean() for m in metrics})
        records.append(record)
    conditions=pd.DataFrame(records)
    assert len(conditions)==1040
    conditions.to_csv(root/'condition_summary.csv',index=False)
    clean=pd.read_csv(root/'readouts/clean.csv')
    compare=[]
    for (kind,k),d in conditions[conditions.arm.isin(['selected','control'])].groupby(['intervention','top_k']):
        selected=d[d.arm=='selected'].iloc[0]
        controls=d[d.arm=='control']
        for metric in metrics:
            compare.append(dict(intervention=kind,top_k=k,metric=metric,inputs=100,controls=len(controls),
                                selected=selected[metric],control_mean=controls[metric].mean(),
                                control_min=controls[metric].min(),control_max=controls[metric].max(),
                                clean=clean[metric].mean()))
    curves=pd.DataFrame(compare)
    curves.to_csv(root/'selected_vs_controls.csv',index=False)
    table=[]
    for k in range(1,9):
        row={'K':k}
        for kind,metric,tag in [('zero','first_is_number','zero_valid'),
                                ('zero','count_restricted_accuracy','zero_numeric'),
                                ('mean','full_vocab_accuracy','mean_full'),
                                ('mean','first_is_number','mean_valid'),
                                ('mean','count_restricted_accuracy','mean_numeric')]:
            s=curves[(curves.intervention==kind)&(curves.top_k==k)&(curves.metric==metric)].iloc[0]
            row[tag+'_selected']=100*s.selected
            row[tag+'_controls']=100*s.control_mean
        table.append(row)
    pd.DataFrame(table).to_csv(root/'compact_table.csv',index=False)
    strata=conditions[conditions.arm=='control'].groupby(['intervention','top_k','has_L1H2']).agg(
        conditions=('heads','size'),valid=('first_is_number','mean'),
        full_accuracy=('full_vocab_accuracy','mean'),numeric_accuracy=('count_restricted_accuracy','mean'),
        numeric_mass=('p_numeric_1to10','mean'))
    strata.to_csv(root/'control_H2_strata.csv')
    ranking=json.loads(args.sites.read_text(encoding='utf-8'))['ranking']['broad']
    scores={f'L{int(l)}H{int(h)}':(rank+1,score) for rank,(l,h,score) in enumerate(ranking)}
    singles=conditions[conditions.top_k.eq(1)].copy()
    singles['broad_rank']=singles['heads'].map(lambda h:scores[h][0])
    singles['broad_score']=singles['heads'].map(lambda h:scores[h][1])
    singles.sort_values(['intervention','broad_rank']).to_csv(root/'all_single_heads.csv',index=False)
    scale=pd.read_csv(root/'L1H2_scaling.csv').groupby('scale')[metrics].mean()
    scale.to_csv(root/'L1H2_scaling_summary.csv')
    archived=pd.read_csv(args.prior/'all_condition_summary.csv')
    archived=archived[(archived['mode']=='nonthinking')&(archived.scope=='sustained')&(archived.support=='primary')]
    bank={'L1H1','L1H2','L1H5','L1H7'}
    release=archived[archived.top_k.eq(3)&archived['heads'].fillna('').map(lambda x:set(x.split(';'))<bank)].copy()
    assert len(release)==4
    release['released_head']=release['heads'].map(lambda h:next(iter(bank-set(h.split(';')))))
    release[['released_head','heads','ar_accuracy','ar_answered','eos_reached']].to_csv(root/'control_head_release_from_existing_sweep.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':14,
                         'axes.titlesize':16,'xtick.labelsize':12,'ytick.labelsize':12,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#161923',
                         'text.color':'#161923','axes.labelcolor':'#161923'})
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.15))
    specs=[('zero','first_is_number','A. Output validity','Numeric first token (%)',(-4,104)),
           ('zero','count_restricted_accuracy','B. Count discrimination','Accuracy within 1–10 (%)',(-1,31)),
           ('mean','full_vocab_accuracy','C. Mean replacement','Correct first token (%)',(-1,31))]
    for ax,(kind,metric,title,ylabel,limits) in zip(axes,specs):
        d=curves[(curves.intervention==kind)&(curves.metric==metric)].sort_values('top_k')
        ax.fill_between(d.top_k,100*d.control_min,100*d.control_max,color='#8190A5',alpha=.17,linewidth=0)
        ax.plot(d.top_k,100*d.selected,'o-',color='#B52F6B',linewidth=2.2,markersize=5,label='Selected')
        ax.plot(d.top_k,100*d.control_mean,'s--',color='#8190A5',linewidth=2,markersize=4,label='Control mean')
        ax.axhline(100*d.clean.iloc[0],color='#161923',linestyle=':',linewidth=1.6,label='Clean')
        ax.set(title=title,xlabel='Top-k',ylabel=ylabel,ylim=limits,xticks=range(1,9))
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#E3E4EA',linewidth=.8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
    fig.subplots_adjust(left=.065,right=.99,bottom=.23,top=.85,wspace=.36)
    for ext in ('pdf','svg','png'):
        fig.savefig(root/f'nonthinking_readout_diagnosis.{ext}',dpi=190,facecolor='white')
    plt.close(fig)
    print('ALL SOURCE HASHES VERIFIED; SUMMARY COMPLETE')
    print(pd.DataFrame(table).round(3).to_string(index=False))
    print('L1H2:\n',singles[singles['heads']=='L1H2'][['intervention']+metrics].to_string(index=False))
    print('SCALING:\n',scale.round(4).to_string())


if __name__=='__main__':
    main()
