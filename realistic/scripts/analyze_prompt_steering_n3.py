"""Validate and summarize the completed N=3 scan; never infer a universal null."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
args=p.parse_args()
out=args.root/'analysis';out.mkdir(parents=True,exist_ok=True)
summaries=[]; paired=[]; audits={}
rng=np.random.default_rng(20260910)
for model,nlayers in [('Qwen3-8B',36),('Gemma4-E4B',42)]:
    directory=args.root/'formal'/model
    done=json.loads((directory/'complete.json').read_text())
    cohort=json.loads((directory/'selection/cohort.json').read_text())
    assert done['status']=='PASS' and len(cohort['seeds'])==10
    rows=[]
    for seed in cohort['seeds']:
        file=directory/'prompts'/f'seed{seed}_N3.jsonl'
        marker=json.loads(file.with_suffix('.complete.json').read_text())
        assert hashlib.sha256(file.read_bytes()).hexdigest()==marker['sha256']
        prompt=[json.loads(s) for s in file.read_text().splitlines()]
        assert len(prompt)==1+nlayers*5
        rows.extend(prompt)
    assert len(rows)==10*(1+nlayers*5)
    clean={r['seed']:r for r in rows if r['condition']=='clean'}
    assert len(clean)==10 and all(r['strict_correct'] for r in clean.values())
    cells={(r['seed'],r['layer'],r['beta'],r['condition']):r for r in rows if r['condition']!='clean'}
    assert len(cells)==nlayers*50
    for r in cells.values():
        a=r['intervention_audit']
        assert a['applications']==1
        if r['condition']=='noop':
            assert r['generated_token_ids']==clean[r['seed']]['generated_token_ids']
        else:
            assert r['span']==2 and a['realized_norm']>0
            assert abs(a['realized_norm_ratio']-1)<.05
    for layer in range(nlayers):
        for beta in [-1.,1.]:
            effect={}
            for condition in ['ridge','random']:
                subset=[cells[s,layer,beta,condition] for s in cohort['seeds']]
                delta=np.array([r['expected_count']-clean[r['seed']]['expected_count'] for r in subset])
                effect[condition]=delta*beta
                generated=[r['parsed_count']-3 for r in subset if r['format_valid'] and not r['generation_truncated']]
                summaries.append(dict(model=model,layer=layer,beta=beta,condition=condition,seeds=10,
                    accuracy=np.mean([r['strict_correct'] for r in subset]),
                    valid_rate=np.mean([r['format_valid'] and not r['generation_truncated'] for r in subset]),
                    prediction_changed_rate=np.mean([r['parsed_count']!=3 or r['generation_truncated'] for r in subset]),
                    mean_expected_count_shift=float(delta.mean()),mean_direction_aligned_shift=float((delta*beta).mean()),
                    mean_generated_shift_valid_only=float(np.mean(generated)) if generated else '',
                    mean_endpoint_probe_shift=np.mean([r['intervention_audit']['endpoint_probe_shift'] for r in subset]),
                    mean_relative_state_norm=np.mean([r['intervention_audit']['relative_state_norm'] for r in subset])))
            differences=effect['ridge']-effect['random']
            bootstrap=differences[rng.integers(0,10,size=(10000,10))].mean(axis=1)
            low,high=np.quantile(bootstrap,[.025,.975])
            paired.append(dict(model=model,layer=layer,beta=beta,seeds=10,
                ridge_minus_random_aligned_shift=float(differences.mean()),
                ci95_low=float(low),ci95_high=float(high),interval_type='pointwise_seed_bootstrap'))
    audits[model]=dict(status='PASS',rows=len(rows),seeds=cohort['seeds'],excluded=cohort['excluded_seeds'],
        all_clean_correct=True,all_layers=nlayers,all_noop_checks_pass=True,
        min_realized_nonzero_norm=min(r['intervention_audit']['realized_norm'] for r in cells.values() if r['condition']!='noop'),
        max_random_norm_error=max(abs(r['intervention_audit']['realized_norm_ratio']-1) for r in cells.values() if r['condition']=='random'))
for name,data in [('layer_summary.csv',summaries),('paired_effects.csv',paired)]:
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(out/'audit.json').write_text(json.dumps(audits,indent=2)+'\n')
print(json.dumps(audits,indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral','DejaVu Serif'],
    'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(6.5,3.8),layout='constrained',sharex='col')
for column,(model,color) in enumerate([('Qwen3-8B','#168DCA'),('Gemma4-E4B','#E87824')]):
    for beta,style in [(-1.,'--'),(1.,'-')]:
        for condition,alpha in [('ridge',1.),('random',.4)]:
            data=[r for r in summaries if r['model']==model and r['beta']==beta and r['condition']==condition]
            x=[r['layer'] for r in data]
            label=f'{condition}, beta={beta:+.0f}'
            axes[0,column].plot(x,[r['mean_expected_count_shift'] for r in data],style,color=color,alpha=alpha,lw=1.2,label=label)
            axes[1,column].plot(x,[r['accuracy']*100 for r in data],style,color=color,alpha=alpha,lw=1.2)
    axes[0,column].set_title(model)
    axes[0,column].axhline(0,color='0.6',lw=.6)
    axes[0,column].legend(frameon=False,ncol=2)
    axes[1,column].set_xlabel('Intervention layer')
    axes[1,column].set_ylim(-2,102)
    for a in axes[:,column]:a.grid(axis='y',alpha=.15)
axes[0,0].set_ylabel('Expected-count change')
axes[1,0].set_ylabel('Accuracy (%)')
fig.savefig(out/'n3_prompt_steering.pdf')
fig.savefig(out/'n3_prompt_steering.png',dpi=220)
plt.close(fig)
