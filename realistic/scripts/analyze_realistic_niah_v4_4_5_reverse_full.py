"""Audit the complete canonical grid and generate split-preserving seed summaries."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from realistic_niah_v4_4_5.reverse_full import (expected_cells, load_journal,
    journal_digest, self_gate, history_gate, atomic_json)


def write_csv(path, rows):
    if not rows:
        path.write_text('no_records\n')
        return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--full-config',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    design=json.loads(Path(args.full_config).read_text())
    root=Path(args.input_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list); normalization=[]; legacy=[]; model_audits={}
    for model,n_layers in design['num_layers'].items():
        folder=root/model;contract=json.loads((folder/'run_contract.json').read_text())
        layers=list(range(n_layers))
        validation_layers=design.get('validation_layers',{}).get(model,layers)
        if contract['design']!=design or contract['counts']!=design['counts'] or contract['layers']!=layers:
            raise RuntimeError('Run is not the complete frozen design')
        if contract['effective_text_backend']!=design['effective_text_backend'][model]:
            raise RuntimeError('Wrong effective backend')
        expected_names={f'seed_{s}_count_{c}.jsonl' for s in design['seeds'] for c in design['counts']}
        if {p.name for p in (folder/'prompts').glob('*.jsonl')}!=expected_names:
            raise RuntimeError('Missing or extra prompt journals')
        audit={'rows':0,'self_checks':0,'history_checks':0,'legacy_reconstructions':0,
               'max_self_delta':0.0,'max_history_primary_delta':0.0,'effective_text_backend':contract['effective_text_backend']}
        for seed in design['seeds']:
            for count in design['counts']:
                path=folder/'prompts'/f'seed_{seed}_count_{count}.jsonl'
                marker=json.loads(path.with_suffix('.complete.json').read_text())
                if marker['status']!='PASS' or marker['journal_sha256']!=journal_digest(path):
                    raise RuntimeError('Missing or changed completion marker')
                rows=load_journal(path,seed=seed,count=count,layers=layers,validation_layers=validation_layers)
                if set(rows)!=expected_cells(layers, validation_layers):raise RuntimeError('Incomplete cell grid')
                meta=json.loads(path.with_suffix('.meta.json').read_text())
                if len(meta['alignment'])!=2 or any(x['status']!='PASS' for x in meta['alignment']):
                    raise RuntimeError('Token alignment failed')
                split='discovery' if seed in design['discovery_seeds'] else 'confirmation'
                if meta['split']!=split:raise RuntimeError('Split mismatch')
                base=rows['clean',-1];corrupt=rows['needle_corrupt',-1]
                gap=base['expected_count']-corrupt['expected_count']
                for condition in ['needle_corrupt','ordinary_corrupt']:
                    denominator=base['expected_count']-rows[condition,-1]['expected_count']
                    normalization.append({'model':model,'seed':seed,'count':count,'split':split,
                        'corruption':condition,'denominator':denominator,'nonpositive':denominator<=0,
                        'clean_strict_correct':base['strict_correct']})
                for (condition,layer),row in rows.items():
                    audit['rows']+=1
                    if row['split']!=split:raise RuntimeError('Row split changed')
                    if layer>=0 and row['hook_applications']!=1:raise RuntimeError('Bad patch count')
                    if 'self_check' in row:
                        if not self_gate(row['self_check']):raise RuntimeError('Self gate failed')
                        audit['self_checks']+=1
                        audit['max_self_delta']=max(audit['max_self_delta'],row['self_check']['expected_count_delta'])
                    if 'history_check' in row:
                        check=row['history_check'];audit['history_checks']+=1
                        audit['max_history_primary_delta']=max(audit['max_history_primary_delta'],check['primary']['expected_count_delta'])
                        if check['status']=='DIRECT_PASS':
                            if not history_gate(check['primary']):raise RuntimeError('Invalid direct history pass')
                        elif check['status']=='LEGACY_REPRODUCED_PRIMARY_RETAINED':
                            if not history_gate(check['legacy_sdpa_reconstruction']):raise RuntimeError('Legacy reconstruction failed')
                            audit['legacy_reconstructions']+=1
                            legacy.append({'model':model,'seed':seed,'count':count,'condition':condition,
                                'layer':layer,'primary_expected_delta':check['primary']['expected_count_delta'],
                                'primary_tv':check['primary']['probability_tv'],
                                'legacy_expected_delta':check['legacy_sdpa_reconstruction']['expected_count_delta'],
                                'legacy_tv':check['legacy_sdpa_reconstruction']['probability_tv']})
                        else:raise RuntimeError('Unexplained historical discrepancy')
                    metrics={}
                    if condition.startswith('reverse'):
                        metrics={k:row[k] for k in ['expected_error_increase','expected_count_drop','normalized_damage','strict_error_increase']}
                        if condition=='reverse_needle_full':
                            metrics['needle_minus_ordinary_error_increase']=row['expected_error_increase']-rows['reverse_ordinary_full',layer]['expected_error_increase']
                    if condition=='restore_needle_full':
                        old_clean=base['history_check']['historical_expected_count']
                        old_corrupt=corrupt['history_check']['historical_expected_count']
                        old_restore=row['history_check']['historical_expected_count'];old_gap=old_clean-old_corrupt
                        metrics={'expected_error_repair':abs(corrupt['expected_count']-count)-abs(row['expected_count']-count),
                            'normalized_restoration':None if abs(gap)<1e-8 else (row['expected_count']-corrupt['expected_count'])/gap,
                            'historical_normalized_restoration':None if abs(old_gap)<1e-8 else (old_restore-old_corrupt)/old_gap,
                            'historical_expected_error_repair':abs(old_corrupt-count)-abs(old_restore-count)}
                    for metric,value in metrics.items():
                        if value is not None:
                            if not np.isfinite(value):raise RuntimeError('Nonfinite summary value')
                            grouped[model,seed,condition,layer,metric].append(value)
        if audit['rows']!=design['all_rows_with_validation'][model]:raise RuntimeError('Wrong total coverage')
        if audit['self_checks']!=300*len(validation_layers) or audit['history_checks']!=300*(3+len(validation_layers)):
            raise RuntimeError('Missing validation checks')
        model_audits[model]=audit
    seed_rows=[];lookup={}
    for (model,seed,condition,layer,metric),values in grouped.items():
        value=float(np.mean(values));lookup[model,seed,condition,layer,metric]=value
        seed_rows.append({'model':model,'seed':seed,'split':'discovery' if seed in design['discovery_seeds'] else 'confirmation',
            'condition':condition,'layer':layer,'metric':metric,'mean':value,'valid_count_cells':len(values)})
    rng=np.random.default_rng(design['bootstrap_seed']);summary=[]
    populations={'discovery':design['discovery_seeds'],'confirmation':design['confirmation_seeds'],'all':design['seeds']}
    groups=sorted({(m,c,l,k) for m,s,c,l,k in lookup})
    for population,seeds in populations.items():
        bootstrap=rng.integers(0,len(seeds),(design['bootstrap_repetitions'],len(seeds)))
        for model,condition,layer,metric in groups:
            if any((model,s,condition,layer,metric) not in lookup for s in seeds):
                raise RuntimeError('An entire seed has undefined metric; requires explicit treatment')
            values=np.array([lookup[model,s,condition,layer,metric] for s in seeds])
            draws=values[bootstrap].mean(axis=1);lo,hi=np.quantile(draws,[.025,.975])
            summary.append({'model':model,'population':population,'condition':condition,'layer':layer,
                'metric':metric,'mean':float(values.mean()),'ci95_low':float(lo),'ci95_high':float(hi),
                'seed_count':len(seeds)})
    write_csv(out/'seed_summary.csv',seed_rows);write_csv(out/'layer_summary.csv',summary)
    write_csv(out/'normalization_audit.csv',normalization);write_csv(out/'legacy_checks.csv',legacy)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for population in populations:
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for j,model in enumerate(design['num_layers']):
            specs=[('reverse_needle_full','expected_error_increase','Full span','#009ec2'),
                   ('reverse_needle_endpoint','expected_error_increase','Endpoint','#8f77b5'),
                   ('reverse_ordinary_full','expected_error_increase','Ordinary control','#8c9298')]
            for i in range(2):
                ax=axes[i,j]
                current=specs if i==0 else [('reverse_needle_full','normalized_damage','Reverse full','#009ec2'),
                    ('restore_needle_full','normalized_restoration','Contemporary restoration','#e79143'),
                    ('restore_needle_full','historical_normalized_restoration','Historical restoration','#444444')]
                for condition,metric,label,color in current:
                    rows=sorted([r for r in summary if r['model']==model and r['population']==population and r['condition']==condition and r['metric']==metric],key=lambda r:r['layer'])
                    xx=[r['layer'] for r in rows];yy=[r['mean'] for r in rows]
                    ax.plot(xx,yy,label=label,color=color,marker='o' if condition=='restore_needle_full' else None,linestyle=':' if label.startswith('Historical') else '-')
                    if not label.startswith('Historical'):
                        ax.fill_between(xx,[r['ci95_low'] for r in rows],[r['ci95_high'] for r in rows],color=color,alpha=.13)
                ax.set(title=model if i==0 else '',xlabel='Intervention layer (zero-based)',
                    ylabel='Change in absolute expected-count error' if i==0 else 'Normalized effect (unclipped)')
                ax.axhline(0,color='#999',linewidth=.7);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
        fig.suptitle(f'Full aligned reverse patch: {population}, {len(populations[population])} seeds x counts 1–10\n95% pointwise seed-bootstrap intervals; normalized effects can have negative denominators',fontsize=11)
        fig.savefig(out/f'reverse_full_{population}.png',dpi=170);plt.close(fig)
    audit={'status':'PASS','models':model_audits,'total_rows':sum(a['rows'] for a in model_audits.values()),
        'legacy_reconstructed_rows':len(legacy),'primary_population':'confirmation',
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(out/'audit.json',audit)
    report=['# Full aligned reverse patch','',f"Coverage: {audit['total_rows']:,} rows; 30 canonical seeds × counts 1–10 per model; all 36/42 layers.",'',
        'The primary inferential population is the original 10 confirmation seeds. Discovery and all-seed descriptive summaries are separate. Counts are averaged within seed before 10,000 seed-bootstrap replicates. Intervals are pointwise, not simultaneous.','',
        f"Historical legacy reconstruction was needed for {len(legacy)} comparison cells. New primary rows always retain same-backend donor/recipient execution; old SDPA-donor/eager-recipient reconstruction is audit-only. See legacy_checks.csv. No numerical tolerance was relaxed.",'',
        'Qwen uses SDPA; Gemma uses eager text attention. All scheduled representative-layer self-patches and restoration checks, all-layer hook counts, coverage, contracts, and explained historical comparisons passed. Contemporary restoration is sampled only at configured validation layers; connecting lines are visual guides. Negative/null normalization denominators are retained and audited. Expected-count metrics condition on candidate answers 1–10 and do not equal strict-generation accuracy.','',
        '![Confirmation results](reverse_full_confirmation.png)','',
        'Full tables: layer_summary.csv, seed_summary.csv, normalization_audit.csv, legacy_checks.csv, audit.json.','',
        'Interpretation is limited to the measured interventions. Full-state replacement is not a count-feature-specific necessity proof. Final-layer source patches have no later cross-position computation and their zero effect is a timing check.']
    (out/'README.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    atomic_json(out/'complete.json',{'status':'PASS','total_rows':audit['total_rows']})
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
