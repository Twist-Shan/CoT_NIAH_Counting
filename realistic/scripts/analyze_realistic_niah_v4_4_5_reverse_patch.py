"""Validate and summarize the descriptive, sparse-layer reverse-patch pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    root=Path(args.input_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    summary=[]; seed_summary=[]; baselines=[]; normalization=[]; audits={}
    for model in ('Qwen3-8B','Gemma4-E4B'):
        folder=root/model
        complete=json.loads((folder/'complete.json').read_text())
        if complete['status']!='PASS': raise RuntimeError('Run incomplete')
        provenance=json.loads((folder/'run_provenance.json').read_text())
        if provenance['schema_version']!='v445_reverse_patch_pilot_v2':
            raise RuntimeError('Only the backend-aligned v2 pilot may enter this summary')
        effective=json.loads((folder/'effective_backend.json').read_text())
        expected_backend='eager' if model=='Gemma4-E4B' else 'sdpa'
        if effective['text']!=expected_backend or any(v!=expected_backend for v in effective['attention_layers']):
            raise RuntimeError('Effective backend differs from the historical alignment protocol')
        rows=jsonl(folder/'detail.jsonl')
        keys=[(r['seed'],r['gold_count'],r['condition'],r['patch_layer']) for r in rows]
        expected={(s,c,k,l) for s in provenance['seeds'] for c in provenance['counts']
                  for k,l in [(b,-1) for b in ('clean','needle_corrupt','ordinary_corrupt')]+
                  [(k,l) for l in provenance['layers'] for k in ('self_needle_full',
                   'reverse_needle_full','reverse_needle_endpoint','reverse_ordinary_full','restore_needle_full')]}
        if len(keys)!=len(set(keys)) or set(keys)!=expected: raise RuntimeError('Incomplete or duplicate cells')
        by_key=dict(zip(keys,rows))
        for seed in provenance['seeds']:
            for count in provenance['counts']:
                clean=by_key[(seed,count,'clean',-1)]['expected_count']
                for condition in ('needle_corrupt','ordinary_corrupt'):
                    corrupt=by_key[(seed,count,condition,-1)]['expected_count']
                    normalization.append({'model':model,'seed':seed,'gold_count':count,
                        'corruption':condition,'clean_expected_count':clean,'corrupt_expected_count':corrupt,
                        'denominator':clean-corrupt,'nonpositive':clean-corrupt<=0})
        align=jsonl(folder/'alignment.jsonl'); selfs=jsonl(folder/'self_patch_audit.jsonl')
        hist=jsonl(folder/'history_alignment.jsonl')
        pair_count=len(provenance['seeds'])*len(provenance['counts'])
        if len(align)!=pair_count*2 or any(r['status']!='PASS' for r in align):
            raise RuntimeError('Missing or failed token alignment audit')
        if len(selfs)!=pair_count*len(provenance['layers']):
            raise RuntimeError('Missing self-patch audit')
        if len(hist)!=pair_count*(3+len(provenance['layers'])):
            raise RuntimeError('Missing historical comparison')
        if any(r['expected_count_delta']>1e-5 or r['probability_tv']>1e-6 or not r['strict_agrees'] for r in selfs):
            raise RuntimeError('Self-patch audit failed')
        if any(r['expected_count_delta']>0.05 or r['probability_tv']>0.01 or not r['strict_agrees'] for r in hist):
            raise RuntimeError('Historical comparison failed')
        if any(r['hook_applications']!=1 for r in rows if r['patch_layer']>=0):
            raise RuntimeError('Patch application audit failed')
        audits[model]={'rows':len(rows),'expected_rows':len(expected),
            'effective_text_backend':expected_backend,
            'nonpositive_needle_denominators':sum(r['nonpositive'] for r in normalization if r['model']==model and r['corruption']=='needle_corrupt'),
            'alignment_rows':len(align),'self_patch_rows':len(selfs),
            'max_self_expected_delta':max(r['expected_count_delta'] for r in selfs),
            'max_self_tv':max(r['probability_tv'] for r in selfs),
            'max_history_expected_delta':max(r['expected_count_delta'] for r in hist),
            'max_history_tv':max(r['probability_tv'] for r in hist),
            'history_strict_agreement':all(r['strict_agrees'] for r in hist),
            'self_strict_agreement':all(r['strict_agrees'] for r in selfs),
            'hook_applications_all_one':all(r['hook_applications']==1 for r in rows if r['patch_layer']>=0)}
        grouped=defaultdict(list)
        for r in rows:
            seed,count,condition,layer=r['seed'],r['gold_count'],r['condition'],r['patch_layer']
            if layer<0:
                baselines.append({'model':model,'seed':seed,'gold_count':count,'condition':condition,
                    'expected_count':r['expected_count'],'strict_prediction':r['strict_prediction']})
                continue
            clean=by_key[(seed,count,'clean',-1)]
            corrupt=by_key[(seed,count,'needle_corrupt',-1)]
            denom=clean['expected_count']-corrupt['expected_count']
            if condition=='restore_needle_full':
                metrics={'expected_error_repair':abs(corrupt['expected_count']-count)-abs(r['expected_count']-count),
                         'normalized_restoration':None if abs(denom)<1e-8 else (r['expected_count']-corrupt['expected_count'])/denom}
            elif condition.startswith('reverse'):
                metrics={k:r[k] for k in ('expected_error_increase','expected_count_drop','normalized_damage','strict_error_increase')}
                if condition=='reverse_needle_full':
                    ordinary=by_key[(seed,count,'reverse_ordinary_full',layer)]
                    metrics['needle_minus_ordinary_error_increase']=r['expected_error_increase']-ordinary['expected_error_increase']
            else: continue
            for metric,value in metrics.items():
                if value is not None: grouped[(condition,layer,metric,seed)].append(value)
        groups=defaultdict(list)
        for (condition,layer,metric,seed),values in grouped.items():
            value=mean(values)
            seed_summary.append({'model':model,'condition':condition,'layer':layer,'metric':metric,
                'seed':seed,'mean_over_counts':value,'valid_count_cells':len(values)})
            groups[(condition,layer,metric)].append(value)
        for (condition,layer,metric),values in groups.items():
            summary.append({'model':model,'condition':condition,'layer':layer,'metric':metric,
                'mean':mean(values),'seed_min':min(values),'seed_max':max(values),'seed_count':len(values)})
    for name,rows in [('summary.csv',summary),('seed_summary.csv',seed_summary),('baselines.csv',baselines),('normalization_audit.csv',normalization)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'audit.json').write_text(json.dumps({'status':'PASS','models':audits,
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inference':'descriptive pilot; no confidence intervals or significance tests'},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors={'reverse_needle_full':'#009EC2','reverse_needle_endpoint':'#8F77B5','reverse_ordinary_full':'#8C9298','restore_needle_full':'#E79143'}
    labels={'reverse_needle_full':'Reverse: full span','reverse_needle_endpoint':'Reverse: endpoint','reverse_ordinary_full':'Reverse: ordinary control','restore_needle_full':'Restoration: full span'}
    for j,model in enumerate(('Qwen3-8B','Gemma4-E4B')):
        for i,metric in enumerate(('expected_error_increase','normalized_damage')):
            ax=axes[i,j]
            conditions=['reverse_needle_full','reverse_needle_endpoint','reverse_ordinary_full'] if i==0 else ['reverse_needle_full','restore_needle_full']
            for condition in conditions:
                m='normalized_restoration' if condition=='restore_needle_full' else metric
                rows=sorted([r for r in summary if r['model']==model and r['condition']==condition and r['metric']==m],key=lambda r:r['layer'])
                ax.plot([r['layer'] for r in rows],[r['mean'] for r in rows],'o--',color=colors[condition],label=labels[condition],markersize=4)
            ax.axhline(0,color='#999999',linewidth=.7)
            title=model+' | '+audits[model]['effective_text_backend']
            if i==1:
                title=f"Signed denominator <= 0: {audits[model]['nonpositive_needle_denominators']}/4 samples"
            ax.set(title=title,xlabel='Intervention layer (zero-based)',ylabel='Change in absolute expected-count error' if i==0 else 'Normalized effect (unclipped)')
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.18); ax.legend(fontsize=8)
    fig.suptitle('Exploratory reverse patch: 2 discovery seeds, counts 2 and 10\nSparse sampled layers; dashed lines are visual guides; no confidence intervals',fontsize=11)
    fig.savefig(out/'reverse_patch_pilot.png',dpi=170)
    plt.close(fig)
    print(json.dumps(audits,indent=2))


if __name__=='__main__': main()
