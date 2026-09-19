"""Seed-equal kth behavior and selected-minus-random local ablation contrasts."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import time


def estimate(values):
    if not values:
        return {'seeds': 0, 'mean': None, 'lower': None, 'upper': None}
    mean=sum(values)/len(values)
    if len(values)==1:
        return {'seeds':1,'mean':mean,'lower':None,'upper':None}
    rng=random.Random(20260906)
    samples=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(2000))
    return {'seeds':len(values),'mean':mean,'lower':samples[49],'upper':samples[1949]}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    started=time.perf_counter()
    with (args.frozen/'cases.jsonl').open(encoding='utf-8') as f:
        cases={c['case_id']:c for c in map(json.loads,f)}
    rows, per_seed= [], defaultdict(lambda:defaultdict(list))
    availability=defaultdict(lambda:defaultdict(int))
    models=['Qwen3-8B','Gemma4-E4B']
    for model in models:
        done=json.loads((args.run/model/'complete.json').read_text(encoding='utf-8'))
        if done['status']!='PASS': raise ValueError('Run incomplete')
        for path in sorted((args.run/model/'captures').glob('*/*/generation.json')):
            c=cases[path.parent.name]
            mode=path.parents[1].name
            g=json.loads(path.read_text(encoding='utf-8'))
            for metric,value in [('accuracy',int(g['correct'])),('parse_ok',int(g['parse_ok'])),
                                 ('truncated',int(g['generation_truncated']))]:
                per_seed[(model,mode,c['split'],'behavior',metric)][c['seed']].append(value)
            retrieval=json.loads((path.parent/'retrieval.json').read_text(encoding='utf-8'))
            a=availability[(model,mode,c['split'])]
            a['cases']+=1
            a['final_query_available']+=retrieval['final_prefix_ids'] is not None
            a['targeted_query_available']+=retrieval['target_prefix_ids'] is not None
            a['trace_episode_available']+=retrieval['trace_record_sites']>0
        for path in sorted((args.run/model/'causal').glob('*/*.json')):
            data=json.loads(path.read_text(encoding='utf-8'))
            for assay,result in data['assays'].items():
                if result['status']!='PASS': continue
                correct={a['name']:int(a['score']['correct']) for a in result['arms']}
                control=sum(correct[f'random_{i}'] for i in range(3))/3
                values={'clean_accuracy':correct['clean'],'selected_accuracy':correct['selected'],
                        'random_accuracy':control,'selected_damage':correct['clean']-correct['selected'],
                        'selected_minus_random_failure':control-correct['selected']}
                for metric,value in values.items():
                    per_seed[(model,data['mode'],'confirmation',assay,metric)][data['seed']].append(value)
    args.output.mkdir(parents=True,exist_ok=False)
    for key,seeds in sorted(per_seed.items()):
        rows.append(dict(zip(('model','mode','split','assay','metric'),key),
                         **estimate([sum(v)/len(v) for v in seeds.values()]),
                         cases=sum(len(v) for v in seeds.values())))
    with (args.output/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    avail=[dict(zip(('model','mode','split'),k),**v) for k,v in availability.items()]
    (args.output/'availability.json').write_text(json.dumps(avail,indent=2),encoding='utf-8')
    (args.output/'analysis.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-started,
        'seed_bootstrap_replicates':2000,'unit':'seed, equal average over available k',
        'limitations':['Availability is not correctness filtering; report missing target mentions.',
                      'Native targeted score is next-city prefix; broad and nonthinking scores are complete city|score.',
                      'Local query ablation does not measure complete thinking rollout causality.']},indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
