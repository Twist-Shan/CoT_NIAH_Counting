"""Summarize completed per-model timings without interpreting discovery smoke effects."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path)
args=p.parse_args()
result={'models':{},'estimates':{}}
for model,layers in [('Qwen3-8B',36),('Gemma4-E4B',42)]:
    root=args.root/'benchmark'/model
    complete=root/'complete.json'
    rows=[]
    for file in (root/'prompts').glob('*.jsonl'):
        rows.extend(json.loads(s) for s in file.read_text().splitlines())
    cells=[r for r in rows if r['condition'] in ('ridge','random')]
    if not cells:
        result['models'][model]={'status':'WAITING','observed_cells':len(rows)}
        continue
    times=[r['elapsed_seconds'] for r in cells]
    summary=dict(status='PASS' if complete.exists() else 'RUNNING',layers=layers,
        observed_intervention_cells=len(cells),mean_seconds=statistics.mean(times),
        median_seconds=statistics.median(times),min_seconds=min(times),max_seconds=max(times),
        sequence_lengths=sorted(set(r['sequence_length'] for r in cells)),
        hook_counts=sorted(set(r['intervention_audit']['applications'] for r in cells)),
        min_changed_fraction=min(r['intervention_audit']['changed_fraction'] for r in cells),
        max_random_norm_relative_error=max(abs(r['intervention_audit']['realized_norm_ratio']-1) for r in cells),
        cache_checks=[json.loads(f.read_text())['status'] for f in root.glob('cache_equivalence*.json')])
    if complete.exists(): summary['completion']=json.loads(complete.read_text())
    result['models'][model]=summary
if all(m['status']=='PASS' for m in result['models'].values()):
    for doses in (6,2):
        work={}
        for model,m in result['models'].items():
            evaluations=100+m['layers']*(100+550*doses*2)
            work[model]={'evaluations':evaluations,'gpu_hours':evaluations*m['mean_seconds']/3600}
        total=sum(m['gpu_hours'] for m in work.values())
        # Parallel independent layer/prompt shards; reserve 15--30% overhead.
        result['estimates'][str(doses)+'_nonzero_doses']={
            'models':work,'single_gpu_compute_hours':total,
            'eight_gpu_ideal_hours':total/8,
            'eight_gpu_planning_hours':[total/8*1.15,total/8*1.30],
            'excludes':'cold model loads, all-layer discovery fitting, different GPU/software performance',
            'unit':'100 prompts/model; 550 single-span targets/model; 78 total layers'}
text=json.dumps(result,indent=2,allow_nan=False)
if args.output: args.output.write_text(text+'\n',encoding='utf-8')
print(text)
