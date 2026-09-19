"""Read concise status for the isolated N=3 experiment without modifying it."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
args=p.parse_args()
r=args.root
result={'root':str(r),'status':(r/'status').read_text().strip() if (r/'status').exists() else 'UNKNOWN','models':{}}
for model,layers in [('Qwen3-8B',36),('Gemma4-E4B',42)]:
    d=r/'formal'/model
    item={'expected_rows':10*(1+5*layers),'probe_files':len(list((d/'probes').glob('L*.npz')))}
    c=d/'selection/cohort.json'
    if c.exists():item['cohort']=json.loads(c.read_text())
    rows=[]
    for f in (d/'prompts').glob('*.jsonl'):
        raw=f.read_bytes()
        # A last line still being written is not a completed cell.
        lines=raw[:raw.rfind(b'\n')+1].decode().splitlines() if b'\n' in raw else []
        rows.extend(json.loads(s) for s in lines)
    item['completed_rows']=len(rows)
    item['complete_prompts']=len(list((d/'prompts').glob('*.complete.json')))
    nonzero=[x for x in rows if x['condition'] in ('ridge','random')]
    if nonzero:
        last=nonzero[-1]
        item['latest_cell']={k:last[k] for k in ('seed','layer','beta','condition')}
        item['median_seconds']=statistics.median(x['elapsed_seconds'] for x in nonzero)
        item['approx_remaining_compute_minutes']=(item['expected_rows']-len(rows))*item['median_seconds']/60
    if (d/'complete.json').exists():item['completion']=json.loads((d/'complete.json').read_text())
    log=r/'logs'/f'{model}.log'
    if log.exists():
        with log.open('rb') as h:
            h.seek(max(0,log.stat().st_size-1000));item['log_tail']=h.read().decode('utf-8',errors='replace')
    result['models'][model]=item
if result['status']=='FAILED':
    result['failure_logs']={}
    for f in (r/'logs').glob('*.log'):
        with f.open('rb') as h:
            h.seek(max(0,f.stat().st_size-1800));result['failure_logs'][f.name]=h.read().decode('utf-8',errors='replace')
print(json.dumps(result,indent=2))
