"""Compact, read-only status for the canonical Enumeration run."""
from prepare_enumeration_default_seed import ROOT, MODE, MODELS, read
for name in ('setup_status.json','pipeline_status.json','pipeline_v2_status.json','update_status.json'):
    if (ROOT/name).exists(): print(name,read(ROOT/name))
for model in MODELS:
    p=ROOT/'fresh_causal_v1/registries'/model/MODE/'ledger.json'
    if p.exists():
        rows=read(p)
        print(model,{role:[r['seed'] for r in rows if r['gold_count']==10 and (r['seed']<1254)==(role=='discovery') and r['update_eligible']]
                     for role in ('discovery','confirmation')})
for p in sorted((ROOT/'execution').rglob('status.json')):
    v=read(p);print(str(p.relative_to(ROOT)), {k:v[k] for k in ('status','gpu','pid','seconds','error') if k in v})
for kind in ('read','retrieve','answer','update'):
    for p in sorted((ROOT/kind).rglob('*status.json')):
        v=read(p);print(str(p.relative_to(ROOT)),{k:v[k] for k in ('status','completed','completed_rows','completed_traces','completed_jobs','total','expected_rows','error') if k in v})
