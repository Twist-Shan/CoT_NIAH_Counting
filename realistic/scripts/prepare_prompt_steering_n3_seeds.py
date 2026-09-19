"""Reproduce one canonical family before freezing held-out supplementary seeds."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from dataset_generation.dynamic_niah import TokenizerAdapter
from realistic_niah_v4.spec import V4Config
from realistic_niah_v4.stimuli import ControlledFreezeSpec, build_controlled_family, load_stimuli

p=argparse.ArgumentParser(description=__doc__)
for name in ('canonical-stimuli','canonical-manifest','source-repo','cache-dir','output'):
    p.add_argument('--'+name,type=Path,required=True)
args=p.parse_args()
out=args.output
out.mkdir(parents=True,exist_ok=True)
manifest=json.loads(args.canonical_manifest.read_text())
base=V4Config.from_mapping(manifest['config'])
config=replace(base,seeds=tuple(range(1234,1284)),discovery_seeds=tuple(range(1234,1254)),
    confirmation_seeds=tuple(range(1254,1284)))
config.validate()
freeze_kwargs=dict(manifest['freeze_spec'])
for key in ('haystack_dir','entities_path','fact_templates_path'):
    freeze_kwargs[key]=str(args.source_repo/freeze_kwargs[key])
freeze_kwargs['tokenizer_cache_dir']=str(args.cache_dir)
spec=ControlledFreezeSpec(config=config,**freeze_kwargs)
tokenizer=TokenizerAdapter(config.canonical_tokenizer,revision=config.canonical_tokenizer_revision,
    cache_dir=str(args.cache_dir))
if tokenizer.backend!='huggingface': raise RuntimeError('Canonical tokenizer did not load')
canonical={(int(r['seed']),int(r['gold_count'])):r for r in load_stimuli(args.canonical_stimuli) if r['design_variant']=='v4.4'}
started=time.perf_counter()
rebuilt,_=build_controlled_family(variant='v4.4',seed=1254,tokenizer=tokenizer,freeze_spec=spec,active_counts=[3])
checked=('passage_sha256','passage','active_needle_spans','slots','gold_pairs','canonical_passage_tokens')
errors=[k for k in checked if rebuilt[0][k]!=canonical[1254,3][k]]
old_design=canonical[1254,3]['design'];new_design=rebuilt[0]['design']
errors.extend('design.'+k for k,v in old_design.items() if k not in new_design or new_design[k]!=v)
added=set(new_design)-set(old_design)
known_added={'slot_nominal_final_starts','insertion_boundary_policy','insertion_boundary_remaps','insertion_boundary_remapped_slots'}
if not added<=known_added: errors.append('unexpected new design keys')
if new_design.get('insertion_boundary_remapped_slots')!=0: errors.append('canonical reproduction remapped a slot')
if errors: raise RuntimeError(f'Generator does not reproduce canonical seed1254/N3: {errors}')
destination=out/'supplementary_n3.jsonl'
audit_file=out/'generation_audit.json'
if destination.exists():
    if not audit_file.exists(): raise RuntimeError('Incomplete supplementary generation; preserve before retry')
    prior=json.loads(audit_file.read_text())
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=prior['supplementary_sha256']:
        raise RuntimeError('Frozen supplementary stimuli changed')
    print('Supplementary seeds already frozen',flush=True)
else:
    families=[];rows=[]
    for seed in range(1264,1284):
        new,metadata=build_controlled_family(variant='v4.4',seed=seed,tokenizer=tokenizer,
            freeze_spec=spec,active_counts=[3])
        rows.extend(new);families.append(metadata)
        print(json.dumps(dict(event='supplementary_seed',seed=seed,elapsed_seconds=time.perf_counter()-started)),flush=True)
    raw=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)
    temporary=destination.with_suffix('.jsonl.tmp')
    temporary.write_text(raw,encoding='utf-8'); temporary.replace(destination)
    evidence=dict(status='PASS',canonical_reproduction_seed=1254,canonical_reproduction_count=3,
        checked_fields=list(checked),old_design_fields_checked=list(old_design),
        new_audit_only_design_fields={k:new_design[k] for k in sorted(added)},
        new_seeds=list(range(1264,1284)),rows=len(rows),
        config=config.to_dict(),freeze_spec=freeze_kwargs,families=families,
        canonical_sha256=hashlib.sha256(args.canonical_stimuli.read_bytes()).hexdigest(),
        supplementary_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started)
    audit_file.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    (out/'extended_config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(event='supplementary_complete',status='PASS',rows=len(rows))),flush=True)
