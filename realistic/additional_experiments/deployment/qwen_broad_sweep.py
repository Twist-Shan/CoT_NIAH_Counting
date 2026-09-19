"""Post-hoc head-count sensitivity, using frozen kth discovery ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, data):
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temp.replace(p)

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--plan-only',action='store_true')
    args=ap.parse_args()
    base=args.base; out=args.output
    # Local archive layout and remote original layout are both supported.
    run=base/('downloaded/full' if (base/'downloaded/full').exists() else 'full')/'Qwen3-8B'
    package=base/'additional_experiments'
    if not package.exists(): package=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(package))
    from kth_retrieval import select_broad, random_bank
    from protocol import encode_ids
    env=read(run/'environment.json'); widths=env['head_counts']
    original=read(run/'frozen_selection.json')
    ranking=[tuple(x['head']) for x in original['broad']['nonthinking']['ranking']]
    assert select_broad(ranking,widths,32)==list(map(tuple,original['broad']['nonthinking']['heads']))
    banks={str(n):{'heads':select_broad(ranking,widths,n)} for n in (64,128,256)}
    for n,b in banks.items():
        b['random_heads']=[random_bank(b['heads'],widths,7000+i) for i in range(3)]
    with (base/'frozen/cases.jsonl').open(encoding='utf-8') as f:
        cases=[c for c in map(json.loads,f) if c['split']=='confirmation']
    assert len(cases)==100
    contract={'type':'posthoc_qwen_nonthinking_broad_head_count','sizes':[64,128,256],
        'selection_sha256':sha(run/'frozen_selection.json'),'base_contract_sha256':sha(run/'contract.json'),
        'frozen_manifest_sha256':sha(base/'frozen/manifest.json'),'script_sha256':sha(Path(__file__)),
        'banks':banks,'cases':[c['case_id'] for c in cases],
        'controls':'three layer-matched banks disjoint from selected; seeds 7000,7001,7002',
        'clean':'reuse original clean output; regenerate clean on k1 and k10 of seed1254 to check consistency',
        'scope':'answer_query','max_new_tokens':64}
    contract=json.loads(json.dumps(contract))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'contract.json').exists(): assert read(out/'contract.json')==contract
    else: save(out/'contract.json',contract)
    if args.plan_only:
        print('PLAN_PASS: nested top64/128/256, 100 cases, 1200 intervention arms + 2 clean checks')
        return
    base_contract=read(run/'contract.json')
    assert all(sha(package/k)==v for k,v in base_contract['code_hashes'].items())
    src=package.parent/'src'
    assert all(sha(src/k)==v for k,v in base_contract['src_hashes'].items())
    manifest=read(base/'frozen/manifest.json')
    assert all(sha(base/'frozen'/k)==v for k,v in manifest['files'].items())
    sys.path.insert(0,str(src))
    import torch
    from run import summarize_generation
    from realistic_niah_v4.modeling import load_registered_model, generate_with_head_ablation
    from realistic_niah_v4.spec import resolve_model_spec
    started=time.perf_counter(); torch.manual_seed(20260906)
    model,tok,adapter=load_registered_model(resolve_model_spec('Qwen3-8B'),cache_dir='outputs/external/lambda_nfs_CoT-Native-thinking-v5_hf_cache')
    tok.padding_side='left'
    save(out/'environment.json',{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'load_seconds':time.perf_counter()-started})
    for c in cases:
        cid=c['case_id']; dest=out/(cid+'.json')
        if dest.exists(): continue
        t=time.perf_counter(); cap=run/'captures/nonthinking'/cid
        for name,h in read(cap/'complete.json')['files'].items(): assert sha(cap/name)==h
        prefix=encode_ids(read(cap/'retrieval.json')['final_prefix_ids'])
        old=read(run/'causal/nonthinking'/(cid+'.json'))['assays']['broad']['arms'][0]
        if c['seed']==1254 and c['level'] in (1,10):
            clean=generate_with_head_ablation(model,tok,adapter,prefix,[],scope='answer_query',max_new_tokens=64)
            assert clean['generated_token_ids']==old['generation']['generated_token_ids'], 'Clean regeneration mismatch'
            save(out/('clean_check_'+cid+'.json'),clean)
        results={}
        for n,b in banks.items():
            arms=[old]
            for name,heads in [('selected',b['heads'])]+[(f'random_{i}',h) for i,h in enumerate(b['random_heads'])]:
                gen=generate_with_head_ablation(model,tok,adapter,prefix,heads,scope='answer_query',max_new_tokens=64)
                arms.append({'name':name,'heads':heads,'generation':gen,'score':summarize_generation(gen,c,mode='nonthinking',prefixed=True)})
            results[n]=arms
        save(dest,{'case_id':cid,'seed':c['seed'],'k':c['level'],'sizes':results,'elapsed_seconds':time.perf_counter()-t})
        print('case_complete',cid,flush=True)
    save(out/'complete.json',{'status':'PASS','cases':100,'intervention_arms':1200,'elapsed_seconds':time.perf_counter()-started})
    print('SWEEP_COMPLETE',flush=True)

if __name__=='__main__': main()
