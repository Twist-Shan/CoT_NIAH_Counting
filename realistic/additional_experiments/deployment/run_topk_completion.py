"""Nested dose curves on frozen additional-task confirmation endpoints."""
import argparse, json, hashlib, random, sys, time
from pathlib import Path
from collections import Counter, defaultdict
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R),str(R/'src'),str(R/'helpers')]
from run_aligned_transfer import read, write, sha, controls

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--canary',action='store_true'); a=ap.parse_args()
    cfg=read(R/'protocol.json')
    for name,h in cfg['files'].items(): assert sha(R/name)==h,(name,'frozen hash')
    import torch
    from protocol import encode_ids
    from run import summarize_generation
    from realistic_niah_v4.modeling import load_registered_model,generate_with_head_ablation,generate_answer_completion,_is_prompt_prefill,_text_config
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v5.causal import _first_generated_city_record
    import re
    torch.manual_seed(20260906)
    model,tok,adapter=load_registered_model(resolve_model_spec(a.model),cache_dir=cfg['cache'])
    widths=list(adapter.num_heads)
    def backend():
        vals=[getattr(c,'_attn_implementation',None) for c in [model.config,_text_config(model)] if c is not None]
        assert vals and all(v=='sdpa' for v in vals),vals
        return vals
    backend()
    stage='canary' if a.canary else 'full'; out=R/stage/a.model; out.mkdir(parents=True,exist_ok=True)
    write(out/'environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),widths=widths,backend=backend(),protocol_sha256=sha(R/'protocol.json')))
    def generate(prefix,heads,persistent):
        enc=encode_ids(prefix); backend()
        if not persistent: result=generate_with_head_ablation(model,tok,adapter,enc,heads,scope='answer_query',max_new_tokens=64)
        elif not heads: result=generate_answer_completion(model,tok,enc,max_new_tokens=128)
        else:
            handles=[]; groups=defaultdict(list); calls=Counter()
            for l,h in heads: groups[l].append(h)
            for l,hs in groups.items():
                def hook(module,args,l=l,hs=hs):
                    value=args[0]; patched=value.clone(); pos=[enc.query_position] if _is_prompt_prefill(value,enc) else slice(None)
                    for h in hs: patched[:,pos,h*adapter.head_dims[l]:(h+1)*adapter.head_dims[l]]=0
                    calls[l]+=1; return (patched,*args[1:])
                handles.append(adapter.output_projections[l].register_forward_pre_hook(hook))
            try: result=generate_answer_completion(model,tok,enc,max_new_tokens=128)
            finally:
                for h in handles: h.remove()
            assert all(calls[l] for l in groups); result['hook_calls']=dict(calls)
        backend(); return result
    for task in ['kth','category']:
        plans=read(R/'plans'/f'{task}_{a.model}.json'); banks=read(R/'banks'/f'{task}_{a.model}.json')
        panel=[p for p in plans if p['split']=='confirmation']; assert len(panel)==200
        if a.canary:
            cid=next(p['case_id'] for p in panel if p['mode']=='native_thinking' and p['target'] and p['broad_prefix'])
            panel=[p for p in panel if p['case_id']==cid]
        for p in panel:
            source=Path(p['source'])
            for name,h in p['source_hashes'].items(): assert sha(source/name)==h
            ids=read(source/'prompt.json')['input_ids']+read(source/'generation.json')['generated_token_ids']
            for assay in ['broad','targeted']:
                if assay=='targeted' and p['mode']=='nonthinking': continue
                length=p['broad_prefix'] if assay=='broad' else (p['target'] or {}).get('prefix_length')
                key=p['mode']+'_'+assay; bank=banks[key]; sizes=bank['sizes']
                if a.canary: sizes=sorted(set([sizes[0],sizes[-1]]))
                dest=out/task/key/p['case_id']; dest.mkdir(parents=True,exist_ok=True)
                if not length:
                    write(dest/'unavailable.json',dict(case_id=p['case_id'],seed=p['seed'],reason=p['unavailable'].get(assay))); continue
                oldpath=Path(cfg['sources'][a.model][task])/'full'/a.model/task/'causal'/f"{p['mode']}_{p['case_id']}.json"
                assert sha(oldpath)==cfg['prior_results'][str(oldpath)]
                previous=read(oldpath)['assays'][assay]['arms']; oldclean=previous[0]
                def score(gen):
                    if assay=='broad': return summarize_generation(gen,p['case'],mode=p['mode'],prefixed=True)
                    raw=gen['completion_text_raw']; pred,start,kind=_first_generated_city_record(raw,[r['city'] for r in p['case']['records']]); hits=[] if pred is None else [(start,pred)]
                    for m in re.finditer(r'(?:city|flower) score audit,\s+([^\n,.]+?) received a score',raw,re.I): hits.append((m.start(1),m[1]))
                    prediction=min(hits)[1] if hits else None
                    return dict(correct=prediction is not None and prediction.casefold()==p['target']['target_city'].casefold(),prediction=prediction,target=p['target']['target_city'],policy='first_semantic_record_with_city_flower_audit_extension')
                assert bool(score(oldclean['generation'])['correct'])==bool(oldclean['score']['correct'])
                if a.canary:
                    gen=generate(ids[:length],[],assay=='targeted'); fresh=score(gen)
                    assert bool(fresh['correct'])==bool(oldclean['score']['correct']),('clean correctness changed',task,p['case_id'],assay)
                    write(dest/'clean_check.json',dict(score=fresh,generation=gen,previous_file=str(oldpath)))
                for k in sizes:
                    f=dest/f'K{k}.json'
                    if f.exists(): continue
                    selected=bank['heads'][:k]; randoms=[controls(selected,widths,(7000 if assay=='broad' else 6000)+i) for i in range(3)]
                    assert len(selected)==k and len(set(map(tuple,selected)))==k
                    arms=[dict(oldclean,reused_from=str(oldpath))]
                    expected=[('selected',selected)]+[(f'random_{i}',h) for i,h in enumerate(randoms)]
                    for name,heads in expected:
                        old=next((x for x in previous if x['name']==name and x['heads']==heads),None)
                        if old is not None and not a.canary:
                            assert bool(score(old['generation'])['correct'])==bool(old['score']['correct'])
                            arms.append(dict(old,reused_from=str(oldpath)))
                        else:
                            gen=generate(ids[:length],heads,assay=='targeted'); arms.append(dict(name=name,heads=heads,score=score(gen),generation=gen))
                    write(f,dict(case_id=p['case_id'],seed=p['seed'],mode=p['mode'],assay=assay,k=k,prefix_length=length,arms=arms,backend=backend(),protocol_sha256=sha(R/'protocol.json')))
                    print(json.dumps(dict(model=a.model,task=task,key=key,case_id=p['case_id'],k=k,time=time.time())),flush=True)
    write(out/'complete.json',dict(status='PASS',protocol_sha256=sha(R/'protocol.json')))

if __name__=='__main__': main()
