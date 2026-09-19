"""Full-span Native Broad: geometry, fresh discovery, frozen banks, and ablation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False),encoding='utf-8');tmp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--stage',choices=['geometry','discovery','canary','full'],required=True)
    a=ap.parse_args();root=a.root.resolve();start=time.perf_counter();cfg=read(root/'protocol.json');ph=sha(root/'protocol.json')
    for rel,h in cfg['files'].items():assert sha(root/rel)==h,rel
    assert a.model in cfg['models'] and cfg['record_scope']=='registered_generated_full_record_spans'
    sys.path[:0]=[str(root),str(root/'src')]
    from native_broad_full_span import full_span_geometry,score_masses,canary_cases
    from category_target_broad import rank_rows
    from local_selection import select_heads,random_control,control_audit
    from task_scoring import score_generation
    from protocol import encode_ids
    from realistic_niah_v4.spec import resolve_model_spec
    spec=resolve_model_spec(a.model)
    out=root/a.stage/a.model
    def inputs(p):
        source=Path(p['source'])
        for name,h in p['source_hashes'].items():assert sha(source/name)==h
        prompt,g=read(source/'prompt.json'),read(source/'generation.json')
        return prompt,g
    if a.stage=='geometry':
        from transformers import AutoTokenizer
        # Use the same pinned tokenizer definition as the registered loader.
        models={'Qwen3-8B':('Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218'),
                'Gemma4-E4B':('google/gemma-4-E4B-it','ee0ef6023621cff504d758262d4e04895a5af4a2')}
        mid,rev=models[a.model]
        tok=AutoTokenizer.from_pretrained(mid,revision=rev,cache_dir=cfg['cache'],local_files_only=True)
        for task in cfg['tasks']:
            plans=read(root/'plans'/f'{task}_{a.model}.json')
            for i,p in enumerate(plans,1):
                dest=out/task/f"{p['case_id']}.json"
                if dest.exists():
                    g=read(dest);assert g['protocol_sha256']==ph
                else:
                    prompt,generation=inputs(p)
                    geo=full_span_geometry(tok,p,prompt,generation,a.model)
                    write(dest,dict(case_id=p['case_id'],seed=p['seed'],split=p['split'],task=task,model=a.model,
                        source_hashes=p['source_hashes'],protocol_sha256=ph,geometry=geo))
                print(json.dumps(dict(stage=a.stage,model=a.model,task=task,completed=i,total=len(plans))),flush=True)
        write(out/'complete.json',dict(status='PASS',protocol_sha256=ph,elapsed_seconds=time.perf_counter()-start));return
    assert read(root/'geometry'/a.model/'complete.json')['status']=='PASS'
    if a.stage=='full':assert read(root/'canary_audit.json')['status']=='PASS'
    import torch
    from realistic_niah_v4.modeling import load_registered_model,query_attention_rows,generate_with_head_ablation,_text_config
    assert torch.cuda.is_available()
    torch.manual_seed(cfg['generation_seed'])
    model,tok,adapter=load_registered_model(spec,cache_dir=cfg['cache']);model.eval()
    widths=list(adapter.num_heads);assert widths==cfg['widths'][a.model]
    def backend():
        vals=[getattr(c,'_attn_implementation',None) for c in [model.config,_text_config(model)] if c is not None]
        assert vals and all(v=='sdpa' for v in vals),vals
        return vals
    write(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
        transformers=importlib.metadata.version('transformers'),gpu=torch.cuda.get_device_name(),widths=widths,
        backend=backend(),protocol_sha256=ph,command=sys.argv))
    def encoding(p):
        prompt,g=inputs(p);ids=prompt['input_ids']+g['generated_token_ids']
        assert len(prompt['input_ids'])<p['broad_prefix']<=len(ids)
        return encode_ids(ids[:p['broad_prefix']])
    def generate(enc,heads):
        backend();g=generate_with_head_ablation(model,tok,adapter,enc,heads,scope='answer_query',max_new_tokens=64);backend()
        return g
    for task in cfg['tasks']:
        plans=read(root/'plans'/f'{task}_{a.model}.json')
        geofiles={p['case_id']:root/'geometry'/a.model/task/f"{p['case_id']}.json" for p in plans}
        geometry={cid:read(path)['geometry'] for cid,path in geofiles.items()}
        bankfile=root/'banks'/f'{task}_{a.model}.json'
        if a.stage=='discovery':
            observations=[];panel=[p for p in plans if p['split']=='discovery']
            for i,p in enumerate(panel,1):
                dest=out/task/f"{p['case_id']}.json";geo=geometry[p['case_id']]
                if dest.exists():row=read(dest);assert row['protocol_sha256']==ph
                else:
                    heads=[];masses=[]
                    if geo['available']:
                        enc=encoding(p);backend();matrices,starts=query_attention_rows(model,adapter,enc);backend()
                        spans=[(r['token_start'],r['token_end']) for r in geo['records']]
                        for layer,(matrix,first) in enumerate(zip(matrices,starts)):
                            for head,alpha in enumerate(matrix):
                                values=[float(alpha[max(0,x-first):min(len(alpha),y-first)].sum()) if y>first else 0. for x,y in spans]
                                heads.append([layer,head,score_masses(values)]);masses.append([layer,head,values])
                    row=dict(case_id=p['case_id'],seed=p['seed'],split='discovery',task=task,model=a.model,
                        available=geo['available'],reason=geo['reason'],geometry_sha256=sha(geofiles[p['case_id']]),
                        heads=heads,record_masses=masses,backend=backend(),prefix_length=p['broad_prefix'],protocol_sha256=ph)
                    write(dest,row)
                if row['heads']:observations.append(row)
                print(json.dumps(dict(stage=a.stage,model=a.model,task=task,completed=i,total=200)),flush=True)
            assert sorted({r['seed'] for r in observations})==cfg['discovery_seeds']
            bank=dict(task=task,model=a.model,mode='native_thinking',assay='broad',record_scope=cfg['record_scope'],
                ranking=rank_rows(observations),sizes=cfg['sizes'][a.model],protocol_sha256=ph,conditions={},
                source_hashes={str(f.relative_to(root)):sha(f) for f in sorted((out/task).glob('*.json'))})
            for k in bank['sizes']:
                selected=select_heads(bank['ranking'],k);randoms=[random_control(selected,widths,s) for s in cfg['random_seeds']]
                bank['conditions'][str(k)]=dict(selected=selected,random=randoms,control_audit=control_audit(selected,widths,randoms))
            if bankfile.exists():assert read(bankfile)==bank
            else:write(bankfile,bank)
        else:
            bank=read(bankfile);assert bank['protocol_sha256']==ph
            panel=canary_cases(plans,geometry) if a.stage=='canary' else [p for p in plans if p['split']=='confirmation']
            sizes=sorted({bank['sizes'][0],bank['sizes'][-1]}) if a.stage=='canary' else bank['sizes']
            expected=sum(geometry[p['case_id']]['available'] for p in panel)*len(sizes);completed=0
            for p in panel:
                dest=out/task/p['case_id'];geo=geometry[p['case_id']]
                if not geo['available']:
                    write(dest/'unavailable.json',dict(reason=geo['reason'],protocol_sha256=ph));continue
                enc=encoding(p);cleanfile=dest/'clean.json'
                if cleanfile.exists():
                    saved=read(cleanfile);assert saved['protocol_sha256']==ph;clean=saved['arm']
                else:
                    gen=generate(enc,[]);clean=dict(name='clean',heads=[],generation=gen,score=score_generation(gen,p['case'],mode='native_thinking',assay='broad'))
                    write(cleanfile,dict(arm=clean,protocol_sha256=ph,prefix_length=enc.sequence_length))
                for k in sizes:
                    path=dest/f'K{k}.json';c=bank['conditions'][str(k)]
                    if path.exists():
                        x=read(path);assert x['protocol_sha256']==ph and x['bank_sha256']==sha(bankfile)
                    else:
                        point_start=time.perf_counter();arms=[clean]
                        for name,heads in [('selected',c['selected'])]+[(f'random_{i}',h) for i,h in enumerate(c['random'])]:
                            gen=generate(enc,heads)
                            arms.append(dict(name=name,heads=heads,generation=gen,score=score_generation(gen,p['case'],mode='native_thinking',assay='broad')))
                        write(path,dict(task=task,model=a.model,case_id=p['case_id'],seed=p['seed'],k=k,arms=arms,
                            prefix_length=enc.sequence_length,max_new_tokens=64,backend=backend(),
                            bank_sha256=sha(bankfile),geometry_sha256=sha(geofiles[p['case_id']]),protocol_sha256=ph,
                            elapsed_seconds=time.perf_counter()-point_start))
                    completed+=1
                    print(json.dumps(dict(stage=a.stage,model=a.model,task=task,completed=completed,total=expected,k=k)),flush=True)
            assert completed==expected
    write(out/'complete.json',dict(status='PASS',protocol_sha256=ph,stage=a.stage,model=a.model,elapsed_seconds=time.perf_counter()-start))


if __name__=='__main__':main()
