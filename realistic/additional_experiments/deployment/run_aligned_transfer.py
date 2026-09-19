"""Frozen additional-task retrieval protocol, aligned endpoint/ablation semantics."""
import argparse,json,hashlib,sys,time,re,math,random
from pathlib import Path
from collections import Counter,defaultdict
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'src'),str(HERE/'helpers')]
from compile_transfer_registry_v3 import token_boundaries
from realistic_niah_v5.parsing import parse_trace_record
from realistic_niah_v5.causal_sites import _rank_event_rows,_structural_event_rows

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,ensure_ascii=False),encoding='utf-8'); tmp.replace(p)
def readlines(p): return [json.loads(x) for x in p.open(encoding='utf-8')]
def source(base,task,model,cid,mode): return base/('kth_uniform_20260906_v2' if task=='kth' else 'category_full_20260906')/('full' if task=='kth' else 'outputs')/model/'captures'/mode/cid

def controls(heads,widths,seed):
    rng=random.Random(seed); selected=set(map(tuple,heads)); out=[]
    for l,n in sorted(Counter(l for l,h in heads).items()):
        pool=[(l,h) for h in range(widths[l]) if (l,h) not in selected]
        if len(pool)<n: raise ValueError('No disjoint layer-matched control')
        out+=rng.sample(pool,n)
    return out

def prepare(a):
    from transformers import AutoTokenizer
    frozen=a.output/'plans'; frozen.mkdir(parents=True,exist_ok=False)
    summaries={}
    for model,mid,rev in MODELS:
        tok=AutoTokenizer.from_pretrained(mid,revision=rev,cache_dir=a.cache,local_files_only=True)
        for task in ['kth','category']:
            root=a.base/('kth_uniform_20260906_v2' if task=='kth' else 'category_full_20260906')
            cases=readlines(root/'frozen/cases.jsonl'); reg={ (r['case_id'],r['mode']):r for r in readlines(a.registry/(task+'_'+model+'.jsonl'))}; plans=[]; stats=Counter()
            for c in cases:
                for mode in ['nonthinking','native_thinking']:
                    d=source(a.base,task,model,c['case_id'],mode); p=read(d/'prompt.json'); g=read(d/'generation.json'); r=reg[c['case_id'],mode]
                    item=dict(case_id=c['case_id'],seed=c['seed'],split='discovery' if c['seed']<1254 else 'confirmation',mode=mode,source=str(d),case=c,broad_prefix=r['answer_query'].get('prefix_length'),trace_positions=[],target=None,unavailable={})
                    if not item['broad_prefix']: item['unavailable']['broad']=r['answer_query'].get('unavailable_reason')
                    if mode=='native_thinking':
                        raw=g['completion_text_raw']; n=len(p['input_ids']); bounds,_,_=token_boundaries(tok,raw,g['generated_token_ids'])
                        parsed=parse_trace_record(dict(g,model_label=model,model_family='qwen3' if model.startswith('Qwen') else 'gemma4',gold_records=c['records']))
                        cut=parsed['parser']; item['sequence_source']=parsed['sequence_source']
                        rows=(_rank_event_rows if parsed['sequence_source']=='rank_supported_episode' else _structural_event_rows)(parsed=parsed,parser=cut)
                        def endpoint(z):
                            # Minimal complete token covering the literal item end.
                            after=sorted(k for k in bounds if k>=z)
                            if not after: return None
                            k=after[0]
                            if raw[z:k].strip(' \t\n\r.,;:!?\"\'`*()[]'): return None
                            return n+bounds[k]
                        item['trace_positions']=sorted({endpoint(e['semantic_end_char'])-1 for e in rows if endpoint(e['semantic_end_char']) is not None})
                        candidates=[]
                        for i in range(1,len(rows)):
                            prev,e=rows[i-1],rows[i]; role='p0_item_end'; z=prev['semantic_end_char']
                            marker=e.get('rank_evidence_end_char'); start=e.get('rank_evidence_start_char')
                            if model.startswith('Qwen') and marker is not None and marker<=e['city_start_char'] and start is not None:
                                role='post_marker'; z=marker
                            prefix=endpoint(z)
                            if prefix is None: continue
                            decoded=tok.decode(g['generated_token_ids'][:prefix-n],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                            if not raw.startswith(decoded) or len(decoded)>e['city_start_char']: continue
                            candidates.append(dict(prefix_length=prefix,role=role,target_city=e['city'],occurrence=i+1,query_char_end=len(decoded),target_city_start=e['city_start_char']))
                        # One deterministic anchor per prompt, independent of correctness.
                        item['target']=candidates[(len(candidates)-1)//2] if candidates else None
                        item['target_candidates']=len(candidates)
                        if not candidates: item['unavailable']['targeted']='no_registered_next_record_transition'
                        if not item['trace_positions']: item['broad_prefix']=None; item['unavailable']['broad']='no_registered_item_endpoint'
                    item['source_hashes']={name:sha(d/name) for name in ['prompt.json','generation.json']}
                    plans.append(item); stats[mode+'_broad']+=bool(item['broad_prefix']); stats[mode+'_targeted']+=bool(item['target'])
            write(frozen/(task+'_'+model+'.json'),plans); summaries[task+'_'+model]=dict(stats)
    banks=read(a.base/'kth_uniform_20260906_v2/frozen/targeted_banks.json'); write(frozen/'targeted_banks.json',banks)
    write(frozen/'summary.json',summaries)
    write(frozen/'manifest.json',dict(files={p.name:sha(p) for p in frozen.glob('*.json')},protocol='aligned_transfer_v1',target_scope='query_and_future_decode_only',broad_scope='one_answer_query',native_broad_keys='legacy_registered_item_end_tokens',selection='discovery_seed_equal',broad_K={'Qwen3-8B':{'nonthinking':128,'native_thinking':32},'Gemma4-E4B':{'nonthinking':6,'native_thinking':6}},target_anchor='middle eligible transition of registered sequence; Qwen rank-before post-marker, otherwise P0; Gemma P0'))
    print('PREPARE_COMPLETE',summaries,flush=True)

def run(a):
    import torch
    from protocol import encode_ids
    from run import summarize_generation
    from realistic_niah_v4.modeling import load_registered_model,query_attention_rows,generate_with_head_ablation,generate_answer_completion,_is_prompt_prefill
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v5.causal import _first_generated_city_record
    frozen=a.output/'plans'; manifest=read(frozen/'manifest.json')
    for name,h in manifest['files'].items(): assert sha(frozen/name)==h
    out=a.output/('canary' if a.canary else 'full')/a.model; out.mkdir(parents=True,exist_ok=True)
    contract=dict(manifest=sha(frozen/'manifest.json'),code={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py')})
    if (out/'contract.json').exists(): assert read(out/'contract.json')==contract
    else: write(out/'contract.json',contract)
    torch.manual_seed(20260906); model,tok,adapter=load_registered_model(resolve_model_spec(a.model),cache_dir=a.cache); widths=list(adapter.num_heads)
    targetbank=read(frozen/'targeted_banks.json')['banks'][a.model]; targetcontrols=[controls(targetbank,widths,6000+i) for i in range(3)]
    def generate(prefix,heads,persistent):
        enc=encode_ids(prefix)
        if not persistent: return generate_with_head_ablation(model,tok,adapter,enc,heads,scope='answer_query',max_new_tokens=64)
        if not heads: return generate_answer_completion(model,tok,enc,max_new_tokens=128)
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
        assert all(calls[l] for l in groups); result['hook_calls']=dict(calls); return result
    for task in ['kth','category']:
        plans=read(frozen/(task+'_'+a.model+'.json'))
        if a.canary:
            chosen=[]
            for seed in [1234,1254]:
                eligible=[p for p in plans if p['seed']==seed and p['mode']=='native_thinking' and p['target'] and p['broad_prefix']]
                assert eligible,'Canary must exercise Broad and Targeted'
                cid=eligible[0]['case_id']; chosen += [p for p in plans if p['case_id']==cid]
            plans=chosen
        attention=[]
        for p in plans:
            f=out/task/'attention'/(p['mode']+'_'+p['case_id']+'.json')
            if f.exists(): attention.append(read(f)); continue
            d=Path(p['source'])
            for name,h in p['source_hashes'].items(): assert sha(d/name)==h
            prompt=read(d/'prompt.json'); g=read(d/'generation.json'); ids=prompt['input_ids']+g['generated_token_ids']
            rows=[]
            if p['broad_prefix']:
                enc=encode_ids(ids[:p['broad_prefix']]); matrix,starts=query_attention_rows(model,adapter,enc)
                if p['mode']=='native_thinking': spans=[(z,z+1) for z in p['trace_positions'] if z<p['broad_prefix']]
                else:
                    offsets=tok(prompt['rendered_prompt'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']; shift=prompt['rendered_prompt'].index(p['case']['passage']); spans=[]
                    for r in p['case']['records']:
                        ix=[i for i,(s,e) in enumerate(offsets) if e>s and s<shift+r['char_end'] and e>shift+r['char_start']]; spans.append((ix[0],ix[-1]+1))
                for l,(mat,start) in enumerate(zip(matrix,starts)):
                    for h,alpha in enumerate(mat):
                        masses=[float(alpha[max(0,x-start):min(len(alpha),y-start)].sum()) if y>start else 0. for x,y in spans]; total=sum(masses); probs=[v/total for v in masses] if total else []; score=total*math.exp(-sum(v*math.log(v) for v in probs if v>0))/len(spans) if spans else 0.
                        rows.append([l,h,score])
            data=dict(case_id=p['case_id'],mode=p['mode'],seed=p['seed'],split=p['split'],heads=rows); write(f,data); attention.append(data)
            print('attention',a.model,task,p['mode'],p['case_id'],flush=True)
        banks={}
        for mode in ['nonthinking','native_thinking']:
            groups=defaultdict(lambda:defaultdict(list))
            for r in attention:
                if r['split']=='discovery' and r['mode']==mode:
                    for l,h,s in r['heads']: groups[l,h][r['seed']].append(s)
            scores={h:sum(sum(v)/len(v) for v in ss.values())/len(ss) for h,ss in groups.items()}; selected=[]; used=Counter(); k=manifest['broad_K'][a.model][mode]
            for l,h in sorted(scores,key=lambda h:(-scores[h],h)):
                if used[l]<widths[l]//2: selected.append((l,h)); used[l]+=1
                if len(selected)==k: break
            assert len(selected)==k; banks[mode]=dict(heads=selected,controls=[controls(selected,widths,7000+i) for i in range(3)])
        write(out/task/'selection.json',banks)
        for p in plans:
            if p['split']!='confirmation': continue
            f=out/task/'causal'/(p['mode']+'_'+p['case_id']+'.json')
            if f.exists(): continue
            prompt=read(Path(p['source'])/'prompt.json'); g=read(Path(p['source'])/'generation.json'); ids=prompt['input_ids']+g['generated_token_ids']; assays={}
            for assay in ['broad','targeted']:
                if assay=='targeted' and p['mode']=='nonthinking': continue
                length=p['broad_prefix'] if assay=='broad' else (p['target'] or {}).get('prefix_length')
                if not length: assays[assay]=dict(status='unavailable',reason=p['unavailable'].get(assay)); continue
                bank=banks[p['mode']]['heads'] if assay=='broad' else targetbank; randoms=banks[p['mode']]['controls'] if assay=='broad' else targetcontrols; arms=[]
                for name,heads in [('clean',[]),('selected',bank)]+[(f'random_{i}',v) for i,v in enumerate(randoms)]:
                    gen=generate(ids[:length],heads,assay=='targeted')
                    if assay=='broad': score=summarize_generation(gen,p['case'],mode=p['mode'],prefixed=True)
                    else:
                        raw=gen['completion_text_raw']; pred,start,kind=_first_generated_city_record(raw,[r['city'] for r in p['case']['records']]); hits=[] if pred is None else [(start,pred)]
                        for m in re.finditer(r'(?:city|flower) score audit,\s+([^\n,.]+?) received a score',raw,re.I): hits.append((m.start(1),m[1]))
                        prediction=min(hits)[1] if hits else None; score=dict(correct=prediction is not None and prediction.casefold()==p['target']['target_city'].casefold(),prediction=prediction,target=p['target']['target_city'],policy='first_semantic_record_with_city_flower_audit_extension')
                    arms.append(dict(name=name,heads=heads,score=score,generation=gen))
                assays[assay]=dict(status='PASS',prefix_length=length,arms=arms)
            write(f,dict(case_id=p['case_id'],mode=p['mode'],seed=p['seed'],assays=assays)); print('causal',a.model,task,p['case_id'],p['mode'],flush=True)
    write(out/'complete.json',dict(status='PASS',canary=a.canary))

MODELS=[('Qwen3-8B','Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218'),('Gemma4-E4B','google/gemma-4-E4B-it','ee0ef6023621cff504d758262d4e04895a5af4a2')]
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--registry',type=Path); p.add_argument('--cache',required=True); p.add_argument('--model'); p.add_argument('--prepare',action='store_true'); p.add_argument('--canary',action='store_true'); a=p.parse_args()
    prepare(a) if a.prepare else run(a)
