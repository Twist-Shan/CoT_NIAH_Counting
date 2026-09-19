"""Compare exact-prefix execution paths without changing any Read population."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+'\n')
    temporary.replace(path)


def freeze(root, stage):
    import shutil
    stage.mkdir(parents=True, exist_ok=False)
    source = root/'fresh_v1/baseline/Gemma4-E4B/enumeration_bullet/generations.jsonl'
    code = root/'fresh_causal_v1/code'
    inputs = {str(source): sha(source), str(root/'fresh_causal_v1/code_manifest.json'): sha(root/'fresh_causal_v1/code_manifest.json')}
    for relative in ['src/realistic_niah_v4/modeling.py', 'scripts/enumeration_fresh_geometry.py']:
        inputs[str(code/relative)] = sha(code/relative)
    shutil.copy2(__file__, stage/Path(__file__).name)
    cfg = dict(root=str(root), source=str(source), code=str(code), input_sha256=inputs,
        cases=[[2026091621,9],[2026091625,7],[2026091625,9],
               [2026091621,8],[2026091625,6],[2026091625,10]],
        cache_dir=str(root.parents[1]/'cache/huggingface'), script_sha256=sha(__file__),
        output_tokens=16, variants=['bf16_original', 'bf16_full', 'bf16_cached_trace',
            'bf16_chunk512', 'fp32_full', 'fp32_cached_trace'],
        utc=datetime.now(timezone.utc).isoformat(), selection='Three previously observed errors and three adjacent-count controls; diagnostic only',
        no_population_change=True)
    write(stage/'manifest.json', cfg)
    print(json.dumps(dict(status='FROZEN',manifest_sha256=sha(stage/'manifest.json'))),flush=True)


def run(stage):
    tick=time.monotonic(); cfg=read(stage/'manifest.json')
    assert sha(__file__)==cfg['script_sha256']
    for p,h in cfg['input_sha256'].items(): assert sha(p)==h,p
    sys.path[:0]=[cfg['code']+'/src',cfg['code']]
    import torch
    from realistic_niah_v4 import modeling
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah.parsing import parse_total
    from scripts.enumeration_fresh_geometry import build_read_geometry
    state=dict(status='RUNNING',phase='LOAD_BF16',pid=os.getpid(),completed=[])
    def save():
        state['seconds']=time.monotonic()-tick; write(stage/'status.json',state)
    save()
    rows={(r['seed'],r['gold_count']):r for r in map(json.loads,Path(cfg['source']).read_text().splitlines())}
    try:
        model, tokenizer, adapter=modeling.load_registered_model(resolve_model_spec('Gemma4-E4B'),cache_dir=Path(cfg['cache_dir']),device_map='auto',torch_dtype='bfloat16',attention_backend='sdpa')
        model.eval(); device=model.get_input_embeddings().weight.device
        runtime=dict(torch=torch.__version__,transformers=importlib.metadata.version('transformers'),
            model_source_sha256=sha(sys.modules[type(model).__module__].__file__),gpu=torch.cuda.get_device_name(),
            model_revision=resolve_model_spec('Gemma4-E4B').revision,backend='sdpa',python=sys.version)
        write(stage/'runtime.json',runtime)
        eos=model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos=set(eos if isinstance(eos,(tuple,list)) else [eos])
        cases=[]
        for key in cfg['cases']:
            row=rows[tuple(key)]; encoding,_,geometry=build_read_geometry(row,tokenizer,mode='enumeration_bullet')
            ids=list(encoding.input_ids); base=list(row['input_ids']); prefix=ids[len(base):]
            assert ids==base+row['output_token_ids'][:len(prefix)]
            assert geometry['selected_site']['alignment_strategy']=='literal_baseline_token_prefix'
            cases.append((row,encoding,geometry,ids,base,prefix))
        def tensor(ids): return torch.tensor([ids],dtype=torch.long,device=device)
        def top(logits):
            values,indices=torch.topk(logits.reshape(-1).float(),10)
            return [dict(token_id=i,text=tokenizer.decode([i]),logit=v) for i,v in zip(indices.cpu().tolist(),values.cpu().tolist())]
        def call(ids,end,cache=None):
            return model(input_ids=tensor(ids), attention_mask=torch.ones((1,end),dtype=torch.long,device=device),
                past_key_values=cache,use_cache=True,**modeling._bounded_logits_kwargs(model))
        def finish(output,total):
            generated=[]; first=top(output.logits[:,-1,:]); cache=output.past_key_values
            logits=output.logits[:,-1,:]
            for i in range(cfg['output_tokens']):
                token=int(logits.argmax(dim=-1).item()); generated.append(token)
                if token in eos: break
                output=call([token],total+len(generated),cache); cache=output.past_key_values; logits=output.logits[:,-1,:]
            text=tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            return dict(generated_token_ids=generated,completion_text=text,prediction=parse_total('Total:'+text),
                        first_top10=first,stopped_on_eos=generated[-1] in eos)
        for precision in ['bf16','fp32']:
            if precision=='fp32':
                gc.collect(); torch.cuda.empty_cache(); model=model.to(dtype=torch.float32); model.eval()
            state['phase']=precision; save()
            with torch.inference_mode():
                for row,encoding,geometry,ids,base,prefix in cases:
                    case_tick=time.monotonic(); result=dict(seed=row['seed'],gold_count=row['gold_count'],precision=precision,
                        prefix_exact=True,prompt_tokens=len(base),trace_prefix_tokens=len(prefix),query_position=encoding.query_position,
                        input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),geometry=geometry,paths={})
                    if precision=='bf16':
                        # Preserve original execution and compare every generated token.
                        original=model.generate(input_ids=tensor(base),attention_mask=tensor(row['attention_mask']),
                            do_sample=False,max_new_tokens=4096,use_cache=True,pad_token_id=tokenizer.pad_token_id)
                        sequence=original[0,len(base):].cpu().tolist(); del original
                        result['original']=dict(generated_token_ids=sequence,matches_saved=sequence==row['output_token_ids'],
                            text=tokenizer.decode(sequence,skip_special_tokens=False,clean_up_tokenization_spaces=False))
                        result['original']['prediction']=parse_total(result['original']['text'])
                        reference=modeling.generate_answer_completion(model,tokenizer,encoding,max_new_tokens=16)
                        result['reference_generate']=reference
                    out=call(ids,len(ids)); result['paths']['full']=finish(out,len(ids)); del out
                    out=call(base,len(base))
                    for offset,token in enumerate(prefix,1): out=call([token],len(base)+offset,out.past_key_values)
                    result['paths']['cached_trace']=finish(out,len(ids)); del out
                    if precision=='bf16':
                        out=None
                        for offset in range(0,len(ids),512):
                            end=min(offset+512,len(ids)); out=call(ids[offset:end],end,None if out is None else out.past_key_values)
                        result['paths']['chunk512']=finish(out,len(ids)); del out
                        result['full_matches_generate']=result['paths']['full']['generated_token_ids']==reference['generated_token_ids']
                        if not result['full_matches_generate']:
                            raise RuntimeError('Manual full-prefill decoder differs from official generate; inspect saved partial result')
                    result['seconds']=time.monotonic()-case_tick
                    filename=f'{precision}_seed{row["seed"]}_N{row["gold_count"]}.json'
                    write(stage/filename,result); state['completed'].append(filename); save()
                    print(json.dumps(dict(file=filename,predictions={k:v['prediction'] for k,v in result['paths'].items()},seconds=result['seconds'])),flush=True)
                    gc.collect(); torch.cuda.empty_cache()
        state.update(status='COMPLETE',phase='PATH_DIAGNOSTIC_COMPLETE_ANALYSIS_PENDING')
    except BaseException as e:
        state.update(status='FAILED',error=repr(e)); raise
    finally: save()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['freeze','run']); p.add_argument('--stage',type=Path,required=True);p.add_argument('--root',type=Path)
    a=p.parse_args(); freeze(a.root,a.stage) if a.action=='freeze' else run(a.stage)
