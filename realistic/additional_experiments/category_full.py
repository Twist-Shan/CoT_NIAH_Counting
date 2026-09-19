"""300 mixed category inputs; natural generation before trace parser development."""
import argparse
import json
from pathlib import Path
import sys
import time
from category_trial import make_pair, prompt
from protocol import read_jsonl, write_json, sha256, text_hash, encode_ids
ROOT=Path(__file__).resolve().parent

def freeze(source, out):
    t=time.perf_counter()
    rows={r['seed']:r for r in read_jsonl(source) if r['design_variant']=='v4.4' and r['gold_count']==10 and 1234<=r['seed']<=1263}
    assert set(rows)==set(range(1234,1264))
    cases=[]
    for seed in sorted(rows):
        for n in (1,3,5,7,9):
            pair=list(make_pair(rows[seed],n))
            assert pair[0]['passage']==pair[1]['passage']
            assert int(pair[0]['gold'])+int(pair[1]['gold'])==10
            for c in pair:
                c['split']='parser_development' if seed<1254 else 'parser_audit'
                assert len(c['records'])==10
                for r in c['records']: assert c['passage'][r['char_start']:r['char_end']]==r['text']
                cases.append(c)
    assert len(cases)==300 and len({c['case_id'] for c in cases})==300
    out.mkdir(parents=True,exist_ok=False)
    prompts=[{'case_id':c['case_id'],'mode':m,'user_text':prompt(c,m),'user_text_sha256':text_hash(prompt(c,m))} for c in cases for m in ('nonthinking','native_thinking')]
    for name,data in [('cases.jsonl',cases),('user_prompts.jsonl',prompts)]:
        with (out/name).open('w',encoding='utf-8') as f:
            for row in data: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json(out/'manifest.json',{'status':'PASS','source_sha256':sha256(source),'cases':300,'generations_per_model':600,'total_generations':1200,'city_counts':[1,3,5,7,9],'seeds':list(rows),'files':{n:sha256(out/n) for n in ('cases.jsonl','user_prompts.jsonl')},'elapsed_seconds':time.perf_counter()-t,'max_tokens':{'nonthinking':64,'native_thinking':4096},'trace_parser_applied':False})
    print('FREEZE_PASS: 300 paired inputs, 600 exact prompts, 1200 model generations')

def run(a):
    import torch
    sys.path.insert(0,str(ROOT.parent/'src'))
    from realistic_niah_v4.modeling import load_registered_model, generate_answer_completion
    from realistic_niah_v4.spec import resolve_model_spec
    from run import summarize_generation
    manifest=json.loads((a.frozen/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['files'].items(): assert sha256(a.frozen/name)==h
    out=a.output/a.model; out.mkdir(parents=True,exist_ok=True)
    contract={'model':a.model,'manifest_sha256':sha256(a.frozen/'manifest.json'),'code_hashes':{p.name:sha256(p) for p in ROOT.glob('*.py')},'src_hashes':{str(p.relative_to(ROOT.parent/'src')):sha256(p) for p in (ROOT.parent/'src').rglob('*.py')},'trace_parser_applied':False}
    if (out/'contract.json').exists(): assert json.loads((out/'contract.json').read_text())==contract
    else: write_json(out/'contract.json',contract)
    t=time.perf_counter(); torch.manual_seed(20260906)
    spec=resolve_model_spec(a.model)
    model,tok,_=load_registered_model(spec,cache_dir=a.cache_dir); tok.padding_side='left'
    write_json(out/'environment.json',{'model_id':spec.model_id,'revision':spec.revision,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':__import__('transformers').__version__,'load_seconds':time.perf_counter()-t})
    cases={c['case_id']:c for c in read_jsonl(a.frozen/'cases.jsonl')}
    for row in read_jsonl(a.frozen/'user_prompts.jsonl'):
        c=cases[row['case_id']]; mode=row['mode']; dest=out/'captures'/mode/c['case_id']
        if (dest/'complete.json').exists():
            saved=json.loads((dest/'complete.json').read_text())
            for name,h in saved['files'].items(): assert sha256(dest/name)==h
            continue
        started=time.perf_counter()
        rendered=tok.apply_chat_template([{'role':'user','content':row['user_text']}],tokenize=False,add_generation_prompt=True,enable_thinking=mode=='native_thinking')
        if mode=='nonthinking': rendered+='Total:'
        ids=tok(rendered,add_special_tokens=False)['input_ids']
        gen=generate_answer_completion(model,tok,encode_ids(ids),max_new_tokens=manifest['max_tokens'][mode])
        gen.update(summarize_generation(gen,c,mode=mode,prefixed=mode=='nonthinking'))
        write_json(dest/'prompt.json',{**row,'rendered_prompt':rendered,'input_ids':ids})
        write_json(dest/'generation.json',gen)
        write_json(dest/'complete.json',{'status':'PASS','files':{n:sha256(dest/n) for n in ('prompt.json','generation.json')},'elapsed_seconds':time.perf_counter()-started})
        print('capture',a.model,mode,c['case_id'],gen['correct'],flush=True)
    assert len(list((out/'captures').glob('*/*/complete.json')))==600
    write_json(out/'complete.json',{'status':'PASS','captures':600,'elapsed_seconds':time.perf_counter()-t})

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__); p.add_argument('--freeze-source',type=Path); p.add_argument('--frozen',type=Path,required=True); p.add_argument('--output',type=Path); p.add_argument('--model'); p.add_argument('--cache-dir',type=Path)
    a=p.parse_args()
    if a.freeze_source: freeze(a.freeze_source,a.frozen)
    else: run(a)
