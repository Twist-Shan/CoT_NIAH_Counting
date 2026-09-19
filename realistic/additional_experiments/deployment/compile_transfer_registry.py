"""Read-only full-panel event/site audit. No model inference or result overwrite."""
import argparse, json, re, sys, hashlib, time
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'src'), str(HERE.parents[1]/'src')]
from realistic_niah_v5.parsing import parse_hybrid_trace

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def token_boundaries(tok, raw, ids):
    enc=tok(raw,add_special_tokens=False,return_offsets_mapping=True)
    exact=list(enc['input_ids'])==ids
    if tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=raw:
        return {},False,'original_decode_mismatch'
    bounds={0:0}
    if exact:
        offsets=enc['offset_mapping']
        for j,(s,e) in enumerate(offsets):
            # Overlapping byte-token offsets must finish the whole character.
            if e>s and (j+1==len(offsets) or offsets[j+1][0]>=e): bounds[e]=j+1
        return bounds,True,'exact_output_reencode'
    # Recovery never substitutes token IDs. Only stable decoded original
    # prefixes become boundaries; incomplete Unicode prefixes are unavailable.
    for j in range(1,len(ids)+1):
        prefix=tok.decode(ids[:j],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if raw.startswith(prefix): bounds[len(prefix)]=j
    return bounds,False,'verified_original_prefix_decode'

def compile_one(c,p,g,tok,model,mode):
    raw=g['completion_text_raw']; ids=g['generated_token_ids']; promptids=p.get('input_ids')
    rendered=p['rendered_prompt']
    encoded_prompt=tok(rendered,add_special_tokens=False)['input_ids']
    if promptids is None: promptids=encoded_prompt
    assert promptids==encoded_prompt,'prompt token mismatch'
    bounds,exact,method=token_boundaries(tok,raw,ids)
    n=len(promptids)
    def endsite(end):
        if end in bounds: return dict(query_position=n+bounds[end]-1,prefix_length=n+bounds[end],char_end=end,alignment='exact_original_boundary')
        return dict(unavailable_reason='no_exact_original_token_boundary',char_end=end)
    def presite(start):
        if start in bounds: return dict(query_position=n+bounds[start]-1,prefix_length=n+bounds[start],char_start=start,alignment='exact_original_boundary')
        # A tokenizer may bind preceding whitespace to the entity token.
        eligible=[z for z in bounds if z<start and raw[z:start].isspace()]
        if eligible:
            z=max(eligible); return dict(query_position=n+bounds[z]-1,prefix_length=n+bounds[z],char_start=start,alignment='leading_whitespace_only')
        return dict(unavailable_reason='entity_start_fused_or_unmapped',char_start=start)
    result=dict(model=model,task=c['task'],case_id=c['case_id'],seed=c['seed'],split=c['split'],mode=mode,
        output_exact_reencode=exact,token_map_method=method,prompt_ids_verified=True,events=[],identities=[],cohort='nonthinking',
        original_output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    if mode=='nonthinking':
        result['answer_query']=dict(prefix_length=n,query_position=n-1,alignment='original_prompt')
        return result
    closematch=re.search(r'</think>|<channel\|>',raw)
    close=closematch.start() if closematch else len(raw)
    result['reasoning_closed']=bool(closematch)
    lookup={r['city'].casefold():r for r in c['records']}
    cut,legacy=parse_hybrid_trace(raw,model_family='qwen3' if model.startswith('Qwen') else 'gemma4',gold_records=c['records'])
    result['legacy_parser']=cut.to_dict(); result['legacy_sequence_source']=legacy['sequence_source']
    mentions=[]
    for r in c['records']:
        for m in re.finditer(r'(?<!\w)'+re.escape(r['city'])+r'(?!\w)',raw[:close],re.I):
            mentions.append((m.start(),m.end(),r))
    mentions.sort(key=lambda x:x[0]); seen=Counter()
    score_pattern=re.compile(r'^\s*(?:\*\*\s*)?(?:received\s+(?:a\s+(?:numeric\s+)?score\s+of\s+)?|with\s+(?:a\s+score\s+of\s+)?|[-–—:|]\s*|\(\s*|\s+)(\d+)\b',re.I)
    for i,(a,b,r) in enumerate(mentions):
        result['identities'].append(dict(entity=r['city'],source_ordinal=r['ordinal'],span=[a,b],pre_city=presite(a)))
        limit=mentions[i+1][0] if i+1<len(mentions) else close
        tail=raw[b:limit]; m=score_pattern.search(tail)
        if not m: continue
        z=b+m.end(); seen[r['ordinal']]+=1
        result['events'].append(dict(entity=r['city'],source_ordinal=r['ordinal'],span=[a,z],text=raw[a:z],
            observed_score=int(m[1]),score_matches_source=int(m[1])==r['score'],visit_number=seen[r['ordinal']],
            pre_city=presite(a),record_end=endsite(z)))
    # Keep original grammar-selected sites in a separate registry. Do not
    # label bare identity/ordinal evidence as a score-supported record.
    result['legacy_item_sites']=[dict(entity=name,span=[a,b],item_end=endsite(b)) for name,a,b in zip(cut.item_gold_cities,cut.item_start_chars,cut.item_end_chars)]
    result['cohort']=('score_evidence_with_repeats' if any(v>1 for v in seen.values()) else 'score_evidence_unique') if seen else ('legacy_structure_only' if cut.detected else 'identity_only' if mentions else 'no_observed_record')
    result['trace_record_sites']=len({e['record_end'].get('query_position') for e in result['events'] if 'query_position' in e['record_end']})
    if c['task']=='kth_needle':
        target=c['records'][c['level']-1]['ordinal']
        hits=[e for e in result['identities'] if e['source_ordinal']==target]
        result['target_query']=hits[0]['pre_city'] if hits else dict(unavailable_reason='target_not_mentioned')
    else:
        result['targeted_event_sites']=sum('query_position' in e['pre_city'] for e in result['events'])
    tag='Needle' if c['task']=='kth_needle' else 'Total'
    # Match the literal tag after the channel boundary, permitting markdown
    # wrappers. Do not substitute a reasoning count or an inferred answer.
    matches=list(re.finditer(r'\b'+tag+r'\s*:',raw[closematch.end():],re.I)) if closematch else []
    result['answer_query']=endsite(closematch.end()+matches[0].end()) if len(matches)==1 else dict(unavailable_reason='missing_or_ambiguous_final_tag')
    result['answer_unit_pre']=presite(closematch.end()+matches[0].start()) if len(matches)==1 else dict(unavailable_reason='missing_or_ambiguous_final_tag')
    return result

def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--cache',required=True); a=p.parse_args()
    a.output.mkdir(exist_ok=False); start=time.time(); summaries={}; failures=[]
    models=[('Qwen3-8B','Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218'),('Gemma4-E4B','google/gemma-4-E4B-it','ee0ef6023621cff504d758262d4e04895a5af4a2')]
    for model,mid,revision in models:
        tok=AutoTokenizer.from_pretrained(mid,revision=revision,cache_dir=a.cache,local_files_only=True)
        for task,folder in [('kth','kth_uniform_20260906_v2'),('category','category_full_20260906')]:
            root=a.base/folder; cases=[json.loads(x) for x in (root/'frozen/cases.jsonl').open(encoding='utf-8')]
            # kth pipeline stores full outputs in full; category in outputs.
            roots=[root/('full' if task=='kth' else 'outputs')/model/'captures']
            assert roots[0].is_dir(),str(roots[0])
            stats=Counter(); records=[]
            for mode in ['nonthinking','native_thinking']:
                for c in cases:
                    d=roots[0]/mode/c['case_id']
                    try:
                        for name,h in read(d/'complete.json')['files'].items():
                            if name in ['prompt.json','generation.json']: assert sha(d/name)==h
                        r=compile_one(c,read(d/'prompt.json'),read(d/'generation.json'),tok,model,mode)
                        stats[mode+'_compiled']+=1; stats[mode+'_exact_output_reencode']+=r['output_exact_reencode']
                        stats[mode+'_original_token_mapping']+=r['token_map_method']!='original_decode_mismatch'
                        stats[mode+'_answer_query']+='query_position' in r['answer_query']
                        if mode=='native_thinking':
                            stats['native_answer_unit_pre']+='query_position' in r['answer_unit_pre']
                            stats['native_with_record_sites']+=r['trace_record_sites']>0
                            stats['native_with_legacy_items']+=bool(r['legacy_item_sites'])
                            stats['native_target_query']+='query_position' in r.get('target_query',{}) if task=='kth' else r['targeted_event_sites']>0
                            stats['cohort_'+r['cohort']]+=1
                        records.append(r)
                    except Exception as e: failures.append(dict(model=model,task=task,mode=mode,case_id=c['case_id'],error=repr(e)))
            with (a.output/(task+'_'+model+'.jsonl')).open('w',encoding='utf-8') as f:
                for r in records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
            summaries[task+'_'+model]=dict(stats); write(a.output/'summary.json',summaries); write(a.output/'failures.json',failures)
            print(task,model,dict(stats),flush=True)
    write(a.output/'audit.json',dict(status='PASS' if not failures else 'FAIL',failures=len(failures),expected_trajectories=2400,elapsed_seconds=time.time()-start,script_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(HERE)):sha(p) for p in (HERE/'src').rglob('*.py')}))
if __name__=='__main__': main()
