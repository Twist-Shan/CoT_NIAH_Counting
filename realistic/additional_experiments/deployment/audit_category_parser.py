import argparse,json,hashlib,sys
from pathlib import Path
from collections import Counter
BASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BASE))
from category_trace_parser import parse_category_trace,CONTRACT
from trace_parser import parse_trace
p=argparse.ArgumentParser(); p.add_argument('--split',choices=['parser_development','parser_audit'],required=True); a=p.parse_args()
root=BASE/'runs/category_full_20260906'; out=root/'analysis'/CONTRACT; out.mkdir(exist_ok=True)
h=hashlib.sha256((BASE/'category_trace_parser.py').read_bytes()).hexdigest()
lock=out/'frozen.json'
if a.split=='parser_audit': assert json.loads(lock.read_text())['sha256']==h
cases=[c for c in map(json.loads,(root/'frozen/cases.jsonl').open(encoding='utf-8')) if c['split']==a.split]
summary={}
with (out/(a.split+'.jsonl')).open('w',encoding='utf-8') as f:
    for model in ['Qwen3-8B','Gemma4-E4B']:
        stats=Counter()
        for c in cases:
            g=json.loads((root/'downloaded/outputs'/model/'captures/native_thinking'/c['case_id']/'generation.json').read_text(encoding='utf-8'))
            raw=g['completion_text_raw']; result=parse_category_trace(c,raw); old=parse_trace(c,raw)
            stats['traces']+=1; stats['old_structured_traces']+=old['status']=='structured'
            stats['traces_with_record_evidence']+=bool(result['events'])
            stats['traces_with_explicit_decision']+=any(e['model_is_target'] is not None for e in result['events'])
            stats['traces_with_explicit_count']+=any(e['model_explicit_count'] is not None for e in result['events'])
            stats['events']+=len(result['events'])
            stats['unknown_decision_events']+=sum(e['decision_status']=='unknown' for e in result['events'])
            stats['conflicting_decision_events']+=sum(e['decision_status']=='conflict' for e in result['events'])
            stats['unknown_source_events']+=sum(e['source_ordinal'] is None for e in result['events'])
            if old.get('episode_span'):
                end=old['episode_span']['end']; oldids={i['source_ordinal'] for i in old['items']}
                stats['traces_with_new_record_after_first_list']+=any(e['record_end']['start']>=end and e['source_ordinal'] not in oldids for e in result['events'])
            for e in result['events']:
                for k in ['record_end','identity_end']:
                    s=e[k]; assert raw[s['start']:s['end']]==s['text']
            f.write(json.dumps(dict(model=model,case_id=c['case_id'],result=result),ensure_ascii=False)+'\n')
        summary[model]=dict(stats)
(out/(a.split+'_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
if a.split=='parser_development': lock.write_text(json.dumps(dict(contract=CONTRACT,sha256=h,scope='freeze before parser_audit',manual_precision_recall_validated=False)),encoding='utf-8')
print(json.dumps(summary,indent=2))
