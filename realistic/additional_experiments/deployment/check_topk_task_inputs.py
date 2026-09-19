import json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
B=Path(__file__).resolve().parents[1];R=B/'runs/topk_completion_20260907_v1'
checks=Counter();reference={}
for f in (R/'downloaded/plans').glob('*.json'):
    plans=json.loads(f.read_text(encoding='utf-8'));groups=defaultdict(list)
    for p in plans:
        c=p['case'];rs=c['records'];assert len(rs)==c['total_records']==10
        assert hashlib.sha256(c['passage'].encode()).hexdigest()==c['passage_sha256']
        assert [r['ordinal'] for r in rs]==list(range(1,11))
        assert [r['char_start'] for r in rs]==sorted(r['char_start'] for r in rs)
        for r in rs:assert c['passage'][r['char_start']:r['char_end']]==r['text']
        if c['task']=='kth_needle':
            rec=rs[c['level']-1];assert c['gold']==f"{rec['city']}|{rec['score']}"
            label=c['level']
        else:
            n=sum(r['category']==c['target_category'] for r in rs);assert str(n)==c['gold']
            label=(sum(r['category']=='city' for r in rs),c['target_category'])
            assert all(r['is_target']==(r['category']==c['target_category']) for r in rs)
        groups[p['mode'],p['seed']].append(label)
        identity=(c['task'],c['case_id']);packed=json.dumps(c,sort_keys=True)
        if identity in reference:assert reference[identity]==packed
        else:reference[identity]=packed
        checks['input_mode_model_rows']+=1
    expected=list(range(1,11)) if 'kth_' in f.name else [(n,t) for n in [1,3,5,7,9] for t in ['city','flower']]
    assert len(groups)==60
    for labels in groups.values():assert sorted(labels)==sorted(expected)
checks['unique_task_inputs']=len(reference)
result=dict(status='PASS',checks=dict(checks),verified=['k uniform1..10 per seed','city count1,3,5,7,9 x target city/flower per seed','same task inputs across models and modes','10 records per input','passage hashes and exact inserted record spans','gold answers recomputed from records'])
(R/'analysis/task_input_recheck.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
