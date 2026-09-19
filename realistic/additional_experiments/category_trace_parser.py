"""Offline mixed-category record evidence, v1. No stopping or gold-based decisions.

Spans are half-open offsets into completion_text_raw. Events represent textual
record mentions, not necessarily independent retrieval operations.
"""
import re
from collections import Counter

CONTRACT = 'category_record_evidence_v1'
FULL = re.compile(r'In the 2024 (city|flower)[ -]score audit,\s+([^\n,.!?]+?)\s+received a score of\s+(\d+)\b', re.I)
SHORT_SCORE = r'\s*(?:received\s+(?:a\s+score\s+of\s+)?|with\s+(?:a\s+score\s+of\s+)?|[-–—:]\s*|\(\s*)(\d+)\b'
COUNT = re.compile(r'\b(?:running\s+)?count\s*(?:[:=]\s*)?(\d+)\b', re.I)
LABEL = re.compile(r'\b(city|flower)[ -]score(?:\s+audit)?(?:\s+record)?\b', re.I)

def span(raw,a,b): return dict(start=a,end=b,text=raw[a:b])

def parse_category_trace(case, raw):
    if case['task'] != 'category_count': raise ValueError('category_count required')
    records=case['records']
    lookup={r['city'].casefold():r for r in records}
    if len(lookup)!=len(records): raise ValueError('unique entity names required')
    close=min([raw.index(c) for c in ('</think>','<channel|>') if c in raw] or [len(raw)])
    text=raw[:close]; candidates=[]
    for m in FULL.finditer(text):
        candidates.append(dict(a=m.start(),b=m.end(),entity=m[2].strip(),score=int(m[3]),identity=(m.start(2),m.end(2)),quoted=(m.start(1),m.end(1),m[1].lower()),form='full_quote'))
    for name,r in lookup.items():
        pattern=re.compile(r'(?<!\w)('+re.escape(r['city'])+r')(?!\w)'+SHORT_SCORE,re.I)
        for m in pattern.finditer(text):
            if any(c['a']<=m.start()<c['b'] for c in candidates): continue
            candidates.append(dict(a=m.start(),b=m.end(),entity=m[1],score=int(m[2]),identity=(m.start(1),m.end(1)),quoted=None,form='abbreviated'))
    candidates.sort(key=lambda c:c['a']); events=[]; seen=Counter()
    for i,c in enumerate(candidates):
        # Only the same line and immediately following metadata lines belong
        # to this record. Prose conclusions and later lists are not inherited.
        stop=candidates[i+1]['a'] if i+1<len(candidates) else close
        line_end=text.find('\n',c['b'],stop)
        end=line_end if line_end>=0 else stop
        if line_end>=0:
            pos=line_end+1
            while pos<stop:
                z=text.find('\n',pos,stop); z=stop if z<0 else z
                if not re.match(r'^\s*[*-]?\s*(?:Flower|City|Score|Type|Count)\s*:',text[pos:z],re.I): break
                end=z; pos=z+1
        tail=text[c['b']:end]
        labels=[dict(value=m[1].lower(),**span(raw,c['b']+m.start(),c['b']+m.end())) for m in LABEL.finditer(tail)]
        # Restrict explicit classification to annotation syntax; arbitrary
        # narrative labels in a sentence are retained as unknown.
        annotation=bool(re.match(r'^[.\s`"*]*(?:\(|->|[-–—]|\n\s*[*-]?\s*(?:Type|Flower|City)\s*:)',tail,re.I))
        labels=labels if annotation else []
        counts=[dict(value=int(m[1]),**span(raw,c['b']+m.start(),c['b']+m.end())) for m in COUNT.finditer(tail)] if annotation else []
        excluded=[span(raw,c['b']+m.start(),c['b']+m.end()) for m in re.finditer(r'\b(?:ignore|excluded|not counted)\b',tail,re.I)] if annotation else []
        vals={x['value'] for x in labels}
        category=next(iter(vals)) if len(vals)==1 else None
        membership={category==case['target_category']} if category else set()
        if excluded: membership.add(False)
        if counts: membership.add(True)
        r=lookup.get(c['entity'].casefold()); key=r['ordinal'] if r else c['entity'].casefold(); seen[key]+=1
        events.append(dict(step=i+1,form=c['form'],entity=c['entity'],source_ordinal=r['ordinal'] if r else None,
            record_end=span(raw,c['a'],c['b']),identity_end=span(raw,*c['identity']),
            observed_score=c['score'],score_matches_source=c['score']==r['score'] if r else None,
            quoted_category=dict(value=c['quoted'][2],**span(raw,*c['quoted'][:2])) if c['quoted'] else None,
            classification_evidence=labels,model_category=category,
            decision_status='conflict' if len(membership)>1 or len(vals)>1 else ('explicit' if membership else 'unknown'),
            model_is_target=next(iter(membership)) if len(membership)==1 and len(vals)<=1 else None,
            exclusion_evidence=excluded,count_evidence=counts,
            model_explicit_count=counts[0]['value'] if len(counts)==1 else None,
            visit_number=seen[key],gold_category=r.get('category') if r else None,
            gold_is_target=r['is_target'] if r else None))
    return dict(contract=CONTRACT,offset_coordinate='completion_text_raw',reasoning_end=close,
        events=events,unique_source_records=len({e['source_ordinal'] for e in events if e['source_ordinal'] is not None}),
        early_stop_validated=False,semantics='record mentions; repetitions do not imply independent retrieval')
