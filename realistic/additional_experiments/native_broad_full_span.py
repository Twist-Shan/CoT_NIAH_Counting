"""Native Broad geometry and score using the main experiment's frozen functions."""
import hashlib
import math
from types import SimpleNamespace


def _sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def geometry_from_sites(tokenizer, raw, generated_ids, prompt_count, prefix_length, sites):
    from realistic_niah_v5.parsing import align_trace_sites
    from realistic_niah_v5.encoding import _visible_item_spans
    visible = prefix_length - prompt_count
    if not 0 < visible <= len(generated_ids):
        raise ValueError('Answer prefix outside original generated tokens')
    decoded = tokenizer.decode(generated_ids[:visible], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not raw.startswith(decoded):
        raise ValueError('Original answer prefix does not decode to raw text')
    token_sites = align_trace_sites(tokenizer, raw_text=raw,
        baseline_output_token_ids=generated_ids, sites=sites)
    spans = _visible_item_spans(token_sites, prompt_tokens=prompt_count,
        prefix=SimpleNamespace(shared_baseline_prefix_tokens=visible))
    registered = {int(s.slot_index):s for s in spans}
    records, excluded = [], []
    for site in token_sites:
        c = site.char_site
        span = registered.get(int(c.occurrence))
        if span is None:
            excluded.append(dict(occurrence=c.occurrence, char_start=c.char_start,char_end=c.char_end,
                reason='no_literal_full_span_before_answer', literal_token_start=site.literal_token_start,
                literal_token_end=site.literal_token_end))
            continue
        a,b = span.start-prompt_count,span.end-prompt_count
        before = tokenizer.decode(generated_ids[:a],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        after = tokenizer.decode(generated_ids[:b],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        assert before == raw[:c.char_start] and after == raw[:c.char_end]
        assert prompt_count <= span.start < span.end <= prefix_length
        records.append(dict(occurrence=c.occurrence,city=c.city,char_start=c.char_start,char_end=c.char_end,
            token_start=span.start,token_end=span.end,text=raw[c.char_start:c.char_end],
            token_ids=list(generated_ids[a:b]),prefix_start_sha256=_sha_text(before),prefix_end_sha256=_sha_text(after)))
    assert [(r['token_start'],r['token_end']) for r in records] == [(s.start,s.end) for s in spans]
    return dict(available=bool(records),reason='' if records else 'no_literal_full_record_span',
        record_scope='registered_generated_full_record_spans',prompt_token_count=prompt_count,
        prefix_length=prefix_length,answer_prefix_sha256=_sha_text(decoded),records=records,
        excluded_records=excluded,registered_records=len(sites),record_count=len(records))


def full_span_geometry(tokenizer, plan, prompt, generation, model):
    from realistic_niah_v5.parsing import parse_trace_record, TraceCharSite
    if not plan['broad_prefix']:
        return dict(available=False,reason=plan['unavailable'].get('broad','baseline_query_unavailable'),
            record_scope='registered_generated_full_record_spans',records=[],excluded_records=[],record_count=0)
    raw=generation['completion_text_raw']
    parsed=parse_trace_record(dict(generation,model_label=model,
        model_family='qwen3' if model.startswith('Qwen') else 'gemma4',gold_records=plan['case']['records']))
    sites=[TraceCharSite(**s) for s in parsed['char_sites'] if s['site_kind']=='item_end']
    geo=geometry_from_sites(tokenizer,raw,generation['generated_token_ids'],len(prompt['input_ids']),plan['broad_prefix'],sites)
    geo['sequence_source']=parsed['sequence_source']
    return geo


def score_masses(masses):
    from realistic_niah_v5.capture import _broad_span_metrics
    if not masses or any(not math.isfinite(x) or x<0 for x in masses):
        raise ValueError('Invalid record attention masses')
    return _broad_span_metrics(masses)['score']


def canary_cases(plans, geometry):
    eligible=[p for p in plans if p['split']=='confirmation' and geometry[p['case_id']]['available']]
    chosen=[]
    for target in (['city','flower'] if plans[0]['case']['task']=='category_count' else ['']):
        subset=[p for p in eligible if p['case'].get('target_category','')==target]
        if not subset:raise ValueError('Canary subgroup has no full-span support')
        first=min(p['seed'] for p in subset)
        subset=sorted([p for p in subset if p['seed']==first],key=lambda p:(int(p['case']['level']),p['case_id']))
        chosen += [subset[0]] if target else [subset[0],subset[-1]]
    if len({p['case_id'] for p in chosen})!=2:raise ValueError('Canary requires two distinct eligible inputs')
    return chosen
