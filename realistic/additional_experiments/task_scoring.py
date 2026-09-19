"""Assay-specific scoring, consistent across both additional tasks."""
import re


def target_token_geometry(tokenizer, raw, ids, target_start, target, prefix_tokens):
    """Locate the saved next entity in original generated IDs, never substitute IDs."""
    if tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False) != raw:
        raise ValueError('Original generated-token decode mismatch')
    end=target_start+len(target)
    if raw[target_start:end].casefold()!=target.casefold():
        raise ValueError('Registered entity character span mismatch')
    encoded=tokenizer(raw,add_special_tokens=False,return_offsets_mapping=True)
    if list(encoded['input_ids'])==list(ids):
        overlap=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>a and a<end and b>target_start]
        if not overlap:raise ValueError('Entity has no original token span')
        start_token,stop_token=overlap[0],overlap[-1]+1
    else:
        # The same stable-original-prefix recovery used by the trace registry.
        bounds=[(0,0)]
        for j in range(1,len(ids)+1):
            prefix=tokenizer.decode(ids[:j],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if raw.startswith(prefix):bounds.append((len(prefix),j))
        start_token=max((n,j) for n,j in bounds if n<=target_start)[1]
        stop_token=min((n,j) for n,j in bounds if n>=end)[1]
    offset=start_token-prefix_tokens
    if offset<0 or stop_token<=start_token:raise ValueError('Target overlaps the frozen query prefix')
    return dict(target_token_ids=list(ids[start_token:stop_token]),target_token_offset=offset)


def score_generation(generation, case, *, mode, assay, target=None, target_token_ids=None, target_token_offset=None):
    if assay == "broad":
        from run import summarize_generation
        result = summarize_generation(generation, case, mode=mode, prefixed=True)
        result["metric"] = "final_task_answer_exact_accuracy"
        return result
    if assay != "targeted" or mode != "native_thinking" or not target:
        raise ValueError("Targeted requires a registered Native next-record target")
    from realistic_niah_v5.causal import _first_generated_city_record
    raw = generation["completion_text_raw"]
    pred, start, _ = _first_generated_city_record(raw, [r["city"] for r in case["records"]])
    hits = [] if pred is None else [(start, pred)]
    for m in re.finditer(r"(?:city|flower) score audit,\s+([^\n,.]+?) received a score", raw, re.I):
        hits.append((m.start(1), m[1]))
    prediction = min(hits)[1] if hits else None
    exact_prefix = bool(target_token_ids and target_token_offset == 0 and
        generation.get('generated_token_ids', [])[:len(target_token_ids)] == list(target_token_ids))
    fallback = prediction is None and exact_prefix
    return dict(prediction=prediction, target=target, exact_target_prefix=exact_prefix,
                token_prefix_fallback_used=fallback,
                correct=(prediction is not None and prediction.casefold() == target.casefold()) or fallback,
                metric="first_semantic_record_entity_accuracy")
