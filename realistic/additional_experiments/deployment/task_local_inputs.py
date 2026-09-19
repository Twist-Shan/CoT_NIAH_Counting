"""Input audits and original-token sites for a fresh Appendix H campaign.

The anchor rule is the retained aligned-transfer preparation rule. Selection
uses discovery attention only; natural-answer correctness never selects sites.
"""
import math
from collections import Counter

MODES = ('nonthinking', 'native_thinking')
BROAD_SIZES = {'Qwen3-8B': [1, 2, 4, 8, 16, 32, 64, 128],
               'Gemma4-E4B': [1, 2, 4, 6, 8]}
TARGET_SIZES = {'Qwen3-8B': [32, 64, 80, 96, 112, 128],
                'Gemma4-E4B': [1, 2, 4, 6, 8]}


def validate_cases(cases, task):
    if len(cases) != 300 or len({c['case_id'] for c in cases}) != 300:
        raise ValueError('Expected 300 distinct task cases')
    if Counter(c['seed'] for c in cases) != Counter({s: 10 for s in range(1234, 1264)}):
        raise ValueError('Expected ten inputs for each registered seed 1234..1263')
    for seed in range(1234, 1264):
        rows = [c for c in cases if c['seed'] == seed]
        levels = Counter(c['level'] for c in rows)
        expected = Counter(range(1, 11)) if task == 'kth' else Counter({n: 2 for n in (1, 3, 5, 7, 9)})
        if levels != expected:
            raise ValueError(f'Incomplete {task} design at seed {seed}')
    for c in cases:
        if c['task'] != ('kth_needle' if task == 'kth' else 'category_count'):
            raise ValueError('Unexpected task identity')
        if len(c['records']) != 10:
            raise ValueError('Each Appendix H passage must contain ten source records')
        for r in c['records']:
            if c['passage'][r['char_start']:r['char_end']] != r['text']:
                raise ValueError('Record character offsets do not match the passage')
        if task == 'kth':
            target = c['records'][c['level']-1]
            expected_gold = f"{target['city']}|{target['score']}"
        else:
            expected_gold = str(sum(r['category'] == c['target_category'] for r in c['records']))
            if expected_gold != str(c['level']):
                raise ValueError('Category level does not equal its source-record count')
        if c['gold'] != expected_gold:
            raise ValueError('Task gold does not match the source records')


def make_plan(case, prompt, generation, tokenizer, model, mode):
    from compile_transfer_registry import compile_one, token_boundaries
    from realistic_niah_v5.parsing import parse_trace_record
    from realistic_niah_v5.causal_sites import _rank_event_rows, _structural_event_rows

    registry = compile_one(case, prompt, generation, tokenizer, model, mode)
    if registry['token_map_method'] == 'original_decode_mismatch':
        raise ValueError('Saved generated token IDs do not decode to the saved completion')
    item = dict(case_id=case['case_id'], seed=case['seed'],
                split='discovery' if case['seed'] < 1254 else 'confirmation',
                mode=mode, case=case, broad_prefix=registry['answer_query'].get('prefix_length'),
                trace_positions=[], target=None, unavailable={})
    if not item['broad_prefix']:
        item['unavailable']['broad'] = registry['answer_query'].get('unavailable_reason')
    if mode == 'native_thinking':
        raw = generation['completion_text_raw']
        n = len(prompt['input_ids'])
        bounds, _, _ = token_boundaries(tokenizer, raw, generation['generated_token_ids'])
        parsed = parse_trace_record(dict(generation, model_label=model,
            model_family='qwen3' if model.startswith('Qwen') else 'gemma4', gold_records=case['records']))
        item['sequence_source'] = parsed['sequence_source']
        rows = (_rank_event_rows if parsed['sequence_source'] == 'rank_supported_episode'
                else _structural_event_rows)(parsed=parsed, parser=parsed['parser'])

        def endpoint(z):
            after = sorted(k for k in bounds if k >= z)
            if not after or raw[z:after[0]].strip(' \t\n\r.,;:!?"\'`*()[]'):
                return None
            return n + bounds[after[0]]

        item['trace_positions'] = sorted({endpoint(e['semantic_end_char']) - 1
            for e in rows if endpoint(e['semantic_end_char']) is not None})
        candidates = []
        for i in range(1, len(rows)):
            prev, event = rows[i-1], rows[i]
            role, z = 'p0_item_end', prev['semantic_end_char']
            marker, start = event.get('rank_evidence_end_char'), event.get('rank_evidence_start_char')
            if model.startswith('Qwen') and marker is not None and marker <= event['city_start_char'] and start is not None:
                role, z = 'post_marker', marker
            prefix = endpoint(z)
            if prefix is None:
                continue
            decoded = tokenizer.decode(generation['generated_token_ids'][:prefix-n],
                                       skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not raw.startswith(decoded) or len(decoded) > event['city_start_char']:
                continue
            candidates.append(dict(prefix_length=prefix, role=role, target_city=event['city'],
                occurrence=i+1, query_char_end=len(decoded), target_city_start=event['city_start_char']))
        item['target'] = candidates[(len(candidates)-1)//2] if candidates else None
        item['target_candidates'] = len(candidates)
        if not candidates:
            item['unavailable']['targeted'] = 'no_registered_next_record_transition'
        if not item['trace_positions']:
            item['broad_prefix'] = None
            item['unavailable']['broad'] = 'no_registered_item_endpoint'
    return item, registry


def broad_spans(plan, prompt, tokenizer):
    """Original campaign: all prompt records, or Native registered endpoint keys."""
    if plan['mode'] == 'native_thinking':
        return [(z, z+1) for z in plan['trace_positions'] if z < plan['broad_prefix']]
    encoded = tokenizer(prompt['rendered_prompt'], add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) != prompt['input_ids']:
        raise ValueError('Prompt token mismatch during Broad discovery')
    shift = prompt['rendered_prompt'].index(plan['case']['passage'])
    spans = []
    for r in plan['case']['records']:
        ix = [i for i, (s, e) in enumerate(encoded['offset_mapping'])
              if e > s and s < shift+r['char_end'] and e > shift+r['char_start']]
        if not ix:
            raise ValueError('Source record has no prompt tokens')
        spans.append((ix[0], ix[-1]+1))
    return spans


def broad_score(masses):
    """Appendix H's epsilon convention, identical to kth_retrieval.broad."""
    if any(not math.isfinite(x) or x < 0 for x in masses):
        raise ValueError('Invalid attention mass')
    total = sum(masses)
    if not masses or total <= 1e-12:
        return 0.
    probs = [v / (total+1e-12) for v in masses]
    return total * math.exp(-sum(v*math.log(v+1e-12) for v in probs)) / len(masses)


def expected_points(plans, model, broad_sizes=None, target_sizes=None):
    broad_sizes = BROAD_SIZES if broad_sizes is None else broad_sizes
    target_sizes = TARGET_SIZES if target_sizes is None else target_sizes
    return sum((len(broad_sizes[model]) if p['broad_prefix'] else 0)
               + (len(target_sizes[model]) if p['mode'] == 'native_thinking' and p['target'] else 0)
               for p in plans if p['split'] == 'confirmation')
