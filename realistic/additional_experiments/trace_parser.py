"""Auditable offline record-block parser; no answer-dependent selection or stopping.

Offsets are half-open character spans in completion_text_raw. Gold metadata is
attached only after structural selection. Unknown semantics stay unknown.
"""
from __future__ import annotations

import re
from collections import Counter

from list_early_stop import first_record_list

CONTRACT = 'record_blocks_v1'
COUNT = re.compile(r'\b(?:running\s+count|count)\s*(?:\*\*)?\s*[:=]\s*(?:\*\*)?\s*(\d+)\b', re.I)
NEGATIVE = re.compile(r"\b(?:does(?:n['’]t| not)\s+count|do(?:n['’]t| not)\s+count|not\s+counted)\b", re.I)
POSITIVE = re.compile(r'\b(?:(?:this\s+)?counts|counted)\b', re.I)
SCORE = re.compile(r'(?:\breceived\s+a\s+score\s+of|\bwith\s+(?:a\s+score\s+of\s+)?|\bscore\s*[:=]?|\|)\s*(\d+)\b|^\s*\((\d+)\)|^\s+(\d+)\b|^\s*[-–—:]\s*(\d+)\b', re.I)


def _span(text, start, end):
    return {'start': start, 'end': end, 'text': text[start:end]}


def _decisions(block, target):
    """Explicit membership evidence only; lexical ambiguity is not resolved by gold."""
    evidence = []
    negatives = list(NEGATIVE.finditer(block))
    if target:
        negatives += list(re.finditer(r'\bnot\s+' + re.escape(target) + r'\b', block, re.I))
    for m in negatives:
        evidence.append((m.start(), m.end(), False, 'explicit_exclusion'))
    for m in POSITIVE.finditer(block):
        if not any(a.start() <= m.start() < a.end() for a in negatives):
            evidence.append((m.start(), m.end(), True, 'explicit_inclusion'))
    if target:
        # Parenthetical topic labels and explicit copular classifications only.
        # A bare mention in a quoted project or question is not classification.
        patterns = [r'\((?:' + re.escape(target) + r')(?:/[^)\n]+)?\)',
                    r'\b(?:is|are|related\s+to)\s+' + re.escape(target) + r'\b',
                    r'\b(?:this\s+is\s+another|this\s+is\s+an?)\s+' + re.escape(target) + r'\s+project\b']
        for pattern in patterns:
            for m in re.finditer(pattern, block, re.I):
                prefix = block[max(0, m.start() - 20):m.start()]
                if re.search(r"\b(?:not|isn't|aren't|isn’t|aren’t)\s*$", prefix, re.I):
                    continue
                evidence.append((m.start(), m.end(), True, 'explicit_topic_label'))
    return sorted(set(evidence))


def parse_trace(case: dict, raw: str):
    records = case['records']
    cities = [r['city'] for r in records]
    if len(set(cities)) != len(cities):
        raise ValueError('This parser contract requires unique source city names')
    closers = [raw.index(s) for s in ('</think>', '<channel|>') if s in raw]
    close = min(closers) if closers else len(raw)
    reasoning = raw[:close]
    boundary, reason = first_record_list(reasoning, records, final=True)
    result = {'contract': CONTRACT, 'offset_coordinate': 'completion_text_raw',
              'reasoning_end': close, 'channel_close_found': bool(closers),
              'selection': 'first_structured_record_episode',
              'status': 'structured' if boundary else 'unavailable',
              'unavailable_reason': reason, 'items': [], 'mention_candidates': []}
    # A separate mention inventory exposes prose and later repetitions without
    # treating each city mention as a completed reasoning step.
    for r in records:
        for m in re.finditer(r'(?<!\w)' + re.escape(r['city']) + r'(?!\w)', reasoning):
            result['mention_candidates'].append({**_span(raw, m.start(), m.end()),
                                                 'ordinal': r['ordinal']})
    result['mention_candidates'].sort(key=lambda x: x['start'])
    if boundary is None:
        return result
    result['episode_span'] = _span(raw, boundary['items'][0]['start'], boundary['end'])
    result['nested'] = boundary['nested']
    seen, target_visits, unique_targets = Counter(), 0, set()
    for step, b in enumerate(boundary['items'], 1):
        r = next(r for r in records if r['ordinal'] == b['ordinal'])
        start, end = b['start'], b['end']
        block = raw[start:end]
        city_match = re.search(r'(?<!\w)' + re.escape(r['city']) + r'(?!\w)', block)
        seen[r['ordinal']] += 1
        target_visits += int(r['is_target'])
        if r['is_target']:
            unique_targets.add(r['ordinal'])
        identity = _span(raw, start + city_match.start(), start + city_match.end())
        # Score evidence must follow this city before a sentence or line break.
        tail = block[city_match.end():]
        clause = re.split(r'[\n\r.!?]', tail, maxsplit=1)[0]
        score = SCORE.search(clause)
        score_value = int(next(v for v in score.groups() if v is not None)) if score else None
        record_span = (_span(raw, identity['start'], start + city_match.end() + score.end())
                       if score else None)
        decisions = ([{'value': value, 'rule': rule, **_span(raw, start + a, start + z)}
                     for a, z, value, rule in _decisions(block, case.get('target_topic'))]
                     if case['task'] == 'topic_count' else [])
        values = {d['value'] for d in decisions}
        decision = decisions[-1] if len(values) == 1 else None
        counts = [{'value': int(m[1]), **_span(raw, start + m.start(), start + m.end())}
                  for m in COUNT.finditer(block)]
        # Multiple count statements remain explicitly ambiguous, even if equal.
        count = counts[0] if len(counts) == 1 else None
        result['items'].append({
            'step': step, 'list_number': b['number'], 'source_ordinal': r['ordinal'],
            'city': r['city'], 'visit_number': seen[r['ordinal']],
            'block': _span(raw, start, end), 'identity_end': identity,
            'record_end': record_span, 'observed_score': score_value,
            'score_matches_source': None if score is None else score_value == r['score'],
            'decision_evidence': decisions, 'decision_end': decision,
            'model_is_target': decision['value'] if decision else None,
            'decision_status': 'conflict' if len(values) > 1 else ('explicit' if decision else 'unknown'),
            'count_evidence': counts, 'count_end': count,
            'model_explicit_count': count['value'] if count else None,
            'count_status': 'multiple' if len(counts) > 1 else ('explicit' if count else 'unknown'),
            'gold_is_target': r['is_target'],
            'gold_target_visits_so_far': target_visits,
            'gold_unique_targets_visited': len(unique_targets),
            'gold_source_prefix_targets': sum(int(x['is_target']) for x in records
                                              if x['ordinal'] <= r['ordinal']),
        })
    result['coverage'] = {'visited_unique_records': len(seen), 'source_records': len(records),
                          'unvisited_source_ordinals': [r['ordinal'] for r in records if not seen[r['ordinal']]],
                          'repeated_visits': sum(seen.values()) - len(seen),
                          'passage_order_monotone': boundary['passage_order_monotone']}
    result['phase_description'] = ('contains_explicit_decisions' if any(i['decision_evidence'] for i in result['items'])
                                   else 'no_explicit_decisions_detected')
    result['early_stop_validated'] = False
    return result


def align_sites(parsed, offsets, *, shift=0):
    """Map spans to exact token ends after caller verifies original token IDs.

    Reject merged future non-span content, including punctuation. This function
    does not tokenize or claim a decoded/re-encoded sequence is identical.
    """
    aligned = []
    for item in parsed['items']:
        for kind in ('identity_end', 'record_end', 'decision_end', 'count_end'):
            span = item[kind]
            if span is None:
                continue
            endpoint = shift + span['end']
            positions = [j for j, (a, b) in enumerate(offsets) if a < endpoint <= b]
            exact = len(positions) == 1 and offsets[positions[0]][1] == endpoint
            aligned.append({'step': item['step'], 'kind': kind,
                            'position': positions[0] if exact else None,
                            'unavailable_reason': None if exact else 'nonexact_or_ambiguous_token_end'})
    return aligned
