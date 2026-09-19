"""First numbered record-list boundary, without consulting the final answer."""
from __future__ import annotations

import re

CONTRACT = 'first_structured_record_episode_v2'
MARKER = re.compile(r'^( *)(?:\*\*)?(\d+)[.)](?:\*\*)?[ \t]+')
RECORD_MARKER = re.compile(r'^( *)(?:[-*+]\s+)?(?:\*\*)?Record\s+(\d+)(?:\*\*)?\s*[:.)-]\s*', re.I)


def _marker(line):
    return RECORD_MARKER.match(line) or MARKER.match(line)


def first_record_list(text: str, records: list[dict], *, final: bool = False):
    """Find the first closed numbered episode containing record city names.

    Indented continuation lines belong to the item. Blank lines do not close
    it. A new non-list paragraph or numbering restart closes the episode.
    Unnumbered prose and later repetitions are never merged into the episode.
    A partial streaming line that might become a number marker stays pending.
    """
    lines, cursor = [], 0
    for line in text.splitlines(keepends=True):
        lines.append((cursor, line))
        cursor += len(line)
    city_patterns = [(r, re.compile(r'(?<!\w)' + re.escape(r['city']) + r'(?!\w)')) for r in records]
    i = 0
    while i < len(lines):
        match = _marker(lines[i][1])
        if match is None or int(match[2]) != 1:
            i += 1
            continue
        indent, blocks, j, closed = len(match[1]), [], i, False
        bad_number = False
        while j < len(lines):
            offset, line = lines[j]
            marker = _marker(line)
            if marker and len(marker[1]) == indent:
                number = int(marker[2])
                if blocks and number == 1:
                    closed = True
                    break
                if number != len(blocks) + 1:
                    bad_number, closed = True, True
                    break
                blocks.append({'start': offset, 'end': offset + len(line.rstrip()), 'number': number})
            elif not line.strip():
                pass
            elif blocks and len(line) - len(line.lstrip(' ')) > indent:
                blocks[-1]['end'] = offset + len(line.rstrip())
            else:
                # "2" or "**2." at the streaming frontier is not yet a paragraph.
                partial = j == len(lines) - 1 and not line.endswith(('\n', '\r')) and not final
                could_be_marker = re.fullmatch(r'\s*(?:\*{0,2})?\d*[.)]?(?:\*{0,2})?\s*', line)
                closed = not (partial and could_be_marker)
                break
            j += 1
        if j == len(lines):
            closed = final
        found = [[r for r, pattern in city_patterns if pattern.search(text[b['start']:b['end']])]
                 for b in blocks]
        # A workflow step can contain the actual record list as a child list.
        # Descend structurally only when the parent spans multiple record cities;
        # never select a later list because its answer is better or more complete.
        for bi, (block, block_records) in enumerate(zip(blocks, found)):
            if len(block_records) <= 1:
                continue
            limit = blocks[bi + 1]['start'] if bi + 1 < len(blocks) else (
                lines[j][0] if j < len(lines) else len(text))
            for offset, line in lines:
                child = _marker(line)
                if (block['start'] < offset < limit and child and int(child[2]) == 1
                        and len(child[1]) > indent):
                    nested, nested_reason = first_record_list(text[offset:limit], records,
                        final=(bi + 1 < len(blocks) or closed))
                    if nested is not None:
                        nested['end'] += offset
                        nested['closure_start'] += offset
                        for item in nested['items']:
                            item['start'] += offset
                            item['end'] += offset
                        nested['nested'] = True
                        return nested, None
                    if nested_reason != 'no_numbered_record_episode':
                        return None, nested_reason
        if any(found):
            if not closed:
                return None, 'record_list_not_closed'
            if bad_number:
                return None, 'first_record_list_numbering_gap'
            if any(len(items) != 1 for items in found):
                return None, 'first_record_list_ambiguous_item'
            ordinals = [items[0]['ordinal'] for items in found]
            items = []
            for block, match_records in zip(blocks, found):
                record = match_records[0]
                items.append({**block, 'ordinal': record['ordinal'], 'is_target': record['is_target'],
                    'city': record['city'], 'explicit_index': True,
                    'line': text[block['start']:block['end']]})
            return {'contract': CONTRACT, 'items': items, 'end': blocks[-1]['end'],
                    'closure_start': lines[j][0] if j < len(lines) else len(text),
                    'nested': False,
                    'repeated_record_visits': len(ordinals) - len(set(ordinals)),
                    'passage_order_monotone': ordinals == sorted(ordinals)}, None
        i = max(i + 1, j)
    return None, 'no_numbered_record_episode'


def exact_list_prefix(tokenizer, prompt: dict, continuation: list[int], boundary: dict):
    """Preserve original token IDs; allow only whitespace past the item end."""
    raw = tokenizer.decode(continuation, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    full = prompt['rendered_prompt'] + raw
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    original = list(prompt['input_ids']) + continuation
    if list(encoded['input_ids']) != original:
        raise ValueError('Decoded list does not reproduce the original token IDs')
    from protocol import token_end
    shift = len(prompt['rendered_prompt'])
    end = shift + boundary['end']
    last = token_end(encoded['offset_mapping'], end - 1, end)
    actual_end = encoded['offset_mapping'][last][1]
    if actual_end > end and not full[end:actual_end].isspace():
        raise ValueError('List-end token contains non-list content')
    if actual_end > shift + boundary['closure_start']:
        raise ValueError('List-end token extends into the following paragraph')
    sites = []
    for item in boundary['items']:
        pos = token_end(encoded['offset_mapping'], shift + item['start'], shift + item['end'])
        sites.append({**item, 'position': pos})
    return original[:last + 1], sites


def generate_to_first_list(model, tokenizer, encoding, records, *, max_new_tokens=4096, check_every=4):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from realistic_niah_v4.modeling import _encoding_tensors

    class StopAtListEnd(StoppingCriteria):
        boundary = None
        def __call__(self, input_ids, scores, **kwargs):
            n = input_ids.shape[1] - encoding.sequence_length
            if n % check_every:
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            ids = input_ids[0, encoding.sequence_length:].tolist()
            raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            self.boundary, _ = first_record_list(raw, records)
            return torch.full((input_ids.shape[0],), self.boundary is not None,
                              dtype=torch.bool, device=input_ids.device)

    stopper = StopAtListEnd()
    ids, mask = _encoding_tensors(model, encoding)
    kwargs = {'input_ids': ids, 'attention_mask': mask, 'do_sample': False,
              'max_new_tokens': max_new_tokens, 'use_cache': True,
              'stopping_criteria': StoppingCriteriaList([stopper])}
    pad = tokenizer.pad_token_id
    if pad is not None:
        kwargs['pad_token_id'] = pad
    generated = model.generate(**kwargs)[0, encoding.sequence_length:].tolist()
    raw = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    boundary, reason = first_record_list(raw, records, final=stopper.boundary is None)
    return {'generated_token_ids': generated, 'completion_text_raw': raw, 'boundary': boundary,
            'unavailable_reason': reason, 'online_stop_triggered': stopper.boundary is not None,
            'check_every_tokens': check_every}
