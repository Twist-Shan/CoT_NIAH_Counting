"""Frozen city/flower behavioral smoke with the original excerpt grammar."""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

from protocol import source_records, read_jsonl, text_hash, sha256, write_json, encode_ids

ROOT = Path(__file__).resolve().parent
FLOWERS = ('tulip', 'rose', 'daisy', 'lily', 'orchid', 'iris', 'poppy', 'peony', 'marigold', 'daffodil')


def make_pair(source, city_count):
    records = source_records(source)
    rng = random.Random(20260906 + source['seed'])
    order = list(range(len(records)))
    rng.shuffle(order)
    city_slots = set(order[:city_count])
    parts, output, cursor, size = [], [], 0, 0
    for i, record in enumerate(records):
        before = source['passage'][cursor:record['char_start']]
        parts.append(before)
        size += len(before)
        category = 'city' if i in city_slots else 'flower'
        name = record['city'] if category == 'city' else FLOWERS[i]
        old = f"In the 2024 city score audit, {record['city']} received a score of {record['score']}."
        new = f"In the 2024 {category} score audit, {name} received a score of {record['score']}."
        if record['text'].count(old) != 1:
            raise ValueError('Unexpected original needle grammar')
        needle = record['text'].replace(old, new)
        output.append({'city': name, 'entity': name, 'category': category,
                       'source_city': record['city'], 'score': int(record['score']), 'ordinal': i + 1,
                       'text': needle, 'char_start': size, 'char_end': size + len(needle)})
        parts.append(needle)
        size += len(needle)
        cursor = record['char_end']
    parts.append(source['passage'][cursor:])
    passage = ''.join(parts)
    for target in ('city', 'flower'):
        gold = sum(r['category'] == target for r in output)
        yield {'case_id': f"category_count_seed{source['seed']}_city{city_count}_target{target}",
               'task': 'category_count', 'seed': source['seed'], 'level': gold,
               'split': 'parser_development_smoke', 'target_category': target, 'target_topic': None,
               'records': [{**r, 'is_target': r['category'] == target} for r in output],
               'passage': passage, 'passage_sha256': text_hash(passage),
               'source_passage_sha256': text_hash(source['passage']),
               'total_records': len(output), 'gold': str(gold), 'answer_prefix': 'Total:'}


def prompt(case, mode):
    target = case['target_category']
    cue = (f'You will need to count only {target}-score audit records in the passage below.\n'
           'A city-score audit record names one city and gives its numeric score.\n'
           'A flower-score audit record names one flower and gives its numeric score.')
    question = (f'How many {target}-score audit records are in the passage?\n'
                'Count only matching audit records. Ignore audit records of the other category and ordinary passage text.')
    instruction = ('Reason concisely without repeating or restarting.\n'
                   'Stop as soon as you determine the count, then output exactly one line:\nTotal:<integer>'
                   if mode == 'native_thinking' else
                   'Do not explain, reason aloud, quote, or list any records.\n'
                   'Write the count using ordinary decimal digits, with no space after the colon.\n'
                   'Your entire response must be exactly one line:\nTotal:<integer>')
    return f"{cue}\n\n<passage>\n{case['passage']}\n</passage>\n\n{question}\n{instruction}"


def freeze(source, output):
    started = time.perf_counter()
    lookup = {r['seed']: r for r in read_jsonl(source) if r['design_variant'] == 'v4.4'
              and r['gold_count'] == 10 and r['seed'] in (1234, 1254)}
    cases = [c for seed, n in ((1234, 2), (1254, 8)) for c in make_pair(lookup[seed], n)]
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'cases.jsonl').open('w', encoding='utf-8') as f:
        for c in cases:
            for r in c['records']:
                assert c['passage'][r['char_start']:r['char_end']] == r['text']
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    prompts = []
    for c in cases:
        for mode in ('nonthinking', 'native_thinking'):
            text = prompt(c, mode)
            prompts.append({'case_id': c['case_id'], 'mode': mode, 'user_text': text,
                            'user_text_sha256': text_hash(text)})
            (output / f"{c['case_id']}__{mode}.txt").write_text(text, encoding='utf-8')
    with (output / 'user_prompts.jsonl').open('w', encoding='utf-8') as f:
        for row in prompts:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(output / 'manifest.json', {'status': 'PASS', 'source_sha256': sha256(source),
        'files': {n: sha256(output / n) for n in ('cases.jsonl', 'user_prompts.jsonl')},
        'cases': len(cases), 'model_mode_generations': 16, 'elapsed_seconds': time.perf_counter() - started,
        'scope': 'Behavior/trace smoke only; two existing development seeds, composition confounded with seed.',
        'entity_compatibility': 'records.city stores the entity identity for the existing trace parser; category is separate.',
        'generation': {'native_max_new_tokens': 4096, 'answer_max_new_tokens': 64, 'do_sample': False}})


def run(args):
    import torch
    sys.path.insert(0, str(ROOT.parent / 'src'))
    from realistic_niah_v4.modeling import load_registered_model, generate_answer_completion
    from realistic_niah_v4.spec import resolve_model_spec
    from run import summarize_generation
    from trace_parser import parse_trace, align_sites
    manifest = json.loads((args.frozen / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if sha256(args.frozen / name) != expected:
            raise ValueError('Frozen input drift')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    out = args.output / args.model
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.manual_seed(20260906)
    spec = resolve_model_spec(args.model)
    model, tokenizer, _ = load_registered_model(spec, cache_dir=args.cache_dir)
    tokenizer.padding_side = 'left'
    write_json(out / 'contract.json', {'model': args.model, 'model_id': spec.model_id,
        'revision': spec.revision, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'dtype': 'bfloat16', 'attention_backend': 'sdpa', 'gpu': torch.cuda.get_device_name(),
        'code_hashes': {p.name: sha256(p) for p in ROOT.glob('*.py')},
        'frozen_manifest_sha256': sha256(args.frozen / 'manifest.json'),
        'model_load_seconds': time.perf_counter() - started})
    cases = {c['case_id']: c for c in read_jsonl(args.frozen / 'cases.jsonl')}
    results = []
    for row in read_jsonl(args.frozen / 'user_prompts.jsonl'):
        case, mode = cases[row['case_id']], row['mode']
        t = time.perf_counter()
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': row['user_text']}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=mode == 'native_thinking')
        if mode == 'nonthinking':
            rendered += 'Total:'
        ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
        generation = generate_answer_completion(model, tokenizer, encode_ids(ids),
                                max_new_tokens=4096 if mode == 'native_thinking' else 64)
        generation.update(summarize_generation(generation, case, mode=mode, prefixed=mode == 'nonthinking'))
        dest = out / 'captures' / mode / case['case_id']
        write_json(dest / 'prompt.json', {**row, 'rendered_prompt': rendered, 'input_ids': ids})
        write_json(dest / 'generation.json', generation)
        if mode == 'native_thinking':
            parsed = parse_trace(case, generation['completion_text_raw'])
            encoded = tokenizer(rendered + generation['completion_text_raw'], add_special_tokens=False,
                                return_offsets_mapping=True)
            identical = encoded['input_ids'] == ids + generation['generated_token_ids']
            parsed['original_ids_reproduced'] = identical
            parsed['token_sites'] = align_sites(parsed, encoded['offset_mapping'], shift=len(rendered)) if identical else []
            write_json(dest / 'trace_parsed.json', parsed)
        result = {'case_id': case['case_id'], 'mode': mode, 'target': case['target_category'],
                  'gold': case['gold'], 'prediction': generation['prediction'], 'correct': generation['correct'],
                  'parse_ok': generation['parse_ok'], 'truncated': generation['generation_truncated'],
                  'tokens': generation['generated_token_count'], 'elapsed_seconds': time.perf_counter() - t}
        results.append(result)
        write_json(out / 'progress.json', results)
        print(json.dumps(result), flush=True)
    write_json(out / 'complete.json', {'status': 'PASS', 'cases': len(results),
               'elapsed_seconds': time.perf_counter() - started, 'results': results})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--freeze-source', type=Path)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--model')
    a = p.parse_args()
    if a.freeze_source:
        freeze(a.freeze_source, a.frozen)
    else:
        run(a)
