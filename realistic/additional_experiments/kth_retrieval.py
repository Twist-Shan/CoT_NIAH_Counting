"""Kth full-panel retrieval assays; no count representation prerequisite."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys
import time

from protocol import make_case, read_jsonl, write_json, sha256, user_prompt, encode_ids

ROOT = Path(__file__).resolve().parent
K_LEVELS = tuple(range(1, 11))


def broad(masses):
    if any(not math.isfinite(x) or x < 0 for x in masses):
        raise ValueError('Invalid attention mass')
    total = sum(masses)
    if not masses or total <= 1e-12:
        return {'mass': total, 'coverage': 0., 'score': 0.}
    probabilities = [x / (total + 1e-12) for x in masses]
    effective = math.exp(-sum(p * math.log(p + 1e-12) for p in probabilities))
    return {'mass': total, 'coverage': effective / len(masses), 'score': total * effective / len(masses)}


def random_bank(bank, widths, seed, *, global_control=False):
    rng = random.Random(seed)
    selected = set(map(tuple, bank))
    if global_control:
        pool = [(l, h) for l, n in enumerate(widths) for h in range(n) if (l, h) not in selected]
        return rng.sample(pool, len(bank))
    output = []
    for l, n in sorted(Counter(l for l, h in bank).items()):
        pool = [(l, h) for h in range(widths[l]) if (l, h) not in selected]
        if len(pool) < n:
            raise ValueError('Disjoint layer-matched control unavailable')
        output.extend(rng.sample(pool, n))
    return output


def select_broad(ranking, widths, n):
    selected, occupied = [], Counter()
    for l, h in ranking:
        if occupied[l] < widths[l] // 2:
            selected.append((l, h))
            occupied[l] += 1
        if len(selected) == n:
            return selected
    raise ValueError('Insufficient heads for broad bank')


def exact_pre_city(raw, city, offsets, shift, reasoning_end):
    """First target-city mention in reasoning, regardless of correctness/list form.

    This is an identity-anchored observational site, not an inferred counter state.
    Reject tokens containing prior nonwhitespace text, or partial city starts.
    """
    m = re.search(r'(?<!\w)' + re.escape(city) + r'(?!\w)', raw[:reasoning_end])
    if not m:
        return None, 'target_city_not_mentioned_in_reasoning'
    a = shift + m.start()
    hits = [j for j, (s, e) in enumerate(offsets) if s <= a < e]
    if len(hits) != 1 or hits[0] < 1:
        return None, 'ambiguous_city_start_token'
    j = hits[0]
    s = offsets[j][0]
    if s < shift or (s < a and not raw[s-shift:m.start()].isspace()):
        return None, 'city_start_token_contains_prior_content'
    return {'prefix_length': j, 'char_start': m.start(), 'city': city,
            'query_position': j - 1, 'anchor': 'first_target_city_mention_pre_token'}, None


def city_correct(raw, city):
    return re.match(r'^[\s*"\'`]*' + re.escape(city) + r'(?!\w)', raw) is not None


def freeze(source, output):
    t = time.perf_counter()
    rows = {r['seed']: r for r in read_jsonl(source) if r['design_variant'] == 'v4.4' and r['gold_count'] == 10}
    config = {'schema': 'kth_retrieval_v1', 'discovery_seeds': list(range(1234, 1254)),
        'confirmation_seeds': list(range(1254, 1264)), 'levels': list(K_LEVELS),
        'modes': ['nonthinking', 'native_thinking'], 'models': ['Qwen3-8B', 'Gemma4-E4B'],
        'max_native_tokens': 4096, 'max_answer_tokens': 64, 'random_replicates': 3,
        'broad_bank_sizes': {'Qwen3-8B': 32, 'Gemma4-E4B': 6},
        'broad_selection': 'new_task_discovery_seed_equal_broad_score; half_heads_per_layer_cap',
        'broad_source': {'nonthinking': 'prompt_full_record_spans', 'native_thinking': 'first_structured_episode_record_end_tokens'},
        'targeted_anchor': 'native first mention of gold kth city before channel close; nonthinking answer prefix',
        'targeted_intervention': 'frozen legacy bank, one-shot pre-city query; no full-trace rollout',
        'targeted_random_matching': {'Qwen3-8B': 'global_disjoint', 'Gemma4-E4B': 'layer_matched_disjoint'},
        'unit': 'seed; equal average over available k within seed',
        'count_representation': False}
    cases = [make_case(rows[s], 'kth_needle', k, 'discovery' if s < 1254 else 'confirmation')
             for s in range(1234, 1264) for k in config['levels']]
    output.mkdir(parents=True, exist_ok=False)
    banks, provenance = {}, {}
    for model, name in [('Qwen3-8B', 'qwen_shared_k128'), ('Gemma4-E4B', 'gemma_shared_k6')]:
        path = ROOT.parent / 'configs' / f'realistic_niah_v5_{name}_targeted_selection_frozen.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        banks[model] = data['development_selection']['primary_bank_heads']
        provenance[model] = {'source': str(path), 'sha256': sha256(path)}
        write_json(output / (model + '_targeted_source.json'), data)
    write_json(output / 'config.json', config)
    write_json(output / 'targeted_banks.json', {'banks': banks, 'provenance': provenance})
    with (output / 'cases.jsonl').open('w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    with (output / 'user_prompts.jsonl').open('w', encoding='utf-8') as f:
        for case in cases:
            for mode in config['modes']:
                f.write(json.dumps({'case_id': case['case_id'], 'mode': mode,
                                    'user_text': user_prompt(case, mode)}, ensure_ascii=False) + '\n')
    write_json(output / 'manifest.json', {'status': 'PASS', 'case_count': len(cases),
        'natural_generations': len(cases) * 4, 'source_sha256': sha256(source),
        'elapsed_seconds': time.perf_counter() - t,
        'files': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})


def execute(args):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT.parent / 'src'))
    from run import render, summarize_generation
    from diagnostics import native_endpoint
    from trace_parser import parse_trace, align_sites
    from realistic_niah_v4.modeling import load_registered_model, generate_answer_completion, query_attention_rows, generate_with_head_ablation
    from realistic_niah_v4.spec import resolve_model_spec
    frozen, out = args.frozen, args.output / args.model
    manifest = json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))
    for n, h in manifest['files'].items():
        if sha256(frozen / n) != h:
            raise ValueError('Frozen input drift: ' + n)
    cfg = json.loads((frozen / 'config.json').read_text(encoding='utf-8'))
    cases = read_jsonl(frozen / 'cases.jsonl')
    frozen_prompts = {(r['case_id'],r['mode']):r['user_text'] for r in read_jsonl(frozen/'user_prompts.jsonl')}
    if args.canary:
        cases = [c for c in cases if c['seed'] in (1234, 1254) and c['level'] in (1, 10)]
    contract = {'config': cfg, 'canary': args.canary, 'model': args.model,
                'frozen_manifest_sha256': sha256(frozen / 'manifest.json'),
                'code_hashes': {p.name: sha256(p) for p in ROOT.glob('*.py')},
                'src_hashes': {str(p.relative_to(ROOT.parent / 'src')): sha256(p) for p in (ROOT.parent / 'src').rglob('*.py')}}
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'contract.json').exists():
        if json.loads((out / 'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Resume contract mismatch')
    else:
        write_json(out / 'contract.json', contract)
    torch.manual_seed(20260906)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    started = time.perf_counter()
    spec = resolve_model_spec(args.model)
    model, tok, adapter = load_registered_model(spec, cache_dir=args.cache_dir)
    widths = list(adapter.num_heads)
    tok.padding_side = 'left'
    write_json(out / 'environment.json', {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'model_id': spec.model_id, 'revision': spec.revision, 'model_load_seconds': time.perf_counter()-started,
        'transformers': __import__('transformers').__version__, 'head_counts': widths,
        'dtype': 'bfloat16', 'attention_backend': 'sdpa', 'seed': 20260906})
    targeted = json.loads((frozen / 'targeted_banks.json').read_text(encoding='utf-8'))['banks'][args.model]
    targeted_controls = [random_bank(targeted, widths, 6000+r, global_control=args.model == 'Qwen3-8B') for r in range(3)]

    def attention(enc, spans, trace_positions, k):
        rows, starts = query_attention_rows(model, adapter, enc)
        result = []
        for l, (matrix, start) in enumerate(zip(rows, starts)):
            for h, alpha in enumerate(matrix):
                def mass(a, b):
                    lo, hi = max(0, a-start), min(len(alpha), b-start)
                    return float(alpha[lo:hi].sum()) if hi > lo else 0.
                masses = [mass(a, b) for a, b in spans]
                trace_masses = [mass(p, p+1) for p in trace_positions]
                result.append({'layer': l, 'head': h, 'record_masses': masses,
                    'prompt_broad': broad(masses), 'trace_broad': broad(trace_masses),
                    'target_mass': masses[k-1], 'target_relative_mass': masses[k-1]/(sum(masses)+1e-12)})
        return result

    for mode in cfg['modes']:
        for case in cases:
            dest = out / 'captures' / mode / case['case_id']
            if (dest / 'complete.json').exists():
                saved=json.loads((dest/'complete.json').read_text(encoding='utf-8'))
                for n,h in saved['files'].items():
                    if sha256(dest/n)!=h:
                        raise ValueError('Changed completed capture: '+str(dest/n))
                continue
            t = time.perf_counter()
            enc, prompt_offsets, prompt = render(case, mode, tok)
            if prompt['user_text'] != frozen_prompts[(case['case_id'],mode)]:
                raise ValueError('Rendered user prompt differs from frozen text')
            generation = generate_answer_completion(model, tok, enc, max_new_tokens=cfg['max_native_tokens'] if mode == 'native_thinking' else 64)
            generation.update(summarize_generation(generation, case, mode=mode, prefixed=mode == 'nonthinking'))
            raw = generation['completion_text_raw']
            full = prompt['rendered_prompt'] + raw
            encoded = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            original = list(enc.input_ids) + generation['generated_token_ids']
            identical = encoded['input_ids'] == original
            offsets = encoded['offset_mapping']
            pstart = prompt['rendered_prompt'].index(case['passage'])
            spans = []
            for record in case['records']:
                a,b = pstart+record['char_start'], pstart+record['char_end']
                ix = [j for j,(s,e) in enumerate(prompt_offsets) if e>s and s<b and e>a]
                spans.append((ix[0], ix[-1]+1))
            parsed, trace_pos, target_anchor, target_reason = None, [], None, None
            if mode == 'native_thinking':
                parsed = parse_trace(case, raw)
                aligned = align_sites(parsed, offsets, shift=len(prompt['rendered_prompt'])) if identical else []
                trace_pos = [x['position'] for x in aligned if x['kind']=='record_end' and x['position'] is not None]
                parsed['token_sites'] = aligned
                endpoint, endpoint_reason = native_endpoint(tok, prompt['rendered_prompt'], generation, 'Needle:')
                final_enc = endpoint[0] if endpoint else None
                city = case['records'][case['level']-1]['city']
                if identical:
                    target_anchor, target_reason = exact_pre_city(raw, city, offsets, len(prompt['rendered_prompt']), parsed['reasoning_end'])
                else:
                    target_reason = 'original_token_ids_not_reproduced'
                targeted_enc = encode_ids(original[:target_anchor['prefix_length']]) if target_anchor else None
            else:
                final_enc, targeted_enc, endpoint_reason = enc, enc, None
                target_anchor = {'anchor': 'nonthinking_answer_query', 'prefix_length': len(enc.input_ids), 'query_position': enc.query_position}
            final_attention = attention(final_enc, spans, trace_pos, case['level']) if final_enc else []
            target_attention = (final_attention if mode == 'nonthinking' else attention(targeted_enc, spans, [], case['level'])) if targeted_enc else []
            write_json(dest / 'prompt.json', prompt)
            write_json(dest / 'generation.json', generation)
            write_json(dest / 'retrieval.json', {'original_ids_reproduced': identical, 'trace_parser': parsed,
                'final_attention': final_attention, 'target_attention': target_attention,
                'final_prefix_ids': list(final_enc.input_ids) if final_enc else None,
                'target_prefix_ids': list(targeted_enc.input_ids) if targeted_enc else None,
                'target_anchor': target_anchor, 'target_unavailable_reason': target_reason,
                'final_unavailable_reason': endpoint_reason, 'trace_record_sites': len(trace_pos)})
            write_json(dest / 'complete.json', {'status': 'PASS', 'elapsed_seconds': time.perf_counter()-t,
                'files': {n:sha256(dest/n) for n in ('prompt.json','generation.json','retrieval.json')}})
            print('capture', args.model, mode, case['case_id'], generation['correct'], flush=True)

    # Seed-equal discovery ranking, frozen before any confirmation intervention.
    selections = {}
    for mode in cfg['modes']:
        scores = defaultdict(lambda: defaultdict(list))
        for case in cases:
            if case['split'] != 'discovery':
                continue
            data = json.loads((out/'captures'/mode/case['case_id']/'retrieval.json').read_text(encoding='utf-8'))
            if mode == 'native_thinking' and data['trace_record_sites'] == 0:
                continue
            for row in data['final_attention']:
                scores[(row['layer'],row['head'])][case['seed']].append(row['trace_broad' if mode=='native_thinking' else 'prompt_broad']['score'])
        means = {h: sum(sum(v)/len(v) for v in seeds.values())/len(seeds) for h,seeds in scores.items()}
        if not means:
            selections[mode] = {'status':'unavailable_no_discovery_sites'}
            continue
        ranking = sorted(means, key=lambda h:(-means[h],h))
        bank = select_broad(ranking, widths, cfg['broad_bank_sizes'][args.model])
        selections[mode] = {'status':'PASS', 'heads':bank,
            'random_heads':[random_bank(bank,widths,7000+r) for r in range(3)],
            'ranking':[{'head':h,'score':means[h], 'seeds':len(scores[h])} for h in ranking]}
    write_json(out/'frozen_selection.json', {'broad':selections,'targeted':targeted,'targeted_random':targeted_controls})
    for mode in cfg['modes']:
        for case in cases:
            if case['split'] != 'confirmation':
                continue
            dest = out/'causal'/mode/(case['case_id']+'.json')
            if dest.exists():
                continue
            t=time.perf_counter()
            data=json.loads((out/'captures'/mode/case['case_id']/'retrieval.json').read_text(encoding='utf-8'))
            results={}
            for assay,key in [('broad','final_prefix_ids'),('targeted','target_prefix_ids')]:
                if not data[key] or (assay=='broad' and selections[mode]['status']!='PASS'):
                    results[assay]={'status':'unavailable'}
                    continue
                prefix=encode_ids(data[key])
                bank=selections[mode]['heads'] if assay=='broad' else targeted
                controls=selections[mode]['random_heads'] if assay=='broad' else targeted_controls
                arms=[]
                for name,heads in [('clean',[]) ,('selected',bank)]+[(f'random_{i}',c) for i,c in enumerate(controls)]:
                    generated=generate_with_head_ablation(model,tok,adapter,prefix,heads,scope='answer_query',max_new_tokens=64)
                    if assay=='targeted' and mode=='native_thinking':
                        score={'correct':city_correct(generated['completion_text_raw'],case['records'][case['level']-1]['city'])}
                    else:
                        score=summarize_generation(generated,case,mode=mode,prefixed=True)
                    arms.append({'name':name,'heads':heads,'score':score,'generation':generated})
                results[assay]={'status':'PASS','arms':arms}
            write_json(dest,{'case_id':case['case_id'],'seed':case['seed'],'k':case['level'],'mode':mode,
                              'assays':results,'elapsed_seconds':time.perf_counter()-t})
            print('causal',args.model,mode,case['case_id'],flush=True)
    write_json(out/'complete.json',{'status':'PASS','captures':len(cases)*2,
        'causal_cases':sum(c['split']=='confirmation' for c in cases)*2,
        'elapsed_seconds':time.perf_counter()-started,'canary':args.canary})


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--freeze-source',type=Path)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--model')
    p.add_argument('--canary',action='store_true')
    args=p.parse_args()
    if args.freeze_source:
        freeze(args.freeze_source,args.frozen)
    else:
        execute(args)
