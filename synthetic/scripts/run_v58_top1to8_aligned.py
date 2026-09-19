"""Exhaustive layer-matched Top-1..8 ablation at fixed clean query prefixes.

Primary: sustained intervention in both modes, per the user's explicit request.
NT answer-query-only intervention is retained as the original main-study reference.
T uses the final N-1 -> N retrieval query from a frozen clean rollout. N=1 is
reported separately. Missing clean anchors are registered before interventions.
"""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT / 'scripts'):
    sys.path.insert(0, str(directory))
import numpy as np
import pandas as pd
import torch
import audit_v58_retrieval_generation as audit
from freeze_v58_control_policy import matched_control_plan


def trim(tokens):
    return tokens[:tokens.index('<EOS>') + 1] if '<EOS>' in tokens else tokens


def final_query_prefix(tokens, count):
    """Use the Nth Sep before the first trace close/answer/end, without retries."""
    start = tokens.index('<Think>') + 1
    end = next((i for i in range(start, len(tokens))
                if tokens[i] in ('</Think>', '<Ans>', '<EOS>')), len(tokens))
    seps = [i for i in range(start, end) if tokens[i] == '<Sep>']
    if len(seps) < count:
        return None
    q = seps[count - 1]
    trace = tokens[start:q + 1]
    # Syntax, not correct marker identity, determines whether this is a query.
    if any(t != '<Sep>' if i % 2 == 0 else not t.startswith('<CH_')
           for i, t in enumerate(trace)):
        return None
    return tokens[:q + 1]


def marker_outcome(continuation, expected):
    first = None
    offset = None
    for i, token in enumerate(continuation):
        if token in ('</Think>', '<Ans>', '<EOS>'):
            break
        if token.startswith('<CH_'):
            first, offset = token, i
            break
    immediate = continuation[0] if continuation else None
    return {'next_marker_pred': first, 'next_marker_offset': offset,
            'next_marker_identifiable': float(first is not None),
            'next_marker_correct': float(first == expected),
            'immediate_marker_valid': float(bool(immediate and immediate.startswith('<CH_'))),
            'immediate_marker_correct': float(immediate == expected)}


def freeze(args):
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    align = args.run_dir / 'analysis/v58_alignment_supplement_20260905'
    previous = args.run_dir / 'analysis/v58_retrieval_sustained_cap64_20260908'
    registry = pd.read_csv(align / 'input_registry.csv')
    support = registry.loc[registry.split.eq('confirmation')]
    assert support.groupby('count').size().to_dict() == {k: 10 for k in range(1, 11)}
    source = args.run_dir / 'analysis/behavior_confirmation_v58/examples.jsonl'
    lookup = {e.prompt_sha256: e for e in [audit.base.example_from_dict(json.loads(s))
              for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]}
    from synthetic_counting_v20.data import V20Vocab
    vocab = V20Vocab.load(args.run_dir / 'vocab.json')
    examples = [lookup[k] for k in support.key]
    paths = [source, align / 'input_registry.csv', Path(__file__), Path(audit.__file__),
             ROOT / 'scripts/freeze_v58_control_policy.py']
    anchors, prefix_rows, arms, plans = [], [], [], {}
    for mode, score in [('thinking', 'targeted'), ('nonthinking', 'broad')]:
        site_path = align / mode / 'frozen_sites.json'
        clean_path = previous / mode / 'generation_trials.csv'
        paths.extend([site_path, clean_path])
        frozen = json.loads(site_path.read_text(encoding='utf-8'))
        ranking = [row[:2] for row in frozen['ranking'][score]]
        assert len(ranking) == len({tuple(h) for h in ranking}) == 32
        plans[mode] = {}
        for k in range(1, 9):
            registered = frozen['controls'].get(str(k), [])
            plan = matched_control_plan(ranking[:k], registered=registered, exhaustive=True)
            plans[mode][str(k)] = plan
            for arm, repeat, heads in [('selected', 0, plan['selected'])] + [
                ('control', j, c['heads']) for j, c in enumerate(plan['controls'])]:
                arms.append({'mode': mode, 'arm': arm, 'top_k': k, 'repeat': repeat,
                             'heads': heads, 'heads_label': audit.canonical(heads),
                             'overlap': len(set(map(tuple, heads)) & set(map(tuple, plan['selected'])))})
        clean = pd.read_csv(clean_path)
        clean = clean.loc[clean.arm.eq('clean')].set_index('prompt_sha256')
        for e in examples:
            item = audit.base.render_v20(e, vocab, mode)
            baseline = trim(clean.loc[e.prompt_sha256, 'generated_tokens'].split())
            prefix = (item.tokens[:item.spans.ans_pos + 1] if mode == 'nonthinking'
                      else final_query_prefix(baseline, e.count))
            marker = item.tokens[item.spans.trace_marker_positions[-1]] if mode == 'thinking' else ''
            record = {'mode': mode, 'key': e.prompt_sha256, 'count': e.count,
                      'anchor_available': prefix is not None,
                      'primary_eligible': prefix is not None and (mode == 'nonthinking' or e.count >= 2),
                      'boundary_count1': mode == 'thinking' and e.count == 1,
                      'reason': 'available' if prefix else 'clean_missing_or_malformed_final_query',
                      'query_position': len(prefix) - 1 if prefix else None,
                      'expected_marker': marker}
            anchors.append(record)
            if prefix is not None:
                assert baseline[:len(prefix)] == prefix
                prefix_rows.append({**record, 'prefix_tokens': prefix,
                                    'clean_full_tokens': baseline})
    pd.DataFrame(anchors).to_csv(out / 'anchor_registry.csv', index=False)
    (out / 'prefixes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in prefix_rows), encoding='utf-8')
    # Runtime reconstructs the same examples from the hash-locked original source.
    checkpoint_hashes = {m: audit.digest(audit.checkpoint_path(args.run_dir, m, 10000))
                         for m in plans}
    assert all(checkpoint_hashes[m] == json.loads((align / m / 'manifest.json').read_text())['checkpoint_sha256']
               for m in plans)
    protocol = {'status': 'frozen_before_new_inference', 'created_unix': time.time(),
                'source_sha256': {str(p.resolve()): audit.digest(p) for p in paths},
                'checkpoint_sha256': checkpoint_hashes, 'checkpoint_step': 10000,
                'examples_source': str(source.resolve()), 'support_keys': support.key.tolist(),
                'prefixes_sha256': audit.digest(out / 'prefixes.jsonl'),
                'anchor_registry_sha256': audit.digest(out / 'anchor_registry.csv'),
                'batch_size': args.batch_size, 'budget': args.budget,
                'control_policy': 'exhaustive_same_layer_disjoint_first_minimum_necessary_overlap',
                'plans': plans, 'arms': arms,
                'scopes': {'thinking': ['sustained'], 'nonthinking': ['sustained', 'answer_query_only']},
                'decoding': 'full-vocabulary greedy, EOS or 64 new tokens, no candidate constraint',
                'prefix_policy': 'Frozen clean natural generation to final retrieval Sep; no correctness filtering; unavailable clean query registered before intervention.',
                'primary_support': 'Thinking: available final query and count>=2; NT: all 100 inputs. Count1 Thinking separately.',
                'main_protocol_difference': 'NT sustained is the user-requested extension; answer_query_only reproduces the original main-study timing.',
                'targeted_metric': 'First CH token after query and before trace close/Ans/EOS; immediate-token accuracy recorded separately. Never search for a later correct marker.',
                'missing_outputs': 'Retained as failures for the relevant accuracy. Numeric shift is missing when clean or treated count is unparsable; report support.',
                'no_outcome_selection': True}
    audit.write_json(out / 'protocol.json', protocol)
    print('FROZEN', {m: {'controls': sum(len(p['controls']) for p in ps.values()),
                        'prefixes': sum(r['mode'] == m for r in prefix_rows)} for m, ps in plans.items()}, flush=True)
    return protocol


@torch.inference_mode()
def generate(model, cfg, vocab, records, examples, heads, scope, batch_size, budget, verify=False):
    rows, checks = [], {'hook_calls': 0, 'unchanged_slices_verified': 0}
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        seqs = [vocab.encode(r['prefix_tokens']) for r in batch]
        prefix_lengths = list(map(len, seqs))
        first_info = [{} for r in batch]
        for step in range(budget):
            active = [s[-1] != vocab.eos_id for s in seqs]
            if not any(active):
                break
            length = max(map(len, seqs))
            if length >= cfg.n_positions:
                raise ValueError('Configured token budget exceeds positional capacity')
            ids = torch.full((len(batch), length), vocab.pad_id, dtype=torch.long, device=cfg.device)
            mask = torch.zeros_like(ids)
            for j, seq in enumerate(seqs):
                ids[j, :len(seq)] = torch.tensor(seq, device=cfg.device)
                mask[j, :len(seq)] = 1
            positions = [[n - 1] if scope == 'answer_query_only' else list(range(n - 1, len(s)))
                         for n, s in zip(prefix_lengths, seqs)]
            ctx = audit._local_attention_edit(model, heads, positions) if heads else contextlib.nullcontext()
            before, handles = {}, []
            probe = verify and start == 0 and bool(heads)
            if probe:
                for layer in sorted({l for l, h in heads}):
                    def capture(module, args, layer=layer):
                        before[layer] = args[0].detach().clone()
                    handles.append(model.layers[layer - 1].attention.output.register_forward_pre_hook(capture))
            try:
                with ctx:
                    if probe:
                        for layer in sorted({l for l, h in heads}):
                            def check(module, args, layer=layer):
                                expected = before[layer].clone()
                                width = cfg.n_embd // cfg.n_head
                                for j, ps in enumerate(positions):
                                    for ll, h in heads:
                                        if ll == layer:
                                            expected[j, ps, h * width:(h + 1) * width] = 0
                                assert torch.equal(expected, args[0]), 'Ablation changed an unintended slice'
                                checks['hook_calls'] += 1
                                checks['unchanged_slices_verified'] += 1
                            handles.append(model.layers[layer - 1].attention.output.register_forward_pre_hook(check))
                    output = model(input_ids=ids, attention_mask=mask).logits
                    logits = output[torch.arange(len(batch), device=cfg.device),
                                    torch.tensor([len(s)-1 for s in seqs], device=cfg.device)]
                    predictions = logits.argmax(-1).cpu().tolist()
                    if step == 0:
                        for j, r in enumerate(batch):
                            token = vocab.id_to_token[predictions[j]]
                            first_info[j] = {'first_token': token}
            finally:
                for handle in handles:
                    handle.remove()
            for j, is_active in enumerate(active):
                if is_active:
                    seqs[j].append(predictions[j])
        for r, seq, n, first in zip(batch, seqs, prefix_lengths, first_info):
            tokens = trim(vocab.decode(seq))
            parsed = audit._parse_generation(tokens, vocab, examples[r['key']], r['mode'])
            continuation = tokens[n:]
            marker = marker_outcome(continuation, r['expected_marker']) if r['mode'] == 'thinking' else {}
            pred = parsed['ar_pred_count']
            rows.append({k: v for k, v in r.items() if k not in ('prefix_tokens','clean_full_tokens')} |
                        {'scope': scope, **parsed, **marker, **first,
                         'eos_reached': float(bool(continuation and continuation[-1] == '<EOS>')),
                         'new_tokens': len(continuation),
                         'generation_capped': float(len(continuation) >= budget and continuation[-1] != '<EOS>'),
                         'answer_and_eos_correct': parsed['ar_accuracy'] * float(tokens[-1] == '<EOS>'),
                         'joint_marker_and_count_failure': float(marker.get('next_marker_correct', 1) == 0 and parsed['ar_accuracy'] == 0) if marker else np.nan})
    assert all(not b.attention.output._forward_pre_hooks for b in model.layers)
    return pd.DataFrame(rows), checks


def run(args, protocol):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.output
    assert audit.digest(out / 'prefixes.jsonl') == protocol['prefixes_sha256']
    assert audit.digest(out / 'anchor_registry.csv') == protocol['anchor_registry_sha256']
    assert all(audit.digest(Path(p)) == h for p, h in protocol['source_sha256'].items())
    records = [json.loads(s) for s in (out / 'prefixes.jsonl').read_text().splitlines()]
    source = Path(protocol['examples_source'])
    examples = {e.prompt_sha256: e for e in [audit.base.example_from_dict(json.loads(s))
                for s in source.read_text().splitlines() if s.strip()]}
    validation = {}
    for mode in ['thinking', 'nonthinking']:
        cp = audit.checkpoint_path(args.run_dir, mode, 10000)
        assert audit.digest(cp) == protocol['checkpoint_sha256'][mode]
        cfg, vocab, _, _, model = audit.base.load_v20_checkpoint_model(args.run_dir, 'rope', mode, step=10000, device='cuda')
        model.eval()
        batch = [r for r in records if r['mode'] == mode]
        dest = out / mode
        (dest / 'arms').mkdir(parents=True, exist_ok=True)
        clean, _ = generate(model,cfg,vocab,batch,examples,[],'sustained',args.batch_size,args.budget)
        clean_map = clean.set_index('key')
        assert all(audit.normalize_tokens(clean_map.loc[r['key'],'generated_tokens']) == ' '.join(r['clean_full_tokens']) for r in batch), 'Clean prefix replay differs from source'
        clean['arm'], clean['top_k'], clean['repeat'], clean['heads'], clean['overlap'] = 'clean', 0, 0, '', 0
        clean.to_csv(dest / 'clean.csv', index=False)
        # Compare unequal right-padded batches with individually evaluated sequences.
        small = [next(r for r in batch if r['count'] == n) for n in range(1, 11)]
        probe_heads = protocol['plans'][mode]['4']['selected']
        alone, _ = generate(model,cfg,vocab,small,examples,probe_heads,'sustained',1,args.budget)
        together, checks = generate(model,cfg,vocab,small,examples,probe_heads,'sustained',len(small),args.budget,verify=True)
        assert alone.generated_tokens.tolist() == together.generated_tokens.tolist(), 'Batch padding changes generated trajectory'
        control_heads = protocol['plans'][mode]['8']['controls'][0]['heads']
        _, multi_checks = generate(model,cfg,vocab,small,examples,control_heads,'sustained',len(small),args.budget,verify=True)
        validation[mode] = {'checkpoint_matches': True,'clean_prefix_replay_exact': True,
                            'batch1_vs_variable_length_batch_exact': True,'hook_checks': checks,
                            'multilayer_hook_checks': multi_checks,'inputs':len(batch),
                            'primary_inputs':sum(r['primary_eligible'] for r in batch)}
        audit.write_json(out/'validation.json',validation)
        arms = [a for a in protocol['arms'] if a['mode'] == mode]
        arms.sort(key=lambda a:(a['arm'] != 'selected',a['top_k'],a['repeat']))
        for scope in protocol['scopes'][mode]:
            for i, arm in enumerate(arms):
                path = dest/'arms'/f"{scope}_{arm['arm']}_k{arm['top_k']}_r{arm['repeat']}.csv"
                started = time.time()
                if path.exists():
                    f = pd.read_csv(path)
                    assert len(f) == len(batch) and set(f.key) == set(clean.key)
                else:
                    f, _ = generate(model,cfg,vocab,batch,examples,arm['heads'],scope,args.batch_size,args.budget)
                    for col in ['arm','top_k','repeat','overlap']:
                        f[col] = arm[col]
                    f['heads'] = arm['heads_label']
                    f['clean_accuracy'] = f.key.map(clean_map.ar_accuracy)
                    f['clean_pred_count'] = f.key.map(clean_map.ar_pred_count)
                    f['absolute_count_shift'] = (pd.to_numeric(f.ar_pred_count) - pd.to_numeric(f.clean_pred_count)).abs()
                    f['normalized_count_shift'] = f.absolute_count_shift / f['count']
                    f.to_csv(path,index=False)
                primary=f.loc[f.primary_eligible]
                if arm['arm']=='selected' or (i+1)%20==0 or i==len(arms)-1:
                    print(mode,scope,i+1,'/',len(arms),arm['arm'],'K',arm['top_k'],'r',arm['repeat'],
                          'answer',round(primary.ar_accuracy.mean(),4),
                          'marker',round(primary.next_marker_correct.mean(),4) if mode=='thinking' else '-',
                          'seconds',round(time.time()-started,2),flush=True)
                audit.write_json(out/'progress.json',{'mode':mode,'scope':scope,'completed':i+1,'scope_total':len(arms),'last_file':str(path),'updated_unix':time.time()})
        validation[mode]['completed_arms_per_scope'] = len(arms)
        audit.write_json(out/'validation.json',validation)
        del model
        torch.cuda.empty_cache()
    assert all(audit.digest(Path(p)) == h for p,h in protocol['source_sha256'].items())
    audit.write_json(out/'manifest.json',{'status':'complete','original_files_unchanged':True,
        'validation':validation,'files':{str(p.relative_to(out)):audit.digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}})
    print('TOP1TO8 AUDIT COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=100)
    p.add_argument('--budget',type=int,default=64)
    p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    protocol=json.loads((args.output/'protocol.json').read_text()) if args.resume else freeze(args)
    assert args.batch_size==protocol['batch_size'] and args.budget==protocol['budget']
    if not args.freeze_only:
        run(args,protocol)


if __name__=='__main__':
    main()
