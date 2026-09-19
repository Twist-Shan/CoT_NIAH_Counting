"""Localize frozen retrieval-control damage by query stage.

Uses the same checkpoint, 100 inputs, head banks, controls and strict parser.
No model training, head re-selection, outcome filtering or significance tests.
"""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    sys.path.insert(0, str(path))
import numpy as np
import pandas as pd
import torch
import audit_v58_retrieval_generation as audit

ATOMS = ['retrieval', 'trace_other', 'bridge', 'answer_marker', 'post_answer']
SCOPES = {
    'nonthinking': ['all', 'original_query', 'answer_marker', 'post_answer'],
    'thinking': ['all', 'trace_phase', 'output_phase', 'retrieval', 'trace_other',
                 'bridge', 'answer_marker', 'post_answer'],
}


def positions_for(tokens, mode, stop, ids):
    """Partition actual queries; prompt Sep is always outside the intervention."""
    n = len(tokens)
    groups = {key: [] for key in ATOMS}
    if mode == 'nonthinking':
        onset, answer, end_trace = stop - 1, stop - 1, stop - 1
    else:
        seps = [p for p in range(stop, n) if tokens[p] == ids['sep']]
        if not seps:
            return {**groups, 'all': [], 'trace_phase': [], 'output_phase': [], 'original_query': []}
        onset = seps[0]
        closes = [p for p in range(onset, n) if tokens[p] == ids['close']]
        answers = [p for p in range(onset, n) if tokens[p] == ids['ans']]
        answer = answers[0] if answers else n
        end_trace = min(closes[0] if closes else n, answer)
    for p in range(onset, n):
        if p < end_trace:
            group = 'retrieval' if tokens[p] == ids['sep'] else 'trace_other'
        elif p < answer:
            group = 'bridge'
        else:
            group = 'answer_marker' if tokens[p] == ids['ans'] else 'post_answer'
        groups[group].append(p)
    groups['all'] = list(range(onset, n))
    groups['trace_phase'] = sorted(groups['retrieval'] + groups['trace_other'])
    groups['output_phase'] = sorted(groups['bridge'] + groups['answer_marker'] + groups['post_answer'])
    groups['original_query'] = [stop - 1] if mode == 'nonthinking' else []
    assert sorted(p for k in ATOMS for p in groups[k]) == groups['all']
    assert not groups['all'] or min(groups['all']) >= (stop - 1 if mode == 'nonthinking' else stop)
    return groups


def verify_partitions():
    ids = {'sep': 1, 'close': 2, 'ans': 3}
    # Includes a prompt Sep, two items, close, answer, digit and EOS.
    tokens = [8, 1, 8, 9, 1, 7, 1, 6, 2, 3, 5, 0]
    g = positions_for(tokens, 'thinking', 4, ids)
    assert g['retrieval'] == [4, 6] and g['trace_other'] == [5, 7]
    assert g['bridge'] == [8] and g['answer_marker'] == [9] and g['post_answer'] == [10, 11]
    # Repeated answer markers must stay in the answer-marker phase.
    g = positions_for([8, 3, 3, 5, 3], 'nonthinking', 2, ids)
    assert g['answer_marker'] == [1, 2, 4] and g['post_answer'] == [3]
    assert positions_for([8, 1, 9], 'thinking', 3, ids)['all'] == []
    # Missing close, absent answer and repeated close remain well-defined.
    for seq in [tokens[:8], [8, 1, 8, 9, 1, 7, 3, 5], [8, 1, 8, 9, 1, 7, 2, 2]]:
        previous = {k: set() for k in ATOMS}
        for length in range(4, len(seq) + 1):
            g = positions_for(seq[:length], 'thinking', 4, ids)
            assert all(previous[k].issubset(g[k]) for k in ATOMS), 'Past query labels changed'
            previous = {k: set(g[k]) for k in ATOMS}


def trim(tokens):
    return tokens[:tokens.index('<EOS>') + 1] if '<EOS>' in tokens else tokens


def trace_tokens(tokens):
    if '<Think>' not in tokens:
        return []
    start = tokens.index('<Think>') + 1
    end = tokens.index('</Think>', start) if '</Think>' in tokens[start:] else len(tokens)
    return tokens[start:end]


@torch.inference_mode()
def generate(model, cfg, vocab, examples, heads, mode, scope, batch_size, budget, clean=None,
             verify_hook=False):
    rows, checks = [], {'head_hook_calls': 0, 'masked_query_rows': 0}
    token_ids = {'sep': vocab.token_to_id['<Sep>'], 'close': vocab.token_to_id['</Think>'],
                 'ans': vocab.token_to_id['<Ans>']}
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        items = [audit.base.render_v20(e, vocab, mode) for e in batch]
        stops = [i.spans.ans_pos + 1 if mode == 'nonthinking' else i.spans.think_pos + 1 for i in items]
        assert len(set(stops)) == 1 and stops[0] + budget <= cfg.n_positions
        generated = torch.tensor([i.input_ids[:s] for i, s in zip(items, stops)], device=cfg.device)
        done = torch.zeros(len(batch), dtype=torch.bool, device=cfg.device)
        exposed = [set() for _ in batch]
        for _ in range(budget):
            groups = [positions_for(row, mode, stop, token_ids) for row, stop in zip(generated.cpu().tolist(), stops)]
            positions = [g[scope] if scope != 'clean' else [] for g in groups]
            for j, ps in enumerate(positions):
                if not bool(done[j]):
                    exposed[j].update(ps)
            before, handles = {}, []
            probe = verify_hook and start == 0 and heads
            layers = sorted({l for l, h in heads}) if probe else []
            if probe:
                for layer in layers:
                    def capture(module, args, layer=layer):
                        before[layer] = args[0].detach().clone()
                    handles.append(model.layers[layer - 1].attention.output.register_forward_pre_hook(capture))
            ctx = audit._local_attention_edit(model, heads, positions) if heads else contextlib.nullcontext()
            try:
                with ctx:
                    if probe:
                        for layer in layers:
                            def check(module, args, layer=layer):
                                expected = before[layer].clone()
                                width = cfg.n_embd // cfg.n_head
                                for j, ps in enumerate(positions):
                                    for ll, h in heads:
                                        if ll == layer:
                                            expected[j, ps, h * width:(h + 1) * width] = 0
                                assert torch.equal(expected, args[0]), 'Actual pre-OV mask differs from stage specification'
                                checks['head_hook_calls'] += 1
                                checks['masked_query_rows'] += sum(map(len, positions))
                            handles.append(model.layers[layer - 1].attention.output.register_forward_pre_hook(check))
                    next_ids = model(input_ids=generated).logits[:, -1].argmax(-1)
            finally:
                for h in handles:
                    h.remove()
            next_ids = torch.where(done, torch.full_like(next_ids, vocab.eos_id), next_ids)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            done |= next_ids.eq(vocab.eos_id)
            if bool(done.all()):
                break
        for e, seq, stop, touched in zip(batch, generated.cpu().tolist(), stops, exposed):
            tokens = trim(vocab.decode(seq))
            parsed = audit._parse_generation(tokens, vocab, e, mode)
            eos = float('<EOS>' in tokens)
            row = {'mode': mode, 'key': e.prompt_sha256, 'count': e.count, 'scope': scope,
                   **parsed, **audit.after_duplicate_diagnostic(tokens, vocab, e.count),
                   'eos_reached': eos, 'answer_and_eos_correct': eos * parsed['ar_accuracy'],
                   'new_tokens': len(tokens) - stop, 'unique_masked_queries': len(touched),
                   'intervention_activated': float(bool(touched) and bool(heads))}
            if clean is not None:
                baseline = clean[e.prompt_sha256]['generated_tokens'].split()
                baseline = trim(baseline)
                old, new = baseline[stop:], tokens[stop:]
                difference = next((i for i in range(min(len(old), len(new))) if old[i] != new[i]), None)
                if difference is None and len(old) != len(new):
                    difference = min(len(old), len(new))
                row['first_changed_token'] = difference
                row['clean_trace_preserved'] = float(trace_tokens(tokens) == trace_tokens(baseline)) if mode == 'thinking' else np.nan
                row['first_changed_query_stage'] = ''
                if difference is not None:
                    query = stop + difference - 1
                    gs = positions_for(vocab.encode(tokens[:query + 1]), mode, stop, token_ids)
                    row['first_changed_query_stage'] = next((g for g in ATOMS if query in gs[g]), 'pre_onset')
                # Output-only masks cannot retroactively alter the emitted trace.
                if mode == 'thinking' and scope in ['output_phase', 'bridge', 'answer_marker', 'post_answer']:
                    assert row['clean_trace_preserved'] == 1, ('Output phase changed earlier trace', scope, e.prompt_sha256)
                # Queries after answer-marker inputs cannot change the first count token.
                if mode == 'nonthinking' and scope == 'post_answer':
                    assert tokens[stop] == baseline[stop], 'Post-answer mask changed earlier count token'
            rows.append(row)
    assert all(not block.attention.output._forward_pre_hooks for block in model.layers)
    return pd.DataFrame(rows), checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--budget', type=int, default=64)
    args = p.parse_args()
    verify_partitions()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    align = args.run_dir / 'analysis/v58_alignment_supplement_20260905'
    previous = args.run_dir / 'analysis/v58_retrieval_sustained_cap64_20260908'
    registry = pd.read_csv(align / 'input_registry.csv')
    assert len(registry) == registry.key.nunique() == 300
    support = registry.loc[registry.split.eq('confirmation')]
    assert support.groupby('count').size().to_dict() == {k: 10 for k in range(1, 11)}
    source = args.run_dir / 'analysis/behavior_confirmation_v58/examples.jsonl'
    lookup = {e.prompt_sha256: e for e in [audit.base.example_from_dict(json.loads(s)) for s in source.read_text().splitlines() if s.strip()]}
    examples = [lookup[k] for k in support.key]
    old_plan = json.loads((args.run_dir / 'analysis/v58_retrieval_generation_audit_20260908/protocol.json').read_text())
    frozen_paths = [source, align / 'input_registry.csv']
    for mode in ['nonthinking', 'thinking']:
        frozen_paths += [align / mode / 'frozen_sites.json', previous / mode / 'generation_trials.csv']
    frozen = {str(s): audit.digest(s) for s in frozen_paths}
    audit.write_json(args.output / 'protocol.json', {
        'created_unix': time.time(), 'script_sha256': audit.digest(Path(__file__)),
        'base_runner_sha256': audit.digest(Path(audit.__file__)), 'source_sha256': frozen,
        'sample_keys': examples and [e.prompt_sha256 for e in examples],
        'checkpoint_step': 10000, 'budget': args.budget, 'batch_size': args.batch_size,
        'arms': {m: old_plan['plans'][m]['arms'] for m in SCOPES}, 'scopes': SCOPES,
        'scope_definitions': {
            'all': 'Every query from original NT Ans or first generated T Sep until EOS/budget.',
            'original_query': 'NT original supplied Ans query only, retained on each full-prefix recomputation.',
            'retrieval': 'T Sep-token queries inside generated trace, excluding prompt Sep.',
            'trace_other': 'All other T trace queries, including item characters that predict separators or trace closure.',
            'bridge': 'Queries from first trace close until first Ans, excluding Ans.',
            'answer_marker': 'Every Ans-token query at or after the first Ans, including repeated Ans.',
            'post_answer': 'Other queries at or after first Ans, including count-token queries that predict EOS.',
            'trace_phase': 'retrieval + trace_other',
            'output_phase': 'bridge + answer_marker + post_answer',
        },
        'interpretation': 'Stage-restricted sustained ablation on freely generated trajectories; interactions are not assumed additive. No filtering for successful outputs.',
        'new_head_selection': False, 'new_significance_tests': False,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
    })
    support.to_csv(args.output / 'input_registry.csv', index=False)
    validations = {}
    for mode in ['nonthinking', 'thinking']:
        started = time.time()
        dest = args.output / mode
        dest.mkdir()
        cp = audit.checkpoint_path(args.run_dir, mode, 10000)
        cp_hash = audit.digest(cp)
        assert cp_hash == json.loads((align / mode / 'manifest.json').read_text())['checkpoint_sha256']
        cfg, vocab, _, _, model = audit.base.load_v20_checkpoint_model(args.run_dir, 'rope', mode, step=10000, device='cuda')
        reference = pd.read_csv(previous / mode / 'generation_trials.csv').fillna({'heads': ''})
        clean, _ = generate(model, cfg, vocab, examples, [], mode, 'clean', args.batch_size, args.budget)
        clean['first_changed_token'] = np.nan
        clean['first_changed_query_stage'] = ''
        clean['clean_trace_preserved'] = 1. if mode == 'thinking' else np.nan
        clean['arm'], clean['top_k'], clean['repeat'], clean['heads'] = 'clean', 0, 0, ''
        clean_map = clean.set_index('key').to_dict('index')
        reference_clean = reference.loc[reference.arm.eq('clean')].set_index('prompt_sha256')
        assert clean.set_index('key').generated_tokens.eq(reference_clean.generated_tokens.reindex(clean.key).map(audit.normalize_tokens)).all()
        clean.to_csv(dest / 'generation_trials.csv', index=False)
        frames, hook_checks, schema = [clean], [], list(clean.columns)
        metric_columns = ['ar_accuracy', 'ar_answered', 'trace_exact', 'trace_ordered_marker_accuracy',
                          'trace_marker_count_accuracy', 'trace_closed', 'trace_format_valid', 'eos_reached',
                          'answer_and_eos_correct', 'duplicate_ans', 'after_duplicate_accuracy',
                          'new_tokens', 'unique_masked_queries', 'intervention_activated', 'clean_trace_preserved']
        group_columns = ['arm', 'top_k', 'repeat', 'heads', 'scope']
        summary_frames = [clean.groupby(group_columns, dropna=False)[metric_columns].mean().reset_index()]
        for arm, k, repeat, heads in old_plan['plans'][mode]['arms']:
            if arm == 'clean':
                continue
            heads = [tuple(h) for h in heads]
            head_text = audit.canonical(heads)
            for scope in SCOPES[mode]:
                detail, checked = generate(model, cfg, vocab, examples, heads, mode, scope,
                    args.batch_size, args.budget, clean=clean_map, verify_hook=(k == 4))
                detail['arm'], detail['top_k'], detail['repeat'], detail['heads'] = arm, k, repeat, head_text
                assert len(detail) == detail.key.nunique() == 100
                if scope == 'all':
                    prior = reference.loc[reference.heads.eq(head_text)].set_index('prompt_sha256')
                    current = detail.set_index('key').generated_tokens
                    assert current.eq(prior.generated_tokens.reindex(current.index).map(audit.normalize_tokens)).all(), ('Sustained reproduction mismatch', mode, head_text)
                assert set(detail.columns) == set(schema)
                frames.append(detail)
                detail.reindex(columns=schema).to_csv(dest / 'generation_trials.csv', mode='a', header=False, index=False)
                summary_frames.append(detail.groupby(group_columns, dropna=False)[metric_columns].mean().reset_index())
                pd.concat(summary_frames, ignore_index=True).to_csv(dest / 'generation_summary.csv', index=False)
                if checked['head_hook_calls']:
                    hook_checks.append(dict(mode=mode, arm=arm, k=k, heads=head_text, scope=scope, **checked))
                print(mode, arm, k, repeat, scope, 'answer', detail.ar_accuracy.mean(),
                      'trace', detail.trace_exact.mean(), 'EOS', detail.eos_reached.mean(), flush=True)
        full = pd.concat(frames, ignore_index=True)
        expected = 100 + 26 * len(SCOPES[mode]) * 100
        assert len(full) == expected and not full.duplicated(['key', 'heads', 'scope']).any()
        validations[mode] = {'checkpoint_sha256': cp_hash, 'checkpoint_matches': True,
                             'clean_reproduced': True, 'all_26_sustained_arms_reproduced': True,
                             'rows': len(full), 'condition_input_count': 100,
                             'query_partition_checks': True, 'output_only_preserves_trace': True,
                             'post_answer_preserves_initial_nt_count': True,
                             'hook_checks': hook_checks, 'seconds': time.time() - started}
        audit.write_json(args.output / 'validation.json', validations)
        del model
        torch.cuda.empty_cache()
    assert all(audit.digest(Path(s)) == h for s, h in frozen.items())
    audit.write_json(args.output / 'manifest.json', {'status': 'complete', 'original_files_unchanged': True,
                     'validation': validations,
                     'files': {str(s.relative_to(args.output)): audit.digest(s) for s in args.output.rglob('*') if s.is_file()}})
    print('STAGE AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
