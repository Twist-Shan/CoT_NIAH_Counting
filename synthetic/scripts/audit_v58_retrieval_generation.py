"""Reproduce frozen ablations and separate answer syntax from count readout.

Adds new audit artifacts only. Original results, heads and parsers are preserved.
Run with the original repo-v58 runtime and the original v58 run directory.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
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
import run_v58_commit_query as base
from run_v58_alignment_supplement import free_running_rows
from run_v22_free_running_topk import _free_running_rows as old_free_running_rows
from run_v22_free_running_topk import _local_attention_edit, _parse_generation
from v58_alignment_core import make_record, rank_discovery


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def canonical(heads):
    return ';'.join(f'L{l}H{h}' for l, h in sorted(heads))


def all_controls(bank, k, n_head):
    selected = [tuple(h) for h in bank[:k]]
    layers = sorted({l for l, h in selected})
    choices = [list(itertools.combinations(
        [(l, h) for h in range(n_head) if (l, h) not in selected],
        sum(ll == l for ll, h in selected))) for l in layers]
    return [list(itertools.chain.from_iterable(parts)) for parts in itertools.product(*choices)]


def after_duplicate_diagnostic(tokens, vocab, count):
    """Only skip contiguous repeated Ans tokens; do not search for a good answer."""
    if '<Ans>' not in tokens:
        return {'duplicate_ans_count': 0, 'duplicate_ans': 0., 'after_duplicate_pred': None,
                'after_duplicate_answered': 0., 'after_duplicate_accuracy': 0.}
    first = tokens.index('<Ans>') + 1
    cursor = first
    while cursor < len(tokens) and tokens[cursor] == '<Ans>':
        cursor += 1
    skipped = cursor - first
    number = []
    while cursor < len(tokens) and tokens[cursor] in vocab.numbers:
        number.append(tokens[cursor])
        cursor += 1
    pred = vocab.decode_number_tokens(number)
    return {'duplicate_ans_count': skipped, 'duplicate_ans': float(skipped > 0),
            'after_duplicate_pred': pred, 'after_duplicate_answered': float(pred is not None),
            'after_duplicate_accuracy': float(pred == count)}


def checkpoint_path(run, mode, step):
    root = run / 'checkpoints/rope' / mode
    path = root / f'step_{step:06d}' / 'checkpoint.pt'
    if not path.exists():
        index = pd.read_csv(root / 'snapshot_index.csv')
        path = root / str(index.loc[index.step.eq(step)].iloc[-1].shard)
    return path


@torch.inference_mode()
def sustained_generation(model, cfg, vocab, examples, heads, batch_size, mode):
    """Ablate every query from the original intervention onset through EOS.

    NT starts at the supplied answer marker. Thinking starts at its first
    generated trace separator (the original targeted retrieval query).
    Recomputing the entire prefix keeps the intervention on all earlier
    post-onset positions as well, rather than restoring old altered states.
    """
    rows = []
    sep_id = vocab.token_to_id['<Sep>']
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        items = [base.render_v20(e, vocab, mode) for e in batch]
        stops = [i.spans.ans_pos + 1 if mode == 'nonthinking' else i.spans.think_pos + 1 for i in items]
        assert len(set(stops)) == 1
        generated = torch.tensor([i.input_ids[:s] for i, s in zip(items, stops)], device=cfg.device)
        done = torch.zeros(len(batch), dtype=torch.bool, device=cfg.device)
        onsets = [i.spans.ans_pos if mode == 'nonthinking' else None for i in items]
        max_new = 4 if mode == 'nonthinking' else cfg.max_render_len - stops[0] + 2
        for _ in range(max_new):  # Identical original generation cap.
            positions = []
            for j, row in enumerate(generated):
                if mode == 'thinking' and onsets[j] is None:
                    candidates = [p for p in torch.nonzero(row.eq(sep_id)).flatten().tolist() if p >= stops[j]]
                    if candidates:
                        onsets[j] = candidates[0]
                ps = list(range(onsets[j], generated.shape[1])) if onsets[j] is not None else []
                assert not ps or ps[0] >= (stops[j] - 1 if mode == 'nonthinking' else stops[j])
                assert not ps or ps[-1] == generated.shape[1] - 1
                positions.append(ps)
            ctx = _local_attention_edit(model, heads, positions) if heads else contextlib.nullcontext()
            with ctx:
                next_ids = model(input_ids=generated).logits[:, -1].argmax(-1)
            next_ids = torch.where(done, torch.full_like(next_ids, vocab.eos_id), next_ids)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            done |= next_ids.eq(vocab.eos_id)
            if bool(done.all()):
                break
        for e, seq, onset in zip(batch, generated.cpu().tolist(), onsets):
            tokens = vocab.decode(seq)
            rows.append({'mode': mode, 'count': e.count, 'prompt_sha256': e.prompt_sha256,
                         'intervention_onset': onset, 'hit_token_cap': float('<EOS>' not in tokens),
                         **_parse_generation(tokens, vocab, e, mode)})
    return pd.DataFrame(rows)


@torch.inference_mode()
def original_query_logits(model, cfg, vocab, examples, heads, batch_size):
    """Read the same original-query logits, without a second generated query."""
    rows = []
    counts = list(range(1, 11))
    count_ids = torch.tensor([vocab.token_to_id[vocab.number_token(n)] for n in counts], device=cfg.device)
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        items = [base.render_v20(e, vocab, 'nonthinking') for e in batch]
        qs = [i.spans.ans_pos for i in items]
        assert len(set(qs)) == 1
        ids = torch.tensor([i.input_ids[:q + 1] for i, q in zip(items, qs)], device=cfg.device)
        assert ids[:, -1].eq(vocab.token_to_id['<Ans>']).all()
        ctx = _local_attention_edit(model, heads, [[q] for q in qs]) if heads else contextlib.nullcontext()
        with ctx:
            logits = model(input_ids=ids).logits[:, -1].float()
        assert torch.isfinite(logits).all()
        numeric = logits[:, count_ids]
        p_numeric = (numeric.logsumexp(-1) - logits.logsumexp(-1)).exp()
        for j, e in enumerate(batch):
            pred = int(numeric[j].argmax()) + 1
            target = int(e.count) - 1
            alternatives = numeric[j].clone()
            alternatives[target] = -float('inf')
            top_id = int(logits[j].argmax())
            row = {'key': e.prompt_sha256, 'count': e.count,
                   'full_vocab_token': vocab.decode([top_id])[0],
                   'full_vocab_accuracy': float(top_id == int(count_ids[target])),
                   'p_numeric_1to10': float(p_numeric[j]),
                   'count_restricted_pred': pred,
                   'count_restricted_accuracy': float(pred == e.count),
                   'count_restricted_margin': float(numeric[j, target] - alternatives.max())}
            row.update({f'count_logit_{n}': float(numeric[j, n - 1]) for n in counts})
            rows.append(row)
    return pd.DataFrame(rows)


@torch.inference_mode()
def attention_diagnostic(model, vocab, records, batch_size):
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        output = base.forward(model, vocab, [r['seq'] for r in batch], attention=True)
        for j, r in enumerate(batch):
            for l, a in enumerate(output.attentions, 1):
                for h in range(a.shape[1]):
                    w = a[j, h, r['q']].float()
                    maxp = int(w.argmax())
                    rows.append({'key': r['key'], 'layer': l, 'head': h,
                                 'answer_self_mass': float(w[r['q']]),
                                 'target_mass': float(w[r['needles']].sum()),
                                 'bos_mass': float(w[0]),
                                 'top_key_position': maxp,
                                 'top_key_token': vocab.decode([r['seq'][maxp]])[0]})
    return pd.DataFrame(rows)


def normalize_tokens(s):
    tokens = s.split()
    # Batch-dependent EOS padding is not generated behavior after the first EOS.
    return ' '.join(tokens[:tokens.index('<EOS>') + 1]) if '<EOS>' in tokens else s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260908)
    torch.backends.cuda.matmul.allow_tf32 = False
    align = run / 'analysis/v58_alignment_supplement_20260905'
    registry = pd.read_csv(align / 'input_registry.csv')
    assert len(registry) == registry.key.nunique() == 300
    assert not registry.duplicated(['set_id', 'corpus_start']).any()
    assert registry.groupby(['split', 'count']).size().to_dict() == {
        (s, n): size for s, size in [('discovery', 20), ('confirmation', 10)] for n in range(1, 11)}
    examples_path = run / 'analysis/behavior_confirmation_v58/examples.jsonl'
    lookup = {e.prompt_sha256: e for e in [base.example_from_dict(json.loads(line))
              for line in examples_path.read_text().splitlines() if line.strip()]}
    chosen = [(lookup[r.key], r.split, r.block) for r in registry.itertuples()]
    confirmation = [e for e, split, block in chosen if split == 'confirmation']
    assert [e.count for e in confirmation] == registry.loc[registry.split.eq('confirmation'), 'count'].tolist()
    plans = {}
    protected = [examples_path, align / 'input_registry.csv']
    for mode in ['nonthinking', 'thinking']:
        site_path = align / mode / 'frozen_sites.json'
        sites = json.loads(site_path.read_text())
        bank = [tuple(h) for h in sites['role_bank']]
        assert len(bank) == len(set(bank)) == 4
        arms = [('clean', 0, 0, [])]
        for k in [1, 2, 4]:
            arms += [('selected', k, 0, bank[:k])]
            arms += [('control', k, i, hs) for i, hs in enumerate(all_controls(bank, k, 8))]
        plans[mode] = {'frozen_sites': sites, 'arms': arms}
        protected += [site_path, align / mode / 'ablation.csv']
    hashes = {str(p): digest(p) for p in protected}
    protocol = {
        'created_unix': time.time(), 'checkpoint_step': 10000, 'training': False,
        'ranking': 'recompute for validation only; use existing frozen heads for every arm',
        'inputs': 'same 200 discovery and 100 confirmation prompts, no outcome filtering',
        'original': 'original free_running_rows, unchanged strict parser, 4-token NT cap',
        'sustained': 'NT: every query from original Ans inclusive; Thinking: every query from first generated trace Sep inclusive; remain active through EOS, including later answer and duplicate Ans queries. Same original generation cap.',
        'count_restricted': 'NT only: argmax over counts 1..10 at original Ans query; diagnostic, not free-generation accuracy',
        'skip_duplicate': 'skip only contiguous Ans repeats before parsing number; diagnostic, not primary scoring',
        'controls': 'all disjoint layer-count-matched sets; also retain original 3-set subset for replication',
        'plans': plans, 'protected_sha256': hashes,
        'script_sha256': digest(Path(__file__)), 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(),
    }
    write_json(out / 'protocol.json', protocol)
    registry.to_csv(out / 'input_registry.csv', index=False)
    validations = {}
    for mode in ['nonthinking', 'thinking']:
        dest = out / mode
        dest.mkdir()
        cp = checkpoint_path(run, mode, 10000)
        expected_sha = json.loads((align / mode / 'manifest.json').read_text())['checkpoint_sha256']
        actual_sha = digest(cp)
        assert actual_sha == expected_sha, ('checkpoint changed', mode)
        cfg, vocab, _, _, model = base.load_v20_checkpoint_model(run, 'rope', mode, step=10000, device='cuda')
        assert cfg.version == 'v58' and cfg.n_layer == 4 and cfg.n_head == 8
        records = [make_record(e, vocab, mode, split, block) for e, split, block in chosen]
        ranking = rank_discovery(model, vocab, [r for r in records if r['split'] == 'discovery'], args.batch_size)
        frozen = plans[mode]['frozen_sites']['ranking']
        role = 'broad' if mode == 'nonthinking' else 'targeted'
        top_match = [tuple(r[:2]) for r in ranking[role][:4]] == [tuple(r[:2]) for r in frozen[role][:4]]
        assert top_match, ('discovery head order changed', mode)
        errors = []
        for name in ['broad', 'targeted']:
            oldmap = {tuple(r[:2]): r[2] for r in frozen[name]}
            errors.extend(abs(r[2] - oldmap[tuple(r[:2])]) for r in ranking[name])
        write_json(dest / 'recomputed_ranking.json', ranking)
        trials, local = [], []
        for arm, k, repeat, heads in plans[mode]['arms']:
            tags = {'arm': arm, 'top_k': k, 'repeat': repeat, 'heads': canonical(heads)}
            data = free_running_rows(model, cfg, vocab, confirmation, mode=mode, heads=heads, batch_size=args.batch_size)
            sustained = sustained_generation(model, cfg, vocab, confirmation, heads, args.batch_size, mode)
            for protocol_name, frame in [('original', data), ('sustained', sustained)]:
                for row in frame.to_dict('records'):
                    row.update(tags, protocol=protocol_name)
                    row.update(after_duplicate_diagnostic(row['generated_tokens'].split(), vocab, row['count']))
                    trials.append(row)
            if mode == 'nonthinking':
                data = original_query_logits(model, cfg, vocab, confirmation, heads, args.batch_size)
                for row in data.to_dict('records'):
                    local.append(dict(row, **tags))
            print(mode, arm, k, repeat, tags['heads'], 'complete', flush=True)
            pd.DataFrame(trials).to_csv(dest / 'generation_trials.csv', index=False)
        frame = pd.DataFrame(trials)
        metrics = ['ar_accuracy', 'ar_answered', 'duplicate_ans', 'after_duplicate_answered',
                   'after_duplicate_accuracy', 'trace_exact']
        frame.groupby(['protocol', 'arm', 'top_k', 'repeat', 'heads'], dropna=False)[metrics].mean().reset_index().to_csv(dest / 'generation_summary.csv', index=False)
        original = frame.loc[frame.protocol.eq('original')].copy()
        sustained_clean = frame.loc[frame.protocol.eq('sustained') & frame.arm.eq('clean')].set_index('prompt_sha256')
        original_clean = original.loc[original.arm.eq('clean')].set_index('prompt_sha256')
        assert sustained_clean.generated_tokens.map(normalize_tokens).eq(original_clean.generated_tokens.reindex(sustained_clean.index).map(normalize_tokens)).all(), 'Clean generators disagree'
        assert len(original) == original[['prompt_sha256', 'heads']].drop_duplicates().shape[0]
        archived = pd.read_csv(align / mode / 'ablation.csv')
        archived['heads'] = archived.heads.map(lambda s: canonical(json.loads(s)))
        matched = archived.merge(original, on=['prompt_sha256', 'heads'], suffixes=('_archived', '_rerun'), validate='one_to_one')
        assert len(matched) == len(archived) == 1100
        numerical = ['ar_accuracy', 'ar_answered', 'ar_pred_count'] + (['trace_exact'] if mode == 'thinking' else [])
        mismatches = {c: int((matched[c + '_archived'].fillna(-999) != matched[c + '_rerun'].fillna(-999)).sum()) for c in numerical}
        token_mismatch = matched.generated_tokens_archived.map(normalize_tokens) != matched.generated_tokens_rerun.map(normalize_tokens)
        matched.loc[token_mismatch].to_csv(dest / 'replication_mismatches.csv', index=False)
        validations[mode] = {'checkpoint_sha256': actual_sha, 'checkpoint_matches_archive': True,
                             'ranking_top4_matches': top_match, 'ranking_max_abs_score_error': max(errors),
                             'archived_rows_matched': len(matched), 'metric_mismatches': mismatches,
                             'token_mismatches_before_first_eos': int(token_mismatch.sum()),
                             'all_control_sets_per_k': {str(k): len(all_controls(plans[mode]['frozen_sites']['role_bank'], k, 8)) for k in [1, 2, 4]}}
        if local:
            logits = pd.DataFrame(local)
            logits.to_csv(dest / 'original_query_logits.csv', index=False)
            logits.groupby(['arm', 'top_k', 'repeat', 'heads'], dropna=False)[['full_vocab_accuracy', 'p_numeric_1to10', 'count_restricted_accuracy', 'count_restricted_margin']].mean().reset_index().to_csv(dest / 'original_query_summary.csv', index=False)
            for _, f in original.groupby('heads'):
                q = logits.loc[logits.heads.eq(f.heads.iloc[0])].set_index('key')
                expected = f.set_index('prompt_sha256').generated_tokens.map(lambda s: s.split()[s.split().index('<Ans>') + 1])
                assert q.full_vocab_token.reindex(expected.index).eq(expected).all(), 'First-step logits disagree with free generation'
            attention_diagnostic(model, vocab, [r for r in records if r['split'] == 'confirmation'], args.batch_size).to_csv(dest / 'clean_answer_attention.csv', index=False)
            # Frozen two-head control containing L1H2: same logits across batch sizes.
            hs = [tuple(x) for x in plans[mode]['frozen_sites']['controls']['2'][0]]
            small = original_query_logits(model, cfg, vocab, confirmation[:8], hs, 1)
            large = original_query_logits(model, cfg, vocab, confirmation[:8], hs, 8)
            cols = [f'count_logit_{n}' for n in range(1, 11)]
            validations[mode]['batch_1_vs_8_max_count_logit_difference'] = float(np.abs(small[cols].values - large[cols].values).max())
            validations[mode]['batch_1_vs_8_tokens_equal'] = bool(small.full_vocab_token.eq(large.full_vocab_token).all())
            # Original implementation vs corrected trace-scope implementation: NT is unchanged.
            old = old_free_running_rows(model, cfg, vocab, confirmation, mode=mode, heads=hs, batch_size=args.batch_size)
            new = original.loc[original.heads.eq(canonical(hs))].set_index('prompt_sha256')
            old = old.set_index('prompt_sha256')
            validations[mode]['old_vs_current_nt_token_mismatches'] = int((old.generated_tokens.map(normalize_tokens) != new.generated_tokens.reindex(old.index).map(normalize_tokens)).sum())
        write_json(out / 'validation.json', validations)
        del model
        torch.cuda.empty_cache()
    assert all(digest(Path(p)) == sha for p, sha in hashes.items())
    failures = {m: v for m, v in validations.items() if v['token_mismatches_before_first_eos'] or any(v['metric_mismatches'].values())}
    write_json(out / 'manifest.json', {'status': 'complete', 'original_files_unchanged': True,
               'replication_exact': not failures, 'validation': validations,
               'files': {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}})
    print('AUDIT COMPLETE', json.dumps(validations), flush=True)


if __name__ == '__main__':
    main()
