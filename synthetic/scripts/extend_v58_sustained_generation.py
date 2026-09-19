"""Continue every capped audit rollout to a fixed 64-token generation budget.

Uncapped rows are retained exactly. Capped rows resume with all post-onset
queries still ablated. No outcome or head selection is performed.
"""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    sys.path.insert(0, str(p))
import pandas as pd
import torch
import audit_v58_retrieval_generation as audit


@torch.inference_mode()
def extend(model, cfg, vocab, rows, lookup, heads, mode, batch_size, budget):
    result = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        tokens = [r['generated_tokens'].split() for r in batch]
        assert all('<EOS>' not in ts for ts in tokens)
        assert len(set(map(len, tokens))) == 1
        examples = [lookup[r['prompt_sha256']] for r in batch]
        items = [audit.base.render_v20(e, vocab, mode) for e in examples]
        stops = [i.spans.ans_pos + 1 if mode == 'nonthinking' else i.spans.think_pos + 1 for i in items]
        assert len(set(stops)) == 1
        generated = torch.tensor([vocab.encode(ts) for ts in tokens], device=cfg.device)
        original_length = generated.shape[1]
        assert stops[0] + budget <= cfg.n_positions
        onsets = [int(r['intervention_onset']) if pd.notna(r['intervention_onset']) else None for r in batch]
        done = torch.zeros(len(batch), dtype=torch.bool, device=cfg.device)
        for _ in range(budget - (original_length - stops[0])):
            positions = []
            for j, row in enumerate(generated):
                if onsets[j] is None:
                    sep = [p for p in torch.nonzero(row.eq(vocab.token_to_id['<Sep>'])).flatten().tolist() if p >= stops[j]]
                    if sep:
                        onsets[j] = sep[0]
                ps = list(range(onsets[j], generated.shape[1])) if onsets[j] is not None else []
                assert not ps or ps[-1] == generated.shape[1] - 1
                positions.append(ps)
            ctx = audit._local_attention_edit(model, heads, positions) if heads else contextlib.nullcontext()
            with ctx:
                next_ids = model(input_ids=generated).logits[:, -1].argmax(-1)
            next_ids = torch.where(done, torch.full_like(next_ids, vocab.eos_id), next_ids)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            done |= next_ids.eq(vocab.eos_id)
            if bool(done.all()):
                break
        for old, e, seq, prefix, onset in zip(batch, examples, generated.cpu().tolist(), tokens, onsets):
            ts = vocab.decode(seq)
            assert ts[:len(prefix)] == prefix, 'Resume changed an earlier generated token'
            result.append(dict(old, **audit._parse_generation(ts, vocab, e, mode),
                               **audit.after_duplicate_diagnostic(ts, vocab, e.count),
                               intervention_onset=onset, hit_token_cap=float('<EOS>' not in ts)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    original = json.loads((args.audit_dir / 'manifest.json').read_text())
    assert original['replication_exact']
    source_files = [args.audit_dir / m / 'generation_trials.csv' for m in ['nonthinking', 'thinking']]
    hashes = {str(s): audit.digest(s) for s in source_files}
    audit.write_json(args.output / 'protocol.json', {
        'created_unix': time.time(), 'budget': args.budget, 'batch_size': args.batch_size,
        'policy': 'Resume all and only EOS-capped sustained rows; retain uncapped rows unchanged. Same head sets, inputs, onset and contiguous-through-end zeroing.',
        'primary_scoring': 'unchanged original strict parser; termination measured separately',
        'original_sha256': hashes, 'script_sha256': audit.digest(Path(__file__)),
        'runner_sha256': audit.digest(Path(audit.__file__)),
    })
    examples_path = args.run_dir / 'analysis/behavior_confirmation_v58/examples.jsonl'
    lookup = {e.prompt_sha256: e for e in [audit.base.example_from_dict(json.loads(s)) for s in examples_path.read_text().splitlines() if s.strip()]}
    validation = {}
    for mode in ['nonthinking', 'thinking']:
        src = pd.read_csv(args.audit_dir / mode / 'generation_trials.csv', keep_default_na=False)
        # Explicit conversion leaves empty clean head lists as empty strings.
        for c in ['hit_token_cap', 'intervention_onset']:
            src[c] = pd.to_numeric(src[c], errors='coerce')
        src = src.loc[src.protocol.eq('sustained')].copy()
        capped = src.loc[src.hit_token_cap.eq(1)]
        assert len(src) == 2700
        cp = audit.checkpoint_path(args.run_dir, mode, 10000)
        assert audit.digest(cp) == original['validation'][mode]['checkpoint_sha256']
        cfg, vocab, _, _, model = audit.base.load_v20_checkpoint_model(args.run_dir, 'rope', mode, step=10000, device='cuda')
        revised = []
        for hs, f in capped.groupby('heads', dropna=False):
            heads = [(int(s.split('H')[0][1:]), int(s.split('H')[1])) for s in hs.split(';') if s]
            revised.extend(extend(model, cfg, vocab, f.to_dict('records'), lookup, heads, mode, args.batch_size, args.budget))
            print(mode, hs, len(f), 'extended', flush=True)
        output = pd.concat([src.loc[~src.hit_token_cap.eq(1)], pd.DataFrame(revised)], ignore_index=True)
        assert len(output) == len(src) and not output.duplicated(['prompt_sha256', 'heads']).any()
        output['eos_reached'] = 1 - output.hit_token_cap.astype(float)
        output['answer_and_eos_correct'] = output.ar_accuracy.astype(float) * output.eos_reached
        output['total_generated_tokens'] = output.generated_tokens.map(lambda s: len(s.split()[:s.split().index('<EOS>') + 1]) if '<EOS>' in s else len(s.split()))
        for i, row in output.iterrows():
            item = audit.base.render_v20(lookup[row.prompt_sha256], vocab, mode)
            output.loc[i, 'total_generated_tokens'] -= item.spans.ans_pos + 1 if mode == 'nonthinking' else item.spans.think_pos + 1
        dest = args.output / mode
        dest.mkdir()
        output.to_csv(dest / 'generation_trials.csv', index=False)
        metrics = ['ar_accuracy', 'ar_answered', 'duplicate_ans', 'after_duplicate_answered', 'after_duplicate_accuracy',
                   'trace_exact', 'hit_token_cap', 'eos_reached', 'answer_and_eos_correct', 'total_generated_tokens']
        for c in metrics:
            output[c] = pd.to_numeric(output[c], errors='coerce')
        output.groupby(['arm', 'top_k', 'repeat', 'heads'], dropna=False)[metrics].mean().reset_index().to_csv(dest / 'generation_summary.csv', index=False)
        prior = src.set_index(['prompt_sha256', 'heads'])
        after = output.set_index(['prompt_sha256', 'heads']).reindex(prior.index)
        validation[mode] = {'input_rows': len(src), 'resumed_rows': len(capped),
                            'still_capped_at_64': int(output.hit_token_cap.sum()),
                            'accuracy_changed_rows': int(prior.ar_accuracy.astype(float).ne(after.ar_accuracy.astype(float)).sum()),
                            'uncapped_tokens_unchanged': bool(prior.loc[prior.hit_token_cap.eq(0), 'generated_tokens'].eq(after.loc[prior.hit_token_cap.eq(0), 'generated_tokens']).all()),
                            'max_generated_tokens': int(output.total_generated_tokens.max())}
        audit.write_json(args.output / 'validation.json', validation)
        del model
        torch.cuda.empty_cache()
    assert all(audit.digest(Path(s)) == h for s, h in hashes.items())
    audit.write_json(args.output / 'manifest.json', {'status': 'complete', 'validation': validation,
                     'original_unchanged': True, 'files': {str(f.relative_to(args.output)): audit.digest(f) for f in args.output.rglob('*') if f.is_file()}})
    print('EXTENSION COMPLETE', json.dumps(validation), flush=True)


if __name__ == '__main__':
    main()
