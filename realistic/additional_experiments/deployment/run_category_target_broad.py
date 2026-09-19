"""Discover task-matching Broad heads, then run the frozen Non-thinking sweep."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    temp.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--model', required=True)
    ap.add_argument('--stage', choices=['discovery', 'canary', 'full'], required=True)
    args = ap.parse_args(); root = args.root.resolve(); start = time.perf_counter()
    cfg = read(root/'protocol.json'); protocol_hash = sha(root/'protocol.json')
    assert cfg['record_scope'] == 'question_target_category' and args.model in cfg['models']
    for rel, expected in cfg['files'].items(): assert sha(root/rel) == expected, rel
    sys.path[:0] = [str(root), str(root/'src')]
    from category_target_broad import record_geometry, score_record_masses, rank_rows, canary_cases
    from local_selection import select_heads, random_control, control_audit
    from protocol import encode_ids
    from task_scoring import score_generation
    import torch
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v4.modeling import load_registered_model, query_attention_rows, generate_with_head_ablation, _text_config
    assert torch.cuda.is_available(), 'CUDA unavailable'
    if args.stage == 'full':
        assert read(root/'canary_audit.json')['status'] == 'PASS'
    torch.manual_seed(cfg['generation_seed'])
    model, tok, adapter = load_registered_model(resolve_model_spec(args.model), cache_dir=cfg['cache'])
    model.eval(); widths = list(adapter.num_heads)
    assert widths == cfg['widths'][args.model]

    def backend():
        values = [getattr(c, '_attn_implementation', None) for c in [model.config, _text_config(model)] if c is not None]
        assert values and all(v == 'sdpa' for v in values), values
        return values

    out = root/args.stage/args.model
    write(out/'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
        transformers=importlib.metadata.version('transformers'), gpu=torch.cuda.get_device_name(),
        widths=widths, backend=backend(), protocol_sha256=protocol_hash, command=sys.argv))
    plans = read(root/'plans'/f'{args.model}.json')
    assert len(plans) == 300 and all(p['mode'] == 'nonthinking' for p in plans)

    def inputs(p):
        source = Path(p['source'])
        for name, h in p['source_hashes'].items(): assert sha(source/name) == h, (p['case_id'], name)
        prompt = read(source/'prompt.json')
        assert p['broad_prefix'] == len(prompt['input_ids'])
        return prompt, encode_ids(prompt['input_ids'])

    def generate(enc, heads):
        backend()
        gen = generate_with_head_ablation(model, tok, adapter, enc, heads,
                                          scope='answer_query', max_new_tokens=cfg['max_new_tokens'])
        backend()
        return gen

    bankpath = root/'banks'/f'{args.model}.json'
    if args.stage == 'discovery':
        observations = []
        for p in plans:
            if p['split'] != 'discovery': continue
            dest = out/'rows'/f"{p['case_id']}.json"
            if dest.exists():
                row = read(dest); assert row['protocol_sha256'] == protocol_hash
            else:
                point_start = time.perf_counter(); prompt, enc = inputs(p)
                encoded = tok(prompt['rendered_prompt'], add_special_tokens=False, return_offsets_mapping=True)
                assert encoded['input_ids'] == prompt['input_ids']
                geometry = record_geometry(p['case'], prompt['rendered_prompt'], encoded['offset_mapping'])
                backend(); matrices, starts = query_attention_rows(model, adapter, enc); backend()
                heads, raw_masses = [], []
                for layer, (matrix, first) in enumerate(zip(matrices, starts)):
                    for head, alpha in enumerate(matrix):
                        masses = [float(alpha[max(0,a-first):min(len(alpha),b-first)].sum()) if b > first else 0.
                                  for a, b in geometry['all_record_token_spans']]
                        heads.append([layer, head, score_record_masses(p['case'], masses)])
                        raw_masses.append([layer, head, masses])
                row = dict(case_id=p['case_id'], seed=p['seed'], split=p['split'], model=args.model,
                    geometry=geometry, heads=heads, all_record_masses=raw_masses,
                    prefix_length=enc.sequence_length, backend=backend(), protocol_sha256=protocol_hash,
                    elapsed_seconds=time.perf_counter()-point_start)
                write(dest, row)
            observations.append(row)
            print(json.dumps(dict(stage=args.stage, model=args.model, completed=len(observations), total=200)), flush=True)
        assert len(observations) == 200
        ranking = rank_rows(observations)
        bank = dict(model=args.model, task='category', mode='nonthinking', assay='broad',
                    record_scope=cfg['record_scope'], ranking=ranking, sizes=cfg['sizes'][args.model],
                    source_hashes={str(p.relative_to(root)):sha(p) for p in sorted((out/'rows').glob('*.json'))},
                    protocol_sha256=protocol_hash, conditions={})
        for k in bank['sizes']:
            heads = select_heads(ranking, k)
            randoms = [random_control(heads, widths, seed) for seed in cfg['random_seeds']]
            bank['conditions'][str(k)] = dict(selected=heads, random=randoms,
                control_audit=control_audit(heads, widths, randoms))
        if bankpath.exists(): assert read(bankpath) == bank
        else: write(bankpath, bank)
    else:
        assert read(root/'discovery'/args.model/'complete.json')['status'] == 'PASS'
        bank = read(bankpath); assert bank['protocol_sha256'] == protocol_hash
        panel = canary_cases(plans) if args.stage == 'canary' else [p for p in plans if p['split'] == 'confirmation']
        sizes = sorted({bank['sizes'][0], bank['sizes'][-1]}) if args.stage == 'canary' else bank['sizes']
        completed = 0
        for p in panel:
            prompt, enc = inputs(p); dest = out/'rows'/p['case_id']
            cleanfile = dest/'clean.json'
            if cleanfile.exists():
                saved = read(cleanfile); assert saved['protocol_sha256'] == protocol_hash
                clean = saved['arm']
            else:
                gen = generate(enc, [])
                clean = dict(name='clean', heads=[], generation=gen,
                             score=score_generation(gen, p['case'], mode='nonthinking', assay='broad'))
                write(cleanfile, dict(arm=clean, protocol_sha256=protocol_hash, prefix_length=enc.sequence_length))
            for k in sizes:
                path = dest/f'K{k}.json'
                if path.exists():
                    saved = read(path)
                    assert saved['protocol_sha256'] == protocol_hash and saved['bank_sha256'] == sha(bankpath)
                else:
                    point_start = time.perf_counter(); condition = bank['conditions'][str(k)]
                    assert condition['control_audit'] == control_audit(condition['selected'], widths, condition['random'])
                    arms = [clean]
                    for name, heads in [('selected', condition['selected'])]+[(f'random_{i}', h) for i, h in enumerate(condition['random'])]:
                        gen = generate(enc, heads)
                        arms.append(dict(name=name, heads=heads, generation=gen,
                            score=score_generation(gen, p['case'], mode='nonthinking', assay='broad')))
                    write(path, dict(model=args.model, task='category', mode='nonthinking', assay='broad',
                        case_id=p['case_id'], seed=p['seed'], target_category=p['case']['target_category'],
                        k=k, arms=arms, prefix_length=enc.sequence_length, max_new_tokens=cfg['max_new_tokens'],
                        backend=backend(), bank_sha256=sha(bankpath), protocol_sha256=protocol_hash,
                        elapsed_seconds=time.perf_counter()-point_start))
                completed += 1
                print(json.dumps(dict(stage=args.stage, model=args.model, completed=completed,
                    total=len(panel)*len(sizes), case_id=p['case_id'], k=k)), flush=True)
    write(out/'complete.json', dict(status='PASS', stage=args.stage, model=args.model,
        protocol_sha256=protocol_hash, elapsed_seconds=time.perf_counter()-start))


if __name__ == '__main__': main()
