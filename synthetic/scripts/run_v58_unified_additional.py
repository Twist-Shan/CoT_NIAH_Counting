"""Common-panel source-to-next retrieval, count-vector controls and geometry dynamics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

import run_v58_commit_query as base
from run_v58_unified_legacy import load_panel, capture_geometry, AR_STEPS
from v58_alignment_core import Engine, fit_space, make_record
from synthetic_counting_v20 import aligned_geometry as geo
from run_v20_phase_transition_audit import iter_checkpoint_models, checkpoint_map


@torch.inference_mode()
def source_next(model, vocab, records, out, batch_size):
    engine = Engine(model, vocab, [], batch_size)
    rows = []
    kinds = ['clean', 'records', 'history', 'recent', 'except_recent', 'early_half']
    for kind in kinds:
        for control in ([False] if kind == 'clean' else [False, True]):
            for start in range(0, len(records), batch_size):
                batch = records[start:start+batch_size]
                inputs, actions, ks = [], [], []
                for r in batch:
                    k = max(1, r['count']//2); ks.append(k)
                    q = r['markers'][k-1]-1
                    previous = [p for marker in r['markers'][:k-1] for p in [marker-1, marker]]
                    recent = previous[-2:]
                    ps = {'clean': [], 'records': r['needles'], 'history': previous, 'recent': recent,
                          'except_recent': previous[:-2], 'early_half': previous[:2*((k-1)//2)]}[kind]
                    if control:
                        ps = r['ordinary'][-len(ps):] if ps else []
                    inputs.append(dict(r, seq=r['seq'][:q+1], q=q)); actions.append({'blank': ps})
                with engine.hooks(inputs, actions, {}):
                    result = base.forward(model, vocab, [r['seq'] for r in inputs])
                for i, (r, modified, k, action) in enumerate(zip(batch, inputs, ks, actions)):
                    target = r['seq'][r['markers'][k-1]]
                    logits = result.logits[i, modified['q']].float()
                    competitors = logits.clone(); competitors[target] = -torch.inf
                    rows.append({'key': r['key'], 'block': r['block'], 'count': r['count'], 'k': k,
                        'arm': kind, 'ordinary_control': control, 'blank_tokens': len(action['blank']),
                        'next_marker_correct': float(logits.argmax()==target),
                        'marker_margin': float(logits[target]-competitors.max())})
    pd.DataFrame(rows).to_csv(out/'source_next_trials.csv', index=False)


def count_vector(model, vocab, records, out, batch_size):
    engine = Engine(model, vocab, [], batch_size)
    caches = {r['key']: c for r, c in zip(records, engine.run(records, capture=True))}
    discovery = [r for r in records if r['split']=='discovery']
    confirmation = [r for r in records if r['split']=='confirmation']
    rows = []
    for layer in range(1, 5):
        space = fit_space(np.stack([caches[r['key']]['hidden'][layer][r['q']].numpy() for r in discovery]), [r['count'] for r in discovery])
        for offset in [-1, 1]:
            subset = [r for r in confirmation if 1 <= r['count']+offset <= 10]
            for arm in ['self', 'count_chord', 'orthogonal_0', 'orthogonal_1', 'orthogonal_2']:
                actions = []
                for r in subset:
                    shift = space['centroids'][r['count']+offset-1]-space['centroids'][r['count']-1]
                    shift = (shift @ space['u'].T) @ space['u']
                    if arm == 'self':
                        shift = torch.zeros_like(shift)
                    elif arm.startswith('orthogonal'):
                        shift = space['v'][int(arm[-1])]*shift.norm()
                    receiver = caches[r['key']]['hidden'][layer][r['q']]
                    actions.append({'patch': [(layer, [r['q']], (receiver+shift)[None])]})
                changed = engine.run(subset, actions)
                for r, row in zip(subset, changed):
                    row.update(arm=arm, layer=layer, offset=offset,
                        donor_count_adoption=float(row['predicted_count']==r['count']+offset),
                        directed_expected_shift=offset*(row['expected_count']-caches[r['key']]['expected_count']))
                    rows.append(row)
    pd.DataFrame(rows).to_csv(out/'count_vector_trials.csv', index=False)


def geometry_dynamics(args, cfg, vocab, records, mode, out):
    selected = json.loads((args.legacy/mode/'geometry/frozen_depths.json').read_text())
    rows = []
    for step, model in iter_checkpoint_models(cfg, vocab, args.run_dir, mode, sorted(AR_STEPS & set(checkpoint_map(args.run_dir, mode)))):
        datasets = capture_geometry(model, vocab, records, batch_size=args.batch_size)
        for endpoint, data in datasets.items():
            layer = selected[endpoint]
            dm = data.metadata.split.eq('discovery'); cm = data.metadata.split.eq('confirmation')
            dx = data.states_by_layer[layer][dm]; cx = data.states_by_layer[layer][cm]
            dy = data.metadata.loc[dm, 'occurrence'].to_numpy(); cy = data.metadata.loc[cm, 'occurrence'].to_numpy()
            _, pred, _ = geo._decoder_predictions(dx, dy, cx, range(1, 11), pca_dim=16, random_state=6201)
            centers = np.stack([dx[dy == k].mean(0) for k in range(1, 11)])
            values = np.linalg.svd(centers-centers.mean(0), compute_uv=False)**2
            probability = values/values.sum() if values.sum() else np.ones(len(values))/len(values)
            rank3 = float(probability[:3].sum())
            ed = float(np.exp(-np.sum(probability*np.log(np.maximum(probability, 1e-30)))))
            for m, prediction in zip(data.metadata.loc[cm].to_dict('records'), pred):
                rows.append({**m, 'step': step, 'layer': layer, 'ncc_correct': float(prediction==m['occurrence']),
                             'centroid_rank3_variance': rank3, 'centroid_effective_dimension': ed})
        pd.DataFrame(rows).to_csv(out/'geometry_dynamics_trials.csv', index=False)
        print('geometry dynamics', mode, step, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--alignment', type=Path, required=True)
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    chosen, registry = load_panel(args.run_dir, args.alignment/'input_registry.csv')
    cfg, vocab, _, _, _ = base._load_bundle(args.run_dir, device='cuda')
    for mode in ['nonthinking', 'thinking']:
        out = args.output/mode; out.mkdir()
        records = [make_record(e, vocab, mode, s, b) for e, s, b in chosen]
        _, _, _, _, model = base.load_v20_checkpoint_model(args.run_dir, 'rope', mode, step=10000, device='cuda')
        count_vector(model, vocab, records, out, args.batch_size)
        if mode == 'thinking':
            source_next(model, vocab, [r for r in records if r['split']=='confirmation'], out, args.batch_size)
        del model; torch.cuda.empty_cache()
        geometry_dynamics(args, cfg, vocab, records, mode, out)
    base.write_json(args.output/'manifest.json', {'status': 'complete', 'registry_sha256': base.digest(args.alignment/'input_registry.csv'),
        'script_sha256': base.digest(Path(__file__)), 'completed_unix': time.time(), 'confirmation_prompts': 100})
    print('ADDITIONAL UNIFIED COMPLETE', flush=True)


if __name__ == '__main__':
    main()
