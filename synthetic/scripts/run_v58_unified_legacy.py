"""Recompute historical body assays on the immutable alignment input registry."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import run_v58_commit_query as base
from v58_alignment_core import controls_for, make_record
from run_v58_alignment_supplement import free_running_rows
from run_v58_thinking_factorial_ablation import _factorial_rows
from synthetic_counting_v20 import aligned_geometry as geo
from synthetic_counting_v20.v10_port_analysis import (
    _local_attention_edit, _retrieval_corruption, _capture_attention_internals,
    _marker_margin, _attention_pattern_value_patch, _residual_patch,
)
from run_v20_phase_transition_audit import checkpoint_map, iter_checkpoint_models

CLASSES = tuple(range(1, 11))
AR_STEPS = {0, 100, 200, 400, 800, 1500, 2500, 4000, 6000, 8000, 10000}


def load_panel(run, registry):
    frame = pd.read_csv(registry)
    assert frame.groupby(['split', 'count']).size().to_dict() == {
        (split, n): amount for split, amount in [('discovery', 20), ('confirmation', 10)] for n in CLASSES}
    assert len(frame) == frame.key.nunique() == 300
    assert len(frame[['set_id', 'corpus_start']].drop_duplicates()) == 300
    examples = {e.prompt_sha256: e for e in [base.example_from_dict(json.loads(line))
        for line in (run/'analysis/behavior_confirmation_v58/examples.jsonl').read_text().splitlines() if line.strip()]}
    assert all(examples[r.key].count == r.count for r in frame.itertuples())
    return [(examples[r.key], r.split, r.block) for r in frame.itertuples()], frame


@torch.inference_mode()
def capture_geometry(model, vocab, records, heads=(), batch_size=16):
    mode = records[0]['mode']
    endpoints = geo.MODE_ENDPOINTS[mode]
    metadata = {ep: [] for ep in endpoints}
    states = {ep: {l: [] for l in range(5)} for ep in endpoints}
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        queries = [[r['q']] if mode == 'nonthinking' else [p-1 for p in r['markers']] for r in batch]
        context = _local_attention_edit(model, heads, queries) if heads else contextlib.nullcontext()
        with context:
            output = base.forward(model, vocab, [r['seq'] for r in batch], hidden=True)
        for i, r in enumerate(batch):
            running = r['needles'] if mode == 'nonthinking' else r['markers']
            for endpoint, sites, labels in [(endpoints[0], running, range(1, r['count']+1)),
                                            (endpoints[1], [r['q']], [r['count']])]:
                for p, k in zip(sites, labels):
                    metadata[endpoint].append({'endpoint': endpoint, 'mode': mode, 'split': r['split'],
                        'group_id': str(r['block']), 'block': r['block'], 'prompt_sha256': r['key'],
                        'occurrence': k, 'total_count': r['count'], 'position': p})
                    for l, h in enumerate(output.hidden_states):
                        states[endpoint][l].append(h[i, p].float().cpu().numpy())
    return {ep: geo.GeometryDataset(pd.DataFrame(metadata[ep]),
                {l: np.stack(v) for l, v in states[ep].items()}) for ep in endpoints}


def evaluate_geometry(model, vocab, records, bank, out, batch_size):
    out.mkdir()
    clean = capture_geometry(model, vocab, records, batch_size=batch_size)
    summaries, selections, clouds, predictions = [], [], [], []
    frozen = {}
    # Discovery CV chooses depths; all confirmation results remain out of selection.
    for endpoint, data in clean.items():
        layer_table, selected, depth = geo.evaluate_geometry_dataset(data, endpoint=endpoint, classes=CLASSES)
        summaries.append(layer_table); selections.append(selected)
        frozen[endpoint] = depth
        data.metadata.to_csv(out/f'{endpoint}_metadata.csv', index=False)
        np.savez_compressed(out/f'{endpoint}_states.npz', **{f'layer_{l}': x for l, x in data.states_by_layer.items()})
        for l in range(1, 5):
            cloud = geo.confirmation_pca_coordinates(data, l)
            cloud['layer'] = l; cloud['k'] = cloud.occurrence; cloud['sample'] = cloud.prompt_sha256
            clouds.append(cloud)
    pd.concat(summaries).to_csv(out/'clean_layer_metrics.csv', index=False)
    pd.concat(selections).to_csv(out/'clean_selections.csv', index=False)
    pd.concat(clouds).to_csv(out/'projection_cloud.csv', index=False)
    base.write_json(out/'frozen_depths.json', frozen)
    confirmation = [r for r in records if r['split'] == 'confirmation']
    arms = [('clean', 0, -1, [])]
    for k in [1, 2, 4]:
        arms += [('selected', k, -1, bank[:k])]
        arms += [('control', k, j, h) for j, h in enumerate(controls_for(bank, k))]
    for condition, k, repeat, heads in arms:
        if condition == 'clean':
            changed = {ep: geo.GeometryDataset(d.metadata.loc[d.metadata.split.eq('confirmation')].reset_index(drop=True),
                       {l: x[d.metadata.split.eq('confirmation')] for l, x in d.states_by_layer.items()}) for ep, d in clean.items()}
        else:
            changed = capture_geometry(model, vocab, confirmation, heads, batch_size)
        for ep, data in changed.items():
            reference = clean[ep]
            dm = reference.metadata.split.eq('discovery')
            cm = reference.metadata.split.eq('confirmation')
            assert data.metadata[['prompt_sha256', 'occurrence']].to_records(index=False).tolist() == reference.metadata.loc[cm, ['prompt_sha256', 'occurrence']].to_records(index=False).tolist()
            y = reference.metadata.loc[dm, 'occurrence'].to_numpy()
            for l in range(1, 5):
                lp, npred, _ = geo._decoder_predictions(reference.states_by_layer[l][dm], y,
                    data.states_by_layer[l], CLASSES, pca_dim=16, random_state=6201)
                meta = data.metadata.copy()
                meta['layer'] = l; meta['condition'] = condition; meta['top_k'] = k; meta['repeat'] = repeat
                meta['ncc_prediction'] = npred; meta['logistic_prediction'] = lp
                meta['ncc_correct'] = (npred == meta.occurrence).astype(float)
                meta['logistic_correct'] = (lp == meta.occurrence).astype(float)
                predictions.append(meta)
        pd.concat(predictions).to_csv(out/'frozen_probe_trials.csv', index=False)
        print('geometry', records[0]['mode'], condition, k, repeat, flush=True)
    return clean


@torch.inference_mode()
def attention_roles(model, vocab, records, batch_size=16, step=10000):
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        output = base.forward(model, vocab, [r['seq'] for r in batch], attention=True)
        for i, r in enumerate(batch):
            for l, weights in enumerate(output.attentions, 1):
                a = weights[i].float()
                mass = a[:, r['q'], r['needles']]
                total = mass.sum(-1)
                p = mass/total[:, None].clamp_min(1e-20)
                entropy = -(p*p.clamp_min(1e-20).log()).sum(-1)
                metric = {'answer_needle_mass': total, 'answer_effective_coverage': entropy.exp()/r['count'],
                          'broad': total*entropy.exp()/r['count']}
                if r['markers']:
                    qs = [p-1 for p in r['markers']]
                    metric['targeted'] = a[:, qs, r['needles']].mean(-1)
                    metric['successor'] = a[:, r['markers'], qs].mean(-1)
                    target_rows = a[:, qs][:, :, r['needles']]
                    correct = torch.arange(r['count'], device=a.device)[None, :]
                    metric['correct_occurrence_top1'] = (target_rows.argmax(-1) == correct).float().mean(-1)
                    metric['targeted_needle_mass'] = target_rows.sum(-1).mean(-1)
                for h in range(8):
                    rows.append({'key': r['key'], 'block': r['block'], 'count': r['count'],
                        'mode': r['mode'], 'step': step, 'layer': l, 'head': h,
                        **{name: float(values[h]) for name, values in metric.items()}})
    return pd.DataFrame(rows)


def run_factorial(model, cfg, vocab, examples, records, sites, out, batch_size):
    rank = attention_roles(model, vocab, [r for r in records if r['split']=='discovery'], batch_size)
    sr = rank.groupby(['layer', 'head']).successor.mean().reset_index().sort_values(['successor', 'layer', 'head'], ascending=[False, True, True])
    sr.to_csv(out/'successor_discovery_ranking.csv', index=False)
    successor = [(int(sr.iloc[0].layer), int(sr.iloc[0]['head']))]
    targeted = [tuple(x[:2]) for x in sites['ranking']['targeted'][:2]]
    broad = [tuple(x[:2]) for x in sites['ranking']['broad'][:2]]
    controls = controls_for(successor, 1)
    arms = [('clean', [], [], []), ('targeted_top2', targeted, [], []),
            ('broad_top2', [], broad, []), ('targeted_plus_broad', targeted, broad, []),
            ('successor_top1', [], [], successor), ('targeted_plus_successor', targeted, [], successor)]
    arms += [(f'successor_control_{i}', [], [], h) for i, h in enumerate(controls)]
    base.write_json(out/'factorial_frozen_arms.json', arms)
    frames = []
    for arm, t, b, s in arms:
        data = _factorial_rows(model, cfg, vocab, examples, targeted_heads=t, broad_heads=b, successor_heads=s, batch_size=batch_size)
        data['arm'] = arm
        frames.append(data)
        pd.concat(frames).to_csv(out/'factorial_trials.csv', index=False)
        print('factorial', arm, flush=True)


@torch.inference_mode()
def run_transport(model, cfg, vocab, records, bank, out, batch_size):
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        ks = [max(1, r['count']//2) for r in batch]
        pairs = [_retrieval_corruption(r['example'], vocab, k) for r, k in zip(batch, ks)]
        clean, damaged, targets, alternatives = [list(x) for x in zip(*pairs)]
        qs = [i.spans.trace_query_positions[k-1] for i, k in zip(clean, ks)]
        ps = [i.prompt_needle_positions[k-1] for i, k in zip(clean, ks)]
        co, values = _capture_attention_internals(model, clean, vocab, cfg.device)
        do, _ = _capture_attention_internals(model, damaged, vocab, cfg.device)
        clean_margin = _marker_margin(co.logits, clean, ks, targets, alternatives)
        corrupt_margin = _marker_margin(do.logits, damaged, ks, targets, alternatives)
        def record(condition, k, repeat, margins):
            for r, occurrence, c, d, m in zip(batch, ks, clean_margin, corrupt_margin, margins):
                rows.append({'key': r['key'], 'block': r['block'], 'count': r['count'], 'occurrence': occurrence,
                    'condition': condition, 'top_k': k, 'repeat': repeat, 'clean_margin': float(c),
                    'corrupt_margin': float(d), 'margin': float(m), 'restoration': float(m-d),
                    'normalized_recovery': float((m-d)/(c-d)) if c-d > 1e-5 else np.nan})
        record('clean', 0, -1, clean_margin); record('damaged', 0, -1, corrupt_margin)
        for k in [1, 2, 4]:
            for condition, repeat, heads in [('value_selected', -1, bank[:k])] + [('value_control', j, h) for j, h in enumerate(controls_for(bank, k))]:
                with _attention_pattern_value_patch(model, heads, qs, ps, donor_values=values):
                    changed = base.forward(model, vocab, [i.input_ids for i in damaged])
                record(condition, k, repeat, _marker_margin(changed.logits, damaged, ks, targets, alternatives))
        for layer in range(1, 5):
            vector = torch.stack([co.hidden_states[layer][i, p] for i, p in enumerate(qs)])
            with _residual_patch(model, layer, qs, vector):
                changed = base.forward(model, vocab, [i.input_ids for i in damaged])
            record(f'residual_L{layer}', 0, -1, _marker_margin(changed.logits, damaged, ks, targets, alternatives))
    pd.DataFrame(rows).to_csv(out/'transport_trials.csv', index=False)


def dynamics(args, cfg, vocab, records, mode, out):
    available = checkpoint_map(args.run_dir, mode)
    other = checkpoint_map(args.run_dir, 'thinking' if mode == 'nonthinking' else 'nonthinking')
    steps = sorted(set(available) & set(other))
    assert steps[0] == 0 and steps[-1] == 10000
    hashes = {str(path): base.digest(path) for path in set(available[s] for s in steps)}
    base.write_json(out/'dynamics_plan.json', {'steps': steps, 'ar_steps': sorted(AR_STEPS & set(steps)),
        'inputs': [r['key'] for r in records], 'checkpoint_shards_sha256': hashes,
        'panel_registry_sha256': base.digest(args.alignment/'input_registry.csv')})
    sites = json.loads((args.alignment/mode/'frozen_sites.json').read_text())
    role = 'broad' if mode == 'nonthinking' else 'targeted'
    bank = [tuple(x[:2]) for x in sites['ranking'][role][:4]]
    frames, ar, causal = [], [], []
    for step, model in iter_checkpoint_models(cfg, vocab, args.run_dir, mode, steps):
        frames.append(attention_roles(model, vocab, records, args.batch_size, step))
        pd.concat(frames).to_csv(out/'dynamics_attention_trials.csv', index=False)
        pd.concat(frames).groupby(['mode', 'step', 'layer', 'head']).mean(numeric_only=True).reset_index().to_csv(out/'dynamics_attention_summary.csv', index=False)
        if step in AR_STEPS:
            frame = free_running_rows(model, cfg, vocab, [r['example'] for r in records], mode=mode, heads=[], batch_size=args.batch_size)
            frame['step'] = step; ar.append(frame)
            pd.concat(ar).to_csv(out/'dynamics_behavior_trials.csv', index=False)
            for condition, repeat, heads in [('selected', -1, bank[:2])] + [('control', j, h) for j, h in enumerate(controls_for(bank, 2))]:
                trial = free_running_rows(model, cfg, vocab, [r['example'] for r in records], mode=mode, heads=heads, batch_size=args.batch_size)
                trial['step'] = step; trial['condition'] = condition; trial['repeat'] = repeat
                causal.append(trial)
            pd.concat(causal).to_csv(out/'dynamics_causal_trials.csv', index=False)
            if mode == 'thinking':
                folder = out/f'transport_step_{step:05d}'; folder.mkdir(exist_ok=True)
                run_transport(model, cfg, vocab, records, bank, folder, args.batch_size)
        print('dynamics', mode, step, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--families', nargs='+', default=['geometry', 'factorial', 'transport', 'dynamics'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    chosen, registry = load_panel(args.run_dir, args.alignment/'input_registry.csv')
    cfg, vocab, _, _, _ = base._load_bundle(args.run_dir, device='cuda')
    assert cfg.version == 'v58' and cfg.train_steps == 10000 and cfg.trace_format == 'separator'
    args.output.mkdir(exist_ok=True, parents=True)
    all_keys = {}
    for mode in ['nonthinking', 'thinking']:
        out = args.output/mode; out.mkdir(exist_ok=True)
        records = [make_record(e, vocab, mode, split, block) for e, split, block in chosen]
        confirmation = [r for r in records if r['split']=='confirmation']
        all_keys[mode] = [r['key'] for r in confirmation]
        sites = json.loads((args.alignment/mode/'frozen_sites.json').read_text())
        role = 'broad' if mode == 'nonthinking' else 'targeted'
        bank = [tuple(x[:2]) for x in sites['ranking'][role][:4]]
        if any(f in args.families for f in ['geometry', 'factorial', 'transport']):
            _, _, _, _, model = base.load_v20_checkpoint_model(args.run_dir, 'rope', mode, step=10000, device='cuda')
            model.eval()
            if 'geometry' in args.families:
                evaluate_geometry(model, vocab, records, bank, out/'geometry', args.batch_size)
            if mode == 'thinking':
                if 'factorial' in args.families:
                    run_factorial(model, cfg, vocab, [r['example'] for r in confirmation], records, sites, out, args.batch_size)
                if 'transport' in args.families:
                    run_transport(model, cfg, vocab, confirmation, bank, out, args.batch_size)
            del model; torch.cuda.empty_cache()
        if 'dynamics' in args.families:
            dynamics(args, cfg, vocab, confirmation, mode, out)
    assert all_keys['thinking'] == all_keys['nonthinking']
    base.write_json(args.output/'manifest.json', {'status': 'complete', 'families': args.families,
        'panel_registry_sha256': base.digest(args.alignment/'input_registry.csv'),
        'script_sha256': base.digest(Path(__file__)), 'confirmation_prompts_per_mode': 100,
        'realized_mode_input_keys_equal': True, 'completed_unix': time.time()})
    print('UNIFIED LEGACY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
