"""Freeze same-layer controls, allowing only structurally necessary overlap.

Previously frozen feasible controls are reused. No behavioral result is read.
This creates an additive control configuration, not new inference results.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def matched_control_plan(selected, *, heads_per_layer=8, repeats=3, seed=20260908,
                         registered=(), exhaustive=False):
    selected = [tuple(h) for h in selected]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Selected heads must be nonempty and unique')
    if heads_per_layer < 1 or repeats < 1:
        raise ValueError('Head count and repeat count must be positive')
    if any(l < 1 or not 0 <= h < heads_per_layer for l, h in selected):
        raise ValueError('Invalid one-based layer or zero-based head index')
    layers = sorted({l for l, _ in selected})
    options, metadata = [], []
    for layer in layers:
        inside = sorted(h for l, h in selected if l == layer)
        outside = [h for h in range(heads_per_layer) if h not in inside]
        k = len(inside)
        overlap = max(0, k - len(outside))
        if overlap == 0:
            choices = list(itertools.combinations(outside, k))
        else:
            choices = [tuple(sorted(outside + list(extra)))
                       for extra in itertools.combinations(inside, overlap)]
        options.append(choices)
        metadata.append({'layer': layer, 'selected_count': k,
                         'nonselected_available': len(outside),
                         'minimum_overlap': overlap,
                         'feasible_combinations': len(choices)})
    total = math.prod(map(len, options))
    minimum_overlap = sum(row['minimum_overlap'] for row in metadata)
    controls = []

    def valid(candidate):
        if len(candidate) != len(selected) or len(set(candidate)) != len(candidate):
            return False
        if {l for l, _ in candidate} != set(layers):
            return False
        return all(tuple(sorted(h for l, h in candidate if l == layer)) in choices
                   for layer, choices in zip(layers, options))

    for bank in registered:
        candidate = tuple(sorted(tuple(h) for h in bank))
        if not valid(candidate):
            raise ValueError('Registered control violates the declared policy')
        if candidate not in controls:
            controls.append(candidate)
    if len(controls) > repeats:
        raise ValueError('Requested repeats would discard registered controls')
    reused = len(controls)
    # Decode indices of the Cartesian product without materializing that product.
    def bank_at(index):
        groups = []
        for layer, choices in reversed(list(zip(layers, options))):
            index, digit = divmod(index, len(choices))
            groups.extend((layer, h) for h in choices[digit])
        return tuple(sorted(groups))

    rng = random.Random(seed)
    target = total if exhaustive else min(repeats, total)
    if exhaustive:
        for index in range(total):
            candidate = bank_at(index)
            if candidate not in controls:
                controls.append(candidate)
    else:
        while len(controls) < target:
            candidate = bank_at(rng.randrange(total))
            if candidate not in controls:
                controls.append(candidate)
    return {'selected': [list(h) for h in selected],
            'layer_counts': metadata, 'minimum_overlap': minimum_overlap,
            'overlap_fallback_used': minimum_overlap > 0,
            'feasible_unique_controls': total,
            'requested_controls': repeats, 'actual_unique_controls': len(controls),
            'reused_registered_controls': reused, 'seed': seed,
            'controls': [{'heads': [list(h) for h in bank],
                          'overlap_count': len(set(bank) & set(selected)),
                          'overlap_heads': [list(h) for h in sorted(set(bank) & set(selected))]}
                         for bank in controls]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'work/v58_final/analysis/v58_alignment_supplement_20260905')
    parser.add_argument('--output', type=Path, default=ROOT / 'work/v58_main_alignment_20260908/control_policy.json')
    args = parser.parse_args()
    result = {'schema_version': 'v58_disjoint_first_controls_v1',
              'status': 'control_configuration_only_no_new_inference',
              'policy': 'Match each layer count; exclude selected where feasible; otherwise use the minimum necessary overlap in that layer.',
              'few_controls_policy': 'Report the actual unique count. A shortage of distinct repeats does not permit extra overlap.',
              'selection': 'Frozen global Top-k across all 32 heads; no re-ranking.',
              'outcome_files_read': [], 'source_sha256': {}, 'modes': {}}
    for mode in ['nonthinking', 'thinking']:
        source = args.source / mode / 'frozen_sites.json'
        raw = source.read_bytes()
        frozen = json.loads(raw)
        result['source_sha256'][str(source.resolve())] = hashlib.sha256(raw).hexdigest()
        bank = frozen['role_bank']
        role = 'broad' if mode == 'nonthinking' else 'targeted'
        if bank != [row[:2] for row in frozen['ranking'][role][:4]]:
            raise ValueError('Frozen bank differs from global Top-4')
        result['modes'][mode] = {str(k): matched_control_plan(
            bank[:k], registered=frozen['controls'][str(k)]) for k in [1, 2, 4]}
    payload = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text(encoding='utf-8') != payload:
        raise FileExistsError('Refusing to replace a different frozen configuration')
    args.output.write_text(payload, encoding='utf-8')
    print(args.output)
    for mode, doses in result['modes'].items():
        print(mode, {k: {'controls': p['actual_unique_controls'],
                         'overlap': p['minimum_overlap']} for k, p in doses.items()})


if __name__ == '__main__':
    main()
