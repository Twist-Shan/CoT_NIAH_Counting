"""Outcome-independent contracts for Native-style N=10 Update selection."""
from __future__ import annotations
import math


def select_discovery(ledger, candidates, confirmation, quota=20):
    if len(candidates) != len(set(candidates)) or set(candidates) & set(confirmation):
        raise ValueError("Discovery candidates duplicate or overlap confirmation")
    lookup = {}
    for row in ledger:
        if int(row['gold_count']) != 10:
            continue
        seed = int(row['seed'])
        if seed in lookup:
            raise ValueError("Duplicate N=10 ledger seed")
        lookup[seed] = row
    if set(candidates) - lookup.keys():
        raise ValueError("Unaccounted discovery candidates")
    selected = [s for s in candidates if lookup[s]['format_eligible'] and lookup[s]['endpoint_eligible']][:quota]
    if len(selected) != quota:
        raise ValueError(f"Discovery quota shortfall: {len(selected)}/{quota}; no adaptive pool extension")
    return selected


def rank_layers(metrics, model):
    last = {'Qwen3-8B': 35, 'Gemma4-E4B': 22}[model]
    candidates = [r for r in metrics if 1 <= r['layer_one_based'] <= last]
    if sorted(r['layer_one_based'] for r in candidates) != list(range(1, last + 1)):
        raise ValueError("Missing or duplicate eligible layer")
    for row in candidates:
        if row['discovery_oof_rows'] != 200 or row['discovery_fold_count'] != 5:
            raise ValueError("Expected 200 discovery states and five seed-grouped folds")
        for key in ('discovery_oof_ncc_balanced_accuracy', 'discovery_oof_logistic_balanced_accuracy'):
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError("Invalid discovery score")
    return sorted(candidates, key=lambda r: (-round(r['discovery_oof_ncc_balanced_accuracy'], 12),
        -round(r['discovery_oof_logistic_balanced_accuracy'], 12), r['layer_one_based']))


def validate_selection(manifest, model, mode, confirmation):
    if manifest.get('status') != 'FROZEN_N10_DISCOVERY_LAYERS' or manifest.get('confirmation_used_for_selection') is not False:
        raise ValueError("Layer selection was not frozen from discovery only")
    cells = [c for c in manifest['cells'] if c['model'] == model and c['mode'] == mode]
    if len(cells) != 1:
        raise ValueError("Missing or duplicate selection cell")
    cell = cells[0]
    if cell['gold_count'] != 10 or cell['discovery_states'] != 200:
        raise ValueError("Expected N=10 selection on 200 states")
    if len(cell['discovery_seeds']) != len(set(cell['discovery_seeds'])) or len(cell['discovery_seeds']) != 20:
        raise ValueError("Expected twenty unique discovery seeds")
    if len(confirmation) != len(set(confirmation)) or len(confirmation) != 10:
        raise ValueError("Expected ten unique confirmation seeds")
    if list(confirmation) != cell['confirmation_seeds'] or set(confirmation) & set(cell['discovery_seeds']):
        raise ValueError("Confirmation cohort mismatch or discovery overlap")
    layer = cell['layer_one_based']
    if not isinstance(layer, int) or not 1 <= layer <= {'Qwen3-8B': 35, 'Gemma4-E4B': 22}[model]:
        raise ValueError("Selected layer has insufficient downstream KV writers")
    return layer - 1
