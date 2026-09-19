"""Validate new curves against the frozen choices, registry, and final layer sweep."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT/'synthetic/work/v58_final/analysis'
RAW = OUT/'data/ncc_selected_dynamics_20260913'
manifest = json.loads((RAW/'manifest.json').read_text())
selection = json.loads((OUT/'selection.json').read_text())
assert manifest['status'] == 'complete'
assert manifest['selection_sha256'] == hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
assert manifest['registry_sha256'] == hashlib.sha256((BASE/'v58_alignment_supplement_20260905/input_registry.csv').read_bytes()).hexdigest()
frames = []
for mode in ['nonthinking', 'thinking']:
    data = pd.read_csv(RAW/mode/'geometry_dynamics_trials.csv')
    assert set(data['split']) == {'confirmation'}
    for endpoint, part in data.groupby('endpoint'):
        assert part.layer.unique().tolist() == [selection['selected'][mode][endpoint]]
        assert sorted(part.step.unique()) == manifest['steps']
        keys = ['prompt_sha256', 'occurrence', 'position']
        reference = None
        for step, at_step in part.groupby('step'):
            assert len(at_step) == (100 if 'answer_query' in endpoint else 550)
            assert at_step.prompt_sha256.nunique() == 100
            assert not at_step.duplicated(keys).any()
            current = set(map(tuple, at_step[keys].to_numpy()))
            if reference is None:
                reference = current
            assert current == reference
    data['mode'] = mode
    frames.append(data)
trials = pd.concat(frames, ignore_index=True)
scores = trials.groupby(['mode','endpoint','step','layer','occurrence']).ncc_correct.mean().groupby(['mode','endpoint','step','layer']).mean().reset_index()
assert len(scores) == 44
final_checks = []
for row in scores[scores.step.eq(10000)].to_dict('records'):
    saved = pd.read_csv(BASE/f"v58_unified_legacy_20260905/{row['mode']}/geometry/clean_layer_metrics.csv")
    ref = saved[saved.endpoint.eq(row['endpoint']) & saved.layer.eq(row['layer'])].iloc[0]
    expected = float(ref['confirmation_ncc_balanced_accuracy'])
    assert np.isclose(row['ncc_correct'], expected, atol=1e-12), (row, expected)
    final_checks.append({**row, 'saved_confirmation_ncc': expected})
# The unchanged Thinking answer layer must reproduce the archived curve exactly.
old = pd.read_csv(BASE/'v58_unified_additional_20260905/thinking/geometry_dynamics_trials.csv')
old = old[old.endpoint.str.contains('answer_query')]
new = trials[trials.endpoint.eq('thinking_answer_query')]
cols = ['step','prompt_sha256','occurrence','position','layer','ncc_correct']
assert old[cols].sort_values(cols[:-1]).reset_index(drop=True).equals(new[cols].sort_values(cols[:-1]).reset_index(drop=True))
scores.to_csv(OUT/'data/synthetic_ncc_dynamics.csv', index=False)
(OUT/'synthetic_dynamics_audit.json').write_text(json.dumps({'status':'PASS','checkpoints_per_endpoint':11,'fixed_inputs':True,'final_checkpoint_matches':final_checks,'unchanged_thinking_answer_curve_reproduced':True}, indent=2))
print(scores.to_string(index=False))
