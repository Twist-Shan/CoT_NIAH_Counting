import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from test_v58_alignment_supplement import fixture
from v58_alignment_core import make_record
from run_v58_unified_legacy import attention_roles, capture_geometry


def test_geometry_preserves_all_paired_occurrence_keys():
    model, vocab, source = fixture()
    example = source['example']
    endpoints = []
    for mode in ['nonthinking', 'thinking']:
        r = make_record(example, vocab, mode, 'confirmation', 3)
        datasets = capture_geometry(model, vocab, [r])
        running, answer = datasets.values()
        assert len(running.metadata) == example.count
        assert len(answer.metadata) == 1
        assert running.metadata.group_id.eq('3').all()
        endpoints.append(running.metadata[['prompt_sha256', 'occurrence']].to_records(index=False).tolist())
    assert endpoints[0] == endpoints[1]


def test_attention_roles_are_probabilities_and_trace_only_mask_preserves_earlier_geometry():
    model, vocab, source = fixture()
    example = source['example']
    r = make_record(example, vocab, 'thinking', 'confirmation', 0)
    clean = capture_geometry(model, vocab, [r])
    changed = capture_geometry(model, vocab, [r], heads=[(4, 0), (4, 1)])
    for ep in clean:
        for layer in range(5):
            np.testing.assert_array_equal(clean[ep].states_by_layer[layer], changed[ep].states_by_layer[layer])
    roles = attention_roles(model, vocab, [r])
    for column in ['broad', 'targeted', 'successor', 'answer_needle_mass']:
        assert roles[column].between(0, 1).all()
