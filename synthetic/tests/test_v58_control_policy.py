from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from freeze_v58_control_policy import matched_control_plan


def test_one_disjoint_complement_does_not_trigger_overlap_or_fake_repeats():
    p = matched_control_plan([(1, h) for h in range(4)])
    assert p['actual_unique_controls'] == 1
    assert not p['overlap_fallback_used']
    assert p['controls'][0]['heads'] == [[1, h] for h in range(4, 8)]


def test_overlap_is_minimal_and_limited_to_infeasible_layers():
    selected = [(1, h) for h in range(5)] + [(2, 0), (2, 1)]
    p = matched_control_plan(selected)
    assert p['minimum_overlap'] == 2
    assert p['actual_unique_controls'] == 3
    for bank in p['controls']:
        hs = {tuple(h) for h in bank['heads']}
        assert len(hs) == 7
        assert {(1, 5), (1, 6), (1, 7)} <= hs
        assert len(hs & set(selected)) == 2
        assert not hs & {(2, 0), (2, 1)}
        assert sum(l == 1 for l, h in hs) == 5


def test_saturated_layer_is_explicit_identical_control():
    p = matched_control_plan([(1, h) for h in range(8)])
    assert p['minimum_overlap'] == 8
    assert p['actual_unique_controls'] == 1
    assert p['controls'][0]['heads'] == p['selected']


def test_existing_feasible_controls_are_preserved():
    registered = [[[1, 1], [1, 2]], [[1, 2], [1, 6]], [[1, 6], [1, 7]]]
    p = matched_control_plan([(1, 3), (1, 0)], registered=registered)
    assert [b['heads'] for b in p['controls']] == registered
    assert p['reused_registered_controls'] == 3


def test_unnecessary_registered_overlap_is_rejected():
    with pytest.raises(ValueError, match='Registered control'):
        matched_control_plan([(1, 0)], registered=[[[1, 0]]])
