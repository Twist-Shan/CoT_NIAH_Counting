from copy import deepcopy
import pytest
from realistic_niah_v6.update_n10 import rank_layers, select_discovery, validate_selection


def test_selection_uses_format_not_final_correctness_and_preserves_order():
    ledger = [dict(seed=s, gold_count=10, format_eligible=s != 0, endpoint_eligible=True,
                   baseline_exact_count=False, intervention_success=False) for s in range(22)]
    assert select_discovery(ledger, list(range(22)), [30], 20) == list(range(1, 21))


@pytest.mark.parametrize('case', ['overlap', 'missing', 'duplicate', 'shortfall'])
def test_invalid_cohort_is_not_silently_repaired(case):
    ledger = [dict(seed=s, gold_count=10, format_eligible=True, endpoint_eligible=True) for s in range(20)]
    candidates, confirmation = list(range(20)), [30]
    if case == 'overlap': confirmation = [0]
    if case == 'missing': ledger.pop()
    if case == 'duplicate': ledger.append(deepcopy(ledger[0]))
    if case == 'shortfall': ledger[0]['endpoint_eligible'] = False
    with pytest.raises(ValueError): select_discovery(ledger, candidates, confirmation)


def metrics():
    return [dict(layer_one_based=i, discovery_oof_ncc_balanced_accuracy=.5,
                 discovery_oof_logistic_balanced_accuracy=.6, discovery_oof_rows=200,
                 discovery_fold_count=5, confirmation_ncc_balanced_accuracy=i/42) for i in range(1,43)]


def test_layer_limit_and_discovery_only_ties():
    rows = metrics()
    rows[41]['discovery_oof_ncc_balanced_accuracy'] = 1
    rows[21]['discovery_oof_logistic_balanced_accuracy'] = .7
    assert rank_layers(rows, 'Gemma4-E4B')[0]['layer_one_based'] == 22
    rows[21]['discovery_oof_logistic_balanced_accuracy'] = .6
    rows[1]['discovery_oof_ncc_balanced_accuracy'] += 1e-14
    assert rank_layers(rows, 'Gemma4-E4B')[0]['layer_one_based'] == 1


@pytest.mark.parametrize('case', ['missing_layer', 'nan', 'population'])
def test_invalid_layer_grid_rejected(case):
    rows = metrics()
    if case == 'missing_layer': rows.pop(0)
    if case == 'nan': rows[0]['discovery_oof_ncc_balanced_accuracy'] = float('nan')
    if case == 'population': rows[0]['discovery_oof_rows'] = 1056
    with pytest.raises(ValueError): rank_layers(rows, 'Qwen3-8B')


def selection():
    return dict(status='FROZEN_N10_DISCOVERY_LAYERS', confirmation_used_for_selection=False,
        cells=[dict(model='Qwen3-8B',mode='enumeration_index',gold_count=10,discovery_states=200,
                    discovery_seeds=list(range(20)),confirmation_seeds=list(range(20,30)),layer_one_based=35)])


def test_valid_selection_can_drive_patching():
    assert validate_selection(selection(),'Qwen3-8B','enumeration_index',list(range(20,30))) == 34


@pytest.mark.parametrize('case', ['leakage', 'wrong_inputs', 'overlap', 'late_layer', 'wrong_population'])
def test_patching_refuses_invalid_selection(case):
    s = selection()
    if case == 'leakage': s['confirmation_used_for_selection'] = True
    if case == 'wrong_inputs': s['cells'][0]['confirmation_seeds'][-1] = 100
    if case == 'overlap': s['cells'][0]['discovery_seeds'][-1] = 20
    if case == 'late_layer': s['cells'][0]['layer_one_based'] = 36
    if case == 'wrong_population': s['cells'][0]['gold_count'] = 9
    with pytest.raises(ValueError): validate_selection(s,'Qwen3-8B','enumeration_index',list(range(20,30)))
