from types import SimpleNamespace
import torch

from test_v58_commit_query import fixture_model
from run_v58_top1to8_aligned import final_query_prefix, marker_outcome, generate
from freeze_v58_control_policy import matched_control_plan


def test_final_query_excludes_prompt_separator_and_does_not_fabricate_missing_anchor():
    seq = ['<Sep>', '<Think>', '<Sep>', '<CH_0061>', '<Sep>', '<CH_0062>', '</Think>', '<Ans>', '<2>', '<EOS>']
    assert final_query_prefix(seq, 2) == seq[:5]
    assert final_query_prefix(seq, 3) is None
    assert final_query_prefix(seq, 1) == seq[:3]
    assert final_query_prefix(['<Think>', '<Sep>', '<Ans>', '<Sep>'], 2) is None


def test_earlier_wrong_marker_is_not_an_outcome_filter():
    seq = ['<Think>', '<Sep>', '<CH_007A>', '<Sep>']
    assert final_query_prefix(seq, 2) == seq


def test_target_marker_success_is_independent_of_later_count_and_stop():
    r = marker_outcome(['<CH_0061>', '</Think>', '<Ans>', '<9>', '<Ans>'], '<CH_0061>')
    assert r['next_marker_correct'] == r['immediate_marker_correct'] == 1


def test_no_search_for_later_correct_marker_or_for_markers_after_answer():
    assert marker_outcome(['<CH_0062>', '<CH_0061>'], '<CH_0061>')['next_marker_correct'] == 0
    assert marker_outcome(['<Ans>', '<CH_0061>'], '<CH_0061>')['next_marker_identifiable'] == 0
    r = marker_outcome(['<Sep>', '<CH_0061>'], '<CH_0061>')
    assert r['next_marker_correct'] == 1 and r['immediate_marker_valid'] == 0


def test_exhaustive_control_cardinality_and_uniqueness_across_layers():
    selected = [(1,h) for h in range(6)] + [(2,0),(2,1)]
    p = matched_control_plan(selected, exhaustive=True)
    assert len(p['controls']) == p['feasible_unique_controls'] == 225
    banks = {tuple(map(tuple,c['heads'])) for c in p['controls']}
    assert len(banks) == 225
    assert all(c['overlap_count'] == 4 for c in p['controls'])


def test_variable_length_generation_matches_individual_runs_and_checks_real_hooks():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model,vocab,e,item=fixture_model()
    cfg=SimpleNamespace(device='cpu',n_embd=64,n_head=8,n_positions=model.config.n_positions)
    records=[]
    for j in [2,5]:
        q=item.spans.trace_marker_positions[j-1]-1
        records.append({'mode':'thinking','key':f'key{j}','count':10,
                        'prefix_tokens':item.tokens[:q+1],'expected_marker':e.needle_markers[j-1],
                        'primary_eligible':True})
    lookup={r['key']:e for r in records}
    for scope in ['sustained','answer_query_only']:
        individual,_=generate(model,cfg,vocab,records,lookup,[(1,0),(4,5)],scope,1,4)
        batched,checks=generate(model,cfg,vocab,records,lookup,[(1,0),(4,5)],scope,2,4,verify=True)
        assert individual.generated_tokens.tolist()==batched.generated_tokens.tolist()
        assert checks['hook_calls']>0
