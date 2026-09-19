import json
from pathlib import Path
import pytest
from realistic_niah_v4_4_5.reverse_full import (expected_cells, frozen_contract,
    load_journal, history_gate, self_gate)


def row(condition='clean', layer=-1):
    return {'seed':1234,'gold_count':1,'condition':condition,'patch_layer':layer,'cell_audit':'PASS'}


def test_full_grid_matches_original_population_plus_validation():
    q=len(expected_cells(range(36)));g=len(expected_cells(range(42)))
    assert (q,g)==(183,213)
    assert 300*q==54900 and 300*g==63900
    assert 300*(3+3*36)+300*(3+3*42)==72000


def test_resume_rejects_changed_contract(tmp_path):
    p=tmp_path/'contract.json'
    frozen_contract(p,{'revision':'a','backend':'eager'})
    frozen_contract(p,{'revision':'a','backend':'eager'})
    with pytest.raises(RuntimeError):frozen_contract(p,{'revision':'a','backend':'sdpa'})


def test_journal_recovers_only_uncommitted_tail_with_evidence(tmp_path):
    p=tmp_path/'cells.jsonl'
    valid=json.dumps(row())+'\n'
    p.write_text(valid+'{"seed":1234',encoding='utf-8')
    loaded=load_journal(p,seed=1234,count=1,layers=[0])
    assert set(loaded)=={('clean',-1)}
    assert p.read_text()==valid
    archived=list(tmp_path.glob('*.interrupted_tail_*'))
    assert len(archived)==1 and archived[0].read_text().endswith('{"seed":1234')


def test_journal_rejects_duplicates_foreign_and_unvalidated_rows(tmp_path):
    p=tmp_path/'cells.jsonl'
    for rows in ([row(),row()],[{**row(),'seed':1235}],[row('unknown',0)],
                 [{**row(),'cell_audit':'FAILED'}]):
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with pytest.raises(RuntimeError):load_journal(p,seed=1234,count=1,layers=[0])
    p.write_text(json.dumps(row())+'\n{broken}\n')
    with pytest.raises(json.JSONDecodeError):load_journal(p,seed=1234,count=1,layers=[0])


def test_numerical_gates_are_not_relaxed_by_legacy_handling():
    exact={'expected_count_delta':0,'probability_tv':0,'strict_agrees':True}
    assert self_gate(exact) and history_gate(exact)
    assert not history_gate({**exact,'probability_tv':0.011499169})
    assert not history_gate({**exact,'strict_agrees':False})
    assert not self_gate({**exact,'expected_count_delta':0.00002})


def test_sparse_validation_keeps_all_primary_cells():
    import json
    from pathlib import Path
    from realistic_niah_v4_4_5.reverse_full import expected_cells
    design = json.loads((Path(__file__).resolve().parents[1] / "configs/realistic_niah_v4_4_5_reverse_full.json").read_text())
    total = 0
    for model, n in design["num_layers"].items():
        cells = expected_cells(range(n), design["validation_layers"][model])
        assert sum(name.startswith("reverse") for name, layer in cells) == 3*n
        assert sum(name == "self_needle_full" for name, layer in cells) == 4
        assert sum(name == "restore_needle_full" for name, layer in cells) == 4
        assert len(cells)*300 == design["all_rows_with_validation"][model]
        total += len(cells)*300
    assert total == 76800
