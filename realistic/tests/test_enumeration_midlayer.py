"""Protect independence and preserve the complete control grid."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('midlayer',Path(__file__).resolve().parents[1]/'scripts/run_enumeration_midlayer_update.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_selection_retains_incorrect_baselines_and_fails_overlap():
    cfg=dict(candidate_seed_start=100,candidate_seed_stop_exclusive=104,confirmation_quota=2)
    ledger=[dict(seed=s,gold_count=10,update_eligible=s!=100,baseline_exact_count=False) for s in range(100,104)]
    assert mod.choose(ledger,cfg,{90})==[101,102]
    assert mod.choose(ledger,cfg,{102})==[101,103]
    with pytest.raises(ValueError,match='Insufficient'):mod.choose(ledger,cfg,{101,102})
    with pytest.raises(ValueError,match='missing'):mod.choose(ledger[:-1],cfg,set())

def test_full_grid_and_smoke_precede_formal():
    cfg=dict(donor_k=[4,6,8],directions=['forward','backward'],scopes=['endpoint','four_token_tail','item_span'])
    jobs=mod.jobs(cfg,list(range(10)))
    assert len(jobs)==36 and len({j['id'] for j in jobs})==36
    assert sum(j['expected_rows'] for j in jobs[:18])==54
    assert sum(j['expected_rows'] for j in jobs[18:])==540
    assert all(j['phase']=='smoke' for j in jobs[:18]) and all(j['phase']=='formal' for j in jobs[18:])
    assert all(j['j']==j['k']+(-1 if j['direction']=='forward' else 1) for j in jobs)
