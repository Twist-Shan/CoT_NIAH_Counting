import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v58_protected_control_policy import protected_control_plan


def test_frozen_global_ranking_all_k():
    ranking=[(1,3),(1,0),(1,4),(1,6),(1,1),(2,6),(2,3),(1,5)]
    expected_counts=[6,10,4,4,10,70,150,90]
    assert sum(expected_counts)==344
    expected_overlap=[0,0,0,1,3,3,3,5]
    for k in range(1,9):
        selected=set(ranking[:k])
        plan=protected_control_plan(ranking[:k])
        assert plan['actual_unique_controls']==expected_counts[k-1]
        assert plan['minimum_overlap']==expected_overlap[k-1]
        banks=[tuple(map(tuple,c['heads'])) for c in plan['controls']]
        assert len(set(banks))==len(banks)==expected_counts[k-1]
        for bank in banks:
            assert (1,2) not in bank
            assert len(bank)==k
            assert len(selected&set(bank))==expected_overlap[k-1]
            assert {l:sum(ll==l for ll,h in bank) for l,h in bank}=={l:sum(ll==l for ll,h in selected) for l,h in selected}


def test_protected_selected_rejected():
    try:
        protected_control_plan([(1,2)])
    except ValueError:
        return
    raise AssertionError('Protection must not silently alter selected heads')
