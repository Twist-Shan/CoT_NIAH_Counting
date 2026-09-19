import pytest
try:
    from launch_enumeration_native_update import mode_seeds
except ImportError:
    from scripts.launch_enumeration_native_update import mode_seeds


def test_mode_quota_is_independent_of_other_modes_and_final_correctness():
    ledger = [{"seed": seed, "gold_count": 10, "update_eligible": True,
               "baseline_exact_count": False} for seed in range(20)]
    assert mode_seeds(ledger, list(range(20)), 10) == list(range(10))
    # One unrelated Thinking cohort of size three cannot change this result.
    assert len(mode_seeds(ledger, list(range(20)), 10)) == 10


def test_mode_selection_is_fixed_order_and_reports_shortfall():
    ledger = [{"seed": seed, "gold_count": 10, "update_eligible": seed in (2, 5, 8)} for seed in range(10)]
    assert mode_seeds(ledger, list(reversed(range(10))), 10) == [2, 5, 8]
    with pytest.raises(ValueError, match="Duplicate"):
        mode_seeds(ledger + [ledger[0]], list(range(10)), 10)
    with pytest.raises(ValueError, match="Missing"):
        mode_seeds(ledger[:-1], list(range(10)), 10)
