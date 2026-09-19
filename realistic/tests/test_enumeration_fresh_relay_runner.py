from copy import deepcopy
import pytest
from scripts.run_enumeration_fresh_relay import selected_relay_tasks, validate_relay_panel


def test_relay_smoke_uses_first_fixed_pair_at_every_available_width():
    cohorts = {"models": {"m": {"pairs": {"4": ["a", "b"], "8": ["b"]}}}}
    assert selected_relay_tasks(cohorts, "m", [4, 8], smoke=True) == [
        {"width": 4, "pair_id": "a"}, {"width": 8, "pair_id": "b"}]
    assert len(selected_relay_tasks(cohorts, "m", [4, 8], smoke=False)) == 3
    cohorts["models"]["m"]["pairs"]["8"] = []
    assert len(selected_relay_tasks(cohorts, "m", [4, 8], smoke=True)) == 1


def test_native_relay_primary_uses_each_mode_without_intersection_shrinkage():
    cohorts = {"cohort_policy": "native_assay_specific_per_mode", "models": {"m": {
        "pairs": {"4": [], "8": []}, "per_mode": {"index": {"pairs": {"4": ["a", "b"], "8": ["b"]}}}}}}
    assert len(selected_relay_tasks(cohorts, "m", [4, 8], smoke=False, mode="index")) == 3
    with pytest.raises(ValueError, match="explicit mode"):
        selected_relay_tasks(cohorts, "m", [4, 8], smoke=False)


def panel():
    pair = {"widths": {"4": {"receiver_patch_positions": [8, 9, 10, 11], "donor_patch_positions": [2, 3, 4, 5]}},
            "query_reset_positions": [14], "post_terminal_reset_positions": [12, 13, 14]}
    trials, audits, raw = [], [], []
    for source in ("self_patch", "full_donor_patch"):
        for relay in ("natural_relay", "answer_query_clean_reset", "post_terminal_suffix_clean_reset"):
            trials.append({"source_condition": source, "relay_condition": relay,
                **pair["widths"]["4"], "patch_token_count": 4, "patch_token_count_capped_by_shorter_span": False,
                "source_patch_hook_applications": {"1": 1, "2": 1},
                "relay_positions": pair["post_terminal_reset_positions"] if relay.startswith("post") else [14],
                "relay_reset_hook_applications": int(relay != "natural_relay"),
                "relay_reset_realized_fro_norm": 0.0, "expected_count_utility": -2.0,
                "completion_text_raw": "", "prediction": None, "generation_truncated": True})
            audits.append({"source_realized_delta_norms": {"1": 0.0 if source == "self_patch" else 1.0,
                                                          "2": 0.0 if source == "self_patch" else 2.0}})
            raw.append({"completion_text_raw": ""})
    return trials, audits, raw, pair


def test_relay_accepts_failed_behavior_but_rejects_broken_interventions():
    rows, audits, raw, pair = panel()
    validate_relay_panel(rows, audits, raw, pair, 4, [1, 2])
    bad = deepcopy(audits)
    bad[0]["source_realized_delta_norms"]["1"] = 0.5
    with pytest.raises(ValueError, match="Self"):
        validate_relay_panel(rows, bad, raw, pair, 4, [1, 2])
    bad = deepcopy(audits)
    bad[-1]["source_realized_delta_norms"] = {"1": 0.0, "2": 0.0}
    with pytest.raises(ValueError, match="no state change"):
        validate_relay_panel(rows, bad, raw, pair, 4, [1, 2])
    bad = deepcopy(rows)
    bad[0]["receiver_patch_positions"] = [7, 8, 9, 10]
    with pytest.raises(ValueError, match="geometry"):
        validate_relay_panel(bad, audits, raw, pair, 4, [1, 2])
    with pytest.raises(ValueError, match="incomplete"):
        validate_relay_panel(rows[:-1], audits, raw, pair, 4, [1, 2])


def test_relay_requires_exact_hook_layers_and_uncapped_width():
    rows, audits, raw, pair = panel()
    bad = deepcopy(rows)
    bad[0]["source_patch_hook_applications"] = {"1": 1}
    with pytest.raises(ValueError, match="every registered layer"):
        validate_relay_panel(bad, audits, raw, pair, 4, [1, 2])
    bad = deepcopy(rows)
    bad[-1]["patch_token_count_capped_by_shorter_span"] = True
    with pytest.raises(ValueError, match="capped"):
        validate_relay_panel(bad, audits, raw, pair, 4, [1, 2])
