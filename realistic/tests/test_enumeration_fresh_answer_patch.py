"""Assay contracts: Native low/high pairs, eligibility and intervention audits."""
import importlib.util
from pathlib import Path
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_low_high_edges_follow_common_count_grid_not_numeric_adjacency():
    module = load("prepare_enumeration_fresh_answer_patch")
    assert module.selected_edges([1, 3, 5, 6, 8, 10]) == [(1, 3), (8, 10)]
    assert module.selected_edges([3, 7, 10]) == [(3, 7), (7, 10)]
    with pytest.raises(ValueError, match="two common edges"):
        module.selected_edges([1, 10])


def test_pairing_matches_models_within_mode_and_records_shortfalls():
    module = load("prepare_enumeration_fresh_answer_patch")
    indexed = {model: {10: {n: {"request_id": f"{model}_{n}"} for n in counts},
                       11: {n: {"request_id": f"{model}_{n}"} for n in [1, 2]}}
               for model, counts in [("Q", [1, 2, 3, 8, 9, 10]), ("G", [1, 3, 8, 10])]}
    panels, common, shortfalls = module.select_pairs(indexed, [10, 11], ["Q", "G"])
    assert common["10"] == [1, 3, 8, 10]
    assert [p["alignment_key"] for p in panels["Q"]] == [p["alignment_key"] for p in panels["G"]]
    assert [p["alignment_key"] for p in panels["Q"]] == [[10, 1, 3], [10, 3, 1], [10, 8, 10], [10, 10, 8]]
    assert [p["seed"] for p in shortfalls] == [11]


def test_answer_patch_requires_correctness_but_does_not_inherit_update_gate():
    module = load("prepare_enumeration_fresh_answer_patch")
    entry = {"read_eligible": True, "format_eligible": False, "update_eligible": False}
    row = {"gold_count": 3, "trace_parse": {"gold_count": 3, "exact_count": True,
        "parser": {"trace_one_to_one": True, "trace_order_class": "reverse"}}}
    assert module.eligibility(entry, row) is None
    row["trace_parse"]["exact_count"] = False
    assert module.eligibility(entry, row) == "baseline_count_incorrect"


def test_audit_preserves_failed_donor_behavior_but_requires_valid_intervention():
    from copy import deepcopy
    module = load("run_enumeration_fresh_answer_patch")
    pair = {"receiver_count": 3}
    trials = [{"condition": condition, "layer": 2, "source_positions": [10], "prediction": prediction,
               "full_delta_norm": 2.0, "completion_text_raw": text}
              for condition, prediction, text in [("self_patch", 3, "3"), ("full_donor_patch", None, "")]]
    audits = [{"positions": [10], "hook_applications": {"2": 1}, "observer_calls": 1,
               "realized_delta_norm": norm, "raw_generation": {"completion_text_raw": text}}
              for norm, text in [(0.0, "3"), (2.0, "")]]
    module.validate_panel(trials, audits, pair, 2, 10)
    wrong = deepcopy(audits)
    wrong[1]["positions"] = [9]
    with pytest.raises(ValueError, match="hook coverage"):
        module.validate_panel(trials, wrong, pair, 2, 10)
    wrong = deepcopy(audits)
    wrong[1]["realized_delta_norm"] = 0.0
    with pytest.raises(ValueError, match="no state change"):
        module.validate_panel(trials, wrong, pair, 2, 10)
    wrong = deepcopy(trials)
    wrong[0]["prediction"] = 5
    with pytest.raises(ValueError, match="Self patch"):
        module.validate_panel(wrong, audits, pair, 2, 10)

def test_clean_regeneration_screen_is_uniform_and_keeps_native_success_guard():
    module = load("prepare_enumeration_fresh_answer_patch")
    entry = {"read_eligible": True}
    row = {"request_id": "r", "gold_count": 9,
           "trace_parse": {"gold_count": 9, "exact_count": True, "parser": {"trace_one_to_one": True}}}
    clean = {"request_id": "r", "gold_count": 9, "condition": "clean", "status": "ok",
             "prediction": 8, "exact_count": False}
    assert module.eligibility(entry, row, clean=clean, require_clean=True) == "answer_query_clean_count_incorrect"
    assert module.eligibility(entry, row) is None
    clean.update(prediction=9, exact_count=True,
                 generated={"generation_truncated": False, "stopped_on_eos": True, "generated_token_ids": [9, 0]})
    assert module.eligibility(entry, row, clean=clean, require_clean=True) is None
    clean["condition"] = "full_donor_patch"
    with pytest.raises(ValueError, match="unmodified"):
        module.eligibility(entry, row, clean=clean, require_clean=True)


def test_native_eligibility_retains_replay_error_with_explicit_technical_reference():
    from copy import deepcopy
    module = load("run_enumeration_fresh_answer_patch")
    generated = {"generated_token_ids": [8, 0], "completion_text_raw": "8", "stopped_on_eos": True, "generation_truncated": False}
    reference = {"prediction": 8, "generated": generated}
    pair = {"receiver_count": 9}
    trials = [{"condition": c, "layer": 2, "source_positions": [10], "prediction": p,
               "full_delta_norm": 2.0, "completion_text_raw": str(p)}
              for c, p in [("self_patch", 8), ("full_donor_patch", 4)]]
    audits = [{"positions": [10], "hook_applications": {"2": 1}, "observer_calls": 1,
               "realized_delta_norm": delta, "raw_generation": {**generated, "completion_text_raw": str(p), "generated_token_ids": [p, 0]}}
              for delta, p in [(0.0, 8), (2.0, 4)]]
    module.validate_panel(trials, audits, pair, 2, 10, clean_reference=reference)
    with pytest.raises(ValueError, match="Self patch"):
        module.validate_panel(trials, audits, pair, 2, 10)
    changed = deepcopy(audits)
    changed[0]["raw_generation"]["generated_token_ids"] = [18, 0]
    with pytest.raises(ValueError, match="unmodified query replay"):
        module.validate_panel(trials, changed, pair, 2, 10, clean_reference=reference)
    changed = deepcopy(audits)
    changed[0]["realized_delta_norm"] = 0.1
    with pytest.raises(ValueError, match="Self patch"):
        module.validate_panel(trials, changed, pair, 2, 10, clean_reference=reference)
