import copy
import pytest
from realistic_niah_v6.read_audit import cluster_mean, json_sha, matched_trials, verify_masks, verify_trial


def test_missing_and_duplicate_arms_are_not_counted_as_failures():
    rows = [{"seed": 1, "gold_count": 2, "condition": c} for c in (
        "clean", "prompt_all_blank", "prompt_records_blank", "trace_all_blank", "prompt_and_trace_blank")]
    assert len(matched_trials(rows, {(1, 2)})) == 5
    with pytest.raises(ValueError, match="Unmatched"):
        matched_trials(rows[:-1], {(1, 2)})
    with pytest.raises(ValueError, match="Duplicate"):
        matched_trials(rows + rows[:1], {(1, 2)})


def test_seed_bootstrap_retains_within_seed_denominators():
    result = cluster_mean([(1, 1), (2, 0), (2, 0), (2, 0)], repetitions=500)
    assert result["value"] == 0.25
    assert result["independent_seed_count"] == 2
    assert result["denominator"] == 4
    assert result == cluster_mean([(1, 1), (2, 0), (2, 0), (2, 0)], repetitions=500)
    assert cluster_mean([(1, 1), (1, 0)])["ci95"] is None


def fixture():
    geometry = {"prompt_token_count": 4, "query_position": 6, "sequence_length": 7,
                "prompt_record_spans": [[1, 3]], "conditions": {}}
    masks = {"clean": [], "prompt_all_blank": [1, 2, 3], "prompt_records_blank": [1, 2],
             "trace_all_blank": [4, 5], "prompt_and_trace_blank": [1, 2, 3, 4, 5]}
    geometry["conditions"] = {k: {"positions_sha256": json_sha(v), "token_count": len(v)}
                              for k, v in masks.items()}
    entry = {"seed": 1, "gold_count": 2, "request_id": "r", "geometry_sha256": "g"}
    generated = {"generated_token_ids": [9, 0], "generated_token_count": 2,
                 "generation_eos_token_ids": [0], "stopped_on_eos": True, "generation_truncated": False,
                 "completion_text_raw": "2<eos>", "completion_text": "2", "full_answer_text": "Total:2"}
    trial = {**entry, "condition": "trace_all_blank", "status": "ok", "blank_token_count": 2,
             "blank_positions_sha256": json_sha([4, 5]), "embedding_zero_probe": [0.0],
             "blank_hook_audit": {"blank_embedding_hook_applications": 1,
                                 "blank_layer_hook_applications": {"0": 1, "1": 1},
                                 "blank_prefill_sequence_lengths": [7]},
             "generated": generated, "prediction": 2, "exact_count": True, "signed_error": 0,
             "absolute_error": 0, "completion_text_raw": "2<eos>", "generated_token_count": 2,
             "generation_truncated": False}
    return geometry, entry, trial


def test_audit_rejects_query_blanking_even_with_consistent_hash():
    geometry, _, _ = fixture()
    verify_masks(geometry)
    geometry["conditions"]["trace_all_blank"] = {"positions_sha256": json_sha([4, 5, 6]), "token_count": 3}
    with pytest.raises(ValueError, match="Mask mismatch"):
        verify_masks(geometry)


@pytest.mark.parametrize("fault", ["metric", "hook", "probe", "eos"])
def test_saved_success_requires_raw_and_hook_evidence(fault):
    geometry, entry, trial = fixture()
    kwargs = dict(layers=2, parse_total=lambda text: int(text.split(":")[-1]), max_new_tokens=2)
    verify_trial(trial, entry, geometry, **kwargs)
    broken = copy.deepcopy(trial)
    if fault == "metric":
        broken["exact_count"] = False
    elif fault == "hook":
        broken["blank_hook_audit"]["blank_layer_hook_applications"]["1"] = 0
    elif fault == "probe":
        broken["embedding_zero_probe"] = [0.01]
    else:
        broken["generated"]["stopped_on_eos"] = False
    with pytest.raises(ValueError):
        verify_trial(broken, entry, geometry, **kwargs)

