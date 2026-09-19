import json
import pytest

from realistic_niah_v5.same_site_progress_transplant import generated_bullet_city_ordinals
from realistic_niah_v6.update_audit import CONDITIONS, jsonl, matched_trials, verify_n10_runtime, verify_trial


@pytest.mark.parametrize("alter", [lambda rows: rows[:-1], lambda rows: rows + rows[:1]])
def test_missing_or_duplicate_control_is_not_analyzed(alter):
    rows = [{"seed": 7, "condition": c} for c in CONDITIONS]
    with pytest.raises(ValueError, match="three-condition"):
        matched_trials(alter(rows), [7])


def sample():
    cities = [f"City{i}" for i in range(1, 11)]
    text = "- City5\n- City6"
    trial = dict(seed=7, condition="donor_to_receiver", layer=2, gold_count=10,
                 donor_occurrence_k=4, receiver_occurrence_j=3, donor_successor=5, receiver_successor=4,
                 patch_scope="fixed_suffix", patch_applications=1, patch_width=1, requested_patch_width=1,
                 donor_item_coverage=.2, receiver_item_coverage=.2, completion_text=text,
                 generated_token_count=2, stopped_on_eos=True, generation_truncated=False,
                 shared_commit_position=19, realized_patch_delta_norm=2.,
                 greedy_donor_successor_adoption=True, greedy_receiver_successor_retention=False,
                 sum_logprob_scores=[-2., -1.], mean_logprob_scores=[-1., -.5], donor_vs_receiver_sum_logodds=1.,
                 **generated_bullet_city_ordinals(text, cities))
    raw = {k: trial[k] for k in ("seed", "condition", "completion_text", "generated_token_count", "stopped_on_eos", "generation_truncated")}
    raw.update(generated_token_ids=[42, 2], generation_eos_token_ids=[2], query_position=19)
    hooks = [dict(seed=7, layer=2, site=19, width=1, applications=1, delta_norm=2., input_ids_sha256="abc") for _ in range(2)]
    args = dict(cities=cities, parser=generated_bullet_city_ordinals, layer=2, k=4,
                direction="forward", scope="endpoint", max_tokens=3)
    return trial, raw, hooks, args


def test_saved_success_cannot_override_actual_generated_text():
    trial, raw, hooks, args = sample()
    verify_trial(trial, raw, hooks, **args)
    trial["generated_known_city_ordinals_any_surface"] = [5, 6, 7]
    with pytest.raises(ValueError, match="Frozen city parser"):
        verify_trial(trial, raw, hooks, **args)


def test_nonzero_target_is_required_at_both_prefills():
    trial, raw, hooks, args = sample()
    hooks[1]["delta_norm"] = 0.
    with pytest.raises(ValueError, match="actual patch norm"):
        verify_trial(trial, raw, hooks, **args)


def test_truncation_flags_must_match_tokens_and_budget():
    trial, raw, hooks, args = sample()
    trial["generation_truncated"] = raw["generation_truncated"] = True
    with pytest.raises(ValueError, match="Truncation flag"):
        verify_trial(trial, raw, hooks, **args)


def test_unicode_line_separator_is_not_a_jsonl_record_boundary(tmp_path):
    path = tmp_path / "raw.jsonl"
    row = {"completion_text": "City5\u2028City6\u2029City7"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert jsonl(path) == [row]


@pytest.mark.parametrize("field,value", [
    ("n10_selection_manifest_sha256", "old-selection"),
    ("patch_layer_zero_based", 16),
    ("runner_sha256", "old-wrapper"),
    ("seeds", [8, 7]),
    ("model_source_sha256", "different-kv-implementation"),
    ("command", ["worker.py", "--selection-manifest", "/fresh_native_update_v1/selection_manifest.json"]),
])
def test_n10_refuses_old_or_mismatched_execution(field, value):
    previous = {k: "unchanged" for k in ("model_revision", "model_source_sha256", "dtype", "backend", "torch",
        "transformers", "layers", "jobs", "attention_readout", "all_three_conditions_generate")}
    runtime = dict(previous, n10_selection_manifest_sha256="new-selection", runner_sha256="new-wrapper",
        patch_layer_zero_based=21, seeds=[7, 8],
        command=["worker.py", "--selection-manifest", "/fresh_n10_update_v1/selection_manifest.json"])
    kwargs = dict(selection_sha256="new-selection", runner_sha256="new-wrapper", seeds=[7, 8], layer=21)
    verify_n10_runtime(runtime, previous, **kwargs)
    runtime[field] = value
    with pytest.raises(ValueError):
        verify_n10_runtime(runtime, previous, **kwargs)
