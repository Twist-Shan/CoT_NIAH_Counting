import pytest

from realistic_niah_v6.fresh_behavior_audit import seed_mean, unique_grid, verify_generation, load_frozen_city_scorer


def test_seed_means_do_not_count_repeats_as_independent_samples():
    stat = seed_mean([(1, 0), (2, 1), (2, 1), (2, 1)], repetitions=1000)
    assert stat["estimate"] == .5
    assert stat["independent_seed_count"] == 2
    assert stat["observation_count"] == 4


def test_one_seed_is_descriptive():
    stat = seed_mean([(1, 0), (1, 1)])
    assert stat["estimate"] == .5 and stat["ci95"] is None


@pytest.mark.parametrize("rows", [[], [(1, float("nan"))], [("1", 1)]])
def test_bad_statistics_inputs_fail(rows):
    with pytest.raises(ValueError):
        seed_mean(rows)


@pytest.mark.parametrize("rows", [[{"x": 1}], [{"x": 1}, {"x": 1}]])
def test_missing_or_duplicate_conditions_fail(rows):
    with pytest.raises(ValueError):
        unique_grid(rows, ["x"], {(1,), (2,)})


def test_eos_at_budget_is_not_truncation():
    raw = {"generated_token_ids": [2, 3], "generated_token_count": 2, "generation_eos_token_ids": [3],
           "stopped_on_eos": True, "generation_truncated": False, "completion_text": "2", "completion_text_raw": "2<eos>"}
    verify_generation(raw, 2)
    with pytest.raises(ValueError):
        verify_generation({**raw, "generation_truncated": True}, 2)


def test_unexplained_early_stop_fails():
    raw = {"generated_token_ids": [2], "generated_token_count": 1, "generation_eos_token_ids": [3],
           "stopped_on_eos": False, "generation_truncated": False, "completion_text": "2", "completion_text_raw": "2"}
    with pytest.raises(ValueError):
        verify_generation(raw, 16)


def test_parser_loader_rejects_missing_definitions(tmp_path):
    path = tmp_path / "incomplete.py"
    path.write_text("x = 1\n")
    with pytest.raises(ValueError):
        load_frozen_city_scorer(path)
