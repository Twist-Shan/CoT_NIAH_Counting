import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("retrieve_prepare", Path(__file__).resolve().parents[1] / "scripts/prepare_enumeration_fresh_retrieve.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_rank_equal_seeds_not_equal_queries():
    def row(seed, i, a, b):
        return {"seed": seed, "query_id": str(i), "heads": [{"layer": 0, "head": 0, "mass": a}, {"layer": 0, "head": 1, "mass": b}]}
    rows = [row(1, i, .9, .1) for i in range(9)] + [row(2, 0, .0, 1.)]
    ranking = m.rank_heads(rows)
    assert ranking[0]["head"] == 1 and ranking[0]["score"] == pytest.approx(.55)
    with pytest.raises(ValueError, match="Duplicate"):
        m.rank_heads(rows + rows[:1])


def test_capacity_rule_preserves_ranking_and_nested_controls():
    # Full bank has three of four heads in L0: K=1 can match; K>=2 cannot.
    ranking = [{"layer": l, "head": h} for l, h in [(0, 0), (0, 1), (0, 2), (1, 0), (0, 3), (1, 1), (1, 2), (1, 3)]]
    bank = m.freeze_banks(ranking, [1, 2, 4], [100, 101], salt="test")
    assert bank["ks"] == [1, 2, 4] and bank["selected_heads"] == [[0, 0], [0, 1], [0, 2], [1, 0]]
    assert [x["k"] for x in bank["layer_matching_infeasible"]] == [2, 4]
    assert bank["random_control_by_k"]["1"] == "layer_matched_random"
    assert bank["random_control_by_k"]["2"] == "global_random_capacity_fallback"
    assert bank == m.freeze_banks(ranking, [1, 2, 4], [100, 101], salt="test")
    for doses in bank["random_banks"].values():
        assert len(doses["1"]) == len(doses["2"]) == len(doses["4"]) == 3
        for repeat in range(3):
            assert doses["1"][repeat] == [[0, 3]]
            assert doses["2"][repeat] == doses["4"][repeat][:2]
            assert {tuple(h) for h in doses["4"][repeat]} == {(0, 3), (1, 1), (1, 2), (1, 3)}


def test_highest_common_count_ignores_correctness_and_nonfinal_queries():
    def q(seed, n, to=None):
        return {"seed": seed, "gold_count": n, "from_occurrence": n - 1 if to is None else to - 1,
                "to_occurrence": n if to is None else to, "split": "confirmation", "baseline_correct": False}
    panels, missing = m.choose_confirmation({"q": [q(1, 10), q(1, 8), q(2, 5)],
                                           "g": [q(1, 8), q(1, 10, 3), q(2, 5)]}, [1, 2, 3])
    assert [x["alignment_key"] for x in panels["q"]] == [[1, 8, 7, 8], [2, 5, 4, 5]]
    assert missing == [3] and panels["q"] == panels["g"]


def test_nan_and_inconsistent_head_universe_fail():
    a = {"seed": 1, "query_id": "a", "heads": [{"layer": 0, "head": 0, "mass": .5}]}
    b = {"seed": 2, "query_id": "b", "heads": [{"layer": 0, "head": 1, "mass": .5}]}
    with pytest.raises(ValueError, match="Inconsistent"):
        m.rank_heads([a, b])
    a["heads"][0]["mass"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        m.rank_heads([a, b])


def test_all_feasible_random_prefixes_match_layers_and_exclude_full_bank():
    ranking = [{"layer": l, "head": h} for h in range(8) for l in range(2)]
    bank = m.freeze_banks(ranking, [1, 2, 4, 8], [100], salt="balanced")
    assert not bank["layer_matching_infeasible"]
    assert set(bank["random_control_by_k"].values()) == {"layer_matched_random"}
    for i in range(3):
        doses = bank["random_banks"]["100"]
        assert doses["4"][i] == doses["8"][i][:4]
        assert all(h >= 4 for _l, h in doses["8"][i])
        assert [h[0] for h in doses["8"][i]] == [h[0] for h in bank["selected_heads"]]


def test_persistent_hook_audit_rejects_unmasked_decode():
    runner_spec = importlib.util.spec_from_file_location("retrieve_runner", Path(__file__).resolve().parents[1] / "scripts/run_enumeration_fresh_retrieve.py")
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    task = {"heads": [[2, 1]]}
    query = {"seed": 1, "gold_count": 10, "query_position": 123}
    trial = {"seed": 1, "gold_count": 10, "intervention_full_sequence_token_indices": [123], "heads": [[2, 1]],
        "head_ablation_decode_steps_requested": -1, "generated_token_ids": [7, 8, 9],
        "head_ablation_prefill_layer_applications": {"2": 1}, "head_ablation_selected_post_zero_max_abs": 0,
        "head_ablation_decode_layer_applications": {"2": 2}}
    runner.validate_trial(trial, task, query, {"2": .4})
    trial["head_ablation_decode_layer_applications"]["2"] = 1
    with pytest.raises(ValueError, match="persist"):
        runner.validate_trial(trial, task, query, {"2": .4})
