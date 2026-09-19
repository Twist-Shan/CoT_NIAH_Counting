from collections import Counter

from realistic_niah.sharding import (
    EXPECTED_FORMAL_REQUESTS,
    expected_request_ids,
    formal_shard_plan,
)


def test_formal_shard_plan_has_exact_registered_matrix() -> None:
    plan = formal_shard_plan()
    tasks = plan["tasks"]

    assert plan["expected_shards"] == 29
    assert plan["expected_requests"] == EXPECTED_FORMAL_REQUESTS == 14_500
    assert len({task["task_id"] for task in tasks}) == 29
    assert all(task["expected_requests"] == 500 for task in tasks)

    counts = Counter(task["model_label"] for task in tasks)
    assert counts == {
        "Qwen3-1.7B": 4,
        "Qwen3-4B": 4,
        "Qwen3-8B": 4,
        "Qwen3-32B": 4,
        "Gemma4-E4B": 4,
        "Gemma4-12B": 4,
        "DeepSeek-R1-0528-Qwen3-8B": 1,
        "GLM-Z1-9B-0414": 1,
        "GLM-4-9B-0414": 3,
    }

    modes = {
        task["model_label"]: {
            candidate["prompt_mode"]
            for candidate in tasks
            if candidate["model_label"] == task["model_label"]
        }
        for task in tasks
    }
    assert modes["DeepSeek-R1-0528-Qwen3-8B"] == {"native_thinking"}
    assert modes["GLM-Z1-9B-0414"] == {"native_thinking"}
    assert modes["GLM-4-9B-0414"] == {
        "direct",
        "enumeration_index",
        "enumeration_bullet",
    }


def test_formal_shards_cover_14500_unique_request_ids() -> None:
    stimulus_ids = tuple(f"stimulus-{index:03d}" for index in range(500))
    all_ids: list[str] = []

    for task in formal_shard_plan()["tasks"]:
        task_ids = expected_request_ids(stimulus_ids, task)
        assert len(task_ids) == 500
        assert len(set(task_ids)) == 500
        all_ids.extend(task_ids)

    assert len(all_ids) == EXPECTED_FORMAL_REQUESTS
    assert len(set(all_ids)) == EXPECTED_FORMAL_REQUESTS


def test_expected_request_id_includes_frozen_layout() -> None:
    task = next(
        task
        for task in formal_shard_plan()["tasks"]
        if task["model_label"] == "Qwen3-8B"
        and task["prompt_mode"] == "native_thinking"
    )
    assert expected_request_ids(("abc",), task) == (
        "Qwen3-8B/native_thinking/cue_before_query_after/abc",
    )
