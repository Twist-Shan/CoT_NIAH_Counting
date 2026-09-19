from realistic_niah.prompt_revision_v2_1 import (
    EXPECTED_PROMPT_REVISION_REQUESTS,
    ENUMERATION_MODEL_LABELS,
    ENUMERATION_MODES,
    expected_request_ids,
    prompt_revision_shard_plan,
)


def test_prompt_revision_plan_has_registered_scope() -> None:
    plan = prompt_revision_shard_plan()
    tasks = plan["tasks"]
    assert plan["expected_shards"] == 15
    assert plan["expected_requests"] == EXPECTED_PROMPT_REVISION_REQUESTS
    assert sum(task["expected_requests"] for task in tasks) == 7_500
    enum_pairs = {
        (task["model_label"], task["prompt_mode"])
        for task in tasks
        if task["analysis_role"] == "replacement_enumeration"
    }
    assert enum_pairs == {
        (model_label, prompt_mode)
        for model_label in ENUMERATION_MODEL_LABELS
        for prompt_mode in ENUMERATION_MODES
    }
    appendix = [
        task
        for task in tasks
        if task["analysis_role"] == "appendix_strict_direct"
    ]
    assert len(appendix) == 1
    assert appendix[0]["model_label"] == "Gemma4-12B"
    assert appendix[0]["prompt_mode"] == "direct"


def test_prompt_revision_request_ids_are_unique() -> None:
    stimuli = [f"stimulus-{index}" for index in range(500)]
    request_ids: list[str] = []
    for task in prompt_revision_shard_plan()["tasks"]:
        request_ids.extend(expected_request_ids(stimuli, task))
    assert len(request_ids) == 7_500
    assert len(set(request_ids)) == 7_500
