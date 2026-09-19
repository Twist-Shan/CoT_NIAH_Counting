from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from realistic_niah.olmo3_extension import (
    EXPECTED_EXTENSION_REQUESTS,
    EXPECTED_EXTENSION_SHARDS,
    EXPECTED_STIMULI_PER_SHARD,
    SOURCE_FORMAL_STIMULI_SHA256,
    expected_request_ids,
    olmo3_extension_plan,
)
from realistic_niah.parsing import evaluate_generation
from realistic_niah.prompts import (
    reasoning_expected,
    render_generation_prompt,
    resolve_model_spec,
)
from realistic_niah.runner import build_requests, decoding_config
from realistic_niah.sharding import formal_shard_plan
from realistic_niah.spec import (
    ALL_MODEL_SPECS,
    EXTENSION_MODEL_REVISIONS,
    EXTENSION_MODEL_SPECS,
    MODEL_SPECS,
    REASONING_EXTENSION_MODEL_SPECS,
)


def _stimulus(index: int) -> dict:
    return {
        "stimulus_id": f"T2000_N3_seed{index}",
        "passage": f"passage {index}",
        "seed": index,
    }


def test_original_formal_panel_remains_frozen() -> None:
    formal = formal_shard_plan()

    assert len(MODEL_SPECS) == 9
    assert formal["expected_shards"] == 29
    assert formal["expected_requests"] == 14_500
    assert not set(EXTENSION_MODEL_SPECS).intersection(MODEL_SPECS)


def test_olmo3_registry_uses_two_immutable_official_checkpoints() -> None:
    instruct = EXTENSION_MODEL_SPECS["Olmo3-7B-Instruct"]
    think = EXTENSION_MODEL_SPECS["Olmo3-7B-Think"]

    assert instruct.model_id == "allenai/Olmo-3-7B-Instruct"
    assert instruct.reasoning_policy == "off_only"
    assert instruct.prompt_modes == (
        "direct",
        "enumeration_index",
        "enumeration_bullet",
    )
    assert think.model_id == "allenai/Olmo-3-7B-Think"
    assert think.reasoning_policy == "always_on"
    assert think.prompt_modes == ("native_thinking",)
    assert set(EXTENSION_MODEL_REVISIONS) == set(EXTENSION_MODEL_SPECS)
    assert all(len(revision) == 40 for revision in EXTENSION_MODEL_REVISIONS.values())
    assert set(ALL_MODEL_SPECS) == (
        set(MODEL_SPECS)
        | set(EXTENSION_MODEL_SPECS)
        | set(REASONING_EXTENSION_MODEL_SPECS)
    )


def test_resolver_accepts_olmo_labels_and_repo_ids() -> None:
    for label, spec in EXTENSION_MODEL_SPECS.items():
        assert resolve_model_spec(label) is spec
        assert resolve_model_spec(spec.model_id) is spec


def test_olmo3_extension_plan_has_four_shards_and_2000_requests() -> None:
    plan = olmo3_extension_plan()
    tasks = plan["tasks"]

    assert plan["expected_shards"] == EXPECTED_EXTENSION_SHARDS == 4
    assert plan["expected_requests"] == EXPECTED_EXTENSION_REQUESTS == 2_000
    assert plan["expected_stimuli_per_shard"] == EXPECTED_STIMULI_PER_SHARD == 500
    assert plan["source_stimuli_sha256"] == SOURCE_FORMAL_STIMULI_SHA256
    assert len({task["task_id"] for task in tasks}) == 4
    assert Counter(task["model_label"] for task in tasks) == {
        "Olmo3-7B-Instruct": 3,
        "Olmo3-7B-Think": 1,
    }
    assert {
        task["prompt_mode"]
        for task in tasks
        if task["model_label"] == "Olmo3-7B-Instruct"
    } == {"direct", "enumeration_index", "enumeration_bullet"}
    assert {
        task["prompt_mode"]
        for task in tasks
        if task["model_label"] == "Olmo3-7B-Think"
    } == {"native_thinking"}


def test_olmo3_extension_request_ids_are_globally_unique() -> None:
    stimulus_ids = tuple(f"stimulus-{index:03d}" for index in range(500))
    request_ids: list[str] = []

    for task in olmo3_extension_plan()["tasks"]:
        task_ids = expected_request_ids(stimulus_ids, task)
        assert len(task_ids) == len(set(task_ids)) == 500
        request_ids.extend(task_ids)

    assert len(request_ids) == len(set(request_ids)) == 2_000


def test_olmo3_build_requests_enforces_checkpoint_mode_split() -> None:
    stimuli = [_stimulus(index) for index in range(2)]
    instruct = build_requests(
        stimuli,
        model_spec=EXTENSION_MODEL_SPECS["Olmo3-7B-Instruct"],
    )
    think = build_requests(
        stimuli,
        model_spec=EXTENSION_MODEL_SPECS["Olmo3-7B-Think"],
    )

    assert len(instruct) == 6
    assert len(think) == 2
    with pytest.raises(ValueError):
        build_requests(
            stimuli,
            model_spec=EXTENSION_MODEL_SPECS["Olmo3-7B-Instruct"],
            prompt_modes=("native_thinking",),
        )


def test_olmo3_decoding_matches_registered_v2_protocol() -> None:
    instruct = EXTENSION_MODEL_SPECS["Olmo3-7B-Instruct"]
    think = EXTENSION_MODEL_SPECS["Olmo3-7B-Think"]

    assert decoding_config(instruct, "direct").max_tokens == 64
    assert decoding_config(instruct, "direct").temperature == 0.0
    assert decoding_config(instruct, "enumeration_index").max_tokens == 1536
    native = decoding_config(think, "native_thinking")
    assert native.max_tokens == 4096
    assert native.temperature == 0.6
    assert native.top_p == 0.95
    assert native.top_k == -1


def test_olmo_templates_do_not_receive_qwen_enable_thinking_kwarg() -> None:
    class RecordingTokenizer:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def apply_chat_template(
            self,
            messages: list[dict],
            **kwargs: object,
        ) -> str:
            self.calls.append(kwargs)
            return "rendered"

    messages = [{"role": "user", "content": "test"}]
    for label, mode in (
        ("Olmo3-7B-Instruct", "direct"),
        ("Olmo3-7B-Think", "native_thinking"),
    ):
        tokenizer = RecordingTokenizer()
        spec = EXTENSION_MODEL_SPECS[label]
        render_generation_prompt(
            tokenizer,
            messages,
            model_spec=spec,
            prompt_mode=mode,
        )
        assert "enable_thinking" not in tokenizer.calls[0]

    assert reasoning_expected(
        EXTENSION_MODEL_SPECS["Olmo3-7B-Instruct"],
        "direct",
    ) is False
    assert reasoning_expected(
        EXTENSION_MODEL_SPECS["Olmo3-7B-Think"],
        "native_thinking",
    ) is True


def test_olmo_think_output_is_split_and_scored_without_special_case() -> None:
    evaluation = evaluate_generation(
        "Counted three records.</think>\nTotal: 3<|endoftext|>",
        prompt_mode="native_thinking",
        reasoning_expected=True,
        gold_pairs=[
            {"city": "A", "score": 1},
            {"city": "B", "score": 2},
            {"city": "C", "score": 3},
        ],
        finish_reason="stop",
        output_tokens=16,
        max_output_tokens=4096,
    )

    assert evaluation["reasoning_text"] == "Counted three records."
    assert evaluation["final_text"] == "Total: 3<|endoftext|>"
    assert evaluation["predicted_count"] == 3
    assert evaluation["response_format_compliant"] is True
    assert evaluation["registered_success"] is True


def test_json_config_matches_registered_extension() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            repo_root / "configs" / "realistic_niah_olmo3_extension.json"
        ).read_text(encoding="utf-8")
    )

    assert config["source_stimuli_sha256"] == SOURCE_FORMAL_STIMULI_SHA256
    assert config["expected_stimuli"] == 500
    assert config["expected_requests_total"] == 2_000
    for label, checkpoint in config["checkpoints"].items():
        spec = EXTENSION_MODEL_SPECS[label]
        assert checkpoint["model_id"] == spec.model_id
        assert checkpoint["revision"] == EXTENSION_MODEL_REVISIONS[label]
        assert tuple(checkpoint["prompt_modes"]) == spec.prompt_modes


def test_inference_environment_pins_compatible_olmo3_versions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    requirements = (
        repo_root / "requirements-inference.txt"
    ).read_text(encoding="utf-8")
    smoke_launcher = (
        repo_root / "scripts" / "run_realistic_niah_olmo3_extension_smoke.sh"
    ).read_text(encoding="utf-8")
    formal_launcher = (
        repo_root / "scripts" / "launch_realistic_niah_olmo3_extension.sh"
    ).read_text(encoding="utf-8")

    assert "transformers==5.5.3" in requirements
    assert "vllm==0.25.1" in requirements
    for launcher in (smoke_launcher, formal_launcher):
        assert 'version("transformers") == "5.5.3"' in launcher
        assert 'version("vllm") == "0.25.1"' in launcher
    assert 'export PATH="$(dirname "${python_bin}"):${PATH}"' in smoke_launcher
