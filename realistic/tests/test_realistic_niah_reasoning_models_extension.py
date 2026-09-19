from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from realistic_niah.parsing import split_reasoning_and_final
from realistic_niah.prompts import (
    MINISTRAL3_REASONING_SYSTEM_PROMPT,
    build_messages,
    render_generation_prompt,
)
from realistic_niah.reasoning_models_extension import (
    EXPECTED_EXTENSION_REQUESTS,
    EXPECTED_EXTENSION_SHARDS,
    LOGICAL_GROUPS,
    reasoning_extension_plan,
    reasoning_extension_tasks,
)
from realistic_niah.runner import decoding_config, model_engine_overrides
from realistic_niah.spec import (
    ALL_MODEL_SPECS,
    FORMAL_PROMPT_MODES,
    NONTHINKING_PROMPT_MODES,
    REASONING_EXTENSION_MODEL_REVISIONS,
    REASONING_EXTENSION_MODEL_SPECS,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        self.calls.append((messages, kwargs))
        return "rendered"


def _render(model_label: str, prompt_mode: str) -> RecordingTokenizer:
    tokenizer = RecordingTokenizer()
    render_generation_prompt(
        tokenizer,
        build_messages("passage", prompt_mode=prompt_mode),
        model_spec=REASONING_EXTENSION_MODEL_SPECS[model_label],
        prompt_mode=prompt_mode,
    )
    return tokenizer


def test_reasoning_extension_has_six_pinned_checkpoints() -> None:
    assert set(REASONING_EXTENSION_MODEL_SPECS) == {
        "Nemotron-Nano-v2-9B",
        "Nemotron-3-Nano-4B",
        "Granite-3.3-Instruct-8B",
        "Cogito-v1-Preview-8B",
        "Ministral-3-Instruct-8B",
        "Ministral-3-Reasoning-8B",
    }
    assert set(REASONING_EXTENSION_MODEL_REVISIONS) == set(
        REASONING_EXTENSION_MODEL_SPECS
    )
    assert all(
        len(revision) == 40
        for revision in REASONING_EXTENSION_MODEL_REVISIONS.values()
    )
    assert set(REASONING_EXTENSION_MODEL_SPECS).issubset(ALL_MODEL_SPECS)


def test_mode_split_matches_same_weight_and_ministral_pair_design() -> None:
    for label in (
        "Nemotron-Nano-v2-9B",
        "Nemotron-3-Nano-4B",
        "Granite-3.3-Instruct-8B",
        "Cogito-v1-Preview-8B",
    ):
        assert (
            REASONING_EXTENSION_MODEL_SPECS[label].prompt_modes
            == FORMAL_PROMPT_MODES
        )
    assert (
        REASONING_EXTENSION_MODEL_SPECS[
            "Ministral-3-Instruct-8B"
        ].prompt_modes
        == NONTHINKING_PROMPT_MODES
    )
    assert REASONING_EXTENSION_MODEL_SPECS[
        "Ministral-3-Reasoning-8B"
    ].prompt_modes == ("native_thinking",)
    assert LOGICAL_GROUPS["Ministral-3-8B"]["comparison_type"] == (
        "separate_instruct_reasoning_checkpoints"
    )


def test_reasoning_extension_plan_has_20_shards_and_10000_requests() -> None:
    tasks = reasoning_extension_tasks()
    plan = reasoning_extension_plan()

    assert len(tasks) == EXPECTED_EXTENSION_SHARDS == 20
    assert EXPECTED_EXTENSION_REQUESTS == 10_000
    assert sum(task["expected_requests"] for task in tasks) == 10_000
    assert len({task["task_id"] for task in tasks}) == 20
    assert plan["expected_requests"] == 10_000
    assert len(plan["tasks_sha256"]) == 64


def test_model_card_specific_chat_template_controls() -> None:
    off = _render("Nemotron-Nano-v2-9B", "direct")
    on = _render("Nemotron-Nano-v2-9B", "native_thinking")
    assert off.calls[0][0][0] == {"role": "system", "content": "/no_think"}
    assert on.calls[0][0][0] == {"role": "system", "content": "/think"}
    assert "enable_thinking" not in off.calls[0][1]

    nemotron3 = _render("Nemotron-3-Nano-4B", "native_thinking")
    assert nemotron3.calls[0][1]["enable_thinking"] is True

    granite = _render("Granite-3.3-Instruct-8B", "native_thinking")
    assert granite.calls[0][1]["thinking"] is True
    assert "enable_thinking" not in granite.calls[0][1]

    cogito = _render("Cogito-v1-Preview-8B", "direct")
    assert cogito.calls[0][1]["enable_thinking"] is False


def test_ministral_reasoning_uses_official_structured_system_prompt() -> None:
    tokenizer = _render("Ministral-3-Reasoning-8B", "native_thinking")
    messages, kwargs = tokenizer.calls[0]

    assert kwargs == {"tokenize": False, "add_generation_prompt": True}
    assert messages[0]["role"] == "system"
    content = messages[0]["content"]
    assert [part["type"] for part in content] == [
        "text",
        "thinking",
        "text",
    ]
    reconstructed = (
        content[0]["text"]
        + "[THINK]"
        + content[1]["thinking"]
        + "[/THINK]"
        + content[2]["text"]
    )
    assert reconstructed == MINISTRAL3_REASONING_SYSTEM_PROMPT


def test_registered_decoding_follows_checkpoint_recommendations() -> None:
    specs = REASONING_EXTENSION_MODEL_SPECS

    nemotron9 = decoding_config(
        specs["Nemotron-Nano-v2-9B"],
        "native_thinking",
    )
    assert (nemotron9.temperature, nemotron9.top_p) == (0.6, 0.95)

    nemotron3 = decoding_config(
        specs["Nemotron-3-Nano-4B"],
        "native_thinking",
    )
    assert (nemotron3.temperature, nemotron3.top_p) == (1.0, 0.95)

    granite = decoding_config(
        specs["Granite-3.3-Instruct-8B"],
        "native_thinking",
    )
    assert granite.temperature == 0.0

    ministral = decoding_config(
        specs["Ministral-3-Reasoning-8B"],
        "native_thinking",
    )
    assert (ministral.temperature, ministral.top_p) == (0.7, 0.95)

    with pytest.raises(ValueError):
        decoding_config(specs["Ministral-3-Instruct-8B"], "native_thinking")


def test_engine_profiles_include_mamba_and_mistral_requirements() -> None:
    specs = REASONING_EXTENSION_MODEL_SPECS

    assert model_engine_overrides(specs["Nemotron-Nano-v2-9B"]) == {
        "mamba_ssm_cache_dtype": "float32"
    }
    assert model_engine_overrides(specs["Nemotron-3-Nano-4B"]) == {
        "mamba_ssm_cache_dtype": "float32"
    }
    assert model_engine_overrides(specs["Ministral-3-Instruct-8B"]) == {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
    }
    assert model_engine_overrides(specs["Cogito-v1-Preview-8B"]) == {}


@pytest.mark.parametrize(
    ("raw", "expected_reasoning"),
    (
        (
            "<think>Counted two.</think><response>Total: 2</response>",
            "Counted two.",
        ),
        ("[THINK]Counted two.[/THINK]Total: 2", "Counted two."),
        (
            "<SPECIAL_34>Counted two.<SPECIAL_35>Total: 2",
            "Counted two.",
        ),
    ),
)
def test_new_reasoning_delimiters_preserve_parseable_final_answer(
    raw: str,
    expected_reasoning: str,
) -> None:
    reasoning, final = split_reasoning_and_final(
        raw,
        prompt_mode="native_thinking",
        reasoning_expected=True,
    )

    assert reasoning == expected_reasoning
    assert final == "Total: 2"


def test_json_config_matches_registered_models() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            repo_root
            / "configs"
            / "realistic_niah_reasoning_models_extension.json"
        ).read_text(encoding="utf-8")
    )

    assert config["expected_shards"] == 20
    assert config["expected_requests_total"] == 10_000
    assert set(config["checkpoints"]) == set(
        REASONING_EXTENSION_MODEL_SPECS
    )
    for label, spec in REASONING_EXTENSION_MODEL_SPECS.items():
        checkpoint = config["checkpoints"][label]
        assert checkpoint["model_id"] == spec.model_id
        assert checkpoint["revision"] == (
            REASONING_EXTENSION_MODEL_REVISIONS[label]
        )
        assert tuple(checkpoint["prompt_modes"]) == spec.prompt_modes
        assert checkpoint["reasoning_policy"] == spec.reasoning_policy
        assert checkpoint["engine_profile"] == spec.engine_profile
