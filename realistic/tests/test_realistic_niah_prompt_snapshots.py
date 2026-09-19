from __future__ import annotations

from realistic_niah.prompts import (
    build_messages,
    reasoning_expected,
    render_generation_prompt,
)
from realistic_niah.spec import MODEL_SPECS

COMMON = """\
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
PASSAGE
</passage>

"""

EXPECTED_SUFFIXES = {
    "direct": """\
How many city-score audit records are in the passage?
Do not explain, reason aloud, quote, or list any records.
Your entire response must be exactly one line:
Total: <integer>""",
    "enumeration_index": """\
How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin the first item with "1. ", the second with "2. ", and continue with ordinary digits.
After each number, write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text.""",
    "enumeration_bullet": """\
How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin each item with "-", then write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text.""",
    "native_thinking": """\
How many city-score audit records are in the passage?
Reason concisely without repeating or restarting.
Stop as soon as you determine the count, then output exactly one line:
Total: <integer>""",
}


def test_all_four_formal_prompts_match_registered_snapshots() -> None:
    for mode, suffix in EXPECTED_SUFFIXES.items():
        messages = build_messages("PASSAGE", prompt_mode=mode)

        assert messages == [
            {
                "role": "user",
                "content": COMMON + suffix,
            }
        ]


def test_switchable_models_enable_thinking_only_for_native_mode() -> None:
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

    tokenizer = RecordingTokenizer()
    model = MODEL_SPECS["Qwen3-4B"]
    messages = [{"role": "user", "content": "test"}]

    for mode in EXPECTED_SUFFIXES:
        render_generation_prompt(
            tokenizer,
            messages,
            model_spec=model,
            prompt_mode=mode,
        )

    assert [call["enable_thinking"] for call in tokenizer.calls] == [
        False,
        False,
        False,
        True,
    ]


def test_always_on_reasoning_models_use_their_native_templates_unchanged() -> None:
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
    for label in (
        "DeepSeek-R1-0528-Qwen3-8B",
        "GLM-Z1-9B-0414",
    ):
        tokenizer = RecordingTokenizer()
        model = MODEL_SPECS[label]
        assert model.prompt_modes == ("native_thinking",)
        for mode in model.prompt_modes:
            render_generation_prompt(
                tokenizer,
                messages,
                model_spec=model,
                prompt_mode=mode,
            )
            assert reasoning_expected(model, mode) is True

        assert all(
            "enable_thinking" not in call
            for call in tokenizer.calls
        )


def test_glm4_matched_control_uses_off_only_native_template() -> None:
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

    tokenizer = RecordingTokenizer()
    model = MODEL_SPECS["GLM-4-9B-0414"]
    messages = [{"role": "user", "content": "test"}]

    render_generation_prompt(
        tokenizer,
        messages,
        model_spec=model,
        prompt_mode="direct",
    )

    assert reasoning_expected(model, "direct") is False
    assert "enable_thinking" not in tokenizer.calls[0]
