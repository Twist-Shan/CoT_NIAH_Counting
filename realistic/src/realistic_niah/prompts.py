from __future__ import annotations

from copy import deepcopy
from typing import Any

from .spec import (
    ALL_MODEL_SPECS,
    FORMAL_PROMPT_MODES,
    QUERY_LAYOUT,
    THINKING_PROMPT_MODES,
    ModelSpec,
)

COMMON_COUNTING_CUE = """\
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score."""

DIRECT_QUERY_BLOCK = """\
How many city-score audit records are in the passage?
Do not explain, reason aloud, quote, or list any records.
Your entire response must be exactly one line:
Total: <integer>"""

ENUMERATION_INDEX_QUERY_BLOCK = """\
How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin the first item with "1. ", the second with "2. ", and continue with ordinary digits.
After each number, write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text."""

ENUMERATION_BULLET_QUERY_BLOCK = """\
How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin each item with "-", then write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text."""

NATIVE_THINKING_QUERY_BLOCK = """\
How many city-score audit records are in the passage?
Reason concisely without repeating or restarting.
Stop as soon as you determine the count, then output exactly one line:
Total: <integer>"""

SUPPORTED_PROMPT_MODES = frozenset(FORMAL_PROMPT_MODES)

MINISTRAL3_REASONING_SYSTEM_PROMPT = """\
# HOW YOU SHOULD THINK AND ANSWER

First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response."""


def query_block(prompt_mode: str) -> str:
    if prompt_mode == "direct":
        return DIRECT_QUERY_BLOCK
    if prompt_mode == "enumeration_index":
        return ENUMERATION_INDEX_QUERY_BLOCK
    if prompt_mode == "enumeration_bullet":
        return ENUMERATION_BULLET_QUERY_BLOCK
    if prompt_mode == "native_thinking":
        return NATIVE_THINKING_QUERY_BLOCK
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def build_messages(
    passage: str,
    *,
    prompt_mode: str,
    query_layout: str = QUERY_LAYOUT,
) -> list[dict[str, str]]:
    if query_layout != QUERY_LAYOUT:
        raise ValueError(f"Unsupported query_layout: {query_layout}")
    task = query_block(prompt_mode)
    passage_block = f"<passage>\n{passage}\n</passage>"
    content = f"{COMMON_COUNTING_CUE}\n\n{passage_block}\n\n{task}"
    return [{"role": "user", "content": content}]


def thinking_enabled(prompt_mode: str) -> bool:
    return prompt_mode in THINKING_PROMPT_MODES


def reasoning_expected(
    model_spec: ModelSpec,
    prompt_mode: str,
) -> bool:
    if model_spec.reasoning_policy == "always_on":
        return True
    if model_spec.reasoning_policy == "switchable":
        return thinking_enabled(prompt_mode)
    if model_spec.reasoning_policy == "off_only":
        return False
    raise ValueError(
        f"Unsupported reasoning policy: {model_spec.reasoning_policy!r}"
    )


def resolve_model_spec(model: str) -> ModelSpec:
    if model in ALL_MODEL_SPECS:
        return ALL_MODEL_SPECS[model]
    for spec in ALL_MODEL_SPECS.values():
        if spec.model_id == model:
            return spec
    raise ValueError(f"Unknown registered model: {model}")


def render_generation_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    model_spec: ModelSpec,
    prompt_mode: str,
) -> str:
    if prompt_mode not in model_spec.prompt_modes:
        raise ValueError(
            f"{model_spec.label} does not support prompt mode {prompt_mode!r}"
        )
    controlled_messages = deepcopy(messages)
    if model_spec.system_prompt_strategy == "nemotron_reasoning_signal":
        signal = "/think" if thinking_enabled(prompt_mode) else "/no_think"
        controlled_messages.insert(
            0,
            {"role": "system", "content": signal},
        )
    elif model_spec.system_prompt_strategy == "ministral3_reasoning":
        before, remainder = MINISTRAL3_REASONING_SYSTEM_PROMPT.split(
            "[THINK]",
            1,
        )
        thinking, after = remainder.split("[/THINK]", 1)
        controlled_messages.insert(
            0,
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": before},
                    {
                        "type": "thinking",
                        "thinking": thinking,
                        "closed": True,
                    },
                    {"type": "text", "text": after},
                ],
            },
        )
    elif model_spec.system_prompt_strategy != "none":
        raise ValueError(
            f"Unsupported system-prompt strategy: "
            f"{model_spec.system_prompt_strategy!r}"
        )

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_spec.reasoning_policy == "switchable":
        enabled = thinking_enabled(prompt_mode)
        if model_spec.chat_template_control == "enable_thinking_kwarg":
            kwargs["enable_thinking"] = enabled
        elif model_spec.chat_template_control == "thinking_kwarg":
            kwargs["thinking"] = enabled
        elif model_spec.chat_template_control != "system_reasoning_signal":
            raise ValueError(
                f"Unsupported switchable chat-template control: "
                f"{model_spec.chat_template_control!r}"
            )
    elif model_spec.reasoning_policy not in {"always_on", "off_only"}:
        raise ValueError(
            f"Unsupported reasoning policy: {model_spec.reasoning_policy!r}"
        )
    return tokenizer.apply_chat_template(controlled_messages, **kwargs)
