from __future__ import annotations

from typing import Any


def apply_generation_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    thinking_mode: bool,
) -> str:
    """Render generation prompt while honoring model-specific thinking controls.

    Qwen3 tokenizers default to thinking mode unless ``enable_thinking`` is passed
    explicitly. Passing the flag through ``apply_chat_template`` keeps rendered
    prompts aligned with the experiment config.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking_mode,
    )
