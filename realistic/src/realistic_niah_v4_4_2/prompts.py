from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from realistic_niah.prompts import COMMON_COUNTING_CUE
from realistic_niah_v4.prompts import (
    TokenSpan,
    V4_NUMERIC_QUERY_BLOCK,
    _encode_text_with_offsets,
    _passage_char_start,
    _span_for_char_interval,
)
from realistic_niah_v4.spec import V4ModelSpec

from .spec import MODES, PROMPT_VARIANTS


@dataclass(frozen=True)
class TracePromptEncoding:
    stimulus_id: str
    design_variant: str
    seed: int
    split: str
    count: int
    model_label: str
    mode: str
    prompt_variant: str
    user_text: str
    model_text: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    prompt_token_count: int
    slot_spans: tuple[TokenSpan, ...]
    needle_spans: tuple[TokenSpan, ...]
    cue_span: tuple[int, int] | None
    passage_span: tuple[int, int]
    question_span: tuple[int, int]
    assistant_prefix_span: tuple[int, int] | None


def build_user_text(passage: str, *, prompt_variant: str) -> str:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown V4.4.2 prompt variant: {prompt_variant}")
    passage_block = f"<passage>\n{passage}\n</passage>"
    components = [passage_block, V4_NUMERIC_QUERY_BLOCK]
    if prompt_variant == "cue_present":
        components.insert(0, COMMON_COUNTING_CUE)
    return "\n\n".join(components)


def _chat_template_kwargs(
    model_spec: V4ModelSpec,
    *,
    mode: str,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unknown V4.4.2 mode: {mode}")
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    enabled = mode == "native_thinking"
    if model_spec.chat_template_control == "enable_thinking_kwarg":
        kwargs["enable_thinking"] = enabled
    elif model_spec.chat_template_control == "thinking_kwarg":
        kwargs["thinking"] = enabled
    else:
        raise ValueError(
            f"Unsupported V4.4.2 chat template control: "
            f"{model_spec.chat_template_control}"
        )
    return kwargs


def render_trace_prompt(
    stimulus: dict[str, Any],
    *,
    tokenizer: Any,
    model_spec: V4ModelSpec,
    mode: str,
    prompt_variant: str,
) -> TracePromptEncoding:
    if str(stimulus["design_variant"]) != "v4.4":
        raise ValueError("V4.4.2 accepts only frozen v4.4 stimuli")
    passage = str(stimulus["passage"])
    user_text = build_user_text(passage, prompt_variant=prompt_variant)
    model_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        **_chat_template_kwargs(model_spec, mode=mode),
    )
    if not isinstance(model_text, str):
        raise RuntimeError("Chat template did not return rendered text")
    assistant_prefix_start_char = None
    if mode == "nonthinking":
        # Preserve the completed V4.4 protocol exactly: the assistant-side
        # ``Total:`` prefix is teacher supplied and generation produces only
        # the decimal continuation. Native thinking must start inside its
        # registered thought channel and therefore cannot use this prefill.
        assistant_prefix_start_char = len(model_text)
        model_text += "Total:"
    input_ids, attention_mask, offsets = _encode_text_with_offsets(
        tokenizer, model_text
    )
    if not input_ids:
        raise RuntimeError("Rendered V4.4.2 prompt is empty")

    passage_start_char = _passage_char_start(model_text, passage)
    passage_token_span = _span_for_char_interval(
        offsets, passage_start_char, passage_start_char + len(passage)
    )
    question_start_char = model_text.find(V4_NUMERIC_QUERY_BLOCK)
    if question_start_char < 0:
        raise RuntimeError("Rendered prompt does not contain the registered query")
    question_span = _span_for_char_interval(
        offsets,
        question_start_char,
        question_start_char + len(V4_NUMERIC_QUERY_BLOCK),
    )
    cue_span = None
    if prompt_variant == "cue_present":
        cue_start_char = model_text.find(COMMON_COUNTING_CUE)
        if cue_start_char < 0:
            raise RuntimeError("Rendered prompt lost the registered opening cue")
        cue_span = _span_for_char_interval(
            offsets,
            cue_start_char,
            cue_start_char + len(COMMON_COUNTING_CUE),
        )
    assistant_prefix_span = None
    if assistant_prefix_start_char is not None:
        assistant_prefix_span = _span_for_char_interval(
            offsets, assistant_prefix_start_char, len(model_text)
        )
        if assistant_prefix_span[1] != len(input_ids):
            raise RuntimeError("Non-thinking Total: prefix is not prompt-final")

    slot_spans: list[TokenSpan] = []
    for slot in stimulus["slots"]:
        start, end = _span_for_char_interval(
            offsets,
            passage_start_char + int(slot["char_start"]),
            passage_start_char + int(slot["char_end"]),
        )
        slot_spans.append(
            TokenSpan(
                slot_index=int(slot["slot_index"]),
                start=start,
                end=end,
                active=bool(slot["active"]),
                kind=str(slot["content_kind"]),
                canonical_length=int(slot["canonical_token_length"]),
                model_token_length=end - start,
            )
        )
    if [span.active for span in slot_spans] != [
        index < int(stimulus["gold_count"]) for index in range(len(slot_spans))
    ]:
        raise RuntimeError("V4.4.2 prompt violates the frozen nested-slot contract")
    return TracePromptEncoding(
        stimulus_id=str(stimulus["stimulus_id"]),
        design_variant="v4.4",
        seed=int(stimulus["seed"]),
        split=str(stimulus["split"]),
        count=int(stimulus["gold_count"]),
        model_label=model_spec.label,
        mode=mode,
        prompt_variant=prompt_variant,
        user_text=user_text,
        model_text=model_text,
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        prompt_token_count=len(input_ids),
        slot_spans=tuple(slot_spans),
        needle_spans=tuple(span for span in slot_spans if span.active),
        cue_span=cue_span,
        passage_span=passage_token_span,
        question_span=question_span,
        assistant_prefix_span=assistant_prefix_span,
    )
