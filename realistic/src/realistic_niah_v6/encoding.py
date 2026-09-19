from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

from realistic_niah_v5 import encoding as _v5_encoding


StructuredTraceEncoding = _v5_encoding.NativeTraceEncoding
PromptRecordSpan = _v5_encoding.PromptRecordSpan


def _enumeration_termination_suffix(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    model_family: str,
) -> str:
    """Isolate the assistant suffix with thinking disabled.

    V5 deliberately renders this suffix inside a native-thinking container.
    V6 enumeration prompts use the same model/tokenizer revisions but disable
    thinking, so candidate answer scores must use the corresponding container.
    """

    user_text = row.get("user_text")
    if not isinstance(user_text, str) or not user_text:
        user_text = "V6 structured-enumeration termination probe"
    sentinel = "V6_ASSISTANT_TERMINATION_SENTINEL_a92f"
    completed = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": sentinel},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(completed, str) or completed.count(sentinel) != 1:
        raise RuntimeError(
            f"Cannot isolate the {model_family} V6 assistant termination suffix"
        )
    suffix = completed.split(sentinel, maxsplit=1)[1]
    if not suffix:
        raise RuntimeError("V6 chat template supplied no assistant termination")
    return suffix


@contextmanager
def _patched_termination() -> Iterator[None]:
    original = _v5_encoding._termination_suffix
    _v5_encoding._termination_suffix = _enumeration_termination_suffix
    try:
        yield
    finally:
        _v5_encoding._termination_suffix = original


def build_structured_trace_encoding(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    site_id: str,
    candidate_counts: Sequence[int] = tuple(range(1, 11)),
    model_family: str | None = None,
) -> StructuredTraceEncoding:
    with _patched_termination():
        return _v5_encoding.build_native_trace_encoding(
            row,
            tokenizer,
            site_id=site_id,
            candidate_counts=candidate_counts,
            model_family=model_family,
        )


def build_structured_causal_encoding(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    query_output_token_index: int,
    sequence_output_token_end: int,
    selected_site: Mapping[str, Any],
    model_family: str | None = None,
) -> StructuredTraceEncoding:
    return _v5_encoding.build_native_causal_encoding(
        row,
        tokenizer,
        query_output_token_index=query_output_token_index,
        sequence_output_token_end=sequence_output_token_end,
        selected_site=selected_site,
        model_family=model_family,
    )


# V6-native names used by adapters.  They intentionally do not replace the
# symbols in the frozen V5 source tree.
build_trace_encoding = build_structured_trace_encoding
build_causal_encoding = build_structured_causal_encoding


__all__ = [
    "PromptRecordSpan",
    "StructuredTraceEncoding",
    "build_causal_encoding",
    "build_structured_causal_encoding",
    "build_structured_trace_encoding",
    "build_trace_encoding",
]
