from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from realistic_niah.parsing import parse_total, split_reasoning_and_final

from .prompts import TracePromptEncoding
from .spec import V442Config


THINK_CLOSE_BY_FAMILY = {
    "qwen3": "</think>",
    "gemma4": "<channel|>",
}
TOTAL_TEXT_RE = re.compile(r"Total\s*:", flags=re.IGNORECASE)


@dataclass(frozen=True)
class TraceBoundaries:
    trace_start: int
    trace_end: int
    final_start: int
    final_end: int
    answer_query_start: int | None
    answer_query_end: int | None
    close_marker_start: int | None
    close_marker_end: int | None
    boundary_status: str

    @property
    def trace_length(self) -> int:
        return self.trace_end - self.trace_start


def _subsequence_positions(values: Sequence[int], pattern: Sequence[int]) -> list[int]:
    if not pattern or len(pattern) > len(values):
        return []
    width = len(pattern)
    return [
        index
        for index in range(len(values) - width + 1)
        if list(values[index : index + width]) == list(pattern)
    ]


def _marker_ids(tokenizer: Any, marker: str) -> tuple[int, ...]:
    encoded = tokenizer(marker, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return tuple(int(value) for value in values)


def _find_total_token_span(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    start: int,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for text in ("Total:", "Total: ", "\nTotal:", "\n\nTotal:"):
        pattern = _marker_ids(tokenizer, text)
        for position in _subsequence_positions(token_ids[start:], pattern):
            candidates.append((start + position, start + position + len(pattern)))
    if candidates:
        return sorted(candidates, key=lambda item: (item[0], item[1] - item[0]))[-1]
    return None


def locate_trace_boundaries(
    tokenizer: Any,
    continuation_ids: Sequence[int],
    *,
    mode: str,
    model_family: str,
) -> TraceBoundaries:
    ids = tuple(int(value) for value in continuation_ids)
    if mode == "nonthinking":
        total = _find_total_token_span(tokenizer, ids, start=0)
        return TraceBoundaries(
            trace_start=0,
            trace_end=0,
            final_start=0,
            final_end=len(ids),
            answer_query_start=None if total is None else total[0],
            answer_query_end=None if total is None else total[1],
            close_marker_start=None,
            close_marker_end=None,
            boundary_status="nonthinking",
        )
    marker = THINK_CLOSE_BY_FAMILY.get(model_family)
    if marker is None:
        raise ValueError(f"No trace delimiter registered for family {model_family}")
    pattern = _marker_ids(tokenizer, marker)
    occurrences = _subsequence_positions(ids, pattern)
    if not occurrences:
        return TraceBoundaries(
            trace_start=0,
            trace_end=len(ids),
            final_start=len(ids),
            final_end=len(ids),
            answer_query_start=None,
            answer_query_end=None,
            close_marker_start=None,
            close_marker_end=None,
            boundary_status="unterminated_trace",
        )
    marker_start = occurrences[-1]
    marker_end = marker_start + len(pattern)
    total = _find_total_token_span(tokenizer, ids, start=marker_end)
    return TraceBoundaries(
        trace_start=0,
        trace_end=marker_start,
        final_start=marker_end,
        final_end=len(ids),
        answer_query_start=None if total is None else total[0],
        answer_query_end=None if total is None else total[1],
        close_marker_start=marker_start,
        close_marker_end=marker_end,
        boundary_status="ok" if total is not None else "missing_total",
    )


def _eos_ids(model: Any, tokenizer: Any) -> list[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, (tuple, list, set)):
        return [int(item) for item in value]
    return [int(value)]


@torch.inference_mode()
def generate_trace(
    model: Any,
    tokenizer: Any,
    encoding: TracePromptEncoding,
    *,
    model_family: str,
    config: V442Config,
    sampling_seed: int,
) -> dict[str, Any]:
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([encoding.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [encoding.attention_mask], dtype=torch.long, device=device
    )
    decoding = config.decoding(encoding.model_label, encoding.mode)
    # Transformers generation does not expose one stable per-request Generator
    # API across the registered versions. Set and record the framework RNGs
    # immediately before this batch-size-one request instead.
    torch.manual_seed(int(sampling_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sampling_seed))
    eos_ids = _eos_ids(model, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = eos_ids[0]
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        **decoding,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)
    generated = model.generate(**kwargs)
    sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
    continuation = tuple(
        int(value)
        for value in sequences[0, input_ids.shape[1] :].detach().cpu().tolist()
    )
    if not continuation:
        raise RuntimeError("V4.4.2 generation returned an empty continuation")
    raw_text = tokenizer.decode(
        continuation,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    clean_text = tokenizer.decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    boundaries = locate_trace_boundaries(
        tokenizer,
        continuation,
        mode=encoding.mode,
        model_family=model_family,
    )
    parsed_raw_text = "Total:" + raw_text if encoding.mode == "nonthinking" else raw_text
    reasoning_text, final_text = split_reasoning_and_final(
        parsed_raw_text,
        prompt_mode=("native_thinking" if encoding.mode == "native_thinking" else "direct"),
        reasoning_expected=encoding.mode == "native_thinking",
    )
    max_tokens = int(decoding["max_new_tokens"])
    stopped_on_eos = bool(eos_ids and continuation[-1] in set(eos_ids))
    return {
        "schema_version": "realistic_niah_v4_4_2_generation_v1",
        "stimulus_id": encoding.stimulus_id,
        "model_label": encoding.model_label,
        "model_family": model_family,
        "mode": encoding.mode,
        "prompt_variant": encoding.prompt_variant,
        "seed": encoding.seed,
        "split": encoding.split,
        "gold_count": encoding.count,
        "sampling_seed": int(sampling_seed),
        "decoding": decoding,
        "prompt_token_count": encoding.prompt_token_count,
        "generated_token_ids": list(continuation),
        "generated_token_count": len(continuation),
        "completion_text_raw": raw_text,
        "completion_text": clean_text,
        "parsed_response_text_raw": parsed_raw_text,
        "reasoning_text": reasoning_text,
        "final_text": final_text,
        "parsed_count": parse_total(final_text),
        "exact_count": parse_total(final_text) == encoding.count,
        "generation_eos_token_ids": eos_ids,
        "stopped_on_eos": stopped_on_eos,
        "generation_truncated": len(continuation) >= max_tokens and not stopped_on_eos,
        "boundaries": asdict(boundaries),
    }
