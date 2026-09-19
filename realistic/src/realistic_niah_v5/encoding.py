from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from realistic_niah_v4.prompts import TokenSpan

from .parsing import (
    TraceCharSite,
    TraceTokenSite,
    align_trace_sites,
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    output_token_ids,
    prompt_token_ids,
    raw_output_text,
    trace_char_sites,
)


@dataclass(frozen=True)
class PromptRecordSpan:
    slot_index: int
    city: str
    score: int | None
    start: int
    end: int


@dataclass(frozen=True)
class NativeTraceEncoding:
    stimulus_id: str
    request_id: str
    design_variant: str
    seed: int
    split: str
    count: int
    model_label: str
    model_family: str
    answer_format: str
    text: str
    generation_prompt: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    query_position: int
    prompt_token_count: int
    raw_prefix_text: str
    selected_site: dict[str, Any]
    prompt_record_spans: tuple[PromptRecordSpan, ...]
    trace_item_spans: tuple[TokenSpan, ...]
    slot_spans: tuple[TokenSpan, ...]
    needle_spans: tuple[TokenSpan, ...]
    hard_negative_spans: tuple[TokenSpan, ...]
    count_candidate_texts: tuple[tuple[int, str], ...]
    count_candidate_answer_token_ids: tuple[tuple[int, tuple[int, ...]], ...]
    count_candidate_token_ids: tuple[tuple[int, tuple[int, ...]], ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    def candidate_token_ids(self, count: int) -> tuple[int, ...]:
        candidates = dict(self.count_candidate_token_ids)
        if int(count) not in candidates:
            raise KeyError(f"No V5 count candidate for {count}")
        return tuple(int(value) for value in candidates[int(count)])


def _as_token_site(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    site_id: str,
    model_family: str,
) -> tuple[TraceTokenSite, list[TraceTokenSite]]:
    raw = raw_output_text(row)
    parser = find_trace_count_sequence(
        raw,
        model_family=model_family,
        gold_records=gold_records(row),
    )
    char_sites = trace_char_sites(raw, parser)
    token_sites = align_trace_sites(
        tokenizer,
        raw_text=raw,
        baseline_output_token_ids=output_token_ids(row),
        sites=char_sites,
    )
    selected = [site for site in token_sites if site.char_site.site_id == site_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one trace site {site_id!r}, found {len(selected)}")
    if not selected[0].alignment_eligible:
        raise ValueError(
            f"Trace site {site_id!r} is not text-exact: "
            f"{selected[0].alignment_status}"
        )
    return selected[0], token_sites


def _visible_item_spans(
    token_sites: Sequence[TraceTokenSite],
    *,
    prompt_tokens: int,
    prefix: TraceTokenSite,
) -> tuple[TokenSpan, ...]:
    spans: list[TokenSpan] = []
    for site in token_sites:
        if site.char_site.site_kind != "item_end":
            continue
        start = site.literal_token_start
        end = site.literal_token_end
        if start is None or end is None or end <= start:
            continue
        # A retokenized suffix is allowed for the selected boundary, but only
        # literal baseline tokens before the divergence can be semantic spans.
        if end > prefix.shared_baseline_prefix_tokens:
            continue
        occurrence = int(site.char_site.occurrence or 0)
        spans.append(
            TokenSpan(
                slot_index=occurrence,
                start=prompt_tokens + int(start),
                end=prompt_tokens + int(end),
                active=True,
                kind="native_trace_item",
                canonical_length=int(end - start),
                model_token_length=int(end - start),
            )
        )
    return tuple(spans)


def _registered_prompt_record_spans(
    row: Mapping[str, Any], *, prompt_token_count: int
) -> tuple[PromptRecordSpan, ...]:
    values = row.get("prompt_record_spans")
    if values is None:
        return ()
    spans: list[PromptRecordSpan] = []
    seen_cities: set[str] = set()
    for value in values:
        city = str(value["city"])
        city_key = city.casefold()
        if city_key in seen_cities:
            raise ValueError(f"Duplicate prompt record city: {city}")
        seen_cities.add(city_key)
        start = int(value["start"])
        end = int(value["end"])
        if not 0 <= start < end <= int(prompt_token_count):
            raise ValueError(f"Prompt record span is out of bounds: [{start}, {end})")
        score = value.get("score")
        spans.append(
            PromptRecordSpan(
                slot_index=int(value["slot_index"]),
                city=city,
                score=None if score is None else int(score),
                start=start,
                end=end,
            )
        )
    gold_cities = {str(value["city"]).casefold() for value in gold_records(row)}
    if seen_cities != gold_cities:
        raise ValueError("Prompt record spans do not match the oracle city registry")
    return tuple(sorted(spans, key=lambda span: span.slot_index))


def _termination_suffix(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    model_family: str,
) -> str:
    user_text = row.get("user_text")
    if not isinstance(user_text, str) or not user_text:
        # The suffix after an already supplied assistant message is independent
        # of the earlier user content. A fixed probe keeps legacy archives
        # usable while the prefix-token audit below still guards the real row.
        user_text = "V5 legacy-archive termination probe"
    sentinel = "V5_ASSISTANT_TERMINATION_SENTINEL_43b7"
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": True,
    }
    completed = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": sentinel},
        ],
        **kwargs,
    )
    if not isinstance(completed, str) or completed.count(sentinel) != 1:
        raise RuntimeError(
            f"Cannot isolate the {model_family} assistant termination suffix"
        )
    suffix = completed.split(sentinel, maxsplit=1)[1]
    if not suffix:
        raise RuntimeError("Native chat template supplied no assistant termination")
    return suffix


def _candidate_sequences(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    raw_prefix: str,
    prefix_ids: tuple[int, ...],
    candidate_counts: Sequence[int],
    model_family: str,
) -> tuple[
    tuple[tuple[int, str], ...],
    tuple[tuple[int, tuple[int, ...]], ...],
    tuple[tuple[int, tuple[int, ...]], ...],
]:
    termination = _termination_suffix(
        tokenizer, row, model_family=model_family
    )
    text_rows: list[tuple[int, str]] = []
    answer_rows: list[tuple[int, tuple[int, ...]]] = []
    score_rows: list[tuple[int, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for value in candidate_counts:
        text = str(int(value))
        answer_full = tuple(
            int(token)
            for token in tokenizer.encode(
                raw_prefix + text, add_special_tokens=False
            )
        )
        score_full = tuple(
            int(token)
            for token in tokenizer.encode(
                raw_prefix + text + termination, add_special_tokens=False
            )
        )
        if answer_full[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(
                f"Count {value} retokenized the native answer-query prefix"
            )
        if score_full[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(
                f"Completed count {value} retokenized the native answer-query prefix"
            )
        answer = answer_full[len(prefix_ids) :]
        scored = score_full[len(prefix_ids) :]
        if not answer or len(scored) <= len(answer) or scored[: len(answer)] != answer:
            raise RuntimeError(
                f"Count {value} lacks a stable answer plus termination sequence"
            )
        if scored in seen:
            raise RuntimeError("Two V5 counts share one completed token sequence")
        seen.add(scored)
        text_rows.append((int(value), text))
        answer_rows.append((int(value), answer))
        score_rows.append((int(value), scored))
    return tuple(text_rows), tuple(answer_rows), tuple(score_rows)


def _unscored_candidate_sequences(
    tokenizer: Any,
    *,
    raw_prefix: str,
    prefix_ids: tuple[int, ...],
    candidate_counts: Sequence[int],
) -> tuple[
    tuple[tuple[int, str], ...],
    tuple[tuple[int, tuple[int, ...]], ...],
    tuple[tuple[int, tuple[int, ...]], ...],
]:
    """Supply generation-only candidates at non-answer sites.

    Full answer+termination scoring is intentionally unavailable here; OV
    candidate scoring is registered only at ``answer_query``.
    """

    text_rows = []
    answer_rows = []
    for value in candidate_counts:
        text = str(int(value))
        full = tuple(
            int(token)
            for token in tokenizer.encode(raw_prefix + text, add_special_tokens=False)
        )
        if full[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(
                f"Count {value} retokenized the selected native trace prefix"
            )
        answer = full[len(prefix_ids) :]
        if not answer:
            raise RuntimeError(f"Count {value} produced no continuation tokens")
        text_rows.append((int(value), text))
        answer_rows.append((int(value), answer))
    rows = tuple(answer_rows)
    return tuple(text_rows), rows, rows


def build_native_trace_encoding(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    site_id: str,
    candidate_counts: Sequence[int] = tuple(range(1, 11)),
    model_family: str | None = None,
) -> NativeTraceEncoding:
    family = infer_model_family(row, model_family)
    selected, token_sites = _as_token_site(
        tokenizer,
        row,
        site_id=site_id,
        model_family=family,
    )
    prompt_ids = prompt_token_ids(row)
    prompt_mask_value = row.get("attention_mask")
    if prompt_mask_value is None:
        prompt_mask = (1,) * len(prompt_ids)
    else:
        prompt_mask = tuple(int(value) for value in prompt_mask_value)
        if len(prompt_mask) != len(prompt_ids):
            raise ValueError("Prompt attention mask length does not match input IDs")
    prefix_ids = tuple(int(value) for value in selected.prefix_token_ids)
    if not prefix_ids:
        raise RuntimeError("A selected V5 site must contain at least one output token")
    full_ids = prompt_ids + prefix_ids
    spans = _visible_item_spans(
        token_sites,
        prompt_tokens=len(prompt_ids),
        prefix=selected,
    )
    prompt_record_spans = _registered_prompt_record_spans(
        row, prompt_token_count=len(prompt_ids)
    )
    raw_prefix = raw_output_text(row)[: selected.char_site.char_end]
    answer_query_kinds = {"answer_query", "answer_query_v2", "answer_query_v3"}
    candidate_builder = (
        _candidate_sequences
        if selected.char_site.site_kind in answer_query_kinds
        else _unscored_candidate_sequences
    )
    candidate_kwargs: dict[str, Any] = {
        "raw_prefix": raw_prefix,
        "prefix_ids": prefix_ids,
        "candidate_counts": candidate_counts,
    }
    if selected.char_site.site_kind in answer_query_kinds:
        candidate_kwargs.update(
            {"row": row, "model_family": family}
        )
    candidate_texts, candidate_answer_ids, candidate_score_ids = (
        candidate_builder(
            tokenizer,
            **candidate_kwargs,
        )
    )
    prompt_text = str(row.get("rendered_prompt", row.get("generation_prompt", "")))
    stimulus_id = str(row.get("stimulus_id", row.get("request_id", "unknown")))
    request_id = str(row.get("request_id", stimulus_id))
    model_label = str(row.get("model_label", row.get("model", family)))
    count = len(gold_records(row))
    seed = int(row.get("seed", -1))
    split = str(row.get("split") or "unregistered")
    return NativeTraceEncoding(
        stimulus_id=stimulus_id,
        request_id=request_id,
        design_variant=str(row.get("design_variant", "v4.4")),
        seed=seed,
        split=split,
        count=count,
        model_label=model_label,
        model_family=family,
        answer_format="numeric",
        text=prompt_text + raw_prefix,
        generation_prompt=prompt_text + raw_prefix,
        input_ids=full_ids,
        attention_mask=prompt_mask + (1,) * len(prefix_ids),
        query_position=len(full_ids) - 1,
        prompt_token_count=len(prompt_ids),
        raw_prefix_text=raw_prefix,
        selected_site=selected.to_dict(),
        prompt_record_spans=prompt_record_spans,
        trace_item_spans=spans,
        slot_spans=spans,
        needle_spans=spans,
        hard_negative_spans=(),
        count_candidate_texts=candidate_texts,
        count_candidate_answer_token_ids=candidate_answer_ids,
        count_candidate_token_ids=candidate_score_ids,
    )


def build_native_causal_encoding(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    query_output_token_index: int,
    sequence_output_token_end: int,
    selected_site: Mapping[str, Any],
    model_family: str | None = None,
) -> NativeTraceEncoding:
    """Build an exact teacher-forced prefix for a registry-defined anchor.

    Unlike :func:`build_native_trace_encoding`, this constructor never asks the
    legacy site parser to invent a site ID.  The grammar-aware causal compiler
    supplies an output-token query and a fixed target endpoint, while the
    stored output token IDs remain the sole model input.
    """

    from .causal_sites import build_output_token_map

    family = infer_model_family(row, model_family)
    prompt_ids = prompt_token_ids(row)
    output_ids = output_token_ids(row)
    query = int(query_output_token_index)
    output_end = int(sequence_output_token_end)
    if not 0 <= query < output_end <= len(output_ids):
        raise ValueError(
            "Causal query/target bounds must satisfy "
            "0 <= query < target_end <= output length"
        )
    prompt_mask_value = row.get("attention_mask")
    if prompt_mask_value is None:
        prompt_mask = (1,) * len(prompt_ids)
    else:
        prompt_mask = tuple(int(value) for value in prompt_mask_value)
        if len(prompt_mask) != len(prompt_ids):
            raise ValueError("Prompt attention mask length does not match input IDs")
    output_prefix = tuple(int(value) for value in output_ids[:output_end])
    full_ids = prompt_ids + output_prefix
    token_map = build_output_token_map(row, tokenizer)
    raw_end = int(token_map.offsets[output_end - 1][1])
    raw_prefix = raw_output_text(row)[:raw_end]
    prompt_text = str(row.get("rendered_prompt", row.get("generation_prompt", "")))
    stimulus_id = str(row.get("stimulus_id", row.get("request_id", "unknown")))
    request_id = str(row.get("request_id", stimulus_id))
    model_label = str(row.get("model_label", row.get("model", family)))
    return NativeTraceEncoding(
        stimulus_id=stimulus_id,
        request_id=request_id,
        design_variant=str(row.get("design_variant", "v4.4")),
        seed=int(row.get("seed", -1)),
        split=str(row.get("split") or "unregistered"),
        count=len(gold_records(row)),
        model_label=model_label,
        model_family=family,
        answer_format="numeric",
        text=prompt_text + raw_prefix,
        generation_prompt=prompt_text + raw_prefix,
        input_ids=full_ids,
        attention_mask=prompt_mask + (1,) * len(output_prefix),
        query_position=len(prompt_ids) + query,
        prompt_token_count=len(prompt_ids),
        raw_prefix_text=raw_prefix,
        selected_site=dict(selected_site),
        prompt_record_spans=_registered_prompt_record_spans(
            row, prompt_token_count=len(prompt_ids)
        ),
        trace_item_spans=(),
        slot_spans=(),
        needle_spans=(),
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )


def item_site_ids(row: Mapping[str, Any], *, model_family: str | None = None) -> list[str]:
    family = infer_model_family(row, model_family)
    raw = raw_output_text(row)
    parser = find_trace_count_sequence(
        raw,
        model_family=family,
        gold_records=gold_records(row),
    )
    return [
        site.site_id
        for site in trace_char_sites(raw, parser)
        if site.site_kind == "item_end"
    ]
