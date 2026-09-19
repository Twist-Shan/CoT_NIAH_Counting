from __future__ import annotations

from dataclasses import dataclass

from realistic_niah_v5.natural_aligned_progress import (
    align_natural_donor_prompt,
    matched_post_item_site_candidates,
    matched_post_item_sites,
    post_item_sites_at_tail_offset,
    resolve_natural_patch_span,
)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int


@dataclass(frozen=True)
class _Encoding:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    query_position: int
    prompt_token_count: int
    prompt_record_spans: tuple[_Span, ...]
    trace_item_spans: tuple[_Span, ...]
    slot_spans: tuple[_Span, ...] = ()
    needle_spans: tuple[_Span, ...] = ()
    hard_negative_spans: tuple[_Span, ...] = ()


@dataclass(frozen=True)
class _Registry:
    prompt_token_count: int
    prompt_records: tuple[tuple[int, int], ...]


class _Tokenizer:
    all_special_ids = (99,)

    def decode(self, ids, **_kwargs):
        return {7: "\n", 5: "."}.get(int(ids[0]), "x")


def test_natural_alignment_matches_site_without_deleting_records() -> None:
    prompt = (1,) * 220
    trace = (2, 5, 7, 3, 5, 7, 4, 5, 7, 8)
    encoding = _Encoding(
        input_ids=prompt + trace,
        attention_mask=(1,) * (len(prompt) + len(trace)),
        query_position=len(prompt) + len(trace) - 1,
        prompt_token_count=len(prompt),
        prompt_record_spans=(_Span(100, 105),),
        trace_item_spans=(
            _Span(220, 222),
            _Span(223, 225),
            _Span(226, 228),
        ),
    )
    items = ((220, 222), (223, 225), (226, 228))
    receiver_site, donor_site, _audit = matched_post_item_sites(
        encoding,
        items,
        receiver_occurrence=1,
        donor_occurrence=2,
        tokenizer=_Tokenizer(),
    )
    assert encoding.input_ids[receiver_site] == encoding.input_ids[donor_site]
    donor, audit = align_natural_donor_prompt(
        encoding,
        _Registry(len(prompt), ((100, 105),)),
        receiver_site=receiver_site,
        donor_site=donor_site,
        tokenizer=_Tokenizer(),
    )
    assert donor.input_ids[receiver_site] == encoding.input_ids[donor_site]
    assert audit["deleted_token_count"] == donor_site - receiver_site
    assert donor.prompt_record_spans[0] == _Span(97, 102)


def test_period_preferred_policy_moves_before_trailing_whitespace() -> None:
    prompt = (1,) * 220
    trace = (2, 5, 7, 3, 5, 7, 4, 5, 7, 8)
    encoding = _Encoding(
        input_ids=prompt + trace,
        attention_mask=(1,) * (len(prompt) + len(trace)),
        query_position=len(prompt) + len(trace) - 1,
        prompt_token_count=len(prompt),
        prompt_record_spans=(),
        trace_item_spans=(
            _Span(220, 223),
            _Span(223, 226),
            _Span(226, 229),
        ),
    )
    items = ((220, 223), (223, 226), (226, 229))
    candidates = matched_post_item_site_candidates(
        encoding,
        items,
        receiver_occurrence=1,
        donor_occurrence=2,
        tokenizer=_Tokenizer(),
        tail_window=12,
    )
    assert {row["shared_commit_token_text"] for row in candidates} >= {".", "\n"}

    latest_receiver, latest_donor, latest_audit = matched_post_item_sites(
        encoding,
        items,
        receiver_occurrence=1,
        donor_occurrence=2,
        tokenizer=_Tokenizer(),
        tail_window=12,
    )
    period_receiver, period_donor, period_audit = matched_post_item_sites(
        encoding,
        items,
        receiver_occurrence=1,
        donor_occurrence=2,
        tokenizer=_Tokenizer(),
        tail_window=12,
        site_policy="period_preferred",
    )
    assert (latest_receiver, latest_donor) == (222, 225)
    assert latest_audit["shared_commit_token_text"] == "\n"
    assert (period_receiver, period_donor) == (221, 224)
    assert period_audit["shared_commit_token_text"] == "."
    assert period_audit["receiver_tail_offset"] == 1


def test_fixed_tail_offset_allows_reverse_direction_and_surface_mismatch() -> None:
    prompt = (1,) * 220
    trace = (2, 5, 7, 3, 5, 8, 4, 5, 9, 10)
    encoding = _Encoding(
        input_ids=prompt + trace,
        attention_mask=(1,) * (len(prompt) + len(trace)),
        query_position=len(prompt) + len(trace) - 1,
        prompt_token_count=len(prompt),
        prompt_record_spans=(),
        trace_item_spans=(
            _Span(220, 223),
            _Span(223, 226),
            _Span(226, 229),
        ),
    )
    receiver_site, donor_site, audit = post_item_sites_at_tail_offset(
        encoding,
        ((220, 223), (223, 226), (226, 229)),
        receiver_occurrence=2,
        donor_occurrence=1,
        tokenizer=_Tokenizer(),
        tail_offset=0,
    )
    assert (receiver_site, donor_site) == (225, 222)
    assert audit["surface_token_matched"] is False
    assert audit["alignment_token_delta"] == -3


def test_item_span_uses_complete_equal_length_items() -> None:
    audit = resolve_natural_patch_span(
        ((10, 15), (20, 25)),
        receiver_occurrence=1,
        donor_occurrence=2,
        receiver_site=14,
        donor_site=24,
        patch_scope="item_span",
    )
    assert audit["effective_patch_width"] == 5
    assert audit["receiver_patch_start"] == 10
    assert audit["donor_patch_start"] == 20
    assert audit["equal_length_complete_item"] is True


def test_item_span_crops_longer_item_from_the_left_without_resampling() -> None:
    audit = resolve_natural_patch_span(
        ((10, 17), (20, 25)),
        receiver_occurrence=1,
        donor_occurrence=2,
        receiver_site=16,
        donor_site=24,
        patch_scope="item_span",
    )
    assert audit["effective_patch_width"] == 5
    assert audit["receiver_patch_start"] == 12
    assert audit["donor_patch_start"] == 20
    assert audit["receiver_item_coverage"] == 5 / 7
    assert audit["donor_item_coverage"] == 1.0
    assert audit["equal_length_complete_item"] is False
    assert audit["hidden_state_resampling"] is False
