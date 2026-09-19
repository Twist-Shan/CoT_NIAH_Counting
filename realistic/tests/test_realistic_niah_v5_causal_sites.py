from __future__ import annotations

import pytest

from realistic_niah_v5.causal_sites import (
    CausalSiteError,
    build_output_token_map,
    compile_causal_site_plan,
)


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        assert not add_special_tokens
        result = {"input_ids": [ord(value) for value in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(value) for value in text]

    def decode(
        self,
        values,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(int(value)) for value in values)


def _row(
    raw: str,
    *,
    cities: list[str],
    family: str = "qwen3",
) -> dict[str, object]:
    prompt = ""
    spans = []
    records = []
    for index, city in enumerate(cities, start=1):
        start = len(prompt)
        score = 60 + index
        prompt += f"{city} received a score of {score}.\n"
        spans.append(
            {
                "slot_index": index,
                "city": city,
                "score": score,
                "start": start,
                "end": len(prompt),
            }
        )
        records.append({"city": city, "score": score})
    return {
        "request_id": "causal-site-test",
        "model_label": "Gemma4-E4B" if family == "gemma4" else "Qwen3-8B",
        "model_family": family,
        "raw_output_text": raw,
        "output_token_ids": [ord(value) for value in raw],
        "input_ids": [ord(value) for value in prompt],
        "prompt_record_spans": spans,
        "gold_records": records,
        "seed": 1254,
        "split": "confirmation",
    }


@pytest.mark.parametrize(
    ("raw", "expected_order"),
    [
        (
            "<think>\n1. Chicago received a score.\n"
            "2. Baku received a score.\n</think>\nTotal: 2",
            "rank_before_city",
        ),
        (
            "<think>\nChicago received a score. Count: 1\n"
            "Baku received a score. Count: 2\n</think>\nTotal: 2",
            "rank_after_city",
        ),
    ],
)
def test_rank_grammar_selects_sites_from_actual_surface_order(
    raw: str, expected_order: str
) -> None:
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku"]), CharacterTokenizer()
    )
    assert plan["causal_cohort"] == "primary_rank_resolved_full_chain"
    assert plan["primary_full_chain_site_complete"] is True
    for event in plan["events"]:
        sites = event["sites"]
        city = sites["city_target_span"]
        query = sites["retrieve_query_state"]
        rank = sites["rank_evidence_core_span"]
        commit = sites["post_update_commit_state"]
        assert event["surface_order"] == expected_order
        assert expected_order in event["grammar_class"]
        assert query["output_token_index"] == city["output_token_start"] - 1
        assert query["token_id"] == ord(raw[query["output_token_index"]])
        assert event["city"].casefold() in event["prompt_source_record"][
            "token_text"
        ].casefold()
        if expected_order == "rank_after_city":
            assert city["output_token_end"] <= rank["output_token_start"]
            assert commit["output_token_index"] >= rank["output_token_end"] - 1
        else:
            assert rank["output_token_end"] <= city["output_token_start"]
            assert event["retrieval_surface_variant"] == (
                "rank_before_city_compact"
            )
            assert event["rank_to_city_has_lexical_content"] is False


def test_rank_before_city_record_clause_is_not_pooled_with_compact_index() -> None:
    raw = (
        '<think>\n1. "In the 2024 city score audit, Chicago received a '
        'score of 61."\n2. "In the 2024 city score audit, Baku received a '
        'score of 62."\n</think>\nTotal: 2'
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku"]), CharacterTokenizer()
    )
    assert [
        event["retrieval_surface_variant"] for event in plan["events"]
    ] == ["rank_before_city_record_clause"] * 2
    transition = plan["transitions"][0]
    assert transition["target_retrieval_surface_variant"] == (
        "rank_before_city_record_clause"
    )
    assert transition["target_rank_to_city_has_lexical_content"] is True
    assert transition["target_rank_to_city_interstitial_char_count"] > 10


def test_continue_and_stop_targets_are_strictly_after_commit() -> None:
    raw = (
        "<think>\nChicago received a score. Count: 1\n"
        "Baku received a score. Count: 2\n</think>\nTotal: 2"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku"]), CharacterTokenizer()
    )
    transition = plan["transitions"][0]
    assert transition["query_output_token_index"] == plan["events"][0]["sites"][
        "post_update_commit_state"
    ]["output_token_index"]
    assert transition["full_continuation_output_token_start"] == (
        transition["query_output_token_index"] + 1
    )
    assert transition["full_continuation_output_token_end"] == transition[
        "next_city_output_token_end"
    ]
    assert transition["primary_transition_eligible"] is True
    terminal = plan["terminal_transition"]
    assert terminal["query_output_token_index"] < terminal[
        "full_continuation_output_token_start"
    ] < terminal["full_continuation_output_token_end"]
    assert terminal["answer_query_token_text"] == "Total: "
    assert terminal["answer_query_output_token_end"] == terminal[
        "full_continuation_output_token_end"
    ]
    assert terminal["primary_terminal_eligible"] is True


def test_transition_anchor_registry_deduplicates_semantic_aliases() -> None:
    raw = (
        "<think>\n1. Chicago received a score.\n"
        "2. Baku received a score.\n</think>\nTotal: 2"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku"]), CharacterTokenizer()
    )
    transition = plan["transitions"][0]
    anchors = transition["anchors"]
    positions = [anchor["output_token_index"] for anchor in anchors]
    assert len(positions) == len(set(positions))
    assert all(
        anchor["output_token_index"]
        < transition["next_city_output_token_start"]
        for anchor in anchors
    )
    aliases = [set(anchor["anchor_roles"]) for anchor in anchors]
    assert {"unit_pre_d1", "pre_marker_d1"} in aliases
    block = next(
        anchor for anchor in anchors if "block_pre_d1" in anchor["anchor_roles"]
    )
    assert block["event_specific"] is False
    assert block["primary_anchor_eligible"] is False
    assert all(anchor["local_anchor_eligible"] for anchor in anchors)


def test_rank_after_city_does_not_invent_pre_or_post_marker_anchors() -> None:
    raw = (
        "<think>\nChicago received a score. Count: 1\n"
        "Baku received a score. Count: 2\n</think>\nTotal: 2"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku"]), CharacterTokenizer()
    )
    transition = plan["transitions"][0]
    resolved_roles = {
        role for anchor in transition["anchors"] for role in anchor["anchor_roles"]
    }
    assert "pre_marker_d1" not in resolved_roles
    assert "post_marker" not in resolved_roles
    by_role = {
        candidate["anchor_role"]: candidate
        for candidate in transition["anchor_candidates"]
    }
    assert by_role["pre_marker_d1"]["not_applicable_reason"] == (
        "rank_marker_not_before_target_city"
    )
    assert by_role["post_marker"]["not_applicable_reason"] == (
        "rank_marker_not_before_target_city"
    )


def test_parenthesis_and_record_clause_anchors_are_surface_grounded() -> None:
    parenthesized = (
        "<think>\n(Record 1: Chicago, 61)\n"
        "(Record 2: Baku, 62)\n</think>\nTotal: 2"
    )
    paren_plan = compile_causal_site_plan(
        _row(parenthesized, cities=["Chicago", "Baku"]),
        CharacterTokenizer(),
    )
    paren_aliases = [
        set(anchor["anchor_roles"])
        for anchor in paren_plan["transitions"][0]["anchors"]
    ]
    assert any("post_open_delimiter" in aliases for aliases in paren_aliases)
    delimiter = paren_plan["events"][1]["sites"]["opening_delimiter_span"]
    assert delimiter["char_text"] == "("

    clause = (
        "<think>\nIn the 2024 city score audit, Chicago received a score of 61. "
        "Count: 1\nIn the 2024 city score audit, Baku received a score of 62. "
        "Count: 2\n</think>\nTotal: 2"
    )
    clause_plan = compile_causal_site_plan(
        _row(clause, cities=["Chicago", "Baku"]), CharacterTokenizer()
    )
    clause_aliases = [
        set(anchor["anchor_roles"])
        for anchor in clause_plan["transitions"][0]["anchors"]
    ]
    assert any("record_clause_pre_d1" in aliases for aliases in clause_aliases)
    clause_span = clause_plan["events"][1]["sites"]["record_clause_span"]
    assert clause_span["char_text"] == "In the 2024 city score audit"


def test_invariant_bullets_enter_marker_neutral_secondary_cohort() -> None:
    raw = (
        "<|channel>thought\nI will enumerate.\n"
        "* Chicago received a score.\n"
        "• Baku received a score.\n"
        "Therefore there are two.\n<channel|>\nTotal: 2"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku"], family="gemma4"),
        CharacterTokenizer(),
    )
    assert plan["sequence_source"] == "structural_fallback"
    assert plan["causal_cohort"] == "secondary_structural_marker_neutral"
    assert plan["grammar_signature"] == "structural_invariant_bullet"
    assert plan["primary_full_chain_site_complete"] is False
    assert all(
        event["marker_semantics"] == "invariant_marker"
        and not event["eligibility"]["marker_control"]
        and not event["eligibility"]["format_shell_control"]
        and event["eligibility"]["invariant_marker_surface_control"]
        and event["eligibility"]["structural_marker_neutral_secondary"]
        for event in plan["events"]
    )
    assert all(
        event["sites"]["rank_evidence_core_span"]["char_text"]
        == event["sites"]["rank_evidence_core_span"]["char_text"].strip()
        for event in plan["events"]
    )


def test_ordinal_rank_span_drops_semantically_empty_trailing_space() -> None:
    raw = (
        "<think>\nFirst excerpt: Chicago received a score.\n"
        "Second excerpt: Baku received a score.\n</think>\nTotal: 2"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku"]), CharacterTokenizer()
    )
    assert plan["causal_cohort"] == "primary_rank_resolved_full_chain"
    assert [
        event["sites"]["rank_evidence_core_span"]["char_text"]
        for event in plan["events"]
    ] == ["First", "Second"]
    assert [
        event["semantic_span_normalization"]["rank_evidence_surface_span"][
            "right_trimmed_chars"
        ]
        for event in plan["events"]
    ] == [1, 1]


def test_rank_core_separates_numeric_marker_from_visible_delimiter() -> None:
    raw = (
        "<think>\n1. Chicago received a score.\n"
        "2. Baku received a score.\n</think>\nTotal: 2"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku"]), CharacterTokenizer()
    )
    assert [
        event["sites"]["rank_evidence_core_span"]["char_text"]
        for event in plan["events"]
    ] == ["1", "2"]
    assert [
        event["sites"]["rank_evidence_surface_span"]["char_text"]
        for event in plan["events"]
    ] == ["1.", "2."]
    assert [
        event["sites"]["rank_visible_format_shell_tokens"]["token_text"]
        for event in plan["events"]
    ] == [".", "."]
    assert all(
        event["eligibility"]["marker_control"]
        and event["eligibility"]["format_shell_control"]
        for event in plan["events"]
    )


def test_partial_unique_trace_has_local_but_not_terminal_primary_estimand() -> None:
    raw = (
        "<think>\nChicago received a score. Count: 1\n"
        "Baku received a score. Count: 2\n</think>\nTotal: 2"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku", "Taipei"]), CharacterTokenizer()
    )
    assert plan["trace_category"] == "partial_unique"
    assert plan["causal_cohort"] == "secondary_local_partial_unique"
    assert plan["transitions"][0]["local_transition_eligible"] is True
    assert plan["transitions"][0]["primary_transition_eligible"] is False
    assert plan["terminal_transition"]["observed_stop_eligible"] is False
    assert plan["terminal_transition"]["primary_terminal_eligible"] is False


def test_duplicate_trace_is_scoped_to_occurrence_retrieval_only() -> None:
    raw = (
        "<think>\n1. Chicago received a score.\n"
        "2. Baku received a score.\n"
        "3. Baku received a score again.\n</think>\nTotal: 2"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku"]), CharacterTokenizer()
    )
    assert plan["causal_cohort"] == "occurrence_retrieval_only_duplicates"
    assert plan["allowed_estimands"] == ["occurrence_source_to_city_retrieval"]
    assert all(event["eligibility"]["retrieval"] for event in plan["events"])
    assert not any(
        event["eligibility"]["progress_commit"]
        or event["eligibility"]["marker_control"]
        for event in plan["events"]
    )
    assert not any(
        transition["local_transition_eligible"]
        for transition in plan["transitions"]
    )
    assert plan["terminal_transition"]["observed_stop_eligible"] is False


def test_score_supported_evidence_sequence_is_exploratory_not_discarded() -> None:
    raw = (
        "<think>\nI found Vancouver with a score of 61, Geneva with 62, "
        "and Prague with 63. Those are the only matching records.\n"
        "</think>\nTotal: 3"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Vancouver", "Geneva", "Prague"]),
        CharacterTokenizer(),
    )
    assert plan["sequence_source"] == "synthetic_evidence_fallback"
    assert plan["causal_cohort"] == "secondary_evidence_sequence_exploratory"
    assert plan["grammar_signature"] == "evidence_sequence_unranked"
    assert plan["plan_exclusion_reasons"] == []
    assert plan["allowed_estimands"] == [
        "score_supported_source_to_city_retrieval",
        "evidence_sequence_continue",
        "evidence_sequence_terminal_stop",
    ]
    assert all(
        event["rank_basis"] == "compiler_occurrence_index_only"
        and event["sites"]["rank_evidence_core_span"] is None
        and event["eligibility"]["retrieval"]
        and event["eligibility"]["progress_commit"]
        and not event["eligibility"]["marker_control"]
        for event in plan["events"]
    )
    assert all(
        transition["local_transition_eligible"]
        and not transition["primary_transition_eligible"]
        for transition in plan["transitions"]
    )
    assert plan["terminal_transition"]["observed_stop_eligible"] is True
    assert plan["terminal_transition"]["primary_terminal_eligible"] is False


def test_partial_evidence_sequence_disables_terminal_count_claim() -> None:
    raw = (
        "<think>\nI found Vancouver with a score of 61 and Geneva with 62. "
        "I did not finish checking the remaining records.\n</think>\nTotal: 2"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Vancouver", "Geneva", "Prague"]),
        CharacterTokenizer(),
    )
    assert plan["sequence_source"] == "synthetic_evidence_fallback"
    assert (
        plan["causal_cohort"]
        == "secondary_evidence_sequence_partial_exploratory"
    )
    assert plan["allowed_estimands"] == [
        "local_score_supported_source_to_city_retrieval",
        "local_evidence_sequence_continue",
    ]
    assert plan["transitions"][0]["local_transition_eligible"] is True
    assert plan["terminal_transition"]["observed_stop_eligible"] is False
    assert plan["terminal_transition"]["primary_terminal_eligible"] is False


def test_structural_extension_is_recap_only() -> None:
    raw = (
        "<think>\n"
        "Chicago received a score of 71. That's one record.\n"
        "Baku received a score of 72. That's the second.\n"
        "Taipei received a score of 73. Third.\n"
        "Complete recap:\n"
        "- Chicago received a score of 71.\n"
        "- Baku received a score of 72.\n"
        "- Taipei received a score of 73.\n"
        "- Harbin received a score of 74.\n"
        "No other matching records.\n"
        "</think>\nTotal: 4"
    )
    plan = compile_causal_site_plan(
        _row(raw, cities=["Chicago", "Baku", "Taipei", "Harbin"]),
        CharacterTokenizer(),
    )
    assert plan["sequence_source"] == "structural_extension"
    assert plan["causal_cohort"] == "secondary_structural_recap"
    assert len(plan["events"]) == 4
    assert all(event["event_source"] == "structural" for event in plan["events"])
    assert not any(
        transition["primary_transition_eligible"]
        for transition in plan["transitions"]
    )


def test_exact_retokenization_and_prompt_source_identity_are_hard_gates() -> None:
    raw = "<think>\n1. Chicago received a score.\n</think>\nTotal: 1"
    mismatched = _row(raw, cities=["Chicago"])
    mismatched["output_token_ids"][-1] += 1
    with pytest.raises(CausalSiteError, match="Re-tokenized output differs"):
        compile_causal_site_plan(mismatched, CharacterTokenizer())

    wrong_source = _row(raw, cities=["Chicago"])
    wrong_source["prompt_record_spans"][0]["start"] += len("Chicago")
    with pytest.raises(CausalSiteError, match="does not decode to its city"):
        compile_causal_site_plan(wrong_source, CharacterTokenizer())


def test_shared_offsets_for_multibyte_byte_tokens_are_valid() -> None:
    class ByteSplitTokenizer:
        def __call__(self, _text: str, **_kwargs):
            return {
                "input_ids": [1, 2, 3, 4],
                "offset_mapping": [(0, 1), (1, 2), (1, 2), (2, 3)],
            }

        def decode(self, values, **_kwargs):
            return "".join(str(value) for value in values)

    token_map = build_output_token_map(
        {"raw_output_text": "a\u2029b", "output_token_ids": [1, 2, 3, 4]},
        ByteSplitTokenizer(),
    )
    shared = token_map.span("unicode_separator", 1, 2)
    assert shared["status"] == "ok"
    assert shared["token_ids"] == [2, 3]
