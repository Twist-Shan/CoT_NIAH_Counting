from __future__ import annotations

from realistic_niah_v5.parsing import parse_and_align_record, parse_trace_record


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


def _row(raw: str, *, family: str, cities: list[str]) -> dict[str, object]:
    return {
        "request_id": "parser-v2-test",
        "model_label": "Gemma4-E4B" if family == "gemma4" else "Qwen3-8B",
        "model_family": family,
        "raw_output_text": raw,
        "output_token_ids": [ord(value) for value in raw],
        "input_ids": [1, 2, 3],
        "gold_records": [
            {"city": city, "score": 50 + index}
            for index, city in enumerate(cities)
        ],
        "seed": 1254,
        "split": "confirmation",
    }


def test_nested_count_markers_recover_only_emitted_one_to_m() -> None:
    raw = (
        "<|channel>thought\n"
        "3. **Execution Scan:**\n"
        "    * *Scan 1:*\n"
        "        * \"In the 2024 city score audit, Athens received a score of 72.\" (Count: 1)\n"
        "    * *Scan 2:*\n"
        "        * \"In the 2024 city score audit, Lisbon received a score of 88.\" (Count: 2)\n"
        "    * *Scan 3:*\n"
        "        * \"In the 2024 city score audit, Zagreb received a score of 63.\" (Count: 3)\n"
        "<channel|>\nTotal: 3"
    )
    result = parse_and_align_record(
        _row(
            raw,
            family="gemma4",
            cities=["Athens", "Lisbon", "Zagreb", "Geneva"],
        ),
        CharacterTokenizer(),
    )
    parser = result["parser"]
    assert parser["marker_kind"] == "inline_count"
    assert parser["item_count"] == 3
    assert parser["item_markers"] == (1, 2, 3)
    assert parser["item_gold_cities"] == ("Athens", "Lisbon", "Zagreb")
    assert parser["missing_gold_cities"] == ("Geneva",)
    assert parser["trace_category"] == "partial_unique"
    for site_kind in (
        "pre_marker",
        "marker_end",
        "pre_city",
        "city_unit_end",
        "item_end",
    ):
        sites = [
            site
            for site in result["token_sites"]
            if site["site_kind"] == site_kind
        ]
        assert [site["occurrence"] for site in sites] == [1, 2, 3]
    pre_markers = [
        site for site in result["token_sites"] if site["site_kind"] == "pre_marker"
    ]
    assert [raw[site["char_end"] :].lstrip(" *(")[:5] for site in pre_markers] == [
        "Count",
        "Count",
        "Count",
    ]


def test_qwen_prose_cardinals_recover_full_sequence() -> None:
    raw = (
        "<think>\n"
        "First excerpt: \"In the 2024 city score audit, Busan received a score of 88.\" That's one record. "
        "Next: \"In the 2024 city score audit, Oslo received a score of 84.\" Second record. "
        "Then: \"Bangkok received a score of 66.\" Three.\n"
        "</think>\nTotal: 3"
    )
    result = parse_trace_record(
        _row(raw, family="qwen3", cities=["Busan", "Oslo", "Bangkok"])
    )
    assert result["parser"]["marker_kind"] == "inline_count"
    assert result["parser"]["item_count"] == 3
    assert result["parser"]["item_markers"] == (1, 2, 3)
    assert len(
        [site for site in result["char_sites"] if site["site_kind"] == "pre_marker"]
    ) == 3


def test_final_total_does_not_set_or_pad_trace_labels() -> None:
    raw = (
        "<think>\n"
        "Paris received a score of 61. Count: 1\n"
        "Rome received a score of 62. Count: 2\n"
        "Oslo received a score of 63. Count: 3\n"
        "</think>\nTotal: 2"
    )
    result = parse_trace_record(
        _row(raw, family="qwen3", cities=["Paris", "Rome", "Oslo", "Baku"])
    )
    assert result["parsed_count"] == 2
    assert result["parser"]["item_count"] == 3
    assert result["parser"]["item_markers"] == (1, 2, 3)
    assert result["parser"]["missing_gold_cities"] == ("Baku",)
    sites = {site["site_kind"]: site for site in result["char_sites"]}
    answer_query = sites["answer_query"]
    assert raw[
        answer_query["char_start"] : answer_query["char_end"]
    ].strip() == "Total:"
    answer_query_v3 = sites["answer_query_v3"]
    assert raw[
        answer_query_v3["char_start"] : answer_query_v3["char_end"]
    ] == "Total: "
    assert raw[answer_query_v3["char_end"]] == "2"


def test_answer_query_v3_survives_a_running_parser_miss() -> None:
    raw = "<think>There is no recoverable city-count episode here.</think>\nTotal: 7"
    result = parse_trace_record(_row(raw, family="qwen3", cities=["Nowhere"]))

    assert result["parser"]["detected"] is False
    sites = {site["site_kind"]: site for site in result["char_sites"]}
    assert set(sites) == {"answer_query", "answer_query_v2", "answer_query_v3"}
    answer_query_v3 = sites["answer_query_v3"]
    assert raw[
        answer_query_v3["char_start"] : answer_query_v3["char_end"]
    ] == "Total: "
    assert raw[answer_query_v3["char_end"]] == "7"


def test_unmarked_score_supported_prose_is_a_trace_sequence() -> None:
    raw = (
        "<think>\n"
        "I found Vancouver with a score of 62, Geneva with 57, and Prague with 65. "
        "Those are the only matching records.\n"
        "</think>\nTotal: 3"
    )
    row = _row(raw, family="qwen3", cities=["Vancouver", "Geneva", "Prague"])
    row["gold_records"] = [
        {"city": "Vancouver", "score": 62},
        {"city": "Geneva", "score": 57},
        {"city": "Prague", "score": 65},
    ]
    result = parse_trace_record(row)
    assert result["parser"]["detected"] is True
    assert result["parser"]["marker_kind"] == "evidence_sequence"
    assert result["parser"]["item_count"] == 3
    assert result["parser"]["item_gold_cities"] == (
        "Vancouver",
        "Geneva",
        "Prague",
    )
    assert result["sequence_source"] == "synthetic_evidence_fallback"
    assert result["parser"]["trace_one_to_one"] is False
    assert result["parser"]["trace_category"] == "synthetic_unverified"
    assert [
        site["occurrence"]
        for site in result["char_sites"]
        if site["site_kind"] == "pre_city"
    ] == [1, 2, 3]


def test_inline_count_sequence_beats_nested_duplicate_bullets() -> None:
    raw = (
        "<|channel>thought\n"
        "* Scan block containing Athens received a score of 72.\n"
        "  * Record 1: Athens, score 72. (Count = 1)\n"
        "* Scan block containing Lisbon received a score of 88.\n"
        "  * Record 2: Lisbon, score 88. (Count = 2)\n"
        "<channel|>\nTotal: 2"
    )
    result = parse_trace_record(
        _row(raw, family="gemma4", cities=["Athens", "Lisbon"])
    )
    assert result["parser"]["marker_kind"] == "inline_count"
    assert result["parser"]["item_count"] == 2
    assert result["parser"]["item_gold_cities"] == ("Athens", "Lisbon")


def test_rank_one_restart_selects_longest_episode_without_cross_splicing() -> None:
    raw = (
        "<think>\n"
        "Athens received a score of 71. Count: 1\n"
        "Lisbon received a score of 72. Count: 2\n"
        "Let me restart carefully.\n"
        "Paris received a score of 73. Count: 1\n"
        "Rome received a score of 74. Count: 2\n"
        "Oslo received a score of 75. Count: 3\n"
        "</think>\nTotal: 3"
    )
    result = parse_trace_record(
        _row(
            raw,
            family="qwen3",
            cities=["Athens", "Lisbon", "Paris", "Rome", "Oslo"],
        )
    )
    assert result["sequence_source"] == "rank_supported_episode"
    assert result["episode_parse"]["raw_sequence_count"] == 2
    assert result["episode_parse"]["selected_terminal_rank"] == 3
    assert result["parser"]["item_gold_cities"] == ("Paris", "Rome", "Oslo")


def test_advancing_rank_preserves_repeated_city_as_non_one_to_one() -> None:
    raw = (
        "<think>\n"
        "Athens received a score of 71. Count: 1\n"
        "Lisbon received a score of 72. Count: 2\n"
        "Lisbon appears again in the recap. Count: 3\n"
        "</think>\nTotal: 3"
    )
    result = parse_trace_record(
        _row(raw, family="qwen3", cities=["Athens", "Lisbon"])
    )
    assert result["parser"]["item_markers"] == (1, 2, 3)
    assert result["parser"]["item_gold_cities"] == (
        "Athens",
        "Lisbon",
        "Lisbon",
    )
    assert result["parser"]["duplicate_gold_city_items"] == 1
    assert result["parser"]["trace_one_to_one"] is False


def test_equal_rank_same_city_restatement_does_not_create_a_restart() -> None:
    raw = (
        "<think>\n"
        "Record 1: Athens received a score of 71. Count: 1\n"
        "Record 2: Lisbon received a score of 72. Count: 2\n"
        "</think>\nTotal: 2"
    )
    result = parse_trace_record(
        _row(raw, family="qwen3", cities=["Athens", "Lisbon"])
    )
    assert result["episode_parse"]["raw_sequence_count"] == 1
    assert result["parser"]["item_gold_cities"] == ("Athens", "Lisbon")
    assert result["parser"]["trace_one_to_one"] is True


def test_same_city_repeated_around_quote_is_one_supported_update() -> None:
    raw = (
        "<think>\n"
        "Chicago received a score of 71. That's one record.\n"
        "Baku received a score of 72. That's the second.\n"
        "Taipei received a score of 73. Third.\n"
        "Another excerpt mentions Harbin: Harbin received a score of 74. "
        "That's the fourth.\n"
        "</think>\nTotal: 4"
    )
    result = parse_trace_record(
        _row(
            raw,
            family="qwen3",
            cities=["Chicago", "Baku", "Taipei", "Harbin"],
        )
    )
    assert result["parser"]["item_markers"] == (1, 2, 3, 4)
    assert result["parser"]["item_gold_cities"] == (
        "Chicago",
        "Baku",
        "Taipei",
        "Harbin",
    )


def test_procedure_heading_does_not_extend_running_count() -> None:
    raw = (
        "<|channel>thought\n"
        "3. Examine the passage:\n"
        "Found 1: Sarajevo, 91.\n"
        "Found 2: Kathmandu, 58.\n"
        "Found 3: Porto, 51.\n"
        "4. Count the Records:\n"
        "* Sarajevo (91) = 1\n"
        "* Kathmandu (58) = 2\n"
        "* Porto (51) = 3\n"
        "<channel|>Total: 3"
    )
    result = parse_trace_record(
        _row(
            raw,
            family="gemma4",
            cities=["Sarajevo", "Kathmandu", "Porto"],
        )
    )
    assert result["parser"]["item_markers"] == (1, 2, 3)
    assert result["parser"]["item_gold_cities"] == (
        "Sarajevo",
        "Kathmandu",
        "Porto",
    )


def test_structural_span_can_extend_an_exact_ranked_city_prefix() -> None:
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
    result = parse_trace_record(
        _row(
            raw,
            family="qwen3",
            cities=["Chicago", "Baku", "Taipei", "Harbin"],
        )
    )
    assert result["sequence_source"] == "structural_extension"
    assert result["parser"]["item_count"] == 4
    assert result["parser"]["item_gold_cities"] == (
        "Chicago",
        "Baku",
        "Taipei",
        "Harbin",
    )
