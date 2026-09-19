from __future__ import annotations

from scripts.run_realistic_niah_v5_tstar_forced_total import (
    FORCED_SUFFIX,
    extract_immediate_integer,
    extract_total,
    summarize_rows,
)


def test_forced_suffix_closes_thinking_and_opens_one_total_line() -> None:
    assert FORCED_SUFFIX == "\n</think>\n\nTotal: "
    assert FORCED_SUFFIX.count("</think>") == 1
    assert FORCED_SUFFIX.count("Total:") == 1


def test_total_parsers_distinguish_immediate_integer_from_extra_text() -> None:
    assert extract_total("<think>x</think>\n\nTotal: 3") == 3
    assert extract_total("Total: 2\nTotal: 3") == 3
    assert extract_total("There are 3 records") is None
    assert extract_immediate_integer("3") == 3
    assert extract_immediate_integer(" 3\n") == 3
    assert extract_immediate_integer("3\nExplanation") is None


def test_summary_keeps_parse_correctness_and_surface_separate() -> None:
    rows = [
        {
            "forced_total": 3,
            "forced_total_gold_correct": True,
            "forced_total_matches_source": True,
            "immediate_integer_only": True,
            "stopped_on_eos": True,
        },
        {
            "forced_total": None,
            "forced_total_gold_correct": False,
            "forced_total_matches_source": False,
            "immediate_integer_only": False,
            "stopped_on_eos": False,
        },
    ]
    assert summarize_rows(rows) == {
        "row_count": 2,
        "forced_total_parsed_n": 1,
        "forced_total_gold_correct_n": 1,
        "forced_total_matches_source_n": 1,
        "immediate_integer_only_n": 1,
        "stopped_on_eos_n": 1,
    }
