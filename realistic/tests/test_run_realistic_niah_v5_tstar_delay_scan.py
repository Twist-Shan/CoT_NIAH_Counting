from __future__ import annotations

import pytest

from scripts.run_realistic_niah_v5_tstar_delay_scan import (
    DELAYS,
    _explicit_cue_only_summaries,
    _summaries,
    choose_boundary_summary,
)
from realistic_niah_v5.unnumbered_counter_restore import explicit_count_cues


def _row(delay: int, *, correct: bool, clean: bool = True) -> dict:
    return {
        "delay_tokens": delay,
        "site_clean": clean,
        "forced_total": 10 if correct else 9,
        "forced_total_gold_correct": correct,
        "forced_total_matches_source": correct,
        "immediate_integer_only": True,
        "stopped_on_eos": True,
    }


def test_delay_grid_and_tie_break_choose_smallest_best_clean_boundary() -> None:
    assert DELAYS == (1, 2, 4, 8)
    rows = []
    for delay, correct_n in ((1, 18), (2, 20), (4, 20), (8, 20)):
        rows.extend(
            _row(delay, correct=index < correct_n) for index in range(20)
        )
    summaries = _summaries(rows)
    chosen = choose_boundary_summary(summaries)
    assert chosen["delay_tokens"] == 2
    assert chosen["forced_total_gold_correct_n"] == 20


def test_unclean_delay_is_ineligible_even_if_more_accurate() -> None:
    rows = []
    for delay in DELAYS:
        rows.extend(_row(delay, correct=True) for _index in range(20))
    rows[-1]["site_clean"] = False
    summaries = _summaries(rows)
    assert summaries[-1]["boundary_eligible"] is False
    assert choose_boundary_summary(summaries)["delay_tokens"] == 1


def test_boundary_selection_fails_when_every_delay_has_leakage() -> None:
    summaries = [
        {
            "delay_tokens": delay,
            "forced_total_gold_correct_n": 20,
            "boundary_eligible": False,
        }
        for delay in DELAYS
    ]
    with pytest.raises(RuntimeError, match="No delay is cue-free"):
        choose_boundary_summary(summaries)


def test_post_tstar_cue_audit_matches_prefix_clean_grammar() -> None:
    cities = ["Paris", "Tokyo"]
    assert explicit_count_cues(" and then pause.", cities=cities, max_index=10) == []
    assert {
        cue["kind"]
        for cue in explicit_count_cues(
            " That is the tenth record. Paris (10)",
            cities=cities,
            max_index=10,
        )
    } == {"labeled_record_index", "city_parenthetical_index"}


def test_v2_allows_repeated_known_city_but_not_explicit_count_cue() -> None:
    rows = []
    for delay in DELAYS:
        for index in range(20):
            row = _row(delay, correct=delay == 4 or index < 12)
            row["explicit_count_cues"] = []
            row["post_tstar_gold_city_mentions"] = ["London"] if index == 0 else []
            rows.append(row)
    rows[-1]["explicit_count_cues"] = [{"kind": "running_progress"}]
    summaries = _explicit_cue_only_summaries(rows)
    assert summaries[0]["boundary_eligible"] is True
    assert summaries[0]["repeated_gold_city_n"] == 1
    assert summaries[-1]["boundary_eligible"] is False
    assert choose_boundary_summary(summaries)["delay_tokens"] == 4
