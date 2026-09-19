from __future__ import annotations

from scripts.scan_realistic_niah_v5_frozen_prompt_noindex_n3 import (
    assigned_split,
    choose_rows,
    format_audit,
)


def _row(seed: int, eligible: bool, *, strict: bool = False) -> dict:
    return {
        "seed": seed,
        "gold_count": 3,
        "noindex_n3_format_audit": {
            "marker_kind": "audit_sentence",
            "primary_eligible_prefix_clean": eligible,
            "global_clean_no_running_index": strict,
            "primary_eligible_no_running_index": eligible,
            "strict_eligible_no_explicit_count_cue": strict,
            "reasons": [],
        },
    }


def _trace_row(reasoning: str, *, marker_kind: str = "audit_sentence") -> dict:
    prefix = "<think>\n"
    raw = prefix + reasoning + "\n</think>\n\nTotal: 3"
    cities = (("Paris", 61), ("Tokyo", 72), ("Lima", 48))
    starts = [raw.index(city, len(prefix)) for city, _score in cities]
    ends = []
    for (city, score), start in zip(cities, starts):
        score_start = raw.index(str(score), start + len(city))
        ends.append(score_start + len(str(score)))
    return {
        "gold_count": 3,
        "gold_records": [
            {"city": city, "score": score} for city, score in cities
        ],
        "raw_output_text": raw,
        "trace_parse": {
            "parser": {
                "trace_one_to_one": True,
                "item_count": 3,
                "reasoning_start_char": len(prefix),
                "reasoning_end_char": raw.index("</think>"),
                "item_start_chars": starts,
                "item_end_chars": ends,
                "marker_kind": marker_kind,
            }
        },
    }


def test_split_policy_matches_existing_causal_supplement() -> None:
    assert all(assigned_split(seed) == "discovery" for seed in range(1234, 1254))
    assert all(assigned_split(seed) == "confirmation" for seed in range(1254, 1264))
    assert assigned_split(1264) == "discovery"
    assert assigned_split(1265) == "confirmation"
    assert [assigned_split(seed) for seed in range(1266, 1272)] == [
        "discovery",
        "discovery",
        "confirmation",
        "discovery",
        "discovery",
        "confirmation",
    ]


def test_selection_is_ascending_within_fixed_split_and_format_only() -> None:
    rows = [_row(seed, eligible=True, strict=seed % 2 == 0) for seed in range(1234, 1280)]
    discovery, confirmation, ledger = choose_rows(list(reversed(rows)))
    assert [row["seed"] for row in discovery] == list(range(1234, 1254))
    assert [row["seed"] for row in confirmation] == list(range(1254, 1264))
    assert len([row for row in ledger if row["selected"]]) == 30
    assert all(row["mechanism_outcomes_accessed"] is False for row in ledger)


def test_ineligible_base_seeds_are_filled_by_later_same_split_seeds() -> None:
    rows = [_row(seed, eligible=seed not in {1234, 1254}) for seed in range(1234, 1272)]
    discovery, confirmation, _ledger = choose_rows(rows)
    assert 1234 not in [row["seed"] for row in discovery]
    assert 1264 in [row["seed"] for row in discovery]
    assert 1254 not in [row["seed"] for row in confirmation]
    assert 1265 in [row["seed"] for row in confirmation]


def test_bullets_with_explicit_running_counts_are_not_noindex() -> None:
    items = [
        "* Paris received 61. (Count = 1)\n",
        "* Tokyo received 72. (Count = 2)\n",
        "* Lima received 48. (Count = 3)\n",
    ]
    prefix = "<think>\n"
    raw = prefix + "".join(items)
    starts = []
    ends = []
    cursor = len(prefix)
    for item in items:
        starts.append(cursor)
        cursor += len(item)
        ends.append(cursor)
    row = {
        "gold_count": 3,
        "gold_records": [
            {"city": "Paris", "score": 61},
            {"city": "Tokyo", "score": 72},
            {"city": "Lima", "score": 48},
        ],
        "raw_output_text": raw,
        "trace_parse": {
            "parser": {
                "trace_one_to_one": True,
                "item_count": 3,
                "reasoning_start_char": 0,
                "item_start_chars": starts,
                "item_end_chars": ends,
                "marker_kind": "bullet",
            }
        },
    }
    audit = format_audit(row)
    assert audit["primary_eligible_no_running_index"] is False
    assert audit["strict_eligible_no_explicit_count_cue"] is False
    assert any("running_progress_before_t_star" in reason for reason in audit["reasons"])


def test_numbered_recap_after_first_complete_pass_is_prefix_clean_only() -> None:
    row = _trace_row(
        "Paris received a score of 61. Tokyo received a score of 72. "
        "Lima received a score of 48.\n"
        "Let me count these: Paris (1), Tokyo (2), Lima (3).",
        marker_kind="indexed",
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is True
    assert audit["global_clean_no_running_index"] is False
    assert "global_explicit_parser_marker_kind:indexed" in audit["global_reasons"]
    assert audit["t_star_char"] < row["raw_output_text"].index("Let me count")
    assert audit["selection_site_kind"] == "first_score_supported_gold_mentions"


def test_spelled_running_count_before_later_records_is_not_prefix_clean() -> None:
    row = _trace_row(
        "Paris received a score of 61. That's one. "
        "Tokyo received a score of 72. That's two. "
        "Lima received a score of 48. That's three."
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is False
    assert audit["global_clean_no_running_index"] is False
    assert any("running_progress_before_t_star" in reason for reason in audit["reasons"])


def test_clean_first_pass_and_terminal_aggregate_are_globally_clean() -> None:
    row = _trace_row(
        "- Paris: 61\n- Tokyo: 72\n- Lima: 48\n"
        "The total number of records is three.",
        marker_kind="bullet",
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is True
    assert audit["global_clean_no_running_index"] is True


def test_labeled_first_mention_before_record_is_not_prefix_clean() -> None:
    row = _trace_row(
        "The first mention is Paris with a score of 61. "
        "Tokyo received a score of 72. Lima received a score of 48."
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is False
    assert audit["global_clean_no_running_index"] is False
    assert any("labeled_record_index_before_t_star" in reason for reason in audit["reasons"])


def test_posthoc_ordinal_is_prefix_clean_but_not_global_clean() -> None:
    row = _trace_row(
        "Paris received a score of 61. Tokyo received a score of 72. "
        "Lima received a score of 48. That's the third."
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is True
    assert audit["global_clean_no_running_index"] is False
    assert "global_labeled_record_index" in audit["global_reasons"]


def test_bare_cardinal_between_records_is_not_prefix_clean() -> None:
    row = _trace_row(
        "Paris received a score of 61. One. "
        "Tokyo received a score of 72. Two. "
        "Lima received a score of 48. Three."
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is False
    assert audit["global_clean_no_running_index"] is False
    assert any("bare_cardinal_sentence_before_t_star" in reason for reason in audit["reasons"])


def test_indexed_parser_marker_alone_excludes_global_not_prefix_clean() -> None:
    row = _trace_row(
        "Paris received a score of 61. Tokyo received a score of 72. "
        "Lima received a score of 48.\n"
        "1. Paris - 61\n2. Tokyo - 72\n3. Lima - 48",
        marker_kind="indexed",
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is True
    assert audit["global_clean_no_running_index"] is False
    assert "global_explicit_parser_marker_kind:indexed" in audit["global_reasons"]


def test_shorthand_numbered_gold_lines_before_tstar_are_not_prefix_clean() -> None:
    row = _trace_row(
        "1. Paris - 61\n2. Tokyo - 72\n3. Lima - 48\n",
        marker_kind="indexed",
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is False
    assert {
        cue["kind"] for cue in audit["prefix_cues"]
    } == {"gold_city_leading_index"}


def test_replayed_gold_evidence_before_full_coverage_is_not_first_pass() -> None:
    row = _trace_row(
        "Paris received a score of 61. Paris received a score of 61. "
        "Tokyo received a score of 72. Lima received a score of 48."
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is False
    assert audit["first_pass_complete"] is False
    assert audit["pre_tstar_score_supported_event_count"] == 4
    assert [
        event["city"] for event in audit["pre_tstar_repeated_gold_evidence"]
    ] == ["Paris"]
    assert "repeated_gold_evidence_before_t_star" in audit["reasons"]


def test_replay_and_numbered_recap_after_tstar_do_not_contaminate_first_pass() -> None:
    row = _trace_row(
        "Paris received a score of 61. Tokyo received a score of 72. "
        "Lima received a score of 48.\n"
        "Let me count again:\n1. Paris - 61\n2. Tokyo - 72\n3. Lima - 48",
        marker_kind="indexed",
    )
    audit = format_audit(row)
    assert audit["primary_eligible_prefix_clean"] is True
    assert audit["first_pass_complete"] is True
    assert audit["pre_tstar_score_supported_event_count"] == 3
    assert audit["pre_tstar_repeated_gold_evidence"] == []
