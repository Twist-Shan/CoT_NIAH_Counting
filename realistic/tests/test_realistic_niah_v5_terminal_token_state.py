from __future__ import annotations

from dataclasses import dataclass

from realistic_niah_v5.terminal_token_state import (
    REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS,
    REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_GEOMETRIES,
    REGISTERED_LOCAL_TERMINAL_TOKEN_STATE_CONDITIONS,
    REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS,
    _grammar_timed_geometry_positions,
    _matched_state_donor_positions,
    _replace_positions,
    _restore_or_identity,
)


@dataclass(frozen=True)
class _Encoding:
    input_ids: tuple[int, ...]


class _Registry:
    query_position = 30
    prompt_token_count = 10

    def positions(self, source_group: str) -> tuple[int, ...]:
        return {
            "terminal_trace_item": tuple(range(20, 29)),
            "trace_other": (10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
            "prompt_records": ((1, 2, 3)),
        }[source_group]


def _span(start: int, end: int) -> dict[str, object]:
    return {
        "status": "ok",
        "full_sequence_token_start": start,
        "full_sequence_token_end": end,
    }


def _state(position: int) -> dict[str, object]:
    return {"status": "ok", "full_sequence_token_index": position}


def test_terminal_token_state_condition_contract_is_unique() -> None:
    assert len(REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS) == 8
    assert len(set(REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS)) == 8
    assert "terminal_token_restore" in REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS
    assert (
        "terminal_token_restore_state_occluded"
        in REGISTERED_TERMINAL_TOKEN_STATE_CONDITIONS
    )
    assert len(REGISTERED_LOCAL_TERMINAL_TOKEN_STATE_CONDITIONS) == 6
    assert len(set(REGISTERED_LOCAL_TERMINAL_TOKEN_STATE_CONDITIONS)) == 6
    assert (
        "ablated_terminal_state_restore"
        in REGISTERED_LOCAL_TERMINAL_TOKEN_STATE_CONDITIONS
    )
    assert len(REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_GEOMETRIES) == 5
    assert len(REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS) == 12
    assert len(set(REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_CONDITIONS)) == 12


def test_replace_positions_preserves_length_and_only_changes_registered_positions() -> None:
    receiver = _Encoding((1, 2, 3, 4))
    donor = _Encoding((9, 8, 7, 6))
    result = _replace_positions(receiver, donor, (1, 3))
    assert result.input_ids == (1, 8, 3, 6)
    assert len(result.input_ids) == len(receiver.input_ids)


def test_empty_diagnostic_partition_is_identity() -> None:
    receiver = _Encoding((1, 2, 3, 4))
    donor = _Encoding((9, 8, 7, 6))
    result = _restore_or_identity(receiver, donor, ())
    assert result is receiver


def test_grammar_timed_update_starts_at_last_semantic_component() -> None:
    registry = _Registry()
    sites = {
        "rank_evidence_core_span": _span(25, 27),
        "city_target_span": _span(21, 23),
        "post_update_commit_state": _state(28),
    }
    after, after_audit = _grammar_timed_geometry_positions(
        registry,
        {"grammar_class": "adjacent_rank_after_city", "sites": sites},
    )
    assert after["grammar_terminal_update"] == (25, 26, 27, 28)
    assert after_audit["grammar_timing_stratum"] == "rank_after_city"

    before_sites = {
        "rank_evidence_core_span": _span(20, 22),
        "city_target_span": _span(24, 26),
        "post_update_commit_state": _state(28),
    }
    before, before_audit = _grammar_timed_geometry_positions(
        registry,
        {"grammar_class": "same_unit_rank_before_city", "sites": before_sites},
    )
    assert before["grammar_terminal_update"] == (24, 25, 26, 27, 28)
    assert before_audit["grammar_timing_stratum"] == "rank_before_city"


def test_matched_state_donor_has_equal_budget_and_excludes_terminal_item() -> None:
    registry = _Registry()
    receivers = (24, 25, 26, 27)
    donors = _matched_state_donor_positions(registry, receivers)
    assert len(donors) == len(receivers)
    assert not set(donors) & set(registry.positions("terminal_trace_item"))
