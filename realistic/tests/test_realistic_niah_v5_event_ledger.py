from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from realistic_niah_v5.event_ledger import (
    build_marker_event_factorial,
    build_semantic_event_factorial,
)


@dataclass(frozen=True)
class _Encoding:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    query_position: int
    prompt_token_count: int
    trace_item_spans: tuple[object, ...] = ()
    slot_spans: tuple[object, ...] = ()
    needle_spans: tuple[object, ...] = ()

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


def _fixture() -> tuple[_Encoding, _Encoding, SimpleNamespace, dict[int, int]]:
    encoding = _Encoding(
        input_ids=tuple(range(42)),
        attention_mask=(1,) * 42,
        query_position=41,
        prompt_token_count=2,
    )
    neutral = _Encoding(
        input_ids=tuple(range(100, 142)),
        attention_mask=(1,) * 42,
        query_position=41,
        prompt_token_count=2,
    )
    registry = SimpleNamespace(
        trace_items=tuple((2 + 4 * index, 5 + 4 * index) for index in range(10)),
        trace_markers=tuple((2 + 4 * index, 3 + 4 * index) for index in range(10)),
    )
    boundaries = {index: 1 + 4 * index for index in range(1, 11)}
    return encoding, neutral, registry, boundaries


def test_three_event_factorial_changes_only_selected_marker_tokens() -> None:
    encoding, neutral, registry, boundaries = _fixture()
    variants, geometry = build_marker_event_factorial(
        encoding,
        neutral,
        registry,
        boundaries,
        receiver=5,
        source_occurrences=(2, 3, 4),
    )

    assert len(variants) == 8
    assert [row["variant_id"] for row in variants] == [
        "markers_000",
        "markers_001",
        "markers_010",
        "markers_011",
        "markers_100",
        "markers_101",
        "markers_110",
        "markers_111",
    ]
    assert geometry["inserted_marker_positions"] == [22, 26, 30]
    assert geometry["target_marker_position"] == 34
    assert geometry["target_boundary"] == 37
    assert geometry["only_marker_token_ids_vary"]
    assert {row["event_count_target"] for row in variants} == {6, 7, 8, 9}

    baseline = variants[0]["encoding"].input_ids
    for variant in variants:
        changed = {
            index
            for index, (left, right) in enumerate(
                zip(baseline, variant["encoding"].input_ids)
            )
            if left != right
        }
        expected = {
            position
            for bit, slot in zip(
                variant["marker_bits"], geometry["inserted_slots"]
            )
            if bit
            for position in slot["marker_positions"]
        }
        assert changed == expected


def test_factorial_rejects_a_source_without_registered_marker() -> None:
    encoding, neutral, registry, boundaries = _fixture()
    registry.trace_markers = tuple(
        span for index, span in enumerate(registry.trace_markers) if index != 2
    )
    try:
        build_marker_event_factorial(
            encoding,
            neutral,
            registry,
            boundaries,
            receiver=5,
            source_occurrences=(2, 3, 4),
        )
    except ValueError as exc:
        assert "lacks marker" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("A markerless source should fail deterministic preflight")


def test_single_semantic_event_factorial_preserves_commit_geometry() -> None:
    encoding, _unused, registry, boundaries = _fixture()
    neutral_ids = list(encoding.input_ids)
    for start, end in registry.trace_items:
        for position in range(start, end):
            neutral_ids[position] += 1000
    neutral = _Encoding(
        input_ids=tuple(neutral_ids),
        attention_mask=encoding.attention_mask,
        query_position=encoding.query_position,
        prompt_token_count=encoding.prompt_token_count,
    )
    variants, geometry = build_semantic_event_factorial(
        encoding,
        neutral,
        registry,
        boundaries,
        receiver=1,
        source_occurrences=(1,),
    )

    assert [row["variant_id"] for row in variants] == ["events_0", "events_1"]
    assert [row["event_count_at_inserted_commit"] for row in variants] == [1, 2]
    assert geometry["factor_count"] == 1
    assert geometry["all_cells_equal_length"]
    assert geometry["only_event_semantic_token_ids_vary"]
    slot = geometry["inserted_slots"][0]
    boundary = int(slot["event_boundary"])
    assert (
        variants[0]["encoding"].input_ids[boundary]
        == variants[1]["encoding"].input_ids[boundary]
    )
    changed = {
        index
        for index, (left, right) in enumerate(
            zip(variants[0]["encoding"].input_ids, variants[1]["encoding"].input_ids)
        )
        if left != right
    }
    assert changed == set(slot["discriminative_positions"])
