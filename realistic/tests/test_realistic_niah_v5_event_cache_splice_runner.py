from __future__ import annotations

from dataclasses import dataclass

from scripts.run_realistic_niah_v5_event_cache_splice import (
    PRIMARY_INVALID_VARIANT,
    VALID_VARIANT,
    build_cache_splice_geometry,
)
from scripts.run_realistic_niah_v5_list_event_edit_scan import (
    build_list_event_variants,
)


@dataclass(frozen=True)
class _Encoding:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    prompt_token_count: int
    query_position: int
    trace_item_spans: tuple[tuple[int, int], ...] = ()
    slot_spans: tuple[tuple[int, int], ...] = ()
    needle_spans: tuple[tuple[int, int], ...] = ()

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class _Registry:
    trace_items: tuple[tuple[int, int], ...]
    trace_markers: tuple[tuple[int, int], ...]


def test_cache_splice_geometry_freezes_equal_length_common_suffix() -> None:
    # Ten 4-token items; token 0 is the marker and token 3 is the shared close.
    items = tuple((5 + 4 * index, 9 + 4 * index) for index in range(10))
    markers = tuple((start, start + 1) for start, _end in items)
    ids = tuple(range(50))
    source = _Encoding(ids, (1,) * len(ids), 5, 48)
    neutral_ids = list(ids)
    for start, end in items:
        neutral_ids[start:end] = (90, 91, 92, ids[end - 1])
    neutral = _Encoding(tuple(neutral_ids), (1,) * len(ids), 5, 48)
    registry = _Registry(items, markers)
    boundaries = {index + 1: end - 1 for index, (_start, end) in enumerate(items)}
    variants = {
        row["event_variant"]: row
        for row in build_list_event_variants(
            source,
            neutral,
            registry,
            receiver=5,
            current_boundary=boundaries[5],
            target_boundary=boundaries[6],
            insert_source_occurrence=4,
            delete_occurrence=3,
        )
    }

    geometry = build_cache_splice_geometry(
        variants[VALID_VARIANT],
        variants[PRIMARY_INVALID_VARIANT],
        registry,
        boundaries,
        receiver=5,
        insert_source_occurrence=4,
    )

    assert geometry["insertion_start"] == items[5][0]
    assert geometry["event_token_count"] == 4
    assert geometry["regions"]["event"] == tuple(range(items[5][0], items[5][0] + 4))
    assert len(geometry["regions"]["marker"]) == 1
    assert len(geometry["regions"]["closing"]) == 1
    assert geometry["closing_surface_token_identical"]
    assert geometry["changed_event_token_positions"] == [items[5][0]]
    assert len(geometry["regions"]["preceding_event_width"]) == 4
