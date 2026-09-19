from __future__ import annotations

from scripts.run_realistic_niah_v5_event_ledger_behavior_regions import (
    compile_region_positions,
)


def test_region_compiler_partitions_marker_and_nonmarker_event_tokens() -> None:
    geometry = {
        "inserted_slots": [
            {"start": 10, "end": 14, "marker_positions": [10], "event_boundary": 13},
            {"start": 14, "end": 19, "marker_positions": [14, 15], "event_boundary": 18},
            {"start": 19, "end": 22, "marker_positions": [19], "event_boundary": 21},
        ]
    }
    regions = compile_region_positions(geometry)
    assert regions["marker"] == (10, 14, 15, 19)
    assert regions["closing"] == (13, 18, 21)
    assert set(regions["marker"]).isdisjoint(regions["nonmarker_event"])
    assert set(regions["marker"]) | set(regions["nonmarker_event"]) == set(
        regions["full_event"]
    )
    assert set(regions["marker_closing"]) == {10, 13, 14, 15, 18, 19, 21}
