from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from scripts.run_realistic_niah_v5_event_commit_movie import (
    build_event_movie_geometry,
    summarize_event_movie,
)
from scripts.run_realistic_niah_v5_list_event_edit_scan import (
    build_list_event_variants,
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


def _variant() -> tuple[dict[str, object], SimpleNamespace, dict[int, int]]:
    encoding = _Encoding(
        input_ids=tuple(range(31)),
        attention_mask=(1,) * 31,
        query_position=30,
        prompt_token_count=2,
    )
    neutral = _Encoding(
        input_ids=tuple(range(100, 131)),
        attention_mask=(1,) * 31,
        query_position=30,
        prompt_token_count=2,
    )
    registry = SimpleNamespace(
        trace_items=((2, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 25), (26, 29)),
        trace_markers=((2, 3), (6, 7), (10, 11), (14, 15), (18, 19), (22, 23), (26, 27)),
    )
    boundaries = {1: 5, 2: 9, 3: 13, 4: 17, 5: 21, 6: 25, 7: 29}
    variants = {
        row["event_variant"]: row
        for row in build_list_event_variants(
            encoding,
            neutral,
            registry,
            receiver=5,
            current_boundary=21,
            target_boundary=25,
            insert_source_occurrence=4,
            delete_occurrence=3,
        )
    }
    return variants["insert_valid_item"], registry, boundaries


def test_event_movie_geometry_tracks_insert_and_target_commit_boundaries() -> None:
    variant, registry, boundaries = _variant()

    geometry = build_event_movie_geometry(
        variant,
        registry,
        boundaries,
        receiver=5,
        insert_source_occurrence=4,
    )

    assert geometry["path_positions"] == tuple(range(21, 30))
    assert geometry["landmarks"]["inserted_marker_end"] == 22
    assert geometry["landmarks"]["inserted_event_boundary"] == 25
    assert geometry["landmarks"]["target_marker_end"] == 26
    assert geometry["landmarks"]["target_boundary"] == 29
    assert geometry["roles"][25] == "inserted_event_boundary"
    assert geometry["roles"][29] == "target_boundary"


def _movie_row(
    *, seed: int, landmark: str, condition: str, donor: int | None, prediction: int, soft: float
) -> dict[str, object]:
    return {
        "seed": seed,
        "event_variant": "insert_valid_item",
        "read_layer": 15,
        "landmarks": [landmark],
        "condition": condition,
        "donor": donor,
        "probe_prediction": prediction,
        "probe_softmax_expected_count": soft,
    }


def test_event_movie_summary_measures_donor_decay_relative_to_current() -> None:
    rows = [
        _movie_row(
            seed=1,
            landmark="target_boundary",
            condition="clean",
            donor=None,
            prediction=7,
            soft=6.8,
        ),
        _movie_row(
            seed=1,
            landmark="current_boundary",
            condition="donor_clamp",
            donor=4,
            prediction=4,
            soft=4.0,
        ),
        _movie_row(
            seed=1,
            landmark="current_boundary",
            condition="donor_clamp",
            donor=6,
            prediction=6,
            soft=6.0,
        ),
        _movie_row(
            seed=1,
            landmark="target_boundary",
            condition="donor_clamp",
            donor=4,
            prediction=7,
            soft=6.7,
        ),
        _movie_row(
            seed=1,
            landmark="target_boundary",
            condition="donor_clamp",
            donor=6,
            prediction=7,
            soft=6.8,
        ),
    ]

    summary = summarize_event_movie(rows)
    target = next(
        cell
        for cell in summary["valid_item_donor_pair_cells"]
        if cell["landmark"] == "target_boundary"
    )

    assert target["donor_invariant_count"] == 1
    assert target["recurrent_separation_2_count"] == 0
    assert abs(target["mean_within_seed_retention"] - 0.05) < 1e-12
