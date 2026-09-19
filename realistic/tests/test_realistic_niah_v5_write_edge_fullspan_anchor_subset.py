from __future__ import annotations

import pytest

from scripts.build_realistic_niah_v5_write_edge_midcount_anchor_subset import (
    filter_one_to_one_band,
)
from scripts.build_realistic_niah_v5_write_edge_fullspan_anchor_subset import (
    select_geometry_eligible_anchor_subset,
)


def _row(seed: int, count: int, request_id: str) -> dict[str, object]:
    return {
        "request_id": request_id,
        "seed": seed,
        "gold_count": count,
        "from_occurrence": count - 1,
        "to_occurrence": count,
        "anchor_equivalence_id": f"{count - 1}->{count}@route-q10",
    }


def test_selector_falls_back_to_highest_geometry_eligible_count() -> None:
    rows = [
        _row(1, 10, "seed1-count10-ineligible"),
        _row(1, 9, "seed1-count9-eligible"),
        _row(1, 8, "seed1-count8-eligible"),
        _row(2, 10, "seed2-count10-eligible"),
    ]
    selected = select_geometry_eligible_anchor_subset(
        rows,
        eligible_request_ids={
            "seed1-count9-eligible",
            "seed1-count8-eligible",
            "seed2-count10-eligible",
        },
        seeds=(1, 2),
    )
    assert [(row["seed"], row["gold_count"]) for row in selected] == [
        (1, 9),
        (2, 10),
    ]
    assert all(row["write_edge_geometry_eligible"] is True for row in selected)
    assert all(row["write_edge_outcome_blind"] is True for row in selected)


def test_selector_rejects_seed_without_geometry_eligible_anchor() -> None:
    with pytest.raises(ValueError, match="eligible anchor for seeds \\[1\\]"):
        select_geometry_eligible_anchor_subset(
            [_row(1, 10, "ineligible")],
            eligible_request_ids=set(),
            seeds=(1,),
        )


def test_selector_still_rejects_selection_rank() -> None:
    row = _row(1, 10, "eligible")
    row["selection_rank"] = 0
    with pytest.raises(ValueError, match="selection_rank"):
        select_geometry_eligible_anchor_subset(
            [row], eligible_request_ids={"eligible"}, seeds=(1,)
        )


def test_selector_accepts_a_pre_filtered_fixed_count_band() -> None:
    rows = [
        _row(1, 8, "seed1-count8"),
        _row(1, 7, "seed1-count7"),
        _row(2, 6, "seed2-count6"),
    ]
    selected = select_geometry_eligible_anchor_subset(
        rows,
        eligible_request_ids={row["request_id"] for row in rows},
        seeds=(1, 2),
    )
    assert [(row["seed"], row["gold_count"]) for row in selected] == [
        (1, 8),
        (2, 6),
    ]


def test_midcount_compiler_filters_parser_cohort_before_selection() -> None:
    rows = [_row(1, 8, "excluded"), _row(1, 7, "eligible")]
    generations = {
        "excluded": {"exclude": True},
        "eligible": {"exclude": False},
    }
    filtered = filter_one_to_one_band(
        rows,
        generations,
        exclusion_fn=lambda row, cohort: (
            "not_one_to_one" if row["exclude"] else None
        ),
    )
    assert [row["request_id"] for row in filtered] == ["eligible"]
