from __future__ import annotations

from scripts.build_realistic_niah_v5_write_edge_anchor_subset import (
    select_anchor_subset,
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


def test_write_edge_anchor_subset_is_highest_count_and_outcome_blind() -> None:
    rows = [
        _row(1, 8, "z"),
        _row(1, 10, "b"),
        _row(1, 10, "a"),
        _row(2, 7, "c"),
        _row(2, 9, "d"),
    ]
    selected = select_anchor_subset(rows, seeds=(1, 2))
    assert [(row["seed"], row["gold_count"], row["request_id"]) for row in selected] == [
        (1, 10, "a"),
        (2, 9, "d"),
    ]
    assert all(row["write_edge_outcome_blind"] is True for row in selected)
    assert all(row["write_edge_selection_rank_used"] is False for row in selected)


def test_write_edge_anchor_subset_rejects_selection_rank() -> None:
    row = _row(1, 10, "a")
    row["selection_rank"] = 1
    try:
        select_anchor_subset([row], seeds=(1,))
    except ValueError as exc:
        assert "selection_rank" in str(exc)
    else:
        raise AssertionError("Write-edge selector accepted selection_rank")
