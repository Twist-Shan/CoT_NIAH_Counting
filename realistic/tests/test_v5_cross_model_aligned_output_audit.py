from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_realistic_niah_v5_cross_model_sample_aligned_outputs import (
    _compare,
    _targeted_counter_write,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_compare_is_multiset_not_set_equality() -> None:
    with pytest.raises(ValueError, match="multiset mismatch"):
        _compare(
            "multiplicity",
            {
                "Qwen3-8B": [("cell",), ("cell",)],
                "Gemma4-E4B": [("cell",)],
            },
            projection="test",
        )


def test_counter_write_projects_only_model_specific_grammar_timing(
    tmp_path: Path,
) -> None:
    for model, timing in zip(
        MODELS, ("rank_after_city", "rank_before_city"), strict=True
    ):
        for phase, seed in (("discovery", 1234), ("confirmation", 1254)):
            _write_jsonl(
                tmp_path
                / model
                / "targeted_counter_write"
                / phase
                / "shards"
                / "worker_00.jsonl",
                [
                    {
                        "status": "ok",
                        "seed": seed,
                        "gold_count": 6,
                        "targeted_from_occurrence": 5,
                        "targeted_to_occurrence": 6,
                        "condition": condition,
                        "grammar_timing_stratum": timing,
                    }
                    for condition in ("clean", "selected_mask")
                ],
            )

    result = _targeted_counter_write(tmp_path)

    assert result["status"] == "PASS"
    assert result["realized_cells_per_model"] == 4
    assert result["model_specific_grammar_timing_counts"]["Qwen3-8B"][
        "discovery"
    ] == {"rank_after_city": 2}
    assert result["model_specific_grammar_timing_counts"]["Gemma4-E4B"][
        "discovery"
    ] == {"rank_before_city": 2}


def test_counter_write_still_rejects_semantic_transition_mismatch(
    tmp_path: Path,
) -> None:
    for model, to_occurrence in zip(MODELS, (6, 7), strict=True):
        for phase, seed in (("discovery", 1234), ("confirmation", 1254)):
            _write_jsonl(
                tmp_path
                / model
                / "targeted_counter_write"
                / phase
                / "shards"
                / "worker_00.jsonl",
                [
                    {
                        "status": "ok",
                        "seed": seed,
                        "gold_count": 7,
                        "targeted_from_occurrence": to_occurrence - 1,
                        "targeted_to_occurrence": to_occurrence,
                        "condition": "clean",
                        "grammar_timing_stratum": "rank_after_city",
                    }
                ],
            )

    with pytest.raises(ValueError, match="multiset mismatch"):
        _targeted_counter_write(tmp_path)
