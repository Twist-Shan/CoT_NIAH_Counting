from __future__ import annotations

from scripts.build_realistic_niah_v5_cross_model_aligned_anchor_panel import (
    build_aligned_panel,
)


def _row(model: str, seed: int, count: int, grammar: str) -> dict[str, object]:
    return {
        "request_id": f"{model}/N{count}/seed{seed}",
        "seed": seed,
        "gold_count": count,
        "from_occurrence": count - 1,
        "to_occurrence": count,
        "target_grammar_class": grammar,
    }


def test_build_aligned_panel_uses_identical_highest_common_transition() -> None:
    qwen: list[dict[str, object]] = []
    gemma: list[dict[str, object]] = []
    for seed in range(1234, 1264):
        qwen.extend(
            [_row("q", seed, 3, "after"), _row("q", seed, 7, "after")]
        )
        gemma.extend(
            [_row("g", seed, 3, "before"), _row("g", seed, 7, "before")]
        )
    shared, qwen_panel, gemma_panel, manifest = build_aligned_panel(qwen, gemma)
    assert len(shared) == len(qwen_panel) == len(gemma_panel) == 30
    assert [row["seed"] for row in shared] == list(range(1234, 1264))
    assert {row["gold_count"] for row in shared} == {7}
    q_keys = [
        (row["seed"], row["gold_count"], row["from_occurrence"], row["to_occurrence"])
        for row in qwen_panel
    ]
    g_keys = [
        (row["seed"], row["gold_count"], row["from_occurrence"], row["to_occurrence"])
        for row in gemma_panel
    ]
    assert q_keys == g_keys
    assert manifest["exact_cross_model_key_equality"] is True
    assert manifest["development_sample_count"] == 20
    assert manifest["confirmation_sample_count"] == 10


def test_build_aligned_panel_rejects_selection_rank() -> None:
    qwen = [_row("q", seed, 3, "after") for seed in range(1234, 1264)]
    gemma = [_row("g", seed, 3, "before") for seed in range(1234, 1264)]
    qwen[0]["selection_rank"] = 1
    try:
        build_aligned_panel(qwen, gemma)
    except ValueError as exc:
        assert "selection_rank" in str(exc)
    else:
        raise AssertionError("selection_rank must fail closed")
