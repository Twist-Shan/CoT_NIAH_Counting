from __future__ import annotations

from scripts.build_realistic_niah_v5_grammar_span_anchor_panel import build_panel


def test_grammar_span_anchor_panel_is_fixed_20_plus_10_and_balanced() -> None:
    rows = []
    for seed in range(1234, 1264):
        for grammar in (
            "adjacent_rank_after_city",
            "adjacent_rank_before_city",
        ):
            rows.append(
                {
                    "seed": seed,
                    "gold_count": 8,
                    "to_occurrence": 8,
                    "request_id": f"request-{seed}-{grammar}",
                    "target_grammar_class": grammar,
                }
            )
    panel, manifest = build_panel(rows)
    assert len(panel) == 30
    assert [row["seed"] for row in panel] == list(range(1234, 1264))
    assert manifest["development_seed_count"] == 20
    assert manifest["confirmation_seed_count"] == 10
    assert manifest["timing_counts_by_phase"] == {
        "development": {"rank_after_city": 10, "rank_before_city": 10},
        "confirmation": {"rank_after_city": 5, "rank_before_city": 5},
    }
    assert all(row["grammar_span_outcome_blind"] for row in panel)
    assert not any(row["grammar_span_selection_rank_used"] for row in panel)
