from __future__ import annotations

import pytest

from scripts.analyze_realistic_niah_v5_event_cache_splice import analyze_batch


def _row(seed: int, direction: str, region: str, progress: float) -> dict:
    donor, receiver = direction.split("_to_")
    return {
        "seed": seed,
        "condition": "cache_splice",
        "read_layer": 24,
        "donor_variant": donor,
        "receiver_variant": receiver,
        "region": region,
        "components": ["key", "value"],
        "donor_axis_progress": progress,
        "splice_audit": {
            "changed_elements": 0 if region in {"b5", "preceding_event_width"} else 1
        },
    }


def test_analyze_batch_applies_bidirectional_frozen_criteria() -> None:
    directions = ("valid_to_invalid", "invalid_to_valid")
    rows = []
    for seed in range(1, 8):
        for direction in directions:
            for region, progress in {
                "marker": 0.9,
                "closing": 0.05,
                "payload": 0.1,
                "event": 1.0,
                "b5": 0.0,
                "preceding_event_width": 0.0,
            }.items():
                rows.append(_row(seed, direction, region, progress))
    plan = {
        "primary_read_layer": 24,
        "primary_intervention": {
            "region": "marker",
            "components": ["key", "value"],
            "directions": list(directions),
            "alpha": 0.05,
            "minimum_effect_size": 0.5,
        },
        "specificity_contrasts": [
            {
                "name": "marker_minus_closing",
                "minimum_effect_size": 0.25,
                "alpha": 0.05,
            },
            {
                "name": "marker_minus_payload",
                "minimum_effect_size": 0.25,
                "alpha": 0.05,
            },
        ],
        "implementation_controls": {
            "identity_regions": ["b5", "preceding_event_width"]
        },
    }

    result = analyze_batch(rows, plan, expected_seeds=range(1, 8))

    assert result["marker_ledger_supported"]
    assert all(result["criteria"].values())
    assert result["primary_marker_axis_progress"]["mean"] == pytest.approx(0.9)
