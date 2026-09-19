from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.analyze_realistic_niah_v4_4_5_state_retention import (
    exact_sign_flip,
    holm_adjust,
)
from scripts.run_realistic_niah_v4_4_5_state_retention import (
    centroid_retention_metrics,
    layer_retention_rows,
    relative_change_metrics,
)


def test_relative_change_metrics_are_full_vector_and_occurrence_resolved() -> None:
    clean = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    changed = torch.tensor([[0.0, 4.0], [0.0, 1.0]])
    result = relative_change_metrics(clean, changed)
    expected_rms = torch.sqrt(torch.tensor((9.0 + 16.0) / 4.0)).item()
    assert result["raw_rms"] == pytest.approx(expected_rms)
    assert result["relative_rms"] > 0
    assert result["per_occurrence_l2"].shape == (2,)
    assert result["per_occurrence_cosine_distance"].shape == (2,)


def test_centroid_retention_detects_correct_and_swapped_states() -> None:
    centroids = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    radii = torch.tensor([1.0, 1.0])
    clean = centroids.clone()
    swapped = centroids.flip(0)
    clean_result = centroid_retention_metrics(clean, centroids, radii)
    swapped_result = centroid_retention_metrics(swapped, centroids, radii)
    assert clean_result["accuracy"] == 1.0
    assert clean_result["mad"] == 0.0
    assert clean_result["mean_correct_distance"] == 0.0
    assert swapped_result["accuracy"] == 0.0
    assert swapped_result["mad"] == 1.0
    assert swapped_result["mean_correct_distance"] == 10.0
    assert clean_result["mean_margin"] > 0
    assert swapped_result["mean_margin"] < 0


def test_layer_rows_make_needle_minus_matched_control_specificity_positive() -> None:
    centroids = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    radii = torch.tensor([1.0, 1.0])
    clean = centroids.clone()
    needle = centroids.flip(0)
    ordinary = centroids + torch.tensor([[0.1, 0.0], [0.1, 0.0]])
    row, occurrences = layer_retention_rows(
        model_label="Qwen3-8B",
        seed=1254,
        layer=3,
        pooling="span_end",
        clean=clean,
        needle=needle,
        ordinary=ordinary,
        centroids=centroids,
        radii=radii,
    )
    assert row["correct_distance_specificity"] > 0
    assert row["margin_damage_specificity"] > 0
    assert row["accuracy_damage_specificity"] > 0
    assert row["relative_rms_specificity"] > 0
    assert len(occurrences) == 2
    assert {item["running_index"] for item in occurrences} == {1, 2}


def test_exact_sign_flip_and_holm_are_deterministic() -> None:
    assert exact_sign_flip([1.0, 1.0]) == 0.5
    assert exact_sign_flip([1.0, -1.0]) == 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_preregistered_metric_families_and_split_are_frozen() -> None:
    config = json.loads(
        Path("configs/realistic_niah_v4_4_5_state_retention.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(config["discovery_seeds"]).isdisjoint(config["confirmation_seeds"])
    assert len(config["discovery_seeds"]) == 20
    assert len(config["confirmation_seeds"]) == 10
    assert config["gold_count"] == 10
    assert config["pooling_sites"] == ["span_end", "span_mean"]
    assert len(config["retention_family"]) == 6
    assert len(config["deformation_family"]) == 4
    assert config["main_display_priority"][0] == config["primary_metric"]
