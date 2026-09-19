from __future__ import annotations

import torch

from scripts.analyze_realistic_niah_v4_4_5_span_state_deformation import exact_sign_flip
from scripts.run_realistic_niah_v4_4_5_span_state_deformation import (
    deformation_metrics,
    select_positions,
)


def test_deformation_metrics_and_specificity_components() -> None:
    clean = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    corrupt = torch.tensor([[0.0, 4.0], [0.0, 1.0]])
    result = deformation_metrics(clean, corrupt)
    expected_rms = torch.sqrt(torch.tensor((9.0 + 16.0) / 4.0)).item()
    assert abs(result["raw_rms_change"] - expected_rms) < 1e-6
    assert result["relative_rms_change"] > 0
    assert result["mean_cosine_distance"] >= 0


def test_select_positions_preserves_requested_order() -> None:
    states = torch.tensor([[10.0], [20.0], [30.0]])
    selected = select_positions(states, (7, 2, 9), (9, 7))
    assert selected.tolist() == [[30.0], [10.0]]


def test_exact_sign_flip_is_two_sided_and_exact() -> None:
    assert exact_sign_flip([1.0, 1.0]) == 0.5
    assert exact_sign_flip([1.0, -1.0]) == 1.0
