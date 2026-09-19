import numpy as np
import torch

from scripts.run_realistic_niah_v5_boundary_equivariance import (
    local_count_tangent,
    quantized_delta,
    quantized_norm_matched_replacement,
    through_origin_slope,
)


def test_local_count_tangent_is_one_central_count_step_in_basis() -> None:
    panel = np.zeros((3, 10, 4), dtype=np.float32)
    for seed in range(3):
        for count in range(1, 11):
            panel[seed, count - 1] = [2.0 * count, -count, seed, 5.0]
    basis = np.eye(4, 2, dtype=np.float32)
    tangent = local_count_tangent(panel, receiver_count=5, basis=basis)
    np.testing.assert_allclose(tangent, [2.0, -1.0, 0.0, 0.0], atol=1e-6)


def test_local_count_tangent_can_be_zero_before_count_is_written() -> None:
    panel = np.zeros((3, 10, 4), dtype=np.float32)
    basis = np.eye(4, 2, dtype=np.float32)
    tangent = local_count_tangent(panel, receiver_count=5, basis=basis)
    np.testing.assert_array_equal(tangent, np.zeros(4, dtype=np.float32))


def test_through_origin_slope_recovers_retention_coefficient() -> None:
    x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    y = [-1.5, -0.75, 0.0, 0.75, 1.5]
    assert through_origin_slope(x, y) == 0.75
    assert through_origin_slope([0.0, 0.0], [1.0, -1.0]) is None


def test_quantized_control_matches_aligned_realized_norm() -> None:
    base = np.linspace(-2.0, 2.0, 4096, dtype=np.float32)
    aligned = np.sin(np.arange(4096, dtype=np.float32)) * 0.01
    _replacement, _delta, target = quantized_delta(
        base, base + aligned, dtype=torch.bfloat16
    )
    rng = np.random.default_rng(1234)
    direction = rng.standard_normal(4096).astype(np.float32)
    _control, _control_delta, realized = quantized_norm_matched_replacement(
        base,
        direction,
        target_norm=target,
        dtype=torch.bfloat16,
    )
    assert abs(realized - target) / target < 0.01
