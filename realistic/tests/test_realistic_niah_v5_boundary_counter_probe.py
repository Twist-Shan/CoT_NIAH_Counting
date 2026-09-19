import numpy as np
import torch
from types import SimpleNamespace

from realistic_niah_v5.boundary_counter_probe import (
    boundary_value_edge_write,
    count_probe_subspace,
    count_prediction_metrics,
    count_probe_predictions,
    fit_dual_ridge_count_probe,
    leave_one_seed_out_probe_metrics,
    norm_matched_orthogonal_replacement,
    projected_donor_replacement,
)


def _separable_panel(seeds: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    labels = []
    seed_ids = []
    for seed in range(seeds):
        for count in range(1, 11):
            vector = np.zeros(12, dtype=np.float32)
            vector[count - 1] = 4.0
            vector[10] = seed * 0.1
            vector[11] = 1.0
            rows.append(vector)
            labels.append(count)
            seed_ids.append(seed)
    return np.stack(rows), np.asarray(labels), np.asarray(seed_ids)


def test_dual_ridge_probe_decodes_held_out_seed() -> None:
    x, y, seeds = _separable_panel()
    train = seeds != 3
    test = ~train
    probe = fit_dual_ridge_count_probe(x[train], y[train])
    predictions = count_probe_predictions(probe, x[test])
    assert count_prediction_metrics(y[test], predictions)["exact_accuracy"] == 1.0


def test_loso_probe_keeps_seeds_grouped() -> None:
    x, y, seeds = _separable_panel()
    result = leave_one_seed_out_probe_metrics(x, y, seeds)
    assert result["exact_accuracy"] == 1.0
    assert len(result["per_seed"]) == 4


def test_count_probe_subspace_contains_only_discriminative_directions() -> None:
    x, y, _seeds = _separable_panel()
    probe = fit_dual_ridge_count_probe(x, y)
    basis = count_probe_subspace(probe)
    assert basis.shape == (12, 9)
    np.testing.assert_allclose(basis.T @ basis, np.eye(9), atol=1e-6)
    shared = np.asarray(probe["weights"]).mean(axis=1)
    np.testing.assert_allclose(shared @ basis, np.zeros(9), atol=1e-5)


def test_projected_swap_and_orthogonal_control_are_norm_matched() -> None:
    basis = np.eye(6, 2, dtype=np.float32)
    receiver = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    donor = np.array([4, 6, 9, 8, 7, 6], dtype=np.float32)
    replacement, projected = projected_donor_replacement(receiver, donor, basis)
    np.testing.assert_allclose(replacement[:2], donor[:2])
    np.testing.assert_allclose(replacement[2:], receiver[2:])
    control, random_delta = norm_matched_orthogonal_replacement(
        receiver, projected, basis, seed=1234
    )
    np.testing.assert_allclose(np.linalg.norm(random_delta), np.linalg.norm(projected))
    np.testing.assert_allclose(random_delta @ basis, np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(control, receiver + random_delta)


def test_boundary_value_edge_write_uses_receiver_attention_and_donor_value() -> None:
    projection = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(4))
    adapter = SimpleNamespace(
        num_heads=(2,),
        head_dims=(2,),
        output_projections=(projection,),
    )
    attention = torch.tensor(
        [[0.0, 0.0, 0.25, 0.0], [0.0, 0.0, 0.50, 0.0]],
        dtype=torch.float32,
    )
    # One KV head is shared by the two query heads.
    value = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    write, audit = boundary_value_edge_write(
        adapter,
        layer=0,
        attention_row=attention,
        key_start=0,
        source_position=2,
        source_value=value,
    )
    torch.testing.assert_close(write, torch.tensor([0.5, 1.0, 1.0, 2.0]))
    assert audit["source_attention_mass_sum"] == 0.75
    assert audit["query_heads"] == 2
    assert audit["kv_heads"] == 1
