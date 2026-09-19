from __future__ import annotations

import torch

from realistic_niah_v4.counter_channel_interventions import (
    norm_matched_orthogonal_delta,
    projected_donor_transform,
    removal_transform,
)


def test_torch_dynamic_transforms_and_matched_control() -> None:
    basis = torch.eye(5, dtype=torch.float32)[:, :2]
    receiver = torch.zeros((1, 1, 5), dtype=torch.float32)
    donor = torch.tensor([[2.0, -1.0, 4.0, 0.5, 3.0]])
    patched = projected_donor_transform(donor, basis)(receiver)
    torch.testing.assert_close(
        patched, torch.tensor([[[2.0, -1.0, 0.0, 0.0, 0.0]]])
    )
    removed = removal_transform(torch.zeros(5), basis)(donor.unsqueeze(0))
    torch.testing.assert_close(
        removed, torch.tensor([[[0.0, 0.0, 4.0, 0.5, 3.0]]])
    )
    projected_delta = torch.tensor([[2.0, -1.0, 0.0, 0.0, 0.0]])
    control = norm_matched_orthogonal_delta(projected_delta, basis, seed=9)
    torch.testing.assert_close(
        torch.linalg.vector_norm(control), torch.linalg.vector_norm(projected_delta)
    )
    torch.testing.assert_close(control @ basis, torch.zeros((1, 2)), atol=1e-6, rtol=0)
