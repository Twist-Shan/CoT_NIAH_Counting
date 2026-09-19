from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from realistic_niah_v5.unified_carrier_transition import (
    interpolated_boundary_targets,
    projected_donor_delta,
    summarize_carrier_trials,
)
from scripts.run_realistic_niah_v5_unified_carrier_transition import (
    _candidate_score_payload,
    _score_native_item_candidates_no_cache,
    _transition_candidates,
    fit_residual_bases,
)


@dataclass(frozen=True)
class _FakeEncoding:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


class _FakeCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 1)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_attentions: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, output_attentions
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 32), -10.0, device=input_ids.device)
        preferred = (input_ids + 1) % 32
        logits.scatter_(2, preferred.unsqueeze(-1), 10.0)
        return SimpleNamespace(logits=logits[:, -int(logits_to_keep) :])


def test_projected_donor_delta_removes_noncount_component() -> None:
    receiver = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    donor = np.asarray([3.0, 20.0, 3.0], dtype=np.float32)
    basis = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)
    assert projected_donor_delta(receiver, donor, basis).tolist() == [2.0, 0.0, 0.0]


def test_interpolated_targets_support_full_and_projected_deltas() -> None:
    receiver = {14: np.asarray([1.0, 2.0], dtype=np.float32)}
    donor = {14: np.asarray([5.0, 10.0], dtype=np.float32)}
    full, full_delta = interpolated_boundary_targets(
        receiver, donor, scale=0.5
    )
    assert full[14].tolist() == [3.0, 6.0]
    assert full_delta[14].tolist() == [2.0, 4.0]

    basis = {14: np.asarray([[1.0], [0.0]], dtype=np.float32)}
    projected, projected_delta = interpolated_boundary_targets(
        receiver, donor, scale=0.5, bases=basis
    )
    assert projected[14].tolist() == [3.0, 2.0]
    assert projected_delta[14].tolist() == [2.0, 0.0]


def test_fit_residual_bases_uses_only_requested_training_rows() -> None:
    panel = np.zeros((3, 1, 10, 4), dtype=np.float32)
    panel[:, 0, :, 0] = np.arange(1, 11, dtype=np.float32)
    panel[2, 0, :, 1] = 1000.0 * np.arange(1, 11, dtype=np.float32)
    bases = fit_residual_bases(
        panel,
        layers=(14,),
        train_indices=(0, 1),
        alpha=1.0,
    )
    basis = bases[14]
    assert basis.shape[0] == 4
    assert abs(float(basis[0, 0])) > 0.99
    assert abs(float(basis[1, 0])) < 1e-5


def test_transition_candidates_share_receiver_separator() -> None:
    encoding = SimpleNamespace(input_ids=tuple(range(100)))
    registry = SimpleNamespace(
        trace_items=tuple((index * 5 + 1, index * 5 + 4) for index in range(10))
    )
    candidates = _transition_candidates(
        encoding, registry, receiver_occurrence=5
    )
    # Receiver item 5 ends at 24 and item 6 starts at 26, so every candidate
    # receives the same two-token separator (24, 25).
    assert candidates[1][:2] == (24, 25)
    assert candidates[6][:2] == (24, 25)
    assert len(candidates) == 10


def test_candidate_score_payload_preserves_sum_and_length_normalized_argmax() -> None:
    candidates = {index: (index,) for index in range(1, 11)}
    candidates[2] = (2, 22)
    scores = [-20.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0]
    payload = _candidate_score_payload(candidates, target=2, sum_scores=scores)
    assert payload["predicted_occurrence_sum_logprob"] == 2
    assert payload["predicted_occurrence_mean_logprob"] == 2
    assert payload["sum_logprob_scores"] == scores
    assert payload["mean_logprob_scores"][1] == pytest.approx(-1.5)
    assert payload["candidate_token_counts"][1] == 2


def test_no_cache_candidate_scoring_aligns_each_token_with_previous_logit() -> None:
    prefix = _FakeEncoding(input_ids=(0,), attention_mask=(1,))
    candidates = {
        occurrence: (occurrence, occurrence + 1) for occurrence in range(1, 11)
    }
    scored, audit = _score_native_item_candidates_no_cache(
        _FakeCausalModel(),
        None,
        prefix,
        candidates,
        receiver_successor=1,
    )
    assert scored["predicted_occurrence_sum_logprob"] == 1
    assert scored["predicted_occurrence_mean_logprob"] == 1
    assert audit["candidate_forward_count"] == 10
    assert audit["sequence_lengths"] == [2] * 10


def test_carrier_summary_uses_realized_first_stage_for_retention() -> None:
    rows = []
    for dose in (-1, 1):
        rows.append(
            {
                "carrier": "count",
                "current_shift": 2.0 * dose,
                "next_shift": 1.0 * dose,
                "current_exact": True,
                "next_exact": dose == 1,
                "receiver_successor_argmax_mean_logprob": True,
                "donor_successor_argmax_mean_logprob": False,
                "receiver_successor_mean_logprob_change": -0.25,
                "donor_vs_receiver_mean_logodds_change": 0.5,
            }
        )
    summary = summarize_carrier_trials(rows)["count"]
    assert summary["pooled_current_to_next_retention"] == pytest.approx(0.5)
    assert summary["current_exact"] == 2
    assert summary["next_exact"] == 1
    assert summary["receiver_successor_argmax_mean_logprob"] == 2
    assert summary["donor_successor_argmax_mean_logprob"] == 0
