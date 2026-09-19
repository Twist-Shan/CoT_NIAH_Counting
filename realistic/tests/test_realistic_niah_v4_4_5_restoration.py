from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from realistic_niah_v4.prompts import PromptEncoding
from realistic_niah_v4_4_3.interventions import clone_prefill_output_for_scoring
from realistic_niah_v4_4_5.restoration import (
    active_broad_metrics,
    generate_answer_completion_from_prefill,
    normalized_recovery,
)
from scripts.run_realistic_niah_v4_4_5_retrieval_subspace import (
    bank_output,
    deterministic_orthogonal_in_bank_span,
)


def test_normalized_recovery_preserves_overshoot() -> None:
    assert normalized_recovery(8.0, 3.0, 6.0) == 0.6
    assert normalized_recovery(8.0, 3.0, 9.0) == 1.2
    assert math.isnan(normalized_recovery(3.0, 3.0, 4.0))


def test_active_broad_metrics_distinguishes_coverage() -> None:
    spans = [SimpleNamespace(start=1, end=2), SimpleNamespace(start=3, end=4)]
    rows = torch.tensor(
        [
            [0.0, 0.2, 0.0, 0.2],
            [0.0, 0.4, 0.0, 0.0],
        ]
    )
    metrics = active_broad_metrics(rows, key_start=0, spans=spans)
    assert metrics[0]["needle_mass"] == pytest.approx(0.4)
    assert metrics[0]["coverage"] == pytest.approx(1.0)
    assert metrics[0]["broad_score"] == pytest.approx(0.4)
    assert metrics[1]["coverage"] == pytest.approx(0.5)
    assert metrics[1]["broad_score"] == pytest.approx(0.2)


import pytest


def test_retrieval_bank_output_and_control_share_output_span() -> None:
    projection = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                ]
            )
        )
    adapter = SimpleNamespace(
        output_projections=(projection,),
        head_dims=(2,),
        num_heads=(2,),
    )
    output = bank_output(
        adapter,
        layer=0,
        heads=(0, 1),
        z=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert torch.allclose(output, torch.tensor([1.0, 2.0, 7.0]))
    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    control = deterministic_orthogonal_in_bank_span(
        adapter,
        layer=0,
        heads=(0, 1),
        basis=basis,
        random_seed=7,
    )
    assert torch.linalg.vector_norm(control).item() == pytest.approx(1.0)
    assert torch.max(torch.abs(control @ basis.T)).item() < 1e-6


class _BranchCache:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.values = self.values.repeat_interleave(int(repeats), dim=0)


def test_candidate_scoring_cache_branch_does_not_mutate_retained_prefill() -> None:
    source = SimpleNamespace(
        logits=torch.tensor([[[0.0, 1.0, 2.0]]]),
        past_key_values=_BranchCache(torch.tensor([[1.0, 2.0]])),
        shared_kv_states={"state": torch.tensor([[3.0]])},
    )
    branch = clone_prefill_output_for_scoring(source)
    branch.past_key_values.batch_repeat_interleave(10)
    branch.shared_kv_states["state"].add_(5.0)
    assert source.past_key_values.values.shape == (1, 2)
    assert branch.past_key_values.values.shape == (10, 2)
    assert torch.equal(source.shared_kv_states["state"], torch.tensor([[3.0]]))


class _ToyTokenizer:
    eos_token_id = 3

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del clean_up_tokenization_spaces
        values = [value for value in token_ids if not (skip_special_tokens and value == 3)]
        return "".join(str(value) for value in values)


class _ToyCachedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.calls: list[dict[str, torch.Tensor | int]] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: _BranchCache,
        use_cache: bool,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        assert use_cache
        assert logits_to_keep == 1
        self.calls.append(
            {
                "token": int(input_ids.item()),
                "mask_length": int(attention_mask.shape[1]),
                "position": int(position_ids.item()),
                "cache_position": int(cache_position.item()),
            }
        )
        logits = torch.full((1, 1, 5), -10.0)
        logits[0, 0, 3] = 10.0
        past_key_values.values = torch.cat(
            (past_key_values.values, input_ids.float()), dim=1
        )
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _generation_encoding() -> PromptEncoding:
    return PromptEncoding(
        stimulus_id="unit",
        design_variant="v4.4",
        seed=1,
        split="unit",
        count=2,
        model_label="unit",
        answer_format="numeric",
        text="",
        generation_prompt="",
        input_ids=(7, 8, 9),
        attention_mask=(1, 1, 1),
        query_position=2,
        slot_spans=(),
        needle_spans=(),
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )


def test_cached_strict_generation_uses_prompt_logits_then_one_token_decode() -> None:
    model = _ToyCachedModel()
    prefill = SimpleNamespace(
        logits=torch.tensor([[[-5.0, -2.0, 8.0, -4.0, -3.0]]]),
        past_key_values=_BranchCache(torch.tensor([[7.0, 8.0, 9.0]])),
    )
    result = generate_answer_completion_from_prefill(
        model,
        _ToyTokenizer(),
        _generation_encoding(),
        prefill,
        max_new_tokens=4,
    )
    assert result["generated_token_ids"] == [2, 3]
    assert result["completion_text_raw"] == "23"
    assert result["completion_text"] == "2"
    assert result["stopped_on_eos"]
    assert model.calls == [
        {"token": 2, "mask_length": 4, "position": 3, "cache_position": 3}
    ]
