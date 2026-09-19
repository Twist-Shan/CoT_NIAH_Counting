from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from realistic_niah_v4_4_3.geometry import v_map_direction
import realistic_niah_v4_4_3.set_interventions as set_interventions
from realistic_niah_v4_4_3.set_analysis import _benjamini_hochberg, _exact_sign_flip
from realistic_niah_v4_4_3.set_geometry import (
    select_candidate_and_control_sets,
    set_reachable_answer_direction,
)
from realistic_niah_v4_4_3.set_interventions import (
    CausalOutput,
    natural_ov_removal_deltas,
    run_with_set_z_deltas,
)
from realistic_niah_v4_4_3.set_spec import V443SetConfig


def _synthetic_config() -> V443SetConfig:
    return V443SetConfig(
        model_labels=("Qwen3-8B",),
        target_output_layers_qwen=(1,),
        set_sizes_qwen=(2, 4),
        set_null_samples=100,
        set_control_norm_pool=8,
        mapping_null_repetitions=100,
    )


def test_set_config_is_frozen_and_valid() -> None:
    config = _synthetic_config()
    config.validate()
    assert config.set_sizes_for("Qwen3-8B") == (2, 4)
    assert "projected_into_selected_set_output_span" in config.set_injection_boundary


def test_nested_set_selection_uses_fit_vectors_and_disjoint_controls() -> None:
    config = _synthetic_config()
    mapped_fit = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, -0.1, 0.0],
            [0.7, 0.0, 0.1],
            [-0.2, 0.9, 0.0],
            [-0.2, -0.9, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    mapped_holdout = mapped_fit.clone()
    directions = {
        1: {
            "mapped_fit_by_head": mapped_fit,
            "mapped_holdout_by_head": mapped_holdout,
            "u_answer_fit": torch.tensor([1.0, 0.0, 0.0]),
            "u_answer_holdout": torch.tensor([1.0, 0.0, 0.0]),
        }
    }
    scores = pd.DataFrame(
        {
            "model_label": ["Qwen3-8B"] * 8,
            "layer": [1] * 8,
            "head": list(range(8)),
        }
    )
    selection, summary = select_candidate_and_control_sets(
        scores,
        directions,
        model_label="Qwen3-8B",
        config=config,
    )
    k2, k4 = selection["candidate_sets"]
    assert set(k2["heads"]).issubset(set(k4["heads"]))
    for candidate in selection["candidate_sets"]:
        control = selection["matched_control_sets"][candidate["set_id"]]
        assert set(candidate["heads"]).isdisjoint(control["heads"])
        assert len(candidate["heads"]) == len(control["heads"])
    assert len(summary) == 4


def test_set_reachable_direction_is_projection_into_registered_o_span() -> None:
    projection = nn.Linear(4, 4, bias=False)
    projection.weight.data.copy_(torch.eye(4))
    adapter = SimpleNamespace(
        output_projections=[projection],
        head_dims=[2],
    )
    unit, cosine = set_reachable_answer_direction(
        adapter,
        layer=0,
        heads=(0,),
        answer_direction=torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    assert torch.allclose(unit, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert abs(cosine - 2 ** -0.5) < 1e-6


def test_natural_v_step_respects_gqa_grouping_and_count_scale() -> None:
    attention = SimpleNamespace(v_proj=nn.Linear(3, 4, bias=False))
    attention.v_proj.weight.data.copy_(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    step, kv_head = v_map_direction(
        attention,
        query_head=3,
        query_heads=4,
        head_dim=2,
        direction=torch.tensor([2.0, 3.0, 4.0]),
    )
    assert kv_head == 1
    assert torch.allclose(step, torch.tensor([8.0, 5.0]))


def test_natural_ov_removal_is_realizable_and_control_stays_in_set_span() -> None:
    projection = nn.Linear(4, 4, bias=False)
    projection.weight.data.copy_(torch.eye(4))
    adapter = SimpleNamespace(
        output_projections=[projection],
        head_dims=[2],
    )
    bundle = SimpleNamespace(
        z_by_layer={0: torch.tensor([2.0, 3.0, 4.0, 5.0])}
    )
    count_steps = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    removal, control, diagnostics = natural_ov_removal_deltas(
        adapter,
        bundle=bundle,
        layer=0,
        heads=(0, 1),
        z_count_steps=count_steps,
        orthogonal_label="unit-test",
    )
    axis = torch.tensor([1.0, 0.0, 0.0, 1.0])
    axis = axis / torch.linalg.vector_norm(axis)
    actual_output = bundle.z_by_layer[0]
    removal_output = removal.reshape(-1)
    control_output = control.reshape(-1)
    assert abs(float(torch.dot(actual_output + removal_output, axis))) < 1e-6
    assert abs(float(torch.dot(control_output, axis))) < 1e-6
    assert torch.allclose(
        torch.linalg.vector_norm(removal_output),
        torch.linalg.vector_norm(control_output),
        atol=1e-6,
    )
    assert abs(diagnostics["control_output_cosine_with_removed_axis"]) < 1e-6
    assert diagnostics["used_count_neutral_z_center"] == 0.0

    center = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    centered_removal, _control, centered_diagnostics = natural_ov_removal_deltas(
        adapter,
        bundle=bundle,
        layer=0,
        heads=(0, 1),
        z_count_steps=count_steps,
        orthogonal_label="centered-unit-test",
        z_center=center,
    )
    centered_output = actual_output + centered_removal.reshape(-1) - center.reshape(-1)
    assert abs(float(torch.dot(centered_output, axis))) < 1e-6
    assert centered_diagnostics["used_count_neutral_z_center"] == 1.0


def test_set_z_delta_is_applied_before_the_models_o_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = nn.Linear(4, 4, bias=False)
            self.o_proj.weight.data.copy_(torch.eye(4))

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.o_proj(hidden)

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.attention = TinyAttention()

        def get_input_embeddings(self) -> nn.Embedding:
            return self.embedding

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            use_cache: bool,
            logits_to_keep: int | None = None,
        ) -> SimpleNamespace:
            del attention_mask, use_cache, logits_to_keep
            hidden = torch.zeros(
                input_ids.shape[0], input_ids.shape[1], 4, device=input_ids.device
            )
            output = self.attention(hidden)
            return SimpleNamespace(logits=torch.zeros(*output.shape[:2], 8))

    def fake_score(
        _model: nn.Module, _encoding: SimpleNamespace, prefill: SimpleNamespace
    ) -> CausalOutput:
        return CausalOutput(logits=prefill.logits[0, -1], candidate_log_scores={})

    monkeypatch.setattr(set_interventions, "_score_candidate_sequences", fake_score)
    model = TinyModel()
    adapter = SimpleNamespace(
        output_projections=[model.attention.o_proj],
        attentions=[model.attention],
        head_dims=[2],
    )
    encoding = SimpleNamespace(
        input_ids=[1, 2],
        attention_mask=[1, 1],
        sequence_length=2,
        query_position=1,
    )
    result = run_with_set_z_deltas(
        model,
        adapter,
        encoding,
        layer=0,
        deltas={0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])},
    )
    assert torch.equal(result.attention_output, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_exact_sign_flip_resolution_for_five_seeds() -> None:
    assert _exact_sign_flip([1, 2, 3, 4, 5], alternative="greater") == 1 / 32


def test_benjamini_hochberg_is_monotone_in_rank_and_restores_input_order() -> None:
    adjusted = _benjamini_hochberg([0.04, 0.001, 0.03, float("nan")])
    assert adjusted[:3] == [0.04, 0.003, 0.04]
    assert math.isnan(adjusted[3])
