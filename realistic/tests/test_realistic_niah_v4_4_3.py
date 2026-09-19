from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from realistic_niah_v4.prompts import PromptEncoding, TokenSpan
from realistic_niah_v4_4_3.geometry import (
    fit_count_direction,
    ov_map_direction,
    project_actual_value_states,
    query_to_kv_head,
    resolve_value_source_layer,
    select_candidate_and_control_heads,
)
from realistic_niah_v4_4_3.interventions import (
    align_attention_row_to_receiver,
    candidate_logit_cache_distribution_diagnostics,
    candidate_logit_cache_deltas,
    scramble_attention_row,
)
from realistic_niah_v4_4_3.io import initialize_isolated_run, validate_filestream_isolation
from realistic_niah_v4_4_3.spec import V443Config


def test_frozen_config_partitions_counts_and_seeds() -> None:
    config = V443Config()
    config.validate()
    assert set(config.fit_counts) | set(config.heldout_counts) == set(range(1, 11))
    assert not set(config.discovery_seeds) & set(config.screen_seeds)
    assert not set(config.screen_seeds) & set(config.confirmation_seeds)
    assert config.directed_patch_pairs == (
        (1, 6),
        (6, 1),
        (3, 8),
        (8, 3),
        (5, 10),
        (10, 5),
    )
    assert not config.write_raw_attention_rows
    assert not config.write_full_hidden_states


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown V4.4.3"):
        V443Config.from_mapping({"not_registered": True})


def test_filestream_namespace_must_be_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "runs" / "v4_source"
    source.mkdir(parents=True)
    namespace = tmp_path / "runs" / "v4_4_3_ov_causal"
    run = namespace / "unit-test-run"
    resolved = validate_filestream_isolation(
        source_run_root=source,
        output_namespace_root=namespace,
        run_root=run,
    )
    assert resolved[2] == run.resolve()
    with pytest.raises(ValueError, match="Overlapping"):
        validate_filestream_isolation(
            source_run_root=source,
            output_namespace_root=source / "v4_4_3_ov_causal",
            run_root=source / "v4_4_3_ov_causal" / "bad-run",
        )


def test_identical_tuple_config_can_resume_json_artifact(tmp_path: Path) -> None:
    source = tmp_path / "runs" / "source"
    source.mkdir(parents=True)
    namespace = tmp_path / "runs" / "v4_4_3_ov_causal"
    run = namespace / "resume-test"
    config = V443Config(mapping_null_repetitions=100)
    initialize_isolated_run(
        source_run_root=source,
        output_namespace_root=namespace,
        run_root=run,
        config=config,
        resume=False,
    )
    resumed = initialize_isolated_run(
        source_run_root=source,
        output_namespace_root=namespace,
        run_root=run,
        config=config,
        resume=True,
    )
    assert resumed == run.resolve()


def test_within_seed_direction_ignores_large_seed_offsets() -> None:
    generator = torch.Generator().manual_seed(443)
    seeds, counts, hidden = 7, 10, 13
    true = torch.randn(hidden, generator=generator)
    true = true / torch.linalg.vector_norm(true)
    seed_offsets = 100 * torch.randn(seeds, 1, hidden, generator=generator)
    count_axis = torch.arange(1, counts + 1, dtype=torch.float32)[None, :, None]
    states = seed_offsets + count_axis * true
    fit = fit_count_direction(states, selected_counts=(1, 3, 5, 7, 9))
    assert torch.dot(fit.unit, true) > 0.9999
    assert fit.projection_count_correlation > 0.9999


def test_gqa_query_to_kv_mapping() -> None:
    assert [
        query_to_kv_head(query_head=head, query_heads=8, kv_heads=2)
        for head in range(8)
    ] == [0, 0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="integral GQA"):
        query_to_kv_head(query_head=0, query_heads=7, kv_heads=2)


def test_ov_map_uses_query_specific_o_and_grouped_v() -> None:
    hidden, query_heads, kv_heads, head_dim = 6, 4, 2, 2

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)

    attention = Attention()
    output = nn.Linear(query_heads * head_dim, hidden, bias=False)
    with torch.no_grad():
        attention.v_proj.weight.zero_()
        attention.v_proj.weight[2:4, :2] = torch.eye(2)
        output.weight.zero_()
        output.weight[:2, 4:6] = torch.eye(2)
    direction = torch.tensor([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    mapped, kv_head = ov_map_direction(
        attention,
        output,
        query_head=2,
        query_heads=query_heads,
        head_dim=head_dim,
        direction=direction,
    )
    assert kv_head == 1
    assert torch.equal(mapped, torch.tensor([3.0, 4.0, 0.0, 0.0, 0.0, 0.0]))


def test_shared_kv_layer_resolves_real_value_provider_and_value_norm() -> None:
    class UnitNorm(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)

    class ProviderAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v_proj = nn.Linear(4, 4, bias=False)
            self.v_norm = UnitNorm()
            self.store_full_length_kv = True
            with torch.no_grad():
                self.v_proj.weight.copy_(torch.eye(4))

    class SharedAttention(nn.Module):
        is_kv_shared_layer = True

    provider = ProviderAttention()
    adapter = SimpleNamespace(
        num_layers=3,
        layer_types=("full_attention", "sliding_attention", "sliding_attention"),
        attentions=(nn.Identity(), provider, SharedAttention()),
    )
    assert resolve_value_source_layer(adapter, 2) == 1
    states = torch.tensor([[[3.0, 4.0, 5.0, 12.0], [8.0, 15.0, 7.0, 24.0]]])
    values = project_actual_value_states(provider, states, head_dim=2)
    assert values.shape == (1, 2, 2, 2)
    assert torch.allclose(
        torch.linalg.vector_norm(values, dim=-1),
        torch.ones(1, 2, 2),
        atol=1e-6,
    )


def test_cache_audit_ignores_only_a_common_candidate_logit_shift() -> None:
    full = torch.tensor([0.0, 1.0, 2.0, 3.0])
    cached = torch.tensor([0.0, 6.0, 7.0, 8.0])
    raw, centered = candidate_logit_cache_deltas(full, cached, (1, 2, 3))
    assert raw == pytest.approx(5.0)
    assert centered == pytest.approx(0.0)
    cached[3] += 0.25
    _raw, centered = candidate_logit_cache_deltas(full, cached, (1, 2, 3))
    assert centered > 0


def test_cache_distribution_diagnostic_reports_argmax_and_total_variation() -> None:
    full = torch.tensor([0.0, 3.0, 1.0, -1.0])
    cached = torch.tensor([0.0, 2.5, 1.1, -1.0])
    agreement, total_variation = candidate_logit_cache_distribution_diagnostics(
        full, cached, (1, 2, 3, 1)
    )
    assert agreement
    assert 0 < total_variation < 0.2
    cached[2] = 4.0
    agreement, _ = candidate_logit_cache_distribution_diagnostics(
        full, cached, (1, 2, 3)
    )
    assert not agreement


def _encoding(lengths: tuple[int, int], *, query_position: int) -> PromptEncoding:
    first_start = 2
    first_end = first_start + lengths[0]
    second_start = first_end + 3
    second_end = second_start + lengths[1]
    spans = (
        TokenSpan(1, first_start, first_end, True, "needle", lengths[0], lengths[0]),
        TokenSpan(2, second_start, second_end, True, "needle", lengths[1], lengths[1]),
    )
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
        input_ids=tuple(range(query_position + 1)),
        attention_mask=(1,) * (query_position + 1),
        query_position=query_position,
        slot_spans=spans,
        needle_spans=spans,
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )


def test_semantic_attention_alignment_preserves_mass_with_width_changes() -> None:
    donor = _encoding((2, 3), query_position=14)
    receiver = _encoding((4, 2), query_position=15)
    row = torch.zeros(15)
    row[donor.slot_spans[0].start : donor.slot_spans[0].end] = 0.2
    row[-1] = 0.6
    aligned = align_attention_row_to_receiver(
        row,
        donor_key_start=0,
        donor_encoding=donor,
        receiver_key_start=0,
        receiver_key_length=16,
        receiver_encoding=receiver,
    )
    assert aligned.shape == (16,)
    assert torch.allclose(aligned.sum(), row.sum(), atol=1e-6)
    assert aligned[-1] > 0.25


def test_position_scramble_preserves_mass_and_self_key() -> None:
    row = torch.arange(1, 21, dtype=torch.float32)
    scrambled = scramble_attention_row(row, fraction=0.37)
    assert torch.equal(scrambled[-1:], row[-1:])
    assert torch.allclose(scrambled.sum(), row.sum())
    assert not torch.equal(scrambled[:-1], row[:-1])


def test_head_selection_uses_fit_score_not_heldout_score() -> None:
    config = V443Config(
        target_output_layers_qwen=(28,),
        qwen_sentinel_heads=(),
        heads_per_layer=1,
    )
    scores = pd.DataFrame(
        [
            {
                "model_label": "Qwen3-8B",
                "layer": 28,
                "head": 0,
                "fit_mapping_cosine": 0.9,
                "heldout_count_mapping_cosine": -0.9,
                "fit_mapped_norm": 1.0,
            },
            {
                "model_label": "Qwen3-8B",
                "layer": 28,
                "head": 1,
                "fit_mapping_cosine": 0.8,
                "heldout_count_mapping_cosine": 0.99,
                "fit_mapped_norm": 1.1,
            },
            {
                "model_label": "Qwen3-8B",
                "layer": 28,
                "head": 2,
                "fit_mapping_cosine": 0.0,
                "heldout_count_mapping_cosine": 0.0,
                "fit_mapped_norm": 0.9,
            },
        ]
    )
    selection = select_candidate_and_control_heads(
        scores, model_label="Qwen3-8B", config=config
    )
    assert selection["candidate_heads"][0]["head"] == 0
    assert not selection["heldout_count_metric_used_for_selection"]


def test_registered_json_matches_dataclass(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "realistic_niah_v4_4_3.json"
    config = V443Config.from_json(source)
    assert config.strict_zo_equivalence_tolerance == 0.05
    assert config.target_layers("Gemma4-E4B") == (36, 37, 38)
