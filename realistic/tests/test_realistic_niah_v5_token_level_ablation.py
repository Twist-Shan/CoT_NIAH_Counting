from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from realistic_niah_v5.token_level_ablation import (
    blank_token_states,
    build_token_blank_registry_from_spans,
    token_blank_condition,
)


_ANALYZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_realistic_niah_v5_token_level_ablation.py"
)
_ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "token_level_analysis_test_module", _ANALYZER_PATH
)
assert _ANALYZER_SPEC is not None and _ANALYZER_SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_ANALYZER_SPEC)
_ANALYZER_SPEC.loader.exec_module(_ANALYZER)

_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_realistic_niah_v5_token_level_ablation.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "token_level_runner_test_module", _RUNNER_PATH
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)


def _registry():
    return build_token_blank_registry_from_spans(
        request_id="request-1",
        query_position=90,
        sequence_length=96,
        prompt_token_count=50,
        prompt_record_spans=((3, 6), (12, 16)),
        trace_item_spans=((50, 57), (58, 65), (66, 75), (76, 91)),
        transition_occurrence=4,
    )


def test_token_blank_registry_partitions_cumulative_and_recent() -> None:
    registry = _registry()
    assert registry.positions("trace_all") == tuple(range(50, 90))
    assert registry.positions("cumulative_trace") == tuple(range(50, 76))
    assert registry.positions("recent_transition") == tuple(range(76, 90))
    assert registry.positions("early_half_trace") == tuple(range(50, 65))
    assert set(registry.positions("cumulative_trace")).isdisjoint(
        registry.positions("recent_transition")
    )


def test_targeting_factorial_includes_joint_blank_arm() -> None:
    registry = _registry()
    clean, clean_audit = token_blank_condition(registry, "clean")
    cumulative, _ = token_blank_condition(registry, "cumulative_trace_blank")
    recent, _ = token_blank_condition(registry, "recent_transition_blank")
    full, full_audit = token_blank_condition(registry, "full_trace_blank")
    assert clean == ()
    assert clean_audit["token_deletion_used"] is False
    assert set(cumulative).isdisjoint(recent)
    assert set(full) == set(cumulative) | set(recent)
    assert full_audit["blank_source_groups"] == [
        "cumulative_trace",
        "recent_transition",
    ]


def test_matched_blank_controls_are_budget_matched_and_deterministic() -> None:
    registry = _registry()
    treatment, _ = token_blank_condition(registry, "full_trace_blank")
    first, first_audit = token_blank_condition(
        registry, "full_trace_matched_control", control_repeat=1
    )
    again, _ = token_blank_condition(
        registry, "full_trace_matched_control", control_repeat=1
    )
    second, _ = token_blank_condition(
        registry, "full_trace_matched_control", control_repeat=2
    )
    assert len(first) == len(treatment)
    assert first == again
    assert first != second
    assert set(first).issubset(registry.positions("ordinary_prompt_control_pool"))
    assert first_audit["matched_control_for"] == "trace_all"


def test_early_trace_arm_is_not_applicable_at_first_transition() -> None:
    registry = build_token_blank_registry_from_spans(
        request_id="first-transition",
        query_position=60,
        sequence_length=64,
        prompt_token_count=50,
        prompt_record_spans=((3, 6),),
        trace_item_spans=((50, 61),),
        transition_occurrence=1,
    )
    assert registry.positions("cumulative_trace") == ()
    assert registry.positions("early_half_trace") == ()
    with pytest.raises(ValueError, match="not applicable"):
        token_blank_condition(registry, "cumulative_trace_blank")
    with pytest.raises(ValueError, match="not applicable"):
        token_blank_condition(registry, "early_half_trace_blank")


def test_city_pre_partial_next_item_is_not_misclassified_as_recent() -> None:
    registry = build_token_blank_registry_from_spans(
        request_id="city-pre",
        query_position=82,
        sequence_length=90,
        prompt_token_count=50,
        prompt_record_spans=((3, 6),),
        # Item 3 has begun (e.g. its marker is visible), but the registered
        # transition is still 2 -> 3, so item 2 is the recent completed event.
        trace_item_spans=((50, 59), (60, 70), (71, 88)),
        transition_occurrence=2,
    )
    assert registry.visible_trace_item_count == 2
    assert registry.positions("recent_transition") == tuple(range(60, 82))
    assert registry.positions("cumulative_trace") == tuple(range(50, 60))


class _AddOne(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.layers = nn.ModuleList([_AddOne(), _AddOne()])

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


def test_blank_token_states_zeroes_embedding_and_every_block() -> None:
    model = _TinyModel()
    adapter = SimpleNamespace(num_layers=2, layers=model.layers)
    token_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with blank_token_states(model, adapter, [1, 3]) as audit:
        hidden = model.embedding(token_ids)
        for layer in model.layers:
            hidden = layer(hidden)
    assert torch.count_nonzero(hidden[:, [1, 3], :]) == 0
    assert torch.count_nonzero(hidden[:, [0, 2, 4], :]) > 0
    assert audit["blank_embedding_hook_applications"] == 1
    assert audit["blank_layer_hook_applications"] == {"0": 1, "1": 1}


def test_factorial_analysis_uses_joint_minus_main_effects() -> None:
    rows = []
    values = {
        "clean": 1.0,
        "cumulative_trace_blank": 0.8,
        "recent_transition_blank": 0.6,
        "full_trace_blank": 0.2,
    }
    for condition, value in values.items():
        rows.append(
            {
                "experiment_id": "targeting_trace_token_blank",
                "model_label": "Qwen3-8B",
                "request_id": "r1",
                "seed": 1254,
                "condition": condition,
                "bank_target_attention_share_of_gold_mass": value,
            }
        )
    result = _ANALYZER.factorial_effects(
        pd.DataFrame(rows),
        metrics=["bank_target_attention_share_of_gold_mass"],
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row[
        "cumulative_trace_blank_effect__bank_target_attention_share_of_gold_mass"
    ] == pytest.approx(-0.2)
    assert row[
        "recent_transition_blank_effect__bank_target_attention_share_of_gold_mass"
    ] == pytest.approx(-0.4)
    assert row[
        "factorial_interaction__bank_target_attention_share_of_gold_mass"
    ] == pytest.approx(-0.2)


def test_registry_route_anchor_and_exact_localizer_join_by_transition() -> None:
    exact_localizer = {
        "from_occurrence": 9,
        "to_occurrence": 10,
        "anchor_equivalence_id": "9->10@q272",
        "anchor_role": "city_pre_d1",
    }
    routed_registry = {
        "from_occurrence": 9,
        "to_occurrence": 10,
        "anchor_equivalence_id": "9->10@route-q267",
    }
    assert _RUNNER._registry_event_matches(
        exact_localizer, routed_registry, match_mode="transition"
    )
    assert not _RUNNER._registry_event_matches(
        exact_localizer, routed_registry, match_mode="exact"
    )
    assert _RUNNER._registry_event_matches(
        {**exact_localizer, "anchor_equivalence_id": "9->10@route-q267"},
        routed_registry,
        match_mode="exact",
    )
    assert _RUNNER._registry_event_matches(
        {**exact_localizer, "anchor_equivalence_id": "9->10@q267"},
        routed_registry,
        match_mode="exact",
    )
    assert not _RUNNER._registry_event_matches(
        exact_localizer,
        {**routed_registry, "to_occurrence": 9},
        match_mode="transition",
    )


@pytest.mark.parametrize("serializer", ["causal", "canonical"])
def test_frozen_bank_accepts_both_registered_plan_serializers(serializer: str) -> None:
    heads = ((1, 2), (3, 4))
    registered_sha = (
        _RUNNER._causal_plan_bank_sha256(heads)
        if serializer == "causal"
        else _RUNNER._sha256_json([list(value) for value in heads])
    )
    plan = pd.DataFrame(
        [
            {
                "heads": "[[1, 2], [3, 4]]",
                "bank_sha256": registered_sha,
                "fold": 0,
            }
        ]
    )
    parsed, audit = _RUNNER._bank_for_task(
        SimpleNamespace(heads_json=None, anchor_role=None),
        plan,
        seed=1254,
    )
    assert parsed == heads
    assert audit["bank_sha256"] == registered_sha
