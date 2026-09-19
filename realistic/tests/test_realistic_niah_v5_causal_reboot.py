from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.run_realistic_niah_v5 import (
    _causal_result_rows,
    _diverse_anchor_subset,
    _evaluation_counts,
    _filter_rows_by_count,
    _load_behavior_anchor_routing,
    _prompt_balanced_anchor_subset,
    _prompt_final_transition_anchor_subset,
    _route_transition_anchors,
    _seed_first_anchor_subset,
    _selection_intervention_site_decoupled,
    _validate_behavior_selection_window,
)
from realistic_niah_v5.causal import (
    _closest_natural_norm_source,
    _continuation_metrics,
    _first_generated_city_record,
    _first_generated_gold_city,
    _fixed_target_head_ablation_logits,
    _fixed_target_head_write_intervention_logits,
    _generate_with_prefill_head_ablation,
    _norm_matched_vector,
    _retrieval_behavior_score,
    _shared_source_value_slice,
    _source_attention_concentration_frame,
    _source_specific_write_decomposition,
    _value_capture_module,
    analyze_paired_causal_results,
)
from realistic_niah_v5.spec import V5Config


def test_formal_count_filter_is_explicit_and_validated() -> None:
    config = V5Config()
    rows = [
        {"gold_count": 1, "stimulus_id": "n1"},
        {"gold_records": [{}, {}, {}, {}, {}], "stimulus_id": "n5"},
        {"gold_count": 6, "stimulus_id": "n6"},
    ]
    selected, counts = _filter_rows_by_count(
        rows,
        config=config,
        requested=[5, 1],
    )
    assert counts == (1, 5)
    assert [row["stimulus_id"] for row in selected] == ["n1", "n5"]
    assert _evaluation_counts(config, None) == tuple(range(1, 11))
    with pytest.raises(ValueError, match="must be unique"):
        _evaluation_counts(config, [1, 1])
    with pytest.raises(ValueError, match="outside the registered config"):
        _evaluation_counts(config, [11])


def test_prompt_final_transition_anchor_subset_is_outcome_blind() -> None:
    tasks = []
    for seed, count in ((1234, 3), (1235, 4)):
        row = {
            "request_id": f"request-{seed}-{count}",
            "seed": seed,
            "gold_count": count,
        }
        for occurrence in range(1, count):
            tasks.append(
                (
                    row,
                    {
                        "from_occurrence": occurrence,
                        "to_occurrence": occurrence + 1,
                        "query_output_token_index": occurrence * 10,
                    },
                )
            )
    selected = _prompt_final_transition_anchor_subset(tasks, limit=10)
    assert len(selected) == 2
    assert {
        (int(row["gold_count"]), int(spec["from_occurrence"]), int(spec["to_occurrence"]))
        for row, spec in selected
    } == {(3, 2, 3), (4, 3, 4)}


def test_source_specific_ov_decomposition_supports_grouped_query_attention() -> None:
    projection = nn.Linear(4, 3, bias=True)
    with torch.no_grad():
        projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 2.0, 0.0],
                    [0.0, 1.0, 0.0, 3.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            )
        )
        projection.bias.copy_(torch.tensor([4.0, 5.0, 6.0]))
    adapter = SimpleNamespace(
        head_dims=[2],
        output_projections=[projection],
    )
    attention = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.5, 0.1, 0.2],
        ]
    )
    # One KV head is shared by two query heads (GQA group size two).
    source_values = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
    rows, writes = _source_specific_write_decomposition(
        adapter,
        attention_rows={0: attention},
        key_starts={0: 0},
        source_values={0: source_values},
        source_start=1,
        source_end=3,
    )
    expected_h0 = projection.weight[:, :2] @ torch.tensor([0.2, 0.6])
    expected_h1 = projection.weight[:, 2:] @ torch.tensor([0.5, 0.2])
    assert torch.allclose(writes[(0, 0)], expected_h0)
    assert torch.allclose(writes[(0, 1)], expected_h1)
    assert [row["source_attention_mass"] for row in rows] == pytest.approx(
        [0.5, 0.6]
    )
    assert all(row["value_head"] == 0 for row in rows)
    assert all(row["query_to_value_group_size"] == 2 for row in rows)
    assert all(
        row["layer_source_write_reconstruction_error"] < 1e-6 for row in rows
    )


def test_source_attention_concentration_compares_every_gold_needle() -> None:
    attention = torch.tensor(
        [
            [0.60, 0.10, 0.10, 0.20],
            [0.10, 0.70, 0.10, 0.10],
        ]
    )
    spans = {
        "Riga": SimpleNamespace(start=0, end=1),
        "Baku": SimpleNamespace(start=1, end=2),
        "Osaka": SimpleNamespace(start=2, end=3),
    }
    frame = _source_attention_concentration_frame(
        {0: attention},
        {0: 0},
        source_spans=spans,
        target_city="Riga",
    ).sort_values("head")
    assert frame["target_source_attention_mass"].tolist() == pytest.approx(
        [0.60, 0.10]
    )
    assert frame["target_source_relative_attention_mass"].tolist() == (
        pytest.approx([0.75, 1.0 / 9.0])
    )
    assert frame["target_source_attention_top1"].tolist() == [True, False]
    assert frame["target_source_attention_rank"].tolist() == [1, 2]
    assert frame["target_minus_max_wrong_source_attention_mass"].tolist() == (
        pytest.approx([0.50, -0.60])
    )


def test_gemma_style_value_capture_prefers_normalized_values_and_shared_kv() -> None:
    v_proj = nn.Linear(4, 4, bias=False)
    v_norm = nn.Identity()
    attention = SimpleNamespace(
        v_proj=v_proj,
        v_norm=v_norm,
        layer_type="full_attention",
    )
    assert _value_capture_module(attention) is v_norm

    values = torch.arange(1 * 2 * 6 * 3, dtype=torch.float32).reshape(1, 2, 6, 3)
    selected = _shared_source_value_slice(
        attention,
        {"full_attention": (torch.zeros_like(values), values)},
        source_start=2,
        source_end=5,
    )
    assert selected is not None
    assert selected.shape == (3, 2, 3)
    assert torch.equal(selected, values[0, :, 2:5, :].transpose(0, 1))


class _CaptureLinear(nn.Linear):
    last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value.detach().clone()
        return super().forward(value)


class _HistoryLinear(nn.Linear):
    def __init__(self, width: int) -> None:
        super().__init__(width, width, bias=False)
        self.inputs: list[torch.Tensor] = []
        with torch.no_grad():
            self.weight.copy_(torch.eye(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value.detach().clone())
        return super().forward(value)


class _TwoStepGenerateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.projection = _HistoryLinear(4)
        self.generation_config = SimpleNamespace(eos_token_id=None)
        with torch.no_grad():
            self.embedding.weight.fill_(1.0)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def generate(self, **kwargs) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        self.projection(self.embedding(input_ids))
        generated = torch.tensor([[7, 8]], device=input_ids.device)
        self.projection(self.embedding(generated[:, :1]))
        return torch.cat([input_ids, generated], dim=1)


class _IntegerTokenizer:
    eos_token_id = None
    pad_token_id = None

    @staticmethod
    def decode(values, **_kwargs) -> str:
        return " ".join(str(int(value)) for value in values)


def test_first_generated_gold_city_uses_first_whole_name_match() -> None:
    city, start = _first_generated_gold_city(
        "Osaka-like filler, then Riga; later Osaka.",
        ["Riga", "Osaka"],
    )
    assert city == "Osaka"
    assert start == 0
    assert _first_generated_gold_city("Rigatoni only", ["Riga"]) == (None, None)


@pytest.mark.parametrize(
    ("text", "known", "expected_city", "evidence"),
    [
        (
            " Sapporo received a score of 82. Wait, next is Copenhagen.",
            ["Copenhagen"],
            "Sapporo",
            "received_score",
        ),
        (
            " [Another entry for Fukuoka with 62]",
            ["Fukuoka"],
            "Fukuoka",
            "entry_or_record_for",
        ),
        (
            " (Mexico City - 64)",
            ["Mexico City"],
            "Mexico City",
            "city_dash_or_colon_score",
        ),
    ],
)
def test_first_generated_city_record_detects_semantic_record_before_correction(
    text, known, expected_city, evidence
) -> None:
    city, start, observed_evidence = _first_generated_city_record(text, known)
    assert city == expected_city
    assert start == text.index(expected_city)
    assert observed_evidence == evidence


def test_retrieval_behavior_score_does_not_let_late_gold_correction_pass() -> None:
    score = _retrieval_behavior_score(
        " Srinagar received a score of 78. Wait, the next is Lahore.",
        expected_city="Lahore",
        gold_cities=["Lahore", "Dublin"],
        exact_target_prefix=False,
    )
    assert score["legacy_correct_next_needle"] is True
    assert score["correct_next_needle"] is False
    assert score["first_generated_city_record"] == "Srinagar"
    assert score["behavior_outcome"] == "wrong_non_gold_city_record"


def test_retrieval_behavior_score_does_not_parse_total_as_a_city() -> None:
    score = _retrieval_behavior_score(
        " I cannot find another record.\n</think>\n\nTotal: 10",
        expected_city="Lahore",
        gold_cities=["Lahore", "Dublin"],
        exact_target_prefix=False,
    )
    assert score["first_generated_city_record"] is None
    assert score["behavior_outcome"] == "no_identifiable_city_record"
    assert score["behavior_scoring_policy"] == (
        "first_semantic_city_record_v3_reserved_label_exclusion"
    )


def test_behavioral_head_ablation_applies_only_during_prefill() -> None:
    model = _TwoStepGenerateModel()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=[2],
        head_dims=[2],
        output_projections=[model.projection],
    )
    encoding = SimpleNamespace(
        input_ids=(1, 2, 3, 4),
        attention_mask=(1, 1, 1, 1),
        sequence_length=4,
    )
    completion = _generate_with_prefill_head_ablation(
        model,
        _IntegerTokenizer(),
        adapter,
        encoding,
        [(0, 1)],
        hook_position=2,
        max_new_tokens=2,
    )
    assert completion["generated_token_ids"] == [7, 8]
    assert len(model.projection.inputs) == 2
    assert torch.equal(
        model.projection.inputs[0][0, 2, 2:], torch.zeros(2)
    )
    assert torch.equal(
        model.projection.inputs[0][0, 2, :2], torch.ones(2)
    )
    assert torch.equal(model.projection.inputs[1], torch.ones(1, 1, 4))
    assert completion["head_ablation_decode_steps_requested"] == 0
    assert completion["head_ablation_decode_steps_observed"] == 0
    assert completion["head_ablation_o_proj_input_width_validated"] is True
    assert completion["head_ablation_selected_post_zero_max_abs"] == 0.0


def test_behavioral_head_ablation_can_cover_a_fixed_decode_window() -> None:
    model = _TwoStepGenerateModel()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=[2],
        head_dims=[2],
        output_projections=[model.projection],
    )
    encoding = SimpleNamespace(
        input_ids=(1, 2, 3, 4),
        attention_mask=(1, 1, 1, 1),
        sequence_length=4,
    )
    completion = _generate_with_prefill_head_ablation(
        model,
        _IntegerTokenizer(),
        adapter,
        encoding,
        [(0, 1)],
        hook_position=[1, 3],
        max_new_tokens=2,
        decode_head_ablation_steps=1,
    )
    assert torch.equal(
        model.projection.inputs[0][0, [1, 3], 2:],
        torch.zeros(2, 2),
    )
    assert torch.equal(
        model.projection.inputs[1][0, 0, 2:], torch.zeros(2)
    )
    assert torch.equal(
        model.projection.inputs[1][0, 0, :2], torch.ones(2)
    )
    assert completion["head_ablation_decode_steps_requested"] == 1
    assert completion["head_ablation_decode_steps_observed"] == 1
    assert completion["head_ablation_decode_layer_applications"] == {"0": 1}
    assert completion["head_ablation_decode_policy"] == (
        "first_n_one_token_cached_decode_forwards"
    )


def test_behavioral_head_ablation_can_persist_through_decode() -> None:
    model = _TwoStepGenerateModel()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=[2],
        head_dims=[2],
        output_projections=[model.projection],
    )
    encoding = SimpleNamespace(
        input_ids=(1, 2, 3, 4),
        attention_mask=(1, 1, 1, 1),
        sequence_length=4,
    )
    completion = _generate_with_prefill_head_ablation(
        model,
        _IntegerTokenizer(),
        adapter,
        encoding,
        [(0, 1)],
        hook_position=3,
        max_new_tokens=2,
        decode_head_ablation_steps=-1,
    )
    assert torch.equal(
        model.projection.inputs[1][0, 0, 2:], torch.zeros(2)
    )
    assert completion["head_ablation_decode_steps_requested"] == -1
    assert completion["head_ablation_decode_steps_observed"] == 1
    assert completion["head_ablation_decode_policy"] == (
        "all_one_token_cached_decode_forwards"
    )


def test_behavioral_head_ablation_supports_a_fixed_prefill_window() -> None:
    model = _TwoStepGenerateModel()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=[2],
        head_dims=[2],
        output_projections=[model.projection],
    )
    encoding = SimpleNamespace(
        input_ids=(1, 2, 3, 4),
        attention_mask=(1, 1, 1, 1),
        sequence_length=4,
    )
    _generate_with_prefill_head_ablation(
        model,
        _IntegerTokenizer(),
        adapter,
        encoding,
        [(0, 1)],
        hook_position=[1, 3],
        max_new_tokens=2,
    )
    assert torch.equal(
        model.projection.inputs[0][0, [1, 3], 2:],
        torch.zeros(2, 2),
    )
    assert torch.equal(
        model.projection.inputs[0][0, [0, 2], 2:],
        torch.ones(2, 2),
    )
    assert torch.equal(model.projection.inputs[1], torch.ones(1, 1, 4))


class _KeptLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.projection = _CaptureLinear(4, 4, bias=False)
        with torch.no_grad():
            self.embedding.weight.fill_(1.0)
            self.projection.weight.copy_(torch.eye(4))
        self.full_logits: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.projection(self.embedding(input_ids))
        length = int(input_ids.shape[1])
        logits = torch.arange(
            length * 5, dtype=torch.float32, device=input_ids.device
        ).reshape(1, length, 5)
        self.full_logits = logits.detach().clone()
        return SimpleNamespace(logits=logits[:, -int(logits_to_keep) :])


def test_head_ablation_scores_target_positions_not_anchor_to_target_path() -> None:
    model = _KeptLogitModel()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=[2],
        head_dims=[2],
        output_projections=[model.projection],
    )
    encoding = SimpleNamespace(
        input_ids=tuple(range(8)),
        attention_mask=(1,) * 8,
        sequence_length=8,
    )
    selected = _fixed_target_head_ablation_logits(
        model,
        adapter,
        encoding,
        [(0, 1)],
        hook_position=2,
        target_full_sequence_token_start=6,
        target_full_sequence_token_end=8,
    )
    assert model.full_logits is not None
    assert torch.equal(selected, model.full_logits[0, 5:7].cpu())
    assert model.projection.last_input is not None
    assert torch.equal(
        model.projection.last_input[0, 2, 2:], torch.zeros(2)
    )
    assert torch.equal(
        model.projection.last_input[0, 2, :2], torch.ones(2)
    )


class _PropagatingLogitModel(_KeptLogitModel):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        projected = self.projection(self.embedding(input_ids))
        context = projected.cumsum(dim=1)
        logits = torch.cat(
            [
                context,
                torch.zeros(
                    (*context.shape[:-1], 1),
                    dtype=context.dtype,
                    device=context.device,
                ),
            ],
            dim=-1,
        )
        self.full_logits = logits.detach().clone()
        return SimpleNamespace(logits=logits[:, -int(logits_to_keep) :])


def test_source_write_intervention_zeroes_heads_then_adds_post_o_delta() -> None:
    model = _PropagatingLogitModel()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=[2],
        head_dims=[2],
        output_projections=[model.projection],
    )
    encoding = SimpleNamespace(
        input_ids=tuple(range(8)),
        attention_mask=(1,) * 8,
        sequence_length=8,
    )
    baseline = _fixed_target_head_write_intervention_logits(
        model,
        adapter,
        encoding,
        hook_position=2,
        target_full_sequence_token_start=6,
        target_full_sequence_token_end=8,
    )
    patched = _fixed_target_head_write_intervention_logits(
        model,
        adapter,
        encoding,
        zero_heads=[(0, 1)],
        output_deltas={0: torch.tensor([0.0, 0.0, 3.0, 4.0])},
        hook_position=2,
        target_full_sequence_token_start=6,
        target_full_sequence_token_end=8,
    )
    assert torch.allclose(
        patched - baseline,
        torch.tensor([[0.0, 0.0, 2.0, 3.0, 0.0]]).repeat(2, 1),
    )


def test_norm_matched_vector_preserves_control_direction() -> None:
    control = torch.tensor([3.0, 4.0])
    reference = torch.tensor([0.0, 10.0])
    matched, scale = _norm_matched_vector(control, reference)
    assert scale == pytest.approx(2.0)
    assert torch.linalg.vector_norm(matched) == pytest.approx(10.0)
    assert torch.allclose(matched / scale, control)


def test_closest_natural_norm_source_does_not_use_direction() -> None:
    reference = torch.tensor([10.0, 0.0])
    city, selected = _closest_natural_norm_source(
        reference,
        {
            "far": torch.tensor([0.0, 2.0]),
            "close": torch.tensor([-9.0, 0.0]),
            "middle": torch.tensor([0.0, 5.0]),
        },
    )
    assert city == "close"
    assert torch.equal(selected, torch.tensor([-9.0, 0.0]))


def test_continuation_metrics_keep_non_saturating_logit_contrasts() -> None:
    logits = torch.tensor(
        [
            [20.0, 19.0, 0.0],
            [0.0, 3.0, 2.0],
        ]
    )
    metrics = _continuation_metrics(logits, [0, 1])
    expected_first_log_odds = 20.0 - torch.logsumexp(
        torch.tensor([19.0, 0.0]), dim=0
    ).item()
    assert metrics["target_city_first_token_logit_margin"] == pytest.approx(1.0)
    assert metrics["target_city_mean_token_logit_margin"] == pytest.approx(1.0)
    assert metrics["target_city_first_token_log_odds"] == pytest.approx(
        expected_first_log_odds
    )
    assert metrics["target_city_mean_target_token_logit"] == pytest.approx(11.5)
    assert metrics["target_city_log_probability"] < 0.0


def test_diverse_anchor_subset_balances_transition_indices_and_seeds() -> None:
    tasks = []
    for seed in (1, 2, 3):
        for occurrence in (1, 2, 3):
            tasks.append(
                (
                    {"request_id": f"request-{seed}", "seed": seed},
                    {
                        "anchor_roles": ["p0_item_end"],
                        "grammar_pair": "ranked -> ranked",
                        "causal_cohort": "primary_rank_resolved_full_chain",
                        "from_occurrence": occurrence,
                        "to_occurrence": occurrence + 1,
                        "query_output_token_index": occurrence,
                    },
                )
            )
    selected = _diverse_anchor_subset(tasks, 6)
    transition_counts: dict[int, int] = {}
    for _row, specification in selected:
        occurrence = int(specification["from_occurrence"])
        transition_counts[occurrence] = transition_counts.get(occurrence, 0) + 1
    assert transition_counts == {1: 2, 2: 2, 3: 2}
    assert {int(row["seed"]) for row, _specification in selected} == {1, 2, 3}


def test_diverse_anchor_subset_keeps_balancing_roles_and_grammars() -> None:
    tasks = []
    for seed in range(1, 7):
        for role in ("p0_item_end", "post_marker", "city_pre_d1"):
            for grammar in ("before -> before", "after -> after"):
                tasks.append(
                    (
                        {"request_id": f"request-{seed}", "seed": seed},
                        {
                            "anchor_role": role,
                            "anchor_roles": [role],
                            "grammar_pair": grammar,
                            "causal_cohort": "primary_rank_resolved_full_chain",
                            "from_occurrence": seed % 3 + 1,
                            "to_occurrence": seed % 3 + 2,
                            "query_output_token_index": seed,
                        },
                    )
                )
    selected = _diverse_anchor_subset(tasks, 12)
    role_counts = {
        role: sum(role in specification["anchor_roles"] for _row, specification in selected)
        for role in ("p0_item_end", "post_marker", "city_pre_d1")
    }
    grammar_counts = {
        grammar: sum(
            specification["grammar_pair"] == grammar
            for _row, specification in selected
        )
        for grammar in ("before -> before", "after -> after")
    }
    assert max(role_counts.values()) - min(role_counts.values()) <= 1
    assert max(grammar_counts.values()) - min(grammar_counts.values()) <= 1


def test_seed_first_anchor_subset_uses_every_seed_before_repeating() -> None:
    tasks = []
    for seed in range(1, 6):
        for occurrence in (1, 2, 3):
            tasks.append(
                (
                    {"request_id": f"request-{seed}", "seed": seed},
                    {
                        "grammar_pair": f"grammar-{occurrence % 2}",
                        "from_occurrence": occurrence,
                        "to_occurrence": occurrence + 1,
                        "query_output_token_index": occurrence,
                    },
                )
            )
    selected = _seed_first_anchor_subset(tasks, 5)
    assert len(selected) == 5
    assert {int(row["seed"]) for row, _specification in selected} == set(range(1, 6))


def test_prompt_balanced_anchor_subset_uses_one_anchor_per_prompt() -> None:
    tasks = []
    for prompt_index in range(1, 7):
        for occurrence, grammar in (
            (1, "common_grammar"),
            (2, "rare_grammar" if prompt_index <= 2 else "common_grammar"),
        ):
            tasks.append(
                (
                    {
                        "request_id": f"request-{prompt_index}",
                        "seed": prompt_index,
                        "gold_count": 3,
                    },
                    {
                        "grammar_pair": f"source -> {grammar}",
                        "target_retrieval_surface_variant": grammar,
                        "from_occurrence": occurrence,
                        "to_occurrence": occurrence + 1,
                        "query_output_token_index": occurrence,
                    },
                )
            )
    selected = _prompt_balanced_anchor_subset(tasks, 100)
    request_ids = [row["request_id"] for row, _specification in selected]
    grammars = [
        specification["grammar_pair"].rsplit(" -> ", 1)[-1]
        for _row, specification in selected
    ]
    assert len(selected) == 6
    assert len(set(request_ids)) == 6
    assert grammars.count("rare_grammar") == 2


def test_target_grammar_routing_builds_one_deduplicated_prefill_window(
    tmp_path,
) -> None:
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "policy_id": "test-route-v1",
                "axis": "target_grammar_class",
                "routes": {
                    "same_unit_rank_before_city": {
                        "required": ["p0_item_end", "post_marker", "city_pre_d1"],
                        "optional": ["post_open_delimiter"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    routing = _load_behavior_anchor_routing(routing_path)
    common = {
        "from_occurrence": 1,
        "to_occurrence": 2,
        "target_output_token_start": 20,
        "target_output_token_end": 21,
        "target_city": "Riga",
        "grammar_pair": (
            "same_unit_rank_before_city -> same_unit_rank_before_city"
        ),
        "event_specific": True,
        "local_anchor_eligible": True,
        "primary_anchor_eligible": True,
    }
    specifications = [
        {
            **common,
            "anchor_equivalence_id": "1->2@q5",
            "anchor_roles": ["p0_item_end"],
            "query_output_token_index": 5,
        },
        {
            **common,
            "anchor_equivalence_id": "1->2@q17",
            "anchor_roles": ["post_marker"],
            "query_output_token_index": 17,
        },
        {
            **common,
            "anchor_equivalence_id": "1->2@q19",
            "anchor_roles": ["post_open_delimiter", "city_pre_d1"],
            "query_output_token_index": 19,
        },
    ]
    routed, excluded = _route_transition_anchors(specifications, routing)
    assert excluded == []
    assert len(routed) == 1
    assert routed[0]["routed_anchor_equivalence_ids"] == [
        "1->2@q5",
        "1->2@q17",
        "1->2@q19",
    ]
    assert routed[0]["routed_query_output_token_indices"] == [5, 17, 19]
    assert routed[0]["routed_anchor_roles_applied"] == [
        "p0_item_end",
        "post_marker",
        "city_pre_d1",
        "post_open_delimiter",
    ]
    assert routed[0]["query_output_token_index"] == 19


def test_multi_site_window_must_include_exact_head_selection_site(
    tmp_path,
) -> None:
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(
        json.dumps(
            {
                "policy_id": "rank-window-v1",
                "axis": "target_grammar_class",
                "routes": {
                    "adjacent_rank_before_city": {
                        "required": [
                            "pre_marker_d1",
                            "post_marker",
                            "city_pre_d1",
                        ],
                        "optional": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    routing = _load_behavior_anchor_routing(routing_path)
    assert _validate_behavior_selection_window(
        routing,
        selection_anchor_role="post_marker",
        target_grammar_class="adjacent_rank_before_city",
    ) == ["pre_marker_d1", "post_marker", "city_pre_d1"]
    with pytest.raises(ValueError, match="must contain the exact site"):
        _validate_behavior_selection_window(
            routing,
            selection_anchor_role="p0_item_end",
            target_grammar_class="adjacent_rank_before_city",
        )
    assert _validate_behavior_selection_window(
        routing,
        selection_anchor_role="p0_item_end",
        target_grammar_class="adjacent_rank_before_city",
        require_selection_anchor=False,
    ) == ["pre_marker_d1", "post_marker", "city_pre_d1"]
    assert _validate_behavior_selection_window(
        routing,
        selection_anchor_role="post_marker",
        target_grammar_class=None,
        require_selection_anchor=False,
    ) == ["pre_marker_d1", "post_marker", "city_pre_d1"]


def test_selection_and_intervention_sites_may_be_explicitly_decoupled() -> None:
    assert _selection_intervention_site_decoupled(
        "post_marker", ["p0_item_end"]
    )
    assert not _selection_intervention_site_decoupled(
        "post_marker", ["p0_item_end", "post_marker"]
    )
    assert not _selection_intervention_site_decoupled(None, ["p0_item_end"])
    with pytest.raises(ValueError, match="At least one intervention"):
        _selection_intervention_site_decoupled("post_marker", [])


def test_causal_result_rows_reads_atomic_output_directory(tmp_path) -> None:
    shards = tmp_path / "behavior" / "shards"
    shards.mkdir(parents=True)
    (shards / "b.jsonl").write_text('{"trial_id":"b"}\n', encoding="utf-8")
    (shards / "a.jsonl").write_text('{"trial_id":"a"}\n', encoding="utf-8")
    assert [row["trial_id"] for row in _causal_result_rows(tmp_path / "behavior")] == [
        "a",
        "b",
    ]


def test_directory_analysis_pairs_before_seed_and_uses_planned_bank_size(
    tmp_path,
) -> None:
    rows = []
    for model_label in ("Qwen3-8B", "Gemma4-E4B"):
        for seed in (1, 2):
            common = {
                "status": "ok",
                "model_label": model_label,
                "seed": seed,
                "request_id": f"{model_label}-{seed}",
                "anchor_equivalence_id": "1->2@q5",
                "anchor_roles": ["city_pre_d1"],
                "mechanism": "retrieval_anchor_localization",
                "transition_phase": "continue",
                "planned_bank_size": 8,
            }
            rows.extend(
                [
                    {
                        **common,
                        "condition": "clean",
                        "bank_size": 0,
                        "target_city_log_probability": -1.0,
                    },
                    {
                        **common,
                        "condition": "selected_bank",
                        "bank_size": 8,
                        "target_city_log_probability": -3.0,
                    },
                ]
            )
    shard_dir = tmp_path / "trials" / "shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = analyze_paired_causal_results(
        tmp_path / "trials",
        tmp_path / "analysis.csv",
        treatment="selected_bank",
        control="clean",
        outcome="target_city_log_probability",
        config=V5Config(bootstrap_samples=200),
        mechanism="retrieval_anchor_localization",
        bank_size=8,
        transition_phase="continue",
        anchor_role="city_pre_d1",
    )
    assert list(result["mean_effect"]) == pytest.approx([-2.0, -2.0])
    assert set(result["n_seeds"]) == {2}
    assert set(result["n_paired_anchor_units"]) == {2}
    assert (
        result["holm_sign_flip_pvalue_across_models"]
        >= result["sign_flip_pvalue"]
    ).all()
