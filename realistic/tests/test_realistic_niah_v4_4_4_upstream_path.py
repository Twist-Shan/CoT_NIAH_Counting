from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from realistic_niah_v4.prompts import PromptEncoding, TokenSpan
from realistic_niah_v4_4_3.interventions import QueryBundle
from realistic_niah_v4_4_4.upstream_path import (
    broad_retrieval_score,
    late_block_and_control,
    slot_positions,
)
from realistic_niah_v4_4_4.upstream_path_analysis import (
    _decision,
    exact_sign_flip_p,
    holm_adjust,
)
from realistic_niah_v4_4_4.upstream_path_report import _interval_svg
from realistic_niah_v4_4_4.upstream_path_spec import V444UpstreamPathConfig


def _encoding() -> PromptEncoding:
    slots = (
        TokenSpan(0, 1, 2, True, "needle", 1, 1),
        TokenSpan(1, 3, 4, True, "needle", 1, 1),
    )
    return PromptEncoding(
        stimulus_id="x",
        design_variant="v4.4",
        seed=1,
        split="confirmation",
        count=2,
        model_label="Qwen3-8B",
        answer_format="numeric",
        text="x",
        generation_prompt="x",
        input_ids=(1, 2, 3, 4, 5, 6),
        attention_mask=(1,) * 6,
        query_position=5,
        slot_spans=slots,
        needle_spans=slots,
        hard_negative_spans=(),
        count_candidate_texts=tuple((count, str(count)) for count in range(1, 11)),
        count_candidate_answer_token_ids=tuple((count, (count,)) for count in range(1, 11)),
        count_candidate_token_ids=tuple((count, (count, 0)) for count in range(1, 11)),
    )


def _bundle() -> QueryBundle:
    alpha = torch.zeros(32, 6)
    alpha[:, 0] = 0.2
    alpha[:, 1] = 0.3
    alpha[:, 2] = 0.1
    alpha[:, 3] = 0.3
    alpha[:, 4] = 0.05
    alpha[:, 5] = 0.05
    return QueryBundle(
        logits=torch.zeros(20),
        candidate_log_scores={count: -float(count) for count in range(1, 11)},
        z_by_layer={23: torch.zeros(32 * 2)},
        value_by_layer={23: torch.zeros(6, 16)},
        attention_output_by_layer={23: torch.zeros(8)},
        alpha_by_layer={23: alpha},
        alpha_key_start_by_layer={23: 0},
        attention_cache_candidate_logit_max_abs_delta=0.0,
        attention_cache_candidate_centered_logit_max_abs_delta=0.0,
    )


def _adapter() -> SimpleNamespace:
    projection = torch.nn.Linear(8, 5, bias=False)
    with torch.no_grad():
        projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
                    [0.2, -0.3, 0.4, 0.1, -0.2, 0.1, 0.3, -0.4],
                    [0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.2, 0.1],
                ]
            )
        )
    return SimpleNamespace(
        head_dims={28: 2},
        num_heads={28: 4},
        output_projections={28: projection},
        num_layers=36,
    )


def test_config_freezes_v442_candidates_and_nested_sets() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "realistic_niah_v4_4_4_upstream_path.json"
    )
    config = V444UpstreamPathConfig.from_json(path)
    assert config.early_set_sizes == (2, 4, 8)
    assert [(item.layer, item.head) for item in config.early_set(2)] == [
        (27, 18),
        (23, 28),
    ]
    stable = [item.stable_score for item in config.early_candidates]
    assert stable == sorted(stable, reverse=True)
    assert config.late_sets[config.primary_late_set] == (16, 19)


def test_slot_positions_and_broad_score_follow_registered_spans() -> None:
    encoding = _encoding()
    assert slot_positions(encoding) == (1, 3)
    result = broad_retrieval_score(_bundle(), encoding, layer=23, head=0)
    assert result["needle_attention_mass"] == pytest.approx(0.6)
    assert result["occurrence_coverage"] == pytest.approx(1.0)
    assert result["broad_retrieval_score"] == pytest.approx(0.6)


def test_l28_block_and_control_are_equal_norm_and_output_orthogonal() -> None:
    adapter = _adapter()
    induced = torch.tensor([[0.4, -0.2], [0.1, 0.3]])
    block, control, diagnostics = late_block_and_control(
        adapter,
        layer=28,
        heads=(0, 1),
        induced_delta=induced,
        label="unit-test",
    )
    assert torch.allclose(block, -induced)
    assert diagnostics["late_control_output_norm"] == pytest.approx(
        diagnostics["late_induced_output_norm"], rel=1e-5
    )
    assert abs(diagnostics["late_control_output_cosine_to_induced"]) < 1e-5
    assert control.shape == induced.shape


def test_exact_sign_flip_and_holm_are_deterministic() -> None:
    assert exact_sign_flip_p(np.ones(10)) == pytest.approx(2 / 1024)
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_decision_promotes_supported_expanded_l28_set() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "realistic_niah_v4_4_4_upstream_path.json"
    )
    config = V444UpstreamPathConfig.from_json(path)
    expanded_set = next(
        name for name in config.late_sets if name != config.primary_late_set
    )
    rows = [
        {
            "late_set": config.primary_late_set,
            "route": route,
            "serial_path_supported": False,
        }
        for route in config.routes
    ]
    rows.append(
        {
            "late_set": expanded_set,
            "route": "slot_state",
            "serial_path_supported": True,
        }
    )

    decision = _decision(pd.DataFrame(rows), config)

    assert decision["base_h16_h19_sufficient"] is False
    assert decision["expanded_l28_set_support"] is True
    assert decision["overall_mechanistic_support"] is True
    assert decision["classification"] == (
        "upstream_read_to_expanded_l28_write_supported_exploratory"
    )


def test_interval_svg_can_label_l28_sets() -> None:
    frame = pd.DataFrame(
        [
            {
                "late_set": "base_h16_h19",
                "early_set": "top4",
                "route": "slot_state",
                "score_mean": 0.1,
                "score_ci_low": 0.05,
                "score_ci_high": 0.15,
            },
            {
                "late_set": "gqa_h16_h19",
                "early_set": "top4",
                "route": "slot_state",
                "score_mean": 0.2,
                "score_ci_low": 0.12,
                "score_ci_high": 0.28,
            },
        ]
    )

    rendered = _interval_svg(
        frame,
        metric="score",
        title="test",
        x_label="x",
        label_column="late_set",
    )

    assert "base_h16_h19" in rendered
    assert "gqa_h16_h19" in rendered
