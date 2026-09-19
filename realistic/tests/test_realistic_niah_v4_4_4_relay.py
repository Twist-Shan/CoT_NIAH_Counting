from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from realistic_niah_v4.prompts import PromptEncoding, TokenSpan
from realistic_niah_v4_4_3.interventions import QueryBundle
from realistic_niah_v4_4_4.relay import (
    contribution_reconstruction_diagnostics,
    edge_z_from_values,
    natural_axis_block_for_patch_delta,
    receiver_alpha_donor_v_delta,
    resolve_position_set,
    source_contribution_vector,
)
from realistic_niah_v4_4_4.relay_analysis import (
    build_relay_seed_metrics,
    summarize_relay_seed_metrics,
)
from realistic_niah_v4_4_4.relay_spec import V444RelayConfig


def _adapter() -> SimpleNamespace:
    projection = torch.nn.Linear(8, 3, bias=False)
    with torch.no_grad():
        projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
                ]
            )
        )
    return SimpleNamespace(
        head_dims={28: 2},
        num_heads={28: 4},
        output_projections={28: projection},
    )


def _bundle(values: torch.Tensor) -> QueryBundle:
    alpha = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [0.5, 0.25, 0.25],
            [0.3, 0.3, 0.4],
            [0.1, 0.2, 0.7],
        ],
        dtype=torch.float32,
    )
    heads = []
    for head in range(4):
        kv = head // 2
        selected = values[:, kv * 2 : kv * 2 + 2]
        heads.append(torch.einsum("k,kd->d", alpha[head], selected))
    z = torch.cat(heads)
    return QueryBundle(
        logits=torch.zeros(12),
        candidate_log_scores={count: float(-count) for count in range(1, 11)},
        z_by_layer={28: z},
        value_by_layer={28: values},
        attention_output_by_layer={28: torch.zeros(3)},
        alpha_by_layer={28: alpha},
        alpha_key_start_by_layer={28: 0},
        attention_cache_candidate_logit_max_abs_delta=0.0,
        attention_cache_candidate_centered_logit_max_abs_delta=0.0,
    )


def _encoding() -> PromptEncoding:
    slots = (
        TokenSpan(0, 0, 1, True, "needle", 1, 1),
        TokenSpan(1, 1, 2, False, "negative", 1, 1),
    )
    return PromptEncoding(
        stimulus_id="x",
        design_variant="v4.4",
        seed=1,
        split="confirmation",
        count=1,
        model_label="Qwen3-8B",
        answer_format="numeric",
        text="x",
        generation_prompt="x",
        input_ids=(1, 2, 3, 4),
        attention_mask=(1, 1, 1, 1),
        query_position=3,
        slot_spans=slots,
        needle_spans=(slots[0],),
        hard_negative_spans=(slots[1],),
        count_candidate_texts=tuple((count, str(count)) for count in range(1, 11)),
        count_candidate_answer_token_ids=tuple(
            (count, (count,)) for count in range(1, 11)
        ),
        count_candidate_token_ids=tuple(
            (count, (count, 0)) for count in range(1, 11)
        ),
    )


def test_receiver_alpha_donor_v_edge_patch_is_exact() -> None:
    adapter = _adapter()
    receiver_values = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0], [1.0, 1.0, 2.0, 2.0]]
    )
    donor_values = receiver_values.clone()
    donor_values[2, :2] += torch.tensor([2.0, -1.0])
    receiver = _bundle(receiver_values)
    donor = _bundle(donor_values)
    delta = receiver_alpha_donor_v_delta(
        receiver,
        donor,
        adapter,
        layer=28,
        heads=(0, 1),
        positions=(2,),
    )
    expected = torch.stack(
        (
            receiver.alpha_by_layer[28][0, 2] * torch.tensor([2.0, -1.0]),
            receiver.alpha_by_layer[28][1, 2] * torch.tensor([2.0, -1.0]),
        )
    )
    assert torch.allclose(delta, expected)


def test_source_contributions_reconstruct_selected_set_output() -> None:
    adapter = _adapter()
    bundle = _bundle(
        torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0], [1.0, 1.0, 2.0, 2.0]]
        )
    )
    axis = torch.tensor([1.0, -0.5, 0.25])
    contribution = source_contribution_vector(
        bundle, adapter, layer=28, heads=(0, 1), output_axis=axis
    )
    edge = edge_z_from_values(
        bundle,
        bundle,
        adapter,
        layer=28,
        heads=(0, 1),
        positions=(0, 1, 2),
    )
    weight = adapter.output_projections[28].weight.detach()
    output = weight[:, :2] @ edge[0] + weight[:, 2:4] @ edge[1]
    expected = torch.dot(output, axis / torch.linalg.vector_norm(axis))
    assert torch.allclose(contribution.sum(), expected, atol=1e-6)
    diagnostics = contribution_reconstruction_diagnostics(
        bundle, adapter, layer=28, heads=(0, 1)
    )
    assert diagnostics["edge_z_reconstruction_relative_l2"] < 1e-6


def test_position_sets_exclude_slots_and_keep_query_explicit() -> None:
    encoding = _encoding()
    contribution = torch.tensor([10.0, 9.0, 1.0, 0.5])
    assert resolve_position_set(encoding, "answer_query_self") == (3,)
    assert resolve_position_set(encoding, "active_needle_endpoints") == (0,)
    assert resolve_position_set(
        encoding, "non_slot_top_1", contribution=contribution
    ) == (2,)


def test_patch_axis_block_removes_only_parallel_component() -> None:
    adapter = _adapter()
    steps = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    patch = torch.tensor([[0.7, -0.2], [0.1, 0.3]])
    block, control, diagnostics = natural_axis_block_for_patch_delta(
        adapter,
        layer=28,
        heads=(0, 1),
        patch_delta_z=patch,
        global_z_count_steps=steps,
        orthogonal_label="unit-test",
    )
    assert block.shape == patch.shape
    assert control.shape == patch.shape
    assert abs(diagnostics["relay_axis_block_residual_projection"]) < 1e-5
    assert abs(diagnostics["relay_axis_control_cosine"]) < 1e-5


def _synthetic_relay_frames(
    config: V444RelayConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    natural = []
    patch = []
    removal = []
    for seed in config.confirmation_seeds:
        for count in config.counts:
            natural.append(
                {
                    "seed": seed,
                    "gold_count": count,
                    "relay_carrier_coefficient": 0.2 * count + 0.001 * seed,
                }
            )
        for receiver, donor in config.relay_pairs:
            gap = donor - receiver
            for intervention, transport in (
                ("receiver_alpha_donor_v_edge_patch", 0.30),
                (
                    "receiver_alpha_donor_v_edge_patch_plus_natural_axis_block",
                    0.05,
                ),
                (
                    "receiver_alpha_donor_v_edge_patch_plus_orthogonal_control",
                    0.28,
                ),
            ):
                patch.append(
                    {
                        "seed": seed,
                        "receiver_count": receiver,
                        "donor_count": donor,
                        "intervention": intervention,
                        "continuous_normalized_transport": transport,
                        "relay_patch_global_axis_coefficient": 0.2 * gap,
                    }
                )
        for count in config.removal_counts:
            removal.extend(
                (
                    {
                        "seed": seed,
                        "gold_count": count,
                        "intervention": "relay_axis_removal",
                        "delta_expected_count_absolute_error": 0.2,
                        "delta_correct_margin": -0.3,
                    },
                    {
                        "seed": seed,
                        "gold_count": count,
                        "intervention": "relay_axis_orthogonal_control",
                        "delta_expected_count_absolute_error": 0.0,
                        "delta_correct_margin": 0.0,
                    },
                )
            )
    return pd.DataFrame(natural), pd.DataFrame(patch), pd.DataFrame(removal)


def test_clean_synthetic_relay_passes_joint_decision() -> None:
    config = V444RelayConfig()
    natural, patch, removal = _synthetic_relay_frames(config)
    seed_metrics = build_relay_seed_metrics(
        natural, patch, removal, relay_config=config
    )
    summary, decision = summarize_relay_seed_metrics(
        seed_metrics, relay_config=config
    )
    assert len(summary) == 6
    assert decision["all_families_pass"] is True
