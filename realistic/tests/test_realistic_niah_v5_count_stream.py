from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_realistic_niah_v5_count_stream import (
    _cohort_exclusion_reason,
    _validate_trace_patch_basis_manifest,
    build_parser,
)
from scripts.analyze_realistic_niah_v5_joint_state_source import (
    factorial_request_effects,
)
from scripts.analyze_realistic_niah_v5_full_state_patch_source import (
    full_state_patch_source_effects,
    summarize_effects as summarize_patch_source_effects,
)
from scripts.analyze_realistic_niah_v5_serial_patch_source import (
    serial_source_claim_gates,
)
from scripts.analyze_realistic_niah_v5_serial_patch_heads import (
    _claim_gates as serial_claim_gates,
    serial_patch_head_effects,
    summarize_serial_effects,
)
from realistic_niah_v5.count_stream import (
    AnswerSourceRegistry,
    NativeCountMechanismSpec,
    _full_state_patch_layers,
    _normalize_head_readout_arm,
    _registered_ordinary_corruption_banks,
    answer_source_mask,
    build_answer_broad_head_plan,
    build_html_aligned_uninformative_trace_encoding,
    build_sparse_trace_patch_sample_plan,
    build_terminal_last_trace_patch_sample_plan,
    build_terminal_serial_pair_plan,
    build_trace_patch_pair_plan,
    deterministic_control_basis,
    fit_count_stream_basis,
    mechanism_decision_ledger,
    rank_answer_broad_heads,
    select_answer_broad_bank_size,
    source_attention_metrics,
    stream_state_retention_metrics,
    summarize_linear_contrasts,
    trace_patch_geometry_positions,
    trace_patch_condition_states,
    valid_trace_patch_receivers,
)
from realistic_niah_v5.encoding import NativeTraceEncoding


def test_serial_head_arm_hashes_frozen_ranking_order_before_canonicalizing() -> None:
    adapter = SimpleNamespace(num_layers=4, num_heads=(2, 2, 2, 2))
    ranked_heads = [[3, 1], [0, 0], [2, 1]]
    frozen_sha = hashlib.sha256(
        json.dumps(
            ranked_heads, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    arm = _normalize_head_readout_arm(
        adapter,
        {
            "condition": "selected_bank",
            "repeat": 0,
            "heads": ranked_heads,
            "bank_sha256": frozen_sha,
        },
    )
    assert arm["heads"] == ((0, 0), (2, 1), (3, 1))
    assert arm["bank_sha256"] == frozen_sha
    assert arm["canonical_bank_sha256"] != frozen_sha


def test_serial_head_arm_rejects_mutated_frozen_plan() -> None:
    adapter = SimpleNamespace(num_layers=2, num_heads=(2, 2))
    with pytest.raises(ValueError, match="bank hash disagrees"):
        _normalize_head_readout_arm(
            adapter,
            {
                "condition": "selected_bank",
                "repeat": 0,
                "heads": [[1, 1], [0, 0]],
                "bank_sha256": "0" * 64,
            },
        )


def test_terminal_serial_pair_plan_uses_every_eligible_seed_without_rank() -> None:
    rows = [
        {
            "request_id": f"serial-{seed}-{count}",
            "model_label": "Qwen3-8B",
            "seed": seed,
            "gold_count": count,
            "mechanism_split": "development",
            "prediction": -999,
        }
        for seed in range(100, 104)
        for count in range(1, 11)
    ]
    plan = build_terminal_serial_pair_plan(rows, model_label="Qwen3-8B")
    repeated = build_terminal_serial_pair_plan(
        [{**row, "prediction": 999} for row in reversed(rows)],
        model_label="Qwen3-8B",
    )
    assert len(plan) == 19 * 4
    assert "selection_rank" not in plan.columns
    assert set(plan["selection_policy"]) == {"all_eligible_registered_seeds"}
    assert set(plan.groupby(["gold_count", "donor_offset"]).size()) == {4}
    assert list(plan["pair_sha256"]) == list(repeated["pair_sha256"])


def test_serial_patch_head_analysis_requires_specific_mediation_and_equivalence() -> None:
    rows = []
    for seed in range(10):
        common = {
            "pair_sha256": f"pair-{seed}",
            "request_id": f"request-{seed}",
            "model_label": "Qwen3-8B",
            "seed": seed,
            "gold_count": 10,
            "donor_offset": -1,
            "receiver_occurrence": 10,
            "donor_occurrence": 9,
            "mechanism_split": "confirmation",
            "status": "ok",
        }
        values = {
            ("self_patch", "clean"): (10.0, 10.0),
            ("full_donor_patch", "clean"): (6.0, 9.0),
            ("self_patch", "selected_bank"): (8.0, 9.0),
            ("full_donor_patch", "selected_bank"): (8.0, 9.0),
        }
        for (patch, head), (margin, expected_count) in values.items():
            rows.append(
                {
                    **common,
                    "patch_condition": patch,
                    "head_condition": head,
                    "head_repeat": 0,
                    "correct_count_margin": margin,
                    "expected_count": expected_count,
                }
            )
        for repeat in (1, 2, 3):
            rows.extend(
                [
                    {
                        **common,
                        "patch_condition": "self_patch",
                        "head_condition": "layer_matched_random",
                        "head_repeat": repeat,
                        "correct_count_margin": 9.8,
                        "expected_count": 9.8,
                    },
                    {
                        **common,
                        "patch_condition": "full_donor_patch",
                        "head_condition": "layer_matched_random",
                        "head_repeat": repeat,
                        "correct_count_margin": 5.8,
                        "expected_count": 8.8,
                    },
                ]
            )
    effects = serial_patch_head_effects(pd.DataFrame(rows))
    assert set(effects["correct_count_margin__patch_damage_clean"]) == {4.0}
    assert set(effects["correct_count_margin__patch_damage_selected"]) == {0.0}
    assert effects["correct_count_margin__specific_head_mediation"].to_numpy() == pytest.approx(4.0)
    assert set(effects["expected_count__adoption_clean"]) == {1.0}
    assert set(effects["expected_count__adoption_selected"]) == {0.0}
    _seed_effects, summary = summarize_serial_effects(
        effects, bootstrap_samples=500, random_seed=7
    )
    claims = serial_claim_gates(summary, phase="confirmation")
    assert claims["serial_readout_pass"] is True
    assert claims["gates"]["selected_bank_residual_equivalence"]["pass"] is True


def _registry() -> AnswerSourceRegistry:
    value = AnswerSourceRegistry(
        request_id="request-1",
        answer_site_id="answer_query_v3",
        sequence_length=40,
        prompt_token_count=25,
        query_position=39,
        prompt_records=((2, 5), (7, 9)),
        trace_context=((25, 39),),
        trace_items=((26, 30), (31, 35)),
        trace_other=((25, 26), (30, 31), (35, 39)),
        trace_markers=((26, 27), (31, 32)),
        trace_nonmarkers=((27, 30), (32, 35)),
        earlier_trace_items=((26, 30),),
        terminal_trace_item=((31, 35),),
    )
    value.validate()
    return value


def test_joint_state_source_interactions_distinguish_parallel_and_serial() -> None:
    states = (
        "clean",
        "aligned_running_state_removal",
        "norm_matched_orthogonal_removal",
    )
    masks = (
        "clean",
        "block_trace_items",
        "block_trace_items_matched_control",
        "block_prompt_records",
        "block_prompt_records_matched_control",
    )
    values = {
        # Prompt masking amplifies aligned state removal: +1 interaction.
        ("norm_matched_orthogonal_removal", "block_prompt_records"): 9.0,
        ("norm_matched_orthogonal_removal", "block_prompt_records_matched_control"): 10.0,
        ("aligned_running_state_removal", "block_prompt_records"): 6.0,
        ("aligned_running_state_removal", "block_prompt_records_matched_control"): 8.0,
        # Trace masking occludes aligned state removal: -1 interaction.
        ("norm_matched_orthogonal_removal", "block_trace_items"): 9.0,
        ("norm_matched_orthogonal_removal", "block_trace_items_matched_control"): 10.0,
        ("aligned_running_state_removal", "block_trace_items"): 8.0,
        ("aligned_running_state_removal", "block_trace_items_matched_control"): 8.0,
    }
    rows = []
    for state in states:
        for mask in masks:
            rows.append(
                {
                    "status": "ok",
                    "model_label": "Qwen3-8B",
                    "seed": 1234,
                    "request_id": "request-1",
                    "gold_count": 5,
                    "dataset_split": "discovery",
                    "state_source_layer": 19,
                    "state_source_scope": "all",
                    "state_condition": state,
                    "mask_condition": mask,
                    "correct_count_margin": values.get((state, mask), 10.0),
                }
            )
    effects = factorial_request_effects(pd.DataFrame(rows))
    assert len(effects) == 1
    assert effects.loc[
        0, "correct_count_margin__prompt_records__specific_interaction"
    ] == pytest.approx(1.0)
    assert effects.loc[
        0, "correct_count_margin__trace_items__specific_interaction"
    ] == pytest.approx(-1.0)


def test_full_state_patch_source_adoption_interaction_has_registered_sign() -> None:
    masks = (
        "clean",
        "block_trace_items",
        "block_trace_items_matched_control",
        "block_prompt_records",
        "block_prompt_records_matched_control",
    )
    donor_expected = {
        "clean": 4.8,
        "block_prompt_records": 4.0,
        "block_prompt_records_matched_control": 4.8,
        "block_trace_items": 4.9,
        "block_trace_items_matched_control": 4.8,
    }
    rows = []
    for patch in ("self_patch", "full_donor_patch"):
        for mask in masks:
            rows.append(
                {
                    "status": "ok",
                    "model_label": "Gemma4-E4B",
                    "seed": 1234,
                    "request_id": "request-1",
                    "pair_sha256": "pair-1",
                    "selection_cell_id": "cell-1",
                    "selection_rank": 1,
                    "gold_count": 5,
                    "receiver_occurrence": 5,
                    "donor_occurrence": 4,
                    "donor_offset": -1,
                    "patch_geometry": "suffix8",
                    "patch_layer_mode": "cumulative_clamp",
                    "layer": 16,
                    "patch_condition": patch,
                    "mask_condition": mask,
                    "expected_count": (
                        5.0 if patch == "self_patch" else donor_expected[mask]
                    ),
                    "correct_count_margin": (
                        10.0
                        if patch == "self_patch"
                        else 10.0 - abs(5.0 - donor_expected[mask])
                    ),
                }
            )
    effects = full_state_patch_source_effects(pd.DataFrame(rows))
    assert effects.loc[
        0, "expected_count__prompt_records__adoption_interaction"
    ] == pytest.approx(0.8)
    assert effects.loc[
        0, "expected_count__trace_items__adoption_interaction"
    ] == pytest.approx(-0.1)

    rank_free = pd.DataFrame(rows).drop(columns="selection_rank")
    rank_free["within_cell_index"] = 1
    rank_free["selection_policy"] = "all_eligible_registered_seeds"
    rank_free["mechanism_split"] = "confirmation"
    rank_free["mask_scope"] = "answer_query_and_answer_tokens"
    rank_free_effects = full_state_patch_source_effects(rank_free)
    assert len(rank_free_effects) == 1


def test_persistent_all_head_serial_source_claim_requires_patch_occlusion() -> None:
    rows = []
    masks = (
        "clean",
        "block_trace_items",
        "block_trace_items_matched_control",
        "block_prompt_records",
        "block_prompt_records_matched_control",
    )
    for seed in range(10):
        for patch in ("self_patch", "full_donor_patch"):
            for mask in masks:
                self_margin = 10.0
                donor_margin = 6.0
                self_expected = 10.0
                donor_expected = 9.0
                if mask == "block_trace_items":
                    self_margin = donor_margin = 8.0
                    self_expected = donor_expected = 9.0
                rows.append(
                    {
                        "status": "ok",
                        "model_label": "Qwen3-8B",
                        "seed": seed,
                        "request_id": f"request-{seed}",
                        "pair_sha256": f"pair-{seed}",
                        "selection_cell_id": "count10_offset-1",
                        "within_cell_index": seed + 1,
                        "selection_policy": "all_eligible_registered_seeds",
                        "mechanism_split": "confirmation",
                        "mask_scope": "answer_query_and_answer_tokens",
                        "gold_count": 10,
                        "receiver_occurrence": 10,
                        "donor_occurrence": 9,
                        "donor_offset": -1,
                        "patch_geometry": "suffix8",
                        "patch_layer_mode": "cumulative_clamp",
                        "layer": 19,
                        "patch_condition": patch,
                        "mask_condition": mask,
                        "correct_count_margin": (
                            self_margin if patch == "self_patch" else donor_margin
                        ),
                        "expected_count": (
                            self_expected if patch == "self_patch" else donor_expected
                        ),
                        "exact_count": (
                            1.0
                            if patch == "self_patch"
                            or mask == "block_trace_items"
                            else 0.0
                        ),
                    }
                )
    effects = full_state_patch_source_effects(pd.DataFrame(rows))
    summary = summarize_patch_source_effects(
        effects, bootstrap_samples=200, random_seed=11
    )
    claims = serial_source_claim_gates(
        effects,
        summary,
        phase="confirmation",
        bootstrap_samples=200,
        random_seed=11,
    )
    assert claims["distributed_serial_readout_pass"] is True
    assert claims["gates"]["trace_mask_residual_equivalence"]["pass"] is True
    assert claims["greedy_exact_count_support_pass"] is True


def test_source_registry_rejects_nonpartitioned_markers() -> None:
    value = _registry()
    invalid = AnswerSourceRegistry(
        **{
            **value.__dict__,
            "trace_nonmarkers": ((17, 20), (23, 25)),
        }
    )
    with pytest.raises(ValueError, match="do not cover"):
        invalid.validate()


def test_answer_source_masks_have_exact_registered_budgets() -> None:
    registry = _registry()
    clean, clean_audit = answer_source_mask(registry, condition="clean")
    trace, trace_audit = answer_source_mask(registry, condition="block_trace_items")
    trace_context, trace_context_audit = answer_source_mask(
        registry, condition="block_trace_context"
    )
    prompt, prompt_audit = answer_source_mask(
        registry, condition="block_prompt_records"
    )
    both, both_audit = answer_source_mask(registry, condition="block_trace_and_prompt")
    trace_control, trace_control_audit = answer_source_mask(
        registry, condition="block_trace_items_matched_control"
    )
    context_control, context_control_audit = answer_source_mask(
        registry, condition="block_trace_context_matched_control"
    )
    marker_control, marker_control_audit = answer_source_mask(
        registry, condition="block_trace_markers_matched_control"
    )
    assert int(clean.sum()) == 40
    assert clean_audit["blocked_token_count"] == 0
    assert trace_audit["blocked_token_count"] == 8
    assert trace_context_audit["blocked_token_count"] == 14
    assert prompt_audit["blocked_token_count"] == 5
    assert both_audit["blocked_token_count"] == 19
    assert trace_control_audit["blocked_token_count"] == 8
    assert marker_control_audit["blocked_token_count"] == 2
    assert context_control_audit["blocked_token_count"] == 14
    assert int(trace_control.sum()) == int(trace.sum())
    assert (
        trace_control_audit["blocked_positions_sha256"]
        != trace_audit["blocked_positions_sha256"]
    )
    assert int(marker_control[:, registry.query_position]) == 1
    assert int(prompt[:, registry.query_position]) == 1
    assert int(both[:, registry.query_position]) == 1


def test_marker_mask_is_explicitly_not_applicable_for_unmarked_trace() -> None:
    value = _registry()
    unmarked = AnswerSourceRegistry(
        **{
            **value.__dict__,
            "trace_markers": (),
            "trace_nonmarkers": value.trace_items,
        }
    )
    unmarked.validate()
    with pytest.raises(ValueError, match="not applicable"):
        answer_source_mask(unmarked, condition="block_trace_markers")


def test_ordinary_corruption_banks_exclude_active_prompt_records() -> None:
    registry = AnswerSourceRegistry(
        request_id="request-banks",
        answer_site_id="answer_query_v3",
        sequence_length=70,
        prompt_token_count=50,
        query_position=69,
        prompt_records=((10, 14), (20, 24)),
        trace_context=((50, 69),),
        trace_items=((52, 56), (58, 62)),
        trace_other=((50, 52), (56, 58), (62, 69)),
        trace_markers=((52, 53), (58, 59)),
        trace_nonmarkers=((53, 56), (59, 62)),
        earlier_trace_items=((52, 56),),
        terminal_trace_item=((58, 62),),
    )
    registry.validate()
    banks = _registered_ordinary_corruption_banks(registry)
    prompt_records = set(registry.positions("prompt_records"))
    flattened = [
        {position for start, end in bank for position in range(start, end)}
        for bank in banks
    ]
    assert all(len(values) == 8 for values in flattened)
    assert all(not values & prompt_records for values in flattened)
    assert not flattened[0] & flattened[1]
    assert not flattened[0] & flattened[2]
    assert not flattened[1] & flattened[2]


def test_source_attention_metrics_separate_mass_and_breadth() -> None:
    attention = torch.tensor(
        [
            [0.2, 0.2, 0.1, 0.1, 0.2, 0.2],
            [0.4, 0.4, 0.1, 0.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    metrics = source_attention_metrics(
        attention,
        key_start=0,
        spans=((0, 2), (4, 6)),
    )
    assert metrics[0]["mass"] == pytest.approx(0.8)
    assert metrics[0]["coverage"] == pytest.approx(1.0)
    assert metrics[0]["broad_score"] == pytest.approx(0.8)
    assert metrics[1]["mass"] == pytest.approx(0.8)
    assert metrics[1]["coverage"] == pytest.approx(0.5)
    assert metrics[1]["broad_score"] == pytest.approx(0.4)


def _capture_frame() -> pd.DataFrame:
    rows = []
    for seed in (1, 2):
        for request in ("a", "b"):
            for layer in (0, 1):
                for head in range(4):
                    score = 0.01
                    if (layer, head) == (0, 0):
                        score = 1.0
                    elif (layer, head) == (1, 0):
                        score = 0.9
                    rows.append(
                        {
                            "request_id": f"{seed}-{request}",
                            "model_label": "Qwen3-8B",
                            "seed": seed,
                            "layer": layer,
                            "head": head,
                            "layer_head_count": 4,
                            "trace_items_broad_score": score,
                        }
                    )
    return pd.DataFrame(rows)


def test_broad_head_plan_freezes_treatment_before_matched_controls() -> None:
    ranking = rank_answer_broad_heads(
        _capture_frame(),
        source_group="trace_items",
        development_seeds=(1, 2),
        model_label="Qwen3-8B",
    )
    assert list(
        ranking.head(2)[["layer", "head"]].itertuples(index=False, name=None)
    ) == [(0, 0), (1, 0)]
    plan = build_answer_broad_head_plan(
        ranking,
        bank_size=2,
        random_controls=3,
        random_seed=17,
    )
    selected = json.loads(
        plan.loc[plan["condition"].eq("selected_bank"), "heads"].iloc[0]
    )
    assert selected == [[0, 0], [1, 0]]
    random_rows = plan.loc[plan["condition"].eq("layer_matched_random")]
    assert len(random_rows) == 3
    assert random_rows["bank_sha256"].nunique() == 3
    for raw in random_rows["heads"]:
        control = json.loads(raw)
        assert len(control) == 2
        assert sorted(layer for layer, _head in control) == [0, 1]
        assert not {tuple(value) for value in control} & {(0, 0), (1, 0)}


def test_control_basis_is_orthonormal_and_deterministic() -> None:
    rng = np.random.default_rng(4)
    basis, _ = np.linalg.qr(rng.standard_normal((12, 3)))
    left = deterministic_control_basis(basis, seed=9)
    right = deterministic_control_basis(basis, seed=9)
    np.testing.assert_allclose(left, right)
    np.testing.assert_allclose(left.T @ left, np.eye(3), atol=2e-5)
    np.testing.assert_allclose(basis.T @ left, np.zeros((3, 3)), atol=2e-5)


def test_count_stream_basis_uses_label_centroids() -> None:
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(1, 5), 6)
    states = rng.normal(scale=0.05, size=(len(labels), 10))
    states[:, 0] += labels
    center, basis, control = fit_count_stream_basis(
        states.astype(np.float32), labels, rank=2, seed=3
    )
    assert center.shape == (10,)
    assert basis.shape == (10, 2)
    assert control.shape == (10, 2)
    np.testing.assert_allclose(basis.T @ control, 0.0, atol=2e-5)


def test_trace_patch_pair_plan_is_directed_and_strictly_intermediate() -> None:
    specifications = [
        {
            "from_occurrence": occurrence,
            "anchor_equivalence_id": f"item-end-{occurrence}",
            "target_city": f"city-{occurrence + 1}",
        }
        for occurrence in range(1, 5)
    ]
    plan = build_trace_patch_pair_plan(specifications, gold_count=5)
    assert [
        (value["donor_occurrence"], value["receiver_occurrence"]) for value in plan
    ] == [(1, 2), (3, 2), (2, 3), (4, 3), (3, 4)]
    assert {value["donor_direction"] for value in plan} == {
        "past_to_later_receiver",
        "future_to_earlier_receiver",
    }
    assert all(1 < value["receiver_occurrence"] < 5 for value in plan)
    with pytest.raises(ValueError, match="strictly intermediate"):
        build_trace_patch_pair_plan(
            specifications,
            gold_count=5,
            receiver_occurrences=(1,),
        )


def test_sparse_trace_patch_plan_freezes_330_local_plus_20_terminal_pairs() -> None:
    rows = [
        {
            "request_id": f"request-{seed}-{count}",
            "model_label": "Qwen3-8B",
            "seed": seed,
            "gold_count": count,
            # Deliberately different outcomes: sampling must not inspect them.
            "prediction": (seed + count) % 10,
            "exact_count": bool((seed + count) % 2),
        }
        for seed in range(100, 130)
        for count in range(1, 11)
    ]
    plan = build_sparse_trace_patch_sample_plan(
        rows,
        model_label="Qwen3-8B",
        seeds_per_cell=10,
        sampling_seed=17,
    )
    repeated = build_sparse_trace_patch_sample_plan(
        [{**row, "prediction": -999, "exact_count": False} for row in rows],
        model_label="Qwen3-8B",
        seeds_per_cell=10,
        sampling_seed=17,
    )
    assert list(plan["pair_sha256"]) == list(repeated["pair_sha256"])
    assert len(plan) == 350
    assert int(plan["panel_kind"].eq("local").sum()) == 330
    assert int(plan["panel_kind"].eq("terminal").sum()) == 20
    cell_sizes = plan.groupby(
        ["panel_kind", "gold_count", "donor_offset"]
    ).size()
    assert len(cell_sizes) == 35
    assert set(cell_sizes) == {10}
    assert plan.groupby(
        ["panel_kind", "gold_count", "donor_offset"]
    )["seed"].nunique().eq(10).all()
    expected_min_count = {-5: 7, -3: 5, -1: 3, 1: 4, 3: 6, 5: 8}
    local = plan.loc[plan["panel_kind"].eq("local")]
    for offset, minimum in expected_min_count.items():
        assert sorted(local.loc[local["donor_offset"].eq(offset), "gold_count"].unique()) == list(
            range(minimum, 11)
        )
    for _cell, frame in local.groupby(
        ["gold_count", "donor_offset"], sort=True
    ):
        frequencies = frame["receiver_occurrence"].value_counts()
        assert int(frequencies.max() - frequencies.min()) <= 1
    terminal = plan.loc[plan["panel_kind"].eq("terminal")]
    assert set(
        terminal[["donor_occurrence", "receiver_occurrence"]].itertuples(
            index=False, name=None
        )
    ) == {(1, 2), (2, 1)}
    assert not terminal["local_next_city_outcome_registered"].any()


def test_terminal_last_plan_freezes_19_natural_receiver_cells() -> None:
    rows = [
        {
            "request_id": f"terminal-{seed}-{count}",
            "model_label": "Qwen3-8B",
            "seed": seed,
            "gold_count": count,
            "prediction": (seed * count) % 11,
        }
        for seed in range(100, 130)
        for count in range(1, 11)
    ]
    plan = build_terminal_last_trace_patch_sample_plan(
        rows,
        model_label="Qwen3-8B",
        seeds_per_cell=10,
        sampling_seed=23,
    )
    repeated = build_terminal_last_trace_patch_sample_plan(
        [{**row, "prediction": -999} for row in rows],
        model_label="Qwen3-8B",
        seeds_per_cell=10,
        sampling_seed=23,
    )
    assert list(plan["pair_sha256"]) == list(repeated["pair_sha256"])
    assert len(plan) == 190
    assert plan.groupby(["gold_count", "donor_offset"]).size().eq(10).all()
    assert plan.groupby(["gold_count", "donor_offset"])["seed"].nunique().eq(10).all()
    expected = {
        -1: list(range(2, 11)),
        -3: list(range(5, 11)),
        -5: list(range(7, 11)),
    }
    for offset, counts in expected.items():
        observed = sorted(
            plan.loc[plan["donor_offset"].eq(offset), "gold_count"].unique()
        )
        assert observed == counts
    assert plan["receiver_occurrence"].eq(plan["gold_count"]).all()
    assert (
        plan["donor_occurrence"]
        == plan["receiver_occurrence"] + plan["donor_offset"]
    ).all()
    assert set(plan["donor_direction"]) == {"past_to_later_receiver"}
    assert plan["receiver_is_terminal"].all()
    assert not plan["local_next_city_outcome_registered"].any()


def test_full_state_patch_geometry_is_exact_and_never_interpolates() -> None:
    registry = _registry()
    receiver, donor, audit = trace_patch_geometry_positions(
        registry,
        receiver_occurrence=2,
        donor_occurrence=1,
        geometry="endpoint",
    )
    assert receiver == (34,)
    assert donor == (29,)
    assert audit["patch_token_count"] == 1
    receiver, donor, audit = trace_patch_geometry_positions(
        registry,
        receiver_occurrence=2,
        donor_occurrence=1,
        geometry="full_span",
    )
    assert receiver == (31, 32, 33, 34)
    assert donor == (26, 27, 28, 29)
    assert audit["patch_position_alignment"] == "right_aligned_relative_token_index"
    with pytest.raises(ValueError, match="not applicable"):
        trace_patch_geometry_positions(
            registry,
            receiver_occurrence=2,
            donor_occurrence=1,
            geometry="suffix8",
        )


def test_full_state_patch_layer_modes_match_one_shot_and_html_clamp() -> None:
    assert _full_state_patch_layers(
        source_layer=3, num_layers=7, layer_mode="one_shot"
    ) == (3,)
    assert _full_state_patch_layers(
        source_layer=3, num_layers=7, layer_mode="cumulative_clamp"
    ) == (3, 4, 5, 6)
    with pytest.raises(ValueError, match="leave a downstream"):
        _full_state_patch_layers(
            source_layer=6, num_layers=7, layer_mode="one_shot"
        )


def test_html_aligned_control_replaces_all_trace_items_without_retokenizing() -> None:
    registry = _registry()
    encoding = NativeTraceEncoding(
        stimulus_id="stimulus-1",
        request_id=registry.request_id,
        design_variant="native_thinking",
        seed=1234,
        split="discovery",
        count=2,
        model_label="Qwen3-8B",
        model_family="qwen3",
        answer_format="number",
        text="fixture",
        generation_prompt="fixture",
        input_ids=tuple(range(registry.sequence_length)),
        attention_mask=(1,) * registry.sequence_length,
        query_position=registry.query_position,
        prompt_token_count=registry.prompt_token_count,
        raw_prefix_text="fixture",
        selected_site={},
        prompt_record_spans=(),
        trace_item_spans=(),
        slot_spans=(),
        needle_spans=(),
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )
    tokenizer = SimpleNamespace(all_special_ids=())
    control, audit = build_html_aligned_uninformative_trace_encoding(
        encoding, registry, tokenizer, random_seed=17
    )
    repeated, repeated_audit = build_html_aligned_uninformative_trace_encoding(
        encoding, registry, tokenizer, random_seed=17
    )
    assert control.input_ids == repeated.input_ids
    assert audit == repeated_audit
    assert len(control.input_ids) == len(encoding.input_ids)
    assert control.attention_mask == encoding.attention_mask
    assert control.input_ids[: registry.prompt_token_count] == encoding.input_ids[
        : registry.prompt_token_count
    ]
    assert all(
        control.input_ids[position] != encoding.input_ids[position]
        for position in registry.positions("trace_items")
    )
    assert audit["control_retokenized"] is False
    assert audit["all_trace_items_replaced"] is True
    assert audit["outcome_fields_accessed"] is False


def test_valid_trace_patch_receiver_ranges_are_signed() -> None:
    assert valid_trace_patch_receivers(3, -1) == (2,)
    assert valid_trace_patch_receivers(3, 1) == ()
    assert valid_trace_patch_receivers(5, -3) == (4,)
    assert valid_trace_patch_receivers(5, 3) == ()
    assert valid_trace_patch_receivers(8, 5) == (2,)


def test_trace_patch_conditions_isolate_progress_and_match_control_norm() -> None:
    receiver = np.zeros(4, dtype=np.float32)
    donor = np.asarray([2.0, 3.0, 0.0, 0.0], dtype=np.float32)
    basis = np.asarray([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    states, audit = trace_patch_condition_states(
        receiver,
        donor,
        basis,
        random_seed=19,
    )
    np.testing.assert_allclose(states["self_patch"], receiver)
    np.testing.assert_allclose(states["full_donor_patch"], donor)
    np.testing.assert_allclose(
        states["progress_projected_patch"],
        np.asarray([2.0, 0.0, 0.0, 0.0]),
    )
    orthogonal = np.asarray(states["norm_matched_orthogonal_patch"])
    assert orthogonal[0] == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(orthogonal) == pytest.approx(2.0, rel=1e-5)
    assert audit["progress_projected_delta_norm"] == pytest.approx(2.0)
    assert audit["orthogonal_control_delta_norm"] == pytest.approx(2.0)


def test_stream_retention_metric_uses_later_item_bases_not_answer_query() -> None:
    clean = np.zeros((2, 3, 4), dtype=np.float32)
    aligned = clean.copy()
    aligned[0, :2, 0] = 2.0
    aligned[1, :2, 1] = 1.0
    aligned[:, 2, 3] = 5.0
    orthogonal = clean.copy()
    orthogonal[0, :2, 2] = 2.0
    orthogonal[1, :2, 3] = 1.0
    bases = {
        4: np.asarray([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32),
        7: np.asarray([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32),
    }
    aligned_metrics = stream_state_retention_metrics(
        clean,
        aligned,
        readout_layers=(4, 7),
        readout_positions=(5, 8, 10),
        query_position=10,
        count_bases=bases,
    )
    orthogonal_metrics = stream_state_retention_metrics(
        clean,
        orthogonal,
        readout_layers=(4, 7),
        readout_positions=(5, 8, 10),
        query_position=10,
        count_bases=bases,
    )
    assert aligned_metrics["downstream_item_readout_count"] == 4
    assert aligned_metrics[
        "downstream_item_progress_subspace_displacement_rms"
    ] == pytest.approx(np.sqrt(2.5))
    assert orthogonal_metrics[
        "downstream_item_progress_subspace_displacement_rms"
    ] == pytest.approx(0.0)
    assert aligned_metrics["answer_query_projected_into_item_basis"] is False
    assert aligned_metrics["answer_query_full_state_displacement_rms"] == 5.0


def test_mechanism_spec_does_not_relabel_seen_seeds_as_confirmation() -> None:
    spec = NativeCountMechanismSpec()
    spec.validate()
    assert spec.development_seeds == tuple(range(1234, 1254))
    assert spec.confirmation_seeds == tuple(range(1254, 1264))
    assert spec.seed_role(1258) == "confirmation"
    assert not spec.formal_inference_eligible
    assert NativeCountMechanismSpec.from_mapping(spec.to_dict()) == spec
    with pytest.raises(ValueError, match="disjoint"):
        NativeCountMechanismSpec(
            status="frozen_confirmation",
            development_seeds=(1, 2),
            confirmation_seeds=(2, 3),
        ).validate()
    with pytest.raises(ValueError, match="20 seeds"):
        NativeCountMechanismSpec(
            development_seeds=tuple(range(1204, 1234)),
        ).validate()
    with pytest.raises(ValueError, match="10 seeds"):
        NativeCountMechanismSpec(
            confirmation_seeds=tuple(range(1316, 1336)),
        ).validate()


def test_broad_discovery_roles_and_odd_even_panels_are_disjoint() -> None:
    spec = NativeCountMechanismSpec()
    spec.validate()
    assert spec.broad_phase(1234) == "ranking_discovery"
    assert spec.broad_phase(1244) == "k_selection_discovery"
    assert spec.broad_counts_for_seed(1234, phase="ranking_discovery") == tuple(
        range(1, 11)
    )
    assert spec.broad_counts_for_seed(1244, phase="k_selection_discovery") == (
        1,
        3,
        5,
        7,
        9,
    )
    assert spec.broad_phase(1263) == "confirmation"
    assert spec.broad_counts_for_seed(1263, phase="confirmation") == (
        2,
        4,
        6,
        8,
        10,
    )


def _broad_k_trials(selected_shifts: dict[int, float]) -> pd.DataFrame:
    rows = []
    for bank_size, shift in selected_shifts.items():
        for seed in (1, 2, 3, 4):
            for request_index in (0, 1):
                request_id = f"{seed}-{request_index}"
                common = {
                    "experiment_id": "answer_broad_head_ablation",
                    "model_label": "Qwen3-8B",
                    "source_group": "trace_items",
                    "bank_size": bank_size,
                    "request_id": request_id,
                    "seed": seed,
                    "status": "ok",
                }
                rows.append(
                    {
                        **common,
                        "condition": "clean",
                        "repeat": 0,
                        "expected_count": 5.0,
                        "correct_count_margin": 2.0,
                    }
                )
                rows.append(
                    {
                        **common,
                        "condition": "selected_bank",
                        "repeat": 0,
                        "expected_count": 5.0 + shift,
                        "correct_count_margin": 2.0 - shift,
                    }
                )
                for repeat in (1, 2, 3):
                    rows.append(
                        {
                            **common,
                            "condition": "layer_matched_random",
                            "repeat": repeat,
                            "expected_count": 5.1,
                            "correct_count_margin": 1.9,
                        }
                    )
    return pd.DataFrame(rows)


def test_broad_k_selection_uses_smallest_one_se_positive_bank() -> None:
    curve, seed_effects, decision = select_answer_broad_bank_size(
        _broad_k_trials({1: 0.2, 2: 1.0, 4: 0.8}),
        model_label="Qwen3-8B",
        source_group="trace_items",
        expected_seeds=(1, 2, 3, 4),
        expected_bank_sizes=(1, 2, 4),
        expected_requests_per_seed=2,
        expected_random_controls=3,
        boundary_extension_bank_size=8,
        bootstrap_samples=100,
        random_seed=3,
    )
    assert list(curve["bank_size"]) == [1, 2, 4]
    assert len(seed_effects) == 12
    assert decision["status"] == "frozen_for_confirmation"
    assert decision["selected_bank_size"] == 2
    assert decision["confirmation_outcomes_used"] is False


def test_broad_k_selection_requires_boundary_extension_when_curve_is_rising() -> None:
    _curve, _seed_effects, decision = select_answer_broad_bank_size(
        _broad_k_trials({1: 0.2, 2: 0.6, 4: 1.1}),
        model_label="Qwen3-8B",
        source_group="trace_items",
        expected_seeds=(1, 2, 3, 4),
        expected_bank_sizes=(1, 2, 4),
        expected_requests_per_seed=2,
        expected_random_controls=3,
        boundary_extension_bank_size=8,
        bootstrap_samples=100,
        random_seed=4,
    )
    assert decision["status"] == "requires_boundary_extension"
    assert decision["selected_bank_size"] is None
    assert decision["required_next_bank_size"] == 8


def test_native_cohort_filter_keeps_correctness_out_of_primary_selection() -> None:
    row = {
        "trace_parse": {
            "exact_count": False,
            "parser": {"detected": True, "trace_one_to_one": True},
        }
    }
    assert _cohort_exclusion_reason(row, "parser_hit") is None
    assert _cohort_exclusion_reason(row, "one_to_one") is None
    assert (
        _cohort_exclusion_reason(row, "one_to_one_correct") == "final_count_incorrect"
    )


def test_trace_patch_cli_registers_full_control_panel() -> None:
    args = build_parser().parse_args(
        [
            "trace-patch",
            "--model",
            "Qwen3-8B",
            "--generations",
            "generations.jsonl",
            "--pair-plan",
            "pair-plan.csv",
            "--basis",
            "basis.npz",
            "--layer",
            "20",
            "--output",
            "trace-patch-output",
        ]
    )
    assert args.command == "trace-patch"
    assert args.max_new_tokens == 48
    assert args.conditions == [
        "clean",
        "self_patch",
        "full_donor_patch",
        "progress_projected_patch",
        "norm_matched_orthogonal_patch",
    ]


def test_full_state_patch_cli_freezes_geometry_and_layer_modes() -> None:
    args = build_parser().parse_args(
        [
            "trace-full-state-patch",
            "--model",
            "Qwen3-8B",
            "--generations",
            "generations.jsonl",
            "--pair-plan",
            "pair-plan.csv",
            "--plan-kind",
            "terminal_last",
            "--basis",
            "basis.npz",
            "--layer",
            "18",
            "--output",
            "full-state-output",
        ]
    )
    assert args.command == "trace-full-state-patch"
    assert args.max_new_tokens == 16
    assert args.geometries == ["endpoint", "suffix4", "suffix8", "full_span"]
    assert args.layer_modes == ["one_shot", "cumulative_clamp"]
    assert args.conditions == ["clean", "self_patch", "full_donor_patch"]


def test_integrated_serial_bridge_resumes_by_default() -> None:
    base_args = [
        "integrated-serial-bridge",
        "--model",
        "Qwen3-8B",
        "--generations",
        "generations.jsonl",
        "--bank-plan",
        "bank-plan.csv",
        "--anchor-registry",
        "anchor-registry.jsonl",
        "--output",
        "integrated-output",
    ]
    assert build_parser().parse_args(base_args).resume is True
    assert build_parser().parse_args([*base_args, "--no-resume"]).resume is False


def test_terminal_last_plan_cli_defaults_to_development() -> None:
    args = build_parser().parse_args(
        [
            "plan-terminal-last-patch",
            "--model",
            "Gemma4-E4B",
            "--generations",
            "generations.jsonl",
            "--output",
            "terminal-plan",
        ]
    )
    assert args.seed_role == "development"
    assert args.row_panel == "trace_patch"
    assert args.seeds_per_cell is None


def test_source_mask_cli_defaults_to_answer_query_only_all_head_assay() -> None:
    args = build_parser().parse_args(
        [
            "source-mask",
            "--model",
            "Qwen3-8B",
            "--generations",
            "generations.jsonl",
            "--output",
            "source-mask-output",
        ]
    )
    assert args.mask_application == "answer_query_only"


def test_serial_patch_source_cli_defaults_to_persistent_rank_free_assay() -> None:
    args = build_parser().parse_args(
        [
            "serial-patch-source",
            "--model",
            "Qwen3-8B",
            "--generations",
            "generations.jsonl",
            "--layer",
            "19",
            "--output",
            "serial-source-output",
        ]
    )
    assert args.command == "serial-patch-source"
    assert args.mask_application == "answer_query_and_answer_tokens"
    assert args.geometry == "suffix8"
    assert args.layer_mode == "cumulative_clamp"
    assert args.mask_conditions == [
        "clean",
        "block_trace_items",
        "block_trace_items_matched_control",
        "block_prompt_records",
        "block_prompt_records_matched_control",
    ]


def test_terminal_relay_cli_uses_frozen_rank_free_panel() -> None:
    args = build_parser().parse_args(
        [
            "terminal-relay-mediation",
            "--model",
            "Qwen3-8B",
            "--generations",
            "generations.jsonl",
            "--source-layer",
            "19",
            "--relay-layer",
            "26",
            "--output",
            "relay-output",
        ]
    )
    assert args.command == "terminal-relay-mediation"
    assert args.source_layer == 19
    assert args.relay_layer == 26
    assert args.geometry == "suffix8"
    assert args.limit is None
    assert args.max_new_tokens == 16


def test_trace_patch_basis_manifest_freezes_site_and_label(tmp_path) -> None:
    artifact = tmp_path / "basis.npz"
    np.savez_compressed(artifact, basis_L1=np.eye(2, dtype=np.float32))
    manifest = artifact.with_suffix(".json")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = {
        "artifact_sha256": digest,
        "site_kind": "item_end",
        "label": "occurrence",
        "confirmation_used_for_fit": False,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path, loaded = _validate_trace_patch_basis_manifest(artifact)
    assert manifest_path == manifest
    assert loaded["label"] == "occurrence"

    payload["label"] = "gold_count"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="occurrence label"):
        _validate_trace_patch_basis_manifest(artifact)


def test_registered_contrasts_are_request_first_and_seed_equal() -> None:
    rows = []
    values = {
        1: {
            "clean": 5.0,
            "block_trace_context": 2.0,
            "block_trace_context_matched_control": 4.0,
        },
        2: {
            "clean": 7.0,
            "block_trace_context": 6.0,
            "block_trace_context_matched_control": 7.0,
        },
    }
    for seed, conditions in values.items():
        for condition, value in conditions.items():
            rows.append(
                {
                    "experiment_id": "answer_source_mask_factorial",
                    "condition": condition,
                    "model_label": "Qwen3-8B",
                    "request_id": f"r{seed}",
                    "seed": seed,
                    "correct_count_margin": value,
                }
            )
    summary, seed_effects = summarize_linear_contrasts(
        pd.DataFrame(rows),
        experiment_id="answer_source_mask_factorial",
        outcome="correct_count_margin",
        bootstrap_samples=200,
    )
    trace = summary.loc[summary["contrast"].eq("trace_damage")].iloc[0]
    specificity = summary.loc[summary["contrast"].eq("trace_source_specificity")].iloc[
        0
    ]
    assert trace["mean_effect"] == pytest.approx(2.0)
    assert specificity["mean_effect"] == pytest.approx(1.5)
    assert np.isfinite(trace["p_value_holm_across_models"])
    assert len(seed_effects.loc[seed_effects["contrast"].eq("trace_damage")]) == 2


def test_restoration_specificity_is_a_difference_of_repairs() -> None:
    rows = []
    values = {
        "trace_token_corrupt": 0.0,
        "trace_corrupt_full_span_restore": 3.0,
        "ordinary_token_corrupt": 1.0,
        "ordinary_corrupt_ordinary_state_restore": 2.0,
    }
    for seed in (1, 2):
        for condition, value in values.items():
            rows.append(
                {
                    "experiment_id": "trace_source_restoration",
                    "condition": condition,
                    "model_label": "Qwen3-8B",
                    "request_id": f"restoration-{seed}",
                    "seed": seed,
                    "correct_count_margin": value,
                }
            )
    summary, _seed_effects = summarize_linear_contrasts(
        pd.DataFrame(rows),
        experiment_id="trace_source_restoration",
        outcome="correct_count_margin",
        bootstrap_samples=100,
    )
    specificity = summary.loc[
        summary["contrast"].eq("full_span_vs_ordinary_repair_specificity")
    ].iloc[0]
    assert specificity["mean_effect"] == pytest.approx(2.0)


def test_decision_ledger_requires_every_component_gate() -> None:
    summary = pd.DataFrame(
        [
            {
                "experiment_id": "stream_state_retention",
                "contrast": "aligned_vs_orthogonal_specificity",
                "model_label": "Qwen3-8B",
                "mean_effect": 1.0,
                "ci_low": 0.2,
            }
        ]
    )
    ledger = mechanism_decision_ledger(summary)
    stream = ledger.loc[ledger["claim"].eq("stream_written_state")].iloc[0]
    assert stream["status"] == "not_established"
    assert "full_donor_vs_self_transport" in stream["missing_gates"]


def test_decision_ledger_requires_forward_trace_transport() -> None:
    rows = []
    for contrast in (
        "full_donor_vs_self_transport",
        "projected_vs_orthogonal_transport",
    ):
        rows.append(
            {
                "experiment_id": "trace_intermediate_state_patching",
                "contrast": contrast,
                "donor_direction": "past_to_later_receiver",
                "model_label": "Qwen3-8B",
                "mean_effect": 1.0,
                "ci_low": 0.2,
            }
        )
    ledger = mechanism_decision_ledger(pd.DataFrame(rows))
    stream = ledger.loc[ledger["claim"].eq("stream_written_state")].iloc[0]
    assert stream["status"] == "passes_registered_gate"
    assert "past_to_later_receiver" in stream["observed_gates"]

    future = pd.DataFrame(rows).assign(donor_direction="future_to_earlier_receiver")
    future_ledger = mechanism_decision_ledger(future)
    future_stream = future_ledger.loc[
        future_ledger["claim"].eq("stream_written_state")
    ].iloc[0]
    assert future_stream["status"] == "not_established"


def test_decision_ledger_keeps_trace_and_prompt_broad_banks_separate() -> None:
    rows = [
        {
            "experiment_id": "answer_source_mask_factorial",
            "contrast": "trace_source_specificity",
            "model_label": "Qwen3-8B",
            "mean_effect": 1.0,
            "ci_low": 0.2,
        },
        {
            "experiment_id": "answer_source_mask_factorial",
            "contrast": "prompt_source_specificity",
            "model_label": "Qwen3-8B",
            "mean_effect": 1.0,
            "ci_low": 0.2,
        },
        {
            "experiment_id": "answer_broad_head_ablation",
            "contrast": "selected_vs_layer_matched_random",
            "source_group": "trace_items",
            "model_label": "Qwen3-8B",
            "mean_effect": 0.8,
            "ci_low": 0.1,
        },
        {
            "experiment_id": "answer_broad_head_ablation",
            "contrast": "selected_vs_layer_matched_random",
            "source_group": "prompt_records",
            "model_label": "Qwen3-8B",
            "mean_effect": -0.4,
            "ci_low": -0.9,
        },
    ]
    ledger = mechanism_decision_ledger(pd.DataFrame(rows))
    trace = ledger.loc[ledger["claim"].eq("answer_time_trace_retrieval")].iloc[0]
    prompt = ledger.loc[ledger["claim"].eq("answer_time_prompt_retrieval")].iloc[0]
    assert trace["status"] == "passes_registered_gate"
    assert prompt["status"] == "not_established"
    assert "trace_items" in trace["observed_gates"]
    assert "prompt_records" not in trace["observed_gates"]


def test_decision_ledger_rejects_ambiguous_duplicate_gate_rows() -> None:
    rows = [
        {
            "experiment_id": "answer_source_mask_factorial",
            "contrast": "trace_source_specificity",
            "model_label": "Qwen3-8B",
            "mean_effect": 1.0,
            "ci_low": 0.2,
        },
    ]
    for bank_size in (8, 16):
        rows.append(
            {
                "experiment_id": "answer_broad_head_ablation",
                "contrast": "selected_vs_layer_matched_random",
                "source_group": "trace_items",
                "bank_size": bank_size,
                "model_label": "Qwen3-8B",
                "mean_effect": 0.8,
                "ci_low": 0.1,
            }
        )
    ledger = mechanism_decision_ledger(pd.DataFrame(rows))
    trace = ledger.loc[ledger["claim"].eq("answer_time_trace_retrieval")].iloc[0]
    assert trace["status"] == "not_established"
    assert "selected_vs_layer_matched_random" in trace["ambiguous_gates"]


def test_analyze_command_writes_registered_outputs(tmp_path) -> None:
    mechanism_config = tmp_path / "mechanism.json"
    mechanism_config.write_text(
        json.dumps(NativeCountMechanismSpec(bootstrap_samples=100).to_dict()),
        encoding="utf-8",
    )
    trials = tmp_path / "trials.jsonl"
    rows = []
    for seed in (1, 2):
        for condition, value in {
            "clean": 2.0,
            "block_trace_context": 0.5,
            "block_trace_context_matched_control": 1.5,
        }.items():
            rows.append(
                {
                    "experiment_id": "answer_source_mask_factorial",
                    "condition": condition,
                    "model_label": "Qwen3-8B",
                    "request_id": f"request-{seed}",
                    "seed": seed,
                    "correct_count_margin": value,
                    "status": "ok",
                }
            )
        for source_group, conditions in {
            "trace_items": {
                "clean": 2.0,
                "selected_bank": 0.0,
                "layer_matched_random": 1.5,
            },
            "prompt_records": {
                "clean": 2.0,
                "selected_bank": 1.0,
                "layer_matched_random": 1.5,
            },
        }.items():
            for condition, value in conditions.items():
                rows.append(
                    {
                        "experiment_id": "answer_broad_head_ablation",
                        "condition": condition,
                        "source_group": source_group,
                        "bank_size": 8,
                        "model_label": "Qwen3-8B",
                        "request_id": f"request-{seed}",
                        "seed": seed,
                        "correct_count_margin": value,
                        "status": "ok",
                    }
                )
    trials.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    args = build_parser().parse_args(
        [
            "analyze",
            "--mechanism-config",
            str(mechanism_config),
            "--trials",
            str(trials),
            "--output",
            str(output),
        ]
    )
    args.func(args)
    assert (output / "estimands.csv").is_file()
    assert (output / "seed_effects.csv").is_file()
    assert (output / "mechanism_decision_ledger.csv").is_file()
    ledger = pd.read_csv(output / "mechanism_decision_ledger.csv")
    assert set(ledger["claim_scope"]) == {"development_only_no_confirmatory_claim"}
    estimands = pd.read_csv(output / "estimands.csv")
    broad = estimands.loc[estimands["experiment_id"].eq("answer_broad_head_ablation")]
    assert set(broad["source_group"]) == {"trace_items", "prompt_records"}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == "analyze"
    assert manifest["formal_inference_eligible"] is False


def test_analyze_trace_patch_separates_temporal_directions(tmp_path) -> None:
    mechanism_config = tmp_path / "mechanism.json"
    mechanism_config.write_text(
        json.dumps(NativeCountMechanismSpec(bootstrap_samples=100).to_dict()),
        encoding="utf-8",
    )
    trials = tmp_path / "trace-patch.jsonl"
    rows = []
    for seed in range(1, 9):
        for direction in (
            "past_to_later_receiver",
            "future_to_earlier_receiver",
        ):
            values = {
                "clean": 0.0,
                "self_patch": 0.0,
                "full_donor_patch": 2.0,
                "progress_projected_patch": 1.5,
                "norm_matched_orthogonal_patch": 0.0,
            }
            for condition, value in values.items():
                rows.append(
                    {
                        "experiment_id": "trace_intermediate_state_patching",
                        "condition": condition,
                        "donor_direction": direction,
                        "model_label": "Qwen3-8B",
                        "request_id": f"trace-{seed}",
                        "seed": seed,
                        "donor_vs_receiver_city_log_odds": value,
                        "status": "ok",
                    }
                )
    trials.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    args = build_parser().parse_args(
        [
            "analyze",
            "--mechanism-config",
            str(mechanism_config),
            "--trials",
            str(trials),
            "--experiment-ids",
            "trace_intermediate_state_patching",
            "--outcome",
            "donor_vs_receiver_city_log_odds",
            "--output",
            str(output),
        ]
    )
    args.func(args)
    estimands = pd.read_csv(output / "estimands.csv")
    assert set(estimands["donor_direction"]) == {
        "past_to_later_receiver",
        "future_to_earlier_receiver",
    }
    ledger = pd.read_csv(output / "mechanism_decision_ledger.csv")
    stream = ledger.loc[ledger["claim"].eq("stream_written_state")].iloc[0]
    assert stream["status"] == "passes_registered_gate"
