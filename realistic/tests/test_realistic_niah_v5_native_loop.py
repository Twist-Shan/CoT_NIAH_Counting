from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scripts.analyze_realistic_niah_v5_native_loop import analyze
from scripts.analyze_realistic_niah_v5_query_mediation import (
    analyze as analyze_query_mediation,
)
from scripts.audit_realistic_niah_v5_query_mediation_span_eligibility import (
    audit as audit_span_eligibility,
)
from realistic_niah_v5.count_stream import (
    _sha256_json,
    trace_patch_geometry_positions,
)
from realistic_niah_v5.native_loop import (
    _score_trace_continuation_with_mediation_hooks,
    build_query_mediation_head_plan,
    build_fixed_native_loop_plan,
    load_frozen_query_mediation_head_plan,
    load_frozen_targeted_bank,
    native_loop_condition_states,
    validate_query_mediation_positions,
)


class _ContinuationHookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Identity()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        position_ids=None,
        cache_position=None,
    ):
        del attention_mask, past_key_values, use_cache, position_ids, cache_position
        sequence = int(input_ids.shape[1])
        values = torch.arange(
            sequence * 4, dtype=torch.float32, device=input_ids.device
        ).reshape(1, sequence, 4)
        values = self.projection(values)
        logits = torch.zeros((1, sequence, 8), device=input_ids.device)
        logits[..., 1] = values[..., 0]
        logits[..., 2] = values[..., 2]
        return SimpleNamespace(logits=logits)


def test_continuation_head_restore_is_path_matched():
    model = _ContinuationHookModel()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=(2,),
        head_dims=(2,),
        output_projections=(model.projection,),
    )
    encoding = SimpleNamespace(attention_mask=(1, 1, 1), sequence_length=3)
    prefill = SimpleNamespace(
        logits=torch.zeros((1, 1, 8)), past_key_values={"cache": [1]}
    )
    score, captures, audit = _score_trace_continuation_with_mediation_hooks(
        model,
        adapter,
        encoding,
        prefill,
        (1, 2, 1),
        city_token_offset=1,
        capture_heads=((0, 1),),
    )
    assert score["token_count"] == 3
    assert captures[0].shape == (1, 2, 4)
    assert audit["continuation_forced_input_token_count"] == 2

    restored, _captures, _audit = (
        _score_trace_continuation_with_mediation_hooks(
            model,
            adapter,
            encoding,
            prefill,
            (1, 2, 1),
            city_token_offset=1,
            capture_heads=((0, 1),),
            head_replacements=captures,
        )
    )
    assert restored == score

    try:
        _score_trace_continuation_with_mediation_hooks(
            model,
            adapter,
            encoding,
            prefill,
            (1, 2),
            city_token_offset=1,
            capture_heads=((0, 1),),
            head_replacements=captures,
        )
    except RuntimeError as exc:
        assert "Path-matched" in str(exc)
    else:
        raise AssertionError("Unequal path replacement shape was accepted")


def _rows(seeds=(1234, 1235)):
    return [
        {
            "request_id": f"Qwen3-8B/N{count}/seed{seed}",
            "model_label": "Qwen3-8B",
            "seed": seed,
            "gold_count": count,
            "gold_records": [
                {"city": f"city-{index}", "score": index}
                for index in range(1, count + 1)
            ],
        }
        for seed in seeds
        for count in range(2, 11)
    ]


def test_native_loop_plan_is_rank_free_and_seed_complete():
    plan = build_fixed_native_loop_plan(
        _rows(),
        model_label="Qwen3-8B",
        seeds=(1234, 1235),
        seed_role="development",
        donor_offsets=(-1, 1),
    )
    assert "selection_rank" not in plan.columns
    assert not plan["selection_rank_used"].astype(bool).any()
    assert not plan["pair_sha256"].duplicated().any()
    local = plan.loc[plan["panel_kind"].eq("p0_local")]
    for _cell, frame in local.groupby("donor_offset"):
        assert set(frame["seed"].astype(int)) == {1234, 1235}
    boundary = plan.loc[plan["panel_kind"].ne("p0_local")]
    for _cell, frame in boundary.groupby("panel_kind"):
        assert set(frame["seed"].astype(int)) == {1234, 1235}
    terminal = plan.loc[plan["panel_kind"].eq("terminal_injection")]
    assert (terminal["donor_occurrence"] == terminal["gold_count"]).all()
    nonterminal = plan.loc[plan["panel_kind"].eq("nonterminal_injection")]
    assert (nonterminal["receiver_occurrence"] == nonterminal["gold_count"]).all()


def test_native_loop_aligned_rows_choose_same_structural_pairs_across_models():
    def aligned_row(model: str, seed: int, count: int):
        return {
            "request_id": f"{model}/N{count}/seed{seed}",
            "model_label": model,
            "seed": seed,
            "gold_count": count,
            "alignment_pair_id": f"seed{seed}_N{count}",
            "cross_model_exact_sample_alignment": True,
            "gold_records": [
                {"city": f"city-{index}", "score": index}
                for index in range(1, count + 1)
            ],
        }

    qwen = build_fixed_native_loop_plan(
        [aligned_row("Qwen3-8B", 1234, 6), aligned_row("Qwen3-8B", 1235, 8)],
        model_label="Qwen3-8B",
        seeds=(1234, 1235),
        seed_role="development",
        donor_offsets=(-1, 1),
    )
    gemma = build_fixed_native_loop_plan(
        [aligned_row("Gemma4-E4B", 1234, 6), aligned_row("Gemma4-E4B", 1235, 8)],
        model_label="Gemma4-E4B",
        seeds=(1234, 1235),
        seed_role="development",
        donor_offsets=(-1, 1),
    )
    columns = [
        "panel_kind",
        "seed",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "alignment_pair_id",
    ]
    assert qwen[columns].to_dict("records") == gemma[columns].to_dict("records")
    assert qwen["cross_model_exact_sample_alignment"].all()
    assert gemma["cross_model_exact_sample_alignment"].all()


def test_native_loop_plan_uses_local_partial_fallback(monkeypatch):
    rows = [row for row in _rows(seeds=(1234,)) if row["gold_count"] <= 7]
    for row in rows:
        row["raw_output_text"] = f"count={row['gold_count']}"

    def fake_parse(_raw_text, *, model_family, gold_records):
        del model_family
        count = len(tuple(gold_records))
        full = count <= 4
        return SimpleNamespace(
            detected=True,
            item_count=count if full else count - 1,
            trace_one_to_one=full,
            trace_category="one_to_one" if full else "partial_unique",
        )

    monkeypatch.setattr(
        "realistic_niah_v5.native_loop.find_trace_count_sequence", fake_parse
    )
    plan = build_fixed_native_loop_plan(
        rows,
        model_label="Qwen3-8B",
        seeds=(1234,),
        seed_role="development",
        donor_offsets=(-3, 3),
    )
    local = plan.loc[plan["panel_kind"].eq("p0_local")]
    assert set(local["donor_offset"].astype(int)) == {-3, 3}
    assert set(local["local_cohort_policy"]) == {
        "partial_unique_local_transition_fallback"
    }
    boundary = plan.loc[plan["panel_kind"].ne("p0_local")]
    assert set(boundary["local_cohort_policy"]) == {"one_to_one_full_trace"}


def test_native_loop_condition_geometry_and_restoration():
    receiver = np.asarray([3.0, 4.0, 7.0, 8.0], dtype=np.float32)
    donor = np.asarray([5.0, 1.0, 9.0, 6.0], dtype=np.float32)
    center = np.zeros(4, dtype=np.float32)
    basis = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    states, audit = native_loop_condition_states(
        receiver, donor, center, basis, random_seed=7
    )
    np.testing.assert_allclose(
        states["count_subspace_transplant"].numpy()[:2], donor[:2]
    )
    np.testing.assert_allclose(
        states["count_subspace_transplant"].numpy()[2:], receiver[2:]
    )
    np.testing.assert_allclose(
        states["count_component_removed"].numpy()[:2], np.zeros(2)
    )
    np.testing.assert_allclose(
        states["count_component_restored"].numpy(), receiver
    )
    orthogonal_delta = (
        states["norm_matched_orthogonal_patch"].numpy() - receiver
    )
    np.testing.assert_allclose(orthogonal_delta[:2], np.zeros(2), atol=1e-6)
    assert audit["restoration_identity_max_abs_error"] == 0.0
    assert (
        audit["condition_audit"]["count_subspace_transplant"]
        ["condition_target_count_fraction"]
        == 1.0
    )


def test_load_frozen_targeted_bank_cross_audits_hash(tmp_path):
    heads = [[2, 1], [3, 0]]
    bank_hash = _sha256_json(heads)
    selection = {
        "model_label": "Qwen3-8B",
        "development_selection": {
            "primary_bank_size": 2,
            "primary_bank_heads": heads,
            "primary_bank_sha256": bank_hash,
        },
    }
    routing = {
        "policy_id": "test-route",
        "head_bank": {"selected_bank_sha256": bank_hash},
        "routes": {
            "adjacent_rank_before_city": {
                "required": ["post_marker"],
                "optional": [],
            }
        },
    }
    selection_path = tmp_path / "selection.json"
    routing_path = tmp_path / "routing.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    routing_path.write_text(json.dumps(routing), encoding="utf-8")
    loaded = load_frozen_targeted_bank(
        selection_path, routing_path, model_label="Qwen3-8B"
    )
    assert loaded["heads"] == ((2, 1), (3, 0))
    assert loaded["bank_sha256"] == bank_hash
    assert loaded["selection_rank_used"] is False


def test_load_frozen_targeted_bank_accepts_legacy_default_json_hash(tmp_path):
    heads = [[29, 4], [17, 2], [17, 1]]
    bank_hash = hashlib.sha256(json.dumps(heads).encode("utf-8")).hexdigest()
    selection = {
        "model_label": "Gemma4-E4B",
        "development_selection": {
            "primary_bank_size": len(heads),
            "primary_bank_heads": heads,
            "primary_bank_sha256": bank_hash,
        },
    }
    routing = {
        "policy_id": "legacy-route",
        "head_bank": {"selected_bank_sha256": bank_hash},
        "routes": {
            "adjacent_rank_after_city": {
                "required": ["p0_item_end"],
                "optional": [],
            }
        },
    }
    selection_path = tmp_path / "selection.json"
    routing_path = tmp_path / "routing.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    routing_path.write_text(json.dumps(routing), encoding="utf-8")
    loaded = load_frozen_targeted_bank(
        selection_path, routing_path, model_label="Gemma4-E4B"
    )
    assert loaded["bank_sha256"] == bank_hash
    assert loaded["bank_hash_encoding"] == "legacy_json_default"
    assert loaded["bank_membership_hashes"]["legacy_json_default"] == bank_hash


def test_query_mediation_head_plan_is_disjoint_and_layer_matched(tmp_path):
    targeted_bank = {
        "model_label": "Qwen3-8B",
        "heads": ((1, 0), (2, 0), (2, 1), (3, 2)),
        "bank_size": 4,
        "bank_sha256": "frozen-bank",
    }
    candidates = [
        (layer, head) for layer in range(1, 4) for head in range(6)
    ]
    plan = build_query_mediation_head_plan(
        targeted_bank,
        candidates,
        source_layer=1,
        random_seed=17,
        candidate_source_sha256="candidate-ranking",
    )
    selected = {tuple(value) for value in plan["active_selected_heads"]}
    random = {tuple(value) for value in plan["layer_matched_random_heads"]}
    assert selected == {(2, 0), (2, 1), (3, 2)}
    assert selected.isdisjoint(random)
    assert plan["layer_composition"] == {"2": 2, "3": 1}
    assert plan["selection_rank_used"] is False
    assert plan["outcome_blind"] is True

    path = tmp_path / "head_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_frozen_query_mediation_head_plan(
        path,
        targeted_bank,
        model_label="Qwen3-8B",
        source_layer=1,
    )
    assert loaded["active_selected_heads"] == ((2, 0), (2, 1), (3, 2))
    assert len(loaded["layer_matched_random_heads"]) == 3


def test_query_mediation_allows_offline_future_donor_capture():
    audit = validate_query_mediation_positions(
        (10,), (30,), query_position=20
    )
    assert audit["receiver_source_before_or_at_query"]
    assert audit["donor_capture_is_offline_counterfactual"]
    assert audit["donor_positions_after_receiver_query_count"] == 1

    try:
        validate_query_mediation_positions((21,), (10,), query_position=20)
    except ValueError as exc:
        assert "receiver position lies after" in str(exc)
    else:
        raise AssertionError("A post-query receiver intervention was accepted")


def test_native_loop_analysis_requires_all_three_links():
    rows = []
    for seed in range(1254, 1264):
        pair = f"p0-{seed}"
        for condition in (
            "count_subspace_transplant",
            "norm_matched_orthogonal_patch",
            "count_component_restored",
            "count_component_removed",
        ):
            positive = condition in {
                "count_subspace_transplant",
                "count_component_restored",
            }
            rows.append(
                {
                    "experiment_id": "p0_count_state_to_targeted_retrieval",
                    "model_label": "Qwen3-8B",
                    "seed": seed,
                    "mechanism_split": "confirmation",
                    "pair_sha256": pair,
                    "condition": condition,
                    "donor_offset": -1,
                    "selection_rank_used": False,
                    "donor_minus_receiver_successor_attention_mass": float(positive),
                    "donor_vs_receiver_city_log_odds": float(positive),
                    "donor_city_adoption": bool(positive),
                    "receiver_city_retention": bool(positive),
                }
            )
        for panel, outcome in (
            ("terminal_injection", "stopped_before_known_city"),
            ("nonterminal_injection", "donor_successor_adoption"),
        ):
            boundary_pair = f"{panel}-{seed}"
            for condition, value in (
                ("count_subspace_transplant", True),
                ("norm_matched_orthogonal_patch", False),
            ):
                rows.append(
                    {
                        "experiment_id": "endpoint_state_update_stop_transplant",
                        "model_label": "Qwen3-8B",
                        "seed": seed,
                        "mechanism_split": "confirmation",
                        "pair_sha256": boundary_pair,
                        "condition": condition,
                        "panel_kind": panel,
                        "selection_rank_used": False,
                        outcome: value,
                    }
                )
    _estimands, _effects, gates = analyze(
        pd.DataFrame(rows),
        phase="confirmation",
        bootstrap_samples=200,
        random_seed=11,
    )
    assert gates["commit_to_retrieval_pass"]
    assert gates["update_pass"]
    assert gates["stop_pass"]
    assert gates["native_loop_pass"]


def test_query_mediation_analysis_requires_state_mask_and_restore():
    rows = []
    values = {
        ("self_patch", "intact"): 0.0,
        ("self_patch", "selected_mask"): 0.0,
        ("self_patch", "layer_matched_random_mask"): 0.0,
        ("full_donor_patch", "intact"): 3.0,
        ("full_donor_patch", "selected_mask"): 0.5,
        ("full_donor_patch", "layer_matched_random_mask"): 2.8,
        ("count_subspace_transplant", "intact"): 2.0,
        ("count_subspace_transplant", "selected_mask"): 0.4,
        ("count_subspace_transplant", "layer_matched_random_mask"): 1.8,
        ("norm_matched_orthogonal_patch", "intact"): 0.0,
        ("norm_matched_orthogonal_patch", "selected_mask"): 0.0,
        ("norm_matched_orthogonal_patch", "layer_matched_random_mask"): 0.0,
        ("full_donor_patch_heads_into_self_patch", "selected_restore"): 2.5,
        (
            "count_subspace_transplant_heads_into_norm_matched_orthogonal_patch",
            "selected_restore",
        ): 1.6,
    }
    for seed in range(1234, 1254):
        for offset in (-3, -2, -1, 1, 2, 3):
            pair = f"{seed}:{offset}"
            for (state, head), value in values.items():
                rows.append(
                    {
                        "experiment_id": "p0_same_trajectory_query_mediation",
                        "model_label": "Qwen3-8B",
                        "seed": seed,
                        "mechanism_split": "development",
                        "pair_sha256": pair,
                        "donor_offset": offset,
                        "patch_geometry": "endpoint",
                        "state_condition": state,
                        "head_condition": head,
                        "selection_rank_used": False,
                        "targeted_bank_sha256": "bank",
                        "head_plan_file_sha256": "head-plan",
                        "donor_vs_receiver_query_city_log_odds": value,
                    }
                )
    _estimands, _effects, gates = analyze_query_mediation(
        pd.DataFrame(rows),
        phase="discovery",
        geometry="endpoint",
        bootstrap_samples=200,
        random_seed=9,
    )
    assert gates["full_state_mediation_pass"]
    assert gates["count_specific_mediation_pass"]
    assert gates["geometry_pass"]
    assert gates["confirmation_eligible"]


def test_query_mediation_span_eligibility_uses_only_pair_metadata(tmp_path):
    root = tmp_path / "endpoint"
    shards = root / "shards"
    shards.mkdir(parents=True)
    rows = [
        {
            "model_label": "Qwen3-8B",
            "seed": 1234,
            "receiver_occurrence": 2,
            "donor_occurrence": 3,
            "donor_offset": 1,
            "receiver_span_token_count": 3,
            "donor_span_token_count": 5,
            "donor_vs_receiver_query_city_log_odds": 999.0,
        },
        {
            "model_label": "Qwen3-8B",
            "seed": 1235,
            "receiver_occurrence": 4,
            "donor_occurrence": 3,
            "donor_offset": -1,
            "receiver_span_token_count": 8,
            "donor_span_token_count": 8,
            "donor_vs_receiver_query_city_log_odds": -999.0,
        },
    ]
    (shards / "part-000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    result = audit_span_eligibility(root)
    assert result["outcome_fields_accessed"] is False
    assert result["minimum_span_length_distribution"] == {"3": 1, "8": 1}
    assert result["eligibility"]["suffix4"]["eligible_pair_count"] == 1
    assert result["eligibility"]["suffix8"]["eligible_pair_count"] == 1
    assert result["eligibility"]["full_span"]["eligible_pair_count"] == 1


def test_capped_suffix_preserves_short_pairs_without_claiming_exact_width():
    registry = SimpleNamespace(trace_items=((10, 13), (20, 25)))
    receiver, donor, audit = trace_patch_geometry_positions(
        registry,
        receiver_occurrence=1,
        donor_occurrence=2,
        geometry="suffix_cap4",
    )
    assert receiver == (10, 11, 12)
    assert donor == (22, 23, 24)
    assert audit["patch_token_count"] == 3
    assert audit["requested_patch_token_count"] == 4
    assert audit["patch_token_count_capped_by_shorter_span"] is True


def test_capped_query_mediation_reports_exact_width_secondary_stratum():
    rows = []
    values = {
        ("self_patch", "intact"): 0.0,
        ("self_patch", "selected_mask"): 0.0,
        ("self_patch", "layer_matched_random_mask"): 0.0,
        ("full_donor_patch", "intact"): 3.0,
        ("full_donor_patch", "selected_mask"): 0.5,
        ("full_donor_patch", "layer_matched_random_mask"): 2.8,
        ("full_donor_patch_heads_into_self_patch", "selected_restore"): 2.5,
    }
    for seed in range(1234, 1254):
        for offset in (-3, -2, -1, 1, 2, 3):
            pair = f"{seed}:{offset}"
            width = 3 if (seed, offset) == (1234, -3) else 4
            for (state, head), value in values.items():
                rows.append(
                    {
                        "experiment_id": "p0_same_trajectory_query_mediation",
                        "model_label": "Qwen3-8B",
                        "seed": seed,
                        "mechanism_split": "development",
                        "pair_sha256": pair,
                        "donor_offset": offset,
                        "patch_geometry": "suffix_cap4",
                        "patch_token_count": width,
                        "requested_patch_token_count": 4,
                        "patch_token_count_capped_by_shorter_span": width < 4,
                        "state_condition": state,
                        "head_condition": head,
                        "selection_rank_used": False,
                        "targeted_bank_sha256": "bank",
                        "head_plan_file_sha256": "head-plan",
                        "donor_vs_receiver_query_city_log_odds": value,
                    }
                )
    _estimands, _effects, gates = analyze_query_mediation(
        pd.DataFrame(rows),
        phase="discovery",
        geometry="suffix_cap4",
        bootstrap_samples=200,
        random_seed=9,
    )
    audit = gates["capped_width_diagnostics"]
    assert audit["pair_count"] == 120
    assert audit["realized_width_pair_counts"] == {"3": 1, "4": 119}
    assert audit["exact_width_pair_count"] == 119
    assert audit["secondary_only"] is True
    assert len(audit["exact_width_estimands"]) == 5
