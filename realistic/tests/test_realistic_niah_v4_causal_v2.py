from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from realistic_niah_v4.attention import matched_random_heads
from realistic_niah_v4.causal_generation import (
    _causal_v2_completion_cache_key,
    _causal_v2_patch_payload,
    _completion_from_generation_label,
    causal_v2_monotonic_span_source_indices,
    causal_v2_prompt_span_alignment_table,
    run_generation_head_ablation_v2,
    run_generation_residual_patching_v2,
)
from realistic_niah_v4.causal_v2 import (
    CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY,
    CausalV2Design,
    _exact_sign_flip_p,
    confirmation_statistics,
    head_ablation_confirmation_seed_effects,
    head_ablation_confirmation_statistics,
    head_phenotype_scores,
    normalized_transport_metrics,
    rank_head_phenotypes,
    stable_layer_k_conditions,
)
from realistic_niah_v4.prompts import PromptEncoding, TokenSpan
from scripts.run_realistic_niah_v4_causal_v2 import (
    _ablation_confirmation_plan,
    _load_attention_phenotype_source,
    _selected_confirmation_detail,
    _write_json_atomic,
)


def _encoding(
    *,
    stimulus_id: str,
    count: int,
    spans: tuple[tuple[int, int], ...] = ((1, 3), (4, 6)),
    seed: int = 1254,
) -> PromptEncoding:
    slot_spans = tuple(
        TokenSpan(
            slot_index=index,
            start=start,
            end=end,
            active=index <= count,
            kind="needle" if index <= count else "length_matched_control",
            canonical_length=end - start,
            model_token_length=end - start,
        )
        for index, (start, end) in enumerate(spans, start=1)
    )
    return PromptEncoding(
        stimulus_id=stimulus_id,
        design_variant="v4.4",
        seed=seed,
        split="confirmation",
        count=count,
        model_label="toy",
        answer_format="numeric",
        text="toy",
        generation_prompt="toy",
        input_ids=tuple(range(8)),
        attention_mask=(1,) * 8,
        query_position=7,
        slot_spans=slot_spans,
        needle_spans=tuple(span for span in slot_spans if span.active),
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )


def test_causal_v2_design_covers_three_anchors_for_every_k() -> None:
    design = CausalV2Design()
    design.validate()
    assert len(design.canonical_pairs) == 9
    assert len(design.directed_pairs) == 18
    assert design.required_counts == (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)
    for k in (1, 3, 5):
        pairs = design.pairs_for_k(k, directed=False)
        assert len(pairs) == 3
        assert {right - left for left, right in pairs} == {k}
        assert pairs[0][0] == 0
        assert pairs[-1][1] == 10
    with pytest.raises(KeyError, match="No registered causal-v2 pairs for k=2"):
        design.pairs_for_k(2)


def test_causal_v2_json_registry_matches_typed_defaults() -> None:
    configured = CausalV2Design.from_json("configs/realistic_niah_v4_causal_v2.json")
    assert configured == CausalV2Design()
    assert configured.ablation_scope == "answer_query"
    assert configured.ablation_top_ns == tuple(range(1, 33))
    assert configured.prompt_full_span_alignment == CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY
    assert configured.answer_multi_layer_protocol == "cumulative_clamp_L_to_final"


def test_causal_v2_json_writer_normalizes_numpy_integer_metadata(tmp_path) -> None:
    output = tmp_path / "complete.json"
    numpy_integer = pd.Series([0], dtype="int64").unique()[0]

    _write_json_atomic(output, {"counts": [numpy_integer]})

    assert json.loads(output.read_text(encoding="utf-8")) == {"counts": [0]}


def test_normalized_transport_keeps_reversal_overshoot_and_invalid_visible() -> None:
    ideal = normalized_transport_metrics(
        baseline_prediction=5,
        intervened_prediction=10,
        receiver_count=5,
        target_count=10,
    )
    assert ideal["normalized_transport"] == pytest.approx(1.0)
    assert ideal["target_conformity"] == pytest.approx(1.0)
    reverse = normalized_transport_metrics(
        baseline_prediction=5,
        intervened_prediction=4,
        receiver_count=5,
        target_count=10,
    )
    assert reverse["normalized_transport"] == pytest.approx(-0.2)
    assert reverse["target_conformity"] == pytest.approx(-0.2)
    overshoot = normalized_transport_metrics(
        baseline_prediction=5,
        intervened_prediction=11,
        receiver_count=5,
        target_count=10,
    )
    assert overshoot["normalized_transport"] == pytest.approx(1.2)
    assert overshoot["target_conformity"] == pytest.approx(0.8)
    invalid = normalized_transport_metrics(
        baseline_prediction=5,
        intervened_prediction=None,
        receiver_count=5,
        target_count=10,
    )
    assert pd.isna(invalid["normalized_transport"])
    assert invalid["strict_normalized_transport"] == pytest.approx(0.0)
    assert invalid["strict_target_hit"] is False


def test_head_phenotypes_separate_uniform_broad_and_first_locator() -> None:
    rows = []
    for seed in (1234, 1235):
        for head, masses, primary in (
            (0, [0.25, 0.25, 0.25], 0.75),
            (1, [0.70, 0.05, 0.05], 0.30),
        ):
            rows.append(
                {
                    "model_label": "toy",
                    "design_variant": "v4.4",
                    "split": "discovery",
                    "seed": seed,
                    "count": 3,
                    "layer": 1,
                    "head": head,
                    "broad_mass": sum(masses),
                    "broad_coverage": primary / sum(masses),
                    "broad_primary": primary,
                    "needle_span_masses": json.dumps(masses),
                }
            )
    scores = head_phenotype_scores(pd.DataFrame(rows))
    rankings = rank_head_phenotypes(scores, top_n=2)
    assert rankings["broad_aggregation"][0] == (1, 0)
    assert rankings["first_locator"][0] == (1, 1)
    locator = scores[scores["head"] == 1].iloc[0]
    assert locator["first_locator_score"] == pytest.approx(0.65)


def test_attention_source_loads_model_capture_index_without_pooling_summary(
    tmp_path,
) -> None:
    capture = tmp_path / "Qwen3-8B" / "numeric" / "attention" / "capture"
    index_rows = []
    for count in (1, 2):
        stimulus_id = f"V4_4_T10000_N{count}_seed1234"
        relative = f"shards/v4.4/{stimulus_id}.csv.gz"
        shard = capture / relative
        shard.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "stimulus_id": stimulus_id,
                    "design_variant": "v4.4",
                    "model_label": "Qwen3-8B",
                    "seed": 1234,
                    "split": "discovery",
                    "count": count,
                    "layer": 0,
                    "head": 0,
                    "broad_mass": 0.2,
                    "broad_coverage": 1.0,
                    "broad_primary": 0.2,
                    "needle_span_masses": json.dumps([0.2] * count),
                }
            ]
        ).to_csv(shard, index=False, compression="gzip")
        index_rows.append(
            {
                "stimulus_id": stimulus_id,
                "design_variant": "v4.4",
                "model_label": "Qwen3-8B",
                "seed": 1234,
                "split": "discovery",
                "count": count,
                "shard_path": relative,
            }
        )
    capture.mkdir(parents=True, exist_ok=True)
    (capture / "attention_capture_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    detail, manifest = _load_attention_phenotype_source(
        tmp_path,
        model_label="Qwen3-8B",
        expected_seeds=(1234,),
        expected_counts=(1, 2),
    )
    assert len(detail) == 2
    assert manifest["source_kind"] == "attention_capture_index"
    assert manifest["file_count"] == 2
    assert manifest["prompt_count"] == 2
    assert manifest["portable_run_relative_source"] == (
        "Qwen3-8B/numeric/attention/capture/attention_capture_index.jsonl"
    )
    assert len(manifest["source_index_sha256"]) == 64
    assert len(manifest["aggregate_sha256"]) == 64


def test_v2_random_baseline_can_sample_full_layer_population() -> None:
    class Adapter:
        num_heads = {0: 2}

    selected = [(0, 0), (0, 1)]
    with pytest.raises(RuntimeError):
        matched_random_heads(selected, Adapter(), seed=1)
    sampled = matched_random_heads(
        selected,
        Adapter(),
        seed=1,
        exclude_selected=False,
    )
    assert set(sampled) == set(selected)


def test_multi_slot_patch_payload_uses_every_endpoint_or_full_span_token() -> None:
    receiver = _encoding(stimulus_id="receiver", count=0)
    source = _encoding(stimulus_id="source", count=2)
    # Captured rows are query, slot-1 token 1/2, slot-2 token 1/2.
    states = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    slices = {1: (1, 3), 2: (3, 5)}
    positions, endpoint_states, ok, reason = _causal_v2_patch_payload(
        site="toggled_needle_end",
        layer_states=states,
        state_slices=slices,
        receiver=receiver,
        source=source,
        slot_indices=(1, 2),
    )
    assert ok and not reason
    assert positions == (2, 5)
    assert torch.equal(endpoint_states, states[[2, 4]])
    positions, span_states, ok, reason = _causal_v2_patch_payload(
        site="toggled_needle_span",
        layer_states=states,
        state_slices=slices,
        receiver=receiver,
        source=source,
        slot_indices=(1, 2),
    )
    assert ok and not reason
    assert positions == (1, 2, 4, 5)
    assert torch.equal(span_states, states[1:])


def test_monotonic_full_span_map_is_endpoint_preserving_and_identity_when_equal() -> (
    None
):
    assert causal_v2_monotonic_span_source_indices(4, 4) == (0, 1, 2, 3)
    assert causal_v2_monotonic_span_source_indices(3, 4) == (0, 2, 3)
    assert causal_v2_monotonic_span_source_indices(4, 3) == (0, 1, 1, 2)
    with pytest.raises(ValueError, match="must both be positive"):
        causal_v2_monotonic_span_source_indices(0, 3)


def test_full_span_patch_maps_unequal_token_lengths_without_state_averaging() -> None:
    receiver = _encoding(stimulus_id="receiver", count=0)
    source = _encoding(
        stimulus_id="source",
        count=1,
        spans=((1, 4), (4, 6)),
    )
    # Captured rows are query followed by the three complete slot-1 vectors.
    states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    positions, patched_states, ok, reason = _causal_v2_patch_payload(
        site="toggled_needle_span",
        layer_states=states,
        state_slices={1: (1, 4)},
        receiver=receiver,
        source=source,
        slot_indices=(1,),
    )
    assert ok and not reason
    assert positions == (1, 2)
    # Receiver endpoints take donor endpoints; no vector is interpolated.
    assert torch.equal(patched_states, states[[1, 3]])


def test_prompt_full_span_alignment_preflight_reports_each_changed_slot() -> None:
    receiver = _encoding(stimulus_id="receiver", count=0)
    donor = _encoding(stimulus_id="donor", count=2)
    aligned = causal_v2_prompt_span_alignment_table(
        (receiver, donor),
        count_pairs=((0, 2),),
        evaluation_seeds=(1254,),
    )
    assert aligned["slot_index"].tolist() == [1, 2]
    assert aligned["exact_model_token_alignment"].all()
    assert aligned["mapping_supported"].all()
    assert set(aligned["alignment_policy"]) == {CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY}

    mismatched_donor = _encoding(
        stimulus_id="mismatched-donor",
        count=2,
        spans=((1, 4), (4, 6)),
    )
    mismatched = causal_v2_prompt_span_alignment_table(
        (receiver, mismatched_donor),
        count_pairs=((0, 2),),
        evaluation_seeds=(1254,),
    )
    assert mismatched["exact_model_token_alignment"].tolist() == [False, True]
    assert mismatched["mapping_supported"].tolist() == [True, True]
    assert json.loads(mismatched.loc[0, "source_index_map"]) == [0, 2]
    assert mismatched.loc[0, "source_tokens_dropped"] == 1


def test_baseline_completion_payload_is_recoverable_for_identity_reuse() -> None:
    recovered = _completion_from_generation_label(
        {
            "completion_text": "10",
            "completion_text_raw": "10<eos>",
            "generated_token_ids": "[123, 456]",
        }
    )
    assert recovered == {
        "completion_text": "10",
        "completion_text_raw": "10<eos>",
        "generated_token_ids": [123, 456],
        "generation_truncated": False,
    }


def test_answer_same_count_cache_ignores_nominal_k_but_donor_transport_does_not() -> (
    None
):
    receiver = _encoding(stimulus_id="receiver", count=5)
    same_count_source = _encoding(stimulus_id="other-seed-N5", count=5)
    shared = {
        "condition": "same_count_seed",
        "receiver": receiver,
        "state_source": same_count_source,
        "site": "answer_query",
        "protocol": "single_layer",
        "start_layer": 7,
    }
    assert _causal_v2_completion_cache_key(
        **shared, slot_indices=(6,)
    ) == _causal_v2_completion_cache_key(**shared, slot_indices=(6, 7, 8, 9, 10))
    donor_shared = {**shared, "condition": "donor_transport"}
    assert _causal_v2_completion_cache_key(
        **donor_shared, slot_indices=(6,)
    ) != _causal_v2_completion_cache_key(**donor_shared, slot_indices=(6, 7, 8, 9, 10))


def test_answer_patching_reuses_identity_and_equivalent_same_count_generations(
    monkeypatch,
) -> None:
    import realistic_niah_v4.causal_generation as causal_generation

    integration_spans = ((1, 2), (2, 3), (3, 4))
    encodings = tuple(
        _encoding(
            stimulus_id=f"seed{seed}-N{count}",
            count=count,
            seed=seed,
            spans=integration_spans,
        )
        for seed in (1254, 1255)
        for count in (0, 1, 3)
    )

    def label(encoding: PromptEncoding) -> dict[str, object]:
        return {
            "stimulus_id": encoding.stimulus_id,
            "model_label": "toy",
            "design_variant": "v4.4",
            "seed": encoding.seed,
            "gold_count": encoding.count,
            "outcome_group": "correct",
            "is_correct": True,
            "format_valid": True,
            "parsed_count": encoding.count,
            "count_error": 0,
            "completion_text": str(encoding.count),
            "completion_text_raw": str(encoding.count),
            "generated_token_ids": json.dumps([encoding.count]),
            "generation_truncated": False,
        }

    labels = {encoding.stimulus_id: label(encoding) for encoding in encodings}
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_capture(_model, _adapter, encoding, positions, *, layers=None):
        selected = tuple(range(2)) if layers is None else tuple(layers)
        return torch.zeros(1), {
            layer: torch.zeros((len(tuple(positions)), 3), dtype=torch.float32)
            for layer in selected
        }

    def fake_generate(_model, _tokenizer, _adapter, encoding, interventions, **_kwargs):
        calls.append((encoding.stimulus_id, tuple(sorted(interventions))))
        return _completion_from_generation_label(labels[encoding.stimulus_id])

    monkeypatch.setattr(causal_generation, "capture_post_block_states", fake_capture)
    monkeypatch.setattr(
        causal_generation, "generate_with_residual_interventions", fake_generate
    )

    class Adapter:
        num_layers = 2

    detail = run_generation_residual_patching_v2(
        None,
        None,
        Adapter(),
        encodings,
        baseline_labels=labels,
        count_pairs=((0, 1), (0, 3)),
        start_layers=(0, 1),
        sites=("answer_query",),
        protocols=("single_layer",),
        control_conditions=("donor_transport", "self_patch", "same_count_seed"),
        evaluation_seeds=(1254,),
        shared_completion_cache={},
    )
    assert len(detail) == 12
    # Four donor transports plus two unique same-count interventions. The two
    # self rows per pair reuse their registered receiver baseline.
    assert len(calls) == 6
    modes = detail.groupby("condition")["generation_reuse_mode"].value_counts()
    assert modes.loc[("self_patch", "baseline_identity_reuse")] == 4
    assert modes.loc[("same_count_seed", "fresh_intervention")] == 2
    assert modes.loc[("same_count_seed", "equivalent_intervention_cache")] == 2


def _screen_rows() -> pd.DataFrame:
    rows = []
    anchors = ((0, 3), (3, 6), (7, 10))
    for layer, treatment_effect in ((3, 0.5), (4, 0.05)):
        for seed in range(1254, 1259):
            for low, high in anchors:
                for receiver, donor in ((low, high), (high, low)):
                    for condition, effect in (
                        ("donor_transport", treatment_effect),
                        ("self_patch", 0.0),
                    ):
                        rows.append(
                            {
                                "model_label": "toy",
                                "site": "answer_query",
                                "patch_protocol": "single_layer",
                                "start_layer": layer,
                                "k": 3,
                                "seed": seed,
                                "receiver_count": receiver,
                                "donor_count": donor,
                                "target_direction": (
                                    "increase" if donor > receiver else "decrease"
                                ),
                                "condition": condition,
                                "strict_normalized_transport": effect,
                                "patched_format_valid": True,
                                "transport_numeric_valid": True,
                            }
                        )
    return pd.DataFrame(rows)


def test_stability_selection_locks_only_replicated_layer_k_conditions() -> None:
    scores, manifest = stable_layer_k_conditions(
        _screen_rows(), family="answer_patching"
    )
    selected = scores[scores["stable"]]
    assert selected["start_layer"].tolist() == [3]
    assert manifest["selected_condition_count"] == 1
    assert manifest["held_out_confirmation_seeds"] == list(range(1259, 1264))
    json.dumps(manifest)


def test_confirmation_statistics_uses_seed_cluster_and_matched_control() -> None:
    result = confirmation_statistics(
        _screen_rows().query("start_layer == 3"),
        family="answer_patching",
        bootstrap_repetitions=200,
    )
    assert len(result) == 1
    assert result.iloc[0]["mean_control_adjusted_transport"] == pytest.approx(0.5)
    assert result.iloc[0]["positive_seed_fraction"] == pytest.approx(1.0)
    assert _exact_sign_flip_p(pd.Series([1.0] * 5).to_numpy()) == pytest.approx(2 / 32)


def test_ablation_confirmation_uses_independent_model_specific_top_n() -> None:
    selection = Path(
        "configs/realistic_niah_v4_causal_v2_ablation_confirmation_selection.json"
    ).resolve()
    _payload, qwen = _ablation_confirmation_plan(
        selection, model_label="Qwen3-8B", repo_root=Path(".").resolve()
    )
    _payload, gemma = _ablation_confirmation_plan(
        selection, model_label="Gemma4-E4B", repo_root=Path(".").resolve()
    )
    assert qwen["top_n"] == 8
    assert gemma["top_n"] == 6
    assert qwen["seeds"] == gemma["seeds"] == tuple(range(1264, 1274))
    assert not set(qwen["seeds"]).intersection(range(1254, 1259))


def test_confirmation_ablation_accepts_one_frozen_bank() -> None:
    class Adapter:
        num_heads = {0: 4}

    with pytest.raises(ValueError, match="No causal-v2 head-ablation rows"):
        run_generation_head_ablation_v2(
            None,
            None,
            Adapter(),
            (),
            baseline_labels={},
            rankings={"broad_aggregation": [(0, 0), (0, 1)]},
            top_ns=(2,),
            random_replicates=3,
            require_full_sweep=False,
        )
    with pytest.raises(ValueError, match="discovery ablation requires"):
        run_generation_head_ablation_v2(
            None,
            None,
            Adapter(),
            (),
            baseline_labels={},
            rankings={"broad_aggregation": [(0, 0), (0, 1)]},
            top_ns=tuple(range(1, 33)),
            require_full_sweep=True,
        )


def test_head_ablation_confirmation_statistics_cluster_by_new_seed() -> None:
    rows = []
    for seed in range(1264, 1274):
        for condition, replicate, accuracy, absolute_error in (
            ("ranked", -1, -1.0, 2.0),
            ("layer_matched_random", 0, 0.0, 0.0),
            ("layer_matched_random", 1, 0.0, 0.0),
            ("layer_matched_random", 2, 0.0, 0.0),
        ):
            rows.append(
                {
                    "model_label": "toy",
                    "stimulus_id": f"seed{seed}-N7",
                    "seed": seed,
                    "gold_count": 7,
                    "head_bank": "broad_aggregation",
                    "top_n": 8,
                    "condition": condition,
                    "random_replicate": replicate,
                    "accuracy_delta": accuracy,
                    "absolute_error_delta": absolute_error,
                    "prediction_changed": float(condition == "ranked"),
                    "patched_format_valid": True,
                }
            )
    seed_effects = head_ablation_confirmation_seed_effects(pd.DataFrame(rows))
    assert len(seed_effects) == 10
    assert set(seed_effects["seed"]) == set(range(1264, 1274))
    statistics = head_ablation_confirmation_statistics(
        seed_effects, bootstrap_repetitions=200
    ).set_index("metric")
    assert statistics.loc[
        "absolute_error_delta", "ranked_minus_random_mean"
    ] == pytest.approx(2.0)
    assert statistics.loc[
        "accuracy_delta", "ranked_minus_random_mean"
    ] == pytest.approx(-1.0)
    assert statistics.loc[
        "absolute_error_delta", "exact_sign_flip_p"
    ] == pytest.approx(2 / 1024)
    assert statistics.loc[
        "absolute_error_delta", "harmful_seed_fraction"
    ] == pytest.approx(1.0)


def test_stability_refuses_skipped_full_span_rows() -> None:
    frame = _screen_rows().query("start_layer == 3").copy()
    frame["status"] = "ok"
    frame["skip_reason"] = ""
    frame.loc[frame.index[0], "status"] = "skipped"
    frame.loc[frame.index[0], "skip_reason"] = "slot_1_model_token_length_mismatch"
    with pytest.raises(ValueError, match="skipped interventions"):
        stable_layer_k_conditions(frame, family="answer_patching")


def test_selected_confirmation_excludes_failed_screen_conditions() -> None:
    screen = _screen_rows()
    confirmation = screen.query("start_layer == 3").copy()
    confirmation["seed"] = confirmation["seed"].astype(int) + 5
    payload = {
        "family": "answer_patching",
        "selection_split": "screen",
        "selected": [
            {
                "model_label": "toy",
                "site": "answer_query",
                "patch_protocol": "single_layer",
                "start_layer": 3,
                "k": 3,
            }
        ],
        "selected_condition_count": 1,
    }
    combined = _selected_confirmation_detail(
        screen,
        confirmation,
        family="answer_patching",
        selection_payload=payload,
        model_label="toy",
        design=CausalV2Design(),
    )
    assert set(combined["start_layer"].astype(int)) == {3}
    assert set(combined["seed"].astype(int)) == set(range(1254, 1264))
    result = confirmation_statistics(
        combined,
        family="answer_patching",
        bootstrap_repetitions=200,
    )
    assert set(result["evidence_scope"]) == {
        "held_out_only",
        "screen_plus_held_out",
    }
    held_out = result[result["evidence_scope"].eq("held_out_only")].iloc[0]
    combined_row = result[result["evidence_scope"].eq("screen_plus_held_out")].iloc[0]
    assert held_out["seeds"] == 5
    assert held_out["screen_seeds"] == 0
    assert held_out["held_out_confirmation_seeds"] == 5
    assert bool(held_out["is_primary_confirmation"])
    assert combined_row["seeds"] == 10
    assert combined_row["screen_seeds"] == 5
    assert combined_row["held_out_confirmation_seeds"] == 5
    assert not bool(combined_row["is_primary_confirmation"])
