from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from realistic_niah_v4.prompts import PromptEncoding, TokenSpan
from realistic_niah_v4_4_5.followup_edges import (
    derive_factorial_encoding,
    distance_bin,
    natural_edge_delta,
    repeated_anchor_candidates,
    select_attention_mass_control,
    select_control_feasible_candidates,
    select_deterministic_random_control,
)
from scripts.run_realistic_niah_v4_4_5_induction_circuit import (
    identity_repeated_pairs,
)
from scripts.analyze_realistic_niah_v4_4_5_induction_circuit import (
    audit_structural_and_relation_rows,
)


def encoding_fixture() -> PromptEncoding:
    spans = (
        TokenSpan(0, 20, 24, True, "needle", 4, 4),
        TokenSpan(1, 50, 54, True, "needle", 4, 4),
    )
    ids = list(range(80))
    ids[20:24] = [7, 100, 9, 10]
    ids[50:54] = [7, 101, 9, 11]
    return PromptEncoding(
        stimulus_id="fixture",
        design_variant="v4.4",
        seed=1,
        split="test",
        count=2,
        model_label="fixture",
        answer_format="numeric",
        text="fixture",
        generation_prompt="fixture",
        input_ids=tuple(ids),
        attention_mask=tuple([1] * len(ids)),
        query_position=79,
        slot_spans=spans,
        needle_spans=spans,
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )


def test_repeated_anchor_registration_requires_unique_in_span_successor():
    candidates = repeated_anchor_candidates(encoding_fixture())
    assert candidates[7] == ((20, 21), (50, 51))
    assert candidates[9] == ((22, 23), (52, 53))
    assert 10 not in candidates


def test_single_needle_is_explicit_no_previous_match_control():
    base = encoding_fixture()
    single = replace(
        base,
        count=1,
        slot_spans=(base.slot_spans[0],),
        needle_spans=(base.needle_spans[0],),
    )
    assert repeated_anchor_candidates(single) == {}
    assert identity_repeated_pairs(single, 7) == ()


def test_structural_no_match_and_relation_rows_have_exact_coverage():
    rows = []
    for count in (1, 2):
        for arm in ("natural", "candidate_edge_block", "mass_distance_control"):
            structural = count == 1
            rows.append(
                {
                    "seed": 1254,
                    "gold_count": count,
                    "arm": arm,
                    "structural_no_previous_match": structural,
                    "registered_edges": 0 if structural else 1,
                    "reachable_edges": 0 if structural else 1,
                    "intervention_sites": (
                        0 if structural or arm == "natural" else 1
                    ),
                    "expected_count": float(count),
                    "strict_absolute_error": 0.0,
                    "retrieval_bank_broad_score_mean": 0.5,
                    "correct_count_margin": 1.0,
                }
            )
    audit_structural_and_relation_rows(
        rows,
        structural_counts={1},
        primary_relation_counts={2},
    )
    rows[1]["intervention_sites"] = 1
    with pytest.raises(RuntimeError, match="structural control"):
        audit_structural_and_relation_rows(
            rows,
            structural_counts={1},
            primary_relation_counts={2},
        )


def test_structural_no_match_keeps_strict_numeric_equality():
    rows = []
    for arm in ("natural", "candidate_edge_block", "mass_distance_control"):
        rows.append(
            {
                "seed": 1254,
                "gold_count": 1,
                "arm": arm,
                "structural_no_previous_match": True,
                "registered_edges": 0,
                "reachable_edges": 0,
                "intervention_sites": 0,
                "expected_count": 1.0,
                "strict_absolute_error": 0.0,
                "retrieval_bank_broad_score_mean": 0.5,
                "correct_count_margin": 1.0,
            }
        )
    rows[1]["expected_count"] += 1e-6
    with pytest.raises(RuntimeError, match="not numerically identical"):
        audit_structural_and_relation_rows(
            rows,
            structural_counts={1},
            primary_relation_counts=set(),
        )


def test_mass_and_random_controls_stay_in_distance_bin():
    row = torch.linspace(0.0, 1.0, 80)
    mass_key, audit = select_attention_mass_control(
        row,
        query=79,
        target_key=50,
        allowed=range(0, 79),
        excluded={20, 21, 22, 23, 50, 51, 52, 53},
        bin_width=16,
    )
    random_key, random_audit = select_deterministic_random_control(
        query=79,
        target_key=50,
        allowed=range(0, 79),
        excluded={20, 21, 22, 23, 50, 51, 52, 53},
        bin_width=16,
        label="fixture",
    )
    assert audit["exact_distance_bin"]
    assert random_audit["exact_distance_bin"]
    assert distance_bin(79, mass_key, width=16) == distance_bin(79, 50, width=16)
    assert distance_bin(79, random_key, width=16) == distance_bin(79, 50, width=16)


def test_control_feasible_candidates_respect_per_bin_capacity_and_rank():
    selected, audit = select_control_feasible_candidates(
        query=128,
        ranked_candidates=[95, 94, 93, 60, 59, 58],
        allowed_controls=[90, 61, 57, 56],
        bin_width=32,
        max_candidates=5,
    )
    # Keys 95/94/93 share a bin with only one control (90), so rank keeps 95.
    # The next bin has three controls and therefore retains 60/59/58.
    assert selected == (95, 60, 59, 58)
    assert audit["selected_candidate_count"] == 4
    assert audit["omitted_exhausted_bin_capacity"] == 2
    assert audit["all_selected_have_unique_exact_bin_capacity"] is True


def test_control_feasible_candidates_skip_bins_without_controls():
    selected, audit = select_control_feasible_candidates(
        query=128,
        ranked_candidates=[127, 95, 60],
        allowed_controls=[90, 61],
        bin_width=32,
        max_candidates=2,
    )
    assert selected == (95, 60)
    assert audit["omitted_no_exact_bin_capacity"] == 1


def test_control_feasible_candidates_regress_qwen_first_unit_capacity_shape():
    # The failed formal unit had 8 halo candidates in each of two bins, but
    # only 48 and 2 distinct non-halo controls in those bins.  Exact matching
    # therefore supports ten, not sixteen, candidate edges.
    selected, audit = select_control_feasible_candidates(
        query=10106,
        ranked_candidates=list(range(1090, 1098)) + list(range(1051, 1059)),
        allowed_controls=list(range(1098, 1146)) + [1018, 1019],
        bin_width=64,
        max_candidates=16,
    )
    assert len(selected) == 10
    assert audit["selected_by_distance_bin"] == {"140": 8, "141": 2}
    assert audit["omitted_exhausted_bin_capacity"] == 6


def test_natural_edge_delta_obeys_gqa_mapping():
    # Two KV heads, width two; query heads 0/1 map KV0 and 2/3 map KV1.
    values = torch.tensor(
        [[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]
    )
    row = torch.tensor([0.25, 0.75])
    delta, audit = natural_edge_delta(
        row,
        values,
        keys=[1],
        key_start=0,
        query_head=3,
        query_heads=4,
        head_dim=2,
    )
    assert torch.allclose(delta, torch.tensor([-22.5, -30.0]))
    assert audit["edge_attention_mass"] == pytest.approx(0.75)


def test_factorial_derivation_preserves_length_query_and_moves_spans():
    base = encoding_fixture()
    replacements = ([7, 200, 9, 12], [7, 201, 9, 13])
    derived, audit = derive_factorial_encoding(
        base,
        identity_replacements=replacements,
        identity=True,
        context=True,
        position=True,
        context_width=2,
    )
    assert derived.sequence_length == base.sequence_length
    assert derived.query_position == base.query_position
    assert derived.input_ids[base.query_position] == base.input_ids[base.query_position]
    assert [span.model_token_length for span in derived.needle_spans] == [4, 4]
    assert [span.start for span in derived.needle_spans] != [20, 50]
    assert audit["identity"] and audit["context"] and audit["position"]
