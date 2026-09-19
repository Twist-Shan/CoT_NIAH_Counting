from __future__ import annotations

import importlib.util
from pathlib import Path

from realistic_niah_v5.count_stream import AnswerSourceRegistry
from realistic_niah_v5.encoding import NativeTraceEncoding
from realistic_niah_v5.single_seed_walkthrough import (
    _target_count_metrics,
    build_uninformative_prompt_trace_encoding,
    matched_ordinary_positions,
    occurrence_counter_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts" / "analyze_realistic_niah_v5_single_seed_walkthrough.py"
SPEC = importlib.util.spec_from_file_location("single_seed_walkthrough_analysis", ANALYZER)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class _Tokenizer:
    all_special_ids = [100]


def _encoding() -> NativeTraceEncoding:
    return NativeTraceEncoding(
        stimulus_id="fixture",
        request_id="Qwen3-8B/native_thinking/v5/fixture",
        design_variant="v5",
        seed=1254,
        split="confirmation",
        count=2,
        model_label="Qwen3-8B",
        model_family="qwen3",
        answer_format="number",
        text="fixture",
        generation_prompt="fixture",
        input_ids=tuple(range(100, 145)),
        attention_mask=tuple([1] * 45),
        query_position=44,
        prompt_token_count=35,
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


def _registry() -> AnswerSourceRegistry:
    value = AnswerSourceRegistry(
        request_id="Qwen3-8B/native_thinking/v5/fixture",
        answer_site_id="answer_query_v3",
        sequence_length=45,
        prompt_token_count=35,
        query_position=44,
        prompt_records=((5, 7), (10, 12)),
        trace_context=((35, 44),),
        trace_items=((35, 38), (39, 42)),
        trace_other=((38, 39), (42, 44)),
        trace_markers=((36, 37), (40, 41)),
        trace_nonmarkers=((35, 36), (37, 38), (39, 40), (41, 42)),
        earlier_trace_items=((35, 38),),
        terminal_trace_item=((39, 42),),
    )
    value.validate()
    return value


def _site(start: int, end: int) -> dict[str, object]:
    return {
        "status": "ok",
        "full_sequence_token_start": start,
        "full_sequence_token_end": end,
    }


def test_uninformative_control_replaces_prompt_and_trace_without_moving_query() -> None:
    clean = _encoding()
    registry = _registry()
    control, audit = build_uninformative_prompt_trace_encoding(
        clean, registry, _Tokenizer(), random_seed=20260822
    )
    sources = set(registry.positions("prompt_records")) | set(
        registry.positions("trace_context")
    )
    assert control.sequence_length == clean.sequence_length
    assert control.input_ids[registry.query_position] == clean.input_ids[registry.query_position]
    assert any(control.input_ids[position] != clean.input_ids[position] for position in sources)
    assert all(
        control.input_ids[position] == clean.input_ids[position]
        for position in range(clean.sequence_length)
        if position not in sources
    )
    assert audit["source_span_count"] == 3
    assert audit["trace_context_span_count"] == 1
    assert audit["control_sequence_length_equal"]
    assert audit["answer_query_token_preserved"]


def test_occurrence_geometry_uses_marker_after_city_and_tail_before_city() -> None:
    registry = _registry()
    after = {
        "grammar_class": "adjacent_rank_after_city",
        "sites": {
            "rank_evidence_core_span": _site(36, 37),
            "city_target_span": _site(35, 36),
            "post_update_commit_state": _site(37, 38),
        },
    }
    before = {
        "grammar_class": "same_unit_rank_before_city",
        "sites": {
            "rank_evidence_core_span": _site(39, 40),
            "city_target_span": _site(40, 41),
            "post_update_commit_state": _site(41, 42),
        },
    }
    first, first_audit = occurrence_counter_geometry(registry, after, 1)
    second, second_audit = occurrence_counter_geometry(registry, before, 2)
    assert first["counter_carrier"] == (36,)
    assert first_audit["counter_carrier_component"] == "marker_core"
    assert second["counter_carrier"] == (40, 41)
    assert second_audit["counter_carrier_component"] == "city_to_commit_tail"
    donors = matched_ordinary_positions(registry, second["counter_carrier"])
    assert len(donors) == 2
    assert not (set(donors) & set(registry.positions("prompt_records")))


def test_target_count_metrics_reinterprets_the_vector_for_occurrence_k() -> None:
    scores = [float(-abs(value - 4)) for value in range(1, 11)]
    probabilities = [0.01] * 10
    probabilities[3] = 0.91
    outcomes = {
        "candidate_counts": ",".join(str(value) for value in range(1, 11)),
        "candidate_log_scores": ",".join(str(value) for value in scores),
        "candidate_probabilities": ",".join(str(value) for value in probabilities),
        "predicted_count_among_candidates": 4,
        "prediction": 4,
    }
    metrics = _target_count_metrics(outcomes, 4)
    assert metrics["candidate_prediction_matches_restored_target"]
    assert metrics["greedy_prediction_matches_restored_target"]
    assert metrics["restored_target_count_probability"] == 0.91
    assert metrics["restored_target_count_margin"] == 1.0


def test_single_seed_analyzer_reports_paths_without_inference() -> None:
    rows = []
    common = {
        "model_label": "Qwen3-8B",
        "seed": 1254,
        "gold_count": 2,
        "request_id": "fixture",
        "source_layer": 19,
        "answer_query_patched": False,
        "case_selected_by_outcome": False,
        "control_sequence_length_equal": True,
        "correct_count_probability": 0.8,
        "correct_count_margin": 1.0,
    }
    for condition, prediction in (("clean", 2), ("uninformative", 7)):
        rows.append(
            {
                **common,
                "condition": condition,
                "restored_occurrence": 0,
                "predicted_count_among_candidates": prediction,
                "expected_count": float(prediction),
                "prediction": prediction,
            }
        )
    for condition in analysis.RESTORE_CONDITIONS:
        for occurrence in (1, 2):
            prediction = occurrence if condition != "counter_carrier_matched_control" else 7
            rows.append(
                {
                    **common,
                    "condition": condition,
                    "restored_occurrence": occurrence,
                    "predicted_count_among_candidates": prediction,
                    "expected_count": float(prediction),
                    "prediction": prediction,
                    "restored_target_count_probability": 0.7,
                    "restored_target_count_margin": 0.5,
                    "counter_carrier_component": "marker_core",
                    "patch_token_count": 1,
                }
            )
    table, summary = analysis.summarize(rows)
    assert len(table) == 6
    assert summary["status"] == "PASS"
    assert summary["case_study_not_inferential"]
    assert summary["conditions"]["counter_carrier_restore"][
        "candidate_exact_path_fraction"
    ] == 1.0
    assert summary["conditions"]["counter_carrier_matched_control"][
        "candidate_exact_path_fraction"
    ] == 0.0
