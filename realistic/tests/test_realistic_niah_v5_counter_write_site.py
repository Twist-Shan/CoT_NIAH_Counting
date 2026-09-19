from __future__ import annotations

from realistic_niah_v5.counter_write_site import (
    build_site_early_stop_encoding,
    directional_count_metrics,
    target_count_restoration_metrics,
)
from realistic_niah_v5.encoding import NativeTraceEncoding


def _encoding() -> NativeTraceEncoding:
    return NativeTraceEncoding(
        stimulus_id="stim",
        request_id="req",
        design_variant="v5",
        seed=1,
        split="discovery",
        count=6,
        model_label="test",
        model_family="qwen3",
        answer_format="numeric",
        text="",
        generation_prompt="",
        input_ids=tuple(range(20)),
        attention_mask=(1,) * 20,
        query_position=18,
        prompt_token_count=5,
        raw_prefix_text="",
        selected_site={},
        prompt_record_spans=(),
        trace_item_spans=(),
        slot_spans=(),
        needle_spans=(),
        hard_negative_spans=(),
        count_candidate_texts=tuple((value, str(value)) for value in range(1, 11)),
        count_candidate_answer_token_ids=tuple(
            (value, (value,)) for value in range(1, 11)
        ),
        count_candidate_token_ids=tuple(
            (value, (value, 99)) for value in range(1, 11)
        ),
    )


def test_site_early_stop_preserves_prefix_and_uses_only_terminal_bridge() -> None:
    encoding, audit = build_site_early_stop_encoding(
        _encoding(), cut_position=9, terminal_suffix_start=16, site_id="pre_marker:4"
    )
    assert encoding.input_ids == tuple(range(10)) + (16, 17, 18)
    assert encoding.query_position == 12
    assert encoding.input_ids[:5] == tuple(range(5))
    assert audit["future_original_tokens_present"] is False
    assert audit["prompt_tokens_changed"] is False


def test_directional_metrics_are_positive_for_earlier_and_later_donors() -> None:
    counts = ",".join(str(value) for value in range(1, 11))
    baseline = {
        "candidate_counts": counts,
        "candidate_log_scores": ",".join("0" for _ in range(10)),
        "expected_count": 5.0,
    }
    later_scores = [0.0] * 10
    later_scores[5] = 2.0
    later = {
        "candidate_counts": counts,
        "candidate_log_scores": ",".join(str(value) for value in later_scores),
        "expected_count": 5.4,
    }
    earlier_scores = [0.0] * 10
    earlier_scores[3] = 2.0
    earlier = {
        "candidate_counts": counts,
        "candidate_log_scores": ",".join(str(value) for value in earlier_scores),
        "expected_count": 4.6,
    }
    later_metrics = directional_count_metrics(
        baseline, later, receiver_progress=5, donor_progress=6
    )
    earlier_metrics = directional_count_metrics(
        baseline, earlier, receiver_progress=5, donor_progress=4
    )
    assert later_metrics["donor_vs_receiver_log_odds_effect"] == 2.0
    assert earlier_metrics["donor_vs_receiver_log_odds_effect"] == 2.0
    assert later_metrics["donor_aligned_expected_count_shift"] > 0
    assert earlier_metrics["donor_aligned_expected_count_shift"] > 0


def test_target_restoration_metrics_track_probability_margin_and_distance() -> None:
    counts = ",".join(str(value) for value in range(1, 11))
    baseline_scores = [0.0] * 10
    baseline_scores[1] = 1.0
    patched_scores = [0.0] * 10
    patched_scores[4] = 3.0
    baseline = {
        "candidate_counts": counts,
        "candidate_log_scores": ",".join(str(value) for value in baseline_scores),
        "expected_count": 3.0,
    }
    patched = {
        "candidate_counts": counts,
        "candidate_log_scores": ",".join(str(value) for value in patched_scores),
        "expected_count": 4.5,
    }
    metrics = target_count_restoration_metrics(
        baseline, patched, target_progress=5
    )
    assert metrics["target_probability_effect"] > 0
    assert metrics["target_margin_effect"] > 0
    assert metrics["expected_distance_improvement"] > 0
