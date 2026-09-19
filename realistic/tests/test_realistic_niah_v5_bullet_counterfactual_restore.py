from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from realistic_niah_v4.prompts import TokenSpan
from realistic_niah_v5.bullet_counterfactual_restore import (
    audit_complete_marker_scrubbable_list,
    build_scrubbed_source_and_blank,
    running_target_metrics,
)
from realistic_niah_v5.encoding import NativeTraceEncoding
from scripts.analyze_realistic_niah_v5_bullet_counterfactual_restore import (
    _confirmation_statistics,
    _freeze_top_three,
)


def _bullet_row(*, nonstructural_bridge: bool = False, numbered: bool = False) -> dict:
    cities = [f"City{chr(65 + index)}" for index in range(10)]
    scores = list(range(70, 80))
    preamble = "<think>First I enumerate all records.\n"
    bullets = []
    starts = []
    ends = []
    cursor = len(preamble)
    for index, (city, score) in enumerate(zip(cities, scores), start=1):
        prefix = f"- {index}. " if numbered and index == 3 else "- "
        text = f"{prefix}{city}: {score}\n"
        starts.append(cursor)
        cursor += len(text)
        ends.append(cursor)
        bullets.append(text)
        if nonstructural_bridge and index == 6:
            bridge = "This prose exits the list episode.\n"
            cursor += len(bridge)
            bullets.append(bridge)
    close = "</think>\nTotal: 10"
    raw = preamble + "".join(bullets) + close
    lines = list(range(2, 12))
    return {
        "gold_count": 10,
        "gold_records": [
            {"city": city, "score": score, "slot_index": index}
            for index, (city, score) in enumerate(zip(cities, scores), start=1)
        ],
        "generation_truncated": False,
        "raw_output_text": raw,
        "trace_parse": {
            "parser": {
                "reasoning_start_char": len("<think>"),
                "reasoning_end_char": raw.index("</think>"),
                "marker_kind": "bullet",
                "trace_one_to_one": True,
                "item_count": 10,
                "item_start_chars": starts,
                "item_end_chars": ends,
                "item_line_numbers": lines,
                "item_markers": ["-"] * 10,
                "item_gold_cities": cities,
            }
        },
    }


def test_list_audit_is_outcome_blind_and_allows_prelist_enumeration() -> None:
    row = _bullet_row()
    row["raw_output_text"] = row["raw_output_text"].replace("Total: 10", "Total: 3")
    audit = audit_complete_marker_scrubbable_list(row)
    assert audit["eligible"] is True
    assert audit["prebullet_contains_explicit_count_or_record_language"] is True
    assert audit["eligibility_uses_final_answer"] is False
    assert audit["final_answer_correctness_accessed"] is False


def test_list_audit_registers_number_marker_and_rejects_prose_bridge() -> None:
    numbered = audit_complete_marker_scrubbable_list(_bullet_row(numbered=True))
    noncontiguous = audit_complete_marker_scrubbable_list(
        _bullet_row(nonstructural_bridge=True)
    )
    assert numbered["eligible"] is True
    assert numbered["item_marker_char_spans"][2]
    assert noncontiguous["eligible"] is False
    assert any(
        value.startswith("nonstructural_interitem_bridge:")
        for value in noncontiguous["reasons"]
    )


class _TokenTokenizer:
    all_special_ids = (999,)

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(97 + (int(value) % 26)) for value in token_ids)


def _encoding() -> NativeTraceEncoding:
    item_pairs = tuple((160 + 4 * index, 164 + 4 * index) for index in range(10))
    spans = tuple(
        TokenSpan(
            slot_index=index,
            start=start,
            end=end,
            active=True,
            kind="native_trace_item",
            canonical_length=end - start,
            model_token_length=end - start,
        )
        for index, (start, end) in enumerate(item_pairs, start=1)
    )
    ids = list(range(1, 205))
    ids[151] = 999
    return NativeTraceEncoding(
        stimulus_id="s",
        request_id="r",
        design_variant="native_thinking",
        seed=1234,
        split="discovery",
        count=10,
        model_label="Qwen3-8B",
        model_family="qwen3",
        answer_format="Total: <integer>",
        text="",
        generation_prompt="",
        input_ids=tuple(ids),
        attention_mask=(1,) * len(ids),
        query_position=len(ids) - 1,
        prompt_token_count=150,
        raw_prefix_text="",
        selected_site={},
        prompt_record_spans=(),
        trace_item_spans=spans,
        slot_spans=spans,
        needle_spans=spans,
        hard_negative_spans=(),
        count_candidate_texts=tuple((value, str(value)) for value in range(1, 11)),
        count_candidate_answer_token_ids=tuple(
            (value, (value,)) for value in range(1, 11)
        ),
        count_candidate_token_ids=tuple(
            (value, (value, 205)) for value in range(1, 11)
        ),
    )


def test_source_blank_scrub_preserves_geometry_and_source_bullets() -> None:
    encoding = _encoding()
    registry = SimpleNamespace(
        prompt_records=((10, 15), (80, 84)),
        trace_items=tuple((160 + 4 * index, 164 + 4 * index) for index in range(10)),
        trace_other=((150, 160), (200, 204)),
        trace_markers=tuple((160 + 4 * index, 161 + 4 * index) for index in range(10)),
        prompt_token_count=150,
    )
    source, blank, audit = build_scrubbed_source_and_blank(
        encoding,
        registry,
        _TokenTokenizer(),
        random_seed=7,
    )
    bullet_positions = range(160, 200)
    assert len(source.input_ids) == len(blank.input_ids) == len(encoding.input_ids)
    assert source.input_ids[151] == blank.input_ids[151] == 999
    marker_positions = {160 + 4 * index for index in range(10)}
    assert all(source.input_ids[position] != encoding.input_ids[position] for position in marker_positions)
    assert all(
        source.input_ids[position] == encoding.input_ids[position]
        for position in set(bullet_positions) - marker_positions
    )
    assert all(
        blank.input_ids[position] != source.input_ids[position]
        for position in bullet_positions
    )
    assert all(
        blank.input_ids[position] == source.input_ids[position]
        for position in list(range(10, 15)) + list(range(80, 84))
    )
    assert audit["source_item_nonmarkers_preserved"] is True
    assert audit["source_explicit_item_markers_scrubbed"] is True
    assert audit["source_blank_base_scrub_identical"] is True
    assert audit["retokenization_used"] is False


def test_running_target_metrics_uses_k_not_source_gold_10() -> None:
    scores = [0.0] * 10
    scores[2] = 4.0
    probabilities = [0.01] * 10
    probabilities[2] = 0.91
    outcomes = {
        "candidate_counts": ",".join(str(value) for value in range(1, 11)),
        "candidate_log_scores": ",".join(str(value) for value in scores),
        "candidate_probabilities": ",".join(str(value) for value in probabilities),
    }
    metrics = running_target_metrics(outcomes, target_k=3)
    assert metrics["running_target_k"] == 3
    assert metrics["predicted_running_count"] == 3
    assert metrics["running_target_exact"] is True
    assert metrics["running_target_margin"] == 4.0


def test_discovery_freezes_three_layers_by_margin_then_exact() -> None:
    layer = pd.DataFrame(
        [
            {
                "source_layer": 0,
                "mean_restoration_target_margin_gain": 1.0,
                "restored_exact_accuracy": 0.8,
            },
            {
                "source_layer": 4,
                "mean_restoration_target_margin_gain": 2.0,
                "restored_exact_accuracy": 0.7,
            },
            {
                "source_layer": 8,
                "mean_restoration_target_margin_gain": 2.0,
                "restored_exact_accuracy": 0.9,
            },
            {
                "source_layer": 12,
                "mean_restoration_target_margin_gain": 0.5,
                "restored_exact_accuracy": 1.0,
            },
        ]
    )
    frozen = _freeze_top_three(
        layer_metrics=layer,
        model_label="Qwen3-8B",
        plan={"seed_count": 20, "rows": []},
    )
    assert frozen["source_layers"] == [8, 4, 0]


def test_confirmation_averages_layers_with_seed_as_unit_and_holm_corrects() -> None:
    rows = []
    for seed in range(10):
        for occurrence in range(1, 11):
            for layer, effect in ((4, 1.0), (8, 2.0), (12, 3.0)):
                rows.append(
                    {
                        "seed": seed,
                        "target_occurrence": occurrence,
                        "source_layer": layer,
                        "source_exact": 1.0,
                        "blank_exact": 0.0,
                        "restored_exact": 1.0,
                        "restoration_exact_gain": 1.0,
                        "restoration_target_margin_gain": effect,
                    }
                )
    per_layer, seed_primary, summary = _confirmation_statistics(
        pd.DataFrame(rows), frozen_layers=(4, 8, 12)
    )
    assert len(seed_primary) == 10
    assert summary["mean_delta_margin"] == 2.0
    assert summary["positive_confirmation_support"] is True
    assert (per_layer["holm_p_across_three_frozen_layers"] < 0.05).all()
