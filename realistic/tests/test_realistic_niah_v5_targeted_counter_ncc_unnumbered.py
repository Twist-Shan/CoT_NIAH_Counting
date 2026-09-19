from __future__ import annotations

from types import SimpleNamespace

import pytest

from realistic_niah_v5.targeted_counter_ncc import transition_carrier_positions
from realistic_niah_v5.pipeline import read_jsonl
from realistic_niah_v5.encoding import NativeTraceEncoding
from realistic_niah_v4.prompts import TokenSpan
from realistic_niah_v5.unnumbered_counter_restore import (
    audit_qwen_thinking_bullets_final_total,
    audit_no_count_enumeration_trace,
    audit_unnumbered_trace,
    build_item_early_stop_encoding,
    inject_unnumbered_instruction,
)
from scripts.run_realistic_niah_v5_unnumbered_generation import (
    unnumbered_reasoning_text,
)
from scripts.run_realistic_niah_v5_frozen_prompt_bullet_pilot import (
    assert_frozen_prompt,
)
from scripts.run_realistic_niah_v5_natural_unnumbered_generation import (
    _rewrite_qwen_extraction_answer_task,
)
from realistic_niah_v5.generation import NativePrompt, build_v5_user_text


def _site(start: int, end: int) -> dict[str, object]:
    return {
        "status": "ok",
        "full_sequence_token_start": start,
        "full_sequence_token_end": end,
    }


def test_transition_carrier_positions_respects_grammar_timing() -> None:
    registry = SimpleNamespace(trace_items=((10, 20), (20, 32)))
    sites = {
        "rank_evidence_core_span": _site(22, 24),
        "city_target_span": _site(25, 27),
        "post_update_commit_state": _site(30, 31),
    }
    after = {"grammar_class": "adjacent_rank_after_city", "sites": sites}
    before = {"grammar_class": "same_unit_rank_before_city", "sites": sites}
    assert transition_carrier_positions(registry, after, occurrence=2) == (
        (22, 23),
        "marker_core",
        "rank_after_city",
    )
    assert transition_carrier_positions(registry, before, occurrence=2) == (
        (25, 26, 27, 28, 29, 30),
        "city_to_commit_tail",
        "rank_before_city",
    )


def _row(items: list[str], tail: str = "\n</think>\n\nTotal: ") -> dict[str, object]:
    prefix = "<think>\n"
    starts = []
    ends = []
    raw = prefix
    for item in items:
        starts.append(len(raw))
        raw += item
        ends.append(len(raw))
    answer_start = len(raw + tail)
    raw += tail + "3"
    return {
        "raw_output_text": raw,
        "gold_count": len(items),
        "trace_parse": {
            "parser": {
                "trace_one_to_one": True,
                "marker_kind": "bullet",
                "item_count": len(items),
                "item_start_chars": starts,
                "item_end_chars": ends,
                "reasoning_start_char": len(prefix),
            },
            "char_sites": [
                {
                    "site_kind": "answer_query",
                    "char_start": answer_start,
                    "char_end": answer_start,
                }
            ],
        },
    }


def test_unnumbered_audit_allows_scores_but_rejects_running_labels() -> None:
    good = _row(
        [
            "- In the 2024 city score audit, Paris received 61.\n",
            "- Tokyo: 72\n",
            "- Lima: 48\n",
        ]
    )
    assert audit_unnumbered_trace(good)["eligible"] is True

    labeled = _row(["- Record 1: Paris, 61\n", "- Tokyo: 72\n"])
    result = audit_unnumbered_trace(labeled)
    assert result["eligible"] is False
    assert "labeled_index:1" in result["reasons"]

    leaked = _row(["- Paris: 61\n", "- Tokyo: 72\n"], tail="\nTwo records total.\nTotal: ")
    assert "pre_answer_tail_count_leak" in audit_unnumbered_trace(leaked)["reasons"]


def test_causal_prefix_audit_allows_bullets_and_future_total() -> None:
    good = _row(
        [
            "- Paris: 61\n",
            "- Tokyo: 72\n",
            "- Lima: 48\n",
        ],
        tail="\nThat's three records.\n</think>\n\nTotal: ",
    )
    result = audit_no_count_enumeration_trace(good)
    assert result["eligible"] is True
    assert result["future_text_after_each_item_not_a_causal_exclusion"] is True


def test_causal_prefix_audit_rejects_late_bullet_and_record_labels() -> None:
    late = _row(["- Paris: 61\n", "- Tokyo: 72\n"])
    late["raw_output_text"] = str(late["raw_output_text"]).replace(
        "<think>\n", "<think>\nThat's two records.\n"
    )
    shift = len("That's two records.\n")
    parser = late["trace_parse"]["parser"]
    parser["item_start_chars"] = [value + shift for value in parser["item_start_chars"]]
    parser["item_end_chars"] = [value + shift for value in parser["item_end_chars"]]
    assert audit_no_count_enumeration_trace(late)["eligible"] is False

    labeled = _row(["- Record 1: Paris, 61\n", "- Record 2: Tokyo, 72\n"])
    assert audit_no_count_enumeration_trace(labeled)["eligible"] is False


def test_causal_prefix_audit_allows_unnumbered_audit_sentences() -> None:
    row = _row(
        [
            '"In the audit, Paris received a score of 61."\n',
            '"In the audit, Tokyo received a score of 72."\n',
        ]
    )
    row["trace_parse"]["parser"]["marker_kind"] = "audit_sentence"
    assert audit_no_count_enumeration_trace(row)["eligible"] is True


def test_unnumbered_instruction_is_inserted_after_passage() -> None:
    source = "before <passage>needle</passage> after"
    value = inject_unnumbered_instruction(source, attempt=1)
    assert value.startswith("before <passage>needle</passage>")
    assert "unnumbered dash bullet" in value
    end_priority = inject_unnumbered_instruction(source, attempt=4)
    assert end_priority.startswith(source)
    assert "IMPORTANT REASONING FORMAT" in end_priority
    with pytest.raises(ValueError):
        inject_unnumbered_instruction(source, attempt=99)


def test_qwen_channel_audit_rejects_repeated_final_bullets() -> None:
    row = _row(["- Paris: 61\n", "- Tokyo: 72\n"])
    row["gold_records"] = [{"city": "Paris"}, {"city": "Tokyo"}]
    row["raw_output_text"] = (
        "- Paris: 61\n- Tokyo: 72\nTotal: 2\n</think>\n\n"
        "- Paris: 61\n- Tokyo: 72\nTotal: 2<|im_end|>"
    )
    result = audit_qwen_thinking_bullets_final_total(row)
    assert result["eligible"] is False
    assert "total_inside_thinking" in result["reasons"]
    assert "bullet_repeated_in_final" in result["reasons"]


def test_qwen_channel_audit_accepts_bullets_then_total_only() -> None:
    row = _row(["- Paris: 61\n", "- Tokyo: 72\n"])
    row["gold_records"] = [{"city": "Paris"}, {"city": "Tokyo"}]
    row["raw_output_text"] = (
        "- Paris: 61\n- Tokyo: 72\n</think>\n\nTotal: 2<|im_end|>"
    )
    result = audit_qwen_thinking_bullets_final_total(row)
    assert result["eligible"] is True
    assert result["final_total_only"] is True
    assert result["final_count_correct"] is True
    assert result["prompt_selection_uses_final_count_correctness"] is False


def test_frozen_prompt_pilot_rejects_any_user_text_change() -> None:
    passage = "Paris received a score of 61."
    expected = build_v5_user_text(passage)
    prompt = NativePrompt(
        stimulus_id="s",
        design_variant="v4.4",
        seed=3000,
        split="discovery",
        gold_count=1,
        model_label="Qwen3-8B",
        model_family="qwen3",
        entity_domain="city",
        user_text=expected,
        rendered_prompt="rendered",
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        gold_records=(),
        prompt_record_spans=(),
    )
    assert assert_frozen_prompt({"passage": passage}, prompt)["status"] == "PASS"
    changed = NativePrompt(**{**prompt.__dict__, "user_text": expected + "\nUse bullets."})
    with pytest.raises(ValueError, match="PROMPT_INTEGRITY_FAILURE"):
        assert_frozen_prompt({"passage": passage}, changed)


def test_official_jsonl_reader_preserves_unicode_line_separator(tmp_path) -> None:
    # str.splitlines() incorrectly treats U+2028 inside a JSON string as a row
    # boundary.  The production reader iterates physical newline-delimited rows.
    path = tmp_path / "unicode-separator.jsonl"
    path.write_text('{"text":"left\u2028right"}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"text": "left\u2028right"}]


def test_teacher_forced_reasoning_has_evidence_but_no_running_index() -> None:
    row = {
        "gold_records": [
            {"city": "Paris", "score": 61},
            {"city": "Tokyo", "score": 72},
        ]
    }
    assert unnumbered_reasoning_text(row) == (
        "\n- Paris: score 61\n- Tokyo: score 72\n"
    )


def test_qwen_extraction_answer_rewrite_removes_conflicting_native_tail() -> None:
    original = build_v5_user_text("A short passage.")
    rewritten = _rewrite_qwen_extraction_answer_task(original, attempt=16)
    assert rewritten.startswith("You will need to count all city-score")
    assert "<passage>\nA short passage.\n</passage>" in rewritten
    assert "Reason concisely without repeating or restarting." not in rewritten
    assert "THINKING JOB — EXTRACT ONLY" in rewritten
    assert rewritten.count("Total: <integer>") == 1


def test_item_early_stop_removes_future_items_and_keeps_terminal_suffix() -> None:
    spans = tuple(
        TokenSpan(
            slot_index=index,
            start=start,
            end=end,
            active=True,
            kind="trace_item",
            canonical_length=end - start,
            model_token_length=end - start,
        )
        for index, (start, end) in enumerate(((4, 7), (7, 11), (11, 14)))
    )
    encoding = NativeTraceEncoding(
        stimulus_id="s",
        request_id="r",
        design_variant="v4.4",
        seed=1,
        split="discovery",
        count=3,
        model_label="Qwen3-8B",
        model_family="qwen3",
        answer_format="numeric",
        text="",
        generation_prompt="",
        input_ids=tuple(range(18)),
        attention_mask=(1,) * 18,
        query_position=17,
        prompt_token_count=4,
        raw_prefix_text="",
        selected_site={},
        prompt_record_spans=(),
        trace_item_spans=spans,
        slot_spans=spans,
        needle_spans=spans,
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )
    registry = SimpleNamespace(trace_items=((4, 7), (7, 11), (11, 14)), query_position=17)
    early, audit = build_item_early_stop_encoding(
        encoding, registry, target_occurrence=2
    )
    assert early.input_ids == tuple(range(11)) + tuple(range(14, 18))
    assert early.query_position == 14
    assert early.input_ids[:11] == encoding.input_ids[:11]
    assert len(early.trace_item_spans) == 2
    assert audit["future_trace_items_removed"] == 1
    assert audit["future_trace_token_count_removed"] == 3
    assert audit["future_trace_tokens_present"] is False
