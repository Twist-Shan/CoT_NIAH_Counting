from __future__ import annotations

from dataset_generation.dynamic_niah import TokenizerAdapter
from realistic_niah.parsing import (
    evaluate_generation,
    split_reasoning_and_final,
)
from realistic_niah.prompts import COMMON_COUNTING_CUE, build_messages
from realistic_niah.spec import (
    FORMAL_PROMPT_MODES,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    QUERY_LAYOUT,
    SEEDS,
    SMOKE_NEEDLE_COUNTS,
    SMOKE_SEEDS,
)
from realistic_niah.stimuli import (
    FreezeSpec,
    audit_frozen_grid,
    freeze_grid,
    freeze_stimulus,
)


def test_registered_v2_grid_contains_500_shared_stimuli() -> None:
    assert 0 not in NEEDLE_COUNTS
    assert 6 in NEEDLE_COUNTS
    assert NEEDLE_COUNTS == (1, 2, 3, 4, 5, 6, 8, 10, 20, 30)
    assert PASSAGE_LENGTHS == (2_000, 3_000, 5_000, 10_000, 20_000)
    assert SEEDS == tuple(range(1234, 1244))
    assert len(PASSAGE_LENGTHS) * len(NEEDLE_COUNTS) * len(SEEDS) == 500
    assert SMOKE_NEEDLE_COUNTS == (6, 20, 30)
    assert SMOKE_SEEDS == (2234, 2235)


def test_v2_prompts_share_cue_and_fixed_query_after_layout() -> None:
    passage = "Alpha passage."
    contents = {
        mode: build_messages(passage, prompt_mode=mode)[0]["content"]
        for mode in FORMAL_PROMPT_MODES
    }

    assert QUERY_LAYOUT == "cue_before_query_after"
    assert len(set(contents.values())) == 4
    for content in contents.values():
        assert content.startswith(COMMON_COUNTING_CUE)
        assert content.index(COMMON_COUNTING_CUE) < content.index("<passage>")
        assert content.index("</passage>") < content.index("How many")
        assert "count all city-score audit records" in content
    assert 'Begin the first item with "1. "' in (
        contents["enumeration_index"]
    )
    assert "do not output placeholders or angle brackets" in (
        contents["enumeration_index"]
    )
    assert 'Begin each item with "-"' in contents["enumeration_bullet"]
    assert "do not output placeholders or angle brackets" in (
        contents["enumeration_bullet"]
    )
    assert "Reason concisely without repeating or restarting" in (
        contents["native_thinking"]
    )
    assert "Stop as soon as you determine the count" in (
        contents["native_thinking"]
    )


def test_indexed_enumeration_parser_handles_six_records() -> None:
    gold = [
        {"city": f"City {index}", "score": 50 + index}
        for index in range(1, 7)
    ]
    output = "\n".join(
        [
            *(f"{index}. City {index}: {50 + index}" for index in range(1, 7)),
            "Total: 6",
        ]
    )
    result = evaluate_generation(
        output,
        prompt_mode="enumeration_index",
        gold_pairs=gold,
        finish_reason="stop",
    )

    assert result["predicted_count"] == 6
    assert result["exact_count"] is True
    assert result["registered_success"] is True
    assert result["pair_recall"] == 1.0
    assert result["listed_total_matches_length"] is True
    assert result["enumeration_format_status"] == "ok"
    assert result["enumeration_format_compliant"] is True


def test_bullet_enumeration_has_independent_strict_parser() -> None:
    gold = [
        {"city": "Madison", "score": 71},
        {"city": "Milwaukee", "score": 83},
    ]
    result = evaluate_generation(
        "- Madison: 71\n- Milwaukee: 83\nTotal: 2",
        prompt_mode="enumeration_bullet",
        gold_pairs=gold,
        finish_reason="stop",
    )

    assert result["exact_count"] is True
    assert result["registered_success"] is True
    assert result["pair_recall"] == 1.0
    assert result["enumeration_format_status"] == "ok"
    assert result["enumeration_format_compliant"] is True


def test_wrong_enumeration_marker_preserves_semantic_pair_audit() -> None:
    gold = [
        {"city": "Madison", "score": 71},
        {"city": "Milwaukee", "score": 83},
    ]
    result = evaluate_generation(
        "1. Madison: 71\n2. Milwaukee: 83\nTotal: 2",
        prompt_mode="enumeration_bullet",
        gold_pairs=gold,
        finish_reason="stop",
    )

    assert result["exact_count"] is True
    assert result["registered_success"] is False
    assert result["pair_recall"] == 1.0
    assert result["enumeration_format_status"] == "wrong_marker"
    assert result["enumeration_format_compliant"] is False
    assert result["response_format_compliant"] is False
    assert result["strict_listed_records"] == []


def test_native_thinking_restarts_are_audited() -> None:
    output = """\
<think>
1. Madison: 71
2. Milwaukee: 83
1. Madison: 71
2. Milwaukee: 83
</think>
Total: 2"""
    result = evaluate_generation(
        output,
        prompt_mode="native_thinking",
        gold_pairs=[
            {"city": "Madison", "score": 71},
            {"city": "Milwaukee", "score": 83},
        ],
        finish_reason="stop",
        output_tokens=120,
        max_output_tokens=4096,
    )

    assert result["exact_count"] is True
    assert result["registered_success"] is True
    assert result["reasoning_enumeration_restart_count"] == 1
    assert result["reasoning_duplicate_record_mentions"] == 2
    assert result["overthinking_flag"] is True
    assert "enumeration_restart" in result["overthinking_signals"]
    assert result["output_budget_fraction"] == 120 / 4096


def test_qwen_reasoning_and_final_are_separated() -> None:
    reasoning, final = split_reasoning_and_final(
        "<think>\nI found six records.\n</think>\n\nTotal: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == "I found six records."
    assert final == "Total: 6"


def test_qwen_prompt_supplied_opening_think_token_is_supported() -> None:
    reasoning, final = split_reasoning_and_final(
        "I found six records.\n</think>\n\nTotal: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == "I found six records."
    assert final == "Total: 6"


def test_always_on_reasoning_is_split_in_a_direct_prompt_mode() -> None:
    reasoning, final = split_reasoning_and_final(
        "I found six records.\n</think>\n\nTotal: 6",
        prompt_mode="direct",
        reasoning_expected=True,
    )

    assert reasoning == "I found six records."
    assert final == "Total: 6"


def test_deepseek_and_glm_end_tokens_are_accepted_after_total() -> None:
    gold = [
        {"city": "Madison", "score": 71},
        {"city": "Milwaukee", "score": 83},
    ]
    for raw_text in (
        "Total: 2<｜end▁of▁sentence｜>",
        "Total: 2<|endoftext|>",
    ):
        result = evaluate_generation(
            raw_text,
            prompt_mode="direct",
            gold_pairs=gold,
            finish_reason="stop",
        )

        assert result["predicted_count"] == 2
        assert result["registered_success"] is True


def test_gemma_reasoning_and_final_are_separated() -> None:
    reasoning, final = split_reasoning_and_final(
        "<|channel>thought\nI found six records.<channel|>Total: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == "I found six records."
    assert final == "Total: 6"


def test_gemma_prompt_supplied_opening_thought_channel_is_supported() -> None:
    reasoning, final = split_reasoning_and_final(
        "I found six records.<channel|>Total: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == "I found six records."
    assert final == "Total: 6"


def test_backend_parsed_native_response_is_treated_as_final() -> None:
    reasoning, final = split_reasoning_and_final(
        "Total: 6",
        prompt_mode="native_thinking",
    )

    assert reasoning == ""
    assert final == "Total: 6"


def test_fixed_post_insertion_length_with_simple_tokenizer() -> None:
    tokenizer = TokenizerAdapter("simple")
    spec = FreezeSpec(
        passage_lengths=(180,),
        needle_counts=(6,),
        seeds=(1234,),
        canonical_tokenizer="simple",
        max_search_attempts=12,
        max_window_retries=2,
        minimum_filler_tokens=40,
    )

    row = freeze_stimulus(
        target_passage_tokens=180,
        num_needles=6,
        seed=1234,
        tokenizer=tokenizer,
        spec=spec,
    )

    assert row["canonical_passage_tokens"] == 180
    assert row["gold_count"] == 6
    assert len(row["needles"]) == 6
    assert row["length_search"]["post_insertion_truncation"] is False
    assert row["haystack"]["source_mode"] == "multi_file_no_repeat"
    assert row["haystack"]["source_repeated_to_target"] is False
    assert row["haystack"]["source_repeat_count"] == 1


def test_frozen_grid_passes_independent_audit(tmp_path) -> None:
    spec = FreezeSpec(
        passage_lengths=(180,),
        needle_counts=(6,),
        seeds=(1234,),
        canonical_tokenizer="simple",
        max_search_attempts=12,
        max_window_retries=2,
        minimum_filler_tokens=40,
    )
    paths = freeze_grid(
        output_dir=tmp_path,
        spec=spec,
        require_huggingface_tokenizer=False,
    )

    report = audit_frozen_grid(
        stimuli_path=paths["stimuli"],
        manifest_path=paths["manifest"],
        require_huggingface_tokenizer=False,
    )

    assert report["passed"] is True
    assert report["rows_checked"] == 1
    assert report["needle_counts"] == [6]
