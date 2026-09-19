from __future__ import annotations

import pytest

from realistic_niah_v5.tstar_prefix import build_tstar_prefix_context


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(
        self,
        values: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(value) for value in values)


class FixedSegmentTokenizer:
    def __init__(self, text: str, segments: list[str]) -> None:
        assert "".join(segments) == text
        self.text = text
        self.segments = segments
        self.ids = list(range(100, 100 + len(segments)))

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert text == self.text
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        cursor = 0
        offsets = []
        for segment in self.segments:
            offsets.append((cursor, cursor + len(segment)))
            cursor += len(segment)
        return {"input_ids": self.ids, "offset_mapping": offsets}

    def decode(
        self,
        values: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        by_id = dict(zip(self.ids, self.segments))
        return "".join(by_id[value] for value in values)


def _row(
    raw: str,
    output_ids: list[int],
    *,
    count: int,
    t_star_char: int,
    occurrences: list[tuple[str, int, int]],
    primary_eligible: bool = True,
) -> dict:
    audit_key = f"noindex_n{count}_format_audit"
    cohort_key = f"noindex_n{count}_cohort"
    return {
        "schema_version": "source-v1",
        "model_label": "Qwen3-8B",
        "model_family": "qwen",
        "seed": 1234,
        "split": "legacy_source_label",
        "request_id": "request/1234",
        "stimulus_id": "stimulus/1234",
        "gold_count": count,
        "gold_records": [
            {"city": city, "score": 50 + index}
            for index, (city, _start, _end) in enumerate(occurrences)
        ],
        "prompt_record_spans": [],
        "input_ids": [900, 901],
        "attention_mask": [1, 1],
        "output_token_ids": output_ids,
        "raw_output_text": raw,
        audit_key: {
            "primary_eligible_prefix_clean": primary_eligible,
            "coverage_complete": True,
            "first_pass_complete": True,
            "t_star_char": t_star_char,
            "first_occurrences": [
                {
                    "occurrence": index,
                    "city": city,
                    "char_start": start,
                    "char_end": end,
                }
                for index, (city, start, end) in enumerate(occurrences, start=1)
            ],
        },
        cohort_key: {"split": "discovery", "rank_within_split": 1},
    }


def test_builds_exact_prompt_plus_tstar_token_prefix() -> None:
    raw = "ABC numbered recap"
    tokenizer = CharacterTokenizer()
    row = _row(
        raw,
        [ord(char) for char in raw],
        count=3,
        t_star_char=3,
        occurrences=[("A", 0, 1), ("B", 1, 2), ("C", 2, 3)],
    )
    context = build_tstar_prefix_context(
        row,
        tokenizer,
        audit_key="noindex_n3_format_audit",
        cohort_key="noindex_n3_cohort",
        cohort_split="discovery",
    )
    assert context["raw_prefix_text"] == "ABC"
    assert context["output_prefix_token_ids"] == [ord("A"), ord("B"), ord("C")]
    assert context["input_ids"] == [900, 901, ord("A"), ord("B"), ord("C")]
    assert context["future_recap_available_to_context"] is False
    assert context["removed_output_char_count"] == len(" numbered recap")
    assert context["split"] == "discovery"


def test_uses_smallest_whole_token_and_records_boundary_spill() -> None:
    raw = "A B! recap"
    tokenizer = FixedSegmentTokenizer(raw, ["A", " B!", " recap"])
    row = _row(
        raw,
        tokenizer.ids,
        count=2,
        t_star_char=3,
        occurrences=[("A", 0, 1), ("B", 2, 3)],
    )
    context = build_tstar_prefix_context(
        row,
        tokenizer,
        audit_key="noindex_n2_format_audit",
        cohort_key="noindex_n2_cohort",
        cohort_split="discovery",
    )
    assert context["raw_prefix_text"] == "A B!"
    assert context["output_prefix_token_ids"] == [100, 101]
    assert context["token_boundary_right_spill_chars"] == 1
    assert context["token_boundary_right_spill_text"] == "!"
    assert context["removed_output_token_count"] == 1


def test_rejects_non_primary_row_and_split_disagreement() -> None:
    raw = "ABC recap"
    tokenizer = CharacterTokenizer()
    ineligible = _row(
        raw,
        [ord(char) for char in raw],
        count=3,
        t_star_char=3,
        occurrences=[("A", 0, 1), ("B", 1, 2), ("C", 2, 3)],
        primary_eligible=False,
    )
    with pytest.raises(ValueError, match="not first-occurrence prefix-clean"):
        build_tstar_prefix_context(
            ineligible,
            tokenizer,
            audit_key="noindex_n3_format_audit",
            cohort_key="noindex_n3_cohort",
        )

    eligible = dict(ineligible)
    eligible["noindex_n3_format_audit"] = dict(
        ineligible["noindex_n3_format_audit"],
        primary_eligible_prefix_clean=True,
    )
    with pytest.raises(ValueError, match="Manifest split disagrees"):
        build_tstar_prefix_context(
            eligible,
            tokenizer,
            audit_key="noindex_n3_format_audit",
            cohort_key="noindex_n3_cohort",
            cohort_split="confirmation",
        )


def test_rejects_replayed_evidence_before_tstar() -> None:
    raw = "ABC recap"
    tokenizer = CharacterTokenizer()
    row = _row(
        raw,
        [ord(char) for char in raw],
        count=3,
        t_star_char=3,
        occurrences=[("A", 0, 1), ("B", 1, 2), ("C", 2, 3)],
    )
    row["noindex_n3_format_audit"]["first_pass_complete"] = False
    with pytest.raises(ValueError, match="repeats score-supported evidence"):
        build_tstar_prefix_context(
            row,
            tokenizer,
            audit_key="noindex_n3_format_audit",
            cohort_key="noindex_n3_cohort",
        )
