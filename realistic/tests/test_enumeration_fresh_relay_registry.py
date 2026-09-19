from dataclasses import dataclass, replace
import pytest
from scripts.prepare_enumeration_fresh_relay import candidate_pairs, common_width_pairs, rebase_actual_answer_geometry, literal_decoded_offsets, relay_baseline_exclusion


def test_native_relay_does_not_inherit_update_noindex_or_final_accuracy_gate():
    entry = {"read_eligible": True, "format_eligible": False, "baseline_exact_count": False}
    parsed = {"detected": True, "trace_one_to_one": True, "trace_order_class": "backward"}
    assert relay_baseline_exclusion(entry, parsed, model="Qwen3-8B", mode="thinking",
        cohort_policy="native_assay_specific_per_mode") is None
    assert relay_baseline_exclusion(entry, parsed, model="Qwen3-8B", mode="thinking",
        cohort_policy="legacy_common_update_gate") is not None
    assert relay_baseline_exclusion(entry, {**parsed, "trace_one_to_one": False}, model="Qwen3-8B", mode="thinking",
        cohort_policy="native_assay_specific_per_mode") == "Native relay cohort: not_one_to_one"


def test_literal_offsets_do_not_require_canonical_retokenization():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join({1: "FOUND:", 2: " ", 3: "Paris", 4: "\n"}[value] for value in ids)
    assert literal_decoded_offsets(Tokenizer(), "FOUND: Paris\n", (1, 2, 3, 4)) == ((0, 6), (6, 7), (7, 12), (12, 13))
    with pytest.raises(ValueError, match="frozen raw text"):
        literal_decoded_offsets(Tokenizer(), "FOUND: Paris\nextra", (1, 2, 3, 4))


def test_literal_offsets_reject_unstable_partial_unicode():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "\ufffd" if len(ids) == 1 else "\u00e9"
    with pytest.raises(ValueError, match="non-monotone"):
        literal_decoded_offsets(Tokenizer(), "\u00e9", (1, 2))


def test_terminal_panel_preserves_historical_count_offset_cells():
    ranges = {"-1": list(range(2, 11)), "-3": list(range(5, 11)), "-5": list(range(7, 11))}
    pairs = candidate_pairs([11, 10], ranges)
    assert len(pairs) == 38
    assert len({row["pair_id"] for row in pairs}) == 38
    assert all(row["receiver_occurrence"] == row["gold_count"] for row in pairs)
    assert all(row["donor_occurrence"] == row["gold_count"] + row["donor_offset"] for row in pairs)
    with pytest.raises(ValueError):
        candidate_pairs([10, 10], ranges)
    with pytest.raises(ValueError):
        candidate_pairs([10], {"-3": [2]})


def test_width_comparison_uses_joint_support_without_shrinking_primary():
    a = [{"pair_id": "a", "widths": {"4": {"eligible": True}, "8": {"eligible": True}}},
         {"pair_id": "b", "widths": {"4": {"eligible": True}, "8": {"eligible": False}}}]
    b = [{"pair_id": "a", "widths": {"4": {"eligible": True}, "8": {"eligible": True}}},
         {"pair_id": "b", "widths": {"4": {"eligible": True}, "8": {"eligible": True}}}]
    assert common_width_pairs([a, b], widths=[4, 8]) == {"4": ["a", "b"], "8": ["a"], "joint_widths": ["a"]}
    with pytest.raises(ValueError, match="same candidate"):
        common_width_pairs([a, b[:1]], widths=[4, 8])


def test_found_geometry_retains_actual_query_and_all_original_tokens():
    @dataclass
    class Encoding:
        input_ids: tuple
        attention_mask: tuple
        query_position: int
        sequence_length: int
        prompt_token_count: int = 2
        trace_item_spans: tuple = ((2, 4), (5, 7))
        slot_spans: tuple = ((2, 4), (5, 7))
        needle_spans: tuple = ((2, 4), (5, 7))
    @dataclass
    class Registry:
        prompt_token_count: int = 2
        query_position: int = 9
        sequence_length: int = 10
        answer_site_id: str = "synthetic"
        trace_items: tuple = ((2, 4), (5, 7))
        trace_context: tuple = ((2, 9),)
        trace_other: tuple = ((4, 5), (7, 9))
        def validate(self):
            assert self.query_position == self.sequence_length - 1
            assert self.trace_context == ((self.prompt_token_count, self.query_position),)
    first = Encoding(tuple(range(10)), (1,) * 10, 9, 10)
    actual = Encoding(tuple(range(7)) + (91, 92), (1,) * 9, 8, 9,
                      trace_item_spans=(), slot_spans=(), needle_spans=())
    encoded, registry = rebase_actual_answer_geometry(first, Registry(), actual)
    assert encoded.input_ids == actual.input_ids
    assert encoded.attention_mask == actual.attention_mask
    assert encoded.query_position == actual.query_position == 8
    assert encoded.trace_item_spans == first.trace_item_spans
    assert registry.trace_other == ((4, 5), (7, 8))
    assert registry.answer_site_id == "answer_query_v3"
    with pytest.raises(ValueError, match="literal source tokens"):
        rebase_actual_answer_geometry(first, Registry(), replace(actual, input_ids=(99,) + actual.input_ids[1:]))
