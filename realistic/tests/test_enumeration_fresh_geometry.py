import pytest

from scripts.enumeration_fresh_geometry import (
    answer_blank_positions, common_eligible_seeds, enumeration_format_eligible,
)


def test_read_masks_preserve_bos_query_and_allow_item_free_answer():
    masks = answer_blank_positions(prompt_count=6, query=9, sequence_length=10,
                                   record_spans=[(2, 4)])
    assert masks["clean"] == ()
    assert masks["prompt_records_blank"] == (2, 3)
    assert masks["prompt_all_blank"] == (1, 2, 3, 4, 5)
    assert masks["trace_all_blank"] == (6, 7, 8)
    assert masks["prompt_and_trace_blank"] == tuple(range(1, 9))
    assert all(0 not in mask and 9 not in mask for mask in masks.values())


def test_read_can_have_empty_trace_region():
    masks = answer_blank_positions(prompt_count=6, query=6, sequence_length=7,
                                   record_spans=[(2, 4)])
    assert masks["trace_all_blank"] == ()
    assert masks["prompt_all_blank"] == masks["prompt_and_trace_blank"]


@pytest.mark.parametrize("spans", [[(0, 2)], [(4, 7)], [], [(4, 4)]])
def test_read_rejects_invalid_record_geometry(spans):
    with pytest.raises(ValueError):
        answer_blank_positions(prompt_count=6, query=9, sequence_length=10,
                               record_spans=spans)


def test_enumeration_gate_does_not_filter_wrong_final_answer():
    parsed = {key: True for key in ("enumeration_format_compliant",
              "exact_ordered_gold_pairs", "marker_kind_compliant", "parser_forward_one_to_one")}
    assert enumeration_format_eligible(dict(parsed, exact_count=False, strict_causal_eligible=False))
    assert enumeration_format_eligible(dict(parsed, exact_count=True, strict_causal_eligible=True))
    assert not enumeration_format_eligible(dict(parsed, marker_kind_compliant=False))


def test_common_cohort_keeps_fixed_order_and_shortfall():
    a = [{"seed": s, "gold_count": 10, "update_eligible": s != 2,
          "intervention_success": False} for s in (1, 2, 3)]
    b = [{"seed": s, "gold_count": 10, "update_eligible": s != 3,
          "intervention_success": True} for s in (3, 2, 1)]
    assert common_eligible_seeds([a, b], [3, 2, 1], field="update_eligible", quota=10) == [1]
    with pytest.raises(ValueError, match="Unaccounted"):
        common_eligible_seeds([a, b], [1, 4], field="update_eligible", quota=10)
    with pytest.raises(ValueError, match="Duplicate"):
        common_eligible_seeds([a + [a[0]], b], [1, 2, 3], field="update_eligible", quota=10)
