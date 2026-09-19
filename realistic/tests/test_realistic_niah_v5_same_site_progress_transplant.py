from __future__ import annotations

from dataclasses import dataclass

import pytest

from realistic_niah_v5.same_site_progress_transplant import (
    canonical_marker_bits,
    donor_receiver_logodds,
    generated_bullet_city_ordinals,
    native_item_candidates,
    query_prefix_before_city,
    select_count_cells,
)


def test_canonical_marker_bits_and_alternative_cells() -> None:
    variants = [
        {"marker_bits": bits, "name": "".join(map(str, bits))}
        for bits in (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        )
    ]
    selected = select_count_cells(
        variants, factor_count=3, donor_valid_counts=(1, 2, 3)
    )
    assert canonical_marker_bits(3, 0) == (0, 0, 0)
    assert selected[1]["primary"]["marker_bits"] == (0, 0, 1)
    assert selected[1]["alternative"]["marker_bits"] == (1, 0, 0)
    assert selected[2]["primary"]["marker_bits"] == (0, 1, 1)
    assert selected[2]["alternative"]["marker_bits"] == (1, 1, 0)
    assert selected[3]["alternative"] is None


def test_marker_bit_contract_rejects_out_of_range_counts() -> None:
    with pytest.raises(ValueError):
        canonical_marker_bits(3, 4)
    with pytest.raises(ValueError):
        canonical_marker_bits(0, 0)


@dataclass(frozen=True)
class _Encoding:
    input_ids: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


def test_native_candidates_and_city_query_prefix() -> None:
    encoding = _Encoding((9, 1, 2, 3, 8, 4, 5, 6, 7))
    candidates = native_item_candidates(encoding, ((1, 4), (5, 9)))
    assert candidates == {1: (1, 2, 3), 2: (4, 5, 6, 7)}
    assert query_prefix_before_city(candidates[2], (6, 7)) == (4, 5)
    with pytest.raises(ValueError):
        query_prefix_before_city(candidates[1], (8,))


def test_generated_city_parser_uses_only_bullet_lines_before_close() -> None:
    result = generated_bullet_city_ordinals(
        "- Beta has score 4.\n"
        "- Gamma has score 9.\n"
        "Summary: Alpha, Beta, Gamma.\n"
        "</think>\n"
        "- Alpha after close should not count.\n",
        ("Alpha", "Beta", "Gamma"),
    )
    assert result["generated_bullet_city_ordinals"] == [2, 3]
    assert result["first_generated_known_city_ordinal"] == 2
    assert result["generated_known_city_bullet_count"] == 2
    assert result["first_generated_bullet_city_ordinal"] == 2
    assert result["reasoning_close_observed"] is True


def test_generated_city_parser_accepts_native_nonbullet_items() -> None:
    result = generated_bullet_city_ordinals(
        " Brussels received a score.\n\n, Beijing received a score.</think>",
        ("Brussels", "Beijing", "Barcelona"),
    )
    assert result["generated_known_city_ordinals_any_surface"] == [1, 2]
    assert result["first_generated_known_city_ordinal"] == 1


def test_donor_receiver_logodds_uses_one_based_ordinals() -> None:
    assert donor_receiver_logodds(
        (-5.0, -3.0, -1.0), donor_successor=3, receiver_successor=2
    ) == 2.0
