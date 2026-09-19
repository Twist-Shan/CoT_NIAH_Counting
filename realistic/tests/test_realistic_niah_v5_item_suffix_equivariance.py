import numpy as np
import pytest

from scripts.run_realistic_niah_v5_item_suffix_equivariance import (
    item_suffix_geometries,
    suffix_positions_for_occurrences,
    union_subspace_basis,
)


class _Registry:
    trace_items = ((3, 9), (10, 18), (20, 25), (26, 33), (34, 42))


def test_item_suffix_geometries_distinguish_anchor_from_contiguous_history() -> None:
    assert item_suffix_geometries(5) == {
        "current_item_suffix": (5,),
        "current_plus_item_2_suffix": (2, 5),
        "last_2_item_suffixes": (4, 5),
        "last_3_item_suffixes": (3, 4, 5),
        "last_4_item_suffixes": (2, 3, 4, 5),
    }


def test_suffix_positions_are_endpoint_aligned_and_nonoverlapping() -> None:
    assert suffix_positions_for_occurrences(
        _Registry(), (2, 5), width=4
    ) == {2: (14, 15, 16, 17), 5: (38, 39, 40, 41)}
    with pytest.raises(ValueError, match="shorter"):
        suffix_positions_for_occurrences(_Registry(), (3,), width=6)


def test_union_subspace_basis_spans_both_inputs_without_duplicate_rank() -> None:
    first = np.eye(5, dtype=np.float32)[:, :2]
    second = np.eye(5, dtype=np.float32)[:, 1:4]
    union = union_subspace_basis(first, second)
    assert union.shape == (5, 4)
    np.testing.assert_allclose(union.T @ union, np.eye(4), atol=1e-6)
    projector = union @ union.T
    np.testing.assert_allclose(projector @ first, first, atol=1e-6)
    np.testing.assert_allclose(projector @ second, second, atol=1e-6)
