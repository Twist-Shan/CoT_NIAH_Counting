import pytest

from scripts.run_realistic_niah_v5_boundary_history_equivariance import (
    _validate_layer_position_panel,
    boundary_history_geometries,
)


def test_boundary_history_geometries_are_nested_and_central() -> None:
    geometries = boundary_history_geometries(5)
    assert geometries == {
        "current_boundary": (5,),
        "current_plus_boundary_2": (2, 5),
        "current_plus_boundary_3": (3, 5),
        "last_2_boundaries": (4, 5),
        "last_3_boundaries": (3, 4, 5),
        "last_4_boundaries": (2, 3, 4, 5),
    }


def test_layer_position_panel_requires_shared_geometry() -> None:
    layers, positions = _validate_layer_position_panel(
        {14: {3: object(), 7: object()}, 15: {3: object(), 7: object()}}
    )
    assert layers == (14, 15)
    assert positions == (3, 7)
    with pytest.raises(ValueError, match="same token positions"):
        _validate_layer_position_panel(
            {14: {3: object()}, 15: {3: object(), 7: object()}}
        )
