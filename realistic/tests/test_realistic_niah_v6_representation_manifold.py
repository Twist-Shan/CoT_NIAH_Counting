from __future__ import annotations

import numpy as np
import pandas as pd

from realistic_niah_v6.representation_manifold import project_layer


def test_manifold_fits_discovery_and_emits_confirmation_only() -> None:
    rng = np.random.default_rng(17)
    discovery = rng.normal(size=(20, 8))
    confirmation_a = rng.normal(size=(10, 8))
    confirmation_b = rng.normal(loc=50.0, scale=9.0, size=(10, 8))
    metadata = pd.DataFrame(
        {
            "split": ["discovery"] * 20 + ["confirmation"] * 10,
            "seed": list(range(20)) + list(range(100, 110)),
            "occurrence": [value % 10 + 1 for value in range(20)]
            + list(range(1, 11)),
        }
    )

    first = project_layer(
        np.vstack([discovery, confirmation_a]),
        metadata,
        label_column="occurrence",
    )
    shifted_confirmation = project_layer(
        np.vstack([discovery, confirmation_b]),
        metadata,
        label_column="occurrence",
    )

    assert first["discovery_rows"] == 20
    assert first["confirmation_rows"] == 10
    assert len(first["rows"]) == 10
    assert {row[0] for row in first["rows"]} == set(range(100, 110))
    assert first["evr"] == shifted_confirmation["evr"]
    assert first["axis_signs"] == shifted_confirmation["axis_signs"]
    assert first["rows"] != shifted_confirmation["rows"]


def test_manifold_projection_is_deterministic() -> None:
    rng = np.random.default_rng(23)
    states = rng.normal(size=(30, 7))
    metadata = pd.DataFrame(
        {
            "split": ["discovery"] * 20 + ["confirmation"] * 10,
            "seed": list(range(30)),
            "gold_count": [value % 10 + 1 for value in range(30)],
        }
    )
    first = project_layer(states, metadata, label_column="gold_count")
    second = project_layer(states, metadata, label_column="gold_count")
    assert first == second
