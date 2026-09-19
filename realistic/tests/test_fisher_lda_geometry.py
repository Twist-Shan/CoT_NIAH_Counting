from __future__ import annotations

import numpy as np
import pandas as pd

from realistic_niah_v5.fisher_lda_geometry import discovery_fitted_fisher_lda3


def _fixture() -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(11)
    rows = []
    states = []
    for split, seeds in (("discovery", range(6)), ("confirmation", range(6, 9))):
        for seed in seeds:
            for label in range(1, 5):
                center = np.array([label * 1.7, label * -0.8, label * 0.4, 0, 0, 0])
                states.append(center + rng.normal(scale=0.45, size=6))
                rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "occurrence": label,
                        "gold_count": label,
                    }
                )
    return np.asarray(states, dtype=np.float32), pd.DataFrame(rows)


def test_fisher_lda3_is_discovery_fitted() -> None:
    states, metadata = _fixture()
    first = discovery_fitted_fisher_lda3(
        states, metadata, (1, 2, 3, 4), pca_dim=6
    )
    changed = states.copy()
    confirmation = metadata["split"].eq("confirmation").to_numpy()
    changed[confirmation] += 500.0
    second = discovery_fitted_fisher_lda3(
        changed, metadata, (1, 2, 3, 4), pca_dim=6
    )

    assert np.allclose(
        first["fit"]["fisher_eigenvalues"],
        second["fit"]["fisher_eigenvalues"],
    )
    first_discovery = [row[3:6] for row in first["points"] if row[0] == "discovery"]
    second_discovery = [row[3:6] for row in second["points"] if row[0] == "discovery"]
    assert np.allclose(first_discovery, second_discovery)


def test_fisher_lda3_exports_frozen_confirmation_diagnostics() -> None:
    states, metadata = _fixture()
    payload = discovery_fitted_fisher_lda3(
        states, metadata, (1, 2, 3, 4), pca_dim=6
    )

    assert payload["fit"]["fit_split"] == "discovery only"
    assert 0.0 < payload["fit"]["top3_fisher_trace_fraction"] <= 1.0
    assert len(payload["discovery_centroids"]) == 4
    assert len(payload["confirmation_centroids"]) == 4
    assert len(payload["points"]) == len(metadata)
    assert np.isfinite(
        payload["metrics"]["confirmation_lda3_class_balanced_silhouette"]
    )
    assert np.isfinite(payload["metrics"]["confirmation_lda3_radius_gap_ratio"])
