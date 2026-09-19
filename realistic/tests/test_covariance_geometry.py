from __future__ import annotations

import numpy as np
import pandas as pd

from realistic_niah_v5.covariance_geometry import (
    class_balanced_scatter,
    fisher_trace,
    ordinal_centroid_rsa,
    select_discovery_winners,
)


def test_class_balanced_scatter_does_not_weight_large_classes_more() -> None:
    values = np.asarray(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [9.0, 0.0],
            [11.0, 0.0],
            [10.0, 0.0],
            [10.0, 0.0],
        ]
    )
    labels = np.asarray([1, 1, 2, 2, 2, 2])
    scatter = class_balanced_scatter(values, labels, (1, 2))
    np.testing.assert_allclose(scatter.centroids[:, 0], [0.0, 10.0])
    np.testing.assert_allclose(scatter.grand_centroid, [5.0, 0.0])
    np.testing.assert_allclose(scatter.between[0, 0], 25.0)
    np.testing.assert_allclose(scatter.within[0, 0], 0.75)


def test_fisher_trace_accounts_for_anisotropic_noise() -> None:
    values = np.asarray(
        [
            [-10.0, -2.0],
            [10.0, -2.0],
            [-10.0, 2.0],
            [10.0, 2.0],
        ]
    )
    labels = np.asarray([1, 1, 2, 2])
    scatter = class_balanced_scatter(values, labels, (1, 2))
    value, _ridge, _condition = fisher_trace(scatter)
    # Separation is entirely on y, while the large x variance is nuisance.
    assert value > 10_000


def test_ordinal_centroid_rsa_is_one_for_equally_spaced_counts() -> None:
    values = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    labels = np.asarray([1, 2, 3, 4])
    assert ordinal_centroid_rsa(values, labels, (1, 2, 3, 4)) == 1.0


def test_layer_selection_never_reads_confirmation_for_ranking() -> None:
    rows = []
    for layer, discovery, confirmation in ((1, 0.8, 0.1), (2, 0.7, 0.9)):
        row = {
            "model_label": "model",
            "endpoint": "running_index",
            "mode": "non_thinking",
            "layer": layer,
            "pca_components": 2,
            "discovery_rows": 20,
            "confirmation_rows": 10,
        }
        for prefix in (
            "oof_isotropic_snr_db",
            "oof_fisher_trace",
            "oof_mahalanobis_silhouette",
            "oof_ordinal_rsa",
        ):
            row[f"discovery_{prefix}"] = discovery
        row["confirmation_isotropic_snr_db"] = confirmation
        row["confirmation_fisher_trace_frozen"] = confirmation
        row["confirmation_mahalanobis_silhouette"] = confirmation
        row["confirmation_ordinal_rsa"] = confirmation
        rows.append(row)
    selected = select_discovery_winners(pd.DataFrame(rows))
    assert set(selected["selected_layer"].tolist()) == {1}
    assert set(selected["confirmation_value"].tolist()) == {0.1}
