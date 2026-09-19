from __future__ import annotations

import numpy as np
import pandas as pd

from realistic_niah_v5.trace_stratified_geometry import (
    _select_rows,
    confirmation_metrics,
    determine_stratum_eligibility,
    grouped_discovery_cv_metrics,
)


def test_each_selector_independently_selects_its_own_best_layer():
    metrics = pd.DataFrame(
        [
            {
                "model_label": "test-model",
                "marker_kind": "indexed",
                "site_kind": site,
                "layer": layer,
                "discovery_selection_score": score,
                "discovery_oof_ncc_balanced_accuracy": score,
                "discovery_oof_logistic_balanced_accuracy": score,
            }
            for site, layer, score in (
                ("item_end", 4, 0.91),
                ("item_end", 9, 0.40),
                ("city_end", 4, 0.50),
                ("city_end", 9, 0.97),
                ("marker_end", 2, 0.99),
                ("marker_end", 4, 0.30),
            )
        ]
    )

    selected = _select_rows(metrics).set_index("selector")

    assert selected.loc["fixed_item_end", "site_kind"] == "item_end"
    assert selected.loc["fixed_item_end", "layer"] == 4
    assert selected.loc["post_marker_site_search", "site_kind"] == "city_end"
    assert selected.loc["post_marker_site_search", "layer"] == 9
    assert selected.loc["all_site_search", "site_kind"] == "marker_end"
    assert selected.loc["all_site_search", "layer"] == 2


def _metadata(marker_kind: str, discovery_seeds, confirmation_seeds, labels):
    return pd.DataFrame(
        [
            {
                "split": split,
                "seed": seed,
                "occurrence": label,
                "marker_kind": marker_kind,
            }
            for split, seeds in (
                ("discovery", discovery_seeds),
                ("confirmation", confirmation_seeds),
            )
            for seed in seeds
            for label in labels
        ]
    )


def test_trace_stratum_eligibility_separates_claim_and_exploratory_support():
    labels = range(1, 6)
    claim = determine_stratum_eligibility(
        _metadata("indexed", range(1, 5), range(5, 8), labels),
        "indexed",
    )
    assert claim.status == "claim_grade"
    assert claim.labels == (1, 2, 3, 4, 5)

    exploratory = determine_stratum_eligibility(
        _metadata("audit_sentence", range(1, 5), [5], labels),
        "audit_sentence",
    )
    assert exploratory.status == "exploratory_only"

    insufficient = determine_stratum_eligibility(
        _metadata("ordinal", [1], [2], [1, 2]),
        "ordinal",
    )
    assert insufficient.status == "not_evaluable"


def test_grouped_discovery_selection_and_confirmation_are_seed_held_out():
    metadata = _metadata("completion_recap", range(1, 5), range(5, 7), range(1, 6))
    rng = np.random.default_rng(4)
    states = []
    for row in metadata.itertuples(index=False):
        vector = np.zeros(12, dtype=np.float32)
        vector[int(row.occurrence) - 1] = 5.0
        vector[5] = float(row.occurrence)
        vector[6] = 0.05 * float(row.seed)
        vector += rng.normal(scale=0.03, size=vector.shape)
        states.append(vector)
    states = np.stack(states)

    discovery = grouped_discovery_cv_metrics(
        states, metadata, range(1, 6), pca_dim=5
    )
    heldout = confirmation_metrics(states, metadata, range(1, 6), pca_dim=5)
    assert discovery["discovery_fold_count"] == 4
    assert discovery["discovery_oof_logistic_balanced_accuracy"] > 0.75
    assert discovery["discovery_oof_ncc_balanced_accuracy"] > 0.9
    assert heldout["confirmation_seed_count"] == 2
    assert heldout["confirmation_logistic_balanced_accuracy"] >= 0.9
    assert heldout["confirmation_ncc_balanced_accuracy"] > 0.9
    assert np.isfinite(heldout["confirmation_class_balanced_snr_db"])
