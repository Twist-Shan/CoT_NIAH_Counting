from __future__ import annotations

from scripts.analyze_realistic_niah_v5_overwrite_mechanism import (
    clean_prediction_cells,
    paired_donor_cells,
)


def _row(*, seed: int, donor: int, next_prediction: int) -> dict[str, object]:
    return {
        "seed": seed,
        "event_variant": "original",
        "read_layer": 15,
        "donor": donor,
        "clean_target_prediction": 6,
        "clean_target_soft": 5.6 + seed / 1000,
        "current_donor_exact": True,
        "next_event_count_exact": next_prediction == 6,
        "next_prediction": next_prediction,
        "current_soft": float(donor),
        "next_soft": float(next_prediction) + donor / 100,
    }


def test_paired_donor_cells_distinguishes_invariance_from_recurrent_separation() -> None:
    rows = [
        _row(seed=1, donor=4, next_prediction=6),
        _row(seed=1, donor=6, next_prediction=6),
        _row(seed=2, donor=4, next_prediction=5),
        _row(seed=2, donor=6, next_prediction=7),
    ]

    cell = paired_donor_cells(rows, variants=("original",))[0]

    assert cell["n_seed_pairs"] == 2
    assert cell["donor_invariant_count"] == 1
    assert cell["donor_invariant_accuracy"] == 0.5
    assert cell["recurrent_separation_exact_count"] == 1
    assert cell["recurrent_separation_accuracy"] == 0.5
    assert cell["current_donor_exact_count"] == 4


def test_clean_prediction_cells_deduplicates_donor_copies() -> None:
    rows = [
        _row(seed=1, donor=4, next_prediction=6),
        _row(seed=1, donor=6, next_prediction=6),
        _row(seed=2, donor=4, next_prediction=5),
        _row(seed=2, donor=6, next_prediction=7),
    ]

    cell = clean_prediction_cells(rows)[0]

    assert cell["n_seeds"] == 2
    assert cell["clean_prediction_counts"] == {"6": 2}
