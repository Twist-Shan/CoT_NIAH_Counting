from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from realistic_niah_v4.counter_channel import (
    LayerStateDataset,
    benchmark_classifiers,
    benchmark_noise_models,
    count_axis,
    count_subspace,
    leave_group_out_noise,
    projected_patch,
    remove_count_component,
    subspace_overlap,
)


def synthetic_dataset() -> LayerStateDataset:
    rng = np.random.default_rng(7)
    rows = []
    counts = []
    seeds = []
    direction = np.asarray([1.0, -0.5, 0.25, 0.0, 0.0, 0.0])
    for seed in range(8):
        seed_offset = rng.normal(scale=0.03, size=6)
        for count in range(1, 6):
            rows.append(count * direction + seed_offset + rng.normal(scale=0.01, size=6))
            counts.append(count)
            seeds.append(seed)
    metadata = pd.DataFrame(
        {
            "sample_id": [f"s{seed}_c{count}" for seed, count in zip(seeds, counts)],
            "seed": seeds,
            "count": counts,
        }
    )
    return LayerStateDataset(
        model_label="mock",
        role="answer_query",
        layer=3,
        states=np.asarray(rows, dtype=np.float32),
        count=np.asarray(counts),
        seed=np.asarray(seeds),
        metadata=metadata,
        source=None,  # type: ignore[arg-type]
    )


def test_count_subspace_and_grouped_classification() -> None:
    dataset = synthetic_dataset()
    dataset.validate()
    basis, explained = count_subspace(dataset.states, dataset.count, rank=1)
    assert explained[0] > 0.99
    assert subspace_overlap(basis, basis) == pytest.approx(1.0)
    axis = count_axis(dataset.states, dataset.count)
    assert abs(float(axis @ basis[:, 0])) > 0.99
    metrics = benchmark_classifiers(
        dataset,
        algorithms=["logistic_l2", "knn_k3_euclidean", "nearest_centroid"],
        folds=4,
        pca_components=3,
    )
    by_algorithm = metrics.set_index("algorithm")["accuracy"]
    assert by_algorithm["logistic_l2"] > 0.9
    assert by_algorithm["nearest_centroid"] > 0.9
    assert by_algorithm["knn_k3_euclidean"] > 0.75


def test_project_remove_and_noise_decomposition() -> None:
    dataset = synthetic_dataset()
    basis, _ = count_subspace(dataset.states, dataset.count, rank=1)
    receiver = dataset.states[0]
    donor = dataset.states[-1]
    patched, complement = projected_patch(receiver, donor, basis)
    delta = donor - receiver
    np.testing.assert_allclose(
        (patched - receiver) + (complement - receiver), delta, atol=1e-6
    )
    removed = remove_count_component(donor, dataset.states.mean(axis=0), basis)
    assert np.linalg.norm(basis.T @ (removed - dataset.states.mean(axis=0))) < 1e-5
    noise = leave_group_out_noise(dataset, rank=1)
    np.testing.assert_allclose(
        noise["noise_total"], noise["noise_parallel"] + noise["noise_orthogonal"], rtol=1e-5
    )
    noise["role_factor"] = "answer_query"
    model_results = benchmark_noise_models(
        noise,
        target="noise_total",
        group_column="seed",
        numeric=["count"],
        categorical=["role_factor"],
        folds=2,
        n_jobs=1,
    )
    assert set(model_results["model"]) == {
        "elastic_net",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    }
