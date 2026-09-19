from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.analyze_realistic_niah_v4_4_5_answer_geometry import (
    load_answer_states,
    select_display_layers,
)
from scripts.analyze_realistic_niah_v4_4_5_span_restoration import (
    audit_patch_hook_applications,
    layerwise_transition_boundaries,
)
from scripts.merge_realistic_niah_v4_4_5_8gpu_shards import expected_keys
from scripts.analyze_realistic_niah_v4_4_5_retrieval_geometry import (
    geometry_metrics,
)
from scripts.analyze_realistic_niah_v4_4_5_retrieval_subspace import (
    paired_effects,
)


def test_retrieval_geometry_uses_frozen_seed_split() -> None:
    rng = np.random.default_rng(7)
    rows = []
    direction = rng.normal(size=16)
    for seed in range(6):
        seed_noise = rng.normal(scale=0.05, size=16)
        for count in (1, 2, 3):
            rows.append(
                {
                    "model_label": "toy",
                    "layer": 2,
                    "seed": seed,
                    "gold_count": count,
                    "strict_correct": True,
                    "vector": count * direction + seed_noise,
                }
            )
    metrics, fitted = geometry_metrics(
        pd.DataFrame(rows),
        {0, 1, 2, 3},
        {4, 5},
        bootstrap_draws=4,
    )
    assert len(metrics) == 1
    assert metrics.iloc[0]["confirmation_rows"] == 6
    assert metrics.iloc[0]["ridge_mad"] < 0.2
    assert ("toy", 2) in fitted


def test_retrieval_subspace_effect_is_aligned_minus_orthogonal() -> None:
    rows = []
    values = {
        "clean_aligned_block": (4.0, 1, False),
        "clean_orthogonal_block": (5.0, 0, True),
        "restored_aligned_block": (3.0, 2, False),
        "restored_orthogonal_block": (5.0, 0, True),
    }
    for condition, (expected, strict_error, correct) in values.items():
        rows.append(
            {
                "model_label": "toy",
                "seed": 1,
                "gold_count": 5,
                "source_patch_layer": 1,
                "retrieval_layer": 2,
                "condition": condition,
                "expected_count": expected,
                "strict_absolute_error": strict_error,
                "strict_correct": correct,
            }
        )
    effect = paired_effects(pd.DataFrame(rows)).iloc[0]
    assert effect["natural_expected_error_specificity"] == 1.0
    assert effect["restoration_mediation_expected_error"] == 2.0
    assert effect["restoration_mediation_accuracy_damage"] == 1.0


def test_answer_geometry_loader_reads_only_clean_baseline(tmp_path: Path) -> None:
    model_root = tmp_path / "toy"
    state_root = model_root / "states"
    state_root.mkdir(parents=True)
    torch.save(
        {"answer_states": {"0": torch.arange(4), "1": torch.arange(4) + 1}},
        state_root / "clean.pt",
    )
    rows = [
        {
            "condition": "clean",
            "patch_layer": -1,
            "state_path": "states/clean.pt",
            "seed": 3,
            "gold_count": 2,
            "strict_correct": True,
        },
        {
            "condition": "needle_corrupt",
            "patch_layer": -1,
            "state_path": "states/not_read.pt",
            "seed": 3,
            "gold_count": 2,
            "strict_correct": False,
        },
    ]
    (model_root / "detail.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    loaded = load_answer_states(tmp_path, ["toy"])
    assert len(loaded) == 2
    assert loaded["layer"].tolist() == [0, 1]
    assert loaded.iloc[1]["vector"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_answer_display_layer_rule_uses_frozen_three_pc_space() -> None:
    rng = np.random.default_rng(11)
    fitted = {}
    metric_rows = []
    for layer, noise in ((0, 20.0), (1, 0.02)):
        y_train = np.repeat(np.arange(1, 4), 5)
        y_test = np.repeat(np.arange(1, 4), 2)
        x_train = np.stack(
            [np.array([count, count**2, count**3, 0.0]) for count in y_train]
        ) + rng.normal(scale=noise, size=(15, 4))
        x_test = np.stack(
            [np.array([count, count**2, count**3, 0.0]) for count in y_test]
        ) + rng.normal(scale=noise, size=(6, 4))
        from sklearn.decomposition import PCA

        fitted[("toy", layer)] = {
            "pca": PCA(n_components=3).fit(x_train),
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        metric_rows.append(
            {"model_label": "toy", "layer": layer, "rank3_all": 0.9}
        )
    selected, candidates = select_display_layers(
        fitted, pd.DataFrame(metric_rows)
    )
    assert selected == {"toy": 1}
    assert candidates["selected_for_display"].sum() == 1


def test_layerwise_boundaries_are_fit_on_discovery_and_read_on_confirmation() -> None:
    discovery_curve = [2.0, 2.0, 2.0, 1.8, 0.9, 0.8, 0.7, 0.08, 0.05, 0.02]
    specificity_rows = []
    endpoint_rows = []
    for population, offset in (("discovery", 0.0), ("confirmation", 0.2)):
        for layer, value in enumerate(discovery_curve):
            specificity_rows.append(
                {
                    "model_label": "toy",
                    "population": population,
                    "patch_layer": layer,
                    "mean_expected_absolute_error_reduction_specificity": value
                    + offset,
                }
            )
            endpoint_rows.append(
                {
                    "model_label": "toy",
                    "population": population,
                    "patch_layer": layer,
                    "mean_expected_absolute_error_reduction_full_minus_endpoint": 0.5,
                }
            )
    boundaries = layerwise_transition_boundaries(
        pd.DataFrame(specificity_rows), pd.DataFrame(endpoint_rows)
    ).set_index("boundary")
    assert boundaries.loc["half_early_plateau", "frozen_boundary_layer"] == 4
    assert boundaries.loc["near_zero_0.10_count", "frozen_boundary_layer"] == 7
    assert (
        boundaries.loc[
            "half_early_plateau", "confirmation_specificity_at_boundary"
        ]
        == 1.1
    )


def test_dense_shard_expected_key_counts() -> None:
    qwen = expected_keys(list(range(1234, 1249)), list(range(36)))
    gemma = expected_keys(list(range(1234, 1239)), list(range(42)))
    assert len(qwen) == 16650
    assert len(gemma) == 6450
    assert (1234, 1, "clean", -1) in qwen
    assert (1248, 10, "restore_ordinary_full", 35) in qwen
    assert (1238, 10, "restore_needle_full", 41) in gemma


def test_patch_hook_audit_tracks_prefill_reuse_path() -> None:
    rows = pd.DataFrame(
        [
            {
                "patch_layer": -1,
                "patch_hook_applications": 0,
                "strict_generation_reused_prefill": True,
            },
            {
                "patch_layer": 4,
                "patch_hook_applications": 2,
                "strict_generation_reused_prefill": True,
            },
            {
                "patch_layer": 5,
                "patch_hook_applications": 3,
                "strict_generation_reused_prefill": False,
            },
        ]
    )
    audit = audit_patch_hook_applications(rows)
    assert audit["status"] == "PASS"
    assert audit["patched_rows"] == 2
    assert audit["reused_prefill_rows_expected_two"] == 1
    assert audit["legacy_rows_expected_three"] == 1

    rows.loc[1, "patch_hook_applications"] = 3
    failed = audit_patch_hook_applications(rows)
    assert failed["status"] == "FAIL"
    assert failed["mismatched_rows"] == 1
