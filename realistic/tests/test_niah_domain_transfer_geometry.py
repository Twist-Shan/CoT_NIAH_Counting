from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from realistic_niah_v5.domain_transfer_geometry import (
    COUNTS,
    DOMAINS,
    EVALUATION_SEEDS,
    SELECTION_SEEDS,
    DomainEndpointDataset,
    city_anchored_pca3,
    evaluate_frozen_layer,
    load_transfer_answer_endpoints,
    select_layer,
)


def _synthetic_panel() -> DomainEndpointDataset:
    rows = []
    layers = {0: [], 1: []}
    for domain_index, domain in enumerate(DOMAINS):
        for seed in EVALUATION_SEEDS:
            for count in COUNTS:
                rows.append(
                    {
                        "entity_domain": domain,
                        "seed": seed,
                        "gold_count": count,
                        "stimulus_id": f"{domain}-{seed}-{count}",
                        "source_stimulus_id": f"V4_4_T10000_N{count}_seed{seed}",
                    }
                )
                rng = np.random.default_rng(10_000 * domain_index + 100 * seed + count)
                noise = rng.normal(0, 0.04, 12)
                layers[0].append(noise + domain_index)
                signal = noise.copy()
                signal[count - 1] += 5.0
                signal[10] += 0.2 * domain_index
                layers[1].append(signal)
    dataset = DomainEndpointDataset(
        mode="non_thinking",
        model_label="synthetic",
        metadata=pd.DataFrame(rows),
        states_by_layer={key: np.asarray(value, dtype=np.float16) for key, value in layers.items()},
    )
    dataset.validate(require_complete=True)
    return dataset


def _synthetic_city_training() -> DomainEndpointDataset:
    rows = []
    layers = {0: [], 1: []}
    for seed in SELECTION_SEEDS + EVALUATION_SEEDS:
        for count in COUNTS:
            rows.append(
                {
                    "entity_domain": "city",
                    "seed": seed,
                    "gold_count": count,
                    "stimulus_id": f"city-{seed}-{count}",
                    "source_stimulus_id": f"V4_4_T10000_N{count}_seed{seed}",
                }
            )
            rng = np.random.default_rng(100 * seed + count)
            noise = rng.normal(0, 0.04, 12)
            layers[0].append(noise)
            signal = noise.copy()
            signal[count - 1] += 5.0
            layers[1].append(signal)
    dataset = DomainEndpointDataset(
        mode="non_thinking",
        model_label="synthetic",
        metadata=pd.DataFrame(rows),
        states_by_layer={
            key: np.asarray(value, dtype=np.float16) for key, value in layers.items()
        },
    )
    dataset.validate()
    return dataset


def test_city_discovery_selection_and_confirmation_transfer_are_disjoint() -> None:
    panel = _synthetic_panel()
    city_training = _synthetic_city_training()
    layer, rows = select_layer(city_training, n_components=4)
    assert layer == 1
    assert len(rows) == 2
    result = evaluate_frozen_layer(
        panel,
        training_dataset=city_training,
        layer=layer,
        dimensions=(1, 2, 4),
    )
    assert result["selection_seeds"] == list(SELECTION_SEEDS)
    assert result["evaluation_seeds"] == list(EVALUATION_SEEDS)
    assert set(result["cross_domain_count"]) == {"flower", "animal"}
    assert result["training_rows"] == 200
    assert result["evaluation_rows"] == 300
    assert result["overall_count"]["logistic_balanced_accuracy"] > 0.9


def test_city_anchored_visualization_contains_all_300_rows() -> None:
    result = city_anchored_pca3(
        _synthetic_panel(), training_dataset=_synthetic_city_training(), layer=1
    )
    assert result["fit_rows"] == 200
    assert len(result["points"]) == 300
    assert {point["analysis_split"] for point in result["points"]} == {
        "confirmation"
    }


def test_transfer_loader_uses_exact_answer_site(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    shard = root / "shards" / "row"
    shard.mkdir(parents=True)
    np.savez(
        shard / "states.npz",
        layer_indices=np.asarray([0, 1]),
        site_states=np.arange(24, dtype=np.float16).reshape(3, 2, 4),
    )
    manifest = {
        "model_label": "Qwen3-8B",
        "stimulus_id": "flower-1254-1",
        "site_rows": [
            {"site_kind": "item_end"},
            {"site_kind": "item_end"},
            {"site_kind": "answer_query_v3"},
        ],
    }
    (shard / "capture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    row = {
        "entity_domain": "flower",
        "seed": 1254,
        "gold_count": 1,
        "split": "confirmation",
        "stimulus_id": "flower-1254-1",
        "source_stimulus_id": "V4_4_T10000_N1_seed1254",
        "states_path": "shards/row/states.npz",
        "manifest_path": "shards/row/capture_manifest.json",
        "running_site_count": 2,
        "answer_site_count": 1,
    }
    (root / "capture_index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_transfer_answer_endpoints(
        root / "capture_index.jsonl", mode="native_thinking"
    )
    assert loaded.states_by_layer[0].tolist() == [[16.0, 17.0, 18.0, 19.0]]
    assert loaded.metadata.iloc[0]["running_site_count"] == 2
