from __future__ import annotations

import json

import numpy as np
import pandas as pd

from realistic_niah_v5.cross_mode_geometry import ModeDataset
from realistic_niah_v5.dual_endpoint_geometry import (
    RUNNING_NATIVE_PRIMARY_SITES,
    RUNNING_NON_THINKING_SITES,
    _final_count_analysis,
    determine_group_eligibility,
    load_native_thinking_final_count,
    load_non_thinking_final_count,
    relabel_seed_panel,
    select_discovery_winners,
)


def test_running_index_primary_sites_are_fixed_single_token_endpoints():
    assert RUNNING_NON_THINKING_SITES == ("span_end",)
    assert RUNNING_NATIVE_PRIMARY_SITES == ("item_end",)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_final_count_loaders_preserve_all_layers_and_gold_labels(tmp_path):
    non_root = tmp_path / "non"
    native_root = tmp_path / "native"
    non_root.mkdir()
    native_root.mkdir()
    non_rows = []
    native_rows = []
    for seed, count in ((1234, 1), (1235, 2)):
        non_shard = non_root / f"non_{seed}_{count}.npz"
        np.savez(
            non_shard,
            layer_indices=np.asarray([0, 1]),
            query_states=np.asarray(
                [[seed, count, 0], [seed, count, 1]], dtype=np.float16
            ),
        )
        non_rows.append(
            {
                "design_variant": "v4.4",
                "split": "discovery",
                "seed": seed,
                "count": count,
                "stimulus_id": f"V4_4_N{count}_seed{seed}",
                "model_label": "toy",
                "position": "prompt_final_total_query",
                "shard_path": non_shard.name,
            }
        )
        native_shard = native_root / f"native_{seed}_{count}.npz"
        np.savez(
            native_shard,
            layer_indices=np.asarray([0, 1]),
            site_states=np.asarray(
                [[[seed, count, 2], [seed, count, 3]]], dtype=np.float16
            ),
        )
        manifest = native_root / f"manifest_{seed}_{count}.json"
        manifest.write_text(
            json.dumps({"site_rows": [{"site_kind": "answer_query_v3"}]}),
            encoding="utf-8",
        )
        native_rows.append(
            {
                "split": "discovery",
                "seed": seed,
                "gold_count": count,
                "stimulus_id": f"V4_4_N{count}_seed{seed}",
                "model_label": "toy",
                "manifest_path": manifest.name,
                "states_path": native_shard.name,
                "request_id": f"toy/{seed}/{count}",
                "exact_count": True,
            }
        )
    non_index = non_root / "capture_index.jsonl"
    native_index = native_root / "capture_index.jsonl"
    _write_jsonl(non_index, non_rows)
    _write_jsonl(native_index, native_rows)

    non = load_non_thinking_final_count(non_index)
    native = load_native_thinking_final_count(native_index)
    assert sorted(non.states_by_layer) == [0, 1]
    assert sorted(native.states_by_layer) == [0, 1]
    assert non.metadata["occurrence"].tolist() == [1, 2]
    assert native.metadata["occurrence"].tolist() == [1, 2]
    np.testing.assert_array_equal(non.states_by_layer[1][:, 2], [1, 1])
    np.testing.assert_array_equal(native.states_by_layer[1][:, 2], [3, 3])


def test_final_count_analysis_uses_registered_trajectory_split(tmp_path):
    non_root = tmp_path / "non_full"
    native_root = tmp_path / "native_full"
    non_root.mkdir()
    native_root.mkdir()
    non_rows = []
    native_rows = []
    for seed in range(1, 6):
        split = "discovery" if seed <= 3 else "confirmation"
        for count in range(1, 11):
            vector = np.asarray(
                [count, count**2 / 10, seed / 10, 1.0], dtype=np.float32
            )
            non_shard = non_root / f"non_{seed}_{count}.npz"
            np.savez(
                non_shard,
                layer_indices=np.asarray([0]),
                query_states=vector[None, :],
            )
            stimulus_id = f"V4_4_N{count}_seed{seed}"
            non_rows.append(
                {
                    "design_variant": "v4.4",
                    "split": split,
                    "seed": seed,
                    "count": count,
                    "stimulus_id": stimulus_id,
                    "model_label": "toy",
                    "position": "prompt_final_total_query",
                    "shard_path": non_shard.name,
                }
            )
            native_shard = native_root / f"native_{seed}_{count}.npz"
            np.savez(
                native_shard,
                layer_indices=np.asarray([0]),
                site_states=(vector + 0.01)[None, None, :],
            )
            manifest = native_root / f"manifest_{seed}_{count}.json"
            manifest.write_text(
                json.dumps({"site_rows": [{"site_kind": "answer_query_v3"}]}),
                encoding="utf-8",
            )
            native_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "gold_count": count,
                    "stimulus_id": stimulus_id,
                    "model_label": "toy",
                    "manifest_path": manifest.name,
                    "states_path": native_shard.name,
                    "request_id": f"toy/{seed}/{count}",
                    "exact_count": True,
                }
            )
    non_index = non_root / "capture_index.jsonl"
    native_index = native_root / "capture_index.jsonl"
    _write_jsonl(non_index, non_rows)
    _write_jsonl(native_index, native_rows)

    _candidates, selected, audit = _final_count_analysis(
        non_index,
        native_index,
        pca_dim=2,
        cv_folds=3,
        random_state=0,
    )
    assert set(selected["evaluation_split_role"]) == {"registered_confirmation"}
    assert set(selected["confirmation_rows"].astype(int)) == {20}
    assert audit["registered_trajectory_counts"] == {
        "discovery": 30,
        "confirmation": 20,
    }
    assert audit["registered_seed_panel"] == {
        "discovery": [1, 2, 3],
        "confirmation": [4, 5],
    }


def test_seed_panel_relabeling_is_disjoint_and_preserves_state_alignment():
    metadata = pd.DataFrame(
        {
            "split": ["source"] * 4,
            "seed": [1, 1, 2, 2],
            "occurrence": [1, 2, 1, 2],
        }
    )
    dataset = ModeDataset(
        mode="non_thinking",
        model_label="toy",
        metadata=metadata,
        states_by_layer={0: np.arange(8).reshape(4, 2)},
    )
    relabeled = relabel_seed_panel(
        dataset, discovery_seeds=[1], confirmation_seeds=[2]
    )
    assert relabeled.metadata["split"].tolist() == ["confirmation"] * 2 + [
        "discovery"
    ] * 2
    np.testing.assert_array_equal(
        relabeled.states_by_layer[0][:, 0], np.asarray([4, 6, 0, 2])
    )


def test_broad_trace_groups_pool_sparse_surface_forms():
    rows = []
    for split, seeds in (("discovery", [1, 2, 3]), ("confirmation", [4, 5])):
        for seed in seeds:
            for label in range(1, 6):
                rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "occurrence": label,
                        "marker_kind": "indexed" if seed % 2 else "ordinal",
                    }
                )
    eligibility = determine_group_eligibility(
        pd.DataFrame(rows), "explicit_ordinal"
    )
    assert eligibility.status == "evaluable"
    assert eligibility.labels == (1, 2, 3, 4, 5)


def test_discovery_winners_are_selected_independently_by_mode():
    rows = []
    for mode, best_layer in (("non_thinking", 1), ("native_thinking", 3)):
        for layer in (1, 2, 3):
            score = 0.9 if layer == best_layer else 0.4
            rows.append(
                {
                    "endpoint": "running_index",
                    "model_label": "toy",
                    "mode": mode,
                    "analysis_group": "all_traces",
                    "selector": "site_search",
                    "token_site": f"site_{layer}",
                    "layer": layer,
                    "discovery_selection_score": score,
                    "discovery_oof_ncc_balanced_accuracy": score,
                    "discovery_oof_logistic_balanced_accuracy": score,
                }
            )
    selected = select_discovery_winners(pd.DataFrame(rows)).set_index("mode")
    assert int(selected.loc["non_thinking", "layer"]) == 1
    assert int(selected.loc["native_thinking", "layer"]) == 3
