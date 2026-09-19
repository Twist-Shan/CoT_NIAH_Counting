from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from realistic_niah_v5.cross_mode_geometry import (
    compare_position_geometry,
    load_native_thinking_capture,
    load_non_thinking_capture,
)


def _write_non_thinking(
    tmp_path,
    *,
    hidden_size=32,
    discovery_seeds=range(1, 5),
    confirmation_seeds=range(5, 8),
    counts=(10,),
):
    base = tmp_path / "nonthinking"
    shards = base / "shards"
    shards.mkdir(parents=True)
    index_rows = []
    rng = np.random.default_rng(1)
    for split, seeds in (
        ("discovery", discovery_seeds),
        ("confirmation", confirmation_seeds),
    ):
        for seed in seeds:
            for count in counts:
                values = np.stack(
                    [
                        np.asarray(
                            [
                                occurrence * 0.8 + rng.normal(scale=0.1)
                                for _ in range(hidden_size)
                            ],
                            dtype=np.float16,
                        )
                        for occurrence in range(1, count + 1)
                    ]
                )[None, :, :]
                shard = shards / f"{split}_{seed}_n{count}.npz"
                np.savez(shard, layer_indices=np.asarray([0]), span_end=values)
                index_rows.append(
                    {
                        "design_variant": "v4.4",
                        "count": count,
                        "split": split,
                        "seed": seed,
                        "stimulus_id": f"stimulus-{split}-{seed}-n{count}",
                        "model_label": "Toy",
                        "shard_path": str(shard.relative_to(base)),
                    }
                )
    index = base / "capture_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in index_rows))
    return index


def _write_native(
    tmp_path,
    *,
    hidden_size=32,
    discovery_seeds=range(1, 5),
    confirmation_seeds=range(5, 8),
    occurrences_by_seed=None,
    marker_kind_by_seed=None,
    counts=(10,),
):
    base = tmp_path / "native"
    shards = base / "shards"
    shards.mkdir(parents=True)
    index_rows = []
    rng = np.random.default_rng(2)
    occurrences_by_seed = occurrences_by_seed or {}
    marker_kind_by_seed = marker_kind_by_seed or {}
    for split, seeds in (
        ("discovery", discovery_seeds),
        ("confirmation", confirmation_seeds),
    ):
        for seed in seeds:
            for count in counts:
                occurrences = tuple(
                    occurrences_by_seed.get(
                        (seed, count),
                        occurrences_by_seed.get(seed, range(1, count + 1)),
                    )
                )
                marker_kind = marker_kind_by_seed.get(
                    (seed, count), marker_kind_by_seed.get(seed, "indexed")
                )
                selected_site_kinds = (
                    ("pre_marker", "marker_end", "item_end")
                    if marker_kind in {"indexed", "ordinal", "inline_count"}
                    else ("item_end",)
                )
                row_dir = shards / f"{split}_{seed}_n{count}"
                row_dir.mkdir()
                site_rows = []
                state_rows = []
                for occurrence in occurrences:
                    for site_kind in selected_site_kinds:
                        site_rows.append(
                            {"site_kind": site_kind, "occurrence": occurrence}
                        )
                        offset = 0.25 if site_kind == "marker_end" else 0.0
                        state_rows.append(
                            np.asarray(
                                [
                                    occurrence * 1.2
                                    + offset
                                    + rng.normal(scale=0.08)
                                    for _ in range(hidden_size)
                                ],
                                dtype=np.float16,
                            )
                        )
                values = np.stack(state_rows)[:, None, :]
                states = row_dir / "states.npz"
                np.savez(states, layer_indices=np.asarray([0]), site_states=values)
                manifest = {
                    "parser": {"marker_kind": marker_kind},
                    "site_rows": site_rows,
                }
                manifest_path = row_dir / "capture_manifest.json"
                manifest_path.write_text(json.dumps(manifest))
                index_rows.append(
                    {
                        "gold_count": count,
                        "split": split,
                        "seed": seed,
                        "stimulus_id": f"stimulus-{split}-{seed}-n{count}",
                        "model_label": "Toy",
                        "trace_one_to_one": len(occurrences) == count,
                        "trace_category": (
                            "one_to_one"
                            if len(occurrences) == count
                            else "partial_unique"
                        ),
                        "marker_kind": marker_kind,
                        "exact_count": True,
                        "states_path": str(states.relative_to(base)),
                        "manifest_path": str(manifest_path.relative_to(base)),
                    }
                )
    index = base / "capture_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in index_rows))
    return index


def test_cross_mode_geometry_uses_matched_seed_panel_and_reports_covariance(tmp_path):
    paths = compare_position_geometry(
        _write_non_thinking(tmp_path),
        _write_native(tmp_path),
        tmp_path / "out",
        # With three confirmation seeds, each delete-one replicate has only
        # two observations per class.  PCA16 therefore exercises the singular
        # within-covariance path used by the real Gemma E3 analysis.
        pca_dim=16,
        layers=[0],
    )
    per_class = pd.read_csv(paths["per_class"])
    assert set(per_class["mode"]) == {"non_thinking", "native_thinking"}
    assert {"min_bhattacharyya", "class_nc1_ratio", "logistic_recall"}.issubset(
        set(per_class["metric"])
    )
    global_metrics = pd.read_csv(paths["global"])
    assert global_metrics["pillai_trace_regularized"].notna().all()
    assert global_metrics["nc1_trace_sigmaw_sigmab_pinv_over_c"].notna().all()
    assert global_metrics["class_balanced_snr"].notna().all()
    assert global_metrics["class_balanced_snr_db"].notna().all()
    assert (global_metrics["class_balanced_snr"] > 0).all()
    assert np.allclose(
        global_metrics["class_balanced_snr_db"],
        10.0 * np.log10(global_metrics["class_balanced_snr"]),
    )
    assert (global_metrics["covariance_ridge"] > 0).all()
    assert (global_metrics["covariance_ridge_attempts"] >= 1).all()
    audit = json.loads(paths["audit"].read_text())
    assert audit["preprocessing_fit_split"] == "discovery only"
    assert audit["analysis_design"] == "fixed_registered_seed_panel_observed_positions"
    assert "mean_k" in audit["snr_definition"]
    assert audit["registered_seed_panel"]["confirmation"] == [5, 6, 7]
    cross = pd.read_csv(paths["cross_mode_trend"])
    assert cross["native_minus_nonthinking"].notna().any()


def test_cross_mode_geometry_keeps_partial_traces_on_full_seed_panel(tmp_path):
    confirmation_seeds = range(5, 9)
    paths = compare_position_geometry(
        _write_non_thinking(tmp_path, confirmation_seeds=confirmation_seeds),
        _write_native(
            tmp_path,
            confirmation_seeds=confirmation_seeds,
            occurrences_by_seed={2: range(1, 9), 6: range(1, 9)},
        ),
        tmp_path / "out_ragged",
        pca_dim=8,
        layers=[0],
    )
    audit = json.loads(paths["audit"].read_text())
    assert audit["registered_seed_panel"] == {
        "discovery": [1, 2, 3, 4],
        "confirmation": [5, 6, 7, 8],
    }
    assert audit["position_support"]["native_thinking"]["discovery"]["10"] == 3
    assert audit["position_support"]["native_thinking"]["confirmation"]["10"] == 3
    assert audit["position_support"]["non_thinking"]["confirmation"]["10"] == 4

    global_metrics = pd.read_csv(paths["global"])
    native = global_metrics.loc[global_metrics["mode"].eq("native_thinking")].iloc[0]
    nonthinking = global_metrics.loc[global_metrics["mode"].eq("non_thinking")].iloc[0]
    assert int(native["n_confirmation"]) == 38
    assert int(native["min_confirmation_per_class"]) == 3
    assert int(nonthinking["n_confirmation"]) == 40

    per_class = pd.read_csv(paths["per_class"])
    late_native = per_class.loc[
        per_class["mode"].eq("native_thinking")
        & per_class["occurrence"].eq(10)
    ]
    assert set(late_native["n_confirmation_class"]) == {3}


def test_cross_mode_geometry_rejects_same_seed_with_different_stimulus(tmp_path):
    nonthinking = _write_non_thinking(tmp_path)
    native = _write_native(tmp_path)
    rows = [json.loads(line) for line in native.read_text().splitlines()]
    rows[0]["stimulus_id"] = "different-stimulus"
    native.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="cross-mode stimulus mismatches"):
        compare_position_geometry(
            nonthinking,
            native,
            tmp_path / "out_mismatch",
            pca_dim=8,
            layers=[0],
        )


def test_trace_aware_native_site_policy_uses_marker_kind(tmp_path):
    index = _write_native(
        tmp_path,
        discovery_seeds=range(1, 3),
        confirmation_seeds=range(3, 5),
        marker_kind_by_seed={
            1: "indexed",
            2: "bullet",
            3: "ordinal",
            4: "audit_sentence",
        },
    )
    dataset = load_native_thinking_capture(
        index,
        site_policy="trace_aware_count_boundary",
        cohort="parser_hit",
    )
    selected = (
        dataset.metadata.groupby("seed")["selected_site_kind"].first().to_dict()
    )
    assert selected == {
        1: "marker_end",
        2: "item_end",
        3: "marker_end",
        4: "item_end",
    }
    assert set(dataset.metadata["native_site_policy"]) == {
        "trace_aware_count_boundary"
    }


def test_all_count_loaders_keep_every_trajectory_and_ragged_event(tmp_path):
    non_index = _write_non_thinking(
        tmp_path,
        discovery_seeds=[1],
        confirmation_seeds=[2],
        counts=(1, 2, 3),
    )
    native_index = _write_native(
        tmp_path,
        discovery_seeds=[1],
        confirmation_seeds=[2],
        counts=(1, 2, 3),
        occurrences_by_seed={(2, 3): (1, 2)},
    )
    non = load_non_thinking_capture(non_index)
    native = load_native_thinking_capture(native_index, cohort="parser_hit")
    assert len(non.metadata) == 12
    assert len(native.metadata) == 11
    assert len(non.metadata[["split", "seed", "gold_count"]].drop_duplicates()) == 6
    assert len(native.metadata[["split", "seed", "gold_count"]].drop_duplicates()) == 6
    assert set(non.metadata["gold_count"]) == {1, 2, 3}
    assert set(native.metadata["gold_count"]) == {1, 2, 3}
    ragged = native.metadata.loc[
        native.metadata["split"].eq("confirmation")
        & native.metadata["gold_count"].eq(3)
    ]
    assert ragged["occurrence"].tolist() == [1, 2]

    one_to_one = load_native_thinking_capture(
        native_index, cohort="one_to_one"
    )
    one_to_one_trajectories = one_to_one.metadata[
        ["split", "seed", "gold_count"]
    ].drop_duplicates()
    assert len(one_to_one_trajectories) == 5
    assert set(
        one_to_one_trajectories.loc[
            one_to_one_trajectories["split"].eq("discovery"), "gold_count"
        ]
    ) == {1, 2, 3}


def test_trace_aware_pre_label_policy_avoids_explicit_k_token(tmp_path):
    index = _write_native(
        tmp_path,
        discovery_seeds=[1, 2],
        confirmation_seeds=[3, 4],
        marker_kind_by_seed={
            1: "indexed",
            2: "bullet",
            3: "inline_count",
            4: "evidence_sequence",
        },
    )
    dataset = load_native_thinking_capture(
        index,
        site_policy="trace_aware_pre_label",
        cohort="parser_hit",
    )
    selected = (
        dataset.metadata.groupby("seed")["selected_site_kind"].first().to_dict()
    )
    assert selected == {
        1: "pre_marker",
        2: "item_end",
        3: "pre_marker",
        4: "item_end",
    }
