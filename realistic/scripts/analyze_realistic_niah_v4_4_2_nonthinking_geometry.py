from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import analyze_realistic_niah_v4_4_2_counter_geometry as core


CONDITIONS = ("cue_present", "cue_absent")
SITE = "answer_query"


def capture_lookup(root: Path) -> dict[tuple[str, str, str], Path]:
    result: dict[tuple[str, str, str], Path] = {}
    pattern = "conditions/*/*/nonthinking/**/capture/capture_manifest.json"
    for path in sorted(root.glob(pattern)):
        manifest = core.read_json(path)
        key = (
            str(manifest["model_label"]),
            str(manifest["stimulus_id"]),
            str(manifest["prompt_variant"]),
        )
        if key in result:
            raise RuntimeError(f"Duplicate capture key: {key}")
        result[key] = path.parent
    return result


def answer_vector(
    capture_dir: Path,
    manifest: dict[str, Any],
    layer: int,
) -> np.ndarray:
    hidden = torch.load(
        capture_dir / f"layer_{layer:02d}_hidden.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    roles = [str(value) for value in manifest["query_roles"]]
    answer_indices = [
        index for index, role in enumerate(roles) if role == "answer_query"
    ]
    if not answer_indices:
        raise RuntimeError(f"Missing answer query role in {capture_dir}")
    return hidden[answer_indices[-1]].numpy()


def analyze(
    root: Path,
    output: Path,
    *,
    permutations: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    lookup = capture_lookup(root)
    models = sorted({key[0] for key in lookup})
    datasets: dict[str, Any] = {}
    statistics: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}

    for model in models:
        identities = sorted(
            {
                key[1]
                for key in lookup
                if key[0] == model
                and (model, key[1], "cue_present") in lookup
                and (model, key[1], "cue_absent") in lookup
            }
        )
        records: list[dict[str, Any]] = []
        for stimulus_id in identities:
            present_dir = lookup[(model, stimulus_id, "cue_present")]
            absent_dir = lookup[(model, stimulus_id, "cue_absent")]
            present_manifest = core.read_json(
                present_dir / "capture_manifest.json"
            )
            absent_manifest = core.read_json(absent_dir / "capture_manifest.json")
            present_generation = core.read_json(
                present_dir.parent / "generation.json"
            )
            absent_generation = core.read_json(
                absent_dir.parent / "generation.json"
            )
            common_layers = sorted(
                {int(row["layer"]) for row in present_manifest["layers"]}
                & {int(row["layer"]) for row in absent_manifest["layers"]}
            )
            records.append(
                {
                    "seed": int(present_manifest["seed"]),
                    "count": int(present_manifest["gold_count"]),
                    "present_dir": present_dir,
                    "absent_dir": absent_dir,
                    "present_manifest": present_manifest,
                    "absent_manifest": absent_manifest,
                    "present_correct": bool(present_generation["exact_count"]),
                    "absent_correct": bool(absent_generation["exact_count"]),
                    "layers": common_layers,
                }
            )
        records.sort(key=lambda row: (row["count"], row["seed"]))
        if not records:
            continue
        layers = sorted(set.intersection(*(set(row["layers"]) for row in records)))
        counts = np.asarray([row["count"] for row in records], dtype=np.int64)
        groups = np.asarray([row["seed"] for row in records], dtype=np.int64)
        coverage[model] = {
            "pairs": len(records),
            "layers": len(layers),
            "counts": {
                str(count): int(np.sum(counts == count))
                for count in range(1, 11)
            },
            "seeds": len(np.unique(groups)),
        }

        for layer in layers:
            present = np.stack(
                [
                    answer_vector(
                        row["present_dir"], row["present_manifest"], layer
                    )
                    for row in records
                ],
                axis=0,
            )
            absent = np.stack(
                [
                    answer_vector(
                        row["absent_dir"], row["absent_manifest"], layer
                    )
                    for row in records
                ],
                axis=0,
            )
            raw_present, raw_absent, raw_evr = core.pca_scores(
                present, absent, cue_centered=False
            )
            centered_present, centered_absent, centered_evr = core.pca_scores(
                present, absent, cue_centered=True
            )
            present_predictions = core.grouped_ridge_predictions(
                raw_present, counts, groups
            )
            absent_predictions = core.grouped_ridge_predictions(
                raw_absent, counts, groups
            )
            rng = np.random.default_rng(982001 + layer * 17)
            decode = core.clustered_decode_comparison(
                counts.astype(np.float64),
                groups,
                present_predictions,
                absent_predictions,
                rng,
                bootstrap_repetitions=bootstrap_repetitions,
            )
            interaction = core.count_cue_interaction(
                present,
                absent,
                counts,
                groups,
                rng,
                permutations=permutations,
            )
            strength = core.counter_strength_comparison(
                present,
                absent,
                counts,
                groups,
                rng,
                permutations=permutations,
                bootstrap_repetitions=bootstrap_repetitions,
            )
            present_centroids = core.centroid_matrix(present, counts)
            absent_centroids = core.centroid_matrix(absent, counts)
            statistics.append(
                {
                    "model": model,
                    "site": SITE,
                    "layer": layer,
                    "centroid_cka": core.linear_cka(
                        present_centroids, absent_centroids
                    ),
                    "path_step_cosine_present": core.path_step_cosine(
                        present_centroids
                    ),
                    "path_step_cosine_absent": core.path_step_cosine(
                        absent_centroids
                    ),
                    **decode,
                    **interaction,
                    **strength,
                }
            )
            rows: list[list[Any]] = []
            for index, record in enumerate(records):
                rows.append(
                    [
                        record["seed"],
                        record["count"],
                        int(record["present_correct"]),
                        int(record["absent_correct"]),
                        *core.rounded(raw_present[index]),
                        *core.rounded(raw_absent[index]),
                        *core.rounded(centered_present[index]),
                        *core.rounded(centered_absent[index]),
                    ]
                )
            datasets[f"{model}|{SITE}|{layer}"] = {
                "model": model,
                "site": SITE,
                "layer": layer,
                "evr_raw": core.rounded(np.asarray(raw_evr), 6),
                "evr_cue_centered": core.rounded(
                    np.asarray(centered_evr), 6
                ),
                "rows": rows,
            }
            print(
                f"[nonthinking-geometry] {model} layer {layer}/{layers[-1]}",
                flush=True,
            )

    statistic_frame = pd.DataFrame(statistics)
    for model in sorted(statistic_frame["model"].unique()):
        mask = statistic_frame["model"] == model
        statistic_frame.loc[mask, "interaction_q"] = core.bh_adjust(
            statistic_frame.loc[mask, "interaction_p"].tolist()
        )
        statistic_frame.loc[mask, "decode_q"] = core.bh_adjust(
            statistic_frame.loc[mask, "decode_p"].tolist()
        )
        statistic_frame.loc[mask, "count_eta_q"] = core.bh_adjust(
            statistic_frame.loc[mask, "count_eta_p"].tolist()
        )
    statistic_frame = statistic_frame.sort_values(["model", "site", "layer"])
    output.mkdir(parents=True, exist_ok=True)
    statistic_path = output / "nonthinking_geometry_layer_statistics.csv"
    statistic_frame.to_csv(statistic_path, index=False)
    statistic_lookup = {
        f"{row.model}|{row.site}|{int(row.layer)}": {
            column: (
                int(value)
                if column == "layer"
                else float(value)
                if isinstance(value, (np.floating, float))
                else value
            )
            for column, value in row._asdict().items()
        }
        for row in statistic_frame.itertuples(index=False)
    }
    payload = {
        "schema_version": "realistic_niah_v4_4_2_nonthinking_geometry_v1",
        "mode": "nonthinking",
        "direction": "cue_absent - cue_present",
        "sites": [SITE],
        "conditions": list(CONDITIONS),
        "landmarks": core.LANDMARKS,
        "coverage": coverage,
        "inference": {
            "count_by_cue_interaction": (
                "paired hidden delta one-way pseudo-F; count labels permuted "
                "within seed; BH adjusted across layers per model/site"
            ),
            "counter_strength": (
                "full-space count eta-squared; paired cue-label permutation p, "
                "seed-cluster bootstrap CI, and BH adjustment across layers"
            ),
            "permutations": permutations,
            "bootstrap_repetitions": bootstrap_repetitions,
            "pca": (
                "six-component pooled shared PCA fit jointly to cue-present and "
                "cue-absent answer-query states at each model/layer"
            ),
        },
        "datasets": datasets,
        "joint": {},
        "statistics": statistic_lookup,
    }
    data_path = output / "nonthinking_geometry_data.json"
    core.write_json(data_path, payload)
    summary = {
        "coverage": coverage,
        "dataset_count": len(datasets),
        "statistics_rows": len(statistic_frame),
        "significant_interaction_fdr_005": int(
            (statistic_frame["interaction_q"] < 0.05).sum()
        ),
        "significant_counter_strength_fdr_005": int(
            (statistic_frame["count_eta_q"] < 0.05).sum()
        ),
        "data_path": str(data_path),
        "statistics_path": str(statistic_path),
    }
    core.write_json(output / "nonthinking_geometry_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = analyze(
        arguments.run_root,
        arguments.output_dir,
        permutations=arguments.permutations,
        bootstrap_repetitions=arguments.bootstrap_repetitions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
