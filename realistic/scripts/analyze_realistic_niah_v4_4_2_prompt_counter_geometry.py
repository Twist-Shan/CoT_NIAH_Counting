from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analyze_realistic_niah_v4_4_2_counter_geometry as core


SITE = "prompt_counter"
CONDITIONS = ("cue_present", "cue_absent")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def capture_lookup(
    root: Path,
) -> dict[tuple[str, int, str], tuple[Path, dict[str, Any]]]:
    result: dict[tuple[str, int, str], tuple[Path, dict[str, Any]]] = {}
    for index_path in sorted(root.glob("*/capture_index.jsonl")):
        model_root = index_path.parent
        for row in read_jsonl(index_path):
            key = (
                str(row["model_label"]),
                int(row["seed"]),
                str(row["prompt_variant"]),
            )
            if key in result:
                raise RuntimeError(f"Duplicate prompt-counter capture: {key}")
            result[key] = (model_root / str(row["shard_path"]), row)
    return result


def load_endpoint_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        layer_indices = np.asarray(payload["layer_indices"], dtype=np.int64)
        span_end = np.asarray(payload["span_end"], dtype=np.float32)
    if span_end.ndim != 3 or span_end.shape[1] != 10:
        raise RuntimeError(
            f"Expected [layers,10,hidden] endpoint tensor in {path}; "
            f"got {span_end.shape}"
        )
    if len(layer_indices) != span_end.shape[0]:
        raise RuntimeError(f"Layer index mismatch in {path}")
    return layer_indices, span_end


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
        seeds = sorted(
            {
                key[1]
                for key in lookup
                if key[0] == model
                and (model, key[1], "cue_present") in lookup
                and (model, key[1], "cue_absent") in lookup
            }
        )
        records: list[dict[str, Any]] = []
        for seed in seeds:
            present_path, present_row = lookup[(model, seed, "cue_present")]
            absent_path, absent_row = lookup[(model, seed, "cue_absent")]
            present_layers, present_states = load_endpoint_shard(present_path)
            absent_layers, absent_states = load_endpoint_shard(absent_path)
            if not np.array_equal(present_layers, absent_layers):
                raise RuntimeError(f"Cue-pair layer mismatch for {model}, seed {seed}")
            if present_states.shape != absent_states.shape:
                raise RuntimeError(f"Cue-pair tensor mismatch for {model}, seed {seed}")
            if str(present_row["stimulus_id"]) != str(absent_row["stimulus_id"]):
                raise RuntimeError(f"Cue-pair stimulus mismatch for {model}, seed {seed}")
            records.append(
                {
                    "seed": seed,
                    "layers": present_layers,
                    "present": present_states,
                    "absent": absent_states,
                }
            )
        if not records:
            continue
        reference_layers = records[0]["layers"]
        if any(not np.array_equal(row["layers"], reference_layers) for row in records):
            raise RuntimeError(f"Layer grid drift across prompt-counter shards for {model}")
        layers = [int(value) for value in reference_layers]
        counts = np.tile(np.arange(1, 11, dtype=np.int64), len(records))
        groups = np.repeat(
            np.asarray([row["seed"] for row in records], dtype=np.int64), 10
        )
        coverage[model] = {
            "paired_prompts": len(records),
            "paired_endpoint_states": int(len(counts)),
            "layers": len(layers),
            "counts": {
                str(count): int(np.sum(counts == count)) for count in range(1, 11)
            },
            "seeds": len(np.unique(groups)),
            "prompt_gold_count": 10,
        }

        for layer_index, layer in enumerate(layers):
            present = np.concatenate(
                [row["present"][layer_index] for row in records], axis=0
            )
            absent = np.concatenate(
                [row["absent"][layer_index] for row in records], axis=0
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
            rng = np.random.default_rng(993117 + layer * 23)
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
            for index in range(len(counts)):
                rows.append(
                    [
                        int(groups[index]),
                        int(counts[index]),
                        None,
                        None,
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
                f"[prompt-counter-geometry] {model} layer {layer}/{layers[-1]}",
                flush=True,
            )

    statistic_frame = pd.DataFrame(statistics)
    if statistic_frame.empty:
        raise RuntimeError("No paired prompt-counter captures were found")
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
    statistic_path = output / "prompt_counter_geometry_layer_statistics.csv"
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
        "schema_version": "realistic_niah_v4_4_2_prompt_counter_geometry_v1",
        "mode": "nonthinking",
        "site": SITE,
        "direction": "cue_absent - cue_present",
        "conditions": list(CONDITIONS),
        "landmarks": core.LANDMARKS,
        "coverage": coverage,
        "counter_definition": (
            "For each seed, use the single N=10 prompt and take the post-block "
            "residual at needle endpoints 1 through 10. Thus count k is the "
            "state after reading the kth occurrence, not an answer-query state "
            "from a separate N=k prompt."
        ),
        "inference": {
            "independent_cluster": "seed",
            "count_by_cue_interaction": (
                "paired hidden delta one-way pseudo-F; occurrence labels "
                "permuted within seed; BH adjusted across layers per model"
            ),
            "counter_strength": (
                "full-space occurrence eta-squared; paired cue-label permutation "
                "p, seed-cluster bootstrap CI, and BH adjustment across layers"
            ),
            "permutations": permutations,
            "bootstrap_repetitions": bootstrap_repetitions,
            "pca": (
                "six-component shared PCA fit jointly to cue-present and "
                "cue-absent endpoint states at each model/layer"
            ),
        },
        "datasets": datasets,
        "statistics": statistic_lookup,
    }
    data_path = output / "prompt_counter_geometry_data.json"
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
    core.write_json(output / "prompt_counter_geometry_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = analyze(
        arguments.capture_root,
        arguments.output_dir,
        permutations=arguments.permutations,
        bootstrap_repetitions=arguments.bootstrap_repetitions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
