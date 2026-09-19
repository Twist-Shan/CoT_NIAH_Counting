from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


SITES = ("trace_last", "trace_mean", "answer_query")
JOINT_SITES = ("trace_last", "answer_query")
CONDITIONS = ("cue_present", "cue_absent")
LANDMARKS = {
    "Qwen3-8B": {"display": 29, "probe": 35},
    "Gemma4-E4B": {"display": 37, "probe": 39},
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def capture_lookup(root: Path) -> dict[tuple[str, str, str], Path]:
    result: dict[tuple[str, str, str], Path] = {}
    pattern = "conditions/*/*/native_thinking/**/capture/capture_manifest.json"
    for path in sorted(root.glob(pattern)):
        manifest = read_json(path)
        key = (
            str(manifest["model_label"]),
            str(manifest["stimulus_id"]),
            str(manifest["prompt_variant"]),
        )
        if key in result:
            raise RuntimeError(f"Duplicate capture key: {key}")
        result[key] = path.parent
    return result


def site_vectors(
    capture_dir: Path,
    manifest: dict[str, Any],
    layer: int,
) -> dict[str, np.ndarray]:
    hidden = torch.load(
        capture_dir / f"layer_{layer:02d}_hidden.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    roles = [str(value) for value in manifest["query_roles"]]
    trace_indices = [index for index, role in enumerate(roles) if role == "trace"]
    answer_indices = [
        index for index, role in enumerate(roles) if role == "answer_query"
    ]
    if not trace_indices or not answer_indices:
        raise RuntimeError(f"Missing trace/answer query roles in {capture_dir}")
    return {
        "trace_last": hidden[trace_indices[-1]].numpy(),
        "trace_mean": hidden[trace_indices].mean(dim=0).numpy(),
        "answer_query": hidden[answer_indices[-1]].numpy(),
    }


def pca_scores(
    present: np.ndarray,
    absent: np.ndarray,
    *,
    cue_centered: bool,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if cue_centered:
        present_fit = present - present.mean(axis=0, keepdims=True)
        absent_fit = absent - absent.mean(axis=0, keepdims=True)
    else:
        present_fit = present
        absent_fit = absent
    pooled = np.concatenate([present_fit, absent_fit], axis=0)
    pca = PCA(n_components=6, svd_solver="randomized", random_state=0)
    projected = pca.fit_transform(pooled)
    size = present.shape[0]
    return (
        projected[:size],
        projected[size:],
        [float(value) for value in pca.explained_variance_ratio_],
    )


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    gram_left = left @ left.T
    gram_right = right @ right.T
    denominator = math.sqrt(
        float(np.sum(gram_left**2)) * float(np.sum(gram_right**2))
    )
    return float(np.sum(gram_left * gram_right) / max(denominator, 1e-12))


def centroid_matrix(
    states: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [states[counts == count].mean(axis=0) for count in range(1, 11)],
        axis=0,
    )


def path_step_cosine(centroids: np.ndarray) -> float:
    steps = np.diff(centroids, axis=0)
    if len(steps) < 2:
        return float("nan")
    left = steps[:-1]
    right = steps[1:]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    values = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)
    return float(values.mean())


def grouped_ridge_predictions(
    states: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    predictions = np.zeros(len(targets), dtype=np.float64)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        x_train = states[train].astype(np.float64, copy=False)
        x_test = states[test].astype(np.float64, copy=False)
        y_train = targets[train].astype(np.float64, copy=False)
        mean_x = x_train.mean(axis=0, keepdims=True)
        centered_train = x_train - mean_x
        centered_test = x_test - mean_x
        scale_sq = float(np.mean(np.sum(centered_train**2, axis=1)))
        scale_sq = max(scale_sq, 1e-12)
        kernel_train = centered_train @ centered_train.T / scale_sq
        mean_y = float(y_train.mean())
        coefficients = np.linalg.solve(
            kernel_train + alpha * np.eye(kernel_train.shape[0]),
            y_train - mean_y,
        )
        kernel_test = centered_test @ centered_train.T / scale_sq
        predictions[test] = mean_y + kernel_test @ coefficients
    return predictions


def r2_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    denominator = float(np.sum((targets - targets.mean()) ** 2))
    return 1.0 - float(np.sum((targets - predictions) ** 2)) / max(
        denominator, 1e-12
    )


def clustered_decode_comparison(
    targets: np.ndarray,
    groups: np.ndarray,
    present_predictions: np.ndarray,
    absent_predictions: np.ndarray,
    rng: np.random.Generator,
    *,
    bootstrap_repetitions: int,
) -> dict[str, float]:
    present_r2 = r2_score(targets, present_predictions)
    absent_r2 = r2_score(targets, absent_predictions)
    unique_groups = np.unique(groups)
    bootstrap_deltas: list[float] = []
    for _ in range(bootstrap_repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        bootstrap_deltas.append(
            r2_score(targets[indices], absent_predictions[indices])
            - r2_score(targets[indices], present_predictions[indices])
        )
    error_delta = (
        (targets - absent_predictions) ** 2
        - (targets - present_predictions) ** 2
    )
    group_effects = np.asarray(
        [error_delta[groups == group].mean() for group in unique_groups],
        dtype=np.float64,
    )
    observed = float(group_effects.mean())
    permutations = 1 << len(unique_groups)
    extreme = 0
    for mask in range(permutations):
        signs = np.asarray(
            [
                1.0 if mask & (1 << index) else -1.0
                for index in range(len(unique_groups))
            ],
            dtype=np.float64,
        )
        if abs(float(np.mean(signs * group_effects))) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "r2_present": present_r2,
        "r2_absent": absent_r2,
        "r2_delta": absent_r2 - present_r2,
        "r2_delta_ci_low": float(np.percentile(bootstrap_deltas, 2.5)),
        "r2_delta_ci_high": float(np.percentile(bootstrap_deltas, 97.5)),
        "decode_p": float(extreme / permutations),
    }


def between_count_statistic(
    gram: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    size = len(labels)
    counts = np.unique(labels)
    global_term = float(gram.sum()) / size
    between = 0.0
    for count in counts:
        indices = np.flatnonzero(labels == count)
        between += float(gram[np.ix_(indices, indices)].sum()) / len(indices)
    between -= global_term
    total = float(np.trace(gram)) - global_term
    within = max(total - between, 1e-12)
    statistic = (between / (len(counts) - 1)) / (within / (size - len(counts)))
    eta_sq = between / max(total, 1e-12)
    return float(statistic), float(eta_sq)


def count_cue_interaction(
    present: np.ndarray,
    absent: np.ndarray,
    counts: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    *,
    permutations: int,
) -> dict[str, float]:
    delta = (absent - present).astype(np.float64, copy=False)
    gram = delta @ delta.T
    observed, eta_sq = between_count_statistic(gram, counts)
    extreme = 0
    for _ in range(permutations):
        permuted = counts.copy()
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            permuted[indices] = rng.permutation(permuted[indices])
        value, _ = between_count_statistic(gram, permuted)
        if value >= observed - 1e-15:
            extreme += 1
    return {
        "interaction_f": observed,
        "interaction_eta_sq": eta_sq,
        "interaction_p": float((extreme + 1) / (permutations + 1)),
    }


def counter_strength_comparison(
    present: np.ndarray,
    absent: np.ndarray,
    counts: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    *,
    permutations: int,
    bootstrap_repetitions: int,
) -> dict[str, float]:
    present64 = present.astype(np.float64, copy=False)
    absent64 = absent.astype(np.float64, copy=False)
    present_gram = present64 @ present64.T
    absent_gram = absent64 @ absent64.T
    _, present_eta = between_count_statistic(present_gram, counts)
    _, absent_eta = between_count_statistic(absent_gram, counts)
    observed = absent_eta - present_eta

    midpoint = (present64 + absent64) * 0.5
    delta = absent64 - present64
    midpoint_gram = midpoint @ midpoint.T
    midpoint_delta = midpoint @ delta.T
    delta_gram = delta @ delta.T
    size = len(counts)
    same_count = (counts[:, None] == counts[None, :]).astype(np.float64)
    count_sizes = np.asarray(
        [np.sum(counts == count) for count in counts], dtype=np.float64
    )
    between_coefficient = same_count / count_sizes[:, None] - 1.0 / size
    total_coefficient = np.eye(size, dtype=np.float64) - 1.0 / size

    def signed_eta_components(
        coefficient: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        base = float(np.sum(coefficient * midpoint_gram))
        linear = np.sum(coefficient * midpoint_delta, axis=0)
        quadratic = 0.25 * coefficient * delta_gram
        return base, linear, quadratic

    between_base, between_linear, between_quadratic = signed_eta_components(
        between_coefficient
    )
    total_base, total_linear, total_quadratic = signed_eta_components(
        total_coefficient
    )
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(permutations, size),
    )
    between_signed = signs @ between_linear
    between_quad = np.einsum(
        "bi,ij,bj->b",
        signs,
        between_quadratic,
        signs,
        optimize=True,
    )
    total_signed = signs @ total_linear
    total_quad = np.einsum(
        "bi,ij,bj->b",
        signs,
        total_quadratic,
        signs,
        optimize=True,
    )
    eta_absent_permuted = (
        between_base + between_signed + between_quad
    ) / np.maximum(total_base + total_signed + total_quad, 1e-12)
    eta_present_permuted = (
        between_base - between_signed + between_quad
    ) / np.maximum(total_base - total_signed + total_quad, 1e-12)
    permuted_deltas = eta_absent_permuted - eta_present_permuted
    extreme = int(
        np.sum(np.abs(permuted_deltas) >= abs(observed) - 1e-15)
    )

    unique_groups = np.unique(groups)
    group_draws = rng.multinomial(
        len(unique_groups),
        np.full(len(unique_groups), 1.0 / len(unique_groups)),
        size=bootstrap_repetitions,
    ).astype(np.float64)
    group_index = {group: index for index, group in enumerate(unique_groups)}
    weights = np.stack(
        [
            group_draws[:, group_index[group]]
            for group in groups
        ],
        axis=1,
    )

    def weighted_eta(gram: np.ndarray) -> np.ndarray:
        global_quadratic = np.einsum(
            "bi,ij,bj->b", weights, gram, weights, optimize=True
        )
        total = (
            weights @ np.diag(gram)
            - global_quadratic / size
        )
        between = np.zeros(bootstrap_repetitions, dtype=np.float64)
        for count in range(1, 11):
            indices = np.flatnonzero(counts == count)
            count_weights = weights[:, indices]
            between += np.einsum(
                "bi,ij,bj->b",
                count_weights,
                gram[np.ix_(indices, indices)],
                count_weights,
                optimize=True,
            ) / len(indices)
        between -= global_quadratic / size
        return between / np.maximum(total, 1e-12)

    bootstrap_deltas = weighted_eta(absent_gram) - weighted_eta(present_gram)
    return {
        "count_eta_present": present_eta,
        "count_eta_absent": absent_eta,
        "count_eta_delta": observed,
        "count_eta_delta_ci_low": float(
            np.percentile(bootstrap_deltas, 2.5)
        ),
        "count_eta_delta_ci_high": float(
            np.percentile(bootstrap_deltas, 97.5)
        ),
        "count_eta_p": float((extreme + 1) / (permutations + 1)),
    }


def bh_adjust(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 1.0
    size = len(array)
    for reverse_rank in range(size - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(array[index]) * size / rank)
        adjusted[index] = running
    return [float(min(1.0, value)) for value in adjusted]


def rounded(values: np.ndarray, digits: int = 5) -> list[float]:
    return [round(float(value), digits) for value in values]


def joint_centroid_projection(
    states: dict[str, dict[str, np.ndarray]],
    counts: np.ndarray,
) -> dict[str, Any]:
    centered_groups: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    for site in JOINT_SITES:
        for condition in CONDITIONS:
            values = states[site][condition]
            centered_groups.append(values - values.mean(axis=0, keepdims=True))
            keys.append((site, condition))
    pooled = np.concatenate(centered_groups, axis=0)
    pca = PCA(n_components=6, svd_solver="randomized", random_state=0)
    projected = pca.fit_transform(pooled)
    cursor = 0
    rows: list[list[Any]] = []
    for (site, condition), values in zip(keys, centered_groups):
        size = len(values)
        group_scores = projected[cursor : cursor + size]
        cursor += size
        for count in range(1, 11):
            center = group_scores[counts == count].mean(axis=0)
            rows.append([site, condition, count, *rounded(center)])
    return {
        "evr": rounded(pca.explained_variance_ratio_, 6),
        "rows": rows,
        "trace_answer_cka_present": linear_cka(
            centroid_matrix(states["trace_last"]["cue_present"], counts),
            centroid_matrix(states["answer_query"]["cue_present"], counts),
        ),
        "trace_answer_cka_absent": linear_cka(
            centroid_matrix(states["trace_last"]["cue_absent"], counts),
            centroid_matrix(states["answer_query"]["cue_absent"], counts),
        ),
    }


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
    joint: dict[str, Any] = {}
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
            present_manifest = read_json(present_dir / "capture_manifest.json")
            absent_manifest = read_json(absent_dir / "capture_manifest.json")
            present_generation = read_json(present_dir.parent / "generation.json")
            absent_generation = read_json(absent_dir.parent / "generation.json")
            common_layers = sorted(
                {int(row["layer"]) for row in present_manifest["layers"]}
                & {int(row["layer"]) for row in absent_manifest["layers"]}
            )
            records.append(
                {
                    "stimulus_id": stimulus_id,
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
            "counts": {str(count): int(np.sum(counts == count)) for count in range(1, 11)},
            "seeds": len(np.unique(groups)),
        }
        for layer in layers:
            state_lists: dict[str, dict[str, list[np.ndarray]]] = {
                site: {condition: [] for condition in CONDITIONS} for site in SITES
            }
            for record in records:
                present_vectors = site_vectors(
                    record["present_dir"], record["present_manifest"], layer
                )
                absent_vectors = site_vectors(
                    record["absent_dir"], record["absent_manifest"], layer
                )
                for site in SITES:
                    state_lists[site]["cue_present"].append(present_vectors[site])
                    state_lists[site]["cue_absent"].append(absent_vectors[site])
            states = {
                site: {
                    condition: np.stack(state_lists[site][condition], axis=0)
                    for condition in CONDITIONS
                }
                for site in SITES
            }
            for site_index, site in enumerate(SITES):
                present = states[site]["cue_present"]
                absent = states[site]["cue_absent"]
                raw_present, raw_absent, raw_evr = pca_scores(
                    present, absent, cue_centered=False
                )
                centered_present, centered_absent, centered_evr = pca_scores(
                    present, absent, cue_centered=True
                )
                # Decode in the same pooled six-PC basis used by the raw 3D
                # explorer.  This keeps the scalar diagnostic stable and makes
                # it explicitly a low-dimensional counter-manifold test; the
                # separate interaction test below remains full-dimensional.
                present_predictions = grouped_ridge_predictions(
                    raw_present, counts, groups
                )
                absent_predictions = grouped_ridge_predictions(
                    raw_absent, counts, groups
                )
                rng = np.random.default_rng(880301 + layer * 17 + site_index * 1009)
                decode = clustered_decode_comparison(
                    counts.astype(np.float64),
                    groups,
                    present_predictions,
                    absent_predictions,
                    rng,
                    bootstrap_repetitions=bootstrap_repetitions,
                )
                interaction = count_cue_interaction(
                    present,
                    absent,
                    counts,
                    groups,
                    rng,
                    permutations=permutations,
                )
                strength = counter_strength_comparison(
                    present,
                    absent,
                    counts,
                    groups,
                    rng,
                    permutations=permutations,
                    bootstrap_repetitions=bootstrap_repetitions,
                )
                present_centroids = centroid_matrix(present, counts)
                absent_centroids = centroid_matrix(absent, counts)
                statistic = {
                    "model": model,
                    "site": site,
                    "layer": layer,
                    "centroid_cka": linear_cka(
                        present_centroids, absent_centroids
                    ),
                    "path_step_cosine_present": path_step_cosine(
                        present_centroids
                    ),
                    "path_step_cosine_absent": path_step_cosine(absent_centroids),
                    **decode,
                    **interaction,
                    **strength,
                }
                statistics.append(statistic)
                rows: list[list[Any]] = []
                for index, record in enumerate(records):
                    rows.append(
                        [
                            record["seed"],
                            record["count"],
                            int(record["present_correct"]),
                            int(record["absent_correct"]),
                            *rounded(raw_present[index]),
                            *rounded(raw_absent[index]),
                            *rounded(centered_present[index]),
                            *rounded(centered_absent[index]),
                        ]
                    )
                datasets[f"{model}|{site}|{layer}"] = {
                    "model": model,
                    "site": site,
                    "layer": layer,
                    "evr_raw": rounded(np.asarray(raw_evr), 6),
                    "evr_cue_centered": rounded(np.asarray(centered_evr), 6),
                    "rows": rows,
                }
            joint[f"{model}|{layer}"] = {
                "model": model,
                "layer": layer,
                **joint_centroid_projection(states, counts),
            }
            print(f"[counter-geometry] {model} layer {layer}/{layers[-1]}", flush=True)

    statistic_frame = pd.DataFrame(statistics)
    for model in sorted(statistic_frame["model"].unique()):
        for site in SITES:
            mask = (statistic_frame["model"] == model) & (
                statistic_frame["site"] == site
            )
            statistic_frame.loc[mask, "interaction_q"] = bh_adjust(
                statistic_frame.loc[mask, "interaction_p"].tolist()
            )
            statistic_frame.loc[mask, "decode_q"] = bh_adjust(
                statistic_frame.loc[mask, "decode_p"].tolist()
            )
            statistic_frame.loc[mask, "count_eta_q"] = bh_adjust(
                statistic_frame.loc[mask, "count_eta_p"].tolist()
            )
    statistic_frame = statistic_frame.sort_values(["model", "site", "layer"])
    statistic_path = output / "counter_geometry_layer_statistics.csv"
    output.mkdir(parents=True, exist_ok=True)
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
        "schema_version": "realistic_niah_v4_4_2_counter_geometry_v1",
        "direction": "cue_absent - cue_present",
        "sites": list(SITES),
        "conditions": list(CONDITIONS),
        "landmarks": LANDMARKS,
        "coverage": coverage,
        "inference": {
            "count_decoder": (
                "fixed-alpha=1 ridge in the pooled shared six-PC basis; "
                "leave-one-seed-out CV"
            ),
            "decode_difference": "seed-cluster bootstrap CI and exact seed-cluster sign-flip p",
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
                "cue-absent states at each model/site/layer"
            ),
        },
        "datasets": datasets,
        "joint": joint,
        "statistics": statistic_lookup,
    }
    data_path = output / "counter_geometry_data.json"
    write_json(data_path, payload)
    summary = {
        "coverage": coverage,
        "dataset_count": len(datasets),
        "joint_dataset_count": len(joint),
        "statistics_rows": len(statistic_frame),
        "significant_interaction_fdr_005": int(
            (statistic_frame["interaction_q"] < 0.05).sum()
        ),
        "significant_decode_difference_fdr_005": int(
            (statistic_frame["decode_q"] < 0.05).sum()
        ),
        "significant_counter_strength_fdr_005": int(
            (statistic_frame["count_eta_q"] < 0.05).sum()
        ),
        "data_path": str(data_path),
        "statistics_path": str(statistic_path),
    }
    write_json(output / "counter_geometry_summary.json", summary)
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
