from __future__ import annotations

"""Estimate adjacent-layer 3D count maps and their seed stability.

All bases and maps are fitted on discovery rows. Grouped cross-validation keeps
entire seeds out of the fit. Prompt-running confirmation rows, when available,
are evaluated without refitting. Raw PCA-coordinate rotation angles are marked
as gauge dependent; gauge-invariant principal angles and prediction metrics are
the primary outputs.
"""

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.layerwise_rotation import (  # noqa: E402
    align_resampled_map_to_reference,
    consecutive_full_operator_metrics,
    evaluate_layer_map,
    fit_layer_map,
    matrix_r2,
    polar_factors,
    principal_angles_degrees,
    proper_rotation_geodesic_degrees,
    relative_frobenius_error,
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def paired_rows(
    source: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    for required in ("sample_id", "states", "count", "seed", "split"):
        if required not in source or required not in target:
            raise KeyError(f"paired datasets require {required}")
    source_index = {str(value): index for index, value in enumerate(source["sample_id"])}
    target_index = {str(value): index for index, value in enumerate(target["sample_id"])}
    if len(source_index) != len(source["sample_id"]):
        raise ValueError("duplicate source sample_id")
    if len(target_index) != len(target["sample_id"]):
        raise ValueError("duplicate target sample_id")
    shared = sorted(set(source_index) & set(target_index))
    if not shared:
        raise ValueError("source and target have no shared sample_id")
    left = np.asarray([source_index[value] for value in shared], dtype=np.int64)
    right = np.asarray([target_index[value] for value in shared], dtype=np.int64)
    for field in ("count", "seed", "split"):
        source_values = np.asarray(source[field][left]).astype(str)
        target_values = np.asarray(target[field][right]).astype(str)
        if not np.array_equal(source_values, target_values):
            raise ValueError(f"paired rows disagree on {field}")
    return left, right


def class_centroids(
    states: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    missing = [value for value in classes if not np.any(labels == value)]
    if missing:
        raise ValueError(f"missing evaluation classes: {missing}")
    return np.stack([states[labels == value].mean(axis=0) for value in classes])


def coordinate_sums(
    fit: Any, source_states: np.ndarray, target_states: np.ndarray
) -> tuple[float, float, float, float]:
    source_coordinates = fit.source.coordinates(source_states)
    target_coordinates = fit.target.coordinates(target_states)
    predicted = source_coordinates @ fit.matrix
    residual = float(np.square(target_coordinates - predicted).sum())
    total = float(
        np.square(target_coordinates - target_coordinates.mean(axis=0, keepdims=True)).sum()
    )
    direct_residual = float(np.square(target_coordinates - source_coordinates).sum())
    return residual, total, direct_residual, float(len(target_coordinates))


def grouped_cv(
    source_states: np.ndarray,
    target_states: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    rank: int,
    folds: int,
    ridge_relative_scale: float,
) -> dict[str, float]:
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(int(folds), len(unique_groups)))
    sample_residual = sample_total = sample_direct = 0.0
    centroid_residual = centroid_total = 0.0
    fold_r2: list[float] = []
    for train, test in splitter.split(source_states, labels, groups):
        fit = fit_layer_map(
            source_states[train],
            target_states[train],
            labels[train],
            rank=rank,
            ridge_relative_scale=ridge_relative_scale,
        )
        residual, total, direct, _ = coordinate_sums(
            fit, source_states[test], target_states[test]
        )
        sample_residual += residual
        sample_total += total
        sample_direct += direct
        fold_r2.append(1.0 - residual / max(total, 1e-24))
        source_centroids = class_centroids(
            source_states[test], labels[test], fit.source.classes
        )
        target_centroids = class_centroids(
            target_states[test], labels[test], fit.target.classes
        )
        residual, total, _, _ = coordinate_sums(
            fit, source_centroids, target_centroids
        )
        centroid_residual += residual
        centroid_total += total
    return {
        "cv_sample_r2": 1.0 - sample_residual / max(sample_total, 1e-24),
        "cv_sample_normalized_rmse": float(
            np.sqrt(sample_residual / max(sample_total, 1e-24))
        ),
        "cv_direct_identity_r2": 1.0 - sample_direct / max(sample_total, 1e-24),
        "cv_centroid_r2": 1.0 - centroid_residual / max(centroid_total, 1e-24),
        "cv_centroid_normalized_rmse": float(
            np.sqrt(centroid_residual / max(centroid_total, 1e-24))
        ),
        "cv_fold_r2_min": float(np.min(fold_r2)),
        "cv_fold_r2_max": float(np.max(fold_r2)),
        "cv_folds": int(len(fold_r2)),
    }


def rdm_spearman(source_centroids: np.ndarray, target_centroids: np.ndarray) -> float:
    def upper(values: np.ndarray) -> np.ndarray:
        differences = values[:, None, :] - values[None, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        return distances[np.triu_indices(len(values), 1)]

    result = spearmanr(upper(source_centroids), upper(target_centroids))
    return float(getattr(result, "statistic", result[0]))


def permutation_control(
    fit: Any, *, repeats: int, rng: np.random.Generator
) -> dict[str, float]:
    source_coordinates = fit.source.coordinates(fit.source.centroids)
    target_coordinates = fit.target.coordinates(fit.target.centroids)
    gram = source_coordinates.T @ source_coordinates
    values = []
    for _ in range(int(repeats)):
        permuted = target_coordinates[rng.permutation(len(target_coordinates))]
        matrix = np.linalg.solve(
            gram + fit.ridge * np.eye(source_coordinates.shape[1]),
            source_coordinates.T @ permuted,
        )
        values.append(matrix_r2(target_coordinates, source_coordinates @ matrix))
    null = np.asarray(values, dtype=float)
    return {
        "permutation_repeats": int(repeats),
        "permutation_r2_mean": float(null.mean()),
        "permutation_r2_ci95_low": float(np.quantile(null, 0.025)),
        "permutation_r2_ci95_high": float(np.quantile(null, 0.975)),
        "permutation_p_ge_observed": float(
            (1 + np.count_nonzero(null >= fit.training_r2 - 1e-15)) / (len(null) + 1)
        ),
    }


def bootstrap_stability(
    source_states: np.ndarray,
    target_states: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    reference: Any,
    *,
    repeats: int,
    rank: int,
    ridge_relative_scale: float,
    rng: np.random.Generator,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if rank != 3:
        raise ValueError("rotation stability is registered only for rank three")
    unique_groups = np.unique(groups)
    rows: list[dict[str, float]] = []
    for repeat in range(int(repeats)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled_groups]
        )
        resampled = fit_layer_map(
            source_states[indices],
            target_states[indices],
            labels[indices],
            rank=rank,
            ridge_relative_scale=ridge_relative_scale,
            required_classes=reference.source.classes,
        )
        aligned_map, _, _ = align_resampled_map_to_reference(resampled, reference)
        _, _, aligned_rotation, _ = polar_factors(aligned_map)
        stretch_singular = np.linalg.svd(aligned_map, compute_uv=False)
        source_angles = principal_angles_degrees(
            resampled.source.basis, reference.source.basis
        )
        target_angles = principal_angles_degrees(
            resampled.target.basis, reference.target.basis
        )
        rows.append(
            {
                "bootstrap": int(repeat),
                "map_relative_frobenius": relative_frobenius_error(
                    aligned_map, reference.matrix
                ),
                "rotation_geodesic_degrees": proper_rotation_geodesic_degrees(
                    aligned_rotation, reference.proper_rotation
                ),
                "source_basis_max_angle_degrees": float(source_angles.max()),
                "target_basis_max_angle_degrees": float(target_angles.max()),
                "stretch_singular_1": float(stretch_singular[0]),
                "stretch_singular_2": float(stretch_singular[1]),
                "stretch_singular_3": float(stretch_singular[2]),
            }
        )
    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {"bootstrap_repeats": int(repeats)}
    for column in (
        "map_relative_frobenius",
        "rotation_geodesic_degrees",
        "source_basis_max_angle_degrees",
        "target_basis_max_angle_degrees",
    ):
        summary[f"bootstrap_{column}_median"] = float(frame[column].median())
        summary[f"bootstrap_{column}_ci95_low"] = float(frame[column].quantile(0.025))
        summary[f"bootstrap_{column}_ci95_high"] = float(frame[column].quantile(0.975))
    for index in (1, 2, 3):
        column = f"stretch_singular_{index}"
        summary[f"bootstrap_{column}_mean"] = float(frame[column].mean())
        summary[f"bootstrap_{column}_sd"] = float(frame[column].std(ddof=1))
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--design-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--roles", nargs="+", default=["prompt_running", "answer_query"])
    parser.add_argument("--ranks", nargs="+", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--bootstraps", type=int)
    parser.add_argument("--permutations", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    design = json.loads(args.design_config.read_text(encoding="utf-8"))
    map_design = design["linear_map"]
    ranks = args.ranks or [int(value) for value in map_design["rank_sensitivity"]]
    folds = int(args.folds or map_design["group_folds"])
    bootstraps = int(
        args.bootstraps
        if args.bootstraps is not None
        else map_design["bootstrap_seed_resamples"]
    )
    permutations = int(
        args.permutations
        if args.permutations is not None
        else map_design["count_pairing_permutations"]
    )
    ridge_scale = float(map_design["ridge_relative_scale"])
    manifest_path = args.packed_root / "layer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {
        (row["model_label"], row["role"], int(row["layer"])): args.packed_root
        / row["path"]
        for row in manifest["datasets"]
    }
    cache: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}

    def dataset(key: tuple[str, str, int]) -> dict[str, np.ndarray]:
        if key not in paths:
            raise KeyError(f"manifest does not contain {key}")
        if key not in cache:
            cache[key] = load_npz(paths[key])
        return cache[key]

    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    boundary_count = 0
    for model_index, model in enumerate(args.models):
        for role_index, role in enumerate(args.roles):
            layers = sorted(
                key[2] for key in paths if key[0] == model and key[1] == role
            )
            if not layers:
                raise RuntimeError(f"no manifest layers for {model}/{role}")
            expected = list(range(layers[0], layers[-1] + 1))
            if layers != expected:
                raise RuntimeError(f"non-contiguous layer manifest for {model}/{role}")
            rank3_fits: list[tuple[int, Any]] = []
            for source_layer, target_layer in zip(layers[:-1], layers[1:]):
                boundary_count += 1
                source = dataset((model, role, source_layer))
                target = dataset((model, role, target_layer))
                source_indices, target_indices = paired_rows(source, target)
                source_states = source["states"][source_indices].astype(np.float64)
                target_states = target["states"][target_indices].astype(np.float64)
                labels = source["count"][source_indices].astype(int)
                groups = source["seed"][source_indices].astype(int)
                splits = source["split"][source_indices].astype(str)
                discovery = splits == "discovery"
                confirmation = splits == "confirmation"
                if len(np.unique(groups[discovery])) != len(design["discovery_seeds"]):
                    raise RuntimeError(
                        f"unexpected discovery seed count for {model}/{role}/L{source_layer}"
                    )
                rng = np.random.default_rng(
                    446000 + model_index * 10000 + role_index * 1000 + source_layer
                )
                reference_rank3 = None
                reference_row_index = None
                for rank in ranks:
                    fit = fit_layer_map(
                        source_states[discovery],
                        target_states[discovery],
                        labels[discovery],
                        rank=int(rank),
                        ridge_relative_scale=ridge_scale,
                    )
                    if int(rank) == 3:
                        reference_rank3 = fit
                    cv = grouped_cv(
                        source_states[discovery],
                        target_states[discovery],
                        labels[discovery],
                        groups[discovery],
                        rank=int(rank),
                        folds=folds,
                        ridge_relative_scale=ridge_scale,
                    )
                    singular = np.linalg.svd(fit.matrix, compute_uv=False)
                    angles = principal_angles_degrees(
                        fit.source.basis, fit.target.basis
                    )
                    row: dict[str, Any] = {
                        "model_label": model,
                        "role": role,
                        "source_layer": int(source_layer),
                        "target_layer": int(target_layer),
                        "normalized_depth": float(target_layer / max(layers[-1], 1)),
                        "rank": int(rank),
                        "discovery_rows": int(np.count_nonzero(discovery)),
                        "discovery_seeds": int(len(np.unique(groups[discovery]))),
                        "confirmation_rows": int(np.count_nonzero(confirmation)),
                        "source_centroid_capture": fit.source.variance_capture,
                        "target_centroid_capture": fit.target.variance_capture,
                        "map_training_r2": fit.training_r2,
                        "map_training_normalized_rmse": fit.training_normalized_rmse,
                        "map_ridge": fit.ridge,
                        "map_singular_1": float(singular[0]),
                        "map_singular_last": float(singular[-1]),
                        "map_condition_number": float(
                            singular[0] / max(singular[-1], 1e-12)
                        ),
                        "orthogonal_factor_determinant": fit.orthogonal_determinant,
                        "subspace_principal_angle_mean_degrees": float(angles.mean()),
                        "subspace_principal_angle_max_degrees": float(angles.max()),
                        **cv,
                    }
                    if int(rank) == 3:
                        cosine = np.clip(
                            (np.trace(fit.proper_rotation) - 1.0) / 2.0,
                            -1.0,
                            1.0,
                        )
                        row["proper_rotation_angle_pca_gauge_degrees"] = float(
                            np.degrees(np.arccos(cosine))
                        )
                        row["centroid_rdm_spearman"] = rdm_spearman(
                            fit.source.centroids, fit.target.centroids
                        )
                        row.update(
                            permutation_control(fit, repeats=permutations, rng=rng)
                        )
                    if np.any(confirmation):
                        confirmation_metrics = evaluate_layer_map(
                            fit,
                            source_states[confirmation],
                            target_states[confirmation],
                        )
                        source_centroids = class_centroids(
                            source_states[confirmation],
                            labels[confirmation],
                            fit.source.classes,
                        )
                        target_centroids = class_centroids(
                            target_states[confirmation],
                            labels[confirmation],
                            fit.target.classes,
                        )
                        centroid_metrics = evaluate_layer_map(
                            fit, source_centroids, target_centroids
                        )
                        row.update(
                            {
                                "confirmation_sample_r2": confirmation_metrics["r2"],
                                "confirmation_sample_normalized_rmse": confirmation_metrics[
                                    "normalized_rmse"
                                ],
                                "confirmation_centroid_r2": centroid_metrics["r2"],
                                "confirmation_centroid_normalized_rmse": centroid_metrics[
                                    "normalized_rmse"
                                ],
                            }
                        )
                    summary_rows.append(row)
                    if int(rank) == 3:
                        reference_row_index = len(summary_rows) - 1
                if reference_rank3 is None:
                    raise RuntimeError("rank three must be included in ranks")
                if reference_row_index is None:
                    raise RuntimeError("rank-three summary row is missing")
                stability, samples = bootstrap_stability(
                    source_states[discovery],
                    target_states[discovery],
                    labels[discovery],
                    groups[discovery],
                    reference_rank3,
                    repeats=bootstraps,
                    rank=3,
                    ridge_relative_scale=ridge_scale,
                    rng=rng,
                )
                summary_rows[reference_row_index].update(stability)
                rank3_fits.append((reference_row_index, reference_rank3))
                for sample in samples:
                    bootstrap_rows.append(
                        {
                            "model_label": model,
                            "role": role,
                            "source_layer": int(source_layer),
                            "target_layer": int(target_layer),
                            **sample,
                        }
                    )
                print(
                    f"[layer-map] {model} {role} L{source_layer}->L{target_layer} "
                    f"cvR2={summary_rows[reference_row_index]['cv_centroid_r2']:+.4f} "
                    f"stability={stability['bootstrap_map_relative_frobenius_median']:.4f}",
                    flush=True,
                )
            for (row_index, current), (_, following) in zip(
                rank3_fits[:-1], rank3_fits[1:]
            ):
                summary_rows[row_index].update(
                    consecutive_full_operator_metrics(current, following)
                )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    if np.isinf(summary.select_dtypes(include=[np.number]).to_numpy()).any():
        raise RuntimeError("infinite numeric values in layer-map summary")
    summary.to_csv(args.output / "layerwise_linear_map_summary.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output / "layerwise_linear_map_bootstrap.csv", index=False
    )
    elapsed = time.perf_counter() - started
    audit = {
        "schema_version": "realistic_niah_v4_4_layerwise_rotation_analysis_v1",
        "status": "PASS",
        "command": sys.argv,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "manifest": str(manifest_path),
        "design_config": str(args.design_config),
        "models": args.models,
        "roles": args.roles,
        "ranks": ranks,
        "group_folds": folds,
        "bootstrap_seed_resamples": bootstraps,
        "count_pairing_permutations": permutations,
        "boundaries": boundary_count,
        "summary_rows": len(summary_rows),
        "bootstrap_rows": len(bootstrap_rows),
        "pca_rotation_warning": "raw PCA-coordinate rotation angle is gauge dependent; use principal angles, prediction, and gauge-aligned bootstrap dispersion as primary metrics",
        "elapsed_seconds": elapsed,
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
