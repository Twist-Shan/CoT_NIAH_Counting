from __future__ import annotations

"""Rotation-invariant and aligned-decoding audit for V4.4.5 counter states.

This script only reads the frozen packed layer files.  It asks whether count
geometry is preserved across layers even when the residual-space axes rotate.
All learned bases, alignments, scalers, and classifiers are fitted inside
seed-held-out folds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def pair_rows(left: dict[str, np.ndarray], right: dict[str, np.ndarray]):
    left_index = {str(value): index for index, value in enumerate(left["sample_id"])}
    right_index = {str(value): index for index, value in enumerate(right["sample_id"])}
    shared = sorted(set(left_index) & set(right_index))
    li = np.asarray([left_index[key] for key in shared], dtype=np.int64)
    ri = np.asarray([right_index[key] for key in shared], dtype=np.int64)
    if not np.array_equal(left["count"][li], right["count"][ri]):
        raise RuntimeError("paired rows disagree on count")
    if not np.array_equal(left["seed"][li], right["seed"][ri]):
        raise RuntimeError("paired rows disagree on seed")
    return li, ri


def centroid_basis(states: np.ndarray, labels: np.ndarray, rank: int):
    classes = np.unique(labels)
    centroids = np.stack([states[labels == value].mean(axis=0) for value in classes])
    center = centroids.mean(axis=0)
    _, singular, vt = np.linalg.svd(centroids - center, full_matrices=False)
    basis = vt[:rank].T
    capture = float(np.square(singular[:rank]).sum() / max(np.square(singular).sum(), 1e-12))
    return classes, centroids, center, basis, capture


def rdm_vector(states: np.ndarray, labels: np.ndarray):
    classes, centroids, _, _, _ = centroid_basis(states, labels, rank=3)
    if len(classes) != 10:
        raise RuntimeError(f"RDM requires ten count classes, got {classes}")
    differences = centroids[:, None, :] - centroids[None, :, :]
    distances = np.sqrt(np.square(differences, dtype=np.float64).sum(axis=2))
    return distances[np.triu_indices(len(classes), 1)]


def rdm_similarity(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], *, bootstraps: int, rng
):
    li, ri = pair_rows(left, right)
    left_states = left["states"][li].astype(np.float32)
    right_states = right["states"][ri].astype(np.float32)
    labels = left["count"][li]
    seeds = left["seed"][li]
    def correlation(a, b) -> float:
        result = spearmanr(a, b)
        return float(getattr(result, "statistic", result[0]))

    estimate = correlation(rdm_vector(left_states, labels), rdm_vector(right_states, labels))
    unique = np.unique(seeds)
    samples = []
    for _ in range(bootstraps):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(seeds == seed) for seed in selected])
        samples.append(
            correlation(
                rdm_vector(left_states[indices], labels[indices]),
                rdm_vector(right_states[indices], labels[indices]),
            )
        )
    return estimate, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), len(unique)


def orthogonal_map(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    # Row-vector convention: target @ R approximates source.
    u, _, vt = np.linalg.svd(target.T @ source, full_matrices=False)
    return u @ vt


def aligned_decode(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], *, rank: int, folds: int, rng,
    shuffle_repeats: int = 1,
):
    li, ri = pair_rows(left, right)
    xs = left["states"][li].astype(np.float32)
    xt = right["states"][ri].astype(np.float32)
    y = left["count"][li].astype(int)
    groups = left["seed"][li]
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    truth, test_groups, aligned_pred, direct_pred, within_pred = [], [], [], [], []
    shuffled_by_repeat: list[list[np.ndarray]] = [[] for _ in range(shuffle_repeats)]
    captures = []
    for train, test in splitter.split(xs, y, groups):
        _, _, cs, us, cap_s = centroid_basis(xs[train], y[train], rank)
        _, _, ct, ut, cap_t = centroid_basis(xt[train], y[train], rank)
        zs_train = (xs[train] - cs) @ us
        zs_test = (xs[test] - cs) @ us
        zt_train = (xt[train] - ct) @ ut
        zt_test = (xt[test] - ct) @ ut
        rotation = orthogonal_map(zt_train, zs_train)

        source_scale = StandardScaler().fit(zs_train)
        source_model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(
            source_scale.transform(zs_train), y[train]
        )
        target_scale = StandardScaler().fit(zt_train)
        target_model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(
            target_scale.transform(zt_train), y[train]
        )
        truth.append(y[test])
        test_groups.append(groups[test])
        aligned_pred.append(source_model.predict(source_scale.transform(zt_test @ rotation)))
        direct_pred.append(source_model.predict(source_scale.transform(zt_test)))
        for repeat in range(shuffle_repeats):
            shuffled = rng.permutation(len(train))
            shuffled_rotation = orthogonal_map(zt_train, zs_train[shuffled])
            shuffled_by_repeat[repeat].append(
                source_model.predict(source_scale.transform(zt_test @ shuffled_rotation))
            )
        within_pred.append(target_model.predict(target_scale.transform(zt_test)))
        captures.append((cap_s, cap_t))
    truth = np.concatenate(truth)
    test_groups = np.concatenate(test_groups)
    aligned_pred = np.concatenate(aligned_pred)
    direct_pred = np.concatenate(direct_pred)
    shuffled_predictions = [np.concatenate(values) for values in shuffled_by_repeat]
    within_pred = np.concatenate(within_pred)
    unique_groups = np.unique(test_groups)
    aligned_seed = np.asarray([
        np.mean(aligned_pred[test_groups == group] == truth[test_groups == group])
        for group in unique_groups
    ])
    shuffled_seed = np.asarray([
        np.mean([
            np.mean(pred[test_groups == group] == truth[test_groups == group])
            for pred in shuffled_predictions
        ])
        for group in unique_groups
    ])
    shuffled_accuracy_null = np.asarray([
        np.mean(pred == truth) for pred in shuffled_predictions
    ])
    bootstrap_indices = rng.integers(
        0, len(unique_groups), size=(10_000, len(unique_groups))
    )
    aligned_bootstrap = aligned_seed[bootstrap_indices].mean(axis=1)
    improvement_bootstrap = (aligned_seed - shuffled_seed)[bootstrap_indices].mean(axis=1)
    return {
        "rows": int(len(truth)),
        "seeds": int(len(np.unique(groups))),
        "aligned_accuracy": float(accuracy_score(truth, aligned_pred)),
        "aligned_accuracy_ci95_low": float(np.quantile(aligned_bootstrap, 0.025)),
        "aligned_accuracy_ci95_high": float(np.quantile(aligned_bootstrap, 0.975)),
        "aligned_mae": float(mean_absolute_error(truth, aligned_pred)),
        "direct_accuracy": float(accuracy_score(truth, direct_pred)),
        "shuffled_alignment_accuracy": float(shuffled_accuracy_null.mean()),
        "shuffled_alignment_ci95_low": float(np.quantile(shuffled_accuracy_null, 0.025)),
        "shuffled_alignment_ci95_high": float(np.quantile(shuffled_accuracy_null, 0.975)),
        "shuffle_repeats": int(shuffle_repeats),
        "shuffle_null_p_greater_equal_aligned": float(
            (1 + np.sum(shuffled_accuracy_null >= accuracy_score(truth, aligned_pred)))
            / (shuffle_repeats + 1)
        ),
        "aligned_minus_shuffled": float(aligned_seed.mean() - shuffled_seed.mean()),
        "aligned_minus_shuffled_ci95_low": float(np.quantile(improvement_bootstrap, 0.025)),
        "aligned_minus_shuffled_ci95_high": float(np.quantile(improvement_bootstrap, 0.975)),
        "within_target_accuracy": float(accuracy_score(truth, within_pred)),
        "source_capture": float(np.mean([x[0] for x in captures])),
        "target_capture": float(np.mean([x[1] for x in captures])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    manifest_path = args.packed_root / "layer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {
        (row["model_label"], row["role"], int(row["layer"])): args.packed_root / row["path"]
        for row in manifest["datasets"]
    }
    cache: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}

    def get(key):
        if key not in cache:
            cache[key] = load_dataset(paths[key])
        return cache[key]

    rng = np.random.default_rng(445)
    rdm_rows = []
    decode_rows = []
    for model in sorted({key[0] for key in paths}):
        for role in ("prompt_running", "answer_query"):
            layers = sorted(key[2] for key in paths if key[0] == model and key[1] == role)
            for left_layer, right_layer in zip(layers[:-1], layers[1:]):
                left, right = get((model, role, left_layer)), get((model, role, right_layer))
                estimate, low, high, seeds = rdm_similarity(
                    left, right, bootstraps=args.bootstraps, rng=rng
                )
                rdm_rows.append({
                    "model_label": model, "comparison": "adjacent_same_role", "left_role": role,
                    "left_layer": left_layer, "right_role": role, "right_layer": right_layer,
                    "rdm_spearman": estimate, "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high, "seeds": seeds,
                })
                result = aligned_decode(left, right, rank=args.rank, folds=args.folds, rng=rng)
                decode_rows.append({
                    "model_label": model, "comparison": "adjacent_same_role", "left_role": role,
                    "left_layer": left_layer, "right_role": role, "right_layer": right_layer, **result,
                })
        prompt_layers = {key[2] for key in paths if key[0] == model and key[1] == "prompt_running"}
        answer_layers = {key[2] for key in paths if key[0] == model and key[1] == "answer_query"}
        for layer in sorted(prompt_layers & answer_layers):
            left, right = get((model, "prompt_running", layer)), get((model, "answer_query", layer))
            estimate, low, high, seeds = rdm_similarity(
                left, right, bootstraps=args.bootstraps, rng=rng
            )
            rdm_rows.append({
                "model_label": model, "comparison": "same_layer_cross_role",
                "left_role": "prompt_running", "left_layer": layer,
                "right_role": "answer_query", "right_layer": layer,
                "rdm_spearman": estimate, "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high, "seeds": seeds,
            })
            result = aligned_decode(left, right, rank=args.rank, folds=args.folds, rng=rng)
            decode_rows.append({
                "model_label": model, "comparison": "same_layer_cross_role",
                "left_role": "prompt_running", "left_layer": layer,
                "right_role": "answer_query", "right_layer": layer, **result,
            })

        mechanism = (28, 29) if model == "Qwen3-8B" else (36, 37)
        left = get((model, "prompt_running", mechanism[0]))
        right = get((model, "answer_query", mechanism[1]))
        estimate, low, high, seeds = rdm_similarity(left, right, bootstraps=args.bootstraps, rng=rng)
        rdm_rows.append({
            "model_label": model, "comparison": "mechanism_prompt_to_answer",
            "left_role": "prompt_running", "left_layer": mechanism[0],
            "right_role": "answer_query", "right_layer": mechanism[1],
            "rdm_spearman": estimate, "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high, "seeds": seeds,
        })
        result = aligned_decode(left, right, rank=args.rank, folds=args.folds, rng=rng)
        decode_rows.append({
            "model_label": model, "comparison": "mechanism_prompt_to_answer",
            "left_role": "prompt_running", "left_layer": mechanism[0],
            "right_role": "answer_query", "right_layer": mechanism[1], **result,
        })

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rdm_rows).to_csv(args.output / "centroid_rdm_transport.csv", index=False)
    pd.DataFrame(decode_rows).to_csv(args.output / "procrustes_aligned_decode.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_5_transport_supplement_v1",
        "source": str(manifest_path),
        "source_read_only": True,
        "rank": args.rank,
        "folds": args.folds,
        "bootstraps": args.bootstraps,
        "rdm_rows": len(rdm_rows),
        "decode_rows": len(decode_rows),
        "status": "PASS",
    }
    (args.output / "transport_supplement_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
