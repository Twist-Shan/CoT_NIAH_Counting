from __future__ import annotations

"""Quantify rank, compression, clustering, and structured noise in V4.4 states.

The script treats the seed as the independent unit.  Prompt-running rows come
from ten endpoint states inside each N=10 prompt.  Answer-query rows come from
separate count-conditioned prompts.  Metrics that need held-out predictions use
the frozen discovery/confirmation split when both are available, otherwise a
five-fold GroupKFold over seeds.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
KS = (1, 2, 3, 5, 9, 16, 32)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "datasets" in payload:
        return list(payload["datasets"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported packed manifest schema: {path}")


def load_dataset(manifest: Path, row: dict[str, Any]) -> dict[str, np.ndarray]:
    source = Path(row["path"])
    if not source.is_absolute():
        source = manifest.parent / source
    with np.load(source, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def svd_metrics(x: np.ndarray) -> dict[str, float | int]:
    centered = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = np.square(singular)
    total = float(variance.sum())
    proportions = variance / max(total, EPS)
    cumulative = np.cumsum(proportions)
    positive = proportions[proportions > 0]
    result: dict[str, float | int] = {
        "stable_rank": float(total / max(float(variance[0]), EPS)),
        "effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "numeric_rank_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "numeric_rank_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "numeric_rank_99": int(np.searchsorted(cumulative, 0.99) + 1),
    }
    for k in KS:
        result[f"total_variance_capture_k{k}"] = float(cumulative[min(k, len(cumulative)) - 1])
    return result


def centroid_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    levels = np.unique(y)
    grand = x.mean(axis=0)
    centroids = np.stack([x[y == level].mean(axis=0) for level in levels])
    weights = np.asarray([(y == level).sum() for level in levels], dtype=np.float64)
    ss_total = float(np.square(x - grand).sum())
    ss_between = float(sum(w * np.square(c - grand).sum() for w, c in zip(weights, centroids)))
    curve = centroids - np.average(centroids, axis=0, weights=weights)
    singular = np.linalg.svd(curve.astype(np.float64), full_matrices=False, compute_uv=False)
    variance = np.square(singular)
    cumulative = np.cumsum(variance / max(float(variance.sum()), EPS))
    out = {"count_eta_squared": ss_between / max(ss_total, EPS)}
    for k in KS:
        out[f"centroid_curve_capture_k{k}"] = float(cumulative[min(k, len(cumulative)) - 1])
    return out


def split_iterator(split: np.ndarray, groups: np.ndarray) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    discovery = np.flatnonzero(split.astype(str) == "discovery")
    confirmation = np.flatnonzero(split.astype(str) == "confirmation")
    if len(discovery) and len(confirmation):
        yield "discovery_to_confirmation", discovery, confirmation
        return
    unique_groups = np.unique(groups)
    folds = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups)):
        yield f"group_fold_{fold}", train, test


def predictive_metrics(x: np.ndarray, y: np.ndarray, groups: np.ndarray, split: np.ndarray) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split_name, train, test in split_iterator(split, groups):
        components = min(32, len(train) - 1, x.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=20260806)
        train_x = pca.fit_transform(x[train].astype(np.float32))
        test_x = pca.transform(x[test].astype(np.float32))
        models = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "knn5_distance": make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=5, weights="distance", metric="euclidean"),
            ),
        }
        for name, model in models.items():
            model.fit(train_x, y[train])
            prediction = model.predict(test_x)
            records.append(
                {
                    "fold": split_name,
                    "algorithm": name,
                    "rows_train": len(train),
                    "rows_test": len(test),
                    "r2": float(r2_score(y[test], prediction)),
                    "mae": float(mean_absolute_error(y[test], prediction)),
                    "pearson_r": float(pearsonr(y[test], prediction).statistic),
                }
            )
    return records


def clustering_metrics(x: np.ndarray, y: np.ndarray, groups: np.ndarray, split: np.ndarray) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split_name, train, test in split_iterator(split, groups):
        components = min(32, len(train) - 1, x.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=20260806)
        pca.fit(x[train].astype(np.float32))
        z = pca.transform(x[test].astype(np.float32))
        records.append(
            {
                "fold": split_name,
                "rows_test": len(test),
                "silhouette_cosine": float(silhouette_score(z, y[test], metric="cosine")),
                "calinski_harabasz": float(calinski_harabasz_score(z, y[test])),
                "davies_bouldin": float(davies_bouldin_score(z, y[test])),
            }
        )
    return records


def prompt_noise_decomposition(x: np.ndarray, count: np.ndarray, seed: np.ndarray) -> dict[str, float]:
    levels = sorted(np.unique(count).tolist())
    seeds = sorted(np.unique(seed).tolist())
    if len(x) != len(levels) * len(seeds):
        raise ValueError("Prompt running-index grid is not balanced")
    index = {(int(s), int(c)): i for i, (s, c) in enumerate(zip(seed, count))}
    cube = np.stack([np.stack([x[index[(s, c)]] for c in levels]) for s in seeds]).astype(np.float64)
    grand = cube.mean(axis=(0, 1), keepdims=True)
    count_effect = cube.mean(axis=0, keepdims=True) - grand
    seed_effect = cube.mean(axis=1, keepdims=True) - grand
    interaction = cube - grand - count_effect - seed_effect
    ss_count = float(len(seeds) * np.square(count_effect).sum())
    ss_seed = float(len(levels) * np.square(seed_effect).sum())
    ss_interaction = float(np.square(interaction).sum())
    total = ss_count + ss_seed + ss_interaction
    return {
        "ss_count": ss_count,
        "ss_seed_context": ss_seed,
        "ss_count_by_seed_interaction": ss_interaction,
        "fraction_count": ss_count / max(total, EPS),
        "fraction_seed_context": ss_seed / max(total, EPS),
        "fraction_count_by_seed_interaction": ss_interaction / max(total, EPS),
    }


def aggregate_folds(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    pieces = []
    for group, part in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, group if isinstance(group, tuple) else (group,)))
        row["folds"] = len(part)
        for metric in metrics:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_min"] = float(part[metric].min())
            row[f"{metric}_max"] = float(part[metric].max())
        pieces.append(row)
    return pd.DataFrame(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--roles", nargs="+", default=["prompt_running", "answer_query"])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest(args.manifest)
    rank_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    clustering_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for row in manifest_rows:
        model = str(row["model_label"])
        role = str(row["role"])
        if model not in args.models or role not in args.roles:
            continue
        data = load_dataset(args.manifest, row)
        x = data["states"].astype(np.float32)
        y = data["count"].astype(np.int64)
        seed = data["seed"].astype(np.int64)
        split = data["split"].astype(str)
        layer = int(row["layer"])
        base = {
            "model_label": model,
            "role": role,
            "layer": layer,
            "rows": len(x),
            "seeds": len(np.unique(seed)),
        }
        fit_mask = split == "discovery" if np.any(split == "confirmation") else np.ones(len(x), dtype=bool)
        rank_rows.append({**base, "fit_population": "discovery" if np.any(split == "confirmation") else "all_available", **svd_metrics(x[fit_mask]), **centroid_metrics(x[fit_mask], y[fit_mask])})
        for metric in predictive_metrics(x, y, seed, split):
            regression_rows.append({**base, **metric})
        for metric in clustering_metrics(x, y, seed, split):
            clustering_rows.append({**base, **metric})
        if role == "prompt_running":
            for population, mask in (("all", np.ones(len(x), dtype=bool)), ("discovery", split == "discovery"), ("confirmation", split == "confirmation")):
                if mask.sum() == 0:
                    continue
                noise_rows.append({**base, "population": population, **prompt_noise_decomposition(x[mask], y[mask], seed[mask])})
        sources.append(str(row["path"]))
        print(f"[geometry] {model} {role} L{layer}", flush=True)

    rank = pd.DataFrame(rank_rows)
    regression = pd.DataFrame(regression_rows)
    clustering = pd.DataFrame(clustering_rows)
    noise = pd.DataFrame(noise_rows)
    rank.to_csv(args.output / "rank_and_compression_by_layer.csv", index=False)
    regression.to_csv(args.output / "count_regression_folds.csv", index=False)
    aggregate_folds(regression, ["model_label", "role", "layer", "algorithm"], ["r2", "mae", "pearson_r"]).to_csv(args.output / "count_regression_summary.csv", index=False)
    clustering.to_csv(args.output / "clustering_folds.csv", index=False)
    aggregate_folds(clustering, ["model_label", "role", "layer"], ["silhouette_cosine", "calinski_harabasz", "davies_bouldin"]).to_csv(args.output / "clustering_summary.csv", index=False)
    noise.to_csv(args.output / "prompt_noise_two_way_decomposition.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_extension_geometry_v1",
        "independent_unit": "seed",
        "prompt_semantics": "ten needle endpoints within each N=10 prompt",
        "answer_semantics": "prompt-final Total: query across count-conditioned prompts",
        "rank_definition": "SVD of centered sample-by-hidden matrix; stable rank=sum(s^2)/max(s^2); effective rank=exp(entropy(s^2/sum(s^2)))",
        "centroid_definition": "count eta squared and rank-k variance of the ten centered count centroids",
        "prediction_split": "discovery-to-confirmation when present, otherwise 5-fold GroupKFold by seed",
        "clustering_space": "PCA32 fit only on training rows and evaluated on held-out rows",
        "noise_decomposition": "balanced two-way Frobenius ANOVA: count + seed/context + count-by-seed interaction",
        "rank_rows": len(rank),
        "regression_rows": len(regression),
        "clustering_rows": len(clustering),
        "noise_rows": len(noise),
        "sources": sorted(set(sources)),
        "status": "PASS",
    }
    atomic_json(args.output / "extension_geometry_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
