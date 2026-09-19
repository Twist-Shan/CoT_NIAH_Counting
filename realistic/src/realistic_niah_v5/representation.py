from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .spec import REGISTERED_COHORTS, V5Config


REPRESENTATION_SCHEMA_VERSION = "realistic_niah_v5_representation_v1"
ANSWER_QUERY_SITE_KINDS = frozenset(
    {"answer_query", "answer_query_v2", "answer_query_v3"}
)


@dataclass(frozen=True)
class RepresentationDataset:
    metadata: pd.DataFrame
    states: np.ndarray

    def validate(self) -> None:
        if self.states.ndim != 2:
            raise ValueError("states must have shape [observations, hidden]")
        if len(self.metadata) != len(self.states):
            raise ValueError("metadata/state row mismatch")
        needed = {
            "request_id",
            "model_label",
            "seed",
            "split",
            "gold_count",
            "occurrence",
            "layer",
            "site_kind",
            "parser_hit",
            "trace_one_to_one",
            "exact_count",
        }
        missing = sorted(needed - set(self.metadata.columns))
        if missing:
            raise ValueError(f"Representation metadata is missing {missing}")
        if not np.isfinite(self.states).all():
            raise ValueError("states contain non-finite values")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL line {line_number}: {error}") from error
    return rows


def load_capture_dataset(
    capture_index: str | Path,
    *,
    site_kinds: Sequence[str] | None = None,
) -> RepresentationDataset:
    """Materialize registered site states from restartable V5 capture shards."""

    index_path = Path(capture_index)
    root = index_path.parent
    allowed = set(site_kinds) if site_kinds is not None else None
    metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for index_row in _read_jsonl(index_path):
        manifest_path = root / str(index_row["manifest_path"])
        states_path = root / str(index_row["states_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive = np.load(states_path, allow_pickle=False)
        states = np.asarray(archive["site_states"], dtype=np.float32)
        layers = np.asarray(archive["layer_indices"], dtype=int)
        site_rows = list(manifest["site_rows"])
        if states.shape[:2] != (len(site_rows), len(layers)):
            raise ValueError(f"Capture shape mismatch in {states_path}")
        parser = manifest["parser"]
        for site_axis, site in enumerate(site_rows):
            if allowed is not None and str(site["site_kind"]) not in allowed:
                continue
            for layer_axis, layer in enumerate(layers):
                occurrence = site.get("occurrence")
                metadata.append(
                    {
                        "request_id": manifest.get("request_id"),
                        "stimulus_id": manifest.get("stimulus_id"),
                        "model_label": manifest.get("model_label"),
                        "model_family": manifest.get("model_family"),
                        "seed": manifest.get("seed"),
                        "split": manifest.get("split"),
                        "gold_count": manifest.get("gold_count"),
                        "parsed_count": manifest.get("parsed_count"),
                        "occurrence": occurrence,
                        "remaining_count": (
                            int(manifest["gold_count"]) - int(occurrence)
                            if occurrence is not None
                            else np.nan
                        ),
                        "normalized_progress": (
                            float(occurrence) / float(manifest["gold_count"])
                            if occurrence is not None and manifest["gold_count"]
                            else np.nan
                        ),
                        "layer": int(layer),
                        "site_id": site["site_id"],
                        "site_kind": site["site_kind"],
                        "alignment_strategy": site.get("alignment_strategy"),
                        "parser_hit": bool(parser.get("detected")),
                        "trace_one_to_one": bool(parser.get("trace_one_to_one")),
                        "trace_category": parser.get("trace_category"),
                        "exact_count": bool(manifest.get("exact_count")),
                    }
                )
                vectors.append(states[site_axis, layer_axis])
    if not vectors:
        raise ValueError("No registered states matched the requested site kinds")
    result = RepresentationDataset(
        metadata=pd.DataFrame(metadata),
        states=np.stack(vectors, axis=0),
    )
    result.validate()
    return result


def cohort_mask(metadata: pd.DataFrame, cohort: str) -> np.ndarray:
    if cohort not in REGISTERED_COHORTS:
        raise ValueError(f"Unknown V5 cohort: {cohort}")
    mask = metadata["parser_hit"].astype(bool).to_numpy()
    if cohort in {"one_to_one", "one_to_one_correct"}:
        mask &= metadata["trace_one_to_one"].astype(bool).to_numpy()
    if cohort == "one_to_one_correct":
        mask &= metadata["exact_count"].astype(bool).to_numpy()
    return mask


def _effective_rank(singular_values: np.ndarray) -> float:
    values = np.square(singular_values.astype(float))
    if values.sum() <= 0:
        return 0.0
    probabilities = values / values.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _rank_at_fraction(singular_values: np.ndarray, fraction: float) -> int:
    values = np.square(singular_values.astype(float))
    if values.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(values) / values.sum(), fraction) + 1)


def rank_metrics(states: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if len(states) < 2:
        raise ValueError("Rank metrics need at least two observations")
    centered = states - states.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    stable = float(energy.sum() / energy.max()) if energy.max() > 0 else 0.0
    unique = np.unique(labels)
    centroids = np.stack([states[labels == value].mean(axis=0) for value in unique])
    centered_centroids = centroids - centroids.mean(axis=0, keepdims=True)
    centroid_singular = np.linalg.svd(
        centered_centroids, full_matrices=False, compute_uv=False
    )
    centroid_energy = np.square(centroid_singular)
    centroid_total = float(centroid_energy.sum())
    return {
        "n_observations": int(len(states)),
        "hidden_size": int(states.shape[1]),
        "n_labels": int(len(unique)),
        "stable_rank": stable,
        "effective_rank": _effective_rank(singular),
        "rank_90": _rank_at_fraction(singular, 0.90),
        "rank_95": _rank_at_fraction(singular, 0.95),
        "centroid_rank_1_fraction": (
            float(centroid_energy[0] / centroid_total) if centroid_total else 0.0
        ),
        "centroid_rank_3_fraction": (
            float(centroid_energy[:3].sum() / centroid_total)
            if centroid_total
            else 0.0
        ),
    }


def _registered_split(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    split = metadata["split"].astype(str).str.lower()
    discovery = split.eq("discovery").to_numpy()
    confirmation = split.eq("confirmation").to_numpy()
    if not discovery.any() or not confirmation.any():
        raise ValueError(
            "Formal V5 probes require explicit discovery and confirmation rows"
        )
    if np.any(discovery & confirmation):
        raise ValueError("Discovery/confirmation rows overlap")
    return discovery, confirmation


def _select_ridge_alpha(
    states: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float(alphas[len(alphas) // 2])
    scores: dict[float, list[float]] = {float(alpha): [] for alpha in alphas}
    for group in unique_groups:
        train = groups != group
        test = groups == group
        if train.sum() < 2 or test.sum() < 1:
            continue
        for alpha in scores:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(states[train], targets[train])
            prediction = model.predict(states[test])
            scores[alpha].append(float(mean_absolute_error(targets[test], prediction)))
    return min(
        scores,
        key=lambda alpha: (
            np.mean(scores[alpha]) if scores[alpha] else np.inf,
            alpha,
        ),
    )


def regression_metrics(
    states: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
    *,
    alphas: Sequence[float],
) -> list[dict[str, Any]]:
    discovery, confirmation = _registered_split(metadata)
    alpha = _select_ridge_alpha(
        states[discovery],
        targets[discovery],
        metadata.loc[discovery, "seed"].to_numpy(),
        alphas,
    )
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        "knn_5": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=min(5, int(discovery.sum()))),
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(states[discovery], targets[discovery])
        prediction = model.predict(states[confirmation])
        rows.append(
            {
                "probe": name,
                "ridge_alpha": alpha if name == "ridge" else np.nan,
                "n_discovery": int(discovery.sum()),
                "n_confirmation": int(confirmation.sum()),
                "confirmation_r2": (
                    float(r2_score(targets[confirmation], prediction))
                    if confirmation.sum() >= 2
                    else np.nan
                ),
                "confirmation_mae": float(
                    mean_absolute_error(targets[confirmation], prediction)
                ),
            }
        )
    return rows


def classification_metrics(
    states: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    *,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    discovery, confirmation = _registered_split(metadata)
    n_neighbors = min(5, int(discovery.sum()))
    models = {
        "knn_5": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors)
        ),
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=random_state),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(), LinearSVC(random_state=random_state)
        ),
        "lda": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        try:
            fitted = clone(model).fit(states[discovery], labels[discovery])
            prediction = fitted.predict(states[confirmation])
            rows.append(
                {
                    "classifier": name,
                    "n_discovery": int(discovery.sum()),
                    "n_confirmation": int(confirmation.sum()),
                    "accuracy": float(accuracy_score(labels[confirmation], prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(labels[confirmation], prediction)
                    ),
                    "macro_f1": float(
                        f1_score(labels[confirmation], prediction, average="macro")
                    ),
                    "status": "ok",
                }
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append(
                {
                    "classifier": name,
                    "n_discovery": int(discovery.sum()),
                    "n_confirmation": int(confirmation.sum()),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    "status": f"not_estimable:{type(error).__name__}",
                }
            )
    return rows


def clustering_metrics(
    states: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    *,
    random_state: int = 0,
) -> dict[str, Any]:
    _discovery, confirmation = _registered_split(metadata)
    x = StandardScaler().fit_transform(states[confirmation])
    y = labels[confirmation]
    unique = np.unique(y)
    result: dict[str, Any] = {
        "n_confirmation": int(len(x)),
        "n_labels": int(len(unique)),
        "silhouette_cosine": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
        "kmeans_adjusted_rand": np.nan,
    }
    if 1 < len(unique) < len(x):
        result.update(
            {
                "silhouette_cosine": float(silhouette_score(x, y, metric="cosine")),
                "calinski_harabasz": float(calinski_harabasz_score(x, y)),
                "davies_bouldin": float(davies_bouldin_score(x, y)),
                "kmeans_adjusted_rand": float(
                    adjusted_rand_score(
                        y,
                        KMeans(
                            n_clusters=len(unique),
                            n_init=20,
                            random_state=random_state,
                        ).fit_predict(x),
                    )
                ),
            }
        )
    return result


def curve_metrics(states: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    values = np.unique(labels)
    values.sort()
    centroids = np.stack([states[labels == value].mean(axis=0) for value in values])
    if len(centroids) < 2:
        return {
            "adjacent_step_mean": np.nan,
            "adjacent_step_cv": np.nan,
            "adjacent_cosine_mean": np.nan,
            "curvature_mean": np.nan,
            "label_distance_spearman": np.nan,
        }
    steps = np.diff(centroids, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    adjacent_cosines = []
    curvatures = []
    for left, right in zip(steps[:-1], steps[1:]):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine = float(np.dot(left, right) / denominator) if denominator else np.nan
        adjacent_cosines.append(cosine)
        curvatures.append(float(np.linalg.norm(right - left)))
    distances = squareform(pdist(centroids, metric="euclidean"))
    label_distances = np.abs(values[:, None] - values[None, :])
    upper = np.triu_indices(len(values), k=1)
    correlation = spearmanr(label_distances[upper], distances[upper]).statistic
    return {
        "adjacent_step_mean": float(norms.mean()),
        "adjacent_step_cv": (
            float(norms.std(ddof=1) / norms.mean())
            if len(norms) > 1 and norms.mean() > 0
            else 0.0
        ),
        "adjacent_cosine_mean": (
            float(np.nanmean(adjacent_cosines)) if adjacent_cosines else np.nan
        ),
        "curvature_mean": float(np.mean(curvatures)) if curvatures else np.nan,
        "label_distance_spearman": float(correlation),
    }


def noise_decomposition(states: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Descriptive centroid signal/residual decomposition, not causal ANOVA."""

    grand = states.mean(axis=0)
    centered = states - grand
    total_ss = float(np.square(centered).sum())
    fitted = np.empty_like(states)
    for label in np.unique(labels):
        fitted[labels == label] = states[labels == label].mean(axis=0)
    signal_ss = float(np.square(fitted - grand).sum())
    residual_ss = float(np.square(states - fitted).sum())
    return {
        "total_sum_squares": total_ss,
        "label_centroid_signal_sum_squares": signal_ss,
        "within_label_residual_sum_squares": residual_ss,
        "label_centroid_fraction": signal_ss / total_ss if total_ss else np.nan,
        "within_label_fraction": residual_ss / total_ss if total_ss else np.nan,
        "decomposition_identity_error": abs(total_ss - signal_ss - residual_ss),
    }


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def analyze_representation(
    capture_index: str | Path,
    output_dir: str | Path,
    *,
    config: V5Config,
    cohorts: Iterable[str] = REGISTERED_COHORTS,
) -> dict[str, Path]:
    """Reproduce the registered V4.4 geometry battery on native trace sites."""

    config.validate()
    dataset = load_capture_dataset(capture_index)
    metadata = dataset.metadata
    states = dataset.states
    output = Path(output_dir)
    summary_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    group_columns = ["model_label", "site_kind", "layer"]
    for cohort in cohorts:
        base_mask = cohort_mask(metadata, cohort)
        for key, indices in metadata.loc[base_mask].groupby(group_columns).groups.items():
            row_indices = np.asarray(list(indices), dtype=int)
            group_meta = metadata.loc[row_indices].reset_index(drop=True)
            group_states = states[row_indices]
            model_label, site_kind, layer = key
            if site_kind not in ANSWER_QUERY_SITE_KINDS:
                finite_label = group_meta["occurrence"].notna().to_numpy()
                group_meta = group_meta.loc[finite_label].reset_index(drop=True)
                group_states = group_states[finite_label]
                labels = group_meta["occurrence"].astype(int).to_numpy()
                label_name = "running_index"
            else:
                labels = group_meta["gold_count"].astype(int).to_numpy()
                label_name = "gold_count"
            if (
                config.representation_n10_only
                and site_kind not in ANSWER_QUERY_SITE_KINDS
            ):
                keep = group_meta["gold_count"].astype(int).eq(10).to_numpy()
                group_meta = group_meta.loc[keep].reset_index(drop=True)
                group_states = group_states[keep]
                labels = labels[keep]
            prefix = {
                "schema_version": REPRESENTATION_SCHEMA_VERSION,
                "model_label": model_label,
                "cohort": cohort,
                "site_kind": site_kind,
                "layer": int(layer),
                "label": label_name,
            }
            if len(group_states) < 3 or len(np.unique(labels)) < 2:
                skipped.append({**prefix, "reason": "insufficient_rows_or_labels"})
                continue
            descriptive = {
                **rank_metrics(group_states, labels),
                **curve_metrics(group_states, labels),
                **noise_decomposition(group_states, labels),
            }
            try:
                descriptive.update(clustering_metrics(group_states, labels, group_meta))
                for row in regression_metrics(
                    group_states,
                    labels.astype(float),
                    group_meta,
                    alphas=config.ridge_alphas,
                ):
                    regression_rows.append({**prefix, **row})
                for row in classification_metrics(group_states, labels, group_meta):
                    classifier_rows.append({**prefix, **row})
            except ValueError as error:
                skipped.append({**prefix, "reason": str(error)})
            summary_rows.append({**prefix, **descriptive})
    paths = {
        "summary": output / "geometry_summary.csv",
        "regression": output / "regression_confirmation.csv",
        "classification": output / "classification_confirmation.csv",
        "audit": output / "representation_audit.json",
    }
    _atomic_csv(paths["summary"], pd.DataFrame(summary_rows))
    _atomic_csv(paths["regression"], pd.DataFrame(regression_rows))
    _atomic_csv(paths["classification"], pd.DataFrame(classifier_rows))
    _atomic_json(
        paths["audit"],
        {
            "schema_version": REPRESENTATION_SCHEMA_VERSION,
            "capture_index": str(Path(capture_index).resolve()),
            "primary_site": config.primary_trace_site,
            "sensitivity_sites": list(config.sensitivity_trace_sites),
            "observed_sites": sorted(metadata["site_kind"].astype(str).unique()),
            "cohorts": list(cohorts),
            "representation_n10_only": config.representation_n10_only,
            "selection_policy": (
                "site/cohort fixed before confirmation; no best-layer selection on "
                "confirmation metrics"
            ),
            "rows_loaded": len(metadata),
            "groups_completed": len(summary_rows),
            "skipped": skipped,
        },
    )
    return paths
