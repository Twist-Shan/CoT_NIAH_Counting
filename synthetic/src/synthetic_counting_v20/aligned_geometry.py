"""Large-model-aligned representation geometry for synthetic v20.

The module mirrors the discovery/confirmation protocol used by the realistic
NiaH reports while adapting the token sites to v20:

* non-thinking prompt occurrence: the kth target character in the prompt;
* thinking item end: the marker that closes the kth trace item;
* non-thinking/thinking answer query: the ``<Ans>`` token.

Layer and metric selection read discovery rows only.  Confirmation rows are
transformed by discovery-fitted preprocessing and are never used to choose a
layer.  This distinction is central to the scientific interpretation, so the
tables retain both the selection score and the frozen confirmation estimate.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .data import V20Example, V20Rendered, V20Vocab, collate_v20, render_v20


ENDPOINT_LABELS = {
    "nonthinking_prompt_occurrence": "Non-thinking prompt occurrence",
    "thinking_item_end": "Thinking trace item end",
    "nonthinking_answer_query": "Non-thinking answer query",
    "thinking_answer_query": "Thinking answer query",
}

MODE_ENDPOINTS = {
    "nonthinking": (
        "nonthinking_prompt_occurrence",
        "nonthinking_answer_query",
    ),
    "thinking": ("thinking_item_end", "thinking_answer_query"),
}


@dataclass(frozen=True)
class GeometryDataset:
    """One endpoint's state matrix and row-level audit metadata."""

    metadata: pd.DataFrame
    states_by_layer: dict[int, np.ndarray]


@dataclass(frozen=True)
class ScatterEstimate:
    between: np.ndarray
    within: np.ndarray
    centroids: np.ndarray
    support: dict[int, int]


def _stable_key(*parts: object) -> str:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _site_candidates(
    item: V20Rendered,
    mode: str,
) -> dict[str, list[tuple[int, int]]]:
    if item.spans is None or item.count is None:
        raise ValueError("task rendering lacks span/count metadata")
    count = int(item.count)
    result: dict[str, list[tuple[int, int]]] = {}
    if mode == "nonthinking":
        result["nonthinking_prompt_occurrence"] = [
            (int(position), occurrence)
            for occurrence, position in enumerate(item.prompt_needle_positions, start=1)
        ]
        result["nonthinking_answer_query"] = [(int(item.spans.ans_pos), count)]
    elif mode == "thinking":
        result["thinking_item_end"] = [
            (int(position), occurrence)
            for occurrence, position in enumerate(item.spans.trace_marker_positions, start=1)
        ]
        result["thinking_answer_query"] = [(int(item.spans.ans_pos), count)]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return result


def _select_records(
    examples: Sequence[V20Example],
    vocab: V20Vocab,
    mode: str,
    split: str,
    per_label: int,
    random_state: int,
) -> tuple[list[V20Rendered], dict[int, list[dict[str, Any]]]]:
    rendered = [render_v20(example, vocab, mode) for example in examples]
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for example_index, (example, item) in enumerate(zip(examples, rendered, strict=True)):
        group = int(example.seed) if example.seed is not None else example.prompt_sha256
        for endpoint, positions in _site_candidates(item, mode).items():
            for position, label in positions:
                row = {
                    "endpoint": endpoint,
                    "mode": mode,
                    "split": split,
                    "example_index": int(example_index),
                    "group_id": str(group),
                    "seed": int(example.seed) if example.seed is not None else -1,
                    "occurrence": int(label),
                    "total_count": int(example.count or 0),
                    "position": int(position),
                    "prompt_sha256": example.prompt_sha256,
                    "corpus_region": example.corpus_region,
                    "corpus_start": int(example.corpus_start),
                }
                row["selection_key"] = _stable_key(
                    random_state,
                    split,
                    endpoint,
                    group,
                    label,
                    example.prompt_sha256,
                )
                candidates.setdefault((endpoint, int(label)), []).append(row)

    selected: list[dict[str, Any]] = []
    expected_labels = sorted({int(example.count or 0) for example in examples})
    for endpoint in MODE_ENDPOINTS[mode]:
        for label in expected_labels:
            bucket = sorted(
                candidates.get((endpoint, label), []),
                key=lambda row: (row["selection_key"], row["example_index"]),
            )
            if len(bucket) < per_label:
                raise ValueError(
                    f"{split}/{endpoint}/label={label} has {len(bucket)} rows; "
                    f"requires {per_label}"
                )
            selected.extend(bucket[:per_label])

    by_example: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        by_example.setdefault(int(row["example_index"]), []).append(row)
    for rows in by_example.values():
        rows.sort(key=lambda row: (row["endpoint"], row["occurrence"], row["position"]))
    return rendered, by_example


@torch.no_grad()
def capture_mode_geometry(
    model: torch.nn.Module,
    vocab: V20Vocab,
    examples: Sequence[V20Example],
    *,
    mode: str,
    split: str,
    per_label: int,
    device: str,
    batch_size: int = 32,
    random_state: int = 6201,
    head_mask: torch.Tensor | None = None,
    forward_context_factory: (
        Callable[[Sequence[V20Rendered]], ContextManager[Any]] | None
    ) = None,
) -> dict[str, GeometryDataset]:
    """Capture balanced endpoint states for one model mode and data split.

    ``head_mask`` supports global cumulative-head ablations.  The optional
    ``forward_context_factory`` receives each rendered mini-batch and can
    install query-local interventions.  Both default to the clean forward, so
    existing geometry callers retain identical behavior.
    """

    rendered, by_example = _select_records(
        examples, vocab, mode, split, per_label, random_state
    )
    endpoint_rows: dict[str, list[dict[str, Any]]] = {
        endpoint: [] for endpoint in MODE_ENDPOINTS[mode]
    }
    endpoint_states: dict[str, dict[int, list[np.ndarray]]] = {
        endpoint: {} for endpoint in MODE_ENDPOINTS[mode]
    }
    selected_indices = sorted(by_example)
    model.eval()
    for start in range(0, len(selected_indices), batch_size):
        indices = selected_indices[start : start + batch_size]
        items = [rendered[index] for index in indices]
        ids, _, mask = collate_v20(items, vocab, device)
        forward_context = (
            contextlib.nullcontext()
            if forward_context_factory is None
            else forward_context_factory(items)
        )
        with forward_context:
            output = model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                head_mask=head_mask,
            )
        hidden_states = output.hidden_states or ()
        if not hidden_states:
            raise RuntimeError("model did not return hidden states")
        for batch_row, example_index in enumerate(indices):
            for metadata in by_example[example_index]:
                endpoint = str(metadata["endpoint"])
                position = int(metadata["position"])
                clean_metadata = {
                    key: value for key, value in metadata.items() if key != "selection_key"
                }
                endpoint_rows[endpoint].append(clean_metadata)
                for layer, hidden in enumerate(hidden_states):
                    endpoint_states[endpoint].setdefault(layer, []).append(
                        hidden[batch_row, position].detach().float().cpu().numpy()
                    )

    result: dict[str, GeometryDataset] = {}
    for endpoint in MODE_ENDPOINTS[mode]:
        metadata = pd.DataFrame(endpoint_rows[endpoint])
        states = {
            layer: np.stack(values).astype(np.float32, copy=False)
            for layer, values in endpoint_states[endpoint].items()
        }
        if not states or any(len(values) != len(metadata) for values in states.values()):
            raise RuntimeError(f"misaligned state capture for {split}/{endpoint}")
        result[endpoint] = GeometryDataset(metadata=metadata, states_by_layer=states)
    return result


def combine_splits(
    discovery: GeometryDataset,
    confirmation: GeometryDataset,
) -> GeometryDataset:
    layers = sorted(set(discovery.states_by_layer) & set(confirmation.states_by_layer))
    if not layers:
        raise ValueError("discovery and confirmation captures share no layers")
    metadata = pd.concat(
        [discovery.metadata, confirmation.metadata], ignore_index=True
    )
    states = {
        layer: np.concatenate(
            [discovery.states_by_layer[layer], confirmation.states_by_layer[layer]], axis=0
        )
        for layer in layers
    }
    return GeometryDataset(metadata=metadata, states_by_layer=states)


def save_geometry_dataset(dataset: GeometryDataset, directory: Path, endpoint: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dataset.metadata.to_csv(directory / f"{endpoint}_metadata.csv", index=False)
    np.savez_compressed(
        directory / f"{endpoint}_states.npz",
        **{f"layer_{layer}": values for layer, values in dataset.states_by_layer.items()},
    )


def balanced_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
    classes: Sequence[int],
) -> float:
    recalls = []
    for label in classes:
        mask = truth == int(label)
        if not np.any(mask):
            raise ValueError(f"balanced accuracy lacks class {label}")
        recalls.append(float(np.mean(prediction[mask] == int(label))))
    return float(np.mean(recalls))


def class_balanced_scatter(
    values: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int],
) -> ScatterEstimate:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    centroids = np.stack([x[y == int(label)].mean(axis=0) for label in classes])
    grand = centroids.mean(axis=0)
    centered = centroids - grand
    between = centered.T @ centered / len(classes)
    within_terms = []
    support: dict[int, int] = {}
    for index, label in enumerate(classes):
        group = x[y == int(label)]
        if not len(group):
            raise ValueError(f"scatter lacks class {label}")
        support[int(label)] = int(len(group))
        residual = group - centroids[index]
        within_terms.append(residual.T @ residual / len(group))
    within = np.mean(np.stack(within_terms), axis=0)
    return ScatterEstimate(
        between=0.5 * (between + between.T),
        within=0.5 * (within + within.T),
        centroids=centroids,
        support=support,
    )


def regularized_precision(
    within: np.ndarray,
    relative_ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    covariance = 0.5 * (np.asarray(within, dtype=np.float64) + np.asarray(within, dtype=np.float64).T)
    dimension = covariance.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(
        np.finfo(float).eps,
        abs(float(np.trace(covariance))) / max(1, dimension),
        float(np.linalg.norm(covariance, ord="fro")) / max(1, dimension),
    )
    ridge = max(relative_ridge * scale, -float(eigenvalues[0]) + relative_ridge * scale)
    values = np.maximum(eigenvalues + ridge, np.finfo(float).eps)
    precision = (eigenvectors * (1.0 / values)[None, :]) @ eigenvectors.T
    inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(values))[None, :]) @ eigenvectors.T
    return precision, inverse_sqrt, float(ridge), float(values[-1] / values[0])


def isotropic_snr(scatter: ScatterEstimate) -> tuple[float, float]:
    signal = float(np.trace(scatter.between))
    noise = float(np.trace(scatter.within))
    if signal <= 0 or noise <= np.finfo(float).eps:
        return np.nan, np.nan
    ratio = signal / noise
    return float(ratio), float(10.0 * np.log10(ratio))


def frozen_fisher_trace(
    reference_within: np.ndarray,
    evaluation: ScatterEstimate,
) -> tuple[float, float]:
    precision, _, _, _ = regularized_precision(reference_within)
    value = max(0.0, float(np.trace(precision @ evaluation.between)))
    calibration = float(
        np.trace(precision @ evaluation.within) / evaluation.within.shape[0]
    )
    return value, calibration


def class_balanced_silhouette(
    values: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int],
) -> float:
    if len(values) <= len(np.unique(labels)):
        return float("nan")
    scores = silhouette_samples(values, labels, metric="euclidean")
    return float(np.mean([scores[labels == int(label)].mean() for label in classes]))


def ordinal_centroid_rsa(
    values: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int],
) -> float:
    centroids = np.stack([values[labels == int(label)].mean(axis=0) for label in classes])
    distances: list[float] = []
    gaps: list[int] = []
    for left in range(len(classes)):
        for right in range(left + 1, len(classes)):
            distances.append(float(np.linalg.norm(centroids[left] - centroids[right])))
            gaps.append(abs(int(classes[left]) - int(classes[right])))
    return float(spearmanr(gaps, distances).statistic)


def _projection(
    train_x: np.ndarray,
    test_x: np.ndarray,
    class_count: int,
    *,
    pca_dim: int,
    random_state: int,
    whiten: bool,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    # ``v10_port_analysis`` uses explicit ``None`` sentinels for unavailable
    # optional pandas accelerators.  scikit-learn 1.9 checks only whether the
    # pyarrow key exists, then dereferences it.  Remove only those sentinels;
    # a real imported module is left untouched.
    for optional in ("pyarrow", "numexpr", "bottleneck"):
        if sys.modules.get(optional) is None:
            sys.modules.pop(optional, None)
    scaler = StandardScaler().fit(train_x.astype(np.float32))
    train_scaled = scaler.transform(train_x.astype(np.float32))
    test_scaled = scaler.transform(test_x.astype(np.float32))
    components = min(pca_dim, train_scaled.shape[1], len(train_scaled) - class_count)
    if components < 2:
        raise ValueError("fewer than two supported PCA components")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        whiten=whiten,
        random_state=random_state,
    ).fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(test_scaled), pca


def _decoder_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    classes: Sequence[int],
    *,
    pca_dim: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_z, test_z, pca = _projection(
        train_x,
        test_x,
        len(classes),
        pca_dim=pca_dim,
        random_state=random_state,
        whiten=True,
    )
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    ).fit(train_z, train_y)
    logistic_prediction = logistic.predict(test_z)
    centroids = np.stack([train_z[train_y == int(label)].mean(axis=0) for label in classes])
    distances = np.square(test_z[:, None, :] - centroids[None, :, :]).sum(axis=-1)
    ncc_prediction = np.asarray(classes, dtype=int)[np.argmin(distances, axis=1)]
    return logistic_prediction, ncc_prediction, int(pca.n_components_)


def _valid_group_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[int],
    requested: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    expected = set(map(int, classes))
    for fold_count in range(min(requested, len(np.unique(groups))), 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=random_state,
        )
        folds = list(splitter.split(np.zeros(len(labels)), labels, groups=groups))
        if all(
            set(labels[train].tolist()) == expected and set(labels[test].tolist()) == expected
            for train, test in folds
        ):
            return folds
    raise ValueError("no grouped discovery split retains every class in every fold")


def evaluate_geometry_layer(
    states: np.ndarray,
    metadata: pd.DataFrame,
    classes: Sequence[int],
    *,
    pca_dim: int = 16,
    random_state: int = 6201,
    folds: int = 5,
) -> dict[str, Any]:
    """Evaluate one layer with discovery-only selection and frozen confirmation."""

    discovery_mask = (
        metadata["split"].astype(str).eq("discovery").to_numpy(copy=True)
    )
    confirmation_mask = (
        metadata["split"].astype(str).eq("confirmation").to_numpy(copy=True)
    )
    label_mask = metadata["occurrence"].astype(int).isin(classes).to_numpy()
    discovery_mask &= label_mask
    confirmation_mask &= label_mask
    discovery_x = states[discovery_mask]
    confirmation_x = states[confirmation_mask]
    discovery_y = metadata.loc[discovery_mask, "occurrence"].to_numpy(dtype=int)
    confirmation_y = metadata.loc[confirmation_mask, "occurrence"].to_numpy(dtype=int)
    groups = metadata.loc[discovery_mask, "group_id"].astype(str).to_numpy()
    if float(np.max(np.std(discovery_x.astype(np.float64), axis=0))) < 1e-10:
        chance = float(1.0 / len(classes))
        confirmation_support = [
            int(np.sum(confirmation_y == int(label))) for label in classes
        ]
        return {
            "discovery_oof_logistic_balanced_accuracy": chance,
            "discovery_oof_ncc_balanced_accuracy": chance,
            "discovery_selection_score": chance,
            "confirmation_logistic_balanced_accuracy": chance,
            "confirmation_ncc_balanced_accuracy": chance,
            "discovery_oof_isotropic_snr_db": np.nan,
            "confirmation_isotropic_snr_db": np.nan,
            "discovery_oof_fisher_trace": np.nan,
            "confirmation_fisher_trace_frozen": np.nan,
            "confirmation_fisher_noise_calibration": np.nan,
            "discovery_oof_mahalanobis_silhouette": np.nan,
            "confirmation_mahalanobis_silhouette": np.nan,
            "discovery_oof_ordinal_rsa": np.nan,
            "confirmation_ordinal_rsa": np.nan,
            "discovery_rows": int(len(discovery_y)),
            "confirmation_rows": int(len(confirmation_y)),
            "discovery_group_count": int(len(np.unique(groups))),
            "discovery_fold_count": 0,
            "discovery_pca_components_min": 0,
            "discovery_pca_components_max": 0,
            "confirmation_pca_components": 0,
            "confirmation_support_min": int(min(confirmation_support)),
            "confirmation_support_max": int(max(confirmation_support)),
            "discovery_covariance_ridge": np.nan,
            "discovery_covariance_condition": np.nan,
            "chance_balanced_accuracy": chance,
        }
    fold_indices = _valid_group_folds(
        discovery_y, groups, classes, folds, random_state
    )

    truths: list[np.ndarray] = []
    logistic_predictions: list[np.ndarray] = []
    ncc_predictions: list[np.ndarray] = []
    snr_values: list[float] = []
    fisher_values: list[float] = []
    silhouette_values: list[float] = []
    rsa_values: list[float] = []
    component_counts: list[int] = []
    for fold_index, (train, test) in enumerate(fold_indices):
        logistic, ncc, components = _decoder_predictions(
            discovery_x[train],
            discovery_y[train],
            discovery_x[test],
            classes,
            pca_dim=pca_dim,
            random_state=random_state + fold_index,
        )
        truths.append(discovery_y[test])
        logistic_predictions.append(logistic)
        ncc_predictions.append(ncc)
        component_counts.append(components)

        train_z, test_z, pca = _projection(
            discovery_x[train],
            discovery_x[test],
            len(classes),
            pca_dim=pca_dim,
            random_state=random_state + fold_index,
            whiten=False,
        )
        scale = np.sqrt(np.maximum(pca.explained_variance_, np.finfo(float).eps))
        _, fold_snr_db = isotropic_snr(
            class_balanced_scatter(test_z / scale, discovery_y[test], classes)
        )
        train_scatter = class_balanced_scatter(train_z, discovery_y[train], classes)
        test_scatter = class_balanced_scatter(test_z, discovery_y[test], classes)
        fold_fisher, _ = frozen_fisher_trace(train_scatter.within, test_scatter)
        _, inverse_sqrt, _, _ = regularized_precision(train_scatter.within)
        test_mahalanobis = test_z @ inverse_sqrt
        snr_values.append(fold_snr_db)
        fisher_values.append(fold_fisher)
        silhouette_values.append(
            class_balanced_silhouette(test_mahalanobis, discovery_y[test], classes)
        )
        rsa_values.append(ordinal_centroid_rsa(test_mahalanobis, discovery_y[test], classes))

    truth = np.concatenate(truths)
    logistic_oof = np.concatenate(logistic_predictions)
    ncc_oof = np.concatenate(ncc_predictions)
    discovery_logistic = balanced_accuracy(truth, logistic_oof, classes)
    discovery_ncc = balanced_accuracy(truth, ncc_oof, classes)

    logistic_confirmation, ncc_confirmation, confirmation_components = _decoder_predictions(
        discovery_x,
        discovery_y,
        confirmation_x,
        classes,
        pca_dim=pca_dim,
        random_state=random_state,
    )
    confirmation_logistic = balanced_accuracy(
        confirmation_y, logistic_confirmation, classes
    )
    confirmation_ncc = balanced_accuracy(confirmation_y, ncc_confirmation, classes)

    discovery_z, confirmation_z, pca = _projection(
        discovery_x,
        confirmation_x,
        len(classes),
        pca_dim=pca_dim,
        random_state=random_state,
        whiten=False,
    )
    scale = np.sqrt(np.maximum(pca.explained_variance_, np.finfo(float).eps))
    _, confirmation_snr_db = isotropic_snr(
        class_balanced_scatter(confirmation_z / scale, confirmation_y, classes)
    )
    discovery_scatter = class_balanced_scatter(discovery_z, discovery_y, classes)
    confirmation_scatter = class_balanced_scatter(
        confirmation_z, confirmation_y, classes
    )
    confirmation_fisher, noise_calibration = frozen_fisher_trace(
        discovery_scatter.within, confirmation_scatter
    )
    _, inverse_sqrt, ridge, condition = regularized_precision(discovery_scatter.within)
    confirmation_mahalanobis = confirmation_z @ inverse_sqrt
    confirmation_silhouette = class_balanced_silhouette(
        confirmation_mahalanobis, confirmation_y, classes
    )
    confirmation_rsa = ordinal_centroid_rsa(
        confirmation_mahalanobis, confirmation_y, classes
    )

    return {
        "discovery_oof_logistic_balanced_accuracy": discovery_logistic,
        "discovery_oof_ncc_balanced_accuracy": discovery_ncc,
        "discovery_selection_score": 0.5 * (discovery_logistic + discovery_ncc),
        "confirmation_logistic_balanced_accuracy": confirmation_logistic,
        "confirmation_ncc_balanced_accuracy": confirmation_ncc,
        "discovery_oof_isotropic_snr_db": float(np.mean(snr_values)),
        "confirmation_isotropic_snr_db": confirmation_snr_db,
        "discovery_oof_fisher_trace": float(np.mean(fisher_values)),
        "confirmation_fisher_trace_frozen": confirmation_fisher,
        "confirmation_fisher_noise_calibration": noise_calibration,
        "discovery_oof_mahalanobis_silhouette": float(np.mean(silhouette_values)),
        "confirmation_mahalanobis_silhouette": confirmation_silhouette,
        "discovery_oof_ordinal_rsa": float(np.mean(rsa_values)),
        "confirmation_ordinal_rsa": confirmation_rsa,
        "discovery_rows": int(len(discovery_y)),
        "confirmation_rows": int(len(confirmation_y)),
        "discovery_group_count": int(len(np.unique(groups))),
        "discovery_fold_count": int(len(fold_indices)),
        "discovery_pca_components_min": int(min(component_counts)),
        "discovery_pca_components_max": int(max(component_counts)),
        "confirmation_pca_components": confirmation_components,
        "confirmation_support_min": int(min(confirmation_scatter.support.values())),
        "confirmation_support_max": int(max(confirmation_scatter.support.values())),
        "discovery_covariance_ridge": ridge,
        "discovery_covariance_condition": condition,
        "chance_balanced_accuracy": float(1.0 / len(classes)),
    }


METRIC_SELECTION = {
    "logistic_balanced_accuracy": (
        "discovery_oof_logistic_balanced_accuracy",
        "confirmation_logistic_balanced_accuracy",
    ),
    "ncc_balanced_accuracy": (
        "discovery_oof_ncc_balanced_accuracy",
        "confirmation_ncc_balanced_accuracy",
    ),
    "isotropic_snr_db": (
        "discovery_oof_isotropic_snr_db",
        "confirmation_isotropic_snr_db",
    ),
    "fisher_trace": (
        "discovery_oof_fisher_trace",
        "confirmation_fisher_trace_frozen",
    ),
    "mahalanobis_silhouette": (
        "discovery_oof_mahalanobis_silhouette",
        "confirmation_mahalanobis_silhouette",
    ),
    "ordinal_rsa": ("discovery_oof_ordinal_rsa", "confirmation_ordinal_rsa"),
}


def evaluate_geometry_dataset(
    dataset: GeometryDataset,
    *,
    endpoint: str,
    classes: Sequence[int] = tuple(range(1, 31)),
    pca_dim: int = 16,
    random_state: int = 6201,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rows = []
    for layer, states in sorted(dataset.states_by_layer.items()):
        rows.append(
            {
                "endpoint": endpoint,
                "endpoint_label": ENDPOINT_LABELS[endpoint],
                "mode": str(dataset.metadata["mode"].iloc[0]),
                "layer": int(layer),
                **evaluate_geometry_layer(
                    states,
                    dataset.metadata,
                    classes,
                    pca_dim=pca_dim,
                    random_state=random_state,
                ),
            }
        )
    per_layer = pd.DataFrame(rows)
    common = per_layer.sort_values(
        ["discovery_selection_score", "layer"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    selected_layer = int(common["layer"])
    selections = []
    for metric, (discovery_column, confirmation_column) in METRIC_SELECTION.items():
        winner = per_layer.sort_values(
            [discovery_column, "layer"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        selections.append(
            {
                "endpoint": endpoint,
                "endpoint_label": ENDPOINT_LABELS[endpoint],
                "mode": str(winner["mode"]),
                "selector": metric,
                "discovery_metric": discovery_column,
                "confirmation_metric": confirmation_column,
                "selected_layer": int(winner["layer"]),
                "discovery_value": float(winner[discovery_column]),
                "confirmation_value": float(winner[confirmation_column]),
                "common_decoder_selected_layer": selected_layer,
            }
        )
    return per_layer, pd.DataFrame(selections), selected_layer


def confirmation_pca_coordinates(
    dataset: GeometryDataset,
    layer: int,
    *,
    components: int = 3,
    random_state: int = 6201,
) -> pd.DataFrame:
    discovery = dataset.metadata["split"].astype(str).eq("discovery").to_numpy()
    confirmation = dataset.metadata["split"].astype(str).eq("confirmation").to_numpy()
    scaler = StandardScaler().fit(dataset.states_by_layer[layer][discovery])
    discovery_scaled = scaler.transform(dataset.states_by_layer[layer][discovery])
    confirmation_scaled = scaler.transform(dataset.states_by_layer[layer][confirmation])
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(discovery_scaled)
    coordinates = pca.transform(confirmation_scaled)
    frame = dataset.metadata.loc[confirmation].reset_index(drop=True).copy()
    for index in range(components):
        frame[f"pc{index + 1}"] = coordinates[:, index]
        frame[f"pc{index + 1}_variance_ratio"] = float(
            pca.explained_variance_ratio_[index]
        )
    frame["selected_layer"] = int(layer)
    return frame


def write_protocol_manifest(path: Path, extra: dict[str, Any] | None = None) -> None:
    value = {
        "schema_version": "v20_aligned_geometry_v1",
        "endpoints": ENDPOINT_LABELS,
        "selection": (
            "StandardScaler + whitened PCA(<=16) and grouped 5-fold discovery CV; "
            "common layer maximizes mean(logistic BA, nearest-centroid BA); "
            "confirmation never selects a layer"
        ),
        "covariance_geometry": {
            "isotropic_snr_db": "10 log10(tr(Sigma_B)/tr(Sigma_W)) in discovery-fitted PCA-whitened space",
            "fisher_trace": "tr((Sigma_W,discovery + ridge I)^-1 Sigma_B,confirmation)",
            "mahalanobis_silhouette": "class-balanced silhouette after discovery within-covariance whitening",
            "ordinal_rsa": "Spearman rho between centroid distance and absolute count gap",
        },
        "claim_boundary": (
            "Residual-state decodability/geometry localizes representations; it does "
            "not establish causal use without the separate intervention suite."
        ),
    }
    if extra:
        value.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ENDPOINT_LABELS",
    "GeometryDataset",
    "capture_mode_geometry",
    "combine_splits",
    "confirmation_pca_coordinates",
    "evaluate_geometry_dataset",
    "evaluate_geometry_layer",
    "save_geometry_dataset",
    "write_protocol_manifest",
]
