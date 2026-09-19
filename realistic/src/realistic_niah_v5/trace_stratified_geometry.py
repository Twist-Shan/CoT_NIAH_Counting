"""Marker-stratified native-thinking token-site geometry.

This module treats parser ``marker_kind`` as a surface-form stratum and keeps
``trace_category`` descriptive.  Within each selector, candidate token sites
and decoder layers are ranked jointly and independently of every other
selector, using only leave-one-discovery-seed-out classification.  There is no
shared-layer constraint.  Confirmation metrics are retained as a
site-sensitivity audit but never enter the programmed ranking; downstream
reporting foregrounds only discovery-frozen winners.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from realistic_niah_v5.cross_mode_geometry import (
    CLASSES,
    ModeDataset,
    load_native_thinking_capture,
)


SCHEMA_VERSION = "realistic_niah_trace_stratified_site_geometry_v2"
LAYER_COMPARISON_POLICY = (
    "independent token-site x layer selection within each selector; "
    "no same-layer matching across selectors or trace strata"
)
SITE_ROLES = {
    "pre_marker": "immediately_before_explicit_count_marker",
    "marker_end": "explicit_or_invariant_marker_endpoint",
    "pre_city": "immediately_before_city_entity",
    "city_end": "entity_endpoint",
    "city_unit_end": "city_sentence_or_line_endpoint",
    "item_end": "completed_item_endpoint",
    "post_boundary": "after_item_boundary",
}
SITE_CANDIDATES_BY_MARKER_KIND = {
    "indexed": (
        "pre_marker",
        "marker_end",
        "pre_city",
        "city_end",
        "city_unit_end",
        "item_end",
        "post_boundary",
    ),
    "ordinal": (
        "pre_marker",
        "marker_end",
        "pre_city",
        "city_end",
        "city_unit_end",
        "item_end",
        "post_boundary",
    ),
    # marker_end is retained as an invariant-marker negative control.  It is
    # excluded from the post-marker selector below.
    "bullet": (
        "pre_marker",
        "marker_end",
        "pre_city",
        "city_end",
        "city_unit_end",
        "item_end",
        "post_boundary",
    ),
    "inline_count": (
        "pre_marker",
        "marker_end",
        "pre_city",
        "city_end",
        "city_unit_end",
        "item_end",
        "post_boundary",
    ),
    "audit_sentence": (
        "pre_city", "city_end", "city_unit_end", "item_end", "post_boundary"
    ),
    "completion_recap": (
        "pre_city", "city_end", "city_unit_end", "item_end", "post_boundary"
    ),
    "evidence_sequence": (
        "pre_city", "city_end", "city_unit_end", "item_end", "post_boundary"
    ),
}
SELECTOR_SITE_FAMILIES = {
    "fixed_item_end": ("item_end",),
    "post_marker_site_search": (
        "city_end", "city_unit_end", "item_end", "post_boundary"
    ),
    "pre_label_site_search": ("pre_marker",),
    "low_leakage_site_search": (
        "pre_marker",
        "pre_city",
        "city_end",
        "city_unit_end",
        "item_end",
        "post_boundary",
    ),
    "all_site_search": (
        "pre_marker",
        "marker_end",
        "pre_city",
        "city_end",
        "city_unit_end",
        "item_end",
        "post_boundary",
    ),
}


@dataclass(frozen=True)
class StratumEligibility:
    marker_kind: str
    status: str
    labels: tuple[int, ...]
    discovery_seed_count: int
    confirmation_seed_count: int
    discovery_support: dict[int, int]
    confirmation_support: dict[int, int]
    reason: str


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


def _balanced_accuracy(
    truth: np.ndarray, prediction: np.ndarray, labels: Sequence[int]
) -> float:
    recalls = []
    for label in labels:
        mask = truth == int(label)
        if not np.any(mask):
            raise ValueError(f"Balanced accuracy lacks label {label}")
        recalls.append(float(np.mean(prediction[mask] == int(label))))
    return float(np.mean(recalls))


def _site_role(marker_kind: str, site_kind: str) -> str:
    if site_kind != "marker_end":
        return SITE_ROLES[site_kind]
    if marker_kind in {"indexed", "ordinal", "inline_count"}:
        return "explicit_count_cue_endpoint"
    if marker_kind == "bullet":
        return "invariant_marker_negative_control"
    raise ValueError(f"{marker_kind} has no registered marker_end site")


def determine_stratum_eligibility(
    metadata: pd.DataFrame,
    marker_kind: str,
    *,
    claim_min_discovery: int = 3,
    claim_min_confirmation: int = 2,
    exploratory_min_discovery: int = 2,
    exploratory_min_confirmation: int = 1,
    min_labels: int = 5,
) -> StratumEligibility:
    frame = metadata.loc[metadata["marker_kind"].astype(str).eq(marker_kind)].copy()
    discovery = frame.loc[frame["split"].astype(str).eq("discovery")]
    confirmation = frame.loc[frame["split"].astype(str).eq("confirmation")]
    discovery_support = {
        label: int((discovery["occurrence"].astype(int) == label).sum())
        for label in CLASSES
    }
    confirmation_support = {
        label: int((confirmation["occurrence"].astype(int) == label).sum())
        for label in CLASSES
    }
    discovery_seed_count = int(discovery["seed"].nunique())
    confirmation_seed_count = int(confirmation["seed"].nunique())
    claim_labels = tuple(
        label
        for label in CLASSES
        if discovery_support[label] >= claim_min_discovery
        and confirmation_support[label] >= claim_min_confirmation
    )
    exploratory_labels = tuple(
        label
        for label in CLASSES
        if discovery_support[label] >= exploratory_min_discovery
        and confirmation_support[label] >= exploratory_min_confirmation
    )
    if (
        discovery_seed_count >= claim_min_discovery
        and confirmation_seed_count >= claim_min_confirmation
        and len(claim_labels) >= min_labels
    ):
        status = "claim_grade"
        labels = claim_labels
        reason = (
            "at least 3 discovery and 2 confirmation seeds; every retained "
            "position has discovery n>=3 and confirmation n>=2"
        )
    elif (
        discovery_seed_count >= exploratory_min_discovery
        and confirmation_seed_count >= exploratory_min_confirmation
        and len(exploratory_labels) >= min_labels
    ):
        status = "exploratory_only"
        labels = exploratory_labels
        reason = (
            "confirmation has only one seed or some retained positions have "
            "confirmation n=1; point estimates are not claim-grade"
        )
    else:
        status = "not_evaluable"
        labels = exploratory_labels
        reason = (
            "insufficient discovery/confirmation seed support or fewer than "
            f"{min_labels} shared position labels"
        )
    return StratumEligibility(
        marker_kind=marker_kind,
        status=status,
        labels=labels,
        discovery_seed_count=discovery_seed_count,
        confirmation_seed_count=confirmation_seed_count,
        discovery_support=discovery_support,
        confirmation_support=confirmation_support,
        reason=reason,
    )


def _fit_projection_and_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    labels: Sequence[int],
    *,
    pca_dim: int,
    random_state: int,
    pca_whiten: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    label_array = np.asarray(labels, dtype=int)
    if set(train_y.tolist()) != set(label_array.tolist()):
        raise ValueError("Training fold does not cover every retained label")
    scaler = StandardScaler().fit(train_x.astype(np.float32))
    train_scaled = scaler.transform(train_x.astype(np.float32))
    test_scaled = scaler.transform(test_x.astype(np.float32))
    components = min(
        int(pca_dim),
        int(train_scaled.shape[0] - len(label_array)),
        int(train_scaled.shape[1]),
    )
    if components < 2:
        raise ValueError("Training fold supports fewer than two PCA components")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        whiten=pca_whiten,
        random_state=random_state,
    ).fit(train_scaled)
    train_projected = pca.transform(train_scaled)
    test_projected = pca.transform(test_scaled)
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    ).fit(train_projected, train_y)
    logistic_prediction = logistic.predict(test_projected)
    centroids = np.stack(
        [train_projected[train_y == label].mean(axis=0) for label in label_array]
    )
    distances = np.square(
        test_projected[:, None, :] - centroids[None, :, :]
    ).sum(axis=-1)
    ncc_prediction = label_array[np.argmin(distances, axis=1)]
    return logistic_prediction, ncc_prediction, test_projected, components


def grouped_discovery_cv_metrics(
    states: np.ndarray,
    metadata: pd.DataFrame,
    labels: Sequence[int],
    *,
    pca_dim: int = 16,
    random_state: int = 0,
    folds: int | None = None,
    pca_whiten: bool = False,
) -> dict[str, Any]:
    discovery_mask = metadata["split"].astype(str).eq("discovery").to_numpy()
    label_mask = metadata["occurrence"].astype(int).isin(labels).to_numpy()
    selected = discovery_mask & label_mask
    x = states[selected].astype(np.float32)
    frame = metadata.loc[selected].reset_index(drop=True)
    y = frame["occurrence"].to_numpy(dtype=int)
    seeds = frame["seed"].to_numpy(dtype=int)
    logistic_truth: list[np.ndarray] = []
    logistic_predictions: list[np.ndarray] = []
    ncc_predictions: list[np.ndarray] = []
    fold_components: list[int] = []
    unique_seeds = sorted(set(seeds.tolist()))
    if len(unique_seeds) < 2:
        raise ValueError("Discovery grouped CV requires at least two seeds")
    if folds is None:
        fold_indices = [
            (np.flatnonzero(seeds != held_seed), np.flatnonzero(seeds == held_seed))
            for held_seed in unique_seeds
        ]
    else:
        fold_count = min(int(folds), len(unique_seeds))
        if fold_count < 2:
            raise ValueError("Discovery grouped CV requires at least two folds")
        splitter = GroupKFold(n_splits=fold_count)
        fold_indices = list(splitter.split(x, y, groups=seeds))
    for train_index, test_index in fold_indices:
        train = np.zeros(len(y), dtype=bool)
        test = np.zeros(len(y), dtype=bool)
        train[train_index] = True
        test[test_index] = True
        if set(y[train].tolist()) != set(map(int, labels)):
            held_seeds = sorted(set(seeds[test].tolist()))
            raise ValueError(
                f"Holding out discovery seeds {held_seeds} removes a retained label"
            )
        logistic, ncc, _projected, components = _fit_projection_and_predict(
            x[train],
            y[train],
            x[test],
            labels,
            pca_dim=pca_dim,
            random_state=random_state,
            pca_whiten=pca_whiten,
        )
        logistic_truth.append(y[test])
        logistic_predictions.append(logistic)
        ncc_predictions.append(ncc)
        fold_components.append(components)
    truth = np.concatenate(logistic_truth)
    logistic = np.concatenate(logistic_predictions)
    ncc = np.concatenate(ncc_predictions)
    logistic_ba = _balanced_accuracy(truth, logistic, labels)
    ncc_ba = _balanced_accuracy(truth, ncc, labels)
    return {
        "discovery_oof_logistic_balanced_accuracy": logistic_ba,
        "discovery_oof_ncc_balanced_accuracy": ncc_ba,
        "discovery_selection_score": 0.5 * (logistic_ba + ncc_ba),
        "discovery_oof_rows": int(len(truth)),
        "discovery_fold_count": int(len(fold_components)),
        "discovery_pca_components_min": int(min(fold_components)),
        "discovery_pca_components_max": int(max(fold_components)),
    }


def confirmation_metrics(
    states: np.ndarray,
    metadata: pd.DataFrame,
    labels: Sequence[int],
    *,
    pca_dim: int = 16,
    random_state: int = 0,
    pca_whiten: bool = False,
) -> dict[str, Any]:
    label_mask = metadata["occurrence"].astype(int).isin(labels).to_numpy()
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy() & label_mask
    confirmation = (
        metadata["split"].astype(str).eq("confirmation").to_numpy() & label_mask
    )
    train_y = metadata.loc[discovery, "occurrence"].to_numpy(dtype=int)
    confirmation_y = metadata.loc[confirmation, "occurrence"].to_numpy(dtype=int)
    logistic, ncc, projected, components = _fit_projection_and_predict(
        states[discovery],
        train_y,
        states[confirmation],
        labels,
        pca_dim=pca_dim,
        random_state=random_state,
        pca_whiten=pca_whiten,
    )
    class_means = np.stack(
        [projected[confirmation_y == label].mean(axis=0) for label in labels]
    )
    balanced_grand = class_means.mean(axis=0)
    signal_power = float(
        np.square(class_means - balanced_grand).sum(axis=1).mean()
    )
    class_noise = np.asarray(
        [
            np.square(
                projected[confirmation_y == label]
                - class_means[label_index]
            )
            .sum(axis=1)
            .mean()
            for label_index, label in enumerate(labels)
        ],
        dtype=float,
    )
    noise_power = float(class_noise.mean())
    if noise_power <= np.finfo(float).eps:
        snr = np.nan
        snr_db = np.nan
    else:
        snr = float(signal_power / noise_power)
        snr_db = float(10.0 * np.log10(snr))
    support = {
        int(label): int(np.sum(confirmation_y == label)) for label in labels
    }
    return {
        "confirmation_logistic_balanced_accuracy": _balanced_accuracy(
            confirmation_y, logistic, labels
        ),
        "confirmation_ncc_balanced_accuracy": _balanced_accuracy(
            confirmation_y, ncc, labels
        ),
        "confirmation_class_balanced_snr": snr,
        "confirmation_class_balanced_snr_db": snr_db,
        "confirmation_signal_power": signal_power,
        "confirmation_noise_power": noise_power,
        "confirmation_rows": int(len(confirmation_y)),
        "confirmation_seed_count": int(
            metadata.loc[confirmation, "seed"].nunique()
        ),
        "confirmation_support_min": int(min(support.values())),
        "confirmation_support_max": int(max(support.values())),
        "confirmation_pca_components": int(components),
        "chance_balanced_accuracy": float(1.0 / len(labels)),
    }


def _subset_stratum(
    dataset: ModeDataset, marker_kind: str, labels: Sequence[int]
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    mask = (
        dataset.metadata["marker_kind"].astype(str).eq(marker_kind)
        & dataset.metadata["occurrence"].astype(int).isin(labels)
    ).to_numpy()
    metadata = dataset.metadata.loc[mask].reset_index(drop=True)
    states = {layer: values[mask] for layer, values in dataset.states_by_layer.items()}
    return metadata, states


def _eligibility_rows(
    model_label: str, eligibility: Iterable[StratumEligibility]
) -> list[dict[str, Any]]:
    rows = []
    for item in eligibility:
        rows.append(
            {
                "model_label": model_label,
                "marker_kind": item.marker_kind,
                "eligibility": item.status,
                "retained_labels": " ".join(map(str, item.labels)),
                "retained_class_count": len(item.labels),
                "discovery_seed_count": item.discovery_seed_count,
                "confirmation_seed_count": item.confirmation_seed_count,
                "discovery_support": json.dumps(item.discovery_support, sort_keys=True),
                "confirmation_support": json.dumps(
                    item.confirmation_support, sort_keys=True
                ),
                "reason": item.reason,
            }
        )
    return rows


def _select_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    selections = []
    for (model, marker_kind), frame in metrics.groupby(
        ["model_label", "marker_kind"], sort=False
    ):
        available_sites = set(frame["site_kind"].astype(str))
        for selector, registered_sites in SELECTOR_SITE_FAMILIES.items():
            candidates = sorted(available_sites & set(registered_sites))
            if not candidates:
                continue
            selected = frame.loc[frame["site_kind"].isin(candidates)].sort_values(
                [
                    "discovery_selection_score",
                    "discovery_oof_ncc_balanced_accuracy",
                    "discovery_oof_logistic_balanced_accuracy",
                    "layer",
                    "site_kind",
                ],
                ascending=[False, False, False, True, True],
                kind="mergesort",
            ).iloc[0]
            row = selected.to_dict()
            row["selector"] = selector
            row["selector_candidate_sites"] = " ".join(candidates)
            selections.append(row)
    return pd.DataFrame(selections)


def analyze_trace_stratified_geometry(
    capture_index: str | Path,
    output_dir: str | Path,
    *,
    pca_dim: int = 16,
    layers: Sequence[int] | None = None,
    random_state: int = 0,
) -> dict[str, Path]:
    index_path = Path(capture_index)
    item_dataset = load_native_thinking_capture(
        index_path, site_kind="item_end", cohort="parser_hit"
    )
    marker_kinds = [
        marker
        for marker in SITE_CANDIDATES_BY_MARKER_KIND
        if marker in set(item_dataset.metadata["marker_kind"].astype(str))
    ]
    eligibility = [
        determine_stratum_eligibility(item_dataset.metadata, marker)
        for marker in marker_kinds
    ]
    eligibility_by_marker = {item.marker_kind: item for item in eligibility}
    selected_layers = (
        sorted(item_dataset.states_by_layer)
        if layers is None
        else sorted(set(map(int, layers)))
    )
    missing_layers = sorted(set(selected_layers) - set(item_dataset.states_by_layer))
    if missing_layers:
        raise ValueError(f"Unavailable layers: {missing_layers}")
    metric_rows: list[dict[str, Any]] = []
    for site_kind in SITE_ROLES:
        relevant_markers = [
            marker
            for marker in marker_kinds
            if site_kind in SITE_CANDIDATES_BY_MARKER_KIND[marker]
            and eligibility_by_marker[marker].status != "not_evaluable"
        ]
        if not relevant_markers:
            continue
        dataset = load_native_thinking_capture(
            index_path, site_kind=site_kind, cohort="parser_hit"
        )
        for marker_kind in relevant_markers:
            item = eligibility_by_marker[marker_kind]
            metadata, states_by_layer = _subset_stratum(
                dataset, marker_kind, item.labels
            )
            if metadata.empty:
                raise ValueError(f"No {marker_kind}/{site_kind} states")
            for layer in selected_layers:
                states = states_by_layer[layer]
                discovery_metrics = grouped_discovery_cv_metrics(
                    states,
                    metadata,
                    item.labels,
                    pca_dim=pca_dim,
                    random_state=random_state,
                )
                heldout_metrics = confirmation_metrics(
                    states,
                    metadata,
                    item.labels,
                    pca_dim=pca_dim,
                    random_state=random_state,
                )
                metric_rows.append(
                    {
                        "model_label": dataset.model_label,
                        "marker_kind": marker_kind,
                        "eligibility": item.status,
                        "retained_labels": " ".join(map(str, item.labels)),
                        "retained_class_count": len(item.labels),
                        "site_kind": site_kind,
                        "site_role": _site_role(marker_kind, site_kind),
                        "layer": int(layer),
                        **discovery_metrics,
                        **heldout_metrics,
                    }
                )
        del dataset
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise ValueError("No evaluable trace strata")
    selections = _select_rows(metrics)
    output = Path(output_dir)
    paths = {
        "eligibility": output / "trace_stratum_eligibility.csv",
        "metrics": output / "trace_stratum_site_layer_metrics.csv",
        "selection": output / "trace_stratum_discovery_selected_sites.csv",
        "audit": output / "trace_stratum_site_sweep_audit.json",
    }
    _atomic_csv(
        paths["eligibility"],
        pd.DataFrame(_eligibility_rows(item_dataset.model_label, eligibility)),
    )
    _atomic_csv(paths["metrics"], metrics)
    _atomic_csv(paths["selection"], selections)
    _atomic_json(
        paths["audit"],
        {
            "schema_version": SCHEMA_VERSION,
            "model_label": item_dataset.model_label,
            "capture_index": str(index_path.resolve()),
            "stratification_variable": "parser marker_kind",
            "trace_category_role": "descriptive only; never selects a token site",
            "candidate_sites_by_marker_kind": {
                key: list(value)
                for key, value in SITE_CANDIDATES_BY_MARKER_KIND.items()
            },
            "selector_site_families": {
                key: list(value) for key, value in SELECTOR_SITE_FAMILIES.items()
            },
            "selection_rule": (
                "within each selector, independently maximize over its candidate "
                "token sites and all registered layers the mean(discovery "
                "leave-one-seed-out logistic balanced accuracy, discovery "
                "leave-one-seed-out nearest-centroid balanced accuracy); tie-break "
                "by NCC, logistic, earlier layer, site name"
            ),
            "layer_comparison_policy": LAYER_COMPARISON_POLICY,
            "confirmation_role": (
                "computed for the site-sensitivity audit but never read by the "
                "programmed site/layer selector; downstream reporting foregrounds "
                "only discovery-frozen winners"
            ),
            "pca_dim_requested": int(pca_dim),
            "preprocessing": (
                "StandardScaler and PCA refit inside each discovery CV fold; final "
                "confirmation transform fit on all discovery rows"
            ),
            "snr_definition": (
                "confirmation-only class-balanced centroid signal power divided by "
                "class-balanced within-class residual power in the frozen discovery "
                "projection"
            ),
            "layers": selected_layers,
            "eligibility": [
                {
                    "marker_kind": item.marker_kind,
                    "status": item.status,
                    "labels": list(item.labels),
                    "discovery_seed_count": item.discovery_seed_count,
                    "confirmation_seed_count": item.confirmation_seed_count,
                    "reason": item.reason,
                }
                for item in eligibility
            ],
        },
    )
    return paths
