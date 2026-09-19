"""Answer-endpoint geometry for the city/flower/animal transfer panel.

The original twenty V4.4 city discovery seeds select a layer and fit the count
probe independently for each model and mode.  The frozen pipeline is evaluated
on all ten confirmation seeds for city, flower, and animal (100 trajectories
per domain).  The 3-D display basis is likewise fitted on city discovery and is
then applied unchanged to the complete 300-row confirmation panel.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from realistic_niah_v5.dual_endpoint_geometry import (
    load_native_thinking_final_count,
    load_non_thinking_final_count,
)


SCHEMA_VERSION = "realistic_niah_domain_transfer_geometry_v2_city_discovery"
DOMAINS = ("city", "flower", "animal")
COUNTS = tuple(range(1, 11))
DISCOVERY_SEEDS = tuple(range(1234, 1254))
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
# Backward-compatible names retained for callers; selection/evaluation now use
# the original city discovery/confirmation split rather than a 5/5 split.
SELECTION_SEEDS = DISCOVERY_SEEDS
EVALUATION_SEEDS = CONFIRMATION_SEEDS
DEFAULT_PCA_DIMENSIONS = (1, 2, 4, 8, 16, 32)


@dataclass
class DomainEndpointDataset:
    mode: str
    model_label: str
    metadata: pd.DataFrame
    states_by_layer: dict[int, np.ndarray]

    def validate(self, *, require_complete: bool = False) -> None:
        required = {
            "entity_domain",
            "seed",
            "gold_count",
            "stimulus_id",
            "source_stimulus_id",
        }
        missing = sorted(required - set(self.metadata.columns))
        if missing:
            raise ValueError(f"{self.mode} metadata is missing {missing}")
        lengths = {len(value) for value in self.states_by_layer.values()}
        if lengths != {len(self.metadata)}:
            raise ValueError(
                f"{self.mode} states/metadata mismatch: {lengths}/{len(self.metadata)}"
            )
        keys = self.metadata[["entity_domain", "seed", "gold_count"]]
        if keys.duplicated().any():
            duplicates = keys[keys.duplicated(keep=False)].head().to_dict("records")
            raise ValueError(f"Duplicate domain/seed/count cells: {duplicates}")
        if not all(np.isfinite(value).all() for value in self.states_by_layer.values()):
            raise ValueError(f"{self.mode} contains non-finite hidden states")
        if not require_complete:
            return
        expected = {
            (domain, seed, count)
            for domain in DOMAINS
            for seed in CONFIRMATION_SEEDS
            for count in COUNTS
        }
        observed = {
            (str(row.entity_domain), int(row.seed), int(row.gold_count))
            for row in self.metadata.itertuples(index=False)
        }
        if observed != expected or len(self.metadata) != len(expected):
            missing_cells = sorted(expected - observed)
            extra_cells = sorted(observed - expected)
            raise ValueError(
                "Domain panel is not the registered 3 x 10 x 10 grid: "
                f"rows={len(self.metadata)} missing={missing_cells[:5]} "
                f"extra={extra_cells[:5]}"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _site_index(manifest: Mapping[str, Any], expected_kind: str) -> int:
    matching = [
        index
        for index, site in enumerate(manifest.get("site_rows", []))
        if str(site.get("site_kind")) == expected_kind
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one {expected_kind} site for "
            f"{manifest.get('stimulus_id')}; found {len(matching)}"
        )
    return matching[0]


def load_transfer_answer_endpoints(
    capture_index: str | Path,
    *,
    mode: str,
) -> DomainEndpointDataset:
    """Load flower/animal answer-query states from a domain-transfer capture."""

    if mode not in {"non_thinking", "native_thinking"}:
        raise ValueError(f"Unsupported mode: {mode}")
    expected_kind = "answer_query" if mode == "non_thinking" else "answer_query_v3"
    index_path = Path(capture_index)
    descriptors: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for row in _read_jsonl(index_path):
        manifest_path = index_path.parent / str(row["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        descriptors.append((row, manifest, _site_index(manifest, expected_kind)))
    descriptors.sort(
        key=lambda item: (
            DOMAINS.index(str(item[0]["entity_domain"])),
            int(item[0]["seed"]),
            int(item[0]["gold_count"]),
        )
    )
    if not descriptors:
        raise ValueError(f"No capture rows in {index_path}")
    first_row, first_manifest, _ = descriptors[0]
    first_states_path = index_path.parent / str(first_row["states_path"])
    with np.load(first_states_path, allow_pickle=False) as archive:
        layer_indices = archive["layer_indices"].astype(int)
        first_states = np.asarray(archive["site_states"])
        hidden_size = int(first_states.shape[-1])
    states = {
        int(layer): np.empty((len(descriptors), hidden_size), dtype=np.float16)
        for layer in layer_indices
    }
    metadata: list[dict[str, Any]] = []
    for row_axis, (row, manifest, site_axis) in enumerate(descriptors):
        states_path = index_path.parent / str(row["states_path"])
        with np.load(states_path, allow_pickle=False) as archive:
            layers = archive["layer_indices"].astype(int)
            values = np.asarray(archive["site_states"])
        if not np.array_equal(layers, layer_indices):
            raise ValueError(f"Layer mismatch in {states_path}")
        if values.ndim != 3 or values.shape[1:] != (len(layer_indices), hidden_size):
            raise ValueError(f"Unexpected site_states shape {values.shape} in {states_path}")
        if site_axis >= values.shape[0]:
            raise ValueError(f"Site axis {site_axis} is absent in {states_path}")
        for layer_axis, layer in enumerate(layer_indices):
            states[int(layer)][row_axis] = values[site_axis, layer_axis]
        metadata.append(
            {
                "entity_domain": str(row["entity_domain"]),
                "seed": int(row["seed"]),
                "gold_count": int(row["gold_count"]),
                "split": str(row.get("split", "confirmation")),
                "stimulus_id": str(row["stimulus_id"]),
                "source_stimulus_id": str(row["source_stimulus_id"]),
                "answer_site_kind": expected_kind,
                "exact_count": row.get("exact_count"),
                "trace_category": row.get("trace_category"),
                "marker_kind": row.get("marker_kind"),
                "generation_truncated": row.get("generation_truncated"),
                "generation_rescue": row.get("generation_rescue"),
                "running_site_count": int(row.get("running_site_count", 0)),
                "answer_site_count": int(row.get("answer_site_count", 0)),
                "states_path": str(states_path.resolve()),
                "manifest_path": str(
                    (index_path.parent / str(row["manifest_path"])).resolve()
                ),
            }
        )
    dataset = DomainEndpointDataset(
        mode=mode,
        model_label=str(first_manifest["model_label"]),
        metadata=pd.DataFrame(metadata),
        states_by_layer=states,
    )
    dataset.validate()
    return dataset


def load_city_answer_endpoints(
    capture_index: str | Path,
    *,
    mode: str,
    seeds: Sequence[int] = DISCOVERY_SEEDS + CONFIRMATION_SEEDS,
) -> DomainEndpointDataset:
    """Load the matching V4.4 city answer-query panel."""

    if mode == "non_thinking":
        source = load_non_thinking_final_count(capture_index)
    elif mode == "native_thinking":
        source = load_native_thinking_final_count(capture_index)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    retained = source.metadata["seed"].astype(int).isin(set(map(int, seeds))).to_numpy()
    metadata = source.metadata.loc[retained].copy().reset_index(drop=True)
    metadata["entity_domain"] = "city"
    metadata["source_stimulus_id"] = metadata["stimulus_id"].astype(str)
    metadata["answer_site_kind"] = metadata["token_site"].astype(str)
    metadata["running_site_count"] = 0
    metadata["answer_site_count"] = 1
    dataset = DomainEndpointDataset(
        mode=mode,
        model_label=source.model_label,
        metadata=metadata,
        states_by_layer={
            int(layer): np.asarray(values[retained])
            for layer, values in source.states_by_layer.items()
        },
    )
    dataset.validate()
    return dataset


def subset_by_seeds(
    dataset: DomainEndpointDataset,
    seeds: Sequence[int],
) -> DomainEndpointDataset:
    retained = dataset.metadata["seed"].astype(int).isin(set(map(int, seeds))).to_numpy()
    result = DomainEndpointDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=dataset.metadata.loc[retained].reset_index(drop=True),
        states_by_layer={
            int(layer): np.asarray(values[retained])
            for layer, values in dataset.states_by_layer.items()
        },
    )
    result.validate()
    return result


def combine_city_and_transfer(
    city: DomainEndpointDataset,
    transfer: DomainEndpointDataset,
) -> DomainEndpointDataset:
    if city.mode != transfer.mode or city.model_label != transfer.model_label:
        raise ValueError("City and transfer datasets have different model/mode labels")
    if set(city.states_by_layer) != set(transfer.states_by_layer):
        raise ValueError("City and transfer captures expose different layer sets")
    frames = [city.metadata.copy(), transfer.metadata.copy()]
    metadata = pd.concat(frames, ignore_index=True)
    metadata["_domain_order"] = metadata["entity_domain"].map(
        {domain: index for index, domain in enumerate(DOMAINS)}
    )
    if metadata["_domain_order"].isna().any():
        raise ValueError("Unknown entity domain in combined panel")
    metadata["_old_index"] = np.arange(len(metadata))
    metadata = metadata.sort_values(
        ["_domain_order", "seed", "gold_count"], kind="mergesort"
    )
    old_index = metadata.pop("_old_index").to_numpy(dtype=int)
    metadata = metadata.drop(columns="_domain_order").reset_index(drop=True)
    states: dict[int, np.ndarray] = {}
    for layer in sorted(city.states_by_layer):
        city_values = np.asarray(city.states_by_layer[layer])
        transfer_values = np.asarray(transfer.states_by_layer[layer])
        if city_values.shape[1:] != transfer_values.shape[1:]:
            raise ValueError(f"Hidden-size mismatch at layer {layer}")
        states[layer] = np.concatenate([city_values, transfer_values], axis=0)[old_index]
    result = DomainEndpointDataset(
        mode=city.mode,
        model_label=city.model_label,
        metadata=metadata,
        states_by_layer=states,
    )
    result.validate(require_complete=True)
    for domain in ("flower", "animal"):
        domain_rows = result.metadata[result.metadata["entity_domain"] == domain]
        mismatched = domain_rows[
            domain_rows["source_stimulus_id"].astype(str)
            != domain_rows.apply(
                lambda row: f"V4_4_T10000_N{int(row.gold_count)}_seed{int(row.seed)}",
                axis=1,
            )
        ]
        if not mismatched.empty:
            raise ValueError(f"{domain} rows are not paired to the registered city cells")
    return result


def _fit_projection(
    train_x: np.ndarray,
    test_x: np.ndarray,
    *,
    n_components: int,
    whiten: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    train = np.asarray(train_x, dtype=np.float32)
    test = np.asarray(test_x, dtype=np.float32)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    maximum = min(train_scaled.shape[0] - 1, train_scaled.shape[1])
    components = min(int(n_components), maximum)
    if components < 1:
        raise ValueError("PCA requires at least two training rows")
    solver = "randomized" if components < min(train_scaled.shape) else "full"
    pca = PCA(
        n_components=components,
        whiten=whiten,
        svd_solver=solver,
        random_state=0,
    )
    return (
        pca.fit_transform(train_scaled),
        pca.transform(test_scaled),
        scaler,
        pca,
    )


def _nearest_centroid_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    labels = np.unique(train_y)
    centroids = np.stack([train_x[train_y == label].mean(axis=0) for label in labels])
    distances = ((test_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return labels[np.argmin(distances, axis=1)]


def _probe_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    n_components: int,
) -> dict[str, float]:
    projected_train, projected_test, _, _ = _fit_projection(
        train_x, test_x, n_components=n_components, whiten=True
    )
    logistic = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=0)
    logistic.fit(projected_train, train_y)
    logistic_prediction = logistic.predict(projected_test)
    ncc_prediction = _nearest_centroid_predict(
        projected_train, train_y, projected_test
    )
    return {
        "logistic_balanced_accuracy": float(
            balanced_accuracy_score(test_y, logistic_prediction)
        ),
        "ncc_balanced_accuracy": float(
            balanced_accuracy_score(test_y, ncc_prediction)
        ),
    }


def select_layer(
    dataset: DomainEndpointDataset,
    *,
    selection_seeds: Sequence[int] = SELECTION_SEEDS,
    selection_domain: str = "city",
    n_components: int = 16,
    cv_folds: int = 5,
) -> tuple[int, list[dict[str, Any]]]:
    """Select one layer with grouped CV on city discovery trajectories."""

    seeds = tuple(map(int, selection_seeds))
    metadata = dataset.metadata
    selector = (
        metadata["seed"].astype(int).isin(seeds)
        & (metadata["entity_domain"].astype(str) == selection_domain)
    ).to_numpy()
    if int(selector.sum()) != len(seeds) * len(COUNTS):
        raise ValueError(
            f"Expected {len(seeds) * len(COUNTS)} {selection_domain} layer-selection "
            f"rows; found {int(selector.sum())}"
        )
    if cv_folds < 2 or cv_folds > len(seeds):
        raise ValueError(f"Invalid grouped CV fold count: {cv_folds}")
    seed_folds = [
        tuple(map(int, fold))
        for fold in np.array_split(np.asarray(seeds, dtype=int), cv_folds)
        if len(fold)
    ]
    labels = metadata["gold_count"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for layer in sorted(dataset.states_by_layer):
        values = dataset.states_by_layer[layer]
        fold_scores: list[dict[str, float]] = []
        for held_seeds in seed_folds:
            test = selector & np.isin(
                metadata["seed"].to_numpy(dtype=int),
                np.asarray(held_seeds, dtype=int),
            )
            train = selector & ~test
            fold_scores.append(
                _probe_scores(
                    values[train],
                    labels[train],
                    values[test],
                    labels[test],
                    n_components=n_components,
                )
            )
        logistic = float(
            np.mean([score["logistic_balanced_accuracy"] for score in fold_scores])
        )
        ncc = float(np.mean([score["ncc_balanced_accuracy"] for score in fold_scores]))
        rows.append(
            {
                "model_label": dataset.model_label,
                "mode": dataset.mode,
                "layer": int(layer),
                "selection_seed_count": len(seeds),
                "selection_domain": selection_domain,
                "selection_fold_count": len(seed_folds),
                "pca_dimensions": int(n_components),
                "cv_logistic_balanced_accuracy": logistic,
                "cv_ncc_balanced_accuracy": ncc,
                "selection_score": 0.5 * (logistic + ncc),
            }
        )
    selected = max(rows, key=lambda row: (row["selection_score"], -row["layer"]))
    return int(selected["layer"]), rows


def _count_residuals(
    train_x: np.ndarray,
    train_count: np.ndarray,
    test_x: np.ndarray,
    test_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centroids = {
        count: np.asarray(train_x[train_count == count], dtype=np.float32).mean(axis=0)
        for count in np.unique(train_count)
    }
    if set(map(int, np.unique(test_count))) - set(map(int, centroids)):
        raise ValueError("Test set contains a count absent from residualization training")
    residual_train = np.stack(
        [np.asarray(row, dtype=np.float32) - centroids[int(count)] for row, count in zip(train_x, train_count)]
    )
    residual_test = np.stack(
        [np.asarray(row, dtype=np.float32) - centroids[int(count)] for row, count in zip(test_x, test_count)]
    )
    return residual_train, residual_test


def evaluate_frozen_layer(
    dataset: DomainEndpointDataset,
    *,
    training_dataset: DomainEndpointDataset,
    layer: int,
    selection_seeds: Sequence[int] = SELECTION_SEEDS,
    evaluation_seeds: Sequence[int] = EVALUATION_SEEDS,
    dimensions: Sequence[int] = DEFAULT_PCA_DIMENSIONS,
) -> dict[str, Any]:
    """Train on city discovery and evaluate all three confirmation domains."""

    if (
        dataset.mode != training_dataset.mode
        or dataset.model_label != training_dataset.model_label
    ):
        raise ValueError("Training and evaluation datasets have different model/mode labels")
    if int(layer) not in dataset.states_by_layer or int(layer) not in training_dataset.states_by_layer:
        raise ValueError(f"Layer {layer} is absent from one endpoint dataset")
    metadata = dataset.metadata
    values = dataset.states_by_layer[int(layer)]
    count = metadata["gold_count"].to_numpy(dtype=int)
    domain = metadata["entity_domain"].astype(str).to_numpy()
    seed = metadata["seed"].to_numpy(dtype=int)
    test = np.isin(seed, np.asarray(evaluation_seeds, dtype=int))
    if int(test.sum()) != 300:
        raise ValueError(f"Expected 300 confirmation evaluation rows; got {int(test.sum())}")

    training_metadata = training_dataset.metadata
    training_values = training_dataset.states_by_layer[int(layer)]
    training_count = training_metadata["gold_count"].to_numpy(dtype=int)
    training_seed = training_metadata["seed"].to_numpy(dtype=int)
    training_domain = training_metadata["entity_domain"].astype(str).to_numpy()
    train = np.isin(training_seed, np.asarray(selection_seeds, dtype=int)) & (
        training_domain == "city"
    )
    if int(train.sum()) != len(selection_seeds) * len(COUNTS):
        raise ValueError(
            f"Expected {len(selection_seeds) * len(COUNTS)} city discovery rows; "
            f"got {int(train.sum())}"
        )

    overall = _probe_scores(
        training_values[train],
        training_count[train],
        values[test],
        count[test],
        n_components=16,
    )
    per_seed = []
    for held_seed in evaluation_seeds:
        seed_test = test & (seed == int(held_seed))
        score = _probe_scores(
            training_values[train],
            training_count[train],
            values[seed_test],
            count[seed_test],
            n_components=16,
        )
        per_seed.append({"seed": int(held_seed), **score})

    domain_count: dict[str, dict[str, float]] = {}
    for target in DOMAINS:
        target_test = test & (domain == target)
        domain_count[target] = _probe_scores(
            training_values[train],
            training_count[train],
            values[target_test],
            count[target_test],
            n_components=16,
        )
    cross_domain_mean = {
        metric: float(
            np.mean([domain_count[target][metric] for target in ("flower", "animal")])
        )
        for metric in (
            "logistic_balanced_accuracy",
            "ncc_balanced_accuracy",
        )
    }

    leakage_folds = []
    for held_seed in evaluation_seeds:
        fold_test = test & (seed == int(held_seed))
        fold_train = test & ~fold_test
        residual_train, residual_test = _count_residuals(
            values[fold_train],
            count[fold_train],
            values[fold_test],
            count[fold_test],
        )
        score = _probe_scores(
            residual_train,
            domain[fold_train],
            residual_test,
            domain[fold_test],
            n_components=16,
        )
        leakage_folds.append({"seed": int(held_seed), **score})
    domain_leakage = {
        metric: float(np.mean([row[metric] for row in leakage_folds]))
        for metric in (
            "logistic_balanced_accuracy",
            "ncc_balanced_accuracy",
        )
    }

    dimension_rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        pooled = _probe_scores(
            training_values[train],
            training_count[train],
            values[test],
            count[test],
            n_components=int(dimension),
        )
        target_scores: dict[str, dict[str, float]] = {}
        for target in DOMAINS:
            target_test = test & (domain == target)
            target_scores[target] = _probe_scores(
                training_values[train],
                training_count[train],
                values[target_test],
                count[target_test],
                n_components=int(dimension),
            )
        dimension_rows.append(
            {
                "dimensions": int(dimension),
                "pooled_logistic_balanced_accuracy": pooled[
                    "logistic_balanced_accuracy"
                ],
                "pooled_ncc_balanced_accuracy": pooled["ncc_balanced_accuracy"],
                "city_logistic_balanced_accuracy": target_scores["city"][
                    "logistic_balanced_accuracy"
                ],
                "city_ncc_balanced_accuracy": target_scores["city"][
                    "ncc_balanced_accuracy"
                ],
                "cross_domain_logistic_balanced_accuracy": float(
                    np.mean(
                        [
                            target_scores[target]["logistic_balanced_accuracy"]
                            for target in ("flower", "animal")
                        ]
                    )
                ),
                "cross_domain_ncc_balanced_accuracy": float(
                    np.mean(
                        [
                            target_scores[target]["ncc_balanced_accuracy"]
                            for target in ("flower", "animal")
                        ]
                    )
                ),
            }
        )

    return {
        "selected_layer": int(layer),
        "selection_seeds": list(map(int, selection_seeds)),
        "evaluation_seeds": list(map(int, evaluation_seeds)),
        "training_rows": int(train.sum()),
        "evaluation_rows": int(test.sum()),
        "count_probe_training_domain": "city",
        "overall_count": overall,
        "overall_count_by_evaluation_seed": per_seed,
        "count_by_evaluation_domain": domain_count,
        "city_confirmation_count": domain_count["city"],
        "cross_domain_count": {
            target: domain_count[target] for target in ("flower", "animal")
        },
        "cross_domain_count_mean": cross_domain_mean,
        "count_residual_domain_leakage": domain_leakage,
        "count_residual_domain_leakage_by_seed": leakage_folds,
        "dimension_sweep": dimension_rows,
        "count_chance": 0.1,
        "domain_chance": 1.0 / 3.0,
    }


def city_anchored_pca3(
    dataset: DomainEndpointDataset,
    *,
    training_dataset: DomainEndpointDataset,
    layer: int,
    selection_seeds: Sequence[int] = SELECTION_SEEDS,
) -> dict[str, Any]:
    """Fit PCA3 on city discovery, then transform the confirmation 300 panel."""

    metadata = dataset.metadata
    values = dataset.states_by_layer[int(layer)]
    training_metadata = training_dataset.metadata
    training_values = training_dataset.states_by_layer[int(layer)]
    fit_mask = (
        (training_metadata["entity_domain"].astype(str) == "city")
        & training_metadata["seed"].astype(int).isin(set(map(int, selection_seeds)))
    ).to_numpy()
    scaler = StandardScaler()
    scaled_fit = scaler.fit_transform(
        np.asarray(training_values[fit_mask], dtype=np.float32)
    )
    scaled_all = scaler.transform(np.asarray(values, dtype=np.float32))
    pca = PCA(n_components=3, whiten=False, svd_solver="randomized", random_state=0)
    pca.fit(scaled_fit)
    coordinates = pca.transform(scaled_all)
    points = []
    for row, xyz in zip(metadata.to_dict("records"), coordinates):
        points.append(
            {
                "entity_domain": str(row["entity_domain"]),
                "seed": int(row["seed"]),
                "gold_count": int(row["gold_count"]),
                "analysis_split": "confirmation",
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            }
        )
    return {
        "basis": "StandardScaler + PCA3 fitted on city discovery rows only",
        "fit_rows": int(fit_mask.sum()),
        "layer": int(layer),
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "points": points,
    }


def capture_audit(dataset: DomainEndpointDataset) -> dict[str, Any]:
    metadata = dataset.metadata
    transfer = metadata[metadata["entity_domain"].astype(str) != "city"]

    def counts(column: str) -> dict[str, int]:
        if column not in transfer:
            return {}
        values = transfer[column].dropna().astype(str)
        return {value: int((values == value).sum()) for value in sorted(values.unique())}

    exact = transfer["exact_count"].dropna() if "exact_count" in transfer else []
    return {
        "model_label": dataset.model_label,
        "mode": dataset.mode,
        "rows_total": int(len(metadata)),
        "rows_by_domain": {
            domain: int((metadata["entity_domain"].astype(str) == domain).sum())
            for domain in DOMAINS
        },
        "transfer_answer_states": int(transfer["answer_site_count"].sum()),
        "transfer_running_states": int(transfer["running_site_count"].sum()),
        "transfer_exact_count_rows": int(sum(bool(value) for value in exact)),
        "transfer_exact_count_denominator": int(len(exact)),
        "transfer_generation_rescue_rows": int(
            sum(bool(value) for value in transfer.get("generation_rescue", []))
        ),
        "trace_category_counts": counts("trace_category"),
        "marker_kind_counts": counts("marker_kind"),
    }


def flatten_dimension_rows(
    model_results: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> Iterable[dict[str, Any]]:
    for model, by_mode in model_results.items():
        for mode, payload in by_mode.items():
            for row in payload["metrics"]["dimension_sweep"]:
                yield {"model_label": model, "mode": mode, **row}
