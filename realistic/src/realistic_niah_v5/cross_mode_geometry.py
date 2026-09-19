"""Position-wise non-thinking versus native-thinking geometry comparison.

Each running/enumeration index is treated as one class. Discovery rows fit all
preprocessing and probes; confirmation rows measure class-specific probe
quality and covariance-aware cluster quality. The primary comparison keeps the
same registered 10-count x 30-seed trajectory panel in both modes. Each trace
contributes its actually observed positions 1..M; native-thinking traces may be
ragged and are retained without requiring a one-to-one trace or a correct final
answer. Position-specific support is reported explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr
from sklearn.covariance import OAS
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    silhouette_samples,
)
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "realistic_niah_cross_mode_position_geometry_v4_all_counts"
CLASSES = tuple(range(1, 11))
NATIVE_SITE_POLICIES = (
    "uniform",
    "trace_aware_count_boundary",
    "trace_aware_pre_label",
)
TRACE_AWARE_SITE_BY_MARKER_KIND = {
    # These marker spans carry an explicit ordinal/index cue.  The resulting
    # geometry is a sensitivity analysis because the cue itself can make k
    # decodable; it is not the primary completed-item estimand.
    "indexed": "marker_end",
    "ordinal": "marker_end",
    "inline_count": "marker_end",
    # A repeated bullet does not identify k, and the fallback parsers have no
    # genuine marker token.  Use the completed item boundary for those traces.
    "bullet": "item_end",
    "audit_sentence": "item_end",
    "completion_recap": "item_end",
    "evidence_sequence": "item_end",
}
TRACE_AWARE_PRE_LABEL_SITE_BY_MARKER_KIND = {
    # These sites are immediately before the explicit k cue.  For inline
    # Count/Record markers this is also after the city evidence has been read.
    "indexed": "pre_marker",
    "ordinal": "pre_marker",
    "inline_count": "pre_marker",
    # Invariant bullets and implicit chains do not expose a count label, so the
    # completed item endpoint is the closest comparable update boundary.
    "bullet": "item_end",
    "audit_sentence": "item_end",
    "completion_recap": "item_end",
    "evidence_sequence": "item_end",
}
QUALITY_DIRECTION = {
    "logistic_precision": "higher",
    "logistic_recall": "higher",
    "logistic_f1": "higher",
    "ncc_recall": "higher",
    "silhouette_cosine": "higher",
    "within_trace": "lower",
    "class_nc1_ratio": "lower",
    "centroid_nearest_squared": "higher",
    "fisher_nearest": "higher",
    "min_bhattacharyya": "higher",
    "mean_bhattacharyya": "higher",
    "nc2_class_cosine_deviation": "lower",
}


@dataclass
class ModeDataset:
    mode: str
    model_label: str
    metadata: pd.DataFrame
    states_by_layer: dict[int, np.ndarray]

    def validate(self) -> None:
        required = {"split", "seed", "occurrence"}
        missing = sorted(required - set(self.metadata.columns))
        if missing:
            raise ValueError(f"{self.mode} metadata is missing {missing}")
        lengths = {len(value) for value in self.states_by_layer.values()}
        if lengths != {len(self.metadata)}:
            raise ValueError(
                f"{self.mode} states/metadata mismatch: {lengths}/{len(self.metadata)}"
            )
        key_columns = ["split", "seed"]
        if "gold_count" in self.metadata.columns:
            key_columns.append("gold_count")
        if "stimulus_id" in self.metadata.columns:
            key_columns.append("stimulus_id")
        key_columns.append("occurrence")
        keys = self.metadata[key_columns]
        if keys.duplicated().any():
            raise ValueError(
                f"{self.mode} has duplicate trajectory/occurrence rows on "
                f"{key_columns}"
            )


@dataclass
class FittedLayer:
    discovery_x: np.ndarray
    discovery_y: np.ndarray
    confirmation_x: np.ndarray
    confirmation_y: np.ndarray
    confirmation_seed: np.ndarray
    logistic_prediction: np.ndarray
    ncc_prediction: np.ndarray
    pca_components: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def load_non_thinking_capture(
    capture_index: str | Path,
    *,
    design_variant: str = "v4.4",
    pooling: str = "span_end",
) -> ModeDataset:
    index_path = Path(capture_index)
    rows = [
        row
        for row in _read_jsonl(index_path)
        if str(row.get("design_variant")) == design_variant
    ]
    rows.sort(key=lambda row: (int(row["seed"]), int(row["count"])))
    if not rows:
        raise ValueError(f"No {design_variant} rows in {index_path}")
    first_path = index_path.parent / str(rows[0]["shard_path"])
    with np.load(first_path, allow_pickle=False) as archive:
        if pooling not in archive.files:
            raise ValueError(f"Pooling {pooling!r} is absent from {first_path}")
        layer_indices = archive["layer_indices"].astype(int)
        hidden_size = int(archive[pooling].shape[-1])
    total = sum(int(row["count"]) for row in rows)
    states = {
        int(layer): np.empty((total, hidden_size), dtype=np.float16)
        for layer in layer_indices
    }
    metadata_rows: list[dict[str, Any]] = []
    offset = 0
    for row in rows:
        count = int(row["count"])
        shard = index_path.parent / str(row["shard_path"])
        with np.load(shard, allow_pickle=False) as archive:
            layers = archive["layer_indices"].astype(int)
            if not np.array_equal(layers, layer_indices):
                raise ValueError(f"Layer mismatch in {shard}")
            values = np.asarray(archive[pooling])
            if values.shape[:2] != (len(layer_indices), count):
                raise ValueError(f"Unexpected {pooling} shape {values.shape} in {shard}")
            for layer_axis, layer in enumerate(layer_indices):
                states[int(layer)][offset : offset + count] = values[layer_axis]
        for occurrence in range(1, count + 1):
            metadata_rows.append(
                {
                    "split": str(row["split"]),
                    "seed": int(row["seed"]),
                    "occurrence": int(occurrence),
                    "gold_count": count,
                    "stimulus_id": str(row["stimulus_id"]),
                }
            )
        offset += count
    dataset = ModeDataset(
        mode="non_thinking",
        model_label=str(rows[0]["model_label"]),
        metadata=pd.DataFrame(metadata_rows),
        states_by_layer=states,
    )
    dataset.validate()
    return dataset


def load_native_thinking_capture(
    capture_index: str | Path,
    *,
    site_kind: str = "item_end",
    site_policy: str = "uniform",
    cohort: str = "one_to_one",
) -> ModeDataset:
    if cohort not in {"parser_hit", "one_to_one", "one_to_one_correct"}:
        raise ValueError(f"Unknown native-thinking cohort: {cohort}")
    if site_policy not in NATIVE_SITE_POLICIES:
        raise ValueError(f"Unknown native-thinking site policy: {site_policy}")
    index_path = Path(capture_index)
    index_rows = []
    descriptors: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
            str,
            str | None,
            list[int],
            list[int],
        ]
    ] = []
    for row in _read_jsonl(index_path):
        if cohort in {"one_to_one", "one_to_one_correct"} and not bool(
            row.get("trace_one_to_one")
        ):
            continue
        if cohort == "one_to_one_correct" and not bool(row.get("exact_count")):
            continue
        manifest_path = index_path.parent / str(row["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row_marker_kind = row.get("marker_kind")
        manifest_marker_kind = manifest.get("parser", {}).get("marker_kind")
        if (
            row_marker_kind is not None
            and manifest_marker_kind is not None
            and str(row_marker_kind) != str(manifest_marker_kind)
        ):
            raise ValueError(
                f"marker_kind mismatch for {row.get('request_id')}: "
                f"index={row_marker_kind!r}, manifest={manifest_marker_kind!r}"
            )
        marker_kind = (
            str(row_marker_kind)
            if row_marker_kind is not None
            else (
                str(manifest_marker_kind)
                if manifest_marker_kind is not None
                else None
            )
        )
        selected_site_kind = site_kind
        if site_policy != "uniform":
            policy = (
                TRACE_AWARE_SITE_BY_MARKER_KIND
                if site_policy == "trace_aware_count_boundary"
                else TRACE_AWARE_PRE_LABEL_SITE_BY_MARKER_KIND
            )
            if marker_kind not in policy:
                raise ValueError(
                    f"{site_policy} requires a registered marker_kind; "
                    f"got {marker_kind!r} for {row.get('request_id')}"
                )
            selected_site_kind = policy[marker_kind]
        site_indices = [
            index
            for index, site in enumerate(manifest["site_rows"])
            if str(site.get("site_kind")) == selected_site_kind
            and site.get("occurrence") is not None
        ]
        occurrences = [
            int(manifest["site_rows"][index]["occurrence"])
            for index in site_indices
        ]
        if not occurrences:
            continue
        if len(occurrences) != len(set(occurrences)):
            raise ValueError(
                f"Duplicate {selected_site_kind} occurrences for {row.get('request_id')}"
            )
        unexpected = sorted(set(occurrences) - set(CLASSES))
        if unexpected:
            raise ValueError(
                f"Unexpected {selected_site_kind} occurrences {unexpected} for "
                f"{row.get('request_id')}"
            )
        if cohort in {"one_to_one", "one_to_one_correct"}:
            expected_occurrences = list(
                range(1, int(row["gold_count"]) + 1)
            )
            if sorted(occurrences) != expected_occurrences:
                raise ValueError(
                    f"Index marks {row.get('request_id')} as {cohort}, but "
                    f"{selected_site_kind} occurrences are {sorted(occurrences)}; "
                    f"expected {expected_occurrences} for gold N="
                    f"{int(row['gold_count'])}"
                )
        order = np.argsort(np.asarray(occurrences, dtype=int))
        ordered_sites = np.asarray(site_indices, dtype=int)[order].tolist()
        ordered_occurrences = np.asarray(occurrences, dtype=int)[order].tolist()
        descriptors.append(
            (
                row,
                manifest,
                selected_site_kind,
                marker_kind,
                ordered_sites,
                ordered_occurrences,
            )
        )
        index_rows.append(row)
    descriptors.sort(key=lambda item: (int(item[0]["seed"]), int(item[0]["gold_count"])))
    index_rows.sort(key=lambda row: (int(row["seed"]), int(row["gold_count"])))
    if not descriptors:
        raise ValueError(
            f"No observed {site_policy}:{site_kind}/{cohort} trajectories "
            f"in {index_path}"
        )
    (
        first_row,
        _first_manifest,
        _first_selected_site_kind,
        _first_marker_kind,
        _first_sites,
        _first_occurrences,
    ) = descriptors[0]
    first_path = index_path.parent / str(first_row["states_path"])
    with np.load(first_path, allow_pickle=False) as archive:
        layer_indices = archive["layer_indices"].astype(int)
        hidden_size = int(archive["site_states"].shape[-1])
    total = sum(
        len(occurrences)
        for _, _, _, _, _, occurrences in descriptors
    )
    states = {
        int(layer): np.empty((total, hidden_size), dtype=np.float16)
        for layer in layer_indices
    }
    metadata_rows: list[dict[str, Any]] = []
    offset = 0
    for (
        row,
        _manifest,
        selected_site_kind,
        marker_kind,
        site_indices,
        occurrences,
    ) in descriptors:
        shard = index_path.parent / str(row["states_path"])
        selected_sites = np.asarray(site_indices, dtype=int)
        with np.load(shard, allow_pickle=False) as archive:
            layers = archive["layer_indices"].astype(int)
            if not np.array_equal(layers, layer_indices):
                raise ValueError(f"Layer mismatch in {shard}")
            values = np.asarray(archive["site_states"])[selected_sites]
            for layer_axis, layer in enumerate(layer_indices):
                states[int(layer)][offset : offset + len(occurrences)] = values[
                    :, layer_axis, :
                ]
        for occurrence in occurrences:
            metadata_rows.append(
                {
                    "split": str(row["split"]),
                    "seed": int(row["seed"]),
                    "occurrence": int(occurrence),
                    "gold_count": int(row["gold_count"]),
                    "stimulus_id": str(row["stimulus_id"]),
                    "trace_category": row.get("trace_category"),
                    "marker_kind": marker_kind,
                    "selected_site_kind": selected_site_kind,
                    "native_site_policy": site_policy,
                }
            )
        offset += len(occurrences)
    dataset = ModeDataset(
        mode="native_thinking",
        model_label=str(index_rows[0]["model_label"]),
        metadata=pd.DataFrame(metadata_rows),
        states_by_layer=states,
    )
    dataset.validate()
    return dataset


def _trajectory_key_columns(metadata: pd.DataFrame) -> list[str]:
    columns = ["split", "seed"]
    if "gold_count" in metadata.columns:
        columns.append("gold_count")
    return columns


def _seed_keys(metadata: pd.DataFrame) -> set[tuple[Any, ...]]:
    columns = _trajectory_key_columns(metadata)
    return {
        tuple(
            str(value) if column == "split" else int(value)
            for column, value in zip(columns, values)
        )
        for values in metadata[columns]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }


def _stimulus_ids_by_seed(metadata: pd.DataFrame) -> dict[tuple[Any, ...], str]:
    columns = _trajectory_key_columns(metadata)
    result: dict[tuple[Any, ...], str] = {}
    for raw_key, frame in metadata.groupby(columns, sort=False):
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        key = tuple(
            str(value) if column == "split" else int(value)
            for column, value in zip(columns, values)
        )
        stimulus_ids = sorted(set(frame["stimulus_id"].astype(str)))
        if len(stimulus_ids) != 1:
            raise ValueError(
                f"Trajectory {key} maps to stimulus IDs {stimulus_ids}"
            )
        result[key] = stimulus_ids[0]
    return result


def _subset_seed_keys(
    dataset: ModeDataset, keys: set[tuple[Any, ...]]
) -> ModeDataset:
    columns = _trajectory_key_columns(dataset.metadata)
    mask = np.fromiter(
        (
            tuple(
                str(value) if column == "split" else int(value)
                for column, value in zip(columns, values)
            )
            in keys
            for values in dataset.metadata[columns].itertuples(
                index=False, name=None
            )
        ),
        dtype=bool,
        count=len(dataset.metadata),
    )
    selected = dataset.metadata.loc[mask].copy()
    selected["_old_index"] = np.flatnonzero(mask)
    selected = selected.sort_values(columns + ["occurrence"])
    old_index = selected.pop("_old_index").to_numpy(dtype=int)
    result = ModeDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=selected.reset_index(drop=True),
        states_by_layer={
            layer: values[old_index] for layer, values in dataset.states_by_layer.items()
        },
    )
    result.validate()
    return result


def _panel_seeds(keys: set[tuple[Any, ...]]) -> dict[str, list[int]]:
    seeds = {
        split: sorted(
            {int(key[1]) for key in keys if str(key[0]) == split}
        )
        for split in ("discovery", "confirmation")
    }
    if not seeds["discovery"] or not seeds["confirmation"]:
        raise ValueError("Both discovery and confirmation need registered seeds")
    return seeds


def _position_support(dataset: ModeDataset) -> dict[str, dict[str, int]]:
    support: dict[str, dict[str, int]] = {}
    for split in ("discovery", "confirmation"):
        frame = dataset.metadata.loc[dataset.metadata["split"].astype(str).eq(split)]
        counts = frame["occurrence"].astype(int).value_counts()
        support[split] = {
            str(label): int(counts.get(label, 0)) for label in CLASSES
        }
    return support


def _trajectory_support(dataset: ModeDataset) -> dict[str, int]:
    columns = _trajectory_key_columns(dataset.metadata)
    trajectories = dataset.metadata[columns].drop_duplicates()
    return {
        split: int(trajectories["split"].astype(str).eq(split).sum())
        for split in ("discovery", "confirmation")
    }


def match_registered_seed_panel(
    non_thinking: ModeDataset, native_thinking: ModeDataset
) -> tuple[ModeDataset, ModeDataset, dict[str, list[int]]]:
    """Require identical split/seed/count trajectories, allowing ragged positions."""

    if non_thinking.model_label != native_thinking.model_label:
        raise ValueError("Cross-mode comparison requires the same model checkpoint")
    non_thinking_keys = _seed_keys(non_thinking.metadata)
    native_thinking_keys = _seed_keys(native_thinking.metadata)
    if non_thinking_keys != native_thinking_keys:
        missing_native = sorted(non_thinking_keys - native_thinking_keys)
        extra_native = sorted(native_thinking_keys - non_thinking_keys)
        raise ValueError(
            "The registered trajectory panels differ: "
            f"missing_native={missing_native}, extra_native={extra_native}"
        )
    non_thinking_stimuli = _stimulus_ids_by_seed(non_thinking.metadata)
    native_thinking_stimuli = _stimulus_ids_by_seed(native_thinking.metadata)
    stimulus_mismatches = {
        key: (non_thinking_stimuli[key], native_thinking_stimuli[key])
        for key in sorted(non_thinking_keys)
        if non_thinking_stimuli[key] != native_thinking_stimuli[key]
    }
    if stimulus_mismatches:
        raise ValueError(
            "The registered trajectory panel has cross-mode stimulus mismatches: "
            f"{stimulus_mismatches}"
        )
    seeds = _panel_seeds(non_thinking_keys)
    return (
        _subset_seed_keys(non_thinking, non_thinking_keys),
        _subset_seed_keys(native_thinking, native_thinking_keys),
        seeds,
    )


def pair_complete_trajectories(
    non_thinking: ModeDataset, native_thinking: ModeDataset
) -> tuple[ModeDataset, ModeDataset, dict[str, list[int]]]:
    if non_thinking.model_label != native_thinking.model_label:
        raise ValueError("Cross-mode comparison requires the same model checkpoint")

    def complete(metadata: pd.DataFrame) -> set[tuple[Any, ...]]:
        columns = _trajectory_key_columns(metadata)
        result: set[tuple[Any, ...]] = set()
        for raw_key, group in metadata.groupby(columns):
            values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            key = tuple(
                str(value) if column == "split" else int(value)
                for column, value in zip(columns, values)
            )
            expected_count = (
                int(group["gold_count"].iloc[0])
                if "gold_count" in group.columns
                else len(CLASSES)
            )
            if sorted(group["occurrence"].astype(int).tolist()) == list(
                range(1, expected_count + 1)
            ):
                result.add(key)
        return result

    common = complete(non_thinking.metadata) & complete(native_thinking.metadata)
    if not common:
        raise ValueError("The two modes have no common complete trajectory")

    left = _subset_seed_keys(non_thinking, common)
    right = _subset_seed_keys(native_thinking, common)
    key_columns = _trajectory_key_columns(left.metadata) + ["occurrence"]
    left_keys = left.metadata[key_columns].to_records(index=False)
    right_keys = right.metadata[key_columns].to_records(index=False)
    if not np.array_equal(left_keys, right_keys):
        raise RuntimeError("Paired cross-mode metadata keys are not identical")
    seeds = _panel_seeds(common)
    return left, right, seeds


def _fit_layer(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    pca_dim: int,
    random_state: int,
) -> FittedLayer:
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy()
    confirmation = metadata["split"].astype(str).eq("confirmation").to_numpy()
    yd = metadata.loc[discovery, "occurrence"].to_numpy(dtype=int)
    yc = metadata.loc[confirmation, "occurrence"].to_numpy(dtype=int)
    if set(yd.tolist()) != set(CLASSES):
        raise ValueError("Discovery rows do not cover every position class")
    if set(yc.tolist()) != set(CLASSES):
        raise ValueError("Confirmation rows do not cover every position class")
    confirmation_support = {label: int(np.sum(yc == label)) for label in CLASSES}
    if min(confirmation_support.values()) < 3:
        raise ValueError(
            "Confirmation geometry needs at least three observations per position "
            "so every delete-one-seed replicate retains covariance support"
        )
    scaler = StandardScaler().fit(states[discovery].astype(np.float32))
    xd_scaled = scaler.transform(states[discovery].astype(np.float32))
    xc_scaled = scaler.transform(states[confirmation].astype(np.float32))
    components = min(
        int(pca_dim),
        int(xd_scaled.shape[0] - len(CLASSES)),
        int(xd_scaled.shape[1]),
    )
    if components < 2:
        raise ValueError("Discovery data support fewer than two PCA components")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(xd_scaled)
    xd = pca.transform(xd_scaled)
    xc = pca.transform(xc_scaled)
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    ).fit(xd, yd)
    logistic_prediction = logistic.predict(xc)
    centroids = np.stack([xd[yd == label].mean(axis=0) for label in CLASSES])
    distances = np.square(xc[:, None, :] - centroids[None, :, :]).sum(axis=-1)
    ncc_prediction = np.asarray(CLASSES, dtype=int)[np.argmin(distances, axis=1)]
    return FittedLayer(
        discovery_x=xd,
        discovery_y=yd,
        confirmation_x=xc,
        confirmation_y=yc,
        confirmation_seed=metadata.loc[confirmation, "seed"].to_numpy(dtype=int),
        logistic_prediction=logistic_prediction,
        ncc_prediction=ncc_prediction,
        pca_components=components,
    )


def _bhattacharyya(
    mean_left: np.ndarray,
    covariance_left: np.ndarray,
    mean_right: np.ndarray,
    covariance_right: np.ndarray,
) -> float:
    pooled = 0.5 * (covariance_left + covariance_right)
    difference = mean_left - mean_right
    sign_p, logdet_p = np.linalg.slogdet(pooled)
    sign_l, logdet_l = np.linalg.slogdet(covariance_left)
    sign_r, logdet_r = np.linalg.slogdet(covariance_right)
    if min(sign_p, sign_l, sign_r) <= 0:
        return np.nan
    mean_term = 0.125 * float(difference @ np.linalg.solve(pooled, difference))
    covariance_term = 0.5 * float(logdet_p - 0.5 * (logdet_l + logdet_r))
    return mean_term + covariance_term


def _evaluate_confirmation(fitted: FittedLayer) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = fitted.confirmation_x
    y = fitted.confirmation_y
    logistic_prediction = fitted.logistic_prediction
    ncc_prediction = fitted.ncc_prediction
    labels = np.asarray(CLASSES, dtype=int)
    class_support = {label: int(np.sum(y == label)) for label in CLASSES}
    precision, recall, f1, _support = precision_recall_fscore_support(
        y,
        logistic_prediction,
        labels=labels,
        zero_division=0,
    )
    _ncc_precision, ncc_recall, _ncc_f1, _ncc_support = (
        precision_recall_fscore_support(
            y,
            ncc_prediction,
            labels=labels,
            zero_division=0,
        )
    )
    silhouette = silhouette_samples(x, y, metric="cosine")
    means = {label: x[y == label].mean(axis=0) for label in CLASSES}
    covariances = {
        label: OAS(store_precision=False).fit(x[y == label]).covariance_
        for label in CLASSES
    }
    grand = x.mean(axis=0)
    centered_means = np.stack([means[label] - grand for label in CLASSES])
    # Report a class-balanced multiclass SNR so ragged late-position support
    # does not silently give early classes more weight.  The signal is the
    # mean squared distance of the ten confirmation centroids from their
    # class-balanced grand centroid; the noise is the macro-average squared
    # residual around each confirmation centroid.  Standardization and PCA
    # were fit on discovery only, so this is a held-out descriptive cluster
    # statistic rather than a training-set score.
    class_means = np.stack([means[label] for label in CLASSES])
    balanced_grand = class_means.mean(axis=0)
    signal_power = float(
        np.square(class_means - balanced_grand).sum(axis=1).mean()
    )
    class_noise_power = np.asarray(
        [
            np.square(x[y == label] - means[label]).sum(axis=1).mean()
            for label in CLASSES
        ],
        dtype=float,
    )
    noise_power = float(class_noise_power.mean())
    snr_ratio = float(signal_power / max(noise_power, np.finfo(float).eps))
    snr_db = float(10.0 * np.log10(max(snr_ratio, np.finfo(float).tiny)))
    norms = np.linalg.norm(centered_means, axis=1)
    normalized = centered_means / np.maximum(norms[:, None], np.finfo(float).eps)
    gram = normalized @ normalized.T
    ideal_off_diagonal = -1.0 / (len(CLASSES) - 1)
    pairwise_bhattacharyya: dict[tuple[int, int], float] = {}
    for left_index, left in enumerate(CLASSES):
        for right in CLASSES[left_index + 1 :]:
            pairwise_bhattacharyya[(left, right)] = _bhattacharyya(
                means[left], covariances[left], means[right], covariances[right]
            )
    rows: list[dict[str, Any]] = []
    for class_index, label in enumerate(CLASSES):
        other_labels = [value for value in CLASSES if value != label]
        centroid_distances = np.asarray(
            [np.square(means[label] - means[other]).sum() for other in other_labels]
        )
        within_trace = float(np.trace(covariances[label]))
        bhattacharyya = np.asarray(
            [
                pairwise_bhattacharyya[tuple(sorted((label, other)))]
                for other in other_labels
            ],
            dtype=float,
        )
        cosine_deviation = np.asarray(
            [
                abs(gram[class_index, other - 1] - ideal_off_diagonal)
                for other in other_labels
            ]
        )
        values = {
            "logistic_precision": float(precision[class_index]),
            "logistic_recall": float(recall[class_index]),
            "logistic_f1": float(f1[class_index]),
            "ncc_recall": float(ncc_recall[class_index]),
            "silhouette_cosine": float(silhouette[y == label].mean()),
            "within_trace": within_trace,
            "class_nc1_ratio": float(
                within_trace / max(float(norms[class_index] ** 2), np.finfo(float).eps)
            ),
            "centroid_nearest_squared": float(centroid_distances.min()),
            "fisher_nearest": float(
                centroid_distances.min() / max(within_trace, np.finfo(float).eps)
            ),
            "min_bhattacharyya": float(np.nanmin(bhattacharyya)),
            "mean_bhattacharyya": float(np.nanmean(bhattacharyya)),
            "nc2_class_cosine_deviation": float(cosine_deviation.mean()),
        }
        for metric, value in values.items():
            direction = QUALITY_DIRECTION[metric]
            rows.append(
                {
                    "occurrence": int(label),
                    "n_confirmation_class": class_support[label],
                    "metric": metric,
                    "value": value,
                    "quality_direction": direction,
                    "quality_value": value if direction == "higher" else -value,
                }
            )

    residuals = np.concatenate([x[y == label] - means[label] for label in CLASSES])
    sigma_w = residuals.T @ residuals / len(residuals)
    sigma_b = centered_means.T @ centered_means / len(CLASSES)
    # Delete-one-seed jackknife replicates can have far fewer residual degrees
    # of freedom than PCA dimensions (for example, Gemma has two confirmation
    # observations per class after one of three seeds is omitted).  In that
    # regime sigma_w is necessarily singular, and a fixed trace-scaled ridge
    # can still be too small for LAPACK's positive-definite factorization.
    # Symmetrize in float64 and increase an explicit, audited ridge until the
    # generalized eigenproblem is numerically well posed.  This changes only
    # the regularized MANOVA diagnostics; class metrics and fitted probes are
    # untouched.
    sigma_w = np.asarray(0.5 * (sigma_w + sigma_w.T), dtype=np.float64)
    sigma_b = np.asarray(0.5 * (sigma_b + sigma_b.T), dtype=np.float64)
    dimension = max(1, sigma_w.shape[0])
    within_scale = max(
        np.finfo(np.float64).eps,
        abs(float(np.trace(sigma_w))) / dimension,
        float(np.linalg.norm(sigma_w, ord="fro")) / dimension,
    )
    minimum_within_eigenvalue = float(np.linalg.eigvalsh(sigma_w)[0])
    ridge = max(
        np.finfo(np.float64).eps,
        1e-6 * within_scale,
        -minimum_within_eigenvalue + 1e-8 * within_scale,
    )
    ridge_attempts = 0
    identity = np.eye(sigma_w.shape[0], dtype=np.float64)
    while True:
        ridge_attempts += 1
        try:
            regularized_within = sigma_w + ridge * identity
            np.linalg.cholesky(regularized_within)
            eigenvalues = eigvalsh(sigma_b, regularized_within)
            break
        except np.linalg.LinAlgError:
            if ridge_attempts >= 12:
                raise
            ridge *= 10.0
    eigenvalues = np.clip(np.real(eigenvalues), 0.0, None)
    ideal = np.full_like(gram, ideal_off_diagonal)
    np.fill_diagonal(ideal, 1.0)
    global_metrics = {
        "n_confirmation": int(len(x)),
        "n_confirmation_seed_clusters": int(len(np.unique(fitted.confirmation_seed))),
        "min_confirmation_per_class": int(min(class_support.values())),
        "max_confirmation_per_class": int(max(class_support.values())),
        "pca_components": int(fitted.pca_components),
        "logistic_accuracy": float(accuracy_score(y, logistic_prediction)),
        "logistic_balanced_accuracy": float(
            balanced_accuracy_score(y, logistic_prediction)
        ),
        "ncc_accuracy": float(accuracy_score(y, ncc_prediction)),
        "ncc_balanced_accuracy": float(balanced_accuracy_score(y, ncc_prediction)),
        "class_balanced_snr": snr_ratio,
        "class_balanced_snr_db": snr_db,
        "class_balanced_signal_power": signal_power,
        "class_balanced_noise_power": noise_power,
        "nc4_probe_ncc_disagreement": float(
            np.mean(logistic_prediction != ncc_prediction)
        ),
        "nc1_trace_sigmaw_sigmab_pinv_over_c": float(
            np.trace(sigma_w @ np.linalg.pinv(sigma_b)) / len(CLASSES)
        ),
        "nc2_etf_gram_relative_error": float(
            np.linalg.norm(gram - ideal) / np.linalg.norm(ideal)
        ),
        "nc2_centered_centroid_norm_cv": float(
            norms.std(ddof=1) / norms.mean() if norms.mean() else np.nan
        ),
        "pillai_trace_regularized": float(np.sum(eigenvalues / (1.0 + eigenvalues))),
        "lawley_hotelling_trace_regularized": float(np.sum(eigenvalues)),
        "wilks_lambda_regularized": float(np.exp(-np.log1p(eigenvalues).sum())),
        "wilks_log_separation_regularized": float(np.log1p(eigenvalues).sum()),
        "covariance_ridge": float(ridge),
        "covariance_ridge_attempts": int(ridge_attempts),
        "covariance_min_eigenvalue_before_ridge": minimum_within_eigenvalue,
        "covariance_within_scale": float(within_scale),
    }
    return pd.DataFrame(rows), global_metrics


def _trend(per_class: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, frame in per_class.groupby("metric", sort=True):
        ordered = frame.sort_values("occurrence")
        x = ordered["occurrence"].to_numpy(dtype=float)
        quality = ordered["quality_value"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, quality, deg=1)[0])
        correlation = (
            float(spearmanr(x, quality).statistic)
            if np.ptp(quality) > 0
            else np.nan
        )
        rows.append(
            {
                "metric": str(metric),
                "quality_direction": str(ordered["quality_direction"].iloc[0]),
                "quality_slope_per_index": slope,
                "quality_spearman": correlation,
                "late_8_10_minus_early_1_3": float(
                    quality[x >= 8].mean() - quality[x <= 3].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _jackknife_se(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return np.nan
    return float(
        np.sqrt((len(finite) - 1) / len(finite) * np.square(finite - finite.mean()).sum())
    )


def _subset_fitted(fitted: FittedLayer, keep: np.ndarray) -> FittedLayer:
    return FittedLayer(
        discovery_x=fitted.discovery_x,
        discovery_y=fitted.discovery_y,
        confirmation_x=fitted.confirmation_x[keep],
        confirmation_y=fitted.confirmation_y[keep],
        confirmation_seed=fitted.confirmation_seed[keep],
        logistic_prediction=fitted.logistic_prediction[keep],
        ncc_prediction=fitted.ncc_prediction[keep],
        pca_components=fitted.pca_components,
    )


def _matched_panel_jackknife_trends(
    fitted_by_mode: Mapping[str, FittedLayer],
    point_trends: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = ("non_thinking", "native_thinking")
    seeds_left = sorted(np.unique(fitted_by_mode[modes[0]].confirmation_seed))
    seeds_right = sorted(np.unique(fitted_by_mode[modes[1]].confirmation_seed))
    if seeds_left != seeds_right:
        raise ValueError("Matched-panel jackknife requires identical confirmation seeds")
    replicates: dict[str, list[pd.DataFrame]] = {mode: [] for mode in modes}
    for omitted_seed in seeds_left:
        for mode in modes:
            fitted = fitted_by_mode[mode]
            keep = fitted.confirmation_seed != int(omitted_seed)
            per_class, _global = _evaluate_confirmation(_subset_fitted(fitted, keep))
            replicate = _trend(per_class)
            replicate["omitted_seed"] = int(omitted_seed)
            replicates[mode].append(replicate)
    mode_rows = []
    cross_rows = []
    summaries = {
        mode: pd.concat(replicates[mode], ignore_index=True) for mode in modes
    }
    trend_columns = (
        "quality_slope_per_index",
        "quality_spearman",
        "late_8_10_minus_early_1_3",
    )
    for mode in modes:
        point = point_trends[mode].set_index("metric")
        for metric, group in summaries[mode].groupby("metric"):
            for estimand in trend_columns:
                estimate = float(point.loc[metric, estimand])
                se = _jackknife_se(group[estimand].to_numpy(dtype=float))
                mode_rows.append(
                    {
                        "mode": mode,
                        "metric": metric,
                        "estimand": estimand,
                        "estimate": estimate,
                        "jackknife_se": se,
                        "ci_low_normal": estimate - 1.96 * se,
                        "ci_high_normal": estimate + 1.96 * se,
                        "jackknife_seed_count": len(seeds_left),
                    }
                )
    left_point = point_trends[modes[0]].set_index("metric")
    right_point = point_trends[modes[1]].set_index("metric")
    merged_replicates = summaries[modes[1]].merge(
        summaries[modes[0]],
        on=["metric", "omitted_seed"],
        suffixes=("_native", "_nonthinking"),
    )
    for metric, group in merged_replicates.groupby("metric"):
        for estimand in trend_columns:
            estimate = float(right_point.loc[metric, estimand] - left_point.loc[metric, estimand])
            differences = (
                group[f"{estimand}_native"].to_numpy(dtype=float)
                - group[f"{estimand}_nonthinking"].to_numpy(dtype=float)
            )
            se = _jackknife_se(differences)
            cross_rows.append(
                {
                    "metric": metric,
                    "estimand": estimand,
                    "native_minus_nonthinking": estimate,
                    "jackknife_se": se,
                    "ci_low_normal": estimate - 1.96 * se,
                    "ci_high_normal": estimate + 1.96 * se,
                    "jackknife_seed_count": len(seeds_left),
                }
            )
    return pd.DataFrame(mode_rows), pd.DataFrame(cross_rows)


def compare_position_geometry(
    non_thinking_capture_index: str | Path,
    native_thinking_capture_index: str | Path,
    output_dir: str | Path,
    *,
    design_variant: str = "v4.4",
    non_thinking_pooling: str = "span_end",
    native_site_kind: str = "item_end",
    native_site_policy: str = "uniform",
    native_cohort: str = "parser_hit",
    pca_dim: int = 32,
    layers: Sequence[int] | None = None,
    random_state: int = 0,
) -> dict[str, Path]:
    non_thinking = load_non_thinking_capture(
        non_thinking_capture_index,
        design_variant=design_variant,
        pooling=non_thinking_pooling,
    )
    native_thinking = load_native_thinking_capture(
        native_thinking_capture_index,
        site_kind=native_site_kind,
        site_policy=native_site_policy,
        cohort=native_cohort,
    )
    if native_cohort == "parser_hit":
        non_thinking, native_thinking, seed_panel = match_registered_seed_panel(
            non_thinking, native_thinking
        )
        analysis_design = "fixed_registered_seed_panel_observed_positions"
    else:
        non_thinking, native_thinking, seed_panel = pair_complete_trajectories(
            non_thinking, native_thinking
        )
        analysis_design = "complete_trajectory_paired_sensitivity"
    available_layers = sorted(
        set(non_thinking.states_by_layer) & set(native_thinking.states_by_layer)
    )
    selected_layers = available_layers if layers is None else sorted(set(map(int, layers)))
    missing = sorted(set(selected_layers) - set(available_layers))
    if missing:
        raise ValueError(f"Requested layers are unavailable in both modes: {missing}")
    per_class_rows = []
    global_rows = []
    trend_rows = []
    jackknife_rows = []
    cross_jackknife_rows = []
    mode_datasets = {
        "non_thinking": non_thinking,
        "native_thinking": native_thinking,
    }
    for layer in selected_layers:
        fitted_by_mode: dict[str, FittedLayer] = {}
        trends_by_mode: dict[str, pd.DataFrame] = {}
        for mode, dataset in mode_datasets.items():
            fitted = _fit_layer(
                dataset.states_by_layer[layer],
                dataset.metadata,
                pca_dim=pca_dim,
                random_state=random_state,
            )
            fitted_by_mode[mode] = fitted
            per_class, global_metrics = _evaluate_confirmation(fitted)
            per_class.insert(0, "layer", int(layer))
            per_class.insert(0, "mode", mode)
            per_class.insert(0, "model_label", dataset.model_label)
            per_class_rows.append(per_class)
            global_rows.append(
                {
                    "model_label": dataset.model_label,
                    "mode": mode,
                    "layer": int(layer),
                    **global_metrics,
                }
            )
            trends = _trend(per_class)
            trends_by_mode[mode] = trends
            trends.insert(0, "layer", int(layer))
            trends.insert(0, "mode", mode)
            trends.insert(0, "model_label", dataset.model_label)
            trend_rows.append(trends)
        mode_jackknife, cross_jackknife = _matched_panel_jackknife_trends(
            fitted_by_mode, trends_by_mode
        )
        for frame in (mode_jackknife, cross_jackknife):
            frame.insert(0, "layer", int(layer))
            frame.insert(0, "model_label", non_thinking.model_label)
        jackknife_rows.append(mode_jackknife)
        cross_jackknife_rows.append(cross_jackknife)

    per_class_frame = pd.concat(per_class_rows, ignore_index=True)
    native = per_class_frame.loc[
        per_class_frame["mode"].eq("native_thinking")
    ]
    nonthinking = per_class_frame.loc[
        per_class_frame["mode"].eq("non_thinking")
    ]
    cross_position = native.merge(
        nonthinking,
        on=["model_label", "layer", "occurrence", "metric"],
        suffixes=("_native", "_nonthinking"),
    )
    cross_position["native_minus_nonthinking_quality"] = (
        cross_position["quality_value_native"]
        - cross_position["quality_value_nonthinking"]
    )
    output = Path(output_dir)
    paths = {
        "per_class": output / "position_class_quality.csv",
        "global": output / "global_covariance_geometry.csv",
        "trends": output / "position_quality_trends.csv",
        "trend_jackknife": output / "position_quality_trend_jackknife.csv",
        "cross_mode_position": output / "native_minus_nonthinking_by_position.csv",
        "cross_mode_trend": output / "native_minus_nonthinking_trend_jackknife.csv",
        "audit": output / "cross_mode_geometry_audit.json",
    }
    _atomic_csv(paths["per_class"], per_class_frame)
    _atomic_csv(paths["global"], pd.DataFrame(global_rows))
    _atomic_csv(paths["trends"], pd.concat(trend_rows, ignore_index=True))
    _atomic_csv(paths["trend_jackknife"], pd.concat(jackknife_rows, ignore_index=True))
    _atomic_csv(paths["cross_mode_position"], cross_position)
    _atomic_csv(
        paths["cross_mode_trend"],
        pd.concat(cross_jackknife_rows, ignore_index=True),
    )
    _atomic_json(
        paths["audit"],
        {
            "schema_version": SCHEMA_VERSION,
            "model_label": non_thinking.model_label,
            "non_thinking_capture_index": str(Path(non_thinking_capture_index).resolve()),
            "native_thinking_capture_index": str(Path(native_thinking_capture_index).resolve()),
            "non_thinking_design_variant": design_variant,
            "non_thinking_pooling": non_thinking_pooling,
            "native_site_kind": native_site_kind,
            "native_site_policy": native_site_policy,
            "native_site_policy_mapping": (
                dict(TRACE_AWARE_SITE_BY_MARKER_KIND)
                if native_site_policy == "trace_aware_count_boundary"
                else (
                    dict(TRACE_AWARE_PRE_LABEL_SITE_BY_MARKER_KIND)
                    if native_site_policy == "trace_aware_pre_label"
                    else None
                )
            ),
            "native_selected_site_support": {
                str(site): int(count)
                for site, count in native_thinking.metadata[
                    "selected_site_kind"
                ].value_counts().sort_index().items()
            },
            "native_marker_kind_support": {
                str(marker): int(count)
                for marker, count in native_thinking.metadata[
                    "marker_kind"
                ].value_counts(dropna=False).sort_index().items()
            },
            "native_cohort": native_cohort,
            "analysis_design": analysis_design,
            "stimulus_alignment": "exact split/seed/gold_count/stimulus_id match",
            "registered_seed_panel": seed_panel,
            "registered_trajectory_panel": {
                "non_thinking": _trajectory_support(non_thinking),
                "native_thinking": _trajectory_support(native_thinking),
            },
            "position_support": {
                "non_thinking": _position_support(non_thinking),
                "native_thinking": _position_support(native_thinking),
            },
            "layers": selected_layers,
            "pca_dim_requested": int(pca_dim),
            "preprocessing_fit_split": "discovery only",
            "probe_fit_split": "discovery only",
            "evaluation_split": "confirmation only",
            "classification_weighting": (
                "class-balanced multinomial logistic fit; balanced accuracy is the "
                "primary summary when position support is ragged"
            ),
            "snr_definition": (
                "confirmation-only class-balanced trace ratio after discovery-fitted "
                "standardization and PCA: mean_k ||mu_k - mean_j(mu_j)||^2 divided "
                "by mean_k mean_i ||x_ki - mu_k||^2; dB = 10 log10(ratio); higher "
                "means more centroid separation per unit within-class scatter"
            ),
            "uncertainty": (
                "matched-panel delete-one-confirmation-seed jackknife; native-thinking "
                "positions may be missing within a seed"
            ),
            "cluster_labels": list(CLASSES),
            "quality_direction": QUALITY_DIRECTION,
            "nc4_note": (
                "No native classifier is trained on these intermediate position labels; "
                "reported NCC accuracy and probe-NCC disagreement are NC4-like diagnostics, "
                "not the original terminal-classifier NC4 statistic."
            ),
            "regularization": (
                "discovery-fitted PCA plus OAS class covariance; MANOVA generalized "
                "eigenvalues use an explicit trace-scaled ridge"
            ),
        },
    )
    return paths
