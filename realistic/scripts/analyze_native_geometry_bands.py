#!/usr/bin/env python3
"""Diagnose apparent two-band structure in native-thinking PCA geometry.

The diagnostic deliberately separates three questions:

1. Is a two-cluster description supported in the displayed PCA3 coordinates?
2. Is cluster membership a trajectory-level nuisance (seed/format/boundary), rather
   than an ordinal-count class?
3. Does the split survive within-trajectory centering in the original hidden space?

The script is descriptive.  K-means labels are never used as counting labels and
the confirmation split remains excluded from the PCA fit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    load_native_thinking_capture,
)
from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    confirmation_metrics,
)


ORDINAL_LABELS = tuple(range(1, 11))
SENSITIVITY_SITES = (
    "pre_city",
    "city_end",
    "city_unit_end",
    "item_end",
    "post_boundary",
)
DISPLAY_YAW = -0.72
DISPLAY_PITCH = 0.46


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def trajectory_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["split"].astype(str)
        + ":"
        + frame["seed"].astype(int).astype(str)
        + ":"
        + frame["gold_count"].astype(int).astype(str)
    )


def center_within_trajectory(states: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    centered = states.astype(np.float32).copy()
    keys = trajectory_key(metadata)
    for key in sorted(keys.unique()):
        mask = keys.eq(key).to_numpy()
        centered[mask] -= centered[mask].mean(axis=0, keepdims=True)
    return centered


def between_trajectory_fraction(states: np.ndarray, metadata: pd.DataFrame) -> float:
    values = states.astype(np.float64)
    grand = values.mean(axis=0, keepdims=True)
    keys = trajectory_key(metadata)
    between = 0.0
    within = 0.0
    for key in sorted(keys.unique()):
        group = values[keys.eq(key).to_numpy()]
        mean = group.mean(axis=0, keepdims=True)
        between += float(group.shape[0] * np.square(mean - grand).sum())
        within += float(np.square(group - mean).sum())
    total = between + within
    return float(between / total) if total > 0 else math.nan


def display_vertical(coordinates: np.ndarray) -> np.ndarray:
    """Match the default report camera's upward screen coordinate."""

    cy, sy = math.cos(DISPLAY_YAW), math.sin(DISPLAY_YAW)
    cp, sp = math.cos(DISPLAY_PITCH), math.sin(DISPLAY_PITCH)
    x, y, z = coordinates.T
    z_after_yaw = -sy * x + cy * z
    return cp * y - sp * z_after_yaw


def two_band_fit(coordinates: np.ndarray, random_state: int) -> dict[str, Any]:
    if len(coordinates) < 4:
        raise ValueError("Two-band diagnostic needs at least four rows")
    labels = KMeans(n_clusters=2, n_init=50, random_state=random_state).fit_predict(
        coordinates
    )
    vertical = display_vertical(coordinates)
    means = {label: float(vertical[labels == label].mean()) for label in (0, 1)}
    upper_label = max(means, key=means.get)
    band = np.asarray(["upper" if value == upper_label else "lower" for value in labels])
    return {
        "band": band,
        "silhouette": float(silhouette_score(coordinates, labels)),
        "cluster_sizes": {
            name: int(np.sum(band == name)) for name in ("upper", "lower")
        },
        "display_vertical_means": {
            name: float(vertical[band == name].mean())
            for name in ("upper", "lower")
        },
    }


def summarize_frozen_bands(
    coordinates: np.ndarray,
    band: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize already-assigned bands without fitting on the summarized rows."""

    selected = np.ones(len(coordinates), dtype=bool) if mask is None else mask
    values = coordinates[selected]
    names = np.asarray(band, dtype=str)[selected]
    if len(values) < 4:
        raise ValueError("Two-band diagnostic needs at least four rows")
    encoded = np.asarray([1 if value == "upper" else 0 for value in names])
    silhouette = (
        float(silhouette_score(values, encoded))
        if len(np.unique(encoded)) == 2
        else math.nan
    )
    vertical = display_vertical(values)
    return {
        "silhouette": silhouette,
        "cluster_sizes": {
            name: int(np.sum(names == name)) for name in ("upper", "lower")
        },
        "display_vertical_means": {
            name: (
                float(vertical[names == name].mean())
                if np.any(names == name)
                else math.nan
            )
            for name in ("upper", "lower")
        },
    }


def fit_discovery_frozen_bands(
    coordinates: np.ndarray,
    discovery: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    """Fit two bands on discovery only and assign every row with frozen centers."""

    if int(discovery.sum()) < 4:
        raise ValueError("Discovery band fit needs at least four rows")
    estimator = KMeans(n_clusters=2, n_init=50, random_state=random_state).fit(
        coordinates[discovery]
    )
    labels = estimator.predict(coordinates)
    discovery_vertical = display_vertical(coordinates[discovery])
    discovery_labels = labels[discovery]
    means = {
        label: float(discovery_vertical[discovery_labels == label].mean())
        for label in (0, 1)
    }
    upper_label = max(means, key=means.get)
    band = np.asarray(
        ["upper" if value == upper_label else "lower" for value in labels]
    )
    centers = estimator.cluster_centers_
    return {
        "band": band,
        "fit_split": "discovery",
        "discovery_cluster_sizes": {
            name: int(np.sum(band[discovery] == name))
            for name in ("upper", "lower")
        },
        "centers": {
            "upper": [float(value) for value in centers[upper_label]],
            "lower": [float(value) for value in centers[1 - upper_label]],
        },
    }


def _class_balanced_snr(
    projected: np.ndarray,
    labels: np.ndarray,
    row_mask: np.ndarray,
    ordinal_labels: Iterable[int],
    *,
    minimum_support: int = 2,
) -> dict[str, Any]:
    """Compute a class-balanced SNR inside one pre-specified row subset."""

    support = {
        int(label): int(np.sum(row_mask & (labels == int(label))))
        for label in ordinal_labels
    }
    retained = [label for label, count in support.items() if count >= minimum_support]
    excluded = [label for label, count in support.items() if count < minimum_support]
    if len(retained) < 2:
        return {
            "status": "insufficient_support",
            "minimum_support": int(minimum_support),
            "retained_labels": retained,
            "excluded_labels": excluded,
            "support": {str(key): value for key, value in support.items()},
            "rows": int(row_mask.sum()),
            "signal_power": math.nan,
            "noise_power": math.nan,
            "snr": math.nan,
            "snr_db": math.nan,
        }
    class_means = np.stack(
        [projected[row_mask & (labels == label)].mean(axis=0) for label in retained]
    )
    balanced_grand = class_means.mean(axis=0)
    signal_power = float(
        np.square(class_means - balanced_grand).sum(axis=1).mean()
    )
    class_noise = np.asarray(
        [
            np.square(
                projected[row_mask & (labels == label)] - class_means[index]
            )
            .sum(axis=1)
            .mean()
            for index, label in enumerate(retained)
        ],
        dtype=float,
    )
    noise_power = float(class_noise.mean())
    if noise_power <= np.finfo(float).eps or signal_power <= 0:
        snr = math.nan
        snr_db = math.nan
    else:
        snr = float(signal_power / noise_power)
        snr_db = float(10.0 * np.log10(snr))
    return {
        "status": "ok",
        "minimum_support": int(minimum_support),
        "retained_labels": retained,
        "excluded_labels": excluded,
        "support": {str(key): value for key, value in support.items()},
        "rows": int(row_mask.sum()),
        "signal_power": signal_power,
        "noise_power": noise_power,
        "snr": snr,
        "snr_db": snr_db,
    }


def band_conditioned_confirmation_snr(
    states: np.ndarray,
    metadata: pd.DataFrame,
    frozen_band: np.ndarray,
    ordinal_labels: Iterable[int] = ORDINAL_LABELS,
    *,
    pca_dim: int = 16,
    random_state: int = 0,
    minimum_support: int = 2,
) -> dict[str, Any]:
    """Measure count SNR within discovery-fitted bands on confirmation rows.

    Standardization/PCA16-whitening and the two-band assignment are both frozen
    before confirmation is evaluated.  The macro value averages signal and
    noise powers across valid bands, so the distance between the band means is
    excluded from the signal and noise terms.
    """

    labels = tuple(map(int, ordinal_labels))
    occurrence = metadata["occurrence"].to_numpy(dtype=int)
    label_mask = metadata["occurrence"].astype(int).isin(labels).to_numpy()
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy() & label_mask
    confirmation = (
        metadata["split"].astype(str).eq("confirmation").to_numpy() & label_mask
    )
    scaler = StandardScaler().fit(states[discovery].astype(np.float32))
    scaled = scaler.transform(states.astype(np.float32))
    components = min(
        int(pca_dim),
        int(discovery.sum() - len(labels)),
        int(scaled.shape[1]),
    )
    if components < 2:
        raise ValueError("Discovery rows support fewer than two PCA components")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        whiten=True,
        random_state=random_state,
    ).fit(scaled[discovery])
    projected = pca.transform(scaled)
    band_array = np.asarray(frozen_band, dtype=str)
    per_band = {
        name: _class_balanced_snr(
            projected,
            occurrence,
            confirmation & (band_array == name),
            labels,
            minimum_support=minimum_support,
        )
        for name in ("upper", "lower")
    }
    valid = [value for value in per_band.values() if value["status"] == "ok"]
    complete_label_coverage = all(
        value["retained_labels"] == list(labels) for value in per_band.values()
    )
    if valid:
        signal = float(np.mean([value["signal_power"] for value in valid]))
        noise = float(np.mean([value["noise_power"] for value in valid]))
        ratio = float(signal / noise) if noise > np.finfo(float).eps else math.nan
        db = float(10.0 * np.log10(ratio)) if ratio > 0 else math.nan
        status = "ok"
    else:
        signal = noise = ratio = db = math.nan
        status = "insufficient_support"
    return {
        "status": status,
        "band_fit_split": "discovery",
        "projection_fit_split": "discovery",
        "projection": f"StandardScaler + PCA{components}-whitened",
        "minimum_class_support_per_band": int(minimum_support),
        "per_band": per_band,
        "macro_within_band": {
            "status": status,
            "valid_band_count": int(len(valid)),
            "complete_label_coverage": complete_label_coverage,
            "signal_power": signal,
            "noise_power": noise,
            "snr": ratio,
            "snr_db": db,
        },
        "definition": (
            "Within each frozen band, average count-centroid separation around "
            "that band's class-balanced grand centroid and divide by the mean "
            "within-(band,count) residual power. Macro SNR is the ratio of the "
            "equal-band mean signal power to equal-band mean noise power."
        ),
        "caveat": (
            "Band membership is a post-hoc nuisance diagnostic derived from "
            "discovery hidden geometry. It is not a primary cross-mode metric."
        ),
    }


def categorical_association(
    frame: pd.DataFrame, column: str, *, band_column: str = "band"
) -> dict[str, Any]:
    values = frame[column].fillna("<missing>").astype(str)
    band = frame[band_column].astype(str)
    table = pd.crosstab(values, band)
    purity = float(table.max(axis=1).sum() / table.to_numpy().sum())
    return {
        "column": column,
        "levels": int(values.nunique()),
        "nmi": float(normalized_mutual_info_score(values, band)),
        "weighted_band_purity": purity,
        "counts": {
            str(index): {
                name: int(table.loc[index].get(name, 0))
                for name in ("upper", "lower")
            }
            for index in table.index
        },
    }


def numeric_association(
    frame: pd.DataFrame, column: str, *, band_column: str = "band"
) -> dict[str, Any]:
    values = pd.to_numeric(frame[column], errors="coerce")
    groups = {
        name: values.loc[frame[band_column].eq(name)].dropna().to_numpy(dtype=float)
        for name in ("upper", "lower")
    }
    upper = groups["upper"]
    lower = groups["lower"]
    pooled_denominator = max(len(upper) + len(lower) - 2, 1)
    pooled_variance = (
        max(len(upper) - 1, 0) * (float(np.var(upper, ddof=1)) if len(upper) > 1 else 0.0)
        + max(len(lower) - 1, 0) * (float(np.var(lower, ddof=1)) if len(lower) > 1 else 0.0)
    ) / pooled_denominator
    effect = (
        float((upper.mean() - lower.mean()) / math.sqrt(pooled_variance))
        if len(upper) and len(lower) and pooled_variance > 0
        else math.nan
    )
    return {
        "column": column,
        "upper_mean": float(upper.mean()) if len(upper) else math.nan,
        "lower_mean": float(lower.mean()) if len(lower) else math.nan,
        "standardized_mean_difference": effect,
        "available_rows": int(len(upper) + len(lower)),
    }


def load_trace_lookup(path: Path | None, request_ids: set[str]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row.get("request_id"))
            if request_id in request_ids:
                result[request_id] = row
    missing = sorted(request_ids - set(result))
    if missing:
        raise ValueError(f"Trace archive lacks request IDs: {missing[:5]}")
    return result


def site_metadata(
    capture_index: Path,
    *,
    site_kind: str,
    trace_archive: Path | None,
) -> pd.DataFrame:
    index_rows = read_jsonl(capture_index)
    index_rows.sort(key=lambda row: (int(row["seed"]), int(row["gold_count"])))
    traces = load_trace_lookup(
        trace_archive, {str(row["request_id"]) for row in index_rows}
    )
    result: list[dict[str, Any]] = []
    for row in index_rows:
        manifest = json.loads(
            (capture_index.parent / str(row["manifest_path"])).read_text(
                encoding="utf-8"
            )
        )
        parser = manifest["parser"]
        raw = str(traces.get(str(row["request_id"]), {}).get("raw_output_text", ""))
        output_token_ids = traces.get(str(row["request_id"]), {}).get(
            "output_token_ids", []
        )
        sites = sorted(
            (
                site
                for site in manifest["site_rows"]
                if str(site.get("site_kind")) == site_kind
                and site.get("occurrence") is not None
            ),
            key=lambda site: int(site["occurrence"]),
        )
        for site in sites:
            occurrence = int(site["occurrence"])
            item_index = occurrence - 1
            start = int(site.get("char_start") or 0)
            end = int(site.get("char_end") or start)
            endpoint = int(site["endpoint_token"])
            result.append(
                {
                    "split": str(row["split"]),
                    "seed": int(row["seed"]),
                    "gold_count": int(row["gold_count"]),
                    "occurrence": occurrence,
                    "request_id": str(row["request_id"]),
                    "stimulus_id": str(row["stimulus_id"]),
                    "trace_category": str(row.get("trace_category")),
                    "marker_kind": str(row.get("marker_kind")),
                    "exact_count": bool(row.get("exact_count")),
                    "parsed_count": row.get("parsed_count"),
                    "boundary_kind": str(site.get("boundary_kind")),
                    "alignment_strategy": str(site.get("alignment_strategy")),
                    "retokenized_suffix_tokens": int(
                        site.get("retokenized_suffix_tokens") or 0
                    ),
                    "endpoint_token": endpoint,
                    "prefix_token_count": int(site["prefix_token_count"]),
                    "shared_baseline_prefix_tokens": int(
                        site.get("shared_baseline_prefix_tokens") or 0
                    ),
                    "prompt_token_count": int(manifest["prompt_token_count"]),
                    "output_token_count": int(manifest["output_token_count"]),
                    "item_char_length": end - start,
                    "item_line_number": (
                        int(parser["item_line_numbers"][item_index])
                        if item_index < len(parser.get("item_line_numbers", []))
                        else None
                    ),
                    "city": str(site.get("city")),
                    "raw_suffix": " ".join(
                        raw[max(0, end - 24) : min(len(raw), end + 24)].split()
                    ),
                    "baseline_endpoint_token_id": (
                        int(output_token_ids[endpoint])
                        if endpoint < len(output_token_ids)
                        else None
                    ),
                }
            )
    return pd.DataFrame(result)


def trajectory_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in points.groupby(
        ["split", "seed", "gold_count", "request_id"], sort=True
    ):
        counts = Counter(group["band"].astype(str))
        majority, majority_count = counts.most_common(1)[0]
        rows.append(
            {
                "split": key[0],
                "seed": int(key[1]),
                "gold_count": int(key[2]),
                "request_id": key[3],
                "states": int(len(group)),
                "majority_band": majority,
                "band_purity": float(majority_count / len(group)),
                "upper_states": int(counts.get("upper", 0)),
                "lower_states": int(counts.get("lower", 0)),
                "marker_kind": str(group["marker_kind"].iloc[0]),
                "trace_category": str(group["trace_category"].iloc[0]),
                "output_token_count": int(group["output_token_count"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def metric_subset(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "confirmation_logistic_balanced_accuracy",
        "confirmation_ncc_balanced_accuracy",
        "confirmation_class_balanced_snr_db",
        "confirmation_signal_power",
        "confirmation_noise_power",
        "confirmation_rows",
        "confirmation_seed_count",
        "confirmation_support_min",
        "confirmation_support_max",
    )
    return {key: value[key] for key in keys}


def nuisance_band_summary(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    random_state: int,
) -> dict[str, Any]:
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy()
    confirmation = metadata["split"].astype(str).eq("confirmation").to_numpy()
    scaler = StandardScaler().fit(states[discovery].astype(np.float32))
    scaled = scaler.transform(states.astype(np.float32))
    pca = PCA(n_components=3, svd_solver="randomized", random_state=random_state).fit(
        scaled[discovery]
    )
    coordinates = pca.transform(scaled)
    raw_frozen = fit_discovery_frozen_bands(coordinates, discovery, random_state)
    raw_band = raw_frozen["band"]
    raw = summarize_frozen_bands(coordinates, raw_band, confirmation)
    frame = metadata.loc[confirmation].reset_index(drop=True).copy()
    frame["band"] = raw_band[confirmation]
    raw_marker_nmi = float(
        normalized_mutual_info_score(
            frame["marker_kind"].fillna("<missing>").astype(str), frame["band"]
        )
    )
    raw_occurrence_nmi = float(
        normalized_mutual_info_score(frame["occurrence"].astype(str), frame["band"])
    )
    raw_seed_nmi = float(
        normalized_mutual_info_score(frame["seed"].astype(str), frame["band"])
    )
    raw_trajectory_purity = float(
        frame.assign(_trajectory=trajectory_key(frame))
        .groupby("_trajectory")["band"]
        .apply(lambda values: values.value_counts().max() / len(values))
        .mean()
    )

    centered_states = center_within_trajectory(states, metadata)
    centered_scaler = StandardScaler().fit(centered_states[discovery])
    centered_scaled = centered_scaler.transform(centered_states)
    centered_pca = PCA(
        n_components=3, svd_solver="randomized", random_state=random_state
    ).fit(centered_scaled[discovery])
    centered_coordinates = centered_pca.transform(centered_scaled)
    centered_frozen = fit_discovery_frozen_bands(
        centered_coordinates, discovery, random_state
    )
    centered_band = centered_frozen["band"]
    centered = summarize_frozen_bands(
        centered_coordinates, centered_band, confirmation
    )
    frame["centered_band"] = centered_band[confirmation]
    centered_marker_nmi = float(
        normalized_mutual_info_score(
            frame["marker_kind"].fillna("<missing>").astype(str),
            frame["centered_band"],
        )
    )
    return {
        "confirmation_states": int(confirmation.sum()),
        "raw_silhouette": float(raw["silhouette"]),
        "raw_marker_kind_nmi": raw_marker_nmi,
        "raw_occurrence_nmi": raw_occurrence_nmi,
        "raw_seed_nmi": raw_seed_nmi,
        "raw_mean_trajectory_purity": raw_trajectory_purity,
        "centered_silhouette": float(centered["silhouette"]),
        "centered_marker_kind_nmi": centered_marker_nmi,
        "raw_vs_centered_band_nmi": float(
            normalized_mutual_info_score(frame["band"], frame["centered_band"])
        ),
    }


def site_layer_sensitivity(
    capture_index: Path,
    *,
    selected_layer: int,
    selected_site: str,
    selected_dataset: Any,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for site in SENSITIVITY_SITES:
        try:
            dataset = (
                selected_dataset
                if site == selected_site
                else load_native_thinking_capture(
                    capture_index, site_kind=site, cohort="parser_hit"
                )
            )
        except ValueError as error:
            rows.append(
                {
                    "sweep": "site_at_selected_layer",
                    "site_kind": site,
                    "layer": int(selected_layer),
                    "status": "unavailable",
                    "reason": str(error),
                }
            )
            continue
        if selected_layer not in dataset.states_by_layer:
            continue
        rows.append(
            {
                "sweep": "site_at_selected_layer",
                "site_kind": site,
                "layer": int(selected_layer),
                "status": "ok",
                "reason": "",
                **nuisance_band_summary(
                    np.asarray(dataset.states_by_layer[selected_layer]),
                    dataset.metadata.reset_index(drop=True),
                    random_state=random_state,
                ),
            }
        )
    for candidate_layer, values in sorted(selected_dataset.states_by_layer.items()):
        rows.append(
            {
                "sweep": "selected_site_across_layers",
                "site_kind": selected_site,
                "layer": int(candidate_layer),
                "status": "ok",
                "reason": "",
                **nuisance_band_summary(
                    np.asarray(values),
                    selected_dataset.metadata.reset_index(drop=True),
                    random_state=random_state,
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze(
    *,
    capture_index: Path,
    trace_archive: Path | None,
    layer: int,
    site_kind: str,
    output_dir: Path,
    random_state: int,
) -> dict[str, Path]:
    dataset = load_native_thinking_capture(
        capture_index, site_kind=site_kind, cohort="parser_hit"
    )
    if layer not in dataset.states_by_layer:
        raise ValueError(f"Layer {layer} is unavailable; have {sorted(dataset.states_by_layer)}")
    metadata = dataset.metadata.reset_index(drop=True)
    enriched = site_metadata(
        capture_index, site_kind=site_kind, trace_archive=trace_archive
    ).reset_index(drop=True)
    alignment_columns = ["split", "seed", "gold_count", "occurrence", "stimulus_id"]
    if not metadata[alignment_columns].equals(enriched[alignment_columns]):
        raise ValueError("Manifest-derived site metadata does not align with state rows")

    states = np.asarray(dataset.states_by_layer[layer], dtype=np.float32)
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy()
    confirmation = metadata["split"].astype(str).eq("confirmation").to_numpy()
    scaler = StandardScaler().fit(states[discovery])
    scaled = scaler.transform(states)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=random_state).fit(
        scaled[discovery]
    )
    coordinates = pca.transform(scaled)
    raw_frozen = fit_discovery_frozen_bands(coordinates, discovery, random_state)
    raw_band_all = raw_frozen["band"]
    raw_band = summarize_frozen_bands(coordinates, raw_band_all, confirmation)
    raw_band_full = summarize_frozen_bands(coordinates, raw_band_all)

    centered_states = center_within_trajectory(states, metadata)
    centered_scaler = StandardScaler().fit(centered_states[discovery])
    centered_scaled = centered_scaler.transform(centered_states)
    centered_pca = PCA(
        n_components=3, svd_solver="randomized", random_state=random_state
    ).fit(centered_scaled[discovery])
    centered_coordinates = centered_pca.transform(centered_scaled)
    centered_frozen = fit_discovery_frozen_bands(
        centered_coordinates, discovery, random_state
    )
    centered_band_all = centered_frozen["band"]
    centered_band = summarize_frozen_bands(
        centered_coordinates, centered_band_all, confirmation
    )
    centered_band_full = summarize_frozen_bands(
        centered_coordinates, centered_band_all
    )

    all_points = enriched.reset_index(drop=True).copy()
    all_points["pc1"] = coordinates[:, 0]
    all_points["pc2"] = coordinates[:, 1]
    all_points["pc3"] = coordinates[:, 2]
    all_points["display_vertical"] = display_vertical(coordinates)
    all_points["band"] = raw_band_all
    all_points["centered_pc1"] = centered_coordinates[:, 0]
    all_points["centered_pc2"] = centered_coordinates[:, 1]
    all_points["centered_pc3"] = centered_coordinates[:, 2]
    all_points["centered_band"] = centered_band_all
    points = all_points.loc[confirmation].reset_index(drop=True).copy()
    points["pc1"] = coordinates[confirmation, 0]
    points["pc2"] = coordinates[confirmation, 1]
    points["pc3"] = coordinates[confirmation, 2]
    points["display_vertical"] = display_vertical(coordinates[confirmation])
    points["band"] = raw_band_all[confirmation]
    points["centered_pc1"] = centered_coordinates[confirmation, 0]
    points["centered_pc2"] = centered_coordinates[confirmation, 1]
    points["centered_pc3"] = centered_coordinates[confirmation, 2]
    points["centered_band"] = centered_band_all[confirmation]
    trajectories = trajectory_summary(points)
    all_trajectories = trajectory_summary(all_points)

    categorical_columns = (
        "seed",
        "marker_kind",
        "trace_category",
        "boundary_kind",
        "alignment_strategy",
        "retokenized_suffix_tokens",
        "item_line_number",
        "baseline_endpoint_token_id",
        "city",
        "occurrence",
    )
    numeric_columns = (
        "occurrence",
        "endpoint_token",
        "prefix_token_count",
        "item_char_length",
        "output_token_count",
        "baseline_endpoint_token_id",
    )
    categorical = [categorical_association(points, column) for column in categorical_columns]
    categorical.sort(key=lambda row: (-row["nmi"], row["column"]))
    numeric = [numeric_association(points, column) for column in numeric_columns]
    numeric.sort(
        key=lambda row: (
            -abs(row["standardized_mean_difference"])
            if math.isfinite(row["standardized_mean_difference"])
            else math.inf,
            row["column"],
        )
    )
    centered_categorical = [
        categorical_association(points, column, band_column="centered_band")
        for column in categorical_columns
    ]
    centered_categorical.sort(key=lambda row: (-row["nmi"], row["column"]))
    full_categorical = [
        categorical_association(all_points, column)
        for column in categorical_columns
    ]
    full_categorical.sort(key=lambda row: (-row["nmi"], row["column"]))

    raw_metrics = confirmation_metrics(
        states,
        metadata,
        ORDINAL_LABELS,
        pca_dim=16,
        random_state=random_state,
        pca_whiten=True,
    )
    centered_metrics = confirmation_metrics(
        centered_states,
        metadata,
        ORDINAL_LABELS,
        pca_dim=16,
        random_state=random_state,
        pca_whiten=True,
    )
    conditioned_snr = band_conditioned_confirmation_snr(
        states,
        metadata,
        raw_band_all,
        ORDINAL_LABELS,
        pca_dim=16,
        random_state=random_state,
        minimum_support=2,
    )
    sensitivity = site_layer_sensitivity(
        capture_index,
        selected_layer=layer,
        selected_site=site_kind,
        selected_dataset=dataset,
        random_state=random_state,
    )
    selected_layer_sites = sensitivity.loc[
        sensitivity["sweep"].eq("site_at_selected_layer")
        & sensitivity["status"].eq("ok")
    ].sort_values("site_kind")
    selected_site_layers = sensitivity.loc[
        sensitivity["sweep"].eq("selected_site_across_layers")
    ].sort_values("layer")
    peak_marker_row = selected_site_layers.sort_values(
        ["raw_marker_kind_nmi", "raw_silhouette", "layer"],
        ascending=[False, False, True],
    ).iloc[0]
    report = {
        "schema_version": "native_geometry_band_diagnostic_v2_frozen_band",
        "model_label": dataset.model_label,
        "capture_index": str(capture_index.resolve()),
        "trace_archive": None if trace_archive is None else str(trace_archive.resolve()),
        "layer": int(layer),
        "site_kind": site_kind,
        "scope": {
            "trajectory_panel": (
                "registered counts "
                + ",".join(
                    map(str, sorted(metadata["gold_count"].astype(int).unique()))
                )
            ),
            "gold_counts": sorted(
                metadata["gold_count"].astype(int).unique().tolist()
            ),
            "full_trajectories": int(
                metadata[["split", "seed", "gold_count"]]
                .drop_duplicates()
                .shape[0]
            ),
            "discovery_trajectories": int(
                metadata.loc[discovery, ["split", "seed", "gold_count"]]
                .drop_duplicates()
                .shape[0]
            ),
            "confirmation_trajectories": int(
                metadata.loc[confirmation, ["split", "seed", "gold_count"]]
                .drop_duplicates()
                .shape[0]
            ),
            "discovery_states": int(discovery.sum()),
            "confirmation_states": int(confirmation.sum()),
            "full_states": int(len(metadata)),
        },
        "display_pca3": {
            "fit_split": "discovery",
            "band_fit_split": "discovery",
            "discovery_band_sizes": raw_frozen["discovery_cluster_sizes"],
            "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "confirmation_two_band": {
                key: value for key, value in raw_band.items() if key != "band"
            },
            "full_panel_two_band": {
                key: value
                for key, value in raw_band_full.items()
                if key != "band"
            },
        },
        "within_trajectory_centered_pca3": {
            "fit_split": "discovery after per-trajectory mean subtraction",
            "band_fit_split": "discovery after per-trajectory mean subtraction",
            "discovery_band_sizes": centered_frozen["discovery_cluster_sizes"],
            "explained_variance_ratio": [
                float(value) for value in centered_pca.explained_variance_ratio_
            ],
            "explained_variance_ratio_sum": float(
                centered_pca.explained_variance_ratio_.sum()
            ),
            "confirmation_two_band": {
                key: value for key, value in centered_band.items() if key != "band"
            },
            "full_panel_two_band": {
                key: value
                for key, value in centered_band_full.items()
                if key != "band"
            },
            "raw_vs_centered_band_nmi": float(
                normalized_mutual_info_score(points["band"], points["centered_band"])
            ),
            "categorical_associations": centered_categorical,
        },
        "hidden_space_variance": {
            "between_trajectory_fraction_after_discovery_scaling": between_trajectory_fraction(
                scaled, metadata
            ),
            "definition": "weighted between-trajectory sum of squares / total sum of squares",
        },
        "ordinal_decodability": {
            "raw": metric_subset(raw_metrics),
            "band_conditioned_confirmation_snr": conditioned_snr,
            "within_trajectory_centered_diagnostic": metric_subset(centered_metrics),
            "centering_caveat": (
                "Per-trajectory centering uses all observed states in each trajectory; "
                "it is a diagnostic nuisance removal, not a deployable causal estimator."
            ),
        },
        "trajectory_band_summary": {
            "trajectory_count": int(len(trajectories)),
            "mean_within_trajectory_band_purity": float(
                trajectories["band_purity"].mean()
            ),
            "fully_single_band_trajectories": int(
                trajectories["band_purity"].eq(1.0).sum()
            ),
            "rows": trajectories.to_dict(orient="records"),
        },
        "full_panel_trajectory_band_summary": {
            "trajectory_count": int(len(all_trajectories)),
            "mean_within_trajectory_band_purity": float(
                all_trajectories["band_purity"].mean()
            ),
            "fully_single_band_trajectories": int(
                all_trajectories["band_purity"].eq(1.0).sum()
            ),
        },
        "categorical_associations": categorical,
        "full_panel_categorical_associations": full_categorical,
        "numeric_associations": numeric,
        "site_layer_sensitivity": {
            "selected_layer_sites": selected_layer_sites.to_dict(orient="records"),
            "selected_site_layer_count": int(len(selected_site_layers)),
            "selected_site_peak_marker_nmi": {
                "layer": int(peak_marker_row["layer"]),
                "raw_marker_kind_nmi": float(
                    peak_marker_row["raw_marker_kind_nmi"]
                ),
                "raw_silhouette": float(peak_marker_row["raw_silhouette"]),
            },
        },
        "interpretation_guardrail": (
            "A high seed/format association or collapse after trajectory centering "
            "supports a trajectory-level offset explanation.  It does not by itself "
            "identify a mechanistic cause, and PCA3 separation is not evidence of a "
            "cleaner ordinal representation."
        ),
    }

    paths = {
        "audit": output_dir / "band_diagnostic.json",
        "points": output_dir / "confirmation_points.csv",
        "all_points": output_dir / "all_points.csv",
        "trajectories": output_dir / "confirmation_trajectories.csv",
        "all_trajectories": output_dir / "all_trajectories.csv",
        "sensitivity": output_dir / "site_layer_sensitivity.csv",
    }
    atomic_json(paths["audit"], report)
    atomic_csv(paths["points"], points)
    atomic_csv(paths["all_points"], all_points)
    atomic_csv(paths["trajectories"], trajectories)
    atomic_csv(paths["all_trajectories"], all_trajectories)
    atomic_csv(paths["sensitivity"], sensitivity)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--trace-archive", type=Path)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--site-kind", default="post_boundary")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    paths = analyze(
        capture_index=args.capture_index.resolve(),
        trace_archive=(
            None if args.trace_archive is None else args.trace_archive.resolve()
        ),
        layer=args.layer,
        site_kind=args.site_kind,
        output_dir=args.output_dir.resolve(),
        random_state=args.random_state,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
