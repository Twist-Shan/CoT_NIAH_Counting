#!/usr/bin/env python3
"""Discovery-select layers for newly captured native-thinking phase sites.

The layer of each site is selected independently by grouped discovery-only
Logistic/NCC accuracy.  Confirmation then reports the frozen classifier and
covariance-aware geometry.  The script also exports raw-state and seed-by-count
mean PCA3 payloads; the latter is a labelled denoised visualization and never
replaces state-level metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v5.covariance_geometry import (  # noqa: E402
    class_balanced_scatter,
    evaluate_covariance_geometry_layer,
    regularized_precision,
)
from realistic_niah_v5.cross_mode_geometry import CLASSES, ModeDataset  # noqa: E402
from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)


SCHEMA = "realistic_niah_v5_native_phase_geometry_analysis_v1"
SITES = ("post_marker", "marker_end", "post_city")
MODELS = ("Qwen3-8B", "Gemma4-E4B")


def truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_phase_capture(capture_index: Path, site: str) -> ModeDataset:
    descriptors: list[tuple[dict[str, Any], list[int], list[dict[str, Any]]]] = []
    for index_row in read_jsonl(capture_index):
        manifest = json.loads(
            (capture_index.parent / str(index_row["manifest_path"])).read_text(
                encoding="utf-8"
            )
        )
        indices = [
            axis
            for axis, row in enumerate(manifest["site_rows"])
            if str(row["site"]) == site
        ]
        rows = [manifest["site_rows"][axis] for axis in indices]
        if not rows:
            continue
        keys = [int(row["occurrence"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate {site} event in {index_row['request_id']}")
        order = np.argsort(np.asarray(keys, dtype=int))
        descriptors.append(
            (
                {**index_row, "stimulus_id": manifest.get("stimulus_id")},
                np.asarray(indices, dtype=int)[order].tolist(),
                [rows[int(axis)] for axis in order],
            )
        )
    if not descriptors:
        raise ValueError(f"No {site} rows in {capture_index}")
    descriptors.sort(key=lambda item: (int(item[0]["seed"]), int(item[0]["gold_count"])))
    first = descriptors[0][0]
    with np.load(capture_index.parent / str(first["states_path"]), allow_pickle=False) as z:
        layers = z["layer_indices"].astype(int)
        hidden = int(z["site_states"].shape[-1])
    total = sum(len(indices) for _row, indices, _sites in descriptors)
    states = {int(layer): np.empty((total, hidden), dtype=np.float16) for layer in layers}
    metadata: list[dict[str, Any]] = []
    offset = 0
    for index_row, indices, sites in descriptors:
        with np.load(
            capture_index.parent / str(index_row["states_path"]), allow_pickle=False
        ) as z:
            now = z["layer_indices"].astype(int)
            if not np.array_equal(now, layers):
                raise ValueError("Phase capture layer grids differ")
            values = np.asarray(z["site_states"])[np.asarray(indices, dtype=int)]
            for layer_axis, layer in enumerate(layers):
                states[int(layer)][offset : offset + len(indices)] = values[:, layer_axis]
        for site_row in sites:
            metadata.append(
                {
                    "request_id": str(index_row["request_id"]),
                    "stimulus_id": str(index_row.get("stimulus_id") or index_row["request_id"]),
                    "split": str(index_row["split"]),
                    "seed": int(index_row["seed"]),
                    "gold_count": int(index_row["gold_count"]),
                    "occurrence": int(site_row["occurrence"]),
                    "site": site,
                    "city": str(site_row.get("city", "")),
                    "grammar_class": str(site_row.get("grammar_class", "")),
                    "surface_order": str(site_row.get("surface_order", "")),
                    "marker_kind": str(site_row.get("marker_kind", "")),
                    "causal_cohort": str(site_row.get("causal_cohort", "")),
                    "primary_full_chain_event": truthy(
                        site_row.get("primary_full_chain_event")
                    ),
                    "progress_commit_eligible": truthy(
                        site_row.get("progress_commit_eligible")
                    ),
                    "token_text": str(site_row.get("token_text", "")),
                    "token_surface_class": str(site_row.get("token_surface_class", "")),
                }
            )
        offset += len(indices)
    dataset = ModeDataset(
        mode="native_thinking",
        model_label=str(descriptors[0][0]["model_label"]),
        metadata=pd.DataFrame(metadata),
        states_by_layer=states,
    )
    dataset.validate()
    return dataset


def support_rows(dataset: ModeDataset, site: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("discovery", "confirmation"):
        frame = dataset.metadata.loc[dataset.metadata["split"].eq(split)]
        counts = frame["occurrence"].astype(int).value_counts()
        for label in CLASSES:
            rows.append(
                {
                    "model_label": dataset.model_label,
                    "site": site,
                    "split": split,
                    "occurrence": label,
                    "states": int(counts.get(label, 0)),
                    "seeds": int(frame.loc[frame["occurrence"].eq(label), "seed"].nunique()),
                }
            )
    return rows


def verify_support(dataset: ModeDataset, site: str) -> None:
    for split in ("discovery", "confirmation"):
        labels = set(
            dataset.metadata.loc[dataset.metadata["split"].eq(split), "occurrence"]
            .astype(int)
            .tolist()
        )
        missing = sorted(set(CLASSES) - labels)
        if missing:
            raise ValueError(f"{dataset.model_label}/{site}/{split} lacks {missing}")


def select_classification_layer(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        [
            "discovery_selection_score",
            "discovery_oof_ncc_balanced_accuracy",
            "discovery_oof_logistic_balanced_accuracy",
            "layer",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def fitted_coordinates(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    discovery = metadata["split"].eq("discovery").to_numpy()
    scaler = StandardScaler().fit(states[discovery].astype(np.float32))
    scaled = scaler.transform(states.astype(np.float32))
    count = min(components, int(discovery.sum() - len(CLASSES)), states.shape[1])
    pca = PCA(n_components=count, svd_solver="randomized", random_state=seed).fit(
        scaled[discovery]
    )
    return pca.transform(scaled), pca.explained_variance_ratio_


def compactness_payload(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    pca_dim: int,
    seed: int,
) -> dict[str, Any]:
    coordinates, evr = fitted_coordinates(
        states, metadata, components=pca_dim, seed=seed
    )
    discovery = metadata["split"].eq("discovery").to_numpy()
    confirmation = metadata["split"].eq("confirmation").to_numpy()
    discovery_y = metadata.loc[discovery, "occurrence"].to_numpy(dtype=int)
    confirmation_y = metadata.loc[confirmation, "occurrence"].to_numpy(dtype=int)
    scatter = class_balanced_scatter(coordinates[discovery], discovery_y, CLASSES)
    _precision, inverse_sqrt, _ridge, _condition = regularized_precision(scatter.within)
    discovery_z = coordinates[discovery] @ inverse_sqrt
    confirmation_z = coordinates[confirmation] @ inverse_sqrt
    centroids = {
        label: discovery_z[discovery_y == label].mean(axis=0) for label in CLASSES
    }
    class_rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    for label in CLASSES:
        own = centroids[label]
        candidates = [
            np.linalg.norm(own - centroids[other])
            for other in (label - 1, label + 1)
            if other in centroids
        ]
        adjacent_gap = float(min(candidates))
        radii = np.linalg.norm(confirmation_z[confirmation_y == label] - own, axis=1)
        radius = float(np.mean(radii))
        ratio = radius / adjacent_gap if adjacent_gap > 0 else np.nan
        ratios.append(ratio)
        class_rows.append(
            {
                "occurrence": label,
                "confirmation_states": int(len(radii)),
                "mean_radius_to_discovery_centroid": radius,
                "nearest_adjacent_discovery_centroid_gap": adjacent_gap,
                "radius_gap_ratio": ratio,
            }
        )
    return {
        "pca_dim": int(coordinates.shape[1]),
        "pca_evr_sum": float(evr.sum()),
        "metric_space": (
            "discovery-standardized PCA followed by discovery within-class "
            "covariance whitening"
        ),
        "class_balanced_radius_gap_ratio": float(np.nanmean(ratios)),
        "interpretation": (
            "lower is tighter: mean confirmation radius around its discovery "
            "count centroid divided by the nearest adjacent-count centroid gap"
        ),
        "classes": class_rows,
    }


def pca3_payload(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, Any]:
    xyz, evr = fitted_coordinates(states, metadata, components=3, seed=seed)
    point_rows = []
    for value, row in zip(xyz, metadata.itertuples(index=False)):
        point_rows.append(
            {
                "x": float(value[0]),
                "y": float(value[1]),
                "z": float(value[2]),
                "split": str(row.split),
                "seed": int(row.seed),
                "gold_count": int(row.gold_count),
                "occurrence": int(row.occurrence),
                "grammar_class": str(row.grammar_class),
                "token_surface_class": str(row.token_surface_class),
            }
        )
    frame = pd.DataFrame(point_rows)
    means = (
        frame.groupby(["split", "seed", "occurrence"], as_index=False)[["x", "y", "z"]]
        .mean()
        .to_dict("records")
    )
    return {
        "evr": evr.tolist(),
        "fit": "discovery-only StandardScaler + PCA3",
        "raw_points": point_rows,
        "seed_count_means": means,
        "warning": (
            "seed_count_means average across available N trajectories and are a "
            "denoised visualization only; every metric uses raw states"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=ROOT / "work/v5_native_phase_geometry",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/v5_native_phase_geometry",
    )
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"schema_version": SCHEMA, "models": {}}
    for model in args.models:
        capture_index = args.capture_root / model / "capture_index.jsonl"
        payload["models"][model] = {"sites": {}}
        for site in SITES:
            dataset = load_phase_capture(capture_index, site)
            verify_support(dataset, site)
            supports.extend(support_rows(dataset, site))
            for (surface, grammar), count in Counter(
                zip(
                    dataset.metadata["token_surface_class"].astype(str),
                    dataset.metadata["grammar_class"].astype(str),
                )
            ).items():
                surface_rows.append(
                    {
                        "model_label": model,
                        "site": site,
                        "token_surface_class": surface,
                        "grammar_class": grammar,
                        "states": int(count),
                    }
                )
            site_candidates: list[dict[str, Any]] = []
            for layer, states in sorted(dataset.states_by_layer.items()):
                metrics = {
                    **grouped_discovery_cv_metrics(
                        states,
                        dataset.metadata,
                        CLASSES,
                        pca_dim=args.pca_dim,
                        random_state=args.seed,
                        folds=args.folds,
                        pca_whiten=True,
                    ),
                    **confirmation_metrics(
                        states,
                        dataset.metadata,
                        CLASSES,
                        pca_dim=args.pca_dim,
                        random_state=args.seed,
                        pca_whiten=True,
                    ),
                }
                row = {
                    "model_label": model,
                    "site": site,
                    "layer": int(layer),
                    "states": int(len(dataset.metadata)),
                    "trajectories": int(dataset.metadata["request_id"].nunique()),
                    **metrics,
                }
                candidates.append(row)
                site_candidates.append(row)
            winner = select_classification_layer(pd.DataFrame(site_candidates)).to_dict()
            layer = int(winner["layer"])
            covariance = evaluate_covariance_geometry_layer(
                dataset.states_by_layer[layer],
                dataset.metadata,
                CLASSES,
                pca_dim=args.pca_dim,
                random_state=args.seed,
                discovery_cv_folds=args.folds,
            )
            winner.update({f"cov_{key}": value for key, value in covariance.items() if key != "metric_definitions"})
            compactness = compactness_payload(
                dataset.states_by_layer[layer],
                dataset.metadata,
                pca_dim=args.pca_dim,
                seed=args.seed,
            )
            winner["confirmation_radius_gap_ratio"] = compactness[
                "class_balanced_radius_gap_ratio"
            ]
            selected.append(winner)
            payload["models"][model]["sites"][site] = {
                "selected_layer": layer,
                "selection": (
                    "grouped discovery-only 5-fold mean Logistic/NCC balanced accuracy"
                ),
                "metrics": winner,
                "compactness": compactness,
                "pca3": pca3_payload(
                    dataset.states_by_layer[layer], dataset.metadata, seed=args.seed
                ),
            }
            print(
                model,
                site,
                f"L{layer}",
                f"confirmation Log/NCC={winner['confirmation_logistic_balanced_accuracy']:.3f}/"
                f"{winner['confirmation_ncc_balanced_accuracy']:.3f}",
                f"silhouette={winner['cov_confirmation_mahalanobis_silhouette']:.3f}",
                f"radius/gap={winner['confirmation_radius_gap_ratio']:.3f}",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates": args.output / "site_layer_candidates.csv",
        "selected": args.output / "site_selected.csv",
        "support": args.output / "site_support.csv",
        "surface": args.output / "site_surface_composition.csv",
        "payload": args.output / "geometry_payload.json",
    }
    atomic_csv(paths["candidates"], pd.DataFrame(candidates))
    atomic_csv(paths["selected"], pd.DataFrame(selected))
    atomic_csv(paths["support"], pd.DataFrame(supports))
    atomic_csv(paths["surface"], pd.DataFrame(surface_rows))
    paths["payload"].write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    audit = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sites": list(SITES),
        "selection": (
            "site fixed semantically; layer selected independently per model/site by "
            "grouped discovery-only 5-fold mean Logistic/NCC balanced accuracy; "
            "confirmation frozen"
        ),
        "post_marker_role": (
            "explicit rank-before-city lexical positive control; excluded from any "
            "claim about implicit counter abstraction"
        ),
        "post_city_role": (
            "first original baseline token after city-containing token; surface "
            "composition is audited because it is not format invariant"
        ),
        "pca_dim": args.pca_dim,
        "outputs": {str(path.resolve()): sha256(path) for path in paths.values()},
    }
    atomic_json(args.output / "audit.json", audit)


if __name__ == "__main__":
    main()
