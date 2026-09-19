#!/usr/bin/env python3
"""Compare running-index and answer-token geometry across entity domains.

Each model x mode x endpoint keeps its own discovery-selected layer and its own
city-discovery PCA/probe basis.  Flower and animal are frozen confirmation-only
transfers.  The two endpoints therefore compare ordering, overlap, and frozen
decodability within a mode; their axes and absolute distances are not shared.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    CLASSES,
    load_native_thinking_capture,
    load_non_thinking_capture,
)
from realistic_niah_v5.dual_endpoint_geometry import (  # noqa: E402
    load_native_thinking_final_count,
    load_non_thinking_final_count,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
MODES = ("non_thinking", "native_thinking")
ENDPOINTS = ("running_index", "answer_token")
SCHEMA = "realistic_niah_domain_endpoint_comparison_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nonthinking_indices(root: Path, model: str) -> tuple[Path, Path]:
    model_root = root / model / "numeric/representation"
    return (
        model_root / "capture/capture_index.jsonl",
        model_root / "answer_query_all_layers_v1/capture_index.jsonl",
    )


def selected_layers(dual_root: Path, domain_payload: Path) -> dict[tuple[str, str, str], int]:
    result: dict[tuple[str, str, str], int] = {}
    for model in MODELS:
        table_path = dual_root / model / "pca16_whiten/running_index_selected.csv"
        frame = pd.read_csv(table_path)
        frame = frame.loc[frame["analysis_group"].astype(str).eq("all_traces")]
        for mode in MODES:
            rows = frame.loc[frame["mode"].astype(str).eq(mode)]
            if len(rows) != 1:
                raise ValueError(f"Expected one running winner for {model}/{mode}; got {len(rows)}")
            result[(model, mode, "running_index")] = int(rows.iloc[0]["layer"])
    payload = json.loads(domain_payload.read_text(encoding="utf-8"))
    for model in MODELS:
        for mode in MODES:
            result[(model, mode, "answer_token")] = int(
                payload["models"][model][mode]["selected_layer"]
            )
    return result


def load_city_selected(
    *,
    model: str,
    mode: str,
    endpoint: str,
    layer: int,
    nonthinking_root: Path,
    native_running_root: Path,
    native_final_root: Path,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    if mode == "non_thinking":
        running_index, answer_index = nonthinking_indices(nonthinking_root, model)
        if endpoint == "running_index":
            dataset = load_non_thinking_capture(
                running_index, design_variant="v4.4", pooling="span_end"
            )
            site = "span_end"
        else:
            dataset = load_non_thinking_final_count(answer_index, design_variant="v4.4")
            site = "answer_query"
    else:
        if endpoint == "running_index":
            dataset = load_native_thinking_capture(
                native_running_root / model / "capture_index.jsonl",
                site_kind="item_end",
                site_policy="uniform",
                cohort="parser_hit",
            )
            site = "item_end"
        else:
            dataset = load_native_thinking_final_count(
                native_final_root / model / "capture_index.jsonl"
            )
            site = "answer_query_v3"
    if layer not in dataset.states_by_layer:
        raise ValueError(f"Layer {layer} absent for {model}/{mode}/{endpoint}")
    metadata = dataset.metadata.copy()
    metadata["entity_domain"] = "city"
    metadata["trajectory_id"] = metadata["stimulus_id"].astype(str)
    metadata["label"] = (
        metadata["occurrence"].astype(int)
        if endpoint == "running_index"
        else metadata["gold_count"].astype(int)
    )
    states = np.asarray(dataset.states_by_layer[layer], dtype=np.float32).copy()
    del dataset
    gc.collect()
    return metadata.reset_index(drop=True), states, site


def transfer_specs(mode: str, layers: dict[str, int]) -> dict[str, dict[str, Any]]:
    if mode == "non_thinking":
        return {
            "running_index": {"site_kind": "running_index", "layer": layers["running_index"]},
            "answer_token": {"site_kind": "answer_query", "layer": layers["answer_token"]},
        }
    return {
        "running_index": {"site_kind": "item_end", "layer": layers["running_index"]},
        "answer_token": {"site_kind": "answer_query_v3", "layer": layers["answer_token"]},
    }


def load_transfer_pair(
    index_path: Path,
    *,
    mode: str,
    layers: dict[str, int],
) -> dict[str, tuple[pd.DataFrame, np.ndarray, str]]:
    specs = transfer_specs(mode, layers)
    site_to_endpoint = {value["site_kind"]: endpoint for endpoint, value in specs.items()}
    raw = [row for row in read_jsonl(index_path) if str(row.get("site_kind")) in site_to_endpoint]
    if not raw:
        raise ValueError(f"No requested transfer sites in {index_path}")

    ordered: dict[str, list[dict[str, Any]]] = {}
    for endpoint, spec in specs.items():
        rows = [row for row in raw if str(row["site_kind"]) == spec["site_kind"]]
        rows.sort(
            key=lambda row: (
                str(row["entity_domain"]),
                int(row["seed"]),
                int(row["gold_count"]),
                -1 if row.get("occurrence") is None else int(row["occurrence"]),
            )
        )
        ordered[endpoint] = rows

    positions: dict[tuple[str, int], tuple[str, int]] = {}
    grouped: dict[str, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for endpoint, rows in ordered.items():
        for output_index, row in enumerate(rows):
            key = (endpoint, id(row))
            positions[key] = (endpoint, output_index)
            grouped[str(row["states_path"])].append((endpoint, output_index, row))

    first_path = index_path.parent / next(iter(grouped))
    with np.load(first_path, allow_pickle=False) as archive:
        hidden_size = int(np.asarray(archive["site_states"]).shape[-1])
    values = {
        endpoint: np.empty((len(rows), hidden_size), dtype=np.float32)
        for endpoint, rows in ordered.items()
    }
    for relative_path, descriptors in grouped.items():
        path = index_path.parent / relative_path
        with np.load(path, allow_pickle=False) as archive:
            layer_indices = np.asarray(archive[descriptors[0][2]["layer_array_key"]]).astype(int)
            state_array = np.asarray(archive[descriptors[0][2]["state_array_key"]])
            for endpoint, output_index, row in descriptors:
                layer = int(specs[endpoint]["layer"])
                axes = np.flatnonzero(layer_indices == layer)
                if len(axes) != 1:
                    raise ValueError(f"Layer {layer} absent or duplicated in {path}")
                values[endpoint][output_index] = state_array[
                    int(row["state_axis"]), int(axes[0])
                ]

    result: dict[str, tuple[pd.DataFrame, np.ndarray, str]] = {}
    for endpoint, rows in ordered.items():
        metadata_rows = []
        for row in rows:
            label = (
                int(row["occurrence"])
                if endpoint == "running_index"
                else int(row["gold_count"])
            )
            trajectory_id = str(
                row.get("request_id") or row.get("stimulus_id") or row.get("source_stimulus_id")
            )
            metadata_rows.append(
                {
                    "split": str(row["split"]),
                    "seed": int(row["seed"]),
                    "occurrence": (
                        int(row["occurrence"]) if row.get("occurrence") is not None else label
                    ),
                    "gold_count": int(row["gold_count"]),
                    "label": label,
                    "entity_domain": str(row["entity_domain"]),
                    "trajectory_id": trajectory_id,
                }
            )
        result[endpoint] = (
            pd.DataFrame(metadata_rows),
            values[endpoint],
            str(specs[endpoint]["site_kind"]),
        )
    return result


def nearest_centroid_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    classes = np.asarray(CLASSES, dtype=int)
    centroids = np.stack([train_x[train_y == label].mean(axis=0) for label in classes])
    distances = ((test_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return classes[np.argmin(distances, axis=1)]


def analyse_endpoint(
    city_metadata: pd.DataFrame,
    city_states: np.ndarray,
    transfer_metadata: pd.DataFrame,
    transfer_states: np.ndarray,
    *,
    model: str,
    mode: str,
    endpoint: str,
    layer: int,
    city_site: str,
    transfer_site: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    discovery = city_metadata["split"].astype(str).eq("discovery").to_numpy()
    confirmation = city_metadata["split"].astype(str).eq("confirmation").to_numpy()
    discovery_y = city_metadata.loc[discovery, "label"].to_numpy(dtype=int)
    if set(discovery_y) != set(CLASSES):
        raise ValueError(f"City discovery does not cover k=1..10 for {model}/{mode}/{endpoint}")

    scaler = StandardScaler().fit(city_states[discovery])
    discovery_scaled = scaler.transform(city_states[discovery])
    pca3 = PCA(n_components=3, random_state=seed).fit(discovery_scaled)
    pca16 = PCA(n_components=16, whiten=True, random_state=seed).fit(discovery_scaled)
    discovery_z = pca16.transform(discovery_scaled)
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=4000,
        random_state=seed,
        solver="lbfgs",
    ).fit(discovery_z, discovery_y)

    domain_frames = {
        "city": (city_metadata.loc[confirmation].reset_index(drop=True), city_states[confirmation]),
    }
    for domain in ("flower", "animal"):
        mask = transfer_metadata["entity_domain"].astype(str).eq(domain).to_numpy()
        domain_frames[domain] = (
            transfer_metadata.loc[mask].reset_index(drop=True), transfer_states[mask]
        )

    points: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for domain, (metadata, states) in domain_frames.items():
        scaled = scaler.transform(states)
        xyz = pca3.transform(scaled)
        z16 = pca16.transform(scaled)
        labels = metadata["label"].to_numpy(dtype=int)
        logistic_prediction = logistic.predict(z16)
        ncc_prediction = nearest_centroid_predict(discovery_z, discovery_y, z16)
        support = {str(label): int(np.sum(labels == label)) for label in CLASSES}
        metric = {
            "states": int(len(metadata)),
            "trajectories": int(metadata["trajectory_id"].nunique()),
            "support": support,
            "logistic_balanced_accuracy": float(
                balanced_accuracy_score(labels, logistic_prediction)
            ),
            "ncc_balanced_accuracy": float(balanced_accuracy_score(labels, ncc_prediction)),
        }
        metrics[domain] = metric
        metric_rows.append(
            {
                "model_label": model,
                "mode": mode,
                "endpoint": endpoint,
                "layer": layer,
                "city_site": city_site,
                "transfer_site": transfer_site,
                "entity_domain": domain,
                "states": metric["states"],
                "trajectories": metric["trajectories"],
                "support_min": min(support.values()),
                "support_max": max(support.values()),
                "logistic_balanced_accuracy": metric["logistic_balanced_accuracy"],
                "ncc_balanced_accuracy": metric["ncc_balanced_accuracy"],
            }
        )
        for value, row in zip(xyz, metadata.itertuples(index=False)):
            points.append(
                {
                    "domain": domain,
                    "count": int(row.label),
                    "seed": int(row.seed),
                    "gold_count": int(row.gold_count),
                    "occurrence": int(row.occurrence),
                    "trajectory_id": str(row.trajectory_id),
                    "x": float(value[0]),
                    "y": float(value[1]),
                    "z": float(value[2]),
                }
            )

    payload = {
        "layer": int(layer),
        "city_site": city_site,
        "transfer_site": transfer_site,
        "label_semantics": "running occurrence k" if endpoint == "running_index" else "gold final count N",
        "pca_fit": "city discovery only; StandardScaler then PCA3",
        "pca3_explained_variance_ratio": pca3.explained_variance_ratio_.tolist(),
        "probe_fit": "city discovery only; StandardScaler + whitened PCA16",
        "metrics": metrics,
        "points": points,
    }
    return payload, metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nonthinking-root",
        type=Path,
        default=ROOT / "work/nonthinking_v44_geometry_300_150_136_166_78",
    )
    parser.add_argument(
        "--native-running-root",
        type=Path,
        default=ROOT / "work/v5_geometry_full_panel/running",
    )
    parser.add_argument(
        "--native-final-root",
        type=Path,
        default=ROOT / "work/v5_geometry_full_panel/final",
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=ROOT / "work/domain_transfer_geometry/full",
    )
    parser.add_argument(
        "--dual-root",
        type=Path,
        default=ROOT / "reports/v5_dual_endpoint_geometry_full300",
    )
    parser.add_argument(
        "--answer-domain-payload",
        type=Path,
        default=ROOT / "work/domain_transfer_geometry/analysis/report_payload.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/v5_domain_endpoint_comparison",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = selected_layers(args.dual_root, args.answer_domain_payload)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "design": (
            "within each model/mode, running and answer use independently selected layers "
            "and independently fitted city-discovery PCA/probe bases"
        ),
        "models": {},
    }
    metric_rows: list[dict[str, Any]] = []
    input_paths: list[Path] = [args.answer_domain_payload]

    for model in MODELS:
        payload["models"][model] = {}
        for mode in MODES:
            endpoint_layers = {
                endpoint: layers[(model, mode, endpoint)] for endpoint in ENDPOINTS
            }
            transfer_folder = "nonthinking" if mode == "non_thinking" else "native"
            transfer_index = args.transfer_root / transfer_folder / model / "site_index.jsonl"
            transfer = load_transfer_pair(
                transfer_index,
                mode=mode,
                layers=endpoint_layers,
            )
            payload["models"][model][mode] = {}
            input_paths.append(transfer_index)
            for endpoint in ENDPOINTS:
                layer = endpoint_layers[endpoint]
                city_metadata, city_states, city_site = load_city_selected(
                    model=model,
                    mode=mode,
                    endpoint=endpoint,
                    layer=layer,
                    nonthinking_root=args.nonthinking_root,
                    native_running_root=args.native_running_root,
                    native_final_root=args.native_final_root,
                )
                transfer_metadata, transfer_states, transfer_site = transfer[endpoint]
                endpoint_payload, rows = analyse_endpoint(
                    city_metadata,
                    city_states,
                    transfer_metadata,
                    transfer_states,
                    model=model,
                    mode=mode,
                    endpoint=endpoint,
                    layer=layer,
                    city_site=city_site,
                    transfer_site=transfer_site,
                    seed=args.seed,
                )
                payload["models"][model][mode][endpoint] = endpoint_payload
                metric_rows.extend(rows)
                summary = endpoint_payload["metrics"]
                print(
                    model,
                    mode,
                    endpoint,
                    f"L{layer}",
                    " | ".join(
                        f"{domain}={value['logistic_balanced_accuracy']:.3f}/"
                        f"{value['ncc_balanced_accuracy']:.3f}"
                        for domain, value in summary.items()
                    ),
                )
                del city_states, transfer_states
                gc.collect()

            if mode == "non_thinking":
                running_index, answer_index = nonthinking_indices(args.nonthinking_root, model)
                input_paths.extend([running_index, answer_index])
            else:
                input_paths.extend(
                    [
                        args.native_running_root / model / "capture_index.jsonl",
                        args.native_final_root / model / "capture_index.jsonl",
                    ]
                )
            input_paths.append(
                args.dual_root / model / "pca16_whiten/running_index_selected.csv"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "domain_endpoint_metrics.csv"
    payload_path = args.output / "geometry_payload.json"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    outputs = [metrics_path, payload_path]
    audit = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": (
            "running layers reuse the all-trace discovery winners from the main report; "
            "answer layers reuse city-discovery winners from the domain-transfer analysis"
        ),
        "visualization": (
            "each model x mode x endpoint uses an independent StandardScaler/PCA3 fitted "
            "only on city discovery; all displayed points are confirmation"
        ),
        "labels": {
            "running_index": "each parser-observed occurrence k; ragged states are not padded",
            "answer_token": "one state per trajectory, labelled by gold final N",
        },
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in sorted(set(input_paths), key=str)
            if path.exists()
        },
        "outputs": {str(path.resolve()): sha256(path) for path in outputs},
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
