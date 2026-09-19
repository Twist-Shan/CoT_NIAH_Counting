"""Discovery-fitted, confirmation-only 3D manifolds for V6 Enumeration.

This module is deliberately descriptive.  It consumes the exact inputs and
discovery-selected defaults sealed by ``native_aligned_representation`` and
never feeds visual coordinates back into model, layer, or claim selection.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from realistic_niah_v5.cross_mode_geometry import ModeDataset
from realistic_niah_v5.dual_endpoint_geometry import (
    load_native_thinking_final_count,
)

from realistic_niah_v6.native_aligned_representation import (
    FINAL_ALIGNMENT_COLUMNS,
    FINAL_SITE,
    RUNNING_ALIGNMENT_COLUMNS,
    RUNNING_SITE,
    Cell,
    _capture_index,
    _cell_slug,
    _expected_trajectory_keys,
    _key_digest,
    _key_tuples,
    _load_running_datasets,
    _subset_dataset,
)
from realistic_niah_v6.spec import MODEL_LABELS, PROMPT_MODES


SCHEMA_VERSION = "realistic_niah_v6_representation_manifold_v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_V3_AGGREGATE_MULTIHOP_REPARSE"
SOURCE_STATUS = "PASS_NATIVE_ANALYSIS_PATH_ALIGNED"
OUTPUT_STATUS = "PASS_DISCOVERY_FIT_CONFIRMATION_PROJECTION"
RANDOM_STATE = 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def project_layer(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    label_column: str,
) -> dict[str, Any]:
    """Fit StandardScaler/PCA3 on discovery and emit confirmation rows only."""

    matrix = np.asarray(states, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(metadata):
        raise ValueError("State matrix and metadata are not row aligned")
    required = {"split", "seed", label_column}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Manifold metadata lacks {missing}")
    discovery_mask = metadata["split"].astype(str).eq("discovery").to_numpy()
    confirmation_mask = metadata["split"].astype(str).eq("confirmation").to_numpy()
    if discovery_mask.sum() < 4 or confirmation_mask.sum() < 1:
        raise ValueError("Manifold split support is insufficient")
    discovery_states = matrix[discovery_mask]
    confirmation_states = matrix[confirmation_mask]
    scaler = StandardScaler(with_mean=True, with_std=True)
    discovery_standard = scaler.fit_transform(discovery_states)
    confirmation_standard = scaler.transform(confirmation_states)
    pca = PCA(
        n_components=3,
        whiten=False,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )
    discovery_coordinates = pca.fit_transform(discovery_standard)
    confirmation_coordinates = pca.transform(confirmation_standard)
    discovery_labels = metadata.loc[discovery_mask, label_column].to_numpy(float)
    axis_signs: list[int] = []
    for axis in range(3):
        correlation = float(
            np.corrcoef(discovery_coordinates[:, axis], discovery_labels)[0, 1]
        )
        sign = -1 if np.isfinite(correlation) and correlation < 0.0 else 1
        discovery_coordinates[:, axis] *= sign
        confirmation_coordinates[:, axis] *= sign
        axis_signs.append(sign)
    confirmation = metadata.loc[
        confirmation_mask, ["seed", label_column]
    ].reset_index(drop=True)
    rows = [
        [
            int(seed),
            int(label),
            round(float(point[0]), 6),
            round(float(point[1]), 6),
            round(float(point[2]), 6),
        ]
        for (seed, label), point in zip(
            confirmation.itertuples(index=False, name=None),
            confirmation_coordinates,
        )
    ]
    return {
        "evr": [round(float(value), 9) for value in pca.explained_variance_ratio_],
        "axis_signs": axis_signs,
        "rows": rows,
        "discovery_rows": int(discovery_mask.sum()),
        "confirmation_rows": int(confirmation_mask.sum()),
        "standard_scaler_zero_variance_features": int(
            np.count_nonzero(np.asarray(scaler.var_) == 0.0)
        ),
    }


def _selected_layers(path: Path) -> dict[Cell, int]:
    frame = pd.read_csv(path)
    result: dict[Cell, int] = {}
    for row in frame.to_dict(orient="records"):
        cell = (str(row["prompt_mode"]), str(row["model_label"]))
        if cell in result:
            raise ValueError(f"Duplicate selected layer for {_cell_slug(cell)}")
        result[cell] = int(row["layer"])
    expected = {
        (prompt_mode, model_label)
        for prompt_mode in PROMPT_MODES
        for model_label in MODEL_LABELS
    }
    if set(result) != expected:
        raise ValueError("Selected-layer table does not cover all four cells")
    return result


def _project_dataset(
    dataset: ModeDataset,
    *,
    label_column: str,
    default_layer: int,
) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer in sorted(dataset.states_by_layer):
        print(
            f"[manifold] {dataset.model_label} {label_column} layer={layer}",
            flush=True,
        )
        layers[str(int(layer))] = project_layer(
            np.asarray(dataset.states_by_layer[layer]),
            dataset.metadata,
            label_column=label_column,
        )
    if str(default_layer) not in layers:
        raise ValueError(f"Frozen default layer L{default_layer} is unavailable")
    first = layers[next(iter(layers))]
    if any(
        (value["discovery_rows"], value["confirmation_rows"])
        != (first["discovery_rows"], first["confirmation_rows"])
        for value in layers.values()
    ):
        raise ValueError("Layerwise manifold support changed")
    return {
        "default_layer": int(default_layer),
        "layers": layers,
        "discovery_rows": int(first["discovery_rows"]),
        "confirmation_rows": int(first["confirmation_rows"]),
    }


def export_representation_manifold(
    *,
    run_root: str | Path,
    output_dir: str | Path,
    protocol_path: str | Path,
    command: str | None = None,
) -> dict[str, Path]:
    run_root = Path(run_root).resolve()
    output = Path(output_dir).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = _read_json(protocol_path)
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("V3 protocol is not at the frozen checkpoint")
    contract = protocol["representation_3d"]
    source_root = run_root / str(contract["source_analysis_root"])
    source_manifest_path = source_root / "analysis_manifest.json"
    source_audit_path = source_root / "alignment_audit.json"
    if _sha256(source_manifest_path) != str(contract["source_analysis_manifest_sha256"]):
        raise ValueError("Native-aligned source manifest hash changed")
    if _sha256(source_audit_path) != str(contract["source_alignment_audit_sha256"]):
        raise ValueError("Native-aligned alignment audit hash changed")
    source_manifest = _read_json(source_manifest_path)
    source_audit = _read_json(source_audit_path)
    if (
        source_manifest.get("status") != SOURCE_STATUS
        or source_audit.get("status") != SOURCE_STATUS
        or source_manifest.get("confirmation_used_for_selection") is not False
    ):
        raise ValueError("Native-aligned source analysis is not sealed")
    for name, filename in (
        ("running_selected", "running_index_selected.csv"),
        ("final_selected", "final_count_selected.csv"),
    ):
        path = source_root / filename
        if _sha256(path) != str(source_manifest["outputs"][name]["sha256"]):
            raise ValueError(f"Native-aligned {name} hash changed")
    running_defaults = _selected_layers(source_root / "running_index_selected.csv")
    final_defaults = _selected_layers(source_root / "final_count_selected.csv")
    cells: tuple[Cell, ...] = tuple(
        (prompt_mode, model_label)
        for prompt_mode in PROMPT_MODES
        for model_label in MODEL_LABELS
    )

    running_raw, input_audits = _load_running_datasets(run_root, cells)
    for cell in cells:
        sealed = source_manifest["inputs"][_cell_slug(cell)]
        observed = input_audits[_cell_slug(cell)]
        if (
            observed["capture_index"]["sha256"]
            != sealed["capture_index"]["sha256"]
            or observed["capture_adapter"]["sha256"]
            != sealed["capture_adapter"]["sha256"]
        ):
            raise ValueError(f"Source capture changed for {_cell_slug(cell)}")
    common_running = set.intersection(
        *(
            set(_key_tuples(dataset.metadata, RUNNING_ALIGNMENT_COLUMNS))
            for dataset in running_raw.values()
        )
    )
    expected_running_digest = str(
        source_audit["running_index"]["common_key_sha256"]
    )
    if _key_digest(common_running) != expected_running_digest:
        raise ValueError("Running exact-common-support digest changed")
    running_aligned = {
        cell: _subset_dataset(
            dataset,
            common_running,
            columns=RUNNING_ALIGNMENT_COLUMNS,
        )
        for cell, dataset in running_raw.items()
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": OUTPUT_STATUS,
        "qualification": str(contract["qualification"]),
        "fit_split": "discovery",
        "display_split": "confirmation",
        "transform": str(contract["transform"]),
        "selection_firewall": str(contract["selection_firewall"]),
        "running": {},
        "final": {},
    }
    for cell in cells:
        prompt_mode, model_label = cell
        block = _project_dataset(
            running_aligned[cell],
            label_column="occurrence",
            default_layer=running_defaults[cell],
        )
        block.update(
            {
                "prompt_mode": prompt_mode,
                "model_label": model_label,
                "token_site": RUNNING_SITE,
                "label": "occurrence",
                "common_key_sha256": expected_running_digest,
            }
        )
        payload["running"][_cell_slug(cell)] = block
    del running_aligned, running_raw
    gc.collect()

    expected_final = _expected_trajectory_keys()
    expected_final_digest = str(
        source_audit["final_count"]["source_identity"]["key_sha256"]
    )
    if _key_digest(expected_final) != expected_final_digest:
        raise ValueError("Final registered-panel digest changed")
    for cell in cells:
        prompt_mode, model_label = cell
        print(f"[manifold] load final {_cell_slug(cell)}", flush=True)
        dataset = load_native_thinking_final_count(_capture_index(run_root, cell))
        keys = _key_tuples(dataset.metadata, FINAL_ALIGNMENT_COLUMNS)
        if len(keys) != 300 or set(keys) != expected_final:
            raise ValueError(f"Final exact panel changed for {_cell_slug(cell)}")
        block = _project_dataset(
            dataset,
            label_column="gold_count",
            default_layer=final_defaults[cell],
        )
        block.update(
            {
                "prompt_mode": prompt_mode,
                "model_label": model_label,
                "token_site": FINAL_SITE,
                "label": "gold_count",
                "trajectory_key_sha256": expected_final_digest,
            }
        )
        payload["final"][_cell_slug(cell)] = block
        del dataset
        gc.collect()

    payload_path = output / "representation_manifold.json"
    manifest_path = output / "representation_manifold_manifest.json"
    complete_path = output / "COMPLETE"
    _atomic_text(
        payload_path,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": OUTPUT_STATUS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "run_root": str(run_root),
        "output_dir": str(output),
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "status": protocol["status"],
        },
        "source_analysis_manifest": {
            "path": str(source_manifest_path),
            "sha256": _sha256(source_manifest_path),
        },
        "source_alignment_audit": {
            "path": str(source_audit_path),
            "sha256": _sha256(source_audit_path),
        },
        "output": {
            "path": str(payload_path),
            "sha256": _sha256(payload_path),
            "bytes": int(payload_path.stat().st_size),
        },
        "cell_count": len(cells),
        "running_layer_count_by_cell": {
            cell: len(value["layers"]) for cell, value in payload["running"].items()
        },
        "final_layer_count_by_cell": {
            cell: len(value["layers"]) for cell, value in payload["final"].items()
        },
        "confirmation_used_for_fit_or_selection": False,
        "new_model_forward_used": False,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": {
                name: _package_version(name)
                for name in ("numpy", "pandas", "scikit-learn")
            },
        },
    }
    _atomic_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(complete_path, "PASS\n")
    return {
        "payload": payload_path,
        "manifest": manifest_path,
        "complete": complete_path,
    }
