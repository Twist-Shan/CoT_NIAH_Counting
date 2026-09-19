#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


RUN_TIMESTAMP_RE = re.compile(r"^run_(\d{8}_\d{6})")
LAYER_RE = re.compile(r"_layer_(\d+)\.pt$")


@dataclass(frozen=True)
class FeatureVectorRecord:
    run_name: str
    run_timestamp: str
    method: str
    position: str | None
    layer: int
    vector_path: Path
    norm: float
    vector: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved counting feature vectors across NIAH run folders."
        )
    )
    parser.add_argument(
        "--parent-dir",
        type=Path,
        default=None,
        help=(
            "Parent NIAH_repo_and_local_runs directory. Defaults to the parent "
            "of the repository root."
        ),
    )
    parser.add_argument(
        "--output-dir-name",
        default="steering_run_analysis",
        help="Output folder name created under the parent directory.",
    )
    parser.add_argument(
        "--metadata-filename",
        default="counting_feature_vector_metadata.json",
        help="JSON output filename.",
    )
    parser.add_argument(
        "--heatmap-filename",
        default="counting_feature_vector_cosine_heatmap.png",
        help="Cosine-similarity heatmap filename.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_parent_dir() -> Path:
    return repo_root().parent


def run_timestamp(run_dir: Path) -> str:
    match = RUN_TIMESTAMP_RE.match(run_dir.name)
    return match.group(1) if match else ""


def layer_from_path(path: Path) -> int:
    match = LAYER_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse layer from vector path: {path}")
    return int(match.group(1))


def finite_vector(vector: torch.Tensor, *, path: Path) -> torch.Tensor:
    out = torch.as_tensor(vector, dtype=torch.float32).detach().cpu().flatten()
    if out.ndim != 1 or out.numel() == 0:
        raise ValueError(f"Vector must be non-empty and one-dimensional: {path}")
    if not torch.isfinite(out).all():
        raise ValueError(f"Vector contains non-finite values: {path}")
    return out


def vector_norm(vector: torch.Tensor, *, path: Path) -> float:
    norm = float(torch.linalg.vector_norm(vector).item())
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Vector has invalid norm {norm}: {path}")
    return norm


def load_mean_difference_vector(path: Path) -> tuple[torch.Tensor, float, str | None]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dictionary payload in vector file: {path}")
    if "direction" not in payload:
        raise ValueError(f"Missing 'direction' in vector file: {path}")
    vector = finite_vector(payload["direction"], path=path)
    if "raw_norm" in payload and payload["raw_norm"] is not None:
        norm = float(payload["raw_norm"])
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"Invalid raw_norm {norm}: {path}")
    elif "raw_direction" in payload and payload["raw_direction"] is not None:
        norm = vector_norm(finite_vector(payload["raw_direction"], path=path), path=path)
    else:
        norm = vector_norm(vector, path=path)
    position = payload.get("position")
    return vector, norm, None if position is None else str(position)


def load_ridge_vector(path: Path) -> tuple[torch.Tensor, float]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "coef" not in payload:
        raise ValueError(f"Expected ridge probe payload with 'coef': {path}")
    coef = finite_vector(payload["coef"], path=path)
    if bool(payload.get("standardize", False)) and payload.get("feature_scale") is not None:
        scale = finite_vector(payload["feature_scale"], path=path)
        if scale.shape != coef.shape:
            raise ValueError(
                f"feature_scale shape {tuple(scale.shape)} does not match coef "
                f"shape {tuple(coef.shape)} for {path}"
            )
        scale = torch.where(scale.abs() < 1e-6, torch.ones_like(scale), scale)
        hidden_vector = coef / scale
    else:
        hidden_vector = coef
    hidden_vector = finite_vector(hidden_vector, path=path)
    return hidden_vector, vector_norm(hidden_vector, path=path)


def iter_run_dirs(parent_dir: Path) -> list[Path]:
    if not parent_dir.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")
    return sorted(
        [path for path in parent_dir.iterdir() if path.is_dir() and path.name.startswith("run_")],
        key=lambda path: run_timestamp(path),
        reverse=True,
    )


def discover_vector_paths(run_dir: Path) -> list[tuple[str, str | None, Path]]:
    root = run_dir / "tensors" / "counting_features"
    if not root.exists():
        return []
    out: list[tuple[str, str | None, Path]] = []
    for path in sorted(root.glob("counterfactual/*/counterfactual_count_layer_*.pt")):
        out.append(("counterfactual", path.parent.name, path))
    for path in sorted(root.glob("contrastive_success/*/contrastive_success_layer_*.pt")):
        out.append(("contrastive-success", path.parent.name, path))
    for path in sorted(root.glob("ridge_probe_layer_*.pt")):
        out.append(("ridge", None, path))
    return out


def sort_key(record: FeatureVectorRecord) -> tuple[str, str, int]:
    position = "" if record.position is None else record.position
    return (record.method, position, record.layer)


def load_records(parent_dir: Path) -> tuple[list[FeatureVectorRecord], int]:
    records: list[FeatureVectorRecord] = []
    run_dirs = iter_run_dirs(parent_dir)
    for run_dir in run_dirs:
        timestamp = run_timestamp(run_dir)
        run_records: list[FeatureVectorRecord] = []
        for method, position_from_path, path in discover_vector_paths(run_dir):
            layer = layer_from_path(path)
            if method == "ridge":
                vector, norm = load_ridge_vector(path)
                position = None
            else:
                vector, norm, position_from_payload = load_mean_difference_vector(path)
                position = position_from_payload or position_from_path
            run_records.append(
                FeatureVectorRecord(
                    run_name=run_dir.name,
                    run_timestamp=timestamp,
                    method=method,
                    position=position,
                    layer=layer,
                    vector_path=path,
                    norm=norm,
                    vector=vector,
                )
            )
        records.extend(sorted(run_records, key=sort_key))
    return records, len(run_dirs)


def cosine_matrix(records: list[FeatureVectorRecord]) -> np.ndarray:
    if not records:
        raise ValueError("No counting feature vectors were found")
    dims = {int(record.vector.numel()) for record in records}
    if len(dims) != 1:
        by_dim: dict[int, list[str]] = {}
        for record in records:
            by_dim.setdefault(int(record.vector.numel()), []).append(
                f"{record.run_name}:{record.method}:{record.position}:L{record.layer}"
            )
        raise ValueError(
            "Cannot compare vectors with different hidden dimensions: "
            + json.dumps(by_dim, indent=2)
        )
    unit_rows = []
    for record in records:
        norm = vector_norm(record.vector, path=record.vector_path)
        unit_rows.append((record.vector / norm).numpy())
    matrix = np.stack(unit_rows, axis=0)
    cosine = matrix @ matrix.T
    if not np.isfinite(cosine).all():
        raise ValueError("Cosine-similarity matrix contains non-finite values")
    return cosine


def write_metadata(
    path: Path,
    *,
    parent_dir: Path,
    analysis_root: Path,
    records: list[FeatureVectorRecord],
    cosine: np.ndarray,
    heatmap_path: Path,
) -> None:
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_dir": str(parent_dir),
        "analysis_root": str(analysis_root),
        "num_vectors": len(records),
        "heatmap_path": str(heatmap_path),
        "vectors": [
            {
                "index": idx,
                "run_name": record.run_name,
                "run_timestamp": record.run_timestamp,
                "method": record.method,
                "position": record.position,
                "layer": record.layer,
                "vector_path": str(record.vector_path),
                "norm": record.norm,
            }
            for idx, record in enumerate(records, start=1)
        ],
        "cosine_similarity_matrix": cosine.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_heatmap(path: Path, cosine: np.ndarray) -> None:
    n = int(cosine.shape[0])
    labels = [str(i) for i in range(1, n + 1)]
    width = max(7.0, min(18.0, 0.35 * n + 4.0))
    height = max(6.0, min(18.0, 0.35 * n + 3.0))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(cosine, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title("Pairwise cosine similarity between saved counting feature vectors")
    ax.set_xlabel("vector index")
    ax.set_ylabel("vector index")
    ax.set_xticks(np.arange(n), labels=labels, rotation=90 if n > 15 else 0)
    ax.set_yticks(np.arange(n), labels=labels)
    fig.colorbar(image, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    parent_dir = (args.parent_dir or default_parent_dir()).resolve()
    analysis_root = parent_dir / args.output_dir_name
    analysis_root.mkdir(parents=True, exist_ok=True)

    records, num_run_dirs = load_records(parent_dir)
    cosine = cosine_matrix(records)
    metadata_path = analysis_root / args.metadata_filename
    heatmap_path = analysis_root / args.heatmap_filename
    write_metadata(
        metadata_path,
        parent_dir=parent_dir,
        analysis_root=analysis_root,
        records=records,
        cosine=cosine,
        heatmap_path=heatmap_path,
    )
    plot_heatmap(heatmap_path, cosine)

    print(f"Scanned run folders: {num_run_dirs}")
    print(f"Found counting feature vectors: {len(records)}")
    print(f"Metadata JSON: {metadata_path}")
    print(f"Cosine heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
