from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.counter_channel import (  # noqa: E402
    count_axis,
    count_centroids,
    count_subspace,
    grouped_cross_layer_decode,
    load_layer_dataset,
    oriented_axis_angle_degrees,
    principal_angles_degrees,
    read_layer_manifest,
    subspace_overlap,
    subset_layer_dataset,
    unoriented_axis_angle_degrees,
)


def analyze(
    manifest: Path,
    output: Path,
    *,
    rank: int,
    folds: int,
    all_decode_pairs: bool,
    fit_splits: tuple[str, ...] | None,
    correct_only: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    manifest_rows = read_layer_manifest(manifest)
    raw_datasets = [load_layer_dataset(manifest, row) for row in manifest_rows]
    datasets = []
    for dataset in raw_datasets:
        mask = np.ones(len(dataset.count), dtype=bool)
        if fit_splits is not None:
            if "split" not in dataset.metadata:
                raise ValueError(
                    f"{dataset.source} has no split field required by --fit-splits"
                )
            mask &= dataset.metadata["split"].astype(str).isin(fit_splits).to_numpy()
        if correct_only:
            if "correct" not in dataset.metadata:
                raise ValueError(
                    f"{dataset.source} has no correct field required by --correct-only"
                )
            correct = dataset.metadata["correct"]
            if correct.dtype == bool:
                mask &= correct.to_numpy()
            else:
                mask &= correct.astype(str).str.lower().isin(
                    {"1", "true", "yes", "correct"}
                ).to_numpy()
        datasets.append(subset_layer_dataset(dataset, mask))
    keys = [(item.model_label, item.role, item.layer) for item in datasets]
    if len(set(keys)) != len(keys):
        raise RuntimeError("layer manifest contains duplicate model/role/layer datasets")

    output.mkdir(parents=True, exist_ok=True)
    basis_root = output / "bases"
    basis_root.mkdir(exist_ok=True)
    bases: dict[tuple[str, str, int], np.ndarray] = {}
    axes: dict[tuple[str, str, int], np.ndarray] = {}
    dataset_rows: list[dict[str, object]] = []
    for dataset in datasets:
        key = (dataset.model_label, dataset.role, dataset.layer)
        basis, explained = count_subspace(dataset.states, dataset.count, rank=rank)
        axis = count_axis(dataset.states, dataset.count)
        centroid_labels, centroids = count_centroids(dataset.states, dataset.count)
        center = dataset.states.astype(np.float64).mean(axis=0)
        bases[key] = basis
        axes[key] = axis
        safe_model = dataset.model_label.replace("/", "_")
        safe_role = dataset.role.replace("/", "_")
        basis_path = basis_root / f"{safe_model}__{safe_role}__L{dataset.layer:02d}.npz"
        np.savez_compressed(
            basis_path,
            basis=basis.astype(np.float32),
            axis=axis.astype(np.float32),
            explained=explained.astype(np.float64),
            center=center.astype(np.float32),
            centroid_labels=centroid_labels.astype(np.int64),
            centroids=centroids.astype(np.float32),
            layer=np.asarray(dataset.layer),
            fit_splits=np.asarray([] if fit_splits is None else fit_splits),
            correct_only=np.asarray(bool(correct_only)),
        )
        dataset_rows.append(
            {
                "model_label": dataset.model_label,
                "role": dataset.role,
                "layer": dataset.layer,
                "rows": len(dataset.count),
                "seeds": len(np.unique(dataset.seed)),
                "count_min": int(dataset.count.min()),
                "count_max": int(dataset.count.max()),
                "rank": rank,
                "centroid_variance_capture": float(explained.sum()),
                "basis_path": str(basis_path.relative_to(output).as_posix()),
                "source": str(dataset.source),
            }
        )

    overlap_rows: list[dict[str, object]] = []
    decode_rows: list[dict[str, object]] = []
    by_model: dict[str, list] = {}
    for dataset in datasets:
        by_model.setdefault(dataset.model_label, []).append(dataset)
    for model, model_datasets in by_model.items():
        model_datasets.sort(key=lambda item: (item.role, item.layer))
        for left_index, left in enumerate(model_datasets):
            for right in model_datasets[left_index + 1 :]:
                left_key = (left.model_label, left.role, left.layer)
                right_key = (right.model_label, right.role, right.layer)
                angles = principal_angles_degrees(bases[left_key], bases[right_key])
                same_role_adjacent = left.role == right.role and abs(left.layer - right.layer) == 1
                same_layer_cross_role = left.role != right.role and left.layer == right.layer
                overlap_rows.append(
                    {
                        "model_label": model,
                        "left_role": left.role,
                        "left_layer": left.layer,
                        "right_role": right.role,
                        "right_layer": right.layer,
                        "subspace_overlap": subspace_overlap(
                            bases[left_key], bases[right_key]
                        ),
                        "axis_angle_degrees": oriented_axis_angle_degrees(
                            axes[left_key], axes[right_key]
                        ),
                        "axis_line_angle_degrees": unoriented_axis_angle_degrees(
                            axes[left_key], axes[right_key]
                        ),
                        "principal_angle_mean_degrees": float(angles.mean()),
                        "principal_angle_max_degrees": float(angles.max()),
                        "same_role_adjacent": same_role_adjacent,
                        "same_layer_cross_role": same_layer_cross_role,
                    }
                )
                if all_decode_pairs or same_role_adjacent or same_layer_cross_role:
                    for source, target in ((left, right), (right, left)):
                        metrics = grouped_cross_layer_decode(
                            source, target, folds=folds
                        )
                        decode_rows.append(
                            {
                                "model_label": model,
                                "train_role": source.role,
                                "train_layer": source.layer,
                                "test_role": target.role,
                                "test_layer": target.layer,
                                **metrics,
                            }
                        )

    pd.DataFrame(dataset_rows).to_csv(output / "subspace_datasets.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output / "subspace_overlap.csv", index=False)
    pd.DataFrame(decode_rows).to_csv(output / "cross_layer_decode.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_counter_subspace_analysis_v1",
        "manifest": str(manifest.resolve()),
        "rank": rank,
        "folds": folds,
        "all_decode_pairs": all_decode_pairs,
        "fit_splits": None if fit_splits is None else list(fit_splits),
        "correct_only": correct_only,
        "datasets": len(datasets),
        "overlap_pairs": len(overlap_rows),
        "decode_directions": len(decode_rows),
        "models": sorted(by_model),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    (output / "subspace_analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cross-layer prompt/trace/answer counter subspaces on CPU"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--all-decode-pairs", action="store_true")
    parser.add_argument(
        "--fit-splits",
        nargs="+",
        help="Optional split labels used to fit every basis (for example discovery)",
    )
    parser.add_argument("--correct-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.manifest.resolve(),
                args.output.resolve(),
                rank=args.rank,
                folds=args.folds,
                all_decode_pairs=args.all_decode_pairs,
                fit_splits=None if args.fit_splits is None else tuple(args.fit_splits),
                correct_only=args.correct_only,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
