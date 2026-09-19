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
    benchmark_classifiers,
    classifier_oof_predictions,
    load_layer_dataset,
    read_layer_manifest,
    subset_layer_dataset,
)


DEFAULT_ALGORITHMS = (
    "logistic_l2",
    "ridge_classifier",
    "linear_svm",
    "rbf_svm",
    "nearest_centroid",
    "shrinkage_lda",
    "gaussian_nb",
    "knn_k1_euclidean",
    "knn_k3_euclidean",
    "knn_k5_euclidean",
    "knn_k7_euclidean",
    "knn_k9_euclidean",
    "knn_k3_cosine",
    "knn_k5_cosine",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped answer-query hidden-state count classification benchmark"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=["answer_query"])
    parser.add_argument("--models", nargs="+", help="Optional model-label subset")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="Optional layer subset; omit for the full layer sweep",
    )
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--correct-only", action="store_true")
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--prediction-algorithms", nargs="+", default=["logistic_l2", "knn_k5_cosine", "extra_trees"])
    args = parser.parse_args()

    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    rows = read_layer_manifest(manifest)
    raw_datasets = [
        load_layer_dataset(manifest, row)
        for row in rows
        if str(row["role"]) in set(args.roles)
        and (args.models is None or str(row["model_label"]) in set(args.models))
        and (args.layers is None or int(row["layer"]) in set(args.layers))
    ]
    datasets = []
    for dataset in raw_datasets:
        mask = np.ones(len(dataset.count), dtype=bool)
        if args.splits is not None:
            if "split" not in dataset.metadata:
                raise ValueError(f"{dataset.source} has no split metadata")
            mask &= dataset.metadata["split"].astype(str).isin(args.splits).to_numpy()
        if args.correct_only:
            if "correct" not in dataset.metadata:
                raise ValueError(f"{dataset.source} has no correct metadata")
            correct = dataset.metadata["correct"]
            if correct.dtype == bool:
                mask &= correct.to_numpy()
            else:
                mask &= correct.astype(str).str.lower().isin(
                    {"1", "true", "yes", "correct"}
                ).to_numpy()
        datasets.append(subset_layer_dataset(dataset, mask))
    if not datasets:
        raise RuntimeError(f"No datasets matched roles={args.roles}")
    metrics: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for dataset in datasets:
        metrics.append(
            benchmark_classifiers(
                dataset,
                algorithms=args.algorithms,
                folds=args.folds,
                pca_components=args.pca_components,
                n_jobs=args.n_jobs,
            )
        )
        for algorithm in args.prediction_algorithms:
            predictions.append(
                classifier_oof_predictions(
                    dataset,
                    algorithm,
                    folds=args.folds,
                    pca_components=args.pca_components,
                    n_jobs=args.n_jobs,
                )
            )
        print(
            f"[classification] {dataset.model_label} {dataset.role} L{dataset.layer}",
            flush=True,
        )
    metric_frame = pd.concat(metrics, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame.to_csv(output / "answer_classifier_metrics.csv", index=False)
    prediction_frame.to_csv(
        output / "answer_classifier_oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    confusion = (
        prediction_frame.groupby(
            ["model_label", "role", "layer", "algorithm", "gold_count", "predicted_count"],
            dropna=False,
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    confusion.to_csv(output / "answer_classifier_confusion.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_answer_classification_v1",
        "manifest": str(manifest),
        "roles": args.roles,
        "models": args.models,
        "layers": args.layers,
        "splits": args.splits,
        "correct_only": args.correct_only,
        "algorithms": args.algorithms,
        "prediction_algorithms": args.prediction_algorithms,
        "folds": args.folds,
        "pca_components": args.pca_components,
        "datasets": len(datasets),
        "metric_rows": len(metric_frame),
        "prediction_rows": len(prediction_frame),
        "split_unit": "seed",
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    (output / "answer_classifier_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
