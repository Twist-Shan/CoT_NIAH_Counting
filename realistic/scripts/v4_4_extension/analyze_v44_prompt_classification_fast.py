from __future__ import annotations

"""Efficient six-classifier layer sweep for prompt running-index states.

PCA and scaling are fit once per held-out seed fold and shared by all six
algorithms.  This is numerically the same leakage boundary as fitting a
separate pipeline for every algorithm but avoids repeating the expensive SVD.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ALGORITHMS = (
    "logistic_l2", "ridge_classifier", "linear_svm",
    "nearest_centroid", "shrinkage_lda", "knn_k5_cosine",
)


def classifier(name: str):
    return {
        "logistic_l2": LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=20260806),
        "ridge_classifier": RidgeClassifier(alpha=1.0),
        "linear_svm": LinearSVC(C=1.0, dual="auto", random_state=20260806, max_iter=10000),
        "nearest_centroid": NearestCentroid(),
        "shrinkage_lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "knn_k5_cosine": KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="uniform"),
    }[name]


def seed_bootstrap(values: pd.DataFrame, reps: int = 10000) -> tuple[float, float]:
    per_seed = values.groupby("seed")["correct"].mean().to_numpy(float)
    rng = np.random.default_rng(20260806)
    means = per_seed[rng.integers(0, len(per_seed), size=(reps, len(per_seed)))].mean(axis=1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pca-components", type=int, default=32)
    args = parser.parse_args(); started = time.perf_counter(); args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [row for row in manifest["datasets"] if row["model_label"] == args.model and row["role"] == "prompt_running"]
    prediction_rows = []
    metric_rows = []
    for row in rows:
        path = args.manifest.parent / row["path"]
        with np.load(path, allow_pickle=False) as z:
            x = np.asarray(z["states"], dtype=np.float32)
            y = np.asarray(z["count"], dtype=int)
            seeds = np.asarray(z["seed"], dtype=int)
        oof = {name: np.empty(len(y), dtype=int) for name in ALGORITHMS}
        folds = np.empty(len(y), dtype=int)
        splitter = GroupKFold(n_splits=5)
        for fold, (train, test) in enumerate(splitter.split(x, y, groups=seeds)):
            pca = PCA(n_components=min(args.pca_components, len(train) - 1), svd_solver="randomized", random_state=20260806)
            z_train = pca.fit_transform(x[train]); z_test = pca.transform(x[test])
            scaler = StandardScaler(); z_train = scaler.fit_transform(z_train); z_test = scaler.transform(z_test)
            for name in ALGORITHMS:
                model = classifier(name); model.fit(z_train, y[train]); oof[name][test] = model.predict(z_test)
            folds[test] = fold
        for name in ALGORITHMS:
            predicted = oof[name]
            frame = pd.DataFrame({"seed": seeds, "correct": predicted == y})
            low, high = seed_bootstrap(frame)
            seed_acc = frame.groupby("seed")["correct"].mean().to_numpy(float)
            metric_rows.append({
                "model_label": args.model, "role": "prompt_running", "layer": int(row["layer"]), "algorithm": name,
                "rows": len(y), "seeds": len(np.unique(seeds)), "count_classes": json.dumps(sorted(np.unique(y).tolist())),
                "count_class_count": len(np.unique(y)), "accuracy": float(accuracy_score(y, predicted)),
                "accuracy_ci95_low": low, "accuracy_ci95_high": high, "seed_accuracy_sd": float(seed_acc.std(ddof=1)),
                "chance_accuracy": 1.0 / len(np.unique(y)), "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
                "count_mae": float(np.mean(np.abs(predicted - y))), "signed_error": float(np.mean(predicted - y)),
                "pca_components": args.pca_components,
            })
            for index in range(len(y)):
                prediction_rows.append({"model_label": args.model, "role": "prompt_running", "layer": int(row["layer"]), "algorithm": name, "seed": int(seeds[index]), "fold": int(folds[index]), "gold_count": int(y[index]), "predicted_count": int(predicted[index]), "correct": int(predicted[index] == y[index])})
        print(f"[fast-classification] {args.model} L{row['layer']}", flush=True)
    metrics = pd.DataFrame(metric_rows); predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output / "answer_classifier_metrics.csv", index=False)
    predictions.to_csv(args.output / "answer_classifier_oof_predictions.csv.gz", index=False, compression="gzip")
    predictions.groupby(["model_label", "role", "layer", "algorithm", "gold_count", "predicted_count"]).size().rename("rows").reset_index().to_csv(args.output / "answer_classifier_confusion.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_prompt_classification_fast_v1", "model": args.model,
        "role": "prompt_running", "algorithms": ALGORITHMS, "folds": 5, "split_unit": "seed",
        "pca_components": args.pca_components, "pca_fit": "inside each training fold and shared by algorithms",
        "datasets": len(rows), "metric_rows": len(metrics), "prediction_rows": len(predictions),
        "elapsed_seconds": time.perf_counter() - started, "status": "PASS",
    }
    (args.output / "answer_classifier_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
