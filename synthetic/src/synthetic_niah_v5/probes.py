from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import run_cache
from .vocab import Vocab


def _majority_acc(y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    values, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def _lookup_baseline(feature: np.ndarray, y: np.ndarray, split: int) -> float:
    if len(y) <= split:
        return float("nan")
    mapping: dict[float, int] = {}
    for value in np.unique(feature[:split]):
        labels = y[:split][feature[:split] == value]
        vals, counts = np.unique(labels, return_counts=True)
        mapping[float(value)] = int(vals[counts.argmax()])
    default = int(np.unique(y[:split], return_counts=True)[0][0])
    pred = np.array([mapping.get(float(v), default) for v in feature[split:]])
    return float((pred == y[split:]).mean())


def _standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def _centroid_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    labels = np.unique(y_train)
    centroids = np.stack([x_train[y_train == label].mean(axis=0) for label in labels])
    distances = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    return labels[distances.argmin(axis=1)]


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    reg = alpha * np.eye(x_train.shape[1])
    reg[-1, -1] = 0.0
    weights = np.linalg.pinv(x_train.T @ x_train + reg) @ x_train.T @ y_train.astype(float)
    return x_test @ weights


def _fit_probe(x: np.ndarray, y: np.ndarray, position: np.ndarray, trace_len: np.ndarray) -> dict[str, float]:
    if len(y) < 8 or len(np.unique(y)) < 2:
        return {
            "accuracy": _majority_acc(y),
            "r2": float("nan"),
            "mae": float("nan"),
            "position_baseline_acc": _majority_acc(y),
            "trace_len_baseline_acc": _majority_acc(y),
        }
    split = max(1, int(0.7 * len(y)))
    if len(np.unique(y[:split])) < 2 or len(np.unique(y[split:])) < 1:
        return {
            "accuracy": _majority_acc(y),
            "r2": float("nan"),
            "mae": float("nan"),
            "position_baseline_acc": _lookup_baseline(position, y, split),
            "trace_len_baseline_acc": _lookup_baseline(trace_len, y, split),
        }
    x_train, x_test = _standardize(x[:split], x[split:])
    pred = _centroid_predict(x_train, y[:split], x_test)
    pred_float = _ridge_predict(x_train, y[:split], x_test)
    ss_res = float(((y[split:].astype(float) - pred_float) ** 2).sum())
    ss_tot = float(((y[split:].astype(float) - y[split:].astype(float).mean()) ** 2).sum())
    return {
        "accuracy": float((pred == y[split:]).mean()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mae": float(np.abs(y[split:].astype(float) - pred_float).mean()),
        "position_baseline_acc": _lookup_baseline(position, y, split),
        "trace_len_baseline_acc": _lookup_baseline(trace_len, y, split),
    }


def run_probes(cfg: dict[str, Any], vocab: Vocab, run_dir: Path) -> pd.DataFrame:
    tables = run_dir / "tables"
    cache_path = run_dir / "cache" / "hidden_cache.npz"
    index_path = tables / "hidden_cache_index.csv"
    if not cache_path.exists() or not index_path.exists():
        run_cache(cfg, vocab, run_dir)
    index = pd.read_csv(index_path)
    hidden = np.load(cache_path)["hidden"]
    rows: list[dict[str, Any]] = []
    group_cols = ["mode", "anchor_name", "target", "layer", "hook_name", "leakage_prone"]
    for keys, group in index.groupby(group_cols, dropna=False):
        mode, anchor_name, target, layer, hook_name, leakage_prone = keys
        x = hidden[group["hidden_index"].to_numpy(dtype=int)]
        y = group["target_value"].to_numpy(dtype=int)
        metrics = _fit_probe(
            x,
            y,
            group["position"].to_numpy(dtype=float),
            group["trace_len"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "mode": mode,
                "anchor_name": anchor_name,
                "target": target,
                "layer": int(layer),
                "hook_name": hook_name,
                "probe_type": "multinomial_logistic_and_ridge",
                "accuracy": metrics["accuracy"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "position_baseline_acc": metrics["position_baseline_acc"],
                "trace_len_baseline_acc": metrics["trace_len_baseline_acc"],
                "leakage_prone": bool(leakage_prone),
            }
        )
    probe_df = pd.DataFrame(rows)
    probe_df.to_csv(tables / "probe_results.csv", index=False)
    return probe_df
