from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cache import HiddenCache, load_hidden_cache
from .metrics import normalize, safe_r2


PREFIX_TARGET_ANCHORS = {"pre_index_k", "post_marker_k", "marker_k_pos", "index_k_pos", "prompt_marker_k"}
PROBE_COLUMNS = [
    "model_type",
    "eval_mode",
    "anchor_name",
    "anchor_k",
    "target",
    "hook_name",
    "layer",
    "probe_type",
    "raw_or_residualized",
    "train_n",
    "test_n",
    "accuracy",
    "r2",
    "mae",
    "ce_loss",
    "position_baseline_acc",
    "token_baseline_acc",
    "trace_len_baseline_acc",
    "leakage_prone",
]


def target_for_anchor(anchor_name: str) -> str:
    return "prefix_count" if anchor_name in PREFIX_TARGET_ANCHORS else "final_count"


def _targets(group: pd.DataFrame, target: str) -> np.ndarray:
    if target == "prefix_count":
        return group["prefix_count"].to_numpy(dtype=int)
    return group["final_count"].to_numpy(dtype=int)


def split_example_ids(example_ids: np.ndarray, train_fraction: float, seed: int = 0) -> tuple[set[str], set[str]]:
    unique = np.array(sorted(set(str(item) for item in example_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    split = max(1, min(len(unique) - 1, int(round(len(unique) * train_fraction)))) if len(unique) > 1 else len(unique)
    train = set(unique[:split].tolist())
    test = set(unique[split:].tolist())
    if not test:
        test = set(unique[:split].tolist())
    return train, test


def _classification_metrics(pipe, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    pred = pipe.predict(x_test)
    acc = float(accuracy_score(y_test, pred))
    try:
        probs = pipe.predict_proba(x_test)
        labels = list(pipe.named_steps["logisticregression"].classes_)
        ce = float(log_loss(y_test, probs, labels=labels))
    except Exception:
        ce = float("nan")
    return acc, ce


def _fit_probe_pair(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    if len(set(y_train.tolist())) < 2 or len(x_train) < 2:
        nan = {"accuracy": float("nan"), "r2": float("nan"), "mae": float("nan"), "ce_loss": float("nan")}
        return nan, nan, np.zeros(x_train.shape[1]), np.zeros(x_train.shape[1])

    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=list(alphas)))
    ridge.fit(x_train, y_train.astype(float))
    ridge_pred = ridge.predict(x_test)
    ridge_round = np.clip(np.rint(ridge_pred), 1, 10).astype(int)
    ridge_metrics = {
        "accuracy": float(accuracy_score(y_test, ridge_round)),
        "r2": safe_r2(y_test.astype(float), ridge_pred.astype(float)),
        "mae": float(mean_absolute_error(y_test.astype(float), ridge_pred.astype(float))),
        "ce_loss": float("nan"),
    }
    ridge_step = ridge.named_steps["ridgecv"]
    scaler = ridge.named_steps["standardscaler"]
    ridge_vec = np.asarray(ridge_step.coef_, dtype=float) / np.maximum(np.asarray(scaler.scale_, dtype=float), 1e-12)

    logistic_metrics = {"accuracy": float("nan"), "r2": float("nan"), "mae": float("nan"), "ce_loss": float("nan")}
    logistic_adjacent = np.zeros(x_train.shape[1], dtype=float)
    try:
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", multi_class="auto"),
        )
        logistic.fit(x_train, y_train)
        acc, ce = _classification_metrics(logistic, x_test, y_test)
        logistic_metrics.update({"accuracy": acc, "ce_loss": ce})
        clf = logistic.named_steps["logisticregression"]
        scaler_l = logistic.named_steps["standardscaler"]
        coef = np.asarray(clf.coef_, dtype=float) / np.maximum(np.asarray(scaler_l.scale_, dtype=float), 1e-12)
        class_to_coef = {int(cls): coef[idx] for idx, cls in enumerate(clf.classes_)}
        diffs = []
        for k in range(1, 10):
            if k in class_to_coef and (k + 1) in class_to_coef:
                diffs.append(class_to_coef[k + 1] - class_to_coef[k])
        if diffs:
            logistic_adjacent = np.mean(np.stack(diffs), axis=0)
    except Exception:
        pass
    return ridge_metrics, logistic_metrics, ridge_vec, logistic_adjacent


def _baseline_acc(feature: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> float:
    x_train = feature[train_mask].reshape(-1, 1)
    x_test = feature[test_mask].reshape(-1, 1)
    if len(set(y_train.tolist())) < 2 or len(x_train) < 2:
        return float("nan")
    try:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced"))
        pipe.fit(x_train, y_train)
        return float(accuracy_score(y_test, pipe.predict(x_test)))
    except Exception:
        return float("nan")


def _residualize_hidden(x: np.ndarray, confounds: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return x
    model = LinearRegression()
    model.fit(confounds[train_mask], x[train_mask])
    return x - model.predict(confounds)


def run_probes(cache: HiddenCache, cfg: Any, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    vector_arrays: dict[str, np.ndarray] = {}
    meta = cache.metadata.copy()
    if meta.empty:
        out = pd.DataFrame(columns=PROBE_COLUMNS)
        out.to_csv(run_dir / "tables" / "probe_results.csv", index=False)
        out.to_csv(run_dir / "tables" / "probe_residualized.csv", index=False)
        pd.DataFrame().to_csv(run_dir / "tables" / "probe_baselines.csv", index=False)
        return out, pd.DataFrame(), out

    meta["hidden_idx"] = np.arange(len(meta))
    group_cols = ["model_type", "eval_mode", "anchor_name", "hook_name", "layer"]
    for keys, group in meta.groupby(group_cols, dropna=False):
        model_type, eval_mode, anchor_name, hook_name, layer = keys
        target = target_for_anchor(str(anchor_name))
        if target == "prefix_count" and group["prefix_count"].replace("", np.nan).dropna().empty:
            continue
        y = _targets(group, target)
        if len(set(y.tolist())) < 2:
            continue
        idx = group["hidden_idx"].to_numpy(dtype=int)
        x = cache.hidden[idx]
        train_ids, test_ids = split_example_ids(group["example_id"].to_numpy(), cfg.probe.train_fraction, seed=int(cfg.train.seed))
        train_mask = group["example_id"].astype(str).isin(train_ids).to_numpy()
        test_mask = group["example_id"].astype(str).isin(test_ids).to_numpy()
        if not train_mask.any() or not test_mask.any():
            continue
        y_train, y_test = y[train_mask], y[test_mask]
        x_train, x_test = x[train_mask], x[test_mask]
        position = group["position"].to_numpy(dtype=float)
        token = group["absolute_token_id"].to_numpy(dtype=float)
        trace_len = pd.to_numeric(group["trace_length_tokens"].replace("", np.nan), errors="coerce").fillna(0).to_numpy(dtype=float)
        anchor_k = pd.to_numeric(group["anchor_k"].replace("", np.nan), errors="coerce").fillna(0).to_numpy(dtype=float)
        pos_acc = _baseline_acc(position, y_train, y_test, train_mask, test_mask)
        tok_acc = _baseline_acc(token, y_train, y_test, train_mask, test_mask)
        trace_acc = _baseline_acc(trace_len, y_train, y_test, train_mask, test_mask)
        index_acc = _baseline_acc(anchor_k, y_train, y_test, train_mask, test_mask)
        baseline_base = {
            "model_type": model_type,
            "eval_mode": eval_mode,
            "anchor_name": anchor_name,
            "anchor_k": "",
            "target": target,
            "hook_name": hook_name,
            "layer": int(layer),
            "position_baseline_acc": pos_acc,
            "token_baseline_acc": tok_acc,
            "trace_len_baseline_acc": trace_acc,
            "index_token_only_baseline_acc": index_acc,
        }
        baseline_rows.append(baseline_base)

        for mode, x_all in [
            ("raw", x),
            (
                "residualized",
                _residualize_hidden(
                    x,
                    np.stack([position, token, trace_len], axis=1),
                    train_mask,
                ),
            ),
        ]:
            x_train_m, x_test_m = x_all[train_mask], x_all[test_mask]
            ridge_metrics, logistic_metrics, ridge_vec, logistic_vec = _fit_probe_pair(
                x_train_m,
                y_train,
                x_test_m,
                y_test,
                cfg.probe.ridge_alpha_grid,
            )
            shuffled_y = np.random.default_rng(int(cfg.train.seed)).permutation(y_train)
            _, _, shuffled_vec, _ = _fit_probe_pair(x_train_m, shuffled_y, x_test_m, y_test, cfg.probe.ridge_alpha_grid)
            base = {
                "model_type": model_type,
                "eval_mode": eval_mode,
                "anchor_name": anchor_name,
                "anchor_k": "",
                "target": target,
                "hook_name": hook_name,
                "layer": int(layer),
                "raw_or_residualized": mode,
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
                "position_baseline_acc": pos_acc,
                "token_baseline_acc": tok_acc,
                "trace_len_baseline_acc": trace_acc,
                "leakage_prone": bool(group["leakage_prone"].astype(bool).any()),
            }
            rows.append({**base, "probe_type": "ridge_scalar", **ridge_metrics})
            rows.append({**base, "probe_type": "multiclass_logistic", **logistic_metrics})
            if mode == "raw":
                for direction_type, vec in [
                    ("ridge", ridge_vec),
                    ("logistic_adjacent", logistic_vec),
                    ("shuffled_label_probe", shuffled_vec),
                ]:
                    key = f"vec_{len(vector_arrays)}"
                    vector_arrays[key] = normalize(vec).astype(np.float32)
                    vector_rows.append(
                        {
                            **{k: base[k] for k in ["model_type", "eval_mode", "anchor_name", "anchor_k", "target", "hook_name", "layer"]},
                            "direction_type": direction_type,
                            "vector_key": key,
                            "scale": float(np.std(x @ normalize(vec))) if np.linalg.norm(vec) > 0 else 1.0,
                        }
                    )

    tables = run_dir / "tables"
    artifacts = run_dir / "artifacts"
    tables.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    probe_df = pd.DataFrame(rows, columns=PROBE_COLUMNS)
    baseline_df = pd.DataFrame(baseline_rows)
    residualized_df = probe_df[probe_df["raw_or_residualized"].eq("residualized")].copy() if not probe_df.empty else probe_df.copy()
    probe_df.to_csv(tables / "probe_results.csv", index=False)
    baseline_df.to_csv(tables / "probe_baselines.csv", index=False)
    residualized_df.to_csv(tables / "probe_residualized.csv", index=False)
    vector_meta = pd.DataFrame(vector_rows)
    vector_meta.to_csv(artifacts / "probe_vectors.csv", index=False)
    np.savez(artifacts / "probe_vectors.npz", **vector_arrays)
    (artifacts / "probe_vector_manifest.json").write_text(json.dumps({"n_vectors": len(vector_arrays)}, indent=2), encoding="utf-8")
    return probe_df, baseline_df, residualized_df


def run_probes_from_disk(cfg: Any, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_probes(load_hidden_cache(run_dir), cfg, run_dir)
