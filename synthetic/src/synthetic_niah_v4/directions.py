from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import HiddenCache, load_hidden_cache
from .metrics import normalize, safe_r2
from .probes import target_for_anchor
from .vocab import Vocab


DIRECTION_COLUMNS = [
    "model_type",
    "eval_mode",
    "anchor_name",
    "anchor_k",
    "hook_name",
    "layer",
    "direction_type",
    "target",
    "norm",
    "projection_slope",
    "projection_r2",
    "cosine_with_ridge",
    "cosine_with_dom",
    "cosine_with_matched_delta",
    "cosine_with_unembedding",
]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _projection_metrics(hidden: np.ndarray, target: np.ndarray, vec: np.ndarray) -> tuple[float, float]:
    if len(hidden) < 2 or float(np.linalg.norm(vec)) == 0.0:
        return float("nan"), float("nan")
    proj = hidden @ vec
    slope = float(np.polyfit(target.astype(float), proj.astype(float), 1)[0]) if len(set(target.tolist())) > 1 else float("nan")
    pred = np.polyval(np.polyfit(target.astype(float), proj.astype(float), 1), target.astype(float)) if len(set(target.tolist())) > 1 else proj
    return slope, safe_r2(proj.astype(float), pred.astype(float))


def _dom_direction(hidden: np.ndarray, target: np.ndarray) -> np.ndarray:
    diffs = []
    for k in range(1, 10):
        left = hidden[target == k]
        right = hidden[target == k + 1]
        if len(left) and len(right):
            diffs.append(right.mean(axis=0) - left.mean(axis=0))
    return normalize(np.mean(np.stack(diffs), axis=0)) if diffs else np.zeros(hidden.shape[1], dtype=float)


def _unembed_direction(model: Any, vocab: Vocab) -> np.ndarray:
    weight = model.gpt2.lm_head.weight.detach().cpu().numpy()
    diffs = []
    ids = vocab.count_ids
    for idx in range(len(ids) - 1):
        diffs.append(weight[ids[idx + 1]] - weight[ids[idx]])
    return normalize(np.mean(np.stack(diffs), axis=0))


def run_directions(cache: HiddenCache, models: dict[str, Any], vocab: Vocab, cfg: Any, run_dir: Path) -> pd.DataFrame:
    artifacts = run_dir / "artifacts"
    tables = run_dir / "tables"
    artifacts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    probe_meta_path = artifacts / "probe_vectors.csv"
    probe_npz_path = artifacts / "probe_vectors.npz"
    probe_meta = pd.read_csv(probe_meta_path) if probe_meta_path.exists() else pd.DataFrame()
    probe_npz = np.load(probe_npz_path) if probe_npz_path.exists() else None

    vectors: dict[str, np.ndarray] = {}
    vector_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    meta = cache.metadata.copy()
    if meta.empty:
        out = pd.DataFrame(columns=DIRECTION_COLUMNS)
        out.to_csv(tables / "direction_metrics.csv", index=False)
        return out
    meta["hidden_idx"] = np.arange(len(meta))
    group_cols = ["model_type", "eval_mode", "anchor_name", "hook_name", "layer"]
    unembed_by_model = {model_type: _unembed_direction(model, vocab) for model_type, model in models.items()}

    def add_vector(base: dict[str, Any], direction_type: str, vec: np.ndarray, scale_hidden: np.ndarray, target_values: np.ndarray) -> None:
        key = f"dir_{len(vectors)}"
        vec_n = normalize(vec).astype(np.float32)
        vectors[key] = vec_n
        scale = float(np.std(scale_hidden @ vec_n)) if len(scale_hidden) and np.linalg.norm(vec_n) > 0 else 1.0
        vector_rows.append({**base, "direction_type": direction_type, "vector_key": key, "scale": scale if scale > 1e-8 else 1.0})

    for keys, group in meta.groupby(group_cols, dropna=False):
        model_type, eval_mode, anchor_name, hook_name, layer = keys
        target_name = target_for_anchor(str(anchor_name))
        if target_name == "prefix_count" and group["prefix_count"].replace("", np.nan).dropna().empty:
            continue
        target_values = (
            group["prefix_count"].to_numpy(dtype=int)
            if target_name == "prefix_count"
            else group["final_count"].to_numpy(dtype=int)
        )
        if len(set(target_values.tolist())) < 2:
            continue
        idx = group["hidden_idx"].to_numpy(dtype=int)
        hidden = cache.hidden[idx]
        base = {
            "model_type": model_type,
            "eval_mode": eval_mode,
            "anchor_name": anchor_name,
            "anchor_k": "",
            "hook_name": hook_name,
            "layer": int(layer),
            "target": target_name,
        }
        dom = _dom_direction(hidden, target_values)
        matched = dom.copy()
        unembed = unembed_by_model.get(model_type, np.zeros(cache.d_model))
        add_vector(base, "dom", dom, hidden, target_values)
        add_vector(base, "matched_delta", matched, hidden, target_values)
        add_vector(base, "unembedding_adjacent", unembed, hidden, target_values)
        if not probe_meta.empty and probe_npz is not None:
            probe_match = probe_meta[
                probe_meta["model_type"].eq(model_type)
                & probe_meta["eval_mode"].eq(eval_mode)
                & probe_meta["anchor_name"].eq(anchor_name)
                & probe_meta["hook_name"].eq(hook_name)
                & probe_meta["layer"].astype(int).eq(int(layer))
                & probe_meta["target"].eq(target_name)
            ]
            for _, row in probe_match.iterrows():
                key = str(row["vector_key"])
                if key in probe_npz:
                    add_vector(base, str(row["direction_type"]), np.asarray(probe_npz[key]), hidden, target_values)

    vector_df = pd.DataFrame(vector_rows)
    if vector_df.empty:
        out = pd.DataFrame(columns=DIRECTION_COLUMNS)
        out.to_csv(tables / "direction_metrics.csv", index=False)
        return out
    for keys, group in vector_df.groupby(["model_type", "eval_mode", "anchor_name", "hook_name", "layer", "target"], dropna=False):
        vec_by_type = {row["direction_type"]: vectors[row["vector_key"]] for _, row in group.iterrows()}
        meta_group = meta[
            meta["model_type"].eq(keys[0])
            & meta["eval_mode"].eq(keys[1])
            & meta["anchor_name"].eq(keys[2])
            & meta["hook_name"].eq(keys[3])
            & meta["layer"].astype(int).eq(int(keys[4]))
        ]
        hidden = cache.hidden[meta_group["hidden_idx"].to_numpy(dtype=int)]
        target_values = (
            meta_group["prefix_count"].to_numpy(dtype=int)
            if keys[5] == "prefix_count"
            else meta_group["final_count"].to_numpy(dtype=int)
        )
        for _, row in group.iterrows():
            vec = vectors[row["vector_key"]]
            slope, r2 = _projection_metrics(hidden, target_values, vec)
            metric_rows.append(
                {
                    **{col: row[col] for col in ["model_type", "eval_mode", "anchor_name", "anchor_k", "hook_name", "layer", "direction_type", "target"]},
                    "norm": float(np.linalg.norm(vec)),
                    "projection_slope": slope,
                    "projection_r2": r2,
                    "cosine_with_ridge": _cos(vec, vec_by_type.get("ridge", np.zeros_like(vec))),
                    "cosine_with_dom": _cos(vec, vec_by_type.get("dom", np.zeros_like(vec))),
                    "cosine_with_matched_delta": _cos(vec, vec_by_type.get("matched_delta", np.zeros_like(vec))),
                    "cosine_with_unembedding": _cos(vec, vec_by_type.get("unembedding_adjacent", np.zeros_like(vec))),
                }
            )

    np.savez(artifacts / "directions.npz", **vectors)
    vector_df.to_csv(artifacts / "directions.csv", index=False)
    metrics = pd.DataFrame(metric_rows, columns=DIRECTION_COLUMNS)
    metrics.to_csv(tables / "direction_metrics.csv", index=False)
    _write_input_geometry(metrics, cache, run_dir)
    return metrics


def _write_input_geometry(direction_metrics: pd.DataFrame, cache: HiddenCache, run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    if not direction_metrics.empty and not cache.metadata.empty:
        top = direction_metrics.sort_values("projection_r2", ascending=False).head(5)
        for _, row in top.iterrows():
            rows.append(
                {
                    "model_type": row["model_type"],
                    "anchor_name": row["anchor_name"],
                    "layer": row["layer"],
                    "direction_type": row["direction_type"],
                    "perturbation": "baseline_projection_by_count",
                    "projection_slope": row["projection_slope"],
                    "projection_r2": row["projection_r2"],
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "tables" / "input_geometry_results.csv", index=False)


def run_directions_from_disk(models: dict[str, Any], vocab: Vocab, cfg: Any, run_dir: Path) -> pd.DataFrame:
    return run_directions(load_hidden_cache(run_dir), models, vocab, cfg, run_dir)
