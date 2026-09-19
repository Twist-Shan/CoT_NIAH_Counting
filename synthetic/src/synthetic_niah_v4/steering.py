from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .cache import extract_anchors
from .data import BaseExample
from .generation import predict_count_from_prefixes
from .hooks import AdditiveSteeringSpec, make_additive_hook
from .metrics import kl_divergence, normalize, spearman_corr
from .render import non_thinking_eval_prefix, render_for_model, thinking_oracle_trace_prefix
from .vocab import Vocab


STEERING_COLUMNS = [
    "model_type",
    "eval_mode",
    "anchor_name",
    "anchor_k",
    "hook_name",
    "layer",
    "direction_type",
    "alpha",
    "n_examples",
    "base_accuracy",
    "steered_accuracy",
    "mean_pred_base",
    "mean_pred_steered",
    "mean_count_shift",
    "mean_gold_logit_change",
    "mean_correct_logprob_change",
    "mean_target_plus_one_logprob_change",
    "monotonicity_score",
    "validity_rate",
    "count_token_rate",
    "kl_count_distribution_vs_base",
    "control_type",
]


FINAL_STEERING_ANCHORS = {"ans_token", "pre_ans_pos", "last_prompt_token", "think_end", "think_start"}
TRACE_STEERING_ANCHORS = {"pre_index_k", "index_k_pos", "marker_k_pos", "post_marker_k", "prompt_marker_k"}
STEERING_DIRECTION_TYPES = {"ridge", "dom", "matched_delta", "logistic_adjacent"}


def _prefix(example: BaseExample, model_type: str, vocab: Vocab) -> list[int]:
    if model_type == "non_thinking":
        return non_thinking_eval_prefix(example, vocab)
    return thinking_oracle_trace_prefix(example, vocab)


def _anchor_position(example: BaseExample, model_type: str, anchor_name: str, anchor_k: Any, vocab: Vocab) -> int | None:
    rendered = render_for_model(example, vocab, model_type)
    anchors = extract_anchors(rendered, example, model_type)
    want_k = None
    if str(anchor_k).strip() not in {"", "nan", "None"}:
        try:
            want_k = int(anchor_k)
        except ValueError:
            want_k = None
    for anchor in anchors:
        if anchor.anchor_name != anchor_name:
            continue
        if want_k is not None and anchor.anchor_k != want_k:
            continue
        return int(anchor.position)
    return None


def _select_direction_configs(direction_df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    if direction_df.empty:
        return direction_df
    if "projection_r2" not in direction_df.columns:
        direction_df = direction_df.copy()
        direction_df["projection_r2"] = 0.0
    typed = direction_df[direction_df["direction_type"].isin(STEERING_DIRECTION_TYPES)].copy()
    final_sub = typed[
        typed["target"].eq("final_count")
        & typed["anchor_name"].isin(FINAL_STEERING_ANCHORS)
    ].copy()
    trace_sub = typed[
        typed["target"].eq("prefix_count")
        & typed["anchor_name"].isin(TRACE_STEERING_ANCHORS)
    ].copy()
    sub = pd.concat([final_sub, trace_sub], ignore_index=True)
    resid = sub[sub["hook_name"].astype(str).str.startswith("resid_post_layer_")]
    if not resid.empty:
        sub = resid
    if sub.empty:
        sub = direction_df.head(int(cfg.steering.max_direction_configs)).copy()
    return sub.sort_values("projection_r2", ascending=False).head(int(cfg.steering.max_direction_configs))


def _load_direction_configs(run_dir: Path) -> pd.DataFrame:
    artifacts = run_dir / "artifacts"
    tables = run_dir / "tables"
    direction_meta_path = artifacts / "directions.csv"
    direction_metrics_path = tables / "direction_metrics.csv"
    direction_df = pd.read_csv(direction_meta_path) if direction_meta_path.exists() else pd.DataFrame()
    if direction_df.empty:
        return direction_df
    metrics_df = pd.read_csv(direction_metrics_path) if direction_metrics_path.exists() and direction_metrics_path.stat().st_size > 0 else pd.DataFrame()
    if metrics_df.empty or "projection_r2" not in metrics_df.columns:
        direction_df = direction_df.copy()
        direction_df["projection_r2"] = 0.0
        direction_df["projection_slope"] = 0.0
        return direction_df
    key_cols = ["model_type", "eval_mode", "anchor_name", "anchor_k", "hook_name", "layer", "direction_type", "target"]
    left = direction_df.copy()
    right = metrics_df.copy()
    for col in key_cols:
        if col not in left.columns:
            left[col] = ""
        if col not in right.columns:
            right[col] = ""
        left[f"__key_{col}"] = left[col].fillna("").astype(str)
        right[f"__key_{col}"] = right[col].fillna("").astype(str)
    merge_cols = [f"__key_{col}" for col in key_cols]
    keep = merge_cols + ["projection_r2", "projection_slope"]
    merged = left.merge(right[keep].drop_duplicates(merge_cols), on=merge_cols, how="left")
    merged["projection_r2"] = merged["projection_r2"].fillna(0.0)
    merged["projection_slope"] = merged["projection_slope"].fillna(0.0)
    return merged.drop(columns=merge_cols)


def _distribution(preds: list[int]) -> np.ndarray:
    arr = np.zeros(10, dtype=float)
    for pred in preds:
        if 1 <= int(pred) <= 10:
            arr[int(pred) - 1] += 1
    return arr / max(float(arr.sum()), 1.0)


def run_steering(
    models: dict[str, Any],
    examples: list[BaseExample],
    vocab: Vocab,
    cfg: Any,
    run_dir: Path,
) -> pd.DataFrame:
    artifacts = run_dir / "artifacts"
    tables = run_dir / "tables"
    direction_npz_path = artifacts / "directions.npz"
    direction_df = _load_direction_configs(run_dir)
    if direction_df.empty or not direction_npz_path.exists():
        out = pd.DataFrame(columns=STEERING_COLUMNS)
        out.to_csv(tables / "steering_results.csv", index=False)
        return out
    direction_npz = np.load(direction_npz_path)
    selected = _select_direction_configs(direction_df, cfg)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(cfg.train.seed))

    for _, cfg_row in selected.iterrows():
        model_type = str(cfg_row["model_type"])
        model = models[model_type]
        direction = normalize(np.asarray(direction_npz[str(cfg_row["vector_key"])], dtype=float))
        random_direction = normalize(rng.normal(size=direction.shape))
        controls = [
            (str(cfg_row["direction_type"]), "none", direction),
            ("random_unit", "random_unit_direction", random_direction),
            (str(cfg_row["direction_type"]), "zero_intervention", np.zeros_like(direction)),
        ]
        model_examples = examples[: max(1, int(cfg.steering.examples_per_count) * 10)]
        prefixes = [_prefix(ex, model_type, vocab) for ex in model_examples]
        gold = [ex.count for ex in model_examples]
        base_rows = predict_count_from_prefixes(model, prefixes, gold, vocab, cfg.device, batch_size=32)
        base_preds = [int(row["pred_count"]) for row in base_rows]
        for direction_type, control_type, vec in controls:
            for alpha in cfg.steering.alpha_grid:
                steered_rows = []
                valid_examples = 0
                for ex, base in zip(model_examples, base_rows):
                    pos = _anchor_position(ex, model_type, str(cfg_row["anchor_name"]), cfg_row.get("anchor_k", ""), vocab)
                    if pos is None:
                        continue
                    hook = make_additive_hook(
                        AdditiveSteeringSpec(
                            hook_name=str(cfg_row["hook_name"]),
                            positions=[pos],
                            direction=torch.tensor(vec, dtype=torch.float32),
                            alpha=float(alpha),
                            scale=float(cfg_row.get("scale", 1.0)),
                        )
                    )
                    one = predict_count_from_prefixes(model, [_prefix(ex, model_type, vocab)], [ex.count], vocab, cfg.device, batch_size=1, hook_fn=hook)[0]
                    steered_rows.append((base, one))
                    valid_examples += 1
                if not steered_rows:
                    continue
                steered_preds = [int(item[1]["pred_count"]) for item in steered_rows]
                base_sub_preds = [int(item[0]["pred_count"]) for item in steered_rows]
                base_acc = np.mean([float(item[0]["final_accuracy"]) for item in steered_rows])
                steered_acc = np.mean([float(item[1]["final_accuracy"]) for item in steered_rows])
                rows.append(
                    {
                        "model_type": model_type,
                        "eval_mode": str(cfg_row["eval_mode"]),
                        "anchor_name": str(cfg_row["anchor_name"]),
                        "anchor_k": cfg_row.get("anchor_k", ""),
                        "hook_name": str(cfg_row["hook_name"]),
                        "layer": int(cfg_row["layer"]),
                        "direction_type": direction_type,
                        "alpha": float(alpha),
                        "n_examples": int(valid_examples),
                        "base_accuracy": float(base_acc),
                        "steered_accuracy": float(steered_acc),
                        "mean_pred_base": float(np.mean(base_sub_preds)),
                        "mean_pred_steered": float(np.mean(steered_preds)),
                        "mean_count_shift": float(np.mean(np.asarray(steered_preds) - np.asarray(base_sub_preds))),
                        "mean_gold_logit_change": float(np.mean([item[1]["gold_logit"] - item[0]["gold_logit"] for item in steered_rows])),
                        "mean_correct_logprob_change": float(np.mean([item[1]["gold_logprob"] - item[0]["gold_logprob"] for item in steered_rows])),
                        "mean_target_plus_one_logprob_change": float(
                            np.mean([item[1]["target_plus_one_logprob"] - item[0]["target_plus_one_logprob"] for item in steered_rows])
                        ),
                        "monotonicity_score": float("nan"),
                        "validity_rate": 1.0,
                        "count_token_rate": float(np.mean([item[1]["count_token_rate"] for item in steered_rows])),
                        "kl_count_distribution_vs_base": kl_divergence(_distribution(steered_preds), _distribution(base_sub_preds)),
                        "control_type": control_type,
                    }
                )

    out = pd.DataFrame(rows, columns=STEERING_COLUMNS)
    if not out.empty:
        group_cols = ["model_type", "eval_mode", "anchor_name", "anchor_k", "hook_name", "layer", "direction_type", "control_type"]
        for keys, group in out.groupby(group_cols, dropna=False):
            mono = spearman_corr(group["alpha"].to_numpy(dtype=float), group["mean_pred_steered"].to_numpy(dtype=float))
            mask = np.ones(len(out), dtype=bool)
            for col, value in zip(group_cols, keys):
                mask &= out[col].astype(str).eq(str(value)).to_numpy()
            out.loc[mask, "monotonicity_score"] = mono
    tables.mkdir(parents=True, exist_ok=True)
    out.to_csv(tables / "steering_results.csv", index=False)
    return out
