from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .config import V6Config
from .data import BaseExample, balanced_examples
from .generation import pad_sequences
from .render import render_for_model
from .vocab import Vocab


FINAL_ANCHORS_NON = ("ans_token", "last_prompt_token")
FINAL_ANCHORS_THINK = ("think_start", "think_end", "ans_token", "pre_ans_token")
PREFIX_ANCHORS_THINK = ("pre_sep_k", "sep_token_k", "marker_token_k", "post_marker_k")


@torch.no_grad()
def _hidden_states_for_rendered(model, rendered, vocab: Vocab, device: str | torch.device):
    input_ids, lengths = pad_sequences([item.input_ids for item in rendered], vocab.pad_id, device)
    attention_mask = (input_ids != vocab.pad_id).long()
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    return out.hidden_states


def _anchor_positions(item, anchor: str) -> list[tuple[int, int, str]]:
    spans = item.spans
    if anchor == "ans_token":
        return [(spans.ans_pos, -1, "final_count")]
    if anchor == "last_prompt_token":
        return [(spans.seq_end_exclusive - 1, -1, "final_count")]
    if anchor == "think_start" and spans.think_open_pos is not None:
        return [(spans.think_open_pos, -1, "final_count")]
    if anchor == "think_end" and spans.think_close_pos is not None:
        return [(spans.think_close_pos, -1, "final_count")]
    if anchor == "pre_ans_token":
        return [(spans.ans_pos - 1, -1, "final_count")]
    out: list[tuple[int, int, str]] = []
    if anchor == "pre_sep_k":
        for k, sep_pos in enumerate(spans.sep_positions, start=1):
            out.append((sep_pos - 1, k, "prefix_count"))
    elif anchor == "sep_token_k":
        for k, sep_pos in enumerate(spans.sep_positions, start=1):
            out.append((sep_pos, k, "prefix_count"))
    elif anchor == "marker_token_k":
        for k, marker_pos in enumerate(spans.trace_marker_positions, start=1):
            out.append((marker_pos, k, "prefix_count"))
    elif anchor == "post_marker_k":
        for k, marker_pos in enumerate(spans.trace_marker_positions, start=1):
            if marker_pos + 1 < len(item.input_ids):
                out.append((marker_pos + 1, k, "prefix_count"))
    return out


def collect_probe_table(models: dict[str, Any], examples: list[BaseExample], vocab: Vocab, cfg: V6Config, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    batch_size = max(1, min(64, int(cfg.train.batch_size)))
    for model_type, model in models.items():
        anchors = list(FINAL_ANCHORS_NON if model_type == "non_thinking" else FINAL_ANCHORS_THINK + PREFIX_ANCHORS_THINK)
        rendered = [render_for_model(example, vocab, model_type) for example in examples]
        for start in tqdm(range(0, len(rendered), batch_size), desc=f"probe cache {model_type} {split}", leave=False):
            chunk = rendered[start : start + batch_size]
            hidden_states = _hidden_states_for_rendered(model, chunk, vocab, cfg.device)
            for local_idx, item in enumerate(chunk):
                example = examples[start + local_idx]
                for layer_idx, hidden in enumerate(hidden_states):
                    layer = layer_idx - 1
                    for anchor in anchors:
                        for pos, prefix_k, label_type in _anchor_positions(item, anchor):
                            label = example.count if label_type == "final_count" else prefix_k
                            rows.append(
                                {
                                    "split": split,
                                    "model_type": model_type,
                                    "example_id": example.example_id,
                                    "count": example.count,
                                    "prefix_count": prefix_k,
                                    "label": label,
                                    "label_type": label_type,
                                    "anchor_type": anchor,
                                    "layer": layer,
                                    "position": pos,
                                    "token_id": item.input_ids[pos],
                                    "trace_length_tokens": len(item.spans.trace_token_positions),
                                    "hidden": hidden[local_idx, pos].detach().cpu().numpy().astype(np.float32),
                                }
                            )
    return pd.DataFrame(rows)


def _baseline_accuracy(feature_train, y_train, feature_test, y_test) -> tuple[float, float]:
    if len(set(y_train.tolist())) <= 1:
        pred = np.full_like(y_test, y_train[0])
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, multi_class="auto"))
        model.fit(np.asarray(feature_train).reshape(-1, 1), y_train)
        pred = model.predict(np.asarray(feature_test).reshape(-1, 1))
    acc = accuracy_score(y_test, pred)
    try:
        r2 = r2_score(y_test, pred)
    except ValueError:
        r2 = math.nan
    return float(acc), float(r2)


def fit_probe_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model_type", "label_type", "anchor_type", "layer"]
    for keys, train_group in tqdm(train_df.groupby(group_cols), desc="fit probes", leave=False):
        model_type, label_type, anchor_type, layer = keys
        test_group = test_df[
            (test_df["model_type"] == model_type)
            & (test_df["label_type"] == label_type)
            & (test_df["anchor_type"] == anchor_type)
            & (test_df["layer"] == layer)
        ]
        if len(train_group) < 5 or len(test_group) < 5:
            continue
        x_train = np.stack(train_group["hidden"].to_numpy())
        x_test = np.stack(test_group["hidden"].to_numpy())
        y_train = train_group["label"].to_numpy(dtype=int)
        y_test = test_group["label"].to_numpy(dtype=int)
        if len(set(y_train.tolist())) <= 1:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, multi_class="auto"))
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0)))
        ridge.fit(x_train, y_train.astype(float))
        ridge_pred_float = ridge.predict(x_test)
        ridge_pred = np.clip(np.rint(ridge_pred_float), 1, 10).astype(int)
        pos_acc, pos_r2 = _baseline_accuracy(train_group["position"].to_numpy(), y_train, test_group["position"].to_numpy(), y_test)
        tok_acc, tok_r2 = _baseline_accuracy(train_group["token_id"].to_numpy(), y_train, test_group["token_id"].to_numpy(), y_test)
        tr_acc, tr_r2 = _baseline_accuracy(
            train_group["trace_length_tokens"].to_numpy(),
            y_train,
            test_group["trace_length_tokens"].to_numpy(),
            y_test,
        )
        shuffled = np.random.default_rng(0).permutation(y_train)
        shuf_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, multi_class="auto"))
        shuf_clf.fit(x_train, shuffled)
        shuf_pred = shuf_clf.predict(x_test)
        rows.append(
            {
                "model_type": model_type,
                "label_type": label_type,
                "anchor_type": anchor_type,
                "layer": layer,
                "probe_accuracy": float(accuracy_score(y_test, pred)),
                "probe_mae": float(mean_absolute_error(y_test, ridge_pred)),
                "probe_r2": float(r2_score(y_test, ridge_pred_float)),
                "ridge_rounded_accuracy": float(accuracy_score(y_test, ridge_pred)),
                "position_only_accuracy": pos_acc,
                "position_only_r2": pos_r2,
                "trace_length_only_accuracy": tr_acc,
                "trace_length_only_r2": tr_r2,
                "token_id_only_accuracy": tok_acc,
                "token_id_only_r2": tok_r2,
                "shuffled_label_accuracy": float(accuracy_score(y_test, shuf_pred)),
                "shuffled_label_r2": math.nan,
                "n_train": len(train_group),
                "n_test": len(test_group),
            }
        )
    return pd.DataFrame(rows)


def run_probes(models: dict[str, Any], vocab: Vocab, cfg: V6Config, run_dir: Path) -> pd.DataFrame:
    train_examples = balanced_examples(cfg.seq_len, cfg.eval.probe_train_examples_per_count, cfg.seed + 20_000)
    test_examples = balanced_examples(cfg.seq_len, cfg.eval.probe_test_examples_per_count, cfg.seed + 30_000)
    train_df = collect_probe_table(models, train_examples, vocab, cfg, "train")
    test_df = collect_probe_table(models, test_examples, vocab, cfg, "test")
    metrics = fit_probe_metrics(train_df, test_df)
    out_dir = run_dir / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "probe_metrics.csv", index=False)
    return metrics

