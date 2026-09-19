from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import BaseExample
from .render import render_for_model
from .vocab import Vocab


def _position_baseline_accuracy(positions: np.ndarray, y: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return 1.0
    split = max(1, int(0.7 * len(y)))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, multi_class="auto", class_weight="balanced"),
    )
    clf.fit(positions[:split, None], y[:split])
    return float(accuracy_score(y[split:], clf.predict(positions[split:, None])))


def _fit_probe(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 8 or len(set(y.tolist())) < 2:
        return {"test_accuracy": float("nan"), "r2": float("nan"), "mae": float("nan")}
    split = max(1, int(0.7 * len(y)))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, multi_class="auto", class_weight="balanced"),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
    ridge.fit(x_train, y_train.astype(float))
    pred_float = ridge.predict(x_test)
    return {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "r2": float(r2_score(y_test.astype(float), pred_float)) if len(set(y_test.tolist())) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_test.astype(float), pred_float)),
    }


@torch.no_grad()
def collect_probe_records(
    model,
    model_type: str,
    examples: list[BaseExample],
    vocab: Vocab,
    device: str,
    checkpoint_step: int,
    seed: int,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for example in examples:
        rendered = render_for_model(example, vocab, model_type)
        input_ids = torch.tensor([rendered.tokens], dtype=torch.long, device=device)
        out = model(input_ids, output_hidden_states=True)
        if out.hidden_states is None:
            continue
        anchors: list[tuple[str, int, int, bool]] = []
        if model_type == "non_thinking":
            anchors.append(("ans_pos", rendered.spans.ans_pos, example.count, False))
            anchors.append(("pre_ans_pos", rendered.spans.ans_pos - 1, example.count, False))
        else:
            anchors.append(("think_open_pos", rendered.spans.think_open_pos or 0, example.count, False))
            anchors.append(("think_close_pos", rendered.spans.think_close_pos or 0, example.count, False))
            anchors.append(("ans_pos", rendered.spans.ans_pos, example.count, False))
            anchors.append(("pre_ans_pos", rendered.spans.ans_pos - 1, example.count, False))
            for idx, index_pos in enumerate(rendered.spans.trace_index_positions, start=1):
                anchors.append(("pre_index_k", index_pos - 1, idx, False))
                anchors.append(("index_k_pos", index_pos, idx, True))
            for idx, marker_pos in enumerate(rendered.spans.trace_marker_positions, start=1):
                anchors.append(("marker_k_pos", marker_pos, idx, False))
                if marker_pos + 1 < len(rendered.tokens):
                    anchors.append(("post_marker_k", marker_pos + 1, idx, False))
        for anchor_type, pos, target, leakage_prone in anchors:
            for layer, hidden in enumerate(out.hidden_states):
                records.append(
                    {
                        "model_type": model_type,
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "seq_len_eval": example.seq_len,
                        "layer": layer,
                        "resid_site": "embed" if layer == 0 else "resid_post",
                        "anchor_type": anchor_type,
                        "target_type": "prefix_count" if anchor_type in {"pre_index_k", "index_k_pos", "marker_k_pos", "post_marker_k"} else "final_count",
                        "target": int(target),
                        "absolute_position": int(pos),
                        "trace_length": int(example.count),
                        "leakage_prone": bool(leakage_prone),
                        "hidden": hidden[0, pos].detach().cpu().numpy(),
                    }
                )
    return records


def run_probes(
    model_by_type: dict[str, Any],
    examples_by_len: dict[int, list[BaseExample]],
    vocab: Vocab,
    cfg: dict[str, Any],
    seed: int,
    checkpoint_step: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    device = cfg["device"]
    examples: list[BaseExample] = []
    for seq_examples in examples_by_len.values():
        examples.extend(seq_examples)
    max_examples = min(len(examples), int(cfg["probe_examples_per_count"]) * 10)
    examples = examples[:max_examples]
    for model_type, model in model_by_type.items():
        records = collect_probe_records(model, model_type, examples, vocab, device, checkpoint_step, seed)
        if not records:
            continue
        df = pd.DataFrame(records)
        for (layer, anchor_type, target_type, leakage_prone), group in df.groupby(
            ["layer", "anchor_type", "target_type", "leakage_prone"], dropna=False
        ):
            x = np.stack(group["hidden"].to_numpy())
            y = group["target"].to_numpy(dtype=int)
            metrics = _fit_probe(x, y)
            pos_acc = _position_baseline_accuracy(group["absolute_position"].to_numpy(dtype=float), y)
            trace_acc = _position_baseline_accuracy(group["trace_length"].to_numpy(dtype=float), y)
            rows.append(
                {
                    "model_type": model_type,
                    "seed": seed,
                    "checkpoint_step": checkpoint_step,
                    "seq_len_eval": ",".join(map(str, sorted(examples_by_len))),
                    "layer": int(layer),
                    "resid_site": "embed" if int(layer) == 0 else "resid_post",
                    "anchor_type": anchor_type,
                    "target_type": target_type,
                    "probe_type": "multinomial_logistic_and_ridge",
                    "train_accuracy": float("nan"),
                    "test_accuracy": metrics["test_accuracy"],
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "position_only_accuracy": pos_acc,
                    "trace_length_only_accuracy": trace_acc if model_type == "thinking" else float("nan"),
                    "embedding_only_accuracy": float("nan"),
                    "leakage_prone": bool(leakage_prone),
                }
            )
    return pd.DataFrame(rows)
