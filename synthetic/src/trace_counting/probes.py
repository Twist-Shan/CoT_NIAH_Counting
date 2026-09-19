from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .eval import resolve_checkpoint_path
from .io_utils import ensure_dir, read_jsonl, save_json
from .model import load_model_from_checkpoint
from .tokenizer import VocabTokenizer

RIDGE_ALPHAS = np.logspace(-4, 4, 17)


def parse_layers(layer_spec: str, n_hidden_states: int) -> list[int]:
    if layer_spec == "all":
        return list(range(n_hidden_states))
    layers = []
    for part in layer_spec.split(","):
        idx = int(part.strip())
        if idx < 0:
            idx = n_hidden_states + idx
        if idx < 0 or idx >= n_hidden_states:
            raise ValueError(f"Layer {part!r} is out of range for {n_hidden_states} hidden-state tensors.")
        layers.append(idx)
    return layers


def layer_name(idx: int) -> str:
    return "embeddings" if idx == 0 else f"layer_{idx}"


def anchor_positions(example: dict, anchors: set[str]) -> list[dict[str, Any]]:
    spans = example["spans"]
    positions: list[dict[str, Any]] = []
    base = {
        "example_id": example["example_id"],
        "seq_len": int(example["seq_len"]),
        "total_count": int(example["count"]),
        "density": int(example["count"]) / max(int(example["seq_len"]), 1),
    }
    fixed = {
        "think_open": spans["think_open_idx"],
        "think_close": spans["think_close_idx"],
        "ans": spans["ans_idx"],
        "count": spans["count_idx"],
    }
    for name, idx in fixed.items():
        if name in anchors and idx is not None:
            positions.append({**base, "anchor": name, "idx": idx})

    if "source" in anchors:
        running = 0
        for idx in range(spans["source_start"], spans["source_end_exclusive"]):
            token = example["full_tokens"][idx]
            is_marker = int(token in {"X", "Y", "Z"})
            running += is_marker
            positions.append(
                {
                    **base,
                    "anchor": "source",
                    "idx": idx,
                    "token": token,
                    "is_marker": is_marker,
                    "running_count": running,
                    "source_pos": idx - spans["source_start"],
                }
            )

    if "source_marker" in anchors:
        for pair in spans["trace_pairs"]:
            positions.append(
                {
                    **base,
                    "anchor": "source_marker",
                    "idx": pair["source_idx"],
                    "token": pair["marker"],
                    "k": pair["k"],
                    "running_count": pair["k"],
                    "is_marker": 1,
                }
            )

    if "trace_index" in anchors or "trace_marker" in anchors:
        for pair in spans["trace_pairs"]:
            if "trace_index" in anchors:
                if pair.get("index_idx") is not None:
                    positions.append(
                        {
                            **base,
                            "anchor": "trace_index",
                            "idx": pair["index_idx"],
                            "k": pair["k"],
                            "token": example["full_tokens"][pair["index_idx"]],
                        }
                    )
            if "trace_marker" in anchors:
                if pair.get("marker_idx") is not None:
                    positions.append(
                        {
                            **base,
                            "anchor": "trace_marker",
                            "idx": pair["marker_idx"],
                            "k": pair["k"],
                            "token": pair["marker"],
                        }
                    )
    return positions


def fit_ridge(X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    if len(X) < 8 or np.unique(y).size < 2:
        return {"r2": np.nan, "mae": np.nan, "n": int(len(X))}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    probe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    probe.fit(X_train, y_train)
    pred = probe.predict(X_test)
    return {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "n": int(len(X)),
    }


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2 or counts.min() < 2 or len(X) < 8:
        return {"accuracy": np.nan, "n_classes": int(len(values)), "n": int(len(X))}
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="auto"),
    )
    probe.fit(X_train, y_train)
    pred = probe.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "n_classes": int(len(values)),
        "n": int(len(X)),
    }


def run_probes(args: argparse.Namespace) -> pd.DataFrame:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    tokenizer = VocabTokenizer.load(data_dir / "vocab.json")
    examples = read_jsonl(data_dir / f"{args.split}.jsonl", limit=args.limit)
    model = load_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    anchors = {part.strip() for part in args.anchors.split(",") if part.strip()}

    feature_store: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    label_store: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    selected_layers: list[int] | None = None
    with torch.no_grad():
        for example in tqdm(examples, desc=f"extract probes {args.split}", dynamic_ncols=True):
            ids = tokenizer.encode(example["full_tokens"])
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if selected_layers is None:
                selected_layers = parse_layers(args.layers, len(hidden_states))
            positions = anchor_positions(example, anchors)
            for layer_idx in selected_layers:
                layer_hidden = hidden_states[layer_idx][0].detach().float().cpu().numpy()
                lname = layer_name(layer_idx)
                for pos in positions:
                    key = (lname, pos["anchor"])
                    feature_store[key].append(layer_hidden[pos["idx"]])
                    label_store[key].append(pos)

    summary_rows: list[dict[str, Any]] = []
    feature_paths: dict[str, str] = {}
    for (lname, anchor), features in sorted(feature_store.items()):
        X = np.stack(features, axis=0)
        labels = pd.DataFrame(label_store[(lname, anchor)])
        if args.save_features:
            stem = f"features_{lname}_{anchor}".replace("/", "_")
            np.savez_compressed(out_dir / f"{stem}.npz", X=X, labels=labels.to_dict(orient="list"))
            feature_paths[f"{lname}/{anchor}"] = f"{stem}.npz"

        for target in ["total_count"]:
            y = labels[target].to_numpy()
            reg = fit_ridge(X, y.astype(float), args.seed)
            cls = fit_logistic(X, y, args.seed)
            summary_rows.append(
                {
                    "layer": lname,
                    "anchor": anchor,
                    "target": target,
                    "probe_type": "ridge",
                    **reg,
                }
            )
            summary_rows.append(
                {
                    "layer": lname,
                    "anchor": anchor,
                    "target": target,
                    "probe_type": "logistic",
                    **cls,
                }
            )

        if "running_count" in labels.columns and labels["running_count"].notna().any():
            y = labels["running_count"].fillna(-1).to_numpy()
            valid = y >= 0
            if valid.any():
                reg = fit_ridge(X[valid], y[valid].astype(float), args.seed)
                cls = fit_logistic(X[valid], y[valid].astype(int), args.seed)
                summary_rows.append({"layer": lname, "anchor": anchor, "target": "running_count", "probe_type": "ridge", **reg})
                summary_rows.append(
                    {"layer": lname, "anchor": anchor, "target": "running_count", "probe_type": "logistic", **cls}
                )

        if "k" in labels.columns and labels["k"].notna().any():
            y = labels["k"].fillna(-1).to_numpy()
            valid = y >= 0
            if valid.any():
                reg = fit_ridge(X[valid], y[valid].astype(float), args.seed)
                cls = fit_logistic(X[valid], y[valid].astype(int), args.seed)
                summary_rows.append({"layer": lname, "anchor": anchor, "target": "trace_k", "probe_type": "ridge", **reg})
                summary_rows.append({"layer": lname, "anchor": anchor, "target": "trace_k", "probe_type": "logistic", **cls})

        if "is_marker" in labels.columns and labels["is_marker"].notna().any():
            y = labels["is_marker"].fillna(-1).to_numpy()
            valid = y >= 0
            if valid.any():
                cls = fit_logistic(X[valid], y[valid].astype(int), args.seed)
                summary_rows.append({"layer": lname, "anchor": anchor, "target": "is_marker", "probe_type": "logistic", **cls})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "probe_summary.csv", index=False)
    save_json(
        {
            "checkpoint": str(checkpoint),
            "split": args.split,
            "limit": args.limit,
            "anchors": sorted(anchors),
            "layers": args.layers,
            "feature_files": feature_paths,
            "rows": summary.fillna("nan").to_dict(orient="records"),
        },
        out_dir / "probe_summary.json",
    )
    print(f"saved probe summary to {out_dir}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract hidden states at task anchors and fit linear probes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="val_id")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--anchors", default="ans,think_open,think_close,source,trace_index,trace_marker")
    parser.add_argument("--layers", default="-1", help="'all' or comma-separated hidden-state indices; -1 is final layer.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_features", action="store_true")
    return parser


def main() -> None:
    run_probes(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
