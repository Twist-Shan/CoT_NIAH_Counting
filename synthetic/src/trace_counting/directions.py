from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .eval import resolve_checkpoint_path
from .io_utils import ensure_dir, read_jsonl, save_json
from .model import load_model_from_checkpoint
from .probes import anchor_positions, layer_name, parse_layers
from .tokenizer import VocabTokenizer

RIDGE_ALPHAS = np.logspace(-4, 4, 17)


def _split_indices(n: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    return indices[n_test:], indices[:n_test]


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std, mean.squeeze(0), std.squeeze(0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    mae = float(np.abs(y_true - y_pred).mean())
    return {"r2": float(r2), "mae": mae}


def fit_ridge_direction(X: np.ndarray, y: np.ndarray, *, seed: int, test_size: float = 0.25) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(X) < 8 or np.unique(y).size < 2:
        return {"ok": False, "reason": "too_few_examples_or_labels", "n": int(len(X))}

    train_idx, test_idx = _split_indices(len(X), test_size, seed)
    X_train, X_test, mean, std = _standardize(X[train_idx], X[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_train_aug = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    X_test_aug = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1)

    best: dict[str, Any] | None = None
    for alpha in RIDGE_ALPHAS:
        eye = np.eye(X_train_aug.shape[1])
        eye[-1, -1] = 0.0
        coef_aug = np.linalg.solve(X_train_aug.T @ X_train_aug + float(alpha) * eye, X_train_aug.T @ y_train)
        pred = X_test_aug @ coef_aug
        metrics = _metrics(y_test, pred)
        if best is None or metrics["r2"] > best["r2"]:
            best = {
                "alpha": float(alpha),
                "coef_standardized": coef_aug[:-1],
                "intercept_standardized": float(coef_aug[-1]),
                **metrics,
            }

    assert best is not None
    coef = best["coef_standardized"] / std
    intercept = best["intercept_standardized"] - float((mean / std) @ best["coef_standardized"])
    projection = X @ coef
    norm = float(np.linalg.norm(coef))
    return {
        "ok": True,
        "coef": coef.astype(np.float32),
        "intercept": float(intercept),
        "alpha": best["alpha"],
        "r2": float(best["r2"]),
        "mae": float(best["mae"]),
        "n": int(len(X)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "direction_norm": norm,
        "projection_mean": float(projection.mean()),
        "projection_std": float(projection.std()),
    }


def _safe_key(*parts: str) -> str:
    return "__".join(part.replace("/", "_") for part in parts)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys() if key != "coef"})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def extract_direction_features(args: argparse.Namespace) -> tuple[dict[tuple[str, str], list[np.ndarray]], dict[tuple[str, str], list[dict[str, Any]]]]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    tokenizer = VocabTokenizer.load(Path(args.data_dir) / "vocab.json")
    examples = read_jsonl(Path(args.data_dir) / f"{args.split}.jsonl", limit=args.limit)
    model = load_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    anchors = {part.strip() for part in args.anchors.split(",") if part.strip()}
    feature_store: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    label_store: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    selected_layers: list[int] | None = None
    with torch.no_grad():
        for example in tqdm(examples, desc=f"directions {args.split}", dynamic_ncols=True):
            input_ids = torch.tensor([tokenizer.encode(example["full_tokens"])], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if selected_layers is None:
                selected_layers = parse_layers(args.layers, len(hidden_states))
            positions = anchor_positions(example, anchors)
            for layer_idx in selected_layers:
                hidden = hidden_states[layer_idx][0].detach().float().cpu().numpy()
                lname = layer_name(layer_idx)
                for pos in positions:
                    key = (lname, pos["anchor"])
                    feature_store[key].append(hidden[pos["idx"]])
                    label_store[key].append(pos)
    return feature_store, label_store


def run_directions(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    feature_store, label_store = extract_direction_features(args)
    targets = [part.strip() for part in args.targets.split(",") if part.strip()]
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for (lname, anchor), features in sorted(feature_store.items()):
        X = np.stack(features, axis=0)
        labels = label_store[(lname, anchor)]
        for target in targets:
            y_values = []
            valid_indices = []
            for idx, label in enumerate(labels):
                value = label.get(target)
                if value is None:
                    continue
                y_values.append(float(value))
                valid_indices.append(idx)
            if not valid_indices:
                continue
            result = fit_ridge_direction(X[np.asarray(valid_indices)], np.asarray(y_values), seed=args.seed)
            row = {
                "layer": lname,
                "anchor": anchor,
                "target": target,
                "split": args.split,
                "seed": args.seed,
                **{key: value for key, value in result.items() if key != "coef"},
            }
            if result.get("ok"):
                key = _safe_key(lname, anchor, target)
                arrays[f"{key}__coef"] = result["coef"]
                row["array_key"] = key
            rows.append(row)

    np.savez_compressed(out_dir / "directions.npz", **arrays)
    _write_csv(rows, out_dir / "direction_summary.csv")
    save_json(
        {
            "checkpoint": str(resolve_checkpoint_path(args.checkpoint)),
            "data_dir": str(args.data_dir),
            "split": args.split,
            "limit": args.limit,
            "anchors": [part.strip() for part in args.anchors.split(",") if part.strip()],
            "layers": args.layers,
            "targets": targets,
            "rows": rows,
        },
        out_dir / "direction_metadata.json",
    )
    print(f"saved ridge directions to {out_dir}")
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit ridge count directions from hidden states.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="val_id")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--anchors", default="ans,think_close,source_marker,trace_index,trace_marker")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--targets", default="total_count,running_count,k")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run_directions(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
