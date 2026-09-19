from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .eval import resolve_checkpoint_path
from .io_utils import ensure_dir, read_jsonl, save_json, write_jsonl
from .model import load_model_from_checkpoint
from .probes import layer_name
from .tokenizer import VocabTokenizer


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_direction(direction_dir: str | Path, *, layer: str, anchor: str, target: str) -> np.ndarray:
    direction_dir = Path(direction_dir)
    key = "__".join([layer, anchor, target])
    arrays = np.load(direction_dir / "directions.npz")
    array_key = f"{key}__coef"
    if array_key not in arrays:
        available = sorted(name.removesuffix("__coef") for name in arrays.files if name.endswith("__coef"))
        raise KeyError(f"Missing direction {key!r}. Available directions: {available}")
    direction = arrays[array_key].astype(np.float32)
    norm = np.linalg.norm(direction)
    if norm <= 1e-8:
        raise ValueError(f"Direction {key!r} has near-zero norm.")
    return direction / norm


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def steered_count_logits(
    *,
    model: torch.nn.Module,
    tokenizer: VocabTokenizer,
    example: dict,
    device: str,
    layer_idx: int,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    ans_idx = example["spans"]["ans_idx"]
    prefix_ids = tokenizer.encode(example["full_tokens"][: ans_idx + 1])
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if layer_idx != len(hidden_states) - 1:
        raise ValueError(
            "This lightweight steering script currently supports final hidden-state directions only. "
            f"Requested layer_idx={layer_idx}, final_idx={len(hidden_states) - 1}."
        )
    hidden = hidden_states[layer_idx][0, -1].float()
    modified = hidden + float(alpha) * direction
    logits = model.lm_head(modified)
    count_ids = torch.tensor(tokenizer.count_token_ids, dtype=torch.long, device=device)
    return logits.index_select(0, count_ids)


def run_steering(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    tokenizer = VocabTokenizer.load(data_dir / "vocab.json")
    examples = read_jsonl(data_dir / f"{args.split}.jsonl", limit=args.limit)
    model = load_model_from_checkpoint(checkpoint).to(device)
    model.eval()

    # Load once to discover the final hidden-state layer name.
    dummy_ids = torch.tensor([tokenizer.encode(examples[0]["full_tokens"][: examples[0]["spans"]["ans_idx"] + 1])], dtype=torch.long, device=device)
    dummy_outputs = model(input_ids=dummy_ids, output_hidden_states=True)
    final_layer_idx = len(dummy_outputs.hidden_states) - 1
    requested_layer = args.layer if args.layer != "final" else layer_name(final_layer_idx)
    if requested_layer != layer_name(final_layer_idx):
        raise ValueError(f"Use --layer final or --layer {layer_name(final_layer_idx)} for this steering script.")

    direction_np = load_direction(args.direction_dir, layer=requested_layer, anchor=args.anchor, target=args.target)
    direction = torch.tensor(direction_np, dtype=torch.float32, device=device)
    alphas = parse_float_list(args.alphas)
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for alpha in alphas:
        correct = 0
        abs_errors = []
        pred_counts = []
        true_counts = []
        true_probs = []
        for example in tqdm(examples, desc=f"steer alpha={alpha:g}", dynamic_ncols=True):
            count_logits = steered_count_logits(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                layer_idx=final_layer_idx,
                direction=direction,
                alpha=alpha,
            )
            pred_count = int(torch.argmax(count_logits).item())
            true_count = int(example["count"])
            probs = F.softmax(count_logits, dim=-1)
            correct += int(pred_count == true_count)
            abs_errors.append(abs(pred_count - true_count))
            pred_counts.append(pred_count)
            true_counts.append(true_count)
            true_probs.append(float(probs[true_count].detach().cpu()))
            prediction_rows.append(
                {
                    "alpha": alpha,
                    "example_id": example["example_id"],
                    "split": args.split,
                    "seq_len": int(example["seq_len"]),
                    "true_count": true_count,
                    "pred_count": pred_count,
                    "true_prob": true_probs[-1],
                }
            )
        summary_rows.append(
            {
                "alpha": alpha,
                "split": args.split,
                "n": len(examples),
                "accuracy": correct / max(len(examples), 1),
                "mae": float(np.mean(abs_errors)) if abs_errors else None,
                "mean_true_count": float(np.mean(true_counts)) if true_counts else None,
                "mean_pred_count": float(np.mean(pred_counts)) if pred_counts else None,
                "mean_true_prob": float(np.mean(true_probs)) if true_probs else None,
                "direction_layer": requested_layer,
                "direction_anchor": args.anchor,
                "direction_target": args.target,
            }
        )

    write_jsonl(prediction_rows, out_dir / "steering_predictions.jsonl")
    _write_csv(summary_rows, out_dir / "steering_summary.csv")
    save_json(
        {
            "checkpoint": str(checkpoint),
            "data_dir": str(data_dir),
            "split": args.split,
            "limit": args.limit,
            "direction_dir": str(args.direction_dir),
            "layer": requested_layer,
            "anchor": args.anchor,
            "target": args.target,
            "alphas": alphas,
        },
        out_dir / "steering_config.json",
    )
    print(f"saved steering results to {out_dir}")
    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Steer final answer hidden states along a ridge count direction.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="val_count_ood")
    parser.add_argument("--direction_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--layer", default="final")
    parser.add_argument("--anchor", default="ans")
    parser.add_argument("--target", default="total_count")
    parser.add_argument("--alphas", default="-4,-2,-1,0,1,2,4")
    return parser


def main() -> None:
    run_steering(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
