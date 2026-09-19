from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .eval import parse_generation, resolve_checkpoint_path
from .io_utils import ensure_dir, read_jsonl, save_json, write_jsonl
from .model import load_model_from_checkpoint
from .steering import load_direction, parse_float_list
from .tokenizer import VocabTokenizer


def parse_direction_specs(value: str) -> list[tuple[str, str, str]]:
    specs = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) != 3:
            raise ValueError(f"Invalid direction spec {part!r}; expected layer:anchor:target")
        specs.append((pieces[0], pieces[1], pieces[2]))
    return specs


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


def _layer_to_block_index(layer: str, model: torch.nn.Module) -> int:
    if layer == "embeddings":
        raise ValueError("Generation steering currently supports transformer block layers, not embeddings.")
    if not layer.startswith("layer_"):
        raise ValueError(f"Unknown layer name {layer!r}")
    block_idx = int(layer.removeprefix("layer_")) - 1
    n_blocks = len(model.transformer.h)
    if block_idx < 0 or block_idx >= n_blocks:
        raise ValueError(f"{layer!r} is out of range for a model with {n_blocks} transformer blocks.")
    return block_idx


def _add_direction_to_last_token(output: Any, direction: torch.Tensor, alpha: float) -> Any:
    if isinstance(output, tuple):
        hidden = output[0]
        hidden = hidden.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + float(alpha) * direction.to(device=hidden.device, dtype=hidden.dtype)
        return (hidden, *output[1:])
    hidden = output.clone()
    hidden[:, -1, :] = hidden[:, -1, :] + float(alpha) * direction.to(device=hidden.device, dtype=hidden.dtype)
    return hidden


@torch.no_grad()
def generate_with_answer_steering(
    *,
    model: torch.nn.Module,
    tokenizer: VocabTokenizer,
    prefix_tokens: list[str],
    task_format: str,
    device: str,
    layer: str,
    direction: torch.Tensor,
    alpha: float,
    max_new_tokens: int,
) -> list[str]:
    ids = tokenizer.encode(prefix_tokens)
    eos_id = tokenizer.eos_token_id
    ans_id = tokenizer.token_to_id["<ANS>"]
    seen_ans = False
    block_idx = _layer_to_block_index(layer, model)
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        if seen_ans and alpha != 0:
            handle = model.transformer.h[block_idx].register_forward_hook(
                lambda _module, _inputs, output: _add_direction_to_last_token(output, direction, alpha)
            )
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                handle.remove()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1]
        next_id = int(torch.argmax(logits).item())
        ids.append(next_id)
        if next_id == ans_id:
            seen_ans = True
        if next_id == eos_id:
            break
    return tokenizer.decode(ids[len(prefix_tokens) :])


def run_generation_steering(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    tokenizer = VocabTokenizer.load(data_dir / "vocab.json")
    examples = read_jsonl(data_dir / f"{args.split}.jsonl", limit=args.limit)
    model = load_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    max_new_tokens = args.max_new_tokens or (3 * int(tokenizer.metadata.get("max_count", 64)) + 4)
    alphas = parse_float_list(args.alphas)
    specs = parse_direction_specs(args.direction_specs)

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for layer, anchor, target in specs:
        method = f"{layer}:{anchor}:{target}"
        try:
            direction_np = load_direction(args.direction_dir, layer=layer, anchor=anchor, target=target)
            _layer_to_block_index(layer, model)
        except KeyError as exc:
            print(f"[skip] missing direction {method}: {exc}", flush=True)
            continue
        except ValueError as exc:
            print(f"[skip] unsupported steering layer {method}: {exc}", flush=True)
            continue
        direction = torch.tensor(direction_np, dtype=torch.float32, device=device)
        for alpha in alphas:
            correct = 0
            valid = 0
            true_counts = []
            pred_counts = []
            abs_errors = []
            for example in tqdm(examples, desc=f"gen-steer {method} alpha={alpha:g}", dynamic_ncols=True):
                prefix = example["full_tokens"][: example["spans"]["source_end_exclusive"]]
                task_format = example.get("task_format", "think_trace")
                generated = generate_with_answer_steering(
                    model=model,
                    tokenizer=tokenizer,
                    prefix_tokens=prefix,
                    task_format=task_format,
                    device=device,
                    layer=layer,
                    direction=direction,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                )
                parsed = parse_generation(generated, task_format=task_format)
                true_count = int(example["count"])
                pred_count = parsed.get("pred_count")
                is_valid = bool(parsed.get("format_valid", False))
                is_correct = pred_count == true_count
                valid += int(is_valid)
                correct += int(is_correct)
                true_counts.append(true_count)
                if pred_count is not None:
                    pred_counts.append(int(pred_count))
                    abs_errors.append(abs(int(pred_count) - true_count))
                prediction_rows.append(
                    {
                        "method": method,
                        "alpha": alpha,
                        "example_id": example["example_id"],
                        "split": args.split,
                        "seq_len": int(example["seq_len"]),
                        "true_count": true_count,
                        "pred_count": pred_count,
                        "format_valid": is_valid,
                        "correct": is_correct,
                        "invalid_reason": parsed.get("invalid_reason"),
                        "generated_tokens": generated,
                    }
                )
            summary_rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "split": args.split,
                    "n": len(examples),
                    "accuracy": correct / max(len(examples), 1),
                    "format_validity": valid / max(len(examples), 1),
                    "mae": float(np.mean(abs_errors)) if abs_errors else None,
                    "mean_true_count": float(np.mean(true_counts)) if true_counts else None,
                    "mean_pred_count": float(np.mean(pred_counts)) if pred_counts else None,
                    "direction_layer": layer,
                    "direction_anchor": anchor,
                    "direction_target": target,
                }
            )

    write_jsonl(prediction_rows, out_dir / "generation_steering_predictions.jsonl")
    _write_csv(summary_rows, out_dir / "generation_steering_summary.csv")
    save_json(
        {
            "checkpoint": str(checkpoint),
            "data_dir": str(data_dir),
            "split": args.split,
            "limit": args.limit,
            "direction_dir": str(args.direction_dir),
            "direction_specs": [":".join(spec) for spec in specs],
            "alphas": alphas,
            "max_new_tokens": max_new_tokens,
        },
        out_dir / "generation_steering_config.json",
    )
    print(f"saved generation steering results to {out_dir}")
    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoregressive answer-span steering for repeated-count answers.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="val_count_ood")
    parser.add_argument("--direction_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--direction_specs", default="layer_4:ans:total_count,layer_4:source_marker:running_count")
    parser.add_argument("--alphas", default="-4,-2,-1,0,1,2,4")
    return parser


def main() -> None:
    run_generation_steering(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
