from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .eval import resolve_checkpoint_path
from .io_utils import ensure_dir, read_jsonl, save_json
from .model import load_model_from_checkpoint
from .tokenizer import VocabTokenizer

PER_HEAD_COLUMNS = [
    "split",
    "example_id",
    "task_format",
    "seq_len",
    "count",
    "query_anchor",
    "layer",
    "head",
    "source_mass",
    "marker_mass",
    "noise_mass",
    "marker_per_token",
    "noise_per_token",
    "marker_enrichment",
    "top_source_is_marker",
]

SUMMARY_COLUMNS = [
    "split",
    "task_format",
    "query_anchor",
    "layer",
    "head",
    "n",
    "mean_count",
    "source_mass",
    "marker_mass",
    "noise_mass",
    "marker_per_token",
    "noise_per_token",
    "marker_enrichment",
    "top_source_marker_rate",
]


def _write_csv(rows: list[dict[str, Any]], path: Path, *, fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    keys = fieldnames or sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def query_indices(example: dict, query_anchors: set[str]) -> dict[str, int]:
    spans = example["spans"]
    out: dict[str, int] = {}
    for anchor in query_anchors:
        if anchor == "ans":
            out[anchor] = spans["ans_idx"]
        elif anchor == "count":
            out[anchor] = spans["count_idx"]
        elif anchor == "think_close" and spans.get("think_close_idx") is not None:
            out[anchor] = spans["think_close_idx"]
        elif anchor == "think_open" and spans.get("think_open_idx") is not None:
            out[anchor] = spans["think_open_idx"]
    return out


@torch.no_grad()
def run_attention_analysis(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    tokenizer = VocabTokenizer.load(data_dir / "vocab.json")
    model = load_model_from_checkpoint(checkpoint, attn_implementation="eager").to(device)
    model.config.output_attentions = True
    model.eval()
    query_anchors = {part.strip() for part in args.query_anchors.split(",") if part.strip()}
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]

    per_head_rows: list[dict[str, Any]] = []
    for split in splits:
        examples = read_jsonl(data_dir / f"{split}.jsonl", limit=args.limit)
        for example in tqdm(examples, desc=f"attention {split}", dynamic_ncols=True):
            ids = tokenizer.encode(example["full_tokens"])
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, output_attentions=True)
            attentions = outputs.attentions
            if not attentions or any(layer_attention is None for layer_attention in attentions):
                raise RuntimeError(
                    "Model did not return attention tensors. This usually means the checkpoint was loaded "
                    "with an attention backend that does not materialize attention weights. "
                    "The attention analysis loads GPT-2 with attn_implementation='eager'; "
                    "if this still fails, check the installed transformers/torch versions."
                )
            spans = example["spans"]
            source_indices = list(range(spans["source_start"], spans["source_end_exclusive"]))
            marker_indices = set(pair["source_idx"] for pair in spans["trace_pairs"])
            noise_indices = [idx for idx in source_indices if idx not in marker_indices]
            anchors = query_indices(example, query_anchors)
            for query_anchor, query_idx in anchors.items():
                for layer_idx, layer_attention in enumerate(attentions, start=1):
                    # shape: batch, heads, query, key
                    layer_np = layer_attention[0, :, query_idx, :].detach().float().cpu().numpy()
                    for head_idx, weights in enumerate(layer_np):
                        source_mass = float(weights[source_indices].sum()) if source_indices else 0.0
                        marker_mass = float(weights[list(marker_indices)].sum()) if marker_indices else 0.0
                        noise_mass = float(weights[noise_indices].sum()) if noise_indices else 0.0
                        marker_per_token = marker_mass / len(marker_indices) if marker_indices else None
                        noise_per_token = noise_mass / len(noise_indices) if noise_indices else None
                        if marker_per_token is not None and noise_per_token is not None and noise_per_token > 0:
                            enrichment = marker_per_token / noise_per_token
                        else:
                            enrichment = None
                        top_source_idx = max(source_indices, key=lambda idx: float(weights[idx])) if source_indices else None
                        per_head_rows.append(
                            {
                                "split": split,
                                "example_id": example["example_id"],
                                "task_format": example.get("task_format", "think_trace"),
                                "seq_len": int(example["seq_len"]),
                                "count": int(example["count"]),
                                "query_anchor": query_anchor,
                                "layer": layer_idx,
                                "head": head_idx,
                                "source_mass": source_mass,
                                "marker_mass": marker_mass,
                                "noise_mass": noise_mass,
                                "marker_per_token": marker_per_token,
                                "noise_per_token": noise_per_token,
                                "marker_enrichment": enrichment,
                                "top_source_is_marker": None if top_source_idx is None else int(top_source_idx in marker_indices),
                            }
                        )

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in per_head_rows:
        key = (row["split"], row["task_format"], row["query_anchor"], row["layer"], row["head"])
        groups[key].append(row)
    summary_rows = []
    for (split, task_format, query_anchor, layer, head), rows in sorted(groups.items()):
        summary_rows.append(
            {
                "split": split,
                "task_format": task_format,
                "query_anchor": query_anchor,
                "layer": layer,
                "head": head,
                "n": len(rows),
                "mean_count": _mean([row["count"] for row in rows]),
                "source_mass": _mean([row["source_mass"] for row in rows]),
                "marker_mass": _mean([row["marker_mass"] for row in rows]),
                "noise_mass": _mean([row["noise_mass"] for row in rows]),
                "marker_per_token": _mean([row["marker_per_token"] for row in rows]),
                "noise_per_token": _mean([row["noise_per_token"] for row in rows]),
                "marker_enrichment": _mean([row["marker_enrichment"] for row in rows]),
                "top_source_marker_rate": _mean([row["top_source_is_marker"] for row in rows]),
            }
        )

    _write_csv(per_head_rows, out_dir / "attention_per_head_examples.csv", fieldnames=PER_HEAD_COLUMNS)
    _write_csv(summary_rows, out_dir / "attention_summary.csv", fieldnames=SUMMARY_COLUMNS)
    save_json(
        {
            "checkpoint": str(checkpoint),
            "data_dir": str(data_dir),
            "splits": splits,
            "limit": args.limit,
            "query_anchors": sorted(query_anchors),
        },
        out_dir / "attention_config.json",
    )
    if not per_head_rows:
        raise RuntimeError(
            "Attention analysis produced zero rows. Check --query_anchors and the dataset spans. "
            "For the default v1 setting, at least the 'ans' anchor should produce rows."
        )
    print(f"saved attention analysis to {out_dir}")
    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze attention from answer/think tokens to source needles.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--splits", default="val_id,val_count_ood")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--query_anchors", default="ans,think_close")
    return parser


def main() -> None:
    run_attention_analysis(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
