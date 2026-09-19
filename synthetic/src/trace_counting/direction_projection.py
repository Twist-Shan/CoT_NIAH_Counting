from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .directions import _safe_key
from .eval import resolve_checkpoint_path
from .io_utils import ensure_dir, read_jsonl, save_json
from .model import load_model_from_checkpoint
from .probes import anchor_positions, layer_name
from .tokenizer import VocabTokenizer


EXAMPLE_COLUMNS = [
    "split",
    "example_id",
    "seq_len",
    "true_count",
    "anchor",
    "layer",
    "target",
    "target_value",
    "projection",
    "pred_value",
]

SUMMARY_COLUMNS = [
    "split",
    "anchor",
    "layer",
    "target",
    "group_key",
    "group_value",
    "n",
    "mean_true_count",
    "mean_target_value",
    "mean_projection",
    "mean_pred_value",
    "mae_pred_value",
]


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def _load_direction_rows(direction_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    with (direction_dir / "direction_summary.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ok") != "True":
                continue
            key = (row["layer"], row["anchor"], row["target"])
            rows[key] = row
    return rows


def _parse_specs(specs: str) -> list[tuple[str, str, str]]:
    out = []
    for part in specs.split(","):
        part = part.strip()
        if not part:
            continue
        fields = [x.strip() for x in part.split(":")]
        if len(fields) != 3:
            raise ValueError(f"Invalid spec {part!r}; expected layer:anchor:target")
        out.append((fields[0], fields[1], fields[2]))
    return out


def _layer_index(layer: str) -> int:
    if layer == "embeddings":
        return 0
    if not layer.startswith("layer_"):
        raise ValueError(f"Unknown layer name {layer!r}")
    return int(layer.removeprefix("layer_"))


def run_projection(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    data_dir = Path(args.data_dir)
    direction_dir = Path(args.direction_dir)
    out_dir = ensure_dir(args.out_dir)

    tokenizer = VocabTokenizer.load(data_dir / "vocab.json")
    examples = read_jsonl(data_dir / f"{args.split}.jsonl", limit=args.limit)
    model = load_model_from_checkpoint(checkpoint).to(device)
    model.eval()

    direction_arrays = np.load(direction_dir / "directions.npz")
    direction_rows = _load_direction_rows(direction_dir)
    specs = _parse_specs(args.specs)
    missing = [spec for spec in specs if spec not in direction_rows]
    if missing:
        for spec in missing:
            print(f"[skip] missing direction row for {':'.join(spec)}", flush=True)
        specs = [spec for spec in specs if spec in direction_rows]

    needed_layers = sorted({_layer_index(layer) for layer, _, _ in specs})
    anchors = {anchor for _, anchor, _ in specs}
    example_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for example in tqdm(examples, desc=f"project directions {args.split}", dynamic_ncols=True):
            input_ids = torch.tensor([tokenizer.encode(example["full_tokens"])], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            positions = anchor_positions(example, anchors)
            positions_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for pos in positions:
                positions_by_anchor[pos["anchor"]].append(pos)

            for layer, anchor, target in specs:
                layer_idx = _layer_index(layer)
                if layer_idx >= len(hidden_states):
                    raise ValueError(f"{layer} is out of range for checkpoint with {len(hidden_states)} hidden-state tensors")
                row = direction_rows[(layer, anchor, target)]
                array_key = _safe_key(layer, anchor, target) + "__coef"
                coef = direction_arrays[array_key].astype(np.float64)
                intercept = float(row["intercept"])
                hidden = hidden_states[layer_idx][0].detach().float().cpu().numpy().astype(np.float64)
                for pos in positions_by_anchor.get(anchor, []):
                    target_value = pos.get(target)
                    if target_value is None:
                        continue
                    projection = float(hidden[pos["idx"]] @ coef)
                    pred_value = projection + intercept
                    example_rows.append(
                        {
                            "split": args.split,
                            "example_id": example["example_id"],
                            "seq_len": int(example["seq_len"]),
                            "true_count": int(example["count"]),
                            "anchor": anchor,
                            "layer": layer,
                            "target": target,
                            "target_value": float(target_value),
                            "projection": projection,
                            "pred_value": pred_value,
                        }
                    )

    summary_rows: list[dict[str, Any]] = []
    group_keys = ["true_count", "target_value"]
    for group_key in group_keys:
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in example_rows:
            key = (row["split"], row["anchor"], row["layer"], row["target"], group_key, row[group_key])
            groups[key].append(row)
        for (split, anchor, layer, target, key_name, key_value), rows in sorted(groups.items()):
            errors = [abs(float(row["pred_value"]) - float(row["target_value"])) for row in rows]
            summary_rows.append(
                {
                    "split": split,
                    "anchor": anchor,
                    "layer": layer,
                    "target": target,
                    "group_key": key_name,
                    "group_value": key_value,
                    "n": len(rows),
                    "mean_true_count": _mean([row["true_count"] for row in rows]),
                    "mean_target_value": _mean([row["target_value"] for row in rows]),
                    "mean_projection": _mean([row["projection"] for row in rows]),
                    "mean_pred_value": _mean([row["pred_value"] for row in rows]),
                    "mae_pred_value": _mean(errors),
                }
            )

    _write_csv(example_rows, out_dir / "direction_projection_examples.csv", EXAMPLE_COLUMNS)
    _write_csv(summary_rows, out_dir / "direction_projection_summary.csv", SUMMARY_COLUMNS)
    save_json(
        {
            "checkpoint": str(checkpoint),
            "data_dir": str(data_dir),
            "direction_dir": str(direction_dir),
            "split": args.split,
            "limit": args.limit,
            "specs": [":".join(spec) for spec in specs],
            "device": device,
        },
        out_dir / "direction_projection_config.json",
    )
    print(f"saved direction projection to {out_dir}")
    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project held-out examples onto fitted ridge count directions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--direction_dir", required=True)
    parser.add_argument("--split", default="val_count_ood")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument(
        "--specs",
        default="layer_1:source_marker:running_count,layer_4:source_marker:running_count,layer_4:ans:total_count",
        help="Comma-separated layer:anchor:target direction specs.",
    )
    return parser


def main() -> None:
    run_projection(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
