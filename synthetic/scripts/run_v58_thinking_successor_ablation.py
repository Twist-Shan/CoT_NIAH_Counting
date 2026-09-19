#!/usr/bin/env python
"""Free-running v58 successor-head ablation on frozen confirmation examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_v22_modes_ncc import ModeSpec, _unique_run  # noqa: E402
from run_v22_free_running_topk import _balanced_reporting_subset, _summary  # noqa: E402
from run_v22_topk_ncc import _format_heads, _heads_from_ranking, _load_model, _rank_heads  # noqa: E402
from run_v58_thinking_factorial_ablation import _factorial_rows  # noqa: E402
from synthetic_counting_v20.data import example_from_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--paired-run-prefix", required=True)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples-per-count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    run_dir = _unique_run(args.results_root.resolve(), args.paired_run_prefix)
    spec = ModeSpec("thinking", args.paired_run_prefix, "thinking", "separator")
    cfg, vocab, _train, selection, _reporting, model = _load_model(
        run_dir, spec, device=args.device
    )
    with args.examples_jsonl.open("r", encoding="utf-8") as handle:
        external = [example_from_dict(json.loads(line)) for line in handle if line.strip()]
    overlap = {
        (item.set_id, int(item.corpus_start)) for item in selection
    } & {(item.set_id, int(item.corpus_start)) for item in external}
    if overlap:
        raise ValueError(f"confirmation/selection overlap={len(overlap)}")
    reporting = _balanced_reporting_subset(
        external,
        args.examples_per_count,
        count_min=int(cfg.count_min),
        count_max=int(cfg.count_max_threshold),
    )

    targeted_ranking = _rank_heads(run_dir, spec, cfg, vocab, selection, model)
    targeted = _heads_from_ranking(targeted_ranking)[:2]
    successor_ranking = pd.read_csv(
        run_dir / "analysis" / "phase_transition" / "tables" / "fixed_head_rankings.csv"
    )
    successor_ranking = successor_ranking[
        successor_ranking["role"].eq("marker_successor")
    ].sort_values("rank", kind="mergesort")
    successor = [
        (int(successor_ranking.iloc[0]["layer"]), int(successor_ranking.iloc[0]["head"]))
    ]
    selected_layer = successor[0][0]
    same_layer = successor_ranking[
        successor_ranking["layer"].astype(int).eq(selected_layer)
        & ~successor_ranking["head"].astype(int).eq(successor[0][1])
    ]
    control_row = same_layer.sort_values("selection_score", kind="mergesort").iloc[0]
    control = [(int(control_row["layer"]), int(control_row["head"]))]

    arms = (
        ("clean", [], []),
        ("targeted_top2", targeted, []),
        ("successor_top1", [], successor),
        ("successor_same_layer_low_score", [], control),
        ("targeted_top2_plus_successor_top1", targeted, successor),
    )
    details = []
    summaries = []
    for arm, targeted_heads, successor_heads in arms:
        print(
            f"[{arm}] targeted={_format_heads(targeted_heads)} "
            f"successor={_format_heads(successor_heads)}",
            flush=True,
        )
        frame = _factorial_rows(
            model,
            cfg,
            vocab,
            reporting,
            targeted_heads=targeted_heads,
            broad_heads=[],
            successor_heads=successor_heads,
            batch_size=args.batch_size,
        )
        frame.insert(0, "successor_heads", _format_heads(successor_heads))
        frame.insert(0, "targeted_heads", _format_heads(targeted_heads))
        frame.insert(0, "arm", arm)
        details.append(frame)
        summaries.append(
            {
                "arm": arm,
                "targeted_heads": _format_heads(targeted_heads),
                "successor_heads": _format_heads(successor_heads),
                **_summary(frame),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(details, ignore_index=True).to_csv(args.output / "successor_detail.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "successor_summary.csv", index=False)
    successor_ranking.to_csv(args.output / "successor_ranking.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v58_thinking_successor_ablation_v1",
                "source_run": run_dir.name,
                "checkpoint_step": int(cfg.train_steps),
                "examples_per_count": int(args.examples_per_count),
                "selection_overlap": len(overlap),
                "targeted_scope": "all generated <Sep> query positions",
                "successor_scope": "generated marker positions immediately following <Sep>",
                "generation": "greedy until EOS; no gold continuation after <Think>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
