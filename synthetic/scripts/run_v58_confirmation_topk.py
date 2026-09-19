#!/usr/bin/env python
"""High-power v58 Top-K ablation on the frozen confirmation suite.

This diagnostic keeps the discovery ranking and intervention implementation
from ``run_v22_free_running_topk.py`` but replaces its 8/count reporting split
with the independently generated, canonical-disjoint v58 confirmation suite.
"""

from __future__ import annotations

import argparse
import hashlib
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
from run_v22_free_running_topk import (  # noqa: E402
    _balanced_reporting_subset,
    _free_running_rows,
    _summary,
)
from run_v22_topk_ncc import (  # noqa: E402
    _format_heads,
    _heads_from_ranking,
    _load_model,
    _matched_control_sets,
    _rank_heads,
)
from synthetic_counting_v20.data import example_from_dict  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path):
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    examples.append(example_from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"invalid example at {path}:{line_number}") from exc
    return examples


def _example_keys(examples) -> set[tuple[str | None, int]]:
    return {(example.set_id, int(example.corpus_start)) for example in examples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--paired-run-prefix", required=True)
    parser.add_argument("--expected-version", default="v58")
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples-per-count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument(
        "--modes", nargs="+", choices=("thinking", "nonthinking"), default=("thinking",)
    )
    args = parser.parse_args()

    run_dir = _unique_run(args.results_root.resolve(), args.paired_run_prefix)
    external_examples = _load_jsonl(args.examples_jsonl.resolve())
    args.output.mkdir(parents=True, exist_ok=True)

    detail_frames = []
    summary_rows = []
    ranking_frames = []
    overlap_by_mode: dict[str, int] = {}
    for mode in args.modes:
        spec = ModeSpec(mode, args.paired_run_prefix, mode, "separator")
        cfg, vocab, _train, selection, _reporting, model = _load_model(
            run_dir, spec, device=args.device
        )
        if cfg.version != args.expected_version:
            raise ValueError(f"expected version={args.expected_version!r}, got {cfg.version!r}")

        overlap = _example_keys(selection) & _example_keys(external_examples)
        overlap_by_mode[mode] = len(overlap)
        if overlap:
            raise ValueError(f"{mode}: confirmation/selection overlap={len(overlap)}")
        reporting = _balanced_reporting_subset(
            external_examples,
            args.examples_per_count,
            count_min=int(cfg.count_min),
            count_max=int(cfg.count_max_threshold),
        )
        ranking = _rank_heads(run_dir, spec, cfg, vocab, selection, model)
        ranked_heads = _heads_from_ranking(ranking)
        ranking_frames.append(ranking.assign(comparison_mode=mode, source_run=run_dir.name))

        arms = [("clean", 0, 0, [])]
        for top_k in args.top_k:
            top_k = int(top_k)
            arms.append(("ranked", 0, top_k, ranked_heads[:top_k]))
            controls = _matched_control_sets(
                ranked_heads, top_k=top_k, n_layer=cfg.n_layer, n_head=cfg.n_head
            )
            arms.extend(
                ("layer_matched_control", path_id, top_k, control)
                for path_id, control in enumerate(controls, start=1)
            )

        for path_kind, path_id, top_k, heads in arms:
            label = _format_heads(heads)
            print(f"[{mode}] confirmation {path_kind} K={top_k} {label}", flush=True)
            detail = _free_running_rows(
                model,
                cfg,
                vocab,
                reporting,
                mode=mode,
                heads=heads,
                batch_size=min(args.batch_size, len(reporting)),
            )
            for column, value in {
                "comparison_mode": mode,
                "path_kind": path_kind,
                "path_id": int(path_id),
                "top_k": int(top_k),
                "heads": label,
            }.items():
                detail.insert(0, column, value)
            detail_frames.append(detail)
            summary_rows.append(
                {
                    "comparison_mode": mode,
                    "scope": "confirmation_role_query_local_free_running",
                    "path_kind": path_kind,
                    "path_id": int(path_id),
                    "top_k": int(top_k),
                    "heads": label,
                    **_summary(detail),
                }
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.concat(ranking_frames, ignore_index=True).to_csv(
        args.output / "head_rankings.csv", index=False
    )
    pd.concat(detail_frames, ignore_index=True).to_csv(
        args.output / "free_running_detail.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(args.output / "free_running_summary.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v58_confirmation_topk_v1",
                "source_run": run_dir.name,
                "checkpoint_step": int(cfg.train_steps),
                "modes": list(args.modes),
                "examples_jsonl": str(args.examples_jsonl.resolve()),
                "examples_sha256": _sha256(args.examples_jsonl.resolve()),
                "examples_per_count": int(args.examples_per_count),
                "selection_overlap_by_mode": overlap_by_mode,
                "selection": "frozen discovery ranking from heldout_head_selection",
                "intervention": "same local attention-output zeroing as run_v22_free_running_topk.py",
                "generation": "greedy until EOS; no gold continuation after mode prefix",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
