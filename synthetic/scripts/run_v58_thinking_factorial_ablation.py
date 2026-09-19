#!/usr/bin/env python
"""Factorial Thinking targeted-query x answer-query broad ablation for v58."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_v22_modes_ncc import ModeSpec, _unique_run  # noqa: E402
from run_v22_free_running_topk import (  # noqa: E402
    _balanced_reporting_subset,
    _summary,
)
from run_v22_topk_ncc import _format_heads, _heads_from_ranking, _load_model, _rank_heads  # noqa: E402
from synthetic_counting_v20.data import example_from_dict, render_v20  # noqa: E402
from synthetic_counting_v20.extended_analysis import _broad_metric_matrices  # noqa: E402
from synthetic_counting_v20.training import _parse_generation  # noqa: E402
from synthetic_counting_v20.v10_port_analysis import _local_attention_edit  # noqa: E402


Head = tuple[int, int]


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [example_from_dict(json.loads(line)) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _thinking_broad_ranking(model, cfg, vocab, examples) -> pd.DataFrame:
    items = [render_v20(example, vocab, "thinking") for example in examples]
    matrices, observations = _broad_metric_matrices(model, cfg, vocab, items)
    rows = []
    for layer in range(cfg.n_layer):
        for head in range(cfg.n_head):
            rows.append(
                {
                    "role": "thinking_broad",
                    "layer": layer + 1,
                    "head": head,
                    "selection_score": float(matrices["broad_score"][layer, head]),
                    "selection_observations": int(observations),
                    "selection_split": "heldout_head_selection",
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["selection_score", "layer", "head"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    result.insert(1, "rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


@torch.no_grad()
def _factorial_rows(
    model,
    cfg,
    vocab,
    examples,
    *,
    targeted_heads: Sequence[Head],
    broad_heads: Sequence[Head],
    successor_heads: Sequence[Head] = (),
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    separator_id = vocab.token_to_id["<Sep>"]
    answer_id = vocab.token_to_id["<Ans>"]
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prefixes = []
        for example in chunk:
            item = render_v20(example, vocab, "thinking")
            assert item.spans is not None
            prefixes.append(item.tokens[: int(item.spans.think_pos) + 1])
        if len({len(prefix) for prefix in prefixes}) != 1:
            raise RuntimeError("generation prefixes must be length-aligned")
        generated = torch.tensor(
            [vocab.encode(prefix) for prefix in prefixes], dtype=torch.long, device=cfg.device
        )
        done = torch.zeros(len(chunk), dtype=torch.bool, device=cfg.device)
        max_new_tokens = int(cfg.max_render_len - len(prefixes[0]) + 2)
        for _ in range(max_new_tokens):
            separator_positions = [
                [p for p in torch.nonzero(row.eq(separator_id), as_tuple=False).flatten().tolist()
                 if p >= len(prefixes[0])]
                for row in generated
            ]
            answer_positions = [
                torch.nonzero(row.eq(answer_id), as_tuple=False).flatten().tolist()
                for row in generated
            ]
            trace_start = len(prefixes[0])
            marker_positions = []
            for row in generated:
                following_separator = (
                    torch.nonzero(row[:-1].eq(separator_id), as_tuple=False).flatten() + 1
                )
                marker_positions.append(
                    [int(position) for position in following_separator if int(position) >= trace_start]
                )
            with ExitStack() as stack:
                if targeted_heads:
                    stack.enter_context(
                        _local_attention_edit(model, targeted_heads, separator_positions)
                    )
                if broad_heads:
                    stack.enter_context(_local_attention_edit(model, broad_heads, answer_positions))
                if successor_heads:
                    stack.enter_context(
                        _local_attention_edit(model, successor_heads, marker_positions)
                    )
                next_ids = model(input_ids=generated).logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(done, torch.full_like(next_ids, vocab.eos_id), next_ids)
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            done |= next_ids.eq(vocab.eos_id)
            if bool(done.all()):
                break
        for offset, (example, token_ids) in enumerate(
            zip(chunk, generated.detach().cpu().tolist(), strict=True)
        ):
            parsed = _parse_generation(vocab.decode(token_ids), vocab, example, "thinking")
            rows.append(
                {
                    "row_id": int(start + offset),
                    "mode": "thinking",
                    "count": int(example.count or 0),
                    "prompt_sha256": example.prompt_sha256,
                    **parsed,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--paired-run-prefix", required=True)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples-per-count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    run_dir = _unique_run(args.results_root.resolve(), args.paired_run_prefix)
    spec = ModeSpec("thinking", args.paired_run_prefix, "thinking", "separator")
    cfg, vocab, _train, selection, _reporting, model = _load_model(
        run_dir, spec, device=args.device
    )
    external = _load_jsonl(args.examples_jsonl.resolve())
    selection_keys = {(item.set_id, int(item.corpus_start)) for item in selection}
    external_keys = {(item.set_id, int(item.corpus_start)) for item in external}
    overlap = selection_keys & external_keys
    if overlap:
        raise ValueError(f"confirmation/selection overlap={len(overlap)}")
    reporting = _balanced_reporting_subset(
        external,
        args.examples_per_count,
        count_min=int(cfg.count_min),
        count_max=int(cfg.count_max_threshold),
    )

    targeted_ranking = _rank_heads(run_dir, spec, cfg, vocab, selection, model)
    broad_ranking = _thinking_broad_ranking(model, cfg, vocab, selection)
    targeted = _heads_from_ranking(targeted_ranking)[: args.top_k]
    broad = _heads_from_ranking(broad_ranking)[: args.top_k]
    arms = (
        ("clean", [], []),
        ("targeted_only", targeted, []),
        ("broad_only", [], broad),
        ("joint", targeted, broad),
    )
    details = []
    summaries = []
    for arm, targeted_heads, broad_heads in arms:
        print(
            f"[{arm}] targeted={_format_heads(targeted_heads)} broad={_format_heads(broad_heads)}",
            flush=True,
        )
        frame = _factorial_rows(
            model,
            cfg,
            vocab,
            reporting,
            targeted_heads=targeted_heads,
            broad_heads=broad_heads,
            batch_size=args.batch_size,
        )
        frame.insert(0, "broad_heads", _format_heads(broad_heads))
        frame.insert(0, "targeted_heads", _format_heads(targeted_heads))
        frame.insert(0, "arm", arm)
        details.append(frame)
        summaries.append(
            {
                "arm": arm,
                "targeted_heads": _format_heads(targeted_heads),
                "broad_heads": _format_heads(broad_heads),
                **_summary(frame),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(details, ignore_index=True).to_csv(args.output / "factorial_detail.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "factorial_summary.csv", index=False)
    targeted_ranking.to_csv(args.output / "targeted_ranking.csv", index=False)
    broad_ranking.to_csv(args.output / "broad_ranking.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v58_thinking_factorial_ablation_v1",
                "source_run": run_dir.name,
                "checkpoint_step": int(cfg.train_steps),
                "examples_sha256": _sha256(args.examples_jsonl.resolve()),
                "examples_per_count": int(args.examples_per_count),
                "selection_overlap": len(overlap),
                "top_k": int(args.top_k),
                "targeted_scope": "all generated <Sep> query positions",
                "broad_scope": "all generated <Ans> query positions",
                "generation": "greedy until EOS; no gold continuation after <Think>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
