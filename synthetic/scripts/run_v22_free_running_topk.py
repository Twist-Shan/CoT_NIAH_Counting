#!/usr/bin/env python
"""Held-out free-running Top-K head-bank ablation for the v22 comparison.

The companion ``run_v22_topk_ncc.py`` intentionally measures teacher-forced
local behavior and clean-frozen downstream geometry.  That protocol is useful
for locating the damaged computation, but in the Thinking model the next gold
trace token can repair the rollout.  This script therefore repeats the ranked
and layer-count-matched interventions during greedy generation.

Non-thinking broad heads are removed at the answer query.  Thinking targeted
heads are removed at every generated ``<Sep>`` query.  Selection examples are
disjoint from the reporting examples, and all arms use the same reporting rows.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from compare_v22_modes_ncc import DEFAULT_RESULTS_ROOT, SPECS, _unique_run  # noqa: E402
from compare_v22_modes_ncc import ModeSpec  # noqa: E402
from run_v22_topk_ncc import (  # noqa: E402
    Head,
    _format_heads,
    _heads_from_ranking,
    _load_model,
    _matched_control_sets,
    _rank_heads,
)
from synthetic_counting_v20.data import render_v20  # noqa: E402
from synthetic_counting_v20.training import _parse_generation  # noqa: E402
from synthetic_counting_v20.v10_port_analysis import _local_attention_edit  # noqa: E402


def _balanced_reporting_subset(
    examples, per_count: int, *, count_min: int, count_max: int
):
    selected = []
    for count in range(count_min, count_max + 1):
        bucket = [example for example in examples if int(example.count or 0) == count]
        if len(bucket) < per_count:
            raise ValueError(
                f"count={count}: reporting split has {len(bucket)} rows, needs {per_count}"
            )
        selected.extend(bucket[:per_count])
    return selected


@torch.no_grad()
def _free_running_rows(
    model,
    cfg,
    vocab,
    examples,
    *,
    mode: str,
    heads: Sequence[Head],
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    separator_id = vocab.token_to_id.get("<Sep>")
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prefixes: list[list[str]] = []
        answer_positions: list[int] = []
        for example in chunk:
            item = render_v20(example, vocab, mode)
            assert item.spans is not None
            stop = (
                item.spans.ans_pos + 1
                if mode == "nonthinking"
                else int(item.spans.think_pos) + 1
            )
            prefixes.append(item.tokens[:stop])
            answer_positions.append(int(item.spans.ans_pos))
        lengths = {len(prefix) for prefix in prefixes}
        if len(lengths) != 1:
            raise RuntimeError("generation prefixes must be length-aligned")
        generated = torch.tensor(
            [vocab.encode(tokens) for tokens in prefixes],
            dtype=torch.long,
            device=cfg.device,
        )
        done = torch.zeros(len(chunk), dtype=torch.bool, device=cfg.device)
        max_new_tokens = (
            4
            if mode == "nonthinking"
            else int(cfg.max_render_len - len(prefixes[0]) + 2)
        )
        for _ in range(max_new_tokens):
            if heads:
                if mode == "nonthinking":
                    positions = [[position] for position in answer_positions]
                else:
                    assert separator_id is not None
                    positions = [
                        torch.nonzero(row.eq(separator_id), as_tuple=False)
                        .flatten()
                        .tolist()
                        for row in generated
                    ]
                intervention = _local_attention_edit(model, heads, positions)
            else:
                intervention = contextlib.nullcontext()
            with intervention:
                next_ids = model(input_ids=generated).logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(
                done,
                torch.full_like(next_ids, vocab.eos_id),
                next_ids,
            )
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            done |= next_ids.eq(vocab.eos_id)
            if bool(done.all()):
                break

        generated_cpu = generated.detach().cpu().tolist()
        for offset, (example, token_ids) in enumerate(
            zip(chunk, generated_cpu, strict=True)
        ):
            parsed = _parse_generation(vocab.decode(token_ids), vocab, example, mode)
            rows.append(
                {
                    "row_id": int(start + offset),
                    "mode": mode,
                    "count": int(example.count or 0),
                    "prompt_sha256": example.prompt_sha256,
                    **parsed,
                }
            )
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> dict[str, float]:
    result = {
        "examples": int(len(frame)),
        "ar_final_accuracy": float(frame["ar_accuracy"].mean()),
        "ar_answer_rate": float(frame["ar_answered"].mean()),
        "ar_abs_error_with_missing_penalty": float(
            frame["ar_abs_error_with_missing_penalty"].mean()
        ),
    }
    for column in (
        "trace_exact",
        "trace_ordered_marker_accuracy",
        "trace_marker_count_accuracy",
        "trace_format_valid",
        "trace_closed",
        "trace_delimiter_count_accuracy",
    ):
        result[column] = (
            float(frame[column].mean()) if frame[column].notna().any() else np.nan
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "work" / "v22_free_running_topk"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples-per-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--paired-run-prefix", default=None)
    parser.add_argument("--expected-version", default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    specs = SPECS
    if args.paired_run_prefix is not None:
        specs = (
            ModeSpec(
                "nonthinking",
                args.paired_run_prefix,
                "nonthinking",
                "separator",
            ),
            ModeSpec(
                "thinking",
                args.paired_run_prefix,
                "thinking",
                "separator",
            ),
        )

    detail_frames = []
    summary_rows = []
    ranking_frames = []
    for spec in specs:
        run_dir = _unique_run(args.results_root.resolve(), spec.run_prefix)
        cfg, vocab, _train, selection, reporting, model = _load_model(
            run_dir, spec, device=args.device
        )
        if args.expected_version is not None and cfg.version != args.expected_version:
            raise ValueError(
                f"{spec.label}: expected version={args.expected_version!r}, "
                f"got {cfg.version!r}"
            )
        reporting = _balanced_reporting_subset(
            reporting,
            args.examples_per_count,
            count_min=int(cfg.count_min),
            count_max=int(cfg.count_max_threshold),
        )
        ranking = _rank_heads(run_dir, spec, cfg, vocab, selection, model)
        ranked_heads = _heads_from_ranking(ranking)
        ranking_frames.append(
            ranking.assign(comparison_mode=spec.label, source_run=run_dir.name)
        )

        arms: list[tuple[str, int, int, list[Head]]] = [("clean", 0, 0, [])]
        for top_k in args.top_k:
            top_k = int(top_k)
            arms.append(("ranked", 0, top_k, ranked_heads[:top_k]))
            for path_id, control in enumerate(
                _matched_control_sets(
                    ranked_heads,
                    top_k=top_k,
                    n_layer=cfg.n_layer,
                    n_head=cfg.n_head,
                ),
                start=1,
            ):
                arms.append(("layer_matched_control", path_id, top_k, control))

        for path_kind, path_id, top_k, heads in arms:
            label = _format_heads(heads)
            print(
                f"[{spec.label}] free-running {path_kind} K={top_k} {label}",
                flush=True,
            )
            detail = _free_running_rows(
                model,
                cfg,
                vocab,
                reporting,
                mode=spec.mode,
                heads=heads,
                batch_size=min(args.batch_size, len(reporting)),
            )
            for column, value in {
                "comparison_mode": spec.label,
                "path_kind": path_kind,
                "path_id": int(path_id),
                "top_k": int(top_k),
                "heads": label,
            }.items():
                detail.insert(0, column, value)
            detail_frames.append(detail)
            summary_rows.append(
                {
                    "comparison_mode": spec.label,
                    "scope": "role_query_local_free_running",
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
    pd.DataFrame(summary_rows).to_csv(
        args.output / "free_running_summary.csv", index=False
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "comparison": (
                    f"paired {args.expected_version or 'separator-trace'} "
                    "Thinking vs Non-thinking"
                    if args.paired_run_prefix is not None
                    else "v22 Thinking vs matched v20 Non-thinking"
                ),
                "paired_run_prefix": args.paired_run_prefix,
                "expected_version": args.expected_version,
                "selection_split": "heldout_head_selection",
                "reporting_split": "disjoint heldout reporting examples",
                "examples_per_count": int(args.examples_per_count),
                "generation": "greedy until EOS; no gold trace tokens after <Think>",
                "scope": {
                    "nonthinking": "selected broad heads zeroed at <Ans>",
                    "thinking": "selected targeted heads zeroed at every generated <Sep> query",
                },
                "controls": (
                    "all disjoint layer-count-matched sets available in the "
                    "configured head inventory"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
