#!/usr/bin/env python
"""Measure how frozen count geometry changes after cumulative head ablation.

The clean discovery split selects each endpoint's layer and fits its decoder.
Every ablated confirmation arm reuses that clean discovery fit.  This makes the
reported NCC change a representation-mediation readout rather than a newly
refitted post-ablation decoder.

Two intervention scopes are retained:

* ``global`` removes each selected head at every sequence position, matching
  standard cumulative head-mask experiments;
* ``role_query_local`` removes Non-thinking broad heads only at ``<Ans>`` and
  Thinking targeted-retrieval heads only at trace separator queries.

For K=1/2, all possible disjoint layer-count-matched control sets are evaluated
when the tiny 4x4 head inventory permits them.  K=4 is reported as a ranked
exploratory dose when no disjoint matched set exists.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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

from compare_v22_modes_ncc import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    SPECS,
    ModeSpec,
    _load_bundle,
    _unique_run,
)
from synthetic_counting_v20.aligned_geometry import (  # noqa: E402
    capture_mode_geometry,
    combine_splits,
    evaluate_geometry_dataset,
)
from synthetic_counting_v20.data import collate_v20, render_v20  # noqa: E402
from synthetic_counting_v20.extended_analysis import _broad_metric_matrices  # noqa: E402
from synthetic_counting_v20.training import load_v20_checkpoint_model  # noqa: E402
from synthetic_counting_v20.v10_port_analysis import _local_attention_edit  # noqa: E402


Head = tuple[int, int]  # one-based layer, zero-based head


def _format_heads(heads: Sequence[Head]) -> str:
    return ";".join(f"L{layer}H{head}" for layer, head in heads)


def _load_model(run_dir: Path, spec: ModeSpec, *, device: str):
    cfg, vocab, train, selection, reporting = _load_bundle(run_dir, device=device)
    _, checkpoint_vocab, _, _, model = load_v20_checkpoint_model(
        run_dir,
        "rope",
        spec.mode,
        step=cfg.train_steps,
        device=device,
    )
    if checkpoint_vocab.fingerprint != vocab.fingerprint:
        raise ValueError(f"{spec.label}: checkpoint vocabulary mismatch")
    return cfg, vocab, train, selection, reporting, model.eval()


def _rank_heads(
    run_dir: Path,
    spec: ModeSpec,
    cfg,
    vocab,
    selection_examples,
    model,
) -> pd.DataFrame:
    if spec.mode == "thinking":
        ranking = pd.read_csv(
            run_dir
            / "analysis"
            / "phase_transition"
            / "tables"
            / "fixed_head_rankings.csv"
        )
        result = ranking.loc[ranking["role"].eq("targeted_retrieval")].copy()
        result = result.sort_values("rank", kind="mergesort")
        result["selection_metric"] = "correct_prompt_needle_mass"
        return result[
            [
                "role",
                "rank",
                "layer",
                "head",
                "selection_metric",
                "selection_score",
                "selection_split",
            ]
        ].reset_index(drop=True)

    items = [render_v20(example, vocab, "nonthinking") for example in selection_examples]
    matrices, observations = _broad_metric_matrices(model, cfg, vocab, items)
    score = matrices["broad_score"]
    rows = []
    for layer in range(cfg.n_layer):
        for head in range(cfg.n_head):
            rows.append(
                {
                    "role": "nonthinking_broad",
                    "layer": layer + 1,
                    "head": head,
                    "selection_metric": "broad_score",
                    "selection_score": float(score[layer, head]),
                    "selection_split": "heldout_head_selection",
                    "selection_observations": int(observations),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["selection_score", "layer", "head"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    result.insert(1, "rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


def _heads_from_ranking(ranking: pd.DataFrame) -> list[Head]:
    return [
        (int(row.layer), int(row.head))
        for row in ranking.sort_values("rank", kind="mergesort").itertuples(index=False)
    ]


def _matched_control_sets(
    ranked: Sequence[Head],
    *,
    top_k: int,
    n_layer: int,
    n_head: int,
) -> list[list[Head]]:
    selected = set(ranked[:top_k])
    required: dict[int, int] = {}
    for layer, _head in ranked[:top_k]:
        required[layer] = required.get(layer, 0) + 1
    per_layer_choices: list[list[tuple[Head, ...]]] = []
    for layer in sorted(required):
        candidates = [
            (layer, head)
            for head in range(n_head)
            if (layer, head) not in selected
        ]
        count = required[layer]
        if len(candidates) < count:
            return []
        per_layer_choices.append(list(itertools.combinations(candidates, count)))
    controls = []
    for product in itertools.product(*per_layer_choices):
        controls.append([head for group in product for head in group])
    return controls


def _global_mask(cfg, heads: Sequence[Head], *, device: str) -> torch.Tensor:
    mask = torch.ones((cfg.n_layer, cfg.n_head), device=device)
    for layer, head in heads:
        mask[layer - 1, head] = 0
    return mask


def _local_context_factory(model, mode: str, heads: Sequence[Head]):
    frozen_heads = tuple(heads)

    def factory(items):
        positions = []
        for item in items:
            assert item.spans is not None
            positions.append(
                [int(item.spans.ans_pos)]
                if mode == "nonthinking"
                else [int(position) for position in item.spans.trace_query_positions]
            )
        return _local_attention_edit(model, frozen_heads, positions)

    return factory


@torch.no_grad()
def _behavior_metrics(
    model,
    cfg,
    vocab,
    examples,
    *,
    mode: str,
    device: str,
    batch_size: int,
    head_mask: torch.Tensor | None = None,
    local_heads: Sequence[Head] | None = None,
) -> dict[str, float]:
    answer_correct: list[float] = []
    answer_margin: list[float] = []
    answer_abs_error: list[float] = []
    trace_correct: list[float] = []
    trace_margin: list[float] = []
    supported_counts = tuple(
        range(int(cfg.count_min), int(cfg.count_max_threshold) + 1)
    )
    number_ids = torch.tensor(
        [vocab.token_to_id[vocab.number_token(count)] for count in supported_counts],
        device=device,
    )
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        items = [render_v20(example, vocab, mode) for example in batch_examples]
        ids, _, attention_mask = collate_v20(items, vocab, device)
        context = (
            _local_context_factory(model, mode, local_heads)(items)
            if local_heads
            else torch.no_grad()
        )
        with context:
            output = model(
                input_ids=ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )
        for row, (example, item) in enumerate(zip(batch_examples, items, strict=True)):
            assert item.spans is not None and example.count is not None
            logits = output.logits[row, item.spans.ans_pos].float()
            gold_id = vocab.token_to_id[vocab.number_token(int(example.count))]
            alternatives = torch.cat((logits[:gold_id], logits[gold_id + 1 :]))
            answer_correct.append(float(int(logits.argmax()) == gold_id))
            answer_margin.append(float((logits[gold_id] - alternatives.max()).cpu()))
            probabilities = torch.softmax(logits[number_ids], dim=0)
            expected = float(
                (
                    probabilities
                    * torch.tensor(
                        supported_counts,
                        dtype=probabilities.dtype,
                        device=device,
                    )
                )
                .sum()
                .cpu()
            )
            answer_abs_error.append(abs(expected - int(example.count)))
            if mode == "thinking":
                for query, target_position in zip(
                    item.spans.trace_query_positions,
                    item.spans.trace_marker_positions,
                    strict=True,
                ):
                    target_id = int(item.input_ids[target_position])
                    values = output.logits[row, query].float()
                    alternatives = torch.cat(
                        (values[:target_id], values[target_id + 1 :])
                    )
                    trace_correct.append(float(int(values.argmax()) == target_id))
                    trace_margin.append(
                        float((values[target_id] - alternatives.max()).cpu())
                    )
    return {
        "teacher_forced_final_accuracy": float(np.mean(answer_correct)),
        "teacher_forced_final_margin": float(np.mean(answer_margin)),
        "teacher_forced_final_expected_abs_error": float(np.mean(answer_abs_error)),
        "teacher_forced_trace_accuracy": (
            float(np.mean(trace_correct)) if trace_correct else math.nan
        ),
        "teacher_forced_trace_margin": (
            float(np.mean(trace_margin)) if trace_margin else math.nan
        ),
    }


def _evaluate_condition(
    *,
    spec: ModeSpec,
    cfg,
    vocab,
    model,
    discovery,
    reporting_examples,
    heads: Sequence[Head],
    scope: str,
    path_kind: str,
    path_id: int,
    top_k: int,
    device: str,
    confirmation_per_label: int,
    batch_size: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    head_mask = None
    local_factory = None
    local_heads = None
    if scope == "global":
        head_mask = _global_mask(cfg, heads, device=device)
    elif scope == "role_query_local":
        local_factory = _local_context_factory(model, spec.mode, heads)
        local_heads = heads
    else:
        raise ValueError(scope)

    confirmation = capture_mode_geometry(
        model,
        vocab,
        reporting_examples,
        mode=spec.mode,
        split="confirmation",
        per_label=confirmation_per_label,
        device=device,
        batch_size=batch_size,
        head_mask=head_mask,
        forward_context_factory=local_factory,
    )
    layer_frames = []
    selected_rows = []
    for endpoint in discovery:
        metrics, _selections, selected_layer = evaluate_geometry_dataset(
            combine_splits(discovery[endpoint], confirmation[endpoint]),
            endpoint=endpoint,
            classes=tuple(
                range(int(cfg.count_min), int(cfg.count_max_threshold) + 1)
            ),
        )
        metrics.insert(0, "comparison_mode", spec.label)
        metrics.insert(1, "scope", scope)
        metrics.insert(2, "path_kind", path_kind)
        metrics.insert(3, "path_id", int(path_id))
        metrics.insert(4, "top_k", int(top_k))
        metrics.insert(5, "heads", _format_heads(heads))
        layer_frames.append(metrics)
        chosen = metrics.loc[metrics["layer"].astype(int).eq(int(selected_layer))].iloc[0]
        selected_rows.append(
            {
                "comparison_mode": spec.label,
                "endpoint": endpoint,
                "scope": scope,
                "path_kind": path_kind,
                "path_id": int(path_id),
                "top_k": int(top_k),
                "heads": _format_heads(heads),
                "clean_discovery_selected_layer": int(selected_layer),
                "confirmation_logistic_balanced_accuracy": float(
                    chosen["confirmation_logistic_balanced_accuracy"]
                ),
                "confirmation_ncc_balanced_accuracy": float(
                    chosen["confirmation_ncc_balanced_accuracy"]
                ),
                "confirmation_isotropic_snr_db": float(
                    chosen["confirmation_isotropic_snr_db"]
                ),
            }
        )
    behavior = _behavior_metrics(
        model,
        cfg,
        vocab,
        reporting_examples,
        mode=spec.mode,
        device=device,
        batch_size=batch_size,
        head_mask=head_mask,
        local_heads=local_heads,
    )
    return pd.concat(layer_frames, ignore_index=True), selected_rows, behavior


def analyze_mode(
    run_dir: Path,
    spec: ModeSpec,
    *,
    device: str,
    discovery_per_label: int,
    confirmation_per_label: int,
    batch_size: int,
    top_ks: Sequence[int],
    expected_version: str | None = None,
):
    cfg, vocab, train, selection, reporting, model = _load_model(
        run_dir, spec, device=device
    )
    if expected_version is not None and cfg.version != expected_version:
        raise ValueError(
            f"{spec.label}: expected version={expected_version!r}, got {cfg.version!r}"
        )
    ranking = _rank_heads(run_dir, spec, cfg, vocab, selection, model)
    ranked_heads = _heads_from_ranking(ranking)
    print(
        f"[{spec.label}] top heads: {_format_heads(ranked_heads[:4])}", flush=True
    )
    discovery = capture_mode_geometry(
        model,
        vocab,
        train,
        mode=spec.mode,
        split="discovery",
        per_label=discovery_per_label,
        device=device,
        batch_size=batch_size,
    )

    layer_frames = []
    selected_rows: list[dict[str, Any]] = []
    behavior_rows = []
    arms: list[tuple[str, int, int, list[Head]]] = [("clean", 0, 0, [])]
    for top_k in top_ks:
        arms.append(("ranked", 0, int(top_k), ranked_heads[:top_k]))
        controls = _matched_control_sets(
            ranked_heads,
            top_k=int(top_k),
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
        )
        for path_id, control in enumerate(controls, start=1):
            arms.append(("layer_matched_control", path_id, int(top_k), control))
    for scope in ("global", "role_query_local"):
        for path_kind, path_id, top_k, heads in arms:
            print(
                f"[{spec.label}] {scope} {path_kind} K={top_k} "
                f"{_format_heads(heads)}",
                flush=True,
            )
            layers, selected, behavior = _evaluate_condition(
                spec=spec,
                cfg=cfg,
                vocab=vocab,
                model=model,
                discovery=discovery,
                reporting_examples=reporting,
                heads=heads,
                scope=scope,
                path_kind=path_kind,
                path_id=path_id,
                top_k=top_k,
                device=device,
                confirmation_per_label=confirmation_per_label,
                batch_size=batch_size,
            )
            layer_frames.append(layers)
            selected_rows.extend(selected)
            behavior_rows.append(
                {
                    "comparison_mode": spec.label,
                    "scope": scope,
                    "path_kind": path_kind,
                    "path_id": path_id,
                    "top_k": top_k,
                    "heads": _format_heads(heads),
                    **behavior,
                }
            )
    return (
        ranking.assign(comparison_mode=spec.label, source_run=run_dir.name),
        pd.concat(layer_frames, ignore_index=True),
        pd.DataFrame(selected_rows),
        pd.DataFrame(behavior_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "work" / "v22_topk_ncc"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--discovery-per-label", type=int, default=10)
    parser.add_argument("--confirmation-per-label", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument(
        "--paired-run-prefix",
        default=None,
        help=(
            "Use both independently trained modes from one separator-trace run; "
            "the default retains the historical v20/v22 comparison"
        ),
    )
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

    rankings = []
    layerwise = []
    selected = []
    behavior = []
    for spec in specs:
        run_dir = _unique_run(args.results_root.resolve(), spec.run_prefix)
        outputs = analyze_mode(
            run_dir,
            spec,
            device=args.device,
            discovery_per_label=args.discovery_per_label,
            confirmation_per_label=args.confirmation_per_label,
            batch_size=args.batch_size,
            top_ks=args.top_k,
            expected_version=args.expected_version,
        )
        rankings.append(outputs[0])
        layerwise.append(outputs[1])
        selected.append(outputs[2])
        behavior.append(outputs[3])

    pd.concat(rankings, ignore_index=True).to_csv(
        args.output / "head_rankings.csv", index=False
    )
    pd.concat(layerwise, ignore_index=True).to_csv(
        args.output / "post_ablation_ncc_layerwise.csv", index=False
    )
    selected_frame = pd.concat(selected, ignore_index=True)
    selected_frame.to_csv(args.output / "post_ablation_ncc_selected.csv", index=False)
    behavior_frame = pd.concat(behavior, ignore_index=True)
    behavior_frame.to_csv(args.output / "post_ablation_behavior.csv", index=False)
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
                "top_k": list(map(int, args.top_k)),
                "decoder_policy": (
                    "clean discovery selects layer and fits PCA/decoder/centroids; "
                    "ablated confirmation is evaluated without refitting"
                ),
                "scopes": {
                    "global": "selected heads removed at every query position",
                    "role_query_local": (
                        "Non-thinking at answer query; Thinking at every trace separator query"
                    ),
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
