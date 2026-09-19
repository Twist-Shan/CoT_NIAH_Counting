#!/usr/bin/env python
"""Aligned clean geometry for v22 Thinking vs its matched Non-thinking baseline.

The v22 run intentionally trained only the separator-trace Thinking model.  Its
registered Non-thinking comparator is the RoPE/atomic-answer model from v20.
The two runs share the exact train and held-out task examples; their vocabulary
fingerprints differ only because v22 removes numeric trace-index tokens.

Layer selection uses discovery examples only.  The selected decoder, PCA, and
class centroids are then frozen and evaluated on the disjoint confirmation set.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = (
    Path(r"\\?\H:/")
    / "\u6211\u7684\u4e91\u7aef\u786c\u76d8"
    / "Colab_Notebooks"
    / "CoT_Counting"
    / "Synthetic_CoT_NiaH_Count"
    / "colab_results"
)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from synthetic_counting_v20.aligned_geometry import (  # noqa: E402
    capture_mode_geometry,
    combine_splits,
    evaluate_geometry_dataset,
)
from synthetic_counting_v20.config import config_from_dict  # noqa: E402
from synthetic_counting_v20.data import (  # noqa: E402
    V20Vocab,
    load_corpus_split,
    load_corpus_text,
    load_suite_manifests,
)
from synthetic_counting_v20.needle_pool import load_needle_pool  # noqa: E402
from synthetic_counting_v20.training import load_v20_checkpoint_model  # noqa: E402


@dataclass(frozen=True)
class ModeSpec:
    label: str
    run_prefix: str
    mode: str
    expected_trace_format: str


SPECS = (
    ModeSpec("nonthinking", "v20_main_RoPE_count1-30_seed1234", "nonthinking", "indexed"),
    ModeSpec("thinking", "v22_main_", "thinking", "separator"),
)


def _unique_run(results_root: Path, prefix: str) -> Path:
    matches = sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and (path.name == prefix or path.name.startswith(prefix))
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one run matching {prefix!r}; found {[path.name for path in matches]}"
        )
    return matches[0]


def _load_bundle(run_dir: Path, *, device: str):
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    cfg = replace(cfg, device=device)
    vocab = V20Vocab.load(run_dir / "vocab.json")
    corpus = load_corpus_text()
    split = load_corpus_split(run_dir / "data" / "corpus_split.json", cfg, corpus)
    pool = load_needle_pool(
        run_dir / "data" / "needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    curves, _ = load_suite_manifests(
        run_dir / "data" / "loss_suite_manifests.json",
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    all_heldout = list(curves["heldout"]["task"])
    selection = []
    reporting = []
    for count in range(cfg.count_min, cfg.count_max_threshold + 1):
        bucket = [example for example in all_heldout if int(example.count or 0) == count]
        selection.extend(bucket[: cfg.phase_head_selection_examples_per_count])
        reporting.extend(bucket[cfg.phase_head_selection_examples_per_count :])
    return cfg, vocab, list(curves["train"]["task"]), selection, reporting


def _example_payload(examples) -> list[dict[str, object]]:
    return [asdict(example) for example in examples]


def analyze_mode(
    run_dir: Path,
    spec: ModeSpec,
    *,
    device: str,
    discovery_per_label: int,
    confirmation_per_label: int,
    batch_size: int,
):
    cfg, vocab, train_examples, _selection_examples, reporting_examples = _load_bundle(
        run_dir, device=device
    )
    if cfg.position_encodings != ("rope",) or cfg.count_tokenization != "atomic":
        raise ValueError(f"{spec.label}: expected RoPE with atomic count tokens")
    if cfg.trace_format != spec.expected_trace_format:
        raise ValueError(
            f"{spec.label}: expected trace_format={spec.expected_trace_format}, got {cfg.trace_format}"
        )
    _, loaded_vocab, _, _, model = load_v20_checkpoint_model(
        run_dir,
        "rope",
        spec.mode,
        step=cfg.train_steps,
        device=device,
    )
    if loaded_vocab.fingerprint != vocab.fingerprint:
        raise ValueError(f"{spec.label}: checkpoint vocabulary mismatch")
    model.eval()
    print(f"[{spec.label}] discovery capture", flush=True)
    discovery = capture_mode_geometry(
        model,
        vocab,
        train_examples,
        mode=spec.mode,
        split="discovery",
        per_label=discovery_per_label,
        device=device,
        batch_size=batch_size,
    )
    print(f"[{spec.label}] confirmation capture", flush=True)
    confirmation = capture_mode_geometry(
        model,
        vocab,
        reporting_examples,
        mode=spec.mode,
        split="confirmation",
        per_label=confirmation_per_label,
        device=device,
        batch_size=batch_size,
    )

    metric_frames = []
    selection_frames = []
    selected_layers = {}
    for endpoint in discovery:
        metrics, selections, selected_layer = evaluate_geometry_dataset(
            combine_splits(discovery[endpoint], confirmation[endpoint]),
            endpoint=endpoint,
        )
        for frame in (metrics, selections):
            frame.insert(0, "comparison_mode", spec.label)
            frame.insert(1, "source_run", run_dir.name)
        metric_frames.append(metrics)
        selection_frames.append(selections)
        selected_layers[endpoint] = int(selected_layer)
        chosen = metrics.loc[metrics["layer"].astype(int).eq(int(selected_layer))].iloc[0]
        print(
            f"[{spec.label}] {endpoint}: L{selected_layer}, "
            f"confirmation Logistic={chosen['confirmation_logistic_balanced_accuracy']:.4f}, "
            f"NCC={chosen['confirmation_ncc_balanced_accuracy']:.4f}",
            flush=True,
        )

    del model, discovery, confirmation
    gc.collect()
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(selection_frames, ignore_index=True),
        selected_layers,
        train_examples,
        reporting_examples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "work" / "ncc_v22_thinking_vs_nonthinking"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--discovery-per-label", type=int, default=10)
    parser.add_argument("--confirmation-per-label", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_selections = []
    selected = {}
    example_payloads = {}
    for spec in SPECS:
        run_dir = _unique_run(args.results_root.resolve(), spec.run_prefix)
        metrics, selections, layers, train_examples, reporting_examples = analyze_mode(
            run_dir,
            spec,
            device=args.device,
            discovery_per_label=args.discovery_per_label,
            confirmation_per_label=args.confirmation_per_label,
            batch_size=args.batch_size,
        )
        all_metrics.append(metrics)
        all_selections.append(selections)
        selected[spec.label] = layers
        example_payloads[spec.label] = (
            _example_payload(train_examples),
            _example_payload(reporting_examples),
        )

    if example_payloads["nonthinking"] != example_payloads["thinking"]:
        raise RuntimeError("v20 Non-thinking and v22 Thinking do not share exact task examples")

    pd.concat(all_metrics, ignore_index=True).to_csv(
        args.output / "geometry_site_layer_metrics.csv", index=False
    )
    pd.concat(all_selections, ignore_index=True).to_csv(
        args.output / "geometry_discovery_selected_metrics.csv", index=False
    )
    (args.output / "selected_layers.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
