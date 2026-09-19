#!/usr/bin/env python
"""Export v58's four aligned geometry endpoints at every residual layer.

This is a read-only analysis pass over the two final checkpoints.  PCA is fit
on discovery states and applied to disjoint confirmation states, matching the
clean NCC protocol.  The resulting compact CSV drives the report's v20-style
2x2, layer-selectable 2D/3D comparison.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from synthetic_counting_v20.aligned_geometry import (  # noqa: E402
    capture_mode_geometry,
    combine_splits,
    confirmation_pca_coordinates,
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_examples(run_dir: Path, *, device: str):
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    cfg = replace(cfg, device=device)
    if cfg.version != "v58" or cfg.count_min != 1 or cfg.count_max_threshold != 10:
        raise ValueError(
            f"expected v58 count 1..10, got {cfg.version} "
            f"count {cfg.count_min}..{cfg.count_max_threshold}"
        )
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
    training = list(curves["train"]["task"])
    all_heldout = list(curves["heldout"]["task"])
    reporting = []
    for count in range(cfg.count_min, cfg.count_max_threshold + 1):
        bucket = [item for item in all_heldout if int(item.count or 0) == count]
        reporting.extend(bucket[cfg.phase_head_selection_examples_per_count :])
    return cfg, vocab, training, reporting


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--discovery-per-label", type=int, default=10)
    parser.add_argument("--confirmation-per-label", type=int, default=8)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "analysis" / "v58_clean_ncc"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg, vocab, training, reporting = load_examples(run_dir, device=args.device)

    parts: list[pd.DataFrame] = []
    for mode in ("nonthinking", "thinking"):
        print(f"[{mode}] load final checkpoint", flush=True)
        _, loaded_vocab, _, _, model = load_v20_checkpoint_model(
            run_dir,
            "rope",
            mode,
            step=cfg.train_steps,
            device=args.device,
        )
        if loaded_vocab.fingerprint != vocab.fingerprint:
            raise ValueError(f"{mode}: checkpoint vocabulary mismatch")
        model.eval()
        discovery = capture_mode_geometry(
            model,
            vocab,
            training,
            mode=mode,
            split="discovery",
            per_label=args.discovery_per_label,
            device=args.device,
            batch_size=args.batch_size,
        )
        confirmation = capture_mode_geometry(
            model,
            vocab,
            reporting,
            mode=mode,
            split="confirmation",
            per_label=args.confirmation_per_label,
            device=args.device,
            batch_size=args.batch_size,
        )
        for endpoint in discovery:
            dataset = combine_splits(discovery[endpoint], confirmation[endpoint])
            for layer in sorted(dataset.states_by_layer):
                frame = confirmation_pca_coordinates(dataset, layer, components=3)
                frame.insert(0, "comparison_mode", mode)
                frame["endpoint"] = endpoint
                frame["layer"] = int(layer)
                frame["sample"] = range(len(frame))
                frame["k"] = frame["occurrence"].astype(int)
                parts.append(
                    frame[
                        [
                            "comparison_mode",
                            "endpoint",
                            "layer",
                            "sample",
                            "k",
                            "pc1",
                            "pc2",
                            "pc3",
                            "pc1_variance_ratio",
                            "pc2_variance_ratio",
                            "pc3_variance_ratio",
                            "total_count",
                            "position",
                            "prompt_sha256",
                        ]
                    ]
                )
        del model, discovery, confirmation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    cloud = pd.concat(parts, ignore_index=True)
    expected_endpoints = {
        "nonthinking_prompt_occurrence",
        "thinking_item_end",
        "nonthinking_answer_query",
        "thinking_answer_query",
    }
    if set(cloud["endpoint"]) != expected_endpoints:
        raise RuntimeError(f"endpoint mismatch: {sorted(set(cloud['endpoint']))}")
    counts = cloud.groupby(["endpoint", "layer"]).size()
    expected_rows = 10 * args.confirmation_per_label
    if len(counts) != 20 or not counts.eq(expected_rows).all():
        raise RuntimeError(f"unbalanced endpoint/layer export:\n{counts}")

    output_path = output_dir / "geometry_confirmation_pca_all_layers.csv"
    cloud.to_csv(output_path, index=False)
    manifest = {
        "schema_version": "v58_geometry_cloud_v1",
        "source_run": str(run_dir),
        "checkpoint_step": int(cfg.train_steps),
        "pca_fit_split": "discovery",
        "projection_split": "confirmation",
        "discovery_per_label": args.discovery_per_label,
        "confirmation_per_label": args.confirmation_per_label,
        "layers": sorted(int(value) for value in cloud["layer"].unique()),
        "endpoints": sorted(expected_endpoints),
        "rows": int(len(cloud)),
        "output_sha256": sha256(output_path),
    }
    (output_dir / "geometry_confirmation_pca_all_layers_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(cloud)} rows)", flush=True)


if __name__ == "__main__":
    main()
