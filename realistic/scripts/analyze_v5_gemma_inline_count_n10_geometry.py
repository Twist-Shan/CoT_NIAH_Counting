#!/usr/bin/env python3
"""Analyze paired Gemma N=10 geometry for one frozen inline-count suffix family."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_v5_pure_trace_n10_geometry import (  # noqa: E402
    choose_layer,
    concatenate_datasets,
    frozen_metrics,
    geometry_payload,
    paired_datasets,
    sha256,
)
from select_v5_gemma_inline_count_n10 import (  # noqa: E402
    GRAMMAR,
    MARKER,
    MODEL,
    TARGET_N,
    select,
)
from realistic_niah_v5.causal_aligned_geometry import (  # noqa: E402
    load_causal_aligned_native_capture,
)
from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    load_non_thinking_capture,
)


SCHEMA = "realistic_niah_v5_gemma_inline_count_n10_geometry_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-thinking-root",
        type=Path,
        default=ROOT / "work/nonthinking_v44_geometry_300_150_136_166_78",
    )
    parser.add_argument(
        "--native-running-root",
        type=Path,
        default=ROOT / "work/v5_geometry_full_panel/running",
    )
    parser.add_argument(
        "--event-registry",
        type=Path,
        default=ROOT / "reports/v5_native_causal_site_review/event_registry.csv",
    )
    parser.add_argument(
        "--supplement-root", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--supplement-event-registry", type=Path, action="append", required=True
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--raw-generation-jsonl", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--family",
        choices=("count_colon", "count_equals", "controlled_prefix_record"),
        required=True,
    )
    parser.add_argument(
        "--native-site-kind",
        default="item_end",
        help="Native endpoint to analyze; use pre_marker to avoid reading Count: k.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    combined_supplement_registry: Path | None = None
    if len(args.supplement_root) == 1 and len(args.supplement_event_registry) > 1:
        combined_supplement_registry = args.output / "combined_supplement_event_registry.csv"
        pd.concat(
            [pd.read_csv(path) for path in args.supplement_event_registry],
            ignore_index=True,
        ).drop_duplicates().to_csv(combined_supplement_registry, index=False)
        supplement_pairs = [
            (args.supplement_root[0], combined_supplement_registry)
        ]
    elif len(args.supplement_root) == len(args.supplement_event_registry):
        supplement_pairs = list(
            zip(args.supplement_root, args.supplement_event_registry)
        )
    else:
        raise ValueError(
            "Supplement roots and event registries must be paired, or one capture "
            "root may be accompanied by multiple registries to concatenate"
        )
    registry = pd.concat(
        [pd.read_csv(args.event_registry)]
        + [pd.read_csv(path) for path in args.supplement_event_registry],
        ignore_index=True,
    )
    selected = (
        pd.read_csv(args.selection) if args.selection is not None else select(registry, args.family)
    )
    if args.family == "controlled_prefix_record" and args.selection is None:
        raise ValueError("controlled_prefix_record requires a frozen --selection")
    grammar = (
        "same_unit_rank_before_city"
        if args.family == "controlled_prefix_record"
        else GRAMMAR
    )
    selected_ids = set(selected["request_id"].astype(str))
    generation_filter_audit = []
    for path in args.raw_generation_jsonl:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        request_ids = {str(row["request_id"]) for row in rows}
        generation_filter_audit.append(
            {
                "path": str(path.resolve()),
                "raw_trajectories": len(rows),
                "retained_trajectories": len(request_ids & selected_ids),
            }
        )
    discovery_n = int(selected["split"].astype(str).eq("discovery").sum())
    confirmation_n = int(selected["split"].astype(str).eq("confirmation").sum())

    base_native_index = args.native_running_root / MODEL / "capture_index.jsonl"
    base_non_index = (
        args.non_thinking_root
        / MODEL
        / "numeric/representation/capture/capture_index.jsonl"
    )
    if args.native_site_kind == "item_end":
        base_native, base_audit = load_causal_aligned_native_capture(
            base_native_index, args.event_registry, site_kind=args.native_site_kind
        )
    else:
        # The original 300-panel capture predates pre_marker. Selected Gemma
        # seeds all come from supplements, so do not pretend that site exists
        # in the base archive.
        base_native = None
        base_audit = {
            "skipped": True,
            "reason": f"base capture lacks {args.native_site_kind}",
        }
    base_non = load_non_thinking_capture(
        base_non_index, design_variant="v4.4", pooling="span_end"
    )
    supplement_native = []
    supplement_non = []
    supplement_audits = []
    supplement_native_indices = []
    supplement_non_indices = []
    for root, event_registry in supplement_pairs:
        native_index = root / MODEL / "native/capture_index.jsonl"
        non_index = root / MODEL / "non_thinking/capture_index.jsonl"
        native_dataset, native_audit = load_causal_aligned_native_capture(
            native_index, event_registry, site_kind=args.native_site_kind
        )
        non_dataset = load_non_thinking_capture(
            non_index, design_variant="v4.4", pooling="span_end"
        )
        supplement_native.append(native_dataset)
        supplement_non.append(non_dataset)
        supplement_audits.append(native_audit)
        supplement_native_indices.append(native_index)
        supplement_non_indices.append(non_index)
    native = concatenate_datasets(
        *([base_native] if base_native is not None else []), *supplement_native
    )
    non = concatenate_datasets(base_non, *supplement_non)
    split_by_seed = {
        int(row.seed): str(row.split)
        for row in selected[["seed", "split"]].itertuples(index=False)
    }
    for dataset in (native, non):
        dataset.metadata["source_split"] = dataset.metadata["split"].astype(str)
        mapped = dataset.metadata["seed"].astype(int).map(split_by_seed)
        eligible = mapped.notna()
        dataset.metadata.loc[eligible, "split"] = mapped.loc[eligible]
    native, non = paired_datasets(native, non, selected, grammar)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "target_gold_count": TARGET_N,
        "filter_unit": "whole trajectory",
        "surface_family": args.family,
        "native_site_kind": args.native_site_kind,
        "generation_filter_audit": generation_filter_audit,
        "selection_rule": (
            f"ten {grammar} items in {args.family}; "
            f"suffix value equals occurrence; native site is {args.native_site_kind}"
        ),
        "models": {
            MODEL: {
                "grammar_class": grammar,
                "marker_kind": MARKER,
                "selected_trace_ids": selected["request_id"].astype(str).tolist(),
                "native_loader_audit": {
                    "base": base_audit,
                    "supplements": supplement_audits,
                },
            }
        },
    }
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    model_payload = payload["models"][MODEL]
    for mode, dataset in (("non_thinking", non), ("native_thinking", native)):
        layer, discovery, candidates = choose_layer(
            dataset, pca_dim=args.pca_dim, folds=args.folds, seed=args.seed
        )
        metrics = frozen_metrics(
            dataset, layer, discovery, pca_dim=args.pca_dim, seed=args.seed
        )
        geometry = geometry_payload(dataset, layer, metrics, seed=args.seed)
        candidate_by_layer = {
            int(candidate["layer"]): candidate for candidate in candidates
        }
        layers: dict[str, Any] = {}
        for candidate_layer in sorted(dataset.states_by_layer):
            if int(candidate_layer) == int(layer):
                layer_metrics = metrics
                layer_geometry = geometry
            else:
                layer_metrics = frozen_metrics(
                    dataset,
                    int(candidate_layer),
                    candidate_by_layer[int(candidate_layer)],
                    pca_dim=args.pca_dim,
                    seed=args.seed,
                )
                layer_geometry = geometry_payload(
                    dataset, int(candidate_layer), layer_metrics, seed=args.seed
                )
            layers[str(int(candidate_layer))] = layer_geometry
        model_payload[mode] = {
            **geometry,
            "selected_layer": int(layer),
            "layers": layers,
        }
        metric_rows.append(
            {
                "model_label": MODEL,
                "mode": mode,
                "grammar_class": grammar,
                "marker_kind": MARKER,
                "surface_family": args.family,
                "site_kind": (
                    "span_end" if mode == "non_thinking" else args.native_site_kind
                ),
                "gold_count": TARGET_N,
                "layer": int(layer),
                "states": int(len(dataset.metadata)),
                "trajectories": int(dataset.metadata["request_id"].nunique()),
                "discovery_trajectories": discovery_n,
                "confirmation_trajectories": confirmation_n,
                **metrics,
                "confirmation_pca3_class_balanced_silhouette": geometry[
                    "confirmation_pca3_class_balanced_silhouette"
                ],
                "confirmation_pca3_ordinal_rsa": geometry[
                    "confirmation_pca3_ordinal_rsa"
                ],
            }
        )
        candidate_rows.extend(
            {
                "model_label": MODEL,
                "mode": mode,
                "surface_family": args.family,
                **row,
            }
            for row in candidates
        )

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "selection": args.output / "selected_trajectories.csv",
        "metrics": args.output / "paired_metrics.csv",
        "candidates": args.output / "layer_candidates.csv",
        "payload": args.output / "geometry_payload.json",
    }
    if combined_supplement_registry is not None:
        paths["combined_supplement_registry"] = combined_supplement_registry
    selected.to_csv(paths["selection"], index=False)
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(candidate_rows).to_csv(paths["candidates"], index=False)
    paths["payload"].write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    inputs = [args.event_registry, base_non_index]
    if base_native is not None:
        inputs.append(base_native_index)
    inputs += list(args.supplement_event_registry) + supplement_native_indices + supplement_non_indices
    if args.selection is not None:
        inputs.append(args.selection)
    inputs.extend(args.raw_generation_jsonl)
    audit = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "surface_family": args.family,
        "native_site_kind": args.native_site_kind,
        "selection_independent_of_hidden_states": True,
        "discovery_trajectories": discovery_n,
        "confirmation_trajectories": confirmation_n,
        "layer_selection": (
            "each mode independently maximizes grouped discovery OOF mean "
            "Logistic/NCC balanced accuracy; confirmation is frozen"
        ),
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "outputs": {str(path.resolve()): sha256(path) for path in paths.values()},
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    non_metrics = model_payload["non_thinking"]["metrics"]
    native_metrics = model_payload["native_thinking"]["metrics"]
    print(
        f"{MODEL} {args.family} D/C={discovery_n}/{confirmation_n} "
        f"non L{model_payload['non_thinking']['layer']}="
        f"{non_metrics['confirmation_logistic_balanced_accuracy']:.3f}/"
        f"{non_metrics['confirmation_ncc_balanced_accuracy']:.3f} "
        f"native L{model_payload['native_thinking']['layer']}="
        f"{native_metrics['confirmation_logistic_balanced_accuracy']:.3f}/"
        f"{native_metrics['confirmation_ncc_balanced_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
