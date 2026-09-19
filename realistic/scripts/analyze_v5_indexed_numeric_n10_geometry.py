#!/usr/bin/env python3
"""Analyze paired N=10 geometry for Qwen's exact numeric-list trace template."""

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
from select_v5_indexed_numeric_n10 import (  # noqa: E402
    GRAMMAR,
    MARKER,
    MODEL,
    STRICT_DASH_SPLIT_SALT,
    TARGET_N,
    select,
    select_strict_dash_20_10,
)
from realistic_niah_v5.causal_aligned_geometry import (  # noqa: E402
    load_causal_aligned_native_capture,
)
from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    load_non_thinking_capture,
)


SCHEMA = "realistic_niah_v5_indexed_numeric_n10_cross_mode_geometry_v1"
STRICT_DASH_SCHEMA = (
    "realistic_niah_v5_indexed_numeric_n10_strict_dash_20_10_geometry_v1"
)


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
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--supplement-event-registry", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/v5_indexed_numeric_n10_cross_mode_geometry",
    )
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strict-dash-20-10",
        action="store_true",
        help=(
            "retain only exact k. city - score items ending on a score digit and "
            "apply the frozen secondary 20-discovery/10-confirmation split"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = pd.concat(
        [
            pd.read_csv(args.event_registry),
            pd.read_csv(args.supplement_event_registry),
        ],
        ignore_index=True,
    )
    selected = (
        select_strict_dash_20_10(registry)
        if args.strict_dash_20_10
        else select(registry)
    )
    schema = STRICT_DASH_SCHEMA if args.strict_dash_20_10 else SCHEMA
    rank_template = (
        "exact_numeric_1_to_10_strict_city_dash_score"
        if args.strict_dash_20_10
        else "exact_numeric_1_to_10"
    )
    support = selected.groupby("split")["request_id"].nunique()
    discovery_n = int(support.get("discovery", 0))
    confirmation_n = int(support.get("confirmation", 0))
    if discovery_n < 2 or confirmation_n < 1:
        raise ValueError(
            f"Indexed numeric panel is under-supported: D/C={discovery_n}/{confirmation_n}"
        )

    base_native_index = args.native_running_root / MODEL / "capture_index.jsonl"
    base_non_index = (
        args.non_thinking_root
        / MODEL
        / "numeric/representation/capture/capture_index.jsonl"
    )
    supplement_native_index = (
        args.supplement_root / MODEL / "native/capture_index.jsonl"
    )
    supplement_non_index = (
        args.supplement_root / MODEL / "non_thinking/capture_index.jsonl"
    )
    base_native, base_audit = load_causal_aligned_native_capture(
        base_native_index,
        args.event_registry,
        site_kind="item_end",
    )
    supplement_native, supplement_audit = load_causal_aligned_native_capture(
        supplement_native_index,
        args.supplement_event_registry,
        site_kind="item_end",
    )
    base_non = load_non_thinking_capture(
        base_non_index,
        design_variant="v4.4",
        pooling="span_end",
    )
    supplement_non = load_non_thinking_capture(
        supplement_non_index,
        design_variant="v4.4",
        pooling="span_end",
    )
    native = concatenate_datasets(base_native, supplement_native)
    non = concatenate_datasets(base_non, supplement_non)
    if args.strict_dash_20_10:
        split_by_seed = {
            int(row.seed): str(row.split)
            for row in selected[["seed", "split"]].itertuples(index=False)
        }
        for dataset in (native, non):
            dataset.metadata["source_split"] = dataset.metadata["split"].astype(str)
            mapped = dataset.metadata["seed"].astype(int).map(split_by_seed)
            eligible = mapped.notna()
            dataset.metadata.loc[eligible, "split"] = mapped.loc[eligible]
    native, non = paired_datasets(native, non, selected, GRAMMAR)

    payload: dict[str, Any] = {
        "schema_version": schema,
        "target_gold_count": TARGET_N,
        "filter_unit": "whole trajectory",
        "surface_template": (
            "exact k. city - score; item_end commit token is a score digit"
            if args.strict_dash_20_10
            else "rank_text only"
        ),
        "secondary_split": (
            {
                "discovery": 20,
                "confirmation": 10,
                "salt": STRICT_DASH_SPLIT_SALT,
                "rule": (
                    "retain all 11 source-discovery seeds; hash-rank the 19 "
                    "source-confirmation seeds, retain the first 10 as secondary "
                    "confirmation, and promote the remaining 9 to discovery"
                ),
            }
            if args.strict_dash_20_10
            else None
        ),
        "models": {
            MODEL: {
                "grammar_class": GRAMMAR,
                "marker_kind": MARKER,
                "rank_template": rank_template,
                "pairing_key": [
                    "secondary split" if args.strict_dash_20_10 else "split",
                    "seed",
                    "gold_count",
                    "occurrence",
                ],
                "selected_trace_ids": selected["request_id"].astype(str).tolist(),
                "native_loader_audit": {
                    "base": base_audit,
                    "supplement": supplement_audit,
                },
            }
        },
    }
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    model_payload = payload["models"][MODEL]
    for mode, dataset in (("non_thinking", non), ("native_thinking", native)):
        layer, discovery, candidates = choose_layer(
            dataset,
            pca_dim=args.pca_dim,
            folds=args.folds,
            seed=args.seed,
        )
        metrics = frozen_metrics(
            dataset,
            layer,
            discovery,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )
        geometry = geometry_payload(dataset, layer, metrics, seed=args.seed)
        candidate_by_layer = {
            int(candidate["layer"]): candidate for candidate in candidates
        }
        layer_payloads: dict[str, Any] = {}
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
                    dataset,
                    int(candidate_layer),
                    layer_metrics,
                    seed=args.seed,
                )
            layer_payloads[str(int(candidate_layer))] = layer_geometry
        model_payload[mode] = {
            **geometry,
            "selected_layer": int(layer),
            "layers": layer_payloads,
        }
        metric_rows.append(
            {
                "model_label": MODEL,
                "mode": mode,
                "grammar_class": GRAMMAR,
                "marker_kind": MARKER,
                "rank_template": rank_template,
                "gold_count": TARGET_N,
                "layer": int(layer),
                "states": int(len(dataset.metadata)),
                "trajectories": int(dataset.metadata["request_id"].nunique()),
                "discovery_trajectories": discovery_n,
                "confirmation_trajectories": confirmation_n,
                "confirmation_per_class": confirmation_n,
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
                "grammar_class": GRAMMAR,
                "rank_template": rank_template,
                **row,
            }
            for row in candidates
        )

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "selection": args.output
        / (
            "selected_strict_dash_20_10_trajectories.csv"
            if args.strict_dash_20_10
            else "selected_indexed_numeric_trajectories.csv"
        ),
        "metrics": args.output / "paired_metrics.csv",
        "candidates": args.output / "layer_candidates.csv",
        "payload": args.output / "geometry_payload.json",
    }
    selected.to_csv(paths["selection"], index=False)
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(candidate_rows).to_csv(paths["candidates"], index=False)
    paths["payload"].write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    input_paths = [
        args.event_registry,
        args.supplement_event_registry,
        base_native_index,
        base_non_index,
        supplement_native_index,
        supplement_non_index,
    ]
    audit = {
        "schema_version": schema,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Qwen N=10 exact one-to-one whole traces; grammar adjacent_rank_before_city; "
            "marker indexed; rank_text exactly equals 1,2,...,10; every item exactly "
            "matches k. city - score and item_end is a score digit"
            if args.strict_dash_20_10
            else "Qwen N=10 exact one-to-one whole traces; grammar "
            "adjacent_rank_before_city; marker indexed; rank_text exactly equals "
            "1,2,...,10"
        ),
        "selection_independent_of_hidden_states": not args.strict_dash_20_10,
        "exploratory_surface_rule_note": (
            "The strict surface family was introduced after diagnosing the prior PCA "
            "split. Within the 30 parser-eligible trajectories, the secondary 20/10 "
            "assignment is seed-hash-only and metric-independent."
            if args.strict_dash_20_10
            else "not applicable"
        ),
        "secondary_split_salt": (
            STRICT_DASH_SPLIT_SALT if args.strict_dash_20_10 else None
        ),
        "discovery_trajectories": discovery_n,
        "confirmation_trajectories": confirmation_n,
        "layer_selection": (
            "each mode independently maximizes grouped discovery OOF mean Logistic/NCC "
            "balanced accuracy; confirmation is frozen"
        ),
        "inputs": {str(path.resolve()): sha256(path) for path in input_paths},
        "outputs": {str(path.resolve()): sha256(path) for path in paths.values()},
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    non_metrics = model_payload["non_thinking"]["metrics"]
    native_metrics = model_payload["native_thinking"]["metrics"]
    print(
        f"{MODEL} {rank_template} D/C={discovery_n}/{confirmation_n} "
        f"non L{model_payload['non_thinking']['layer']}="
        f"{non_metrics['confirmation_logistic_balanced_accuracy']:.3f}/"
        f"{non_metrics['confirmation_ncc_balanced_accuracy']:.3f} "
        f"native L{model_payload['native_thinking']['layer']}="
        f"{native_metrics['confirmation_logistic_balanced_accuracy']:.3f}/"
        f"{native_metrics['confirmation_ncc_balanced_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
