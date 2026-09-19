#!/usr/bin/env python3
"""Compare N=10 pure-grammar native traces with paired non-thinking traces.

Unlike the event-level grammar filter, this analysis retains a trajectory only
when all ten registered progress commits form one complete one-to-one sequence,
use one grammar class and one marker kind, and carry ranks 1..10 exactly.  The
native grammar is selected by the number of qualifying discovery trajectories;
hidden states and confirmation outcomes never select the grammar.  Non-thinking
then uses exactly the same split/seed/N/running-k cells.  Each mode independently
selects its layer using grouped discovery OOF balanced accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v5.causal_aligned_geometry import (  # noqa: E402
    load_causal_aligned_native_capture,
)
from realistic_niah_v5.covariance_geometry import (  # noqa: E402
    class_balanced_silhouette,
    ordinal_centroid_rsa,
)
from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    CLASSES,
    ModeDataset,
    load_non_thinking_capture,
)
from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
KEY_COLUMNS = ("split", "seed", "gold_count", "occurrence")
TARGET_N = 10
SCHEMA = "realistic_niah_v5_pure_trace_n10_cross_mode_geometry_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subset(dataset: ModeDataset, indices: np.ndarray) -> ModeDataset:
    result = ModeDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=dataset.metadata.iloc[indices].reset_index(drop=True),
        states_by_layer={
            layer: values[indices] for layer, values in dataset.states_by_layer.items()
        },
    )
    result.validate()
    return result


def concatenate_datasets(*datasets: ModeDataset) -> ModeDataset:
    if not datasets:
        raise ValueError("At least one mode dataset is required")
    first = datasets[0]
    layers = set(first.states_by_layer)
    for dataset in datasets[1:]:
        if dataset.mode != first.mode or dataset.model_label != first.model_label:
            raise ValueError("Cannot concatenate different models or modes")
        if set(dataset.states_by_layer) != layers:
            raise ValueError("Base and supplement layer grids differ")
        for layer in layers:
            if dataset.states_by_layer[layer].shape[1] != first.states_by_layer[layer].shape[1]:
                raise ValueError("Base and supplement hidden sizes differ")
    result = ModeDataset(
        mode=first.mode,
        model_label=first.model_label,
        metadata=pd.concat(
            [dataset.metadata for dataset in datasets], ignore_index=True
        ),
        states_by_layer={
            layer: np.concatenate(
                [dataset.states_by_layer[layer] for dataset in datasets], axis=0
            )
            for layer in sorted(layers)
        },
    )
    result.validate()
    return result


def _truth(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna(False).astype(str).str.lower().eq("true")


def pure_trace_registry(registry: pd.DataFrame) -> pd.DataFrame:
    """Return parser-only N=10 trajectories whose ten items share one grammar."""

    eligible = registry.loc[
        _truth(registry, "primary_full_chain_event")
        & _truth(registry, "progress_commit_eligible")
        & _truth(registry, "progress_commit_site_resolved")
        & _truth(registry, "exact_count")
        & registry["trace_category"].astype(str).eq("one_to_one")
        & registry["gold_count"].astype(int).eq(TARGET_N)
        & registry["parsed_count"].astype(int).eq(TARGET_N)
    ].copy()
    rows: list[dict[str, Any]] = []
    for (model, request_id), group in eligible.groupby(
        ["model_label", "request_id"], sort=True
    ):
        occurrences = group["occurrence"].astype(int).tolist()
        ranks = group["rank"].astype(int).tolist()
        grammars = sorted(set(group["grammar_class"].astype(str)))
        marker_kinds = sorted(set(group["marker_kind"].astype(str)))
        if len(group) != TARGET_N:
            continue
        if sorted(occurrences) != list(CLASSES) or sorted(ranks) != list(CLASSES):
            continue
        if any(rank != occurrence for rank, occurrence in zip(ranks, occurrences)):
            continue
        if len(grammars) != 1 or len(marker_kinds) != 1:
            continue
        first = group.iloc[0]
        rows.append(
            {
                "model_label": str(model),
                "request_id": str(request_id),
                "split": str(first["split"]),
                "seed": int(first["seed"]),
                "gold_count": TARGET_N,
                "grammar_class": grammars[0],
                "marker_kind": marker_kinds[0],
                "states": TARGET_N,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No parser-defined pure N=10 trajectories were found")
    return result.sort_values(
        ["model_label", "split", "seed", "grammar_class"], kind="mergesort"
    ).reset_index(drop=True)


def grammar_support(pure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, grammar, marker), group in pure.groupby(
        ["model_label", "grammar_class", "marker_kind"], sort=True
    ):
        discovery = group.loc[group["split"].astype(str).eq("discovery")]
        confirmation = group.loc[group["split"].astype(str).eq("confirmation")]
        rows.append(
            {
                "model_label": str(model),
                "grammar_class": str(grammar),
                "marker_kind": str(marker),
                "discovery_trajectories": int(len(discovery)),
                "confirmation_trajectories": int(len(confirmation)),
                "total_trajectories": int(len(group)),
                "discovery_seeds": ",".join(map(str, sorted(discovery["seed"].astype(int)))),
                "confirmation_seeds": ",".join(
                    map(str, sorted(confirmation["seed"].astype(int)))
                ),
            }
        )
    return pd.DataFrame(rows)


def select_grammar(support: pd.DataFrame, model: str) -> dict[str, Any]:
    candidates = support.loc[support["model_label"].astype(str).eq(model)].copy()
    if candidates.empty:
        raise ValueError(f"No pure N=10 grammar candidates for {model}")
    # Only discovery support and deterministic names select the grammar.  The
    # confirmation count is audited after selection and never enters ordering.
    winner = candidates.sort_values(
        ["discovery_trajectories", "grammar_class", "marker_kind"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    if int(winner["discovery_trajectories"]) < 2:
        raise ValueError(f"Pure N=10 winner for {model} has fewer than two discovery seeds")
    if int(winner["confirmation_trajectories"]) < 1:
        raise ValueError(f"Pure N=10 winner for {model} has no confirmation trajectory")
    return winner.to_dict()


def _sort_indices(frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.reset_index().sort_values(list(KEY_COLUMNS), kind="mergesort")
    return ordered["index"].to_numpy(dtype=int)


def paired_datasets(
    native: ModeDataset,
    non_thinking: ModeDataset,
    selected_traces: pd.DataFrame,
    grammar: str,
) -> tuple[ModeDataset, ModeDataset]:
    request_ids = set(selected_traces["request_id"].astype(str))
    native_mask = (
        native.metadata["request_id"].astype(str).isin(request_ids)
        & native.metadata["grammar_class"].astype(str).eq(grammar)
        & native.metadata["gold_count"].astype(int).eq(TARGET_N)
    ).to_numpy()
    native_filtered = subset(native, np.flatnonzero(native_mask))
    native_filtered = subset(native_filtered, _sort_indices(native_filtered.metadata))
    observed_requests = set(native_filtered.metadata["request_id"].astype(str))
    if observed_requests != request_ids:
        missing = sorted(request_ids - observed_requests)
        raise ValueError(f"Archived native states lack pure traces: {missing}")
    native_sizes = native_filtered.metadata.groupby("request_id").size()
    if not native_sizes.eq(TARGET_N).all():
        raise ValueError("Pure native trajectories do not each contain ten states")
    native_keys = list(
        native_filtered.metadata[list(KEY_COLUMNS)].itertuples(index=False, name=None)
    )
    if len(set(native_keys)) != len(native_keys):
        raise ValueError("Pure native split/seed/N/k cells are not unique")

    key_set = set(native_keys)
    non_keys = non_thinking.metadata[list(KEY_COLUMNS)].apply(tuple, axis=1)
    non_mask = non_keys.isin(key_set).to_numpy()
    non_filtered = subset(non_thinking, np.flatnonzero(non_mask))
    non_filtered = subset(non_filtered, _sort_indices(non_filtered.metadata))
    paired_non_keys = list(
        non_filtered.metadata[list(KEY_COLUMNS)].itertuples(index=False, name=None)
    )
    if native_keys != paired_non_keys:
        raise ValueError(
            f"Pure N=10 pairing failed: native={len(native_keys)}, "
            f"non-thinking={len(paired_non_keys)}"
        )
    non_filtered.metadata["request_id"] = native_filtered.metadata["request_id"].astype(str)
    non_filtered.metadata["grammar_class"] = grammar
    return native_filtered, non_filtered


def discovery_metrics(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    pca_dim: int,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    return grouped_discovery_cv_metrics(
        states,
        metadata,
        CLASSES,
        pca_dim=pca_dim,
        folds=folds,
        random_state=seed,
        pca_whiten=True,
    )


def choose_layer(
    dataset: ModeDataset,
    *,
    pca_dim: int,
    folds: int,
    seed: int,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for layer, states in sorted(dataset.states_by_layer.items()):
        candidates.append(
            {
                "layer": int(layer),
                **discovery_metrics(
                    states,
                    dataset.metadata,
                    pca_dim=pca_dim,
                    folds=folds,
                    seed=seed,
                ),
            }
        )
    winner = sorted(
        candidates,
        key=lambda row: (
            -float(row["discovery_selection_score"]),
            -float(row["discovery_oof_ncc_balanced_accuracy"]),
            -float(row["discovery_oof_logistic_balanced_accuracy"]),
            int(row["layer"]),
        ),
    )[0]
    return int(winner["layer"]), winner, candidates


def frozen_metrics(
    dataset: ModeDataset,
    layer: int,
    discovery: dict[str, Any],
    *,
    pca_dim: int,
    seed: int,
) -> dict[str, Any]:
    return {
        **discovery,
        **confirmation_metrics(
            dataset.states_by_layer[layer],
            dataset.metadata,
            CLASSES,
            pca_dim=pca_dim,
            random_state=seed,
            pca_whiten=True,
        ),
    }


def geometry_payload(
    dataset: ModeDataset,
    layer: int,
    metrics: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    discovery = dataset.metadata["split"].astype(str).eq("discovery").to_numpy()
    states = dataset.states_by_layer[layer].astype(np.float32)
    scaler = StandardScaler().fit(states[discovery])
    scaled = scaler.transform(states)
    pca = PCA(n_components=3, random_state=seed).fit(scaled[discovery])
    xyz = pca.transform(scaled)
    confirmation = dataset.metadata["split"].astype(str).eq("confirmation").to_numpy()
    labels = dataset.metadata.loc[confirmation, "occurrence"].to_numpy(dtype=int)
    points = []
    for value, row in zip(xyz, dataset.metadata.itertuples(index=False)):
        points.append(
            {
                "x": float(value[0]),
                "y": float(value[1]),
                "z": float(value[2]),
                "split": str(row.split),
                "seed": int(row.seed),
                "gold_count": int(row.gold_count),
                "occurrence": int(row.occurrence),
                "request_id": str(row.request_id),
            }
        )
    return {
        "layer": int(layer),
        "pca3_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "confirmation_pca3_class_balanced_silhouette": float(
            class_balanced_silhouette(xyz[confirmation], labels, CLASSES)
        ),
        "confirmation_pca3_ordinal_rsa": float(
            ordinal_centroid_rsa(xyz[confirmation], labels, CLASSES)
        ),
        "metrics": metrics,
        "points": points,
    }


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
        "--output",
        type=Path,
        default=ROOT / "reports/v5_pure_trace_n10_cross_mode_geometry",
    )
    parser.add_argument(
        "--supplement-root",
        type=Path,
        help=(
            "Optional paired capture root containing <model>/native and "
            "<model>/non_thinking capture indexes"
        ),
    )
    parser.add_argument(
        "--supplement-event-registry",
        type=Path,
        help="Event registry compiled from the isolated supplement generations",
    )
    parser.add_argument(
        "--frozen-grammar-selection",
        type=Path,
        help=(
            "Optional pre-supplement selected_pure_trace_grammar.csv. When set, "
            "the grammar/marker target is frozen and new discovery traces cannot "
            "change it."
        ),
    )
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.supplement_root is None) != (args.supplement_event_registry is None):
        raise ValueError(
            "--supplement-root and --supplement-event-registry must be provided together"
        )
    registry_paths = [args.event_registry]
    if args.supplement_event_registry is not None:
        registry_paths.append(args.supplement_event_registry)
    registry = pd.concat(
        [pd.read_csv(path) for path in registry_paths], ignore_index=True
    )
    analysis_input_paths = list(registry_paths)
    if args.frozen_grammar_selection is not None:
        analysis_input_paths.append(args.frozen_grammar_selection)
    pure = pure_trace_registry(registry)
    support = grammar_support(pure)
    frozen_by_model: dict[str, dict[str, Any]] = {}
    if args.frozen_grammar_selection is not None:
        frozen = pd.read_csv(args.frozen_grammar_selection)
        frozen_by_model = {
            str(row["model_label"]): row.to_dict() for _, row in frozen.iterrows()
        }
        if set(frozen_by_model) != set(MODELS):
            raise ValueError("Frozen grammar selection does not cover both models")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "target_gold_count": TARGET_N,
        "filter_unit": "whole trajectory",
        "models": {},
    }
    selected_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for model in MODELS:
        if frozen_by_model:
            frozen = frozen_by_model[model]
            grammar = str(frozen["grammar_class"])
            marker_kind = str(frozen["marker_kind"])
            matches = support.loc[
                support["model_label"].astype(str).eq(model)
                & support["grammar_class"].astype(str).eq(grammar)
                & support["marker_kind"].astype(str).eq(marker_kind)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Frozen pure grammar {model}/{grammar}/{marker_kind} "
                    "is absent or non-unique after supplementation"
                )
            selected = matches.iloc[0].to_dict()
        else:
            selected = select_grammar(support, model)
            grammar = str(selected["grammar_class"])
        selected_rows.append(selected)
        trace_rows = pure.loc[
            pure["model_label"].astype(str).eq(model)
            & pure["grammar_class"].astype(str).eq(grammar)
        ].copy()
        base_native_index = args.native_running_root / model / "capture_index.jsonl"
        base_non_index = (
            args.non_thinking_root
            / model
            / "numeric/representation/capture/capture_index.jsonl"
        )
        analysis_input_paths.extend((base_native_index, base_non_index))
        native, native_audit = load_causal_aligned_native_capture(
            base_native_index,
            args.event_registry,
            site_kind="item_end",
        )
        non = load_non_thinking_capture(
            base_non_index,
            design_variant="v4.4",
            pooling="span_end",
        )
        if args.supplement_root is not None:
            supplement_native_index = (
                args.supplement_root / model / "native/capture_index.jsonl"
            )
            supplement_non_index = (
                args.supplement_root
                / model
                / "non_thinking/capture_index.jsonl"
            )
            analysis_input_paths.extend(
                (supplement_native_index, supplement_non_index)
            )
            supplement_native, supplement_native_audit = (
                load_causal_aligned_native_capture(
                    supplement_native_index,
                    args.supplement_event_registry,
                    site_kind="item_end",
                )
            )
            supplement_non = load_non_thinking_capture(
                supplement_non_index,
                design_variant="v4.4",
                pooling="span_end",
            )
            native = concatenate_datasets(native, supplement_native)
            non = concatenate_datasets(non, supplement_non)
            native_audit = {
                "base": native_audit,
                "supplement": supplement_native_audit,
            }
        native, non = paired_datasets(native, non, trace_rows, grammar)

        model_payload: dict[str, Any] = {
            "grammar_class": grammar,
            "marker_kind": str(selected["marker_kind"]),
            "pairing_key": list(KEY_COLUMNS),
            "pure_trace_filter": (
                "N=10; one_to_one; exact final count; ten unique commits; ranks 1..10; "
                "one grammar_class and one marker_kind across the whole trace"
            ),
            "selected_trace_ids": trace_rows["request_id"].astype(str).tolist(),
            "native_loader_audit": native_audit,
        }
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
            model_payload[mode] = geometry
            support_by_split = dataset.metadata.groupby("split")["request_id"].nunique()
            metric_rows.append(
                {
                    "model_label": model,
                    "mode": mode,
                    "grammar_class": grammar,
                    "marker_kind": str(selected["marker_kind"]),
                    "gold_count": TARGET_N,
                    "layer": int(layer),
                    "states": int(len(dataset.metadata)),
                    "trajectories": int(dataset.metadata["request_id"].nunique()),
                    "discovery_trajectories": int(support_by_split.get("discovery", 0)),
                    "confirmation_trajectories": int(
                        support_by_split.get("confirmation", 0)
                    ),
                    "confirmation_per_class": int(
                        support_by_split.get("confirmation", 0)
                    ),
                    **metrics,
                    "confirmation_pca3_class_balanced_silhouette": geometry[
                        "confirmation_pca3_class_balanced_silhouette"
                    ],
                    "confirmation_pca3_ordinal_rsa": geometry[
                        "confirmation_pca3_ordinal_rsa"
                    ],
                }
            )
            for row in candidates:
                candidate_rows.append(
                    {"model_label": model, "mode": mode, "grammar_class": grammar, **row}
                )
        payload["models"][model] = model_payload
        non_metrics = model_payload["non_thinking"]["metrics"]
        native_metrics = model_payload["native_thinking"]["metrics"]
        print(
            model,
            grammar,
            f"traces D/C={selected['discovery_trajectories']}/"
            f"{selected['confirmation_trajectories']}",
            f"non L{model_payload['non_thinking']['layer']}="
            f"{non_metrics['confirmation_logistic_balanced_accuracy']:.3f}/"
            f"{non_metrics['confirmation_ncc_balanced_accuracy']:.3f}",
            f"native L{model_payload['native_thinking']['layer']}="
            f"{native_metrics['confirmation_logistic_balanced_accuracy']:.3f}/"
            f"{native_metrics['confirmation_ncc_balanced_accuracy']:.3f}",
        )

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "support": args.output / "pure_trace_grammar_support.csv",
        "selection": args.output / "selected_pure_trace_grammar.csv",
        "metrics": args.output / "paired_metrics.csv",
        "candidates": args.output / "layer_candidates.csv",
        "payload": args.output / "geometry_payload.json",
    }
    support.to_csv(paths["support"], index=False)
    pd.DataFrame(selected_rows).to_csv(paths["selection"], index=False)
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(candidate_rows).to_csv(paths["candidates"], index=False)
    paths["payload"].write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    audit = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_gold_count": TARGET_N,
        "grammar_selection": (
            "grammar and marker kind frozen from the pre-supplement discovery panel; "
            "supplement discovery improves layer/PCA fitting but cannot change the subgroup"
            if args.frozen_grammar_selection is not None
            else "maximize the number of parser-qualified pure N=10 discovery trajectories; "
            "tie by grammar_class then marker_kind; no hidden state or confirmation metric"
        ),
        "trace_filter": (
            "whole-trajectory one_to_one exact-count filter; ten unique eligible progress "
            "commits; rank equals occurrence 1..10; one grammar and marker kind"
        ),
        "layer_selection": (
            "each mode independently maximizes grouped discovery OOF mean Logistic/NCC "
            "balanced accuracy; confirmation evaluated only after freeze"
        ),
        "pairing_key": list(KEY_COLUMNS),
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in analysis_input_paths
        },
        "outputs": {str(path.resolve()): sha256(path) for path in paths.values()},
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
