#!/usr/bin/env python3
"""Analyse a fixed, explicit ``index + city`` native-thinking grammar.

This is deliberately narrower than the discovery-selected single-grammar
appendix.  The grammar is fixed from the causal route before looking at
geometry, and the primary representation site is the final token of the city
(``city_end``).  Thus the prefix contains both the explicit running index and
the retrieved city.  ``item_end`` is retained only as a same-grammar control.
Layers are selected independently for the two sites with grouped discovery
OOF metrics; confirmation is evaluated only after the layer is frozen.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v5.causal_aligned_geometry import (  # noqa: E402
    load_causal_aligned_native_capture,
)
from realistic_niah_v5.cross_mode_geometry import CLASSES, ModeDataset  # noqa: E402
from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
FIXED_GRAMMAR = {
    "Qwen3-8B": "adjacent_rank_before_city",
    "Gemma4-E4B": "same_unit_rank_before_city",
}
SITES = ("city_end", "item_end")
SCHEMA = "realistic_niah_v5_native_index_city_geometry_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subset(dataset: ModeDataset, mask: np.ndarray) -> ModeDataset:
    result = ModeDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=dataset.metadata.loc[mask].reset_index(drop=True),
        states_by_layer={layer: values[mask] for layer, values in dataset.states_by_layer.items()},
    )
    result.validate()
    return result


def support_table(metadata: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = (
        metadata.groupby(["split", "occurrence"], sort=True)
        .size()
        .rename("n")
        .reset_index()
    )
    expected = set(CLASSES)
    for split in ("discovery", "confirmation"):
        observed = set(frame.loc[frame["split"].eq(split), "occurrence"].astype(int))
        if observed != expected:
            raise ValueError(
                f"Fixed grammar lacks k=1..10 in {split}: "
                f"missing={sorted(expected - observed)}"
            )
    confirmation_min = int(frame.loc[frame["split"].eq("confirmation"), "n"].min())
    if confirmation_min < 2:
        raise ValueError(
            "Fixed grammar has fewer than two confirmation states in at least one class"
        )
    return frame, confirmation_min


def layer_metrics(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    pca_dim: int,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    return {
        **grouped_discovery_cv_metrics(
            states,
            metadata,
            CLASSES,
            pca_dim=pca_dim,
            random_state=seed,
            folds=folds,
            pca_whiten=True,
        ),
        **confirmation_metrics(
            states,
            metadata,
            CLASSES,
            pca_dim=pca_dim,
            random_state=seed,
            pca_whiten=True,
        ),
    }


def select_layer(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        [
            "discovery_selection_score",
            "discovery_oof_ncc_balanced_accuracy",
            "discovery_oof_logistic_balanced_accuracy",
            "layer",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--running-root",
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
        default=ROOT / "reports/v5_native_index_city_geometry",
    )
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "definition": (
            "a priori rank-before-city grammar; primary site city_end so the prefix "
            "contains the explicit index and the retrieved city; item_end is a control"
        ),
        "models": {},
    }

    for model in MODELS:
        grammar = FIXED_GRAMMAR[model]
        payload["models"][model] = {"fixed_grammar": grammar, "sites": {}}
        for site in SITES:
            dataset, loader_audit = load_causal_aligned_native_capture(
                args.running_root / model / "capture_index.jsonl",
                args.event_registry,
                site_kind=site,
            )
            mask = dataset.metadata["grammar_class"].astype(str).eq(grammar).to_numpy()
            fixed = subset(dataset, mask)
            support, confirmation_min = support_table(fixed.metadata)
            supports.append(
                {
                    "model_label": model,
                    "fixed_grammar": grammar,
                    "site": site,
                    "states": int(len(fixed.metadata)),
                    "trajectories": int(fixed.metadata["request_id"].nunique()),
                    "confirmation_min_per_class": confirmation_min,
                    "support": json.dumps(support.to_dict("records"), ensure_ascii=False),
                }
            )

            site_rows: list[dict[str, Any]] = []
            for layer, states in sorted(fixed.states_by_layer.items()):
                row = {
                    "model_label": model,
                    "fixed_grammar": grammar,
                    "site": site,
                    "layer": int(layer),
                    "states": int(len(fixed.metadata)),
                    "trajectories": int(fixed.metadata["request_id"].nunique()),
                    **layer_metrics(
                        states,
                        fixed.metadata,
                        pca_dim=args.pca_dim,
                        folds=args.folds,
                        seed=args.seed,
                    ),
                }
                site_rows.append(row)
                candidates.append(row)
            winner = select_layer(pd.DataFrame(site_rows)).to_dict()
            selected.append(winner)
            layer = int(winner["layer"])
            discovery = fixed.metadata["split"].astype(str).eq("discovery").to_numpy()
            pca = PCA(n_components=3, random_state=args.seed).fit(
                fixed.states_by_layer[layer][discovery].astype(np.float32)
            )
            xyz = pca.transform(fixed.states_by_layer[layer].astype(np.float32))
            payload["models"][model]["sites"][site] = {
                "layer": layer,
                "pca3_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "metrics": {
                    "discovery_selection_score": float(winner["discovery_selection_score"]),
                    "confirmation_logistic": float(
                        winner["confirmation_logistic_balanced_accuracy"]
                    ),
                    "confirmation_ncc": float(
                        winner["confirmation_ncc_balanced_accuracy"]
                    ),
                    "confirmation_snr_db": float(
                        winner["confirmation_class_balanced_snr_db"]
                    ),
                },
                "support": support.to_dict("records"),
                "loader_audit": loader_audit,
                "points": [
                    {
                        "x": float(value[0]),
                        "y": float(value[1]),
                        "z": float(value[2]),
                        "split": str(row.split),
                        "seed": int(row.seed),
                        "occurrence": int(row.occurrence),
                        "gold_count": int(row.gold_count),
                        "request_id": str(row.request_id),
                    }
                    for value, row in zip(xyz, fixed.metadata.itertuples(index=False))
                ],
            }
            print(
                model,
                grammar,
                site,
                f"L{layer}",
                f"confirmation={winner['confirmation_logistic_balanced_accuracy']:.3f}/"
                f"{winner['confirmation_ncc_balanced_accuracy']:.3f}",
            )

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates": args.output / "site_layer_candidates.csv",
        "selected": args.output / "site_selected.csv",
        "support": args.output / "site_support.csv",
        "payload": args.output / "geometry_payload.json",
    }
    pd.DataFrame(candidates).to_csv(paths["candidates"], index=False)
    pd.DataFrame(selected).to_csv(paths["selected"], index=False)
    pd.DataFrame(supports).to_csv(paths["support"], index=False)
    paths["payload"].write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    audit = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixed_grammar": FIXED_GRAMMAR,
        "primary_site": "city_end",
        "control_site": "item_end",
        "selection_rule": (
            "grammar and sites fixed a priori; each site's layer selected independently "
            "by grouped discovery OOF mean Logistic/NCC balanced accuracy; confirmation "
            "is frozen evaluation"
        ),
        "pca_visualization": "PCA3 fitted on discovery states of the fixed grammar/site/layer",
        "outputs": {str(path.resolve()): sha256(path) for path in paths.values()},
    }
    audit_path = args.output / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
