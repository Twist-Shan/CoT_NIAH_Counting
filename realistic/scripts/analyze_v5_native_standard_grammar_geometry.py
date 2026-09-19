#!/usr/bin/env python3
"""Grammar-stratified sweep over the five archived native semantic sites.

This CPU-only analysis complements ``analyze_v5_native_phase_grammar_geometry``.
It joins archived all-layer states to the frozen causal event registry, then
selects ``site x layer`` using seed-grouped discovery CV within every grammar
that covers counts 1..10 in both splits.  Confirmation remains frozen.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_v5_native_phase_geometry import (
    MODELS,
    ROOT,
    atomic_csv,
    atomic_json,
    compactness_payload,
    pca3_payload,
    select_classification_layer,
    sha256,
)
from analyze_v5_native_phase_grammar_geometry import (
    eligible,
    grammar_support,
    subset_dataset,
)

import sys

sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v5.causal_aligned_geometry import (  # noqa: E402
    load_causal_aligned_native_capture,
)
from realistic_niah_v5.covariance_geometry import (  # noqa: E402
    evaluate_covariance_geometry_layer,
)
from realistic_niah_v5.cross_mode_geometry import CLASSES  # noqa: E402
from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)


SCHEMA = "realistic_niah_v5_native_standard_grammar_geometry_v1"
SITES = ("pre_city", "city_end", "city_unit_end", "item_end", "post_boundary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-running-root",
        type=Path,
        default=ROOT / "work" / "v5_geometry_full_panel" / "running",
    )
    parser.add_argument(
        "--event-registry",
        type=Path,
        default=ROOT / "reports" / "v5_native_causal_site_review" / "event_registry.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "work" / "v5_native_standard_grammar_geometry",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    support_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"schema_version": SCHEMA, "models": {}}
    inputs = [args.event_registry.resolve()]

    for model in args.models:
        capture_index = (args.native_running_root / model / "capture_index.jsonl").resolve()
        inputs.append(capture_index)
        datasets = {}
        loader_audits = {}
        for site in SITES:
            dataset, audit = load_causal_aligned_native_capture(
                capture_index, args.event_registry, site_kind=site
            )
            dataset.metadata = dataset.metadata.copy()
            dataset.metadata["token_surface_class"] = "archived_semantic_site"
            datasets[site] = dataset
            loader_audits[site] = audit
        grammars = sorted(
            set().union(
                *(set(ds.metadata["grammar_class"].astype(str)) for ds in datasets.values())
            )
        )
        model_payload: dict[str, Any] = {
            "grammars": {},
            "loader_audits": loader_audits,
        }
        for grammar in grammars:
            grammar_candidates: list[dict[str, Any]] = []
            model_payload["grammars"][grammar] = {"sites": {}}
            for site, full_dataset in datasets.items():
                dataset = subset_dataset(full_dataset, grammar)
                rows = grammar_support(dataset, site, grammar)
                support_rows.extend(rows)
                ok, reason = eligible(pd.DataFrame(rows))
                eligibility_rows.append(
                    {
                        "model_label": model,
                        "grammar_class": grammar,
                        "site": site,
                        "eligible": ok,
                        "reason": reason,
                        "states": int(len(dataset.metadata)),
                        "trajectories": int(dataset.metadata["request_id"].nunique())
                        if len(dataset.metadata)
                        else 0,
                    }
                )
                if not ok:
                    continue
                site_candidates: list[dict[str, Any]] = []
                cv_error = ""
                for layer, states in sorted(dataset.states_by_layer.items()):
                    try:
                        metrics = {
                            **grouped_discovery_cv_metrics(
                                states,
                                dataset.metadata,
                                CLASSES,
                                pca_dim=args.pca_dim,
                                random_state=args.seed,
                                folds=args.folds,
                                pca_whiten=True,
                            ),
                            **confirmation_metrics(
                                states,
                                dataset.metadata,
                                CLASSES,
                                pca_dim=args.pca_dim,
                                random_state=args.seed,
                                pca_whiten=True,
                            ),
                        }
                    except ValueError as error:
                        cv_error = str(error)
                        break
                    row = {
                        "model_label": model,
                        "grammar_class": grammar,
                        "site": site,
                        "layer": int(layer),
                        "states": int(len(dataset.metadata)),
                        **metrics,
                    }
                    candidate_rows.append(row)
                    grammar_candidates.append(row)
                    site_candidates.append(row)
                if cv_error:
                    eligibility_rows[-1]["eligible"] = False
                    eligibility_rows[-1]["reason"] = f"grouped CV invalid: {cv_error}"
                    continue
                site_winner = select_classification_layer(pd.DataFrame(site_candidates))
                model_payload["grammars"][grammar]["sites"][site] = {
                    "best_layer": int(site_winner["layer"]),
                    "discovery_selection_score": float(
                        site_winner["discovery_selection_score"]
                    ),
                }
            if not grammar_candidates:
                continue
            winner = select_classification_layer(pd.DataFrame(grammar_candidates)).to_dict()
            site = str(winner["site"])
            layer = int(winner["layer"])
            dataset = subset_dataset(datasets[site], grammar)
            covariance = evaluate_covariance_geometry_layer(
                dataset.states_by_layer[layer],
                dataset.metadata,
                CLASSES,
                pca_dim=args.pca_dim,
                random_state=args.seed,
                discovery_cv_folds=args.folds,
            )
            winner.update(
                {f"cov_{key}": value for key, value in covariance.items() if key != "metric_definitions"}
            )
            winner["confirmation_radius_gap_ratio"] = compactness_payload(
                dataset.states_by_layer[layer],
                dataset.metadata,
                pca_dim=args.pca_dim,
                seed=args.seed,
            )["class_balanced_radius_gap_ratio"]
            selected_rows.append(winner)
            model_payload["grammars"][grammar]["winner"] = winner
            model_payload["grammars"][grammar]["pca3"] = pca3_payload(
                dataset.states_by_layer[layer], dataset.metadata, seed=args.seed
            )
        payload["models"][model] = model_payload

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "support": args.output / "standard_grammar_site_support.csv",
        "eligibility": args.output / "standard_grammar_site_eligibility.csv",
        "candidates": args.output / "standard_grammar_site_layer_candidates.csv",
        "selected": args.output / "standard_grammar_selected.csv",
        "payload": args.output / "standard_grammar_geometry_payload.json",
    }
    atomic_csv(paths["support"], pd.DataFrame(support_rows))
    atomic_csv(paths["eligibility"], pd.DataFrame(eligibility_rows))
    atomic_csv(paths["candidates"], pd.DataFrame(candidate_rows))
    atomic_csv(paths["selected"], pd.DataFrame(selected_rows))
    atomic_json(paths["payload"], payload)
    audit = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sites": list(SITES),
        "classes": list(CLASSES),
        "selection": "grammar-specific semantic site x layer selected by grouped discovery CV; confirmation frozen",
        "causal_alignment": "every site is retained only after exact archived item_end == causal commit token",
        "inputs": {str(path): sha256(path) for path in inputs},
        "outputs": {str(path): sha256(path) for path in paths.values()},
    }
    atomic_json(args.output / "audit.json", audit)
    for row in selected_rows:
        print(
            row["model_label"],
            row["grammar_class"],
            row["site"],
            f"L{int(row['layer'])}",
            f"C-Log/NCC={row['confirmation_logistic_balanced_accuracy']:.3f}/"
            f"{row['confirmation_ncc_balanced_accuracy']:.3f}",
            f"silhouette={row['cov_confirmation_mahalanobis_silhouette']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
