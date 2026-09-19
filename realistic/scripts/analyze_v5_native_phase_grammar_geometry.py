#!/usr/bin/env python3
"""Discovery-only grammar x token-site ablation for native counting states.

For every sufficiently supported trace grammar, this script selects a
``site x layer`` winner using seed-grouped discovery CV and evaluates the
frozen winner on confirmation.  It additionally constructs a conservative
combined view: all grammars share one discovery-selected layer, while each
grammar may use its own discovery-selected token site.  A raw and a
discovery-fitted grammar-centered PCA3 view are both exported; centering is a
labelled nuisance-removal diagnostic, not the primary representation result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analyze_v5_native_phase_geometry import (
    MODELS,
    ROOT,
    SITES,
    atomic_csv,
    atomic_json,
    compactness_payload,
    load_phase_capture,
    pca3_payload,
    select_classification_layer,
    sha256,
)

import sys

sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v5.covariance_geometry import (  # noqa: E402
    evaluate_covariance_geometry_layer,
)
from realistic_niah_v5.cross_mode_geometry import CLASSES, ModeDataset  # noqa: E402
from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)


SCHEMA = "realistic_niah_v5_native_phase_grammar_geometry_v1"


def subset_dataset(dataset: ModeDataset, grammar: str) -> ModeDataset:
    mask = dataset.metadata["grammar_class"].astype(str).eq(grammar).to_numpy()
    result = ModeDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=dataset.metadata.loc[mask].reset_index(drop=True),
        states_by_layer={layer: states[mask] for layer, states in dataset.states_by_layer.items()},
    )
    result.validate()
    return result


def primary_progress_dataset(dataset: ModeDataset) -> ModeDataset:
    required = {"primary_full_chain_event", "progress_commit_eligible"}
    missing = sorted(required - set(dataset.metadata.columns))
    if missing:
        raise ValueError(f"Phase capture metadata lacks primary flags: {missing}")
    mask = (
        dataset.metadata["primary_full_chain_event"].astype(bool)
        & dataset.metadata["progress_commit_eligible"].astype(bool)
    ).to_numpy()
    result = ModeDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=dataset.metadata.loc[mask].reset_index(drop=True),
        states_by_layer={layer: states[mask] for layer, states in dataset.states_by_layer.items()},
    )
    result.validate()
    return result


def grammar_support(dataset: ModeDataset, site: str, grammar: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("discovery", "confirmation"):
        frame = dataset.metadata.loc[dataset.metadata["split"].astype(str).eq(split)]
        for label in CLASSES:
            selected = frame.loc[frame["occurrence"].astype(int).eq(label)]
            rows.append(
                {
                    "model_label": dataset.model_label,
                    "grammar_class": grammar,
                    "site": site,
                    "split": split,
                    "occurrence": label,
                    "states": int(len(selected)),
                    "seeds": int(selected["seed"].nunique()),
                }
            )
    return rows


def eligible(support: pd.DataFrame) -> tuple[bool, str]:
    missing = support.loc[support["states"].astype(int).eq(0)]
    if not missing.empty:
        labels = ", ".join(
            f"{row.split}:k{int(row.occurrence)}" for row in missing.itertuples()
        )
        return False, f"missing full 1..10 support ({labels})"
    discovery = support.loc[support["split"].eq("discovery")]
    if int(discovery["seeds"].min()) < 2:
        return False, "a discovery count is supported by fewer than two seeds"
    return True, "all 10 counts in both splits; >=2 discovery seeds per count"


def grammar_centered_pca3(
    states: np.ndarray, metadata: pd.DataFrame, *, seed: int
) -> dict[str, Any]:
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy()
    grammars = metadata["grammar_class"].astype(str).to_numpy()
    centered = states.astype(np.float32).copy()
    means: dict[str, np.ndarray] = {}
    for grammar in sorted(set(grammars.tolist())):
        fit = discovery & (grammars == grammar)
        means[grammar] = centered[fit].mean(axis=0)
        centered[grammars == grammar] -= means[grammar]
    scaler = StandardScaler().fit(centered[discovery])
    scaled = scaler.transform(centered)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=seed).fit(
        scaled[discovery]
    )
    xyz = pca.transform(scaled)
    points: list[dict[str, Any]] = []
    for value, row in zip(xyz, metadata.itertuples(index=False)):
        points.append(
            {
                "x": float(value[0]),
                "y": float(value[1]),
                "z": float(value[2]),
                "split": str(row.split),
                "seed": int(row.seed),
                "gold_count": int(row.gold_count),
                "occurrence": int(row.occurrence),
                "grammar_class": str(row.grammar_class),
                "site": str(row.selected_site),
            }
        )
    frame = pd.DataFrame(points)
    denoised = (
        frame.groupby(
            ["split", "seed", "occurrence", "grammar_class", "site"], as_index=False
        )[["x", "y", "z"]]
        .mean()
        .to_dict("records")
    )
    return {
        "fit": "discovery grammar-mean residual + StandardScaler + PCA3",
        "evr": pca.explained_variance_ratio_.tolist(),
        "raw_points": points,
        "seed_count_means": denoised,
        "warning": (
            "Grammar means are estimated on discovery and subtracted only as an "
            "explicit nuisance-removal sensitivity analysis; all primary metrics "
            "use the uncentered raw states."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    shared_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"schema_version": SCHEMA, "models": {}}
    input_paths: list[Path] = []

    for model in args.models:
        capture_index = (args.capture_root / model / "capture_index.jsonl").resolve()
        input_paths.append(capture_index)
        datasets = {
            site: primary_progress_dataset(load_phase_capture(capture_index, site))
            for site in SITES
        }
        grammars = sorted(
            set().union(
                *(set(ds.metadata["grammar_class"].astype(str)) for ds in datasets.values())
            )
        )
        model_payload: dict[str, Any] = {"grammars": {}, "shared_layer": {}}
        eligible_grammars: list[str] = []

        for grammar in grammars:
            model_payload["grammars"][grammar] = {"sites": {}}
            grammar_candidates: list[dict[str, Any]] = []
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
            eligible_grammars.append(grammar)
            model_payload["grammars"][grammar]["winner"] = winner
            model_payload["grammars"][grammar]["pca3"] = pca3_payload(
                dataset.states_by_layer[layer], dataset.metadata, seed=args.seed
            )

        # Constrained combined ablation: one layer for the model, but each
        # eligible grammar can choose its best site at that layer.
        model_candidates = pd.DataFrame(candidate_rows)
        model_candidates = model_candidates.loc[
            model_candidates["model_label"].eq(model)
            & model_candidates["grammar_class"].isin(eligible_grammars)
        ]
        layer_options: list[dict[str, Any]] = []
        for layer in sorted(model_candidates["layer"].unique().tolist()):
            chosen: list[pd.Series] = []
            for grammar in eligible_grammars:
                frame = model_candidates.loc[
                    model_candidates["grammar_class"].eq(grammar)
                    & model_candidates["layer"].eq(layer)
                ]
                if frame.empty:
                    break
                chosen.append(select_classification_layer(frame))
            if len(chosen) == len(eligible_grammars) and chosen:
                layer_options.append(
                    {
                        "layer": int(layer),
                        "mean_discovery_selection_score": float(
                            np.mean([float(row["discovery_selection_score"]) for row in chosen])
                        ),
                    }
                )
        if layer_options:
            shared_layer = int(
                sorted(
                    layer_options,
                    key=lambda row: (-row["mean_discovery_selection_score"], row["layer"]),
                )[0]["layer"]
            )
            metadata_parts: list[pd.DataFrame] = []
            state_parts: list[np.ndarray] = []
            selected_sites: dict[str, str] = {}
            for grammar in eligible_grammars:
                frame = model_candidates.loc[
                    model_candidates["grammar_class"].eq(grammar)
                    & model_candidates["layer"].eq(shared_layer)
                ]
                row = select_classification_layer(frame).to_dict()
                site = str(row["site"])
                selected_sites[grammar] = site
                shared_rows.append({**row, "shared_layer": shared_layer})
                dataset = subset_dataset(datasets[site], grammar)
                metadata = dataset.metadata.copy()
                metadata["selected_site"] = site
                metadata_parts.append(metadata)
                state_parts.append(dataset.states_by_layer[shared_layer])
            pooled_metadata = pd.concat(metadata_parts, ignore_index=True)
            pooled_states = np.concatenate(state_parts, axis=0)
            order = np.lexsort(
                (
                    pooled_metadata["occurrence"].to_numpy(dtype=int),
                    pooled_metadata["gold_count"].to_numpy(dtype=int),
                    pooled_metadata["seed"].to_numpy(dtype=int),
                    pooled_metadata["split"].astype(str).to_numpy(),
                )
            )
            pooled_metadata = pooled_metadata.iloc[order].reset_index(drop=True)
            pooled_states = pooled_states[order]
            pooled_metrics = {
                **grouped_discovery_cv_metrics(
                    pooled_states,
                    pooled_metadata,
                    CLASSES,
                    pca_dim=args.pca_dim,
                    random_state=args.seed,
                    folds=args.folds,
                    pca_whiten=True,
                ),
                **confirmation_metrics(
                    pooled_states,
                    pooled_metadata,
                    CLASSES,
                    pca_dim=args.pca_dim,
                    random_state=args.seed,
                    pca_whiten=True,
                ),
            }
            pooled_covariance = evaluate_covariance_geometry_layer(
                pooled_states,
                pooled_metadata,
                CLASSES,
                pca_dim=args.pca_dim,
                random_state=args.seed,
                discovery_cv_folds=args.folds,
            )
            pooled_metrics.update(
                {f"cov_{key}": value for key, value in pooled_covariance.items() if key != "metric_definitions"}
            )
            pooled_metrics["confirmation_radius_gap_ratio"] = compactness_payload(
                pooled_states,
                pooled_metadata,
                pca_dim=args.pca_dim,
                seed=args.seed,
            )["class_balanced_radius_gap_ratio"]
            model_payload["shared_layer"] = {
                "layer": shared_layer,
                "selected_sites": selected_sites,
                "eligible_grammars": eligible_grammars,
                "metrics": pooled_metrics,
                "raw_pca3": pca3_payload(pooled_states, pooled_metadata, seed=args.seed),
                "grammar_centered_pca3": grammar_centered_pca3(
                    pooled_states, pooled_metadata, seed=args.seed
                ),
                "selection": (
                    "discovery-only: maximize equal-grammar mean grouped-CV score "
                    "over a shared layer, with a grammar-specific site at that layer"
                ),
            }
        payload["models"][model] = model_payload

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "support": args.output / "grammar_site_support.csv",
        "eligibility": args.output / "grammar_site_eligibility.csv",
        "candidates": args.output / "grammar_site_layer_candidates.csv",
        "selected": args.output / "grammar_selected.csv",
        "shared": args.output / "shared_layer_selected_sites.csv",
        "payload": args.output / "grammar_geometry_payload.json",
    }
    atomic_csv(paths["support"], pd.DataFrame(support_rows))
    atomic_csv(paths["eligibility"], pd.DataFrame(eligibility_rows))
    atomic_csv(paths["candidates"], pd.DataFrame(candidate_rows))
    atomic_csv(paths["selected"], pd.DataFrame(selected_rows))
    atomic_csv(paths["shared"], pd.DataFrame(shared_rows))
    atomic_json(paths["payload"], payload)
    audit = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sites": list(SITES),
        "classes": list(CLASSES),
        "selection": (
            "primary_full_chain_event and progress_commit_eligible only; grammar-specific "
            "site x layer winner selected only by grouped discovery CV; confirmation is frozen"
        ),
        "shared_layer_selection": (
            "one layer per model selected on equal-grammar mean discovery score; "
            "site may vary by grammar; raw and explicitly grammar-centered views exported"
        ),
        "inputs": {str(path): sha256(path) for path in input_paths},
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
