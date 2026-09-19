#!/usr/bin/env python3
"""Re-evaluate native counter geometry on frozen causal progress events.

This is a CPU-only analysis over archived all-layer hidden states.  It does not
run either language model.  Site and layer selection use discovery seeds only;
confirmation seeds are evaluated once after the winner is frozen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal_aligned_geometry import (  # noqa: E402
    PRIMARY_PROGRESS_FILTER,
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


MODELS = ("Qwen3-8B", "Gemma4-E4B")
DEFAULT_SITES = ("pre_city", "city_end", "city_unit_end", "item_end", "post_boundary")
PRIMARY_SITE = "item_end"
SCHEMA_VERSION = "realistic_niah_v5_native_causal_aligned_geometry_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_one(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        [
            "discovery_selection_score",
            "discovery_oof_ncc_balanced_accuracy",
            "discovery_oof_logistic_balanced_accuracy",
            "layer",
            "site_kind",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).iloc[0]


def validate_support(metadata: pd.DataFrame, *, context: str) -> dict[str, Any]:
    support = (
        metadata.groupby(["split", "occurrence"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    for split in ("discovery", "confirmation"):
        observed = set(
            support.loc[support["split"].astype(str).eq(split), "occurrence"]
            .astype(int)
            .tolist()
        )
        if observed != set(CLASSES):
            raise ValueError(
                f"{context}/{split} does not cover all ten counters: {sorted(observed)}"
            )
    confirmation_min = int(
        support.loc[support["split"].astype(str).eq("confirmation"), "count"].min()
    )
    if confirmation_min < 3:
        raise ValueError(
            f"{context} needs at least three held-out rows per counter; "
            f"minimum={confirmation_min}"
        )
    return {
        "rows": int(len(metadata)),
        "trajectories": int(metadata["request_id"].nunique()),
        "confirmation_min_per_counter": confirmation_min,
        "support": support.to_dict(orient="records"),
    }


def candidate_metrics(
    states,
    metadata: pd.DataFrame,
    *,
    pca_dim: int,
    folds: int,
    random_state: int,
) -> dict[str, Any]:
    discovery = grouped_discovery_cv_metrics(
        states,
        metadata,
        CLASSES,
        pca_dim=pca_dim,
        random_state=random_state,
        folds=folds,
        pca_whiten=True,
    )
    confirmation = confirmation_metrics(
        states,
        metadata,
        CLASSES,
        pca_dim=pca_dim,
        random_state=random_state,
        pca_whiten=True,
    )
    return {**discovery, **confirmation}


def legacy_row(legacy_root: Path, model: str) -> dict[str, Any]:
    path = legacy_root / model / "pca16_whiten" / "running_index_selected.csv"
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["mode"].astype(str).eq("native_thinking")
        & frame["analysis_group"].astype(str).eq("all_traces")
        & frame["token_site"].astype(str).eq(PRIMARY_SITE)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one legacy native running row in {path}")
    return selected.iloc[0].to_dict()


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
        "--legacy-dual-root",
        type=Path,
        default=ROOT / "reports" / "v5_dual_endpoint_geometry_full300",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "v5_native_causal_aligned_geometry",
    )
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--discovery-cv-folds", type=int, default=5)
    parser.add_argument("--relative-ridge", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sites = tuple(dict.fromkeys(map(str, args.sites)))
    if PRIMARY_SITE not in sites:
        raise ValueError(f"Primary causal progress site {PRIMARY_SITE!r} is required")
    all_candidates: list[dict[str, Any]] = []
    site_selected: list[dict[str, Any]] = []
    loader_audits: dict[str, dict[str, Any]] = {}
    model_key_sha: dict[str, str] = {}
    support_audit: dict[str, dict[str, Any]] = {}

    for model in MODELS:
        capture_index = args.native_running_root / model / "capture_index.jsonl"
        for site in sites:
            print(f"[causal-geometry] loading {model} / {site}", flush=True)
            dataset, loader_audit = load_causal_aligned_native_capture(
                capture_index,
                args.event_registry,
                site_kind=site,
            )
            key_sha = str(loader_audit["metadata_key_sha256"])
            if model in model_key_sha and model_key_sha[model] != key_sha:
                raise ValueError(
                    f"Site support is not paired for {model}: "
                    f"{model_key_sha[model]} versus {key_sha} at {site}"
                )
            model_key_sha.setdefault(model, key_sha)
            loader_audits[f"{model}/{site}"] = loader_audit
            support_audit.setdefault(
                model, validate_support(dataset.metadata, context=f"{model}/{site}")
            )

            site_rows: list[dict[str, Any]] = []
            for layer, states in sorted(dataset.states_by_layer.items()):
                metrics = candidate_metrics(
                    states,
                    dataset.metadata,
                    pca_dim=args.pca_dim,
                    folds=args.discovery_cv_folds,
                    random_state=args.random_state,
                )
                row = {
                    "model_label": model,
                    "site_kind": site,
                    "layer": int(layer),
                    "pca_dim": int(args.pca_dim),
                    **metrics,
                }
                site_rows.append(row)
                all_candidates.append(row)
            site_frame = pd.DataFrame(site_rows)
            winner = select_one(site_frame).to_dict()
            selected_layer = int(winner["layer"])
            covariance = evaluate_covariance_geometry_layer(
                dataset.states_by_layer[selected_layer],
                dataset.metadata,
                CLASSES,
                pca_dim=args.pca_dim,
                random_state=args.random_state,
                relative_ridge=args.relative_ridge,
                discovery_cv_folds=args.discovery_cv_folds,
            )
            covariance.pop("metric_definitions", None)
            site_selected.append(
                {
                    **winner,
                    **{f"cov_{key}": value for key, value in covariance.items()},
                }
            )
            print(
                "[causal-geometry] selected "
                f"{model}/{site} L{selected_layer} "
                f"disc={float(winner['discovery_selection_score']):.3f} "
                f"conf-log={float(winner['confirmation_logistic_balanced_accuracy']):.3f} "
                f"conf-ncc={float(winner['confirmation_ncc_balanced_accuracy']):.3f} "
                f"snr={float(winner['confirmation_class_balanced_snr_db']):.2f}dB",
                flush=True,
            )
            del dataset
            gc.collect()

    candidates_frame = pd.DataFrame(all_candidates).sort_values(
        ["model_label", "site_kind", "layer"], kind="mergesort"
    )
    selected_frame = pd.DataFrame(site_selected).sort_values(
        ["model_label", "site_kind"], kind="mergesort"
    )
    model_winners = pd.DataFrame(
        [
            select_one(frame).to_dict()
            for _model, frame in selected_frame.groupby("model_label", sort=True)
        ]
    ).sort_values("model_label", kind="mergesort")

    legacy_rows: list[dict[str, Any]] = []
    for model in MODELS:
        old = legacy_row(args.legacy_dual_root, model)
        current = selected_frame.loc[
            selected_frame["model_label"].eq(model)
            & selected_frame["site_kind"].eq(PRIMARY_SITE)
        ].iloc[0]
        legacy_rows.append(
            {
                "model_label": model,
                "legacy_scope": "all parser-observed item_end events",
                "causal_scope": "exact causal primary progress commits",
                "legacy_layer": int(old["layer"]),
                "causal_layer": int(current["layer"]),
                "legacy_confirmation_logistic_balanced_accuracy": float(
                    old["confirmation_logistic_balanced_accuracy"]
                ),
                "causal_confirmation_logistic_balanced_accuracy": float(
                    current["confirmation_logistic_balanced_accuracy"]
                ),
                "causal_minus_legacy_logistic": float(
                    current["confirmation_logistic_balanced_accuracy"]
                    - old["confirmation_logistic_balanced_accuracy"]
                ),
                "legacy_confirmation_ncc_balanced_accuracy": float(
                    old["confirmation_ncc_balanced_accuracy"]
                ),
                "causal_confirmation_ncc_balanced_accuracy": float(
                    current["confirmation_ncc_balanced_accuracy"]
                ),
                "causal_minus_legacy_ncc": float(
                    current["confirmation_ncc_balanced_accuracy"]
                    - old["confirmation_ncc_balanced_accuracy"]
                ),
                "legacy_confirmation_snr_db": float(
                    old["confirmation_class_balanced_snr_db"]
                ),
                "causal_confirmation_snr_db": float(
                    current["confirmation_class_balanced_snr_db"]
                ),
                "causal_minus_legacy_snr_db": float(
                    current["confirmation_class_balanced_snr_db"]
                    - old["confirmation_class_balanced_snr_db"]
                ),
                "legacy_confirmation_noise_power": float(
                    old["confirmation_noise_power"]
                ),
                "causal_confirmation_noise_power": float(
                    current["confirmation_noise_power"]
                ),
            }
        )
    legacy_frame = pd.DataFrame(legacy_rows)

    output = args.output.resolve()
    paths = {
        "candidates": output / "site_layer_candidates.csv",
        "site_selected": output / "site_selected.csv",
        "model_winners": output / "model_site_winners.csv",
        "legacy_comparison": output / "legacy_vs_causal_item_end.csv",
        "audit": output / "audit.json",
    }
    atomic_csv(paths["candidates"], candidates_frame)
    atomic_csv(paths["site_selected"], selected_frame)
    atomic_csv(paths["model_winners"], model_winners)
    atomic_csv(paths["legacy_comparison"], legacy_frame)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS),
        "sites": list(sites),
        "primary_counter_site": PRIMARY_SITE,
        "primary_counter_role": "causal p0_item_end / archived item_end exact token",
        "causal_event_filter": dict(PRIMARY_PROGRESS_FILTER),
        "registered_split": {
            "discovery": list(range(1234, 1254)),
            "confirmation": list(range(1254, 1264)),
        },
        "classes": list(CLASSES),
        "pca_dim": int(args.pca_dim),
        "pca_whiten": True,
        "discovery_cv_folds": int(args.discovery_cv_folds),
        "selection_rule": (
            "for every archived token site, choose decoder layer by the mean of "
            "seed-grouped discovery OOF Logistic and NCC balanced accuracy; choose "
            "the exploratory site winner by the same discovery-only score; never "
            "rank sites or layers on confirmation"
        ),
        "interpretation_boundary": (
            "item_end reuses an archived state only after exact equality with the "
            "causal commit token. Changes versus the legacy panel combine compiler "
            "event/cohort alignment with geometry; they are not a new forward pass. "
            "post_marker was not archived and is not inferred."
        ),
        "paired_site_metadata_sha256": model_key_sha,
        "support": support_audit,
        "loader_audits": loader_audits,
        "inputs": {
            str(args.event_registry.resolve()): sha256(args.event_registry.resolve()),
            **{
                str((args.native_running_root / model / "capture_index.jsonl").resolve()): sha256(
                    (args.native_running_root / model / "capture_index.jsonl").resolve()
                )
                for model in MODELS
            },
        },
        "outputs": {
            str(path.resolve()): sha256(path)
            for name, path in paths.items()
            if name != "audit"
        },
    }
    atomic_json(paths["audit"], audit)
    print(legacy_frame.to_string(index=False), flush=True)
    print(model_winners[["model_label", "site_kind", "layer"]].to_string(index=False))


if __name__ == "__main__":
    main()
