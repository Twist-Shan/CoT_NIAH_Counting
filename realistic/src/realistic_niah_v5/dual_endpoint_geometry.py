"""CPU analysis for fixed-end running-index and final-count geometry.

The two modes are never forced onto a common decoder layer.  Running-index
sites are fixed a priori to the last token of the prompt evidence span and the
last token of the parsed thinking item.  Layers are ranked separately inside
each mode using discovery-only grouped cross-validation, and the frozen winner
is then evaluated on held-out seeds.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from realistic_niah_v5.cross_mode_geometry import (
    CLASSES,
    ModeDataset,
    load_native_thinking_capture,
    load_non_thinking_capture,
)
from realistic_niah_v5.trace_stratified_geometry import (
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)


SCHEMA_VERSION = "realistic_niah_dual_endpoint_geometry_v5_fixed_end_all_counts"
RUNNING_NON_THINKING_SITES = ("span_end",)
RUNNING_NATIVE_PRIMARY_SITES = ("item_end",)
TRACE_GROUP_MEMBERS = {
    "all_traces": None,
    "explicit_count_marker": ("indexed", "ordinal", "inline_count"),
    # Backward-compatible narrow slice retained for comparison with the first
    # N=10 report; the broader group above is the current primary explicit cue.
    "explicit_ordinal": ("indexed", "ordinal"),
    "implicit_or_invariant_progress": (
        "bullet",
        "audit_sentence",
        "completion_recap",
        "evidence_sequence",
    ),
    "non_explicit_progress": (
        "bullet",
        "audit_sentence",
        "completion_recap",
        "evidence_sequence",
    ),
}
PCA_WHITEN = True


@dataclass(frozen=True)
class GroupEligibility:
    analysis_group: str
    status: str
    labels: tuple[int, ...]
    discovery_seed_count: int
    confirmation_seed_count: int
    discovery_support: dict[int, int]
    confirmation_support: dict[int, int]
    reason: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def load_non_thinking_final_count(
    capture_index: str | Path,
    *,
    design_variant: str = "v4.4",
) -> ModeDataset:
    """Load prompt-final ``Total:`` states with gold count as the class label."""

    index_path = Path(capture_index)
    rows = [
        row
        for row in _read_jsonl(index_path)
        if str(row.get("design_variant")) == design_variant
    ]
    rows.sort(key=lambda row: (int(row["seed"]), int(row["count"])))
    if not rows:
        raise ValueError(f"No {design_variant} rows in {index_path}")
    first_path = index_path.parent / str(rows[0]["shard_path"])
    with np.load(first_path, allow_pickle=False) as archive:
        layer_indices = archive["layer_indices"].astype(int)
        first_states = np.asarray(archive["query_states"])
        hidden_size = int(first_states.shape[-1])
    states = {
        int(layer): np.empty((len(rows), hidden_size), dtype=np.float16)
        for layer in layer_indices
    }
    metadata_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        shard = index_path.parent / str(row["shard_path"])
        with np.load(shard, allow_pickle=False) as archive:
            layers = archive["layer_indices"].astype(int)
            values = np.asarray(archive["query_states"])
            if not np.array_equal(layers, layer_indices):
                raise ValueError(f"Layer mismatch in {shard}")
            if values.shape != (len(layer_indices), hidden_size):
                raise ValueError(f"Unexpected query_states shape {values.shape} in {shard}")
            for layer_axis, layer in enumerate(layer_indices):
                states[int(layer)][row_index] = values[layer_axis]
        metadata_rows.append(
            {
                "split": str(row["split"]),
                "seed": int(row["seed"]),
                "occurrence": int(row["count"]),
                "gold_count": int(row["count"]),
                "stimulus_id": str(row["stimulus_id"]),
                "token_site": str(row.get("position", "prompt_final_total_query")),
                "design_variant": str(row["design_variant"]),
            }
        )
    dataset = ModeDataset(
        mode="non_thinking",
        model_label=str(rows[0]["model_label"]),
        metadata=pd.DataFrame(metadata_rows),
        states_by_layer=states,
    )
    dataset.validate()
    return dataset


def load_native_thinking_final_count(capture_index: str | Path) -> ModeDataset:
    """Load the token immediately before the native-thinking numeric answer."""

    index_path = Path(capture_index)
    descriptors: list[tuple[dict[str, Any], int]] = []
    for row in _read_jsonl(index_path):
        manifest_path = index_path.parent / str(row["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matching = [
            site_index
            for site_index, site in enumerate(manifest.get("site_rows", []))
            if str(site.get("site_kind")) == "answer_query_v3"
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Expected one answer_query_v3 site for {row.get('request_id')}; "
                f"found {len(matching)}"
            )
        descriptors.append((row, matching[0]))
    descriptors.sort(key=lambda item: (int(item[0]["seed"]), int(item[0]["gold_count"])))
    if not descriptors:
        raise ValueError(f"No final-count rows in {index_path}")
    first_row, _ = descriptors[0]
    first_path = index_path.parent / str(first_row["states_path"])
    with np.load(first_path, allow_pickle=False) as archive:
        layer_indices = archive["layer_indices"].astype(int)
        hidden_size = int(archive["site_states"].shape[-1])
    states = {
        int(layer): np.empty((len(descriptors), hidden_size), dtype=np.float16)
        for layer in layer_indices
    }
    metadata_rows: list[dict[str, Any]] = []
    for row_index, (row, site_index) in enumerate(descriptors):
        shard = index_path.parent / str(row["states_path"])
        with np.load(shard, allow_pickle=False) as archive:
            layers = archive["layer_indices"].astype(int)
            values = np.asarray(archive["site_states"])
            if not np.array_equal(layers, layer_indices):
                raise ValueError(f"Layer mismatch in {shard}")
            if values.ndim != 3 or values.shape[1:] != (
                len(layer_indices),
                hidden_size,
            ):
                raise ValueError(f"Unexpected site_states shape {values.shape} in {shard}")
            for layer_axis, layer in enumerate(layer_indices):
                states[int(layer)][row_index] = values[site_index, layer_axis]
        metadata_rows.append(
            {
                "split": str(row["split"]),
                "seed": int(row["seed"]),
                "occurrence": int(row["gold_count"]),
                "gold_count": int(row["gold_count"]),
                "stimulus_id": str(row["stimulus_id"]),
                "token_site": "answer_query_v3",
                "marker_kind": row.get("marker_kind"),
                "trace_category": row.get("trace_category"),
                "exact_count": bool(row.get("exact_count")),
            }
        )
    dataset = ModeDataset(
        mode="native_thinking",
        model_label=str(first_row["model_label"]),
        metadata=pd.DataFrame(metadata_rows),
        states_by_layer=states,
    )
    dataset.validate()
    return dataset


def relabel_seed_panel(
    dataset: ModeDataset,
    *,
    discovery_seeds: Sequence[int],
    confirmation_seeds: Sequence[int],
) -> ModeDataset:
    discovery_set = set(map(int, discovery_seeds))
    confirmation_set = set(map(int, confirmation_seeds))
    overlap = sorted(discovery_set & confirmation_set)
    if overlap:
        raise ValueError(f"Discovery/confirmation seeds overlap: {overlap}")
    retained = dataset.metadata["seed"].astype(int).isin(
        discovery_set | confirmation_set
    ).to_numpy()
    selected = dataset.metadata.loc[retained].copy()
    selected["source_split"] = selected["split"].astype(str)
    selected["split"] = np.where(
        selected["seed"].astype(int).isin(discovery_set),
        "discovery",
        "confirmation",
    )
    selected["_old_index"] = np.flatnonzero(retained)
    selected = selected.sort_values(["split", "seed", "occurrence"], kind="mergesort")
    old_index = selected.pop("_old_index").to_numpy(dtype=int)
    result = ModeDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=selected.reset_index(drop=True),
        states_by_layer={
            layer: values[old_index] for layer, values in dataset.states_by_layer.items()
        },
    )
    result.validate()
    observed_discovery = set(
        result.metadata.loc[result.metadata["split"].eq("discovery"), "seed"].astype(int)
    )
    observed_confirmation = set(
        result.metadata.loc[
            result.metadata["split"].eq("confirmation"), "seed"
        ].astype(int)
    )
    if observed_discovery != discovery_set:
        raise ValueError(
            "Missing requested discovery seeds: "
            f"{sorted(discovery_set - observed_discovery)}"
        )
    if observed_confirmation != confirmation_set:
        raise ValueError(
            "Missing requested confirmation seeds: "
            f"{sorted(confirmation_set - observed_confirmation)}"
        )
    return result


def _trace_group_mask(metadata: pd.DataFrame, analysis_group: str) -> np.ndarray:
    if analysis_group not in TRACE_GROUP_MEMBERS:
        raise ValueError(f"Unknown trace group: {analysis_group}")
    members = TRACE_GROUP_MEMBERS[analysis_group]
    if members is None:
        return np.ones(len(metadata), dtype=bool)
    return metadata["marker_kind"].astype(str).isin(members).to_numpy()


def determine_group_eligibility(
    metadata: pd.DataFrame,
    analysis_group: str,
    *,
    min_discovery_support: int = 3,
    min_confirmation_support: int = 2,
    min_classes: int = 5,
) -> GroupEligibility:
    grouped = metadata.loc[_trace_group_mask(metadata, analysis_group)].copy()
    discovery = grouped.loc[grouped["split"].astype(str).eq("discovery")]
    confirmation = grouped.loc[grouped["split"].astype(str).eq("confirmation")]
    discovery_support = {
        label: int((discovery["occurrence"].astype(int) == label).sum())
        for label in CLASSES
    }
    confirmation_support = {
        label: int((confirmation["occurrence"].astype(int) == label).sum())
        for label in CLASSES
    }
    labels = tuple(
        label
        for label in CLASSES
        if discovery_support[label] >= min_discovery_support
        and confirmation_support[label] >= min_confirmation_support
    )
    if len(labels) >= min_classes:
        status = "evaluable"
        reason = (
            f"retained {len(labels)} labels with discovery support >= "
            f"{min_discovery_support} and confirmation support >= "
            f"{min_confirmation_support}"
        )
    else:
        status = "not_evaluable"
        reason = (
            f"only {len(labels)} labels passed the support gate; "
            f"requires at least {min_classes}"
        )
    return GroupEligibility(
        analysis_group=analysis_group,
        status=status,
        labels=labels,
        discovery_seed_count=int(discovery["seed"].nunique()),
        confirmation_seed_count=int(confirmation["seed"].nunique()),
        discovery_support=discovery_support,
        confirmation_support=confirmation_support,
        reason=reason,
    )


def _subset_group(
    dataset: ModeDataset,
    analysis_group: str,
    labels: Sequence[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = _trace_group_mask(dataset.metadata, analysis_group)
    mask &= dataset.metadata["occurrence"].astype(int).isin(labels).to_numpy()
    return dataset.metadata.loc[mask].reset_index(drop=True), mask


def _candidate_row(
    *,
    endpoint: str,
    dataset: ModeDataset,
    analysis_group: str,
    selector: str,
    token_site: str,
    layer: int,
    states: np.ndarray,
    metadata: pd.DataFrame,
    labels: Sequence[int],
    pca_dim: int,
    cv_folds: int,
    random_state: int,
) -> dict[str, Any]:
    metrics = grouped_discovery_cv_metrics(
        states,
        metadata,
        labels,
        pca_dim=pca_dim,
        random_state=random_state,
        folds=cv_folds,
        pca_whiten=PCA_WHITEN,
    )
    heldout = confirmation_metrics(
        states,
        metadata,
        labels,
        pca_dim=pca_dim,
        random_state=random_state,
        pca_whiten=PCA_WHITEN,
    )
    return {
        "endpoint": endpoint,
        "model_label": dataset.model_label,
        "mode": dataset.mode,
        "analysis_group": analysis_group,
        "selector": selector,
        "token_site": token_site,
        "layer": int(layer),
        "retained_labels": " ".join(map(str, labels)),
        "retained_class_count": len(labels),
        **metrics,
        **heldout,
    }


def select_discovery_winners(
    candidates: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (
        "endpoint",
        "model_label",
        "mode",
        "analysis_group",
        "selector",
    ),
) -> pd.DataFrame:
    if candidates.empty:
        raise ValueError("Cannot select from an empty candidate table")
    rows = []
    for _, frame in candidates.groupby(list(group_columns), sort=False, dropna=False):
        selected = frame.sort_values(
            [
                "discovery_selection_score",
                "discovery_oof_ncc_balanced_accuracy",
                "discovery_oof_logistic_balanced_accuracy",
                "layer",
                "token_site",
            ],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        ).iloc[0]
        rows.append(selected.to_dict())
    return pd.DataFrame(rows)


def _heldout_metrics(
    states: np.ndarray,
    metadata: pd.DataFrame,
    labels: Sequence[int],
    *,
    pca_dim: int,
    random_state: int,
) -> dict[str, Any]:
    return confirmation_metrics(
        states,
        metadata,
        labels,
        pca_dim=pca_dim,
        random_state=random_state,
        pca_whiten=PCA_WHITEN,
    )


def _running_index_analysis(
    non_thinking_index: Path,
    native_thinking_index: Path,
    *,
    pca_dim: int,
    cv_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate_rows: list[dict[str, Any]] = []
    non_thinking_reference: ModeDataset | None = None
    for pooling in RUNNING_NON_THINKING_SITES:
        dataset = load_non_thinking_capture(
            non_thinking_index,
            design_variant="v4.4",
            pooling=pooling,
        )
        if non_thinking_reference is None:
            non_thinking_reference = dataset
        for layer, states in sorted(dataset.states_by_layer.items()):
            candidate_rows.append(
                _candidate_row(
                    endpoint="running_index",
                    dataset=dataset,
                    analysis_group="all_traces",
                    selector="fixed_span_end_layer_search",
                    token_site=pooling,
                    layer=layer,
                    states=states,
                    metadata=dataset.metadata,
                    labels=CLASSES,
                    pca_dim=pca_dim,
                    cv_folds=cv_folds,
                    random_state=random_state,
                )
            )

    item_dataset = load_native_thinking_capture(
        native_thinking_index,
        site_kind="item_end",
        cohort="parser_hit",
    )
    if non_thinking_reference is None:
        raise RuntimeError("Non-thinking running-index candidates were not loaded")
    if non_thinking_reference.model_label != item_dataset.model_label:
        raise ValueError(
            "Running-index inputs use different model labels: "
            f"{non_thinking_reference.model_label!r} versus "
            f"{item_dataset.model_label!r}"
        )

    def seed_panel(frame: pd.DataFrame) -> dict[str, list[int]]:
        return {
            split: sorted(
                frame.loc[frame["split"].astype(str).eq(split), "seed"]
                .astype(int)
                .unique()
                .tolist()
            )
            for split in ("discovery", "confirmation")
        }

    def trajectory_panel(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
        required = {"split", "seed", "gold_count"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Running-index metadata lacks {missing}")
        return {
            (str(split), int(seed), int(gold_count))
            for split, seed, gold_count in frame[
                ["split", "seed", "gold_count"]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }

    non_thinking_panel = seed_panel(non_thinking_reference.metadata)
    native_panel = seed_panel(item_dataset.metadata)
    if non_thinking_panel != native_panel:
        raise ValueError(
            "Running-index inputs do not share the same seed panel: "
            f"non_thinking={non_thinking_panel}, native_thinking={native_panel}"
        )
    non_thinking_trajectories = trajectory_panel(non_thinking_reference.metadata)
    native_trajectories = trajectory_panel(item_dataset.metadata)
    if non_thinking_trajectories != native_trajectories:
        raise ValueError(
            "Running-index inputs do not share the same 10-count x 30-seed "
            "trajectory panel: "
            f"missing_native={sorted(non_thinking_trajectories - native_trajectories)}, "
            f"extra_native={sorted(native_trajectories - non_thinking_trajectories)}"
        )
    # The report's core estimand is the pooled registered panel.  Marker-format
    # strata are parser diagnostics with small, unequal supports; searching a
    # separate site x layer winner inside every stratum adds multiplicity but
    # does not answer the cross-mode representation question.
    eligibility = [
        determine_group_eligibility(item_dataset.metadata, "all_traces")
    ]
    eligible = {
        item.analysis_group: item
        for item in eligibility
        if item.status == "evaluable"
    }
    for site in RUNNING_NATIVE_PRIMARY_SITES:
        dataset = load_native_thinking_capture(
            native_thinking_index,
            site_kind=site,
            cohort="parser_hit",
        )
        for analysis_group, item in eligible.items():
            metadata, mask = _subset_group(dataset, analysis_group, item.labels)
            for layer, states in sorted(dataset.states_by_layer.items()):
                candidate_rows.append(
                    _candidate_row(
                        endpoint="running_index",
                        dataset=dataset,
                        analysis_group=analysis_group,
                        selector="fixed_item_end_layer_search",
                        token_site=site,
                        layer=layer,
                        states=states[mask],
                        metadata=metadata,
                        labels=item.labels,
                        pca_dim=pca_dim,
                        cv_folds=cv_folds,
                        random_state=random_state,
                    )
                )

    candidates = pd.DataFrame(candidate_rows)
    winners = select_discovery_winners(candidates)
    selected_rows: list[dict[str, Any]] = []

    non_thinking_winners = winners.loc[winners["mode"].eq("non_thinking")]
    for pooling, frame in non_thinking_winners.groupby("token_site", sort=False):
        dataset = load_non_thinking_capture(
            non_thinking_index,
            design_variant="v4.4",
            pooling=str(pooling),
        )
        for winner in frame.to_dict(orient="records"):
            labels = tuple(map(int, str(winner["retained_labels"]).split()))
            metrics = _heldout_metrics(
                dataset.states_by_layer[int(winner["layer"])],
                dataset.metadata,
                labels,
                pca_dim=pca_dim,
                random_state=random_state,
            )
            selected_rows.append(
                {
                    **winner,
                    "evaluation_split_role": "original_confirmation",
                    **metrics,
                }
            )

    native_winners = winners.loc[winners["mode"].eq("native_thinking")]
    for site, frame in native_winners.groupby("token_site", sort=False):
        dataset = load_native_thinking_capture(
            native_thinking_index,
            site_kind=str(site),
            cohort="parser_hit",
        )
        for winner in frame.to_dict(orient="records"):
            group = str(winner["analysis_group"])
            labels = tuple(map(int, str(winner["retained_labels"]).split()))
            metadata, mask = _subset_group(dataset, group, labels)
            metrics = _heldout_metrics(
                dataset.states_by_layer[int(winner["layer"])][mask],
                metadata,
                labels,
                pca_dim=pca_dim,
                random_state=random_state,
            )
            selected_rows.append(
                {
                    **winner,
                    "evaluation_split_role": "original_confirmation",
                    **metrics,
                }
            )

    eligibility_rows = [
        {
            "model_label": item_dataset.model_label,
            "analysis_group": item.analysis_group,
            "status": item.status,
            "retained_labels": " ".join(map(str, item.labels)),
            "retained_class_count": len(item.labels),
            "discovery_seed_count": item.discovery_seed_count,
            "confirmation_seed_count": item.confirmation_seed_count,
            "discovery_support": json.dumps(item.discovery_support, sort_keys=True),
            "confirmation_support": json.dumps(
                item.confirmation_support, sort_keys=True
            ),
            "reason": item.reason,
        }
        for item in eligibility
    ]
    audit = {
        "non_thinking_layers": sorted(non_thinking_reference.states_by_layer),
        "native_thinking_layers": sorted(item_dataset.states_by_layer),
        "non_thinking_seed_panel": non_thinking_panel,
        "native_thinking_seed_panel": native_panel,
        "non_thinking_trajectory_count": len(non_thinking_trajectories),
        "native_thinking_trajectory_count": len(native_trajectories),
    }
    return (
        candidates,
        pd.DataFrame(selected_rows),
        pd.DataFrame(eligibility_rows),
        audit,
    )


def _final_count_analysis(
    non_thinking_index: Path,
    native_thinking_index: Path,
    *,
    pca_dim: int,
    cv_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_non_thinking = load_non_thinking_final_count(non_thinking_index)
    raw_native = load_native_thinking_final_count(native_thinking_index)
    if raw_non_thinking.model_label != raw_native.model_label:
        raise ValueError(
            "Final-count inputs use different model labels: "
            f"{raw_non_thinking.model_label!r} versus {raw_native.model_label!r}"
        )
    datasets = [raw_non_thinking, raw_native]

    def trajectory_keys(dataset: ModeDataset) -> set[tuple[str, int, int]]:
        return {
            (str(split), int(seed), int(gold_count))
            for split, seed, gold_count in dataset.metadata[
                ["split", "seed", "gold_count"]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }

    non_thinking_keys = trajectory_keys(raw_non_thinking)
    native_keys = trajectory_keys(raw_native)
    if non_thinking_keys != native_keys:
        raise ValueError(
            "Final-count inputs do not share the same registered trajectory "
            "panel: "
            f"missing_native={sorted(non_thinking_keys - native_keys)}, "
            f"extra_native={sorted(native_keys - non_thinking_keys)}"
        )
    candidate_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        token_sites = sorted(set(dataset.metadata["token_site"].astype(str)))
        if len(token_sites) != 1:
            raise ValueError(f"Final-count dataset has token sites {token_sites}")
        for layer, states in sorted(dataset.states_by_layer.items()):
            candidate_rows.append(
                _candidate_row(
                    endpoint="final_count",
                    dataset=dataset,
                    analysis_group="all_counts",
                    selector="independent_layer_search",
                    token_site=token_sites[0],
                    layer=layer,
                    states=states,
                    metadata=dataset.metadata,
                    labels=CLASSES,
                    pca_dim=pca_dim,
                    cv_folds=cv_folds,
                    random_state=random_state,
                )
            )
    candidates = pd.DataFrame(candidate_rows)
    winners = select_discovery_winners(candidates)
    by_mode = {dataset.mode: dataset for dataset in datasets}
    selected_rows: list[dict[str, Any]] = []
    for winner in winners.to_dict(orient="records"):
        mode = str(winner["mode"])
        dataset = by_mode[mode]
        layer = int(winner["layer"])
        heldout_metrics = _heldout_metrics(
            dataset.states_by_layer[layer],
            dataset.metadata,
            CLASSES,
            pca_dim=pca_dim,
            random_state=random_state,
        )
        selected = {
            **winner,
            "evaluation_split_role": "registered_confirmation",
            **heldout_metrics,
        }
        selected_rows.append(selected)

    support_audit: dict[str, Any] = {}
    for dataset in datasets:
        support_audit[dataset.mode] = {
            split: {
                str(label): int(
                    (
                        dataset.metadata.loc[
                            dataset.metadata["split"].astype(str).eq(split),
                            "occurrence",
                        ].astype(int)
                        == label
                    ).sum()
                )
                for label in CLASSES
            }
            for split in ("discovery", "confirmation")
        }
    audit = {
        "registered_seed_panel": {
            split: sorted(
                raw_non_thinking.metadata.loc[
                    raw_non_thinking.metadata["split"].astype(str).eq(split),
                    "seed",
                ]
                .astype(int)
                .unique()
                .tolist()
            )
            for split in ("discovery", "confirmation")
        },
        "registered_trajectory_counts": {
            split: sum(key[0] == split for key in non_thinking_keys)
            for split in ("discovery", "confirmation")
        },
        "support": support_audit,
        "non_thinking_layers": sorted(raw_non_thinking.states_by_layer),
        "native_thinking_layers": sorted(raw_native.states_by_layer),
    }
    return candidates, pd.DataFrame(selected_rows), audit


def analyze_dual_endpoint_geometry(
    *,
    non_thinking_running_index: str | Path,
    native_thinking_running_index: str | Path,
    non_thinking_final_count: str | Path,
    native_thinking_final_count: str | Path,
    output_dir: str | Path,
    pca_dim: int = 16,
    cv_folds: int = 5,
    random_state: int = 0,
    command: str | None = None,
) -> dict[str, Path]:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    phase_started = time.perf_counter()
    running_candidates, running_selected, eligibility, running_audit = (
        _running_index_analysis(
            Path(non_thinking_running_index),
            Path(native_thinking_running_index),
            pca_dim=pca_dim,
            cv_folds=cv_folds,
            random_state=random_state,
        )
    )
    timings["running_index_seconds"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    final_candidates, final_selected, final_audit = _final_count_analysis(
        Path(non_thinking_final_count),
        Path(native_thinking_final_count),
        pca_dim=pca_dim,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    timings["final_count_seconds"] = time.perf_counter() - phase_started
    timings["total_seconds"] = time.perf_counter() - started

    output = Path(output_dir)
    paths = {
        "running_candidates": output / "running_index_candidate_metrics.csv",
        "running_selected": output / "running_index_selected.csv",
        "running_eligibility": output / "running_index_group_eligibility.csv",
        "final_candidates": output / "final_count_candidate_metrics.csv",
        "final_selected": output / "final_count_selected.csv",
        "audit": output / "dual_endpoint_geometry_audit.json",
        "runtime": output / "runtime_log.json",
    }
    _atomic_csv(paths["running_candidates"], running_candidates)
    _atomic_csv(paths["running_selected"], running_selected)
    _atomic_csv(paths["running_eligibility"], eligibility)
    _atomic_csv(paths["final_candidates"], final_candidates)
    _atomic_csv(paths["final_selected"], final_selected)
    _atomic_json(
        paths["audit"],
        {
            "schema_version": SCHEMA_VERSION,
            "model_label": str(running_candidates.iloc[0]["model_label"]),
            "estimands": {
                "running_index": (
                    "decodability of k=1..10 from non-thinking prompt occurrence "
                    "states versus native-thinking parsed thinking-trace item states"
                ),
                "final_count": (
                    "decodability of gold N=1..10 at the prompt-final Total: query "
                    "state versus the thinking-trace token immediately before the "
                    "numeric final answer"
                ),
            },
            "independent_layer_rule": (
                "running token sites are fixed to span_end/item_end; decoder "
                "layers are selected separately within each mode and layer "
                "numbers are never matched across modes"
            ),
            "selection_rule": (
                "with token site fixed, maximize the mean of discovery grouped-CV "
                "logistic and nearest-centroid balanced accuracy over layers; "
                "tie-break by NCC, logistic, then earlier within-model layer"
            ),
            "confirmation_rule": (
                "all-layer held-out curves are saved for transparent diagnostics, "
                "but the programmed winner reads discovery columns only; the "
                "reported selected row is therefore discovery-frozen"
            ),
            "pca_dim_requested": int(pca_dim),
            "pca_whiten": PCA_WHITEN,
            "cv_folds_requested": int(cv_folds),
            "random_state": int(random_state),
            "preprocessing": (
                "StandardScaler and whitened PCA refit inside every grouped "
                "discovery fold; the held-out projection is fit on all selection "
                "rows"
            ),
            "metrics": (
                "multiclass balanced accuracy for class-weighted logistic regression "
                "and nearest centroid; class-balanced centroid signal/noise ratio in "
                "the frozen discovery projection"
            ),
            "running_index_token_sites": {
                "non_thinking_primary": {
                    "span_end": "last token of each prompt evidence span",
                },
                "native_thinking_primary": {
                    "item_end": "last token of a completed parsed trace item",
                },
                "excluded_from_primary_search": (
                    "span_mean and all alternative native token sites are omitted; "
                    "the primary comparison uses one boundary-token state per "
                    "observed occurrence in both modes"
                ),
            },
            "running_index_analysis_unit": (
                "all 10 registered counts x 30 seeds = 300 trajectories per model; "
                "trajectory i contributes only its observed labels 1..M_i"
            ),
            "trace_groups": {
                "evaluated": ["all_traces"],
                "marker_strata_role": (
                    "descriptive appendix only; no stratum-specific site/layer "
                    "winner search in the primary analysis"
                ),
            },
            "final_count_token_sites": {
                "non_thinking": (
                    "prompt_final_total_query: final prompt Total: token immediately "
                    "before answer generation"
                ),
                "native_thinking": (
                    "answer_query_v3: last literal thinking-trace token immediately "
                    "before the numeric final answer"
                ),
            },
            "final_count_split_caveat": (
                "both modes use the registered 20-seed discovery / 10-seed "
                "confirmation split over all ten gold-count conditions; each "
                "confirmation panel therefore contains 100 trajectories"
            ),
            "inputs": {
                "non_thinking_running_index": str(
                    Path(non_thinking_running_index).resolve()
                ),
                "native_thinking_running_index": str(
                    Path(native_thinking_running_index).resolve()
                ),
                "non_thinking_final_count": str(
                    Path(non_thinking_final_count).resolve()
                ),
                "native_thinking_final_count": str(
                    Path(native_thinking_final_count).resolve()
                ),
            },
            "running_index_audit": running_audit,
            "final_count_audit": final_audit,
        },
    )
    _atomic_json(
        paths["runtime"],
        {
            "schema_version": SCHEMA_VERSION,
            "command": command,
            "device": "CPU",
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": {
                package: _package_version(package)
                for package in ("numpy", "pandas", "scikit-learn")
            },
            "timings_seconds": timings,
        },
    )
    return paths
