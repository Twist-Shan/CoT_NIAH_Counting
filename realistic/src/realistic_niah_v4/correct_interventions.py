from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .causal_v2 import CAUSAL_V2_CANONICAL_PAIRS


PATCH_TARGET_ACCURACY_DEFINITION = "patched_count_equals_donor_gold_count"
PATCH_CLUSTER_TARGET = 5
CORRECT_ABLATION_CLUSTER_TARGET = 10
CORRECT_ABLATION_COUNTS = (1, 2, 3, 4, 5)
ABLATION_TOP_NS = tuple(range(1, 33))
PARALLEL_WORKER_COUNT = 4
PARALLEL_ASSIGNMENT_METHOD = "sorted_round_robin_v1"
PATCH_PAIR_KEY_COLUMNS = (
    "model_label",
    "seed",
    "receiver_count",
    "donor_count",
    "k",
    "target_direction",
)
ABLATION_STIMULUS_KEY_COLUMNS = (
    "model_label",
    "stimulus_id",
    "seed",
    "gold_count",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _normalized_key_records(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    integer_columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Return unique, deterministically sorted JSON-safe work identities."""

    missing = sorted(set(key_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Parallel work identities are missing columns: {missing}")
    keys = frame[list(key_columns)].drop_duplicates().copy()
    if len(keys) != len(frame):
        raise ValueError("Parallel work identities contain duplicate key rows")
    integer_set = {str(value) for value in integer_columns}
    for column in key_columns:
        if column in integer_set:
            keys[column] = pd.to_numeric(keys[column], errors="raise").astype(int)
        else:
            keys[column] = keys[column].astype(str)
    keys = keys.sort_values(list(key_columns), kind="stable").reset_index(drop=True)
    return [
        {
            column: (
                int(value) if column in integer_set else str(value)
            )
            for column, value in zip(key_columns, row)
        }
        for row in keys.itertuples(index=False, name=None)
    ]


def build_parallel_work_plan(
    *,
    model_label: str,
    added_pairs: pd.DataFrame,
    ablation_stimuli: pd.DataFrame,
    worker_count: int = PARALLEL_WORKER_COUNT,
) -> dict[str, Any]:
    """Freeze a disjoint, exhaustive round-robin partition for GPU workers.

    Assignment depends only on sorted registered work identities.  It therefore
    cannot inspect an intervention outcome and is invariant to process timing.
    """

    if int(worker_count) < 1:
        raise ValueError("Parallel worker_count must be positive")
    model = str(model_label)
    for name, frame in (("patch", added_pairs), ("ablation", ablation_stimuli)):
        observed_models = set(frame["model_label"].astype(str))
        if observed_models and observed_models != {model}:
            raise ValueError(
                f"{name} work contains models {sorted(observed_models)}; expected {model}"
            )
    patch_records = _normalized_key_records(
        added_pairs,
        key_columns=PATCH_PAIR_KEY_COLUMNS,
        integer_columns=("seed", "receiver_count", "donor_count", "k"),
    )
    ablation_records = _normalized_key_records(
        ablation_stimuli,
        key_columns=ABLATION_STIMULUS_KEY_COLUMNS,
        integer_columns=("seed", "gold_count"),
    )
    for index, record in enumerate(patch_records):
        record["worker_index"] = int(index % int(worker_count))
    for index, record in enumerate(ablation_records):
        record["worker_index"] = int(index % int(worker_count))
    return {
        "schema_version": "realistic_niah_v4_4_parallel_work_plan_v1",
        "model_label": model,
        "worker_count": int(worker_count),
        "assignment_method": PARALLEL_ASSIGNMENT_METHOD,
        "patch_pair_key_columns": list(PATCH_PAIR_KEY_COLUMNS),
        "ablation_stimulus_key_columns": list(ABLATION_STIMULUS_KEY_COLUMNS),
        "patch_pairs": patch_records,
        "ablation_stimuli": ablation_records,
        "patch_pair_count": int(len(patch_records)),
        "ablation_stimulus_count": int(len(ablation_records)),
        "worker_loads": [
            {
                "worker_index": int(worker_index),
                "patch_pairs": int(
                    sum(
                        int(row["worker_index"]) == worker_index
                        for row in patch_records
                    )
                ),
                "ablation_stimuli": int(
                    sum(
                        int(row["worker_index"]) == worker_index
                        for row in ablation_records
                    )
                ),
            }
            for worker_index in range(int(worker_count))
        ],
    }


def parallel_plan_records(
    plan: dict[str, Any],
    *,
    work_kind: str,
    worker_index: int | None = None,
) -> list[dict[str, Any]]:
    """Read and validate one work-kind slice from a frozen parallel plan."""

    if plan.get("schema_version") != (
        "realistic_niah_v4_4_parallel_work_plan_v1"
    ):
        raise ValueError("Unexpected parallel work-plan schema")
    if plan.get("assignment_method") != PARALLEL_ASSIGNMENT_METHOD:
        raise ValueError("Unexpected parallel assignment method")
    worker_count = int(plan.get("worker_count", 0))
    if worker_count < 1:
        raise ValueError("Parallel work plan has invalid worker_count")
    if work_kind not in {"patch_pairs", "ablation_stimuli"}:
        raise ValueError(f"Unknown parallel work kind: {work_kind}")
    records = list(plan.get(work_kind, []))
    expected = int(
        plan[
            "patch_pair_count"
            if work_kind == "patch_pairs"
            else "ablation_stimulus_count"
        ]
    )
    if len(records) != expected:
        raise ValueError(f"Parallel plan {work_kind} count does not match manifest")
    assignments = [int(row["worker_index"]) for row in records]
    if any(value < 0 or value >= worker_count for value in assignments):
        raise ValueError(f"Parallel plan {work_kind} has out-of-range worker index")
    if worker_index is None:
        return records
    index = int(worker_index)
    if index < 0 or index >= worker_count:
        raise ValueError(f"worker_index={index} is outside [0, {worker_count})")
    return [row for row in records if int(row["worker_index"]) == index]


def select_parallel_work_frame(
    frame: pd.DataFrame,
    *,
    plan: dict[str, Any],
    work_kind: str,
    worker_index: int,
) -> pd.DataFrame:
    """Select exactly the rows assigned to one worker, preserving full columns."""

    key_columns = (
        PATCH_PAIR_KEY_COLUMNS
        if work_kind == "patch_pairs"
        else ABLATION_STIMULUS_KEY_COLUMNS
    )
    records = parallel_plan_records(
        plan, work_kind=work_kind, worker_index=worker_index
    )
    keys = {
        tuple(str(record[column]) for column in key_columns)
        for record in records
    }
    selected = frame[
        frame[list(key_columns)]
        .astype(str)
        .apply(tuple, axis=1)
        .isin(keys)
    ].copy()
    observed = {
        tuple(str(value) for value in row)
        for row in selected[list(key_columns)].itertuples(index=False, name=None)
    }
    if observed != keys or len(selected) != len(keys):
        raise RuntimeError(
            f"Worker {worker_index} {work_kind} selection is incomplete or duplicated"
        )
    return selected.reset_index(drop=True)


def clean_correct_baselines(
    labels: pd.DataFrame,
    *,
    counts: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Select format-valid clean generations that exactly match the gold count."""

    required = {
        "stimulus_id",
        "model_label",
        "seed",
        "gold_count",
        "outcome_group",
        "format_valid",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Baseline labels are missing columns: {missing}")
    selected = labels[
        labels["outcome_group"].astype(str).eq("correct")
        & _as_bool(labels["format_valid"])
    ].copy()
    if counts is not None:
        selected = selected[
            pd.to_numeric(selected["gold_count"], errors="raise")
            .astype(int)
            .isin({int(value) for value in counts})
        ].copy()
    return selected.reset_index(drop=True)


def eligible_directed_pairs(
    labels: pd.DataFrame,
    *,
    canonical_pairs: Sequence[tuple[int, int]] = CAUSAL_V2_CANONICAL_PAIRS,
) -> pd.DataFrame:
    """Return same-seed directed pairs for which receiver and donor are correct."""

    correct = clean_correct_baselines(labels)
    keys = correct[["model_label", "seed", "gold_count", "stimulus_id"]].copy()
    keys["seed"] = pd.to_numeric(keys["seed"], errors="raise").astype(int)
    keys["gold_count"] = pd.to_numeric(keys["gold_count"], errors="raise").astype(int)
    if keys.duplicated(["model_label", "seed", "gold_count"]).any():
        raise ValueError("Baseline labels are not unique by model/seed/count")
    by_key = {
        (str(row.model_label), int(row.seed), int(row.gold_count)): str(row.stimulus_id)
        for row in keys.itertuples(index=False)
    }
    models = sorted(str(value) for value in labels["model_label"].unique())
    seeds = sorted(
        pd.to_numeric(labels["seed"], errors="raise").astype(int).unique().tolist()
    )
    rows: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            for lower, upper in canonical_pairs:
                lower_key = (model, seed, int(lower))
                upper_key = (model, seed, int(upper))
                if lower_key not in by_key or upper_key not in by_key:
                    continue
                for receiver, donor in ((lower, upper), (upper, lower)):
                    rows.append(
                        {
                            "model_label": model,
                            "seed": int(seed),
                            "receiver_count": int(receiver),
                            "donor_count": int(donor),
                            "receiver_stimulus_id": by_key[
                                (model, seed, int(receiver))
                            ],
                            "donor_stimulus_id": by_key[(model, seed, int(donor))],
                            "k": abs(int(donor) - int(receiver)),
                            "target_direction": (
                                "increase" if int(donor) > int(receiver) else "decrease"
                            ),
                        }
                    )
    columns = [
        "model_label",
        "seed",
        "receiver_count",
        "donor_count",
        "receiver_stimulus_id",
        "donor_stimulus_id",
        "k",
        "target_direction",
    ]
    return pd.DataFrame(rows, columns=columns)


def existing_clean_pair_instances(detail: pd.DataFrame) -> pd.DataFrame:
    """Extract unique eligible treatment pair instances from a patching detail."""

    required = {
        "model_label",
        "condition",
        "seed",
        "receiver_count",
        "donor_count",
        "receiver_stimulus_id",
        "donor_stimulus_id",
        "k",
        "target_direction",
        "baseline_is_correct",
        "baseline_format_valid",
        "donor_baseline_outcome",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Patching detail is missing columns: {missing}")
    treatment = detail[detail["condition"].astype(str).eq("donor_transport")].copy()
    eligible = treatment[
        _as_bool(treatment["baseline_is_correct"])
        & _as_bool(treatment["baseline_format_valid"])
        & treatment["donor_baseline_outcome"].astype(str).eq("correct")
    ].copy()
    columns = [
        "model_label",
        "seed",
        "receiver_count",
        "donor_count",
        "receiver_stimulus_id",
        "donor_stimulus_id",
        "k",
        "target_direction",
    ]
    return eligible[columns].drop_duplicates().reset_index(drop=True)


def _pair_support(
    pairs: pd.DataFrame,
    *,
    target: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in (1, 3, 5):
        for direction in ("increase", "decrease"):
            group = pairs[
                pd.to_numeric(pairs["k"], errors="coerce").fillna(-1).astype(int).eq(k)
                & pairs["target_direction"].astype(str).eq(direction)
            ]
            clusters = int(group["seed"].nunique()) if not group.empty else 0
            rows.append(
                {
                    "k": int(k),
                    "target_direction": direction,
                    "eligible_pair_instances": int(len(group)),
                    "seed_clusters": clusters,
                    "target_seed_clusters": int(target),
                    "missing_seed_clusters": max(0, int(target) - clusters),
                    "quota_met": bool(clusters >= int(target)),
                }
            )
    return rows


def select_sequential_supplement(
    *,
    existing_pairs: pd.DataFrame,
    existing_ablation_baselines: pd.DataFrame,
    candidate_baselines: pd.DataFrame,
    reserve_seeds: Sequence[int],
    patch_cluster_target: int = PATCH_CLUSTER_TARGET,
    ablation_cluster_target: int = CORRECT_ABLATION_CLUSTER_TARGET,
    ablation_counts: Sequence[int] = CORRECT_ABLATION_COUNTS,
    canonical_pairs: Sequence[tuple[int, int]] = CAUSAL_V2_CANONICAL_PAIRS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Choose the first baseline-only seed prefix that satisfies both quotas.

    The stopping rule can inspect only unmodified greedy baseline correctness.
    It never reads patching or ablation outcomes.
    """

    if patch_cluster_target < 1 or ablation_cluster_target < 1:
        raise ValueError("Supplement targets must be positive")
    models = sorted(str(value) for value in candidate_baselines["model_label"].unique())
    if len(models) != 1:
        raise ValueError("Sequential supplement selection expects exactly one model")
    model = models[0]
    existing_models = set(
        str(value) for value in existing_pairs["model_label"].unique()
    )
    if existing_models and existing_models != {model}:
        raise ValueError("Existing and candidate pair models differ")
    candidate = candidate_baselines.copy()
    candidate["seed"] = pd.to_numeric(candidate["seed"], errors="raise").astype(int)
    available = set(candidate["seed"].unique().tolist())
    ordered = [int(value) for value in reserve_seeds]
    if len(set(ordered)) != len(ordered):
        raise ValueError("Reserve seeds must be unique")
    missing_seeds = sorted(set(ordered) - available)
    if missing_seeds:
        raise ValueError(
            f"Candidate baselines lack reserve seeds: {missing_seeds[:10]}"
        )

    initial_pairs = existing_pairs.drop_duplicates().reset_index(drop=True)
    initial_ablation = clean_correct_baselines(
        existing_ablation_baselines, counts=ablation_counts
    )
    initial_patch_support = _pair_support(initial_pairs, target=patch_cluster_target)
    initial_ablation_clusters = int(initial_ablation["seed"].nunique())
    initial_ablation_shortage = max(
        0, int(ablation_cluster_target) - initial_ablation_clusters
    )

    scanned: list[int] = []
    selected_labels = candidate.iloc[0:0].copy()
    fresh_pairs = eligible_directed_pairs(
        selected_labels, canonical_pairs=canonical_pairs
    )
    fresh_ablation = clean_correct_baselines(selected_labels, counts=ablation_counts)
    final_pairs = initial_pairs.copy()
    final_ablation = initial_ablation.copy()
    quota_met = False
    for seed in ordered:
        scanned.append(int(seed))
        selected_labels = candidate[candidate["seed"].isin(scanned)].copy()
        fresh_pairs = eligible_directed_pairs(
            selected_labels, canonical_pairs=canonical_pairs
        )
        fresh_ablation = clean_correct_baselines(
            selected_labels, counts=ablation_counts
        )
        final_pairs = pd.concat(
            [initial_pairs, fresh_pairs], ignore_index=True, sort=False
        ).drop_duplicates()
        final_ablation = pd.concat(
            [initial_ablation, fresh_ablation], ignore_index=True, sort=False
        ).drop_duplicates(["model_label", "seed", "gold_count"])
        patch_support = _pair_support(final_pairs, target=patch_cluster_target)
        quota_met = all(bool(row["quota_met"]) for row in patch_support) and int(
            final_ablation["seed"].nunique()
        ) >= int(ablation_cluster_target)
        if quota_met:
            break

    selected_pair_parts: list[pd.DataFrame] = []
    for initial_row in initial_patch_support:
        k = int(initial_row["k"])
        direction = str(initial_row["target_direction"])
        deficit = int(initial_row["missing_seed_clusters"])
        if deficit == 0:
            continue
        existing_seeds = set(
            pd.to_numeric(
                initial_pairs.loc[
                    pd.to_numeric(initial_pairs["k"], errors="raise").astype(int).eq(k)
                    & initial_pairs["target_direction"].astype(str).eq(direction),
                    "seed",
                ],
                errors="raise",
            )
            .astype(int)
            .tolist()
        )
        candidates = fresh_pairs[
            pd.to_numeric(fresh_pairs["k"], errors="raise").astype(int).eq(k)
            & fresh_pairs["target_direction"].astype(str).eq(direction)
            & ~pd.to_numeric(fresh_pairs["seed"], errors="raise")
            .astype(int)
            .isin(existing_seeds)
        ].copy()
        candidates = candidates.sort_values(["seed", "receiver_count", "donor_count"])
        chosen: list[pd.DataFrame] = []
        for _seed, seed_frame in candidates.groupby("seed", sort=True):
            chosen.append(seed_frame.head(1))
            if len(chosen) == deficit:
                break
        if len(chosen) != deficit:
            continue
        selected_pair_parts.append(pd.concat(chosen, ignore_index=True))
    selected_fresh_pairs = (
        pd.concat(selected_pair_parts, ignore_index=True, sort=False)
        if selected_pair_parts
        else fresh_pairs.iloc[0:0].copy()
    )
    selected_final_pairs = pd.concat(
        [initial_pairs, selected_fresh_pairs], ignore_index=True, sort=False
    ).drop_duplicates()
    final_patch_support = _pair_support(
        selected_final_pairs, target=patch_cluster_target
    )
    quota_met = quota_met and all(bool(row["quota_met"]) for row in final_patch_support)
    added_pair_seeds = sorted(
        pd.to_numeric(selected_fresh_pairs["seed"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    observed_fresh_ablation_seeds = sorted(
        pd.to_numeric(fresh_ablation["seed"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    initial_ablation_seeds = set(
        pd.to_numeric(initial_ablation["seed"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    added_ablation_seeds = [
        seed
        for seed in observed_fresh_ablation_seeds
        if seed not in initial_ablation_seeds
    ][:initial_ablation_shortage]
    if added_ablation_seeds:
        discovery_prefix_end = ordered.index(added_ablation_seeds[-1]) + 1
        shared_discovery_seed_prefix = ordered[:discovery_prefix_end]
    else:
        shared_discovery_seed_prefix = []
    selected_fresh_ablation = fresh_ablation[
        pd.to_numeric(fresh_ablation["seed"], errors="raise")
        .astype(int)
        .isin(added_ablation_seeds)
    ].copy()
    selected_final_ablation = pd.concat(
        [initial_ablation, selected_fresh_ablation],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(["model_label", "seed", "gold_count"])
    quota_met = quota_met and int(selected_final_ablation["seed"].nunique()) >= int(
        ablation_cluster_target
    )
    per_seed: list[dict[str, Any]] = []
    for seed in scanned:
        seed_labels = selected_labels[selected_labels["seed"].eq(seed)]
        seed_pairs = fresh_pairs[fresh_pairs["seed"].eq(seed)]
        selected_seed_pairs = selected_fresh_pairs[
            selected_fresh_pairs["seed"].eq(seed)
        ]
        correct_counts = sorted(
            pd.to_numeric(
                clean_correct_baselines(seed_labels)["gold_count"], errors="raise"
            )
            .astype(int)
            .unique()
            .tolist()
        )
        per_seed.append(
            {
                "seed": int(seed),
                "correct_counts": correct_counts,
                "eligible_directed_pairs": [
                    [int(row.receiver_count), int(row.donor_count)]
                    for row in seed_pairs.itertuples(index=False)
                ],
                "selected_added_directed_pairs": [
                    [int(row.receiver_count), int(row.donor_count)]
                    for row in selected_seed_pairs.itertuples(index=False)
                ],
                "contributes_patching": bool(seed in added_pair_seeds),
                "contributes_correct_ablation": bool(seed in added_ablation_seeds),
            }
        )
    manifest = {
        "schema_version": "realistic_niah_v4_4_correct_intervention_supplement_v2",
        "model_label": model,
        "selection_status": "complete" if quota_met else "insufficient_reserve",
        "stopping_rule": (
            "first ordered reserve-seed prefix meeting baseline-correctness support "
            "quotas; intervention outcomes are never inspected"
        ),
        "patching": {
            "target_seed_clusters_per_model_k_direction": int(patch_cluster_target),
            "initial_support": initial_patch_support,
            "final_support": final_patch_support,
            "added_eligible_pair_seeds": added_pair_seeds,
            "selected_added_pair_instances": int(len(selected_fresh_pairs)),
        },
        "correct_only_ablation": {
            "counts": [int(value) for value in ablation_counts],
            "target_seed_clusters": int(ablation_cluster_target),
            "initial_seed_clusters": initial_ablation_clusters,
            "initial_missing_seed_clusters": initial_ablation_shortage,
            "observed_fresh_eligible_seed_clusters": int(
                len(observed_fresh_ablation_seeds)
            ),
            "final_seed_clusters": int(selected_final_ablation["seed"].nunique()),
            "added_eligible_seeds": added_ablation_seeds,
            "unselected_fresh_eligible_seeds": [
                seed
                for seed in observed_fresh_ablation_seeds
                if seed not in set(added_ablation_seeds)
            ],
            "shared_discovery_seed_prefix": shared_discovery_seed_prefix,
            "shared_discovery_expected_all_example_stimuli": int(
                len(shared_discovery_seed_prefix) * len(ablation_counts)
            ),
        },
        "reserve_seeds": ordered,
        "scanned_supplement_seeds": scanned,
        "unused_reserve_seeds": ordered[len(scanned) :],
        "per_scanned_seed": per_seed,
    }
    return (
        manifest,
        selected_fresh_pairs.reset_index(drop=True),
        selected_fresh_ablation.reset_index(drop=True),
    )


def _cluster_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    label: str,
    repetitions: int,
) -> tuple[float, float, float, int]:
    cluster_totals = (
        frame.assign(_value=pd.to_numeric(frame[value_column], errors="coerce"))
        .dropna(subset=["_value"])
        .groupby("seed")["_value"]
        .agg(["sum", "count"])
    )
    if cluster_totals.empty:
        return math.nan, math.nan, math.nan, 0
    sums = cluster_totals["sum"].to_numpy(dtype=float)
    counts = cluster_totals["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(_stable_seed(label))
    indices = rng.integers(0, len(sums), size=(int(repetitions), len(sums)))
    distribution = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return (
        float(sums.sum() / counts.sum()),
        float(low),
        float(high),
        int(len(sums)),
    )


def summarize_average_patching_accuracy(
    detail: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    """Summarize donor-target and receiver-retention accuracy for every group."""

    required = {
        *group_columns,
        "condition",
        "seed",
        "receiver_count",
        "donor_count",
        "strict_target_hit",
        "patched_is_correct",
        "patched_format_valid",
        "generated_count_shift",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Patching detail is missing summary columns: {missing}")
    work = detail[detail["condition"].astype(str).eq("donor_transport")].copy()
    if "status" in work.columns:
        work = work[work["status"].astype(str).eq("ok")].copy()
    if work.empty:
        raise ValueError("No successful donor-transport rows were supplied")
    work["strict_target_hit"] = _as_bool(work["strict_target_hit"]).astype(float)
    work["patched_is_correct"] = _as_bool(work["patched_is_correct"]).astype(float)
    work["patched_format_valid"] = _as_bool(work["patched_format_valid"]).astype(float)
    rows: list[dict[str, Any]] = []
    for keys, frame in work.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip(group_columns, keys))
        target_mean, target_low, target_high, target_clusters = _cluster_bootstrap(
            frame,
            "strict_target_hit",
            label="patch-target:" + ":".join(str(value) for value in keys),
            repetitions=bootstrap_repetitions,
        )
        receiver_mean, receiver_low, receiver_high, _ = _cluster_bootstrap(
            frame,
            "patched_is_correct",
            label="patch-receiver:" + ":".join(str(value) for value in keys),
            repetitions=bootstrap_repetitions,
        )
        rows.append(
            {
                **metadata,
                "pair_instances": int(len(frame)),
                "seed_clusters": int(frame["seed"].nunique()),
                "average_patching_acc": target_mean,
                "average_patching_acc_ci95_low": target_low,
                "average_patching_acc_ci95_high": target_high,
                "patching_acc_seed_clusters": target_clusters,
                "patching_acc_successes": int(frame["strict_target_hit"].sum()),
                "patching_acc_denominator": int(len(frame)),
                "patching_acc_definition": PATCH_TARGET_ACCURACY_DEFINITION,
                "average_post_patch_receiver_acc": receiver_mean,
                "average_post_patch_receiver_acc_ci95_low": receiver_low,
                "average_post_patch_receiver_acc_ci95_high": receiver_high,
                "patched_valid_rate": float(frame["patched_format_valid"].mean()),
                "average_signed_count_shift_valid": float(
                    pd.to_numeric(
                        frame["generated_count_shift"], errors="coerce"
                    ).mean()
                ),
                "bootstrap_repetitions": int(bootstrap_repetitions),
            }
        )
    return pd.DataFrame(rows).sort_values(list(group_columns)).reset_index(drop=True)


def summarize_ablation_population(
    detail: pd.DataFrame,
    *,
    population: str,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    """Summarize signed all-example or failure-induction correct-only ablation."""

    if population not in {"all_examples_signed", "clean_correct_only"}:
        raise ValueError(f"Unknown ablation population: {population}")
    required = {
        "model_label",
        "stimulus_id",
        "seed",
        "gold_count",
        "head_bank",
        "top_n",
        "condition",
        "baseline_is_correct",
        "baseline_format_valid",
        "patched_is_correct",
        "patched_format_valid",
        "accuracy_delta",
        "absolute_error_delta",
        "generated_count_shift",
        "prediction_changed",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Ablation detail is missing summary columns: {missing}")
    work = detail.copy()
    if population == "clean_correct_only":
        work = work[
            _as_bool(work["baseline_is_correct"])
            & _as_bool(work["baseline_format_valid"])
        ].copy()
    if work.empty:
        raise ValueError(f"No rows remain for ablation population {population}")
    for column in (
        "baseline_is_correct",
        "patched_is_correct",
        "patched_format_valid",
        "prediction_changed",
    ):
        work[column] = _as_bool(work[column]).astype(float)
    work["absolute_count_shift"] = pd.to_numeric(
        work["generated_count_shift"], errors="coerce"
    ).abs()
    identifiers = [
        "model_label",
        "stimulus_id",
        "seed",
        "gold_count",
        "head_bank",
        "top_n",
    ]
    metrics = [
        "accuracy_delta",
        "absolute_error_delta",
        "generated_count_shift",
        "absolute_count_shift",
        "prediction_changed",
    ]
    ranked = work[work["condition"].astype(str).eq("ranked")].copy()
    random = work[work["condition"].astype(str).eq("layer_matched_random")].copy()
    random_mean = random.groupby(identifiers, as_index=False, dropna=False).agg(
        **{f"{metric}_random": (metric, "mean") for metric in metrics}
    )
    paired = ranked.merge(
        random_mean, on=identifiers, how="left", validate="one_to_one"
    )
    for metric in metrics:
        paired[f"{metric}_ranked_minus_random"] = pd.to_numeric(
            paired[metric], errors="coerce"
        ) - pd.to_numeric(paired[f"{metric}_random"], errors="coerce")
    groups = ["model_label", "head_bank", "top_n"]
    rows: list[dict[str, Any]] = []
    for keys, frame in paired.groupby(groups, sort=True, dropna=False):
        metadata = dict(zip(groups, keys))
        signed_mean, signed_low, signed_high, signed_clusters = _cluster_bootstrap(
            frame,
            "generated_count_shift",
            label=f"ablation:{population}:signed:" + ":".join(map(str, keys)),
            repetitions=bootstrap_repetitions,
        )
        failure = 1.0 - pd.to_numeric(frame["patched_is_correct"], errors="raise")
        frame = frame.assign(correct_to_wrong=failure)
        failure_mean, failure_low, failure_high, failure_clusters = _cluster_bootstrap(
            frame,
            "correct_to_wrong",
            label=f"ablation:{population}:failure:" + ":".join(map(str, keys)),
            repetitions=bootstrap_repetitions,
        )
        rows.append(
            {
                **metadata,
                "analysis_population": population,
                "examples": int(len(frame)),
                "seed_clusters": int(frame["seed"].nunique()),
                "baseline_accuracy": float(frame["baseline_is_correct"].mean()),
                "post_ablation_accuracy": float(frame["patched_is_correct"].mean()),
                "mean_accuracy_delta": float(
                    pd.to_numeric(frame["accuracy_delta"], errors="raise").mean()
                ),
                "mean_signed_count_shift_valid": signed_mean,
                "mean_signed_count_shift_ci95_low": signed_low,
                "mean_signed_count_shift_ci95_high": signed_high,
                "signed_shift_seed_clusters": signed_clusters,
                "mean_absolute_count_shift_valid": float(
                    pd.to_numeric(frame["absolute_count_shift"], errors="coerce").mean()
                ),
                "mean_absolute_error_delta": float(
                    pd.to_numeric(frame["absolute_error_delta"], errors="raise").mean()
                ),
                "prediction_changed_rate": float(frame["prediction_changed"].mean()),
                "patched_valid_rate": float(frame["patched_format_valid"].mean()),
                "correct_to_wrong_rate": (
                    failure_mean if population == "clean_correct_only" else math.nan
                ),
                "correct_to_wrong_ci95_low": (
                    failure_low if population == "clean_correct_only" else math.nan
                ),
                "correct_to_wrong_ci95_high": (
                    failure_high if population == "clean_correct_only" else math.nan
                ),
                "correct_to_wrong_seed_clusters": (
                    failure_clusters if population == "clean_correct_only" else 0
                ),
                "ranked_minus_random_signed_shift": float(
                    pd.to_numeric(
                        frame["generated_count_shift_ranked_minus_random"],
                        errors="coerce",
                    ).mean()
                ),
                "ranked_minus_random_absolute_shift": float(
                    pd.to_numeric(
                        frame["absolute_count_shift_ranked_minus_random"],
                        errors="coerce",
                    ).mean()
                ),
                "ranked_minus_random_accuracy_delta": float(
                    pd.to_numeric(
                        frame["accuracy_delta_ranked_minus_random"],
                        errors="coerce",
                    ).mean()
                ),
                "ranked_minus_random_absolute_error_delta": float(
                    pd.to_numeric(
                        frame["absolute_error_delta_ranked_minus_random"],
                        errors="coerce",
                    ).mean()
                ),
                "bootstrap_repetitions": int(bootstrap_repetitions),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def summarize_ablation_n_diagnostics(
    detail: pd.DataFrame,
    *,
    population: str,
    head_bank: str = "broad_aggregation",
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    """Compare every ablation dose without automatically selecting top-n.

    For all examples, the discovery endpoint is the ranked-minus-random
    increase in absolute count shift.  For clean-correct examples, it is the
    ranked-minus-random increase in correct-to-wrong failures.  Signed shifts
    remain separate direction-bearing endpoints.
    """

    if population not in {"all_examples_signed", "clean_correct_only"}:
        raise ValueError(f"Unknown ablation population: {population}")
    required = {
        "model_label",
        "stimulus_id",
        "seed",
        "gold_count",
        "head_bank",
        "top_n",
        "condition",
        "baseline_is_correct",
        "baseline_format_valid",
        "patched_is_correct",
        "patched_format_valid",
        "accuracy_delta",
        "absolute_error_delta",
        "generated_count_shift",
        "prediction_changed",
        "ranked_random_head_overlap",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Ablation detail is missing diagnostic columns: {missing}")
    work = detail[detail["head_bank"].astype(str).eq(str(head_bank))].copy()
    if population == "clean_correct_only":
        work = work[
            _as_bool(work["baseline_is_correct"])
            & _as_bool(work["baseline_format_valid"])
        ].copy()
    if work.empty:
        raise ValueError(
            f"No {head_bank} rows remain for ablation population {population}"
        )
    for column in (
        "baseline_is_correct",
        "patched_is_correct",
        "patched_format_valid",
        "prediction_changed",
    ):
        work[column] = _as_bool(work[column]).astype(float)
    work["signed_shift"] = pd.to_numeric(work["generated_count_shift"], errors="coerce")
    work["absolute_shift"] = work["signed_shift"].abs()
    work["correct_to_wrong"] = 1.0 - work["patched_is_correct"]

    identifiers = [
        "model_label",
        "stimulus_id",
        "seed",
        "gold_count",
        "head_bank",
        "top_n",
    ]
    metrics = (
        "signed_shift",
        "absolute_shift",
        "correct_to_wrong",
        "accuracy_delta",
        "absolute_error_delta",
        "prediction_changed",
        "patched_format_valid",
    )
    ranked = work[work["condition"].astype(str).eq("ranked")].copy()
    random = work[work["condition"].astype(str).eq("layer_matched_random")].copy()
    random_mean = random.groupby(identifiers, as_index=False, dropna=False).agg(
        **{f"{metric}_random": (metric, "mean") for metric in metrics},
        random_replicates_observed=("condition", "size"),
        random_overlap_mean=("ranked_random_head_overlap", "mean"),
    )
    paired = ranked.merge(
        random_mean, on=identifiers, how="inner", validate="one_to_one"
    )
    if len(paired) != len(ranked):
        raise ValueError("Ablation n diagnostics have incomplete random controls")
    for metric in metrics:
        paired[f"{metric}_effect"] = pd.to_numeric(
            paired[metric], errors="coerce"
        ) - pd.to_numeric(paired[f"{metric}_random"], errors="coerce")

    groups = ["model_label", "head_bank", "top_n"]
    rows: list[dict[str, Any]] = []
    for keys, frame in paired.groupby(groups, sort=True, dropna=False):
        metadata = dict(zip(groups, keys))
        signed_effect, signed_low, signed_high, clusters = _cluster_bootstrap(
            frame,
            "signed_shift_effect",
            label=f"ablation-n:{population}:signed:" + ":".join(map(str, keys)),
            repetitions=bootstrap_repetitions,
        )
        absolute_effect, absolute_low, absolute_high, _ = _cluster_bootstrap(
            frame,
            "absolute_shift_effect",
            label=f"ablation-n:{population}:absolute:" + ":".join(map(str, keys)),
            repetitions=bootstrap_repetitions,
        )
        failure_effect, failure_low, failure_high, _ = _cluster_bootstrap(
            frame,
            "correct_to_wrong_effect",
            label=f"ablation-n:{population}:failure:" + ":".join(map(str, keys)),
            repetitions=bootstrap_repetitions,
        )
        if population == "all_examples_signed":
            primary_metric = "ranked_minus_random_absolute_count_shift"
            primary_effect = absolute_effect
            primary_low = absolute_low
            primary_high = absolute_high
        else:
            primary_metric = "ranked_minus_random_correct_to_wrong"
            primary_effect = failure_effect
            primary_low = failure_low
            primary_high = failure_high
        rows.append(
            {
                **metadata,
                "analysis_population": population,
                "selection_status": "discovery_only_unfrozen",
                "examples": int(len(frame)),
                "seed_clusters": int(frame["seed"].nunique()),
                "random_replicates_min": int(frame["random_replicates_observed"].min()),
                "ranked_mean_signed_count_shift": float(
                    pd.to_numeric(frame["signed_shift"], errors="coerce").mean()
                ),
                "random_mean_signed_count_shift": float(
                    pd.to_numeric(frame["signed_shift_random"], errors="coerce").mean()
                ),
                "ranked_minus_random_signed_count_shift": signed_effect,
                "ranked_minus_random_signed_count_shift_ci95_low": signed_low,
                "ranked_minus_random_signed_count_shift_ci95_high": signed_high,
                "ranked_mean_absolute_count_shift": float(
                    pd.to_numeric(frame["absolute_shift"], errors="coerce").mean()
                ),
                "random_mean_absolute_count_shift": float(
                    pd.to_numeric(
                        frame["absolute_shift_random"], errors="coerce"
                    ).mean()
                ),
                "ranked_minus_random_absolute_count_shift": absolute_effect,
                "ranked_minus_random_absolute_count_shift_ci95_low": absolute_low,
                "ranked_minus_random_absolute_count_shift_ci95_high": absolute_high,
                "ranked_correct_to_wrong_rate": (
                    float(frame["correct_to_wrong"].mean())
                    if population == "clean_correct_only"
                    else math.nan
                ),
                "random_correct_to_wrong_rate": (
                    float(frame["correct_to_wrong_random"].mean())
                    if population == "clean_correct_only"
                    else math.nan
                ),
                "ranked_minus_random_correct_to_wrong": (
                    failure_effect if population == "clean_correct_only" else math.nan
                ),
                "ranked_minus_random_correct_to_wrong_ci95_low": (
                    failure_low if population == "clean_correct_only" else math.nan
                ),
                "ranked_minus_random_correct_to_wrong_ci95_high": (
                    failure_high if population == "clean_correct_only" else math.nan
                ),
                "ranked_minus_random_accuracy_delta": float(
                    pd.to_numeric(
                        frame["accuracy_delta_effect"], errors="coerce"
                    ).mean()
                ),
                "ranked_minus_random_absolute_error_delta": float(
                    pd.to_numeric(
                        frame["absolute_error_delta_effect"], errors="coerce"
                    ).mean()
                ),
                "ranked_minus_random_prediction_changed": float(
                    pd.to_numeric(
                        frame["prediction_changed_effect"], errors="coerce"
                    ).mean()
                ),
                "ranked_valid_rate": float(frame["patched_format_valid"].mean()),
                "random_valid_rate": float(frame["patched_format_valid_random"].mean()),
                "random_overlap_mean": float(frame["random_overlap_mean"].mean()),
                "primary_metric": primary_metric,
                "primary_effect": primary_effect,
                "primary_effect_ci95_low": primary_low,
                "primary_effect_ci95_high": primary_high,
                "primary_ci95_excludes_zero_positive": bool(primary_low > 0),
                "effect_seed_clusters": clusters,
                "bootstrap_repetitions": int(bootstrap_repetitions),
            }
        )
    result = pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)
    result["primary_rank_within_model_bank"] = (
        result.groupby(["model_label", "head_bank"])["primary_effect"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return result
