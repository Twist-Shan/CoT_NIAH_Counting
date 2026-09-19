from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .prompts import PromptEncoding, TokenSpan

POOLINGS = ("span_end", "span_mean")
ASSESSMENT_THRESHOLDS = {
    "first_occurrence_share": 0.80,
    "winner_first_rate": 0.90,
    "stable_absolute_depth_bin_rate": 0.80,
    "near_uniform_local_effective_fraction": 0.80,
    "broader_span_mean_effective_number": 3.00,
    "moderately_distributed_local_effective_fraction": 0.70,
    "prefix_row_mass": 0.50,
}


def occurrence_attention_values(
    attention_row: np.ndarray,
    spans: Sequence[TokenSpan],
    *,
    key_start: int,
    pooling: str,
) -> np.ndarray:
    """Return one comparable attention value per occurrence.

    `span_end` uses the last token in each exact model-token span. `span_mean`
    averages all tokens in each span, so realistic records with slightly
    different tokenizer lengths remain comparable.
    """

    if pooling not in POOLINGS:
        raise ValueError(f"Unknown occurrence pooling: {pooling}")
    row = np.asarray(attention_row, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError("attention_row must be one-dimensional")
    if not spans:
        raise ValueError("At least one occurrence span is required")
    values: list[float] = []
    for span in spans:
        start = int(span.start) - int(key_start)
        end = int(span.end) - int(key_start)
        if start < 0 or end > len(row) or end <= start:
            raise ValueError(
                "Occurrence span is not fully visible in the saved query row: "
                f"slot={span.slot_index} span=[{span.start},{span.end}) "
                f"key_start={key_start} key_length={len(row)}"
            )
        value = row[end - 1] if pooling == "span_end" else row[start:end].mean()
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def depth_bin_masses(attention_row: np.ndarray, *, bins: int) -> np.ndarray:
    """Sum a query row into equal-width relative-depth bins."""

    row = np.asarray(attention_row, dtype=np.float64)
    if row.ndim != 1 or row.size == 0:
        raise ValueError("attention_row must be a non-empty vector")
    if int(bins) < 2:
        raise ValueError("bins must be at least two")
    edges = np.linspace(0, len(row), int(bins) + 1, dtype=int)
    return np.asarray(
        [row[edges[index] : edges[index + 1]].sum() for index in range(bins)],
        dtype=np.float64,
    )


def partition_sample_metrics(
    values: np.ndarray,
    normalized_depths: np.ndarray,
    *,
    partitions: int,
    depth_bins: int = 10,
) -> dict[str, float | int | np.ndarray]:
    """Measure ordinal selection and uniformity inside the winning partition."""

    values = np.asarray(values, dtype=np.float64)
    depths = np.asarray(normalized_depths, dtype=np.float64)
    if values.ndim != 1 or depths.ndim != 1 or len(values) != len(depths):
        raise ValueError("values and normalized_depths must be aligned vectors")
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("values must be finite, non-negative, and non-empty")
    if np.any(~np.isfinite(depths)) or np.any((depths < 0) | (depths > 1)):
        raise ValueError("normalized_depths must lie in [0, 1]")
    if int(partitions) < 2 or int(depth_bins) < 2:
        raise ValueError("partitions and depth_bins must be at least two")

    total = float(values.sum())
    shares = values / total if total > 0 else np.zeros_like(values)
    effective = float(1.0 / np.square(shares).sum()) if total > 0 else 0.0
    winner = int(np.argmax(values))
    partition_ids = np.minimum((depths * partitions).astype(int), partitions - 1)
    depth_ids = np.minimum((depths * depth_bins).astype(int), depth_bins - 1)
    partition_mass = np.bincount(
        partition_ids,
        weights=values,
        minlength=int(partitions),
    )
    winner_partition = int(np.argmax(partition_mass))
    local = values[partition_ids == winner_partition]
    local_total = float(local.sum())
    local_shares = local / local_total if local_total > 0 else np.zeros_like(local)
    local_effective = (
        float(1.0 / np.square(local_shares).sum()) if local_total > 0 else 0.0
    )
    local_count = int(len(local))
    local_fraction = local_effective / local_count if local_count else np.nan
    local_mean = float(local.mean()) if local_count else 0.0
    local_cv = (
        float(local.std(ddof=0) / local_mean) if local_mean > 0 else np.nan
    )
    return {
        "normalized_shares": shares,
        "winner_occurrence_index": winner + 1,
        "winner_share": float(shares[winner]) if total > 0 else 0.0,
        "winner_depth": float(depths[winner]),
        "winner_depth_bin": int(depth_ids[winner]),
        "winner_partition": winner_partition,
        "effective_number": effective,
        "first_occurrence_share": float(shares[0]) if total > 0 else 0.0,
        "winner_is_first": int(winner == 0),
        "local_needle_count": local_count,
        "local_effective_number": local_effective,
        "local_effective_fraction": float(local_fraction),
        "local_cv": local_cv,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv_atomic(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression="gzip" if gzip else None,
    )
    temporary.replace(path)


def _write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_bootstrap_mean(
    values: pd.Series,
    seeds: pd.Series,
    *,
    repetitions: int,
    random_seed: int,
) -> tuple[float, float, float]:
    frame = pd.DataFrame({"value": values, "seed": seeds}).dropna()
    by_seed = frame.groupby("seed", sort=True)["value"].mean().to_numpy(dtype=float)
    if by_seed.size == 0:
        return np.nan, np.nan, np.nan
    estimate = float(by_seed.mean())
    if by_seed.size == 1 or repetitions <= 0:
        return estimate, estimate, estimate
    rng = np.random.default_rng(int(random_seed))
    indices = rng.integers(0, len(by_seed), size=(int(repetitions), len(by_seed)))
    draws = by_seed[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def _mode_and_frequency(values: pd.Series) -> tuple[int, float]:
    counts = Counter(int(value) for value in values)
    mode, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return int(mode), float(count / len(values))


def _ranking_heads(
    path: Path,
    *,
    top_k: int,
    all_candidates: bool,
) -> list[tuple[int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = (
        payload.get("full_candidate_ranking", [])
        if all_candidates
        else payload.get("top_heads", [])[: int(top_k)]
    )
    heads = [
        (
            int(row.get("rank", index)),
            int(row["layer"]),
            int(row["head"]),
        )
        for index, row in enumerate(source, start=1)
    ]
    pairs = [(layer, head) for _rank, layer, head in heads]
    if len(heads) < int(top_k) or len(set(pairs)) != len(pairs):
        raise ValueError(f"Ranking does not contain {top_k} unique heads: {path}")
    heads.sort(key=lambda item: item[0])
    return heads


def _head_summary(
    samples: pd.DataFrame,
    occurrences: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
    bootstrap_top_k: int,
    random_seed: int,
) -> pd.DataFrame:
    keys = ["model_label", "design_variant", "pooling", "head_rank", "layer", "head"]
    rows: list[dict[str, Any]] = []
    occurrence_groups = occurrences.groupby(keys, sort=False)
    for group_index, (key, frame) in enumerate(samples.groupby(keys, sort=True)):
        item = dict(zip(keys, key))
        occurrence = occurrence_groups.get_group(key)
        mean_share = occurrence.groupby("occurrence_index")[
            "normalized_share"
        ].mean()
        dominant_occurrence = int(mean_share.idxmax())
        winner_mode, winner_mode_frequency = _mode_and_frequency(
            frame["winner_occurrence_index"]
        )
        depth_mode, depth_mode_frequency = _mode_and_frequency(
            frame["winner_depth_bin"]
        )
        partition_mode, partition_mode_frequency = _mode_and_frequency(
            frame["winner_partition"]
        )
        item.update(
            {
                "examples": int(len(frame)),
                "seeds": int(frame["seed"].nunique()),
                "dominant_occurrence": dominant_occurrence,
                "dominant_occurrence_mean_share": float(
                    mean_share.loc[dominant_occurrence]
                ),
                "winner_occurrence_mode": winner_mode,
                "winner_occurrence_mode_frequency": winner_mode_frequency,
                "winner_depth_bin_mode": depth_mode,
                "winner_depth_bin_mode_frequency": depth_mode_frequency,
                "winner_partition_mode": partition_mode,
                "winner_partition_mode_frequency": partition_mode_frequency,
            }
        )
        metrics = (
            "first_occurrence_share",
            "winner_is_first",
            "winner_depth",
            "effective_number",
            "local_needle_count",
            "local_effective_number",
            "local_effective_fraction",
            "local_cv",
        )
        for metric_index, metric in enumerate(metrics):
            run_bootstrap = int(item["head_rank"]) <= int(bootstrap_top_k)
            if run_bootstrap:
                mean, low, high = _seed_bootstrap_mean(
                    frame[metric],
                    frame["seed"],
                    repetitions=bootstrap_repetitions,
                    random_seed=(
                        int(random_seed) + 1009 * group_index + 97 * metric_index
                    ),
                )
            else:
                mean = float(frame[metric].mean())
                low = high = np.nan
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95_low"] = low if run_bootstrap else np.nan
            item[f"{metric}_ci95_high"] = high if run_bootstrap else np.nan
        rows.append(item)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _depth_summary(
    detail: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
    bootstrap_top_k: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "model_label",
        "design_variant",
        "head_rank",
        "layer",
        "head",
        "depth_bin",
    ]
    for group_index, (key, frame) in enumerate(detail.groupby(keys, sort=True)):
        rank = int(frame["head_rank"].iloc[0])
        run_bootstrap = rank <= int(bootstrap_top_k)
        if run_bootstrap:
            mean, low, high = _seed_bootstrap_mean(
                frame["row_mass_fraction"],
                frame["seed"],
                repetitions=bootstrap_repetitions,
                random_seed=int(random_seed) + 7919 * group_index,
            )
        else:
            mean = float(frame["row_mass_fraction"].mean())
            low = high = np.nan
        rows.append(
            {
                **dict(zip(keys, key)),
                "bin_start": float(frame["bin_start"].iloc[0]),
                "bin_end": float(frame["bin_end"].iloc[0]),
                "seeds": int(frame["seed"].nunique()),
                "row_mass_fraction_mean": mean,
                "row_mass_fraction_ci95_low": low if run_bootstrap else np.nan,
                "row_mass_fraction_ci95_high": high if run_bootstrap else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def assess_rank1_partitioning(
    head_summary: pd.DataFrame,
    depth_detail: pd.DataFrame,
    *,
    depth_bins: int,
) -> dict[str, Any]:
    """Apply transparent descriptive rules to each variant's rank-1 head."""

    assessments: list[dict[str, Any]] = []
    for variant in sorted(head_summary["design_variant"].unique()):
        selected = head_summary[
            (head_summary["design_variant"] == variant)
            & (head_summary["head_rank"] == 1)
        ]
        by_pool = {str(row.pooling): row for row in selected.itertuples()}
        if set(by_pool) != set(POOLINGS):
            raise ValueError(f"Rank-1 dual-pooling summary is incomplete for {variant}")
        endpoint = by_pool["span_end"]
        span_mean = by_pool["span_mean"]
        depth = depth_detail[
            (depth_detail["design_variant"] == variant)
            & (depth_detail["head_rank"] == 1)
        ]
        prefix_bins = max(1, int(depth_bins) // 4)
        prefix_mass = float(
            depth.groupby("depth_bin")["row_mass_fraction"].mean().iloc[
                :prefix_bins
            ].sum()
        )
        selector = bool(
            endpoint.first_occurrence_share_mean
            >= ASSESSMENT_THRESHOLDS["first_occurrence_share"]
            and endpoint.winner_is_first_mean
            >= ASSESSMENT_THRESHOLDS["winner_first_rate"]
        )
        endpoint_uniform = bool(
            endpoint.local_effective_fraction_mean
            >= ASSESSMENT_THRESHOLDS[
                "near_uniform_local_effective_fraction"
            ]
        )
        broader_mean = bool(
            span_mean.effective_number_mean
            >= ASSESSMENT_THRESHOLDS["broader_span_mean_effective_number"]
            and span_mean.local_effective_fraction_mean
            >= ASSESSMENT_THRESHOLDS[
                "moderately_distributed_local_effective_fraction"
            ]
        )
        stable_absolute_depth = bool(
            endpoint.winner_depth_bin_mode_frequency
            >= ASSESSMENT_THRESHOLDS["stable_absolute_depth_bin_rate"]
        )
        prefix_gated = bool(
            prefix_mass >= ASSESSMENT_THRESHOLDS["prefix_row_mass"]
        )
        if selector and not endpoint_uniform and broader_mean:
            classification = (
                "first_occurrence_endpoint_selector_with_broader_span_mean"
            )
        elif selector and not endpoint_uniform:
            classification = "first_occurrence_endpoint_selector"
        elif stable_absolute_depth and endpoint_uniform:
            classification = "depth_partition_local_endpoint_aggregator"
        else:
            classification = "mixed_or_unsupported"
        assessments.append(
            {
                "design_variant": variant,
                "layer": int(endpoint.layer),
                "head": int(endpoint.head),
                "classification": classification,
                "first_occurrence_endpoint_selector": selector,
                "stable_absolute_depth_bin": stable_absolute_depth,
                "prefix_gated_full_query_row": prefix_gated,
                "near_uniform_endpoint_aggregation_inside_winning_partition": (
                    endpoint_uniform
                ),
                "broader_span_mean_distribution": broader_mean,
                "endpoint_first_occurrence_share": float(
                    endpoint.first_occurrence_share_mean
                ),
                "endpoint_winner_first_rate": float(endpoint.winner_is_first_mean),
                "endpoint_winner_depth_bin_mode_frequency": float(
                    endpoint.winner_depth_bin_mode_frequency
                ),
                "endpoint_local_needle_count": float(
                    endpoint.local_needle_count_mean
                ),
                "endpoint_local_effective_number": float(
                    endpoint.local_effective_number_mean
                ),
                "endpoint_local_effective_fraction": float(
                    endpoint.local_effective_fraction_mean
                ),
                "span_mean_effective_number": float(
                    span_mean.effective_number_mean
                ),
                "span_mean_local_effective_fraction": float(
                    span_mean.local_effective_fraction_mean
                ),
                "full_row_prefix_quarter_mass": prefix_mass,
            }
        )
    return {
        "schema_version": "realistic_niah_v4_partition_assessment_v1",
        "thresholds": ASSESSMENT_THRESHOLDS,
        "interpretation": (
            "Classification is descriptive. A partition-local causal counting "
            "claim still requires ablation or patching against matched controls."
        ),
        "assessments": assessments,
    }


def classify_candidate_heads(
    head_summary: pd.DataFrame,
    depth_detail: pd.DataFrame,
    *,
    depth_bins: int,
) -> pd.DataFrame:
    """Assign transparent endpoint-attention phenotypes to every candidate."""

    keys = ["model_label", "design_variant", "head_rank", "layer", "head"]
    metric_columns = [
        "dominant_occurrence_mean_share",
        "winner_occurrence_mode",
        "winner_occurrence_mode_frequency",
        "winner_depth_bin_mode",
        "winner_depth_bin_mode_frequency",
        "first_occurrence_share_mean",
        "winner_is_first_mean",
        "effective_number_mean",
        "local_needle_count_mean",
        "local_effective_number_mean",
        "local_effective_fraction_mean",
        "local_cv_mean",
    ]
    endpoint = head_summary[head_summary["pooling"] == "span_end"]
    span_mean = head_summary[head_summary["pooling"] == "span_mean"]
    endpoint = endpoint[keys + metric_columns].rename(
        columns={column: f"endpoint_{column}" for column in metric_columns}
    )
    span_mean = span_mean[keys + metric_columns].rename(
        columns={column: f"span_mean_{column}" for column in metric_columns}
    )
    result = endpoint.merge(span_mean, on=keys, validate="one_to_one")

    depth = depth_detail.copy()
    depth["row_quartile"] = np.minimum(
        (depth["depth_bin"].to_numpy(dtype=int) * 4) // int(depth_bins),
        3,
    )
    per_sample_quarter = (
        depth.groupby(keys + ["stimulus_id", "seed", "row_quartile"], sort=True)[
            "row_mass_fraction"
        ]
        .sum()
        .reset_index()
    )
    quarter = per_sample_quarter.pivot_table(
        index=keys,
        columns="row_quartile",
        values="row_mass_fraction",
        aggfunc="mean",
        fill_value=0.0,
    ).reset_index()
    quarter = quarter.rename(
        columns={index: f"row_quartile_{index}_mass" for index in range(4)}
    )
    result = result.merge(quarter, on=keys, how="left", validate="one_to_one")
    quarter_columns = [f"row_quartile_{index}_mass" for index in range(4)]
    result["row_dominant_quartile"] = result[quarter_columns].to_numpy().argmax(
        axis=1
    )
    result["row_dominant_quartile_mass"] = result[quarter_columns].max(axis=1)
    result["span_mean_minus_endpoint_effective_number"] = (
        result["span_mean_effective_number_mean"]
        - result["endpoint_effective_number_mean"]
    )

    phenotypes: list[str] = []
    for row in result.itertuples():
        global_aggregator = bool(
            row.endpoint_effective_number_mean >= 6.0
            and row.endpoint_dominant_occurrence_mean_share <= 0.25
        )
        occurrence_selector = bool(
            row.endpoint_effective_number_mean <= 2.0
            and row.endpoint_winner_occurrence_mode_frequency >= 0.80
        )
        local_aggregator = bool(
            not global_aggregator
            and row.endpoint_local_needle_count_mean >= 2.0
            and row.endpoint_local_effective_fraction_mean >= 0.80
            and row.row_dominant_quartile_mass >= 0.50
        )
        span_mean_only = bool(
            not global_aggregator
            and row.span_mean_effective_number_mean >= 6.0
            and row.span_mean_dominant_occurrence_mean_share <= 0.25
        )
        if global_aggregator:
            phenotype = "global_endpoint_aggregator"
        elif local_aggregator:
            phenotype = "partition_local_endpoint_aggregator"
        elif occurrence_selector and span_mean_only:
            phenotype = "endpoint_selector_broad_span_mean"
        elif occurrence_selector:
            phenotype = "occurrence_endpoint_selector"
        elif span_mean_only:
            phenotype = "broad_span_mean_only"
        else:
            phenotype = "mixed"
        phenotypes.append(phenotype)
    result["phenotype"] = phenotypes
    return result.sort_values(keys).reset_index(drop=True)


def _profile_effective_number(profile: np.ndarray) -> float:
    profile = np.asarray(profile, dtype=np.float64)
    total = float(profile.sum())
    if total <= 0:
        return 0.0
    shares = profile / total
    return float(1.0 / np.square(shares).sum())


def _sample_raw_ensemble_effective(
    occurrence_detail: pd.DataFrame,
    selected_heads: Sequence[tuple[int, int]],
) -> float:
    if not selected_heads:
        return 0.0
    observed = pd.MultiIndex.from_arrays(
        [occurrence_detail["layer"], occurrence_detail["head"]]
    )
    selected = occurrence_detail[observed.isin(set(selected_heads))]
    summed = (
        selected.groupby(["stimulus_id", "occurrence_index"])[
            "raw_attention_value"
        ]
        .sum()
        .unstack("occurrence_index", fill_value=0.0)
    )
    values = summed.to_numpy(dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    shares = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    effective = np.divide(
        1.0,
        np.square(shares).sum(axis=1),
        out=np.zeros(len(shares), dtype=float),
        where=np.square(shares).sum(axis=1) > 0,
    )
    return float(effective.mean())


def head_bank_coverage(
    occurrence_detail: pd.DataFrame,
    *,
    diagnostic_top_k: int,
) -> pd.DataFrame:
    """Compare rank-ordered heads with a diversity-seeking candidate bank."""

    endpoint = occurrence_detail[occurrence_detail["pooling"] == "span_end"]
    rows: list[dict[str, Any]] = []
    for variant, frame in endpoint.groupby("design_variant", sort=True):
        profiles = frame.pivot_table(
            index=["head_rank", "layer", "head"],
            columns="occurrence_index",
            values="normalized_share",
            aggfunc="mean",
            fill_value=0.0,
        ).sort_index(level="head_rank")
        labels = list(profiles.index)
        vectors = profiles.to_numpy(dtype=float)
        rank_order = list(range(min(int(diagnostic_top_k), len(labels))))
        greedy = [0]
        current = vectors[0].copy()
        while len(greedy) < min(int(diagnostic_top_k), len(labels)):
            candidates = [index for index in range(len(labels)) if index not in greedy]
            scored = [
                (
                    _profile_effective_number(current + vectors[index]),
                    -int(labels[index][0]),
                    index,
                )
                for index in candidates
            ]
            chosen = max(scored)[2]
            greedy.append(chosen)
            current += vectors[chosen]
        for strategy, order in (
            ("registered_rank_order", rank_order),
            ("greedy_complement_from_rank1", greedy),
        ):
            selected: list[int] = []
            for step, index in enumerate(order, start=1):
                selected.append(index)
                head_pairs = [
                    (int(labels[item][1]), int(labels[item][2]))
                    for item in selected
                ]
                profile = vectors[selected].sum(axis=0)
                rank, layer, head = labels[index]
                rows.append(
                    {
                        "design_variant": variant,
                        "strategy": strategy,
                        "head_count": step,
                        "added_rank": int(rank),
                        "added_layer": int(layer),
                        "added_head": int(head),
                        "selected_heads": json.dumps(
                            [
                                {
                                    "rank": int(labels[item][0]),
                                    "layer": pair[0],
                                    "head": pair[1],
                                }
                                for item, pair in zip(selected, head_pairs)
                            ]
                        ),
                        "equal_head_profile_effective_number": (
                            _profile_effective_number(profile)
                        ),
                        "raw_attention_ensemble_effective_number": (
                            _sample_raw_ensemble_effective(frame, head_pairs)
                        ),
                    }
                )
        all_pairs = [(int(layer), int(head)) for _rank, layer, head in labels]
        rows.append(
            {
                "design_variant": variant,
                "strategy": "all_broad_candidates",
                "head_count": len(labels),
                "added_rank": np.nan,
                "added_layer": np.nan,
                "added_head": np.nan,
                "selected_heads": "all",
                "equal_head_profile_effective_number": (
                    _profile_effective_number(vectors.sum(axis=0))
                ),
                "raw_attention_ensemble_effective_number": (
                    _sample_raw_ensemble_effective(frame, all_pairs)
                ),
            }
        )
    return pd.DataFrame(rows)


def phenotype_bank_coverage(
    occurrence_detail: pd.DataFrame,
    phenotypes: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the joint endpoint coverage of every head phenotype.

    The equal-profile metric gives every head the same total weight. The raw
    metric preserves the observed attention magnitude, so disagreement between
    them exposes a candidate bank whose spatial coverage exists but is
    dominated by a small number of high-mass heads.
    """

    keys = ["model_label", "design_variant", "head_rank", "layer", "head"]
    endpoint = occurrence_detail[occurrence_detail["pooling"] == "span_end"]
    endpoint = endpoint.merge(
        phenotypes[keys + ["phenotype"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (variant, phenotype), frame in endpoint.groupby(
        ["design_variant", "phenotype"], sort=True
    ):
        profiles = frame.pivot_table(
            index=["head_rank", "layer", "head"],
            columns="occurrence_index",
            values="normalized_share",
            aggfunc="mean",
            fill_value=0.0,
        ).sort_index(level="head_rank")
        labels = list(profiles.index)
        pairs = [(int(layer), int(head)) for _rank, layer, head in labels]
        per_head_mass = frame.groupby(
            ["stimulus_id", "head_rank", "layer", "head"], sort=False
        )["raw_attention_value"].sum()
        bank_mass = frame.groupby("stimulus_id", sort=False)[
            "raw_attention_value"
        ].sum()
        rows.append(
            {
                "design_variant": variant,
                "phenotype": phenotype,
                "head_count": len(labels),
                "minimum_rank": min(int(label[0]) for label in labels),
                "median_rank": float(np.median([label[0] for label in labels])),
                "maximum_rank": max(int(label[0]) for label in labels),
                "equal_head_profile_effective_number": (
                    _profile_effective_number(profiles.to_numpy(dtype=float).sum(axis=0))
                ),
                "raw_attention_ensemble_effective_number": (
                    _sample_raw_ensemble_effective(frame, pairs)
                ),
                "mean_per_head_total_endpoint_mass": float(per_head_mass.mean()),
                "mean_summed_bank_endpoint_mass": float(bank_mass.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["design_variant", "phenotype"]
    ).reset_index(drop=True)


def _plot_occurrence_profiles(detail: pd.DataFrame, output: Path) -> None:
    rank1 = detail[detail["head_rank"] == 1]
    variants = sorted(rank1["design_variant"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, variant in zip(axes.flat, variants):
        selected = rank1[rank1["design_variant"] == variant]
        for pooling, color in (("span_end", "#c43c39"), ("span_mean", "#2878b5")):
            frame = selected[selected["pooling"] == pooling]
            summary = frame.groupby("occurrence_index")["normalized_share"].agg(
                ["mean", "std", "count"]
            )
            sem = summary["std"] / np.sqrt(summary["count"])
            x = summary.index.to_numpy(dtype=int)
            y = summary["mean"].to_numpy(dtype=float)
            axis.plot(x, y, marker="o", label=pooling, color=color)
            axis.fill_between(
                x,
                y - 1.96 * sem,
                y + 1.96 * sem,
                color=color,
                alpha=0.15,
            )
        axis.set_title(variant)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Needle occurrence index")
    axes[1, 1].set_xlabel("Needle occurrence index")
    axes[0, 0].set_ylabel("Within-needle normalized share")
    axes[1, 0].set_ylabel("Within-needle normalized share")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Rank-1 Qwen span-end head: endpoint spike versus full-span mean")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_depth_profiles(summary: pd.DataFrame, output: Path) -> None:
    summary = summary[summary["head_rank"] == 1]
    figure, axis = plt.subplots(figsize=(10, 5))
    for variant, frame in summary.groupby("design_variant", sort=True):
        x = (frame["bin_start"] + frame["bin_end"]) / 2
        axis.plot(
            x,
            frame["row_mass_fraction_mean"],
            marker="o",
            markersize=3,
            label=variant,
        )
    axis.axvspan(0, 0.25, color="#777777", alpha=0.08, label="prefix quarter")
    axis.set_xlabel("Relative key depth")
    axis.set_ylabel("Fraction of full query-row attention")
    axis.set_title("Rank-1 head full-query-row depth profile")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_top_head_heatmaps(
    detail: pd.DataFrame,
    output: Path,
    *,
    top_k: int,
) -> None:
    endpoint = detail[
        (detail["pooling"] == "span_end") & (detail["head_rank"] <= int(top_k))
    ]
    variants = sorted(endpoint["design_variant"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    image = None
    for axis, variant in zip(axes.flat, variants):
        frame = endpoint[endpoint["design_variant"] == variant]
        pivot = frame.pivot_table(
            index=["head_rank", "layer", "head"],
            columns="occurrence_index",
            values="normalized_share",
            aggfunc="mean",
        ).sort_index(level="head_rank")
        image = axis.imshow(
            pivot.to_numpy(),
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="magma",
        )
        axis.set_title(variant)
        axis.set_yticks(range(len(pivot)))
        axis.set_yticklabels(
            [f"r{rank}:L{layer}H{head}" for rank, layer, head in pivot.index],
            fontsize=8,
        )
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns)
    axes[1, 0].set_xlabel("Needle occurrence index")
    axes[1, 1].set_xlabel("Needle occurrence index")
    if image is not None:
        figure.colorbar(image, ax=axes, label="Mean normalized endpoint share")
    figure.suptitle("Top-8 span-end heads: occurrence specialization")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_candidate_phenotypes(frame: pd.DataFrame, output: Path) -> None:
    counts = frame.pivot_table(
        index="design_variant",
        columns="phenotype",
        values="head_rank",
        aggfunc="count",
        fill_value=0,
    )
    axis = counts.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="tab20")
    axis.set_xlabel("Design variant")
    axis.set_ylabel("Discovery broad-candidate heads")
    axis.set_title("All Qwen span-end broad candidates by N=10 phenotype")
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_candidate_scatter(frame: pd.DataFrame, output: Path) -> None:
    variants = sorted(frame["design_variant"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    phenotype_values = sorted(frame["phenotype"].unique())
    colors = {
        phenotype: plt.get_cmap("tab10")(index % 10)
        for index, phenotype in enumerate(phenotype_values)
    }
    for axis, variant in zip(axes.flat, variants):
        selected = frame[frame["design_variant"] == variant]
        for phenotype, group in selected.groupby("phenotype", sort=True):
            axis.scatter(
                group["endpoint_effective_number_mean"],
                group["row_dominant_quartile_mass"],
                s=20,
                alpha=0.7,
                color=colors[phenotype],
                label=phenotype,
            )
        axis.axvline(6, color="#555555", linestyle="--", linewidth=1)
        axis.axhline(0.5, color="#555555", linestyle=":", linewidth=1)
        axis.set_title(variant)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Endpoint effective number at N=10")
    axes[1, 1].set_xlabel("Endpoint effective number at N=10")
    axes[0, 0].set_ylabel("Dominant full-row quartile mass")
    axes[1, 0].set_ylabel("Dominant full-row quartile mass")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="lower center", ncol=3)
    figure.suptitle("Broad candidates: endpoint breadth versus spatial gating")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_head_bank(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[frame["strategy"] != "all_broad_candidates"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for axis, metric in zip(
        axes,
        (
            "equal_head_profile_effective_number",
            "raw_attention_ensemble_effective_number",
        ),
    ):
        for (variant, strategy), group in selected.groupby(
            ["design_variant", "strategy"], sort=True
        ):
            linestyle = "-" if strategy == "greedy_complement_from_rank1" else "--"
            axis.plot(
                group["head_count"],
                group[metric],
                marker="o",
                linestyle=linestyle,
                label=f"{variant} / {strategy}",
            )
        axis.set_xlabel("Heads in ensemble")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Effective number across 10 needle endpoints")
    axes[0].set_title("Equalized head profiles")
    axes[1].set_title("Raw attention-weighted ensemble")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="lower center", ncol=2)
    figure.suptitle("Do multiple broad candidates cover complementary partitions?")
    figure.tight_layout(rect=(0, 0.18, 1, 1))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_phenotype_bank(frame: pd.DataFrame, output: Path) -> None:
    variants = sorted(frame["design_variant"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for axis, variant in zip(axes.flat, variants):
        selected = frame[frame["design_variant"] == variant].sort_values(
            "raw_attention_ensemble_effective_number"
        )
        positions = np.arange(len(selected))
        axis.barh(
            positions - 0.18,
            selected["equal_head_profile_effective_number"],
            height=0.36,
            label="equal head profiles",
            color="#2878b5",
        )
        axis.barh(
            positions + 0.18,
            selected["raw_attention_ensemble_effective_number"],
            height=0.36,
            label="raw attention weighted",
            color="#c43c39",
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(selected["phenotype"], fontsize=8)
        axis.set_xlim(0, 10.2)
        axis.set_title(variant)
        axis.grid(axis="x", alpha=0.25)
    axes[1, 0].set_xlabel("Effective number across 10 needle endpoints")
    axes[1, 1].set_xlabel("Effective number across 10 needle endpoints")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="lower center", ncol=2)
    figure.suptitle("Joint endpoint coverage of all heads in each phenotype")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_partitioned_attention(
    *,
    attention_index_path: str | Path,
    encodings: Iterable[PromptEncoding],
    rankings_dir: str | Path,
    output_dir: str | Path,
    design_variants: Sequence[str],
    count: int = 10,
    top_k: int = 8,
    all_candidates: bool = True,
    partitions: int = 4,
    depth_bins: int = 20,
    bootstrap_repetitions: int = 10_000,
    random_seed: int = 20260731,
) -> dict[str, Path]:
    """Test whether top span-end heads are selectors or local aggregators."""

    index_path = Path(attention_index_path)
    capture_root = index_path.parent
    ranking_root = Path(rankings_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoding_by_id = {item.stimulus_id: item for item in encodings}
    records = [
        row
        for row in _read_jsonl(index_path)
        if int(row["count"]) == int(count)
        and str(row["design_variant"]) in set(design_variants)
    ]
    expected_ids = set(encoding_by_id)
    observed_ids = {str(row["stimulus_id"]) for row in records}
    if observed_ids != expected_ids:
        raise ValueError(
            "Selected attention and prompt grids disagree: "
            f"attention={len(observed_ids)} encodings={len(expected_ids)}"
        )
    rankings = {
        variant: _ranking_heads(
            ranking_root / f"{variant.replace('.', '_')}_span_end.json",
            top_k=top_k,
            all_candidates=all_candidates,
        )
        for variant in design_variants
    }

    occurrence_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        stimulus_id = str(record["stimulus_id"])
        encoding = encoding_by_id[stimulus_id]
        variant = str(record["design_variant"])
        raw_value = record.get("raw_attention_shard_path")
        if not raw_value:
            raise ValueError(f"Raw query row is missing for {stimulus_id}")
        raw_path = capture_root / str(raw_value)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        normalized_depths = np.asarray(
            [
                (span.start + span.end - 1) / (2 * encoding.query_position)
                for span in encoding.needle_spans
            ],
            dtype=np.float64,
        )
        heads = rankings[variant]
        with np.load(raw_path, allow_pickle=False) as raw:
            key_starts = np.asarray(raw["key_starts"], dtype=int)
            loaded_layers = {
                layer: np.asarray(raw[f"layer_{layer:03d}"], dtype=np.float32)
                for layer in sorted({layer for _rank, layer, _head in heads})
            }
            for head_rank, layer, head in heads:
                row = loaded_layers[layer][head]
                key_start = int(key_starts[layer])
                binned = depth_bin_masses(row, bins=depth_bins)
                row_total = float(row.sum(dtype=np.float64))
                for depth_bin, mass in enumerate(binned):
                    depth_rows.append(
                        {
                            "stimulus_id": stimulus_id,
                            "model_label": encoding.model_label,
                            "design_variant": variant,
                            "seed": int(encoding.seed),
                            "split": encoding.split,
                            "count": int(encoding.count),
                            "head_rank": head_rank,
                            "layer": layer,
                            "head": head,
                            "depth_bin": depth_bin,
                            "bin_start": depth_bin / depth_bins,
                            "bin_end": (depth_bin + 1) / depth_bins,
                            "row_mass": float(mass),
                            "row_mass_fraction": (
                                float(mass / row_total) if row_total > 0 else 0.0
                            ),
                        }
                    )
                for pooling in POOLINGS:
                    values = occurrence_attention_values(
                        row,
                        encoding.needle_spans,
                        key_start=key_start,
                        pooling=pooling,
                    )
                    metrics = partition_sample_metrics(
                        values,
                        normalized_depths,
                        partitions=partitions,
                    )
                    base = {
                        "stimulus_id": stimulus_id,
                        "model_label": encoding.model_label,
                        "design_variant": variant,
                        "seed": int(encoding.seed),
                        "split": encoding.split,
                        "count": int(encoding.count),
                        "pooling": pooling,
                        "head_rank": head_rank,
                        "layer": layer,
                        "head": head,
                    }
                    sample_rows.append(
                        {
                            **base,
                            **{
                                key: value
                                for key, value in metrics.items()
                                if key != "normalized_shares"
                            },
                        }
                    )
                    shares = np.asarray(metrics["normalized_shares"], dtype=float)
                    for occurrence_index, (span, value, share, depth) in enumerate(
                        zip(
                            encoding.needle_spans,
                            values,
                            shares,
                            normalized_depths,
                        ),
                        start=1,
                    ):
                        occurrence_rows.append(
                            {
                                **base,
                                "occurrence_index": occurrence_index,
                                "slot_index": int(span.slot_index),
                                "normalized_depth": float(depth),
                                "raw_attention_value": float(value),
                                "normalized_share": float(share),
                            }
                        )
        if record_index % 10 == 0 or record_index == len(records):
            print(
                "[v4 partition analysis] "
                f"{record_index}/{len(records)} {stimulus_id}",
                flush=True,
            )

    occurrences = pd.DataFrame(occurrence_rows)
    samples = pd.DataFrame(sample_rows)
    depth_detail = pd.DataFrame(depth_rows)
    summary = _head_summary(
        samples,
        occurrences,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_top_k=top_k,
        random_seed=random_seed,
    )
    depth_summary = _depth_summary(
        depth_detail,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_top_k=1,
        random_seed=random_seed + 1,
    )
    assessment = assess_rank1_partitioning(
        summary,
        depth_detail,
        depth_bins=depth_bins,
    )
    phenotypes = classify_candidate_heads(
        summary,
        depth_detail,
        depth_bins=depth_bins,
    )
    phenotype_counts = (
        phenotypes.groupby(["design_variant", "phenotype"], sort=True)
        .size()
        .rename("heads")
        .reset_index()
    )
    head_bank = head_bank_coverage(
        occurrences,
        diagnostic_top_k=top_k,
    )
    phenotype_bank = phenotype_bank_coverage(occurrences, phenotypes)

    split_summaries: list[pd.DataFrame] = []
    split_phenotypes: list[pd.DataFrame] = []
    split_head_banks: list[pd.DataFrame] = []
    split_phenotype_banks: list[pd.DataFrame] = []
    for split in sorted(samples["split"].unique()):
        split_samples = samples[samples["split"] == split]
        split_occurrences = occurrences[occurrences["split"] == split]
        split_depth = depth_detail[depth_detail["split"] == split]
        split_summary = _head_summary(
            split_samples,
            split_occurrences,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_top_k=top_k,
            random_seed=random_seed + 10_000,
        )
        split_phenotype = classify_candidate_heads(
            split_summary,
            split_depth,
            depth_bins=depth_bins,
        )
        split_head_bank = head_bank_coverage(
            split_occurrences,
            diagnostic_top_k=top_k,
        )
        split_phenotype_bank = phenotype_bank_coverage(
            split_occurrences,
            split_phenotype,
        )
        for frame in (
            split_summary,
            split_phenotype,
            split_head_bank,
            split_phenotype_bank,
        ):
            frame.insert(1, "split", split)
        split_summaries.append(split_summary)
        split_phenotypes.append(split_phenotype)
        split_head_banks.append(split_head_bank)
        split_phenotype_banks.append(split_phenotype_bank)
    summary_by_split = pd.concat(split_summaries, ignore_index=True)
    phenotypes_by_split = pd.concat(split_phenotypes, ignore_index=True)
    phenotype_counts_by_split = (
        phenotypes_by_split.groupby(
            ["split", "design_variant", "phenotype"], sort=True
        )
        .size()
        .rename("heads")
        .reset_index()
    )
    head_bank_by_split = pd.concat(split_head_banks, ignore_index=True)
    phenotype_bank_by_split = pd.concat(
        split_phenotype_banks, ignore_index=True
    )

    occurrence_path = output / "occurrence_detail.csv.gz"
    sample_path = output / "head_sample_detail.csv.gz"
    summary_path = output / "head_partition_summary.csv"
    depth_detail_path = output / "all_candidate_full_row_depth_detail.csv.gz"
    depth_summary_path = output / "all_candidate_full_row_depth_summary.csv"
    assessment_path = output / "partition_hypothesis_assessment.json"
    phenotype_path = output / "all_candidate_head_phenotypes.csv"
    phenotype_count_path = output / "all_candidate_phenotype_counts.csv"
    head_bank_path = output / "multi_head_partition_coverage.csv"
    phenotype_bank_path = output / "phenotype_bank_coverage.csv"
    summary_by_split_path = output / "head_partition_summary_by_split.csv"
    phenotype_by_split_path = output / "all_candidate_head_phenotypes_by_split.csv"
    phenotype_count_by_split_path = (
        output / "all_candidate_phenotype_counts_by_split.csv"
    )
    head_bank_by_split_path = output / "multi_head_partition_coverage_by_split.csv"
    phenotype_bank_by_split_path = output / "phenotype_bank_coverage_by_split.csv"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    occurrence_figure = figures / "rank1_endpoint_vs_span_mean_by_occurrence.png"
    depth_figure = figures / "rank1_full_row_depth_profile.png"
    heatmap_figure = figures / "top8_span_end_occurrence_specialization.png"
    phenotype_figure = figures / "all_candidate_phenotype_counts.png"
    scatter_figure = figures / "all_candidate_breadth_vs_spatial_gating.png"
    head_bank_figure = figures / "multi_head_partition_coverage.png"
    phenotype_bank_figure = figures / "phenotype_bank_coverage.png"
    _write_csv_atomic(occurrences, occurrence_path, gzip=True)
    _write_csv_atomic(samples, sample_path, gzip=True)
    _write_csv_atomic(summary, summary_path)
    _write_csv_atomic(depth_detail, depth_detail_path, gzip=True)
    _write_csv_atomic(depth_summary, depth_summary_path)
    _write_json_atomic(assessment, assessment_path)
    _write_csv_atomic(phenotypes, phenotype_path)
    _write_csv_atomic(phenotype_counts, phenotype_count_path)
    _write_csv_atomic(head_bank, head_bank_path)
    _write_csv_atomic(phenotype_bank, phenotype_bank_path)
    _write_csv_atomic(summary_by_split, summary_by_split_path)
    _write_csv_atomic(phenotypes_by_split, phenotype_by_split_path)
    _write_csv_atomic(phenotype_counts_by_split, phenotype_count_by_split_path)
    _write_csv_atomic(head_bank_by_split, head_bank_by_split_path)
    _write_csv_atomic(phenotype_bank_by_split, phenotype_bank_by_split_path)
    _plot_occurrence_profiles(occurrences, occurrence_figure)
    _plot_depth_profiles(depth_summary, depth_figure)
    _plot_top_head_heatmaps(occurrences, heatmap_figure, top_k=top_k)
    _plot_candidate_phenotypes(phenotypes, phenotype_figure)
    _plot_candidate_scatter(phenotypes, scatter_figure)
    _plot_head_bank(head_bank, head_bank_figure)
    _plot_phenotype_bank(phenotype_bank, phenotype_bank_figure)

    outputs = {
        "occurrence_detail": occurrence_path,
        "head_sample_detail": sample_path,
        "head_partition_summary": summary_path,
        "all_candidate_full_row_depth_detail": depth_detail_path,
        "all_candidate_full_row_depth_summary": depth_summary_path,
        "assessment": assessment_path,
        "all_candidate_phenotypes": phenotype_path,
        "all_candidate_phenotype_counts": phenotype_count_path,
        "multi_head_partition_coverage": head_bank_path,
        "phenotype_bank_coverage": phenotype_bank_path,
        "head_partition_summary_by_split": summary_by_split_path,
        "all_candidate_phenotypes_by_split": phenotype_by_split_path,
        "all_candidate_phenotype_counts_by_split": phenotype_count_by_split_path,
        "multi_head_partition_coverage_by_split": head_bank_by_split_path,
        "phenotype_bank_coverage_by_split": phenotype_bank_by_split_path,
        "occurrence_figure": occurrence_figure,
        "depth_figure": depth_figure,
        "top8_heatmap": heatmap_figure,
        "phenotype_figure": phenotype_figure,
        "candidate_scatter": scatter_figure,
        "head_bank_figure": head_bank_figure,
        "phenotype_bank_figure": phenotype_bank_figure,
    }
    _write_json_atomic(
        {
            "schema_version": "realistic_niah_v4_partition_analysis_manifest_v1",
            "model_label": str(samples["model_label"].iloc[0]),
            "design_variants": list(design_variants),
            "count": int(count),
            "top_k": int(top_k),
            "candidate_scope": "all_discovery_broad_candidates"
            if all_candidates
            else "registered_top_k",
            "candidate_counts": {
                variant: len(rankings[variant]) for variant in design_variants
            },
            "partitions": int(partitions),
            "depth_bins": int(depth_bins),
            "bootstrap_repetitions": int(bootstrap_repetitions),
            "examples": int(samples["stimulus_id"].nunique()),
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        output / "partition_analysis_manifest.json",
    )
    outputs["manifest"] = output / "partition_analysis_manifest.json"
    return outputs
