"""Registered position-wise attention-head comparisons for V5.

The module deliberately separates selection from evaluation:

* V5 targeted-retrieval heads are ranked on discovery rows only.
* The V4.4 first-locator ranking is loaded from its frozen discovery registry.
* Position and bank contrasts are evaluated on confirmation rows and clustered
  by seed.  Exact prompt-needle raw and relative attention masses are retained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


SCHEMA_VERSION = "realistic_niah_v5_position_head_analysis_v1"
REQUIRED_COLUMNS = {
    "request_id",
    "model_label",
    "seed",
    "split",
    "gold_count",
    "trace_one_to_one",
    "mechanism",
    "occurrence",
    "layer",
    "head",
    "target_needle_raw_mass",
    "target_needle_relative_mass",
    "target_needle_top1",
}
MASS_COLUMNS = (
    "target_needle_raw_mass",
    "target_needle_relative_mass",
)


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


def _truth_mask(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy()
    return values.astype(str).str.lower().isin({"true", "1"}).to_numpy()


def _validate_attention(attention: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(attention.columns))
    if missing:
        raise ValueError(f"Attention table is missing columns: {missing}")
    frame = attention.copy()
    frame = frame.loc[
        frame["mechanism"].astype(str).eq("targeted_retrieval")
        & _truth_mask(frame["trace_one_to_one"])
    ].copy()
    if frame.empty:
        raise ValueError("No one-to-one targeted-retrieval rows are available")
    frame["occurrence"] = frame["occurrence"].astype(int)
    for column in MASS_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(MASS_COLUMNS)].isna().all(axis=None):
        raise ValueError("Exact target-needle mass columns contain no finite values")
    return frame


def rank_position_heads(
    attention: pd.DataFrame,
    *,
    region: str,
    split: str = "discovery",
) -> pd.DataFrame:
    """Rank targeted-retrieval heads separately at the first/later positions."""

    if region not in {"first", "later", "all"}:
        raise ValueError(f"Unknown targeted-retrieval region: {region}")
    frame = _validate_attention(attention)
    mask = frame["split"].astype(str).str.lower().eq(split.lower())
    if region == "first":
        # Match the support of the later-position comparison: N=1 requests
        # have no within-request later position and therefore cannot identify
        # a first-versus-later difference.
        mask &= frame["occurrence"].eq(1) & frame["gold_count"].astype(int).ge(2)
    elif region == "later":
        mask &= frame["occurrence"].ge(2)
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No {split}/{region} targeted-retrieval rows")
    # Give every request equal weight.  Without this reduction an N=10 request
    # contributes nine later queries while an N=2 request contributes one.
    request_weighted = (
        selected.groupby(
            ["request_id", "model_label", "seed", "layer", "head"],
            as_index=False,
        )
        .agg(
            target_needle_raw_mass=("target_needle_raw_mass", "mean"),
            target_needle_relative_mass=("target_needle_relative_mass", "mean"),
            target_needle_top1=("target_needle_top1", "mean"),
            request_queries=("occurrence", "size"),
        )
    )
    grouped = (
        request_weighted.groupby(["model_label", "layer", "head"], as_index=False)
        .agg(
            target_needle_raw_mass=("target_needle_raw_mass", "mean"),
            target_needle_relative_mass=("target_needle_relative_mass", "mean"),
            target_needle_top1=("target_needle_top1", "mean"),
            n_queries=("request_queries", "sum"),
            n_requests=("request_id", "size"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(
            ["model_label", "target_needle_raw_mass", "layer", "head"],
            ascending=[True, False, True, True],
        )
        .reset_index(drop=True)
    )
    grouped["rank"] = grouped.groupby("model_label").cumcount() + 1
    grouped["region"] = region
    grouped["selection_split"] = split
    grouped["selection_metric"] = "request_weighted_mean_target_needle_raw_mass"
    return grouped


def load_first_locator_heads(path: str | Path) -> tuple[str, list[tuple[int, int]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    definition = str(payload.get("first_locator_definition", ""))
    expected = "first needle span mass"
    if expected not in definition.lower():
        raise ValueError(
            "Registry does not contain the frozen V4.4 first-locator definition"
        )
    values = payload.get("rankings", {}).get("first_locator")
    if not isinstance(values, list) or not values:
        raise ValueError("Registry has no first_locator ranking")
    ordered = sorted(values, key=lambda row: int(row["rank"]))
    heads = [(int(row["layer"]), int(row["head"])) for row in ordered]
    if len(heads) != len(set(heads)):
        raise ValueError("First-locator registry contains duplicate heads")
    return str(payload.get("model_label", "")), heads


def _heads(frame: pd.DataFrame, size: int) -> list[tuple[int, int]]:
    ordered = frame.sort_values("rank").head(int(size))
    return [(int(row.layer), int(row.head)) for row in ordered.itertuples()]


def _head_mask(frame: pd.DataFrame, heads: Iterable[tuple[int, int]]) -> np.ndarray:
    wanted = {(int(layer), int(head)) for layer, head in heads}
    return np.fromiter(
        (
            (int(layer), int(head)) in wanted
            for layer, head in zip(frame["layer"], frame["head"])
        ),
        dtype=bool,
        count=len(frame),
    )


def _bootstrap_seed_mean(
    values: pd.DataFrame,
    *,
    value_column: str,
    samples: int,
    random_state: int,
) -> dict[str, Any]:
    by_seed = values.groupby("seed", as_index=False)[value_column].mean()
    finite = by_seed[value_column].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_seeds": 0,
        }
    rng = np.random.default_rng(random_state)
    draws = finite[
        rng.integers(0, len(finite), size=(int(samples), len(finite)))
    ].mean(axis=1)
    return {
        "estimate": float(finite.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_seeds": int(len(finite)),
    }


def position_contrasts(
    attention: pd.DataFrame,
    *,
    ranked_heads: pd.DataFrame,
    bank_sizes: Sequence[int],
    bootstrap_samples: int,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate k=1 minus the within-request mean of k>=2 on confirmation."""

    frame = _validate_attention(attention)
    frame = frame.loc[
        frame["split"].astype(str).str.lower().eq("confirmation")
    ].copy()
    detail_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for size in bank_sizes:
        selected_heads = _heads(ranked_heads, int(size))
        selected = frame.loc[_head_mask(frame, selected_heads)].copy()
        for metric in MASS_COLUMNS:
            per_request = (
                selected.assign(
                    region=np.where(selected["occurrence"].eq(1), "first", "later")
                )
                .groupby(["request_id", "seed", "region"], as_index=False)[metric]
                .mean()
                .pivot(index=["request_id", "seed"], columns="region", values=metric)
                .reset_index()
            )
            if not {"first", "later"}.issubset(per_request.columns):
                continue
            per_request["difference_first_minus_later"] = (
                per_request["first"] - per_request["later"]
            )
            per_request["metric"] = metric
            per_request["bank_size"] = int(size)
            per_request["bank"] = "targeted_first_discovery_ranked"
            detail_rows.append(per_request)
            stats = _bootstrap_seed_mean(
                per_request,
                value_column="difference_first_minus_later",
                samples=bootstrap_samples,
                random_state=random_state + int(size),
            )
            summary_rows.append(
                {
                    "comparison": "first_minus_later",
                    "bank": "targeted_first_discovery_ranked",
                    "bank_size": int(size),
                    "metric": metric,
                    "n_requests": int(len(per_request)),
                    **stats,
                }
            )
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return detail, pd.DataFrame(summary_rows)


def bank_comparisons(
    attention: pd.DataFrame,
    *,
    targeted_first_ranking: pd.DataFrame,
    first_locator_heads: Sequence[tuple[int, int]],
    bank_sizes: Sequence[int],
    bootstrap_samples: int,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare V5 targeted-first and frozen V4.4 first-locator banks at k=1."""

    frame = _validate_attention(attention)
    frame = frame.loc[
        frame["split"].astype(str).str.lower().eq("confirmation")
        & frame["occurrence"].eq(1)
    ].copy()
    detail_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for size in bank_sizes:
        targeted = _heads(targeted_first_ranking, int(size))
        locator = list(first_locator_heads[: int(size)])
        banks = {"targeted_first": targeted, "first_locator": locator}
        for metric in MASS_COLUMNS:
            wide_parts = []
            raw_distributions: dict[str, np.ndarray] = {}
            for bank_name, heads in banks.items():
                selected = frame.loc[_head_mask(frame, heads)].copy()
                raw_distributions[bank_name] = selected[metric].to_numpy(dtype=float)
                aggregate = (
                    selected.groupby(["request_id", "seed"], as_index=False)[metric]
                    .mean()
                    .rename(columns={metric: bank_name})
                )
                wide_parts.append(aggregate)
                values = raw_distributions[bank_name]
                values = values[np.isfinite(values)]
                distribution_rows.append(
                    {
                        "bank_size": int(size),
                        "metric": metric,
                        "bank": bank_name,
                        "n_head_queries": int(len(values)),
                        "mean": float(np.mean(values)) if len(values) else np.nan,
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                        "q05": float(np.quantile(values, 0.05)) if len(values) else np.nan,
                        "q50": float(np.quantile(values, 0.50)) if len(values) else np.nan,
                        "q95": float(np.quantile(values, 0.95)) if len(values) else np.nan,
                    }
                )
            paired = wide_parts[0].merge(wide_parts[1], on=["request_id", "seed"])
            paired["difference_targeted_minus_locator"] = (
                paired["targeted_first"] - paired["first_locator"]
            )
            paired["metric"] = metric
            paired["bank_size"] = int(size)
            detail_rows.append(paired)
            stats = _bootstrap_seed_mean(
                paired,
                value_column="difference_targeted_minus_locator",
                samples=bootstrap_samples,
                random_state=random_state + 10_000 + int(size),
            )
            left = raw_distributions["targeted_first"]
            right = raw_distributions["first_locator"]
            left = left[np.isfinite(left)]
            right = right[np.isfinite(right)]
            summary_rows.append(
                {
                    "comparison": "targeted_first_minus_first_locator",
                    "bank_size": int(size),
                    "metric": metric,
                    "n_requests": int(len(paired)),
                    "wasserstein_distance_descriptive": (
                        float(wasserstein_distance(left, right))
                        if len(left) and len(right)
                        else np.nan
                    ),
                    "ks_statistic_descriptive": (
                        float(ks_2samp(left, right).statistic)
                        if len(left) and len(right)
                        else np.nan
                    ),
                    **stats,
                }
            )
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return detail, pd.DataFrame(summary_rows), pd.DataFrame(distribution_rows)


def ranking_overlap(
    first: pd.DataFrame,
    later: pd.DataFrame,
    first_locator_heads: Sequence[tuple[int, int]],
    *,
    bank_sizes: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    merged = first.merge(
        later,
        on=["model_label", "layer", "head"],
        suffixes=("_first", "_later"),
    )
    correlation = spearmanr(
        merged["target_needle_raw_mass_first"],
        merged["target_needle_raw_mass_later"],
    ).statistic
    max_layer = int(max(first["layer"].max(), later["layer"].max()))
    for size in bank_sizes:
        banks = {
            "targeted_later": set(_heads(later, int(size))),
            "first_locator": set(first_locator_heads[: int(size)]),
        }
        first_set = set(_heads(first, int(size)))
        for comparison, other in banks.items():
            union = first_set | other
            overlap = first_set & other
            left_hist = np.bincount(
                [layer for layer, _head in first_set], minlength=max_layer + 1
            ).astype(float)
            right_hist = np.bincount(
                [layer for layer, _head in other], minlength=max_layer + 1
            ).astype(float)
            if left_hist.sum():
                left_hist /= left_hist.sum()
            if right_hist.sum():
                right_hist /= right_hist.sum()
            rows.append(
                {
                    "comparison": f"targeted_first_vs_{comparison}",
                    "bank_size": int(size),
                    "overlap_heads": int(len(overlap)),
                    "jaccard": float(len(overlap) / len(union)) if union else np.nan,
                    "layer_js_distance": float(jensenshannon(left_hist, right_hist)),
                    "all_head_score_spearman_first_vs_later": (
                        float(correlation)
                        if comparison == "targeted_later"
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze_position_heads(
    attention_csv: str | Path,
    first_locator_registry: str | Path,
    output_dir: str | Path,
    *,
    bank_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32),
    bootstrap_samples: int = 5000,
    random_state: int = 0,
) -> dict[str, Path]:
    attention = pd.read_csv(attention_csv)
    frame = _validate_attention(attention)
    models = sorted(frame["model_label"].astype(str).unique())
    if len(models) != 1:
        raise ValueError("Position-head analysis expects exactly one model")
    registry_model, locator_heads = load_first_locator_heads(first_locator_registry)
    if registry_model and registry_model != models[0]:
        raise ValueError(
            f"First-locator registry is for {registry_model}, attention is {models[0]}"
        )
    available_heads = int(frame[["layer", "head"]].drop_duplicates().shape[0])
    sizes = sorted(
        {
            int(size)
            for size in bank_sizes
            if 0 < int(size) <= min(available_heads, len(locator_heads))
        }
    )
    if not sizes:
        raise ValueError("No requested bank size is estimable")
    first = rank_position_heads(frame, region="first")
    later = rank_position_heads(frame, region="later")
    all_positions = rank_position_heads(frame, region="all")
    position_detail, position_summary = position_contrasts(
        frame,
        ranked_heads=first,
        bank_sizes=sizes,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    bank_detail, bank_summary, distributions = bank_comparisons(
        frame,
        targeted_first_ranking=first,
        first_locator_heads=locator_heads,
        bank_sizes=sizes,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    overlap = ranking_overlap(
        first, later, locator_heads, bank_sizes=sizes
    )
    output = Path(output_dir)
    paths = {
        "rankings": output / "position_head_rankings.csv",
        "ranking_overlap": output / "ranking_overlap.csv",
        "position_detail": output / "first_vs_later_confirmation_detail.csv",
        "position_summary": output / "first_vs_later_confirmation_summary.csv",
        "bank_detail": output / "targeted_vs_first_locator_confirmation_detail.csv",
        "bank_summary": output / "targeted_vs_first_locator_confirmation_summary.csv",
        "bank_distributions": output / "targeted_vs_first_locator_distributions.csv",
        "audit": output / "position_head_analysis_audit.json",
    }
    _atomic_csv(paths["rankings"], pd.concat([first, later, all_positions]))
    _atomic_csv(paths["ranking_overlap"], overlap)
    _atomic_csv(paths["position_detail"], position_detail)
    _atomic_csv(paths["position_summary"], position_summary)
    _atomic_csv(paths["bank_detail"], bank_detail)
    _atomic_csv(paths["bank_summary"], bank_summary)
    _atomic_csv(paths["bank_distributions"], distributions)
    _atomic_json(
        paths["audit"],
        {
            "schema_version": SCHEMA_VERSION,
            "model_label": models[0],
            "attention_csv": str(Path(attention_csv).resolve()),
            "first_locator_registry": str(Path(first_locator_registry).resolve()),
            "targeted_head_selection_split": "discovery",
            "first_locator_selection_split": "V4.4 non-thinking discovery",
            "evaluation_split": "confirmation",
            "cohort": "trace_one_to_one",
            "position_estimand": "within-request k=1 minus mean(k>=2)",
            "mass_contract": list(MASS_COLUMNS),
            "bank_sizes": sizes,
            "bootstrap_unit": "seed",
            "bootstrap_samples": int(bootstrap_samples),
            "ks_and_wasserstein_status": "descriptive_only; clustered bootstrap contrast is inferential",
            "confirmation_used_for_selection": False,
        },
    )
    return paths
