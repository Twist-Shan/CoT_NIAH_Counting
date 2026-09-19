from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "realistic_niah_v4_prompt_counter_dynamics_v1"
MODELS = ("Qwen3-8B", "Gemma4-E4B")
POOLING_SPECS = {
    "span_end": {
        "key_pooling": "needle_end",
        "total": "needle_end_total_mass",
        "effective": "needle_end_effective_number",
        "coverage": "needle_end_relative_coverage",
        "current_share": "needle_end_current_share",
        "profile": "needle_end_mass",
    },
    "span_mean": {
        "key_pooling": "needle_span_sum",
        "total": "needle_span_total_mass",
        "effective": "needle_span_effective_number",
        "coverage": "needle_span_relative_coverage",
        "current_share": "needle_span_current_share",
        "profile": "needle_span_mass",
    },
}
ATTENTION_METRICS = (
    "row_effective_tokens",
    "row_effective_fraction",
    "row_top1_mass",
    "needle_total_mass",
    "needle_effective_number",
    "needle_relative_coverage",
    "needle_current_share",
)
ASSOCIATION_METRICS = (
    "row_effective_fraction",
    "needle_effective_number",
    "needle_relative_coverage",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _write_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if gzip else ".tmp"
    temporary = path.with_name(path.name + suffix)
    frame.to_csv(
        temporary,
        index=False,
        compression="gzip" if gzip else None,
    )
    temporary.replace(path)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(int(replicates), len(values)))
    estimates = values[draws].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) <= 0:
        return math.nan
    centered = x[valid] - x[valid].mean()
    return float(np.sum(centered * (y[valid] - y[valid].mean())) / np.sum(centered**2))


def _residualized_correlation(frame: pd.DataFrame, metric: str) -> float:
    """Correlation after removing occurrence fixed effects."""

    selected = frame[["query_occurrence", metric, "counter_noise"]].dropna().copy()
    if len(selected) < 3:
        return math.nan
    selected["attention_residual"] = selected[metric] - selected.groupby(
        "query_occurrence"
    )[metric].transform("mean")
    selected["noise_residual"] = selected["counter_noise"] - selected.groupby(
        "query_occurrence"
    )["counter_noise"].transform("mean")
    if (
        selected["attention_residual"].std(ddof=0) <= 0
        or selected["noise_residual"].std(ddof=0) <= 0
    ):
        return math.nan
    return float(
        np.corrcoef(
            selected["attention_residual"], selected["noise_residual"]
        )[0, 1]
    )


def _bootstrap_seed_correlation(
    frame: pd.DataFrame,
    metric: str,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    seeds = np.asarray(sorted(frame["seed"].astype(int).unique()), dtype=int)
    if len(seeds) < 2:
        return math.nan, math.nan
    attention = (
        frame.pivot(index="seed", columns="query_occurrence", values=metric)
        .reindex(index=seeds, columns=range(1, 11))
        .to_numpy(dtype=float)
    )
    noise = (
        frame.pivot(
            index="seed", columns="query_occurrence", values="counter_noise"
        )
        .reindex(index=seeds, columns=range(1, 11))
        .to_numpy(dtype=float)
    )
    if not np.isfinite(attention).all() or not np.isfinite(noise).all():
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(
        0, len(seeds), size=(int(replicates), len(seeds))
    )
    attention_draws = attention[draws]
    noise_draws = noise[draws]
    attention_residual = attention_draws - attention_draws.mean(
        axis=1, keepdims=True
    )
    noise_residual = noise_draws - noise_draws.mean(axis=1, keepdims=True)
    numerator = np.sum(attention_residual * noise_residual, axis=(1, 2))
    denominator = np.sqrt(
        np.sum(attention_residual**2, axis=(1, 2))
        * np.sum(noise_residual**2, axis=(1, 2))
    )
    estimates = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return math.nan, math.nan
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _primary_layers(model_root: Path) -> dict[str, int]:
    summary = _read_json(
        model_root / "representation" / "analysis" / "representation_summary.json"
    )
    return {
        str(pooling): int(layer)
        for pooling, layer in summary["primary_layer_selection"]["layers"].items()
    }


def _behavior_lookup(model_root: Path) -> dict[tuple[str, int], str]:
    labels = pd.read_csv(
        model_root / "behavior" / "capture" / "generation_labels.csv"
    )
    labels = labels[pd.to_numeric(labels["gold_count"]).astype(int) == 10]
    result: dict[tuple[str, int], str] = {}
    for row in labels.to_dict("records"):
        if _as_bool(row["is_correct"]):
            outcome = "correct"
        elif _as_bool(row["format_valid"]):
            outcome = "wrong"
        else:
            outcome = "invalid"
        result[(str(row["design_variant"]), int(row["seed"]))] = outcome
    return result


def _load_attention_metrics(model_root: Path, model: str) -> pd.DataFrame:
    capture_root = (
        model_root / "representation" / "prompt_counter_attention_v1"
    )
    records = _read_jsonl(capture_root / "capture_index.jsonl")
    frames: list[pd.DataFrame] = []
    usecols = [
        "design_variant",
        "seed",
        "split",
        "query_occurrence",
        "layer",
        "head",
        "row_effective_tokens",
        "row_effective_fraction",
        "row_top1_mass",
        "needle_end_total_mass",
        "needle_end_effective_number",
        "needle_end_relative_coverage",
        "needle_end_current_share",
        "needle_span_total_mass",
        "needle_span_effective_number",
        "needle_span_relative_coverage",
        "needle_span_current_share",
    ]
    for record in records:
        frames.append(
            pd.read_csv(capture_root / str(record["shard_path"]), usecols=usecols)
        )
    if len(frames) != 120:
        raise RuntimeError(f"{model}: expected 120 prompt-attention metric shards")
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "model", model)
    return frame


def _select_head_banks(
    metrics: pd.DataFrame,
    *,
    primary_layers: dict[str, int],
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    discovery = metrics[
        (metrics["split"] == "discovery")
        & (pd.to_numeric(metrics["query_occurrence"]) >= 2)
    ].copy()
    for pooling, spec in POOLING_SPECS.items():
        discovery["broad_score"] = (
            pd.to_numeric(discovery[spec["total"]])
            * pd.to_numeric(discovery[spec["coverage"]])
        )
        grouped = (
            discovery.groupby(
                ["model", "design_variant", "layer", "head"], as_index=False
            )["broad_score"]
            .mean()
            .sort_values(
                ["model", "design_variant", "layer", "broad_score", "head"],
                ascending=[True, True, True, False, True],
            )
        )
        grouped["rank"] = grouped.groupby(
            ["model", "design_variant", "layer"]
        ).cumcount() + 1
        grouped = grouped[grouped["rank"] <= int(top_n)]
        for row in grouped.to_dict("records"):
            rows.append(
                {
                    **row,
                    "hidden_pooling": pooling,
                    "key_pooling": spec["key_pooling"],
                    "selection_split": "discovery",
                    "selection_occurrences": "2-10",
                    "score_definition": "mean(total_needle_mass*relative_coverage)",
                    "top_n": int(top_n),
                    "primary_probe_layer": bool(
                        int(row["layer"]) == int(primary_layers[pooling])
                    ),
                }
            )
    return pd.DataFrame(rows)


def _bank_sample_metrics(
    metrics: pd.DataFrame,
    banks: pd.DataFrame,
    outcomes: dict[tuple[str, int], str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for pooling, spec in POOLING_SPECS.items():
        selected = banks[banks["hidden_pooling"] == pooling][
            ["model", "design_variant", "layer", "head", "rank"]
        ]
        merged = metrics.merge(
            selected,
            on=["model", "design_variant", "layer", "head"],
            how="inner",
            validate="many_to_one",
        )
        renamed = merged.rename(
            columns={
                spec["total"]: "needle_total_mass",
                spec["effective"]: "needle_effective_number",
                spec["coverage"]: "needle_relative_coverage",
                spec["current_share"]: "needle_current_share",
            }
        )
        keys = [
            "model",
            "design_variant",
            "seed",
            "split",
            "query_occurrence",
            "layer",
        ]
        aggregated = renamed.groupby(keys, as_index=False)[
            list(ATTENTION_METRICS)
        ].mean()
        aggregated["hidden_pooling"] = pooling
        aggregated["key_pooling"] = spec["key_pooling"]
        aggregated["outcome"] = [
            outcomes[(str(variant), int(seed))]
            for variant, seed in zip(
                aggregated["design_variant"], aggregated["seed"]
            )
        ]
        pieces.append(aggregated)
    return pd.concat(pieces, ignore_index=True)


def _all_head_sample_metrics(
    metrics: pd.DataFrame,
    outcomes: dict[tuple[str, int], str],
) -> pd.DataFrame:
    """Aggregate the same diagnostics over every head as a selection control."""

    pieces: list[pd.DataFrame] = []
    keys = [
        "model",
        "design_variant",
        "seed",
        "split",
        "query_occurrence",
        "layer",
    ]
    for pooling, spec in POOLING_SPECS.items():
        renamed = metrics.rename(
            columns={
                spec["total"]: "needle_total_mass",
                spec["effective"]: "needle_effective_number",
                spec["coverage"]: "needle_relative_coverage",
                spec["current_share"]: "needle_current_share",
            }
        )
        aggregated = renamed.groupby(keys, as_index=False)[
            list(ATTENTION_METRICS)
        ].mean()
        aggregated["hidden_pooling"] = pooling
        aggregated["key_pooling"] = spec["key_pooling"]
        aggregated["outcome"] = [
            outcomes[(str(variant), int(seed))]
            for variant, seed in zip(
                aggregated["design_variant"], aggregated["seed"]
            )
        ]
        pieces.append(aggregated)
    return pd.concat(pieces, ignore_index=True)


def _hidden_counter_noise(
    model_root: Path,
    model: str,
    *,
    primary_layers: dict[str, int],
) -> pd.DataFrame:
    capture_root = model_root / "representation" / "capture"
    records = _read_jsonl(capture_root / "capture_index.jsonl")
    rows: list[dict[str, Any]] = []
    for variant in sorted({str(record["design_variant"]) for record in records}):
        variant_records = [
            record for record in records if str(record["design_variant"]) == variant
        ]
        arrays: dict[str, list[np.ndarray]] = {
            pooling: [] for pooling in POOLING_SPECS
        }
        metadata: list[tuple[int, str]] = []
        layer_indices: np.ndarray | None = None
        for record in variant_records:
            with np.load(
                capture_root / str(record["shard_path"]), allow_pickle=False
            ) as payload:
                current_layers = np.asarray(payload["layer_indices"], dtype=int)
                if layer_indices is None:
                    layer_indices = current_layers
                elif not np.array_equal(layer_indices, current_layers):
                    raise RuntimeError(f"{model}/{variant}: hidden layer grid changed")
                for pooling in POOLING_SPECS:
                    arrays[pooling].append(np.asarray(payload[pooling]))
            metadata.append((int(record["seed"]), str(record["split"])))
        if layer_indices is None or len(metadata) != 30:
            raise RuntimeError(f"{model}/{variant}: expected 30 hidden shards")
        discovery = np.asarray([split == "discovery" for _seed, split in metadata])
        confirmation = np.asarray(
            [split == "confirmation" for _seed, split in metadata]
        )
        for pooling in POOLING_SPECS:
            tensor = np.stack(arrays[pooling], axis=0).astype(np.float32)
            centroids = tensor[discovery].mean(axis=0)
            centered_centroids = centroids - centroids.mean(axis=1, keepdims=True)
            signal_rms = np.sqrt(
                np.mean(np.sum(centered_centroids**2, axis=-1), axis=1)
            )
            for sample_axis in np.flatnonzero(confirmation):
                seed = metadata[int(sample_axis)][0]
                residual = tensor[int(sample_axis)] - centroids
                residual_norm = np.linalg.norm(residual, axis=-1)
                normalized = np.divide(
                    residual_norm,
                    signal_rms[:, None],
                    out=np.full_like(residual_norm, np.nan),
                    where=signal_rms[:, None] > 0,
                )
                for layer_axis, layer in enumerate(layer_indices):
                    for occurrence in range(1, 11):
                        rows.append(
                            {
                                "model": model,
                                "design_variant": variant,
                                "seed": int(seed),
                                "split": "confirmation",
                                "hidden_pooling": pooling,
                                "layer": int(layer),
                                "query_occurrence": int(occurrence),
                                "counter_residual_norm": float(
                                    residual_norm[int(layer_axis), occurrence - 1]
                                ),
                                "count_signal_rms": float(
                                    signal_rms[int(layer_axis)]
                                ),
                                "counter_noise": float(
                                    normalized[int(layer_axis), occurrence - 1]
                                ),
                                "primary_probe_layer": bool(
                                    int(layer) == int(primary_layers[pooling])
                                ),
                            }
                        )
            del tensor
    return pd.DataFrame(rows)


def _slope_summary(
    bank: pd.DataFrame,
    noise: pd.DataFrame,
    *,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    slope_rows: list[dict[str, Any]] = []
    confirmation = bank[bank["split"] == "confirmation"]
    group_keys = ["model", "design_variant", "hidden_pooling", "layer"]
    for keys, group in confirmation.groupby(group_keys, sort=True):
        for metric in ATTENTION_METRICS:
            seed_slopes = []
            for _seed, seed_frame in group.groupby("seed"):
                seed_slopes.append(
                    _linear_slope(
                        (seed_frame["query_occurrence"].to_numpy() - 1.0) / 9.0,
                        seed_frame[metric].to_numpy(),
                    )
                )
            values = np.asarray(seed_slopes, dtype=float)
            low, high = _bootstrap_mean_interval(
                values,
                seed=_stable_seed(*keys, metric, "attention_slope"),
                replicates=bootstrap_replicates,
            )
            slope_rows.append(
                {
                    **dict(zip(group_keys, keys)),
                    "quantity": "attention",
                    "metric": metric,
                    "seed_count": int(np.isfinite(values).sum()),
                    "mean_slope_per_full_1_to_10_range": float(
                        np.nanmean(values)
                    ),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    for keys, group in noise.groupby(group_keys, sort=True):
        seed_slopes = []
        for _seed, seed_frame in group.groupby("seed"):
            seed_slopes.append(
                _linear_slope(
                    (seed_frame["query_occurrence"].to_numpy() - 1.0) / 9.0,
                    seed_frame["counter_noise"].to_numpy(),
                )
            )
        values = np.asarray(seed_slopes, dtype=float)
        low, high = _bootstrap_mean_interval(
            values,
            seed=_stable_seed(*keys, "counter_noise_slope"),
            replicates=bootstrap_replicates,
        )
        slope_rows.append(
            {
                **dict(zip(group_keys, keys)),
                "quantity": "hidden_counter",
                "metric": "counter_noise",
                "seed_count": int(np.isfinite(values).sum()),
                "mean_slope_per_full_1_to_10_range": float(np.nanmean(values)),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    result = pd.DataFrame(slope_rows)
    primary = noise[
        [*group_keys, "primary_probe_layer"]
    ].drop_duplicates(group_keys)
    return result.merge(primary, on=group_keys, how="left", validate="many_to_one")


def _attention_noise_association(
    bank: pd.DataFrame,
    noise: pd.DataFrame,
    *,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    confirmation = bank[bank["split"] == "confirmation"].merge(
        noise,
        on=[
            "model",
            "design_variant",
            "seed",
            "split",
            "hidden_pooling",
            "layer",
            "query_occurrence",
        ],
        how="inner",
        validate="one_to_one",
    )
    group_keys = ["model", "design_variant", "hidden_pooling", "layer"]
    rows: list[dict[str, Any]] = []
    for keys, group in confirmation.groupby(group_keys, sort=True):
        for metric in ASSOCIATION_METRICS:
            estimate = _residualized_correlation(group, metric)
            low, high = _bootstrap_seed_correlation(
                group,
                metric,
                seed=_stable_seed(*keys, metric, "residualized_correlation"),
                replicates=bootstrap_replicates,
            )
            rows.append(
                {
                    **dict(zip(group_keys, keys)),
                    "attention_metric": metric,
                    "correlation": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "seed_count": int(group["seed"].nunique()),
                    "sample_count": int(len(group)),
                    "primary_probe_layer": bool(
                        group["primary_probe_layer"].iloc[0]
                    ),
                    "adjustment": "within-occurrence demeaning",
                    "estimand": (
                        "confirmation seed variation after removing occurrence means"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _profile_maps(
    model_root: Path,
    model: str,
    banks: pd.DataFrame,
) -> pd.DataFrame:
    capture_root = (
        model_root / "representation" / "prompt_counter_attention_v1"
    )
    records = [
        record
        for record in _read_jsonl(capture_root / "capture_index.jsonl")
        if str(record["split"]) == "confirmation"
    ]
    accumulators: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    model_banks = banks[banks["model"] == model]
    for record in records:
        variant = str(record["design_variant"])
        with np.load(
            capture_root / str(record["profile_path"]), allow_pickle=False
        ) as payload:
            layers = np.asarray(payload["layer_indices"], dtype=int)
            heads = np.asarray(payload["head_indices"], dtype=int)
            for pooling, spec in POOLING_SPECS.items():
                values = np.asarray(payload[spec["profile"]], dtype=np.float32)
                selected = model_banks[
                    (model_banks["design_variant"] == variant)
                    & (model_banks["hidden_pooling"] == pooling)
                ]
                for layer, group in selected.groupby("layer"):
                    indices = []
                    for head in group["head"].astype(int):
                        match = np.flatnonzero(
                            (layers == int(layer)) & (heads == int(head))
                        )
                        if len(match) != 1:
                            raise RuntimeError(
                                f"{model}/{variant}/{pooling}/L{layer}H{head}: "
                                "profile head lookup failed"
                            )
                        indices.append(int(match[0]))
                    accumulators[(variant, pooling, int(layer))].append(
                        values[np.asarray(indices, dtype=int)].mean(axis=0)
                    )
    rows: list[dict[str, Any]] = []
    for (variant, pooling, layer), maps in sorted(accumulators.items()):
        mean_map = np.stack(maps).mean(axis=0)
        row_totals = mean_map.sum(axis=1, keepdims=True)
        shares = np.divide(
            mean_map,
            row_totals,
            out=np.zeros_like(mean_map),
            where=row_totals > 0,
        )
        for query_occurrence in range(1, 11):
            for key_occurrence in range(1, 11):
                rows.append(
                    {
                        "model": model,
                        "design_variant": variant,
                        "hidden_pooling": pooling,
                        "key_pooling": POOLING_SPECS[pooling]["key_pooling"],
                        "layer": int(layer),
                        "query_occurrence": int(query_occurrence),
                        "key_occurrence": int(key_occurrence),
                        "visible_by_causality": bool(
                            key_occurrence <= query_occurrence
                        ),
                        "mean_attention_mass": float(
                            mean_map[query_occurrence - 1, key_occurrence - 1]
                        ),
                        "within_needle_share": float(
                            shares[query_occurrence - 1, key_occurrence - 1]
                        ),
                        "confirmation_seed_count": int(len(maps)),
                    }
                )
    return pd.DataFrame(rows)


def analyze_prompt_counter_dynamics(
    run_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_n: int = 8,
    bootstrap_replicates: int = 2000,
) -> dict[str, Path]:
    run_root = Path(run_root)
    output = (
        Path(output_dir)
        if output_dir is not None
        else run_root / "analysis" / "prompt_counter_dynamics_v1"
    )
    bank_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    all_head_sample_frames: list[pd.DataFrame] = []
    noise_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []
    for model in MODELS:
        model_root = run_root / model / "numeric"
        primary_layers = _primary_layers(model_root)
        outcomes = _behavior_lookup(model_root)
        metrics = _load_attention_metrics(model_root, model)
        banks = _select_head_banks(
            metrics,
            primary_layers=primary_layers,
            top_n=int(top_n),
        )
        samples = _bank_sample_metrics(metrics, banks, outcomes)
        all_head_samples = _all_head_sample_metrics(metrics, outcomes)
        noise = _hidden_counter_noise(
            model_root,
            model,
            primary_layers=primary_layers,
        )
        profiles = _profile_maps(model_root, model, banks)
        bank_frames.append(banks)
        sample_frames.append(samples)
        all_head_sample_frames.append(all_head_samples)
        noise_frames.append(noise)
        profile_frames.append(profiles)
        del metrics
    bank = pd.concat(bank_frames, ignore_index=True)
    samples = pd.concat(sample_frames, ignore_index=True)
    all_head_samples = pd.concat(all_head_sample_frames, ignore_index=True)
    noise = pd.concat(noise_frames, ignore_index=True)
    profiles = pd.concat(profile_frames, ignore_index=True)
    slopes = _slope_summary(
        samples,
        noise,
        bootstrap_replicates=int(bootstrap_replicates),
    )
    associations = _attention_noise_association(
        samples,
        noise,
        bootstrap_replicates=int(bootstrap_replicates),
    )
    all_head_slopes = _slope_summary(
        all_head_samples,
        noise,
        bootstrap_replicates=int(bootstrap_replicates),
    )
    all_head_associations = _attention_noise_association(
        all_head_samples,
        noise,
        bootstrap_replicates=int(bootstrap_replicates),
    )
    paths = {
        "selected_head_bank": output / "selected_head_bank.csv",
        "attention_bank_by_sample": output / "attention_bank_by_sample.csv.gz",
        "hidden_counter_noise_by_sample": output
        / "hidden_counter_noise_by_sample.csv.gz",
        "occurrence_slope_summary": output / "occurrence_slope_summary.csv",
        "attention_noise_association": output
        / "attention_noise_association.csv",
        "all_head_attention_by_sample": output
        / "all_head_attention_by_sample.csv.gz",
        "all_head_occurrence_slope_summary": output
        / "all_head_occurrence_slope_summary.csv",
        "all_head_attention_noise_association": output
        / "all_head_attention_noise_association.csv",
        "profile_maps": output / "profile_maps.csv.gz",
        "manifest": output / "analysis_manifest.json",
    }
    _write_csv(bank, paths["selected_head_bank"])
    _write_csv(samples, paths["attention_bank_by_sample"], gzip=True)
    _write_csv(noise, paths["hidden_counter_noise_by_sample"], gzip=True)
    _write_csv(slopes, paths["occurrence_slope_summary"])
    _write_csv(associations, paths["attention_noise_association"])
    _write_csv(
        all_head_samples,
        paths["all_head_attention_by_sample"],
        gzip=True,
    )
    _write_csv(
        all_head_slopes,
        paths["all_head_occurrence_slope_summary"],
    )
    _write_csv(
        all_head_associations,
        paths["all_head_attention_noise_association"],
    )
    _write_csv(profiles, paths["profile_maps"], gzip=True)
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "models": list(MODELS),
            "hidden_poolings": list(POOLING_SPECS),
            "query_site": "needle_end",
            "key_poolings": {
                pooling: spec["key_pooling"]
                for pooling, spec in POOLING_SPECS.items()
            },
            "head_selection": {
                "split": "discovery",
                "occurrences": "2-10",
                "top_n_per_model_variant_layer_pooling": int(top_n),
                "score": "mean(total_needle_mass*relative_coverage)",
            },
            "head_scope_control": (
                "the same sample metrics, slopes, and occurrence-adjusted "
                "associations averaged over every head in each layer"
            ),
            "hidden_noise": (
                "confirmation full-space residual norm to the matching "
                "variant/occurrence discovery centroid, divided by discovery "
                "count-centroid RMS"
            ),
            "association_adjustment": "within-occurrence demeaning",
            "inference_unit": "confirmation seed",
            "bootstrap_replicates": int(bootstrap_replicates),
            "causal_claim": False,
            "row_counts": {
                "selected_head_bank": int(len(bank)),
                "attention_bank_by_sample": int(len(samples)),
                "hidden_counter_noise_by_sample": int(len(noise)),
                "occurrence_slope_summary": int(len(slopes)),
                "attention_noise_association": int(len(associations)),
                "all_head_attention_by_sample": int(len(all_head_samples)),
                "all_head_occurrence_slope_summary": int(len(all_head_slopes)),
                "all_head_attention_noise_association": int(
                    len(all_head_associations)
                ),
                "profile_maps": int(len(profiles)),
            },
        },
        paths["manifest"],
    )
    return paths
