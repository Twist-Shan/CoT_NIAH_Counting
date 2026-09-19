from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .causal_audit import (
    CONFIRMATION_SEEDS,
    MODELS,
    STRICT_GREEDY_METRIC,
    VARIANTS,
    ScreenDesign,
    _read_json,
    _require,
    _table_row_count,
    _validate_behavior_alignment,
    _validate_csv_capture,
    _validate_generation_rows,
)


PROFILE = "answer_query_dense_v1"
DIRECTED_COUNT_PAIRS = (
    (5, 6),
    (6, 5),
    (7, 8),
    (8, 7),
    (9, 10),
    (10, 9),
    (5, 10),
    (10, 5),
)
EXPECTED_LAYERS = {
    "Qwen3-8B": (0, 9, 18, 26, 29, 32, 34, 35),
    "Gemma4-E4B": (0, 10, 20, 31, 35, 38, 40, 41),
}


def _matches_answer_query_design(model: str, design: dict[str, Any]) -> bool:
    return (
        design.get("family") == "generation_residual_patching_v1"
        and design.get("model_label") == model
        and design.get("answer_format") == "numeric"
        and design.get("behavior_metric") == STRICT_GREEDY_METRIC
        and design.get("confirmation_variants") == list(VARIANTS)
        and design.get("confirmation_seeds") == list(CONFIRMATION_SEEDS)
        and design.get("confirmation_counts") == list(range(1, 11))
        and design.get("layers") == list(EXPECTED_LAYERS[model])
        and design.get("directed_count_pairs")
        == [list(pair) for pair in DIRECTED_COUNT_PAIRS]
        and design.get("sites") == ["answer_query"]
        and design.get("needle_protocols") == ["single_layer"]
    )


def find_answer_query_designs(
    run_root: str | Path,
) -> dict[str, ScreenDesign]:
    root = Path(run_root).resolve()
    selected: dict[str, ScreenDesign] = {}
    for model in MODELS:
        family_root = (
            root / model / "numeric" / "causal" / "generation_residual_patching_v1"
        )
        matches: list[ScreenDesign] = []
        for design_root in sorted(family_root.glob("design_*")):
            design_path = design_root / "design.json"
            if not design_path.is_file():
                continue
            design = _read_json(design_path)
            if _matches_answer_query_design(model, design):
                matches.append(
                    ScreenDesign(model, "answer-query-patching", design_root, design)
                )
        _require(
            len(matches) == 1,
            f"Expected one {model} {PROFILE} design, found {len(matches)}",
        )
        selected[model] = matches[0]
    return selected


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    _require(
        normalized.isin({"true", "false"}).all(),
        f"Cannot parse boolean column {series.name}",
    )
    return normalized.eq("true")


def _as_nullable_bool_numeric(series: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    present = series.notna()
    if not present.any():
        return result
    if pd.api.types.is_bool_dtype(series):
        result.loc[present] = series.loc[present].astype(float)
        return result
    normalized = series.loc[present].astype(str).str.strip().str.lower()
    _require(
        normalized.isin({"true", "false"}).all(),
        f"Cannot parse nullable boolean column {series.name}",
    )
    result.loc[present] = normalized.eq("true").astype(float)
    return result


def _numeric(series: pd.Series, *, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    _require(values.notna().all(), f"{name} contains nonnumeric values")
    return values


def add_answer_query_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    frame = detail.copy()
    baseline = pd.to_numeric(frame["baseline_predicted_count"], errors="coerce")
    donor = pd.to_numeric(frame["donor_baseline_predicted_count"], errors="coerce")
    patched = pd.to_numeric(frame["patched_predicted_count"], errors="coerce")
    receiver_count = _numeric(frame["receiver_count"], name="receiver_count")
    donor_count = _numeric(frame["donor_count"], name="donor_count")
    generated_shift = pd.to_numeric(frame["generated_count_shift"], errors="coerce")
    direction_sign = np.sign(donor_count - receiver_count)

    frame["patched_valid_numeric"] = _as_bool(frame["patched_format_valid"]).astype(
        float
    )
    frame["prediction_changed_numeric"] = _as_nullable_bool_numeric(
        frame["prediction_changed"]
    )
    frame["moved_toward_donor_gold_numeric"] = _as_nullable_bool_numeric(
        frame["moved_toward_donor_gold"]
    )
    frame["follows_donor_gold_numeric"] = _as_nullable_bool_numeric(
        frame["follows_donor_gold"]
    )
    frame["follows_donor_prediction_numeric"] = _as_nullable_bool_numeric(
        frame["follows_donor_prediction"]
    )
    frame["direction_aligned_shift"] = generated_shift * direction_sign
    frame["canonical_pair"] = [
        f"{min(int(receiver), int(target))}<->{max(int(receiver), int(target))}"
        for receiver, target in zip(receiver_count, donor_count)
    ]

    eligible = baseline.notna() & donor.notna() & baseline.ne(donor)
    frame["donor_prediction_eligible"] = eligible
    frame["donor_prediction_adopted"] = np.where(
        eligible, patched.eq(donor).astype(float), np.nan
    )
    baseline_distance = (baseline - donor).abs()
    patched_distance = (patched - donor).abs()
    frame["moved_toward_donor_prediction"] = np.where(
        eligible, patched_distance.lt(baseline_distance).astype(float), np.nan
    )
    frame["donor_prediction_distance_reduction"] = np.where(
        eligible,
        (baseline_distance - patched_distance) / baseline_distance,
        np.nan,
    )
    frame["donor_prediction_transport_fraction"] = np.where(
        eligible, (patched - baseline) / (donor - baseline), np.nan
    )
    return frame


def _validate_donor_alignment(
    detail: pd.DataFrame, labels: pd.DataFrame, *, context: str
) -> None:
    donor = detail[
        [
            "donor_stimulus_id",
            "donor_count",
            "donor_baseline_outcome",
            "donor_baseline_predicted_count",
        ]
    ].drop_duplicates()
    _require(
        not donor.duplicated("donor_stimulus_id").any(),
        f"{context}: inconsistent donor baseline fields",
    )
    expected = labels[
        ["stimulus_id", "gold_count", "outcome_group", "parsed_count"]
    ].rename(columns={"stimulus_id": "donor_stimulus_id"})
    merged = donor.merge(
        expected,
        on="donor_stimulus_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    _require(
        merged["_merge"].eq("both").all(),
        f"{context}: donor lacks a behavior label",
    )
    comparisons = (
        ("donor_count", "gold_count"),
        ("donor_baseline_outcome", "outcome_group"),
        ("donor_baseline_predicted_count", "parsed_count"),
    )
    for left, right in comparisons:
        equal = merged[left].eq(merged[right]) | (
            merged[left].isna() & merged[right].isna()
        )
        _require(equal.all(), f"{context}: donor field {left} disagrees")


def _validate_derived_fields(detail: pd.DataFrame, *, context: str) -> None:
    _require(
        not _as_bool(detail["patched_generation_truncated"]).any(),
        f"{context}: at least one generation was truncated",
    )
    valid = _as_bool(detail["patched_format_valid"])
    baseline = _numeric(
        detail["baseline_predicted_count"], name="baseline_predicted_count"
    )
    patched = pd.to_numeric(detail["patched_predicted_count"], errors="coerce")
    donor_prediction = _numeric(
        detail["donor_baseline_predicted_count"],
        name="donor_baseline_predicted_count",
    )
    donor_gold = _numeric(detail["donor_count"], name="donor_count")
    generated_shift = pd.to_numeric(detail["generated_count_shift"], errors="coerce")
    _require(
        patched.loc[valid].notna().all() and patched.loc[~valid].isna().all(),
        f"{context}: patched numeric value disagrees with format validity",
    )
    observed_changed = _as_nullable_bool_numeric(detail["prediction_changed"])
    _require(
        np.array_equal(
            observed_changed.loc[valid].astype(bool).to_numpy(),
            baseline.loc[valid].ne(patched.loc[valid]).to_numpy(),
        ),
        f"{context}: prediction_changed is inconsistent",
    )
    _require(
        observed_changed.loc[~valid].isna().all(),
        f"{context}: invalid rows must not have prediction_changed labels",
    )
    _require(
        np.allclose(
            generated_shift.loc[valid].to_numpy(),
            (patched.loc[valid] - baseline.loc[valid]).to_numpy(),
        )
        and generated_shift.loc[~valid].isna().all(),
        f"{context}: generated_count_shift is inconsistent",
    )
    observed_donor_gold = _as_nullable_bool_numeric(detail["follows_donor_gold"])
    _require(
        np.array_equal(
            observed_donor_gold.loc[valid].astype(bool).to_numpy(),
            patched.loc[valid].eq(donor_gold.loc[valid]).to_numpy(),
        ),
        f"{context}: follows_donor_gold is inconsistent",
    )
    observed_donor_prediction = _as_nullable_bool_numeric(
        detail["follows_donor_prediction"]
    )
    _require(
        np.array_equal(
            observed_donor_prediction.loc[valid].astype(bool).to_numpy(),
            patched.loc[valid].eq(donor_prediction.loc[valid]).to_numpy(),
        ),
        f"{context}: follows_donor_prediction is inconsistent",
    )
    expected_moved = (
        (patched.loc[valid] - donor_gold.loc[valid])
        .abs()
        .lt((baseline.loc[valid] - donor_gold.loc[valid]).abs())
    )
    observed_moved = _as_nullable_bool_numeric(detail["moved_toward_donor_gold"])
    _require(
        np.array_equal(
            observed_moved.loc[valid].astype(bool).to_numpy(),
            expected_moved.to_numpy(),
        ),
        f"{context}: moved_toward_donor_gold is inconsistent",
    )
    for name, observed in (
        ("follows_donor_gold", observed_donor_gold),
        ("follows_donor_prediction", observed_donor_prediction),
        ("moved_toward_donor_gold", observed_moved),
    ):
        _require(
            observed.loc[~valid].isna().all(),
            f"{context}: invalid rows must not have {name} labels",
        )


def _validate_family_cartesian(
    detail: pd.DataFrame, *, model: str, context: str
) -> None:
    expected = {
        (receiver, donor, layer)
        for receiver, donor in DIRECTED_COUNT_PAIRS
        for layer in EXPECTED_LAYERS[model]
    }
    for (variant, seed), family in detail.groupby(
        ["design_variant", "seed"], sort=True
    ):
        observed = {
            (int(row.receiver_count), int(row.donor_count), int(row.start_layer))
            for row in family.itertuples(index=False)
        }
        _require(
            len(family) == 64 and observed == expected,
            f"{context}: incomplete cartesian family {variant}/seed={seed}",
        )


def audit_answer_query_patching(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    status = _read_json(root / "answer_query_patching.status")
    marker = _read_json(root / "answer_query_patching_dense_v1.complete")
    _require(status.get("state") == "COMPLETE", "Campaign status is not COMPLETE")
    _require(marker.get("state") == "COMPLETE", "Completion marker is not COMPLETE")
    _require(status.get("profile") == PROFILE, "Unexpected campaign profile")
    designs = find_answer_query_designs(root)
    report: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_answer_query_patching_audit_v1",
        "run_root": str(root),
        "profile": PROFILE,
        "campaign_updated_utc": status.get("updated_utc"),
        "models": {},
    }
    for model, design in designs.items():
        context = f"{model}/{PROFILE}"
        labels = pd.read_csv(
            root / model / "numeric" / "behavior" / "capture" / "generation_labels.csv"
        )
        detail, counts = _validate_csv_capture(
            design,
            capture_subdir="capture",
            expected_index_rows=40,
            expected_rows_per_shard=64,
            expected_detail_rows=2560,
            unique_columns=[
                "receiver_stimulus_id",
                "donor_stimulus_id",
                "site",
                "patch_protocol",
                "start_layer",
            ],
        )
        _validate_generation_rows(detail, context=context)
        _validate_behavior_alignment(detail, labels, context=context)
        _validate_donor_alignment(detail, labels, context=context)
        _validate_derived_fields(detail, context=context)
        _validate_family_cartesian(detail, model=model, context=context)
        _require(set(detail["status"]) == {"ok"}, f"{context}: skipped rows")
        _require(
            set(detail["site"]) == {"answer_query"}
            and set(detail["patch_protocol"]) == {"single_layer"}
            and set(pd.to_numeric(detail["patched_layer_count"]).astype(int)) == {1},
            f"{context}: unexpected site, protocol, or patched layer count",
        )
        enriched = add_answer_query_metrics(detail)
        invalid_detail = detail[~_as_bool(detail["patched_format_valid"])]
        counts.update(
            {
                "design": str(design.root.relative_to(root)),
                "successful_rows": int(detail["status"].eq("ok").sum()),
                "skipped_rows": int(detail["status"].ne("ok").sum()),
                "patched_valid_rows": int(
                    _as_bool(detail["patched_format_valid"]).sum()
                ),
                "patched_invalid_rows": int(
                    (~_as_bool(detail["patched_format_valid"])).sum()
                ),
                "summary_rows": _table_row_count(
                    design.root / "summary.csv",
                    required_columns={
                        "start_layer",
                        "direction",
                        "prediction_changed_rate",
                    },
                ),
                "layers": list(EXPECTED_LAYERS[model]),
                "directed_count_pairs": [list(pair) for pair in DIRECTED_COUNT_PAIRS],
                "eligible_donor_prediction_rows": int(
                    enriched["donor_prediction_eligible"].sum()
                ),
                "invalid_examples": invalid_detail[
                    [
                        "design_variant",
                        "seed",
                        "receiver_stimulus_id",
                        "donor_stimulus_id",
                        "receiver_count",
                        "donor_count",
                        "start_layer",
                        "patched_completion_text_raw",
                        "patched_generated_token_ids",
                    ]
                ].to_dict("records"),
            }
        )
        report["models"][model] = counts

    error_terms = ("Traceback", "CUDA out of memory", "OOM", "FAILED")
    error_hits: list[str] = []
    for path in sorted((root / "logs").glob("answer_query_dense_v1*.log")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if any(term in line for term in error_terms):
                error_hits.append(f"{path.name}:{line_number}:{line}")
    _require(not error_hits, f"Answer-query logs contain errors: {error_hits[:3]}")
    report["log_error_hits"] = error_hits
    report["validated"] = True
    return report


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")


def _seed_estimate(
    frame: pd.DataFrame,
    metric: str,
    *,
    label: str,
    repetitions: int,
    expected_seed_count: int | None = len(CONFIRMATION_SEEDS),
) -> dict[str, float | int]:
    usable = frame[["seed", metric]].dropna()
    seed_values = usable.groupby("seed", sort=True)[metric].mean().to_numpy(dtype=float)
    if expected_seed_count is None:
        _require(len(seed_values) > 0, f"{label}: no usable seed clusters")
    else:
        _require(
            len(seed_values) == expected_seed_count,
            f"{label}: expected {expected_seed_count} seed clusters, "
            f"found {len(seed_values)}",
        )
    rng = np.random.default_rng(_stable_seed(label))
    indices = rng.integers(
        0, len(seed_values), size=(int(repetitions), len(seed_values))
    )
    bootstrap = seed_values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "estimate": float(seed_values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "seed_clusters": int(len(seed_values)),
    }


def _exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    _require(values.ndim == 1 and 0 < len(values) <= 20, "Invalid sign-flip input")
    observed = abs(float(values.mean()))
    masks = np.arange(1 << len(values), dtype=np.uint64)[:, None]
    bits = ((masks >> np.arange(len(values), dtype=np.uint64)) & 1).astype(float)
    signs = bits * 2.0 - 1.0
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-12))


def _holm(values: Iterable[float]) -> list[float]:
    p_values = [float(value) for value in values]
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _layer_contrast(
    detail: pd.DataFrame,
    *,
    layer: int,
    baseline_layer: int,
    metric: str,
    label: str,
    repetitions: int,
) -> dict[str, float]:
    identity = [
        "design_variant",
        "seed",
        "receiver_stimulus_id",
        "donor_stimulus_id",
    ]
    selected = detail[
        pd.to_numeric(detail["start_layer"])
        .astype(int)
        .isin({int(layer), int(baseline_layer)})
    ]
    pivot = selected.pivot(index=identity, columns="start_layer", values=metric)
    _require(
        {layer, baseline_layer}.issubset(pivot.columns),
        f"{label}: missing layer in paired contrast",
    )
    paired = pivot[[baseline_layer, layer]].dropna()
    differences = (paired[layer] - paired[baseline_layer]).rename("difference")
    by_seed = differences.reset_index().groupby("seed", sort=True)["difference"].mean()
    _require(len(by_seed) == 10, f"{label}: expected ten paired seeds")
    estimate = _seed_estimate(
        by_seed.rename("difference").reset_index(),
        "difference",
        label=label,
        repetitions=repetitions,
    )
    estimate["p_raw"] = _exact_sign_flip_p(by_seed.to_numpy(dtype=float))
    return estimate


def analyze_answer_query_patching(
    run_root: str | Path,
    *,
    bootstrap_repetitions: int = 20_000,
) -> dict[str, pd.DataFrame]:
    root = Path(run_root).resolve()
    designs = find_answer_query_designs(root)
    layer_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    invalid_frames: list[pd.DataFrame] = []
    for model, design in designs.items():
        raw = pd.read_csv(design.root / "detail.csv.gz", compression="gzip")
        detail = add_answer_query_metrics(raw)
        labels = pd.read_csv(
            root / model / "numeric" / "behavior" / "capture" / "generation_labels.csv"
        ).set_index("stimulus_id")
        invalid = detail[detail["patched_valid_numeric"].eq(0.0)].copy()
        if not invalid.empty:
            for role in ("receiver", "donor"):
                ids = invalid[f"{role}_stimulus_id"]
                invalid[f"{role}_baseline_completion_text_raw"] = ids.map(
                    labels["completion_text_raw"]
                )
                invalid[f"{role}_baseline_generated_token_ids"] = ids.map(
                    labels["generated_token_ids"]
                )
            invalid.insert(0, "model", model)
            invalid_frames.append(
                invalid[
                    [
                        "model",
                        "design_variant",
                        "seed",
                        "receiver_stimulus_id",
                        "donor_stimulus_id",
                        "receiver_count",
                        "donor_count",
                        "baseline_predicted_count",
                        "donor_baseline_predicted_count",
                        "receiver_baseline_completion_text_raw",
                        "receiver_baseline_generated_token_ids",
                        "donor_baseline_completion_text_raw",
                        "donor_baseline_generated_token_ids",
                        "start_layer",
                        "patched_outcome",
                        "patched_completion_text",
                        "patched_completion_text_raw",
                        "patched_generated_token_ids",
                        "patched_generation_truncated",
                    ]
                ]
            )
        baseline_layer = min(EXPECTED_LAYERS[model])
        model_layer_rows: list[dict[str, Any]] = []
        for layer in EXPECTED_LAYERS[model]:
            selected = detail[
                pd.to_numeric(detail["start_layer"]).astype(int).eq(layer)
            ]
            row: dict[str, Any] = {
                "model": model,
                "layer": int(layer),
                "rows": int(len(selected)),
                "baseline_correct_rate": float(
                    _as_bool(selected["baseline_is_correct"]).mean()
                ),
                "eligible_donor_prediction_rows": int(
                    selected["donor_prediction_eligible"].sum()
                ),
            }
            metrics = {
                "patched_valid_rate": "patched_valid_numeric",
                "changed_rate": "prediction_changed_numeric",
                "moved_toward_donor_gold_rate": "moved_toward_donor_gold_numeric",
                "follows_donor_gold_rate": "follows_donor_gold_numeric",
                "follows_donor_prediction_rate": "follows_donor_prediction_numeric",
                "eligible_donor_adoption_rate": "donor_prediction_adopted",
                "eligible_donor_distance_reduction_rate": "moved_toward_donor_prediction",
                "mean_donor_distance_reduction": "donor_prediction_distance_reduction",
                "mean_direction_aligned_shift": "direction_aligned_shift",
            }
            for output_name, metric in metrics.items():
                estimate = _seed_estimate(
                    selected,
                    metric,
                    label=f"answer-query-{model}-L{layer}-{metric}",
                    repetitions=bootstrap_repetitions,
                )
                row[output_name] = estimate["estimate"]
                row[f"{output_name}_ci95_low"] = estimate["ci95_low"]
                row[f"{output_name}_ci95_high"] = estimate["ci95_high"]
                row[f"{output_name}_seed_clusters"] = estimate["seed_clusters"]
            if layer == baseline_layer:
                for metric_name in ("eligible_donor_adoption", "aligned_shift"):
                    row[f"{metric_name}_vs_layer0"] = 0.0
                    row[f"{metric_name}_vs_layer0_ci95_low"] = 0.0
                    row[f"{metric_name}_vs_layer0_ci95_high"] = 0.0
                    row[f"{metric_name}_vs_layer0_p_raw"] = 1.0
            else:
                contrasts = {
                    "eligible_donor_adoption": "donor_prediction_adopted",
                    "aligned_shift": "direction_aligned_shift",
                }
                for output_name, metric in contrasts.items():
                    contrast = _layer_contrast(
                        detail,
                        layer=layer,
                        baseline_layer=baseline_layer,
                        metric=metric,
                        label=f"answer-query-{model}-L{layer}-vs-L{baseline_layer}-{metric}",
                        repetitions=bootstrap_repetitions,
                    )
                    row[f"{output_name}_vs_layer0"] = contrast["estimate"]
                    row[f"{output_name}_vs_layer0_ci95_low"] = contrast["ci95_low"]
                    row[f"{output_name}_vs_layer0_ci95_high"] = contrast["ci95_high"]
                    row[f"{output_name}_vs_layer0_p_raw"] = contrast["p_raw"]
            model_layer_rows.append(row)

        for metric_name in ("eligible_donor_adoption", "aligned_shift"):
            nonbaseline = [
                row for row in model_layer_rows if int(row["layer"]) != baseline_layer
            ]
            adjusted = _holm(
                row[f"{metric_name}_vs_layer0_p_raw"] for row in nonbaseline
            )
            for row, value in zip(nonbaseline, adjusted):
                row[f"{metric_name}_vs_layer0_p_holm"] = value
            for row in model_layer_rows:
                if int(row["layer"]) == baseline_layer:
                    row[f"{metric_name}_vs_layer0_p_holm"] = 1.0
        layer_rows.extend(model_layer_rows)

        summary_metrics = {
            "patched_valid_rate": "patched_valid_numeric",
            "changed_rate": "prediction_changed_numeric",
            "moved_toward_donor_gold_rate": "moved_toward_donor_gold_numeric",
            "follows_donor_gold_rate": "follows_donor_gold_numeric",
            "follows_donor_prediction_rate": "follows_donor_prediction_numeric",
            "eligible_donor_adoption_rate": "donor_prediction_adopted",
            "mean_direction_aligned_shift": "direction_aligned_shift",
        }
        pair_groups = [
            "start_layer",
            "receiver_count",
            "donor_count",
            "canonical_pair",
            "direction",
        ]
        for keys, selected in detail.groupby(pair_groups, sort=True):
            values = dict(zip(pair_groups, keys))
            row = {
                "model": model,
                "layer": int(values["start_layer"]),
                "receiver_count": int(values["receiver_count"]),
                "donor_count": int(values["donor_count"]),
                "canonical_pair": str(values["canonical_pair"]),
                "direction": str(values["direction"]),
                "rows": int(len(selected)),
                "seed_clusters": int(selected["seed"].nunique()),
                "eligible_donor_prediction_rows": int(
                    selected["donor_prediction_eligible"].sum()
                ),
            }
            for output_name, metric in summary_metrics.items():
                estimate = _seed_estimate(
                    selected,
                    metric,
                    label=(
                        f"answer-query-{model}-L{values['start_layer']}-"
                        f"{values['receiver_count']}-to-{values['donor_count']}-{metric}"
                    ),
                    repetitions=bootstrap_repetitions,
                    expected_seed_count=None,
                )
                row[output_name] = estimate["estimate"]
                row[f"{output_name}_ci95_low"] = estimate["ci95_low"]
                row[f"{output_name}_ci95_high"] = estimate["ci95_high"]
                row[f"{output_name}_seed_clusters"] = estimate["seed_clusters"]
            pair_rows.append(row)

        for (layer, variant), selected in detail.groupby(
            ["start_layer", "design_variant"], sort=True
        ):
            row = {
                "model": model,
                "layer": int(layer),
                "design_variant": str(variant),
                "rows": int(len(selected)),
                "seed_clusters": int(selected["seed"].nunique()),
                "eligible_donor_prediction_rows": int(
                    selected["donor_prediction_eligible"].sum()
                ),
            }
            for output_name, metric in summary_metrics.items():
                estimate = _seed_estimate(
                    selected,
                    metric,
                    label=f"answer-query-{model}-L{layer}-{variant}-{metric}",
                    repetitions=bootstrap_repetitions,
                    expected_seed_count=None,
                )
                row[output_name] = estimate["estimate"]
                row[f"{output_name}_ci95_low"] = estimate["ci95_low"]
                row[f"{output_name}_ci95_high"] = estimate["ci95_high"]
                row[f"{output_name}_seed_clusters"] = estimate["seed_clusters"]
            variant_rows.append(row)

        for (layer, outcome), selected in detail.groupby(
            ["start_layer", "baseline_outcome"], sort=True
        ):
            row = {
                "model": model,
                "layer": int(layer),
                "baseline_outcome": str(outcome),
                "rows": int(len(selected)),
                "seed_clusters": int(selected["seed"].nunique()),
                "eligible_donor_prediction_rows": int(
                    selected["donor_prediction_eligible"].sum()
                ),
            }
            for output_name, metric in summary_metrics.items():
                estimate = _seed_estimate(
                    selected,
                    metric,
                    label=f"answer-query-{model}-L{layer}-{outcome}-{metric}",
                    repetitions=bootstrap_repetitions,
                    expected_seed_count=None,
                )
                row[output_name] = estimate["estimate"]
                row[f"{output_name}_ci95_low"] = estimate["ci95_low"]
                row[f"{output_name}_ci95_high"] = estimate["ci95_high"]
                row[f"{output_name}_seed_clusters"] = estimate["seed_clusters"]
            outcome_rows.append(row)

        stratum_groups = [
            "start_layer",
            "direction",
            "canonical_pair",
            "design_variant",
            "baseline_outcome",
        ]
        for keys, selected in detail.groupby(stratum_groups, sort=True):
            values = dict(zip(stratum_groups, keys))
            stratum_rows.append(
                {
                    "model": model,
                    "layer": int(values["start_layer"]),
                    "direction": str(values["direction"]),
                    "canonical_pair": str(values["canonical_pair"]),
                    "design_variant": str(values["design_variant"]),
                    "baseline_outcome": str(values["baseline_outcome"]),
                    "rows": int(len(selected)),
                    "seed_clusters": int(selected["seed"].nunique()),
                    "patched_valid_rate": float(
                        selected["patched_valid_numeric"].mean()
                    ),
                    "changed_rate_valid_only": float(
                        selected["prediction_changed_numeric"].mean()
                    ),
                    "moved_toward_donor_gold_rate_valid_only": float(
                        selected["moved_toward_donor_gold_numeric"].mean()
                    ),
                    "follows_donor_gold_rate_valid_only": float(
                        selected["follows_donor_gold_numeric"].mean()
                    ),
                    "follows_donor_prediction_rate_valid_only": float(
                        selected["follows_donor_prediction_numeric"].mean()
                    ),
                    "eligible_donor_prediction_rows": int(
                        selected["donor_prediction_eligible"].sum()
                    ),
                    "eligible_donor_adoption_rate_invalid_as_failure": float(
                        selected["donor_prediction_adopted"].mean()
                    ),
                    "mean_direction_aligned_shift_valid_only": float(
                        selected["direction_aligned_shift"].mean()
                    ),
                }
            )

    invalid_rows = (
        pd.concat(invalid_frames, ignore_index=True)
        if invalid_frames
        else pd.DataFrame(
            columns=[
                "model",
                "design_variant",
                "seed",
                "receiver_stimulus_id",
                "donor_stimulus_id",
                "receiver_count",
                "donor_count",
                "baseline_predicted_count",
                "donor_baseline_predicted_count",
                "receiver_baseline_completion_text_raw",
                "receiver_baseline_generated_token_ids",
                "donor_baseline_completion_text_raw",
                "donor_baseline_generated_token_ids",
                "start_layer",
                "patched_outcome",
                "patched_completion_text",
                "patched_completion_text_raw",
                "patched_generated_token_ids",
                "patched_generation_truncated",
            ]
        )
    )

    return {
        "layer_summary": pd.DataFrame(layer_rows),
        "pair_summary": pd.DataFrame(pair_rows),
        "variant_summary": pd.DataFrame(variant_rows),
        "outcome_summary": pd.DataFrame(outcome_rows),
        "stratum_summary": pd.DataFrame(stratum_rows),
        "invalid_rows": invalid_rows,
    }


def write_answer_query_analysis(
    tables: dict[str, pd.DataFrame], output_dir: str | Path
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def write_answer_query_audit(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
