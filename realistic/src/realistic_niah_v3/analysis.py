from __future__ import annotations

import gzip
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest, t
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

from .accuracy_distributions import (
    ACCURACY_FAMILIES,
    ACCURACY_FAMILY_BY_NAME,
    AccuracyFamily,
    fit_accuracy_distribution,
    inverse_link,
    predictive_log_likelihood,
    quantile_residual_diagnostics,
    randomized_quantile_residuals,
)
from .spec import (
    MATCHED_REASONING_PAIRS,
    PROTOCOL_VERSION,
)
from .native_thinking import (
    NATIVE_THINKING_STYLE_ORDER,
    classify_native_thinking_style,
    native_thinking_style_flags,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple[str, ...]


CANDIDATES = (
    Candidate("intercept_only", ()),
    Candidate("linear_N", ("N",)),
    Candidate("linear_L", ("L_k",)),
    Candidate("log_N", ("ln_N",)),
    Candidate("log_L", ("ln_L_k",)),
    Candidate("linear_additive", ("N", "L_k")),
    Candidate("log_additive", ("ln_N", "ln_L_k")),
    Candidate("N_logL", ("N", "ln_L_k")),
    Candidate("logN_L", ("ln_N", "L_k")),
    Candidate("density", ("density_per_1k",)),
    Candidate(
        "linear_interaction",
        ("N", "L_k", "N_x_L_k"),
    ),
    Candidate(
        "log_interaction",
        ("ln_N", "ln_L_k", "ln_N_x_ln_L_k"),
    ),
)

CONTINUOUS_TARGETS = (
    "signed_mean_deviation",
    "absolute_mean_deviation",
    "signed_median_deviation",
    "signed_trimmed_mean_deviation",
    "signed_deviation_sample_variance",
)
REGISTERED_MODE_ORDER = (
    "direct",
    "enumeration_index",
    "enumeration_bullet",
    "native_thinking",
)

FEATURE_LABELS = {
    "N": "N",
    "L_k": "L/1000",
    "ln_N": "ln(N)",
    "ln_L_k": "ln(L/1000)",
    "density_per_1k": "N/(L/1000)",
    "N_x_L_k": "N(L/1000)",
    "ln_N_x_ln_L_k": "ln(N)ln(L/1000)",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison_slot(model_label: str) -> str:
    if model_label in MATCHED_REASONING_PAIRS:
        control = MATCHED_REASONING_PAIRS[model_label]
        if model_label.startswith("GLM"):
            return "GLM-4/Z1-9B"
        if model_label.startswith("Ministral"):
            return "Ministral-3-8B pair"
        return f"{control}/{model_label}"
    if model_label in set(MATCHED_REASONING_PAIRS.values()):
        reasoning = next(
            target
            for target, control in MATCHED_REASONING_PAIRS.items()
            if control == model_label
        )
        if model_label.startswith("GLM"):
            return "GLM-4/Z1-9B"
        if model_label.startswith("Ministral"):
            return "Ministral-3-8B pair"
        return f"{model_label}/{reasoning}"
    return model_label


def _discover_request_files(run_root: Path) -> list[Path]:
    files: list[Path] = []
    for collection in ("models", "matched_controls", "matched_reasoning"):
        files.extend(
            sorted((run_root / collection).glob("*/main/requests.jsonl"))
        )
    if not files:
        raise FileNotFoundError(
            f"No canonical V3 request files were found below {run_root}"
        )
    return files


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def _source_record(
    path: Path,
    *,
    root: Path,
    rows: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def load_request_table(
    run_root: str | Path,
    *,
    require_final_audit: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root = Path(run_root).resolve()
    audit_path = root / "orchestration" / "final_shard_audit.json"
    if require_final_audit:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("passed") is not True
            or audit.get("protocol_version") != PROTOCOL_VERSION
            or audit.get("audit_only") is not False
        ):
            raise RuntimeError(
                "V3 final merge audit has not passed with canonical outputs"
            )

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    provenance_paths = (
        root / "dataset" / "stimuli.jsonl",
        root / "dataset" / "manifest.json",
        root / "dataset" / "audit_report.json",
        root / "orchestration" / "formal_shards.json",
        audit_path,
    )
    for path in provenance_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/non-empty V3 provenance file: {path}")
        row_count = None
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object: {path}")
        elif path.name.endswith(".jsonl"):
            row_count = sum(1 for _ in _load_jsonl(path))
        sources.append(_source_record(path, root=root, rows=row_count))

    for path in _discover_request_files(root):
        file_rows = list(_load_jsonl(path))
        manifest_path = path.with_name("run_manifest.json")
        qc_path = path.with_name("qc_report.json")
        if (
            not manifest_path.is_file()
            or manifest_path.stat().st_size == 0
            or not qc_path.is_file()
            or qc_path.stat().st_size == 0
        ):
            raise FileNotFoundError(
                f"Canonical request file lacks manifest/QC siblings: {path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol_version") != PROTOCOL_VERSION
            or int(manifest.get("completed_requests", -1)) != len(file_rows)
            or qc.get("protocol_version") != PROTOCOL_VERSION
            or qc.get("passed") is not True
            or int(qc.get("completed_requests", -1)) != len(file_rows)
        ):
            raise RuntimeError(
                f"Canonical manifest/QC failed source validation: {path}"
            )
        sources.extend(
            (
                _source_record(path, root=root, rows=len(file_rows)),
                _source_record(manifest_path, root=root),
                _source_record(qc_path, root=root),
            )
        )
        for row in file_rows:
            if row.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError(f"Non-V3 request row in {path}")
            evaluation = row["evaluation"]
            predicted = evaluation.get("predicted_count")
            gold = int(row["gold_count"])
            signed = None if predicted is None else int(predicted) - gold
            prompt_mode = str(row["prompt_mode"])
            reasoning_text = str(evaluation.get("reasoning_text") or "")
            native_flags = native_thinking_style_flags(reasoning_text)
            native_style = (
                classify_native_thinking_style(reasoning_text)
                if prompt_mode == "native_thinking"
                else "not_applicable"
            )
            rows.append(
                {
                    "request_id": str(row["request_id"]),
                    "model_label": str(row["model_label"]),
                    "comparison_slot": _comparison_slot(
                        str(row["model_label"])
                    ),
                    "model_id": str(row["model_id"]),
                    "model_revision": str(row["model_revision"]),
                    "prompt_mode": prompt_mode,
                    "stimulus_id": str(row["stimulus_id"]),
                    "seed": int(row["seed"]),
                    "N": gold,
                    "L": int(row["target_passage_tokens"]),
                    "predicted_count": (
                        np.nan if predicted is None else int(predicted)
                    ),
                    "parse_success": (
                        evaluation["parse_status"] != "parse_fail"
                    ),
                    "parse_status": str(evaluation["parse_status"]),
                    "exact_count": bool(evaluation["exact_count"]),
                    "strict_registered_success": bool(
                        evaluation["registered_success"]
                    ),
                    "format_compliant": bool(
                        evaluation["response_format_compliant"]
                    ),
                    "truncated": bool(evaluation["truncated"]),
                    "signed_deviation": (
                        np.nan if signed is None else float(signed)
                    ),
                    "absolute_deviation": (
                        np.nan if signed is None else float(abs(signed))
                    ),
                    "output_tokens": int(row.get("output_tokens", 0)),
                    "finish_reason": row.get("finish_reason"),
                    "reasoning_expected": bool(
                        row.get("reasoning_expected", False)
                    ),
                    "reasoning_text": reasoning_text,
                    "final_text": str(evaluation.get("final_text") or ""),
                    "native_thinking_style": native_style,
                    **native_flags,
                    "source_file": str(path.relative_to(root)),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("The V3 request table is empty")
    duplicates = table["request_id"].duplicated(keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate V3 request IDs: "
            f"{table.loc[duplicates, 'request_id'].head(3).tolist()}"
        )
    return add_derived_predictors(table), sources


def add_derived_predictors(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["L_k"] = result["L"].astype(float) / 1000.0
    result["ln_N"] = np.log(result["N"].astype(float))
    result["ln_L_k"] = np.log(result["L_k"])
    result["density_per_1k"] = result["N"] / result["L_k"]
    result["N_x_L_k"] = result["N"] * result["L_k"]
    result["ln_N_x_ln_L_k"] = result["ln_N"] * result["ln_L_k"]
    return result


def exclusive_outcome_class(row: pd.Series) -> str:
    if bool(row["truncated"]):
        return "truncation"
    if not bool(row["parse_success"]):
        return "parse_failure"
    if row["signed_deviation"] < 0:
        return "undercount"
    if row["signed_deviation"] > 0:
        return "overcount"
    if not bool(row["format_compliant"]):
        return "format_only_failure"
    return "strict_success"


def behavior_tables(
    requests: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = requests.copy()
    working["outcome_class"] = working.apply(
        exclusive_outcome_class,
        axis=1,
    )
    group = [
        "comparison_slot",
        "model_label",
        "prompt_mode",
    ]
    summary = (
        working.groupby(group, dropna=False)
        .agg(
            requests=("request_id", "size"),
            parse_rate=("parse_success", "mean"),
            parseable_exact_accuracy=("exact_count", "mean"),
            strict_registered_accuracy=(
                "strict_registered_success",
                "mean",
            ),
            format_compliance_rate=("format_compliant", "mean"),
            truncation_rate=("truncated", "mean"),
            mean_signed_deviation=("signed_deviation", "mean"),
            median_signed_deviation=("signed_deviation", "median"),
            mean_absolute_deviation=("absolute_deviation", "mean"),
            median_absolute_deviation=("absolute_deviation", "median"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
    )
    by_condition = (
        working.groupby(
            [
                "comparison_slot",
                "model_label",
                "prompt_mode",
                "N",
                "L",
            ],
            dropna=False,
        )
        .agg(
            requests=("request_id", "size"),
            parse_rate=("parse_success", "mean"),
            parseable_exact_accuracy=("exact_count", "mean"),
            strict_registered_accuracy=(
                "strict_registered_success",
                "mean",
            ),
            mean_signed_deviation=("signed_deviation", "mean"),
            median_signed_deviation=("signed_deviation", "median"),
            mean_absolute_deviation=("absolute_deviation", "mean"),
            signed_deviation_sample_variance=(
                "signed_deviation",
                "var",
            ),
        )
        .reset_index()
    )
    outcomes = (
        working.groupby(
            ["comparison_slot", "prompt_mode", "outcome_class"],
            dropna=False,
        )
        .size()
        .rename("requests")
        .reset_index()
    )
    totals = outcomes.groupby(
        ["comparison_slot", "prompt_mode"]
    )["requests"].transform("sum")
    outcomes["proportion"] = outcomes["requests"] / totals
    return summary, by_condition, outcomes


def native_thinking_style_tables(
    requests: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize observable native-thinking counting devices.

    The summary is descriptive: a textual marker does not prove that the
    corresponding visible list or tally causally produced the final answer.
    Style shares use every native-thinking request as the denominator.
    """

    native = requests.loc[
        requests["prompt_mode"] == "native_thinking"
    ].copy()
    if native.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    if "native_thinking_style" not in native:
        reasoning = native.get(
            "reasoning_text",
            pd.Series("", index=native.index),
        )
        native["native_thinking_style"] = reasoning.map(
            classify_native_thinking_style
        )
    native["native_thinking_style"] = pd.Categorical(
        native["native_thinking_style"],
        categories=NATIVE_THINKING_STYLE_ORDER,
        ordered=True,
    )
    grouping = [
        "comparison_slot",
        "model_label",
        "native_thinking_style",
    ]
    summary = (
        native.groupby(grouping, observed=True, dropna=False)
        .agg(
            requests=("request_id", "size"),
            parse_rate=("parse_success", "mean"),
            parseable_exact_accuracy=("exact_count", "mean"),
            strict_registered_accuracy=(
                "strict_registered_success",
                "mean",
            ),
            format_compliance_rate=("format_compliant", "mean"),
            truncation_rate=("truncated", "mean"),
            mean_signed_deviation=("signed_deviation", "mean"),
            mean_absolute_deviation=("absolute_deviation", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
    )
    totals = summary.groupby(
        ["comparison_slot", "model_label"],
        observed=True,
    )["requests"].transform("sum")
    summary["style_share_within_model"] = summary["requests"] / totals

    by_condition = (
        native.groupby(
            grouping + ["N", "L"],
            observed=True,
            dropna=False,
        )
        .agg(
            requests=("request_id", "size"),
            parse_rate=("parse_success", "mean"),
            parseable_exact_accuracy=("exact_count", "mean"),
            truncation_rate=("truncated", "mean"),
            mean_signed_deviation=("signed_deviation", "mean"),
        )
        .reset_index()
    )

    example_columns = [
        "comparison_slot",
        "model_label",
        "native_thinking_style",
        "request_id",
        "stimulus_id",
        "seed",
        "N",
        "L",
        "exact_count",
        "parse_success",
        "truncated",
        "reasoning_text",
        "final_text",
    ]
    examples = (
        native.sort_values(
            [
                "model_label",
                "native_thinking_style",
                "truncated",
                "exact_count",
                "N",
                "L",
                "request_id",
            ],
            ascending=[True, True, True, False, True, True, True],
        )
        .groupby(
            ["model_label", "native_thinking_style"],
            observed=True,
            as_index=False,
        )
        .head(1)[example_columns]
        .reset_index(drop=True)
    )
    return summary, by_condition, examples


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, original_index in enumerate(order):
        running = max(running, (total - rank) * values[original_index])
        adjusted[original_index] = min(1.0, running)
    return pd.Series(adjusted, index=p_values.index)


def paired_mode_comparisons(
    requests: pd.DataFrame,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_730,
) -> pd.DataFrame:
    """Compare exact accuracy using the shared stimulus as the paired unit.

    Confidence intervals resample the ten stimulus seeds as clusters, so all
    N×L cells derived from one seed remain together. Exact McNemar p-values
    use the request-level discordant pairs and are Holm-adjusted within each
    behavior-comparison slot.
    """

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    required = {
        "comparison_slot",
        "prompt_mode",
        "stimulus_id",
        "seed",
        "exact_count",
    }
    missing = sorted(required - set(requests.columns))
    if missing:
        raise ValueError(
            "Paired mode comparison is missing columns: "
            + ", ".join(missing)
        )
    duplicate = requests.duplicated(
        ["comparison_slot", "prompt_mode", "stimulus_id"],
        keep=False,
    )
    if duplicate.any():
        raise ValueError(
            "Each comparison slot must have at most one request per "
            "prompt mode and stimulus"
        )

    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    for slot, slot_rows in requests.groupby("comparison_slot", sort=True):
        observed_modes = set(slot_rows["prompt_mode"].astype(str))
        modes = [
            mode for mode in REGISTERED_MODE_ORDER if mode in observed_modes
        ]
        for mode_a, mode_b in combinations(modes, 2):
            left = slot_rows.loc[
                slot_rows["prompt_mode"] == mode_a,
                ["stimulus_id", "seed", "exact_count"],
            ].rename(columns={"exact_count": "correct_a"})
            right = slot_rows.loc[
                slot_rows["prompt_mode"] == mode_b,
                ["stimulus_id", "seed", "exact_count"],
            ].rename(
                columns={
                    "seed": "seed_b",
                    "exact_count": "correct_b",
                }
            )
            paired = left.merge(
                right,
                on="stimulus_id",
                how="inner",
                validate="one_to_one",
            )
            if paired.empty:
                continue
            if not (paired["seed"] == paired["seed_b"]).all():
                raise ValueError("Paired prompt modes disagree on seed")
            paired["difference"] = (
                paired["correct_b"].astype(float)
                - paired["correct_a"].astype(float)
            )
            seed_effects = (
                paired.groupby("seed")["difference"]
                .mean()
                .to_numpy(dtype=float)
            )
            sampled = rng.integers(
                0,
                len(seed_effects),
                size=(bootstrap_replicates, len(seed_effects)),
            )
            bootstrap = seed_effects[sampled].mean(axis=1)
            b_wins = int(
                (
                    ~paired["correct_a"].astype(bool)
                    & paired["correct_b"].astype(bool)
                ).sum()
            )
            a_wins = int(
                (
                    paired["correct_a"].astype(bool)
                    & ~paired["correct_b"].astype(bool)
                ).sum()
            )
            discordant = a_wins + b_wins
            p_value = (
                float(
                    binomtest(
                        b_wins,
                        n=discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant
                else 1.0
            )
            rows.append(
                {
                    "comparison_slot": str(slot),
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "paired_stimuli": len(paired),
                    "paired_seeds": len(seed_effects),
                    "accuracy_a": float(
                        paired["correct_a"].astype(float).mean()
                    ),
                    "accuracy_b": float(
                        paired["correct_b"].astype(float).mean()
                    ),
                    "risk_difference_b_minus_a": float(
                        paired["difference"].mean()
                    ),
                    "cluster_bootstrap_ci95_low": float(
                        np.quantile(bootstrap, 0.025)
                    ),
                    "cluster_bootstrap_ci95_high": float(
                        np.quantile(bootstrap, 0.975)
                    ),
                    "a_only_correct": a_wins,
                    "b_only_correct": b_wins,
                    "mcnemar_exact_p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["mcnemar_holm_p_value_within_slot"] = (
        result.groupby("comparison_slot", group_keys=False)[
            "mcnemar_exact_p_value"
        ].apply(_holm_adjust)
    )
    return result


def _trimmed_mean(values: pd.Series, proportion: float = 0.1) -> float:
    clean = np.sort(values.dropna().to_numpy(dtype=float))
    if not len(clean):
        return math.nan
    trim = int(math.floor(proportion * len(clean)))
    if trim == 0 or 2 * trim >= len(clean):
        return float(clean.mean())
    return float(clean[trim:-trim].mean())


def aggregate_continuous_target(
    requests: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    keys = ["comparison_slot", "N", "L"]
    if target == "signed_mean_deviation":
        grouped = requests.groupby(keys)["signed_deviation"]
        result = grouped.mean().rename("target").reset_index()
        counts = grouped.count().rename("observations").reset_index()
    elif target == "absolute_mean_deviation":
        grouped = requests.groupby(keys)["absolute_deviation"]
        result = grouped.mean().rename("target").reset_index()
        counts = grouped.count().rename("observations").reset_index()
    elif target == "signed_median_deviation":
        grouped = requests.groupby(keys)["signed_deviation"]
        result = grouped.median().rename("target").reset_index()
        counts = grouped.count().rename("observations").reset_index()
    elif target == "signed_trimmed_mean_deviation":
        grouped = requests.groupby(keys)["signed_deviation"]
        result = grouped.apply(_trimmed_mean).rename("target").reset_index()
        counts = grouped.count().rename("observations").reset_index()
    elif target == "signed_deviation_sample_variance":
        grouped = requests.groupby(keys)["signed_deviation"]
        result = grouped.var(ddof=1).rename("target").reset_index()
        counts = grouped.count().rename("observations").reset_index()
    else:
        raise ValueError(f"Unsupported continuous target: {target}")
    result = result.merge(counts, on=keys, validate="one_to_one")
    return add_derived_predictors(result.dropna(subset=["target"]))


def aggregate_accuracy_target(requests: pd.DataFrame) -> pd.DataFrame:
    """Aggregate request-level Bernoulli outcomes into Binomial cells."""

    keys = ["comparison_slot", "N", "L"]
    result = (
        requests.groupby(keys, dropna=False)["exact_count"]
        .agg(successes="sum", trials="size")
        .reset_index()
    )
    result["successes"] = result["successes"].astype(int)
    result["trials"] = result["trials"].astype(int)
    return add_derived_predictors(result)


def _design_matrix(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    model_levels: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]
    labels = frame["comparison_slot"].astype(str)
    for level in model_levels[1:]:
        columns.append((labels == level).to_numpy(dtype=float))
        names.append(f"model[{level}]")
    for feature in features:
        values = frame[feature].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in feature {feature}")
        columns.append(values)
        names.append(feature)
    return np.column_stack(columns), names


def _ols_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residual = y - fitted
    degrees = max(1, len(y) - x.shape[1])
    sigma2 = float(residual @ residual) / degrees
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    statistic = np.divide(
        beta,
        standard_error,
        out=np.zeros_like(beta),
        where=standard_error > 0,
    )
    p_value = 2.0 * t.sf(np.abs(statistic), df=degrees)
    return beta, fitted, standard_error, p_value


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return math.nan
    return float(r2_score(y_true, y_pred))


def _coefficient_rows(
    *,
    target: str,
    prompt_mode: str,
    candidate: Candidate,
    names: list[str],
    beta: np.ndarray,
    standard_error: np.ndarray,
    p_value: np.ndarray,
    distribution_family: str = "gaussian_ols",
    selected: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, estimate, error, probability in zip(
        names,
        beta,
        standard_error,
        p_value,
    ):
        rows.append(
            {
                "target": target,
                "prompt_mode": prompt_mode,
                "candidate": candidate.name,
                "distribution_family": distribution_family,
                "term": name,
                "estimate": float(estimate),
                "standard_error": float(error),
                "p_value": float(probability),
                "ci95_low": float(estimate - 1.96 * error),
                "ci95_high": float(estimate + 1.96 * error),
                "is_feature": name in candidate.features,
                "is_interaction": "_x_" in name,
                "selected_candidate": selected,
            }
        )
    return rows


def _interaction_status(
    coefficient_rows: list[dict[str, Any]],
) -> tuple[bool, float | None]:
    interactions = [
        row for row in coefficient_rows if bool(row["is_interaction"])
    ]
    if not interactions:
        return True, None
    maximum = max(float(row["p_value"]) for row in interactions)
    return maximum < 0.05, maximum


def cross_validate_continuous(
    requests: pd.DataFrame,
    *,
    prompt_mode: str,
    target: str,
    candidate: Candidate,
    n_splits: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mode_rows = requests.loc[
        requests["prompt_mode"] == prompt_mode
    ].copy()
    groups = np.array(sorted(mode_rows["seed"].unique()))
    splits = min(n_splits, len(groups))
    if splits < 2:
        raise ValueError("Grouped CV requires at least two unique seeds")
    levels = tuple(sorted(mode_rows["comparison_slot"].unique()))
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=splits)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(mode_rows, groups=mode_rows["seed"]),
        start=1,
    ):
        train = aggregate_continuous_target(
            mode_rows.iloc[train_index],
            target,
        )
        test = aggregate_continuous_target(
            mode_rows.iloc[test_index],
            target,
        )
        x_train, _ = _design_matrix(train, candidate.features, levels)
        x_test, _ = _design_matrix(test, candidate.features, levels)
        beta, _, _, _ = _ols_fit(
            x_train,
            train["target"].to_numpy(dtype=float),
        )
        predicted = x_test @ beta
        observed = test["target"].to_numpy(dtype=float)
        fold_rows.append(
            {
                "fold": fold,
                "r2": _safe_r2(observed, predicted),
                "mae": float(mean_absolute_error(observed, predicted)),
                "rmse": float(
                    mean_squared_error(observed, predicted) ** 0.5
                ),
                "test_cells": len(test),
                "test_seeds": ",".join(
                    str(value)
                    for value in sorted(
                        mode_rows.iloc[test_index]["seed"].unique()
                    )
                ),
            }
        )

    full = aggregate_continuous_target(mode_rows, target)
    x_full, names = _design_matrix(full, candidate.features, levels)
    beta, fitted, standard_error, p_value = _ols_fit(
        x_full,
        full["target"].to_numpy(dtype=float),
    )
    coefficients = _coefficient_rows(
        target=target,
        prompt_mode=prompt_mode,
        candidate=candidate,
        names=names,
        beta=beta,
        standard_error=standard_error,
        p_value=p_value,
    )
    interaction_significant, interaction_p = _interaction_status(coefficients)
    fold_frame = pd.DataFrame(fold_rows)
    result = {
        "target": target,
        "prompt_mode": prompt_mode,
        "candidate": candidate.name,
        "distribution_family": "gaussian_ols",
        "distribution": "gaussian",
        "link": "identity",
        "distribution_extra_parameters": 1,
        "features": ",".join(candidate.features),
        "feature_count": len(candidate.features),
        "cv_folds": len(fold_rows),
        "cv_r2_mean": float(fold_frame["r2"].mean()),
        "cv_r2_sd": float(fold_frame["r2"].std(ddof=1)),
        "cv_mae_mean": float(fold_frame["mae"].mean()),
        "cv_mae_sd": float(fold_frame["mae"].std(ddof=1)),
        "cv_rmse_mean": float(fold_frame["rmse"].mean()),
        "full_r2": _safe_r2(
            full["target"].to_numpy(dtype=float),
            fitted,
        ),
        "full_mae": float(
            mean_absolute_error(full["target"], fitted)
        ),
        "interaction_significant_0_05": interaction_significant,
        "maximum_interaction_p_value": interaction_p,
        "converged": True,
    }
    return result, coefficients


def _aggregate_bernoulli_log_loss(
    successes: np.ndarray,
    trials: np.ndarray,
    probability: np.ndarray,
) -> float:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    loss = -(
        successes * np.log(probability)
        + (trials - successes) * np.log1p(-probability)
    ).sum()
    return float(loss / trials.sum())


def _aggregate_brier_score(
    successes: np.ndarray,
    trials: np.ndarray,
    probability: np.ndarray,
) -> float:
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    probability = np.asarray(probability, dtype=float)
    total = (
        successes * (1.0 - probability) ** 2
        + (trials - successes) * probability**2
    ).sum()
    return float(total / trials.sum())


def cross_validate_accuracy(
    requests: pd.DataFrame,
    *,
    prompt_mode: str,
    candidate: Candidate,
    family: AccuracyFamily = ACCURACY_FAMILIES[0],
    n_splits: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one explicit accuracy distribution with grouped seed CV."""

    mode_rows = requests.loc[
        requests["prompt_mode"] == prompt_mode
    ].copy()
    groups = mode_rows["seed"].to_numpy()
    splits = min(n_splits, mode_rows["seed"].nunique())
    if splits < 2:
        raise ValueError("Grouped CV requires at least two unique seeds")
    levels = tuple(sorted(mode_rows["comparison_slot"].unique()))
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=splits)
    all_converged = True
    for fold, (train_index, test_index) in enumerate(
        splitter.split(mode_rows, groups=groups),
        start=1,
    ):
        train_requests = mode_rows.iloc[train_index]
        test_requests = mode_rows.iloc[test_index]
        train = aggregate_accuracy_target(train_requests)
        test = aggregate_accuracy_target(test_requests)
        x_train, _ = _design_matrix(train, candidate.features, levels)
        x_test, _ = _design_matrix(test, candidate.features, levels)
        fit = fit_accuracy_distribution(
            x_train,
            train["successes"].to_numpy(dtype=int),
            train["trials"].to_numpy(dtype=int),
            family,
            compute_inference=False,
        )
        probability = inverse_link(x_test @ fit.beta, family.link)
        successes = test["successes"].to_numpy(dtype=int)
        trials = test["trials"].to_numpy(dtype=int)
        all_converged = all_converged and fit.converged
        predictive_nlpd = -predictive_log_likelihood(
            successes,
            trials,
            probability,
            family,
            concentration=fit.concentration,
        ) / float(trials.sum())
        probability_loss = _aggregate_bernoulli_log_loss(
            successes,
            trials,
            probability,
        )
        model_prevalence = train_requests.groupby("comparison_slot")[
            "exact_count"
        ].mean()
        fallback = float(train_requests["exact_count"].mean())
        baseline = (
            test["comparison_slot"]
            .map(model_prevalence)
            .fillna(fallback)
            .clip(1e-8, 1 - 1e-8)
            .to_numpy(dtype=float)
        )
        baseline_loss = _aggregate_bernoulli_log_loss(
            successes,
            trials,
            baseline,
        )
        fold_rows.append(
            {
                "fold": fold,
                "predictive_nlpd": float(predictive_nlpd),
                "log_loss": probability_loss,
                "brier": _aggregate_brier_score(
                    successes,
                    trials,
                    probability,
                ),
                "deviance_explained": (
                    1.0 - probability_loss / baseline_loss
                    if baseline_loss > 0
                    else math.nan
                ),
                "test_requests": int(trials.sum()),
                "test_cells": len(test),
            }
        )

    full = aggregate_accuracy_target(mode_rows)
    x_full, names = _design_matrix(full, candidate.features, levels)
    successes_full = full["successes"].to_numpy(dtype=int)
    trials_full = full["trials"].to_numpy(dtype=int)
    fit = fit_accuracy_distribution(
        x_full,
        successes_full,
        trials_full,
        family,
        compute_inference=True,
    )
    coefficients = _coefficient_rows(
        target="parseable_exact_accuracy",
        prompt_mode=prompt_mode,
        candidate=candidate,
        names=names,
        beta=fit.beta,
        standard_error=fit.standard_error,
        p_value=fit.p_value,
        distribution_family=family.name,
    )
    interaction_significant, interaction_p = _interaction_status(coefficients)
    fold_frame = pd.DataFrame(fold_rows)
    full_probability_loss = _aggregate_bernoulli_log_loss(
        successes_full,
        trials_full,
        fit.probability,
    )
    full_predictive_nlpd = -fit.log_likelihood / float(trials_full.sum())
    concentration = fit.concentration
    result = {
        "target": "parseable_exact_accuracy",
        "prompt_mode": prompt_mode,
        "candidate": candidate.name,
        "distribution_family": family.name,
        "distribution": family.distribution,
        "link": family.link,
        "distribution_extra_parameters": family.extra_parameters,
        "features": ",".join(candidate.features),
        "feature_count": len(candidate.features),
        "cv_folds": len(fold_rows),
        "cv_predictive_nlpd_mean": float(
            fold_frame["predictive_nlpd"].mean()
        ),
        "cv_predictive_nlpd_sd": float(
            fold_frame["predictive_nlpd"].std(ddof=1)
        ),
        "cv_log_loss_mean": float(fold_frame["log_loss"].mean()),
        "cv_log_loss_sd": float(fold_frame["log_loss"].std(ddof=1)),
        "cv_brier_mean": float(fold_frame["brier"].mean()),
        "cv_brier_sd": float(fold_frame["brier"].std(ddof=1)),
        "cv_deviance_explained_mean": float(
            fold_frame["deviance_explained"].mean()
        ),
        "full_predictive_nlpd": float(full_predictive_nlpd),
        "full_log_loss": float(full_probability_loss),
        "full_brier": _aggregate_brier_score(
            successes_full,
            trials_full,
            fit.probability,
        ),
        "beta_binomial_concentration": concentration,
        "beta_binomial_intraclass_correlation": (
            None if concentration is None else 1.0 / (concentration + 1.0)
        ),
        "interaction_significant_0_05": interaction_significant,
        "maximum_interaction_p_value": interaction_p,
        "converged": bool(fit.converged and all_converged),
    }
    return result, coefficients


def _select_model(
    comparison: pd.DataFrame,
    target: str,
) -> tuple[str, str]:
    """Select one response surface and, for accuracy, one distribution.

    The held-out predictive distribution is primary for accuracy.  Among
    models within a one-standard-error-style tolerance, we prefer fewer
    response-surface terms and then fewer distributional parameters.
    """

    eligible = comparison.loc[comparison["converged"].astype(bool)].copy()
    interaction_ok = eligible["interaction_significant_0_05"].astype(bool)
    eligible = eligible.loc[
        interaction_ok
        | ~eligible["candidate"].str.contains("interaction")
    ]
    if eligible.empty:
        raise RuntimeError(
            f"No converged, eligible candidate remained for target {target}"
        )
    if target == "parseable_exact_accuracy":
        best_index = eligible["cv_predictive_nlpd_mean"].idxmin()
        best = eligible.loc[best_index]
        tolerance = max(
            0.002,
            float(best["cv_predictive_nlpd_sd"])
            / math.sqrt(float(best["cv_folds"])),
        )
        near = eligible.loc[
            eligible["cv_predictive_nlpd_mean"]
            <= float(best["cv_predictive_nlpd_mean"]) + tolerance
        ]
        selected = near.sort_values(
            [
                "feature_count",
                "distribution_extra_parameters",
                "cv_brier_mean",
                "distribution_family",
                "candidate",
            ]
        ).iloc[0]
        return (
            str(selected["candidate"]),
            str(selected["distribution_family"]),
        )
    best_index = eligible["cv_r2_mean"].idxmax()
    best = eligible.loc[best_index]
    tolerance = max(
        0.02,
        float(best["cv_r2_sd"])
        / math.sqrt(float(best["cv_folds"])),
    )
    near = eligible.loc[
        eligible["cv_r2_mean"]
        >= float(best["cv_r2_mean"]) - tolerance
    ]
    selected = near.sort_values(
        ["feature_count", "cv_mae_mean", "candidate"]
    ).iloc[0]
    return (
        str(selected["candidate"]),
        str(selected.get("distribution_family", "gaussian_ols")),
    )


def _select_candidate(comparison: pd.DataFrame, target: str) -> str:
    """Compatibility wrapper used by focused selection tests."""

    return _select_model(comparison, target)[0]


def fit_candidate_grid(
    requests: pd.DataFrame,
    *,
    targets: tuple[str, ...] = (
        "parseable_exact_accuracy",
        *CONTINUOUS_TARGETS,
    ),
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    modes = tuple(sorted(requests["prompt_mode"].unique()))
    for target in targets:
        for prompt_mode in modes:
            if target == "parseable_exact_accuracy":
                for family in ACCURACY_FAMILIES:
                    for candidate in CANDIDATES:
                        result, terms = cross_validate_accuracy(
                            requests,
                            prompt_mode=prompt_mode,
                            candidate=candidate,
                            family=family,
                            n_splits=n_splits,
                        )
                        comparisons.append(result)
                        coefficients.extend(terms)
            else:
                for candidate in CANDIDATES:
                    result, terms = cross_validate_continuous(
                        requests,
                        prompt_mode=prompt_mode,
                        target=target,
                        candidate=candidate,
                        n_splits=n_splits,
                    )
                    comparisons.append(result)
                    coefficients.extend(terms)

    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["selected"] = False
    selected_rows: list[dict[str, Any]] = []
    for (target, prompt_mode), group in comparison_frame.groupby(
        ["target", "prompt_mode"],
        sort=True,
    ):
        selected_name, selected_family = _select_model(group, str(target))
        selected = group.loc[
            (group["candidate"] == selected_name)
            & (group["distribution_family"] == selected_family)
        ].iloc[0]
        selected_rows.append(selected.to_dict())
        comparison_frame.loc[
            (
                (comparison_frame["target"] == target)
                & (comparison_frame["prompt_mode"] == prompt_mode)
                & (comparison_frame["candidate"] == selected_name)
                & (
                    comparison_frame["distribution_family"]
                    == selected_family
                )
            ),
            "selected",
        ] = True
    comparison_frame["selected"] = comparison_frame["selected"].astype(bool)
    selected_frame = pd.DataFrame(selected_rows)
    coefficient_frame = pd.DataFrame(coefficients)
    selected_keys = {
        (
            str(row["target"]),
            str(row["prompt_mode"]),
            str(row["candidate"]),
            str(row["distribution_family"]),
        )
        for row in selected_rows
    }
    coefficient_frame["selected_candidate"] = [
        (
            str(row.target),
            str(row.prompt_mode),
            str(row.candidate),
            str(row.distribution_family),
        )
        in selected_keys
        for row in coefficient_frame.itertuples()
    ]
    return comparison_frame, selected_frame, coefficient_frame


def predict_selected_law(
    requests: pd.DataFrame,
    *,
    target: str,
    prompt_mode: str,
    candidate_name: str,
    distribution_family: str = "binomial_logit",
) -> pd.DataFrame:
    candidate = next(
        item for item in CANDIDATES if item.name == candidate_name
    )
    mode_rows = requests.loc[
        requests["prompt_mode"] == prompt_mode
    ].copy()
    levels = tuple(sorted(mode_rows["comparison_slot"].unique()))
    if target == "parseable_exact_accuracy":
        family = ACCURACY_FAMILY_BY_NAME[distribution_family]
        condition = aggregate_accuracy_target(mode_rows)
        x, _ = _design_matrix(condition, candidate.features, levels)
        fit = fit_accuracy_distribution(
            x,
            condition["successes"].to_numpy(dtype=int),
            condition["trials"].to_numpy(dtype=int),
            family,
            compute_inference=False,
        )
        condition["observed"] = condition["successes"] / condition["trials"]
        condition["observations"] = condition["trials"]
        condition["predicted"] = fit.probability
        return condition
    condition = aggregate_continuous_target(mode_rows, target)
    x, _ = _design_matrix(condition, candidate.features, levels)
    beta, _, _, _ = _ols_fit(
        x,
        condition["target"].to_numpy(dtype=float),
    )
    condition = condition.rename(columns={"target": "observed"})
    condition["predicted"] = x @ beta
    return condition


def formula_text(
    target: str,
    candidate_name: str,
    distribution_family: str = "binomial_logit",
) -> str:
    candidate = next(
        item for item in CANDIDATES if item.name == candidate_name
    )
    if target == "parseable_exact_accuracy":
        family = ACCURACY_FAMILY_BY_NAME[distribution_family]
        link_label = {
            "logit": "logit",
            "probit": "probit",
            "cloglog": "cloglog",
        }[family.link]
        response = f"{link_label}(P(correct))"
    else:
        response = target
    pieces = ["alpha_model"]
    pieces.extend(
        f"beta_{index + 1}*{FEATURE_LABELS[feature]}"
        for index, feature in enumerate(candidate.features)
    )
    return f"{response} = " + " + ".join(pieces)


def accuracy_distribution_diagnostics(
    requests: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    random_seed: int = 20260802,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit selected accuracy laws and compute Dunn--Smyth Q--Q diagnostics."""

    selected_accuracy = selected.loc[
        selected["target"] == "parseable_exact_accuracy"
    ].sort_values("prompt_mode")
    residual_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for mode_index, row in enumerate(selected_accuracy.itertuples()):
        prompt_mode = str(row.prompt_mode)
        candidate_name = str(row.candidate)
        family_name = str(row.distribution_family)
        candidate = next(
            item for item in CANDIDATES if item.name == candidate_name
        )
        family = ACCURACY_FAMILY_BY_NAME[family_name]
        mode_rows = requests.loc[
            requests["prompt_mode"] == prompt_mode
        ].copy()
        condition = aggregate_accuracy_target(mode_rows)
        levels = tuple(sorted(condition["comparison_slot"].unique()))
        design, _ = _design_matrix(condition, candidate.features, levels)
        successes = condition["successes"].to_numpy(dtype=int)
        trials = condition["trials"].to_numpy(dtype=int)
        fit = fit_accuracy_distribution(
            design,
            successes,
            trials,
            family,
            compute_inference=False,
        )
        seed = random_seed + mode_index
        residuals = randomized_quantile_residuals(
            successes,
            trials,
            fit.probability,
            family,
            concentration=fit.concentration,
            random_seed=seed,
        )
        ordered_indices = np.argsort(residuals)
        theoretical = np.full(len(residuals), np.nan, dtype=float)
        theoretical[ordered_indices] = scipy.stats.norm.ppf(
            (np.arange(len(residuals)) + 0.5) / len(residuals)
        )
        for condition_row, probability, residual, quantile in zip(
            condition.to_dict(orient="records"),
            fit.probability,
            residuals,
            theoretical,
        ):
            residual_rows.append(
                {
                    "prompt_mode": prompt_mode,
                    "candidate": candidate_name,
                    "distribution_family": family_name,
                    "comparison_slot": condition_row["comparison_slot"],
                    "N": int(condition_row["N"]),
                    "L": int(condition_row["L"]),
                    "successes": int(condition_row["successes"]),
                    "trials": int(condition_row["trials"]),
                    "fitted_probability": float(probability),
                    "randomized_quantile_residual": float(residual),
                    "theoretical_normal_quantile": float(quantile),
                    "random_seed": seed,
                }
            )
        diagnostics = quantile_residual_diagnostics(residuals)
        diagnostic_rows.append(
            {
                "prompt_mode": prompt_mode,
                "candidate": candidate_name,
                "distribution_family": family_name,
                "distribution": family.distribution,
                "link": family.link,
                "converged": fit.converged,
                "beta_binomial_concentration": fit.concentration,
                "beta_binomial_intraclass_correlation": (
                    None
                    if fit.concentration is None
                    else 1.0 / (fit.concentration + 1.0)
                ),
                "random_seed": seed,
                **diagnostics,
            }
        )
    return pd.DataFrame(residual_rows), pd.DataFrame(diagnostic_rows)


def write_request_table_gzip(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        table.to_csv(handle, index=False)
    temporary.replace(path)


def analysis_manifest(
    *,
    run_root: Path,
    output_root: Path,
    sources: list[dict[str, Any]],
    requests: pd.DataFrame,
    output_files: Iterable[Path],
) -> dict[str, Any]:
    files = [
        {
            "path": str(path),
            "relative_path": str(path.relative_to(output_root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in output_files
        if path.is_file()
    ]
    return {
        "schema_version": "realistic_niah_v3_analysis_manifest_v2",
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "request_rows": len(requests),
        "unique_request_ids": int(requests["request_id"].nunique()),
        "candidate_grid": [asdict(candidate) for candidate in CANDIDATES],
        "accuracy_distribution_grid": [
            asdict(family) for family in ACCURACY_FAMILIES
        ],
        "native_thinking_style_order": list(
            NATIVE_THINKING_STYLE_ORDER
        ),
        "continuous_targets": list(CONTINUOUS_TARGETS),
        "primary_accuracy": (
            "predicted count is parsed and exactly equals N; parse, format, "
            "and truncation diagnostics are reported separately"
        ),
        "bias_estimand": (
            "predicted_count - N conditional on parse success; parse failures "
            "are never imputed and parse rate is reported alongside bias"
        ),
        "cross_validation": (
            "five-fold grouped by seed; every seed is wholly held out across "
            "all N, L, models, and modes in its fold"
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "sources": sources,
        "outputs": files,
    }
