from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from realistic_niah_v3.analysis import (
    paired_mode_comparisons as paired_mode_comparisons,
)

from .spec import (
    BIAS_TRIM_PROPORTION,
    EXPECTED_REQUESTS,
    MATCHED_REASONING_PAIRS,
    MINIMUM_PARSEABLE_PER_BIAS_CELL,
    PROTOCOL_VERSION,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison_slot(model_label: str) -> str:
    if model_label in MATCHED_REASONING_PAIRS:
        if model_label.startswith("GLM"):
            return "GLM-4/Z1-9B"
        if model_label.startswith("Ministral"):
            return "Ministral-3-8B pair"
    if model_label in set(MATCHED_REASONING_PAIRS.values()):
        if model_label.startswith("GLM"):
            return "GLM-4/Z1-9B"
        if model_label.startswith("Ministral"):
            return "Ministral-3-8B pair"
    return model_label


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def _discover_request_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for collection in ("models", "matched_controls", "matched_reasoning"):
        files.extend(sorted((root / collection).glob("*/main/requests.jsonl")))
    if not files:
        raise FileNotFoundError(f"No canonical V3.1 request files below {root}")
    return files


def add_derived_predictors(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["L_k"] = result["L"].astype(float) / 1000.0
    result["logN"] = np.log(result["N"].astype(float))
    result["logL"] = np.log(result["L_k"])
    result["N_x_L_k"] = result["N"].astype(float) * result["L_k"]
    result["logN_x_logL"] = result["logN"] * result["logL"]
    result["N_x_logL"] = result["N"].astype(float) * result["logL"]
    result["logN_x_L_k"] = result["logN"] * result["L_k"]
    return result


def load_request_table(
    run_root: str | Path,
    *,
    require_final_audit: bool = True,
    require_complete: bool = True,
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
            raise RuntimeError("V3.1 final merge audit has not passed")

    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in _discover_request_files(root):
        file_rows = list(_load_jsonl(path))
        manifest_path = path.with_name("run_manifest.json")
        qc_path = path.with_name("qc_report.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol_version") != PROTOCOL_VERSION
            or int(manifest.get("completed_requests", -1)) != len(file_rows)
            or qc.get("protocol_version") != PROTOCOL_VERSION
            or qc.get("passed") is not True
            or int(qc.get("completed_requests", -1)) != len(file_rows)
        ):
            raise RuntimeError(f"V3.1 manifest/QC validation failed: {path}")
        sources.extend(
            {
                "path": str(source.relative_to(root)),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
                **({"rows": len(file_rows)} if source == path else {}),
            }
            for source in (path, manifest_path, qc_path)
        )
        for row in file_rows:
            if row.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError(f"Non-V3.1 request row in {path}")
            evaluation = row["evaluation"]
            predicted = evaluation.get("predicted_count")
            gold = int(row["gold_count"])
            signed = None if predicted is None else int(predicted) - gold
            rows.append(
                {
                    "request_id": str(row["request_id"]),
                    "model_label": str(row["model_label"]),
                    "comparison_slot": _comparison_slot(str(row["model_label"])),
                    "model_id": str(row["model_id"]),
                    "model_revision": str(row["model_revision"]),
                    "prompt_mode": str(row["prompt_mode"]),
                    "stimulus_id": str(row["stimulus_id"]),
                    "seed": int(row["seed"]),
                    "N": gold,
                    "L": int(row["target_passage_tokens"]),
                    "predicted_count": np.nan if predicted is None else int(predicted),
                    "parse_success": evaluation["parse_status"] != "parse_fail",
                    "parse_status": str(evaluation["parse_status"]),
                    "exact_count": bool(evaluation["exact_count"]),
                    "strict_registered_success": bool(evaluation["registered_success"]),
                    "format_compliant": bool(evaluation["response_format_compliant"]),
                    "truncated": bool(evaluation["truncated"]),
                    "signed_deviation": np.nan if signed is None else float(signed),
                    "absolute_deviation": np.nan
                    if signed is None
                    else float(abs(signed)),
                    "output_tokens": int(row.get("output_tokens", 0)),
                    "finish_reason": row.get("finish_reason"),
                    "reasoning_expected": bool(row.get("reasoning_expected", False)),
                    "reasoning_text": str(evaluation.get("reasoning_text") or ""),
                    "final_text": str(evaluation.get("final_text") or ""),
                    "raw_output_text": str(row.get("raw_output_text") or ""),
                    "separate_reasoning_text": str(row.get("reasoning_content") or ""),
                    "source_file": str(path.relative_to(root)),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("The V3.1 request table is empty")
    duplicates = table["request_id"].duplicated(keep=False)
    if duplicates.any():
        raise ValueError("Duplicate V3.1 request IDs")
    if require_complete and len(table) != EXPECTED_REQUESTS:
        raise RuntimeError(
            f"Expected {EXPECTED_REQUESTS:,} V3.1 rows, found {len(table):,}"
        )
    return add_derived_predictors(table), sources


def symmetric_trimmed_mean(
    values: Iterable[float],
    proportion: float = BIAS_TRIM_PROPORTION,
) -> float:
    clean = np.sort(np.asarray(list(values), dtype=float))
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return math.nan
    if not 0.0 <= proportion < 0.5:
        raise ValueError("trim proportion must be in [0, 0.5)")
    trim = int(math.floor(proportion * len(clean)))
    if trim == 0:
        return float(clean.mean())
    return float(clean[trim:-trim].mean())


def _quantile(values: pd.Series, probability: float) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    return math.nan if not len(clean) else float(np.quantile(clean, probability))


def _top_error_share(values: pd.Series, proportion: float = 0.05) -> float:
    clean = np.sort(values.dropna().to_numpy(dtype=float))
    total = float(clean.sum())
    if not len(clean) or total <= 0:
        return 0.0 if len(clean) else math.nan
    count = max(1, int(math.ceil(proportion * len(clean))))
    return float(clean[-count:].sum() / total)


def bias_condition_table(
    requests: pd.DataFrame,
    *,
    minimum_parseable: int = MINIMUM_PARSEABLE_PER_BIAS_CELL,
) -> pd.DataFrame:
    keys = ["comparison_slot", "model_label", "prompt_mode", "N", "L"]
    rows: list[dict[str, Any]] = []
    for key, group in requests.groupby(keys, sort=True, dropna=False):
        signed = group["signed_deviation"].dropna()
        absolute = group["absolute_deviation"].dropna()
        parseable = int(signed.size)
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_total": int(len(group)),
                "n_parseable": parseable,
                "parse_rate": float(group["parse_success"].mean()),
                "trim_count_each_tail": int(math.floor(0.1 * parseable)),
                "trimmed_signed_bias_10": symmetric_trimmed_mean(signed),
                "mean_signed_deviation": float(signed.mean())
                if parseable
                else math.nan,
                "median_signed_deviation": float(signed.median())
                if parseable
                else math.nan,
                "mean_absolute_deviation": float(absolute.mean())
                if parseable
                else math.nan,
                "median_absolute_deviation": float(absolute.median())
                if parseable
                else math.nan,
                "absolute_deviation_q90": _quantile(absolute, 0.90),
                "absolute_deviation_q95": _quantile(absolute, 0.95),
                "maximum_absolute_deviation": float(absolute.max())
                if parseable
                else math.nan,
                "undercount_rate_among_parseable": (
                    float((signed < 0).mean()) if parseable else math.nan
                ),
                "overcount_rate_among_parseable": (
                    float((signed > 0).mean()) if parseable else math.nan
                ),
                "top_5_percent_absolute_error_share": _top_error_share(absolute),
                "bias_law_eligible": parseable >= minimum_parseable,
                "bias_coverage_status": (
                    "eligible"
                    if parseable >= minimum_parseable
                    else "insufficient_conditional_bias_coverage"
                ),
            }
        )
    return add_derived_predictors(pd.DataFrame(rows))


def accuracy_condition_table(requests: pd.DataFrame) -> pd.DataFrame:
    keys = ["comparison_slot", "model_label", "prompt_mode", "N", "L"]
    working = requests.copy()
    working["correct_and_format"] = working["exact_count"].astype(bool) & working[
        "format_compliant"
    ].astype(bool)
    table = (
        working.groupby(keys, sort=True, dropna=False)
        .agg(
            n_total=("request_id", "size"),
            n_parseable=("parse_success", "sum"),
            n_correct_parsed=("exact_count", "sum"),
            n_format_compliant=("format_compliant", "sum"),
            n_correct_and_format_compliant=(
                "correct_and_format",
                "sum",
            ),
            n_strict_success=("strict_registered_success", "sum"),
            n_truncated=("truncated", "sum"),
        )
        .reset_index()
    )
    table["parse_rate"] = table["n_parseable"] / table["n_total"]
    table["parsed_exact_accuracy"] = table["n_correct_parsed"] / table["n_total"]
    table["conditional_numeric_accuracy"] = np.where(
        table["n_parseable"] > 0,
        table["n_correct_parsed"] / table["n_parseable"],
        np.nan,
    )
    table["format_compliance_rate"] = table["n_format_compliant"] / table["n_total"]
    table["strict_accuracy"] = table["n_strict_success"] / table["n_total"]
    table["truncation_rate"] = table["n_truncated"] / table["n_total"]
    return add_derived_predictors(table)


def exclusive_outcome_class(row: pd.Series) -> str:
    if bool(row["truncated"]):
        return "truncation"
    if not bool(row["parse_success"]):
        return "parse_failure"
    if float(row["signed_deviation"]) < 0:
        return "undercount"
    if float(row["signed_deviation"]) > 0:
        return "overcount"
    if not bool(row["format_compliant"]):
        return "format_only_failure"
    return "strict_success"


def behavior_tables(
    requests: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = requests.copy()
    working["correct_and_format"] = working["exact_count"].astype(bool) & working[
        "format_compliant"
    ].astype(bool)
    group = ["comparison_slot", "model_label", "prompt_mode"]
    summary = (
        working.groupby(group, sort=True, dropna=False)
        .agg(
            n_total=("request_id", "size"),
            n_parseable=("parse_success", "sum"),
            n_correct_parsed=("exact_count", "sum"),
            n_format_compliant=("format_compliant", "sum"),
            n_correct_and_format_compliant=("correct_and_format", "sum"),
            n_truncated=("truncated", "sum"),
            strict_successes=("strict_registered_success", "sum"),
        )
        .reset_index()
    )
    summary["parse_rate"] = summary["n_parseable"] / summary["n_total"]
    summary["parsed_exact_accuracy"] = summary["n_correct_parsed"] / summary["n_total"]
    summary["conditional_numeric_accuracy"] = np.where(
        summary["n_parseable"] > 0,
        summary["n_correct_parsed"] / summary["n_parseable"],
        np.nan,
    )
    summary["format_compliance_rate"] = (
        summary["n_format_compliant"] / summary["n_total"]
    )
    summary["strict_accuracy"] = summary["strict_successes"] / summary["n_total"]
    summary["truncation_rate"] = summary["n_truncated"] / summary["n_total"]
    accuracy_cells = accuracy_condition_table(working)
    bias_cells = bias_condition_table(working)
    working["outcome_class"] = working.apply(exclusive_outcome_class, axis=1)
    outcomes = (
        working.groupby(["comparison_slot", "prompt_mode", "outcome_class"])
        .size()
        .rename("requests")
        .reset_index()
    )
    totals = outcomes.groupby(["comparison_slot", "prompt_mode"])["requests"].transform(
        "sum"
    )
    outcomes["proportion"] = outcomes["requests"] / totals
    return summary, accuracy_cells, bias_cells, outcomes
