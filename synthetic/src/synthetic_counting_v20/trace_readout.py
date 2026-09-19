from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .training import atomic_csv


REQUIRED_COLUMNS = {
    "position_encoding",
    "mode",
    "step",
    "count",
    "ar_accuracy",
    "trace_exact",
    "trace_marker_count_accuracy",
    "trace_generated_marker_count",
    "trace_format_valid",
    "trace_closed",
}


def _wilson_interval(successes: float, observations: int) -> tuple[float, float]:
    z = 1.959963984540054
    estimate = successes / max(1, observations)
    denominator = 1.0 + z * z / observations
    center = (estimate + z * z / (2 * observations)) / denominator
    radius = z * math.sqrt(
        estimate * (1.0 - estimate) / observations
        + z * z / (4 * observations * observations)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def trace_consistent_predictions(
    frame: pd.DataFrame,
    *,
    count_max_threshold: int,
) -> pd.DataFrame:
    """Decode a count only from the model's own closed, well-formed trace.

    Prediction never reads the gold count.  The gold column is used only after
    prediction to score accuracy.  A malformed, unclosed, empty, or out-of-range
    trace is treated as abstention and therefore as an incorrect answer.
    """

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"trace readout requires columns: {missing}")
    output = frame[frame["mode"].eq("thinking")].copy()
    generated_count = pd.to_numeric(
        output["trace_generated_marker_count"], errors="coerce"
    )
    well_formed = (
        output["trace_format_valid"].eq(1.0)
        & output["trace_closed"].eq(1.0)
        & generated_count.between(1, int(count_max_threshold), inclusive="both")
    )
    output["trace_readout_pred_count"] = generated_count.where(well_formed)
    output["trace_readout_answered"] = well_formed.astype(float)
    output["trace_readout_accuracy"] = (
        output["trace_readout_pred_count"].eq(output["count"]) & well_formed
    ).astype(float)
    output["trace_readout_abs_error"] = (
        output["trace_readout_pred_count"] - output["count"]
    ).abs()
    output["trace_readout_abs_error_with_missing_penalty"] = output[
        "trace_readout_abs_error"
    ].fillna(int(count_max_threshold))
    return output


def write_trace_readout_analysis(
    run_dir: str | Path,
    *,
    count_max_threshold: int,
) -> dict[str, Any] | None:
    run_dir = Path(run_dir)
    source = run_dir / "tables" / "final_autoregressive_detail.csv"
    if not source.exists():
        return None
    detail = trace_consistent_predictions(
        pd.read_csv(source), count_max_threshold=count_max_threshold
    )
    if detail.empty:
        return None

    by_count = detail.groupby(
        ["position_encoding", "mode", "step", "count"], as_index=False
    ).agg(
        examples=("trace_readout_accuracy", "size"),
        trace_readout_accuracy=("trace_readout_accuracy", "mean"),
        trace_readout_answer_rate=("trace_readout_answered", "mean"),
        trace_readout_abs_error_answered_only=("trace_readout_abs_error", "mean"),
        raw_ar_accuracy=("ar_accuracy", "mean"),
        trace_exact=("trace_exact", "mean"),
        trace_marker_count_accuracy=("trace_marker_count_accuracy", "mean"),
    )

    summary_rows: list[dict[str, Any]] = []
    for (position_encoding, mode, step), group in detail.groupby(
        ["position_encoding", "mode", "step"]
    ):
        per_count = by_count[
            by_count["position_encoding"].eq(position_encoding)
            & by_count["mode"].eq(mode)
            & by_count["step"].eq(step)
        ]
        successes = float(group["trace_readout_accuracy"].sum())
        observations = int(len(group))
        low, high = _wilson_interval(successes, observations)
        overall = successes / max(1, observations)
        minimum = float(per_count["trace_readout_accuracy"].min())
        maximum = float(per_count["trace_readout_accuracy"].max())
        spread = maximum - minimum
        trace_exact = float(group["trace_exact"].mean())
        success = bool(
            overall >= 0.90
            and minimum >= 0.85
            and spread <= 0.10
            and trace_exact >= 0.90
        )
        summary_rows.append(
            {
                "position_encoding": position_encoding,
                "mode": mode,
                "step": int(step),
                "examples": observations,
                "examples_per_count": int(group.groupby("count").size().min()),
                "trace_readout_accuracy": overall,
                "trace_readout_accuracy_wilson95_low": low,
                "trace_readout_accuracy_wilson95_high": high,
                "trace_readout_answer_rate": float(
                    group["trace_readout_answered"].mean()
                ),
                "minimum_count_accuracy": minimum,
                "maximum_count_accuracy": maximum,
                "count_accuracy_spread": spread,
                "raw_ar_accuracy": float(group["ar_accuracy"].mean()),
                "trace_exact": trace_exact,
                "success_criteria_met": success,
            }
        )
    summary = pd.DataFrame(summary_rows)

    tables = run_dir / "tables"
    atomic_csv(detail, tables / "trace_readout_detail.csv")
    atomic_csv(by_count, tables / "trace_readout_by_count.csv")
    atomic_csv(summary, tables / "trace_readout_summary.csv")

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    plot = by_count.sort_values("count")
    axis.plot(
        plot["count"], plot["raw_ar_accuracy"], "o--", linewidth=1.8,
        label="Raw LM answer",
    )
    axis.plot(
        plot["count"], plot["trace_readout_accuracy"], "o-", linewidth=2.2,
        label="Trace-consistent readout",
    )
    axis.axhline(
        0.85,
        color="0.55",
        linestyle=":",
        linewidth=1.2,
        label="Per-count criterion",
    )
    axis.set(xlabel="True count", ylabel="Exact answer accuracy", ylim=(-0.03, 1.03))
    axis.set_xticks(sorted(plot["count"].unique()))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="lower left")
    figure.tight_layout()
    figure.savefig(figure_dir / "trace_readout_by_count.png", dpi=180)
    plt.close(figure)

    manifest = {
        "method": "closed_well_formed_trace_marker_pair_count",
        "prediction_uses_gold_count": False,
        "abstain_if_trace_unclosed_or_malformed": True,
        "count_support": [1, int(count_max_threshold)],
        "success_thresholds": {
            "overall_accuracy": 0.90,
            "minimum_count_accuracy": 0.85,
            "maximum_count_accuracy_spread": 0.10,
            "trace_exact": 0.90,
        },
        "summary": summary.to_dict(orient="records"),
    }
    manifest_path = run_dir / "analysis" / "trace_readout" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


__all__ = ["trace_consistent_predictions", "write_trace_readout_analysis"]
