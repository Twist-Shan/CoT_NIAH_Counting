from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


GATE_THRESHOLDS = {
    "thinking_accuracy_min": 0.90,
    "thinking_min_count_accuracy_min": 0.80,
    "thinking_count_spread_max": 0.20,
    "thinking_trace_exact_min": 0.90,
    "thinking_minus_nonthinking_gap_min": 0.10,
}


def _single_mode_row(frame: pd.DataFrame, mode: str) -> pd.Series:
    rows = frame[frame["mode"].eq(mode)]
    if len(rows) != 1:
        raise ValueError(f"expected one {mode!r} summary row, found {len(rows)}")
    return rows.iloc[0]


def evaluate_behavior_gate(
    run_dir: str | Path,
    *,
    expected_version: str = "v44",
    expected_final_step: int = 8_000,
) -> dict[str, Any]:
    """Evaluate the fixed behavior gate without checkpoint selection."""

    root = Path(run_dir)
    summary = pd.read_csv(root / "tables" / "final_autoregressive_summary.csv")
    by_count = pd.read_csv(root / "tables" / "final_autoregressive_by_count.csv")
    thinking = _single_mode_row(summary, "thinking")
    nonthinking = _single_mode_row(summary, "nonthinking")

    expected_counts = list(range(1, 11))
    mode_count_rows: dict[str, pd.DataFrame] = {}
    for mode in ("nonthinking", "thinking"):
        rows = by_count[by_count["mode"].eq(mode)].sort_values("count")
        observed = rows["count"].astype(int).tolist()
        if observed != expected_counts:
            raise ValueError(
                f"{mode} final table must contain counts 1..10 exactly; got {observed}"
            )
        mode_count_rows[mode] = rows

    thinking_counts = mode_count_rows["thinking"]
    thinking_accuracy = float(thinking["ar_final_accuracy"])
    nonthinking_accuracy = float(nonthinking["ar_final_accuracy"])
    thinking_per_count = {
        str(int(row["count"])): float(row["ar_final_accuracy"])
        for _, row in thinking_counts.iterrows()
    }
    nonthinking_per_count = {
        str(int(row["count"])): float(row["ar_final_accuracy"])
        for _, row in mode_count_rows["nonthinking"].iterrows()
    }
    metrics = {
        "thinking_accuracy": thinking_accuracy,
        "nonthinking_accuracy": nonthinking_accuracy,
        "thinking_minus_nonthinking_gap": thinking_accuracy - nonthinking_accuracy,
        "thinking_min_count_accuracy": min(thinking_per_count.values()),
        "thinking_count_spread": (
            max(thinking_per_count.values()) - min(thinking_per_count.values())
        ),
        "thinking_trace_exact": float(thinking["trace_exact"]),
        "thinking_per_count_accuracy": thinking_per_count,
        "nonthinking_per_count_accuracy": nonthinking_per_count,
    }
    checks = {
        "thinking_accuracy": (
            metrics["thinking_accuracy"]
            >= GATE_THRESHOLDS["thinking_accuracy_min"]
        ),
        "thinking_min_count_accuracy": (
            metrics["thinking_min_count_accuracy"]
            >= GATE_THRESHOLDS["thinking_min_count_accuracy_min"]
        ),
        "thinking_count_spread": (
            metrics["thinking_count_spread"]
            <= GATE_THRESHOLDS["thinking_count_spread_max"]
        ),
        "thinking_trace_exact": (
            metrics["thinking_trace_exact"]
            >= GATE_THRESHOLDS["thinking_trace_exact_min"]
        ),
        "thinking_minus_nonthinking_gap": (
            metrics["thinking_minus_nonthinking_gap"]
            >= GATE_THRESHOLDS["thinking_minus_nonthinking_gap_min"]
        ),
    }
    return {
        "version": expected_version,
        "endpoint_policy": (
            f"fixed final {expected_final_step}-step checkpoint; no checkpoint selection"
        ),
        "thresholds": GATE_THRESHOLDS,
        "metrics": metrics,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fixed v44 behavior gate")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = evaluate_behavior_gate(args.run_dir)
    output = args.output or args.run_dir / "analysis" / "behavior_gate_v44.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2), flush=True)
    print(f"BEHAVIOR_GATE_OUTPUT={output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
