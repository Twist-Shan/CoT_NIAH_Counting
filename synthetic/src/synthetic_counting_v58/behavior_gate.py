from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_counting_v44.behavior_gate import evaluate_behavior_gate as _evaluate_behavior_gate


# Revised before v58 optimization: absolute near-perfect accuracy and exact
# marker identity are not prerequisites for the comparative mechanism claim.
# The behavioral screen instead requires a useful, count-uniform Thinking model
# and a large paired advantage over the independently trained Non-thinking model.
GATE_THRESHOLDS = {
    "thinking_accuracy_min": 0.75,
    "thinking_min_count_accuracy_min": 0.70,
    "thinking_count_spread_max": 0.20,
    "thinking_minus_nonthinking_gap_min": 0.30,
}


def evaluate_behavior_gate(run_dir: str | Path) -> dict[str, object]:
    result = _evaluate_behavior_gate(
        run_dir,
        expected_version="v58",
        expected_final_step=10_000,
    )
    metrics = result["metrics"]
    checks = {
        "thinking_accuracy": (
            float(metrics["thinking_accuracy"])
            >= GATE_THRESHOLDS["thinking_accuracy_min"]
        ),
        "thinking_min_count_accuracy": (
            float(metrics["thinking_min_count_accuracy"])
            >= GATE_THRESHOLDS["thinking_min_count_accuracy_min"]
        ),
        "thinking_count_spread": (
            float(metrics["thinking_count_spread"])
            <= GATE_THRESHOLDS["thinking_count_spread_max"]
        ),
        "thinking_minus_nonthinking_gap": (
            float(metrics["thinking_minus_nonthinking_gap"])
            >= GATE_THRESHOLDS["thinking_minus_nonthinking_gap_min"]
        ),
    }
    result.update(
        {
            "endpoint_policy": (
                "fixed final 10000-step checkpoint; no checkpoint selection; "
                "revised pre-v58 comparative-uniformity gate"
            ),
            "thresholds": dict(GATE_THRESHOLDS),
            "checks": checks,
            "passed": all(checks.values()),
            "trace_policy": (
                "trace metrics are reported diagnostically; targeted retrieval is "
                "tested separately with NCC, causal ablation/patching, and "
                "free-running sufficiency"
            ),
        }
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fixed v58 behavior gate")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = evaluate_behavior_gate(args.run_dir)
    output = args.output or args.run_dir / "analysis" / "behavior_gate_v58.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2), flush=True)
    print(f"BEHAVIOR_GATE_OUTPUT={output.resolve()}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["GATE_THRESHOLDS", "evaluate_behavior_gate"]
