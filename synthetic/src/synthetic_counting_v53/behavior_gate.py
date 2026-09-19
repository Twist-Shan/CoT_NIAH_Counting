from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_counting_v44.behavior_gate import (
    GATE_THRESHOLDS,
    evaluate_behavior_gate as _evaluate_behavior_gate,
)


def evaluate_behavior_gate(run_dir: str | Path) -> dict[str, object]:
    return _evaluate_behavior_gate(
        run_dir,
        expected_version="v53",
        expected_final_step=10_000,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fixed v53 behavior gate")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = evaluate_behavior_gate(args.run_dir)
    output = args.output or args.run_dir / "analysis" / "behavior_gate_v53.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2), flush=True)
    print(f"BEHAVIOR_GATE_OUTPUT={output.resolve()}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["GATE_THRESHOLDS", "evaluate_behavior_gate"]
