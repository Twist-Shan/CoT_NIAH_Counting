from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_counting_v44.behavior_gate import evaluate_behavior_gate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fixed v45 behavior gate")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_behavior_gate(args.run_dir, expected_version="v45")
    output = args.run_dir / "analysis" / "behavior_gate_v45.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2), flush=True)
    print(f"BEHAVIOR_GATE_OUTPUT={output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
