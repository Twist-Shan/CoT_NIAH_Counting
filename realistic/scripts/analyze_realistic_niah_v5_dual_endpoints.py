#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.dual_endpoint_geometry import (  # noqa: E402
    analyze_dual_endpoint_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select token sites and decoder layers independently within non-thinking "
            "and native-thinking for running-index and final-count geometry."
        )
    )
    parser.add_argument("--non-thinking-running-index", type=Path, required=True)
    parser.add_argument("--native-thinking-running-index", type=Path, required=True)
    parser.add_argument("--non-thinking-final-count", type=Path, required=True)
    parser.add_argument("--native-thinking-final-count", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    command = " ".join(shlex.quote(argument) for argument in sys.argv)
    paths = analyze_dual_endpoint_geometry(
        non_thinking_running_index=args.non_thinking_running_index,
        native_thinking_running_index=args.native_thinking_running_index,
        non_thinking_final_count=args.non_thinking_final_count,
        native_thinking_final_count=args.native_thinking_final_count,
        output_dir=args.output,
        pca_dim=args.pca_dim,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        command=command,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
