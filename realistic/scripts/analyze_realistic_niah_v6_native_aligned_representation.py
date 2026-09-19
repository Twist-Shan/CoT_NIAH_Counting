#!/usr/bin/env python3
"""Run the frozen Native-thinking representation path across all V6 cells."""

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

from realistic_niah_v6.native_aligned_representation import (  # noqa: E402
    analyze_native_aligned_representation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--alignment-contract",
        type=Path,
        default=ROOT / "configs/realistic_niah_v6_native_analysis_alignment_v1.json",
    )
    args = parser.parse_args()
    output = (
        args.output.resolve()
        if args.output is not None
        else (args.run_root.resolve() / "native_aligned_representation")
    )
    paths = analyze_native_aligned_representation(
        run_root=args.run_root.resolve(),
        output_dir=output,
        contract_path=args.alignment_contract.resolve(),
        project_root=ROOT,
        command=" ".join(shlex.quote(value) for value in sys.argv),
    )
    print(
        json.dumps(
            {"status": "PASS", "outputs": {key: str(path) for key, path in paths.items()}},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
