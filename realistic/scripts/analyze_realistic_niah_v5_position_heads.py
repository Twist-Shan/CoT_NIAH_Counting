#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.position_head_analysis import analyze_position_heads


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare V5 targeted retrieval at k=1 versus later positions and "
            "against the frozen V4.4 first-locator head bank"
        )
    )
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--first-locator-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bank-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    paths = analyze_position_heads(
        args.attention,
        args.first_locator_registry,
        args.output,
        bank_sizes=args.bank_sizes,
        bootstrap_samples=args.bootstrap_samples,
        random_state=args.random_state,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
