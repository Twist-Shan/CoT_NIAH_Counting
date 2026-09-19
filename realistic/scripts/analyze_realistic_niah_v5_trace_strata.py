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

from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    analyze_trace_stratified_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep native-thinking token sites within parser marker_kind strata. "
            "Site/layer selection uses discovery grouped CV only."
        )
    )
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    paths = analyze_trace_stratified_geometry(
        args.capture_index,
        args.output,
        pca_dim=args.pca_dim,
        layers=args.layers,
        random_state=args.random_state,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
