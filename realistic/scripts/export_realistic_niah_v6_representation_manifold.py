#!/usr/bin/env python3
"""Export all-layer V6 Enumeration running/final PCA3 manifolds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:  # The remote V6 source package is deliberately read-only after sealing.
    from realistic_niah_v6.representation_manifold import (  # noqa: E402
        export_representation_manifold,
    )
except ModuleNotFoundError:  # pragma: no cover - remote sealed-package layout
    from realistic_niah_v6_representation_manifold import (  # type: ignore  # noqa: E402
        export_representation_manifold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    paths = export_representation_manifold(
        run_root=args.run_root,
        output_dir=args.output,
        protocol_path=args.protocol,
        command=" ".join(sys.argv),
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
