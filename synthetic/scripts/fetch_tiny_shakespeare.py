from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_counting_v11.data import TINY_SHAKESPEARE_URL, ensure_tiny_shakespeare_corpus


def main() -> None:
    path = ensure_tiny_shakespeare_corpus()
    print(f"Tiny Shakespeare ready: {path}")
    print(f"Source: {TINY_SHAKESPEARE_URL}")


if __name__ == "__main__":
    main()
