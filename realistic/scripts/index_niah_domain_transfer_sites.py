from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from realistic_niah_v5.domain_transfer import (  # noqa: E402
    write_capture_site_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a flat answer/running-site catalog for one or more "
            "entity-domain-transfer capture directories."
        )
    )
    parser.add_argument(
        "capture_dirs",
        nargs="+",
        type=Path,
        help="Directories containing capture_index.jsonl and shard manifests.",
    )
    args = parser.parse_args()
    for capture_dir in args.capture_dirs:
        site_index, manifest = write_capture_site_index(capture_dir)
        print(f"{capture_dir}: {site_index.name}, {manifest.name}")


if __name__ == "__main__":
    main()
