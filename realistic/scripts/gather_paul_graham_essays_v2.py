from __future__ import annotations

import argparse
import json
from pathlib import Path

from sync_paul_graham_essays import sync


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather Paul Graham essay text files for dynamic NIAH v2 generation."
    )
    parser.add_argument("--out-dir", default="data/haystacks/paul_graham")
    parser.add_argument(
        "--min-file-bytes",
        type=int,
        default=5 * 1024,
        help="Minimum UTF-8 byte size for kept text files (default: 5KB).",
    )
    args = parser.parse_args()

    result = sync(Path(args.out_dir), min_file_bytes=args.min_file_bytes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
