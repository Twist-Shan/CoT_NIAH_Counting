from __future__ import annotations

import argparse
import json

from realistic_niah.corpus import sync_full_paul_graham_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a content-deduplicated Paul Graham haystack corpus from "
            "the complete URL list registered by NVIDIA/RULER."
        )
    )
    parser.add_argument(
        "--url-list",
        default=(
            "data/haystacks/paul_graham/"
            "ruler_paulgraham_urls.txt"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-file-bytes", type=int, default=5 * 1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--minimum-included-files", type=int, default=100)
    parser.add_argument("--maximum-failures", type=int, default=50)
    args = parser.parse_args()

    manifest = sync_full_paul_graham_corpus(
        url_list_path=args.url_list,
        output_dir=args.output_dir,
        min_file_bytes=args.min_file_bytes,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        minimum_included_files=args.minimum_included_files,
        maximum_failures=args.maximum_failures,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
