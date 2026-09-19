from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from realistic_niah_v3_1.stimuli import (
    audit_v31_grid,
    default_freeze_spec,
    freeze_v31_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze and audit the registered 3,360-stimulus V3.1 grid."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canonical-tokenizer-revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--haystack-dir", required=True)
    parser.add_argument("--haystack-corpus-manifest", required=True)
    parser.add_argument("--haystack-corpus-manifest-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    spec = replace(
        default_freeze_spec(),
        haystack_dir=str(Path(args.haystack_dir).resolve()),
        haystack_corpus_manifest=str(Path(args.haystack_corpus_manifest).resolve()),
        haystack_corpus_manifest_sha256=args.haystack_corpus_manifest_sha256,
    )
    paths = freeze_v31_grid(
        output_dir=output,
        spec=spec,
        canonical_tokenizer_revision=args.canonical_tokenizer_revision,
        tokenizer_cache_dir=args.cache_dir,
        overwrite=args.overwrite,
    )
    audit = audit_v31_grid(
        stimuli_path=paths["stimuli"],
        manifest_path=paths["manifest"],
        cache_dir=args.cache_dir,
    )
    audit_path = output / "audit_report.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not audit["passed"]:
        raise RuntimeError(f"V3.1 frozen-grid audit failed: {audit['errors']}")
    print(
        json.dumps(
            {
                **{key: str(value) for key, value in paths.items()},
                "audit": str(audit_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
