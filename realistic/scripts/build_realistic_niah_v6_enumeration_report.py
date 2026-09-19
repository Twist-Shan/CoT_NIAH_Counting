#!/usr/bin/env python3
"""Build the self-contained 20-frame V6 index/bullet replication report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v6.reporting import (  # noqa: E402
    build_report_document,
    validate_report_html,
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document, summary = build_report_document(args.run_root, args.completion_audit)
    _atomic_text(args.output, document)
    summary_path = args.output.with_suffix(".summary.json")
    _atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    validation = validate_report_html(args.output)
    validation_path = args.output.with_suffix(".validation.json")
    _atomic_text(
        validation_path, json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    if validation["status"] != "PASS":
        raise SystemExit(f"V6 report validation failed: {validation['errors']}")
    args.output.with_suffix(".COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()

