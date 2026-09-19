#!/usr/bin/env python3
"""Fail-closed completion audit for the full four-cell V6 enumeration suite."""

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

from realistic_niah_v6.completion import audit_full_suite  # noqa: E402


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_full_suite(args.run_root)
    output = args.output
    if output.suffix.lower() != ".json":
        output = output / "suite_completion_audit.json"
    _atomic_json(output, audit)
    marker = output.with_name("suite_completion.COMPLETE")
    marker.write_text("PASS\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "audit": str(output.resolve()),
                "audit_sha256": audit["audit_sha256"],
                "ordinary_failure_count": audit["replacement_policy"][
                    "ordinary_cell_failure_count"
                ],
                "coherent_replacement_trajectory_count": audit[
                    "replacement_policy"
                ]["coherent_replacement_trajectory_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
