from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v3_1.stimuli import audit_v31_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a frozen V3.1 grid.")
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_v31_grid(
        stimuli_path=args.stimuli,
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
