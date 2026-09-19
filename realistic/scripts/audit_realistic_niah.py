from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah.stimuli import audit_frozen_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently re-audit a frozen Realistic NIAH grid."
    )
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = audit_frozen_grid(
        stimuli_path=args.stimuli,
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
    )
    payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
