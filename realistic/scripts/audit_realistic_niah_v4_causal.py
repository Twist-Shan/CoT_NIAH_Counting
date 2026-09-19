from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v4.causal_audit import audit_screen_8h, write_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the completed Realistic NIAH V4 screen_8h_v1 causal campaign."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_screen_8h(args.run_root)
    if args.output is not None:
        write_audit(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
