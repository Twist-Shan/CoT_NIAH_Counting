from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v3_1.merge import audit_and_merge


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and merge all V3.1 shards.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = audit_and_merge(Path(args.run_root), audit_only=args.audit_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
