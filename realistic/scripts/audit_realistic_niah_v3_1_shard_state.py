from __future__ import annotations

import argparse
import json

from realistic_niah_v3_1.shard_state import audit_shard_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V3.1 Slurm shard state.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bundle-id")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = audit_shard_state(args.run_root, args.bundle_id)
    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
