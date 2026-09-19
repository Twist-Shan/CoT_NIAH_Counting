from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah.extension_runtime import audit_and_merge_extension
from realistic_niah.reasoning_models_extension import (
    PLAN_BASENAME,
    reasoning_extension_plan,
)


def audit_and_merge(run_root: Path, *, audit_only: bool) -> dict:
    return audit_and_merge_extension(
        run_root,
        plan=reasoning_extension_plan(),
        plan_basename=PLAN_BASENAME,
        schema_prefix="realistic_niah_reasoning_models_extension",
        audit_only=audit_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and losslessly merge all reasoning-model extension shards."
        )
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = audit_and_merge(
        Path(args.run_root),
        audit_only=args.audit_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
