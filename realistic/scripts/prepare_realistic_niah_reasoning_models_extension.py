from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah.extension_runtime import prepare_extension
from realistic_niah.reasoning_models_extension import (
    PLAN_BASENAME,
    RUN_NAME_PREFIXES,
    reasoning_extension_plan,
)


def prepare(
    source_formal_run_root: Path,
    run_root: Path,
    repo_root: Path,
) -> dict:
    return prepare_extension(
        source_formal_run_root,
        run_root,
        repo_root,
        plan=reasoning_extension_plan(),
        plan_basename=PLAN_BASENAME,
        run_name_prefixes=RUN_NAME_PREFIXES,
        schema_prefix="realistic_niah_reasoning_models_extension",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the five-group reasoning-model extension from the "
            "audited V2 formal stimuli."
        )
    )
    parser.add_argument("--source-formal-run-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = prepare(
        Path(args.source_formal_run_root),
        Path(args.run_root),
        Path(args.repo_root),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
