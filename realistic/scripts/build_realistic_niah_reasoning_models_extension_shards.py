from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah.extension_runtime import atomic_json, atomic_text
from realistic_niah.reasoning_models_extension import reasoning_extension_plan


def write_plan(json_output: Path, tsv_output: Path) -> dict:
    plan = reasoning_extension_plan()
    fields = (
        "task_id",
        "priority",
        "model_label",
        "prompt_mode",
        "output_collection",
        "expected_requests",
        "model_revision",
    )
    lines = ["\t".join(fields)]
    lines.extend(
        "\t".join(str(task[field]) for field in fields)
        for task in plan["tasks"]
    )
    atomic_json(json_output, plan)
    atomic_text(tsv_output, "\n".join(lines) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the immutable 20-shard reasoning-model plan."
    )
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--tsv-output", required=True)
    args = parser.parse_args()
    plan = write_plan(Path(args.json_output), Path(args.tsv_output))
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
