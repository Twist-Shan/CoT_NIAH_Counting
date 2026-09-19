from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah.olmo3_extension import olmo3_extension_plan


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_plan(json_output: Path, tsv_output: Path) -> dict:
    plan = olmo3_extension_plan()
    json_payload = json.dumps(
        plan,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
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
    for task in plan["tasks"]:
        lines.append("\t".join(str(task[field]) for field in fields))
    tsv_payload = "\n".join(lines) + "\n"

    _atomic_text(json_output, json_payload)
    _atomic_text(tsv_output, tsv_payload)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the immutable four-shard OLMo 3 V2 extension plan."
    )
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--tsv-output", required=True)
    args = parser.parse_args()

    plan = write_plan(Path(args.json_output), Path(args.tsv_output))
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
