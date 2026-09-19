from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v3.sharding import formal_shard_plan


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the immutable 48-shard Realistic NIAH V3 plan."
    )
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--tsv-output", required=True)
    args = parser.parse_args()

    plan = formal_shard_plan()
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
    _atomic_text(Path(args.json_output), json_payload)
    _atomic_text(Path(args.tsv_output), "\n".join(lines) + "\n")
    print(json_payload, end="")


if __name__ == "__main__":
    main()
