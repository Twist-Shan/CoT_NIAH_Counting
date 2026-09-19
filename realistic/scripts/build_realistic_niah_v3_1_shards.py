from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v3_1.sharding import formal_shard_plan


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the immutable V3.1 shard plan.")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--tsv-output", required=True)
    args = parser.parse_args()
    plan = formal_shard_plan()
    payload = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
        "\t".join(str(task[field]) for field in fields) for task in plan["tasks"]
    )
    _atomic_text(Path(args.json_output), payload)
    _atomic_text(Path(args.tsv_output), "\n".join(lines) + "\n")
    print(payload, end="")


if __name__ == "__main__":
    main()
