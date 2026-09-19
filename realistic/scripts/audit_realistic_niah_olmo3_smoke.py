from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "requests": len(rows),
        "parse_failures": sum(
            row["evaluation"]["parse_status"] == "parse_fail" for row in rows
        ),
        "format_failures": sum(
            not bool(row["evaluation"]["response_format_compliant"])
            for row in rows
        ),
        "truncations": sum(bool(row["evaluation"]["truncated"]) for row in rows),
        "exact_count_correct": sum(
            bool(row["evaluation"]["exact_count"]) for row in rows
        ),
        "registered_successes": sum(
            bool(row["evaluation"]["registered_success"]) for row in rows
        ),
    }


def audit(smoke_root: Path) -> dict[str, Any]:
    paths = {
        "Olmo3-7B-Instruct": (
            smoke_root / "models" / "Olmo3-7B-Instruct" / "smoke"
            / "requests.jsonl"
        ),
        "Olmo3-7B-Think": (
            smoke_root / "models" / "Olmo3-7B-Think" / "smoke"
            / "requests.jsonl"
        ),
    }
    expected = {
        "Olmo3-7B-Instruct": {
            "requests": 12,
            "modes": {
                "direct",
                "enumeration_index",
                "enumeration_bullet",
            },
        },
        "Olmo3-7B-Think": {
            "requests": 4,
            "modes": {"native_thinking"},
        },
    }
    audits: dict[str, Any] = {}
    all_ids: set[str] = set()
    for model_label, path in paths.items():
        rows = _load_jsonl(path)
        ids = [str(row["request_id"]) for row in rows]
        modes = {str(row["prompt_mode"]) for row in rows}
        if (
            len(rows) != expected[model_label]["requests"]
            or len(ids) != len(set(ids))
            or modes != expected[model_label]["modes"]
            or {str(row["model_label"]) for row in rows} != {model_label}
            or all_ids.intersection(ids)
        ):
            raise RuntimeError(f"Smoke structural audit failed: {model_label}")
        all_ids.update(ids)
        audits[model_label] = {
            "passed": True,
            "prompt_modes": sorted(modes),
            **_summary(rows),
        }
    result = {
        "schema_version": "realistic_niah_olmo3_smoke_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "requests": len(all_ids),
        "unique_request_ids": len(all_ids),
        "models": audits,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(Path(args.smoke_root).resolve())
    payload = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    print(payload, end="")


if __name__ == "__main__":
    main()
