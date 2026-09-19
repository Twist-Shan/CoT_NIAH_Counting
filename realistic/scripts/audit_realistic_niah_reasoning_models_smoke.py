from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realistic_niah.extension_runtime import atomic_json, load_jsonl
from realistic_niah.spec import REASONING_EXTENSION_MODEL_SPECS


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
    audits: dict[str, Any] = {}
    all_ids: set[str] = set()
    stimuli_per_mode = 4
    for model_label, spec in REASONING_EXTENSION_MODEL_SPECS.items():
        path = (
            smoke_root / "models" / model_label / "smoke" / "requests.jsonl"
        )
        rows = load_jsonl(path)
        ids = [str(row["request_id"]) for row in rows]
        modes = {str(row["prompt_mode"]) for row in rows}
        expected_requests = stimuli_per_mode * len(spec.prompt_modes)
        if (
            len(rows) != expected_requests
            or len(ids) != len(set(ids))
            or modes != set(spec.prompt_modes)
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
        "schema_version": (
            "realistic_niah_reasoning_models_smoke_audit_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "requests": len(all_ids),
        "unique_request_ids": len(all_ids),
        "models": audits,
    }
    if len(all_ids) != 80:
        raise RuntimeError("Reasoning-model smoke must contain 80 requests")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(Path(args.smoke_root).resolve())
    atomic_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
