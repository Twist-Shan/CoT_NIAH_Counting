#!/usr/bin/env python3
"""Find complete unnumbered bullet episodes with no within-item progress marker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.bullet_counterfactual_restore import (  # noqa: E402
    audit_complete_marker_scrubbable_list,
)
from scripts.run_realistic_niah_v5_count_stream import _atomic_jsonl  # noqa: E402


def _iter_valid_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield object rows while making any damaged JSONL records auditable."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                print(
                    json.dumps(
                        {
                            "event": "invalid_jsonl_row_skipped",
                            "path": str(path),
                            "line_number": line_number,
                            "error": str(error),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if not isinstance(value, dict):
                print(
                    json.dumps(
                        {
                            "event": "non_object_jsonl_row_skipped",
                            "path": str(path),
                            "line_number": line_number,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            yield value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--min-seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_request: dict[str, dict[str, object]] = {}
    seen_requests: set[str] = set()
    status_counts: dict[str, int] = {}
    for path in args.inputs:
        for row in _iter_valid_jsonl(path):
            if str(row.get("model_label")) != str(args.model):
                continue
            if int(row.get("seed", -1)) < int(args.min_seed):
                continue
            request_id = str(row.get("request_id", ""))
            if not request_id or request_id in seen_requests:
                continue
            seen_requests.add(request_id)
            audit = audit_complete_marker_scrubbable_list(row)
            if not audit["eligible"]:
                status = "broad_gate_fail"
            elif str(audit["marker_kind"]) != "bullet":
                status = "indexed_not_bullet"
            elif any(audit["item_marker_char_spans"]):
                status = "bullet_contains_explicit_progress_marker"
            else:
                status = "strict_unnumbered_bullet_pass"
                by_request[request_id] = {
                    **row,
                    "strict_unnumbered_bullet_audit": audit,
                    "strict_gate_uses_final_answer": False,
                }
            status_counts[status] = status_counts.get(status, 0) + 1
    selected = sorted(
        by_request.values(), key=lambda value: (int(value["seed"]), str(value["request_id"]))
    )
    _atomic_jsonl(args.output, selected)
    print(
        json.dumps(
            {
                "model_label": str(args.model),
                "strict_pass_count": len(selected),
                "seeds": [int(value["seed"]) for value in selected],
                "status_counts": status_counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
