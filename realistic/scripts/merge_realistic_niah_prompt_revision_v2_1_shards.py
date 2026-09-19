from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from realistic_niah.prompt_revision_v2_1 import (
    EXPECTED_PROMPT_REVISION_REQUESTS,
    PROTOCOL_VERSION,
    expected_request_ids,
    prompt_revision_shard_plan,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_id_digest(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _evaluation_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "parse_failures": sum(
            row["evaluation"]["parse_status"] == "parse_fail" for row in rows
        ),
        "format_failures": sum(
            not bool(row["evaluation"]["response_format_compliant"])
            for row in rows
        ),
        "truncations": sum(
            bool(row["evaluation"]["truncated"]) for row in rows
        ),
        "exact_count_correct": sum(
            bool(row["evaluation"]["exact_count"]) for row in rows
        ),
        "registered_successes": sum(
            bool(row["evaluation"]["registered_success"]) for row in rows
        ),
    }


def _canonical_output(
    run_root: Path,
    task: dict[str, Any],
) -> Path:
    role = str(task["analysis_role"])
    model_label = str(task["model_label"])
    if role == "replacement_enumeration":
        return (
            run_root
            / "replacement_enumeration"
            / str(task["output_collection"])
            / model_label
            / "main"
        )
    if role == "appendix_strict_direct":
        return (
            run_root
            / "appendix"
            / model_label
            / "direct_strict"
            / "main"
        )
    raise RuntimeError(f"Unregistered analysis role: {role}")


def audit_and_merge(run_root: Path, *, audit_only: bool) -> dict[str, Any]:
    plan = prompt_revision_shard_plan()
    orchestration = run_root / "orchestration"
    saved_plan_path = orchestration / "prompt_revision_shards.json"
    stimuli_path = run_root / "dataset" / "stimuli.jsonl"
    commit_path = orchestration / "git_commit.txt"
    prompts_sha_path = orchestration / "prompts_sha256.txt"

    saved_plan = _load_json(saved_plan_path)
    if saved_plan != plan:
        raise RuntimeError("Saved V2.1 plan differs from the registered plan")
    frozen_commit = commit_path.read_text(encoding="utf-8").strip()
    frozen_prompts_sha = prompts_sha_path.read_text(encoding="utf-8").strip()

    stimuli = _load_jsonl(stimuli_path)
    stimulus_ids = tuple(str(row["stimulus_id"]) for row in stimuli)
    if len(stimulus_ids) != 500 or len(set(stimulus_ids)) != 500:
        raise RuntimeError("V2.1 dataset must contain 500 unique stimulus IDs")

    all_rows: list[dict[str, Any]] = []
    output_rows: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    output_tasks: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    shard_audits: list[dict[str, Any]] = []
    global_ids: set[str] = set()
    source_files: list[dict[str, Any]] = []

    for task in plan["tasks"]:
        task_id = str(task["task_id"])
        shard_dir = run_root / "shards" / task_id / "main"
        requests_path = shard_dir / "requests.jsonl"
        manifest_path = shard_dir / "run_manifest.json"
        required = (requests_path, manifest_path)
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError(f"Missing/non-empty shard artifact: {task_id}")

        rows = _load_jsonl(requests_path)
        manifest = _load_json(manifest_path)
        expected = set(expected_request_ids(stimulus_ids, task))
        actual_ids = [str(row["request_id"]) for row in rows]
        actual = set(actual_ids)
        if (
            len(rows) != 500
            or len(actual) != 500
            or actual != expected
            or int(manifest["expected_requests"]) != 500
            or int(manifest["completed_requests"]) != 500
            or manifest["model"]["label"] != task["model_label"]
            or manifest["model_revision"] != task["model_revision"]
            or manifest["prompt_modes"] != [task["prompt_mode"]]
            or manifest["git"]["commit"] != frozen_commit
        ):
            raise RuntimeError(f"Structural shard audit failed: {task_id}")
        if any(
            str(row["model_label"]) != str(task["model_label"])
            or str(row["prompt_mode"]) != str(task["prompt_mode"])
            for row in rows
        ):
            raise RuntimeError(f"Row/task identity mismatch: {task_id}")
        overlap = global_ids.intersection(actual)
        if overlap:
            raise RuntimeError(
                f"Cross-shard duplicate IDs in {task_id}: "
                f"{sorted(overlap)[:3]}"
            )

        global_ids.update(actual)
        sorted_rows = sorted(rows, key=lambda row: str(row["request_id"]))
        all_rows.extend(sorted_rows)
        output = _canonical_output(run_root, task)
        output_rows[output].extend(sorted_rows)
        output_tasks[output].append(task)
        for source_path in required:
            source_files.append(
                {
                    "path": str(source_path.relative_to(run_root)),
                    "sha256": _sha256(source_path),
                    "bytes": source_path.stat().st_size,
                }
            )
        shard_audits.append(
            {
                "task_id": task_id,
                "model_label": task["model_label"],
                "prompt_mode": task["prompt_mode"],
                "analysis_role": task["analysis_role"],
                "requests": len(rows),
                "request_ids_sha256": _ordered_id_digest(sorted(actual)),
                "evaluation": _evaluation_counts(rows),
                "passed": True,
            }
        )

    if len(all_rows) != EXPECTED_PROMPT_REVISION_REQUESTS:
        raise RuntimeError("Merged V2.1 row count is not 7,500")
    if len(global_ids) != EXPECTED_PROMPT_REVISION_REQUESTS:
        raise RuntimeError("Merged V2.1 IDs are not 7,500 unique values")

    created_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema_version": "realistic_niah_prompt_revision_shard_audit_v2_1",
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": created_at,
        "passed": True,
        "audit_only": audit_only,
        "run_root": str(run_root),
        "git_commit": frozen_commit,
        "prompts_sha256": frozen_prompts_sha,
        "plan_tasks_sha256": plan["tasks_sha256"],
        "stimuli_sha256": _sha256(stimuli_path),
        "stimuli": len(stimuli),
        "shards": len(shard_audits),
        "requests": len(all_rows),
        "unique_request_ids": len(global_ids),
        "request_ids_sha256": _ordered_id_digest(sorted(global_ids)),
        "source_files": source_files,
        "shard_audits": shard_audits,
        "evaluation": _evaluation_counts(all_rows),
        "scope": {
            "replacement_enumeration_requests": 7_000,
            "appendix_strict_direct_requests": 500,
            "old_enumeration_included": False,
        },
    }

    if not audit_only:
        for output, rows in output_rows.items():
            rows = sorted(rows, key=lambda row: str(row["request_id"]))
            tasks = output_tasks[output]
            request_ids = [str(row["request_id"]) for row in rows]
            modes = sorted({str(row["prompt_mode"]) for row in rows})
            model_label = str(tasks[0]["model_label"])
            manifest = {
                "schema_version": (
                    "realistic_niah_prompt_revision_merged_manifest_v2_1"
                ),
                "protocol_version": PROTOCOL_VERSION,
                "created_at_utc": created_at,
                "analysis_role": tasks[0]["analysis_role"],
                "model_label": model_label,
                "model_id": tasks[0]["model_id"],
                "model_revision": tasks[0]["model_revision"],
                "git_commit": frozen_commit,
                "prompts_sha256": frozen_prompts_sha,
                "query_layout": plan["query_layout"],
                "prompt_modes": modes,
                "expected_requests": len(rows),
                "completed_requests": len(rows),
                "request_ids_sha256": _ordered_id_digest(request_ids),
                "stimuli_sha256": audit["stimuli_sha256"],
                "source_shards": [str(task["task_id"]) for task in tasks],
                "prompt_revision_plan_tasks_sha256": plan["tasks_sha256"],
                "old_enumeration_included": False,
            }
            qc = {
                "schema_version": (
                    "realistic_niah_prompt_revision_structural_qc_v2_1"
                ),
                "created_at_utc": created_at,
                "passed": True,
                "analysis_role": tasks[0]["analysis_role"],
                "model_label": model_label,
                "expected_requests": len(rows),
                "completed_requests": len(rows),
                "unique_request_ids": len(set(request_ids)),
                "all_jsonl_rows_parsed": True,
                "evaluation": _evaluation_counts(rows),
            }
            _atomic_jsonl(output / "requests.jsonl", rows)
            _atomic_json(output / "run_manifest.json", manifest)
            _atomic_json(output / "qc_report.json", qc)

    _atomic_json(orchestration / "final_shard_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and merge the V2.1 prompt-revision GPU shards."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = audit_and_merge(
        Path(args.run_root).resolve(),
        audit_only=args.audit_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
