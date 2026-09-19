from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from realistic_niah.olmo3_extension import (
    EXPECTED_EXTENSION_REQUESTS,
    EXPECTED_STIMULI_PER_SHARD,
    OLMO3_LOGICAL_MODEL_LABEL,
    SOURCE_FORMAL_STIMULI_SHA256,
    expected_request_ids,
    olmo3_extension_plan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                raise ValueError(f"Expected object at {path}:{line_number}")
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _evaluation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
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


def audit_and_merge(run_root: Path, *, audit_only: bool) -> dict[str, Any]:
    run_root = run_root.resolve()
    plan = olmo3_extension_plan()
    saved_plan_path = (
        run_root / "orchestration" / "olmo3_extension_shards.json"
    )
    stimuli_path = run_root / "dataset" / "stimuli.jsonl"
    provenance_path = (
        run_root / "orchestration" / "source_formal_run_provenance.json"
    )
    saved_plan = _load_json(saved_plan_path)
    provenance = _load_json(provenance_path)
    if saved_plan != plan:
        raise RuntimeError("Saved OLMo extension plan differs from registration")
    if _sha256(stimuli_path) != SOURCE_FORMAL_STIMULI_SHA256:
        raise RuntimeError("OLMo extension stimuli SHA256 is not the frozen V2 SHA")
    if provenance.get("stimuli_sha256") != SOURCE_FORMAL_STIMULI_SHA256:
        raise RuntimeError("Source provenance has a mismatched stimuli SHA256")

    stimuli = _load_jsonl(stimuli_path)
    stimulus_ids = tuple(str(row["stimulus_id"]) for row in stimuli)
    if (
        len(stimulus_ids) != EXPECTED_STIMULI_PER_SHARD
        or len(set(stimulus_ids)) != EXPECTED_STIMULI_PER_SHARD
    ):
        raise RuntimeError("OLMo extension dataset must have 500 unique stimuli")

    all_rows: list[dict[str, Any]] = []
    model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shard_audits: list[dict[str, Any]] = []
    global_ids: set[str] = set()
    source_files: list[dict[str, Any]] = []
    expected_git_commit = str(provenance["git_commit"])

    for task in plan["tasks"]:
        task_id = str(task["task_id"])
        shard_dir = run_root / "shards" / task_id / "main"
        requests_path = shard_dir / "requests.jsonl"
        manifest_path = shard_dir / "run_manifest.json"
        if not requests_path.is_file() or requests_path.stat().st_size == 0:
            raise RuntimeError(f"Missing/non-empty shard requests: {task_id}")
        if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
            raise RuntimeError(f"Missing/non-empty shard manifest: {task_id}")

        rows = _load_jsonl(requests_path)
        manifest = _load_json(manifest_path)
        expected = set(expected_request_ids(stimulus_ids, task))
        actual_ids = [str(row["request_id"]) for row in rows]
        actual = set(actual_ids)
        row_revisions = {str(row.get("model_revision")) for row in rows}
        row_models = {str(row.get("model_label")) for row in rows}
        row_modes = {str(row.get("prompt_mode")) for row in rows}
        if (
            len(rows) != EXPECTED_STIMULI_PER_SHARD
            or len(actual) != EXPECTED_STIMULI_PER_SHARD
            or actual != expected
            or int(manifest["expected_requests"])
            != EXPECTED_STIMULI_PER_SHARD
            or int(manifest["completed_requests"])
            != EXPECTED_STIMULI_PER_SHARD
            or manifest["model"]["label"] != task["model_label"]
            or manifest["model"]["model_id"] != task["model_id"]
            or manifest["model_revision"] != task["model_revision"]
            or manifest["prompt_modes"] != [task["prompt_mode"]]
            or manifest["stimuli_sha256"] != SOURCE_FORMAL_STIMULI_SHA256
            or manifest["git"]["commit"] != expected_git_commit
            or row_revisions != {str(task["model_revision"])}
            or row_models != {str(task["model_label"])}
            or row_modes != {str(task["prompt_mode"])}
        ):
            raise RuntimeError(f"Structural shard audit failed: {task_id}")
        overlap = global_ids.intersection(actual)
        if overlap:
            raise RuntimeError(
                f"Cross-shard duplicate request IDs in {task_id}: "
                f"{sorted(overlap)[:3]}"
            )
        global_ids.update(actual)
        sorted_rows = sorted(rows, key=lambda row: str(row["request_id"]))
        all_rows.extend(sorted_rows)
        model_rows[str(task["model_label"])].extend(sorted_rows)
        source_files.extend(
            (
                {
                    "path": requests_path.relative_to(run_root).as_posix(),
                    "sha256": _sha256(requests_path),
                    "bytes": requests_path.stat().st_size,
                },
                {
                    "path": manifest_path.relative_to(run_root).as_posix(),
                    "sha256": _sha256(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                },
            )
        )
        shard_audits.append(
            {
                "task_id": task_id,
                "model_label": task["model_label"],
                "prompt_mode": task["prompt_mode"],
                "requests": len(rows),
                "request_ids_sha256": _ordered_id_digest(sorted(actual)),
                "evaluation": _evaluation_counts(rows),
                "passed": True,
            }
        )

    if (
        len(all_rows) != EXPECTED_EXTENSION_REQUESTS
        or len(global_ids) != EXPECTED_EXTENSION_REQUESTS
    ):
        raise RuntimeError("Merged OLMo extension must have 2,000 unique requests")

    created_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema_version": "realistic_niah_olmo3_extension_audit_v1",
        "created_at_utc": created_at,
        "passed": True,
        "audit_only": audit_only,
        "run_root": str(run_root),
        "logical_model_label": OLMO3_LOGICAL_MODEL_LABEL,
        "plan_tasks_sha256": plan["tasks_sha256"],
        "stimuli_sha256": _sha256(stimuli_path),
        "stimuli": len(stimuli),
        "shards": len(shard_audits),
        "requests": len(all_rows),
        "unique_request_ids": len(global_ids),
        "request_ids_sha256": _ordered_id_digest(sorted(global_ids)),
        "git_commit": expected_git_commit,
        "source_files": source_files,
        "shard_audits": shard_audits,
        "evaluation": _evaluation_counts(all_rows),
    }

    if not audit_only:
        variant_outputs: list[dict[str, Any]] = []
        for model_label, rows in sorted(model_rows.items()):
            output = run_root / "models" / model_label / "main"
            rows = sorted(rows, key=lambda row: str(row["request_id"]))
            modes = sorted({str(row["prompt_mode"]) for row in rows})
            request_ids = [str(row["request_id"]) for row in rows]
            model_tasks = [
                task
                for task in plan["tasks"]
                if task["model_label"] == model_label
            ]
            model_manifest = {
                "schema_version": "realistic_niah_merged_run_manifest_v2",
                "protocol_version": plan["protocol_version"],
                "created_at_utc": created_at,
                "logical_model_label": OLMO3_LOGICAL_MODEL_LABEL,
                "model_label": model_label,
                "model_id": model_tasks[0]["model_id"],
                "model_revision": model_tasks[0]["model_revision"],
                "query_layout": plan["query_layout"],
                "prompt_modes": modes,
                "expected_requests": len(rows),
                "completed_requests": len(rows),
                "request_ids_sha256": _ordered_id_digest(request_ids),
                "stimuli_sha256": audit["stimuli_sha256"],
                "source_shards": [str(task["task_id"]) for task in model_tasks],
                "extension_plan_tasks_sha256": plan["tasks_sha256"],
                "git_commit": expected_git_commit,
            }
            model_qc = {
                "schema_version": "realistic_niah_structural_qc_v2",
                "created_at_utc": created_at,
                "passed": True,
                "logical_model_label": OLMO3_LOGICAL_MODEL_LABEL,
                "model_label": model_label,
                "expected_requests": len(rows),
                "completed_requests": len(rows),
                "unique_request_ids": len(set(request_ids)),
                "all_jsonl_rows_parsed": True,
                "evaluation": _evaluation_counts(rows),
            }
            _atomic_jsonl(output / "requests.jsonl", rows)
            _atomic_json(output / "run_manifest.json", model_manifest)
            _atomic_json(output / "qc_report.json", model_qc)
            variant_outputs.append(
                {
                    "model_label": model_label,
                    "model_id": model_tasks[0]["model_id"],
                    "model_revision": model_tasks[0]["model_revision"],
                    "prompt_modes": modes,
                    "requests": len(rows),
                    "path": output.relative_to(run_root).as_posix(),
                }
            )

        family_manifest = {
            "schema_version": "realistic_niah_model_family_manifest_v1",
            "created_at_utc": created_at,
            "logical_model_label": OLMO3_LOGICAL_MODEL_LABEL,
            "note": (
                "Instruct and Think are distinct immutable checkpoints. "
                "They are grouped only for analysis; request rows are not duplicated."
            ),
            "expected_requests": EXPECTED_EXTENSION_REQUESTS,
            "completed_requests": len(all_rows),
            "stimuli_sha256": audit["stimuli_sha256"],
            "extension_plan_tasks_sha256": plan["tasks_sha256"],
            "variants": variant_outputs,
        }
        _atomic_json(
            run_root
            / "models"
            / OLMO3_LOGICAL_MODEL_LABEL
            / "family_manifest.json",
            family_manifest,
        )

    _atomic_json(
        run_root / "orchestration" / "final_shard_audit.json",
        audit,
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and losslessly merge the OLMo 3 extension shards."
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
