from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .sharding import expected_request_ids, formal_shard_plan
from .spec import (
    CANONICAL_TOKENIZER,
    CANONICAL_TOKENIZER_REVISION,
    EXPECTED_REQUESTS,
    EXPECTED_SHARDS,
    EXPECTED_STIMULI,
    INSERTION_DEPTH_MAX_FRACTION,
    INSERTION_DEPTH_MIN_FRACTION,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    PROTOCOL_VERSION,
    SEEDS,
    V3_FREEZE_PROTOCOL,
    V3_RUN_PROTOCOL,
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
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
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
    total = len(rows)
    exact = sum(bool(row["evaluation"]["exact_count"]) for row in rows)
    parsed = sum(
        row["evaluation"]["parse_status"] != "parse_fail" for row in rows
    )
    strict = sum(
        bool(row["evaluation"]["registered_success"]) for row in rows
    )
    return {
        "requests": total,
        "parse_successes": parsed,
        "parse_failures": total - parsed,
        "format_failures": sum(
            not bool(row["evaluation"]["response_format_compliant"])
            for row in rows
        ),
        "truncations": sum(
            bool(row["evaluation"]["truncated"]) for row in rows
        ),
        "exact_count_correct": exact,
        "registered_successes": strict,
        "parse_rate": parsed / total if total else None,
        "parseable_exact_accuracy": exact / total if total else None,
        "strict_registered_accuracy": strict / total if total else None,
    }


def _audit_request_depths(row: dict[str, Any]) -> None:
    policy = row.get("insertion_depth_policy")
    if not isinstance(policy, dict):
        raise ValueError("request row lacks insertion_depth_policy")
    if (
        float(policy["minimum_inclusive"])
        != INSERTION_DEPTH_MIN_FRACTION
        or float(policy["maximum_inclusive"])
        != INSERTION_DEPTH_MAX_FRACTION
    ):
        raise ValueError("request row has the wrong insertion-depth policy")
    needles = row.get("needles")
    if not isinstance(needles, list) or len(needles) != int(row["num_needles"]):
        raise ValueError("request row needle metadata is incomplete")
    for needle in needles:
        depth = float(needle["normalized_depth"])
        if not (
            INSERTION_DEPTH_MIN_FRACTION
            <= depth
            <= INSERTION_DEPTH_MAX_FRACTION
        ):
            raise ValueError(f"request row contains out-of-range depth: {depth}")


def audit_and_merge(run_root: Path, *, audit_only: bool = False) -> dict[str, Any]:
    run_root = run_root.resolve()
    plan = formal_shard_plan()
    orchestration = run_root / "orchestration"
    dataset = run_root / "dataset"
    saved_plan = _load_json(orchestration / "formal_shards.json")
    if saved_plan != plan:
        raise RuntimeError("Saved V3 shard plan differs from the registry")

    dataset_manifest = _load_json(dataset / "manifest.json")
    dataset_audit = _load_json(dataset / "audit_report.json")
    dataset_spec = dataset_manifest.get("spec", {})
    if (
        dataset_manifest.get("schema_version")
        != V3_FREEZE_PROTOCOL.manifest_schema_version
        or dataset_manifest.get("protocol_version") != PROTOCOL_VERSION
        or int(dataset_manifest.get("rows", -1)) != EXPECTED_STIMULI
        or dataset_spec.get("canonical_tokenizer") != CANONICAL_TOKENIZER
        or dataset_spec.get("canonical_tokenizer_revision")
        != CANONICAL_TOKENIZER_REVISION
        or tuple(dataset_spec.get("passage_lengths", ())) != PASSAGE_LENGTHS
        or tuple(dataset_spec.get("needle_counts", ())) != NEEDLE_COUNTS
        or tuple(dataset_spec.get("seeds", ())) != SEEDS
        or dataset_audit.get("passed") is not True
        or dataset_audit.get("protocol_version") != PROTOCOL_VERSION
        or int(dataset_audit.get("rows_checked", -1)) != EXPECTED_STIMULI
    ):
        raise RuntimeError("Frozen V3 dataset provenance/audit is invalid")

    stimuli_path = dataset / "stimuli.jsonl"
    stimuli = _load_jsonl(stimuli_path)
    stimulus_ids = tuple(str(row["stimulus_id"]) for row in stimuli)
    registered_stimulus_ids = {
        f"{V3_FREEZE_PROTOCOL.stimulus_id_prefix}"
        f"T{length}_N{count}_seed{seed}"
        for length in PASSAGE_LENGTHS
        for count in NEEDLE_COUNTS
        for seed in SEEDS
    }
    if (
        len(stimulus_ids) != EXPECTED_STIMULI
        or len(set(stimulus_ids)) != EXPECTED_STIMULI
        or set(stimulus_ids) != registered_stimulus_ids
    ):
        raise RuntimeError(
            "V3 dataset must contain the exact 980 registered stimulus IDs"
        )
    for stimulus in stimuli:
        if (
            stimulus.get("schema_version")
            != V3_FREEZE_PROTOCOL.stimulus_schema_version
            or stimulus.get("protocol_version") != PROTOCOL_VERSION
            or int(stimulus["target_passage_tokens"])
            != int(stimulus["canonical_passage_tokens"])
            or int(stimulus["gold_count"]) != int(stimulus["num_needles"])
        ):
            raise RuntimeError("V3 stimulus schema or registered count is invalid")
        _audit_request_depths(stimulus)

    prepare_audit = _load_json(orchestration / "prepare_audit.json")
    if (
        prepare_audit.get("passed") is not True
        or prepare_audit.get("protocol_version") != PROTOCOL_VERSION
        or prepare_audit.get("git", {}).get("dirty") is not False
        or prepare_audit.get("plan", {}).get("tasks_sha256")
        != plan["tasks_sha256"]
        or prepare_audit.get("dataset", {}).get("stimuli_sha256")
        != _sha256(stimuli_path)
    ):
        raise RuntimeError("V3 prepare provenance audit is invalid")
    prepared_commit = str(prepare_audit["git"]["commit"])

    all_rows: list[dict[str, Any]] = []
    model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shard_audits: list[dict[str, Any]] = []
    global_ids: set[str] = set()
    source_files: list[dict[str, Any]] = []

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
        if (
            len(rows) != EXPECTED_STIMULI
            or len(actual) != EXPECTED_STIMULI
            or actual != expected
            or manifest.get("schema_version")
            != V3_RUN_PROTOCOL.run_manifest_schema_version
            or manifest.get("protocol_version") != PROTOCOL_VERSION
            or int(manifest["expected_requests"]) != EXPECTED_STIMULI
            or int(manifest["completed_requests"]) != EXPECTED_STIMULI
            or manifest["model"]["label"] != task["model_label"]
            or manifest["model_revision"] != task["model_revision"]
            or manifest["prompt_modes"] != [task["prompt_mode"]]
            or manifest.get("git", {}).get("commit") != prepared_commit
            or manifest.get("git", {}).get("dirty") is not False
        ):
            raise RuntimeError(f"Structural V3 shard audit failed: {task_id}")
        for row in rows:
            if (
                row.get("schema_version")
                != V3_RUN_PROTOCOL.request_schema_version
                or row.get("protocol_version") != PROTOCOL_VERSION
                or row.get("stimulus_schema_version")
                != V3_FREEZE_PROTOCOL.stimulus_schema_version
            ):
                raise RuntimeError(f"V3 row schema mismatch in {task_id}")
            _audit_request_depths(row)
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
                    "path": str(requests_path.relative_to(run_root)),
                    "sha256": _sha256(requests_path),
                    "bytes": requests_path.stat().st_size,
                },
                {
                    "path": str(manifest_path.relative_to(run_root)),
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

    if len(all_rows) != EXPECTED_REQUESTS:
        raise RuntimeError("Merged V3 row count is not 47,040")
    if len(global_ids) != EXPECTED_REQUESTS:
        raise RuntimeError("Merged V3 request IDs are not all unique")
    if len(shard_audits) != EXPECTED_SHARDS:
        raise RuntimeError("Merged V3 shard count is not 48")

    created_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema_version": "realistic_niah_formal_shard_audit_v3",
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": created_at,
        "passed": True,
        "audit_only": audit_only,
        "run_root": str(run_root),
        "plan_tasks_sha256": plan["tasks_sha256"],
        "git_commit": prepared_commit,
        "prepare_audit_sha256": _sha256(
            orchestration / "prepare_audit.json"
        ),
        "stimuli_sha256": _sha256(stimuli_path),
        "stimuli": len(stimuli),
        "shards": len(shard_audits),
        "requests": len(all_rows),
        "unique_request_ids": len(global_ids),
        "request_ids_sha256": _ordered_id_digest(sorted(global_ids)),
        "insertion_depth_interval": [
            INSERTION_DEPTH_MIN_FRACTION,
            INSERTION_DEPTH_MAX_FRACTION,
        ],
        "source_files": source_files,
        "shard_audits": shard_audits,
        "evaluation": _evaluation_counts(all_rows),
    }

    canonical_files: list[dict[str, Any]] = []
    if not audit_only:
        for model_label, rows in model_rows.items():
            model_tasks = [
                task
                for task in plan["tasks"]
                if task["model_label"] == model_label
            ]
            collection = str(model_tasks[0]["output_collection"])
            output = run_root / collection / model_label / "main"
            rows = sorted(rows, key=lambda row: str(row["request_id"]))
            request_ids = [str(row["request_id"]) for row in rows]
            expected_rows = sum(
                int(task["expected_requests"]) for task in model_tasks
            )
            if len(rows) != expected_rows:
                raise RuntimeError(
                    f"Merged model count mismatch for {model_label}"
                )
            model_manifest = {
                "schema_version": "realistic_niah_merged_run_manifest_v3",
                "protocol_version": PROTOCOL_VERSION,
                "created_at_utc": created_at,
                "model_label": model_label,
                "model_id": model_tasks[0]["model_id"],
                "model_revision": model_tasks[0]["model_revision"],
                "query_layout": plan["query_layout"],
                "prompt_modes": sorted(
                    {str(row["prompt_mode"]) for row in rows}
                ),
                "expected_requests": expected_rows,
                "completed_requests": len(rows),
                "request_ids_sha256": _ordered_id_digest(request_ids),
                "stimuli_sha256": audit["stimuli_sha256"],
                "source_shards": [
                    str(task["task_id"]) for task in model_tasks
                ],
                "formal_plan_tasks_sha256": plan["tasks_sha256"],
            }
            model_qc = {
                "schema_version": "realistic_niah_structural_qc_v3",
                "protocol_version": PROTOCOL_VERSION,
                "created_at_utc": created_at,
                "passed": True,
                "model_label": model_label,
                "expected_requests": expected_rows,
                "completed_requests": len(rows),
                "unique_request_ids": len(set(request_ids)),
                "all_jsonl_rows_parsed": True,
                "all_depths_in_registered_interval": True,
                "evaluation": _evaluation_counts(rows),
            }
            canonical_requests = output / "requests.jsonl"
            canonical_manifest = output / "run_manifest.json"
            canonical_qc = output / "qc_report.json"
            _atomic_jsonl(canonical_requests, rows)
            _atomic_json(canonical_manifest, model_manifest)
            _atomic_json(canonical_qc, model_qc)
            readback_rows = _load_jsonl(canonical_requests)
            readback_manifest = _load_json(canonical_manifest)
            readback_qc = _load_json(canonical_qc)
            if (
                len(readback_rows) != expected_rows
                or len(
                    {
                        str(row["request_id"])
                        for row in readback_rows
                    }
                )
                != expected_rows
                or int(readback_manifest["completed_requests"])
                != expected_rows
                or readback_qc.get("passed") is not True
                or int(readback_qc["completed_requests"]) != expected_rows
            ):
                raise RuntimeError(
                    f"Canonical V3 readback failed for {model_label}"
                )
            canonical_files.extend(
                {
                    "path": str(path.relative_to(run_root)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in (
                    canonical_requests,
                    canonical_manifest,
                    canonical_qc,
                )
            )

    audit["canonical_files"] = canonical_files
    _atomic_json(orchestration / "final_shard_audit.json", audit)
    return audit
