from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realistic_niah_v3.merge import (
    _atomic_json,
    _atomic_jsonl,
    _evaluation_counts,
    _load_json,
    _load_jsonl,
    _ordered_id_digest,
    _sha256,
)

from .sharding import expected_request_ids, formal_bundle_plan, formal_shard_plan
from .resume import audit_resume_manifests, parse_resume_commits
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
    V31_FREEZE_PROTOCOL,
    V31_RUN_PROTOCOL,
)


def _audit_request_depths(row: dict[str, Any]) -> None:
    policy = row.get("insertion_depth_policy")
    if not isinstance(policy, dict):
        raise ValueError("V3.1 row lacks insertion_depth_policy")
    if (
        float(policy["minimum_inclusive"]) != INSERTION_DEPTH_MIN_FRACTION
        or float(policy["maximum_inclusive"]) != INSERTION_DEPTH_MAX_FRACTION
    ):
        raise ValueError("V3.1 row has the wrong insertion-depth policy")
    needles = row.get("needles")
    if not isinstance(needles, list) or len(needles) != int(row["num_needles"]):
        raise ValueError("V3.1 row needle metadata is incomplete")
    for needle in needles:
        depth = float(needle["normalized_depth"])
        if not (INSERTION_DEPTH_MIN_FRACTION <= depth <= INSERTION_DEPTH_MAX_FRACTION):
            raise ValueError(f"V3.1 row contains out-of-range depth: {depth}")


def audit_and_merge(run_root: Path, *, audit_only: bool = False) -> dict[str, Any]:
    run_root = run_root.resolve()
    plan = formal_shard_plan()
    orchestration = run_root / "orchestration"
    dataset = run_root / "dataset"
    saved_plan = _load_json(orchestration / "formal_shards.json")
    if saved_plan != plan:
        raise RuntimeError("Saved V3.1 shard plan differs from the registry")
    bundle_plan = formal_bundle_plan()
    if _load_json(orchestration / "formal_bundles.json") != bundle_plan:
        raise RuntimeError("Saved V3.1 bundle plan differs from the registry")

    dataset_manifest = _load_json(dataset / "manifest.json")
    dataset_audit = _load_json(dataset / "audit_report.json")
    dataset_spec = dataset_manifest.get("spec", {})
    if (
        dataset_manifest.get("schema_version")
        != V31_FREEZE_PROTOCOL.manifest_schema_version
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
        raise RuntimeError("Frozen V3.1 dataset provenance/audit is invalid")

    stimuli_path = dataset / "stimuli.jsonl"
    stimuli = _load_jsonl(stimuli_path)
    stimulus_ids = tuple(str(row["stimulus_id"]) for row in stimuli)
    registered_stimulus_ids = {
        f"{V31_FREEZE_PROTOCOL.stimulus_id_prefix}T{length}_N{count}_seed{seed}"
        for length in PASSAGE_LENGTHS
        for count in NEEDLE_COUNTS
        for seed in SEEDS
    }
    if (
        len(stimulus_ids) != EXPECTED_STIMULI
        or len(set(stimulus_ids)) != EXPECTED_STIMULI
        or set(stimulus_ids) != registered_stimulus_ids
    ):
        raise RuntimeError("V3.1 dataset does not contain the exact registered IDs")
    for stimulus in stimuli:
        if (
            stimulus.get("schema_version")
            != V31_FREEZE_PROTOCOL.stimulus_schema_version
            or stimulus.get("protocol_version") != PROTOCOL_VERSION
            or int(stimulus["target_passage_tokens"])
            != int(stimulus["canonical_passage_tokens"])
            or int(stimulus["gold_count"]) != int(stimulus["num_needles"])
        ):
            raise RuntimeError("V3.1 stimulus schema or count is invalid")
        _audit_request_depths(stimulus)

    prepare_audit = _load_json(orchestration / "prepare_audit.json")
    if (
        prepare_audit.get("passed") is not True
        or prepare_audit.get("protocol_version") != PROTOCOL_VERSION
        or prepare_audit.get("git", {}).get("dirty") is not False
        or prepare_audit.get("plan", {}).get("tasks_sha256") != plan["tasks_sha256"]
        or prepare_audit.get("physical_bundle_plan", {}).get("bundles_sha256")
        != bundle_plan["bundles_sha256"]
        or prepare_audit.get("dataset", {}).get("stimuli_sha256")
        != _sha256(stimuli_path)
    ):
        raise RuntimeError("V3.1 prepare provenance audit is invalid")
    prepared_commit = str(prepare_audit["git"]["commit"])
    resume_audit = audit_resume_manifests(
        run_root,
        current_commit=prepared_commit,
        allowed_commits=parse_resume_commits(
            os.environ.get("REALISTIC_NIAH_RESUME_FROM_COMMITS", "")
        ),
    )
    permitted_manifest_commits = {
        prepared_commit,
        *resume_audit["allowed_commits"],
    }

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
            raise RuntimeError(f"Missing V3.1 shard requests: {task_id}")
        if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
            raise RuntimeError(f"Missing V3.1 shard manifest: {task_id}")
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
            != V31_RUN_PROTOCOL.run_manifest_schema_version
            or manifest.get("protocol_version") != PROTOCOL_VERSION
            or int(manifest.get("expected_requests", -1)) != EXPECTED_STIMULI
            or int(manifest.get("completed_requests", -1)) != EXPECTED_STIMULI
            or manifest.get("model", {}).get("label") != task["model_label"]
            or manifest.get("model_revision") != task["model_revision"]
            or manifest.get("prompt_modes") != [task["prompt_mode"]]
            or manifest.get("git", {}).get("commit")
            not in permitted_manifest_commits
            or manifest.get("git", {}).get("dirty") is not False
        ):
            raise RuntimeError(f"Structural V3.1 shard audit failed: {task_id}")
        for row in rows:
            if (
                row.get("schema_version") != V31_RUN_PROTOCOL.request_schema_version
                or row.get("protocol_version") != PROTOCOL_VERSION
                or row.get("stimulus_schema_version")
                != V31_FREEZE_PROTOCOL.stimulus_schema_version
            ):
                raise RuntimeError(f"V3.1 row schema mismatch in {task_id}")
            _audit_request_depths(row)
        overlap = global_ids.intersection(actual)
        if overlap:
            raise RuntimeError(f"Cross-shard duplicate V3.1 IDs in {task_id}")
        global_ids.update(actual)
        sorted_rows = sorted(rows, key=lambda row: str(row["request_id"]))
        all_rows.extend(sorted_rows)
        model_rows[str(task["model_label"])].extend(sorted_rows)
        source_files.extend(
            {
                "path": str(path.relative_to(run_root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (requests_path, manifest_path)
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

    if len(all_rows) != EXPECTED_REQUESTS or len(global_ids) != EXPECTED_REQUESTS:
        raise RuntimeError("Merged V3.1 request accounting is not 161,280 unique rows")
    if len(shard_audits) != EXPECTED_SHARDS:
        raise RuntimeError("Merged V3.1 shard count is not 48")

    created_at = datetime.now(timezone.utc).isoformat()
    audit: dict[str, Any] = {
        "schema_version": "realistic_niah_formal_shard_audit_v3_1",
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": created_at,
        "passed": True,
        "audit_only": audit_only,
        "run_root": str(run_root),
        "plan_tasks_sha256": plan["tasks_sha256"],
        "bundle_plan_sha256": bundle_plan["bundles_sha256"],
        "physical_model_loads": bundle_plan["physical_model_loads"],
        "git_commit": prepared_commit,
        "resume_manifest_audit": resume_audit,
        "stimuli_sha256": _sha256(stimuli_path),
        "stimuli": len(stimuli),
        "shards": len(shard_audits),
        "requests": len(all_rows),
        "unique_request_ids": len(global_ids),
        "request_ids_sha256": _ordered_id_digest(sorted(global_ids)),
        "source_files": source_files,
        "shard_audits": shard_audits,
        "evaluation": _evaluation_counts(all_rows),
    }

    canonical_files: list[dict[str, Any]] = []
    if not audit_only:
        for model_label, rows in model_rows.items():
            model_tasks = [
                task for task in plan["tasks"] if task["model_label"] == model_label
            ]
            collection = str(model_tasks[0]["output_collection"])
            output = run_root / collection / model_label / "main"
            rows = sorted(rows, key=lambda row: str(row["request_id"]))
            request_ids = [str(row["request_id"]) for row in rows]
            expected_rows = sum(int(task["expected_requests"]) for task in model_tasks)
            if len(rows) != expected_rows or len(set(request_ids)) != expected_rows:
                raise RuntimeError(f"V3.1 model merge mismatch: {model_label}")
            model_manifest = {
                "schema_version": "realistic_niah_merged_run_manifest_v3_1",
                "protocol_version": PROTOCOL_VERSION,
                "created_at_utc": created_at,
                "model_label": model_label,
                "model_id": model_tasks[0]["model_id"],
                "model_revision": model_tasks[0]["model_revision"],
                "query_layout": plan["query_layout"],
                "prompt_modes": sorted({str(row["prompt_mode"]) for row in rows}),
                "expected_requests": expected_rows,
                "completed_requests": len(rows),
                "request_ids_sha256": _ordered_id_digest(request_ids),
                "stimuli_sha256": audit["stimuli_sha256"],
                "source_shards": [str(task["task_id"]) for task in model_tasks],
                "formal_plan_tasks_sha256": plan["tasks_sha256"],
            }
            model_qc = {
                "schema_version": "realistic_niah_structural_qc_v3_1",
                "protocol_version": PROTOCOL_VERSION,
                "created_at_utc": created_at,
                "passed": True,
                "model_label": model_label,
                "expected_requests": expected_rows,
                "completed_requests": len(rows),
                "unique_request_ids": len(set(request_ids)),
                "evaluation": _evaluation_counts(rows),
            }
            paths = (
                output / "requests.jsonl",
                output / "run_manifest.json",
                output / "qc_report.json",
            )
            _atomic_jsonl(paths[0], rows)
            _atomic_json(paths[1], model_manifest)
            _atomic_json(paths[2], model_qc)
            if len(_load_jsonl(paths[0])) != expected_rows:
                raise RuntimeError(f"Canonical V3.1 readback failed: {model_label}")
            canonical_files.extend(
                {
                    "path": str(path.relative_to(run_root)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in paths
            )

    audit["canonical_files"] = canonical_files
    _atomic_json(orchestration / "final_shard_audit.json", audit)
    return audit
