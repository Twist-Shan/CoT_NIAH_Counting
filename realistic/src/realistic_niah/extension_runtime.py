from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .spec import NEEDLE_COUNTS, PASSAGE_LENGTHS, QUERY_LAYOUT, SEEDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_digest(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _dataset_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(
        candidate for candidate in dataset_root.rglob("*")
        if candidate.is_file()
    ):
        files.append(
            {
                "path": path.relative_to(dataset_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def _copy_dataset_verified(
    source_root: Path,
    destination_root: Path,
    manifest: list[dict[str, Any]],
) -> None:
    for item in manifest:
        relative = Path(str(item["path"]))
        source = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if (
                destination.stat().st_size != int(item["bytes"])
                or sha256_file(destination) != str(item["sha256"])
            ):
                raise RuntimeError(
                    f"Refusing to overwrite mismatched dataset file: "
                    f"{destination}"
                )
            continue
        shutil.copy2(source, destination)
        if (
            destination.stat().st_size != int(item["bytes"])
            or sha256_file(destination) != str(item["sha256"])
        ):
            raise RuntimeError(
                f"Copied dataset verification failed: {destination}"
            )


def _validate_stimuli(
    stimuli_path: Path,
    *,
    expected_sha256: str,
    expected_stimuli: int,
) -> list[dict[str, Any]]:
    if sha256_file(stimuli_path) != expected_sha256:
        raise RuntimeError("Source stimuli SHA256 does not match frozen V2 data")
    rows = load_jsonl(stimuli_path)
    ids = [str(row["stimulus_id"]) for row in rows]
    if len(rows) != expected_stimuli or len(set(ids)) != expected_stimuli:
        raise RuntimeError(
            f"Source dataset must contain {expected_stimuli} unique stimuli"
        )
    actual_grid = {
        (
            int(row["target_passage_tokens"]),
            int(row["num_needles"]),
            int(row["seed"]),
        )
        for row in rows
    }
    expected_grid = {
        (passage_length, needle_count, seed)
        for passage_length in PASSAGE_LENGTHS
        for needle_count in NEEDLE_COUNTS
        for seed in SEEDS
    }
    if actual_grid != expected_grid:
        raise RuntimeError("Source stimuli do not cover the exact frozen V2 grid")
    return rows


def write_extension_plan(
    run_root: Path,
    *,
    plan: dict[str, Any],
    plan_basename: str,
) -> None:
    orchestration = run_root / "orchestration"
    json_path = orchestration / f"{plan_basename}.json"
    tsv_path = orchestration / f"{plan_basename}.tsv"
    if json_path.exists() and load_json(json_path) != plan:
        raise RuntimeError("Existing extension plan differs from registration")
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
        "\t".join(str(task[field]) for field in fields)
        for task in plan["tasks"]
    )
    atomic_json(json_path, plan)
    atomic_text(tsv_path, "\n".join(lines) + "\n")


def prepare_extension(
    source_formal_run_root: Path,
    run_root: Path,
    repo_root: Path,
    *,
    plan: dict[str, Any],
    plan_basename: str,
    run_name_prefixes: tuple[str, ...],
    schema_prefix: str,
) -> dict[str, Any]:
    source_formal_run_root = source_formal_run_root.resolve()
    run_root = run_root.resolve()
    repo_root = repo_root.resolve()
    if source_formal_run_root == run_root:
        raise ValueError("Extension run root must differ from source formal run")
    if not run_root.name.startswith(run_name_prefixes):
        raise ValueError(
            f"Run directory must start with one of {run_name_prefixes}"
        )

    source_dataset = source_formal_run_root / "dataset"
    source_stimuli = source_dataset / "stimuli.jsonl"
    source_audit_path = (
        source_formal_run_root / "orchestration" / "final_shard_audit.json"
    )
    if not source_stimuli.is_file() or source_stimuli.stat().st_size == 0:
        raise RuntimeError("Source formal stimuli are missing or empty")
    if not source_audit_path.is_file() or source_audit_path.stat().st_size == 0:
        raise RuntimeError("Source formal final audit is missing or empty")

    expected_sha = str(plan["source_stimuli_sha256"])
    expected_stimuli = int(plan["expected_stimuli_per_shard"])
    stimuli = _validate_stimuli(
        source_stimuli,
        expected_sha256=expected_sha,
        expected_stimuli=expected_stimuli,
    )
    source_audit = load_json(source_audit_path)
    if (
        source_audit.get("passed") is not True
        or int(source_audit.get("requests", -1)) != 14_500
        or int(source_audit.get("unique_request_ids", -1)) != 14_500
        or source_audit.get("stimuli_sha256") != expected_sha
    ):
        raise RuntimeError("Source V2 formal final audit did not pass exactly")

    dataset_files = _dataset_manifest(source_dataset)
    if not dataset_files:
        raise RuntimeError("Source dataset directory is empty")
    _copy_dataset_verified(
        source_dataset,
        run_root / "dataset",
        dataset_files,
    )
    destination_stimuli = run_root / "dataset" / "stimuli.jsonl"
    if sha256_file(destination_stimuli) != expected_sha:
        raise RuntimeError("Destination stimuli verification failed")

    write_extension_plan(
        run_root,
        plan=plan,
        plan_basename=plan_basename,
    )
    git = _git_provenance(repo_root)
    provenance_path = (
        run_root / "orchestration" / "source_formal_run_provenance.json"
    )
    stable_provenance = {
        "schema_version": f"{schema_prefix}_source_provenance_v1",
        "source_formal_run_root": str(source_formal_run_root),
        "source_final_audit": {
            "path": str(source_audit_path),
            "sha256": sha256_file(source_audit_path),
            "requests": 14_500,
            "unique_request_ids": 14_500,
            "passed": True,
        },
        "dataset_files": dataset_files,
        "stimuli": len(stimuli),
        "stimuli_sha256": expected_sha,
        "extension_plan_tasks_sha256": plan["tasks_sha256"],
        "repo_root": str(repo_root),
        "git_commit": git["commit"],
    }
    if provenance_path.exists():
        existing = load_json(provenance_path)
        comparable = {
            key: value
            for key, value in existing.items()
            if key != "created_at_utc"
        }
        if comparable != stable_provenance:
            raise RuntimeError(
                "Existing provenance differs; use a new extension run root"
            )
        provenance = existing
    else:
        provenance = {
            **stable_provenance,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(provenance_path, provenance)

    result = {
        "schema_version": f"{schema_prefix}_prepare_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "run_root": str(run_root),
        "source_formal_run_root": str(source_formal_run_root),
        "stimuli": len(stimuli),
        "stimuli_sha256": sha256_file(destination_stimuli),
        "dataset_files": len(dataset_files),
        "expected_shards": int(plan["expected_shards"]),
        "expected_requests": int(plan["expected_requests"]),
        "plan_tasks_sha256": plan["tasks_sha256"],
        "git": git,
    }
    atomic_json(run_root / "orchestration" / "prepare_audit.json", result)
    return result


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


def _expected_request_ids(
    stimulus_ids: Iterable[str],
    task: dict[str, Any],
) -> set[str]:
    return {
        f"{task['model_label']}/{task['prompt_mode']}/"
        f"{QUERY_LAYOUT}/{stimulus_id}"
        for stimulus_id in stimulus_ids
    }


def audit_and_merge_extension(
    run_root: Path,
    *,
    plan: dict[str, Any],
    plan_basename: str,
    schema_prefix: str,
    audit_only: bool,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    saved_plan = load_json(
        run_root / "orchestration" / f"{plan_basename}.json"
    )
    provenance = load_json(
        run_root / "orchestration" / "source_formal_run_provenance.json"
    )
    stimuli_path = run_root / "dataset" / "stimuli.jsonl"
    if saved_plan != plan:
        raise RuntimeError("Saved extension plan differs from registration")
    if sha256_file(stimuli_path) != str(plan["source_stimuli_sha256"]):
        raise RuntimeError("Extension stimuli SHA256 is not the frozen V2 SHA")
    if (
        provenance.get("stimuli_sha256")
        != str(plan["source_stimuli_sha256"])
    ):
        raise RuntimeError("Source provenance has a mismatched stimuli SHA256")
    if provenance.get("extension_plan_tasks_sha256") != plan["tasks_sha256"]:
        raise RuntimeError("Source provenance has a mismatched plan SHA256")

    stimuli = load_jsonl(stimuli_path)
    stimulus_ids = tuple(str(row["stimulus_id"]) for row in stimuli)
    expected_stimuli = int(plan["expected_stimuli_per_shard"])
    if (
        len(stimulus_ids) != expected_stimuli
        or len(set(stimulus_ids)) != expected_stimuli
    ):
        raise RuntimeError(
            f"Extension dataset must have {expected_stimuli} unique stimuli"
        )

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

        rows = load_jsonl(requests_path)
        manifest = load_json(manifest_path)
        expected = _expected_request_ids(stimulus_ids, task)
        actual_ids = [str(row["request_id"]) for row in rows]
        actual = set(actual_ids)
        expected_requests = int(task["expected_requests"])
        row_revisions = {str(row.get("model_revision")) for row in rows}
        row_models = {str(row.get("model_label")) for row in rows}
        row_modes = {str(row.get("prompt_mode")) for row in rows}
        manifest_model = manifest.get("model", {})
        if (
            len(rows) != expected_requests
            or len(actual) != expected_requests
            or actual != expected
            or int(manifest["expected_requests"]) != expected_requests
            or int(manifest["completed_requests"]) != expected_requests
            or manifest_model.get("label") != task["model_label"]
            or manifest_model.get("model_id") != task["model_id"]
            or manifest_model.get("reasoning_policy")
            != task["reasoning_policy"]
            or manifest_model.get("chat_template_control")
            != task["chat_template_control"]
            or manifest_model.get("system_prompt_strategy")
            != task["system_prompt_strategy"]
            or manifest_model.get("engine_profile") != task["engine_profile"]
            or manifest["model_revision"] != task["model_revision"]
            or manifest["prompt_modes"] != [task["prompt_mode"]]
            or manifest["stimuli_sha256"]
            != str(plan["source_stimuli_sha256"])
            or manifest["git"]["commit"] != expected_git_commit
            or row_revisions != {str(task["model_revision"])}
            or row_models != {str(task["model_label"])}
            or row_modes != {str(task["prompt_mode"])}
        ):
            raise RuntimeError(f"Structural shard audit failed: {task_id}")
        overlap = global_ids.intersection(actual)
        if overlap:
            raise RuntimeError(
                f"Cross-shard duplicate IDs in {task_id}: "
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
                    "sha256": sha256_file(requests_path),
                    "bytes": requests_path.stat().st_size,
                },
                {
                    "path": manifest_path.relative_to(run_root).as_posix(),
                    "sha256": sha256_file(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                },
            )
        )
        shard_audits.append(
            {
                "task_id": task_id,
                "logical_model_label": task["logical_model_label"],
                "model_label": task["model_label"],
                "prompt_mode": task["prompt_mode"],
                "requests": len(rows),
                "request_ids_sha256": ordered_id_digest(sorted(actual)),
                "evaluation": _evaluation_counts(rows),
                "passed": True,
            }
        )

    expected_total = int(plan["expected_requests"])
    if len(all_rows) != expected_total or len(global_ids) != expected_total:
        raise RuntimeError(
            f"Merged extension must have {expected_total} unique requests"
        )

    created_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema_version": f"{schema_prefix}_audit_v1",
        "created_at_utc": created_at,
        "passed": True,
        "audit_only": audit_only,
        "run_root": str(run_root),
        "plan_tasks_sha256": plan["tasks_sha256"],
        "stimuli_sha256": sha256_file(stimuli_path),
        "stimuli": len(stimuli),
        "shards": len(shard_audits),
        "requests": len(all_rows),
        "unique_request_ids": len(global_ids),
        "request_ids_sha256": ordered_id_digest(sorted(global_ids)),
        "git_commit": expected_git_commit,
        "source_files": source_files,
        "shard_audits": shard_audits,
        "evaluation": _evaluation_counts(all_rows),
    }

    if not audit_only:
        variant_outputs: dict[str, dict[str, Any]] = {}
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
                "logical_model_label": model_tasks[0][
                    "logical_model_label"
                ],
                "model_label": model_label,
                "model_id": model_tasks[0]["model_id"],
                "model_revision": model_tasks[0]["model_revision"],
                "query_layout": plan["query_layout"],
                "prompt_modes": modes,
                "expected_requests": len(rows),
                "completed_requests": len(rows),
                "request_ids_sha256": ordered_id_digest(request_ids),
                "stimuli_sha256": audit["stimuli_sha256"],
                "source_shards": [
                    str(task["task_id"]) for task in model_tasks
                ],
                "extension_plan_tasks_sha256": plan["tasks_sha256"],
                "git_commit": expected_git_commit,
            }
            model_qc = {
                "schema_version": "realistic_niah_structural_qc_v2",
                "created_at_utc": created_at,
                "passed": True,
                "logical_model_label": model_tasks[0][
                    "logical_model_label"
                ],
                "model_label": model_label,
                "expected_requests": len(rows),
                "completed_requests": len(rows),
                "unique_request_ids": len(set(request_ids)),
                "all_jsonl_rows_parsed": True,
                "evaluation": _evaluation_counts(rows),
            }
            atomic_jsonl(output / "requests.jsonl", rows)
            atomic_json(output / "run_manifest.json", model_manifest)
            atomic_json(output / "qc_report.json", model_qc)
            variant_outputs[model_label] = {
                "model_label": model_label,
                "model_id": model_tasks[0]["model_id"],
                "model_revision": model_tasks[0]["model_revision"],
                "prompt_modes": modes,
                "requests": len(rows),
                "path": output.relative_to(run_root).as_posix(),
            }

        logical_groups = plan["logical_groups"]
        planned_models = {
            str(task["model_label"]) for task in plan["tasks"]
        }
        grouped_models = {
            str(variant)
            for group in logical_groups.values()
            for variant in group["variants"]
        }
        if grouped_models != planned_models:
            raise RuntimeError("Logical groups do not cover planned models")
        for logical_label, group in sorted(logical_groups.items()):
            variants = [
                variant_outputs[str(label)]
                for label in group["variants"]
            ]
            family_manifest = {
                "schema_version": (
                    "realistic_niah_model_family_manifest_v1"
                ),
                "created_at_utc": created_at,
                "logical_model_label": logical_label,
                "comparison_type": group["comparison_type"],
                "expected_requests": sum(
                    int(variant["requests"]) for variant in variants
                ),
                "completed_requests": sum(
                    int(variant["requests"]) for variant in variants
                ),
                "stimuli_sha256": audit["stimuli_sha256"],
                "extension_plan_tasks_sha256": plan["tasks_sha256"],
                "variants": variants,
            }
            atomic_json(
                run_root
                / "models"
                / logical_label
                / "family_manifest.json",
                family_manifest,
            )

    atomic_json(
        run_root / "orchestration" / "final_shard_audit.json",
        audit,
    )
    return audit
