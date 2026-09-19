from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realistic_niah.olmo3_extension import (
    EXPECTED_STIMULI_PER_SHARD,
    SOURCE_FORMAL_STIMULI_SHA256,
    olmo3_extension_plan,
)
from realistic_niah.spec import NEEDLE_COUNTS, PASSAGE_LENGTHS, SEEDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


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
    for path in sorted(candidate for candidate in dataset_root.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": path.relative_to(dataset_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
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
                or _sha256(destination) != str(item["sha256"])
            ):
                raise RuntimeError(
                    f"Refusing to overwrite mismatched dataset file: {destination}"
                )
            continue
        shutil.copy2(source, destination)
        if (
            destination.stat().st_size != int(item["bytes"])
            or _sha256(destination) != str(item["sha256"])
        ):
            raise RuntimeError(f"Copied dataset verification failed: {destination}")


def _validate_stimuli(stimuli_path: Path) -> list[dict[str, Any]]:
    if _sha256(stimuli_path) != SOURCE_FORMAL_STIMULI_SHA256:
        raise RuntimeError("Source stimuli SHA256 does not match frozen V2 data")
    rows = _load_jsonl(stimuli_path)
    stimulus_ids = [str(row["stimulus_id"]) for row in rows]
    if (
        len(rows) != EXPECTED_STIMULI_PER_SHARD
        or len(set(stimulus_ids)) != EXPECTED_STIMULI_PER_SHARD
    ):
        raise RuntimeError("Source dataset must contain 500 unique stimuli")
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


def _write_plan(run_root: Path) -> dict[str, Any]:
    plan = olmo3_extension_plan()
    orchestration = run_root / "orchestration"
    json_path = orchestration / "olmo3_extension_shards.json"
    tsv_path = orchestration / "olmo3_extension_shards.tsv"
    if json_path.exists() and _load_json(json_path) != plan:
        raise RuntimeError("Existing OLMo extension plan differs from registration")
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
    _atomic_json(json_path, plan)
    _atomic_text(tsv_path, "\n".join(lines) + "\n")
    return plan


def prepare(
    source_formal_run_root: Path,
    run_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    source_formal_run_root = source_formal_run_root.resolve()
    run_root = run_root.resolve()
    repo_root = repo_root.resolve()
    if source_formal_run_root == run_root:
        raise ValueError("Extension run root must differ from source formal run")
    if not run_root.name.startswith(("olmo3_7b_extension_", "olmo3_7b_smoke_")):
        raise ValueError(
            "Run directory name must start with olmo3_7b_extension_ "
            "or olmo3_7b_smoke_"
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

    stimuli = _validate_stimuli(source_stimuli)
    source_audit = _load_json(source_audit_path)
    if (
        source_audit.get("passed") is not True
        or int(source_audit.get("requests", -1)) != 14_500
        or int(source_audit.get("unique_request_ids", -1)) != 14_500
        or source_audit.get("stimuli_sha256") != SOURCE_FORMAL_STIMULI_SHA256
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
    if _sha256(destination_stimuli) != SOURCE_FORMAL_STIMULI_SHA256:
        raise RuntimeError("Destination stimuli verification failed")

    plan = _write_plan(run_root)
    git = _git_provenance(repo_root)
    provenance_path = (
        run_root / "orchestration" / "source_formal_run_provenance.json"
    )
    stable_provenance = {
        "schema_version": "realistic_niah_olmo3_source_provenance_v1",
        "source_formal_run_root": str(source_formal_run_root),
        "source_final_audit": {
            "path": str(source_audit_path),
            "sha256": _sha256(source_audit_path),
            "requests": 14_500,
            "unique_request_ids": 14_500,
            "passed": True,
        },
        "dataset_files": dataset_files,
        "stimuli": len(stimuli),
        "stimuli_sha256": SOURCE_FORMAL_STIMULI_SHA256,
        "extension_plan_tasks_sha256": plan["tasks_sha256"],
        "repo_root": str(repo_root),
        "git_commit": git["commit"],
    }
    if provenance_path.exists():
        existing = _load_json(provenance_path)
        comparable = {
            key: value
            for key, value in existing.items()
            if key != "created_at_utc"
        }
        if comparable != stable_provenance:
            raise RuntimeError(
                "Existing source provenance differs; use a new extension run root"
            )
        provenance = existing
    else:
        provenance = {
            **stable_provenance,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(provenance_path, provenance)

    result = {
        "schema_version": "realistic_niah_olmo3_prepare_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "run_root": str(run_root),
        "source_formal_run_root": str(source_formal_run_root),
        "stimuli": len(stimuli),
        "stimuli_sha256": _sha256(destination_stimuli),
        "dataset_files": len(dataset_files),
        "expected_shards": plan["expected_shards"],
        "expected_requests": plan["expected_requests"],
        "plan_tasks_sha256": plan["tasks_sha256"],
        "git": git,
    }
    _atomic_json(
        run_root / "orchestration" / "prepare_audit.json",
        result,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a self-contained OLMo 3 extension from the audited V2 "
            "formal stimuli."
        )
    )
    parser.add_argument("--source-formal-run-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = prepare(
        Path(args.source_formal_run_root),
        Path(args.run_root),
        Path(args.repo_root),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
