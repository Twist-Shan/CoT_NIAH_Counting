from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realistic_niah_v3.sharding import formal_shard_plan
from realistic_niah_v3.spec import EXPECTED_STIMULI, PROTOCOL_VERSION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _mount_snapshot(path: Path) -> dict[str, Any]:
    if platform.system() != "Linux":
        return {"available": False, "reason": "findmnt is Linux-only"}
    result = subprocess.run(
        ["findmnt", "-J", "-T", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"available": True, "findmnt": json.loads(result.stdout)}


def prepare(run_root: Path, repo_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    repo_root = repo_root.resolve()
    dataset = run_root / "dataset"
    orchestration = run_root / "orchestration"
    stimuli_path = dataset / "stimuli.jsonl"
    manifest_path = dataset / "manifest.json"
    audit_path = dataset / "audit_report.json"
    for path in (stimuli_path, manifest_path, audit_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/non-empty V3 dataset file: {path}")

    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        dataset_manifest.get("protocol_version") != PROTOCOL_VERSION
        or dataset_audit.get("protocol_version") != PROTOCOL_VERSION
        or dataset_audit.get("passed") is not True
        or int(dataset_audit.get("rows_checked", -1)) != EXPECTED_STIMULI
    ):
        raise RuntimeError("The frozen V3 dataset has not passed its audit")

    dirty = _git(repo_root, "status", "--short")
    if dirty:
        raise RuntimeError("Formal V3 preparation requires a clean worktree")
    plan = formal_shard_plan()
    orchestration.mkdir(parents=True, exist_ok=True)
    json_path = orchestration / "formal_shards.json"
    tsv_path = orchestration / "formal_shards.tsv"
    _atomic_text(
        json_path,
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    fields = (
        "task_id",
        "priority",
        "model_label",
        "prompt_mode",
        "output_collection",
        "expected_requests",
        "model_revision",
        "gpus_required",
        "tensor_parallel_size",
        "request_batch_size",
        "max_num_seqs",
        "gpu_memory_utilization",
        "max_model_len",
    )
    lines = ["\t".join(fields)]
    lines.extend(
        "\t".join(str(task[field]) for field in fields)
        for task in plan["tasks"]
    )
    _atomic_text(tsv_path, "\n".join(lines) + "\n")

    audit = {
        "schema_version": "realistic_niah_v3_prepare_audit_v1",
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "git": {
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": False,
        },
        "mount": _mount_snapshot(run_root),
        "dataset": {
            "stimuli": EXPECTED_STIMULI,
            "stimuli_sha256": _sha256(stimuli_path),
            "manifest_sha256": _sha256(manifest_path),
            "audit_sha256": _sha256(audit_path),
        },
        "plan": {
            "path": str(json_path),
            "sha256": _sha256(json_path),
            "tasks_sha256": plan["tasks_sha256"],
            "shards": plan["expected_shards"],
            "requests": plan["expected_requests"],
        },
    }
    _atomic_text(
        orchestration / "prepare_audit.json",
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and prepare an already frozen formal V3 run."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = prepare(Path(args.run_root), Path(args.repo_root))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
