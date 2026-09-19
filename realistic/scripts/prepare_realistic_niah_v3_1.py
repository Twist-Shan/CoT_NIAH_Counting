from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realistic_niah_v3_1.integrity import validate_frozen_dataset
from realistic_niah_v3_1.sharding import formal_bundle_plan, formal_shard_plan
from realistic_niah_v3_1.spec import EXPECTED_STIMULI, PROTOCOL_VERSION


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def prepare(
    run_root: Path,
    repo_root: Path,
    expected_commit: str | None = None,
    *,
    require_source_revision: bool = False,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    repo_root = repo_root.resolve()
    dataset = run_root / "dataset"
    orchestration = run_root / "orchestration"
    stimuli_path = dataset / "stimuli.jsonl"
    manifest_path = dataset / "manifest.json"
    audit_path = dataset / "audit_report.json"
    for path in (stimuli_path, manifest_path, audit_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing V3.1 dataset file: {path}")
    dataset_integrity = validate_frozen_dataset(
        dataset, require_source_revision=require_source_revision
    )
    dirty = _git(repo_root, "status", "--short")
    if dirty:
        raise RuntimeError("Formal V3.1 preparation requires a clean worktree")
    actual_commit = _git(repo_root, "rev-parse", "HEAD")
    resolved_expected_commit = expected_commit or actual_commit
    if actual_commit != resolved_expected_commit:
        raise RuntimeError(
            f"Formal V3.1 commit mismatch: {actual_commit} != {resolved_expected_commit}"
        )
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
    )
    lines = ["\t".join(fields)]
    lines.extend(
        "\t".join(str(task[field]) for field in fields) for task in plan["tasks"]
    )
    _atomic_text(tsv_path, "\n".join(lines) + "\n")
    bundle_plan = formal_bundle_plan()
    bundle_json_path = orchestration / "formal_bundles.json"
    bundle_tsv_path = orchestration / "formal_bundles.tsv"
    _atomic_text(
        bundle_json_path,
        json.dumps(bundle_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    bundle_fields = (
        "bundle_id",
        "priority",
        "model_label",
        "expected_logical_shards",
        "expected_requests",
        "model_revision",
        "prompt_modes",
        "logical_task_ids",
    )
    bundle_lines = ["\t".join(bundle_fields)]
    for bundle in bundle_plan["bundles"]:
        values = dict(bundle)
        values["prompt_modes"] = ",".join(bundle["prompt_modes"])
        values["logical_task_ids"] = ",".join(bundle["logical_task_ids"])
        bundle_lines.append("\t".join(str(values[field]) for field in bundle_fields))
    _atomic_text(bundle_tsv_path, "\n".join(bundle_lines) + "\n")
    audit = {
        "schema_version": "realistic_niah_v3_1_prepare_audit_v1",
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "git": {
            "commit": actual_commit,
            "expected_commit": resolved_expected_commit,
            "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": False,
        },
        "mount": _mount_snapshot(run_root),
        "dataset": {
            "stimuli": EXPECTED_STIMULI,
            "dataset_id": dataset_integrity["dataset_id"],
            "revision": dataset_integrity["revision"],
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
        "physical_bundle_plan": {
            "path": str(bundle_json_path),
            "sha256": _sha256(bundle_json_path),
            "bundles_sha256": bundle_plan["bundles_sha256"],
            "physical_bundles": bundle_plan["physical_bundles"],
            "loads_avoided": bundle_plan["loads_avoided"],
        },
    }
    _atomic_text(
        orchestration / "prepare_audit.json",
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a frozen V3.1 run.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-source-revision", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                Path(args.run_root),
                Path(args.repo_root),
                args.expected_commit,
                require_source_revision=args.require_source_revision,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
