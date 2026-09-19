"""Archive completed N10 Update records and code, referencing frozen tensor/input archives."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tarfile
import time

from backup_enumeration_fresh_read import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "input-backup", "discovery-backup", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    tick = time.monotonic()
    root = a.root.resolve()
    stage = root / "fresh_n10_update_v1"
    state = read(stage / "gpu_pipeline_status.json")
    if state["status"] != "COMPLETE" or state["phase"] != "N10_UPDATE_GPU_COMPLETE_ANALYSIS_PENDING":
        raise ValueError("N10 GPU queue is not complete")
    manifest, assembly = [read(stage / f) for f in ("manifest.json", "assembly_audit.json")]
    if assembly["status"] != "PASS" or assembly["primary_rows"] != manifest["expected_formal_rows"] or manifest["expected_formal_rows"] != 2160:
        raise ValueError("N10 assembly grid mismatch")
    if assembly["primary_rows"] != 2160 or assembly["selection_manifest_sha256"] != sha(stage / "selection_manifest.json"):
        raise ValueError("N10 assembly selection mismatch")
    dependencies, inventories = {}, {}
    for name, folder in (("inputs", a.input_backup), ("discovery", a.discovery_backup)):
        status = read(folder / "backup_status.json")
        if status["status"] != "COMPLETE" or sha(status["archive"]) != status["archive_sha256"]:
            raise ValueError(f"Dependency archive changed: {name}")
        if sha(folder / "backup_manifest.json") != status["manifest_sha256"]:
            raise ValueError(f"Dependency inventory changed: {name}")
        inventories[name] = read(folder / "backup_manifest.json")
        dependencies[name] = {k: status[k] for k in ("archive", "archive_sha256", "manifest_sha256")}
    if inventories["discovery"]["n10_manifest_sha256"] != sha(stage / "manifest.json"):
        raise ValueError("Discovery is from a different N10 run")
    files, tensor_dependencies = {}, {}

    def add(path, expected=None):
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Unsafe archive member: {path}")
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Frozen source changed: {path}")
        name = path.relative_to(root).as_posix()
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}

    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc" or any(v in path.parts for v in ("__pycache__", ".pytest_cache")):
            continue
        if path.suffix == ".npz":
            name = path.relative_to(root).as_posix()
            expected = inventories["discovery"]["files"][name]
            if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError("Frozen discovery tensor changed")
            tensor_dependencies[name] = expected
        else:
            add(path)
    for relative, expected in manifest["code_sha256"].items():
        add(stage / "code" / relative, expected)
    for relative, expected in manifest["contract_sha256"].items():
        add(stage / relative, expected)
    for path, expected in {**manifest["source_sha256"], **assembly["source_files_sha256"]}.items():
        add(path, expected)
    jobs = read(stage / "jobs.json")
    if [j["id"] for j in state["jobs"]] != [j["id"] for j in jobs]:
        raise ValueError("Pipeline jobs incomplete or reordered")
    for job in state["jobs"]:
        status = read(job["status_path"])
        if job["status"] != "COMPLETE" or status["status"] != "COMPLETE":
            raise ValueError(f"Incomplete job: {job['id']}")
        if job["expected_rows"] is not None and job["rows"] != job["expected_rows"]:
            raise ValueError("Job row count mismatch")
    inventory = {"schema": "enumeration_n10_update_incremental_backup_v1", "utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root), "files": files, "tensor_dependencies": tensor_dependencies, "dependencies": dependencies,
        "manifest_sha256": sha(stage / "manifest.json"), "selection_manifest_sha256": sha(stage / "selection_manifest.json"),
        "formal_rows": 2160, "smoke_rows": 216, "script_sha256": sha(__file__), "command": sys.argv,
        "scope": "Completed N10 Update raw outputs, hooks, logs, runtime, assembly, frozen code and contracts; tensors and adapted input geometry remain in referenced backups"}
    a.output.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(inventory, indent=2) + "\n").encode()
    (a.output / "backup_manifest.json").write_bytes(raw)
    archive = a.output / "n10_update_completed.tar.gz"
    with tarfile.open(archive, "x:gz", compresslevel=3) as tar:
        info = tarfile.TarInfo("backup_manifest.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
        for name, record in files.items():
            tar.add(root / name, arcname=name, recursive=False)
            if sha(root / name) != record["sha256"]:
                raise ValueError(f"Source changed while archiving: {name}")
    result = {"status": "COMPLETE", "archive": str(archive), "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size, "files": len(files), "formal_rows": 2160, "smoke_rows": 216,
        "manifest_sha256": sha(a.output / "backup_manifest.json"), "seconds": time.monotonic() - tick}
    (a.output / "backup_status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
