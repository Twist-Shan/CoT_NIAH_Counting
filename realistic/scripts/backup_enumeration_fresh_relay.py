"""Incrementally archive completed Relay outputs; reference the verified Update input archive."""
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
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--input-backup", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    root = a.root.resolve()
    stage = root / "fresh_relay_native_gpu_v1"
    state, manifest = read(stage / "gpu_pipeline_status.json"), read(stage / "manifest.json")
    if state["status"] != "COMPLETE":
        raise ValueError("Relay queue is not complete")
    base = read(a.input_backup / "backup_status.json")
    if base["status"] != "COMPLETE" or sha(base["archive"]) != base["archive_sha256"]:
        raise ValueError("Frozen input dependency archive mismatch")
    dependency = read(a.input_backup / "backup_manifest.json")
    for name, expected in (("fresh_v1/manifest.json", manifest["baseline_manifest_sha256"]),
                           ("fresh_causal_v1/code_manifest.json", manifest["causal_code_manifest_sha256"])):
        if dependency["files"][name]["sha256"] != expected:
            raise ValueError("Input backup does not match Relay")
    registry = Path(manifest["registry_root"]).resolve()
    if not registry.is_relative_to(root) or sha(registry / "common_cohorts.json") != manifest["cohorts_sha256"]:
        raise ValueError("Wrong frozen relay registry")
    for name, expected in manifest["entrypoints_sha256"].items():
        if sha(stage / "code" / name) != expected:
            raise ValueError(f"Frozen Relay code changed: {name}")
    for name, key in (("protocol.json", "protocol_copy_sha256"), ("job_manifest.json", "job_manifest_sha256")):
        if sha(stage / name) != manifest[key]:
            raise ValueError("Relay contract changed")
    jobs = read(stage / "job_manifest.json")["jobs"]
    for job in jobs:
        status = read(Path(job["output"]) / "status.json")
        if status["status"] != "COMPLETE" or status["completed_rows"] != job["expected_rows"]:
            raise ValueError(f"Incomplete Relay job: {job['id']}")
    files = {}
    for folder in (stage, registry):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError("Unsafe backup member")
                files[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    inventory = {"schema": "enumeration_relay_incremental_backup_v1", "utc": datetime.now(timezone.utc).isoformat(),
                 "source_root": str(root), "files": files, "script_sha256": sha(__file__), "command": sys.argv,
                 "formal_rows": sum(j["expected_rows"] for j in jobs if j["phase"] == "formal"),
                 "smoke_rows": sum(j["expected_rows"] for j in jobs if j["phase"] == "smoke"),
                 "dependency": {"archive": base["archive"], "archive_sha256": base["archive_sha256"],
                                "manifest_sha256": base["manifest_sha256"],
                                "purpose": "Frozen causal kernel, baseline protocol and adapted model inputs are in the completed Update backup."},
                 "scope": "All completed Relay outputs, runtime logs, stage code and per-mode registry; statistical audit pending"}
    a.output.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(inventory, indent=2, ensure_ascii=True) + "\n").encode()
    (a.output / "backup_manifest.json").write_bytes(raw)
    archive = a.output / "relay_completed.tar.gz"
    with tarfile.open(archive, "x:gz", compresslevel=3) as tar:
        info = tarfile.TarInfo("backup_manifest.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
        for name, item in files.items():
            tar.add(root / name, arcname=name, recursive=False)
            if sha(root / name) != item["sha256"]:
                raise ValueError("Relay output changed while archiving")
    result = {"status": "COMPLETE", "archive": str(archive), "archive_sha256": sha(archive),
              "archive_bytes": archive.stat().st_size, "files": len(files),
              "formal_rows": inventory["formal_rows"], "smoke_rows": inventory["smoke_rows"],
              "manifest_sha256": sha(a.output / "backup_manifest.json"), "seconds": time.monotonic() - start}
    (a.output / "backup_status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
