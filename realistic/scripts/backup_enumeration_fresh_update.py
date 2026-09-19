"""Archive completed primary Update results, exact source rows, and frozen inputs."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tick = time.monotonic()
    root = args.root.resolve()
    update, causal, baseline = (root / name for name in
                              ("fresh_native_update_v1", "fresh_causal_v1", "fresh_v1"))
    paths = set()

    def add(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid backup input: {path}")
        if expected is not None and sha(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
        paths.add(path)

    state, manifest, assembly = (read(update / name) for name in
                                ("gpu_pipeline_status.json", "manifest.json", "assembly_audit.json"))
    if state["status"] != "COMPLETE" or assembly["status"] != "PASS":
        raise ValueError("Primary Update is not complete and audited")
    if assembly["primary_rows"] != manifest["primary_expected_rows"]:
        raise ValueError("Primary row count mismatch")
    for path in update.rglob("*"):
        if path.is_file():
            add(path)
    for path, expected in assembly["source_files_sha256"].items():
        add(path, expected)
        parent = Path(path).parent
        for name in ("technical_audit.json", "manifest.json", "command.json", "status.json", "geometry_audit.jsonl", "summary.jsonl"):
            if (parent / name).is_file():
                add(parent / name)
    for name, key in (("protocol.json", "protocol_sha256"), ("job_manifest.json", "job_manifest_sha256"),
                      ("launch_enumeration_native_update.py", "entrypoint_sha256"),
                      ("test_enumeration_native_update.py", "test_sha256")):
        add(update / name, manifest[key])
    add(causal / "code_manifest.json", manifest["causal_code_manifest_sha256"])
    add(causal / "cohorts.json", manifest["old_cohorts_sha256"])
    add(baseline / "manifest.json", manifest["baseline_manifest_sha256"])
    add(baseline / "protocol.json", read(baseline / "manifest.json")["protocol_sha256"])
    code = read(causal / "code_manifest.json")
    for name, expected in {**code["original_code_sha256"], **code["additive_code_sha256"]}.items():
        if name.startswith(("src/", "scripts/")):
            add(causal / "code" / name, expected)
    for cell in manifest["cells"]:
        add(cell["cohort"], cell["cohort_sha256"])
        registry = Path(cell["registry"])
        add(registry / "manifest.json", cell["registry_sha256"])
        registered = read(registry / "manifest.json")
        add(registry / "ledger.json", registered["ledger_sha256"])
        add(registry / "adapted_generations.jsonl", registered["adapted_generations_sha256"])
        for row in read(registry / "ledger.json"):
            if row["seed"] in cell["selected_seeds"] and row["gold_count"] == 10:
                add(registry / row["geometry_file"], row["geometry_sha256"])
    files = {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(paths)}
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = {"schema": "enumeration_update_backup_v1", "utc": datetime.now(timezone.utc).isoformat(),
                 "source_root": str(root), "files": files, "primary_rows": assembly["primary_rows"],
                 "command": sys.argv, "backup_script_sha256": sha(__file__),
                 "helper_sha256": sha(Path(__file__).with_name("backup_enumeration_fresh_read.py")),
                 "inventory_seconds": time.monotonic() - tick,
                 "scope": "Completed primary Update; original source outputs; frozen code, sampling and adapted model inputs"}
    raw = (json.dumps(inventory, indent=2, ensure_ascii=True) + "\n").encode()
    (args.output / "backup_manifest.json").write_bytes(raw)
    archive = args.output / "update_completed.tar.gz"
    with tarfile.open(archive, "x:gz", compresslevel=3) as tar:
        info = tarfile.TarInfo("backup_manifest.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
        for name, record in files.items():
            tar.add(root / name, arcname=name, recursive=False)
            if sha(root / name) != record["sha256"]:
                raise ValueError(f"Input changed while archiving: {name}")
    result = {"status": "COMPLETE", "archive": str(archive), "archive_sha256": sha(archive),
              "archive_bytes": archive.stat().st_size, "files": len(files),
              "source_bytes": sum(x["bytes"] for x in files.values()),
              "primary_rows": assembly["primary_rows"], "manifest_sha256": sha(args.output / "backup_manifest.json"),
              "seconds": time.monotonic() - tick}
    (args.output / "backup_status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
