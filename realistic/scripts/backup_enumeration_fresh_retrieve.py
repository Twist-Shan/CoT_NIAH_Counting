"""Archive the completed Retrieve localization, frozen banks and behavior grid."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tarfile

from backup_enumeration_fresh_answer_patch import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--input-backup", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    stage = root / "fresh_retrieve_gpu_v2"
    m = read(stage / "manifest.json")
    assert read(stage / "gpu_pipeline_status.json")["status"] == "COMPLETE"
    registry = Path(m["registry_root"])
    assert registry.resolve().is_relative_to(root)
    assert sha(registry / "manifest.json") == m["registry_manifest_sha256"]
    reg = read(registry / "manifest.json")
    base, dependency = read(a.input_backup / "backup_status.json"), read(a.input_backup / "backup_manifest.json")
    assert base["status"] == "COMPLETE" and sha(base["archive"]) == base["archive_sha256"]
    for name, key in (("fresh_v1/manifest.json", "baseline_manifest_sha256"),
                      ("fresh_causal_v1/code_manifest.json", "causal_code_manifest_sha256")):
        assert dependency["files"][name]["sha256"] == reg[key]
    for name, expected in m["entrypoints_sha256"].items():
        assert sha(stage / "code" / name) == expected, name
    assert sha(stage / "protocol.json") == m["protocol_sha256"]
    assert sha(stage / "jobs.json") == m["jobs_sha256"]
    jobs = read(stage / "jobs.json")["jobs"]
    for job in jobs:
        folder = Path(job["output"])
        if job["action"] == "banks":
            plan = read(folder / "plan.json")
            assert plan["status"] == "FROZEN_BEFORE_BEHAVIOR" and not plan["uses_confirmation_outcomes"]
            assert sha(folder / "banks.json") == plan["banks_sha256"]
        else:
            state = read(folder / "status.json")
            assert state["status"] == "COMPLETE" and state["completed_rows"] == state["expected_rows"]
            if job["expected_rows"] is not None:
                assert state["completed_rows"] == job["expected_rows"]
            for key, name in (("trials_sha256", "trials.jsonl"), ("observations_sha256", "observations.jsonl"), ("ranking_sha256", "ranking.json")):
                if key in state:
                    assert sha(folder / name) == state[key], str(folder / name)
    files = {}
    for folder in (stage, registry):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                assert not path.is_symlink() and path.resolve().is_relative_to(root)
                files[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    inventory = {"schema": "enumeration_retrieve_incremental_backup_v1", "utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root), "files": files, "formal_rows": m["expected_formal_rows"],
        "localization_queries": m["expected_localization_queries"],
        "dependency": {"archive": base["archive"], "archive_sha256": base["archive_sha256"], "manifest_sha256": base["manifest_sha256"]},
        "script_sha256": sha(__file__), "helper_sha256": sha(Path(__file__).with_name("backup_enumeration_fresh_answer_patch.py")),
        "scope": "Complete frozen Retrieve code, input registries, localization observations/rankings, banks, all smoke and formal behavior outputs; statistical audit pending", "command": sys.argv}
    a.output.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(inventory, indent=2) + "\n").encode()
    (a.output / "backup_manifest.json").write_bytes(raw)
    archive = a.output / "retrieve_completed.tar.gz"
    with tarfile.open(archive, "x:gz", compresslevel=3) as tar:
        info = tarfile.TarInfo("backup_manifest.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
        for name, item in files.items():
            tar.add(root / name, arcname=name, recursive=False)
            assert sha(root / name) == item["sha256"], name
    result = {"status": "COMPLETE", "archive": str(archive), "archive_sha256": sha(archive),
              "archive_bytes": archive.stat().st_size, "files": len(files),
              "formal_rows": m["expected_formal_rows"], "localization_queries": m["expected_localization_queries"],
              "manifest_sha256": sha(a.output / "backup_manifest.json")}
    (a.output / "backup_status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
