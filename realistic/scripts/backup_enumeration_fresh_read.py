"""Back up completed fresh Read trials and their immutable inputs; no GPU work."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tick = time.monotonic()
    root = args.root.resolve()
    causal = root / "fresh_causal_v1"
    baseline = root / "fresh_v1"
    paths = set()

    def add(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid backup input: {path}")
        if expected is not None and sha(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
        paths.add(path)

    for path in (baseline / "manifest.json", baseline / "protocol.json", causal / "code_manifest.json"):
        add(path)
    code = read(causal / "code_manifest.json")
    baseline_manifest = read(baseline / "manifest.json")
    add(baseline / "manifest.json", code["baseline_manifest_sha256"])
    add(baseline / "protocol.json", baseline_manifest["protocol_sha256"])
    for name, expected in code["original_code_sha256"].items():
        if name.startswith(("src/", "scripts/")):
            add(causal / "code" / name, expected)
    for name, expected in code["additive_code_sha256"].items():
        add(causal / "code" / name, expected)
    cells = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for mode in ("enumeration_index", "enumeration_bullet", "thinking"):
            registry = causal / "registries" / model / mode
            manifest = read(registry / "manifest.json")
            add(registry / "ledger.json", manifest["ledger_sha256"])
            add(registry / "adapted_generations.jsonl", manifest["adapted_generations_sha256"])
            add(registry / "manifest.json")
            add(registry / "status.json")
            for entry in read(registry / "ledger.json"):
                add(registry / entry["geometry_file"], entry["geometry_sha256"])
            original = baseline / "baseline" / model / mode
            original_status = read(original / "status.json")
            if original_status["status"] != "COMPLETE":
                raise ValueError(f"Incomplete baseline: {model}/{mode}")
            add(original / "generations.jsonl", manifest["baseline_generations_sha256"])
            add(original / "status.json")
            for stage in ("smoke", "formal"):
                folder = causal / "gpu_jobs" / stage / "read" / model / mode
                status = read(folder / "status.json")
                if status["status"] != "COMPLETE" or status["completed"] != status["total"]:
                    raise ValueError(f"Incomplete Read cell: {folder}")
                add(folder / "trials.jsonl", status["trials_sha256"])
                for name in ("status.json", "runtime.json", "coverage.json"):
                    add(folder / name)
                cells.append({"model": model, "mode": mode, "stage": stage, "rows": status["total"]})
    files = {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(paths)}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "enumeration_read_backup_v1", "utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(root), "cells": cells, "files": files, "command": sys.argv,
                "backup_script_sha256": sha(__file__), "inventory_seconds": time.monotonic() - tick,
                "scope": "Completed Read smoke/formal, all six frozen registries and baseline inputs, frozen src/scripts"}
    raw = (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode()
    (args.output / "backup_manifest.json").write_bytes(raw)
    archive = args.output / "read_completed.tar.gz"
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
              "source_bytes": sum(x["bytes"] for x in files.values()), "cells": cells,
              "manifest_sha256": sha(args.output / "backup_manifest.json"),
              "seconds": time.monotonic() - tick}
    (args.output / "backup_status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

