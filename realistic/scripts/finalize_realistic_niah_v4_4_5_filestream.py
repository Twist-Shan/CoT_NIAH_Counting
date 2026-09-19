from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_SHA256 = "da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"
RETRIEVAL_LABELS = (
    "Qwen_L21",
    "Qwen_L23",
    "Qwen_L24",
    "Qwen_L26",
    "Qwen_L27",
    "Gemma_L29",
    "Gemma_L35",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pid_is_original_rsync(pid: int, source: Path, destination: Path) -> bool:
    command_path = Path(f"/proc/{pid}/cmdline")
    try:
        command = command_path.read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        )
    except FileNotFoundError:
        return False
    return (
        "rsync" in command
        and str(source) in command
        and str(destination) in command
    )


def run_rsync(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def rsync_dry_run(source: Path, destination: Path, output_path: Path) -> str:
    completed = subprocess.run(
        [
            "rsync",
            "-aHn",
            "--delete",
            "--itemize-changes",
            "--out-format=%i %n%L",
            f"{source}/",
            f"{destination}/",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_path.write_text(completed.stdout, encoding="utf-8")
    changes = [line for line in completed.stdout.splitlines() if line.strip()]
    if changes:
        raise RuntimeError(f"Filestream dry-run found {len(changes)} differences")
    return completed.stdout


def inventory(root: Path) -> dict[str, Any]:
    file_count = 0
    file_bytes = 0
    symlink_count = 0
    symlink_digest = hashlib.sha256()
    inode_paths: dict[tuple[int, int], list[str]] = defaultdict(list)
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(directory_names + file_names):
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                symlink_count += 1
                symlink_digest.update(relative.encode("utf-8"))
                symlink_digest.update(b"\0")
                symlink_digest.update(os.readlink(path).encode("utf-8"))
                symlink_digest.update(b"\n")
                continue
            if not path.is_file():
                continue
            stat = path.stat()
            file_count += 1
            file_bytes += int(stat.st_size)
            if stat.st_nlink > 1:
                inode_paths[(int(stat.st_dev), int(stat.st_ino))].append(relative)

    hardlink_groups = [sorted(paths) for paths in inode_paths.values() if len(paths) > 1]
    hardlink_groups.sort()
    hardlink_digest = hashlib.sha256()
    for paths in hardlink_groups:
        hardlink_digest.update("\0".join(paths).encode("utf-8"))
        hardlink_digest.update(b"\n")
    return {
        "file_count": file_count,
        "file_bytes": file_bytes,
        "symlink_count": symlink_count,
        "symlink_topology_sha256": symlink_digest.hexdigest(),
        "hardlink_group_count": len(hardlink_groups),
        "hardlink_topology_sha256": hardlink_digest.hexdigest(),
    }


def load_pass(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"Audit is not PASS: {path}: {payload}")
    return payload


def audit_science_outputs(destination: Path, expected_sha: str) -> dict[str, Any]:
    stimulus = (
        destination
        / "dataset"
        / "canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl"
    )
    actual_sha = sha256(stimulus)
    if actual_sha != expected_sha:
        raise RuntimeError(f"Destination stimulus hash mismatch: {actual_sha}")

    merge = load_pass(destination / "canonical_merged" / "merge_audit.json")
    expected_detail = {"Qwen3-8B": 33300, "Gemma4-E4B": 38700}
    for model, rows in expected_detail.items():
        if int(merge["models"][model]["detail_rows"]) != rows:
            raise RuntimeError(f"Destination merge row mismatch for {model}")

    span = load_pass(destination / "analysis" / "span_restoration" / "analysis_audit.json")
    if int(span.get("detail_rows", -1)) != 72000 or int(span.get("restoration_rows", -1)) != 70200:
        raise RuntimeError("Destination span-analysis row audit failed")

    answer = load_pass(destination / "analysis" / "answer_geometry" / "analysis_audit.json")
    if int(answer.get("state_rows", -1)) != 23400:
        raise RuntimeError("Destination answer-geometry row audit failed")

    retrieval = load_pass(destination / "analysis" / "retrieval_geometry" / "geometry_audit.json")
    if int(retrieval.get("rows", -1)) != 3000:
        raise RuntimeError("Destination retrieval-geometry row audit failed")

    gpu = load_pass(destination / "retrieval_subspace_8gpu_complete.json")
    if int(gpu.get("total_rows", -1)) != 2800:
        raise RuntimeError("Destination retrieval-subspace combined audit failed")

    layer_audits = {}
    for label in RETRIEVAL_LABELS:
        layer = load_pass(
            destination / "analysis" / "retrieval_subspace" / label / "analysis_audit.json"
        )
        if int(layer.get("rows", -1)) != 400 or int(layer.get("paired_units", -1)) != 100:
            raise RuntimeError(f"Destination retrieval-subspace audit failed for {label}")
        layer_audits[label] = layer["status"]

    cross_layer = load_pass(
        destination
        / "analysis"
        / "retrieval_subspace_cross_layer"
        / "cross_layer_audit.json"
    )
    if int(cross_layer.get("primary_rows", -1)) != 7:
        raise RuntimeError("Destination cross-layer audit failed")

    return {
        "stimulus_sha256": actual_sha,
        "merge_status": merge["status"],
        "span_status": span["status"],
        "answer_geometry_status": answer["status"],
        "retrieval_geometry_status": retrieval["status"],
        "retrieval_subspace_status": gpu["status"],
        "retrieval_subspace_layer_statuses": layer_audits,
        "cross_layer_status": cross_layer["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--wait-pids", nargs="*", type=int, default=[])
    parser.add_argument("--expected-stimulus-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    lock_path = source / "locks" / "filestream_finalize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("A Filestream finalizer already holds the lock") from error

    while any(
        pid_is_original_rsync(pid, source, destination) for pid in args.wait_pids
    ):
        time.sleep(args.poll_seconds)

    incremental_log = Path("/tmp/nonthinking_v445_filestream_incremental.log")
    run_rsync(
        [
            "rsync",
            "-aH",
            "--partial",
            "--info=progress2",
            f"{source}/",
            f"{destination}/",
        ],
        incremental_log,
    )
    source_log = source / "logs" / "filestream_incremental_rsync.log"
    destination_log = destination / "logs" / "filestream_incremental_rsync.log"
    source_log.parent.mkdir(parents=True, exist_ok=True)
    destination_log.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(incremental_log, source_log)
    shutil.copy2(incremental_log, destination_log)
    temporary_dry_run = Path("/tmp/nonthinking_v445_filestream_rsync_dry_run.txt")
    rsync_dry_run(source, destination, temporary_dry_run)
    dry_run_path = source / "analysis" / "filestream_copy" / "rsync_dry_run.txt"
    destination_dry_run_path = (
        destination / "analysis" / "filestream_copy" / "rsync_dry_run.txt"
    )
    dry_run_path.parent.mkdir(parents=True, exist_ok=True)
    destination_dry_run_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(temporary_dry_run, dry_run_path)
    shutil.copy2(temporary_dry_run, destination_dry_run_path)

    source_inventory = inventory(source)
    destination_inventory = inventory(destination)
    if source_inventory != destination_inventory:
        raise RuntimeError(
            "Source/destination inventory mismatch: "
            f"source={source_inventory}, destination={destination_inventory}"
        )
    science = audit_science_outputs(destination, args.expected_stimulus_sha256)

    payload = {
        "schema_version": "realistic_niah_v4_4_5_filestream_copy_audit_v1",
        "status": "PASS",
        "source": str(source),
        "destination": str(destination),
        "source_inventory_before_audit_file": source_inventory,
        "destination_inventory_before_audit_file": destination_inventory,
        "rsync_dry_run_changes": 0,
        "science_audits": science,
    }
    source_audit = source / "analysis" / "filestream_copy" / "filestream_copy_audit.json"
    destination_audit = (
        destination / "analysis" / "filestream_copy" / "filestream_copy_audit.json"
    )
    atomic_json(source_audit, payload)
    destination_audit.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_audit, destination_audit)

    final_dry_run = source / "analysis" / "filestream_copy" / "rsync_final_dry_run.txt"
    final_dry_run.write_text("", encoding="utf-8")
    destination_final_dry_run = (
        destination / "analysis" / "filestream_copy" / "rsync_final_dry_run.txt"
    )
    destination_final_dry_run.write_text("", encoding="utf-8")
    rsync_dry_run(source, destination, final_dry_run)
    shutil.copy2(final_dry_run, destination_final_dry_run)

    audit_sha = sha256(destination_audit)
    marker = destination / ".FILESTREAM_COPY_COMPLETE"
    marker.write_text(
        json.dumps(
            {"status": "PASS", "filestream_copy_audit_sha256": audit_sha},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "marker": str(marker), "audit_sha256": audit_sha}))


if __name__ == "__main__":
    main()
