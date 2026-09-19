from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchiveMetadata:
    source_dir: str
    archive_path: str
    size_bytes: int
    sha256: str
    md5: str
    created_at_utc: str


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_archive(
    source_dir: str | Path,
    archive_path: str | Path,
) -> ArchiveMetadata:
    source = Path(source_dir).resolve()
    destination = Path(archive_path).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(source, arcname=source.name)
    temporary.replace(destination)
    return ArchiveMetadata(
        source_dir=str(source),
        archive_path=str(destination),
        size_bytes=destination.stat().st_size,
        sha256=_digest(destination, "sha256"),
        md5=_digest(destination, "md5"),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def sync_archive_to_drive(
    metadata: ArchiveMetadata,
    *,
    remote_path: str,
    rclone_binary: str = "rclone",
) -> dict[str, Any]:
    if shutil.which(rclone_binary) is None:
        raise FileNotFoundError(
            f"{rclone_binary!r} is not installed or not on PATH"
        )
    archive = Path(metadata.archive_path)
    subprocess.run(
        [
            rclone_binary,
            "copyto",
            str(archive),
            remote_path,
            "--checksum",
            "--stats-one-line",
        ],
        check=True,
    )
    result = subprocess.run(
        [rclone_binary, "lsjson", remote_path, "--files-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote_items = json.loads(result.stdout)
    if len(remote_items) != 1:
        raise RuntimeError(f"Expected one remote archive, got {remote_items}")
    remote = remote_items[0]
    if int(remote["Size"]) != metadata.size_bytes:
        raise RuntimeError(
            f"Remote size mismatch: {remote['Size']} != {metadata.size_bytes}"
        )
    hashes = {str(k).lower(): str(v).lower() for k, v in remote.get("Hashes", {}).items()}
    if hashes.get("md5") and hashes["md5"] != metadata.md5:
        raise RuntimeError("Remote MD5 does not match the local archive")
    return {
        "verified": True,
        "remote_path": remote_path,
        "remote_size_bytes": int(remote["Size"]),
        "remote_hashes": hashes,
        "local": asdict(metadata),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def archive_and_sync_run(
    *,
    source_dir: str | Path,
    archive_dir: str | Path,
    remote_dir: str,
    rclone_binary: str = "rclone",
) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    archive = Path(archive_dir).resolve() / f"{source.name}.tar.gz"
    metadata = build_run_archive(source, archive)
    remote_path = f"{remote_dir.rstrip('/')}/{archive.name}"
    verification = sync_archive_to_drive(
        metadata,
        remote_path=remote_path,
        rclone_binary=rclone_binary,
    )
    manifest_path = archive.with_suffix(archive.suffix + ".sync.json")
    manifest_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verification
