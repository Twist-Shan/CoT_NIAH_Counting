from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .spec import EXPECTED_STIMULI, PROTOCOL_VERSION

DATASET_SEAL_FILENAME = "dataset_seal.json"
SEALED_DATASET_FILES = (
    "stimuli.jsonl",
    "manifest.json",
    "audit_report.json",
    "contamination_audit.json",
    "cell_counts.json",
    "SHA256SUMS",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def seal_frozen_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    dataset = Path(dataset_dir).resolve()
    with (dataset / "stimuli.jsonl").open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != EXPECTED_STIMULI:
        raise RuntimeError(
            f"Cannot seal {rows} stimuli; expected {EXPECTED_STIMULI}"
        )
    audit = json.loads((dataset / "audit_report.json").read_text(encoding="utf-8"))
    if (
        audit.get("passed") is not True
        or audit.get("protocol_version") != PROTOCOL_VERSION
        or int(audit.get("rows_checked", -1)) != EXPECTED_STIMULI
    ):
        raise RuntimeError("Cannot seal a dataset that has not passed the full audit")
    files: dict[str, dict[str, Any]] = {}
    for filename in SEALED_DATASET_FILES:
        path = dataset / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        files[filename] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    seal = {
        "schema_version": "realistic_niah_dataset_seal_v3_3_long_context",
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": EXPECTED_STIMULI,
        "files": files,
    }
    _atomic_json(dataset / DATASET_SEAL_FILENAME, seal)
    return seal


def validate_frozen_dataset(
    dataset_dir: str | Path,
    *,
    expected_seal_sha256: str | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_dir).resolve()
    seal_path = dataset / DATASET_SEAL_FILENAME
    if not seal_path.is_file():
        raise FileNotFoundError(f"Missing frozen dataset seal: {seal_path}")
    observed_seal_sha256 = sha256_file(seal_path)
    if expected_seal_sha256 is not None and (
        observed_seal_sha256 != expected_seal_sha256
    ):
        raise RuntimeError(
            "Dataset seal SHA256 mismatch: "
            f"{observed_seal_sha256} != {expected_seal_sha256}"
        )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("schema_version")
        != "realistic_niah_dataset_seal_v3_3_long_context"
        or seal.get("protocol_version") != PROTOCOL_VERSION
        or int(seal.get("rows", -1)) != EXPECTED_STIMULI
    ):
        raise RuntimeError("Frozen dataset seal metadata is invalid")
    files = seal.get("files")
    if not isinstance(files, dict) or set(files) != set(SEALED_DATASET_FILES):
        raise RuntimeError("Frozen dataset seal file list is invalid")
    observed: dict[str, dict[str, Any]] = {}
    for filename in SEALED_DATASET_FILES:
        path = dataset / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen dataset file: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        expected = files[filename]
        if size != int(expected["bytes"]) or digest != expected["sha256"]:
            raise RuntimeError(
                f"Frozen dataset mismatch for {filename}: "
                f"bytes={size}, sha256={digest}"
            )
        observed[filename] = {"bytes": size, "sha256": digest}
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((dataset / "audit_report.json").read_text(encoding="utf-8"))
    with (dataset / "stimuli.jsonl").open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if (
        rows != EXPECTED_STIMULI
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or int(manifest.get("rows", -1)) != EXPECTED_STIMULI
        or audit.get("protocol_version") != PROTOCOL_VERSION
        or audit.get("passed") is not True
        or int(audit.get("rows_checked", -1)) != EXPECTED_STIMULI
    ):
        raise RuntimeError("Frozen dataset manifest or audit metadata is invalid")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "rows": rows,
        "seal_path": str(seal_path),
        "seal_sha256": observed_seal_sha256,
        "files": observed,
    }
