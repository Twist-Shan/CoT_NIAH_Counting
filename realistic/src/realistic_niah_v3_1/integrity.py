from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .spec import EXPECTED_STIMULI, PROTOCOL_VERSION


# Anonymous provenance labels; the frozen bytes remain pinned by SHA256 below.
DATASET_ID = "anonymous-source/realistic-niah-count-empirical-law"
DATASET_REVISION = "anonymous-snapshot"
DATASET_SOURCE_FILENAME = "source_revision.json"
EXPECTED_DATASET_FILES = {
    "stimuli.jsonl": {
        "bytes": 184_690_729,
        "sha256": "afed18fe24d3c684b7f342a3c5cc119fe3bd4033487d25c97cbbe4fc21c0d159",
    },
    "manifest.json": {
        "bytes": 2_322,
        "sha256": "c2362bfcfa6c5242723a480d257f541b15c98fdf0adbfdcc1308eb6330bbf388",
    },
    "audit_report.json": {
        "bytes": 923,
        "sha256": "168bbec73cf11fd5b44cfcc66e10ded3f34a45caaee6c296650468ec1ccbf615",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def validate_frozen_dataset(
    dataset_dir: str | Path,
    *,
    require_source_revision: bool = True,
    record_source_revision: bool = False,
) -> dict[str, Any]:
    dataset = Path(dataset_dir).resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")

    observed: dict[str, dict[str, Any]] = {}
    for filename, expected in EXPECTED_DATASET_FILES.items():
        path = dataset / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen V3.1 dataset file: {path}")
        size = path.stat().st_size
        digest = _sha256(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise RuntimeError(
                f"Frozen V3.1 dataset mismatch for {filename}: "
                f"bytes={size}, sha256={digest}"
            )
        observed[filename] = {"bytes": size, "sha256": digest}

    stimuli_path = dataset / "stimuli.jsonl"
    with stimuli_path.open("rb") as handle:
        rows = sum(1 for _ in handle)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((dataset / "audit_report.json").read_text(encoding="utf-8"))
    if (
        rows != EXPECTED_STIMULI
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or audit.get("protocol_version") != PROTOCOL_VERSION
        or audit.get("passed") is not True
        or int(audit.get("rows_checked", -1)) != EXPECTED_STIMULI
    ):
        raise RuntimeError("The frozen V3.1 dataset metadata/audit is invalid")

    source_path = dataset / DATASET_SOURCE_FILENAME
    expected_source = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
    }
    if record_source_revision:
        _atomic_json(source_path, expected_source)
    if require_source_revision:
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing dataset source record: {source_path}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source != expected_source:
            raise RuntimeError(
                f"Frozen V3.1 dataset source mismatch: {source} != {expected_source}"
            )

    return {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "rows": rows,
        "files": observed,
        "source_record": str(source_path),
    }
