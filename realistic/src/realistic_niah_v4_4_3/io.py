from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch

from .spec import V443Config


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_bytes(destination, encoded)
    return destination


def atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    payload = b"".join(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    _atomic_bytes(destination, payload)
    return destination


def atomic_csv_gzip(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    _atomic_bytes(destination, gzip.compress(csv_bytes, mtime=0))
    return destination


def atomic_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    _atomic_bytes(destination, text.encode("utf-8"))
    return destination


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".tmp.", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def validate_filestream_isolation(
    *,
    source_run_root: str | Path,
    output_namespace_root: str | Path,
    run_root: str | Path,
) -> tuple[Path, Path, Path]:
    source = Path(source_run_root).resolve()
    namespace = Path(output_namespace_root).resolve()
    run = Path(run_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"V4.4 source run is missing: {source}")
    try:
        run.relative_to(namespace)
    except ValueError as error:
        raise ValueError(
            f"V4.4.3 run root must be below its namespace: {namespace}"
        ) from error
    if run == namespace:
        raise ValueError("V4.4.3 run root must include a unique run_id")
    for left, right, label in (
        (source, namespace, "source/output namespace"),
        (source, run, "source/run root"),
    ):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError(f"Overlapping {label} violates filestream isolation")
    if namespace.name != "v4_4_3_ov_causal":
        raise ValueError(
            "The shared output namespace must end in v4_4_3_ov_causal"
        )
    return source, namespace, run


def initialize_isolated_run(
    *,
    source_run_root: str | Path,
    output_namespace_root: str | Path,
    run_root: str | Path,
    config: V443Config,
    resume: bool,
    repo_commit: str | None = None,
) -> Path:
    config.validate()
    source, namespace, run = validate_filestream_isolation(
        source_run_root=source_run_root,
        output_namespace_root=output_namespace_root,
        run_root=run_root,
    )
    namespace.mkdir(parents=True, exist_ok=True)
    config_path = run / "resolved_config.json"
    if run.exists():
        if not resume:
            raise FileExistsError(
                f"V4.4.3 run root already exists; choose another run_id: {run}"
            )
        if not config_path.is_file():
            raise RuntimeError("Existing run has no resolved_config.json")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        normalized_config = json.loads(
            json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True)
        )
        if existing != normalized_config:
            raise RuntimeError("Refusing to resume with a different frozen config")
        return run
    run.mkdir()
    atomic_json(config_path, config.to_dict())
    owner = {
        "schema_version": "realistic_niah_v4_4_3_owner_v1",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_unix": time.time(),
        "python": platform.python_version(),
        "source_run_root": str(source),
        "output_namespace_root": str(namespace),
        "run_root": str(run),
        "repo_commit": repo_commit,
        "write_contract": {
            "source_run_root": "read_only",
            "run_root": "exclusive_v4_4_3_writer",
            "raw_attention_rows": False,
            "full_hidden_states": False,
            "shards": "temporary_then_os_replace",
        },
    }
    atomic_json(run / "owner.json", owner)
    return run


def source_input_manifest(source_run_root: str | Path) -> dict[str, Any]:
    source = Path(source_run_root).resolve()
    required = {
        "stimuli": source / "dataset" / "stimuli.jsonl",
        "dataset_manifest": source / "dataset" / "manifest.json",
        "dataset_checksums": source / "dataset" / "SHA256SUMS",
        "numeric_representation_complete": source / "numeric_representation.complete",
        "answer_query_all_layers_complete": source
        / "answer_query_all_layers_v1.complete",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete V4.4 source run; missing={missing}")
    files = {}
    for name, path in required.items():
        stat = path.stat()
        files[name] = {
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "realistic_niah_v4_4_3_input_manifest_v1",
        "source_access": "read_only",
        "source_run_root": str(source),
        "files": files,
    }


def stage_root(run_root: str | Path, model_label: str, stage: str) -> Path:
    if not model_label or "/" in model_label or "\\" in model_label:
        raise ValueError("Unsafe model label")
    if not stage or "/" in stage or "\\" in stage:
        raise ValueError("Unsafe stage label")
    root = Path(run_root) / "models" / model_label / stage
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_stage_status(
    run_root: str | Path,
    *,
    model_label: str,
    stage: str,
    state: str,
    detail: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "schema_version": "realistic_niah_v4_4_3_stage_status_v1",
        "model_label": model_label,
        "stage": stage,
        "state": state,
        "updated_unix": time.time(),
        "detail": dict(detail or {}),
    }
    return atomic_json(stage_root(run_root, model_label, stage) / "status.json", payload)
