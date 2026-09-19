from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from realistic_niah.runner import (
    _normalized_manifest_engine,
    _ordered_id_digest,
    model_engine_overrides,
)
from realistic_niah.spec import QUERY_LAYOUT

from .engine import formal_engine_config
from .sharding import expected_request_ids, formal_shard_plan
from .spec import MODEL_SPECS, PROTOCOL_VERSION, V31_RUN_PROTOCOL


_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def parse_resume_commits(raw: str) -> set[str]:
    if not raw:
        return set()
    commits = raw.split(":")
    if any(_COMMIT_RE.fullmatch(commit) is None for commit in commits):
        raise ValueError("Resume commits must be colon-separated full Git SHAs")
    return set(commits)


def manifest_resume_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "protocol_version": manifest.get("protocol_version"),
        "model_id": manifest.get("model", {}).get("model_id"),
        "query_layout": manifest.get("query_layout"),
        "stimuli_sha256": manifest.get("stimuli_sha256"),
        "request_ids_sha256": manifest.get("request_ids_sha256"),
        "engine": _normalized_manifest_engine(manifest.get("engine")),
        "model_engine_overrides": manifest.get("model_engine_overrides", {}),
        "prompt_payload_storage": manifest.get("prompt_payload_storage", "full"),
        "checkpoint_strategy": manifest.get(
            "checkpoint_strategy",
            "legacy_full_file_rewrite",
        ),
    }


def expected_resume_signature(
    task: dict[str, Any],
    *,
    stimuli_sha256: str,
    request_ids_sha256: str,
) -> dict[str, Any]:
    model_label = str(task["model_label"])
    return {
        "schema_version": V31_RUN_PROTOCOL.run_manifest_schema_version,
        "protocol_version": PROTOCOL_VERSION,
        "model_id": str(task["model_id"]),
        "query_layout": QUERY_LAYOUT,
        "stimuli_sha256": stimuli_sha256,
        "request_ids_sha256": request_ids_sha256,
        "engine": asdict(formal_engine_config(model_label)),
        "model_engine_overrides": model_engine_overrides(MODEL_SPECS[model_label]),
        "prompt_payload_storage": "sha256_only",
        "checkpoint_strategy": "atomic_batch_parts_then_single_canonical_merge",
    }


def validate_resume_manifest(
    manifest: dict[str, Any],
    task: dict[str, Any],
    *,
    stimuli_sha256: str,
    request_ids_sha256: str,
    current_commit: str,
    allowed_commits: set[str],
) -> str:
    existing = manifest_resume_signature(manifest)
    expected = expected_resume_signature(
        task,
        stimuli_sha256=stimuli_sha256,
        request_ids_sha256=request_ids_sha256,
    )
    if existing != expected:
        raise RuntimeError(
            "Existing V3.1 manifest is incompatible with the frozen run: "
            f"task={task['task_id']}, existing={existing}, expected={expected}"
        )
    if (
        manifest.get("model", {}).get("label") != task["model_label"]
        or manifest.get("model_revision") != task["model_revision"]
        or manifest.get("prompt_modes") != [task["prompt_mode"]]
        or manifest.get("git", {}).get("dirty") is not False
    ):
        raise RuntimeError(
            f"Existing V3.1 manifest provenance is incompatible: {task['task_id']}"
        )
    manifest_commit = str(manifest.get("git", {}).get("commit", ""))
    if manifest_commit not in {current_commit, *allowed_commits}:
        raise RuntimeError(
            "Existing V3.1 manifest commit is not explicitly authorized: "
            f"task={task['task_id']}, commit={manifest_commit!r}"
        )
    return manifest_commit


def audit_resume_manifests(
    run_root: str | Path,
    *,
    current_commit: str,
    allowed_commits: set[str],
) -> dict[str, Any]:
    if _COMMIT_RE.fullmatch(current_commit) is None:
        raise ValueError("Current commit must be a full lowercase Git SHA")
    invalid_allowed = sorted(
        commit for commit in allowed_commits if _COMMIT_RE.fullmatch(commit) is None
    )
    if invalid_allowed:
        raise ValueError(
            f"Allowed resume commits must be full lowercase Git SHAs: {invalid_allowed}"
        )
    root = Path(run_root).resolve()
    stimuli_path = root / "dataset" / "stimuli.jsonl"
    stimuli_bytes = stimuli_path.read_bytes()
    stimuli_sha256 = hashlib.sha256(stimuli_bytes).hexdigest()
    stimulus_ids = tuple(
        str(json.loads(line)["stimulus_id"])
        for line in stimuli_bytes.decode("utf-8").splitlines()
        if line
    )
    tasks = formal_shard_plan()["tasks"]
    expected_paths = {
        root / "shards" / str(task["task_id"]) / "main" / "run_manifest.json": task
        for task in tasks
    }
    actual_paths = set(root.glob("shards/*/main/run_manifest.json"))
    extra_paths = actual_paths - set(expected_paths)
    if extra_paths:
        raise RuntimeError(
            f"Unexpected V3.1 shard manifests: {sorted(str(path) for path in extra_paths)}"
        )
    observed_commits: set[str] = set()
    for path in sorted(actual_paths):
        task = expected_paths[path]
        request_ids_sha256 = _ordered_id_digest(
            expected_request_ids(stimulus_ids, task)
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        observed_commits.add(
            validate_resume_manifest(
                manifest,
                task,
                stimuli_sha256=stimuli_sha256,
                request_ids_sha256=request_ids_sha256,
                current_commit=current_commit,
                allowed_commits=allowed_commits,
            )
        )
    return {
        "passed": True,
        "manifests": len(actual_paths),
        "current_commit": current_commit,
        "allowed_commits": sorted(allowed_commits),
        "observed_commits": sorted(observed_commits),
    }
