from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from realistic_niah.runner import (
    LoadedVLLMRuntime,
    load_vllm_runtime,
    request_id,
    run_vllm_experiment,
)
from realistic_niah.spec import QUERY_LAYOUT
from realistic_niah_v3_1.spec import MODEL_SPECS

from .engine import formal_engine_config
from .integrity import sha256_file
from .spec import (
    EXPECTED_REQUESTS,
    EXPECTED_REQUESTS_PER_MODEL,
    FORMAL_PROMPT_MODES,
    MODEL_CONTEXT_ENGINE_OVERRIDES,
    MODEL_IDS,
    MODEL_LABELS,
    MODEL_REVISIONS,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    PREFLIGHT_LENGTH,
    PREFLIGHT_NEEDLE_COUNT,
    PREFLIGHT_SEED,
    PROTOCOL_VERSION,
    SEEDS,
    V33_LONG_CONTEXT_RUN_PROTOCOL,
    WORKER_LENGTHS,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected object in {path}:{line_number}")
            rows.append(value)
    return rows


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_is_dirty(repo_root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def run_long_context_experiment(
    *,
    model_label: str,
    stimuli_path: str | Path,
    output_dir: str | Path,
    passage_lengths: Iterable[int],
    needle_counts: Iterable[int] = NEEDLE_COUNTS,
    seeds: Iterable[int] = SEEDS,
    prompt_modes: Iterable[str] = FORMAL_PROMPT_MODES,
    cache_dir: str | Path | None = None,
    repo_root: str | Path = ".",
    require_clean_git: bool = False,
    loaded_runtime: LoadedVLLMRuntime | None = None,
) -> dict[str, Any]:
    if model_label not in MODEL_LABELS:
        raise ValueError(f"Unregistered long-context model: {model_label}")
    lengths = tuple(int(value) for value in passage_lengths)
    counts = tuple(int(value) for value in needle_counts)
    paired_seeds = tuple(int(value) for value in seeds)
    modes = tuple(str(value) for value in prompt_modes)
    if not lengths or not set(lengths).issubset(PASSAGE_LENGTHS):
        raise ValueError("Requested lengths fall outside the registered grid")
    if not counts or not set(counts).issubset(NEEDLE_COUNTS):
        raise ValueError("Requested needle counts fall outside the registered grid")
    if not paired_seeds or not set(paired_seeds).issubset(SEEDS):
        raise ValueError("Requested seeds fall outside the registered paired seeds")
    if not modes or not set(modes).issubset(FORMAL_PROMPT_MODES):
        raise ValueError("Only direct and native-thinking modes are registered")
    return run_vllm_experiment(
        stimuli_path=stimuli_path,
        output_dir=output_dir,
        model=model_label,
        revision=MODEL_REVISIONS[model_label],
        passage_lengths=lengths,
        needle_counts=counts,
        seeds=paired_seeds,
        prompt_modes=modes,
        query_layout=QUERY_LAYOUT,
        engine_config=formal_engine_config(model_label),
        cache_dir=cache_dir,
        repo_root=repo_root,
        require_clean_git=require_clean_git,
        registered_model_spec=MODEL_SPECS[model_label],
        protocol=V33_LONG_CONTEXT_RUN_PROTOCOL,
        loaded_runtime=loaded_runtime,
        additional_engine_overrides=MODEL_CONTEXT_ENGINE_OVERRIDES[model_label],
    )


def _expected_request_ids(
    stimuli_path: str | Path,
    *,
    model_label: str,
    lengths: Iterable[int],
    modes: Iterable[str] = FORMAL_PROMPT_MODES,
) -> set[str]:
    selected_lengths = set(int(value) for value in lengths)
    selected_modes = tuple(str(value) for value in modes)
    if model_label not in MODEL_LABELS:
        raise ValueError(f"Unregistered long-context model: {model_label}")
    model_spec = MODEL_SPECS[model_label]
    expected: set[str] = set()
    cells: dict[tuple[int, int], set[int]] = {}
    with Path(stimuli_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            length = int(row["target_passage_tokens"])
            if length not in selected_lengths:
                continue
            count = int(row["num_needles"])
            seed = int(row["seed"])
            if count not in NEEDLE_COUNTS or seed not in SEEDS:
                raise RuntimeError(
                    f"Unregistered grid value in stimuli at line {line_number}"
                )
            cells.setdefault((length, count), set()).add(seed)
            stimulus_id = str(row["stimulus_id"])
            for mode in selected_modes:
                identifier = request_id(
                    model_spec=model_spec,
                    prompt_mode=mode,
                    query_layout=QUERY_LAYOUT,
                    stimulus_id=stimulus_id,
                    namespace=V33_LONG_CONTEXT_RUN_PROTOCOL.request_id_namespace,
                )
                if identifier in expected:
                    raise RuntimeError(f"Duplicate expected request ID: {identifier}")
                expected.add(identifier)
    expected_cells = {
        (length, count) for length in selected_lengths for count in NEEDLE_COUNTS
    }
    if set(cells) != expected_cells or any(
        values != set(SEEDS) for values in cells.values()
    ):
        raise RuntimeError("Frozen stimuli do not contain the exact registered grid")
    expected_count = (
        len(selected_lengths) * len(NEEDLE_COUNTS) * len(SEEDS) * len(selected_modes)
    )
    if len(expected) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} request IDs, constructed {len(expected)}"
        )
    return expected


def validate_output(
    *,
    model_label: str,
    stimuli_path: str | Path,
    output_dir: str | Path,
    lengths: Iterable[int],
    modes: Iterable[str] = FORMAL_PROMPT_MODES,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    manifest_path = output / "run_manifest.json"
    requests_path = output / "requests.jsonl"
    if not manifest_path.is_file() or not requests_path.is_file():
        raise FileNotFoundError(f"Incomplete output directory: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_lengths = tuple(int(value) for value in lengths)
    selected_modes = tuple(str(value) for value in modes)
    expected_ids = _expected_request_ids(
        stimuli_path,
        model_label=model_label,
        lengths=selected_lengths,
        modes=selected_modes,
    )
    if (
        manifest.get("schema_version")
        != V33_LONG_CONTEXT_RUN_PROTOCOL.run_manifest_schema_version
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("model", {}).get("label") != model_label
        or manifest.get("model", {}).get("model_id") != MODEL_IDS[model_label]
        or manifest.get("model_revision") != MODEL_REVISIONS[model_label]
        or manifest.get("query_layout") != QUERY_LAYOUT
        or tuple(manifest.get("prompt_modes", ())) != selected_modes
        or manifest.get("engine") != asdict(formal_engine_config(model_label))
        or manifest.get("model_engine_overrides")
        != MODEL_CONTEXT_ENGINE_OVERRIDES[model_label]
        or int(manifest.get("expected_requests", -1)) != len(expected_ids)
        or int(manifest.get("completed_requests", -1)) != len(expected_ids)
        or "completed_at_utc" not in manifest
    ):
        raise RuntimeError(f"Run manifest is incompatible or incomplete: {output}")
    rows = _read_jsonl(requests_path)
    observed_ids = [str(row.get("request_id")) for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError(f"Duplicate request IDs in {requests_path}")
    if set(observed_ids) != expected_ids:
        missing = sorted(expected_ids - set(observed_ids))[:3]
        extra = sorted(set(observed_ids) - expected_ids)[:3]
        raise RuntimeError(
            f"Request grid mismatch in {requests_path}: missing={missing}, extra={extra}"
        )
    allowed_lengths = set(selected_lengths)
    allowed_modes = set(selected_modes)
    for row in rows:
        if (
            row.get("schema_version")
            != V33_LONG_CONTEXT_RUN_PROTOCOL.request_schema_version
            or row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("model_label") != model_label
            or row.get("model_id") != MODEL_IDS[model_label]
            or row.get("model_revision") != MODEL_REVISIONS[model_label]
            or int(row.get("target_passage_tokens", -1)) not in allowed_lengths
            or int(row.get("num_needles", -1)) not in NEEDLE_COUNTS
            or int(row.get("seed", -1)) not in SEEDS
            or row.get("prompt_mode") not in allowed_modes
            or row.get("query_layout") != QUERY_LAYOUT
        ):
            raise RuntimeError(f"Incompatible request row: {row.get('request_id')}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "output_dir": str(output),
        "lengths": list(selected_lengths),
        "modes": list(selected_modes),
        "requests": len(rows),
        "requests_sha256": sha256_file(requests_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _preflight_paths(run_root: Path, model_label: str) -> tuple[Path, Path]:
    if model_label not in MODEL_LABELS:
        raise ValueError(f"Unregistered long-context model: {model_label}")
    output = run_root / "preflight" / model_label / "main"
    return output, output / "_SUCCESS.json"


def _validate_preflight_subset(
    *,
    model_label: str,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = output_dir / "run_manifest.json"
    requests_path = output_dir / "requests.jsonl"
    if not manifest_path.is_file() or not requests_path.is_file():
        raise FileNotFoundError("Preflight output is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != V33_LONG_CONTEXT_RUN_PROTOCOL.run_manifest_schema_version
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("model", {}).get("label") != model_label
        or manifest.get("model", {}).get("model_id") != MODEL_IDS[model_label]
        or manifest.get("model_revision") != MODEL_REVISIONS[model_label]
        or manifest.get("engine") != asdict(formal_engine_config(model_label))
        or manifest.get("model_engine_overrides")
        != MODEL_CONTEXT_ENGINE_OVERRIDES[model_label]
        or tuple(manifest.get("prompt_modes", ())) != FORMAL_PROMPT_MODES
        or int(manifest.get("expected_requests", -1)) != 2
        or int(manifest.get("completed_requests", -1)) != 2
        or "completed_at_utc" not in manifest
    ):
        raise RuntimeError("Preflight manifest is incompatible or incomplete")
    rows = _read_jsonl(requests_path)
    observed = {
        (
            int(row.get("target_passage_tokens", -1)),
            int(row.get("num_needles", -1)),
            int(row.get("seed", -1)),
            str(row.get("prompt_mode")),
        )
        for row in rows
    }
    expected = {
        (PREFLIGHT_LENGTH, PREFLIGHT_NEEDLE_COUNT, PREFLIGHT_SEED, mode)
        for mode in FORMAL_PROMPT_MODES
    }
    if len(rows) != 2 or observed != expected:
        raise RuntimeError("Preflight request subset is incorrect")
    model_spec = MODEL_SPECS[model_label]
    stimulus_id = (
        f"V33LC_T{PREFLIGHT_LENGTH}_N{PREFLIGHT_NEEDLE_COUNT}_"
        f"seed{PREFLIGHT_SEED}"
    )
    expected_request_ids = {
        request_id(
            model_spec=model_spec,
            prompt_mode=mode,
            query_layout=QUERY_LAYOUT,
            stimulus_id=stimulus_id,
            namespace=V33_LONG_CONTEXT_RUN_PROTOCOL.request_id_namespace,
        )
        for mode in FORMAL_PROMPT_MODES
    }
    observed_request_ids = [str(row.get("request_id")) for row in rows]
    if (
        len(observed_request_ids) != len(set(observed_request_ids))
        or set(observed_request_ids) != expected_request_ids
    ):
        raise RuntimeError("Preflight request IDs are incorrect")
    if any(
        row.get("schema_version")
        != V33_LONG_CONTEXT_RUN_PROTOCOL.request_schema_version
        or row.get("protocol_version") != PROTOCOL_VERSION
        or row.get("model_label") != model_label
        or row.get("model_id") != MODEL_IDS[model_label]
        or row.get("model_revision") != MODEL_REVISIONS[model_label]
        or row.get("query_layout") != QUERY_LAYOUT
        for row in rows
    ):
        raise RuntimeError("Preflight request metadata is incorrect")
    return {
        "requests": 2,
        "manifest_sha256": sha256_file(manifest_path),
        "requests_sha256": sha256_file(requests_path),
    }


def validate_preflight(
    *,
    model_label: str,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    output, marker = _preflight_paths(root, model_label)
    audit = _validate_preflight_subset(
        model_label=model_label,
        output_dir=output,
    )
    if not marker.is_file():
        raise FileNotFoundError(f"Missing preflight success marker: {marker}")
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if saved.get("manifest_sha256") != audit["manifest_sha256"]:
        raise RuntimeError("Preflight success marker does not match its manifest")
    return audit


def wait_for_preflight(
    *,
    model_label: str,
    run_root: str | Path,
    timeout_seconds: float = 2700.0,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    output, marker = _preflight_paths(root, model_label)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if marker.is_file():
            audit = _validate_preflight_subset(
                model_label=model_label,
                output_dir=output,
            )
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("manifest_sha256") != audit["manifest_sha256"]:
                raise RuntimeError("Preflight success marker is stale")
            return audit
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for preflight marker: {marker}")


def run_worker(
    *,
    model_label: str,
    worker_index: int,
    stimuli_path: str | Path,
    run_root: str | Path,
    cache_dir: str | Path | None,
    repo_root: str | Path,
    preflight_timeout_seconds: float = 2700.0,
) -> dict[str, Any]:
    if model_label not in MODEL_LABELS:
        raise ValueError(f"Unregistered long-context model: {model_label}")
    if worker_index not in WORKER_LENGTHS:
        raise ValueError(f"Unknown worker index: {worker_index}")
    root = Path(run_root).resolve()
    repo = Path(repo_root).resolve()
    if _git_is_dirty(repo):
        raise RuntimeError("Formal V3.3 long-context run requires a clean Git worktree")
    commit = _git_commit(repo)
    engine = formal_engine_config(model_label)

    if worker_index == 0:
        runtime = load_vllm_runtime(
            model_spec=MODEL_SPECS[model_label],
            revision=MODEL_REVISIONS[model_label],
            engine_config=engine,
            cache_dir=cache_dir,
            additional_engine_overrides=(
                MODEL_CONTEXT_ENGINE_OVERRIDES[model_label]
            ),
        )
        preflight_output, preflight_marker = _preflight_paths(root, model_label)
        run_long_context_experiment(
            model_label=model_label,
            stimuli_path=stimuli_path,
            output_dir=preflight_output,
            passage_lengths=(PREFLIGHT_LENGTH,),
            needle_counts=(PREFLIGHT_NEEDLE_COUNT,),
            seeds=(PREFLIGHT_SEED,),
            prompt_modes=FORMAL_PROMPT_MODES,
            cache_dir=cache_dir,
            repo_root=repo,
            require_clean_git=True,
            loaded_runtime=runtime,
        )
        preflight_audit = _validate_preflight_subset(
            model_label=model_label,
            output_dir=preflight_output,
        )
        _atomic_json(
            preflight_marker,
            {
                "schema_version": "realistic_niah_preflight_success_v3_3_long_context",
                "protocol_version": PROTOCOL_VERSION,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit,
                "model_label": model_label,
                **preflight_audit,
            },
        )
        print(
            f"V33_LONG_CONTEXT_100K_PREFLIGHT_OK model={model_label}",
            flush=True,
        )
    else:
        preflight_audit = wait_for_preflight(
            model_label=model_label,
            run_root=root,
            timeout_seconds=preflight_timeout_seconds,
        )
        runtime = load_vllm_runtime(
            model_spec=MODEL_SPECS[model_label],
            revision=MODEL_REVISIONS[model_label],
            engine_config=engine,
            cache_dir=cache_dir,
            additional_engine_overrides=(
                MODEL_CONTEXT_ENGINE_OVERRIDES[model_label]
            ),
        )

    lengths = WORKER_LENGTHS[worker_index]
    output = root / "formal" / model_label / f"worker-{worker_index}" / "main"
    manifest = run_long_context_experiment(
        model_label=model_label,
        stimuli_path=stimuli_path,
        output_dir=output,
        passage_lengths=lengths,
        cache_dir=cache_dir,
        repo_root=repo,
        require_clean_git=True,
        loaded_runtime=runtime,
    )
    audit = validate_output(
        model_label=model_label,
        stimuli_path=stimuli_path,
        output_dir=output,
        lengths=lengths,
    )
    marker = output / "_SUCCESS.json"
    _atomic_json(
        marker,
        {
            "schema_version": "realistic_niah_worker_success_v3_3_long_context",
            "protocol_version": PROTOCOL_VERSION,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "model_label": model_label,
            "worker_index": worker_index,
            "preflight_manifest_sha256": preflight_audit["manifest_sha256"],
            **audit,
        },
    )
    return {
        "model_label": model_label,
        "worker_index": worker_index,
        "lengths": list(lengths),
        "completed_requests": manifest["completed_requests"],
        "success_marker": str(marker),
    }


def required_worker_markers(run_root: str | Path) -> tuple[Path, ...]:
    root = Path(run_root).resolve()
    return tuple(
        root
        / "formal"
        / model_label
        / f"worker-{worker_index}"
        / "main"
        / "_SUCCESS.json"
        for model_label in MODEL_LABELS
        for worker_index in WORKER_LENGTHS
    )


def finalize_run(
    *,
    stimuli_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    audits: list[dict[str, Any]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    for model_label in MODEL_LABELS:
        model_request_count = 0
        for worker_index, lengths in WORKER_LENGTHS.items():
            output = (
                root
                / "formal"
                / model_label
                / f"worker-{worker_index}"
                / "main"
            )
            marker = output / "_SUCCESS.json"
            if not marker.is_file():
                raise FileNotFoundError(f"Missing worker success marker: {marker}")
            audit = validate_output(
                model_label=model_label,
                stimuli_path=stimuli_path,
                output_dir=output,
                lengths=lengths,
            )
            saved_marker = json.loads(marker.read_text(encoding="utf-8"))
            if saved_marker.get("manifest_sha256") != audit["manifest_sha256"]:
                raise RuntimeError(
                    f"{model_label} worker {worker_index} success marker is stale"
                )
            audits.append(audit)
            model_request_count += int(audit["requests"])
            for row in _read_jsonl(output / "requests.jsonl"):
                identifier = str(row["request_id"])
                if identifier in rows_by_id:
                    raise RuntimeError(
                        f"Duplicate request across workers: {identifier}"
                    )
                rows_by_id[identifier] = row
        if model_request_count != EXPECTED_REQUESTS_PER_MODEL:
            raise RuntimeError(
                f"{model_label} has {model_request_count} requests, expected "
                f"{EXPECTED_REQUESTS_PER_MODEL}"
            )

    expected_ids: set[str] = set()
    for model_label in MODEL_LABELS:
        model_ids = _expected_request_ids(
            stimuli_path,
            model_label=model_label,
            lengths=PASSAGE_LENGTHS,
            modes=FORMAL_PROMPT_MODES,
        )
        if expected_ids.intersection(model_ids):
            raise RuntimeError("Model-specific request namespaces overlap")
        expected_ids.update(model_ids)
    if set(rows_by_id) != expected_ids or len(rows_by_id) != EXPECTED_REQUESTS:
        missing = sorted(expected_ids - set(rows_by_id))[:3]
        extra = sorted(set(rows_by_id) - expected_ids)[:3]
        raise RuntimeError(
            "Final request grid is incomplete: "
            f"observed={len(rows_by_id)}, missing={missing}, extra={extra}"
        )
    cells: dict[tuple[str, str, int, int], set[int]] = {}
    for row in rows_by_id.values():
        key = (
            str(row["model_label"]),
            str(row["prompt_mode"]),
            int(row["target_passage_tokens"]),
            int(row["num_needles"]),
        )
        cells.setdefault(key, set()).add(int(row["seed"]))
    expected_cells = {
        (model_label, mode, length, count)
        for model_label in MODEL_LABELS
        for mode in FORMAL_PROMPT_MODES
        for length in PASSAGE_LENGTHS
        for count in NEEDLE_COUNTS
    }
    if set(cells) != expected_cells or any(
        values != set(SEEDS) for values in cells.values()
    ):
        raise RuntimeError("Final cells do not contain all 30 paired seeds")

    results = root / "results"
    requests_path = results / "requests.jsonl"
    _atomic_jsonl(requests_path, (rows_by_id[key] for key in sorted(rows_by_id)))
    audit = {
        "schema_version": "realistic_niah_final_audit_v3_3_long_context",
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "models": [
            {
                "model_label": label,
                "model_id": MODEL_IDS[label],
                "model_revision": MODEL_REVISIONS[label],
                "requests": EXPECTED_REQUESTS_PER_MODEL,
            }
            for label in MODEL_LABELS
        ],
        "prompt_modes": list(FORMAL_PROMPT_MODES),
        "passage_lengths": list(PASSAGE_LENGTHS),
        "needle_counts": list(NEEDLE_COUNTS),
        "seeds": list(SEEDS),
        "cells": len(cells),
        "requests": len(rows_by_id),
        "unique_request_ids": len(rows_by_id),
        "requests_sha256": sha256_file(requests_path),
        "worker_audits": audits,
    }
    audit_path = results / "final_audit.json"
    _atomic_json(audit_path, audit)
    _atomic_json(
        results / "_SUCCESS.json",
        {
            "schema_version": "realistic_niah_final_success_v3_3_long_context",
            "protocol_version": PROTOCOL_VERSION,
            "final_audit_sha256": sha256_file(audit_path),
            "requests_sha256": audit["requests_sha256"],
            "requests": EXPECTED_REQUESTS,
        },
    )
    return audit


def finalize_if_ready(
    *,
    stimuli_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    missing = [
        str(path)
        for path in required_worker_markers(run_root)
        if not path.is_file()
    ]
    if missing:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "complete": False,
            "missing_worker_markers": missing,
        }
    audit = finalize_run(stimuli_path=stimuli_path, run_root=run_root)
    return {"complete": True, "audit": audit}
