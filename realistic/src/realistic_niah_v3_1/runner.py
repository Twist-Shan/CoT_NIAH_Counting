from __future__ import annotations

import gc
import subprocess
from pathlib import Path
from typing import Any, Iterable

from realistic_niah.runner import (
    EngineConfig,
    LoadedVLLMRuntime,
    load_vllm_runtime,
    run_vllm_experiment,
)
from realistic_niah.spec import QUERY_LAYOUT

from .sharding import _task_id
from .spec import MODEL_REVISIONS, V31_RUN_PROTOCOL, resolve_model_spec


def run_v31_experiment(
    *,
    stimuli_path: str | Path,
    output_dir: str | Path,
    model: str,
    revision: str | None = None,
    passage_lengths: Iterable[int] | None = None,
    needle_counts: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    prompt_modes: Iterable[str] | None = None,
    query_layout: str = QUERY_LAYOUT,
    engine_config: EngineConfig | None = None,
    cache_dir: str | Path | None = None,
    repo_root: str | Path = ".",
    require_clean_git: bool = False,
    loaded_runtime: LoadedVLLMRuntime | None = None,
) -> dict[str, Any]:
    model_spec = resolve_model_spec(model)
    registered_revision = MODEL_REVISIONS[model_spec.label]
    if revision is not None and revision != registered_revision:
        raise ValueError(
            f"V3.1 revision mismatch for {model_spec.label}: "
            f"{revision} != {registered_revision}"
        )
    return run_vllm_experiment(
        stimuli_path=stimuli_path,
        output_dir=output_dir,
        model=model,
        revision=registered_revision,
        passage_lengths=passage_lengths,
        needle_counts=needle_counts,
        seeds=seeds,
        prompt_modes=prompt_modes,
        query_layout=query_layout,
        engine_config=engine_config,
        cache_dir=cache_dir,
        repo_root=repo_root,
        require_clean_git=require_clean_git,
        registered_model_spec=model_spec,
        protocol=V31_RUN_PROTOCOL,
        loaded_runtime=loaded_runtime,
    )


def run_v31_model_bundle(
    *,
    stimuli_path: str | Path,
    run_root: str | Path,
    model: str,
    revision: str | None = None,
    passage_lengths: Iterable[int] | None = None,
    needle_counts: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    query_layout: str = QUERY_LAYOUT,
    engine_config: EngineConfig | None = None,
    cache_dir: str | Path | None = None,
    repo_root: str | Path = ".",
    require_clean_git: bool = False,
) -> dict[str, Any]:
    """Run every registered mode for one model with one model load."""

    model_spec = resolve_model_spec(model)
    registered_revision = MODEL_REVISIONS[model_spec.label]
    if revision is not None and revision != registered_revision:
        raise ValueError(
            f"V3.1 revision mismatch for {model_spec.label}: "
            f"{revision} != {registered_revision}"
        )
    repo = Path(repo_root).resolve()
    if require_clean_git:
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("Formal bundle run requires a clean Git worktree")
    engine = engine_config or EngineConfig()
    runtime = load_vllm_runtime(
        model_spec=model_spec,
        revision=registered_revision,
        engine_config=engine,
        cache_dir=cache_dir,
    )
    root = Path(run_root).resolve()
    manifests: dict[str, dict[str, Any]] = {}
    for prompt_mode in model_spec.prompt_modes:
        task_id = _task_id(model_spec.label, prompt_mode)
        manifests[prompt_mode] = run_v31_experiment(
            stimuli_path=stimuli_path,
            output_dir=root / "shards" / task_id / "main",
            model=model_spec.label,
            revision=registered_revision,
            passage_lengths=passage_lengths,
            needle_counts=needle_counts,
            seeds=seeds,
            prompt_modes=(prompt_mode,),
            query_layout=query_layout,
            engine_config=engine,
            cache_dir=cache_dir,
            repo_root=repo,
            require_clean_git=require_clean_git,
            loaded_runtime=runtime,
        )
        gc.collect()
    return {
        "protocol_version": V31_RUN_PROTOCOL.protocol_version,
        "model_label": model_spec.label,
        "model_revision": registered_revision,
        "physical_model_loads": 1,
        "logical_shards": len(manifests),
        "completed_requests": sum(
            int(manifest["completed_requests"]) for manifest in manifests.values()
        ),
        "prompt_modes": list(manifests),
    }
