from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from realistic_niah.runner import EngineConfig, run_vllm_experiment
from realistic_niah.spec import QUERY_LAYOUT

from .spec import MODEL_REVISIONS, V3_RUN_PROTOCOL, resolve_model_spec


def run_v3_experiment(
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
) -> dict[str, Any]:
    model_spec = resolve_model_spec(model)
    registered_revision = MODEL_REVISIONS[model_spec.label]
    if revision is not None and revision != registered_revision:
        raise ValueError(
            f"V3 revision mismatch for {model_spec.label}: "
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
        protocol=V3_RUN_PROTOCOL,
    )
