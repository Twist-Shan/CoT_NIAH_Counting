from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from realistic_niah.stimuli import (
    FreezeSpec,
    audit_frozen_grid,
    freeze_grid,
)

from .spec import (
    CANONICAL_TOKENIZER,
    CANONICAL_TOKENIZER_REVISION,
    EXPECTED_STIMULI,
    INSERTION_DEPTH_MAX_FRACTION,
    INSERTION_DEPTH_MIN_FRACTION,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    SEEDS,
    V3_FREEZE_PROTOCOL,
)


def default_freeze_spec() -> FreezeSpec:
    return FreezeSpec(
        passage_lengths=PASSAGE_LENGTHS,
        needle_counts=NEEDLE_COUNTS,
        seeds=SEEDS,
        canonical_tokenizer=CANONICAL_TOKENIZER,
        canonical_tokenizer_revision=CANONICAL_TOKENIZER_REVISION,
        insertion_depth_min_fraction=INSERTION_DEPTH_MIN_FRACTION,
        insertion_depth_max_fraction=INSERTION_DEPTH_MAX_FRACTION,
    )


def freeze_v3_grid(
    *,
    output_dir: str | Path,
    spec: FreezeSpec | None = None,
    canonical_tokenizer_revision: str | None = None,
    tokenizer_cache_dir: str | None = None,
    require_huggingface_tokenizer: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    resolved = spec or default_freeze_spec()
    if resolved.passage_lengths != PASSAGE_LENGTHS:
        raise ValueError("Formal V3 freezing requires the registered length grid")
    if resolved.needle_counts != NEEDLE_COUNTS:
        raise ValueError("Formal V3 freezing requires the registered needle grid")
    if resolved.seeds != SEEDS:
        raise ValueError("Formal V3 freezing requires the registered paired seeds")
    if (
        resolved.insertion_depth_min_fraction
        != INSERTION_DEPTH_MIN_FRACTION
        or resolved.insertion_depth_max_fraction
        != INSERTION_DEPTH_MAX_FRACTION
    ):
        raise ValueError("Formal V3 freezing requires the 5%-95% depth interval")
    if canonical_tokenizer_revision not in (
        None,
        CANONICAL_TOKENIZER_REVISION,
    ):
        raise ValueError(
            "Formal V3 freezing requires the registered canonical tokenizer "
            f"revision {CANONICAL_TOKENIZER_REVISION}"
        )
    if resolved.canonical_tokenizer_revision not in (
        None,
        CANONICAL_TOKENIZER_REVISION,
    ):
        raise ValueError(
            "Formal V3 freeze spec has an unregistered tokenizer revision"
        )
    resolved = replace(
        resolved,
        canonical_tokenizer_revision=CANONICAL_TOKENIZER_REVISION,
        tokenizer_cache_dir=(
            tokenizer_cache_dir
            if tokenizer_cache_dir is not None
            else resolved.tokenizer_cache_dir
        ),
    )
    return freeze_grid(
        output_dir=output_dir,
        spec=resolved,
        require_huggingface_tokenizer=require_huggingface_tokenizer,
        overwrite=overwrite,
        protocol=V3_FREEZE_PROTOCOL,
    )


def audit_v3_grid(
    *,
    stimuli_path: str | Path,
    manifest_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    require_huggingface_tokenizer: bool = True,
) -> dict[str, Any]:
    stimuli_file = Path(stimuli_path)
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else stimuli_file.with_name("manifest.json")
    )
    report = audit_frozen_grid(
        stimuli_path=stimuli_file,
        manifest_path=manifest_file,
        cache_dir=cache_dir,
        require_huggingface_tokenizer=require_huggingface_tokenizer,
        protocol=V3_FREEZE_PROTOCOL,
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    saved = manifest.get("spec", {})
    registered = (
        tuple(saved.get("passage_lengths", ())) == PASSAGE_LENGTHS
        and tuple(saved.get("needle_counts", ())) == NEEDLE_COUNTS
        and tuple(saved.get("seeds", ())) == SEEDS
        and saved.get("canonical_tokenizer") == CANONICAL_TOKENIZER
        and saved.get("canonical_tokenizer_revision")
        == CANONICAL_TOKENIZER_REVISION
        and float(saved.get("insertion_depth_min_fraction", -1))
        == INSERTION_DEPTH_MIN_FRACTION
        and float(saved.get("insertion_depth_max_fraction", -1))
        == INSERTION_DEPTH_MAX_FRACTION
    )
    if report["rows_checked"] != EXPECTED_STIMULI:
        report["passed"] = False
        report["errors"].append(
            "V3 requires "
            f"{EXPECTED_STIMULI} stimuli, found {report['rows_checked']}"
        )
    if not registered:
        report["passed"] = False
        report["errors"].append(
            "Frozen manifest does not match the registered V3 matrix, "
            "tokenizer revision, or insertion-depth interval"
        )
    return report
