from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from realistic_niah.stimuli import FreezeSpec, audit_frozen_grid, freeze_grid

from .spec import (
    CANONICAL_TOKENIZER,
    CANONICAL_TOKENIZER_REVISION,
    EXPECTED_STIMULI,
    INSERTION_DEPTH_MAX_FRACTION,
    INSERTION_DEPTH_MIN_FRACTION,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    SEEDS,
    V33_LONG_CONTEXT_FREEZE_PROTOCOL,
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


def _validate_registered_spec(spec: FreezeSpec) -> None:
    expected = default_freeze_spec()
    immutable_fields = (
        "passage_lengths",
        "needle_counts",
        "seeds",
        "canonical_tokenizer",
        "canonical_tokenizer_revision",
        "haystack_source_mode",
        "insertion_depth_min_fraction",
        "insertion_depth_max_fraction",
    )
    mismatches = [
        name
        for name in immutable_fields
        if getattr(spec, name) != getattr(expected, name)
    ]
    if mismatches:
        raise ValueError(
            "V3.3 long-context freeze spec changed immutable fields: "
            + ", ".join(mismatches)
        )


def freeze_long_context_grid(
    *,
    output_dir: str | Path,
    spec: FreezeSpec | None = None,
    tokenizer_cache_dir: str | None = None,
    require_huggingface_tokenizer: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    resolved = spec or default_freeze_spec()
    _validate_registered_spec(resolved)
    resolved = replace(
        resolved,
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
        protocol=V33_LONG_CONTEXT_FREEZE_PROTOCOL,
    )


def audit_long_context_grid(
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
        protocol=V33_LONG_CONTEXT_FREEZE_PROTOCOL,
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
        and saved.get("haystack_source_mode") == "multi_file_no_repeat"
        and float(saved.get("insertion_depth_min_fraction", -1))
        == INSERTION_DEPTH_MIN_FRACTION
        and float(saved.get("insertion_depth_max_fraction", -1))
        == INSERTION_DEPTH_MAX_FRACTION
    )
    errors = report.setdefault("errors", [])
    if int(report.get("rows_checked", -1)) != EXPECTED_STIMULI:
        report["passed"] = False
        errors.append(
            f"V3.3 long-context requires {EXPECTED_STIMULI} stimuli, "
            f"found {report.get('rows_checked')}"
        )
    if not registered:
        report["passed"] = False
        errors.append(
            "Frozen manifest does not match the registered V3.3 long-context grid"
        )
    return report
