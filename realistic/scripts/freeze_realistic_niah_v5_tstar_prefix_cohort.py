#!/usr/bin/env python3
"""Freeze exact token-boundary early-stop contexts for a no-index cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.tstar_prefix import (  # noqa: E402
    TSTAR_PREFIX_SCHEMA,
    build_tstar_prefix_context,
)


CONTEXTS_FILENAME = "tstar_first_pass_contexts_v2.jsonl"
MANIFEST_FILENAME = "tstar_first_pass_manifest_v2.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Any, *, immutable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if immutable and path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f"Frozen file changed: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


def atomic_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, immutable: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    if immutable and path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f"Frozen file changed: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


def _ordered_source_rows(
    rows: list[dict[str, Any]], manifest: Mapping[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in by_seed:
            raise ValueError(f"Duplicate selected row seed={seed}")
        by_seed[seed] = row
    discovery = [int(value) for value in manifest.get("discovery_seeds", ())]
    confirmation = [int(value) for value in manifest.get("confirmation_seeds", ())]
    ordered = [("discovery", seed) for seed in discovery] + [
        ("confirmation", seed) for seed in confirmation
    ]
    expected = {seed for _split, seed in ordered}
    if set(by_seed) != expected:
        raise ValueError("Selected rows and frozen manifest contain different seeds")
    if len(discovery) != 20 or len(confirmation) != 10:
        raise ValueError("Frozen cohort must contain 20 discovery and 10 confirmation rows")
    return [(split, by_seed[seed]) for split, seed in ordered]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-rows", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = read_json(args.source_manifest)
    if source_manifest.get("status") != "FROZEN":
        raise ValueError("Source cohort manifest is not FROZEN")
    expected_source_hash = dict(source_manifest.get("files") or {}).get(
        args.selected_rows.name
    )
    actual_source_hash = sha256_file(args.selected_rows)
    if expected_source_hash != actual_source_hash:
        raise ValueError("Selected rows SHA256 does not match the source manifest")
    fixed_count = int(source_manifest["fixed_count"])
    audit_key = f"noindex_n{fixed_count}_format_audit"
    cohort_key = f"noindex_n{fixed_count}_cohort"
    ordered = _ordered_source_rows(read_jsonl(args.selected_rows), source_manifest)

    from transformers import AutoTokenizer

    spec = resolve_model_spec(str(args.model))
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    contexts: list[dict[str, Any]] = []
    for split, row in ordered:
        if int(row.get("gold_count", -1)) != fixed_count:
            raise ValueError("Source row gold_count disagrees with manifest")
        if str(row.get("model_label")) != str(args.model):
            raise ValueError("Source row model label disagrees with --model")
        contexts.append(
            build_tstar_prefix_context(
                row,
                tokenizer,
                audit_key=audit_key,
                cohort_key=cohort_key,
                cohort_split=split,
            )
        )

    contexts_path = args.output_dir / CONTEXTS_FILENAME
    atomic_jsonl(contexts_path, contexts)
    spill_counts = [int(row["token_boundary_right_spill_chars"]) for row in contexts]
    removed_tokens = [int(row["removed_output_token_count"]) for row in contexts]
    global_clean = {
        split: sum(
            bool(source_row[audit_key]["global_clean_eligible"])
            for source_split, source_row in ordered
            if source_split == split
        )
        for split in ("discovery", "confirmation")
    }
    manifest = {
        "schema_version": TSTAR_PREFIX_SCHEMA,
        "status": "FROZEN",
        "model_label": str(args.model),
        "fixed_count": fixed_count,
        "selection_population": "first_pass_noindex_enumeration",
        "stopping_rule": (
            "smallest whole-output-token prefix covering t_star, where t_star is "
            "the end of the K-th unique locally score-supported gold-record first "
            "occurrence in a cue-free, repetition-free first evidence pass"
        ),
        "source_generation_preserved": True,
        "source_selected_rows": str(args.selected_rows.resolve()),
        "source_selected_rows_sha256": actual_source_hash,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "selection_used_final_answer": False,
        "mechanism_outcomes_accessed": False,
        "future_recap_available_to_context": False,
        "discovery_seeds": [
            int(row["seed"]) for row in contexts if row["split"] == "discovery"
        ],
        "confirmation_seeds": [
            int(row["seed"]) for row in contexts if row["split"] == "confirmation"
        ],
        "context_count": len(contexts),
        "global_clean_sensitivity": global_clean,
        "token_boundary_right_spill_chars": {
            "max": max(spill_counts),
            "nonzero_rows": sum(value > 0 for value in spill_counts),
        },
        "removed_output_tokens": {
            "min": min(removed_tokens),
            "median": statistics.median(removed_tokens),
            "max": max(removed_tokens),
        },
        "files": {CONTEXTS_FILENAME: sha256_file(contexts_path)},
    }
    atomic_json(args.output_dir / MANIFEST_FILENAME, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
