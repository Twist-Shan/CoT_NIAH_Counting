#!/usr/bin/env python3
"""Build preregistered V4.4 seed pools for format-only natural-trace replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_generation.dynamic_niah import TokenizerAdapter  # noqa: E402
from realistic_niah_v4.spec import V4Config  # noqa: E402
from realistic_niah_v4.stimuli import (  # noqa: E402
    ControlledFreezeSpec,
    build_controlled_family,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--confirmation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=(9, 10))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--haystack-dir", default="data/haystacks/paul_graham")
    parser.add_argument("--entities", default="data/entities/cities.csv")
    parser.add_argument(
        "--fact-templates", default="data/templates/niah_fact_single_template.txt"
    )
    args = parser.parse_args()

    discovery = tuple(int(value) for value in args.discovery_seeds)
    confirmation = tuple(int(value) for value in args.confirmation_seeds)
    counts = tuple(sorted({int(value) for value in args.counts}))
    if not discovery or not confirmation or set(discovery) & set(confirmation):
        raise ValueError("Supplement discovery/confirmation pools must be nonempty and disjoint")
    if counts != (9, 10):
        raise ValueError("Natural no-enumeration supplement is frozen to N=9 and N=10")
    if any(seed <= 1263 for seed in discovery + confirmation):
        raise ValueError("Supplement seeds must be outside the original 1234..1263 panel")

    config = V4Config(
        seeds=discovery + confirmation,
        discovery_seeds=discovery,
        confirmation_seeds=confirmation,
    )
    config.validate()
    tokenizer = TokenizerAdapter(
        config.canonical_tokenizer,
        revision=config.canonical_tokenizer_revision,
        cache_dir=str(args.cache_dir),
    )
    if tokenizer.backend != "huggingface":
        raise RuntimeError(f"Pinned Hugging Face tokenizer unavailable: {tokenizer.load_error}")
    freeze_spec = ControlledFreezeSpec(
        config=config,
        haystack_dir=args.haystack_dir,
        entities_path=args.entities,
        fact_templates_path=args.fact_templates,
        tokenizer_cache_dir=str(args.cache_dir),
    )
    rows: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    role_by_seed = {
        **{seed: "discovery" for seed in discovery},
        **{seed: "confirmation" for seed in confirmation},
    }
    for seed in discovery + confirmation:
        family, metadata = build_controlled_family(
            variant="v4.4", seed=seed, tokenizer=tokenizer, freeze_spec=freeze_spec
        )
        selected = [row for row in family if int(row["gold_count"]) in set(counts)]
        if {int(row["gold_count"]) for row in selected} != set(counts):
            raise RuntimeError(f"Supplement family seed={seed} is missing N=9 or N=10")
        for row in selected:
            value = dict(row)
            value["split"] = role_by_seed[seed]
            value["supplemental"] = True
            value["supplement_seed_role"] = role_by_seed[seed]
            value["supplement_selection_uses_patch_outcome"] = False
            rows.append(value)
        families.append({"seed": seed, "role": role_by_seed[seed], **metadata})

    rows.sort(key=lambda row: (int(row["seed"]), int(row["gold_count"])))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    stimuli = output / "stimuli.jsonl"
    _atomic_jsonl(stimuli, rows)
    _atomic_json(
        output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_natural_seed_pool_v1",
            "status": "PASS",
            "discovery_seeds": list(discovery),
            "confirmation_seeds": list(confirmation),
            "counts": list(counts),
            "selection_rule": "ascending seed, format eligibility only; patch outcomes unavailable",
            "outcome_blind": True,
            "selection_rank_used": False,
            "stimulus_rows": len(rows),
            "stimuli_sha256": _sha256(stimuli),
            "families": families,
        },
    )


if __name__ == "__main__":
    main()
