#!/usr/bin/env python3
"""Build an isolated, deterministic N=10 candidate batch for V5 supplementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_generation.dynamic_niah import TokenizerAdapter
from realistic_niah_v4.spec import V4Config
from realistic_niah_v4.stimuli import ControlledFreezeSpec, build_controlled_family
from realistic_niah_v5.mechanism_dataset import paired_record
from realistic_niah_v5.spec import V5Config


SCHEMA_VERSION = "realistic_niah_v5_one_to_one_candidates_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part) for part in value.split(",") if part.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seed list must be non-empty and unique")
    return seeds


def build_candidates(
    output_dir: Path,
    *,
    discovery_seeds: tuple[int, ...],
    confirmation_seeds: tuple[int, ...],
    cache_dir: Path,
    haystack_dir: str,
    entities: str,
    fact_templates: str,
) -> dict[str, Any]:
    overlap = sorted(set(discovery_seeds) & set(confirmation_seeds))
    if overlap:
        raise ValueError(f"Discovery/confirmation seed overlap: {overlap}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Candidate output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_seeds = discovery_seeds + confirmation_seeds
    v4_config = V4Config(
        seeds=all_seeds,
        discovery_seeds=discovery_seeds,
        confirmation_seeds=confirmation_seeds,
    )
    v4_config.validate()
    tokenizer = TokenizerAdapter(
        v4_config.canonical_tokenizer,
        revision=v4_config.canonical_tokenizer_revision,
        cache_dir=str(cache_dir),
    )
    if tokenizer.backend != "huggingface":
        raise RuntimeError(
            "Pinned Hugging Face tokenizer is required: " + str(tokenizer.load_error)
        )
    freeze_spec = ControlledFreezeSpec(
        config=v4_config,
        haystack_dir=haystack_dir,
        entities_path=entities,
        fact_templates_path=fact_templates,
        tokenizer_cache_dir=str(cache_dir),
    )

    split_by_seed = {
        **{seed: "discovery" for seed in discovery_seeds},
        **{seed: "confirmation" for seed in confirmation_seeds},
    }
    stimuli: list[dict[str, Any]] = []
    family_metadata: list[dict[str, Any]] = []
    for seed in all_seeds:
        family, metadata = build_controlled_family(
            variant="v4.4",
            seed=seed,
            tokenizer=tokenizer,
            freeze_spec=freeze_spec,
        )
        candidates = []
        for stimulus in family:
            row = paired_record(stimulus, mode="native_thinking")
            if int(row["gold_count"]) == 10:
                candidates.append(row)
        if len(candidates) != 1:
            raise RuntimeError(f"Seed {seed} produced {len(candidates)} N=10 rows")
        row = candidates[0]
        if str(row["split"]) != split_by_seed[seed]:
            raise RuntimeError(
                f"Seed {seed} split mismatch: {row['split']} != {split_by_seed[seed]}"
            )
        row.update(
            {
                "supplemental": True,
                "supplement_schema_version": SCHEMA_VERSION,
                "supplement_purpose": "strict_one_to_one_n10",
                "merge_into_primary_300": False,
            }
        )
        stimuli.append(row)
        family_metadata.append({"split": split_by_seed[seed], **metadata})

    stimuli_path = output_dir / "stimuli_native_thinking_n10.jsonl"
    config_path = output_dir / "realistic_niah_v5_one_to_one_candidates.json"
    _atomic_jsonl(stimuli_path, stimuli)
    _atomic_json(
        config_path,
        V5Config(
            discovery_seeds=discovery_seeds,
            confirmation_seeds=confirmation_seeds,
        ).to_dict(),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "strict one-to-one N=10 candidate supplementation",
        "merge_into_primary_300": False,
        "discovery_seeds": list(discovery_seeds),
        "confirmation_seeds": list(confirmation_seeds),
        "candidate_rows": len(stimuli),
        "candidate_counts": sorted({int(row["gold_count"]) for row in stimuli}),
        "family_metadata": family_metadata,
        "files": {
            stimuli_path.name: _sha256(stimuli_path),
            config_path.name: _sha256(config_path),
        },
    }
    _atomic_json(output_dir / "candidate_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--discovery-seeds", type=_parse_seed_list, required=True)
    parser.add_argument("--confirmation-seeds", type=_parse_seed_list, required=True)
    parser.add_argument("--haystack-dir", default="data/haystacks/paul_graham")
    parser.add_argument("--entities", default="data/entities/cities.csv")
    parser.add_argument(
        "--fact-templates", default="data/templates/niah_fact_single_template.txt"
    )
    args = parser.parse_args()
    manifest = build_candidates(
        args.output_dir,
        discovery_seeds=args.discovery_seeds,
        confirmation_seeds=args.confirmation_seeds,
        cache_dir=args.cache_dir,
        haystack_dir=args.haystack_dir,
        entities=args.entities,
        fact_templates=args.fact_templates,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
