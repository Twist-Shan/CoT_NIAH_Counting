#!/usr/bin/env python3
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


SCHEMA_VERSION = "realistic_niah_v5_supplement_v1"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(
    exclusions: Iterable[Mapping[str, Any]],
    *,
    discovery_seed: int,
    confirmation_seed: int,
) -> list[dict[str, Any]]:
    seed_by_split = {
        "discovery": int(discovery_seed),
        "confirmation": int(confirmation_seed),
    }
    rows: list[dict[str, Any]] = []
    for exclusion in sorted(
        exclusions,
        key=lambda row: (
            str(row.get("model_label")),
            str(row.get("split")),
            int(row.get("gold_count", -1)),
            int(row.get("seed", -1)),
        ),
    ):
        split = str(exclusion["split"])
        if split not in seed_by_split:
            raise ValueError(f"Unsupported exclusion split: {split}")
        if exclusion.get("reason_code") != "no_aligned_registered_v5_trace_sites":
            raise ValueError(
                "Supplementation is limited to registered no-site exclusions"
            )
        count = int(exclusion["gold_count"])
        replacement_seed = seed_by_split[split]
        replacement_id = f"V4_4_T10000_N{count}_seed{replacement_seed}"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "model_label": exclusion.get("model_label"),
                "split": split,
                "gold_count": count,
                "original_request_id": exclusion.get("request_id"),
                "original_stimulus_id": exclusion.get("stimulus_id"),
                "original_seed": int(exclusion["seed"]),
                "original_reason_code": exclusion["reason_code"],
                "original_parser_status": exclusion.get("parser_status"),
                "replacement_seed": replacement_seed,
                "replacement_stimulus_id": replacement_id,
                "replacement_row_id": f"native_thinking:{replacement_id}",
                "analysis_role": "supplemental_replacement",
                "merge_into_primary_300": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build two separate V4.4 seed families and a deterministic mapping "
            "for V5 no-site exclusions."
        )
    )
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--discovery-seed", type=int, default=1264)
    parser.add_argument("--confirmation-seed", type=int, default=1265)
    parser.add_argument("--haystack-dir", default="data/haystacks/paul_graham")
    parser.add_argument("--entities", default="data/entities/cities.csv")
    parser.add_argument(
        "--fact-templates",
        default="data/templates/niah_fact_single_template.txt",
    )
    args = parser.parse_args()

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Supplement output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    exclusions = _load_jsonl(args.exclusions)
    if not exclusions:
        raise ValueError("No capture exclusions were supplied")
    mapping = _mapping(
        exclusions,
        discovery_seed=args.discovery_seed,
        confirmation_seed=args.confirmation_seed,
    )

    v4_config = V4Config(
        seeds=(args.discovery_seed, args.confirmation_seed),
        discovery_seeds=(args.discovery_seed,),
        confirmation_seeds=(args.confirmation_seed,),
    )
    v4_config.validate()
    tokenizer = TokenizerAdapter(
        v4_config.canonical_tokenizer,
        revision=v4_config.canonical_tokenizer_revision,
        cache_dir=str(args.cache_dir),
    )
    if tokenizer.backend != "huggingface":
        raise RuntimeError(
            "Pinned Hugging Face tokenizer is required for supplemental V4.4 seeds: "
            f"{tokenizer.load_error}"
        )
    freeze_spec = ControlledFreezeSpec(
        config=v4_config,
        haystack_dir=args.haystack_dir,
        entities_path=args.entities,
        fact_templates_path=args.fact_templates,
        tokenizer_cache_dir=str(args.cache_dir),
    )

    stimuli: list[dict[str, Any]] = []
    family_metadata: list[dict[str, Any]] = []
    for split, seed in (
        ("discovery", args.discovery_seed),
        ("confirmation", args.confirmation_seed),
    ):
        family, metadata = build_controlled_family(
            variant="v4.4",
            seed=seed,
            tokenizer=tokenizer,
            freeze_spec=freeze_spec,
        )
        for stimulus in family:
            row = paired_record(stimulus, mode="native_thinking")
            row["supplemental"] = True
            row["supplement_schema_version"] = SCHEMA_VERSION
            row["supplement_seed_role"] = split
            row["merge_into_primary_300"] = False
            stimuli.append(row)
        family_metadata.append({"split": split, **metadata})

    replacement_ids = {str(row["replacement_stimulus_id"]) for row in mapping}
    stimulus_ids = {str(row["stimulus_id"]) for row in stimuli}
    missing = sorted(replacement_ids - stimulus_ids)
    if missing:
        raise RuntimeError(f"Replacement stimuli were not generated: {missing}")

    stimuli_path = output / "stimuli_native_thinking.jsonl"
    mapping_path = output / "replacement_mapping.jsonl"
    config_path = output / "realistic_niah_v5_supplement.json"
    _atomic_jsonl(stimuli_path, stimuli)
    _atomic_jsonl(mapping_path, mapping)
    _atomic_json(
        config_path,
        V5Config(
            discovery_seeds=(args.discovery_seed,),
            confirmation_seeds=(args.confirmation_seed,),
        ).to_dict(),
    )
    _atomic_json(
        output / "supplement_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_exclusions": str(args.exclusions),
            "source_exclusions_sha256": _sha256(args.exclusions),
            "replacement_policy": (
                "one full held-separate V4.4 seed family per original split"
            ),
            "merge_into_primary_300": False,
            "discovery_seed": args.discovery_seed,
            "confirmation_seed": args.confirmation_seed,
            "stimulus_rows": len(stimuli),
            "replacement_rows": len(mapping),
            "family_metadata": family_metadata,
            "files": {
                stimuli_path.name: _sha256(stimuli_path),
                mapping_path.name: _sha256(mapping_path),
                config_path.name: _sha256(config_path),
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "stimulus_rows": len(stimuli),
                "replacement_rows": len(mapping),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
