from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from realistic_niah_v4.spec import V4Config
from realistic_niah_v4.stimuli import ControlledFreezeSpec, freeze_v4_causal_v2_grid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the ordered fresh-seed reserve for V4.4 correct interventions."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-config", default="configs/realistic_niah_v4.json")
    parser.add_argument(
        "--definition",
        default="configs/realistic_niah_v4_4_correct_interventions.json",
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-simple-tokenizer", action="store_true")
    args = parser.parse_args()

    definition_path = Path(args.definition).resolve()
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if definition.get("schema_version") != (
        "realistic_niah_v4_4_correct_interventions_v2"
    ):
        raise ValueError("Unexpected correct-intervention definition schema")
    start = int(definition["reserve_seed_start"])
    end = int(definition["reserve_seed_end_inclusive"])
    if end < start:
        raise ValueError("Reserve seed range is reversed")
    seeds = tuple(range(start, end + 1))
    if len(seeds) < 2:
        raise ValueError("Fresh-seed reserve must contain at least two seeds")
    split = len(seeds) // 2
    base_path = Path(args.base_config).resolve()
    base = V4Config.from_json(base_path)
    config = replace(
        base,
        seeds=seeds,
        discovery_seeds=seeds[:split],
        confirmation_seeds=seeds[split:],
    )
    config.validate()
    outputs = freeze_v4_causal_v2_grid(
        output_dir=args.output_dir,
        freeze_spec=ControlledFreezeSpec(
            config=config,
            tokenizer_cache_dir=args.cache_dir,
        ),
        require_huggingface_tokenizer=not args.allow_simple_tokenizer,
        overwrite=args.overwrite,
    )
    provenance_path = Path(args.output_dir) / "correct_intervention_freeze.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "realistic_niah_v4_4_correct_intervention_freeze_v2"
                ),
                "definition_path": str(definition_path),
                "definition_sha256": _sha256(definition_path),
                "base_config_path": str(base_path),
                "base_config_sha256": _sha256(base_path),
                "reserve_seeds": list(seeds),
                "rows": len(seeds) * 11,
                "stimuli_sha256": _sha256(outputs["stimuli"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **{name: str(path) for name, path in outputs.items()},
                "freeze_provenance": str(provenance_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
