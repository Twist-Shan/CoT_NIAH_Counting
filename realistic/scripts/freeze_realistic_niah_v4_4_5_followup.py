from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from realistic_niah_v4.spec import V4Config
from realistic_niah_v4.stimuli import ControlledFreezeSpec, freeze_v4_causal_v2_grid


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze fresh V4.4-only stimuli for the V4.4.5 follow-up."
    )
    parser.add_argument(
        "--config", default="configs/realistic_niah_v4_4_5_stimuli.json"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = V4Config.from_json(config_path)
    outputs = freeze_v4_causal_v2_grid(
        output_dir=args.output_dir,
        freeze_spec=ControlledFreezeSpec(
            config=config,
            tokenizer_cache_dir=args.cache_dir,
        ),
        require_huggingface_tokenizer=True,
        overwrite=args.overwrite,
    )
    provenance = {
        "schema_version": "realistic_niah_v4_4_5_followup_freeze_v1",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "seeds": list(config.seeds),
        "discovery_seeds": list(config.discovery_seeds),
        "confirmation_seeds": list(config.confirmation_seeds),
        "stimuli_sha256": sha256(Path(outputs["stimuli"])),
        "rows_expected": len(config.seeds) * 11,
    }
    provenance_path = Path(args.output_dir) / "v4_4_5_freeze_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                **{key: str(value) for key, value in outputs.items()},
                "provenance": str(provenance_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
