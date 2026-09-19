from __future__ import annotations

import argparse
import json

from realistic_niah_v4.spec import V4Config
from realistic_niah_v4.stimuli import (
    ControlledFreezeSpec,
    freeze_v4_causal_v2_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the V4.4-only N=0..10 nested stimulus extension used by "
            "the causal-v2 ablation, patching, and steering study."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/realistic_niah_v4.json")
    parser.add_argument("--base-stimuli")
    parser.add_argument("--cache-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-simple-tokenizer",
        action="store_true",
        help="Testing only; formal freezes require the registered HF tokenizer.",
    )
    args = parser.parse_args()
    config = V4Config.from_json(args.config)
    outputs = freeze_v4_causal_v2_grid(
        output_dir=args.output_dir,
        freeze_spec=ControlledFreezeSpec(
            config=config,
            tokenizer_cache_dir=args.cache_dir,
        ),
        base_stimuli_path=args.base_stimuli,
        require_huggingface_tokenizer=not args.allow_simple_tokenizer,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
