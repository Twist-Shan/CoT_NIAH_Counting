from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v4.spec import V4Config
from realistic_niah_v4.stimuli import (
    ControlledFreezeSpec,
    audit_v4_grid,
    freeze_v4_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze and audit the controlled Realistic NIAH V4 grid."
    )
    parser.add_argument(
        "--config",
        default="configs/realistic_niah_v4.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--haystack-dir", default="data/haystacks/paul_graham")
    parser.add_argument("--entities", default="data/entities/cities.csv")
    parser.add_argument(
        "--fact-templates",
        default="data/templates/niah_fact_single_template.txt",
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-simple-tokenizer",
        action="store_true",
        help="Testing only; formal freezes must use the pinned HF tokenizer.",
    )
    args = parser.parse_args()

    config = V4Config.from_json(args.config)
    freeze_spec = ControlledFreezeSpec(
        config=config,
        haystack_dir=args.haystack_dir,
        entities_path=args.entities,
        fact_templates_path=args.fact_templates,
        tokenizer_cache_dir=args.cache_dir,
    )
    paths = freeze_v4_grid(
        output_dir=args.output_dir,
        freeze_spec=freeze_spec,
        require_huggingface_tokenizer=not args.allow_simple_tokenizer,
        overwrite=args.overwrite,
    )
    audit = audit_v4_grid(
        stimuli_path=paths["stimuli"],
        manifest_path=paths["manifest"],
        cache_dir=args.cache_dir,
        require_huggingface_tokenizer=not args.allow_simple_tokenizer,
    )
    audit_path = Path(args.output_dir) / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not audit["passed"]:
        raise SystemExit("V4 freeze audit failed:\n- " + "\n- ".join(audit["errors"]))
    print(
        json.dumps(
            {
                **{key: str(value) for key, value in paths.items()},
                "audit": str(audit_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
