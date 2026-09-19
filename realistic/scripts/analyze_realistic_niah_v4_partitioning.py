from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v4.modeling import load_registered_tokenizer
from realistic_niah_v4.partitioned_attention import analyze_partitioned_attention
from realistic_niah_v4.pipeline import render_encodings, select_stimuli
from realistic_niah_v4.spec import DESIGN_VARIANTS, V4Config, resolve_model_spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether V4 span-end heads select one occurrence or aggregate "
            "uniformly inside a positional partition."
        )
    )
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--config", default="configs/realistic_niah_v4.json")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--answer-format", default="numeric")
    parser.add_argument("--cache-dir")
    parser.add_argument("--variants", default=",".join(DESIGN_VARIANTS))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of ranked heads receiving bootstrap CIs and top-head plots.",
    )
    parser.add_argument(
        "--top-k-only",
        action="store_true",
        help="Analyze only top-k heads instead of every discovery broad candidate.",
    )
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--depth-bins", type=int, default=20)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260731)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = sorted(set(variants) - set(DESIGN_VARIANTS))
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    config = V4Config.from_json(args.config)
    model_spec = resolve_model_spec(args.model)
    tokenizer = load_registered_tokenizer(model_spec, cache_dir=args.cache_dir)
    selected = select_stimuli(
        args.stimuli,
        variants=variants,
        counts=(args.count,),
    )
    model_root = Path(args.run_root) / model_spec.label / args.answer_format
    output = (
        Path(args.output_dir)
        if args.output_dir
        else model_root / "attention" / "analysis" / "partitioning"
    )
    outputs = analyze_partitioned_attention(
        attention_index_path=(
            model_root / "attention" / "capture" / "attention_capture_index.jsonl"
        ),
        encodings=render_encodings(
            selected,
            tokenizer=tokenizer,
            model_label=model_spec.label,
            config=config,
            answer_format=args.answer_format,
        ),
        rankings_dir=model_root / "attention" / "analysis" / "rankings",
        output_dir=output,
        design_variants=variants,
        count=args.count,
        top_k=args.top_k,
        all_candidates=not args.top_k_only,
        partitions=args.partitions,
        depth_bins=args.depth_bins,
        bootstrap_repetitions=args.bootstrap_repetitions,
        random_seed=args.random_seed,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
