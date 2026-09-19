from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v4.modeling import load_registered_model
from realistic_niah_v4.spec import resolve_model_spec
from realistic_niah_v4_4_2.pipeline import (
    analyze_existing_captures,
    attach_legacy_v44_baseline,
    build_filestream_index,
    estimate_storage,
    run_condition,
    write_runtime_provenance,
)
from realistic_niah_v4_4_2.spec import MODES, MODELS, PROMPT_VARIANTS, V442Config


def _csv_ints(value: str | None):
    return None if not value else tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated Realistic NIAH V4.4.2 trace hidden-state/QK analysis."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("preflight", "generate", "capture", "generate-capture", "analyze", "analyze-existing", "all", "attach-baseline", "index", "aggregate", "estimate-storage"),
    )
    parser.add_argument("--config", default="configs/realistic_niah_v4_4_2.json")
    parser.add_argument("--stimuli")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--legacy-v4-root")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS)
    parser.add_argument("--seeds")
    parser.add_argument("--counts")
    parser.add_argument("--split", choices=("discovery", "confirmation"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--analysis-device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-legacy-baseline", action="store_true")
    parser.add_argument("--estimate-prompt-tokens", type=int, default=11000)
    parser.add_argument("--estimate-trace-tokens", type=int, default=4096)
    args = parser.parse_args()

    config = V442Config.from_json(args.config)
    write_runtime_provenance(args.output_dir, config=config)
    if args.stage == "attach-baseline":
        if not args.legacy_v4_root:
            parser.error("--legacy-v4-root is required for attach-baseline")
        result = {"legacy_reference": str(attach_legacy_v44_baseline(args.legacy_v4_root, args.output_dir))}
    elif args.stage == "index":
        result = {"filestream_index": str(build_filestream_index(args.output_dir))}
    elif args.stage == "aggregate":
        from realistic_niah_v4_4_2.aggregate import aggregate_run

        result = aggregate_run(args.output_dir)
    elif args.stage == "analyze-existing":
        result = analyze_existing_captures(
            args.output_dir,
            config=config,
            model_label=args.model,
            mode=args.mode,
            prompt_variant=args.prompt_variant,
            split=args.split,
            analysis_device=args.analysis_device,
            overwrite=args.overwrite,
        )
    elif args.stage == "estimate-storage":
        if not args.model:
            parser.error("--model is required for estimate-storage")
        spec = resolve_model_spec(args.model)
        model, _tokenizer, adapter = load_registered_model(
            spec,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            attention_backend=args.attention_backend,
        )
        result = estimate_storage(
            model,
            adapter,
            config=config,
            prompt_tokens=args.estimate_prompt_tokens,
            trace_tokens=args.estimate_trace_tokens,
        )
    else:
        for required in ("stimuli", "model", "mode", "prompt_variant"):
            if getattr(args, required) is None:
                parser.error(f"--{required.replace('_', '-')} is required for {args.stage}")
        result = run_condition(
            stimuli_path=args.stimuli,
            output_root=args.output_dir,
            config=config,
            model_label=args.model,
            mode=args.mode,
            prompt_variant=args.prompt_variant,
            stage=args.stage,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            attention_backend=args.attention_backend,
            analysis_device=args.analysis_device,
            seeds=_csv_ints(args.seeds),
            counts=_csv_ints(args.counts),
            split=args.split,
            overwrite=args.overwrite,
            reuse_legacy_baseline=not args.rerun_legacy_baseline,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
