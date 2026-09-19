from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.pipeline import (
    run_labeled_attention_analysis,
    run_model_stage,
    run_representation_analysis,
    write_runtime_provenance,
)
from realistic_niah_v4.spec import ANSWER_FORMATS, DESIGN_VARIANTS, MODEL_SPECS

MODEL_STAGES = (
    "preflight",
    "behavior",
    "representation-capture",
    "answer-query-representation-capture",
    "prompt-counter-attention-capture",
    "attention",
    "ablation",
    "patching",
    "geometric-steering",
)


def _csv_strings(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value.split(",") if item.strip())


def _csv_floats(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(item) for item in value.split(",") if item.strip())


def _count_pairs(value: str | None) -> tuple[tuple[int, int], ...] | None:
    if value is None:
        return None
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "count pairs must use LOW:HIGH comma-separated syntax"
            )
        result.append((int(parts[0]), int(parts[1])))
    if not result:
        raise argparse.ArgumentTypeError("at least one count pair is required")
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Qwen3-8B or Gemma4-E4B Realistic NIAH V4 "
            "non-thinking mechanistic stage."
        )
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            *MODEL_STAGES,
            "representation-analyze",
            "attention-analyze",
            "all",
        ),
    )
    parser.add_argument("--stimuli", required=True)
    parser.add_argument(
        "--config",
        default="configs/realistic_niah_v4.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument(
        "--answer-format",
        default="numeric",
        choices=ANSWER_FORMATS,
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--variants")
    parser.add_argument("--seeds")
    parser.add_argument("--counts")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--representation-all-counts",
        action="store_true",
        help=(
            "Capture every selected N for representation-capture. The historical "
            "default remains the configured N=10 representation row."
        ),
    )
    parser.add_argument("--forward-smoke", action="store_true")
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=16,
        help=(
            "Maximum continuation length for deterministic behavior labeling; "
            "the registered numeric answers need at most two answer tokens."
        ),
    )
    parser.add_argument(
        "--causal-layers",
        help="Explicit comma-separated decoder layers; overrides layer fractions.",
    )
    parser.add_argument(
        "--causal-top-ns",
        help="Comma-separated top-N head set sizes for ablation.",
    )
    parser.add_argument(
        "--causal-random-replicates",
        type=int,
        help="Same-layer random-head control replicates.",
    )
    parser.add_argument(
        "--causal-count-pairs",
        type=_count_pairs,
        help="Canonical LOW:HIGH pairs for residual patching.",
    )
    parser.add_argument(
        "--ablation-scopes",
        help="Comma-separated subset of answer_query,global.",
    )
    parser.add_argument(
        "--ablation-poolings",
        help="Comma-separated subset of span_end,span_mean.",
    )
    parser.add_argument(
        "--residual-patch-sites",
        help=(
            "Comma-separated subset of answer_query,toggled_needle_end,"
            "toggled_needle_span."
        ),
    )
    parser.add_argument(
        "--residual-patch-protocols",
        help="Comma-separated subset of single_layer,cumulative_from_layer.",
    )
    parser.add_argument(
        "--steering-count-pairs",
        type=_count_pairs,
        help="Canonical LOW:HIGH pairs for geometric steering.",
    )
    parser.add_argument(
        "--steering-methods",
        help=(
            "Comma-separated subset of centroid_transplant,centroid_delta,"
            "chord,polyline."
        ),
    )
    parser.add_argument(
        "--steering-alphas",
        help="Comma-separated interpolation fractions in (0,1].",
    )
    parser.add_argument(
        "--steering-random-replicates",
        type=int,
        help="Orthogonal norm-matched steering controls; zero disables them.",
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    variants = _csv_strings(args.variants)
    if variants is not None:
        unknown = sorted(set(variants) - set(DESIGN_VARIANTS))
        if unknown:
            parser.error(f"unknown design variants: {unknown}")
    provenance = write_runtime_provenance(
        output_dir=Path(args.output_dir) / args.model / args.answer_format,
        config_path=args.config,
        stimuli_path=args.stimuli,
        model_label=args.model,
        answer_format=args.answer_format,
        repo_root=args.repo_root,
    )
    common = {
        "stimuli_path": args.stimuli,
        "config_path": args.config,
        "output_dir": args.output_dir,
        "model_label": args.model,
        "answer_format": args.answer_format,
        "cache_dir": args.cache_dir,
        "device_map": args.device_map,
        "variants": variants,
        "seeds": _csv_ints(args.seeds),
        "counts": _csv_ints(args.counts),
        "representation_all_counts": args.representation_all_counts,
        "overwrite": args.overwrite,
        "forward_smoke": args.forward_smoke,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "causal_layers": _csv_ints(args.causal_layers),
        "causal_top_ns": _csv_ints(args.causal_top_ns),
        "causal_random_replicates": args.causal_random_replicates,
        "causal_count_pairs": args.causal_count_pairs,
        "ablation_scopes": _csv_strings(args.ablation_scopes),
        "ablation_poolings": _csv_strings(args.ablation_poolings),
        "residual_patch_sites": _csv_strings(args.residual_patch_sites),
        "residual_patch_protocols": _csv_strings(
            args.residual_patch_protocols
        ),
        "steering_count_pairs": args.steering_count_pairs,
        "steering_methods": _csv_strings(args.steering_methods),
        "steering_alphas": _csv_floats(args.steering_alphas),
        "steering_random_replicates": args.steering_random_replicates,
    }
    outputs: dict[str, object] = {"provenance": str(provenance)}
    if args.stage == "representation-analyze":
        outputs[args.stage] = run_representation_analysis(
            config_path=args.config,
            output_dir=args.output_dir,
            model_label=args.model,
            answer_format=args.answer_format,
        )
    elif args.stage == "attention-analyze":
        outputs[args.stage] = run_labeled_attention_analysis(
            stimuli_path=args.stimuli,
            config_path=args.config,
            output_dir=args.output_dir,
            model_label=args.model,
            answer_format=args.answer_format,
            cache_dir=args.cache_dir,
            variants=variants,
            seeds=_csv_ints(args.seeds),
            counts=_csv_ints(args.counts),
            overwrite_pooling_metrics=args.overwrite,
        )
    elif args.stage == "all":
        # Separate model loads keep every expensive stage restartable and make
        # failure provenance unambiguous. For cluster jobs, invoke each stage
        # separately instead of `all`.
        for stage in MODEL_STAGES[1:]:
            outputs[stage] = run_model_stage(stage=stage, **common)
            if stage == "representation-capture":
                outputs["representation-analyze"] = run_representation_analysis(
                    config_path=args.config,
                    output_dir=args.output_dir,
                    model_label=args.model,
                    answer_format=args.answer_format,
                )
            if stage == "attention":
                outputs["attention-analyze"] = run_labeled_attention_analysis(
                    stimuli_path=args.stimuli,
                    config_path=args.config,
                    output_dir=args.output_dir,
                    model_label=args.model,
                    answer_format=args.answer_format,
                    cache_dir=args.cache_dir,
                    variants=variants,
                    seeds=_csv_ints(args.seeds),
                    counts=_csv_ints(args.counts),
                    overwrite_pooling_metrics=args.overwrite,
                )
    else:
        outputs[args.stage] = run_model_stage(stage=args.stage, **common)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
