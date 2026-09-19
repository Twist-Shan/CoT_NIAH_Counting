from __future__ import annotations

import argparse

from .config import (
    ALL_MODEL_VARIANTS,
    SUPPORTED_TRAINING_COUNT_DISTRIBUTIONS,
    SUPPORTED_TASK_OUTPUT_LOSS_REDUCTIONS,
    VERSION_SPECS,
    preset_config,
)


def build_parser(version: str = "v20") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Run {version}: query-first RoPE synthetic counting (counts 1..30; "
            f"trace={VERSION_SPECS[version]['trace_format']})"
        )
    )
    parser.add_argument("--preset", choices=("debug", "main"), default="debug")
    parser.add_argument(
        "--stage",
        default="all",
        help="all or comma-separated prepare,train,phase,causal,extended,attention,state,plots",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--precision", choices=("float32", "bf16"), default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--recovery-every", type=int, default=None)
    parser.add_argument("--snapshot-shard-every", type=int, default=None)
    parser.add_argument("--snapshot-dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--ar-eval-every", type=int, default=None)
    parser.add_argument("--max-steps-for-language-pred", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--eval-examples-per-count", type=int, default=None)
    parser.add_argument("--final-examples-per-count", type=int, default=None)
    parser.add_argument("--ar-examples-per-count", type=int, default=None)
    parser.add_argument("--permutation-examples-per-count", type=int, default=None)
    parser.add_argument("--phase-examples-per-count", type=int, default=None)
    parser.add_argument("--phase-head-selection-examples-per-count", type=int, default=None)
    parser.add_argument("--final-count-loss-weight", type=float, default=None)
    parser.add_argument("--cot-trace-loss-weight", type=float, default=None)
    parser.add_argument("--task-output-count-weight", type=float, default=None)
    parser.add_argument("--task-output-trace-weight", type=float, default=None)
    parser.add_argument("--task-output-structure-weight", type=float, default=None)
    parser.add_argument("--answer-query-contrastive-weight", type=float, default=None)
    parser.add_argument("--answer-query-contrastive-temperature", type=float, default=None)
    parser.add_argument(
        "--task-output-scheduled-sampling-max-probability",
        type=float,
        default=None,
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--n-positions", type=int, default=None)
    parser.add_argument("--count-max-threshold", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-inner", type=int, default=None)
    parser.add_argument("--task-occurrence-ratio", type=float, default=None)
    parser.add_argument(
        "--training-count-distribution",
        choices=SUPPORTED_TRAINING_COUNT_DISTRIBUTIONS,
        default=None,
    )
    parser.add_argument(
        "--task-output-loss-reduction",
        choices=SUPPORTED_TASK_OUTPUT_LOSS_REDUCTIONS,
        default=None,
    )
    parser.add_argument("--needle-pool-size", type=int, default=None)
    parser.add_argument("--needle-pool-frequency-threshold", type=float, default=None)
    parser.add_argument("--needle-pool-frequency-bins", type=int, default=None)
    parser.add_argument("--needle-pool-seed", type=int, default=None)
    parser.add_argument("--candidate-filter-max-attempts", type=int, default=None)
    parser.add_argument(
        "--model-variant",
        action="append",
        choices=ALL_MODEL_VARIANTS,
        default=None,
    )
    safe_version = version.replace("-", "_").replace(".", "_")
    parser.add_argument("--out-root", default=f"runs/synthetic_counting_{safe_version}")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-sync-root", default=None)
    parser.add_argument("--skip-completed", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, version: str = "v20") -> None:
    parser = build_parser(version)
    args = parser.parse_args(argv)
    names = (
        "device",
        "seed",
        "train_steps",
        "lr_decay_steps",
        "min_lr",
        "batch_size",
        "precision",
        "checkpoint_every",
        "recovery_every",
        "snapshot_shard_every",
        "snapshot_dtype",
        "eval_every",
        "ar_eval_every",
        "max_steps_for_language_pred",
        "weight_decay",
        "eval_examples_per_count",
        "final_examples_per_count",
        "ar_examples_per_count",
        "permutation_examples_per_count",
        "phase_examples_per_count",
        "phase_head_selection_examples_per_count",
        "final_count_loss_weight",
        "cot_trace_loss_weight",
        "task_output_count_weight",
        "task_output_trace_weight",
        "task_output_structure_weight",
        "answer_query_contrastive_weight",
        "answer_query_contrastive_temperature",
        "task_output_scheduled_sampling_max_probability",
        "seq_len",
        "n_positions",
        "count_max_threshold",
        "n_layer",
        "n_head",
        "n_embd",
        "n_inner",
        "task_occurrence_ratio",
        "training_count_distribution",
        "task_output_loss_reduction",
        "needle_pool_size",
        "needle_pool_frequency_threshold",
        "needle_pool_frequency_bins",
        "needle_pool_seed",
        "candidate_filter_max_attempts",
    )
    overrides = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    version_spec = VERSION_SPECS[version]
    canonical_batch_size = version_spec.get("batch_size")
    canonical_final_weight = version_spec.get("final_count_loss_weight")
    canonical_task_output_count_weight = version_spec.get(
        "task_output_count_weight"
    )
    canonical_task_output_trace_weight = version_spec.get(
        "task_output_trace_weight"
    )
    canonical_task_output_structure_weight = version_spec.get(
        "task_output_structure_weight"
    )
    canonical_count_max = version_spec.get("count_max_threshold")
    canonical_n_layer = version_spec.get("n_layer")
    canonical_n_head = version_spec.get("n_head")
    canonical_n_embd = version_spec.get("n_embd")
    canonical_n_inner = version_spec.get("n_inner")
    canonical_pool_threshold = version_spec.get("needle_pool_frequency_threshold")
    canonical_pool_size = version_spec.get("needle_pool_size")
    canonical_count_distribution = version_spec.get("training_count_distribution")
    has_canonical_joint_sampler_cap = (
        "joint_sampler_max_starts_per_cell" in version_spec
    )
    canonical_joint_sampler_cap = version_spec.get(
        "joint_sampler_max_starts_per_cell"
    )
    canonical_context_permutation = version_spec.get(
        "permute_task_context_tokens"
    )
    canonical_task_output_reduction = version_spec.get("task_output_loss_reduction")
    canonical_delimiter_partition = version_spec.get(
        "task_output_trace_delimiters_as_structure"
    )
    canonical_tying = version_spec.get("tie_word_embeddings")
    canonical_partial_untie = version_spec.get("untie_atomic_count_readout")
    canonical_contrastive_weight = version_spec.get(
        "answer_query_contrastive_weight"
    )
    canonical_contrastive_temperature = version_spec.get(
        "answer_query_contrastive_temperature"
    )
    canonical_scheduled_sampling = version_spec.get(
        "task_output_scheduled_sampling_max_probability"
    )
    canonical_train_steps = version_spec.get("train_steps")
    canonical_lr_decay_steps = version_spec.get("lr_decay_steps")
    canonical_min_lr = version_spec.get("min_lr")
    canonical_phase_cloud_steps = version_spec.get("phase_cloud_steps")
    if (
        canonical_batch_size is not None
        and args.batch_size is not None
        and int(args.batch_size) != int(canonical_batch_size)
    ):
        parser.error(f"{version} fixes --batch-size at {canonical_batch_size}")
    if (
        canonical_final_weight is not None
        and args.final_count_loss_weight is not None
        and float(args.final_count_loss_weight) != float(canonical_final_weight)
    ):
        parser.error(
            f"{version} fixes --final-count-loss-weight at {canonical_final_weight:g}"
        )
    if (
        canonical_task_output_count_weight is not None
        and args.task_output_count_weight is not None
        and float(args.task_output_count_weight)
        != float(canonical_task_output_count_weight)
    ):
        parser.error(
            f"{version} fixes --task-output-count-weight at "
            f"{canonical_task_output_count_weight:g}"
        )
    if (
        canonical_task_output_trace_weight is not None
        and args.task_output_trace_weight is not None
        and float(args.task_output_trace_weight)
        != float(canonical_task_output_trace_weight)
    ):
        parser.error(
            f"{version} fixes --task-output-trace-weight at "
            f"{canonical_task_output_trace_weight:g}"
        )
    if (
        canonical_task_output_structure_weight is not None
        and args.task_output_structure_weight is not None
        and float(args.task_output_structure_weight)
        != float(canonical_task_output_structure_weight)
    ):
        parser.error(
            f"{version} fixes --task-output-structure-weight at "
            f"{canonical_task_output_structure_weight:g}"
        )
    if (
        canonical_count_max is not None
        and args.count_max_threshold is not None
        and int(args.count_max_threshold) != int(canonical_count_max)
    ):
        parser.error(
            f"{version} fixes --count-max-threshold at {canonical_count_max}"
        )
    if (
        canonical_n_layer is not None
        and args.n_layer is not None
        and int(args.n_layer) != int(canonical_n_layer)
    ):
        parser.error(f"{version} fixes --n-layer at {canonical_n_layer}")
    if (
        canonical_n_head is not None
        and args.n_head is not None
        and int(args.n_head) != int(canonical_n_head)
    ):
        parser.error(f"{version} fixes --n-head at {canonical_n_head}")
    if (
        canonical_n_embd is not None
        and args.n_embd is not None
        and int(args.n_embd) != int(canonical_n_embd)
    ):
        parser.error(f"{version} fixes --n-embd at {canonical_n_embd}")
    if (
        canonical_n_inner is not None
        and args.n_inner is not None
        and int(args.n_inner) != int(canonical_n_inner)
    ):
        parser.error(f"{version} fixes --n-inner at {canonical_n_inner}")
    if (
        canonical_pool_threshold is not None
        and args.needle_pool_frequency_threshold is not None
        and float(args.needle_pool_frequency_threshold)
        != float(canonical_pool_threshold)
    ):
        parser.error(
            f"{version} fixes --needle-pool-frequency-threshold at "
            f"{canonical_pool_threshold:g}"
        )
    if (
        canonical_pool_size is not None
        and args.needle_pool_size is not None
        and int(args.needle_pool_size) != int(canonical_pool_size)
    ):
        parser.error(f"{version} fixes --needle-pool-size at {canonical_pool_size}")
    if (
        canonical_count_distribution is not None
        and args.training_count_distribution is not None
        and args.training_count_distribution != canonical_count_distribution
    ):
        parser.error(
            f"{version} fixes --training-count-distribution at "
            f"{canonical_count_distribution}"
        )
    if (
        canonical_task_output_reduction is not None
        and args.task_output_loss_reduction is not None
        and args.task_output_loss_reduction != canonical_task_output_reduction
    ):
        parser.error(
            f"{version} fixes --task-output-loss-reduction at "
            f"{canonical_task_output_reduction}"
        )
    if (
        canonical_contrastive_weight is not None
        and args.answer_query_contrastive_weight is not None
        and float(args.answer_query_contrastive_weight)
        != float(canonical_contrastive_weight)
    ):
        parser.error(
            f"{version} fixes --answer-query-contrastive-weight at "
            f"{canonical_contrastive_weight:g}"
        )
    if (
        canonical_contrastive_temperature is not None
        and args.answer_query_contrastive_temperature is not None
        and float(args.answer_query_contrastive_temperature)
        != float(canonical_contrastive_temperature)
    ):
        parser.error(
            f"{version} fixes --answer-query-contrastive-temperature at "
            f"{canonical_contrastive_temperature:g}"
        )
    if (
        canonical_scheduled_sampling is not None
        and args.task_output_scheduled_sampling_max_probability is not None
        and float(args.task_output_scheduled_sampling_max_probability)
        != float(canonical_scheduled_sampling)
    ):
        parser.error(
            f"{version} fixes --task-output-scheduled-sampling-max-probability "
            f"at {canonical_scheduled_sampling:g}"
        )
    if (
        canonical_train_steps is not None
        and args.train_steps is not None
        and int(args.train_steps) != int(canonical_train_steps)
    ):
        parser.error(f"{version} fixes --train-steps at {canonical_train_steps}")
    if (
        canonical_lr_decay_steps is not None
        and args.lr_decay_steps is not None
        and int(args.lr_decay_steps) != int(canonical_lr_decay_steps)
    ):
        parser.error(
            f"{version} fixes --lr-decay-steps at {canonical_lr_decay_steps}"
        )
    if (
        canonical_min_lr is not None
        and args.min_lr is not None
        and float(args.min_lr) != float(canonical_min_lr)
    ):
        parser.error(f"{version} fixes --min-lr at {canonical_min_lr:g}")
    overrides.update(
        version=version,
        count_tokenization=version_spec["count_tokenization"],
        trace_format=version_spec["trace_format"],
    )
    if canonical_batch_size is not None:
        overrides["batch_size"] = canonical_batch_size
    if canonical_final_weight is not None:
        overrides["final_count_loss_weight"] = canonical_final_weight
    if canonical_task_output_count_weight is not None:
        overrides["task_output_count_weight"] = canonical_task_output_count_weight
    if canonical_task_output_trace_weight is not None:
        overrides["task_output_trace_weight"] = canonical_task_output_trace_weight
    if canonical_task_output_structure_weight is not None:
        overrides["task_output_structure_weight"] = (
            canonical_task_output_structure_weight
        )
    if canonical_count_max is not None:
        overrides["count_max_threshold"] = canonical_count_max
    if canonical_n_layer is not None:
        overrides["n_layer"] = canonical_n_layer
    if canonical_n_head is not None:
        overrides["n_head"] = canonical_n_head
    if canonical_n_embd is not None:
        overrides["n_embd"] = canonical_n_embd
    if canonical_n_inner is not None:
        overrides["n_inner"] = canonical_n_inner
    if canonical_pool_threshold is not None:
        overrides["needle_pool_frequency_threshold"] = canonical_pool_threshold
    if canonical_pool_size is not None:
        overrides["needle_pool_size"] = canonical_pool_size
    if canonical_count_distribution is not None:
        overrides["training_count_distribution"] = canonical_count_distribution
    if has_canonical_joint_sampler_cap:
        overrides["joint_sampler_max_starts_per_cell"] = canonical_joint_sampler_cap
    if canonical_context_permutation is not None:
        overrides["permute_task_context_tokens"] = canonical_context_permutation
    if canonical_task_output_reduction is not None:
        overrides["task_output_loss_reduction"] = canonical_task_output_reduction
    if canonical_delimiter_partition is not None:
        overrides["task_output_trace_delimiters_as_structure"] = (
            canonical_delimiter_partition
        )
    if canonical_tying is not None:
        overrides["tie_word_embeddings"] = canonical_tying
    if canonical_partial_untie is not None:
        overrides["untie_atomic_count_readout"] = canonical_partial_untie
    if canonical_contrastive_weight is not None:
        overrides["answer_query_contrastive_weight"] = canonical_contrastive_weight
    if canonical_contrastive_temperature is not None:
        overrides["answer_query_contrastive_temperature"] = (
            canonical_contrastive_temperature
        )
    if canonical_scheduled_sampling is not None:
        overrides["task_output_scheduled_sampling_max_probability"] = (
            canonical_scheduled_sampling
        )
    if canonical_train_steps is not None:
        overrides["train_steps"] = canonical_train_steps
    if canonical_lr_decay_steps is not None:
        overrides["lr_decay_steps"] = canonical_lr_decay_steps
    if canonical_min_lr is not None:
        overrides["min_lr"] = canonical_min_lr
    if canonical_phase_cloud_steps is not None:
        overrides["phase_cloud_steps"] = tuple(canonical_phase_cloud_steps)
    if args.model_variant is not None:
        overrides["enabled_model_variants"] = tuple(args.model_variant)
    elif version == "v22":
        # The nonthinking objective is identical to v20, so the canonical v22
        # run trains only the changed separator-trace Thinking model.
        overrides["enabled_model_variants"] = ("rope/thinking",)
    elif version in {"v23", "v24", "v24.2", "v24.3", "v24.4", "v24.5", "v24.6", "v24.7", "v25", "v26", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49"}:
        # Both objectives are retrained whenever the loss or count distribution
        # changes so the within-version Thinking/Non-thinking comparison stays
        # controlled.
        overrides["enabled_model_variants"] = (
            "rope/nonthinking",
            "rope/thinking",
        )
    cfg = preset_config(args.preset, **overrides)
    from .pipeline import run_v20_pipeline

    run_v20_pipeline(
        cfg,
        stage=args.stage,
        out_root=args.out_root,
        run_name=args.run_name,
        checkpoint_sync_root=args.checkpoint_sync_root,
        skip_completed=args.skip_completed,
    )


__all__ = ["build_parser", "main"]
