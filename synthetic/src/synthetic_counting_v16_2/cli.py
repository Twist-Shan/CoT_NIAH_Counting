from __future__ import annotations

import argparse

from .config import ALL_MODEL_VARIANTS, preset_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated synthetic counting v16_2")
    parser.add_argument("--preset", choices=("debug", "main"), default="debug")
    parser.add_argument(
        "--stage",
        default="all",
        help="all or comma-separated prepare,train,attention,state,plots",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="save numeric checkpoints at this cadence; the objective boundary and final step are also retained",
    )
    parser.add_argument(
        "--max-steps-for-language-pred",
        type=int,
        default=None,
        help="use all-sequence language loss through this step, then train only the mode-specific task-output span",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW decoupled weight-decay coefficient; 0 disables decay",
    )
    parser.add_argument(
        "--eval-examples-per-count",
        type=int,
        default=None,
        help="fixed evaluation examples for each accepted count (100 means 1000 per suite for counts 1..10)",
    )
    parser.add_argument(
        "--final-count-loss-weight",
        type=float,
        default=None,
        help="training-loss weight for the final numeric count target",
    )
    parser.add_argument(
        "--cot-trace-loss-weight",
        type=float,
        default=None,
        help="training-loss weight for thinking trace index and marker targets",
    )
    rpe_distance = parser.add_mutually_exclusive_group()
    rpe_distance.add_argument(
        "--rpe-max-update",
        dest="rpe_max_update",
        action="store_true",
        default=None,
        help="set RPE max_relative_distance to max_render_len - 1",
    )
    rpe_distance.add_argument(
        "--no-rpe-max-update",
        dest="rpe_max_update",
        action="store_false",
        help="retain the legacy fixed max_relative_distance",
    )
    parser.add_argument(
        "--model-variant",
        action="append",
        choices=ALL_MODEL_VARIANTS,
        default=None,
        help="model variant to run; repeat for any subset of the four variants",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument(
        "--count-max-threshold",
        type=int,
        default=None,
        help="accepted-count/output-number maximum; cfg.count_max is only a read-only alias",
    )
    parser.add_argument(
        "--task-occurrence-ratio",
        type=float,
        default=None,
        help="example-level probability that a training window uses counting-task formatting",
    )
    parser.add_argument("--needle-pool-size", type=int, default=None)
    parser.add_argument("--needle-pool-frequency-threshold", type=float, default=None)
    parser.add_argument("--needle-pool-frequency-bins", type=int, default=None)
    parser.add_argument("--needle-pool-seed", type=int, default=None)
    parser.add_argument("--candidate-filter-max-attempts", type=int, default=None)
    parser.add_argument("--corpus-train-fraction", type=float, default=None)
    parser.add_argument("--corpus-validation-fraction", type=float, default=None)
    order = parser.add_mutually_exclusive_group()
    order.add_argument(
        "--shuffle-needle-set-order",
        dest="shuffle_needle_set_order",
        action="store_true",
        default=None,
        help="shuffle the three prefix members per example (default)",
    )
    order.add_argument(
        "--canonical-needle-set-order",
        dest="shuffle_needle_set_order",
        action="store_false",
        help="render the three prefix members in canonical code-point order",
    )
    parser.add_argument("--out-root", default="runs/synthetic_counting_v16_2")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-sync-root", default=None)
    parser.add_argument("--skip-completed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    names = (
        "device",
        "seed",
        "train_steps",
        "checkpoint_every",
        "max_steps_for_language_pred",
        "weight_decay",
        "eval_examples_per_count",
        "final_count_loss_weight",
        "cot_trace_loss_weight",
        "rpe_max_update",
        "seq_len",
        "count_max_threshold",
        "task_occurrence_ratio",
        "needle_pool_size",
        "needle_pool_frequency_threshold",
        "needle_pool_frequency_bins",
        "needle_pool_seed",
        "candidate_filter_max_attempts",
        "corpus_train_fraction",
        "corpus_validation_fraction",
        "shuffle_needle_set_order",
    )
    overrides = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if args.model_variant is not None:
        overrides["enabled_model_variants"] = tuple(args.model_variant)
    cfg = preset_config(args.preset, **overrides)
    from .pipeline import run_v16_2_pipeline

    run_v16_2_pipeline(
        cfg,
        stage=args.stage,
        out_root=args.out_root,
        run_name=args.run_name,
        checkpoint_sync_root=args.checkpoint_sync_root,
        skip_completed=args.skip_completed,
    )
