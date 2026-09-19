from __future__ import annotations

import argparse

from .config import SUPPORTED_VERSIONS, preset_config
from .pipeline import run_pipeline


def build_parser(version: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run synthetic counting {version}")
    parser.add_argument("--preset", choices=("debug", "main"), default="debug")
    parser.add_argument("--stage", default="all", help="all or comma-separated train,attention,state,plots")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument(
        "--count-sampling",
        choices=("power", "exponential"),
        default=None,
        help="v17 training-count sampler; balanced validation is unchanged",
    )
    parser.add_argument("--power-alpha", type=float, default=None)
    parser.add_argument("--exponential-beta", type=float, default=None)
    parser.add_argument(
        "--min-candidate-windows",
        type=int,
        default=None,
        help=(
            "v16.1 minimum number of split-local windows required for a "
            "(target character, count) stratum to enter training"
        ),
    )
    parser.add_argument("--out-root", default=f"runs/synthetic_counting_{version}")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-sync-root", default=None)
    parser.add_argument("--skip-completed", action="store_true")
    return parser


def main(version: str, argv: list[str] | None = None) -> None:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(version)
    args = build_parser(version).parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "device": args.device,
            "seed": args.seed,
            "train_steps": args.train_steps,
            "count_sampling": args.count_sampling,
            "power_alpha": args.power_alpha,
            "exponential_beta": args.exponential_beta,
            "min_candidate_windows": args.min_candidate_windows,
        }.items()
        if value is not None
    }
    cfg = preset_config(version, args.preset, **overrides)
    run_pipeline(
        cfg,
        stage=args.stage,
        out_root=args.out_root,
        run_name=args.run_name,
        checkpoint_sync_root=args.checkpoint_sync_root,
        skip_completed=args.skip_completed,
    )
