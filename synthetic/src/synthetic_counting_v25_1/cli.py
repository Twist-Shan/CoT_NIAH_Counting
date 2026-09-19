from __future__ import annotations

import argparse

import torch

from .calibration import run_v25_1_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Symmetrically calibrate v25's existing atomic-number LM-head rows"
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-per-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2478)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_v25_1_calibration(
        args.source_run,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        validation_per_count=args.validation_per_count,
        seed=args.seed,
    )


__all__ = ["build_parser", "main"]
