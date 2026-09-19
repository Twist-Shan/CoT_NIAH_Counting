from __future__ import annotations

import argparse
from pathlib import Path

from .config import build_config, write_config
from .vocab import Vocab


def run_dir_for_args(args: argparse.Namespace) -> Path:
    root = Path(args.out_root)
    return root / args.run_name if args.run_name else root


def run(args: argparse.Namespace) -> Path:
    cfg = build_config(args)
    run_dir = run_dir_for_args(args)
    for sub in ["checkpoints", "tables", "figures", "cache"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    vocab = Vocab.build(include_trace_indices=bool(cfg["trace_indices"]))
    cfg["model"]["vocab_size"] = len(vocab.id_to_token)
    cfg["model"]["bos_token_id"] = vocab.bos_id
    cfg["model"]["eos_token_id"] = vocab.eos_id
    cfg["model"]["pad_token_id"] = vocab.pad_id
    write_config(run_dir, cfg)
    vocab.save(run_dir / "vocab.json")

    stages = ["train", "eval", "cache", "probe", "attention", "plots", "report"] if args.stage == "all" else [args.stage]
    if "train" in stages:
        from .train import train_model

        train_model(cfg, vocab, run_dir)
    if "eval" in stages:
        from .evaluation import run_evaluation

        run_evaluation(cfg, vocab, run_dir)
    if "cache" in stages:
        from .cache import run_cache

        run_cache(cfg, vocab, run_dir)
    if "probe" in stages:
        from .probes import run_probes

        run_probes(cfg, vocab, run_dir)
    if "attention" in stages:
        from .attention import run_attention

        run_attention(cfg, vocab, run_dir)
    if "plots" in stages:
        from .plots import make_plots

        make_plots(run_dir)
    if "report" in stages:
        from .report import make_report

        make_report(run_dir)

    print(f"FINAL_RUN_DIR {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic NIAH Counting v5 explicit soft-switch runner")
    parser.add_argument("--preset", choices=["debug", "main"], default="debug")
    parser.add_argument("--stage", choices=["train", "eval", "cache", "probe", "attention", "plots", "report", "all"], default="all")
    parser.add_argument("--out-root", "--out_root", dest="out_root", default="outputs/v5_explicit_switch")
    parser.add_argument("--run-name", "--run_name", dest="run_name", default="")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=None)
    parser.add_argument("--count-min", "--count_min", dest="count_min", type=int, default=None)
    parser.add_argument("--count-max", "--count_max", dest="count_max", type=int, default=None)
    parser.add_argument("--train-steps", "--train_steps", dest="train_steps", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--thinking-fraction", "--thinking_fraction", dest="thinking_fraction", type=float, default=None)
    parser.add_argument("--eval-examples-per-count", "--eval_examples_per_count", dest="eval_examples_per_count", type=int, default=None)
    parser.add_argument("--probe-examples-per-count", "--probe_examples_per_count", dest="probe_examples_per_count", type=int, default=None)
    parser.add_argument("--attention-examples-per-count", "--attention_examples_per_count", dest="attention_examples_per_count", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument(
        "--trace-indices",
        "--trace_indices",
        dest="trace_indices",
        action="store_true",
        help="Render indexed traces as <I1> M1 ... <In> Mn (default).",
    )
    trace_group.add_argument(
        "--no-trace-indices",
        "--no_trace_indices",
        dest="trace_indices",
        action="store_false",
        help="Ablation: render marker-only traces as M1 ... Mn.",
    )
    parser.set_defaults(trace_indices=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
