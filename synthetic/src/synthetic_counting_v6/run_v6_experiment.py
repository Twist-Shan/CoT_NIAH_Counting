from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import V6Config, ensure_run_dirs, load_config, save_config
from .data import balanced_examples, load_examples, save_examples
from .evaluation import evaluate_all
from .train import load_final_models, train_models
from .vocab import Vocab

if False:  # pragma: no cover - typing only, avoids importing pandas/sklearn-heavy stages eagerly.
    import pandas as pd


STAGES = ("all", "train", "final_eval", "probes", "attention", "plots", "report")


def _apply_overrides(cfg: V6Config, args: argparse.Namespace) -> V6Config:
    for name in ["seq_len", "seed"]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, int(value))
    if args.device:
        cfg.device = args.device
    if args.train_steps is not None:
        cfg.train.train_steps = int(args.train_steps)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.eval_every is not None:
        cfg.train.eval_every = int(args.eval_every)
    if args.log_every is not None:
        cfg.train.log_every = int(args.log_every)
    if args.test_examples_per_count is not None:
        cfg.eval.test_examples_per_count = int(args.test_examples_per_count)
    if args.val_examples_per_count is not None:
        cfg.eval.val_examples_per_count = int(args.val_examples_per_count)
    if args.probe_examples_per_count is not None:
        cfg.eval.probe_train_examples_per_count = int(args.probe_examples_per_count)
        cfg.eval.probe_test_examples_per_count = int(args.probe_examples_per_count)
    if args.attention_examples_per_count is not None:
        cfg.eval.attention_examples_per_count = int(args.attention_examples_per_count)
    cfg.run_probes = not args.no_probes
    cfg.run_attention = not args.no_attention
    min_positions = cfg.seq_len + 2 * cfg.max_count + 8
    if cfg.model.n_positions < min_positions:
        cfg.model.n_positions = min_positions
    return cfg


def _default_run_dir(cfg: V6Config) -> Path:
    return Path(f"runs/v6_separator_trace_{cfg.preset}_seed{cfg.seed}")


def _pool_path(run_dir: Path, name: str) -> Path:
    return run_dir / "data" / f"{name}_pool.jsonl"


def _get_or_create_pool(run_dir: Path, name: str, seq_len: int, examples_per_count: int, seed: int, skip_completed: bool) -> list:
    path = _pool_path(run_dir, name)
    if skip_completed and path.exists():
        return load_examples(path)
    examples = balanced_examples(seq_len, examples_per_count, seed=seed)
    save_examples(path, examples)
    return examples


def prepare_pools(cfg: V6Config, run_dir: Path, skip_completed: bool) -> dict[str, list]:
    return {
        "val": _get_or_create_pool(run_dir, "val", cfg.seq_len, cfg.eval.val_examples_per_count, cfg.seed + 10_000, skip_completed),
        "test": _get_or_create_pool(run_dir, "test", cfg.seq_len, cfg.eval.test_examples_per_count, cfg.seed + 11_000, skip_completed),
    }


def write_manifest(run_dir: Path, cfg: V6Config) -> None:
    manifest = {
        "experiment": "trace_count_v6_separator_trace",
        "run_dir": str(run_dir),
        "vocab_size": 91,
        "trace_format": "<Sep> marker_1 <Sep> marker_2 ... <Sep> marker_n",
        "final_answer_tokens": "<1> ... <10>",
        "config": cfg.to_dict(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_final_eval(cfg: V6Config, run_dir: Path, vocab: Vocab, examples: list):
    models = load_final_models(cfg, vocab, run_dir)
    detail, by_count, by_bin = evaluate_all(
        models,
        examples,
        vocab,
        cfg.device,
        step=int(cfg.train.train_steps),
        batch_size=max(1, min(64, int(cfg.train.batch_size))),
    )
    detail.to_csv(run_dir / "metrics" / "metrics_final_test_detail.csv", index=False)
    by_count.to_csv(run_dir / "metrics" / "metrics_final_test_by_count.csv", index=False)
    by_bin.to_csv(run_dir / "metrics" / "metrics_final_test_by_bin.csv", index=False)
    detail.to_csv(run_dir / "metrics_final_test_detail.csv", index=False)
    by_count.to_csv(run_dir / "metrics_final_test_by_count.csv", index=False)
    by_bin.to_csv(run_dir / "metrics_final_test_by_bin.csv", index=False)
    return detail, by_count, by_bin


def run_pipeline(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config, args.preset)
    cfg = _apply_overrides(cfg, args)
    run_dir = Path(args.run_dir) if args.run_dir else _default_run_dir(cfg)
    ensure_run_dirs(run_dir)
    save_config(cfg, run_dir / "config.yaml")
    write_manifest(run_dir, cfg)
    vocab = Vocab.build()
    vocab.save(run_dir / "vocab.json")
    pools = prepare_pools(cfg, run_dir, args.skip_completed)

    requested = args.stage
    stages = ["train", "final_eval", "probes", "attention", "plots", "report"] if requested == "all" else [requested]
    models = None
    for stage in stages:
        print(f"[v6] stage={stage}", flush=True)
        if stage == "train":
            models, _train_df = train_models(cfg, vocab, run_dir, skip_completed=args.skip_completed, eval_examples=pools["val"])
        elif stage == "final_eval":
            run_final_eval(cfg, run_dir, vocab, pools["test"])
        elif stage == "probes":
            if cfg.run_probes:
                from .probes import run_probes

                models = models or load_final_models(cfg, vocab, run_dir)
                run_probes(models, vocab, cfg, run_dir)
        elif stage == "attention":
            if cfg.run_attention:
                from .attention import run_attention

                models = models or load_final_models(cfg, vocab, run_dir)
                run_attention(models, vocab, cfg, run_dir)
        elif stage == "plots":
            from .plots import make_all_plots

            make_all_plots(run_dir)
        elif stage == "report":
            from .report import generate_report

            generate_report(run_dir)
        else:
            raise ValueError(f"unknown stage={stage}")
    print(f"FINAL_RUN_DIR {run_dir}", flush=True)
    print(f"FINAL_REPORT {run_dir / 'report' / 'report.html'}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace Count v6: separator-delimited thinking trace experiment")
    parser.add_argument("--config", default=None, help="YAML config path. If omitted, --preset chooses built-in defaults.")
    parser.add_argument("--preset", choices=["debug", "main"], default="debug")
    parser.add_argument("--stage", choices=list(STAGES), default="all")
    parser.add_argument("--run-dir", "--run_dir", default="")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--test-examples-per-count", type=int, default=None)
    parser.add_argument("--val-examples-per-count", type=int, default=None)
    parser.add_argument("--probe-examples-per-count", type=int, default=None)
    parser.add_argument("--attention-examples-per-count", type=int, default=None)
    parser.add_argument("--no-probes", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser


def main() -> None:
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
