from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .config import config_from_dict, prepare_run_dir, preset_config
from .core import Vocab
from .training import train_both_models


STAGES = ("train", "attention", "state", "plots")


def _sync_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "seed",
        "train_steps",
        "batch_size",
        "eval_every",
        "ar_eval_every",
        "checkpoint_every",
        "eval_examples_per_count",
        "ar_examples_per_count",
        "attention_examples_per_count",
        "attention_causal_examples_per_count",
        "state_train_examples_per_count",
        "state_eval_examples_per_count",
        "state_causal_examples_per_count",
        "early_stop_patience",
    )
    values = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if args.device:
        values["device"] = args.device
    return values


def run(args: argparse.Namespace) -> Path:
    requested = list(STAGES) if args.stage == "all" else [args.stage]
    cfg = preset_config(args.preset, **_overrides(args))
    run_dir = prepare_run_dir(args.out_root, cfg, args.run_name or None)
    sync_run_dir = Path(args.checkpoint_sync_root) / run_dir.name if args.checkpoint_sync_root else None
    if sync_run_dir is not None:
        sync_run_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_completed:
            # A Colab reconnect starts with an empty /content filesystem. Restore
            # the saved run before resolving config/checkpoints or analysis stages.
            _sync_tree(sync_run_dir, run_dir)

    config_path = run_dir / "config.json"
    vocab_path = run_dir / "vocab.json"
    if config_path.exists() and args.skip_completed:
        cfg = config_from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    else:
        config_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    vocab = Vocab.load(vocab_path) if vocab_path.exists() else Vocab.build(cfg)
    if not vocab_path.exists():
        vocab.save(vocab_path)
    if sync_run_dir is not None:
        shutil.copy2(config_path, sync_run_dir / "config.json")
        shutil.copy2(vocab_path, sync_run_dir / "vocab.json")

    print("=" * 88, flush=True)
    print("[v10] two separate v2-style GPT-2 counting models", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"stages={','.join(requested)} preset={cfg.preset} device={cfg.device}", flush=True)
    print(
        f"prompt_length={cfg.seq_len} count={cfg.count_min}-{cfg.count_max} "
        f"steps={cfg.train_steps} eval_every={cfg.eval_every} checkpoint_every={cfg.checkpoint_every}",
        flush=True,
    )
    print(
        "early_stopping="
        + (f"patience {cfg.early_stop_patience}" if cfg.early_stop_patience else "disabled (full learning dynamics)"),
        flush=True,
    )
    print("=" * 88, flush=True)

    if "train" in requested:
        train_both_models(
            cfg,
            vocab,
            run_dir,
            sync_run_dir=sync_run_dir,
            skip_completed=args.skip_completed,
        )
    if "attention" in requested:
        from .attention_causal import run_attention_causal

        marker = run_dir / "analysis" / "attention_causal" / "tables" / "topn_ablation_summary.csv"
        if not (args.skip_completed and marker.exists()):
            run_attention_causal(cfg, vocab, run_dir)
    if "state" in requested:
        from .state_causal import run_state_causal

        marker = run_dir / "analysis" / "state_causal" / "tables" / "trace_progress_transplant_summary.csv"
        if not (args.skip_completed and marker.exists()):
            run_state_causal(cfg, vocab, run_dir)
    if "plots" in requested:
        from .plots import make_all_plots

        make_all_plots(run_dir, cfg)
    if sync_run_dir is not None:
        _sync_tree(run_dir, sync_run_dir)
    print(f"V10_RUN_DIR={run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v10 two-model counting dynamics and causal mechanism pipeline")
    parser.add_argument("--preset", choices=["debug", "main"], default="debug")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--out-root", default="runs/synthetic_counting_v10")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--checkpoint-sync-root", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--ar-eval-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--eval-examples-per-count", type=int, default=None)
    parser.add_argument("--ar-examples-per-count", type=int, default=None)
    parser.add_argument("--attention-examples-per-count", type=int, default=None)
    parser.add_argument("--attention-causal-examples-per-count", type=int, default=None)
    parser.add_argument("--state-train-examples-per-count", type=int, default=None)
    parser.add_argument("--state-eval-examples-per-count", type=int, default=None)
    parser.add_argument("--state-causal-examples-per-count", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
