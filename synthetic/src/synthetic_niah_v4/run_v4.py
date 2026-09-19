from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import STAGES, build_config, ensure_output_dirs
from .data import balanced_examples
from .train import checkpoint_is_compatible, checkpoint_path, load_models, train_all
from .vocab import Vocab


def _write_config(run_dir: Path, cfg) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def _examples(cfg, per_count: int, offset: int):
    return balanced_examples(
        cfg.seq_len,
        int(per_count),
        seed=int(cfg.train.seed) + offset,
        counts=list(range(int(cfg.count_min), int(cfg.count_max) + 1)),
    )


def _ensure_models(cfg, vocab: Vocab, run_dir: Path, skip_completed: bool):
    missing = [
        checkpoint_path(run_dir, model_type, seed)
        for seed in cfg.seeds
        for model_type in ["non_thinking", "thinking"]
        if not checkpoint_is_compatible(checkpoint_path(run_dir, model_type, seed), cfg, vocab)
    ]
    if missing:
        print("[v4] missing/incompatible checkpoints; training required:", flush=True)
        for path in missing:
            print(f"  - {path}", flush=True)
        train_all(cfg, vocab, run_dir, skip_completed=skip_completed)
    return load_models(cfg, vocab, run_dir, seed=cfg.seeds[0])


def run_stage(stage: str, cfg, run_dir: Path, vocab: Vocab, skip_completed: bool = False):
    if stage == "train":
        return train_all(cfg, vocab, run_dir, skip_completed=skip_completed)
    if stage == "behavior_eval":
        from .generation import evaluate_behavior

        models = _ensure_models(cfg, vocab, run_dir, skip_completed)
        examples = _examples(cfg, cfg.eval_examples_per_count, 1000)
        df = evaluate_behavior(models, examples, vocab, cfg.device, batch_size=min(64, cfg.train.batch_size))
        df.to_csv(run_dir / "tables" / "behavior_eval.csv", index=False)
        summary = df.groupby(["model_type", "eval_mode", "count"], as_index=False)["final_accuracy"].mean()
        summary.to_csv(run_dir / "tables" / "behavior_accuracy_by_count.csv", index=False)
        return df
    if stage == "cache":
        from .cache import collect_hidden_cache, save_hidden_cache

        models = _ensure_models(cfg, vocab, run_dir, skip_completed)
        examples = _examples(cfg, cfg.cache_examples_per_count, 2000)
        cache = collect_hidden_cache(models, examples, vocab, cfg)
        save_hidden_cache(cache, run_dir)
        return cache.metadata
    if stage == "probe":
        from .cache import load_hidden_cache
        from .probes import run_probes

        cache = load_hidden_cache(run_dir)
        return run_probes(cache, cfg, run_dir)[0]
    if stage == "directions":
        from .directions import run_directions_from_disk

        models = _ensure_models(cfg, vocab, run_dir, skip_completed)
        return run_directions_from_disk(models, vocab, cfg, run_dir)
    if stage == "patching":
        from .patching import run_patching

        models = _ensure_models(cfg, vocab, run_dir, skip_completed)
        examples = _examples(cfg, max(cfg.eval_examples_per_count, cfg.steering.examples_per_count), 3000)
        return run_patching(models, examples, vocab, cfg, run_dir)
    if stage == "steering":
        from .steering import run_steering

        models = _ensure_models(cfg, vocab, run_dir, skip_completed)
        examples = _examples(cfg, cfg.steering.examples_per_count, 4000)
        return run_steering(models, examples, vocab, cfg, run_dir)
    if stage == "plots":
        from .plots import make_all_plots

        make_all_plots(run_dir)
        return None
    if stage == "report":
        from .report import generate_report

        return generate_report(run_dir, cfg)
    raise ValueError(f"unknown stage={stage}")


def run(args: argparse.Namespace) -> Path:
    cfg = build_config(args)
    run_dir = cfg.run_dir
    ensure_output_dirs(run_dir)
    _write_config(run_dir, cfg)
    vocab = Vocab.build()
    vocab.save(run_dir / "vocab.json")
    requested = args.stage
    stages = [
        "train",
        "behavior_eval",
        "cache",
        "probe",
        "directions",
        "patching",
        "steering",
        "plots",
        "report",
    ] if requested == "all" else [requested]
    for stage in stages:
        print(f"[v4] stage={stage}", flush=True)
        if stage in {"probe", "directions", "patching", "steering"}:
            if not (run_dir / "cache" / "hidden_cache_metadata.csv").exists():
                run_stage("cache", cfg, run_dir, vocab, args.skip_completed)
            if stage in {"directions", "patching", "steering"} and not (run_dir / "tables" / "probe_results.csv").exists():
                run_stage("probe", cfg, run_dir, vocab, args.skip_completed)
            if stage in {"patching", "steering"} and not (run_dir / "tables" / "direction_metrics.csv").exists():
                run_stage("directions", cfg, run_dir, vocab, args.skip_completed)
        run_stage(stage, cfg, run_dir, vocab, args.skip_completed)
    print(f"FINAL_RUN_DIR {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic NIAH Counting v4 steering pipeline")
    parser.add_argument("--preset", choices=["debug", "main"], default="debug")
    parser.add_argument("--stage", choices=list(STAGES), default="all")
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=None)
    parser.add_argument("--count-min", dest="count_min", type=int, default=None)
    parser.add_argument("--count-max", dest="count_max", type=int, default=None)
    parser.add_argument("--train-steps", dest="train_steps", type=int, default=None)
    parser.add_argument("--eval-examples-per-count", dest="eval_examples_per_count", type=int, default=None)
    parser.add_argument("--probe-examples-per-count", dest="probe_examples_per_count", type=int, default=None)
    parser.add_argument("--steering-examples-per-count", dest="steering_examples_per_count", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--out-root", dest="out_root", default="outputs/v4")
    parser.add_argument("--run-name", dest="run_name", default="")
    parser.add_argument("--skip-completed", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
