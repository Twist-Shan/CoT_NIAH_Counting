from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


PRESETS: dict[str, dict[str, Any]] = {
    "debug": {
        "preset": "debug",
        "train_seq_len": 256,
        "seq_lens_eval": [256, 512],
        "train_steps": 200,
        "batch_size": 32,
        "eval_every": 50,
        "log_every": 10,
        "checkpoint_every": 100,
        "test_examples_per_count": 20,
        "probe_examples_per_count": 50,
        "attention_examples_per_count": 20,
        "corrupt_examples_per_count": 20,
        "seeds": [1234],
    },
    "main": {
        "preset": "main",
        "train_seq_len": 256,
        "seq_lens_eval": [256, 512, 1024],
        "train_steps": 10000,
        "batch_size": 128,
        "eval_every": 500,
        "log_every": 50,
        "checkpoint_every": 1000,
        "test_examples_per_count": 1000,
        "probe_examples_per_count": 500,
        "attention_examples_per_count": 100,
        "corrupt_examples_per_count": 200,
        "seeds": [1234, 1235, 1236, 1237, 1238],
    },
}


MODEL_CONFIG = {
    "vocab_size": 90,
    "n_layers": 4,
    "n_heads": 4,
    "d_model": 256,
    "d_mlp": 1024,
    "dropout": 0.0,
    "context_len": 2048,
    "learning_rate": 3e-4,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "warmup_steps": 500,
    "grad_clip_norm": 1.0,
}


def parse_seeds(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(PRESETS[args.preset])
    cfg.update(MODEL_CONFIG)
    cfg["warmup_steps"] = 20 if args.preset == "debug" else MODEL_CONFIG["warmup_steps"]
    cfg["seeds"] = parse_seeds(args.seeds, cfg["seeds"])
    cfg["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return cfg


def write_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    serializable = dict(cfg)
    serializable["betas"] = list(serializable["betas"])
    (run_dir / "config.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    lines = []
    for key, value in serializable.items():
        lines.append(f"{key}: {json.dumps(value)}")
    (run_dir / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_checkpoint_path(run_dir: Path, model_type: str, seed: int) -> Path:
    return run_dir / "checkpoints" / f"{model_type}_seed{seed}" / "final.pt"


def train_all(cfg: dict[str, Any], run_dir: Path, vocab: Vocab, skip_completed: bool) -> pd.DataFrame:
    import pandas as pd

    from .train import train_model

    rows = []
    for seed in cfg["seeds"]:
        for model_type in ["non_thinking", "thinking"]:
            print(f"[train] {model_type} seed={seed}", flush=True)
            _, train_log = train_model(cfg, model_type, seed, vocab, run_dir, skip_completed=skip_completed)
            rows.append(train_log)
    train_log = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not train_log.empty:
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        train_log.to_csv(run_dir / "metrics" / "train_log.csv", index=False)
    return train_log


def ensure_checkpoints(cfg: dict[str, Any], run_dir: Path, vocab: Vocab, skip_completed: bool, skip_training: bool) -> pd.DataFrame:
    import pandas as pd

    missing = [
        model_checkpoint_path(run_dir, model_type, seed)
        for seed in cfg["seeds"]
        for model_type in ["non_thinking", "thinking"]
        if not model_checkpoint_path(run_dir, model_type, seed).exists()
    ]
    if missing and skip_training:
        raise FileNotFoundError(f"Missing checkpoints and --skip-training was set: {missing[:3]}")
    if missing:
        return train_all(cfg, run_dir, vocab, skip_completed=skip_completed)
    logs = sorted((run_dir / "metrics").glob("train_log_*_seed*.csv"))
    if logs:
        return pd.concat([pd.read_csv(path) for path in logs], ignore_index=True)
    train_log = run_dir / "metrics" / "train_log.csv"
    return pd.read_csv(train_log) if train_log.exists() else pd.DataFrame()


def examples_for_round(cfg: dict[str, Any], seed: int, per_count: int, offset: int = 0) -> dict[int, list]:
    from .data import balanced_examples

    return {
        seq_len: balanced_examples(seq_len, per_count, seed=seed + offset + seq_len * 17)
        for seq_len in cfg["seq_lens_eval"]
    }


def run_round1(cfg: dict[str, Any], run_dir: Path, vocab: Vocab) -> pd.DataFrame:
    import pandas as pd

    from .eval import evaluate_model, summarize_example_rows, threshold_table
    from .model import make_model
    from .train import checkpoint_steps_for_model, load_checkpoint

    rows = []
    for seed in cfg["seeds"]:
        examples_by_len = examples_for_round(cfg, seed, int(cfg["test_examples_per_count"]), offset=1000)
        for model_type in ["non_thinking", "thinking"]:
            for step, ckpt_path in checkpoint_steps_for_model(run_dir, model_type, seed, int(cfg["train_steps"])):
                print(f"[round1] {model_type} seed={seed} step={step}", flush=True)
                model = make_model(cfg, cfg["device"])
                load_checkpoint(model, ckpt_path, cfg["device"])
                rows.append(
                    evaluate_model(
                        model,
                        model_type,
                        examples_by_len,
                        vocab,
                        cfg["device"],
                        seed,
                        step,
                        batch_size=min(256, int(cfg["batch_size"])),
                    )
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    eval_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    tables = run_dir / "tables"
    metrics = run_dir / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    if not eval_df.empty:
        eval_df.to_csv(tables / "round1_eval_by_step.csv", index=False)
        eval_df.to_csv(metrics / "eval_by_step.csv", index=False)
        final_step = int(cfg["train_steps"])
        final_by_count = summarize_example_rows(
            eval_df[eval_df["checkpoint_step"].eq(final_step)],
            ["model_type", "seed", "checkpoint_step", "seq_len_eval", "count", "count_bin", "eval_mode"],
        )
        final_by_count.to_csv(tables / "round1_final_checkpoint_by_count.csv", index=False)
        by_bin = summarize_example_rows(
            eval_df,
            ["model_type", "seed", "checkpoint_step", "seq_len_eval", "count_bin", "eval_mode"],
        )
        by_bin.to_csv(metrics / "eval_by_bin.csv", index=False)
        threshold_table(by_bin).to_csv(tables / "round1_step_to_thresholds.csv", index=False)
    return eval_df


def run_round2(cfg: dict[str, Any], run_dir: Path, vocab: Vocab) -> pd.DataFrame:
    import pandas as pd

    from .corrupted_trace import run_corrupted_trace_eval, summarize_follow_rules
    from .model import make_model
    from .train import load_checkpoint

    rows = []
    for seed in cfg["seeds"]:
        examples_by_len = examples_for_round(cfg, seed, int(cfg["corrupt_examples_per_count"]), offset=2000)
        ckpt = model_checkpoint_path(run_dir, "thinking", seed)
        print(f"[round2] thinking corrupted traces seed={seed}", flush=True)
        model = make_model(cfg, cfg["device"])
        load_checkpoint(model, ckpt, cfg["device"])
        rows.append(
            run_corrupted_trace_eval(
                model,
                examples_by_len,
                vocab,
                cfg["device"],
                seed,
                int(cfg["train_steps"]),
                batch_size=min(256, int(cfg["batch_size"])),
            )
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    corrupt_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    corrupt_df.to_csv(tables / "round2_corrupted_trace_results.csv", index=False)
    summarize_follow_rules(corrupt_df).to_csv(tables / "round2_follow_rule_summary.csv", index=False)
    return corrupt_df


def run_round3(cfg: dict[str, Any], run_dir: Path, vocab: Vocab) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import pandas as pd

    from .attention import attention_leaderboard, run_attention_analysis
    from .interventions import run_head_ablation
    from .model import make_model
    from .probes import run_probes
    from .train import load_checkpoint

    probe_rows = []
    attention_rows = []
    ablation_rows = []
    masking_rows = []
    for seed in cfg["seeds"]:
        examples_by_len = examples_for_round(
            cfg,
            seed,
            max(int(cfg["probe_examples_per_count"]), int(cfg["attention_examples_per_count"])),
            offset=3000,
        )
        models = {}
        for model_type in ["non_thinking", "thinking"]:
            model = make_model(cfg, cfg["device"])
            load_checkpoint(model, model_checkpoint_path(run_dir, model_type, seed), cfg["device"])
            models[model_type] = model
        print(f"[round3] probes seed={seed}", flush=True)
        probe_rows.append(run_probes(models, examples_by_len, vocab, cfg, seed, int(cfg["train_steps"])))
        print(f"[round3] attention seed={seed}", flush=True)
        attention_df = run_attention_analysis(models, examples_by_len, vocab, cfg, seed, int(cfg["train_steps"]))
        attention_rows.append(attention_df)
        print(f"[round3] head ablation seed={seed}", flush=True)
        ablation_df, masking_df = run_head_ablation(models, attention_df, examples_by_len, vocab, cfg, seed, int(cfg["train_steps"]))
        ablation_rows.append(ablation_df)
        masking_rows.append(masking_df)
        for model in models.values():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    probe_df = pd.concat(probe_rows, ignore_index=True) if probe_rows else pd.DataFrame()
    attention_df = pd.concat(attention_rows, ignore_index=True) if attention_rows else pd.DataFrame()
    ablation_df = pd.concat(ablation_rows, ignore_index=True) if ablation_rows else pd.DataFrame()
    masking_df = pd.concat(masking_rows, ignore_index=True) if masking_rows else pd.DataFrame()
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    probe_df.to_csv(tables / "round3_probe_results.csv", index=False)
    attention_df.to_csv(tables / "round3_attention_head_metrics.csv", index=False)
    attention_leaderboard(attention_df).to_csv(tables / "round3_attention_leaderboard.csv", index=False)
    ablation_df.to_csv(tables / "round3_head_ablation_results.csv", index=False)
    masking_df.to_csv(tables / "round3_attention_masking_results.csv", index=False)
    return probe_df, attention_df, ablation_df, masking_df


def existing_or_empty(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def write_summary_json(run_dir: Path, cfg: dict[str, Any], eval_df, corrupt_df, attention_df) -> dict[str, Any]:
    generated = eval_df[eval_df["eval_mode"].isin(["direct", "generated_trace"])] if not eval_df.empty else eval_df

    def acc_by_len(model_type: str) -> dict[str, float]:
        if generated.empty:
            return {}
        sub = generated[generated["model_type"].eq(model_type)]
        if sub.empty:
            return {}
        return {str(k): float(v) for k, v in sub.groupby("seq_len_eval")["final_accuracy"].mean().to_dict().items()}

    trace = (
        eval_df[(eval_df["model_type"].eq("thinking")) & (eval_df["eval_mode"].eq("generated_trace"))]
        if not eval_df.empty
        else eval_df
    )
    summary = {
        "run_name": run_dir.name,
        "preset": cfg["preset"],
        "train_seq_len": cfg["train_seq_len"],
        "seq_lens_eval": cfg["seq_lens_eval"],
        "count_range": [1, 10],
        "seeds": cfg["seeds"],
        "non_thinking_final_accuracy_by_len": acc_by_len("non_thinking"),
        "thinking_final_accuracy_by_len": acc_by_len("thinking"),
        "thinking_trace_exact_by_len": {
            str(k): float(v) for k, v in trace.groupby("seq_len_eval")["trace_exact_rate"].mean().to_dict().items()
        }
        if not trace.empty
        else {},
        "round1_main_takeaway": "Round 1 isolates length/noise generalization at fixed count range 1..10.",
        "round2_main_takeaway": "Round 2 checks whether thinking final answers follow prompt count or corrupted trace-derived shortcuts.",
        "round3_main_takeaway": "Round 3 separates probe/attention diagnostics from causal single-head ablation evidence.",
        "limitations": [
            "All data are symbolic.",
            "Counts are limited to 1..10.",
            "The trace exposes count length, so final readout may exploit trace length or last-index shortcuts.",
            "Probe decodability is not causal evidence.",
            "Attention patterns are not causal unless ablation or masking changes behavior.",
            "There is no loss-mask ablation in this version by design.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> Path:
    from .vocab import Vocab

    cfg = build_cfg(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_{cfg['preset']}"
    run_dir = Path(args.out_root) / run_name
    for sub in ["checkpoints", "metrics", "tables", "figures"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    write_config(run_dir, cfg)
    vocab = Vocab.build()
    vocab.save(run_dir / "vocab.json")

    requested = args.round
    if requested in {"all", "1_hard_eval"}:
        train_log = ensure_checkpoints(cfg, run_dir, vocab, args.skip_completed, args.skip_training)
    else:
        train_log = ensure_checkpoints(cfg, run_dir, vocab, args.skip_completed, args.skip_training)

    eval_df = existing_or_empty(run_dir / "tables" / "round1_eval_by_step.csv")
    corrupt_df = existing_or_empty(run_dir / "tables" / "round2_corrupted_trace_results.csv")
    probe_df = existing_or_empty(run_dir / "tables" / "round3_probe_results.csv")
    attention_df = existing_or_empty(run_dir / "tables" / "round3_attention_head_metrics.csv")
    ablation_df = existing_or_empty(run_dir / "tables" / "round3_head_ablation_results.csv")

    if requested in {"all", "1_hard_eval"}:
        eval_df = run_round1(cfg, run_dir, vocab)
    if requested in {"all", "2_corrupted_trace"}:
        corrupt_df = run_round2(cfg, run_dir, vocab)
    if requested in {"all", "3_mechanistic"}:
        probe_df, attention_df, ablation_df, _ = run_round3(cfg, run_dir, vocab)

    from .plots import make_round1_plots, make_round2_plots, make_round3_plots

    figures_dir = run_dir / "figures"
    make_round1_plots(train_log, eval_df, figures_dir)
    make_round2_plots(corrupt_df, figures_dir)
    make_round3_plots(probe_df, attention_df, ablation_df, figures_dir)
    write_summary_json(run_dir, cfg, eval_df, corrupt_df, attention_df)
    print(f"FINAL_RUN_DIR {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic NIAH Counting v3 no-loss-ablation runner")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="debug")
    parser.add_argument("--round", choices=["all", "1_hard_eval", "2_corrupted_trace", "3_mechanistic"], default="all")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list, e.g. 1234,1235")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--out_root", default="runs/syn_v3_no_loss")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
