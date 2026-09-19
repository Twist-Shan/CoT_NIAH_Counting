from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import run_attention_analysis, run_state_analysis
from .config import canonical_run_specs, debug_run_specs, preset_config, select_specs
from .plots import make_plots
from .training import summarize, train_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v19 shared-digit explicit-trace suite")
    parser.add_argument("--preset", choices=("debug", "main"), default="debug")
    parser.add_argument("--suite", choices=("all", "power", "uniform"), default="all")
    parser.add_argument(
        "--stage",
        choices=("all", "train", "attention", "state", "plots"),
        default="all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-root", default="runs/synthetic_counting_v19")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-sync-root", default=None)
    parser.add_argument("--skip-completed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    overrides = {"seed": args.seed}
    if args.device is not None:
        overrides["device"] = args.device
    cfg = preset_config(args.preset, **overrides)
    run_name = args.run_name or f"v19_{args.preset}_{args.suite}_seed{cfg.seed}"
    run_dir = Path(args.out_root) / run_name
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    base_specs = debug_run_specs() if args.preset == "debug" else canonical_run_specs()
    specs = select_specs(args.suite, base_specs)
    if not specs:
        raise ValueError(
            f"suite={args.suite!r} selects no runs for preset={args.preset!r}; "
            "the debug preset contains only its compact uniform direct/CoT pair"
        )
    pd.DataFrame([spec.__dict__ for spec in specs]).to_csv(run_dir / "tables" / "suite_manifest.csv", index=False)
    if args.stage in {"all", "train"}:
        sync = Path(args.checkpoint_sync_root) if args.checkpoint_sync_root else None
        for index, spec in enumerate(specs, 1):
            print(f"[v19] run {index}/{len(specs)}: {spec.name}", flush=True)
            train_run(cfg, spec, run_dir, skip_completed=args.skip_completed, checkpoint_sync_root=sync)
        summarize(run_dir)
    if args.stage in {"all", "attention"}:
        run_attention_analysis(cfg, specs, run_dir)
    if args.stage in {"all", "state"}:
        run_state_analysis(cfg, specs, run_dir)
    if args.stage in {"all", "plots"}:
        summarize(run_dir)
        make_plots(run_dir)
    print(f"FINAL_RUN_DIR={run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
