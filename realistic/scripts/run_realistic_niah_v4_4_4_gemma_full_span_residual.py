from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from realistic_niah_v4_4_3.io import atomic_json
from realistic_niah_v4_4_4.gemma_residual_analysis import analyze_campaign
from realistic_niah_v4_4_4.gemma_residual_pipeline import run_model_campaign
from realistic_niah_v4_4_4.gemma_full_span_residual_spec import (
    GemmaFullSpanResidualConfig,
)
from realistic_niah_v4_4_4.pipeline import (
    freeze_confirmation_dataset,
    initialize_campaign,
)


DEFAULT_SOURCE = (
    "runs/"
    "run_20260731_v4_numeric_presentation_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one hash-locked Gemma full-span-ranked residual K dose."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("initialize", "dataset", "model", "analyze", "campaign"),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-namespace-root", required=True)
    parser.add_argument("--source-run-root", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--config",
        default="configs/realistic_niah_v4_4_4_gemma_full_span_residual_k2.json",
    )
    parser.add_argument("--v4-config", default="configs/realistic_niah_v4.json")
    parser.add_argument(
        "--cache-dir", default="outputs/external/lambda_nfs_CoT-Non-thinking-v4_hf-cache"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _status(run_root: Path, state: str, **detail: object) -> None:
    if run_root.exists():
        atomic_json(
            run_root / "gemma_residual.status.json",
            {
                "schema_version": "realistic_niah_v4_4_4_residual_status_v1",
                "state": state,
                "updated_unix": time.time(),
                "detail": detail,
            },
        )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    config = GemmaFullSpanResidualConfig.from_json(_resolve(repo_root, args.config))
    if args.stage in {"initialize", "campaign"}:
        initialize_campaign(
            source_run_root=args.source_run_root,
            output_namespace_root=args.output_namespace_root,
            run_root=run_root,
            config=config,  # type: ignore[arg-type]
            repo_root=repo_root,
            resume=args.resume,
        )
    elif not run_root.is_dir():
        raise FileNotFoundError("Residual-path run root is not initialized")
    if args.stage == "initialize":
        return 0
    if args.stage in {"dataset", "campaign"}:
        freeze_confirmation_dataset(
            run_root=run_root,
            repo_root=repo_root,
            v4_config_path=_resolve(repo_root, args.v4_config),
            cache_dir=args.cache_dir,
            config=config,  # type: ignore[arg-type]
            resume=args.resume,
        )
        if args.stage == "dataset":
            return 0
    if args.stage in {"model", "campaign"}:
        _status(run_root, "RUNNING", stage="model")
        result = run_model_campaign(
            run_root=run_root,
            config=config,
            v4_config_path=_resolve(repo_root, args.v4_config),
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            resume=args.resume,
        )
        if args.stage == "model":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    if args.stage in {"analyze", "campaign"}:
        analysis = analyze_campaign(run_root=run_root, config=config)
        _status(
            run_root,
            "COMPLETE",
            full_residual_count_path_support=analysis["primary_decision"][
                "full_residual_count_path_support"
            ],
            audit=analysis["audit"],
        )
        print(json.dumps(analysis["primary_decision"], ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        try:
            parsed = parse_args()
            _status(
                Path(parsed.run_root).resolve(),
                "FAILED",
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
        except Exception:
            pass
        traceback.print_exc()
        raise SystemExit(1)
