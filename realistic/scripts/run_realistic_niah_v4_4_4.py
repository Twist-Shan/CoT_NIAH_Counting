from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from realistic_niah_v4_4_3.io import atomic_json
from realistic_niah_v4_4_4.analysis import analyze_campaign, audit_campaign
from realistic_niah_v4_4_4.pipeline import (
    freeze_confirmation_dataset,
    initialize_campaign,
    run_model_campaign,
)
from realistic_niah_v4_4_4.spec import V444Config


DEFAULT_SOURCE = (
    "runs/"
    "run_20260731_v4_numeric_presentation_v3"
)
DEFAULT_NAMESPACE = (
    "runs/v4_4_4_natural_ov"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated V4.4.4 natural-OV confirmation."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "initialize",
            "dataset",
            "model",
            "analyze",
            "audit",
            "campaign",
        ),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-run-root", default=DEFAULT_SOURCE)
    parser.add_argument("--output-namespace-root", default=DEFAULT_NAMESPACE)
    parser.add_argument("--config", default="configs/realistic_niah_v4_4_4.json")
    parser.add_argument("--v4-config", default="configs/realistic_niah_v4.json")
    parser.add_argument(
        "--cache-dir", default="outputs/external/lambda_nfs_CoT-Non-thinking-v4_hf-cache"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _write_state(run_root: Path, state: str, **detail: object) -> None:
    if not run_root.exists():
        return
    atomic_json(
        run_root / "campaign.status.json",
        {
            "schema_version": "realistic_niah_v4_4_4_campaign_status_v1",
            "state": state,
            "updated_unix": time.time(),
            "detail": detail,
        },
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    config = V444Config.from_json(_resolve(repo_root, args.config))
    v4_config_path = _resolve(repo_root, args.v4_config)
    run_root = Path(args.run_root).resolve()
    if args.stage in {"initialize", "campaign"}:
        initialize_campaign(
            source_run_root=args.source_run_root,
            output_namespace_root=args.output_namespace_root,
            run_root=run_root,
            config=config,
            repo_root=repo_root,
            resume=args.resume,
        )
    elif not run_root.is_dir():
        raise FileNotFoundError("V4.4.4 run root is not initialized")
    if args.stage == "initialize":
        print(json.dumps({"state": "INITIALIZED", "run_root": str(run_root)}))
        return 0
    if args.stage in {"dataset", "campaign"}:
        freeze_confirmation_dataset(
            run_root=run_root,
            repo_root=repo_root,
            v4_config_path=v4_config_path,
            cache_dir=args.cache_dir,
            config=config,
            resume=args.resume,
        )
        if args.stage == "dataset":
            print(json.dumps({"state": "DATASET_COMPLETE", "run_root": str(run_root)}))
            return 0
    if args.stage in {"model", "campaign"}:
        _write_state(run_root, "RUNNING", stage="natural_ov_model_campaign")
        model_result = run_model_campaign(
            source_run_root=args.source_run_root,
            run_root=run_root,
            repo_root=repo_root,
            config=config,
            v4_config_path=v4_config_path,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            resume=args.resume,
        )
        if args.stage == "model":
            print(json.dumps(model_result, ensure_ascii=False, indent=2))
            return 0
    if args.stage in {"analyze", "campaign"}:
        analysis = analyze_campaign(run_root=run_root, config=config)
        audit = analysis["audit"]
        _write_state(
            run_root,
            "COMPLETE" if audit["all_checks_pass"] else "AUDIT_FAILED",
            primary_decision=analysis["primary_decision"],
            audit=audit,
        )
        if not audit["all_checks_pass"]:
            raise RuntimeError("V4.4.4 final audit failed")
        print(json.dumps(analysis["primary_decision"], ensure_ascii=False, indent=2))
        return 0
    if args.stage == "audit":
        audit = audit_campaign(run_root, config)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["all_checks_pass"] else 1
    raise AssertionError(args.stage)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        try:
            parsed = parse_args()
            _write_state(
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

