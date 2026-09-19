from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from realistic_niah_v4_4_3.analysis import analyze_campaign, audit_campaign
from realistic_niah_v4_4_3.io import atomic_json
from realistic_niah_v4_4_3.pipeline import initialize_campaign, run_model_campaign
from realistic_niah_v4_4_3.spec import V443Config


DEFAULT_SOURCE = (
    "runs/"
    "run_20260731_v4_numeric_presentation_v3"
)
DEFAULT_NAMESPACE = (
    "runs/v4_4_3_ov_causal"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated V4.4.3 OV causal tests."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("initialize", "model", "analyze", "audit", "campaign"),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-run-root", default=DEFAULT_SOURCE)
    parser.add_argument("--output-namespace-root", default=DEFAULT_NAMESPACE)
    parser.add_argument(
        "--config", default="configs/realistic_niah_v4_4_3.json"
    )
    parser.add_argument(
        "--v4-config", default="configs/realistic_niah_v4.json"
    )
    parser.add_argument(
        "--cache-dir", default="outputs/external/lambda_nfs_CoT-Non-thinking-v4_hf-cache"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _write_campaign_state(run_root: Path, state: str, **detail: object) -> None:
    if not run_root.exists():
        return
    atomic_json(
        run_root / "campaign.status.json",
        {
            "schema_version": "realistic_niah_v4_4_3_campaign_status_v1",
            "state": state,
            "updated_unix": time.time(),
            "detail": detail,
        },
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    v4_config_path = Path(args.v4_config)
    if not v4_config_path.is_absolute():
        v4_config_path = repo_root / v4_config_path
    config = V443Config.from_json(config_path)
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
        raise FileNotFoundError(
            "Run root is not initialized; run --stage initialize first"
        )
    if args.stage == "initialize":
        print(json.dumps({"state": "INITIALIZED", "run_root": str(run_root)}))
        return 0
    if args.stage == "model":
        if args.model is None:
            raise ValueError("--stage model requires --model")
        result = run_model_campaign(
            source_run_root=args.source_run_root,
            run_root=run_root,
            model_label=args.model,
            config=config,
            v4_config_path=v4_config_path,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            resume=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "campaign":
        _write_campaign_state(run_root, "RUNNING", stage="model_campaigns")
        models = [args.model] if args.model else list(config.model_labels)
        results = {}
        for model in models:
            results[model] = run_model_campaign(
                source_run_root=args.source_run_root,
                run_root=run_root,
                model_label=model,
                config=config,
                v4_config_path=v4_config_path,
                cache_dir=args.cache_dir,
                device_map=args.device_map,
                resume=args.resume,
            )
        if set(models) == set(config.model_labels):
            analysis = analyze_campaign(run_root, config=config)
            audit = audit_campaign(run_root, config=config)
            _write_campaign_state(
                run_root,
                "COMPLETE" if audit["complete"] else "AUDIT_FAILED",
                analysis=analysis,
                audit=audit,
            )
            if not audit["complete"]:
                raise RuntimeError("V4.4.3 final audit did not pass")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "analyze":
        result = analyze_campaign(run_root, config=config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "audit":
        result = audit_campaign(run_root, config=config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["complete"] else 1
    raise AssertionError(args.stage)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        try:
            parsed = parse_args()
            _write_campaign_state(
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
