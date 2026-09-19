from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from realistic_niah_v4_4_3.io import atomic_json
from realistic_niah_v4_4_4.relay_analysis import (
    analyze_relay_campaign,
    audit_relay_campaign,
)
from realistic_niah_v4_4_4.relay_pipeline import run_relay_model_campaign
from realistic_niah_v4_4_4.relay_report import build_relay_html_report
from realistic_niah_v4_4_4.relay_spec import V444RelayConfig
from realistic_niah_v4_4_4.spec import V444Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V4.4.4 relay-to-OV serial-mediation supplement."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("model", "analyze", "audit", "report", "campaign"),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--relay-config", default="configs/realistic_niah_v4_4_4_relay.json"
    )
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
    if not run_root.is_dir():
        return
    atomic_json(
        run_root / "relay_campaign.status.json",
        {
            "schema_version": "realistic_niah_v4_4_4_relay_campaign_status_v1",
            "state": state,
            "updated_unix": time.time(),
            "detail": detail,
        },
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    if not run_root.is_dir():
        raise FileNotFoundError("The completed V4.4.4 run root is missing")
    resolved_base = run_root / "resolved_config.json"
    if not resolved_base.is_file():
        raise FileNotFoundError("The V4.4.4 run has no resolved config")
    base_config = V444Config.from_json(resolved_base)
    relay_config = V444RelayConfig.from_json(
        _resolve(repo_root, args.relay_config)
    )
    relay_config.validate_against_base(base_config)
    if args.stage in {"model", "campaign"}:
        _write_state(run_root, "RUNNING", stage="relay_model_campaign")
        result = run_relay_model_campaign(
            run_root=run_root,
            base_config=base_config,
            relay_config=relay_config,
            v4_config_path=_resolve(repo_root, args.v4_config),
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            resume=args.resume,
        )
        if args.stage == "model":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    if args.stage in {"analyze", "campaign"}:
        analysis = analyze_relay_campaign(
            run_root=run_root,
            base_config=base_config,
            relay_config=relay_config,
        )
        report = build_relay_html_report(
            run_root=run_root, relay_config=relay_config
        )
        _write_state(
            run_root,
            "COMPLETE" if analysis["audit"]["all_checks_pass"] else "AUDIT_FAILED",
            primary_decision=analysis["primary_decision"],
            report=str(report),
            audit=analysis["audit"],
        )
        if not analysis["audit"]["all_checks_pass"]:
            raise RuntimeError("V4.4.4 relay final audit failed")
        print(json.dumps(analysis["primary_decision"], ensure_ascii=False, indent=2))
        return 0
    if args.stage == "audit":
        audit = audit_relay_campaign(
            run_root,
            base_config=base_config,
            relay_config=relay_config,
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["all_checks_pass"] else 1
    if args.stage == "report":
        report = build_relay_html_report(
            run_root=run_root, relay_config=relay_config
        )
        print(json.dumps({"report": str(report)}, ensure_ascii=False))
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
