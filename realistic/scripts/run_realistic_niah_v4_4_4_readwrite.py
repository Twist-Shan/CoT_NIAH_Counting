from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from realistic_niah_v4_4_3.io import atomic_json
from realistic_niah_v4_4_4.spec import V444Config
from realistic_niah_v4_4_4.readwrite_analysis import analyze_campaign, audit_campaign
from realistic_niah_v4_4_4.readwrite_pipeline import run_model_campaign
from realistic_niah_v4_4_4.readwrite_spec import V444ReadWriteConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V4.4.4 factorized read / downstream OV-write supplement."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("model", "analyze", "audit", "campaign"),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--config", default="configs/realistic_niah_v4_4_4_readwrite.json"
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
        run_root / "v4_4_4_read_write_campaign.status.json",
        {
            "schema_version": (
                "realistic_niah_v4_4_4_read_write_campaign_status_v1"
            ),
            "state": state,
            "updated_unix": time.time(),
            "detail": detail,
        },
    )


def _register_base_snapshot(run_root: Path, model_label: str) -> None:
    protected = (
        Path("resolved_config.json"),
        Path("dataset/stimuli.jsonl"),
        Path(f"models/{model_label}/directions/complete.json"),
        Path(f"models/{model_label}/center_controls/complete.json"),
        Path(f"models/{model_label}/center_controls/selection.json"),
        Path(f"models/{model_label}/center_controls/artifacts.pt"),
        Path(f"models/{model_label}/smoke/complete.json"),
        Path(f"models/{model_label}/confirmation/complete.json"),
    )
    missing = [str(path) for path in protected if not (run_root / path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Frozen V4.4.4 base artifacts are missing: {missing}"
        )
    payload = {
        "schema_version": "realistic_niah_v4_4_4_read_write_base_snapshot_v1",
        "files": {
            path.as_posix(): {
                "bytes": (run_root / path).stat().st_size,
                "sha256": hashlib.sha256((run_root / path).read_bytes()).hexdigest(),
            }
            for path in protected
        },
    }
    path = run_root / "v4_4_4_read_write_base_snapshot.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("Frozen V4.4.4 base artifacts changed")
        return
    atomic_json(path, payload)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    resolved_base = run_root / "resolved_config.json"
    if not resolved_base.is_file():
        raise FileNotFoundError("Read/write supplement requires a completed V4.4.4 run root")
    base_config = V444Config.from_json(resolved_base)
    config = V444ReadWriteConfig.from_json(_resolve(repo_root, args.config))
    config.validate_against_base(base_config)
    _register_base_snapshot(run_root, config.model_label)
    if args.stage in {"model", "campaign"}:
        _write_state(run_root, "RUNNING", stage="model")
        result = run_model_campaign(
            run_root=run_root,
            base_config=base_config,
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
        analysis = analyze_campaign(run_root, config=config)
        state = "COMPLETE" if analysis["audit"]["all_checks_pass"] else "AUDIT_FAILED"
        _write_state(
            run_root,
            state,
            primary_decision=analysis["primary_decision"],
            audit=analysis["audit"],
        )
        if not analysis["audit"]["all_checks_pass"]:
            raise RuntimeError("V4.4.4 read/write supplement final audit failed")
        print(json.dumps(analysis["primary_decision"], ensure_ascii=False, indent=2))
        return 0
    if args.stage == "audit":
        audit = audit_campaign(run_root, config=config)
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
