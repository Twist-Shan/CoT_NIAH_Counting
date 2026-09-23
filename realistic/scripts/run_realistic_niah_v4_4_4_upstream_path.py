from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch

from realistic_niah_v4_4_3.io import atomic_json
from realistic_niah_v4_4_4.spec import V444Config
from realistic_niah_v4_4_4.upstream_path_analysis import analyze_campaign, audit_campaign
from realistic_niah_v4_4_4.upstream_path_pipeline import run_model_stage
from realistic_niah_v4_4_4.upstream_path_spec import V444UpstreamPathConfig


STATUS_NAME = "v4_4_4_upstream_path_campaign.status.json"
SNAPSHOT_NAME = "v4_4_4_upstream_path_base_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the append-only V4.4.4 frozen-broad -> L28 OV path pilot."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("model-base", "model-expanded", "analyze", "audit", "campaign"),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--config", default="configs/realistic_niah_v4_4_4_upstream_path.json"
    )
    parser.add_argument("--v4-config", default="configs/realistic_niah_v4.json")
    parser.add_argument("--cache-dir", default="outputs/external/lambda_nfs_CoT-Non-thinking-v4_hf-cache")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _write_state(run_root: Path, state: str, **detail: object) -> None:
    if run_root.is_dir():
        atomic_json(
            run_root / STATUS_NAME,
            {
                "schema_version": "realistic_niah_v4_4_4_upstream_campaign_status_v1",
                "state": state,
                "updated_unix": time.time(),
                "detail": detail,
            },
        )


def _protected_paths(model_label: str) -> tuple[Path, ...]:
    return (
        Path("resolved_config.json"),
        Path("dataset/stimuli.jsonl"),
        Path(f"models/{model_label}/directions/complete.json"),
        Path(f"models/{model_label}/center_controls/complete.json"),
        Path(f"models/{model_label}/center_controls/selection.json"),
        Path(f"models/{model_label}/center_controls/artifacts.pt"),
        Path(f"models/{model_label}/smoke/complete.json"),
        Path(f"models/{model_label}/confirmation/complete.json"),
        Path(f"models/{model_label}/read_write_discovery/complete.json"),
        Path(f"models/{model_label}/read_write_smoke/complete.json"),
        Path(f"models/{model_label}/read_write_evaluation/complete.json"),
        Path(f"models/{model_label}/read_write_analysis/complete.json"),
        Path(f"models/{model_label}/relay_discovery/complete.json"),
        Path(f"models/{model_label}/relay_smoke/complete.json"),
        Path(f"models/{model_label}/relay_confirmation/complete.json"),
    )


def _snapshot_payload(run_root: Path, model_label: str) -> dict[str, object]:
    protected = _protected_paths(model_label)
    missing = [str(path) for path in protected if not (run_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen V4.4.4 artifacts are missing: {missing}")
    return {
        "schema_version": "realistic_niah_v4_4_4_upstream_base_snapshot_v1",
        "files": {
            path.as_posix(): {
                "bytes": (run_root / path).stat().st_size,
                "sha256": hashlib.sha256((run_root / path).read_bytes()).hexdigest(),
            }
            for path in protected
        },
    }


def _register_or_verify_snapshot(run_root: Path, model_label: str) -> None:
    payload = _snapshot_payload(run_root, model_label)
    path = run_root / SNAPSHOT_NAME
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("A protected V4.4.4 artifact changed")
    else:
        atomic_json(path, payload)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    resolved = run_root / "resolved_config.json"
    if not resolved.is_file():
        raise FileNotFoundError("Upstream supplement requires a completed V4.4.4 run root")
    base_config = V444Config.from_json(resolved)
    config = V444UpstreamPathConfig.from_json(_resolve(repo_root, args.config))
    config.validate_against_base(base_config)
    _register_or_verify_snapshot(run_root, config.model_label)

    def model(expanded: bool) -> dict[str, object]:
        _write_state(
            run_root,
            "RUNNING",
            stage="model-expanded" if expanded else "model-base",
        )
        result = run_model_stage(
            run_root=run_root,
            base_config=base_config,
            config=config,
            v4_config_path=_resolve(repo_root, args.v4_config),
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            expanded=expanded,
            resume=args.resume,
        )
        _release_cuda()
        _register_or_verify_snapshot(run_root, config.model_label)
        return result

    if args.stage == "model-base":
        print(json.dumps(model(False), ensure_ascii=False, indent=2))
        return 0
    if args.stage == "model-expanded":
        print(json.dumps(model(True), ensure_ascii=False, indent=2))
        return 0
    if args.stage == "analyze":
        result = analyze_campaign(run_root, config=config)
        print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
        return 0
    if args.stage == "audit":
        audit = audit_campaign(run_root, config=config)
        _register_or_verify_snapshot(run_root, config.model_label)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["all_checks_pass"] else 1
    if args.stage == "campaign":
        model(False)
        analysis = analyze_campaign(run_root, config=config)
        if analysis["decision"]["requires_expanded_stage"]:
            _write_state(
                run_root,
                "RUNNING",
                stage="model-expanded",
                reason="base H16/H19 did not support a constrained serial path",
            )
            model(True)
            analysis = analyze_campaign(run_root, config=config)
        _register_or_verify_snapshot(run_root, config.model_label)
        audit = analysis["audit"]
        state = "COMPLETE" if audit["all_checks_pass"] else "AUDIT_FAILED"
        _write_state(
            run_root,
            state,
            decision=analysis["decision"],
            audit=audit,
        )
        if not audit["all_checks_pass"]:
            raise RuntimeError("V4.4.4 upstream-path audit failed")
        print(json.dumps(analysis["decision"], ensure_ascii=False, indent=2))
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

