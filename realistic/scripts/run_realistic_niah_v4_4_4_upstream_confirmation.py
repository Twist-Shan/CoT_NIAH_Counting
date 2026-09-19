from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch

from realistic_niah_v4.spec import V4Config
from realistic_niah_v4_4_3.io import atomic_json
from realistic_niah_v4_4_4.pipeline import (
    _direction_config,
    _load_model,
    v444_v4_config,
)
from realistic_niah_v4_4_4.spec import V444Config
from realistic_niah_v4_4_4.upstream_confirmation_analysis import analyze_campaign
from realistic_niah_v4_4_4.upstream_confirmation_spec import (
    V444UpstreamConfirmationConfig,
)
from realistic_niah_v4_4_4.upstream_path_pipeline import run_stage


STATUS_NAME = "v4_4_4_upstream_confirmation.status.json"
PREREGISTRATION_NAME = "v4_4_4_upstream_confirmation.preregistered.json"
DEFAULT_EXPLORATORY_RUN = (
    "runs/v4_4_4_natural_ov/"
    "run_20260803_v4_4_4_natural_ov_qwen_l28_a100_1501368870_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent confirmation of the V4.4.4 upstream slot-state path."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("model-primary", "model-loo", "analyze", "campaign"),
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--config",
        default="configs/realistic_niah_v4_4_4_upstream_confirmation.json",
    )
    parser.add_argument("--v4-config", default="configs/realistic_niah_v4.json")
    parser.add_argument(
        "--exploratory-run-root", default=DEFAULT_EXPLORATORY_RUN
    )
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_status(run_root: Path, state: str, **detail: object) -> None:
    atomic_json(
        run_root / STATUS_NAME,
        {
            "schema_version": "realistic_niah_v4_4_4_upstream_confirmation_status_v1",
            "state": state,
            "updated_unix": time.time(),
            "detail": detail,
        },
    )


def _register_or_verify_preregistration(
    *,
    run_root: Path,
    exploratory_run_root: Path,
    config_path: Path,
    config: V444UpstreamConfirmationConfig,
) -> None:
    dataset = run_root / "dataset" / "stimuli.jsonl"
    exploratory_analysis = (
        exploratory_run_root
        / "models"
        / config.model_label
        / "upstream_path_analysis"
        / "realistic_niah_v4_4_4_upstream_path_analysis.json"
    )
    for path in (dataset, exploratory_analysis, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required preregistration input is missing: {path}")
    exploratory = json.loads(exploratory_analysis.read_text(encoding="utf-8"))
    old_seeds = set(int(seed) for seed in exploratory["config"]["evaluation_seeds"])
    new_seeds = set(config.evaluation_seeds)
    if old_seeds & new_seeds:
        raise RuntimeError("Independent confirmation overlaps exploratory seeds")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_upstream_confirmation_preregistration_v1",
        "registered_before_model_stage": True,
        "config": config.to_dict(),
        "primary_hypothesis": (
            "early V4.4.2 broad-retrieval top-4 slot-state patch has positive donor "
            "log-odds gain and the induced effect is specifically mediated by the "
            "L28 H16--H19 pre-O Z channel relative to a same-span equal-norm "
            "orthogonal control"
        ),
        "primary_decision_rule": (
            "intersection-union: early mean > 0 and natural-vs-orthogonal mediation "
            "mean > 0, with max(two-sided exact paired sign-flip p) < 0.05"
        ),
        "secondary_family": (
            "paired full-minus-leave-one-out mediation decrement; Holm across H16,H17,H18,H19"
        ),
        "exploratory_seed_registry": sorted(old_seeds),
        "confirmation_seed_registry": list(config.evaluation_seeds),
        "sha256": {
            "confirmation_config": _sha256(config_path),
            "frozen_dataset": _sha256(dataset),
            "exploratory_analysis": _sha256(exploratory_analysis),
        },
        "exploratory_analysis_path": str(exploratory_analysis),
    }
    path = run_root / PREREGISTRATION_NAME
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("Preregistered design or source hashes changed")
    else:
        atomic_json(path, payload)


def _load_campaign_inputs(args: argparse.Namespace):
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    resolved_base = run_root / "resolved_config.json"
    dataset_complete = run_root / "dataset" / "complete.json"
    if not resolved_base.is_file() or not dataset_complete.is_file():
        raise FileNotFoundError(
            "The isolated derivative run must be initialized and its dataset frozen"
        )
    base_config = V444Config.from_json(resolved_base)
    config_path = _resolve(repo_root, args.config)
    config = V444UpstreamConfirmationConfig.from_json(config_path)
    config.validate_against_base(base_config)
    _register_or_verify_preregistration(
        run_root=run_root,
        exploratory_run_root=Path(args.exploratory_run_root).resolve(),
        config_path=config_path,
        config=config,
    )
    return repo_root, run_root, base_config, config


def _run_models(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    base_config: V444Config,
    config: V444UpstreamConfirmationConfig,
    run_primary: bool,
    run_loo: bool,
) -> None:
    base_v4 = V4Config.from_json(_resolve(repo_root, args.v4_config))
    v4_config = v444_v4_config(base_v4, base_config)
    model, tokenizer, adapter = _load_model(
        config.model_label,
        config=_direction_config(base_config),
        cache_dir=args.cache_dir,
        device_map=args.device_map,
    )
    model.eval()
    try:
        if run_primary:
            _write_status(run_root, "RUNNING", stage="primary_full_h16_h19")
            run_stage(
                model,
                tokenizer,
                adapter,
                run_root=run_root,
                base_config=base_config,
                config=config,
                v4_config=v4_config,
                expanded=False,
                resume=args.resume,
            )
        if run_loo:
            _write_status(run_root, "RUNNING", stage="leave_one_out")
            run_stage(
                model,
                tokenizer,
                adapter,
                run_root=run_root,
                base_config=base_config,
                config=config,
                v4_config=v4_config,
                expanded=True,
                resume=args.resume,
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    repo_root, run_root, base_config, config = _load_campaign_inputs(args)
    if args.stage in {"model-primary", "campaign"}:
        _run_models(
            args=args,
            repo_root=repo_root,
            run_root=run_root,
            base_config=base_config,
            config=config,
            run_primary=True,
            run_loo=args.stage == "campaign",
        )
        if args.stage == "model-primary":
            return 0
    if args.stage == "model-loo":
        _run_models(
            args=args,
            repo_root=repo_root,
            run_root=run_root,
            base_config=base_config,
            config=config,
            run_primary=False,
            run_loo=True,
        )
        return 0
    if args.stage in {"analyze", "campaign"}:
        _write_status(run_root, "RUNNING", stage="analysis")
        payload = analyze_campaign(run_root, config=config)
        decision = payload["primary_decision"]
        audit = payload["audit"]
        _write_status(
            run_root,
            "COMPLETE" if audit["all_checks_pass"] else "AUDIT_FAILED",
            primary_decision=decision,
            audit=audit,
        )
        print(json.dumps(decision, ensure_ascii=False, indent=2))
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
            root = Path(parsed.run_root).resolve()
            if root.is_dir():
                _write_status(
                    root,
                    "FAILED",
                    error_type=type(error).__name__,
                    error=str(error),
                    traceback=traceback.format_exc(),
                )
        except Exception:
            pass
        traceback.print_exc()
        raise SystemExit(1)
