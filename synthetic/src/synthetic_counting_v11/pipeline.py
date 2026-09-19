from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .analysis import run_attention_analysis, run_state_analysis
from .config import ExperimentConfig, prepare_run_dir, same_experiment_except_version
from .data import Vocab
from .plots import make_all_plots
from .training import sync_tree, train_all_models


STAGES = ("train", "attention", "state", "plots")


def _stage_list(stage: str | Iterable[str]) -> list[str]:
    if isinstance(stage, str):
        values = [part.strip() for part in stage.split(",") if part.strip()]
    else:
        values = [str(part).strip() for part in stage if str(part).strip()]
    if values == ["all"] or "all" in values:
        return list(STAGES)
    invalid = sorted(set(values) - set(STAGES))
    if invalid:
        raise ValueError(f"Unknown stages {invalid}; choose from {STAGES} or all")
    return values


def _write_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)


def _update_manifest(run_dir: Path, cfg: ExperimentConfig, stage: str, status: str) -> None:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "version": cfg.version,
        "preset": cfg.preset,
        "seed": cfg.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    manifest["version"] = cfg.version
    manifest["preset"] = cfg.preset
    manifest["seed"] = cfg.seed
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["stages"][stage] = {
        "status": status,
        "updated_at_utc": manifest["updated_at_utc"],
    }
    _write_json(manifest, path)


def _make_time_to_threshold(run_dir: Path, threshold: float = 0.99) -> None:
    path = run_dir / "tables" / "eval_by_bin.csv"
    if not path.exists() or not path.stat().st_size:
        return
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["position_encoding", "mode", "count_bin"], sort=False):
        position_encoding, mode, count_bin = keys
        for metric in ("tf_final_accuracy", "tf_trace_marker_accuracy", "tf_trace_index_accuracy"):
            valid = group[group[metric].ge(threshold)].sort_values("step")
            rows.append(
                {
                    "position_encoding": position_encoding,
                    "mode": mode,
                    "count_bin": count_bin,
                    "metric": metric,
                    "threshold": threshold,
                    "first_step_at_threshold": int(valid.iloc[0]["step"]) if not valid.empty else None,
                    "reached_threshold": not valid.empty,
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "tables" / "time_to_99.csv", index=False)


def run_pipeline(
    cfg: ExperimentConfig,
    *,
    stage: str | Iterable[str] = "all",
    out_root: str | Path = "runs/synthetic_counting_v11",
    run_name: str | None = None,
    checkpoint_sync_root: str | Path | None = None,
    skip_completed: bool = True,
) -> Path:
    cfg.validate()
    run_dir = prepare_run_dir(out_root, cfg, run_name)
    sync_run_dir = (
        Path(checkpoint_sync_root) / run_dir.name if checkpoint_sync_root is not None else None
    )
    if sync_run_dir is not None and sync_run_dir.exists():
        print(f"[restore] {sync_run_dir} -> {run_dir}", flush=True)
        sync_tree(sync_run_dir, run_dir)

    config_path = run_dir / "config.json"
    requested_config = cfg.to_dict()
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != requested_config and same_experiment_except_version(existing, requested_config):
            print(f"[migrate] run metadata version -> {cfg.version}", flush=True)
            _write_json(requested_config, config_path)
        elif existing != requested_config:
            raise ValueError(
                f"Run directory {run_dir} contains a different config. "
                "Use a new --run-name instead of mixing experiments."
            )
    else:
        _write_json(requested_config, config_path)
    vocab = Vocab.build(cfg)
    vocab_path = run_dir / "vocab.json"
    if vocab_path.exists():
        existing_vocab = Vocab.load(vocab_path)
        if existing_vocab.fingerprint != vocab.fingerprint:
            raise ValueError("Run directory vocabulary does not match the requested experiment")
    else:
        vocab.save(vocab_path)

    selected = _stage_list(stage)
    for current in selected:
        print(f"[{cfg.version}] stage={current}", flush=True)
        _update_manifest(run_dir, cfg, current, "running")
        try:
            if current == "train":
                train_all_models(
                    cfg,
                    vocab,
                    run_dir,
                    sync_run_dir=sync_run_dir,
                    skip_completed=skip_completed,
                )
                _make_time_to_threshold(run_dir)
            elif current == "attention":
                run_attention_analysis(cfg, vocab, run_dir)
            elif current == "state":
                run_state_analysis(cfg, vocab, run_dir)
            elif current == "plots":
                make_all_plots(cfg, run_dir)
        except Exception:
            _update_manifest(run_dir, cfg, current, "failed")
            if sync_run_dir is not None:
                sync_tree(run_dir, sync_run_dir)
            raise
        _update_manifest(run_dir, cfg, current, "complete")
        if sync_run_dir is not None:
            sync_tree(run_dir, sync_run_dir)
    print(f"FINAL_RUN_DIR={run_dir.resolve()}", flush=True)
    return run_dir
