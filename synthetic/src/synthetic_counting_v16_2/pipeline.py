from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .analysis import run_v16_2_attention_analysis, run_v16_2_state_analysis
from .config import V16_2Config, config_from_dict, prepare_run_dir
from .data import (
    CorpusSplit,
    V16_2Example,
    V16_2Vocab,
    build_corpus_split,
    build_loss_suite_manifests,
    build_test_suite_manifests,
    load_corpus_split,
    load_corpus_text,
    load_suite_manifests,
    save_corpus_split,
    save_suite_manifests,
)
from .needle_pool import (
    NeedlePool,
    build_needle_pool,
    load_needle_pool,
    plot_needle_pool,
    save_needle_pool,
)
from .plots import make_all_v16_2_plots
from .training import sync_tree, train_v16_2_models
from .timing import timed_event


STAGES = ("prepare", "train", "attention", "state", "plots")


def _stage_list(stage: str | Iterable[str]) -> list[str]:
    values = [part.strip() for part in stage.split(",")] if isinstance(stage, str) else [str(part).strip() for part in stage]
    values = [value for value in values if value]
    if "all" in values:
        return list(STAGES)
    invalid = sorted(set(values) - set(STAGES))
    if invalid:
        raise ValueError(f"unknown stages {invalid}; choose from {STAGES} or all")
    return values


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _update_manifest(run_dir: Path, cfg: V16_2Config, stage: str, status: str) -> None:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "version": cfg.version,
        "preset": cfg.preset,
        "seed": cfg.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    now = datetime.now(timezone.utc).isoformat()
    manifest["updated_at_utc"] = now
    manifest["stages"][stage] = {"status": status, "updated_at_utc": now}
    _write_json(manifest, path)


def _prepared_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return (
        run_dir / "data" / "corpus_split.json",
        run_dir / "data" / "needle_pool.json",
        run_dir / "data" / "loss_suite_manifests.json",
    )


def prepare_v16_2_data(
    cfg: V16_2Config,
    vocab: V16_2Vocab,
    text: str,
    run_dir: Path,
) -> tuple[
    CorpusSplit,
    NeedlePool,
    dict[str, dict[str, list[V16_2Example]]],
    dict[str, list[V16_2Example]],
]:
    split_path, pool_path, suites_path = _prepared_paths(run_dir)
    if any(path.exists() for path in (split_path, pool_path, suites_path)):
        if not all(path.exists() for path in (split_path, pool_path, suites_path)):
            raise RuntimeError("prepare artifacts are incomplete; use a fresh run name")
        return load_prepared_v16_2_data(cfg, vocab, text, run_dir)
    split = build_corpus_split(cfg, text)
    save_corpus_split(split, split_path)
    pool = build_needle_pool(cfg, text, split, vocab.fingerprint)
    save_needle_pool(pool, run_dir)
    plot_needle_pool(pool, run_dir / "figures" / "needle_pool_frequency_distribution.png")
    curve_suites = build_loss_suite_manifests(cfg, vocab, text, split, pool)
    test_suites = build_test_suite_manifests(cfg, vocab, text, split, pool)
    save_suite_manifests(
        curve_suites,
        test_suites,
        suites_path,
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    split_rows = []
    for region in (split.train, split.validation, split.test):
        split_rows.append(
            {"region": region.name, "start": region.start, "end": region.end, "length": region.length, "sha256": region.sha256}
        )
    pd.DataFrame(split_rows).to_csv(run_dir / "tables" / "corpus_split.csv", index=False)
    viability_rows = []
    for region in (split.train, split.validation, split.test):
        region_text = text[region.start : region.end]
        for item in pool.sets:
            viability_rows.append(
                {
                    "region": region.name,
                    "set_id": item.set_id,
                    "viable": any(character in region_text for character in item.characters),
                }
            )
    pd.DataFrame(viability_rows).to_csv(
        run_dir / "tables" / "regional_pool_viability.csv", index=False
    )
    manifest_rows = []
    for source, suites in {**curve_suites, "test": test_suites}.items():
        for suite, examples in suites.items():
            manifest_rows.append(
                {
                    "curve_source": source,
                    "suite": suite,
                    "num_examples": len(examples),
                    "task_examples": sum(item.example_kind == "counting_task" for item in examples),
                    "raw_examples": sum(item.example_kind == "raw_lm" for item in examples),
                    "source_region": examples[0].corpus_region,
                }
            )
    pd.DataFrame(manifest_rows).to_csv(run_dir / "tables" / "loss_suite_manifest_summary.csv", index=False)
    return split, pool, curve_suites, test_suites


def load_prepared_v16_2_data(
    cfg: V16_2Config,
    vocab: V16_2Vocab,
    text: str,
    run_dir: Path,
) -> tuple[
    CorpusSplit,
    NeedlePool,
    dict[str, dict[str, list[V16_2Example]]],
    dict[str, list[V16_2Example]],
]:
    split_path, pool_path, suites_path = _prepared_paths(run_dir)
    missing = [str(path) for path in (split_path, pool_path, suites_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "v16_2 preparation is required before training/analysis; missing " + ", ".join(missing)
        )
    split = load_corpus_split(split_path, cfg, text)
    pool = load_needle_pool(
        pool_path,
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    curve_suites, test_suites = load_suite_manifests(
        suites_path,
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    return split, pool, curve_suites, test_suites


def run_v16_2_pipeline(
    cfg: V16_2Config,
    *,
    stage: str | Iterable[str] = "all",
    out_root: str | Path = "runs/synthetic_counting_v16_2",
    run_name: str | None = None,
    checkpoint_sync_root: str | Path | None = None,
    skip_completed: bool = True,
) -> Path:
    cfg.validate()
    run_dir = prepare_run_dir(out_root, cfg, run_name)
    sync_run_dir = Path(checkpoint_sync_root) / run_dir.name if checkpoint_sync_root else None
    if sync_run_dir is not None and sync_run_dir.exists():
        with timed_event(
            run_dir, scope="pipeline", block="drive_restore", device=cfg.device
        ):
            sync_tree(sync_run_dir, run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        saved_cfg = config_from_dict(json.loads(config_path.read_text(encoding="utf-8")))
        if replace(saved_cfg, device=cfg.device) != cfg:
            raise ValueError("run directory contains a different v16_2 config; use a new run name")
    else:
        _write_json(cfg.to_dict(), config_path)
    text = load_corpus_text()
    vocab = V16_2Vocab.build(cfg, text)
    vocab_path = run_dir / "vocab.json"
    if vocab_path.exists():
        if V16_2Vocab.load(vocab_path).fingerprint != vocab.fingerprint:
            raise ValueError("run directory vocabulary does not match this corpus/config")
    else:
        vocab.save(vocab_path)

    prepared = None
    for current in _stage_list(stage):
        print(f"[v16_2] stage={current}", flush=True)
        _update_manifest(run_dir, cfg, current, "running")
        try:
            with timed_event(
                run_dir,
                scope="pipeline",
                block=current,
                device=cfg.device,
            ):
                if current == "prepare":
                    prepared = prepare_v16_2_data(cfg, vocab, text, run_dir)
                else:
                    if prepared is None:
                        prepared = load_prepared_v16_2_data(cfg, vocab, text, run_dir)
                    split, pool, curve_suites, test_suites = prepared
                    if current == "train":
                        train_v16_2_models(
                            cfg,
                            vocab,
                            text,
                            split,
                            pool,
                            curve_suites,
                            test_suites,
                            run_dir,
                            sync_run_dir=sync_run_dir,
                            skip_completed=skip_completed,
                        )
                    elif current == "attention":
                        run_v16_2_attention_analysis(cfg, vocab, run_dir, curve_suites["heldout"]["task"])
                    elif current == "state":
                        run_v16_2_state_analysis(
                            cfg,
                            vocab,
                            run_dir,
                            curve_suites["train"]["task"],
                            curve_suites["heldout"]["task"],
                        )
                    elif current == "plots":
                        make_all_v16_2_plots(cfg, run_dir)
        except Exception:
            _update_manifest(run_dir, cfg, current, "failed")
            if sync_run_dir is not None:
                with timed_event(
                    run_dir,
                    scope="pipeline",
                    block=f"drive_failure_sync_{current}",
                    device=cfg.device,
                ):
                    sync_tree(run_dir, sync_run_dir)
            raise
        _update_manifest(run_dir, cfg, current, "complete")
        if sync_run_dir is not None:
            with timed_event(
                run_dir,
                scope="pipeline",
                block=f"drive_sync_{current}",
                device=cfg.device,
            ):
                sync_tree(run_dir, sync_run_dir)
    print(f"FINAL_RUN_DIR={run_dir.resolve()}", flush=True)
    return run_dir
