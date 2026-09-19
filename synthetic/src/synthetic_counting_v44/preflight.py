from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import (
    V20Vocab,
    build_corpus_split,
    load_corpus_text,
)
from synthetic_counting_v20.needle_pool import build_needle_pool
from synthetic_counting_v20.pipeline import load_prepared_v20_data
from synthetic_counting_v20.training import (
    _JOINT_SET_COUNT_SAMPLER_CACHE,
    _joint_set_count_sampler,
)
from synthetic_counting_v35.config import preset_config as v35_preset_config


def run_preflight(
    run_dir: str | Path,
    *,
    expected_version: str = "v44",
) -> dict[str, Any]:
    """Audit the fixed count-1-to-10 data before either model is trained."""

    root = Path(run_dir)
    cfg = config_from_dict(json.loads((root / "config.json").read_text(encoding="utf-8")))
    if cfg.version != expected_version:
        raise ValueError(f"expected a {expected_version} run, got {cfg.version!r}")
    if cfg.count_max_threshold != 10 or cfg.trace_format != "separator":
        raise ValueError("v44 preflight requires counts 1..10 and separator trace")
    if cfg.joint_sampler_max_starts_per_cell is not None:
        raise ValueError("v44 preflight requires full within-cell support")

    text = load_corpus_text()
    vocab = V20Vocab.build(cfg, text)
    split, pool, _, _ = load_prepared_v20_data(cfg, vocab, text, root)
    joint = _joint_set_count_sampler(cfg, text, split, pool)
    plan = joint.plan.copy()
    feasible = plan[plan["feasible"]].copy()
    if feasible.empty:
        raise ValueError("v44 sampler has no feasible set x count cells")
    if not feasible["full_window_count"].eq(feasible["retained_window_count"]).all():
        raise ValueError("v44 full-support audit found retained-window truncation")
    if not plan["within_cell_sampling_policy"].eq("all_legal_starts").all():
        raise ValueError("v44 sampler policy is not all_legal_starts")
    observed_counts = sorted(feasible["count"].astype(int).unique().tolist())
    if observed_counts != list(range(1, 11)):
        raise ValueError(f"v44 feasible support does not cover 1..10: {observed_counts}")

    pivot = plan.pivot(
        index="set_id", columns="count", values="target_probability"
    ).fillna(0.0)
    if pivot.columns.astype(int).tolist() != list(range(1, 11)):
        raise ValueError("v44 maximum-entropy target does not contain counts 1..10")
    probability = pivot.to_numpy(dtype=float)
    set_marginal = probability.sum(axis=1, keepdims=True)
    count_marginal = probability.sum(axis=0, keepdims=True)
    mask = probability > 0
    mutual_information_bits = float(
        (
            probability[mask]
            * np.log2(
                (probability / (set_marginal @ count_marginal))[mask]
            )
        ).sum()
    )
    set_only_bayes_accuracy = float(probability.max(axis=1).sum())
    if mutual_information_bits >= 0.07:
        raise ValueError(f"set-count MI is too high: {mutual_information_bits}")
    if set_only_bayes_accuracy >= 0.12:
        raise ValueError(
            f"set-only Bayes count accuracy is too high: {set_only_bayes_accuracy}"
        )

    baseline = v35_preset_config("main", seed=cfg.seed, device=cfg.device)
    baseline_split = build_corpus_split(baseline, text)
    baseline_vocab = V20Vocab.build(baseline, text)
    baseline_pool = build_needle_pool(
        baseline,
        text,
        baseline_split,
        baseline_vocab.fingerprint,
    )
    marker_sets_identical_to_v35 = [item.characters for item in pool.sets] == [
        item.characters for item in baseline_pool.sets
    ]
    if not marker_sets_identical_to_v35:
        raise ValueError("v44 marker sets differ from v35")

    audit_path = root / "tables" / "pretraining_full_support_audit.csv"
    plan.to_csv(audit_path, index=False)
    result = {
        "version": cfg.version,
        "count_support": observed_counts,
        "trace_format": cfg.trace_format,
        "marker_set_count": len(pool.sets),
        "marker_sets_identical_to_v35": marker_sets_identical_to_v35,
        "feasible_cells": int(len(feasible)),
        "full_legal_starts": int(feasible["full_window_count"].sum()),
        "retained_legal_starts": int(feasible["retained_window_count"].sum()),
        "retention_fraction": 1.0,
        "within_cell_sampling_policy": "all_legal_starts",
        "maxent_target_set_count_mi_bits": mutual_information_bits,
        "maxent_target_set_only_bayes_accuracy": set_only_bayes_accuracy,
        "count_chance_accuracy": 0.10,
        "max_count_marginal_error_from_uniform": float(
            np.abs(count_marginal.ravel() - 0.10).max()
        ),
        "max_set_marginal_error_from_uniform": float(
            np.abs(set_marginal.ravel() - 0.01).max()
        ),
        "audit_table": str(audit_path.resolve()),
        "passed": True,
    }
    safe_version = expected_version.replace(".", "_")
    output = root / "analysis" / f"preflight_{safe_version}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    _JOINT_SET_COUNT_SAMPLER_CACHE.clear()
    del joint
    gc.collect()
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit v44 before optimization")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    result = run_preflight(args.run_dir)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
