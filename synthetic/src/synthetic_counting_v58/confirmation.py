from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import (
    V20Example,
    V20Vocab,
    balanced_v20_examples,
    example_to_dict,
    load_corpus_split,
    load_corpus_text,
    load_suite_manifests,
)
from synthetic_counting_v20.needle_pool import load_needle_pool
from synthetic_counting_v20.training import (
    autoregressive_task_evaluation,
    load_v20_checkpoint_model,
)
from synthetic_counting_v58.behavior_gate import GATE_THRESHOLDS


DEFAULT_EXAMPLES_PER_COUNT = 200
DEFAULT_SEED_OFFSET = 181_000
OUTPUT_SCHEMA_VERSION = "v58_independent_behavior_confirmation_v1"


def _example_key(example: V20Example) -> tuple[str | None, int]:
    """Identify a task before its stochastic context permutation is rendered."""

    return example.set_id, int(example.corpus_start)


def select_disjoint_balanced(
    candidates: Iterable[V20Example],
    excluded: Iterable[V20Example],
    *,
    count_max: int,
    examples_per_count: int,
) -> list[V20Example]:
    """Select a balanced suite with no canonical-test (set, start) overlap."""

    excluded_keys = {_example_key(example) for example in excluded}
    buckets: dict[int, list[V20Example]] = {
        count: [] for count in range(1, count_max + 1)
    }
    selected_keys: set[tuple[str | None, int]] = set()
    for example in candidates:
        if example.count is None or int(example.count) not in buckets:
            continue
        key = _example_key(example)
        bucket = buckets[int(example.count)]
        if (
            key in excluded_keys
            or key in selected_keys
            or len(bucket) >= examples_per_count
        ):
            continue
        bucket.append(example)
        selected_keys.add(key)
    missing = {
        count: examples_per_count - len(values)
        for count, values in buckets.items()
        if len(values) != examples_per_count
    }
    if missing:
        raise RuntimeError(
            "independent confirmation candidates did not fill every count bucket: "
            f"{missing}"
        )
    return [example for count in sorted(buckets) for example in buckets[count]]


def build_confirmation_examples(
    cfg: Any,
    vocab: V20Vocab,
    text: str,
    split: Any,
    pool: Any,
    canonical_test: list[V20Example],
    *,
    examples_per_count: int,
    seed: int,
) -> list[V20Example]:
    """Generate a deterministic balanced test suite disjoint from the canonical one."""

    excluded_by_count = pd.Series(
        [int(example.count) for example in canonical_test]
    ).value_counts()
    maximum_excluded = int(excluded_by_count.max()) if len(excluded_by_count) else 0
    candidates = balanced_v20_examples(
        cfg,
        vocab,
        text,
        split,
        pool,
        examples_per_count + maximum_excluded,
        seed,
        region_name="test",
    )
    selected = select_disjoint_balanced(
        candidates,
        canonical_test,
        count_max=cfg.count_max_threshold,
        examples_per_count=examples_per_count,
    )
    random.Random(seed + 1).shuffle(selected)
    return selected


def _wilson_interval(successes: float, observations: int) -> tuple[float, float]:
    z = 1.959963984540054
    estimate = successes / observations
    denominator = 1.0 + z * z / observations
    center = (estimate + z * z / (2 * observations)) / denominator
    radius = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / observations
            + z * z / (4 * observations * observations)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_confirmation(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_modes = {"nonthinking", "thinking"}
    if set(detail["mode"].unique()) != required_modes:
        raise ValueError("confirmation detail must contain exactly both modes")

    by_count_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for mode in ("nonthinking", "thinking"):
        mode_frame = detail[detail["mode"].eq(mode)]
        count_sizes = mode_frame.groupby("count").size()
        if count_sizes.nunique() != 1:
            raise ValueError(f"{mode} confirmation suite is not count-balanced")
        for count, group in mode_frame.groupby("count", sort=True):
            low, high = _wilson_interval(float(group["ar_accuracy"].sum()), len(group))
            row = {
                "mode": mode,
                "count": int(count),
                "examples": int(len(group)),
                "ar_final_accuracy": float(group["ar_accuracy"].mean()),
                "ar_final_accuracy_wilson95_low": low,
                "ar_final_accuracy_wilson95_high": high,
                "ar_answer_rate": float(group["ar_answered"].mean()),
            }
            for name in (
                "trace_exact",
                "trace_ordered_marker_accuracy",
                "trace_marker_count_accuracy",
                "trace_format_valid",
                "trace_closed",
                "trace_delimiter_count_accuracy",
            ):
                if mode == "thinking" and name in group:
                    row[name] = float(group[name].mean())
            by_count_rows.append(row)

        low, high = _wilson_interval(
            float(mode_frame["ar_accuracy"].sum()), len(mode_frame)
        )
        row = {
            "mode": mode,
            "examples": int(len(mode_frame)),
            "examples_per_count": int(count_sizes.iloc[0]),
            "ar_final_accuracy": float(mode_frame["ar_accuracy"].mean()),
            "ar_final_accuracy_wilson95_low": low,
            "ar_final_accuracy_wilson95_high": high,
            "ar_answer_rate": float(mode_frame["ar_answered"].mean()),
        }
        for name in (
            "trace_exact",
            "trace_ordered_marker_accuracy",
            "trace_marker_count_accuracy",
            "trace_format_valid",
            "trace_closed",
            "trace_delimiter_count_accuracy",
        ):
            if mode == "thinking" and name in mode_frame:
                row[name] = float(mode_frame[name].mean())
        summary_rows.append(row)

    by_count = pd.DataFrame(by_count_rows)
    summary = pd.DataFrame(summary_rows)
    thinking_accuracy = float(
        summary.loc[summary["mode"].eq("thinking"), "ar_final_accuracy"].iloc[0]
    )
    nonthinking_accuracy = float(
        summary.loc[summary["mode"].eq("nonthinking"), "ar_final_accuracy"].iloc[0]
    )
    thinking_counts = by_count[by_count["mode"].eq("thinking")]
    per_count = {
        str(int(row["count"])): float(row["ar_final_accuracy"])
        for _, row in thinking_counts.iterrows()
    }
    metrics = {
        "thinking_accuracy": thinking_accuracy,
        "nonthinking_accuracy": nonthinking_accuracy,
        "thinking_minus_nonthinking_gap": thinking_accuracy - nonthinking_accuracy,
        "thinking_min_count_accuracy": min(per_count.values()),
        "thinking_count_spread": max(per_count.values()) - min(per_count.values()),
        "thinking_trace_exact": float(
            summary.loc[summary["mode"].eq("thinking"), "trace_exact"].iloc[0]
        ),
        "thinking_per_count_accuracy": per_count,
        "nonthinking_per_count_accuracy": {
            str(int(row["count"])): float(row["ar_final_accuracy"])
            for _, row in by_count[by_count["mode"].eq("nonthinking")].iterrows()
        },
    }
    checks = {
        "thinking_accuracy": metrics["thinking_accuracy"]
        >= GATE_THRESHOLDS["thinking_accuracy_min"],
        "thinking_min_count_accuracy": metrics["thinking_min_count_accuracy"]
        >= GATE_THRESHOLDS["thinking_min_count_accuracy_min"],
        "thinking_count_spread": metrics["thinking_count_spread"]
        <= GATE_THRESHOLDS["thinking_count_spread_max"],
        "thinking_minus_nonthinking_gap": metrics["thinking_minus_nonthinking_gap"]
        >= GATE_THRESHOLDS["thinking_minus_nonthinking_gap_min"],
    }
    gate = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "thresholds": dict(GATE_THRESHOLDS),
        "metrics": metrics,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "trace_policy": "diagnostic only; not a behavioral gate",
    }
    return summary, by_count, gate


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_confirmation(
    run_dir: Path,
    *,
    device: str,
    examples_per_count: int,
    seed_offset: int,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text()))
    if cfg.version != "v58":
        raise ValueError(f"v58 confirmation requires version='v58', found {cfg.version!r}")
    cfg = replace(cfg, device=device)
    vocab = V20Vocab.load(run_dir / "vocab.json")
    corpus = load_corpus_text()
    split = load_corpus_split(run_dir / "data/corpus_split.json", cfg, corpus)
    pool = load_needle_pool(
        run_dir / "data/needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    _, test_suites = load_suite_manifests(
        run_dir / "data/loss_suite_manifests.json",
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    seed = int(cfg.seed + seed_offset)
    examples = build_confirmation_examples(
        cfg,
        vocab,
        corpus,
        split,
        pool,
        test_suites["task"],
        examples_per_count=examples_per_count,
        seed=seed,
    )
    output_dir = run_dir / "analysis" / "behavior_confirmation_v58"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.jsonl"
    temporary_examples = examples_path.with_suffix(".jsonl.tmp")
    with temporary_examples.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example_to_dict(example), sort_keys=True) + "\n")
    temporary_examples.replace(examples_path)

    frames: list[pd.DataFrame] = []
    for mode in ("nonthinking", "thinking"):
        _, loaded_vocab, _, _, model = load_v20_checkpoint_model(
            run_dir,
            "rope",
            mode,
            step=cfg.train_steps,
            device=device,
        )
        frame = autoregressive_task_evaluation(
            model,
            cfg,
            loaded_vocab,
            examples,
            position_encoding="rope",
            mode=mode,
            step=cfg.train_steps,
        )
        frames.append(frame)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detail_with_generations = pd.concat(frames, ignore_index=True)
    failures = (
        detail_with_generations[detail_with_generations["ar_accuracy"].lt(1)]
        .sort_values(["mode", "count", "row_id"])
        .groupby(["mode", "count"], as_index=False)
        .head(5)
    )
    detail = detail_with_generations.drop(columns=["generated_tokens"])
    summary, by_count, gate = summarize_confirmation(detail)
    paths = {
        "detail": output_dir / "autoregressive_detail.csv",
        "failures": output_dir / "autoregressive_failures.csv",
        "summary": output_dir / "autoregressive_summary.csv",
        "by_count": output_dir / "autoregressive_by_count.csv",
        "gate": output_dir / "behavior_gate.json",
    }
    _atomic_csv(detail, paths["detail"])
    _atomic_csv(failures, paths["failures"])
    _atomic_csv(summary, paths["summary"])
    _atomic_csv(by_count, paths["by_count"])
    _atomic_json(gate, paths["gate"])
    canonical_keys = {_example_key(example) for example in test_suites["task"]}
    confirmation_keys = {_example_key(example) for example in examples}
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "checkpoint_step": int(cfg.train_steps),
        "modes": ["nonthinking", "thinking"],
        "examples_per_count": int(examples_per_count),
        "total_examples_per_mode": int(len(examples)),
        "generator_seed": seed,
        "seed_offset": int(seed_offset),
        "region": "test",
        "canonical_exclusion_key": ["set_id", "corpus_start"],
        "canonical_test_examples": int(len(test_suites["task"])),
        "canonical_overlap": int(len(canonical_keys & confirmation_keys)),
        "unique_confirmation_keys": int(len(confirmation_keys)),
        "files": {},
    }
    for name, path in {"examples": examples_path, **paths}.items():
        manifest["files"][name] = {
            "path": str(path.relative_to(run_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest, manifest_path)
    return {"gate": gate, "manifest": manifest, "output_dir": str(output_dir)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a disjoint, count-balanced final-checkpoint confirmation for v58"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--examples-per-count", type=int, default=DEFAULT_EXAMPLES_PER_COUNT
    )
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    args = parser.parse_args(argv)
    if args.examples_per_count <= 0:
        parser.error("--examples-per-count must be positive")
    result = run_confirmation(
        args.run_dir,
        device=args.device,
        examples_per_count=args.examples_per_count,
        seed_offset=args.seed_offset,
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"CONFIRMATION_OUTPUT={result['output_dir']}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_EXAMPLES_PER_COUNT",
    "DEFAULT_SEED_OFFSET",
    "build_confirmation_examples",
    "run_confirmation",
    "select_disjoint_balanced",
    "summarize_confirmation",
]
