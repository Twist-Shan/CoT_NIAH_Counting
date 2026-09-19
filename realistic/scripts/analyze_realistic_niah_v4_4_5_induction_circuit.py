from __future__ import annotations

"""Audit and analyze follow-up 22 without pooling models."""

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_by_seed(rows: Sequence[dict], value: Callable[[dict], float]) -> np.ndarray:
    arms: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        arms.setdefault((int(row["seed"]), int(row["gold_count"])), {})[str(row["arm"])] = row
    per_seed: dict[int, list[float]] = {}
    for (seed, _count), group in arms.items():
        if set(group) != {"natural", "candidate_edge_block", "mass_distance_control"}:
            raise RuntimeError("A canonical unit lacks one frozen arm")
        per_seed.setdefault(seed, []).append(
            value(group["candidate_edge_block"]) - value(group["mass_distance_control"])
        )
    return np.asarray([np.mean(per_seed[seed]) for seed in sorted(per_seed)], dtype=float)


def audit_structural_and_relation_rows(
    rows: Sequence[dict],
    *,
    structural_counts: set[int],
    primary_relation_counts: set[int],
) -> None:
    by_unit: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        by_unit.setdefault(
            (int(row["seed"]), int(row["gold_count"])), {}
        )[str(row["arm"])] = row
    expected_arms = {"natural", "candidate_edge_block", "mass_distance_control"}
    compared = (
        "expected_count",
        "strict_absolute_error",
        "retrieval_bank_broad_score_mean",
        "correct_count_margin",
    )
    for (_seed, count), group in by_unit.items():
        if set(group) != expected_arms:
            raise RuntimeError("A canonical unit lacks one frozen arm")
        if count in structural_counts:
            if not all(bool(row.get("structural_no_previous_match")) for row in group.values()):
                raise RuntimeError("N=1 rows are not marked as structural no-match controls")
            if any(
                int(row["registered_edges"]) != 0
                or int(row["reachable_edges"]) != 0
                or int(row["intervention_sites"]) != 0
                for row in group.values()
            ):
                raise RuntimeError("N=1 structural control contains an intervention edge")
            natural = group["natural"]
            for arm in ("candidate_edge_block", "mass_distance_control"):
                if any(
                    abs(float(group[arm][name]) - float(natural[name])) > 1e-9
                    for name in compared
                ):
                    raise RuntimeError("N=1 no-op arms are not numerically identical")
            continue
        if count not in primary_relation_counts:
            raise RuntimeError("Canonical count is outside the frozen analysis registry")
        if any(bool(row.get("structural_no_previous_match")) for row in group.values()):
            raise RuntimeError("A relation-present row is marked as structural no-match")
        expected_edges = count - 1
        if any(
            int(row["registered_edges"]) != expected_edges
            or int(row["reachable_edges"]) != expected_edges
            for row in group.values()
        ):
            raise RuntimeError("A relation-present unit lacks exact registered edge coverage")
        if int(group["natural"]["intervention_sites"]) != 0:
            raise RuntimeError("Natural arm unexpectedly applies an intervention")
        if any(
            int(group[arm]["intervention_sites"]) != expected_edges
            for arm in ("candidate_edge_block", "mass_distance_control")
        ):
            raise RuntimeError("Candidate/control intervention-site coverage differs")


def bootstrap(values: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(int(seed))
    estimates = np.mean(rng.choice(values, size=(int(draws), len(values)), replace=True), axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "positive_seed_fraction": float(np.mean(values > 0)),
        "seed_count": int(len(values)),
    }


def analyze_model(
    root: Path,
    model: str,
    *,
    draws: int,
    structural_counts: set[int],
    primary_relation_counts: set[int],
) -> dict:
    model_root = root / model
    completion = json.loads((model_root / "complete.json").read_text(encoding="utf-8"))
    synthetic = json.loads((model_root / "synthetic_audit.json").read_text(encoding="utf-8"))
    if completion["status"] == "complete_no_retained_head":
        result = {
            "model": model,
            "status": "PASS",
            "scientific_decision": "not_supported",
            "failed_stage": "synthetic_relation_gate",
            "synthetic": synthetic,
            "canonical_rows": 0,
        }
        (model_root / "analysis_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (model_root / "analysis_audit.json").write_text(json.dumps({"status": "PASS", "canonical_expected": False}, indent=2) + "\n", encoding="utf-8")
        return result
    if completion.get("status") != "complete":
        raise RuntimeError(
            f"Unexpected completion status for {model}: {completion.get('status')}"
        )
    rows = load_jsonl(model_root / "detail.jsonl")
    registration_coverage = json.loads(
        (model_root / "canonical_registration_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        registration_coverage.get("status") != "PASS"
        or int(registration_coverage.get("units", -1)) != 100
    ):
        raise RuntimeError("Canonical registration coverage audit did not pass")
    warmup = json.loads(
        (model_root / "canonical_noop_warmup.json").read_text(encoding="utf-8")
    )
    if (
        warmup.get("status") != "PASS"
        or bool(warmup.get("recorded", True))
        or int(warmup.get("intervention_sites", -1)) != 0
        or int(warmup.get("gold_count", -1)) not in structural_counts
        or not bool(warmup.get("source_capture_present"))
    ):
        raise RuntimeError("Discarded canonical no-op warm-up audit did not pass")
    keys = {(row["model_label"], int(row["seed"]), int(row["gold_count"]), row["arm"]) for row in rows}
    if len(rows) != 300 or len(keys) != 300:
        raise RuntimeError(f"{model} expected 300 canonical rows/keys, got {len(rows)}/{len(keys)}")
    if any(not np.isfinite(float(row["expected_count"])) for row in rows):
        raise RuntimeError("Canonical expected count contains nonfinite values")
    audit_structural_and_relation_rows(
        rows,
        structural_counts=structural_counts,
        primary_relation_counts=primary_relation_counts,
    )
    relation_rows = [
        row
        for row in rows
        if int(row["gold_count"]) in primary_relation_counts
    ]
    expected_error = paired_by_seed(relation_rows, lambda row: abs(float(row["expected_count"]) - int(row["gold_count"])))
    strict_error = paired_by_seed(relation_rows, lambda row: float(row["strict_absolute_error"]))
    broad_damage = paired_by_seed(relation_rows, lambda row: -float(row["retrieval_bank_broad_score_mean"]))
    margin_damage = paired_by_seed(relation_rows, lambda row: -float(row["correct_count_margin"]))
    metrics = {
        "expected_absolute_error_candidate_minus_control": bootstrap(expected_error, draws=draws, seed=4452201),
        "strict_absolute_error_candidate_minus_control": bootstrap(strict_error, draws=draws, seed=4452202),
        "broad_score_damage_candidate_minus_control": bootstrap(broad_damage, draws=draws, seed=4452203),
        "correct_margin_damage_candidate_minus_control": bootstrap(margin_damage, draws=draws, seed=4452204),
    }
    full_panel_expected_error = paired_by_seed(
        rows,
        lambda row: abs(float(row["expected_count"]) - int(row["gold_count"])),
    )
    synthetic_pass = bool(synthetic.get("retained_heads"))
    behavior_pass = metrics["expected_absolute_error_candidate_minus_control"]["ci95_low"] > 0
    result = {
        "model": model,
        "status": "PASS",
        "scientific_decision": "supported" if synthetic_pass and behavior_pass else "not_supported",
        "synthetic_relation_gate": synthetic_pass,
        "canonical_matched_block_gate": behavior_pass,
        "canonical_rows": len(rows),
        "primary_relation_rows": len(relation_rows),
        "structural_no_previous_match_rows": len(rows) - len(relation_rows),
        "paired_seed_count": len(expected_error),
        "metrics": metrics,
        "full_panel_expected_absolute_error_candidate_minus_control": bootstrap(
            full_panel_expected_error, draws=draws, seed=4452205
        ),
        "boundary": "Supports or rejects the classical induction label for the frozen primary head; it does not alter the established distributed span-evidence mechanism.",
    }
    (model_root / "analysis_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (model_root / "analysis_audit.json").write_text(json.dumps({"status": "PASS", "rows": len(rows), "unique_keys": len(keys), "registration_coverage_units": int(registration_coverage["units"]), "discarded_noop_warmup": True, "primary_relation_rows": len(relation_rows), "structural_no_previous_match_rows": len(rows) - len(relation_rows), "paired_seeds": len(expected_error)}, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/realistic_niah_v4_4_5_induction_circuit.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    structural_counts = {
        int(value)
        for value in config["canonical"]["structural_no_previous_match_counts"]
    }
    primary_relation_counts = {
        int(value) for value in config["canonical"]["primary_relation_counts"]
    }
    if structural_counts | primary_relation_counts != {
        int(value) for value in config["counts"]
    } or structural_counts & primary_relation_counts:
        raise RuntimeError("Canonical structural/primary count registry is not a partition")
    summaries = [
        analyze_model(
            args.run_root.resolve(),
            model,
            draws=args.bootstrap_draws,
            structural_counts=structural_counts,
            primary_relation_counts=primary_relation_counts,
        )
        for model in args.models
    ]
    combined = {"schema_version": "realistic_niah_v4_4_5_induction_analysis_v1", "status": "PASS", "models": summaries, "models_pooled": False}
    output = args.run_root.resolve() / "analysis_summary.json"
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
