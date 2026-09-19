from __future__ import annotations

"""Audit and summarize V4.4.5 same-forward serial mediation outputs."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def discover_files(roots: Iterable[Path], name: str) -> list[Path]:
    result: list[Path] = []
    for root in roots:
        direct = root / name
        if direct.is_file():
            result.append(direct)
        result.extend(path for path in root.glob(f"*/{name}") if path.is_file())
    unique = sorted(set(path.resolve() for path in result))
    if not unique:
        raise FileNotFoundError(f"No {name} files found under the supplied roots")
    return unique


def as_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite three-coordinate readout, got {value}")
    return array


def exact_sign_flip_p_two_sided(values: np.ndarray) -> tuple[float, int]:
    """Return the exact paired sign-flip p-value over independent units."""

    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or not len(sample) or not np.isfinite(sample).all():
        raise ValueError("Sign-flip input must be a non-empty finite vector")
    if len(sample) > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 units")
    permutations = 1 << len(sample)
    masks = np.arange(permutations, dtype=np.uint64)[:, None]
    bits = (masks >> np.arange(len(sample), dtype=np.uint64)) & 1
    signs = np.where(bits == 1, 1.0, -1.0)
    null_statistics = np.abs((signs @ sample) / len(sample))
    observed = abs(float(sample.mean()))
    p_value = float(np.mean(null_statistics >= observed - 1e-12))
    return p_value, permutations


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize seed-equal means and resample only independent seed clusters.

    Counts are registered repeated conditions within a seed.  They are averaged
    inside each seed before any uncertainty calculation; bootstrap and exact
    sign-flip inference then operate on the resulting seed means.
    """

    raw = np.asarray(values, dtype=float)
    cluster_values = np.asarray(clusters)
    if raw.ndim != 1 or cluster_values.ndim != 1:
        raise ValueError("Cluster bootstrap inputs must be vectors")
    if len(raw) != len(cluster_values):
        raise ValueError("Values and clusters must have equal length")
    if int(draws) <= 0:
        raise ValueError("Bootstrap draws must be positive")

    grouped: dict[Any, list[float]] = {}
    for value, cluster in zip(raw, cluster_values, strict=True):
        if pd.isna(cluster):
            raise ValueError("Cluster identifiers must be defined")
        cluster_key = cluster.item() if hasattr(cluster, "item") else cluster
        grouped.setdefault(cluster_key, []).append(float(value))
    if not grouped:
        raise ValueError("Cluster bootstrap input has no clusters")

    means: list[float] = []
    finite_cluster_sizes: list[int] = []
    undefined_clusters = 0
    for cluster_rows in grouped.values():
        finite = np.asarray(cluster_rows, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            undefined_clusters += 1
            continue
        means.append(float(finite.mean()))
        finite_cluster_sizes.append(int(len(finite)))
    sample = np.asarray(means, dtype=float)
    if not len(sample):
        raise ValueError("Cluster bootstrap input has no finite cluster means")

    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(sample), size=(int(draws), len(sample)))
    bootstrap_means = sample[indices].mean(axis=1)
    sign_flip_p, sign_flip_permutations = exact_sign_flip_p_two_sided(sample)
    finite_raw = int(np.isfinite(raw).sum())
    return {
        "mean": float(sample.mean()),
        "median": float(np.median(sample)),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "units": int(len(sample)),
        "independent_seed_clusters": int(len(sample)),
        "registered_seed_clusters": int(len(grouped)),
        "paired_seed_count_units": int(len(raw)),
        "finite_paired_seed_count_units": finite_raw,
        "undefined_units": int(len(raw) - finite_raw),
        "undefined_clusters": int(undefined_clusters),
        "finite_rows_per_cluster_min": int(min(finite_cluster_sizes)),
        "finite_rows_per_cluster_max": int(max(finite_cluster_sizes)),
        "bootstrap_draws": int(draws),
        "bootstrap_method": "seed-cluster percentile bootstrap",
        "cluster_column": "seed",
        "exact_sign_flip_p_two_sided": sign_flip_p,
        "exact_sign_flip_permutations": int(sign_flip_permutations),
    }


def max_pairwise(values: list[np.ndarray | float]) -> float:
    arrays = [np.atleast_1d(np.asarray(value, dtype=float)) for value in values]
    return float(
        max(
            np.max(np.abs(left - right))
            for index, left in enumerate(arrays)
            for right in arrays[index + 1 :]
        )
    ) if len(arrays) > 1 else 0.0


def expected_error(row: Mapping[str, Any]) -> float:
    return abs(float(row["expected_count"]) - int(row["gold_count"]))


def unit_effects(group: pd.DataFrame) -> dict[str, Any]:
    rows = {str(row["arm"]): row for row in group.to_dict("records")}
    required = {
        "C", "O", "S", "S_Rorth", "S_Raligned", "S_Torth", "S_Taligned",
        "S_Rorth_Torth", "S_Raligned_Torth", "S_Rorth_Taligned",
        "S_Raligned_Taligned",
    }
    if set(rows) != required:
        raise RuntimeError(f"Unit has wrong arms: {sorted(set(rows) ^ required)}")
    error = {name: expected_error(row) for name, row in rows.items()}
    strict = {name: float(row["strict_absolute_error"]) for name, row in rows.items()}
    retrieval_before = {
        name: as_vector(row["retrieval_coordinates_before"])
        for name, row in rows.items()
    }
    late_before = {
        name: as_vector(row["late_coordinates_before"])
        for name, row in rows.items()
    }
    broad = {
        name: float(row["retrieval_bank_broad_score_mean"])
        for name, row in rows.items()
    }
    rorth_variants = ("S_Rorth", "S_Rorth_Torth", "S_Rorth_Taligned")
    raligned_variants = ("S_Raligned", "S_Raligned_Torth", "S_Raligned_Taligned")
    rnone_variants = ("S", "S_Torth", "S_Taligned")
    source_s_variants = tuple(name for name in required if name.startswith("S"))
    source_repair = error["O"] - error["S"]
    remaining_repair = error["O"] - error["S_Raligned_Taligned"]
    accounted = (
        1.0 - remaining_repair / source_repair
        if abs(source_repair) > 1e-12
        else float("nan")
    )
    return {
        "model_label": str(group.iloc[0]["model_label"]),
        "seed": int(group.iloc[0]["seed"]),
        "gold_count": int(group.iloc[0]["gold_count"]),
        "source_repair": source_repair,
        "retrieval_mediation": error["S_Raligned"] - error["S_Rorth"],
        "late_mediation": error["S_Taligned"] - error["S_Torth"],
        "joint_interaction": (
            error["S_Raligned_Taligned"] - error["S_Rorth_Taligned"]
        ) - (
            error["S_Raligned_Torth"] - error["S_Rorth_Torth"]
        ),
        "remaining_repair": remaining_repair,
        "accounted_fraction_unclipped": accounted,
        "source_strict_repair": strict["O"] - strict["S"],
        "retrieval_strict_mediation": strict["S_Raligned"] - strict["S_Rorth"],
        "late_strict_mediation": strict["S_Taligned"] - strict["S_Torth"],
        "source_broad_score_change": broad["S"] - broad["O"],
        "source_retrieval_coordinate_shift_l2": float(
            np.linalg.norm(retrieval_before["S"] - retrieval_before["O"])
        ),
        "retrieval_aligned_late_radius_reduction": float(
            np.linalg.norm(late_before["S_Rorth"])
            - np.linalg.norm(late_before["S_Raligned"])
        ),
        "late_to_retrieval_invariance_max_abs": max(
            max_pairwise([retrieval_before[name] for name in rnone_variants]),
            max_pairwise([retrieval_before[name] for name in rorth_variants]),
            max_pairwise([retrieval_before[name] for name in raligned_variants]),
        ),
        "downstream_to_broad_invariance_max_abs": max_pairwise(
            [broad[name] for name in source_s_variants]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/realistic_niah_v4_4_5_serial_mediation.json"),
    )
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--seeds", type=csv_ints)
    parser.add_argument("--counts", type=csv_ints)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()

    config_path = args.experiment_config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    arms = tuple(str(value) for value in config["arms"])
    seeds = tuple(args.seeds or tuple(int(value) for value in config["confirmation_seeds"]))
    counts = tuple(args.counts or tuple(int(value) for value in config["counts"]))
    roots = [path.resolve() for path in args.run_roots]
    detail_files = discover_files(roots, "detail.jsonl")
    broad_files = discover_files(roots, "broad_metrics.jsonl")
    details = pd.DataFrame(row for path in detail_files for row in read_jsonl(path))
    broad = pd.DataFrame(row for path in broad_files for row in read_jsonl(path))
    details = details[details["model_label"].isin(args.models)].copy()
    broad = broad[broad["model_label"].isin(args.models)].copy()
    key_columns = ["model_label", "seed", "gold_count", "arm"]
    if details.duplicated(key_columns).any():
        raise RuntimeError("Duplicate serial detail keys")
    broad_key = key_columns + ["layer", "head"]
    if broad.duplicated(broad_key).any():
        raise RuntimeError("Duplicate serial broad keys")
    expected = {
        (model, seed, count, arm)
        for model in args.models
        for seed in seeds
        for count in counts
        for arm in arms
    }
    observed = {
        (str(row.model_label), int(row.seed), int(row.gold_count), str(row.arm))
        for row in details.itertuples()
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(f"Serial key coverage differs; missing={missing[:5]}, extra={extra[:5]}")

    hook_failures: list[dict[str, Any]] = []
    norm_tolerance = float(config["realized_norm_relative_tolerance"])
    orth_tolerance = float(config["orthogonality_max_abs_cosine_tolerance"])
    for row in details.to_dict("records"):
        expected_source = 0 if row["source_mode"] == "none" else 2
        failures = []
        if int(row["source_patch_applications"]) != expected_source:
            failures.append("source_applications")
        if int(row["retrieval_applications"]) != 1:
            failures.append("retrieval_applications")
        if int(row["late_applications"]) != 1:
            failures.append("late_applications")
        for prefix in ("retrieval", "late"):
            mode = str(row[f"{prefix}_mode"])
            if mode != "none" and abs(float(row[f"{prefix}_norm_ratio"]) - 1.0) > norm_tolerance:
                failures.append(f"{prefix}_norm")
            if mode == "orthogonal" and float(
                row[f"{prefix}_orthogonality_max_abs_cosine"]
            ) > orth_tolerance:
                failures.append(f"{prefix}_orthogonality")
        if failures:
            hook_failures.append({"key": [row[key] for key in key_columns], "failures": failures})
    if hook_failures:
        raise RuntimeError(f"Hook audit failures: {hook_failures[:3]}")

    expected_broad_rows = sum(
        len(config["stages"][model]["retrieval_heads"])
        * len(seeds) * len(counts) * len(arms)
        for model in args.models
    )
    if len(broad) != expected_broad_rows:
        raise RuntimeError(f"Broad rows {len(broad)} != {expected_broad_rows}")
    unit_rows = [
        unit_effects(group)
        for _key, group in details.groupby(["model_label", "seed", "gold_count"], sort=True)
    ]
    effects = pd.DataFrame(unit_rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Pin LF so retained CSV hashes are stable across Linux analysis nodes and
    # Windows recovery/reanalysis hosts.
    paired_path = output / "paired_serial_effects.csv"
    detail_flat_path = output / "detail_flat.csv"
    effects.to_csv(paired_path, index=False, lineterminator="\n")
    details.to_csv(detail_flat_path, index=False, lineterminator="\n")

    metric_names = [
        "source_repair",
        "retrieval_mediation",
        "late_mediation",
        "joint_interaction",
        "remaining_repair",
        "accounted_fraction_unclipped",
        "source_strict_repair",
        "retrieval_strict_mediation",
        "late_strict_mediation",
        "source_broad_score_change",
        "source_retrieval_coordinate_shift_l2",
        "retrieval_aligned_late_radius_reduction",
        "late_to_retrieval_invariance_max_abs",
        "downstream_to_broad_invariance_max_abs",
    ]
    summary: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_4_5_serial_mediation_summary_v2",
        "status": "PASS",
        "claim_scope": config["claim_scope"],
        "inference": {
            "independent_unit": "seed",
            "within_seed_aggregation": "equal-weight mean over registered counts",
            "between_seed_aggregation": "equal-weight mean over seeds",
            "bootstrap_method": "seed-cluster percentile bootstrap",
            "bootstrap_draws": int(args.bootstrap_draws),
            "secondary_test": "two-sided exact sign-flip over seed means",
        },
        "models": {},
    }
    for model_index, model in enumerate(args.models):
        selected = effects[effects["model_label"] == model]
        model_summary = {
            name: cluster_bootstrap_mean(
                selected[name].to_numpy(dtype=float),
                selected["seed"].to_numpy(),
                draws=int(args.bootstrap_draws),
                seed=20260814 + model_index * 100 + index,
            )
            for index, name in enumerate(metric_names)
        }
        model_summary["ordered_criterion_diagnostics"] = {
            "i_source_changes_retrieval": {
                "source_repair_positive": model_summary["source_repair"]["mean"] > 0,
                "broad_change_mean": model_summary["source_broad_score_change"]["mean"],
                "retrieval_shift_l2_mean": model_summary[
                    "source_retrieval_coordinate_shift_l2"
                ]["mean"],
            },
            "ii_retrieval_precedes_late": {
                "retrieval_mediation_positive": model_summary[
                    "retrieval_mediation"
                ]["mean"] > 0,
                "late_radius_reduction_mean": model_summary[
                    "retrieval_aligned_late_radius_reduction"
                ]["mean"],
            },
            "iii_late_affects_output_not_earlier_readout": {
                "late_mediation_positive": model_summary["late_mediation"]["mean"] > 0,
                "retrieval_invariance_max_over_units": float(
                    selected["late_to_retrieval_invariance_max_abs"].max()
                ),
            },
            "note": "These are preregistered directional diagnostics, not a unique-pathway test.",
        }
        summary["models"][model] = model_summary
    summary_path = output / "serial_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": "realistic_niah_v4_4_5_serial_mediation_analysis_audit_v2",
        "status": "PASS",
        "detail_files": [str(path) for path in detail_files],
        "broad_files": [str(path) for path in broad_files],
        "detail_rows": int(len(details)),
        "unique_detail_keys": int(len(observed)),
        "broad_rows": int(len(broad)),
        "expected_broad_rows": int(expected_broad_rows),
        "paired_units": int(len(effects)),
        "paired_seed_count_units": int(len(effects)),
        "inference_unit": "seed",
        "independent_seed_clusters": int(len(args.models) * len(seeds)),
        "seed_clusters_per_model": int(len(seeds)),
        "counts_per_seed_cluster": int(len(counts)),
        "within_seed_aggregation": "equal-weight mean over registered counts",
        "between_seed_aggregation": "equal-weight mean over seeds",
        "bootstrap_method": "seed-cluster percentile bootstrap",
        "bootstrap_draws": int(args.bootstrap_draws),
        "exact_sign_flip_unit": "seed",
        "input_files": {
            "experiment_config": file_record(config_path),
            "detail": [file_record(path) for path in detail_files],
            "broad": [file_record(path) for path in broad_files],
        },
        "output_files": {
            "paired_serial_effects": file_record(paired_path),
            "detail_flat": file_record(detail_flat_path),
            "serial_summary": file_record(summary_path),
        },
        "hook_failures": 0,
        "models": list(args.models),
        "seeds": list(seeds),
        "counts": list(counts),
        "arms": list(arms),
    }
    (output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
