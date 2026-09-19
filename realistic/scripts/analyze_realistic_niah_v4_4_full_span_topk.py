from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCHEMA = "realistic_niah_v4_4_full_span_topk_analysis_v2"
MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _exact_sign_flip_p(values: Iterable[float], *, chunk_size: int = 65_536) -> float:
    """Truly enumerate every sign assignment for at most twenty seed effects."""

    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if not 1 <= len(vector) <= 20:
        raise ValueError("Exact sign-flip enumeration requires 1..20 finite values")
    observed = abs(float(vector.mean()))
    extreme = 0
    total = 1 << len(vector)
    bit_positions = np.arange(len(vector), dtype=np.uint64)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        masks = np.arange(start, stop, dtype=np.uint64)[:, None]
        bits = (masks >> bit_positions) & 1
        signs = np.where(bits == 0, -1.0, 1.0)
        draws = np.abs((signs * vector[None, :]).mean(axis=1))
        extreme += int(np.count_nonzero(draws >= observed - 1e-15))
    return extreme / total


def _holm(values: Iterable[float]) -> list[float]:
    raw = np.asarray(list(values), dtype=float)
    adjusted = np.full(len(raw), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(raw))
    order = finite[np.argsort(raw[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(raw[index]) * (len(order) - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def _seed_mean_bootstrap_ci(
    values: Iterable[float], *, label: str, repetitions: int = 10_000
) -> tuple[float, float]:
    """Bootstrap the equal-seed estimand used by the exact sign-flip test.

    Correct-only eligibility varies by seed, so an equal-seed mean and a pooled
    eligible-example mean are different estimands.  The primary report treats
    seed as the independent replication unit; its confidence interval must
    therefore resample the twenty seed effects rather than reuse the pooled-row
    interval emitted by the generic dual-population summary.
    """

    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if not len(vector):
        raise ValueError("Seed bootstrap requires at least one finite effect")
    stable_seed = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(stable_seed)
    indices = rng.integers(0, len(vector), size=(int(repetitions), len(vector)))
    distribution = vector[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def _stage_root(run_root: Path, model: str) -> Path:
    family = (
        run_root
        / model
        / "numeric"
        / "causal_v2"
        / "answer_query_head_ablation"
    )
    candidates = sorted(family.glob("confirmation_*/complete.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one complete confirmation for {model}; got {len(candidates)}")
    return candidates[0].parent


def _seed_effects(detail: pd.DataFrame, *, model: str) -> pd.DataFrame:
    detail = detail.copy()
    detail["seed"] = pd.to_numeric(detail["seed"], errors="raise").astype(int)
    detail["top_n"] = pd.to_numeric(detail["top_n"], errors="raise").astype(int)
    detail["generated_count_shift"] = pd.to_numeric(
        detail["generated_count_shift"], errors="coerce"
    ).fillna(0.0)
    detail["baseline_correct"] = _as_bool(detail["baseline_is_correct"])
    detail["baseline_valid"] = _as_bool(detail["baseline_format_valid"])
    detail["patched_correct"] = _as_bool(detail["patched_is_correct"])
    rows: list[dict[str, object]] = []
    for top_n in sorted(detail["top_n"].unique()):
        dose = detail[detail["top_n"].eq(top_n)]
        for seed in sorted(dose["seed"].unique()):
            seed_rows = dose[dose["seed"].eq(seed)]
            ranked = seed_rows[seed_rows["condition"].astype(str).eq("ranked")]
            random = seed_rows[
                seed_rows["condition"].astype(str).eq("layer_matched_random")
            ]
            if len(ranked) != 5 or len(random) != 15:
                raise RuntimeError(
                    f"Unexpected rows for {model} K={top_n} seed={seed}: "
                    f"ranked={len(ranked)} random={len(random)}"
                )
            all_effect = float(ranked["generated_count_shift"].abs().mean()) - float(
                random["generated_count_shift"].abs().mean()
            )
            ranked_clean = ranked[ranked["baseline_correct"] & ranked["baseline_valid"]]
            random_clean = random[random["baseline_correct"] & random["baseline_valid"]]
            clean_ids = set(ranked_clean["stimulus_id"].astype(str))
            if clean_ids != set(random_clean["stimulus_id"].astype(str)):
                raise RuntimeError("Ranked/random clean-correct populations are misaligned")
            if not clean_ids:
                raise RuntimeError(f"No clean-correct examples for {model} seed={seed}")
            clean_effect = float((~ranked_clean["patched_correct"]).mean()) - float(
                (~random_clean["patched_correct"]).mean()
            )
            rows.extend(
                (
                    {
                        "model_label": model,
                        "top_n": int(top_n),
                        "seed": int(seed),
                        "analysis_population": "all_examples_signed",
                        "primary_metric": "ranked_minus_random_absolute_count_shift",
                        "primary_seed_effect": all_effect,
                        "examples": len(ranked),
                    },
                    {
                        "model_label": model,
                        "top_n": int(top_n),
                        "seed": int(seed),
                        "analysis_population": "clean_correct_only",
                        "primary_metric": "ranked_minus_random_correct_to_wrong",
                        "primary_seed_effect": clean_effect,
                        "examples": len(clean_ids),
                    },
                )
            )
    return pd.DataFrame(rows)


def analyze(run_root: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_frames: list[pd.DataFrame] = []
    dual_frames: list[pd.DataFrame] = []
    membership_rows: list[dict[str, object]] = []
    inputs: dict[str, object] = {}
    for model in MODELS:
        stage = _stage_root(run_root, model)
        detail_path = stage / "detail.csv.gz"
        dual_path = stage / "analysis" / "dual_population_seed_extrapolation_summary.csv"
        ranking_path = run_root / "inputs" / f"{model}.full_span_head_rankings.json"
        detail = pd.read_csv(detail_path, compression="gzip", low_memory=False)
        dual = pd.read_csv(dual_path, low_memory=False)
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        if ranking.get("mass_definition") != (
            "sum of answer-query attention over every token in each active needle span"
        ):
            raise RuntimeError(f"{model} registry is not full-span literal mass")
        bank = ranking["rankings"]["broad_aggregation"]
        top_ns = [
            int(value)
            for value in sorted(
                pd.to_numeric(dual["top_n"], errors="raise").astype(int).unique()
            )
        ]
        for top_n in top_ns:
            selected = bank[:top_n]
            for row in selected:
                membership_rows.append(
                    {
                        "model_label": model,
                        "top_n": int(top_n),
                        "rank": int(row["rank"]),
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "head_label": f"L{int(row['layer'])}H{int(row['head'])}",
                    }
                )
        seed_frames.append(_seed_effects(detail, model=model))
        dual_frames.append(dual)
        inputs[model] = {
            "stage_root": str(stage),
            "detail_sha256": _sha256(detail_path),
            "dual_summary_sha256": _sha256(dual_path),
            "ranking_sha256": _sha256(ranking_path),
            "top_ns": top_ns,
        }

    seed_effects = pd.concat(seed_frames, ignore_index=True)
    dual = pd.concat(dual_frames, ignore_index=True)
    summaries: list[dict[str, object]] = []
    for (model, top_n, population, metric), group in seed_effects.groupby(
        ["model_label", "top_n", "analysis_population", "primary_metric"],
        sort=True,
    ):
        values = group.sort_values("seed")["primary_seed_effect"].to_numpy(dtype=float)
        if len(values) != 20:
            raise RuntimeError(f"Expected 20 seed effects for {model} K={top_n} {population}")
        dual_row = dual[
            dual["model_label"].astype(str).eq(model)
            & pd.to_numeric(dual["top_n"], errors="raise").astype(int).eq(int(top_n))
            & dual["analysis_population"].astype(str).eq(population)
        ]
        if len(dual_row) != 1:
            raise RuntimeError("Could not align dual-population summary")
        record = dual_row.iloc[0]
        effect = float(values.mean())
        ci_low, ci_high = _seed_mean_bootstrap_ci(
            values,
            label=f"{model}:K{int(top_n)}:{population}:equal-seed-mean",
        )
        pooled_example_effect = float(record["primary_effect"])
        summaries.append(
            {
                "model_label": model,
                "top_n": int(top_n),
                "analysis_population": population,
                "primary_metric": metric,
                "primary_effect": effect,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "pooled_eligible_example_effect": pooled_example_effect,
                "equal_seed_minus_pooled_effect": effect - pooled_example_effect,
                "exact_sign_flip_p": _exact_sign_flip_p(values),
                "seed_clusters": len(values),
                "positive_seed_fraction": float(np.mean(values > 0)),
                "nonnegative_seed_fraction": float(np.mean(values >= 0)),
            }
        )
    statistics = pd.DataFrame(summaries).sort_values(
        ["analysis_population", "model_label", "top_n"]
    ).reset_index(drop=True)
    statistics["holm_p_within_primary_endpoint_12_tests"] = np.nan
    for population, indices in statistics.groupby("analysis_population").groups.items():
        statistics.loc[indices, "holm_p_within_primary_endpoint_12_tests"] = _holm(
            statistics.loc[indices, "exact_sign_flip_p"]
        )
    statistics["significant_raw_0_05"] = statistics["exact_sign_flip_p"] <= 0.05
    statistics["significant_holm_0_05"] = (
        statistics["holm_p_within_primary_endpoint_12_tests"] <= 0.05
    )

    seed_path = output_dir / "full_span_topk_seed_effects.csv"
    stats_path = output_dir / "full_span_topk_primary_statistics.csv"
    membership_frame = pd.DataFrame(membership_rows)
    membership_path = output_dir / "full_span_topk_membership.csv"
    seed_effects.to_csv(seed_path, index=False)
    statistics.to_csv(stats_path, index=False)
    membership_frame.to_csv(membership_path, index=False)
    report_models: dict[str, dict[str, object]] = {}
    for model in MODELS:
        doses: dict[str, object] = {}
        for top_n in sorted(statistics.loc[statistics["model_label"].eq(model), "top_n"].unique()):
            rows = statistics[
                statistics["model_label"].eq(model)
                & statistics["top_n"].eq(int(top_n))
            ]
            all_row = rows[rows["analysis_population"].eq("all_examples_signed")].iloc[0]
            clean_row = rows[rows["analysis_population"].eq("clean_correct_only")].iloc[0]
            heads = membership_frame[
                membership_frame["model_label"].eq(model)
                & membership_frame["top_n"].eq(int(top_n))
            ].sort_values("rank")["head_label"].tolist()
            doses[str(int(top_n))] = {
                "heads": heads,
                "all_absolute_shift": {
                    "effect": float(all_row["primary_effect"]),
                    "ci95_low": float(all_row["ci95_low"]),
                    "ci95_high": float(all_row["ci95_high"]),
                    "pooled_eligible_example_effect": float(
                        all_row["pooled_eligible_example_effect"]
                    ),
                    "two_sided_exact_seed_sign_flip_p": float(all_row["exact_sign_flip_p"]),
                    "holm_p_across_twelve_frozen_sets": float(
                        all_row["holm_p_within_primary_endpoint_12_tests"]
                    ),
                },
                "clean_correct_to_wrong": {
                    "effect": float(clean_row["primary_effect"]),
                    "ci95_low": float(clean_row["ci95_low"]),
                    "ci95_high": float(clean_row["ci95_high"]),
                    "pooled_eligible_example_effect": float(
                        clean_row["pooled_eligible_example_effect"]
                    ),
                    "two_sided_exact_seed_sign_flip_p": float(clean_row["exact_sign_flip_p"]),
                    "holm_p_across_twelve_frozen_sets": float(
                        clean_row["holm_p_within_primary_endpoint_12_tests"]
                    ),
                },
            }
        report_models[model] = doses
    audit_path = (
        run_root
        / "audit"
        / "ablation_seed_extrapolation"
        / "ablation_seed_extrapolation_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    report_summary = {
        "schema_version": "realistic_niah_v4_4_full_span_topk_report_summary_v2",
        "run_root": str(run_root),
        "ranking_definition": {
            "mass": "sum of answer-query attention over every token in each active needle span",
            "score": "mean(broad_mass * exp(entropy(per-needle mass))/needle_count)",
            "query": "answer_query",
        },
        "design": {
            "seeds": list(range(1316, 1336)),
            "counts": [1, 2, 3, 4, 5],
            "examples_per_model": 100,
            "head_bank": "broad_aggregation",
            "frozen_top_n": {model: [1, 2, 4, 8, 16, 32] for model in MODELS},
            "layer_matched_random_replicates": 3,
            "bootstrap_repetitions": 10_000,
            "exact_sign_flip_assignments_per_test": 2**20,
            "holm_family_size_per_primary_endpoint": 12,
        },
        "audit": audit,
        "models": report_models,
    }
    report_summary_path = output_dir / "seed_extrapolation_summary_v2.json"
    report_summary_path.write_text(
        json.dumps(report_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    significant_holm = [
        {
            "model_label": str(row["model_label"]),
            "top_n": int(row["top_n"]),
            "analysis_population": str(row["analysis_population"]),
            "primary_effect": float(row["primary_effect"]),
            "holm_p_within_primary_endpoint_12_tests": float(
                row["holm_p_within_primary_endpoint_12_tests"]
            ),
        }
        for _, row in statistics.loc[statistics["significant_holm_0_05"]].iterrows()
    ]
    summary = {
        "schema_version": SCHEMA,
        "status": "complete",
        "ranking_definition": {
            "mass": "full needle-span literal attention mass",
            "score": "mean(broad_mass * occurrence coverage)",
            "query": "answer query",
        },
        "top_ns": sorted(statistics["top_n"].unique().astype(int).tolist()),
        "models": list(MODELS),
        "primary_endpoint_families": 2,
        "tests_per_primary_endpoint_family": 12,
        "exact_sign_flip_assignments_per_test": 2**20,
        "inputs": inputs,
        "outputs": {
            "seed_effects": str(seed_path),
            "primary_statistics": str(stats_path),
            "membership": str(membership_path),
            "report_summary": str(report_summary_path),
        },
        "significant_holm": significant_holm,
    }
    summary_path = output_dir / "full_span_topk_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_root / "analysis" / "full_span_topk"
    )
    print(json.dumps(analyze(run_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
