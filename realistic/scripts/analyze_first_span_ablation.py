#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


KS = (1, 2, 4, 8, 16, 32)


def bootstrap_ci(values: np.ndarray, *, seed: int, draws: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def signflip_p(values: np.ndarray, *, seed: int, draws: int = 200000) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    hits = 0
    done = 0
    while done < draws:
        n = min(10000, draws - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(values)))
        hits += int(np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
        done += n
    return float((hits + 1) / (draws + 1))


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    order = np.argsort(pvalues.to_numpy())
    raw = pvalues.to_numpy()[order]
    adjusted = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw)
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return pd.Series(out, index=pvalues.index)


def detail_path(run_root: Path, model: str) -> Path:
    hits = sorted((run_root / model / "numeric" / "causal_v2" / "answer_query_head_ablation").glob("confirmation_*/detail.csv.gz"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one detail file for {model}, found {hits}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--campaign-root", type=Path, required=True)
    args = ap.parse_args()
    out_dir = args.run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []

    for model_index, model in enumerate(("Qwen3-8B", "Gemma4-E4B")):
        frame = pd.read_csv(detail_path(args.run_root, model))
        frame["baseline_is_correct"] = frame["baseline_is_correct"].astype(bool)
        frame["patched_is_correct"] = frame["patched_is_correct"].astype(bool)
        ranking_path = args.campaign_root / "inputs" / f"{model}.first_span_absolute_mass_rankings.json"
        ranking = json.loads(ranking_path.read_text())["rankings"]["first_locator"]
        for k in KS:
            head_rows.append({
                "model": model,
                "top_k": k,
                "heads": ",".join(f"L{x['layer']}H{x['head']}" for x in ranking[:k]),
            })
            subset_k = frame.loc[frame["top_n"].eq(k)].copy()
            endpoint_specs = (
                ("all_absolute_error_increase", subset_k, "absolute_error_delta"),
                (
                    "correct_to_wrong_rate",
                    subset_k.loc[subset_k["baseline_is_correct"]].assign(
                        correct_to_wrong=lambda x: (~x["patched_is_correct"]).astype(float)
                    ),
                    "correct_to_wrong",
                ),
            )
            for endpoint_index, (endpoint, population, metric) in enumerate(endpoint_specs):
                ranked = population.loc[population["condition"].eq("ranked")].groupby("seed")[metric].mean()
                random = population.loc[population["condition"].eq("layer_matched_random")].groupby("seed")[metric].mean()
                common = ranked.index.intersection(random.index)
                diff = (ranked.loc[common] - random.loc[common]).to_numpy(float)
                ci_low, ci_high = bootstrap_ci(diff, seed=440000 + model_index * 1000 + endpoint_index * 100 + k)
                p = signflip_p(diff, seed=441000 + model_index * 1000 + endpoint_index * 100 + k)
                rows.append({
                    "model": model,
                    "top_k": k,
                    "endpoint": endpoint,
                    "seed_clusters": len(common),
                    "ranked_mean": float(ranked.loc[common].mean()),
                    "random_mean": float(random.loc[common].mean()),
                    "ranked_minus_random": float(diff.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "signflip_p": p,
                })

    summary = pd.DataFrame(rows)
    summary["holm_p_within_model_endpoint"] = np.nan
    for _, idx in summary.groupby(["model", "endpoint"]).groups.items():
        summary.loc[idx, "holm_p_within_model_endpoint"] = holm_adjust(summary.loc[idx, "signflip_p"])
    summary["holm_significant_0_05"] = summary["holm_p_within_model_endpoint"] < 0.05
    summary.to_csv(out_dir / "first_span_ablation_statistics.csv", index=False)
    pd.DataFrame(head_rows).to_csv(out_dir / "first_span_head_sets.csv", index=False)

    audit = {
        "schema_version": "realistic_niah_v4_first_span_ablation_audit_v1",
        "status": "complete",
        "ranking_definition": "descending discovery mean absolute attention mass over the complete first needle literal span",
        "control_family": "layer_matched_random",
        "models": sorted(summary["model"].unique()),
        "top_k": list(KS),
        "endpoints": sorted(summary["endpoint"].unique()),
        "rows": json.loads(summary.to_json(orient="records")),
    }
    (out_dir / "first_span_ablation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# V4.4 complete-first-span answer-query ablation",
        "",
        "Heads are frozen by discovery mean absolute attention mass over every token in the complete first needle literal span. Positive ranked-minus-random values mean that this frozen set damages counting more than three layer-matched random sets.",
        "",
        "```text",
        summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "```",
        "",
    ]
    (out_dir / "first_span_ablation_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
