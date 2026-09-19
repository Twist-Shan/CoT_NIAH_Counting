from __future__ import annotations

"""Create dimensionless prompt-vs-trace running-counter noise comparisons."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def correct_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "correct"})


def trace_signal(source_run: Path, model: str) -> dict[str, dict[str, float | int]]:
    analysis = source_run / "geometry" / model / "analysis"
    primary = pd.read_csv(analysis / "primary_layers.csv")
    primary = primary.loc[
        (primary.estimand == "running_index") & (primary.stratum == "all")
    ]
    output = {}
    with np.load(analysis / "directions_and_centroids.npz", allow_pickle=False) as payload:
        for row in primary.itertuples(index=False):
            layer = int(row.layer)
            prefix = f"running_index__{row.site}__all__layer_{layer:03d}__discovery_centroid_"
            centroids = np.stack([payload[prefix + str(label)] for label in range(1, 11)]).astype(np.float64)
            centered = centroids - centroids.mean(axis=0, keepdims=True)
            signal = math.sqrt(float(np.square(centered).sum(axis=1).mean()) / centroids.shape[1])
            output[str(row.site)] = {"layer": layer, "hidden": centroids.shape[1], "signal_rms": signal}
    return output


def aggregate_trace(path: Path, signals: dict[str, dict[str, float | int]]) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], list[float]] = {}
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        chunk = chunk.loc[chunk["split"].astype(str) == "confirmation"]
        for population, scoped in (
            ("all_eligible", chunk),
            ("baseline_correct_only", chunk.loc[correct_mask(chunk["baseline_correct"])]),
            ("matched_n10_population", chunk.loc[chunk["gold_count"] == 10]),
            (
                "matched_n10_baseline_correct_only",
                chunk.loc[(chunk["gold_count"] == 10) & correct_mask(chunk["baseline_correct"])],
            ),
        ):
            grouped = scoped.groupby("site")["noise_total"].agg(["sum", "count"])
            for site, row in grouped.iterrows():
                key = (population, str(site))
                current = totals.setdefault(key, [0.0, 0.0])
                current[0] += float(row["sum"])
                current[1] += float(row["count"])
    rows = []
    for (population, site), (energy, observations) in totals.items():
        signal = signals[site]
        noise = math.sqrt(energy / (observations * int(signal["hidden"])))
        rows.append({
            "role": "trace_running",
            "site": site,
            "layer": signal["layer"],
            "population": population,
            "observations": int(observations),
            "signal_rms": signal["signal_rms"],
            "noise_rms": noise,
            "noise_to_signal_ratio": noise / max(float(signal["signal_rms"]), 1e-12),
        })
    return rows


def aggregate_prompt(path: Path, geometry_path: Path, matched_ids: set[str]) -> list[dict[str, object]]:
    with np.load(geometry_path, allow_pickle=False) as payload:
        primary = int(payload["primary_layer"])
        hidden = int(payload["centroids"].shape[-1])
        signal = float(payload["signal_rms"][primary])
    totals = {
        "all_n10": [0.0, 0],
        "baseline_correct_only_n10": [0.0, 0],
        "matched_trace_eligible_n10": [0.0, 0],
        "matched_trace_eligible_correct_only_n10": [0.0, 0],
    }
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        chunk = chunk.loc[chunk["split"].astype(str) == "confirmation"]
        totals["all_n10"][0] += float(chunk["noise_total"].sum())
        totals["all_n10"][1] += len(chunk)
        correct = chunk.loc[correct_mask(chunk["baseline_correct"])]
        totals["baseline_correct_only_n10"][0] += float(correct["noise_total"].sum())
        totals["baseline_correct_only_n10"][1] += len(correct)
        matched = chunk.loc[chunk["stimulus_id"].astype(str).isin(matched_ids)]
        totals["matched_trace_eligible_n10"][0] += float(matched["noise_total"].sum())
        totals["matched_trace_eligible_n10"][1] += len(matched)
        matched_correct = matched.loc[correct_mask(matched["baseline_correct"])]
        totals["matched_trace_eligible_correct_only_n10"][0] += float(
            matched_correct["noise_total"].sum()
        )
        totals["matched_trace_eligible_correct_only_n10"][1] += len(matched_correct)
    rows = []
    for population, (energy, observations) in totals.items():
        if observations == 0:
            continue
        noise = math.sqrt(energy / (observations * hidden))
        rows.append({
            "role": "prompt_running",
            "site": "needle_span_end",
            "layer": primary,
            "population": population,
            "observations": observations,
            "signal_rms": signal,
            "noise_rms": noise,
            "noise_to_signal_ratio": noise / max(signal, 1e-12),
        })
    return rows


def matched_trace_ids(path: Path) -> set[str]:
    output: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["stimulus_id", "split", "gold_count"], chunksize=100_000):
        scoped = chunk.loc[(chunk["split"] == "confirmation") & (chunk["gold_count"] == 10)]
        output.update(scoped["stimulus_id"].astype(str))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        trace_path = args.trace_root.resolve() / model / "trace_noise_rows.csv.gz"
        prompt_dir = args.prompt_root.resolve() / model
        model_rows = aggregate_trace(trace_path, trace_signal(args.source_run.resolve(), model))
        matched = matched_trace_ids(trace_path)
        model_rows.extend(
            aggregate_prompt(
                prompt_dir / "prompt_noise_rows.csv.gz",
                prompt_dir / "frozen_prompt_counter_geometry.npz",
                matched,
            )
        )
        for row in model_rows:
            row["model_label"] = model
        rows.extend(model_rows)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prompt_trace_noise_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema_version": "realistic_niah_v4_4_prompt_trace_noise_comparison_v1",
        "normalization": "RMS within-count residual divided by RMS between-count centroid signal",
        "basis_fit_split": "discovery",
        "evaluation_split": "confirmation",
        "rows": len(rows),
        "status": "PASS",
    }
    (output / "comparison_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
