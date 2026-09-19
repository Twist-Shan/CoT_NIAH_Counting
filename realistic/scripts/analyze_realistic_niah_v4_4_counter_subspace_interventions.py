from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def _cluster_means(frame: pd.DataFrame, metric: str) -> np.ndarray:
    valid = frame[np.isfinite(pd.to_numeric(frame[metric], errors="coerce"))]
    if valid.empty:
        return np.asarray([], dtype=float)
    return (
        valid.groupby("seed", sort=True)[metric]
        .mean()
        .to_numpy(dtype=float)
    )


def _bootstrap_ci(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(sampled, [0.025, 0.975]))


def _signflip_p_greater(values: np.ndarray, *, draws: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = float(values.mean())
    if len(values) <= 20:
        masks = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(len(values), dtype=np.uint64)) & 1
        signs = bits.astype(np.int8) * 2 - 1
        null = (signs * values[None, :]).mean(axis=1)
        return float(np.mean(null >= observed - 1e-15))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    chunk = 10_000
    while completed < draws:
        size = min(chunk, draws - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        exceed += int(np.sum((signs * values).mean(axis=1) >= observed - 1e-15))
        completed += size
    return float((exceed + 1) / (draws + 1))


def _outcome_count(block: dict) -> float:
    value = block.get("outcome", {}).get("parsed_count")
    return float("nan") if value is None else float(value)


def load_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"design.json", "complete.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "realistic_niah_v4_4_counter_subspace_intervention_shard_v1":
            continue
        receiver_count = int(payload["receiver_count"])
        donor_count = int(payload["donor_count"])
        gap = donor_count - receiver_count
        if gap == 0:
            raise ValueError(f"donor and receiver counts are equal in {path}")
        clean_count = _outcome_count(payload["clean_receiver"])
        job = payload["job"]
        for condition, block in payload["conditions"].items():
            predicted = _outcome_count(block)
            valid_pair = bool(np.isfinite(clean_count) and np.isfinite(predicted))
            aligned_shift = (
                float(np.sign(gap) * (predicted - clean_count))
                if valid_pair
                else float("nan")
            )
            rows.append(
                {
                    "model_label": payload["model_label"],
                    "job_id": str(job["job_id"]),
                    "seed": int(payload["seed"]),
                    "receiver_count": receiver_count,
                    "donor_count": donor_count,
                    "count_gap": gap,
                    "source_layer": int(job["source_layer"]),
                    "source_site_kind": str(job["source_site"]["kind"]),
                    "mediator_layer": job.get("mediator_layer"),
                    "removal_dose": float(job.get("removal_dose", 1.0)),
                    "condition": condition,
                    "clean_receiver_prediction": clean_count,
                    "condition_prediction": predicted,
                    "valid_pair": valid_pair,
                    "invalid_condition": not np.isfinite(predicted),
                    "donor_aligned_shift": aligned_shift,
                    "normalized_donor_aligned_shift": (
                        aligned_shift / abs(gap) if valid_pair else float("nan")
                    ),
                    "donor_adoption": (
                        float(predicted == donor_count)
                        if np.isfinite(predicted)
                        else float("nan")
                    ),
                    "receiver_retention": (
                        float(predicted == receiver_count)
                        if np.isfinite(predicted)
                        else float("nan")
                    ),
                    "source": str(path),
                }
            )
    if not rows:
        raise RuntimeError(f"No intervention shards found below {root}")
    return pd.DataFrame(rows)


def paired_contrasts(rows: pd.DataFrame) -> pd.DataFrame:
    index = [
        "model_label",
        "job_id",
        "seed",
        "receiver_count",
        "donor_count",
        "count_gap",
        "source_layer",
        "source_site_kind",
        "mediator_layer",
        "removal_dose",
    ]
    wide = rows.pivot(index=index, columns="condition", values="normalized_donor_aligned_shift")
    output = wide.reset_index()
    if {"projected_patch", "orthogonal_norm_matched"} <= set(wide.columns):
        output["projected_minus_orthogonal"] = (
            wide["projected_patch"] - wide["orthogonal_norm_matched"]
        ).to_numpy()
    if {"projected_patch", "projected_patch_plus_removal"} <= set(wide.columns):
        output["mediation_attenuation"] = (
            wide["projected_patch"] - wide["projected_patch_plus_removal"]
        ).to_numpy()
    return output


def summarize(
    rows: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    bootstrap_draws: int,
    signflip_draws: int,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    grouping = [
        "model_label",
        "source_layer",
        "source_site_kind",
        "mediator_layer",
        "removal_dose",
    ]
    for keys, group in rows.groupby(grouping, dropna=False):
        for condition, condition_rows in group.groupby("condition"):
            values = _cluster_means(condition_rows, "normalized_donor_aligned_shift")
            lo, hi = _bootstrap_ci(values, draws=bootstrap_draws, seed=442)
            summaries.append(
                {
                    **dict(zip(grouping, keys)),
                    "estimand": f"{condition}_vs_clean",
                    "jobs": condition_rows["job_id"].nunique(),
                    "seed_clusters": len(values),
                    "eligible_rows": int(condition_rows["valid_pair"].sum()),
                    "total_rows": len(condition_rows),
                    "invalid_rate": float(condition_rows["invalid_condition"].mean()),
                    "equal_seed_mean": float(values.mean()) if len(values) else float("nan"),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "one_sided_signflip_p": _signflip_p_greater(
                        values, draws=signflip_draws, seed=442
                    ),
                }
            )
    contrast_metrics = [
        metric
        for metric in ("projected_minus_orthogonal", "mediation_attenuation")
        if metric in contrasts
    ]
    for keys, group in contrasts.groupby(grouping, dropna=False):
        for metric in contrast_metrics:
            values = _cluster_means(group, metric)
            lo, hi = _bootstrap_ci(values, draws=bootstrap_draws, seed=443)
            summaries.append(
                {
                    **dict(zip(grouping, keys)),
                    "estimand": metric,
                    "jobs": group["job_id"].nunique(),
                    "seed_clusters": len(values),
                    "eligible_rows": int(np.isfinite(group[metric]).sum()),
                    "total_rows": len(group),
                    "invalid_rate": float(1.0 - np.isfinite(group[metric]).mean()),
                    "equal_seed_mean": float(values.mean()) if len(values) else float("nan"),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "one_sided_signflip_p": _signflip_p_greater(
                        values, draws=signflip_draws, seed=443
                    ),
                }
            )
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU analysis for V4.4 projected patch/removal/mediation shards"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--signflip-draws", type=int, default=200_000)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.run_root.resolve())
    contrasts = paired_contrasts(rows)
    summary = summarize(
        rows,
        contrasts,
        bootstrap_draws=args.bootstrap_draws,
        signflip_draws=args.signflip_draws,
    )
    rows.to_csv(output / "counter_subspace_intervention_rows.csv.gz", index=False, compression="gzip")
    contrasts.to_csv(output / "counter_subspace_intervention_contrasts.csv", index=False)
    summary.to_csv(output / "counter_subspace_intervention_statistics.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_counter_subspace_intervention_analysis_v1",
        "run_root": str(args.run_root.resolve()),
        "rows": len(rows),
        "jobs": int(rows["job_id"].nunique()),
        "models": sorted(rows["model_label"].unique()),
        "bootstrap_unit": "seed",
        "bootstrap_draws": args.bootstrap_draws,
        "signflip_unit": "seed",
        "signflip_draws_if_not_exact": args.signflip_draws,
        "primary_contrasts": ["projected_minus_orthogonal", "mediation_attenuation"],
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    (output / "counter_subspace_intervention_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
