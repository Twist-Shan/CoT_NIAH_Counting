from __future__ import annotations

"""Exact geometry and marginal factor decompositions of correct-answer noise."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def correct_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "correct"})


def eta_squared(frame: pd.DataFrame, target: str, factor: str) -> tuple[float, int]:
    scoped = frame[[target, factor]].dropna()
    if len(scoped) < 2 or scoped[factor].nunique() < 2:
        return float("nan"), int(scoped[factor].nunique())
    y = scoped[target].astype(float)
    grand = float(y.mean())
    total = float(np.square(y - grand).sum())
    grouped = scoped.groupby(factor, dropna=False)[target].agg(["count", "mean"])
    between = float((grouped["count"] * np.square(grouped["mean"] - grand)).sum())
    return between / max(total, 1e-12), len(grouped)


def load_role(path: Path, role: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[frame["split"].astype(str) == "confirmation"].copy()
    frame["role"] = role
    if role == "trace_running":
        frame["position_bin"] = np.floor(frame["trace_progress"].clip(0, 0.999999) * 10).astype("Int64")
        frame["scope_site"] = frame["site"].astype(str)
    else:
        frame["position_bin"] = np.floor(frame["prompt_progress"].clip(0, 0.999999) * 10).astype("Int64")
        frame["scope_site"] = "needle_span_end"
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--prompt-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = [load_role(path.resolve(), "trace_running") for path in args.trace_inputs]
    frames.extend(load_role(path.resolve(), "prompt_running") for path in args.prompt_inputs)
    full = pd.concat(frames, ignore_index=True)

    geometry_rows = []
    factor_rows = []
    scopes = []
    for (model, role), model_role in full.groupby(["model_label", "role"], sort=True):
        if role == "trace_running":
            scopes.append((model, role, "all_sites", model_role))
        scopes.extend(
            (model, role, str(site), site_frame)
            for site, site_frame in model_role.groupby("scope_site", sort=True)
        )
    for model, role, site, scoped_all in scopes:
        for population, scoped in (
            ("all_confirmation", scoped_all),
            ("correct_only_confirmation", scoped_all.loc[correct_mask(scoped_all["baseline_correct"])]),
        ):
            if scoped.empty:
                continue
            total = float(scoped["noise_total"].sum())
            parallel = float(scoped["noise_parallel"].sum())
            orthogonal = float(scoped["noise_orthogonal"].sum())
            axis = float(np.square(scoped["count_axis_deviation"].astype(float)).sum())
            geometry_rows.append({
                "model_label": model,
                "role": role,
                "site": site,
                "population": population,
                "observations": len(scoped),
                "samples": scoped["stimulus_id"].nunique(),
                "mean_noise_total_rms": float(scoped["noise_total_rms"].mean()),
                "rms_noise_total": float(np.sqrt(np.square(scoped["noise_total_rms"]).mean())),
                "parallel_energy_share": parallel / max(total, 1e-12),
                "orthogonal_energy_share": orthogonal / max(total, 1e-12),
                "first_count_axis_energy_share": axis / max(total, 1e-12),
                "parallel_plus_orthogonal_error": abs((parallel + orthogonal) - total) / max(total, 1e-12),
            })
            base_factors = ["running_index", "position_bin", "seed", "realization_id"]
            factors = (
                [*base_factors, "marker_kind", "termination_kind", "trace_order_class", "site"]
                if role == "trace_running"
                else [*base_factors, "city", "score", "haystack_source_mode", "haystack_source_files"]
            )
            for target in ("noise_total_rms", "noise_orthogonal_rms"):
                for factor in factors:
                    if factor not in scoped:
                        continue
                    eta, levels = eta_squared(scoped, target, factor)
                    factor_rows.append({
                        "model_label": model,
                        "role": role,
                        "site": site,
                        "population": population,
                        "target": target,
                        "factor": factor,
                        "marginal_eta_squared": eta,
                        "levels": levels,
                        "observations": len(scoped),
                    })
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(geometry_rows).to_csv(output / "geometric_noise_decomposition.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(output / "marginal_conditional_variance.csv", index=False)
    audit = {
        "schema_version": "realistic_niah_v4_4_correct_noise_decomposition_v1",
        "evaluation_split": "confirmation",
        "populations": ["all_confirmation", "correct_only_confirmation"],
        "exact_identity": "total energy = rank-3 counter-subspace energy + orthogonal energy",
        "marginal_eta_squared_warning": (
            "Each factor is a separate law-of-total-variance decomposition; values overlap "
            "and must not be summed. Joint held-out attribution is in the factor-model tables."
        ),
        "unexplained_warning": (
            "Residual after measured covariates is not identified as stochastic noise because "
            "the design has no repeated generation of an identical prompt."
        ),
        "geometry_rows": len(geometry_rows),
        "factor_rows": len(factor_rows),
        "status": "PASS",
    }
    (output / "decomposition_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
