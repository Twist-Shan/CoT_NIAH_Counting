from __future__ import annotations

"""Link discovery map stability to held-out layerwise transport causality."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MAP_PREDICTORS = (
    "cv_centroid_r2",
    "bootstrap_map_relative_frobenius_median",
    "bootstrap_rotation_geodesic_degrees_median",
    "subspace_principal_angle_max_degrees",
    "full_operator_cosine_to_next",
    "full_operator_relative_drift_to_next",
)


def exact_signflip_p(values: Iterable[float]) -> float:
    vector = np.asarray(list(values), dtype=float)
    vector = vector[np.isfinite(vector)]
    if not 1 <= len(vector) <= 20:
        raise ValueError("exact sign-flip requires 1..20 finite seed effects")
    observed = abs(float(vector.mean()))
    extreme = 0
    total = 1 << len(vector)
    for signs in itertools.product((-1.0, 1.0), repeat=len(vector)):
        draw = abs(float(np.mean(vector * np.asarray(signs))))
        extreme += int(draw >= observed - 1e-15)
    return float(extreme / total)


def bootstrap_ci(values: np.ndarray, *, label: str, draws: int) -> tuple[float, float]:
    stable = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(stable)
    distribution = values[
        rng.integers(0, len(values), size=(draws, len(values)))
    ].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def holm(values: Iterable[float]) -> list[float]:
    raw = np.asarray(list(values), dtype=float)
    adjusted = np.full(len(raw), np.nan)
    finite = np.flatnonzero(np.isfinite(raw))
    order = finite[np.argsort(raw[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, raw[index] * (len(order) - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-analysis", type=Path, required=True)
    parser.add_argument("--transport-analysis", type=Path, required=True)
    parser.add_argument("--design-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=50_000)
    args = parser.parse_args()

    design = json.loads(args.design_config.read_text(encoding="utf-8"))
    link_design = design["map_causal_link"]
    map_audit = json.loads(
        (args.map_analysis / "analysis_audit.json").read_text(encoding="utf-8")
    )
    transport_audit = json.loads(
        (args.transport_analysis / "analysis_audit.json").read_text(encoding="utf-8")
    )
    if map_audit["status"] != "PASS" or transport_audit["status"] != "PASS":
        raise RuntimeError("both source analyses must pass before linkage")
    maps = pd.read_csv(args.map_analysis / "layerwise_linear_map_summary.csv")
    maps = maps.loc[
        (maps["role"] == link_design["role"])
        & (maps["rank"] == int(link_design["rank"]))
    ].copy()
    effects = pd.read_csv(
        args.transport_analysis / "layerwise_transport_seed_effects.csv"
    )
    merge_keys = ["model_label", "source_layer", "target_layer"]
    if maps.duplicated(merge_keys).any():
        raise RuntimeError("duplicate rank/role map boundary")
    registered = {
        (model, int(source), int(target))
        for model, boundaries in design["answer_transport"]["boundaries"].items()
        for source, target in boundaries
    }
    observed = {
        (str(row.model_label), int(row.source_layer), int(row.target_layer))
        for row in effects[merge_keys].drop_duplicates().itertuples()
    }
    if observed != registered:
        raise RuntimeError(
            f"causal boundary mismatch: missing={registered-observed}, "
            f"unexpected={observed-registered}"
        )
    selected_maps = maps.merge(
        effects[merge_keys].drop_duplicates(), on=merge_keys, how="inner", validate="one_to_one"
    )
    if len(selected_maps) != len(registered):
        raise RuntimeError("not every registered causal boundary has a map")
    r2_threshold = float(link_design["stable_cv_centroid_r2_min"])
    dispersion_threshold = float(
        link_design["stable_bootstrap_map_relative_frobenius_median_max"]
    )
    selected_maps["locally_stable"] = (
        (selected_maps["cv_centroid_r2"] >= r2_threshold)
        & (
            selected_maps["bootstrap_map_relative_frobenius_median"]
            <= dispersion_threshold
        )
    )
    for model, group in selected_maps.groupby("model_label"):
        if group["locally_stable"].nunique() != 2:
            raise RuntimeError(f"{model} does not have both stability regimes")

    linked = effects.merge(
        selected_maps[
            merge_keys
            + ["locally_stable", "cv_centroid_r2", *MAP_PREDICTORS[1:]]
        ],
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    if linked["locally_stable"].isna().any():
        raise RuntimeError("missing stability label after merge")

    regime_rows: list[dict[str, float | int | str | bool]] = []
    for (model, contrast, metric), group in linked.groupby(
        ["model_label", "contrast", "metric"]
    ):
        seed_regimes = group.groupby(["seed", "locally_stable"])["effect"].mean().unstack()
        if False not in seed_regimes or True not in seed_regimes:
            raise RuntimeError(f"missing stability regime for {model}/{contrast}/{metric}")
        stable_minus_unstable = (
            seed_regimes[True] - seed_regimes[False]
        ).to_numpy(float)
        label = f"{model}/{contrast}/{metric}/stable-minus-unstable"
        low, high = bootstrap_ci(
            stable_minus_unstable, label=label, draws=args.bootstraps
        )
        map_group = selected_maps[selected_maps["model_label"] == model]
        regime_rows.append(
            {
                "model_label": model,
                "contrast": contrast,
                "metric": metric,
                "is_primary": bool(
                    contrast == link_design["primary_contrast"]
                    and metric == link_design["primary_metric"]
                ),
                "stable_boundaries": int(map_group["locally_stable"].sum()),
                "unstable_boundaries": int((~map_group["locally_stable"]).sum()),
                "seeds": len(stable_minus_unstable),
                "mean_stable_minus_unstable": float(stable_minus_unstable.mean()),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "exact_seed_signflip_p_two_sided": exact_signflip_p(
                    stable_minus_unstable
                ),
                "seed_effect_min": float(stable_minus_unstable.min()),
                "seed_effect_max": float(stable_minus_unstable.max()),
            }
        )
    regimes = pd.DataFrame(regime_rows)
    regimes["holm_p_within_model_six_tests"] = np.nan
    for _, indices in regimes.groupby("model_label").groups.items():
        regimes.loc[indices, "holm_p_within_model_six_tests"] = holm(
            regimes.loc[indices, "exact_seed_signflip_p_two_sided"]
        )

    boundary_effects = (
        linked.groupby(merge_keys + ["contrast", "metric"], as_index=False)
        .agg(mean_causal_effect=("effect", "mean"), seeds=("seed", "nunique"))
        .merge(selected_maps, on=merge_keys, validate="many_to_one")
    )
    correlation_rows: list[dict[str, float | int | str]] = []
    for (model, contrast, metric), group in boundary_effects.groupby(
        ["model_label", "contrast", "metric"]
    ):
        for predictor in MAP_PREDICTORS:
            complete = group[["mean_causal_effect", predictor]].dropna()
            if len(complete) < 2:
                raise RuntimeError(
                    f"too few complete boundaries for {model}/{contrast}/{metric}/{predictor}"
                )
            correlation_rows.append(
                {
                    "model_label": model,
                    "contrast": contrast,
                    "metric": metric,
                    "predictor": predictor,
                    "boundaries": len(complete),
                    "spearman_rho_descriptive": float(
                        complete["mean_causal_effect"].corr(
                            complete[predictor], method="spearman"
                        )
                    ),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    selected_maps.to_csv(args.output / "registered_boundary_stability.csv", index=False)
    boundary_effects.to_csv(args.output / "boundary_map_causal_effects.csv", index=False)
    regimes.to_csv(args.output / "stable_minus_unstable_tests.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(
        args.output / "boundary_spearman_descriptive.csv", index=False
    )
    primary = regimes.loc[regimes["is_primary"]].to_dict(orient="records")
    audit = {
        "schema_version": "realistic_niah_v4_4_map_causal_link_v1",
        "status": "PASS",
        "design_config": str(args.design_config),
        "map_analysis": str(args.map_analysis),
        "transport_analysis": str(args.transport_analysis),
        "map_source_status": map_audit["status"],
        "transport_source_status": transport_audit["status"],
        "role": link_design["role"],
        "rank": link_design["rank"],
        "stable_rule": {
            "cv_centroid_r2_min": r2_threshold,
            "bootstrap_map_relative_frobenius_median_max": dispersion_threshold,
        },
        "primary_estimand": link_design["primary_estimand"],
        "primary_results": primary,
        "multiplicity": design["multiplicity"]["map_causal_link"],
        "boundary_correlations": "descriptive only; registered depth landmarks are not treated as a random population",
        "bootstrap_draws": args.bootstraps,
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
