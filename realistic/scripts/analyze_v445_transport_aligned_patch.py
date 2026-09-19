from __future__ import annotations

"""Aggregate the transport-aligned causal patch with seed-level inference."""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_signflip_p(values: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value on seed-mean paired contrasts."""
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def seed_bootstrap_ci(values: np.ndarray, *, draws: int = 50000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(442)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise RuntimeError("empty transport-aligned table")
    for row in rows:
        row["seed"] = int(row["seed"])
        row["receiver_count"] = int(row["receiver_count"])
        row["donor_count"] = int(row["donor_count"])
        for key in (
            "replacement_delta_norm",
            "clean_donor_log_odds",
            "condition_donor_log_odds",
            "donor_log_odds_gain",
            "target_donor_fraction",
        ):
            row[key] = float(row[key])
        row["argmax_token_changed"] = int(row["argmax_token_changed"])

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    cells: dict[tuple[str, int, int, int, str, str], dict] = {}
    for row in rows:
        grouped[(row["model_label"], row["support"], row["condition"])].append(row)
        cells[
            (
                row["model_label"],
                row["seed"],
                row["receiver_count"],
                row["donor_count"],
                row["support"],
                row["condition"],
            )
        ] = row

    summary = []
    for (model, support, condition), group in sorted(grouped.items()):
        summary.append(
            {
                "model_label": model,
                "support": support,
                "condition": condition,
                "rows": len(group),
                "seeds": len({row["seed"] for row in group}),
                "mean_target_donor_fraction": float(
                    np.mean([row["target_donor_fraction"] for row in group])
                ),
                "mean_donor_log_odds_gain": float(
                    np.mean([row["donor_log_odds_gain"] for row in group])
                ),
                "argmax_change_rate": float(
                    np.mean([row["argmax_token_changed"] for row in group])
                ),
                "mean_replacement_delta_norm": float(
                    np.mean([row["replacement_delta_norm"] for row in group])
                ),
            }
        )

    contrast_specs = [
        ("aligned_dose_1_minus_orthogonal", "aligned_dose_1", "matched_orthogonal"),
        ("aligned_dose_2_minus_orthogonal", "aligned_dose_2", "matched_orthogonal"),
        ("dose_2_minus_dose_1", "aligned_dose_2", "aligned_dose_1"),
    ]
    contrast_rows = []
    models = sorted({row["model_label"] for row in rows})
    supports = sorted({row["support"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    pairs = sorted({(row["receiver_count"], row["donor_count"]) for row in rows})
    for model in models:
        for support in supports:
            for contrast_name, left, right in contrast_specs:
                for metric in ("target_donor_fraction", "donor_log_odds_gain"):
                    seed_values = []
                    for seed in seeds:
                        paired = []
                        for receiver, donor in pairs:
                            left_row = cells[(model, seed, receiver, donor, support, left)]
                            right_row = cells[(model, seed, receiver, donor, support, right)]
                            paired.append(float(left_row[metric]) - float(right_row[metric]))
                        seed_values.append(float(np.mean(paired)))
                    seed_values_np = np.asarray(seed_values)
                    lo, hi = seed_bootstrap_ci(seed_values_np)
                    contrast_rows.append(
                        {
                            "model_label": model,
                            "support": support,
                            "contrast": contrast_name,
                            "metric": metric,
                            "seeds": len(seed_values),
                            "pairs_per_seed": len(pairs),
                            "mean_contrast": float(seed_values_np.mean()),
                            "bootstrap_95ci_low": lo,
                            "bootstrap_95ci_high": hi,
                            "exact_seed_signflip_p_two_sided": seed_signflip_p(seed_values_np),
                            "seed_mean_min": float(seed_values_np.min()),
                            "seed_mean_max": float(seed_values_np.max()),
                        }
                    )

    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("condition_summary.csv", summary), ("seed_contrasts.csv", contrast_rows)):
        with (args.output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    audit = {
        "schema_version": "realistic_niah_v4_4_5_transport_aligned_analysis_v1",
        "input": str(args.input.resolve()),
        "rows": len(rows),
        "models": models,
        "seeds": seeds,
        "pairs": pairs,
        "inference_unit": "seed mean across directed count pairs",
        "confidence_interval": "50,000-draw nonparametric bootstrap over seeds",
        "p_value": "exact two-sided sign-flip test over seed-mean paired contrasts",
        "status": "PASS",
    }
    (args.output / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
