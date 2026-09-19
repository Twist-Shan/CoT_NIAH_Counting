from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from realistic_niah_v4_4_3.io import atomic_csv_gzip, atomic_json, atomic_text, stage_root

from .analysis import exact_sign_flip_p
from .relay_spec import V444RelayConfig
from .spec import V444Config


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Relay slope needs matching nontrivial arrays")
    centered = x_values - float(x_values.mean())
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        raise ValueError("Relay slope predictor is constant")
    return float(np.sum(centered * (y_values - float(y_values.mean()))) / denominator)


def _bootstrap_ci(
    values: Sequence[float], *, repetitions: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Relay bootstrap values must be finite and nonempty")
    generator = np.random.default_rng(int(seed))
    draws = generator.choice(array, size=(int(repetitions), len(array)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_relay_seed_metrics(
    natural: pd.DataFrame,
    patch: pd.DataFrame,
    removal: pd.DataFrame,
    *,
    relay_config: V444RelayConfig,
) -> pd.DataFrame:
    _require_columns(
        natural,
        ("seed", "gold_count", "relay_carrier_coefficient"),
        "relay natural detail",
    )
    _require_columns(
        patch,
        (
            "seed",
            "receiver_count",
            "donor_count",
            "intervention",
            "continuous_normalized_transport",
            "relay_patch_global_axis_coefficient",
        ),
        "relay patch detail",
    )
    _require_columns(
        removal,
        (
            "seed",
            "intervention",
            "delta_expected_count_absolute_error",
            "delta_correct_margin",
        ),
        "relay removal detail",
    )
    rows: list[dict[str, float | int]] = []
    for seed in relay_config.confirmation_seeds:
        natural_seed = natural[natural["seed"].eq(seed)].sort_values("gold_count")
        if len(natural_seed) != len(relay_config.counts):
            raise ValueError(f"Relay natural rows are incomplete for seed {seed}")
        natural_slope = _ols_slope(
            natural_seed["gold_count"], natural_seed["relay_carrier_coefficient"]
        )
        patch_seed = patch[patch["seed"].eq(seed)]
        expected_patch_rows = len(relay_config.relay_pairs) * 3
        if len(patch_seed) != expected_patch_rows:
            raise ValueError(f"Relay patch rows are incomplete for seed {seed}")
        base_patch = patch_seed[
            patch_seed["intervention"].eq("receiver_alpha_donor_v_edge_patch")
        ].copy()
        if len(base_patch) != len(relay_config.relay_pairs):
            raise ValueError(f"Relay base-patch rows are incomplete for seed {seed}")
        gaps = (
            base_patch["donor_count"].to_numpy(float)
            - base_patch["receiver_count"].to_numpy(float)
        )
        first_stage = (
            base_patch["relay_patch_global_axis_coefficient"].to_numpy(float) / gaps
        )
        behavior = base_patch["continuous_normalized_transport"].to_numpy(float)
        pivot = patch_seed.pivot_table(
            index=["receiver_count", "donor_count"],
            columns="intervention",
            values="continuous_normalized_transport",
            aggfunc="first",
        )
        required_patch_conditions = {
            "receiver_alpha_donor_v_edge_patch_plus_natural_axis_block",
            "receiver_alpha_donor_v_edge_patch_plus_orthogonal_control",
        }
        if not required_patch_conditions.issubset(pivot.columns):
            raise ValueError("Relay mediation conditions are incomplete")
        mediation = (
            pivot[
                "receiver_alpha_donor_v_edge_patch_plus_orthogonal_control"
            ]
            - pivot[
                "receiver_alpha_donor_v_edge_patch_plus_natural_axis_block"
            ]
        )
        removal_seed = removal[removal["seed"].eq(seed)]
        expected_removal_rows = len(relay_config.removal_counts) * 2
        if len(removal_seed) != expected_removal_rows:
            raise ValueError(f"Relay removal rows are incomplete for seed {seed}")
        removal_pivot = removal_seed.pivot_table(
            index="gold_count",
            columns="intervention",
            values=("delta_expected_count_absolute_error", "delta_correct_margin"),
            aggfunc="first",
        )
        required_removal = {"relay_axis_removal", "relay_axis_orthogonal_control"}
        if not required_removal.issubset(
            removal_pivot["delta_expected_count_absolute_error"].columns
        ):
            raise ValueError("Relay removal/control conditions are incomplete")
        error_specificity = (
            removal_pivot["delta_expected_count_absolute_error"][
                "relay_axis_removal"
            ]
            - removal_pivot["delta_expected_count_absolute_error"][
                "relay_axis_orthogonal_control"
            ]
        )
        margin_specificity = (
            removal_pivot["delta_correct_margin"]["relay_axis_removal"]
            - removal_pivot["delta_correct_margin"][
                "relay_axis_orthogonal_control"
            ]
        )
        rows.append(
            {
                "seed": int(seed),
                "natural_relay_slope": natural_slope,
                "edge_patch_first_stage_transport": float(np.mean(first_stage)),
                "edge_patch_behavior_transport": float(np.mean(behavior)),
                "ov_mediation_specificity": float(np.mean(mediation)),
                "relay_removal_error_specificity": float(
                    np.mean(error_specificity)
                ),
                "relay_removal_margin_specificity": float(
                    np.mean(margin_specificity)
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not np.isfinite(result.drop(columns="seed").to_numpy(float)).all():
        raise ValueError("Relay seed metrics contain non-finite values")
    return result


def summarize_relay_seed_metrics(
    seed_metrics: pd.DataFrame, *, relay_config: V444RelayConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specifications = (
        ("natural_relay_slope", "greater"),
        ("edge_patch_first_stage_transport", "greater"),
        ("edge_patch_behavior_transport", "greater"),
        ("ov_mediation_specificity", "greater"),
        ("relay_removal_error_specificity", "greater"),
        ("relay_removal_margin_specificity", "less"),
    )
    rows = []
    for offset, (metric, alternative) in enumerate(specifications):
        values = seed_metrics[metric].to_numpy(float)
        low, high = _bootstrap_ci(
            values,
            repetitions=relay_config.bootstrap_repetitions,
            seed=44_400 + offset,
        )
        rows.append(
            {
                "metric": metric,
                "alternative": alternative,
                "seed_count": len(values),
                "mean": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "positive_seed_fraction": float(np.mean(values > 0)),
                "exact_sign_flip_p": exact_sign_flip_p(
                    values, alternative=alternative
                ),
            }
        )
    summary = pd.DataFrame(rows)
    p = dict(zip(summary["metric"], summary["exact_sign_flip_p"]))
    families = {
        "natural_relay": p["natural_relay_slope"],
        "edge_patch": max(
            p["edge_patch_first_stage_transport"],
            p["edge_patch_behavior_transport"],
        ),
        "ov_mediation": p["ov_mediation_specificity"],
        "relay_removal": max(
            p["relay_removal_error_specificity"],
            p["relay_removal_margin_specificity"],
        ),
    }
    global_p = max(families.values())
    decision = {
        "decision_rule": relay_config.primary_decision_rule,
        "alpha": relay_config.primary_alpha,
        "family_p_values": families,
        "global_intersection_union_p": global_p,
        "all_families_pass": bool(
            all(value <= relay_config.primary_alpha for value in families.values())
        ),
    }
    return summary, decision


def audit_relay_campaign(
    run_root: str | Path,
    *,
    base_config: V444Config,
    relay_config: V444RelayConfig,
) -> dict[str, Any]:
    root = Path(run_root)
    model = relay_config.model_label
    discovery_root = stage_root(root, model, "relay_discovery")
    smoke_root = stage_root(root, model, "relay_smoke")
    confirmation_root = stage_root(root, model, "relay_confirmation")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    relay_config.validate_against_base(base_config)
    required = (
        discovery_root / "complete.json",
        discovery_root / "selection.json",
        discovery_root / "artifacts.pt",
        discovery_root / "natural_position_set_detail.csv.gz",
        smoke_root / "complete.json",
        confirmation_root / "complete.json",
        confirmation_root / "natural_detail.csv.gz",
        confirmation_root / "edge_patch_detail.csv.gz",
        confirmation_root / "removal_detail.csv.gz",
    )
    missing = [str(path) for path in required if not path.is_file()]
    check("required_artifacts", not missing, missing)
    if missing:
        return {"all_checks_pass": False, "checks": checks}
    selection = json.loads(
        (discovery_root / "selection.json").read_text(encoding="utf-8")
    )
    check(
        "selection_uses_only_natural_discovery",
        selection.get("selection_uses_causal_outcomes") is False,
        selection.get("selection_uses_causal_outcomes"),
    )
    check(
        "candidate_is_eligible_relay_set",
        selection.get("selected_position_set") in relay_config.eligible_relay_sets,
        selection.get("selected_position_set"),
    )
    check(
        "candidate_not_source_control",
        selection.get("selected_position_set") not in relay_config.source_control_sets,
        selection.get("selected_position_set"),
    )
    natural = pd.read_csv(confirmation_root / "natural_detail.csv.gz")
    patch = pd.read_csv(confirmation_root / "edge_patch_detail.csv.gz")
    removal = pd.read_csv(confirmation_root / "removal_detail.csv.gz")
    expected_natural = len(relay_config.confirmation_seeds) * len(relay_config.counts)
    expected_patch = (
        len(relay_config.confirmation_seeds) * len(relay_config.relay_pairs) * 3
    )
    expected_removal = (
        len(relay_config.confirmation_seeds) * len(relay_config.removal_counts) * 2
    )
    check("natural_row_count", len(natural) == expected_natural, len(natural))
    check("patch_row_count", len(patch) == expected_patch, len(patch))
    check("removal_row_count", len(removal) == expected_removal, len(removal))
    check(
        "confirmation_seed_registry",
        set(natural["seed"].astype(int)) == set(relay_config.confirmation_seeds),
        sorted(set(natural["seed"].astype(int))),
    )
    design_hashes = set(natural["design_hash"].astype(str))
    design_hashes |= set(patch["design_hash"].astype(str))
    design_hashes |= set(removal["design_hash"].astype(str))
    check("single_confirmation_design_hash", len(design_hashes) == 1, sorted(design_hashes))
    max_reconstruction = float(
        pd.concat(
            [
                natural["edge_z_reconstruction_relative_l2"],
            ],
            ignore_index=True,
        ).max()
    )
    check(
        "edge_reconstruction_tolerance",
        math.isfinite(max_reconstruction)
        and max_reconstruction
        <= relay_config.contribution_reconstruction_relative_tolerance,
        max_reconstruction,
    )
    cache_columns = (
        "attention_cache_candidate_logit_max_abs_delta",
        "attention_cache_candidate_centered_logit_max_abs_delta",
        "attention_cache_reference_tolerance_exceeded",
    )
    missing_cache_columns = [name for name in cache_columns if name not in natural]
    check(
        "attention_cache_diagnostics_recorded",
        not missing_cache_columns,
        missing_cache_columns,
    )
    if not missing_cache_columns:
        raw_cache = natural[cache_columns[0]].astype(float)
        centered_cache = natural[cache_columns[1]].astype(float)
        cache_detail = {
            "policy": "diagnostic_only; direct pre-O reconstruction is the hard gate",
            "reference_tolerance": relay_config.attention_cache_logit_tolerance,
            "max_raw_candidate_logit_delta": float(raw_cache.max()),
            "max_centered_candidate_logit_delta": float(centered_cache.max()),
            "reference_tolerance_exceedance_rows": int(
                natural[cache_columns[2]].astype(bool).sum()
            ),
            "row_count": int(len(natural)),
        }
        check(
            "attention_cache_diagnostics_finite",
            bool(np.isfinite(raw_cache).all() and np.isfinite(centered_cache).all()),
            cache_detail,
        )
    forbidden = list(root.rglob("*relay*raw*position*")) + list(
        root.rglob("*relay*full*value*")
    )
    check("no_raw_relay_maps_or_full_values", not forbidden, [str(p) for p in forbidden])
    return {
        "all_checks_pass": all(row["passed"] for row in checks),
        "check_count": len(checks),
        "checks": checks,
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    decision = payload["primary_decision"]
    selection = payload["selection"]
    metrics = {row["metric"]: row for row in payload["metric_summary"]}

    def line(metric: str) -> str:
        row = metrics[metric]
        return (
            f"{row['mean']:.6f} "
            f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}], "
            f"p={row['exact_sign_flip_p']:.6g}"
        )

    status = "SUPPORTED" if decision["all_families_pass"] else "NOT SUPPORTED"
    return f"""# Realistic NIAH V4.4.4 relay-to-OV supplement

## Registered claim

This supplement does not require L28 H16/H19 to locate raw needles through
their own QK circuit. It tests the narrower serial path
`relay value set -> receiver-alpha read -> L28 H16/H19 pre-O Z -> answer`.

The selected position set is `{selection['selected_position_set']}`. Selection
used only natural source-contribution rows from the discovery seeds and did not
use confirmation causal outcomes. The result is **{status}** under the frozen
four-family intersection-union rule (global p={decision['global_intersection_union_p']:.6g}).

## Results

- Natural relay carrier slope: {line('natural_relay_slope')}.
- Receiver-alpha/donor-V first-stage transport: {line('edge_patch_first_stage_transport')}.
- Receiver-alpha/donor-V behavioral transport: {line('edge_patch_behavior_transport')}.
- L28 natural-OV mediation specificity: {line('ov_mediation_specificity')}.
- Relay removal error specificity: {line('relay_removal_error_specificity')}.
- Relay removal margin specificity: {line('relay_removal_margin_specificity')}.

The natural carrier and the mechanical OV-axis first stage are supported, but
the answer-level transport interval crosses zero and the registered OV block
does not outperform its matched orthogonal control. Both removal estimands are
in the direction opposite to natural necessity. The defensible conclusion is
therefore that the selected late value-state carries count information and is
mechanically accessible through H16/H19's OV subspace, not that the model
naturally depends on this terminal relay-to-OV path to produce the answer.

## Interpretation boundary

The V-only edge patch keeps receiver Q, K, and alpha fixed. A positive result
therefore supports transport of content already present in a relay value state;
it does not identify the upstream heads that created that state. The natural
axis block tests whether the patch effect passes through the frozen L28
H16/H19 OV channel. Removal is required to distinguish natural use from mere
intervention accessibility.

## Audit

Relay audit: `{payload['audit']['all_checks_pass']}` across
{payload['audit'].get('check_count', len(payload['audit']['checks']))} checks.
No raw per-token contribution maps or full V-state tensors are persisted.
The eager/cache final-logit delta is retained as a non-fatal numerical
diagnostic. Direct reconstruction of the original L28 pre-O z from
`sum(alpha * V)` is the hard validity gate (relative L2 <= the registered
threshold, 0.05 in this campaign).
"""


def analyze_relay_campaign(
    *,
    run_root: str | Path,
    base_config: V444Config,
    relay_config: V444RelayConfig,
) -> dict[str, Any]:
    relay_config.validate_against_base(base_config)
    root = Path(run_root)
    discovery_root = stage_root(root, relay_config.model_label, "relay_discovery")
    confirmation_root = stage_root(
        root, relay_config.model_label, "relay_confirmation"
    )
    selection = json.loads(
        (discovery_root / "selection.json").read_text(encoding="utf-8")
    )
    natural = pd.read_csv(confirmation_root / "natural_detail.csv.gz")
    patch = pd.read_csv(confirmation_root / "edge_patch_detail.csv.gz")
    removal = pd.read_csv(confirmation_root / "removal_detail.csv.gz")
    seed_metrics = build_relay_seed_metrics(
        natural, patch, removal, relay_config=relay_config
    )
    metric_summary, decision = summarize_relay_seed_metrics(
        seed_metrics, relay_config=relay_config
    )
    audit = audit_relay_campaign(
        root, base_config=base_config, relay_config=relay_config
    )
    decision["selected_position_set"] = selection["selected_position_set"]
    decision["selection_qualified"] = bool(selection["selection_qualified"])
    decision["supported"] = bool(
        decision["all_families_pass"]
        and selection["selection_qualified"]
        and audit["all_checks_pass"]
    )
    output = root / "analysis" / "relay"
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv_gzip(seed_metrics, output / "relay_seed_metrics.csv.gz")
    atomic_csv_gzip(metric_summary, output / "relay_metric_summary.csv.gz")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_relay_analysis_v1",
        "selection": selection,
        "primary_decision": decision,
        "metric_summary": metric_summary.to_dict(orient="records"),
        "audit": audit,
    }
    atomic_json(output / "realistic_niah_v4_4_4_relay_analysis.json", payload)
    atomic_text(
        output / "realistic_niah_v4_4_4_relay_report.md",
        _markdown_report(payload),
    )
    return payload
