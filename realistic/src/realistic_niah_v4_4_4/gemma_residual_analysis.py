from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from realistic_niah_v4_4_3.io import (
    atomic_csv_gzip,
    atomic_json,
    atomic_text,
    stage_root,
)

from .analysis import _bootstrap_ci, _json_records, exact_sign_flip_p
from .gemma_residual_pipeline import (
    CONFIRMATION_STAGE,
    DISCOVERY_STAGE,
    SMOKE_STAGE,
)
from .gemma_residual_spec import GemmaResidualConfig


PATH_ENDPOINTS = (
    "source_donor_transport",
    "exact_residual_mediation",
    "count_axis_mediation",
    "terminal_count_adoption",
)
CLEAN_NECESSITY_ENDPOINTS = (
    "clean_correct_failure_rate",
    "clean_delta_absolute_error",
)


def endpoint_registry(config: GemmaResidualConfig) -> tuple[str, ...]:
    clean = CLEAN_NECESSITY_ENDPOINTS if config.require_clean_necessity else ()
    return (*clean, *PATH_ENDPOINTS)


def build_seed_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    required_conditions = {
        "source_patch",
        "source_patch_plus_exact_block",
        "source_patch_plus_exact_orthogonal",
        "source_patch_plus_count_axis_block",
        "source_patch_plus_count_axis_orthogonal",
    }
    rows: list[dict[str, Any]] = []
    for (set_id, set_role, seed), group in detail.groupby(
        ["set_id", "set_role", "seed"], sort=True
    ):
        observed = set(group["condition"].astype(str))
        if observed != required_conditions:
            raise RuntimeError(
                f"Residual seed lacks a condition: {set_id}/{seed}={observed}"
            )
        by_condition = {
            condition: subgroup
            for condition, subgroup in group.groupby("condition", sort=False)
        }
        patch = by_condition["source_patch"]
        exact_block = by_condition["source_patch_plus_exact_block"]
        exact_control = by_condition["source_patch_plus_exact_orthogonal"]
        count_block = by_condition["source_patch_plus_count_axis_block"]
        count_control = by_condition["source_patch_plus_count_axis_orthogonal"]
        pair_columns = ["receiver_count", "donor_count"]
        for frame in by_condition.values():
            if frame[pair_columns].duplicated().any():
                raise RuntimeError("Residual condition contains duplicate donor pairs")

        def paired_difference(left: pd.DataFrame, right: pd.DataFrame) -> float:
            merged = left.merge(
                right,
                on=pair_columns,
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            return float(
                (
                    merged["continuous_normalized_transport_left"]
                    - merged["continuous_normalized_transport_right"]
                ).mean()
            )

        values = {
            "source_donor_transport": float(
                patch["continuous_normalized_transport"].mean()
            ),
            "exact_residual_mediation": paired_difference(exact_control, exact_block),
            "count_axis_mediation": paired_difference(count_control, count_block),
            "terminal_count_adoption": float(patch["terminal_count_adoption"].mean()),
        }
        rows.extend(
            {
                "endpoint": endpoint,
                "set_id": str(set_id),
                "set_role": str(set_role),
                "seed": int(seed),
                "value": float(value),
            }
            for endpoint, value in values.items()
        )
    return pd.DataFrame(rows)


def build_clean_seed_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "set_id",
        "set_role",
        "seed",
        "gold_count",
        "baseline_correct",
        "clean_correct_failure",
        "delta_expected_count_absolute_error",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise RuntimeError(f"Clean ablation detail lacks columns: {missing}")
    rows: list[dict[str, Any]] = []
    for (set_id, set_role, seed), group in detail.groupby(
        ["set_id", "set_role", "seed"], sort=True
    ):
        if group["gold_count"].duplicated().any():
            raise RuntimeError("Clean ablation contains duplicate count rows")
        eligible = group[group["baseline_correct"].astype(bool)]
        if eligible.empty:
            raise RuntimeError(
                f"Clean-correct failure rate is undefined for {set_id}/{seed}"
            )
        values = {
            "clean_correct_failure_rate": float(
                eligible["clean_correct_failure"].mean()
            ),
            "clean_delta_absolute_error": float(
                group["delta_expected_count_absolute_error"].mean()
            ),
        }
        rows.extend(
            {
                "endpoint": endpoint,
                "set_id": str(set_id),
                "set_role": str(set_role),
                "seed": int(seed),
                "value": float(value),
            }
            for endpoint, value in values.items()
        )
    return pd.DataFrame(rows)


def add_candidate_specificity(
    metrics: pd.DataFrame, selection: Mapping[str, Any]
) -> pd.DataFrame:
    candidate_id = str(selection["candidate"]["set_id"])
    control_ids = {str(entry["set_id"]) for entry in selection["matched_controls"]}
    candidate = metrics[metrics["set_id"].eq(candidate_id)]
    controls = (
        metrics[metrics["set_id"].isin(control_ids)]
        .groupby(["endpoint", "seed"], as_index=False)["value"]
        .mean()
    )
    merged = candidate.merge(
        controls,
        on=["endpoint", "seed"],
        how="inner",
        suffixes=("_candidate", "_control"),
        validate="one_to_one",
    )
    expected = candidate[["endpoint", "seed"]].drop_duplicates().shape[0]
    if len(merged) != expected:
        raise RuntimeError("Residual candidate/control specificity grid is incomplete")
    specificity = pd.DataFrame(
        {
            "endpoint": merged["endpoint"],
            "set_id": candidate_id,
            "set_role": "candidate_specificity",
            "seed": merged["seed"].astype(int),
            "value": merged["value_candidate"] - merged["value_control"],
        }
    )
    return pd.concat([metrics, specificity], ignore_index=True)


def summarize_metrics(
    metrics: pd.DataFrame, config: GemmaResidualConfig
) -> pd.DataFrame:
    rows = []
    for (endpoint, set_id, set_role), group in metrics.groupby(
        ["endpoint", "set_id", "set_role"], sort=True
    ):
        values = group.sort_values("seed")["value"].to_numpy(float)
        mean, low, high = _bootstrap_ci(
            values,
            repetitions=config.bootstrap_repetitions,
            seed=446_000 + sum(ord(char) for char in str(endpoint) + str(set_role)),
        )
        rows.append(
            {
                "endpoint": str(endpoint),
                "set_id": str(set_id),
                "set_role": str(set_role),
                "mean": float(mean),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "one_sided_exact_sign_flip_p": float(
                    exact_sign_flip_p(values, alternative="greater")
                ),
                "seeds": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def primary_decision(
    summary: pd.DataFrame, config: GemmaResidualConfig
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    all_components: list[dict[str, Any]] = []
    for endpoint in endpoint_registry(config):
        components = []
        for role in ("candidate_core", "candidate_specificity"):
            selected = summary[
                summary["endpoint"].eq(endpoint) & summary["set_role"].eq(role)
            ]
            if len(selected) != 1:
                raise RuntimeError(f"Residual summary lacks {endpoint}/{role}")
            row = selected.iloc[0]
            component = {
                "endpoint": endpoint,
                "role": role,
                "mean": float(row["mean"]),
                "ci95_low": float(row.ci95_low),
                "ci95_high": float(row.ci95_high),
                "p": float(row.one_sided_exact_sign_flip_p),
                "passes_p": bool(
                    float(row.one_sided_exact_sign_flip_p) <= config.primary_alpha
                ),
                "passes_ci": bool(float(row.ci95_low) > 0),
            }
            components.append(component)
            all_components.append(component)
        family_p = max(component["p"] for component in components)
        families[endpoint] = {
            "intersection_union_p": float(family_p),
            "passes_alpha_and_ci": bool(
                family_p <= config.primary_alpha
                and all(component["passes_ci"] for component in components)
            ),
            "components": components,
        }
    global_p = max(component["p"] for component in all_components)
    supported = bool(
        global_p <= config.primary_alpha
        and all(component["passes_ci"] for component in all_components)
    )
    return {
        "alpha": float(config.primary_alpha),
        "decision_rule": (
            "registered_endpoints_candidate_core_and_matched_specificity_"
            "all_positive_with_ci_and_exact_p"
        ),
        "families": families,
        "global_intersection_union_p": float(global_p),
        "full_residual_count_path_support": supported,
        "interpretation": (
            (
                "Frozen broad bank is naturally important for counting and "
                "causally writes a distributed count-aligned residual state "
                "used by the terminal answer computation."
                if config.require_clean_necessity
                else "Frozen broad bank causally writes a distributed count-aligned "
                "residual state used by the terminal answer computation."
            )
            if supported
            else "At least one required residual-path evidence gate failed."
        ),
    }


def audit_campaign(run_root: str | Path, config: GemmaResidualConfig) -> dict[str, Any]:
    root = Path(run_root)
    discovery = stage_root(root, config.model_label, DISCOVERY_STAGE)
    confirmation = stage_root(root, config.model_label, CONFIRMATION_STAGE)
    detail = pd.read_csv(confirmation / "residual_detail.csv.gz")
    clean_path = confirmation / "clean_ablation_detail.csv.gz"
    selection = json.loads((discovery / "selection.json").read_text(encoding="utf-8"))
    expected_rows = (
        len(config.confirmation_seeds)
        * (1 + len(config.matched_control_sets))
        * len(config.donor_pairs)
        * 5
    )
    observed_seeds = set(detail["seed"].astype(int))
    expected_clean_rows = (
        len(config.confirmation_seeds)
        * (1 + len(config.matched_control_sets))
        * len(config.counts)
        if config.require_clean_necessity
        else 0
    )
    clean_detail = (
        pd.read_csv(clean_path)
        if config.require_clean_necessity and clean_path.is_file()
        else pd.DataFrame()
    )
    checks = {
        "dataset_rows": sum(1 for _ in (root / "dataset/stimuli.jsonl").open("rb"))
        == len(config.dataset_seeds) * len(config.counts),
        "confirmation_rows": len(detail) == expected_rows,
        "confirmation_seed_shards": len(
            list((confirmation / "shards").glob("seed*.csv.gz"))
        )
        == len(config.confirmation_seeds),
        "clean_ablation_rows": (
            not config.require_clean_necessity
            or len(clean_detail) == expected_clean_rows
        ),
        "clean_ablation_seed_shards": (
            not config.require_clean_necessity
            or len(list((confirmation / "clean_ablation_shards").glob("seed*.csv.gz")))
            == len(config.confirmation_seeds)
        ),
        "clean_ablation_seed_grid": (
            not config.require_clean_necessity
            or set(clean_detail.get("seed", pd.Series(dtype=int)).astype(int))
            == set(config.confirmation_seeds)
        ),
        "clean_ablation_condition": (
            not config.require_clean_necessity
            or set(clean_detail.get("condition", pd.Series(dtype=str)).astype(str))
            == {"clean_zero_z_bank_ablation"}
        ),
        "clean_ablation_design_hash": (
            not config.require_clean_necessity
            or (
                clean_detail.get("design_hash", pd.Series(dtype=str)).nunique() == 1
                and set(clean_detail["design_hash"].astype(str))
                == set(detail["design_hash"].astype(str))
            )
        ),
        "all_confirmation_seeds": observed_seeds == set(config.confirmation_seeds),
        "three_frozen_controls": len(selection["matched_controls"])
        == len(config.matched_control_sets),
        "frozen_candidate_exact": tuple(
            tuple(int(part) for part in site)
            for site in selection["candidate"]["sites"]
        )
        == tuple(tuple(site.to_list()) for site in config.candidate_sites),
        "frozen_controls_exact": tuple(
            tuple(tuple(int(part) for part in site) for site in entry["sites"])
            for entry in selection["matched_controls"]
        )
        == tuple(
            tuple(tuple(site.to_list()) for site in site_set)
            for site_set in config.matched_control_sets
        ),
        "frozen_variant_recorded": selection.get("mechanism_variant")
        == config.mechanism_variant,
        "control_sampling_seed_recorded": selection.get("matched_control_sampling_seed")
        == config.matched_control_sampling_seed,
        "source_selection_precedes_outcomes": not bool(
            selection["selection_uses_causal_outcomes"]
        ),
        "residual_selection_precedes_confirmation": not bool(
            selection["residual_layer_selection_uses_confirmation_outcomes"]
        ),
        "single_design_hash": detail["design_hash"].nunique() == 1,
        "discovery_complete": (discovery / "complete.json").is_file(),
        "smoke_complete": stage_root(root, config.model_label, SMOKE_STAGE)
        .joinpath("complete.json")
        .is_file(),
        "model_complete": (
            root / "models" / config.model_label / "residual_path_complete.json"
        ).is_file(),
        "no_raw_attention": not any(root.rglob("*raw_attention*")),
        "no_full_hidden": not any(root.rglob("*full_hidden*")),
    }
    return {
        "schema_version": "realistic_niah_v4_4_4_residual_audit_v1",
        "expected": {
            "confirmation_rows": expected_rows,
            "clean_ablation_rows": expected_clean_rows,
        },
        "observed": {
            "confirmation_rows": len(detail),
            "clean_ablation_rows": len(clean_detail),
        },
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["primary_decision"]
    return "\n".join(
        (
            "# V4.4.4 Gemma frozen-bank residual mediation",
            "",
            f"Selected mediator: L{payload['selected_mediator_layer']} (discovery only).",
            f"Global IUT p: {decision['global_intersection_union_p']:.8g}.",
            f"Full residual count-path support: {decision['full_residual_count_path_support']}.",
            "",
            "This gated fallback does not identify a unique downstream head and does not "
            "replace the retained negative localized-OV results.",
        )
    )


def analyze_campaign(
    *, run_root: str | Path, config: GemmaResidualConfig
) -> dict[str, Any]:
    root = Path(run_root)
    discovery = stage_root(root, config.model_label, DISCOVERY_STAGE)
    confirmation = stage_root(root, config.model_label, CONFIRMATION_STAGE)
    detail = pd.read_csv(confirmation / "residual_detail.csv.gz")
    selection = json.loads((discovery / "selection.json").read_text(encoding="utf-8"))
    seed_metrics = build_seed_metrics(detail)
    if config.require_clean_necessity:
        clean_detail = pd.read_csv(confirmation / "clean_ablation_detail.csv.gz")
        seed_metrics = pd.concat(
            [seed_metrics, build_clean_seed_metrics(clean_detail)],
            ignore_index=True,
        )
    seed_metrics = add_candidate_specificity(seed_metrics, selection)
    summary = summarize_metrics(seed_metrics, config)
    decision = primary_decision(summary, config)
    audit = audit_campaign(root, config)
    if not audit["all_checks_pass"]:
        raise RuntimeError(f"Residual campaign audit failed: {audit['checks']}")
    payload = {
        "schema_version": "realistic_niah_v4_4_4_residual_analysis_v1",
        "config": config.to_dict(),
        "selected_mediator_layer": int(selection["residual_mediator"]["layer"]),
        "discovery_layer_scores": selection["residual_mediator"]["scores"],
        "primary_decision": decision,
        "summary": _json_records(summary),
        "audit": audit,
    }
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv_gzip(seed_metrics, output / "residual_seed_metrics.csv.gz")
    atomic_csv_gzip(summary, output / "residual_endpoint_summary.csv.gz")
    atomic_json(output / "realistic_niah_v4_4_4_residual_analysis.json", payload)
    atomic_json(output / "residual_audit.json", audit)
    atomic_text(output / "realistic_niah_v4_4_4_residual_report.md", _markdown(payload))
    return payload
