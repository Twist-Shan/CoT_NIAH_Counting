#!/usr/bin/env python3
"""Run the preregistered V2 focused empirical-law method on V3.1 data.

This adapter deliberately changes only the model set and the N/L grid.  The
candidate laws, five-fold held-condition cross-validation rule, HC3 inference,
BH correction, shared-law selection thresholds, and leave-one-model-out (LOMO)
stability analysis are imported from the archived V2 analysis script.

The adapter does *not* run the later V3.1 nested held-seed/held-N/held-L
bootstrap design.  That design was never part of the earlier focused report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MODE_ORDER = [
    "direct",
    "enumeration_index",
    "enumeration_bullet",
    "native_thinking",
]

FAMILY_ORDER = [
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "Gemma4-26B-A4B",
    "Gemma4-31B",
    "Nemotron-3-Nano-4B",
    "Nemotron-Nano-v2-9B",
    "GLM 9B pair",
    "Ministral 8B pair",
]

MODEL_BY_FAMILY_MODE = {
    (family, mode): family
    for family in [
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
        "Gemma4-E4B",
        "Gemma4-12B",
        "Gemma4-26B-A4B",
        "Gemma4-31B",
        "Nemotron-3-Nano-4B",
        "Nemotron-Nano-v2-9B",
    ]
    for mode in MODE_ORDER
} | {
    ("GLM 9B pair", "direct"): "GLM-4-9B-0414",
    ("GLM 9B pair", "enumeration_index"): "GLM-4-9B-0414",
    ("GLM 9B pair", "enumeration_bullet"): "GLM-4-9B-0414",
    ("GLM 9B pair", "native_thinking"): "GLM-Z1-9B-0414",
    ("Ministral 8B pair", "direct"): "Ministral-3-Instruct-8B",
    ("Ministral 8B pair", "enumeration_index"): "Ministral-3-Instruct-8B",
    ("Ministral 8B pair", "enumeration_bullet"): "Ministral-3-Instruct-8B",
    ("Ministral 8B pair", "native_thinking"): "Ministral-3-Reasoning-8B",
}

N_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20]
L_LEVELS = [1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000]
SEEDS_PER_CELL = 30
REQUESTS_PER_MODEL_MODE = len(N_LEVELS) * len(L_LEVELS) * SEEDS_PER_CELL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--legacy-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_legacy_module(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_empirical_law", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy method from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FAMILY_ORDER = FAMILY_ORDER
    module.MODE_ORDER = MODE_ORDER
    module.MODEL_BY_FAMILY_MODE = MODEL_BY_FAMILY_MODE
    module.N_LEVELS = N_LEVELS
    module.L_LEVELS = L_LEVELS
    return module


def load_current_requests(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "request_id",
        "model_label",
        "model_revision",
        "prompt_mode",
        "stimulus_id",
        "seed",
        "L",
        "N",
        "predicted_count",
        "signed_deviation",
        "absolute_deviation",
    ]
    data = pd.read_csv(input_csv, usecols=usecols).rename(
        columns={
            "model_label": "model",
            "model_revision": "source_version",
            "prompt_mode": "mode",
            "signed_deviation": "signed_error",
            "absolute_deviation": "absolute_error",
        }
    )
    parts: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        for mode in MODE_ORDER:
            source_model = MODEL_BY_FAMILY_MODE[(family, mode)]
            part = data[
                data["model"].eq(source_model) & data["mode"].eq(mode)
            ].copy()
            if len(part) != REQUESTS_PER_MODEL_MODE:
                raise ValueError(
                    f"Expected {REQUESTS_PER_MODEL_MODE} requests for "
                    f"{family}/{mode} from {source_model}; found {len(part)}"
                )
            revisions = part["source_version"].dropna().unique()
            if len(revisions) != 1:
                raise ValueError(
                    f"Expected one revision for {source_model}/{mode}; "
                    f"found {revisions.tolist()}"
                )
            part["analysis_family"] = family
            parts.append(part)
            mapping_rows.append(
                {
                    "analysis_family": family,
                    "mode": mode,
                    "source_model": source_model,
                    "source_version": str(revisions[0]),
                    "requests": len(part),
                }
            )
    focused = pd.concat(parts, ignore_index=True)
    expected = len(FAMILY_ORDER) * len(MODE_ORDER) * REQUESTS_PER_MODEL_MODE
    if len(focused) != expected:
        raise ValueError(f"Expected {expected} focused rows; found {len(focused)}")
    if focused["request_id"].duplicated().any():
        duplicated = int(focused["request_id"].duplicated().sum())
        raise ValueError(f"Focused request IDs are not unique: {duplicated}")
    observed_n = sorted(focused["N"].unique().tolist())
    observed_l = sorted(focused["L"].unique().tolist())
    if observed_n != N_LEVELS or observed_l != L_LEVELS:
        raise ValueError(
            f"Unexpected grid: N={observed_n}, L={observed_l}; "
            f"expected N={N_LEVELS}, L={L_LEVELS}"
        )
    return focused, pd.DataFrame(mapping_rows)


def build_condition_table(data: pd.DataFrame) -> pd.DataFrame:
    working = data.copy()
    working["signed_error"] = pd.to_numeric(
        working["signed_error"], errors="coerce"
    )
    keys = ["analysis_family", "mode", "N", "L"]
    base = (
        working.groupby(keys, as_index=False)
        .agg(
            total_n=("request_id", "size"),
            source_model=("model", "first"),
            source_version=("source_version", "first"),
        )
    )
    parsed = working[working["signed_error"].notna()]
    summary = (
        parsed.groupby(keys, as_index=False)
        .agg(
            parsed_n=("signed_error", "size"),
            signed_mean_deviation=("signed_error", "mean"),
            signed_sd=("signed_error", "std"),
        )
    )
    cells = base.merge(summary, on=keys, how="left", validate="one_to_one")
    cells["parsed_n"] = cells["parsed_n"].fillna(0).astype(int)
    cells["parse_rate"] = cells["parsed_n"] / cells["total_n"]
    cells["L_k"] = cells["L"] / 1000.0
    cells["lnN"] = np.log(cells["N"])
    cells["lnL_k"] = np.log(cells["L_k"])
    expected = len(FAMILY_ORDER) * len(MODE_ORDER) * len(N_LEVELS) * len(L_LEVELS)
    if len(cells) != expected:
        raise ValueError(f"Expected {expected} condition rows; found {len(cells)}")
    if not cells["total_n"].eq(SEEDS_PER_CELL).all():
        bad = cells.loc[~cells["total_n"].eq(SEEDS_PER_CELL), keys + ["total_n"]]
        raise ValueError(f"Every N/L cell must have 30 seeds; examples:\n{bad.head()}")
    if cells["signed_mean_deviation"].isna().any():
        bad = cells.loc[cells["signed_mean_deviation"].isna(), keys]
        raise ValueError(f"At least one N/L cell has no parsed response:\n{bad.head()}")
    return cells.sort_values(keys).reset_index(drop=True)


def make_fit_candidate(legacy):
    def fit_candidate(
        cells: pd.DataFrame,
        family: str,
        mode: str,
        formula: tuple[str, ...],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        subset = cells[
            cells["analysis_family"].eq(family)
            & cells["mode"].eq(mode)
            & cells["signed_mean_deviation"].notna()
        ].sort_values(["N", "L"]).reset_index(drop=True)
        expected_conditions = len(N_LEVELS) * len(L_LEVELS)
        if len(subset) != expected_conditions:
            raise ValueError(
                f"Expected {expected_conditions} defined conditions for "
                f"{family}/{mode}; found {len(subset)}"
            )
        observed = subset["signed_mean_deviation"].to_numpy(dtype=float)
        raw_features = legacy.predictor_matrix(subset, formula)
        raw_design = np.column_stack([np.ones(len(subset)), raw_features])
        beta = np.linalg.lstsq(raw_design, observed, rcond=None)[0]
        fitted = raw_design @ beta
        full_metrics = legacy.regression_metrics(
            observed, fitted, parameters=len(formula) + 1
        )

        standardized, _, _ = legacy.standardized_design(raw_features)
        standardized_beta = np.linalg.lstsq(standardized, observed, rcond=None)[0]
        y_sd = float(observed.std(ddof=1))
        standardized_effects = (
            standardized_beta[1:] / y_sd
            if y_sd > 1e-12
            else np.full(len(formula), np.nan)
        )
        condition_number = float(np.linalg.cond(standardized))

        oof = np.full(len(subset), np.nan)
        folds = legacy.condition_fold(subset)
        for fold in range(5):
            test_mask = folds == fold
            train_design, center, scale = legacy.standardized_design(
                raw_features[~test_mask]
            )
            test_design, _, _ = legacy.standardized_design(
                raw_features[test_mask], center, scale
            )
            fold_beta = np.linalg.lstsq(
                train_design, observed[~test_mask], rcond=None
            )[0]
            oof[test_mask] = test_design @ fold_beta
        cv_metrics = legacy.regression_metrics(observed, oof)

        robust_se, robust_p, confidence = legacy.robust_coefficient_table(
            raw_design, observed, beta
        )
        names = ["intercept", *formula]
        effect_by_name = dict(zip(formula, standardized_effects))
        coefficient_rows: list[dict[str, object]] = []
        for index, name in enumerate(names):
            coefficient_rows.append(
                {
                    "analysis_family": family,
                    "mode": mode,
                    "formula": legacy.FORMULA_ID[formula],
                    "formula_label": legacy.FORMULA_LABEL[legacy.FORMULA_ID[formula]],
                    "term": name,
                    "coefficient": float(beta[index]),
                    "hc3_se": float(robust_se[index]),
                    "hc3_ci_low": float(confidence[index, 0]),
                    "hc3_ci_high": float(confidence[index, 1]),
                    "hc3_p": float(robust_p[index]),
                    "standardized_effect": (
                        float(effect_by_name[name])
                        if name in effect_by_name
                        else math.nan
                    ),
                }
            )
        formula_id = legacy.FORMULA_ID[formula]
        fit_row = {
            "analysis_family": family,
            "mode": mode,
            "source_model": subset["source_model"].iloc[0],
            "formula": formula_id,
            "formula_label": legacy.FORMULA_LABEL[formula_id],
            "formula_class": legacy.FORMULA_CLASS[formula_id],
            "special_term": legacy.SPECIAL_TERM[formula_id],
            "parent_formula": legacy.PARENT_FORMULA[formula_id],
            "terms": "+".join(formula),
            "predictors": len(formula),
            "parameters": len(formula) + 1,
            "n_conditions": len(subset),
            "mean_parse_rate": float(subset["parse_rate"].mean()),
            "minimum_parsed_n": int(subset["parsed_n"].min()),
            "condition_number_standardized": condition_number,
            "in_sample_r2": full_metrics["r2"],
            "adjusted_r2": full_metrics["adjusted_r2"],
            "in_sample_rmse": full_metrics["rmse"],
            "in_sample_mae": full_metrics["mae"],
            "aic": full_metrics["aic"],
            "aicc": full_metrics["aicc"],
            "bic": full_metrics["bic"],
            "cv_r2": cv_metrics["r2"],
            "cv_rmse": cv_metrics["rmse"],
            "cv_mae": cv_metrics["mae"],
        }
        return fit_row, coefficient_rows

    return fit_candidate


def write_outputs(
    output: Path,
    input_csv: Path,
    legacy_script: Path,
    legacy,
    mapping: pd.DataFrame,
    cells: pd.DataFrame,
    fits: pd.DataFrame,
    coefficients: pd.DataFrame,
    summary: pd.DataFrame,
    choices: pd.DataFrame,
    selected: pd.DataFrame,
    selected_coefficients: pd.DataFrame,
    lomo: pd.DataFrame,
) -> None:
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frames = {
        "model_mode_mapping.csv": mapping,
        "condition_signed_bias.csv": cells,
        "candidate_fit_metrics.csv": fits,
        "candidate_coefficients.csv": coefficients,
        "mode_candidate_summary.csv": summary,
        "selected_mode_laws.csv": choices,
        "selected_model_fit_metrics.csv": selected,
        "selected_model_coefficients.csv": selected_coefficients,
        "lomo_structure_selection.csv": lomo,
        "selected_term_support.csv": legacy.build_term_support(selected_coefficients),
    }
    for name, frame in frames.items():
        frame.to_csv(tables / name, index=False)

    # The legacy candidate heatmaps are dimensioned from FAMILY_ORDER and are
    # therefore safe to reuse for the expanded 12-family model set.
    legacy.plot_candidate_heatmaps_by_mode(fits, choices, figures)

    selected_records = choices[
        [
            "mode",
            "selected_formula",
            "selected_formula_label",
            "formula_class",
            "median_cv_r2",
            "q25_cv_r2",
            "median_cv_mae",
            "lomo_formula_stability",
            "evidence_reading",
        ]
    ].to_dict(orient="records")
    manifest = {
        "analysis": "v3_1_model_set_with_v2_focused_empirical_law_method",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method_change_timing": (
            "The decision to reuse the V2 method was made before inspecting "
            "the V3.1 selected-law results."
        ),
        "input": str(input_csv.resolve()),
        "input_sha256": legacy.file_sha256(input_csv),
        "legacy_method_script": str(legacy_script.resolve()),
        "legacy_method_script_sha256": legacy.file_sha256(legacy_script),
        "requests": len(FAMILY_ORDER) * len(MODE_ORDER) * REQUESTS_PER_MODEL_MODE,
        "analysis_families": len(FAMILY_ORDER),
        "physical_models": int(mapping["source_model"].nunique()),
        "model_mode_slots": len(mapping),
        "N_levels": N_LEVELS,
        "L_levels": L_LEVELS,
        "seeds_per_cell": SEEDS_PER_CELL,
        "condition_rows": len(cells),
        "candidate_laws": len(legacy.FORMULAS),
        "candidate_fits": len(fits),
        "outcome": (
            "mean signed deviation among parseable responses in each "
            "model-family x mode x N x L cell"
        ),
        "candidate_classes": {
            "additive": len(legacy.ADDITIVE_FORMULAS),
            "hierarchical_first_order_interaction": len(legacy.INTERACTION_FORMULAS),
            "density": len(legacy.DENSITY_FORMULAS),
        },
        "validation": {
            "type": "five-fold held-condition cross-validation",
            "fold_rule": "(index(N) + index(L)) mod 5",
            "held_seed_nested_refit": False,
            "held_N_nested_refit": False,
            "held_L_nested_refit": False,
        },
        "inference": {
            "coefficient_covariance": "HC3",
            "multiple_testing": "Benjamini-Hochberg",
            "cross_model_special_term_test": "one-sided Wilcoxon with BH",
            "leave_one_model_out_structure_stability": True,
            "bootstrap_repetitions": 0,
        },
        "selection_thresholds": {
            "practical_standardized_effect": legacy.PRACTICAL_EFFECT_THRESHOLD,
            "special_term_significant_model_fraction": legacy.SPECIAL_SIGNIFICANT_FRACTION_THRESHOLD,
            "special_cv_r2_gain": legacy.SPECIAL_CV_GAIN_THRESHOLD,
            "special_cv_gain_q": legacy.SPECIAL_CV_GAIN_Q_THRESHOLD,
            "median_cv_r2_tolerance": legacy.MEDIAN_CV_TOLERANCE,
            "q25_cv_r2_tolerance": legacy.Q25_CV_TOLERANCE,
        },
        "selected_laws": selected_records,
        "tables": {name: len(frame) for name, frame in frames.items()},
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state = {
        "stage": "complete",
        "candidate_fits_completed": len(fits),
        "candidate_fits_expected": (
            len(FAMILY_ORDER) * len(MODE_ORDER) * len(legacy.FORMULAS)
        ),
        "selected_laws": selected_records,
    }
    (output / "analysis_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def verify_outputs(output: Path, legacy) -> dict[str, object]:
    tables = output / "tables"
    cells = pd.read_csv(tables / "condition_signed_bias.csv")
    fits = pd.read_csv(tables / "candidate_fit_metrics.csv")
    choices = pd.read_csv(tables / "selected_mode_laws.csv")
    selected = pd.read_csv(tables / "selected_model_fit_metrics.csv")
    lomo = pd.read_csv(tables / "lomo_structure_selection.csv")
    expected_cells = len(FAMILY_ORDER) * len(MODE_ORDER) * len(N_LEVELS) * len(L_LEVELS)
    expected_fits = len(FAMILY_ORDER) * len(MODE_ORDER) * len(legacy.FORMULAS)
    checks = {
        "condition_rows": (len(cells), expected_cells),
        "candidate_fits": (len(fits), expected_fits),
        "selected_laws": (len(choices), len(MODE_ORDER)),
        "selected_model_mode_fits": (
            len(selected), len(FAMILY_ORDER) * len(MODE_ORDER)
        ),
        "lomo_rows": (len(lomo), len(FAMILY_ORDER) * len(MODE_ORDER)),
    }
    bad = {name: values for name, values in checks.items() if values[0] != values[1]}
    if bad:
        raise ValueError(f"Output row-count verification failed: {bad}")
    if set(choices["mode"]) != set(MODE_ORDER):
        raise ValueError("Selected-law table does not contain exactly four modes")
    if not set(choices["selected_formula"]).issubset(legacy.FORMULA_BY_ID):
        raise ValueError("Selected-law table contains an unknown candidate formula")
    if not fits["n_conditions"].eq(len(N_LEVELS) * len(L_LEVELS)).all():
        raise ValueError("At least one candidate fit used an incomplete condition grid")
    return {
        "passed": True,
        **{name: actual for name, (actual, _) in checks.items()},
        "selected": choices[
            [
                "mode",
                "selected_formula_label",
                "median_cv_r2",
                "q25_cv_r2",
                "median_cv_mae",
                "lomo_formula_stability",
            ]
        ].to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    input_csv = args.input.resolve()
    legacy_script = args.legacy_script.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    legacy = load_legacy_module(legacy_script)
    legacy.fit_candidate = make_fit_candidate(legacy)

    focused, mapping = load_current_requests(input_csv)
    cells = build_condition_table(focused)
    fits, coefficients = legacy.fit_all_candidates(cells)
    summary, choices, selected, lomo = legacy.select_mode_laws(
        fits, coefficients
    )
    selected_coefficients = legacy.selected_coefficients(coefficients, choices)
    write_outputs(
        output,
        input_csv,
        legacy_script,
        legacy,
        mapping,
        cells,
        fits,
        coefficients,
        summary,
        choices,
        selected,
        selected_coefficients,
        lomo,
    )
    result = verify_outputs(output, legacy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
