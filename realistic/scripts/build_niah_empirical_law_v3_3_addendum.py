#!/usr/bin/env python3
"""Add the audited V3.3 long-context regression extension to the V3.2 report.

The script is idempotent: it replaces the delimited V3.3 section if present.
It never refits the V3.2 confirmatory laws.  All V3.3 candidate selection,
fixed-N diagnostics, and N-fixed shared-slope models are labelled exploratory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_realistic_niah_v3_3_regression_scan import (
    BIAS_FAMILY,
    HEADLINE_ACCURACY,
    MAE_FAMILY,
    MODELS,
    MODES,
    N_LEVELS,
    add_predictors,
    build_count_error_cells,
    coefficient_prediction,
    normalize_requests,
)


START = "<!-- V3_3_LONG_CONTEXT_START -->"
END = "<!-- V3_3_LONG_CONTEXT_END -->"
STYLE_START = "<!-- V3_3_STYLE_START -->"
STYLE_END = "<!-- V3_3_STYLE_END -->"

AURORA = {
    "indigo": "#23165C",
    "violet": "#6750E8",
    "cyan": "#00C2FF",
    "yellow": "#F6E36A",
    "teal": "#00D4B4",
    "pink": "#FF5FA2",
    "black": "#161923",
    "paper": "#F8FBFF",
    "gray": "#8190A5",
}
SCALING = {
    "purple": "#2C115F",
    "magenta": "#9C179E",
    "orange": "#ED7953",
    "yellow": "#F0F921",
}

MODE_LABEL = {"direct": "Non-thinking", "native_thinking": "Native-thinking"}
OUTCOME_LABEL = {
    HEADLINE_ACCURACY: "Accuracy",
    MAE_FAMILY: "10% trimmed conditional MAE",
    BIAS_FAMILY: "10% trimmed signed bias",
}
FORMULA = {
    "intercept": "1",
    "N": "N",
    "L_k": "L_k",
    "logN": r"\ln N",
    "logL": r"\ln L_k",
    "N__L_k": "N + L_k",
    "logN__logL": r"\ln N + \ln L_k",
    "N__logL": r"N + \ln L_k",
    "logN__L_k": r"\ln N + L_k",
    "N__L_k__N_x_L_k": r"N + L_k + NL_k",
    "logN__logL__logN_x_logL": r"\ln N + \ln L_k + (\ln N)(\ln L_k)",
    "N__logL__N_x_logL": r"N + \ln L_k + N\ln L_k",
    "logN__L_k__logN_x_L_k": r"\ln N + L_k + L_k\ln N",
    "invN": r"N^{-1}",
    "invN__L_k": r"N^{-1} + L_k",
    "invN__logL": r"N^{-1} + \ln L_k",
    "invN__L_k__invN_x_L_k": r"N^{-1} + L_k + L_k/N",
    "invN__logL__invN_x_logL": r"N^{-1} + \ln L_k + (\ln L_k)/N",
}
# Plain-text equivalents for Matplotlib annotations.  The HTML equations use
# MathJax, but chart annotations should never expose raw TeX commands.
PLOT_FORMULA = {
    "intercept": "1",
    "N": "N",
    "L_k": "L/1k",
    "logN": "ln N",
    "logL": "ln(L/1k)",
    "N__L_k": "N + L/1k",
    "logN__logL": "ln N + ln(L/1k)",
    "N__logL": "N + ln(L/1k)",
    "logN__L_k": "ln N + L/1k",
    "N__L_k__N_x_L_k": "N + L/1k + N(L/1k)",
    "logN__logL__logN_x_logL": "ln N + ln(L/1k) + ln N ln(L/1k)",
    "N__logL__N_x_logL": "N + ln(L/1k) + N ln(L/1k)",
    "logN__L_k__logN_x_L_k": "ln N + L/1k + (L/1k) ln N",
    "invN": "1/N",
    "invN__L_k": "1/N + L/1k",
    "invN__logL": "1/N + ln(L/1k)",
    "invN__L_k__invN_x_L_k": "1/N + L/1k + L/(1k N)",
    "invN__logL__invN_x_logL": "1/N + ln(L/1k) + ln(L/1k)/N",
}
TERM_LATEX = {
    "intercept": "1",
    "N": "N",
    "L_k": "L_k",
    "logN": r"\ln N",
    "logL": r"\ln L_k",
    "N_x_L_k": "NL_k",
    "logN_x_logL": r"(\ln N)(\ln L_k)",
    "N_x_logL": r"N\ln L_k",
    "logN_x_L_k": r"L_k\ln N",
    "invN": r"N^{-1}",
    "invN_x_L_k": r"L_k/N",
    "invN_x_logL": r"(\ln L_k)/N",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    analysis = (
        root
        / "outputs"
        / "anvil_realistic_niah_v3_3_long_context_20260906_holdout"
        / "analysis"
        / "v3_3_regression_scan"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=analysis)
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "NiaH_Empirical-law_report.html",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=root / "reports" / "assets" / "niah_empirical_law_v3_3",
    )
    parser.add_argument(
        "--frozen-accuracy-coefficients",
        type=Path,
        default=(
            root
            / "outputs"
            / "anvil_realistic_niah_v3_1_20260819_formal"
            / "analysis"
            / "v3_2_inverse_n_candidate_extension"
            / "tables"
            / "selected_model_coefficients.csv"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def format_number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def html_table(frame: pd.DataFrame) -> str:
    return frame.to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=False,
        justify="left",
    )


def scaling_colors(levels: list[int]) -> list[str]:
    colors = matplotlib.colormaps["plasma_r"](
        np.linspace(0.0, 1.0, len(levels))
    )
    return [mcolors.to_hex(color) for color in colors]


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": AURORA["paper"],
            "axes.facecolor": AURORA["paper"],
            "axes.edgecolor": "#D8DEE9",
            "axes.labelcolor": AURORA["black"],
            "axes.titlecolor": AURORA["black"],
            "text.color": AURORA["black"],
            "xtick.color": "#5F6B7A",
            "ytick.color": "#5F6B7A",
            "grid.color": AURORA["gray"],
            "grid.alpha": 0.20,
            "grid.linewidth": 0.55,
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
        }
    )


def add_discrete_colorbar(
    fig: plt.Figure,
    axes: np.ndarray,
    levels: list[int],
    colors: list[str],
    label: str,
    labels: list[str] | None = None,
) -> None:
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(len(levels) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(
        scalar,
        ax=list(np.ravel(axes)),
        ticks=np.arange(len(levels)),
        fraction=0.024,
        pad=0.018,
        aspect=32,
    )
    colorbar.ax.set_yticklabels(labels or [str(item) for item in levels], fontsize=7.2)
    colorbar.set_label(label, fontsize=8.4)
    colorbar.outline.set_edgecolor("#D8DEE9")


def selected_prediction(
    frame: pd.DataFrame,
    selections: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    mode: str,
    outcome: str,
) -> tuple[np.ndarray, str, float]:
    selected = selections.loc[
        selections["dataset"].eq(dataset)
        & selections["model_label"].eq(model)
        & selections["prompt_mode"].eq(mode)
        & selections["outcome_family"].eq(outcome)
    ].iloc[0]
    candidate = str(selected["selected_candidate"])
    coef = coefficients.loc[
        coefficients["dataset"].eq(dataset)
        & coefficients["model_label"].eq(model)
        & coefficients["prompt_mode"].eq(mode)
        & coefficients["outcome_family"].eq(outcome)
        & coefficients["candidate"].eq(candidate)
    ]
    prediction = coefficient_prediction(frame, coef)
    if outcome == HEADLINE_ACCURACY:
        prediction = expit(prediction)
    return prediction, candidate, float(selected["selected_cv_score"])


def response_cells(requests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model_label", "prompt_mode", "N", "L"]
    accuracy = (
        requests.groupby(keys, observed=True, sort=True)
        .agg(observed_accuracy=("exact_count", "mean"), n_requests=("request_id", "size"))
        .reset_index()
    )
    accuracy = add_predictors(accuracy)
    count_error = add_predictors(build_count_error_cells(requests)).rename(
        columns={"comparison_slot": "model_label"}
    )
    return accuracy, count_error


def plot_joint_law(
    cells: pd.DataFrame,
    selections: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    outcome: str,
    observed_column: str,
    horizontal: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.8, 8.1),
        sharey=(outcome == HEADLINE_ACCURACY),
    )
    if horizontal == "L":
        x_levels = sorted(cells["L"].unique().astype(int))
        curve_levels = list(N_LEVELS)
        curve_key = "N"
    else:
        x_levels = list(N_LEVELS)
        curve_levels = sorted(cells["L"].unique().astype(int))
        curve_key = "L"
    colors = scaling_colors(curve_levels)
    for row, mode in enumerate(MODES):
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            block = cells.loc[
                cells["model_label"].eq(model)
                & cells["prompt_mode"].eq(mode)
            ].copy()
            prediction, candidate, score = selected_prediction(
                block,
                selections,
                coefficients,
                dataset="combined_1k_100k",
                model=model,
                mode=mode,
                outcome=outcome,
            )
            block["prediction"] = prediction
            for level, color in zip(curve_levels, colors, strict=True):
                part = block.loc[block[curve_key].eq(level)].sort_values(horizontal)
                ax.plot(
                    part[horizontal],
                    part["prediction"],
                    color=color,
                    linewidth=1.4,
                    alpha=0.92,
                    zorder=2,
                )
                ax.scatter(
                    part[horizontal],
                    part[observed_column],
                    s=17,
                    facecolor=AURORA["paper"],
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.95,
                    zorder=3,
                )
            if horizontal == "L":
                ax.set_xscale("log")
                ax.set_xticks(x_levels)
                ax.set_xticklabels(
                    [f"{value // 1000}k" for value in x_levels],
                    rotation=42,
                    ha="right",
                    fontsize=6.8,
                )
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            else:
                ax.set_xscale("log", base=2)
                ax.set_xticks(x_levels)
                ax.set_xticklabels(
                    [str(value) for value in x_levels],
                    rotation=42,
                    ha="right",
                    fontsize=7.0,
                )
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if outcome == HEADLINE_ACCURACY:
                ax.set_ylim(-0.025, 1.025)
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            elif outcome == BIAS_FAMILY:
                ax.axhline(0, color=AURORA["gray"], linewidth=0.8, zorder=0)
            else:
                lower = min(0.0, float(block["prediction"].min()) * 1.05)
                upper = max(float(block[observed_column].max()), float(block["prediction"].max()))
                ax.set_ylim(lower, upper * 1.06 if upper > 0 else 1.0)
            ax.grid(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{model} · {MODE_LABEL[mode]}", loc="left", fontsize=11.2, fontweight="bold")
            ax.text(
                0.01,
                0.985,
                f"{PLOT_FORMULA[candidate]} · CV {'D²' if outcome == HEADLINE_ACCURACY else 'R²'}={score:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
                color="#475467",
                bbox={"facecolor": AURORA["paper"], "edgecolor": "none", "alpha": 0.83, "pad": 2},
            )
            if row == 1:
                ax.set_xlabel("Passage length L (tokens; log positions)" if horizontal == "L" else "Target count N (log2 positions)")
            if column == 0:
                ax.set_ylabel(OUTCOME_LABEL[outcome])
    fig.suptitle(
        f"V3.3 exploratory {OUTCOME_LABEL[outcome].lower()} laws · {horizontal} on the horizontal axis",
        x=0.075,
        y=0.995,
        ha="left",
        fontsize=15.0,
        fontweight="bold",
    )
    add_discrete_colorbar(
        fig,
        axes,
        curve_levels,
        colors,
        "Target count N" if curve_key == "N" else "Passage length L",
        None if curve_key == "N" else [f"{value // 1000}k" for value in curve_levels],
    )
    fig.text(
        0.075,
        0.015,
        "Open markers: observed condition means · solid curves: selected combined-range law",
        fontsize=8.5,
        color="#667085",
    )
    fig.subplots_adjust(left=0.075, right=0.90, top=0.92, bottom=0.12, hspace=0.30, wspace=0.14)
    fig.savefig(path, dpi=185, bbox_inches="tight", facecolor=AURORA["paper"])
    plt.close(fig)


def plot_frozen_accuracy(
    holdout: pd.DataFrame,
    frozen_coefficients: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.2), sharey=True)
    mode_color = {"direct": SCALING["orange"], "native_thinking": SCALING["purple"]}
    for ax, model in zip(axes, MODELS, strict=True):
        for mode in MODES:
            block = holdout.loc[
                holdout["model_label"].eq(model)
                & holdout["prompt_mode"].eq(mode)
            ].copy()
            coef = frozen_coefficients.loc[
                frozen_coefficients["comparison_slot"].eq(model)
                & frozen_coefficients["prompt_mode"].eq(mode)
                & frozen_coefficients["outcome_family"].eq(HEADLINE_ACCURACY)
            ]
            block["frozen_prediction"] = expit(coefficient_prediction(block, coef))
            summary = (
                block.groupby("L", sort=True)
                .agg(
                    observed=("exact_count", "mean"),
                    predicted=("frozen_prediction", "mean"),
                )
                .reset_index()
            )
            color = mode_color[mode]
            ax.plot(summary["L"], summary["observed"], color=color, linewidth=2.2, marker="o", markersize=4.3)
            ax.plot(summary["L"], summary["predicted"], color=color, linewidth=1.8, linestyle="--", marker="^", markersize=4.0, markerfacecolor=AURORA["paper"])
        levels = sorted(block["L"].unique().astype(int))
        ax.set_xscale("log")
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{value // 1000}k" for value in levels], rotation=35, ha="right")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(model, loc="left", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("V3.3 holdout passage length L")
    axes[0].set_ylabel("Mean exact accuracy across N")
    handles = []
    for mode in MODES:
        for observed, linestyle, marker in [(True, "-", "o"), (False, "--", "^")]:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=mode_color[mode],
                    linestyle=linestyle,
                    marker=marker,
                    markerfacecolor=(mode_color[mode] if observed else AURORA["paper"]),
                    label=f"{MODE_LABEL[mode]} · {'observed' if observed else 'frozen V3.2'}",
                )
            )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 0.945), fontsize=8.2)
    fig.suptitle("Frozen V3.2 accuracy laws on the untouched 25k–100k holdout", x=0.07, y=1.02, ha="left", fontsize=14.5, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.20, wspace=0.12)
    fig.savefig(path, dpi=185, bbox_inches="tight", facecolor=AURORA["paper"])
    plt.close(fig)


def plot_fixed_n10(
    accuracy_cells: pd.DataFrame,
    error_cells: pd.DataFrame,
    selections: pd.DataFrame,
    coefficients: pd.DataFrame,
    path: Path,
) -> None:
    outcomes = [
        (HEADLINE_ACCURACY, "observed_accuracy"),
        (MAE_FAMILY, MAE_FAMILY),
        (BIAS_FAMILY, BIAS_FAMILY),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13.8, 9.2), sharex=True)
    mode_style = {
        "direct": (SCALING["orange"], "-", "o"),
        "native_thinking": (SCALING["purple"], "--", "^"),
    }
    for row, (outcome, observed_column) in enumerate(outcomes):
        source = accuracy_cells if outcome == HEADLINE_ACCURACY else error_cells
        for column, model in enumerate(MODELS):
            ax = axes[row, column]
            for mode in MODES:
                block = source.loc[
                    source["model_label"].eq(model)
                    & source["prompt_mode"].eq(mode)
                    & source["N"].eq(10)
                ].sort_values("L").copy()
                prediction, candidate, score = selected_prediction(
                    block,
                    selections,
                    coefficients,
                    dataset="combined_1k_100k_fixed_N10",
                    model=model,
                    mode=mode,
                    outcome=outcome,
                )
                color, linestyle, marker = mode_style[mode]
                ax.plot(block["L"], prediction, color=color, linestyle=linestyle, linewidth=2.0)
                ax.scatter(block["L"], block[observed_column], s=28, marker=marker, facecolor=AURORA["paper"], edgecolor=color, linewidth=1.0, zorder=3)
                ax.text(
                    0.015,
                    0.96 - 0.105 * MODES.index(mode),
                    f"{MODE_LABEL[mode]}: {PLOT_FORMULA[candidate]} · CV {'D²' if outcome == HEADLINE_ACCURACY else 'R²'}={score:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7.2,
                    color=color,
                )
            ax.set_xscale("log")
            levels = sorted(block["L"].unique().astype(int))
            ax.set_xticks(levels)
            ax.set_xticklabels([f"{value // 1000}k" for value in levels], rotation=40, ha="right", fontsize=6.8)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.grid(True)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(model, loc="left", fontsize=11.5, fontweight="bold")
                ax.set_ylim(-0.02, 1.02)
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            if row == 2:
                ax.axhline(0, color=AURORA["gray"], linewidth=0.8)
                ax.set_xlabel("Passage length L (tokens; log positions)")
            if column == 0:
                ax.set_ylabel(OUTCOME_LABEL[outcome])
    fig.suptitle("Fixed N=10: length-only exploratory regressions", x=0.075, y=0.995, ha="left", fontsize=14.8, fontweight="bold")
    fig.text(0.075, 0.015, "Open markers: observed cell estimands · curves: selected L or ln L law", fontsize=8.5, color="#667085")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.09, hspace=0.28, wspace=0.12)
    fig.savefig(path, dpi=185, bbox_inches="tight", facecolor=AURORA["paper"])
    plt.close(fig)


def equation_from_coefficients(
    coefficients: pd.DataFrame,
    *,
    outcome: str,
) -> str:
    ordered = coefficients.reset_index(drop=True)
    expression = ""
    for index, row in ordered.iterrows():
        estimate = float(row["estimate"])
        term = str(row["term"])
        symbol = TERM_LATEX.get(term, term.replace("_", r"\_"))
        magnitude = f"{abs(estimate):.5g}"
        if index == 0:
            expression = ("-" if estimate < 0 else "") + magnitude
            if term != "intercept":
                expression += rf"\,{symbol}"
        else:
            expression += f" {'-' if estimate < 0 else '+'} {magnitude}"
            expression += rf"\,{symbol}"
    left = (
        r"\operatorname{logit}\widehat p"
        if outcome == HEADLINE_ACCURACY
        else (r"\widehat{\operatorname{tMAE}}_{10}" if outcome == MAE_FAMILY else r"\widehat b_{10}")
    )
    return rf"\({left}={expression}\)"


def selected_equations(
    selections: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    dataset: str,
) -> str:
    rows = []
    subset = selections.loc[selections["dataset"].eq(dataset)].sort_values(
        ["model_label", "prompt_mode", "outcome_family"]
    )
    for item in subset.itertuples(index=False):
        coef = coefficients.loc[
            coefficients["dataset"].eq(dataset)
            & coefficients["model_label"].eq(item.model_label)
            & coefficients["prompt_mode"].eq(item.prompt_mode)
            & coefficients["outcome_family"].eq(item.outcome_family)
            & coefficients["candidate"].eq(item.selected_candidate)
        ]
        rows.append(
            {
                "Model": html.escape(str(item.model_label)),
                "Mode": MODE_LABEL[str(item.prompt_mode)],
                "Outcome": OUTCOME_LABEL[str(item.outcome_family)],
                "Equation": equation_from_coefficients(coef, outcome=str(item.outcome_family)),
            }
        )
    return html_table(pd.DataFrame(rows))


def frozen_summary_table(tables: Path) -> pd.DataFrame:
    accuracy = pd.read_csv(tables / "frozen_v3_2_accuracy_holdout_metrics.csv")
    mae = pd.read_csv(tables / "frozen_v3_2_mae_holdout_metrics.csv")
    accuracy = accuracy.loc[accuracy["L"].astype(str).eq("all")]
    mae = mae.loc[mae["L"].astype(str).eq("all")]
    rows = []
    for model in MODELS:
        for mode in MODES:
            a = accuracy.loc[
                accuracy["model_label"].eq(model)
                & accuracy["prompt_mode"].eq(mode)
            ].iloc[0]
            e = mae.loc[
                mae["model_label"].eq(model) & mae["prompt_mode"].eq(mode)
            ].iloc[0]
            rows.append(
                {
                    "Model": model,
                    "Mode": MODE_LABEL[mode],
                    "Observed accuracy": format_number(a["observed_accuracy"]),
                    "Frozen mean p": format_number(a["mean_predicted_probability"]),
                    "Log loss": format_number(a["log_loss"]),
                    "Brier": format_number(a["brier_score"]),
                    "Observed tMAE₁₀": format_number(e["observed_mean"]),
                    "Frozen mean tMAE₁₀": format_number(e["predicted_mean"]),
                    "Cell prediction MAE": format_number(e["mean_absolute_prediction_error"]),
                }
            )
    return pd.DataFrame(rows)


def selected_law_table(selections: pd.DataFrame) -> pd.DataFrame:
    subset = selections.loc[selections["dataset"].eq("combined_1k_100k")].copy()
    rows = []
    for item in subset.sort_values(["outcome_family", "model_label", "prompt_mode"]).itertuples(index=False):
        rows.append(
            {
                "Model": item.model_label,
                "Mode": MODE_LABEL[item.prompt_mode],
                "Outcome": OUTCOME_LABEL[item.outcome_family],
                "Selected basis": rf"\({FORMULA[item.selected_candidate]}\)",
                "CV score": format_number(item.selected_cv_score),
                "Best registry score": format_number(item.best_cv_score),
                "CV loss": format_number(item.selected_cv_loss),
            }
        )
    return pd.DataFrame(rows)


def fixed_n10_table(selections: pd.DataFrame, coefficients: pd.DataFrame) -> pd.DataFrame:
    subset = selections.loc[
        selections["dataset"].eq("combined_1k_100k_fixed_N10")
    ]
    rows = []
    for item in subset.sort_values(["outcome_family", "model_label", "prompt_mode"]).itertuples(index=False):
        coef = coefficients.loc[
            coefficients["dataset"].eq(item.dataset)
            & coefficients["model_label"].eq(item.model_label)
            & coefficients["prompt_mode"].eq(item.prompt_mode)
            & coefficients["outcome_family"].eq(item.outcome_family)
            & coefficients["candidate"].eq(item.selected_candidate)
        ]
        slope = coef.loc[coef["term"].ne("intercept"), "estimate"]
        rows.append(
            {
                "Model": item.model_label,
                "Mode": MODE_LABEL[item.prompt_mode],
                "Outcome": OUTCOME_LABEL[item.outcome_family],
                "Length form": rf"\({FORMULA[item.selected_candidate]}\)",
                "Slope": format_number(slope.iloc[0] if len(slope) else math.nan, 5),
                "CV score": format_number(item.selected_cv_score),
                "Interpretation": (
                    "D² from log loss"
                    if item.outcome_family == HEADLINE_ACCURACY
                    else "held-condition R²"
                ),
            }
        )
    return pd.DataFrame(rows)


def n_fixed_comparison_table(
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for mode in MODES:
            for outcome in (HEADLINE_ACCURACY, MAE_FAMILY, BIAS_FAMILY):
                block = metrics.loc[
                    metrics["model_label"].eq(model)
                    & metrics["prompt_mode"].eq(mode)
                    & metrics["outcome_family"].eq(outcome)
                ]
                cv = block.loc[
                    block["evaluation_scheme"].eq("combined_5fold_held_condition_cv")
                    & block["length_term"].isin(["L_k", "logL"])
                ].copy()
                hold = block.loc[
                    block["evaluation_scheme"].eq("fit_v3_1_score_v3_3_holdout")
                ].copy()
                if outcome == HEADLINE_ACCURACY:
                    cv_gain = "cv_d2_vs_N_fixed"
                    hold_gain = "holdout_d2_vs_N_fixed"
                else:
                    cv_gain = "cv_relative_sse_reduction_vs_N_fixed"
                    hold_gain = "holdout_relative_sse_reduction_vs_N_fixed"
                cv_best = cv.sort_values(cv_gain, ascending=False).iloc[0]
                hold[hold_gain] = hold[hold_gain].fillna(0.0)
                hold_best = hold.sort_values(hold_gain, ascending=False).iloc[0]
                coef = coefficients.loc[
                    coefficients["model_label"].eq(model)
                    & coefficients["prompt_mode"].eq(mode)
                    & coefficients["outcome_family"].eq(outcome)
                    & coefficients["evaluation_scheme"].eq(
                        "combined_5fold_held_condition_cv"
                    )
                    & coefficients["length_term"].eq(cv_best["length_term"])
                    & coefficients["term"].str.startswith("beta_")
                ]
                slope = coef.iloc[0] if not coef.empty else None
                slope_text = "—"
                if slope is not None:
                    slope_text = format_number(slope["estimate"], 4)
                    if pd.notna(slope["ci95_low"]):
                        slope_text += f" [{format_number(slope['ci95_low'],4)}, {format_number(slope['ci95_high'],4)}]"
                    else:
                        slope_text += " [HC3 CI unavailable]"
                hold_spec = (
                    r"\(\alpha_N\)"
                    if hold_best["length_term"] == "none"
                    else (
                        r"\(\alpha_N+\beta L_k\)"
                        if hold_best["length_term"] == "L_k"
                        else r"\(\alpha_N+\beta\ln L_k\)"
                    )
                )
                rows.append(
                    {
                        "Model": model,
                        "Mode": MODE_LABEL[mode],
                        "Outcome": OUTCOME_LABEL[outcome],
                        "Combined best length": (
                            r"\(L_k\)" if cv_best["length_term"] == "L_k" else r"\(\ln L_k\)"
                        ),
                        "Shared slope β [95% HC3 CI]": slope_text,
                        "Combined gain vs α_N": format_number(cv_best[cv_gain]),
                        "Best V3.1→V3.3 form": hold_spec,
                        "Holdout gain vs α_N": format_number(hold_best[hold_gain]),
                    }
                )
    return pd.DataFrame(rows)


def inject(report: str, section: str) -> str:
    style = """<!-- V3_3_STYLE_START --><style>
#piecewise-n3{order:8} #v3-3-long-context{order:9} #repro{order:10} .footer{order:11}
#v3-3-long-context .v33-scope{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:22px 0}
#v3-3-long-context .v33-scope>div{padding:15px 16px;border:1px solid var(--line);background:#fff}
#v3-3-long-context .v33-scope strong{display:block;color:var(--indigo);font:700 22px ui-monospace,SFMono-Regular,Consolas,monospace}
#v3-3-long-context .v33-scope span{display:block;margin-top:4px;color:#596579;font-size:12px;line-height:1.45}
#v3-3-long-context .v33-wide .data-table{table-layout:fixed;font-size:10.5px}
#v3-3-long-context .v33-wide .data-table th,#v3-3-long-context .v33-wide .data-table td{padding:7px 6px;white-space:normal;overflow-wrap:anywhere}
@media(max-width:800px){#v3-3-long-context .v33-scope{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><!-- V3_3_STYLE_END -->"""
    if STYLE_START in report:
        report = re.sub(
            re.escape(STYLE_START) + r".*?" + re.escape(STYLE_END),
            lambda _: style,
            report,
            flags=re.DOTALL,
        )
    else:
        report = report.replace("</head>", style + "</head>", 1)
    if START in report:
        report = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            lambda _: section,
            report,
            flags=re.DOTALL,
        )
    else:
        anchor = '<section id="repro">'
        if anchor not in report:
            raise ValueError("Reproducibility section anchor not found")
        report = report.replace(anchor, section + "\n\n" + anchor, 1)
    nav_link = '<a href="#v3-3-long-context">Appendix F · V3.3 long context</a>'
    if nav_link not in report:
        report = report.replace('<a href="#repro">Reproducibility</a>', nav_link + '<a href="#repro">Reproducibility</a>', 1)
    report = re.sub(
        r"<title>.*?</title>",
        "<title>NiaH Empirical-law Report · V3.2 + V3.3 long-context extension</title>",
        report,
        count=1,
        flags=re.DOTALL,
    )
    return report


def main() -> int:
    args = parse_args()
    analysis = args.analysis_dir.resolve()
    tables = analysis / "tables"
    state = json.loads((analysis / "analysis_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((analysis / "analysis_manifest.json").read_text(encoding="utf-8"))
    if state.get("stage") != "complete":
        raise ValueError(f"V3.3 analysis is not complete: {state}")
    old_path = Path(manifest["inputs"]["old"]["path"])
    holdout_path = Path(manifest["inputs"]["holdout"]["path"])
    old = normalize_requests(old_path, "v3_1_1k_20k")
    holdout = normalize_requests(holdout_path, "v3_3_25k_100k")
    combined = pd.concat([old, holdout], ignore_index=True, sort=False)
    accuracy_cells, error_cells = response_cells(combined)

    selections = pd.read_csv(tables / "exploratory_selected_laws.csv")
    coefficients = pd.read_csv(tables / "exploratory_candidate_coefficients.csv")
    frozen_coefficients = pd.read_csv(args.frozen_accuracy_coefficients)

    assets = args.assets_dir.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    figure_paths = {
        "frozen_accuracy": assets / "v3_3_frozen_accuracy_holdout.png",
        "accuracy_l": assets / "v3_3_accuracy_l_horizontal.png",
        "accuracy_n": assets / "v3_3_accuracy_n_horizontal.png",
        "mae_l": assets / "v3_3_trimmed_mae_l_horizontal.png",
        "mae_n": assets / "v3_3_trimmed_mae_n_horizontal.png",
        "fixed_n10": assets / "v3_3_fixed_n10_length_regressions.png",
    }
    plot_frozen_accuracy(holdout, frozen_coefficients, figure_paths["frozen_accuracy"])
    plot_joint_law(
        accuracy_cells,
        selections,
        coefficients,
        outcome=HEADLINE_ACCURACY,
        observed_column="observed_accuracy",
        horizontal="L",
        path=figure_paths["accuracy_l"],
    )
    plot_joint_law(
        accuracy_cells,
        selections,
        coefficients,
        outcome=HEADLINE_ACCURACY,
        observed_column="observed_accuracy",
        horizontal="N",
        path=figure_paths["accuracy_n"],
    )
    eligible_error = error_cells.loc[error_cells["mae_law_eligible"].astype(bool)]
    plot_joint_law(
        eligible_error,
        selections,
        coefficients,
        outcome=MAE_FAMILY,
        observed_column=MAE_FAMILY,
        horizontal="L",
        path=figure_paths["mae_l"],
    )
    plot_joint_law(
        eligible_error,
        selections,
        coefficients,
        outcome=MAE_FAMILY,
        observed_column=MAE_FAMILY,
        horizontal="N",
        path=figure_paths["mae_n"],
    )
    plot_fixed_n10(
        accuracy_cells,
        eligible_error,
        selections,
        coefficients,
        figure_paths["fixed_n10"],
    )

    frozen_table = html_table(frozen_summary_table(tables))
    law_table = html_table(selected_law_table(selections))
    fixed_table = html_table(fixed_n10_table(selections, coefficients))
    combined_equations = selected_equations(
        selections, coefficients, dataset="combined_1k_100k"
    )
    fixed_equations = selected_equations(
        selections, coefficients, dataset="combined_1k_100k_fixed_N10"
    )

    section = r"""<!-- V3_3_LONG_CONTEXT_START -->
<section id="v3-3-long-context" class="appendix"><div class="section-head"><div class="section-no">Appendix F · V3.3</div><div><h2>Long-context extension: two 32B models over 1k–100k tokens</h2><p class="lede">本节把 Gemma4-31B 与 Qwen3-32B 的新增长上下文数据补入报告。预注册问题是：冻结的 V3.2 经验律能否从 1k–20k 外推到 25k–100k；其余重新选律与固定 N=10 回归均为事后探索。</p></div></div>
<div class="v33-scope"><div><strong>15,120</strong><span>新增且未用于 V3.2 拟合的 requests</span></div><div><strong>28,560</strong><span>两模型在 1k–100k 的合并 requests</span></div><div><strong>14 × 17</strong><span>目标数 N × passage length L 条件网格</span></div><div><strong>0</strong><span>bootstrap；沿用五折 held-condition CV</span></div></div>

<h3>F.1 设计、估计目标与证据边界</h3>
<p>两个固定模型 revision 均包含 Non-thinking 与 Native-thinking；每个 \((N,L)\) cell 有 30 个固定 seeds。新增长度为 \(L\in\{25,30,40,50,60,70,80,90,100\}\,\mathrm{k}\)，目标数仍为 \(N\in\{1,2,3,4,5,6,7,8,9,10,12,15,18,20\}\)。Accuracy 是 request-level exact-match probability；MAE 与 Bias 仍条件于可解析整数，并分别对绝对误差与有符号误差两端各删除 10% 后取均值。</p>
<p>本节包含冻结公式外推检验与两模型各自的探索性重新拟合（包括固定 N=10 的长度回归）。重新拟合不能回写为预注册成功；N 固定效应比较归入 V3.2 补充。</p>
<div class="conclusion"><strong>F.1 当前结论：</strong>新增数据构成真正的长度外推检验；在观察新数据后重新选出的公式只描述 1k–100k 范围内的两模型行为。</div>

<h3>F.2 冻结 V3.2 law 的 25k–100k 外推</h3>
<p>Accuracy 用 log loss 与 Brier score 评分。log loss 为 \(-n^{-1}\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)]\)，Brier score 为 \(n^{-1}\sum_i(y_i-p_i)^2\)；两者越小越好。MAE 表中的 “Cell prediction MAE” 是冻结 law 对各 \((N,L)\) cell 的 \(\operatorname{tMAE}_{10}\) 预测绝对误差的平均值，不是模型的计数误差本身。</p>
<div class="table-wrap">@@FROZEN_TABLE@@</div>
<figure><img src="@@FROZEN_FIG@@" alt="Frozen V3.2 accuracy laws evaluated on the 25k to 100k holdout"><figcaption><strong>FIGURE F1</strong><span><b>冻结 V3.2 Accuracy law 的长上下文检验。</b> 横轴为未参与 V3.2 拟合的 9 个 passage lengths；纵轴为在 14 个 N 与每个 N 的 30 个 seeds 上等权汇总的 exact accuracy。实线圆点是观测值，虚线三角是冻结预测；橙色为 Non-thinking，紫色为 Native-thinking。每个观测点含 420 requests。</span></figcaption></figure>
<p>Gemma4-31B Non-thinking 的平均预测 (0.207) 接近观测 (0.262)，但另外三组存在明显校准差异：Gemma Native-thinking 被低估（(0.283) 对 (0.776)），Qwen Non-thinking 被低估（(0.116) 对 (0.270)），Qwen Native-thinking 被高估（(0.723) 对 (0.518)）。MAE 也显示均值接近并不保证 cell-level 预测准确，例如 Qwen Non-thinking 的观测/预测均值接近，但 cell prediction MAE 为 1.352 count units。</p>
<div class="conclusion"><strong>F.2 当前结论：</strong>冻结 V3.2 公式不是普适的 100k 外推律。Gemma Non-thinking 的外推相对接近；其余模式至少存在系统校准偏差，因此需要在新长度区间重新描述，而不能声称 V3.2 law 已获整体确认。</div>

<h3>F.3 两模型分别在 N–L 网格上的探索性回归</h3>
<p>合并 1k–100k 后，每个模型、mode 与 outcome 分别扫描 18 个冻结候选 basis。Accuracy 仍在 logit scale 拟合；MAE/Bias 在原始 count units 使用 identity OLS。五折规则为 \((\operatorname{index}(N)+\operatorname{index}(L))\bmod 5\)，同一 cell 的 30 个 seeds 始终在同一折。候选选择沿用“距最佳 CV score 不超过 0.02 时优先较少 predictors、较低 loss 与较早 registry order”。</p>
<div class="table-wrap">@@LAW_TABLE@@</div>
<figure><img src="@@ACC_L_FIG@@" alt="Two-model V3.3 accuracy laws with passage length on the horizontal axis"><figcaption><strong>FIGURE F2</strong><span><b>Accuracy law，L-horizontal。</b> 每个面板对应一个固定模型与 mode；横轴覆盖全部 17 个 \(L\)，纵轴是 exact accuracy。黄→紫编码全部 14 个 \(N\)（黄色为较小 \(N\)，紫色为较大 \(N\)）。空心点为每个 30-seed cell 的观测准确率，实线为该面板在合并区间选择的 logit-law 预测。面板内给出 basis 与 held-condition CV \(D^2\)。</span></figcaption></figure>
<figure><img src="@@ACC_N_FIG@@" alt="Two-model V3.3 accuracy laws with target count on the horizontal axis"><figcaption><strong>FIGURE F3</strong><span><b>Accuracy law，N-horizontal。</b> F2 的同一拟合改以 N 为横轴；黄→紫编码全部 17 个 L（黄色为短上下文，紫色为长上下文）。该转置视图用于检查 N 主效应与 N–L interaction，未进行第二次拟合。</span></figcaption></figure>
<figure><img src="@@MAE_L_FIG@@" alt="Two-model V3.3 trimmed MAE laws with passage length on the horizontal axis"><figcaption><strong>FIGURE F4</strong><span><b>10% trimmed conditional MAE law，L-horizontal。</b> 纵轴单位为 counts；每个 cell 对可解析响应的绝对误差排序后，删除最低与最高各 10%，再平均中间 80%。空心点为 cell estimand，实线为 identity-OLS 预测；颜色覆盖全部 N。</span></figcaption></figure>
<figure><img src="@@MAE_N_FIG@@" alt="Two-model V3.3 trimmed MAE laws with target count on the horizontal axis"><figcaption><strong>FIGURE F5</strong><span><b>10% trimmed conditional MAE law，N-horizontal。</b> F4 的同一拟合改以 \(N\) 为横轴；颜色覆盖全部 \(L\)。identity link 不强制非负，因此局部负预测若出现应解释为线性近似失配，不能解释为负误差幅度。</span></figcaption></figure>
<p>两模型的 Non-thinking Accuracy 需要不同 basis：Gemma 选择 \(N^{-1}+L_k\)，Qwen 选择含 \((\ln N)(\ln L_k)\) 的交互式；两种 Native-thinking 均选择 \(\ln N+\ln L_k\)。这支持“mode 影响误差几何”的描述，但不能把相同 basis 当作相同机制，因为截距和斜率仍按模型分别估计。</p>
<details class="equations"><summary>展开：1k–100k 的 12 个模型特异方程</summary><div class="table-wrap">@@COMBINED_EQUATIONS@@</div></details>
<div class="conclusion"><strong>F.3 当前结论：</strong>扩大到 100k 后，Native-thinking Accuracy 在两个 32B 模型上都可由对数 \(N\) 与对数 \(L\) 的加性 log-odds 近似；Non-thinking 的 \(N\) 形状依模型而异。MAE 的高 CV \(R^2\) 主要出现在 Gemma Non-thinking，其他组的可预测程度较低或更依赖交互项。</div>

<h3>F.4 固定 \(N=10\)：只比较 \(L\) 与 \(\ln L\)</h3>
<p>固定 \(N=10\) 后，\(N\) 不再是 predictor。三个候选为截距、\(L_k=L/1000\) 与 \(\ln L_k\)：</p>
<div class="math-block">\[g\{\mu(10,L)\}=\alpha+\beta L_k\quad\text{or}\quad g\{\mu(10,L)\}=\alpha+\beta\ln L_k,\]</div>
<p>其中 Accuracy 使用 \(g=\operatorname{logit}\)，MAE/Bias 使用 identity。线性 \(L_k\) 表示每增加 1k tokens 有相同的 link-scale 变化；\(\ln L_k\) 表示 passage length 按相同比例增长时有相同变化。CV score 对 Accuracy 为 \(D^2\)，对 MAE/Bias 为 \(R^2\)。</p>
<div class="table-wrap">@@FIXED_TABLE@@</div>
<figure><img src="@@FIXED_FIG@@" alt="Fixed N equals 10 length-only regressions for accuracy, trimmed MAE, and trimmed bias"><figcaption><strong>FIGURE F6</strong><span><b>固定 \(N=10\) 的长度回归。</b> 两列为两个模型，三行为 Accuracy、10% trimmed conditional MAE 与 10% trimmed signed bias；横轴为全部 17 个 \(L\)。橙色实线/圆点为 Non-thinking，紫色虚线/三角为 Native-thinking。空心点是观测 cell estimand，曲线是 \(L_k\) 或 \(\ln L_k\) 中按同一 CV 规则选出的模型。</span></figcaption></figure>
<p>两个 Native-thinking Accuracy 都选择 \(\ln L_k\)，而 Gemma Non-thinking 选择负的线性 \(L_k\) slope。Qwen Non-thinking Accuracy 也选择 \(L_k\)，但 CV \(D^2=0.064\)，其正 slope 只能视为弱且模型特异的描述。Qwen Non-thinking MAE 的 CV \(R^2=0.010\)，说明固定 \(N=10\) 后没有稳定的单调长度信号。部分 MAE identity fits 在短长度给出负预测，进一步表明单一全区间直线只是一阶近似。</p>
<details class="equations"><summary>展开：固定 N=10 的 12 个方程</summary><div class="table-wrap">@@FIXED_EQUATIONS@@</div></details>
<div class="conclusion"><strong>F.4 当前结论：</strong>固定 N 后，Native-thinking Accuracy 的长度退化在两个模型上都更接近乘法尺度；Non-thinking 不共享这一简单规律。MAE 对长度的响应更不稳定，尤其不能从 Qwen Non-thinking 推断通用 slope。</div>

<h3>F.5 固定效应分析的版本范围</h3>
<p>原有 12 个比较槽的固定效应比较使用 V3.2 的 1k–20k 数据，见<a href="#v3-2-n-fixed-effects">V3.2 固定效应补充</a>。本次另在两个模型的 1k–100k 范围，使用全部 N 比较共同的线性与对数长度系数，见报告前部的<a href="#all-n-linear-length">全部 N 长度回归</a>。这八组结果已重新拟合并与原有数值核验；此前其他 V3.3 固定效应探索记录继续保留。</p>

<h3>F.6 Reproducibility record</h3>
<p>本节使用两份紧凑 request-level 输入：V3.1 两模型子集 13,440 rows 与 V3.3 holdout 15,120 rows；合并后 request IDs 唯一。分析未运行 bootstrap，也未重新运行模型推理。完整 CSV 包括候选指标与系数、固定 N=10 结果；先前生成的 N-fixed CSV 仅作为历史探索记录归档。</p>
<div class="methods">analysis schema: @@SCHEMA@@
old input sha256: @@OLD_SHA@@
holdout input sha256: @@HOLDOUT_SHA@@
candidate metric rows: @@CANDIDATE_ROWS@@
bootstrap repetitions: 0

Rebuild:
.venv\Scripts\python.exe scripts\analyze_realistic_niah_v3_3_regression_scan.py [...]
.venv\Scripts\python.exe scripts\build_niah_empirical_law_v3_3_addendum.py</div>
<div class="conclusion"><strong>Appendix F 总结：</strong>冻结的 1k–20k 公式不能直接覆盖两模型所有 mode 到 100k。长上下文效应的函数形状随模型和 mode 改变；各自重新拟合的结果属于探索性描述。</div></section>
<!-- V3_3_LONG_CONTEXT_END -->"""
    replacements = {
        "@@FROZEN_TABLE@@": frozen_table,
        "@@LAW_TABLE@@": law_table,
        "@@FIXED_TABLE@@": fixed_table,
        "@@COMBINED_EQUATIONS@@": combined_equations,
        "@@FIXED_EQUATIONS@@": fixed_equations,
        "@@FROZEN_FIG@@": image_uri(figure_paths["frozen_accuracy"]),
        "@@ACC_L_FIG@@": image_uri(figure_paths["accuracy_l"]),
        "@@ACC_N_FIG@@": image_uri(figure_paths["accuracy_n"]),
        "@@MAE_L_FIG@@": image_uri(figure_paths["mae_l"]),
        "@@MAE_N_FIG@@": image_uri(figure_paths["mae_n"]),
        "@@FIXED_FIG@@": image_uri(figure_paths["fixed_n10"]),
        "@@SCHEMA@@": html.escape(str(manifest["schema_version"])),
        "@@OLD_SHA@@": html.escape(str(manifest["inputs"]["old"]["sha256"])),
        "@@HOLDOUT_SHA@@": html.escape(str(manifest["inputs"]["holdout"]["sha256"])),
        "@@CANDIDATE_ROWS@@": str(manifest["outputs"]["candidate_metric_rows"]),
    }
    for key, value in replacements.items():
        section = section.replace(key, value)
    if "@@" in section:
        raise ValueError("Unresolved V3.3 report placeholder")

    report_path = args.report.resolve()
    report = report_path.read_text(encoding="utf-8")
    report = inject(report, section)
    if "\ufffd" in report:
        raise ValueError("Report contains a Unicode replacement character")
    report_path.write_text(report, encoding="utf-8")
    build_manifest = {
        "schema_version": "niah_empirical_law_v3_3_addendum_build_v1",
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "analysis_manifest_sha256": sha256(analysis / "analysis_manifest.json"),
        "analysis_state": state,
        "figures": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in figure_paths.items()
        },
    }
    (assets / "v3_3_addendum_build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "report": str(report_path),
                "bytes": report_path.stat().st_size,
                "figures": len(figure_paths),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
