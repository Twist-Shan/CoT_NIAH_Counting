#!/usr/bin/env python3
"""Build the audited V3.2 behavioral empirical-law article.

The report is intentionally self-contained: figures are embedded as base64 PNGs,
while the source PNGs and a build manifest are retained beside the report.  Every
number in the article is derived from the frozen V3.2 outputs, the audited V3.1
behavior tables, or the separately labelled post-hoc untrimmed-bias sensitivity
analysis.  The latter never replaces the preregistered trimmed-bias estimand.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Fixed Aurora palette supplied for the paper figures. Neutral tints may be used
# for grids and rules, but every data-bearing mark must come from this registry.
AURORA = {
    "indigo": "#23165C",
    "violet": "#6750E8",
    "cyan": "#00C2FF",
    "yellow": "#F6E36A",
    "teal": "#00D4B4",
    "green": "#39E58C",
    "magenta": "#C04DFF",
    "pink": "#FF5FA2",
    "black": "#161923",
    "white": "#F8FBFF",
    "gray": "#8190A5",
    "brown": "#765347",
}

# A four-stop approximation to the purple-to-yellow Plasma sequence used in
# classic neural scaling-law figures.  We reverse it for passage length so the
# easiest/shortest condition is yellow and the hardest/longest is purple.
SCALING_LAW = {
    "purple": "#2C115F",
    "magenta": "#9C179E",
    "orange": "#ED7953",
    "yellow": "#F0F921",
}

MODE_ORDER = ["direct", "native_thinking", "enumeration_index", "enumeration_bullet"]
MODE_ZH = {
    "direct": "Non-thinking（直接作答）",
    "native_thinking": "Native-thinking（原生思考）",
    "enumeration_index": "索引枚举",
    "enumeration_bullet": "项目符号枚举",
}
MODE_SHORT = {
    "direct": "Non-thinking",
    "native_thinking": "Native-thinking",
    "enumeration_index": "Index",
    "enumeration_bullet": "Bullet",
}
MODE_COLOR = {
    "direct": SCALING_LAW["yellow"],
    "native_thinking": SCALING_LAW["purple"],
    "enumeration_index": SCALING_LAW["magenta"],
    "enumeration_bullet": SCALING_LAW["orange"],
}
MAE_FAMILY = "trimmed_conditional_mae_10"
def length_colors(levels: list[int]) -> list[str]:
    """Sample the scaling-law Plasma ramp for every registered L level."""
    colors = matplotlib.colormaps["plasma_r"](
        np.linspace(0.0, 1.0, len(levels))
    )
    return [mcolors.to_hex(color).upper() for color in colors]


def target_count_colors(levels: list[int]) -> list[str]:
    """Sample the same yellow-to-purple scaling-law ramp for N curves."""
    return length_colors(levels)


def apply_all_n_ticks(ax: plt.Axes, n_levels: list[int], *, fontsize: float) -> None:
    """Show every registered N value on the log2 axis."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_levels)
    ax.set_xticklabels(
        [str(value) for value in n_levels],
        rotation=52,
        ha="right",
        rotation_mode="anchor",
        fontsize=fontsize,
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def apply_all_l_ticks(ax: plt.Axes, l_levels: list[int], *, fontsize: float) -> None:
    """Show all registered passage lengths on a log-positioned L axis."""
    ax.set_xscale("log")
    ax.set_xticks(l_levels)
    ax.set_xticklabels(
        [f"{value // 1000}k" for value in l_levels],
        rotation=36,
        ha="right",
        rotation_mode="anchor",
        fontsize=fontsize,
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

SLOT_ORDER = [
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
    "GLM-4/Z1-9B",
    "Ministral-3-8B pair",
]
MATCHED_PAIR_SLOTS = {"GLM-4/Z1-9B", "Ministral-3-8B pair"}

FORMULA = {
    "intercept": "1",
    "N": "N",
    "L_k": "Lₖ",
    "logN": "ln N",
    "logL": "ln Lₖ",
    "N__L_k": "N + Lₖ",
    "logN__logL": "ln N + ln Lₖ",
    "N__logL": "N + ln Lₖ",
    "logN__L_k": "ln N + Lₖ",
    "N__L_k__N_x_L_k": "N + Lₖ + N×Lₖ",
    "logN__logL__logN_x_logL": "ln N + ln Lₖ + ln N×ln Lₖ",
    "N__logL__N_x_logL": "N + ln Lₖ + N×ln Lₖ",
    "logN__L_k__logN_x_L_k": "ln N + Lₖ + ln N×Lₖ",
    "invN": "1/N",
    "invN__L_k": "1/N + Lₖ",
    "invN__logL": "1/N + ln Lₖ",
    "invN__L_k__invN_x_L_k": "1/N + Lₖ + (1/N)×Lₖ",
    "invN__logL__invN_x_logL": "1/N + ln Lₖ + (1/N)×ln Lₖ",
}

PLOT_FORMULA = {
    key: value.replace("Lₖ", "L/1k") for key, value in FORMULA.items()
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    download = root / "outputs" / "anvil_realistic_niah_v3_1_20260819_formal"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-root",
        type=Path,
        default=download,
        help="Local root containing the audited run and analysis directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "NiaH_Empirical-law_report.html",
    )
    parser.add_argument(
        "--v3-1-config",
        type=Path,
        default=root / "configs" / "realistic_niah_v3_1.json",
    )
    parser.add_argument(
        "--v3-2-config",
        type=Path,
        default=root / "configs" / "realistic_niah_v3_2_empirical_law_analysis.json",
    )
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict:
    return json.loads(require(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float, digits: int = 1) -> str:
    return "—" if pd.isna(value) else f"{100 * float(value):.{digits}f}%"


def num(value: float, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def coefficient_effect_summary(
    frame: pd.DataFrame,
    *,
    outcome_family: str,
    prompt_mode: str,
    term: str,
) -> dict[str, float]:
    """Summarize one selected-law term across the comparison slots."""
    subset = frame[
        frame["outcome_family"].eq(outcome_family)
        & frame["prompt_mode"].eq(prompt_mode)
        & frame["term"].eq(term)
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No coefficient rows for {outcome_family}/{prompt_mode}/{term}"
        )
    effects = pd.to_numeric(subset["standardized_effect"], errors="coerce")
    q_values = pd.to_numeric(subset["hc3_q"], errors="coerce")
    return {
        "median_abs": float(effects.abs().median()),
        "negative_fraction": float((effects < 0).mean()),
        "positive_fraction": float((effects > 0).mean()),
        "q05_fraction": float((q_values < 0.05).mean()),
    }


def b64_png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def table_html(frame: pd.DataFrame, classes: str = "data-table") -> str:
    return frame.to_html(index=False, border=0, classes=classes, escape=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": AURORA["white"],
            "axes.facecolor": AURORA["white"],
            "axes.edgecolor": "#D5DCE6",
            "axes.labelcolor": AURORA["black"],
            "xtick.color": "#536176",
            "ytick.color": "#536176",
            "text.color": AURORA["black"],
            "grid.color": "#DCE3ED",
            "grid.linewidth": 0.65,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=AURORA["white"])
    plt.close(fig)


def weighted_mode_summary(summary: pd.DataFrame, slots: set[str] | None = None) -> pd.DataFrame:
    data = summary.copy()
    if slots is not None:
        data = data[data["comparison_slot"].isin(slots)]
    rows = []
    for mode in MODE_ORDER:
        block = data[data["prompt_mode"].eq(mode)]
        total = float(block["n_total"].sum())
        rows.append(
            {
                "prompt_mode": mode,
                "accuracy": float(block["n_correct_parsed"].sum() / total),
                "parse_rate": float(block["n_parseable"].sum() / total),
                "strict_accuracy": float(block["strict_successes"].sum() / total),
                "requests": int(total),
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_dumbbell(summary: pd.DataFrame, path: Path) -> None:
    pivot = summary.pivot(index="comparison_slot", columns="prompt_mode", values="parsed_exact_accuracy")
    pivot = pivot.reindex(SLOT_ORDER)
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(13.8, 4.15))
    for idx, (slot, row) in enumerate(pivot.iterrows()):
        xx = x[idx]
        ax.plot([xx, xx], [row["direct"], row["native_thinking"]], color=AURORA["gray"], alpha=0.45, lw=2.2, zorder=1)
        ax.scatter(xx, row["direct"], s=58, color=MODE_COLOR["direct"], edgecolor=AURORA["indigo"], lw=0.65, marker="o", zorder=3)
        ax.scatter(xx, row["native_thinking"], s=64, color=MODE_COLOR["native_thinking"], edgecolor=AURORA["indigo"], lw=0.65, marker="^", zorder=3)
        delta = row["native_thinking"] - row["direct"]
        ax.text(xx, 1.045, f"{100 * delta:+.0f}", ha="center", va="bottom", fontsize=7.4, color=AURORA["brown"])
    labels = [f"{slot}{'†' if slot in MATCHED_PAIR_SLOTS else ''}" for slot in pivot.index]
    ax.set_xticks(x, labels, rotation=24, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Parsed exact accuracy")
    ax.set_xlabel("Comparison slot")
    ax.set_title("Parsed exact accuracy by comparison slot", loc="left", pad=18)
    ax.text(0, 1.025, "Vertical lines pair the same slot; labels show Native − Non-thinking (percentage points)", transform=ax.transAxes, fontsize=9.2, color=AURORA["gray"])
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.7)
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=MODE_COLOR["direct"], markeredgecolor=AURORA["indigo"], markersize=8, label="Non-thinking"),
        plt.Line2D([], [], marker="^", color="none", markerfacecolor=MODE_COLOR["native_thinking"], markeredgecolor=AURORA["indigo"], markersize=8, label="Native-thinking"),
        plt.Line2D([], [], color="none", label="† matched checkpoints"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=3)
    fig.tight_layout()
    savefig(fig, path)


def design_values(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = frame["N"].to_numpy(float)
    lk = frame["L"].to_numpy(float) / 1000.0
    logn = np.log(n)
    logl = np.log(lk)
    invn = 1.0 / n
    return {
        "N": n,
        "L_k": lk,
        "logN": logn,
        "logL": logl,
        "N_x_L_k": n * lk,
        "logN_x_logL": logn * logl,
        "N_x_logL": n * logl,
        "logN_x_L_k": logn * lk,
        "invN": invn,
        "invN_x_L_k": invn * lk,
        "invN_x_logL": invn * logl,
    }


def median_fitted_surface(
    coefficients: pd.DataFrame,
    mode: str,
    family: str,
    n_levels: list[int],
    l_levels: list[int],
) -> np.ndarray:
    grid = pd.DataFrame([(n, length) for n in n_levels for length in l_levels], columns=["N", "L"])
    values = design_values(grid)
    predictions = []
    for slot in SLOT_ORDER:
        block = coefficients[
            coefficients["comparison_slot"].eq(slot)
            & coefficients["prompt_mode"].eq(mode)
            & coefficients["outcome_family"].eq(family)
        ]
        if block.empty:
            continue
        pred = np.zeros(len(grid), dtype=float)
        for row in block.itertuples(index=False):
            if row.term == "intercept":
                pred += float(row.estimate)
            else:
                pred += float(row.estimate) * values[row.term]
        if family.startswith("accuracy_"):
            pred = 1.0 / (1.0 + np.exp(-np.clip(pred, -30, 30)))
        predictions.append(pred)
    median = np.median(np.vstack(predictions), axis=0)
    return median.reshape(len(n_levels), len(l_levels))


def predictions_by_slot(
    coefficients: pd.DataFrame,
    mode: str,
    family: str,
    n_levels: list[int],
    l_levels: list[int],
) -> np.ndarray:
    """Return selected-law predictions as slot × N × L.

    The V3.2 law shares an item structure across comparison slots but never a
    pooled coefficient vector.  Keeping the slot dimension here lets the plot
    show model heterogeneity rather than inventing an "average model".
    """
    grid = pd.DataFrame(
        [(n, length) for n in n_levels for length in l_levels],
        columns=["N", "L"],
    )
    values = design_values(grid)
    predictions = []
    for slot in SLOT_ORDER:
        block = coefficients[
            coefficients["comparison_slot"].eq(slot)
            & coefficients["prompt_mode"].eq(mode)
            & coefficients["outcome_family"].eq(family)
        ]
        if block.empty:
            continue
        eta = np.zeros(len(grid), dtype=float)
        for row in block.itertuples(index=False):
            if row.term == "intercept":
                eta += float(row.estimate)
            else:
                eta += float(row.estimate) * values[row.term]
        if family.startswith("accuracy_"):
            eta = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        elif family == MAE_FAMILY:
            # V3.2 reports the requested identity-scale OLS fit.  Do not clip
            # negative fitted values: they are scientifically useful diagnostics
            # of where a local linear approximation violates MAE's support.
            eta = eta
        predictions.append(eta)
    if not predictions:
        raise ValueError(f"No selected coefficients for {mode} / {family}")
    return np.vstack(predictions).reshape(-1, len(n_levels), len(l_levels))


def plot_model_law_panels(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    family: str,
    column: str,
    modes: tuple[str, str],
    x_axis: str,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Render 12 model-resolved panels with every registered N and L.

    One axis is horizontal and every level of the other axis becomes a curve.
    Both prompt modes are overlaid, so no median model is invented.  MAE uses a
    symmetric-log display only to retain negative identity-link predictions;
    the regression itself remains in raw count units.
    """
    if x_axis not in {"N", "L"}:
        raise ValueError(f"Unsupported x_axis={x_axis}")
    n_levels = sorted(int(x) for x in cells["N"].unique())
    l_levels = sorted(int(x) for x in cells["L"].unique())
    curve_levels = l_levels if x_axis == "N" else n_levels
    curve_colors = length_colors(curve_levels)
    x_levels = n_levels if x_axis == "N" else l_levels
    fitted_by_mode = {
        mode: predictions_by_slot(coefficients, mode, family, n_levels, l_levels)
        for mode in modes
    }
    for mode, values in fitted_by_mode.items():
        if values.shape[0] != len(SLOT_ORDER):
            raise ValueError(
                f"Expected {len(SLOT_ORDER)} slots for {mode}/{family}, got {values.shape[0]}"
            )

    fig, axes = plt.subplots(3, 4, figsize=(16.4, 11.1), sharex=True, sharey=True)
    for slot_index, slot in enumerate(SLOT_ORDER):
        row_index, column_index = divmod(slot_index, 4)
        ax = axes[row_index, column_index]
        if family in {"trimmed_signed_bias_10", MAE_FAMILY}:
            ax.axhline(0, color=AURORA["gray"], linewidth=0.7, alpha=0.58, zorder=0)
        for mode_index, mode in enumerate(modes):
            line_style = "-" if mode_index == 0 else "--"
            marker = "o" if mode_index == 0 else "^"
            for curve_index, (curve_level, color) in enumerate(zip(curve_levels, curve_colors)):
                if x_axis == "N":
                    observed = (
                        cells[
                            cells["comparison_slot"].eq(slot)
                            & cells["prompt_mode"].eq(mode)
                            & cells["L"].eq(curve_level)
                        ]
                        .set_index("N")
                        .reindex(n_levels)[column]
                        .to_numpy(float)
                    )
                    fitted = fitted_by_mode[mode][slot_index, :, curve_index]
                else:
                    observed = (
                        cells[
                            cells["comparison_slot"].eq(slot)
                            & cells["prompt_mode"].eq(mode)
                            & cells["N"].eq(curve_level)
                        ]
                        .set_index("L")
                        .reindex(l_levels)[column]
                        .to_numpy(float)
                    )
                    fitted = fitted_by_mode[mode][slot_index, curve_index, :]
                ax.plot(
                    x_levels,
                    fitted,
                    color=color,
                    linewidth=1.25,
                    linestyle=line_style,
                    alpha=0.92,
                )
                ax.scatter(
                    x_levels,
                    observed,
                    s=11 if mode_index == 0 else 13,
                    marker=marker,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.62,
                    alpha=0.88,
                    zorder=3,
                )
        if x_axis == "N":
            apply_all_n_ticks(ax, n_levels, fontsize=5.7)
            ax.set_xlim(0.9, 22)
        else:
            apply_all_l_ticks(ax, l_levels, fontsize=6.0)
        if family.startswith("accuracy_"):
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        elif family == MAE_FAMILY:
            ax.set_yscale("symlog", linthresh=0.25, linscale=0.75)
        ax.grid(color=AURORA["gray"], alpha=0.20, linewidth=0.52)
        ax.spines[["top", "right"]].set_visible(False)
        dagger = "†" if slot in MATCHED_PAIR_SLOTS else ""
        ax.set_title(f"{slot}{dagger}", loc="left", fontsize=9.0, fontweight="bold", pad=5)
        if row_index == 2:
            ax.set_xlabel(
                "Target count N (log2 positions)" if x_axis == "N" else "Passage length L (tokens; log positions)",
                fontsize=8.0,
            )
        if column_index == 0:
            ax.set_ylabel(ylabel, fontsize=8.2)
        ax.tick_params(labelsize=6.8)

    mode_handles = [
        plt.Line2D([], [], color=AURORA["indigo"], linestyle="-", marker="o", markerfacecolor="white", linewidth=1.6, markersize=4, label=MODE_SHORT[modes[0]]),
        plt.Line2D([], [], color=AURORA["indigo"], linestyle="--", marker="^", markerfacecolor="white", linewidth=1.6, markersize=4.2, label=MODE_SHORT[modes[1]]),
    ]
    curve_handles = [
        plt.Line2D(
            [], [], color=color, linewidth=1.8,
            label=(f"L={level // 1000}k" if x_axis == "N" else f"N={level}"),
        )
        for level, color in zip(curve_levels, curve_colors)
    ]
    legend_columns = 5 if x_axis == "N" else 8
    fig.legend(
        handles=mode_handles + curve_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=legend_columns,
        frameon=False,
        fontsize=6.7,
        handlelength=2.1,
        columnspacing=0.9,
    )
    fig.suptitle(title, x=0.055, y=0.997, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.895), h_pad=1.25, w_pad=0.95)
    savefig(fig, path)


def plot_aggregate_law_curves(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    family: str,
    column: str,
    modes: tuple[str, str],
    x_axis: str,
    path: Path,
    title: str,
    ylabel: str,
    fitted_override: dict[str, np.ndarray] | None = None,
    piecewise_breakpoint: int | None = None,
) -> None:
    """Render a two-mode scaling-law summary without pooling coefficients.

    Points and error bars are the median and interquartile range of the 12
    observed model slots.  Lines and ribbons are the same summaries of 12
    independently fitted slot-specific equations.  The ribbons therefore
    describe model heterogeneity, not sampling uncertainty or confidence.
    """
    if x_axis not in {"N", "L"}:
        raise ValueError(f"Unsupported x_axis={x_axis}")
    n_levels = sorted(int(value) for value in cells["N"].unique())
    l_levels = sorted(int(value) for value in cells["L"].unique())
    x_levels = n_levels if x_axis == "N" else l_levels
    curve_levels = l_levels if x_axis == "N" else n_levels
    colors = length_colors(curve_levels)
    fitted_by_mode = fitted_override if fitted_override is not None else {
        mode: predictions_by_slot(
            coefficients, mode, family, n_levels, l_levels
        )
        for mode in modes
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.35), sharey=True)
    for ax, mode in zip(axes, modes):
        fitted = fitted_by_mode[mode]
        law = selected[
            selected["outcome_family"].eq(family)
            & selected["prompt_mode"].eq(mode)
        ].iloc[0]
        for curve_index, (curve_level, color) in enumerate(
            zip(curve_levels, colors)
        ):
            observed_by_slot: list[np.ndarray] = []
            for slot in SLOT_ORDER:
                block = cells[
                    cells["comparison_slot"].eq(slot)
                    & cells["prompt_mode"].eq(mode)
                ]
                if x_axis == "N":
                    observed = (
                        block[block["L"].eq(curve_level)]
                        .set_index("N")
                        .reindex(n_levels)[column]
                        .to_numpy(float)
                    )
                else:
                    observed = (
                        block[block["N"].eq(curve_level)]
                        .set_index("L")
                        .reindex(l_levels)[column]
                        .to_numpy(float)
                    )
                observed_by_slot.append(observed)
            observed_matrix = np.vstack(observed_by_slot)
            observed_q25, observed_q50, observed_q75 = np.nanquantile(
                observed_matrix, [0.25, 0.50, 0.75], axis=0
            )
            fitted_matrix = (
                fitted[:, :, curve_index]
                if x_axis == "N"
                else fitted[:, curve_index, :]
            )
            fitted_q25, fitted_q50, fitted_q75 = np.nanquantile(
                fitted_matrix, [0.25, 0.50, 0.75], axis=0
            )

            # Do not interpolate across an unconstrained piecewise boundary.
            xs = np.asarray(x_levels)
            segments = [np.ones(len(xs), dtype=bool)]
            if piecewise_breakpoint is not None and x_axis == "N":
                segments = [xs <= piecewise_breakpoint, xs > piecewise_breakpoint]
            for segment in segments:
                ax.fill_between(xs[segment], fitted_q25[segment], fitted_q75[segment],
                                color=color, alpha=0.095, linewidth=0, zorder=1)
                ax.plot(xs[segment], fitted_q50[segment], color=color,
                        linewidth=1.9, alpha=0.96, zorder=2)
            ax.errorbar(
                x_levels,
                observed_q50,
                yerr=[
                    observed_q50 - observed_q25,
                    observed_q75 - observed_q50,
                ],
                fmt="o",
                markersize=4.0,
                markerfacecolor=AURORA["white"],
                markeredgecolor=color,
                markeredgewidth=1.05,
                ecolor=color,
                elinewidth=0.72,
                capsize=1.7,
                alpha=0.88,
                zorder=3,
            )

        if x_axis == "N":
            if piecewise_breakpoint is not None:
                ax.axvspan(0.9, piecewise_breakpoint, color=AURORA["gray"], alpha=0.08)
            apply_all_n_ticks(ax, n_levels, fontsize=7.0)
            ax.set_xlim(0.9, 22)
            ax.set_xlabel("Target count N (log2 positions)")
        else:
            apply_all_l_ticks(ax, l_levels, fontsize=7.2)
            ax.set_xlabel("Passage length L (tokens; log positions)")
        if family.startswith("accuracy_"):
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(1.0)
            )
            score_name = "D²"
        elif family == MAE_FAMILY:
            ax.set_yscale("symlog", linthresh=0.25, linscale=0.75)
            ax.axhline(
                0, color=AURORA["gray"], linewidth=0.8, alpha=0.65
            )
            score_name = "R²"
        else:
            ax.axhline(
                0, color=AURORA["gray"], linewidth=0.8, alpha=0.65
            )
            score_name = "R²"
        ax.set_title(
            f"{MODE_SHORT[mode]}\n"
            f"{PLOT_FORMULA[law.selected_candidate]} · "
            f"median CV {score_name}={law.median_primary_score:.2f}"
            + (" (N>=4)" if piecewise_breakpoint is not None else ""),
            loc="left",
            fontsize=11.2,
            pad=10,
        )
        ax.grid(color=AURORA["gray"], alpha=0.22, linewidth=0.62)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel)

    curve_handles = [
        plt.Line2D(
            [],
            [],
            color=color,
            linewidth=2.0,
            label=(
                f"L={level // 1000}k" if x_axis == "N" else f"N={level}"
            ),
        )
        for level, color in zip(curve_levels, colors)
    ]
    fig.legend(
        handles=curve_handles,
        title="Passage length" if x_axis == "N" else "Target count",
        loc="upper center",
        bbox_to_anchor=(0.51, 0.935),
        ncol=4 if x_axis == "N" else 7,
        frameon=False,
        fontsize=7.1,
        title_fontsize=7.5,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.suptitle(
        title,
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.6)
    savefig(fig, path)


def accuracy_predictions_by_slot(
    coefficients: pd.DataFrame,
    mode: str,
    n_levels: list[int],
    l_levels: list[int],
) -> np.ndarray:
    return predictions_by_slot(
        coefficients,
        mode,
        "accuracy_bernoulli_logit",
        n_levels,
        l_levels,
    )


def plot_l_axis_law_curves(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    family: str,
    column: str,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Transpose the law view: L on x and all registered N as curves."""
    modes = ["direct", "native_thinking"]
    n_levels = sorted(int(x) for x in cells["N"].unique())
    l_levels = sorted(int(x) for x in cells["L"].unique())
    colors = target_count_colors(n_levels)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.5), sharey=True)
    for ax, mode in zip(axes, modes):
        fitted = predictions_by_slot(
            coefficients, mode, family, n_levels, l_levels
        )
        law = selected[
            selected["outcome_family"].eq(family)
            & selected["prompt_mode"].eq(mode)
        ].iloc[0]
        for n_index, (target, color) in enumerate(zip(n_levels, colors)):
            observed_by_slot = []
            for slot in SLOT_ORDER:
                observed_by_slot.append(
                    cells[
                        cells["comparison_slot"].eq(slot)
                        & cells["prompt_mode"].eq(mode)
                        & cells["N"].eq(target)
                    ]
                    .set_index("L")
                    .reindex(l_levels)[column]
                    .to_numpy(float)
                )
            observed = np.vstack(observed_by_slot)
            q25, q50, q75 = np.nanquantile(observed, [0.25, 0.50, 0.75], axis=0)
            fitted_q50 = np.nanmedian(fitted[:, n_index, :], axis=0)
            ax.vlines(
                l_levels,
                q25,
                q75,
                color=color,
                linewidth=0.55,
                alpha=0.28,
                zorder=1,
            )
            ax.plot(
                l_levels,
                fitted_q50,
                color=color,
                linewidth=1.55,
                alpha=0.94,
                zorder=2,
            )
            ax.scatter(
                l_levels,
                q50,
                s=15,
                facecolor=AURORA["white"],
                edgecolor=color,
                linewidth=0.7,
                zorder=3,
            )
        apply_all_l_ticks(ax, l_levels, fontsize=7.6)
        if family == "trimmed_signed_bias_10":
            ax.axhline(0, color=AURORA["gray"], linewidth=0.8, alpha=0.7)
        if family.startswith("accuracy_"):
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        elif family == MAE_FAMILY:
            ax.set_ylim(bottom=-0.02)
        ax.set_xlabel("Passage length L (tokens; log scale)")
        ax.set_title(
            f"{MODE_SHORT[mode]}\n{PLOT_FORMULA[law.selected_candidate]} · median CV R²={law.median_primary_score:.2f}"
            if not family.startswith("accuracy_")
            else f"{MODE_SHORT[mode]}\n{PLOT_FORMULA[law.selected_candidate]} · median CV D²={law.median_primary_score:.2f}",
            loc="left",
            fontsize=11.2,
            pad=10,
        )
        ax.grid(alpha=0.42)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel)
    handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=f"N={target}")
        for target, color in zip(n_levels, colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.935),
        ncol=7,
        frameon=False,
        fontsize=7.3,
    )
    fig.suptitle(title, x=0.06, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    savefig(fig, path)


def plot_accuracy_law_curves(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    selected: pd.DataFrame,
    path: Path,
) -> None:
    """Scaling-law view: exact accuracy points plus selected-law curves."""
    n_levels = sorted(int(x) for x in cells["N"].unique())
    l_levels = sorted(int(x) for x in cells["L"].unique())
    colors = length_colors(l_levels)
    modes = ["direct", "native_thinking"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)

    for ax, mode in zip(axes, modes):
        predictions = accuracy_predictions_by_slot(
            coefficients, mode, n_levels, l_levels
        )
        law = selected[
            selected["outcome_family"].eq("accuracy_bernoulli_logit")
            & selected["prompt_mode"].eq(mode)
        ].iloc[0]
        for length_index, (length, color) in enumerate(zip(l_levels, colors)):
            observed = (
                cells[
                    cells["prompt_mode"].eq(mode)
                    & cells["L"].eq(length)
                ]
                .pivot(index="comparison_slot", columns="N", values="parsed_exact_accuracy")
                .reindex(index=SLOT_ORDER, columns=n_levels)
                .to_numpy(float)
            )
            observed_median = np.nanmedian(observed, axis=0)
            observed_q25 = np.nanquantile(observed, 0.25, axis=0)
            observed_q75 = np.nanquantile(observed, 0.75, axis=0)
            fitted = predictions[:, :, length_index]
            fitted_median = np.nanmedian(fitted, axis=0)
            fitted_q25 = np.nanquantile(fitted, 0.25, axis=0)
            fitted_q75 = np.nanquantile(fitted, 0.75, axis=0)

            ax.fill_between(
                n_levels,
                fitted_q25,
                fitted_q75,
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(
                n_levels,
                fitted_median,
                color=color,
                linewidth=2.1,
                label=f"L={length // 1000}k",
            )
            ax.errorbar(
                n_levels,
                observed_median,
                yerr=[observed_median - observed_q25, observed_q75 - observed_median],
                fmt="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
                ecolor=color,
                elinewidth=0.8,
                capsize=1.8,
                alpha=0.88,
                zorder=3,
            )

        apply_all_n_ticks(ax, n_levels, fontsize=7.2)
        ax.set_xlim(0.9, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(color=AURORA["gray"], alpha=0.24, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Target count N (log2 scale)")
        ax.set_title(
            f"{MODE_SHORT[mode]}\n{PLOT_FORMULA[law.selected_candidate]} · median CV D²={law.median_primary_score:.2f}",
            loc="left",
            fontsize=11.5,
            pad=12,
        )
    axes[0].set_ylabel("Parsed exact accuracy")
    axes[1].legend(
        title="Passage length",
        loc="lower left",
        frameon=False,
        ncol=4,
        fontsize=7.4,
        title_fontsize=7.8,
    )
    fig.suptitle(
        "Observed exact accuracy and selected empirical-law curves",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, path)


def plot_accuracy_model_panels(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    path: Path,
    modes: tuple[str, str] = ("direct", "native_thinking"),
    title: str = "Model-resolved observed accuracy and selected-law fits",
) -> None:
    """Show all 12 comparison slots, overlaying both modes in each panel."""
    n_levels = sorted(int(x) for x in cells["N"].unique())
    l_levels = sorted(int(x) for x in cells["L"].unique())
    colors = length_colors(l_levels)
    fitted_by_mode = {
        mode: accuracy_predictions_by_slot(coefficients, mode, n_levels, l_levels)
        for mode in modes
    }
    for mode, values in fitted_by_mode.items():
        if values.shape[0] != len(SLOT_ORDER):
            raise ValueError(
                f"Expected {len(SLOT_ORDER)} accuracy slots for {mode}, got {values.shape[0]}"
            )

    fig, axes = plt.subplots(3, 4, figsize=(15.8, 10.8), sharex=True, sharey=True)
    for slot_index, slot in enumerate(SLOT_ORDER):
        row_index = slot_index // 4
        column_index = slot_index % 4
        ax = axes[row_index, column_index]
        for mode_index, mode in enumerate(modes):
            line_style = "-" if mode_index == 0 else "--"
            marker = "o" if mode_index == 0 else "^"
            for length_index, (length, color) in enumerate(zip(l_levels, colors)):
                observed = (
                    cells[
                        cells["comparison_slot"].eq(slot)
                        & cells["prompt_mode"].eq(mode)
                        & cells["L"].eq(length)
                    ]
                    .set_index("N")
                    .reindex(n_levels)["parsed_exact_accuracy"]
                    .to_numpy(float)
                )
                fitted = fitted_by_mode[mode][slot_index, :, length_index]
                ax.plot(
                    n_levels,
                    fitted,
                    color=color,
                    linewidth=1.5,
                    linestyle=line_style,
                    alpha=0.96,
                )
                ax.scatter(
                    n_levels,
                    observed,
                    s=14 if mode == "direct" else 16,
                    marker=marker,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.75,
                    zorder=3,
                )
        apply_all_n_ticks(ax, n_levels, fontsize=6.1)
        ax.set_xlim(0.9, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(color=AURORA["gray"], alpha=0.22, linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        dagger = "†" if slot in MATCHED_PAIR_SLOTS else ""
        ax.set_title(
            f"{slot}{dagger}",
            loc="left",
            fontsize=9.2,
            fontweight="bold",
            pad=6,
        )
        if row_index == 2:
            ax.set_xlabel("Target count N (log2)", fontsize=8.5)
        if column_index == 0:
            ax.set_ylabel("Exact accuracy", fontsize=8.5)
        ax.tick_params(labelsize=7.5)

    length_handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=f"L={length // 1000}k")
        for length, color in zip(l_levels, colors)
    ]
    mode_handles = [
        plt.Line2D([], [], color=AURORA["indigo"], linewidth=1.8, linestyle="-", marker="o", markerfacecolor=AURORA["white"], markersize=4.2, label=MODE_SHORT[modes[0]]),
        plt.Line2D([], [], color=AURORA["indigo"], linewidth=1.8, linestyle="--", marker="^", markerfacecolor=AURORA["white"], markersize=4.5, label=MODE_SHORT[modes[1]]),
    ]
    fig.legend(
        handles=mode_handles + length_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=5,
        frameon=False,
        fontsize=7.6,
    )
    fig.suptitle(
        title,
        x=0.055,
        y=0.997,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.905), h_pad=1.35, w_pad=1.05)
    savefig(fig, path)


def plot_bias_model_panels(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    path: Path,
) -> None:
    """Show model-specific observed and fitted 10% trimmed signed bias."""
    n_levels = sorted(int(x) for x in cells["N"].unique())
    l_levels = sorted(int(x) for x in cells["L"].unique())
    colors = length_colors(l_levels)
    modes = ("direct", "native_thinking")
    fitted_by_mode = {
        mode: predictions_by_slot(
            coefficients, mode, "trimmed_signed_bias_10", n_levels, l_levels
        )
        for mode in modes
    }
    observed_values = cells[
        cells["prompt_mode"].isin(modes) & cells["L"].isin(l_levels)
    ]["trimmed_signed_bias_10"].to_numpy(float)
    fitted_values = np.concatenate([value.ravel() for value in fitted_by_mode.values()])
    lower = min(float(np.nanmin(observed_values)), float(np.nanmin(fitted_values)))
    upper = max(float(np.nanmax(observed_values)), float(np.nanmax(fitted_values)))
    padding = max(0.5, 0.05 * (upper - lower))

    fig, axes = plt.subplots(3, 4, figsize=(15.8, 10.8), sharex=True, sharey=True)
    for slot_index, slot in enumerate(SLOT_ORDER):
        row_index, column_index = divmod(slot_index, 4)
        ax = axes[row_index, column_index]
        ax.axhline(0, color=AURORA["gray"], linewidth=0.8, alpha=0.65, zorder=0)
        for mode_index, mode in enumerate(modes):
            line_style = "-" if mode_index == 0 else "--"
            marker = "o" if mode_index == 0 else "^"
            for length_index, (length, color) in enumerate(zip(l_levels, colors)):
                observed = (
                    cells[
                        cells["comparison_slot"].eq(slot)
                        & cells["prompt_mode"].eq(mode)
                        & cells["L"].eq(length)
                    ]
                    .set_index("N")
                    .reindex(n_levels)["trimmed_signed_bias_10"]
                    .to_numpy(float)
                )
                fitted = fitted_by_mode[mode][slot_index, :, length_index]
                ax.plot(
                    n_levels,
                    fitted,
                    color=color,
                    linewidth=1.5,
                    linestyle=line_style,
                    alpha=0.96,
                )
                ax.scatter(
                    n_levels,
                    observed,
                    s=14 if mode_index == 0 else 16,
                    marker=marker,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.75,
                    zorder=3,
                )
        apply_all_n_ticks(ax, n_levels, fontsize=6.1)
        ax.set_xlim(0.9, 22)
        ax.set_ylim(lower - padding, upper + padding)
        ax.grid(color=AURORA["gray"], alpha=0.22, linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        dagger = "†" if slot in MATCHED_PAIR_SLOTS else ""
        ax.set_title(f"{slot}{dagger}", loc="left", fontsize=9.2, fontweight="bold", pad=6)
        if row_index == 2:
            ax.set_xlabel("Target count N (log2)", fontsize=8.5)
        if column_index == 0:
            ax.set_ylabel("10% trimmed signed bias", fontsize=8.5)
        ax.tick_params(labelsize=7.5)

    length_handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=f"L={length // 1000}k")
        for length, color in zip(l_levels, colors)
    ]
    mode_handles = [
        plt.Line2D([], [], color=AURORA["indigo"], linewidth=1.8, linestyle="-", marker="o", markerfacecolor=AURORA["white"], markersize=4.2, label="Non-thinking"),
        plt.Line2D([], [], color=AURORA["indigo"], linewidth=1.8, linestyle="--", marker="^", markerfacecolor=AURORA["white"], markersize=4.5, label="Native-thinking"),
    ]
    fig.legend(
        handles=mode_handles + length_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=5,
        frameon=False,
        fontsize=7.6,
    )
    fig.suptitle(
        "Model-resolved observed bias and selected-law fits",
        x=0.055,
        y=0.997,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.905), h_pad=1.35, w_pad=1.05)
    savefig(fig, path)


def observed_surface(cells: pd.DataFrame, mode: str, column: str, n_levels: list[int], l_levels: list[int]) -> np.ndarray:
    data = cells[cells["prompt_mode"].eq(mode)]
    grouped = data.groupby(["N", "L"], as_index=False)[column].median()
    return grouped.pivot(index="N", columns="L", values=column).reindex(index=n_levels, columns=l_levels).to_numpy(float)


def plot_surfaces(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    selected: pd.DataFrame,
    family: str,
    column: str,
    path: Path,
) -> None:
    n_levels = sorted(int(x) for x in cells["N"].unique())
    l_levels = sorted(int(x) for x in cells["L"].unique())
    modes = ["direct", "native_thinking"]
    if family.startswith("accuracy_"):
        vmin, vmax = 0.0, 1.0
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "report_acc",
            [SCALING_LAW["purple"], SCALING_LAW["magenta"], SCALING_LAW["orange"], SCALING_LAW["yellow"]],
        )
        label = "Parsed exact accuracy"
    else:
        observed = np.concatenate([observed_surface(cells, mode, column, n_levels, l_levels).ravel() for mode in modes])
        vmax = float(np.nanquantile(np.abs(observed), 0.98))
        vmax = max(vmax, 0.5)
        vmin = -vmax
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "report_bias", [SCALING_LAW["purple"], AURORA["white"], SCALING_LAW["yellow"]]
        )
        label = "10% trimmed signed bias"
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.1), sharex=True, sharey=True)
    image = None
    for row_idx, mode in enumerate(modes):
        observed = observed_surface(cells, mode, column, n_levels, l_levels)
        fitted = median_fitted_surface(coefficients, mode, family, n_levels, l_levels)
        law = selected[selected["outcome_family"].eq(family) & selected["prompt_mode"].eq(mode)].iloc[0]
        for col_idx, (matrix, kind) in enumerate([(observed, "Observed median"), (fitted, "Median fitted")]):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{MODE_SHORT[mode]} · {kind}", loc="left", fontsize=11.5)
            ax.set_xticks(np.arange(len(l_levels)), [f"{x // 1000}k" for x in l_levels], rotation=0)
            ax.set_yticks(np.arange(len(n_levels)), n_levels)
            ax.set_xlabel("Passage length L (tokens)")
            ax.set_ylabel("Target count N")
            if col_idx == 1:
                score_name = "D²" if family.startswith("accuracy_") else "R²"
                ax.text(0.02, 0.98, f"{PLOT_FORMULA[law.selected_candidate]}\nmedian CV {score_name}={law.median_primary_score:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=8.8, color=AURORA["black"], bbox={"facecolor": AURORA["white"], "edgecolor": "none", "alpha": 0.84, "pad": 5})
    fig.subplots_adjust(right=0.90, hspace=0.28, wspace=0.16)
    cax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(label)
    savefig(fig, path)


def plot_three_estimand_surfaces(
    accuracy_cells: pd.DataFrame,
    accuracy_coefficients: pd.DataFrame,
    accuracy_selected: pd.DataFrame,
    mae_cells: pd.DataFrame,
    mae_coefficients: pd.DataFrame,
    mae_selected: pd.DataFrame,
    path: Path,
) -> None:
    """Show the three report estimands as observed points over fitted surfaces.

    Each panel aggregates *after* fitting: the points are cell-wise medians
    across the 12 comparison slots, while the translucent surface is the
    pointwise median of the 12 slot-specific selected-law predictions.  This
    preserves the report's model-resolved estimand and never invents a pooled
    coefficient vector.
    """
    n_levels = sorted(int(x) for x in accuracy_cells["N"].unique())
    l_levels = sorted(int(x) for x in accuracy_cells["L"].unique())
    n_grid, l_grid_k = np.meshgrid(
        np.asarray(n_levels, dtype=float),
        np.asarray(l_levels, dtype=float) / 1000.0,
        indexing="ij",
    )
    modes = ["direct", "native_thinking"]
    specs = [
        {
            "label": "Parsed exact accuracy",
            "family": "accuracy_bernoulli_logit",
            "column": "parsed_exact_accuracy",
            "cells": accuracy_cells,
            "coefficients": accuracy_coefficients,
            "selected": accuracy_selected,
            "score": "D²",
        },
        {
            "label": "10% trimmed conditional MAE",
            "family": MAE_FAMILY,
            "column": MAE_FAMILY,
            "cells": mae_cells,
            "coefficients": mae_coefficients,
            "selected": mae_selected,
            "score": "R²",
        },
        {
            "label": "10% trimmed signed bias",
            "family": "trimmed_signed_bias_10",
            "column": "trimmed_signed_bias_10",
            "cells": accuracy_cells,
            "coefficients": accuracy_coefficients,
            "selected": accuracy_selected,
            "score": "R²",
        },
    ]
    cmap = matplotlib.colormaps["plasma_r"]
    fig = plt.figure(figsize=(15.8, 16.8))
    axes = np.empty((3, 2), dtype=object)
    row_scalars: list[tuple[plt.cm.ScalarMappable, str, bool]] = []
    for row_idx, spec in enumerate(specs):
        observed_by_mode = {
            mode: observed_surface(
                spec["cells"], mode, spec["column"], n_levels, l_levels
            )
            for mode in modes
        }
        fitted_by_mode = {
            mode: median_fitted_surface(
                spec["coefficients"], mode, spec["family"], n_levels, l_levels
            )
            for mode in modes
        }
        finite = np.concatenate(
            [
                matrix[np.isfinite(matrix)]
                for matrix in [*observed_by_mode.values(), *fitted_by_mode.values()]
            ]
        )
        if spec["family"].startswith("accuracy_"):
            vmin, vmax = 0.0, 1.0
        elif spec["family"] == "trimmed_signed_bias_10":
            vmax = max(float(np.nanmax(np.abs(finite))), 0.5)
            vmin = -vmax
        else:
            vmin = min(float(np.nanmin(finite)), 0.0)
            vmax = max(float(np.nanmax(finite)), 0.5)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for col_idx, mode in enumerate(modes):
            ax = fig.add_subplot(3, 2, row_idx * 2 + col_idx + 1, projection="3d")
            axes[row_idx, col_idx] = ax
            observed = observed_by_mode[mode]
            fitted = fitted_by_mode[mode]
            ax.plot_surface(
                n_grid,
                l_grid_k,
                fitted,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                alpha=0.70,
                linewidth=0.32,
                edgecolor=(0.14, 0.09, 0.36, 0.24),
                antialiased=True,
                rstride=1,
                cstride=1,
            )
            ax.scatter(
                n_grid.ravel(),
                l_grid_k.ravel(),
                observed.ravel(),
                c=observed.ravel(),
                cmap=cmap,
                norm=norm,
                s=18,
                linewidth=0.45,
                edgecolor=AURORA["indigo"],
                depthshade=False,
                alpha=0.96,
            )
            law = spec["selected"][
                spec["selected"]["outcome_family"].eq(spec["family"])
                & spec["selected"]["prompt_mode"].eq(mode)
            ].iloc[0]
            ax.set_title(
                f"{spec['label']} · {MODE_SHORT[mode]}\n"
                f"{PLOT_FORMULA[law.selected_candidate]} · median CV {spec['score']}={law.median_primary_score:.2f}",
                loc="left",
                fontsize=10.5,
                pad=12,
                fontweight="bold",
            )
            ax.set_xlabel("Target count N", labelpad=7)
            ax.set_ylabel("Passage length L (k tokens)", labelpad=8)
            if col_idx == 0:
                ax.set_zlabel(spec["label"], labelpad=8)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_yticks([1, 5, 10, 15, 20])
            ax.set_xlim(1, 20)
            ax.set_ylim(1, 20)
            ax.set_zlim(vmin, vmax)
            if spec["family"].startswith("accuracy_"):
                ax.zaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.view_init(elev=25, azim=-128)
            ax.set_box_aspect((1.18, 1.0, 0.72))
            ax.grid(True, alpha=0.23)
            ax.xaxis.pane.set_facecolor((0.97, 0.98, 1.0, 0.72))
            ax.yaxis.pane.set_facecolor((0.97, 0.98, 1.0, 0.72))
            ax.zaxis.pane.set_facecolor((0.97, 0.98, 1.0, 0.72))
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        row_scalars.append(
            (scalar, spec["label"], spec["family"].startswith("accuracy_"))
        )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=AURORA["white"],
            markeredgecolor=AURORA["indigo"],
            markersize=6,
            label="Observed median across 12 model slots",
        ),
        plt.Line2D(
            [],
            [],
            color=SCALING_LAW["magenta"],
            linewidth=7,
            alpha=0.62,
            label="Median of 12 slot-specific fitted surfaces",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Three estimands on the registered N–L grid",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.03,
        right=0.86,
        top=0.90,
        bottom=0.025,
        hspace=0.24,
        wspace=0.02,
    )
    colorbar_positions = [
        (0.888, 0.684, 0.014, 0.175),
        (0.888, 0.365, 0.014, 0.175),
        (0.888, 0.047, 0.014, 0.175),
    ]
    short_labels = ["Accuracy", "Trimmed MAE", "Trimmed bias"]
    for (scalar, _label, is_accuracy), position, short_label in zip(
        row_scalars, colorbar_positions, short_labels
    ):
        cax = fig.add_axes(position)
        cbar = fig.colorbar(scalar, cax=cax)
        cbar.set_label(
            f"{short_label} · yellow low → purple high",
            fontsize=8.2,
            labelpad=7,
        )
        cbar.ax.tick_params(labelsize=7.5)
        if is_accuracy:
            cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    savefig(fig, path)


def build_three_estimand_interactive_payload(
    accuracy_cells: pd.DataFrame,
    accuracy_coefficients: pd.DataFrame,
    accuracy_selected: pd.DataFrame,
    mae_cells: pd.DataFrame,
    mae_coefficients: pd.DataFrame,
    mae_selected: pd.DataFrame,
) -> dict[str, object]:
    """Build a compact, auditable payload for the rotatable Appendix-D view.

    The observed markers remain on the registered 14-by-8 grid.  The displayed
    surface is evaluated on a denser grid only to make rotation legible; it is
    still the pointwise median of the same 12 slot-specific selected-law
    predictions, not a new fit or a pooled coefficient vector.
    """
    n_levels = sorted(int(x) for x in accuracy_cells["N"].unique())
    l_levels = sorted(int(x) for x in accuracy_cells["L"].unique())
    n_fine = np.linspace(float(min(n_levels)), float(max(n_levels)), 48)
    l_fine = np.linspace(float(min(l_levels)), float(max(l_levels)), 48)
    n_obs_grid, l_obs_grid_k = np.meshgrid(
        np.asarray(n_levels, dtype=float),
        np.asarray(l_levels, dtype=float) / 1000.0,
        indexing="ij",
    )
    modes = ["direct", "native_thinking"]
    specs = [
        {
            "key": "accuracy",
            "label": "Exact accuracy",
            "short_label": "Accuracy",
            "family": "accuracy_bernoulli_logit",
            "column": "parsed_exact_accuracy",
            "cells": accuracy_cells,
            "coefficients": accuracy_coefficients,
            "selected": accuracy_selected,
            "score": "D²",
            "unit": "probability",
            "tickformat": ".0%",
        },
        {
            "key": "mae",
            "label": "10% trimmed conditional MAE",
            "short_label": "Trimmed MAE",
            "family": MAE_FAMILY,
            "column": MAE_FAMILY,
            "cells": mae_cells,
            "coefficients": mae_coefficients,
            "selected": mae_selected,
            "score": "R²",
            "unit": "count units",
            "tickformat": ".2f",
        },
        {
            "key": "bias",
            "label": "10% trimmed signed bias",
            "short_label": "Trimmed bias",
            "family": "trimmed_signed_bias_10",
            "column": "trimmed_signed_bias_10",
            "cells": accuracy_cells,
            "coefficients": accuracy_coefficients,
            "selected": accuracy_selected,
            "score": "R²",
            "unit": "count units",
            "tickformat": ".2f",
        },
    ]
    payload: dict[str, object] = {
        "schema_version": "niah_appendix_d_plotly_v1",
        "registered_grid": {"N": n_levels, "L_tokens": l_levels},
        "surface_grid_points": int(len(n_fine) * len(l_fine)),
        "colorscale": [
            [0.0, SCALING_LAW["yellow"]],
            [0.34, SCALING_LAW["orange"]],
            [0.68, SCALING_LAW["magenta"]],
            [1.0, SCALING_LAW["purple"]],
        ],
        "metrics": {},
    }
    for spec in specs:
        mode_payloads: dict[str, object] = {}
        scale_values: list[np.ndarray] = []
        for mode in modes:
            observed = observed_surface(
                spec["cells"], mode, spec["column"], n_levels, l_levels
            )
            fitted_registered = median_fitted_surface(
                spec["coefficients"], mode, spec["family"], n_levels, l_levels
            )
            fitted_fine = median_fitted_surface(
                spec["coefficients"],
                mode,
                spec["family"],
                n_fine.tolist(),
                l_fine.tolist(),
            )
            law = spec["selected"][
                spec["selected"]["outcome_family"].eq(spec["family"])
                & spec["selected"]["prompt_mode"].eq(mode)
            ].iloc[0]
            scale_values.extend([observed.ravel(), fitted_fine.ravel()])
            mode_payloads[mode] = {
                "mode_label": MODE_SHORT[mode],
                "formula": PLOT_FORMULA[law.selected_candidate],
                "score_name": spec["score"],
                "median_score": round(float(law.median_primary_score), 6),
                "observed": {
                    "x": n_obs_grid.ravel().astype(float).tolist(),
                    "y": l_obs_grid_k.ravel().astype(float).tolist(),
                    "z": np.round(observed.ravel(), 8).tolist(),
                    "fitted": np.round(fitted_registered.ravel(), 8).tolist(),
                    "residual": np.round(
                        (observed - fitted_registered).ravel(), 8
                    ).tolist(),
                },
                "surface": {
                    "x": np.round(n_fine, 7).tolist(),
                    "y": np.round(l_fine / 1000.0, 7).tolist(),
                    # Plotly surface rows follow y and columns follow x.
                    "z": np.round(fitted_fine.T, 8).tolist(),
                },
            }
        finite = np.concatenate(
            [values[np.isfinite(values)] for values in scale_values]
        )
        if spec["key"] == "accuracy":
            zmin, zmax = 0.0, 1.0
        elif spec["key"] == "bias":
            limit = max(float(np.max(np.abs(finite))), 0.5)
            zmin, zmax = -limit, limit
        else:
            zmin = min(float(np.min(finite)), 0.0)
            zmax = max(float(np.max(finite)), 0.5)
        payload["metrics"][spec["key"]] = {
            "label": spec["label"],
            "short_label": spec["short_label"],
            "unit": spec["unit"],
            "tickformat": spec["tickformat"],
            "zmin": round(zmin, 8),
            "zmax": round(zmax, 8),
            "modes": mode_payloads,
        }
    return payload


def plot_fit_scores(metrics: pd.DataFrame, path: Path) -> None:
    family_map = {"accuracy_bernoulli_logit": ("Accuracy CV D²", "primary_score"), "trimmed_signed_bias_10": ("Bias CV R²", "primary_score")}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.4), sharey=True)
    y = np.arange(len(SLOT_ORDER))[::-1]
    for ax, (family, (title, value_col)) in zip(axes, family_map.items()):
        data = metrics[metrics["outcome_family"].eq(family) & metrics["prompt_mode"].isin(["direct", "native_thinking"])].copy()
        pivot = data.pivot(index="comparison_slot", columns="prompt_mode", values=value_col).reindex(SLOT_ORDER)
        for idx, (slot, row) in enumerate(pivot.iterrows()):
            yy = y[idx]
            ax.plot([row["direct"], row["native_thinking"]], [yy, yy], color="#D4DCE7", lw=2.5)
            ax.scatter(row["direct"], yy, s=55, color=MODE_COLOR["direct"], edgecolor=AURORA["indigo"], lw=0.5, zorder=3)
            ax.scatter(row["native_thinking"], yy, s=55, color=MODE_COLOR["native_thinking"], edgecolor=AURORA["indigo"], lw=0.5, zorder=3)
        ax.axvline(0, color=AURORA["gray"], lw=1, ls="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Held-condition score (higher is better)")
        ax.grid(axis="x")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, SLOT_ORDER)
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=MODE_COLOR["direct"], markersize=7, label="Non-thinking"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=MODE_COLOR["native_thinking"], markersize=7, label="Native-thinking"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.51, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, path)


def plot_interactions(coefficients: pd.DataFrame, path: Path) -> None:
    data = coefficients[
        coefficients["outcome_family"].eq("trimmed_signed_bias_10")
        & coefficients["prompt_mode"].isin(["direct", "native_thinking"])
        & coefficients["term"].eq("N_x_L_k")
    ].copy()
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    y = np.arange(len(SLOT_ORDER))[::-1]
    offsets = {"direct": 0.13, "native_thinking": -0.13}
    for mode in ["direct", "native_thinking"]:
        block = data[data["prompt_mode"].eq(mode)].set_index("comparison_slot").reindex(SLOT_ORDER)
        for idx, (slot, row) in enumerate(block.iterrows()):
            yy = y[idx] + offsets[mode]
            significant = bool(row["hc3_q"] <= 0.05) if pd.notna(row["hc3_q"]) else False
            ax.scatter(row["standardized_effect"], yy, s=70, facecolor=MODE_COLOR[mode] if significant else AURORA["white"], edgecolor=MODE_COLOR[mode], lw=1.5, zorder=3)
    ax.axvline(0, color=AURORA["gray"], lw=1.1)
    ax.axvspan(-0.1, 0.1, color="#E8EDF4", alpha=0.8, zorder=0)
    ax.set_yticks(y, SLOT_ORDER)
    ax.set_xlabel("Standardized coefficient of N × (L/1,000) (signed SD units)")
    ax.set_title("Model-level standardized N×L interaction effects", loc="left", pad=18)
    ax.text(0, 1.025, "Filled: within-slot HC3 coefficient has BH-adjusted q≤0.05; gray band: |effect|<0.1", transform=ax.transAxes, fontsize=9.3, color="#637086")
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=MODE_COLOR["direct"], markeredgecolor=MODE_COLOR["direct"], markersize=8, label="Non-thinking"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=MODE_COLOR["native_thinking"], markeredgecolor=MODE_COLOR["native_thinking"], markersize=8, label="Native-thinking"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=AURORA["white"], markeredgecolor=AURORA["gray"], markersize=8, label="q>0.05"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    savefig(fig, path)


def plot_all_mode_accuracy(summary: pd.DataFrame, path: Path) -> None:
    pivot = summary.pivot(index="comparison_slot", columns="prompt_mode", values="parsed_exact_accuracy").reindex(index=SLOT_ORDER, columns=MODE_ORDER)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "report_acc",
        [SCALING_LAW["purple"], SCALING_LAW["magenta"], SCALING_LAW["orange"], SCALING_LAW["yellow"]],
    )
    fig, ax = plt.subplots(figsize=(9.8, 7.4))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(MODE_ORDER)), [MODE_SHORT[x] for x in MODE_ORDER], rotation=18, ha="right")
    ax.set_yticks(np.arange(len(SLOT_ORDER)), SLOT_ORDER)
    for i in range(len(SLOT_ORDER)):
        for j in range(len(MODE_ORDER)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{100 * value:.0f}", ha="center", va="center", fontsize=8.2, color=AURORA["white"] if value < 0.48 else AURORA["black"])
    ax.set_title("Parsed exact accuracy across all four prompt modes", loc="left", pad=15)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Parsed exact accuracy (0–1)")
    savefig(fig, path)


def plot_enumeration_degeneracy(cells: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    modes = ["enumeration_index", "enumeration_bullet"]
    rows = []
    for (slot, mode), block in cells[cells["prompt_mode"].isin(modes)].groupby(["comparison_slot", "prompt_mode"]):
        zero_fraction = float(np.mean(np.isclose(block["trimmed_signed_bias_10"], 0.0, atol=1e-12)))
        fit = metrics[
            metrics["comparison_slot"].eq(slot)
            & metrics["prompt_mode"].eq(mode)
            & metrics["outcome_family"].eq("trimmed_signed_bias_10")
        ].iloc[0]
        acc = summary[summary["comparison_slot"].eq(slot) & summary["prompt_mode"].eq(mode)]["parsed_exact_accuracy"].iloc[0]
        rows.append({"slot": slot, "mode": mode, "zero_fraction": zero_fraction, "r2": fit.primary_score, "accuracy": acc})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    for mode in modes:
        block = data[data["mode"].eq(mode)]
        ax.scatter(block["zero_fraction"], block["r2"], s=45 + 150 * block["accuracy"], color=MODE_COLOR[mode], edgecolor=AURORA["indigo"], lw=0.6, alpha=0.86, label=MODE_SHORT[mode])
    ax.axhline(0, color=AURORA["gray"], lw=1, ls="--")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Fraction of N × L cells with exactly zero 10% trimmed bias")
    ax.set_ylabel("Held-condition CV R² of the selected bias law")
    ax.set_title("Zero-bias cell fraction and held-condition bias R²", loc="left", pad=18)
    ax.text(0, 1.025, "Point area increases with slot accuracy; low R² does not imply low accuracy", transform=ax.transAxes, fontsize=9.3, color="#637086")
    ax.grid()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    savefig(fig, path)


def law_rows(selected: pd.DataFrame, modes: list[str], families: list[str]) -> pd.DataFrame:
    data = selected[selected["prompt_mode"].isin(modes) & selected["outcome_family"].isin(families)].copy()
    data["Outcome"] = data["outcome_family"].map(
        {
            "accuracy_bernoulli_logit": "Accuracy · logit",
            "accuracy_bernoulli_probit": "Accuracy · probit",
            "accuracy_bernoulli_cloglog": "Accuracy · cloglog",
            "trimmed_signed_bias_10": "10% trimmed bias",
            MAE_FAMILY: "Conditional MAE · 10% symmetric trimming · identity scale",
        }
    )
    data["Mode"] = data["prompt_mode"].map(MODE_SHORT)
    data["Selected law"] = data["selected_candidate"].map(FORMULA)
    data["Median CV score"] = data["median_primary_score"].map(lambda x: num(x, 3))
    data["Q25 CV score"] = data["q25_primary_score"].map(lambda x: num(x, 3))
    data["Median CV loss"] = data["median_primary_loss"].map(lambda x: num(x, 3))
    data["LOMO stability"] = data["lomo_formula_stability"].map(lambda x: pct(x, 1))
    data["LOMO held score"] = data["lomo_median_held_primary_score"].map(
        lambda x: num(x, 3)
    )
    data["Reading"] = data["evidence_reading"]
    return data[
        [
            "Outcome",
            "Mode",
            "Selected law",
            "Median CV score",
            "Q25 CV score",
            "Median CV loss",
            "LOMO stability",
            "LOMO held score",
            "Reading",
        ]
    ]


TERM_SYMBOL = {
    "N": "N",
    "L_k": "Lₖ",
    "logN": "ln N",
    "logL": "ln Lₖ",
    "N_x_L_k": "N·Lₖ",
    "logN_x_logL": "ln N·ln Lₖ",
    "N_x_logL": "N·ln Lₖ",
    "logN_x_L_k": "ln N·Lₖ",
    "invN": "1/N",
    "invN_x_L_k": "(1/N)·Lₖ",
    "invN_x_logL": "(1/N)·ln Lₖ",
}
TERM_ORDER = [
    "N",
    "logN",
    "invN",
    "L_k",
    "logL",
    "N_x_L_k",
    "logN_x_logL",
    "N_x_logL",
    "logN_x_L_k",
    "invN_x_L_k",
    "invN_x_logL",
]


def fitted_equation_table(
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
    mode: str,
    family: str,
) -> pd.DataFrame:
    """Create an auditable table of the 12 slot-specific fitted equations."""
    rows: list[dict[str, str]] = []
    metric_block = metrics[
        metrics["prompt_mode"].eq(mode)
        & metrics["outcome_family"].eq(family)
    ].set_index("comparison_slot")
    for slot in SLOT_ORDER:
        block = coefficients[
            coefficients["comparison_slot"].eq(slot)
            & coefficients["prompt_mode"].eq(mode)
            & coefficients["outcome_family"].eq(family)
        ]
        if block.empty:
            raise ValueError(f"Missing coefficients for {slot} / {mode} / {family}")
        estimates = {str(row.term): float(row.estimate) for row in block.itertuples(index=False)}
        if family.startswith("accuracy_"):
            lhs = "logit(p̂)"
        elif family == MAE_FAMILY:
            lhs = "MAE-hat"
        else:
            lhs = "b̂₁₀"
        expression = f"{lhs} = {estimates['intercept']:.5g}"
        for term in TERM_ORDER:
            if term not in estimates:
                continue
            value = estimates[term]
            sign = "+" if value >= 0 else "−"
            expression += f" {sign} {abs(value):.5g} {TERM_SYMBOL[term]}"
        metric = metric_block.loc[slot]
        rows.append(
            {
                "Model": f"{slot}{'†' if slot in MATCHED_PAIR_SLOTS else ''}",
                "Fitted equation": expression,
                "Held-condition D²" if family.startswith("accuracy_") else "Held-condition R²": num(metric.primary_score, 3),
            }
        )
    return pd.DataFrame(rows)


TERM_LATEX = {
    "N": "N",
    "L_k": "L_k",
    "logN": r"\ln N",
    "logL": r"\ln L_k",
    "N_x_L_k": r"N L_k",
    "logN_x_logL": r"(\ln N)(\ln L_k)",
    "N_x_logL": r"N\ln L_k",
    "logN_x_L_k": r"L_k\ln N",
    "invN": r"N^{-1}",
    "invN_x_L_k": r"L_kN^{-1}",
    "invN_x_logL": r"N^{-1}\ln L_k",
}


def latex_equation_details(
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
    mode: str,
    family: str,
    summary: str,
) -> str:
    """Create collapsible MathJax equations without pooling slot coefficients."""
    metric_block = metrics[
        metrics["prompt_mode"].eq(mode)
        & metrics["outcome_family"].eq(family)
    ].set_index("comparison_slot")
    rows: list[str] = []
    for slot in SLOT_ORDER:
        block = coefficients[
            coefficients["comparison_slot"].eq(slot)
            & coefficients["prompt_mode"].eq(mode)
            & coefficients["outcome_family"].eq(family)
        ]
        estimates = {str(row.term): float(row.estimate) for row in block.itertuples(index=False)}
        if family.startswith("accuracy_"):
            lhs = rf"\operatorname{{logit}}\widehat{{p}}_{{\mathrm{{{mode}}},s}}"
            score_name = "D²"
        elif family == MAE_FAMILY:
            lhs = rf"\widehat{{\operatorname{{MAE}}}}_{{\mathrm{{{mode}}},s}}"
            score_name = "R²"
        else:
            lhs = rf"\widehat{{b}}_{{10,\mathrm{{{mode}}},s}}"
            score_name = "R²"
        expression = f"{lhs}={estimates['intercept']:.5g}"
        for term in TERM_ORDER:
            if term not in estimates:
                continue
            value = estimates[term]
            operator = "+" if value >= 0 else "-"
            expression += f"{operator}{abs(value):.5g}{TERM_LATEX[term]}"
        score = float(metric_block.loc[slot, "primary_score"])
        rows.append(
            "<tr><td>"
            + html.escape(f"{slot}{'†' if slot in MATCHED_PAIR_SLOTS else ''}")
            + "</td><td>\\("
            + expression
            + "\\)</td><td>"
            + html.escape(f"{score_name}={score:.3f}")
            + "</td></tr>"
        )
    return (
        '<details class="equations"><summary>'
        + html.escape(summary)
        + '</summary><div class="table-wrap"><table class="data-table equation-table">'
        + '<thead><tr><th>Model</th><th>Fitted equation</th><th>Held-condition score</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div></details>"
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    download = args.download_root.resolve()
    run_root = require(download / "20260819_formal")
    behavior_dir = require(download / "analysis" / "v3_1_behavior_empirical_law")
    frozen_law_dir = require(download / "analysis" / "v3_2_empirical_law")
    law_dir = require(
        download / "analysis" / "v3_2_inverse_n_candidate_extension"
    )
    sensitivity_dir = require(
        download / "analysis" / "v3_2_untrimmed_bias_sensitivity"
    )
    count_error_dir = require(
        download / "analysis" / "v3_2_trimmed_count_error_extension"
    )
    behavior_tables = require(behavior_dir / "tables")
    law_tables = require(law_dir / "tables")
    sensitivity_tables = require(sensitivity_dir / "tables")
    count_error_tables = require(count_error_dir / "tables")
    audit_path = require(run_root / "orchestration" / "final_shard_audit.json")
    audit = read_json(audit_path)
    manifest = read_json(law_dir / "analysis_manifest.json")
    frozen_manifest = read_json(frozen_law_dir / "analysis_manifest.json")
    sensitivity_manifest = read_json(sensitivity_dir / "analysis_manifest.json")
    count_error_manifest = read_json(count_error_dir / "analysis_manifest.json")
    state = read_json(law_dir / "analysis_state.json")
    v31 = read_json(args.v3_1_config)
    v32 = read_json(args.v3_2_config)

    if not (audit.get("passed") and audit.get("requests") == 161280 and audit.get("unique_request_ids") == 161280):
        raise ValueError("Frozen inference audit did not pass the 161,280-request gate")
    if not (
        state.get("stage") == "complete"
        and manifest.get("status") == "complete"
        and manifest.get("bootstrap_repetitions") == 0
        and manifest.get("base_candidates") == 13
        and manifest.get("expanded_candidates") == 18
        and manifest.get("selection_and_validation_changed") is False
    ):
        raise ValueError("The V3.2 inverse-count candidate extension is incomplete or mismatched")
    if not (
        sensitivity_manifest.get("status") == "complete"
        and sensitivity_manifest.get("exploratory") is True
        and sensitivity_manifest.get("changed_estimand_only") is True
        and sensitivity_manifest.get("input_sha256") == frozen_manifest.get("input_sha256")
    ):
        raise ValueError("Untrimmed-bias sensitivity analysis is incomplete or mismatched")
    if not (
        count_error_manifest.get("status") == "complete"
        and count_error_manifest.get("requests") == 161280
        and count_error_manifest.get("input_sha256") == frozen_manifest.get("input_sha256")
        and count_error_manifest.get("bootstrap_repetitions") == 0
        and count_error_manifest.get("candidate_registry_size") == 18
        and count_error_manifest.get("parent_v3_2_config_verified") is True
        and count_error_manifest.get("outcome", {}).get("id") == MAE_FAMILY
        and count_error_manifest.get("outcome", {}).get(
            "symmetric_trim_fraction_each_tail"
        ) == 0.10
    ):
        raise ValueError("Trimmed count-error extension is incomplete or mismatched")

    summary = pd.read_csv(require(behavior_tables / "model_mode_summary.csv"))
    cells = pd.read_csv(require(law_tables / "cell_outcomes.csv.gz"))
    selected = pd.read_csv(require(law_tables / "selected_mode_laws.csv"))
    candidate_summary = pd.read_csv(require(law_tables / "mode_candidate_summary.csv"))
    metrics = pd.read_csv(require(law_tables / "selected_model_fit_metrics.csv"))
    coefficients = pd.read_csv(require(law_tables / "selected_model_coefficients.csv"))
    try:
        beta_metrics = pd.read_csv(require(law_tables / "beta_binomial_fit_metrics.csv"))
    except pd.errors.EmptyDataError:
        # The inverse-N extension may be rebuilt with --skip-beta-binomial;
        # robustness metrics are immutable and remain available in frozen V3.2.
        beta_metrics = pd.read_csv(
            require(frozen_law_dir / "tables" / "beta_binomial_fit_metrics.csv")
        )
    inverse_selection_comparison = pd.read_csv(
        require(law_tables / "baseline_vs_inverse_n_selection.csv")
    )
    sensitivity_comparison = pd.read_csv(
        require(sensitivity_tables / "trimmed_vs_untrimmed_comparison.csv")
    )
    sensitivity_diagnostics = pd.read_csv(
        require(sensitivity_tables / "estimand_diagnostics.csv")
    )
    sensitivity_coefficients = pd.read_csv(
        require(sensitivity_tables / "selected_model_coefficients.csv")
    )
    count_error_cells = pd.read_csv(
        require(count_error_tables / "count_error_cells.csv.gz")
    )
    mae_selected = pd.read_csv(
        require(count_error_tables / "mae_selected_mode_laws.csv")
    )
    mae_metrics = pd.read_csv(
        require(count_error_tables / "mae_selected_model_fit_metrics.csv")
    )
    mae_coefficients = pd.read_csv(
        require(count_error_tables / "mae_selected_model_coefficients.csv")
    )
    bias_influence = pd.read_csv(
        require(count_error_tables / "bias_cell_influence_diagnostics.csv")
    )
    bias_influence_summary = pd.read_csv(
        require(count_error_tables / "bias_influence_summary.csv")
    )
    bias_request_tails = pd.read_csv(
        require(count_error_tables / "bias_request_tail_diagnostics.csv")
    )

    switchable = set(v31["switchable_models"])
    switchable_slots = switchable.intersection(set(summary["comparison_slot"]))
    agg_switch = weighted_mode_summary(summary, switchable_slots)
    agg_all = weighted_mode_summary(summary)

    assets = args.output.parent / "niah_empirical_law_v3_2_assets"
    assets.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "accuracy_aggregate_n": assets / "fig00a_accuracy_aggregate_by_N.png",
        "accuracy_aggregate_l": assets / "fig00b_accuracy_aggregate_by_L.png",
        "accuracy_n": assets / "fig01_accuracy_by_N_all_models.png",
        "accuracy_l": assets / "fig02_accuracy_by_L_all_models.png",
        "mae_aggregate_n": assets / "fig02a_trimmed_mae_aggregate_by_N.png",
        "mae_aggregate_l": assets / "fig02b_trimmed_mae_aggregate_by_L.png",
        "mae_n": assets / "fig03_trimmed_mae_by_N_all_models.png",
        "mae_l": assets / "fig04_trimmed_mae_by_L_all_models.png",
        "interaction": assets / "figA1_shared_interaction_strength.png",
        "bias_aggregate_n": assets / "figB0a_bias_aggregate_by_N.png",
        "bias_aggregate_l": assets / "figB0b_bias_aggregate_by_L.png",
        "bias_n": assets / "figB1_bias_by_N_all_models.png",
        "bias_l": assets / "figB2_bias_by_L_all_models.png",
        "enum_accuracy_n": assets / "figC1_enumeration_accuracy_by_N.png",
        "enum_accuracy_l": assets / "figC2_enumeration_accuracy_by_L.png",
        "enum_mae_n": assets / "figC3_enumeration_mae_by_N.png",
        "enum_mae_l": assets / "figC4_enumeration_mae_by_L.png",
        "appendix_d_3d": assets / "figD1_three_estimands_3d.png",
    }
    set_plot_style()
    for x_axis, key in [
        ("N", "accuracy_aggregate_n"),
        ("L", "accuracy_aggregate_l"),
    ]:
        plot_aggregate_law_curves(
            cells,
            coefficients,
            selected,
            family="accuracy_bernoulli_logit",
            column="parsed_exact_accuracy",
            modes=("direct", "native_thinking"),
            x_axis=x_axis,
            path=figure_paths[key],
            title=(
                "Observed exact accuracy and selected empirical-law curves: "
                + ("target count" if x_axis == "N" else "passage length")
                + " on the horizontal axis"
            ),
            ylabel="Parsed exact accuracy",
        )
    plot_model_law_panels(
        cells,
        coefficients,
        family="accuracy_bernoulli_logit",
        column="parsed_exact_accuracy",
        modes=("direct", "native_thinking"),
        x_axis="N",
        path=figure_paths["accuracy_n"],
        title="Accuracy laws across all models: target count on the horizontal axis",
        ylabel="Parsed exact accuracy",
    )
    for x_axis, key in [
        ("N", "mae_aggregate_n"),
        ("L", "mae_aggregate_l"),
    ]:
        plot_aggregate_law_curves(
            count_error_cells,
            mae_coefficients,
            mae_selected,
            family=MAE_FAMILY,
            column=MAE_FAMILY,
            modes=("direct", "native_thinking"),
            x_axis=x_axis,
            path=figure_paths[key],
            title=(
                "Observed 10% trimmed conditional MAE and selected empirical-law curves: "
                + ("target count" if x_axis == "N" else "passage length")
                + " on the horizontal axis"
            ),
            ylabel="10% trimmed conditional MAE (count units; symlog)",
        )
    plot_model_law_panels(
        cells,
        coefficients,
        family="accuracy_bernoulli_logit",
        column="parsed_exact_accuracy",
        modes=("direct", "native_thinking"),
        x_axis="L",
        path=figure_paths["accuracy_l"],
        title="Accuracy laws across all models: passage length on the horizontal axis",
        ylabel="Parsed exact accuracy",
    )
    plot_model_law_panels(
        count_error_cells,
        mae_coefficients,
        family=MAE_FAMILY,
        column=MAE_FAMILY,
        modes=("direct", "native_thinking"),
        x_axis="N",
        path=figure_paths["mae_n"],
        title="10% trimmed conditional MAE laws across all models: target count on the horizontal axis",
        ylabel="10% trimmed conditional MAE (count units; symlog)",
    )
    plot_model_law_panels(
        count_error_cells,
        mae_coefficients,
        family=MAE_FAMILY,
        column=MAE_FAMILY,
        modes=("direct", "native_thinking"),
        x_axis="L",
        path=figure_paths["mae_l"],
        title="10% trimmed conditional MAE laws across all models: passage length on the horizontal axis",
        ylabel="10% trimmed conditional MAE (count units; symlog)",
    )
    plot_interactions(coefficients, figure_paths["interaction"])
    for x_axis, key in [
        ("N", "bias_aggregate_n"),
        ("L", "bias_aggregate_l"),
    ]:
        plot_aggregate_law_curves(
            cells,
            coefficients,
            selected,
            family="trimmed_signed_bias_10",
            column="trimmed_signed_bias_10",
            modes=("direct", "native_thinking"),
            x_axis=x_axis,
            path=figure_paths[key],
            title=(
                "Observed 10% trimmed signed bias and selected empirical-law curves: "
                + ("target count" if x_axis == "N" else "passage length")
                + " on the horizontal axis"
            ),
            ylabel="10% trimmed signed bias (count units)",
        )
    for x_axis, key in [("N", "bias_n"), ("L", "bias_l")]:
        plot_model_law_panels(
            cells,
            coefficients,
            family="trimmed_signed_bias_10",
            column="trimmed_signed_bias_10",
            modes=("direct", "native_thinking"),
            x_axis=x_axis,
            path=figure_paths[key],
            title=f"10% trimmed signed-bias laws: {'target count' if x_axis == 'N' else 'passage length'} on the horizontal axis",
            ylabel="10% trimmed signed bias (count units)",
        )
    plot_three_estimand_surfaces(
        cells,
        coefficients,
        selected,
        count_error_cells,
        mae_coefficients,
        mae_selected,
        figure_paths["appendix_d_3d"],
    )
    for x_axis, key in [("N", "enum_accuracy_n"), ("L", "enum_accuracy_l")]:
        plot_model_law_panels(
            cells,
            coefficients,
            family="accuracy_bernoulli_logit",
            column="parsed_exact_accuracy",
            modes=("enumeration_index", "enumeration_bullet"),
            x_axis=x_axis,
            path=figure_paths[key],
            title=f"Enumeration accuracy laws: {'target count' if x_axis == 'N' else 'passage length'} on the horizontal axis",
            ylabel="Parsed exact accuracy",
        )
    for x_axis, key in [("N", "enum_mae_n"), ("L", "enum_mae_l")]:
        plot_model_law_panels(
            count_error_cells,
            mae_coefficients,
            family=MAE_FAMILY,
            column=MAE_FAMILY,
            modes=("enumeration_index", "enumeration_bullet"),
            x_axis=x_axis,
            path=figure_paths[key],
            title=f"Enumeration 10% trimmed conditional MAE laws: {'target count' if x_axis == 'N' else 'passage length'} on the horizontal axis",
            ylabel="10% trimmed conditional MAE (count units; symlog)",
        )
    figures = {key: b64_png(path) for key, path in figure_paths.items()}
    # Compatibility aliases keep the historical template evaluable below; the
    # final artifact is replaced by the focused two-part V3.2 template.
    figures.update(
        {
            "accuracy_dumbbell": figures["accuracy_n"],
            "accuracy_surfaces": figures["accuracy_n"],
            "accuracy_l_axis": figures["accuracy_l"],
            "accuracy_model_panels": figures["accuracy_n"],
            "bias_surfaces": figures["bias_n"],
            "bias_l_axis": figures["bias_l"],
            "bias_model_panels": figures["bias_n"],
            "mae_l_axis": figures["mae_l"],
            "enumeration_model_panels": figures["enum_accuracy_n"],
        }
    )
    appendix_d_payload = build_three_estimand_interactive_payload(
        cells,
        coefficients,
        selected,
        count_error_cells,
        mae_coefficients,
        mae_selected,
    )
    appendix_d_payload_json = json.dumps(
        appendix_d_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    plotly_bundle_path = require(root / "scripts" / "vendor" / "plotly-3.6.0.min.js")
    plotly_bundle = plotly_bundle_path.read_text(encoding="utf-8").replace(
        "</script", "<\\/script"
    )
    appendix_d_script = r"""
    (() => {
      const shell = document.getElementById("appendix-d-interactive");
      const payloadNode = document.getElementById("appendix-d-payload");
      if (!shell || !payloadNode || typeof Plotly === "undefined") return;
      const explorer = shell.querySelector(".d3-explorer");
      // Plotly must see the final grid width during its first layout pass.
      // Keep the explorer in flow but invisible until both canvases are ready.
      explorer.style.display = "block";
      explorer.style.visibility = "hidden";
      const payload = JSON.parse(payloadNode.textContent);
      const plots = {
        direct: document.getElementById("appendix-d-direct"),
        native_thinking: document.getElementById("appendix-d-native")
      };
      const labels = {
        direct: document.getElementById("appendix-d-direct-law"),
        native_thinking: document.getElementById("appendix-d-native-law")
      };
      const metricNote = document.getElementById("appendix-d-metric-note");
      const buttons = Array.from(shell.querySelectorAll("[data-d3-metric]"));
      const resetButton = document.getElementById("appendix-d-reset");
      const camera = {eye:{x:1.58,y:1.58,z:1.08},center:{x:0,y:0,z:-0.05},up:{x:0,y:0,z:1}};
      const plotConfig = {
        responsive:true,
        scrollZoom:true,
        displaylogo:false,
        modeBarButtonsToRemove:["toImage","sendDataToCloud","select2d","lasso2d"],
        showTips:true
      };
      let metricKey = "accuracy";
      let syncingCamera = false;

      function tracesFor(metric, modeKey) {
        const mode = metric.modes[modeKey];
        const obs = mode.observed;
        const observedCustom = obs.fitted.map((value, index) => [value, obs.residual[index]]);
        const showScale = modeKey === "native_thinking";
        return [
          {
            type:"surface",
            x:mode.surface.x,
            y:mode.surface.y,
            z:mode.surface.z,
            name:"Median fitted surface",
            colorscale:payload.colorscale,
            cmin:metric.zmin,
            cmax:metric.zmax,
            opacity:0.80,
            showscale:showScale,
            colorbar:showScale ? {
              title:{text:metric.short_label,side:"right",font:{size:11,color:"#475467"}},
              thickness:12,
              len:0.66,
              x:1.02,
              tickfont:{size:10,color:"#667085"},
              tickformat:metric.tickformat,
              outlinewidth:0
            } : undefined,
            contours:{
              z:{show:true,usecolormap:true,project:{z:true},highlightcolor:"#F8FBFF",width:1}
            },
            lighting:{ambient:0.72,diffuse:0.78,specular:0.18,roughness:0.72,fresnel:0.08},
            lightposition:{x:80,y:-110,z:170},
            hovertemplate:
              "<b>Median fitted surface</b><br>Target count N=%{x:.2f}" +
              "<br>Passage length L=%{y:.2f}k" +
              "<br>Fitted " + metric.short_label + "=%{z:" + metric.tickformat + "}<extra></extra>"
          },
          {
            type:"scatter3d",
            mode:"markers",
            x:obs.x,
            y:obs.y,
            z:obs.z,
            customdata:observedCustom,
            name:"Observed median",
            marker:{
              size:4.5,
              color:obs.z,
              colorscale:payload.colorscale,
              cmin:metric.zmin,
              cmax:metric.zmax,
              opacity:0.98,
              line:{color:"#23165C",width:1.25},
              showscale:false
            },
            hovertemplate:
              "<b>Observed median across 12 slots</b><br>Target count N=%{x:.0f}" +
              "<br>Passage length L=%{y:.0f}k" +
              "<br>Observed " + metric.short_label + "=%{z:" + metric.tickformat + "}" +
              "<br>Median fitted=%{customdata[0]:" + metric.tickformat + "}" +
              "<br>Observed − fitted=%{customdata[1]:.3f}<extra></extra>"
          }
        ];
      }

      function layoutFor(metric, modeKey) {
        const mode = metric.modes[modeKey];
        return {
          autosize:true,
          margin:{l:0,r:modeKey === "native_thinking" ? 58 : 10,b:0,t:8},
          paper_bgcolor:"rgba(0,0,0,0)",
          plot_bgcolor:"rgba(0,0,0,0)",
          showlegend:false,
          uirevision:"niah-appendix-d-camera",
          scene:{
            bgcolor:"#F8FAFC",
            camera:camera,
            dragmode:"orbit",
            aspectmode:"manual",
            aspectratio:{x:1.08,y:1,z:0.78},
            xaxis:{
              title:{text:"Target N",font:{size:11,color:"#344054"}},
              range:[1,20],tickvals:payload.registered_grid.N,
              tickfont:{size:9,color:"#667085"},gridcolor:"#D7DEE8",zerolinecolor:"#8190A5",
              showbackground:true,backgroundcolor:"#F8FAFC",showspikes:false
            },
            yaxis:{
              title:{text:"Length L (k tokens)",font:{size:11,color:"#344054"}},
              range:[1,20],tickvals:payload.registered_grid.L_tokens.map(x => x/1000),
              tickfont:{size:9,color:"#667085"},gridcolor:"#D7DEE8",zerolinecolor:"#8190A5",
              showbackground:true,backgroundcolor:"#F8FAFC",showspikes:false
            },
            zaxis:{
              title:{text:metric.short_label,font:{size:11,color:"#344054"}},
              range:[metric.zmin,metric.zmax],tickformat:metric.tickformat,
              tickfont:{size:9,color:"#667085"},gridcolor:"#D7DEE8",zerolinecolor:"#8190A5",
              showbackground:true,backgroundcolor:"#F8FAFC",showspikes:false
            }
          },
          hoverlabel:{bgcolor:"#161923",bordercolor:"#6750E8",font:{color:"#F8FBFF",size:12}}
        };
      }

      function renderMetric(nextMetric) {
        metricKey = nextMetric;
        const metric = payload.metrics[metricKey];
        buttons.forEach(button => {
          const active = button.dataset.d3Metric === metricKey;
          button.classList.toggle("is-active", active);
          button.setAttribute("aria-pressed", String(active));
        });
        metricNote.textContent = metric.label + " · " + metric.unit +
          " · yellow = lower, purple = higher";
        for (const modeKey of Object.keys(plots)) {
          const mode = metric.modes[modeKey];
          labels[modeKey].textContent = mode.formula + " · median CV " +
            mode.score_name + "=" + mode.median_score.toFixed(2);
        }
        return Promise.all(Object.keys(plots).map(modeKey =>
          Plotly.react(plots[modeKey], tracesFor(metric, modeKey), layoutFor(metric, modeKey), plotConfig)
        ));
      }

      function mirrorCamera(sourceKey, targetKey) {
        plots[sourceKey].on("plotly_relayout", event => {
          const nextCamera = event["scene.camera"];
          if (!nextCamera || syncingCamera) return;
          syncingCamera = true;
          Plotly.relayout(plots[targetKey], {"scene.camera":nextCamera})
            .finally(() => { syncingCamera = false; });
        });
      }

      buttons.forEach(button => button.addEventListener("click", () =>
        renderMetric(button.dataset.d3Metric)
      ));
      resetButton.addEventListener("click", () => {
        Plotly.relayout(plots.direct, {"scene.camera":camera});
        Plotly.relayout(plots.native_thinking, {"scene.camera":camera});
      });
      renderMetric(metricKey).then(() => {
        mirrorCamera("direct", "native_thinking");
        mirrorCamera("native_thinking", "direct");
        shell.classList.add("is-ready");
        explorer.style.visibility = "visible";
        Plotly.Plots.resize(plots.direct);
        Plotly.Plots.resize(plots.native_thinking);
      });
    })();
    """

    headline = selected[selected["outcome_family"].isin(["accuracy_bernoulli_logit", "trimmed_signed_bias_10"])]
    direct_acc = headline[headline["outcome_family"].eq("accuracy_bernoulli_logit") & headline["prompt_mode"].eq("direct")].iloc[0]
    native_acc = headline[headline["outcome_family"].eq("accuracy_bernoulli_logit") & headline["prompt_mode"].eq("native_thinking")].iloc[0]
    direct_bias = headline[headline["outcome_family"].eq("trimmed_signed_bias_10") & headline["prompt_mode"].eq("direct")].iloc[0]
    native_bias = headline[headline["outcome_family"].eq("trimmed_signed_bias_10") & headline["prompt_mode"].eq("native_thinking")].iloc[0]
    direct_interaction = selected[
        selected["outcome_family"].eq("trimmed_signed_bias_10")
        & selected["prompt_mode"].eq("direct")
    ].iloc[0]
    native_interaction = selected[
        selected["outcome_family"].eq("trimmed_signed_bias_10")
        & selected["prompt_mode"].eq("native_thinking")
    ].iloc[0]

    accuracy_coefficients = coefficients[
        coefficients["outcome_family"].eq("accuracy_bernoulli_logit")
        & coefficients["prompt_mode"].isin(["direct", "native_thinking"])
    ]

    def accuracy_coefficient_summary(mode: str, term: str) -> tuple[float, float]:
        values = accuracy_coefficients[
            accuracy_coefficients["prompt_mode"].eq(mode)
            & accuracy_coefficients["term"].eq(term)
        ]["estimate"]
        return float(values.median()), float((values < 0).mean())

    direct_log_n, direct_log_n_negative = accuracy_coefficient_summary("direct", "logN")
    direct_l_k, direct_l_k_negative = accuracy_coefficient_summary("direct", "L_k")
    native_n, native_n_negative = accuracy_coefficient_summary("native_thinking", "N")
    native_log_l, native_log_l_negative = accuracy_coefficient_summary("native_thinking", "logL")

    def coefficient_estimate_summary(
        frame: pd.DataFrame,
        outcome_family: str,
        mode: str,
        term: str,
    ) -> tuple[float, float, float]:
        values = pd.to_numeric(
            frame[
                frame["outcome_family"].eq(outcome_family)
                & frame["prompt_mode"].eq(mode)
                & frame["term"].eq(term)
            ]["estimate"],
            errors="coerce",
        ).dropna()
        if values.empty:
            raise ValueError(f"No coefficient estimates for {outcome_family}/{mode}/{term}")
        return float(values.median()), float(values.min()), float(values.max())

    direct_mae_n_raw = coefficient_estimate_summary(
        mae_coefficients, MAE_FAMILY, "direct", "N"
    )
    direct_mae_l_raw = coefficient_estimate_summary(
        mae_coefficients, MAE_FAMILY, "direct", "L_k"
    )
    direct_mae_nl_raw = coefficient_estimate_summary(
        mae_coefficients, MAE_FAMILY, "direct", "N_x_L_k"
    )
    native_mae_n_raw = coefficient_estimate_summary(
        mae_coefficients, MAE_FAMILY, "native_thinking", "N"
    )
    native_mae_l_raw = coefficient_estimate_summary(
        mae_coefficients, MAE_FAMILY, "native_thinking", "L_k"
    )
    native_mae_nl_raw = coefficient_estimate_summary(
        mae_coefficients, MAE_FAMILY, "native_thinking", "N_x_L_k"
    )

    accuracy_candidates = candidate_summary[
        candidate_summary["outcome_family"].eq("accuracy_bernoulli_logit")
        & candidate_summary["prompt_mode"].isin(["direct", "native_thinking"])
    ]
    direct_best_raw = accuracy_candidates[
        accuracy_candidates["prompt_mode"].eq("direct")
    ].sort_values("median_primary_score", ascending=False).iloc[0]
    direct_inverse_candidates = accuracy_candidates[
        accuracy_candidates["prompt_mode"].eq("direct")
        & accuracy_candidates["candidate"].str.startswith("invN")
    ]
    direct_best_inverse_raw = direct_inverse_candidates.sort_values(
        "median_primary_score", ascending=False
    ).iloc[0]
    direct_best_inverse_eligible = direct_inverse_candidates[
        direct_inverse_candidates["selection_gate_pass"].astype(bool)
    ].sort_values("median_primary_score", ascending=False).iloc[0]
    direct_loglog = accuracy_candidates[
        accuracy_candidates["prompt_mode"].eq("direct")
        & accuracy_candidates["candidate"].eq("logN__logL")
    ].iloc[0]
    native_loglog = accuracy_candidates[
        accuracy_candidates["prompt_mode"].eq("native_thinking")
        & accuracy_candidates["candidate"].eq("logN__logL")
    ].iloc[0]

    direct_raw = agg_switch[agg_switch["prompt_mode"].eq("direct")].iloc[0]
    native_raw = agg_switch[agg_switch["prompt_mode"].eq("native_thinking")].iloc[0]
    raw_delta = native_raw.accuracy - direct_raw.accuracy
    overall = audit["evaluation"]
    overall_accuracy = overall["exact_count_correct"] / overall["requests"]

    direct_mae = mae_selected[
        mae_selected["prompt_mode"].eq("direct")
    ].iloc[0]
    native_mae = mae_selected[
        mae_selected["prompt_mode"].eq("native_thinking")
    ].iloc[0]
    index_acc = selected[
        selected["outcome_family"].eq("accuracy_bernoulli_logit")
        & selected["prompt_mode"].eq("enumeration_index")
    ].iloc[0]
    bullet_acc = selected[
        selected["outcome_family"].eq("accuracy_bernoulli_logit")
        & selected["prompt_mode"].eq("enumeration_bullet")
    ].iloc[0]
    index_mae = mae_selected[mae_selected["prompt_mode"].eq("enumeration_index")].iloc[0]
    bullet_mae = mae_selected[mae_selected["prompt_mode"].eq("enumeration_bullet")].iloc[0]
    direct_mae_peak = count_error_cells[
        count_error_cells["prompt_mode"].eq("direct")
    ].sort_values(MAE_FAMILY, ascending=False).iloc[0]
    native_mae_peak = count_error_cells[
        count_error_cells["prompt_mode"].eq("native_thinking")
    ].sort_values(MAE_FAMILY, ascending=False).iloc[0]

    direct_accuracy_n_effect = coefficient_effect_summary(
        coefficients,
        outcome_family="accuracy_bernoulli_logit",
        prompt_mode="direct",
        term="logN",
    )
    direct_accuracy_l_effect = coefficient_effect_summary(
        coefficients,
        outcome_family="accuracy_bernoulli_logit",
        prompt_mode="direct",
        term="L_k",
    )
    native_accuracy_n_effect = coefficient_effect_summary(
        coefficients,
        outcome_family="accuracy_bernoulli_logit",
        prompt_mode="native_thinking",
        term="N",
    )
    native_accuracy_l_effect = coefficient_effect_summary(
        coefficients,
        outcome_family="accuracy_bernoulli_logit",
        prompt_mode="native_thinking",
        term="logL",
    )
    direct_mae_n_effect = coefficient_effect_summary(
        mae_coefficients,
        outcome_family=MAE_FAMILY,
        prompt_mode="direct",
        term="N",
    )
    direct_mae_l_effect = coefficient_effect_summary(
        mae_coefficients,
        outcome_family=MAE_FAMILY,
        prompt_mode="direct",
        term="L_k",
    )
    direct_mae_interaction_effect = coefficient_effect_summary(
        mae_coefficients,
        outcome_family=MAE_FAMILY,
        prompt_mode="direct",
        term="N_x_L_k",
    )
    native_mae_n_effect = coefficient_effect_summary(
        mae_coefficients,
        outcome_family=MAE_FAMILY,
        prompt_mode="native_thinking",
        term="N",
    )
    native_mae_l_effect = coefficient_effect_summary(
        mae_coefficients,
        outcome_family=MAE_FAMILY,
        prompt_mode="native_thinking",
        term="L_k",
    )
    native_mae_interaction_effect = coefficient_effect_summary(
        mae_coefficients,
        outcome_family=MAE_FAMILY,
        prompt_mode="native_thinking",
        term="N_x_L_k",
    )

    law_main = law_rows(selected, ["direct", "native_thinking"], ["accuracy_bernoulli_logit", "trimmed_signed_bias_10"])
    law_all = law_rows(selected, MODE_ORDER, ["accuracy_bernoulli_logit", "accuracy_bernoulli_probit", "accuracy_bernoulli_cloglog", "trimmed_signed_bias_10"])
    enumeration_accuracy_fit = law_rows(
        selected,
        ["enumeration_index", "enumeration_bullet"],
        ["accuracy_bernoulli_logit"],
    )
    direct_accuracy_equations = fitted_equation_table(coefficients, metrics, "direct", "accuracy_bernoulli_logit")
    native_accuracy_equations = fitted_equation_table(coefficients, metrics, "native_thinking", "accuracy_bernoulli_logit")
    direct_bias_equations = fitted_equation_table(coefficients, metrics, "direct", "trimmed_signed_bias_10")
    native_bias_equations = fitted_equation_table(coefficients, metrics, "native_thinking", "trimmed_signed_bias_10")
    index_accuracy_equations = fitted_equation_table(coefficients, metrics, "enumeration_index", "accuracy_bernoulli_logit")
    bullet_accuracy_equations = fitted_equation_table(coefficients, metrics, "enumeration_bullet", "accuracy_bernoulli_logit")
    mae_main = law_rows(
        mae_selected,
        ["direct", "native_thinking"],
        [MAE_FAMILY],
    )
    direct_mae_equations = fitted_equation_table(
        mae_coefficients, mae_metrics, "direct", MAE_FAMILY
    )
    native_mae_equations = fitted_equation_table(
        mae_coefficients, mae_metrics, "native_thinking", MAE_FAMILY
    )
    index_mae_equations = fitted_equation_table(
        mae_coefficients, mae_metrics, "enumeration_index", MAE_FAMILY
    )
    bullet_mae_equations = fitted_equation_table(
        mae_coefficients, mae_metrics, "enumeration_bullet", MAE_FAMILY
    )
    equation_details = {
        "direct_accuracy": latex_equation_details(coefficients, metrics, "direct", "accuracy_bernoulli_logit", "Non-thinking：12 个模型的 Accuracy 方程"),
        "native_accuracy": latex_equation_details(coefficients, metrics, "native_thinking", "accuracy_bernoulli_logit", "Native-thinking：12 个模型的 Accuracy 方程"),
        "direct_mae": latex_equation_details(mae_coefficients, mae_metrics, "direct", MAE_FAMILY, "Non-thinking：12 个模型的 trimmed-MAE 方程"),
        "native_mae": latex_equation_details(mae_coefficients, mae_metrics, "native_thinking", MAE_FAMILY, "Native-thinking：12 个模型的 trimmed-MAE 方程"),
        "direct_bias": latex_equation_details(coefficients, metrics, "direct", "trimmed_signed_bias_10", "Non-thinking：12 个模型的 Bias 方程"),
        "native_bias": latex_equation_details(coefficients, metrics, "native_thinking", "trimmed_signed_bias_10", "Native-thinking：12 个模型的 Bias 方程"),
        "index_accuracy": latex_equation_details(coefficients, metrics, "enumeration_index", "accuracy_bernoulli_logit", "Index enumeration：12 个模型的 Accuracy 方程"),
        "bullet_accuracy": latex_equation_details(coefficients, metrics, "enumeration_bullet", "accuracy_bernoulli_logit", "Bullet enumeration：12 个模型的 Accuracy 方程"),
        "index_mae": latex_equation_details(mae_coefficients, mae_metrics, "enumeration_index", MAE_FAMILY, "Index enumeration：12 个模型的 trimmed-MAE 方程"),
        "bullet_mae": latex_equation_details(mae_coefficients, mae_metrics, "enumeration_bullet", MAE_FAMILY, "Bullet enumeration：12 个模型的 trimmed-MAE 方程"),
    }
    registered_n = sorted(int(x) for x in count_error_cells["N"].unique())
    registered_l = sorted(int(x) for x in count_error_cells["L"].unique())
    mae_prediction_diagnostics = {}
    for mode in MODE_ORDER:
        raw_predictions = predictions_by_slot(
            mae_coefficients,
            mode,
            MAE_FAMILY,
            registered_n,
            registered_l,
        )
        mae_prediction_diagnostics[mode] = {
            "minimum": float(raw_predictions.min()),
            "negative_fraction": float(np.mean(raw_predictions < 0)),
            "negative_cells": int(np.sum(raw_predictions < 0)),
            "total_cells": int(raw_predictions.size),
        }

    bias_tail_table = bias_request_tails[
        bias_request_tails["prompt_mode"].isin(["direct", "native_thinking"])
    ].copy()
    bias_tail_table["Mode"] = bias_tail_table["prompt_mode"].map(MODE_SHORT)
    bias_tail_table["Max |raw request error|"] = bias_tail_table[
        "max_abs_raw_request_error"
    ].map(lambda x: num(x, 1))
    bias_tail_table["Max |error| retained"] = bias_tail_table[
        "max_abs_error_retained_after_cell_trimming"
    ].map(lambda x: num(x, 1))
    bias_tail_table["Median |raw−trimmed cell bias|"] = bias_tail_table[
        "median_abs_raw_vs_trimmed_cell_bias_change"
    ].map(lambda x: num(x, 3))
    bias_tail_table["Q95 |raw−trimmed|"] = bias_tail_table[
        "q95_abs_raw_vs_trimmed_cell_bias_change"
    ].map(lambda x: num(x, 3))
    bias_tail_table["Max |raw−trimmed|"] = bias_tail_table[
        "max_abs_raw_vs_trimmed_cell_bias_change"
    ].map(lambda x: num(x, 3))
    bias_tail_table["Pearson r"] = bias_tail_table[
        "pearson_raw_vs_trimmed_cell_bias"
    ].map(lambda x: num(x, 3))
    bias_tail_table = bias_tail_table[
        [
            "Mode",
            "Max |raw request error|",
            "Max |error| retained",
            "Median |raw−trimmed cell bias|",
            "Q95 |raw−trimmed|",
            "Max |raw−trimmed|",
            "Pearson r",
        ]
    ]

    bias_influence_table = bias_influence_summary[
        bias_influence_summary["prompt_mode"].isin(["direct", "native_thinking"])
    ].copy()
    bias_influence_table["Mode"] = bias_influence_table["prompt_mode"].map(MODE_SHORT)
    bias_influence_table["Cook-flagged cells"] = bias_influence_table.apply(
        lambda row: f"{int(row.influential_cells)}/{int(row.cells_evaluated)} "
        f"({100 * row.influential_cells / row.cells_evaluated:.1f}%)",
        axis=1,
    )
    bias_influence_table["Median curve r"] = bias_influence_table[
        "median_surface_correlation"
    ].map(lambda x: num(x, 3))
    bias_influence_table["Worst curve r"] = bias_influence_table[
        "minimum_surface_correlation"
    ].map(lambda x: num(x, 3))
    bias_influence_table["Median curve RMSE"] = bias_influence_table[
        "median_surface_rmse"
    ].map(lambda x: num(x, 3))
    bias_influence_table["Largest pointwise change"] = bias_influence_table[
        "maximum_surface_change"
    ].map(lambda x: num(x, 3))
    bias_influence_table = bias_influence_table[
        [
            "Mode",
            "Cook-flagged cells",
            "Median curve r",
            "Worst curve r",
            "Median curve RMSE",
            "Largest pointwise change",
        ]
    ]

    bias_influence_details = bias_influence[
        bias_influence["prompt_mode"].isin(["direct", "native_thinking"])
    ].copy()
    bias_influence_details["Model"] = bias_influence_details["comparison_slot"]
    bias_influence_details["Mode"] = bias_influence_details["prompt_mode"].map(MODE_SHORT)
    bias_influence_details["Flagged"] = bias_influence_details["influential_cells"].astype(int)
    bias_influence_details["Max Cook's D"] = bias_influence_details["max_cooks_d"].map(
        lambda x: num(x, 3)
    )
    bias_influence_details["Max-D cell"] = bias_influence_details.apply(
        lambda row: (
            f"N={int(row.max_cooks_N)}, L={int(row.max_cooks_L) // 1000}k"
            if pd.notna(row.max_cooks_N)
            else "not estimable"
        ),
        axis=1,
    )
    bias_influence_details["Refit curve r"] = bias_influence_details[
        "surface_correlation_after_dropping_influential"
    ].map(lambda x: num(x, 3))
    bias_influence_details["Max refit change"] = bias_influence_details[
        "surface_max_abs_change_after_dropping_influential"
    ].map(lambda x: num(x, 3))
    bias_influence_details = bias_influence_details[
        ["Model", "Mode", "Flagged", "Max Cook's D", "Max-D cell", "Refit curve r", "Max refit change"]
    ]

    concept_definitions = pd.DataFrame(
        [
            {
                "Concept": "Request",
                "Calculation / construction": "One comparison slot × prompt mode × N × L × random seed",
                "Meaning": "The atomic model call. There are 30 seeds for every registered slot–mode–N–L condition.",
            },
            {
                "Concept": "N × L cell",
                "Calculation / construction": "All requests sharing comparison slot, prompt mode, N, and L; normally n=30",
                "Meaning": "The unit used to compute a condition accuracy or signed-bias value.",
            },
            {
                "Concept": "Comparison slot",
                "Calculation / construction": "One behavior-comparison unit in the 12-panel grid",
                "Meaning": "Usually the same checkpoint across modes; † GLM/Ministral slots are registered instruct–reasoning checkpoint pairs.",
            },
            {
                "Concept": "Parsed exact accuracy",
                "Calculation / construction": "Σ 1[an integer is parsed and equals N] / number of all requests",
                "Meaning": "Parse failures remain in the denominator and count as incorrect.",
            },
            {
                "Concept": "Signed error",
                "Calculation / construction": "dᵢ = predicted countᵢ − Nᵢ",
                "Meaning": "Positive is over-counting; negative is under-counting; defined only when an integer is parseable.",
            },
            {
                "Concept": "10% trimmed signed bias",
                "Calculation / construction": "Sort m parseable d values, k=floor(0.10m), then average d(k+1)…d(m−k)",
                "Meaning": "Cell-level directional error after removing k values from each tail; a full m=30 cell averages the middle 24.",
            },
            {
                "Concept": "10% symmetrically trimmed conditional MAE",
                "Calculation / construction": "Let aᵢ=|predicted countᵢ−N| for m≥20 parseable responses; sort a, set k=floor(0.10m), and average a(k+1)…a(m−k)",
                "Meaning": "Unsigned error magnitude for the middle 80% of parseable absolute errors. A full m=30 cell removes the 3 smallest and 3 largest absolute errors, then averages the middle 24.",
            },
            {
                "Concept": "Lₖ, ln N, ln Lₖ, 1/N",
                "Calculation / construction": "Lₖ=L/1,000; ln is the natural logarithm; invN=1/N (well-defined here because every registered N>0)",
                "Meaning": "Registered transformations used by the finite 18-candidate law registry. The five inverse-count additions pair 1/N with Lₖ or ln Lₖ, with or without a hierarchical first-order interaction.",
            },
            {
                "Concept": "Basis / topology",
                "Calculation / construction": "The selected set φ(N,L) of main effects and any hierarchical interaction",
                "Meaning": "Shared within a prompt mode across 12 slots; intercepts and coefficients remain slot-specific.",
            },
            {
                "Concept": "Bernoulli-logit / inverse-logit",
                "Calculation / construction": "logit(p)=ln[p/(1−p)]; the fitted probability is σ(z)=1/(1+e^(−z))",
                "Meaning": "Accuracy is modeled on the log-odds scale and then mapped back to a probability between 0 and 1.",
            },
            {
                "Concept": "OLS with identity link",
                "Calculation / construction": "Choose β to minimize Σ(b−Xβ)²; the fitted bias is b̂=Xβ",
                "Meaning": "Bias remains in count units; no inverse transformation is applied.",
            },
            {
                "Concept": "Trimmed-MAE identity law",
                "Calculation / construction": "Fit tMAE₁₀=Xβ by OLS and report tMAE-hat=Xβ without clipping or transformation",
                "Meaning": "Coefficients are directly in count units for the middle-80% estimand. Negative fitted values are retained as diagnostics that identity OLS is only a local descriptive approximation.",
            },
            {
                "Concept": "First-order N×L interaction",
                "Calculation / construction": "Multiply the registered parent terms (for example N×Lₖ) and include both parent main effects",
                "Meaning": "The marginal effect of N may change with L, and the marginal effect of L may change with N.",
            },
            {
                "Concept": "Aggregated mode accuracy",
                "Calculation / construction": "Σ correct requests / Σ requests over the included comparison slots",
                "Meaning": "A request-weighted overall rate, not the unweighted mean of model percentages.",
            },
            {
                "Concept": "Post-hoc sensitivity",
                "Calculation / construction": "Rerun the frozen pipeline after changing only the cell estimand from the 10% trimmed mean to the untrimmed arithmetic mean",
                "Meaning": "A diagnostic robustness analysis defined after seeing the main result; it does not replace the preregistered estimand.",
            },
            {
                "Concept": "Bootstrap repetitions = 0",
                "Calculation / construction": "Generate no resampled pseudo-datasets and compute no bootstrap interval",
                "Meaning": "Model choice and stability are assessed by held-condition CV and LOMO rather than the discarded heavy-bootstrap design.",
            },
        ]
    )

    visual_definitions = pd.DataFrame(
        [
            {
                "Graphic element": "Median",
                "Calculation": "50th percentile across the 12 comparison-slot values at the same N and L",
                "Used for": "The central observed or fitted curve/surface in aggregate figures.",
            },
            {
                "Graphic element": "Q25 / Q75",
                "Calculation": "25th / 75th percentile across those same 12 slot values",
                "Used for": "The lower and upper endpoints of aggregate variability.",
            },
            {
                "Graphic element": "IQR",
                "Calculation": "IQR=Q75−Q25; plotted as the interval [Q25,Q75] around the median",
                "Used for": "Error bars for observed medians and shaded bands for fitted medians; it is model heterogeneity, not a confidence interval.",
            },
            {
                "Graphic element": "Observed marker",
                "Calculation": "Aggregate figure: median of 12 slot-level cell values; 3×4 figure: one slot-level N×L cell value",
                "Used for": "Hollow circles/triangles show data rather than fitted predictions.",
            },
            {
                "Graphic element": "Fitted line",
                "Calculation": "Evaluate the selected slot-specific regression equation at every displayed N and L",
                "Used for": "Solid/dashed curves; no smoothing is applied beyond the fitted law itself.",
            },
            {
                "Graphic element": "Yellow→purple curve color",
                "Calculation": "Equally sample a reversed Plasma color map in ascending order of the variable encoded by color",
                "Used for": "N-horizontal and 3×4 figures color all 8 L values; L-horizontal figures color all 14 N values. Yellow is the smallest value and dark purple the largest.",
            },
            {
                "Graphic element": "Log2-positioned N axis",
                "Calculation": "Horizontal position is proportional to log₂(N), while labels show the original N",
                "Used for": "Separates small N values without changing the N or ln N term used by the regression.",
            },
            {
                "Graphic element": "Log-positioned L axis",
                "Calculation": "Horizontal position is proportional to ln(L), while tick labels show the original length in thousands of tokens",
                "Used for": "Makes all 8 registered L values legible in the transposed Accuracy, Bias, and MAE views; it does not alter the fitted L term.",
            },
            {
                "Graphic element": "Percentage-point delta",
                "Calculation": "100 × (Native accuracy − Non-thinking accuracy)",
                "Used for": "Numbers above the paired markers in Figure 1.",
            },
            {
                "Graphic element": "Appendix-D 3D points and surface",
                "Calculation": "At each registered N×L coordinate, the point is the median observed cell value across 12 slots; the translucent surface is the pointwise median of 12 separately fitted slot-specific predictions",
                "Used for": "A perspective view of the same fitted laws. Surface color redundantly encodes z (yellow=low, purple=high); it is not a fourth variable and no pooled coefficient vector is fitted.",
            },
            {
                "Graphic element": "98th-percentile color cap",
                "Calculation": "Take the empirical 0.98 quantile of all aggregate observed |bias| pixels (linear interpolation between adjacent ordered values), then use max(0.5, that quantile)",
                "Used for": "The symmetric ± color limit in Figure 4; values beyond it saturate instead of stretching the scale.",
            },
        ]
    )

    metric_definitions = pd.DataFrame(
        [
            {
                "Quantity": "Five-fold held-condition CV / OOF prediction",
                "Calculation": "fold=(index(N)+index(L)) mod 5; fit on four folds and predict the fifth, so each out-of-fold (OOF) row is predicted by a fit that never used its N×L condition",
                "Meaning": "Tests interpolation to unseen registered conditions while preventing requests from one cell from leaking across train and validation.",
            },
            {
                "Quantity": "Accuracy CV log loss",
                "Calculation": "−mean[y log(p̂)+(1−y) log(1−p̂)] on concatenated out-of-fold requests",
                "Meaning": "Lower is better; directly scores probability predictions and penalizes confident errors.",
            },
            {
                "Quantity": "Accuracy CV D²",
                "Calculation": "1 − CV log loss(law) / CV log loss(intercept-only)",
                "Meaning": "1 is perfect; 0 matches the held-out constant baseline; negative is worse than the baseline.",
            },
            {
                "Quantity": "Bias CV R²",
                "Calculation": "1 − Σ(b−b̂)² / Σ(b−mean(b))² using concatenated out-of-fold N×L cells",
                "Meaning": "Fraction of held-condition cell-level bias variance explained; negative means worse than predicting the mean.",
            },
            {
                "Quantity": "Bias CV MAE",
                "Calculation": "mean(|b−b̂|) over concatenated out-of-fold N×L cells",
                "Meaning": "Prediction error in count units; remains interpretable when R² is unstable near zero variance.",
            },
            {
                "Quantity": "Conditional-MAE CV R² / CV MAE",
                "Calculation": "For each cell, first compute the 10% symmetrically trimmed conditional MAE; fit it with identity OLS, then compute R²=1−Σ(tMAE−tMAE-hat)²/Σ(tMAE−mean_train(tMAE))² and mean |tMAE−tMAE-hat| on held conditions",
                "Meaning": "R² measures held-condition variance explained; CV MAE is the typical prediction error in count units for the middle-80% estimand. Neither quantity is a confidence interval.",
            },
            {
                "Quantity": "Cook's D influence diagnostic",
                "Calculation": "For each slot-level selected bias OLS, Dᵢ=[eᵢ²/(p·MSE)]·hᵢᵢ/(1−hᵢᵢ)²; flag Dᵢ>4/n, then refit the same topology after dropping flagged cells",
                "Meaning": "Descriptive screen for cells that jointly have unusual residual and leverage. The 4/n rule is a heuristic, not an outlier hypothesis test; refit-curve change measures practical sensitivity.",
            },
            {
                "Quantity": "Median / Q25 across slots",
                "Calculation": "50th / 25th percentile of the 12 slot-specific CV scores",
                "Meaning": "Median is the typical model; Q25 reveals whether support extends beyond only the easiest models. Neither is a confidence interval.",
            },
            {
                "Quantity": "Near-best selection",
                "Calculation": "Keep candidates within 0.02 of the best median CV score, then within 0.05 of the best Q25 among those; break remaining ties by fewer predictors, lower median loss, higher median/Q25 score, then registry order",
                "Meaning": "Avoids selecting a more complex expression for a practically negligible validation gain.",
            },
            {
                "Quantity": "Frozen interaction gate",
                "Calculation": "Require median |standardized interaction| ≥0.10, BH-adjusted HC3 q<0.05 in ≥50% of slots, median CV-score gain over the no-interaction parent ≥0.02, and one-sided Wilcoxon gain BH q≤0.05",
                "Meaning": "An interaction is eligible only when it is practically nontrivial, recurrent across models, and improves held-condition prediction.",
            },
            {
                "Quantity": "Standardized coefficient effect / BH-adjusted HC3 q",
                "Calculation": "Accuracy: E=β·SD(x), measured in log-odds; bias: E=β·SD(x)/SD(b); trimmed conditional MAE: E=β·SD(x)/SD(tMAE₁₀). HC3 uses residual inflation 1/(1−hᵢᵢ)²; Benjamini–Hochberg adjusts the coefficient p-values within each registered family.",
                "Meaning": "|E| permits within-outcome comparison of differently scaled predictors. The report calls a term interpretively dominant only when its cross-slot median |E| is at least twice the runner-up; this does not delete smaller terms from prediction. HC3 is heteroskedasticity-robust and BH controls the false-discovery rate.",
            },
            {
                "Quantity": "LOMO structure stability",
                "Calculation": "(1/12) Σₛ 1[selected basis on 11 slots = selected basis on all 12]",
                "Meaning": "Whether the law topology survives removal of each model; it does not test coefficient transfer.",
            },
            {
                "Quantity": "LOMO held score",
                "Calculation": "medianₛ Qₛ(basis selected without slot s)",
                "Meaning": "The omitted model’s held-condition score under the topology selected by the other 11; coefficients are still estimated within the omitted model.",
            },
        ]
    )

    sensitivity_fit = sensitivity_comparison.copy()
    sensitivity_fit["Mode"] = sensitivity_fit["prompt_mode"].map(MODE_SHORT)
    sensitivity_fit["Trimmed law"] = sensitivity_fit["trimmed_selected_candidate"].map(FORMULA)
    sensitivity_fit["Trimmed CV R²"] = sensitivity_fit["trimmed_median_primary_score"].map(
        lambda x: num(x, 3)
    )
    sensitivity_fit["Trimmed LOMO"] = sensitivity_fit[
        "trimmed_lomo_formula_stability"
    ].map(lambda x: pct(x, 1))
    sensitivity_fit["Untrimmed law"] = sensitivity_fit[
        "untrimmed_selected_candidate"
    ].map(FORMULA)
    sensitivity_fit["Untrimmed CV R²"] = sensitivity_fit[
        "untrimmed_median_primary_score"
    ].map(lambda x: num(x, 3))
    sensitivity_fit["Untrimmed Q25"] = sensitivity_fit[
        "untrimmed_q25_primary_score"
    ].map(lambda x: num(x, 3))
    sensitivity_fit["Untrimmed LOMO"] = sensitivity_fit[
        "untrimmed_lomo_formula_stability"
    ].map(lambda x: pct(x, 1))
    sensitivity_fit["Untrimmed LOMO held R²"] = sensitivity_fit[
        "untrimmed_lomo_median_held_primary_score"
    ].map(lambda x: num(x, 3))
    sensitivity_fit["Reading"] = sensitivity_fit["untrimmed_evidence_reading"]
    sensitivity_fit = sensitivity_fit[
        [
            "Mode",
            "Trimmed law",
            "Trimmed CV R²",
            "Trimmed LOMO",
            "Untrimmed law",
            "Untrimmed CV R²",
            "Untrimmed Q25",
            "Untrimmed LOMO",
            "Untrimmed LOMO held R²",
            "Reading",
        ]
    ]

    sensitivity_zero = sensitivity_diagnostics.copy()
    sensitivity_zero["Mode"] = sensitivity_zero["prompt_mode"].map(MODE_SHORT)
    sensitivity_zero["Zero cells · trimmed"] = sensitivity_zero[
        "zero_fraction_trimmed"
    ].map(lambda x: pct(x, 1))
    sensitivity_zero["Zero cells · untrimmed"] = sensitivity_zero[
        "zero_fraction_untrimmed"
    ].map(lambda x: pct(x, 1))
    sensitivity_zero["Median |estimand change|"] = sensitivity_zero[
        "median_abs_estimand_change"
    ].map(lambda x: num(x, 3))
    sensitivity_zero["Trimmed–untrimmed correlation"] = sensitivity_zero[
        "correlation_trimmed_untrimmed"
    ].map(lambda x: num(x, 3))
    sensitivity_zero = sensitivity_zero[
        [
            "Mode",
            "Zero cells · trimmed",
            "Zero cells · untrimmed",
            "Median |estimand change|",
            "Trimmed–untrimmed correlation",
        ]
    ]

    untrimmed_term_medians = (
        sensitivity_coefficients.loc[
            sensitivity_coefficients["term"].isin(["N", "L_k", "N_x_L_k"])
        ]
        .groupby(["prompt_mode", "term"], observed=True)["estimate"]
        .median()
    )

    def untrimmed_term(mode: str, term: str) -> str:
        key = (mode, term)
        return num(untrimmed_term_medians.loc[key], 5) if key in untrimmed_term_medians.index else "—"

    untrimmed_coefficient_comparison = pd.DataFrame(
        [
            {
                "Mode": MODE_SHORT[mode],
                "median βN": untrimmed_term(mode, "N"),
                "median βL": untrimmed_term(mode, "L_k"),
                "median βNL": untrimmed_term(mode, "N_x_L_k"),
            }
            for mode in ["enumeration_index", "enumeration_bullet", "native_thinking"]
        ]
    )
    sensitivity_by_mode = sensitivity_comparison.set_index("prompt_mode")
    index_sensitivity = sensitivity_by_mode.loc["enumeration_index"]
    bullet_sensitivity = sensitivity_by_mode.loc["enumeration_bullet"]

    slot_table = summary[summary["prompt_mode"].isin(["direct", "native_thinking"])].pivot(index="comparison_slot", columns="prompt_mode", values="parsed_exact_accuracy").reindex(SLOT_ORDER).reset_index()
    slot_table.columns = ["比较槽", "Non-thinking accuracy", "Native-thinking accuracy"]
    slot_table["差值"] = slot_table["Native-thinking accuracy"] - slot_table["Non-thinking accuracy"]
    for column in ["Non-thinking accuracy", "Native-thinking accuracy", "差值"]:
        slot_table[column] = slot_table[column].map(lambda x: pct(x, 1))
    slot_table["可比性"] = slot_table["比较槽"].map(lambda x: "配对 checkpoint；含 checkpoint 混杂" if x in MATCHED_PAIR_SLOTS else "同一 checkpoint；提示模式切换")

    revision_rows = []
    for model, revision in v31["model_revisions"].items():
        revision_rows.append({"物理模型": model, "固定 revision": revision})
    revisions = pd.DataFrame(revision_rows)

    zero_stats = []
    for mode in MODE_ORDER:
        block = cells[cells["prompt_mode"].eq(mode)]
        per_slot = block.groupby("comparison_slot")["trimmed_signed_bias_10"].apply(lambda s: float(np.mean(np.isclose(s, 0.0, atol=1e-12))))
        zero_stats.append({"模式": MODE_SHORT[mode], "单元零偏差比例中位数": pct(per_slot.median(), 1), "单元偏差 SD 中位数": num(block.groupby("comparison_slot")["trimmed_signed_bias_10"].std().median(), 3), "绝对偏差中位数": num(block.groupby("comparison_slot")["trimmed_signed_bias_10"].apply(lambda s: np.mean(np.abs(s))).median(), 3)})
    zero_table = pd.DataFrame(zero_stats)

    bb_summary = beta_metrics.groupby("prompt_mode", as_index=False).agg(median_rho=("rho", "median"), median_kappa=("kappa", "median"), converged=("converged", "sum"))
    bb_summary["模式"] = bb_summary["prompt_mode"].map(MODE_SHORT)
    bb_summary["median ρ"] = bb_summary["median_rho"].map(lambda x: num(x, 4))
    bb_summary["median κ"] = bb_summary["median_kappa"].map(lambda x: num(x, 2))
    bb_summary["收敛槽"] = bb_summary["converged"].map(lambda x: f"{int(x)}/12")
    bb_table = bb_summary[["模式", "median ρ", "median κ", "收敛槽"]]

    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    css = f"""
    :root{{--indigo:{AURORA['indigo']};--violet:{AURORA['violet']};--cyan:{AURORA['cyan']};--yellow:{AURORA['yellow']};--teal:{AURORA['teal']};--green:{AURORA['green']};--magenta:{AURORA['magenta']};--pink:{AURORA['pink']};--ink:{AURORA['black']};--paper:{AURORA['white']};--gray:{AURORA['gray']};--brown:{AURORA['brown']};--line:#D7DEE8;--soft:#EDF2F8;--shadow:0 14px 40px rgba(35,22,92,.08)}}
    *{{box-sizing:border-box}}html{{scroll-behavior:smooth;max-width:100%;overflow-x:hidden}}body{{margin:0;max-width:100%;overflow-x:hidden;background:var(--paper);color:var(--ink);font-family:Inter,"Segoe UI","Noto Sans SC","Microsoft YaHei",sans-serif;line-height:1.72;font-size:17px}}a{{color:var(--indigo)}}.page{{max-width:1240px;margin:auto;padding:0 34px 100px}}.hero{{position:relative;overflow:hidden;background:var(--ink);color:white;min-height:690px;border-bottom:1px solid rgba(255,255,255,.12)}}.hero:before{{content:"";position:absolute;inset:-20% -5% auto auto;width:72vw;height:72vw;max-width:980px;max-height:980px;background:radial-gradient(circle at 45% 45%,rgba(0,194,255,.78),transparent 27%),radial-gradient(circle at 63% 38%,rgba(255,95,162,.62),transparent 25%),radial-gradient(circle at 54% 64%,rgba(103,80,232,.85),transparent 36%);filter:blur(28px);opacity:.88;transform:rotate(-12deg)}}.hero .page{{position:relative;z-index:1;padding-top:46px;padding-bottom:72px}}.brand{{display:flex;justify-content:space-between;align-items:center;font-size:.76rem;letter-spacing:.12em;text-transform:uppercase;color:#D5DDEA}}.hero-grid{{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(300px,.6fr);gap:60px;align-items:end;padding-top:110px}}.kicker{{color:var(--yellow);font-size:.82rem;text-transform:uppercase;letter-spacing:.14em;font-weight:700}}h1{{font-family:Georgia,"Noto Serif SC",serif;font-weight:500;font-size:clamp(3.4rem,7.5vw,7.1rem);line-height:.93;letter-spacing:-.055em;margin:.26em 0 .3em;max-width:900px}}.dek{{max-width:790px;font-size:clamp(1.1rem,2vw,1.38rem);line-height:1.48;color:#E8EEF6}}.hero-aside{{border-top:1px solid rgba(255,255,255,.45);padding-top:22px;color:#E3E8F2;font-size:.92rem}}.hero-aside strong{{display:block;color:var(--yellow);font-size:2.7rem;line-height:1;margin-bottom:12px}}.meta{{display:flex;flex-wrap:wrap;gap:18px;margin-top:38px;color:#B9C4D4;font-size:.78rem}}.nav{{position:sticky;top:0;z-index:10;background:rgba(248,251,255,.94);backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}}.nav .page{{display:flex;gap:24px;align-items:center;overflow:auto;padding-top:12px;padding-bottom:12px;white-space:nowrap;font-size:.79rem}}.nav a{{text-decoration:none;color:#435169}}.nav strong{{color:var(--indigo)}}main.page{{padding-top:48px}}section{{padding:66px 0 34px;border-bottom:1px solid var(--line)}}.section-head{{display:grid;grid-template-columns:180px minmax(0,780px);gap:32px;margin-bottom:32px}}.section-no{{font-size:.76rem;color:var(--violet);font-weight:800;letter-spacing:.12em;text-transform:uppercase;padding-top:11px}}h2{{font-family:Georgia,"Noto Serif SC",serif;font-size:clamp(2.25rem,4.3vw,4.25rem);font-weight:500;line-height:1.05;letter-spacing:-.035em;margin:0 0 18px}}h3{{font-size:1.28rem;margin:46px 0 12px;letter-spacing:-.015em}}.lede{{font-size:1.21rem;line-height:1.55;color:#354258;max-width:820px}}p,ul,ol{{max-width:830px}}.wide{{max-width:none}}.key-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin:30px 0 18px}}.key{{background:var(--paper);padding:26px 22px;min-height:130px}}.key .value{{font-size:2.25rem;font-weight:700;letter-spacing:-.04em;color:var(--indigo)}}.key .label{{font-size:.82rem;color:#5C687A;margin-top:4px}}.pullquote{{font-family:Georgia,"Noto Serif SC",serif;font-size:clamp(1.75rem,3vw,2.65rem);line-height:1.24;max-width:970px;margin:42px 0;padding-left:26px;border-left:6px solid var(--yellow)}}.conclusion{{max-width:930px;margin:38px 0 8px;padding:20px 24px;background:linear-gradient(110deg,rgba(0,194,255,.10),rgba(255,95,162,.08));border-left:4px solid var(--indigo);font-size:.96rem}}.conclusion strong{{color:var(--indigo)}}figure{{margin:40px 0 22px;border-top:1px solid var(--line);padding-top:18px}}figure img{{display:block;width:100%;height:auto;background:var(--paper)}}figcaption{{display:grid;grid-template-columns:95px minmax(0,820px);gap:20px;color:#526075;font-size:.88rem;line-height:1.58;margin-top:16px}}figcaption strong{{color:var(--indigo);letter-spacing:.04em}}.formula-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:24px 0}}.formula{{background:var(--ink);color:#EAF0F8;padding:24px;border-radius:2px;min-height:145px}}.formula .tag{{color:var(--yellow);font-size:.76rem;text-transform:uppercase;letter-spacing:.12em}}.formula code{{display:block;background:none;color:white;font-size:1.18rem;margin:16px 0 8px;white-space:normal}}.formula small{{color:#AEBACA}}.note{{max-width:900px;background:#FFFBE0;border:1px solid #E8D86F;padding:18px 22px;margin:22px 0}}.table-wrap{{overflow:auto;border-top:2px solid var(--ink);border-bottom:1px solid var(--line);margin:26px 0;max-width:100%}}.data-table{{border-collapse:collapse;width:100%;font-size:.78rem}}.data-table th{{text-align:left;color:#37445A;background:#EEF3F8;font-weight:700}}.data-table th,.data-table td{{padding:11px 12px;border-bottom:1px solid #E1E7EF;vertical-align:top;white-space:nowrap}}.data-table tr:last-child td{{border-bottom:none}}details{{border-top:1px solid var(--line);padding:17px 0}}summary{{cursor:pointer;font-weight:700;color:var(--indigo)}}.methods{{font-family:"Cascadia Code",Consolas,monospace;font-size:.81rem;white-space:pre-wrap;background:#EDF2F7;padding:18px;overflow:auto;max-width:1000px}}.appendix{{background:#F1F4F9;margin-left:calc(50% - 50vw);margin-right:calc(50% - 50vw);padding-left:max(34px,calc((100vw - 1172px)/2));padding-right:max(34px,calc((100vw - 1172px)/2))}}.footer{{padding:48px 0;color:#607087;font-size:.8rem}}.legend-dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}@media(max-width:800px){{body{{font-size:15px}}.page{{padding-left:20px;padding-right:20px}}.hero{{min-height:auto}}.hero-grid{{grid-template-columns:1fr;padding-top:72px}}.hero-aside{{max-width:330px}}.section-head{{grid-template-columns:1fr;gap:5px}}.key-grid{{grid-template-columns:repeat(2,1fr)}}.formula-grid{{grid-template-columns:1fr}}figcaption{{grid-template-columns:1fr;gap:3px}}.appendix{{padding-left:20px;padding-right:20px}}}}@media print{{.nav{{display:none}}.hero{{min-height:auto;background:white;color:var(--ink);border-bottom:2px solid var(--ink)}}.hero:before{{display:none}}.hero .page{{padding-top:20px;padding-bottom:30px}}.hero .dek,.hero-aside,.brand,.meta{{color:var(--ink)}}section{{break-inside:auto}}figure,.formula,.key-grid{{break-inside:avoid}}.appendix{{background:white}}}}
    """

    # Match the established NiaH_Non-thinking_report.html rather than using a
    # standalone blog/launch-page treatment.  This second assignment is
    # intentional: it leaves the earlier palette definition available to old
    # build manifests while making the emitted artifact visually conservative.
    css = """
    :root { --ink:#161923; --muted:#5f6b7a; --line:#d8dee9; --paper:#F8FBFF; --wash:#f4f6f8;
      --indigo:#23165C; --violet:#6750E8; --cyan:#00C2FF; --yellow:#F6E36A;
      --teal:#00D4B4; --green:#39E58C; --magenta:#C04DFF; --pink:#FF5FA2;
      --gray:#8190A5; --brown:#765347; }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; max-width:100%; overflow-x:hidden; }
    body { margin:0; color:var(--ink); background:#eef1f5; font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.68; }
    a { color:var(--indigo); text-decoration:none; }
    .page { width:min(1180px,calc(100% - 32px)); margin-left:auto; margin-right:auto; }
    .hero { width:min(1180px,calc(100% - 32px)); margin:24px auto 0; background:var(--paper); box-shadow:0 18px 50px rgba(23,32,51,.10); }
    .hero .page { width:100%; padding:48px 64px 34px; background:#f8fafc; border-bottom:1px solid var(--line); }
    .brand { display:flex; justify-content:space-between; gap:20px; color:var(--muted); font-size:12px; letter-spacing:.08em; text-transform:uppercase; }
    .hero-grid { display:block; padding-top:34px; }
    .kicker,.section-no { color:var(--indigo); font-size:12px; font-weight:750; letter-spacing:.10em; text-transform:uppercase; }
    h1 { max-width:900px; margin:7px 0 0; font-family:Georgia,"Noto Serif SC",serif; font-size:clamp(34px,5vw,54px); line-height:1.12; letter-spacing:-.02em; font-weight:600; }
    .dek { max-width:900px; margin:18px 0 0; color:#3c4858; font-size:18px; line-height:1.58; }
    .meta { display:flex; gap:18px; flex-wrap:wrap; margin-top:22px; color:var(--muted); font-size:13px; }
    .hero-aside { display:flex; align-items:baseline; gap:14px; max-width:900px; margin:24px 0 0; padding:15px 18px; border-left:4px solid var(--teal); background:rgba(0,212,180,.08); color:#344054; font-size:14px; }
    .hero-aside strong { flex:0 0 auto; color:var(--indigo); font:700 27px ui-monospace,SFMono-Regular,Consolas,monospace; }
    .nav { position:sticky; top:0; z-index:20; width:min(1180px,calc(100% - 32px)); margin:0 auto; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); box-shadow:0 8px 20px rgba(23,32,51,.05); backdrop-filter:blur(10px); }
    .nav .page { width:100%; display:flex; gap:20px; padding:11px 64px; overflow-x:auto; white-space:nowrap; font-size:13px; }
    .nav a { color:#475467; }
    .nav strong { color:var(--indigo); }
    main.page { display:flex; flex-direction:column; margin-top:0; margin-bottom:72px; padding:0 64px 64px; background:var(--paper); box-shadow:0 18px 50px rgba(23,32,51,.10); }
    #design { order:1; } #accuracy { order:2; } #mae { order:3; } #appendix-a { order:4; }
    #appendix-b { order:5; } #appendix-c { order:6; } #appendix-d { order:7; } #repro { order:8; } .footer { order:9; }
    section { padding:50px 0 24px; border-bottom:1px solid var(--line); }
    section:last-of-type { border-bottom:0; }
    .section-head { display:grid; grid-template-columns:135px minmax(0,820px); gap:24px; margin-bottom:24px; }
    .section-no { padding-top:9px; }
    h2 { margin:0 0 14px; font-family:Georgia,"Noto Serif SC",serif; font-size:32px; line-height:1.25; font-weight:600; letter-spacing:-.012em; }
    h3 { margin:32px 0 10px; font-size:20px; line-height:1.35; }
    p { margin:10px 0; }
    p,ul,ol { max-width:920px; }
    .lede { max-width:900px; color:#344054; font-size:17px; }
    .wide { max-width:none; }
    .key-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:24px 0; }
    .key { min-height:112px; padding:17px 18px; border:1px solid var(--line); background:#fff; }
    .key .value { color:var(--indigo); font:700 25px ui-monospace,SFMono-Regular,Consolas,monospace; letter-spacing:-.035em; }
    .key .label { margin-top:6px; color:#596579; font-size:12px; line-height:1.45; }
    .pullquote { max-width:930px; margin:24px 0; padding:16px 19px; border-left:4px solid var(--cyan); background:rgba(0,194,255,.07); color:#2f3b4d; font-family:Georgia,"Noto Serif SC",serif; font-size:18px; line-height:1.55; }
    .conclusion { max-width:930px; margin:24px 0 8px; padding:15px 18px; border-left:4px solid var(--teal); background:rgba(0,212,180,.08); color:#23413f; font-size:14px; }
    .conclusion strong { color:var(--indigo); }
    figure { margin:30px 0 36px; }
    figure img { display:block; width:100%; height:auto; border:1px solid var(--line); background:#fff; }
    figcaption { display:grid; grid-template-columns:92px minmax(0,900px); gap:18px; margin:10px auto 0; color:#586579; font-size:13px; line-height:1.55; }
    figcaption strong { color:var(--indigo); letter-spacing:.04em; }
    .d3-shell { position:relative; min-height:300px; border:1px solid var(--line); background:#fff; }
    .d3-explorer { display:none; }
    .d3-shell.is-ready .d3-explorer { display:block; }
    .d3-shell.is-ready .interactive-fallback { display:none; }
    .interactive-fallback { display:block; width:100%; height:auto; border:0; }
    .d3-toolbar { display:flex; align-items:stretch; justify-content:space-between; gap:14px; padding:14px 16px; border-bottom:1px solid var(--line); background:#f8fafc; }
    .d3-metrics { display:flex; flex:1 1 auto; gap:7px; }
    .d3-metric,.d3-reset { appearance:none; border:1px solid #ccd4df; background:#fff; color:#475467; cursor:pointer; font:650 12px/1.25 Inter,ui-sans-serif,sans-serif; }
    .d3-metric { min-width:0; padding:9px 13px; }
    .d3-metric span { display:block; margin-top:2px; color:#7a8494; font-size:10px; font-weight:500; }
    .d3-metric:hover,.d3-reset:hover { border-color:var(--violet); color:var(--indigo); }
    .d3-metric:focus-visible,.d3-reset:focus-visible { outline:3px solid rgba(0,194,255,.35); outline-offset:2px; }
    .d3-metric.is-active { border-color:var(--indigo); background:var(--indigo); color:#fff; box-shadow:inset 0 -3px 0 var(--yellow); }
    .d3-metric.is-active span { color:#d9d7ed; }
    .d3-reset { flex:0 0 auto; padding:9px 13px; }
    .d3-status { display:flex; justify-content:space-between; gap:18px; padding:9px 16px; border-bottom:1px solid #e5e9f0; color:#667085; font-size:11px; }
    .d3-grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); }
    .d3-panel { min-width:0; padding:12px 8px 5px; }
    .d3-panel + .d3-panel { border-left:1px solid var(--line); }
    .d3-panel-head { min-height:54px; padding:0 12px; }
    .d3-panel-head strong { display:block; color:var(--ink); font-size:14px; }
    .d3-panel-head span { display:block; margin-top:2px; color:#667085; font:500 11px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace; }
    .d3-plot { width:100%; height:520px; }
    .formula-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:20px 0; }
    .formula { min-height:132px; padding:18px; border:1px solid var(--line); background:#f8fafc; }
    .formula .tag { color:var(--indigo); font-size:11px; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
    .formula code { display:block; margin:13px 0 7px; color:#203251; background:none; font-family:"Iowan Old Style",Cambria,Georgia,serif; font-size:17px; white-space:normal; }
    .math-block { max-width:930px; margin:17px 0; padding:14px 18px; overflow-x:auto; border-left:3px solid var(--violet); background:#f7f7fb; color:#20243a; }
    .math-block mjx-container[display="true"] { display:block; max-width:100%; overflow-x:auto; overflow-y:hidden; padding-bottom:3px; }
    .derivation { max-width:930px; padding:2px 0 2px 18px; border-left:2px solid #d9d4f5; }
    .warning { max-width:930px; margin:20px 0; padding:16px 18px; border-left:4px solid var(--pink); background:rgba(255,95,162,.07); color:#513447; }
    .formula small { color:#667085; }
    .law-metrics { display:flex; flex-wrap:wrap; gap:7px; margin-top:13px; }
    .law-metrics span { padding:4px 7px; border:1px solid #d8dee9; background:#fff; color:#475467; font:600 11px ui-monospace,SFMono-Regular,Consolas,monospace; }
    .law-reading { margin-top:11px; color:#475467; font-size:13px; line-height:1.55; }
    .evidence-order { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); margin:20px 0 26px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
    .evidence-step { padding:15px 17px; border-right:1px solid var(--line); }
    .evidence-step:last-child { border-right:0; }
    .evidence-step strong { display:block; margin-bottom:4px; color:var(--ink); font-size:13px; }
    .evidence-step span { color:#667085; font-size:12px; line-height:1.5; }
    .note { max-width:930px; margin:20px 0; padding:16px 18px; border-left:4px solid var(--yellow); background:rgba(246,227,106,.17); color:#5b4930; }
    .table-wrap { max-width:100%; margin:22px 0; overflow:auto; border-top:2px solid var(--ink); border-bottom:1px solid var(--line); }
    .data-table { width:100%; border-collapse:collapse; font-size:12px; line-height:1.45; }
    .data-table th { text-align:left; color:#475467; background:#f8fafc; font-weight:650; }
    .data-table th,.data-table td { padding:9px 10px; border-bottom:1px solid #e5e9f0; vertical-align:top; white-space:nowrap; }
    .definition-table { table-layout:fixed; min-width:760px; }
    .definition-table th,.definition-table td { white-space:normal; line-height:1.48; }
    .definition-table th:first-child,.definition-table td:first-child { width:18%; }
    .definition-table th:nth-child(2),.definition-table td:nth-child(2) { width:43%; }
    .definition-table th:nth-child(3),.definition-table td:nth-child(3) { width:39%; }
    .equation-table { min-width:900px; }
    .equation-table td:nth-child(2) { color:#203251; font:12px/1.55 "Cascadia Code",Consolas,monospace; }
    details.equations { margin:14px 0; padding:15px 17px; border:1px solid var(--line); background:#fff; }
    details.equations > summary { display:flex; justify-content:space-between; gap:18px; }
    details.equations > summary::after { content:"expand"; color:#667085; font:500 11px ui-monospace,SFMono-Regular,Consolas,monospace; text-transform:uppercase; letter-spacing:.06em; }
    details.equations[open] > summary::after { content:"collapse"; }
    details { border-top:1px solid var(--line); padding:15px 0; }
    summary { cursor:pointer; color:var(--indigo); font-weight:700; }
    .methods { max-width:1000px; padding:16px; overflow:auto; background:#edf2f7; font:12px/1.55 "Cascadia Code",Consolas,monospace; white-space:pre-wrap; }
    .appendix { margin-left:-64px; margin-right:-64px; padding-left:64px; padding-right:64px; background:#f8fafc; }
    .footer { padding:42px 0 0; color:#607087; font-size:12px; }
    .legend-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
    @media(max-width:800px) {
      .hero .page,main.page { padding-left:28px; padding-right:28px; }
      .nav .page { padding-left:28px; }
      .section-head { grid-template-columns:1fr; gap:5px; }
      .key-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .formula-grid { grid-template-columns:1fr; }
      .evidence-order { grid-template-columns:1fr; }
      .evidence-step { border-right:0; border-bottom:1px solid var(--line); }
      .evidence-step:last-child { border-bottom:0; }
      figcaption { grid-template-columns:1fr; gap:3px; }
      .d3-toolbar { align-items:stretch; flex-direction:column; }
      .d3-metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); }
      .d3-metric { padding:9px 6px; }
      .d3-status { flex-direction:column; gap:2px; }
      .d3-grid { grid-template-columns:1fr; }
      .d3-panel + .d3-panel { border-left:0; border-top:1px solid var(--line); }
      .d3-plot { height:450px; }
      .appendix { margin-left:-28px; margin-right:-28px; padding-left:28px; padding-right:28px; }
      .hero-aside { align-items:flex-start; flex-direction:column; gap:4px; }
    }
    @media(max-width:560px) {
      .hero,.nav,.page { width:100%; }
      .hero { margin-top:0; }
      .hero .page,main.page { padding-left:18px; padding-right:18px; }
      .nav .page { padding-left:18px; }
      .key-grid { grid-template-columns:1fr 1fr; }
      h1 { font-size:34px; }
      h2 { font-size:27px; }
      main.page,section,.formula-grid,.formula,.math-block { min-width:0; max-width:100%; }
      .d3-metric span { display:none; }
      .d3-plot { height:410px; }
      .appendix { margin-left:-18px; margin-right:-18px; padding-left:18px; padding-right:18px; }
    }
    @media print { body { background:#fff; } .hero,.nav,main.page { width:100%; margin:0; box-shadow:none; } .nav { display:none; } section,figure { break-inside:avoid; } .d3-explorer { display:none!important; } .interactive-fallback { display:block!important; } }
    """

    report = rf"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>NiaH Empirical-law Report · V3.2 + 1/N audit</title><style>{css}</style></head>
<body><header class="hero"><div class="page"><div class="brand"><span>NiaH Empirical-law Report</span><span>V3.2 · 1/N candidate audit</span></div><div class="hero-grid"><div><div class="kicker">Behavioral Study: Distinct Error Laws</div><h1>Non-thinking and Native-thinking show distinct error geometry.</h1><p class="dek">本报告只保留能够支撑论文主张的证据：两种推理模式的行为差异，以及 Accuracy、10% 截尾有符号偏差与 conditional MAE 如何随目标数 N 和上下文长度 L 系统变化。两种显式 enumeration 作为控制条件放在附录。冻结的 13 候选 V3.2 结果保持不变；另行加入 5 个含 1/N 的候选并按同一规则重选。</p><div class="meta"><span>161,280 requests</span><span>14 fixed model revisions</span><span>12 comparison slots</span><span>18 candidate laws</span><span>14 N × 8 L × 30 seeds</span><span>commit {html.escape(audit['git_commit'][:12])}</span></div></div><aside class="hero-aside"><strong>{pct(raw_delta,1)}</strong><span>10 个同 checkpoint 槽中，Native-thinking 相对 Non-thinking 的聚合 parsed exact accuracy 增量。该差值描述总体表现；经验律进一步刻画表现怎样随 N 与 L 改变。</span></aside></div></div></header>
<nav class="nav"><div class="page"><strong>目录</strong><a href="#finding">1. Behavior</a><a href="#design">2. Meaning & methods</a><a href="#accuracy">3. Accuracy law</a><a href="#bias">4. Bias law</a><a href="#mae">5. MAE law</a><a href="#appendix">Appendix</a><a href="#repro">Reproducibility</a></div></nav>
<main class="page">
<section id="finding"><div class="section-head"><div class="section-no">01 · Behavior</div><div><h2>Thinking improves accuracy, but not by a constant offset.</h2><p class="lede">Native-thinking 在所有 12 个比较槽上都有更高的描述性准确率。更重要的是，提升量随模型而变；因此后续不能只报告一个平均百分点，而要研究 N–L 难度曲面的形状。</p></div></div>
<div class="key-grid"><div class="key"><div class="value">{pct(direct_raw.accuracy,1)}</div><div class="label">Non-thinking accuracy<br>10 个同 checkpoint 槽</div></div><div class="key"><div class="value">{pct(native_raw.accuracy,1)}</div><div class="label">Native-thinking accuracy<br>10 个同 checkpoint 槽</div></div><div class="key"><div class="value">{num(direct_acc.median_primary_score,2)}</div><div class="label">Non-thinking accuracy law<br>median held-condition D²</div></div><div class="key"><div class="value">{num(native_acc.median_primary_score,2)}</div><div class="label">Native-thinking accuracy law<br>median held-condition D²</div></div></div>
<figure><img src="{figures['accuracy_dumbbell']}" alt="12 个模型比较槽的 Non-thinking 与 Native-thinking 准确率横向窄条图"><figcaption><strong>FIGURE 1</strong><span><b>模型层面的行为差异。</b> 横轴依次列出 12 个模型比较槽，纵轴是 parsed exact accuracy=<code>Σ1[解析整数=N]/3,360</code>；解析失败计错。Scaling Yellow 圆点为 Non-thinking，Scaling Purple 三角为 Native-thinking，连线只标识同一比较槽，不表示连续轨迹。顶部数字按 <code>100×(Native accuracy−Non-thinking accuracy)</code> 计算，单位为 percentage points。† 表示 GLM/Ministral 的 instruct–reasoning matched checkpoints，因此同时包含 checkpoint 差异。</span></figcaption></figure>
<div class="conclusion"><strong>本段结论：</strong>Native-thinking 的总体优势是真实且跨模型可见的，但增益不是统一常数。需要经验律来回答更精细的问题：当 N 或 L 增加时，两种 mode 的性能以什么速率退化。</div></section>

<section id="design"><div class="section-head"><div class="section-no">02 · Meaning & methods</div><div><h2>What an empirical law means here.</h2><p class="lede">这里的 law 不是“参数量越大性能越高”的模型尺寸 scaling law，而是行为响应面：把每个模型在 112 个 N×L 条件上的误差曲面压缩为少数可解释系数，并检验它能否预测被整块留出的 N×L 条件。</p></div></div>
<div class="formula"><div class="tag">General model-resolved empirical law</div><code>gₘ(Yₛₘ(N,L)) = αₛₘ + θₛₘᵀ φₘ(N,L)</code><p class="law-reading"><b>s</b> 表示模型 comparison slot，<b>m</b> 表示 prompt mode，<b>φₘ</b> 是该 mode 跨 12 个槽共享的自变量 basis；截距 α 与系数向量 θ 对每个模型单独估计。Accuracy 的 g 是 logit，Bias 的 g 是 identity，conditional MAE 的 g 是 ln(1+MAE)。</p></div>
<p>这个表达式区分两层“规律”。第一层是 <b>topology</b>：N、L、ln N、ln L 或交互项中，哪些进入表达式；它回答难度曲面的几何形状。第二层是 <b>strength</b>：每个模型的具体系数；它回答同一形状在该模型上有多强。因此“同一种 law”只表示 basis 相同，不表示所有模型或两种 mode 共用系数。</p>
<h3>Notation and analysis units</h3><div class="table-wrap">{table_html(concept_definitions, 'data-table definition-table')}</div>
<h3>How every visual quantity is calculated</h3><div class="table-wrap">{table_html(visual_definitions, 'data-table definition-table')}</div>
<h3>Three distinct estimands</h3><p><b>Estimand</b> 指分析希望估计的目标量，而不是使用的拟合算法。这里分别估计“请求完全正确的概率”“可解析错误的方向性均值”和“可解析错误的绝对幅度”，三者不能互相替代。</p><div class="formula-grid"><div class="formula"><div class="tag">Accuracy · request level</div><code>yᵢ = 1[parsed countᵢ = Nᵢ]</code><p class="law-reading">解析失败计作 0；Bernoulli-logit 直接估计完全正确的概率。</p></div><div class="formula"><div class="tag">Bias · N×L cell level</div><code>b₁₀ = 10%-trimmed mean(predicted count − N)</code><p class="law-reading">只用可解析响应；将误差排序后两端各删去 floor(0.10m) 个值，m≥20；正值为多计，负值为少计。</p></div><div class="formula"><div class="tag">Conditional MAE · N×L cell level</div><code>MAE = m⁻¹ Σ |predicted count − N|</code><p class="law-reading">对 m≥20 个可解析响应取绝对误差均值；不截尾，因此保留每一个可解析的极端计数误差。</p></div></div>
<p>Accuracy、bias 与 conditional MAE 是三个 estimand。Accuracy 询问“是否完全正确”；bias 询问“剩余错误倾向于少计还是多计”；MAE 询问“可解析错误平均偏离多少”，但不区分方向且不包括解析失败。因此 Native-thinking 可以具有很高 accuracy，同时在少量剩余错误上仍呈现可拟合的方向性与幅度结构。正文严格采用 V3.1 预注册的 10% 截尾 bias；附录的未截尾 bias 只作 post-hoc sensitivity，不替换 confirmatory estimand。</p>
<h3>Selection and validation</h3><p>候选集包含冻结 V3.2 的 13 个结构，以及 5 个 post-freeze inverse-count 扩展：<code>1/N</code>、<code>1/N+Lₖ</code>、<code>1/N+ln Lₖ</code> 和各自满足层级原则的一阶交互，共 18 个候选。五折 held-condition CV 用 <code>(index(N)+index(L)) mod 5</code> 将完整 N×L 条件整块分到五折。每次用四折拟合、对第五折预测，最后拼接所有 out-of-fold（OOF）预测再计算分数；OOF 指每一行都由一个未使用该行所属 N×L 条件的模型产生预测。结构选择按 12 个模型槽的中位分数、Q25、预测损失、复杂度和同一交互门限完成；Bootstrap repetitions = 0 表示没有生成重采样伪数据集，也没有 bootstrap 区间。</p>
<p class="note"><strong>1/N candidate audit：</strong>新增项只扩展函数族，没有改变数据、estimand、fold、near-best rule、interaction gate 或 LOMO。Accuracy 三种 link、10% trimmed bias 与 conditional MAE 共 {len(inverse_selection_comparison)} 个 mode-family 选择中，selection change={int(inverse_selection_comparison['selection_changed'].sum())}，最终选择以 1/N 开头的结构={int(inverse_selection_comparison['selected_is_inverse_count'].sum())}。需要区分“原始中位分数最高”和“通过预定门限后被选择”：Non-thinking logit 中，最佳 inverse interaction <code>{FORMULA[direct_best_inverse_raw.candidate]}</code> 的 median D²={num(direct_best_inverse_raw.median_primary_score,3)}，但相对其无交互 parent 的 median gain={num(direct_best_inverse_raw.median_cv_score_gain_over_parent,3)}，未达到 0.02 interaction gate；最佳可选 inverse main-effect 结构 <code>{FORMULA[direct_best_inverse_eligible.candidate]}</code> 的 median D²={num(direct_best_inverse_eligible.median_primary_score,3)}，仍低于最终 <code>{FORMULA[direct_acc.selected_candidate]}</code>。因此当前 headline law 对该非线性小-N 候选稳健；这是一项 post-freeze specification audit，而不是把扩展伪装成预注册候选。</p>
<p><b>LOMO（leave-one-model-out）检验结构而不是零样本预测。</b> 对每个比较槽 s，将它完全排除，只用其余 11 个槽重新选择 basis；重复 12 次。LOMO structure stability 是这 12 次所选 basis 与全模型 basis 相同的比例。LOMO held score 则读取被留出槽在该 basis 下的五折 held-condition 分数；该槽的回归系数仍由它自己的训练折估计，所以它回答“结构能否跨模型迁移”，不回答“能否不看新模型数据直接预测其系数”。</p>
<div class="table-wrap">{table_html(metric_definitions, 'data-table definition-table')}</div>
<p class="note"><strong>读数原则：</strong>D²/R²=1 表示完美 held-condition 预测，0 表示不优于常数基线，负值表示比基线更差。Median 与 Q25 是 12 个模型槽的分布摘要，不是置信区间；LOMO stability 高说明 topology 不由单个模型决定，但并不保证每个模型都具有高拟合度。</p>
<div class="note"><strong>解释边界：</strong>经验律证明的是冻结网格内的低维描述与 held-condition 预测，不是内部算法的因果识别，也不是 N/L 网格外的外推定律。GLM/Ministral 的 † 槽还同时改变 checkpoint。</div>
<div class="conclusion"><strong>本段结论：</strong>经验律的科学意义是把“thinking 更好”推进为可检验的几何陈述：哪一种 N–L basis 能跨模型复现、每个模型的系数是多少、以及该低维结构能否预测未参与拟合的条件。</div></section>

<section id="accuracy"><div class="section-head"><div class="section-no">03 · Accuracy law</div><div><h2>Thinking redistributes sensitivity between count and context.</h2><p class="lede">Non-thinking 由 <strong>{FORMULA[direct_acc.selected_candidate]}</strong> 概括；Native-thinking 由 <strong>{FORMULA[native_acc.selected_candidate]}</strong> 概括。两者使用同一 Bernoulli-logit 框架，却选择不同 basis，因此不是同一条 law 的截距平移。</p></div></div>
<div class="formula-grid"><div class="formula"><div class="tag">Non-thinking · total expression</div><code>p̂ₛ(N,L)=σ(β₀,ₛ + βN,ₛ ln N + βL,ₛ Lₖ)</code><div class="law-metrics"><span>median CV D² {num(direct_acc.median_primary_score,3)}</span><span>Q25 {num(direct_acc.q25_primary_score,3)}</span><span>LOMO stability {pct(direct_acc.lomo_formula_stability,1)}</span><span>LOMO held D² {num(direct_acc.lomo_median_held_primary_score,3)}</span></div><p class="law-reading"><code>∂logit(p̂)/∂N=βN,ₛ/N</code>；小 N 时数量惩罚最陡。<code>∂logit(p̂)/∂Lₖ=βL,ₛ</code>；长度惩罚近似恒定。</p></div><div class="formula"><div class="tag">Native-thinking · total expression</div><code>p̂ₛ(N,L)=σ(γ₀,ₛ + γN,ₛ N + γL,ₛ ln Lₖ)</code><div class="law-metrics"><span>median CV D² {num(native_acc.median_primary_score,3)}</span><span>Q25 {num(native_acc.q25_primary_score,3)}</span><span>LOMO stability {pct(native_acc.lomo_formula_stability,1)}</span><span>LOMO held D² {num(native_acc.lomo_median_held_primary_score,3)}</span></div><p class="law-reading"><code>∂logit(p̂)/∂N=γN,ₛ</code>；每增加一个目标有近似恒定惩罚。<code>∂logit(p̂)/∂Lₖ=γL,ₛ/Lₖ</code>；额外长度的边际惩罚递减。</p></div></div>
<p>σ(z)=1/(1+e<sup>−z</sup>)。上述导数作用在 log-odds，不是准确率百分点；两种 basis 下的原始系数大小也不能直接比较。方向则高度一致：Non-thinking 的 ln N 与 Lₖ 系数在 {pct(direct_log_n_negative,0)}、{pct(direct_l_k_negative,0)} 的槽中为负；Native-thinking 的 N 与 ln Lₖ 系数在 {pct(native_n_negative,0)}、{pct(native_log_l_negative,0)} 的槽中为负。因此“更多目标、更长上下文降低准确率”不是少数模型驱动。</p>
<p><b>主导效应按标准化系数而不是原始系数判定。</b> 对 Accuracy 定义 <code>E=β·SD(x)</code>，表示 predictor 增加一个样本标准差时 log-odds 改变多少；先在每个槽独立计算，再对 12 个 <code>|E|</code> 取中位数。Non-thinking 的 ln N 为 {num(direct_accuracy_n_effect['median_abs'],2)}，Lₖ 为 {num(direct_accuracy_l_effect['median_abs'],2)}，前者约为后者 {num(direct_accuracy_n_effect['median_abs'] / direct_accuracy_l_effect['median_abs'],1)} 倍。Native-thinking 的 ln Lₖ 为 {num(native_accuracy_l_effect['median_abs'],2)}，N 为 {num(native_accuracy_n_effect['median_abs'],2)}，前者约为后者 {num(native_accuracy_l_effect['median_abs'] / native_accuracy_n_effect['median_abs'],1)} 倍。因此正文的机制解释只突出 <b>Non-thinking 的 numerosity sensitivity</b> 与 <b>Native-thinking 的 context-length sensitivity</b>；较小项仍保留在预测方程中，不能把它们当作精确的零。</p>
<figure><img src="{figures['accuracy_surfaces']}" alt="Non-thinking 与 Native-thinking 的总体 accuracy scaling-law 曲线"><figcaption><strong>FIGURE 2</strong><span><b>总体 scaling-law 视图。</b> 横轴列出全部 14 个真实目标数 N=1,2,3,4,5,6,7,8,9,10,12,15,18,20（位置为 log₂N），纵轴是 parsed exact accuracy；黄色→紫色连续色带依次表示全部 8 个长度 L=1k,2k,3k,5k,8k,10k,15k,20k。对每个 mode、N、L，先取得 12 个槽的 cell accuracy；空心点是其 Q50，中间误差棒下端/上端分别为 Q25/Q75。再用 12 套槽特异方程计算 12 个预测；实线是预测 Q50，阴影覆盖预测 [Q25,Q75]。<code>IQR=Q75−Q25</code>，这里表示模型异质性，不是抽样置信区间。</span></figcaption></figure>
<figure><img src="{figures['accuracy_l_axis']}" alt="以 L 为横轴、N 为曲线的 accuracy 经验律"><figcaption><strong>FIGURE 3</strong><span><b>同一 Accuracy law 的 L-horizontal 视图。</b> 横轴为全部 8 个 L，横向位置按 ln L 排列、标签保留原始 token 数；纵轴为 parsed exact accuracy。黄→紫的 14 条曲线按 N=1→20 排列。对每个 N、L，空心点是 12 个槽观测 accuracy 的 Q50，细竖线连接 Q25 与 Q75；实线是在相同 N、L 上计算 12 个槽特异预测后取 Q50。该转置视图没有重拟合模型，只用于直接比较在固定 N 下随 L 增长的变化。</span></figcaption></figure>
<figure><img src="{figures['accuracy_model_panels']}" alt="12 个比较槽的 accuracy 经验律 3×4 小倍图"><figcaption><strong>FIGURE 4</strong><span><b>Accuracy 的模型分辨 3×4 图。</b> 每个面板对应一个比较槽；横轴位置为 log₂N、标签保留原始 N，纵轴为 parsed exact accuracy。Non-thinking 用实线/空心圆，Native-thinking 用虚线/空心三角；黄→紫八种颜色显示全部 8 个 L。每个观测点按一个 cell 内 <code>Σ1[解析整数=N]/30</code> 计算；曲线是在相同 N、L 上计算 <code>σ(Xβₛ)</code> 得到的槽特异 Bernoulli-logit 预测。共享的是 mode-specific basis，不是 βₛ。† 表示 matched checkpoint pair。</span></figcaption></figure>
<h3>Why these laws are credible—and what remains tentative</h3><p>结构选择依据 held-condition 预测而不是训练内拟合。Non-thinking 最终式在 near-best 候选中具有更低的中位 held-out log loss，并在 LOMO 中保持 {pct(direct_acc.lomo_formula_stability,1)}；Native-thinking 的 N+ln Lₖ 与 ln N+ln Lₖ 在当前网格上非常接近，最终式 median D²={num(native_acc.median_primary_score,3)}，但 LOMO 只有 {pct(native_acc.lomo_formula_stability,1)}，所以“线性 N”这一精确细节应标为暂定。</p>
<p>与主导效应相容的解释是：Non-thinking 的主要瓶颈是形成精确的全局 numerosity 表征。若内部 count uncertainty 随 N 的尺度扩张，则落入一个固定宽度 exact-count 区间的概率会呈幂律下降，因而在 log-odds 上出现强 ln N 项。Native-thinking 的主要瓶颈转为长上下文检索：标准 softmax 在更多 token 间分配固定注意力质量，若相关证据的有效质量近似按 L 的幂下降，就会出现强 ln Lₖ 项。这里不再用较弱的 Native N 项构造“逐项串行处理”主张；上述两种解释都是可检验的行为机制假说，不是由回归本身证明的内部算法。</p>
<details class="equations"><summary>Non-thinking：12 个模型的 Accuracy 具体表达式</summary><div class="table-wrap">{table_html(direct_accuracy_equations, 'data-table equation-table')}</div></details>
<details class="equations"><summary>Native-thinking：12 个模型的 Accuracy 具体表达式</summary><div class="table-wrap">{table_html(native_accuracy_equations, 'data-table equation-table')}</div></details>
<details><summary>Accuracy law 的选择统计</summary><div class="table-wrap">{table_html(law_main[law_main['Outcome'].str.contains('Accuracy')])}</div></details>
<div class="conclusion"><strong>本段结论：</strong>完整方程仍同时包含 N 与 L，但主导效应不同：Non-thinking 的最大标准化项是 ln N，支持“精确全局数量表征随 numerosity 退化”的解释；Native-thinking 的最大标准化项是 ln Lₖ，支持“thinking 的剩余瓶颈主要在长上下文检索”的解释。较小项用于保持预测完整性，不承担正文的机制主张。</div></section>

<section id="bias"><div class="section-head"><div class="section-no">04 · Bias law</div><div><h2>A shared interaction topology, with distinct strength.</h2><p class="lede">对可解析响应的 10% 截尾有符号偏差，两种主 mode 都选择 N + Lₖ + N×Lₖ。这里的区别不在 basis，而在每个模型的系数、残差幅度和跨模型稳定性。</p></div></div>
<div class="formula-grid"><div class="formula"><div class="tag">Non-thinking · total expression</div><code>b̂₁₀,ₛ=β₀,ₛ+βN,ₛN+βL,ₛLₖ+βNL,ₛNLₖ</code><div class="law-metrics"><span>median CV R² {num(direct_bias.median_primary_score,3)}</span><span>Q25 {num(direct_bias.q25_primary_score,3)}</span><span>LOMO stability {pct(direct_bias.lomo_formula_stability,1)}</span><span>LOMO held R² {num(direct_bias.lomo_median_held_primary_score,3)}</span></div></div><div class="formula"><div class="tag">Native-thinking · total expression</div><code>b̂₁₀,ₛ=γ₀,ₛ+γN,ₛN+γL,ₛLₖ+γNL,ₛNLₖ</code><div class="law-metrics"><span>median CV R² {num(native_bias.median_primary_score,3)}</span><span>Q25 {num(native_bias.q25_primary_score,3)}</span><span>LOMO stability {pct(native_bias.lomo_formula_stability,1)}</span><span>LOMO held R² {num(native_bias.lomo_median_held_primary_score,3)}</span></div></div></div>
<p>交互项使数量效应依赖长度：<code>∂b̂₁₀/∂N=βN,ₛ+βNL,ₛLₖ</code>；反过来，长度效应也依赖 N。它表达的不是“两个困难因素简单相加”，而是上下文越长时，新增目标对少计/多计方向的边际影响可能改变。每个槽的 CV 增益按“含交互模型的 held-condition score − 无交互父模型的 score”计算；再对 12 个槽取中位数。Non-thinking 的中位 CV 增益为 {num(direct_interaction.median_cv_score_gain_over_parent,3)}，Native-thinking 为 {num(native_interaction.median_cv_score_gain_over_parent,3)}；两者都通过冻结交互门，但 Native-thinking 的剩余偏差更小、跨模型预测力更分散。</p>
<figure><img src="{figures['bias_surfaces']}" alt="Non-thinking 与 Native-thinking 的观测和拟合截尾偏差曲面"><figcaption><strong>FIGURE 5</strong><span><b>Bias law 的总体曲面。</b> 横轴为 L，纵轴为 N；每个左侧像素先取得相同 mode、N、L 下的 12 个槽级 <code>b₁₀</code>，再取 Q50；右侧像素先用 12 套槽特异 OLS 方程预测，再取预测 Q50。Scaling Purple 表示负值（少计）、Scaling Yellow 表示正值（多计）、白色为 0。共用色标以 0 对称，其绝对端点为 <code>max(0.5, aggregate observed |bias| 的 98th percentile)</code>；超过端点的值颜色饱和。Bias 只对可解析响应定义，不能替代 accuracy。</span></figcaption></figure>
<figure><img src="{figures['bias_l_axis']}" alt="以 L 为横轴、N 为曲线的截尾偏差经验律"><figcaption><strong>FIGURE 6</strong><span><b>同一 Bias law 的 L-horizontal 视图。</b> 横轴为全部 8 个 L（位置按 ln L，标签为原始 token 数），纵轴为 10% trimmed signed bias；0 表示没有方向性误差。黄→紫的 14 条曲线表示 N=1→20。空心点是 12 个槽观测 <code>b₁₀</code> 的 Q50，细竖线是 [Q25,Q75]，实线是 12 个槽特异 OLS 预测的 Q50。该图把 Figure 5 的 N×L 曲面按固定 N 切成 14 条剖面，没有另行拟合。</span></figcaption></figure>
<figure><img src="{figures['bias_model_panels']}" alt="12 个比较槽的 10% 截尾偏差经验律 3×4 小倍图"><figcaption><strong>FIGURE 7</strong><span><b>Bias 的模型分辨 3×4 图。</b> 每个面板对应一个比较槽；横轴位置为 log₂N、标签保留原始 N，纵轴为 10% trimmed signed bias，水平 0 表示无方向性误差。Non-thinking 为实线/空心圆，Native-thinking 为虚线/空心三角；黄→紫八种颜色显示全部 8 个 L。每个点是该 cell 将 m 个可解析 signed errors 排序、两端各删 <code>floor(0.10m)</code> 后的均值；线是在相同 N、L 上代入该槽 OLS 方程得到的预测。共享 topology 不等于共享系数。</span></figcaption></figure>
<h3>Outlier and influence audit</h3>
<p>这里检查两种不同层级的异常影响。第一层是 <b>request tail</b>：每个 cell 在计算 <code>b₁₀</code> 前，按 signed error 排序并对称删除两端各 <code>floor(0.10m)</code> 个响应。第二层是 <b>regression influence</b>：即使 cell 内已经截尾，某个 N×L cell 仍可能同时具有大残差和高 leverage。对每个槽的已选 bias OLS 计算 <b>Cook's D</b>，其中 <code>Dᵢ=[eᵢ²/(p·MSE)]·hᵢᵢ/(1−hᵢᵢ)²</code>；<code>eᵢ</code> 是残差，<code>hᵢᵢ</code> 是帽子矩阵对角线，<code>p</code> 是参数数目，<code>MSE</code> 是残差均方。<code>Dᵢ&gt;4/n</code> 只是描述性筛查阈值，不是“该点必为错误样本”的假设检验。</p>
<div class="table-wrap">{table_html(bias_tail_table)}</div>
<p>原始 request 尾部确实包含极端值：Non-thinking 的最大绝对 signed error 为 2,021，但 10% 对称截尾后进入 cell bias 的最大保留绝对误差降至 19。Native-thinking 相应从 73 降至 26。Non-thinking 的 raw 与 trimmed cell bias 相关仅为 0.706，说明若直接用普通均值，少数尾部失败会实质改变 estimand；当前正式 bias 拟合没有使用这些被截去的尾部响应。</p>
<div class="table-wrap">{table_html(bias_influence_table)}</div>
<p>再将 Cook-flagged cells 删除，并在每个槽内保持同一已选 topology 重拟合。表中的 <b>curve r</b> 是原拟合与删点重拟合在 112 个注册 N×L 条件上的 Pearson 相关；<b>curve RMSE</b> 是两条拟合曲面之差的均方根；<b>largest pointwise change</b> 是任一注册条件上的最大绝对变化，均以 count units 表示。典型槽的曲面相关在两种 mode 下都约为 0.999，但最弱槽分别为 0.969 与 0.941，说明总体 topology 并非由单一异常 cell 决定，不过个别模型的局部系数强度仍受高影响条件牵引。</p>
<details><summary>逐模型 Cook's D 与固定-topology 删点重拟合</summary><div class="table-wrap">{table_html(bias_influence_details)}</div></details>
<details class="equations"><summary>Non-thinking：12 个模型的 Bias 具体表达式</summary><div class="table-wrap">{table_html(direct_bias_equations, 'data-table equation-table')}</div></details>
<details class="equations"><summary>Native-thinking：12 个模型的 Bias 具体表达式</summary><div class="table-wrap">{table_html(native_bias_equations, 'data-table equation-table')}</div></details>
<details><summary>Bias law 的选择统计</summary><div class="table-wrap">{table_html(law_main[law_main['Outcome'].str.contains('bias', case=False)])}</div></details>
<div class="conclusion"><strong>本段结论：</strong>Bias 的区别应写成 <em>shared topology, distinct strength</em>：两种 mode 都需要 N×L 交互，但 Non-thinking 的方向性误差曲面更强、更稳定。异常值审计进一步表明：cell 内截尾成功阻断了极端 request 尾部；固定 topology 的删点重拟合通常几乎不改变曲面，但个别模型仍有局部高影响条件，故模型特异系数不应被解读为精确常数。</div></section>

<section id="mae"><div class="section-head"><div class="section-no">05 · MAE law</div><div><h2>Error magnitude adds a third, deliberately conditional view.</h2><p class="lede">Conditional MAE 不关心多计还是少计，只度量可解析响应平均偏离目标多少。它补充 Accuracy 与 Bias，但由于排除 parse failures 且保留所有可解析极端误差，应作为幅度诊断而不是新的主终点。</p></div></div>
<div class="formula-grid"><div class="formula"><div class="tag">Non-thinking · conditional MAE</div><code>ηₛ=ln(1+MAEₛ)=β₀,ₛ+θₛᵀφ(N,L)</code><div class="law-metrics"><span>selected {FORMULA[direct_mae.selected_candidate]}</span><span>median CV R² {num(direct_mae.median_primary_score,3)}</span><span>Q25 {num(direct_mae.q25_primary_score,3)}</span><span>LOMO stability {pct(direct_mae.lomo_formula_stability,1)}</span></div><p class="law-reading"><code>MAE-hat=max(0,exp(η-hat)−1)</code>；CV 指标在反变换后的 count-error 尺度计算。</p></div><div class="formula"><div class="tag">Native-thinking · conditional MAE</div><code>ηₛ=ln(1+MAEₛ)=γ₀,ₛ+θₛᵀφ(N,L)</code><div class="law-metrics"><span>selected {FORMULA[native_mae.selected_candidate]}</span><span>median CV R² {num(native_mae.median_primary_score,3)}</span><span>Q25 {num(native_mae.q25_primary_score,3)}</span><span>LOMO stability {pct(native_mae.lomo_formula_stability,1)}</span></div><p class="law-reading">同一非负反变换；共享的是 mode-level basis，12 个槽仍分别估计系数。</p></div></div>
<p>每个 cell 的 conditional MAE 按 <code>(1/m)Σ|predicted−N|</code> 计算，只纳入可解析整数且要求 <code>m≥20</code>。先拟合 <code>ln(1+MAE)</code> 是因为 MAE 非负且右尾较长；反变换时截在 0，避免恒等链接产生不可能的负预测。候选 registry、五折 held-condition CV、near-best 规则和 LOMO 与 Accuracy/Bias 保持一致，没有 bootstrap。</p>
<p><b>MAE 的主导项与 Accuracy 不同。</b> 这里的标准化效应定义为 <code>E=β·SD(x)/SD[ln(1+MAE)]</code>。Non-thinking 的 N 项中位 <code>|E|</code>={num(direct_mae_n_effect['median_abs'],2)}，约为 ln Lₖ 项 {num(direct_mae_l_effect['median_abs'],2)} 的 {num(direct_mae_n_effect['median_abs'] / direct_mae_l_effect['median_abs'],1)} 倍，而且 12/12 槽方向为正。这说明一旦只看可解析输出，错误幅度主要随目标数增长：Non-thinking 不仅更容易失去 exactness，失误时也会在更大的 numerosity 上偏离得更远。Accuracy 的强 ln N 描述“零误差概率”如何下降，MAE 的强 N 描述“非零误差分布有多宽”；它们是同一误差分布的不同函数，因此不要求选择相同变换。</p>
<p>Native-thinking 的主效应较小且方向不稳定：N 与 Lₖ 的中位 <code>|E|</code> 分别只有 {num(native_mae_n_effect['median_abs'],2)} 与 {num(native_mae_l_effect['median_abs'],2)}；真正主导的是正的 N×Lₖ 交互，中位 <code>|E|</code>={num(native_mae_interaction_effect['median_abs'],2)}，约为较大的单轴主效应 {num(native_mae_interaction_effect['median_abs'] / max(native_mae_n_effect['median_abs'], native_mae_l_effect['median_abs']),1)} 倍，12/12 槽为正、其中 {pct(native_mae_interaction_effect['q05_fraction'],0)} 的槽在 HC3+BH 后 q&lt;0.05。其行为解释不是“单独增加 N 或 L 就稳定放大误差”，而是<b>只有大 N 与长 L 同时出现时，剩余可解析错误的幅度明显放大</b>。这与检索—聚合耦合相容：更长上下文提高每个目标的漏检/重复计入风险，而更多目标增加这种风险转化为大 count error 的机会；回归只能支持这种联合瓶颈的几何，不能识别具体内部步骤。</p>
<figure><img src="{figures['mae_l_axis']}" alt="以 L 为横轴、N 为曲线的 conditional MAE 经验律"><figcaption><strong>FIGURE 8</strong><span><b>Conditional MAE 的 L-horizontal 视图。</b> 横轴为全部 8 个 L（ln L 位置），纵轴为可解析响应的平均绝对计数误差。黄→紫的 14 条曲线表示全部 N。空心点是 12 个槽 cell MAE 的 Q50，细竖线是 [Q25,Q75]；实线先在每个槽计算所选 <code>η=Xβ</code>，再用 <code>max(0,exp(η)−1)</code> 回到 count units，最后取 12 个预测的 Q50。IQR 是模型异质性而非置信区间。</span></figcaption></figure>
<p>典型 held-condition 预测力较高：Non-thinking median CV R²={num(direct_mae.median_primary_score,3)}，Native-thinking={num(native_mae.median_primary_score,3)}。但 topology 稳定性不同，LOMO 分别为 {pct(direct_mae.lomo_formula_stability,1)} 与 {pct(native_mae.lomo_formula_stability,1)}；因此不能把所有模型的精确系数当作普适常数。最高 Non-thinking cell 10% trimmed MAE={num(direct_mae_peak[MAE_FAMILY],2)}，来自 {html.escape(str(direct_mae_peak.comparison_slot))}、N={int(direct_mae_peak.N)}、L={int(direct_mae_peak.L)//1000}k；该 cell 截尾后保留的最大绝对误差为 {num(direct_mae_peak.max_abs_retained_for_mae,0)}。</p>
<div class="table-wrap">{table_html(mae_main)}</div>
<details class="equations"><summary>Non-thinking：12 个模型的 conditional MAE 具体表达式</summary><p><code>η=ln(1+MAE)</code>；表中方程给出 η，原尺度预测统一按 <code>max(0,exp(η)−1)</code> 计算。</p><div class="table-wrap">{table_html(direct_mae_equations, 'data-table equation-table')}</div></details>
<details class="equations"><summary>Native-thinking：12 个模型的 conditional MAE 具体表达式</summary><p><code>η=ln(1+MAE)</code>；表中方程给出 η，原尺度预测统一按 <code>max(0,exp(η)−1)</code> 计算。</p><div class="table-wrap">{table_html(native_mae_equations, 'data-table equation-table')}</div></details>
<div class="conclusion"><strong>本段结论：</strong>只按大系数解释时，Non-thinking MAE 的主导因素是 N，表示 numerosity 决定错误幅度；Native-thinking MAE 的主导因素是 N×Lₖ，表示大目标数与长上下文共同触发幅度放大。精确 topology 的 LOMO stability 仍只有 {pct(direct_mae.lomo_formula_stability,1)} / {pct(native_mae.lomo_formula_stability,1)}，且 conditional MAE 保留极端可解析错误，所以这些是主导模式而非普适常数。</div></section>

<section id="appendix" class="appendix"><div class="section-head"><div class="section-no">Appendix A</div><div><h2>Enumeration controls.</h2><p class="lede">Index 与 Bullet 是显式枚举控制，不参与正文的 Non-thinking vs Native-thinking 论证。Accuracy 使用冻结 V3.2 分析；Bias 同时披露预注册的 10% 截尾结果和只改变 estimand 的未截尾敏感性分析。</p></div></div>
<figure><img src="{figures['enumeration_model_panels']}" alt="12 个比较槽的 enumeration accuracy 经验律 3×4 小倍图"><figcaption><strong>FIGURE A1</strong><span><b>Enumeration accuracy 的模型分辨 3×4 图。</b> 每个面板对应一个比较槽；横轴位置为 log₂N、标签保留原始 N，纵轴为 parsed exact accuracy。Index 为实线/空心圆，Bullet 为虚线/空心三角；黄→紫八种颜色显示全部 8 个 L。每个点按该 cell 的 <code>Σ1[解析整数=N]/30</code> 计算；曲线是在相同 N、L 上代入该槽独立系数并应用 inverse-logit 后的预测。两种 enumeration 共享所选 basis ln N+ln Lₖ，但不共享系数。</span></figcaption></figure>
<p>两种 enumeration 的 logit accuracy 都选择 ln N + ln Lₖ。下面的 CV D² 衡量未见 N×L 条件上的概率预测相对常数准确率基线改善多少；LOMO stability 衡量删除任一模型后是否仍选中同一 basis；LOMO held D² 则衡量由其他 11 个模型决定的 basis 在被留出模型内部是否仍能预测未见条件。</p>
<div class="table-wrap">{table_html(enumeration_accuracy_fit)}</div>
<h3>Post-hoc sensitivity: untrimmed signed-mean bias</h3>
<p>每个 cell 约有 30 个 seed。10% 截尾会从两端各删 3 个误差；当准确率很高、非零错误本来就很稀疏时，这可能把一个 cell 的全部方向性错误删掉。敏感性分析保持相同的 13 个候选、五折条件划分、交互门限、模型特异系数和 LOMO，只把 <code>b₁₀</code> 换成所有可解析输出的 <code>mean(predicted−N)</code>。</p>
<div class="table-wrap">{table_html(sensitivity_fit)}</div>
<p>取消截尾后，Index 由 <strong>{FORMULA[index_sensitivity.trimmed_selected_candidate]}</strong>、median CV R²={num(index_sensitivity.trimmed_median_primary_score,3)}，变为 <strong>{FORMULA[index_sensitivity.untrimmed_selected_candidate]}</strong>、median CV R²={num(index_sensitivity.untrimmed_median_primary_score,3)}、Q25={num(index_sensitivity.untrimmed_q25_primary_score,3)}、LOMO stability={pct(index_sensitivity.untrimmed_lomo_formula_stability,1)}。它与 Native-thinking 选择同一 N+Lₖ+N×Lₖ topology。Bullet 变为 <strong>{FORMULA[bullet_sensitivity.untrimmed_selected_candidate]}</strong>，median CV R²={num(bullet_sensitivity.untrimmed_median_primary_score,3)}、LOMO stability={pct(bullet_sensitivity.untrimmed_lomo_formula_stability,1)}；含交互候选略有更高分数，但未通过冻结的跨模型交互支持门限。</p>
<details><summary>为什么截尾会改变 enumeration bias law</summary><div class="table-wrap">{table_html(sensitivity_zero)}</div><p><b>Zero-cell fraction</b> 按 <code>#{'{'}cell: |bias|≤1e−12{'}'} / 1,344</code> 计算，其中每个 mode 有 <code>12 slots×14 N×8 L=1,344</code> 个 cell。<b>SST</b>（total sum of squares）按 <code>Σᵢ(bᵢ−b̄)²</code> 计算，衡量该 estimand 在 cell 间的总变异。截尾后 Index/Bullet 的大量 cell 被压成 0，使 SST 和可预测方差同时缩小。表中的相关系数是 <b>Pearson r</b>：<code>r=Σ(b₁−b̄₁)(b₂−b̄₂)/√[Σ(b₁−b̄₁)²Σ(b₂−b̄₂)²]</code>，其中 b₁/b₂ 是同一 cell 的截尾/未截尾 bias。较高 r 说明两个 estimand 有线性关联，但不表示数值相等。</p></details>
<details><summary>未截尾 law 的跨模型中位系数</summary><div class="table-wrap">{table_html(untrimmed_coefficient_comparison)}</div><p>每个系数先在 12 个槽内分别拟合；表中“跨模型中位系数”是这 12 个槽系数的 Q50，不是把数据混在一起重拟合的系数。Index 与 Native-thinking 的 N、Lₖ 和 N×Lₖ 中位系数方向及量级接近。Bullet 的交互项没有达到共享结构门限，因此正式选择只保留 N 与 Lₖ。</p></details>
<p class="note"><strong>敏感性分析边界：</strong>未截尾均值估计 expected signed error=<code>Σ(predicted−N)/m</code>，因此会保留罕见的极端多计误差（这是描述性称呼，未预设分类阈值）；Index 的最大正误差达到 +285。因此较强 R² 可能部分描述尾部失败随 N/L 的变化，而不是典型错误。高 LOMO 表明结构并非只由单一模型决定，但不能消除均值对极端值敏感这一事实。</p>
<details class="equations"><summary>Index：12 个模型的 Accuracy 具体表达式</summary><div class="table-wrap">{table_html(index_accuracy_equations, 'data-table equation-table')}</div></details>
<details class="equations"><summary>Bullet：12 个模型的 Accuracy 具体表达式</summary><div class="table-wrap">{table_html(bullet_accuracy_equations, 'data-table equation-table')}</div></details>
<details><summary>所有模式与链接函数的结构选择</summary><div class="table-wrap">{table_html(law_all)}</div></details>
<details><summary>逐槽 Accuracy 与可比性</summary><div class="table-wrap">{table_html(slot_table)}</div></details>
<div class="conclusion"><strong>本段结论：</strong>Enumeration 作为控制条件支持“显式结构化输出通常改善准确率”。预注册截尾 estimand 下的 bias law 较弱，但未截尾敏感性显示 Index 与 Native-thinking 共享 N+Lₖ+N×Lₖ topology，Bullet 也获得可迁移的 N+Lₖ law。最合理的解释是 10% cell-wise trimming 在高准确率区间删除了稀疏方向性错误；该结果应作为 post-hoc robustness evidence，而不是替换预注册主结果。</div></section>

<section id="repro"><div class="section-head"><div class="section-no">Appendix B</div><div><h2>Reproducibility and claim boundary.</h2><p class="lede">推理输出、请求 ID、刺激与经验律输入均由哈希锁定。报告可在本地 CPU 上重建；任何新模型、新长度或新 seed 都属于外部验证，而不是当前结果的静默扩展。</p></div></div>
<details><summary>Frozen model revisions</summary><div class="table-wrap">{table_html(revisions)}</div></details>
<details><summary>Files, hashes, and rebuild command</summary><div class="methods">protocol: {html.escape(audit['protocol_version'])}
inference commit: {html.escape(audit['git_commit'])}
requests: {audit['requests']:,}
unique_request_ids: {audit['unique_request_ids']:,}
request_ids_sha256: {html.escape(audit['request_ids_sha256'])}
stimuli_sha256: {html.escape(audit['stimuli_sha256'])}
analysis input sha256: {html.escape(manifest['input_sha256'])}
base analysis config sha256: {html.escape(manifest['base_config_sha256'])}
inverse-count extension config sha256: {html.escape(manifest['extension_config_sha256'])}
analysis elapsed: {manifest['elapsed_seconds']:.1f} seconds
software: {html.escape(json.dumps(manifest['software'], ensure_ascii=False))}

Rebuild:
.venv\\Scripts\\python.exe scripts\\build_niah_empirical_law_v3_2_report.py</div></details>
<h3>Limits that remain</h3><ul><li><strong>Grid-bound:</strong> 规律只在 N∈{html.escape(str(v32['immutable_input']['N_levels']))}、L∈{html.escape(str(v32['immutable_input']['L_levels']))} 上成立；图外推不属于证据。</li><li><strong>Model-set-bound:</strong> LOMO 只针对当前 12 个比较槽。</li><li><strong>Behavioral, not mechanistic:</strong> basis 与系数描述输出误差曲面，不识别内部计数机制。</li><li><strong>Parsing matters:</strong> Bias 与 conditional MAE 只在可解析响应上定义；只有 Accuracy 把解析失败计为失败。</li><li><strong>MAE tail sensitivity:</strong> Conditional MAE 不截尾，极端但可解析的错误会进入均值；其 topology stability 也低于主 Accuracy/Bias 结果。</li></ul>
<div class="conclusion"><strong>本段结论：</strong>当前证据足以支持 Behavioral Study: Distinct Error Laws：Accuracy 呈 mode-specific topology，Bias 呈 shared topology with distinct strength；MAE 提供有用但较不稳定的幅度证据。结果不支持网格外预测、全模型普适性或内部机制因果解释。</div></section>
<footer class="footer">NiaH Empirical-law Report · V3.2 frozen analysis + post-freeze 1/N candidate audit · generated {html.escape(generated)}.</footer>
</main></body></html>"""

    # Focused V3.2 paper-style report. The historical template above is kept as
    # provenance, but this is the only artifact written to disk.
    report = rf"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light">
<title>NiaH Empirical-law Report · V3.2</title><style>{css}</style>
<script>{plotly_bundle}</script>
<script>window.MathJax={{tex:{{inlineMath:[["\\(","\\)"],["$","$"]],displayMath:[["\\[","\\]"]]}},svg:{{fontCache:"global"}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script></head>
<body><header class="hero"><div class="page"><div class="brand"><span>NiaH Empirical-law Report</span><span>V3.2 · 10% trimmed MAE</span></div><div class="hero-grid"><div><div class="kicker">Behavioral Study: Distinct Error Laws</div><h1>Accuracy and central error magnitude obey different behavioral laws.</h1><p class="dek">报告分为两个正文结果：完全正确概率（Accuracy）与可解析计数误差中间 80% 的绝对幅度（10% symmetrically trimmed conditional MAE）。每个 prompt mode 共享函数结构，但 12 个模型槽分别估计系数；两种 enumeration 以及方向性 Bias 放在附录。所有公式由 MathJax 渲染，所有曲线均覆盖注册的 14 个 N 与 8 个 L。</p><div class="meta"><span>161,280 requests</span><span>14 fixed revisions</span><span>12 comparison slots</span><span>14 N × 8 L × 30 seeds</span><span>18 candidate laws</span><span>0 bootstrap</span><span>commit {html.escape(audit['git_commit'][:12])}</span></div></div><aside class="hero-aside"><strong>V3.2</strong><span>MAE 对绝对误差排序后左右各删 10%，再在 count units 上回归；没有 log 变换，也没有把负预测静默截为 0。</span></aside></div></div></header>
<nav class="nav"><div class="page"><strong>目录</strong><a href="#design">0. 设定与方法</a><a href="#accuracy">1. Accuracy</a><a href="#mae">2. MAE</a><a href="#appendix-a">Appendix A · Interaction</a><a href="#appendix-b">Appendix B · Bias</a><a href="#appendix-c">Appendix C · Enumeration</a><a href="#appendix-d">Appendix D · 3D</a><a href="#repro">Reproducibility</a></div></nav>
<main class="page">

<section id="design"><div class="section-head"><div class="section-no">0 · Definitions</div><div><h2>基础设定、estimands 与计算方法</h2><p class="lede">经验律在固定模型集合与固定 N–L 网格上总结行为几何；它不是模型参数量 scaling law，也不是内部机制的因果模型。</p></div></div>
<div class="key-grid"><div class="key"><div class="value">12</div><div class="label">comparison slots；每个槽独立估计系数</div></div><div class="key"><div class="value">112</div><div class="label">每槽每 mode 的 N×L 条件</div></div><div class="key"><div class="value">30</div><div class="label">每个条件的固定随机 seeds</div></div><div class="key"><div class="value">5-fold</div><div class="label">整块 held-condition validation</div></div></div>
<h3>0.1 Atomic units and three estimands</h3>
<p>一次 request 由模型槽、prompt mode、目标数 \(N\)、passage length \(L\) 与 seed 唯一确定。一个 cell 汇集同一槽、mode、N、L 的 30 个 requests。令第 \(i\) 次输出解析出的整数为 \(\widehat C_i\)；若解析失败，则 \(\widehat C_i\) 对 Accuracy 记为错误，而 Bias 与 MAE 只在可解析子集上定义。</p>
<div class="math-block">\[Y_i=\mathbf 1\{{\widehat C_i=N_i\}},\qquad \widehat p=\frac1n\sum_{{i=1}}^nY_i.\]</div>
<p>Accuracy 的 estimand 是 request-level exact-match probability。对一个含 \(m\ge20\) 个可解析响应的 cell，先令绝对误差 \(a_i=|\widehat C_i-N|\)，排序为 \(a_{{(1)}}\le\cdots\le a_{{(m)}}\)，并令 \(k=\lfloor0.1m\rfloor\)。正文的 MAE estimand 定义为</p>
<div class="math-block">\[\operatorname{{tMAE}}_{{10}}(N,L)=\frac1{{m-2k}}\sum_{{i=k+1}}^{{m-k}}a_{{(i)}}.\]</div>
<p><strong>这是双侧 10% 截尾，不只是删大 outliers。</strong>当 \(m=30\) 时，先删除 3 个最小与 3 个最大的绝对误差，再平均中间 24 个。删除低端值意味着该量不是 raw expected absolute error；它描述可解析响应绝对误差分布的 central 80% magnitude，并降低少量极大计数错误对拟合的支配。</p>
<p class="note"><strong>审计边界：</strong>这一 MAE estimand 是在 V3.2 推理数据与候选/验证设计冻结后，按本次明确要求加入的 post-freeze extension；161,280 个请求、模型 revisions、N/L 网格、五折划分、选择门限与 0-bootstrap 设定均未改变。旧的 raw-MAE 产物保留作审计，但不进入本报告的正式结果。</p>
<p>附录中的 10% trimmed signed bias 先令 \(d_i=\widehat C_i-N\)，排序为 \(d_{{(1)}}\le\cdots\le d_{{(m)}}\)，再取 \(k=\lfloor0.1m\rfloor\) 后的中间均值：</p>
<div class="math-block">\[b_{{10}}=\frac1{{m-2k}}\sum_{{i=k+1}}^{{m-k}}d_{{(i)}}.\]</div>
<div class="conclusion"><strong>定义结论：</strong>Accuracy 测“零误差事件”的概率；\(\operatorname{{tMAE}}_{{10}}\) 测可解析绝对误差中间 80% 的幅度；Bias 测 signed error 中间 80% 的方向性位置。三者是不同 estimands，不能由其中一个替代另一个。</div>

<h3>0.2 Predictor registry and shared topology</h3>
<p>令 \(L_k=L/1000\)，并考虑 \(N,L_k,\ln N,\ln L_k,N^{{-1}}\) 及满足 hierarchy 的一阶交互。V3.2 的 13 个基础候选加 5 个 \(1/N\) 候选共 18 个；这次扩展没有改变任何最终选择。对于同一 mode，所有 12 个槽共享所选 basis \(\phi(N,L)\)，但每个槽 \(s\) 具有独立的截距与系数。</p>
<div class="math-block">\[g\!\left(\mu_{{s,m}}(N,L)\right)=\alpha_{{s,m}}+\boldsymbol\beta_{{s,m}}^\top\phi_m(N,L).\]</div>
<p>Accuracy 使用 \(g=\operatorname{{logit}}\)，MAE 与 Bias 使用 identity link。所谓 topology 指 \(\phi_m\) 中包含哪些项；“shared topology”不表示系数相同。</p>

<h3>0.3 Selection, validation, uncertainty, and plot grammar</h3>
<p>112 个条件按 \((\operatorname{{index}}(N)+\operatorname{{index}}(L))\bmod5\) 分成五折。每次整块留出 N×L 条件，而不是随机拆散同一条件的 seeds。Accuracy 的主损失是 request-level log loss，报告 \(D^2=1-\mathcal L_{{model}}/\mathcal L_{{intercept}}\)；MAE/Bias 报告 held-condition \(R^2=1-\sum(y-\widehat y)^2/\sum(y-\bar y_{{train}})^2\)。分数 1 为完美、0 为不优于常数、负数为更差。</p>
<p>Near-best rule 要求 median score 距最佳不超过 0.02、Q25 距最佳不超过 0.05，然后按更少参数、较低损失、较高 median/Q25 与 registry 顺序破同分。交互还须通过预测增益与跨槽支持门限。系数不确定性使用 HC3 sandwich covariance；同一 family×mode×candidate×term 的 12 个槽 p 值用 Benjamini–Hochberg 控制 FDR。</p>
<p>LOMO（leave-one-model-out）每次完全删除一个模型槽，用另外 11 个槽重新选 topology，共 12 次。Structure stability 是 12 次中仍选中全模型 topology 的比例；LOMO held score 再检验该 topology 在被删槽的 held conditions 上的预测力。它验证“结构可迁移”，不声称可以在没有该模型数据时猜出它的系数。</p>
<p>图中实线/圆点表示 Non-thinking（或 Index），虚线/三角表示 Native-thinking（或 Bullet）；点是观测 cell 值，线是同一模型槽的 fitted value。黄色到紫色仅编码另一坐标轴的离散水平。所有 N 与 L 均显示。若出现 Q25/Q75，则 \(Q_q\) 是样本排序后的 q 分位数，\(\operatorname{{IQR}}=Q_{{0.75}}-Q_{{0.25}}\)；它描述模型异质性，不是置信区间。MAE 图的 symlog 轴定义为零附近线性、超过 0.25 count 后对正负绝对值取对数，仅改变显示，不改变回归。</p>
<div class="conclusion"><strong>方法结论：</strong>模型选择由未见 N×L 条件上的预测表现决定，系数按槽估计，LOMO 检查 topology 是否依赖单一模型；没有使用重型 bootstrap 或额外 nested held-seed/N/L 设计。</div>
<details><summary>完整概念与计算表</summary><div class="table-wrap">{table_html(concept_definitions, 'data-table definition-table')}</div><div class="table-wrap">{table_html(visual_definitions, 'data-table definition-table')}</div></details>
</section>

<section id="accuracy"><div class="section-head"><div class="section-no">1 · Accuracy</div><div><h2>Exactness follows mode-specific odds laws.</h2><p class="lede">正文比较 Non-thinking 与 Native-thinking。二者都用 Bernoulli-logit，但 held-condition selection 选择了不同的 N/L 坐标。</p></div></div>
<h3>1.1 Selected equations</h3>
<div class="formula-grid"><div class="formula"><div class="tag">Non-thinking</div><div class="math-block">\[\operatorname{{logit}}\widehat p_{{s,NT}}=\alpha_s+\beta_{{N,s}}\ln N+\beta_{{L,s}}L_k.\]</div><div class="law-metrics"><span>median CV D² {num(direct_acc.median_primary_score,3)}</span><span>Q25 {num(direct_acc.q25_primary_score,3)}</span><span>LOMO {pct(direct_acc.lomo_formula_stability,1)}</span></div></div><div class="formula"><div class="tag">Native-thinking</div><div class="math-block">\[\operatorname{{logit}}\widehat p_{{s,T}}=\alpha_s+\gamma_{{N,s}}N+\gamma_{{L,s}}\ln L_k.\]</div><div class="law-metrics"><span>median CV D² {num(native_acc.median_primary_score,3)}</span><span>Q25 {num(native_acc.q25_primary_score,3)}</span><span>LOMO {pct(native_acc.lomo_formula_stability,1)}</span></div></div></div>
<p>由于 \(\widehat p=\sigma(\eta)=(1+e^{{-\eta}})^{{-1}}\)，两个 law 等价于对 odds 的乘法分解：</p>
<div class="math-block">\[\frac{{\widehat p_{{s,NT}}}}{{1-\widehat p_{{s,NT}}}}=e^{{\alpha_s}}N^{{\beta_{{N,s}}}}e^{{\beta_{{L,s}}L_k}},\qquad \frac{{\widehat p_{{s,T}}}}{{1-\widehat p_{{s,T}}}}=e^{{\alpha_s}}e^{{\gamma_{{N,s}}N}}L_k^{{\gamma_{{L,s}}}}.\]</div>
<p>因此，Non-thinking 的 \(\ln N\) 项不是“准确率对 N 线性”，而是 exactness odds 对 N 呈幂律；Native-thinking 的 \(\ln L_k\) 则表示 odds 对 passage length 呈幂律。严格地说，固定其他坐标后，把 N 乘以 \(c>0\) 会使 Non-thinking 的预测 odds 乘以 \(c^{{\beta_N}}\)；把 L 乘以 c 会使 Native-thinking 的预测 odds 乘以 \(c^{{\gamma_L}}\)。而每增加一个 target，Native-thinking odds 乘以 \(e^{{\gamma_N}}\)；每增加 1k tokens，Non-thinking odds 乘以 \(e^{{\beta_L}}\)。</p>
<div class="math-block">\[\frac{{\partial\widehat p_{{NT}}}}{{\partial N}}=\widehat p_{{NT}}(1-\widehat p_{{NT}})\frac{{\beta_N}}N,\quad \frac{{\partial\widehat p_{{T}}}}{{\partial L}}=\widehat p_{{T}}(1-\widehat p_{{T}})\frac{{\gamma_L}}L.\]</div>
<p>因子 \(\widehat p(1-\widehat p)\) 表明，同一个 log-odds 变化在 \(p\approx0.5\) 时产生最大的绝对概率变化，在接近 0 或 1 时被压缩。标准化效应 \(E_j=\beta_j\operatorname{{SD}}(x_j)/\operatorname{{SD}}(\eta)\) 只用于比较不同单位的 predictor 对线性预测量 \(\eta\) 的相对贡献。Non-thinking 中 \(|E_{{\ln N}}|\) 的跨槽中位数为 {num(direct_accuracy_n_effect['median_abs'],2)}，Native-thinking 中 \(|E_{{\ln L}}|\) 为 {num(native_accuracy_l_effect['median_abs'],2)}。</p>
<h3>1.2 Mechanism hypothesis: a mode-dependent bottleneck exchange</h3>
<p>下面不是在解释“为什么回归选了某个 basis”，而是在问：<em>什么样的计算瓶颈会自然地产生观测到的 odds 几何？</em> 先把跨 12 个模型槽的系数中位数当作一个描述性“典型槽”（并非 pooled model）。四个斜率在 12/12 槽中都为负，对应的典型 odds multiplier 为：</p>
<div class="formula-grid"><div class="formula"><div class="tag">Non-thinking · multiplicative count scale</div><div class="math-block">\[\frac{{O_{{NT}}(2N,L)}}{{O_{{NT}}(N,L)}}=2^{{{num(direct_log_n,3)}}}={num(2 ** direct_log_n,3)},\qquad \frac{{O_{{NT}}(N,L_k+1)}}{{O_{{NT}}(N,L_k)}}=e^{{{num(direct_l_k,3)}}}={num(math.exp(direct_l_k),3)}.\]</div><p>典型槽中，N 翻倍后 exactness odds 只保留约 {pct(2 ** direct_log_n,1)}；每增加 1k tokens，odds 保留约 {pct(math.exp(direct_l_k),1)}。</p></div><div class="formula"><div class="tag">Native-thinking · serial count cost + relative retrieval cost</div><div class="math-block">\[\frac{{O_{{T}}(N+1,L)}}{{O_{{T}}(N,L)}}=e^{{{num(native_n,3)}}}={num(math.exp(native_n),3)},\qquad \frac{{O_{{T}}(N,2L)}}{{O_{{T}}(N,L)}}=2^{{{num(native_log_l,3)}}}={num(2 ** native_log_l,3)}.\]</div><p>典型槽中，每多一个 target，odds 保留约 {pct(math.exp(native_n),1)}；L 翻倍后保留约 {pct(2 ** native_log_l,1)}。</p></div></div>
<div class="derivation"><p><strong>Non-thinking：近似标量化的 numerosity noise。</strong> 设模型不显式逐项维护计数轨迹，而是形成一个全局近似数感 \(\widetilde N=N+\varepsilon\)，其中误差尺度随数量按 \(\operatorname{{SD}}(\varepsilon\mid N)=cN^q\) 增长。若 exact count 近似要求 \(|\varepsilon|&lt;1/2\)，则在噪声尺度较大时</p><div class="math-block">\[P(|\varepsilon|&lt;1/2\mid N)\approx \frac{{1}}{{\sqrt{{2\pi}}cN^q}}\propto N^{{-q}}.\]</div><p>在较低成功率区域 \(\operatorname{{logit}}p\approx\ln p\)，于是自然出现 \(-q\ln N\)。这与人类 approximate-number 行为中的 scalar variability 只是结构类比，不是 LLM 机制证据（参见 <a href="https://doi.org/10.1111/1467-9280.00120">Whalen, Gallistel &amp; Gelman, 1999</a>）。长度项则可由近似恒定的 context-interference hazard 产生：若每 1k tokens 维持正确全局状态的 survival factor 是 \(s&lt;1\)，则总因子为 \(s^{{L_k}}=e^{{L_k\ln s}}\)。因此 Non-thinking law 对应“相对数量噪声 × 累积上下文干扰”的乘法分解。</p>
<p><strong>Native-thinking：一次检索路由加逐项目标聚合。</strong> 若显式 reasoning 把任务改写为“先从上下文中路由目标证据，再逐项验证/聚合”，一个最小模型是</p><div class="math-block">\[O_T(N,L)\approx A\,S_{{route}}(L)\,r^N,\qquad S_{{route}}(L)\propto L^{{-\kappa}},\quad 0&lt;r&lt;1.\]</div><p>其中 \(r^N=e^{{N\ln r}}\) 表示每增加一个需要正确处理的 target 就多一个近似恒定的可靠性因子；\(L^{{-\kappa}}\) 表示相关证据在更多候选 token 中受到竞争或稀释。标准 softmax attention 的归一化分母随候选数增长，为这种相对长度成本提供了一个 heuristic 起点（<a href="https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html">Vaswani et al., 2017</a>），但网络可以通过改变 logits 抵消稀释，所以它不是定理；长上下文检索确实会受位置和长度影响（<a href="https://aclanthology.org/2024.tacl-1.9/">Liu et al., 2024</a>），而 self-attention 的长度敏感性也存在理论限制（<a href="https://doi.org/10.1162/TACL_A_00306">Hahn, 2020</a>）。该机制正好给出 \(\ln O_T=\ln A+N\ln r-\kappa\ln L\)。</p>
<p><strong>核心 insight：reasoning 不是简单减小同一个系数，而是交换主导瓶颈。</strong> Non-thinking 更像一次压缩后的全局数感，其 exactness 主要随 N 的<em>相对尺度</em>恶化；Native-thinking 外显了逐项处理，缓解这类 scalar compression，却把瓶颈转移到长上下文中的证据路由与逐项聚合。因此两种 mode 可以有相近甚至很高的平均 accuracy，却沿 N–L 平面的不同方向失效。所选 law 中没有 \(N\ln L\) interaction，与“一次路由、随后聚合”相容；若每个 target 都独立重搜整段上下文，更自然会出现 \(Nr(L)\) 或 \(N\ln L\) 型耦合。但 interaction 未被选中只是不支持该额外复杂度，不能证明处理阶段独立。</p></div>
<p class="warning"><strong>机制边界与不确定点：</strong>Non-thinking 的 \(\ln N+L_k\) topology 在 LOMO 中保持 {pct(direct_acc.lomo_formula_stability,1)}。Native-thinking 的长度幂律较可信，但“严格线性 N”仍暂定：\(N+\ln L_k\) 的 median CV D²={num(native_acc.median_primary_score,3)}，而 \(\ln N+\ln L_k\) 为 {num(native_loglog.median_primary_score,3)}，差值仅 {num(native_acc.median_primary_score-native_loglog.median_primary_score,3)}；LOMO stability 只有 {pct(native_acc.lomo_formula_stability,1)}。因此当前最强的机制命题是“Native-thinking 受相对长度检索和目标聚合约束”，不是“每个 target 必然造成完全相同的独立失败概率”。</p>
<details><summary>Three falsifiable follow-ups for the mechanism hypothesis</summary><ol><li><strong>Scalar-noise test：</strong>在 Non-thinking 中回归 robust count-error scale（如 MAD 或 trimmed SD）的 \(\ln\) 对 \(\ln N\)。若标量噪声成立，scale 应近似按 \(N^q\) 增长，而不仅是 Accuracy 曲线碰巧选择 \(\ln N\)。</li><li><strong>Serial-reliability test：</strong>从 Native-thinking trace 中标注 target retrieval、state update 与 final aggregation failure；若 \(r^N\) 解释成立，新增 target 应带来近似稳定的条件 log-odds decrement，并可由可定位的逐项错误预测。</li><li><strong>Retrieval-dilution test：</strong>固定 N 与语义内容，分别增加无关 padding、近似 distractors、改变目标位置。若 \(L^{{-\kappa}}\) 来自竞争，近似 distractors 与中间位置应比等量无信息 padding 造成更强退化；若三者等效，则常量 token hazard 更合理。</li></ol></details>
<div class="conclusion"><strong>Accuracy 公式结论：</strong>Non-thinking 的主要可迁移特征是 N 的乘法尺度，Native-thinking 的主要特征是 L 的乘法尺度；“distinct laws”指 odds 曲面使用不同坐标，而非只比较两个平均准确率。</div>

<h3>1.3 Aggregate scaling-law view</h3>
<figure><img src="{figures['accuracy_aggregate_n']}" alt="aggregate observed and fitted accuracy with N on the horizontal axis"><figcaption><strong>FIGURE 1</strong><span><b>Accuracy 汇总图，N-horizontal。</b> 左/右面板分别为 Non-thinking 与 Native-thinking；横轴显示全部 14 个 N（log₂ 位置），纵轴为 parsed exact accuracy。黄→紫八条曲线固定 L=1k…20k。空心点为 12 个模型槽观测 cell accuracy 的 Q50，误差条为 [Q25,Q75]；实线为 12 套槽特异 logit 方程预测的 Q50，半透明带为预测 [Q25,Q75]。IQR 描述模型异质性，不是 seed uncertainty 或置信区间。</span></figcaption></figure>
<figure><img src="{figures['accuracy_aggregate_l']}" alt="aggregate observed and fitted accuracy with L on the horizontal axis"><figcaption><strong>FIGURE 2</strong><span><b>Accuracy 汇总图，L-horizontal。</b> 横轴显示全部 8 个 passage lengths（对数位置、原始 token 标签），黄→紫 14 条曲线固定 N=1…20；点、误差条、线和带仍分别表示跨 12 槽的观测 Q50/IQR 与预测 Q50/IQR。这是 Figure 1 同一批方程的转置切片，没有重新拟合。</span></figcaption></figure>
<h3>1.4 Model-resolved evidence</h3>
<figure><img src="{figures['accuracy_n']}" alt="12 models, observed and fitted accuracy with N on the horizontal axis"><figcaption><strong>FIGURE 3</strong><span><b>Accuracy 模型分辨图，N-horizontal。</b> 3×4 面板覆盖 12 个模型槽；横轴显示全部 14 个 N，纵轴为 parsed exact accuracy。每种黄→紫颜色固定一个 L；实线/圆点为 Non-thinking，虚线/三角为 Native-thinking。点为 30-seed cell accuracy，线为该槽方程在相同 N,L 上的预测。</span></figcaption></figure>
<figure><img src="{figures['accuracy_l']}" alt="12 models, observed and fitted accuracy with L on the horizontal axis"><figcaption><strong>FIGURE 4</strong><span><b>Accuracy 模型分辨图，L-horizontal。</b> 同一组 12 套槽特异方程的转置视图；横轴显示全部 8 个 L，黄→紫 14 条曲线固定全部 N。它揭示汇总 IQR 背后的具体模型差异。</span></figcaption></figure>
<div class="table-wrap">{table_html(law_main[law_main['Outcome'].str.contains('Accuracy', case=False)])}</div>
{equation_details['direct_accuracy']}{equation_details['native_accuracy']}
<div class="conclusion"><strong>Accuracy 证据结论：</strong>Non-thinking 的 topology 在 LOMO 中保持 {pct(direct_acc.lomo_formula_stability,1)}，Native-thinking 只有 {pct(native_acc.lomo_formula_stability,1)}；后者的“线性 N”细节仍较暂定。可靠主张是两种 mode 在 N/L 难度曲面上的主导坐标不同，而不是每个模型都具有相同系数或完全相同退化速率。</div>
</section>

<section id="mae"><div class="section-head"><div class="section-no">2 · Trimmed conditional MAE</div><div><h2>Central error magnitude follows a shared N–L interaction law.</h2><p class="lede">\(\operatorname{{tMAE}}_{{10}}\) 只对成功解析为整数的响应计量：每个 cell 的绝对误差左右各删 10%，再对中间 80% 取均值，并直接在 count units 上做 OLS。它回答“典型可解析误差幅度如何随 N 与 L 联合变化”，不是 raw expected absolute error。</p></div></div>
<h3>2.1 Trimmed-MAE equations and marginal effects</h3>
<div class="formula-grid"><div class="formula"><div class="tag">Non-thinking · identity OLS</div><div class="math-block">\[\widehat{{\operatorname{{tMAE}}}}_{{10,s,NT}}=a_s+b_{{N,s}}N+b_{{L,s}}L_k+b_{{NL,s}}NL_k.\]</div><div class="law-metrics"><span>median CV R² {num(direct_mae.median_primary_score,3)}</span><span>Q25 {num(direct_mae.q25_primary_score,3)}</span><span>LOMO {pct(direct_mae.lomo_formula_stability,1)}</span></div></div><div class="formula"><div class="tag">Native-thinking · identity OLS</div><div class="math-block">\[\widehat{{\operatorname{{tMAE}}}}_{{10,s,T}}=c_s+d_{{N,s}}N+d_{{L,s}}L_k+d_{{NL,s}}NL_k.\]</div><div class="law-metrics"><span>median CV R² {num(native_mae.median_primary_score,3)}</span><span>Q25 {num(native_mae.q25_primary_score,3)}</span><span>LOMO {pct(native_mae.lomo_formula_stability,1)}</span></div></div></div>
<p>这里的 OLS 参数是有限网格上 trimmed cell functional 的线性投影：若 \(T=\operatorname{{tMAE}}_{{10}}\) 且 \(X=(1,N,L_k,NL_k)^\top\)，则 \(\beta^*=\arg\min_\beta\operatorname{{E}}[(T-X^\top\beta)^2]\)，并满足 normal equations \(\operatorname{{E}}[X(T-X^\top\beta^*)]=0\)。它估计的是 middle-80% absolute-error surface，不是完整误差分布，也不提供 non-negativity guarantee。两种 mode 的条件边际效应具有同一形式：</p>
<div class="math-block">\[\frac{{\partial\widehat T_m}}{{\partial N}}=\beta_{{N,m}}+\beta_{{NL,m}}L_k,\qquad \frac{{\partial\widehat T_m}}{{\partial L}}=\frac{{\beta_{{L,m}}+\beta_{{NL,m}}N}}{{1000}},\quad m\in\{{NT,T\}}.\]</div>
<p>共享的是 basis，不是强度。Non-thinking 的标准化 \(|E|\) 中位数为：N {num(direct_mae_n_effect['median_abs'],2)}、\(L_k\) {num(direct_mae_l_effect['median_abs'],2)}、\(NL_k\) {num(direct_mae_interaction_effect['median_abs'],2)}；N 与交互都很大，而 L 主效应较小。Native-thinking 则为 N {num(native_mae_n_effect['median_abs'],2)}、\(L_k\) {num(native_mae_l_effect['median_abs'],2)}、\(NL_k\) {num(native_mae_interaction_effect['median_abs'],2)}；交互约为较大单轴主效应的 {num(native_mae_interaction_effect['median_abs'] / max(native_mae_n_effect['median_abs'], native_mae_l_effect['median_abs']),1)} 倍，12/12 槽为正，{pct(native_mae_interaction_effect['q05_fraction'],0)} 的槽在 HC3+BH 后 q&lt;0.05。</p>
<h3>2.2 Mechanism hypothesis: intrinsic numerosity noise versus retrieval–aggregation coupling</h3>
<div class="derivation"><p><strong>从可积累的“误差机会”出发。</strong> 令 \(q_m(L)\) 表示 mode \(m\) 在长度 L 下，每个 target 对 central absolute-error scale 贡献的一阶机会，令 \(r_m(L)\) 表示与 target 数无关的 routing/context burden。若稀疏局部失误的绝对幅度近似可加、强抵消与多重同时失误进入高阶项，则</p><div class="math-block">\[T_m(N,L)\approx r_m(L)+Nq_m(L).\]</div><p>在有限的注册长度区间作局部展开 \(r_m(L)\approx r_0+r_1L_k\)、\(q_m(L)\approx q_0+q_1L_k\)，得到</p><div class="math-block">\[T_m(N,L)\approx r_0+q_0N+r_1L_k+q_1NL_k.\]</div><p>这不是为回归 basis 作事后同义改写：它给出一个可证伪的机制含义——\(q_0\) 是短上下文中的每目标 central-error burden，\(q_1\) 是 passage length 对该每目标 burden 的调制。若错误是独立、零均值且大量相互抵消，典型绝对和更可能呈 \(\sqrt N\) 而非 N；若大误差由单次全局崩溃主导，也不应稳定分解为 \(r+Nq\)。因此当前线性–交互几何更接近“稀疏失误机会或方向一致的聚合误差”，而不是任意噪声模型。</p>
<p><strong>Non-thinking：数量本身已经制造 central error。</strong> 其跨槽原始系数中位数为 \(b_N={num(direct_mae_n_raw[0],3)}\)、\(b_L={num(direct_mae_l_raw[0],3)}\)、\(b_{{NL}}={num(direct_mae_nl_raw[0],3)}\)。较大的正 N 主效应意味着即使短上下文中，增加 target load 也会扩大典型可解析误差；正交互又表示长上下文进一步放大这一 numerosity burden。这与 Accuracy 部分的“全局近似数感”一致：压缩表征不仅更容易错，而且一旦错，其 central magnitude 随 N 扩张。这里不要求每个 target 被显式逐项处理；N 项也可以是有限网格上全局噪声尺度的一阶投影。</p>
<p><strong>Native-thinking：单轴负担被压低，联合路由–聚合才暴露。</strong> 其系数中位数为 \(d_N={num(native_mae_n_raw[0],3)}\)、\(d_L={num(native_mae_l_raw[0],3)}\)、\(d_{{NL}}={num(native_mae_nl_raw[0],3)}\)，且 12/12 槽的 interaction 为正。近零的典型单轴项配合稳定正交互，支持这样的 operating regime：N 小时，即使 L 长，只有少量证据需要路由；L 短时，即使 N 大，目标证据仍易访问；只有 N 与 L 同时大时，每项目标的检索/状态更新风险 \(q_T(L)\) 才累积成可见误差。这就是 retrieval–aggregation coupling，而不是“reasoning 对所有条件统一加一个常数优势”。</p>
<p><strong>Accuracy 与 tMAE 为什么给出不同 law。</strong> Accuracy 是 \(P(\widehat C=N)\)，对“是否发生任何导致最终不精确的事件”敏感；tMAE 是可解析响应绝对误差 middle 80% 的位置，对“发生偏离后典型偏多远”敏感。串行过程可使零错误概率近似按可靠性乘积衰减，同时让累计误差幅度由 \(Nq(L)\) 控制。因此 Native-thinking 可以保持很高 Accuracy，却在少数困难区域出现结构化的正 \(NL\) magnitude；这不是矛盾，而是 event probability 与 error severity 的分离。</p>
<p><strong>截尾与 identity link 的机制边界。</strong> 最低和最高各 10% absolute errors 被删除，所以这个机制只针对 central regime：它既不描述最容易的近零质量，也不描述 catastrophic tail。且 identity OLS 允许负预测，说明上述式子是注册网格上的一阶行为近似，不是非负误差的生成分布。</p></div>
<details><summary>Falsifiable MAE checks</summary><ol><li>按错误类型区分遗漏、重复计数和状态覆盖；若 \(Nq(L)\) 成立，正 interaction 应能由每目标错误率随 L 上升解释，而不是由少数全局崩溃解释。</li><li>比较 trimmed variance、MAD 与 tMAE 的 N-scaling；若只有均值呈线性交互而 robust scale 不随之变化，可能是误差方向混合或 cell composition，而非统一 burden。</li><li>在固定 N、L 下外部提供可验证的中间计数状态；若 Native-thinking 的 \(NL\) 项来自状态维护/聚合，它应显著减弱；若来自最初检索，它可能保持。</li></ol></details>
<div class="warning"><strong>Identity-link 限制：</strong>观测 \(\operatorname{{tMAE}}_{{10}}\) 必然非负，但未截断 OLS 在注册网格内给出 Non-thinking 最小预测 {num(mae_prediction_diagnostics['direct']['minimum'],2)}、Native-thinking {num(mae_prediction_diagnostics['native_thinking']['minimum'],2)} count；负预测比例分别为 {pct(mae_prediction_diagnostics['direct']['negative_fraction'],2)} 与 {pct(mae_prediction_diagnostics['native_thinking']['negative_fraction'],2)}。报告保留这些值，不作静默裁剪。它们意味着 identity OLS 只能作为网格内局部近似，不能外推为非负生成分布。</div>
<div class="conclusion"><strong>MAE 公式结论：</strong>双侧 10% 截尾后，两种 mode 都选择 \(N+L_k+NL_k\)。Non-thinking 的 held-condition 预测更强但 topology 稳定性为 {pct(direct_mae.lomo_formula_stability,1)}；Native-thinking 的精度较低、Q25 更弱，却在 LOMO 中保持 {pct(native_mae.lomo_formula_stability,1)}，且正交互跨槽一致。可靠结论是共享联合暴露几何、不同系数强度与稳定性，而不是两个 mode 的误差分布相同。</div>

<h3>2.3 Aggregate scaling-law view</h3>
<figure><img src="{figures['mae_aggregate_n']}" alt="aggregate 10% trimmed conditional MAE with N on the horizontal axis"><figcaption><strong>FIGURE 5</strong><span><b>10% trimmed conditional MAE 汇总图，N-horizontal。</b> 左/右面板分别为 Non-thinking 与 Native-thinking；横轴覆盖全部 N，黄→紫曲线固定全部 L。空心点与误差条是 12 槽观测 \(\operatorname{{tMAE}}_{{10}}\) 的 Q50 与 [Q25,Q75]；实线与带是 12 套 identity OLS 预测的 Q50 与 [Q25,Q75]。每个 cell 先对绝对误差双侧各删 10%；图中 IQR 是模型异质性，不是置信区间。</span></figcaption></figure>
<figure><img src="{figures['mae_aggregate_l']}" alt="aggregate 10% trimmed conditional MAE with L on the horizontal axis"><figcaption><strong>FIGURE 6</strong><span><b>10% trimmed conditional MAE 汇总图，L-horizontal。</b> 横轴覆盖全部 L，黄→紫曲线固定全部 N。symlog 在 \([-0.25,0.25]\) 附近线性、外侧对正负绝对值取对数；负的 fitted value 被保留为 identity-link 失配诊断。该图与 Figure 5 使用同一方程，没有重新拟合。</span></figcaption></figure>
<h3>2.4 Model-resolved trimmed-MAE fits</h3>
<figure><img src="{figures['mae_n']}" alt="12 models, 10% trimmed conditional MAE fits with N on the horizontal axis"><figcaption><strong>FIGURE 7</strong><span><b>10% trimmed conditional MAE 模型分辨图，N-horizontal。</b> 横轴为全部 14 个 N，颜色固定全部 8 个 L。点是每 cell 中间 80% 绝对误差的均值，线是 identity-scale OLS 预测。水平 0 将物理可行观测区与负预测诊断区分开。</span></figcaption></figure>
<figure><img src="{figures['mae_l']}" alt="12 models, 10% trimmed conditional MAE fits with L on the horizontal axis"><figcaption><strong>FIGURE 8</strong><span><b>10% trimmed conditional MAE 模型分辨图，L-horizontal。</b> 横轴为全部 8 个 L，颜色固定全部 14 个 N；其余编码与 Figure 7 相同。两张图共同显示不同 N 曲线随 L 的扇出及其模型异质性。</span></figcaption></figure>
<div class="table-wrap">{table_html(mae_main)}</div>
{equation_details['direct_mae']}{equation_details['native_mae']}
<div class="conclusion"><strong>MAE 证据结论：</strong>Non-thinking median held-condition \(R^2={num(direct_mae.median_primary_score,3)}\)，Native-thinking 为 {num(native_mae.median_primary_score,3)}；LOMO topology stability 分别为 {pct(direct_mae.lomo_formula_stability,1)} / {pct(native_mae.lomo_formula_stability,1)}。因此 \(N\times L_k\) 联合放大是共同主模式；Non-thinking 更可预测，Native-thinking 的 topology 更稳定但跨模型预测下四分位更弱。</div>
</section>

<section id="appendix-a" class="appendix"><div class="section-head"><div class="section-no">Appendix A</div><div><h2>A shared interaction topology, with distinct strength</h2><p class="lede">这句话专指预注册的 10% trimmed signed-bias law：两种 mode 都选择 \(N+L_k+NL_k\)，但交互系数的大小与显著性分布不同。</p></div></div>
<figure><img src="{figures['interaction']}" alt="standardized N by L interaction coefficients for bias"><figcaption><strong>FIGURE A1</strong><span><b>Bias 的标准化 \(N\times L_k\) 交互强度。</b> 每行一个模型槽，横轴为 \(E_{{NL}}=\beta_{{NL}}\operatorname{{SD}}(NL_k)/\operatorname{{SD}}(b_{{10}})\)。实心点表示槽内 HC3 p 值经 12 槽 BH 校正后 q≤0.05；空心点表示未过该阈值。灰带 \(|E|&lt;0.1\) 是预定 practical-effect 区域，不是置信区间。</span></figcaption></figure>
<p>共享 topology 的数学含义仅是同一组 predictors。对任一 mode，\(\partial b_{{10}}/\partial N=\beta_N+\beta_{{NL}}L_k\)，所以交互系数决定 passage length 如何改变目标数的方向性误差斜率。Non-thinking 与 Native-thinking 系数不同，因此相同 topology 可以产生不同强度与不同曲面。</p>
<div class="conclusion"><strong>Appendix A 结论：</strong>“shared”描述 basis，“distinct strength”描述槽特异标准化系数；它不是说两种 mode 具有相同误差分布。</div></section>

<section id="appendix-b" class="appendix"><div class="section-head"><div class="section-no">Appendix B</div><div><h2>Bias regression and outlier control</h2><p class="lede">Bias 使用 V3.1 预注册的 10% cell-wise symmetric trimming，普通均值版本不作为正式 estimand。</p></div></div>
<div class="math-block">\[\widehat b_{{10,s,m}}=\alpha_{{s,m}}+\beta_{{N,s,m}}N+\beta_{{L,s,m}}L_k+\beta_{{NL,s,m}}NL_k.\]</div>
<p>当 m=30 时，\(k=3\)，所以 Bias 平均排序后中间 24 个 signed errors；这是一个不同于原始均值的 population target，不应解释为对 raw mean 的无偏估计。它在定义上阻止每端最极端的三个响应直接进入 cell location，但不能消除剩余 24 个值的变异，也不保证对任意污染机制稳健。</p>
<p>交互项给出条件边际效应 \(\partial\widehat b_{{10}}/\partial N=\beta_N+\beta_{{NL}}L_k\) 与 \(\partial\widehat b_{{10}}/\partial L=(\beta_L+\beta_{{NL}}N)/1000\)。因此正 \(\beta_{{NL}}\) 表示上下文越长，目标数增加所伴随的 over-counting slope 越大；若截距或主效应为负，局部区域仍可表现为 under-counting，不能只凭交互符号判断整个曲面方向。</p>
<figure><img src="{figures['bias_aggregate_n']}" alt="aggregate trimmed signed bias with N on the horizontal axis"><figcaption><strong>FIGURE B1</strong><span><b>10% trimmed signed bias 汇总图，N-horizontal。</b> 横轴覆盖全部 N，黄→紫曲线固定全部 L；0 为无方向性偏差，正值为多计、负值为少计。空心点/误差条为 12 槽观测 \(b_{{10}}\) 的 Q50/[Q25,Q75]，实线/带为 12 套槽特异 OLS 预测的 Q50/[Q25,Q75]。这里的 IQR 是模型异质性，不是 bias 的置信区间。</span></figcaption></figure>
<figure><img src="{figures['bias_aggregate_l']}" alt="aggregate trimmed signed bias with L on the horizontal axis"><figcaption><strong>FIGURE B2</strong><span><b>10% trimmed signed bias 汇总图，L-horizontal。</b> 横轴覆盖全部 L，黄→紫曲线固定全部 N。该图是 B1 同一 N×L 曲面的转置切片；曲线随 N 扇出对应 \(N\times L_k\) interaction。</span></figcaption></figure>
<figure><img src="{figures['bias_n']}" alt="12 models, trimmed signed bias with N on the horizontal axis"><figcaption><strong>FIGURE B3</strong><span><b>10% trimmed signed bias 模型分辨图，N-horizontal。</b> 3×4 面板逐槽展示全部 N 与 L。点是 cell 内两端各删除 \(\lfloor0.1m\rfloor\) 个 signed errors 后的均值，线是该槽 identity OLS 预测。</span></figcaption></figure>
<figure><img src="{figures['bias_l']}" alt="12 models, trimmed signed bias with L on the horizontal axis"><figcaption><strong>FIGURE B4</strong><span><b>10% trimmed signed bias 模型分辨图，L-horizontal。</b> 横轴为全部 L，颜色固定全部 N；它展示汇总 IQR 背后的槽间系数差异。</span></figcaption></figure>
<div class="table-wrap">{table_html(law_main[law_main['Outcome'].str.contains('bias', case=False)])}</div>
<p>Cook's distance 用 \(D_i=[e_i^2/(p\operatorname{{MSE}})]h_{{ii}}/(1-h_{{ii}})^2\) 筛查 cell-level influence，阈值 \(4/n\) 仅作诊断。删除被标记 cells 后按固定 topology 重拟合，典型曲面与原曲面的相关接近 1；这说明主 topology 不是由单个 cell 驱动，但不保证每个系数精确稳定。</p>
<details><summary>Request-tail 与 Cook's D 审计</summary><div class="table-wrap">{table_html(bias_tail_table)}</div><div class="table-wrap">{table_html(bias_influence_table)}</div><div class="table-wrap">{table_html(bias_influence_details)}</div></details>
{equation_details['direct_bias']}{equation_details['native_bias']}
<div class="conclusion"><strong>Appendix B 结论：</strong>截尾避免极端 request errors 主导方向性均值；两种 mode 共享 \(N\times L_k\) interaction topology，但 Non-thinking 的方向性放大通常更强。</div></section>

<section id="appendix-c" class="appendix"><div class="section-head"><div class="section-no">Appendix C</div><div><h2>Enumeration controls repeat the Accuracy and MAE views</h2><p class="lede">Index 与 Bullet enumeration 不进入正文主比较；这里用完全相同的坐标、selection 与 validation 展示，避免只报告聚合数字。</p></div></div>
<h3>C.1 Enumeration Accuracy</h3><p>Index 与 Bullet 的 logit Accuracy 都选择 \(\ln N+\ln L_k\)：\(\operatorname{{logit}}\widehat p=\alpha+\beta_N\ln N+\beta_L\ln L_k\)。它等价于 odds \(\propto N^{{\beta_N}}L_k^{{\beta_L}}\)，即两个难度轴都按乘法尺度作用。</p>
<figure><img src="{figures['enum_accuracy_n']}" alt="enumeration accuracy with N on the horizontal axis"><figcaption><strong>FIGURE C1</strong><span><b>Enumeration Accuracy，N-horizontal。</b> Index 为实线/圆点，Bullet 为虚线/三角；全部 8 个 L 以黄→紫编码，12 个模型槽分别作图。</span></figcaption></figure>
<figure><img src="{figures['enum_accuracy_l']}" alt="enumeration accuracy with L on the horizontal axis"><figcaption><strong>FIGURE C2</strong><span><b>Enumeration Accuracy，L-horizontal。</b> 同一方程改以 L 为横轴，并用全部 14 个 N 着色。Index median CV D²={num(index_acc.median_primary_score,3)}，Bullet={num(bullet_acc.median_primary_score,3)}。</span></figcaption></figure>
{equation_details['index_accuracy']}{equation_details['bullet_accuracy']}
<h3>C.2 Enumeration 10% trimmed conditional MAE</h3><p>Index 选择 \({FORMULA[index_mae.selected_candidate]}\)，Bullet 选择 \({FORMULA[bullet_mae.selected_candidate]}\)。两者都使用正文相同的双侧截尾与 identity OLS；负预测保留并显示在 symlog 轴上。Index 的 Q25 held \(R^2={num(index_mae.q25_primary_score,3)}\)，说明典型 fit 很弱；Bullet 的 LOMO topology stability 只有 {pct(bullet_mae.lomo_formula_stability,1)}，因此其精确对数交互也不宜作强结论。</p>
<figure><img src="{figures['enum_mae_n']}" alt="enumeration 10% trimmed conditional MAE with N on the horizontal axis"><figcaption><strong>FIGURE C3</strong><span><b>Enumeration 10% trimmed MAE，N-horizontal。</b> 点为每个 cell 的 middle-80% absolute-error mean，线为未截断预测；颜色覆盖全部 L。</span></figcaption></figure>
<figure><img src="{figures['enum_mae_l']}" alt="enumeration 10% trimmed conditional MAE with L on the horizontal axis"><figcaption><strong>FIGURE C4</strong><span><b>Enumeration 10% trimmed MAE，L-horizontal。</b> 颜色覆盖全部 N。Index median CV R²={num(index_mae.median_primary_score,3)}，Bullet={num(bullet_mae.median_primary_score,3)}。若某一槽的 trimmed response 恒为 0，则其槽内 \(R^2\) 不定义；跨槽汇总按有限 scores 计算。</span></figcaption></figure>
{equation_details['index_mae']}{equation_details['bullet_mae']}
<div class="conclusion"><strong>Appendix C 结论：</strong>Enumeration Accuracy 的两轴均呈幂律 odds；双侧截尾 MAE 仍显示 N/L 结构，但 Index 的预测力弱、Bullet 的 topology 不稳定。它们与 Native-thinking 有部分几何相似，不能据此断言相同内部算法。</div></section>

<section id="appendix-d" class="appendix"><div class="section-head"><div class="section-no">Appendix D</div><div><h2>Three estimands on the joint N–L grid</h2><p class="lede">三维图把正文与附录的三类因变量放回同一个注册网格：x 轴是目标数 \(N\)，y 轴是 passage length \(L_k=L/1000\)，z 轴依次是 Accuracy、\(\operatorname{{tMAE}}_{{10}}\) 与 \(b_{{10}}\)。左列为 Non-thinking，右列为 Native-thinking。</p></div></div>
<p>对每个注册坐标 \((N,L)\)，空心边框点的高度是 12 个 comparison slots 的观测 cell 值中位数；半透明曲面的高度是 12 套槽特异 selected-law 预测在该坐标上的逐点中位数：</p>
<div class="math-block">\[z_{{\mathrm{{obs}}}}(N,L)=Q_{{0.5}}\!\left\{{y_s(N,L)\right\}}_{{s=1}}^{{12}},\qquad z_{{\mathrm{{fit}}}}(N,L)=Q_{{0.5}}\!\left\{{\widehat y_s(N,L)\right\}}_{{s=1}}^{{12}}.\]</div>
<p>这里没有对 12 个模型系数取中位数后重建一个“平均模型”，因为一般而言 \(Q_{{0.5}}(X\beta_s)\neq XQ_{{0.5}}(\beta_s)\)。报告先用每个槽自己的方程计算 \(\widehat y_s(N,L)\)，再在预测值上取中位数。黄→紫颜色仅重复编码 z 轴高度（yellow=low，purple=high），不是第四个变量。Accuracy z 轴是概率；MAE 与 Bias z 轴是 count units，因此三个 estimand 的垂直刻度不可直接比较。观测点只位于冻结的 14×8 注册网格；为使旋转时曲面连续可读，selected law 在网格边界内的 48×48 坐标上重新求值，但没有重拟合、插入新观测或外推到注册范围之外。</p>
<figure><div id="appendix-d-interactive" class="d3-shell"><img class="interactive-fallback" src="{figures['appendix_d_3d']}" alt="static fallback: three by two 3D surfaces for accuracy, trimmed conditional MAE, and trimmed signed bias over target count and passage length"><div class="d3-explorer"><div class="d3-toolbar"><div class="d3-metrics" role="group" aria-label="Choose the Appendix D estimand"><button class="d3-metric is-active" type="button" data-d3-metric="accuracy" aria-pressed="true">Accuracy<span>probability</span></button><button class="d3-metric" type="button" data-d3-metric="mae" aria-pressed="false">Trimmed MAE<span>count units</span></button><button class="d3-metric" type="button" data-d3-metric="bias" aria-pressed="false">Trimmed bias<span>count units</span></button></div><button id="appendix-d-reset" class="d3-reset" type="button">Reset view</button></div><div class="d3-status"><span id="appendix-d-metric-note">Exact accuracy · probability · yellow = lower, purple = higher</span><span>Drag to rotate · scroll to zoom · hover for observed and fitted values</span></div><div class="d3-grid"><article class="d3-panel"><div class="d3-panel-head"><strong>Non-thinking</strong><span id="appendix-d-direct-law"></span></div><div id="appendix-d-direct" class="d3-plot" aria-label="Interactive Non-thinking N by L response surface"></div></article><article class="d3-panel"><div class="d3-panel-head"><strong>Native-thinking</strong><span id="appendix-d-native-law"></span></div><div id="appendix-d-native" class="d3-plot" aria-label="Interactive Native-thinking N by L response surface"></div></article></div></div></div><script id="appendix-d-payload" type="application/json">{appendix_d_payload_json}</script><script>{appendix_d_script}</script><figcaption><strong>FIGURE D1</strong><span><b>可旋转的 Accuracy、10% trimmed conditional MAE 与 10% trimmed signed bias 三维 N–L 几何。</b> 上方按钮切换 z 轴因变量；左右面板始终共享该因变量的 z 范围与颜色尺度。拖动任一面板会旋转并同步另一面板的相机，滚轮缩放，Reset view 恢复论文初始视角。深色描边点是跨 12 槽观测中位数，半透明曲面是跨 12 套槽特异预测的逐点中位数；hover 同时给出观测、拟合及残差。MAE 曲面可能出现负的 identity-link 预测，仍按正文规则保留为局部线性失配诊断。打印或脚本不可用时自动显示静态 3×2 fallback。</span></figcaption></figure>
<p class="warning"><strong>透视图的阅读限制：</strong>三维投影会压缩远端距离，并可能遮挡点；不能用屏幕上的视觉角度替代系数、边际导数、CV 或 LOMO。定量主张仍应以正文的两轴切片、公式和 held-condition 指标为准。D1 的用途是检查曲面整体形状、mode 间弯曲/扇出差异以及观测点是否系统性偏离 fitted surface。</p>
<div class="conclusion"><strong>Appendix D 结论：</strong>三维视角把三类 estimand 的区别集中显示出来：Accuracy 描述零误差概率曲面；trimmed MAE 描述 central absolute-error magnitude；trimmed Bias 描述方向性位置。相同 N–L 网格上的不同 z 几何说明“更容易出错”“典型误差更大”和“更倾向多计/少计”是三个不可互换的行为命题。</div></section>

<section id="repro"><div class="section-head"><div class="section-no">Reproducibility</div><div><h2>Audit trail and claim boundary</h2><p class="lede">结果只在注册网格与冻结模型集合内成立；网格外外推、未见架构与内部机制均不属于当前证据。</p></div></div>
<details><summary>Frozen model revisions</summary><div class="table-wrap">{table_html(revisions)}</div></details>
<details><summary>Files, hashes, and rebuild command</summary><div class="methods">protocol: {html.escape(audit['protocol_version'])}<br>inference commit: {html.escape(audit['git_commit'])}<br>requests: {audit['requests']:,}<br>unique_request_ids: {audit['unique_request_ids']:,}<br>request_ids_sha256: {html.escape(audit['request_ids_sha256'])}<br>stimuli_sha256: {html.escape(audit['stimuli_sha256'])}<br>analysis input sha256: {html.escape(manifest['input_sha256'])}<br>analysis state: {html.escape(state['stage'])}<br>bootstrap repetitions: {manifest['bootstrap_repetitions']}<br><br>Rebuild:<br>.venv\Scripts\python.exe scripts\build_niah_empirical_law_v3_2_report.py</div></details>
<ul><li><strong>Grid-bound:</strong> \(N\in\{{1,2,3,4,5,6,7,8,9,10,12,15,18,20\}}\)，\(L\in\{{1,2,3,5,8,10,15,20\}}\)k。</li><li><strong>Model-set-bound:</strong> LOMO 只覆盖当前 12 个 comparison slots。</li><li><strong>Parsing-bound:</strong> Accuracy 将 parse failure 计错；Bias/MAE 条件于可解析整数。</li><li><strong>Trimming-bound:</strong> \(\operatorname{{tMAE}}_{{10}}\) 同时删除绝对误差的最低与最高 10%，只描述 middle-80% central magnitude，不是 raw request-level expected loss，也不描述 catastrophic tail。</li><li><strong>Identity-MAE-bound:</strong> 负 fitted trimmed MAE 明示该线性模型不是非负生成模型。</li><li><strong>Mechanism-bound:</strong> 边际导数与 odds 分解是回归的数学后果；检索—聚合解释是理论启发，不是机制识别。</li></ul>
<div class="conclusion"><strong>总括结论：</strong>最稳健的正文发现是：Native-thinking 改变了 exactness 对 N 与 L 的相对敏感度；而可解析错误的幅度主要由 N–L 联合暴露控制。Bias 进一步显示两种 mode 可以共享交互 topology，却具有不同强度。</div></section>
<footer class="footer">NiaH Empirical-law Report · V3.2 · 10% symmetrically trimmed conditional MAE identity regression · generated {html.escape(generated)}.</footer>
</main></body></html>"""

    if "�" in report:
        raise ValueError("Report contains a Unicode replacement character")
    forbidden_bootstrap_phrases = [
        "bootstrap repetitions = 2,000",
        "bootstrap_repetitions: 2000",
        "2000 次 bootstrap",
        "2,000 次 bootstrap",
    ]
    if any(phrase in report for phrase in forbidden_bootstrap_phrases):
        raise ValueError("The report reintroduced the discarded heavy-bootstrap design")
    # Preserve the separately audited, post-hoc plateau/additive comparison on rebuild.
    from build_niah_empirical_piecewise_addendum import OUT as piecewise_out, inject as inject_piecewise
    if (piecewise_out / "tables" / "report_version_summary.csv").exists():
        report = inject_piecewise(report)
    from build_niah_empirical_n_fixed_addendum import OUT as n_fixed_out, inject as inject_n_fixed
    if (n_fixed_out / "manifest.json").exists():
        report = inject_n_fixed(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    build_manifest = {
        "schema_version": "niah_empirical_law_report_v3_2_build_v1",
        "created_at": generated,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "analysis_version": manifest["analysis_version"],
        "analysis_state": state["stage"],
        "requests": audit["requests"],
        "unique_request_ids": audit["unique_request_ids"],
        "physical_model_revisions": v32["immutable_input"]["physical_model_revisions"],
        "comparison_slots": v32["immutable_input"]["comparison_slots"],
        "cells": manifest["cells"],
        "bootstrap_repetitions": manifest["bootstrap_repetitions"],
        "figures": {key: {"path": str(value.resolve()), "sha256": sha256(value)} for key, value in figure_paths.items()},
        "interactive_appendix_d": {
            "engine": "Plotly.js 3.6.0",
            "bundle_path": str(plotly_bundle_path.resolve()),
            "bundle_sha256": sha256(plotly_bundle_path),
            "payload_schema": appendix_d_payload["schema_version"],
            "payload_sha256": hashlib.sha256(
                appendix_d_payload_json.encode("utf-8")
            ).hexdigest(),
            "registered_points_per_mode": int(
                len(appendix_d_payload["registered_grid"]["N"])
                * len(appendix_d_payload["registered_grid"]["L_tokens"])
            ),
            "surface_grid_points": appendix_d_payload["surface_grid_points"],
            "refit": False,
        },
        "headline_accuracy_laws": {mode: selected[selected["outcome_family"].eq("accuracy_bernoulli_logit") & selected["prompt_mode"].eq(mode)]["selected_candidate"].iloc[0] for mode in ["direct", "native_thinking"]},
        "headline_bias_laws": {mode: selected[selected["outcome_family"].eq("trimmed_signed_bias_10") & selected["prompt_mode"].eq(mode)]["selected_candidate"].iloc[0] for mode in ["direct", "native_thinking"]},
        "headline_trimmed_conditional_mae_laws": {mode: mae_selected[mae_selected["prompt_mode"].eq(mode)]["selected_candidate"].iloc[0] for mode in ["direct", "native_thinking"]},
        "sources": {
            "audit": str(audit_path.resolve()),
            "behavior_tables": str(behavior_tables.resolve()),
            "v3_2_tables": str(law_tables.resolve()),
            "untrimmed_bias_sensitivity_tables": str(sensitivity_tables.resolve()),
            "untrimmed_bias_sensitivity_manifest_sha256": sha256(
                sensitivity_dir / "analysis_manifest.json"
            ),
            "trimmed_count_error_extension_tables": str(count_error_tables.resolve()),
            "trimmed_count_error_extension_manifest_sha256": sha256(
                count_error_dir / "analysis_manifest.json"
            ),
            "inverse_n_candidate_extension_tables": str(law_tables.resolve()),
            "inverse_n_candidate_extension_manifest_sha256": sha256(
                law_dir / "analysis_manifest.json"
            ),
            "v3_1_config": str(args.v3_1_config.resolve()),
            "v3_2_config": str(args.v3_2_config.resolve()),
        },
    }
    manifest_path = assets / "report_build_manifest.json"
    manifest_path.write_text(json.dumps(build_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(args.output), "bytes": args.output.stat().st_size, "figures": len(figure_paths), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
